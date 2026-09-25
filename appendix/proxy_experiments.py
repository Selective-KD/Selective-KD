import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.isotonic import IsotonicRegression

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXTENT, SIGMA, FLOOR_FRAC, N_TRAIN, GRID_SIDE, SEED = 6.0, 1.6, 0.06, 30000, 250, 0
K, HIDDEN, DEPTH, EPOCHS, BATCH_SIZE, LR = 3, 256, 3, 250, 256, 2e-3
PRIOR_SD, RHO_INIT, KL_SCALE = 1.0, -4.0, 1.0
T_EVAL, N_ENSEMBLE, WEIGHT_DECAY, PRIOR_PREC = 100, 20, 1e-4, 1.0
N_DOTS = 1200
TEACHERS = ("vi", "laplace", "ensemble")
TAGS = ("A", "B", "C")
DETAIL_TAG, DETAIL_KIND = "A", "laplace"
METRIC_SEED = 0
EPS = 1e-12


def _f64(p):
    return np.asarray(p, dtype=np.float64)


def ent(p, axis=-1):
    p = _f64(p)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis)


def gini(p, axis=-1):
    p = _f64(p)
    return 1.0 - (p ** 2).sum(axis)


def kl(p, q, axis=-1):
    p, q = _f64(p), _f64(q)
    return (p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(q, EPS, None)))).sum(axis)


def l2(p, q, axis=-1):
    return np.sqrt(((_f64(p) - _f64(q)) ** 2).sum(axis))


def true_quantities(p_star, t):
    p_star, t = _f64(p_star), _f64(t)
    return {"hard_shannon": ent(p_star), "hard_gini": gini(p_star),
            "err_kl_fwd": kl(p_star, t), "err_kl_rev": kl(t, p_star), "err_l2": l2(p_star, t)}


def sample_inputs(n, extent, sigma, floor_frac, rng):
    n_floor = int(round(n * floor_frac))
    n_core = n - n_floor
    parts = []
    if n_core > 0:
        got, chunks = 0, []
        while got < n_core:
            c = rng.normal(0.0, sigma, size=(2 * (n_core - got) + 64, 2))
            c = c[np.abs(c).max(1) <= extent]
            chunks.append(c)
            got += len(c)
        parts.append(np.concatenate(chunks)[:n_core])
    if n_floor > 0:
        parts.append(rng.uniform(-extent, extent, size=(n_floor, 2)))
    X = np.concatenate(parts).astype(np.float32)
    rng.shuffle(X, axis=0)
    return X


def draw_labels(p_star, rng):
    p = np.asarray(p_star, dtype=np.float64)
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0
    u = rng.random((len(p), 1))
    return np.clip((u > c).sum(1), 0, p.shape[1] - 1).astype(np.int64)


def grid(extent, n_side):
    g = np.linspace(-extent, extent, n_side)
    GX, GY = np.meshgrid(g, g)
    return np.column_stack([GX.ravel(), GY.ravel()]).astype(np.float32), (n_side, n_side)


A_FIELD_SEED, A_NF, A_LS, A_B0, A_BS = 11, 256, 1.3, 1.6, 0.45
_a_rng = np.random.default_rng(A_FIELD_SEED)
A_W = _a_rng.normal(0.0, 1.0 / A_LS, size=(A_NF, 2))
A_B = _a_rng.uniform(0.0, 2.0 * np.pi, size=A_NF)
A_A = _a_rng.normal(0.0, 1.0, size=(K, A_NF))


def p_star_A(X):
    X = np.asarray(X, dtype=np.float64)
    Z = np.sqrt(2.0 / A_NF) * np.cos(X @ A_W.T + A_B)
    f = Z @ A_A.T
    r = np.linalg.norm(X, axis=1, keepdims=True)
    z = (A_B0 * (1.0 + A_BS * r)) * f
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


B_SEED, B_CELL, B_JIT, B_W, B_B0, B_BS = 5, 1.5, 0.55, 0.85, 2.2, 0.35
_b_pad = B_CELL * 2.0
_b_g = np.arange(-EXTENT - _b_pad, EXTENT + _b_pad + 1e-9, B_CELL)
_B_CX, _B_CY = np.meshgrid(_b_g, _b_g)
_b_rng = np.random.default_rng(B_SEED)
B_C = np.column_stack([_B_CX.ravel(), _B_CY.ravel()])
B_C = B_C + _b_rng.uniform(-B_JIT, B_JIT, size=B_C.shape)
B_CLS = _b_rng.integers(0, K, size=len(B_C))
B_MASKS = [(B_CLS == k) for k in range(K)]


def p_star_B(X, chunk=20000):
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((len(X), K))
    for a in range(0, len(X), chunk):
        xb = X[a:a + chunk]
        d2 = ((xb[:, None, :] - B_C[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2.0 * B_W ** 2))
        s = np.column_stack([w[:, m].sum(1) for m in B_MASKS])
        r = np.linalg.norm(xb, axis=1, keepdims=True)
        z = (B_B0 * (1.0 + B_BS * r)) * s
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[a:a + chunk] = e / e.sum(1, keepdims=True)
    return out


C_R0, C_BLEND, C_BLOB_R, C_BLOB_SD = 2.6, 0.55, 1.35, 1.05
C_OUT_SEED, C_N_SECT, C_N_RING, C_RING_W, C_SHARP, C_SMOOTH = 9, 7, 4, 1.1, 4.5, 0.35
_c_ang = np.arange(K) * 2.0 * np.pi / K + np.pi / 2.0
C_CEN = C_BLOB_R * np.column_stack([np.cos(_c_ang), np.sin(_c_ang)])
C_BLOCK = np.random.default_rng(C_OUT_SEED).integers(0, K, size=(C_N_RING, C_N_SECT))
_c_I, _c_J = np.meshgrid(np.arange(C_N_RING) + 0.5, np.arange(C_N_SECT) + 0.5, indexing="ij")
C_IDX = np.column_stack([_c_I.ravel(), _c_J.ravel()])
C_BCLS = C_BLOCK.ravel()


def _c_inner_logits(X):
    d2 = ((X[:, None, :] - C_CEN[None, :, :]) ** 2).sum(-1)
    return -d2 / (2.0 * C_BLOB_SD ** 2)


def _c_outer_logits(X):
    r = np.linalg.norm(X, axis=1)
    a = np.arctan2(X[:, 1], X[:, 0])
    u = np.clip((r - C_R0) / C_RING_W, 0.5, C_N_RING - 0.5)
    v = (a + np.pi) / (2.0 * np.pi) * C_N_SECT
    du = u[:, None] - C_IDX[None, :, 0]
    dv = np.abs(v[:, None] - C_IDX[None, :, 1])
    dv = np.minimum(dv, C_N_SECT - dv)
    w = np.exp(-(du ** 2 + dv ** 2) / (2.0 * C_SMOOTH ** 2))
    w = w / np.clip(w.sum(1, keepdims=True), 1e-300, None)
    s = np.column_stack([w[:, C_BCLS == k].sum(1) for k in range(K)])
    return C_SHARP * s


def p_star_C(X):
    X = np.asarray(X, dtype=np.float64)
    r = np.linalg.norm(X, axis=1, keepdims=True)
    w = 1.0 / (1.0 + np.exp(-(r - C_R0) / C_BLEND))
    z = (1.0 - w) * _c_inner_logits(X) + w * _c_outer_logits(X)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


WORLDS = {"A": p_star_A, "B": p_star_B, "C": p_star_C}


class BayesLinear(nn.Module):
    def __init__(self, d_in, d_out, prior_sd=1.0, rho_init=-4.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(d_out, d_in).normal_(0.0, 1.0 / np.sqrt(d_in)))
        self.w_rho = nn.Parameter(torch.full((d_out, d_in), float(rho_init)))
        self.b_mu = nn.Parameter(torch.zeros(d_out))
        self.b_rho = nn.Parameter(torch.full((d_out,), float(rho_init)))
        self.prior_sd = float(prior_sd)

    def sample_weights(self):
        w = self.w_mu + F.softplus(self.w_rho) * torch.randn_like(self.w_rho)
        b = self.b_mu + F.softplus(self.b_rho) * torch.randn_like(self.b_rho)
        return w, b

    def forward(self, x, sample=True):
        w, b = self.sample_weights() if sample else (self.w_mu, self.b_mu)
        return F.linear(x, w, b)

    def kl(self):
        def kg(mu, rho):
            sd = F.softplus(rho)
            return (np.log(self.prior_sd) - torch.log(sd) + (sd ** 2 + mu ** 2) / (2.0 * self.prior_sd ** 2) - 0.5).sum()
        return kg(self.w_mu, self.w_rho) + kg(self.b_mu, self.b_rho)


class VIMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden, depth, prior_sd=1.0, rho_init=-4.0):
        super().__init__()
        dims = [d_in] + [hidden] * (depth - 1) + [n_classes]
        self.layers = nn.ModuleList([BayesLinear(dims[i], dims[i + 1], prior_sd, rho_init) for i in range(len(dims) - 1)])
        self.n_classes = n_classes

    def forward(self, x, sample=True):
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def sample_params(self):
        return [layer.sample_weights() for layer in self.layers]

    def forward_with(self, params, x):
        for i, (w, b) in enumerate(params):
            x = F.linear(x, w, b)
            if i < len(params) - 1:
                x = F.relu(x)
        return x

    def kl(self):
        return sum(layer.kl() for layer in self.layers)


def train_vi(model, X, y, epochs, batch_size, lr, kl_scale, seed=0):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64))
    n = len(Xt)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            b = perm[i:i + batch_size]
            loss = F.cross_entropy(model(Xt[b], sample=True), yt[b]) + kl_scale * model.kl() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


class MAPMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden, depth):
        super().__init__()
        dims = [d_in] + [hidden] * (depth - 1) + [n_classes]
        self.body = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 2)])
        self.head = nn.Linear(dims[-2], dims[-1])
        self.n_classes = n_classes

    def features(self, x):
        for layer in self.body:
            x = F.relu(layer(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


def train_map(model, X, y, epochs, batch_size, lr, weight_decay, seed=0):
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64))
    n = len(Xt)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            b = perm[i:i + batch_size]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_ensemble(X, y, n_members, n_classes, hidden, depth, epochs, batch_size, lr, weight_decay, seed=0):
    models = []
    for m in range(int(n_members)):
        ms = int(seed) * 1000 + m
        torch.manual_seed(ms)
        net = MAPMLP(int(np.asarray(X).shape[1]), n_classes, hidden, depth)
        train_map(net, X, y, epochs, batch_size, lr, weight_decay, ms)
        models.append(net)
    return models


@torch.no_grad()
def _phi(model, X, chunk=20000):
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    out = []
    model.eval()
    for a in range(0, len(Xt), chunk):
        f = model.features(Xt[a:a + chunk]).double().numpy()
        out.append(np.column_stack([f, np.ones(len(f))]))
    return np.concatenate(out)


@torch.no_grad()
def fit_last_layer_laplace(model, X_train, prior_prec, chunk=20000):
    model.eval()
    Phi = _phi(model, X_train, chunk)
    n, hf = Phi.shape
    k = model.n_classes
    Wb = np.column_stack([model.head.weight.double().numpy(), model.head.bias.double().numpy()[:, None]])
    logits = Phi @ Wb.T
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(1, keepdims=True)
    H = np.zeros((k * hf, k * hf))
    for a in range(k):
        for b in range(k):
            w = P[:, a] * ((1.0 if a == b else 0.0) - P[:, b])
            H[a * hf:(a + 1) * hf, b * hf:(b + 1) * hf] = Phi.T @ (w[:, None] * Phi)
    H[np.diag_indices_from(H)] += float(prior_prec)
    H = 0.5 * (H + H.T)
    return {"Wb": Wb, "L": np.linalg.cholesky(H), "hf": hf, "k": k}


def _accumulate(n, k):
    return dict(s1=np.zeros((n, k)), s2=np.zeros((n, k)), sH=np.zeros(n), sHH=np.zeros(n),
                sG=np.zeros(n), sC=np.zeros(n), sCC=np.zeros(n))


def _add(acc, p, sl=slice(None)):
    h, g, c = ent(p), gini(p), p.max(1)
    acc["s1"][sl] += p
    acc["s2"][sl] += p * p
    acc["sH"][sl] += h
    acc["sHH"][sl] += h * h
    acc["sG"][sl] += g
    acc["sC"][sl] += c
    acc["sCC"][sl] += c * c


def _finish(acc, T):
    t = acc["s1"] / T
    var = np.maximum(acc["s2"] / T - t * t, 0.0)
    au_shannon, au_gini = acc["sH"] / T, acc["sG"] / T
    conf_mean = acc["sC"] / T
    return {"t": t, "var": var, "var_sum": var.sum(1),
            "au_shannon": au_shannon, "eu_shannon": np.maximum(ent(t) - au_shannon, 0.0),
            "au_gini": au_gini, "eu_gini": np.maximum(gini(t) - au_gini, 0.0),
            "conf_mean": conf_mean, "conf_std": np.sqrt(np.maximum(acc["sCC"] / T - conf_mean ** 2, 0.0)),
            "ent_std": np.sqrt(np.maximum(acc["sHH"] / T - (acc["sH"] / T) ** 2, 0.0))}


@torch.no_grad()
def mc_stats(model, X, T, seed=0, point_chunk=20000):
    model.eval()
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    n, k = Xt.shape[0], model.n_classes
    acc = _accumulate(n, k)
    torch.manual_seed(seed)
    for _ in range(int(T)):
        params = model.sample_params()
        for a in range(0, n, point_chunk):
            p = F.softmax(model.forward_with(params, Xt[a:a + point_chunk]), dim=1).double().numpy()
            _add(acc, p, slice(a, a + p.shape[0]))
    return _finish(acc, T)


@torch.no_grad()
def mc_stats_laplace(model, post, X, T, seed=0, point_chunk=20000):
    Phi = _phi(model, X, point_chunk)
    n, k, hf = Phi.shape[0], post["k"], post["hf"]
    Wb, L = post["Wb"], post["L"]
    rng = np.random.default_rng(seed)
    acc = _accumulate(n, k)
    for _ in range(int(T)):
        Wb_s = Wb + np.linalg.solve(L.T, rng.standard_normal(k * hf)).reshape(k, hf)
        for a in range(0, n, point_chunk):
            lg = Phi[a:a + point_chunk] @ Wb_s.T
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(1, keepdims=True)
            _add(acc, p, slice(a, a + p.shape[0]))
    return _finish(acc, T)


@torch.no_grad()
def mc_stats_ensemble(models, X, T=None, seed=0, point_chunk=20000):
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    n, k = Xt.shape[0], models[0].n_classes
    M = len(models) if T is None else min(int(T), len(models))
    acc = _accumulate(n, k)
    for net in models[:M]:
        net.eval()
        _add(acc, F.softmax(net(Xt), dim=1).double().numpy())
    return _finish(acc, M)


def save_teacher(path, kind, cfg, mu, sd, X_tr, y_tr, model, post=None):
    blob = {"kind": kind, "cfg": dict(cfg), "mu": np.asarray(mu), "sd": np.asarray(sd),
            "X_tr": np.asarray(X_tr), "y_tr": np.asarray(y_tr)}
    if kind == "ensemble":
        blob["state_dicts"] = [m.state_dict() for m in model]
    else:
        blob["state_dict"] = model.state_dict()
    if post is not None:
        blob["post"] = {k: post[k] for k in ("Wb", "L", "hf", "k")}
    torch.save(blob, path)


def load_teacher(path):
    blob = torch.load(path, weights_only=False)
    cfg, kind = blob["cfg"], blob["kind"]
    d_in = int(np.asarray(blob["X_tr"]).shape[1])

    def norm(a):
        return ((np.asarray(a, dtype=np.float32) - blob["mu"]) / blob["sd"]).astype(np.float32)

    if kind == "vi":
        model = VIMLP(d_in, cfg["n_classes"], cfg["hidden"], cfg["depth"], cfg["prior_sd"], cfg["rho_init"])
        model.load_state_dict(blob["state_dict"])
        stats_fn = lambda X_raw, T, seed=0: mc_stats(model, norm(X_raw), T, seed)
    elif kind == "laplace":
        model = MAPMLP(d_in, cfg["n_classes"], cfg["hidden"], cfg["depth"])
        model.load_state_dict(blob["state_dict"])
        stats_fn = lambda X_raw, T, seed=0: mc_stats_laplace(model, blob["post"], norm(X_raw), T, seed)
    else:
        model = []
        for sdict in blob["state_dicts"]:
            m = MAPMLP(d_in, cfg["n_classes"], cfg["hidden"], cfg["depth"])
            m.load_state_dict(sdict)
            model.append(m)
        stats_fn = lambda X_raw, T, seed=0: mc_stats_ensemble(model, norm(X_raw), T, seed)
    blob["model"], blob["norm"], blob["stats_fn"] = model, norm, stats_fn
    return blob


def cfg_for(kind):
    cfg = dict(extent=EXTENT, sigma=SIGMA, floor_frac=FLOOR_FRAC, n_train=N_TRAIN, grid_side=GRID_SIDE,
               n_classes=K, hidden=HIDDEN, depth=DEPTH, prior_sd=PRIOR_SD, rho_init=RHO_INIT, kl_scale=KL_SCALE,
               epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, mc_samples=T_EVAL, seed=SEED, n_dots=N_DOTS,
               teacher_kind=kind)
    if kind == "laplace":
        cfg.update(weight_decay=WEIGHT_DECAY, prior_prec=PRIOR_PREC)
    elif kind == "ensemble":
        cfg.update(n_ensemble=N_ENSEMBLE, mc_samples=N_ENSEMBLE, weight_decay=WEIGHT_DECAY)
    return cfg


def run_cell(name, out_dir, cfg, p_star_fn):
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(16, (os.cpu_count() or 4) // 2)))
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    X_tr = sample_inputs(cfg["n_train"], cfg["extent"], cfg["sigma"], cfg["floor_frac"], rng)
    y_tr = draw_labels(p_star_fn(X_tr.astype(np.float64)), rng)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    norm = lambda a: ((np.asarray(a, dtype=np.float32) - mu) / sd).astype(np.float32)

    kind, post = cfg["teacher_kind"], None
    if kind == "vi":
        model = VIMLP(2, cfg["n_classes"], cfg["hidden"], cfg["depth"], cfg["prior_sd"], cfg["rho_init"])
        train_vi(model, norm(X_tr), y_tr, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["kl_scale"], cfg["seed"])
        stats_fn = lambda Xn, sd_: mc_stats(model, Xn, cfg["mc_samples"], sd_)
    elif kind == "laplace":
        model = MAPMLP(2, cfg["n_classes"], cfg["hidden"], cfg["depth"])
        train_map(model, norm(X_tr), y_tr, cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["seed"])
        post = fit_last_layer_laplace(model, norm(X_tr), cfg["prior_prec"])
        stats_fn = lambda Xn, sd_: mc_stats_laplace(model, post, Xn, cfg["mc_samples"], sd_)
    else:
        model = train_ensemble(norm(X_tr), y_tr, cfg["n_ensemble"], cfg["n_classes"], cfg["hidden"], cfg["depth"],
                               cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["seed"])
        stats_fn = lambda Xn, sd_: mc_stats_ensemble(model, Xn, cfg["n_ensemble"], sd_)
    save_teacher(os.path.join(out_dir, name + "_teacher.pt"), kind, cfg, mu, sd, X_tr, y_tr, model, post)

    X_g, shape = grid(cfg["extent"], cfg["grid_side"])
    st_g = stats_fn(norm(X_g), cfg["seed"])
    ps_g = p_star_fn(X_g.astype(np.float64))
    truths = true_quantities(ps_g, st_g["t"])
    raws = {k: st_g[k] for k in ("au_shannon", "au_gini", "eu_shannon", "eu_gini")}
    r_g = np.linalg.norm(X_g.astype(np.float64), axis=1)
    inner = r_g <= cfg["sigma"] * 1.5
    acc_inner = float((st_g["t"].argmax(1) == ps_g.argmax(1))[inner].mean())
    dots_rng = np.random.default_rng(cfg["seed"] + 1000)
    dots = X_tr[dots_rng.choice(len(X_tr), size=min(cfg["n_dots"], len(X_tr)), replace=False)]

    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump({"name": name, "cfg": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in cfg.items()},
                   "teacher_acc_inner": acc_inner, "n_train": int(len(X_tr))}, fh, indent=2)
    np.savez_compressed(os.path.join(out_dir, name + "_maps.npz"),
                        X_grid=X_g, shape=np.array(shape), extent=np.float64(cfg["extent"]), dots=dots,
                        p_star=ps_g, t=st_g["t"],
                        **{"true_" + k: v for k, v in truths.items()}, **{"raw_" + k: v for k, v in raws.items()},
                        conf_mean=st_g["conf_mean"], conf_std=st_g["conf_std"], ent_std=st_g["ent_std"], var=st_g["var"])
    return acc_inner


def kl_moments_across_draws(out_dir, tag, kind, seed=0):
    npz = np.load(os.path.join(out_dir, "%s_%s_maps.npz" % (tag, kind)))
    X = npz["X_grid"].astype(np.float64)
    t_mean = np.asarray(npz["t"], float)
    blob = load_teacher(os.path.join(out_dir, "%s_%s_teacher.pt" % (tag, kind)))
    T = int(blob["cfg"]["mc_samples"])
    post = blob["post"]
    Phi = _phi(blob["model"], blob["norm"](X))
    Wb, L, k, hf = post["Wb"], post["L"], post["k"], post["hf"]
    rng = np.random.default_rng(seed)
    s1, s2 = np.zeros(len(X)), np.zeros(len(X))
    for _ in range(T):
        Wb_s = Wb + np.linalg.solve(L.T, rng.standard_normal(k * hf)).reshape(k, hf)
        lg = Phi @ Wb_s.T
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        d = kl(p, t_mean)
        s1 += d
        s2 += d * d
    mean = s1 / T
    return mean, np.maximum(s2 / T - mean ** 2, 0.0)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def iso_r2(a, b):
    inc = stats.spearmanr(a, b).statistic >= 0
    h = np.random.default_rng(METRIC_SEED).permutation(len(a))
    f0, f1 = h[:len(a) // 2], h[len(a) // 2:]
    pred = np.empty_like(b)
    for fit, ev in ((f0, f1), (f1, f0)):
        pred[ev] = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(a[fit], b[fit]).predict(a[ev])
    return float(1.0 - np.sum((b - pred) ** 2) / np.sum((b - b.mean()) ** 2))


def measures(a, b):
    rng = np.random.default_rng(METRIC_SEED)
    return dict(pearson=float(stats.pearsonr(a, b).statistic), spearman=float(stats.spearmanr(a, b).statistic),
                cos=_cos(a, b), cos_null=_cos(rng.permutation(a), b), isoR2=iso_r2(a, b))


def _r90_of(cfg):
    rng = np.random.default_rng(int(cfg["seed"]))
    X = sample_inputs(int(cfg["n_train"]), cfg["extent"], cfg["sigma"], cfg["floor_frac"], rng)
    return float(np.percentile(np.linalg.norm(X.astype(np.float64), axis=1), 90))


def make_metrics(out_dir):
    pairs = [("diversity", "raw_au_shannon", "true_hard_gini"), ("error", "raw_eu_shannon", "true_err_kl_fwd")]
    out = {}
    for pair, pk, tk in pairs:
        out[pair] = {}
        for tag in TAGS:
            for kind in TEACHERS:
                base = os.path.join(out_dir, "%s_%s" % (tag, kind))
                d = np.load(base + "_maps.npz")
                cfg = json.load(open(base + ".json"))
                cut = _r90_of(cfg["cfg"])
                r = np.linalg.norm(d["X_grid"].astype(np.float64), axis=1)
                A, B = np.asarray(d[pk], float), np.asarray(d[tk], float)
                for reg, m in [("all", np.ones(len(r), bool)), ("inner<=%.2f" % cut, r <= cut), ("outer>%.2f" % cut, r > cut)]:
                    key = "%s_%s_" % (tag, kind) + reg.split("<")[0].split(">")[0]
                    out[pair][key] = dict(n=int(m.sum()), region=reg, **measures(A[m], B[m]))
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def make_alternatives_tables(out_dir, tag, kind):
    d = np.load(os.path.join(out_dir, "%s_%s_maps.npz" % (tag, kind)))
    kl_mean, kl_var = kl_moments_across_draws(out_dir, tag, kind)
    np.savez_compressed(os.path.join(out_dir, "%s_%s_klvar.npz" % (tag, kind)), kl_mean=kl_mean, kl_var=kl_var)
    div_t = [("Gini(p*)", np.asarray(d["true_hard_gini"], float)), ("H(p*)", np.asarray(d["true_hard_shannon"], float))]
    div_p = [("AU Shannon", np.asarray(d["raw_au_shannon"], float)), ("AU Gini", np.asarray(d["raw_au_gini"], float))]
    err_t = [("KL(p*||t)", np.asarray(d["true_err_kl_fwd"], float)), ("KL(t||p*)", np.asarray(d["true_err_kl_rev"], float)),
             ("||p*-t||", np.asarray(d["true_err_l2"], float))]
    err_p = [("EU mutual info", np.asarray(d["raw_eu_shannon"], float)), ("EU pred. variance", np.asarray(d["raw_eu_gini"], float)),
             ("Var_m KL(t_m||t)", kl_var)]
    res = {"experiment": tag, "teacher": kind, "diversity": {}, "error": {}}
    for block, targets, proxies in (("diversity", div_t, div_p), ("error", err_t, err_p)):
        for pname, a in proxies:
            for tname, b in targets:
                res[block]["%s | %s" % (pname, tname)] = measures(a, b)
    with open(os.path.join(out_dir, "alternatives_%s_%s.json" % (tag, kind)), "w") as fh:
        json.dump(res, fh, indent=1)


def make_mc_sweep(out_dir, tag, n_sub=6000, reps=5):
    m_big = list(range(1, 11)) + [12, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100]
    pairs = [("AU-diversity", "au_shannon", "true_hard_gini"), ("EU-error", "eu_shannon", "true_err_kl_fwd")]

    def spear(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(stats.spearmanr(a, b).statistic)

    res = {}
    for kind in ("laplace", "ensemble", "vi"):
        npz = np.load(os.path.join(out_dir, "%s_%s_maps.npz" % (tag, kind)))
        blob = load_teacher(os.path.join(out_dir, "%s_%s_teacher.pt" % (tag, kind)))
        X = npz["X_grid"].astype(np.float64)
        idx = np.random.default_rng(0).choice(len(X), n_sub, replace=False)
        Xs = X[idx]
        targets = {t: np.asarray(npz[t], float)[idx] for _, _, t in pairs}
        n_mem = len(blob["model"]) if kind == "ensemble" else None
        Ms = list(range(1, n_mem + 1)) if kind == "ensemble" else m_big
        out = {name: {"M": [], "corr_mean": [], "corr_std": [], "rel_mean": [], "rel_std": []} for name, _, _ in pairs}
        for M in Ms:
            acc = {name: {"c": [], "r": []} for name, _, _ in pairs}
            for rep in range(reps):
                if kind == "ensemble":
                    perm = np.random.default_rng(100 + rep).permutation(n_mem)
                    mods = [blob["model"][i] for i in perm]
                    sa = mc_stats_ensemble(mods[:M], blob["norm"](Xs), M)
                    sb = mc_stats_ensemble(mods[M:2 * M], blob["norm"](Xs), M) if 2 * M <= n_mem else None
                else:
                    sa = blob["stats_fn"](Xs, M, 1000 * rep + 1)
                    sb = blob["stats_fn"](Xs, M, 1000 * rep + 2)
                for name, pk, tk in pairs:
                    acc[name]["c"].append(spear(sa[pk], targets[tk]))
                    if sb is not None:
                        acc[name]["r"].append(spear(sa[pk], sb[pk]))
            for name, _, _ in pairs:
                c = [v for v in acc[name]["c"] if np.isfinite(v)]
                r = [v for v in acc[name]["r"] if np.isfinite(v)]
                out[name]["M"].append(M)
                out[name]["corr_mean"].append(float(np.mean(c)) if c else float("nan"))
                out[name]["corr_std"].append(float(np.std(c)) if c else float("nan"))
                out[name]["rel_mean"].append(float(np.mean(r)) if r else float("nan"))
                out[name]["rel_std"].append(float(np.std(r)) if r else float("nan"))
        res[kind] = out
    with open(os.path.join(out_dir, "mc_sweep_%s.json" % tag), "w") as fh:
        json.dump({"n_sub": n_sub, "reps": reps, "experiment": tag, "res": res}, fh, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    for tag in TAGS:
        for kind in TEACHERS:
            name = "%s_%s" % (tag, kind)
            t0 = time.time()
            acc = run_cell(name, OUT, cfg_for(kind), WORLDS[tag])
            print("%-18s inner acc %.4f   %6.0fs" % (name, acc, time.time() - t0), flush=True)
    make_metrics(OUT)
    make_alternatives_tables(OUT, DETAIL_TAG, DETAIL_KIND)
    make_mc_sweep(OUT, DETAIL_TAG)
    print("done: %s" % OUT)


if __name__ == "__main__":
    main()
