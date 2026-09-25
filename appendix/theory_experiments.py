import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, K = 30, 8
CENTER_SEED, DATA_SEED, REF_SEED, SPLIT_SEED = 0, 0, 777, 11
N_TRAIN, N_VAL, N_TEST, N_TEST_ROUTING = 5_000, 2_000, 10_000, 10_000
HIDDEN, ETA, N_ITERS, EVAL_EVERY = 128, 0.05, 20_000, 100
TAIL_ITERS, WARMUP = 5_000, 5_000
C0, EPS, S_MAX = 2.0, 1e-6, 0.5

TEACHER_ERROR_GINI = 0.20
CURVES_S, CURVES_SEEDS = [0.0, 0.1, 0.2], 5
SWEEP_S, SWEEP_SEEDS = [round(0.05 * i, 2) for i in range(21)], 10
CURVES_GINIS = [0.001, 0.05, 0.2, 0.5]
SWEEP_GINIS = [round(0.01 + 0.02 * i, 2) for i in range(25)] + [0.5]
HEATMAP_GINIS = [0.01, 0.02, 0.05, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58]
HEATMAP_S = [0.0, 0.05, 0.10, 0.15, 0.20, 0.251, 0.301, 0.351, 0.401, 0.451]
HEATMAP_SEEDS = 10
ROUTING_GINI, ROUTING_SEEDS = 0.50, 10
ROUTING_F = [round(0.1 * i, 2) for i in range(11)]
ROUTING_LAMBDAS = [round(0.1 * i, 2) for i in range(11)]
CROSSOVER_F = {"halfspace": 0.40, "diversity": 0.17, "radial": 0.25}

MAX_WORKERS = os.cpu_count() or 1
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def draw_centers(seed=CENTER_SEED):
    return np.random.default_rng(seed).integers(-1, 2, size=(K, D)).astype(np.float64)


def true_bcp(X, mu):
    lo = X @ mu.T - 0.5 * np.sum(mu * mu, 1)[None, :]
    lo -= lo.max(1, keepdims=True)
    e = np.exp(lo)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def sample(mu, n, rng):
    y = rng.integers(0, K, n)
    return (rng.standard_normal((n, D)) + mu[y]).astype(np.float32), y.astype(np.int64)


def point_gini(P):
    return 1 - np.sum(P.astype(np.float64) ** 2, 1)


def world_stats(mu, n=20_000, seed=999):
    x, _ = sample(mu, n, np.random.default_rng(seed))
    P = true_bcp(x, mu)
    return (float(point_gini(P).mean()), float(1 - np.mean(np.max(P, 1))),
            float(np.mean(-np.sum(P * np.log(np.clip(P, 1e-12, 1)), 1))))


def alphas_for_target_gini(mu, targets):
    grid = np.linspace(0.05, 3.0, 150)
    g = np.array([world_stats(a * mu)[0] for a in grid])
    o = np.argsort(g)
    return [float(np.interp(t, g[o], grid[o])) for t in targets]


MU = draw_centers()


def teacher_mean(P, s):
    if s <= 0:
        return P.astype(np.float64)
    m = P.astype(np.float64).copy()
    idx = np.argsort(-P, axis=1)
    r = np.arange(len(P))
    m[r, idx[:, 0]] -= s
    m[r, idx[:, 1]] += s
    m = np.clip(m, EPS, None)
    return m / m.sum(1, keepdims=True)


def teacher_draw(mean_row, s, rng):
    return mean_row.astype(np.float32) if s <= 0 else rng.dirichlet((C0 / s) * mean_row).astype(np.float32)


def runnerup_onehot(P):
    idx = np.argsort(-P, axis=1)
    cw = np.zeros_like(P, dtype=np.float32)
    cw[np.arange(len(P)), idx[:, 1]] = 1.0
    return cw


def routing_targets(good, Ptr, oh, kind, lam):
    if kind == "route":
        return np.where(good[:, None], Ptr, oh).astype(np.float32)
    Tea = Ptr.copy()
    Tea[~good] = runnerup_onehot(Ptr[~good])
    return ((1 - lam) * oh + lam * Tea).astype(np.float32)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, K))

    def forward(self, x):
        return self.net(x)


def make_split(mu, n_test=N_TEST):
    rng = np.random.default_rng(DATA_SEED)
    Xtr, ytr = sample(mu, N_TRAIN, rng)
    Xva, yva = sample(mu, N_VAL, rng)
    Xte, yte = sample(mu, n_test, rng)
    return Xtr, ytr, true_bcp(Xtr, mu), Xva, yva, Xte, yte


def train(target, Xtr, Xva, yva, Xte, yte, seed):
    torch.manual_seed(seed)
    m = MLP()
    trng = np.random.default_rng(7000 + seed)
    order = np.random.default_rng(500 + seed).integers(0, N_TRAIN, N_ITERS)
    Xtr_t, Xva_t, Xte_t = torch.from_numpy(Xtr), torch.from_numpy(Xva), torch.from_numpy(Xte)
    yva_t, yte_t = torch.from_numpy(yva), torch.from_numpy(yte)
    its, te_acc, te_ce, va_acc = [], [], [], []
    for it in range(N_ITERS):
        idx = order[it]
        t = torch.from_numpy(np.asarray(target(idx, trng), np.float32)).unsqueeze(0)
        m.zero_grad(set_to_none=True)
        (-(t * F.log_softmax(m(Xtr_t[idx:idx + 1]), 1)).sum()).backward()
        with torch.no_grad():
            for p in m.parameters():
                p -= ETA * p.grad
        if it % EVAL_EVERY == 0:
            with torch.no_grad():
                lp = F.log_softmax(m(Xte_t), 1)
                its.append(it)
                te_acc.append((lp.argmax(1) == yte_t).float().mean().item() * 100)
                te_ce.append(F.nll_loss(lp, yte_t).item())
                va_acc.append((m(Xva_t).argmax(1) == yva_t).float().mean().item() * 100)
    return np.array(its), np.array(te_acc), np.array(te_ce), np.array(va_acc)


def onehot_target(ytr):
    eye = np.eye(K, dtype=np.float32)
    return lambda idx, rng: eye[ytr[idx]]


def teacher_target(Ptr, s):
    mean = teacher_mean(Ptr, s)
    return lambda idx, rng: teacher_draw(mean[idx], s, rng)


def matrix_target(T):
    return lambda idx, rng: T[idx]


def readout_plateau(its, te):
    return float(te[its >= (N_ITERS - TAIL_ITERS)].mean())


def readout_bestval(its, te, va, warmup=0):
    ok = its >= warmup
    return float(te[ok][int(np.argmax(va[ok]))])


def parallel(fn, tasks, desc):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        for i, r in enumerate(ex.map(fn, tasks, chunksize=1), 1):
            out.append(r)
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                print("    [%s] %d/%d  (%.0fs)" % (desc, i, len(tasks), time.time() - t0), flush=True)
    return out


def world(alpha):
    mu = alpha * MU
    g, be, bce = world_stats(mu)
    return mu, dict(alpha=alpha, gini=g, bayes_acc=(1 - be) * 100, bayes_ce=bce)


TEACHER_ERROR_ALPHA = alphas_for_target_gini(MU, [TEACHER_ERROR_GINI])[0]
CURVES_ALPHAS = alphas_for_target_gini(MU, CURVES_GINIS)
SWEEP_ALPHAS = alphas_for_target_gini(MU, SWEEP_GINIS)
HEATMAP_ALPHAS = alphas_for_target_gini(MU, HEATMAP_GINIS)
ROUTING_ALPHA = alphas_for_target_gini(MU, [ROUTING_GINI])[0]
ROUTING_MU = ROUTING_ALPHA * MU
REF_X = sample(ROUTING_MU, 20_000, np.random.default_rng(REF_SEED))[0]
HALFSPACE_V = np.random.default_rng(SPLIT_SEED).standard_normal(D)
HALFSPACE_V /= np.linalg.norm(HALFSPACE_V)


def teacher_error_worker(task):
    torch.set_num_threads(1)
    si, seed, kind = task
    Xtr, ytr, Ptr, Xva, yva, Xte, yte = make_split(TEACHER_ERROR_ALPHA * MU)
    if kind == "onehot":
        return si, seed, kind, train(onehot_target(ytr), Xtr, Xva, yva, Xte, yte, seed)
    s = (CURVES_S if kind == "curves" else SWEEP_S)[si]
    return si, seed, kind, train(teacher_target(Ptr, s), Xtr, Xva, yva, Xte, yte, seed)


def teacher_error_curves():
    print("[teacher_error_curves]", flush=True)
    _, w = world(TEACHER_ERROR_ALPHA)
    nL, nS = len(CURVES_S), CURVES_SEEDS
    got = parallel(teacher_error_worker, [(si, s, "curves") for s in range(nS) for si in range(nL)], "teacher_error_curves")
    its = got[0][3][0]
    T = len(its)
    acc = np.full((nS, nL, T), np.nan)
    ce, val = acc.copy(), acc.copy()
    for si, s, kind, (_, a, c, v) in got:
        acc[s, si], ce[s, si], val[s, si] = a, c, v
    p = "teacher_error_curves_"
    return {p + "iters": its, p + "s": np.array(CURVES_S), p + "smax": S_MAX, p + "seeds": nS, p + "gini": w["gini"],
            p + "alpha": w["alpha"], p + "bayes_acc": w["bayes_acc"], p + "bayes_ce": w["bayes_ce"],
            p + "acc": acc, p + "ce": ce, p + "val": val}


def teacher_error_sweep():
    print("[teacher_error_sweep]", flush=True)
    _, w = world(TEACHER_ERROR_ALPHA)
    nL, nS = len(SWEEP_S), SWEEP_SEEDS
    tasks = [(si, s, "sweep") for si in range(nL) for s in range(nS)] + [(0, s, "onehot") for s in range(nS)]
    got = parallel(teacher_error_worker, tasks, "teacher_error_sweep")
    T = len(got[0][3][0])
    acc = np.full((nL, nS, T), np.nan, np.float32)
    ce, val = acc.copy(), acc.copy()
    oacc = np.full((nS, T), np.nan, np.float32)
    oce, oval = oacc.copy(), oacc.copy()
    for si, s, kind, (its, a, c, v) in got:
        if kind == "onehot":
            oacc[s], oce[s], oval[s] = a, c, v
        else:
            acc[si, s], ce[si, s], val[si, s] = a, c, v
    p = "teacher_error_sweep_"
    return {p + "s": np.array(SWEEP_S), p + "smax": S_MAX, p + "gini": w["gini"], p + "alpha": w["alpha"],
            p + "bayes_acc": w["bayes_acc"], p + "bayes_ce": w["bayes_ce"], p + "iters": got[0][3][0],
            p + "acc": acc, p + "ce": ce, p + "val": val, p + "oh_acc": oacc, p + "oh_ce": oce, p + "oh_val": oval}


def difficulty_worker(task):
    torch.set_num_threads(1)
    which, j, seed, kind = task
    Xtr, ytr, Ptr, Xva, yva, Xte, yte = make_split((CURVES_ALPHAS if which == "curves" else SWEEP_ALPHAS)[j] * MU)
    target = onehot_target(ytr) if kind == "onehot" else matrix_target(Ptr)
    return which, j, seed, kind, train(target, Xtr, Xva, yva, Xte, yte, seed)


def difficulty_common(which, alphas, ginis, seeds, p):
    ws = [world(a)[1] for a in alphas]
    nG, nS, arms = len(alphas), seeds, ["bcp", "onehot"]
    got = parallel(difficulty_worker, [(which, j, s, k) for j in range(nG) for s in range(nS) for k in arms], p.rstrip("_"))
    T = len(got[0][4][0])
    acc = np.full((nG, nS, 2, T), np.nan, np.float32)
    ce, val = acc.copy(), acc.copy()
    for _, j, s, k, (its, a, c, v) in got:
        acc[j, s, arms.index(k)], ce[j, s, arms.index(k)], val[j, s, arms.index(k)] = a, c, v
    return {p + "ginis_target": np.array(ginis), p + "gini": np.array([w["gini"] for w in ws]),
            p + "alpha": np.array([w["alpha"] for w in ws]), p + "bayes_acc": np.array([w["bayes_acc"] for w in ws]),
            p + "bayes_ce": np.array([w["bayes_ce"] for w in ws]), p + "iters": got[0][4][0],
            p + "arms": np.array(arms), p + "acc": acc, p + "ce": ce, p + "val": val}


def difficulty_curves():
    print("[difficulty_curves]", flush=True)
    return difficulty_common("curves", CURVES_ALPHAS, CURVES_GINIS, CURVES_SEEDS, "difficulty_curves_")


def difficulty_sweep():
    print("[difficulty_sweep]", flush=True)
    return difficulty_common("sweep", SWEEP_ALPHAS, SWEEP_GINIS, SWEEP_SEEDS, "difficulty_sweep_")


def heatmap_worker(task):
    torch.set_num_threads(1)
    seed, j = task
    Xtr, ytr, Ptr, Xva, yva, Xte, yte = make_split(HEATMAP_ALPHAS[j] * MU)
    curves = [train(onehot_target(ytr), Xtr, Xva, yva, Xte, yte, seed)]
    for s in HEATMAP_S:
        curves.append(train(teacher_target(Ptr, s), Xtr, Xva, yva, Xte, yte, seed))
    return seed, j, curves


def heatmap():
    print("[heatmap]", flush=True)
    ws = [world(a)[1] for a in HEATMAP_ALPHAS]
    nG, nS, nR = len(HEATMAP_GINIS), len(HEATMAP_S), HEATMAP_SEEDS
    got = parallel(heatmap_worker, [(r, j) for r in range(nR) for j in range(nG)], "heatmap")
    its = got[0][2][0][0]
    T = len(its)
    acc = np.full((nR, nG, 1 + nS, T), np.nan, np.float32)
    ce, val = acc.copy(), acc.copy()
    for seed, j, curves in got:
        for a, (_, te, tc, va) in enumerate(curves):
            acc[seed, j, a], ce[seed, j, a], val[seed, j, a] = te, tc, va

    def grid(readout):
        d = np.zeros((nR, nS, nG))
        for r in range(nR):
            for j in range(nG):
                oh = readout(its, acc[r, j, 0], val[r, j, 0])
                for si in range(nS):
                    d[r, si, j] = oh - readout(its, acc[r, j, 1 + si], val[r, j, 1 + si])
        return d.mean(0)

    p = "heatmap_"
    return {p + "ginis_target": np.array(HEATMAP_GINIS), p + "gini": np.array([w["gini"] for w in ws]),
            p + "alpha": np.array([w["alpha"] for w in ws]), p + "bayes_acc": np.array([w["bayes_acc"] for w in ws]),
            p + "slevels": np.array(HEATMAP_S), p + "snorm": np.array([s / max(HEATMAP_S) for s in HEATMAP_S]),
            p + "iters": its, p + "arms": np.array(["onehot"] + ["s=%s" % s for s in HEATMAP_S]),
            p + "acc": acc, p + "ce": ce, p + "val": val,
            p + "diff_plateau": grid(lambda i, te, va: readout_plateau(i, te)),
            p + "diff_bestval": grid(lambda i, te, va: readout_bestval(i, te, va)),
            p + "diff_bestval_warm": grid(lambda i, te, va: readout_bestval(i, te, va, WARMUP))}


def split_score(X, split):
    if split == "halfspace":
        return X @ HALFSPACE_V
    if split == "radial":
        return -np.linalg.norm(X, axis=1)
    return point_gini(true_bcp(X, ROUTING_MU))


def good_mask(X, f, split):
    if f <= 0:
        return np.ones(len(X), bool)
    if f >= 1:
        return np.zeros(len(X), bool)
    return split_score(X, split) >= float(np.quantile(split_score(REF_X, split), f))


def routing_worker(task):
    torch.set_num_threads(1)
    split, f, seed, kind, lam = task
    Xtr, ytr, Ptr, Xva, yva, Xte, yte = make_split(ROUTING_MU, n_test=N_TEST_ROUTING)
    good = good_mask(Xtr, f, split)
    tgt = routing_targets(good, Ptr, np.eye(K, dtype=np.float32)[ytr], kind, lam)
    curves = train(matrix_target(tgt), Xtr, Xva, yva, Xte, yte, seed)
    return split, f, seed, kind, lam, curves, float(1 - good_mask(Xte, f, split).mean())


def routing_readouts(its, acc, val):
    def bv(w):
        ok = its >= w
        return np.take_along_axis(acc[..., ok], val[..., ok].argmax(-1)[..., None], -1)[..., 0]
    return bv(0), bv(WARMUP)


def routing(split):
    p = "routing_%s_" % split
    print("[routing_%s]" % split, flush=True)
    _, w = world(ROUTING_ALPHA)
    fs, cross = list(ROUTING_F), CROSSOVER_F[split]
    extra = [] if any(np.isclose(cross, fs)) else [cross]
    nL, nS = len(ROUTING_LAMBDAS), ROUTING_SEEDS
    tasks = ([(split, f, s, "const", lam) for f in fs + extra for lam in ROUTING_LAMBDAS for s in range(nS)]
             + [(split, f, s, "route", None) for f in fs + extra for s in range(nS)])
    got = parallel(routing_worker, tasks, p.rstrip("_"))
    its = got[0][5][0]
    T = len(its)

    def block(f_list):
        n = len(f_list)
        cacc = np.full((n, nL, nS, T), np.nan, np.float32)
        cce, cval = cacc.copy(), cacc.copy()
        racc = np.full((n, nS, T), np.nan, np.float32)
        rce, rval = racc.copy(), racc.copy()
        bad = np.zeros((n, nS))
        for _, f, seed, kind, lam, (_, a, c, v), bf in got:
            if not any(np.isclose(f, f_list)):
                continue
            fi = int(np.argmin(np.abs(np.array(f_list) - f)))
            bad[fi, seed] = bf
            if kind == "route":
                racc[fi, seed], rce[fi, seed], rval[fi, seed] = a, c, v
            else:
                li = int(np.argmin(np.abs(np.array(ROUTING_LAMBDAS) - lam)))
                cacc[fi, li, seed], cce[fi, li, seed], cval[fi, li, seed] = a, c, v
        cbv, cbvw = routing_readouts(its, cacc, cval)
        rbv, rbvw = routing_readouts(its, racc, rval)
        return dict(const_acc=cacc, const_ce=cce, const_val=cval, route_acc=racc, route_ce=rce, route_val=rval,
                    bad_frac_test=bad.mean(1), const_bv=cbv, const_bv_warm=cbvw, route_bv=rbv, route_bv_warm=rbvw)

    out = {p + "split": split, p + "f": np.array(fs), p + "lambdas": np.array(ROUTING_LAMBDAS), p + "gini": w["gini"],
           p + "alpha": w["alpha"], p + "bayes_acc": w["bayes_acc"], p + "iters": its, p + "crossover_f": cross,
           p + "warmup": WARMUP}
    for k, v in block(fs).items():
        out[p + k] = v
    for k, v in block([cross]).items():
        out[p + "crossover_" + k] = v[0]
    return out


EXPERIMENTS = [teacher_error_curves, teacher_error_sweep, difficulty_curves, difficulty_sweep, heatmap,
               lambda: routing("halfspace"), lambda: routing("diversity"), lambda: routing("radial")]


def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "theory_results.npz")
    results = {}
    for run in EXPERIMENTS:
        results.update(run())
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in results.items()})
        print("    saved %s  (%.0fs)" % (path, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
