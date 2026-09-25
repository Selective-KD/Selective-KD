"""TGeo-KD, Hu et al. 2024."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .base import Method, kl, teacher_target


class TGeo(Method):
    name = "tgeo"
    DEFAULTS = {"hidden": (32, 16), "ctrl_lr": 0.1, "ctrl_wd": 0.05}
    needs_val = True

    def __init__(self, T, params):
        self.T = float(T)
        self.hidden = tuple(params["hidden"])
        self.ctrl_lr, self.ctrl_wd = float(params["ctrl_lr"]), float(params["ctrl_wd"])
        self.net = self.opt = self.tbar = None

    def prepare(self, P, Z, y, seed):
        K = P.shape[2]
        t = P.mean(0)
        self.tbar = torch.stack([t[y == c].mean(0) if (y == c).any() else torch.full((K,), 1.0 / K)
                                 for c in range(K)])
        g = torch.Generator().manual_seed(int(seed))
        h1, h2 = self.hidden
        self.net = nn.Sequential(nn.Linear(9 * K, h1), nn.LeakyReLU(), nn.Linear(h1, h2), nn.LeakyReLU(),
                                 nn.Linear(h2, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                nn.init.uniform_(m.bias, -1.0 / m.in_features ** 0.5, 1.0 / m.in_features ** 0.5, generator=g)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.ctrl_lr, weight_decay=self.ctrl_wd)

    def alpha(self, S, P, y):
        t = P.mean(0)
        G = F.one_hot(y, t.shape[1]).float()
        Tb = self.tbar.to(t.device)[y]
        state = torch.cat([S, t, G, Tb, (Tb - S) ** 2, (G - S) ** 2, (t - S) ** 2, (t - G) ** 2, (Tb - G) ** 2], 1)
        return torch.sigmoid(self.net.to(t.device)(state)).squeeze(1)

    def objective(self, logits, alpha, P, Z, y):
        pt = teacher_target(P, Z, self.T)
        kd = kl(pt, F.log_softmax(logits / self.T, 1))
        ce = F.cross_entropy(logits, y, reduction="none")
        return (alpha * self.T ** 2 * kd + (1.0 - alpha) * ce).mean()

    def outer_step(self, model, batch, val, lr):
        xs, y, P, Z = batch
        xv, yv, _, _ = val
        names, params = zip(*[(n, p) for n, p in model.named_parameters()])
        fwd = lambda ps, x: functional_call(model, dict(zip(names, ps)), (x,))
        logits = fwd(params, xs)
        with torch.no_grad():
            S = F.softmax(logits, 1)
        alpha = self.alpha(S, P, y)
        inner = self.objective(logits, alpha, P, Z, y)
        grads = torch.autograd.grad(inner, params, create_graph=True)
        fast = [p - lr * g for p, g in zip(params, grads)]
        val_loss = F.cross_entropy(fwd(fast, xv), yv)
        self.opt.zero_grad()
        val_loss.backward()
        self.opt.step()

    def loss(self, logits, features, idx, P, Z, y):
        with torch.no_grad():
            alpha = self.alpha(F.softmax(logits, 1), P, y)
        return self.objective(logits, alpha, P, Z, y)
