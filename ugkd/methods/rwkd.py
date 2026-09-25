"""RW-KD, Lu et al. 2021, Algorithm 1."""
import torch
from torch.func import functional_call

from .base import Method, kd_terms


class RWKD(Method):
    name = "rwkd"
    DEFAULTS = {"delta": 1e-8, "beta": 1.0}
    needs_val = True

    def __init__(self, T, params):
        self.T, self.delta, self.beta = float(T), float(params["delta"]), float(params["beta"])
        self.lam = None

    def _terms(self, logits, P, Z, y):
        ce, soft, _, _ = kd_terms(logits, P, Z, y, self.T)
        return ce, soft

    def outer_step(self, model, batch, val, lr):
        xs, y, P, Z = batch
        xv, yv, Pv, Zv = val
        names, params = zip(*[(n, p) for n, p in model.named_parameters()])
        fwd = lambda ps, x: functional_call(model, dict(zip(names, ps)), (x,))
        n = len(y)
        e_ce = torch.zeros(n, device=y.device, requires_grad=True)
        e_kd = torch.zeros(n, device=y.device, requires_grad=True)
        l_ce, l_kd = self._terms(fwd(params, xs), P, Z, y)
        inner = (e_ce * l_ce + e_kd * l_kd).sum()
        grads = torch.autograd.grad(inner, params, create_graph=True)
        fast = [p - lr * g for p, g in zip(params, grads)]
        v_ce, v_kd = self._terms(fwd(fast, xv), Pv, Zv, yv)
        meta = (v_ce + v_kd).mean()
        u_ce, u_kd = torch.autograd.grad(meta, [e_ce, e_kd])
        u_ce, u_kd = (-self.beta * u_ce).clamp_min(self.delta), (-self.beta * u_kd).clamp_min(self.delta)
        self.lam = (u_kd / (u_ce + u_kd)).detach()

    def loss(self, logits, features, idx, P, Z, y):
        lam = self.lam if self.lam is not None else torch.full((len(y),), 0.5, device=logits.device)
        ce, soft = self._terms(logits, P, Z, y)
        return ((1.0 - lam) * ce + lam * soft).mean()
