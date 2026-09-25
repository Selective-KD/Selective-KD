"""<method>_ugkd: our routing around a published objective."""
import torch
import torch.nn.functional as F

from .abkd import ABKD
from .base import Method, balance_weights, head_grad_norms
from .dkd import DKD
from .lsd import LSD
from .ours import R_MAX, fit_routing
from .wls import WLS
from .wttm import WTTM


class Fused(Method):
    inner_cls = None

    def __init__(self, T, params):
        self.T = float(T)
        self.inner = self.inner_cls(T, params)
        self.router, self.lam = None, None

    @property
    def needs_steps(self):
        return self.inner.needs_steps

    def set_total_steps(self, n):
        self.inner.set_total_steps(n)

    def prepare(self, P, Z, y, seed):
        self.router, self.lam = fit_routing(P, seed)
        self.inner.prepare(P, Z, y, seed)

    def loss(self, logits, features, idx, P, Z, y):
        lam = self.lam[idx].to(logits.device)
        ce = F.cross_entropy(logits, y, reduction="none")
        per = (1.0 - lam) * ce + lam * self.inner.per_sample(logits, P, Z, y)
        resid = torch.autograd.grad(per.sum(), logits, retain_graph=True)[0]
        with torch.no_grad():
            w = balance_weights(head_grad_norms(resid, features), lam, R_MAX)
        return (w * per).mean()

    def summary(self):
        return {"frac_to_teacher": float(self.lam.mean())}


def _fusion(inner):
    return type(inner.name.upper() + "UGKD", (Fused,), {
        "name": inner.name + "_ugkd",
        "inner_cls": inner,
        "DEFAULTS": inner.DEFAULTS,
    })


DKDUGKD, ABKDUGKD, LSDUGKD, WTTMUGKD, WLSUGKD = (_fusion(c) for c in (DKD, ABKD, LSD, WTTM, WLS))
