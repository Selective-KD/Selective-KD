"""ugkd."""
import torch

from ..router import EpistemicFractionRouter
from ..uncertainty import decompose, epistemic_fraction
from .base import Method, balance_weights, head_grad_norms, kd_terms

R_MAX = 100.0


def fit_routing(P, seed):
    u = decompose(P)
    r = epistemic_fraction(u["au"], u["eu"]).cpu().numpy()
    router = EpistemicFractionRouter(seed).fit(r)
    return router, torch.as_tensor(router.assign(r))


class UGKD(Method):
    name = "ugkd"
    DEFAULTS = {}

    def __init__(self, T, params):
        self.T = float(T)
        self.router, self.lam = None, None

    def prepare(self, P, Z, y, seed):
        self.router, self.lam = fit_routing(P, seed)

    def loss(self, logits, features, idx, P, Z, y):
        lam = self.lam[idx].to(logits.device)
        ce, soft, resid_ce, resid_soft = kd_terms(logits, P, Z, y, self.T)
        per = (1.0 - lam) * ce + lam * soft
        with torch.no_grad():
            resid = (1.0 - lam)[:, None] * resid_ce + lam[:, None] * resid_soft
            w = balance_weights(head_grad_norms(resid, features), lam, R_MAX)
        return (w * per).mean()

    def summary(self):
        return {"frac_to_teacher": float(self.lam.mean())}
