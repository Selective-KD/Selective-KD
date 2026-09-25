"""LUD, Guo et al. 2024, Eq. 6."""
from .base import Method, kd_terms


class LUD(Method):
    name = "lud"
    DEFAULTS = {}

    def __init__(self, T, params):
        self.T = float(T)
        self.lam = None

    def prepare(self, P, Z, y, seed):
        conf = P.mean(0).max(1).values
        self.lam = (conf > conf.median()).float()

    def loss(self, logits, features, idx, P, Z, y):
        lam = self.lam[idx].to(logits.device)
        ce, soft, _, _ = kd_terms(logits, P, Z, y, self.T)
        return ((1.0 - lam) * ce + lam * soft).mean()
