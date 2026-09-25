"""Hinton, Vinyals & Dean 2015."""
from .base import Method, kd_terms


class Hinton(Method):
    name = "hinton"
    DEFAULTS = {"lambda": 1.0}

    def __init__(self, T, params):
        self.T, self.lam = float(T), float(params["lambda"])

    def loss(self, logits, features, idx, P, Z, y):
        ce, soft, _, _ = kd_terms(logits, P, Z, y, self.T)
        return ((1.0 - self.lam) * ce + self.lam * soft).mean()
