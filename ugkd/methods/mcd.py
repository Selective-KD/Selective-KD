"""MCD, Mishra, Mishra & Xiong 2026, Eq. 4."""
import torch.nn.functional as F

from .base import Method, kl, teacher_target


class MCD(Method):
    name = "mcd"
    DEFAULTS = {"alpha": 12.0, "beta": 2.0}

    def __init__(self, T, params):
        self.T, self.alpha, self.beta = float(T), float(params["alpha"]), float(params["beta"])

    def loss(self, logits, features, idx, P, Z, y):
        t = P.mean(0)
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = teacher_target(P, Z, self.T)
        kd = self.T ** 2 * kl(pt, F.log_softmax(logits / self.T, 1))
        corr = (t.argmax(1) == y).float()
        return (corr * (self.alpha * kd + ce) + (1.0 - corr) * (kd + self.beta * ce)).mean()
