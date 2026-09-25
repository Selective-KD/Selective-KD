"""ABKD, Wang et al. 2025, following their released code."""
import torch.nn.functional as F

from .base import EPS, Method, teacher_target


def ab_divergence(p_t, p_s, a, b):
    p, q = p_t.clamp_min(EPS), p_s.clamp_min(EPS)
    if a == 0.0 and b == 0.0:
        return 0.5 * ((q.log() - p.log()) ** 2).sum(1)
    if a == 0.0:
        qb, pb = q.pow(b), p.pow(b)
        return (qb * (qb / pb).log() - qb + pb).sum(1) / b
    if b == 0.0:
        pa, qa = p.pow(a), q.pow(a)
        return (pa * (pa / qa).log() - pa + qa).sum(1) / a
    if a + b == 0.0:
        r = q.pow(a) / p.pow(a)
        return ((r.log() + r.reciprocal() - 1.0) / a).sum(1)
    first = p.pow(a) * q.pow(b)
    second = (a / (a + b)) * p.pow(a + b)
    third = (b / (a + b)) * q.pow(a + b)
    return -(first - second - third).sum(1) / (a * b)


class ABKD(Method):
    name = "abkd"
    DEFAULTS = {"alpha": 0.9, "beta": 0.2, "ce_weight": 1.0, "kd_weight": 32.0}

    def __init__(self, T, params):
        self.T = float(T)
        self.alpha, self.beta = float(params["alpha"]), float(params["beta"])
        self.ce_weight, self.kd_weight = float(params["ce_weight"]), float(params["kd_weight"])

    def per_sample(self, logits, P, Z, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        d = ab_divergence(teacher_target(P, Z, self.T), F.softmax(logits / self.T, 1), self.alpha, self.beta)
        return self.ce_weight * ce + self.kd_weight * d

    def loss(self, logits, features, idx, P, Z, y):
        return self.per_sample(logits, P, Z, y).mean()
