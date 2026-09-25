"""LSD, Sun et al. 2024, following their released code; tau is the script temperature."""
import torch.nn.functional as F

from .base import Method, kl, log_prob


def zscore(x, tau):
    return (x - x.mean(-1, keepdim=True)) / (1e-7 + x.std(-1, keepdim=True)) / tau


class LSD(Method):
    name = "lsd"
    DEFAULTS = {"ce_weight": 0.1, "kd_weight": 9.0}

    def __init__(self, T, params):
        self.tau = float(T)
        self.ce_weight, self.kd_weight = float(params["ce_weight"]), float(params["kd_weight"])

    def per_sample(self, logits, P, Z, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        q_t = F.softmax(zscore(log_prob(P.mean(0)), self.tau), 1)
        logq_s = F.log_softmax(zscore(logits, self.tau), 1)
        return self.ce_weight * ce + self.kd_weight * self.tau ** 2 * kl(q_t, logq_s)

    def loss(self, logits, features, idx, P, Z, y):
        return self.per_sample(logits, P, Z, y).mean()
