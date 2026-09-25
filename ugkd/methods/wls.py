"""WLS, Zhou et al. 2021, following their released code."""
import torch
import torch.nn.functional as F

from .base import Method, log_prob, teacher_target


class WLS(Method):
    name = "wls"
    DEFAULTS = {"alpha": 2.25}

    def __init__(self, T, params):
        self.T, self.alpha = float(T), float(params["alpha"])

    def per_sample(self, logits, P, Z, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = teacher_target(P, Z, self.T)
        soft = -(pt * F.log_softmax(logits / self.T, 1)).sum(1)
        with torch.no_grad():
            ce_t = -log_prob(P.mean(0).gather(1, y[:, None]).squeeze(1))
            w = 1.0 - torch.exp(-ce / (ce_t + 1e-7))
        return ce + self.alpha * self.T ** 2 * w * soft

    def loss(self, logits, features, idx, P, Z, y):
        return self.per_sample(logits, P, Z, y).mean()
