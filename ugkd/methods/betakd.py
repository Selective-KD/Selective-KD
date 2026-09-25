"""Beta-KD, Sun et al., task level, Eq. 15."""
import math

import torch
import torch.nn.functional as F

from .base import Method, kl, teacher_target

LOG_MIN, LOG_MAX = -5.0, 5.0


class BetaKD(Method):
    name = "betakd"
    DEFAULTS = {"init_beta": 1.0}

    def __init__(self, T, params):
        self.T = float(T)
        init = float(params["init_beta"])
        self.s = torch.nn.Parameter(torch.tensor([math.log(math.expm1(init))]))
        self.d = None

    def prepare(self, P, Z, y, seed):
        self.d = float(P.shape[2])

    def extra_params(self):
        return [self.s]

    def beta(self):
        raw = F.softplus(self.s).squeeze(0)
        z = torch.sigmoid(raw.log() - 0.5 * (LOG_MIN + LOG_MAX))
        return torch.exp(LOG_MIN + (LOG_MAX - LOG_MIN) * z)

    def loss(self, logits, features, idx, P, Z, y):
        ce = F.cross_entropy(logits, y)
        pt = teacher_target(P, Z, self.T)
        kd = self.T ** 2 * kl(pt, F.log_softmax(logits / self.T, 1)).mean()
        beta = self.beta().to(logits.device)
        return ce + beta * kd - 0.5 * self.d * beta.log()
