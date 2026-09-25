"""WTTM, Zheng & Yang 2024, Eq. 22; gamma = 1 / T."""
import torch
import torch.nn.functional as F

from .base import Method, kl, log_prob


def power_transform(p_t, gamma):
    logp = log_prob(p_t)
    logU = torch.logsumexp(gamma * logp, dim=1)
    return (gamma * logp - logU[:, None]).exp(), logU.exp()


class WTTM(Method):
    name = "wttm"
    DEFAULTS = {"lambda": 0.5, "ce_weight": 1.0}

    def __init__(self, T, params):
        self.T, self.gamma = float(T), 1.0 / float(T)
        self.lam_kd, self.ce_weight = float(params["lambda"]), float(params["ce_weight"])
        self.beta = None

    def prepare(self, P, Z, y, seed):
        u_mean = float(power_transform(P.mean(0), self.gamma)[1].mean())
        self.beta = self.lam_kd / (1.0 - self.lam_kd) * self.T / u_mean

    def per_sample(self, logits, P, Z, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        phat, U = power_transform(P.mean(0), self.gamma)
        return self.ce_weight * ce + self.beta * U * kl(phat, F.log_softmax(logits, 1))

    def loss(self, logits, features, idx, P, Z, y):
        return self.per_sample(logits, P, Z, y).mean()
