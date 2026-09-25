"""UBKD, Hemmatian, Shahzadi & Mozaffari 2024, Eq. 20; T0 is the script temperature."""
import torch
import torch.nn.functional as F

from ..uncertainty import variance_decompose
from .base import Method

PUB_A0, PUB_BETA_A = 0.95, 0.6


class UBKD(Method):
    name = "ubkd"
    DEFAULTS = {"a0": 0.95, "beta_a": 0.6, "beta_T": 0.0, "anchor_q": 50.0}

    def __init__(self, T, params):
        self.T0, self.a0 = float(T), float(params["a0"])
        self.beta_a, self.beta_T, self.anchor_q = params["beta_a"], float(params["beta_T"]), float(params["anchor_q"])
        self.a = self.Ti = None

    def prepare(self, P, Z, y, seed):
        u = variance_decompose(P)
        K = P.shape[2]
        if self.beta_a == "auto":
            a_min = PUB_A0 - PUB_BETA_A * (1.0 - 1.0 / K)
            scale = torch.quantile(u["epistemic"], self.anchor_q / 100.0)
            self.beta_a = float((self.a0 - a_min) / scale)
        self.a = (self.a0 - float(self.beta_a) * u["epistemic"]).clamp(0.0, 1.0)
        self.Ti = (self.T0 - self.beta_T * u["aleatoric"]).clamp(1.0, self.T0)

    def loss(self, logits, features, idx, P, Z, y):
        a, Ti = self.a[idx].to(logits.device), self.Ti[idx].to(logits.device)
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = F.softmax(Z / Ti[None, :, None], -1).mean(0)
        soft = -(pt * F.log_softmax(logits / Ti[:, None], 1)).sum(1)
        return ((1.0 - a) * ce + a * Ti ** 2 * soft).mean()
