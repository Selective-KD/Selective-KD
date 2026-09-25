"""DKD, Zhao et al. 2022, following mdistiller."""
import torch
import torch.nn.functional as F

from .base import Method, kl, log_prob, teacher_target

LOGIT_MASK = -1000.0


class DKD(Method):
    name = "dkd"
    DEFAULTS = {"alpha": 1.0, "beta": 8.0, "ce_weight": 1.0, "warmup_frac": 20.0 / 240.0}
    needs_steps = True

    def __init__(self, T, params):
        self.T = float(T)
        self.alpha, self.beta = float(params["alpha"]), float(params["beta"])
        self.ce_weight, self.warmup_frac = float(params["ce_weight"]), float(params["warmup_frac"])
        self.warmup_steps, self.step = 1.0, 0

    def set_total_steps(self, n):
        self.warmup_steps = max(1.0, self.warmup_frac * n)

    def per_sample(self, logits, P, Z, y):
        self.step += 1
        K = logits.shape[1]
        ce = F.cross_entropy(logits, y, reduction="none")
        p_s, p_t = F.softmax(logits / self.T, 1), teacher_target(P, Z, self.T)
        mask = F.one_hot(y, K).bool()
        b_s = torch.stack([p_s[mask], 1.0 - p_s[mask]], 1)
        b_t = torch.stack([p_t[mask], 1.0 - p_t[mask]], 1)
        tckd = kl(b_t, log_prob(b_s)) * self.T ** 2
        logphat_s = F.log_softmax(logits / self.T + LOGIT_MASK * mask, 1)
        phat_t = F.softmax(log_prob(p_t) + LOGIT_MASK * mask, 1)
        nckd = kl(phat_t, logphat_s) * self.T ** 2
        warm = min(self.step / self.warmup_steps, 1.0)
        return self.ce_weight * ce + warm * (self.alpha * tckd + self.beta * nckd)

    def loss(self, logits, features, idx, P, Z, y):
        return self.per_sample(logits, P, Z, y).mean()
