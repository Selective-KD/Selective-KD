import torch
import torch.nn.functional as F

EPS = 1e-12


class Method:
    name = "base"
    DEFAULTS = {}

    needs_val = False
    needs_steps = False

    def __init__(self, T, params):
        self.T = float(T)

    def extra_params(self):
        return []

    def set_total_steps(self, n):
        pass

    def prepare(self, P, Z, y, seed):
        pass

    def loss(self, logits, features, idx, P, Z, y):
        raise NotImplementedError

    def summary(self):
        return {}


def teacher_target(P, Z, T):
    return P.mean(0) if T == 1.0 else F.softmax(Z / T, -1).mean(0)


def one_hot(y, num_classes):
    return F.one_hot(y, num_classes).float()


def log_prob(p):
    return p.clamp_min(EPS).log()


def kl(p, logq):
    return (p * (log_prob(p) - logq)).sum(1)


def kd_terms(logits, P, Z, y, T):
    K = logits.shape[1]
    g = one_hot(y, K)
    logp = F.log_softmax(logits, 1)
    ce = -(g * logp).sum(1)
    if T == 1.0:
        t = P.mean(0)
        soft = -(t * logp).sum(1)
        return ce, soft, logp.exp() - g, logp.exp() - t
    pt = teacher_target(P, Z, T)
    logp_T = F.log_softmax(logits / T, 1)
    soft = T * T * (-(pt * logp_T).sum(1))
    return ce, soft, logp.exp() - g, T * (logp_T.exp() - pt)


def head_grad_norms(resid, features):
    return resid.norm(dim=1) * (features * features).sum(1).add(1.0).sqrt()


def balance_weights(g, lam, r_max):
    teach = lam > 0.5
    label = ~teach
    if teach.any() and label.any():
        ratio = (g[label].mean() / g[teach].mean()).clamp(1.0, r_max)
    else:
        ratio = torch.ones((), device=g.device)
    w = 1.0 + (ratio - 1.0) * lam
    return w / w.mean()
