import torch

EPS = 1e-12


def entropy(p, dim=-1):
    return -(p * torch.log(p.clamp_min(EPS))).sum(dim)


def decompose(P):
    t = P.mean(0)
    au = entropy(P).mean(0)
    eu = entropy(t) - au
    return {"t": t, "au": au, "eu": eu}


def gini(p, dim=-1):
    return 1.0 - (p * p).sum(dim)


def variance_decompose(P):
    """Kwon et al. 2020."""
    return {"epistemic": P.var(0, unbiased=False).sum(1), "aleatoric": gini(P).mean(0)}


def epistemic_fraction(au, eu, eps=EPS):
    return eu / (eu + au + eps)
