import numpy as np


def permute_split(n, frac, seed):
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1), got %r" % frac)
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def holdout(idx, frac, seed):
    idx = np.asarray(idx)
    if frac <= 0.0:
        return idx, idx[:0]
    held, kept = permute_split(len(idx), frac, seed)
    return idx[kept], idx[held]
