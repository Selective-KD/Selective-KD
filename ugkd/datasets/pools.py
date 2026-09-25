import numpy as np


def draw(y, candidates, n, rng, spent):
    cand = np.setdiff1d(candidates, spent)
    classes = np.unique(y)
    per = n // len(classes)
    out = []
    for c in classes:
        cc = cand[y[cand] == c]
        if len(cc) < per:
            raise ValueError("class %d has %d samples available, need %d" % (c, len(cc), per))
        out.append(rng.choice(cc, size=per, replace=False))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx


def balanced_mask(y, f, rng):
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        mask[rng.permutation(idx)[:int(round(f * len(idx)))]] = True
    return mask


def draw_pools(y, in_a, in_b, *, n_teacher, n_student, n_test, f, seed):
    rng = np.random.default_rng(seed)
    a_idx, b_idx = np.flatnonzero(in_a), np.flatnonzero(in_b)
    spent = np.array([], dtype=np.int64)
    out = {}

    def take(name, cands, n, group):
        nonlocal spent
        idx = draw(y, cands, n, rng, spent)
        spent = np.concatenate([spent, idx])
        return idx, np.full(len(idx), group, dtype=np.int64)

    out["teacher_train"] = take("teacher_train", a_idx, n_teacher, 0)
    for name, n in (("student_train", n_student), ("test", n_test)):
        n_b = int(round(f * n))
        ia, ga = take(name, a_idx, n - n_b, 0)
        ib, gb = take(name, b_idx, n_b, 1) if n_b > 0 else (a_idx[:0], ga[:0])
        idx, group = np.concatenate([ia, ib]), np.concatenate([ga, gb])
        perm = rng.permutation(len(idx))
        out[name] = (idx[perm], group[perm])
    return out
