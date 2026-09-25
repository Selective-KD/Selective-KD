import hashlib

import numpy as np

from .corruptions import corrupt
from .pools import balanced_mask, draw


def _rng(seed, tag):
    return np.random.default_rng(int(hashlib.sha256(("%d|%s" % (seed, tag)).encode()).hexdigest()[:8], 16))


def compose(x, y, is_b, region_a, region_b, seed, tag, frost_dir):
    out = np.empty_like(x)
    if (~is_b).any():
        out[~is_b] = corrupt(region_a, x[~is_b], _rng(seed, tag + "|" + region_a), frost_dir)
    if is_b.any():
        out[is_b] = corrupt(region_b, x[is_b], _rng(seed, tag + "|" + region_b), frost_dir)
    return out


def build(x_train, y_train, x_test, y_test, *, region_a, region_b, f, n_teacher, n_student, seed,
          frost_dir="frost", log=print):
    rng = np.random.default_rng(seed)
    t_idx = draw(y_train, np.arange(len(y_train)), n_teacher, rng, np.array([], dtype=np.int64))
    s_idx = draw(y_train, np.arange(len(y_train)), n_student, rng, t_idx)
    out = {}
    log("teacher_train: %d images under %s" % (len(t_idx), region_a))
    xt = compose(x_train[t_idx], y_train[t_idx], np.zeros(len(t_idx), bool), region_a, region_b, seed, "teacher", frost_dir)
    out["teacher_train"] = (xt, y_train[t_idx], np.zeros(len(t_idx), dtype=np.int64))
    for name, x, y, idx in (("student_train", x_train, y_train, s_idx), ("test", x_test, y_test, np.arange(len(y_test)))):
        is_b = balanced_mask(y[idx], f, rng)
        log("%s: %d images, %d under %s, %d under %s" % (name, len(idx), (~is_b).sum(), region_a, is_b.sum(), region_b))
        xc = compose(x[idx], y[idx], is_b, region_a, region_b, seed, name, frost_dir)
        out[name] = (xc, y[idx], is_b.astype(np.int64))
    return out
