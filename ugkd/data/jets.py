import numpy as np

from .standardize import Standardizer


def load_npz(path):
    f = np.load(path)
    if "x" not in f or "y" not in f:
        raise ValueError("%s must contain arrays 'x' and 'y'" % path)
    x, y = np.asarray(f["x"], dtype=np.float32), np.asarray(f["y"], dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("expected x (N, d) and y (N,), got %s and %s" % (x.shape, y.shape))
    group = np.asarray(f["group"], dtype=np.int64) if "group" in f else None
    return x, y, group


def standardize(x):
    st = Standardizer(per_feature=True)
    return st.fit_transform(x), st
