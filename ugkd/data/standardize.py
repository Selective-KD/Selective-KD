import numpy as np


class Standardizer:
    def __init__(self, per_feature=True, eps=1e-8):
        self.per_feature, self.eps = per_feature, eps
        self.mean = self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.per_feature:
            self.mean, self.std = x.mean(0), x.std(0)
        else:
            self.mean, self.std = x.mean(), x.std()
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def state_dict(self):
        return {"per_feature": self.per_feature, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        s = cls(per_feature=d["per_feature"])
        s.mean, s.std = d["mean"], d["std"]
        return s
