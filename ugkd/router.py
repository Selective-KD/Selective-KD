import numpy as np

GMM_KWARGS = dict(n_components=2, covariance_type="full", n_init=10, reg_covar=1e-6,
                  max_iter=100, tol=1e-3)


class EpistemicFractionRouter:
    def __init__(self, seed=0):
        self.seed = seed
        self.gm = self.ignorant = None

    def fit(self, r):
        from sklearn.mixture import GaussianMixture
        r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
        self.gm = GaussianMixture(random_state=self.seed, **GMM_KWARGS).fit(r)
        self.ignorant = int(np.argmax(self.gm.means_.ravel()))
        return self

    def assign(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 1)
        return (self.gm.predict(r) != self.ignorant).astype(np.float32)
