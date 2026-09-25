import torch

from ..train.batches import iterate


class _Loader:
    def __init__(self, batches, batch_size):
        self.batches, self.batch_size, self.dataset = batches, batch_size, range(batches.n)

    def __iter__(self):
        for idx in iterate(self.batches, self.batch_size, train=False):
            yield self.batches.get(idx, train=False)


class LibraryLaplace:
    def __init__(self, la, structure, tau):
        self.la, self.structure, self.tau = la, structure, float(tau)

    @staticmethod
    def _make(model, structure, tau):
        from laplace import Laplace
        return Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure=structure,
                       prior_precision=1.0 if tau == "marglik" else float(tau), last_layer_name="head")

    @classmethod
    def fit(cls, model, batches, structure, tau, batch_size=512, device="cpu"):
        model.eval()
        la = cls._make(model, structure, tau)
        la.fit(_Loader(batches, batch_size))
        if tau == "marglik":
            la.optimize_prior_precision(method="marglik")
        return cls(la, structure, float(la.prior_precision))

    @torch.no_grad()
    def draws(self, features, n_draws, seed, temperature=1.0):
        gen = torch.Generator(device=features.device).manual_seed(int(seed))
        theta = self.la.sample(n_draws, generator=gen)
        d = features.shape[1]
        K = theta.shape[1] // (d + 1)
        W, b = theta[:, :K * d].reshape(n_draws, K, d), theta[:, K * d:]
        Z = (features.float() @ W.transpose(1, 2) + b[:, None, :]) / float(temperature)
        return torch.softmax(Z, -1), Z

    def state_dict(self):
        return {"backend": "library", "structure": self.structure, "tau": self.tau, "state": self.la.state_dict()}

    @classmethod
    def from_state_dict(cls, d, model):
        la = cls._make(model, d["structure"], d["tau"])
        la.load_state_dict(d["state"])
        return cls(la, d["structure"], d["tau"])
