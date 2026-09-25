import math

import numpy as np
import torch

from ..train.batches import iterate

TAU_GRID = np.logspace(-4, 4, 81)


@torch.no_grad()
def collect(model, batches, batch_size, device):
    model.eval()
    feats, probs, ys = [], [], []
    for idx in iterate(batches, batch_size, train=False):
        x, y = batches.get(idx, train=False)
        h = model.features(x)
        feats.append(h.double().cpu())
        probs.append(torch.softmax(model.head(h).double(), 1).cpu())
        ys.append(y.cpu())
    H = torch.cat(feats)
    Hb = torch.cat([H, torch.ones(len(H), 1, dtype=torch.float64)], 1)
    return Hb, torch.cat(probs), torch.cat(ys)


def _log_marginal(tau, loglik, w_sq, n_params, eig):
    return loglik - 0.5 * tau * w_sq + 0.5 * n_params * math.log(tau) - 0.5 * torch.log(eig + tau).sum().item()


def select_tau_marglik(loglik, w_sq, n_params, eig, grid=TAU_GRID):
    scores = [_log_marginal(float(t), loglik, w_sq, n_params, eig) for t in grid]
    return float(grid[int(np.argmax(scores))])


class LastLayerLaplace:
    def __init__(self, kind, W, tau, n_train, factors):
        self.kind, self.W, self.tau, self.n_train, self.factors = kind, W, float(tau), int(n_train), factors
        self.num_classes, self.dim = W.shape

    @classmethod
    def fit(cls, model, batches, kind, tau, batch_size=512, device="cpu"):
        Hb, P, y = collect(model, batches, batch_size, device)
        W = torch.cat([model.head.weight.detach().double().cpu(),
                       model.head.bias.detach().double().cpu()[:, None]], 1)
        N, K = P.shape
        loglik = torch.log(P[torch.arange(N), y].clamp_min(1e-12)).sum().item()
        w_sq = (W * W).sum().item()
        if kind == "exact":
            G = _exact_ggn(Hb, P)
            if tau == "marglik":
                eig = torch.linalg.eigvalsh(G).clamp_min(0.0)
                tau = select_tau_marglik(loglik, w_sq, G.shape[0], eig)
            prec = G + float(tau) * torch.eye(G.shape[0], dtype=torch.float64)
            L = torch.linalg.cholesky(0.5 * (prec + prec.T))
            return cls(kind, W, tau, N, {"L": L})
        if kind == "kfac":
            A = (Hb.T @ Hb) / N
            Lam = torch.diag_embed(P) - P[:, :, None] * P[:, None, :]
            B = Lam.mean(0)
            if tau == "marglik":
                a, b = torch.linalg.eigvalsh(A).clamp_min(0.0), torch.linalg.eigvalsh(B).clamp_min(0.0)
                eig = (N * b[:, None] * a[None, :]).reshape(-1)
                tau = select_tau_marglik(loglik, w_sq, K * Hb.shape[1], eig)
            s = math.sqrt(float(tau) / N)
            A = A + s * torch.eye(A.shape[0], dtype=torch.float64)
            B = B + s * torch.eye(B.shape[0], dtype=torch.float64)
            return cls(kind, W, tau, N, {"Ai_h": _inv_sqrt(A), "Bi_h": _inv_sqrt(B)})
        raise ValueError("unknown laplace kind %r; choose exact or kfac" % kind)

    def sample_weights(self, n_draws, seed):
        gen = torch.Generator().manual_seed(int(seed))
        if self.kind == "exact":
            z = torch.randn(n_draws, self.num_classes * self.dim, generator=gen, dtype=torch.float64)
            delta = torch.linalg.solve_triangular(self.factors["L"].T, z.T, upper=True)
            return (self.W[None] + delta.T.reshape(n_draws, self.num_classes, self.dim)).float()
        E = torch.randn(n_draws, self.num_classes, self.dim, generator=gen, dtype=torch.float64)
        delta = (self.factors["Bi_h"] @ E @ self.factors["Ai_h"]) / math.sqrt(self.n_train)
        return (self.W[None] + delta).float()

    @torch.no_grad()
    def draws(self, features, n_draws, seed, temperature=1.0):
        Ws = self.sample_weights(n_draws, seed).to(features.device)
        Hb = torch.cat([features.float(), torch.ones(len(features), 1, device=features.device)], 1)
        Z = (Hb @ Ws.transpose(1, 2)) / float(temperature)
        return torch.softmax(Z, -1), Z

    def state_dict(self):
        return {"kind": self.kind, "W": self.W, "tau": self.tau, "n_train": self.n_train,
                "factors": self.factors}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["kind"], d["W"], d["tau"], d["n_train"], d["factors"])


def _exact_ggn(Hb, P):
    N, hf = Hb.shape
    K = P.shape[1]
    G = torch.zeros(K * hf, K * hf, dtype=torch.float64)
    for a in range(K):
        for b in range(K):
            w = P[:, a] * ((1.0 if a == b else 0.0) - P[:, b])
            G[a * hf:(a + 1) * hf, b * hf:(b + 1) * hf] = Hb.T @ (w[:, None] * Hb)
    return 0.5 * (G + G.T)


def _inv_sqrt(M, eps=1e-10):
    w, V = torch.linalg.eigh(M)
    return V @ torch.diag(w.clamp_min(eps).rsqrt()) @ V.T
