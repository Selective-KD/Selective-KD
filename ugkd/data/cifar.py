import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root, train=True, download=True):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root, train=train, download=download)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def load_npz(path):
    f = np.load(path)
    x, y = np.asarray(f["x"], dtype=np.uint8), np.asarray(f["y"], dtype=np.int64)
    if x.shape[1:] != (32, 32, 3):
        raise ValueError("expected x (N, 32, 32, 3) uint8, got %s" % (x.shape,))
    group = np.asarray(f["group"], dtype=np.int64) if "group" in f else None
    return x, y, group


class Augment:
    def __init__(self, device, train=True, pad=4, flip_p=0.5):
        self.train, self.pad, self.flip_p = train, pad, flip_p
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def __call__(self, x_uint8, generator=None):
        x = x_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        if self.train:
            x = self._crop(x, generator)
            x = self._flip(x, generator)
        return (x - self.mean) / self.std

    def _crop(self, x, generator):
        b, p = x.shape[0], self.pad
        x = F.pad(x, (p, p, p, p))
        ox = torch.randint(0, 2 * p + 1, (b,), device=x.device, generator=generator)
        oy = torch.randint(0, 2 * p + 1, (b,), device=x.device, generator=generator)
        rows = torch.arange(32, device=x.device).view(1, 32) + oy.view(b, 1)
        cols = torch.arange(32, device=x.device).view(1, 32) + ox.view(b, 1)
        bi = torch.arange(b, device=x.device).view(b, 1, 1)
        return x[bi, :, rows.view(b, 32, 1), cols.view(b, 1, 32)].permute(0, 3, 1, 2)

    def _flip(self, x, generator):
        m = torch.rand(x.shape[0], device=x.device, generator=generator) < self.flip_p
        return torch.where(m.view(-1, 1, 1, 1), x.flip(3), x)
