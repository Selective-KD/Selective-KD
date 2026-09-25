import numpy as np
import torch

from ..data.cifar import Augment


class ArrayBatches:
    def __init__(self, x, y, device):
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.device, self.n = device, len(self.y)

    def get(self, idx, train=True):
        return self.x[idx].to(self.device, non_blocking=True), self.y[idx].to(self.device)


class ImageBatches:
    def __init__(self, x_uint8, y, device):
        self.x = torch.as_tensor(np.asarray(x_uint8, dtype=np.uint8))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.device, self.n = device, len(self.y)
        self.aug_train, self.aug_eval = Augment(device, train=True), Augment(device, train=False)
        self.generator = None

    def get(self, idx, train=True):
        x = self.x[idx].to(self.device, non_blocking=True)
        x = self.aug_train(x, self.generator) if train else self.aug_eval(x)
        return x, self.y[idx].to(self.device)


def iterate(batches, batch_size, train=True, rng=None):
    order = (rng or np.random).permutation(batches.n) if train else np.arange(batches.n)
    for i in range(0, batches.n, batch_size):
        yield order[i:i + batch_size]
