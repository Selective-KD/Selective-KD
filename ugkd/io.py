import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_bundle(path, bundle):
    torch.save(bundle, path)


def load_bundle(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
