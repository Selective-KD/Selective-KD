import torch.nn as nn
import torch.nn.functional as F

NAMES = {
    "mlp_256": (256, 256),
    "mlp_64": (64, 64),
}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.body = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
        self.head = nn.Linear(dims[-1], num_classes)
        self.feature_dim, self.num_classes = dims[-1], num_classes

    def features(self, x):
        for layer in self.body:
            x = F.relu(layer(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


def build(name, in_dim, num_classes):
    if name not in NAMES:
        raise ValueError("unknown model %r; choose from %s" % (name, sorted(NAMES)))
    return MLP(in_dim, NAMES[name], num_classes)
