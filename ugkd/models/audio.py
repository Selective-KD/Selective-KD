import torch.nn as nn

CNN_CFGS = {
    "cnn8": ((32, 64, 128, 128), 2, 512),
    "cnn8n": ((16, 32, 64, 64), 2, 512),
    "cnn4": ((16, 32, 64, 64), 1, 128),
    "cnn4n": ((8, 16, 32, 32), 1, 64),
}
RESNET_CFGS = {
    "resnet14": ((32, 64, 128), 2, 512),
    "resnet8": ((16, 32, 64), 1, 128),
}
NAMES = sorted(CNN_CFGS) + sorted(RESNET_CFGS)


def _mlp_head(in_dim, hidden, dropout):
    return nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_dim, hidden),
                         nn.ReLU(), nn.Dropout(dropout))


class CNN(nn.Module):
    def __init__(self, channels, n_convs, hidden, num_classes, n_mels, n_frames, dropout):
        super().__init__()
        blocks, cin = [], 1
        for c in channels:
            for j in range(n_convs):
                blocks += [nn.Conv2d(cin if j == 0 else c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
            blocks += [nn.MaxPool2d(2)]
            cin = c
        f, t = n_mels, n_frames
        for _ in channels:
            f, t = f // 2, t // 2
        self.body = nn.Sequential(*blocks, _mlp_head(cin * f * t, hidden, dropout))
        self.head = nn.Linear(hidden, num_classes)
        self.feature_dim, self.num_classes = hidden, num_classes

    def features(self, x):
        return self.body(x.unsqueeze(1))

    def forward(self, x):
        return self.head(self.features(x))


class _BasicBlock(nn.Module):
    def __init__(self, cin, c, stride):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, c, 3, stride, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.shortcut = (nn.Identity() if stride == 1 and cin == c else
                         nn.Sequential(nn.Conv2d(cin, c, 1, stride, bias=False), nn.BatchNorm2d(c)))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.f(x) + self.shortcut(x))


class FreqResNet(nn.Module):
    def __init__(self, channels, n_blocks, hidden, num_classes, n_mels, n_frames, dropout):
        super().__init__()
        layers = [nn.Conv2d(1, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU()]
        cin, f = channels[0], n_mels
        for i, c in enumerate(channels):
            for j in range(n_blocks):
                layers.append(_BasicBlock(cin, c, 2 if i > 0 and j == 0 else 1))
                cin = c
            if i > 0:
                f = (f + 1) // 2
        layers.append(nn.AdaptiveAvgPool2d((None, 1)))
        self.body = nn.Sequential(*layers, _mlp_head(cin * f, hidden, dropout))
        self.head = nn.Linear(hidden, num_classes)
        self.feature_dim, self.num_classes = hidden, num_classes

    def features(self, x):
        return self.body(x.unsqueeze(1))

    def forward(self, x):
        return self.head(self.features(x))


def build(name, num_classes, n_mels, n_frames, dropout=0.0):
    if name in CNN_CFGS:
        channels, n_convs, hidden = CNN_CFGS[name]
        return CNN(channels, n_convs, hidden, num_classes, n_mels, n_frames, dropout)
    if name in RESNET_CFGS:
        channels, n_blocks, hidden = RESNET_CFGS[name]
        return FreqResNet(channels, n_blocks, hidden, num_classes, n_mels, n_frames, dropout)
    raise ValueError("unknown model %r; choose from %s" % (name, NAMES))
