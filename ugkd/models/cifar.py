import torch
import torch.nn as nn
import torch.nn.functional as F

CONVNET_CFGS = {
    "convnet6": ([32, 64, 128], 2),
    "convnet3": ([32, 64, 128], 1),
}
RESNET_DEPTHS = {"resnet20": 3, "resnet32": 5}
WRN_CFGS = {"wrn_16_1": (16, 1), "wrn_16_2": (16, 2),
            "wrn_40_1": (40, 1), "wrn_40_2": (40, 2)}
NO_DROPOUT = set(RESNET_DEPTHS) | {"shufflenet_v1"}
NAMES = sorted(CONVNET_CFGS) + sorted(RESNET_DEPTHS) + sorted(WRN_CFGS) + ["shufflenet_v1"]


def _init_conv_bn(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _conv_stage(in_ch, out_ch, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ConvNet(nn.Module):
    def __init__(self, channels, n_convs, num_classes, dropout):
        super().__init__()
        stages, in_ch = [], 3
        for i, out_ch in enumerate(channels):
            stages += [_conv_stage(in_ch, out_ch, n_convs),
                       nn.Identity() if i == 0 else nn.MaxPool2d(2),
                       nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            in_ch = out_ch
        self.body = nn.Sequential(*stages, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(channels[-1], num_classes)
        self.feature_dim, self.num_classes = channels[-1], num_classes
        _init_conv_bn(self)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


class ResBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    """He et al. 2016, sec. 4.2."""
    def __init__(self, n, num_classes):
        super().__init__()
        widths = [16, 32, 64]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        layers, in_planes = [], widths[0]
        for stage, planes in enumerate(widths):
            for block in range(n):
                layers.append(ResBasicBlock(in_planes, planes, 2 if stage > 0 and block == 0 else 1))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], num_classes)
        self.feature_dim, self.num_classes = widths[-1], num_classes
        _init_conv_bn(self)

    def features(self, x):
        out = self.layers(F.relu(self.bn1(self.conv1(x))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.shortcut = None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        residual = x if self.equal_in_out else out
        out = F.relu(self.bn2(self.conv1(out)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return residual + out


def _wrn_group(n_blocks, in_planes, out_planes, stride, dropout):
    blocks = [WRNBasicBlock(in_planes if i == 0 else out_planes, out_planes,
                            stride if i == 0 else 1, dropout) for i in range(n_blocks)]
    return nn.Sequential(*blocks)


class WideResNet(nn.Module):
    """Zagoruyko & Komodakis 2016."""
    def __init__(self, depth, widen, num_classes, dropout):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n + 4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.group1 = _wrn_group(n, widths[0], widths[1], 1, dropout)
        self.group2 = _wrn_group(n, widths[1], widths[2], 2, dropout)
        self.group3 = _wrn_group(n, widths[2], widths[3], 2, dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.head = nn.Linear(widths[3], num_classes)
        self.feature_dim, self.num_classes = widths[3], num_classes
        _init_conv_bn(self)
        nn.init.zeros_(self.head.bias)

    def features(self, x):
        out = self.group3(self.group2(self.group1(self.conv1(x))))
        return F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


class _ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        return x.view(n, self.groups, c // self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)


class _ShuffleBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super().__init__()
        self.stride = stride
        mid = out_planes // 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid, 1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.shuffle = _ChannelShuffle(g)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_planes, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.AvgPool2d(3, stride=2, padding=1) if stride == 2 else nn.Identity()

    def forward(self, x):
        out = self.shuffle(F.relu(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        return F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)


class ShuffleNetV1(nn.Module):
    """Zhang et al. 2018, the CIFAR variant."""
    def __init__(self, num_classes, out_planes=(240, 480, 960), num_blocks=(4, 8, 4), groups=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        layers, in_planes = [], 24
        for planes, n in zip(out_planes, num_blocks):
            for i in range(n):
                cat_planes = in_planes if i == 0 else 0
                layers.append(_ShuffleBottleneck(in_planes, planes - cat_planes, 2 if i == 0 else 1, groups))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(out_planes[-1], num_classes)
        self.feature_dim, self.num_classes = out_planes[-1], num_classes

    def features(self, x):
        out = self.layers(F.relu(self.bn1(self.conv1(x))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


def build(name, num_classes, dropout=0.0):
    if name in NO_DROPOUT and dropout > 0:
        print("note: %s has no dropout; --dropout %g ignored" % (name, dropout))
    if name in CONVNET_CFGS:
        channels, n_convs = CONVNET_CFGS[name]
        return ConvNet(channels, n_convs, num_classes, dropout)
    if name in RESNET_DEPTHS:
        return CifarResNet(RESNET_DEPTHS[name], num_classes)
    if name in WRN_CFGS:
        depth, widen = WRN_CFGS[name]
        return WideResNet(depth, widen, num_classes, dropout)
    if name == "shufflenet_v1":
        return ShuffleNetV1(num_classes)
    raise ValueError("unknown model %r; choose from %s" % (name, NAMES))
