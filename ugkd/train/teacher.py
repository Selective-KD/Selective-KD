import numpy as np
import torch
import torch.nn as nn

from .batches import iterate


def make_optimizer(model, name, lr, momentum, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("unknown optimizer %r" % name)


def make_schedule(opt, name, epochs):
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    raise ValueError("unknown schedule %r" % name)


def train(model, batches, *, epochs, batch_size, optimizer, lr, momentum, weight_decay,
          schedule, label_smoothing, amp, seed, device, log=print):
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, momentum, weight_decay)
    sch = make_schedule(opt, schedule, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        total, count = 0.0, 0
        for idx in iterate(batches, batch_size, train=True, rng=rng):
            x, y = batches.get(idx, train=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
            count += len(idx)
        sch.step()
        log("epoch %d/%d  loss %.4f" % (ep + 1, epochs, total / max(count, 1)))
    model.eval()
    return model
