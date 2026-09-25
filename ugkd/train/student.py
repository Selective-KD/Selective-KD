import numpy as np
import torch

from .batches import iterate
from .teacher import make_optimizer, make_schedule


class PairedArrays:
    def __init__(self, xs, xt, y, device):
        self.xs = torch.as_tensor(np.asarray(xs, dtype=np.float32))
        self.xt = torch.as_tensor(np.asarray(xt, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.device, self.n = device, len(self.y)

    def get(self, idx, train=True):
        return (self.xs[idx].to(self.device, non_blocking=True),
                self.xt[idx].to(self.device, non_blocking=True), self.y[idx].to(self.device))

    def subset(self, idx):
        return PairedArrays(self.xs[idx], self.xt[idx], self.y[idx], self.device)


class PairedImages:
    def __init__(self, x_uint8, y, device):
        from ..data.cifar import Augment
        self.x = torch.as_tensor(np.asarray(x_uint8, dtype=np.uint8))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.device, self.n = device, len(self.y)
        self.aug_train, self.aug_eval = Augment(device, train=True), Augment(device, train=False)

    def get(self, idx, train=True):
        x = self.x[idx].to(self.device, non_blocking=True)
        x = self.aug_train(x) if train else self.aug_eval(x)
        return x, x, self.y[idx].to(self.device)

    def subset(self, idx):
        return PairedImages(self.x[idx], self.y[idx], self.device)


@torch.no_grad()
def read_pool(teacher, batches, n_draws, seed, batch_size=512):
    Ps, Zs = [], []
    for idx in iterate(batches, batch_size, train=False):
        _, xt, _ = batches.get(idx, train=False)
        P, Z = teacher.draws(xt, n_draws, seed)
        Ps.append(P.cpu())
        Zs.append(Z.cpu())
    return torch.cat(Ps, 1), torch.cat(Zs, 1)


def train(model, batches, teacher, method, *, n_draws, online_teacher, epochs, batch_size,
          optimizer, lr, momentum, weight_decay, schedule, label_smoothing, amp, seed, device,
          val_batches=None, log=print):
    if method.needs_val and val_batches is None:
        raise ValueError("method %s needs validation data: set --val-frac > 0" % method.name)
    P_pool, Z_pool = read_pool(teacher, batches, n_draws, seed)
    method.prepare(P_pool, Z_pool, batches.y, seed)
    if "frac_to_teacher" in method.summary():
        log("routed %.3f of the pool to the teacher" % method.summary()["frac_to_teacher"])
    if method.needs_val:
        P_val, Z_val = read_pool(teacher, val_batches, n_draws, seed + 1)
        val_rng = np.random.default_rng(seed + 1)

    model.to(device)
    steps_per_epoch = -(-batches.n // batch_size)
    method.set_total_steps(epochs * steps_per_epoch)
    opt = make_optimizer(model, optimizer, lr, momentum, weight_decay)
    for prm in method.extra_params():
        prm.data = prm.data.to(device)
        opt.add_param_group({"params": [prm], "weight_decay": 0.0})
    sch = make_schedule(opt, schedule, epochs)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        model.train()
        total, count = 0.0, 0
        for idx in iterate(batches, batch_size, train=True, rng=rng):
            xs, xt, y = batches.get(idx, train=True)
            if online_teacher:
                P, Z = teacher.draws(xt, n_draws, seed * 1_000_003 + step)
            else:
                P, Z = P_pool[:, idx].to(device), Z_pool[:, idx].to(device)
            if method.needs_val:
                vidx = val_rng.choice(val_batches.n, min(batch_size, val_batches.n), replace=False)
                xv, _, yv = val_batches.get(vidx, train=False)
                method.outer_step(model, (xs, y, P, Z),
                                  (xv, yv, P_val[:, vidx].to(device), Z_val[:, vidx].to(device)),
                                  opt.param_groups[0]["lr"])
            with torch.autocast("cuda", enabled=use_amp):
                features = model.features(xs)
                logits = model.head(features)
            loss = method.loss(logits.float(), features.float(), idx, P, Z, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
            count += len(idx)
            step += 1
        sch.step()
        log("epoch %d/%d  loss %.4f" % (ep + 1, epochs, total / max(count, 1)))
    model.eval()
    return model
