import argparse

import numpy as np


def tau_type(s):
    return "marglik" if s == "marglik" else float(s)


def add_training_args(p, *, optimizer, lr, momentum, weight_decay, epochs, batch_size,
                      schedule, label_smoothing, amp):
    g = p.add_argument_group("training")
    g.add_argument("--optimizer", choices=["sgd", "adam"], default=optimizer)
    g.add_argument("--lr", type=float, default=lr)
    g.add_argument("--momentum", type=float, default=momentum, help="sgd only")
    g.add_argument("--weight-decay", type=float, default=weight_decay)
    g.add_argument("--epochs", type=int, default=epochs)
    g.add_argument("--batch-size", type=int, default=batch_size)
    g.add_argument("--schedule", choices=["constant", "cosine"], default=schedule)
    g.add_argument("--label-smoothing", type=float, default=label_smoothing)
    g.add_argument("--amp", type=int, choices=[0, 1], default=int(amp), help="fp16 autocast on GPU")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")


def add_laplace_args(p, *, laplace, tau):
    from .. import posterior
    g = p.add_argument_group("posterior")
    g.add_argument("--laplace", choices=posterior.KINDS, default=laplace)
    g.add_argument("--tau", type=tau_type, default=tau, help="prior precision, a number or 'marglik'")


def add_method_args(p, *, mc_samples, temperature):
    from .. import methods
    g = p.add_argument_group("distillation")
    g.add_argument("--method", choices=methods.NAMES, required=True)
    g.add_argument("--method-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="override one of the method's own parameters; repeatable")
    g.add_argument("--mc-samples", type=int, default=mc_samples, help="posterior draws per teacher read")
    g.add_argument("--temperature", type=float, default=temperature)
    g.add_argument("--val-frac", type=float, default=0.0,
                   help="share of the training data held out for methods that need it (tgeo, rwkd)")
    p.epilog = "method parameters and their defaults:\n" + methods.defaults_text()
    p.formatter_class = argparse.RawDescriptionHelpFormatter


def method_overrides(a):
    import ast
    out = {}
    for item in a.method_arg:
        key, _, value = item.partition("=")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def split_validation(batches, val_frac, seed):
    from ..data.split import holdout
    if val_frac <= 0.0:
        return batches, None
    keep, held = holdout(np.arange(batches.n), val_frac, seed)
    return batches.subset(keep), batches.subset(held)


def build_method(a):
    from .. import methods
    overrides = method_overrides(a)
    params = {**methods.REGISTRY[a.method].DEFAULTS, **overrides}
    print("method %s  T %g  %s" % (a.method, a.temperature, params))
    return methods.build(a.method, a.temperature, overrides)


def training_kwargs(a, device):
    return dict(epochs=a.epochs, batch_size=a.batch_size, optimizer=a.optimizer, lr=a.lr,
                momentum=a.momentum, weight_decay=a.weight_decay, schedule=a.schedule,
                label_smoothing=a.label_smoothing, amp=bool(a.amp), seed=a.seed, device=device)


def parser(description):
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
