import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from ugkd.data import cifar as data
from ugkd.data.split import permute_split
from ugkd.io import pick_device, save_bundle, set_seed
from ugkd.models import cifar as models
from ugkd import posterior
from ugkd.train import cli, teacher
from ugkd.train.batches import ImageBatches

NUM_CLASSES = 10


def load_images(a):
    if (a.data is None) == (a.data_root is None):
        raise SystemExit("give exactly one of --data-root or --data")
    return data.load_npz(a.data)[:2] if a.data else data.load_cifar10(a.data_root, train=True)


def main():
    p = cli.parser("Train a CIFAR-10 teacher.")
    p.add_argument("--data-root", default=None, help="torchvision CIFAR-10 folder (downloaded if absent)")
    p.add_argument("--data", default=None, help="instead of --data-root: an .npz with x uint8 (N, 32, 32, 3) and y")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--teacher-frac", type=float, default=0.8,
                   help="share of the training images the teacher sees; 1.0 = all of them (a dataset file)")
    p.add_argument("--split-seed", type=int, default=0, help="seed of the teacher/student split")
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    cli.add_training_args(p, optimizer="sgd", lr=0.1, momentum=0.9, weight_decay=5e-4, epochs=200,
                          batch_size=100, schedule="cosine", label_smoothing=0.0, amp=False)
    cli.add_laplace_args(p, laplace="kfac", tau="marglik")
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    x, y = load_images(a)
    teacher_idx = np.arange(len(y)) if a.teacher_frac >= 1.0 else permute_split(len(y), a.teacher_frac, a.split_seed)[0]
    batches = ImageBatches(x[teacher_idx], y[teacher_idx], device)

    model = models.build(a.model, NUM_CLASSES, dropout=a.dropout)
    teacher.train(model, batches, **cli.training_kwargs(a, device))
    post = posterior.fit(model, batches, a.laplace, a.tau, device=device)
    print("laplace %s  tau %g" % (a.laplace, post.tau))

    save_bundle(a.out, {
        "modality": "cifar",
        "model": a.model,
        "num_classes": NUM_CLASSES,
        "input": {"dropout": a.dropout},
        "state_dict": model.cpu().state_dict(),
        "standardizer": None,
        "posterior": post.state_dict(),
        "split": {"seed": a.split_seed, "teacher_frac": a.teacher_frac, "teacher_idx": teacher_idx},
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
