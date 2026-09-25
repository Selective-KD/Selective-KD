import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from ugkd.data import cifar as data
from ugkd.data.split import holdout
from ugkd.io import load_bundle, pick_device, save_bundle, set_seed
from ugkd.models import cifar as models
from ugkd.teacher import Teacher
from ugkd.train import cli, student
from ugkd.train.student import PairedImages

NUM_CLASSES = 10


def load_images(a):
    if (a.data is None) == (a.data_root is None):
        raise SystemExit("give exactly one of --data-root or --data")
    return data.load_npz(a.data)[:2] if a.data else data.load_cifar10(a.data_root, train=True)


def main():
    p = cli.parser("Train a CIFAR-10 student.")
    p.add_argument("--teacher", required=True, help="teacher bundle (.pt)")
    p.add_argument("--data-root", default=None, help="torchvision CIFAR-10 folder (downloaded if absent)")
    p.add_argument("--data", default=None, help="instead of --data-root: an .npz with x uint8 (N, 32, 32, 3) and y")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--student-frac", type=float, default=1.0,
                   help="share of the images the teacher did not see that the student trains on")
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    cli.add_method_args(p, mc_samples=30, temperature=1.0)
    cli.add_training_args(p, optimizer="sgd", lr=0.05, momentum=0.9, weight_decay=5e-4, epochs=600,
                          batch_size=64, schedule="cosine", label_smoothing=0.0, amp=False)
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    teacher = Teacher(load_bundle(a.teacher), device)
    x, y = load_images(a)
    split = teacher.bundle["split"]
    student_idx = (np.arange(len(y)) if split["teacher_frac"] >= 1.0
                   else np.setdiff1d(np.arange(len(y)), split["teacher_idx"]))
    if a.student_frac < 1.0:
        _, student_idx = holdout(student_idx, a.student_frac, split["seed"])
    batches = PairedImages(x[student_idx], y[student_idx], device)
    batches, val_batches = cli.split_validation(batches, a.val_frac, a.seed)

    model = models.build(a.model, teacher.num_classes, dropout=a.dropout)
    method = cli.build_method(a)
    student.train(model, batches, teacher, method, n_draws=a.mc_samples, online_teacher=True,
                  val_batches=val_batches, **cli.training_kwargs(a, device))

    save_bundle(a.out, {
        "modality": "cifar",
        "model": a.model,
        "num_classes": teacher.num_classes,
        "input": {"dropout": a.dropout},
        "state_dict": model.cpu().state_dict(),
        "standardizer": None,
        "split": {"seed": split["seed"], "student_idx": student_idx},
        "method": {"name": a.method, **method.summary()},
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
