import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from ugkd.data import cifar as data
from ugkd.train import cli
from ugkd.datasets import cifar
from ugkd.datasets.corruptions import NAMES


def main():
    p = cli.parser("Build the CIFAR-10 dataset.")
    p.add_argument("--data-root", required=True, help="torchvision CIFAR-10 folder (downloaded if absent)")
    p.add_argument("--region-a", choices=NAMES, default="gaussian_blur", help="the teacher's corruption")
    p.add_argument("--region-b", choices=NAMES, default="glass_blur", help="the corruption the teacher never sees")
    p.add_argument("--f", type=float, default=0.3, help="share of region B in the student and test sets")
    p.add_argument("--n-teacher", type=int, default=40_000, help="training images for the teacher, class-balanced")
    p.add_argument("--n-student", type=int, default=10_000, help="training images for the student, class-balanced, disjoint")
    p.add_argument("--frost-dir", default="frost", help="where the frost photographs are fetched to")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    x_train, y_train = data.load_cifar10(a.data_root, train=True)
    x_test, y_test = data.load_cifar10(a.data_root, train=False)
    files = cifar.build(x_train, y_train, x_test, y_test, region_a=a.region_a, region_b=a.region_b, f=a.f,
                        n_teacher=a.n_teacher, n_student=a.n_student, seed=a.seed, frost_dir=a.frost_dir)
    for name, (x, y, group) in files.items():
        np.savez_compressed(os.path.join(a.out_dir, name + ".npz"), x=x, y=y, group=group)
        print("%-14s n %6d  region B %.2f" % (name, len(y), group.mean()))


if __name__ == "__main__":
    main()
