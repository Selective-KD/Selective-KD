import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from ugkd.train import cli
from ugkd.datasets import jets
from ugkd.datasets.pools import draw_pools


def main():
    p = cli.parser("Build the JetClass dataset.")
    p.add_argument("--jetclass", default=None, help="a JetClass tar or a folder of .root files")
    p.add_argument("--features", default=None, help="instead of --jetclass: an .npz with x, y and optionally axis")
    p.add_argument("--max-per-class", type=int, default=None, help="cap on jets read per class")
    p.add_argument("--axis-column", type=int, default=0,
                   help="column of x that defines the regions when the file has no `axis` array (0 = jet pt)")
    p.add_argument("--region-a", type=float, nargs=2, default=(500.0, 600.0), metavar=("LO", "HI"))
    p.add_argument("--region-b", type=float, nargs=2, default=(850.0, 1000.0), metavar=("LO", "HI"))
    p.add_argument("--f", type=float, default=0.3, help="share of region B in the student and test pools")
    p.add_argument("--n-teacher", type=int, default=100_000)
    p.add_argument("--n-student", type=int, default=1_500)
    p.add_argument("--n-test", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    if (a.jetclass is None) == (a.features is None):
        raise SystemExit("give exactly one of --jetclass or --features")
    if a.jetclass:
        x, y = jets.extract(a.jetclass, a.max_per_class)
        axis = x[:, a.axis_column]
        np.savez_compressed(os.path.join(a.out_dir, "features.npz"), x=x, y=y, axis=axis)
        print("features.npz  %d jets, %d features" % (len(y), x.shape[1]))
    else:
        f = np.load(a.features)
        x, y = np.asarray(f["x"], dtype=np.float32), np.asarray(f["y"], dtype=np.int64)
        axis = np.asarray(f["axis"], dtype=np.float64) if "axis" in f else x[:, a.axis_column]

    in_a = (axis >= a.region_a[0]) & (axis < a.region_a[1])
    in_b = (axis >= a.region_b[0]) & (axis <= a.region_b[1])
    pools = draw_pools(y, in_a, in_b, n_teacher=a.n_teacher, n_student=a.n_student, n_test=a.n_test,
                       f=a.f, seed=a.seed)
    for name, (idx, group) in pools.items():
        np.savez_compressed(os.path.join(a.out_dir, name + ".npz"), x=x[idx], y=y[idx], group=group)
        print("%-14s n %6d  region B %.2f" % (name, len(idx), group.mean()))


if __name__ == "__main__":
    main()
