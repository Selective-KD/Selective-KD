import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from ugkd.data.audio import LogMel
from ugkd.io import pick_device
from ugkd.train import cli
from ugkd.datasets import audio


def main():
    p = cli.parser("Build the UrbanSound8K dataset.")
    p.add_argument("--us8k", required=True, help="UrbanSound8K folder (metadata/ and audio/)")
    p.add_argument("--cache", default=None, help="decoded-clip cache (default: <out-dir>/clips.npy)")
    p.add_argument("--teacher-frac", type=float, default=0.6, help="share of each class's recordings for the teacher")
    p.add_argument("--student-frac", type=float, default=0.2, help="share for the student; the rest is the test set")
    p.add_argument("--beta", type=float, default=0.8, help="weight of the dominant source in a mixture")
    p.add_argument("--mask-bins", type=int, default=16, help="mel bins masked in region B")
    p.add_argument("--f", type=float, default=0.3, help="share of region B in the student and test sets")
    p.add_argument("--n-teacher", type=int, default=20_000, help="rendered mixtures for the teacher")
    p.add_argument("--n-student", type=int, default=1_500)
    p.add_argument("--n-test", type=int, default=4_000)
    g = p.add_argument_group("front end")
    g.add_argument("--sr", type=int, default=22050)
    g.add_argument("--clip-seconds", type=float, default=4.0)
    g.add_argument("--n-mels", type=int, default=64)
    g.add_argument("--n-fft", type=int, default=1024)
    g.add_argument("--hop", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rows = audio.read_metadata(a.us8k)
    bank, rms = audio.decode(rows, a.cache or os.path.join(a.out_dir, "clips.npy"), a.sr, a.clip_seconds)
    pools = audio.split_sources(rows, rms, a.teacher_frac, a.student_frac, a.seed)
    for name, pool in pools.items():
        print("%-8s %d clips" % (name, sum(len(v) for v in pool.values())))
    front = LogMel(pick_device(a.device), sr=a.sr, clip_seconds=a.clip_seconds, n_mels=a.n_mels,
                   n_fft=a.n_fft, hop=a.hop)
    files = audio.build(pools, bank, front, beta=a.beta, mask_bins=a.mask_bins, f=a.f, n_teacher=a.n_teacher,
                        n_student=a.n_student, n_test=a.n_test, seed=a.seed)
    frontend = dict(sr=a.sr, clip_seconds=a.clip_seconds, n_mels=a.n_mels, n_fft=a.n_fft, hop=a.hop)
    for name, (x, y, group) in files.items():
        np.savez_compressed(os.path.join(a.out_dir, name + ".npz"), x=x, y=y, group=group, **frontend)
        print("%-14s n %6d  region B %.2f" % (name, len(y), group.mean()))


if __name__ == "__main__":
    main()
