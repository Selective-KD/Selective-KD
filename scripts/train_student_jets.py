import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ugkd.data import jets as data
from ugkd.io import load_bundle, pick_device, save_bundle, set_seed
from ugkd.models import jets as models
from ugkd.teacher import Teacher
from ugkd.train import cli, student
from ugkd.train.student import PairedArrays


def main():
    p = cli.parser("Train a tabular student.")
    p.add_argument("--teacher", required=True, help="teacher bundle (.pt)")
    p.add_argument("--train", required=True, help=".npz with arrays x (N, d) and y (N,)")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    cli.add_method_args(p, mc_samples=30, temperature=1.0)
    cli.add_training_args(p, optimizer="sgd", lr=0.03, momentum=0.0, weight_decay=1e-3, epochs=400,
                          batch_size=128, schedule="constant", label_smoothing=0.0, amp=False)
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    teacher = Teacher(load_bundle(a.teacher), device)
    x, y, _ = data.load_npz(a.train)
    xs, standardizer = data.standardize(x)
    xt = teacher.standardizer.transform(x)
    batches = PairedArrays(xs, xt, y, device)
    batches, val_batches = cli.split_validation(batches, a.val_frac, a.seed)

    model = models.build(a.model, x.shape[1], teacher.num_classes)
    method = cli.build_method(a)
    student.train(model, batches, teacher, method, n_draws=a.mc_samples, online_teacher=False,
                  val_batches=val_batches, **cli.training_kwargs(a, device))

    save_bundle(a.out, {
        "modality": "jets",
        "model": a.model,
        "num_classes": teacher.num_classes,
        "input": {"in_dim": x.shape[1]},
        "state_dict": model.cpu().state_dict(),
        "standardizer": standardizer.state_dict(),
        "method": {"name": a.method, **method.summary()},
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
