import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ugkd.data import audio as data
from ugkd.io import load_bundle, pick_device, save_bundle, set_seed
from ugkd.models import audio as models
from ugkd.teacher import Teacher
from ugkd.train import cli, student
from ugkd.train.student import PairedArrays


def main():
    p = cli.parser("Train an audio student.")
    p.add_argument("--teacher", required=True, help="teacher bundle (.pt); its front-end settings are reused")
    p.add_argument("--train-csv", default=None, help="CSV with columns path,label")
    p.add_argument("--root", default=None, help="folder the CSV paths are relative to")
    p.add_argument("--data", default=None, help="instead of --train-csv: an .npz of rendered log-mel spectrograms")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    cli.add_method_args(p, mc_samples=30, temperature=1.0)
    cli.add_training_args(p, optimizer="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4, epochs=100,
                          batch_size=128, schedule="cosine", label_smoothing=0.0, amp=True)
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    teacher = Teacher(load_bundle(a.teacher), device)
    inp = teacher.bundle["input"]
    if (a.data is None) == (a.train_csv is None):
        raise SystemExit("give exactly one of --train-csv or --data")
    if a.data:
        x, y, _, _ = data.load_npz(a.data)
        if x.shape[1:] != (inp["n_mels"], inp["n_frames"]):
            raise SystemExit("spectrograms are %s but the teacher expects (%d, %d)" % (x.shape[1:], inp["n_mels"], inp["n_frames"]))
    else:
        front = data.LogMel(device, sr=inp["sr"], clip_seconds=inp["clip_seconds"], n_mels=inp["n_mels"],
                            n_fft=inp["n_fft"], hop=inp["hop"])
        paths, y, _ = data.load_csv(a.train_csv, a.root)
        x = front.render(paths)
    xs, standardizer = data.standardize(x)
    xt = teacher.standardizer.transform(x)
    batches = PairedArrays(xs, xt, y, device)
    batches, val_batches = cli.split_validation(batches, a.val_frac, a.seed)

    model = models.build(a.model, teacher.num_classes, inp["n_mels"], inp["n_frames"], dropout=a.dropout)
    method = cli.build_method(a)
    student.train(model, batches, teacher, method, n_draws=a.mc_samples, online_teacher=False,
                  val_batches=val_batches, **cli.training_kwargs(a, device))

    save_bundle(a.out, {
        "modality": "audio",
        "model": a.model,
        "num_classes": teacher.num_classes,
        "input": {k: inp[k] for k in ("sr", "clip_seconds", "n_mels", "n_fft", "hop", "n_frames")} | {"dropout": a.dropout},
        "state_dict": model.cpu().state_dict(),
        "standardizer": standardizer.state_dict(),
        "method": {"name": a.method, **method.summary()},
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
