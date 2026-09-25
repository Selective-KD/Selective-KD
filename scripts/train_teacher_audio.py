import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ugkd.data import audio as data
from ugkd.io import pick_device, save_bundle, set_seed
from ugkd.models import audio as models
from ugkd import posterior
from ugkd.train import cli, teacher
from ugkd.train.batches import ArrayBatches


def add_frontend_args(p):
    g = p.add_argument_group("front end")
    g.add_argument("--sr", type=int, default=22050, help="sample rate clips are resampled to")
    g.add_argument("--clip-seconds", type=float, default=4.0, help="clips are padded or cropped to this length")
    g.add_argument("--n-mels", type=int, default=64)
    g.add_argument("--n-fft", type=int, default=1024)
    g.add_argument("--hop", type=int, default=512)


def frontend(a, device):
    return data.LogMel(device, sr=a.sr, clip_seconds=a.clip_seconds, n_mels=a.n_mels, n_fft=a.n_fft, hop=a.hop)


def main():
    p = cli.parser("Train an audio teacher.")
    p.add_argument("--train-csv", default=None, help="CSV with columns path,label")
    p.add_argument("--root", default=None, help="folder the CSV paths are relative to")
    p.add_argument("--data", default=None, help="instead of --train-csv: an .npz of rendered log-mel spectrograms")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num-classes", type=int, default=None, help="default: max(label) + 1")
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    add_frontend_args(p)
    cli.add_training_args(p, optimizer="adam", lr=1e-3, momentum=0.9, weight_decay=1e-4, epochs=20,
                          batch_size=128, schedule="cosine", label_smoothing=0.01, amp=True)
    cli.add_laplace_args(p, laplace="kfac", tau=1.0)
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    if (a.data is None) == (a.train_csv is None):
        raise SystemExit("give exactly one of --train-csv or --data")
    if a.data:
        x, y, _, fe = data.load_npz(a.data)
        if fe:
            a.sr, a.clip_seconds, a.n_mels, a.n_fft, a.hop = int(fe["sr"]), fe["clip_seconds"], int(fe["n_mels"]), int(fe["n_fft"]), int(fe["hop"])
        n_mels, n_frames = x.shape[1], x.shape[2]
    else:
        front = frontend(a, device)
        paths, y, _ = data.load_csv(a.train_csv, a.root)
        x = front.render(paths)
        n_mels, n_frames = front.n_mels, front.n_frames
    x, standardizer = data.standardize(x)
    num_classes = a.num_classes or int(y.max()) + 1
    batches = ArrayBatches(x, y, device)

    model = models.build(a.model, num_classes, n_mels, n_frames, dropout=a.dropout)
    teacher.train(model, batches, **cli.training_kwargs(a, device))
    post = posterior.fit(model, batches, a.laplace, a.tau, device=device)
    print("laplace %s  tau %g" % (a.laplace, post.tau))

    save_bundle(a.out, {
        "modality": "audio",
        "model": a.model,
        "num_classes": num_classes,
        "input": {"sr": a.sr, "clip_seconds": a.clip_seconds, "n_mels": a.n_mels, "n_fft": a.n_fft,
                  "hop": a.hop, "n_frames": n_frames, "dropout": a.dropout},
        "state_dict": model.cpu().state_dict(),
        "standardizer": standardizer.state_dict(),
        "posterior": post.state_dict(),
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
