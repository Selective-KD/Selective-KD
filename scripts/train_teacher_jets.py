import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ugkd.data import jets as data
from ugkd.io import pick_device, save_bundle, set_seed
from ugkd.models import jets as models
from ugkd import posterior
from ugkd.train import cli, teacher
from ugkd.train.batches import ArrayBatches


def main():
    p = cli.parser("Train a tabular teacher.")
    p.add_argument("--train", required=True, help=".npz with arrays x (N, d) and y (N,)")
    p.add_argument("--model", choices=models.NAMES, required=True)
    p.add_argument("--num-classes", type=int, default=None, help="default: max(y) + 1")
    p.add_argument("--out", required=True, help="output bundle (.pt)")
    cli.add_training_args(p, optimizer="sgd", lr=0.01, momentum=0.0, weight_decay=0.0, epochs=2560,
                          batch_size=1024, schedule="constant", label_smoothing=0.0, amp=False)
    cli.add_laplace_args(p, laplace="exact", tau=0.1)
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)
    x, y, _ = data.load_npz(a.train)
    x, standardizer = data.standardize(x)
    num_classes = a.num_classes or int(y.max()) + 1
    batches = ArrayBatches(x, y, device)

    model = models.build(a.model, x.shape[1], num_classes)
    teacher.train(model, batches, **cli.training_kwargs(a, device))
    post = posterior.fit(model, batches, a.laplace, a.tau, device=device)
    print("laplace %s  tau %g" % (a.laplace, post.tau))

    save_bundle(a.out, {
        "modality": "jets",
        "model": a.model,
        "num_classes": num_classes,
        "input": {"in_dim": x.shape[1]},
        "state_dict": model.cpu().state_dict(),
        "standardizer": standardizer.state_dict(),
        "posterior": post.state_dict(),
        "args": vars(a),
    })
    print("saved", a.out)


if __name__ == "__main__":
    main()
