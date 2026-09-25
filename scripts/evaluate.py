import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from ugkd.data import audio, cifar, jets
from ugkd.data.standardize import Standardizer
from ugkd.io import load_bundle, pick_device
from ugkd.teacher import build_model
from ugkd.train import cli
from ugkd.train.batches import ArrayBatches, ImageBatches, iterate


def load_inputs(a, bundle, device):
    m, inp = bundle["modality"], bundle["input"]
    st = Standardizer.from_state_dict(bundle["standardizer"]) if bundle.get("standardizer") else None
    if m == "jets":
        x, y, group = jets.load_npz(a.data)
        return ArrayBatches(st.transform(x), y, device), group
    if m == "audio":
        if a.data:
            x, y, group, _ = audio.load_npz(a.data)
        else:
            paths, y, group = audio.load_csv(a.csv, a.root)
            front = audio.LogMel(device, sr=inp["sr"], clip_seconds=inp["clip_seconds"], n_mels=inp["n_mels"],
                                 n_fft=inp["n_fft"], hop=inp["hop"])
            x = front.render(paths)
        return ArrayBatches(st.transform(x), y, device), group
    if m == "cifar":
        if a.data:
            x, y, group = cifar.load_npz(a.data)
        else:
            (x, y), group = cifar.load_cifar10(a.data_root, train=False), None
        return ImageBatches(x, y, device), group
    raise ValueError("unknown modality %r" % m)


@torch.no_grad()
def predict(model, batches, batch_size):
    logp = []
    for idx in iterate(batches, batch_size, train=False):
        x, _ = batches.get(idx, train=False)
        logp.append(torch.log_softmax(model(x).float(), 1).cpu())
    return torch.cat(logp)


def main():
    p = cli.parser("Evaluate a bundle.")
    p.add_argument("--model", required=True, help="teacher or student bundle (.pt)")
    p.add_argument("--data", default=None, help="an .npz with x, y and optionally group (jets, CIFAR, or audio spectrograms)")
    p.add_argument("--csv", default=None, help="audio CSV with path,label and optionally group")
    p.add_argument("--root", default=None, help="folder the CSV paths are relative to")
    p.add_argument("--data-root", default=None, help="torchvision CIFAR-10 folder; evaluates on its test split")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = pick_device(a.device)
    bundle = load_bundle(a.model)
    model = build_model(bundle)
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    batches, group = load_inputs(a, bundle, device)

    logp = predict(model, batches, a.batch_size)
    y = batches.y
    correct = (logp.argmax(1) == y).numpy()
    nll = -logp[torch.arange(len(y)), y].numpy()
    print("n %d  accuracy %.4f  nll %.4f" % (len(y), correct.mean(), nll.mean()))
    if group is not None:
        for g in np.unique(group):
            m = group == g
            print("group %d  n %d  accuracy %.4f  nll %.4f" % (g, m.sum(), correct[m].mean(), nll[m].mean()))


if __name__ == "__main__":
    main()
