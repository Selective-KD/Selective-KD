import csv
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .standardize import Standardizer


def load_csv(csv_path, root=None):
    paths, labels, groups = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("%s must have columns 'path' and 'label'" % csv_path)
        has_group = "group" in reader.fieldnames
        for row in reader:
            p = row["path"]
            paths.append(os.path.join(root, p) if root else p)
            labels.append(int(row["label"]))
            if has_group:
                groups.append(int(row["group"]))
    return paths, np.asarray(labels, dtype=np.int64), (np.asarray(groups, dtype=np.int64) if has_group else None)


class LogMel:
    def __init__(self, device, sr=22050, clip_seconds=4.0, n_mels=64, n_fft=1024,
                 hop=512, power=2.0, floor=1e-6, rms_eps=1e-9):
        self.device, self.sr, self.hop = device, sr, hop
        self.clip_samples = int(round(clip_seconds * sr))
        self.n_frames = self.clip_samples // hop + 1
        self.n_mels, self.floor, self.rms_eps = n_mels, floor, rms_eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=power).to(device)

    def load(self, path):
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(w.mean(1))
        if sr != self.sr:
            w = torchaudio.functional.resample(w, sr, self.sr)
        n = self.clip_samples
        if len(w) >= n:
            w = w[:n]
        else:
            w = torch.nn.functional.pad(w, (0, n - len(w)))
        return w

    @torch.no_grad()
    def __call__(self, waves):
        w = waves.to(self.device)
        w = w / (w.pow(2).mean(-1, keepdim=True).sqrt() + self.rms_eps)
        return torch.log(self.mel(w) + self.floor)

    def render(self, paths, batch_size=256):
        out = np.empty((len(paths), self.n_mels, self.n_frames), dtype=np.float32)
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            waves = torch.stack([self.load(p) for p in chunk])
            out[i:i + len(chunk)] = self(waves).cpu().numpy()
        return out


FRONTEND_KEYS = ("sr", "clip_seconds", "n_mels", "n_fft", "hop")


def load_npz(path):
    f = np.load(path)
    x, y = np.asarray(f["x"], dtype=np.float32), np.asarray(f["y"], dtype=np.int64)
    if x.ndim != 3:
        raise ValueError("expected x (N, n_mels, n_frames), got %s" % (x.shape,))
    group = np.asarray(f["group"], dtype=np.int64) if "group" in f else None
    frontend = {k: float(f[k]) for k in FRONTEND_KEYS} if all(k in f for k in FRONTEND_KEYS) else None
    return x, y, group, frontend


def standardize(x):
    st = Standardizer(per_feature=False)
    return st.fit_transform(x), st
