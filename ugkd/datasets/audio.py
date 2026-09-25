import csv
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

INT16 = 32767.0
RMS_MIN = 1e-4
RMS_EPS = 1e-9


def read_metadata(root):
    rows = []
    with open(os.path.join(root, "metadata", "UrbanSound8K.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"path": os.path.join(root, "audio", "fold%s" % r["fold"], r["slice_file_name"]),
                         "source": int(r["fsID"]), "label": int(r["classID"])})
    return rows


def _best_window(x, L):
    if len(x) <= L:
        return np.pad(x, (0, L - len(x)))
    best, best_e = None, -1.0
    for off in (0, L // 8, L // 4):
        w = x[off:off + L]
        if len(w) < L:
            continue
        e = float((w ** 2).mean())
        if e > best_e:
            best, best_e = w, e
    return best


def decode(rows, cache_path, sr, clip_seconds, log=print):
    L = int(round(clip_seconds * sr))
    rms_path = cache_path + ".rms.npy"
    if os.path.exists(cache_path) and os.path.exists(rms_path):
        mm = np.load(cache_path, mmap_mode="r")
        if mm.shape == (len(rows), L):
            log("cache present: %s" % cache_path)
            return mm, np.load(rms_path)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    mm = np.lib.format.open_memmap(cache_path, "w+", np.int16, (len(rows), L))
    rms = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        try:
            x, fs = sf.read(r["path"], dtype="float32", always_2d=True)
            x = x.mean(1)
            if fs != sr:
                x = torchaudio.functional.resample(torch.from_numpy(x), fs, sr).numpy()
            x = _best_window(x, L)
        except Exception as e:
            log("skip %s: %s" % (r["path"], e))
            x = np.zeros(L, np.float32)
        rms[i] = float(np.sqrt((x ** 2).mean()))
        mm[i] = np.clip(x, -1, 1) * INT16
        if i % 500 == 0:
            log("  decoded %d/%d" % (i, len(rows)))
    mm.flush()
    np.save(rms_path, rms)
    return np.load(cache_path, mmap_mode="r"), rms


def split_sources(rows, rms, teacher_frac, student_frac, seed):
    labels = np.array([r["label"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    keep = rms >= RMS_MIN
    owner = {}
    for s in np.unique(sources[keep]):
        m = keep & (sources == s)
        owner[s] = int(np.bincount(labels[m]).argmax())
    rng = np.random.default_rng(seed)
    part = {}
    for c in np.unique(labels):
        ids = np.array(sorted(s for s, o in owner.items() if o == c))
        rng.shuffle(ids)
        n1 = int(round(teacher_frac * len(ids)))
        n2 = int(round((teacher_frac + student_frac) * len(ids)))
        for name, chunk in (("teacher", ids[:n1]), ("student", ids[n1:n2]), ("test", ids[n2:])):
            for s in chunk:
                part[s] = name
    pools = {name: {} for name in ("teacher", "student", "test")}
    for i in np.flatnonzero(keep):
        pools[part[sources[i]]].setdefault(labels[i], []).append(i)
    K = len(np.unique(labels))
    for name, p in pools.items():
        missing = [c for c in range(K) if c not in p]
        if missing:
            raise ValueError("pool %s has no clips for classes %s" % (name, missing))
        pools[name] = {c: np.array(v) for c, v in p.items()}
    return pools


def render(pool, n, *, bank, front, beta, mask_bins, rng, batch_size=256):
    K = len(pool)
    half = K // 2
    partner = lambda c: c + half if c < half else c - half
    dom = rng.integers(0, K, n)
    iP = np.array([rng.choice(pool[int(c)]) for c in dom])
    iQ = np.array([rng.choice(pool[partner(int(c))]) for c in dom])
    out = np.empty((n, front.n_mels, front.n_frames), np.float32)
    floor = float(np.log(front.floor))
    unit = lambda w: w / (w.pow(2).mean(-1, keepdim=True).sqrt() + RMS_EPS)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            P = torch.from_numpy(np.asarray(bank[np.sort(iP[sl])], np.float32))[np.argsort(np.argsort(iP[sl]))] / INT16
            Q = torch.from_numpy(np.asarray(bank[np.sort(iQ[sl])], np.float32))[np.argsort(np.argsort(iQ[sl]))] / INT16
            mix = unit(beta * unit(P) + (1.0 - beta) * unit(Q))
            s = front(mix.to(front.device))
            if mask_bins > 0:
                B, F_, _ = s.shape
                start = torch.as_tensor(rng.integers(0, F_ - mask_bins + 1, B), device=s.device)
                ar = torch.arange(F_, device=s.device)[None, :]
                band = (ar >= start[:, None]) & (ar < (start + mask_bins)[:, None])
                s = s.masked_fill(band[:, :, None], floor)
            out[sl] = s.cpu().numpy()
    return out, dom.astype(np.int64)


def build(pools, bank, front, *, beta, mask_bins, f, n_teacher, n_student, n_test, seed, log=print):
    rng = np.random.default_rng(seed)
    out = {}
    log("teacher_train: %d mixtures" % n_teacher)
    x, y = render(pools["teacher"], n_teacher, bank=bank, front=front, beta=beta, mask_bins=0, rng=rng)
    out["teacher_train"] = (x, y, np.zeros(n_teacher, dtype=np.int64))
    for name, pool, n in (("student_train", pools["student"], n_student), ("test", pools["test"], n_test)):
        n_b = int(round(f * n))
        log("%s: %d mixtures, %d masked" % (name, n, n_b))
        xa, ya = render(pool, n - n_b, bank=bank, front=front, beta=beta, mask_bins=0, rng=rng)
        xb, yb = render(pool, n_b, bank=bank, front=front, beta=beta, mask_bins=mask_bins, rng=rng)
        x, y = np.concatenate([xa, xb]), np.concatenate([ya, yb])
        group = np.concatenate([np.zeros(n - n_b, dtype=np.int64), np.ones(n_b, dtype=np.int64)])
        perm = rng.permutation(n)
        out[name] = (x[perm], y[perm], group[perm])
    return out
