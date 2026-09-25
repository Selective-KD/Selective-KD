"""Hendrycks & Dietterich 2019, the CIFAR-C corruptions at severity 3."""
import os

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

SEVERITY = 3
_S = SEVERITY - 1

FROST_URL = "https://raw.githubusercontent.com/hendrycks/robustness/master/ImageNet-C/create_c/"
FROST_FILES = ["frost1.png", "frost2.png", "frost3.png", "frost4.jpg", "frost5.jpg"]
_FROST = {}


def _to_uint8(x01):
    return np.uint8(np.clip(x01, 0, 1) * 255)


def gaussian_blur(x, rng):
    sigma = [.4, .6, 0.7, .8, 1][_S]
    return _to_uint8(gaussian_filter(np.asarray(x, dtype=np.float64) / 255.0, sigma=(sigma, sigma, 0)))


def glass_blur(x, rng):
    sigma, max_delta, iterations = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][_S]
    x01 = gaussian_filter(np.asarray(x, dtype=np.float64) / 255.0, sigma=(sigma, sigma, 0), mode="reflect")
    h, w = x01.shape[:2]
    for _ in range(iterations):
        for i in range(h - max_delta, max_delta, -1):
            for j in range(w - max_delta, max_delta, -1):
                dx, dy = rng.integers(-max_delta, max_delta, size=2)
                i2, j2 = i + dy, j + dx
                x01[i, j], x01[i2, j2] = x01[i2, j2].copy(), x01[i, j].copy()
    return _to_uint8(gaussian_filter(x01, sigma=(sigma, sigma, 0), mode="reflect"))


def _disk_kernel(radius, alias_blur):
    L = np.arange(-8, 9) if radius <= 8 else np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    k = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float64)
    return gaussian_filter(k / k.sum(), sigma=alias_blur)


def defocus_blur(x, rng):
    radius, alias = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1.0, 0.2), (1.5, 0.1)][_S]
    x01 = np.asarray(x, dtype=np.float64) / 255.0
    k = _disk_kernel(radius, alias)
    return _to_uint8(np.stack([convolve(x01[..., d], k, mode="reflect") for d in range(3)], -1))


def impulse_noise(x, rng):
    amount = [.01, .02, .03, .05, .07][_S]
    x01 = np.asarray(x, dtype=np.float64) / 255.0
    flipped = rng.random(x01.shape) < amount
    salted = rng.random(x01.shape) < 0.5
    x01 = x01.copy()
    x01[flipped & salted] = 1.0
    x01[flipped & ~salted] = 0.0
    return _to_uint8(x01)


def gaussian_noise(x, rng):
    c = [0.04, 0.06, .08, .09, .10][_S]
    return _to_uint8(np.asarray(x, dtype=np.float64) / 255.0 + rng.normal(size=x.shape, scale=c))


def spatter(x, rng):
    import cv2
    c = [(0.62, 0.1, 0.7, 0.7, 0.5, 0), (0.65, 0.1, 0.8, 0.7, 0.5, 0), (0.65, 0.3, 1, 0.69, 0.5, 0),
         (0.65, 0.1, 0.7, 0.69, 0.6, 1), (0.65, 0.1, 0.5, 0.68, 0.6, 1)][_S]
    x01 = np.array(x, dtype=np.float32) / 255.
    liquid = gaussian_filter(rng.normal(loc=c[0], scale=c[1], size=x01.shape[:2]), sigma=c[2], mode="nearest", truncate=4.0)
    liquid[liquid < c[3]] = 0
    if c[5] == 0:
        liquid = (liquid * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        dist = cv2.filter2D(dist, cv2.CV_8U, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]))
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        m = cv2.cvtColor(liquid * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]), 238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x01 = cv2.cvtColor(x01, cv2.COLOR_BGR2BGRA)
        return _to_uint8(cv2.cvtColor(np.clip(x01 + m * color, 0, 1), cv2.COLOR_BGRA2BGR))
    m = gaussian_filter(np.where(liquid > c[3], 1, 0).astype(np.float32), sigma=c[4], mode="nearest", truncate=4.0)
    m[m < 0.8] = 0
    color = np.concatenate((63 / 255. * np.ones_like(x01[..., :1]), 42 / 255. * np.ones_like(x01[..., :1]),
                            20 / 255. * np.ones_like(x01[..., :1])), axis=2) * m[..., np.newaxis]
    return _to_uint8(np.clip(x01 * (1 - m[..., np.newaxis]) + color, 0, 1))


def ensure_frost(frost_dir):
    import urllib.request
    os.makedirs(frost_dir, exist_ok=True)
    for f in FROST_FILES:
        p = os.path.join(frost_dir, f)
        if not os.path.exists(p):
            print("fetching %s" % f)
            urllib.request.urlretrieve(FROST_URL + f, p)


def _frost_images(frost_dir):
    import cv2
    if frost_dir not in _FROST:
        ensure_frost(frost_dir)
        imgs = []
        for f in FROST_FILES:
            im = cv2.imread(os.path.join(frost_dir, f))
            if im is None:
                raise ValueError("unreadable frost image %s" % f)
            imgs.append(cv2.resize(im, (0, 0), fx=0.2, fy=0.2))
        _FROST[frost_dir] = imgs
    return _FROST[frost_dir]


def frost(x, rng, frost_dir="frost"):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][_S]
    fr = _frost_images(frost_dir)[int(rng.integers(5))]
    xs, ys = int(rng.integers(0, fr.shape[0] - 32)), int(rng.integers(0, fr.shape[1] - 32))
    fr = fr[xs:xs + 32, ys:ys + 32][..., [2, 1, 0]]
    return np.uint8(np.clip(c[0] * np.array(x, dtype=np.float64) + c[1] * fr, 0, 255))


CORRUPTIONS = {
    "gaussian_blur": gaussian_blur, "glass_blur": glass_blur, "defocus_blur": defocus_blur,
    "impulse_noise": impulse_noise, "gaussian_noise": gaussian_noise, "spatter": spatter, "frost": frost,
}
NAMES = sorted(CORRUPTIONS)


def corrupt(name, images, rng, frost_dir="frost"):
    fn = CORRUPTIONS[name]
    if name == "frost":
        return np.stack([fn(img, rng, frost_dir) for img in images])
    return np.stack([fn(img, rng) for img in images])
