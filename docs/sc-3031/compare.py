#!/usr/bin/env python
"""sc-3031 A/B compare: metrics between the Rust new-adapter PNG and the Python
old-adapter PNG. Independent reimplementations won't be bit-exact; we report the
same family of metrics the mlx-gen engine goldens use (mean-abs, max-abs, px>8
fraction) plus PSNR/SSIM for a human-readable similarity read."""
import sys

import numpy as np
from PIL import Image


def load(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float64)


def ssim(a, b):
    # global single-window SSIM on luma (rough; for a quick read, not WCAG-grade)
    ya = a @ [0.299, 0.587, 0.114]
    yb = b @ [0.299, 0.587, 0.114]
    mu_a, mu_b = ya.mean(), yb.mean()
    va, vb = ya.var(), yb.var()
    cov = ((ya - mu_a) * (yb - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    )


rust_path, py_path = sys.argv[1], sys.argv[2]
a = load(rust_path)
b = load(py_path, size=Image.open(rust_path).size)
diff = np.abs(a - b)
mean_abs = diff.mean()
max_abs = diff.max()
px_gt8 = float((diff.max(axis=2) > 8).mean()) * 100.0
mse = (diff**2).mean()
psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)
print(
    f"mean_abs={mean_abs:.2f} max_abs={max_abs:.0f} px>8={px_gt8:.1f}% "
    f"psnr={psnr:.2f}dB ssim={ssim(a, b):.4f}"
)
