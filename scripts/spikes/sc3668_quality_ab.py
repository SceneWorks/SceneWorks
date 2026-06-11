#!/usr/bin/env python
"""sc-3668 — quality A/B: AuraSR vs Real-ESRGAN x4.

Decision criterion (from the story): is AuraSR's quality enough better than the
already-ported Real-ESRGAN x4 to justify the (large) Mac port cost?

Two comparisons:
  (A) Objective: HR -> bicubic /4 -> SR x4 -> compare to HR (PSNR/SSIM).
  (B) Real-use: upscale a detail crop x4 and save a side-by-side strip for the
      eye (this is what users actually do; no ground truth).

Caveat: the poses/* images are synthetic clean renders, which UNDER-show AuraSR's
texture-hallucination strength and slightly favour the conservative Real-ESRGAN.
We note this; the decision is cost-dominated regardless.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OUT = Path("/tmp/sc3668")
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("poses/standing_09.png")  # same image family as the sc-3489 spike
AURA_SNAP = Path(
    "/Users/michael/.cache/huggingface/hub/models--fal--AuraSR-v2/"
    "snapshots/ff452185a7c8b51206dd62c21c292e7baad5c3a3"
)
ESRGAN_PTH = Path(
    "/Users/michael/Library/Application Support/SceneWorks/data/models/"
    "nateraw__real-esrgan/RealESRGAN_x4plus.pth"
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    # lightweight global SSIM on luma
    af = a.astype(np.float64).mean(2)
    bf = b.astype(np.float64).mean(2)
    mu_a, mu_b = af.mean(), bf.mean()
    va, vb = af.var(), bf.var()
    cov = ((af - mu_a) * (bf - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


def load_aura():
    import aura_sr
    from safetensors.torch import load_file

    cfg = json.loads((AURA_SNAP / "config.json").read_text())
    m = aura_sr.AuraSR(cfg, device="cpu")
    m.upsampler.load_state_dict(load_file(str(AURA_SNAP / "model.safetensors")), strict=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m.upsampler.to(dev).eval()
    return m


def aura_4x(m, img: Image.Image) -> Image.Image:
    return m.upscale_4x_overlapped(img.convert("RGB"), max_batch_size=8)


def load_esrgan():
    # Reuse the repo's pure-torch RRDBNet (upscalers.py) — avoids the basicsr/
    # torchvision functional_tensor break and matches what SceneWorks actually ships.
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "sw_upscalers", "apps/worker/scene_worker/upscalers.py"
    )
    up = importlib.util.module_from_spec(spec)
    sys.modules["sw_upscalers"] = up  # dataclass introspection needs the module registered
    spec.loader.exec_module(up)
    state = up._load_state_dict(torch, ESRGAN_PTH)
    num_blocks = up._infer_rrdb_blocks(state)
    RRDBNet = up._rrdbnet_class(torch)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=num_blocks, num_grow_ch=32)
    net.load_state_dict(state, strict=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net.to(dev).eval()
    return net


@torch.no_grad()
def esrgan_4x(net, img: Image.Image) -> Image.Image:
    dev = next(net.parameters()).device
    x = torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    y = net(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


def detail_crop(img: Image.Image, box) -> Image.Image:
    return img.crop(box)


def main() -> None:
    hr = Image.open(SRC).convert("RGB")
    W, H = hr.size
    print(f"HR source {SRC} = {W}x{H}")

    aura = load_aura()
    esr = load_esrgan()

    # (A) objective: downscale /4 then SR x4, compare to HR
    lr = hr.resize((W // 4, H // 4), Image.BICUBIC)
    t = time.time(); a_sr = aura_4x(aura, lr); ta = time.time() - t
    t = time.time(); e_sr = esrgan_4x(esr, lr); te = time.time() - t
    a_sr = a_sr.resize((W, H)); e_sr = e_sr.resize((W, H))
    bic = lr.resize((W, H), Image.BICUBIC)
    hr_a = np.asarray(hr)
    metrics = {
        "aura":   {"psnr": psnr(np.asarray(a_sr), hr_a), "ssim": ssim_gray(np.asarray(a_sr), hr_a), "sec": ta},
        "esrgan": {"psnr": psnr(np.asarray(e_sr), hr_a), "ssim": ssim_gray(np.asarray(e_sr), hr_a), "sec": te},
        "bicubic": {"psnr": psnr(np.asarray(bic), hr_a), "ssim": ssim_gray(np.asarray(bic), hr_a)},
    }
    print("(A) reference PSNR/SSIM vs HR (downscale/4 -> SR x4):")
    print(json.dumps(metrics, indent=2))

    # (B) real-use: upscale a 128px detail crop x4 -> 512px, side-by-side strip
    cx, cy = W // 2, H // 3
    box = (cx - 64, cy - 64, cx + 64, cy + 64)  # 128x128 crop
    crop = detail_crop(hr, box)
    a_up = aura_4x(aura, crop)          # 512x512
    e_up = esrgan_4x(esr, crop)         # 512x512
    bic_up = crop.resize((512, 512), Image.NEAREST)
    strip = Image.new("RGB", (512 * 3 + 20, 512 + 30), (20, 20, 20))
    for i, (im, label) in enumerate([(bic_up, "nearest"), (e_up, "Real-ESRGAN x4"), (a_up, "AuraSR x4")]):
        strip.paste(im.resize((512, 512)), (i * (512 + 10), 30))
    strip.save(OUT / "quality_strip.png")
    a_up.save(OUT / "crop_aura_x4.png")
    e_up.save(OUT / "crop_esrgan_x4.png")
    print(f"\n(B) detail strip -> {OUT/'quality_strip.png'} (order: nearest | Real-ESRGAN | AuraSR)")

    (OUT / "quality_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
