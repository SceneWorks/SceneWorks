#!/usr/bin/env python
"""sc-3668 — texture-crop A/B: AuraSR vs Real-ESRGAN x4 on a real photo.

Gives AuraSR its best shot: a real photographic image with fine texture (the
rusty robot / candle / curtain scene), upscaling texture-rich crops x4 so the
GAN's detail-hallucination strength (vs Real-ESRGAN's conservative output) is
visible to the eye. No CoreML, no MLX — both run on torch/MPS (the path the
models use today). This is the deciding input for MLX-port-vs-drop.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

OUT = Path("/tmp/sc3668")
SRC = Path("docs/sc-3494/images/candle-sdxl-seed42.png")  # 1024x1024 real render
AURA_SNAP = Path(
    "~/.cache/huggingface/hub/models--fal--AuraSR-v2/"
    "snapshots/ff452185a7c8b51206dd62c21c292e7baad5c3a3"
)
ESRGAN_PTH = Path(
    "~/Library/Application Support/SceneWorks/data/models/"
    "nateraw__real-esrgan/RealESRGAN_x4plus.pth"
)
# texture-rich crops in the 1024^2 image: (name, cx, cy, size)
CROPS = [
    ("rusty_head", 660, 250, 200),
    ("candle_flame", 250, 470, 200),
    ("torso_rivets", 560, 720, 200),
]


def load_aura():
    import aura_sr
    from safetensors.torch import load_file

    cfg = json.loads((AURA_SNAP / "config.json").read_text())
    m = aura_sr.AuraSR(cfg, device="cpu")
    m.upsampler.load_state_dict(load_file(str(AURA_SNAP / "model.safetensors")), strict=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m.upsampler.to(dev).eval()
    return m


def load_esrgan():
    spec = importlib.util.spec_from_file_location(
        "sw_upscalers", "apps/worker/scene_worker/upscalers.py"
    )
    up = importlib.util.module_from_spec(spec)
    sys.modules["sw_upscalers"] = up
    spec.loader.exec_module(up)
    state = up._load_state_dict(torch, ESRGAN_PTH)
    RRDBNet = up._rrdbnet_class(torch)
    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
        num_block=up._infer_rrdb_blocks(state), num_grow_ch=32,
    )
    net.load_state_dict(state, strict=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net.to(dev).eval()
    return net


def aura_4x(m, img):
    return m.upscale_4x_overlapped(img.convert("RGB"), max_batch_size=8)


@torch.no_grad()
def esrgan_4x(net, img):
    dev = next(net.parameters()).device
    x = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    y = net(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 8 + 8, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return im


def main():
    src = Image.open(SRC).convert("RGB")
    aura, esr = load_aura(), load_esrgan()
    timing = {}
    panels = []
    for name, cx, cy, size in CROPS:
        h = size // 2
        crop = src.crop((cx - h, cy - h, cx + h, cy + h))  # size x size
        out = size * 4
        t = time.time(); a = aura_4x(aura, crop).resize((out, out)); timing[f"{name}_aura_s"] = time.time() - t
        t = time.time(); e = esrgan_4x(esr, crop).resize((out, out)); timing[f"{name}_esrgan_s"] = time.time() - t
        n = crop.resize((out, out), Image.NEAREST)
        row = Image.new("RGB", (out * 3 + 20, out + 4), (15, 15, 15))
        for i, (im, lab) in enumerate([(n, f"{name}: nearest"), (e, "Real-ESRGAN x4"), (a, "AuraSR x4")]):
            row.paste(label(im.copy(), lab), (i * (out + 10), 2))
        panels.append(row)
    W = max(p.width for p in panels)
    H = sum(p.height for p in panels) + 10 * (len(panels) - 1)
    strip = Image.new("RGB", (W, H), (15, 15, 15))
    y = 0
    for p in panels:
        strip.paste(p, (0, y)); y += p.height + 10
    strip.save(OUT / "quality_crops.png")
    print("timing:", json.dumps(timing, indent=2))
    print(f"saved -> {OUT/'quality_crops.png'} ({strip.size})")


if __name__ == "__main__":
    main()
