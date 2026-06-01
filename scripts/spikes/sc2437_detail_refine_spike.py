"""sc-2437 spike — tile-based diffusion detail refine on MPS (GO/NO-GO for sc-2438).

Question: does a low-denoise SDXL/RealVisXL img2img pass run OVER TILES ("creative
upscale" / SUPIR-lite) add fine detail while preserving composition, with no visible
seams, on M5 Max (MPS, bf16) at acceptable VRAM + time? This is the net-new value over
the Phase-1 upscalers (Real-ESRGAN/AuraSR sharpen/enlarge but don't hallucinate new
micro-texture). Reuses the SDXL checkpoint already shipped — no new download.

It runs the REAL diffusers img2img pipeline (the same one the SDXL adapter uses for edit)
in a tiled, feather-blended refine — the exact loop sc-2438 would add to the adapter — so
the recipe transfers. Sweeps a couple of denoise strengths; reports a detail score
(high-frequency energy), a structure-preservation diff (low = composition kept), seams are
eyeballed from the PNGs, plus peak MPS memory and time/tile.

Run with the SceneWorks desktop venv (torch/diffusers/PIL/numpy):
  "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
      scripts/spikes/sc2437_detail_refine_spike.py \
      --checkpoint "stabilityai/stable-diffusion-xl-base-1.0" \
      --source /path/to/photo.png

Outputs PNGs + a JSON summary under --out (default /tmp/sc2437_detail).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SCENEWORKS_GPU_ID", "mps")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _peak_mem_gb(device: str) -> float | None:
    try:
        if device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        return None
    return None


# High-frequency energy: mean abs difference from a blurred copy. Higher = more fine
# detail/texture. Used to show the refine ADDED detail vs the (upscaled) input.
def detail_score(img: Image.Image) -> float:
    gray = img.convert("L")
    blur = gray.filter(ImageFilter.GaussianBlur(2))
    a = np.asarray(gray, dtype=np.float32)
    b = np.asarray(blur, dtype=np.float32)
    return round(float(np.abs(a - b).mean()), 3)


# Composition preservation: mean abs luma diff at low resolution. Low = same structure
# (we want detail added WITHOUT moving the picture around).
def structure_diff(a: Image.Image, b: Image.Image, size: int = 96) -> float:
    ga = np.asarray(a.convert("L").resize((size, size)), dtype=np.float32)
    gb = np.asarray(b.convert("L").resize((size, size)), dtype=np.float32)
    return round(float(np.abs(ga - gb).mean()), 2)


def _feather(tile_w: int, tile_h: int, overlap: int) -> np.ndarray:
    """Raised-cosine alpha ramp over the overlap borders so tiles blend seamlessly."""
    def ramp(n: int) -> np.ndarray:
        w = np.ones(n, dtype=np.float32)
        if overlap > 0:
            edge = np.array(
                [0.5 - 0.5 * math.cos(math.pi * (i + 0.5) / overlap) for i in range(overlap)],
                dtype=np.float32,
            )
            w[:overlap] = edge
            w[-overlap:] = edge[::-1]
        return w
    return np.outer(ramp(tile_h), ramp(tile_w))


def refine_tiled(pipe, image, *, prompt, negative, strength, tile, overlap, steps, guidance, seed):
    W, H = image.size
    step = max(tile - overlap, 1)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    xs = list(range(0, max(W - overlap, 1), step))
    ys = list(range(0, max(H - overlap, 1), step))
    tiles = 0
    for y in ys:
        for x in xs:
            x0, y0 = min(x, max(W - tile, 0)), min(y, max(H - tile, 0))
            crop = image.crop((x0, y0, x0 + tile, y0 + tile))
            tw, th = crop.size
            gen = torch.Generator(device="cpu").manual_seed(seed + tiles)
            refined = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=crop,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            ).images[0].resize((tw, th))
            feather = _feather(tw, th, overlap)[:, :, None]
            acc[y0:y0 + th, x0:x0 + tw] += np.asarray(refined, dtype=np.float32) * feather
            wsum[y0:y0 + th, x0:x0 + tw] += feather
            tiles += 1
    wsum[wsum == 0] = 1.0
    out = np.clip(acc / wsum, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), tiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="SDXL/RealVisXL HF id (from_pretrained, cached) — refine backbone.")
    parser.add_argument("--source", required=True, help="Source image (the detail target).")
    parser.add_argument("--prompt", default="ultra detailed, sharp focus, fine texture, high quality")
    parser.add_argument("--negative-prompt", default="blurry, soft, lowres, smooth, plastic")
    parser.add_argument("--preupscale", type=float, default=2.0,
                        help="Bicubic pre-upscale factor before refine (simulates refine-after-upscale).")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--strengths", default="0.25,0.4", help="Denoise strengths to sweep.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="/tmp/sc2437_detail")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device()

    src = Image.open(args.source).convert("RGB")
    if args.preupscale and args.preupscale != 1.0:
        src = src.resize((round(src.width * args.preupscale), round(src.height * args.preupscale)), Image.BICUBIC)
    src.save(out / "input_upscaled.png")
    base_detail = detail_score(src)
    print(f"input {src.size} (bicubic x{args.preupscale}) detailScore={base_detail}", flush=True)

    from diffusers import StableDiffusionXLImg2ImgPipeline

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    print(f"=== loading {args.checkpoint} img2img on {device} ===", flush=True)
    t_load = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.checkpoint, torch_dtype=dtype, variant="fp16")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    print(f"  loaded in {round(time.time() - t_load, 1)}s", flush=True)

    summary = {
        "device": device, "input": list(src.size), "preupscale": args.preupscale,
        "tile": args.tile, "overlap": args.overlap, "steps": args.steps,
        "inputDetailScore": base_detail, "runs": [],
    }
    for strength in [float(s) for s in args.strengths.split(",") if s.strip()]:
        t0 = time.time()
        try:
            refined, tiles = refine_tiled(
                pipe, src, prompt=args.prompt, negative=args.negative_prompt,
                strength=strength, tile=args.tile, overlap=args.overlap,
                steps=args.steps, guidance=args.guidance, seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            summary["runs"].append({"strength": strength, "error": str(exc)})
            print(f"  strength {strength} FAILED: {exc}", file=sys.stderr)
            continue
        elapsed = time.time() - t0
        refined.save(out / f"refined_s{strength}.png")
        run = {
            "strength": strength, "tiles": tiles, "seconds": round(elapsed, 1),
            "secPerTile": round(elapsed / max(tiles, 1), 1), "peakMemGb": _peak_mem_gb(device),
            "detailScore": detail_score(refined),
            "detailGain": round(detail_score(refined) - base_detail, 3),
            "structureDiff": structure_diff(src, refined),
        }
        summary["runs"].append(run)
        print(f"  strength {strength}: {tiles} tiles {run['seconds']}s ({run['secPerTile']}s/tile) | "
              f"detail {base_detail}->{run['detailScore']} (+{run['detailGain']}) | "
              f"structureDiff {run['structureDiff']} | peak {run['peakMemGb']}GB", flush=True)

    # GO: detail meaningfully up while structure stays close (composition preserved).
    good = [r for r in summary["runs"] if "error" not in r and r["detailGain"] > 0.3 and r["structureDiff"] <= 12]
    summary["verdict"] = "GO" if good else "REVIEW"
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"VERDICT: {summary['verdict']}  (PNGs + summary.json in {out})")
    print("GO = detailGain > 0.3 AND structureDiff ≤ 12 on ≥1 strength, and seams look clean.")
    print("Eyeball refined_*.png at 100%: texture added, tile boundaries invisible, no identity drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
