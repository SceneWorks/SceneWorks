"""sc-2758 — SDXL acceleration A/B: LCM-LoRA vs SDXL-Lightning vs Hyper-SD.

Spike question (epic 2755 S3): characterize all three SDXL few-step acceleration methods on the
two torch-path bases SceneWorks ships (`sdxl`, `realvisxl`) and LOCK the per-variant defaults
(steps / CFG / scheduler) that feed BOTH the torch impl (sc-2760/2761) and the already-merged
MLX impl re-tune (sc-2769 → sc-2907). The few-step schedulers (LCMScheduler / EulerDiscrete
`timestep_spacing=trailing` / TCDScheduler / DDIM-trailing) live only in diffusers, so this runs
on the torch+diffusers stack (here: the packaged SceneWorks worker venv, MPS).

Unlike the mlx-gen golden tool (`tools/dump_sdxl_accel_golden.py`, which forces the 512 micro-
conditioning convention + single 4-step renders for MLX bit-parity), this renders at the REAL
1024² micro-conditioning so the output reflects production quality — this is a quality A/B, not a
parity dump.

Matrix per (base, prompt):
  standard      : 30 steps, CFG 7.0, base EulerDiscrete           (quality anchor)
  lcm           : LCMScheduler, CFG 1.0, steps {2,4,8} + a CFG-2.0 probe at 4
  lightning     : EulerDiscrete(trailing), CFG 1.0, steps {2,4,8} (each its matching N-step LoRA)
  hyper (TCD)   : TCDScheduler eta=0, CFG 1.0, steps {1,2,4,8}    (each its matching N-step LoRA)
  hyper (DDIM)  : DDIMScheduler(trailing), CFG 1.0 — 1-step (timesteps=[800]) + 4-step tie-break

Outputs to /tmp/sc2758_sdxl_accel/:
  <base>__<tag>.png             individual renders
  grid__<base>__<prompt>.png    labelled comparison montage (method rows × step columns)
  results.json                  every render's exact config + wall-clock + s/step

Run (packaged worker venv has torch+diffusers+MPS):
  PY="/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python"
  "$PY" scripts/spikes/sc2758_sdxl_acceleration_spike.py            # full sweep (2 bases x 2 prompts)
  "$PY" scripts/spikes/sc2758_sdxl_acceleration_spike.py --smoke    # sdxl + fox only, fast subset
  "$PY" scripts/spikes/sc2758_sdxl_acceleration_spike.py --bases sdxl --prompts fox
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is cached; never hit the network

OUT_DIR = "/tmp/sc2758_sdxl_accel"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SDXL_SEED", "42"))
W = int(os.environ.get("SDXL_W", "1024"))
H = int(os.environ.get("SDXL_H", "1024"))

BASES = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "realvisxl": "SG161222/RealVisXL_V5.0",
}

PROMPTS = {
    "fox": "a majestic red fox standing in a misty autumn forest at dawn, volumetric light, "
    "highly detailed fur, sharp focus, photorealistic",
    "portrait": "professional studio photograph of a young woman with freckles, soft window "
    "light, 85mm lens, shallow depth of field, highly detailed realistic skin texture",
}

LCM_REPO = "latent-consistency/lcm-lora-sdxl"
LCM_FILE = "pytorch_lora_weights.safetensors"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
HYPER_REPO = "ByteDance/Hyper-SD"


@dataclass
class Render:
    tag: str              # unique id within a base
    method: str           # standard|lcm|lightning|hyper
    scheduler: str        # euler|lcm|euler_trailing|tcd|ddim_trailing
    steps: int
    cfg: float
    lora_repo: Optional[str] = None
    lora_file: Optional[str] = None
    col: str = ""         # montage column key (e.g. "4")
    eta: Optional[float] = None
    timesteps: Optional[list[int]] = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_matrix() -> list[Render]:
    r: list[Render] = []
    r.append(Render("standard_30", "standard", "euler", 30, 7.0, col="30"))

    # LCM-LoRA — one LoRA file, vary steps/CFG.
    for n in (2, 4, 8):
        r.append(Render(f"lcm_{n}", "lcm", "lcm", n, 1.0, LCM_REPO, LCM_FILE, col=str(n)))
    r.append(Render("lcm_4_cfg2", "lcm", "lcm", 4, 2.0, LCM_REPO, LCM_FILE, col="4·cfg2"))

    # SDXL-Lightning — matching N-step LoRA per step count, Euler-trailing, CFG off.
    for n in (2, 4, 8):
        r.append(
            Render(
                f"lightning_{n}", "lightning", "euler_trailing", n, 1.0,
                LIGHTNING_REPO, f"sdxl_lightning_{n}step_lora.safetensors", col=str(n),
            )
        )

    # Hyper-SD — matching N-step LoRA per step count. Primary scheduler = TCD (eta=0).
    hyper_file = {1: "Hyper-SDXL-1step-lora.safetensors", 2: "Hyper-SDXL-2steps-lora.safetensors",
                  4: "Hyper-SDXL-4steps-lora.safetensors", 8: "Hyper-SDXL-8steps-lora.safetensors"}
    for n in (1, 2, 4, 8):
        r.append(
            Render(
                f"hyper_tcd_{n}", "hyper", "tcd", n, 1.0,
                HYPER_REPO, hyper_file[n], col=str(n), eta=0.0,
            )
        )
    # Scheduler tie-breaks: official 1-step DDIM(trailing, t=800) recipe + a 4-step DDIM head-to-head.
    r.append(
        Render("hyper_ddim_1", "hyper", "ddim_trailing", 1, 1.0, HYPER_REPO, hyper_file[1],
               col="1·ddim", timesteps=[800])
    )
    r.append(
        Render("hyper_ddim_4", "hyper", "ddim_trailing", 4, 1.0, HYPER_REPO, hyper_file[4],
               col="4·ddim")
    )
    return r


SMOKE_TAGS = {"standard_30", "lcm_4", "lightning_4", "hyper_tcd_4", "hyper_ddim_4"}


def make_scheduler(kind: str, base_config):
    from diffusers import (
        DDIMScheduler,
        EulerDiscreteScheduler,
        LCMScheduler,
        TCDScheduler,
    )

    if kind == "euler":
        return EulerDiscreteScheduler.from_config(base_config)
    if kind == "lcm":
        return LCMScheduler.from_config(base_config)
    if kind == "euler_trailing":
        return EulerDiscreteScheduler.from_config(base_config, timestep_spacing="trailing")
    if kind == "tcd":
        return TCDScheduler.from_config(base_config)
    if kind == "ddim_trailing":
        return DDIMScheduler.from_config(base_config, timestep_spacing="trailing")
    raise ValueError(kind)


def load_pipe(repo: str):
    from diffusers import StableDiffusionXLPipeline

    last_err = None
    for kw in (dict(variant="fp16", use_safetensors=True), dict(use_safetensors=True)):
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(repo, torch_dtype=torch.float16, **kw)
            pipe.to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
            return pipe
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"failed to load {repo}: {last_err}")


def run_render(pipe, base_config, prompt: str, rnd: Render) -> dict[str, Any]:
    pipe.scheduler = make_scheduler(rnd.scheduler, base_config)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    kw: dict[str, Any] = dict(
        prompt=prompt, width=W, height=H, guidance_scale=rnd.cfg, generator=g, output_type="pil",
    )
    if rnd.timesteps is not None:
        kw["timesteps"] = rnd.timesteps
    else:
        kw["num_inference_steps"] = rnd.steps
    if rnd.eta is not None:
        kw["eta"] = rnd.eta

    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    img = pipe(**kw).images[0]
    if DEVICE == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0

    return {"image": img, "seconds": round(dt, 3), "s_per_step": round(dt / max(rnd.steps, 1), 3)}


def _font(size: int):
    for p in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label(img: Image.Image, text: str, thumb: int = 320) -> Image.Image:
    im = img.convert("RGB").resize((thumb, thumb))
    bar = 46
    canvas = Image.new("RGB", (thumb, thumb + bar), (18, 18, 18))
    canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    f = _font(15)
    for i, line in enumerate(text.split("\n")[:2]):
        d.text((6, 4 + i * 19), line, fill=(235, 235, 235), font=f)
    return canvas


def build_grid(base: str, prompt_key: str, rendered: list[tuple[Render, dict]]):
    """Montage: one row per method, columns ordered by step count (+ probes)."""
    methods = ["standard", "lcm", "lightning", "hyper"]
    by_method: dict[str, list[tuple[Render, dict]]] = {m: [] for m in methods}
    for rnd, res in rendered:
        by_method.setdefault(rnd.method, []).append((rnd, res))

    thumb, bar, pad, hdr = 320, 46, 8, 30
    ncols = max((len(v) for v in by_method.values()), default=1)
    cell_w, cell_h = thumb + pad, thumb + bar + pad
    grid_w = pad + max(ncols, 1) * cell_w + 150
    grid_h = hdr + len(methods) * cell_h + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (10, 10, 10))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), f"sc-2758  base={base}  prompt={prompt_key}  seed={SEED}  {W}x{H}  ({DEVICE})",
           fill=(255, 220, 120), font=_font(17))

    for ri, m in enumerate(methods):
        y = hdr + ri * cell_h
        d.text((grid_w - 144, y + thumb // 2), m.upper(), fill=(180, 200, 255), font=_font(16))
        cells = sorted(by_method.get(m, []), key=lambda x: (x[0].steps, x[0].tag))
        for ci, (rnd, res) in enumerate(cells):
            x = pad + ci * cell_w
            sched = rnd.scheduler.replace("euler_trailing", "euler-tr").replace("ddim_trailing", "ddim-tr")
            txt = f"{rnd.steps}st cfg{rnd.cfg:g} {sched}\n{res['seconds']}s ({res['s_per_step']}s/st)"
            canvas.paste(label(res["image"], txt, thumb), (x, y))
    out = os.path.join(OUT_DIR, f"grid__{base}__{prompt_key}.png")
    canvas.save(out)
    print(f"  grid → {out}")
    return out


def free(pipe):
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="sdxl + fox, key subset only")
    ap.add_argument("--bases", nargs="*", choices=list(BASES), default=None)
    ap.add_argument("--prompts", nargs="*", choices=list(PROMPTS), default=None)
    args = ap.parse_args()

    bases = args.bases or (["sdxl"] if args.smoke else list(BASES))
    prompts = args.prompts or (["fox"] if args.smoke else list(PROMPTS))
    matrix = build_matrix()
    if args.smoke:
        matrix = [r for r in matrix if r.tag in SMOKE_TAGS]

    import diffusers

    print(f"diffusers {diffusers.__version__} torch {torch.__version__} on {DEVICE}")
    print(f"bases={bases} prompts={prompts} renders/base/prompt={len(matrix)}")

    results: list[dict] = []
    results_path = os.path.join(OUT_DIR, "results.json")

    # Group renders by LoRA file so we load+fuse each LoRA group exactly once (fresh pipe per group
    # avoids fp16 unfuse residue across different LoRAs).
    def group_key(r: Render):
        return (r.lora_repo or "", r.lora_file or "")

    for base in bases:
        repo = BASES[base]
        order: list = []
        seen = set()
        for r in matrix:
            k = group_key(r)
            if k not in seen:
                seen.add(k)
                order.append(k)
        for prompt_key in prompts:
            prompt = PROMPTS[prompt_key]
            rendered: list[tuple[Render, dict]] = []
            for gk in order:
                grp = [r for r in matrix if group_key(r) == gk]
                lora_repo, lora_file = gk
                print(f"\n[{base}/{prompt_key}] group lora={lora_file or '(none)'} "
                      f"({len(grp)} render(s))")
                pipe = load_pipe(repo)
                base_config = dict(pipe.scheduler.config)
                if lora_repo:
                    try:
                        path = hf_hub_download(lora_repo, lora_file)
                        pipe.load_lora_weights(os.path.dirname(path),
                                               weight_name=os.path.basename(path))
                        pipe.fuse_lora()
                        pipe.unload_lora_weights()
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! LoRA load failed ({lora_file}): {e}")
                        free(pipe)
                        continue
                for rnd in grp:
                    try:
                        res = run_render(pipe, base_config, prompt, rnd)
                        png = os.path.join(OUT_DIR, f"{base}__{prompt_key}__{rnd.tag}.png")
                        res["image"].convert("RGB").save(png)
                        rendered.append((rnd, res))
                        rec = {
                            "base": base, "prompt_key": prompt_key, "tag": rnd.tag,
                            "method": rnd.method, "scheduler": rnd.scheduler, "steps": rnd.steps,
                            "cfg": rnd.cfg, "eta": rnd.eta, "timesteps": rnd.timesteps,
                            "lora_file": lora_file, "seconds": res["seconds"],
                            "s_per_step": res["s_per_step"], "png": png,
                        }
                        results.append(rec)
                        print(f"  ✓ {rnd.tag:16s} {rnd.steps:>2}st cfg{rnd.cfg:<3g} "
                              f"{rnd.scheduler:14s} {res['seconds']:>6.2f}s "
                              f"({res['s_per_step']:.2f}s/st)")
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=2)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ✗ {rnd.tag}: {type(e).__name__}: {e}")
                free(pipe)
            if rendered:
                build_grid(base, prompt_key, rendered)

    print(f"\nDONE — {len(results)} renders. results.json + grids in {OUT_DIR}")


if __name__ == "__main__":
    main()
