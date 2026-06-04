"""sc-2758 — InstantID-under-acceleration viability.

Spike sub-question: does InstantID face identity survive when the SDXL UNet is driven by an
acceleration LoRA (Lightning / Hyper) + a few-step scheduler instead of the standard 30-step
Euler? InstantID is an SDXL ControlNet+IP-Adapter stack on RealVisXL; if identity holds at
4-8 steps it unlocks an ~8x faster character path. If the few-step schedulers collapse the
ControlNet guidance, InstantID stays standard-only.

Mirrors `scene_worker/instantid_adapter.py` (identity-only `_run_pipeline` path): InstantX
vendored `StableDiffusionXLInstantIDPipeline`, IdentityNet ControlNet from `InstantX/InstantID`,
antelopev2 ArcFace embedding + letterboxed kps control image, bf16/MPS (fp16 NaNs on Metal).

Measures ArcFace cosine(reference, generated) per render — the same metric as the sc-2009 /
qwen-2511 identity spikes (InstantID no-restore baseline ~0.68-0.88) — so the accel deltas are
read on the established scale.

Run (packaged worker venv has insightface+onnxruntime+diffusers):
  PY="/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python"
  "$PY" scripts/spikes/sc2758_instantid_accel_spike.py
  "$PY" scripts/spikes/sc2758_instantid_accel_spike.py --ref ~/Datasets/Kelsie/kelsie_0073.png
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_DIR = "/tmp/sc2758_instantid_accel"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_REPO = "SG161222/RealVisXL_V5.0"  # InstantID ships on RealVisXL in SceneWorks
INSTANT_REPO = "InstantX/InstantID"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float16  # fp16 NaNs on Metal (sc-2009)
SEED = 42
W = H = 1024

PROMPT = "cinematic photo of a person sitting at a cafe by a window, warm afternoon light, " \
    "candid, highly detailed, photorealistic, 50mm"
NEG = "blurry, low quality, distorted, deformed, cartoon, painting, watermark, text"

LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
HYPER_REPO = "ByteDance/Hyper-SD"

# Acceleration configs to test on top of InstantID. Standard = the production baseline.
# CFG off (1.0) for the distilled methods. ip/cn scales held at the production 0.8.
CONFIGS = [
    dict(tag="standard_30", method="standard", sched="euler", steps=30, cfg=5.0, lora=None),
    dict(tag="lightning_8", method="lightning", sched="euler_trailing", steps=8, cfg=1.0,
         lora=(LIGHTNING_REPO, "sdxl_lightning_8step_lora.safetensors")),
    dict(tag="lightning_4", method="lightning", sched="euler_trailing", steps=4, cfg=1.0,
         lora=(LIGHTNING_REPO, "sdxl_lightning_4step_lora.safetensors")),
    dict(tag="hyper_8", method="hyper", sched="tcd", steps=8, cfg=1.0, eta=0.0,
         lora=(HYPER_REPO, "Hyper-SDXL-8steps-lora.safetensors")),
    dict(tag="hyper_4", method="hyper", sched="tcd", steps=4, cfg=1.0, eta=0.0,
         lora=(HYPER_REPO, "Hyper-SDXL-4steps-lora.safetensors")),
    dict(tag="lcm_8", method="lcm", sched="lcm", steps=8, cfg=1.0,
         lora=("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors")),
]

_VENDOR = Path(__file__).resolve().parents[2] / "apps" / "worker" / "scene_worker" / "_vendor" / "instantid"


def _import_instantid():
    v = str(_VENDOR)
    if v not in sys.path:
        sys.path.insert(0, v)
    import importlib

    mod = importlib.import_module("pipeline_stable_diffusion_xl_instantid")
    return mod.StableDiffusionXLInstantIDPipeline, mod.draw_kps


def _ensure_antelopev2() -> Path:
    root = Path(os.environ.get("INSTANTID_INSIGHTFACE_ROOT", str(Path.home() / ".insightface")))
    dest = root / "models" / "antelopev2"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"):
        if not (dest / name).exists():
            data = Path(hf_hub_download("DIAMONIK7777/antelopev2", name)).read_bytes()
            (dest / name).write_bytes(data)
    return root


_FA = None


def face_analysis():
    global _FA
    if _FA is None:
        from insightface.app import FaceAnalysis

        root = _ensure_antelopev2()
        app = FaceAnalysis(name="antelopev2", root=str(root), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _FA = app
    return _FA


def letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image.resize((nw, nh), Image.LANCZOS), ((width - nw) // 2, (height - nh) // 2))
    return canvas


def largest_face(pil: Image.Image):
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    faces = face_analysis().get(bgr)
    if not faces:
        raise RuntimeError("no face detected")
    return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))[-1]


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def make_scheduler(kind, base_config):
    from diffusers import DDIMScheduler, EulerDiscreteScheduler, LCMScheduler, TCDScheduler

    if kind == "euler":
        return EulerDiscreteScheduler.from_config(base_config)
    if kind == "euler_trailing":
        return EulerDiscreteScheduler.from_config(base_config, timestep_spacing="trailing")
    if kind == "lcm":
        return LCMScheduler.from_config(base_config)
    if kind == "tcd":
        return TCDScheduler.from_config(base_config)
    if kind == "ddim_trailing":
        return DDIMScheduler.from_config(base_config, timestep_spacing="trailing")
    raise ValueError(kind)


def build_pipe():
    from diffusers import ControlNetModel

    pipeline_class, draw_kps = _import_instantid()
    identitynet = ControlNetModel.from_pretrained(INSTANT_REPO, subfolder="ControlNetModel", torch_dtype=DTYPE)
    try:
        pipe = pipeline_class.from_pretrained(BASE_REPO, controlnet=identitynet, torch_dtype=DTYPE)
    except Exception:
        pipe = pipeline_class.from_pretrained(BASE_REPO, controlnet=identitynet, torch_dtype=DTYPE)
    ip_bin = hf_hub_download(INSTANT_REPO, "ip-adapter.bin")
    pipe.load_ip_adapter_instantid(ip_bin)
    pipe.to(DEVICE)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe, draw_kps


def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(Path.home() / "Datasets" / "Kelsie" / "kelsie_0073.png"))
    args = ap.parse_args()

    ref_path = os.path.expanduser(args.ref)
    if not os.path.exists(ref_path):
        # fall back to the ip-adapter spike reference
        alt = str(Path.home() / ".sdxl-ipadapter-spike" / "ref_kelsie.png")
        ref_path = alt if os.path.exists(alt) else ref_path
    print(f"reference: {ref_path}")
    print(f"diffusers on {DEVICE} dtype={DTYPE}")

    reference = Image.open(ref_path)
    canvas = letterbox(reference, W, H)
    ref_face = largest_face(canvas)
    ref_emb = ref_face["embedding"]
    ref_norm = ref_face.normed_embedding

    pipe, draw_kps = build_pipe()
    base_config = dict(pipe.scheduler.config)
    face_kps = draw_kps(canvas, ref_face["kps"])  # identity-only control image, output aspect
    pipe.set_ip_adapter_scale(0.8)

    results = []
    fused_for = None  # which lora is currently fused
    for cfg in CONFIGS:
        # Rebuild the pipe whenever we need a different LoRA fusion (avoids fp16/bf16 unfuse residue).
        if cfg["lora"] != fused_for:
            del pipe
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            pipe, draw_kps = build_pipe()
            pipe.set_ip_adapter_scale(0.8)
            if cfg["lora"] is not None:
                repo, fname = cfg["lora"]
                path = hf_hub_download(repo, fname)
                pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path))
                pipe.fuse_lora()
                pipe.unload_lora_weights()
            fused_for = cfg["lora"]

        pipe.scheduler = make_scheduler(cfg["sched"], base_config)
        gen = torch.Generator("cpu").manual_seed(SEED)
        call = dict(
            prompt=PROMPT, negative_prompt=NEG, image_embeds=ref_emb, image=face_kps,
            controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8,
            width=W, height=H, guidance_scale=cfg["cfg"], generator=gen,
            num_inference_steps=cfg["steps"],
        )
        if cfg.get("eta") is not None:
            call["eta"] = cfg["eta"]
        if DEVICE == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        img = pipe(**call).images[0].convert("RGB")
        if DEVICE == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        png = os.path.join(OUT_DIR, f"instantid__{cfg['tag']}.png")
        img.save(png)
        try:
            gen_face = largest_face(img)
            cos = cosine(ref_norm, gen_face.normed_embedding)
            face_ok = True
        except Exception as e:  # noqa: BLE001
            cos = float("nan")
            face_ok = False
            print(f"  (no face in {cfg['tag']}: {e})")
        rec = dict(tag=cfg["tag"], method=cfg["method"], sched=cfg["sched"], steps=cfg["steps"],
                   cfg=cfg["cfg"], seconds=round(dt, 2), s_per_step=round(dt / cfg["steps"], 3),
                   arcface_cosine=round(cos, 4) if face_ok else None, face_detected=face_ok, png=png)
        results.append(rec)
        print(f"  {cfg['tag']:14s} {cfg['steps']:>2}st cfg{cfg['cfg']:<3g} {cfg['sched']:14s} "
              f"{dt:6.2f}s  ArcFace={rec['arcface_cosine']}")
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({"reference": ref_path, "results": results}, f, indent=2)

    # montage with the reference + scored renders
    thumb = 360
    cells = [("reference", canvas)] + [(r["tag"], Image.open(r["png"])) for r in results]
    cols = len(cells)
    bar = 50
    montage = Image.new("RGB", (cols * thumb, thumb + bar), (12, 12, 12))
    d = ImageDraw.Draw(montage)
    by_tag = {r["tag"]: r for r in results}
    for i, (tag, im) in enumerate(cells):
        montage.paste(im.convert("RGB").resize((thumb, thumb)), (i * thumb, bar))
        if tag == "reference":
            label = "REFERENCE"
        else:
            r = by_tag[tag]
            label = f"{tag}\n{r['steps']}st {r['seconds']}s  ArcFace {r['arcface_cosine']}"
        for j, line in enumerate(label.split("\n")):
            d.text((i * thumb + 6, 4 + j * 22), line, fill=(240, 240, 240), font=_font(16))
    mout = os.path.join(OUT_DIR, "grid__instantid_accel.png")
    montage.save(mout)
    print(f"\nmontage → {mout}")
    print("\nSUMMARY (ArcFace cosine vs reference):")
    for r in results:
        print(f"  {r['tag']:14s} steps={r['steps']:>2}  ArcFace={r['arcface_cosine']}  {r['seconds']}s")


if __name__ == "__main__":
    main()
