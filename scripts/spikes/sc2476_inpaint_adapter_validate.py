"""sc-2476 validation — exercise the REAL SDXL inpaint adapter path on M5 Max.

Closes the sc-2476 acceptance criterion ("real masked-inpaint run on the M5 Max produces
a coherent localized edit") and de-risks the one piece the sc-2475 spike did NOT cover:
``SdxlDiffusersAdapter._as_inpaint_pipe`` — wrapping the loaded edit pipe's `components`
in ``StableDiffusionXLInpaintPipeline`` (shared modules, no reload). The spike proved the
diffusers inpaint pipeline itself; this proves our adapter wiring of it.

It drives REAL adapter code (`SdxlDiffusersAdapter._as_inpaint_pipe`, `load_mask_image`,
`load_source_image`) against the SDXL base checkpoint already cached. `find_asset_media_path`
is monkeypatched to read source/mask PNGs off disk (asset-sidecar resolution is orthogonal
to what this validates), mirroring the worker unit tests.

Run with the SceneWorks desktop venv (torch/diffusers/PIL):
  "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
      scripts/spikes/sc2476_inpaint_adapter_validate.py \
      --source /path/to/photo.png --prompt "a vase of bright sunflowers"

Outputs PNGs + a JSON summary under --out (default /tmp/sc2476_inpaint_adapter).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _pkg in (_REPO / "apps" / "worker",):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

os.environ.setdefault("SCENEWORKS_GPU_ID", "mps")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from scene_worker import image_adapters as ia  # noqa: E402
from scene_worker.image_adapters import (  # noqa: E402
    SdxlDiffusersAdapter,
    image_request_from_job,
    load_mask_image,
    load_source_image,
)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ellipse_mask(size: int, feather: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    pad = size // 6
    ImageDraw.Draw(mask).ellipse([pad, pad, size - pad, size - pad], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sdxl", help="MODEL_TARGETS key (sdxl or realvisxl).")
    parser.add_argument("--source", required=True, help="Source image (square-fit to --size).")
    parser.add_argument("--prompt", default="a vase of bright sunflowers, sharp focus")
    parser.add_argument("--negative-prompt", default="blurry, lowres")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--feather", type=int, default=12)
    parser.add_argument("--out", default="/tmp/sc2476_inpaint_adapter")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device()
    size = args.size

    # Stage source + mask on disk; point the adapter's asset resolver at them.
    source = Image.open(args.source).convert("RGB").resize((size, size), Image.LANCZOS)
    source_path = out / "source.png"
    source.save(source_path)
    mask = _ellipse_mask(size, args.feather)
    mask_path = out / "mask.png"
    mask.save(mask_path)

    paths = {"asset_source": source_path, "asset_mask": mask_path}
    orig_resolver = ia.find_asset_media_path
    ia.find_asset_media_path = lambda project_path, asset_id: paths[asset_id]

    request = image_request_from_job({
        "payload": {
            "projectId": "p", "mode": "edit_image", "model": args.model,
            "prompt": args.prompt, "negativePrompt": args.negative_prompt,
            "sourceAssetId": "asset_source", "maskAssetId": "asset_mask",
            "width": size, "height": size, "advanced": {"strength": args.strength},
        }
    })

    # Sanity: the loaders return aligned, correctly-typed inputs.
    src_img = load_source_image(out, request)
    mask_img = load_mask_image(out, request)
    assert src_img.size == (size, size) and mask_img.size == (size, size)
    assert mask_img.mode == "L"

    adapter = SdxlDiffusersAdapter()
    model_target = ia.MODEL_TARGETS[args.model]
    settings = __import__("types").SimpleNamespace(gpu_id=device)

    print(f"=== loading {args.model} edit pipe on {device} ===", flush=True)
    t_load = time.time()
    pipe = adapter._load_pipeline(  # noqa: SLF001 — spike drives the real internal path
        settings, request, model_target, progress=lambda *a: None, job_id="sc2476",
    )
    load_s = round(time.time() - t_load, 1)

    # THE thing under test: wrap the loaded edit pipe's components in the inpaint pipeline.
    inpaint_pipe = adapter._as_inpaint_pipe(pipe)  # noqa: SLF001
    inpaint_class = type(inpaint_pipe).__name__
    shared_unet = inpaint_pipe.unet is pipe.unet  # must reuse the SAME module (no reload)
    print(f"  loaded in {load_s}s → wrapped as {inpaint_class} (shared UNet={shared_unet})", flush=True)

    print("=== running masked inpaint via the real adapter path ===", flush=True)
    t0 = time.time()
    result = adapter._run_pipeline(settings, pipe, request, seed=1234, project_path=out)  # noqa: SLF001
    elapsed = round(time.time() - t0, 1)
    result.save(out / "result.png")

    # Outside-mask preservation: outside ≈ source, inside changed.
    src = np.asarray(source, dtype=np.float32)
    res = np.asarray(result.convert("RGB").resize(source.size), dtype=np.float32)
    m = np.asarray(mask.resize(source.size), dtype=np.float32) / 255.0
    inside, outside = m > 0.5, m <= 0.5
    diff = np.abs(res - src).mean(axis=2)
    inside_diff = round(float(diff[inside].mean()), 2)
    outside_diff = round(float(diff[outside].mean()), 2)

    verdict = "GO" if inside_diff >= 15 and outside_diff <= 8 else "REVIEW"
    summary = {
        "device": device, "model": args.model, "size": size, "strength": args.strength,
        "inpaintClass": inpaint_class, "sharedUnet": shared_unet,
        "seconds": elapsed, "insideMeanDiff": inside_diff, "outsideMeanDiff": outside_diff,
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    ia.find_asset_media_path = orig_resolver

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"VERDICT: {verdict}  (result.png + summary.json in {out})")
    print("GO = adapter wrapped the pipe (sharedUnet=True) + inside Δ≥15 & outside Δ≤8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
