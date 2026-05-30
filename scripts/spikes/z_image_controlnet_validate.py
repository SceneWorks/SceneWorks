#!/usr/bin/env python3
"""sc-2257 — real-weights validation of the Z-Image strict pose ControlNet port.

Runs in the mflux SIDECAR venv (the one with the ported mflux fork +
Fun-Controlnet-Union weights). Renders a bundled COCO-18 pose skeleton, loads the
ported ``ZImageControl`` (base Z-Image-Turbo + Fun-Controlnet-Union-2.1-8steps),
and generates two images at a shared seed/prompt:

  - ``control_context_scale = 0.0`` — must reproduce base Z-Image (parity gate;
    the control hints are multiplied by 0).
  - ``control_context_scale = <scale>`` — pose-locked output that should follow
    the rendered skeleton.

USAGE (on the Mac, sidecar venv):
    python scripts/spikes/z_image_controlnet_validate.py \
        --cn /path/to/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
        --poses apps/web/public/poses/index.json \
        --category tpose --steps 8 --scale 1.0 --out /tmp/sc2257_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load_pose_keypoints(poses_index: Path, category: str) -> tuple[str, list]:
    data = json.loads(poses_index.read_text())
    poses = data.get("poses", [])
    for p in poses:
        if p.get("category") == category:
            return p.get("id"), p.get("keypoints")
    # fall back to the first pose
    return poses[0].get("id"), poses[0].get("keypoints")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True, help="Fun-Controlnet-Union safetensors path")
    ap.add_argument("--poses", default="apps/web/public/poses/index.json")
    ap.add_argument("--category", default="tpose")
    ap.add_argument("--prompt", default="a full body studio photograph of a person, plain grey background, sharp focus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/sc2257_out")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np
    from PIL import Image

    # SceneWorks skeleton renderer (worker module on sys.path via repo root)
    sys.path.insert(0, str(Path("apps/worker").resolve()))
    from scene_worker.openpose_skeleton import draw_bodypose, normalize_keypoints

    from mflux.models.z_image.variants.z_image_control import ZImageControl

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pose_id, raw_kp = _load_pose_keypoints(Path(args.poses), args.category)
    keypoints = normalize_keypoints(raw_kp)
    skel = draw_bodypose(args.size, args.size, keypoints)
    skel_path = out / f"skeleton_{pose_id}.png"
    Image.fromarray(skel).save(skel_path)
    print(f"[validate] pose={pose_id} skeleton -> {skel_path}", flush=True)

    t0 = time.time()
    model = ZImageControl(control_weights_path=args.cn)
    print(f"[validate] ZImageControl loaded in {time.time() - t0:.1f}s (bits={model.bits})", flush=True)

    common = dict(
        seed=args.seed,
        prompt=args.prompt,
        control_image_path=str(skel_path),
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
    )

    t1 = time.time()
    img0 = model.generate_image(control_context_scale=0.0, **common)
    p0 = out / f"scale0_{pose_id}.png"
    img0.save(p0)
    print(f"[validate] scale=0.0 -> {p0}  ({time.time() - t1:.1f}s)", flush=True)

    t2 = time.time()
    img1 = model.generate_image(control_context_scale=args.scale, **common)
    p1 = out / f"scale{args.scale}_{pose_id}.png"
    img1.save(p1)
    print(f"[validate] scale={args.scale} -> {p1}  ({time.time() - t2:.1f}s)", flush=True)

    # mflux generate_image returns a GeneratedImage wrapper; .image is the PIL image
    a0 = np.asarray(getattr(img0, "image", img0), dtype=np.float32)
    a1 = np.asarray(getattr(img1, "image", img1), dtype=np.float32)
    diff = float(np.abs(a0 - a1).mean())
    print(f"[validate] mean|scale0 - scale{args.scale}| = {diff:.3f} (0-255 scale); "
          f"expect >0 (control steers). Inspect {p1} vs skeleton for pose lock.", flush=True)
    print("[validate] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
