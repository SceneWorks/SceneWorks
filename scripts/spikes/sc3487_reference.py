#!/usr/bin/env python3
"""sc-3487 spike — Python reference dump for the Rust DWPose port.

Runs the SHIPPED detector stack (rtmlib Wholebody, performance mode = yolox_m +
rtmw-dw-x-l, COCO-WholeBody-133, ``to_openpose=False`` exactly like
``pose_adapters.load_pose_detector``) and dumps the RAW intermediate + final
tensors so the Rust ``ort`` port can be validated for numerical parity:

  - yolox person bboxes (xyxy, original-image px)
  - per person: 133 keypoints (x,y px) + 133 scores  (pre-OpenPose-conversion)
  - timings + the onnxruntime provider actually used

Dumped per provider so we can separate (a) algorithm-port parity (Rust-CPU vs
Python-CPU — same ORT numerics, so any delta is the pre/post port) from (b) EP
equivalence (CoreML vs CPU).

USAGE (dwpose-spike venv):
  ~/.dwpose-spike/venv/bin/python scripts/spikes/sc3487_reference.py \
      --images "/tmp/sc3487/sources/*.png" --out /tmp/sc3487/ref --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# The exact weights pose_adapters pins via SCENEWORKS_DWPOSE_DET/POSE; default to
# the rtmlib performance-preset cache so this matches production.
CACHE = Path.home() / ".cache/rtmlib/hub/checkpoints"
DET = os.environ.get(
    "SCENEWORKS_DWPOSE_DET", str(CACHE / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx")
)
POSE = os.environ.get(
    "SCENEWORKS_DWPOSE_POSE",
    str(CACHE / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="/tmp/sc3487/ref")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from rtmlib import Wholebody

    t0 = time.time()
    model = Wholebody(
        det=DET,
        det_input_size=(640, 640),
        pose=POSE,
        pose_input_size=(288, 384),
        to_openpose=False,
        backend="onnxruntime",
        device=args.device,
    )
    # what provider did ORT actually bind?
    det_prov = model.det_model.session.get_providers()
    pose_prov = model.pose_model.session.get_providers()
    print(f"[ref] load {time.time()-t0:.1f}s det={det_prov} pose={pose_prov}", flush=True)

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"[ref] no images match {args.images}")
        return 1

    records = []
    for path in paths:
        img = cv2.imread(path)  # BGR uint8 (what rtmlib consumes)
        if img is None:
            print(f"[ref] SKIP unreadable {path}")
            continue
        h, w = img.shape[:2]

        t1 = time.time()
        bboxes = model.det_model(img)  # (N,4) xyxy original px
        det_ms = (time.time() - t1) * 1000.0

        t2 = time.time()
        keypoints, scores = model.pose_model(img, bboxes=bboxes)  # (N,133,2),(N,133)
        pose_ms = (time.time() - t2) * 1000.0

        n = 0 if keypoints is None else len(keypoints)
        rec = {
            "source": Path(path).name,
            "width": w,
            "height": h,
            "device": args.device,
            "detProviders": det_prov,
            "poseProviders": pose_prov,
            "detMs": round(det_ms, 1),
            "poseMs": round(pose_ms, 1),
            "bboxes": np.asarray(bboxes, dtype=float).round(3).tolist(),
            "persons": [
                {
                    "keypoints": np.asarray(keypoints[i], dtype=float).round(4).tolist(),
                    "scores": np.asarray(scores[i], dtype=float).round(5).tolist(),
                }
                for i in range(n)
            ],
        }
        records.append(rec)
        (out / f"{Path(path).stem}.{args.device}.json").write_text(json.dumps(rec))
        print(
            f"[ref] {Path(path).name}: {n} person(s) det={det_ms:.0f}ms "
            f"pose={pose_ms:.0f}ms bbox0={rec['bboxes'][:1]}",
            flush=True,
        )

    (out / f"index.{args.device}.json").write_text(
        json.dumps([{"source": r["source"], "n": len(r["persons"])} for r in records])
    )
    print(f"[ref] DONE {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
