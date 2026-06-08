#!/usr/bin/env python3
"""sc-3487 spike — parity report: Rust `ort` detector vs Python rtmlib.

Compares the raw 133 COCO-WholeBody keypoints + SimCC scores produced by the Rust
port (scripts/spikes/sc3487_ort_pose) against the Python reference
(sc3487_reference.py), per image and per person, reporting keypoint pixel error and
score deltas. Two comparisons matter:

  A) algorithm-port parity:  rust.cpu   vs python.cpu    (same ORT numerics ->
     any delta is the Rust pre/post port)
  B) EP equivalence:         rust.coreml vs rust.cpu     (CoreML vs CPU)

USAGE:
  python3 scripts/spikes/sc3487_compare.py --ref /tmp/sc3487/ref --rust /tmp/sc3487/rust \
      --a-device cpu --b-device coreml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(d: Path, stem: str, dev: str):
    p = d / f"{stem}.{dev}.json"
    return json.loads(p.read_text()) if p.exists() else None


def kp_errors(ka, kb):
    """Euclidean pixel errors between two (N,2) keypoint lists."""
    errs = []
    for (ax, ay), (bx, by) in zip(ka, kb):
        errs.append(math.hypot(ax - bx, ay - by))
    return errs


def summarize(errs):
    errs = sorted(errs)
    n = len(errs)
    mean = sum(errs) / n
    p95 = errs[min(n - 1, int(0.95 * n))]
    return mean, p95, errs[-1]


def compare(name, a, b, diag):
    print(f"\n=== {name} ===")
    if a is None or b is None:
        print("   MISSING one side"); return
    na, nb = len(a["persons"]), len(b["persons"])
    if na != nb:
        print(f"   person count differs: A={na} B={nb}")
    # bbox deltas
    for i, (ba, bb) in enumerate(zip(a["bboxes"], b["bboxes"])):
        d = max(abs(x - y) for x, y in zip(ba, bb))
        print(f"   bbox[{i}] max|Δ|={d:.2f}px  A={[round(v,1) for v in ba]} B={[round(v,1) for v in bb]}")
    for i in range(min(na, nb)):
        pa, pb = a["persons"][i], b["persons"][i]
        errs = kp_errors(pa["keypoints"], pb["keypoints"])
        mean, p95, mx = summarize(errs)
        sderr = [abs(x - y) for x, y in zip(pa["scores"], pb["scores"])]
        smean = sum(sderr) / len(sderr)
        smax = max(sderr)
        print(
            f"   person[{i}] kp px-err mean={mean:.3f} p95={p95:.3f} max={mx:.3f} "
            f"(diag={diag:.0f}px -> mean={100*mean/diag:.3f}% of diag) | "
            f"score |Δ| mean={smean:.4f} max={smax:.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/sc3487/ref")
    ap.add_argument("--rust", default="/tmp/sc3487/rust")
    ap.add_argument("--a-device", default="cpu")
    ap.add_argument("--b-device", default="coreml")
    args = ap.parse_args()

    refd, rustd = Path(args.ref), Path(args.rust)
    stems = sorted({p.name.split(".")[0] for p in rustd.glob("*.json")})

    print("########## A) algorithm-port parity: rust.cpu vs python.cpu ##########")
    for stem in stems:
        rust_cpu = load(rustd, stem, "cpu")
        py_cpu = load(refd, stem, "cpu")
        diag = math.hypot(rust_cpu["width"], rust_cpu["height"]) if rust_cpu else 0
        compare(f"{stem}: rust.cpu vs python.cpu", py_cpu, rust_cpu, diag)

    print("\n\n########## B) EP equivalence: rust.coreml vs rust.cpu ##########")
    for stem in stems:
        rust_cpu = load(rustd, stem, "cpu")
        rust_cm = load(rustd, stem, args.b_device)
        diag = math.hypot(rust_cpu["width"], rust_cpu["height"]) if rust_cpu else 0
        compare(f"{stem}: rust.{args.b_device} vs rust.cpu", rust_cpu, rust_cm, diag)

    print("\n\n########## C) cross-check: python.coreml vs python.cpu ##########")
    for stem in stems:
        py_cpu = load(refd, stem, "cpu")
        py_cm = load(refd, stem, "mps")
        diag = math.hypot(py_cpu["width"], py_cpu["height"]) if py_cpu else 0
        compare(f"{stem}: python.coreml vs python.cpu", py_cpu, py_cm, diag)


if __name__ == "__main__":
    main()
