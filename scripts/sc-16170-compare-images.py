#!/usr/bin/env python3
"""Compare two fixed-seed RGB outputs for SC-16170 calibration provenance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-threshold", type=float, default=0.15)
    parser.add_argument("--mean-threshold", type=float, default=0.002)
    args = parser.parse_args()

    reference = Image.open(args.reference).convert("RGB")
    candidate = Image.open(args.candidate).convert("RGB")
    if reference.size != candidate.size:
        raise SystemExit(f"image geometry differs: {reference.size} != {candidate.size}")
    deltas = [
        abs(left - right)
        for left, right in zip(reference.tobytes(), candidate.tobytes(), strict=True)
    ]
    maximum = max(deltas)
    mean = sum(deltas) / len(deltas)
    print(
        f"ZIMAGE_QUALITY max_rgb8={maximum} mean_rgb8={mean:.9f} "
        f"max_normalized={maximum / 255:.9f} mean_normalized={mean / 255:.9f}"
    )
    print(
        f"QUALITY_CONTRACT maximum_threshold={args.max_threshold:.9f} "
        f"mean_threshold={args.mean_threshold:.9f}"
    )
    for path in (args.reference, args.candidate):
        print(f"OUTPUT_SHA256 {path} {digest(path)}")
    if maximum / 255 > args.max_threshold or mean / 255 > args.mean_threshold:
        raise SystemExit(
            f"quality contract failed: {maximum / 255:.9f}/{mean / 255:.9f} "
            f"> {args.max_threshold}/{args.mean_threshold}"
        )
    print("RESULT passed")


if __name__ == "__main__":
    main()
