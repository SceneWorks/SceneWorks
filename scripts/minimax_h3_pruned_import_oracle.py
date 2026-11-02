#!/usr/bin/env python3
"""Regenerate the synthetic managed-import AdaLN oracle, offline and CPU-only.

The reference directory contains WanGP's pinned textual pruning/transformer sources;
no checkpoint or downloaded table is used. Inference does not depend on Python.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REVISION = "47b06ff920c16b4b7b36de068367a5de0e0968e9"
SOURCES = {
    "prune_checkpoint.py": (
        "91b0c55a1fa7dcff0acaad5d89df68a40cc54545f14c1a9d37342f703027d1f1",
        {"_interpolated_table"},
    ),
    "transformer.py": (
        "5f0b81959522333830c7f3d422c2b0138b5f4fd37287a98ebb1705fd04f0ca50",
        {"AdalnProj"},
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    namespace = {"torch": torch, "nn": nn, "F": F}
    for name, (expected_hash, definitions) in SOURCES.items():
        raw = (args.reference_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"pinned reference hash mismatch: {name}")
        tree = ast.parse(raw, filename=name)
        chosen = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions
        ]
        if {node.name for node in chosen} != definitions:
            raise ValueError(f"missing pinned definitions: {name}")
        exec(compile(ast.Module(body=chosen, type_ignores=[]), name, "exec"), namespace)
    rows = torch.arange(1025, dtype=torch.float32)[:, None]
    columns = torch.arange(1, 9, dtype=torch.float32)[None, :]
    table = ((rows / 1024) - .5) * columns / 32
    # LCM(7, 5) = 35 covers every weight/bias row pattern in the Rust source writer.
    weight = ((torch.arange(35)[:, None] % 7 + 1) * columns / 512).to(torch.bfloat16)
    bias = ((torch.arange(35) % 5 - 2).float() / 32).to(torch.float16)
    projection = namespace["AdalnProj"](
        8, 35, 1, modalities=1, apply_silu=False, dtype=torch.float32, device="cpu"
    )
    with torch.no_grad():
        projection.linear.weight.copy_(weight.float())
        projection.linear.bias.copy_(bias.float())
        timesteps = torch.tensor(
            [-.1, 0, .12548828125, .23, .7, .999, .9995, 1, 1.1], dtype=torch.float32
        )
        coordinates = namespace["_interpolated_table"](table, timesteps)
        outputs = projection(coordinates)[0]
        cases = [
            dict(timestep=float(timesteps[i]), coordinates=coordinates[i].tolist(),
                 projection_period=outputs[i].tolist())
            for i in range(len(timesteps))
        ]
    result = {
        "reference_revision": REVISION,
        "reference_sha256": {name: value[0] for name, value in SOURCES.items()},
        "torch": torch.__version__, "device": "cpu", "grid": 1025, "rank": 8,
        "source_dtypes": {"table": "F32", "weight": "BF16", "folded_bias": "F16"},
        "projection_compute": "F32; pinned curve _lock_dtype; lossless storage-rounded values",
        "coordinates_abs": 2e-6, "projection_abs": 2e-5, "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"generated {len(cases)} pinned CPU cases, rank8, projection row-period35; no source payload")


if __name__ == "__main__":
    main()
