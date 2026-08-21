#!/usr/bin/env python3
"""Provision one immutable public Hugging Face artifact for the epic-20738 harness.

This helper is intentionally narrow: it accepts a controller-written JSON request, downloads only
the reviewed allow-list at an exact commit with anonymous access, and reports the resolved requested
subdirectory. The Node controller owns sequencing, inventory hashing, receipts, and scratch cleanup.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SHA40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def parse_request(path: Path) -> dict:
    request = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "repository", "revision", "subdirectory", "allowPatterns", "destination"}
    if set(request) != required:
        raise ValueError(f"artifact request fields must be exactly {sorted(required)}")
    if not REPOSITORY.fullmatch(request["repository"]):
        raise ValueError("repository must be an exact owner/name identity")
    if not SHA40.fullmatch(request["revision"]):
        raise ValueError("revision must be an exact lowercase 40-hex commit")
    patterns = request["allowPatterns"]
    if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) and p for p in patterns):
        raise ValueError("allowPatterns must be a non-empty string array")
    if any(".." in Path(pattern).parts or Path(pattern).is_absolute() for pattern in patterns):
        raise ValueError("allowPatterns cannot escape the exact snapshot")
    subdirectory = Path(request["subdirectory"])
    if subdirectory.is_absolute() or ".." in subdirectory.parts:
        raise ValueError("subdirectory cannot escape the exact snapshot")
    return request


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = parse_request(args.request.resolve())
    destination = Path(request["destination"]).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing non-empty artifact destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    resolved = Path(
        snapshot_download(
            repo_id=request["repository"],
            repo_type="model",
            revision=request["revision"],
            allow_patterns=request["allowPatterns"],
            local_dir=destination,
            token=False,
            max_workers=4,
        )
    ).resolve()
    if resolved != destination:
        raise RuntimeError(f"snapshot_download resolved outside the requested destination: {resolved}")
    selected = (resolved / request["subdirectory"]).resolve()
    if selected != resolved and resolved not in selected.parents:
        raise RuntimeError("selected artifact subdirectory escaped the exact snapshot")
    if not selected.is_dir():
        raise RuntimeError(f"selected artifact subdirectory is missing: {selected}")
    if not any(path.is_file() for path in selected.rglob("*")):
        raise RuntimeError(f"selected artifact subdirectory is empty: {selected}")
    print(json.dumps({"id": request["id"], "snapshotRoot": str(resolved), "selectedRoot": str(selected)}))


if __name__ == "__main__":
    main()
