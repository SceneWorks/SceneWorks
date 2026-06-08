#!/usr/bin/env python3
"""Stage the CoreML-enabled onnxruntime dylib for the Rust DWPose detector (sc-3487).

The Rust worker links `ort` with `load-dynamic`, so it dlopens onnxruntime at
runtime from `ORT_DYLIB_PATH` (set by the desktop `setup.rs`). For a packaged,
Python-free Mac we must bundle that dylib. The macOS arm64 onnxruntime PyPI wheel
ships exactly the CoreML-enabled `libonnxruntime.<ver>.dylib` the Python rtmlib path
uses (and that ort 2.0.0-rc.12's ORT API 24 accepts), so we pip-download that wheel
(a zip) and extract its `capi/libonnxruntime*.dylib`.

Pinned to a known-compatible onnxruntime so the bundle is deterministic. Invoked by
build-sidecar.mjs on macOS only.

USAGE: python3 stage-onnxruntime.py <dest-dylib-path>
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import zipfile

# ORT 2.0.0-rc.12 requests ONNX Runtime API 24 (>= onnxruntime 1.24); 1.26.0 is the
# version validated in the sc-3487 spike. Bump in lockstep with the `ort` crate.
ONNXRUNTIME_VERSION = "1.26.0"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage-onnxruntime.py <dest-dylib-path>", file=sys.stderr)
        return 2
    dest = sys.argv[1]

    with tempfile.TemporaryDirectory() as tmp:
        # Let pip pick the wheel matching this (macOS arm64) build host.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                f"onnxruntime=={ONNXRUNTIME_VERSION}",
                "--only-binary=:all:", "--no-deps", "-d", tmp,
            ],
            check=True,
        )
        wheels = glob.glob(os.path.join(tmp, "onnxruntime-*.whl"))
        if not wheels:
            print("stage-onnxruntime: no onnxruntime wheel downloaded", file=sys.stderr)
            return 1
        with zipfile.ZipFile(wheels[0]) as zf:
            dylibs = [n for n in zf.namelist() if n.endswith(".dylib") and "libonnxruntime" in n]
            if not dylibs:
                print("stage-onnxruntime: wheel has no libonnxruntime dylib", file=sys.stderr)
                return 1
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(dylibs[0]) as src, open(dest, "wb") as out:
                out.write(src.read())
        os.chmod(dest, 0o755)
    print(f"stage-onnxruntime: staged {dest} (onnxruntime {ONNXRUNTIME_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
