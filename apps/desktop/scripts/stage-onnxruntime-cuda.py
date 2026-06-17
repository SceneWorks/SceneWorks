#!/usr/bin/env python3
"""Stage the CUDA-enabled onnxruntime + its CUDA-12 deps for the candle worker's `ort`
paths (sc-5496, epic 5482) — the Windows analogue of `stage-onnxruntime.py`'s CoreML
dylib staging on macOS.

The Rust worker links `ort` with `load-dynamic`, so it dlopens onnxruntime at runtime
from `ORT_DYLIB_PATH` (set by the desktop `setup.rs`). For a packaged, CUDA-Toolkit-free
machine we must bundle a CUDA-enabled onnxruntime AND the exact CUDA-12 runtime + cuDNN-9
DLLs its `onnxruntime_providers_cuda.dll` depends on (a `torch` import normally arranges
these on `PATH`; the Python stack is retired off-Mac). The cleanest version-matched source
is the PyPI `onnxruntime-gpu` wheel + its `nvidia-*-cu12` dependency wheels: `pip download`
resolves the precise cuDNN/cuFFT/cuBLAS/cudart set the build was tested against.

We stage onnxruntime's three DLLs into the `onnxruntime` resource dir, and the deps the
toolkit redist list doesn't cover — cuDNN + cuFFT + nvJitLink + nvRTC — into the `cuda`
resource dir (which build-sidecar.mjs also fills with the CUDA-12 runtime redist —
cudart/cublas/cublasLt/curand — from the toolkit, shared with cudarc). `setup.rs` then
points `ORT_DYLIB_PATH` at the staged onnxruntime.dll and `SCENEWORKS_ORT_CUDA_DIR`/
`SCENEWORKS_ORT_CUDNN_DIR` at the `cuda` dir; `ort_cuda::preload_cuda_dylibs` preloads
them and puts the dir on the loader search path so cuDNN's lazily-loaded sub-engine DLLs
(`cudnn_engines_tensor_ir64_9.dll` et al.) resolve at inference time.

Validated on RTX PRO 6000 against onnxruntime-gpu 1.26.0 + cuDNN-cu12 9.23 (DWPose CUDA EP
engaged end-to-end). Bump in lockstep with the `ort` crate's ORT API requirement (rc.12 ⇒
API 24 ⇒ onnxruntime >= 1.24). Invoked by build-sidecar.mjs on the Windows candle build.

USAGE: python stage-onnxruntime-cuda.py <onnxruntime-dir> <cuda-dir>
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import zipfile

# ORT 2.0.0-rc.12 requests ONNX Runtime API 24 (>= onnxruntime 1.24); 1.26.0 is the
# version validated in the sc-5496 GPU run. Bump in lockstep with the `ort` crate.
ONNXRUNTIME_GPU_VERSION = "1.26.0"

# onnxruntime's own DLLs (the CUDA execution provider is in providers_cuda; providers_shared
# is the EP bridge). TensorRT is deliberately omitted — the worker only uses the CUDA EP.
ORT_DLLS = (
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
)

# The CUDA-12 deps onnxruntime's CUDA provider + cuDNN need that the toolkit redist list
# (build-sidecar.mjs: cudart/cublas/cublasLt/curand/nvrtc) doesn't guarantee: cuDNN-9 (incl.
# the lazily-loaded compute-engine sub-DLLs), cuFFT (a hard import of providers_cuda),
# nvJitLink + nvRTC (cuDNN's runtime-compiled engines JIT through them). cudart/cublas/
# cublasLt come from the toolkit copy (CUDA-12, shared with cudarc); not re-staged here.
#
# IMPORTANT: onnxruntime-gpu does NOT declare the nvidia-*-cu12 wheels as hard dependencies,
# so `pip download onnxruntime-gpu` alone does NOT pull them — they must be requested
# explicitly (the cu12 line, matching onnxruntime-gpu's CUDA-12 build). Unpinned ⇒ the
# latest cu12 release, which is what the sc-5496 GPU run validated (cuDNN 9.23 / cuFFT 11.4).
CUDA_DEP_PACKAGES = (
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-cuda-nvrtc-cu12",
)
# Wheel filename prefixes (underscored) the packages above resolve to, for harvesting.
CUDA_DEP_WHEEL_PREFIXES = tuple(p.replace("-", "_") + "-" for p in CUDA_DEP_PACKAGES)


def _extract_dlls(wheel: str, names: list[str] | None, dest: str) -> int:
    """Extract DLLs from a wheel (zip) into dest by basename. names=None → all *.dll."""
    os.makedirs(dest, exist_ok=True)
    count = 0
    with zipfile.ZipFile(wheel) as zf:
        for entry in zf.namelist():
            base = os.path.basename(entry)
            if not base.lower().endswith(".dll"):
                continue
            if names is not None and base not in names:
                continue
            with zf.open(entry) as src, open(os.path.join(dest, base), "wb") as out:
                out.write(src.read())
            count += 1
    return count


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: stage-onnxruntime-cuda.py <onnxruntime-dir> <cuda-dir>", file=sys.stderr)
        return 2
    ort_dir, cuda_dir = sys.argv[1], sys.argv[2]

    with tempfile.TemporaryDirectory() as tmp:
        # Resolve onnxruntime-gpu + the cu12 deps it needs at runtime (it doesn't declare
        # them, so they're listed explicitly) — CUDA-12 wheels for this Windows host.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                f"onnxruntime-gpu=={ONNXRUNTIME_GPU_VERSION}",
                *CUDA_DEP_PACKAGES,
                "--only-binary=:all:", "-d", tmp,
            ],
            check=True,
        )

        def _one(pattern: str) -> str:
            hits = glob.glob(os.path.join(tmp, pattern))
            if not hits:
                print(f"stage-onnxruntime-cuda: no wheel matching {pattern}", file=sys.stderr)
                raise SystemExit(1)
            return hits[0]

        n = _extract_dlls(_one("onnxruntime_gpu-*.whl"), list(ORT_DLLS), ort_dir)
        if n < len(ORT_DLLS):
            print(f"stage-onnxruntime-cuda: only {n}/{len(ORT_DLLS)} onnxruntime DLLs found", file=sys.stderr)
            return 1
        print(f"stage-onnxruntime-cuda: staged {n} onnxruntime DLLs into {ort_dir}")

        # cuDNN-9 (incl. lazily-loaded sub-engines) + cuFFT + nvJitLink + nvRTC into the
        # shared cuda dir, from the matched nvidia-*-cu12 dep wheels pip resolved.
        for prefix in CUDA_DEP_WHEEL_PREFIXES:
            wheel = _one(f"{prefix}*.whl")
            count = _extract_dlls(wheel, None, cuda_dir)
            print(f"stage-onnxruntime-cuda: staged {count} DLLs from {os.path.basename(wheel)} into {cuda_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
