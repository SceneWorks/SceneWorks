# sc-5560 — Bundled CUDA runtime: provenance, requirements & licensing

Epic 5558 (candle on the Windows desktop). The native **candle (CUDA)** generation
backend — built into the desktop sidecar by [sc-5559](https://app.shortcut.com/trefry/story/5559)
when `SCENEWORKS_DESKTOP_CANDLE=1` — links `cudarc` with **dynamic-linking**, so it
`LoadLibrary`s the CUDA runtime libraries by name at runtime instead of static-linking
them. The desktop must therefore ship those libraries; the standalone candle worker
(sc-3676 / `package-cuda.ps1`) already does this, and this story brings the same set
into the Tauri installer.

## What we ship

Staged by `apps/desktop/scripts/build-sidecar.mjs` (Windows candle build only) into
the `apps/desktop/cuda/` Tauri resource dir (`tauri.conf.json` `resources` ->
`cuda/**/*`), copied from the CUDA Toolkit 12.9 `bin` dir (`$CUDA_PATH\bin`):

| DLL | Library |
| --- | --- |
| `cudart64_12.dll` | CUDA Runtime |
| `cublas64_12.dll` | cuBLAS |
| `cublasLt64_12.dll` | cuBLAS Lt |
| `curand64_10.dll` | cuRAND |
| `nvrtc64_120_0.dll` | NVRTC (runtime kernel compilation) |
| `nvrtc-builtins64_129.dll` | NVRTC built-ins |

The DLLs are matched by version-agnostic prefix (e.g. `cudart64_\d+.dll`) so a minor
CUDA point-release still resolves; the `.alt` NVRTC variant is excluded.

**Not bundled:** the CUDA *driver* API (`nvcuda.dll`). It ships with the user's NVIDIA
display driver, which also JIT-compiles the binary's `compute_80` PTX forward onto
newer architectures (through Blackwell / sm_120, per sc-3676).

## Runtime requirements (CUDA-only — product decision 2026-06-14)

- **An NVIDIA (CUDA-capable) GPU.** SceneWorks generation on Windows is CUDA-only:
  no CPU fallback (3B-14B models can't run on CPU) and no AMD/ROCm. A non-NVIDIA
  Windows machine is unsupported off-Mac once Python is dropped (sc-5563); the app
  must say so clearly (preflight gate — sc-5561) rather than silently fail.
- **NVIDIA display driver >= 576.02** — the minimum that supports the CUDA 12.9
  runtime and forward-JITs the bundled `compute_80` PTX (documented in sc-3676 /
  `package-cuda.ps1`).

## How the loader finds the DLLs

The DLLs land in a `cuda` resource sub-dir, not next to the sidecar `.exe`, so they
are not on the default Windows DLL search path. `apps/desktop/src/setup.rs`
(`resolve_bundled_cuda_dir` + `spawn_api`) prepends the resolved `cuda` resource dir
to the sidecar process's `PATH` on Windows, which the `LoadLibrary` search order
includes. The resolver returns `None` on a plain (non-candle) build — that build
ships only a placeholder `README.txt` in `cuda/`, so `PATH` is left untouched.

## Licensing

These are NVIDIA CUDA **redistributable** runtime libraries, redistributed under the
NVIDIA CUDA Toolkit EULA (which enumerates the redistributable runtime libraries in
its "Attachment A"): <https://docs.nvidia.com/cuda/eula/index.html>. The tracked
notice — `apps/desktop/licenses/cuda/NOTICE.txt` — is staged next to the DLLs and is
the single source of truth for the in-app **About -> Licenses** screen
(`apps/desktop/licenses/manifest.json` + `apps/web/src/data/bundledLicenses.js`,
sc-3778), the same pattern as the bundled ffmpeg/onnxruntime notices.
