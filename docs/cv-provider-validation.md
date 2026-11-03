# Native CV ownership and provider proof

SW-6775 / SW-6680 move the shipped CV engines to inference's
`crates/media/gen-cv` (`sceneworks-gen-cv`), reexported by the existing
`runtime-macos::cv` and `runtime-cuda::cv` through their explicit `cv` feature.
The worker enables that feature on both platforms. The extracted crate retains
SceneWorks' AGPL-3.0-or-later terms and source provenance in its LICENSE/NOTICE;
default runtime media consumers do not acquire the CV dependency. SceneWorks owns model
resolution, JSON/render/job behavior, cancellation between jobs/images, and the
existing process caches. The engine owns YOLO11m's entire MLX network and shared
letterbox/decode/NMS, DWPose's YOLOX/RTMW execution and numerical pre/postprocessing,
and Real-ESRGAN x2/x4 execution and padded tiling.

Every CV ORT load calls the same inference-owned session builder. Its required
`SessionLoader` uses `cv::ModelIoLoader` here: platform dependency preparation
remains in `ort_cuda`, and final session loading always calls
`model_io::ort_session`. External initializer validation and byte-level progress
remain on that path. Engine errors retain worker classifications, including
caller-owned load failures. CoreML registration, CUDA's error-on-failure policy,
CPU retry, paired DWPose reload, device labels, and cache keys are preserved.

## Ordinary validation

Inference's existing CPU provider and macOS lanes run the CV numerical fixtures
and compile/lint provider code. These tests never load a model or initialize ORT.
They cover both YOLO input layouts, decode/NMS, DWPose SimCC/OpenPose conversion,
Real-ESRGAN tiling/CHW/padding/output validation, and grouped EP fallback using
plain tokens. SceneWorks compiles its existing ignored provider proofs and runs
worker cache, resolver, cancellation and error-classification fixtures.

## Explicit provider validation

Dispatch `.github/workflows/cv-provider-validation.yml` twice against the clean,
stable paired source: once with `backend=mlx`, once with `backend=cuda`. Inputs:

- `source_sha`: full SceneWorks commit, identical to the dispatch ref.
- `inference_sha`: full reachable inference commit pinned at every consumer site.
- `fixture_dir`: existing absolute path to a **dedicated** directory containing
  the files below and all referenced ONNX external-data companions. No symlinks.
- `ort_dylib`: absolute existing ONNX Runtime 1.26 library with CoreML on Mac or
  CUDA on Windows. Runtime provisioning remains the standard desktop/runner setup.
- `build_only=false`: enables the selected ignored bodies; `true` compiles and
  checks registration only, and explicitly cannot produce a runtime PASS.
- `mac_go=true`: only for an authorized developer-Mac execution dispatch.

| Fixture | Source / purpose |
| --- | --- |
| `people.jpg` | Existing 810x1080 person photo from the sc-3734 oracle fixture set |
| `yolo11m_fused_mlx.safetensors` (Mac) | `SceneWorks/yolo11m-person-detect-mlx` at `d7027d3a8812bdebbf7862fc1c7dcfdeebb0f777` |
| `refs.safetensors`, `refs_ext.safetensors`, `ref_people.json` (Mac) | Existing per-block and Ultralytics oracle outputs from `docs/sc-3734/torch_ref.py` and accompanying fixture tooling; stage matching artifacts before dispatch |
| `yolo11m.onnx` (Windows) | `SceneWorks/yolo11m-person-detect-onnx` at `3cffbaccca4f239ae6301d8b66ba721401ecfa8d` |
| `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` | `SceneWorks/dwpose-onnx` at `c330a9609201343576b8b69d222e38c1486d2854` |
| `rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx` | Same DWPose snapshot |
| `real_esrgan_x2.onnx`, `real_esrgan_x4.onnx` | `SceneWorks/real-esrgan-onnx` at `09f741bac80a246b407da3ee902bf5f3291b602f` |

The controller downloads nothing. It records every fixture file's SHA-256,
including external model data, the runtime library, executable, lockfile, source
and pin, host/device identity, exact case/settings, duration, complete-body
receipts and outputs. Missing assets, missing/ignored bodies, failed assertions,
or fallback to CPU fail the selected hardware proof. Each ORT graph must also
record nonzero executions on its required provider in the retained, proof-only
ORT profile; a configured device label alone is insufficient. Raw DWPose outputs and
normalized results are retained; both Real-ESRGAN factors save PNG outputs from
an odd 513x33 image crossing a 512-pixel tile boundary. Mac YOLO checks the retained
whole-network, per-block, and photo oracle assertions; its attention retains the
existing explicitly selected CPU stream while other MLX work uses the GPU.

Mac work enters the shared Metal execution lease. Windows requests a real-weight
runner and uses the existing execution controller with physical GPU1 exposed as
logical device 0. Distinct dispatches preserve queueing and never cancel running
work. This is one bounded correctness pass per provider, with no memory matrix or
calibration campaign. Ordinary PR/main workflows never invoke it.

A passing compile, fixture test, or `build_only` run is not native provider proof.
Keep both actual dispatch artifacts with the acceptance report before closing
SW-6775/SW-6680.
