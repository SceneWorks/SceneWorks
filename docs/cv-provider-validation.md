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
stable paired source: once with `backend=mlx`, once with `backend=cuda`. Its five
inputs are `source_sha` (the exact dispatch-ref SHA), `inference_sha` (the full
reachable revision pinned everywhere by that source), `backend`, `build_only`,
and `mac_go`. Actual evidence requires `build_only=false`; Mac execution also
requires `mac_go=true`. Build-only skips provisioning and all model execution.

The workflow provisions its own fixtures and native runtime. No pre-existing
oracle directory, manually exported model, or user-supplied runtime is required.
The registered `nax-kernel-validation.yml` dispatcher can call this reusable
workflow with `suite=cv` after the coordinator integrates that bridge; keep its
existing default NAX and aesthetic suites unchanged. The bridge is not part of
this source change. Neither CV workflow has an automatic push/PR/schedule trigger.

### Independent oracle production

For Mac, a separate manual-only reusable workflow,
`.github/workflows/cv-oracle-provision.yml`, runs before hardware validation. It
uses the real CPU forward of **Ultralytics 8.3.0**, upstream commit
`6e43d1e1e5db72afbf686dee6745669bcb124b0a`, and the exact published fused YOLO
weights. The maintained producer is `scripts/cv_oracle_producer.py`; it does not
call the Rust engine or depend on archived spike scripts. The [historical torch reference](https://github.com/SceneWorks/SceneWorks/blob/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f/docs/sc-3734/torch_ref.py)
requires an existing base oracle and only prints its
comparison deltas, so it is not used as a substitute for complete provisioning.

The producer creates the upstream YOLO11m graph, fuses its Conv/BN modules, and
strictly loads **all 225** pinned tensors, reversing only the documented OHWI to
OIHW storage transpose. Every key, shape, dtype and loaded value must match;
partial loading and random-weight references are rejected. It checks the
installed upstream architecture/math files against the pinned commit's digests,
then captures actual upstream block and head outputs through hooks. The
otherwise unexposed attention intermediate must reconstruct the upstream
projection input within `1e-5`. All required base/extended tensors have asserted
names, shapes, float32 dtype and finite values; the final tensor must be
`1x84x8400` with bounded class probabilities. Upstream NMS/box scaling must produce
exactly the documented four people on the pinned 810x1080 bus photograph.

`refs.safetensors`, `refs_ext.safetensors` and `ref_people.json` are thus new,
independent references with recorded provenance, not claims to have recovered
historical files whose digests were never recorded. They preserve every existing
Rust numerical assertion: base block4 `<1e-4`, other base blocks `<1e-3`, final
boxes `<1px`, final classes `<1e-3`, isolated blocks/attention/PE `<1e-3`, and photo
corners `<2px`/confidence `<0.02`. Missing, partial or failed oracle artifacts
reject the hardware run.

Tools install in a run-private CPython 3.12 venv. The primary wheel hashes pin
Ultralytics 8.3.0, PyTorch 2.8.0, torchvision 0.23.0, NumPy 1.26.4,
safetensors 0.6.2, OpenCV 4.10.0.84 and Pillow 11.3.0. Dependency resolution is
recorded; every selected wheel is length/hash checked before an offline install.
The tool installer and producer reject ordinary/local execution before any
model-capable import. The CPU job uses one thread and never selects MPS/CUDA.
Its tensors, source-file hashes, model/image identities, package versions,
installation reports and complete producer receipt are retained as artifacts.

### Pinned fixtures and runtime

`scripts/cv-fixture-manifest.json` records immutable URLs, byte sizes and SHA-256s.
`scripts/cv_provision.py` checks the supported HF cache roots for those exact
snapshots, verifies cached bytes, and copies them into a fresh regular-file
fixture directory. It never writes to an application/HF cache. Missing assets
come from the pinned public URLs. Corrupt cached bytes fail explicitly.

| Fixture | Immutable source |
| --- | --- |
| `people.jpg` | Ultralytics `ultralytics/assets/bus.jpg` at `790f9c067c9a3548acf7c6925cea14de9ad77787`; SHA-256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| `yolo11m_fused_mlx.safetensors` (Mac) | `SceneWorks/yolo11m-person-detect-mlx` at `d7027d3a8812bdebbf7862fc1c7dcfdeebb0f777` |
| `yolo11m.onnx` (Windows) | `SceneWorks/yolo11m-person-detect-onnx` at `3cffbaccca4f239ae6301d8b66ba721401ecfa8d` |
| `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` | `SceneWorks/dwpose-onnx` at `c330a9609201343576b8b69d222e38c1486d2854` |
| `rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx` | Same DWPose snapshot |
| `real_esrgan_x2.onnx`, `real_esrgan_x4.onnx` | `SceneWorks/real-esrgan-onnx` at `09f741bac80a246b407da3ee902bf5f3291b602f` |

All pinned ONNX exports are expected to be self-contained. A metadata-only
protobuf walk checks graph initializers, nested graphs, attribute tensors,
sparse tensors and functions. Any external tensor reference fails provisioning;
no external-data companion is silently omitted. Product loading still passes
through `model_io::ort_session` and retains its independent validation/progress.

The runtime is extracted from the same SHA-pinned ORT **1.26.0** wheels as the
product provisioners. Mac uses the CoreML-enabled arm64 dylib. Windows stages the
three ORT CUDA DLLs and the complete pinned CUDA 12.9/cuDNN 9.23 dependency wheel
set from `apps/desktop/src/cuda_provision.rs`, under run-private directories.
`SCENEWORKS_ORT_CUDA_DIR` and `SCENEWORKS_ORT_CUDNN_DIR` select those copies;
installed/global runtimes are unchanged. Every extracted library is hashed.

Payload limits are enforced: 3 GiB of asset downloads per provisioning job,
6 GiB of extracted runtime libraries, 512 MiB of oracle wheels and 128 MiB of
oracle outputs. Cold payloads are approximately 563 MB for Mac model/photo/ORT
staging and 2.400 GB for Windows model/photo/full runtime staging. The separate
Mac oracle inputs are 80.5 MB before cache reuse, and its seven primary tool
wheels total approximately 150 MB plus recorded bounded dependencies. The oracle
job has a 35-minute timeout; hardware proof retains its 180-minute timeout and
Windows execution controller's 170-minute budget. These are bounded correctness
passes, not memory matrices or calibration campaigns.

### Acceptance and artifacts

The controller independently rechecks pinned fixture hashes, the provisioned
runtime hash, oracle source/pin/weight/photo provenance and output hashes. It
records lockfile, executable, host/device, exact case/settings, duration,
complete-body receipts and outputs. Missing bodies/assets, failed assertions,
or CPU fallback fail the selected hardware proof. Each ORT graph must record
nonzero executions on its required provider in a retained proof-only ORT profile;
a configured device label alone is insufficient. Raw DWPose outputs and
normalized results are retained. Both Real-ESRGAN factors save PNGs from an odd
513x33 image crossing a 512-pixel tile boundary. Mac YOLO runs every retained
whole-network, per-block and photo oracle assertion; its attention retains the
existing explicitly selected CPU stream while other MLX work uses the GPU.

Mac hardware work enters the existing shared Metal execution lease. Windows
requests a real-weight runner and uses the existing controller with physical
GPU1 exposed as logical device 0. Distinct dispatches queue without canceling
running work. The CPU oracle workflow does not use the GPU. Normal CI never
invokes either learned workflow.

A passing compile, metadata/mock test, provisioning step or `build_only` run is
not native provider proof. Keep the actual independent-oracle and both hardware
validation artifacts with the acceptance report before closing SW-6775/SW-6680.

## Registered dispatcher

The existing registered `nax-kernel-validation.yml` dispatcher accepts `suite=cv`
and forwards only `source_sha`, `inference_sha`, `backend`, `build_only`, and
`mac_go` to the reusable CV workflow. Select the published consumer ref and pass
its exact 40-character SHA, plus the inference SHA in that source's lockfile.
For actual Mac MLX/CoreML proofs select `backend=mlx`, `build_only=false`,
`mac_go=true`; for Windows CUDA select `backend=cuda`, `build_only=false`,
`mac_go=false`. The NAX default and aesthetic CPU routes remain separate manual
suites. Ordinary CI cannot select CV. An actual Mac run first creates independent
CPU oracle artifacts in the separate reusable oracle job, then validates all six
provider cases; Windows validates all four cases on physical GPU1/logical0.
