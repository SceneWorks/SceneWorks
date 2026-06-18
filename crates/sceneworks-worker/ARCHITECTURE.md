# SceneWorks Worker Capability Matrix

SceneWorks runs **two parallel worker implementations** behind one HTTP job
contract:

- **Python** `apps/worker/scene_worker/` — Diffusers/PyTorch, GPU (CUDA) or CPU.
- **Rust** `apps/rust-worker/` + `crates/sceneworks-worker/` — the
  `sceneworks-rust-worker` binary, which plays **two roles from one binary**:
  - a **CPU utility worker** on Docker/Windows/Linux (`SCENEWORKS_GPU_ID=cpu`);
  - the full **MLX GPU worker** on macOS desktop (`gpu_id=mlx`).

Feature parity between the two is maintained by hand. This document is the
human-readable companion to the code's source of truth so a new job kind does
not get silently unsupported in one worker.

## Source of truth

- **Job kinds:** `crates/sceneworks-core/src/contracts.rs` → `enum JobType`
  (canonical; the `string_enum!` macro adds an `Unknown(String)` forward-compat
  variant, so the enum is the complete set of *known* kinds).
- **Routing oracle:** `crates/sceneworks-core/src/jobs_store.rs` →
  `mac_rust_supported` enumerates every kind's Rust/MLX-vs-Python-torch decision.
  It is a Rust `match` over every `JobType` variant with no wildcard for known
  kinds, so the compiler guarantees exhaustiveness — **a new `JobType` will not
  compile until it is added here.** That property is what makes the matrix below
  provably complete.
- **Dispatch sites:** Rust `crates/sceneworks-worker/src/lib.rs::run_utility_job`;
  Python `apps/worker/scene_worker/runtime.py` (job-type groups at lines 63–143
  plus the if/elif chain at ~1528).
- **Mirror discipline:** `runtime.py:68-143` carries explicit "Keep in sync with
  `contracts.rs::JobType / WorkerCapability`" comments.

## How a job reaches a worker

Routing is by **capability advertisement**, not by queue or static config:

1. Each worker advertises capabilities at registration, gated on backend probes
   (Python `worker_capabilities`, `runtime.py:171-213`) or native engine
   registration (Rust `engines::registry_capabilities`).
2. `jobs_store::worker_supports_job` matches `required_capability(job)` against
   the worker's advertised set. A real (non-dry-run) training job additionally
   requires `lora_train_execute`; a `preview:true` person job maps to the
   `*_preview` capability instead.
3. The macOS `mlx` worker **refuses** (leaves queued for the Python torch
   worker) any job whose model/mode is not MLX-eligible (`*_mlx_eligible`
   gates). A parallel candle (Windows/CUDA) lane mirrors this for narrow
   txt2img/txt2video.
4. CPU-utility kinds (`NON_GPU_JOB_TYPES`: `model_download`, `model_import`,
   `model_convert`, `lora_import`) never route to GPU workers.
5. Route decisions are logged (`RouteDecision`:
   `deferred_to_mlx | claimed_by_mlx | fell_back_to_torch | explicit_gpu`).

> "Rust ✅" in the matrix means **"in its capable configuration"** — CPU-utility
> off macOS, MLX GPU on macOS. Off-macOS, GPU arms in `lib.rs` are
> `#[cfg(target_os = "macos")]`-gated or never advertised.

### MLX generator cache residency

The macOS MLX worker keeps one generator resident across jobs so repeated image
or video requests do not cold-load weights every time. To avoid leaving a
multi-GB Metal/MLX allocation resident while the desktop app is idle, the cache
evicts its resident generator after 300 idle seconds by default and clears the
MLX backend cache. Tune this with `SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS`; set
it to `0` to disable idle eviction.

## Capability matrix

Legend: ✅ handled · ❌ never dispatched (explicit fail-arm) · ⚠️ handled but
conditional/partial.

| Job kind (`JobType`) | Python worker | Rust worker | Proof (file:line) |
|---|---|---|---|
| `placeholder` | ❌ (no adapter → else) | ✅ utility | py `runtime.py:1552` / rs `lib.rs:623` |
| `image_generate` | ✅ diffusers/torch | ✅ MLX (Mac, if eligible) | py `1528`→`run_image_job` / rs `lib.rs:629`, gate `jobs_store.rs:4093` |
| `image_edit` | ✅ torch (image handler) | ⚠️ MLX-eligible edit models only | py `1528` / rs `lib.rs:637`, oracle `jobs_store.rs:2027` |
| `image_vqa` | ✅ torch (SenseNova-U1) | ⚠️ MLX, SenseNova-U1 only | py `1534`→`run_vqa_job` / rs `lib.rs:652`, oracle `2040` |
| `image_interleave` | ✅ torch | ⚠️ MLX, SenseNova-U1 only | py `1536`→`run_interleave_job` / rs `lib.rs:655`, oracle `2040` |
| `image_detail` | ✅ torch | ⚠️ MLX, SDXL/RealVisXL only | py `1532`→`run_detail_job` / rs `lib.rs:643`, oracle `2029` |
| `image_upscale` | ✅ torch (Real-ESRGAN + AuraSR) | ⚠️ MLX Real-ESRGAN/SeedVR2; **AuraSR dropped on Mac** | py `1530`→`run_upscale_job` / rs `lib.rs:752`, oracle `2099` |
| `video_generate` | ✅ torch (Wan/LTX/SVD) | ⚠️ MLX-eligible models | py `1538`→`run_video_job` / rs `lib.rs:669`, oracle `2047` |
| `video_extend` | ✅ torch | ⚠️ MLX, LTX IC-LoRA + Wan TI2V-5B only | py `1538` / rs `lib.rs:669`, oracle `2053` |
| `video_bridge` | ✅ torch | ⚠️ MLX, LTX IC-LoRA + Wan TI2V-5B only | py `1538` / rs `lib.rs:669`, oracle `2053` |
| `person_replace` | ✅ torch (Wan-VACE) | ⚠️ MLX Wan-VACE/SCAIL-2 only | py `1538` (`replace_person` mode, `video_adapters.py:543`) / rs `lib.rs:682`, oracle `2066` |
| `video_upscale` | ❌ **no torch path** | ⚠️ **Mac-only**, MLX SeedVR2 | py: else `1552` / rs `lib.rs:761` (`cfg(macos)`), oracle `2116` |
| `person_detect` | ✅ torch (YOLO/SAM2) | ✅ MLX (Mac) + CPU procedural preview | py `1540`→`run_person_job` / rs `lib.rs:726`, oracle `2079` |
| `person_track` | ✅ torch (ByteTrack) | ✅ MLX (Mac) + CPU procedural preview | py `1540` / rs `lib.rs:764`, oracle `2079` |
| `pose_detect` | ✅ torch (rtmlib, if backend) | ✅ MLX RTMW (Mac-only arm) | py `1542`→`run_pose_job` / rs `lib.rs:734` (`cfg(macos)`), oracle `2084` |
| `kps_extract` | ✅ torch (InsightFace, if backend) | ✅ MLX SCRFD (Mac-only arm) | py `1544`→`run_kps_extract_job` / rs `lib.rs:742` (`cfg(macos)`), oracle `2090` |
| `lora_train` | ✅ torch (real exec needs backend) | ⚠️ MLX-native families only | py `1546`→`run_lora_train_job` / rs `lib.rs:690`, oracle `2131` |
| `training_caption` | ✅ torch | ⚠️ MLX JoyCaption only | py `1548`→`run_training_caption_worker_job` / rs `lib.rs:696`, oracle `2133` |
| `prompt_refine` | ✅ torch (PromptRefiner fallback) | ✅ MLX/candle TextLlm | py `1550`→`run_prompt_refine_job` / rs `lib.rs:705`, oracle `2021` |
| `model_download` | ❌ (Python dropped fallbacks) | ✅ utility | rs `lib.rs:708`, oracle `2016` |
| `model_import` | ❌ | ✅ utility | rs `lib.rs:714`, oracle `2017` |
| `model_convert` | ❌ | ✅ utility | rs `lib.rs:717`, oracle `2129` |
| `lora_import` | ❌ | ✅ utility | rs `lib.rs:711`, oracle `2018` |
| `frame_extract` | ❌ | ✅ utility (FFmpeg) | rs `lib.rs:720`, oracle `2019` |
| `timeline_export` | ❌ | ✅ utility (FFmpeg MP4) | rs `lib.rs:723`, oracle `2020` |

**Not job kinds** (routing/readiness capabilities, not dispatchable rows):
`person_detect_preview`, `person_track_preview` (Rust CPU procedural),
`person_segment` (Python SAM2 readiness sub-capability for replace),
`lora_train_execute` (real-training gate), and the `cpu`/`gpu` markers — all in
`contracts.rs::WorkerCapability`.

## Parity gaps

- **Rust-only (Python explicitly fails it):** `placeholder`, `model_download`,
  `model_import`, `model_convert`, `lora_import`, `frame_extract`,
  `timeline_export` — the CPU-utility family. The Python worker `else` arm
  (`runtime.py:1552`) returns "No adapter exists for this job type yet"; the
  Python worker no longer advertises or runs utility fallbacks.
- **Rust-only AND macOS-only (true single point of failure):** `video_upscale` —
  there is **no torch path on any platform** (`jobs_store.rs:2112`,
  `lib.rs:758`). Unsupported off macOS by design.
- **Generation kinds:** the Python torch worker handles the broadest set of
  model/payload shapes; the macOS MLX worker serves the MLX-eligible subset of
  each ⚠️ row and defers the rest back to Python.

## Maintenance

When adding a `JobType` variant, the compiler will force you to update
`mac_rust_supported` in `jobs_store.rs`. Also update: the dispatch arm in
`lib.rs` (Rust) and/or the if/elif chain + job-type groups in `runtime.py`
(Python), the capability mirrors in `runtime.py:63-143`, and **this matrix**.
