# SceneWorks Worker Capability Matrix

SceneWorks runs a **single native worker** behind one HTTP job contract. The
retired Python (`apps/worker/scene_worker/`) Diffusers/PyTorch worker was deleted
in epic 8283 (Python eradication); everything now runs natively.

- **Rust** `apps/rust-worker/` + `crates/sceneworks-worker/` — the
  `sceneworks-rust-worker` binary, which plays **three roles from one binary**:
  - a **CPU utility worker** on Docker/Windows/Linux (`SCENEWORKS_GPU_ID=cpu`);
  - the full **MLX GPU worker** on macOS desktop (`gpu_id=mlx`);
  - the **candle GPU worker** on the Windows/CUDA build (`backend-candle`).

This document is the human-readable companion to the code's source of truth so a
new job kind does not get silently unsupported.

## Source of truth

- **Job kinds:** `crates/sceneworks-core/src/contracts.rs` → `enum JobType`
  (canonical; the `string_enum!` macro adds an `Unknown(String)` forward-compat
  variant, so the enum is the complete set of *known* kinds).
- **Capability gate:** `crates/sceneworks-core/src/jobs_store.rs` →
  `required_capability` maps a job to the capability a worker must advertise
  (including the person-preview and control-training exceptions).
- **Routing oracles:** `crates/sceneworks-core/src/jobs_store/routing/gaps.rs` →
  `mac_rust_supported` and `candle_supported`, backed by the eligibility
  predicates in `routing/mlx.rs` and `routing/candle.rs`. Each oracle is a Rust
  `match` over every known `JobType`, so adding a variant forces an explicit
  routing decision.
- **Dispatch site:** `crates/sceneworks-worker/src/lib.rs::run_utility_job`.
- **Drift guard:** `crates/sceneworks-worker/src/architecture_tests.rs` extracts
  the canonical `JobType` declarations and fails unless this matrix has exactly
  one row for every known kind and `run_utility_job` has an explicit dispatch
  arm for every variant.

## How a job reaches a worker

Routing is by **capability advertisement**, not by queue or static config:

1. Each worker advertises capabilities at registration, gated on native engine
   registration (`engines::registry_capabilities`) and the GPU probe (`gpu.rs`).
2. `jobs_store::worker_supports_job` matches `required_capability(job)` against
   the worker's advertised set. A real (non-dry-run) training job additionally
   requires `lora_train_execute`; a `preview:true` person job maps to the
   `*_preview` capability instead.
3. The macOS `mlx` worker **refuses** (leaves queued) any job whose model/mode is
   not MLX-eligible (`*_mlx_eligible` gates). A parallel candle (Windows/CUDA)
   lane mirrors this for its supported families. With the Python torch worker
   retired from every surface, a job outside the native lanes is not silently
   downgraded — it is refused (`no-torch-fallback`, sc-5968) and stays queued
   until a worker that can serve it registers.
4. CPU-utility kinds (`NON_GPU_JOB_TYPES`: `model_download`, `model_import`,
   `model_convert`, `lora_import`) never route to GPU workers.
5. Route decisions are logged (`RouteDecision`:
   `deferred_to_mlx | claimed_by_mlx | claimed_by_candle | claimed_by_gpu | explicit_gpu`).

> "✅" in the matrix means **"in its capable configuration"** — CPU-utility off
> macOS, MLX GPU on macOS, candle GPU on the Windows/CUDA build. Off-macOS, MLX
> arms in `lib.rs` are `#[cfg(target_os = "macos")]`-gated or never advertised.

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

<!-- job-matrix:start -->
| Job kind (`JobType`) | Native worker | Durable code anchors |
|---|---|---|
| `placeholder` | ✅ CPU utility | `run_utility_job` → `run_placeholder_job`; `cpu_gpu` |
| `image_generate` | ⚠️ MLX / candle, eligible models and modes only | `run_utility_job` → `run_image_generate_job`; `job_is_mlx_eligible`; `image_job_is_candle_eligible` |
| `image_edit` | ⚠️ MLX / candle, eligible edit models only | `run_utility_job` → `run_image_generate_job`; `job_is_mlx_eligible`; `image_job_is_candle_eligible` |
| `image_vqa` | ⚠️ MLX / candle, SenseNova-U1 only | `run_utility_job` → `run_vqa_job`; `understanding_job_is_mlx_eligible` |
| `image_interleave` | ⚠️ MLX / candle, SenseNova-U1 only | `run_utility_job` → `run_interleave_job`; `understanding_job_is_mlx_eligible` |
| `video_generate` | ⚠️ MLX / candle, eligible models and modes only | `run_utility_job` → `run_video_generate_job`; `video_job_is_mlx_eligible`; `video_job_is_candle_eligible` |
| `video_extend` | ⚠️ MLX LTX/Wan eligible paths; candle Wan-VACE eligible path | `run_utility_job` → `run_video_generate_job`; `video_job_is_mlx_eligible`; `video_job_is_candle_eligible` |
| `video_bridge` | ⚠️ MLX LTX/Wan eligible paths; candle Wan-VACE eligible path | `run_utility_job` → `run_video_generate_job`; `video_job_is_mlx_eligible`; `video_job_is_candle_eligible` |
| `person_detect` | ✅ MLX / candle model-backed; CPU procedural preview | `run_utility_job` → `run_person_detect_job`; `required_capability`; `mlx_gpu`; `with_candle_capabilities` |
| `person_track` | ✅ MLX / candle model-backed; CPU procedural preview | `run_utility_job` → `run_person_track_job`; `required_capability`; `mlx_gpu`; `with_candle_capabilities` |
| `person_replace` | ⚠️ MLX / candle, eligible Wan-VACE/SCAIL paths | `run_utility_job` → `run_video_generate_job`; `video_job_is_mlx_eligible`; `video_job_is_candle_eligible` |
| `audio_generate` | ⚠️ native candle audio registry on Mac / CUDA when linked | `run_utility_job` → `run_audio_generate_job`; `inference_runtime::audio`; `mlx_gpu`; `with_candle_capabilities` |
| `pose_detect` | ✅ MLX/CoreML on Mac / candle/CUDA off-Mac | `run_utility_job` → `run_pose_detect_job`; `mlx_gpu`; `with_candle_capabilities` |
| `kps_extract` | ✅ MLX SCRFD on Mac / candle SCRFD off-Mac | `run_utility_job` → `run_kps_extract_job`; `mlx_gpu`; `with_candle_capabilities` |
| `image_upscale` | ⚠️ MLX / candle Real-ESRGAN or SeedVR2; AuraSR dropped | `run_utility_job` → `run_image_upscale_job`; `upscale_job_is_mlx_eligible`; `upscale_job_is_candle_eligible` |
| `image_detail` | ⚠️ MLX, SDXL/RealVisXL detail models only | `run_utility_job` → `run_image_detail_job`; `job_is_mlx_eligible`; `mlx_gpu` |
| `image_segment` | ⚠️ Mac-only MLX SAM3 | `run_utility_job` → `run_image_segment_job`; `mlx_gpu`; `mac_rust_supported` |
| `video_upscale` | ⚠️ MLX on Mac / candle off-Mac, SeedVR2 only | `run_utility_job` → `run_video_upscale_job`; `video_upscale_job_is_mlx_eligible`; `video_upscale_job_is_candle_eligible` |
| `frame_extract` | ✅ CPU utility (FFmpeg) | `run_utility_job` → `run_frame_extract_job`; `cpu_gpu` |
| `timeline_export` | ✅ CPU utility (FFmpeg MP4) | `run_utility_job` → `run_timeline_export_job`; `cpu_gpu` |
| `model_download` | ✅ CPU utility | `run_utility_job` → `run_model_download_job`; `cpu_gpu` |
| `model_import` | ✅ CPU utility | `run_utility_job` → `run_model_import_job`; `cpu_gpu` |
| `model_convert` | ✅ CPU utility | `run_utility_job` → `run_model_convert_job`; `cpu_gpu` |
| `lora_import` | ✅ CPU utility | `run_utility_job` → `run_lora_import_job`; `cpu_gpu` |
| `lora_download` | ✅ CPU utility | `run_utility_job` → `run_lora_download_job`; `cpu_gpu` |
| `lora_train` | ⚠️ MLX / candle, native trainable families only | `run_utility_job` → `run_lora_train_job`; `training_job_is_mlx_eligible`; `training_job_is_candle_eligible` |
| `control_training` | ⚠️ candle ControlNet trainer only | `run_utility_job` → `run_control_training_job`; `required_capability`; `is_real_training_job`; `training_job_is_candle_eligible` |
| `training_caption` | ⚠️ MLX / candle JoyCaption only | `run_utility_job` → `run_training_caption_job`; `caption_job_is_mlx_eligible` |
| `dataset_parquet_import` | ✅ CPU utility (Parquet scan + public image fetch) | `run_utility_job` → `run_dataset_parquet_import_job`; `NON_GPU_JOB_TYPES`; `mac_rust_supported`; `candle_supported` |
| `dataset_analysis` | ⚠️ MLX CLIP lane when linked; no candle CLIP lane yet | `run_utility_job` → `run_dataset_analysis_job`; `mlx_gpu`; `mac_rust_supported` |
| `dataset_upscale` | ✅ MLX/CoreML on Mac / candle/CUDA off-Mac | `run_utility_job` → `run_dataset_upscale_job`; `mlx_gpu`; `with_candle_capabilities` |
| `dataset_face_analysis` | ✅ MLX face stack on Mac / candle face stack off-Mac | `run_utility_job` → `run_dataset_face_analysis_job`; `mlx_gpu`; `with_candle_capabilities` |
| `face_likeness_compare` | ✅ MLX face stack on Mac / candle face stack off-Mac | `run_utility_job` → `run_face_likeness_compare_job`; `mlx_gpu`; `with_candle_capabilities` |
| `prompt_refine` | ⚠️ native TextLlm provider when its registry lane is linked | `run_utility_job` → `run_prompt_refine_job`; `registry_capabilities`; `mac_rust_supported`; `candle_supported` |
<!-- job-matrix:end -->

**Not job kinds** (routing/readiness capabilities, not dispatchable rows):
`person_detect_preview`, `person_track_preview` (Rust CPU procedural),
`person_segment` (SAM readiness sub-capability for replace),
`lora_train_execute` (real-training gate), and the `cpu`/`gpu` markers — all in
`contracts.rs::WorkerCapability`.

## Coverage notes

- **Utility family (CPU, any platform):** `placeholder`, `model_download`,
  `model_import`, `model_convert`, `lora_import`, `lora_download`,
  `frame_extract`, `timeline_export` — served by the Rust CPU utility worker.
- **Platform-specific:** `image_segment` is Mac-only MLX. `video_upscale` is
  SeedVR2 on both native GPU lanes: MLX on Mac and candle/CUDA off-Mac.
- **Generation kinds:** each ⚠️ row serves the MLX/candle-eligible subset of its
  model/payload shapes; a shape outside the native lane is refused (no torch
  fallback), so it stays queued rather than silently downgraded.

## Maintenance

When adding a `JobType` variant, the routing-oracle matches force an explicit
backend decision. Also update the dispatch arm in `lib.rs::run_utility_job`, the
capability mirrors in `contracts.rs`, and this matrix. The
`architecture_matrix_covers_every_known_job_type_and_dispatch_arm` test guards
the enum/dispatch/documentation boundary against silent drift.
