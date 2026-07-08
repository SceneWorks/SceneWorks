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
- **Routing oracle:** `crates/sceneworks-core/src/jobs_store.rs` →
  `mac_rust_supported` enumerates every kind's MLX/candle-eligibility decision.
  It is a Rust `match` over every `JobType` variant with no wildcard for known
  kinds, so the compiler guarantees exhaustiveness — **a new `JobType` will not
  compile until it is added here.** That property is what makes the matrix below
  provably complete.
- **Dispatch site:** `crates/sceneworks-worker/src/lib.rs::run_utility_job`.

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

| Job kind (`JobType`) | Native worker | Proof (file:line) |
|---|---|---|
| `placeholder` | ✅ utility | rs `lib.rs:623` |
| `image_generate` | ✅ MLX (Mac) / candle (Win), if eligible | rs `lib.rs:629`, gate `jobs_store.rs:4093` |
| `image_edit` | ⚠️ MLX/candle-eligible edit models only | rs `lib.rs:637`, oracle `jobs_store.rs:2027` |
| `image_vqa` | ⚠️ MLX, SenseNova-U1 only | rs `lib.rs:652`, oracle `2040` |
| `image_interleave` | ⚠️ MLX, SenseNova-U1 only | rs `lib.rs:655`, oracle `2040` |
| `image_detail` | ⚠️ MLX, SDXL/RealVisXL only | rs `lib.rs:643`, oracle `2029` |
| `image_upscale` | ⚠️ MLX Real-ESRGAN/SeedVR2; **AuraSR dropped on Mac** | rs `lib.rs:752`, oracle `2099` |
| `video_generate` | ⚠️ MLX/candle-eligible models | rs `lib.rs:669`, oracle `2047` |
| `video_extend` | ⚠️ MLX, LTX IC-LoRA + Wan TI2V-5B only | rs `lib.rs:669`, oracle `2053` |
| `video_bridge` | ⚠️ MLX, LTX IC-LoRA + Wan TI2V-5B only | rs `lib.rs:669`, oracle `2053` |
| `person_replace` | ⚠️ MLX Wan-VACE/SCAIL-2 only | rs `lib.rs:682`, oracle `2066` |
| `video_upscale` | ⚠️ **Mac-only**, MLX SeedVR2 | rs `lib.rs:761` (`cfg(macos)`), oracle `2116` |
| `person_detect` | ✅ MLX/candle + CPU procedural preview | rs `lib.rs:726`, oracle `2079` |
| `person_track` | ✅ MLX/candle + CPU procedural preview | rs `lib.rs:764`, oracle `2079` |
| `pose_detect` | ✅ MLX RTMW (Mac) / candle (Win) | rs `lib.rs:734`, oracle `2084` |
| `kps_extract` | ✅ MLX SCRFD (Mac) / candle (Win) | rs `lib.rs:742`, oracle `2090` |
| `lora_train` | ⚠️ MLX/candle-native families only | rs `lib.rs:690`, oracle `2131` |
| `training_caption` | ⚠️ MLX/candle JoyCaption only | rs `lib.rs:696`, oracle `2133` |
| `prompt_refine` | ✅ MLX/candle TextLlm | rs `lib.rs:705`, oracle `2021` |
| `model_download` | ✅ utility | rs `lib.rs:708`, oracle `2016` |
| `model_import` | ✅ utility | rs `lib.rs:714`, oracle `2017` |
| `model_convert` | ✅ utility | rs `lib.rs:717`, oracle `2129` |
| `lora_import` | ✅ utility | rs `lib.rs:711`, oracle `2018` |
| `frame_extract` | ✅ utility (FFmpeg) | rs `lib.rs:720`, oracle `2019` |
| `timeline_export` | ✅ utility (FFmpeg MP4) | rs `lib.rs:723`, oracle `2020` |

**Not job kinds** (routing/readiness capabilities, not dispatchable rows):
`person_detect_preview`, `person_track_preview` (Rust CPU procedural),
`person_segment` (SAM readiness sub-capability for replace),
`lora_train_execute` (real-training gate), and the `cpu`/`gpu` markers — all in
`contracts.rs::WorkerCapability`.

## Coverage notes

- **Utility family (CPU, any platform):** `placeholder`, `model_download`,
  `model_import`, `model_convert`, `lora_import`, `frame_extract`,
  `timeline_export` — served by the Rust CPU utility worker.
- **macOS-only:** `video_upscale` — native SeedVR2 has no lane off macOS
  (`jobs_store.rs:2112`, `lib.rs:758`). Unsupported elsewhere by design.
- **Generation kinds:** each ⚠️ row serves the MLX/candle-eligible subset of its
  model/payload shapes; a shape outside the native lane is refused (no torch
  fallback), so it stays queued rather than silently downgraded.

## Maintenance

When adding a `JobType` variant, the compiler will force you to update
`mac_rust_supported` in `jobs_store.rs`. Also update: the dispatch arm in
`lib.rs::run_utility_job`, the capability mirrors in `contracts.rs`, and **this
matrix**.
