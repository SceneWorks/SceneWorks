use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use reqwest::header;
use reqwest::StatusCode;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, JobSnapshot, JobStatus, JobType, JsonObject,
    ProgressRequest, ProgressStage, WorkerCapability, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot, WorkerStatus, WorkerUtilizationSnapshot,
};
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
// The single source of truth for which `mlx.converter` discriminators the native converters handle.
// `resolve_convert_plan` rejects anything not on it up front so this worker's converter set can never
// drift from the convert-gap gate that derives its allow-list from the same const (sc-10573).
use sceneworks_core::jobs_store::NATIVE_CONVERTERS;
use sceneworks_core::jsonc::strip_jsonc_comments;
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, detect_lora_family, detect_model_family, first_safetensors_path,
    read_safetensors_header, reconcile_detected_family, FamilyMismatch, SafetensorsHeaderError,
};
// Only the cfg-gated adapter resolvers (image `resolve_adapters`, video
// `resolve_lora_file`) use this, so gate the import identically or the parity
// build (no `backend-candle`) trips `-D unused-imports` (sc-10221).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::lora_family::resolve_adapter_in_dir;
use sceneworks_core::lora_url::{
    lora_source_url_file_name, lora_source_url_file_stem, parse_lora_source_url_with_private,
    validate_public_ip,
};
use sceneworks_core::project_store::{ProjectStore, ProjectStoreError};
use sceneworks_core::slug::slugify;
use sceneworks_core::time::{format_unix_seconds, now_unix_seconds};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::MissedTickBehavior;
use tracing::Level;
use uuid::Uuid;

// Shared `advanced` knob accessors (sc-4281). The MLX image/video job paths are macOS-gated; the
// candle InstantID lane (sc-5491) is the first off-Mac caller, so the module also compiles on the
// Windows candle build. The candle lane calls only a subset (`flag`/`str`/`f32_clamped`), so allow
// dead_code there (the rest are MLX-only) — same pattern as `openpose_skeleton`. On a non-candle
// Windows/Linux build it stays excluded, so its accessors are never uncalled-dead there.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod advanced;
mod api_client;
// Lazy, on-demand download-credential pull from the macOS desktop credential socket
// (sc-5891). Compiles on all targets; the socket I/O is `cfg(unix)` and inert unless
// the desktop injects `SCENEWORKS_CRED_IPC_*`, so server/Docker/Windows are unaffected.
mod credentials_ipc;
// Generic single-resident, dedicated-thread model cache scaffolding (sc-11191, F-019): the
// `CacheThread<K, M>` + `Fingerprint` + panic/idle-timeout/oneshot-seam machinery shared verbatim by
// `generator_cache` and `refine_model_cache`. All-targets like its two consumers; off macOS the
// production seams are cfg'd out, so allow dead_code there (mirrors the generator_cache precedent).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod cache_thread;
// Backend-neutral generator load/run cache (epic 3720, sc-3724). Typed entirely against
// `gen_core::*` (no tensor types leak), so it links on ALL targets — the production load seam
// (`with_cached_generator`) is reached only from the macOS image/video paths, but the all-targets
// stub test exercises the load→progress→cancel→output contract with no backend linked. Off macOS
// the production caller is cfg'd out, so allow dead_code there (the engines.rs precedent).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod generator_cache;
// Resident-model cache for the native prompt-refine / caption / describe LLM (sc-8840, F-038): the
// text-LLM sibling of `generator_cache`. Typed entirely against the tensor-free
// `gen_core::core_llm::*` contract, so it links on ALL targets — the production seam
// (`with_cached_refiner`) is reached only from the native refine path (macOS MLX / Windows candle),
// so off both natives it is dead (allow it, mirroring the `generator_cache` precedent). Caches the
// ~16 GB refine model keyed by weights dir + selection reqs with the SAME idle-eviction window as
// `generator_cache` so a single setting bounds resident model memory across both lanes.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
mod refine_model_cache;
use api_client::*;
// Backend-neutral engine dispatch table + registry-derived capability advertisement
// (sc-3723). All-targets: the table is pure data and the derivation runs off-macOS off an
// (empty) registry, so a future candle backend lights up with zero worker changes. Off
// macOS the only consumers are the (all-targets) registry-derivation tests — the production
// caller (`mlx_gpu`) is macOS-gated — so allow dead_code on the non-macOS lib build (the
// person_replace pattern); the stub test still exercises it on every target.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod engines;
mod gpu;
use gpu::*;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod mlx_fit_gate;
// CUDA/candle VRAM fit-gate + small-card emulation (epic 10765 Phase 0, sc-10766). Pure helpers wired
// into `generate_candle_stream`; gated to the same candle lane as that consumer so the pub(crate)
// helpers aren't dead code (→ `-D warnings`) in the non-candle / macOS builds.
mod job_metrics;
mod supervisor;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod vram_gate;
// Krea pose-ControlNet VRAM fit ladder (sc-11754, epic 8459 → epic 10765). The dedicated fit-gate for the
// control lane, which is diverted around the base.rs `generate_candle_stream` gate. Same candle cfg as
// `vram_gate` (its only consumer, krea_control_candle.rs, is under that cfg) so its pub(crate) helpers
// aren't dead code under `-D warnings` on the non-candle / macOS builds.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod krea_control_fit;
use supervisor::*;
mod model_jobs;
use model_jobs::*;
mod media_jobs;
use media_jobs::*;
// Image-decode backstop (sc-6143): transcodes a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/
// BMP/GIF) to PNG at decode time. Compiles on all targets; the transcoder is the shared
// `sceneworks_core::media_convert` routine (sips on macOS, ffmpeg elsewhere).
mod image_decode;
mod image_jobs;
use image_jobs::*;
// Ideogram 4 mandatory JSON-caption conditioning + placeholder detect-and-recover (epic 4725,
// sc-6501). Pure prompt-guard + post-render heuristic, compiled cross-platform so its unit tests run
// on the Linux parity lane. sc-6610: its functions are called only from the macOS MLX generate path
// (`image_jobs/base.rs` `generate_stream`, `#[cfg(target_os = "macos")]`) — off-Mac, Ideogram routes
// to candle (txt2img) or the torch worker, neither of which applies the caption guard, so they read
// as dead code on EVERY non-macOS build (the candle `backend-candle` lane included; the prior
// `not(feature = "backend-candle")` carve-out wrongly assumed the candle path called them).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod ideogram_caption;
// SenseNova-U1 understanding + interleave jobs (epic 3180, sc-3905 — Path B). VQA + Document
// Studio (interleave) consume the concrete `T2iModel` directly (the `Generator` contract emits
// Images/Video only). The handlers are compiled cross-platform (with non-macOS error stubs); the
// real in-process MLX work is macOS-gated inside the module.
mod sensenova_jobs;
use sensenova_jobs::*;
mod video_jobs;
use video_jobs::*;
// Replace-person mask pipeline (epic 3040, sc-3521): cross-platform mask rasterization /
// resample / stored-seg-mask load, so the mask-port-vs-Python parity test runs on the
// Linux CI lane. Its masks are consumed only by the macOS Wan-VACE path in `video_jobs`,
// so off macOS the items are otherwise unused (the parity tests still build + run).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod person_replace;
mod training_jobs;
use training_jobs::*;
mod caption_jobs;
use caption_jobs::*;
// The shared scaffold both dataset-analysis jobs route through (sc-8836, F-034) — the `CancelJoinGuard`
// select loop, per-item progress ramp, and sidecar POST extracted out of the two near-duplicate modules.
// Gated to the same lanes as its only callers (the real `run_*_analysis_job` in the two modules below);
// on the parity lane those fall back to no-op stubs, so the scaffold has no consumer and must not compile.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod analysis_jobs_common;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use analysis_jobs_common::*;
mod dataset_analysis_jobs;
use dataset_analysis_jobs::*;
mod face_analysis_jobs;
use face_analysis_jobs::*;
// sc-4407 — the shared, generator-agnostic face-likeness scorer (epic 4406): the backbone identity-
// likeness component the Angles (sc-4409) / Poses (sc-4410) / With-Character (sc-4411) surfaces call as
// a post-pass over a finished generation. Its public seam (`FaceLikenessScorer`) has no production
// caller YET — the consuming surfaces are separate stories — so allow the unused seam here; the pure
// scoring core is exercised by the module's unit tests and the seam by the ignored real-weight test.
#[allow(dead_code)]
mod face_likeness;
// sc-4415 — on-demand "compare image to another" likeness tool (epic 4406): scores a CANDIDATE asset
// against a SOURCE identity reference asset through the shared `face_likeness` scorer. Lives in
// Character Studio Assets; routed as the `face_likeness_compare` job type.
mod face_likeness_compare_jobs;
use face_likeness_compare_jobs::*;
mod prompt_refine_jobs;
use prompt_refine_jobs::*;
mod downloads;
// sc-6541 closed-loop study: test-only LoRA output-quality eval harness (research instrument) —
// see the module doc + docs/sc-6541/closed-loop-protocol.md.
#[cfg(all(test, target_os = "macos"))]
mod lora_eval_harness;
// sc-6541 closed-loop study: native-Rust LoRA train→generate driver (research instrument) —
// see the module doc + docs/sc-6541/closed-loop-protocol.md.
#[cfg(all(test, target_os = "macos"))]
mod lora_train_driver;
// Shared test-support helpers for the real-weight smoke harnesses (sc-8866, epic 8800): the
// byte-identical `env_or` + RGB8 degenerate-decode floor checks (`image_mean`/`image_std`/
// `is_all_zero`/`save_png`) that were copy-pasted across every `*_mlx_smoke.rs` (macOS) and
// `*_gpu_smoke.rs` (off-Mac candle) file + `footprint_measure.rs`. Gated on the SUPERSET of both
// smoke lanes so it compiles exactly where a smoke that imports it does.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod smoke_support;
// Real-weight GPU smoke for the candle SCAIL-2 lane (sc-7078). Test-only + candle-only; never built
// in normal compiles. Drives the shipped worker conditioning + `gen_core::load("scail2_14b")`.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod scail2_gpu_smoke;
// Real-weight GPU smoke for the candle RealVisXL Lightning lane (sc-7176). Test-only + candle-only;
// drives `gen_core::load("sdxl")` with the forced `lightning` sampler against the distilled checkpoint.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod realvisxl_lightning_gpu_smoke;
// Real-weight GPU smoke for the candle SDXL edit + PiD super-resolving decode (epic 7840, sc-8044).
// Test-only + candle-only; drives the bespoke `candle_gen_sdxl::SdxlEdit` provider (inpaint) with the
// `pid_sdxl` student attached, asserting the PiD decode super-resolves the render-sized native decode.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod sdxl_edit_pid_gpu_smoke;
// Real-weight GPU smoke for the candle FLUX.2-dev lane (epic 6564 sc-7458). Test-only + candle-only;
// drives `gen_core::load("flux2_dev")` with a Q4 LoadSpec (CPU-stage → quantize-onto-GPU) against the
// dense diffusers snapshot — the worker-lane validation backing the off-Mac candle routing wire.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_dev_gpu_smoke;
// Real-weight GPU smoke for the candle Anima 2B lane (epic 10512, sc-10625 — the hardware-gated
// acceptance extracted from sc-10525). Test-only + candle-only; drives `gen_core::load("anima_base" |
// "anima_aesthetic" | "anima_turbo")` against the dense bf16 circlestone-labs/Anima split_files
// snapshot (± an official LoRA/LoKr), proving the candle Anima port renders coherently on real CUDA —
// the evidence that unblocks flipping `macOnly: false` / `candle_routed = true` (sc-10625).
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod anima_gpu_smoke;
// Real-weight GPU smoke for the candle SANA 1600M lane (epic 8485, sc-11780). Test-only + candle-only;
// drives the WORKER's `resolve_weights_dir("sana_1600m")` (the diffusers-snapshot-root resolution) +
// `gen_core::load("sana_1600m")` against the whole `Efficient-Large-Model/Sana_1600M_1024px_diffusers`
// snapshot, proving the candle SANA port renders a coherent true-CFG 1024² image on real CUDA — the
// hardware evidence backing `macOnly: false` / `candle_routed = true`.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod sana_candle_gpu_smoke;
// Real-weight GPU smoke for the candle InstantID + PiD super-resolving decode (epic 7840, sc-8386).
// Test-only + candle-only; drives the bespoke `candle_gen_instantid::InstantId` provider across
// Identity/Angle/Pose with the `pid_sdxl` student attached, asserting the PiD decode 4×-super-resolves
// the native decode AND the ArcFace identity likeness survives. Validates the sc-8373 InstantID lane.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod instantid_pid_gpu_smoke;
// Real-weight GPU smoke for the candle Z-Image + PiD decode (epic 7840, sc-8033). Test-only +
// candle-only; drives `gen_core::load("z_image_turbo", spec.with_pid(pid_flux, gemma))` — the generic
// candle t2i lane (sc-9727) — proving Z-Image's flux-aliased latent decodes through the pid_flux
// student at 4× (native 1024² -> 4096²). Z-Image has no dedicated pid_zimage; it reuses pid_flux.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_pid_gpu_smoke;
// Real-weight MLX smoke for the Krea 2 Turbo worker lane (epic 7565 sc-7575). Test-only + macOS-only;
// drives `gen_core::load("krea_2_turbo")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir —
// the worker-lane validation (the crate links + drives the engine), not just the mlx-gen-krea crate.
#[cfg(all(test, target_os = "macos"))]
mod krea_turbo_mlx_smoke;
// Real-weight MLX smoke for the Krea 2 Turbo pose-ControlNet worker lane on a PACKED Q8 base (sc-11796).
// Test-only + macOS-only; drives `gen_core::load("krea_2_turbo_control")` with the exact packed-q8
// `LoadSpec` `krea_control_spec` builds and asserts the pose steers the render vs a base passthrough —
// the worker-lane proof that pose control is honored on the installed quant tier (not silently dropped).
#[cfg(all(test, target_os = "macos"))]
mod krea_control_mlx_smoke;
// Real-weight MLX smoke for the FLUX.1-dev strict-control worker lane (sc-8244; engine E2 sc-8239).
// Test-only + macOS-only; drives `gen_core::load("flux1_dev_control")` (Dir base + Shakker control
// overlay) per control mode (pose/canny/depth) and asserts a control-vs-control-free steer — the
// worker-lane validation that the crate links + drives the registered control generator end-to-end.
#[cfg(all(test, target_os = "macos"))]
mod flux1_control_mlx_smoke;
// Real-weight MLX smokes for the SD3.5 worker lane (epic 7841 S6 sc-7875 — the MLX-path validation
// boundary). Test-only + macOS-only; drive `gen_core::load("sd3_5_large" | "sd3_5_large_turbo" |
// "sd3_5_medium")` against the gated stabilityai/* diffusers snapshots (the worker crate links + drives
// all three registered generators + the LoRA `with_adapters` apply seam), not just mlx-gen-sd3 in
// isolation.
#[cfg(all(test, target_os = "macos"))]
mod sd3_5_mlx_smoke;
// Real-weight MLX smoke for the SDXL base 1.0 Q8 worker lane (sc-8746, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("sdxl")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir.
// Closes the stale sc-1975 Q8-on-SDXL loop on-device: asserts the fixed mlx-gen Q8 path (sc-2641) renders
// non-degenerate AND specifically NOT all-zero (the retired Apple recipe's exact failure signature).
#[cfg(all(test, target_os = "macos"))]
mod sdxl_base_q8_mlx_smoke;
// Real-weight MLX train→apply smoke for the Illustrious-XL SDXL-family lane (sc-10618, epic 10609).
// Test-only + macOS-only; drives `mlx_gen_sdxl::load_trainer` from the Illustrious turnkey's dense
// `bf16/` tier, trains a tiny LoRA/LoKr, then renders WITHOUT vs WITH the adapter via
// `mlx_gen_sdxl::load(...).with_adapters` and asserts it visibly changes the output — the E2E evidence
// (not a registry entry + a green unit test) the training half of the epic demands. For LoKr it also
// asserts no `mid_block` factors were emitted (sc-2640: the SDXL LoKr surface is down/up attention only).
#[cfg(all(test, target_os = "macos"))]
mod illustrious_train_apply_mlx_smoke;
// Real-weight MLX smoke for the Lens-Turbo Q4 worker lane (sc-8763, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("lens_turbo")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/lens-turbo-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_turbo_q4_mlx_smoke;
// Real-weight MLX smoke for the recovered base Lens Q4 worker lane (sc-8767, epic 8506 Group-B).
// Test-only + macOS-only; drives `gen_core::load("lens")` with a Q4 LoadSpec against the packed `q4/`
// turnkey subdir. On-device evidence that the SceneWorks/lens-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_base_q4_mlx_smoke;
// Real-weight MLX smoke for the Chroma1-Base Q4 worker lane (sc-8777, epic 8506 Group-B). Test-only +
// macOS-only; drives `gen_core::load("chroma1_base")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/chroma1-base-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate. Chroma packs ONLY the transformer per-tier (the T5-XXL TE + VAE stay dense — chroma
// never quantizes its T5, so no denseTextEncoderTier). hd/flash share this crate + layout.
#[cfg(all(test, target_os = "macos"))]
mod chroma1_base_q4_mlx_smoke;
// Real-weight MLX smoke for the PiD 2K/4K output tier (epic 7840, sc-10054). Test-only + macOS-only;
// drives the REAL `pid_output_tier` + `pid_effective_dims` mapping then renders z_image_turbo through
// `gen_core::load(...).with_pid(pid_flux, gemma)` + `use_pid`, asserting `pidTarget:"2k"` yields a 2048²
// image (base 512 × 4) and `"4k"` yields 4096² (base 1024 × 4) — the on-device evidence that the tier
// mapping actually changes the output resolution on real weights.
#[cfg(all(test, target_os = "macos"))]
mod pid_tier_mlx_smoke;
// Real-weight MLX smoke for the SANA quant-matrix lane (sc-8489/sc-8513, epic 8506). macOS-only;
// drives `gen_core::load("sana_1600m"|"sana_sprint_1600m")` with a per-tier LoadSpec against the
// packed q4/q8 + dense bf16 turnkey subdirs. On-device evidence that the SceneWorks/Sana_*_mlx
// pre-quantized turnkeys load through the worker packed path (`STANDARD_TIER_MODELS` →
// `standard_tier_subdir` resolves the tier) and render non-degenerate at EVERY downloaded tier. SANA
// packs the Linear-DiT transformer + Gemma-2 CHI TE per-tier (DC-AE VAE dense); q4/q8 are a no-op on
// the already-packed weights (packed-detected) and bf16 loads dense (Quant::None).
#[cfg(all(test, target_os = "macos"))]
mod sana_mlx_smoke;
// On-device per-tier memory-footprint measurement harness (sc-8516, epic 8506). Test-only + macOS-only;
// #[ignore]d real-weight smokes that drive `gen_core::load(id)` + ONE generation while sampling the MLX
// process-global memory counters (mlx_rs::memory::{reset_peak_memory, get_active_memory, get_peak_memory})
// generator_cache.rs already publishes — producing measured resident + peak footprint per (model, tier)
// to calibrate the sc-8509 RAM→tier suggestion (apps/web/src/tierSuggestion.js) and backfill the sc-8508
// manifest footprint fields.
#[cfg(all(test, target_os = "macos"))]
mod footprint_measure;
// On-device build helper for the Wan2.2 T2V-A14B quant matrix (sc-9942, epic 8506). Test-only +
// macOS-only; an #[ignore]d helper that drives `mlx_gen_wan::convert::convert_t2v_14b` once per tier
// (bf16/q8/q4) against the native checkpoint to produce the self-contained hosted tier subdirs, then
// copies the tokenizer the converter omits. Run one-off to build the artifacts for
// `SceneWorks/wan2.2-t2v-a14b-mlx`; not exercised in CI (needs the ~126GB native weights).
#[cfg(all(test, target_os = "macos"))]
mod wan_t2v_14b_tier_build;
// On-device build helper for the Wan2.2 I2V-A14B quant matrix (sc-9943, epic 8506). The image→video
// sibling of the above; drives `mlx_gen_wan::convert::convert_i2v_14b` (in_dim 36 image-concat) once
// per tier (bf16/q8/q4) against the native checkpoint to produce the self-contained hosted tier
// subdirs, then copies the tokenizer the converter omits. Run one-off to build the artifacts for
// `SceneWorks/wan2.2-i2v-a14b-mlx`; not exercised in CI (needs the ~126GB native weights).
#[cfg(all(test, target_os = "macos"))]
mod wan_i2v_14b_tier_build;
// On-device build helper for the Wan2.2 TI2V-5B quant matrix (sc-9941, epic 8506). The single-expert
// sibling of the A14B helpers: drives `mlx_gen_wan::convert::convert_ti2v_5b` for the dense bf16 tier,
// then derives the q8/q4 tiers worker-side (load the bf16 `model.safetensors` →
// `quantize_wan_transformer` → save + reuse the shared dense T5/VAE/tokenizer + a `config.json` quant
// patch) — byte-identical to an inline convert, no mlx-gen change. Run one-off to build the artifacts
// for `SceneWorks/wan2.2-ti2v-5b-mlx`; not exercised in CI (needs the native checkpoint).
#[cfg(all(test, target_os = "macos"))]
mod wan_ti2v_5b_tier_build;
// On-device build helper for the Bernini quant matrix (sc-9945, epic 8506). Composite model: derives
// all three tiers (bf16/q8/q4) worker-side from the already-hosted lean bf16 snapshot — copy the dense
// remainder, quantize the planner backbone (`mlx_gen_bernini::convert::quantize_qwen_planner_backbone`)
// + both renderer experts (`mlx_gen_wan::convert::quantize_wan_transformer`), patch the two config
// sidecars. Run one-off to build the artifacts for `SceneWorks/bernini-mlx`; not exercised in CI.
#[cfg(all(test, target_os = "macos"))]
mod bernini_tier_build;
// The DWPose skeleton rasterizer is consumed only by the macOS Z-Image strict-pose
// control path; on Mac AND the off-Mac candle DWPose lane (sc-5496) it backs the
// `pose_jobs` skeleton render; on a candle-disabled box off Mac it still builds +
// unit-tests (cross-platform raster) but its items are otherwise unused — so allow
// dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod openpose_skeleton;
// Native canny edge-map preprocessor for the Fun-Controlnet-Union canny head
// (epic 8236, sc-8240). Pure CPU raster (cross-platform + testable everywhere),
// sibling of `openpose_skeleton`: arbitrary image → `ControlKind::Canny` control
// image. Consumed by the shared strict-control driver (sc-8243) on macOS AND the
// off-Mac candle strict-control trio (sc-8304); on a candle-disabled box off Mac
// it still builds + unit-tests but its items are otherwise unused — so allow
// dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod canny;
// Depth-map preprocessor for the Fun-Controlnet-Union depth head (epic 8236): arbitrary image →
// `ControlKind::Depth` control image via a Depth Anything V2 port. Sibling of `canny` /
// `openpose_skeleton`, but — unlike those pure raster preprocessors — depth needs neural
// inference, so it is backend-gated: macOS = `mlx-gen-depth` (sc-8242), off-Mac + `backend-candle`
// = `candle-gen-depth` (sc-8413, the Windows/CUDA sibling). Consumed by the shared strict-control
// driver (sc-8243 mac) AND the off-Mac candle strict-control trio (sc-8304, which wires the candle
// estimator into `preprocess_control_entry`); on a candle-disabled box off Mac the estimator stays
// unused — so allow dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod depth;
// DWPose pose detection via onnxruntime (epic 3482, sc-3487). On Mac the CoreML EP +
// on the off-Mac candle GPU-worker lane the CUDA EP (sc-5496, epic 5482) run the same
// RTMW detector in-process; on a candle-disabled box the Python rtmlib path stays the
// Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod pose_jobs;
// Control-type preprocessor registry (ControlNet Training Studio A1, sc-10160, epic 10159): the
// single `ControlKind`-keyed mapping from a target image to its condition image, wrapping the
// existing pose (pose_jobs + openpose_skeleton) / canny / depth preprocessors so train-prep
// (folder-ingest A2, bring-your-own-dataset A3) and the strict-control inference lanes render the
// condition with identical code (automatic convention-match). Cross-platform like `canny` (the
// pose/depth arms are internally backend-gated).
//
// A1 lands this registry ahead of its first non-test consumer: the folder-ingest data-prep
// pipeline (A2, sc-10161) is what resolves + drives a preprocessor over a dataset, and the
// bring-your-own-dataset adapter (A3, sc-10171) reuses it for annotated-render/convention checks.
// The studio job (B1, sc-10162) is the first real consumer — it renders conditions via this registry
// (through A2). That caller exists only on the neural-inference builds (macOS / off-Mac candle), so
// the module is fully wired there; on a candle-disabled off-Mac ("neither") build there is still no
// consumer of the neural path, so keep the dead_code allowance only for that build.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod control_preprocess;
// Folder-ingest control-dataset prep (Training Studio A2, sc-10161, epic 10159): the "create your
// own" GENERATE core — raw target images → render each condition via the A1 `control_preprocess`
// registry → square-canonical letterbox for alignment → write `(target, control, caption)` triples
// + `manifest.jsonl` in the layout the Krea control trainer (B2) and the bring-your-own adapter (A3)
// consume. Cross-platform for the canny path; pose/depth/person-filter resolve through the
// backend-gated registry/detector. The studio job (B1, sc-10162, `control_training_jobs`) is the
// first real caller — it renders conditions from an existing dataset then trains. That caller exists
// only on the neural-inference builds; on a candle-disabled off-Mac ("neither") build nothing drives
// the pipeline, so keep the dead_code allowance only for that build.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod control_dataset_prep;
// Bring-your-own-dataset ingest adapter (Training Studio A3, sc-10171, epic 10159): the second
// dataset-input path — map a PROVIDED dataset into the same on-disk layout A2 emits, skipping what
// the source supplies. Prepared pairs (target + rendered condition) are convention-validated then
// ingested as-is / normalized / regenerated via the A1 preprocessor; annotated COCO
// (person_keypoints + captions) renders the OpenPose-18 skeleton from ground-truth keypoints (no
// detection) — cross-platform. Reuses A2's square letterbox + write/manifest tail. B1 (sc-10162)
// ships only the render-from-an-existing-dataset source; wiring this bring-your-own path into the
// studio job (source provisioning + convention-warning surfacing) is a scoped follow-up, so this
// module keeps its dead_code allowance until that lands.
#[allow(dead_code)]
mod control_dataset_byo;
// ControlNet Training Studio orchestration job (B1, sc-10162, epic 10159): renders the per-image
// control condition from an existing captioned dataset via `control_preprocess`/`control_dataset_prep`
// (A1/A2), then trains the control branch through the shared `training_jobs` executor
// (`krea_control` → `krea_2_control`). Cross-platform module shell with a neural-build-gated real impl
// + a loud stub, mirroring `training_jobs`.
mod control_training_jobs;
// CUDA execution-provider dependency preloading for the off-Mac candle `ort` paths
// (sc-6209, epic 5482): `ort::ep::cuda::preload_dylibs` dlopens the CUDA-12 runtime +
// cuDNN-9 DLLs the onnxruntime CUDA EP needs, so it engages the GPU regardless of PATH
// (the Mac CoreML path needs no equivalent). Shared by pose_jobs (DWPose, sc-5496) +
// person_jobs (YOLO, sc-5498), and Real-ESRGAN (sc-5499) next — gated to the candle GPU
// lane only.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod ort_cuda;
// SCRFD 5-point face-landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac, plus the
// candle SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482) — the same
// InstantID face-stack detector reused in-process for the Key Point Library "extract kps from this
// image" capability. So the module compiles on Mac AND the candle lane; on a candle-disabled box the
// Python InsightFace path stays the Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod kps_jobs;
// Image upscaling: Real-ESRGAN (epic 3482, sc-3489) RRDBNet x2/x4 via `ort`/CoreML on Mac, plus the
// SeedVR2 one-step diffusion upscaler — native MLX on Mac (sc-4815) and the candle CUDA backend on
// Windows (sc-5928). So the module compiles on Mac AND the Windows/CUDA candle lane; the ort/CoreML
// Real-ESRGAN path inside stays Mac-gated (the Python torch Real-ESRGAN / AuraSR path is the
// Windows/Linux backend), while the SeedVR2 path is backend-neutral (`gen_core::load("seedvr2")`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod upscale_jobs;
// YOLO11 person detection + selected-person ByteTrack tracking (epic 3482, sc-3488/sc-3633;
// off-Mac candle lane sc-5498, epic 5482). Native-MLX YOLO11m on Mac, `ort`/CUDA on the off-Mac
// candle GPU-worker lane (the pure-Rust ByteTrack in `person_track` is backend-neutral). So both
// modules compile on Mac AND the candle lane; on a candle-disabled box the Python Ultralytics
// path stays the Windows/Linux backend. Person *segmentation* (SAM masks) stays Mac-only
// (`person_segment*` below) — off-Mac tracks are box-only; a candle SAM backport is epic 3792.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_jobs;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_track;
// Native-MLX SAM2 person segmentation (epic 3704, sc-3709): the `mlx-gen-sam2`
// box-prompt segmenter generates per-frame masks in `run_person_track`. macOS-only
// like person_jobs (mlx-gen builds MLX from source); the Python SAM2 path stays the
// Windows/Linux backend.
#[cfg(target_os = "macos")]
mod person_segment;
// SAM3 text-concept (PCS) person segmenter — the box-prompt-free upgrade of `person_segment`
// (epic 4910, sc-4926). macOS-only (native MLX `mlx-gen-sam3`); the off-Mac Windows/CUDA candle
// sibling is `person_segment_sam3_candle` below.
#[cfg(target_os = "macos")]
mod person_segment_sam3;
// Smart-select image segmentation (epic 6087, sc-6105): the `image_segment` job runs SAM3
// box-prompt segmentation in-process to produce an inpaint mask asset for the Image Editor.
// macOS-only like its `person_segment_sam3` (SAM3) dependency; no torch/candle image-segment path.
#[cfg(target_os = "macos")]
mod segment_jobs;
// Off-Mac candle SAM3 text-concept person segmenter (sc-6247, epic 5482 under sc-5062) — the
// Windows/CUDA sibling of `person_segment_sam3`, driving `candle-gen-sam3`'s `Sam3VideoModel` to
// replace the SAM2 box-prompt STUB in the off-Mac person-track (`media_jobs` `maskState = "missing"`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod person_segment_sam3_candle;
// Backend-neutral SAM3 person-segmentation helpers (sc-8847, F-045): the weight resolution,
// RGB→CHW normalization, and mask/association MATH shared VERBATIM by the two cfg-exclusive SAM3
// modules (`person_segment_sam3` MLX / `person_segment_sam3_candle` candle). Extracted here ONCE so a
// fix lands on both platforms and they cannot silently diverge; the per-backend files keep only their
// tensor/model/device seam. Same superset gate as `scail2_masks` (both SAM3 modules or neither).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_segment_sam3_common;
// SCAIL-2 color-coded segmentation-mask painting (epic 5439, sc-5448): turns native SAM3
// per-person masks into the palette-painted RGB masks the SCAIL-2 engine consumes. Backend-neutral
// (pure pixel painting over `AllPersonMasks`); available on both the macOS MLX lane (sc-5448) and the
// off-Mac candle lane (sc-6837, the candle SCAIL-2 sibling), each over its own SAM3 module's
// structurally-identical `AllPersonMasks`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod scail2_masks;
use downloads::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use kps_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use pose_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use upscale_jobs::*;

mod credentials;
pub use credentials::*;
mod error;
pub use error::*;
mod manifest;
pub(crate) use manifest::*;
mod paths;
pub use paths::*;
mod payload;
pub(crate) use payload::*;
mod settings;
pub use settings::*;

mod imports;
pub use imports::*;
mod progress;
pub(crate) use progress::*;
mod util;
pub use util::*;
mod preflight;
pub use preflight::*;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const DEFAULT_MAX_LORA_URL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_MODEL_URL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const DEFAULT_TRANSITION_DURATION_SECONDS: f64 = 0.5;
// One source of truth for the person-track sample cadence (sc-8914 / F-112): the sidecar
// `sampleRateFps` the media handlers record and the sampler `person_track` uses must never drift, so
// on the lanes that build `person_track` (macOS / off-Mac candle) these alias its constants directly.
// The `person_track` module is cfg'd out on the bare parity lane (no MLX, no candle), so there the
// aliases fall back to the literal values — kept in lockstep by
// `person_track_sample_constants_are_a_single_source_of_truth`, which asserts the equality on the
// lanes where both exist.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PERSON_TRACK_SAMPLE_RATE_FPS: f64 = person_track::SAMPLE_RATE_FPS;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PERSON_TRACK_MAX_SAMPLES: usize = person_track::MAX_SAMPLES;
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
const PERSON_TRACK_SAMPLE_RATE_FPS: f64 = 2.0;
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
const PERSON_TRACK_MAX_SAMPLES: usize = 24;
const PERSON_TRACK_X_DRIFT: f64 = 0.018;

#[derive(Debug, Clone, PartialEq)]
struct DiscoveredGpu {
    id: String,
    name: String,
    capabilities: Vec<WorkerCapability>,
    utilization: Option<WorkerUtilizationSnapshot>,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        // Windows has no per-child SIGTERM, so a supervised child ALSO treats stdin-EOF
        // as a graceful-shutdown request: the supervisor holds the write end of the
        // child's piped stdin and closing it (see `supervisor::terminate_child`)
        // delivers EOF here — the Windows analogue of the unix SIGTERM path (sc-11184 /
        // F-014). This trips the same sc-8845 graceful-cancel wind-down, so an in-flight
        // job posts a terminal `Canceled` instead of dying mid-GPU-write. The top-level
        // supervisor process is NOT a child (its stdin is the real console, not a
        // supervisor-held pipe), so it keeps Ctrl-C only; gate the stdin path to child
        // workers via the `SCENEWORKS_WORKER_CHILD` marker the supervisor sets.
        if std::env::var_os("SCENEWORKS_WORKER_CHILD").is_some() {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = wait_for_parent_stdin_close() => {}
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Resolve once the supervisor closes the write end of this child's stdin pipe
/// (sc-11184 / F-014) — the Windows graceful-shutdown signal, standing in for the unix
/// SIGTERM the supervisor cannot deliver per-child on Windows.
///
/// `shutdown_signal()` is awaited on every worker-loop turn, so opening a fresh reader
/// per call would risk parking a blocking thread on each turn. Instead a SINGLE
/// background reader is started once (guarded by a `OnceLock`) and its EOF result is
/// fanned out over a `watch` channel; every caller just subscribes. The channel latches
/// `true` on close and stays there, so a caller that subscribes AFTER the pipe already
/// closed still returns immediately. The reader drains stdin on std's blocking handle
/// inside `spawn_blocking` (rather than `tokio::io::stdin`, which needs the `io-std`
/// feature this crate does not enable).
#[cfg(not(unix))]
async fn wait_for_parent_stdin_close() {
    use std::sync::OnceLock;
    use tokio::sync::watch;

    static CLOSED: OnceLock<watch::Sender<bool>> = OnceLock::new();
    let sender = CLOSED.get_or_init(|| {
        let (tx, _rx) = watch::channel(false);
        let signal = tx.clone();
        // One dedicated blocking reader for the whole process, so repeated turns never
        // each park a fresh thread. Detached: dropping the JoinHandle lets it run on.
        tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut stdin = std::io::stdin();
            let mut scratch = [0_u8; 64];
            loop {
                match stdin.read(&mut scratch) {
                    // EOF: the supervisor dropped the write end → graceful shutdown.
                    Ok(0) => break,
                    // A worker child consumes stdin for nothing else, so discard any
                    // stray bytes; only closure is meaningful.
                    Ok(_) => continue,
                    // Treat a read error as closure too, so shutdown never hangs on it.
                    Err(_) => break,
                }
            }
            // `send_replace` (NOT `send`) so the latch is updated UNCONDITIONALLY even
            // when `receiver_count() == 0`: `send` returns `Err` WITHOUT storing the value
            // if no receiver is currently subscribed, and receivers only exist while a
            // `wait_for_parent_stdin_close` future is being polled. In the synchronous gap
            // between the poll-phase `select!` and the run_job `select!` no receiver is
            // subscribed, so an EOF landing in that window would be lost forever and every
            // later waiter would block on `changed()` indefinitely (the reader is
            // single-shot and has exited). `send_replace` latches `true` regardless, so the
            // next `subscribe()`'s `borrow_and_update()` observes it immediately (sc-11184).
            let _ = signal.send_replace(true);
        });
        tx
    });

    let mut receiver = sender.subscribe();
    if *receiver.borrow_and_update() {
        return;
    }
    // Wait until the reader latches `true`. `changed()` cannot error from a dropped
    // sender: the `OnceLock` holds it for the process lifetime.
    while receiver.changed().await.is_ok() {
        if *receiver.borrow() {
            return;
        }
    }
}

/// Emit a pre-built structured-event object (already carrying its `event` key) at a
/// **declared** level through the `tracing` backbone. The format-adaptive subscriber
/// renders the `{ event, level, reportedAt, ... }` line on stdout (captured into the
/// per-process log file + the in-app Logs buffer); `reportedAt` is stamped at render
/// time. Replaces the old `println!` of the same JSON so the level is now authoritative
/// rather than inferred from the line text downstream.
fn emit_event_value(level: Level, payload: Value) {
    sceneworks_core::observability::emit_event(level, payload);
}

/// Emit a structured worker event at **info** level (the per-generation lifecycle
/// events — pipeline load / inference start+complete — that the Rust MLX path mirrors
/// from the torch worker, sc-3450). `event` is injected into `payload`.
// Only the macOS image-generation path emits these today; on other targets the
// generation code is cfg'd out, so the helper would be dead code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn emit_event(event: &str, payload: Value) {
    let mut value = payload;
    if let Some(object) = value.as_object_mut() {
        object.insert("event".to_owned(), Value::String(event.to_owned()));
    }
    emit_event_value(Level::INFO, value);
}

pub async fn run() -> WorkerResult<()> {
    // Install the tracing backbone before anything emits (covers both the
    // standalone `sceneworks-rust-worker` binary and the API's GPU-worker path,
    // which both funnel here). Idempotent — a second call is a no-op.
    sceneworks_core::observability::init_logging();
    // Host mode (no HF cache env set): default HF_HOME to the shared ~/.cache/
    // huggingface so downloads land in the OS cache rather than the private data
    // dir (sc-1904 follow-up). Set before spawning child workers so they inherit
    // it; desktop/Compose already inject HF_HOME, making this a no-op there.
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        tracing::info!(
            event = "hf_home_defaulted",
            home = %home.display(),
            "rust_worker defaulting HF_HOME"
        );
    }
    let settings = Settings::from_env();
    if !settings.is_child_worker {
        if settings.gpu_id == "auto" {
            return supervise_auto_workers(settings).await;
        }
        if settings.gpu_id == "cpu" && settings.utility_workers > 1 {
            let specs = utility_worker_specs(&settings.worker_id, settings.utility_workers);
            return supervise_children(settings, specs).await;
        }
    }
    run_worker_loop(settings).await
}

pub async fn run_worker_loop(settings: Settings) -> WorkerResult<()> {
    // sc-4482 (epic 3720): log the resolved backend-neutral gen-core contract version at startup
    // so a pin skew that slips past the CI guard (`scripts/check-gen-core-skew.sh`) is
    // diagnosable from one log line. One shared contract version backs every linked backend.
    tracing::info!(
        event = "gen_core_contract_version",
        version = %gen_core::VERSION,
        gpuId = %settings.gpu_id,
        "rust_worker gen-core contract version"
    );
    // sc-7820 (epic 7819): apply the user's GPU memory ceiling to the MLX runtime once at startup,
    // before any model load. The MLX limit is process-global, so this single call covers
    // generations, upscales, AND LoRA training. No-op when unset (0) and on non-macOS/candle builds.
    generator_cache::apply_gpu_memory_limit(settings.gpu_memory_limit_bytes);
    // sc-7825 (epic 7819): on the MLX GPU worker only, publish live MLX memory telemetry to the
    // shared config dir for the Settings readout. Gated to `mlx` so the CPU utility workers (which
    // do no MLX work) don't clobber the file with zeros.
    if settings.gpu_id == "mlx" {
        generator_cache::spawn_gpu_telemetry(settings.config_dir.clone());
    }
    let gpu = discover_gpu(&settings).await;
    let api = ApiClient::new(&settings);
    let http_client = crate::downloads::streaming_download_client();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    let mut lock_failures = 0_u32;
    let mut idle_heartbeat = IdleHeartbeat::new(progress_report_interval(&settings));
    loop {
        // sc-8845 (F-043): shutdown is observed ONLY here, around the claim / idle-sleep phase —
        // NOT around full job execution. `poll_once` does no long GPU work (memory-sync, idle
        // heartbeat, the transactional claim POST, and the idle sleep), so racing it against
        // shutdown and dropping it at an await point loses no job state: nothing is claimed on the
        // idle path, and a claimed job is handled below OUTSIDE this select. Before this change the
        // select raced the WHOLE `poll_once` (claim + the entire job) against shutdown, so a
        // graceful quit mid-job dropped the in-flight future at an arbitrary await, left the job
        // `running` until the 90s stale sweep marked it `interrupted`, and killed spawn_blocking GPU
        // work mid-write. Now a mid-job shutdown trips the job's cancel and posts a prompt terminal
        // `Canceled` (see `run_job_with_shutdown`).
        let claim = tokio::select! {
            result = poll_once(&api, &settings, &mut idle_heartbeat) => result,
            _ = shutdown_signal() => {
                // Clean-idle shutdown: no job in flight, so the pre-existing Offline heartbeat +
                // return is preserved exactly.
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        };
        match claim {
            Ok(None) => lock_failures = 0,
            Ok(Some(job)) => {
                lock_failures = 0;
                // Execute the claimed job WITHOUT racing (and dropping) the whole future against
                // shutdown. `run_job_with_shutdown` supervises execution: on a mid-job shutdown it
                // trips the job's cancel flag, lets the in-flight future wind down (never dropped
                // mid-write), and posts a terminal `Canceled` for the job before returning
                // `ShutdownDuringJob` so the loop exits with the job in a prompt terminal state.
                match run_job_with_shutdown(&api, &settings, &http_client, job).await {
                    // `run_utility_job` already posts a terminal Idle heartbeat at its end, so the
                    // scheduler should treat that as the just-sent one and wait a full interval —
                    // marking it *due* here made the next `poll_once` fire a redundant second Idle
                    // heartbeat right away (sc-8952).
                    JobOutcome::Completed => idle_heartbeat.mark_sent(),
                    JobOutcome::ShutdownDuringJob => {
                        let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                        return Ok(());
                    }
                }
            }
            Err(error) if is_database_locked(&error) => {
                // SQLite claim contention. With busy_timeout + BEGIN IMMEDIATE in the
                // store this should be rare, but back off (instead of hammering at the
                // flat poll interval) and make it visible so an MLX-eligible job lost to
                // lock contention is explained rather than silently retried into torch.
                lock_failures = lock_failures.saturating_add(1);
                let delay = retry_delay(settings.poll_seconds, lock_failures);
                emit_event_value(
                    Level::WARN,
                    json!({
                        "event": "claim_lock_contention",
                        "workerId": settings.worker_id,
                        "gpuId": settings.gpu_id,
                        "consecutiveFailures": lock_failures,
                        "retryInSeconds": delay,
                        "error": error.to_string(),
                    }),
                );
                // The back-off sleep is a between-jobs wait, so it too must observe shutdown rather
                // than blocking a graceful quit for up to `delay` seconds.
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                    _ = shutdown_signal() => {
                        let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                        return Ok(());
                    }
                }
            }
            Err(error) => {
                lock_failures = 0;
                tracing::error!(
                    event = "rust_worker_poll_failed",
                    error = %error,
                    "worker claim poll failed"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))) => {}
                    _ = shutdown_signal() => {
                        let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// Outcome of supervising one claimed job through [`run_job_with_shutdown`].
enum JobOutcome {
    /// The job ran to its own terminal state (success, failure, or user cancel) with no shutdown
    /// observed; the loop continues to the next claim.
    Completed,
    /// SIGTERM/Ctrl-C arrived while the job was in flight. The job's cancel flag was tripped, the
    /// in-flight future was awaited to wind-down (never dropped mid-write), and a terminal
    /// `Canceled` was posted for it. The loop must now exit.
    ShutdownDuringJob,
}

/// Run one claimed job while keeping shutdown observable WITHOUT dropping the in-flight future
/// (sc-8845, F-043).
///
/// The whole-`poll_once`-vs-shutdown `select!` this replaces cancelled the job future at an
/// arbitrary await on a graceful quit: no terminal job-state write happened, so the claimed job sat
/// `running` until the API's 90s stale sweep relabelled it `interrupted`, and any `spawn_blocking`
/// GPU work was killed mid-write (partial outputs left behind). Here, execution is bound to a
/// process-shutdown [`CancelFlag`]; on shutdown we:
///   1. trip the flag so handlers that thread it (the generate/edit/detail/video/upscale/train
///      paths via `run_utility_job`) stop at their next checkpoint instead of running to natural
///      end, then
///   2. keep awaiting the SAME job future — it is never dropped, so no write is interrupted — for up
///      to `shutdown_timeout_seconds`, then
///   3. post a terminal `Canceled` for the job (unless the handler already wrote a terminal state
///      itself), so the job lands a prompt, specific terminal state instead of a delayed generic
///      `interrupted`.
///
/// The bounded wait guarantees a graceful quit is never blocked indefinitely by an un-interruptible
/// compute path: if the future has not resolved by the grace window we still post `Canceled` and
/// return, having already tripped the flag so the underlying task winds down.
async fn run_job_with_shutdown(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
) -> JobOutcome {
    let job_id = job.id.clone();
    let shutdown = gen_core::CancelFlag::new();
    let job_future = run_utility_job(api, settings, http_client, job, shutdown.clone());
    tokio::pin!(job_future);

    tokio::select! {
        () = &mut job_future => return JobOutcome::Completed,
        _ = shutdown_signal() => {}
    }

    // Shutdown fired mid-job. Trip the shared flag so a handler that observes it winds down
    // promptly, then AWAIT the same (un-dropped) future to its checkpoint / natural end, bounded by
    // the grace window so an un-interruptible path can't hang the quit.
    emit_event_value(
        Level::WARN,
        json!({
            "event": "worker_shutdown_during_job",
            "workerId": settings.worker_id,
            "gpuId": settings.gpu_id,
            "jobId": job_id,
        }),
    );
    shutdown.cancel();
    let grace = Duration::from_secs(settings.shutdown_timeout_seconds.max(1));
    let _ = tokio::time::timeout(grace, &mut job_future).await;
    // Post the terminal `Canceled`. If the handler already wrote its own terminal state (it observed
    // the flag and posted `Canceled`, or completed/failed in the race window) the API rejects this
    // as a no-op/409 — harmless; the point is that the job never dangles `running`.
    let _ = mark_job_canceled(api, &job_id, "Worker shut down before the job completed.").await;
    JobOutcome::ShutdownDuringJob
}

/// True when an error ultimately stems from SQLite reporting the jobs database as locked.
/// The claim travels worker→API→store, so a lock surfaces as an `Api { detail }` whose
/// message embeds the SQLite text; match on the rendered string rather than a typed variant.
fn is_database_locked(error: &WorkerError) -> bool {
    error
        .to_string()
        .to_ascii_lowercase()
        .contains("database is locked")
}

async fn register_worker_with_retry(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<()> {
    let mut attempt = 0_u32;
    loop {
        match register_worker(api, settings, gpu).await {
            Ok(_) => return Ok(()),
            Err(error) => {
                attempt = attempt.saturating_add(1);
                let delay = retry_delay(settings.poll_seconds, attempt);
                tracing::warn!(
                    event = "rust_worker_register_failed",
                    attempt,
                    retryInSeconds = delay,
                    error = %error,
                    "worker registration failed; will retry"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                    _ = shutdown_signal() => return Err(WorkerError::Canceled(
                        "Worker shutdown requested before registration completed.".to_owned(),
                    )),
                }
            }
        }
    }
}

/// The claim / idle phase of one loop turn (sc-8845, F-043). Returns the claimed job for the caller
/// to execute (outside the shutdown `select!`), or `None` when nothing was claimed (already having
/// slept the idle poll interval). Deliberately does NO job execution: the caller races only THIS
/// future against shutdown, so a graceful quit between jobs drops nothing load-bearing (no claimed
/// job, no GPU work — just the memory-sync, idle heartbeat, transactional claim POST, and idle
/// sleep). Job execution is supervised separately by `run_job_with_shutdown`.
async fn poll_once(
    api: &ApiClient,
    settings: &Settings,
    idle_heartbeat: &mut IdleHeartbeat,
) -> WorkerResult<Option<JobSnapshot>> {
    // sc-7824 (epic 7819): pick up a live GPU-memory-limit change here, before claiming the next
    // job, so a Settings slider move applies between jobs (not mid-flight) with no worker restart.
    // No-op unless this is the MLX worker and the desktop has written the live-handoff file.
    if settings.gpu_id == "mlx" {
        generator_cache::sync_gpu_memory_limit(&settings.config_dir);
    }
    if idle_heartbeat.should_send() {
        heartbeat(api, settings, WorkerStatus::Idle, None).await?;
        idle_heartbeat.mark_sent();
    }
    let claim: ClaimResponse = api
        .post_json(
            "/api/v1/jobs/claim",
            &ClaimRequest {
                worker_id: settings.worker_id.clone(),
                extra: BTreeMap::new(),
            },
        )
        .await?;
    let Some(job) = claim.job else {
        tokio::time::sleep(Duration::from_secs(settings.poll_seconds)).await;
        return Ok(None);
    };
    Ok(Some(job))
}

struct IdleHeartbeat {
    interval: Duration,
    next_due: Instant,
}

impl IdleHeartbeat {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_due: Instant::now(),
        }
    }

    fn should_send(&self) -> bool {
        Instant::now() >= self.next_due
    }

    fn mark_sent(&mut self) {
        self.next_due = Instant::now() + self.interval;
    }
}

async fn register_worker(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<WorkerSnapshot> {
    api.post_json(
        "/api/v1/workers/register",
        &WorkerRegisterRequest {
            worker_id: settings.worker_id.clone(),
            gpu_id: gpu.id.clone(),
            gpu_name: Some(gpu.name.clone()),
            capabilities: worker_capabilities(gpu),
            loaded_models: Vec::new(),
            utilization: gpu.utilization.clone(),
            extra: BTreeMap::new(),
        },
    )
    .await
}

/// Post a worker heartbeat. A transport-level failure (`WorkerError::Http`: the API
/// is briefly unreachable — a restart, a transient network blip) is logged and
/// swallowed rather than propagated: a running job must not be torn down for
/// telemetry we can simply resend. The next heartbeat (≤15s) refreshes the worker's
/// `last_seen` well inside the API's stale-sweep window (default 90s), so a brief
/// outage no longer false-positives a live job to `interrupted`; a sustained outage
/// (> the timeout) still lets the sweep fire — the API stays the authority on
/// declaring a worker gone. A non-transport error (the API answered and rejected
/// the heartbeat, e.g. the worker is no longer registered) is a real signal and is
/// still propagated. (sc-6320)
pub(crate) async fn heartbeat(
    api: &ApiClient,
    settings: &Settings,
    status: WorkerStatus,
    current_job_id: Option<&str>,
) -> WorkerResult<()> {
    // Capture the label before `status` is moved into the request, for the log line.
    let status_label = status.as_str().to_owned();
    let outcome: WorkerResult<WorkerSnapshot> = api
        .post_json(
            &format!("/api/v1/workers/{}/heartbeat", settings.worker_id),
            &WorkerHeartbeatRequest {
                status,
                current_job_id: current_job_id.map(str::to_owned),
                loaded_models: Vec::new(),
                utilization: gpu_utilization(&settings.gpu_id).await,
                extra: BTreeMap::new(),
            },
        )
        .await;
    match outcome {
        Ok(_) => Ok(()),
        Err(WorkerError::Http(error)) => {
            emit_event_value(
                Level::ERROR,
                json!({
                    "event": "worker_heartbeat_transport_failed",
                    "workerId": settings.worker_id,
                    "jobId": current_job_id,
                    "status": status_label,
                    "error": error.to_string(),
                }),
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
}

/// Dispatch one claimed job to its handler and reconcile the terminal state.
///
/// `shutdown` (sc-8845, F-043) is the process-shutdown [`CancelFlag`] tripped by
/// `run_job_with_shutdown` when SIGTERM/Ctrl-C arrives mid-job. The caller's
/// bounded-wait-then-terminal-`Canceled` write is what GUARANTEES no job dangles `running` even for a
/// handler that cannot observe the flag. The always-compiled placeholder path threads it directly so
/// the shutdown-during-job behavior is exercised on every target (the GPU handlers are macOS/candle-
/// gated). sc-9618 (F-043 follow-up): the dispatch below is scoped in [`with_shutdown_flag`], binding
/// the flag as a task-local so the per-engine GPU consumer loops (image `consume_gen_events`, video
/// `generate_video`, training `consume_training_events`, the shared `run_batched_analysis_job`, and the
/// image-detail loop) consult it via `shutdown_requested()` at their existing per-step cancel
/// checkpoints — tripping the engine cancel mid-step on quit instead of winding down at the grace
/// window. MLX and candle twins stay in sync because both funnel through those SAME shared consumers.
async fn run_utility_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
    shutdown: gen_core::CancelFlag,
) {
    // Bind the process-shutdown flag as a task-local for the whole dispatch (sc-9618, F-043 follow-up)
    // so the per-engine GPU consumer loops awaited below honor it at their per-step cancel checkpoints
    // (via `shutdown_requested()`), stopping a gen/prompt mid-step on quit instead of waiting out the
    // grace window — without threading the flag through every stream-handler signature. The placeholder
    // path keeps its explicit `&shutdown` (it's the always-compiled reference implementation).
    // Per-run metrics probe (epic 10402, sc-10404): reset the MLX peak-memory
    // window and start the background GPU-load/memory sampler before the job
    // runs, so peak memory + peak load cover the whole job. Consumed after the
    // handler returns to POST the hardware/timing block; settings + phase
    // timings are posted separately by the handlers and coalesce-merge server-side.
    let metrics_probe = job_metrics::JobMetricsProbe::start(&settings.gpu_id);
    let result = with_shutdown_flag(shutdown.clone(), async {
        match job.job_type {
            JobType::Placeholder => run_placeholder_job(api, settings, &job, &shutdown)
                .await
                .map_err(|error| ("Placeholder job failed.", error)),
            // Native MLX image generation, served in-process by the linked mlx-gen
            // engine on the macOS Apple-Silicon GPU worker (epic 3018). Off macOS the
            // capability is never advertised, so this arm is unreachable there.
            JobType::ImageGenerate => run_image_generate_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Image generation failed.", error)),
            // Plain Image Edit (sc-3513): the distinct `image_edit` job type (`mode=edit_image`
            // + `sourceAssetId`, epic 2427) shares the generate handler — it dispatches on
            // payload model+mode (qwen/flux2/sdxl edit streams), not job type. The API only
            // routes MLX-eligible edit models here (jobs_store::image_job_is_mlx_eligible); off
            // macOS the `image_edit` capability is never advertised, so this arm is unreachable.
            JobType::ImageEdit => run_image_generate_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Image edit failed.", error)),
            // Native MLX tile-ControlNet detail refine (epic 3041, sc-3060), served in-process
            // by the engine on the macOS Apple-Silicon GPU worker. Off macOS the capability is
            // never advertised, so this arm is unreachable there (image_detail runs on torch).
            JobType::ImageDetail => run_image_detail_job(api, settings, &job)
                .await
                .map_err(|error| ("Image detail enhancement failed.", error)),
            // SenseNova-U1 visual question answering + Document Studio interleave (epic 3180,
            // sc-3905). These bypass the `Generator` registry and call the concrete `T2iModel`
            // directly (text / text+images output the `GenerationOutput` contract can't express).
            // The API routes them here only on Mac (`understanding_job_is_mlx_eligible`); off macOS
            // the `image_vqa`/`image_interleave` capabilities are never advertised, so these arms
            // are unreachable there (the Python torch worker serves them on Windows/Linux).
            JobType::ImageVqa => run_vqa_job(api, settings, &job)
                .await
                .map_err(|error| ("Visual question answering failed.", error)),
            JobType::ImageInterleave => run_interleave_job(api, settings, &job)
                .await
                .map_err(|error| ("Interleaved generation failed.", error)),
            // Native MLX video generation, served in-process by the linked mlx-gen engine
            // on the macOS Apple-Silicon GPU worker (epic 3018). sc-3033 ships the runtime
            // + procedural stub; the real Wan (sc-3034) / LTX+audio (sc-3035) models link
            // their provider crates. Off macOS the capability is never advertised, so this
            // arm is unreachable there.
            // The clip-conditioning advanced video modes (epic 3040, sc-3522) share the video
            // generation handler — `run_video_generate_job` dispatches `extend_clip` /
            // `video_bridge` by the request `mode` into the LTX IC-LoRA `VideoClip` path. The API
            // only routes the LTX-eligible jobs here (`video_job_is_mlx_eligible`); off macOS the
            // VideoExtend/VideoBridge capabilities are never advertised, so these arms are
            // unreachable there (the procedural stub would otherwise ignore the conditioning).
            JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
                run_video_generate_job(api, settings, &job)
                    .await
                    .map_err(|error| ("Video generation failed.", error))
            }
            // replace_person → native Wan-VACE (epic 3040, sc-3521): the `PersonReplace` job
            // type (and `video_generate` mode=`replace_person`) shares the video handler, which
            // dispatches on `mode == "replace_person"` to the engine `wan_vace` provider — the
            // native equivalent of the torch `WanVACEPipeline` path. The API routes only
            // MLX-eligible replace_person jobs here (`jobs_store::video_job_is_mlx_eligible`);
            // off macOS the `person_replace` capability is never advertised, so this arm only
            // produces a real video on the macOS MLX worker (and the Python torch path serves
            // Windows/Linux + non-VACE replacement).
            JobType::PersonReplace => run_video_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Person replacement failed.", error)),
            // Native MLX LoRA/LoKr training (epic 3039, sc-3043/3049), served in-process
            // by the linked mlx-gen engine on the macOS Apple-Silicon GPU worker. The API
            // routes only MLX-native families here (jobs_store::training_job_is_mlx_eligible);
            // kolors/lens + LoKr-on-Wan stay on the Python torch worker, which is also the
            // Windows/Linux path. Off macOS the execute capability is never advertised.
            JobType::LoraTrain => run_lora_train_job(api, settings, &job)
                .await
                .map_err(|error| ("LoRA training failed.", error)),
            // ControlNet Training Studio (epic 10159, sc-10162): render the per-image control
            // condition from the plan's dataset (A1/A2), then train the control branch through the
            // same native executor as LoRA (`krea_control` → `krea_2_control`). Candle-only today; the
            // routing gate keeps it on a candle worker (or the linked mlx build), the stub fails loudly
            // elsewhere.
            JobType::ControlTraining => {
                control_training_jobs::run_control_training_job(api, settings, http_client, &job)
                    .await
                    .map_err(|error| ("ControlNet training failed.", error))
            }
            // Native MLX JoyCaption dataset captioning (epic 3550, sc-3556). The API
            // routes only `captioner=joy_caption` jobs here; Windows/Linux and
            // explicit non-MLX GPU choices keep the Python torch captioner fallback.
            JobType::TrainingCaption => run_training_caption_job(api, settings, &job)
                .await
                .map_err(|error| ("Training captioning failed.", error)),
            // Dataset Doctor CLIP-embedding analysis (sc-6535): the macOS MLX worker embeds every dataset
            // image (clip_vit_l14) and POSTs the content-hash sidecar; off-Mac the handler returns a
            // precise unsupported error (no candle CLIP embedder yet).
            JobType::DatasetAnalysis => run_dataset_analysis_job(api, settings, &job)
                .await
                .map_err(|error| ("Dataset analysis failed.", error)),
            // Dataset Doctor face pass (sc-6538): the native SCRFD+ArcFace stack embeds the largest face of
            // each Person-dataset image and POSTs the face sidecar. MLX on Mac (`mlx-gen-face`), candle on
            // the candle lane; off both the handler returns a precise unsupported error.
            JobType::DatasetFaceAnalysis => run_dataset_face_analysis_job(api, settings, &job)
                .await
                .map_err(|error| ("Dataset face analysis failed.", error)),
            // On-demand "compare image to another" likeness tool (sc-4415): scores a CANDIDATE asset
            // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer. MLX on Mac,
            // candle off-Mac; off both the handler returns a precise unsupported error. Like the
            // dataset-face pass, the job-type capability is gpu.rs-hardcoded (the face stack has no gen-core
            // registry), so a job stays queued rather than mis-claimed where the stack isn't linked.
            JobType::FaceLikenessCompare => run_face_likeness_compare_job(api, settings, &job)
                .await
                .map_err(|error| ("Face likeness compare failed.", error)),
            // Native candle prompt refinement (epic 5095, sc-5525; consolidated onto candle-llm in sc-7404):
            // routes `prompt_refine` to the candle `core_llm::TextLlm` provider (candle-llama, resolved
            // model-first). The candle worker advertises `prompt_refine` only when `backend_candle_enabled`
            // (engines::registry_capabilities from the registered core_llm provider); off the Windows candle
            // build the capability is never advertised, so this arm is unreachable there and the Python torch
            // refiner serves the job (sc-5525 keeps it as the Mac + default-installer fallback).
            JobType::PromptRefine => run_prompt_refine_job(api, settings, &job)
                .await
                .map_err(|error| ("Prompt refinement failed.", error)),
            JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Model download failed.", error)),
            JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("LoRA import failed.", error)),
            JobType::LoraDownload => run_lora_download_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("LoRA download failed.", error)),
            JobType::ModelImport => run_model_import_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Model import failed.", error)),
            JobType::ModelConvert => run_model_convert_job(api, settings, &job)
                .await
                .map_err(|error| ("Model conversion failed.", error)),
            JobType::FrameExtract => run_frame_extract_job(api, settings, &job)
                .await
                .map_err(|error| ("Frame extraction failed.", error)),
            JobType::TimelineExport => run_timeline_export_job(api, settings, &job)
                .await
                .map_err(|error| ("Timeline export failed.", error)),
            JobType::PersonDetect => run_person_detect_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Person detection failed.", error)),
            // DWPose whole-body pose detection (epic 3482, sc-3487 Mac / sc-5496 off-Mac):
            // RTMW via onnxruntime, replacing the Python rtmlib path — CoreML EP on the
            // macOS MLX worker, CUDA EP on the off-Mac candle GPU worker. Available on Mac
            // AND the candle lane; on a candle-disabled box `PoseDetect` is never advertised
            // by the Rust worker (the Python worker handles it), so this falls to the `_`
            // arm there.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::PoseDetect => run_pose_detect_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Pose detection failed.", error)),
            // SCRFD 5-point landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac + the candle
            // SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482), served in-process
            // for the Key Point Library. Available on Mac AND the candle lane; on a candle-disabled box
            // `KpsExtract` is never advertised by the Rust worker (the Python InsightFace path handles it),
            // so this falls to the `_` arm there.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::KpsExtract => run_kps_extract_job(api, settings, &job)
                .await
                .map_err(|error| ("Keypoint extraction failed.", error)),
            // Image upscaling, served in-process by `upscale_jobs::run_image_upscale_job`: Real-ESRGAN
            // RRDBNet x2/x4 via onnxruntime/CoreML (epic 3482, sc-3489, Mac) + SeedVR2 one-step diffusion
            // (native MLX on Mac sc-4815 / candle CUDA on Windows sc-5928). Available on Mac AND the
            // Windows/CUDA candle lane; on a plain Windows/Linux box `ImageUpscale` is never advertised by
            // the Rust worker, so it falls to the `_` arm (Python Real-ESRGAN/AuraSR). The routing oracle
            // refuses `engine=seedvr2` on torch and `engine=real-esrgan`/`aura-sr` on the candle worker.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::ImageUpscale => run_image_upscale_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Image upscale failed.", error)),
            // Dataset Doctor one-tap upscale (sc-6539): Real-ESRGAN over flagged low-res items, then
            // re-point each via the API. Same engine + worker lanes as image_upscale.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::DatasetUpscale => run_dataset_upscale_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Dataset upscale failed.", error)),
            // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt segmentation,
            // served in-process by `segment_jobs::run_image_segment_job` — a box prompt → a binary
            // inpaint mask asset for the Image Editor. macOS-only (the capability is advertised only by
            // `mlx_gpu`), so off-Mac this arm is absent and a segment job is never claimed there.
            #[cfg(target_os = "macos")]
            JobType::ImageSegment => {
                segment_jobs::run_image_segment_job(api, settings, http_client, &job)
                    .await
                    .map_err(|error| ("Smart-select segmentation failed.", error))
            }
            // SeedVR2 video upscaling (epic 4811): one-step super-resolution — native MLX on Mac (sc-4816)
            // / candle CUDA on Windows (sc-5928). SceneWorks' first video upscaler: decodes the source
            // clip, runs the temporal-chunked 5D upscale, re-encodes, and passes the source audio through.
            // Available on Mac + the Windows/CUDA candle lane; elsewhere `VideoUpscale` is never advertised
            // (no torch path), so it falls to the `_` arm and the routing oracle reports it unsupported.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::VideoUpscale => run_video_upscale_job(api, settings, &job)
                .await
                .map_err(|error| ("Video upscale failed.", error)),
            JobType::PersonTrack => run_person_track_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Person tracking failed.", error)),
            _ => {
                let result = fail_job(
                    api,
                    &job.id,
                    "No utility exists for this job type.",
                    Some(format!(
                        "Unsupported utility job type: {}",
                        job.job_type.as_str()
                    )),
                )
                .await;
                result.map_err(|error| ("Utility job failed.", error))
            }
        }
    })
    .await;
    if matches!(job.job_type, JobType::LoraImport | JobType::ModelImport) {
        let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    }
    if let Err((message, error)) = result {
        match error {
            WorkerError::Canceled(_) => {}
            error => {
                let _ = fail_job(api, &job.id, message, Some(error.to_string())).await;
                tracing::error!(
                    event = "utility_job_failed",
                    jobId = %job.id,
                    error = %error,
                    "{message}"
                );
            }
        }
    }
    // Capture + POST the run's hardware metrics for every job type — including
    // failed/canceled runs, which still carry a meaningful peak + wall-clock
    // (epic 10402, sc-10404). Best-effort: never fails the job.
    let metrics = metrics_probe.finish().await;
    job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
    let _ = heartbeat(api, settings, WorkerStatus::Idle, None).await;
}

async fn run_placeholder_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    shutdown: &gen_core::CancelFlag,
) -> WorkerResult<()> {
    let stages = [
        (
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Preparing placeholder job.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.35,
            "Running placeholder step 1.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.65,
            "Running placeholder step 2.",
        ),
        (
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving placeholder result.",
        ),
    ];

    for (status, stage, progress, message) in stages {
        // sc-8845 (F-043): a process shutdown mid-job is a cancel checkpoint too — trip the same
        // terminal `Canceled` write as a user cancel so the job lands a prompt terminal state
        // instead of being dropped `running`. Checked before the user-cancel GET so a graceful quit
        // is honored even if the snapshot fetch is momentarily failing.
        let shutting_down = shutdown.is_cancelled();
        let snapshot_cancel = if shutting_down {
            false
        } else {
            let snapshot: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{}", job.id)).await?;
            snapshot.cancel_requested
        };
        if shutting_down || snapshot_cancel {
            let message = if shutting_down {
                "Worker shut down before the job completed."
            } else {
                "Worker canceled the job before completion."
            };
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Canceled,
                    ProgressStage::Canceled,
                    progress,
                    message,
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            return Err(WorkerError::Canceled(message.to_owned()));
        }

        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        update_job(
            api,
            &job.id,
            progress_payload(status, stage, progress, message, None, None, None),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    let mut result = JsonObject::new();
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    result.insert("output".to_owned(), Value::String("placeholder".to_owned()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Placeholder job completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

fn progress_report_interval(settings: &Settings) -> Duration {
    Duration::from_secs(settings.heartbeat_seconds.clamp(5, 15))
}

fn retry_delay(poll_seconds: u64, attempt: u32) -> u64 {
    let multiplier = 2_u64.saturating_pow(attempt.saturating_sub(1).min(4));
    poll_seconds.max(1).saturating_mul(multiplier).clamp(1, 30)
}

#[cfg(test)]
mod tests;
