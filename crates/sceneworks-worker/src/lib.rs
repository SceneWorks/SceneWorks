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
    apply_adapter_metadata_to_manifest_entry, apply_model_manifest_defaults, detect_model_family,
    first_safetensors_path, inspect_adapter_in_dir, reconcile_detected_family,
    validate_minimax_h3_trainer_header, FamilyMismatch, SafetensorsHeaderError,
};
// Only the cfg-gated adapter resolvers (image `resolve_adapters` / `classify_adapter`, video
// `resolve_lora_file`) use these, so gate the import identically or the parity
// build (no `backend-candle`) trips `-D unused-imports` (sc-10221).
//
// `read_safetensors_header` joined this block in sc-14057: the LoRA-import path used to read the
// header itself, and now delegates to `lora_family::inspect_adapter_in_dir`, leaving
// `classify_adapter` as the only caller — which is gated.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::lora_family::read_safetensors_header;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
use sceneworks_core::lora_family::resolve_adapter_in_dir;
use sceneworks_core::lora_url::{
    lora_source_url_file_name, lora_source_url_file_stem, parse_lora_source_url_with_private,
    validate_lora_url_host, validate_public_ip,
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
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod asset_media;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod image_sampling;
// Shared one-child PNG persistence for upscale (Mac + candle) and smart-select (Mac).
// Keep the include site on the callers' superset so the neither-backend lane does not
// compile an otherwise dead helper — plus `test`, because this seam is where the standalone
// upscale's embedded workflow is written (sc-15948) and that contract should be asserted on every
// lane rather than only where an upscaler backend compiles. The non-test build is unchanged.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
mod single_child_asset;
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
// The single pre-loader model-source guard (sc-19708): every dispatched job's generic model
// carriers are reduced to this platform's exact requirement closure and judged through the one
// shared availability resolver before any handler constructs a loader.
mod external_library_runtime;
mod inference_runtime;
mod text_encoder_selection;
pub use text_encoder_selection::{
    image_text_encoder_options, resolve_image_text_encoder_selection, ImageTextEncoderOption,
};
// Promotion activation (sc-19706): the idle-time producer that turns a successful source-tier load
// into an app-owned resolved bundle. The guard above records an I/O-free hint; this drains it.
mod resolved_cache_promotion;
// Backend-neutral generator load/run cache (epic 3720, sc-3724). Typed entirely against
// `gen_core::*` (no tensor types leak), so it links on ALL targets — the production load seam
// (`with_cached_generator`) is reached only from the macOS image/video paths, but the all-targets
// stub test exercises the load→progress→cancel→output contract with no backend linked. Off macOS
// the production caller is cfg'd out, so allow dead_code there (the engines.rs precedent).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod generator_cache;
// sc-21027: MiniMax-H3's FL2VA provider labels its two conditioning-force boundaries. A Metal
// watchdog timeout there poisons this process's command queue, so record the first timeout, refuse
// the next claim, and clean-exit exactly once for the existing auto-worker supervisor to replace.
// All-targets so the lifecycle and ordering regressions run weights-free on neither/Candle lanes.
mod mlx_worker_recovery;
// Request-scoped execution planning (sc-18317, epic 18304 P2): the warm-hit execution-policy
// decision the `LoadIdentity`/`ExecutionPolicy` split (sc-18305) left owing, plus selection of
// gen-core's typed execution domains (graph-eval cadence, FFN chunk, CFG batching) from what each
// provider declares. Backend-neutral and typed entirely against `gen_core::*`, so it links on all
// targets exactly like `generator_cache`; off macOS its production callers are cfg'd out.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod execution_planner;
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
// Stage 1 of the engine-capability pipeline (sc-16965, epic 16948): the weights-free dumper that
// turns the LINKED provider registry into checked-in, per-backend facts files that stage 2's
// generator + vitest drift guard read on every PR. `pub` because `src/bin/dump-engine-capabilities`
// is a separate crate and can only reach the public surface. All-targets on purpose: the
// empty-registry refusal is the whole point of the module, so it must compile — and be unit-tested —
// on the lanes that link no engines at all.
pub mod engine_capability_facts;
mod gpu;
pub mod memory_route_registry;
use gpu::*;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod candle_memory_strategy;
mod fit_gate;
// Margin constants derived from repeat-capture variance in the calibration evidence (sc-18094,
// epic 18093). Consumed by the stale-closure widening (sc-18095) and estimate-backed admission
// (sc-18096/18097) follow-ups; pinned to `scripts/derive-ladder-margins.mjs` by
// `scripts/derive-ladder-margins.test.mjs`.
pub mod ladder_margin_policy;
pub mod memory_strategy;
// The worker half of the VIDEO memory gate (sc-18814, epic 18803): implements
// `sceneworks_core::video_request`'s selector seam by calling `memory_strategy::select_strategy`.
// Cross-platform on purpose — the video lane spans both backends and the gate must not imply the
// two are symmetric, so one module serves both and names the difference explicitly.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod mlx_fit_gate;
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
mod video_admission;
// The full base fine-tune memory-envelope gate (sc-14056) lives beside the generation MLX fit gate
// (it reuses that module's byte-summing + unified-memory budget probe). Re-exported for the rust-api
// training submit gate, which calls it alongside `training_base_model_status`/`training_disk_space_error`.
pub use mlx_fit_gate::full_finetune_memory_error;
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
// Candle conditioning-overlay admission gate (sc-16069, epic 15448). Every candle route that overlays a
// second network on the base — ControlNet / IP-Adapter / identity encoder — is diverted around BOTH the
// `generate_candle_stream` `vram_gate` and the `generator_cache` `apply_residency_policy`, so before this
// eleven of them allocated with no pre-flight check at all. Same candle cfg as `vram_gate` (its consumers,
// the `image_jobs` conditioning lanes, are under that cfg) so its pub(crate) helpers aren't dead code
// under `-D warnings` on the non-candle / macOS builds.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod conditioning_fit;
use supervisor::*;
mod model_jobs;
pub use model_jobs::recover_stranded_model_conversions;
// The convert pre-flight the rust-api calls before queueing a `model_convert` job, so a convert
// requested against a still-downloading source is refused at the request boundary instead of failing
// in the worker (see `convert_source_state`).
use model_jobs::*;
pub use model_jobs::{convert_source_state, ConvertSourceState};
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
// to candle for eligible shapes or remains queued, and neither path applies the caption guard, so they read
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
pub use video_jobs::{text_encoder_options_for_adapter, TextEncoderOption};
// Pure audio generation — the SceneWorks Audio Studio job path (epic 13400 / sc-13404). Compiled on
// every platform (the dispatch arm is uniform); the actual candle audio lane is resolved through
// `inference_runtime::load_audio`, which errors clearly on a build that ships no audio registry (a
// non-native desktop worker never advertises `audio_generate`, so the arm is unreachable there).
mod audio_jobs;
use audio_jobs::*;
// The Voice Clone "register a voice" embed path (sc-13517): the rust-api calls
// `voice_register::embed_reference_clip` to compute a reference clip's Chatterbox-VE speaker vector
// for the saved-voice registry. Public because it is invoked from another crate (rust-api), not the
// worker job loop.
pub mod voice_register;
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
// LAION-style URL/caption Parquet materialization belongs to the non-GPU
// utility worker; the existing ControlNet trainer consumes the resulting
// ordinary SceneWorks dataset.
mod dataset_parquet_jobs;
use dataset_parquet_jobs::*;
pub mod catalog_analysis;
pub mod catalog_image_fetch;
pub mod catalog_parquet_scanner;
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
mod catalog_semantic_jobs;
use catalog_semantic_jobs::*;
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
// sc-12409: parity guard tying the shipped video manifest's `limits.maxPixels` to the area cap of
// the ENGINE PINNED by Cargo.toml (mlx via `runtime-macos` on macOS, candle via `runtime-cuda`
// off-mac under backend-candle). Not a smoke — no weights, no GPU; just the shipped manifest bytes
// vs the pinned `MAX_AREA_*` const. Gated on the same "engine bundle present" superset as
// `smoke_support` because it needs a backend crate in scope to read the const.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod pinned_engine_geometry;
// sc-17607: the COMPOSITION half of the SC-15833 FLUX.2 audit — is `flux2_dev` still registered by
// `candle-gen-flux2` in the bundle the worker links? A codegen digest cannot answer that, because
// the measurement binary links neither `runtime-cuda` nor `candle-gen-catalog`. Test-only and
// candle-only: the composition under test IS the CUDA bundle, so there is no neutral version.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_composition_audit;
// Real-weight GPU smoke for the candle SCAIL-2 lane (sc-7078). Test-only + candle-only; never built
// in normal compiles. Drives the shipped worker conditioning + `crate::inference_runtime::load("scail2_14b")`.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod scail2_gpu_smoke;
// SC-20945's single terminal epic-20738 CUDA entrypoint. Test-only, candle-only, and #[ignore]d:
// the checked-in controller selects one reviewed profile cell per fresh process and serializes all
// 19 cells in one workflow job. Ordinary tests compile the source but never touch weights/hardware.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod epic_20738_terminal_cuda_smoke;
// Real-weight GPU smoke for the candle RealVisXL Lightning lane (sc-7176). Test-only + candle-only;
// drives `crate::inference_runtime::load("sdxl")` with the forced `lightning` sampler against the distilled checkpoint.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod realvisxl_lightning_gpu_smoke;
// Real-weight GPU smoke for the candle SDXL edit + PiD super-resolving decode (epic 7840, sc-8044).
// Test-only + candle-only; drives the bespoke `runtime_cuda::providers::sdxl::SdxlEdit` provider (inpaint) with the
// `pid_sdxl` student attached, asserting the PiD decode super-resolves the render-sized native decode.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod sdxl_edit_pid_gpu_smoke;
// Real-weight GPU smoke for the candle FLUX.2-dev lane (epic 6564 sc-7458). Test-only + candle-only;
// drives `crate::inference_runtime::load("flux2_dev")` with a Q4 LoadSpec (CPU-stage → quantize-onto-GPU) against the
// dense diffusers snapshot — the worker-lane validation backing the off-Mac candle routing wire.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod flux2_dev_gpu_smoke;
// Real-weight GPU smoke for the candle Anima 2B lane (epic 10512, sc-10625 — the hardware-gated
// acceptance extracted from sc-10525). Test-only + candle-only; drives `crate::inference_runtime::load("anima_base" |
// "anima_aesthetic" | "anima_turbo")` against the dense bf16 circlestone-labs/Anima split_files
// snapshot (± an official LoRA/LoKr), proving the candle Anima port renders coherently on real CUDA —
// the evidence that unblocks flipping `macOnly: false` / `candle_routed = true` (sc-10625).
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod anima_gpu_smoke;
// Real-weight GPU smoke for the candle SenseNova-U1 8B lane (sc-13817, epic 13678). Test-only +
// candle-only; drives `crate::inference_runtime::load("sensenova_u1_8b{,_fast}")` against the DENSE
// bf16 turnkey tier. SenseNova was the one candle image family with no GPU smoke — and its lane had
// never worked, because the tier resolver handed the dense-only loader an MLX-packed q8 tier. This is
// the hardware evidence for the sc-13817 dense-force fix.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod sensenova_gpu_smoke;
// SC-18902's retained real-weight evidence harness for the former `ltx_2_3_eros` Candle route.
// Test-only + candle-only, and itself #[ignore]d: the exact-head Windows CUDA capture proved the
// undistilled route unusable, so product routing now rejects Eros off-Mac. The harness remains as
// reproducible historical evidence and does not advertise or restore that route.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod ltx_eros_gpu_smoke;
// Real-weight GPU smoke for the candle SANA 1600M lane (epic 8485, sc-11780). Test-only + candle-only;
// drives the WORKER's `resolve_weights_dir("sana_1600m")` (the diffusers-snapshot-root resolution) +
// `gen_core::load("sana_1600m")` against the whole `Efficient-Large-Model/Sana_1600M_1024px_diffusers`
// snapshot, proving the candle SANA port renders a coherent true-CFG 1024² image on real CUDA — the
// hardware evidence backing `macOnly: false` / `candle_routed = true`.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod sana_candle_gpu_smoke;
// Hardware-gated evidence for the CUDA primary-context lease the generator cache takes around a
// cold load. Test-only + candle-only; proves an unbound thread cannot read device memory at all,
// that the lease makes the pre-load snapshot readable, and that releasing it destroys the context —
// the three facts `generator_cache::bind_backend_load_context` is built on, none of which a unit
// seam can see (the seams stub both helpers away).
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod generator_cache_context_gpu_smoke;
// Hardware-gated evidence that a FAILED `cuda_preflight` does not poison the process (sc-16260 AC 4).
// Test-only + candle-only; hides the devices with `CUDA_VISIBLE_DEVICES=-1`, probes (must fail),
// restores visibility and probes again in the SAME process — the exact move `recheck_gpu_health`
// makes. Without that property the recovery re-check would be dead code however correct its Rust.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod cuda_preflight_gpu_smoke;
// Real-weight GPU smoke for the candle Qwen-Image-Edit lane (sc-13534). Test-only + candle-only; drives
// the WORKER's `resolve_qwen_edit_candle_base` (the tier/gate reconciliation this story landed) + a
// bespoke `runtime_cuda::providers::qwen_image::QwenEdit` load + render, proving the resolver lands on the
// packed q4 tier subdir of the `SceneWorks/qwen-image-edit-2511-mlx` turnkey (NOT the upstream snapshot
// the pre-fix code reached) and that q4 packed-loads + renders a coherent edit on real CUDA.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod conditioned_image_gpu_smoke;
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod qwen_edit_candle_gpu_smoke;
// Real-weight GPU smoke for the candle InstantID + PiD super-resolving decode (epic 7840, sc-8386).
// Test-only + candle-only; drives the bespoke `runtime_cuda::providers::instantid::InstantId` provider across
// Identity/Angle/Pose with the `pid_sdxl` student attached, asserting the PiD decode 4×-super-resolves
// the native decode AND the ArcFace identity likeness survives. Validates the sc-8373 InstantID lane.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod instantid_pid_gpu_smoke;
// Real-weight GPU smoke for the candle Z-Image + PiD decode (epic 7840, sc-8033). Test-only +
// candle-only; drives `crate::inference_runtime::load("z_image_turbo", spec.with_pid(pid_flux, gemma))` — the generic
// candle t2i lane (sc-9727) — proving Z-Image's flux-aliased latent decodes through the pid_flux
// student at 4× (native 1024² -> 4096²). Z-Image has no dedicated pid_zimage; it reuses pid_flux.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod zimage_pid_gpu_smoke;
// Real-weight MLX smoke for the Krea 2 Turbo worker lane (epic 7565 sc-7575). Test-only + macOS-only;
// drives `crate::inference_runtime::load("krea_2_turbo")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir —
// the worker-lane validation (the crate links + drives the engine), not just the mlx-gen-krea crate.
#[cfg(all(test, target_os = "macos"))]
mod krea_turbo_mlx_smoke;
// Real-weight MLX smoke for the SenseNova-U1 `_fast` worker lane (sc-17396). Test-only + macOS-only;
// drives `crate::inference_runtime::load("sensenova_u1_8b_fast")` with a packed-tier Q8 `LoadSpec`
// against the HF CACHE. Distinct from the `sensenova_jobs` packed-tier smokes, which build the model
// via `load_sensenova_model` (`load_raw` + `from_weights`, the dense VQA/interleave shape) and so
// never reach the engine's own `load_fast` — the gap that let the pinned-artifact regression ship.
#[cfg(all(test, target_os = "macos"))]
mod sensenova_fast_q8_mlx_smoke;
// Real-weight MLX smoke for the Krea 2 Turbo pose-ControlNet worker lane on a PACKED Q8 base (sc-11796).
// Test-only + macOS-only; drives `gen_core::load("krea_2_turbo_control")` with the exact packed-q8
// `LoadSpec` `krea_control_spec` builds and asserts the pose steers the render vs a base passthrough —
// the worker-lane proof that pose control is honored on the installed quant tier (not silently dropped).
#[cfg(all(test, target_os = "macos"))]
mod krea_control_mlx_smoke;
// Real-weight MLX smoke for the FLUX.1-dev strict-control worker lane (sc-8244; engine E2 sc-8239).
// Test-only + macOS-only; drives `crate::inference_runtime::load("flux1_dev_control")` (Dir base + Shakker control
// overlay) per control mode (pose/canny/depth) and asserts a control-vs-control-free steer — the
// worker-lane validation that the crate links + drives the registered control generator end-to-end.
#[cfg(all(test, target_os = "macos"))]
mod flux1_control_mlx_smoke;
// Real-weight MLX smokes for the SD3.5 worker lane (epic 7841 S6 sc-7875 — the MLX-path validation
// boundary). Test-only + macOS-only; drive `crate::inference_runtime::load("sd3_5_large" | "sd3_5_large_turbo" |
// "sd3_5_medium")` against the gated stabilityai/* diffusers snapshots (the worker crate links + drives
// all three registered generators + the LoRA `with_adapters` apply seam), not just mlx-gen-sd3 in
// isolation.
#[cfg(all(test, target_os = "macos"))]
mod sd3_5_mlx_smoke;
// Real-weight MLX smoke for the SDXL base 1.0 Q8 worker lane (sc-8746, epic 8506 Group-B). Test-only +
// macOS-only; drives `crate::inference_runtime::load("sdxl")` with a Q8 LoadSpec against the packed `q8/` turnkey subdir.
// Closes the stale sc-1975 Q8-on-SDXL loop on-device: asserts the fixed mlx-gen Q8 path (sc-2641) renders
// non-degenerate AND specifically NOT all-zero (the retired Apple recipe's exact failure signature).
#[cfg(all(test, target_os = "macos"))]
mod sdxl_base_q8_mlx_smoke;
// Real-weight MLX train→apply smoke for the Illustrious-XL SDXL-family lane (sc-10618, epic 10609).
// Test-only + macOS-only; drives `runtime_macos::providers::sdxl::load_trainer` from the Illustrious turnkey's dense
// `bf16/` tier, trains a tiny LoRA/LoKr, then renders WITHOUT vs WITH the adapter via
// `runtime_macos::providers::sdxl::load(...).with_adapters` and asserts it visibly changes the output — the E2E evidence
// (not a registry entry + a green unit test) the training half of the epic demands. For LoKr it also
// asserts no `mid_block` factors were emitted (sc-2640: the SDXL LoKr surface is down/up attention only).
#[cfg(all(test, target_os = "macos"))]
mod illustrious_train_apply_mlx_smoke;
// Real-weight MLX smoke for the Lens-Turbo Q4 worker lane (sc-8763, epic 8506 Group-B). Test-only +
// macOS-only; drives `crate::inference_runtime::load("lens_turbo")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/lens-turbo-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_turbo_q4_mlx_smoke;

// Real-weight MLX smoke for the Mage-Flow q4 worker lane (sc-14980 / sc-14979). Mage is the first
// family whose tier subdir is NOT engine-complete — `<snapshot>/q4/` is DiT-only, and the shared text
// encoder + VAE arrive as per-tier co-requisites — so this is the on-device proof that the manifest's
// per-tier downloads and the pinned engine agree. It fails loudly against an engine pinned before the
// sc-14979 split, which is exactly the ordering hazard the pin bump closes.
#[cfg(all(test, target_os = "macos"))]
mod mage_flow_q8_mlx_smoke;
// Real-weight MLX smoke for the recovered base Lens Q4 worker lane (sc-8767, epic 8506 Group-B).
// Test-only + macOS-only; drives `crate::inference_runtime::load("lens")` with a Q4 LoadSpec against the packed `q4/`
// turnkey subdir. On-device evidence that the SceneWorks/lens-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate (both transformer + gpt-oss MoE TE are packed per-tier; NOT a dense-TE model).
#[cfg(all(test, target_os = "macos"))]
mod lens_base_q4_mlx_smoke;
// Real-weight MLX smoke for the Chroma1-Base Q4 worker lane (sc-8777, epic 8506 Group-B). Test-only +
// macOS-only; drives `crate::inference_runtime::load("chroma1_base")` with a Q4 LoadSpec against the packed `q4/` turnkey
// subdir. On-device evidence that the SceneWorks/chroma1-base-mlx pre-quantized q4 tier loads through the
// worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and renders
// non-degenerate. Chroma packs ONLY the transformer per-tier (the T5-XXL TE + VAE stay dense — chroma
// never quantizes its T5, so no denseTextEncoderTier). hd/flash share this crate + layout.
#[cfg(all(test, target_os = "macos"))]
mod chroma1_base_q4_mlx_smoke;
// Real-weight MLX smoke for the PiD 2K/4K output tier (epic 7840, sc-10054). Test-only + macOS-only;
// drives the REAL `pid_output_tier` + `pid_effective_dims` mapping then renders z_image_turbo through
// `crate::inference_runtime::load(...).with_pid(pid_flux, gemma)` + `use_pid`, asserting `pidTarget:"2k"` yields a 2048²
// image (base 512 × 4) and `"4k"` yields 4096² (base 1024 × 4) — the on-device evidence that the tier
// mapping actually changes the output resolution on real weights.
#[cfg(all(test, target_os = "macos"))]
mod pid_tier_mlx_smoke;
// Real-weight MLX smoke for the SANA quant-matrix lane (sc-8489/sc-8513, epic 8506). macOS-only;
// drives `crate::inference_runtime::load("sana_1600m"|"sana_sprint_1600m")` with a per-tier LoadSpec against the
// packed q4/q8 + dense bf16 turnkey subdirs. On-device evidence that the SceneWorks/Sana_*_mlx
// pre-quantized turnkeys load through the worker packed path (`STANDARD_TIER_MODELS` →
// `standard_tier_subdir` resolves the tier) and render non-degenerate at EVERY downloaded tier. SANA
// packs the Linear-DiT transformer + Gemma-2 CHI TE per-tier (DC-AE VAE dense); q4/q8 are a no-op on
// the already-packed weights (packed-detected) and bf16 loads dense (Quant::None).
#[cfg(all(test, target_os = "macos"))]
mod sana_mlx_smoke;
// Real-weight MLX smoke for the Mochi 1 quant-matrix video lane (epic 1788, sc-11992). macOS-only;
// drives `crate::inference_runtime::load("mochi_1")` with a per-tier LoadSpec against the
// pre-quantized q4/q8 + dense bf16 tier subdirs of the SceneWorks/mochi-1-mlx turnkey. On-device
// evidence that the worker tier path loads (`WeightsSource::Dir` = the TIER dir, with the shared
// T5-XXL/tokenizer/AsymmVAE resolved from that dir's PARENT — the A6 sibling layout) and renders a
// non-degenerate, MOVING clip at every downloaded tier, with monotonic progress that reaches decode
// (the video job lane has no background heartbeat during a generation).
#[cfg(all(test, target_os = "macos"))]
mod mochi_mlx_smoke;
// Real-weight smoke for the Voice Clone two-call chain (sc-13411, C4): Kokoro base TTS → OpenVoice V2
// tone-color conversion → Chatterbox-VE evidence that the converted clip's speaker identity moved
// toward the reference. Test-only + macOS-only; #[ignore]d — drives the exact
// `runtime_macos::catalog().audio()` seams the worker's voice-clone job uses, minus the API/job plumbing.
#[cfg(all(test, target_os = "macos"))]
mod voiceclone_smoke;
// On-device per-tier memory-footprint measurement harness (sc-8516, epic 8506). Test-only + macOS-only;
// #[ignore]d real-weight smokes that drive `crate::inference_runtime::load(id)` + ONE generation while sampling the MLX
// process-global memory counters (mlx_rs::memory::{reset_peak_memory, get_active_memory, get_peak_memory})
// generator_cache.rs already publishes — producing measured resident + peak footprint per (model, tier)
// to calibrate the sc-8509 RAM→tier suggestion (apps/web/src/tierSuggestion.js) and backfill the sc-8508
// manifest footprint fields.
#[cfg(all(test, target_os = "macos"))]
mod footprint_measure;
// On-device RESOLUTION sweep of the MLX activation transient (sc-16195, epic 15448). Test-only +
// macOS-only; the sibling of footprint_measure with the axis rotated — that one measures ONE
// resolution across tiers, this one measures ONE tier across resolutions, sampling the same
// process-global MLX counters. Supplies the shape the mlx_fit_gate request estimator's headroom term
// is fitted to, replacing the linear-in-megapixels scaling of a 1024²-only calibration.
#[cfg(all(test, target_os = "macos"))]
mod resolution_sweep;
// On-device end-to-end validation of the epic 18093 memory-ladder apparatus (sc-18101). Test-only +
// macOS-only; four `#[ignore]`d scenarios that drive the real `mlx_fit_gate::evaluate_request` seam
// against live loaded providers under `SCENEWORKS_MLX_MEMORY_CAP_GB` — an unmeasured cell engaging a
// deep rung and rendering, a measured-current cell whose selection is diffed against a pre-epic
// checkout, a stale-closure lane admitting at the widened peak, and an oversized request refusing.
#[cfg(all(test, target_os = "macos"))]
mod ladder_e2e_sc18101;
// On-device validation of z_image_turbo's DeferredMaterialization Sequential cold load (sc-18409).
// Test-only + macOS-only; one `#[ignore]`d scenario proving PR #2215's `apply_residency_policy`
// coupling (Sequential branch ⇒ deferred load shape for z_image_turbo) on real bf16 weights, with
// a real render and an observed-peak-vs-admitted-ceiling comparison.
#[cfg(all(test, target_os = "macos"))]
mod ladder_e2e_sc18409;
// On-device build helper for the Wan2.2 T2V-A14B quant matrix (sc-9942, epic 8506). Test-only +
// macOS-only; an #[ignore]d helper that drives `runtime_macos::providers::wan::convert::convert_t2v_14b` once per tier
// (bf16/q8/q4) against the native checkpoint to produce the self-contained hosted tier subdirs, then
// copies the tokenizer the converter omits. Run one-off to build the artifacts for
// `SceneWorks/wan2.2-t2v-a14b-mlx`; not exercised in CI (needs the ~126GB native weights).
#[cfg(all(test, target_os = "macos"))]
mod wan_t2v_14b_tier_build;
// On-device build helper for the Wan2.2 I2V-A14B quant matrix (sc-9943, epic 8506). The image→video
// sibling of the above; drives `runtime_macos::providers::wan::convert::convert_i2v_14b` (in_dim 36 image-concat) once
// per tier (bf16/q8/q4) against the native checkpoint to produce the self-contained hosted tier
// subdirs, then copies the tokenizer the converter omits. Run one-off to build the artifacts for
// `SceneWorks/wan2.2-i2v-a14b-mlx`; not exercised in CI (needs the ~126GB native weights).
#[cfg(all(test, target_os = "macos"))]
mod wan_i2v_14b_tier_build;
// On-device build helper for the Wan2.2 TI2V-5B quant matrix (sc-9941, epic 8506). The single-expert
// sibling of the A14B helpers: drives `runtime_macos::providers::wan::convert::convert_ti2v_5b` for the dense bf16 tier,
// then derives the q8/q4 tiers worker-side (load the bf16 `model.safetensors` →
// `quantize_wan_transformer` → save + reuse the shared dense T5/VAE/tokenizer + a `config.json` quant
// patch) — byte-identical to an inline convert, no mlx-gen change. Run one-off to build the artifacts
// for `SceneWorks/wan2.2-ti2v-5b-mlx`; not exercised in CI (needs the native checkpoint).
#[cfg(all(test, target_os = "macos"))]
mod wan_ti2v_5b_tier_build;
// On-device build helper for the Bernini quant matrix (sc-9945, epic 8506). Composite model: derives
// all three tiers (bf16/q8/q4) worker-side from the already-hosted lean bf16 snapshot — copy the dense
// remainder, quantize the planner backbone (`runtime_macos::providers::bernini::convert::quantize_qwen_planner_backbone`)
// + both renderer experts (`runtime_macos::providers::wan::convert::quantize_wan_transformer`), patch the two config
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
// RTMW detector in-process; on a candle-disabled box the capability is not advertised and the job
// remains queued.
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
// capability is not advertised and the job remains queued.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod kps_jobs;
// Image upscaling: Real-ESRGAN (epic 3482, sc-3489) RRDBNet x2/x4 via `ort`/CoreML on Mac, plus the
// SeedVR2 one-step diffusion upscaler — native MLX on Mac (sc-4815) and the candle CUDA backend on
// Windows (sc-5928). So the module compiles on Mac AND the Windows/CUDA candle lane; the ort/CoreML
// Real-ESRGAN path inside stays Mac-gated while the candle worker supplies the off-Mac Real-ESRGAN
// lane, and SeedVR2 is backend-neutral (`crate::inference_runtime::load("seedvr2")`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod upscale_jobs;
// YOLO11 person detection + selected-person ByteTrack tracking (epic 3482, sc-3488/sc-3633;
// off-Mac candle lane sc-5498, epic 5482). Native-MLX YOLO11m on Mac, `ort`/CUDA on the off-Mac
// candle GPU-worker lane (the pure-Rust ByteTrack in `person_track` is backend-neutral). So both
// modules compile on Mac AND the candle lane; a candle-disabled off-Mac build refuses the jobs.
// Person segmentation uses MLX SAM2/SAM3 on Mac and native candle SAM3 off-Mac.
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
// box-prompt segmenter generates per-frame masks in `run_person_track`. On Mac this is SAM2; off-Mac
// the candle SAM3 seam in `media_jobs::run_candle_segmenter` supplies real masks.
#[cfg(target_os = "macos")]
mod person_segment;
// SAM3 text-concept (PCS) person segmenter — the box-prompt-free upgrade of `person_segment`
// (epic 4910, sc-4926). macOS-only (native MLX `mlx-gen-sam3`); the off-Mac Windows/CUDA candle
// sibling is `person_segment_sam3_candle` below.
#[cfg(target_os = "macos")]
mod person_segment_sam3;
// Smart-select image segmentation: native SAM3 box-PVS on both registered runtimes. Candle's
// pinned SAM3 surface has no point-prompt API, so segment_jobs rejects points before any I/O.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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

mod checkpoint_catalog_migration;
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
    use std::sync::OnceLock;
    use tokio::sync::watch;

    static SHUTDOWN: OnceLock<watch::Sender<bool>> = OnceLock::new();
    let sender = SHUTDOWN.get_or_init(|| {
        let (sender, _receiver) = watch::channel(false);
        let signal = sender.clone();
        tokio::spawn(async move {
            wait_for_process_shutdown_source().await;
            // Latch the request even if it arrives between two `select!` calls,
            // when no receiver is currently subscribed.
            signal.send_replace(true);
        });
        sender
    });
    let mut receiver = sender.subscribe();
    wait_for_shutdown_latch(&mut receiver).await;
}

async fn wait_for_shutdown_latch(receiver: &mut tokio::sync::watch::Receiver<bool>) {
    if *receiver.borrow_and_update() {
        return;
    }
    while receiver.changed().await.is_ok() {
        if *receiver.borrow_and_update() {
            return;
        }
    }
}

async fn wait_for_process_shutdown_source() {
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
/// The process-wide shutdown source calls this once on a supervised Windows child.
/// A SINGLE background reader is guarded by a `OnceLock`, and its EOF result is
/// latched over a `watch` channel. The reader drains stdin on std's blocking handle
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
    // Only the top-level worker process performs conversion recovery. It runs before any utility
    // children are spawned, and the lifecycle lock excludes independently running finalizers.
    // Child restarts must never sweep while sibling utility workers may be converting.
    if !settings.is_child_worker {
        recover_stranded_model_conversions(&settings.data_dir).await?;
        // sc-20651 (epic 20398): compile pre-epic imported catalog entries into the checkpoint
        // plan store. Started here rather than in `run_worker_loop` for the same reason
        // `recover_stranded_model_conversions` is: the loop runs in every child utility worker
        // too, and four of them re-reading every legacy checkpoint at once would be four times
        // the disk for one result. Started and NEVER awaited — an unmigrated entry still routes
        // through the bespoke lane it always did, so nothing downstream waits on this.
        {
            let config_dir = settings.config_dir.clone();
            let data_dir = settings.data_dir.clone();
            tokio::spawn(async move {
                match checkpoint_catalog_migration::migrate_legacy_checkpoint_catalog(
                    &config_dir,
                    &data_dir,
                )
                .await
                {
                    Ok(summary)
                        if summary.attempted == 0 && summary.declined_containers.is_empty() => {}
                    Ok(summary) => tracing::info!(
                        event = "checkpoint_catalog_migrated",
                        attempted = summary.attempted,
                        migrated = summary.migrated,
                        failed = summary.failed(),
                        declined = summary.declined_containers.len(),
                        vanished = summary.vanished.len(),
                        skipped = summary.skipped,
                        "migrated pre-epic imported models into the checkpoint plan store"
                    ),
                    Err(error) => tracing::warn!(
                        error = %error,
                        "pre-epic imported-model catalog migration failed"
                    ),
                }
            });
        }
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

/// Which worker loops should run the startup CUDA probe: only a GPU worker on a candle-enabled
/// build. Pure so the gate is testable without a GPU — every one of these three conditions is
/// load-bearing, and inverting any of them is silent. `cpu` covers both the standalone utility
/// pool and the API's in-process loops (`spawn_inprocess_utility_worker`); those must never build
/// a CUDA context. `mlx` is the macOS GPU worker, which has its own Metal probe. And with the
/// candle backend off there is nothing to probe for.
fn should_run_cuda_preflight(settings: &Settings) -> bool {
    settings.backend_candle_enabled && settings.gpu_id != "cpu" && settings.gpu_id != "mlx"
}

/// Server/Docker-lane CUDA preflight (sc-16247, GH #1966).
///
/// The desktop gets a real setup screen: `run_startup` runs `cuda_device_preflight` and refuses to
/// start on an unusable GPU. The server/Docker lane has no such screen and, before this, no GPU
/// preflight of ANY kind — `run_worker()` goes straight into the worker loop, so a driver-stack
/// mismatch there surfaces exactly as GH #1966 described: first at the first model load, as a job
/// failure, per job, forever.
///
/// This runs the same probe and logs the actionable reason ONCE at startup, loudly, at `error`
/// level — so the container's logs name the host-side fix rather than leaving an operator to infer
/// it from a stream of failed jobs.
///
/// **sc-16260 acts on the verdict instead of only logging it.** The returned [`GpuHealth`] is
/// threaded into `discover_gpu`, which withholds every candle capability when the GPU is unusable,
/// and into the worker loop, which reports `WorkerStatus::Unhealthy` carrying the same reason.
/// Together those mean a driver-mismatch host leaves generation QUEUED with a visible explanation
/// instead of claiming and failing every routed job forever.
///
/// It deliberately does **not** abort the process. Tearing the process down here would turn a
/// diagnosable, recoverable state into a crash loop with its own message, and the worker still has
/// a job to do while degraded: hold its registration, report why, and re-probe (see
/// [`GPU_HEALTH_RECHECK`]) so a host fixed without restarting the container recovers on its own.
///
/// Gated to the GPU worker by [`should_run_cuda_preflight`]: the CPU utility loops never touch
/// CUDA, and probing from each of them would build a throwaway CUDA context per utility process.
/// A no-op on every build without the candle lane linked, and on macOS.
///
/// The probe acquires device 0, which is faithful for the deployments that set
/// `CUDA_VISIBLE_DEVICES`: under the `auto` supervisor each per-GPU child is spawned with
/// `CUDA_VISIBLE_DEVICES=<its gpu id>` (`supervisor::child_environment`), so device 0 IS that
/// child's own GPU. A server deployment that pins `SCENEWORKS_CANDLE_GPU_ID` without also setting
/// `CUDA_VISIBLE_DEVICES` gets physical device 0 instead — but so does its generation, because
/// `runtime_cuda::media::default_device()` is itself a hardcoded `new_cuda(0)`. The probe is
/// therefore still testing the device that lane will use; it does not fix, and must not be read as
/// endorsing, that pinning gap.
///
/// Runs on the blocking pool: `cuInit` plus the first kernel launch is ~0.25-0.5 s of synchronous
/// driver work, and `run_worker_loop` is also called IN-PROCESS by the API's utility pool
/// (`spawn_inprocess_utility_worker`). That pool is `cpu` by default and so skips the probe
/// entirely, but `SCENEWORKS_RUST_WORKER_GPU_ID` can override it — and a GPU-id override there
/// would otherwise stall the API's runtime thread during startup.
async fn cuda_startup_health(settings: &Settings) -> GpuHealth {
    let health = probe_cuda_health(settings).await;
    if let Some(reason) = health.reason() {
        tracing::error!(
            event = "cuda_preflight_failed",
            gpuId = %settings.gpu_id,
            reason = %reason,
            "SceneWorks GPU worker cannot acquire a CUDA device; generation capabilities withheld"
        );
    }
    health
}

/// Run the probe and classify the outcome, with no logging of its own — so the startup call
/// ([`cuda_startup_health`]) can be loud exactly once while the recovery re-check
/// ([`recheck_gpu_health`]) stays quiet about a failure it has already reported.
///
/// A lane that does not probe ([`should_run_cuda_preflight`]) reports [`GpuHealth::Usable`]: the
/// CPU utility loops and the macOS `mlx` worker must behave precisely as they did before this
/// existed, so "no probe ran" is deliberately indistinguishable from "the probe passed".
///
/// **Only a BLOCKING failure makes the worker unhealthy** — the severity split
/// [`cuda_failure_is_blocking`] already draws for the desktop setup screen, applied here to the
/// worker's own advertisement. That split exists because the two directions cost wildly different
/// amounts, and this path is no different: over-reporting is the expensive one. A transient CUDA
/// OOM — another process, or an orphaned worker from a crashed session, currently holding the GPU
/// — is a real probe failure that says nothing about the driver stack. Treating it as unhealthy
/// would strip every capability, hand the operator the GENERIC "check that nvidia-smi lists a
/// supported GPU" text (no `CUDA_ERROR_*` token matches, so there is no specific remedy to give),
/// and — with `SCENEWORKS_CANDLE_REQUIRED=1` — fail queued work over a condition that clears by
/// itself. The desktop deliberately starts the app in that state; the worker must likewise stay
/// advertising. A transient failure is logged and stepped over, and if a job does then run, the
/// classified message from [`crate::classify_engine_error`] explains what happened.
async fn probe_cuda_health(settings: &Settings) -> GpuHealth {
    if !should_run_cuda_preflight(settings) {
        return GpuHealth::Usable;
    }
    let probe = tokio::task::spawn_blocking(cuda_preflight).await;
    // A JoinError can only be a panic inside `cuda_preflight`, which already catches its own
    // (see `preflight::cuda_preflight`) — report rather than propagate either way. It is NOT
    // evidence of a driver-class fault, so it is folded into the same `Err` the classifier then
    // routes down the advisory path.
    let outcome = match probe {
        Ok(result) => result,
        Err(error) => Err(format!(
            "the CUDA preflight probe did not complete: {error}"
        )),
    };
    classify_probe_outcome(outcome, &settings.gpu_id)
}

/// The probe outcome → health verdict decision, sync and GPU-free so the severity split is
/// unit-testable on any machine (see `only_a_driver_class_probe_failure_makes_the_worker_unhealthy`).
///
/// Kept apart from [`probe_cuda_health`] deliberately: that function's only other job is running
/// the probe, which needs real CUDA, and a rule this consequential — it decides whether a worker
/// withdraws every capability it serves — must not be reachable only from hardware.
fn classify_probe_outcome(probe: Result<(), String>, gpu_id: &str) -> GpuHealth {
    let reason = match probe {
        Ok(()) => return GpuHealth::Usable,
        Err(reason) => reason,
    };
    if !cuda_failure_is_blocking(&reason) {
        tracing::warn!(
            event = "cuda_preflight_transient",
            gpuId = %gpu_id,
            reason = %reason,
            "SceneWorks GPU probe failed for a reason that may clear on its own; keeping the \
             worker's capabilities advertised"
        );
        return GpuHealth::Usable;
    }
    GpuHealth::Unusable { reason }
}

/// How often an UNHEALTHY worker re-runs the CUDA probe (sc-16260 AC 4).
///
/// The startup probe is a single sample, so without this a host repaired underneath a running
/// container would stay stranded with its capabilities withheld until someone restarted it. Only
/// an unhealthy worker re-probes; once the GPU is usable the loop stops entirely, so a healthy
/// worker never pays for this and never builds a second CUDA context behind a running job.
///
/// **The transition is therefore ONE-WAY: unhealthy → healthy only.** A worker that starts healthy
/// is never re-probed, so a driver that dies mid-life (an Xid, a device falling off the bus) leaves
/// it reporting `idle` and claiming jobs, which then fail individually with the classified
/// driver-error text from [`crate::classify_engine_error`] — i.e. exactly the pre-sc-16260
/// behaviour, for that case only. Detecting mid-life GPU death is a different problem (it wants a
/// signal from the failing job, not a poll) and is deliberately out of this story's scope.
///
/// A minute is chosen against what actually gets fixed on the other side. The dominant failure
/// (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, GH #1966) needs a host reboot, which restarts the
/// container anyway — the re-check cannot help there and is not meant to. What it does cover is
/// the genuinely transient family: `CUDA_ERROR_SYSTEM_NOT_READY` while the driver/fabric is still
/// initializing, a GPU briefly held by another process, or a device hot-attached to the container.
/// Those clear on the order of seconds-to-minutes, so a minute recovers promptly without spinning
/// `cuInit` against a wedged driver every poll turn.
const GPU_HEALTH_RECHECK: Duration = Duration::from_secs(60);

/// How often an idle worker runs a resolved-cache retention checkpoint (sc-19710). A pass walks
/// every entry and re-verifies the source of anything it intends to evict, so it is deliberately
/// far rarer than the poll interval; retention is a housekeeping activity, not a hot path.
const RESOLVED_CACHE_RETENTION_INTERVAL: Duration = Duration::from_secs(600);

/// Opens the resolved-cache retention driver when there is anything to drive.
///
/// The store is never created as a side effect: an opt-out install must not grow a managed cache
/// root. A store that already exists is still reconciled even when the policy was since disabled,
/// so entries materialized while it was on cannot be stranded by turning it off.
fn resolved_cache_retention(
    data_dir: &std::path::Path,
) -> Option<sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheRetention> {
    use sceneworks_core::model_artifacts::resolved_cache::{
        ResolvedCachePolicy, ResolvedCacheRetention, ResolvedCacheStore,
    };

    // Derived here rather than read from `Settings::resolved_cache`, which is `cfg(not(test))` and
    // therefore absent from test builds. It is the same value: that field is itself populated by
    // `from_env_or_safe_default`, which fails closed to the finite, disabled default.
    let policy = ResolvedCachePolicy::from_env_or_safe_default();
    let exists = data_dir.join("models").join("resolved").is_dir();
    if !policy.enabled && !exists {
        return None;
    }
    let store = match ResolvedCacheStore::open(data_dir) {
        Ok(store) => store,
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache store unavailable; retention is skipped");
            return None;
        }
    };
    match ResolvedCacheRetention::new(store, policy) {
        Ok(retention) => Some(retention),
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache retention policy is invalid; retention is skipped");
            None
        }
    }
}

/// Starts one retention checkpoint on the blocking pool and returns its handle **without awaiting
/// it**.
///
/// Retention is housekeeping and must never sit on the claim path: a sweep that evicts walks every
/// entry and re-hashes each candidate's source, so awaiting one would make a job submitted just
/// afterwards wait for the whole sweep, and awaiting the startup pass would delay the very first
/// claim by a full recover-plus-retention cycle. Detaching is safe by construction — the eviction
/// tombstone is durable, so a sweep cut short by process exit converges on the next pass rather
/// than leaving a half-removed entry. The gating (policy check, store open) happens inside the
/// blocking task too, so no filesystem work touches the async runtime.
///
/// Retention failures are never fatal: the cache is an optimization.
fn spawn_retention_checkpoint(
    data_dir: std::path::PathBuf,
    startup: bool,
) -> tokio::task::JoinHandle<()> {
    use sceneworks_core::model_artifacts::resolved_cache::{
        RetentionCheckpointOutcome, RetentionHold,
    };

    tokio::task::spawn_blocking(move || {
        let Some(retention) = resolved_cache_retention(&data_dir) else {
            return;
        };
        let now = sceneworks_core::time::now_unix_seconds().max(0) as u64;
        let outcome = if startup {
            retention.run_after_recovery(now)
        } else {
            retention.run_if_idle(true, now)
        };
        match outcome {
            Ok(RetentionCheckpointOutcome::Ran(report)) => {
                // Entries held because their authoritative source could not be verified are the
                // half of "never strand silently" that eviction counts alone would hide: with both
                // delete routes now reconciling, a held entry means a source that went away
                // outside the API. It is deliberately only reported — a disconnected external
                // library is a disconnect, never an uninstall, and must never trigger removal.
                let unverified = report
                    .retained
                    .iter()
                    .filter(|record| record.hold == RetentionHold::SourceUnverified)
                    .count();
                if !report.evicted.is_empty() || !report.failed.is_empty() || unverified != 0 {
                    emit_event_value(
                        Level::INFO,
                        json!({
                            "event": "resolved_cache_retention",
                            "startup": startup,
                            "evicted": report.evicted.len(),
                            "failed": report.failed.len(),
                            "sourceUnverified": unverified,
                            "bytesBefore": report.complete_bytes_before,
                            "bytesAfter": report.complete_bytes_after,
                            "limitSatisfied": report.limit_satisfied,
                        }),
                    );
                }
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(error = %error, startup, "resolved-cache retention checkpoint failed")
            }
        }
    })
}

/// Starts one resolved-cache promotion drain on the blocking pool and returns its handle **without
/// awaiting it** (sc-19706).
///
/// Same rule as the retention checkpoint, for the same reason: the drain resolves whole closures
/// against the source library and then copies and hashes a bundle, so awaiting it would make the
/// next job wait for a promotion it has nothing to do with. Failures are never fatal — the cache is
/// an optimization, and a closure that could not be promoted is simply recorded again the next time
/// that model loads.
fn spawn_promotion_drain(settings: Settings) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || resolved_cache_promotion::drain_intake(&settings))
}

/// Re-run the CUDA probe for a worker that is currently unhealthy, and act on any change
/// (sc-16260 AC 4).
///
/// On recovery this RE-REGISTERS. That is the load-bearing half: the capability set is what the
/// API routes on, and it was frozen at the withheld set when the worker started, so simply
/// flipping the status back to `idle` would leave a healthy worker advertising nothing and the
/// queue still stalled. `register_worker` is an upsert on `worker_id`, so this restores the full
/// candle set on the existing row and clears the stored reason; the next heartbeat reports `idle`.
///
/// Logging is asymmetric on purpose. Recovery is an event worth an `info` line. A failure
/// IDENTICAL to the one already reported at startup is not — repeating it every minute would bury
/// the loud line it was meant to make findable. A failure whose reason CHANGED is new information
/// (`SYSTEM_NOT_READY` resolving into `NO_DEVICE` says the driver came up but the GPU did not),
/// so that gets its own `warn`.
async fn recheck_gpu_health(
    api: &ApiClient,
    settings: &Settings,
    health: &mut GpuHealth,
) -> WorkerResult<()> {
    let previous = health.reason().map(str::to_owned);
    let next = probe_cuda_health(settings).await;
    let reason = next.reason().map(str::to_owned);
    *health = next;
    match reason {
        None => {
            tracing::info!(
                event = "cuda_preflight_recovered",
                gpuId = %settings.gpu_id,
                "SceneWorks GPU worker re-acquired a CUDA device; re-advertising generation capabilities"
            );
            let gpu = discover_gpu(settings, health).await;
            // A `Canceled` here means shutdown arrived during the re-registration backoff, not a
            // recovery failure. Swallow it: propagating would exit `run_worker_loop` with `Err`,
            // skipping the terminal `Offline` heartbeat that both other shutdown paths post and
            // leaving the row reading `unhealthy` for the full 90 s stale window. The loop's own
            // `shutdown_signal()` arm handles it one turn later, cleanly.
            match register_worker_with_retry(api, settings, &gpu).await {
                Ok(()) | Err(WorkerError::Canceled(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Some(reason) if previous.as_deref() != Some(reason.as_str()) => {
            tracing::warn!(
                event = "cuda_preflight_changed",
                gpuId = %settings.gpu_id,
                reason = %reason,
                "SceneWorks GPU worker still cannot acquire a CUDA device; the reason changed"
            );
        }
        Some(_) => {}
    }
    Ok(())
}

/// How long a managed-import staging tree must have gone untouched before a worker treats it as a
/// crash orphan rather than another process's live transfer (sc-20636).
///
/// Generous on purpose: reclaiming late costs disk, reclaiming early destroys a running import.
const IMPORT_STAGING_ORPHAN_AGE: Duration = Duration::from_secs(60 * 60);

/// Reclaim managed-import staging trees left behind by a crash (sc-20636).
///
/// The crash is the one ingest failure no destructor covers: the commit rename never ran, so no
/// install, plan, or catalog record exists — only the staging bytes, which nothing references and
/// nothing else will ever remove. Age-gated via
/// [`sceneworks_core::checkpoint_ingest::active_staging_ids`] because several worker processes share one data dir,
/// so "in flight" is not knowable from this process's own state.
///
/// Separate from [`spawn_retention_checkpoint`]: that recovers the resolved-model cache, a
/// different store with its own interval and its own maintenance slot. This one is startup-only —
/// a crash is the only thing that produces an orphan.
fn reclaim_import_staging(
    data_dir: &Path,
) -> Result<usize, sceneworks_core::checkpoint_ingest::ManagedIngestError> {
    let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir);
    let active =
        sceneworks_core::checkpoint_ingest::active_staging_ids(&store, IMPORT_STAGING_ORPHAN_AGE);
    let in_flight: Vec<&str> = active.iter().map(String::as_str).collect();
    sceneworks_core::checkpoint_ingest::sweep_staging(&store, &in_flight)
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
    // sc-7820 (epic 7819): apply the GPU memory ceiling to the MLX runtime once at startup, before
    // any model load. The MLX limit is process-global, so this single call covers generations,
    // upscales, AND LoRA training. When the user configured no ceiling (0) a default derived from
    // physical RAM is applied instead (GitHub #1932) — MLX's own default budget is ~all of unified
    // memory, which starves macOS on a small Mac. No-op on non-macOS/candle builds.
    generator_cache::apply_gpu_memory_limit(settings.gpu_memory_limit_bytes);
    // sc-7825 (epic 7819): on the MLX GPU worker only, publish live MLX memory telemetry to the
    // shared config dir for the Settings readout. Gated to `mlx` so the CPU utility workers (which
    // do no MLX work) don't clobber the file with zeros.
    if settings.gpu_id == "mlx" {
        generator_cache::spawn_gpu_telemetry(settings.config_dir.clone());
    }
    // sc-16247 / sc-16260: probe the CUDA device ONCE before advertising anything, and let the
    // verdict shape what this worker claims to be able to do. `discover_gpu` withholds the whole
    // candle capability block on an unusable GPU, so the registration below advertises only the
    // placeholder set and no generation job can route here.
    let mut health = cuda_startup_health(&settings).await;
    let gpu = discover_gpu(&settings, &health).await;
    let api = ApiClient::new(&settings);
    let http_client = crate::downloads::streaming_download_client();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    let mut lock_failures = 0_u32;
    let mut idle_heartbeat = IdleHeartbeat::new(progress_report_interval(&settings));
    // sc-16260 AC 4: the startup probe is one sample, so an unhealthy worker re-probes on an
    // interval and re-advertises if the host is repaired underneath it. Seeded a full interval
    // out — the startup probe just ran, and re-running it immediately would say nothing new.
    let mut next_gpu_recheck = Instant::now() + GPU_HEALTH_RECHECK;
    // sc-19710 / sc-19706: ONE resolved-cache maintenance slot, shared by the retention checkpoint
    // and the promotion drain. A single slot is deliberate rather than one handle each: a sweep
    // walks and re-hashes eviction candidates while a promotion copies and hashes a whole bundle,
    // and running both at once would have them competing for the same disk and the same entries —
    // retention deciding what fits while promotion is still adding to it.
    //
    // The startup occupant is the retention checkpoint, which recovers the store (finishing any
    // eviction interrupted by a crash) and then enforces retention. It is started, never awaited —
    // the first job claim must not queue behind a recover-plus-retention pass.
    let mut maintenance_task = Some(spawn_retention_checkpoint(settings.data_dir.clone(), true));
    let mut next_retention_checkpoint = Instant::now() + RESOLVED_CACHE_RETENTION_INTERVAL;
    // sc-20636: the other startup reclamation. Spawned blocking and never awaited, for the same
    // reason the retention checkpoint is not: removing an abandoned multi-gigabyte staging tree
    // must not delay the first job claim.
    {
        let data_dir = settings.data_dir.clone();
        tokio::task::spawn_blocking(move || match reclaim_import_staging(&data_dir) {
            Ok(0) => {}
            Ok(reclaimed) => tracing::info!(
                event = "import_staging_reclaimed",
                reclaimed,
                "reclaimed abandoned model-import staging directories"
            ),
            Err(error) => tracing::warn!(error = %error, "model-import staging sweep failed"),
        });
    }
    loop {
        // A Metal watchdog timeout poisons the command queue for this OS process. The just-failed
        // job was already made terminal by `run_utility_job`; stop BEFORE `poll_once` can claim a
        // successor, mark this instance unhealthy, and clean-exit. Under `GPU_ID=auto` the existing
        // supervisor observes that single exit and spawns a fresh process with fresh Metal state.
        // Clean exit is intentional: an abnormal-exit attribution would manufacture a second job
        // failure after the truthful first timeout has already been persisted.
        if settings.gpu_id == "mlx" {
            if let Some(recycle) = mlx_worker_recovery::global().begin_recycle() {
                emit_event_value(
                    Level::ERROR,
                    json!({
                        "event": "mlx_h3_i2v_poisoned_worker_recycle",
                        "workerId": settings.worker_id,
                        "gpuId": settings.gpu_id,
                        "firstTimeout": recycle.first_timeout,
                        "sawSubmissionsIgnored": recycle.saw_submissions_ignored,
                    }),
                );
                let _ = heartbeat_with_reason(
                    &api,
                    &settings,
                    WorkerStatus::Unhealthy,
                    None,
                    Some(recycle.reason),
                )
                .await;
                return Ok(());
            }
            if let Some(reason) = mlx_worker_recovery::global().quarantine_reason() {
                // This is reachable only when shutdown interrupted the active job's terminal-write
                // retry. Stay unclaimable rather than recycling on an unconfirmed failure or
                // falling through to `poll_once`; the shutdown arm normally resolves immediately.
                let _ = heartbeat_with_reason(
                    &api,
                    &settings,
                    WorkerStatus::Unhealthy,
                    None,
                    Some(reason),
                )
                .await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))) => {}
                    _ = shutdown_signal() => return Ok(()),
                }
                continue;
            }
        }
        if !health.is_usable() && Instant::now() >= next_gpu_recheck {
            next_gpu_recheck = Instant::now() + GPU_HEALTH_RECHECK;
            recheck_gpu_health(&api, &settings, &mut health).await?;
        }
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
            result = poll_once(&api, &settings, &mut idle_heartbeat, &health) => result,
            _ = shutdown_signal() => {
                // Clean-idle shutdown: no job in flight, so the pre-existing Offline heartbeat +
                // return is preserved exactly.
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        };
        match claim {
            Ok(None) => {
                lock_failures = 0;
                // Claiming nothing is the proof of idleness both maintenance activities require:
                // no job is in flight, so neither a sweep nor a bundle copy can compete with one.
                // Every artifact lock a sweep takes is non-blocking, so an in-use model is skipped
                // rather than waited on.
                //
                // The handle is polled, never awaited: this arm sits directly on the claim path,
                // so the loop must come straight back round to claim the next job while
                // maintenance is still running. Work is also skipped entirely while a predecessor
                // is in flight, so slow sweeps and slow copies cannot stack up.
                if maintenance_task
                    .as_ref()
                    .is_some_and(tokio::task::JoinHandle::is_finished)
                {
                    maintenance_task = None;
                }
                if maintenance_task.is_none() {
                    if Instant::now() >= next_retention_checkpoint {
                        // Retention wins whenever it is due. The reverse ordering could starve
                        // retention indefinitely on a worker with a steady promotion stream, and
                        // retention is what keeps the cache inside its size limit — the limit
                        // promotion admission itself is judged against.
                        //
                        // This does NOT make starving promotion impossible, only bounded by an
                        // assumption: the next checkpoint is armed when this one is STARTED, so a
                        // sweep that runs longer than the interval is already due again when it
                        // finishes and sweeps run back to back, with no idle turn left over for a
                        // drain. That holds only while sweep duration stays under the interval.
                        // It is self-stabilizing rather than a defect — a sweep that slow means
                        // the cache is over its limit, which is the condition promotion must not
                        // be adding to — so the ordering stands, but the guarantee is
                        // "promotion yields to retention", not "promotion always runs".
                        next_retention_checkpoint =
                            Instant::now() + RESOLVED_CACHE_RETENTION_INTERVAL;
                        maintenance_task =
                            Some(spawn_retention_checkpoint(settings.data_dir.clone(), false));
                    } else if resolved_cache_promotion::work_pending(&settings.data_dir) {
                        // sc-19706: in-memory check only — no store open, no stat — because this
                        // sits on the claim path exactly like the checkpoint gate above.
                        maintenance_task = Some(spawn_promotion_drain(settings.clone()));
                    }
                }
            }
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
                // lock contention is explained rather than silently losing the claim to another poller.
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

/// True when the API reports SQLite's typed BUSY/LOCKED class for the jobs database.
fn is_database_locked(error: &WorkerError) -> bool {
    matches!(
        error,
        WorkerError::Api {
            code: Some(code), ..
        } if code == "database_locked"
    )
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
    health: &GpuHealth,
) -> WorkerResult<Option<JobSnapshot>> {
    // sc-7824 (epic 7819): pick up a live GPU-memory-limit change here, before claiming the next
    // job, so a Settings slider move applies between jobs (not mid-flight) with no worker restart.
    // No-op unless this is the MLX worker and the desktop has written the live-handoff file.
    if settings.gpu_id == "mlx" {
        generator_cache::sync_gpu_memory_limit(&settings.config_dir);
    }
    if idle_heartbeat.should_send() {
        // sc-16260: an unusable GPU reports `unhealthy` + the host-side remedy here instead of
        // `idle`. It keeps heartbeating on the same cadence — the process IS alive, it must stay
        // out of the API's stale sweep, and it has to be able to recover — but `idle` on a worker
        // that has withdrawn every capability it serves is the misleading state this story exists
        // to remove: an operator would read "Ready" off a worker that will never claim anything.
        match health.reason() {
            None => heartbeat(api, settings, WorkerStatus::Idle, None).await?,
            Some(reason) => {
                heartbeat_with_reason(
                    api,
                    settings,
                    WorkerStatus::Unhealthy,
                    None,
                    Some(reason.to_owned()),
                )
                .await?;
            }
        }
        idle_heartbeat.mark_sent();
    }
    // The claim POST still goes out while unhealthy, deliberately. The store refuses it twice
    // over (no advertised capability, plus the `Unhealthy` backstop in `worker_supports_job`), so
    // this costs one cheap request per poll and keeps the loop shape identical — which is what
    // makes recovery instant: the moment the re-probe passes and re-registration lands, the very
    // next claim is served with no extra transition to get right.
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
    heartbeat_with_reason(api, settings, status, current_job_id, None).await
}

/// [`heartbeat`] plus the `status_reason` a [`WorkerStatus::Unhealthy`] worker carries
/// (sc-16260) — the host-side remedy, so the Queue screen can explain a stalled queue.
///
/// Separate from [`heartbeat`] rather than an extra parameter on it: `heartbeat` has ~50 call
/// sites, every one of which reports `Idle`/`Busy`/`Offline` and would have to pass `None`.
/// Only the idle-poll path in [`poll_once`] ever has a reason to send.
pub(crate) async fn heartbeat_with_reason(
    api: &ApiClient,
    settings: &Settings,
    status: WorkerStatus,
    current_job_id: Option<&str>,
    status_reason: Option<String>,
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
                status_reason,
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
        // sc-19708: the ONE pre-loader guard. Typed model-source admission happens here for every
        // job type, before any handler runs; handlers never carry availability or cache policy.
        //
        // sc-19707: run it on the BLOCKING pool. The guard walks the resolved-cache journal, stats
        // whole closures, and — when it leases a local bundle — takes the entry's shared artifact
        // lock through the blocking `FileExt::lock_shared`. An evictor holding that lock
        // exclusively would otherwise stall this runtime worker thread, not just this job.
        let guard_job_type = job.job_type.clone();
        let guard_payload = job.payload.clone();
        let guard_settings = settings.clone();
        let guard_task = tokio::task::spawn_blocking(move || {
            external_library_runtime::RuntimeSourceGuard::begin(
                &guard_job_type,
                &guard_payload,
                &guard_settings,
            )
        });
        // The guard's admission pass can re-verify a multi-GB resolved bundle, which outlasts the
        // API's stale-worker timeout — the sc-13856 hazard at a new call site. Awaiting the handle
        // bare let the sweep mark a still-healthy worker offline mid-verification (the Krea bf16
        // 90s lost-heartbeat incident), so the wait must pump heartbeats on the progress interval.
        let source_guard = progress::heartbeat_while_blocking(
            api,
            settings,
            &job.id,
            "model source guard",
            guard_task,
        )
        .await
        .map_err(|error| ("Model source unavailable.", error))?;
        let dispatch_result = match job.job_type {
            JobType::Placeholder => run_placeholder_job(api, settings, &job, &shutdown)
                .await
                .map_err(|error| ("Placeholder job failed.", error)),
            // Native MLX image generation, served in-process by the linked mlx-gen
            // engine on the macOS Apple-Silicon GPU worker (epic 3018). Off macOS the
            // capability is never advertised, so this arm is unreachable there.
            JobType::ImageGenerate => run_image_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Image generation failed.", error)),
            // Plain Image Edit (sc-3513): the distinct `image_edit` job type (`mode=edit_image`
            // + `sourceAssetId`, epic 2427) shares the generate handler — it dispatches on
            // payload model+mode (qwen/flux2/sdxl edit streams), not job type. The API only
            // routes MLX-eligible edit models here (jobs_store::image_job_is_mlx_eligible); off
            // macOS the `image_edit` capability is never advertised, so this arm is unreachable.
            JobType::ImageEdit => run_image_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Image edit failed.", error)),
            // Native MLX tile-ControlNet detail refine (epic 3041, sc-3060), served in-process
            // by the engine on the macOS Apple-Silicon GPU worker. Off macOS the capability is
            // never advertised, so this arm is unreachable there and the job remains queued.
            JobType::ImageDetail => run_image_detail_job(api, settings, &job)
                .await
                .map_err(|error| ("Image detail enhancement failed.", error)),
            // SenseNova-U1 visual question answering + Document Studio interleave (epic 3180,
            // sc-3905). These bypass the `Generator` registry and call the concrete `T2iModel`
            // directly (text / text+images output the `GenerationOutput` contract can't express).
            // On Mac the API routes them here through `understanding_job_is_mlx_eligible`; off-Mac
            // the candle worker advertises the same capabilities and uses the candle handlers below.
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
            // MLX-eligible replace_person jobs here (`jobs_store::video_job_is_mlx_eligible`). The
            // off-Mac candle lane serves eligible Wan-VACE/SCAIL-2 replacement; unsupported models
            // are refused and remain queued.
            JobType::PersonReplace => run_video_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Person replacement failed.", error)),
            // Pure audio synthesis (SceneWorks Audio Studio, epic 13400 / sc-13404): Kokoro TTS (and
            // future SFX/music) served in-process by the runtime's candle audio lane
            // (`inference_runtime::load_audio` → `catalog().audio()`). Advertised only by a worker
            // that links the audio registry (the macOS mlx worker, whose `runtime-macos` bundle ships
            // it default-on); a worker without the lane never advertises `audio_generate`, so this arm
            // is unreachable there and the job stays queued for a capable worker.
            JobType::AudioGenerate => run_audio_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Audio generation failed.", error)),
            // Native MLX LoRA/LoKr training (epic 3039, sc-3043/3049), served in-process
            // by the linked mlx-gen engine on the macOS Apple-Silicon GPU worker. The API
            // routes only MLX-native families here (jobs_store::training_job_is_mlx_eligible);
            // unsupported shapes are refused and remain queued for a compatible native trainer.
            // Off macOS the execute capability is advertised only by a linked native trainer.
            JobType::LoraTrain => run_lora_train_job(api, settings, &job)
                .await
                .map_err(|error| ("LoRA training failed.", error)),
            // ControlNet Training Studio (epic 10159, sc-10162): render the per-image control
            // condition from the plan's dataset (A1/A2), then train the control branch through the
            // same native executor as LoRA (`krea_control` → `krea_2_control`). Candle-only today; the
            // routing gate keeps it on a candle worker (or the linked mlx build), the stub fails loudly
            // elsewhere.
            JobType::ControlTraining => {
                control_training_jobs::run_control_training_job(api, settings, &job)
                    .await
                    .map_err(|error| ("ControlNet training failed.", error))
            }
            // Native MLX JoyCaption dataset captioning (epic 3550, sc-3556). The API
            // routes only `captioner=joy_caption` jobs here; other captioners remain queued unless
            // another compatible native captioner registers.
            JobType::TrainingCaption => run_training_caption_job(api, settings, &job)
                .await
                .map_err(|error| ("Training captioning failed.", error)),
            JobType::DatasetParquetImport => run_dataset_parquet_import_job(api, settings, &job)
                .await
                .map_err(|error| ("Parquet dataset import failed.", error)),
            // Dataset Doctor CLIP-embedding analysis (sc-6535): the native MLX or Candle worker embeds
            // every dataset image (clip_vit_l14) and POSTs the content-hash sidecar.
            JobType::DatasetAnalysis => run_dataset_analysis_job(api, settings, &job)
                .await
                .map_err(|error| ("Dataset analysis failed.", error)),
            JobType::CatalogAnalysis => run_catalog_analysis_job(api, settings, &job)
                .await
                .map_err(|error| ("Catalog analysis failed.", error)),
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
            // (engines::registry_capabilities from the registered core_llm provider); without a native
            // text provider the capability is not advertised and the job remains queued.
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
            JobType::PersonDetect => run_person_detect_job(api, settings, &job)
                .await
                .map_err(|error| ("Person detection failed.", error)),
            // DWPose whole-body pose detection (epic 3482, sc-3487 Mac / sc-5496 off-Mac):
            // RTMW via onnxruntime, replacing the Python rtmlib path — CoreML EP on the
            // macOS MLX worker, CUDA EP on the off-Mac candle GPU worker. Available on Mac
            // AND the candle lane; on a candle-disabled box `PoseDetect` is never advertised, so the
            // job remains queued and this falls to the `_` arm only if called defensively.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::PoseDetect => run_pose_detect_job(api, settings, &job)
                .await
                .map_err(|error| ("Pose detection failed.", error)),
            // SCRFD 5-point landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac + the candle
            // SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482), served in-process
            // for the Key Point Library. Available on Mac AND the candle lane; on a candle-disabled box
            // `KpsExtract` is never advertised, so the job remains queued and this falls to the `_` arm
            // only if called defensively.
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
            // Windows/CUDA candle lane; on a build without either lane `ImageUpscale` is never advertised,
            // so the job remains queued. The routing oracle admits only the engines each native lane serves.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::ImageUpscale => run_image_upscale_job(api, settings, &job)
                .await
                .map_err(|error| ("Image upscale failed.", error)),
            // Dataset Doctor one-tap upscale (sc-6539): Real-ESRGAN over flagged low-res items, then
            // re-point each via the API. Same engine + worker lanes as image_upscale.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::DatasetUpscale => run_dataset_upscale_job(api, settings, &job)
                .await
                .map_err(|error| ("Dataset upscale failed.", error)),
            // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt segmentation,
            // served in-process by `segment_jobs::run_image_segment_job` — a box prompt → a binary
            // inpaint mask asset for the Image Editor. Advertised by both native workers; Candle
            // point prompts fail closed because the pinned provider exposes box PVS only.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::ImageSegment => segment_jobs::run_image_segment_job(api, settings, &job)
                .await
                .map_err(|error| ("Smart-select segmentation failed.", error)),
            // SeedVR2 video upscaling (epic 4811): one-step super-resolution — native MLX on Mac (sc-4816)
            // / candle CUDA on Windows (sc-5928). SceneWorks' first video upscaler: decodes the source
            // clip, runs the temporal-chunked 5D upscale, re-encodes, and passes the source audio through.
            // Available on Mac + the Windows/CUDA candle lane; elsewhere `VideoUpscale` is never advertised
            // (no torch path), so it falls to the `_` arm and the routing oracle reports it unsupported.
            #[cfg(any(
                target_os = "macos",
                all(not(target_os = "macos"), feature = "backend-candle")
            ))]
            JobType::VideoUpscale => {
                video_jobs::seedvr2::run_video_upscale_job(api, settings, &job)
                    .await
                    .map_err(|error| ("Video upscale failed.", error))
            }
            JobType::PersonTrack => run_person_track_job(api, settings, &job)
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
        };
        // Success releases the operation-owned source sessions; failure re-probes the exact bound
        // sources so a mid-load disconnect surfaces as the typed unavailable condition instead of
        // a raw loader error (sc-19708). An error with the source still present stays verbatim.
        match dispatch_result {
            Ok(()) => source_guard
                .finish_success()
                .map_err(|error| ("Model source session cleanup failed.", error)),
            Err((message, error)) => Err((message, source_guard.classify_failure(settings, error))),
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
                // sc-16247: this detail is what `QueueScreen` renders as `job.error`, so it is the
                // last point before a raw `DriverError(...)` reaches the user. `classify_engine_error`
                // already annotates the lanes that route through it (the reported krea_2_turbo path),
                // but ~35 other load seams build `WorkerError::Engine` directly — a host driver
                // problem hits all of them identically. Annotating here catches every one, and is a
                // no-op when the guidance is already present.
                let original_detail = error.to_string();
                let recovery_detail = if settings.gpu_id == "mlx" {
                    mlx_worker_recovery::global().observe(&original_detail)
                } else {
                    None
                };
                if let Some(detail) = &recovery_detail {
                    emit_event_value(
                        Level::ERROR,
                        json!({
                            "event": "mlx_h3_i2v_worker_poisoned",
                            "workerId": settings.worker_id,
                            "gpuId": settings.gpu_id,
                            "jobId": job.id,
                            "originalError": original_detail,
                            "jobError": detail,
                        }),
                    );
                }
                let poisoned = recovery_detail.is_some();
                let detail = recovery_detail
                    .unwrap_or_else(|| annotate_cuda_driver_failure(&original_detail));
                if poisoned {
                    // A clean child exit is not an abnormal death, so the supervisor deliberately
                    // will not invent a second terminal attribution for it. Persist the original
                    // timeout before arming that exit. Transport/API failures keep this process
                    // quarantined and retry without claiming; shutdown cancellation stops the
                    // retry and leaves the loop's no-claim backstop armed.
                    let mut attempt = 0_u32;
                    let _ = mlx_worker_recovery::persist_terminal_failure_with(
                        mlx_worker_recovery::global(),
                        || {
                            attempt = attempt.saturating_add(1);
                            let post_attempt = attempt;
                            let detail = detail.clone();
                            let job_id = job.id.clone();
                            let shutdown = shutdown.clone();
                            async move {
                                let Err(terminal_error) =
                                    fail_job(api, &job_id, message, Some(detail)).await
                                else {
                                    return mlx_worker_recovery::TerminalPersistenceAttempt::Persisted;
                                };
                                emit_event_value(
                                    Level::ERROR,
                                    json!({
                                        "event": "mlx_h3_i2v_terminal_failure_retry",
                                        "workerId": settings.worker_id,
                                        "gpuId": settings.gpu_id,
                                        "jobId": &job_id,
                                        "attempt": post_attempt,
                                        "error": terminal_error.to_string(),
                                    }),
                                );
                                if shutdown.is_cancelled() {
                                    return mlx_worker_recovery::TerminalPersistenceAttempt::Stop;
                                }
                                let _ = heartbeat_with_reason(
                                    api,
                                    settings,
                                    WorkerStatus::Unhealthy,
                                    Some(&job_id),
                                    mlx_worker_recovery::global().quarantine_reason(),
                                )
                                .await;
                                tokio::time::sleep(Duration::from_secs(
                                    settings.poll_seconds.max(1),
                                ))
                                .await;
                                mlx_worker_recovery::TerminalPersistenceAttempt::Retry
                            }
                        },
                    )
                    .await;
                } else {
                    let _ = fail_job(api, &job.id, message, Some(detail)).await;
                }
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
    // Do not advertise Idle after a poison latch. The next loop turn reports Unhealthy and exits
    // before polling, so no scheduler observation can mistake this process for claimable between
    // the active job's truthful failure and the supervisor's replacement.
    if settings.gpu_id != "mlx" || mlx_worker_recovery::global().can_claim() {
        let _ = heartbeat(api, settings, WorkerStatus::Idle, None).await;
    }
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
mod test_env;

// NTFS disk discipline for the multi-gigabyte weight fixtures (StorageFull incident, 2026-08-20).
// The `set_len`-extended fixtures the memory gates need are holes on APFS/ext4 and FULL allocations
// on NTFS, so on Windows they have to be marked sparse explicitly. See the module docs.
#[cfg(test)]
mod test_fixture_disk;

// Reads pinned download entries (repo/revision/files) out of the embedded builtin catalog so
// provisioning harnesses follow a manifest pin bump instead of mirroring it (sc-13810).
#[cfg(test)]
mod manifest_pins;

#[cfg(test)]
mod architecture_tests;

// Source-level drift guard for the bespoke candle preview wiring (epic 16948, sc-16962). Deliberately
// NOT cfg-gated to `backend-candle`: the lanes it guards are, and both candle CI lanes are
// dispatch-only, so a compiled test over them would never run on an ordinary PR — exactly when the
// "make it compile with `preview: Default::default()`" regression lands.
#[cfg(test)]
mod candle_preview_wiring_tests;

// The epic-17625 regression gate (sc-17637, AC9): no new job-time download, no new
// `<data_dir>/cache` weight destination. Deliberately NOT cfg-gated for the same reason as the guard
// above, only more so — every download helper and all of its call sites are gated
// `macos || backend-candle`, so on the required ubuntu/default-features `parity` lane none of that
// code is compiled at all and a gate inheriting those cfgs would never run. This one reads source
// text, so it fires on every platform and every PR.
#[cfg(test)]
mod job_time_download_guard;

// Pinned-snapshot provisioning helpers + the install-layout smokes (sc-13797/sc-13810). Compiled on
// EVERY platform — the download/layout code is platform-agnostic; only the live-network smoke inside
// carries `#[cfg(target_os = "macos")]` for lane confinement.
#[cfg(test)]
mod snapshot_install;

#[cfg(test)]
mod tests;
