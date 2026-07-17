//! Native video-generation jobs — runtime pipeline + procedural stub (epic 3018, sc-3033).
//!
//! Parses the job into a [`VideoRequest`], produces a single video (one mp4 asset,
//! unlike images which batch `count`), and reports a flat "fact" the Rust API turns
//! into an indexed asset (mirroring `video_generation_result` in the Python worker's
//! `video_adapters.py`). The shared encode pipeline takes the engine's video output
//! shape — RGB8 `frames` + `fps` + an optional synchronized `audio` track — writes
//! the frames to an mp4 (libx264), muxes a 16-bit-PCM WAV as AAC when audio is present
//! (`-shortest`), remuxes `+faststart` (WKWebView range-seek), and extracts a poster
//! frame. It reuses [`crate::media_jobs::run_ffmpeg`] (binary resolution + the
//! periodic-heartbeat / cooperative-cancel loop).
//!
//! sc-3033 ships only the **procedural stub** generator (a moving gradient + a quiet
//! synchronized tone for the LTX family, mirroring the engine: LTX emits audio, Wan
//! does not). The real in-process MLX video models — Wan2.2 (sc-3034) and LTX-2.3 +
//! audio (sc-3035) — link the `mlx-gen-wan` / `mlx-gen-ltx` provider crates and decode
//! `gen_core::GenerationOutput::Video { frames, fps, audio }` into the same
//! [`DecodedVideo`], so the encode/mux/poster path below is unchanged for them.

use std::f32::consts::PI;
use std::path::Path;

use sceneworks_core::video_request::{
    duration_limit_error, fps_limit_error, is_ltx_model, VideoRequest,
};

// Used only by the video generation metrics builders below, which are themselves
// gated to the macOS / backend-candle lanes (the shared `generate_video` funnel) —
// so gate the import to match, or the Linux-no-candle "neither" build sees it as
// unused (`-D warnings`, the sc-10404 dead-code trap).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::contracts::GenerationMetrics;

use super::*;
use crate::media_jobs::{run_ffmpeg, FfmpegContext};

// Real MLX Wan2.2 generation (macOS, sc-3034). `runtime-macos` explicitly includes all three Wan
// registrations in its validated media catalog.
#[cfg(target_os = "macos")]
use crate::image_jobs::{classify_adapter, load_reference_image, lora_path};
// epic 3720 (sc-3724): the backend-neutral generation contract types come from `gen_core` while the
// compile-time runtime bundle decides which backend catalog this module loads from.
// Backend-neutral contract types shared by the macOS MLX video path AND the Windows candle video
// lane (sc-5097): the streaming driver (`generate_video`), the output decode
// (`run_loaded_video_generation`), and `VideoGenInput`/`video_load_spec` are all backend-neutral, so
// these types compile on both lanes. `cfg(target_os)` decides which provider crate registered the
// video engine, not which contract types this module names.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    AdapterSpec, CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Generator,
    LoadPhase, LoadSpec, Precision, Progress, Quant, WeightsSource,
};
// MLX-only contract types (LoRA classification, MoE experts) — the candle video lane uses none of these.
#[cfg(target_os = "macos")]
use gen_core::{AdapterKind, MoeExpert};
// VACE conditioning + replacement types are shared by the MLX Wan-VACE path and the candle Wan-VACE
// lane (sc-5494), so they are available on both the macos and windows+backend-candle builds.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{Image, ReplacementMode};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::character_store::CharacterStore;
// Frame-count stride coercion (Wan needs frames ≡ 1 mod 4; LTX snaps to 8k+1; Mochi snaps to 6k+1)
// — used by the MLX path and the candle entry (sc-5097).
//
// Mochi's stride is NOT interchangeable with Wan's (sc-11992, epic 1788): `wan_frame_count(150)` is
// 149, `149 % 6 == 5`, and the Mochi runtime's `validate_request` hard-rejects anything but `1 + 6k`
// — so the shipped 5 s @ 30 fps default (150 raw) fails outright on the Wan stride. `mochi_frame_count`
// (B3/sc-11993) snaps to the NEAREST 6k+1 (151), the count the manifest documents for that default.
// See `frames_are_the_mochi_lattice_not_wans` for the exact failure mode.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use sceneworks_core::video_request::{video_frame_count, wan_frame_count};
// `ltx_frame_count` is the MLX LTX arm's own stride expression (the candle lane resolves LTX through
// the shared `video_frame_count` ladder, so the candle LIB never names it) and is also asserted by
// the lattice + asset-fact tests, which run on EVERY cfg — hence `any(macos, test)`.
// `mochi_frame_count` is named only by tests that are themselves macOS/candle-gated.
// Each is gated to exactly the configs that use it: an import left dead on a single cfg fails the
// parity lane's `-D warnings` while a macOS check stays green (sc-10404's trap).
#[cfg(any(target_os = "macos", test))]
use sceneworks_core::video_request::ltx_frame_count;
#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
use sceneworks_core::video_request::mochi_frame_count;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use std::time::{Duration, Instant};

/// Stub adapter id recorded on generated assets — matches the Python
/// `ProceduralVideoAdapter.id` so the asset sidecar reads identically.
const STUB_ADAPTER: &str = "procedural_video";
const CANCEL_MESSAGE: &str = "Video generation canceled by user.";

/// Decoded video ready for muxing — the worker-side shape both the procedural stub
/// (sc-3033) and the real engine output (`gen_core::GenerationOutput::Video`,
/// sc-3034/3035) feed into [`encode_media`]. Mirrors the engine contract: `frames`
/// are RGB8, `audio` is `Some` for LTX (a synchronized track) and `None` for Wan.
/// The frames are held in memory (the engine returns them that way); the duration
/// clamp (≤30s) in [`VideoRequest`] bounds the footprint.
struct DecodedVideo {
    frames: Vec<RgbFrame>,
    fps: u32,
    audio: Option<AudioTrack>,
}

/// One RGB8 frame, row-major, `pixels.len() == width * height * 3` (the engine's
/// `Image`).
struct RgbFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Interleaved PCM audio — the engine's `AudioTrack` (LTX-2.3 synchronized audio).
struct AudioTrack {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

/// The native (MLX) video engine a `run_video_generate_job` request routes to — the routing DECISION,
/// separated from execution (sc-8828, F-026; mirrors the image lane's `ImageRoute`). `resolve_video_route`
/// runs the predicate ladder ONCE and returns this; the caller `match`es to run the (non-uniformly-typed)
/// generators. Preserves the historical predicate order + per-family engine exactly, so routing is
/// byte-identical. `replace_person` (mode-gated, resolve-or-error) is three distinct variants — SCAIL-2,
/// dual-expert Wan2.2 VACE-Fun, single-expert Wan-VACE — because each drives a different backend/checkpoint.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VideoRoute {
    /// `replace_person` on a `scail2_*` model → SCAIL-2 cross-identity replacement (sc-5452). Carries the
    /// resolved engine id.
    ReplacePersonScail2(&'static str),
    /// `replace_person` on `wan_2_2_vace_fun_14b` → the dual-expert Wan2.2 VACE-Fun engine (sc-3459).
    ReplacePersonWanVaceFun,
    /// `replace_person` on any other replace-capable model → single-expert Wan-VACE (sc-3521).
    ReplacePersonWanVace,
    /// `extend_clip` / `video_bridge` on `wan_2_2_ti2v_5b` (weights available) → native Wan-VACE
    /// extend/bridge, falling back to the TI2V-5B keyframe path if the VACE snapshot is unprovisioned
    /// (sc-3812). The fallback is an execution-time detail handled in the dispatch arm.
    WanVaceExtendBridge,
    /// Wan txt2video / i2v on a resolvable Wan model (sc-3034). Carries the resolved engine id.
    Wan(&'static str),
    /// LTX+audio (sc-3035). Carries the resolved engine id.
    Ltx(&'static str),
    /// Stable Video Diffusion. Carries the resolved engine id.
    Svd(&'static str),
    /// Bernini (epic 4699): Qwen2.5-VL planner + Wan2.2-T2V-A14B renderer. Carries the resolved engine id.
    Bernini(&'static str),
    /// SCAIL-2 standalone character animation (epic 5439). Carries the resolved engine id.
    Scail2(&'static str),
    /// Mochi 1 text-to-video (epic 1788 / sc-11992). Carries the resolved engine id. t2v ONLY —
    /// [`mochi_available`] gates the mode, since `conditioning: []` means there is no other shape.
    Mochi(&'static str),
    /// No native engine matched (or weights unresolved) → the procedural stub, after
    /// `ensure_video_engine_weights` fails a known-but-unprovisioned engine loudly (sc-4176).
    Stub,
}

/// Run the native video dispatch predicate ladder ONCE and return the [`VideoRoute`]. Mirrors the
/// historical inline ladder EXACTLY — same predicate order, same per-family engine — so routing is
/// byte-identical (sc-8828). Pure decision: no I/O, no generation. `replace_person` is dispatched by
/// mode first (resolve-or-error semantics live in the execution arms), then the engine-id/availability
/// ladder for every other mode.
#[cfg(target_os = "macos")]
fn resolve_video_route(request: &VideoRequest, settings: &Settings) -> VideoRoute {
    if request.mode == "replace_person" {
        if let Some(engine_id) = scail2_engine_id(&request.model) {
            VideoRoute::ReplacePersonScail2(engine_id)
        } else if request.model == "wan_2_2_vace_fun_14b" {
            VideoRoute::ReplacePersonWanVaceFun
        } else {
            VideoRoute::ReplacePersonWanVace
        }
    } else if matches!(request.mode.as_str(), "extend_clip" | "video_bridge")
        && wan_engine_id(&request.model) == Some("wan2_2_ti2v_5b")
        && wan_available(request, settings)
    {
        VideoRoute::WanVaceExtendBridge
    } else if let Some(engine_id) =
        wan_engine_id(&request.model).filter(|_| wan_available(request, settings))
    {
        VideoRoute::Wan(engine_id)
    } else if let Some(engine_id) =
        ltx_engine_id(&request.model).filter(|_| ltx_available(request, settings))
    {
        VideoRoute::Ltx(engine_id)
    } else if let Some(engine_id) =
        svd_engine_id(&request.model).filter(|_| svd_available(request, settings))
    {
        VideoRoute::Svd(engine_id)
    } else if let Some(engine_id) =
        bernini_engine_id(&request.model).filter(|_| bernini_available(request, settings))
    {
        VideoRoute::Bernini(engine_id)
    } else if let Some(engine_id) =
        scail2_engine_id(&request.model).filter(|_| scail2_available(request, settings))
    {
        VideoRoute::Scail2(engine_id)
    } else if let Some(engine_id) =
        mochi_engine_id(&request.model).filter(|_| mochi_available(request, settings))
    {
        // Mochi 1 (epic 1788 / sc-11992). Appended at the TAIL of the ladder: `mochi_engine_id`
        // matches only `mochi_1`, and no earlier predicate can match that id, so routing for every
        // pre-existing model stays byte-identical. `mochi_available` folds in the t2v-only mode gate.
        VideoRoute::Mochi(engine_id)
    } else {
        VideoRoute::Stub
    }
}

/// The candle (Windows/CUDA/Linux) video engine a `run_video_generate_job` request routes to — the
/// candle-lane sibling of [`VideoRoute`] (sc-8828, F-026). Every arm is gated on
/// `settings.backend_candle_enabled`; when that is off (default) the resolver returns
/// [`CandleVideoRoute::Stub`] so routing is unchanged until parity is accepted. Conditioning shapes
/// never reach the candle lane — the router's `video_job_is_candle_eligible` confines it — so this is
/// a narrow replace/animate/extend/txt2video ladder.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandleVideoRoute {
    /// `replace_person` on a `scail2_*` model → candle SCAIL-2 replacement (sc-6837). Carries the id.
    ReplacePersonScail2(&'static str),
    /// `replace_person` on any other candle-VACE model → candle Wan-VACE replacement (sc-5494).
    ReplacePersonWanVace,
    /// `animate_character` on a `scail2_*` model → candle SCAIL-2 animation (sc-6837). Carries the id.
    AnimateScail2(&'static str),
    /// `extend_clip` / `video_bridge` → candle Wan-VACE extend/bridge (sc-5494).
    WanVaceExtendBridge,
    /// A `bernini` VIDEO job (t2v + the editing/reference/multi-source modes) → `generate_candle_bernini`
    /// (sc-10997, epic 6562). A DISTINCT engine (the full Qwen planner + Wan2.2-T2V-A14B renderer), NOT a
    /// wan/ltx `is_candle_video_engine` id, so it is routed off the model id before the generic arm below.
    /// Carries the resolved engine id.
    Bernini(&'static str),
    /// A candle txt2video engine id → `generate_candle_video` (sc-5097).
    CandleVideo,
    /// An in-place ComfyUI Wan2.2 base model (`external_base_*`) → `generate_candle_wan_comfyui`
    /// (epic 10451 Phase 2c, sc-10671). Not an `is_candle_video_engine` id — routed off the forwarded row.
    WanComfyui,
    /// Candle disabled, or no candle engine matched → the procedural stub.
    Stub,
}

/// Run the candle video dispatch predicate ladder ONCE and return the [`CandleVideoRoute`]. Mirrors the
/// historical inline ladder EXACTLY — same predicate order + `backend_candle_enabled` gating — so
/// routing is byte-identical (sc-8828). Pure decision: no I/O, no generation.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_video_route(request: &VideoRequest, settings: &Settings) -> CandleVideoRoute {
    if !settings.backend_candle_enabled {
        return CandleVideoRoute::Stub;
    }
    if request.mode == "replace_person" {
        match candle_scail2_engine_id(&request.model) {
            Some(engine_id) => CandleVideoRoute::ReplacePersonScail2(engine_id),
            None => CandleVideoRoute::ReplacePersonWanVace,
        }
    } else if request.mode == "animate_character"
        && candle_scail2_engine_id(&request.model).is_some()
    {
        let engine_id = candle_scail2_engine_id(&request.model).expect("scail2 model");
        CandleVideoRoute::AnimateScail2(engine_id)
    } else if matches!(request.mode.as_str(), "extend_clip" | "video_bridge") {
        CandleVideoRoute::WanVaceExtendBridge
    } else if let Some(engine_id) = candle_bernini_engine_id(&request.model) {
        // Bernini (sc-10997, epic 6562): the full Qwen planner + Wan2.2-T2V-A14B renderer serves t2v +
        // the editing/reference/multi-source modes (v2v / r2v / rv2v / mv2v / ads2v). A DISTINCT engine
        // (`crate::inference_runtime::load("bernini")`), NOT a wan/ltx `is_candle_video_engine` id, so it is routed off the
        // model id BEFORE the generic candle-video arm below. Routed by model id, not weight availability —
        // `generate_candle_bernini` resolves-or-errors loudly if the `SceneWorks/bernini` snapshot is
        // unprovisioned (sc-11003), never degrading to a stub. The per-mode source media is validated when
        // the conditioning is assembled (`resolve_candle_bernini_conditioning`), mirroring the MLX lane.
        CandleVideoRoute::Bernini(engine_id)
    } else if wan_comfyui_available(request, settings) {
        // In-place ComfyUI Wan2.2 base (sc-10671): an `external_base_*` id, so it matches no
        // `is_candle_video_engine` arm below — route it off the forwarded `modelManifestEntry`.
        CandleVideoRoute::WanComfyui
    } else if is_candle_video_engine(&request.model) {
        CandleVideoRoute::CandleVideo
    } else {
        CandleVideoRoute::Stub
    }
}

/// The payload invariants every video job must satisfy before the worker does anything expensive —
/// the pure, synchronously testable seam [`run_video_generate_job`] delegates them to (sc-12297).
///
/// It exists as its own function for the reason `mochi_preflight` does: a decision left inline in an
/// `async` arm is one no test currently asserts the CALL of — the sc-11992 review caught exactly that,
/// a dropped gate surviving 835 green tests. Both checks here are refusals whose absence is invisible
/// until a render is already underway.
///
/// Note the reason is "nothing asserts the call", NOT "the arm is unreachable from a test". The latter
/// was the standing belief and it was false (sc-12318): an `async` arm taking `ApiClient`/`JobSnapshot`
/// is drivable — `ApiClient::new` does no I/O and `JobSnapshot` deserializes from a literal. See
/// `generate_mochi_using`, which pins its arm's decisions directly. `run_video_generate_job` has no
/// such harness yet, so this seam is still the only thing asserting these checks; that is a gap to
/// close if it ever matters, not a law of the code.
///
/// EVERY video job funnels through this: `VideoGenerate` / `VideoExtend` / `VideoBridge` /
/// `PersonReplace` all dispatch to `run_video_generate_job` (lib.rs). That is what makes it the
/// backstop for the API's `create_video_job` gate — which is the one that returns a caller a 400,
/// but only covers what IT enqueues, not a job replayed from a pre-sc-12297 row or produced by any
/// future non-HTTP path.
fn video_preflight(request: &VideoRequest) -> WorkerResult<()> {
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    // The model's declared `limits.hardMaxDuration`. NOT redundant with `mochi_preflight`'s fit
    // gate (sc-11992) — the closest thing that already existed: that one is macOS-only (so the
    // candle lane has no duration ceiling but this), Mochi-only, and answers a DIFFERENT question
    // — "does this fit THIS machine's RAM", not "is this within what the model can do". Memory is
    // not capability: a machine with the headroom to decode a 30s Mochi clip would render 30s off
    // a model trained for 5. Neither gate subsumes the other.
    if let Some(message) = duration_limit_error(
        &request.model,
        request.duration,
        &request.model_manifest_entry,
    ) {
        return Err(WorkerError::InvalidPayload(message));
    }
    // The model's declared `limits.fps`. NOT redundant with the duration cap above — they bound
    // different axes, and the cost/quality axis is `frames = duration × fps`, which only both
    // together bound. A *legally 5-second* mochi_1 request (cap 5 ✓) at 60 fps is 301 frames,
    // double the shipped 5s default's 151, and `301 % 6 == 1` clears the engine's own check, so
    // nothing downstream says no (sc-12347).
    //
    // `request.fps` is already resolved against the model's `defaults.fps` by `from_payload`, so a
    // payload that names no fps is judged on the value the model itself declares — not the blanket
    // 25, which is off-menu for 7 of the 10 shipped video models and would make this gate reject
    // every fps-less payload.
    if let Some(message) =
        fps_limit_error(&request.model, request.fps, &request.model_manifest_entry)
    {
        return Err(WorkerError::InvalidPayload(message));
    }
    Ok(())
}

/// Dispatch handler for `JobType::VideoGenerate`: generate, encode, and stream a
/// single video asset through the Rust GPU worker.
pub(crate) async fn run_video_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = VideoRequest::from_payload(&job.payload);
    video_preflight(&request)?;
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let plan = VideoPlan::new(&request, &project_path);
    if let Some(parent) = plan.media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing video.",
            None,
            backend,
        ),
    )
    .await?;

    // sc-3033 ships the procedural stub only; the real MLX video models (Wan sc-3034,
    // LTX+audio sc-3035) decode `GenerationOutput::Video` into a `DecodedVideo` here.
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    let seed = resolve_video_seed(&request);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.2,
            "Rendering frames.",
            None,
            backend,
        ),
    )
    .await?;
    // sc-3459 (epic 3456): Wan2.2 VACE-Fun A14B routes to the NEW dual-expert VACE engine
    // `wan2_2_vace_fun_14b`. macOS is served natively (mlx-gen sc-6604, merged + pinned) via
    // `generate_wan_vace_fun` in the macOS block below. The native **candle** engine (sc-6605) is
    // not done, so on Windows/Linux (candle) and the no-backend stub path a VACE-Fun job must fail
    // honestly here — it must NEVER fall through to the Wan2.1 `generate_candle_wan_vace` /
    // `generate_wan_vace` backend, which would silently render with the WRONG checkpoint (the exact
    // failure the epic forbids).
    #[cfg(not(target_os = "macos"))]
    if request.model == "wan_2_2_vace_fun_14b" {
        return Err(WorkerError::InvalidPayload(
            "wan_2_2_vace_fun_14b: the native Wan2.2 VACE-Fun engine is macOS-only for now (the \
             candle backend is pending sc-6605). The job will not be routed to the Wan2.1 VACE \
             backend. Choose another model on this platform."
                .to_owned(),
        ));
    }
    // Generate: real MLX on macOS for Wan (sc-3034) / LTX+audio (sc-3035) models with
    // resolvable weights, else the procedural stub (non-macOS or missing weights = stub).
    // replace_person (sc-3521) always routes to the native Wan-VACE provider regardless of the
    // user-picked (replace-capable) model — the native equivalent of the torch `WanVACEPipeline`
    // path. It errors clearly if the VACE snapshot is unprovisioned rather than degrading to the
    // procedural stub (a stubbed person-replace would be meaningless). It also reports the honest
    // `replacementStatus` the asset sidecar folds in (project_store::build_video_sidecar_parts).
    #[cfg(target_os = "macos")]
    let (decoded, adapter, raw_settings, replacement_status) =
        match resolve_video_route(&request, settings) {
            // sc-5452: SCAIL-2 is a higher-quality cross-identity replacement backend behind the same
            // YOLO11 → ByteTrack → SAM3 person-track pipeline. A `scail2_14b` person-replace job routes
            // to SCAIL-2 (the tracked person's masks + the character reference → the engine's
            // replacement conditioning, `replace_flag = true`); every other replace-capable model keeps
            // native Wan-VACE (sc-3521). Routed by model id, not weight availability: like the Wan-VACE
            // path, `generate_scail2_replace` resolves-or-errors loudly if the snapshot is unprovisioned
            // (a person-replace must never silently degrade to a different backend or the stub). Both
            // report the honest `replacementStatus` the asset sidecar folds in.
            VideoRoute::ReplacePersonScail2(engine_id) => {
                let (decoded, status) = generate_scail2_replace(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    // The replace_person path doesn't resolve user LoRAs (sc-5452), so no lightning recipe.
                    decoded,
                    SCAIL2_ADAPTER,
                    scail2_raw_settings(&request, false),
                    Some(status),
                )
            }
            VideoRoute::ReplacePersonWanVaceFun => {
                // sc-3459: the native dual-expert Wan2.2 VACE-Fun engine (`wan2_2_vace_fun_14b`,
                // mlx-gen sc-6604). Same replace_person conditioning as single-expert Wan-VACE, but the
                // dual-expert snapshot + engine — `generate_wan_vace_fun` resolves-or-errors loudly,
                // never falling back to the Wan2.1 `wan_vace` checkpoint.
                let (decoded, status) =
                    generate_wan_vace_fun(api, settings, job, &request, &project_path, backend)
                        .await?;
                (
                    decoded,
                    WAN_VACE_FUN_ADAPTER,
                    wan_vace_raw_settings(&request, "wan2_2_vace_fun_14b"),
                    Some(status),
                )
            }
            VideoRoute::ReplacePersonWanVace => {
                let (decoded, status) =
                    generate_wan_vace(api, settings, job, &request, &project_path, backend).await?;
                (
                    decoded,
                    WAN_VACE_ADAPTER,
                    wan_vace_raw_settings(&request, "wan_vace"),
                    Some(status),
                )
            }
            VideoRoute::WanVaceExtendBridge => {
                // sc-3812 (tier C): route Wan extend/bridge to native Wan-VACE for genuine motion
                // continuity (the model attends to real source frames, not one boundary still). Falls
                // back to the sc-3357 single-frame TI2V-5B keyframe path when the VACE snapshot is
                // unprovisioned, so the mode keeps working on the weights the user already has. The
                // engine substitution under the `wan_2_2` pick is recorded honestly in raw-settings.
                match resolve_wan_vace_model_dir(settings) {
                    Ok(model_dir) => (
                        generate_wan_vace_extend_bridge(
                            api,
                            settings,
                            job,
                            &request,
                            &project_path,
                            backend,
                            model_dir,
                        )
                        .await?,
                        WAN_VACE_ADAPTER,
                        wan_vace_extend_raw_settings(&request),
                        None,
                    ),
                    Err(_) => {
                        let engine_id = "wan2_2_ti2v_5b";
                        (
                            generate_wan(
                                api,
                                settings,
                                job,
                                &request,
                                &project_path,
                                engine_id,
                                backend,
                            )
                            .await?,
                            WAN_ADAPTER,
                            wan_raw_settings(&request, engine_id),
                            None,
                        )
                    }
                }
            }
            VideoRoute::Wan(engine_id) => (
                generate_wan(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?,
                WAN_ADAPTER,
                wan_raw_settings(&request, engine_id),
                None,
            ),
            VideoRoute::Ltx(engine_id) => (
                generate_ltx(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?,
                LTX_ADAPTER,
                ltx_raw_settings(&request),
                None,
            ),
            VideoRoute::Svd(engine_id) => (
                generate_svd(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?,
                SVD_ADAPTER,
                svd_raw_settings(&request),
                None,
            ),
            VideoRoute::Bernini(engine_id) => {
                // Bernini (epic 4699 / sc-4707 + sc-4703): the full Qwen2.5-VL planner + Wan2.2-T2V-A14B
                // renderer. Serves text_to_video + the editing/reference video modes (video_to_video /
                // reference_to_video / reference_video_to_video); `generate_bernini` maps the SceneWorks mode
                // to the engine guidance task and resolves the source media into the planner conditioning.
                // (t2i/i2i image companion = a separate image-typed catalog id, tracked under epic 4699.)
                (
                    generate_bernini(
                        api,
                        settings,
                        job,
                        &request,
                        &project_path,
                        engine_id,
                        backend,
                    )
                    .await?,
                    BERNINI_ADAPTER,
                    bernini_raw_settings(&request),
                    None,
                )
            }
            VideoRoute::Scail2(engine_id) => {
                // SCAIL-2 (epic 5439 / sc-5448): Wan2.1-14B I2V character animation. `generate_scail2`
                // segments the reference image + driving frames with native SAM3, paints the color-coded
                // masks, and maps the SceneWorks mode to the engine task (animate_character → animation;
                // replace_person → replacement is wired in sc-5452). No torch path (mac-only engine).
                // It returns the resolved `lightning` bool so the effective-recipe record uses the
                // same resolution instead of a second `unwrap_or_default` pass (F-118).
                let (decoded, lightning) = generate_scail2(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    SCAIL2_ADAPTER,
                    scail2_raw_settings(&request, lightning),
                    None,
                )
            }
            VideoRoute::Mochi(engine_id) => {
                // Mochi 1 (epic 1788 / sc-11992): 10B AsymmDiT text-to-video, true CFG, pre-quantized
                // tier dirs. `generate_mochi` fetches an opted-into tier on demand, resolves the tier
                // dir + its shared T5/VAE co-requisite, runs the pre-flight memory fit gate (the
                // untiled AsymmVAE decode scales with clip length and MLX's OOM handler is an
                // unmappable `exit(-1)`), then drives the shared `generate_video` funnel. It returns
                // its own `rawSettings` so the record names the tier that was actually loaded.
                let (decoded, raw_settings) =
                    generate_mochi(api, settings, job, &request, engine_id, backend).await?;
                (decoded, MOCHI_ADAPTER, raw_settings, None)
            }
            VideoRoute::Stub => {
                // An MLX-routed video model whose snapshot didn't resolve must fail
                // loudly with the resolver's precise error instead of completing with
                // procedural stub output (sc-4176, epic 3482 "unsupported jobs error
                // loudly"). replace_person above already follows this rule; the stub
                // remains only for ids outside the engine families (test models,
                // not-yet-ported families).
                ensure_video_engine_weights(&request, settings)?;
                (
                    generate_stub_video(&request, seed),
                    STUB_ADAPTER,
                    stub_raw_settings(&request),
                    None,
                )
            }
        };
    // Windows/CUDA candle video lane (sc-5097): a real wan/ltx txt2video job runs through
    // `generate_candle_video` (the same neutral encode/mux path as MLX + the stub); anything the
    // candle lane doesn't serve stubs exactly as before. Gated on `backend_candle_enabled` (default
    // off → routing unchanged until parity). Conditioning shapes never reach here — the router's
    // `video_job_is_candle_eligible` confines the candle worker to txt2video.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (decoded, adapter, raw_settings, replacement_status) =
        match resolve_candle_video_route(&request, settings) {
            // sc-6837 (epic 6563): SCAIL-2 is a distinct cross-identity replacement backend (NOT Wan-VACE)
            // behind the same person-track pipeline. A `scail2_14b` replace_person job routes to the candle
            // SCAIL-2 engine (the character reference + the tracked person's color masks, `replace_flag`),
            // mirroring the macOS `generate_scail2_replace` dispatch; every other replace-capable model
            // keeps candle Wan-VACE. Routed by model id, not weight availability — `generate_candle_scail2_
            // replace` resolves-or-errors loudly (a person-replace must never silently degrade to a stub).
            CandleVideoRoute::ReplacePersonScail2(engine_id) => {
                let (decoded, status) = generate_candle_scail2_replace(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_SCAIL2_ADAPTER,
                    // The replace_person path doesn't resolve user LoRAs (sc-5452), so no lightning recipe.
                    candle_scail2_raw_settings(&request, false),
                    Some(status),
                )
            }
            CandleVideoRoute::ReplacePersonWanVace => {
                // Candle Wan-VACE person replacement (sc-5494) — the candle equivalent of the MLX
                // `wan_vace` path. The router (`video_request_candle_vace_eligible`) already confirmed the
                // candle-VACE model + the source clip + person track before this job reached the candle
                // worker; `generate_candle_wan_vace` resolves-or-errors loudly (a person-replace must never
                // silently degrade to a stub).
                let (decoded, status) =
                    generate_candle_wan_vace(api, settings, job, &request, &project_path, backend)
                        .await?;
                (
                    decoded,
                    CANDLE_WAN_VACE_ADAPTER,
                    wan_vace_raw_settings(&request, "wan_vace"),
                    Some(status),
                )
            }
            CandleVideoRoute::AnimateScail2(engine_id) => {
                // sc-6837: SCAIL-2 standalone character animation — a reference character + a driving video →
                // an animated clip. `generate_candle_scail2` segments the reference + driving frames with the
                // candle SAM3 segmenter, paints the color-coded masks, and runs the `scail2_14b` engine
                // (`animate_character` → engine task `animation`). The candle sibling of the macOS
                // `generate_scail2`; resolves-or-errors loudly (no torch fallback — a distinct candle engine).
                let (decoded, adapter, raw_settings) = generate_candle_scail2(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (decoded, adapter, raw_settings, None::<Value>)
            }
            CandleVideoRoute::WanVaceExtendBridge => {
                // Candle Wan-VACE extend/bridge (sc-5494): real source frames pinned at the kept positions +
                // a generated span (the candle equivalent of the MLX `generate_wan_vace_extend_bridge`).
                let decoded = generate_candle_wan_vace_extend_bridge(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_WAN_VACE_ADAPTER,
                    wan_vace_extend_raw_settings(&request),
                    None::<Value>,
                )
            }
            CandleVideoRoute::Bernini(engine_id) => {
                // sc-10997 (epic 6562): the candle Bernini VIDEO lane — the full Qwen planner +
                // Wan2.2-T2V-A14B renderer. `generate_candle_bernini` maps the SceneWorks mode to the engine
                // `video_mode` task (t2v / v2v / r2v / rv2v / mv2v / ads2v) and resolves the source media into
                // the planner conditioning (`resolve_candle_bernini_conditioning`) — empty for t2v, one or
                // more `VideoClip`s for the edit modes, `MultiReference` for the reference modes. The off-Mac
                // sibling of the macOS `generate_bernini`; resolves-or-errors loudly (no torch fallback — a
                // distinct candle engine, GPU-val gated on the `SceneWorks/bernini` weights, sc-11003).
                let decoded = generate_candle_bernini(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (
                    decoded,
                    CANDLE_BERNINI_ADAPTER,
                    candle_bernini_raw_settings(&request),
                    None::<Value>,
                )
            }
            CandleVideoRoute::CandleVideo => {
                let (decoded, adapter, raw_settings) =
                    generate_candle_video(api, settings, job, &request, &project_path, backend)
                        .await?;
                (decoded, adapter, raw_settings, None::<Value>)
            }
            CandleVideoRoute::WanComfyui => {
                let (decoded, adapter, raw_settings) = generate_candle_wan_comfyui(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    backend,
                )
                .await?;
                (decoded, adapter, raw_settings, None::<Value>)
            }
            CandleVideoRoute::Stub => (
                generate_stub_video(&request, seed),
                STUB_ADAPTER,
                stub_raw_settings(&request),
                None::<Value>,
            ),
        };
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    let (decoded, adapter, raw_settings, replacement_status) = (
        generate_stub_video(&request, seed),
        STUB_ADAPTER,
        stub_raw_settings(&request),
        None::<Value>,
    );
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding video.",
            None,
            backend,
        ),
    )
    .await?;
    // sc-12371: measure the clip BEFORE it is moved into the encoder. This is the single seam every
    // video job funnels through — whatever lane, engine or route produced `decoded` — so measuring
    // once here is what makes "the sidecar can lie about clip length" structurally impossible rather
    // than merely fixed for today's models. `video_asset_fact` cannot be called without it.
    let clip = EncodedClip::measure(&decoded);
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    encode_media(&plan.media_path, decoded, Some(ctx)).await?;

    let fact = video_asset_fact(&plan, seed, adapter, raw_settings, replacement_status, clip);
    let result = streaming_result(&plan, &fact, adapter);
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Generated video.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Per-job invariants for the single video this job produces.
struct VideoPlan {
    request: VideoRequest,
    genset_id: String,
    asset_id: String,
    created_at: String,
    family: String,
    /// `assets/videos/<genset>/<date>_<model>_<slug>.mp4` (project-relative).
    media_rel: String,
    /// Absolute path to the media file.
    media_path: PathBuf,
}

impl VideoPlan {
    fn new(request: &VideoRequest, project_path: &Path) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let asset_id = fresh_asset_id();
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "video", Some(42));
        // Sanitize the untrusted model id before it becomes a path component: it arrives
        // verbatim from the job payload, and a `../` / `\` / absolute id would otherwise
        // traverse out of the project dir here (F-003 / sc-11159). rust-api rejects such ids
        // at enqueue, but the worker is the trust boundary and must re-confine — slugify
        // collapses any separator/`..` to a single readable component (mirrors write_image_asset).
        let model_slug = slugify(&request.model, "model", None);
        // Nest under the per-generation id so two renders sharing date+model+slug
        // cannot collide on a flat path (mirrors the image + Python video adapters).
        let media_rel = format!(
            "assets/videos/{genset_id}/{}_{}_{slug}.mp4",
            &created_at[..10],
            model_slug
        );
        let media_path = project_path.join(&media_rel);
        Self {
            request: request.clone(),
            genset_id,
            asset_id,
            created_at,
            family,
            media_rel,
            media_path,
        }
    }
}

/// Resolve the seed, matching the Python `resolve_seed(seed, prompt)`: an explicit
/// seed wins, else the first 4 bytes of `sha256(prompt)` (its `hexdigest()[:8]`).
fn resolve_video_seed(request: &VideoRequest) -> i64 {
    if let Some(seed) = request.seed {
        return seed;
    }
    let digest = Sha256::digest(request.prompt.as_bytes());
    u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as i64
}

/// The asset's video family, from the resolved manifest entry when present, else
/// inferred from the model id (parity with the Python `VIDEO_MODEL_TARGETS` family).
fn resolve_family(request: &VideoRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
    {
        if !family.trim().is_empty() {
            return family.to_owned();
        }
    }
    if is_ltx_model(&request.model) {
        "ltx-video".to_owned()
    } else if request.model.starts_with("wan") {
        "wan-video".to_owned()
    } else {
        "video".to_owned()
    }
}

// ---------------------------------------------------------------------------
// Procedural stub generator (sc-3033). Real MLX models land in sc-3034/3035.
// ---------------------------------------------------------------------------

/// Build a deterministic placeholder clip: `frame_count` moving-gradient frames at
/// the request fps, plus a quiet synchronized tone for the LTX family (the engine
/// emits audio for LTX and none for Wan, so the stub mirrors that split — exercising
/// both the audio-mux and video-only encode paths).
fn generate_stub_video(request: &VideoRequest, seed: i64) -> DecodedVideo {
    let frame_count = request.frame_count();
    let fps = request.fps.max(1);
    let (width, height) = (request.width, request.height);
    let frames = (0..frame_count)
        .map(|index| RgbFrame {
            width,
            height,
            pixels: stub_video_rgb8(width, height, seed, index, frame_count),
        })
        .collect();
    let audio = is_ltx_model(&request.model).then(|| stub_audio_track(frame_count, fps));
    DecodedVideo { frames, fps, audio }
}

/// Deterministic per-frame pixels: a vertical gradient from a per-seed base colour to
/// white, with a bright vertical band that sweeps left→right across the clip so frames
/// differ (visible motion). Exactly `width * height * 3` RGB8 bytes.
fn stub_video_rgb8(width: u32, height: u32, seed: i64, index: u32, frame_count: u32) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let v_span = height.saturating_sub(1).max(1) as f32;
    // The sweeping band's centre column for this frame.
    let progress = index as f32 / frame_count.max(1) as f32;
    let band_centre = progress * width.saturating_sub(1).max(1) as f32;
    let band_half = (width as f32 * 0.06).max(1.0);
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / v_span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for x in 0..width {
            let dist = (x as f32 - band_centre).abs();
            if dist <= band_half {
                // Brighten toward white inside the band (1.0 at centre → 0 at edge).
                let highlight = 1.0 - dist / band_half;
                buffer.push(lerp(row[0], highlight));
                buffer.push(lerp(row[1], highlight));
                buffer.push(lerp(row[2], highlight));
            } else {
                buffer.extend_from_slice(&row);
            }
        }
    }
    buffer
}

fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

/// A quiet 220 Hz mono tone matching the clip length (`frame_count / fps` seconds) at
/// 48 kHz — enough to exercise the WAV-write + AAC-mux + `-shortest` path end to end.
fn stub_audio_track(frame_count: u32, fps: u32) -> AudioTrack {
    let sample_rate = 48_000u32;
    let duration = frame_count as f32 / fps.max(1) as f32;
    let n = (sample_rate as f32 * duration).round().max(1.0) as usize;
    let freq = 220.0f32;
    let samples = (0..n)
        .map(|i| (2.0 * PI * freq * (i as f32 / sample_rate as f32)).sin() * 0.2)
        .collect();
    AudioTrack {
        samples,
        sample_rate,
        channels: 1,
    }
}

// ---------------------------------------------------------------------------
// Encode pipeline: frames → mp4 (+ optional AAC audio) → faststart → poster.
// Reuses `media_jobs::run_ffmpeg`. Pure of the API except the optional `ctx`
// (heartbeat/cancel), so it is exercisable in tests with a real ffmpeg.
// ---------------------------------------------------------------------------

/// Write `decoded` to `media_path` as an mp4: frames → libx264, an optional 16-bit
/// PCM WAV muxed as AAC (`-shortest`), then a best-effort `+faststart` remux and
/// `.poster.jpg`. `media_path` is created (atomically renamed from a temp) only on
/// success; all intermediates are removed regardless of outcome.
async fn encode_media(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let frames_dir = media_path.with_extension("frames");
    let enc_tmp = media_path.with_extension("enc.mp4");
    let wav_tmp = media_path.with_extension("audio.wav");
    let mux_tmp = media_path.with_extension("mux.mp4");
    let result = encode_inner(
        media_path,
        decoded,
        ctx,
        &frames_dir,
        &enc_tmp,
        &wav_tmp,
        &mux_tmp,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let _ = tokio::fs::remove_file(&enc_tmp).await;
    let _ = tokio::fs::remove_file(&wav_tmp).await;
    let _ = tokio::fs::remove_file(&mux_tmp).await;
    if result.is_err() {
        // A failure (or cancel) before the atomic rename leaves no media_path; if the
        // rename itself half-completed, drop the partial so the asset never points at it.
        let _ = tokio::fs::remove_file(media_path).await;
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn encode_inner(
    media_path: &Path,
    decoded: DecodedVideo,
    ctx: Option<FfmpegContext<'_>>,
    frames_dir: &Path,
    enc_tmp: &Path,
    wav_tmp: &Path,
    mux_tmp: &Path,
) -> WorkerResult<()> {
    let fps = decoded.fps.max(1);
    let audio = decoded.audio;
    let frames = decoded.frames;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }

    // 1. Write the frame sequence (blocking PNG encodes off the async runtime).
    tokio::fs::create_dir_all(frames_dir).await?;
    let dir = frames_dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (index, frame) in frames.into_iter().enumerate() {
            let RgbFrame {
                width,
                height,
                pixels,
            } = frame;
            let image = image::RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
                WorkerError::InvalidPayload("video frame buffer size mismatch".to_owned())
            })?;
            let path = dir.join(format!("frame_{index:05}.png"));
            image
                .save_with_format(&path, image::ImageFormat::Png)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;

    // 2. Frames → mp4 (libx264, yuv420p — request dims are multiples of 32, so even).
    let pattern = frames_dir.join("frame_%05d.png");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-framerate".to_owned(),
            fps.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            pattern.to_string_lossy().into_owned(),
            "-c:v".to_owned(),
            "libx264".to_owned(),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            "-r".to_owned(),
            fps.to_string(),
            enc_tmp.to_string_lossy().into_owned(),
        ],
        ctx,
    )
    .await?;

    // 3. Mux the audio track (LTX) as AAC, else the video-only mp4 is the result.
    let finished_tmp = if let Some(audio) = audio {
        write_wav_pcm16(&audio, wav_tmp)?;
        run_ffmpeg(
            vec![
                "ffmpeg".to_owned(),
                "-nostdin".to_owned(),
                "-y".to_owned(),
                "-i".to_owned(),
                enc_tmp.to_string_lossy().into_owned(),
                "-i".to_owned(),
                wav_tmp.to_string_lossy().into_owned(),
                "-c:v".to_owned(),
                "copy".to_owned(),
                "-c:a".to_owned(),
                "aac".to_owned(),
                "-shortest".to_owned(),
                mux_tmp.to_string_lossy().into_owned(),
            ],
            ctx,
        )
        .await?;
        mux_tmp
    } else {
        enc_tmp
    };

    // 4. Publish atomically, then best-effort faststart + poster (mirrors Python).
    tokio::fs::rename(finished_tmp, media_path).await?;
    faststart_mp4(media_path).await;
    write_poster_frame(media_path).await;
    Ok(())
}

/// Peak-normalize the f32 PCM to 16-bit and write a canonical WAV. Silence (peak 0)
/// stays silent rather than dividing by zero.
fn write_wav_pcm16(audio: &AudioTrack, path: &Path) -> WorkerResult<()> {
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    let scale = if peak > 0.0 {
        i16::MAX as f32 / peak
    } else {
        0.0
    };
    let mut pcm = Vec::with_capacity(audio.samples.len() * 2);
    for &sample in &audio.samples {
        let value = (sample * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        pcm.extend_from_slice(&value.to_le_bytes());
    }

    let channels = audio.channels.max(1);
    let bits_per_sample = 16u16;
    let block_align = channels * bits_per_sample / 8;
    let byte_rate = audio.sample_rate * block_align as u32;
    let data_len = pcm.len() as u32;

    let mut buffer = Vec::with_capacity(44 + pcm.len());
    buffer.extend_from_slice(b"RIFF");
    buffer.extend_from_slice(&(36 + data_len).to_le_bytes());
    buffer.extend_from_slice(b"WAVE");
    buffer.extend_from_slice(b"fmt ");
    buffer.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buffer.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buffer.extend_from_slice(&channels.to_le_bytes());
    buffer.extend_from_slice(&audio.sample_rate.to_le_bytes());
    buffer.extend_from_slice(&byte_rate.to_le_bytes());
    buffer.extend_from_slice(&block_align.to_le_bytes());
    buffer.extend_from_slice(&bits_per_sample.to_le_bytes());
    buffer.extend_from_slice(b"data");
    buffer.extend_from_slice(&data_len.to_le_bytes());
    buffer.extend_from_slice(&pcm);
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Best-effort `+faststart` remux (moov atom to the front so WKWebView can start
/// playback without a tail byte-range seek). A missing/failing ffmpeg leaves the
/// original untouched — the API's byte-range support is the load-bearing guarantee.
async fn faststart_mp4(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let remuxed = media_path.with_extension("faststart.mp4");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            remuxed.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if ok {
        let _ = tokio::fs::rename(&remuxed, media_path).await;
    } else {
        let _ = tokio::fs::remove_file(&remuxed).await;
    }
}

/// Best-effort poster extraction to `<name>.poster.jpg` (WKWebView does not paint a
/// `<video>`'s first frame on its own). A missing/failing ffmpeg leaves no poster.
async fn write_poster_frame(media_path: &Path) {
    if !media_path.exists() {
        return;
    }
    let poster = media_path.with_extension("poster.jpg");
    let ok = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-q:v".to_owned(),
            "3".to_owned(),
            poster.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .is_ok();
    if !ok {
        let _ = tokio::fs::remove_file(&poster).await;
    }
}

// ---------------------------------------------------------------------------
// Asset fact + streaming result (mirrors `video_generation_result`).
// ---------------------------------------------------------------------------

/// The MEASURED facts of the clip that was encoded — counted off the `DecodedVideo` itself, never
/// predicted from the request (sc-12371).
///
/// Everything under an asset's `file` block is a claim about bytes on disk: `frameCount`, `fps` and
/// `duration` must describe the mp4 that exists, not the one that was ordered. They used to be
/// predictions — `request.frame_count()` re-ran the temporal lattice and `request.duration` echoed
/// the user's ask — and the predictions were wrong: the arms driving a Wan engine under a non-Wan
/// model id (`bernini`, `scail2_14b`, `external_base_*`) rendered `wan_frame_count(raw)` = 149 while
/// the asset claimed 150 frames at the requested 6.0 s. Nothing failed loudly; a wrong clip length
/// is invisible in the output (epic 1788).
///
/// Making the predictions agree with the engine would only fix today's models. Measuring removes the
/// prediction: there is no second computation left to drift, and it also covers what agreement never
/// could — an engine returning a count nobody predicted (SVD's burst clamps to its own `numFrames`;
/// a `validate()` re-snap or truncated decode would too).
///
/// [`video_asset_fact`] takes one of these BY VALUE for that reason: an asset fact cannot be built
/// without measuring the clip, so the record cannot silently fall back to a prediction. That is a
/// compile error rather than a test we would have to remember to write.
///
/// Mirrors [`run_video_upscale_job`], which has always recorded its real `out_count` and
/// `out_count / out_fps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodedClip {
    /// `decoded.frames.len()` — `encode_inner` writes exactly one PNG per entry, so this IS the
    /// file's frame count.
    frames: usize,
    /// The framerate the file is encoded at. `decoded.fps.max(1)` — the SAME clamp `encode_inner`
    /// applies when it hands `-framerate` to ffmpeg, so the record matches the container.
    fps: u32,
}

impl EncodedClip {
    /// Measure the clip that is about to be written.
    fn measure(decoded: &DecodedVideo) -> Self {
        Self {
            frames: decoded.frames.len(),
            fps: decoded.fps.max(1),
        }
    }

    /// The clip's real running time. Not `request.duration`: at the 6 s x 25 fps default a Wan
    /// engine renders 149 frames, so the file is 5.96 s and an asset claiming 6.0 s is lying about
    /// a file sitting right next to it.
    fn duration_seconds(&self) -> f64 {
        self.frames as f64 / f64::from(self.fps)
    }

    /// Stamp the measured frame count onto a builder's raw settings — THE ONE WRITER of
    /// `frameCount`. No `*_raw_settings` builder may record one (pinned by
    /// `no_raw_settings_builder_records_its_own_frame_count`), so this is the only opinion there is.
    fn record_frame_count(&self, raw_settings: Value) -> Value {
        match raw_settings {
            Value::Object(mut map) => {
                map.insert("frameCount".to_owned(), json!(self.frames));
                Value::Object(map)
            }
            // Every builder returns an object; a non-object carries no keys to correct, so pass it
            // through rather than fabricating a record around it.
            other => other,
        }
    }
}

/// The flat per-asset fact the Rust API turns into an indexed video asset (every key
/// is consumed by the API's video sidecar builder). Mirrors `video_generation_result`.
/// `adapter` is the generating adapter id (`procedural_video` stub / `mlx_wan` real)
/// and `raw_settings` its recorded knobs.
///
/// `clip` is the MEASURED [`EncodedClip`] and is required, not optional: `frameCount`, `fps` and
/// `duration` all land under the API's `asset.file` block — claims about the mp4 on disk — so they
/// are derived from the clip that was encoded rather than predicted from the request (sc-12371).
fn video_asset_fact(
    plan: &VideoPlan,
    seed: i64,
    adapter: &str,
    raw_settings: Value,
    replacement_status: Option<Value>,
    clip: EncodedClip,
) -> Value {
    let request = &plan.request;
    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = if title.is_empty() {
        "Generated video".to_owned()
    } else {
        title.to_owned()
    };
    let timeline_context = request
        .advanced
        .get("timelineContext")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut fact = json!({
        "type": "video",
        "assetId": plan.asset_id,
        "mediaPath": plan.media_rel,
        "mimeType": "video/mp4",
        "width": request.width,
        "height": request.height,
        // REQUESTED, deliberately (sc-12371). These two are the knobs the user PICKED off the
        // model's `limits.durations` / `limits.fps` menus, and `build_video_sidecar_parts` feeds
        // them to `recipe.normalizedSettings` — which is what "re-run this generation" rebuilds the
        // payload from (sc-12324/12345). Replacing them with the measured values would replay a 6 s
        // ask as 5.96 s: off-menu, and sc-12347 now enforces those menus server-side, so it could be
        // refused outright. The MEASURED pair below is what the `file` block uses.
        "duration": request.duration,
        "fps": request.fps,
        // MEASURED off the encoded clip (sc-12371) — the `asset.file` block's honest running time
        // and cadence. `file.duration` used to echo `request.duration`, so a 6 s ask on a Wan engine
        // produced a 149-frame / 5.96 s file that the asset described as 6.0 s: exactly the "claims
        // a duration the file does not have" this story was filed for. Kept as separate keys rather
        // than overwriting the two above because a knob and a measurement are different facts that
        // only happened to share a name — see `recipe_fields.js`, which already splits "the dims the
        // app RAN at" from "the dims the user PICKED" for the same reason.
        "encodedFrameCount": clip.frames,
        "encodedDuration": clip.duration_seconds(),
        "encodedFps": clip.fps,
        "quality": request.quality,
        "family": plan.family,
        "seed": seed,
        "displayName": display_name,
        "createdAt": plan.created_at,
        "mode": request.mode,
        "model": request.model,
        "adapter": adapter,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "rawAdapterSettings": clip.record_frame_count(raw_settings),
        "sourceAssetId": request.source_asset_id,
        "lastFrameAssetId": request.last_frame_asset_id,
        "sourceClipAssetId": request.source_clip_asset_id,
        "bridgeRightClipAssetId": request.bridge_right_clip_asset_id,
        // The multi-source ids and the fit are top-level payload fields, NOT `advanced` — so the
        // `advanced.clone()` every real `*_raw_settings` builder starts with does not carry them.
        // They must be written here or the recipe cannot reproduce the modes that use them
        // (mv2v / reference_to_video / reference_video_to_video / ads2v / animate_character, and
        // the fit for image_to_video / first_last_frame). sc-12345, prereq for sc-12324 replay.
        "fitMode": request.fit_mode,
        "sourceClipAssetIds": request.source_clip_asset_ids,
        "referenceAssetIds": request.reference_asset_ids,
        "referenceClipAssetId": request.reference_clip_asset_id,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "personTrackId": request.person_track_id,
        "replacementMode": request.replacement_mode,
        "timelineContext": timeline_context,
    });
    // replace_person reports its honest mask/track provenance (mirrors the torch
    // `video_generation_result` `replacementStatus` fold; sc-3521).
    if let (Some(status), Some(object)) = (replacement_status, fact.as_object_mut()) {
        object.insert("replacementStatus".to_owned(), status);
    }
    fact
}

/// Raw-settings recorded on a procedural stub asset — the dispatched KNOBS (`duration` here is what
/// was ASKED for, which is what `rawAdapterSettings` is for). Like every other builder it records no
/// `frameCount`: [`EncodedClip::record_frame_count`] stamps the clip's real length (sc-12371).
fn stub_raw_settings(request: &VideoRequest) -> Value {
    json!({
        "model": request.model,
        "fps": request.fps,
        "duration": request.duration,
        "quality": request.quality,
        "stub": true,
    })
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact. A video job always reports exactly one asset (`expectedCount` 1).
fn streaming_result(plan: &VideoPlan, fact: &Value, adapter: &str) -> JsonObject {
    let request = &plan.request;
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": 1,
        "adapter": adapter,
        "model": request.model,
        "generationSet": {
            "id": plan.genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": plan.created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Progress payload with the worker's real backend label (mirrors `image_progress`).
fn video_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

// ---------------------------------------------------------------------------
// SeedVR2 video upscale (epic 4811, sc-4816): the net-new `video_upscale` job —
// SceneWorks' first video upscaler. Decode the source clip -> native-MLX SeedVR2
// one-step super-resolution (temporal chunking + overlap is internal to the engine)
// -> encode + source-audio passthrough. macOS-only (no torch path). Reuses the shared
// encode pipeline (`encode_media`) + the streaming engine driver (`generate_video`).
// ---------------------------------------------------------------------------

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::seedvr2::video as seedvr2_video;
/// The SeedVR2 provider's pure temporal-chunk planning/blend module (`video::plan_chunks`,
/// `video::assemble_overlap`, `DEFAULT_OVERLAP`, `Chunk`), reused ONE LEVEL UP for worker-window
/// streaming (sc-9595). Both provider crates expose an identical `video` module over `gen_core::Image`
/// (byte-identical seam math), so the worker-window cross-fade is the engine's own — the worker never
/// reimplements the blend. Aliased per platform: MLX on Mac, candle on the Windows/CUDA lane.
#[cfg(target_os = "macos")]
use runtime_macos::providers::seedvr2::video as seedvr2_video;

/// HF repo hosting the raw SeedVR2 checkpoint (`numz/SeedVR2_comfyUI`); the engine converts it
/// in-memory at load (no Python). Override the staged dir with `SCENEWORKS_SEEDVR2_DIR`.
// SeedVR2 video upscale runs on Mac (native MLX) AND the Windows/CUDA candle lane (sc-5928); these
// constants/helpers are backend-neutral (gen_core + ffmpeg + the shared streaming driver).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_REPO: &str = "numz/SeedVR2_comfyUI";
/// Pinned SeedVR2 checkpoint revision (sc-8879 / sc-9879). `numz/SeedVR2_comfyUI` is a
/// third-party mirror with a fixed (non-overridable) repo here — fetching the mutable `main`
/// branch would let an upstream re-push silently swap the 3B fp16 DiT + VAE weights we load.
/// Pin the exact commit so downloads are reproducible; HF's tree API still reports each file's
/// `lfs.oid`, which `ensure_hf_cached_file` verifies the content against. MUST equal the
/// image-upscale lane's `upscale_jobs::SEEDVR2_REVISION` (same repo + files) — the
/// `seedvr2_video_revision_matches_image_lane` agreement test locks them together so they
/// can't drift.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_REVISION: &str = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_VAE_FILE: &str = "ema_vae_fp16.safetensors";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_DIT_3B_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
/// The engine registry id wired for video upscale (3B; 7B = sc-5197 / sc-5927).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ENGINE_ID: &str = "seedvr2_3b";
/// Adapter id recorded on the result asset for provenance (mirrors the other `mlx_*` video adapters;
/// SeedVR2 itself takes no LoRA — this is metadata only).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_ADAPTER: &str = "mlx_seedvr2";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_CANCEL_MESSAGE: &str = "Video upscale canceled by user.";

/// Snap a dimension to the SeedVR2 VAE/patch stride (a multiple of 16, the engine's hard
/// requirement), rounding to nearest and clamping to the engine's `[16, 4096]` size range.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn snap_seedvr2_dim(value: u32) -> u32 {
    let rounded = value.saturating_add(8) / 16 * 16;
    rounded.clamp(16, 4096)
}

// ---------------------------------------------------------------------------
// Disk-space guard for the streaming SeedVR2 output (sc-9646, sc-9595 follow-up)
// ---------------------------------------------------------------------------
// sc-9595 removed the sc-8829 host-RAM cap by streaming the upscale in temporal windows, so peak host
// RAM is now bounded to ~one window regardless of clip length. But the constraint MOVED from RAM to
// DISK: the full upscaled PNG sequence is written to a worker scratch dir before the final encode, so
// a multi-minute / 4K clip can now write many GB with NO guard (the RAM cap previously bounded the
// whole operation). This mirrors the removed `check_seedvr2_host_ram`: a generous, machine-derived,
// fail-loud-before-the-window-loop preflight that estimates the output PNG footprint and rejects a
// clip that would fill the scratch volume, so the disk is not silently exhausted mid-run.

/// Fraction of the scratch volume's CURRENTLY-AVAILABLE space the streamed output PNG sequence is
/// allowed to occupy. Deliberately generous — the estimate itself uses raw RGB8 per frame (a real
/// upper bound on PNG-compressed output), and we still leave headroom for the source frames already on
/// disk, the eventual encoded MP4, and everything else sharing the volume.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_DISK_OUTPUT_FRACTION: f64 = 0.8;

/// Estimated peak on-disk bytes for the streamed output: `frame_count` PNG frames at
/// `out_w × out_h`, sized as raw RGB8 (`w·h·3`) per frame. PNG compression only ever makes the real
/// footprint SMALLER, so this is a safe upper bound (matching the generous shape of the removed
/// `seedvr2_estimated_host_bytes`). Pure so the estimate is unit-testable without a filesystem.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn seedvr2_estimated_output_bytes(frame_count: u64, out_w: u64, out_h: u64) -> u64 {
    let per_frame = out_w.saturating_mul(out_h).saturating_mul(3);
    frame_count.saturating_mul(per_frame)
}

/// Bytes currently AVAILABLE on the volume backing `path`, best-effort and portable with NO new crate
/// dependency (mirrors the removed `total_physical_ram_bytes`, which likewise shelled out): macOS +
/// Linux run POSIX `df -k -P <path>` and read the 4th column (available 1K blocks); Windows (candle
/// lane) runs `fsutil volume diskfree <path>`. Returns `None` if the probe fails; the caller then
/// skips the guard rather than falsely rejecting a job.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn available_disk_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        // `df -k -P` forces POSIX one-line-per-filesystem output in 1024-byte blocks, so the columns
        // are stable regardless of locale/long device names:
        //   Filesystem 1024-blocks Used Available Capacity Mounted on
        let out = std::process::Command::new("df")
            .args(["-k", "-P"])
            .arg(path)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        // The data row is the 2nd line (after the header); `-P` guarantees it is a single line.
        let row = text.lines().nth(1)?;
        let available_kib = row.split_whitespace().nth(3)?.parse::<u64>().ok()?;
        Some(available_kib.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    {
        // Windows (candle lane): `fsutil volume diskfree <path>` prints the free bytes; the
        // "Total # of avail free bytes" line is the per-user available figure. Dependency-free.
        let out = std::process::Command::new("fsutil")
            .args(["volume", "diskfree"])
            .arg(path)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("avail") && lower.contains("free bytes") {
                // Grab the last whitespace-separated token, stripping thousands separators.
                if let Some(token) = line.split_whitespace().next_back() {
                    let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
                    if let Ok(bytes) = digits.parse::<u64>() {
                        return Some(bytes);
                    }
                }
            }
        }
        None
    }
}

/// Fail loud (before the window loop / before any GPU work) when the estimated streamed-output PNG
/// footprint would exceed the generous fraction of the scratch volume's currently-available space.
/// `Ok(())` when it fits, when the frame count / dimensions are unknown (0), or when the free-space
/// probe is unavailable (we do not falsely reject a job on a probe failure). The error names the
/// estimate AND the available space so the user knows exactly what to trim. Mirrors the shape of the
/// removed `check_seedvr2_host_ram` (sc-9646).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn check_seedvr2_output_disk(
    scratch_dir: &Path,
    frame_count: u64,
    out_w: u64,
    out_h: u64,
) -> WorkerResult<()> {
    if frame_count == 0 || out_w == 0 || out_h == 0 {
        return Ok(());
    }
    let Some(available) = available_disk_bytes(scratch_dir) else {
        // Probe failed (unknown platform / error): skip the guard rather than block a valid job.
        return Ok(());
    };
    let needed = seedvr2_estimated_output_bytes(frame_count, out_w, out_h);
    let budget = ((available as f64) * SEEDVR2_DISK_OUTPUT_FRACTION) as u64;
    if needed > budget {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        // Largest frame count that fits the budget, so the message is actionable.
        let per_frame = seedvr2_estimated_output_bytes(1, out_w, out_h).max(1);
        let max_frames = budget / per_frame;
        return Err(WorkerError::InvalidPayload(format!(
            "Not enough disk space to upscale this clip: {frame_count} output frames at \
             {out_w}×{out_h} would write ~{needed:.1} GB of PNG frames to the scratch volume, over \
             the ~{budget:.1} GB usable of ~{available:.1} GB free. Trim the clip to about \
             {max_frames} frames (or fewer), lower the target resolution, or free up disk space.",
            needed = needed as f64 / GIB,
            budget = budget as f64 / GIB,
            available = available as f64 / GIB,
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming worker-window chunking for the SeedVR2 upscale (sc-9595, removes the sc-8829 host-RAM cap)
// ---------------------------------------------------------------------------
// sc-8829 (F-027) bounded host RGB8 RAM with a machine-derived frame cap that FAILED LOUD before decode:
// `decode_seedvr2_source_frames` materialized EVERY source frame into a `Vec<Image>` up front, and the
// up-to-4× output was likewise held whole before encode (the engine's whole-clip API returns the entire
// `Vec<Image>`), so a few-minute 1080p clip meant tens of GB of RGB8 → OOM. The cap rejected such clips
// — a capability narrowing.
//
// This story removes the cap by streaming the upscale in temporal WORKER WINDOWS: we plan windows over
// the real frame count with the engine's own `video::plan_chunks` (a valid chunk length + the
// `DEFAULT_OVERLAP=4` cross-fade), decode + upscale ONE window at a time through the shared
// `generate_video` funnel, and cross-fade across worker-window boundaries with the engine's own
// `video::assemble_overlap` (fed a local 2-window plan) so the seam handling is byte-identical to the
// engine's internal chunking. Finalized frames stream straight to a numbered PNG sequence on disk and
// are encoded once at the end (same ffmpeg args as the old whole-clip path), so peak host RAM is bounded
// to ~one worker window's frames + a ≤4-frame overlap tail regardless of clip length.
//
// Seam identity: `plan_chunks`/`assemble_overlap` are the SAME pure functions the engine uses
// internally, and each engine chunk's output is a deterministic function of its source pixel window +
// seed (`pipeline::preprocess_chunk`). When a worker window is processed by the engine as a single
// internal chunk (the common case — `SEEDVR2_WORKER_CHUNK_FRAMES` fits the engine's budget-sized chunk),
// the streamed output is bit-identical to the whole-clip run. If the engine sub-chunks a worker window
// under tight GPU budget, the cross-fade still closes every seam (identical blend math) but the
// upscaled pixels near an internal boundary can differ slightly from a whole-clip run's internal
// boundary — a real-weights fidelity nuance that only a GPU golden run can measure (see the PR notes).

/// The worker-level temporal window size (pixel frames). A valid engine chunk length (mult of 4, ≥8),
/// chosen at the engine's `MAX_CHUNK_FRAMES` ceiling so that on any machine whose budget-sized chunk is
/// ≥ this, the engine processes a whole worker window as ONE internal chunk (`plan_chunks(64,64,4)` = a
/// single chunk) → the streamed output is bit-identical to a whole-clip run. Larger windows mean fewer
/// worker-window seams and a closer match to whole-clip chunking, traded against a larger per-window
/// host footprint (≈ `64 · out_w · out_h · 3` bytes of RGB8, ~1.6 GB at 4×-of-1080p — bounded).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SEEDVR2_WORKER_CHUNK_FRAMES: i32 = 64;

/// Streaming cross-window assembler (sc-9595): feeds each upscaled worker window into the engine's own
/// `video::assemble_overlap` and emits finalized frames in order, holding only a ≤`ov`-frame tail plus
/// the current window in memory. The cross-fade at each worker-window boundary is byte-identical to the
/// engine's internal chunk assembly — proven equivalent to a single whole-clip `assemble_overlap` over
/// synthetic frames (see `seedvr2_stream_matches_whole_clip_assembly`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct Seedvr2StreamAssembler {
    /// Cross-fade overlap (frames) between adjacent worker windows — the engine's `DEFAULT_OVERLAP`.
    ov: i32,
    /// Total real output frame count (== source frame count); trailing chunk padding past this is dropped.
    n: i32,
    /// The retained, blended-but-not-yet-final tail: the assembled frames from `retained_start` onward
    /// that may still be cross-faded by the NEXT window. Its absolute start is `retained_start`.
    retained: Vec<Image>,
    /// Absolute frame index of `retained[0]`.
    retained_start: i32,
    /// Whether `push_window` has been called yet (the first window seeds `retained` with no blend).
    seeded: bool,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Seedvr2StreamAssembler {
    fn new(n: i32, ov: i32) -> Self {
        Self {
            ov,
            n,
            retained: Vec::new(),
            retained_start: 0,
            seeded: false,
        }
    }

    /// Feed the upscaled frames for the worker window at absolute `start`. Returns the newly finalized
    /// frames (in order) that no later window can touch; the caller streams them to disk immediately.
    /// Uses the engine's `video::assemble_overlap` over a local `[retained, window]` 2-window plan so
    /// the blend is the engine's own.
    fn push_window(&mut self, start: i32, mut frames: Vec<Image>) -> Vec<Image> {
        // Clip trailing chunk padding past the real frame count.
        let visible = (self.n - start).clamp(0, frames.len() as i32);
        frames.truncate(visible.max(0) as usize);
        if !self.seeded {
            self.seeded = true;
            self.retained_start = start;
            self.retained = frames;
        } else {
            let off = start - self.retained_start;
            let local_plan = [
                seedvr2_video::Chunk {
                    start: 0,
                    len: self.retained.len() as i32,
                },
                seedvr2_video::Chunk {
                    start: off,
                    len: frames.len() as i32,
                },
            ];
            let n_local = off + frames.len() as i32;
            let inputs = [std::mem::take(&mut self.retained), frames];
            // The engine's own cross-fade closes the worker-window seam (identical to its internal one).
            self.retained = seedvr2_video::assemble_overlap(&local_plan, &inputs, n_local, self.ov);
        }
        // Finalize everything except the last `ov` frames (the next window may still blend them).
        let keep = self.ov.max(0) as usize;
        self.drain_prefix(keep)
    }

    /// Flush the remaining tail once every window has been pushed.
    fn finish(&mut self) -> Vec<Image> {
        self.drain_prefix(0)
    }

    /// Emit finalized frames from the front of `retained`, keeping `keep` frames retained (and never
    /// emitting past the real frame count `n`).
    fn drain_prefix(&mut self, keep: usize) -> Vec<Image> {
        // How many front frames are finalized this call: everything past `keep`, but never emitting
        // past the real frame count `n` (`n - retained_start` remaining slots). `Vec::drain` shifts
        // the tail once, so this is O(n) rather than the O(n²) of a `remove(0)`-per-frame loop.
        let by_keep = self.retained.len().saturating_sub(keep);
        let by_count = (self.n - self.retained_start).max(0) as usize;
        let boundary = by_keep.min(by_count);
        let out: Vec<Image> = self.retained.drain(..boundary).collect();
        self.retained_start += boundary as i32;
        out
    }
}

/// Resolve a project-relative asset path safely under `project_path` (reject `..` / absolute
/// components — same guard as `upscale_jobs::resolve_source`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn safe_join(project_path: &Path, rel: &str) -> Option<PathBuf> {
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => return None,
        }
    }
    Some(path)
}

/// Provision the SeedVR2 checkpoint dir: an env-pinned dir (pre-staged for local validation) wins,
/// else the app cache (download the VAE + 3B DiT from `numz/SeedVR2_comfyUI` on first use). Returns
/// the dir to hand the engine as `WeightsSource::Dir`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_seedvr2_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("seedvr2-mlx"));
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Video upscale canceled while fetching SeedVR2 weights.",
        fresh_download: false,
    };
    for file in [SEEDVR2_VAE_FILE, SEEDVR2_DIT_3B_FILE] {
        crate::downloads::ensure_hf_cached_file(
            &context,
            SEEDVR2_REPO,
            SEEDVR2_REVISION,
            file,
            &dir.join(file),
        )
        .await?;
    }
    Ok(dir)
}

/// Decode every frame of `source` to a numbered PNG sequence ON DISK (native resolution — the engine
/// bicubic-upscales internally to the target) and return the ordered PNG paths, WITHOUT loading any
/// pixels into RAM (sc-9595). Uses the bundled ffmpeg (`run_ffmpeg`); `-fps_mode passthrough` keeps the
/// exact source frame count. The caller loads each temporal WINDOW of these paths on demand via
/// [`load_seedvr2_window`], so host RGB8 RAM is bounded to one window instead of the whole clip. The
/// returned paths live under a job-scoped temp dir the caller is responsible for removing.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn decode_seedvr2_source_to_disk(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    source: &Path,
    frames_dir: &Path,
) -> WorkerResult<Vec<PathBuf>> {
    let _ = tokio::fs::remove_dir_all(frames_dir).await;
    tokio::fs::create_dir_all(frames_dir).await?;
    let ctx = FfmpegContext::new(api, settings, job_id, SEEDVR2_CANCEL_MESSAGE);
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-fps_mode".to_owned(),
            "passthrough".to_owned(),
            frames_dir
                .join("in_%05d.png")
                .to_string_lossy()
                .into_owned(),
        ],
        Some(ctx),
    )
    .await?;
    let dir = frames_dir.to_path_buf();
    let paths = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<PathBuf>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
            .collect();
        paths.sort();
        Ok(paths)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;
    if paths.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "source video produced no frames to upscale".to_owned(),
        ));
    }
    Ok(paths)
}

/// Load the temporal window `paths[start .. start+len]` (clamped to the sequence end) into engine
/// [`Image`]s on demand (sc-9595). Real frames only — the engine's `preprocess_chunk` pads a partial
/// trailing window with last-frame repeats internally, matching the whole-clip path. Runs the blocking
/// PNG decode off the async runtime.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn load_seedvr2_window(
    paths: &[PathBuf],
    start: usize,
    len: usize,
) -> WorkerResult<Vec<Image>> {
    let end = start.saturating_add(len).min(paths.len());
    let window: Vec<PathBuf> = paths.get(start..end).unwrap_or(&[]).to_vec();
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut frames = Vec::with_capacity(window.len());
        for path in window {
            let image = crate::image_decode::decode_image_any(&path)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
                .to_rgb8();
            frames.push(rgb_image_to_engine(image));
        }
        Ok(frames)
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
}

/// Append RGB8 frames to a numbered PNG sequence on disk, starting at `next_index`, off the async
/// runtime (sc-9595). Returns the next free index. The shared frames dir is later encoded once by
/// [`encode_seedvr2_stream`] with the exact ffmpeg args the whole-clip `encode_media` used, so the
/// output is byte-identical while peak host RAM stays bounded to one worker window.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn append_seedvr2_frames(
    frames_dir: &Path,
    next_index: usize,
    frames: Vec<Image>,
) -> WorkerResult<usize> {
    if frames.is_empty() {
        return Ok(next_index);
    }
    let dir = frames_dir.to_path_buf();
    let count = frames.len();
    tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        for (offset, frame) in frames.into_iter().enumerate() {
            let index = next_index + offset;
            let img = image::RgbImage::from_raw(frame.width, frame.height, frame.pixels)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("video frame buffer size mismatch".to_owned())
                })?;
            let path = dir.join(format!("frame_{index:05}.png"));
            img.save_with_format(&path, image::ImageFormat::Png)
                .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        }
        Ok(())
    })
    .await
    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??;
    Ok(next_index + count)
}

/// Encode a pre-written numbered PNG sequence (`frame_%05d.png`, `frame_count` frames from index 0) to
/// the final mp4 (silent) + faststart + poster (sc-9595). The ffmpeg encode args are byte-identical to
/// the whole-clip `encode_media` path (`libx264` / `yuv420p` / `-framerate fps` / `-r fps`), so the
/// streamed output matches the old path frame-for-frame. Audio muxing stays the caller's source
/// passthrough step (SeedVR2 emits no audio).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn encode_seedvr2_stream(
    media_path: &Path,
    frames_dir: &Path,
    frame_count: usize,
    fps: u32,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if frame_count == 0 {
        return Err(WorkerError::InvalidPayload(
            "video generation produced no frames".to_owned(),
        ));
    }
    let fps = fps.max(1);
    let enc_tmp = media_path.with_extension("enc.mp4");
    let pattern = frames_dir.join("frame_%05d.png");
    let result = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-framerate".to_owned(),
            fps.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            pattern.to_string_lossy().into_owned(),
            "-c:v".to_owned(),
            "libx264".to_owned(),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            "-r".to_owned(),
            fps.to_string(),
            enc_tmp.to_string_lossy().into_owned(),
        ],
        ctx,
    )
    .await;
    match result {
        Ok(()) => {
            // Publish atomically, then best-effort faststart + poster (mirrors `encode_media`).
            tokio::fs::rename(&enc_tmp, media_path).await?;
            faststart_mp4(media_path).await;
            write_poster_frame(media_path).await;
            Ok(())
        }
        Err(error) => {
            let _ = tokio::fs::remove_file(&enc_tmp).await;
            let _ = tokio::fs::remove_file(media_path).await;
            Err(error)
        }
    }
}

/// Result of the streamed SeedVR2 upscale (sc-9595): the metadata the caller needs to build the asset
/// fact + encode the pre-written PNG sequence. The upscaled frames themselves are already on disk
/// (`out_frames_dir`), never held whole in RAM.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct Seedvr2Stream {
    /// Real output frame count (== source frame count).
    frame_count: usize,
    fps: u32,
    out_w: u32,
    out_h: u32,
    src_w: u32,
    src_h: u32,
    seed: u64,
}

/// RAII guard for a worker-owned scratch directory (sc-9595). The streamed 4× PNG sequence can be many
/// GB, and it now outlives the stream call while the caller runs an `update_job` progress POST + an
/// `ffmpeg` encode — a span where any `?` (a transient POST failure, a 409 stale-sweep reclaim, an
/// encode error, or a between-step cancel) would otherwise leak the whole dir on disk. `Drop` removes
/// it on EVERY exit path (success, encode/create_dir_all/update_job failure, cancel, panic) so cleanup
/// can never be skipped. Call [`ScratchDir::disarm`] after the caller has already removed the dir to
/// avoid a redundant (harmless) second removal. `Drop` must use the sync `std::fs` API.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct ScratchDir {
    path: std::path::PathBuf,
    armed: bool,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ScratchDir {
    /// Guard `path`. Does not create it — the caller populates it; the guard only guarantees removal.
    fn new(path: std::path::PathBuf) -> Self {
        Self { path, armed: true }
    }

    /// Stop guarding: the caller has already removed the dir (e.g. right after a successful encode, to
    /// free disk before the mux step). A no-op `Drop` follows.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Drop for ScratchDir {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: a missing dir yields a benign NotFound we intentionally ignore.
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Stream the SeedVR2 upscale in temporal worker windows (sc-9595). Decodes the source to disk, plans
/// worker windows over the real frame count with the engine's `video::plan_chunks`
/// (`SEEDVR2_WORKER_CHUNK_FRAMES` + `DEFAULT_OVERLAP`), upscales ONE window at a time through the shared
/// `generate_video` funnel, cross-fades across worker-window boundaries with the engine's own
/// `video::assemble_overlap`, and appends each finalized frame to a numbered PNG sequence on disk. Peak
/// host RGB8 RAM is bounded to ~one worker window + a ≤`ov`-frame tail. Each per-window `generate_video`
/// call is independently heartbeat-covered + cancellable (the funnel's watchdog), and cancel is polled
/// between windows too — no long silent blocking span.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn run_seedvr2_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    req: &sceneworks_core::contracts::VideoUpscaleRequest,
    source_path: &Path,
    src_frames_dir: &Path,
    out_frames_dir: &Path,
    factor: u32,
    source_fps: u32,
    weights_dir: PathBuf,
) -> WorkerResult<Seedvr2Stream> {
    let seed = req.seed.unwrap_or(0);
    // Decode the whole source to numbered PNGs on disk (RAM-free), then peek the first frame's dims.
    let paths =
        decode_seedvr2_source_to_disk(api, settings, &job.id, source_path, src_frames_dir).await?;
    let n = paths.len();
    let first = load_seedvr2_window(&paths, 0, 1).await?;
    let src_w = first[0].width;
    let src_h = first[0].height;
    drop(first);

    // Resolve + snap the target output resolution (identical to the whole-clip path).
    let (target_w, target_h) = match (req.target_width, req.target_height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => (src_w.saturating_mul(factor), src_h.saturating_mul(factor)),
    };
    let target_w = snap_seedvr2_dim(target_w);
    let target_h = snap_seedvr2_dim(target_h);

    tokio::fs::create_dir_all(out_frames_dir).await?;

    // Disk-space preflight (sc-9646): sc-9595 streams the whole upscaled PNG sequence to this scratch
    // dir before the final encode, so a long / high-res clip can write many GB with no bound (the
    // removed sc-8829 host-RAM cap previously bounded the whole operation). Fail loud HERE — before the
    // first window's decode + GPU work — when the estimated output footprint would exceed a generous
    // fraction of the scratch volume's free space, so we don't fill the disk mid-run. Probed on a
    // blocking thread (it shells out) but cheap and one-shot.
    {
        let guard_dir = out_frames_dir.to_path_buf();
        let (frames, gw, gh) = (n as u64, u64::from(target_w), u64::from(target_h));
        tokio::task::spawn_blocking(move || check_seedvr2_output_disk(&guard_dir, frames, gw, gh))
            .await
            .map_err(|error| task_join_error("seedvr2 disk preflight", error))??;
    }

    // Plan the worker windows over the REAL frame count with the engine's own planner: a valid chunk
    // length + `DEFAULT_OVERLAP` cross-fade, so the worker-window seam handling matches the engine's
    // internal chunking exactly.
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    let plan = seedvr2_video::plan_chunks(n as i32, SEEDVR2_WORKER_CHUNK_FRAMES, ov);
    let mut assembler = Seedvr2StreamAssembler::new(n as i32, ov);
    let mut next_index = 0usize;
    let mut out_w = target_w;
    let mut out_h = target_h;
    let window_total = plan.len().max(1);

    for (window_idx, chunk) in plan.iter().enumerate() {
        check_cancel(api, &job.id, SEEDVR2_CANCEL_MESSAGE).await?;
        // Real frames only for this window; the engine's preprocess_chunk pads a partial tail internally.
        let start = chunk.start.max(0) as usize;
        if start >= n {
            break;
        }
        let len = chunk.len.max(0) as usize;
        let window_frames = load_seedvr2_window(&paths, start, len).await?;
        if window_frames.is_empty() {
            continue;
        }
        let window_len = window_frames.len() as u32;

        // Per-window progress spanning the Generating band (0.18 → 0.55) so the bar advances per window
        // even though each window's own denoise progress is reported inside `generate_video`.
        let frac = 0.18 + 0.37 * (window_idx as f64 / window_total as f64);
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                frac,
                &format!("Upscaling window {}/{window_total}.", window_idx + 1),
                None,
                backend,
            ),
        )
        .await?;

        // Upscale this window through the shared streaming driver (generator cache + stall watchdog +
        // cancel + per-step progress). Same seed for every window → deterministic per-chunk blend.
        let input = VideoGenInput {
            engine_id: SEEDVR2_ENGINE_ID,
            model_dir: weights_dir.clone(),
            conditioning: vec![Conditioning::VideoClip {
                frames: window_frames,
                frame_idx: 0,
                strength: 1.0,
            }],
            width: target_w,
            height: target_h,
            frames: window_len,
            fps: source_fps,
            seed,
            softness: Some(req.softness),
            ..Default::default()
        };
        // SeedVR2 upscale is one-step with no per-generation sampler/scheduler knobs, so the
        // advanced block generate_video reads for those is empty here (F-118).
        let decoded =
            generate_video(api, settings, job, backend, &JsonObject::new(), input).await?;
        if let Some(frame) = decoded.frames.first() {
            out_w = frame.width;
            out_h = frame.height;
        }
        let upscaled: Vec<Image> = decoded
            .frames
            .into_iter()
            .map(|frame| Image {
                width: frame.width,
                height: frame.height,
                pixels: frame.pixels,
            })
            .collect();

        // Cross-fade this window against the retained tail (the engine's own blend) and stream the
        // now-finalized frames to disk.
        let finalized = assembler.push_window(chunk.start, upscaled);
        next_index = append_seedvr2_frames(out_frames_dir, next_index, finalized).await?;
    }

    // Flush the final tail.
    let tail = assembler.finish();
    next_index = append_seedvr2_frames(out_frames_dir, next_index, tail).await?;

    if next_index == 0 {
        return Err(WorkerError::InvalidPayload(
            "source video produced no frames to upscale".to_owned(),
        ));
    }

    Ok(Seedvr2Stream {
        frame_count: next_index,
        fps: source_fps,
        out_w,
        out_h,
        src_w,
        src_h,
        seed,
    })
}

/// Validate a requested video-upscale factor. SeedVR2 supports only 2x and 4x, so any other value
/// (3x, 8x, 1x, 0) is rejected with a clear error rather than silently coerced (F-118). Returns the
/// factor widened to `u32` for the downstream dimension math.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_video_upscale_factor(factor: u8) -> WorkerResult<u32> {
    match factor {
        2 | 4 => Ok(u32::from(factor)),
        other => Err(WorkerError::InvalidPayload(format!(
            "Video upscale supports only factor 2 or 4 (got {other})."
        ))),
    }
}

/// Dispatch handler for `JobType::VideoUpscale`: decode the source clip, run the SeedVR2 upscaler
/// (native MLX on Mac / candle CUDA on Windows, sc-5928), re-encode, pass the source audio through,
/// and stream a single upscaled video asset.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_video_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(Value::Object(job.payload.clone())).map_err(|error| {
            WorkerError::InvalidPayload(format!("Invalid video_upscale payload: {error}"))
        })?;
    if req.source_asset_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Video upscale jobs require a source video asset.".to_owned(),
        ));
    }
    let engine = req.engine.trim().to_ascii_lowercase();
    if !matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b") {
        return Err(WorkerError::InvalidPayload(format!(
            "This video upscaler supports only engine=seedvr2 (got {engine})."
        )));
    }
    let project_id = req
        .project_id
        .clone()
        .or_else(|| job.project_id.clone())
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload("Missing payload.projectId".to_owned()))?;
    // Reject an unsupported factor early rather than silently coercing it to 2 (F-118).
    let factor = resolve_video_upscale_factor(req.factor)?;
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Loading source video.",
            None,
            backend,
        ),
    )
    .await?;

    // Resolve the source video asset (on-disk path + fps + display name) from its sidecar.
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&project_id)?;
    let project_path = PathBuf::from(project.path);
    let asset = store
        .get_asset(&project_id, &req.source_asset_id)
        .map_err(|_| WorkerError::InvalidPayload("Source video asset not found.".to_owned()))?;
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset has no media file.".to_owned()))?;
    let rel = file.get("path").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload("Source asset media path missing.".to_owned())
    })?;
    let source_path = safe_join(&project_path, rel)
        .filter(|path| path.exists())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Source media file is unavailable.".to_owned())
        })?;
    let source_fps = file
        .get("fps")
        .and_then(Value::as_f64)
        .map(|fps| fps.round() as u32)
        .filter(|fps| *fps > 0)
        .unwrap_or(24);
    let source_display = asset
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);

    // sc-9595: no host-RAM cap. The upscale streams in temporal worker windows (decode → upscale →
    // append-encode one window at a time), so peak host RGB8 RAM is bounded to ~one window regardless
    // of clip length — the sc-8829 whole-clip frame cap that rejected long/high-res clips is gone.

    check_cancel(api, &job.id, SEEDVR2_CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Fetching SeedVR2 weights.",
            None,
            backend,
        ),
    )
    .await?;
    let weights_dir = ensure_seedvr2_weights(api, settings, job).await?;

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.18,
            "Decoding source frames.",
            None,
            backend,
        ),
    )
    .await?;
    // Decode the whole source to a numbered PNG sequence ON DISK (disk-bounded, no RAM), then load each
    // temporal window on demand (sc-9595). `-fps_mode passthrough` preserves the exact source frame
    // count / order; the output frame count/order/fps therefore stay identical to the whole-clip path.
    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()`. Mirrors the person-track work dir sanitization.
    let safe_job = safe_download_dir(&job.id);
    let src_frames_dir = std::env::temp_dir().join(format!("sceneworks_seedvr2_src_{safe_job}"));
    let out_frames_dir = std::env::temp_dir().join(format!("sceneworks_seedvr2_out_{safe_job}"));
    let _ = tokio::fs::remove_dir_all(&out_frames_dir).await;
    // RAII-guard the output PNG scratch so it is removed on EVERY exit after the stream: not just the
    // stream error arm, but the create_dir_all / update_job progress POST / encode span below, any of
    // which can `?`-return (transient POST failure, 409 stale-sweep reclaim, encode error, cancel).
    // Without this the full multi-GB 4× sequence would leak on those paths (sc-9595 review).
    let mut out_scratch = ScratchDir::new(out_frames_dir.clone());
    let stream_result = run_seedvr2_stream(
        api,
        settings,
        job,
        backend,
        &req,
        &source_path,
        &src_frames_dir,
        &out_frames_dir,
        factor,
        source_fps,
        weights_dir,
    )
    .await;
    // Always drop the source PNG scratch (it's disk-only; the output dir is owned by `out_scratch`).
    let _ = tokio::fs::remove_dir_all(&src_frames_dir).await;
    // On stream failure, `out_scratch`'s Drop removes the output dir as the function returns.
    let stream = stream_result?;
    let Seedvr2Stream {
        frame_count: out_count,
        fps: out_fps,
        out_w,
        out_h,
        src_w,
        src_h,
        seed,
    } = stream;
    let duration = out_count as f64 / out_fps.max(1) as f64;

    // Plan the output asset path (nested under the per-generation id, like VideoPlan).
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let media_rel = format!(
        "assets/videos/{genset_id}/{}_seedvr2_upscale.mp4",
        &created_at[..10]
    );
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.6,
            "Encoding upscaled video.",
            None,
            backend,
        ),
    )
    .await?;
    // Encode the streamed PNG sequence to a (silent) mp4 + poster + faststart (byte-identical ffmpeg
    // args to the old whole-clip `encode_media` path), then drop the output scratch dir.
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    let encode_result =
        encode_seedvr2_stream(&media_path, &out_frames_dir, out_count, out_fps, Some(ctx)).await;
    // Free the multi-GB PNG scratch as soon as the encode returns (before the mux step), on BOTH the
    // ok and err arms; then disarm the guard so its Drop doesn't redundantly re-remove. `encode_result`
    // is propagated AFTER cleanup — an encode error still leaves no scratch behind (and if this early
    // removal is itself skipped by an unwind, the still-armed guard's Drop is the backstop).
    let _ = tokio::fs::remove_dir_all(&out_frames_dir).await;
    out_scratch.disarm();
    encode_result?;

    // Source-audio passthrough: remux the source's audio onto the upscaled video. `-map 1:a:0?`
    // makes audio optional, so a source with no audio yields a clean video-only file (no error);
    // `-c:v copy` keeps the upscaled video stream untouched, `+faststart` is preserved.
    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Muxing,
            0.85,
            "Muxing source audio.",
            None,
            backend,
        ),
    )
    .await?;
    let mux_tmp = media_path.with_extension("audiomux.mp4");
    let ctx = FfmpegContext::new(api, settings, &job.id, SEEDVR2_CANCEL_MESSAGE);
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            source_path.to_string_lossy().into_owned(),
            "-map".to_owned(),
            "0:v:0".to_owned(),
            "-map".to_owned(),
            "1:a:0?".to_owned(),
            "-c:v".to_owned(),
            "copy".to_owned(),
            "-c:a".to_owned(),
            "aac".to_owned(),
            "-movflags".to_owned(),
            "+faststart".to_owned(),
            "-shortest".to_owned(),
            mux_tmp.to_string_lossy().into_owned(),
        ],
        Some(ctx),
    )
    .await?;
    tokio::fs::rename(&mux_tmp, &media_path).await?;

    let display_name = req
        .display_name
        .clone()
        .unwrap_or_else(|| match &source_display {
            Some(name) => format!("{name} ({factor}x upscaled)"),
            None => format!("Upscaled video ({factor}x)"),
        });
    let raw_settings = json!({
        "engine": "seedvr2",
        "model": req.model,
        "factor": factor,
        "softness": req.softness,
        "sourceAssetId": req.source_asset_id,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "targetWidth": out_w,
        "targetHeight": out_h,
        "frameCount": out_count,
    });
    let fact = json!({
        "type": "video",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "video/mp4",
        "width": out_w,
        "height": out_h,
        "duration": duration,
        "fps": out_fps,
        "quality": "best",
        "family": "video",
        "seed": seed as i64,
        "displayName": display_name,
        "createdAt": created_at,
        "mode": "video_upscale",
        "model": req.model,
        "adapter": SEEDVR2_ADAPTER,
        "prompt": "",
        "negativePrompt": Value::Null,
        "loras": [],
        "rawAdapterSettings": raw_settings,
        "sourceAssetId": req.source_asset_id,
        "timelineContext": json!({}),
    });
    let result = json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": SEEDVR2_ADAPTER,
        "model": req.model,
        "generationSet": {
            "id": genset_id,
            "mode": "video_upscale",
            "model": req.model,
            "prompt": "",
            "negativePrompt": Value::Null,
            "count": 1,
            "createdAt": created_at,
        },
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal");

    update_job(
        api,
        &job.id,
        video_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Video upscale complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend-neutral video helpers shared by the MLX (macOS) and candle
// (Windows/CUDA) lanes (sc-8830, F-028). These collapse the byte-identical
// twin pairs that had drifted apart between the two backends: the dense-adapter
// resolver, the LoRA-file resolver (path-confined + core `first_safetensors_path`),
// the LoRA strength reader, the MLX quant reader, and the negative-prompt idiom.
// Backend-specific behavior (which SAM3 module, which driving-clip loader) is
// injected by the callers, never re-branched at runtime.
// ---------------------------------------------------------------------------

/// At most 5 user LoRAs per job (mirrors the image path's `MAX_JOB_LORAS`). Shared by every
/// dense (single-transformer) video family on both backends.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MAX_JOB_LORAS: usize = 5;

/// The trimmed `negative_prompt`, or `None` when empty/whitespace — the single source for the
/// idiom the video dispatchers all repeated (sc-8830). Empty negatives must be `None`, not
/// `Some("")`, so the engines treat them as "no negative" rather than conditioning on the empty
/// string.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn non_empty_negative_prompt(request: &VideoRequest) -> Option<String> {
    let trimmed = request.negative_prompt.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// A LoRA spec's strength (`weight`, default 0.8 — matches the image path). Shared by both
/// backends (sc-8830; formerly the byte-identical `lora_scale` / `candle_scail2_lora_scale`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn lora_scale(lora: &Value) -> f32 {
    lora.get("weight")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(0.8) as f32
}

/// Resolve a LoRA spec's file (a directory → its first `.safetensors`, recursively via core's
/// [`first_safetensors_path`]), verifying it exists. The `path` originates from
/// attacker-controllable job payload, so it is first confined to an app-managed root
/// (sc-5723 / WKA-002) via [`normalize_app_managed_lora_path`] before any on-disk use.
///
/// Shared by both backends (sc-8830). The old candle twin `candle_resolve_lora_file`
/// re-implemented the directory scan with a shallow (non-recursive) case-sensitive `read_dir`,
/// which missed nested `subdir/model.safetensors` snapshots and uppercase `.SAFETENSORS`
/// extensions that the macOS path (core `first_safetensors_path`) resolved. Converging on the
/// core helper fixes that latent candle-lane bug; path confinement is preserved identically.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_lora_file(
    settings: &Settings,
    path: PathBuf,
    declared: Option<&str>,
) -> WorkerResult<PathBuf> {
    let path = crate::normalize_app_managed_lora_path(settings, &path)?;
    let file = if path.is_dir() {
        // Prefer the manifest-declared adapter over an arbitrary directory scan so a
        // trained LoRA loads its final adapter, not a step checkpoint (sc-10221).
        crate::resolve_adapter_in_dir(&path, declared).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "LoRA has no .safetensors under {}",
                path.display()
            ))
        })?
    } else {
        path
    };
    if !file.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "LoRA file is missing: {}",
            file.display()
        )));
    }
    Ok(file)
}

/// Build the adapter specs for a **dense** (single-transformer) video family — Wan-VACE and
/// SCAIL-2 (both backends). Unlike the MoE Wan path there is no Lightning distill pair and no
/// high/low experts, so every user LoRA/LoKr/LoHa is applied shared (`moe_expert: None`); the
/// engine sniffs the format and merges it. `classify_adapter` tags SceneWorks peft LoKr as
/// `Lokr` and everything else (incl. third-party LyCORIS) as `Lora`. Parameterized by the
/// per-family max-LoRA cap so the confinement + count guard lives in one place (sc-8830).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_dense_adapters(
    settings: &Settings,
    request: &VideoRequest,
    max_loras: usize,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > max_loras {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {max_loras} LoRAs per job."
        )));
    }
    let mut specs: Vec<AdapterSpec> = Vec::new();
    for lora in &request.loras {
        let path = crate::image_jobs::lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = crate::image_jobs::classify_adapter(&file)?;
        specs.push(AdapterSpec {
            path: file,
            scale: lora_scale(lora),
            kind,
            pass_scales: None,
            moe_expert: None,
        });
    }
    Ok(specs)
}

/// MLX quantization for a dense video load (Bernini / SCAIL-2). Explicit `mlxQuantize`: `<= 0` ⇒ bf16
/// (power users with ample RAM), `<= 4` ⇒ Q4, `>= 5` ⇒ Q8. **No explicit pick ⇒ Q4** — a deliberate,
/// owned exception to epic 10721's app-wide **Q8** generation default (sc-10750, owner-confirmed
/// 2026-07-11). It is NOT an oversight, and it does NOT hold dense video back from "Q8 app-wide":
///
/// - The primary default path is the turnkey tier machinery [`bernini_tier_order`] /
///   [`scail2_tier_order`], which ALREADY defaults to Q8 clamped-to-installed (sc-10726). So a modern
///   tier-subdir install resolves Q8 whenever the `q8/` tier is on disk. This resolver only feeds two
///   narrow spots, and Q4 is the safe answer for BOTH:
///     1. `legacy_quant` for a LEGACY flat snapshot (no tier subdirs, pre-sc-9944/9945): the flat bf16
///        layout (~93 GB Bernini) is quantized AT LOAD, so Q4 (~37 GB resident) is the OOM-safe floor;
///        Q8 (~55 GB) would risk an OOM on the default box for zero gain on a deprecated path.
///     2. the on-demand tier-fetch gate [`ensure_scail2_tier_present`]: a Q8 default here would
///        auto-pull the ~55 GB `q8/` tier on EVERY default job — Q4 keeps a default job from
///        triggering a huge unrequested download (Q8 is fetched only on an explicit pick).
///
/// Never defaults to bf16 — the bf16 snapshots are far too large for the default box. The
/// epic-consistent upgrade (default to the highest tier that FITS the machine's *video-runtime* memory
/// budget, falling back to Q4 on tight boxes) is the deferred capability / `Auto` work — sc-10733 (S8),
/// which owns the Apple-Silicon, video-workload-aware transient calibration; it must NOT be duplicated
/// with a divergent probe here (epic 10721 R4: one capability calc, no drift).
///
/// Shared by both MLX dense families (sc-8830; formerly the byte-identical `resolve_bernini_quant`
/// / `resolve_scail2_quant`).
#[cfg(target_os = "macos")]
fn resolve_mlx_dense_quant(request: &VideoRequest) -> Option<Quant> {
    match request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        Some(bits) if bits <= 0 => None,
        Some(bits) if bits <= 4 => Some(Quant::Q4),
        Some(_) => Some(Quant::Q8),
        None => Some(Quant::Q4),
    }
}

/// Cancel message shared by every SCAIL-2 person-segmentation pass (both backends).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SCAIL2_SEGMENT_CANCEL_MESSAGE: &str = "SCAIL-2 canceled during person segmentation.";

/// Run a SCAIL-2 person-segmentation-and-paint pass on the blocking pool under the heartbeat
/// keepalive (sc-8390 / sc-8807). The cold multi-GB SAM3 checkpoint parse + per-frame propagation
/// can exceed the API's 90s stale-sweep, so the keepalive drives progress and its cancel poll trips
/// the flag the engine's per-frame propagate contract observes between frames. Backend-neutral
/// (sc-8830): the caller's `segment` closure captures whichever SAM3 module the build links (MLX
/// `person_segment_sam3` vs candle `person_segment_sam3_candle`) plus the paint background, so the
/// heartbeat orchestration lives in exactly one place instead of a per-backend twin.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn scail2_segment_blocking<R, F>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    task_label: &'static str,
    segment: F,
) -> WorkerResult<R>
where
    R: Send + 'static,
    F: FnOnce(gen_core::CancelFlag) -> WorkerResult<R> + Send + 'static,
{
    let cancel = gen_core::CancelFlag::new();
    let flag = cancel.clone();
    run_blocking_with_heartbeat(
        api,
        settings,
        job_id,
        Some(cancel),
        SCAIL2_SEGMENT_CANCEL_MESSAGE,
        task_label,
        crate::no_cancel_ack(),
        tokio::task::spawn_blocking(move || segment(flag)),
    )
    .await
}

/// Assemble the SCAIL-2 `animate_character` conditioning (`Reference` + reference `Mask` +
/// `ControlClip`) from an already-loaded reference image + driving frames. The two SAM3
/// segmentation passes (reference → single painted mask, driving clip → per-frame painted masks)
/// are supplied as closures so the backend-specific SAM3 module + paint background convention live
/// at the call site while this orchestration (heartbeat, `ControlClip` shape) is shared (sc-8830 —
/// collapses the ~100-line MLX/candle `resolve_scail2_conditioning` twin).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn assemble_scail2_animate_conditioning<FR, FD>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    reference: Image,
    driving: Vec<Image>,
    segment_reference: FR,
    segment_driving: FD,
) -> WorkerResult<Vec<Conditioning>>
where
    FR: FnOnce(gen_core::CancelFlag) -> WorkerResult<Image> + Send + 'static,
    FD: FnOnce(gen_core::CancelFlag) -> WorkerResult<Vec<Image>> + Send + 'static,
{
    let ref_mask = scail2_segment_blocking(
        api,
        settings,
        job_id,
        "scail2 reference segment task",
        segment_reference,
    )
    .await?;
    let driving_mask = scail2_segment_blocking(
        api,
        settings,
        job_id,
        "scail2 driving segment task",
        segment_driving,
    )
    .await?;
    Ok(vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_mask,
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ])
}

// ---------------------------------------------------------------------------
// Real MLX Wan2.2 generation (macOS, via mlx-gen-wan, sc-3034): T2V/TI2V (5B
// dense, z48 VAE), T2V/I2V (A14B dual-expert MoE) + MoE/Lightning LoRA. Decodes
// the engine's `GenerationOutput::Video { frames, fps, audio: None }` into a
// `DecodedVideo` and reuses the [`encode_media`] pipeline above. LTX (sc-3035) and
// every other model keep the procedural stub.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan asset (mirrors the image `mlx_*` convention).
#[cfg(target_os = "macos")]
const WAN_ADAPTER: &str = "mlx_wan";

/// Raw-settings recorded on a real MLX Wan asset: the request's `advanced` knobs plus
/// the real-inference markers (mirrors the image `mlx_raw_settings`). Also records the
/// effective sampler the worker actually dispatched (sc-4997) — the 5B interim default / the
/// 14B Lightning preset — so the chosen steps/CFG is inspectable on the asset, not silent.
#[cfg(target_os = "macos")]
fn wan_raw_settings(request: &VideoRequest, engine_id: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    let (steps, guidance) = wan_sampling(engine_id, request);
    if let Some(steps) = steps {
        raw.insert("effectiveSteps".to_owned(), json!(steps));
    }
    if let Some(guidance) = guidance {
        raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
    }
    Value::Object(raw)
}

/// SceneWorks Wan model id → mlx-gen registry id, or `None` if `model` is not a Wan
/// family id this worker serves.
#[cfg(target_os = "macos")]
fn wan_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        _ => None,
    }
}

/// Whether the linked Wan engine can serve this request now: a Wan model id with
/// resolvable on-disk weights. Off macOS / non-Wan / weights-absent → the stub
/// (mirrors the image `mlx_available` weights gate).
/// Fail-loud gate for the stub fallback (sc-4176): when the requested model id
/// maps to an MLX video engine family (Wan/LTX/SVD) but its weights/snapshot
/// can't be resolved, surface the resolver's precise re-download error instead
/// of silently degrading the job to procedural stub output. Non-engine model
/// ids pass through (the stub is their intended path).
#[cfg(target_os = "macos")]
pub(crate) fn ensure_video_engine_weights(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    if let Some(engine_id) = wan_engine_id(&request.model) {
        resolve_wan_model_dir(settings, &request.model, engine_id)?;
    }
    if ltx_engine_id(&request.model).is_some() {
        resolve_ltx_model_dir(settings, request)?;
    }
    if svd_engine_id(&request.model).is_some() {
        if request.source_asset_id.is_none() {
            return Err(WorkerError::InvalidPayload(
                "SVD image-to-video requires a source image asset.".to_owned(),
            ));
        }
        resolve_svd_model_dir(settings)?;
    }
    if bernini_engine_id(&request.model).is_some() {
        resolve_bernini_model_dir(settings)?;
    }
    if scail2_engine_id(&request.model).is_some() {
        resolve_scail2_model_dir(settings)?;
    }
    // Mochi 1 (epic 1788 / sc-11992). Without this arm a Mochi job whose weights don't resolve falls
    // to `VideoRoute::Stub` and the user is handed a PROCEDURAL FAKE VIDEO instead of the resolver's
    // precise "download the tier" / "the shared components are missing" error — exactly the silent
    // degradation sc-4176 added this gate to prevent.
    if mochi_engine_id(&request.model).is_some() {
        resolve_mochi_model_dir(settings, request)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn wan_available(request: &VideoRequest, settings: &Settings) -> bool {
    match wan_engine_id(&request.model) {
        Some(engine_id) => resolve_wan_model_dir(settings, &request.model, engine_id).is_ok(),
        None => false,
    }
}

/// Resolve the converted MLX snapshot directory for a Wan model (mirrors the Python
/// `_resolve_wan_mlx`): an env override, then the app-managed `<data>/models/mlx/<id>`,
/// then (T2V-14B only) the turnkey HF MLX snapshot. Errors clearly if none is present.
#[cfg(target_os = "macos")]
fn resolve_wan_model_dir(
    settings: &Settings,
    model: &str,
    _engine_id: &str,
) -> WorkerResult<PathBuf> {
    let (env, local_id, hf_repo): (&str, &str, Option<&str>) = match model {
        "wan_2_2" => (
            "SCENEWORKS_MLX_WAN5B_DIR",
            "wan_2_2",
            Some("SceneWorks/wan2.2-ti2v-5b-mlx"),
        ),
        "wan_2_2_t2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_T2V_DIR",
            "wan_2_2_t2v_14b",
            Some("SceneWorks/wan2.2-t2v-a14b-mlx"),
        ),
        "wan_2_2_i2v_14b" => (
            "SCENEWORKS_MLX_WAN14B_I2V_DIR",
            "wan_2_2_i2v_14b",
            Some("SceneWorks/wan2.2-i2v-a14b-mlx"),
        ),
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "not a Wan model: {other}"
            )))
        }
    };
    if let Some(dir) = local_mlx_dir(settings, env, local_id) {
        return Ok(dir);
    }
    if let Some(repo) = hf_repo {
        if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, repo) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{model}: no MLX weights found. Convert/download the Wan checkpoint into {}{}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join(local_id)
            .display(),
        hf_repo
            .map(|repo| format!(" (or download the turnkey repo {repo})"))
            .unwrap_or_default(),
    )))
}

/// A locally-converted MLX dir for the model (env override, then
/// `<data>/models/mlx/<id>`), counted only when it holds a `config.json` — mirrors the
/// Python `_local_mlx_dir`, so a locally-quantized conversion supersedes a turnkey download.
#[cfg(target_os = "macos")]
fn local_mlx_dir(settings: &Settings, env: &str, local_id: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(override_dir) = std::env::var(env) {
        let trimmed = override_dir.trim();
        if !trimmed.is_empty() {
            candidates.push(PathBuf::from(trimmed));
        }
    }
    candidates.push(settings.data_dir.join("models").join("mlx").join(local_id));
    candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file())
}

/// The turnkey SceneWorks Wan2.2 **T2V-A14B** MLX repo (sc-9942, epic 8506). Hosts the quant matrix
/// as self-contained tier subdirs `q4/` (default) + `q8/` + `bf16/`, each a COMPLETE dual-expert
/// snapshot (both MoE experts + UMT5 T5 encoder + z16 VAE + tokenizer + `config.json`). This replaces
/// the flat dense-bf16 layout (which quantized at LOAD, staging the full bf16 experts first); the
/// worker now descends into the chosen tier so a pre-packed snapshot loads with no install-time
/// convert peak. The flat root files are kept for back-compat with already-shipped workers that
/// resolve the repo root (a cleanup story drops them once those age out); a new worker only ever
/// resolves the tier subdirs.
#[cfg(target_os = "macos")]
const WAN_T2V_14B_REPO: &str = "SceneWorks/wan2.2-t2v-a14b-mlx";

/// Pinned revision for [`WAN_T2V_14B_REPO`] (mirrors [`LTX_BUNDLE_REVISION`], sc-9879). The repo is a
/// hard-coded const — no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches —
/// so pulling the mutable `main` branch would let an upstream re-push silently swap a checkpoint we
/// load. Pin the exact commit that adds the `q4/`/`q8/`/`bf16/` tier subdirs for defense-in-depth
/// (the native downloader still verifies each file's own hash on download). This is the commit that added the
/// `q4/`/`q8/`/`bf16/` tier subdirs (sc-9942).
#[cfg(target_os = "macos")]
const WAN_T2V_14B_REVISION: &str = "991eb255c544bbb2e1f1e07da4355c2f0a5337b7";

/// The turnkey SceneWorks Wan2.2 **I2V-A14B** MLX repo (sc-9943, epic 8506). The image→video sibling
/// of [`WAN_T2V_14B_REPO`]: same self-contained `q4/`/`q8/`/`bf16/` tier layout (both MoE experts +
/// UMT5 T5 + z16 VAE + tokenizer + `config.json`), differing only in the experts' `in_dim` (36
/// image-concat conditioning vs 16 text-only). The worker descends into the chosen tier so a
/// pre-packed snapshot loads with no install-time convert peak; the legacy flat root files stay for
/// already-shipped workers.
#[cfg(target_os = "macos")]
const WAN_I2V_14B_REPO: &str = "SceneWorks/wan2.2-i2v-a14b-mlx";

/// Pinned revision for [`WAN_I2V_14B_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The commit that adds
/// the `q4/`/`q8/`/`bf16/` tier subdirs to the I2V-A14B repo (sc-9943); pinning the exact commit (not
/// the mutable `main`) stops an upstream re-push from silently swapping a checkpoint the on-demand
/// `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each file's own hash on download).
#[cfg(target_os = "macos")]
const WAN_I2V_14B_REVISION: &str = "c6c786170031eccc3a1fac0f98f1ad4ff988271e";

/// The turnkey SceneWorks Wan2.2 **TI2V-5B** MLX repo (sc-9941, epic 8506). The single-expert sibling
/// of the A14B repos: same self-contained `q4/`/`q8/`/`bf16/` tier layout, but ONE transformer
/// (`model.safetensors`) rather than the dual `high/low_noise_model` MoE experts (still + UMT5 T5 +
/// z16 VAE + tokenizer + `config.json`). The worker descends into the chosen tier so a pre-packed
/// snapshot loads with no install-time convert peak; the legacy flat root files stay for
/// already-shipped workers (cleanup sc-9977).
#[cfg(target_os = "macos")]
const WAN_TI2V_5B_REPO: &str = "SceneWorks/wan2.2-ti2v-5b-mlx";

/// Pinned revision for [`WAN_TI2V_5B_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The commit that adds
/// the `q4/`/`q8/`/`bf16/` tier subdirs to the TI2V-5B repo (sc-9941); pinning the exact commit (not
/// the mutable `main`) stops an upstream re-push from silently swapping a checkpoint the on-demand
/// `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each file's own hash on download).
#[cfg(target_os = "macos")]
const WAN_TI2V_5B_REVISION: &str = "bb1b055249614cf9d7cf4373fbdbc184b77dee88";

/// Pinned commit revision for the A14B Lightning distill-LoRA repo `lightx2v/Wan2.2-Lightning` (sc-11168 /
/// F-007 — completes the sc-9879 rollout). Both the MLX (`ensure_wan_lightning_present`) and candle
/// (`candle_ensure_wan_lightning_present`) self-heal fetches were pulling the mutable `main` branch, so an
/// upstream re-push (or a compromised token) could silently swap the high/low distill weights we load.
/// Pin the exact commit for defense-in-depth (the native downloader still verifies each file's own hash on
/// download). Shared by BOTH lanes so the twins agree. Gated to the lanes that actually fetch it (macOS
/// MLX or the candle build) so a Linux-non-candle build doesn't flag it dead.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const WAN_LIGHTNING_REVISION: &str = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e";

/// The files that make an **A14B** (dual-expert MoE) Wan tier subdir COMPLETE: both experts + the T5
/// encoder + VAE + tokenizer + `config.json`.
#[cfg(target_os = "macos")]
const WAN_A14B_TIER_FILES: &[&str] = &[
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
];

/// The files that make a **TI2V-5B** (single-expert) Wan tier subdir COMPLETE: the one transformer
/// (`model.safetensors`) + the T5 encoder + VAE + tokenizer + `config.json`.
#[cfg(target_os = "macos")]
const WAN_TI2V_5B_TIER_FILES: &[&str] = &[
    "model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
];

/// The tier-completeness file set for a Wan quant-matrix model: the single-expert TI2V-5B ships one
/// `model.safetensors`, the A14B MoE models ship the two `high/low_noise_model.safetensors` experts.
#[cfg(target_os = "macos")]
fn wan_tier_files(model: &str) -> &'static [&'static str] {
    if model == "wan_2_2" {
        WAN_TI2V_5B_TIER_FILES
    } else {
        WAN_A14B_TIER_FILES
    }
}

/// Map a Wan quant-matrix video model id to its `(quant-matrix repo, pinned revision)` for the
/// on-demand tier fetch, or `None` for a model with no hosted tier matrix. The TI2V-5B (sc-9941),
/// T2V-A14B (sc-9942) and I2V-A14B (sc-9943) turnkeys host the SAME self-contained
/// `q4/`/`q8/`/`bf16/` tier layout (epic 8506); only the repo + pinned commit (and the single- vs
/// dual-expert file set, see [`wan_tier_files`]) differ, so the whole tier-resolve/fetch path is
/// shared and keyed only here. `request.model` is `"wan_2_2"` for the TI2V-5B engine.
#[cfg(target_os = "macos")]
fn wan_tier_repo(model: &str) -> Option<(&'static str, &'static str)> {
    match model {
        "wan_2_2" => Some((WAN_TI2V_5B_REPO, WAN_TI2V_5B_REVISION)),
        "wan_2_2_t2v_14b" => Some((WAN_T2V_14B_REPO, WAN_T2V_14B_REVISION)),
        "wan_2_2_i2v_14b" => Some((WAN_I2V_14B_REPO, WAN_I2V_14B_REVISION)),
        _ => None,
    }
}

/// Parse `advanced.mlxQuantize` (int or numeric string) for the Wan quant-matrix tier selector.
#[cfg(target_os = "macos")]
fn wan_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// The Wan2.2 quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`ltx_bundle_tier_order`]): `mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`, an explicit `1..=4` ⇒
/// `q4`, and — with NO explicit `mlxQuantize` — the **`q4`** default (q4-first).
///
/// The video lane deliberately does NOT take epic 10721's app-wide **Q8** default (sc-10726); it keeps
/// the pre-sc-10726 q4-first default (sc-10859). Rationale: the MLX video lane has no user Q8 lever
/// (Video Studio's quant control is the GGUF/torch path), so a silent Q8 default gives no UI-accessible
/// quality benefit and only ever surfaces as an *accidental* default when the Q8 tier landed on disk via
/// a side lane — where it risks a video-runtime OOM at heavy res/frame counts (the install-fit clamp
/// doesn't help: the sc-8516 budget is 1024²-image-calibrated). Q8/bf16 stay reachable on an explicit
/// pick; `bf16` stays OUT of the default order so a default job never pulls the huge dense tier. The
/// precise "highest tier that fits the video-runtime budget" default is the deferred sc-10733 (S8).
#[cfg(target_os = "macos")]
fn wan_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match wan_quant_bits(request) {
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        // No explicit pick (`None`) OR an explicit `1..=4` ⇒ q4-first (sc-10859 video carve-out).
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained Wan2.2 tier snapshot, given the model's expected tier
/// file set (`files`, from [`wan_tier_files`]): the transformer(s), the T5 encoder, VAE, tokenizer,
/// and `config.json`. A partially-downloaded tier fails this so [`wan_tier_subdir`] falls through to
/// a smaller complete tier rather than half-loading.
#[cfg(target_os = "macos")]
fn wan_tier_is_complete(dir: &Path, files: &[&str]) -> bool {
    files.iter().all(|file| dir.join(file).is_file())
}

/// Descend a resolved Wan2.2 quant-matrix repo `root` into the requested quant tier subdir
/// (sc-9941 TI2V-5B / sc-9942 T2V / sc-9943 I2V, epic 8506), mirroring [`ltx_bundle_subdir`]. Returns
/// the first COMPLETE tier in [`wan_tier_order`] (all of a model's weights — one transformer for the
/// 5B, both experts for the A14B — live in the SAME subdir, so one resolution covers the model), or
/// `None` when the repo has no complete tier subdir — a legacy flat snapshot, where the caller keeps
/// the root + load-time quant.
#[cfg(target_os = "macos")]
fn wan_tier_subdir(root: &Path, request: &VideoRequest) -> Option<PathBuf> {
    let files = wan_tier_files(&request.model);
    wan_tier_order(request)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| wan_tier_is_complete(dir, files))
}

/// Resolve the Wan2.2 `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9941 TI2V-5B / sc-9942 T2V / sc-9943 I2V).
/// A pre-packed
/// tier's `config.json` is authoritative — [`WanTransformer::from_weights`] builds the experts at the
/// stored bits and `resolve_load_time_quant` rejects a conflicting `spec.quantize` as a hard error —
/// so a resolved tier loads with `quant = None`: `mlxQuantize` selects WHICH tier, never a load-time
/// requant (the `bf16/` tier is dense, so `None` ⇒ dense too). A legacy flat snapshot (no tier
/// subdirs) keeps today's behavior: load the root and quantize at load per [`resolve_wan_quant`].
#[cfg(target_os = "macos")]
fn resolve_wan_tier_dir_and_quant(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &'static str,
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_wan_model_dir(settings, &request.model, engine_id)?;
    match wan_tier_subdir(&root, request) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, resolve_wan_quant(request))),
    }
}

/// On-demand fetch of a non-default Wan2.2 quant-matrix tier subdir (sc-9941 TI2V-5B / sc-9942 T2V /
/// sc-9943 I2V, mirrors [`ensure_ltx_q8_present`] / [`ensure_ltx_bf16_present`]). The macOS default
/// download is the lean `q4/` tier; a job that opts into a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`,
/// `>= 8` ⇒ `q8`) pulls just that subdir from that model's FIXED [`wan_tier_repo`] revision the first
/// time it is requested so [`wan_tier_subdir`] can resolve it. No-op for a model with no hosted tier
/// matrix, a `q4` (default)
/// job, when the repo snapshot isn't downloaded yet (resolve surfaces the clear error), or when the
/// tier is already complete. Fails loud on a real download error — fast, before any compute; a
/// tier that isn't published yet stays absent so resolve falls back to a smaller complete
/// tier.
#[cfg(target_os = "macos")]
async fn ensure_wan_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    let Some((repo, revision)) = wan_tier_repo(&request.model) else {
        return Ok(());
    };
    let tier = match wan_quant_bits(request) {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b >= 8 => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) else {
        return Ok(());
    };
    if wan_tier_is_complete(&root.join(tier), wan_tier_files(&request.model)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, repo, revision, &files)
        .await
        .map(|_| ())
}

/// On-demand fetch of the 4-step Lightning distill LoRA pair (`lightx2v/Wan2.2-Lightning`) for the
/// A14B MoE models (sc-10030). Normally the pair installs as a manifest `coRequisite` alongside the
/// model (sc-9696), but a worker that installed the model BEFORE the coRequisite was added has the
/// tiers without the LoRA — and [`resolve_wan_adapters`] then hard-errors when the toggle is on. This
/// self-heals that case: it pulls just the per-architecture high/low pair the first time a gen needs
/// it (twin of [`ensure_wan_tier_present`] / the candle `ensure_qwen_lightning_lora_cached`). No-op
/// when the Lightning toggle is off (sc-10047 — the native multi-step recipe needs no LoRA), for a
/// non-A14B engine, or when the pair is already cached. A pair still missing after the fetch makes
/// resolve surface the clear "fetch it via the model manager" error. Fails loud on a real download error —
/// fast, before any compute.
#[cfg(target_os = "macos")]
async fn ensure_wan_lightning_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<()> {
    // sc-10047: Lightning is a default-on toggle now. When the job opted out (`advanced.lightning`
    // = false), the native multi-step CFG recipe runs with no Lightning adapter, so we need nothing
    // here. Default-on (or explicitly on) still wants the pair present and self-heals if absent.
    if !wan_lightning_on(engine_id, request) {
        return Ok(());
    }
    // Per-architecture subdir (NOT cross-compatible, sc-4997); must match `resolve_lightning_loras`.
    let subdir = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        // Only the A14B MoE models bake Lightning — every other engine needs nothing here.
        _ => return Ok(()),
    };
    const REPO: &str = "lightx2v/Wan2.2-Lightning";
    // Fast path: both halves already materialized in the hub cache (the common case after install).
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, REPO) {
        let base = snapshot.join(subdir);
        if base.join("high_noise_model.safetensors").is_file()
            && base.join("low_noise_model.safetensors").is_file()
        {
            return Ok(());
        }
    }
    let files = vec![
        format!("{subdir}/high_noise_model.safetensors"),
        format!("{subdir}/low_noise_model.safetensors"),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        REPO,
        WAN_LIGHTNING_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// The 4-step Lightning distill LoRA pair (high/low) for an A14B MoE model
/// (`lightx2v/Wan2.2-Lightning`, the rank-64 Seko distill). The subdir is architecture-specific:
/// T2V-A14B (V1.1) and I2V-A14B (V1) ship distinct LoRAs that are NOT cross-compatible (sc-4997).
/// Errors if not downloaded / the per-architecture subdir is missing.
#[cfg(target_os = "macos")]
fn resolve_lightning_loras(
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<(PathBuf, PathBuf)> {
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, "lightx2v/Wan2.2-Lightning")
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: the Lightning distill LoRA (lightx2v/Wan2.2-Lightning) is not \
                 downloaded — fetch it via the model manager"
            ))
        })?;
    let base = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "{other}: no Lightning distill LoRA — only the A14B MoE models bake Lightning"
            )))
        }
    };
    let high = snapshot.join(base).join("high_noise_model.safetensors");
    let low = snapshot.join(base).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id}: Lightning LoRA file missing: {}",
                file.display()
            )));
        }
    }
    Ok((high, low))
}

/// The `.low_noise.safetensors` sibling of a Wan A14B MoE high-noise LoRA file, or
/// `None` when the file is not the high-noise half of a pair (port of the Python
/// `wan_moe_low_noise_sibling`; case-insensitive `.high_noise.safetensors` suffix).
#[cfg(target_os = "macos")]
fn wan_moe_low_noise_sibling(primary: &Path) -> Option<PathBuf> {
    const HIGH: &str = ".high_noise.safetensors";
    let name = primary.file_name()?.to_str()?;
    if !name.to_ascii_lowercase().ends_with(HIGH) {
        return None;
    }
    let stem = &name[..name.len() - HIGH.len()];
    let sibling = primary.with_file_name(format!("{stem}.low_noise.safetensors"));
    sibling.is_file().then_some(sibling)
}

/// Build the adapter specs for a Wan generation (sc-3034): the Lightning distill pair
/// (both A14B MoE models — T2V + I2V — tagged high/low, sc-4997) followed by the user LoRAs.
/// On the MoE models a user
/// `*.high_noise.safetensors` with a `.low_noise` sibling tags high→High / low→Low; a
/// single-file LoRA is shared (both experts on MoE, the single model on the 5B). peft LoKr AND
/// third-party LyCORIS (LoHa / non-peft LoKr) both apply on the MLX Wan/LTX paths now (epic 3641,
/// sc-3671) — `classify_adapter` returns `Lora` for third-party and the engine detects + merges it.
#[cfg(target_os = "macos")]
fn resolve_wan_adapters(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    let mut specs: Vec<AdapterSpec> = Vec::new();

    // Lightning distill (both A14B MoE models — T2V + I2V, sc-4997): 4-step, applied per-expert at
    // strength 1.0 through the standard adapter path. As of sc-10047 this is a **default-on toggle**
    // (`advanced.lightning`) rather than mandatory — the mlx-gen additive path (epic 10043) applies
    // it on the quantized tiers, so the pair is added only when the toggle is on. When off, the
    // native multi-step CFG recipe runs ([`wan_sampling`]) with no Lightning adapter. User LoRAs
    // below are honored in both states. The subdir is resolved per architecture (not cross-compatible).
    if is_moe && wan_lightning_on(engine_id, request) {
        let (high, low) = resolve_lightning_loras(settings, engine_id)?;
        specs.push(moe_adapter(high, 1.0, AdapterKind::Lora, MoeExpert::High));
        specs.push(moe_adapter(low, 1.0, AdapterKind::Lora, MoeExpert::Low));
    }

    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = classify_adapter(&file)?;
        let scale = lora_scale(lora);
        match (is_moe, wan_moe_low_noise_sibling(&file)) {
            (true, Some(low)) => {
                // A MoE pair → high half to the high-noise expert, the sibling to the low.
                let low_kind = classify_adapter(&low)?;
                specs.push(moe_adapter(file, scale, kind, MoeExpert::High));
                specs.push(moe_adapter(low, scale, low_kind, MoeExpert::Low));
            }
            _ => {
                // Single-file → shared (both experts on MoE; the dense single model on the 5B).
                specs.push(AdapterSpec {
                    path: file,
                    scale,
                    kind,
                    pass_scales: None,
                    moe_expert: None,
                });
            }
        }
    }
    Ok(specs)
}

#[cfg(target_os = "macos")]
fn moe_adapter(path: PathBuf, scale: f32, kind: AdapterKind, expert: MoeExpert) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind,
        pass_scales: None,
        moe_expert: Some(expert),
    }
}

/// Build the adapter specs for a Wan-VACE generation (sc-3893 worker routing). Unlike the base Wan
/// path, VACE-1.3B is a **single dense** transformer: no Lightning distill, no MoE high/low experts.
/// So every user LoRA/LoKr is applied shared with `moe_expert: None` — the engine `wan_vace` provider
/// merges diffusers-named LoRA/LoKr (mlx-gen #184) and rejects `moe_expert` tags. `classify_adapter`
/// tags SceneWorks peft LoKr as `Lokr` and everything else (incl. third-party LyCORIS LoHa / non-peft
/// LoKr) as `Lora`, which the engine then detects + merges by key sniff (epic 3641). Delegates to
/// the shared [`resolve_dense_adapters`] (sc-8830).
#[cfg(target_os = "macos")]
fn resolve_wan_vace_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// Build the adapter specs for a SCAIL-2 generation (sc-5451 inference LoRA path, mlx-gen #462).
/// SCAIL-2 is a single **dense** Wan2.1-14B-I2V transformer — like Wan-VACE, no Lightning distill and
/// no MoE high/low experts — so every LoRA is applied shared with `moe_expert: None`. The engine
/// installs a standard `lora_down/up` (PEFT/diffusers/kohya/LoKr) adapter as a forward-time residual
/// over the (Q4/Q8) base; `classify_adapter` tags SceneWorks peft LoKr as `Lokr` and everything else
/// (incl. third-party LyCORIS) as `Lora`. This carries both a user-selected SCAIL-2 LoRA and the
/// bundled Bias-Aware DPO quality LoRA (both surface through `request.loras`). A lightx2v diff-patch
/// "lightning" LoRA installs via the engine's in-place diff-patch merge (sc-5684); selecting it makes
/// the worker apply the step-distill recipe (`scail2_sampling`, sc-5700). Delegates to the shared
/// [`resolve_dense_adapters`] (sc-8830) — the MLX Wan-VACE / SCAIL-2 twin.
#[cfg(target_os = "macos")]
fn resolve_scail2_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// The first-frame conditioning for a Wan generation: required for I2V-14B, optional for
/// the TI2V-5B (present → image-conditioned mask-blend, absent → pure T2V), and ignored
/// by the T2V-14B (text-only). Loads `source_asset_id` to an in-memory RGB8 image.
#[cfg(target_os = "macos")]
fn resolve_wan_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // first_last_frame is Wan-native only on the TI2V-5B mask-blend keyframe path (sc-3357);
    // the routing gate (`video_mode_is_mlx_eligible`) already restricts FLF to `wan_2_2`, but
    // guard here too so a mis-routed 14B MoE job fails clearly instead of silently dropping it.
    if request.mode == "first_last_frame" {
        if engine_id != "wan2_2_ti2v_5b" {
            return Err(WorkerError::InvalidPayload(format!(
                "first_last_frame is only supported on wan_2_2 (TI2V-5B), not {engine_id}."
            )));
        }
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    let required = engine_id == "wan2_2_i2v_14b";
    let accepts = required || engine_id == "wan2_2_ti2v_5b";
    if !accepts {
        return Ok(Vec::new());
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit to the output W×H by the chosen crop/pad mode (sc-6139) — see
            // `resolve_ltx_conditioning`; without it the provider VAE-encodes a stretched
            // first frame into its channel-concat `y`.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None if required => Err(WorkerError::InvalidPayload(
            "wan_2_2_i2v_14b: image-to-video requires a source image (sourceAssetId).".to_owned(),
        )),
        None => Ok(Vec::new()),
    }
}

/// Which boundary frame of a source clip to extract for Wan-native clip conditioning (sc-3357).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ClipFramePosition {
    /// The clip's first decoded frame (the right-side clip's head for `video_bridge`).
    First,
    /// The clip's last decoded frame (the source tail for `extend_clip` / the left-side clip).
    Last,
}

/// Build the Wan-native boundary [`Conditioning::Keyframe`] set for extend_clip / video_bridge
/// (sc-3357). Wan TI2V-5B has no in-context clip-append path (LTX's IC-LoRA `VideoClip`); its only
/// clip primitive is the single-frame mask-blend `Keyframe` (the same one Wan FLF rides). So the
/// faithful Wan-native form — matching the torch Wan reference, which routed these modes to plain
/// i2v (`_pipeline_kind` → `"image"`, never IC-LoRA/VACE) — pins the clip *boundary* frame(s):
/// - **extend_clip** → the source clip's last frame pinned at latent frame `0` (continue from it),
///   strength `videoConditioningStrength`.
/// - **video_bridge** → the left clip's last frame at `0` (`videoConditioningStrength`) + the right
///   clip's first frame at latent frame `-1` (the engine's negative-from-end index), strength
///   `bridgeRightVideoConditioningStrength`. Mechanically identical to first_last_frame.
///
/// Both strengths default to `1.0` (fully pinned), mirroring [`build_video_clip_conditioning`] and
/// the torch `_advanced_float` defaults. This is the single-frame fidelity ceiling for Wan; richer
/// motion-tail continuity is the LTX IC-LoRA path or native Wan-VACE (sc-3385 routing matrix).
#[cfg(target_os = "macos")]
fn build_wan_boundary_conditioning(
    request: &VideoRequest,
    left_frame: Image,
    right_frame: Option<Image>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::Keyframe {
        image: left_frame,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frame.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::Keyframe {
            image: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve extend_clip / video_bridge into Wan-native boundary [`Conditioning::Keyframe`]s
/// (sc-3357). Wan-native clip conditioning is **only** the TI2V-5B mask-blend keyframe path, so
/// guard the engine (the routing gate `video_mode_is_mlx_eligible` already restricts these to
/// `wan_2_2`, but fail clearly here too if a 14B MoE job is mis-routed). Extracts the boundary
/// frame(s) — the source clip's last frame (+ the right clip's first frame for bridge) — then maps
/// them via [`build_wan_boundary_conditioning`]. Unlike the LTX path this needs **no** IC-LoRA.
#[cfg(target_os = "macos")]
async fn resolve_wan_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    if engine_id != "wan2_2_ti2v_5b" {
        return Err(WorkerError::InvalidPayload(format!(
            "{} is only supported on wan_2_2 (TI2V-5B), not {engine_id}.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_frame = extract_clip_boundary_frame(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        ClipFramePosition::Last,
    )
    .await?;
    let right_frame = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_boundary_frame(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    build_wan_boundary_conditioning(request, left_frame, right_frame)
}

/// Decode a single boundary frame (first or last) of a source clip into an [`Image`], fit to the
/// output `width`×`height` by contain+pad (letterbox, `FRAME_PAD_COLOR`) so a clip whose aspect
/// differs from the output is not distorted — sc-6229, matching the `load_source_video_frames`
/// recipe (sc-3357, the Wan boundary-keyframe conditioning input). The last frame
/// uses ffmpeg `-sseof` to seek near the end + `-update 1` so each decoded frame overwrites the lone
/// output, leaving the final frame; the first frame is a plain `-frames:v 1`. Extracted via the
/// shared [`run_ffmpeg`] (binary resolution + heartbeat/cancel), then loaded off the async runtime.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn extract_clip_boundary_frame(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    position: ClipFramePosition,
) -> WorkerResult<Image> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let out = frames_dir.join("boundary.png");
    let mut args = vec!["ffmpeg".to_owned(), "-nostdin".to_owned(), "-y".to_owned()];
    if position == ClipFramePosition::Last {
        // Seek to ~2s before EOF; short clips clamp to the start (whole clip decoded). `-update 1`
        // overwrites the single output per frame, so the final decoded frame is what remains.
        args.push("-sseof".to_owned());
        args.push("-2".to_owned());
    }
    args.push("-i".to_owned());
    args.push(clip_path.display().to_string());
    args.push("-vf".to_owned());
    // Contain+pad (letterbox) to the output dims so a source clip whose aspect differs from the
    // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe.
    args.push(format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
    ));
    if position == ClipFramePosition::Last {
        args.push("-update".to_owned());
        args.push("1".to_owned());
    } else {
        args.push("-frames:v".to_owned());
        args.push("1".to_owned());
    }
    args.push(out.display().to_string());
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(args, Some(ctx)).await;
    let load = async {
        result?;
        let path = out.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Image> {
            let decoded = crate::image_decode::decode_image_any(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "boundary conditioning frame {}: {error}",
                        path.display()
                    ))
                })?
                .to_rgb8();
            Ok(Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            })
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frame = load.await;
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    frame
}

/// Map `advanced.mlxQuantize` to a quant level (≤0 → dense, ≤4 → Q4, else Q8). Absent →
/// `None`: dense bf16, or the engine builds it quantized from a pre-quantized snapshot.
#[cfg(target_os = "macos")]
fn resolve_wan_quant(request: &VideoRequest) -> Option<Quant> {
    let bits = request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })?;
    match bits {
        b if b <= 0 => None,
        b if b <= 4 => Some(Quant::Q4),
        _ => Some(Quant::Q8),
    }
}

/// Interim step count for the dense TI2V-5B until a 5B distill LoRA ships (sc-4999): half the
/// engine's 40-step default, so an out-of-the-box 1280×720 job no longer runs the ~40-min /
/// GPU-wedging 40-step+CFG schedule that wedged the GPU (sc-4986 / sc-4997). CFG is retained
/// (no 5B distill exists, so dropping it would hurt prompt adherence); the user can still dial
/// `steps`/`guidanceScale` lower from VideoStudio, and the engine pre-flight guard (sc-4986) is
/// the memory backstop. The full few-step / no-CFG preset lands once the 5B distill LoRA exists.
#[cfg(target_os = "macos")]
const WAN5B_INTERIM_STEPS: u32 = 20;

/// An optional positive-integer `advanced` knob (`steps`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_opt_u32(request: &VideoRequest, key: &str) -> Option<u32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    })
}

/// An optional float `advanced` knob (`guidanceScale`); accepts a number or a numeric string.
/// Shared by the MLX path and the candle video lane (sc-5097).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn advanced_opt_f32(request: &VideoRequest, key: &str) -> Option<f32> {
    request.advanced.get(key).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    })
}

/// `true` if the A14B MoE Lightning distill is engaged for this request (sc-10047). The Lightning
/// 4-step distill is now a **default-on toggle** (`advanced.lightning`) rather than mandatory: the
/// mlx-gen additive path (epic 10043) applies the high/low pair on the quantized tiers, so a job can
/// opt out and run the native multi-step CFG recipe instead. Only the two A14B MoE models (T2V + I2V)
/// bake Lightning — for every other engine (the dense 5B, non-Wan) this is irrelevant and returns
/// `false`. Backward compatible: an absent flag on an A14B job defaults to `true` (the prior
/// always-on behavior). A strict-bool `false` opts out; `true` (or absent) opts in.
#[cfg(target_os = "macos")]
fn wan_lightning_on(engine_id: &str, request: &VideoRequest) -> bool {
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    if !is_moe {
        return false;
    }
    // Absent ⇒ default-on for A14B; only an explicit strict-bool `false` opts out.
    request
        .advanced
        .get("lightning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

/// Per-model sampling for the base Wan path (sc-3034 / sc-4997 / sc-10047). On the A14B MoE models
/// (T2V + I2V) the recipe is now conditional on the Lightning toggle ([`wan_lightning_on`]):
/// - toggle **on** (default) → the 4-step Lightning distill preset: forced 4 steps / CFG-off
///   (guide 1.0), unchanged from before.
/// - toggle **off** → the native Wan2.2 A14B multi-step + CFG recipe: honor an explicit user
///   `steps`/`guidanceScale`, else `None` so the engine's own config.json A14B defaults (40 steps,
///   dual CFG) stand exactly.
///
/// The dense TI2V-5B has no distill LoRA yet (sc-4999) and no toggle: honor an explicit user
/// `steps`/`guidanceScale`, else apply the interim default ([`WAN5B_INTERIM_STEPS`], CFG retained).
/// `None` ⇒ the engine config default.
#[cfg(target_os = "macos")]
fn wan_sampling(engine_id: &str, request: &VideoRequest) -> (Option<u32>, Option<f32>) {
    if engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b" {
        if wan_lightning_on(engine_id, request) {
            // Lightning distill (default): 4 steps / CFG-off. The distill is applied as an
            // adapter (resolve_wan_adapters), so a user `steps`/`guidanceScale` can't break it.
            return (Some(4), Some(1.0));
        }
        // Toggle off: native multi-step CFG. Honor an explicit user override, else `None` so the
        // engine's config.json A14B non-distill defaults (multi-step + CFG on) stand exactly.
        let steps = advanced_opt_u32(request, "steps");
        let guidance = advanced_opt_f32(request, "guidanceScale");
        return (steps, guidance);
    }
    // wan2_2_ti2v_5b (dense): user override wins, else the interim default; CFG left to the
    // engine (guide 5.0) unless the user disables it via `guidanceScale`.
    let steps = advanced_opt_u32(request, "steps").or(Some(WAN5B_INTERIM_STEPS));
    let guidance = advanced_opt_f32(request, "guidanceScale");
    (steps, guidance)
}

/// The lightx2v lightning step-distill recipe (sc-5684 / sc-5700): 8 steps, CFG off, scheduler shift 1.
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_STEPS: u32 = 8;
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_GUIDANCE: f32 = 1.0;
#[cfg(target_os = "macos")]
const SCAIL2_LIGHTNING_SHIFT: f32 = 1.0;

/// SCAIL-2 sampling recipe `(steps, guidance, scheduler_shift)`. When a lightx2v diff-patch
/// "lightning" LoRA is selected (`lightning`), apply the step-distill recipe so the toggle yields the
/// ~10× fewer-DiT-passes speedup: CFG off (guidance 1.0 → the engine short-circuits to a single DiT
/// forward per step) and scheduler shift 1.0 are the lightning invariants (forced), and the step count
/// defaults to 8 but honors an explicit user `advanced.steps` override. Without a lightning LoRA, return
/// all-`None` so the engine's quality defaults (40 steps, guide 5.0, shift 5.0) stand exactly as before
/// — this path is unchanged. The chosen knobs are recorded as `effective*` in [`scail2_raw_settings`]
/// so what actually ran is inspectable on the asset (mirrors [`wan_raw_settings`]).
#[cfg(target_os = "macos")]
fn scail2_sampling(
    request: &VideoRequest,
    lightning: bool,
) -> (Option<u32>, Option<f32>, Option<f32>) {
    if !lightning {
        return (None, None, None);
    }
    (
        advanced_opt_u32(request, "steps").or(Some(SCAIL2_LIGHTNING_STEPS)),
        Some(SCAIL2_LIGHTNING_GUIDANCE),
        Some(SCAIL2_LIGHTNING_SHIFT),
    )
}

/// `true` if any resolved adapter is a lightx2v diff-patch ("lightning") LoRA — the engine's own
/// detector (a file carrying full-rank `.diff`/`.diff_b` tensors), so the recipe keys off the actual
/// format, not a catalog id or filename. A file that can't be read is treated as non-lightning (the
/// engine surfaces the real load error downstream).
#[cfg(target_os = "macos")]
fn scail2_adapters_have_lightning(adapters: &[AdapterSpec]) -> bool {
    adapters
        .iter()
        .any(|a| runtime_macos::providers::scail2::has_diff_patch_keys(&a.path).unwrap_or(false))
}

/// In-place ComfyUI Wan2.2 A14B experts for the sc-10671 base lane (epic 10451 Phase 2c). When set on a
/// [`VideoGenInput`], [`generate_video`] builds the two experts from these files (key remap +
/// scaled-fp8 dequant, `runtime_cuda::providers::wan::load_from_comfyui_experts`) via the uncached bespoke load path
/// instead of the registry snapshot. The UMT5 TE + VAE are read in place too when `te_file` / `vae_file`
/// are set (sc-10909), else they come from `model_dir` (a resident Wan snapshot tier); the tiny
/// tokenizer always comes from `model_dir`. Read in place, never copied.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone)]
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
struct ComfyuiWanExperts {
    /// The high-noise expert file (ComfyUI `*_high_noise_*`), read in place → candle `transformer/`.
    high_file: PathBuf,
    /// The low-noise expert file (ComfyUI `*_low_noise_*`), read in place → candle `transformer_2/`.
    low_file: PathBuf,
    /// The UMT5-XXL text encoder (`umt5_xxl_fp8_e4m3fn_scaled`, companion scaled-fp8), read in place
    /// when present (sc-10909); `None` ⇒ the snapshot `text_encoder/`.
    te_file: Option<PathBuf>,
    /// The Wan VAE (`wan_2.1_vae.safetensors`, native WAN-VAE keys), read in place when present
    /// (sc-10909); `None` ⇒ the snapshot `vae/`.
    vae_file: Option<PathBuf>,
    /// I2V (channel-concat) vs T2V — selects the Wan config (`patch_embedding` in-channels differ).
    i2v: bool,
}

/// The resolved inputs for one video generation (engine load + request build), shared by
/// Wan (sc-3034) and LTX (sc-3035) — split out so the engine call is unit-testable on real
/// weights without the API/job plumbing. The LTX-only knobs (`video_mode` no_audio,
/// prompt-enhance) default off for Wan; the Wan-only `moe_expert` rides on `adapters`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct VideoGenInput {
    engine_id: &'static str,
    model_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    conditioning: Vec<Conditioning>,
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    steps: Option<u32>,
    guidance: Option<f32>,
    /// Flow-matching scheduler shift (`req.scheduler_shift`); `None` ⇒ the engine default. Set by the
    /// SCAIL-2 lightning recipe (shift 1.0, sc-5700); the other models leave it at the engine default.
    scheduler_shift: Option<f32>,
    /// Per-generation sampler / scheduler (epic 7114 P5, sc-7127). Left `None` by the handlers; the
    /// shared funnel [`generate_video`] reads them from the job's `advanced` block and N3-guards them
    /// against the resolved engine descriptor's advertised surface before they reach `req`. A video
    /// engine that does not advertise the curated sampler/scheduler axis (everything but the Wan
    /// fold-in + the SVD/LTX sampler-only outliers, until candle adoption) leaves these `None`.
    sampler: Option<String>,
    scheduler: Option<String>,
    seed: u64,
    /// Per-request control-clip conditioning scale (Wan-VACE `conditioning_scale`, sc-3441 /
    /// sc-3521); `None` ⇒ the engine default (1.0). Unused by the non-control paths.
    control_scale: Option<f32>,
    // LTX-only knobs (sc-3035); left at defaults by Wan + the other models.
    video_mode: Option<String>,
    enhance_prompt: bool,
    use_uncensored_enhancer: bool,
    enhance_max_tokens: Option<u32>,
    enhance_temperature: Option<f32>,
    // SVD-only micro-conditioning knobs (sc-3523); `None` on the other models.
    motion_bucket_id: Option<f32>,
    noise_aug_strength: Option<f32>,
    decode_chunk_size: Option<u32>,
    // SVD motion-conditioning fps, decoupled from the output `fps` (sc-3764); `None` elsewhere.
    conditioning_fps: Option<u32>,
    // SeedVR2 input pre-blur (sc-4816); `None` on the other models.
    softness: Option<f32>,
    // LTX-only external Gemma-3 text-encoder snapshot dir (sc-8827): rides `LoadSpec::text_encoder` so
    // the LTX provider locates its Gemma encoder from the spec instead of the process-global
    // `$LTX_GEMMA_DIR` env var (the old `set_var` seam was unsound on the multithreaded runtime,
    // F-025). `None` on every other model (they bundle their TE) and when no override resolves.
    text_encoder_dir: Option<PathBuf>,
    /// In-place ComfyUI Wan MoE experts (epic 10451 Phase 2c, sc-10671). `Some` ⇒ [`generate_video`]
    /// takes the bespoke uncached load path (`load_from_comfyui_experts`) instead of the registry
    /// snapshot; `None` on every other job.
    comfyui: Option<ComfyuiWanExperts>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Default for VideoGenInput {
    fn default() -> Self {
        Self {
            engine_id: "",
            model_dir: PathBuf::new(),
            quant: None,
            adapters: Vec::new(),
            conditioning: Vec::new(),
            prompt: String::new(),
            negative_prompt: None,
            width: 0,
            height: 0,
            frames: 0,
            fps: 0,
            steps: None,
            guidance: None,
            scheduler_shift: None,
            sampler: None,
            scheduler: None,
            seed: 0,
            control_scale: None,
            video_mode: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            text_encoder_dir: None,
            comfyui: None,
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_load_spec(input: &VideoGenInput) -> LoadSpec {
    LoadSpec {
        weights: WeightsSource::Dir(input.model_dir.clone()),
        quantize: input.quant,
        precision: Precision::Bf16,
        control: None,
        // MultiControlNet (sc-3378) is image-only; video providers ignore it.
        extra_controls: Vec::new(),
        ip_adapter: None,
        adapters: input.adapters.clone(),
        // PiD super-resolving decode (epic 7840) is an image-only latent-space swap; video
        // providers have no PiD backbone, so never request it.
        pid: None,
        // Video providers are not face-ID models — no identity sub-model weights.
        identity: None,
        // LTX's external Gemma-3 text encoder rides the spec (sc-8827); `None` ⇒ the provider's
        // `$LTX_GEMMA_DIR` / `<root>/text_encoder` fallback.
        text_encoder: input.text_encoder_dir.clone().map(WeightsSource::Dir),
        // Video providers have not wired sequential residency (sc-10821) — stays Resident.
        offload_policy: Default::default(),
    }
}

/// Run one generation to a [`DecodedVideo`] (RGB8 frames + fps + optional audio) against an already
/// loaded video generator, streaming denoise progress via `on_progress` and honoring `cancel`.
/// The engine fills the audio track (LTX) or leaves it `None` (Wan).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn run_loaded_video_generation(
    generator: &dyn Generator,
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let req = GenerationRequest {
        prompt: input.prompt,
        negative_prompt: input.negative_prompt,
        width: input.width,
        height: input.height,
        frames: Some(input.frames),
        fps: Some(input.fps),
        steps: input.steps,
        guidance: input.guidance,
        scheduler_shift: input.scheduler_shift,
        // Per-generation sampler / scheduler (sc-7127), already N3-guarded against the engine's
        // advertised surface in `generate_video`, so an unsupported name was dropped to `None` (the
        // engine default) before reaching here — `validate_request` only ever sees an advertised name.
        sampler: input.sampler,
        scheduler: input.scheduler,
        seed: Some(input.seed),
        conditioning: input.conditioning,
        control_scale: input.control_scale,
        video_mode: input.video_mode,
        enhance_prompt: input.enhance_prompt,
        use_uncensored_enhancer: input.use_uncensored_enhancer,
        enhance_max_tokens: input.enhance_max_tokens,
        enhance_temperature: input.enhance_temperature,
        motion_bucket_id: input.motion_bucket_id,
        noise_aug_strength: input.noise_aug_strength,
        decode_chunk_size: input.decode_chunk_size,
        conditioning_fps: input.conditioning_fps,
        softness: input.softness,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&req, on_progress)
        .map_err(|error| crate::classify_engine_error("video generation failed", error))?;
    match output {
        GenerationOutput::Video { frames, fps, audio } => Ok(DecodedVideo {
            frames: frames
                .into_iter()
                .map(|image| RgbFrame {
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels,
                })
                .collect(),
            fps,
            audio: audio.map(|track| AudioTrack {
                samples: track.samples,
                sample_rate: track.sample_rate,
                channels: track.channels,
            }),
        }),
        GenerationOutput::Images(_) => Err(WorkerError::Engine(
            "video model returned images, expected video frames".to_owned(),
        )),
    }
}

#[cfg(all(target_os = "macos", test))]
fn load_video_generation_for_tests(input: &VideoGenInput) -> WorkerResult<Box<dyn Generator>> {
    let spec = video_load_spec(input);
    crate::inference_runtime::load(input.engine_id, &spec)
        .map_err(|error| crate::classify_engine_error("video load failed", error))
}

#[cfg(all(target_os = "macos", test))]
fn run_video_generation(
    input: VideoGenInput,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<DecodedVideo> {
    let generator = load_video_generation_for_tests(&input)?;
    run_loaded_video_generation(generator.as_ref(), input, cancel, on_progress)
}

/// Forward-progress watchdog: if the engine emits no progress event (no denoise `Step`, no
/// `Decoding`) for this long — covering both the silent cold model-load phase and the gap
/// between steps — the generation is treated as wedged and the job is failed with a clear
/// error instead of heartbeating indefinitely. Tuned well above any legitimate single load or
/// step on the current video models; override via `SCENEWORKS_VIDEO_STALL_SECS` for an
/// unusually large/slow model or disk.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const VIDEO_STALL_TIMEOUT: Duration = Duration::from_secs(600);

/// Grace period granted after a stall is detected and engine cancellation is requested, before
/// the still-running blocking task is abandoned. A cooperative engine bails between steps well
/// within this window (the manual-cancel path proves it honors the flag); the abandon escape
/// only matters for a hard Metal wedge that never re-checks cancel, and keeps the watchdog from
/// itself re-hanging on the join.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const VIDEO_STALL_GRACE: Duration = Duration::from_secs(60);

/// The effective forward-progress stall timeout: `SCENEWORKS_VIDEO_STALL_SECS` (a positive
/// integer number of seconds) when set, else [`VIDEO_STALL_TIMEOUT`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_stall_timeout() -> Duration {
    parse_stall_timeout(std::env::var("SCENEWORKS_VIDEO_STALL_SECS").ok())
}

/// Parse the `SCENEWORKS_VIDEO_STALL_SECS` override (a positive integer number of seconds),
/// falling back to [`VIDEO_STALL_TIMEOUT`] when unset, blank, non-numeric, or zero.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn parse_stall_timeout(raw: Option<String>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or(VIDEO_STALL_TIMEOUT)
}

/// First-detection handling for the in-loop video cancel poller (sc-5516): trip the engine
/// `CancelFlag` and post a NON-terminal "Cancelling…" update (indeterminate progress bar —
/// `running` + fraction 0.0 renders the "Working" animation, not a backward jump). The terminal
/// `Canceled` is posted only after the blocking generation actually stops (see `generate_video`),
/// so the worker row — and therefore the next queued job — is not freed until the GPU is genuinely
/// idle, and the UI honestly shows "Cancelling…" until completion. Best-effort: a failed status
/// update here is non-fatal because the post-run terminal write is what ultimately frees the
/// worker. Mirrors the image path's `begin_image_cancel` (sc-5515).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn begin_video_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        video_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — finishing the current step…",
            None,
            backend,
        ),
    )
    .await;
}

/// The `(samplers, schedulers)` a video engine advertises (epic 7114), read from its registered
/// gen-core descriptor by engine id — the same `Capabilities` surface `validate_request` enforces, so
/// the N3 guard in [`generate_video`] mirrors the image lane's `model.descriptor.capabilities` read.
/// Empty (so every name N3-falls back to the engine default) when the id isn't registered on this
/// backend — e.g. a candle video engine before it adopts the unified framework.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_engine_sampling_surface(engine_id: &str) -> (Vec<&'static str>, Vec<&'static str>) {
    crate::inference_runtime::generators()
        .map(|reg| (reg.descriptor)())
        .find(|descriptor| descriptor.id == engine_id)
        .map(|descriptor| {
            (
                descriptor.capabilities.samplers,
                descriptor.capabilities.schedulers,
            )
        })
        .unwrap_or_default()
}

/// The effective video settings captured from the resolved [`VideoGenInput`] just
/// before it moves into the blocking generation task (epic 10402, sc-10418). Sourced
/// from the single resolved funnel so it reflects exactly what reached the engine —
/// the tier-resolved quant, the N3-guarded sampler/scheduler, and the recipe-resolved
/// steps/guidance — rather than re-deriving them from the sparse `advanced` payload
/// (the video engines' quant/guidance rules are engine-specific and would drift from
/// what actually ran). Captured like `log_engine_id`, before the move at the spawn.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct VideoSettingsSnapshot {
    quant: Option<Quant>,
    sampler: Option<String>,
    scheduler: Option<String>,
    scheduler_shift: Option<f32>,
    guidance: Option<f32>,
    width: u32,
    height: u32,
    seed: u64,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl VideoSettingsSnapshot {
    fn from_input(input: &VideoGenInput) -> Self {
        Self {
            quant: input.quant,
            sampler: input.sampler.clone(),
            scheduler: input.scheduler.clone(),
            scheduler_shift: input.scheduler_shift,
            guidance: input.guidance,
            width: input.width,
            height: input.height,
            seed: input.seed,
        }
    }
}

/// Normalized quant label + bit-width for a resolved video [`Quant`] (epic 10402,
/// sc-10418): `Q8` → ("q8", 8), `Q4` → ("q4", 4), `Nvfp4` → ("nvfp4", None), `None`
/// (dense/bf16) → ("bf16", None). Mirrors the image lane's `effective_quant_label` mapping
/// so the Stats charts group video and image runs on the same tier labels.
///
/// **MATCH THE VARIANT — never derive a tier label from [`Quant::bits`] (sc-11042, epic 11037 SC#5).**
/// This function used to `format!("q{bits}")` from `q.bits()`, which was correct only while `Quant` was
/// `{Q4, Q8}`. `Quant::Nvfp4::bits()` returns **4** (its E2M1 elements are 4-bit), so the bits-derived
/// form stamped an NVFP4 video render as `"q4"` + bits 4 in Stats telemetry — falsely reporting one
/// creative choice as another, exactly the tier aliasing this epic forbids. **The compiler could not
/// catch it**: reading `.bits()` raises no E0004 when a variant is added, so it compiled silently on the
/// `backend-candle` lane. The explicit arms below make any future `Quant` variant a hard compile error
/// here instead of a silent mislabel — do not collapse them back into a catch-all or a `bits()` map.
///
/// NVFP4 reports **no** bit count: it is ~4.5 EFFECTIVE bits/weight (E2M1 elements + FP8-E4M3 block
/// scales + an FP32 per-tensor scale), so `Some(4)` would re-introduce the same `q4` aliasing in the
/// `quant_bits` column that the label fix removes. `None` is the honest "no integer width applies" —
/// the same signal the dense/bf16 arm uses, and the same reason `flux2_comfyui_raw_settings` writes
/// `mlxQuantize: null` for this tier.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_quant_label(quant: Option<Quant>) -> (Option<String>, Option<u32>) {
    match quant {
        Some(Quant::Q4) => (Some("q4".to_owned()), Some(4)),
        Some(Quant::Q8) => (Some("q8".to_owned()), Some(8)),
        Some(Quant::Nvfp4) => (Some("nvfp4".to_owned()), None),
        None => (Some("bf16".to_owned()), None),
    }
}

/// Fold the effective video settings + model + observed step count into the
/// phase-timing metrics block for a finished video job (epic 10402, sc-10418). A
/// video job produces one output (sc-10426). Sampler/scheduler fall back to
/// "default" (engine-native) so the comparison charts always have a non-blank group,
/// mirroring the image lane. Guidance / scheduler-shift stay `None` when the engine's
/// own config default was used (not overridden) — an honest "not captured" rather
/// than a fabricated value the worker can't know without loading the engine config.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_video_metrics(
    mut metrics: GenerationMetrics,
    settings: &VideoSettingsSnapshot,
    model: Option<String>,
    effective_steps: Option<u32>,
) -> GenerationMetrics {
    let (quant_label, quant_bits) = video_quant_label(settings.quant);
    metrics.model = model;
    metrics.quant_label = quant_label;
    metrics.quant_bits = quant_bits;
    metrics.sampler = Some(
        settings
            .sampler
            .clone()
            .unwrap_or_else(|| "default".to_owned()),
    );
    metrics.scheduler = Some(
        settings
            .scheduler
            .clone()
            .unwrap_or_else(|| "default".to_owned()),
    );
    metrics.scheduler_shift = settings
        .scheduler_shift
        .and_then(|shift| serde_json::Number::from_f64(shift as f64));
    metrics.steps = effective_steps;
    metrics.image_count = Some(1); // one video output per job (sc-10426)
    metrics.guidance_scale = settings
        .guidance
        .and_then(|scale| serde_json::Number::from_f64(scale as f64));
    metrics.guidance_method = Some("cfg".to_owned());
    metrics.width = Some(settings.width);
    metrics.height = Some(settings.height);
    metrics.seed = Some(settings.seed as i64);
    metrics
}

/// Drive a `run_video_generation` on a blocking thread, forwarding its streamed denoise
/// progress to the async worker (Generating stage ~0.25..0.58) + polling cancel ~every 2s.
/// The shared blocking + mpsc + cancel plumbing for Wan and LTX. A forward-progress watchdog
/// ([`video_stall_timeout`]) fails a wedged job loudly rather than letting it look alive forever.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn generate_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    advanced: &JsonObject,
    input: VideoGenInput,
) -> WorkerResult<DecodedVideo> {
    generate_video_using(
        api,
        settings,
        job,
        backend,
        advanced,
        input,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_video`] with the engine loader supplied by the caller (sc-12318).
///
/// The `_using` half of the same pair [`crate::generator_cache::with_cached_generator`] already splits
/// one level down, and it exists for the same reason: with the loader threaded in, a test can drive an
/// async per-family arm (`generate_mochi`, `generate_candle_video`) against a stub `Generator` and
/// assert on the [`VideoGenInput`] that actually reached the engine. Without it, every decision an arm
/// makes inline — the frame lattice, the Mochi fit gate — is reachable only as the free function it
/// delegates to, never as the call itself.
///
/// SCOPE: the injected loader covers the registry **cached** path only. The in-place ComfyUI Wan MoE
/// branch builds its generator from per-file expert weights through `with_uncached_generator`, which has
/// no `(engine_id, spec)` key to load from, so it ignores `load_generator` and stays uncovered here.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn generate_video_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    backend: &str,
    advanced: &JsonObject,
    mut input: VideoGenInput,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<DecodedVideo> {
    // Per-generation sampler / scheduler axis for video (epic 7114 P5, sc-7127). The handlers leave
    // `input.sampler`/`scheduler` `None`; read them from the caller's already-parsed `advanced` block
    // here — the single funnel every Wan / LTX / SVD path passes through — and N3-guard each against
    // the resolved engine descriptor's advertised surface. A name the engine does not advertise (every
    // video engine but the Wan fold-in + the SVD/LTX sampler-only outliers, until candle adoption) is
    // dropped to the engine default + a `sampling_knob_unsupported` event, never a hard-fail. Taking
    // `advanced` by reference avoids re-parsing the whole payload into a throwaway VideoRequest per
    // generation (F-118).
    {
        let (raw_sampler, raw_scheduler, raw_shift) =
            crate::image_jobs::read_advanced_sampling_knobs(advanced);
        let (samplers, schedulers) = video_engine_sampling_surface(input.engine_id);
        input.sampler = crate::image_jobs::normalize_sampling_knob(
            raw_sampler,
            &samplers,
            "sampler",
            input.engine_id,
            &job.id,
            backend,
        );
        input.scheduler = crate::image_jobs::normalize_sampling_knob(
            raw_scheduler,
            &schedulers,
            "scheduler",
            input.engine_id,
            &job.id,
            backend,
        );
        // Schedule shift: only when the handler hasn't already forced it (the SCAIL-2 lightning recipe
        // sets shift 1.0), so the user knob can't clobber a model's required recipe. Parity with the
        // image lane's `advanced.schedulerShift` / `timestepShift` read.
        if input.scheduler_shift.is_none() {
            input.scheduler_shift = raw_shift;
        }
    }
    let cancel = CancelFlag::new();
    let stall_timeout = video_stall_timeout();
    let log_engine_id = input.engine_id;
    // Snapshot the effective settings before `input` moves into the blocking task
    // (sc-10418), so the completion-time metrics POST reports exactly what reached
    // the engine (resolved quant / sampler / scheduler / guidance / dims / seed).
    let video_settings = VideoSettingsSnapshot::from_input(&input);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Progress>(64);
    let blocking = {
        let cancel = cancel.clone();
        let spec = video_load_spec(&input);
        let engine_id = input.engine_id;
        // sc-10671: an in-place ComfyUI Wan MoE takes the bespoke **uncached** load path
        // (`load_from_comfyui_experts` — two experts read in place + remapped + dequant'd), which frees
        // any resident cached generator first; every other job takes the registry cached path. On the
        // non-candle/macOS build `comfyui` is always `None`, so only the cached path is compiled.
        let comfyui_load = input.comfyui.as_ref().map(|e| {
            (
                e.high_file.clone(),
                e.low_file.clone(),
                e.te_file.clone(),
                e.vae_file.clone(),
                input.model_dir.clone(),
                e.i2v,
            )
        });
        tokio::spawn(async move {
            let run = move |generator: &dyn Generator| {
                let mut on_progress = |progress: Progress| {
                    // A closed channel means the consumer loop returned early (POST failure /
                    // 409); trip the engine flag so the denoise bails instead of running unheard
                    // (sc-8804, F-003 — the swallowed-closed-channel leak).
                    if tx.blocking_send(progress).is_err() {
                        cancel.cancel();
                    }
                };
                run_loaded_video_generation(generator, input, &cancel, &mut on_progress)
            };
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let result = match comfyui_load {
                Some((high, low, te, vae, snapshot, i2v)) => {
                    crate::generator_cache::with_uncached_generator(
                        move || {
                            runtime_cuda::providers::wan::wan14b::load_from_comfyui_experts(
                                high, low, te, vae, snapshot, i2v,
                            )
                            .map_err(|error| {
                                crate::classify_engine_error("video load failed", error)
                            })
                        },
                        run,
                    )
                    .await
                }
                None => {
                    crate::generator_cache::with_cached_generator_using(
                        engine_id,
                        spec,
                        "video load failed",
                        load_generator,
                        run,
                    )
                    .await
                }
            };
            #[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
            let result = {
                let _ = comfyui_load;
                crate::generator_cache::with_cached_generator_using(
                    engine_id,
                    spec,
                    "video load failed",
                    load_generator,
                    run,
                )
                .await
            };
            result
        })
    };

    // Bind the blocking generation task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` in the loop below returns early on a transient POST failure or a 409
    // (stale-sweep reclaim); on that early return this guard trips the engine `CancelFlag` and
    // aborts the still-running denoise instead of leaving it burning GPU memory alongside the next
    // claimed job. The stall/abandon watchdog and final join reach through `guard.handle_mut()` /
    // `guard.into_handle()`. `cancel` is kept alongside (it's `Clone`) for the in-loop pollers.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut canceled = false;
    // Set when the watchdog (not the user) tripped, so the job is failed with a stall error
    // rather than reported as a clean user cancellation.
    let mut stalled = false;
    // Once a stall is detected we request engine cancel and wait at most `VIDEO_STALL_GRACE`
    // for the blocking task to unwind; past this deadline we abandon it (a hard Metal wedge)
    // so the watchdog never re-hangs on the join.
    let mut abandon_deadline: Option<Instant> = None;
    let mut abandoned = false;
    let mut last_cancel = Instant::now();
    // Time of the most recent progress event; the forward-progress watchdog fails the job if
    // this goes stale for `stall_timeout` (covers both the silent load phase and step-to-step).
    let mut last_progress = Instant::now();
    // Interval arm so the cold model-load phase (crate::inference_runtime::load emits no progress)
    // still heartbeats and polls cancel, instead of looking dead to the API's
    // staleness check until the first denoise step (sc-4276 / F-MLXW-12; mirrors
    // the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(crate::progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Per-phase wall-clock (epic 10402, sc-10405): load = start→first Step,
    // sample = Step→Decoding, decode = Decoding→engine-return (video emits no
    // per-frame decode-done event). Posted best-effort at clean completion.
    let mut phase_timer = crate::job_metrics::PhaseTimer::new(Instant::now());
    // Effective denoise step count from the Step event (sc-10406).
    let mut video_effective_steps: Option<u32> = None;
    // Run the progress loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003). The stall/
    // abandon watchdog inside the loop still handles the hard-wedge case via `abandoned`.
    let loop_result: WorkerResult<()> = async {
        loop {
            tokio::select! {
                maybe_progress = rx.recv() => {
                    let Some(progress) = maybe_progress else {
                        break;
                    };
                    last_progress = Instant::now(); // forward progress — reset the stall watchdog.
                    if canceled {
                        continue; // drain so the blocking sender never blocks.
                    }
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll so a quit stops the gen at this frame step, matching a user cancel.
                    if shutdown_requested() {
                        begin_video_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                        continue;
                    }
                    if last_cancel.elapsed() >= Duration::from_secs(2) {
                        last_cancel = Instant::now();
                        if cancel_requested_peek(api, &job.id).await {
                            begin_video_cancel(api, &job.id, &cancel, backend).await;
                            canceled = true;
                            continue;
                        }
                        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    }
                    // Phase-boundary capture (sc-10405), borrowing so `progress` is
                    // still owned by the fraction/message match below.
                    match &progress {
                        Progress::Step { total, .. } => {
                            phase_timer.mark_sample_step(Instant::now());
                            if video_effective_steps.is_none() {
                                video_effective_steps = Some(*total);
                            }
                        }
                        Progress::Decoding => phase_timer.mark_decoding(Instant::now()),
                        Progress::Loading(_) => {}
                    }
                    let (status, stage, fraction, message) = match progress {
                        Progress::Step { current, total } => (
                            JobStatus::Running,
                            ProgressStage::Generating,
                            0.25 + 0.30 * (current as f64 / total.max(1) as f64),
                            format!("Generating frames — step {current}/{total}."),
                        ),
                        Progress::Decoding => (
                            JobStatus::Running,
                            ProgressStage::Generating,
                            0.58,
                            "Decoding frames.".to_owned(),
                        ),
                        Progress::Loading(phase) => (
                            JobStatus::LoadingModel,
                            ProgressStage::LoadingModel,
                            0.24,
                            match phase {
                                LoadPhase::TextEncoder => "Loading text encoder.",
                                LoadPhase::Renderer => "Loading render components.",
                            }
                            .to_owned(),
                        ),
                    };
                    update_job(
                        api,
                        &job.id,
                        video_progress(
                            status,
                            stage,
                            fraction,
                            &message,
                            None,
                            backend,
                        ),
                    )
                    .await?;
                }
                _ = interval.tick() => {
                    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    // sc-9618: honor a process shutdown on every tick (local flag read, unthrottled).
                    if !canceled && (shutdown_requested()
                        || (last_cancel.elapsed() >= Duration::from_secs(2) && {
                            last_cancel = Instant::now();
                            cancel_requested_peek(api, &job.id).await
                        }))
                    {
                        begin_video_cancel(api, &job.id, &cancel, backend).await;
                        canceled = true;
                    }
                    // Forward-progress watchdog: a wedged engine keeps this async loop heartbeating
                    // (the block runs on a separate thread), so the API sees a healthy job forever.
                    // If no progress has arrived for `stall_timeout`, request engine cancel and start
                    // the abandon countdown.
                    if !canceled && last_progress.elapsed() >= stall_timeout {
                        tracing::warn!(
                            event = "rust_worker_video_stalled",
                            jobId = %job.id,
                            engine = %log_engine_id,
                            stallSeconds = stall_timeout.as_secs(),
                            "no progress within the stall window — requesting engine cancel"
                        );
                        cancel.cancel();
                        canceled = true;
                        stalled = true;
                        abandon_deadline = Some(Instant::now() + VIDEO_STALL_GRACE);
                    }
                    if let Some(deadline) = abandon_deadline {
                        if Instant::now() >= deadline {
                            abandoned = true;
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    if abandoned {
        // The engine never honored the cancel flag within the grace window (a hard Metal wedge).
        // Detach the still-running blocking task instead of awaiting it — awaiting would re-hang
        // the very failure path this watchdog exists to break. The thread (and the GPU it holds)
        // leaks until the worker is restarted by the supervisor.
        tracing::error!(
            event = "rust_worker_video_abandoned",
            jobId = %job.id,
            engine = %log_engine_id,
            graceSeconds = VIDEO_STALL_GRACE.as_secs(),
            "engine did not respond to cancellation within the grace window — exiting the worker \
             so the supervisor can recover the wedged GPU task"
        );
        guard.handle_mut().abort();
        std::process::exit(70);
    }
    // Loop exited cleanly — reclaim the handle (disarming the drop-guard) and join the finished task.
    let result = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("video task join", error))?;
    if stalled {
        return Err(WorkerError::Engine(format!(
            "Video generation stalled: no progress for {}s. The job was canceled.",
            stall_timeout.as_secs()
        )));
    }
    if canceled {
        // Reached only on a genuine user cancel — the stall/abandon watchdog returns above.
        // Generation has actually stopped now, so post the TERMINAL Canceled here (not at the
        // earlier cancel poll, which only tripped the flag + showed "Cancelling…"). This terminal
        // write is what frees the worker row (`jobs_store::update_job_progress`), so it lands as
        // the worker returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free (sc-5516; mirrors the image path sc-5515).
        update_job(
            api,
            &job.id,
            video_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                CANCEL_MESSAGE,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    // Post the video metrics (epic 10402): the resolved effective settings
    // (quant / sampler / scheduler / guidance / dims / seed, sc-10418) + model +
    // effective steps (sc-10406) folded with the per-phase timing (sc-10405).
    // into_metrics closes the decode span still open at completion (video emits no
    // decode-done event). Best-effort; coalesce-merges with the S2 hardware block
    // server-side.
    let timing = phase_timer.into_metrics(Instant::now()).unwrap_or_default();
    let model = job
        .payload
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_owned);
    let metrics = build_video_metrics(timing, &video_settings, model, video_effective_steps);
    crate::job_metrics::post_generation_metrics(api, &job.id, &metrics).await;
    result
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) video lane (sc-5097, epic 5095). The candle wan/ltx providers serve a narrow
// **txt2video-only** first slice (no image/VACE conditioning, audio, LoRA, or quant). This is the
// video sibling of the candle image lane (image_jobs.rs `generate_candle_stream`): it builds a
// `VideoGenInput` and drives the SAME neutral streaming harness (`generate_video` →
// `run_loaded_video_generation` → the registry-resolved candle generator), reusing the shared
// encode/mux/poster path. Reached only when `backend_candle_enabled` (default off).
// ---------------------------------------------------------------------------

/// Per-asset adapter ids for the candle video engines (`candle_<family>`), the candle siblings of
/// the MLX `mlx_wan` / `mlx_ltx` labels (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_ADAPTER: &str = "candle_wan";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_ADAPTER: &str = "candle_ltx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SVD_ADAPTER: &str = "candle_svd";
/// Adapter id recorded on a candle Mochi 1 asset (epic 1788 / sc-11992). Distinct from the MLX
/// [`MOCHI_ADAPTER`] so the sidecar records which backend rendered the clip, matching the
/// `candle_wan`/`mlx_wan` and `candle_ltx`/`mlx_ltx` pairs.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_MOCHI_ADAPTER: &str = "candle_mochi";

/// Default HuggingFace repos the candle video providers load (overridable via the manifest `repo`).
/// The candle wan providers read a Wan2.2 diffusers snapshot — the TI2V-5B, or the T2V-A14B /
/// I2V-A14B 14B MoE (sc-5175); ltx reads the LTX-2.3 checkpoint plus a separate Gemma-3-12B encoder
/// snapshot (`LTX_GEMMA_DIR`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_5B_REPO: &str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_T2V_14B_REPO: &str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_I2V_14B_REPO: &str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_REPO: &str = "Lightricks/LTX-2.3";
// The `ltx_2_3_eros` weights repo (sc-5495): a full dense LTX-2.3 fine-tune (the candle provider
// loads its bf16 single-file checkpoint like the base; same architecture, same Gemma encoder).
// The pinned checkpoint version is manifest-driven (`downloads[].files` / `mlx.convertSourceFile`),
// so only that one file is fetched even though the repo also ships older + fp8 variants.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_EROS_REPO: &str = "TenStrip/LTX2.3-10Eros";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_LTX_GEMMA_REPO: &str = "google/gemma-3-12b-it";

/// Per-asset adapter id for the candle Wan-VACE controllable-video lane (sc-5494) — the candle sibling
/// of the MLX `mlx_wan_vace` label.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_VACE_ADAPTER: &str = "candle_wan_vace";

/// The diffusers Wan2.1-VACE-14B snapshot the candle `wan_vace` provider reads (`transformer/` +
/// `text_encoder/` + `vae/` + `tokenizer/`). Overridable via `SCENEWORKS_CANDLE_WAN_VACE_DIR`. The 14B
/// repo matches the provider's `WanVaceConfig::vace_14b` dims (dim 5120, 40 layers).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN_VACE_REPO: &str = "Wan-AI/Wan2.1-VACE-14B-diffusers";

/// SceneWorks video model id → candle registry engine id, or `None` for an id the candle video lane
/// does not serve. Note ltx maps to `ltx_2_3_distilled` (the candle provider's id), not the MLX
/// `ltx_2_3`. Covers the base txt2video ids (5B + ltx) plus the Wan2.2 **14B** dual-expert MoE pair
/// (sc-5174 / sc-5175): `wan_2_2_t2v_14b` (text→video) and `wan_2_2_i2v_14b` (image→video), plus `svd`
/// (image→video, sc-5493 / epic 5481). `ltx_2_3_eros` (sc-5495) maps to the same `ltx_2_3_distilled`
/// engine as the base — it's a full dense LTX-2.3 fine-tune — differing only in the weights repo.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
        "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
        // Base + eros both load the one candle LTX-2.3 engine (the eros merge is a full dense LTX-2.3
        // checkpoint, sc-5495); they differ only in the weights repo (see `candle_video_repo`).
        "ltx_2_3" | "ltx_2_3_eros" => Some("ltx_2_3_distilled"),
        // SVD-XT image→video (sc-5493 / epic 5481): the candle-gen-svd provider's `svd_xt` engine.
        "svd" => Some("svd_xt"),
        // Mochi 1 (epic 1788 / sc-11992). The sceneworks id IS the engine id: `candle-gen-mochi`
        // registers the SAME `MODEL_ID = "mochi_1"` as `mlx-gen-mochi` (no `_distilled`-style split),
        // and its descriptor is `mac_only: false` — the off-Mac lane is real and CUDA-validated on
        // Blackwell (sc-11990), ingesting the same hosted mlx-affine tiers.
        //
        // Load-bearing: without this arm `is_candle_video_engine` is false, `resolve_candle_video_route`
        // never reaches the generic arm, and a Windows Mochi job falls to `CandleVideoRoute::Stub` —
        // handing the user a PROCEDURAL FAKE VIDEO instead of an error. B1 already routed Windows
        // (`candle_video_routed = true`), so that promise must be served here.
        "mochi_1" => Some("mochi_1"),
        _ => None,
    }
}

/// Whether `model` is served by the candle video lane (sc-5097).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_candle_video_engine(model: &str) -> bool {
    candle_video_engine_id(model).is_some()
}

/// The adapter id recorded on a candle video asset. Every engine that is NOT wan MUST have an
/// explicit arm: the `_` fall-through is the Wan default, so a missing arm silently stamps a
/// different model's provenance onto the asset sidecar + telemetry (sc-11992 — `mochi_1` landed in
/// `_` and was labelled a Wan adapter).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_adapter_label(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_ADAPTER,
        "svd_xt" => CANDLE_SVD_ADAPTER,
        "mochi_1" => CANDLE_MOCHI_ADAPTER,
        _ => CANDLE_WAN_ADAPTER,
    }
}

/// The candle default weights repo for a video engine id (the per-variant Wan2.2 diffusers snapshot,
/// or the LTX-2.3 checkpoint). Used when the manifest entry omits `repo`. Like
/// [`candle_video_adapter_label`], the `_` arm is the Wan default — a non-wan engine without an
/// explicit arm inherits Wan's repo (sc-11992).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_default_repo(engine_id: &str) -> &'static str {
    match engine_id {
        "ltx_2_3_distilled" => CANDLE_LTX_REPO,
        "svd_xt" => SVD_REPO,
        "wan2_2_t2v_14b" => CANDLE_WAN_T2V_14B_REPO,
        "wan2_2_i2v_14b" => CANDLE_WAN_I2V_14B_REPO,
        // Mochi 1 (epic 1788 / sc-11992): ONE repo serves BOTH backends — candle ingests the same
        // mlx-affine tiers via A6's `.scales`-detect seam, and `SceneWorks/mochi-1-candle` was never
        // published. No manifest entry carries a top-level `repo`, so this default is what the candle
        // lane actually resolves.
        "mochi_1" => MOCHI_REPO,
        // `wan2_2_ti2v_5b` (and any other wan id) → the 5B TI2V snapshot.
        _ => CANDLE_WAN_5B_REPO,
    }
}

/// The candle weights repo for a video engine: the manifest `repo` wins, else — for the Wan
/// quant-matrix models whose per-tier candle repos live in `downloads[]` with no top-level `repo`
/// (`SceneWorks/wan2.2-*-candle`, sc-10027) — the platform-appropriate candle tier repo matching the
/// requested tier (default q4), else `ltx_2_3_eros`'s own fine-tune repo, else the candle default repo.
///
/// Without the `downloads[]` resolution the Windows/Linux Wan-14B lane fell back to the DENSE
/// `Wan-AI/*-Diffusers` default — a different (bf16, ~72 GB) repo that the packed-tier install never
/// fetches — so a candle Wan-14B job errored "snapshot not found" even with the q4 tier present (sc-10539).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_repo(request: &VideoRequest, engine_id: &str) -> String {
    if let Some(repo) = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return repo.to_owned();
    }
    if let Some(repo) = candle_wan_tier_repo_from_downloads(request, engine_id) {
        return repo;
    }
    if request.model == "ltx_2_3_eros" {
        CANDLE_LTX_EROS_REPO.to_owned()
    } else {
        candle_video_default_repo(engine_id).to_owned()
    }
}

/// The candle Wan tier repo from the manifest `downloads[]` for THIS platform (sc-10539). The Wan
/// quant-matrix (sc-10027) hosts each candle tier as a per-`variant` download entry — `q4`/`q8` in the
/// packed `SceneWorks/wan2.2-*-candle` repo, `bf16` in the dense `Wan-AI/*-Diffusers` repo — rather than
/// a single top-level `repo`, so `candle_video_repo` must consult them. Picks the repo for the highest-
/// preference tier present for this OS (default **q8-first** when the manifest lists it for the platform,
/// clamping to q4 otherwise — epic 10721 / sc-10726 — mirroring [`candle_wan_tier_subdir`] so the
/// resolved repo is the one whose tier subdir the loader then selects). `None` for non-Wan engines or a
/// manifest without matching platform downloads.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_repo_from_downloads(request: &VideoRequest, engine_id: &str) -> Option<String> {
    if !engine_id.starts_with("wan2_2") {
        return None;
    }
    let downloads = request.model_manifest_entry.get("downloads")?.as_array()?;
    let platform = if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    let order: &[&str] = match candle_wan_quant_bits(request) {
        None => &["q8", "q4"],
        Some(bits) if bits <= 0 => &["bf16", "q8", "q4"],
        Some(bits) if bits >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    };
    order.iter().find_map(|&tier| {
        downloads.iter().find_map(|download| {
            if download.get("variant").and_then(Value::as_str) != Some(tier) {
                return None;
            }
            let on_platform = match download.get("platforms").and_then(Value::as_array) {
                Some(platforms) => platforms
                    .iter()
                    .any(|value| value.as_str() == Some(platform)),
                None => true,
            };
            if !on_platform {
                return None;
            }
            download
                .get("repo")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
    })
}

/// Resolve the candle weights snapshot dir for `repo`. Errors loudly (no procedural-stub fallback)
/// when the snapshot is absent, so a missing model surfaces a re-download error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_snapshot_dir(settings: &Settings, repo: &str) -> WorkerResult<PathBuf> {
    huggingface_snapshot_dir(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "candle video weights snapshot not found for {repo}"
        ))
    })
}

/// (sc-10027) The `advanced.mlxQuantize` bits for a candle wan tier-select — a number or numeric string;
/// `None` when unset.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// (sc-10027) Whether `dir` is a complete candle wan tier — a diffusers-layout snapshot with the DiT
/// transformer(s), the T5 encoder, the VAE and the tokenizer. The A14B MoE carries a second expert
/// (`transformer_2/`); the TI2V-5B is a single transformer.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_complete(dir: &Path, a14b: bool) -> bool {
    let has = |sub: &str| dir.join(sub).is_dir();
    has("transformer")
        && has("text_encoder")
        && has("vae")
        && has("tokenizer")
        && (!a14b || has("transformer_2"))
}

/// (sc-10027) Resolve the candle wan quant tier subdir (`q4`/`q8`/`bf16`) + its quant marker under a
/// `SceneWorks/wan2.2-*-candle` snapshot `root`, per `advanced.mlxQuantize` (default **q8** when
/// installed, clamping to q4 — epic 10721 / sc-10726 — falling back through the tier order), or `None`
/// for a non-wan engine or a flat repo with no tier subdirs (e.g. the
/// dense `Wan-AI/*-Diffusers` fallback, which loads as-is). A resolved subdir **is** the diffusers-layout
/// snapshot the sc-10025 packed-detect seam loads — the quant is baked into the tier, so the `Quant`
/// returned is a tier-select marker (`spec.quantize` is a no-op on the candle wan load). Candle analog of
/// the macOS `wan_tier_subdir` / `resolve_wan_tier_dir_and_quant` (sc-9079).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_tier_subdir(
    root: &Path,
    engine_id: &str,
    request: &VideoRequest,
) -> Option<(PathBuf, Option<Quant>)> {
    if !engine_id.starts_with("wan2_2") {
        return None;
    }
    // Tier preference by requested bits (mirrors the macOS `wan_tier_order`): no explicit pick → q8
    // (the app-wide default, clamped to the installed tier, so q4-only installs still resolve q4);
    // explicit ≤4 → q4. `bf16` stays out of the default order (never auto-loaded on a default job).
    let order: &[&str] = match candle_wan_quant_bits(request) {
        None => &["q8", "q4"],
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    };
    let a14b = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    order.iter().find_map(|&tier| {
        let dir = root.join(tier);
        candle_wan_tier_complete(&dir, a14b).then(|| {
            let quant = match tier {
                "q4" => Some(Quant::Q4),
                "q8" => Some(Quant::Q8),
                _ => None, // bf16
            };
            (dir, quant)
        })
    })
}

/// Resolve the Gemma-3-12B encoder snapshot dir for the candle LTX provider (sc-8827). Returns the
/// HF-cache snapshot path so the caller can thread it onto `LoadSpec::text_encoder` — no more mutating
/// the process-global `$LTX_GEMMA_DIR` at job time on the multithreaded runtime (the old `set_var`
/// seam was unsound, F-025). Honors an explicit operator `$LTX_GEMMA_DIR` (returns `None` so the
/// provider reads the env override itself). Best-effort: if the Gemma snapshot isn't in the HF cache
/// we return `None` so the provider tries its `<root>/text_encoder` fallback and emits its own clear
/// "set LTX_GEMMA_DIR …" error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_ltx_gemma_dir(settings: &Settings) -> Option<PathBuf> {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return None; // honor an explicit operator override (the provider reads the env var).
    }
    huggingface_snapshot_dir(&settings.data_dir, CANDLE_LTX_GEMMA_REPO)
}

/// Raw-settings recorded on a candle video asset (mirrors `wan_raw_settings`, trimmed to the
/// txt2video surface): the request `advanced` knobs plus the real-inference markers.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_video_raw_settings(request: &VideoRequest, repo: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Per-request conditioning for a candle video generation. The Wan2.2 **14B I2V** engine
/// (sc-5174 / sc-5175) and **SVD-XT** (sc-5493) are conditioned: each requires a source image, loaded
/// to a single [`Conditioning::Reference`] — for Wan, the channel-concat first frame the provider
/// VAE-encodes into its `y` (`in_dim=36`); for SVD, the CLIP-encoded + noise-aug VAE-encoded driving
/// frame. The candle analog of the MLX i2v / SVD conditioning ([`resolve_wan_conditioning`]).
/// Every other candle video engine (5B, T2V-14B, ltx) is txt2video-only, so this returns an empty set.
/// The router's `video_request_candle_eligible` already guarantees the i2v shape carries a source and
/// the txt2video ids do not, but the source is required here too so a mis-routed job fails clearly.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_video_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &str,
) -> WorkerResult<Vec<Conditioning>> {
    // The Wan2.2 14B I2V engine (sc-5175) and SVD-XT (sc-5493) condition on a single source image; every
    // other candle video engine (5B, T2V-14B, ltx) is txt2video-only (empty conditioning).
    if engine_id != "wan2_2_i2v_14b" && engine_id != "svd_xt" {
        return Ok(Vec::new());
    }
    let asset_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{engine_id}: image-to-video requires a source image (sourceAssetId)."
            ))
        })?;
    let image = crate::image_jobs::load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    // Pre-fit the source to the output W×H by the chosen crop/pad mode (sc-6139), the
    // same as the macOS LTX/Wan paths, so the Windows/CUDA candle I2V engine conditions
    // on an undistorted frame instead of an internal stretch.
    let image = crate::image_jobs::fit_engine_image(
        image,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    Ok(vec![Conditioning::Reference {
        image,
        strength: None,
    }])
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) Wan Lightning toggle + adapter resolution (sc-10138) — the off-Mac analog of
// the macOS `wan_lightning_on` / `wan_sampling` / `ensure_wan_lightning_present` / `resolve_wan_adapters`
// (all `#[cfg(target_os = "macos")]`). The candle Wan engine now ACCEPTS adapters on a packed q4/q8 tier
// via the additive branch (candle-gen sc-10094/10095, epic 10043), so off-Mac the A14B can get its 4-step
// distill through the Lightning toggle and user LoRAs apply on the candle Wan video path. These are
// candle-lane copies (the macOS lane keeps its own), reusing the backend-neutral helpers `lora_scale` /
// `resolve_lora_file` / `crate::image_jobs::{lora_path, classify_adapter}` / `MAX_JOB_LORAS` /
// `advanced_opt_*` / `huggingface_snapshot_dir` / `ensure_hf_files_cached`.
// ---------------------------------------------------------------------------

/// (sc-10138) The interim step default for the dense candle TI2V-5B (no distill LoRA exists yet) — the
/// candle analog of the macOS `WAN5B_INTERIM_STEPS`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_WAN5B_INTERIM_STEPS: u32 = 20;

/// (sc-10138) `true` if the A14B MoE 4-step Lightning distill is engaged for this candle request — the
/// candle analog of `wan_lightning_on`. A **default-on toggle** (`advanced.lightning`): absent or `true`
/// opts in, a strict-bool `false` opts out (native multi-step CFG). Only the two A14B MoE models bake
/// Lightning; every other engine returns `false`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_lightning_on(engine_id: &str, request: &VideoRequest) -> bool {
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    if !is_moe {
        return false;
    }
    request
        .advanced
        .get("lightning")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

/// (sc-10138) Per-model sampling for the candle Wan path — the candle analog of `wan_sampling`. On the
/// A14B MoE models the recipe is conditional on the Lightning toggle: on (default) → 4 steps / CFG-off
/// (the distill rides an adapter, so a user override can't break it); off → native multi-step CFG
/// (honor an explicit user `steps`/`guidanceScale`, else `None` so the engine's config defaults stand).
/// The dense TI2V-5B has no distill: user override wins, else the interim default.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_sampling(engine_id: &str, request: &VideoRequest) -> (Option<u32>, Option<f32>) {
    if engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b" {
        if candle_wan_lightning_on(engine_id, request) {
            return (Some(4), Some(1.0));
        }
        return (
            advanced_opt_u32(request, "steps"),
            advanced_opt_f32(request, "guidanceScale"),
        );
    }
    (
        advanced_opt_u32(request, "steps").or(Some(CANDLE_WAN5B_INTERIM_STEPS)),
        advanced_opt_f32(request, "guidanceScale"),
    )
}

/// (sc-10138) The `.low_noise.safetensors` sibling of a Wan A14B MoE high-noise LoRA file, or `None`
/// when the file is not the high-noise half of a pair — the candle analog of `wan_moe_low_noise_sibling`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_wan_moe_low_noise_sibling(primary: &Path) -> Option<PathBuf> {
    const HIGH: &str = ".high_noise.safetensors";
    let name = primary.file_name()?.to_str()?;
    if !name.to_ascii_lowercase().ends_with(HIGH) {
        return None;
    }
    let stem = &name[..name.len() - HIGH.len()];
    let sibling = primary.with_file_name(format!("{stem}.low_noise.safetensors"));
    sibling.is_file().then_some(sibling)
}

/// (sc-10138) The per-architecture 4-step Lightning distill LoRA pair (high/low) for an A14B MoE model
/// (`lightx2v/Wan2.2-Lightning`, rank-64 Seko) — the candle analog of `resolve_lightning_loras`. T2V-A14B
/// (V1.1) and I2V-A14B (V1) ship distinct, NOT cross-compatible LoRAs (sc-4997).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_resolve_lightning_loras(
    settings: &Settings,
    engine_id: &str,
) -> WorkerResult<(PathBuf, PathBuf)> {
    let snapshot = crate::model_jobs::huggingface_snapshot_dir(
        &settings.data_dir,
        "lightx2v/Wan2.2-Lightning",
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{engine_id}: the Lightning distill LoRA (lightx2v/Wan2.2-Lightning) is not \
                     downloaded — fetch it via the model manager"
        ))
    })?;
    let base = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "{other}: no Lightning distill LoRA — only the A14B MoE models bake Lightning"
            )))
        }
    };
    let high = snapshot.join(base).join("high_noise_model.safetensors");
    let low = snapshot.join(base).join("low_noise_model.safetensors");
    for file in [&high, &low] {
        if !file.is_file() {
            return Err(WorkerError::InvalidPayload(format!(
                "{engine_id}: Lightning LoRA file missing: {}",
                file.display()
            )));
        }
    }
    Ok((high, low))
}

/// (sc-10138) On-demand fetch of the A14B Lightning distill pair for the candle lane — the analog of
/// `ensure_wan_lightning_present`. Self-heals a worker that installed the tiers before the Lightning
/// `coRequisite`. No-op when the toggle is off, for a non-A14B engine, or when the pair is cached. A
/// pair still missing after the fetch makes resolve surface the clear "fetch it via the model manager" error.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn candle_ensure_wan_lightning_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<()> {
    if !candle_wan_lightning_on(engine_id, request) {
        return Ok(());
    }
    let subdir = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        _ => return Ok(()),
    };
    const REPO: &str = "lightx2v/Wan2.2-Lightning";
    if let Some(snapshot) = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, REPO) {
        let base = snapshot.join(subdir);
        if base.join("high_noise_model.safetensors").is_file()
            && base.join("low_noise_model.safetensors").is_file()
        {
            return Ok(());
        }
    }
    let files = vec![
        format!("{subdir}/high_noise_model.safetensors"),
        format!("{subdir}/low_noise_model.safetensors"),
    ];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        REPO,
        WAN_LIGHTNING_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// (sc-10138) Tag an A14B MoE Lightning/user LoRA to a specific expert — the candle analog of
/// `moe_adapter`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_moe_adapter(
    path: PathBuf,
    scale: f32,
    kind: gen_core::AdapterKind,
    expert: gen_core::MoeExpert,
) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind,
        pass_scales: None,
        moe_expert: Some(expert),
    }
}

/// (sc-10138) Build the adapter specs for a candle Wan generation — the candle analog of
/// `resolve_wan_adapters`. The Lightning distill pair (both A14B MoE models, tagged high/low, only when
/// the toggle is on) followed by the user LoRAs. On the MoE models a user `*.high_noise.safetensors` with
/// a `.low_noise` sibling tags high→High / low→Low; a single-file LoRA is shared (both experts on MoE,
/// the single model on the 5B). The candle Wan engine applies these additively on a packed q4/q8 tier
/// (sc-10094/10095) or folds them on a dense tier.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_resolve_wan_adapters(
    settings: &Settings,
    request: &VideoRequest,
    engine_id: &str,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let is_moe = engine_id == "wan2_2_t2v_14b" || engine_id == "wan2_2_i2v_14b";
    let mut specs: Vec<AdapterSpec> = Vec::new();

    // Lightning distill (both A14B MoE models): 4-step, per-expert at strength 1.0, added only when the
    // toggle is on. When off, the native multi-step CFG recipe runs with no Lightning adapter.
    if is_moe && candle_wan_lightning_on(engine_id, request) {
        let (high, low) = candle_resolve_lightning_loras(settings, engine_id)?;
        specs.push(candle_moe_adapter(
            high,
            1.0,
            gen_core::AdapterKind::Lora,
            gen_core::MoeExpert::High,
        ));
        specs.push(candle_moe_adapter(
            low,
            1.0,
            gen_core::AdapterKind::Lora,
            gen_core::MoeExpert::Low,
        ));
    }

    for lora in &request.loras {
        let path = crate::image_jobs::lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = crate::image_jobs::classify_adapter(&file)?;
        let scale = lora_scale(lora);
        match (is_moe, candle_wan_moe_low_noise_sibling(&file)) {
            (true, Some(low)) => {
                let low_kind = crate::image_jobs::classify_adapter(&low)?;
                specs.push(candle_moe_adapter(
                    file,
                    scale,
                    kind,
                    gen_core::MoeExpert::High,
                ));
                specs.push(candle_moe_adapter(
                    low,
                    scale,
                    low_kind,
                    gen_core::MoeExpert::Low,
                ));
            }
            _ => {
                specs.push(AdapterSpec {
                    path: file,
                    scale,
                    kind,
                    pass_scales: None,
                    moe_expert: None,
                });
            }
        }
    }
    Ok(specs)
}

/// The candle Mochi pre-flight's gated result (sc-12306): the tier's baked-in quant marker, obtainable
/// ONLY by passing the VRAM fit gate.
///
/// Bundling the marker into the gated return is deliberate, and mirrors the MLX lane's [`MochiPreflight`]
/// — which adopted the shape after a review mutation that deleted a free-standing `mochi_fit_check(...)?`
/// call still compiled and still rendered, silently un-gating the lane. With the marker only obtainable
/// here, the generation arm cannot reach a quant on a path that skipped the gate.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MochiVramPreflight {
    quant: Option<Quant>,
}

/// Live pre-flight Mochi VRAM admission check for the candle lane (sc-12306) — the seam
/// [`generate_candle_video`] calls before the load + 64-step denoise.
///
/// Sums the on-disk bytes the load will hold resident via the SHARED [`crate::mlx_fit_gate::mochi_resident_bytes`]
/// (the tier dir's AsymmDiT plus the `text_encoder/` + `vae/` siblings from its parent): despite the
/// module name that scan describes the hosted repo layout, which is one repo serving both lanes, and
/// summing only the tier dir would miss the ~9.7 GiB T5-XXL + VAE — over half the resident footprint.
///
/// `budget` arrives resolved so this stays free of the GPU probe and is unit-testable without CUDA. No
/// budget signal ⇒ admits. `Err` is the actionable pre-denoise rejection.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn mochi_vram_preflight(
    model_label: &str,
    tier_dir: &Path,
    frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<MochiVramPreflight> {
    match crate::vram_gate::mochi_fit_error(
        model_label,
        crate::mlx_fit_gate::mochi_resident_bytes(tier_dir),
        frames,
        width,
        height,
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(MochiVramPreflight {
            quant: mochi_tier_quant(tier_dir),
        }),
    }
}

/// Live pre-flight Wan VRAM admission check for the candle lane (sc-12344) — the non-Mochi half of this
/// lane's gate, and the seam [`generate_candle_video`] calls before the load + denoise.
///
/// Budgets on the on-disk bytes of the components the Wan loader actually reads
/// ([`crate::vram_gate::wan_weight_bytes`]), which for Wan ARE the resident set: both A14B MoE experts
/// stay co-resident and every file in each component dir loads. `ltx`/`svd` read `0` there and are
/// admitted unchanged — their on-disk bytes are not their loaded set, so a floor would wall-reject
/// working cards (the exemption is recorded on `vram_gate::wan_weight_components`).
///
/// **Takes and returns `model_dir` rather than borrowing it**, so the gate cannot be deleted without
/// breaking the build — the same "unskippable by construction" property [`MochiVramPreflight`] gets from
/// bundling its quant marker, and the shape `mlx_fit_gate::apply_residency_policy(spec, engine_id) ->
/// WorkerResult<LoadSpec>` already uses at the MLX cache seam. A free-standing `check(&dir)?` would
/// still compile after a review mutation deleted it, silently un-gating the lane.
///
/// `budget` arrives resolved so this stays free of the GPU probe and is unit-testable without CUDA. No
/// budget signal (or an exempt engine) ⇒ admits. `Err` is the actionable pre-load rejection.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_vram_preflight(
    engine_id: &str,
    model_dir: PathBuf,
    gpu_id: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> WorkerResult<PathBuf> {
    match crate::vram_gate::video_weights_fit_error(
        engine_id,
        crate::vram_gate::wan_weight_bytes(engine_id, &model_dir),
        gpu_id,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(model_dir),
    }
}

/// The candle video lane's live VRAM budget: the real `nvidia-smi` reading, the
/// `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card emulation folded over it, then this process's reclaimable
/// cudarc pool added back (sc-11023).
///
/// The reclaimable fold IS correct here, unlike `krea_control_candle.rs` which deliberately omits it:
/// video routes through `generator_cache::with_cached_generator` (the `comfyui` in-place MoE is the one
/// uncached exception, and it is not Mochi), so the single exclusive cache slot evicts its occupant
/// BEFORE the incoming load and cudarc reuses those pages in-process. Without the fold, a warm re-gate
/// would be measured against a `free` that still counts the model it is about to replace. Matches
/// `generate_candle_stream` (image_jobs/base.rs).
///
/// The reverse direction is deliberately NOT wired: an admitted Mochi job does not call
/// [`crate::vram_gate::note_loaded_peak`], so it contributes nothing to the reclaimable high-water the
/// image lane reads. That keeps today's behavior (the video lane has never recorded a peak) rather than
/// guessing: Mochi's predicted peak is dominated by a TRANSIENT decode, not by resident weights, and
/// publishing ~81 GB as "reclaimable" on the strength of a derived floor would relax later image gates
/// on a number nothing has measured. Under-reporting the pool only ever fails conservative (a spurious
/// reject, never an OOM). Revisit once B5/sc-11995 backfills real `footprint.peakMemoryBytes`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn candle_video_vram_budget(settings: &Settings) -> Option<crate::vram_gate::VramBudget> {
    let budget = crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    );
    budget.map(|budget| {
        crate::vram_gate::with_reclaimable(
            budget,
            crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id),
        )
    })
}

/// Windows/CUDA candle video path (sc-5097 txt2video; sc-5175 adds the Wan2.2 14B MoE T2V + I2V).
/// Resolves the engine + weights, provisions the LTX Gemma encoder, resolves any i2v source-image
/// conditioning, builds a `VideoGenInput`, and runs it through the shared [`generate_video`] streaming
/// driver. Returns the decoded clip + the candle adapter label.
///
/// Every engine passes a VRAM fit gate before the load: Mochi's frame-dependent decode gate (sc-12306),
/// and the Wan weights-floor gate (sc-12344). `ltx`/`svd` are explicitly EXEMPT — their on-disk bytes are
/// not their loaded set, so any byte-derived floor would wall-reject working cards; the reason is
/// recorded on `vram_gate::wan_weight_components`. Neither gate reads `candle.vramGbByTier` (no candle
/// video model carries a `candle` block, so that lookup would admit unconditionally — sc-12344).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_video(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    generate_candle_video_using(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_candle_video`] with the engine loader supplied by the caller (sc-12318) — the candle
/// sibling of [`generate_mochi_using`], and for the same reason.
///
/// [`video_frame_count`] is a large part of the pre-load exposure here: swapping it for
/// `wan_frame_count` puts every non-Wan family (Mochi's `6k+1`, LTX's `8k+1`) off its engine's lattice,
/// which `validate_request` hard-rejects. `generate_candle_video_using_*` pins that call at the caller.
/// (Mochi's fit gate reads the coerced count too, but Wan's weights floor is frame-blind and LTX is
/// exempt, so the lattice remains this arm's own pin for those families.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_video_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let engine_id = candle_video_engine_id(&request.model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{} is not a candle video engine", request.model))
    })?;
    let adapter = candle_video_adapter_label(engine_id);
    let repo = candle_video_repo(request, engine_id);
    // Mochi (epic 1788 / sc-11992) resolves its tier dir through the SHARED resolver both lanes use —
    // `SceneWorks/mochi-1-mlx` serves candle too (A6's `.scales`-detect seam ingests the mlx-affine
    // tiers 1:1), so the tier-dir semantics, the on-demand q8/bf16 fetch and the shared-parent
    // co-requisite are identical off-Mac. It must NOT fall through to the wan tier-select below: the
    // Mochi tier layout is `<root>/{q4|q8|bf16}/transformer/` with the T5/VAE/tokenizer as siblings of
    // the tier dir, which `candle_wan_tier_subdir` does not understand.
    //
    // Keyed off the RESOLVED engine id, mirroring the `is_ltx` binding below — the id is already
    // resolved through `candle_video_engine_id`, so re-deriving the family from the model string here
    // would be a second, drift-prone source of truth.
    let is_mochi = engine_id == "mochi_1";
    if is_mochi {
        ensure_mochi_q8_present(api, settings, job, request).await?;
        ensure_mochi_bf16_present(api, settings, job, request).await?;
    }
    let snapshot_dir = if is_mochi {
        // Resolve-or-error; never a stub (the candle generic arm has no stub fallback once
        // `candle_video_engine_id` resolves the id).
        resolve_mochi_model_dir(settings, request)?
    } else {
        candle_video_snapshot_dir(settings, &repo)?
    };
    // Coerce the requested frame count onto the engine's temporal stride — the ONE shared ladder both
    // lanes use (sc-11992), so the candle stride can never drift from the MLX one. Computed HERE, above
    // the tier binding, because Mochi's fit gate (sc-12306) needs the coerced count: the decode peak is
    // linear in frames, so gating on the raw request would size the check against a length that never
    // renders. (The SVD arm below returns before this is read; it derives its own model-fixed burst.)
    let frames = video_frame_count(&request.model, request.raw_frame_count());
    // Wan quant-matrix tier-select (sc-10027): a candle wan tier repo (`SceneWorks/wan2.2-*-candle`) ships
    // q4/q8/bf16 subdirs — resolve the one matching `advanced.mlxQuantize` (default q4) and load from it
    // (the packed-detect seam reads the baked-in quant). A flat/dense repo (no subdirs, e.g. the
    // `Wan-AI/*-Diffusers` fallback) stays as-is with no quant marker.
    let (model_dir, wan_quant) = if is_mochi {
        // `resolve_mochi_model_dir` already returned the TIER dir. The VRAM fit gate (sc-12306) runs
        // here, and the quant marker comes back OUT of it — see `mochi_vram_preflight` for why the
        // marker is bundled into the gated return rather than read alongside a free-standing check.
        let MochiVramPreflight { quant } = mochi_vram_preflight(
            engine_id,
            &snapshot_dir,
            frames,
            request.width,
            request.height,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?;
        (snapshot_dir, quant)
    } else {
        let (dir, quant) = match candle_wan_tier_subdir(&snapshot_dir, engine_id, request) {
            Some((tier_dir, quant)) => (tier_dir, quant),
            None => (snapshot_dir, None),
        };
        // The Wan weights-floor VRAM fit gate (sc-12344), the non-Mochi half of this lane's admission
        // check. Runs on the RESOLVED tier dir (so it sizes the tier that will actually load, not the
        // manifest's default — the sc-12090 lesson) and BEFORE `candle_ensure_wan_lightning_present`
        // below, so a card that cannot hold the weights is refused without first paying for the
        // Lightning fetch. A no-op for `ltx`/`svd`, which are exempt — see `vram_gate::wan_weight_components`.
        let dir = wan_vram_preflight(
            engine_id,
            dir,
            &settings.gpu_id,
            candle_video_vram_budget(settings).await,
        )?;
        (dir, quant)
    };
    // ltx needs the separate Gemma-3-12B encoder (its only conditioning input). Resolve its snapshot
    // dir here and thread it onto the LoadSpec below (sc-8827) instead of mutating `$LTX_GEMMA_DIR`.
    let is_ltx = engine_id == "ltx_2_3_distilled";
    let ltx_gemma_dir = if is_ltx {
        resolve_ltx_gemma_dir(settings)
    } else {
        None
    };
    // Wan 14B I2V conditions on a source image (`Conditioning::Reference`); every other candle video
    // engine is txt2video-only (empty conditioning).
    let conditioning =
        resolve_candle_video_conditioning(settings, request, project_path, engine_id)?;

    // SVD-XT (sc-5493): image→video only — no prompt / negative / guidance (the engine uses its
    // frame-wise CFG ramp), a model-fixed burst (≤25 frames), the user `fps` as the playback cadence,
    // and the motion micro-conditioning knobs (motion_bucket_id / noise_aug_strength / decode_chunk /
    // conditioning_fps). Mirrors the MLX `generate_svd`; the conditioning is the source `Reference`
    // resolved above.
    if engine_id == "svd_xt" {
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id,
            model_dir,
            conditioning,
            width: request.width,
            height: request.height,
            frames: svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32,
            fps: request.fps,
            steps: Some(svd_steps(request)),
            seed: resolve_video_seed(request) as u64,
            motion_bucket_id: Some(
                svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
            ),
            noise_aug_strength: Some(svd_f32(
                request,
                "noiseAugStrength",
                "noiseAugStrength",
                0.02,
            )),
            decode_chunk_size: Some(
                svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32,
            ),
            conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
            ..VideoGenInput::default()
        };
        let mut raw_settings = svd_raw_settings(request);
        if let Value::Object(map) = &mut raw_settings {
            map.insert("repo".to_owned(), Value::String(repo.clone()));
        }
        let decoded = generate_video_using(
            api,
            settings,
            job,
            backend,
            &request.advanced,
            input,
            load_generator,
        )
        .await?;
        return Ok((decoded, adapter, raw_settings));
    }

    let is_wan = engine_id == "wan2_2_ti2v_5b"
        || engine_id == "wan2_2_t2v_14b"
        || engine_id == "wan2_2_i2v_14b";

    // Wan Lightning toggle + adapters (sc-10138): self-heal the A14B Lightning distill pair when the
    // toggle is on, then resolve the adapter specs (Lightning + user LoRAs, per-expert on the MoE). The
    // candle Wan engine applies these additively on a packed q4/q8 tier (candle-gen sc-10094/10095) or
    // folds them on a dense tier. `Vec::new()` for the non-Wan (ltx) engine.
    let adapters = if is_wan {
        candle_ensure_wan_lightning_present(api, settings, job, request, engine_id).await?;
        candle_resolve_wan_adapters(settings, request, engine_id)?
    } else {
        Vec::new()
    };

    // Descriptor-narrowed sampling surface: wan (5B + 14B) takes guidance + a negative prompt; the
    // distilled ltx takes neither (single-stage, no CFG). Wan uses the Lightning-aware recipe
    // ([`candle_wan_sampling`]: 4-step/CFG-off when the toggle is on, else native multi-step + CFG);
    // ltx keeps its own step default and no CFG.
    //
    // Mochi needs its OWN arm (sc-11992) — it is not distilled, so it takes true CFG (negative prompt
    // + guidance), but it is also not a Wan model: falling through to `candle_wan_sampling` would hit
    // that function's dense-5B tail and force `CANDLE_WAN5B_INTERIM_STEPS` (20) on it, silently
    // overriding the AsymmDiT's own 64-step default with a Wan tuning constant. `None` ⇒ the engine's
    // DEFAULT_STEPS (64) / DEFAULT_GUIDANCE (4.5) stand.
    let (steps, guidance, negative_prompt) = if is_ltx {
        (advanced_opt_u32(request, "steps"), None, None)
    } else if is_mochi {
        (
            advanced_opt_u32(request, "steps"),
            advanced_opt_f32(request, "guidanceScale"),
            non_empty_negative_prompt(request),
        )
    } else {
        let (steps, guidance) = candle_wan_sampling(engine_id, request);
        (steps, guidance, non_empty_negative_prompt(request))
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        // Wan quant-matrix tier marker (sc-10027): `Some(Q4/Q8)` when a packed candle tier subdir was
        // resolved, else `None` (bf16 tier / dense repo / ltx). A no-op on the candle wan load (the
        // packed-detect seam reads the tier's baked-in quant), carried for the LoadSpec + asset record.
        quant: wan_quant,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        // ltx's Gemma-3 encoder dir rides the LoadSpec (sc-8827); `None` for wan (bundled TE).
        text_encoder_dir: ltx_gemma_dir,
        ..VideoGenInput::default()
    };
    let raw_settings = candle_video_raw_settings(request, &repo);
    let decoded = generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        load_generator,
    )
    .await?;
    Ok((decoded, adapter, raw_settings))
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) in-place ComfyUI Wan2.2 base generation (epic 10451 Phase 2c, sc-10671): the
// video sibling of the z-image/qwen ComfyUI base image lanes. Reads a user's two ComfyUI Wan A14B
// expert files in place (native-Wan keys + companion scaled-fp8), remapped + dequant'd off-Mac via
// `runtime_cuda::providers::wan::load_from_comfyui_experts`. The UMT5 TE + VAE are read in place too when the tree
// carries them (sc-10909, folded into `components[]` by the API); the tokenizer (and either component
// when absent) comes from a resident `SceneWorks/wan2.2-*-candle` snapshot tier. T2V only for now
// (I2V's channel-concat reference conditioning is a follow-up); the model id is an `external_base_*`
// catalog row.
// ---------------------------------------------------------------------------

/// The candle Wan2.2 T2V-A14B engine id the ComfyUI base experts load into.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_T2V_ENGINE: &str = "wan2_2_t2v_14b";

/// The engine label recorded on candle ComfyUI Wan assets + telemetry.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_CANDLE_ADAPTER: &str = "candle_wan_comfyui";

/// The resident Wan2.2 T2V-A14B snapshot repo supplying the dense UMT5 TE / VAE / tokenizer (the
/// experts are read from the ComfyUI tree). Any complete tier subdir serves — the TE/VAE stay dense
/// across tiers; only the transformer (which we don't use here) is quantized.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_SNAPSHOT_REPO: &str = "SceneWorks/wan2.2-t2v-a14b-candle";

/// Tier subdirs probed for the dense TE/VAE/tokenizer (first fully-present tree wins).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_COMFYUI_SNAPSHOT_TIERS: &[&str] = &["q8", "q4", "bf16"];

/// Wan T2V DiT `patch_embedding` in-channels (16 latent). I2V is channel-concat (36) and needs the
/// reference-conditioning lane, so this slice serves only T2V experts.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const WAN_T2V_IN_CHANNELS: u64 = 16;

/// The two in-place ComfyUI expert files + the resident snapshot tier, plus the optional in-place UMT5
/// TE / VAE files (sc-10909). The snapshot tier always supplies the tokenizer (and the TE/VAE when their
/// files are absent).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
struct ComfyuiWanPaths {
    high: PathBuf,
    low: PathBuf,
    /// In-place UMT5 TE (`text_encoder` component), confined; `None` ⇒ snapshot `text_encoder/`.
    te: Option<PathBuf>,
    /// In-place Wan VAE (`vae` component), confined; `None` ⇒ snapshot `vae/`.
    vae: Option<PathBuf>,
    snapshot_dir: PathBuf,
}

/// Peek a safetensors header (8-byte length + JSON) and return the `patch_embedding.weight`
/// in-channels (`shape[1]`) — the T2V(16)/I2V(36) discriminator. `None` on any read/parse miss.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_expert_in_channels(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let header_len = u64::from_le_bytes(len) as usize;
    // Guard against a corrupt/huge length before allocating (headers are KB-scale).
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; header_len];
    file.read_exact(&mut buf).ok()?;
    let header: Value = serde_json::from_slice(&buf).ok()?;
    header
        .get("patch_embedding.weight")?
        .get("shape")?
        .as_array()?
        .get(1)?
        .as_u64()
}

/// Resolve the ComfyUI Wan expert paths + a resident snapshot tier from the forwarded `external_base_*`
/// row. `Ok(None)` (router falls through) when this is not a runnable ComfyUI Wan T2V job: wrong family,
/// not usable, missing an expert component, no resident Wan snapshot, or an I2V expert (36-channel — the
/// reference-conditioning lane is deferred). Each expert path is confined by
/// `normalize_app_managed_model_path` (the sc-10668-widened external-roots allow-list); the snapshot dir
/// is a fixed-repo/cache path (never payload-derived), so it needs no confinement.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_wan_comfyui_paths(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<Option<ComfyuiWanPaths>> {
    let entry = &request.model_manifest_entry;
    if entry.get("family").and_then(Value::as_str) != Some("wan-video") {
        return Ok(None);
    }
    if entry.get("usable").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let Some(components) = entry.get("components").and_then(Value::as_array) else {
        return Ok(None);
    };
    let path_for = |role: &str| -> Option<&str> {
        components
            .iter()
            .find(|component| component.get("role").and_then(Value::as_str) == Some(role))
            .and_then(|component| component.get("path").and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let (Some(high), Some(low)) = (path_for("transformer_high"), path_for("transformer_low"))
    else {
        return Ok(None);
    };
    let Some(snapshot_root) =
        huggingface_snapshot_dir(&settings.data_dir, WAN_COMFYUI_SNAPSHOT_REPO)
    else {
        return Ok(None);
    };
    let Some(snapshot_dir) = WAN_COMFYUI_SNAPSHOT_TIERS
        .iter()
        .map(|tier| snapshot_root.join(tier))
        .find(|dir| {
            dir.join("text_encoder").is_dir()
                && dir.join("vae").is_dir()
                && dir.join("tokenizer").join("tokenizer.json").is_file()
        })
    else {
        return Ok(None);
    };
    let high = crate::paths::normalize_app_managed_model_path(
        settings,
        high,
        "ComfyUI Wan high-noise expert",
    )?;
    let low = crate::paths::normalize_app_managed_model_path(
        settings,
        low,
        "ComfyUI Wan low-noise expert",
    )?;
    // T2V only: an I2V expert (36 in-channels) would load into the wrong config; decline it here (the
    // channel-concat reference lane is a follow-up) rather than surface a shape error at generate.
    if wan_expert_in_channels(&high) != Some(WAN_T2V_IN_CHANNELS) {
        return Ok(None);
    }
    // Optional in-place UMT5 TE + Wan VAE (sc-10909): the API folds them into `components[]` as
    // `text_encoder` / `vae` when the tree carries them; each is confined like the experts. Absent ⇒
    // `None` ⇒ the resident snapshot tier supplies that component (the row is complete either way).
    let te = path_for("text_encoder")
        .map(|te| {
            crate::paths::normalize_app_managed_model_path(settings, te, "ComfyUI Wan UMT5 encoder")
        })
        .transpose()?;
    let vae = path_for("vae")
        .map(|vae| crate::paths::normalize_app_managed_model_path(settings, vae, "ComfyUI Wan VAE"))
        .transpose()?;
    Ok(Some(ComfyuiWanPaths {
        high,
        low,
        te,
        vae,
        snapshot_dir,
    }))
}

/// True when this is a candle-runnable in-place ComfyUI Wan2.2 T2V job: an `external_base_*` model whose
/// forwarded row is a usable wan-video with both expert components + a resident snapshot. Mirrors the
/// image comfyui availability predicates.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn wan_comfyui_available(request: &VideoRequest, settings: &Settings) -> bool {
    request.model.starts_with("external_base_")
        && matches!(resolve_wan_comfyui_paths(request, settings), Ok(Some(_)))
}

/// Real candle in-place ComfyUI Wan2.2 T2V generation: resolve + confine the two expert paths and the
/// snapshot tier, then drive the shared [`generate_video`] funnel with `input.comfyui` set (the bespoke
/// uncached `load_from_comfyui_experts` path). Non-distilled base — native multi-step + per-expert CFG
/// (guidance `None` ⇒ the engine's per-expert defaults); no Lightning distill (the base lane folds no
/// adapters).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_wan_comfyui(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let _ = project_path;
    let paths = resolve_wan_comfyui_paths(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "ComfyUI Wan components could not be resolved (family/usable/experts/snapshot)"
                .to_owned(),
        )
    })?;
    let input = VideoGenInput {
        engine_id: WAN_COMFYUI_T2V_ENGINE,
        // The snapshot tier supplies the tokenizer (and the metrics `model_dir`, and the UMT5 TE / VAE
        // when their tree files are absent); the experts — and the TE/VAE when present — are read in
        // place via `comfyui` below (sc-10909).
        model_dir: paths.snapshot_dir.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: non_empty_negative_prompt(request),
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        // Non-Lightning base: honor a requested step count, else the engine default; per-expert CFG
        // defaults (guidance `None`). No adapters — the ComfyUI base lane does not fold LoRAs.
        steps: advanced_opt_u32(request, "steps"),
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        conditioning: Vec::new(),
        comfyui: Some(ComfyuiWanExperts {
            high_file: paths.high,
            low_file: paths.low,
            te_file: paths.te,
            vae_file: paths.vae,
            i2v: false,
        }),
        ..VideoGenInput::default()
    };
    let raw_settings = candle_video_raw_settings(request, WAN_COMFYUI_SNAPSHOT_REPO);
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, WAN_COMFYUI_CANDLE_ADAPTER, raw_settings))
}

// ---------------------------------------------------------------------------
// Candle (Windows/CUDA) SCAIL-2 generation (sc-6837, epic 6563): the off-Mac sibling of the macOS
// `generate_scail2` / `generate_scail2_replace` (epic 5439). Same end-to-end shape — a reference
// character image + a driving video → an animated clip (`animate_character`), or cross-identity
// `replace_person` (engine `replace_flag`) over the saved YOLO11 → ByteTrack → SAM3 person track. The
// worker paints the color-coded masks from the candle SAM3 segmenter (`person_segment_sam3_candle`);
// the painters (`scail2_masks`) are shared with the MLX lane. A distinct candle engine, NOT VACE — no
// torch fallback (`crate::inference_runtime::load("scail2_14b")` resolves the `candle_gen_scail2` provider, sc-6836).
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real candle SCAIL-2 asset (the candle sibling of `mlx_scail2`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SCAIL2_ADAPTER: &str = "candle_scail2";

/// SceneWorks SCAIL-2 model id → candle registry id, or `None` if `model` is not SCAIL-2.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_engine_id(model: &str) -> Option<&'static str> {
    (model == "scail2_14b").then_some("scail2_14b")
}

/// Map a SceneWorks video mode to the SCAIL-2 engine `video_mode` task string. `replace_person`
/// (cross-identity) flips the engine `replace_flag`; everything else (`animate_character`) is plain
/// animation. Mirrors the macOS `scail2_engine_video_mode`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "replace_person" => "replacement",
        _ => "animation",
    }
}

/// Resolve the candle SCAIL-2 snapshot dir from the `SCENEWORKS_CANDLE_SCAIL2_DIR` override, else the
/// app-managed `<data>/models/candle/scail2`. The sentinel is the `transformer/` subdir (the converted
/// SCAIL2Model DiT): the `candle_gen_scail2` provider builds its `Scail2Config` from hardcoded
/// Wan2.1-14B dims and reads only the component subdirs (transformer, vae, text_encoder, clip, and
/// `tokenizer/tokenizer.json`), not a root `config.json`, so that is the right marker. Errors loudly
/// when absent — like the candle Wan-VACE resolver, a missing checkpoint surfaces a clear error
/// instead of degrading to a stub (a character animation / replacement must never silently produce
/// meaningless output). The provider's `load` then validates each subdir and reports the precise gap.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_scail2_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_SCAIL2_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("transformer").is_dir() {
            return Ok(path);
        }
    }
    let managed = settings
        .data_dir
        .join("models")
        .join("candle")
        .join("scail2");
    if managed.join("transformer").is_dir() {
        return Ok(managed);
    }
    Err(WorkerError::InvalidPayload(format!(
        "scail2 (candle): no weights found. Place a candle-layout SCAIL-2 snapshot (transformer/ + \
         vae/ + text_encoder/ + clip/ + tokenizer/) at {} or set $SCENEWORKS_CANDLE_SCAIL2_DIR.",
        managed.display(),
    )))
}

/// Raw-settings recorded on a candle SCAIL-2 asset (mirrors the macOS `scail2_raw_settings`). When a
/// lightx2v lightning LoRA is applied (`lightning`, sc-6838), records the effective step-distill recipe
/// the worker dispatched so the chosen steps/CFG/shift is inspectable on the asset, not silent.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_raw_settings(request: &VideoRequest, lightning: bool) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "scail2Task".to_owned(),
        Value::String(candle_scail2_engine_video_mode(&request.mode).to_owned()),
    );
    if lightning {
        let (steps, guidance, shift) = candle_scail2_sampling(request, true);
        raw.insert("scail2Lightning".to_owned(), Value::Bool(true));
        if let Some(steps) = steps {
            raw.insert("effectiveSteps".to_owned(), json!(steps));
        }
        if let Some(guidance) = guidance {
            raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
        }
        if let Some(shift) = shift {
            raw.insert("effectiveSchedulerShift".to_owned(), json!(shift));
        }
    }
    Value::Object(raw)
}

/// The lightx2v lightning step-distill recipe (sc-6838, the candle sibling of the MLX sc-5684/5700
/// recipe): 8 steps, CFG off, scheduler shift 1.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SCAIL2_LIGHTNING_STEPS: u32 = 8;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SCAIL2_LIGHTNING_GUIDANCE: f32 = 1.0;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_SCAIL2_LIGHTNING_SHIFT: f32 = 1.0;

/// Build the candle SCAIL-2 adapter specs from `request.loras` — the candle sibling of the macOS
/// `resolve_scail2_adapters` (sc-5451). SCAIL-2 is a single dense Wan2.1-14B-I2V transformer (no MoE
/// high/low), so every adapter is shared (`moe_expert: None`); the engine merges LoRA / LoKr / LoHa and
/// the lightx2v lightning diff-patch into the dense DiT before build ([`runtime_cuda::providers::scail2::merge_adapters`]).
/// Carries both a user-selected SCAIL-2 LoRA and the bundled Bias-Aware DPO quality LoRA (both surface
/// through `request.loras`); selecting a lightning diff-patch LoRA makes the worker apply the
/// step-distill recipe ([`candle_scail2_sampling`]). Delegates to the shared [`resolve_dense_adapters`]
/// (sc-8830) — the byte-identical MLX/candle twin, now one implementation whose LoRA-file resolver is
/// core's recursive [`first_safetensors_path`] (the old candle twin's shallow scan is gone).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_resolve_scail2_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    resolve_dense_adapters(settings, request, MAX_JOB_LORAS)
}

/// `true` if any resolved adapter is a lightx2v diff-patch ("lightning") LoRA — the engine's own
/// detector (a file carrying full-rank `.diff`/`.diff_b` tensors), so the recipe keys off the actual
/// format, not a catalog id or filename. A file that can't be read is treated as non-lightning (the
/// engine surfaces the real load error downstream). The candle sibling of the macOS
/// `scail2_adapters_have_lightning`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_adapters_have_lightning(adapters: &[AdapterSpec]) -> bool {
    adapters
        .iter()
        .any(|a| runtime_cuda::providers::scail2::has_diff_patch_keys(&a.path).unwrap_or(false))
}

/// SCAIL-2 sampling recipe `(steps, guidance, scheduler_shift)`. When a lightx2v diff-patch "lightning"
/// LoRA is selected (`lightning`), apply the step-distill recipe (CFG off → the engine short-circuits to
/// a single DiT forward per step, scheduler shift 1.0; step count defaults to 8 but honors an explicit
/// `advanced.steps`). Without it, all-`None` so the engine's quality defaults stand. The candle sibling
/// of the macOS `scail2_sampling`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_scail2_sampling(
    request: &VideoRequest,
    lightning: bool,
) -> (Option<u32>, Option<f32>, Option<f32>) {
    if !lightning {
        return (None, None, None);
    }
    (
        advanced_opt_u32(request, "steps").or(Some(CANDLE_SCAIL2_LIGHTNING_STEPS)),
        Some(CANDLE_SCAIL2_LIGHTNING_GUIDANCE),
        Some(CANDLE_SCAIL2_LIGHTNING_SHIFT),
    )
}

/// Resolve a candle SCAIL-2 `animate_character` request into the engine conditioning — the candle
/// sibling of the macOS `resolve_scail2_conditioning`. Loads the reference character image + the
/// driving clip, segments both with the candle SAM3 PCS segmenter (every person → a palette color,
/// left-to-right), paints the color-coded masks (animation: reference bg white, driving bg black), and
/// assembles `Reference` + `Mask` + `ControlClip`. Segmentation + painting run on the blocking pool.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_scail2_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // The character: a reference image (referenceAssetIds first, else the i2v sourceAssetId).
    let ref_id = request
        .reference_asset_ids
        .first()
        .map(String::as_str)
        .or(request.source_asset_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "scail2 animate_character requires a reference character image (referenceAssetIds \
                 or sourceAssetId)."
                    .into(),
            )
        })?;
    let reference = crate::image_jobs::load_reference_image(
        &settings.data_dir,
        &request.project_id,
        ref_id,
        project_path,
    )?;

    // The driving video → frames at the output size (the engine re-resizes internally). Reuses the
    // candle Wan-VACE source-clip loader (`load_source_video_frames`, which reads `sourceClipAssetId`
    // and aspect-fits to W×H) — the candle-lane sibling of the macOS path's `extract_clip_frames`
    // (that helper is macOS-only).
    if request
        .source_clip_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .is_none()
    {
        return Err(WorkerError::InvalidPayload(
            "scail2 animate_character requires a driving video (sourceClipAssetId).".into(),
        ));
    }
    let driving = load_source_video_frames(
        api,
        settings,
        job,
        request,
        project_path,
        wan_frame_count(request.raw_frame_count()) as usize,
    )
    .await?;

    // SAM3 segmenter weights (download-on-first-use), shared by both segmentation passes.
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3_candle::ensure_segmenter_weights(settings, &context).await?;

    // Decode the engine `Image`s to `RgbImage`s for SAM3 (it normalizes RGB internally).
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    let driving_rgb: Vec<image::RgbImage> = driving
        .iter()
        .map(|f| image::RgbImage::from_raw(f.width, f.height, f.pixels.clone()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| WorkerError::InvalidPayload("scail2 driving frame is malformed".into()))?;

    // Segment + paint via the shared orchestrator (sc-8830). Animation keeps the reference's world
    // (ref bg white) and drops the driving world (driving bg black); the candle SAM3 module is the
    // off-Mac twin, whose per-frame propagate contract (sc-8972) observes the tripped cancel flag
    // between frames.
    let (rm, rt) = (sam_model.clone(), sam_tokenizer.clone());
    assemble_scail2_animate_conditioning(
        api,
        settings,
        &job.id,
        reference,
        driving,
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &rm,
                &rt,
                std::slice::from_ref(&ref_rgb),
                Some(flag),
                None,
            )?;
            crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_WHITE)
        },
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                &driving_rgb,
                Some(flag),
                Some(Box::new(|frame, total| {
                    tracing::debug!(event = "scail2_sam3_propagate_progress", frame, total);
                })),
            )?;
            crate::scail2_masks::paint_driving_masks(&masks, crate::scail2_masks::BG_BLACK)
        },
    )
    .await
}

/// Real candle SCAIL-2 character animation (sc-6837 + sc-6838): build the `VideoGenInput` and run the
/// shared `generate_video` path. `animate_character` → engine task `animation`; the source media becomes
/// the SAM3-painted conditioning. Inference LoRA / LoKr / LoHa + the Bias-Aware DPO LoRA + the lightx2v
/// lightning diff-patch resolve from `request.loras` and merge into the dense DiT (sc-6838); a lightning
/// LoRA also applies the step-distill recipe (8 steps / CFG-off / shift 1). Frame count uses the Wan
/// 1-mod-4 stride (the renderer is Wan2.1); the engine stitches > 81-frame clips into overlapping segments.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_scail2(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, &'static str, Value)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_candle_scail2_conditioning(api, settings, job, request, project_path).await?;
    // Inference adapters (DPO / lightning / user LoRA) + the lightning step-distill recipe.
    let adapters = candle_resolve_scail2_adapters(settings, request)?;
    let lightning = candle_scail2_adapters_have_lightning(&adapters);
    let (steps, guidance, scheduler_shift) = candle_scail2_sampling(request, lightning);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: resolve_candle_scail2_model_dir(settings)?,
        adapters,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(candle_scail2_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((
        decoded,
        CANDLE_SCAIL2_ADAPTER,
        candle_scail2_raw_settings(request, lightning),
    ))
}

/// Resolve a candle SCAIL-2 `replace_person` request into cross-identity replacement conditioning — the
/// candle sibling of the macOS `resolve_scail2_replace_conditioning` (sc-5452). Reuses the masks
/// SceneWorks already computed: the saved person track (YOLO11 → ByteTrack → SAM3, corrections applied)
/// supplies the per-frame driving masks; the character's approved reference is the identity. Driving
/// frames load exactly as the candle Wan-VACE backend loads them (`load_source_video_frames`) so the
/// resampled track masks stay frame-aligned 1:1. Replacement keeps the driving clip's world (driving
/// mask bg white, reference mask bg black); `video_mode = "replacement"` flips the engine `replace_flag`.
/// SCAIL-2 replaces the whole tracked person, so face_only/full_person + maskingStrength are inert.
/// Returns the conditioning plus the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_scail2_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Driving frames + their per-frame binary person masks — the same source the candle Wan-VACE
    // backend consumes, loaded identically so the resampled masks align 1:1 with the frames.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let driving =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let frame_total = driving.len();
    let (binary_masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frame_total,
    )?;
    // The tracked person → blue (person 0); replacement keeps the driving's world → white bg.
    let driving_masks = crate::scail2_masks::paint_track_driving_masks(
        &binary_masks,
        crate::scail2_masks::BG_WHITE,
    );

    // The character identity: the first approved reference image (multi-ref = the engine-contract
    // extension, sc-5583 on the MLX side).
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let reference = references.into_iter().next().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        )
    })?;

    // The reference color mask: a candle SAM3 pass on the reference image → the primary person painted
    // blue on a black background (replacement discards the reference's surrounding world).
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3_candle::ensure_segmenter_weights(settings, &context).await?;
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    // Heartbeat keepalive + user cancel across the cold SAM3 parse + single-frame propagate
    // (sc-8390 / sc-8807), via the shared blocking-segment helper (sc-8830); the engine's per-frame
    // propagate contract (sc-8972) observes the tripped flag between frames, beyond the coarse seams
    // (cold parse / model build).
    let ref_mask = scail2_segment_blocking(
        api,
        settings,
        &job.id,
        "scail2 reference segment task",
        move |flag| {
            let masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                std::slice::from_ref(&ref_rgb),
                Some(flag),
                None,
            )?;
            crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_BLACK)
        },
    )
    .await?;

    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        1.0,
        reference_count,
        frame_total,
        CANDLE_SCAIL2_ADAPTER,
    );
    Ok((conditioning, status))
}

/// Real candle SCAIL-2 cross-identity replacement (sc-6837): the candle sibling of the MLX
/// `generate_scail2_replace`. Builds the replacement conditioning from the saved person track +
/// character reference and runs the shared `generate_video` path with `video_mode = "replacement"`
/// (engine `replace_flag = true`). Returns the decoded video + the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_scail2_replace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let (conditioning, status) =
        resolve_candle_scail2_replace_conditioning(api, settings, job, request, project_path)
            .await?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: resolve_candle_scail2_model_dir(settings)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some("replacement".to_owned()),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, status))
}

/// Resolve the candle Wan-VACE diffusers snapshot dir (sc-5494): `SCENEWORKS_CANDLE_WAN_VACE_DIR`
/// override (when it holds a `transformer/config.json`), else the HF [`CANDLE_WAN_VACE_REPO`] snapshot.
/// Errors loudly when absent (no stub fallback — a missing VACE checkpoint surfaces a re-download error).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_CANDLE_WAN_VACE_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("transformer").join("config.json").is_file() {
            return Ok(path);
        }
    }
    candle_video_snapshot_dir(settings, CANDLE_WAN_VACE_REPO)
}

/// Windows/CUDA candle Wan-VACE `replace_person` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace`]. Resolves the diffusers VACE snapshot, extracts the source-clip control frames,
/// builds the per-frame person mask from the saved track + the character references, and runs the
/// `wan_vace` engine. Person detect/track/segment stays upstream (the masks are pre-saved); the
/// conditioning builders are shared with the MLX path. No quant / LoRA (the candle VACE provider rejects
/// them). Returns the decoded clip + the honest `replacementStatus`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();

    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;
    let negative_prompt = non_empty_negative_prompt(request);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        CANDLE_WAN_VACE_ADAPTER,
    );
    Ok((decoded, status))
}

/// Windows/CUDA candle Wan-VACE `extend_clip` / `video_bridge` (sc-5494): the candle sibling of the MLX
/// [`generate_wan_vace_extend_bridge`]. Loads the real source-clip anchor frames (the left clip's tail
/// for extend; both clips' boundaries for bridge), builds the source-at-kept-positions + generated-span
/// ControlClip, and runs the `wan_vace` engine. No reference images, no quant / LoRA.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let model_dir = resolve_candle_wan_vace_model_dir(settings)?;
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = non_empty_negative_prompt(request);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

/// Resolve a Wan request into a [`VideoGenInput`] and run it (sc-3034).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_wan(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let (steps, guidance) = wan_sampling(engine_id, request);
    let negative_prompt = non_empty_negative_prompt(request);
    // extend_clip / video_bridge build single-frame boundary `Keyframe` conditioning from the
    // source clip(s) (async ffmpeg frame extraction, sc-3357); every other mode resolves
    // keyframe/reference conditioning synchronously from images.
    let conditioning = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => {
            resolve_wan_clip_conditioning(api, settings, job, request, project_path, engine_id)
                .await?
        }
        _ => resolve_wan_conditioning(settings, request, project_path, engine_id)?,
    };
    // Wan quant matrix (sc-9941 TI2V-5B / sc-9942 T2V / sc-9943 I2V, epic 8506): the macOS default
    // install is the lean q4 tier; a q8/bf16 job fetches that subdir on demand before resolving. No-op
    // for a model with no hosted tier matrix, a q4 job, or an already-present tier.
    ensure_wan_tier_present(api, settings, job, request).await?;
    // The A14B MoE recipe uses the Lightning distill LoRA by default (sc-10030 / sc-10047 default-on
    // toggle). It normally installs as a manifest coRequisite, but self-heal a worker that installed
    // the model before the coRequisite existed so resolve_wan_adapters below doesn't dead-end. No-op
    // for the 5B model (no Lightning), an already-cached pair, or a job that opted out of Lightning.
    ensure_wan_lightning_present(api, settings, job, request, engine_id).await?;
    // Descend into the chosen quant-matrix tier subdir when the turnkey ships them; a pre-packed tier
    // loads with quant=None (config.json is authoritative). A legacy flat snapshot (or a model with no
    // hosted matrix) keeps the root + load-time quant.
    let (model_dir, quant) = if wan_tier_repo(&request.model).is_some() {
        resolve_wan_tier_dir_and_quant(settings, request, engine_id)?
    } else {
        (
            resolve_wan_model_dir(settings, &request.model, engine_id)?,
            resolve_wan_quant(request),
        )
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        adapters: resolve_wan_adapters(settings, request, engine_id)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Real MLX Bernini generation (macOS, via mlx-gen-bernini, epic 4699 / sc-4707 + sc-4703 + sc-5425):
// the full Qwen2.5-VL semantic planner + Wan2.2-T2V-A14B dual-expert renderer. Serves the planner
// video task surface: text_to_video (t2v), video_to_video (v2v — source-clip edit),
// reference_to_video (r2v — subject references → video), reference_video_to_video (rv2v — source clip
// + references), multi_video_to_video (mv2v — multiple source clips), ads2v (source video + reference
// video + references). The SceneWorks mode maps to the engine `video_mode` task string and the source
// media is resolved into the planner's `VideoClip` / `MultiReference` conditioning. Q4 default / Q8
// opt-in at load. The turnkey `SceneWorks/bernini-mlx` snapshot is self-contained. (t2i/i2i image
// companion = a separate image-typed catalog id, tracked under epic 4699.)
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Bernini asset.
#[cfg(target_os = "macos")]
const BERNINI_ADAPTER: &str = "mlx_bernini";

/// SceneWorks Bernini model id → mlx-gen registry id, or `None` if `model` is not the Bernini family.
#[cfg(target_os = "macos")]
fn bernini_engine_id(model: &str) -> Option<&'static str> {
    (model == "bernini").then_some("bernini")
}

/// Whether the linked Bernini engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn bernini_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_bernini_model_dir(settings).is_ok()
}

/// Resolve the Bernini MLX snapshot dir: env override (`SCENEWORKS_MLX_BERNINI_DIR`) → app-managed
/// `<data>/models/mlx/bernini` → the turnkey download-on-first-use `SceneWorks/bernini-mlx` snapshot
/// (mirrors `resolve_wan_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bernini_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_BERNINI_DIR", "bernini") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/bernini-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "bernini: no MLX weights found. Download the turnkey SceneWorks/bernini-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_BERNINI_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("bernini")
            .display(),
    )))
}

/// MLX quantization for a Bernini load: Q4 default (the validated 128 GB-fitting tier, sc-4709 ~44 GB
/// peak), Q8 opt-in via the advanced `mlxQuantize: 8` control, explicit `<= 0` ⇒ bf16 (power users
/// with ample RAM). Never defaults to bf16 — the snapshot is ~93 GB at bf16. Delegates to the shared
/// [`resolve_mlx_dense_quant`] (sc-8830) — the byte-identical Bernini/SCAIL-2 twin.
#[cfg(target_os = "macos")]
fn resolve_bernini_quant(request: &VideoRequest) -> Option<Quant> {
    resolve_mlx_dense_quant(request)
}

// --- Bernini quant-matrix tiers (sc-9945, epic 8506) ---------------------------------------------
// The composite sibling of the Wan quant-matrix tiers. `SceneWorks/bernini-mlx` hosts self-contained
// `q4/` (default) + `q8/` + `bf16/` tier subdirs, each a COMPLETE composite snapshot (the Qwen2.5-VL
// planner components + both Wan renderer experts + the shared dense T5/VAE/tokenizer + config
// sidecars). The legacy flat dense-bf16 layout quantized at LOAD, staging the ~56 GB experts + ~14 GB
// planner backbone as bf16 first; a pre-packed tier loads with no dense-staging peak. Both the video
// (`bernini`) and image (`bernini_image`) load paths resolve through the same tier machinery. The flat
// root files stay for already-shipped workers; a new worker resolves the tier subdirs. Mirrors the
// `wan_tier_*` helpers, but keyed on raw `mlxQuantize` bits so it serves the image lane too.

/// The turnkey SceneWorks Bernini MLX repo (sc-9945). Hosts the `q4/`/`q8/`/`bf16/` tier subdirs.
#[cfg(target_os = "macos")]
const BERNINI_REPO: &str = "SceneWorks/bernini-mlx";

/// Pinned revision for [`BERNINI_REPO`] — the commit that adds the `q4/`/`q8/`/`bf16/` tier subdirs
/// (sc-9945). Pinning the exact commit (not the mutable `main`) stops an upstream re-push from silently
/// swapping a checkpoint the on-demand `q8/*` + `bf16/*` fetch loads (the native downloader still verifies each
/// file's own hash on download). This is the commit that added the `q4/`/`q8/`/`bf16/` tier subdirs
/// (sc-9945), with the exact hosted sizes: q4 37,815,703,819 / q8 55,129,270,617 / bf16 87,591,990,679.
#[cfg(target_os = "macos")]
const BERNINI_REVISION: &str = "533d688f16c8f33dc832890c1e16c11921a2019a";

/// The runtime files that make a Bernini tier subdir COMPLETE for the load path: the planner
/// components + both renderer experts + the shared dense T5/VAE + tokenizer + config sidecars + the
/// planner's `mllm/tokenizer.json` (mirrors `bernini_tier_build::TIER_FILES`). The three packable
/// weights (`qwen2_5_vl.safetensors`, `high/low_noise_model.safetensors`) are present in every tier;
/// only their contents differ (packed vs dense).
#[cfg(target_os = "macos")]
const BERNINI_TIER_FILES: &[&str] = &[
    "qwen2_5_vl.safetensors",
    "qwen2_5_vl_config.json",
    "connector.safetensors",
    "vit_decoder.safetensors",
    "mask_tokens.safetensors",
    "bernini_planner.json",
    "high_noise_model.safetensors",
    "low_noise_model.safetensors",
    "t5_encoder.safetensors",
    "vae.safetensors",
    "tokenizer.json",
    "config.json",
    "bernini_renderer.json",
    "mllm/tokenizer.json",
];

/// Bernini VIDEO-lane default tier order (no explicit `mlxQuantize`): **q4-first** (sc-10859). The MLX
/// video lane has no Q8 lever, so a silent Q8 default only risks a video-runtime OOM at heavy
/// res/frame counts — see [`wan_tier_order`]. Passed by the video generation path.
#[cfg(target_os = "macos")]
pub(crate) const BERNINI_VIDEO_DEFAULT_TIER_ORDER: &[&str] = &["q4", "q8"];

/// Bernini IMAGE-lane default tier order (no explicit `mlxQuantize`): **q8-first** — epic 10721 /
/// sc-10726's app-wide Q8 default, kept for the image lane (Image Studio has a Q8 picker and the
/// sc-8516 budget's 1024²-image transient is exactly what applies). Passed by `image_jobs::bernini`.
#[cfg(target_os = "macos")]
pub(crate) const BERNINI_IMAGE_DEFAULT_TIER_ORDER: &[&str] = &["q8", "q4"];

/// The Bernini quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`wan_tier_order`]): `mlxQuantize <= 0` ⇒ `bf16`; `>= 8` ⇒ `q8`; an explicit `q4` ⇒ q4-first; and —
/// with NO explicit pick — `default_order`. `bf16` stays OUT of the default order, so a default job
/// never pulls the huge dense tier by accident.
///
/// Bernini is SHARED by the video (`bernini`) and image (`bernini_image`) lanes, whose *default* tiers
/// diverge (sc-10859): the caller passes [`BERNINI_VIDEO_DEFAULT_TIER_ORDER`] (q4-first, video OOM
/// carve-out) or [`BERNINI_IMAGE_DEFAULT_TIER_ORDER`] (q8-first, epic-10721 app-wide default). Only the
/// no-explicit-pick (`None`) arm consults it; every explicit pick is lane-independent.
#[cfg(target_os = "macos")]
fn bernini_tier_order(
    bits: Option<i64>,
    default_order: &'static [&'static str],
) -> &'static [&'static str] {
    match bits {
        None => default_order,
        Some(b) if b <= 0 => &["bf16", "q8", "q4"],
        Some(b) if b >= 8 => &["q8", "q4"],
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained Bernini tier snapshot (all [`BERNINI_TIER_FILES`]). A
/// partially-downloaded tier fails this so [`bernini_tier_subdir`] falls through to a smaller complete
/// tier rather than half-loading.
#[cfg(target_os = "macos")]
fn bernini_tier_is_complete(dir: &Path) -> bool {
    BERNINI_TIER_FILES
        .iter()
        .all(|file| dir.join(file).is_file())
}

/// Descend a resolved Bernini repo `root` into the requested quant tier subdir (sc-9945), mirroring
/// [`wan_tier_subdir`]. Returns the first COMPLETE tier in [`bernini_tier_order`], or `None` when the
/// repo has no complete tier subdir — a legacy flat snapshot, where the caller keeps the root +
/// load-time quant.
#[cfg(target_os = "macos")]
fn bernini_tier_subdir(
    root: &Path,
    bits: Option<i64>,
    default_order: &'static [&'static str],
) -> Option<PathBuf> {
    bernini_tier_order(bits, default_order)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| bernini_tier_is_complete(dir))
}

/// Resolve the Bernini `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9945). A pre-packed tier's config sidecars
/// are authoritative — the planner (`Qwen25VlText::from_weights`, via the `quantization` block) and
/// both renderer experts (`WanTransformer::from_weights`) build packed — so a resolved tier loads with
/// `quant = None`: `mlxQuantize` selects WHICH tier, never a load-time requant (the `bf16/` tier is
/// dense, so `None` ⇒ dense too). A legacy flat snapshot (no tier subdirs) keeps today's behavior: load
/// the root and quantize at load per `legacy_quant`. Shared by the video + image lanes (each passes its
/// own parsed `bits` + `legacy_quant` + `default_order`: [`BERNINI_VIDEO_DEFAULT_TIER_ORDER`] for video,
/// [`BERNINI_IMAGE_DEFAULT_TIER_ORDER`] for image — the no-explicit-pick default diverges per sc-10859).
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bernini_tier_dir_and_quant(
    settings: &Settings,
    bits: Option<i64>,
    legacy_quant: Option<Quant>,
    default_order: &'static [&'static str],
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_bernini_model_dir(settings)?;
    match bernini_tier_subdir(&root, bits, default_order) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, legacy_quant)),
    }
}

/// Parse `advanced.mlxQuantize` (int or numeric string) for the Bernini tier selector — the raw bits
/// the tier order keys on (the video lane's twin of the image lane's `resolve_bernini_image_quant`
/// bits). `resolve_bernini_quant` maps the same value to a `Quant`; this keeps the tier selection and
/// the legacy load-time quant in sync off one source.
#[cfg(target_os = "macos")]
fn bernini_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// On-demand fetch of a non-default Bernini quant-matrix tier subdir (sc-9945, mirrors
/// [`ensure_wan_tier_present`]). The macOS default download is the lean `q4/` tier; a job that opts into
/// a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`) pulls just that subdir from the FIXED
/// [`BERNINI_REVISION`] the first time it is requested so [`bernini_tier_subdir`] can resolve it. No-op
/// for a `q4` (default) job, when the repo snapshot isn't downloaded yet (resolve surfaces the clear
/// error), or when the tier is already complete. Fails loud on a real download error — fast, before any
/// compute; a tier that isn't published yet stays absent so resolve falls back to a smaller complete tier.
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_bernini_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    bits: Option<i64>,
) -> WorkerResult<()> {
    let tier = match bits {
        Some(b) if b <= 0 => "bf16",
        Some(b) if b >= 8 => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, BERNINI_REPO) else {
        return Ok(());
    };
    if bernini_tier_is_complete(&root.join(tier)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        BERNINI_REPO,
        BERNINI_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Raw-settings recorded on a real MLX Bernini asset (mirrors `wan_raw_settings`).
#[cfg(target_os = "macos")]
fn bernini_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine guidance task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "berniniTask".to_owned(),
        Value::String(bernini_engine_video_mode(&request.mode).to_owned()),
    );
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the Bernini engine `video_mode` task string (which selects the
/// renderer guidance mode). The engine also infers the mode from the supplied conditioning, but the
/// explicit task keeps the mapping unambiguous. Unknown / `text_to_video` ⇒ plain `t2v`.
#[cfg(target_os = "macos")]
fn bernini_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "video_to_video" => "v2v",
        "reference_to_video" => "r2v",
        "reference_video_to_video" => "rv2v",
        // Multi-source-video modes (sc-5425): mv2v (multiple source clips) and ads2v
        // (source video + reference video + reference images). Both resolve to the
        // engine's `V2vApg` guidance via `task_to_vit_mode`; they differ only in the
        // supplied media.
        "multi_video_to_video" => "mv2v",
        "ads2v" => "ads2v",
        _ => "t2v",
    }
}

/// Resolve the source media for a Bernini editing/reference request into the planner conditioning:
/// source clips → [`Conditioning::VideoClip`] (the edit structure, VAE/ViT-encoded by the engine)
/// and subject reference images → [`Conditioning::MultiReference`]. `text_to_video` needs none.
/// The single-clip modes (v2v / rv2v / ads2v) use `sourceClipAssetId`; mv2v supplies several via
/// `sourceClipAssetIds`; ads2v additionally appends the reference video (`referenceClipAssetId`) as a
/// second clip after the source clip (sc-5425). Clips are emitted videos-first, in submission order,
/// then images — matching the engine's `collect_conditioning` / `assign_source_ids` ordering. Each
/// mode's required media is enforced here (defense in depth — the API validates the same), so a
/// mis-built request fails loudly instead of silently rendering an unconditioned clip. Every clip is
/// decoded to the output frame count (the engine resamples to its `target_fps` grid).
#[cfg(target_os = "macos")]
async fn resolve_bernini_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let mode = request.mode.as_str();

    // Source video clips, in the order the engine assigns source ids (videos first; for ads2v the
    // source clip leads the reference video).
    let mut clip_ids: Vec<&str> = Vec::new();
    match mode {
        "video_to_video" | "reference_video_to_video" | "ads2v" => {
            let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "bernini {} requires a source clip (sourceClipAssetId).",
                    request.mode.replace('_', " ")
                ))
            })?;
            clip_ids.push(clip_id);
        }
        "multi_video_to_video" => {
            if request.source_clip_asset_ids.len() < 2 {
                return Err(WorkerError::InvalidPayload(
                    "bernini multi video to video requires at least two source clips \
                     (sourceClipAssetIds)."
                        .to_owned(),
                ));
            }
            clip_ids.extend(request.source_clip_asset_ids.iter().map(String::as_str));
        }
        _ => {}
    }
    if mode == "ads2v" {
        let ref_clip_id = request.reference_clip_asset_id.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "bernini ads2v requires a reference video (referenceClipAssetId).".to_owned(),
            )
        })?;
        clip_ids.push(ref_clip_id);
    }

    let mut conditioning = Vec::new();
    for clip_id in clip_ids {
        let frames = extract_clip_frames(
            api,
            settings,
            job,
            &request.project_id,
            project_path,
            clip_id,
            request.width,
            request.height,
            wan_frame_count(request.raw_frame_count()),
        )
        .await?;
        conditioning.push(Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        });
    }

    // Subject reference images → MultiReference (r2v / rv2v / ads2v).
    let needs_refs = matches!(
        mode,
        "reference_to_video" | "reference_video_to_video" | "ads2v"
    );
    if needs_refs {
        if request.reference_asset_ids.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "bernini {} requires at least one reference image (referenceAssetIds).",
                request.mode.replace('_', " ")
            )));
        }
        let mut images = Vec::with_capacity(request.reference_asset_ids.len());
        for asset_id in &request.reference_asset_ids {
            images.push(load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?);
        }
        conditioning.push(Conditioning::MultiReference { images });
    }

    Ok(conditioning)
}

/// Real MLX Bernini video generation (epic 4699 / sc-4707 + sc-4703 + sc-5425): build the
/// `VideoGenInput` and run the shared `generate_video` path. The SceneWorks mode resolves to the
/// engine `video_mode` task ([`bernini_engine_video_mode`]) and the source media into the planner
/// conditioning ([`resolve_bernini_conditioning`]) — empty for t2v, one or more `VideoClip`s for
/// v2v/mv2v/rv2v/ads2v, and `MultiReference` for r2v/rv2v/ads2v. No LoRA (the engine reports
/// `supports_lora=false`); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the renderer is Wan).
#[cfg(target_os = "macos")]
async fn generate_bernini(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_bernini_conditioning(api, settings, job, request, project_path).await?;
    // sc-9945: fetch the requested quant tier subdir (q8/bf16) if it isn't the shipped q4 default, then
    // descend into it — a pre-packed tier loads with `quant = None` (config sidecars authoritative); a
    // legacy flat snapshot keeps load-time quant.
    let bits = bernini_quant_bits(request);
    ensure_bernini_tier_present(api, settings, job, bits).await?;
    // Video lane: no explicit pick defaults q4-first (sc-10859 OOM carve-out), unlike the image lane.
    let (model_dir, quant) = resolve_bernini_tier_dir_and_quant(
        settings,
        bits,
        resolve_bernini_quant(request),
        BERNINI_VIDEO_DEFAULT_TIER_ORDER,
    )?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(bernini_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Real candle Bernini VIDEO generation (Windows/CUDA, via candle-gen-bernini, sc-10997 / epic 6562):
// the off-Mac sibling of the macOS `generate_bernini` above. The full Qwen2.5-VL planner + Wan2.2-T2V-
// A14B renderer registers under `bernini` (`Modality::Video`); the video path reaches it via
// `crate::inference_runtime::load("bernini")` through runtime-cuda's explicit catalog. Serves
// `text_to_video` + the editing/reference/multi-source video
// modes (v2v / r2v / rv2v / mv2v / ads2v); `generate_candle_bernini` maps the SceneWorks mode to the
// engine guidance task and resolves the source media into the planner conditioning. Loads the converted
// `SceneWorks/bernini` snapshot DENSE (the candle loader reads the tree as-is; the off-Mac
// packed-tier select is deferred WITH the still lane until the `SceneWorks/bernini` tier layout lands —
// GPU-val is gated on sc-11003). No LoRA (the engine reports
// `supports_lora=false`). A distinct candle engine — NO torch fallback (a missing snapshot fails loud
// at load, never a silent stub).
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real candle Bernini VIDEO asset — the `candle_<family>` sibling of the MLX
/// `mlx_bernini` label (same spelling as the still lane's `CANDLE_BERNINI_IMAGE_ADAPTER`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_BERNINI_ADAPTER: &str = "candle_bernini";

/// SceneWorks Bernini model id → candle registry id, or `None` if `model` is not the Bernini video
/// family. The candle sibling of the macOS `bernini_engine_id`; drives the `CandleVideoRoute::Bernini`
/// arm (routed on the model id, not weight availability).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_engine_id(model: &str) -> Option<&'static str> {
    (model == "bernini").then_some("bernini")
}

/// Map a SceneWorks video mode to the Bernini engine `video_mode` task string (which selects the
/// renderer guidance mode). The candle sibling of the macOS `bernini_engine_video_mode` — the identical
/// mapping so the two lanes render the same mode for the same request. Unknown / `text_to_video` ⇒
/// plain `t2v`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "video_to_video" => "v2v",
        "reference_to_video" => "r2v",
        "reference_video_to_video" => "rv2v",
        // Multi-source-video modes (sc-5425): mv2v (multiple source clips) and ads2v (source video +
        // reference video + reference images). Both resolve to the engine's `V2vApg` guidance; they
        // differ only in the supplied media.
        "multi_video_to_video" => "mv2v",
        "ads2v" => "ads2v",
        _ => "t2v",
    }
}

/// Raw-settings recorded on a real candle Bernini VIDEO asset (mirrors the macOS `bernini_raw_settings`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_bernini_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine guidance task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "berniniTask".to_owned(),
        Value::String(candle_bernini_engine_video_mode(&request.mode).to_owned()),
    );
    Value::Object(raw)
}

/// Resolve the source media for a candle Bernini editing/reference request into the planner
/// conditioning — the candle sibling of the macOS `resolve_bernini_conditioning`. Source clips →
/// [`Conditioning::VideoClip`] (the edit structure, VAE/ViT-encoded by the engine) and subject
/// reference images → [`Conditioning::MultiReference`]. `text_to_video` needs none. The single-clip
/// modes (v2v / rv2v / ads2v) use `sourceClipAssetId`; mv2v supplies several via `sourceClipAssetIds`;
/// ads2v additionally appends the reference video (`referenceClipAssetId`) as a second clip after the
/// source clip (sc-5425). Clips are emitted videos-first, in submission order, then images — matching
/// the engine's `collect_conditioning` / `assign_source_ids` ordering. Each mode's required media is
/// enforced here (defense in depth — the API validates the same), so a mis-built request fails loud
/// instead of silently rendering an unconditioned clip. Clips decode to the output frame count via the
/// candle-shared [`extract_clip_frames`]; reference images load via the shared `load_reference_image`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn resolve_candle_bernini_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let mode = request.mode.as_str();

    // Source video clips, in the order the engine assigns source ids (videos first; for ads2v the
    // source clip leads the reference video).
    let mut clip_ids: Vec<&str> = Vec::new();
    match mode {
        "video_to_video" | "reference_video_to_video" | "ads2v" => {
            let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "bernini {} requires a source clip (sourceClipAssetId).",
                    request.mode.replace('_', " ")
                ))
            })?;
            clip_ids.push(clip_id);
        }
        "multi_video_to_video" => {
            if request.source_clip_asset_ids.len() < 2 {
                return Err(WorkerError::InvalidPayload(
                    "bernini multi video to video requires at least two source clips \
                     (sourceClipAssetIds)."
                        .to_owned(),
                ));
            }
            clip_ids.extend(request.source_clip_asset_ids.iter().map(String::as_str));
        }
        _ => {}
    }
    if mode == "ads2v" {
        let ref_clip_id = request.reference_clip_asset_id.as_deref().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "bernini ads2v requires a reference video (referenceClipAssetId).".to_owned(),
            )
        })?;
        clip_ids.push(ref_clip_id);
    }

    let mut conditioning = Vec::new();
    for clip_id in clip_ids {
        let frames = extract_clip_frames(
            api,
            settings,
            job,
            &request.project_id,
            project_path,
            clip_id,
            request.width,
            request.height,
            wan_frame_count(request.raw_frame_count()),
        )
        .await?;
        conditioning.push(Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        });
    }

    // Subject reference images → MultiReference (r2v / rv2v / ads2v).
    let needs_refs = matches!(
        mode,
        "reference_to_video" | "reference_video_to_video" | "ads2v"
    );
    if needs_refs {
        if request.reference_asset_ids.is_empty() {
            return Err(WorkerError::InvalidPayload(format!(
                "bernini {} requires at least one reference image (referenceAssetIds).",
                request.mode.replace('_', " ")
            )));
        }
        let mut images = Vec::with_capacity(request.reference_asset_ids.len());
        for asset_id in &request.reference_asset_ids {
            images.push(crate::image_jobs::load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?);
        }
        conditioning.push(Conditioning::MultiReference { images });
    }

    Ok(conditioning)
}

/// Real candle Bernini video generation (sc-10997 / epic 6562): build the `VideoGenInput` and run the
/// shared [`generate_video`] path. The SceneWorks mode resolves to the engine `video_mode` task
/// ([`candle_bernini_engine_video_mode`]) and the source media into the planner conditioning
/// ([`resolve_candle_bernini_conditioning`]) — empty for t2v, one or more `VideoClip`s for
/// v2v/mv2v/rv2v/ads2v, and `MultiReference` for r2v/rv2v/ads2v. The converted `SceneWorks/bernini`
/// snapshot descends into the requested quant tier subfolder (`bf16/`|`q8/`|`q4/`, sc-11003) via the
/// shared [`crate::image_jobs::resolve_candle_bernini_tier_dir_and_quant`] so video + still load from
/// the same tier: no explicit `mlxQuantize` ⇒ bf16 dense, `:4`|`:8` opt into the packed tiers. No LoRA
/// (the engine reports `supports_lora=false`); steps / guidance stay at the engine defaults. Frame
/// count uses the Wan 1-mod-4 stride (the renderer is Wan).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn generate_candle_bernini(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_candle_bernini_conditioning(api, settings, job, request, project_path).await?;
    // Select the published tier subfolder + matching load quant (sc-11003): parse `mlxQuantize` (int
    // or numeric string) the same way the still lane does, defaulting to bf16 dense.
    let tier_bits = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()));
    let (model_dir, quant) =
        crate::image_jobs::resolve_candle_bernini_tier_dir_and_quant(settings, tier_bits)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(candle_bernini_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Real MLX SCAIL-2 generation (macOS, via mlx-gen-scail2, epic 5439 / sc-5448): end-to-end character
// animation — a reference character image + a driving video → an animated clip of the character
// performing the driving motion. The worker paints the color-coded segmentation masks the engine
// needs from native SAM3 (no user masks): the reference image and the driving frames are each
// segmented (every person → a distinct palette color, left-to-right) and painted onto the
// whose-world-to-keep background (animation: driving bg black, ref bg white). Conditioning =
// `Reference` (the character) + `Mask` (its color mask) + `ControlClip{frames, mask}` (driving video +
// per-frame color masks). `replace_person` (cross-identity, replace_flag=true) is the same engine,
// wired in sc-5452; multi-character (paired ref+mask) awaits the engine request-contract extension.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SCAIL-2 asset.
#[cfg(target_os = "macos")]
const SCAIL2_ADAPTER: &str = "mlx_scail2";

/// SceneWorks SCAIL-2 model id → mlx-gen registry id, or `None` if `model` is not SCAIL-2.
#[cfg(target_os = "macos")]
fn scail2_engine_id(model: &str) -> Option<&'static str> {
    (model == "scail2_14b").then_some("scail2_14b")
}

/// Whether the linked SCAIL-2 engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn scail2_available(_request: &VideoRequest, settings: &Settings) -> bool {
    resolve_scail2_model_dir(settings).is_ok()
}

/// Resolve the SCAIL-2 MLX snapshot dir: env override (`SCENEWORKS_MLX_SCAIL2_DIR`) → app-managed
/// `<data>/models/mlx/scail2` → the turnkey download-on-first-use `SceneWorks/scail2-mlx` snapshot
/// (mirrors `resolve_bernini_model_dir`). Errors clearly if none is present (no stub fallback).
#[cfg(target_os = "macos")]
fn resolve_scail2_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Some(dir) = local_mlx_dir(settings, "SCENEWORKS_MLX_SCAIL2_DIR", "scail2") {
        return Ok(dir);
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, "SceneWorks/scail2-mlx") {
        return Ok(dir);
    }
    Err(WorkerError::InvalidPayload(format!(
        "scail2: no MLX weights found. Download the turnkey SceneWorks/scail2-mlx snapshot via the \
         Model Manager, set $SCENEWORKS_MLX_SCAIL2_DIR, or place a converted snapshot at {}.",
        settings
            .data_dir
            .join("models")
            .join("mlx")
            .join("scail2")
            .display(),
    )))
}

/// MLX quantization for a SCAIL-2 load: Q4 default, Q8 opt-in via the advanced `mlxQuantize: 8`
/// control, explicit `<= 0` ⇒ bf16 (power users with ample RAM). Delegates to the shared
/// [`resolve_mlx_dense_quant`] (sc-8830) — the byte-identical `resolve_bernini_quant` twin.
#[cfg(target_os = "macos")]
fn resolve_scail2_quant(request: &VideoRequest) -> Option<Quant> {
    resolve_mlx_dense_quant(request)
}

/// The turnkey SceneWorks SCAIL-2 MLX repo (sc-9944, epic 8506). Hosts the quant matrix as
/// self-contained tier subdirs `q4/` (default) + `q8/` + `bf16/`, each a COMPLETE snapshot (the DiT
/// `dit.safetensors` at the tier's precision + the shared dense Wan2.1 z16 VAE + UMT5 T5 encoder +
/// open-CLIP ViT-H/14 visual tower + tokenizer + `config.json` carrying the quant manifest). This
/// augments the flat Q4 layout the repo shipped before (a single pre-quantized snapshot, sc-5445);
/// the worker now descends into the chosen tier so a pre-packed snapshot loads with no install-time
/// convert peak. The flat root files stay for back-compat with already-shipped workers that resolve
/// the repo root; a new worker only ever resolves the tier subdirs.
#[cfg(target_os = "macos")]
const SCAIL2_REPO: &str = "SceneWorks/scail2-mlx";

/// Pinned revision for [`SCAIL2_REPO`] (mirrors [`WAN_T2V_14B_REVISION`]). The repo is a hard-coded
/// const — no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches — so pulling
/// the mutable `main` branch would let an upstream re-push silently swap a checkpoint we load. This is
/// the commit that added the `q4/`/`q8/`/`bf16/` tier subdirs (sc-9944); the native downloader still verifies
/// each file's own hash on download.
#[cfg(target_os = "macos")]
const SCAIL2_REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";

/// The files that make a SCAIL-2 tier subdir COMPLETE — the six files the snapshot loader opens
/// (`mlx-gen-scail2` `generate.rs`): the DiT plus the shared dense Wan2.1 z16 VAE, UMT5 T5 encoder,
/// open-CLIP ViT-H/14 visual tower, UMT5 tokenizer, and `config.json` (which carries the quant
/// manifest for `q4`/`q8`, or none for the dense `bf16` tier). A partially-downloaded tier fails this
/// so [`scail2_tier_subdir`] falls through to a smaller complete tier rather than half-loading.
#[cfg(target_os = "macos")]
const SCAIL2_TIER_FILES: &[&str] = &[
    "dit.safetensors",
    "vae.safetensors",
    "t5_encoder.safetensors",
    "clip.safetensors",
    "tokenizer.json",
    "config.json",
];

/// Map a SCAIL-2 model id to its `(quant-matrix repo, pinned revision)` for the on-demand tier fetch,
/// or `None` for a non-SCAIL-2 id. Keyed here so the whole tier-resolve/fetch path (mirroring the Wan
/// [`wan_tier_repo`] machinery) routes on `scail2_tier_repo(..).is_some()`.
#[cfg(target_os = "macos")]
fn scail2_tier_repo(model: &str) -> Option<(&'static str, &'static str)> {
    (model == "scail2_14b").then_some((SCAIL2_REPO, SCAIL2_REVISION))
}

/// The SCAIL-2 quant-matrix tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a repo missing the preferred subdir still loads (mirrors
/// [`wan_tier_order`]): `<= 0` ⇒ `bf16`; `>= 8` ⇒ `q8`; and — with NO explicit `mlxQuantize` OR an
/// explicit `q4` — the **`q4`** default (q4-first). `bf16` stays OUT of the default order, so a default
/// job never pulls the huge dense tier by accident.
///
/// The video lane keeps the pre-sc-10726 q4-first default (sc-10859), NOT epic 10721's app-wide Q8 —
/// see [`wan_tier_order`] for the rationale (no MLX video Q8 lever ⇒ a silent Q8 only risks a
/// video-runtime OOM). This also lets the resolver drop its old `mlxQuantize`-presence check: the
/// no-explicit-pick default and an explicit `q4` now BOTH resolve q4-first, so [`resolve_scail2_quant`]
/// mapping both to `Some(Quant::Q4)` is exactly what we want.
#[cfg(target_os = "macos")]
fn scail2_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    match resolve_scail2_quant(request) {
        None => &["bf16", "q8", "q4"],
        Some(Quant::Q8) => &["q8", "q4"],
        // No explicit pick OR an explicit `q4` ⇒ q4-first (sc-10859 video carve-out).
        _ => &["q4", "q8"],
    }
}

/// Whether `dir` is a COMPLETE self-contained SCAIL-2 tier snapshot ([`SCAIL2_TIER_FILES`]). A
/// partially-downloaded tier fails this so [`scail2_tier_subdir`] falls through to a smaller complete
/// tier rather than half-loading.
#[cfg(target_os = "macos")]
fn scail2_tier_is_complete(dir: &Path) -> bool {
    SCAIL2_TIER_FILES
        .iter()
        .all(|file| dir.join(file).is_file())
}

/// Descend a resolved SCAIL-2 quant-matrix repo `root` into the requested quant tier subdir (sc-9944,
/// epic 8506), mirroring [`wan_tier_subdir`]. Returns the first COMPLETE tier in [`scail2_tier_order`],
/// or `None` when the repo has no complete tier subdir — a legacy flat snapshot, where the caller
/// keeps the root + load-time quant.
#[cfg(target_os = "macos")]
fn scail2_tier_subdir(root: &Path, request: &VideoRequest) -> Option<PathBuf> {
    scail2_tier_order(request)
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| scail2_tier_is_complete(dir))
}

/// Resolve the SCAIL-2 `(model_dir, load-time quant)` for a generation, descending into the
/// quant-matrix tier subdir when the turnkey ships them (sc-9944). A pre-packed tier's `config.json`
/// is authoritative — [`Scail2Config::from_model_dir`] reads its `quantization` block and the DiT
/// loader builds each Linear from packed parts directly — so a resolved tier loads with `quant = None`
/// (`mlxQuantize` selects WHICH tier, never a load-time requant; the `bf16/` tier is dense, so `None`
/// ⇒ dense too). A legacy flat snapshot (no tier subdirs) keeps today's behavior: load the root and
/// quantize at load per [`resolve_scail2_quant`].
#[cfg(target_os = "macos")]
fn resolve_scail2_tier_dir_and_quant(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<(PathBuf, Option<Quant>)> {
    let root = resolve_scail2_model_dir(settings)?;
    match scail2_tier_subdir(&root, request) {
        Some(tier) => Ok((tier, None)),
        None => Ok((root, resolve_scail2_quant(request))),
    }
}

/// On-demand fetch of a non-default SCAIL-2 quant-matrix tier subdir (sc-9944, mirrors
/// [`ensure_wan_tier_present`]). The macOS default download is the lean `q4/` tier; a job that opts
/// into a heavier tier (`mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`) pulls just that subdir from the
/// FIXED [`scail2_tier_repo`] revision the first time it is requested so [`scail2_tier_subdir`] can
/// resolve it. No-op for a non-SCAIL-2 model, a `q4` (default) job, when the repo snapshot isn't
/// downloaded yet (resolve surfaces the clear error), or when the tier is already complete. Fails loud
/// on a real download error — fast, before any compute; a tier that isn't published yet stays absent so
/// resolve falls back to a smaller complete tier.
#[cfg(target_os = "macos")]
async fn ensure_scail2_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    let Some((repo, revision)) = scail2_tier_repo(&request.model) else {
        return Ok(());
    };
    let tier = match resolve_scail2_quant(request) {
        None => "bf16",
        Some(Quant::Q8) => "q8",
        // q4 default — ships with the base install, nothing to fetch on demand.
        _ => return Ok(()),
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, repo) else {
        return Ok(());
    };
    if scail2_tier_is_complete(&root.join(tier)) {
        return Ok(());
    }
    let files = vec![format!("{tier}/*")];
    crate::model_jobs::ensure_hf_files_cached(api, settings, job, repo, revision, &files)
        .await
        .map(|_| ())
}

/// Raw-settings recorded on a real MLX SCAIL-2 asset (mirrors `bernini_raw_settings`). When the
/// lightx2v lightning LoRA is applied (`lightning`, sc-5700), records the effective step-distill recipe
/// the worker dispatched — so the chosen steps/CFG/shift is inspectable on the asset, not silent
/// (mirrors `wan_raw_settings`).
#[cfg(target_os = "macos")]
fn scail2_raw_settings(request: &VideoRequest, lightning: bool) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    // The engine task the SceneWorks mode resolved to (lineage / observability).
    raw.insert(
        "scail2Task".to_owned(),
        Value::String(scail2_engine_video_mode(&request.mode).to_owned()),
    );
    if lightning {
        let (steps, guidance, shift) = scail2_sampling(request, true);
        raw.insert("scail2Lightning".to_owned(), Value::Bool(true));
        if let Some(steps) = steps {
            raw.insert("effectiveSteps".to_owned(), json!(steps));
        }
        if let Some(guidance) = guidance {
            raw.insert("effectiveGuidanceScale".to_owned(), json!(guidance));
        }
        if let Some(shift) = shift {
            raw.insert("effectiveSchedulerShift".to_owned(), json!(shift));
        }
    }
    Value::Object(raw)
}

/// Map a SceneWorks video mode to the SCAIL-2 engine `video_mode` task string. `replace_person`
/// (cross-identity, sc-5452) flips the engine `replace_flag`; everything else (`animate_character`)
/// is plain animation.
#[cfg(target_os = "macos")]
fn scail2_engine_video_mode(mode: &str) -> &'static str {
    match mode {
        "replace_person" => "replacement",
        _ => "animation",
    }
}

/// Resolve a SCAIL-2 request into the engine conditioning: load the reference character image and the
/// driving clip, segment both with native SAM3 (every person → a palette color), paint the color-coded
/// masks (animation background convention), and assemble `Reference` + `Mask` + `ControlClip`. The
/// segmentation + painting run on the blocking pool (GPU inference). The reference is
/// `referenceAssetIds[0]` (preferred) or `sourceAssetId`; the driving clip is `sourceClipAssetId`.
#[cfg(target_os = "macos")]
async fn resolve_scail2_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    // The character: a reference image (referenceAssetIds first, else the i2v sourceAssetId).
    let ref_id = request
        .reference_asset_ids
        .first()
        .map(String::as_str)
        .or(request.source_asset_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "scail2 animate_character requires a reference character image (referenceAssetIds \
                 or sourceAssetId)."
                    .into(),
            )
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        ref_id,
        project_path,
    )?;

    // The driving video → frames at the output size (the engine re-resizes internally).
    let clip_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "scail2 animate_character requires a driving video (sourceClipAssetId).".into(),
        )
    })?;
    let driving = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        clip_id,
        request.width,
        request.height,
        wan_frame_count(request.raw_frame_count()),
    )
    .await?;

    // SAM3 segmenter weights (download-on-first-use), shared by both segmentation passes.
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::ensure_segmenter_weights(settings, &context).await?;

    // Decode the engine `Image`s to `RgbImage`s for SAM3 (it normalizes RGB internally).
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    let driving_rgb: Vec<image::RgbImage> = driving
        .iter()
        .map(|f| image::RgbImage::from_raw(f.width, f.height, f.pixels.clone()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| WorkerError::InvalidPayload("scail2 driving frame is malformed".into()))?;

    // Segment + paint via the shared orchestrator (sc-8830). Animation keeps the reference's world
    // (ref bg white) and drops the driving world (driving bg black); the native SAM3 module is the
    // per-frame 1008² MLX twin.
    let (rm, rt) = (sam_model.clone(), sam_tokenizer.clone());
    assemble_scail2_animate_conditioning(
        api,
        settings,
        &job.id,
        reference,
        driving,
        move |flag| {
            let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                &rm,
                &rt,
                std::slice::from_ref(&ref_rgb),
                Some(flag),
                None,
            )?;
            crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_WHITE)
        },
        move |flag| {
            let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                &driving_rgb,
                Some(flag),
                Some(Box::new(|frame, total| {
                    tracing::debug!(event = "scail2_sam3_propagate_progress", frame, total);
                })),
            )?;
            crate::scail2_masks::paint_driving_masks(&masks, crate::scail2_masks::BG_BLACK)
        },
    )
    .await
}

/// Real MLX SCAIL-2 generation (epic 5439 / sc-5448): build the `VideoGenInput` and run the shared
/// `generate_video` path. The SceneWorks mode resolves to the engine `video_mode` task
/// ([`scail2_engine_video_mode`]) and the source media into the SAM3-painted conditioning
/// ([`resolve_scail2_conditioning`]). A user-selected SCAIL-2 LoRA and the bundled Bias-Aware DPO
/// quality LoRA install as forward-time residuals over the Q4 base ([`resolve_scail2_adapters`],
/// sc-5451 / mlx-gen #462); steps/guidance stay at the engine defaults. Frame count uses the Wan
/// 1-mod-4 stride coercion (the
/// renderer is Wan2.1); the engine stitches > 81-frame clips into overlapping segments internally.
///
/// Returns the decoded clip plus the resolved `lightning` bool so the caller records the effective
/// recipe on the asset without re-resolving the adapters (which discarded errors via
/// `unwrap_or_default`, risking a lightning-flag inconsistency — F-118).
#[cfg(target_os = "macos")]
async fn generate_scail2(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, bool)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let conditioning =
        resolve_scail2_conditioning(api, settings, job, request, project_path).await?;
    // Selecting a lightx2v diff-patch "lightning" LoRA flips the worker to the step-distill recipe
    // (8 steps, CFG off, shift 1.0) so the toggle yields the speedup; otherwise steps/guidance/shift
    // stay `None` and the engine's quality defaults stand (sc-5700).
    let adapters = resolve_scail2_adapters(settings, request)?;
    let lightning = scail2_adapters_have_lightning(&adapters);
    let (steps, guidance, scheduler_shift) = scail2_sampling(request, lightning);
    // SCAIL-2 quant matrix (sc-9944, epic 8506): the macOS default install is the lean q4 tier; a
    // q8/bf16 job fetches that subdir on demand before resolving. No-op for a q4 job or an
    // already-present tier. Then descend into the chosen tier subdir when the turnkey ships them (a
    // pre-packed tier loads with quant=None, config.json authoritative); a legacy flat snapshot keeps
    // the root + load-time quant.
    ensure_scail2_tier_present(api, settings, job, request).await?;
    let (model_dir, quant) = resolve_scail2_tier_dir_and_quant(settings, request)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps,
        guidance,
        scheduler_shift,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        adapters,
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, lightning))
}

/// Resolve a `replace_person` request into SCAIL-2 cross-identity replacement conditioning (sc-5452,
/// the **integrated** surface). Unlike the standalone `animate_character` path
/// ([`resolve_scail2_conditioning`], which segments a fresh driving clip), this reuses the masks
/// SceneWorks already computed: the saved person track (native YOLO11 → ByteTrack → SAM3,
/// corrections applied) supplies the per-frame driving masks, and the character's approved reference
/// image is the identity. Driving frames come from the source clip exactly as the Wan-VACE backend
/// loads them ([`load_source_video_frames`]), so the resampled track masks stay frame-aligned 1:1.
/// Replacement keeps the **driving** clip's world (driving mask bg white, reference mask bg black);
/// `video_mode = "replacement"` flips the engine `replace_flag`. SCAIL-2 is a full-character model —
/// it replaces the whole tracked person, so the face_only/full_person `replacementMode` knob and
/// `maskingStrength` are inert (the engine reads only the color masks). Multi-character (the extra
/// references) awaits the engine request-contract extension (sc-5583), so only the first reference
/// is used. Returns the conditioning plus the honest `replacementStatus` for the asset sidecar.
#[cfg(target_os = "macos")]
async fn resolve_scail2_replace_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<(Vec<Conditioning>, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Driving frames + their per-frame binary person masks — the same source the Wan-VACE backend
    // consumes, loaded identically so the resampled masks align 1:1 with the frames.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let driving =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let frame_total = driving.len();
    let (binary_masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frame_total,
    )?;
    // The tracked person → blue (person 0); replacement keeps the driving's world → white bg.
    let driving_masks = crate::scail2_masks::paint_track_driving_masks(
        &binary_masks,
        crate::scail2_masks::BG_WHITE,
    );

    // The character identity: the first approved reference image (multi-ref = sc-5583).
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let reference = references.into_iter().next().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        )
    })?;

    // The reference color mask: a fresh native-SAM3 pass on the reference image → the primary person
    // painted blue on a black background (replacement discards the reference's surrounding world).
    let client = crate::downloads::streaming_download_client();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SCAIL-2 canceled while fetching the SAM3 segmenter weights.",
        fresh_download: false,
    };
    let (sam_model, sam_tokenizer) =
        crate::person_segment_sam3::ensure_segmenter_weights(settings, &context).await?;
    let ref_rgb =
        image::RgbImage::from_raw(reference.width, reference.height, reference.pixels.clone())
            .ok_or_else(|| {
                WorkerError::InvalidPayload("scail2 reference image is malformed".into())
            })?;
    // Heartbeat keepalive + user cancel across the cold SAM3 parse + single-frame propagate
    // (sc-8390 / sc-8807), via the shared blocking-segment helper (sc-8830).
    let ref_mask = scail2_segment_blocking(
        api,
        settings,
        &job.id,
        "scail2 reference segment task",
        move |flag| {
            let masks = crate::person_segment_sam3::segment_all_persons_in_memory(
                &sam_model,
                &sam_tokenizer,
                std::slice::from_ref(&ref_rgb),
                Some(flag),
                None,
            )?;
            crate::scail2_masks::paint_reference_mask(&masks, crate::scail2_masks::BG_BLACK)
        },
    )
    .await?;

    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            // masking_strength / start_frame / mode are inert for SCAIL-2 (it reads only the color
            // masks); carried at neutral defaults for the shared ControlClip contract.
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    // maskingStrength is recorded as 1.0 — SCAIL-2 always does a full-character replacement, so the
    // Wan-VACE partial-mask knob does not apply.
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        1.0,
        reference_count,
        frame_total,
        SCAIL2_ADAPTER,
    );
    Ok((conditioning, status))
}

/// Real MLX SCAIL-2 cross-identity replacement (epic 5439 / sc-5452): the integrated backend behind
/// the existing `replace_person` pipeline. Builds the replacement conditioning from the saved person
/// track + character reference ([`resolve_scail2_replace_conditioning`]) and runs the shared
/// `generate_video` path with `video_mode = "replacement"` (engine `replace_flag = true`). Returns
/// the decoded video plus the honest `replacementStatus` (adapter `mlx_scail2`). Mirrors
/// [`generate_wan_vace`]'s return shape so the dispatch folds the status identically.
#[cfg(target_os = "macos")]
async fn generate_scail2_replace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let negative_prompt = non_empty_negative_prompt(request);
    let (conditioning, status) =
        resolve_scail2_replace_conditioning(api, settings, job, request, project_path).await?;
    // Same quant-matrix tier resolution as the animate path (sc-9944): fetch a toggled q8/bf16 tier on
    // demand, then descend into the chosen tier subdir (pre-packed ⇒ quant=None) or keep the flat root
    // + load-time quant for a legacy snapshot.
    ensure_scail2_tier_present(api, settings, job, request).await?;
    let (model_dir, quant) = resolve_scail2_tier_dir_and_quant(settings, request)?;
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: wan_frame_count(request.raw_frame_count()),
        fps: request.fps,
        seed: resolve_video_seed(request) as u64,
        video_mode: Some(scail2_engine_video_mode(&request.mode).to_owned()),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    Ok((decoded, status))
}

// ---------------------------------------------------------------------------
// Real MLX LTX-2.3 generation (macOS, via mlx-gen-ltx, sc-3035): T2V/I2V with
// SYNCHRONIZED AUDIO (the 2-stage distilled A/V pipeline; CFG forced 1.0). One
// engine model `ltx_2_3` serves both `ltx_2_3` + `ltx_2_3_eros` (the checkpoint dir
// selects quant via split_model.json). The Gemma-3 text encoder is resolved by the
// engine ($LTX_GEMMA_DIR / the HF cache). Audio rides the sc-3033 WAV→AAC mux path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX LTX asset.
#[cfg(target_os = "macos")]
const LTX_ADAPTER: &str = "mlx_ltx";

/// SceneWorks LTX model id → mlx-gen registry id (one engine model serves both), or
/// `None` if not an LTX family id.
#[cfg(target_os = "macos")]
fn ltx_engine_id(model: &str) -> Option<&'static str> {
    matches!(model, "ltx_2_3" | "ltx_2_3_eros").then_some("ltx_2_3")
}

/// Whether the linked LTX engine can serve this request now (resolvable weights).
#[cfg(target_os = "macos")]
fn ltx_available(request: &VideoRequest, settings: &Settings) -> bool {
    ltx_engine_id(&request.model).is_some() && resolve_ltx_model_dir(settings, request).is_ok()
}

/// The turnkey SceneWorks LTX-2.3 MLX bundle (sc-5608, epic 5594; replaces the third-party
/// `notapalindrome/ltx23-mlx-av-q4` + `mlx-community/gemma-3-12b-it-bf16` mirrors). One repo with
/// the LTX `q4/` (default) + `q8/` (opt-in) checkpoint subdirs — each the full audio+I2V component
/// set — plus the bundled `gemma/` text encoder the engine reads via `$LTX_GEMMA_DIR`.
#[cfg(target_os = "macos")]
const LTX_BUNDLE_REPO: &str = "SceneWorks/ltx-2.3-mlx";
/// Pinned revision for the fixed [`LTX_BUNDLE_REPO`] (sc-9879, F-077 follow-up). The bundle repo is a
/// hard-coded const (no manifest/payload override reaches the on-demand `q8/*` + `bf16/*` fetches), so
/// pulling the mutable `main` branch would let an upstream re-push silently swap a checkpoint we load.
/// Pin the exact commit for defense-in-depth (mirrors the SeedVR2/Real-ESRGAN pins, sc-8879/sc-9682).
/// The native downloader still verifies each file's own hash on download. Bumped to the commit that added the
/// dense `bf16/` tier (sc-8513) — a superset of the prior commit, so the q8 fetch is unaffected.
#[cfg(target_os = "macos")]
const LTX_BUNDLE_REVISION: &str = "01df27d308466533aa09d251e3aebdcc627d07eb";

/// Whether `dir` is a converted LTX snapshot **complete for the current engine** — it must
/// carry the audio `vocoder` + I2V `vae_encoder` + single `upsampler`/`vae_decoder` the
/// engine `load()` reads. Older conversions (`spatial_/temporal_upscaler_*`, no vocoder)
/// fail this, so a stale local dir is skipped in favour of the turnkey snapshot.
#[cfg(target_os = "macos")]
fn ltx_dir_is_complete(dir: &Path) -> bool {
    [
        "connector.safetensors",
        "transformer.safetensors",
        "upsampler.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]
    .iter()
    .all(|file| dir.join(file).is_file())
}

/// Whether `dir` is a complete Gemma-3 text-encoder snapshot the LTX engine can load: its
/// `config.json`, the sharded `model.safetensors.index.json`, and every shard that index maps (or a
/// lone `model.safetensors` for a single-file checkpoint). Used so the eros gemma fetch
/// ([`ensure_ltx_gemma_present`]) no-ops only when gemma is genuinely present and a half-downloaded
/// dir never shadows a re-fetch ([`resolve_ltx_eros_gemma_dir`]).
#[cfg(target_os = "macos")]
fn ltx_gemma_dir_is_complete(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    let Ok(index_raw) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) else {
        // No shard index ⇒ a single-file checkpoint is complete once its lone weights file exists.
        return dir.join("model.safetensors").is_file();
    };
    let Ok(index) = serde_json::from_str::<Value>(&index_raw) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return false;
    };
    let shards: std::collections::BTreeSet<String> = weight_map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    shards.iter().all(|shard| dir.join(shard).is_file())
}

/// Parse `advanced.mlxQuantize` (int or numeric string) → the requested bit width, if present.
#[cfg(target_os = "macos")]
fn ltx_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// Whether the request opts into the higher-quality Q8 LTX checkpoint (`advanced.mlxQuantize: 8`,
/// accepted as int or string). The default is Q4 (sc-5608).
#[cfg(target_os = "macos")]
fn ltx_wants_q8(request: &VideoRequest) -> bool {
    ltx_quant_bits(request)
        .map(|bits| bits >= 8)
        .unwrap_or(false)
}

/// Whether the request opts into the dense **bf16** LTX checkpoint (`advanced.mlxQuantize <= 0`,
/// int or string) — the ~47 GB power-user tier (sc-8513, epic 8506). Never the default: absent ⇒ Q4,
/// so the big bf16 bundle is a deliberate opt-in (mirrors [`resolve_mlx_dense_quant`]'s `<= 0` rule).
#[cfg(target_os = "macos")]
fn ltx_wants_bf16(request: &VideoRequest) -> bool {
    ltx_quant_bits(request)
        .map(|bits| bits <= 0)
        .unwrap_or(false)
}

/// The SceneWorks LTX bundle tier search order for a request — preferred tier first, then the
/// always-smaller fallback tiers so a bundle missing the preferred subdir still loads (sc-8513):
/// `mlxQuantize <= 0` ⇒ `bf16`, `>= 8` ⇒ `q8`, and — with an explicit `1..=4` OR NO explicit
/// `mlxQuantize` — the **`q4`** default (q4-first). bf16 stays OUT of the default order, so a default
/// job never loads the huge dense tier by accident.
///
/// The video lane keeps the pre-sc-10726 q4-first default (sc-10859), NOT epic 10721's app-wide Q8 —
/// see [`wan_tier_order`] for the rationale (no MLX video Q8 lever ⇒ a silent Q8 only risks a
/// video-runtime OOM). The no-explicit-pick and explicit-`q4` cases now share one q4-first arm.
#[cfg(target_os = "macos")]
fn ltx_bundle_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    if ltx_wants_bf16(request) {
        &["bf16", "q8", "q4"]
    } else if ltx_wants_q8(request) {
        &["q8", "q4"]
    } else {
        // No explicit pick OR an explicit `1..=4` ⇒ q4-first (sc-10859 video carve-out).
        &["q4", "q8"]
    }
}

/// Pick the engine-complete `bf16/`/`q8/`/`q4/` checkpoint subdir of a SceneWorks LTX bundle `root`,
/// trying `order` (preferred tier first, sc-5608/sc-8513). Returns the first **complete**
/// ([`ltx_dir_is_complete`]) subdir — so a partially-downloaded bundle falls through rather than
/// half-loading — or `None`.
#[cfg(target_os = "macos")]
fn ltx_bundle_subdir(root: &Path, order: &[&str]) -> Option<PathBuf> {
    order
        .iter()
        .map(|sub| root.join(sub))
        .find(|dir| ltx_dir_is_complete(dir))
}

/// Resolve the converted LTX MLX snapshot dir. Env override (`SCENEWORKS_MLX_LTX_DIR` /
/// `…_EROS_DIR`) → `<data>/models/mlx/<candidate>` → (base only) the turnkey SceneWorks bundle
/// [`LTX_BUNDLE_REPO`], descending into its `q4/`/`q8/` subdir. Only a dir **complete for the
/// current engine** ([`ltx_dir_is_complete`]) counts, so a stale local conversion is skipped. For
/// the base model the Q4 checkpoint is the default (`mlxQuantize: 8` prefers the Q8 one); the engine
/// reads the actual bits from `split_model.json`, so this only picks *which* dir to load.
#[cfg(target_os = "macos")]
fn resolve_ltx_model_dir(settings: &Settings, request: &VideoRequest) -> WorkerResult<PathBuf> {
    let eros = request.model == "ltx_2_3_eros";
    let env = if eros {
        "SCENEWORKS_MLX_LTX_EROS_DIR"
    } else {
        "SCENEWORKS_MLX_LTX_DIR"
    };
    if let Ok(override_dir) = std::env::var(env) {
        let path = PathBuf::from(override_dir.trim());
        if ltx_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let wants_bf16 = ltx_wants_bf16(request);
    let wants_q8 = ltx_wants_q8(request);
    let candidates: &[&str] = if eros {
        &["ltx_2_3_eros"]
    } else if wants_bf16 {
        // No local bf16 conversion id exists (install-time convert only emits Q4/Q8), so don't let a
        // local quantized dir shadow the dense turnkey tier — fall straight through to the bundle's
        // bf16/ subdir below.
        &[]
    } else if wants_q8 {
        &["ltx_2_3_base_q8", "ltx_2_3_base_q4", "ltx_2_3"]
    } else {
        &["ltx_2_3_base_q4", "ltx_2_3_base_q8", "ltx_2_3"]
    };
    for id in candidates {
        let dir = settings.data_dir.join("models").join("mlx").join(id);
        if ltx_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    // Turnkey SceneWorks bundle for the base model (sc-5608): one repo with `bf16/` + `q8/` + `q4/`
    // LTX subdirs (+ a bundled `gemma/` the engine reads via $LTX_GEMMA_DIR). Pick the preferred tier
    // subdir; the engine reads the actual bits from split_model.json, so this only selects which to
    // load.
    if !eros {
        if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) {
            if let Some(dir) = ltx_bundle_subdir(&root, ltx_bundle_tier_order(request)) {
                return Ok(dir);
            }
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: no complete converted LTX MLX weights found under {} (expected one of {candidates:?} \
         with the audio vocoder + i2v vae_encoder; or the turnkey {LTX_BUNDLE_REPO} q4/ or q8/ \
         subdir; or set ${env})",
        request.model,
        settings.data_dir.join("models").join("mlx").display(),
    )))
}

/// The Gemma-3 text encoder bundled beside a resolved LTX dir, if present: `<parent>/gemma`
/// (sc-5608). The SceneWorks bundle ships it as a sibling of the `q4/`/`q8/` checkpoint dir; a
/// local/legacy conversion has none (→ the engine falls back to the HF-cache gemma snapshot).
#[cfg(target_os = "macos")]
fn bundled_ltx_gemma_dir(model_dir: &Path) -> Option<PathBuf> {
    let gemma = model_dir.parent()?.join("gemma");
    gemma.is_dir().then_some(gemma)
}

/// Resolve the Gemma-3 text encoder bundled beside the resolved LTX dir (sc-5608), returning it so the
/// caller can thread it onto `LoadSpec::text_encoder` (sc-8827) — a fresh install is self-contained (no
/// separate `mlx-community/gemma` download) without mutating the process-global `$LTX_GEMMA_DIR` at job
/// time on the multithreaded runtime (the old `set_var` seam was unsound, F-025). Best-effort +
/// non-destructive: returns `None` when an explicit operator `$LTX_GEMMA_DIR` is set (the provider
/// reads the env var itself), and `None` when no bundled `gemma/` sibling exists
/// ([`bundled_ltx_gemma_dir`]) so the provider falls back to the HF-cache gemma snapshot.
///
/// `pub(crate)` so the LoRA trainer path reuses it (sc-9989): training resolves the TE identically to
/// inference, so a self-contained install trains without a separate `mlx-community/gemma` download.
#[cfg(target_os = "macos")]
pub(crate) fn resolve_bundled_ltx_gemma_dir(model_dir: &Path) -> Option<PathBuf> {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return None; // honor an explicit operator override (the provider reads the env var).
    }
    bundled_ltx_gemma_dir(model_dir)
}

/// Resolve the Gemma-3 text encoder for an **eros** generation. Unlike the base model — whose turnkey
/// bundle ships `gemma/` beside its checkpoint ([`resolve_bundled_ltx_gemma_dir`]) — the eros install
/// is a bare local conversion under `models/mlx/ltx_2_3_eros/` with no bundled TE, so gemma is
/// provisioned separately ([`ensure_ltx_gemma_present`]) and resolved here: a `models/mlx/gemma`
/// sibling of the checkpoint (the `<parent>/gemma` convention [`bundled_ltx_gemma_dir`] already uses)
/// first, then the fetched [`LTX_BUNDLE_REPO`] snapshot's `gemma/`. Returns `None` when an operator
/// `$LTX_GEMMA_DIR` is set (the provider reads the env var) or nothing complete is on disk (the
/// provider surfaces the clear "set LTX_GEMMA_DIR" error) — a partial dir never wins.
#[cfg(target_os = "macos")]
fn resolve_ltx_eros_gemma_dir(settings: &Settings, model_dir: &Path) -> Option<PathBuf> {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return None; // honor an explicit operator override (the provider reads the env var).
    }
    if let Some(sibling) = bundled_ltx_gemma_dir(model_dir) {
        if ltx_gemma_dir_is_complete(&sibling) {
            return Some(sibling);
        }
    }
    let bundle_gemma = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO)?.join("gemma");
    ltx_gemma_dir_is_complete(&bundle_gemma).then_some(bundle_gemma)
}

/// On-demand fetch of the bundle's `q8/` subdir (sc-5679). The macOS default download is lean
/// (`q4/` + `gemma/`); when a job opts into Q8 ([`ltx_wants_q8`]) and the bundle's `q8/` isn't already
/// complete, pull just `q8/*` from [`LTX_BUNDLE_REPO`] into the HF cache so [`resolve_ltx_model_dir`]
/// can load it. Base model only (eros has its own single-dir conversion). No-op when Q8 isn't
/// requested, the bundle snapshot isn't downloaded yet (resolve surfaces the clear "download the
/// bundle" error), or `q8/` is already present. Fails loud on a real download error — fast, before
/// any compute; a `q8/` tier that isn't published yet stays absent so resolve falls back to Q4.
/// Mirrors the eros [`ensure_ltx_upscaler_cached`] on-demand fetch.
#[cfg(target_os = "macos")]
async fn ensure_ltx_q8_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model == "ltx_2_3_eros" || !ltx_wants_q8(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) else {
        return Ok(());
    };
    if ltx_dir_is_complete(&root.join("q8")) {
        return Ok(());
    }
    let files = vec!["q8/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Fetch the SceneWorks LTX bundle's dense `bf16/` subdir on demand (sc-8513, epic 8506). The macOS
/// default download is lean (`q4/` + `gemma/`); a bf16 job ([`ltx_wants_bf16`]) pulls the ~47 GB
/// `bf16/*` from the FIXED [`LTX_BUNDLE_REVISION`] the first time it is requested. No-op for eros, for
/// non-bf16 jobs, or when `bf16/` is already complete. Mirrors [`ensure_ltx_q8_present`].
#[cfg(target_os = "macos")]
async fn ensure_ltx_bf16_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model == "ltx_2_3_eros" || !ltx_wants_bf16(request) {
        return Ok(());
    }
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) else {
        return Ok(());
    };
    if ltx_dir_is_complete(&root.join("bf16")) {
        return Ok(());
    }
    let files = vec!["bf16/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// Ensure the Gemma-3 text encoder an **eros** generation needs is on disk (the eros gate over
/// [`ensure_ltx_bundle_gemma_present`]). No-op for the base model, which bundles gemma with its
/// turnkey checkpoint. Called just before resolving the LTX text encoder so an eros job that was
/// installed before install-time provisioning existed still self-heals on first generation.
#[cfg(target_os = "macos")]
async fn ensure_ltx_gemma_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if request.model != "ltx_2_3_eros" {
        return Ok(());
    }
    ensure_ltx_bundle_gemma_present(api, settings, job).await
}

/// Ensure the eros Gemma-3 text encoder is on disk, fetching the bundle's `gemma/` on demand. The
/// eros install produces a bare converted checkpoint under `models/mlx/ltx_2_3_eros/` with no bundled
/// TE (unlike the base turnkey bundle, which ships `gemma/` beside its `q4/`), so without this an
/// eros generation dead-ends on "gemma snapshot not found". Pulls just `gemma/*` (~24 GB) from the
/// FIXED [`LTX_BUNDLE_REVISION`] — the same SceneWorks re-host the base model uses, so no separate
/// `mlx-community/gemma-3-12b-it-bf16` download. No-op when an operator `$LTX_GEMMA_DIR` is set, when
/// a local `models/mlx/gemma` sibling is already complete, or when the bundle snapshot's `gemma/` is
/// already complete. `pub(crate)` so the eros convert job provisions gemma at install time
/// ([`crate::model_jobs::run_model_convert_job`]) — the generation path is the self-healing backstop.
/// Mirrors [`ensure_ltx_q8_present`].
#[cfg(target_os = "macos")]
pub(crate) async fn ensure_ltx_bundle_gemma_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return Ok(());
    }
    // A complete `<data>/models/mlx/gemma` sibling already satisfies the eros resolver — nothing to
    // fetch (also short-circuits an operator who provisioned gemma there by hand).
    let local_sibling = settings.data_dir.join("models").join("mlx").join("gemma");
    if ltx_gemma_dir_is_complete(&local_sibling) {
        return Ok(());
    }
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO) {
        if ltx_gemma_dir_is_complete(&root.join("gemma")) {
            return Ok(());
        }
    }
    let files = vec!["gemma/*".to_owned()];
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        LTX_BUNDLE_REPO,
        LTX_BUNDLE_REVISION,
        &files,
    )
    .await
    .map(|_| ())
}

/// LoRAs for an LTX generation: the manifest-declared auto distill LoRA (when present) followed by
/// the user LoRAs (sc-3035). A model that declares `mlx.autoDistillLora` (10Eros) is NOT
/// pre-distilled, so its distill LoRA must be injected at runtime with per-pass strengths or its
/// video degrades to noise — see [`resolve_ltx_distill_adapter`]. Every user LoRA applies at a
/// uniform per-pass strength (`pass_scales` left `None` → the engine uses `scale` on every distilled
/// stage). peft LoKr allowed (engine residual), LyCORIS rejected.
#[cfg(target_os = "macos")]
fn resolve_ltx_adapters(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len() + 1);
    // The auto distill LoRA is the model's base recipe (per-pass 1.0/0.4); user LoRAs stack on top.
    if let Some(distill) = resolve_ltx_distill_adapter(settings, request)? {
        specs.push(distill);
    }
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = resolve_lora_file(
            settings,
            path,
            crate::image_jobs::declared_adapter_file(lora),
        )?;
        let kind = classify_adapter(&file)?;
        specs.push(AdapterSpec::new(file, lora_scale(lora), kind));
    }
    Ok(specs)
}

/// The auto-injected per-pass distill LoRA for an LTX model that declares `mlx.autoDistillLora`
/// (`ltx_2_3_eros` today), or `None` when the model declares none or the user opted out via
/// `advanced.useDistillLora = false`. 10Eros's base checkpoint is not pre-distilled, so without this
/// LoRA its MLX video collapses to noise a few frames in (the manifest documents that exact symptom).
///
/// The LoRA is the `coRequisite` `resources.distilledLora` (the cond_safe variant, sc-9696), so it
/// installs alongside the checkpoint and is resolved from the HF cache here. Strengths come from
/// `mlx.autoDistillLora` (`stage1Strength` full first pass / `stage2Strength` reduced spatial-upscale
/// pass — TenStrip's guidance for rank<=72 cond_safe LoRAs), overridable via
/// `advanced.distillStage1Strength` / `distillStage2Strength`, and applied as
/// `pass_scales = [stage1, stage2]` (the engine's LTX per-pass feature, sc-2687). Declared-but-missing
/// fails with an actionable error rather than silently producing noise.
///
/// Ported from the deleted Python `MlxVideoAdapter` (b821d74e): the injection was lost when video
/// generation moved to the Rust worker in the sc-3037 cutover, which is why 10Eros regressed to noise.
#[cfg(target_os = "macos")]
fn resolve_ltx_distill_adapter(
    settings: &Settings,
    request: &VideoRequest,
) -> WorkerResult<Option<AdapterSpec>> {
    let Some(auto) = request
        .model_manifest_entry
        .get("mlx")
        .and_then(Value::as_object)
        .and_then(|mlx| mlx.get("autoDistillLora"))
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    // Opt-out knob (default on): the distill LoRA is a required runtime component for these models.
    let enabled = request
        .advanced
        .get("useDistillLora")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if !enabled {
        return Ok(None);
    }
    let stage1 = advanced::f32(
        &request.advanced,
        "distillStage1Strength",
        auto.get("stage1Strength")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32,
    );
    let stage2 = advanced::f32(
        &request.advanced,
        "distillStage2Strength",
        auto.get("stage2Strength")
            .and_then(Value::as_f64)
            .unwrap_or(0.4) as f32,
    );
    // The distill LoRA repo/file live in `resources.distilledLora` (the recommended cond_safe variant).
    let (repo, file) = request
        .model_manifest_entry
        .get("resources")
        .and_then(Value::as_object)
        .and_then(|res| res.get("distilledLora"))
        .and_then(Value::as_object)
        .and_then(|d| {
            Some((
                d.get("repo").and_then(Value::as_str)?,
                d.get("file").and_then(Value::as_str)?,
            ))
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "This model declares mlx.autoDistillLora but resources.distilledLora is missing its \
                 repo/file, so the distill LoRA cannot be resolved."
                    .to_owned(),
            )
        })?;
    // The LoRA is a download co-requisite (sc-9696), so it is expected in the HF cache. Fail with an
    // actionable message if it is absent rather than silently degrading the output to noise.
    let path = huggingface_snapshot_dir(&settings.data_dir, repo)
        .map(|dir| dir.join(file))
        .filter(|candidate| candidate.is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "The required distill LoRA for this model is not installed ({repo}/{file}). \
                 Re-download the model to fetch its co-requisite distill LoRA."
            ))
        })?;
    let kind = classify_adapter(&path)?;
    Ok(Some(
        AdapterSpec::new(path, stage1, kind).with_pass_scales(vec![stage1, stage2]),
    ))
}

/// Optional I2V conditioning for LTX: a `source_asset_id` → a single `Reference` image
/// (image→video); absent → pure text→video. `first_last_frame` → two `Keyframe`s (sc-3055).
/// (Audio is produced either way.)
#[cfg(target_os = "macos")]
fn resolve_ltx_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if request.mode == "first_last_frame" {
        return resolve_keyframe_conditioning(settings, request, project_path);
    }
    match request.source_asset_id.as_deref() {
        Some(asset_id) => {
            let image = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                asset_id,
                project_path,
            )?;
            // Pre-fit the starting image to the output W×H by the chosen crop/pad mode
            // (sc-6139) — without this the engine resizes it internally = stretch. Reuses
            // the image-edit lane's helper; a pre-fit-to-exact-dims reference is a no-op
            // for any further internal resize.
            let image = crate::image_jobs::fit_engine_image(
                image,
                request.width,
                request.height,
                &request.fit_mode,
            )?;
            Ok(vec![Conditioning::Reference {
                image,
                strength: None,
            }])
        }
        None => Ok(Vec::new()),
    }
}

/// Read an `advanced` boolean flag (JSON bool), default `false` (Python `bool(.get(k))`).
/// First/last-frame conditioning (sc-3055 cutover): two [`Conditioning::Keyframe`]s — the source
/// image pinned at latent frame 0 and the last-frame image at latent frame `-1` (the engine's
/// Python-style negative-from-end index, so the worker needs no latent-frame math; the engine
/// bounds-checks it). Mirrors the torch `_ltx_conditioning_images` first_last_frame path: first @
/// `imageConditioningStrength`, last @ `lastFrameConditioningStrength` (both default 1.0 = fully
/// pinned). Shared by LTX (`ltx_2_3`) and Wan TI2V-5B (`wan_2_2`), the engines whose providers
/// advertise `Keyframe`. `imageFrameIndex` (default 0) is forwarded as the first keyframe's latent
/// index — for the universal FLF case (0) latent 0 == output 0.
#[cfg(target_os = "macos")]
fn resolve_keyframe_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let first_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let last_id = request.last_frame_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "first_last_frame requires a last-frame image (lastFrameAssetId).".to_owned(),
        )
    })?;
    let first = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        first_id,
        project_path,
    )?;
    let last = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        last_id,
        project_path,
    )?;
    // Fit both keyframes to the output W×H by the chosen crop/pad mode (sc-6139) so a
    // square first/last frame letterboxes (pad) or fills+trims (crop) into an off-aspect
    // clip instead of the engine stretching each internally.
    let first = crate::image_jobs::fit_engine_image(
        first,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    let last = crate::image_jobs::fit_engine_image(
        last,
        request.width,
        request.height,
        &request.fit_mode,
    )?;
    Ok(vec![
        Conditioning::Keyframe {
            image: first,
            frame_idx: advanced::i32(&request.advanced, "imageFrameIndex", 0),
            strength: advanced::f32(&request.advanced, "imageConditioningStrength", 1.0),
        },
        Conditioning::Keyframe {
            image: last,
            frame_idx: -1,
            strength: advanced::f32(&request.advanced, "lastFrameConditioningStrength", 1.0),
        },
    ])
}

/// Whether the job's LoRA set includes an IC-LoRA — the in-context conditioning adapter the
/// LTX extend/bridge keyframe-append path needs (without it the appended clip tokens are inert,
/// per the engine `apply_ltx_adapters` seam). Port of the torch `lora_looks_like_ic_lora`
/// (lora_adapters.py): an explicit `icLora`/`isIcLora` flag, a `conditioningRole: ic_lora`, or an
/// "ic-lora" / "ltx-2-3-ic-" marker anywhere in the id / name / path / file list. The IC-LoRA is a
/// user-installed LoRA flowing through `request.loras` (not an auto-provisioned fixed repo), so it
/// rides the existing [`resolve_ltx_adapters`] seam with no new adapter-loading code.
#[cfg(target_os = "macos")]
fn loras_contain_ic_lora(loras: &[Value]) -> bool {
    loras.iter().any(lora_looks_like_ic_lora)
}

#[cfg(target_os = "macos")]
fn lora_looks_like_ic_lora(lora: &Value) -> bool {
    let Some(obj) = lora.as_object() else {
        // A bare string lora id: sniff the string itself.
        return lora
            .as_str()
            .map(|id| ic_lora_marker(&id.to_lowercase().replace('_', "-")))
            .unwrap_or(false);
    };
    if obj.get("icLora") == Some(&Value::Bool(true))
        || obj.get("isIcLora") == Some(&Value::Bool(true))
    {
        return true;
    }
    if let Some(role) = obj.get("conditioningRole").and_then(Value::as_str) {
        if role.trim().to_lowercase().replace('-', "_") == "ic_lora" {
            return true;
        }
    }
    let source = obj.get("source").and_then(Value::as_object);
    // Gather every id/name/path/file string the torch heuristic inspects.
    let mut haystacks: Vec<String> = Vec::new();
    for key in [
        "id",
        "loraId",
        "name",
        "displayName",
        "installedPath",
        "sourcePath",
        "path",
    ] {
        if let Some(value) = obj.get(key).and_then(Value::as_str) {
            haystacks.push(value.to_owned());
        }
    }
    if let Some(source) = source {
        for key in ["repo", "file", "path"] {
            if let Some(value) = source.get(key).and_then(Value::as_str) {
                haystacks.push(value.to_owned());
            }
        }
    }
    // `files` (or `source.files`) may be a list or a single string.
    let files = source
        .and_then(|s| s.get("files"))
        .or_else(|| obj.get("files"));
    match files {
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(value) = item.as_str() {
                    haystacks.push(value.to_owned());
                }
            }
        }
        Some(Value::String(value)) => haystacks.push(value.clone()),
        _ => {}
    }
    let text = haystacks.join(" ").to_lowercase().replace('_', "-");
    ic_lora_marker(&text)
}

/// The torch `lora_looks_like_ic_lora` text test (already `_`→`-` normalised + lowercased).
#[cfg(target_os = "macos")]
fn ic_lora_marker(text: &str) -> bool {
    text.contains("ic-lora") || text.contains("ltx-2-3-ic-")
}

/// Build the in-context [`Conditioning::VideoClip`] set for extend_clip / video_bridge (sc-3522).
/// Source-of-truth = torch `_ltx_video_conditioning` (video_adapters.py) + the engine consumer
/// `runtime_macos::providers::ltx::build_clips`: each source clip's frames are appended as IC-LoRA in-context tokens
/// at an output **latent** frame index, with a `1 − strength` denoise mask.
/// - **extend_clip** → one clip pinned at latent frame `0`, strength `videoConditioningStrength`.
/// - **video_bridge** → a left clip at `0` (strength `videoConditioningStrength`) + a right clip at
///   latent frame `-1` (the engine's negative-from-end index, `lf + idx`, so the worker needs no
///   latent-frame math), strength `bridgeRightVideoConditioningStrength`.
///
/// Both strengths default to `1.0` (fully pinned), mirroring the torch `_advanced_float` defaults.
#[cfg(target_os = "macos")]
fn build_video_clip_conditioning(
    request: &VideoRequest,
    left_frames: Vec<Image>,
    right_frames: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let mut conditioning = vec![Conditioning::VideoClip {
        frames: left_frames,
        frame_idx: 0,
        strength: advanced::f32(&request.advanced, "videoConditioningStrength", 1.0),
    }];
    if request.mode == "video_bridge" {
        let right = right_frames.ok_or_else(|| {
            WorkerError::InvalidPayload(
                "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                    .to_owned(),
            )
        })?;
        conditioning.push(Conditioning::VideoClip {
            frames: right,
            frame_idx: -1,
            strength: advanced::f32(
                &request.advanced,
                "bridgeRightVideoConditioningStrength",
                1.0,
            ),
        });
    }
    Ok(conditioning)
}

/// Resolve an asset id to its on-disk media file path (the source clip mp4), mirroring the asset
/// lookup in [`load_reference_image`] but returning the path for ffmpeg frame extraction (the
/// Rust equivalent of the torch `source_asset_media_path`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_clip_media_path(
    settings: &Settings,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<PathBuf> {
    let asset = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("source clip asset {asset_id} has no media path"))
        })?;
    // file.path is sidecar-sourced (user-editable on disk), so guard it through
    // safe_project_path instead of a bare join so a poisoned sidecar can't escape
    // the project to read an arbitrary file as the source clip (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    if !path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip file is missing for asset {asset_id}: {}",
            path.display()
        )));
    }
    Ok(path)
}

/// Decode the first `count` frames of a source clip into [`Image`]s for in-context conditioning.
/// Mirrors the torch reference `decode_video_by_frame(starting_frame=0, frame_cap=num_frames)` /
/// `video_preprocess` (ltx_pipelines): **sequential** frames from the start at the clip's native
/// cadence (no fps resample), fit to the output `width`×`height` by contain+pad (letterbox,
/// `FRAME_PAD_COLOR`) so a clip whose aspect differs from the output is not distorted (sc-6229;
/// the engine `build_clips` LANCZOS-downsizes each frame to stage-1 half-res, so this only bounds
/// memory). `count` is the
/// generation's snapped frame count (`8k+1`); a clip shorter than `count` yields fewer frames,
/// which the engine VAE encode accepts. Extracted via the shared [`run_ffmpeg`] (binary
/// resolution + heartbeat/cancel), then loaded off the async runtime.
///
/// Shared by the macOS MLX Bernini conditioning and the candle Bernini VIDEO lane (sc-10997): both
/// resolve arbitrary source-clip asset ids (mv2v supplies several via `sourceClipAssetIds`, ads2v
/// appends the reference video) into planner `VideoClip` conditioning, so the per-asset-id loader is
/// gated to both lanes (unlike `load_source_video_frames`, which reads only the request's single
/// `sourceClipAssetId`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn extract_clip_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: u32,
) -> WorkerResult<Vec<Image>> {
    let clip_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    let frames_dir = project_path
        .join("assets")
        .join(".cond_clips")
        .join(Uuid::new_v4().simple().to_string());
    tokio::fs::create_dir_all(&frames_dir).await?;
    let pattern = frames_dir.join("frame_%05d.png");
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let result = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            clip_path.display().to_string(),
            // Contain+pad (letterbox) to the output dims so a clip whose aspect differs from the
            // requested W×H is not stretched (sc-6229); reuses the `FRAME_PAD_COLOR` recipe. The
            // engine re-resizes to stage-1 half-res, so this only bounds the extracted footprint.
            "-vf".to_owned(),
            format!(
                "scale={width}:{height}:force_original_aspect_ratio=decrease,\
                 pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24"
            ),
            // First `count` decoded frames, sequential from the start at native cadence.
            "-frames:v".to_owned(),
            count.to_string(),
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    // Load the extracted PNGs (sorted by frame index) into `Image`s, off the async runtime.
    let load = async {
        result?;
        let dir = frames_dir.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                .collect();
            paths.sort();
            let mut frames = Vec::with_capacity(paths.len());
            for path in paths {
                let decoded = crate::image_decode::decode_image_any(&path)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "conditioning frame {}: {error}",
                            path.display()
                        ))
                    })?
                    .to_rgb8();
                frames.push(Image {
                    width: decoded.width(),
                    height: decoded.height(),
                    pixels: decoded.into_raw(),
                });
            }
            Ok(frames)
        })
        .await
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?
    };
    let frames = load.await;
    // Best-effort cleanup of the scratch frame dir regardless of outcome.
    let _ = tokio::fs::remove_dir_all(&frames_dir).await;
    let frames = frames?;
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip {asset_id} produced no decodable frames for conditioning"
        )));
    }
    Ok(frames)
}

/// Resolve extend_clip / video_bridge into the in-context [`Conditioning::VideoClip`] set (sc-3522).
/// Requires an installed IC-LoRA (the keyframe-append adapter) — mirrors the torch gate
/// (`_uses_ic_lora_pipeline` + the "requires at least one installed LTX-compatible LoRA" error),
/// since without it the appended clip tokens are inert. Then decodes each source clip's first
/// `num_frames` frames and builds the clips. `num_frames` is the generation's snapped frame count,
/// the same value [`generate_ltx`] passes to the engine.
#[cfg(target_os = "macos")]
async fn resolve_video_clip_conditioning(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    if !loras_contain_ic_lora(&request.loras) {
        return Err(WorkerError::InvalidPayload(format!(
            "{} requires an installed IC-LoRA (in-context conditioning adapter) — add an \
             LTX IC-LoRA to the selected preset; without it the source-clip conditioning is inert.",
            request.mode.replace('_', " ")
        )));
    }
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let num_frames = ltx_frame_count(request.raw_frame_count());
    let left_frames = extract_clip_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        num_frames,
    )
    .await?;
    let right_frames = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            extract_clip_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                num_frames,
            )
            .await?,
        )
    } else {
        None
    };
    build_video_clip_conditioning(request, left_frames, right_frames)
}

/// Raw-settings recorded on a real MLX LTX asset (`advanced` knobs + real-inference markers).
#[cfg(target_os = "macos")]
fn ltx_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    Value::Object(raw)
}

/// Resolve an LTX request into a [`VideoGenInput`] and run it (sc-3035). Distilled 2-stage
/// → no negative prompt / guidance (CFG 1.0); quant is checkpoint-driven (`None`); frames
/// snap to `8k+1`; `advanced.noAudio` → `video_mode = "no_audio"` (full A/V denoise, audio
/// decode skipped); prompt-enhance + per-pass LoRA flow through.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_ltx(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let video_mode = advanced::bool(&request.advanced, "noAudio").then(|| "no_audio".to_owned());
    let enhance_max_tokens = request
        .advanced
        .get("enhanceMaxTokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let enhance_temperature = request
        .advanced
        .get("enhanceTemperature")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    // extend_clip / video_bridge build in-context VideoClip conditioning from decoded source
    // clips (async ffmpeg extraction); every other mode resolves keyframe/reference conditioning
    // synchronously from images.
    let conditioning = match request.mode.as_str() {
        "extend_clip" | "video_bridge" => {
            resolve_video_clip_conditioning(api, settings, job, request, project_path).await?
        }
        _ => resolve_ltx_conditioning(settings, request, project_path)?,
    };
    // The macOS default download is lean (q4 + gemma); a Q8 / bf16 job fetches the bundle's q8/ or
    // bf16/ on demand before resolving (sc-5679 / sc-8513). No-op unless that tier is requested and
    // its subdir is absent.
    ensure_ltx_q8_present(api, settings, job, request).await?;
    ensure_ltx_bf16_present(api, settings, job, request).await?;
    // The eros install ships no bundled TE, so provision the bundle's `gemma/` on demand (no-op for
    // the base model, which bundles gemma with its checkpoint). Self-heals installs that predate this.
    ensure_ltx_gemma_present(api, settings, job, request).await?;
    let model_dir = resolve_ltx_model_dir(settings, request)?;
    // Thread the Gemma-3 text encoder onto the LoadSpec (sc-8827, was `$LTX_GEMMA_DIR`). Base: the
    // SceneWorks bundle subdir's sibling `gemma/`. Eros: the separately-provisioned gemma
    // ([`ensure_ltx_gemma_present`]) — a `models/mlx/gemma` sibling or the bundle snapshot's `gemma/`.
    // `None` ⇒ the engine falls back to the HF-cache gemma snapshot.
    let text_encoder_dir = if request.model == "ltx_2_3_eros" {
        resolve_ltx_eros_gemma_dir(settings, &model_dir)
    } else {
        resolve_bundled_ltx_gemma_dir(&model_dir)
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant: None,
        adapters: resolve_ltx_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: ltx_frame_count(request.raw_frame_count()),
        fps: request.fps,
        steps: None,
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        control_scale: None,
        video_mode,
        enhance_prompt: advanced::bool(&request.advanced, "enhancePrompt"),
        use_uncensored_enhancer: advanced::bool(&request.advanced, "useUncensoredEnhancer"),
        enhance_max_tokens,
        enhance_temperature,
        text_encoder_dir,
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Mochi 1 (epic 1788 / sc-11992): a 10B AsymmDiT text-to-video model with a 6×-temporal AsymmVAE,
// served natively on BOTH backends — `mlx-gen-mochi` (macOS) and `candle-gen-mochi` (Windows/Linux
// CUDA). Two facts shape this whole block, and both DEVIATE from the LTX template it otherwise
// mirrors:
//
//  1. ONE engine id, TWO backends. Both providers register `MODEL_ID = "mochi_1"` (unlike LTX's
//     `ltx_2_3` + `ltx_2_3_distilled` split), so `mochi_engine_id` is lane-agnostic.
//  2. ONE repo, TWO backends. Every tier lives in `SceneWorks/mochi-1-mlx` with no `platforms` tag,
//     because A6's `.scales`-detect seam lets candle ingest the mlx-affine tiers 1:1 (CUDA-validated
//     on Blackwell, sc-11990). So the tier resolver + the on-demand fetches below are compiled for
//     the SUPERSET of both lanes and serve candle exactly as they serve MLX.
//
// A tier IS a directory, not a requant toggle: both descriptors advertise `supported_quants: &[]`, so
// `advanced.mlxQuantize` selects WHICH DIR to load and the provider only ASSERTS the tier's baked-in
// level from its `split_model.json` (a mismatch is a hard load error, never a silent bf16 run).
// Installed layout — the tier dirs are SIBLINGS of the shared components, which the provider resolves
// from the tier dir's PARENT (`resolve_component_root`):
//
//     <model_dir>/{q4,q8,bf16}/{split_model.json, transformer/}
//     <model_dir>/{text_encoder,tokenizer,vae}/          <- the shared coRequisite (~10.4 GB)
//
// Text-to-video ONLY (`conditioning: []` on both descriptors), no LoRA/LoKr, no sampler/scheduler
// axis (one fixed flow-match Euler integrator), `max_count: 1`.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Mochi asset (the candle sibling is [`CANDLE_MOCHI_ADAPTER`]).
#[cfg(target_os = "macos")]
const MOCHI_ADAPTER: &str = "mlx_mochi";

/// The ONE turnkey repo serving BOTH backends (epic 1788 / A6 sc-11990). Deliberately NOT the LTX
/// shape (MLX rehost for macOS + upstream torch repo off-Mac): candle ingests these mlx-affine tiers
/// directly, `SceneWorks/mochi-1-candle` was never published, and pointing every platform here also
/// sidesteps sc-12113 (upstream `genmo/mochi-1-preview` ships both a bf16 and an fp32 `vae/` and the
/// loader's dir-glob picks fp32 non-deterministically — this rehost ships bf16 only).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_REPO: &str = "SceneWorks/mochi-1-mlx";

/// Pinned revision for the fixed [`MOCHI_REPO`]. The repo is a hard-coded const (no manifest/payload
/// override reaches the on-demand `q8/*` + `bf16/*` fetches below), so pulling the mutable `main`
/// branch would let an upstream re-push silently swap a checkpoint we load. Pin the exact commit for
/// defense-in-depth, mirroring [`LTX_BUNDLE_REVISION`] / the SeedVR2 + Real-ESRGAN pins. This is the
/// same revision B1's manifest sizes were read from (HF API rev `90a87786`); the native downloader
/// still verifies each file's own hash on download.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_REVISION: &str = "90a87786d9b5b592c3b3c083e5fcdf130007b1de";

/// Operator override for the Mochi model root (or a single tier dir) — the smoke's seam and the
/// escape hatch for a hand-staged conversion, mirroring `$SCENEWORKS_MLX_LTX_DIR`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_DIR_ENV: &str = "SCENEWORKS_MLX_MOCHI_DIR";

/// SceneWorks Mochi model id → the gen-core registry id, or `None` if not a Mochi family id.
///
/// ONE id covers BOTH backends (both providers register `MODEL_ID = "mochi_1"`) — contrast
/// [`ltx_engine_id`], which folds two sceneworks ids onto one engine, and the candle LTX arm, which
/// maps to a DIFFERENT `ltx_2_3_distilled` id. Despite that, this is the **MLX lane's** resolver: it
/// is the `resolve_video_route` / `mochi_available` / `ensure_video_engine_weights` key. The candle
/// lane resolves the same id through its own [`candle_video_engine_id`] table and then keys off the
/// resolved `engine_id` (mirroring its local `is_ltx`), so it never calls this — hence the macOS gate
/// rather than the superset one the shared tier helpers below carry.
#[cfg(target_os = "macos")]
fn mochi_engine_id(model: &str) -> Option<&'static str> {
    (model == "mochi_1").then_some("mochi_1")
}

/// Whether `dir` is a loadable Mochi TIER dir: the `split_model.json` the provider reads to resolve
/// the tier's baked-in quant, plus the AsymmDiT weights themselves. A tier dir mid-download (manifest
/// written, `transformer/` absent, or vice versa) fails this, so it never half-loads.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_tier_dir_is_complete(dir: &Path) -> bool {
    dir.join("split_model.json").is_file()
        && dir.join("transformer").join("model.safetensors").is_file()
}

/// Whether `root` carries the SHARED T5-XXL text encoder + tokenizer + AsymmVAE the provider resolves
/// from the tier dir's PARENT. These ship as a `coRequisite` (sc-9696) — a separate ~10.4 GB download
/// tracked once and excluded from the tier matrix — so a user can genuinely have a complete `q4/` and
/// no `text_encoder/`. Checking the tier dir alone would then resolve "successfully" and dead-end
/// inside the provider on a missing-file error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_shared_is_complete(root: &Path) -> bool {
    ["text_encoder", "tokenizer", "vae"]
        .iter()
        .all(|component| root.join(component).is_dir())
}

/// Parse `advanced.mlxQuantize` (int or numeric string) → the requested bit width, if present.
/// Mirrors [`ltx_quant_bits`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_quant_bits(request: &VideoRequest) -> Option<i64> {
    request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
}

/// Whether the request opts into the Q8 tier (`advanced.mlxQuantize >= 8`, int or string).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_wants_q8(request: &VideoRequest) -> bool {
    mochi_quant_bits(request).is_some_and(|bits| (8..=15).contains(&bits))
}

/// Whether the request opts into the dense **bf16** tier (`advanced.mlxQuantize <= 0`, or an explicit
/// `>= 16`). Never the default: absent ⇒ q4, so the ~20 GB dense tier is a deliberate opt-in
/// (mirrors [`ltx_wants_bf16`] / [`resolve_mlx_dense_quant`]'s `<= 0` rule, and additionally accepts
/// a literal `16` since bf16 IS 16-bit and a caller may say so).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_wants_bf16(request: &VideoRequest) -> bool {
    mochi_quant_bits(request).is_some_and(|bits| bits <= 0 || bits >= 16)
}

/// The Mochi tier search order — preferred tier first, then the always-SMALLER fallback tiers so a
/// partially-installed model still loads: `mlxQuantize <= 0`/`>= 16` ⇒ bf16, `8..=15` ⇒ q8, and — with
/// an explicit `1..=4` OR no explicit pick — the **q4** default (q4-first).
///
/// bf16 stays OUT of the default order so a default job never loads the dense tier by accident, and
/// the video lane keeps the pre-sc-10726 q4-first default (sc-10859) rather than epic 10721's app-wide
/// Q8 — see [`wan_tier_order`] / [`ltx_bundle_tier_order`] for that carve-out's rationale. For Mochi
/// the argument is sharper still: the untiled decode already dominates the footprint, so a silent Q8
/// buys quality the machine may not have the memory to spend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_tier_order(request: &VideoRequest) -> &'static [&'static str] {
    if mochi_wants_bf16(request) {
        &["bf16", "q8", "q4"]
    } else if mochi_wants_q8(request) {
        &["q8", "q4"]
    } else {
        &["q4", "q8"]
    }
}

/// Pick the first COMPLETE tier subdir of a Mochi model `root`, trying `order` (preferred first).
/// Requires the shared co-requisite at `root` too — a tier dir without the T5/VAE siblings is not
/// loadable, so it must not resolve.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_tier_subdir(root: &Path, order: &[&str]) -> Option<PathBuf> {
    if !mochi_shared_is_complete(root) {
        return None;
    }
    order
        .iter()
        .map(|tier| root.join(tier))
        .find(|dir| mochi_tier_dir_is_complete(dir))
}

/// The tier's baked-in quant level, read from the resolved tier dir's `split_model.json` — the same
/// manifest the provider reads. `Some(Q4)`/`Some(Q8)` for a packed tier, `None` for dense bf16.
///
/// Threaded onto the `LoadSpec` so (a) the asset record + metrics report the tier that was actually
/// loaded rather than the tier that was asked for, and (b) the provider's assert becomes a real
/// cross-check that our tier selection and its manifest read agree. Deriving this from the RESOLVED
/// dir rather than from `advanced.mlxQuantize` is what keeps a fallback (requested q8, only q4
/// installed) from turning into the provider's hard "spec.quantize disagrees with the tier" error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_tier_quant(tier_dir: &Path) -> Option<Quant> {
    let raw = std::fs::read_to_string(tier_dir.join("split_model.json")).ok()?;
    let manifest: Value = serde_json::from_str(&raw).ok()?;
    if !manifest.get("quantized").and_then(Value::as_bool)? {
        return None;
    }
    match manifest.get("quantization_bits").and_then(Value::as_i64)? {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// The dir the PRE-DOWNLOAD fit gate measures: the REQUESTED tier's path, whether or not it has been
/// downloaded yet. `None` when no model root is locatable at all ⇒ no signal ⇒ [`mochi_precheck`]
/// skips (never blocks without evidence, matching the gate's own fail-open rule).
///
/// Deliberately NOT [`resolve_mochi_model_dir`]: that one requires a COMPLETE tier and falls back to a
/// smaller installed one, and it hard-errors when nothing is installed — which is correct AFTER the
/// download and wrong before it (a first-ever q8 job would be rejected as "not downloaded" instead of
/// downloading). This returns a bare path and lets the gate's on-disk scan decide.
///
/// Naming the REQUESTED tier (not the fallback) is what keeps the weights term a strict lower bound on
/// what THIS job will hold resident: the scan finds the shared co-requisites plus this tier's own bytes
/// if present, never a different tier's.
#[cfg(target_os = "macos")]
fn mochi_precheck_dir(settings: &Settings, request: &VideoRequest) -> Option<PathBuf> {
    // The preferred tier — `mochi_tier_order`'s head is the one the request actually asked for; the
    // rest of the order is fallback, which the pre-check must not measure.
    let requested = mochi_tier_order(request).first().copied()?;
    let root = match std::env::var(MOCHI_DIR_ENV) {
        Ok(override_dir) => {
            let root = PathBuf::from(override_dir.trim());
            // A self-contained dir that IS one tier — the components live under it, so it is already
            // the thing to measure (and no download can change it).
            if mochi_tier_dir_is_complete(&root) && mochi_shared_is_complete(&root) {
                return Some(root);
            }
            root
        }
        Err(_) => huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO)?,
    };
    Some(root.join(requested))
}

/// PRE-DOWNLOAD Mochi admission check (sc-12322) — the same gate as [`mochi_preflight`], run BEFORE
/// `ensure_mochi_*_present` fetches a ~13-20 GiB tier, so a job that cannot fit is refused without
/// paying for the download first.
///
/// This is sound because the weights term is a conservative FLOOR here, not zero. Mochi's
/// `text_encoder/` + `tokenizer/` + `vae/` are a `coRequisite: true` download (~9.7 GiB) that is
/// already on disk before an on-demand `<tier>/*` fetch, and `mochi_resident_bytes` sums those shared
/// siblings from the tier dir's PARENT even when the tier dir itself is absent. So:
///
/// ```text
/// floor = shared co-requisites + (requested tier, if already downloaded)  ≤  actual resident weights
/// ⇒ needed(floor) ≤ needed(actual) ⇒ {pre-check refuses} ⊆ {full gate refuses}
/// ```
///
/// i.e. it can only refuse when refusal is CERTAIN — a config that fits is never rejected early, which
/// is the property that makes an early gate safe at all. Anything it admits still faces the unchanged
/// full check after the download.
///
/// The floor is what makes this bite on the machine the story is named for: at the shipped 151-frame
/// default the shared components alone give 9.73 + 60.56 decode + 2 OS = 72.29 GiB > 64. A
/// WEIGHTS-FREE pre-check (decode + OS only) would total 62.56 and ADMIT on a 64 GB Mac — downloading
/// the tier just to refuse it, which is the whole bug.
///
/// Lives beside [`mochi_preflight`] and for the same reason: one seam, asserted directly across
/// budgets and tier layouts. Its ORDER against the `ensure_mochi_*_present` fetches — the thing that
/// makes it worth having at all — is pinned separately, at the call site, by
/// `generate_mochi_using_refuses_before_paying_for_the_tier_download` (sc-12318).
#[cfg(target_os = "macos")]
fn mochi_precheck(
    model: &str,
    engine_id: &str,
    tier_dir: Option<&Path>,
    raw_frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<()> {
    let Some(tier_dir) = tier_dir else {
        return Ok(());
    };
    // The COERCED count — the frames the decode will really pay for, exactly as `mochi_preflight`
    // computes it. Pre-checking the raw request would gate on a length the engine never renders.
    let frames = video_frame_count(model, raw_frames);
    crate::mlx_fit_gate::mochi_fit_check(engine_id, tier_dir, frames, width, height)
}

/// Resolve the Mochi tier dir to load: the `$SCENEWORKS_MLX_MOCHI_DIR` override (accepted as either a
/// model root carrying tier subdirs or a single self-contained tier dir), then the turnkey
/// [`MOCHI_REPO`] snapshot's tier subdir. Only a COMPLETE tier + its shared co-requisite counts.
///
/// Shared by BOTH lanes (one repo, both backends), so the candle generation arm calls this instead of
/// `candle_video_snapshot_dir` — the flat-snapshot resolver has no notion of the tier layout.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_mochi_model_dir(settings: &Settings, request: &VideoRequest) -> WorkerResult<PathBuf> {
    let order = mochi_tier_order(request);
    if let Ok(override_dir) = std::env::var(MOCHI_DIR_ENV) {
        let root = PathBuf::from(override_dir.trim());
        // A model root with `q4/`/`q8/`/`bf16/` subdirs …
        if let Some(dir) = mochi_tier_subdir(&root, order) {
            return Ok(dir);
        }
        // … or a self-contained dir that IS one tier (components under it, the provider's
        // `resolve_component_root` self-resolution case).
        if mochi_tier_dir_is_complete(&root) && mochi_shared_is_complete(&root) {
            return Ok(root);
        }
    }
    if let Some(root) = huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO) {
        if let Some(dir) = mochi_tier_subdir(&root, order) {
            return Ok(dir);
        }
        // The snapshot exists but nothing loadable resolved — say WHICH half is missing rather than a
        // blanket "not found", since the tiers and the shared components are separate downloads.
        if !mochi_shared_is_complete(&root) {
            return Err(WorkerError::InvalidPayload(format!(
                "{}: the shared Mochi text encoder / tokenizer / VAE are not installed (expected \
                 text_encoder/, tokenizer/ and vae/ beside the tier dirs under {}). Re-download \
                 Mochi 1 in the Model Manager — the shared components install alongside whichever \
                 tier you pick.",
                request.model,
                root.display()
            )));
        }
        return Err(WorkerError::InvalidPayload(format!(
            "{}: no complete Mochi tier found under {} (looked for {order:?}). Download the tier you \
             selected in the Model Manager, or set ${MOCHI_DIR_ENV} to a model dir.",
            request.model,
            root.display()
        )));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{}: Mochi 1 is not downloaded (no {MOCHI_REPO} snapshot under {}). Install it from the \
         Model Manager, or set ${MOCHI_DIR_ENV} to a model dir.",
        request.model,
        settings.data_dir.display()
    )))
}

/// Whether the linked Mochi engine can serve this request now: a Mochi model id, in the ONE mode it
/// supports, with resolvable weights.
///
/// **text_to_video only** — both descriptors declare `conditioning: []`, so there is no i2v /
/// first-last / extend / bridge / replace surface. This gate matters because `VideoRequest`'s mode
/// DEFAULTS to `image_to_video` when the payload omits one, so a mode check is what keeps a
/// conditioning-shaped Mochi job out of the route rather than letting it reach the engine's
/// `validate_request` (or, worse, an unrelated arm of the ladder).
#[cfg(target_os = "macos")]
fn mochi_available(request: &VideoRequest, settings: &Settings) -> bool {
    mochi_engine_id(&request.model).is_some()
        && request.mode == "text_to_video"
        && resolve_mochi_model_dir(settings, request).is_ok()
}

/// Fetch the `q8/` tier on demand (mirrors [`ensure_ltx_q8_present`]). The default install is the lean
/// q4 tier + the shared co-requisite; a job that toggles `advanced.mlxQuantize: 8` without a
/// pre-installed q8 pulls just `q8/*` from the FIXED [`MOCHI_REVISION`] before any compute. No-op when
/// q8 isn't requested, when the snapshot isn't downloaded at all (resolve surfaces the clear
/// "not downloaded" error), or when `q8/` is already complete. Fails loud on a real download error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_mochi_q8_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if !mochi_wants_q8(request) {
        return Ok(());
    }
    ensure_mochi_tier_present(api, settings, job, "q8").await
}

/// Fetch the dense `bf16/` tier on demand (~20 GB) — the [`ensure_mochi_q8_present`] sibling for the
/// power-user tier ([`mochi_wants_bf16`]).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_mochi_bf16_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
) -> WorkerResult<()> {
    if !mochi_wants_bf16(request) {
        return Ok(());
    }
    ensure_mochi_tier_present(api, settings, job, "bf16").await
}

/// The shared body behind the two on-demand tier fetches: pull `<tier>/*` from the pinned
/// [`MOCHI_REVISION`] when the snapshot exists but that tier doesn't.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn ensure_mochi_tier_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    tier: &str,
) -> WorkerResult<()> {
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, MOCHI_REPO) else {
        return Ok(());
    };
    if mochi_tier_dir_is_complete(&root.join(tier)) {
        return Ok(());
    }
    crate::model_jobs::ensure_hf_files_cached(
        api,
        settings,
        job,
        MOCHI_REPO,
        MOCHI_REVISION,
        &[format!("{tier}/*")],
    )
    .await
    .map(|_| ())
}

/// The `rawSettings` recorded on a Mochi asset. `mochiTier` names the tier that was actually LOADED
/// (from the resolved dir's `split_model.json`), not the one the request asked for — those differ
/// when a requested tier isn't installed and the order falls back.
#[cfg(target_os = "macos")]
fn mochi_raw_settings(request: &VideoRequest, quant: Option<Quant>) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "mochiTier".to_owned(),
        Value::String(
            match quant {
                Some(Quant::Q4) => "q4",
                Some(Quant::Q8) => "q8",
                _ => "bf16",
            }
            .to_owned(),
        ),
    );
    Value::Object(raw)
}

/// Everything a Mochi generation must settle BEFORE the load, resolved as ONE unit by
/// [`mochi_preflight`] (sc-11992).
///
/// These travel together on purpose. Both fields are things [`generate_mochi`] cannot build its
/// [`VideoGenInput`] without, and [`mochi_preflight`] is the only thing that produces them — so the
/// generation arm cannot obtain a frame count or a quant marker on a path that skipped the fit gate.
/// Splitting them back into free-standing calls in the caller is what let the gate be silently dropped
/// (the mutation that survived the first round of this story's review).
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MochiPreflight {
    /// The requested frame count coerced onto Mochi's `6k+1` lattice.
    frames: u32,
    /// The tier dir's quant marker (`None` = the dense bf16 tier).
    quant: Option<Quant>,
}

/// The Mochi pre-flight: coerce the clip length onto the engine's lattice, refuse a job that cannot
/// fit this machine, and report the tier's quant — the whole "before we load anything" decision, as a
/// pure, synchronously testable seam (sc-11992).
///
/// Both decisions here are invisible in the output when wrong — a mis-coerced frame count is a hard
/// engine reject, a skipped gate is a `SIGKILL` — so they live in one seam where `mochi_preflight_*`
/// can assert them directly by model id, budget, and clip length.
///
/// This function was originally justified by the claim that [`generate_mochi`] is "unreachable from a
/// unit test" because it is `async` and takes a live `ApiClient`/`JobSnapshot`. **That was wrong**
/// (sc-12318): `ApiClient::new` does no I/O, `JobSnapshot` deserializes from a literal, and
/// `generate_mochi_using` now drives the whole arm against a stub engine. Keep the seam anyway — it
/// gives the gate cheap, direct coverage across budgets and clip lengths, and returning
/// `{frames, quant}` as a unit is what stops the arm obtaining either on a path that skipped the gate.
/// But do NOT reason from that claim again: a decision put here is *additionally* covered, not
/// *only* coverable here.
///
/// `Ok(MochiPreflight)` admits; `Err` is the actionable pre-crash rejection.
#[cfg(target_os = "macos")]
fn mochi_preflight(
    model: &str,
    engine_id: &str,
    tier_dir: &Path,
    raw_frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<MochiPreflight> {
    // The shared stride ladder, dispatched BY MODEL — so Mochi lands on `mochi_frame_count`'s 6k+1
    // lattice and never on Wan's 4k+1 stride. NOT `wan_frame_count` (see `video_frame_count`): the
    // shipped 5 s @ 30 fps default takes `wan_frame_count(150) = 149`, and `149 % 6 == 5` is OFF the
    // lattice, which the engine's `validate_request` hard-rejects.
    let frames = video_frame_count(model, raw_frames);

    // PRE-FLIGHT FIT GATE (epic AC: "an unsupported environment shows an actionable error, not a
    // crash"). This MUST run before the load: Mochi holds ~18.7 GiB of weights resident for the whole
    // run (`supports_sequential_offload: false`) and its AsymmVAE decode is UNTILED, so the peak grows
    // linearly with clip length (sc-12291) — ~60 GiB of decode alone at the shipped 5 s default. MLX's
    // default error handler is `exit(-1)`, a hard process kill that CANNOT be mapped to a job error
    // after the fact, so an over-budget job has to be refused here or it takes the worker down with
    // it. The generic `mlx_fit_gate` cannot cover this: it is resolution-blind by construction and its
    // weights-fit floor would admit on the 18.7 GiB weights alone.
    //
    // It takes the COERCED `frames` — the count the decode will actually pay for — not the raw
    // request, and not a constant.
    crate::mlx_fit_gate::mochi_fit_check(engine_id, tier_dir, frames, width, height)?;

    Ok(MochiPreflight {
        frames,
        quant: mochi_tier_quant(tier_dir),
    })
}

/// Resolve a Mochi request into a [`VideoGenInput`] and run it (epic 1788 / sc-11992) — the MLX lane.
///
/// Text-to-video only, true CFG (not distilled, so negative prompt + guidance are real), frames snap
/// to `6k+1`, and the tier dir carries the quant. Progress/heartbeat bridging, cooperative cancel, the
/// forward-progress stall watchdog and the completion metrics all come from the shared
/// [`generate_video`] funnel — the same one Wan/LTX/SVD use — so Mochi inherits the "no background
/// heartbeat during a job, the progress callback IS the keepalive" contract without re-implementing it.
///
/// Deliberately thin: every pre-load decision lives in [`mochi_precheck`] (before the download) or
/// [`mochi_preflight`] (after it), each a pure seam its own tests assert directly.
///
/// The ORDER of those two against the `ensure_mochi_*_present` fetches is the one decision this arm
/// itself owns — the pre-check must precede the download to be worth having (sc-12322) — and it is no
/// longer unpinned: `generate_mochi_using`'s tests (sc-12318) drive this whole body against a stub
/// engine, including `generate_mochi_using_refuses_before_paying_for_the_tier_download`, which fails if
/// the pre-check is moved after the fetches.
#[cfg(target_os = "macos")]
async fn generate_mochi(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    generate_mochi_using(
        api,
        settings,
        job,
        request,
        engine_id,
        backend,
        crate::inference_runtime::load,
    )
    .await
}

/// [`generate_mochi`] with the engine loader supplied by the caller (sc-12318) — the seam that makes
/// this arm's pre-load decisions assertable.
///
/// Everything `generate_mochi` does lives here; `generate_mochi` is the one-line delegation that binds
/// the real `inference_runtime::load`. That delegation carries no logic, so the only thing it can drift
/// on is the loader itself — whereas the decisions below are invisible in the output when wrong (a
/// mis-coerced count is a hard engine reject; a skipped gate is a `SIGKILL`; a pre-check that runs too
/// late is a ~13-20 GiB download the answer never depended on), and are covered by
/// `generate_mochi_using_*`.
#[cfg(target_os = "macos")]
async fn generate_mochi_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    engine_id: &'static str,
    backend: &str,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<(DecodedVideo, Value)> {
    // Refuse a job that cannot possibly fit BEFORE paying for its tier download (sc-12322). Runs on a
    // weights FLOOR (the already-present shared co-requisites), so it only ever refuses when the full
    // gate below would too — see `mochi_precheck`. MUST stay ahead of the fetches below: that ordering
    // is the whole point, and `generate_mochi_using_refuses_before_paying_for_the_tier_download` is
    // what holds it there.
    mochi_precheck(
        &request.model,
        engine_id,
        mochi_precheck_dir(settings, request).as_deref(),
        request.raw_frame_count(),
        request.width,
        request.height,
    )?;
    // A tier the job toggled but never downloaded is fetched before any compute (no-op otherwise).
    ensure_mochi_q8_present(api, settings, job, request).await?;
    ensure_mochi_bf16_present(api, settings, job, request).await?;
    let model_dir = resolve_mochi_model_dir(settings, request)?;
    // Frame lattice + fit gate + tier quant, as one gated unit — see `mochi_preflight`.
    let MochiPreflight { frames, quant } = mochi_preflight(
        &request.model,
        engine_id,
        &model_dir,
        request.raw_frame_count(),
        request.width,
        request.height,
    )?;

    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        // No adapter path on either backend (`supports_lora`/`supports_lokr` = false).
        adapters: Vec::new(),
        // t2v only — `conditioning: []` on the descriptor.
        conditioning: Vec::new(),
        prompt: request.prompt.clone(),
        // Not distilled ⇒ true CFG, so a negative prompt is real conditioning here (contrast the
        // distilled LTX path, which passes `None`).
        negative_prompt: non_empty_negative_prompt(request),
        width: request.width,
        height: request.height,
        frames,
        fps: request.fps,
        // `None` ⇒ the engine's own DEFAULT_STEPS (64) / DEFAULT_GUIDANCE (4.5).
        steps: advanced_opt_u32(request, "steps"),
        guidance: advanced_opt_f32(request, "guidanceScale"),
        seed: resolve_video_seed(request) as u64,
        ..VideoGenInput::default()
    };
    let raw_settings = mochi_raw_settings(request, quant);
    let decoded = generate_video_using(
        api,
        settings,
        job,
        backend,
        &request.advanced,
        input,
        load_generator,
    )
    .await?;
    Ok((decoded, raw_settings))
}

// ---------------------------------------------------------------------------
// Real MLX Stable Video Diffusion (SVD-XT) generation (macOS, via mlx-gen-svd, sc-3523):
// image→video ONLY — animates one source image into a fixed ~25-frame burst (no text prompt,
// no audio) via the `motion_bucket_id` / `noise_aug_strength` / conditioning-fps
// micro-conditioning. One engine model `svd_xt`. Source-of-truth = the torch
// `DiffusersVideoAdapter` `svd_video` path (`StableVideoDiffusionPipeline`, video_adapters.py).
// The engine loads the stock diffusers fp16 snapshot directly (vae/ + unet/ + image_encoder/),
// so there is no conversion step (unlike Wan/LTX).
//
// fps (sc-3764): the engine decouples the two cadences — the motion micro-conditioning fps
// (`added_time_ids` = fps − 1) rides `conditioning_fps` (manifest `condFps`, default 7 — the value
// the model was trained on, so MOTION stays correct), while the output/playback fps is the user's
// `request.fps` (mirroring the torch `export_to_video(fps=request.fps)`). So the burst now plays at
// the requested cadence with correct motion — full parity with the torch `svd_video` path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SVD asset — matches the torch `svd_video` adapter id so the
/// asset sidecar reads identically across the two backends.
#[cfg(target_os = "macos")]
const SVD_ADAPTER: &str = "svd_video";

/// The diffusers SVD-XT repo the engine loads directly (fp16 `vae/` + `unet/` + `image_encoder/`).
/// Shared by the MLX (macOS) lane and the candle (Windows/CUDA) lane (sc-5493).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const SVD_REPO: &str = "stabilityai/stable-video-diffusion-img2vid-xt";

/// SceneWorks model id → mlx-gen registry id for the SVD family (only `svd` → `svd_xt`), or `None`.
#[cfg(target_os = "macos")]
fn svd_engine_id(model: &str) -> Option<&'static str> {
    (model == "svd").then_some("svd_xt")
}

/// Whether the linked SVD engine can serve this request now (image→video with resolvable weights).
/// SVD is image-conditioned only, so a request without a `sourceAssetId` can never run on it.
#[cfg(target_os = "macos")]
fn svd_available(request: &VideoRequest, settings: &Settings) -> bool {
    svd_engine_id(&request.model).is_some()
        && request.source_asset_id.is_some()
        && resolve_svd_model_dir(settings).is_ok()
}

/// Whether `dir` is a usable SVD-XT snapshot — each component subdir carries the safetensors the
/// engine reads (preferring the on-disk `.fp16` variant, else the full-precision file).
#[cfg(target_os = "macos")]
fn svd_dir_is_complete(dir: &Path) -> bool {
    let has = |sub: &str, stem: &str| {
        dir.join(sub)
            .join(format!("{stem}.fp16.safetensors"))
            .is_file()
            || dir.join(sub).join(format!("{stem}.safetensors")).is_file()
    };
    has("vae", "diffusion_pytorch_model")
        && has("unet", "diffusion_pytorch_model")
        && has("image_encoder", "model")
}

/// Resolve the SVD-XT snapshot dir: env override (`SCENEWORKS_MLX_SVD_DIR`) → the cached HF snapshot
/// of [`SVD_REPO`]. Only a dir carrying the three component subdirs ([`svd_dir_is_complete`]) counts.
#[cfg(target_os = "macos")]
fn resolve_svd_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if svd_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, SVD_REPO) {
        if svd_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "svd: no complete SVD-XT weights found (expected vae/ + unet/ + image_encoder/ under the \
         cached {SVD_REPO} snapshot, or set $SCENEWORKS_MLX_SVD_DIR)"
    )))
}

/// Read an SVD integer knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`,
/// then clamp to `[min, max]`. Mirrors the torch `safe_int(advanced.get(adv_key),
/// target.get(manifest_key, default), min, max)` (advanced overrides the manifest, which overrides
/// the builtin default; the resolved value is clamped).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_i32(
    request: &VideoRequest,
    adv_key: &str,
    manifest_key: &str,
    default: i32,
    min: i32,
    max: i32,
) -> i32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(default)
        .clamp(min, max)
}

/// Read an SVD float knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`
/// (no clamp). Mirrors the torch `float(advanced.get(adv_key, target.get(manifest_key, default)))`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_f32(request: &VideoRequest, adv_key: &str, manifest_key: &str, default: f32) -> f32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as f32)
        .unwrap_or(default)
}

/// Inference steps for an SVD request: `advanced.steps` → `modelManifestEntry.steps[quality]` (else
/// its `balanced`) → the builtin quality ladder (fast 15 / balanced 25 / best 30), clamped 1..=80.
/// Mirrors the torch `_num_inference_steps` for the `svd_video` adapter.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_steps(request: &VideoRequest) -> u32 {
    let builtin = match request.quality.as_str() {
        "fast" => 15,
        "best" => 30,
        _ => 25,
    };
    let manifest_default = request
        .model_manifest_entry
        .get("steps")
        .and_then(Value::as_object)
        .and_then(|steps| {
            steps
                .get(&request.quality)
                .or_else(|| steps.get("balanced"))
        })
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(builtin);
    request
        .advanced
        .get("steps")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(manifest_default)
        .clamp(1, 80) as u32
}

/// The single `Reference` conditioning image (image→video source). SVD is image-conditioned only,
/// so a missing `sourceAssetId` is a hard error (the routing gate [`svd_available`] already
/// requires it; this guards the direct-call path).
#[cfg(target_os = "macos")]
fn resolve_svd_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let asset_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "svd image→video requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(vec![Conditioning::Reference {
        image,
        strength: None,
    }])
}

/// Raw-settings recorded on a real SVD asset (the resolved knobs + real-inference markers). Shared by
/// the MLX (macOS) and candle (Windows/CUDA, sc-5493) lanes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn svd_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert(
        "numFrames".to_owned(),
        json!(svd_i32(request, "numFrames", "numFrames", 25, 1, 25)),
    );
    raw.insert(
        "motionBucketId".to_owned(),
        json!(svd_i32(
            request,
            "motionBucketId",
            "motionBucketId",
            127,
            1,
            255
        )),
    );
    raw.insert(
        "conditioningFps".to_owned(),
        json!(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30)),
    );
    // The output/playback cadence (decoupled from conditioningFps; sc-3764).
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "noiseAugStrength".to_owned(),
        json!(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02
        )),
    );
    raw.insert(
        "decodeChunkSize".to_owned(),
        json!(svd_i32(
            request,
            "decodeChunkSize",
            "decodeChunkSize",
            8,
            1,
            64
        )),
    );
    raw.insert("steps".to_owned(), json!(svd_steps(request)));
    Value::Object(raw)
}

/// Resolve an SVD request into a [`VideoGenInput`] and run it (sc-3523). image→video only: no
/// prompt / negative / guidance (the engine uses its frame-wise CFG ramp); `frames` is the
/// model-fixed burst length (≤25); `fps` carries the motion-conditioning cadence (see the module
/// note); `motion_bucket_id` / `noise_aug_strength` / `decode_chunk_size` drive the SVD knobs.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_svd(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: resolve_svd_model_dir(settings)?,
        quant: None,
        adapters: Vec::new(),
        conditioning: resolve_svd_conditioning(settings, request, project_path)?,
        prompt: String::new(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32,
        // Output/playback cadence = the user's `fps` (mirrors the torch `export_to_video(fps=request.fps)`);
        // the motion cadence rides `conditioning_fps` below (sc-3764).
        fps: request.fps,
        steps: Some(svd_steps(request)),
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        motion_bucket_id: Some(
            svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
        ),
        noise_aug_strength: Some(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02,
        )),
        decode_chunk_size: Some(
            svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32,
        ),
        conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

// ---------------------------------------------------------------------------
// Real MLX Wan-VACE replace_person generation (macOS, via mlx-gen-wan, sc-3521):
// route the `replace_person` mode / `PersonReplace` job to the native `wan_vace`
// provider — the equivalent of the torch `DiffusersVideoAdapter` `WanVACEPipeline`
// path. The worker builds the masked-control inputs (source clip frames + the
// onnx-track person mask + character refs) and the engine does the
// masking/neutralization + denoise. Person detect/track/segment stays upstream.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan-VACE replace_person asset.
#[cfg(target_os = "macos")]
const WAN_VACE_ADAPTER: &str = "mlx_wan_vace";

/// Per-asset adapter label for the native dual-expert Wan2.2 VACE-Fun replace_person backend
/// (`wan2_2_vace_fun_14b`, sc-3459) — distinct from single-expert `mlx_wan_vace` so the asset
/// honestly records which VACE engine produced the replacement.
#[cfg(target_os = "macos")]
const WAN_VACE_FUN_ADAPTER: &str = "mlx_wan_vace_fun";

/// Letterbox pad colour for extracted source-clip frames — matches the Python `fit_frame`
/// background (`0x12110f` = RGB 18,17,15) so the box masks (rasterized from the same
/// normalized boxes at W×H) stay aligned with the control frames through the engine's
/// identity-resize preprocess.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const FRAME_PAD_COLOR: &str = "0x12110f";

/// Raw-settings recorded on a real Wan-VACE asset (`advanced` knobs + the real-inference
/// markers; the engine id is `wan_vace`, not the user-picked replace-capable model).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn wan_vace_raw_settings(request: &VideoRequest, model: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(model.to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "replacementMode".to_owned(),
        Value::String(request.replacement_mode.clone()),
    );
    Value::Object(raw)
}

/// SceneWorks `replacementMode` string → engine [`ReplacementMode`] (default FaceOnly).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn replacement_mode_from(value: &str) -> ReplacementMode {
    match value {
        "full_person_keep_outfit" => ReplacementMode::FullPersonKeepOutfit,
        "full_person_replace_outfit" => ReplacementMode::FullPersonReplaceOutfit,
        _ => ReplacementMode::FaceOnly,
    }
}

/// Whether `dir` is a load-ready assembled Wan-VACE snapshot — the diffusers VACE
/// `transformer/` plus the shared base-Wan UMT5/VAE/tokenizer that `crate::inference_runtime::load("wan_vace")`
/// reads (sc-3467 `assemble_wan_vace_snapshot` layout).
#[cfg(target_os = "macos")]
fn wan_vace_dir_is_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Resolve (assembling on first use) the converted Wan-VACE snapshot dir. Env override
/// (`SCENEWORKS_MLX_WAN_VACE_DIR`) → the app-managed `<data>/models/mlx/wan_vace` → assemble
/// it from the diffusers VACE transformer (HF `Wan-AI/Wan2.1-VACE-1.3B-diffusers`,
/// `transformer/`) + a converted base-Wan 14B snapshot's shared UMT5/z16-VAE/tokenizer
/// (sc-3467 `assemble_wan_vace_snapshot` — packaging, not conversion). Errors clearly when a
/// component is missing rather than degrading to the stub.
#[cfg(target_os = "macos")]
fn resolve_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if wan_vace_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let out_dir = settings
        .data_dir
        .join("models")
        .join("mlx")
        .join("wan_vace");
    if wan_vace_dir_is_complete(&out_dir) {
        return Ok(out_dir);
    }
    // Assemble on first use: the VACE transformer is diffusers-layout (no conversion); the
    // shared T5/VAE/tokenizer come from a converted base-Wan 14B snapshot (z16 VAE, shared
    // with VACE since both are Wan2.1-based).
    let vace_repo = "Wan-AI/Wan2.1-VACE-1.3B-diffusers";
    let transformer_dir = huggingface_snapshot_dir(&settings.data_dir, vace_repo)
        .map(|snapshot| snapshot.join("transformer"))
        .filter(|dir| dir.join("config.json").is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "replace_person: the Wan-VACE transformer ({vace_repo}) is not downloaded — \
                 fetch it via the model manager."
            ))
        })?;
    let base_wan = ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]
        .into_iter()
        .find_map(|model| resolve_wan_model_dir(settings, model, model).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "replace_person: Wan-VACE needs a converted base-Wan 14B snapshot (its shared \
                 UMT5 text encoder + z16 VAE + tokenizer). Convert/download wan_2_2_t2v_14b or \
                 wan_2_2_i2v_14b first."
                    .to_owned(),
            )
        })?;
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    runtime_macos::providers::wan::convert::assemble_wan_vace_snapshot(
        &out_dir,
        &transformer_dir,
        &base_wan,
        true,
    )
    .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "replace_person: failed to assemble the Wan-VACE snapshot: {error}"
        ))
    })?;
    Ok(out_dir)
}

/// Whether `dir` is a load-ready assembled Wan2.2 VACE-Fun snapshot — BOTH diffusers VACE-Fun
/// expert dirs (`transformer/` high-noise + `transformer_2/` low-noise) plus the shared base-Wan
/// UMT5/VAE/tokenizer that `crate::inference_runtime::load("wan2_2_vace_fun_14b")` reads (sc-6604
/// `assemble_wan_vace_fun_snapshot` layout).
#[cfg(target_os = "macos")]
fn wan_vace_fun_dir_is_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("transformer_2").join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Resolve (assembling on first use) the dual-expert Wan2.2 VACE-Fun snapshot dir (sc-3459). Env
/// override (`SCENEWORKS_MLX_WAN_VACE_FUN_DIR`) → the app-managed `<data>/models/mlx/wan_2_2_vace_fun`
/// → assemble it from the diffusers VACE-Fun experts (HF `linoyts/Wan2.2-VACE-Fun-14B-diffusers`,
/// `transformer/` + `transformer_2/`) + a converted base-Wan 14B snapshot's shared UMT5/z16-VAE/
/// tokenizer (sc-6604 `assemble_wan_vace_fun_snapshot` — packaging, not conversion). Errors clearly
/// when a component is missing rather than degrading to the Wan2.1 VACE backend or the stub.
#[cfg(target_os = "macos")]
fn resolve_wan_vace_fun_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_FUN_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if wan_vace_fun_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let out_dir = settings
        .data_dir
        .join("models")
        .join("mlx")
        .join("wan_2_2_vace_fun");
    if wan_vace_fun_dir_is_complete(&out_dir) {
        return Ok(out_dir);
    }
    // Assemble on first use: the two VACE-Fun experts are diffusers-layout (read directly, no
    // conversion); the shared T5/VAE/tokenizer come from a converted base-Wan 14B snapshot (z16 VAE,
    // shared with VACE-Fun since it is Wan2.2-A14B-based with the Wan2.1 z16 VAE).
    let vace_repo = "linoyts/Wan2.2-VACE-Fun-14B-diffusers";
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, vace_repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: the VACE-Fun transformers ({vace_repo}) are not downloaded — \
             fetch the model via the model manager."
        ))
    })?;
    let high = snapshot.join("transformer");
    let low = snapshot.join("transformer_2");
    if !high.join("config.json").is_file() || !low.join("config.json").is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: the {vace_repo} download is incomplete (missing transformer/ or \
             transformer_2/) — re-fetch it via the model manager."
        )));
    }
    let base_wan = ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]
        .into_iter()
        .find_map(|model| resolve_wan_model_dir(settings, model, model).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "wan_2_2_vace_fun_14b: VACE-Fun needs a converted base-Wan 14B snapshot (its shared \
                 UMT5 text encoder + z16 VAE + tokenizer). Convert/download wan_2_2_t2v_14b or \
                 wan_2_2_i2v_14b first."
                    .to_owned(),
            )
        })?;
    // CARVE-OUT(epic 3720): backend-specific weight packager; not a registry contract.
    runtime_macos::providers::wan::convert::assemble_wan_vace_fun_snapshot(
        &out_dir, &high, &low, &base_wan, true,
    )
    .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: failed to assemble the VACE-Fun snapshot: {error}"
        ))
    })?;
    Ok(out_dir)
}

/// Decode the source clip into exactly `count` RGB frames at `width × height` (letterboxed,
/// `FRAME_PAD_COLOR`), evenly resampled across the clip — the new shared frame-extraction
/// helper (Python `load_source_video_frames`; also the seam extend/bridge will reuse). The
/// frames are the (un-neutralized) Wan-VACE control video; the engine masks them.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn load_source_video_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    count: usize,
) -> WorkerResult<Vec<Image>> {
    let asset_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a source clip (sourceClipAssetId).".to_owned(),
        )
    })?;
    let asset = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_asset(&request.project_id, asset_id)
        .map_err(|error| WorkerError::InvalidPayload(format!("source clip {asset_id}: {error}")))?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("source clip {asset_id} has no media path"))
        })?;
    let media_path = crate::safe_project_path(project_path, rel)?;
    if !tokio::fs::try_exists(&media_path).await? {
        return Err(WorkerError::InvalidPayload(format!(
            "source clip file is missing: {}",
            media_path.display()
        )));
    }

    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()`. Mirrors `sw-person-track-{safe_download_dir(job.id)}` in media_jobs.
    let work_dir =
        std::env::temp_dir().join(format!("sw-replace-frames-{}", safe_download_dir(&job.id)));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
        width = request.width,
        height = request.height,
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_extracted_frames(work_dir.clone(), count).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Collect the extracted PNG frames in `work_dir`, resample them to `count` evenly-spaced
/// indices (Python `evenly_spaced_indices` — the same arithmetic as the mask resample), and
/// decode the selected frames to engine [`Image`]s. Blocking IO/decoding runs off the runtime.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_extracted_frames(work_dir: PathBuf, count: usize) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let indices = crate::person_replace::resample_indices(paths.len(), count);
        indices
            .into_iter()
            .map(|index| decode_png_image(&paths[index]))
            .collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// The approved character reference images (≤4) for the replacement (Python
/// `character_reference_images`): the selected look's `approvedReferenceIds`, else the
/// character's approved `references`. Errors when none are readable (the torch
/// `_validate_inputs` parity). The engine cover-fits each to the output size.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_character_references(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let character_id = request.character_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload("replace_person requires a character (characterId).".to_owned())
    })?;
    let character = CharacterStore::new(&settings.data_dir, project_path.to_path_buf())
        .get_character(&request.project_id, character_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("character {character_id}: {error}"))
        })?;
    let mut ids: Vec<String> = Vec::new();
    if let Some(look_id) = request.character_look_id.as_deref() {
        if let Some(looks) = character.get("looks").and_then(Value::as_array) {
            for look in looks {
                if look.get("id").and_then(Value::as_str) == Some(look_id) {
                    if let Some(approved) =
                        look.get("approvedReferenceIds").and_then(Value::as_array)
                    {
                        ids.extend(approved.iter().filter_map(Value::as_str).map(str::to_owned));
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        if let Some(references) = character.get("references").and_then(Value::as_array) {
            for reference in references {
                if reference
                    .get("approved")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Some(asset_id) = reference.get("assetId").and_then(Value::as_str) {
                        ids.push(asset_id.to_owned());
                    }
                }
            }
        }
    }
    // The approved references we will attempt (capped at the engine's 4-reference contract). A
    // reference that fails to load must NOT be dropped silently: a corrupted approved reference
    // otherwise quietly weakens identity conditioning with zero signal (sc-8922, F-120). Warn per
    // skipped reference (asset id + error) and, when some — but not all — loaded, emit a summary the
    // operator can compare against the approved count.
    let attempted: Vec<String> = ids
        .into_iter()
        .filter(|id| !id.is_empty())
        .take(4)
        .collect();
    let approved_count = attempted.len();
    let mut images = Vec::new();
    for asset_id in attempted {
        match load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &asset_id,
            project_path,
        ) {
            Ok(image) => images.push(image),
            Err(error) => {
                tracing::warn!(
                    event = "character_reference_load_failed",
                    characterId = %character_id,
                    assetId = %asset_id,
                    error = %error,
                    "skipping an unreadable approved character reference — identity conditioning \
                     will use fewer references than approved"
                );
            }
        }
    }
    if images.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        ));
    }
    if images.len() < approved_count {
        tracing::warn!(
            event = "character_references_partially_loaded",
            characterId = %character_id,
            loaded = images.len(),
            approved = approved_count,
            "loaded fewer character references than were approved — {} of {} approved references \
             could not be read; identity conditioning is reduced",
            approved_count - images.len(),
            approved_count
        );
    }
    Ok(images)
}

/// Convert an `image::RgbImage` (the rasterized mask) to an engine [`Image`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn rgb_image_to_engine(image: image::RgbImage) -> Image {
    Image {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    }
}

/// Build the Wan-VACE conditioning: one [`Conditioning::ControlClip`] (source frames + the
/// per-frame person mask; the engine neutralizes the masked region) plus one
/// [`Conditioning::Reference`] per character reference image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_vace_conditioning(
    frames: Vec<Image>,
    masks: Vec<image::RgbImage>,
    references: Vec<Image>,
    masking_strength: f32,
    mode: ReplacementMode,
) -> WorkerResult<Vec<Conditioning>> {
    if frames.len() != masks.len() {
        return Err(WorkerError::InvalidPayload(format!(
            "replace_person: control frames ({}) and masks ({}) length mismatch",
            frames.len(),
            masks.len()
        )));
    }
    let mask_images: Vec<Image> = masks.into_iter().map(rgb_image_to_engine).collect();
    let mut conditioning = Vec::with_capacity(1 + references.len());
    conditioning.push(Conditioning::ControlClip {
        frames,
        mask: mask_images,
        masking_strength,
        start_frame: 0,
        mode,
    });
    for image in references {
        conditioning.push(Conditioning::Reference {
            image,
            strength: None,
        });
    }
    Ok(conditioning)
}

/// The honest `replacementStatus` recorded on the asset fact (mirrors the torch
/// `replacement_status`); the API folds it into the video sidecar's normalizedSettings.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn replacement_status_value(
    track: &Value,
    track_id: &str,
    mask_mode: &str,
    masking_strength: f32,
    reference_count: usize,
    frame_count: usize,
    adapter: &str,
) -> Value {
    let status = track.get("status").and_then(Value::as_object);
    let person_tracking_active = status
        .and_then(|s| s.get("personTrackingActive"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mask_state = status
        .and_then(|s| s.get("maskState"))
        .and_then(Value::as_str)
        .unwrap_or("missing")
        .to_owned();
    let corrections = track.get("corrections").and_then(Value::as_array);
    let correction_count = corrections.map(|list| list.len()).unwrap_or(0);
    let resolved_track_id = track.get("id").and_then(Value::as_str).unwrap_or(track_id);
    json!({
        "personDetectionActive": true,
        "personTrackingActive": person_tracking_active,
        "replacementActive": true,
        "replacementAdapter": adapter,
        "maskMode": mask_mode,
        "maskState": mask_state,
        "maskingStrength": masking_strength,
        "personTrackId": resolved_track_id,
        "characterReferenceCount": reference_count,
        "controlFrameCount": frame_count,
        "usedCorrections": correction_count > 0,
        "correctionCount": correction_count,
    })
}

/// Resolve a replace_person request into a Wan-VACE generation: assemble/resolve the snapshot,
/// extract the source-clip control frames, build the per-frame person mask from the saved
/// track (corrections applied), load the character refs, run the engine, and return the decoded
/// video plus the honest `replacementStatus`. Person detect/track/segment stays upstream.
#[cfg(target_os = "macos")]
async fn generate_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_wan_vace_model_dir(settings)?;
    generate_wan_vace_engine(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        "wan_vace",
        model_dir,
        WAN_VACE_ADAPTER,
        resolve_wan_quant(request),
    )
    .await
}

/// The dual-expert Wan2.2 VACE-Fun replace_person dispatch (sc-3459) — identical conditioning to
/// single-expert [`generate_wan_vace`], but resolves the dual-expert snapshot
/// ([`resolve_wan_vace_fun_model_dir`]) + the `wan2_2_vace_fun_14b` engine. Forces **Q4** by default
/// (the validated real-weight footprint; both 14B experts at bf16 would risk OOM on a 128 GB Mac),
/// still overridable via the `mlxQuantize` advanced knob.
#[cfg(target_os = "macos")]
async fn generate_wan_vace_fun(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_wan_vace_fun_model_dir(settings)?;
    let quant = resolve_wan_quant(request).or(Some(Quant::Q4));
    generate_wan_vace_engine(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        "wan2_2_vace_fun_14b",
        model_dir,
        WAN_VACE_FUN_ADAPTER,
        quant,
    )
    .await
}

/// Shared replace_person engine dispatch for both VACE backends (single-expert `wan_vace` +
/// dual-expert `wan2_2_vace_fun_14b`): builds the source-frame + person-mask + character-reference
/// control conditioning, runs the resolved engine, and returns the decoded video + the honest
/// `replacementStatus`. Only `engine_id` / `model_dir` / `adapter` / `quant` differ between the two.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_wan_vace_engine(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    engine_id: &'static str,
    model_dir: PathBuf,
    adapter: &'static str,
    quant: Option<Quant>,
) -> WorkerResult<(DecodedVideo, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Source frames + masks must match in count and be `1 + 4·k` (one z16 VAE temporal chunk),
    // which `wan_frame_count` guarantees — the engine `validate()` enforces it too.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();

    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;

    let negative_prompt = non_empty_negative_prompt(request);
    let steps = request.advanced.get("steps").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    });
    let guidance = request.advanced.get("guidanceScale").and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    });
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        adapter,
    );
    Ok((decoded, status))
}

// ---------------------------------------------------------------------------
// Wan extend_clip / video_bridge — native Wan-VACE ControlClip (sc-3812, tier C).
//
// The TI2V-5B single-frame path (`build_wan_boundary_conditioning`, sc-3357) conditions on one
// boundary still, so it morphs *from* a frozen frame and cannot inherit the source clip's motion.
// Routing these modes to the `wan_vace` engine instead lets the model attend to *several real*
// source frames pinned at the kept positions (mask black = keep) while it generates the rest of the
// timeline freely (mask white = regenerate over a neutral-gray control video). That is the whole
// point of extend/bridge — genuine motion continuity — at the cost of the smaller VACE-1.3B base
// (vs TI2V-5B), so the single-frame path stays the baseline/fallback. No reference images: the
// content comes from the kept frames, not a character (the engine's reference path is optional).
// Raw-settings record `model = wan_vace` + `fidelityTier = vace_controlclip` so the engine
// substitution under the user's `wan_2_2` pick is an inspectable fact on the asset, not a black box.

/// Mid-gray (≈0 after the engine's `2·x/255 − 1` normalization) control frame for the
/// to-generate span: a neutral `reactive = video·mask` signal so the masked region is generated
/// freely from the kept frames + prompt, never biased toward a frozen filler image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn neutral_control_frame(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: vec![128u8; (width as usize) * (height as usize) * 3],
    }
}

/// A solid W×H mask (`0` = keep the control frame, `255` = regenerate; the engine binarizes at
/// 0.5), matching the `image::RgbImage` form `person_track_masks` produces for replace_person.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn solid_mask(width: u32, height: u32, value: u8) -> image::RgbImage {
    image::RgbImage::from_pixel(width, height, image::Rgb([value, value, value]))
}

/// How many real source frames to pin as the motion anchor per kept boundary (sc-3812). More =
/// truer continuity but fewer freely-generated frames. Overridable via advanced `motionAnchorFrames`
/// (per side); defaults to ~⅓ of the output budget (split across the two boundaries for bridge), and
/// is clamped so at least 5 frames (one z16 chunk) are left to generate.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn extend_anchor_frames(request: &VideoRequest, frame_count: usize) -> usize {
    let per_side = if request.mode == "video_bridge" { 2 } else { 1 };
    let max_total = frame_count.saturating_sub(5).max(1);
    let max_per_side = (max_total / per_side).max(1);
    let default = (frame_count / 3 / per_side).max(1);
    let requested = request
        .advanced
        .get("motionAnchorFrames")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as usize)
        .unwrap_or(default);
    requested.clamp(1, max_per_side)
}

/// Decode the `take`-end `count` frames of a source clip (its head or tail) to letterboxed W×H
/// engine [`Image`]s, in temporal order (sc-3812). Unlike [`load_source_video_frames`] — which
/// resamples the *whole* clip evenly — this keeps *consecutive* real frames so the model sees the
/// clip's actual motion velocity at the boundary. Decodes only the kept subset (`decode_png_image`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
async fn load_clip_anchor_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    let media_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()` even with the uuid suffix. Mirrors the person-track work dir.
    let work_dir = std::env::temp_dir().join(format!(
        "sw-anchor-frames-{}-{}",
        safe_download_dir(&job.id),
        Uuid::new_v4().simple()
    ));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_anchor_frames(work_dir.clone(), count, take).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Pick the head/tail `count` consecutive PNGs from `work_dir` (sorted) and decode them to engine
/// [`Image`]s, preserving temporal order. Fewer available than `count` ⇒ all of them (short clip).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_anchor_frames(
    work_dir: PathBuf,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let take_n = count.min(paths.len());
        let selected = match take {
            ClipFramePosition::Last => &paths[paths.len() - take_n..],
            ClipFramePosition::First => &paths[..take_n],
        };
        selected.iter().map(|path| decode_png_image(path)).collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// Decode one RGB PNG into an engine [`Image`] (shared by the resample + anchor frame selectors).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn decode_png_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source frame {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// Build the Wan-VACE extend/bridge ControlClip (sc-3812): real source frames pinned at the kept
/// positions (mask black) and a neutral-gray generated span (mask white). For `extend_clip` the
/// left-clip tail anchors the start and the continuation is generated; for `video_bridge` both
/// clips' boundary anchors are pinned at the two ends and the gap between them is generated. The
/// control clip is `frame_count` long (`1 + 4·k`, the engine's z16-chunk constraint) with no
/// reference images. `masking_strength`/`mode` are inert in the WanVACE mask math (carried for the
/// shared [`Conditioning::ControlClip`] contract), so they take the neutral defaults.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_extend_bridge_vace_conditioning(
    request: &VideoRequest,
    width: u32,
    height: u32,
    frame_count: usize,
    left_anchor: Vec<Image>,
    right_anchor: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let neutral = neutral_control_frame(width, height);
    let keep_mask = solid_mask(width, height, 0);
    let gen_mask = solid_mask(width, height, 255);
    let mut frames: Vec<Image> = Vec::with_capacity(frame_count);
    let mut masks: Vec<image::RgbImage> = Vec::with_capacity(frame_count);
    let left_n = left_anchor.len();
    match request.mode.as_str() {
        "extend_clip" => {
            if left_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "extend_clip: {left_n} anchor frames leave no room to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in left_n..frame_count {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
        }
        "video_bridge" => {
            let right = right_anchor.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
            let right_n = right.len();
            if left_n + right_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "video_bridge: {left_n}+{right_n} anchor frames leave no gap to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in 0..(frame_count - left_n - right_n) {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
            for frame in right {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
        }
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "build_extend_bridge_vace_conditioning: unexpected mode {other}"
            )))
        }
    }
    build_vace_conditioning(frames, masks, Vec::new(), 1.0, ReplacementMode::default())
}

/// Raw-settings for a Wan-VACE extend/bridge asset: the request `advanced` knobs + the real-inference
/// markers, recording the actual engine (`wan_vace`) and `fidelityTier` so the substitution under the
/// user's `wan_2_2` pick is an inspectable fact (sc-3812). Unlike [`wan_vace_raw_settings`] there is
/// no `replacementMode` (these modes are not person-replacement).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn wan_vace_extend_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String("wan_vace".to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "fidelityTier".to_owned(),
        Value::String("vace_controlclip".to_owned()),
    );
    Value::Object(raw)
}

/// Resolve an extend_clip / video_bridge request into a native Wan-VACE generation (sc-3812, tier C).
/// Loads the real source-clip anchor frames (the left clip's tail for extend; both clips' boundaries
/// for bridge), builds the source-at-kept-positions + generated-span ControlClip, and runs the
/// `wan_vace` engine. The TI2V-5B single-frame path ([`generate_wan`]) remains the baseline/fallback,
/// chosen by the dispatch seam when the VACE snapshot is unprovisioned.
#[cfg(target_os = "macos")]
async fn generate_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    model_dir: PathBuf,
) -> WorkerResult<DecodedVideo> {
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = non_empty_negative_prompt(request);
    let steps = request.advanced.get("steps").and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as u32)
    });
    let guidance = request.advanced.get("guidanceScale").and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())
            .map(|value| value as f32)
    });
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        quant: resolve_wan_quant(request),
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> VideoRequest {
        VideoRequest::from_payload(&value.as_object().cloned().unwrap())
    }

    /// sc-12297: the worker's backstop refuses a clip past the model's declared
    /// `limits.hardMaxDuration` BEFORE any weights load — pinned at the seam
    /// `run_video_generate_job` actually calls, not just as a sceneworks-core free function.
    ///
    /// Kills the mutation that matters: deleting the `duration_limit_error` call from
    /// `video_preflight`. Every core test stays green through that — the core function is still
    /// correct, it is simply no longer CALLED — which is the precise shape of the dropped-gate bug
    /// the sc-11992 review caught surviving 835 green tests.
    ///
    /// Ungated on purpose. `mochi_preflight`'s fit gate is macOS-only, so the candle lane's ONLY
    /// duration ceiling is this one; a `#[cfg(target_os = "macos")]` here would leave the platform
    /// that has no other protection untested.
    #[test]
    fn video_preflight_refuses_a_clip_past_the_models_declared_hard_max_duration() {
        // The story's mochi_1 shape: cap 5s, asked for 30s. 30 x 30fps = 900 raw frames, which snap
        // to 901 on Mochi's 6k+1 lattice — and `901 % 6 == 1`, so the engine's own validate_request
        // ACCEPTS it. On the candle lane nothing else says no, which is what makes this gate the
        // whole defense rather than a nicety.
        let over = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
            "duration": 30, "fps": 30,
            "modelManifestEntry": { "limits": { "hardMaxDuration": 5 } }
        }));
        assert_eq!(over.raw_frame_count(), 900);
        assert_eq!(
            over.frame_count(),
            901,
            "on-lattice, and the engine would accept it"
        );
        let Err(WorkerError::InvalidPayload(message)) = video_preflight(&over) else {
            panic!("a 30s clip on a 5s-capped model must be refused before the load");
        };
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(message.contains("5s"), "states the cap: {message}");
        assert!(message.contains("30s"), "states what was asked: {message}");

        // At-cap admits — the shipped 5s default must still run, or the gate has bricked the model.
        let at_cap = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
            "duration": 5, "fps": 30,
            "modelManifestEntry": { "limits": { "hardMaxDuration": 5 } }
        }));
        assert!(
            video_preflight(&at_cap).is_ok(),
            "5s is exactly the cap and is the model's own shipped default"
        );

        // A job carrying no manifest entry (the stub lane, an uncatalogued id whose entry resolves
        // to {}) is UNCONSTRAINED — the gate never blocks without a declared cap.
        let no_entry = request(json!({
            "projectId": "p", "model": "stub-model", "mode": "text_to_video", "prompt": "p",
            "duration": 30
        }));
        assert!(
            video_preflight(&no_entry).is_ok(),
            "no cap declared => no cap"
        );

        // The pre-existing projectId invariant still holds at this seam.
        let no_project = request(json!({ "model": "mochi_1", "mode": "text_to_video" }));
        assert!(matches!(
            video_preflight(&no_project),
            Err(WorkerError::InvalidPayload(m)) if m.contains("projectId")
        ));
    }

    /// sc-12347: the same backstop on the fps axis — refuses a rate the model does not advertise,
    /// and admits a payload that names no rate at all.
    ///
    /// Kills the same class of mutation: deleting the `fps_limit_error` call from `video_preflight`
    /// leaves every core test green, because the core function is still correct — just not CALLED.
    ///
    /// Ungated for the same reason as the duration test: `mochi_preflight`'s fit gate is macOS-only
    /// AND is a *memory* test, so it cannot refuse an off-spec-but-affordable rate on any lane.
    #[test]
    fn video_preflight_refuses_an_fps_the_model_does_not_advertise() {
        let mochi_entry = json!({
            "limits": { "hardMaxDuration": 5, "fps": [30] }, "defaults": { "fps": 30 }
        });

        // The story's shape: a request that is LEGALLY 5 seconds — sc-12297's cap admits it — but
        // asks for 60fps. 5 x 60 = 300 raw frames snapping to 301, double the shipped default's
        // 151, and `301 % 6 == 1` so the engine's own validate_request ACCEPTS it. The duration gate
        // cannot see this; only the fps menu refuses it.
        let off_menu = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
            "duration": 5, "fps": 60, "modelManifestEntry": mochi_entry
        }));
        assert_eq!(off_menu.raw_frame_count(), 300);
        assert_eq!(
            off_menu.frame_count(),
            301,
            "on-lattice, and the engine would accept it"
        );
        assert_eq!(
            duration_limit_error(
                &off_menu.model,
                off_menu.duration,
                &off_menu.model_manifest_entry
            ),
            None,
            "the duration cap ADMITS this request — it is legally 5 seconds"
        );
        let Err(WorkerError::InvalidPayload(message)) = video_preflight(&off_menu) else {
            panic!("60fps on a model that advertises only 30 must be refused before the load");
        };
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("30 fps"),
            "states what is allowed: {message}"
        );
        assert!(
            message.contains("60 fps"),
            "states what was asked: {message}"
        );

        // The model's own advertised rate admits — the shipped default must still run.
        let on_menu = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
            "duration": 5, "fps": 30, "modelManifestEntry": mochi_entry
        }));
        assert!(
            video_preflight(&on_menu).is_ok(),
            "30 is what mochi advertises"
        );
        assert_eq!(on_menu.frame_count(), 151);

        // THE REGRESSION THIS STORY NEARLY SHIPPED: a payload naming NO fps must be ADMITTED. It
        // resolves to the model's declared 30, not the blanket 25 — which is off mochi's menu and
        // would make this gate refuse a perfectly ordinary job (7 of the 10 shipped video models
        // are in that position, so this is the common case, not an edge).
        let no_fps = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
            "duration": 5, "modelManifestEntry": mochi_entry
        }));
        assert_eq!(
            no_fps.fps, 30,
            "resolved from the model's declared defaults.fps"
        );
        assert!(
            video_preflight(&no_fps).is_ok(),
            "an fps-less payload must not be refused by the menu"
        );
        assert_eq!(
            no_fps.frame_count(),
            151,
            "the frame count the manifest documents"
        );

        // A job carrying no manifest entry is UNCONSTRAINED — never block without a declared menu.
        let no_entry = request(json!({
            "projectId": "p", "model": "stub-model", "mode": "text_to_video", "prompt": "p",
            "duration": 5, "fps": 60
        }));
        assert!(
            video_preflight(&no_entry).is_ok(),
            "no menu declared => no menu"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Mochi 1 (epic 1788 / sc-11992). These sit on the SUPERSET cfg because the tier resolver, the
    // stride and the on-demand fetches are SHARED by both lanes (one repo, both backends) — so they
    // run on the macOS lane AND on the windows-candle lane, which is what makes the candle half
    // genuinely covered rather than asserted.
    // -----------------------------------------------------------------------------------------

    /// Build a Mochi model root in the A6 installed layout: tier dirs as SIBLINGS of the shared
    /// T5/tokenizer/VAE co-requisite. `tiers` lists the tier dirs to materialize.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn mochi_root(tag: &str, tiers: &[&str], shared: bool) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "mochi_root_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        for tier in tiers {
            let transformer = root.join(tier).join("transformer");
            std::fs::create_dir_all(&transformer).unwrap();
            std::fs::write(transformer.join("model.safetensors"), b"w").unwrap();
            let (quantized, bits) = match *tier {
                "q4" => (true, 4),
                "q8" => (true, 8),
                _ => (false, 0),
            };
            let manifest = if quantized {
                json!({ "quantized": true, "quantization_bits": bits, "quantization_group_size": 64 })
            } else {
                json!({ "quantized": false })
            };
            std::fs::write(
                root.join(tier).join("split_model.json"),
                serde_json::to_string(&manifest).unwrap(),
            )
            .unwrap();
        }
        if shared {
            for component in ["text_encoder", "tokenizer", "vae"] {
                std::fs::create_dir_all(root.join(component)).unwrap();
            }
        }
        root
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn mochi_request(advanced: Value) -> VideoRequest {
        request(json!({
            "projectId": "p",
            "model": "mochi_1",
            "mode": "text_to_video",
            "prompt": "a calico kitten",
            "advanced": advanced,
        }))
    }

    /// `advanced.mlxQuantize` selects WHICH TIER DIR to load — the story's "selecting q4/q8/bf16
    /// loads the right tier" AC. A tier is a DIRECTORY, not a requant toggle (`supported_quants: &[]`
    /// on both descriptors), so this mapping is the entire quant mechanism.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_tier_order_maps_mlx_quantize_to_a_tier_dir() {
        // No explicit pick ⇒ q4-first (the video lane's sc-10859 carve-out; bf16 never a default).
        assert_eq!(mochi_tier_order(&mochi_request(json!({}))), &["q4", "q8"]);
        assert!(!mochi_tier_order(&mochi_request(json!({}))).contains(&"bf16"));
        // Explicit small ⇒ still q4-first.
        assert_eq!(
            mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 4 }))),
            &["q4", "q8"]
        );
        // q8 ⇒ q8-first. Accepted as int OR string (the manifest/UI can send either).
        assert_eq!(
            mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 8 }))),
            &["q8", "q4"]
        );
        assert_eq!(
            mochi_tier_order(&mochi_request(json!({ "mlxQuantize": "8" }))),
            &["q8", "q4"]
        );
        // Dense bf16 ⇒ the `<= 0` opt-in rule, and a literal 16 (bf16 IS 16-bit).
        for dense in [json!({ "mlxQuantize": 0 }), json!({ "mlxQuantize": -1 })] {
            assert_eq!(
                mochi_tier_order(&mochi_request(dense)),
                &["bf16", "q8", "q4"]
            );
        }
        assert_eq!(
            mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 16 }))),
            &["bf16", "q8", "q4"]
        );
    }

    /// The resolver returns the TIER DIR (not the model root), honors the requested tier, and folds
    /// in the shared co-requisite gate. Pins the `WeightsSource::Dir` = tier-dir contract the
    /// provider's parent-resolution (`resolve_component_root`) depends on.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn resolve_mochi_model_dir_picks_the_requested_tier_dir() {
        let root = mochi_root("resolve", &["q4", "q8", "bf16"], true);
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            ..Settings::from_env()
        };
        // Route the resolver at the fixture via the operator override.
        temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
            for (advanced, want) in [
                (json!({}), "q4"),
                (json!({ "mlxQuantize": 4 }), "q4"),
                (json!({ "mlxQuantize": 8 }), "q8"),
                (json!({ "mlxQuantize": 0 }), "bf16"),
            ] {
                let dir = resolve_mochi_model_dir(&settings, &mochi_request(advanced.clone()))
                    .unwrap_or_else(|e| panic!("resolve {advanced} failed: {e:?}"));
                assert_eq!(
                    dir,
                    root.join(want),
                    "advanced {advanced} must resolve the {want} TIER dir"
                );
                // The tier dir's parent is the model root — where the provider finds the shared
                // T5/VAE. If the resolver ever returned the root itself this breaks.
                assert_eq!(dir.parent().unwrap(), root);
            }
        });
        std::fs::remove_dir_all(&root).ok();
    }

    /// A complete tier with NO shared co-requisite must NOT resolve. The T5/tokenizer/VAE are a
    /// SEPARATE download (`coRequisite`, sc-9696), so "q4 present, text_encoder absent" is a real
    /// user state — and resolving it would dead-end inside the provider on a missing-file error
    /// instead of telling the user what to re-download.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn resolve_mochi_model_dir_requires_the_shared_corequisite() {
        let root = mochi_root("noshared", &["q4"], false);
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            ..Settings::from_env()
        };
        temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
            assert!(
                mochi_tier_subdir(&root, &["q4"]).is_none(),
                "a tier without the shared T5/VAE siblings is not loadable"
            );
            assert!(
                resolve_mochi_model_dir(&settings, &mochi_request(json!({}))).is_err(),
                "must error rather than resolve an unloadable tier"
            );
        });
        std::fs::remove_dir_all(&root).ok();
    }

    /// A half-downloaded tier (manifest without weights, or weights without the manifest) must fall
    /// THROUGH to the next tier in the order rather than half-loading.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_tier_subdir_skips_an_incomplete_tier() {
        let root = mochi_root("partial", &["q4"], true);
        // A `q8/` that carries only the manifest — the shape a mid-flight download leaves on disk.
        std::fs::create_dir_all(root.join("q8")).unwrap();
        std::fs::write(root.join("q8").join("split_model.json"), b"{}").unwrap();

        assert!(!mochi_tier_dir_is_complete(&root.join("q8")));
        assert_eq!(
            mochi_tier_subdir(&root, &["q8", "q4"]),
            Some(root.join("q4")),
            "a torn q8 must fall through to the complete q4, never half-load"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// The quant marker rides the RESOLVED tier's own `split_model.json` — the same manifest the
    /// provider asserts against — so the spec can never disagree with the dir we picked (which would
    /// be a hard load error), and the asset record names the tier actually loaded.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_tier_quant_reads_the_tier_manifest() {
        let root = mochi_root("quant", &["q4", "q8", "bf16"], true);
        assert_eq!(mochi_tier_quant(&root.join("q4")), Some(Quant::Q4));
        assert_eq!(mochi_tier_quant(&root.join("q8")), Some(Quant::Q8));
        // Dense tier ⇒ no quant assertion (the provider loads it dense).
        assert_eq!(mochi_tier_quant(&root.join("bf16")), None);
        // A manifest-less dir is dense-by-absence, never a guess.
        assert_eq!(mochi_tier_quant(&root.join("nope")), None);
        std::fs::remove_dir_all(&root).ok();
    }

    /// `mochi_1` is the engine id on BOTH backends — no `_distilled`-style split, unlike LTX. (The
    /// candle lane's equivalent is pinned by `candle_mochi_resolves_the_engine_and_never_falls_to_the_stub`.)
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_engine_id_is_the_model_id_and_matches_nothing_else() {
        assert_eq!(mochi_engine_id("mochi_1"), Some("mochi_1"));
        for other in [
            "wan_2_2",
            "ltx_2_3",
            "svd",
            "scail2_14b",
            "bernini",
            "mochi",
        ] {
            assert!(mochi_engine_id(other).is_none(), "{other} is not mochi_1");
        }
    }

    /// **The frame-stride seam** (sc-11992): Mochi must use its own `6k+1` snap, not the
    /// `wan_frame_count` else-arm every non-LTX model used to fall into.
    ///
    /// The failure mode is worth stating precisely, because the story brief mis-stated it. It claimed
    /// `wan_frame_count(150) = 145`, that `145 % 6 == 1` makes the runtime ACCEPT it, and that a 5 s
    /// request would therefore silently render 4.83 s. The arithmetic does not hold:
    /// `wan_frame_count(150)` is **149** (`150 − ((150−1) % 4)` = 150 − 1), and `149 % 6 == 5`, so the
    /// engine's `validate_request` **hard-rejects** it. The real bug is a LOUD failure — every
    /// default-duration Mochi job dies on "num_frames must be 1 + 6·k (got 149)" — not a silent
    /// wrong-length clip.
    ///
    /// Stronger still, a silent wrong length is IMPOSSIBLE on this pairing, and the test pins that:
    /// whenever the Wan stride's answer happens to land on Mochi's lattice it is provably EQUAL to
    /// Mochi's snap, so the substitution either agrees or errors — it never lies. (Why: if
    /// `wan(raw) = W` is `≡ 1 mod 6` then `raw − W ∈ 0..=3`, so `W` is Mochi's `lower` and
    /// `raw − W ≤ 3 ≤ 6 − (raw − W)` makes Mochi's nearest-with-ties-low pick `W` too.)
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn frames_are_the_mochi_lattice_not_wans() {
        // THE WIRING (not just the functions): the shared ladder BOTH generation lanes call must map
        // the mochi_1 MODEL ID onto the 6k+1 stride. This is what fails if a lane is (re-)pointed at
        // `wan_frame_count` — asserting `mochi_frame_count` vs `wan_frame_count` in isolation would
        // NOT, since both functions stay correct while the caller picks the wrong one.
        assert_eq!(
            video_frame_count("mochi_1", 150),
            151,
            "mochi_1 must resolve the 6k+1 stride through the shared ladder"
        );
        assert_eq!(
            video_frame_count("mochi_1", 150),
            mochi_frame_count(150),
            "the mochi arm must BE mochi_frame_count"
        );
        assert_ne!(
            video_frame_count("mochi_1", 150),
            wan_frame_count(150),
            "routing mochi through the wan stride is the bug this pins"
        );
        // The other families keep their own strides — adding Mochi must not perturb them.
        assert_eq!(video_frame_count("ltx_2_3", 150), ltx_frame_count(150));
        assert_eq!(video_frame_count("ltx_2_3_eros", 150), ltx_frame_count(150));
        assert_eq!(video_frame_count("wan_2_2", 150), wan_frame_count(150));
        assert_eq!(video_frame_count("scail2_14b", 150), wan_frame_count(150));
        assert_eq!(video_frame_count("bernini", 150), wan_frame_count(150));

        // The shipped default: 5 s x 30 fps = 150 raw frames.
        assert_eq!(
            mochi_frame_count(150),
            151,
            "the manifest's documented 5 s count"
        );
        assert_eq!(
            wan_frame_count(150),
            149,
            "NOT 145 — the brief's arithmetic was wrong"
        );
        assert_ne!(mochi_frame_count(150), wan_frame_count(150));
        // The default job's real failure on the wan arm: OFF-lattice ⇒ the engine rejects.
        assert_eq!(
            wan_frame_count(150) % 6,
            5,
            "149 is off Mochi's 6k+1 lattice, so validate_request hard-rejects the 5 s default"
        );

        // Mochi's own snap is always ON-lattice — the engine accepts every value we can produce.
        for raw in [1_u32, 6, 7, 30, 90, 150, 151, 163, 900] {
            assert_eq!(
                mochi_frame_count(raw) % 6,
                1,
                "mochi_frame_count({raw}) must sit on the 6k+1 lattice"
            );
        }

        // The invariant: the wan stride can never SILENTLY render a wrong length — over the whole
        // supported duration range it either equals mochi's snap or is rejected by the engine.
        let mut accepted = 0;
        for raw in 1_u32..=1_800 {
            let wan = wan_frame_count(raw);
            if wan % 6 == 1 {
                accepted += 1;
                assert_eq!(
                    wan,
                    mochi_frame_count(raw),
                    "raw {raw}: a wan value the mochi runtime ACCEPTS must equal mochi's own snap, \
                     or the substitution would silently change clip length"
                );
            }
        }
        assert!(accepted > 0, "the invariant must be exercised, not vacuous");
    }

    /// sc-12371: **no `*_raw_settings` builder may have an opinion about clip length.**
    ///
    /// This is the structural half of the fix, and the reason the class is dead rather than merely
    /// patched. Every builder used to compute its own `frameCount` from the REQUEST, independently
    /// of the arm that had just told the engine how many frames to render — two answers to one
    /// question. They diverged: the arms driving a Wan engine under a non-Wan model id (`bernini`,
    /// `scail2_14b`, `external_base_*`) rendered `wan_frame_count(raw)` = 149 while their builder
    /// recorded the unsnapped 150, and nothing failed loudly.
    ///
    /// Making the two computations agree would only fix today's models. Deleting the second
    /// computation means a builder has nothing to drift FROM: `record_frame_count` stamps the count
    /// off the encoded clip at the one seam every video job funnels through. This test is what stops
    /// a future builder from quietly reintroducing a second opinion.
    #[test]
    #[cfg(target_os = "macos")]
    fn no_raw_settings_builder_records_its_own_frame_count() {
        let req = |model: &str| {
            request(json!({
                "projectId": "p", "model": model, "mode": "text_to_video",
                "duration": 6.0, "fps": 25, "advanced": { "userKnob": "keep-me" }
            }))
        };
        let builders: Vec<(&str, Value)> = vec![
            ("stub", stub_raw_settings(&req("stub"))),
            ("wan", wan_raw_settings(&req("wan_2_2"), "wan2_2_ti2v_5b")),
            ("ltx", ltx_raw_settings(&req("ltx_2_3"))),
            ("mochi", mochi_raw_settings(&req("mochi_1"), None)),
            ("bernini", bernini_raw_settings(&req("bernini"))),
            ("scail2", scail2_raw_settings(&req("scail2_14b"), false)),
            ("svd", svd_raw_settings(&req("svd"))),
            (
                "wan_vace",
                wan_vace_raw_settings(&req("wan_2_2"), "wan_vace"),
            ),
            (
                "wan_vace_extend",
                wan_vace_extend_raw_settings(&req("wan_2_2")),
            ),
        ];
        for (name, raw) in &builders {
            let map = raw.as_object().expect("raw settings is an object");
            assert!(
                !map.contains_key("frameCount"),
                "{name}_raw_settings must not record a frameCount — `record_frame_count` counts it \
                 off the encoded clip; a builder that recomputes it is the sc-12371 bug returning"
            );
            // The builders still carry their own knobs — this is not vacuously passing on an empty
            // object, and SVD keeps `numFrames` (its engine burst KNOB, a different thing).
            assert!(
                !map.is_empty(),
                "{name}_raw_settings records nothing at all"
            );
        }
        // SVD's `numFrames` is a dispatched knob, not a clip length — it must survive.
        let svd = svd_raw_settings(&req("svd"));
        assert_eq!(svd["numFrames"], json!(25));
    }

    /// sc-12371: the stamped `frameCount` is the ENCODED CLIP's length, not any prediction of it.
    ///
    /// DISCRIMINATING BY CONSTRUCTION: every probe stamps a count that differs from what the request
    /// would have predicted, so a `record_frame_count` that reached back to `request.frame_count()`
    /// (or to any lattice at all) is RED. `bernini` at 6 s x 25 fps predicts 149 from the ladder and
    /// 150 from the old raw arm — the stamp records neither, because it records the clip.
    ///
    /// This is the property the "make both sides agree" fix could NOT buy: the engine is free to
    /// return a count that is nobody's prediction (SVD's fixed burst clamps to its own `numFrames`;
    /// a `validate()` re-snap or a truncated decode would too). The asset follows the file.
    #[test]
    #[cfg(target_os = "macos")]
    fn record_frame_count_records_the_encoded_clip_not_the_request() {
        let bernini = request(json!({
            "projectId": "p", "model": "bernini", "mode": "text_to_video",
            "duration": 6.0, "fps": 25
        }));
        assert_eq!(bernini.raw_frame_count(), 150);
        assert_eq!(bernini.frame_count(), 149, "the ladder predicts 149 here");

        // A clip whose length matches NEITHER prediction — the shape sc-12318's stub probe produces
        // (asked for 151, returned 1). The record must follow the frames that exist.
        let odd = EncodedClip { frames: 7, fps: 25 };
        let stamped = odd.record_frame_count(bernini_raw_settings(&bernini));
        assert_eq!(
            stamped["frameCount"],
            json!(7),
            "the asset records the clip that was encoded, not what the request predicted"
        );
        assert_ne!(stamped["frameCount"], json!(bernini.frame_count()));
        assert_ne!(stamped["frameCount"], json!(bernini.raw_frame_count()));
        // The builder's own knobs survive the stamp.
        assert_eq!(stamped["model"], json!("bernini"));
        assert_eq!(stamped["realModelInference"], json!(true));

        // The real pairing: `generate_bernini` hands the engine `wan_frame_count(raw)` and the
        // engine returns that many frames, so the asset records 149 — never the raw 150 the old
        // builder wrote.
        let rendered = wan_frame_count(bernini.raw_frame_count()) as usize;
        assert_eq!(rendered, 149);
        let clip = EncodedClip {
            frames: rendered,
            fps: 25,
        };
        let stamped = clip.record_frame_count(bernini_raw_settings(&bernini));
        assert_eq!(stamped["frameCount"], json!(149));

        // An already-stamped record is corrected, not duplicated (the stamp is the ONE writer).
        let restamped = EncodedClip { frames: 5, fps: 25 }.record_frame_count(stamped);
        assert_eq!(restamped["frameCount"], json!(5));

        // A non-object carries no keys to correct and passes through rather than being wrapped.
        assert_eq!(clip.record_frame_count(Value::Null), Value::Null);
    }

    /// sc-12371: `EncodedClip::measure` reads the clip, and `duration_seconds` is the FILE's running
    /// time — the second half of the story's harm ("claims a duration/frame count the file does not
    /// have"). The asset used to echo `request.duration`, so a 6.0 s ask on a Wan engine produced a
    /// 149-frame / 5.96 s file described as 6.0 s.
    ///
    /// DISCRIMINATING: 149 frames at 25 fps is 5.96 s, which is NOT the requested 6.0 — so a
    /// `duration` that reached back to `request.duration` is RED here.
    #[test]
    fn encoded_clip_measures_the_file_not_the_request() {
        let frame = |()| RgbFrame {
            width: 2,
            height: 2,
            pixels: vec![0; 12],
        };
        // What a bernini 6 s x 25 fps job actually produces: the Wan stride's 149 frames.
        let decoded = DecodedVideo {
            frames: (0..149).map(|_| frame(())).collect(),
            fps: 25,
            audio: None,
        };
        let clip = EncodedClip::measure(&decoded);
        assert_eq!(clip.frames, 149, "counted off the frames, not predicted");
        assert_eq!(clip.fps, 25);
        assert!(
            (clip.duration_seconds() - 5.96).abs() < 1e-9,
            "149 frames at 25 fps is 5.96 s — the file's real length, not the 6.0 s requested"
        );
        assert_ne!(
            clip.duration_seconds(),
            6.0,
            "the probe must discriminate: a `duration` echoing request.duration passes otherwise"
        );

        // `encode_inner` clamps the framerate with `.max(1)`; the record mirrors that clamp so the
        // asset always describes the container ffmpeg was actually handed.
        let zero_fps = DecodedVideo {
            frames: vec![frame(()), frame(())],
            fps: 0,
            audio: None,
        };
        let clip = EncodedClip::measure(&zero_fps);
        assert_eq!(clip.fps, 1, "mirrors encode_inner's `decoded.fps.max(1)`");
        assert_eq!(clip.duration_seconds(), 2.0);
    }

    /// The candle-lane half of [`no_raw_settings_builder_records_its_own_frame_count`] (sc-12371).
    /// Same invariant, different `#[cfg]`: these builders were the other half of the live bug —
    /// `candle_bernini` / `candle_scail2`(`_replace`) and `candle_wan_comfyui` (under an
    /// `external_base_*` id) all drive Wan engines whose ids `is_wan_model` does not match.
    #[test]
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    fn no_candle_raw_settings_builder_records_its_own_frame_count() {
        let req = |model: &str| {
            request(json!({
                "projectId": "p", "model": model, "mode": "text_to_video",
                "duration": 6.0, "fps": 25, "advanced": { "userKnob": "keep-me" }
            }))
        };
        let builders: Vec<(&str, Value)> = vec![
            (
                "candle_video",
                candle_video_raw_settings(&req("wan_2_2"), "repo"),
            ),
            (
                "candle_video/comfyui",
                candle_video_raw_settings(&req("external_base_wan22_comfyui"), "repo"),
            ),
            (
                "candle_scail2",
                candle_scail2_raw_settings(&req("scail2_14b"), false),
            ),
            (
                "candle_bernini",
                candle_bernini_raw_settings(&req("bernini")),
            ),
            (
                "wan_vace",
                wan_vace_raw_settings(&req("wan_2_2"), "wan_vace"),
            ),
            (
                "wan_vace_extend",
                wan_vace_extend_raw_settings(&req("wan_2_2")),
            ),
        ];
        for (name, raw) in &builders {
            let map = raw.as_object().expect("raw settings is an object");
            assert!(
                !map.contains_key("frameCount"),
                "{name}_raw_settings must not record a frameCount — `record_frame_count` counts it \
                 off the encoded clip; a builder that recomputes it is the sc-12371 bug returning"
            );
            assert!(
                !map.is_empty(),
                "{name}_raw_settings records nothing at all"
            );
        }
        // And the stamp still writes the clip's real length on this lane.
        let stamped = EncodedClip { frames: 7, fps: 25 }
            .record_frame_count(candle_bernini_raw_settings(&req("bernini")));
        assert_eq!(stamped["frameCount"], json!(7));
    }

    /// Mochi's REAL hosted q4 resident footprint, split by component: AsymmDiT 9.007 GiB + T5-XXL
    /// bf16 8.871 + AsymmVAE 0.856 = 18.73 GiB. Mirrors `mlx_fit_gate`'s `MOCHI_Q4_RESIDENT_BYTES`
    /// (which asserts the same total against the pure gate) — the preflight tests need them SPLIT
    /// across the A6 sibling layout so the on-disk scan has to fold the shared components to get the
    /// total right.
    ///
    /// On the SUPERSET cfg (sc-12306): both lanes ingest these same hosted tiers, so both gates budget
    /// against these exact bytes.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    const MOCHI_Q4_DIT_BYTES: u64 = 9_670_883_602;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    const MOCHI_Q4_TE_BYTES: u64 = 9_524_669_250;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    const MOCHI_Q4_VAE_BYTES: u64 = 919_551_200;

    /// A Mochi A6-layout root (q4 tier + shared `text_encoder`/`vae`/`tokenizer` siblings) whose
    /// `.safetensors` report the REAL hosted byte sizes, so the on-disk scan behind either lane's fit
    /// gate resolves the true ~18.73 GiB resident footprint instead of the 1-byte stubs `mochi_root`
    /// writes.
    ///
    /// The files are SPARSE: `set_len` sets the apparent size with zero allocated blocks on APFS/NTFS,
    /// and `sum_safetensors_bytes` reads `metadata().len()`. So this is instant and costs no disk —
    /// materializing 18.7 GiB of real zeros per test would not be viable.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn mochi_root_real_sized(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "mochi_preflight_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        for (relative, len) in [
            ("q4/transformer/model.safetensors", MOCHI_Q4_DIT_BYTES),
            ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
            ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(
            root.join("q4").join("split_model.json"),
            serde_json::to_string(
                &json!({ "quantized": true, "quantization_bits": 4, "quantization_group_size": 64 }),
            )
            .unwrap(),
        )
        .unwrap();
        root
    }

    /// sc-11992 review: pin the frame lattice AT THE SEAM THE GENERATION ARM CALLS, not just as a free
    /// function — `mochi_preflight` is the seam `generate_mochi` delegates the decision to. The arm
    /// ITSELF is pinned by `generate_mochi_using_*` (sc-12318); this stays the cheap, direct check of
    /// the seam's own contract across budgets and clip lengths.
    ///
    /// Kills the review mutation `video_frame_count(...)` → `wan_frame_count(...)`, which survived
    /// 835/0 green when this logic sat inline in `generate_mochi`: the wan stride answers 149 for the
    /// shipped 5 s default, and `149 % 6 == 5` is off the lattice the engine accepts.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_preflight_coerces_the_frame_count_on_mochis_lattice_by_model_id() {
        let root = mochi_root_real_sized("lattice");
        // A budget no clip can overflow, so ONLY the lattice is under test here.
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "512", || {
            mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 150, 848, 480)
        });
        std::fs::remove_dir_all(&root).ok();

        let preflight = out.expect("a 151-frame clip fits a 512 GB budget");
        assert_eq!(
            preflight.frames, 151,
            "the seam must snap the shipped 5 s default onto the 6k+1 lattice"
        );
        assert_eq!(
            preflight.frames,
            mochi_frame_count(150),
            "the seam's mochi arm must BE mochi_frame_count"
        );
        assert_ne!(
            preflight.frames,
            wan_frame_count(150),
            "routing the seam through the wan stride is the bug this story exists to fix"
        );
        assert_eq!(
            preflight.quant,
            Some(Quant::Q4),
            "the tier dir's split_model.json carries the quant the load asserts"
        );
    }

    /// sc-11992 review / epic AC "a missing-weights or unsupported environment shows an ACTIONABLE
    /// ERROR, not a crash": the fit gate must run on the generation seam, with the REAL coerced frame
    /// count. MLX's default error handler is `exit(-1)` — a hard process kill that cannot be mapped to
    /// a job error after the fact — so an over-budget job must be refused here or it takes the worker
    /// down with it.
    ///
    /// Kills BOTH remaining review mutations, each of which survived 835/0 green:
    ///   * hardcoding the gate's `frames` argument to `7` ⇒ the gate sees 4.66 GiB of decode instead
    ///     of 60.56, totals ~25 GiB, and ADMITS this job ⇒ `expect_err` fails.
    ///   * deleting the `mochi_fit_check(...)?` call ⇒ nothing rejects ⇒ `expect_err` fails.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_preflight_rejects_the_5s_default_on_a_64gb_mac() {
        let root = mochi_root_real_sized("reject");
        // A 64 GB Mac — the machine the epic's crash report names.
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 150, 848, 480)
        });
        std::fs::remove_dir_all(&root).ok();

        let message = out
            .expect_err(
                "the shipped 5 s default (151 frames) needs 18.73 GiB weights + 60.56 GiB untiled \
                 decode + 2 GiB OS reserve = 81.3 GiB, which does NOT fit a 64 GB Mac — admitting it \
                 is exactly the SIGKILL this gate exists to prevent",
            )
            .to_string();
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("Shorten the clip"),
            "leads with the only lever that moves the dominant term: {message}"
        );
    }

    /// The other half of the gate's contract: on the SAME 64 GB Mac a short clip is ADMITTED. Without
    /// this, `mochi_preflight_rejects_the_5s_default_on_a_64gb_mac` would still pass against a gate
    /// that blanket-refuses everything — so this is what makes the pair prove the gate is
    /// FRAME-SENSITIVE, which is the whole point of a decode peak that scales with clip length.
    ///
    /// Also independently kills the `wan_frame_count` substitution: `wan_frame_count(19) = 17`.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_preflight_admits_a_short_clip_on_the_same_64gb_mac() {
        let root = mochi_root_real_sized("admit");
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 19, 848, 480)
        });
        std::fs::remove_dir_all(&root).ok();

        let preflight = out.expect(
            "19 frames (the engine's own DEFAULT_FRAMES) needs 18.73 + 9.32 + 2 = 30.1 GiB, which \
             fits a 64 GB Mac and must NOT be rejected",
        );
        assert_eq!(
            preflight.frames, 19,
            "19 already sits on the 6k+1 lattice, so the snap is a no-op"
        );
    }

    // -----------------------------------------------------------------------------------------
    // Mochi 1 candle/CUDA VRAM fit gate (sc-12306). These run on the windows-candle.yml lane, which
    // is the ONLY lane that compiles `generate_candle_video` — the macOS lane and the Linux `parity`
    // job never see this code. `mochi_vram_preflight` takes its budget as a parameter precisely so the
    // whole decision is exercisable here with no CUDA driver and no GPU.
    // -----------------------------------------------------------------------------------------

    /// An RTX 5090 — the biggest consumer NVIDIA card, and the machine the story names. 32 GB total,
    /// all free (a cold card with nothing loaded).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    fn rtx_5090() -> Option<crate::vram_gate::VramBudget> {
        crate::vram_gate::apply_vram_cap(None, Some(32.0))
    }

    /// THE story: the shipped 5 s / 151-frame default is refused BEFORE the load + 64-step denoise, with
    /// a message naming the clip-length lever. Needs 18.73 GiB weights + 60.56 GiB untiled decode + 2 GiB
    /// headroom ≈ 81.3 GB against 32 GB — so on consumer hardware this is the DEFAULT path, not an edge
    /// case.
    ///
    /// Kills the mutations that a compile alone would not:
    ///   * hardcoding the gate's `frames` to a small number (e.g. B2's 7-frame smoke geometry) ⇒ the gate
    ///     sees 4.66 GiB of decode instead of 60.56, totals ~25 GB, and ADMITS ⇒ `expect_err` fails.
    ///   * passing `request.raw_frame_count()` instead of the coerced count ⇒ 150 not 151 — caught by
    ///     `mochi_vram_preflight_coerces_the_frame_count_on_the_candle_lane` below.
    ///   * dropping the frames term entirely (reusing the resolution-blind `predicted_peak_gb` shape) ⇒
    ///     admits ⇒ `expect_err` fails.
    ///   * reusing the MLX message ⇒ the "unified memory" assert fails.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn mochi_vram_preflight_rejects_the_5s_default_on_an_rtx_5090() {
        let root = mochi_root_real_sized("candle_reject");
        let out = mochi_vram_preflight(
            "mochi_1",
            &root.join("q4"),
            video_frame_count("mochi_1", 150),
            848,
            480,
            "0",
            rtx_5090(),
        );
        std::fs::remove_dir_all(&root).ok();

        let message = out
            .expect_err(
                "the shipped 5 s default (151 frames) needs ~81 GB and NO consumer NVIDIA GPU has \
                 that — admitting it burns a full 64-step denoise before a raw CUDA OOM",
            )
            .to_string();
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("151-frame"),
            "names the clip length that was refused, so the user can act on it: {message}"
        );
        assert!(
            message.contains("Shorten the clip"),
            "leads with the only lever that moves the dominant term — Mochi has one trained bucket \
             and the tier delta is ~11 GiB against a ~60 GiB decode: {message}"
        );
        assert!(
            message.contains("VRAM") && !message.contains("unified memory"),
            "must be CUDA-worded, not the MLX lane's Mac prose: {message}"
        );
        assert!(
            !message.contains("run on a Mac"),
            "telling a Windows/CUDA user to buy a Mac is the MLX message leaking: {message}"
        );
    }

    /// The other half of the contract: on the SAME 32 GB card a short clip is ADMITTED. Without this,
    /// the reject test above would pass against a gate that blanket-refuses Mochi on every consumer
    /// card — so this pair is what proves the gate is FRAME-SENSITIVE, which is the whole point of a
    /// decode peak that scales with clip length.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn mochi_vram_preflight_admits_a_short_clip_on_the_same_rtx_5090() {
        let root = mochi_root_real_sized("candle_admit");
        // 7 frames: 18.73 weights + 4.66 decode + 2 headroom ≈ 25.4 GB, which fits 32 GB.
        let out = mochi_vram_preflight("mochi_1", &root.join("q4"), 7, 848, 480, "0", rtx_5090());
        std::fs::remove_dir_all(&root).ok();

        let preflight = out.expect(
            "a 7-frame clip needs ~25 GB and FITS a 32 GB card — a gate that refuses this has \
             wall-rejected hardware that works",
        );
        assert_eq!(
            preflight.quant,
            Some(Quant::Q4),
            "the tier dir's split_model.json carries the quant the candle loader asserts — and it is \
             reachable ONLY through the gate"
        );
    }

    /// The gate must budget on the COERCED frame count, not the raw request: the decode peak is linear
    /// in frames, so gating on a length that never renders sizes the check against fiction. 150 raw
    /// snaps to 151 on Mochi's 6k+1 lattice.
    ///
    /// Kills the `wan_frame_count` substitution independently of the MLX lane's copy of this check:
    /// `wan_frame_count(150) = 149`, and `149 % 6 == 5` is off the lattice the engine accepts.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn mochi_vram_preflight_coerces_the_frame_count_on_the_candle_lane() {
        let root = mochi_root_real_sized("candle_lattice");
        let tier = root.join("q4");
        // A budget nothing overflows, so ONLY the frame arithmetic is under test.
        let huge = crate::vram_gate::apply_vram_cap(None, Some(512.0));

        // The seam the generation arm passes: `video_frame_count(&request.model, raw)`.
        let frames = video_frame_count("mochi_1", 150);
        assert_eq!(
            frames, 151,
            "the shipped 5 s default snaps onto the 6k+1 lattice"
        );
        assert_ne!(
            frames,
            wan_frame_count(150),
            "routing the candle lane through the wan stride would gate on 149 — off Mochi's lattice"
        );
        assert!(
            mochi_vram_preflight("mochi_1", &tier, frames, 848, 480, "0", huge).is_ok(),
            "151 frames fits a 512 GB budget — the gate rejects by BUDGET, never by duration alone"
        );

        // Frame-sensitivity at the seam: the SAME tier + card, two lengths, two verdicts. A 48 GB card
        // sits between the 7-frame (~25 GB) and 151-frame (~81 GB) totals.
        let card_48 = crate::vram_gate::apply_vram_cap(None, Some(48.0));
        assert!(
            mochi_vram_preflight("mochi_1", &tier, 7, 848, 480, "0", card_48).is_ok(),
            "a short clip fits 48 GB"
        );
        assert!(
            mochi_vram_preflight("mochi_1", &tier, frames, 848, 480, "0", card_48).is_err(),
            "the 5 s default does not fit the SAME 48 GB card — the gate must see the frame count"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-12344: the Wan weights-floor gate is keyed on the ENGINE id, and [`wan_vram_preflight`] is
    /// handed `engine_id` — pin that the two agree for every candle video model the router serves.
    ///
    /// This is the mutation nothing else catches. The sceneworks model id and the engine id differ only
    /// by underscores (`wan_2_2_t2v_14b` vs `wan2_2_t2v_14b`, `wan_2_2` vs `wan2_2_ti2v_5b`), so passing
    /// `&request.model` at the call site instead of `engine_id` still compiles, still renders, and reads
    /// **0 bytes** ⇒ no signal ⇒ the gate silently admits every job — dead code that reads as coverage,
    /// the exact failure this story was filed to avoid. It also fails loudly if a future engine is added
    /// to `candle_video_engine_id` without a decision about its fit gate.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn every_candle_video_model_maps_to_a_gated_or_recorded_exempt_engine() {
        let root = std::env::temp_dir().join(format!(
            "sc12344_engine_ids_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        // A populated Wan component tree, so a 0 read can only mean "not keyed to this engine".
        for component in ["transformer", "transformer_2", "text_encoder", "vae"] {
            let path = root.join(component).join("model.safetensors");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path)
                .unwrap()
                .set_len(1_000)
                .unwrap();
        }

        // The GATED engines: every Wan model must read real bytes through its resolved engine id.
        for model in ["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"] {
            let engine_id = candle_video_engine_id(model).expect("a candle video engine");
            assert!(
                crate::vram_gate::wan_weight_bytes(engine_id, &root) > 0,
                "{model} → {engine_id} must be gated; a 0 read means the component map and the engine \
                 id disagree, and the gate is INERT for this model"
            );
            // …and the sceneworks id is NOT the key. This is the slip above, made concrete.
            assert_eq!(
                crate::vram_gate::wan_weight_bytes(model, &root),
                0,
                "{model} is the sceneworks id, not the engine id — passing it reads no bytes"
            );
        }

        // The EXEMPT engines (sc-12344's recorded reason: their on-disk bytes are not their loaded set,
        // so a byte-derived floor would wall-reject working cards — see `vram_gate::wan_weight_components`).
        for model in ["ltx_2_3", "ltx_2_3_eros", "svd"] {
            let engine_id = candle_video_engine_id(model).expect("a candle video engine");
            assert_eq!(
                crate::vram_gate::wan_weight_bytes(engine_id, &root),
                0,
                "{model} → {engine_id} is exempt by decision; gating it on a dir sum would over-count"
            );
        }

        // Mochi rides its own frame-dependent gate (sc-12306), never this one.
        assert_eq!(
            crate::vram_gate::wan_weight_bytes(
                candle_video_engine_id("mochi_1").expect("a candle video engine"),
                &root
            ),
            0
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// The gate NO-OPS without a budget signal (the story's explicit AC) and without a weight signal.
    /// A worker on a card `nvidia-smi` cannot read, or pointed at a weights dir it cannot scan, must
    /// keep rendering exactly as it did before this gate existed — a fit gate that blocks on missing
    /// evidence is a regression, not a safety net (sc-12179: never wall-reject a machine that worked).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn mochi_vram_preflight_no_ops_without_a_budget_or_weight_signal() {
        let root = mochi_root_real_sized("candle_nosignal");

        // No budget: `nvidia_vram_budget_gb` → None (non-NVIDIA / unreadable) and no cap set.
        assert!(
            mochi_vram_preflight("mochi_1", &root.join("q4"), 151, 848, 480, "0", None).is_ok(),
            "no budget signal ⇒ admit — the 5 s default is refused ONLY against a real reading"
        );
        assert_eq!(
            crate::vram_gate::apply_vram_cap(None, None),
            None,
            "no real reading + no cap ⇒ no budget, so the wiring above really can pass None"
        );

        // No weights: a dir with no safetensors scans to 0 bytes ⇒ unmeasurable ⇒ admit, even on a
        // tiny card. The fixture needs its OWN root, for two reasons: `mochi_resident_bytes` folds the
        // shared `text_encoder/` + `vae/` siblings from the tier dir's PARENT, so (a) an "empty" tier
        // under `root` above would still scan ~9.7 GiB and be REJECTED — testing the exact opposite of
        // the intent — and (b) hanging it directly off `temp_dir()` would make the parent scan read
        // /tmp, so an unrelated `/tmp/text_encoder` would flake it. A private root has neither problem.
        let bare_root = std::env::temp_dir().join(format!(
            "mochi_candle_nosignal_bare_{}_{}",
            std::process::id(),
            line!()
        ));
        let bare = bare_root.join("q4");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(
            crate::mlx_fit_gate::mochi_resident_bytes(&bare),
            0,
            "the fixture must really have no weight signal, or the assert below proves nothing"
        );
        let out = mochi_vram_preflight(
            "mochi_1",
            &bare,
            151,
            848,
            480,
            "0",
            crate::vram_gate::apply_vram_cap(None, Some(4.0)),
        );
        std::fs::remove_dir_all(&bare_root).ok();
        std::fs::remove_dir_all(&root).ok();
        assert!(out.is_ok(), "unmeasurable weights ⇒ no signal ⇒ admit");
    }

    /// The on-disk scan must fold the SHARED `text_encoder/` + `vae/` siblings from the tier dir's
    /// PARENT — both providers set `supports_sequential_offload: false`, so all three components are
    /// held for the whole run. Summing only the tier dir would miss ~9.7 GiB (T5-XXL + VAE), over half
    /// the resident footprint, and silently under-gate every candle Mochi job.
    ///
    /// This pins that the candle lane reuses the SHARED scan rather than growing its own tier-only one.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn mochi_vram_preflight_folds_the_shared_siblings_on_the_candle_lane() {
        let root = mochi_root_real_sized("candle_siblings");
        assert_eq!(
            crate::mlx_fit_gate::mochi_resident_bytes(&root.join("q4")),
            MOCHI_Q4_DIT_BYTES + MOCHI_Q4_TE_BYTES + MOCHI_Q4_VAE_BYTES,
            "the candle gate must budget on DiT + T5 + VAE (~18.73 GiB), not the tier dir alone"
        );

        // Behavioral consequence: a card sized to fit the DiT alone must still refuse. DiT-only is
        // 9.01 + 4.66 decode + 2 = ~15.7 GB; the true total is 18.73 + 4.66 + 2 = ~25.4 GB. A 20 GB
        // card admits the former and must reject the latter.
        let out = mochi_vram_preflight(
            "mochi_1",
            &root.join("q4"),
            7,
            848,
            480,
            "0",
            crate::vram_gate::apply_vram_cap(None, Some(20.0)),
        );
        std::fs::remove_dir_all(&root).ok();
        assert!(
            out.is_err(),
            "a tier-only scan would admit this job on a 20 GB card and then OOM on the T5 + VAE"
        );
    }

    // -----------------------------------------------------------------------------------------
    // sc-12318 — driving the ASYNC generation arms.
    //
    // The `mochi_preflight_*` tests above pin the pure seam, but nothing asserted that
    // `generate_mochi` CALLS it: the sc-11992 review found a bypass (destructure the preflight, then
    // re-bind `frames = wan_frame_count(...)`) that stayed clippy-clean and green, because a unit test
    // could reach the free function but never the caller.
    //
    // The arm was believed undrivable — `async`, with a live `ApiClient`/`JobSnapshot`. That premise
    // was wrong on every count. `ApiClient::new` is pure (a reqwest client + a base URL, no I/O);
    // `JobSnapshot` builds from `serde_json::from_value` as it already does elsewhere; and both
    // `ensure_mochi_*_present` calls return `Ok(())` on their first line for a request that names no
    // tier. The ONLY real obstacle was `generate_video`'s hardcoded `inference_runtime::load`, which
    // `generate_video_using` now takes as a parameter — the same split `with_cached_generator` /
    // `with_cached_generator_using` already had one level down.
    //
    // No stub HTTP server is needed: `update_job` is the one API call the progress loop `?`-propagates,
    // and it fires only per progress event, so a silent probe generator never reaches it (`heartbeat`
    // swallows `WorkerError::Http`; `cancel_requested_peek` swallows everything).
    // -----------------------------------------------------------------------------------------

    /// What `generate_mochi_using` actually handed the engine, captured from inside the load+run.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[derive(Clone, Default)]
    struct ArmProbe {
        /// The `GenerationRequest` the arm built, or `None` if generation never ran.
        request: std::sync::Arc<std::sync::Mutex<Option<GenerationRequest>>>,
        /// The `LoadSpec` the arm resolved — carries the tier dir + quant marker.
        spec: std::sync::Arc<std::sync::Mutex<Option<LoadSpec>>>,
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    impl ArmProbe {
        /// The loader to hand `generate_mochi_using`. Records the spec, then yields a generator that
        /// records the request — so a test can assert on both the load and the generate side of the arm
        /// without a tensor backend, weights, or a GPU.
        fn loader(
            &self,
        ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static
        {
            let seen_spec = std::sync::Arc::clone(&self.spec);
            let seen_request = std::sync::Arc::clone(&self.request);
            move |engine_id, spec| {
                *seen_spec.lock().unwrap() = Some(spec.clone());
                Ok(Box::new(ProbeGenerator {
                    descriptor: gen_core::ModelDescriptor {
                        // Leaked so the descriptor's `&'static str` id reflects the engine actually
                        // asked for; the probe outlives nothing, so the leak is bounded by the test.
                        id: Box::leak(engine_id.to_owned().into_boxed_str()),
                        family: "test",
                        backend: "probe",
                        modality: gen_core::Modality::Video,
                        capabilities: gen_core::Capabilities::default(),
                    },
                    request: seen_request,
                }))
            }
        }

        /// The frame count that reached the engine. Panics if generation never ran.
        fn engine_frames(&self) -> u32 {
            self.request
                .lock()
                .unwrap()
                .as_ref()
                .expect("the arm reached the engine")
                .frames
                .expect("a video request carries a frame count")
        }

        /// Whether the arm got as far as loading an engine at all. macOS-only: the pre-load assertion
        /// it serves belongs to the fit gate, which is an MLX-lane concern — the candle lane has none,
        /// so carrying this there would be dead code under `clippy --all-targets -D warnings`.
        #[cfg(target_os = "macos")]
        fn loaded(&self) -> bool {
            self.spec.lock().unwrap().is_some()
        }
    }

    /// A backend-neutral `Generator` that records the request and returns a minimal clip.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    struct ProbeGenerator {
        descriptor: gen_core::ModelDescriptor,
        request: std::sync::Arc<std::sync::Mutex<Option<GenerationRequest>>>,
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    impl Generator for ProbeGenerator {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            *self.request.lock().unwrap() = Some(req.clone());
            // Deliberately silent: a `Progress` event would drive `update_job`, the only API call the
            // progress loop hard-fails on, and this test has no server behind `api`.
            Ok(GenerationOutput::Video {
                frames: vec![gen_core::Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0u8; 12],
                }],
                fps: req.fps.unwrap_or(30),
                audio: None,
            })
        }
    }

    /// A `JobSnapshot` for a Mochi video job. `payload.model` is what the completion metrics read.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn mochi_job_snapshot() -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job-mochi-1",
            "type": "video_generate",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": { "model": "mochi_1" },
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": "test-worker",
            "progress": 0.0,
            "stage": "queued",
            "message": "queued",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 1,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": false,
            "createdAt": "2026-07-16T00:00:00Z",
            "updatedAt": "2026-07-16T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        }))
        .expect("the mochi job snapshot deserializes")
    }

    /// Drive `generate_mochi_using` against `probe` with the Mochi tier at `root` and an MLX budget of
    /// `cap_gb`, from a plain `#[test]` — the env overrides are process-global, so the whole async run
    /// has to sit inside the [`ENV_LOCK`] the sync `temp_env_vars` holds.
    ///
    /// `api` points at a closed port on purpose: reaching the network would be a bug in this test, and
    /// an unroutable URL makes that fail loudly rather than depending on a stub's fidelity.
    #[cfg(target_os = "macos")]
    fn drive_mochi_arm(
        root: &Path,
        cap_gb: &str,
        probe: &ArmProbe,
        request: &VideoRequest,
    ) -> WorkerResult<(DecodedVideo, Value)> {
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            api_url: "http://127.0.0.1:0".to_owned(),
            ..Settings::from_env()
        };
        let job = mochi_job_snapshot();
        let loader = probe.loader();
        temp_env_vars(
            &[
                (MOCHI_DIR_ENV, root.to_str().expect("utf-8 fixture root")),
                (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, cap_gb),
            ],
            || {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("test runtime builds")
                    .block_on(generate_mochi_using(
                        &ApiClient::new(&settings),
                        &settings,
                        &job,
                        request,
                        "mochi_1",
                        "mlx",
                        loader,
                    ))
            },
        )
    }

    /// **The caller-side pin the story asked for.** Drives `generate_mochi_using` end to end and asserts
    /// on the frame count that REACHED the engine — so it fails for any way the arm can get the lattice
    /// wrong, not just the one the pure seam covers.
    ///
    /// Kills the sc-11992 review bypass that survived `mochi_preflight`: destructuring the preflight and
    /// then re-binding `frames = wan_frame_count(request.raw_frame_count())` is clippy-clean and leaves
    /// the gate running on the correct 151, so every existing test stays green — but the engine is handed
    /// 149, and `149 % 6 == 5` is off Mochi's lattice, which `validate_request` hard-rejects.
    #[cfg(target_os = "macos")]
    #[test]
    fn generate_mochi_using_hands_the_engine_the_mochi_lattice_frame_count() {
        let root = mochi_root_real_sized("arm_lattice");
        let probe = ArmProbe::default();
        // A budget no clip can overflow, so the fit gate cannot be what fails this test.
        let out = drive_mochi_arm(&root, "512", &probe, &mochi_request(json!({})));
        std::fs::remove_dir_all(&root).ok();

        let (decoded, raw_settings) = out.expect("a 151-frame clip fits a 512 GB budget");
        assert_eq!(
            probe.engine_frames(),
            151,
            "the ARM must hand the engine the shipped 5 s default snapped onto Mochi's 6k+1 lattice"
        );
        assert_ne!(
            probe.engine_frames(),
            wan_frame_count(150),
            "routing the arm through the wan stride is the bug epic 1788 exists to fix — the engine \
             hard-rejects 149 (149 % 6 == 5)"
        );
        assert_eq!(
            probe
                .spec
                .lock()
                .unwrap()
                .as_ref()
                .expect("a load ran")
                .quantize,
            Some(Quant::Q4),
            "the arm must carry the resolved tier's quant onto the LoadSpec"
        );
        // sc-12371 REPLACED this pin's premise. It used to assert that the arm's `raw_settings`
        // carried a `frameCount` equal to `probe.engine_frames()` — i.e. that core's INDEPENDENT
        // mirror of the lattice happened to agree with the arm. That mirror is gone: the builder no
        // longer has an opinion, and the asset's `frameCount` is stamped from the ENCODED clip at
        // the funnel (`record_frame_count`). So the arm must carry no count...
        assert!(
            raw_settings.get("frameCount").is_none(),
            "the arm must not record a clip length — the funnel stamps it off the encoded frames"
        );
        // ...and this probe is the exact case that proves why recording beats predicting: the arm
        // asked the engine for 151 frames and the stub returned ONE. The old builder would have
        // written 151 onto a 1-frame clip — a sidecar lying about a file that is right there. The
        // stamp records what actually came back.
        assert_eq!(
            decoded.frames.len(),
            1,
            "the probe returns a one-frame clip"
        );
        assert_eq!(
            EncodedClip::measure(&decoded).record_frame_count(raw_settings)["frameCount"],
            json!(1),
            "the asset records the clip that exists (1), not the 151 the arm requested"
        );
        assert_ne!(
            u64::from(probe.engine_frames()),
            1,
            "the probe must discriminate: if the stub returned exactly what was requested, \
             recording and predicting would agree and this would pin nothing"
        );
    }

    /// The gate half of the same pin: on a 64 GB Mac the shipped 5 s default must be REFUSED, and
    /// refused **before the engine is loaded** — MLX's default error handler is `exit(-1)`, so an
    /// over-budget job that reaches the load takes the worker down with it rather than failing the job.
    ///
    /// `probe.loaded()` is what makes this a pre-load assertion rather than just an error check: it is
    /// only ever set from inside the loader, so it proves the arm short-circuited on the gate. Kills the
    /// "inline the seam" bypass (`video_frame_count` + `mochi_tier_quant`, no `mochi_fit_check`), which
    /// is green today and caught only by clippy's dead-code cascade.
    #[cfg(target_os = "macos")]
    #[test]
    fn generate_mochi_using_refuses_an_over_budget_clip_before_loading_the_engine() {
        let root = mochi_root_real_sized("arm_reject");
        let probe = ArmProbe::default();
        let out = drive_mochi_arm(&root, "64", &probe, &mochi_request(json!({})));
        std::fs::remove_dir_all(&root).ok();

        let message = out
            .err()
            .expect(
                "the shipped 5 s default (151 frames) needs 18.73 GiB weights + 60.56 GiB untiled \
                 decode + 2 GiB OS reserve = 81.3 GiB, which does NOT fit a 64 GB Mac",
            )
            .to_string();
        assert!(
            message.contains("Shorten the clip"),
            "the arm must surface the gate's actionable error: {message}"
        );
        assert!(
            !probe.loaded(),
            "the fit gate must refuse BEFORE the engine load — a load that starts over-budget is the \
             exit(-1) SIGKILL the gate exists to prevent, and cannot be mapped to a job error after \
             the fact"
        );
    }

    /// **The floor-vs-full seam** — the one case where sc-12322's two gates DISAGREE, and the only
    /// thing that pins the post-download `mochi_preflight` call now that `mochi_precheck` runs the same
    /// `mochi_fit_check` earlier.
    ///
    /// Why the sibling tests can't cover this: they stage a COMPLETE tier, so the pre-check measures the
    /// full weights and refuses first — the preflight becomes unobservable, and inlining it away stays
    /// green (caught only by clippy's dead-code cascade, not by `cargo test`; sc-12318 AC #1).
    ///
    /// This makes the two verdicts diverge, with **no download and no network**:
    ///   * `mlxQuantize: 8` ⇒ the pre-check names the **absent** `q8/` dir, so `mochi_resident_bytes`
    ///     folds only the shared `text_encoder/` + `vae/` siblings — a 9.73 GiB FLOOR (`q4/` is not a
    ///     folded component). At 115 frames: 9.73 + 46.58 + 2 = **58.31 ⇒ ADMITS** a 64 GB budget.
    ///   * `ensure_mochi_q8_present` then no-ops rather than fetching: `drive_mochi_arm`'s `data_dir`
    ///     has no HF cache, so `huggingface_snapshot_dir` is `None` and `ensure_mochi_tier_present`
    ///     early-returns `Ok(())`. That is what keeps this hermetic — the divergence normally needs a
    ///     real ~13 GiB download to materialize the tier.
    ///   * `resolve_mochi_model_dir` falls back to the installed `q4/`, so `mochi_preflight` sees the
    ///     FULL 18.73 GiB: 18.73 + 46.58 + 2 = **67.32 ⇒ REFUSES**.
    ///
    /// So the job is admitted by the floor and refused by the full check — exactly the case the
    /// post-download gate exists for. It is REACHABLE in production (a 64 GB Mac, bf16/q8 not yet
    /// installed, ~2.8-4.2 s clip — well under `hardMaxDuration`'s 151 frames), and without the
    /// preflight those jobs run into MLX's `exit(-1)`.
    ///
    /// Kills the sc-12318 AC #1 mutation — inline `video_frame_count` + `mochi_tier_quant` in place of
    /// `mochi_preflight` — which passes 856/0 on main: the pre-check admits here, so nothing else
    /// refuses, the engine loads, and both assertions below fail.
    #[cfg(target_os = "macos")]
    #[test]
    fn generate_mochi_using_refuses_after_the_fetch_when_the_precheck_floor_admitted() {
        // q4 installed at its real hosted size; q8 is NOT — the fixture stages no other tier.
        let root = mochi_root_real_sized("arm_floor_vs_full");
        let probe = ArmProbe::default();
        // 3.8 s x 30 fps = 114 raw ⇒ 115 frames at 848x480, inside the 109..127 divergence window on a
        // 64 GB Mac. Every geometry term is EXPLICIT because `VideoRequest`'s defaults are none of
        // Mochi's, and each default silently moves this OFF the seam — into the region where the floor
        // and the full check agree — leaving a green test that proves nothing. The window is only ~9 GiB
        // wide (the q4 tier), so it is unforgiving of drift:
        //   * `fps` defaults to 25, not Mochi's 30 ⇒ 95 raw ⇒ 97 frames.
        //   * the frame defaults to 768x512, not Mochi's one trained 848x480 bucket.
        //   * `modelManifestEntry` carries `requiresDimensionsMultipleOf: 16` (Mochi's declared value)
        //     because the DEFAULT stride is 64 — under which 848 floors to 832 and the decode term,
        //     which is linear in pixels, lands ~2% low. Production passes the manifest entry; a fixture
        //     that omits it is not running the geometry it claims to.
        let request = request(json!({
            "projectId": "p",
            "model": "mochi_1",
            "mode": "text_to_video",
            "prompt": "a calico kitten",
            "duration": 3.8,
            "fps": 30,
            "width": 848,
            "height": 480,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } },
            "advanced": { "mlxQuantize": 8 },
        }));
        // Pin the fixture's own premises — the window is narrow, so a drifted default would silently
        // move this off the seam and the test would prove nothing.
        assert_eq!(request.raw_frame_count(), 114, "3.8 s x 30 fps");
        assert_eq!(
            mochi_frame_count(114),
            115,
            "115 sits in the divergence window"
        );
        assert_eq!((request.width, request.height), (848, 480));
        assert_eq!(
            mochi_tier_order(&request).first().copied(),
            Some("q8"),
            "the pre-check must name the ABSENT q8 tier, or it would measure q4 and refuse early"
        );
        assert!(!root.join("q8").exists(), "q8 must NOT be installed");
        assert_eq!(
            crate::mlx_fit_gate::mochi_resident_bytes(&root.join("q8")),
            MOCHI_Q4_TE_BYTES + MOCHI_Q4_VAE_BYTES,
            "the pre-check's floor must be the shared siblings ONLY — if it folded q4 it would refuse \
             before the fetch and this test would collapse into its sibling"
        );

        let out = drive_mochi_arm(&root, "64", &probe, &request);
        std::fs::remove_dir_all(&root).ok();

        let message = out
            .err()
            .expect(
                "the 9.73 GiB shared FLOOR admits this clip (58.31 < 64), but the resolved q4 tier's \
                 full 18.73 GiB does not (67.32 > 64) — the post-download gate must refuse it",
            )
            .to_string();
        assert!(
            message.contains("Shorten the clip"),
            "the refusal must be the fit gate's, i.e. `mochi_preflight` really ran: {message}"
        );
        assert!(
            !probe.loaded(),
            "an over-budget job that reaches the load is the exit(-1) SIGKILL the gate exists to \
             prevent — the pre-check's floor cannot catch this one, only the preflight can"
        );
    }

    /// Stage a REAL HuggingFace hub cache holding Mochi's shared `coRequisite` components and **no
    /// tier dir** — the disk state of a machine that has the ~9.7 GiB shared download but has not
    /// fetched a tier. Returns the hub dir to point `HF_HUB_CACHE` at.
    ///
    /// Goes through `safe_repo_dir_name` rather than hand-writing `models--SceneWorks--mochi-1-mlx`, so
    /// the fixture cannot drift from the slug the real resolver computes. Sparse, like
    /// [`mochi_root_real_sized`].
    #[cfg(target_os = "macos")]
    fn mochi_hf_cache_shared_only(tag: &str) -> PathBuf {
        let hub = std::env::temp_dir().join(format!(
            "mochi_hf_hub_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        let repo_dir = hub.join(format!(
            "models--{}",
            sceneworks_core::hf_home::safe_repo_dir_name(MOCHI_REPO).expect("the mochi repo slug")
        ));
        let snapshot = repo_dir.join("snapshots").join(MOCHI_REVISION);
        for (relative, len) in [
            ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
            ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
        ] {
            let path = snapshot.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }
        std::fs::create_dir_all(snapshot.join("tokenizer")).unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), MOCHI_REVISION).unwrap();
        hub
    }

    /// **Closes the residual sc-12322 recorded against this story.** Its own mutation table ends with
    /// *"pre-check moved after the fetches ⇒ survives everything. Structural to the untestable async
    /// arm; tracked by sc-12318"* — the ORDER of `mochi_precheck` against `ensure_mochi_*_present` is
    /// the one decision `generate_mochi` itself owns, and the whole point of sc-12322: refusing after
    /// the download is the bug, not the fix.
    ///
    /// How it discriminates. The job asks for **q8**, and the staged cache has the shared components
    /// but no q8 tier — so `ensure_mochi_q8_present` is past its `!mochi_wants_q8` early-out and MUST
    /// attempt a real `ensure_hf_files_cached` fetch. The two orders then give different errors:
    ///   * pre-check first (correct) ⇒ the gate's actionable "Shorten the clip", no network touched.
    ///   * pre-check after the fetches ⇒ a transport error from the download instead ⇒ RED.
    ///
    /// **`huggingface_base_url` is what makes that hermetic, and it is NOT `api_url`.** The fetch dials
    /// `settings.huggingface_base_url` (`HuggingFaceSnapshot::resolve` → `{base_url}/api/models/…/tree/…`);
    /// `api_url` only carries progress/heartbeat. `Settings::from_env()` defaults the former to the real
    /// `https://huggingface.co`, so overriding only `api_url` would send this test's FAILURE path to the
    /// live internet: it still goes red (the download's `report_download_progress` hard-`?`s on a
    /// heartbeat to the closed `api_url`), but only after really resolving the tree — and the tier is a
    /// public ungated repo, so that resolve succeeds and bytes can start landing before the heartbeat
    /// trips. Pinning BOTH at a closed port keeps the whole thing offline and instant. Verified by
    /// pointing the base at a sentinel host and reading it back out of the mutation's error.
    ///
    /// `HF_HUB_CACHE` is pinned at the fixture because `huggingface_hub_cache_dir` reads it (and
    /// `HF_HOME`) BEFORE `data_dir` — left ambient, a dev box's real cache would decide this test.
    /// `SCENEWORKS_MLX_MOCHI_DIR` is cleared for the same reason: it would bypass the HF resolution
    /// this test is built on.
    #[cfg(target_os = "macos")]
    #[test]
    fn generate_mochi_using_refuses_before_paying_for_the_tier_download() {
        let hub = mochi_hf_cache_shared_only("arm_precheck");
        let probe = ArmProbe::default();
        let request = mochi_request(json!({ "mlxQuantize": 8 }));
        let settings = Settings {
            data_dir: hub.join("unused-data-dir"),
            api_url: "http://127.0.0.1:0".to_owned(),
            // The URL the tier fetch actually dials — see above. Unroutable ⇒ the failure path can
            // never reach the real hub.
            huggingface_base_url: "http://127.0.0.1:0".to_owned(),
            ..Settings::from_env()
        };
        let job = mochi_job_snapshot();
        let loader = probe.loader();
        let out = temp_env_vars(
            &[
                ("HF_HUB_CACHE", hub.to_str().expect("utf-8 fixture hub")),
                (MOCHI_DIR_ENV, ""),
                (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64"),
            ],
            || {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("test runtime builds")
                    .block_on(generate_mochi_using(
                        &ApiClient::new(&settings),
                        &settings,
                        &job,
                        &request,
                        "mochi_1",
                        "mlx",
                        loader,
                    ))
            },
        );
        std::fs::remove_dir_all(&hub).ok();

        let message = out
            .err()
            .expect("a 151-frame clip cannot fit a 64 GB Mac even on the shared-components floor")
            .to_string();
        assert!(
            message.contains("Shorten the clip"),
            "the refusal must come from the PRE-DOWNLOAD gate, before `ensure_mochi_q8_present` \
             charges ~13-20 GiB for an answer it never needed. A transport error here means the \
             pre-check now runs after the fetches: {message}"
        );
        assert!(!probe.loaded(), "nothing may load on a refused job");
    }

    /// The candle half of the sc-12318 pin, and the reason the probe harness sits on the superset cfg:
    /// `generate_candle_video`'s `video_frame_count(&request.model, request.raw_frame_count())` is the
    /// same unpinned caller `generate_mochi` had — swapping the call for `wan_frame_count` hands the
    /// engine 149 for the shipped 5 s Mochi default, which `validate_request` hard-rejects
    /// (`149 % 6 == 5`).
    ///
    /// Runs on the windows-candle lane (`cargo test -p sceneworks-worker --features backend-candle`),
    /// which is what makes this arm genuinely covered rather than asserted — the macOS lane never
    /// compiles it.
    ///
    /// 1-byte stub weights are enough: unlike the MLX arm there is no on-disk byte scan to satisfy, and
    /// the residency policy reads them as negligible and stays `Resident`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn generate_candle_video_using_hands_the_engine_the_mochi_lattice_frame_count() {
        let root = mochi_root("arm_lattice_candle", &["q4"], true);
        let probe = ArmProbe::default();
        let request = mochi_request(json!({}));
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            api_url: "http://127.0.0.1:0".to_owned(),
            ..Settings::from_env()
        };
        let job = mochi_job_snapshot();
        let loader = probe.loader();
        let out = temp_env_var(
            MOCHI_DIR_ENV,
            root.to_str().expect("utf-8 fixture root"),
            || {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("test runtime builds")
                    .block_on(generate_candle_video_using(
                        &ApiClient::new(&settings),
                        &settings,
                        &job,
                        &request,
                        std::path::Path::new(""),
                        "candle",
                        loader,
                    ))
            },
        );
        std::fs::remove_dir_all(&root).ok();

        let (_decoded, adapter, _raw_settings) =
            out.expect("the candle mochi arm runs to completion");
        assert_eq!(
            probe.engine_frames(),
            151,
            "the candle ARM must hand the engine the 5 s default snapped onto Mochi's 6k+1 lattice"
        );
        assert_ne!(
            probe.engine_frames(),
            wan_frame_count(150),
            "routing this call through the wan stride hands the engine 149, which it hard-rejects"
        );
        assert_eq!(
            adapter, CANDLE_MOCHI_ADAPTER,
            "a candle-rendered clip records the candle adapter, not the MLX one"
        );
    }

    /// The PRE-DOWNLOAD disk state (sc-12322): the shared `coRequisite` components at their REAL
    /// hosted sizes, and NO tier dir — exactly what a machine looks like when a job has toggled q8/bf16
    /// but `ensure_mochi_*_present` has not run yet. Sparse, like [`mochi_root_real_sized`].
    #[cfg(target_os = "macos")]
    fn mochi_root_shared_only(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "mochi_precheck_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        for (relative, len) in [
            ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
            ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        root
    }

    /// THE sc-12322 behavior: the 5 s default is refused on a 64 GB Mac with NO tier on disk — i.e.
    /// before the ~13-20 GiB download the old ordering charged for the same answer.
    ///
    /// The fixture's premise — the decision is reached with NO tier dir on disk — is what makes this a
    /// PRE-download test rather than a second copy of the `mochi_preflight` pair.
    ///
    /// The three mutations, MEASURED not assumed:
    ///   * pre-check unconditionally admits ⇒ **RED here** (this test + the floor test). This is the
    ///     story's required mutation-check.
    ///   * `mochi_precheck(...)?` deleted from `generate_mochi` ⇒ **RED** at
    ///     `generate_mochi_using_refuses_before_paying_for_the_tier_download` (sc-12318), which drives
    ///     the arm itself. Previously this was green-but-for-clippy's dead-code cascade.
    ///   * pre-check MOVED to after `ensure_mochi_*_present` ⇒ **RED**, same test: the refusal comes
    ///     back as a transport error from the download instead of the gate's message. This was recorded
    ///     here as "survives everything — structural to the untestable async arm"; that premise was
    ///     wrong (the arm IS drivable) and the residual is now closed.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_precheck_refuses_the_5s_default_before_any_tier_is_downloaded() {
        let root = mochi_root_shared_only("reject");
        // The requested tier does NOT exist yet — the pre-check must still decide.
        let tier_dir = root.join("bf16");
        assert!(
            !tier_dir.exists(),
            "the fixture must model the pre-download state"
        );
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_precheck("mochi_1", "mochi_1", Some(&tier_dir), 150, 848, 480)
        });
        std::fs::remove_dir_all(&root).ok();

        let message = out
            .expect_err(
                "the shipped 5 s default (151 frames) needs 9.73 GiB of already-present shared \
                 components + 60.56 GiB untiled decode + 2 GiB OS reserve = 72.29 GiB, which does NOT \
                 fit a 64 GB Mac — so the answer is knowable BEFORE the ~20 GiB bf16 download",
            )
            .to_string();
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("Shorten the clip"),
            "the early refusal must carry the same actionable lever as the full gate: {message}"
        );
    }

    /// Why the pre-check measures the WEIGHTS FLOOR and not just the decode term (sc-12322's own
    /// description proposed decode-only). On a 64 GB Mac the decode + OS reserve for the shipped
    /// default is 62.56 GiB — it FITS, by 1.44 GiB. So a weights-free pre-check admits exactly the job
    /// in this story's title, downloads ~20 GiB, and only then refuses: the bug, unfixed.
    ///
    /// The already-present shared co-requisites are what push the floor over the line. This test fails
    /// the moment someone "simplifies" the pre-check to decode-only.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_precheck_needs_the_shared_weights_floor_to_catch_the_64gb_default() {
        // Decode + OS alone does NOT bust a 64 GB budget at the shipped default …
        // (`mochi_decode_peak_gb` moved to the backend-neutral `fit_gate` in sc-12306, when the candle
        // video lane grew the same frame-dependent gate; the arithmetic is unchanged.)
        let decode_only = crate::fit_gate::mochi_decode_peak_gb(151, 848, 480) + 2.0;
        assert!(
            decode_only < 64.0,
            "a weights-free pre-check would ADMIT the titled case ({decode_only:.2} < 64) — which is \
             precisely why the floor is load-bearing"
        );

        // … but the floor the shared components already put on disk does.
        let root = mochi_root_shared_only("floor");
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_precheck(
                "mochi_1",
                "mochi_1",
                Some(&root.join("bf16")),
                150,
                848,
                480,
            )
        });
        std::fs::remove_dir_all(&root).ok();
        assert!(
            out.is_err(),
            "the shared co-requisites (~9.73 GiB) are already on disk before any tier fetch, and \
             {decode_only:.2} + 9.73 > 64 — the pre-check must use them"
        );
    }

    /// The other direction, and the property that makes an early gate SAFE: a clip that fits is NOT
    /// refused early. Without this the pre-check could blanket-reject and still pass the test above —
    /// and a pre-check that refuses a fitting config is strictly worse than the late refusal it
    /// replaces, because no download can ever redeem it.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_precheck_admits_a_short_clip_on_the_same_64gb_mac() {
        let root = mochi_root_shared_only("admit");
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_precheck("mochi_1", "mochi_1", Some(&root.join("bf16")), 19, 848, 480)
        });
        std::fs::remove_dir_all(&root).ok();

        out.expect(
            "19 frames needs 9.73 floor + 9.32 decode + 2 OS = 21.1 GiB — it fits, so the pre-check \
             must let the download proceed",
        );
    }

    /// Fail-open, twice over: no locatable root (`None`) and a root with nothing measurable on disk
    /// both ADMIT. The pre-check never blocks without evidence — same rule as the gate it calls
    /// (`weight_bytes == 0` ⇒ no signal). A first-ever Mochi job must reach the download, not be
    /// refused for not having downloaded yet.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_precheck_admits_without_a_signal() {
        // No root at all ⇒ nothing to measure.
        temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_precheck("mochi_1", "mochi_1", None, 150, 848, 480)
        })
        .expect("no locatable model root ⇒ no signal ⇒ admit, never a phantom refusal");

        // An empty root ⇒ the scan sums 0 bytes ⇒ still no signal, even though the clip is huge.
        let root =
            std::env::temp_dir().join(format!("mochi_precheck_empty_{}", std::process::id()));
        std::fs::create_dir_all(root.join("bf16")).unwrap();
        let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
            mochi_precheck(
                "mochi_1",
                "mochi_1",
                Some(&root.join("bf16")),
                150,
                848,
                480,
            )
        });
        std::fs::remove_dir_all(&root).ok();
        out.expect("unmeasurable weights ⇒ no signal ⇒ admit and let the full gate decide later");
    }

    /// The pre-check must measure the tier the job ASKED for, not the smaller one already installed.
    /// `resolve_mochi_model_dir` deliberately falls back to a smaller complete tier — reusing it here
    /// would measure q4's bytes for a bf16 job and, worse, hard-error when nothing is installed at all.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_precheck_dir_names_the_requested_tier_not_the_installed_fallback() {
        let root = mochi_root("precheck_dir", &["q4"], true);
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            ..Settings::from_env()
        };
        let dir = temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
            // bf16 requested; only q4 is on disk.
            mochi_precheck_dir(&settings, &mochi_request(json!({ "mlxQuantize": 16 })))
        });
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            dir.as_deref(),
            Some(root.join("bf16").as_path()),
            "the requested tier's path, existence NOT required — measuring the q4 fallback would \
             under-count a bf16 job's weights against a budget it must clear"
        );
    }

    /// Mochi is text-to-video ONLY (`conditioning: []` on both descriptors). `VideoRequest`'s mode
    /// DEFAULTS to `image_to_video` when the payload omits one, so the mode gate is what keeps a
    /// conditioning-shaped job out of the Mochi route — without it an unmoded payload would route to
    /// Mochi and fail deep in the engine.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_available_is_text_to_video_only() {
        let root = mochi_root("mode", &["q4"], true);
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            ..Settings::from_env()
        };
        temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
            let t2v = request(json!({
                "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
            }));
            assert!(
                mochi_available(&t2v, &settings),
                "a t2v mochi job with resolvable weights must route to the Mochi engine"
            );
            // Every non-t2v mode — including the DEFAULT the DTO applies when `mode` is absent.
            for mode in [
                "image_to_video",
                "first_last_frame",
                "extend_clip",
                "video_bridge",
                "replace_person",
                "animate_character",
            ] {
                let req = request(json!({
                    "projectId": "p", "model": "mochi_1", "mode": mode, "prompt": "p"
                }));
                assert!(
                    !mochi_available(&req, &settings),
                    "mochi has no {mode} surface (conditioning: [])"
                );
            }
            let unmoded = request(json!({ "projectId": "p", "model": "mochi_1", "prompt": "p" }));
            assert_eq!(unmoded.mode, "image_to_video", "the DTO's default mode");
            assert!(
                !mochi_available(&unmoded, &settings),
                "a payload with no mode defaults to i2v and must NOT reach the t2v-only engine"
            );
        });
        std::fs::remove_dir_all(&root).ok();
    }

    /// A Mochi t2v job with resolvable weights routes to `VideoRoute::Mochi` — and, critically, a
    /// Mochi job whose weights DON'T resolve must NOT silently become `VideoRoute::Stub` output:
    /// `ensure_video_engine_weights` (the sc-4176 fail-loud gate the Stub arm calls) must error.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_routes_to_the_mochi_engine_and_never_degrades_to_a_fake_video() {
        let root = mochi_root("route", &["q4"], true);
        let settings = Settings {
            data_dir: root.join("unused-data-dir"),
            ..Settings::from_env()
        };
        let req = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        }));
        temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
            assert_eq!(
                resolve_video_route(&req, &settings),
                VideoRoute::Mochi("mochi_1")
            );
            // Adding Mochi must not perturb any pre-existing route.
            let wan = request(json!({
                "projectId": "p", "model": "wan_2_2", "mode": "text_to_video", "prompt": "p"
            }));
            assert!(!matches!(
                resolve_video_route(&wan, &settings),
                VideoRoute::Mochi(_)
            ));
        });

        // Weights absent (no override, empty data dir) ⇒ the route falls to Stub, BUT the Stub arm's
        // fail-loud gate must refuse rather than hand back a procedural fake video.
        let empty = std::env::temp_dir().join(format!("mochi_empty_{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let bare = Settings {
            data_dir: empty.clone(),
            ..Settings::from_env()
        };
        temp_env_var(MOCHI_DIR_ENV, "", || {
            assert_eq!(resolve_video_route(&req, &bare), VideoRoute::Stub);
            let err = ensure_video_engine_weights(&req, &bare)
                .expect_err("an unprovisioned mochi MUST fail loudly, never render a fake video");
            let WorkerError::InvalidPayload(message) = err else {
                panic!("expected an actionable InvalidPayload");
            };
            assert!(
                message.contains("mochi_1") && message.contains("Model Manager"),
                "the error must name the model and tell the user what to do: {message}"
            );
        });
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    /// The on-demand tier fetches are gated on the request's tier toggle — the "fetched on demand if
    /// toggled but not downloaded" AC. Pins that a default (q4) job asks for NEITHER extra tier, so
    /// no job accidentally pulls the ~13 GB q8 / ~20 GB bf16.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_on_demand_tier_fetch_is_gated_on_the_toggle() {
        // Default ⇒ neither on-demand tier is wanted.
        assert!(!mochi_wants_q8(&mochi_request(json!({}))));
        assert!(!mochi_wants_bf16(&mochi_request(json!({}))));
        // q8 toggle ⇒ q8 only.
        assert!(mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 8 }))));
        assert!(!mochi_wants_bf16(&mochi_request(
            json!({ "mlxQuantize": 8 })
        )));
        // bf16 toggle ⇒ bf16 only (the two must be mutually exclusive, or a bf16 job would also drag
        // the q8 tier down).
        assert!(mochi_wants_bf16(&mochi_request(
            json!({ "mlxQuantize": 0 })
        )));
        assert!(!mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 0 }))));
        // An explicit q4 pick pulls nothing extra.
        assert!(!mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 4 }))));
        assert!(!mochi_wants_bf16(&mochi_request(
            json!({ "mlxQuantize": 4 })
        )));
    }

    /// The pinned revision must be a full 40-char commit, not a mutable branch: the repo is a
    /// hard-coded const, so `main` would let an upstream re-push swap a checkpoint we load.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_repo_revision_is_pinned_to_a_commit() {
        assert_eq!(MOCHI_REPO, "SceneWorks/mochi-1-mlx");
        assert_eq!(MOCHI_REVISION.len(), 40, "a full commit sha, not a branch");
        assert!(MOCHI_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
        // The revision B1's manifest sizes were read from.
        assert!(MOCHI_REVISION.starts_with("90a87786"));
    }

    /// The MLX asset record names the tier that was actually LOADED, so provenance can't drift from
    /// the tier the resolver picked when a requested tier isn't installed.
    #[cfg(target_os = "macos")]
    #[test]
    fn mochi_raw_settings_record_the_loaded_tier() {
        let req = mochi_request(json!({ "mlxQuantize": 8 }));
        // Requested q8 but q4 was what resolved ⇒ the record says q4, not q8.
        let raw = mochi_raw_settings(&req, Some(Quant::Q4));
        assert_eq!(raw["mochiTier"], json!("q4"));
        assert_eq!(raw["realModelInference"], json!(true));
        assert_eq!(raw["model"], json!("mochi_1"));
        // No frameCount: the builder records the dispatched KNOBS; the clip's length is stamped from
        // the encoded frames at the funnel (sc-12371, `record_frame_count`).
        assert!(raw.get("frameCount").is_none());
        assert_eq!(
            mochi_raw_settings(&req, Some(Quant::Q8))["mochiTier"],
            json!("q8")
        );
        assert_eq!(mochi_raw_settings(&req, None)["mochiTier"], json!("bf16"));
    }

    /// Set `key` to `value` (empty ⇒ removed) for the duration of `body`, then restore. The Mochi
    /// resolver reads an operator override from the environment, and `std::env::set_var` is
    /// process-global, so the tests that use it are serialized behind one mutex.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn temp_env_var<T>(key: &str, value: &str, body: impl FnOnce() -> T) -> T {
        temp_env_vars(&[(key, value)], body)
    }

    /// The single lock every env-scoped test serializes on. Shared by [`temp_env_var`] and
    /// [`temp_env_vars`] so a one-var and a multi-var test can never interleave their mutations of the
    /// process-global environment.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// [`temp_env_var`] for several vars set together, under ONE acquisition of the shared
    /// [`ENV_LOCK`]. Nesting the single-var helper to get a second var would self-deadlock — the lock
    /// is not reentrant — so a test needing e.g. both the Mochi dir override and the MLX memory cap
    /// must come through here.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn temp_env_vars<T>(vars: &[(&str, &str)], body: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let restore: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var(key).ok();
                if value.is_empty() {
                    std::env::remove_var(key);
                } else {
                    std::env::set_var(key, value);
                }
                ((*key).to_owned(), previous)
            })
            .collect();
        let out = body();
        for (key, previous) in restore {
            match previous {
                Some(prior) => std::env::set_var(&key, prior),
                None => std::env::remove_var(&key),
            }
        }
        out
    }

    /// sc-10418: the resolved video quant maps to the same normalized tier labels the
    /// image lane uses, so the Stats charts group video + image runs on one axis.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn video_quant_label_maps_tiers() {
        assert_eq!(
            video_quant_label(Some(Quant::Q8)),
            (Some("q8".to_owned()), Some(8))
        );
        assert_eq!(
            video_quant_label(Some(Quant::Q4)),
            (Some("q4".to_owned()), Some(4))
        );
        assert_eq!(video_quant_label(None), (Some("bf16".to_owned()), None));
    }

    /// sc-11042 / epic 11037 SC#5 — **the NVFP4 aliasing guard**.
    ///
    /// `Quant::Nvfp4::bits()` returns **4** (E2M1 elements are 4-bit), so the previous
    /// `format!("q{bits}")` implementation stamped an NVFP4 video render as `"q4"` + bits 4 — reporting
    /// the int4-affine tier the user did NOT pick. The compiler cannot catch that class of bug (reading
    /// `.bits()` raises no E0004 on a new variant), so it is pinned here instead.
    ///
    /// The `bits()`-is-4 assertion is deliberate: it documents WHY the label must be matched from the
    /// variant, and fails loudly if a future contract change makes the old bits-derived form look safe
    /// again.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn video_quant_label_never_aliases_nvfp4_to_q4() {
        let (label, bits) = video_quant_label(Some(Quant::Nvfp4));
        // The label names the tier the user actually picked — NOT "q4".
        assert_eq!(label, Some("nvfp4".to_owned()));
        assert_ne!(
            label,
            Some("q4".to_owned()),
            "NVFP4 must never be stamped as the q4 tier (epic 11037 SC#5)"
        );
        // No integer bit-width is honest for NVFP4 (~4.5 EFFECTIVE bits/weight), so `quant_bits` stays
        // empty rather than repeating the q4 aliasing in the numeric column.
        assert_eq!(bits, None);
        // The trap this guards: the element width really is 4, which is exactly why a bits-derived
        // label silently aliased onto q4.
        assert_eq!(Quant::Nvfp4.bits(), 4);
        assert_eq!(Quant::Q4.bits(), 4);
        // And the q4 tier still reports itself truthfully — NVFP4 took nothing away from it.
        assert_eq!(
            video_quant_label(Some(Quant::Q4)),
            (Some("q4".to_owned()), Some(4))
        );
    }

    /// sc-10418 (mirrors the S4 image `image_settings_metrics` tests): the video
    /// settings builder folds the resolved effective settings onto the phase-timing
    /// block WITHOUT clobbering the timings, records one output (sc-10426), and carries
    /// quant / sampler / scheduler / guidance / dims / seed through.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn build_video_metrics_folds_settings_onto_timing() {
        let timing = GenerationMetrics {
            load_ms: Some(3000),
            sample_ms: Some(12000),
            decode_ms: Some(2000),
            ..Default::default()
        };
        let settings = VideoSettingsSnapshot {
            quant: Some(Quant::Q4),
            sampler: Some("euler".to_owned()),
            scheduler: Some("karras".to_owned()),
            scheduler_shift: Some(3.0),
            guidance: Some(5.0),
            width: 832,
            height: 480,
            seed: 1234,
        };
        let metrics = build_video_metrics(
            timing,
            &settings,
            Some("wan2_2_t2v_14b".to_owned()),
            Some(4),
        );
        // Phase timings survive the fold.
        assert_eq!(metrics.load_ms, Some(3000));
        assert_eq!(metrics.sample_ms, Some(12000));
        assert_eq!(metrics.decode_ms, Some(2000));
        // Effective settings folded in.
        assert_eq!(metrics.model.as_deref(), Some("wan2_2_t2v_14b"));
        assert_eq!(metrics.quant_label.as_deref(), Some("q4"));
        assert_eq!(metrics.quant_bits, Some(4));
        assert_eq!(metrics.sampler.as_deref(), Some("euler"));
        assert_eq!(metrics.scheduler.as_deref(), Some("karras"));
        assert_eq!(
            metrics.scheduler_shift.as_ref().and_then(|n| n.as_f64()),
            Some(3.0)
        );
        assert_eq!(metrics.steps, Some(4));
        assert_eq!(metrics.image_count, Some(1));
        assert_eq!(
            metrics.guidance_scale.as_ref().and_then(|n| n.as_f64()),
            Some(5.0)
        );
        assert_eq!(metrics.guidance_method.as_deref(), Some("cfg"));
        assert_eq!(metrics.width, Some(832));
        assert_eq!(metrics.height, Some(480));
        assert_eq!(metrics.seed, Some(1234));
    }

    /// sc-10418: a default-settings video run (no user sampler/scheduler override,
    /// engine-default guidance) still reports non-blank sampler/scheduler so the charts
    /// have a group, but leaves guidance / scheduler-shift `None` — an honest "engine
    /// default, not captured" rather than a fabricated value.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn build_video_metrics_defaults_when_unset() {
        let settings = VideoSettingsSnapshot {
            quant: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance: None,
            width: 1280,
            height: 720,
            seed: 7,
        };
        let metrics = build_video_metrics(GenerationMetrics::default(), &settings, None, None);
        assert_eq!(metrics.quant_label.as_deref(), Some("bf16"));
        assert_eq!(metrics.quant_bits, None);
        assert_eq!(metrics.sampler.as_deref(), Some("default"));
        assert_eq!(metrics.scheduler.as_deref(), Some("default"));
        assert!(metrics.scheduler_shift.is_none());
        assert!(metrics.guidance_scale.is_none());
        assert_eq!(metrics.image_count, Some(1));
        assert_eq!(metrics.width, Some(1280));
        assert_eq!(metrics.height, Some(720));
    }

    /// sc-8879 / sc-9879 (F-077 follow-up): the video SeedVR2 upscale fetches the same fixed
    /// third-party mirror as the image lane, and must pin an exact commit rather than the mutable
    /// `main` branch so an upstream re-push can't silently swap the 3B DiT + VAE weights. Lock the
    /// constant to a real 40-hex lowercase commit id (mirrors `seedvr2_revision_is_pinned_commit_not_main`
    /// in the image-upscale lane).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_video_revision_is_pinned_commit_not_main() {
        assert_ne!(
            SEEDVR2_REVISION, "main",
            "video SeedVR2 must pin a fixed revision"
        );
        assert_eq!(
            SEEDVR2_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            SEEDVR2_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned revision must be lowercase hex"
        );
    }

    /// sc-9879 (F-077 follow-up): the image (`upscale_jobs`) and video (`video_jobs`) SeedVR2 fetches
    /// pull the IDENTICAL files from the IDENTICAL fixed mirror, so their pinned commit must agree.
    /// Two independent consts (a shared source of truth would need a cfg-split hoist across the two
    /// modules) — lock them together here so a bump to one lane without the other fails loudly
    /// (mirrors this PR's InstantID/PuLID shared-repo sha-agreement tests).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_video_revision_matches_image_lane() {
        assert_eq!(
            SEEDVR2_REVISION,
            crate::upscale_jobs::SEEDVR2_REVISION,
            "video and image SeedVR2 fetch the same mirror + files; their pinned commit must match"
        );
        assert_eq!(
            SEEDVR2_REPO,
            crate::upscale_jobs::SEEDVR2_REPO,
            "video and image SeedVR2 must reference the same upstream mirror repo"
        );
    }

    /// sc-11168 / F-007 (completes the sc-9879 rollout on the video lanes): both the MLX
    /// (`ensure_wan_lightning_present`) and candle (`candle_ensure_wan_lightning_present`) A14B Lightning
    /// self-heal fetches pull the FIXED `lightx2v/Wan2.2-Lightning` distill pair from the SHARED
    /// `WAN_LIGHTNING_REVISION` const, so it must pin an exact commit rather than the mutable `main`
    /// branch — an upstream re-push would otherwise silently swap the high/low distill weights we load.
    /// Lock the pin to a real 40-hex lowercase commit id (mirrors the SeedVR2 format test above).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn wan_lightning_revision_is_pinned_commit_not_main() {
        assert_ne!(
            WAN_LIGHTNING_REVISION, "main",
            "Wan Lightning distill pair must pin a fixed revision"
        );
        assert_eq!(
            WAN_LIGHTNING_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            WAN_LIGHTNING_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned revision must be lowercase hex"
        );
    }

    /// sc-9879 (F-077 follow-up): `ensure_ltx_q8_present` pulls `q8/*` from the FIXED SceneWorks LTX-2.3
    /// bundle const (non-overridable here), so it must pin an exact commit rather than the mutable `main`
    /// branch — an upstream re-push would otherwise silently swap the Q8 checkpoint we load. Lock the pin
    /// to a real 40-hex lowercase commit id (mirrors the SeedVR2 format test above).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_bundle_revision_is_pinned_commit_not_main() {
        assert_ne!(
            LTX_BUNDLE_REVISION, "main",
            "LTX q8 bundle must pin a fixed revision"
        );
        assert_eq!(
            LTX_BUNDLE_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            LTX_BUNDLE_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned revision must be lowercase hex"
        );
    }

    // sc-8828 (F-026): `resolve_video_route` is the extracted native (MLX) dispatch decision — the
    // predicate ladder pulled out of `run_video_generate_job`'s 4-tuple match. These lock the branches
    // whose routing depends only on the mode + model id (not on staged weights), so a future family edit
    // that reorders the ladder or mis-routes `replace_person` fails loudly here.
    #[cfg(target_os = "macos")]
    #[test]
    fn video_route_replace_person_dispatches_by_model() {
        let settings = Settings::from_env();

        // `replace_person` on the SCAIL-2 model → the SCAIL-2 replacement backend (sc-5452), carrying
        // the resolved engine id — NOT Wan-VACE.
        let scail2 = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "replace_person",
        }));
        assert_eq!(
            resolve_video_route(&scail2, &settings),
            VideoRoute::ReplacePersonScail2("scail2_14b"),
        );

        // `replace_person` on the dual-expert Wan2.2 VACE-Fun model → its own variant (sc-3459), never
        // the single-expert Wan-VACE checkpoint.
        let vace_fun = request(json!({
            "projectId": "p", "model": "wan_2_2_vace_fun_14b", "mode": "replace_person",
        }));
        assert_eq!(
            resolve_video_route(&vace_fun, &settings),
            VideoRoute::ReplacePersonWanVaceFun,
        );

        // `replace_person` on any other replace-capable model → single-expert Wan-VACE (sc-3521).
        let other = request(json!({
            "projectId": "p", "model": "wan_2_2_ti2v_5b", "mode": "replace_person",
        }));
        assert_eq!(
            resolve_video_route(&other, &settings),
            VideoRoute::ReplacePersonWanVace,
        );

        // An unknown model with no engine + no resolvable weights → the stub (the loud-fail path lives in
        // the execution arm via `ensure_video_engine_weights`).
        let unknown = request(json!({
            "projectId": "p", "model": "definitely_not_a_video_engine", "mode": "text_to_video",
        }));
        assert_eq!(resolve_video_route(&unknown, &settings), VideoRoute::Stub);
    }

    // sc-8828 (F-026): the candle sibling — `resolve_candle_video_route`. Locks the `backend_candle_enabled`
    // gate (off → Stub, so routing is unchanged until parity) + the mode/model dispatch.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn candle_video_route_gates_on_backend_flag_then_mode() {
        let mut settings = Settings::from_env();

        // Candle disabled (default) → always the stub, regardless of model/mode.
        settings.backend_candle_enabled = false;
        let scail2_replace = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "replace_person",
        }));
        assert_eq!(
            resolve_candle_video_route(&scail2_replace, &settings),
            CandleVideoRoute::Stub,
        );

        // Enabled: `replace_person` on the candle SCAIL-2 model → the SCAIL-2 replacement variant;
        // any other replace-capable model → candle Wan-VACE.
        settings.backend_candle_enabled = true;
        assert_eq!(
            resolve_candle_video_route(&scail2_replace, &settings),
            CandleVideoRoute::ReplacePersonScail2(candle_scail2_engine_id("scail2_14b").unwrap()),
        );
        let extend = request(json!({
            "projectId": "p", "model": "wan_2_2_ti2v_5b", "mode": "extend_clip",
        }));
        assert_eq!(
            resolve_candle_video_route(&extend, &settings),
            CandleVideoRoute::WanVaceExtendBridge,
        );
    }

    /// sc-10997 (epic 6562): the candle Bernini VIDEO lane routes t2v + every editing/reference/
    /// multi-source mode to `CandleVideoRoute::Bernini` (a DISTINCT engine, NOT the generic wan/ltx
    /// txt2video arm), and only when the backend-candle flag is on. This per-mode dispatch is the
    /// story's validation (GPU-val is gated on the `SceneWorks/bernini` weights, sc-11003).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn candle_video_route_bernini_every_mode() {
        let mut settings = Settings::from_env();
        settings.backend_candle_enabled = true;
        let engine = candle_bernini_engine_id("bernini").expect("bernini engine id");
        for mode in [
            "text_to_video",
            "video_to_video",
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ] {
            let req = request(json!({ "projectId": "p", "model": "bernini", "mode": mode }));
            assert_eq!(
                resolve_candle_video_route(&req, &settings),
                CandleVideoRoute::Bernini(engine),
                "bernini {mode} must route to CandleVideoRoute::Bernini",
            );
        }
        // Backend-candle off → Stub (routing unchanged until the flag is set), even for bernini.
        settings.backend_candle_enabled = false;
        let off = request(json!({
            "projectId": "p", "model": "bernini", "mode": "text_to_video",
        }));
        assert_eq!(
            resolve_candle_video_route(&off, &settings),
            CandleVideoRoute::Stub,
        );
    }

    /// sc-10997: the candle Bernini engine id + the SceneWorks-mode → engine `video_mode` task mapping,
    /// the byte-identical twin of the MLX `bernini_engine_id` / `bernini_engine_video_mode`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn candle_bernini_engine_id_and_video_mode_mapping() {
        assert_eq!(candle_bernini_engine_id("bernini"), Some("bernini"));
        // The still `bernini_image` id + every other video family keep their own routing.
        assert_eq!(candle_bernini_engine_id("bernini_image"), None);
        assert_eq!(candle_bernini_engine_id("wan_2_2"), None);
        assert_eq!(candle_bernini_engine_id("scail2_14b"), None);
        assert_eq!(candle_bernini_engine_id(""), None);

        assert_eq!(candle_bernini_engine_video_mode("text_to_video"), "t2v");
        assert_eq!(candle_bernini_engine_video_mode("video_to_video"), "v2v");
        assert_eq!(
            candle_bernini_engine_video_mode("reference_to_video"),
            "r2v"
        );
        assert_eq!(
            candle_bernini_engine_video_mode("reference_video_to_video"),
            "rv2v"
        );
        assert_eq!(
            candle_bernini_engine_video_mode("multi_video_to_video"),
            "mv2v"
        );
        assert_eq!(candle_bernini_engine_video_mode("ads2v"), "ads2v");
        // Unknown / image_to_video ⇒ plain t2v (the renderer is text-conditioned Wan2.2-T2V).
        assert_eq!(candle_bernini_engine_video_mode("image_to_video"), "t2v");
        assert_eq!(candle_bernini_engine_video_mode(""), "t2v");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stall_timeout_override_parses_or_falls_back() {
        // A valid positive override wins.
        assert_eq!(
            parse_stall_timeout(Some("120".to_owned())),
            Duration::from_secs(120)
        );
        assert_eq!(
            parse_stall_timeout(Some("  90 ".to_owned())),
            Duration::from_secs(90)
        );
        // Unset, blank, non-numeric, or zero all fall back to the default.
        assert_eq!(parse_stall_timeout(None), VIDEO_STALL_TIMEOUT);
        assert_eq!(
            parse_stall_timeout(Some(String::new())),
            VIDEO_STALL_TIMEOUT
        );
        assert_eq!(
            parse_stall_timeout(Some("nope".to_owned())),
            VIDEO_STALL_TIMEOUT
        );
        assert_eq!(
            parse_stall_timeout(Some("0".to_owned())),
            VIDEO_STALL_TIMEOUT
        );
    }

    #[test]
    fn plan_builds_nested_media_path() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox runs"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        assert!(plan
            .media_rel
            .starts_with(&format!("assets/videos/{}/", plan.genset_id)));
        assert!(plan.media_rel.ends_with(".mp4"));
        // The model id is slugified into the filename (F-003 / sc-11159): `ltx_2_3` -> `ltx-2-3`.
        assert!(plan.media_rel.contains("_ltx-2-3_"));
        assert!(plan.asset_id.starts_with("asset_"));
        assert_eq!(plan.family, "ltx-video");
        assert_eq!(
            plan.media_path,
            Path::new("/tmp/project").join(&plan.media_rel)
        );
    }

    /// F-003 / sc-11159: a path-traversal / absolute model id in the payload must NOT let the
    /// rendered `.mp4` escape the project dir. The model id becomes a filename component in
    /// `media_rel`, so a `../`, `..\`, or `/abs` id would otherwise place the media path (and
    /// its later `create_dir_all` + write) outside `project_path`. Assert confinement per vector.
    #[test]
    fn plan_confines_malicious_model_id() {
        let project = Path::new("/tmp/project");
        for evil in [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\evil",
            "/etc/cron.d/pwn",
            "sub/dir/model",
        ] {
            let request = request(json!({
                "projectId": "p", "model": evil, "prompt": "A red fox runs"
            }));
            let plan = VideoPlan::new(&request, project);
            assert!(
                !plan.media_rel.contains(".."),
                "media_rel must not contain `..` for model id {evil:?}: {}",
                plan.media_rel
            );
            assert!(
                plan.media_rel
                    .starts_with(&format!("assets/videos/{}/", plan.genset_id)),
                "media_rel escaped the videos dir for model id {evil:?}: {}",
                plan.media_rel
            );
            assert!(
                plan.media_path.starts_with(project),
                "media_path {:?} escaped project dir for model id {evil:?}",
                plan.media_path
            );
        }
    }

    #[test]
    fn family_prefers_manifest_then_infers_from_model() {
        let manifest = request(json!({
            "projectId": "p", "model": "ltx_2_3",
            "modelManifestEntry": { "family": "ltx-custom" }
        }));
        assert_eq!(resolve_family(&manifest), "ltx-custom");
        let wan = request(json!({ "projectId": "p", "model": "wan_2_2_t2v_14b" }));
        assert_eq!(resolve_family(&wan), "wan-video");
        let other = request(json!({ "projectId": "p", "model": "mystery" }));
        assert_eq!(resolve_family(&other), "video");
    }

    #[test]
    fn resolve_seed_prefers_explicit_then_hashes_prompt() {
        let explicit = request(json!({ "projectId": "p", "seed": 123 }));
        assert_eq!(resolve_video_seed(&explicit), 123);
        // No seed → deterministic from the prompt (re-run reproduces).
        let a = request(json!({ "projectId": "p", "prompt": "sunset" }));
        let b = request(json!({ "projectId": "p", "prompt": "sunset" }));
        assert_eq!(resolve_video_seed(&a), resolve_video_seed(&b));
        let c = request(json!({ "projectId": "p", "prompt": "sunrise" }));
        assert_ne!(resolve_video_seed(&a), resolve_video_seed(&c));
    }

    #[test]
    fn stub_video_frames_have_correct_size_and_audio_split() {
        // LTX → audio present; the frame buffers are exactly width*height*3.
        let ltx = request(json!({
            "projectId": "p", "model": "ltx_2_3", "width": 256, "height": 256,
            "duration": 1.0, "fps": 9
        }));
        let decoded = generate_stub_video(&ltx, 7);
        assert_eq!(decoded.frames.len(), ltx.frame_count() as usize);
        assert_eq!(decoded.fps, 9);
        for frame in &decoded.frames {
            assert_eq!(frame.pixels.len(), 256 * 256 * 3);
        }
        let audio = decoded.audio.expect("LTX stub emits audio");
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());

        // Wan → no audio track (mirrors the engine).
        let wan = request(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 1.0, "fps": 16
        }));
        assert!(generate_stub_video(&wan, 7).audio.is_none());
    }

    #[test]
    fn stub_frames_differ_across_time() {
        // The sweeping band makes frame 0 and a later frame differ (real motion).
        let request = request(json!({
            "projectId": "p", "model": "wan_2_2", "width": 256, "height": 64,
            "duration": 1.0, "fps": 16
        }));
        let decoded = generate_stub_video(&request, 3);
        assert!(decoded.frames.len() >= 2);
        assert_ne!(
            decoded.frames[0].pixels,
            decoded.frames[decoded.frames.len() - 1].pixels
        );
    }

    #[test]
    fn wav_header_is_canonical_and_peak_normalized() {
        let audio = AudioTrack {
            samples: vec![0.0, 0.5, -0.25, 0.5],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        // 4 mono 16-bit samples → 8 bytes of PCM, 44-byte header.
        assert_eq!(bytes.len(), 44 + 8);
        // Peak (0.5) maps to i16::MAX; the matching trough (-0.25) is half-scale negative.
        let first = i16::from_le_bytes([bytes[44], bytes[45]]);
        let peak = i16::from_le_bytes([bytes[46], bytes[47]]);
        let trough = i16::from_le_bytes([bytes[48], bytes[49]]);
        assert_eq!(first, 0);
        assert_eq!(peak, i16::MAX);
        assert_eq!(trough, -(i16::MAX / 2) - 1); // -0.25/0.5 * 32767, rounded
    }

    #[test]
    fn silent_audio_does_not_divide_by_zero() {
        let audio = AudioTrack {
            samples: vec![0.0; 16],
            sample_rate: 48_000,
            channels: 1,
        };
        let dir = std::env::temp_dir().join(format!("sw_wav_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("silent.wav");
        write_wav_pcm16(&audio, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes[44..].iter().all(|&b| b == 0));
    }

    #[test]
    fn asset_fact_and_streaming_result_shape() {
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox",
            "duration": 4.0, "fps": 24, "width": 768, "height": 512,
            "sourceAssetId": "asset_src", "personTrackId": "track_1"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        // Drive the real stub arm and measure what it produced — the funnel's exact sequence.
        let decoded = generate_stub_video(&request, 42);
        let clip = EncodedClip::measure(&decoded);
        let fact = video_asset_fact(
            &plan,
            42,
            "procedural_video",
            stub_raw_settings(&request),
            None,
            clip,
        );
        // Exhaustive, mirroring the image lane's key sweep (`image_jobs/tests.rs`). The recipe is
        // the ONLY record of what a user asked for, so a field silently dropped here is a field
        // the replay path (sc-12324) cannot reproduce — which is exactly how `fitMode` and the
        // multi-source ids went missing until sc-12345. Add a key here when you add one to the
        // fact; a spot-check would let the next one through.
        for key in [
            "type",
            "assetId",
            "mediaPath",
            "mimeType",
            "width",
            "height",
            "duration",
            "fps",
            "quality",
            "family",
            "seed",
            "displayName",
            "createdAt",
            "mode",
            "model",
            "adapter",
            "prompt",
            "negativePrompt",
            "loras",
            "rawAdapterSettings",
            "sourceAssetId",
            "lastFrameAssetId",
            "sourceClipAssetId",
            "bridgeRightClipAssetId",
            "fitMode",
            "sourceClipAssetIds",
            "referenceAssetIds",
            "referenceClipAssetId",
            "characterId",
            "characterLookId",
            "personTrackId",
            "replacementMode",
            "timelineContext",
        ] {
            assert!(fact.get(key).is_some(), "fact missing key {key}");
        }
        assert_eq!(fact["type"], json!("video"));
        assert_eq!(fact["mimeType"], json!("video/mp4"));
        assert_eq!(fact["mediaPath"], json!(plan.media_rel));
        assert_eq!(fact["adapter"], json!("procedural_video"));
        assert_eq!(fact["seed"], json!(42));
        // sc-12371 — THE SPLIT. LTX snaps 4.0 s x 24 fps (96 raw) onto its 8k+1 lattice, so the clip
        // is NOT 4.0 s. The fact must carry BOTH truths, because they feed different consumers:
        //
        //   `duration`/`fps`              -> `recipe.normalizedSettings` -> "re-run this generation"
        //   `encodedDuration`/`encodedFps` -> `asset.file`               -> what the mp4 really is
        //
        // Collapsing them either way is a bug: measured-into-normalizedSettings replays a 4 s ask as
        // 4.04 s (off the model's `limits.durations` menu, which sc-12347 now enforces), and
        // requested-into-file is the sc-12371 lie this story was filed for.
        let rendered = ltx_frame_count(96) as usize;
        assert_eq!(decoded.frames.len(), rendered);
        assert_ne!(
            rendered, 96,
            "the probe must discriminate: LTX has to actually snap 96 here, or requested and \
             measured agree and this pins nothing"
        );
        // The REPLAY knobs: exactly what the user picked, untouched.
        assert_eq!(fact["duration"], json!(4.0));
        assert_eq!(fact["fps"], json!(24));
        // The FILE facts: measured off the clip.
        assert_eq!(fact["encodedFrameCount"], json!(rendered));
        assert_eq!(fact["encodedFps"], json!(24));
        assert_eq!(fact["rawAdapterSettings"]["frameCount"], json!(rendered));
        assert!(
            (fact["encodedDuration"].as_f64().expect("encodedDuration") - rendered as f64 / 24.0)
                .abs()
                < 1e-9,
            "encodedDuration must be the encoded frames over the encoded fps"
        );
        assert_ne!(
            fact["encodedDuration"],
            json!(4.0),
            "the REQUESTED 4.0 s is not the file's duration — if this ever matches, a \
             `request.duration` regression in the file block would pass unnoticed"
        );
        assert_eq!(fact["sourceAssetId"], json!("asset_src"));
        assert_eq!(fact["personTrackId"], json!("track_1"));
        assert_eq!(fact["displayName"], json!("A red fox"));
        assert_eq!(fact["rawAdapterSettings"]["stub"], json!(true));

        let result = streaming_result(&plan, &fact, "procedural_video");
        assert_eq!(result["expectedCount"], json!(1));
        assert_eq!(result["adapter"], json!("procedural_video"));
        assert_eq!(result["assetWrites"].as_array().unwrap().len(), 1);
        assert_eq!(result["generationSet"]["count"], json!(1));
    }

    /// The fit and the list-valued source ids reach the fact (sc-12345). These arrive as
    /// TOP-LEVEL payload fields, so the `advanced.clone()` every real `*_raw_settings` builder
    /// starts with does not carry them — `video_asset_fact` is their only path onto the recipe.
    /// ads2v is the densest mode: source clip + reference clip + subject references at once.
    #[test]
    fn asset_fact_records_fit_and_multi_source_ids() {
        let ads2v = request(json!({
            "projectId": "p", "model": "bernini_2", "mode": "ads2v",
            "prompt": "the hero drives past",
            "sourceClipAssetId": "clip_main",
            "referenceClipAssetId": "clip_ref",
            "referenceAssetIds": ["ref_1", "ref_2"],
            "fitMode": "pad",
        }));
        let plan = VideoPlan::new(&ads2v, Path::new("/tmp/project"));
        // The clip these ids ride on is irrelevant to them, but it is not optional: `video_asset_fact`
        // requires the MEASURED clip so no caller can record a predicted length (sc-12371).
        let clip = EncodedClip {
            frames: 81,
            fps: 16,
        };
        let fact = video_asset_fact(&plan, 5, "mlx_bernini", json!({}), None, clip);
        assert_eq!(fact["referenceClipAssetId"], json!("clip_ref"));
        assert_eq!(fact["referenceAssetIds"], json!(["ref_1", "ref_2"]));
        assert_eq!(fact["fitMode"], json!("pad"));

        // mv2v carries the clip array instead; the other list stays empty rather than absent.
        let mv2v = request(json!({
            "projectId": "p", "model": "bernini_2", "mode": "multi_video_to_video",
            "prompt": "stitch them", "sourceClipAssetIds": ["clip_a", "clip_b"],
        }));
        let mv2v_plan = VideoPlan::new(&mv2v, Path::new("/tmp/project"));
        let mv2v_fact = video_asset_fact(&mv2v_plan, 5, "mlx_bernini", json!({}), None, clip);
        assert_eq!(mv2v_fact["sourceClipAssetIds"], json!(["clip_a", "clip_b"]));
        assert_eq!(mv2v_fact["referenceAssetIds"], json!([]));
    }

    /// A replace_person asset fact carries the `replacementStatus` object the API folds into
    /// the video sidecar (sc-3521); a non-replace fact omits it.
    #[test]
    fn asset_fact_embeds_replacement_status_when_present() {
        let request = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "replace_person",
            "prompt": "swap the hero", "personTrackId": "track_9"
        }));
        let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
        let status = json!({ "replacementActive": true, "maskMode": "segmentation" });
        let clip = EncodedClip {
            frames: 81,
            fps: 16,
        };
        let fact = video_asset_fact(&plan, 7, "mlx_wan_vace", json!({}), Some(status), clip);
        assert_eq!(fact["replacementStatus"]["replacementActive"], json!(true));
        assert_eq!(fact["replacementStatus"]["maskMode"], json!("segmentation"));
        // Without a status the key is absent (the non-replace paths).
        let bare = video_asset_fact(&plan, 7, "mlx_wan", json!({}), None, clip);
        assert!(bare.get("replacementStatus").is_none());
    }

    /// SceneWorks `replacementMode` strings → engine `ReplacementMode` (default FaceOnly).
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_mode_maps_the_three_granularities() {
        assert_eq!(
            replacement_mode_from("face_only"),
            ReplacementMode::FaceOnly
        );
        assert_eq!(
            replacement_mode_from("full_person_keep_outfit"),
            ReplacementMode::FullPersonKeepOutfit
        );
        assert_eq!(
            replacement_mode_from("full_person_replace_outfit"),
            ReplacementMode::FullPersonReplaceOutfit
        );
        assert_eq!(replacement_mode_from("nonsense"), ReplacementMode::FaceOnly);
    }

    /// SCAIL-2 maps the SceneWorks video mode to the engine `video_mode` task: cross-identity
    /// `replace_person` → "replacement" (engine flips `replace_flag`), everything else (standalone
    /// `animate_character`) → "animation" (sc-5448 / sc-5452).
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_mode_maps_replacement_vs_animation() {
        assert_eq!(scail2_engine_video_mode("replace_person"), "replacement");
        assert_eq!(scail2_engine_video_mode("animate_character"), "animation");
        assert_eq!(scail2_engine_video_mode("text_to_video"), "animation");
    }

    /// The replacement status records the actual engine adapter, so a SCAIL-2-backed person-replace
    /// (sc-5452) reports `mlx_scail2` (not the Wan-VACE default).
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_status_records_scail2_adapter() {
        let track = json!({ "id": "trk_1", "status": { "maskState": "active" } });
        let status =
            replacement_status_value(&track, "trk_1", "segmentation", 1.0, 1, 81, SCAIL2_ADAPTER);
        assert_eq!(status["replacementAdapter"], json!("mlx_scail2"));
        assert_eq!(status["replacementActive"], json!(true));
        assert_eq!(status["controlFrameCount"], json!(81));
    }

    /// The Wan-VACE conditioning is one ControlClip (frames + per-frame mask) followed by one
    /// Reference per character image; mismatched frame/mask counts fail clearly.
    #[cfg(target_os = "macos")]
    #[test]
    fn vace_conditioning_builds_control_clip_plus_references() {
        let frame = |v: u8| Image {
            width: 2,
            height: 2,
            pixels: vec![v; 12],
        };
        let mask = || image::RgbImage::from_pixel(2, 2, image::Rgb([255, 255, 255]));
        let conditioning = build_vace_conditioning(
            vec![frame(10), frame(20)],
            vec![mask(), mask()],
            vec![frame(30)],
            0.75,
            ReplacementMode::FullPersonKeepOutfit,
        )
        .expect("conditioning builds");
        assert_eq!(conditioning.len(), 2); // 1 ControlClip + 1 Reference
        match &conditioning[0] {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(mask.len(), 2);
                assert_eq!(*masking_strength, 0.75);
                assert_eq!(*start_frame, 0);
                assert_eq!(*mode, ReplacementMode::FullPersonKeepOutfit);
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
        assert!(matches!(conditioning[1], Conditioning::Reference { .. }));
        // A frame/mask count mismatch is rejected.
        assert!(build_vace_conditioning(
            vec![frame(1)],
            vec![mask(), mask()],
            Vec::new(),
            1.0,
            ReplacementMode::FaceOnly,
        )
        .is_err());
    }

    /// sc-3812 extend: the ControlClip pins the real tail frames at the front (mask black = keep)
    /// and fills the rest of the budget with a neutral-gray generated span (mask white), no refs.
    #[cfg(target_os = "macos")]
    #[test]
    fn extend_vace_conditioning_pins_tail_and_generates_rest() {
        let anchor = |v: u8| Image {
            width: 2,
            height: 2,
            pixels: vec![v; 12],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
            "prompt": "keep walking", "sourceClipAssetId": "clip_a"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            2,
            2,
            5,
            vec![anchor(11), anchor(22)],
            None,
        )
        .expect("extend conditioning builds");
        assert_eq!(conditioning.len(), 1); // ControlClip only, no Reference
        match &conditioning[0] {
            Conditioning::ControlClip { frames, mask, .. } => {
                assert_eq!(frames.len(), 5);
                assert_eq!(mask.len(), 5);
                // First two are the real tail frames, kept (black mask).
                assert_eq!(frames[0].pixels[0], 11);
                assert_eq!(frames[1].pixels[0], 22);
                assert_eq!(mask[0].pixels[0], 0);
                assert_eq!(mask[1].pixels[0], 0);
                // The rest is the neutral-gray generated span (white mask).
                assert_eq!(frames[2].pixels[0], 128);
                assert_eq!(frames[4].pixels[0], 128);
                assert_eq!(mask[2].pixels[0], 255);
                assert_eq!(mask[4].pixels[0], 255);
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
    }

    /// sc-3812 bridge: both clips' boundary anchors are kept at the two ends; the gap between them
    /// is the generated span. A missing right clip / over-budget anchors fail clearly.
    #[cfg(target_os = "macos")]
    #[test]
    fn bridge_vace_conditioning_keeps_both_ends_generates_gap() {
        let anchor = |v: u8| Image {
            width: 1,
            height: 1,
            pixels: vec![v; 3],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "video_bridge",
            "prompt": "connect", "sourceClipAssetId": "left", "bridgeRightClipAssetId": "right"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            1,
            1,
            5,
            vec![anchor(10)],
            Some(vec![anchor(90)]),
        )
        .expect("bridge conditioning builds");
        match &conditioning[0] {
            Conditioning::ControlClip { frames, mask, .. } => {
                assert_eq!(frames.len(), 5);
                // Left end kept, gap generated, right end kept.
                assert_eq!((frames[0].pixels[0], mask[0].pixels[0]), (10, 0));
                assert_eq!((frames[1].pixels[0], mask[1].pixels[0]), (128, 255));
                assert_eq!((frames[3].pixels[0], mask[3].pixels[0]), (128, 255));
                assert_eq!((frames[4].pixels[0], mask[4].pixels[0]), (90, 0));
            }
            other => panic!("expected ControlClip, got {other:?}"),
        }
        // video_bridge without a right clip is rejected.
        assert!(
            build_extend_bridge_vace_conditioning(&req, 1, 1, 5, vec![anchor(10)], None).is_err()
        );
        // Anchors that leave no gap are rejected.
        assert!(build_extend_bridge_vace_conditioning(
            &req,
            1,
            1,
            5,
            vec![anchor(1), anchor(2), anchor(3)],
            Some(vec![anchor(4), anchor(5)]),
        )
        .is_err());
    }

    /// sc-3812 motion anchor: defaults to ~⅓ of the budget (halved per side for bridge), honors an
    /// explicit `motionAnchorFrames`, and always clamps so ≥5 frames stay generatable.
    #[cfg(target_os = "macos")]
    #[test]
    fn extend_anchor_frames_defaults_and_clamps() {
        let extend = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x"
        }));
        assert_eq!(extend_anchor_frames(&extend, 81), 27); // 81/3
        let bridge = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "video_bridge", "prompt": "x"
        }));
        assert_eq!(extend_anchor_frames(&bridge, 81), 13); // (81/3)/2

        let explicit = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
            "advanced": { "motionAnchorFrames": 4 }
        }));
        assert_eq!(extend_anchor_frames(&explicit, 81), 4);

        // Over-budget request clamps so 5 frames remain to generate (81 - 5 = 76).
        let greedy = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
            "advanced": { "motionAnchorFrames": 999 }
        }));
        assert_eq!(extend_anchor_frames(&greedy, 81), 76);
        // Minimum-length clip still yields a usable anchor.
        assert_eq!(extend_anchor_frames(&extend, 5), 1);
    }

    /// `replacement_status_value` reports the honest mask/track provenance the sidecar folds in.
    #[cfg(target_os = "macos")]
    #[test]
    fn replacement_status_reads_track_and_counts() {
        let track = json!({
            "id": "track_42",
            "status": { "maskState": "active", "personTrackingActive": true },
            "corrections": [ { "frameIndex": 0 }, { "frameIndex": 3 } ]
        });
        // 0.5 is exactly representable as f32 so the JSON widen to f64 is exact.
        let status = replacement_status_value(
            &track,
            "ignored",
            "segmentation",
            0.5,
            2,
            81,
            WAN_VACE_ADAPTER,
        );
        assert_eq!(status["personDetectionActive"], json!(true));
        assert_eq!(status["personTrackingActive"], json!(true));
        assert_eq!(status["replacementActive"], json!(true));
        assert_eq!(status["replacementAdapter"], json!("mlx_wan_vace"));
        assert_eq!(status["maskMode"], json!("segmentation"));
        assert_eq!(status["maskState"], json!("active"));
        assert_eq!(status["maskingStrength"], json!(0.5));
        assert_eq!(status["personTrackId"], json!("track_42"));
        assert_eq!(status["characterReferenceCount"], json!(2));
        assert_eq!(status["controlFrameCount"], json!(81));
        assert_eq!(status["usedCorrections"], json!(true));
        assert_eq!(status["correctionCount"], json!(2));
    }

    /// An assembled Wan-VACE snapshot dir if one is present (env override or the app-managed
    /// default), else `None` so the real-weight smoke skips.
    #[cfg(target_os = "macos")]
    fn wan_vace_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
            let path = PathBuf::from(dir.trim());
            if wan_vace_dir_is_complete(&path) {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join("Library/Application Support/SceneWorks/data/models/mlx/wan_vace");
        wan_vace_dir_is_complete(&path).then_some(path)
    }

    /// Real in-process Wan-VACE replace_person through the engine: load the assembled snapshot
    /// and denoise a tiny 5-frame clip from a synthetic control clip (gray frames + a centered
    /// box mask) + one reference, asserting frames come back RGB8-sized with streamed progress.
    /// `#[ignore]` — the weights live outside CI; run manually on a Mac where the snapshot is
    /// assembled (the real-Mac GPU parity gate, sc-3521).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
    #[test]
    fn wan_vace_real_weights() {
        let Some(model_dir) = wan_vace_dir() else {
            eprintln!("skipping wan_vace_real_weights: no assembled wan_vace snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let gray = || Image {
            width: w,
            height: h,
            pixels: vec![118u8; (w * h * 3) as usize],
        };
        let frames: Vec<Image> = (0..5).map(|_| gray()).collect();
        let masks: Vec<image::RgbImage> = (0..5)
            .map(|_| {
                crate::person_replace::box_mask(
                    Some(&json!({ "x": 0.3, "y": 0.2, "width": 0.4, "height": 0.6 })),
                    w,
                    h,
                )
            })
            .collect();
        let conditioning =
            build_vace_conditioning(frames, masks, vec![gray()], 1.0, ReplacementMode::FaceOnly)
                .expect("conditioning builds");
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan_vace",
            model_dir,
            conditioning,
            prompt: "a person walking, cinematic".to_owned(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            control_scale: Some(1.0),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("VACE generation");
        assert!(decoded.fps >= 1);
        assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// A Bernini MLX snapshot dir if present: env override (`SCENEWORKS_MLX_BERNINI_DIR`), then the
    /// bring-up staging / cache dirs, then the app-managed default. `None` ⇒ the smoke skips.
    #[cfg(target_os = "macos")]
    fn bernini_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_BERNINI_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        for rel in [
            ".cache/mlx-gen-models/bernini-mlx-upload",
            ".cache/mlx-gen-models/bernini_full_mlx_bf16",
            "Library/Application Support/SceneWorks/data/models/mlx/bernini",
        ] {
            let path = PathBuf::from(&home).join(rel);
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        None
    }

    /// Test-only override for the quant tier a Bernini smoke loads (sc-4709): `SCENEWORKS_BERNINI_
    /// SMOKE_QUANT` = `8` → Q8, `0` → bf16 (no quant), anything else / unset → Q4 (the committed
    /// default). Lets the manual smokes profile peak memory at either tier without changing the
    /// product default (e.g. capturing the Q8 video peak this story tracks).
    #[cfg(target_os = "macos")]
    fn bernini_smoke_quant() -> Option<Quant> {
        match std::env::var("SCENEWORKS_BERNINI_SMOKE_QUANT")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
        {
            Some(bits) if bits <= 0 => None,
            Some(bits) if bits > 4 => Some(Quant::Q8),
            _ => Some(Quant::Q4),
        }
    }

    /// A test-only `usize`/`u32` env override, falling back to `default` when unset/unparseable.
    #[cfg(target_os = "macos")]
    fn bernini_smoke_u32(key: &str, default: u32) -> u32 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(default)
    }

    /// Real in-process Bernini text-to-video through the WORKER registry path (epic 4699 / sc-4707):
    /// `crate::inference_runtime::load("bernini")` (proves runtime-macos includes Bernini — the
    /// "no generator registered" trap) → Q4 → a tiny t2v clip, asserting RGB8 frames
    /// stream back with denoise progress. Also confirms the lean published `SceneWorks/bernini-mlx`
    /// snapshot loads. `#[ignore]` — weights live outside CI; run on a Mac with the snapshot present.
    /// The quant tier (`SCENEWORKS_BERNINI_SMOKE_QUANT`) and dims (`..._W`/`_H`/`_FRAMES`/`_STEPS`)
    /// are env-overridable for memory profiling (sc-4709 Q8 peak); defaults reproduce the Q4 run.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
    #[test]
    fn bernini_t2v_real_weights() {
        let Some(model_dir) = bernini_dir() else {
            eprintln!("skipping bernini_t2v_real_weights: no Bernini MLX snapshot found");
            return;
        };
        let w = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_W", 832);
        let h = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_H", 480);
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "bernini",
            model_dir,
            quant: bernini_smoke_quant(),
            prompt: "a golden retriever puppy running across a sunlit meadow, cinematic".to_owned(),
            width: w,
            height: h,
            frames: bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_FRAMES", 17),
            fps: 16,
            steps: Some(bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_STEPS", 20)),
            seed: 7,
            video_mode: Some("t2v".to_owned()),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("bernini t2v generation");
        assert!(decoded.fps >= 1);
        assert!(steps > 0, "denoise progress streamed");
        assert!(!decoded.frames.is_empty(), "frames returned");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// The SceneWorks mode → engine guidance-task mapping (sc-4703). Pure; runs in CI on Mac.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_engine_video_mode_maps_each_sceneworks_mode() {
        assert_eq!(bernini_engine_video_mode("text_to_video"), "t2v");
        assert_eq!(bernini_engine_video_mode("video_to_video"), "v2v");
        assert_eq!(bernini_engine_video_mode("reference_to_video"), "r2v");
        assert_eq!(
            bernini_engine_video_mode("reference_video_to_video"),
            "rv2v"
        );
        // Multi-source modes (sc-5425).
        assert_eq!(bernini_engine_video_mode("multi_video_to_video"), "mv2v");
        assert_eq!(bernini_engine_video_mode("ads2v"), "ads2v");
        // Unknown / unset falls back to plain text-to-video.
        assert_eq!(bernini_engine_video_mode("image_to_video"), "t2v");
        assert_eq!(bernini_engine_video_mode(""), "t2v");
    }

    /// Only the `bernini` catalog id routes to the Bernini engine (sc-4707). Other video
    /// families (Wan/LTX/SVD/image-typed ids) are `None` so they keep their own routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_engine_id_maps_only_the_bernini_family() {
        assert_eq!(bernini_engine_id("bernini"), Some("bernini"));
        assert_eq!(bernini_engine_id("wan_2_2"), None);
        assert_eq!(bernini_engine_id("wan_2_2_t2v_14b"), None);
        assert_eq!(bernini_engine_id("ltx_2_3"), None);
        assert_eq!(bernini_engine_id("z_image_turbo"), None);
        assert_eq!(bernini_engine_id(""), None);
    }

    /// Bernini load quantization (sc-4709): Q4 is the default (the validated 64 GB-fitting tier),
    /// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), and the control parses from a JSON
    /// number or a string. The snapshot is ~93 GB at bf16 so a missing control NEVER means bf16.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_resolve_quant_defaults_q4_and_honors_override() {
        let quant = |advanced: Value| {
            resolve_bernini_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
        };
        // No control → Q4 default (never bf16).
        assert!(matches!(quant(json!({})), Some(Quant::Q4)));
        // Explicit tiers (number).
        assert!(matches!(
            quant(json!({ "mlxQuantize": 4 })),
            Some(Quant::Q4)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 8 })),
            Some(Quant::Q8)
        ));
        // `<= 0` opts into bf16 (no quantization) for power users with ample RAM.
        assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
        assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
        // String forms parse the same (the advanced map can carry stringly-typed values).
        assert!(matches!(
            quant(json!({ "mlxQuantize": "8" })),
            Some(Quant::Q8)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": " 4 " })),
            Some(Quant::Q4)
        ));
        assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
        // A bits value between 4 and 8 rounds to the nearest supported tier (> 4 ⇒ Q8).
        assert!(matches!(
            quant(json!({ "mlxQuantize": 6 })),
            Some(Quant::Q8)
        ));
    }

    /// Raw-settings lineage captured on a real Bernini asset (sc-4709): the real-inference marker,
    /// the catalog model id, the produced frame count / fps, the resolved engine guidance task
    /// (`berniniTask`), and a pass-through of the user's advanced controls.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_raw_settings_capture_lineage_and_task() {
        let req = request(json!({
            "projectId": "p",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "duration": 5,
            "fps": 16,
            "advanced": { "mlxQuantize": 8, "userKnob": "keep-me" }
        }));
        let raw = bernini_raw_settings(&req);
        let raw = raw.as_object().expect("raw settings is an object");
        assert_eq!(raw["realModelInference"], json!(true));
        assert_eq!(raw["model"], json!("bernini"));
        assert_eq!(raw["fps"], json!(req.fps));
        // No frameCount: this builder is exactly where the sc-12371 bug lived (it recorded
        // `request.frame_count()` = the raw 150 while `generate_bernini` rendered 149). The clip's
        // length is now stamped from the encoded frames at the funnel — see
        // `no_raw_settings_builder_records_its_own_frame_count`.
        assert!(raw.get("frameCount").is_none());
        // The SceneWorks mode resolved to its engine guidance task for observability/lineage.
        assert_eq!(raw["berniniTask"], json!("rv2v"));
        // The user's advanced controls survive verbatim (provenance).
        assert_eq!(raw["userKnob"], json!("keep-me"));
        assert_eq!(raw["mlxQuantize"], json!(8));
    }

    /// Bernini conditioning resolution enforces each editing/reference mode's required media
    /// (sc-4703 / sc-4709), failing loudly BEFORE any IO when it is missing — defense in depth
    /// behind the API-side validation. `text_to_video` needs none and resolves to empty
    /// conditioning. The guards fire before touching the api / job / disk, so a minimal snapshot
    /// suffices (mirrors `video_clip_conditioning_requires_ic_lora`).
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn bernini_conditioning_enforces_required_media() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-bernini-1",
            "type": "video_generate",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-14T00:00:00Z",
            "updatedAt": "2026-06-14T00:00:00Z"
        }))
        .expect("job snapshot");
        let resolve = |mode: &str, extra: Value| {
            let mut payload = json!({ "projectId": "p", "model": "bernini", "prompt": "go" });
            payload
                .as_object_mut()
                .unwrap()
                .insert("mode".to_owned(), json!(mode));
            for (k, v) in extra.as_object().cloned().unwrap_or_default() {
                payload.as_object_mut().unwrap().insert(k, v);
            }
            let req = request(payload);
            let api = &api;
            let settings = &settings;
            let job = &job;
            async move {
                resolve_bernini_conditioning(api, settings, job, &req, Path::new("/tmp/p")).await
            }
        };

        // text_to_video needs no source media → empty conditioning, no IO.
        let t2v = resolve("text_to_video", json!({}))
            .await
            .expect("t2v resolves");
        assert!(t2v.is_empty(), "t2v needs no conditioning");

        // video_to_video / reference_video_to_video require a source clip.
        let v2v_err = resolve("video_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            v2v_err.contains("source clip"),
            "v2v missing-clip error: {v2v_err}"
        );
        let rv2v_err = resolve("reference_video_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            rv2v_err.contains("source clip"),
            "rv2v missing-clip error: {rv2v_err}"
        );

        // reference_to_video requires at least one reference image.
        let r2v_err = resolve("reference_to_video", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            r2v_err.contains("reference image"),
            "r2v missing-refs error: {r2v_err}"
        );
    }

    /// Real in-process Bernini editing/reference/multi-source video modes through the engine
    /// (sc-4703 / sc-4709 / sc-5425): drive v2v (synthetic source clip → `VideoClip`), r2v
    /// (synthetic reference → `MultiReference`), rv2v (both), mv2v (two source clips), and ads2v
    /// (source clip + reference clip + reference), asserting each mode loads, consumes its
    /// conditioning, and streams RGB8 frames. `#[ignore]` — weights live outside CI; run on a Mac
    /// with the `SceneWorks/bernini-mlx` snapshot.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
    #[test]
    fn bernini_editing_reference_modes_real_weights() {
        let Some(model_dir) = bernini_dir() else {
            eprintln!("skipping bernini_editing_reference_modes_real_weights: no snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let frame = |shade: u8| Image {
            width: w,
            height: h,
            pixels: vec![shade; (w * h * 3) as usize],
        };
        // Two 5-frame synthetic source clips and one synthetic subject reference.
        let clip: Vec<Image> = (0..5).map(|i| frame(60 + i * 12)).collect();
        let clip_b: Vec<Image> = (0..5).map(|i| frame(40 + i * 16)).collect();
        let reference = frame(150);
        let video_clip = |frames: Vec<Image>| Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        };

        let cases: &[(&str, Vec<Conditioning>)] = &[
            ("v2v", vec![video_clip(clip.clone())]),
            (
                "r2v",
                vec![Conditioning::MultiReference {
                    images: vec![reference.clone()],
                }],
            ),
            (
                "rv2v",
                vec![
                    video_clip(clip.clone()),
                    Conditioning::MultiReference {
                        images: vec![reference.clone()],
                    },
                ],
            ),
            // mv2v: two source clips, no references.
            (
                "mv2v",
                vec![video_clip(clip.clone()), video_clip(clip_b.clone())],
            ),
            // ads2v: source clip + reference clip + reference image (videos first, then images).
            (
                "ads2v",
                vec![
                    video_clip(clip.clone()),
                    video_clip(clip_b.clone()),
                    Conditioning::MultiReference {
                        images: vec![reference.clone()],
                    },
                ],
            ),
        ];

        for (task, conditioning) in cases {
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: "bernini",
                model_dir: model_dir.clone(),
                quant: Some(Quant::Q4),
                conditioning: conditioning.clone(),
                prompt: "the subject walks through a neon-lit street, cinematic".to_owned(),
                width: w,
                height: h,
                frames: 5,
                fps: 16,
                steps: Some(8),
                seed: 11,
                video_mode: Some((*task).to_owned()),
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|error| panic!("bernini {task} generation: {error}"));
            assert!(steps > 0, "{task}: denoise progress streamed");
            assert!(!decoded.frames.is_empty(), "{task}: frames returned");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{task}: frames are RGB8-sized"
            );
        }
    }

    // ===================== SCAIL-2 validation (epic 5439 / sc-5450) =====================

    /// Only the `scail2_14b` catalog id routes to the SCAIL-2 engine (sc-5448). Every other video
    /// family (Wan/LTX/SVD/Bernini/image ids) is `None`, so they keep their own routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_engine_id_maps_only_the_scail2_family() {
        assert_eq!(scail2_engine_id("scail2_14b"), Some("scail2_14b"));
        assert_eq!(scail2_engine_id("bernini"), None);
        assert_eq!(scail2_engine_id("wan_2_2"), None);
        assert_eq!(scail2_engine_id("ltx_2_3"), None);
        assert_eq!(scail2_engine_id("scail2"), None);
        assert_eq!(scail2_engine_id(""), None);
    }

    /// SCAIL-2 load quantization (sc-5450): Q4 is the default (the validated ~16 GB tier),
    /// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), parsing a JSON number or string. The
    /// bf16 snapshot is ~47 GB so a missing control NEVER means bf16. Mirrors the Bernini quant test.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_resolve_quant_defaults_q4_and_honors_override() {
        let quant = |advanced: Value| {
            resolve_scail2_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
        };
        assert!(matches!(quant(json!({})), Some(Quant::Q4)));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 4 })),
            Some(Quant::Q4)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": 8 })),
            Some(Quant::Q8)
        ));
        assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
        assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
        // String forms parse the same; a tier between 4 and 8 rounds up to Q8.
        assert!(matches!(
            quant(json!({ "mlxQuantize": "8" })),
            Some(Quant::Q8)
        ));
        assert!(matches!(
            quant(json!({ "mlxQuantize": " 4 " })),
            Some(Quant::Q4)
        ));
        assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
        assert!(matches!(
            quant(json!({ "mlxQuantize": 6 })),
            Some(Quant::Q8)
        ));
    }

    /// Raw-settings lineage on a real SCAIL-2 asset (sc-5450): the real-inference marker, the catalog
    /// model id, the produced frame count / fps, the resolved engine task (`scail2Task`), and a
    /// pass-through of the user's advanced controls. `replace_person` → "replacement",
    /// `animate_character` → "animation".
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_raw_settings_capture_lineage_and_task() {
        let raw_for = |mode: &str| {
            let req = request(json!({
                "projectId": "p",
                "model": "scail2_14b",
                "mode": mode,
                "duration": 5,
                "fps": 16,
                "advanced": { "mlxQuantize": 4, "userKnob": "keep-me" }
            }));
            let raw = scail2_raw_settings(&req, false);
            (raw, req)
        };
        let (raw, req) = raw_for("animate_character");
        let raw = raw.as_object().expect("raw settings is an object");
        assert_eq!(raw["realModelInference"], json!(true));
        assert_eq!(raw["model"], json!("scail2_14b"));
        assert_eq!(raw["fps"], json!(req.fps));
        // No frameCount — the other half of the sc-12371 bug (recorded the raw count while
        // `generate_scail2` rendered the Wan stride). Stamped at the funnel now.
        assert!(raw.get("frameCount").is_none());
        assert_eq!(raw["scail2Task"], json!("animation"));
        assert_eq!(raw["userKnob"], json!("keep-me"));
        // No lightning LoRA ⇒ no effective-recipe override recorded (engine quality defaults stand).
        assert!(raw.get("scail2Lightning").is_none());
        assert!(raw.get("effectiveSteps").is_none());
        // replace_person resolves to the replacement task (flips the engine replace_flag).
        let (raw_replace, _) = raw_for("replace_person");
        assert_eq!(raw_replace["scail2Task"], json!("replacement"));
    }

    /// The lightx2v lightning toggle (sc-5700): selecting a diff-patch LoRA applies the step-distill
    /// recipe — 8 steps, CFG off (guidance 1.0), scheduler shift 1.0 — with an explicit user step
    /// count still honored, and the effective recipe is recorded on the asset. Without lightning the
    /// sampling is all-`None` (the engine's quality defaults stand, unchanged).
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_lightning_recipe_overrides_sampling_and_records() {
        let req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
            "duration": 5, "fps": 16, "advanced": {}
        }));
        // Non-lightning: untouched (engine defaults).
        assert_eq!(scail2_sampling(&req, false), (None, None, None));
        // Lightning: CFG off + shift 1.0 forced, steps default 8.
        assert_eq!(
            scail2_sampling(&req, true),
            (Some(8), Some(1.0), Some(1.0)),
            "lightning applies the 8-step CFG-off recipe"
        );
        // An explicit user step count is honored; CFG/shift stay forced.
        let req_steps = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
            "duration": 5, "fps": 16, "advanced": { "steps": 4 }
        }));
        assert_eq!(
            scail2_sampling(&req_steps, true),
            (Some(4), Some(1.0), Some(1.0))
        );
        // The effective recipe is recorded for observability when lightning is active.
        let raw = scail2_raw_settings(&req, true);
        let raw = raw.as_object().unwrap();
        assert_eq!(raw["scail2Lightning"], json!(true));
        assert_eq!(raw["effectiveSteps"], json!(8));
        assert_eq!(raw["effectiveGuidanceScale"], json!(1.0));
        assert_eq!(raw["effectiveSchedulerShift"], json!(1.0));
    }

    /// SCAIL-2 conditioning resolution fails loudly BEFORE any IO when its required media is missing
    /// (sc-5450 — defense in depth behind the API-side validation): `animate_character` needs a
    /// reference character image, and the integrated `replace_person` backend needs a person track.
    /// Both guards fire before touching the api / job / disk, so a minimal job suffices.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn scail2_conditioning_guards_fire_before_io() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-scail2-1",
            "type": "video_generate",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-14T00:00:00Z",
            "updatedAt": "2026-06-14T00:00:00Z"
        }))
        .expect("job snapshot");

        // animate_character with no reference image / source asset → reference guard.
        let animate_req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "animate_character", "prompt": "go"
        }));
        let animate_err =
            resolve_scail2_conditioning(&api, &settings, &job, &animate_req, Path::new("/tmp/p"))
                .await
                .unwrap_err()
                .to_string();
        assert!(
            animate_err.contains("reference character image"),
            "animate missing-reference error: {animate_err}"
        );

        // replace_person with no person track → person-track guard.
        let replace_req = request(json!({
            "projectId": "p", "model": "scail2_14b", "mode": "replace_person", "prompt": "go"
        }));
        let replace_err = resolve_scail2_replace_conditioning(
            &api,
            &settings,
            &job,
            &replace_req,
            Path::new("/tmp/p"),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            replace_err.contains("person track"),
            "replace missing-track error: {replace_err}"
        );
    }

    /// A SCAIL-2 MLX snapshot dir if present: env override (`SCENEWORKS_MLX_SCAIL2_DIR`), then the
    /// bring-up convert/staging dirs, then the app-managed default. `None` ⇒ the smoke skips.
    #[cfg(target_os = "macos")]
    fn scail2_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SCAIL2_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        for rel in [
            ".cache/scail2-mlx-convert",
            ".cache/mlx-gen-models/scail2-mlx-upload",
            "Library/Application Support/SceneWorks/data/models/mlx/scail2",
        ] {
            let path = PathBuf::from(&home).join(rel);
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        None
    }

    /// A synthetic SCAIL-2 color-coded mask: a centered blue rectangle (person 0) on a solid `bg`,
    /// matching the engine's 7-class color scheme (only the chromatic channels matter, not the exact
    /// silhouette — the smoke proves the engine consumes the shape and streams frames, not parity).
    #[cfg(target_os = "macos")]
    fn scail2_smoke_color_mask(w: u32, h: u32, bg: [u8; 3]) -> Image {
        let (wu, hu) = (w as usize, h as usize);
        let mut px = vec![0u8; wu * hu * 3];
        for chunk in px.chunks_exact_mut(3) {
            chunk.copy_from_slice(&bg);
        }
        for y in (hu / 4)..(3 * hu / 4) {
            for x in (wu / 4)..(3 * wu / 4) {
                let o = (y * wu + x) * 3;
                px[o..o + 3].copy_from_slice(&crate::scail2_masks::PALETTE[0]);
            }
        }
        Image {
            width: w,
            height: h,
            pixels: px,
        }
    }

    /// Real in-process SCAIL-2 through the WORKER registry path (epic 5439 / sc-5450): drive both
    /// engine tasks — `animation` (standalone `animate_character`) and `replacement` (cross-identity
    /// `replace_person`) — with a synthetic reference + color mask + driving clip + per-frame color
    /// masks, asserting each `video_mode` loads via `crate::inference_runtime::load("scail2_14b")`
    /// (proving runtime-macos includes SCAIL-2 — the "no generator registered"
    /// trap), consumes the conditioning, and streams RGB8 frames with denoise progress. The driving
    /// background follows the replacement convention (animation → black, replacement → white).
    /// `#[ignore]` — the ~47 GB snapshot lives outside CI; run manually on a Mac where it is present
    /// (Q4 ≈ 16 GB peak). The bg conventions mirror `scail2_masks` (sc-5448 / sc-5452).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real SCAIL-2 snapshot (~47 GB); run manually on a Mac where the scail2 weights are present"]
    #[test]
    fn scail2_animation_and_replacement_real_weights() {
        let Some(model_dir) = scail2_dir() else {
            eprintln!("skipping scail2_animation_and_replacement_real_weights: no snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        let frame = |shade: u8| Image {
            width: w,
            height: h,
            pixels: vec![shade; (w * h * 3) as usize],
        };
        // 5 driving frames → one clean VAE-aligned segment (the engine keeps ((T-1)/4)*4 + 1).
        let driving: Vec<Image> = (0..5).map(|i| frame(50 + i * 20)).collect();
        let reference = frame(160);

        // animation keeps the reference's world (driving bg black, ref bg white); cross-identity
        // replacement keeps the driving's world (driving bg white, ref bg black) — sc-5448 / sc-5452.
        for (task, driving_bg, reference_bg) in [
            (
                "animation",
                crate::scail2_masks::BG_BLACK,
                crate::scail2_masks::BG_WHITE,
            ),
            (
                "replacement",
                crate::scail2_masks::BG_WHITE,
                crate::scail2_masks::BG_BLACK,
            ),
        ] {
            let reference_mask = scail2_smoke_color_mask(w, h, reference_bg);
            let driving_masks: Vec<Image> = driving
                .iter()
                .map(|_| scail2_smoke_color_mask(w, h, driving_bg))
                .collect();
            let conditioning = vec![
                Conditioning::Reference {
                    image: reference.clone(),
                    strength: None,
                },
                Conditioning::Mask {
                    image: reference_mask,
                },
                Conditioning::ControlClip {
                    frames: driving.clone(),
                    mask: driving_masks,
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: ReplacementMode::default(),
                },
            ];
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: "scail2_14b",
                model_dir: model_dir.clone(),
                quant: Some(Quant::Q4),
                conditioning,
                prompt: "the character walks forward, cinematic".to_owned(),
                width: w,
                height: h,
                frames: 5,
                fps: 16,
                steps: Some(8),
                seed: 11,
                video_mode: Some(task.to_owned()),
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|error| panic!("scail2 {task} generation: {error}"));
            assert!(steps > 0, "{task}: denoise progress streamed");
            assert!(!decoded.frames.is_empty(), "{task}: frames returned");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{task}: frames are RGB8-sized"
            );
        }
    }

    /// Real in-process Wan-VACE extend/bridge through the engine (sc-3812): build the tier-C
    /// control clip (real anchor frames pinned + a neutral generated span, no references) and run
    /// the assembled snapshot, asserting RGB8 frames stream back. `#[ignore]` — the weights live
    /// outside CI; run manually on a Mac with the snapshot assembled (the real-Mac gate; the A/B
    /// vs the TI2V-5B single-frame path is the practical fidelity judge, sc-3800).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
    #[test]
    fn wan_vace_extend_bridge_real_weights() {
        let Some(model_dir) = wan_vace_dir() else {
            eprintln!("skipping wan_vace_extend_bridge_real_weights: no assembled snapshot found");
            return;
        };
        let (w, h) = (256u32, 256u32);
        // Two distinct real "source" frames as the extend motion anchor; the engine generates the
        // remaining 3 frames of the 5-frame budget over the neutral span.
        let anchor = |v: u8| Image {
            width: w,
            height: h,
            pixels: vec![v; (w * h * 3) as usize],
        };
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
            "prompt": "the camera keeps gliding forward, cinematic",
            "sourceClipAssetId": "clip_a"
        }));
        let conditioning = build_extend_bridge_vace_conditioning(
            &req,
            w,
            h,
            5,
            vec![anchor(90), anchor(110)],
            None,
        )
        .expect("extend conditioning builds");
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan_vace",
            model_dir,
            conditioning,
            prompt: req.prompt.clone(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 11,
            control_scale: Some(1.0),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("VACE extend generation");
        assert_eq!(decoded.frames.len(), 5);
        assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Wan model-id → engine-id mapping + the family predicates that drive routing.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_engine_id_maps_the_three_models() {
        assert_eq!(wan_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
        assert_eq!(wan_engine_id("wan_2_2_t2v_14b"), Some("wan2_2_t2v_14b"));
        assert_eq!(wan_engine_id("wan_2_2_i2v_14b"), Some("wan2_2_i2v_14b"));
        assert_eq!(wan_engine_id("ltx_2_3"), None);
        assert_eq!(wan_engine_id("z_image_turbo"), None);
        // sc-3459: VACE-Fun is a replace_person/control engine, NOT a base Wan T2V/I2V engine —
        // it must stay out of `wan_engine_id` so a replace_person job routes to `generate_wan_vace_fun`
        // (the dual-expert `wan2_2_vace_fun_14b` engine), never the base txt/img→video path.
        assert_eq!(wan_engine_id("wan_2_2_vace_fun_14b"), None);
    }

    /// Per-model sampling (sc-4997 / sc-10047): with the Lightning toggle on (the default) both A14B
    /// MoE models (T2V + I2V) force the 4-step Lightning preset (CFG off); the dense 5B honors an
    /// explicit user `steps`/`guidanceScale` and otherwise applies the interim default with CFG retained.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_sampling_overrides_both_14b_and_5b_interim() {
        let req = request(json!({ "projectId": "p" }));
        // Both A14B MoE models default to Lightning on → forced 4-step / guide 1.0.
        assert_eq!(wan_sampling("wan2_2_t2v_14b", &req), (Some(4), Some(1.0)));
        assert_eq!(wan_sampling("wan2_2_i2v_14b", &req), (Some(4), Some(1.0)));
        // 5B, no override → interim default steps, CFG left to the engine (None ⇒ guide 5.0).
        assert_eq!(
            wan_sampling("wan2_2_ti2v_5b", &req),
            (Some(WAN5B_INTERIM_STEPS), None)
        );
        // 5B honors an explicit user steps + guidanceScale (e.g. the ComfyUI fast settings).
        let over = request(json!({
            "projectId": "p", "advanced": { "steps": 6, "guidanceScale": 1.0 }
        }));
        assert_eq!(wan_sampling("wan2_2_ti2v_5b", &over), (Some(6), Some(1.0)));
        // The 14B Lightning preset (toggle on) ignores user steps/guidance — the distill forces 4/1.0.
        assert_eq!(wan_sampling("wan2_2_t2v_14b", &over), (Some(4), Some(1.0)));
        assert_eq!(wan_sampling("wan2_2_i2v_14b", &over), (Some(4), Some(1.0)));
    }

    /// sc-10047: `wan_lightning_on` — default-on toggle. Absent / explicit `true` on an A14B MoE model
    /// ⇒ on; explicit `false` ⇒ off; the dense 5B and any non-Wan engine are never Lightning.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_lightning_on_defaults_on_for_a14b_and_honors_toggle() {
        let absent = request(json!({ "projectId": "p" }));
        // A14B MoE: absent ⇒ default-on (backward compatible with the prior always-on behavior).
        assert!(wan_lightning_on("wan2_2_t2v_14b", &absent));
        assert!(wan_lightning_on("wan2_2_i2v_14b", &absent));
        // Explicit true ⇒ on.
        let on = request(json!({ "projectId": "p", "advanced": { "lightning": true } }));
        assert!(wan_lightning_on("wan2_2_t2v_14b", &on));
        assert!(wan_lightning_on("wan2_2_i2v_14b", &on));
        // Explicit false ⇒ off.
        let off = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
        assert!(!wan_lightning_on("wan2_2_t2v_14b", &off));
        assert!(!wan_lightning_on("wan2_2_i2v_14b", &off));
        // The dense 5B and non-Wan engines are never Lightning regardless of the flag.
        assert!(!wan_lightning_on("wan2_2_ti2v_5b", &absent));
        assert!(!wan_lightning_on("wan2_2_ti2v_5b", &on));
        assert!(!wan_lightning_on("some_other_engine", &on));
    }

    /// sc-10047: with the Lightning toggle OFF, the A14B MoE models run the native multi-step CFG
    /// recipe — honor an explicit user `steps`/`guidanceScale`, else `None` so the engine's config.json
    /// A14B non-distill defaults (40 steps, dual CFG) stand. 5B is unaffected by the toggle.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_sampling_toggle_off_runs_native_multistep_cfg() {
        // Toggle off, no user override → (None, None): the engine's A14B non-distill defaults stand.
        let off = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
        assert_eq!(wan_sampling("wan2_2_t2v_14b", &off), (None, None));
        assert_eq!(wan_sampling("wan2_2_i2v_14b", &off), (None, None));
        // Toggle off, user override → honored (multi-step + CFG on).
        let off_over = request(json!({
            "projectId": "p",
            "advanced": { "lightning": false, "steps": 30, "guidanceScale": 4.0 }
        }));
        assert_eq!(
            wan_sampling("wan2_2_t2v_14b", &off_over),
            (Some(30), Some(4.0))
        );
        assert_eq!(
            wan_sampling("wan2_2_i2v_14b", &off_over),
            (Some(30), Some(4.0))
        );
        // Toggle on with the same overrides → still the forced 4-step Lightning preset.
        let on_over = request(json!({
            "projectId": "p",
            "advanced": { "lightning": true, "steps": 30, "guidanceScale": 4.0 }
        }));
        assert_eq!(
            wan_sampling("wan2_2_t2v_14b", &on_over),
            (Some(4), Some(1.0))
        );
        // 5B ignores the lightning flag entirely (no toggle) — interim default steps still apply.
        let five_b = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
        assert_eq!(
            wan_sampling("wan2_2_ti2v_5b", &five_b),
            (Some(WAN5B_INTERIM_STEPS), None)
        );
    }

    /// `advanced.mlxQuantize` maps to a quant level; absent → dense / engine-resolved.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_quant_maps_mlx_quantize() {
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(resolve_wan_quant(&q4), Some(Quant::Q4));
        let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(resolve_wan_quant(&q8), Some(Quant::Q8));
        let dense = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(resolve_wan_quant(&dense), None);
        let absent = request(json!({ "projectId": "p" }));
        assert_eq!(resolve_wan_quant(&absent), None);
    }

    /// Write the six files that make a Wan2.2 A14B tier subdir COMPLETE
    /// ([`wan_tier_is_complete`]), so [`wan_tier_subdir`] treats it as present.
    #[cfg(target_os = "macos")]
    fn write_complete_wan_tier(root: &Path, tier: &str) {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        for file in [
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "config.json",
        ] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
    }

    /// `mlxQuantize` selects the preferred A14B tier, then falls back to the always-smaller
    /// present tiers (bf16 is only ever tried when explicitly requested).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_tier_order_prefers_then_falls_back() {
        let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(wan_tier_order(&bf16), &["bf16", "q8", "q4"]);
        let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(wan_tier_order(&q8), &["q8", "q4"]);
        // An explicit Q4 pick stays q4-first (never overridden by the q8 default).
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(wan_tier_order(&q4), &["q4", "q8"]);
        // sc-10859: an absent knob defaults q4-first on the video lane (the sc-10726 app-wide q8 default
        // is NOT applied to video — no MLX video Q8 lever, so a silent q8 only risks an OOM). bf16 is
        // never in a default job's search path (no OOM by accident).
        let absent = request(json!({ "projectId": "p" }));
        assert_eq!(wan_tier_order(&absent), &["q4", "q8"]);
    }

    /// [`wan_tier_subdir`] resolves the requested tier, falls back to a smaller COMPLETE
    /// tier, ignores a partially-downloaded one, and returns `None` for a legacy flat root.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_tier_subdir_resolves_and_falls_back() {
        let root = std::env::temp_dir().join(format!("sw_wan_tier_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        // Legacy flat root (no tier subdirs) → None (caller keeps root + load-time quant).
        let q8_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(wan_tier_subdir(&root, &q8_req), None);

        // Only q4 present: a q8 request falls back to the smaller complete q4 tier.
        write_complete_wan_tier(&root, "q4");
        assert_eq!(wan_tier_subdir(&root, &q8_req), Some(root.join("q4")));

        // A partial q8 (missing a file) is skipped, still falling back to q4.
        std::fs::create_dir_all(root.join("q8")).unwrap();
        std::fs::write(root.join("q8").join("config.json"), b"x").unwrap();
        assert_eq!(wan_tier_subdir(&root, &q8_req), Some(root.join("q4")));

        // Completed q8 now wins for an explicit q8 request; but a DEFAULT job (no mlxQuantize) resolves
        // q4-first on the video lane even with q8 installed (sc-10859: the sc-10726 app-wide q8 default
        // is not applied to video — no MLX video Q8 lever). An explicit q4 pick still resolves q4.
        write_complete_wan_tier(&root, "q8");
        assert_eq!(wan_tier_subdir(&root, &q8_req), Some(root.join("q8")));
        let default_req = request(json!({ "projectId": "p" }));
        assert_eq!(wan_tier_subdir(&root, &default_req), Some(root.join("q4")));
        let q4_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(wan_tier_subdir(&root, &q4_req), Some(root.join("q4")));

        // bf16 request with no bf16 tier falls back to q8 (never silently to a default).
        let bf16_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(wan_tier_subdir(&root, &bf16_req), Some(root.join("q8")));
        std::fs::remove_dir_all(&root).ok();
    }

    /// Every Wan quant-matrix tier repo (TI2V-5B sc-9941 / T2V sc-9942 / I2V sc-9943) must pin an
    /// exact commit (not the mutable `main`) so an upstream re-push can't swap a checkpoint the
    /// on-demand fetch loads (mirrors the LTX/SeedVR2 pins). `wan_tier_repo` also routes each model id
    /// to its own repo+revision.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_tier_revisions_are_pinned_commits_not_main() {
        for (label, revision) in [
            ("TI2V-5B", WAN_TI2V_5B_REVISION),
            ("T2V", WAN_T2V_14B_REVISION),
            ("I2V", WAN_I2V_14B_REVISION),
        ] {
            assert_ne!(
                revision, "main",
                "the {label} tier repo must pin a fixed revision before release"
            );
            assert_eq!(
                revision.len(),
                40,
                "a pinned HF revision is a 40-char commit sha ({label})"
            );
            assert!(
                revision
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "the pinned {label} revision must be lowercase hex"
            );
        }
        // Each quant-matrix model id routes to its own repo + revision (the 5B engine's request.model
        // is `wan_2_2`); a model with no hosted matrix has no tier repo.
        assert_eq!(
            wan_tier_repo("wan_2_2"),
            Some((WAN_TI2V_5B_REPO, WAN_TI2V_5B_REVISION))
        );
        assert_eq!(
            wan_tier_repo("wan_2_2_t2v_14b"),
            Some((WAN_T2V_14B_REPO, WAN_T2V_14B_REVISION))
        );
        assert_eq!(
            wan_tier_repo("wan_2_2_i2v_14b"),
            Some((WAN_I2V_14B_REPO, WAN_I2V_14B_REVISION))
        );
        assert_eq!(wan_tier_repo("bernini"), None);
    }

    /// sc-9945: the Bernini quant-matrix repo must pin an exact commit (not the mutable `main`) so an
    /// upstream re-push can't swap a checkpoint the on-demand tier fetch loads. INTENTIONALLY red until
    /// the tiers are hosted and `BERNINI_REVISION` is pinned to the real commit (mirrors sc-9941's flow).
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_tier_revision_is_pinned_commit_not_main() {
        assert_ne!(
            BERNINI_REVISION, "main",
            "the Bernini tier repo must pin a fixed revision before release"
        );
        assert_eq!(
            BERNINI_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            BERNINI_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned Bernini revision must be lowercase hex"
        );
    }

    /// sc-9945: a COMPLETE Bernini tier (all [`BERNINI_TIER_FILES`], incl. the planner's nested
    /// `mllm/tokenizer.json`) resolves for the requested quant, and a missing preferred tier falls
    /// through to the next smaller complete tier — never silently to the wrong one. Composite: the three
    /// packable weights live in the SAME tier subdir as the dense remainder, so one resolution covers
    /// both the planner and the renderer.
    #[cfg(target_os = "macos")]
    #[test]
    fn bernini_tier_subdir_resolves_complete_tier() {
        let root = std::env::temp_dir().join(format!("bernini-tier-test-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let make_tier = |tier: &str| {
            for file in BERNINI_TIER_FILES {
                let path = root.join(tier).join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, b"x").unwrap();
            }
        };
        make_tier("q4");
        make_tier("q8");

        // No explicit pick: the default is LANE-DEPENDENT (sc-10859), clamped to what's on disk. The
        // VIDEO lane defaults q4-first (OOM carve-out); the IMAGE lane keeps epic-10721's q8 default.
        assert_eq!(
            bernini_tier_subdir(&root, None, BERNINI_VIDEO_DEFAULT_TIER_ORDER),
            Some(root.join("q4"))
        );
        assert_eq!(
            bernini_tier_subdir(&root, None, BERNINI_IMAGE_DEFAULT_TIER_ORDER),
            Some(root.join("q8"))
        );
        // Explicit picks are lane-independent (`default_order` is consulted ONLY for `None`).
        // An explicit Q4 pick stays q4 (never overridden by a default).
        assert_eq!(
            bernini_tier_subdir(&root, Some(4), BERNINI_IMAGE_DEFAULT_TIER_ORDER),
            Some(root.join("q4"))
        );
        // Q8 (bits >= 8) → q8.
        assert_eq!(
            bernini_tier_subdir(&root, Some(8), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
            Some(root.join("q8"))
        );
        // bf16 (bits <= 0) with no bf16 tier falls back to q8 (never silently to a default).
        assert_eq!(
            bernini_tier_subdir(&root, Some(0), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
            Some(root.join("q8"))
        );
        // An incomplete tier (missing the nested planner tokenizer) is not resolved.
        std::fs::remove_file(root.join("q8").join("mllm/tokenizer.json")).unwrap();
        assert_eq!(
            bernini_tier_subdir(&root, Some(8), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
            Some(root.join("q4"))
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// The single-expert TI2V-5B tier (sc-9941) is COMPLETE with one `model.safetensors` (not the
    /// A14B two-expert set), and `wan_tier_subdir` resolves it for the `wan_2_2` model id.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_tier_ti2v_5b_single_expert_completeness() {
        assert_eq!(wan_tier_files("wan_2_2"), WAN_TI2V_5B_TIER_FILES);
        assert_eq!(wan_tier_files("wan_2_2_t2v_14b"), WAN_A14B_TIER_FILES);

        let root = std::env::temp_dir().join(format!("sw_wan5b_{}", Uuid::new_v4().simple()));
        let dir = root.join("q4");
        std::fs::create_dir_all(&dir).unwrap();
        for file in WAN_TI2V_5B_TIER_FILES {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
        // Complete for the 5B (single transformer) but NOT for the A14B set (missing the experts).
        assert!(wan_tier_is_complete(&dir, WAN_TI2V_5B_TIER_FILES));
        assert!(!wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES));

        let req = request(json!({ "projectId": "p", "model": "wan_2_2" }));
        assert_eq!(wan_tier_subdir(&root, &req), Some(dir));
        std::fs::remove_dir_all(&root).ok();
    }

    /// Write the six files that make a SCAIL-2 tier subdir COMPLETE ([`scail2_tier_is_complete`]), so
    /// [`scail2_tier_subdir`] treats it as present.
    #[cfg(target_os = "macos")]
    fn write_complete_scail2_tier(root: &Path, tier: &str) {
        let dir = root.join(tier);
        std::fs::create_dir_all(&dir).unwrap();
        for file in SCAIL2_TIER_FILES {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
    }

    /// `mlxQuantize` selects the preferred SCAIL-2 tier, then falls back to the always-smaller present
    /// tiers (bf16 is only ever tried when explicitly requested) — mirrors [`wan_tier_order`].
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_tier_order_prefers_then_falls_back() {
        let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(scail2_tier_order(&bf16), &["bf16", "q8", "q4"]);
        let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(scail2_tier_order(&q8), &["q8", "q4"]);
        // An explicit Q4 pick stays q4-first — the SAME arm as the no-pick default now (sc-10859),
        // which is exactly why the resolver no longer needs its old raw-mlxQuantize-presence check.
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(scail2_tier_order(&q4), &["q4", "q8"]);
        // sc-10859: an absent knob defaults q4-first on the video lane (NOT the sc-10726 app-wide q8 —
        // no MLX video Q8 lever); bf16 is never in a default job's search path (no OOM by accident).
        let absent = request(json!({ "projectId": "p" }));
        assert_eq!(scail2_tier_order(&absent), &["q4", "q8"]);
    }

    /// [`scail2_tier_subdir`] resolves the requested tier, falls back to a smaller COMPLETE tier,
    /// ignores a partially-downloaded one, and returns `None` for a legacy flat root.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_tier_subdir_resolves_and_falls_back() {
        let root = std::env::temp_dir().join(format!("sw_scail2_tier_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        // Legacy flat root (no tier subdirs) → None (caller keeps root + load-time quant).
        let q8_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(scail2_tier_subdir(&root, &q8_req), None);

        // Only q4 present: a q8 request falls back to the smaller complete q4 tier.
        write_complete_scail2_tier(&root, "q4");
        assert_eq!(scail2_tier_subdir(&root, &q8_req), Some(root.join("q4")));

        // A partial q8 (missing files) is skipped, still falling back to q4.
        std::fs::create_dir_all(root.join("q8")).unwrap();
        std::fs::write(root.join("q8").join("config.json"), b"x").unwrap();
        assert_eq!(scail2_tier_subdir(&root, &q8_req), Some(root.join("q4")));

        // Completed q8 now wins for an explicit q8 request; but a DEFAULT job (no mlxQuantize) resolves
        // q4-first on the video lane even with q8 installed (sc-10859: the sc-10726 app-wide q8 default
        // is not applied to video — no MLX video Q8 lever). An explicit q4 pick still resolves q4.
        write_complete_scail2_tier(&root, "q8");
        assert_eq!(scail2_tier_subdir(&root, &q8_req), Some(root.join("q8")));
        let default_req = request(json!({ "projectId": "p" }));
        assert_eq!(
            scail2_tier_subdir(&root, &default_req),
            Some(root.join("q4"))
        );
        let q4_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(scail2_tier_subdir(&root, &q4_req), Some(root.join("q4")));

        // bf16 request with no bf16 tier falls back to q8 (never silently to a default).
        let bf16_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(scail2_tier_subdir(&root, &bf16_req), Some(root.join("q8")));
        std::fs::remove_dir_all(&root).ok();
    }

    /// The SCAIL-2 quant-matrix tier repo (sc-9944) must pin an exact commit (not the mutable `main`)
    /// so an upstream re-push can't swap a checkpoint the on-demand fetch loads (mirrors the Wan pins),
    /// and `scail2_tier_repo` routes only the `scail2_14b` id to it.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_tier_revision_is_pinned_commit_not_main() {
        assert_ne!(
            SCAIL2_REVISION, "main",
            "the SCAIL-2 tier repo must pin a fixed revision before release"
        );
        assert_eq!(
            SCAIL2_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            SCAIL2_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned SCAIL-2 revision must be lowercase hex"
        );
        assert_eq!(
            scail2_tier_repo("scail2_14b"),
            Some((SCAIL2_REPO, SCAIL2_REVISION))
        );
        assert_eq!(scail2_tier_repo("wan_2_2"), None);
        assert_eq!(scail2_tier_repo("bernini"), None);
    }

    /// The `.high_noise.safetensors` → `.low_noise.safetensors` sibling convention
    /// (case-insensitive; only fires when the sibling file exists).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_moe_sibling_pairs_high_and_low() {
        let dir = std::env::temp_dir().join(format!("sw_moe_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let high = dir.join("char.high_noise.safetensors");
        let low = dir.join("char.low_noise.safetensors");
        std::fs::write(&high, b"x").unwrap();
        // No sibling yet → None.
        assert_eq!(wan_moe_low_noise_sibling(&high), None);
        std::fs::write(&low, b"x").unwrap();
        assert_eq!(wan_moe_low_noise_sibling(&high), Some(low));
        // A single-file (non-high-noise) LoRA never pairs.
        let single = dir.join("plain.safetensors");
        std::fs::write(&single, b"x").unwrap();
        assert_eq!(wan_moe_low_noise_sibling(&single), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal valid safetensors (8-byte LE header length + JSON header), optionally stamping
    /// `__metadata__.networkType` so `classify_adapter` can distinguish peft LoKr from plain LoRA.
    #[cfg(target_os = "macos")]
    fn write_lora_fixture(path: &Path, network_type: Option<&str>) {
        let mut meta = serde_json::Map::new();
        meta.insert("format".to_owned(), json!("pt"));
        if let Some(nt) = network_type {
            meta.insert("networkType".to_owned(), json!(nt));
        }
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), Value::Object(meta));
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        std::fs::write(path, buffer).unwrap();
    }

    /// Wan-VACE is single-dense: each user LoRA/LoKr resolves to one shared spec with
    /// `moe_expert: None` (no Lightning, no high/low split), the kind set by the file's metadata,
    /// and the scale taken from the request `weight` (sc-3893).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_vace_adapters_are_single_dense() {
        let dir = std::env::temp_dir().join(format!("sw_vace_lora_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("style.safetensors");
        let lokr = dir.join("char.safetensors");
        write_lora_fixture(&plain, None);
        write_lora_fixture(&lokr, Some("lokr"));

        let req = request(json!({
            "projectId": "p",
            "loras": [
                { "path": plain.to_string_lossy(), "weight": 0.5 },
                { "path": lokr.to_string_lossy(), "weight": 0.9 },
            ],
        }));
        // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
        // the fixture dir the temp LoRAs live in.
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        let specs = resolve_wan_vace_adapters(&settings, &req).expect("resolve vace adapters");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, plain.canonicalize().unwrap());
        assert_eq!(specs[0].kind, AdapterKind::Lora);
        assert!((specs[0].scale - 0.5).abs() < 1e-6);
        assert!(specs[0].moe_expert.is_none(), "VACE is single-dense");
        assert!(specs[0].pass_scales.is_none());
        assert_eq!(specs[1].kind, AdapterKind::Lokr);
        assert!((specs[1].scale - 0.9).abs() < 1e-6);
        assert!(specs[1].moe_expert.is_none());

        // Over the per-job cap → a clear payload error (mirrors the base Wan path).
        let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
            .map(|_| json!({ "path": plain.to_string_lossy() }))
            .collect();
        let over = request(json!({ "projectId": "p", "loras": many }));
        assert!(matches!(
            resolve_wan_vace_adapters(&settings, &over),
            Err(WorkerError::InvalidPayload(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Lay down a fake `lightx2v/Wan2.2-Lightning` HF snapshot under `data_dir` with the
    /// per-architecture high/low pair, so [`resolve_lightning_loras`] resolves the Lightning
    /// distill in a hermetic test (mirrors `write_complete_wan_tier`). Returns the two file paths.
    #[cfg(target_os = "macos")]
    fn write_fake_wan_lightning(data_dir: &Path, engine_id: &str) -> (PathBuf, PathBuf) {
        let subdir = match engine_id {
            "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
            "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            other => panic!("no Lightning subdir for {other}"),
        };
        let base = data_dir
            .join("cache")
            .join("huggingface")
            .join("hub")
            .join("models--lightx2v--Wan2.2-Lightning")
            .join("snapshots")
            .join("deadbeef")
            .join(subdir);
        std::fs::create_dir_all(&base).unwrap();
        let high = base.join("high_noise_model.safetensors");
        let low = base.join("low_noise_model.safetensors");
        std::fs::write(&high, b"x").unwrap();
        std::fs::write(&low, b"x").unwrap();
        (high, low)
    }

    /// sc-10047: `resolve_wan_adapters` gates the A14B Lightning distill pair on the default-on
    /// toggle. Toggle on (default) → the high/low Lightning pair is prepended (per-expert tagged),
    /// then user LoRAs. Toggle off → NO Lightning adapters, but user LoRAs are still honored. Only an
    /// explicit `lightning:false` opts out; absent defaults to on (backward compatible). This test is
    /// hermetic: env vars pointing HF cache elsewhere would break the fixture, so skip when set.
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_adapters_gate_lightning_on_toggle() {
        // The fake HF snapshot only resolves when the cache-dir env overrides are unset (else the
        // real cache is consulted). Skip rather than assert-false in that unusual local config.
        if std::env::var_os("HF_HUB_CACHE").is_some()
            || std::env::var_os("HUGGINGFACE_HUB_CACHE").is_some()
            || std::env::var_os("HF_HOME").is_some()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("sw_wan_light_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let (high, low) = write_fake_wan_lightning(&dir, "wan2_2_t2v_14b");
        // A user LoRA lives under data_dir so the confinement check passes.
        let user_lora = dir.join("style.safetensors");
        write_lora_fixture(&user_lora, None);

        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };

        // Toggle ON (default / absent) → Lightning high/low pair, then the user LoRA (3 specs).
        for adv in [json!({}), json!({ "lightning": true })] {
            let req = request(json!({
                "projectId": "p",
                "advanced": adv,
                "loras": [ { "path": user_lora.to_string_lossy(), "weight": 0.5 } ],
            }));
            let specs = resolve_wan_adapters(&settings, &req, "wan2_2_t2v_14b")
                .expect("resolve adapters (lightning on)");
            assert_eq!(specs.len(), 3, "lightning pair + user lora");
            assert_eq!(specs[0].path, high);
            assert_eq!(specs[0].moe_expert, Some(MoeExpert::High));
            assert_eq!(specs[1].path, low);
            assert_eq!(specs[1].moe_expert, Some(MoeExpert::Low));
            // The user LoRA is the single-file shared spec (no high_noise sibling).
            assert_eq!(specs[2].path, user_lora.canonicalize().unwrap());
            assert!(specs[2].moe_expert.is_none());
            assert!((specs[2].scale - 0.5).abs() < 1e-6);
        }

        // Toggle OFF → NO Lightning adapters; the user LoRA is still applied (1 spec).
        let off = request(json!({
            "projectId": "p",
            "advanced": { "lightning": false },
            "loras": [ { "path": user_lora.to_string_lossy(), "weight": 0.5 } ],
        }));
        let specs = resolve_wan_adapters(&settings, &off, "wan2_2_t2v_14b")
            .expect("resolve adapters (lightning off)");
        assert_eq!(specs.len(), 1, "no Lightning, user LoRA only");
        assert_eq!(specs[0].path, user_lora.canonicalize().unwrap());
        assert!(specs[0].moe_expert.is_none());

        // Toggle OFF works even when the Lightning snapshot is absent (nothing to resolve): remove
        // the fake snapshot and confirm the off path still succeeds with just the user LoRA.
        std::fs::remove_file(&high).unwrap();
        std::fs::remove_file(&low).unwrap();
        let specs_no_light = resolve_wan_adapters(&settings, &off, "wan2_2_t2v_14b")
            .expect("lightning-off needs no Lightning snapshot");
        assert_eq!(specs_no_light.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-5723 (WKA-002): a LoRA path from the (attacker-controllable) payload that
    /// resolves outside every app-managed root is rejected before the file is opened —
    /// the worker must not be pointed at an arbitrary host `.safetensors`. The fixture
    /// lives in a sibling temp dir that is NOT under the configured `data_dir`.
    #[cfg(target_os = "macos")]
    #[test]
    fn lora_path_outside_app_managed_root_is_rejected() {
        let data_dir =
            std::env::temp_dir().join(format!("sw_lora_data_{}", Uuid::new_v4().simple()));
        let outside =
            std::env::temp_dir().join(format!("sw_lora_evil_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let evil = outside.join("evil.safetensors");
        write_lora_fixture(&evil, None);

        let settings = Settings {
            data_dir: data_dir.clone(),
            ..Settings::from_env()
        };
        let req = request(json!({
            "projectId": "p",
            "loras": [ { "path": evil.to_string_lossy(), "weight": 0.5 } ],
        }));
        // The HF cache roots are env-derived; this temp path is under neither, so it
        // must be refused. (Guard against the host's real HF cache happening to be a
        // parent — vanishingly unlikely for a fresh uuid temp dir.)
        let result = resolve_wan_vace_adapters(&settings, &req);
        assert!(
            matches!(result, Err(WorkerError::InvalidPayload(_))),
            "expected out-of-root LoRA to be rejected, got {result:?}"
        );

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// SCAIL-2 is single-dense like Wan-VACE (sc-5451/sc-5686): each user LoRA/LoKr (and the bundled
    /// Bias-Aware DPO LoRA, which arrives the same way) resolves to one shared spec with
    /// `moe_expert: None` and no `pass_scales`, the kind set by the file's metadata, scale from
    /// `weight`; over the per-job cap is a clear payload error.
    #[cfg(target_os = "macos")]
    #[test]
    fn scail2_adapters_are_single_dense() {
        let dir = std::env::temp_dir().join(format!("sw_scail2_lora_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("dpo.safetensors");
        let lokr = dir.join("char.safetensors");
        write_lora_fixture(&plain, None);
        write_lora_fixture(&lokr, Some("lokr"));

        let req = request(json!({
            "projectId": "p",
            "loras": [
                { "path": plain.to_string_lossy(), "weight": 1.0 },
                { "path": lokr.to_string_lossy(), "weight": 0.7 },
            ],
        }));
        // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
        // the fixture dir the temp LoRAs live in.
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        let specs = resolve_scail2_adapters(&settings, &req).expect("resolve scail2 adapters");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, plain.canonicalize().unwrap());
        assert_eq!(specs[0].kind, AdapterKind::Lora);
        assert!((specs[0].scale - 1.0).abs() < 1e-6);
        assert!(specs[0].moe_expert.is_none(), "SCAIL-2 is single-dense");
        assert!(specs[0].pass_scales.is_none());
        assert_eq!(specs[1].kind, AdapterKind::Lokr);
        assert!((specs[1].scale - 0.7).abs() < 1e-6);
        assert!(specs[1].moe_expert.is_none());

        let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
            .map(|_| json!({ "path": plain.to_string_lossy() }))
            .collect();
        let over = request(json!({ "projectId": "p", "loras": many }));
        assert!(matches!(
            resolve_scail2_adapters(&settings, &over),
            Err(WorkerError::InvalidPayload(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-8830 drift-fix regression: the shared [`resolve_lora_file`] resolves a `.safetensors`
    /// nested in a **subdirectory** of the LoRA dir (via core's recursive `first_safetensors_path`).
    /// The old candle twin `candle_resolve_lora_file` used a shallow non-recursive `read_dir`, so it
    /// returned "LoRA has no .safetensors" for exactly this shape — a latent candle-lane bug that
    /// converging on the core helper fixes. Both backends now resolve it identically.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_lora_file_finds_nested_safetensors() {
        let dir = std::env::temp_dir().join(format!("sw_lora_nested_{}", Uuid::new_v4().simple()));
        let nested = dir.join("adapter");
        std::fs::create_dir_all(&nested).unwrap();
        let weight = nested.join("model.safetensors");
        write_lora_fixture(&weight, None);

        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        // Point the resolver at the LoRA dir (not the file); it must recurse into `adapter/`.
        // No declared file → falls back to the recursive scan.
        let resolved = resolve_lora_file(&settings, dir.clone(), None)
            .expect("nested .safetensors must resolve");
        assert_eq!(resolved, weight.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-10221: a trained LoRA's folder holds step checkpoints alongside the final
    /// adapter; the video resolver must load the manifest-declared final file, not an
    /// arbitrary sibling the directory scan might pick.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_lora_file_prefers_declared_over_checkpoint() {
        let dir = std::env::temp_dir().join(format!("sw_lora_ckpt_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let final_adapter = dir.join("my_style.safetensors");
        write_lora_fixture(&dir.join("my_style-step250.safetensors"), None);
        write_lora_fixture(&final_adapter, None);

        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };
        let resolved = resolve_lora_file(&settings, dir.clone(), Some("my_style.safetensors"))
            .expect("declared adapter must resolve");
        assert_eq!(resolved, final_adapter.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-8830: the shared [`resolve_dense_adapters`] enforces exactly the `max_loras` cap it is
    /// handed — one shared count guard for every dense family (Wan-VACE / SCAIL-2, both backends).
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_dense_adapters_honors_the_max_lora_cap() {
        let dir = std::env::temp_dir().join(format!("sw_dense_cap_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("style.safetensors");
        write_lora_fixture(&plain, None);
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };

        // Two LoRAs under a cap of 1 → rejected; the same two under a cap of 2 → accepted.
        let two: Vec<Value> = (0..2)
            .map(|_| json!({ "path": plain.to_string_lossy(), "weight": 0.5 }))
            .collect();
        let req = request(json!({ "projectId": "p", "loras": two }));
        assert!(matches!(
            resolve_dense_adapters(&settings, &req, 1),
            Err(WorkerError::InvalidPayload(_))
        ));
        let specs = resolve_dense_adapters(&settings, &req, 2).expect("under cap resolves");
        assert_eq!(specs.len(), 2);
        assert!(specs.iter().all(|s| s.moe_expert.is_none()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-8830: [`non_empty_negative_prompt`] trims and maps empty/whitespace to `None` (so engines
    /// see "no negative", not the empty string) and passes a real negative through trimmed.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn non_empty_negative_prompt_trims_and_drops_empty() {
        let empty = request(json!({ "projectId": "p", "negativePrompt": "   " }));
        assert_eq!(non_empty_negative_prompt(&empty), None);
        let missing = request(json!({ "projectId": "p" }));
        assert_eq!(non_empty_negative_prompt(&missing), None);
        let real = request(json!({ "projectId": "p", "negativePrompt": "  blurry  " }));
        assert_eq!(non_empty_negative_prompt(&real), Some("blurry".to_owned()));
    }

    /// sc-8830: the shared [`resolve_mlx_dense_quant`] (Bernini / SCAIL-2) defaults to Q4, honors
    /// `mlxQuantize: 8` → Q8, and treats `<= 0` as bf16 (`None`) — never defaulting to bf16.
    ///
    /// The Q4 default is the OWNED exception to epic 10721's app-wide Q8 default (sc-10750): this
    /// resolver is the legacy flat-snapshot load-time quant + the on-demand tier-fetch gate, where Q4 is
    /// the OOM-safe / no-surprise-download floor. The modern turnkey default (Q8-clamped-to-installed)
    /// lives in [`bernini_tier_order`] / [`scail2_tier_order`], not here. This assertion guards that
    /// decision — do not flip it to Q8 without the sc-10733 (S8) capability clamp.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_mlx_dense_quant_defaults_q4_and_honors_override() {
        let default = request(json!({ "projectId": "p" }));
        assert_eq!(resolve_mlx_dense_quant(&default), Some(Quant::Q4));
        let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
        assert_eq!(resolve_mlx_dense_quant(&q8), Some(Quant::Q8));
        let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
        assert_eq!(resolve_mlx_dense_quant(&q4), Some(Quant::Q4));
        let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
        assert_eq!(resolve_mlx_dense_quant(&bf16), None);
        // String forms parse the same way (the payload knob can arrive as a JSON string).
        let q8_str = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": "8" } }));
        assert_eq!(resolve_mlx_dense_quant(&q8_str), Some(Quant::Q8));
    }

    /// A locally-converted Wan2.2 TI2V-5B dir if one is present (env override or the
    /// app-managed default), else `None` so the real-weight smoke skips.
    #[cfg(target_os = "macos")]
    fn wan_5b_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN5B_DIR") {
            let path = PathBuf::from(dir.trim());
            if path.join("config.json").is_file() {
                return Some(path);
            }
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home)
            .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2");
        path.join("config.json").is_file().then_some(path)
    }

    /// Real in-process Wan2.2 TI2V-5B T2V through the engine (the lightest Wan model —
    /// ~10 GB, safe on a 128 GB Mac). Loads the converted 5B snapshot and denoises a
    /// tiny 5-frame clip, asserting frames come back RGB8-sized with streamed progress.
    /// `#[ignore]` — the weights live outside CI; run manually where they are present.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
    #[test]
    fn wan_5b_real_weights() {
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping wan_5b_real_weights: no converted TI2V-5B dir found");
            return;
        };
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width: 256,
            height: 256,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("5B T2V generation");
        assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
        assert!(decoded.fps >= 1);
        assert!(decoded.audio.is_none(), "Wan emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Real in-process load+generate verification for the Wan2.2 T2V-A14B quant-matrix tiers
    /// (sc-9942, epic 8506). For each tier subdir present under `$SCENEWORKS_WAN_T2V_14B_TIER_OUT`
    /// (`bf16/`/`q8/`/`q4/`, e.g. the wan_t2v_14b_tier_build output), loads the tier through the SAME
    /// engine path a job uses (`run_video_generation`, `quant: None` — the pre-packed config.json is
    /// authoritative) and denoises a tiny 5-frame clip, asserting the frames come back RGB8-sized and
    /// NON-degenerate (real pixel variance, so a broken packed load surfaces instead of silent noise).
    /// This is the on-device evidence that every hosted tier loads packed with no install-time convert
    /// peak. Runs a few CFG steps WITHOUT the Lightning distill so it is self-contained (no LoRA
    /// download); output is a coherence check, not a quality bar. `#[ignore]` — needs the built tiers
    /// (~28–69 GB each) + a big-memory Mac; the cache limit is capped so loading bf16 then q8 then q4
    /// in one process does not accumulate residue (sc-5567).
    ///
    /// ```sh
    /// export SCENEWORKS_WAN_T2V_14B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
    /// cargo test -p sceneworks-worker --release --lib wan_t2v_14b_tier_real_weights -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "loads the built Wan2.2 T2V-A14B tiers; run manually on a Mac where they are present"]
    #[test]
    fn wan_t2v_14b_tier_real_weights() {
        let Some(root) = std::env::var_os("SCENEWORKS_WAN_T2V_14B_TIER_OUT").map(PathBuf::from)
        else {
            eprintln!(
                "skipping wan_t2v_14b_tier_real_weights: set SCENEWORKS_WAN_T2V_14B_TIER_OUT"
            );
            return;
        };
        // Cap the buffer cache so three sequential heavy loads release between tiers (sc-5567).
        mlx_rs::memory::set_cache_limit(0);
        let mut verified = 0;
        for tier in ["bf16", "q8", "q4"] {
            let dir = root.join(tier);
            if !wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
                eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
                continue;
            }
            eprintln!("verifying {tier} tier → {}", dir.display());
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: "wan2_2_t2v_14b",
                model_dir: dir.clone(),
                // Pre-packed tier: config.json carries the quant; the engine reconstructs the experts
                // packed and rejects a conflicting override, so load with None (the worker path does
                // the same in resolve_wan_tier_dir_and_quant).
                quant: None,
                prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
                // A few CFG steps WITHOUT the Lightning distill — enough to exercise the packed
                // experts + VAE decode and produce non-degenerate frames for a load check.
                width: 256,
                height: 256,
                frames: 5,
                fps: 16,
                steps: Some(6),
                guidance: Some(5.0),
                seed: 7,
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|e| panic!("{tier} tier T2V generation failed: {e:?}"));
            // The engine's temporal VAE fixes the decoded frame count (a 5-latent-frame request
            // decodes to 8 RGB frames for the A14B); assert a real multi-frame clip, not an exact
            // count.
            assert!(
                decoded.frames.len() > 1,
                "{tier}: got {} frames, expected a multi-frame clip",
                decoded.frames.len()
            );
            assert!(steps > 0, "{tier}: denoise progress streamed");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{tier}: frames are RGB8-sized"
            );
            // Non-degenerate: a broken packed load tends to decode to a flat/NaN frame. Require real
            // spatial variance in the first frame.
            let f0 = &decoded.frames[0];
            let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
            let var = f0
                .pixels
                .iter()
                .map(|&p| (p as f64 - mean).powi(2))
                .sum::<f64>()
                / f0.pixels.len() as f64;
            assert!(
                var.sqrt() > 3.0,
                "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
                var.sqrt()
            );
            mlx_rs::memory::clear_cache();
            verified += 1;
        }
        assert!(
            verified > 0,
            "no tier subdirs found under SCENEWORKS_WAN_T2V_14B_TIER_OUT"
        );
        eprintln!("verified {verified} tier(s)");
    }

    /// Locate a tier subdir (`q4`/`q8`/`bf16`) for a hosted Wan A14B mirror, honoring an explicit
    /// env override first, then the HF hub cache (`~/.cache/huggingface/hub/models--<repo>/snapshots/
    /// <rev>/<tier>`). Returns the tier dir only when it is a COMPLETE A14B tier (the six-file set),
    /// else `None` so the on-device test skips cleanly. Used by
    /// [`wan_a14b_lightning_additive_ondevice`] to drive the real q4 tier + Lightning additive path
    /// (sc-10049, epic 10043) exactly as a job does.
    #[cfg(target_os = "macos")]
    fn wan_a14b_cached_tier(env_var: &str, repo: &str, tier: &str) -> Option<PathBuf> {
        if let Ok(root) = std::env::var(env_var) {
            let dir = PathBuf::from(root.trim()).join(tier);
            if wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
                return Some(dir);
            }
        }
        // HF hub cache: models--<org>--<name>/snapshots/<rev>/<tier>.
        let home = std::env::var("HOME").ok()?;
        let cache = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        let snapshots = std::fs::read_dir(&cache).ok()?;
        for entry in snapshots.flatten() {
            let dir = entry.path().join(tier);
            if wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
                return Some(dir);
            }
        }
        None
    }

    /// Resolve the cached Lightning distill high/low pair for an A14B engine straight from the HF
    /// hub cache (the same layout [`resolve_lightning_loras`] reads under `data_dir`, but pointed at
    /// `~/.cache/huggingface`). Returns `None` when the pair is absent so the on-device test skips.
    #[cfg(target_os = "macos")]
    fn wan_a14b_cached_lightning(engine_id: &str) -> Option<(PathBuf, PathBuf)> {
        let subdir = match engine_id {
            "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
            "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            _ => return None,
        };
        let home = std::env::var("HOME").ok()?;
        let cache = PathBuf::from(&home)
            .join(".cache/huggingface/hub/models--lightx2v--Wan2.2-Lightning/snapshots");
        for entry in std::fs::read_dir(&cache).ok()?.flatten() {
            let base = entry.path().join(subdir);
            let high = base.join("high_noise_model.safetensors");
            let low = base.join("low_noise_model.safetensors");
            if high.is_file() && low.is_file() {
                return Some((high, low));
            }
        }
        None
    }

    /// **sc-10049 (epic 10043) — the on-device close of the loop on sc-10030.** Drives the REAL
    /// Wan2.2 T2V-A14B **q4** tier through the exact engine path a job uses (`run_video_generation`
    /// → `crate::inference_runtime::load` → `Generator::generate`) with the Lightning distill pair installed as
    /// **forward-time additive adapters** on the pre-quantized (packed) experts. This is the exact
    /// case that FAILED before this epic with the mlx-gen `model.rs:614` rejection —
    /// `"LoRA adapters on a pre-quantized snapshot need dequantize-then-merge … not yet wired"` —
    /// and must now complete a 4-step distilled clip with the base STAYING packed (the low-memory-Mac
    /// guarantee this epic exists for: NO ~28 GB/expert dense dequant spike).
    ///
    /// Three sub-cases on the same cached q4 tier:
    ///   1. **Lightning ON, 4 steps** — the Lightning high/low pair (per-expert tagged) as additive
    ///      adapters; asserts a coherent multi-frame clip (the additive path completed, no 614 error).
    ///   2. **Lightning OFF, multi-step CFG** — no adapters, several CFG steps; the native recipe
    ///      still runs on the packed tier (regression guard the toggle-off path is intact).
    ///   3. **Lightning ON + a plain user LoRA** — the distill pair PLUS a single-file plain LoRA
    ///      (the additive user-LoRA path on a packed base). A real Civitai-style LoRA is used when
    ///      `SCENEWORKS_WAN_USER_LORA` points at one; otherwise the low-noise Lightning half doubles
    ///      as a valid single-file plain-LoRA fixture (same PEFT `diffusion_model.`-prefixed keys),
    ///      still exercising the plain-LoRA additive install onto the packed experts.
    ///
    /// Peak RSS is sampled around each generation and printed so the run's log is the memory evidence
    /// (a dequant-then-merge of a 14B bf16 expert would spike ~28 GB/expert; the packed q4 path holds
    /// far below that). `#[ignore]` — needs the cached q4 tier + the Lightning pair + a Metal device.
    ///
    /// ```sh
    /// # tiers + Lightning are auto-discovered in ~/.cache/huggingface; override the tier root with
    /// # SCENEWORKS_WAN_T2V_14B_TIER_OUT and a user LoRA with SCENEWORKS_WAN_USER_LORA if desired.
    /// cargo test -p sceneworks-worker --release --lib \
    ///   wan_a14b_lightning_additive_ondevice -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 T2V-A14B q4 tier + Lightning pair; run manually on a Mac (sc-10049)"]
    #[test]
    fn wan_a14b_lightning_additive_ondevice() {
        const ENGINE: &str = "wan2_2_t2v_14b";
        let Some(tier) = wan_a14b_cached_tier(
            "SCENEWORKS_WAN_T2V_14B_TIER_OUT",
            "SceneWorks/wan2.2-t2v-a14b-mlx",
            "q4",
        ) else {
            eprintln!(
                "skipping wan_a14b_lightning_additive_ondevice: no complete q4 T2V-A14B tier in \
                 ~/.cache/huggingface (or SCENEWORKS_WAN_T2V_14B_TIER_OUT/q4)"
            );
            return;
        };
        let Some((light_high, light_low)) = wan_a14b_cached_lightning(ENGINE) else {
            eprintln!(
                "skipping wan_a14b_lightning_additive_ondevice: Lightning distill pair not cached"
            );
            return;
        };
        eprintln!("q4 tier   → {}", tier.display());
        eprintln!("lightning → {}", light_high.display());

        // Cap the buffer cache so the sequential heavy loads release between sub-cases (sc-5567).
        mlx_rs::memory::set_cache_limit(0);

        // Sub-case common frame/coherence assertions.
        fn assert_coherent(label: &str, decoded: &DecodedVideo, steps: u32) {
            assert!(
                decoded.frames.len() > 1,
                "{label}: got {} frames, expected a multi-frame clip",
                decoded.frames.len()
            );
            assert!(steps > 0, "{label}: denoise progress streamed");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{label}: frames are RGB8-sized"
            );
            let f0 = &decoded.frames[0];
            let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
            let var = f0
                .pixels
                .iter()
                .map(|&p| (p as f64 - mean).powi(2))
                .sum::<f64>()
                / f0.pixels.len() as f64;
            assert!(
                var.sqrt() > 3.0,
                "{label}: frame 0 looks degenerate (std {:.2}) — additive load likely broken",
                var.sqrt()
            );
            eprintln!(
                "{label}: OK — {} frames, {steps} steps, frame0 std {:.2}",
                decoded.frames.len(),
                var.sqrt()
            );
        }

        let run = |label: &str, adapters: Vec<AdapterSpec>, steps: u32, guidance: Option<f32>| {
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: ENGINE,
                model_dir: tier.clone(),
                // Pre-packed q4 tier: config.json is authoritative, so load with None (the worker
                // path does the same). Adapters install ADDITIVELY on the packed experts.
                quant: None,
                adapters,
                prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
                width: 256,
                height: 256,
                frames: 5,
                fps: 16,
                steps: Some(steps),
                guidance,
                seed: 7,
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut n_steps = 0u32;
            let mut on_progress = |p: Progress| {
                if let Progress::Step { .. } = p {
                    n_steps += 1;
                }
            };
            // Reset the MLX GPU allocator's peak counter so `get_peak_memory` below reports THIS
            // sub-case's peak — the number a dense dequant-then-merge would blow up (a 14B bf16
            // expert dequant is ~28 GB/expert; the packed q4 additive path stays far below that).
            mlx_rs::memory::reset_peak_memory();
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|e| panic!("{label} generation failed: {e:?}"));
            assert_coherent(label, &decoded, n_steps);
            let mlx_peak_gib =
                mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!(
                "{label}: MLX peak GPU mem {mlx_peak_gib:.2} GiB, process RSS {} MiB",
                peak_rss_mib()
            );
            // Guard: the packed q4 additive path must hold well under a dense dequant spike. Two
            // dequantized 14B bf16 experts alone would be ~56 GB resident; assert the MLX peak stays
            // below a generous 40 GB ceiling so a regression that dequantizes-then-merges is caught.
            assert!(
                mlx_peak_gib < 40.0,
                "{label}: MLX peak {mlx_peak_gib:.2} GiB — a dense per-expert dequant spike \
                 (~28 GB/expert) appears to have happened; the packed q4 additive path regressed"
            );
            mlx_rs::memory::clear_cache();
        };

        // 1) Lightning ON, 4 steps — the exact pre-epic model.rs:614 failure case, now additive.
        run(
            "q4+lightning(4-step)",
            vec![
                moe_adapter(light_high.clone(), 1.0, AdapterKind::Lora, MoeExpert::High),
                moe_adapter(light_low.clone(), 1.0, AdapterKind::Lora, MoeExpert::Low),
            ],
            4,
            None,
        );

        // 2) Lightning OFF — native multi-step CFG on the packed tier (regression guard).
        run("q4+lightning-off(6-step CFG)", Vec::new(), 6, Some(5.0));

        // 3) Lightning ON + a plain single-file user LoRA (additive user-LoRA on a packed base).
        let user_lora = std::env::var("SCENEWORKS_WAN_USER_LORA")
            .ok()
            .map(|s| PathBuf::from(s.trim().to_owned()))
            .filter(|p| p.is_file())
            // Fall back to the low-noise Lightning half as a valid single-file plain LoRA fixture.
            .unwrap_or_else(|| light_low.clone());
        eprintln!("user LoRA → {}", user_lora.display());
        run(
            "q4+lightning+userLoRA",
            vec![
                moe_adapter(light_high.clone(), 1.0, AdapterKind::Lora, MoeExpert::High),
                moe_adapter(light_low.clone(), 1.0, AdapterKind::Lora, MoeExpert::Low),
                // Single-file → shared across both experts (moe_expert: None), the plain-LoRA
                // additive path (the exact user-Civitai-LoRA shape).
                AdapterSpec::new(user_lora, 0.6, AdapterKind::Lora),
            ],
            4,
            None,
        );
    }

    /// Best-effort current-process resident set size in MiB, read dependency-free via `ps -o rss=`
    /// (macOS reports RSS in KiB). Used only to print memory evidence in the on-device tests — a
    /// coarse guard that the packed q4 additive path holds well below a dense per-expert dequant
    /// spike (~28 GB/expert). Returns 0 when `ps` is unavailable / unparsable (evidence-only).
    #[cfg(target_os = "macos")]
    fn peak_rss_mib() -> u64 {
        let pid = std::process::id().to_string();
        let out = match std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
        {
            Ok(o) if o.status.success() => o.stdout,
            _ => return 0,
        };
        String::from_utf8_lossy(&out)
            .trim()
            .parse::<u64>()
            .map(|kib| kib / 1024)
            .unwrap_or(0)
    }

    /// **sc-10049 (epic 10043) — the I2V sibling of [`wan_a14b_lightning_additive_ondevice`].**
    /// Drives the REAL Wan2.2 **I2V**-A14B **q4** tier with a synthesized start frame + the Lightning
    /// distill pair installed additively on the packed experts. Same close-the-loop assertion as the
    /// T2V case (the pre-epic `model.rs:614` rejection is gone; a 4-step distilled clip completes with
    /// the base staying packed), for the image-to-video engine. A start image is REQUIRED for I2V, so
    /// the test builds a small gradient RGB8 frame (no asset store needed) and passes it as the
    /// `Conditioning::Reference`. Prints the MLX GPU peak per sub-case as the memory evidence and
    /// asserts it holds below the dense-dequant ceiling. `#[ignore]` — needs the cached I2V q4 tier +
    /// the I2V Lightning pair + a Metal device.
    ///
    /// ```sh
    /// cargo test -p sceneworks-worker --release --lib \
    ///   wan_i2v_a14b_lightning_additive_ondevice -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 I2V-A14B q4 tier + Lightning pair; run manually on a Mac (sc-10049)"]
    #[test]
    fn wan_i2v_a14b_lightning_additive_ondevice() {
        const ENGINE: &str = "wan2_2_i2v_14b";
        let Some(tier) = wan_a14b_cached_tier(
            "SCENEWORKS_WAN_I2V_14B_TIER_OUT",
            "SceneWorks/wan2.2-i2v-a14b-mlx",
            "q4",
        ) else {
            eprintln!(
                "skipping wan_i2v_a14b_lightning_additive_ondevice: no complete q4 I2V-A14B tier"
            );
            return;
        };
        let Some((light_high, light_low)) = wan_a14b_cached_lightning(ENGINE) else {
            eprintln!(
                "skipping wan_i2v_a14b_lightning_additive_ondevice: Lightning pair not cached"
            );
            return;
        };
        eprintln!("q4 I2V tier → {}", tier.display());

        mlx_rs::memory::set_cache_limit(0);

        // A small non-flat start frame (diagonal gradient) so the VAE-encoded `y` is meaningful.
        let (w, h) = (256u32, 256u32);
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                pixels.push((x % 256) as u8);
                pixels.push((y % 256) as u8);
                pixels.push(((x + y) % 256) as u8);
            }
        }
        let start = gen_core::Image {
            width: w,
            height: h,
            pixels,
        };
        let conditioning = vec![Conditioning::Reference {
            image: start,
            strength: None,
        }];

        let run = |label: &str, adapters: Vec<AdapterSpec>, steps: u32, guidance: Option<f32>| {
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: ENGINE,
                model_dir: tier.clone(),
                quant: None,
                adapters,
                conditioning: conditioning.clone(),
                prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
                width: w,
                height: h,
                frames: 5,
                fps: 16,
                steps: Some(steps),
                guidance,
                seed: 7,
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut n_steps = 0u32;
            let mut on_progress = |p: Progress| {
                if let Progress::Step { .. } = p {
                    n_steps += 1;
                }
            };
            mlx_rs::memory::reset_peak_memory();
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|e| panic!("{label} I2V generation failed: {e:?}"));
            assert!(
                decoded.frames.len() > 1,
                "{label}: expected a multi-frame clip, got {}",
                decoded.frames.len()
            );
            assert!(n_steps > 0, "{label}: denoise progress streamed");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{label}: frames RGB8-sized"
            );
            let f0 = &decoded.frames[0];
            let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
            let var = f0
                .pixels
                .iter()
                .map(|&p| (p as f64 - mean).powi(2))
                .sum::<f64>()
                / f0.pixels.len() as f64;
            assert!(
                var.sqrt() > 3.0,
                "{label}: frame 0 degenerate (std {:.2})",
                var.sqrt()
            );
            let mlx_peak_gib =
                mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!(
                "{label}: OK — {} frames, {n_steps} steps, std {:.2}, MLX peak {mlx_peak_gib:.2} GiB",
                decoded.frames.len(),
                var.sqrt()
            );
            assert!(
                mlx_peak_gib < 40.0,
                "{label}: MLX peak {mlx_peak_gib:.2} GiB — dense dequant spike; q4 additive regressed"
            );
            mlx_rs::memory::clear_cache();
        };

        // I2V q4 Lightning ON (4-step) — the close-the-loop case for the image-to-video engine.
        run(
            "i2v-q4+lightning(4-step)",
            vec![
                moe_adapter(light_high, 1.0, AdapterKind::Lora, MoeExpert::High),
                moe_adapter(light_low, 1.0, AdapterKind::Lora, MoeExpert::Low),
            ],
            4,
            None,
        );
    }

    /// Real in-process load+generate verification for the Wan2.2 **TI2V-5B** quant-matrix tiers
    /// (sc-9941, epic 8506) — the single-expert sibling of [`wan_t2v_14b_tier_real_weights`]. For each
    /// tier subdir present under `$SCENEWORKS_WAN_TI2V_5B_TIER_OUT` (`bf16/`/`q8/`/`q4/`, the
    /// wan_ti2v_5b_tier_build output), loads the tier through the SAME engine path a job uses
    /// (`run_video_generation`, `quant: None` — the pre-packed config.json is authoritative) and
    /// denoises a tiny text-to-video clip, asserting the frames come back RGB8-sized and NON-degenerate
    /// (real pixel variance, so a broken packed load surfaces instead of silent noise). This is the
    /// on-device evidence that every hosted 5B tier loads packed (single `model.safetensors`
    /// transformer) with no install-time convert peak. Runs a few CFG steps so it is self-contained;
    /// output is a coherence check, not a quality bar. `#[ignore]` — needs the built tiers + a Metal
    /// device; the cache limit is capped so loading bf16 then q8 then q4 in one process does not
    /// accumulate residue (sc-5567).
    ///
    /// ```sh
    /// export SCENEWORKS_WAN_TI2V_5B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
    /// cargo test -p sceneworks-worker --release --lib wan_ti2v_5b_tier_real_weights -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "loads the built Wan2.2 TI2V-5B tiers; run manually on a Mac where they are present"]
    #[test]
    fn wan_ti2v_5b_tier_real_weights() {
        let Some(root) = std::env::var_os("SCENEWORKS_WAN_TI2V_5B_TIER_OUT").map(PathBuf::from)
        else {
            eprintln!(
                "skipping wan_ti2v_5b_tier_real_weights: set SCENEWORKS_WAN_TI2V_5B_TIER_OUT"
            );
            return;
        };
        // Cap the buffer cache so three sequential loads release between tiers (sc-5567).
        mlx_rs::memory::set_cache_limit(0);
        let mut verified = 0;
        for tier in ["bf16", "q8", "q4"] {
            let dir = root.join(tier);
            if !wan_tier_is_complete(&dir, WAN_TI2V_5B_TIER_FILES) {
                eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
                continue;
            }
            eprintln!("verifying {tier} tier → {}", dir.display());
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: "wan2_2_ti2v_5b",
                model_dir: dir.clone(),
                // Pre-packed tier: config.json carries the quant; the engine reconstructs the
                // transformer packed and rejects a conflicting override, so load with None (the worker
                // path does the same in resolve_wan_tier_dir_and_quant).
                quant: None,
                prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
                width: 256,
                height: 256,
                frames: 5,
                fps: 24,
                steps: Some(8),
                seed: 7,
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|e| panic!("{tier} tier TI2V-5B generation failed: {e:?}"));
            assert!(
                decoded.frames.len() > 1,
                "{tier}: got {} frames, expected a multi-frame clip",
                decoded.frames.len()
            );
            assert!(steps > 0, "{tier}: denoise progress streamed");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{tier}: frames are RGB8-sized"
            );
            // Non-degenerate: a broken packed load tends to decode to a flat/NaN frame. Require real
            // spatial variance in the first frame.
            let f0 = &decoded.frames[0];
            let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
            let var = f0
                .pixels
                .iter()
                .map(|&p| (p as f64 - mean).powi(2))
                .sum::<f64>()
                / f0.pixels.len() as f64;
            assert!(
                var.sqrt() > 3.0,
                "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
                var.sqrt()
            );
            mlx_rs::memory::clear_cache();
            verified += 1;
        }
        assert!(
            verified > 0,
            "no tier subdirs found under SCENEWORKS_WAN_TI2V_5B_TIER_OUT"
        );
        eprintln!("verified {verified} tier(s)");
    }

    /// Real in-process load+generate verification for the Wan2.2 **I2V-A14B** quant-matrix tiers
    /// (sc-9943, epic 8506) — the image→video sibling of [`wan_t2v_14b_tier_real_weights`]. For each
    /// tier subdir present under `$SCENEWORKS_WAN_I2V_14B_TIER_OUT` (`bf16/`/`q8/`/`q4/`, the
    /// wan_i2v_14b_tier_build output), loads the tier through the SAME engine path a job uses
    /// (`run_video_generation`, `quant: None` — the pre-packed config.json is authoritative) and,
    /// conditioning on a synthetic START IMAGE (the I2V in_dim-36 image-concat path a T2V run does not
    /// exercise), denoises a tiny clip — asserting the frames come back RGB8-sized and NON-degenerate
    /// (real pixel variance, so a broken packed load or a mis-wired image-cond path surfaces instead of
    /// silent noise). This is the on-device evidence that every hosted I2V tier loads packed with no
    /// install-time convert peak AND drives the image conditioning. Runs a few CFG steps WITHOUT the
    /// Lightning distill so it is self-contained (no LoRA download); output is a coherence check, not a
    /// quality bar. `#[ignore]` — needs the built tiers (~28–69 GB each) + a big-memory Mac; the cache
    /// limit is capped so loading bf16 then q8 then q4 in one process does not accumulate residue
    /// (sc-5567).
    ///
    /// ```sh
    /// export SCENEWORKS_WAN_I2V_14B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
    /// cargo test -p sceneworks-worker --release --lib wan_i2v_14b_tier_real_weights -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[ignore = "loads the built Wan2.2 I2V-A14B tiers; run manually on a Mac where they are present"]
    #[test]
    fn wan_i2v_14b_tier_real_weights() {
        let Some(root) = std::env::var_os("SCENEWORKS_WAN_I2V_14B_TIER_OUT").map(PathBuf::from)
        else {
            eprintln!(
                "skipping wan_i2v_14b_tier_real_weights: set SCENEWORKS_WAN_I2V_14B_TIER_OUT"
            );
            return;
        };
        const W: u32 = 256;
        const H: u32 = 256;
        // A non-degenerate synthetic start frame (a smooth two-axis gradient) so the image-concat
        // conditioning carries real content the engine can propagate — fit through the SAME
        // `fit_engine_image` the I2V resolve path calls, so the reference reaches the engine exactly
        // as a job's would.
        let source = {
            let mut pixels = Vec::with_capacity((W * H * 3) as usize);
            for y in 0..H {
                for x in 0..W {
                    pixels.push((x * 255 / W) as u8);
                    pixels.push((y * 255 / H) as u8);
                    pixels.push(128);
                }
            }
            crate::image_jobs::fit_engine_image(
                Image {
                    width: W,
                    height: H,
                    pixels,
                },
                W,
                H,
                "pad",
            )
            .expect("fit synthetic I2V start frame")
        };
        // Cap the buffer cache so three sequential heavy loads release between tiers (sc-5567).
        mlx_rs::memory::set_cache_limit(0);
        let mut verified = 0;
        for tier in ["bf16", "q8", "q4"] {
            let dir = root.join(tier);
            if !wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
                eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
                continue;
            }
            eprintln!("verifying {tier} tier → {}", dir.display());
            let input = VideoGenInput {
                sampler: None,
                scheduler: None,
                engine_id: "wan2_2_i2v_14b",
                model_dir: dir.clone(),
                // Pre-packed tier: config.json carries the quant; the engine reconstructs the experts
                // packed and rejects a conflicting override, so load with None (the worker path does
                // the same in resolve_wan_tier_dir_and_quant).
                quant: None,
                // The I2V-defining input: a source frame the engine VAE-encodes into its channel-concat
                // conditioning (the in_dim-36 path). Without it the I2V engine errors, so this also
                // proves the image-cond wiring on each packed tier.
                conditioning: vec![Conditioning::Reference {
                    image: source.clone(),
                    strength: None,
                }],
                prompt: "the camera slowly pushes in, cinematic".to_owned(),
                // A few CFG steps WITHOUT the Lightning distill — enough to exercise the packed
                // experts + VAE decode and produce non-degenerate frames for a load check.
                width: W,
                height: H,
                frames: 5,
                fps: 16,
                steps: Some(6),
                guidance: Some(5.0),
                seed: 7,
                ..VideoGenInput::default()
            };
            let cancel = CancelFlag::new();
            let mut steps = 0u32;
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { .. } = progress {
                    steps += 1;
                }
            };
            let decoded = run_video_generation(input, &cancel, &mut on_progress)
                .unwrap_or_else(|e| panic!("{tier} tier I2V generation failed: {e:?}"));
            // The engine's temporal VAE fixes the decoded frame count (a 5-latent-frame request
            // decodes to a multi-frame clip for the A14B); assert a real multi-frame clip.
            assert!(
                decoded.frames.len() > 1,
                "{tier}: got {} frames, expected a multi-frame clip",
                decoded.frames.len()
            );
            assert!(steps > 0, "{tier}: denoise progress streamed");
            assert!(
                decoded
                    .frames
                    .iter()
                    .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
                "{tier}: frames are RGB8-sized"
            );
            // Non-degenerate: a broken packed load or dropped image-cond tends to decode to a
            // flat/NaN frame. Require real spatial variance in the first frame.
            let f0 = &decoded.frames[0];
            let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
            let var = f0
                .pixels
                .iter()
                .map(|&p| (p as f64 - mean).powi(2))
                .sum::<f64>()
                / f0.pixels.len() as f64;
            assert!(
                var.sqrt() > 3.0,
                "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
                var.sqrt()
            );
            mlx_rs::memory::clear_cache();
            verified += 1;
        }
        assert!(
            verified > 0,
            "no tier subdirs found under SCENEWORKS_WAN_I2V_14B_TIER_OUT"
        );
        eprintln!("verified {verified} tier(s)");
    }

    /// Real-weight image→video fit smoke (sc-6139): proves the Crop/Pad pre-fit reaches the
    /// engine and the engine renders the chosen off-aspect output without re-stretching. A
    /// solid-color SQUARE source is fit to a 448×256 landscape via the SAME `fit_engine_image`
    /// the I2V resolve paths call — `pad` letterboxes (black side bars), `crop` fills+trims —
    /// asserted deterministically; then the pad-fit reference drives a tiny TI2V-5B I2V clip and
    /// the decoded frames are asserted to be exactly 448×256 (a re-stretch would change the
    /// aspect). Frame 0 is dumped to `$TMPDIR/sc6139_i2v_pad_frame0.png` for a visual bar check.
    /// `#[ignore]` — needs the converted TI2V-5B weights; run manually on a Mac where present.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
    #[test]
    fn image_to_video_fit_real_weights() {
        // Off-aspect output: a square source must letterbox (pad) or fill+trim (crop).
        const OUT_W: u32 = 448; // 14·32
        const OUT_H: u32 = 256; // 8·32
                                // A solid non-black square so pad's black bars are unambiguous against the content.
        let mk_square = || Image {
            width: 256,
            height: 256,
            pixels: [200u8, 80, 40].repeat((256 * 256) as usize),
        };
        let is_black_col = |img: &Image, x: u32| -> bool {
            (0..img.height).all(|y| {
                let i = ((y * img.width + x) * 3) as usize;
                img.pixels[i] == 0 && img.pixels[i + 1] == 0 && img.pixels[i + 2] == 0
            })
        };

        // PAD: contained (fit height), centered → black bars on the left/right edges.
        let padded =
            crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "pad").expect("pad fit");
        assert_eq!((padded.width, padded.height), (OUT_W, OUT_H));
        assert!(is_black_col(&padded, 0), "pad: left edge is a black bar");
        assert!(
            is_black_col(&padded, OUT_W - 1),
            "pad: right edge is a black bar"
        );
        let ci = ((OUT_H / 2 * OUT_W + OUT_W / 2) * 3) as usize;
        assert_eq!(
            &padded.pixels[ci..ci + 3],
            &[200, 80, 40],
            "pad: center keeps the source color"
        );

        // CROP: covered (fit width), center-cropped → fully filled, no black border.
        let cropped = crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "crop")
            .expect("crop fit");
        assert_eq!((cropped.width, cropped.height), (OUT_W, OUT_H));
        assert!(
            !is_black_col(&cropped, 0),
            "crop: left edge is filled, not a bar"
        );
        assert!(
            !is_black_col(&cropped, OUT_W - 1),
            "crop: right edge is filled, not a bar"
        );

        // Real-weight run: the pad-fit reference drives a tiny TI2V-5B I2V clip. If the engine
        // re-stretched the reference it could not land at the output aspect; asserting the decoded
        // frames are exactly OUT_W×OUT_H confirms the pre-fit reference is consumed as-is.
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping image_to_video_fit_real_weights: no converted TI2V-5B dir found");
            return;
        };
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            conditioning: vec![Conditioning::Reference {
                image: padded,
                strength: None,
            }],
            prompt: "a slow gentle camera move, cinematic".to_owned(),
            width: OUT_W,
            height: OUT_H,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .expect("TI2V-5B I2V generation from a pad-fit square reference");
        assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
        assert!(steps > 0, "denoise progress streamed");
        for f in &decoded.frames {
            assert_eq!(
                (f.width, f.height),
                (OUT_W, OUT_H),
                "engine renders the requested off-aspect output (no re-stretch)"
            );
            assert_eq!(f.pixels.len(), (f.width * f.height * 3) as usize);
        }
        // Dump frame 0 so the pad letterbox is visually verifiable.
        let frame0 = &decoded.frames[0];
        if let Some(buf) =
            image::RgbImage::from_raw(frame0.width, frame0.height, frame0.pixels.clone())
        {
            let out = std::env::temp_dir().join("sc6139_i2v_pad_frame0.png");
            let _ = buf.save(&out);
            eprintln!("wrote pad-fit I2V frame 0 to {}", out.display());
        }
    }

    /// Real-weight perf probe for the TI2V-5B (sc-4997): measures the load / DiT-denoise /
    /// VAE-decode wall-clock split at a configurable resolution, frame count, step count, and
    /// CFG — to ground the "under 10 min" target on real numbers instead of estimates. Env-driven
    /// so configs run without recompiling. MUST be `--release` (debug MLX timing is meaningless):
    ///   SCENEWORKS_MLX_WAN5B_DIR=~/.cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16 \
    ///   WAN_TIMING_W=1280 WAN_TIMING_H=720 WAN_TIMING_FRAMES=121 WAN_TIMING_STEPS=20 WAN_TIMING_CFG=on \
    ///   cargo test -p sceneworks-worker --release --lib wan_5b_timing -- --ignored --nocapture
    /// CFG: `off`/`1` ⇒ guide 1.0 (no CFG); a number ⇒ that scale; `on`/unset ⇒ engine default (5.0).
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight perf probe; needs the converted TI2V-5B snapshot + a Metal device"]
    fn wan_5b_timing() {
        use std::time::Instant;
        let Some(model_dir) = wan_5b_dir() else {
            eprintln!("skipping wan_5b_timing: no converted TI2V-5B dir found");
            return;
        };
        let env_u32 = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(d)
        };
        let width = env_u32("WAN_TIMING_W", 1280);
        let height = env_u32("WAN_TIMING_H", 720);
        let frames = env_u32("WAN_TIMING_FRAMES", 121);
        let steps = env_u32("WAN_TIMING_STEPS", 20);
        let guidance = match std::env::var("WAN_TIMING_CFG")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("off") | Some("1") | Some("1.0") => Some(1.0_f32),
            Some("on") | Some("") | None => None,
            Some(other) => other.parse().ok(),
        };
        // Optional MLX memory-limit override (GB): lowers `get_memory_limit()` so the z48 vae22
        // decode tile planner (`auto_tiling_budgeted`) picks SMALLER tiles — to test whether a
        // smaller decode working-set is faster than the default "largest tile under budget" (sc-5089).
        let mem_limit_gb = env_u32("WAN_TIMING_MEM_LIMIT_GB", 0);
        if mem_limit_gb > 0 {
            let prev =
                mlx_rs::memory::set_memory_limit((mem_limit_gb as usize) * 1024 * 1024 * 1024);
            eprintln!(
                "WAN5B_TIMING mem_limit set to {mem_limit_gb} GB (was {:.0} GB)",
                prev as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        mlx_rs::memory::reset_peak_memory();
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_ti2v_5b",
            model_dir,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width,
            height,
            frames,
            fps: 24,
            steps: Some(steps),
            guidance,
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let start = Instant::now();
        let mut first_step: Option<Instant> = None;
        let mut last_step: Option<Instant> = None;
        let mut decode_at: Option<Instant> = None;
        let mut dit_peak_bytes: Option<usize> = None;
        let mut on_progress = |progress: Progress| match progress {
            Progress::Step { .. } => {
                let now = Instant::now();
                first_step.get_or_insert(now);
                last_step = Some(now);
            }
            Progress::Decoding => {
                decode_at.get_or_insert(Instant::now());
                // Peak so far = the DiT-denoise peak (the VAE decode is the *next* stage).
                dit_peak_bytes.get_or_insert(mlx_rs::memory::get_peak_memory());
            }
            Progress::Loading(_) => {}
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("5B generation");
        let end = Instant::now();
        let total_peak_bytes = mlx_rs::memory::get_peak_memory();
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let secs = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64();
        let fs = first_step.unwrap_or(start);
        let dec = decode_at.or(last_step).unwrap_or(end);
        eprintln!(
            "WAN5B_TIMING {width}x{height} frames={frames} steps={steps} cfg={} => \
             total={:.1}s load={:.1}s dit={:.1}s vae={:.1}s | peak_dit={:.0}GB peak_total={:.0}GB \
             out_frames={}",
            guidance.map_or_else(|| "5.0(default)".to_owned(), |g| format!("{g}")),
            secs(start, end),
            secs(start, fs),
            secs(fs, dec),
            secs(dec, end),
            gib(dit_peak_bytes.unwrap_or(0)),
            gib(total_peak_bytes),
            decoded.frames.len(),
        );
    }

    /// LTX model-id → engine-id mapping (both base + eros load the one engine model).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_engine_id_maps_base_and_eros() {
        assert_eq!(ltx_engine_id("ltx_2_3"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("ltx_2_3_eros"), Some("ltx_2_3"));
        assert_eq!(ltx_engine_id("wan_2_2"), None);
        assert_eq!(ltx_engine_id("z_image_turbo"), None);
    }

    /// SceneWorks bundle resolution (sc-5608): `ltx_bundle_subdir` picks the requested quant subdir,
    /// prefers a complete one over an incomplete sibling, and `bundled_ltx_gemma_dir` finds `gemma/`.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_bundle_subdir_picks_quant_and_finds_gemma() {
        fn write_complete_ltx_dir(dir: &Path) {
            std::fs::create_dir_all(dir).unwrap();
            for file in [
                "connector.safetensors",
                "transformer.safetensors",
                "upsampler.safetensors",
                "vae_decoder.safetensors",
                "vae_encoder.safetensors",
                "audio_vae.safetensors",
                "vocoder.safetensors",
            ] {
                std::fs::write(dir.join(file), b"x").unwrap();
            }
        }
        let root = std::env::temp_dir().join(format!("sw_ltx_bundle_{}", Uuid::new_v4().simple()));
        let (q4, q8, bf16) = (root.join("q4"), root.join("q8"), root.join("bf16"));
        write_complete_ltx_dir(&q4);
        write_complete_ltx_dir(&q8);
        write_complete_ltx_dir(&bf16);
        std::fs::create_dir_all(root.join("gemma")).unwrap();

        // Each tier order prefers its own subdir (default q4, mlxQuantize:8 q8, mlxQuantize<=0 bf16).
        assert_eq!(
            ltx_bundle_subdir(&root, &["q4", "q8"]).as_deref(),
            Some(q4.as_path())
        );
        assert_eq!(
            ltx_bundle_subdir(&root, &["q8", "q4"]).as_deref(),
            Some(q8.as_path())
        );
        assert_eq!(
            ltx_bundle_subdir(&root, &["bf16", "q8", "q4"]).as_deref(),
            Some(bf16.as_path())
        );
        // The default order never loads the huge bf16 tier even when present.
        assert_eq!(
            ltx_bundle_subdir(&root, &["q4", "q8"]).as_deref(),
            Some(q4.as_path())
        );

        // The gemma encoder is found as a sibling of the loaded quant dir.
        assert_eq!(
            bundled_ltx_gemma_dir(&q4).as_deref(),
            Some(root.join("gemma").as_path())
        );

        // An incomplete preferred subdir falls back to the complete sibling (q8 → q4, bf16 → q8).
        std::fs::remove_file(q8.join("vocoder.safetensors")).unwrap();
        assert_eq!(
            ltx_bundle_subdir(&root, &["q8", "q4"]).as_deref(),
            Some(q4.as_path())
        );
        std::fs::remove_file(bf16.join("vocoder.safetensors")).unwrap();
        assert_eq!(
            ltx_bundle_subdir(&root, &["bf16", "q8", "q4"]).as_deref(),
            Some(q4.as_path())
        );

        // No complete subdir → None; no gemma sibling → None.
        let bare = std::env::temp_dir().join(format!("sw_ltx_bare_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(bare.join("q4")).unwrap();
        assert!(ltx_bundle_subdir(&bare, &["q4", "q8"]).is_none());
        assert!(bundled_ltx_gemma_dir(&bare.join("q4")).is_none());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&bare);
    }

    /// Lay down a complete Gemma-3 text-encoder snapshot at `dir`: `config.json`, a two-shard
    /// `model.safetensors.index.json`, and the shards it maps. Mirrors the real bundle `gemma/` so the
    /// completeness + eros-resolution tests are hermetic.
    #[cfg(target_os = "macos")]
    fn write_complete_gemma_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"x").unwrap();
    }

    /// `ltx_gemma_dir_is_complete`: needs `config.json` + every shard the index maps (or a lone
    /// single-file checkpoint); a missing shard, missing config, or bad index all fail.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_gemma_completeness_requires_config_and_all_shards() {
        let root = std::env::temp_dir().join(format!("sw_gemma_ok_{}", Uuid::new_v4().simple()));
        write_complete_gemma_dir(&root);
        assert!(ltx_gemma_dir_is_complete(&root));

        // A missing shard the index references → incomplete (a partial download must not pass).
        std::fs::remove_file(root.join("model-00002-of-00002.safetensors")).unwrap();
        assert!(!ltx_gemma_dir_is_complete(&root));

        // Missing config.json → incomplete even with weights present.
        std::fs::write(root.join("model-00002-of-00002.safetensors"), b"x").unwrap();
        std::fs::remove_file(root.join("config.json")).unwrap();
        assert!(!ltx_gemma_dir_is_complete(&root));

        // Single-file checkpoint (no index) is complete once config + the lone weights file exist.
        let single = std::env::temp_dir().join(format!("sw_gemma_1f_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&single).unwrap();
        std::fs::write(single.join("config.json"), b"{}").unwrap();
        assert!(!ltx_gemma_dir_is_complete(&single));
        std::fs::write(single.join("model.safetensors"), b"x").unwrap();
        assert!(ltx_gemma_dir_is_complete(&single));

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&single);
    }

    /// `resolve_ltx_eros_gemma_dir`: a complete `models/mlx/gemma` sibling of the eros checkpoint wins;
    /// an incomplete/absent sibling with no bundle snapshot → `None` (provider surfaces the clear
    /// "set LTX_GEMMA_DIR" error). Skipped when an operator `$LTX_GEMMA_DIR` is set (the override path
    /// returns `None` by design, exercised by [`resolve_bundled_ltx_gemma_dir`]'s own coverage).
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_ltx_eros_gemma_prefers_local_sibling() {
        if std::env::var_os("LTX_GEMMA_DIR").is_some() {
            return;
        }
        let data = std::env::temp_dir().join(format!("sw_eros_gemma_{}", Uuid::new_v4().simple()));
        let mlx = data.join("models").join("mlx");
        let eros = mlx.join("ltx_2_3_eros");
        std::fs::create_dir_all(&eros).unwrap();
        let settings = Settings {
            data_dir: data.clone(),
            ..Settings::from_env()
        };

        // No sibling gemma and no bundle snapshot in this fresh data dir → None.
        assert!(resolve_ltx_eros_gemma_dir(&settings, &eros).is_none());

        // A complete `models/mlx/gemma` sibling is resolved as the eros TE.
        let gemma = mlx.join("gemma");
        write_complete_gemma_dir(&gemma);
        assert_eq!(
            resolve_ltx_eros_gemma_dir(&settings, &eros).as_deref(),
            Some(gemma.as_path())
        );

        // An incomplete sibling does not win (falls through to the absent bundle → None).
        std::fs::remove_file(gemma.join("model-00001-of-00002.safetensors")).unwrap();
        assert!(resolve_ltx_eros_gemma_dir(&settings, &eros).is_none());

        let _ = std::fs::remove_dir_all(&data);
    }

    /// sc-8827 (F-025): the LTX Gemma-encoder dir rides `LoadSpec::text_encoder` (via
    /// `VideoGenInput::text_encoder_dir`) instead of the process-global `$LTX_GEMMA_DIR`. This asserts
    /// the path flows through `video_load_spec` onto the spec — `None` maps to `None` (env/`<root>`
    /// fallback in the provider), a dir maps to `Some(WeightsSource::Dir(..))`.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn video_load_spec_threads_text_encoder_dir() {
        let base = VideoGenInput {
            engine_id: "ltx_2_3",
            model_dir: PathBuf::from("/models/ltx/q4"),
            ..VideoGenInput::default()
        };
        // No override → text_encoder None (the provider reads its env/`<root>` fallback).
        assert!(
            video_load_spec(&base).text_encoder.is_none(),
            "no text_encoder_dir ⇒ no spec override"
        );
        // An explicit gemma dir rides the spec as a Dir source.
        let gemma = PathBuf::from("/models/ltx/gemma");
        let with_te = VideoGenInput {
            text_encoder_dir: Some(gemma.clone()),
            ..base
        };
        assert!(
            matches!(video_load_spec(&with_te).text_encoder, Some(WeightsSource::Dir(ref p)) if *p == gemma),
            "text_encoder_dir rides LoadSpec::text_encoder as a Dir source"
        );
    }

    /// Q8 opt-in detection (sc-5679): `advanced.mlxQuantize: 8` (int or string) → true; absent / Q4
    /// → false. Drives both the resolve quant preference and the on-demand q8 fetch.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_wants_q8_reads_mlx_quantize() {
        let with = |adv: Value| {
            request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": adv }))
        };
        assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": 8 }))));
        assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": "8" }))));
        assert!(!ltx_wants_q8(&with(json!({ "mlxQuantize": 4 }))));
        assert!(!ltx_wants_q8(&request(
            json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x" })
        )));
    }

    /// bf16 opt-in detection (sc-8513): `advanced.mlxQuantize <= 0` (int or string) → true; Q4/Q8 /
    /// absent → false. Drives the dense-tier preference + the on-demand bf16 fetch. Also asserts the
    /// tier order the three cases resolve to (bf16 only ever tried on an explicit opt-in).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_wants_bf16_and_tier_order() {
        let with = |adv: Value| {
            request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": adv }))
        };
        assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": 0 }))));
        assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": -1 }))));
        assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": "0" }))));
        assert!(!ltx_wants_bf16(&with(json!({ "mlxQuantize": 4 }))));
        assert!(!ltx_wants_bf16(&with(json!({ "mlxQuantize": 8 }))));
        let plain = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x" }));
        assert!(!ltx_wants_bf16(&plain));

        assert_eq!(
            ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 0 }))),
            &["bf16", "q8", "q4"]
        );
        assert_eq!(
            ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 8 }))),
            &["q8", "q4"]
        );
        // sc-10859: a plain request (no mlxQuantize) defaults q4-first on the video lane (NOT the
        // sc-10726 app-wide q8 — no MLX video Q8 lever); bf16 stays out of the default search path.
        assert_eq!(ltx_bundle_tier_order(&plain), &["q4", "q8"]);
        // An explicit Q4 pick stays q4-first.
        assert_eq!(
            ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 4 }))),
            &["q4", "q8"]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn svd_engine_id_maps_only_svd() {
        assert_eq!(svd_engine_id("svd"), Some("svd_xt"));
        assert_eq!(svd_engine_id("ltx_2_3"), None);
        assert_eq!(svd_engine_id("wan_2_2"), None);
        assert_eq!(svd_engine_id("svd_xt"), None);
    }

    /// SVD knobs resolve advanced → manifest entry → builtin default, then clamp; `conditioningFps`
    /// reads the `condFps` manifest key. Mirrors the torch `svd_video` `_pipeline_kwargs` mapping.
    #[cfg(target_os = "macos")]
    #[test]
    fn svd_knobs_resolve_advanced_over_manifest_over_default() {
        // Bare request → builtin defaults.
        let bare = request(json!({ "projectId": "p", "model": "svd", "sourceAssetId": "a" }));
        assert_eq!(svd_i32(&bare, "numFrames", "numFrames", 25, 1, 25), 25);
        assert_eq!(
            svd_i32(&bare, "motionBucketId", "motionBucketId", 127, 1, 255),
            127
        );
        assert_eq!(svd_i32(&bare, "conditioningFps", "condFps", 7, 1, 30), 7);
        assert_eq!(
            svd_i32(&bare, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            8
        );
        assert_eq!(
            svd_f32(&bare, "noiseAugStrength", "noiseAugStrength", 0.02),
            0.02
        );
        assert_eq!(svd_steps(&bare), 25); // balanced

        // Manifest entry overrides the builtin default; advanced overrides the manifest.
        let layered = request(json!({
            "projectId": "p", "model": "svd", "sourceAssetId": "a", "quality": "fast",
            "modelManifestEntry": {
                "motionBucketId": 180, "condFps": 6, "noiseAugStrength": 0.1,
                "steps": { "fast": 12, "balanced": 25, "best": 30 }
            },
            "advanced": { "motionBucketId": 200, "decodeChunkSize": "16" }
        }));
        // advanced wins for motionBucketId; manifest wins for condFps + noiseAug.
        assert_eq!(
            svd_i32(&layered, "motionBucketId", "motionBucketId", 127, 1, 255),
            200
        );
        assert_eq!(svd_i32(&layered, "conditioningFps", "condFps", 7, 1, 30), 6);
        assert_eq!(
            svd_f32(&layered, "noiseAugStrength", "noiseAugStrength", 0.02),
            0.1
        );
        // numeric string parses.
        assert_eq!(
            svd_i32(&layered, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            16
        );
        // steps from manifest's quality ladder (fast).
        assert_eq!(svd_steps(&layered), 12);

        // Out-of-range values clamp to the engine-safe bounds.
        let extreme = request(json!({
            "projectId": "p", "model": "svd", "sourceAssetId": "a",
            "advanced": { "motionBucketId": 999, "numFrames": 99, "decodeChunkSize": 0 }
        }));
        assert_eq!(
            svd_i32(&extreme, "motionBucketId", "motionBucketId", 127, 1, 255),
            255
        );
        assert_eq!(svd_i32(&extreme, "numFrames", "numFrames", 25, 1, 25), 25);
        assert_eq!(
            svd_i32(&extreme, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
            1
        );
    }

    /// Locate the cached SVD-XT diffusers snapshot (the stock HF repo the engine loads directly), or
    /// `None` if absent — `$SCENEWORKS_MLX_SVD_DIR` else the default HF hub cache.
    #[cfg(target_os = "macos")]
    fn svd_real_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
            let path = PathBuf::from(dir.trim());
            if svd_dir_is_complete(&path) {
                return Some(path);
            }
        }
        let snaps = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub")
            .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
            .join("snapshots");
        std::fs::read_dir(&snaps)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| svd_dir_is_complete(path))
    }

    /// Real in-process SVD-XT image→video through the engine: load the stock diffusers snapshot,
    /// animate a synthetic reference image into a tiny clip, and assert the decode seam returns the
    /// requested RGB8 frames at the OUTPUT/playback fps (decoupled from the conditioning fps; sc-3764)
    /// with NO audio and streamed denoise progress. Exercises the worker's `run_video_generation`
    /// path (with the sc-3523 motion knobs) end to end. `#[ignore]` — the weights live outside CI.
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real SVD-XT weights; run manually on a Mac with the checkpoint cached"]
    #[test]
    fn svd_real_weights_image_to_video() {
        let Some(model_dir) = svd_real_dir() else {
            eprintln!("skipping svd_real_weights_image_to_video: no SVD-XT checkpoint found");
            return;
        };
        let (w, h) = (64u32, 64u32);
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                pixels[i] = (x * 255 / w) as u8;
                pixels[i + 1] = (y * 255 / h) as u8;
                pixels[i + 2] = 128;
            }
        }
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "svd_xt",
            model_dir,
            width: 256,
            height: 256,
            frames: 4,
            // Playback fps (24) distinct from the conditioning fps (7) to prove the decouple (sc-3764).
            fps: 24,
            steps: Some(2),
            seed: 7,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: w,
                    height: h,
                    pixels,
                },
                strength: None,
            }],
            motion_bucket_id: Some(127.0),
            noise_aug_strength: Some(0.02),
            decode_chunk_size: Some(2),
            conditioning_fps: Some(7),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("svd i2v generation");
        assert_eq!(decoded.frames.len(), 4, "4 frames");
        assert_eq!(
            decoded.fps, 24,
            "output fps follows the playback fps, not the conditioning fps (sc-3764)"
        );
        assert!(decoded.audio.is_none(), "SVD emits no audio");
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// SeedVR2 video upscale (epic 4811 / sc-4816) end-to-end against the real 3B weights:
    /// drives the same `run_loaded_video_generation` path the `video_upscale` handler uses
    /// (a `VideoClip` of LR frames + a target size + `softness`), asserting an upscaled,
    /// frame-count-preserving clip comes back. `#[ignore]` — the ~7 GB checkpoint lives outside
    /// CI; run manually with `SCENEWORKS_SEEDVR2_DIR` pointed at a `numz/SeedVR2_comfyUI` snapshot
    /// (e.g. the HF cache dir), on a Metal device.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "loads the real SeedVR2 3B weights (~7GB); run manually with SCENEWORKS_SEEDVR2_DIR set"]
    fn seedvr2_video_upscale_real_weights() {
        let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
            .expect("set SCENEWORKS_SEEDVR2_DIR to a numz/SeedVR2_comfyUI checkpoint snapshot dir");
        // 8 low-res frames (a multiple of 4 ≥ 8 so the 4:1 causal-VAE compression preserves the
        // count) fed as the LR `VideoClip`; target = 2× (both ÷16).
        let frames: Vec<Image> = (0..8)
            .map(|i| Image {
                width: 64,
                height: 48,
                pixels: stub_video_rgb8(64, 48, 7, i, 8),
            })
            .collect();
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "seedvr2_3b",
            model_dir: PathBuf::from(dir),
            conditioning: vec![Conditioning::VideoClip {
                frames,
                frame_idx: 0,
                strength: 1.0,
            }],
            width: 128,
            height: 96,
            frames: 8,
            fps: 16,
            seed: 0,
            softness: Some(0.3),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let decoded = run_video_generation(input, &cancel, &mut |_| {})
            .expect("seedvr2 video upscale generation");
        assert_eq!(
            decoded.frames.len(),
            8,
            "frame count preserved (chunk multiple-of-4, ≥8)"
        );
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.width == 128 && f.height == 96),
            "every frame upscaled to the target 128x96"
        );
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "RGB8 buffers are well-formed"
        );
        assert!(
            decoded.audio.is_none(),
            "the engine emits no audio (worker muxes the source)"
        );
    }

    // sc-9595: streaming worker-window chunking for the SeedVR2 upscale (replaces the sc-8829 host-RAM
    // cap). The seam-correctness guarantee is that WORKER-WINDOWED streaming assembly is FRAME-IDENTICAL
    // to a single whole-clip `video::assemble_overlap` — same frame count, order, and cross-fade blend —
    // reusing the engine's own public `plan_chunks`/`assemble_overlap` one level up. These tests model
    // the engine as a deterministic per-chunk upscale (a pure function of its source pixel window +
    // seed, which is what `pipeline::preprocess_chunk` guarantees), exactly as the real driver does, and
    // exercise the `Seedvr2StreamAssembler`. Real-weight PIXEL identity of an internally sub-chunked
    // worker window is a GPU-golden concern the ignore-gated smokes don't run (see the PR notes).

    /// A single-channel synthetic frame (1×1 RGB, all bytes = `v`) — enough to test frame identity,
    /// ordering, and the cross-fade blend byte-exactly.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn syn_frame(v: u8) -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![v, v, v],
        }
    }

    /// Model the engine's whole-clip upscale over synthetic frames: `plan_chunks` at engine chunk `c`,
    /// each chunk = the (identity-)upscaled source window (clamp-padded past the end, like
    /// `preprocess_chunk`), then `assemble_overlap`. Mirrors `pipeline::generate_video` exactly, minus
    /// the real tensor pass — the deterministic per-chunk structure is what worker windowing must
    /// reproduce.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn engine_whole_clip_synthetic(src: &[Image], c: i32, ov: i32) -> Vec<Image> {
        let n = src.len() as i32;
        let plan = seedvr2_video::plan_chunks(n, c, ov);
        let mut chunk_frames: Vec<Vec<Image>> = Vec::with_capacity(plan.len());
        for chunk in &plan {
            let mut frames = Vec::with_capacity(chunk.len as usize);
            for j in 0..chunk.len {
                let idx = (chunk.start + j).clamp(0, n - 1) as usize;
                frames.push(src[idx].clone());
            }
            chunk_frames.push(frames);
        }
        seedvr2_video::assemble_overlap(&plan, &chunk_frames, n, ov)
    }

    /// Drive the worker-window streaming path over synthetic frames: `plan_chunks` at the worker chunk
    /// size, upscale each window via `engine_whole_clip_synthetic` (the engine re-plans internally at
    /// engine chunk `engine_c` over the LOCAL window), and cross-fade across worker windows with the
    /// `Seedvr2StreamAssembler`. Returns the streamed frame sequence + the max retained-tail length
    /// observed (the memory bound).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn worker_streamed_synthetic(
        src: &[Image],
        worker_chunk: i32,
        engine_c: i32,
        ov: i32,
    ) -> (Vec<Image>, usize) {
        let n = src.len() as i32;
        let plan = seedvr2_video::plan_chunks(n, worker_chunk, ov);
        let mut assembler = Seedvr2StreamAssembler::new(n, ov);
        let mut out: Vec<Image> = Vec::new();
        let mut max_tail = 0usize;
        for chunk in &plan {
            let start = chunk.start.max(0) as usize;
            if start >= src.len() {
                break;
            }
            let end = (start + chunk.len.max(0) as usize).min(src.len());
            let window_src: Vec<Image> = src[start..end].to_vec();
            let upscaled = engine_whole_clip_synthetic(&window_src, engine_c, ov);
            out.extend(assembler.push_window(chunk.start, upscaled));
            max_tail = max_tail.max(assembler.retained.len());
        }
        out.extend(assembler.finish());
        (out, max_tail)
    }

    /// THE core seam-correctness guarantee (sc-9595): windowed streaming assembly is frame-identical to
    /// a single whole-clip `assemble_overlap`, across a sweep of frame counts, worker-chunk sizes, and
    /// engine-chunk sizes — including where the worker chunk and engine chunk differ. Byte-exact on the
    /// synthetic frames, so both the frame ORDER and the cross-fade BLEND match the whole-clip path.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_stream_matches_whole_clip_assembly() {
        let ov = seedvr2_video::DEFAULT_OVERLAP;
        let mut cases = 0;
        for n in [1usize, 5, 8, 12, 16, 20, 28, 40, 41, 64, 100, 137, 256, 257] {
            let src: Vec<Image> = (0..n)
                .map(|i| syn_frame(((i * 13 + 5) % 251) as u8))
                .collect();
            for &worker_chunk in &[16i32, 20, 32, 64] {
                for &engine_c in &[8i32, 12, 16, 24, 32] {
                    let whole = engine_whole_clip_synthetic(&src, engine_c, ov);
                    let (streamed, _) = worker_streamed_synthetic(&src, worker_chunk, engine_c, ov);
                    assert_eq!(
                        streamed.len(),
                        n,
                        "frame count preserved (n={n}, worker={worker_chunk}, engine={engine_c})"
                    );
                    assert_eq!(
                        streamed, whole,
                        "windowed == whole-clip (n={n}, worker={worker_chunk}, engine={engine_c})"
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases > 200, "swept a broad matrix ({cases} cases)");
    }

    /// The streaming assembler holds BOUNDED memory: the retained tail never exceeds one worker window
    /// plus the overlap, regardless of clip length — the property that removes the whole-clip host-RAM
    /// footprint the sc-8829 cap existed to bound.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_stream_retains_bounded_tail() {
        let ov = seedvr2_video::DEFAULT_OVERLAP;
        let worker_chunk = SEEDVR2_WORKER_CHUNK_FRAMES;
        // A long clip that would be tens of GB of RGB8 whole — the exact case the cap rejected.
        let n = 4000usize;
        let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
        let (streamed, max_tail) = worker_streamed_synthetic(&src, worker_chunk, 16, ov);
        assert_eq!(streamed.len(), n, "every frame emitted, in order");
        // The retained tail is at most one merged worker window (worker_chunk + a partial next window's
        // overlap), never the whole clip — orders of magnitude below `n`.
        assert!(
            max_tail as i32 <= worker_chunk + ov + 1,
            "retained tail {max_tail} bounded by one window (~{}), not the {n}-frame clip",
            worker_chunk + ov
        );
    }

    /// Frame ORDER is preserved end-to-end: each streamed frame carries its source ordinal through the
    /// (identity) upscale + cross-fade, so the sequence is monotonic and complete (no drops/dupes/swaps)
    /// even across worker-window boundaries with a partial trailing window.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_stream_preserves_frame_order() {
        let ov = seedvr2_video::DEFAULT_OVERLAP;
        // 150 frames, worker chunk 64 → 3 worker windows with a partial last one; a value per ordinal.
        let n = 150usize;
        let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
        let (streamed, _) = worker_streamed_synthetic(&src, SEEDVR2_WORKER_CHUNK_FRAMES, 16, ov);
        assert_eq!(streamed.len(), n);
        // Non-overlap frames pass through unblended → equal to the source ordinal value; overlap frames
        // are cross-faded but between two IDENTICAL-value windows here only at seams. Compare to the
        // whole-clip oracle for the exact expected values.
        let whole = engine_whole_clip_synthetic(&src, 16, ov);
        assert_eq!(
            streamed, whole,
            "streamed frames match the whole-clip oracle"
        );
    }

    /// Cap removal (sc-9595): a clip that the sc-8829 cap would have REJECTED (huge whole-clip RGB8
    /// footprint) now streams to completion with the correct frame count and no error — the capability
    /// narrowing is gone. (The removed `check_seedvr2_host_ram` / `seedvr2_estimated_host_bytes`
    /// helpers no longer exist; this test would not compile if a hard cap were reintroduced in the
    /// streaming path.)
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn seedvr2_stream_upscales_formerly_capped_clip() {
        let ov = seedvr2_video::DEFAULT_OVERLAP;
        // ~4300 frames — the finding's pathological few-minute 1080p clip the old cap rejected outright.
        let n = 4300usize;
        let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
        let (streamed, max_tail) =
            worker_streamed_synthetic(&src, SEEDVR2_WORKER_CHUNK_FRAMES, 16, ov);
        assert_eq!(
            streamed.len(),
            n,
            "the formerly-capped clip streams in full"
        );
        assert!(
            (max_tail as i32) < n as i32 / 10,
            "peak retained frames stay far below the whole clip (bounded memory)"
        );
    }

    /// The `ScratchDir` RAII guard removes the (multi-GB, in prod) output PNG scratch on the ERROR path
    /// (sc-9595 review): dropping an ARMED guard deletes a populated dir — this is the guarantee that
    /// covers a create_dir_all / progress-POST / encode `?`-return or a cancel between the stream and the
    /// encode, where the old code leaked the whole 4× sequence. Disarming (the success path, after the
    /// caller already removed the dir) makes Drop a no-op so it can't fight a concurrent same-name reuse.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn scratch_dir_guard_cleans_up_on_drop_and_respects_disarm() {
        let base = std::env::temp_dir().join(format!(
            "sceneworks_seedvr2_scratchtest_{}",
            Uuid::new_v4().simple()
        ));

        // ARMED guard dropped (error path): a populated dir + a nested file are removed on Drop.
        let armed_dir = base.join("armed");
        std::fs::create_dir_all(&armed_dir).unwrap();
        std::fs::write(armed_dir.join("frame_00001.png"), b"not-a-real-png").unwrap();
        assert!(armed_dir.exists(), "precondition: scratch populated");
        {
            let _guard = ScratchDir::new(armed_dir.clone());
            // Simulate the post-stream encode span returning Err before any manual cleanup: the guard
            // goes out of scope here without disarming.
        }
        assert!(
            !armed_dir.exists(),
            "armed guard's Drop must remove the leaked output scratch on the error path"
        );

        // DISARMED guard dropped (success path): the caller already removed the dir, and Drop must NOT
        // re-remove — modelled by a dir the guard deliberately leaves intact.
        let disarmed_dir = base.join("disarmed");
        std::fs::create_dir_all(&disarmed_dir).unwrap();
        {
            let mut guard = ScratchDir::new(disarmed_dir.clone());
            guard.disarm();
        }
        assert!(
            disarmed_dir.exists(),
            "disarmed guard's Drop must be a no-op (caller owns cleanup)"
        );

        // A guard over an ALREADY-removed dir drops cleanly (benign NotFound, no panic).
        let missing_dir = base.join("missing");
        {
            let _guard = ScratchDir::new(missing_dir.clone());
        }
        assert!(!missing_dir.exists());

        let _ = std::fs::remove_dir_all(&base);
    }

    /// `advanced.noAudio` maps to the engine's `video_mode = "no_audio"`; enhance flags
    /// flow through. Asserts the LTX request build (the VideoGenInput, pre-load).
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_advanced_flags_map_to_video_gen_input() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "advanced": { "noAudio": true, "enhancePrompt": true }
        }));
        assert!(advanced::bool(&req.advanced, "noAudio"));
        assert!(advanced::bool(&req.advanced, "enhancePrompt"));
        assert!(!advanced::bool(&req.advanced, "useUncensoredEnhancer"));
        // LTX adapters: a plain user LoRA is uniform (no per-pass schedule, no moe tag).
        let settings = Settings::from_env();
        let none = resolve_ltx_adapters(&settings, &req).unwrap();
        assert!(none.is_empty());
    }

    /// Lay down a fake HF snapshot file under `data_dir`
    /// (`cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/<file>`) and return its path, so
    /// [`resolve_ltx_distill_adapter`] resolves hermetically (mirrors `write_fake_wan_lightning`).
    #[cfg(target_os = "macos")]
    fn write_fake_hf_lora(data_dir: &Path, repo: &str, file: &str) -> PathBuf {
        let snapshot = data_dir
            .join("cache")
            .join("huggingface")
            .join("hub")
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join("deadbeef");
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join(file);
        write_lora_fixture(&path, None);
        path
    }

    /// The `ltx_2_3_eros` manifest shape [`resolve_ltx_distill_adapter`] reads: `mlx.autoDistillLora`
    /// per-pass strengths plus the `resources.distilledLora` repo/file.
    #[cfg(target_os = "macos")]
    fn eros_manifest_entry(repo: &str, file: &str) -> Value {
        json!({
            "mlx": { "autoDistillLora": { "stage1Strength": 1.0, "stage2Strength": 0.4 } },
            "resources": { "distilledLora": { "repo": repo, "file": file } },
        })
    }

    /// Regression (10Eros noise): `ltx_2_3_eros` is not pre-distilled, so `resolve_ltx_adapters` must
    /// auto-inject its cond_safe distill LoRA at per-pass strengths (1.0 first pass / 0.4 upscale) ahead
    /// of any user LoRAs — the injection was lost in the sc-3037 Python→Rust video cutover, leaving the
    /// undistilled base to collapse to noise. Strengths honor `advanced.distillStage*Strength`;
    /// `advanced.useDistillLora = false` opts out.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_eros_auto_injects_distill_lora_per_pass() {
        // Hold ENV_LOCK for the whole test. This fixture resolves ONLY while the HF cache-dir env
        // overrides are unset (`huggingface_hub_cache_dir` reads them BEFORE `data_dir`), and the
        // check below reads them once, at entry. Without the lock a concurrent `temp_env_var(s)`
        // caller — e.g. `generate_mochi_using_refuses_before_paying_for_the_tier_download`, which
        // pins `HF_HUB_CACHE` to its own fixture hub — can set the var AFTER that check passes and
        // BEFORE `resolve_ltx_adapters` reads it, pointing the resolver at the wrong cache: the
        // distill LoRA is then "not installed" and this test fails for a reason that has nothing to
        // do with it. `set_var` is process-global; only the lock makes the read-then-use atomic.
        // Deterministic (12/12) when the two tests are selected together (sc-12306).
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // The fake HF snapshot only resolves when the cache-dir env overrides are unset (else the real
        // cache is consulted). Skip rather than assert-false in that unusual local config.
        if std::env::var_os("HF_HUB_CACHE").is_some()
            || std::env::var_os("HUGGINGFACE_HUB_CACHE").is_some()
            || std::env::var_os("HF_HOME").is_some()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("sw_eros_distill_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let repo = "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments";
        let file = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors";
        let distill = write_fake_hf_lora(&dir, repo, file);
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };

        // Default: the distill LoRA is injected alone, at per-pass 1.0/0.4.
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
            "modelManifestEntry": eros_manifest_entry(repo, file),
        }));
        let specs = resolve_ltx_adapters(&settings, &req).expect("resolve eros adapters");
        assert_eq!(specs.len(), 1, "the distill LoRA must be injected");
        assert_eq!(specs[0].path, distill);
        assert_eq!(specs[0].kind, AdapterKind::Lora);
        assert_eq!(specs[0].pass_scales, Some(vec![1.0, 0.4]));

        // A user LoRA stacks AFTER the distill (the distill is the model's base recipe).
        let user = dir.join("style.safetensors");
        write_lora_fixture(&user, None);
        let req_user = request(json!({
            "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
            "modelManifestEntry": eros_manifest_entry(repo, file),
            "loras": [ { "path": user.to_string_lossy(), "weight": 0.5 } ],
        }));
        let specs = resolve_ltx_adapters(&settings, &req_user).expect("resolve eros + user");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].pass_scales, Some(vec![1.0, 0.4]), "distill first");
        assert!(
            specs[1].pass_scales.is_none(),
            "user LoRA is uniform per-pass"
        );
        assert!((specs[1].scale - 0.5).abs() < 1e-6);

        // `advanced` overrides the per-pass strengths.
        let req_override = request(json!({
            "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
            "modelManifestEntry": eros_manifest_entry(repo, file),
            "advanced": { "distillStage1Strength": 0.8, "distillStage2Strength": 0.2 },
        }));
        let specs = resolve_ltx_adapters(&settings, &req_override).expect("override strengths");
        assert_eq!(specs[0].pass_scales, Some(vec![0.8, 0.2]));

        // Opt-out disables the injection entirely (no user LoRAs → empty).
        let req_off = request(json!({
            "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
            "modelManifestEntry": eros_manifest_entry(repo, file),
            "advanced": { "useDistillLora": false },
        }));
        assert!(resolve_ltx_adapters(&settings, &req_off)
            .unwrap()
            .is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A model that declares `mlx.autoDistillLora` but whose co-requisite distill LoRA is not installed
    /// fails with an actionable payload error rather than silently degrading to noise; a model that
    /// declares no `autoDistillLora` (base `ltx_2_3`) injects nothing.
    #[cfg(target_os = "macos")]
    #[test]
    fn ltx_distill_lora_missing_errors_and_base_model_is_noop() {
        if std::env::var_os("HF_HUB_CACHE").is_some()
            || std::env::var_os("HUGGINGFACE_HUB_CACHE").is_some()
            || std::env::var_os("HF_HOME").is_some()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("sw_eros_missing_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let settings = Settings {
            data_dir: dir.clone(),
            ..Settings::from_env()
        };

        // Declared distill LoRA, but nothing on disk → actionable error (not silent noise).
        let repo = "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments";
        let file = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors";
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
            "modelManifestEntry": eros_manifest_entry(repo, file),
        }));
        assert!(matches!(
            resolve_ltx_adapters(&settings, &req),
            Err(WorkerError::InvalidPayload(_))
        ));

        // Base `ltx_2_3` declares no `autoDistillLora` → no injection.
        let base = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "modelManifestEntry": { "family": "ltx-video" },
        }));
        assert!(resolve_ltx_adapters(&settings, &base).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The FLF keyframe knobs (sc-3055) parse from JSON numbers + numeric strings and fall back
    /// to their defaults — these drive the two `Keyframe` strengths + the first keyframe's index.
    #[cfg(target_os = "macos")]
    #[test]
    fn advanced_numeric_helpers_parse_flf_keyframe_knobs() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame",
            "advanced": {
                "imageConditioningStrength": 0.8,        // JSON number
                "lastFrameConditioningStrength": "0.65",  // numeric string
                "imageFrameIndex": 2
            }
        }));
        assert_eq!(
            advanced::f32(&req.advanced, "imageConditioningStrength", 1.0),
            0.8
        );
        assert_eq!(
            advanced::f32(&req.advanced, "lastFrameConditioningStrength", 1.0),
            0.65
        );
        assert_eq!(advanced::i32(&req.advanced, "imageFrameIndex", 0), 2);
        // Absent keys → the fully-pinned defaults (strength 1.0, first index 0).
        let bare = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "a fox" }));
        assert_eq!(
            advanced::f32(&bare.advanced, "imageConditioningStrength", 1.0),
            1.0
        );
        assert_eq!(
            advanced::f32(&bare.advanced, "lastFrameConditioningStrength", 1.0),
            1.0
        );
        assert_eq!(advanced::i32(&bare.advanced, "imageFrameIndex", 0), 0);
    }

    /// `resolve_keyframe_conditioning` fails clearly when an FLF source/last-frame asset id is
    /// missing (the guards run before any project/image IO, so no fixture is needed).
    #[cfg(target_os = "macos")]
    #[test]
    fn keyframe_conditioning_requires_both_frame_assets() {
        let settings = Settings::from_env();
        // No sourceAssetId.
        let no_first = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame", "lastFrameAssetId": "asset_last"
        }));
        let err = resolve_keyframe_conditioning(&settings, &no_first, Path::new("/tmp/p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("source image"), "got: {err}");
        // sourceAssetId but no lastFrameAssetId.
        let no_last = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
            "mode": "first_last_frame", "sourceAssetId": "asset_first"
        }));
        let err = resolve_keyframe_conditioning(&settings, &no_last, Path::new("/tmp/p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("last-frame image"), "got: {err}");
    }

    /// FLF on a 14B Wan MoE engine is rejected at the conditioning resolver (defence-in-depth
    /// behind the routing gate, which already restricts FLF to `wan_2_2`/TI2V-5B).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_flf_rejected_on_non_ti2v_engine() {
        let settings = Settings::from_env();
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "prompt": "a fox",
            "mode": "first_last_frame",
            "sourceAssetId": "a", "lastFrameAssetId": "b"
        }));
        let err = resolve_wan_conditioning(&settings, &req, Path::new("/tmp/p"), "wan2_2_t2v_14b")
            .unwrap_err()
            .to_string();
        assert!(err.contains("TI2V-5B"), "got: {err}");
    }

    /// A 1×1 RGB [`Image`] for clip-conditioning construction tests (the engine resizes; the
    /// content is irrelevant — only the variant / frame_idx / strength mapping is under test).
    #[cfg(target_os = "macos")]
    fn pixel(n: u8) -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![n, n, n],
        }
    }

    /// extend_clip → one `VideoClip` pinned at latent frame 0, strength `videoConditioningStrength`
    /// (default 1.0); the bridge-only right knob is ignored.
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_extend_maps_single_clip_at_zero() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
            "mode": "extend_clip",
            "advanced": { "videoConditioningStrength": 0.7 }
        }));
        let cond = build_video_clip_conditioning(&req, vec![pixel(1), pixel(2)], None).unwrap();
        assert_eq!(cond.len(), 1);
        match &cond[0] {
            Conditioning::VideoClip {
                frames,
                frame_idx,
                strength,
            } => {
                assert_eq!(frames.len(), 2);
                assert_eq!(*frame_idx, 0);
                assert_eq!(*strength, 0.7);
            }
            other => panic!("expected VideoClip, got {other:?}"),
        }
    }

    /// video_bridge → left clip at 0 (`videoConditioningStrength`) + right clip at -1
    /// (`bridgeRightVideoConditioningStrength`); both default to 1.0 when absent.
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_bridge_maps_left_zero_right_tail() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
            "mode": "video_bridge",
            "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
        }));
        let cond =
            build_video_clip_conditioning(&req, vec![pixel(1)], Some(vec![pixel(2), pixel(3)]))
                .unwrap();
        assert_eq!(cond.len(), 2);
        match (&cond[0], &cond[1]) {
            (
                Conditioning::VideoClip {
                    frames: left,
                    frame_idx: left_idx,
                    strength: left_strength,
                },
                Conditioning::VideoClip {
                    frames: right,
                    frame_idx: right_idx,
                    strength: right_strength,
                },
            ) => {
                assert_eq!(left.len(), 1);
                assert_eq!(*left_idx, 0);
                assert_eq!(*left_strength, 1.0); // default
                assert_eq!(right.len(), 2);
                assert_eq!(*right_idx, -1); // engine negative-from-end (lf + idx)
                assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
            }
            other => panic!("expected two VideoClips, got {other:?}"),
        }
    }

    /// video_bridge without the right clip frames is a construction error (defence behind the
    /// resolver's `bridgeRightClipAssetId` guard).
    #[cfg(target_os = "macos")]
    #[test]
    fn video_clip_conditioning_bridge_requires_right_clip() {
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
            "mode": "video_bridge"
        }));
        let err = build_video_clip_conditioning(&req, vec![pixel(1)], None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("right-side source clip"), "got: {err}");
    }

    /// Wan extend_clip → one boundary `Keyframe` at latent frame 0 (the source clip's last frame),
    /// strength `videoConditioningStrength` (default 1.0); the right frame is ignored (sc-3357).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_extend_pins_last_frame_at_zero() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "extend it",
            "mode": "extend_clip",
            "advanced": { "videoConditioningStrength": 0.7 }
        }));
        let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
        assert_eq!(cond.len(), 1);
        match &cond[0] {
            Conditioning::Keyframe {
                frame_idx,
                strength,
                ..
            } => {
                assert_eq!(*frame_idx, 0);
                assert_eq!(*strength, 0.7);
            }
            other => panic!("expected Keyframe, got {other:?}"),
        }
    }

    /// Wan video_bridge → left clip's last frame at 0 (`videoConditioningStrength`) + right clip's
    /// first frame at -1 (`bridgeRightVideoConditioningStrength`); mechanically FLF (sc-3357).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_bridge_pins_both_boundaries() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
            "mode": "video_bridge",
            "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
        }));
        let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
        assert_eq!(cond.len(), 2);
        match (&cond[0], &cond[1]) {
            (
                Conditioning::Keyframe {
                    frame_idx: left_idx,
                    strength: left_strength,
                    ..
                },
                Conditioning::Keyframe {
                    frame_idx: right_idx,
                    strength: right_strength,
                    ..
                },
            ) => {
                assert_eq!(*left_idx, 0);
                assert_eq!(*left_strength, 1.0); // default
                assert_eq!(*right_idx, -1); // engine negative-from-end
                assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
            }
            other => panic!("expected two Keyframes, got {other:?}"),
        }
    }

    /// Wan video_bridge without the right boundary frame is a construction error (defence behind
    /// the resolver's `bridgeRightClipAssetId` guard).
    #[cfg(target_os = "macos")]
    #[test]
    fn wan_boundary_conditioning_bridge_requires_right_frame() {
        let req = request(json!({
            "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
            "mode": "video_bridge"
        }));
        let err = build_wan_boundary_conditioning(&req, pixel(1), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("right-side source clip"), "got: {err}");
    }

    /// The IC-LoRA detector matches the torch `lora_looks_like_ic_lora` markers (flags, role,
    /// and "ic-lora" / "ltx-2-3-ic-" in id/name/path/files) and rejects an ordinary LoRA.
    #[cfg(target_os = "macos")]
    #[test]
    fn ic_lora_detection_matches_torch_markers() {
        // Explicit flags.
        assert!(loras_contain_ic_lora(&[json!({ "icLora": true })]));
        assert!(loras_contain_ic_lora(&[json!({ "isIcLora": true })]));
        // conditioningRole (with the `-`/`_` normalisation).
        assert!(loras_contain_ic_lora(&[
            json!({ "conditioningRole": "IC-Lora" })
        ]));
        // Name / id / path markers.
        assert!(loras_contain_ic_lora(&[
            json!({ "name": "LTX-2.3-22b-IC-LoRA-Union-Control" })
        ]));
        assert!(loras_contain_ic_lora(&[
            json!({ "id": "ltx_2_3_ic_union" })
        ]));
        assert!(loras_contain_ic_lora(&[
            json!({ "source": { "files": ["my-ic-lora.safetensors"] } })
        ]));
        // A bare string id.
        assert!(loras_contain_ic_lora(&[json!("some-ic-lora-v2")]));
        // An ordinary LoRA is not an IC-LoRA.
        assert!(!loras_contain_ic_lora(&[
            json!({ "name": "cinematic-style", "path": "/loras/cinematic.safetensors" })
        ]));
        assert!(!loras_contain_ic_lora(&[]));
    }

    /// extend/bridge conditioning fails clearly (before any IO) when no IC-LoRA is installed —
    /// mirrors the torch gate; without the adapter the appended clip tokens are inert.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn video_clip_conditioning_requires_ic_lora() {
        let settings = Settings::from_env();
        let api = ApiClient::new(&settings);
        // The IC-LoRA gate is the resolver's first check, so it returns before touching `job`
        // / the api / disk — a minimal snapshot suffices.
        let job: JobSnapshot = serde_json::from_value(json!({
            "id": "job-extend-1",
            "type": "video_extend",
            "status": "preparing",
            "projectId": "p",
            "projectName": "P",
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": null,
            "progress": 0,
            "stage": "preparing",
            "message": "",
            "error": null,
            "etaSeconds": null,
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-09T00:00:00Z",
            "updatedAt": "2026-06-09T00:00:00Z"
        }))
        .expect("job snapshot");
        let req = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
            "mode": "extend_clip", "sourceClipAssetId": "clip_a",
            "loras": [{ "name": "cinematic-style" }]
        }));
        let err = resolve_video_clip_conditioning(&api, &settings, &job, &req, Path::new("/tmp/p"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("IC-LoRA"), "got: {err}");
    }

    /// An LTX-2.3 snapshot **complete for the current engine** ([`ltx_dir_is_complete`]),
    /// else `None` so the smoke skips. Checks `$SCENEWORKS_MLX_LTX_DIR`, the turnkey SceneWorks
    /// bundle's `q4/`/`q8/` subdirs in the HF cache ([`LTX_BUNDLE_REPO`], `SceneWorks/ltx-2.3-mlx`),
    /// and the app-managed `<data>/models/mlx/*` dirs (which predate the audio+i2v layout, so skip).
    #[cfg(target_os = "macos")]
    fn ltx_complete_dir() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(dir) = std::env::var("SCENEWORKS_MLX_LTX_DIR") {
            candidates.push(PathBuf::from(dir.trim()));
        }
        if let Ok(home) = std::env::var("HOME") {
            let hub = PathBuf::from(&home).join(".cache/huggingface/hub");
            // The turnkey SceneWorks bundle: each snapshot carries `q4/` + `q8/` checkpoint subdirs.
            let snapshots = hub
                .join("models--SceneWorks--ltx-2.3-mlx")
                .join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                for snapshot in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
                    candidates.push(snapshot.join("q4"));
                    candidates.push(snapshot.join("q8"));
                }
            }
            let base =
                PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx");
            for id in [
                "ltx_2_3_base_q4",
                "ltx_2_3_base_q8",
                "ltx_2_3_eros",
                "ltx_2_3",
            ] {
                candidates.push(base.join(id));
            }
        }
        candidates.into_iter().find(|dir| ltx_dir_is_complete(dir))
    }

    /// Real in-process LTX-2.3 T2V **with synchronized audio** through the engine. Loads a
    /// complete converted snapshot + the cached Gemma TE and denoises a tiny 9-frame clip,
    /// asserting frames come back RGB8-sized **and an audio track is produced** with streamed
    /// progress. `#[ignore]` + skips unless a complete snapshot is present (the cached
    /// `ltx_2_3_base_*` dirs predate the engine's vocoder/vae_encoder layout).
    #[cfg(target_os = "macos")]
    #[ignore = "loads the real LTX-2.3 weights + Gemma TE; needs a snapshot complete for the current engine"]
    #[test]
    fn ltx_real_weights_with_audio() {
        let Some(model_dir) = ltx_complete_dir() else {
            eprintln!("skipping ltx_real_weights_with_audio: no complete LTX snapshot found");
            return;
        };
        // The bundle ships gemma beside the q4/q8 dir; thread it onto the LoadSpec (matches the worker
        // path, sc-8827) so the smoke needs no separate mlx-community/gemma snapshot in the HF cache.
        let text_encoder_dir = resolve_bundled_ltx_gemma_dir(&model_dir);
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "ltx_2_3",
            model_dir,
            prompt: "a calm ocean wave at sunset, gentle surf".to_owned(),
            width: 256,
            height: 256,
            frames: 9,
            fps: 24,
            seed: 7,
            text_encoder_dir,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded =
            run_video_generation(input, &cancel, &mut on_progress).expect("LTX A/V generation");
        assert_eq!(decoded.frames.len(), 9, "9 frames (1 + 8·1)");
        let audio = decoded
            .audio
            .expect("LTX produces a synchronized audio track");
        assert!(audio.sample_rate >= 1 && !audio.samples.is_empty());
        assert!(steps > 0, "denoise progress streamed");
        assert!(decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
    }

    /// Full encode → mp4 + poster, exercised against a real ffmpeg. Skips when no
    /// ffmpeg is reachable (SCENEWORKS_FFMPEG or `ffmpeg` on PATH) so it never fails a
    /// host without the binary; CI with ffmpeg runs it for real.
    #[tokio::test]
    async fn encode_stub_to_mp4_with_audio_and_poster() {
        if !ffmpeg_reachable() {
            eprintln!("skipping encode_stub_to_mp4_with_audio_and_poster: ffmpeg not found");
            return;
        }
        let request = request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
            "duration": 1.0, "fps": 9, "width": 128, "height": 128
        }));
        let decoded = generate_stub_video(&request, 11);
        assert!(decoded.audio.is_some());
        let dir = std::env::temp_dir().join(format!("sw_vid_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let media_path = dir.join("clip.mp4");
        encode_media(&media_path, decoded, None).await.unwrap();
        assert!(media_path.exists(), "mp4 must be written");
        assert!(media_path.metadata().unwrap().len() > 0);
        assert!(
            media_path.with_extension("poster.jpg").exists(),
            "poster must be extracted"
        );
        // Intermediates are cleaned up.
        assert!(!media_path.with_extension("frames").exists());
        assert!(!media_path.with_extension("enc.mp4").exists());
        assert!(!media_path.with_extension("audio.wav").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ffmpeg_reachable() -> bool {
        if let Ok(path) = std::env::var("SCENEWORKS_FFMPEG") {
            if !path.trim().is_empty() && Path::new(&path).exists() {
                return true;
            }
        }
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// F-118: video upscale accepts only 2x and 4x. Any other factor is rejected with a clear error
    /// rather than silently coerced to 2x (which produced a quietly-different output). Compiled on
    /// both the macOS and candle lanes, matching the gate on [`resolve_video_upscale_factor`].
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn video_upscale_factor_accepts_2_and_4_rejects_others() {
        assert_eq!(resolve_video_upscale_factor(2).expect("2x"), 2);
        assert_eq!(resolve_video_upscale_factor(4).expect("4x"), 4);
        for bad in [0u8, 1, 3, 5, 8] {
            let err = resolve_video_upscale_factor(bad)
                .expect_err("unsupported factor must be rejected, not coerced");
            assert!(
                matches!(err, WorkerError::InvalidPayload(ref m) if m.contains("factor 2 or 4")),
                "factor {bad} should yield a clear InvalidPayload error, got {err:?}"
            );
        }
    }
}

// Candle video lane labeling + engine-mapping unit tests (sc-5099). Windows/candle-gated; pure maps.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_video_label_tests {
    use super::*;

    #[test]
    fn candle_video_engine_ids_map_5b_ltx_and_14b() {
        assert_eq!(candle_video_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
        // The Wan2.2 14B dual-expert MoE pair (sc-5175): T2V (text→video) + I2V (image→video).
        assert_eq!(
            candle_video_engine_id("wan_2_2_t2v_14b"),
            Some("wan2_2_t2v_14b")
        );
        assert_eq!(
            candle_video_engine_id("wan_2_2_i2v_14b"),
            Some("wan2_2_i2v_14b")
        );
        // ltx maps to the candle distilled id, not the MLX `ltx_2_3`.
        assert_eq!(candle_video_engine_id("ltx_2_3"), Some("ltx_2_3_distilled"));
        // SVD maps to the candle `svd_xt` engine (sc-5493).
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        // eros now shares the one candle LTX-2.3 engine with the base (sc-5495 wired the eros
        // dense checkpoint through `ltx_2_3_distilled`; they differ only in `candle_video_repo`).
        assert_eq!(
            candle_video_engine_id("ltx_2_3_eros"),
            Some("ltx_2_3_distilled")
        );
        assert!(is_candle_video_engine("ltx_2_3_eros"));
        for model in [
            "wan_2_2",
            "wan_2_2_t2v_14b",
            "wan_2_2_i2v_14b",
            "ltx_2_3",
            "ltx_2_3_eros",
            "svd",
        ] {
            assert!(is_candle_video_engine(model), "{model}");
        }
    }

    #[test]
    fn candle_video_adapter_labels_are_per_family() {
        // Every wan engine (5B + 14B T2V/I2V) reports the shared `candle_wan` adapter.
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            assert_eq!(
                candle_video_adapter_label(engine_id),
                "candle_wan",
                "{engine_id}"
            );
        }
        assert_eq!(
            candle_video_adapter_label("ltx_2_3_distilled"),
            "candle_ltx"
        );
        assert_eq!(candle_video_adapter_label("svd_xt"), "candle_svd");
        // Mochi (sc-11992) must have its OWN arm. The `_` fall-through is the Wan default, so a
        // missing arm silently stamps `candle_wan` onto a Mochi asset's sidecar + telemetry — wrong
        // provenance with no error anywhere.
        assert_eq!(candle_video_adapter_label("mochi_1"), "candle_mochi");
        assert_ne!(
            candle_video_adapter_label("mochi_1"),
            CANDLE_WAN_ADAPTER,
            "mochi_1 must not fall through to the Wan adapter label"
        );
    }

    /// **The candle-lane routing trap** (sc-11992). B1 declared Windows served
    /// (`VideoModelCaps::new("mochi_1", true, true, false, false)` ⇒ `candle_video_routed`), and A6
    /// validated the lane on Blackwell against the very same hosted tiers. If `candle_video_engine_id`
    /// does not resolve `mochi_1`, `is_candle_video_engine` is false, `resolve_candle_video_route`
    /// never reaches its generic arm, and the job lands on `CandleVideoRoute::Stub` — handing the user
    /// a PROCEDURAL FAKE VIDEO rather than an error. That is a silent wrong-output bug, so it is
    /// pinned here rather than left to the descriptor.
    #[test]
    fn candle_mochi_resolves_the_engine_and_never_falls_to_the_stub() {
        // ONE engine id on BOTH backends — no `_distilled`-style split (contrast ltx above).
        assert_eq!(candle_video_engine_id("mochi_1"), Some("mochi_1"));
        assert!(
            is_candle_video_engine("mochi_1"),
            "mochi_1 MUST be a candle video engine or Windows silently renders a procedural stub"
        );

        // The generic candle arm is reached for a t2v mochi job (the shape B1's router allows
        // through: text_to_video, no source asset, no LoRA).
        let settings = Settings {
            backend_candle_enabled: true,
            ..Settings::from_env()
        };
        let payload = json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        });
        let req = VideoRequest::from_payload(payload.as_object().expect("object"));
        assert_eq!(
            resolve_candle_video_route(&req, &settings),
            CandleVideoRoute::CandleVideo,
            "a t2v mochi job must route to the candle video engine, NOT the stub"
        );
    }

    /// The candle default repo must be Mochi's own. No Mochi manifest entry carries a top-level
    /// `repo`, so `candle_video_repo` falls through to this default — and the `_` arm is Wan's, which
    /// would send the loader at a Wan diffusers snapshot.
    #[test]
    fn candle_mochi_default_repo_is_the_shared_mochi_turnkey() {
        assert_eq!(
            candle_video_default_repo("mochi_1"),
            "SceneWorks/mochi-1-mlx"
        );
        assert_ne!(
            candle_video_default_repo("mochi_1"),
            CANDLE_WAN_5B_REPO,
            "mochi_1 must not inherit Wan's default repo"
        );
        // ONE repo serves BOTH backends — the candle default IS the MLX repo (A6's `.scales`-detect
        // seam ingests the mlx-affine tiers 1:1; `SceneWorks/mochi-1-candle` was never published).
        assert_eq!(candle_video_default_repo("mochi_1"), MOCHI_REPO);

        // With no manifest `repo` (the shipped Mochi entry), the resolved repo is that default.
        let payload = json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        });
        let req = VideoRequest::from_payload(payload.as_object().expect("object"));
        assert_eq!(candle_video_repo(&req, "mochi_1"), "SceneWorks/mochi-1-mlx");
    }

    #[test]
    fn candle_video_default_repos_are_per_engine() {
        assert_eq!(
            candle_video_default_repo("wan2_2_ti2v_5b"),
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_t2v_14b"),
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_i2v_14b"),
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("ltx_2_3_distilled"),
            "Lightricks/LTX-2.3"
        );
        // SVD-XT loads the stock diffusers img2vid-xt snapshot directly (sc-5493).
        assert_eq!(
            candle_video_default_repo("svd_xt"),
            "stabilityai/stable-video-diffusion-img2vid-xt"
        );
    }

    #[test]
    fn candle_video_engine_ids_map_svd() {
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        assert!(is_candle_video_engine("svd"));
    }

    #[test]
    fn candle_video_conditioning_only_for_i2v() {
        let settings = crate::Settings::from_env();
        let project_path = std::path::Path::new("");
        // txt2video engines never build conditioning (even if a stray source asset is present).
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "ltx_2_3_distilled"] {
            let payload = json!({ "sourceAssetId": "asset_1" });
            let request = VideoRequest::from_payload(payload.as_object().expect("object"));
            let conditioning =
                resolve_candle_video_conditioning(&settings, &request, project_path, engine_id)
                    .expect("txt2video conditioning resolves");
            assert!(
                conditioning.is_empty(),
                "{engine_id} must be txt2video-only"
            );
        }
        // The 14B I2V + SVD-XT require a source image — a request without one errors before touching
        // disk (sc-5175 / sc-5493).
        for engine_id in ["wan2_2_i2v_14b", "svd_xt"] {
            let payload = json!({ "mode": "image_to_video" });
            let no_source = VideoRequest::from_payload(payload.as_object().expect("object"));
            assert!(
                resolve_candle_video_conditioning(&settings, &no_source, project_path, engine_id)
                    .is_err(),
                "{engine_id} without a source image must error"
            );
        }
    }

    /// The candle Wan Lightning toggle (sc-10138) drives the sampling recipe: A14B MoE default-on
    /// (absent flag) ⇒ 4-step / CFG-off; explicit opt-out ⇒ native multi-step CFG (engine defaults, or a
    /// user override); the dense 5B has no toggle and applies the interim step default.
    #[test]
    fn candle_wan_lightning_toggle_drives_sampling() {
        let req = |advanced: Value| {
            VideoRequest::from_payload(json!({ "advanced": advanced }).as_object().unwrap())
        };

        for moe in ["wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            // Default-on (absent flag): the 4-step / CFG-off Lightning recipe.
            let on = req(json!({}));
            assert!(candle_wan_lightning_on(moe, &on), "{moe} default-on");
            assert_eq!(
                candle_wan_sampling(moe, &on),
                (Some(4), Some(1.0)),
                "{moe} lightning recipe"
            );

            // Explicit opt-out → native multi-step CFG: no user override ⇒ (None, None) so the engine
            // config defaults stand; a user override is honored verbatim.
            let off = req(json!({ "lightning": false }));
            assert!(!candle_wan_lightning_on(moe, &off), "{moe} opt-out");
            assert_eq!(
                candle_wan_sampling(moe, &off),
                (None, None),
                "{moe} native defaults"
            );
            let off_override =
                req(json!({ "lightning": false, "steps": 30, "guidanceScale": 3.5 }));
            assert_eq!(
                candle_wan_sampling(moe, &off_override),
                (Some(30), Some(3.5))
            );
        }

        // Dense TI2V-5B: no Lightning toggle (always off), interim step default, user override wins.
        let five_b = req(json!({}));
        assert!(!candle_wan_lightning_on("wan2_2_ti2v_5b", &five_b));
        assert_eq!(
            candle_wan_sampling("wan2_2_ti2v_5b", &five_b),
            (Some(CANDLE_WAN5B_INTERIM_STEPS), None)
        );
        assert_eq!(
            candle_wan_sampling("wan2_2_ti2v_5b", &req(json!({ "steps": 12 }))).0,
            Some(12)
        );
    }

    /// (sc-10027 / sc-10726) `candle_wan_tier_subdir` picks the q4/q8 subdir per `advanced.mlxQuantize`
    /// (default q8 when installed, clamped to q4), falls back through the tier order, requires a complete
    /// tier, and returns `None` for a non-wan engine or a flat/dense repo with no tier subdirs.
    #[test]
    fn candle_wan_tier_subdir_resolves_by_mlx_quantize() {
        let root = std::env::temp_dir().join(format!("sc10027_wan_tier_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Complete q4 + q8 A14B tiers (transformer + transformer_2 + text_encoder + vae + tokenizer).
        for tier in ["q4", "q8"] {
            for sub in [
                "transformer",
                "transformer_2",
                "text_encoder",
                "vae",
                "tokenizer",
            ] {
                std::fs::create_dir_all(root.join(tier).join(sub)).unwrap();
            }
        }
        let req = |adv: Value| {
            VideoRequest::from_payload(json!({ "advanced": adv }).as_object().unwrap())
        };

        // Default (no mlxQuantize) → q8 when installed (sc-10726), clamped to what's on disk.
        let (d, q) = candle_wan_tier_subdir(&root, "wan2_2_t2v_14b", &req(json!({}))).unwrap();
        assert!(d.ends_with("q8"));
        assert_eq!(q, Some(Quant::Q8));
        // An explicit Q4 pick stays q4 (never overridden by the q8 default).
        let (d, q) =
            candle_wan_tier_subdir(&root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 4 })))
                .unwrap();
        assert!(d.ends_with("q4"));
        assert_eq!(q, Some(Quant::Q4));
        // mlxQuantize = 8 → q8.
        let (d, q) =
            candle_wan_tier_subdir(&root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 8 })))
                .unwrap();
        assert!(d.ends_with("q8"));
        assert_eq!(q, Some(Quant::Q8));
        // bf16 requested but no bf16 tier present → falls back through q8/q4 (never a missing dir).
        let (d, _) =
            candle_wan_tier_subdir(&root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 0 })))
                .unwrap();
        assert!(d.ends_with("q8") || d.ends_with("q4"));
        // Non-wan engine → None (ltx loads its flat snapshot).
        assert!(candle_wan_tier_subdir(&root, "ltx_2_3_distilled", &req(json!({}))).is_none());
        // A flat repo (the dense Wan-AI/*-Diffusers fallback — no tier subdirs) → None.
        let flat = std::env::temp_dir().join(format!("sc10027_flat_{}", std::process::id()));
        std::fs::create_dir_all(&flat).unwrap();
        assert!(candle_wan_tier_subdir(&flat, "wan2_2_t2v_14b", &req(json!({}))).is_none());

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&flat).ok();
    }
}
