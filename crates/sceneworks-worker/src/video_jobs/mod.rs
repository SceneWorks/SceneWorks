//! Native video-generation jobs — runtime pipeline + procedural stub (epic 3018, sc-3033).
//!
//! Parses the job into a [`VideoRequest`], produces a single video (one mp4 asset,
//! unlike images which batch `count`), and reports a flat "fact" the Rust API turns
//! into an indexed asset (mirroring `video_generation_result` in the Python worker's
//! `video_adapters.py`). The shared encode pipeline takes the engine's video output
//! shape — RGB8 `frames` + `fps` + an optional synchronized `audio` track — writes
//! the frames to an mp4 (libx264), muxes a 16-bit-PCM WAV as AAC when audio is present
//! (bounded at the PICTURE's length — see [`audio_mux_args`]), remuxes `+faststart`
//! (WKWebView range-seek), and extracts a poster frame. It reuses
//! [`crate::media_jobs::run_ffmpeg`] (binary resolution + the periodic-heartbeat /
//! cooperative-cancel loop).
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
    duration_limit_error, fps_limit_error, is_ltx_model, reference_limit_error, requested_steps,
    steps_limit_error, VideoRequest,
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
use crate::media_jobs::{run_ffmpeg, run_ffmpeg_with_stdin_chunks, FfmpegContext};

/// Shared worker-level dependencies used by the engine-family modules. Keeping this
/// list explicit prevents a family from silently inheriting newly imported parent
/// names or flattening another engine family's namespace.
mod prelude {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[allow(unused_imports)]
    pub(super) use super::load_reference_image;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle"),
        test
    ))]
    #[allow(unused_imports)]
    pub(super) use super::ltx_frame_count;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[allow(unused_imports)]
    pub(super) use super::scail2::scail2_segment_blocking;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[allow(unused_imports)]
    pub(super) use super::{
        advanced, assemble_scail2_animate_conditioning, lora_scale, non_empty_negative_prompt,
        resolve_dense_adapters, resolve_lora_file, video_frame_count, wan_frame_count, AdapterSpec,
        CancelFlag, CharacterStore, Conditioning, GenerationMetrics, GenerationOutput,
        GenerationRequest, Generator, Image, LoadPhase, LoadSpec, OffloadPolicy, Precision,
        Progress, Quant, ReplacementMode, WeightsSource, MAX_JOB_LORAS,
    };
    #[allow(unused_imports)]
    pub(super) use super::{
        backend_label, cancel_requested_peek, check_cancel, faststart_mp4, fresh_asset_id,
        heartbeat, huggingface_snapshot_dir, json, now_rfc3339, picture_bound_seconds,
        resolve_video_seed, run_ffmpeg, safe_download_dir, shutdown_requested, task_join_error,
        update_job, video_progress, write_poster_frame, ApiClient, AudioTrack, BTreeMap,
        CancelJoinGuard, DecodedVideo, Duration, FfmpegContext, Instant, JobSnapshot, JobStatus,
        JsonObject, Path, PathBuf, ProgressStage, ProjectStore, RgbFrame, Settings, Uuid, Value,
        VideoRequest, WorkerError, WorkerResult, WorkerStatus, CANCEL_MESSAGE,
    };
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[allow(unused_imports)]
    pub(super) use super::{classify_adapter, lora_path};
    #[cfg(target_os = "macos")]
    #[allow(unused_imports)]
    pub(super) use super::{resolve_mlx_dense_quant, AdapterKind, MoeExpert};
}

// Real MLX Wan2.2 generation (macOS, sc-3034). `runtime-macos` explicitly includes all three Wan
// registrations in its validated media catalog.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::image_jobs::load_reference_image;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::image_jobs::{classify_adapter, lora_path};
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
    LoadPhase, LoadSpec, OffloadPolicy, Precision, Progress, Quant, WeightsSource,
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
// `ltx_frame_count` is used by both native LTX conditioning lanes and by cross-platform tests.
// `mochi_frame_count` is named only by tests that are themselves macOS/candle-gated.
// Each is gated to exactly the configs that use it: an import left dead on a single cfg fails the
// parity lane's `-D warnings` while a macOS check stays green (sc-10404's trap).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle"),
    test
))]
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
    /// Actual provider-owned adapter install outcomes from this generation. Empty for providers that
    /// do not expose partial-install reports.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    adapter_apply_reports: Vec<gen_core::AdapterApplyReport>,
}

/// One RGB8 frame, row-major, `pixels.len() == width * height * 3` (the engine's
/// `Image`).
struct RgbFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Interleaved PCM audio — the engine's `AudioTrack` (LTX-2.3 synchronized audio). `pub(crate)` so
/// the pure-audio job path (`audio_jobs`, sc-13404) reuses the same shape + the `write_wav_pcm16`
/// writer below for its Kokoro output rather than duplicating the WAV plumbing.
pub(crate) struct AudioTrack {
    pub(crate) samples: Vec<f32>,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
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
    /// Krea Realtime 14B autoregressive video (epic 8431 / sc-8443). t2v + i2v (`Reference`) + v2v
    /// (`VideoClip`); mac-only. Carries the resolved engine id.
    KreaRealtime(&'static str),
    /// Mochi 1 text-to-video (epic 1788 / sc-11992). Carries the resolved engine id. t2v ONLY —
    /// [`mochi_available`] gates the mode, since `conditioning: []` means there is no other shape.
    Mochi(&'static str),
    /// MiniMax-H3 / Hailuo 3.0 joint audio+video (epic 17137 / sc-19508). Serves t2va, fl2va (0/1/2
    /// keyframes) and Ref2VA; mac-only. Carries the resolved engine id — ONE id for BOTH catalog
    /// entries, because the two DiT partitions are directories of a single provider, not two
    /// providers. [`minimax_h3_available`] folds in the registry check, the partition/shape
    /// agreement check and weights resolution.
    MiniMaxH3(&'static str),
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
    // MiniMax-H3 readiness is the ONE ladder input that cannot be reached from a test today: it
    // requires the engine to be REGISTERED in the linked inference bundle, which is false at the
    // pinned revision (sc-18650 owns the bump). Left inline, the tail arm below would be dead code
    // no test can distinguish from a deleted one — declaration without reachability, the exact trap
    // this epic keeps hitting. Threaded in instead, so `resolve_video_route_with` can be driven
    // with the readiness the pin bump will supply and the arm is covered NOW.
    //
    // Evaluated here rather than in the ladder, but NOT eagerly: `minimax_h3_engine_id` is a pure
    // string check, so every other model short-circuits before any filesystem touch and routing
    // stays byte-identical. For a MiniMax-H3 id the probe simply runs earlier than it would have —
    // no earlier predicate can match those ids, so the outcome is unchanged.
    let minimax_h3_ready =
        minimax_h3_engine_id(&request.model).is_some() && minimax_h3_available(request, settings);
    resolve_video_route_with(request, settings, minimax_h3_ready)
}

/// [`resolve_video_route`] with the MiniMax-H3 readiness probe supplied by the caller — the seam
/// that makes the family's tail arm reachable before the inference pin moves (sc-19508).
///
/// `resolve_video_route` is the one-line delegation that binds the real probe. Everything the
/// ladder decides lives here.
#[cfg(target_os = "macos")]
fn resolve_video_route_with(
    request: &VideoRequest,
    settings: &Settings,
    minimax_h3_ready: bool,
) -> VideoRoute {
    // Lives at the top of `_with`, not in the thin `resolve_video_route` wrapper: the wrapper is a
    // one-line delegation and every routing decision belongs on the seam tests drive directly.
    if request.model == "wan_2_2_vace_fun_14b" && request.mode != "replace_person" {
        return VideoRoute::Stub;
    }
    if request.mode == "replace_person" {
        if let Some(engine_id) = scail2_engine_id(&request.model) {
            VideoRoute::ReplacePersonScail2(engine_id)
        } else if let Some(engine_id) = ltx_engine_id(&request.model) {
            VideoRoute::Ltx(engine_id)
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
    } else if let Some(engine_id) = krea_realtime_engine_id(&request.model)
        .filter(|_| krea_realtime_available(request, settings))
    {
        // Krea Realtime 14B (epic 8431 / sc-8443 S10). `krea_realtime_engine_id` matches only
        // `krea_realtime_14b`, and no earlier predicate can match that id, so routing for every
        // pre-existing model stays byte-identical. Weights unresolved ⇒ falls to Stub, whose fail-loud
        // gate refuses rather than handing back a procedural fake clip.
        VideoRoute::KreaRealtime(engine_id)
    } else if let Some(engine_id) =
        mochi_engine_id(&request.model).filter(|_| mochi_available(request, settings))
    {
        // Mochi 1 (epic 1788 / sc-11992). Appended at the TAIL of the ladder: `mochi_engine_id`
        // matches only `mochi_1`, and no earlier predicate can match that id, so routing for every
        // pre-existing model stays byte-identical. `mochi_available` folds in the t2v-only mode gate.
        VideoRoute::Mochi(engine_id)
    } else if let Some(engine_id) =
        minimax_h3_engine_id(&request.model).filter(|_| minimax_h3_ready)
    {
        // MiniMax-H3 (epic 17137 / sc-19508). Appended at the NEW tail: `minimax_h3_engine_id`
        // matches only `minimax_h3` / `minimax_h3_ref`, and no earlier predicate can match either
        // id, so routing for every pre-existing model stays byte-identical.
        //
        // `minimax_h3_available` folds in three gates that all have to hold before an arm may run:
        // the engine is actually REGISTERED in the linked inference bundle (false at the current
        // pin, and the reason this arm is dormant rather than broken), the conditioning shape
        // agrees with the entry's DiT partition, and both the tier and its shared components
        // resolve. Any of them failing drops to `Stub`, where `ensure_video_engine_weights` re-runs
        // the same checks and surfaces the precise reason instead of a procedural fake clip.
        VideoRoute::MiniMaxH3(engine_id)
    } else {
        VideoRoute::Stub
    }
}

/// The candle (Windows/CUDA/Linux) video engine a `run_video_generate_job` request routes to — the
/// candle-lane sibling of [`VideoRoute`] (sc-8828, F-026). The Eros terminal refusal precedes the
/// backend flag so even a directly invoked legacy job cannot produce a stub; every supported arm is
/// gated on `settings.backend_candle_enabled`, and returns [`CandleVideoRoute::Stub`] when disabled.
/// The ladder preserves each accepted model-native conditioned provider: base LTX owns I2V, FLF,
/// extend, bridge, and replacement;
/// Wan TI2V-5B owns I2V/FLF; VACE-Fun owns its dedicated dual-expert replacement route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandleVideoRoute {
    /// `ltx_2_3_eros` is MLX-only after its undistilled Candle route failed exact-head CUDA
    /// acceptance (sc-18902). Keep a worker-side terminal backstop for replayed/legacy jobs that
    /// bypass the current scheduler refusal; they must never become procedural stub output.
    UnsupportedEros,
    /// `replace_person` on a `scail2_*` model → candle SCAIL-2 replacement (sc-6837). Carries the id.
    ReplacePersonScail2(&'static str),
    /// `replace_person` on `wan_2_2_vace_fun_14b` → the dedicated dual-expert Candle engine.
    ReplacePersonWanVaceFun,
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
    /// MiniMax-H3's joint audio/video Candle provider. This remains unreachable through the
    /// scheduler until the separately-owned `candle_video_routed` capability flip, but direct or
    /// replayed off-Mac jobs must use the real provider rather than procedural stub output.
    MiniMaxH3(&'static str),
    /// An in-place ComfyUI Wan2.2 base model (`external_base_*`) → `generate_candle_wan_comfyui`
    /// (epic 10451 Phase 2c, sc-10671). Not an `is_candle_video_engine` id — routed off the forwarded row.
    WanComfyui,
    /// Candle disabled, or no candle engine matched → the procedural stub.
    Stub,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn reject_unsupported_candle_video_route(route: CandleVideoRoute) -> Result<(), WorkerError> {
    match route {
        CandleVideoRoute::UnsupportedEros => Err(WorkerError::InvalidPayload(
            "LTX-2.3 10Eros is not supported on Candle/CUDA: its validated recipe requires the \
             MLX two-pass cond_safe distill adapter, while the undistilled Candle route produced \
             unusable noise in SC-18902 acceptance. Use an Apple Silicon MLX worker or select the \
             base LTX-2.3 model."
                .to_owned(),
        )),
        _ => Ok(()),
    }
}

/// Run the candle video dispatch predicate ladder ONCE and return the [`CandleVideoRoute`]. Mirrors the
/// historical inline ladder EXACTLY — same predicate order + `backend_candle_enabled` gating — so
/// routing is byte-identical (sc-8828).
///
/// # Why this is fallible (sc-20644 review blocker 2)
///
/// The ladder's terminal arm is [`CandleVideoRoute::Stub`], which COMPLETES a job with procedural
/// video. That is the right answer for a model this backend does not serve, and the wrong answer for
/// a lane that had a typed refusal to give: `wan_comfyui_available` used to answer a `bool`, so
/// every refusal the Wan plan route raises — a drifted or tampered plan, a missing snapshot tier, a
/// missing or ambiguous expert role, an unconsumed layer — turned into `false` and the job SUCCEEDED
/// with stub output.
///
/// This is the video lane's version of what `CheckpointPlanSelection::into_unclaimed_refusal` does
/// on the image side: a lane's refusal is retained rather than discarded, and reaching the end of
/// the ladder re-raises it instead of stubbing. Every other arm is unchanged and still infallible;
/// only a lane that can REFUSE participates.
/// Test-facing accessor for [`resolve_candle_video_route`], so a lane's own test module can assert
/// that its refusal reaches the ROUTER rather than only its resolver. The distinction matters: a
/// refusal that stops at the resolver and is swallowed on the way out is exactly the defect
/// (review blocker 2).
/// Returns only whether the route RESOLVED, not which route, so the accessor does not have to leak
/// the private `CandleVideoRoute` out of this module. The distinction the caller needs is exactly
/// "did this refuse or did it fall through", and the refusal message is the interesting half.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_video_route_for_test(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<()> {
    resolve_candle_video_route(request, settings).map(|_| ())
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_video_route(
    request: &VideoRequest,
    settings: &Settings,
) -> WorkerResult<CandleVideoRoute> {
    // This precedes the backend gate and every mode arm deliberately. A replayed Eros job must fail
    // even on a disabled Candle worker, never misroute to Wan-VACE or procedural stub output.
    if request.model == "ltx_2_3_eros" {
        return Ok(CandleVideoRoute::UnsupportedEros);
    }
    // A plan-backed entry is decided HERE, ahead of the backend gate and every mode arm — each of
    // those is wrong for it in its own way, and exactly one candle video lane loads a plan. See
    // [`candle::plan_backed_wan_video_route`], which claims that lane or refuses by name (sc-20651).
    if let Some(checkpoint_id) =
        sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(&request.model_manifest_entry)
    {
        candle::plan_backed_wan_video_route(request, settings, checkpoint_id)?;
        return Ok(CandleVideoRoute::WanComfyui);
    }
    if !settings.backend_candle_enabled {
        return Ok(CandleVideoRoute::Stub);
    }
    if request.model == "wan_2_2_vace_fun_14b" && request.mode != "replace_person" {
        return Ok(CandleVideoRoute::Stub);
    }
    Ok(if request.mode == "replace_person" {
        match scail2_engine_id(&request.model) {
            Some(engine_id) => CandleVideoRoute::ReplacePersonScail2(engine_id),
            None if request.model == "wan_2_2_vace_fun_14b" => {
                CandleVideoRoute::ReplacePersonWanVaceFun
            }
            None if candle::candle_video_engine_id(&request.model) == Some("ltx_2_3_distilled") => {
                CandleVideoRoute::CandleVideo
            }
            None => CandleVideoRoute::ReplacePersonWanVace,
        }
    } else if request.mode == "animate_character" && scail2_engine_id(&request.model).is_some() {
        let engine_id = scail2_engine_id(&request.model).expect("scail2 model");
        CandleVideoRoute::AnimateScail2(engine_id)
    } else if matches!(request.mode.as_str(), "extend_clip" | "video_bridge")
        && candle::candle_video_engine_id(&request.model) == Some("ltx_2_3_distilled")
    {
        CandleVideoRoute::CandleVideo
    } else if matches!(request.mode.as_str(), "extend_clip" | "video_bridge") {
        CandleVideoRoute::WanVaceExtendBridge
    } else if let Some(engine_id) = bernini_engine_id(&request.model) {
        // Bernini (sc-10997, epic 6562): the full Qwen planner + Wan2.2-T2V-A14B renderer serves t2v +
        // the editing/reference/multi-source modes (v2v / r2v / rv2v / mv2v / ads2v). A DISTINCT engine
        // (`crate::inference_runtime::load("bernini")`), NOT a wan/ltx `is_candle_video_engine` id, so it is routed off the
        // model id BEFORE the generic candle-video arm below. Routed by model id, not weight availability —
        // `generate_candle_bernini` resolves-or-errors loudly if the `SceneWorks/bernini` snapshot is
        // unprovisioned (sc-11003), never degrading to a stub. The per-mode source media is validated when
        // the conditioning is assembled (`resolve_candle_bernini_conditioning`), mirroring the MLX lane.
        CandleVideoRoute::Bernini(engine_id)
    } else if wan_comfyui_claims(request, settings)? {
        // In-place ComfyUI Wan2.2 base (sc-10671): an `external_base_*` id, or a plan-backed entry
        // (sc-20644), so it matches no `is_candle_video_engine` arm below — route it off the
        // forwarded `modelManifestEntry`. `?` rather than a bool: a refusal this lane raises must
        // reach the job as a typed error, never fall past here into `Stub` and COMPLETE as
        // procedural video (review blocker 2).
        CandleVideoRoute::WanComfyui
    } else if let Some(engine_id) = minimax_h3_engine_id(&request.model) {
        CandleVideoRoute::MiniMaxH3(engine_id)
    } else if is_candle_video_engine(&request.model) {
        CandleVideoRoute::CandleVideo
    } else {
        CandleVideoRoute::Stub
    })
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
    // The model's declared `limits.hardMinSteps` and `limits.steps` (sc-19426, sc-19502 — one call,
    // because `steps_limit_error` owns both). Bounds the SAMPLING axis, which neither
    // bound above touches: `frames = duration × fps` says nothing about how many denoise steps run
    // over those frames, so a request legal on both can still hand the engine a schedule it cannot
    // build (LTX-2.3 bakes an 8-step sigma table and refuses anything else) or one that is legal but
    // useless (MiniMax-H3 at 1 evaluation is a single Euler jump from pure noise — see that entry's
    // `hardMinSteps` note for why its floor is a product judgement rather than an engine limit).
    //
    // The unit read here is MODEL EVALUATIONS, matching `advanced.steps`, `defaults.steps` and each
    // turbo adapter's declared `sampling.steps`. No ±1 is applied anywhere on this side; the
    // MiniMax-H3 engine appends its own terminal sigma grid point (sc-18726).
    //
    // The API gate is the one that returns a caller a 400, but it only covers what IT enqueues — a
    // job replayed from a pre-sc-19426 row, or produced by any future non-HTTP path, reaches the
    // engine through here. Read through `requested_steps`, the same reader `advanced_opt_u32` is,
    // so this judges exactly the number the adapters go on to pass down.
    if let Some(steps) = requested_steps(&request.advanced) {
        if let Some(message) =
            steps_limit_error(&request.model, steps, &request.model_manifest_entry)
        {
            return Err(WorkerError::InvalidPayload(message));
        }
    }
    // The model's declared reference-media caps (sc-17160). Bounds a THIRD axis the two above do
    // not touch: how much conditioning media the engine is handed. The API gate is the one that
    // returns a caller a 400, but it only covers what IT enqueues — a job replayed from a
    // pre-sc-17160 row, or produced by any future non-HTTP path, reaches the engine through here.
    //
    // It matters most for the audio references, whose default cap is 0: an engine handed
    // `Conditioning::ReferenceAudio` it does not consume renders unconditioned, and an
    // unconditioned render is invisible in the output rather than an error (epic 1788). Refusing
    // at the funnel is what keeps "this model takes no audio references" from degrading into
    // "this model quietly ignored them".
    if let Some(message) = reference_limit_error(
        &request.model,
        request.reference_asset_ids.len(),
        request.source_clip_asset_ids.len(),
        request.reference_audio_asset_ids.len(),
        &request.model_manifest_entry,
    ) {
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
    // Prove the audio references resolve BEFORE the job is marked Running — the same posture as
    // `resolve_voice_clone_plan`, which resolves its reference up front so a missing or
    // undecodable clip fails in the first second rather than deep inside a render that has
    // already cost minutes of GPU time. This runs the whole path: project-scoped asset lookup,
    // the `safe_project_path` guard, the ffmpeg normalization onto the engine's rate, the WAV
    // decode, and the `Conditioning::ReferenceAudio` construction.
    //
    // The vector is discarded here because the MiniMax-H3 arm re-resolves it inside
    // `resolve_minimax_h3_conditioning`, which is where the value is actually consumed. For every
    // already-shipped model the list is empty (their `limits.maxReferenceAudioAssets` defaults to
    // 0, so `video_preflight` above has already refused a non-empty one) and this is a no-op that
    // touches no disk and spawns no process. Only a Ref2VA payload pays for it twice, and one
    // ffmpeg transcode of a reference clip is milliseconds against a render measured in minutes —
    // which is the trade the early refusal is worth.
    resolve_reference_audio_conditioning(api, settings, job, &request, &project_path).await?;
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
    // `wan2_2_vace_fun_14b`. macOS is served natively via MLX and Windows/Linux via Candle. A worker
    // binary built without either native backend must fail honestly here — it must NEVER fall
    // through to the Wan2.1 `generate_wan_vace` backend, which would silently render with the wrong
    // checkpoint (the exact failure the epic forbids).
    #[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
    if request.model == "wan_2_2_vace_fun_14b" {
        return Err(WorkerError::InvalidPayload(
            "wan_2_2_vace_fun_14b requires the native MLX worker on macOS or a worker built with \
             Candle backend support on Windows/Linux. This worker has neither backend, and the job \
             will not be routed to the incompatible Wan2.1 VACE engine."
                .to_owned(),
        ));
    }
    // Krea Realtime 14B (epic 8431 / sc-8443) is `mac_only` (descriptor) and has NO candle engine, so
    // an off-Mac job must fail honestly here rather than fall through to the candle stub / a different
    // backend — the same silent-degradation trap the VACE-Fun guard above closes. The full candle port
    // is out of this story's scope.
    #[cfg(not(target_os = "macos"))]
    if request.model == "krea_realtime_14b" {
        return Err(WorkerError::InvalidPayload(
            "krea_realtime_14b: the native Krea Realtime engine is macOS-only (there is no candle \
             backend). The job will not be routed to a different backend. Choose another model on \
             this platform."
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
                let (decoded, status, lightning) = generate_scail2_replace(
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
            VideoRoute::Ltx(engine_id) => {
                let (decoded, status) = generate_ltx(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (decoded, LTX_ADAPTER, ltx_raw_settings(&request), status)
            }
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
            VideoRoute::KreaRealtime(engine_id) => {
                // Krea Realtime 14B (epic 8431 / sc-8443 S10): autoregressive self-forcing video whose
                // shipped checkpoint is Wan 2.1 T2V 14B weight-for-weight. `generate_krea_realtime` maps
                // the supplied media to the engine conditioning (a reference still → i2v `Reference`,
                // strength no-op; a source clip → v2v `VideoClip`, strength honored; else t2v) and drives
                // the shared `generate_video` heartbeat funnel. CFG off; Wan-family LoRAs ride
                // `LoadSpec::adapters` (sc-15017 S15). It returns its own `rawSettings` because the tier
                // that actually LOADED (sc-15258) and any partial LoRA application are only knowable
                // inside the arm — so the record names what actually ran, not what the request asked for.
                let (decoded, raw_settings) = generate_krea_realtime(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (decoded, KREA_REALTIME_ADAPTER, raw_settings, None)
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
            VideoRoute::MiniMaxH3(engine_id) => {
                // MiniMax-H3 (epic 17137 / sc-19508): joint audio+video in ONE denoise pass.
                // `generate_minimax_h3` maps the supplied media to the engine conditioning — 0/1/2
                // keyframes for t2va/fl2va, ordered image + audio references for Ref2VA — resolves
                // the tier's DiT partition and the upstream shared components (two different
                // repos), and drives the shared `generate_video` funnel. The synchronized stereo
                // soundtrack rides `DecodedVideo::audio` out of the funnel exactly as LTX's does;
                // no bespoke audio plumbing.
                //
                // It returns its own `rawSettings` because the tier that actually LOADED and the
                // task that actually DENOISED — i.e. which of the two structurally-identical
                // 18.78 GB DiT partitions ran — are only knowable inside the arm, and a
                // wrong-partition render is invisible in the output.
                let (decoded, raw_settings) = generate_minimax_h3(
                    api,
                    settings,
                    job,
                    &request,
                    &project_path,
                    engine_id,
                    backend,
                )
                .await?;
                (decoded, MINIMAX_H3_ADAPTER, raw_settings, None)
            }
            VideoRoute::Stub => {
                // An MLX-routed video model whose snapshot didn't resolve must fail
                // loudly with the resolver's precise error instead of completing with
                // procedural stub output (sc-4176, epic 3482 "unsupported jobs error
                // loudly"). replace_person above already follows this rule; the stub
                // remains only for ids outside the engine families (test models,
                // not-yet-ported families).
                wan::ensure_video_engine_weights(&request, settings)?;
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
    let (decoded, adapter, raw_settings, replacement_status) = {
        let candle_route = resolve_candle_video_route(&request, settings)?;
        reject_unsupported_candle_video_route(candle_route)?;
        match candle_route {
            // The rejection above is the execution backstop. Keep this arm unreachable so adding a
            // new match site cannot accidentally turn the unsupported route into successful output.
            CandleVideoRoute::UnsupportedEros => {
                unreachable!("unsupported Eros route was rejected")
            }
            // sc-6837 (epic 6563): SCAIL-2 is a distinct cross-identity replacement backend (NOT Wan-VACE)
            // behind the same person-track pipeline. A `scail2_14b` replace_person job routes to the candle
            // SCAIL-2 engine (the character reference + the tracked person's color masks, `replace_flag`),
            // mirroring the macOS `generate_scail2_replace` dispatch; every other replace-capable model
            // keeps candle Wan-VACE. Routed by model id, not weight availability — `generate_candle_scail2_
            // replace` resolves-or-errors loudly (a person-replace must never silently degrade to a stub).
            CandleVideoRoute::ReplacePersonScail2(engine_id) => {
                let (decoded, status, lightning) = generate_candle_scail2_replace(
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
                    scail2_raw_settings(&request, lightning),
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
            CandleVideoRoute::ReplacePersonWanVaceFun => {
                let (decoded, status) = generate_candle_wan_vace_fun(
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
                    CANDLE_WAN_VACE_FUN_ADAPTER,
                    wan_vace_raw_settings(&request, "wan2_2_vace_fun_14b"),
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
                    bernini_raw_settings(&request),
                    None::<Value>,
                )
            }
            CandleVideoRoute::CandleVideo => {
                let (decoded, adapter, raw_settings, status) =
                    generate_candle_video(api, settings, job, &request, &project_path, backend)
                        .await?;
                (decoded, adapter, raw_settings, status)
            }
            CandleVideoRoute::MiniMaxH3(engine_id) => {
                let (decoded, raw_settings) = generate_candle_minimax_h3(
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
                    CANDLE_MINIMAX_H3_ADAPTER,
                    raw_settings,
                    None::<Value>,
                )
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
        }
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
    // sc-15956: the sanitized workflow, written beside the clip as an `ffmetadata` document for
    // the encoder to read. `None` means "encode exactly as before" — the user turned the setting
    // off, the job carries no payload, or the envelope is over the recording ceiling — which is
    // the same three-reasons-one-`Option` shape the image seam's `workflow_source` has.
    let workflow_metadata = plan.media_path.with_extension("workflow.ffmeta");
    let embedded =
        video_workflow_metadata(settings, job, &request, &plan, seed, &workflow_metadata);
    let encoded = encode_media(
        &plan.media_path,
        decoded,
        embedded.then_some(workflow_metadata.as_path()),
        Some(ctx),
    )
    .await;
    // The scratch goes on EVERY path, and the `?` waits for it. `encode_media(...).await?` here
    // propagated FIRST, so a failed or cancelled encode left `*.workflow.ffmeta` — the whole
    // envelope, prompt in plaintext — sitting in the user's project directory permanently, next to
    // no video. A metadata document is a file the user never asked for; it must not outlive the
    // encode it was written for. `encode_media` already keeps its own three temporaries on this
    // exact shape (`let result = …; remove; result`), and `seedvr2.rs` uses an RAII `ScratchDir`
    // for the same reason. Pinned by `the_workflow_metadata_scratch_does_not_outlive_a_failed_
    // encode` in `crates/sceneworks-worker/src/video_jobs/tests.rs`.
    let _ = tokio::fs::remove_file(&workflow_metadata).await;
    encoded?;

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

/// Build the sanitized workflow envelope for this clip and write it beside the media as an
/// `ffmetadata` document, returning whether the encoder should attach it (sc-15956).
///
/// **The video lane's write seam** — the counterpart of `image_jobs::workflow_source` plus
/// `write_image_asset`, collapsed into one function because a video job produces exactly one file.
/// Declared in `WORKFLOW_WRITE_SEAMS` as `Embeds`, naming the six video builders sc-15956 promoted
/// into `ADVANCED_BUILDERS`.
///
/// Returns `false` — "encode exactly as before" — for the same three reasons the image seam has,
/// and logs which one at `debug`, because a user asking why a clip has no recipe needs to know
/// whether they turned it off or the payload was empty:
///
/// * the job carries no payload to describe (a stub or dry-run write);
/// * `embedWorkflowInImages` did not resolve to true;
/// * the envelope is over the recording ceiling, or could not be written.
///
/// The last one degrades to no-workflow rather than to a failed job. A clip that generated fine is
/// not worth failing over a metadata document, and the same reasoning already governs
/// `faststart_mp4` and `write_poster_frame` below.
///
/// # The setting
///
/// Video honours `embedWorkflowInImages` — the SAME switch as images, not one of its own. The
/// setting is a privacy control over what leaves in a file, and the reason someone turns it off
/// ("my prompts are not going out in the files I share") does not stop applying at the container
/// boundary. Giving video its own switch would mean a user who had deliberately opted out started
/// embedding again the day this lane shipped, silently, which is the exact inversion
/// `embed_workflow_in_images`'s fails-closed rule exists to prevent. The stored key keeps its
/// `embedWorkflowInImages` name for back-compatibility — renaming it would reset every existing
/// opt-out to the default-on — and the UI copy is what names both media, since that is what the
/// user actually reads.
fn video_workflow_metadata(
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    plan: &VideoPlan,
    seed: i64,
    metadata_path: &Path,
) -> bool {
    if job.payload.is_empty() {
        tracing::debug!(
            reason = "empty_job_payload",
            "not embedding a workflow: the job carries no payload to describe"
        );
        return false;
    }
    if !sceneworks_core::app_paths::embed_workflow_in_images(&settings.config_dir) {
        tracing::debug!(
            reason = "preference_off",
            config_dir = %settings.config_dir.display(),
            "not embedding a workflow: `embedWorkflowInImages` did not resolve to true"
        );
        return false;
    }
    let facts = sceneworks_core::workflow_share::WorkflowAssetFacts {
        mode: request.mode.clone(),
        model: request.model.clone(),
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        // The seed this clip actually rendered with, resolved by `resolve_video_seed` — never the
        // payload's `seed`, which is `None` whenever the run derived one from the prompt.
        seed,
        width: Some(request.width),
        height: Some(request.height),
    };
    let Some(share) =
        sceneworks_core::workflow_share::embeddable_video_workflow_share(&facts, &job.payload)
    else {
        tracing::debug!(
            reason = "over_recording_ceiling",
            "not embedding a workflow: the envelope is larger than the recording ceiling"
        );
        return false;
    };
    match sceneworks_core::workflow_mp4::write_workflow_metadata_file(&share, metadata_path) {
        Ok(()) => {
            tracing::debug!(
                asset_id = %plan.asset_id,
                "embedding the sanitized workflow in the clip this job writes"
            );
            true
        }
        Err(error) => {
            tracing::warn!(
                %error,
                "not embedding a workflow: the metadata document could not be written"
            );
            false
        }
    }
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
    resolve_catalog_video_family(&request.model, &request.model_manifest_entry)
}

/// Resolve the catalog family without constructing a second [`VideoRequest`]. Both asset planning
/// and the pre-generation admission funnel call this exact policy so a custom manifest family can
/// neither inherit the built-in curve nor be recorded under a different family than was graded.
fn resolve_catalog_video_family(model_id: &str, model_manifest_entry: &JsonObject) -> String {
    if let Some(family) = model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .filter(|family| !family.trim().is_empty())
    {
        return family.to_owned();
    }
    if is_ltx_model(model_id) {
        "ltx-video".to_owned()
    } else if model_id.starts_with("wan") {
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
    DecodedVideo {
        frames,
        fps,
        audio,
        adapter_apply_reports: Vec::new(),
    }
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
/// 48 kHz — enough to exercise the WAV-write + AAC-mux path (see [`audio_mux_args`]) end to end.
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

/// Write `decoded` to `media_path` as an mp4: raw RGB frames streamed to libx264, an optional 16-bit
/// PCM WAV muxed as AAC and bounded at the picture's length ([`audio_mux_args`]), then a
/// best-effort `+faststart` remux and
/// `.poster.jpg`. `media_path` is created (atomically renamed from a temp) only on
/// success; all intermediates are removed regardless of outcome.
async fn encode_media(
    media_path: &Path,
    decoded: DecodedVideo,
    workflow_metadata: Option<&Path>,
    ctx: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let enc_tmp = media_path.with_extension("enc.mp4");
    let wav_tmp = media_path.with_extension("audio.wav");
    let mux_tmp = media_path.with_extension("mux.mp4");
    let result = encode_inner(
        media_path,
        decoded,
        workflow_metadata,
        ctx,
        &enc_tmp,
        &wav_tmp,
        &mux_tmp,
    )
    .await;
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
    workflow_metadata: Option<&Path>,
    ctx: Option<FfmpegContext<'_>>,
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

    let width = frames[0].width;
    let height = frames[0].height;
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| WorkerError::InvalidPayload("video frame dimensions overflow".to_owned()))?;
    for frame in &frames {
        if frame.width != width || frame.height != height {
            return Err(WorkerError::InvalidPayload(
                "video frames must all have the same dimensions".to_owned(),
            ));
        }
        if frame.pixels.len() != expected_bytes {
            return Err(WorkerError::InvalidPayload(
                "video frame buffer size mismatch".to_owned(),
            ));
        }
    }

    // Captured before `frames` is consumed below: step 2 bounds the file at the PICTURE's length,
    // and that length is this count — never anything read back off the soundtrack (sc-19425).
    let frame_count = frames.len();

    // 1. Stream the engine-owned RGB buffers directly into FFmpeg. This moves one existing frame
    // buffer at a time through the pipe: no per-frame PNG encode, no multi-GB scratch tree, and no
    // second whole-video concatenation.
    //
    // The workflow envelope (sc-15956) rides in HERE rather than in a post-hoc rewrite, so the tag
    // is part of the file from the moment it exists: there is no window in which a clip is on disk
    // without its recipe, and no extra pass over what can be gigabytes. It survives the two steps
    // below — measured, and pinned by `the_envelope_survives_the_whole_encode_chain` in
    // `crates/sceneworks-worker/src/video_jobs/tests.rs`.
    let chunks = frames.into_iter().map(|frame| frame.pixels).collect();
    let mut args = vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-f".to_owned(),
        "rawvideo".to_owned(),
        "-pix_fmt".to_owned(),
        "rgb24".to_owned(),
        "-video_size".to_owned(),
        format!("{width}x{height}"),
        "-framerate".to_owned(),
        fps.to_string(),
        "-i".to_owned(),
        "pipe:0".to_owned(),
    ];
    if let Some(metadata_path) = workflow_metadata {
        // Input 1. The frames are input 0 and stay mapped by ffmpeg's own stream selection (an
        // `ffmetadata` input carries no streams to compete with), so only the metadata source has
        // to be named.
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_input_args(
            metadata_path,
        ));
    }
    args.extend([
        "-c:v".to_owned(),
        "libx264".to_owned(),
        "-pix_fmt".to_owned(),
        "yuv420p".to_owned(),
        "-r".to_owned(),
        fps.to_string(),
    ]);
    if workflow_metadata.is_some() {
        args.extend(sceneworks_core::workflow_mp4::ffmetadata_map_args(1));
    }
    args.push(enc_tmp.to_string_lossy().into_owned());
    run_ffmpeg_with_stdin_chunks(args, chunks, ctx).await?;

    // 2. Mux the audio track (LTX, MiniMax-H3) as AAC, else the video-only mp4 is the result.
    let finished_tmp = if let Some(audio) = audio {
        write_wav_pcm16(&audio, wav_tmp)?;
        run_ffmpeg(
            audio_mux_args(enc_tmp, wav_tmp, mux_tmp, frame_count, fps),
            ctx,
        )
        .await?;
        mux_tmp
    } else {
        enc_tmp
    };

    // 3. Publish atomically, then best-effort faststart + poster (mirrors Python).
    tokio::fs::rename(finished_tmp, media_path).await?;
    faststart_mp4(media_path).await;
    write_poster_frame(media_path).await;
    Ok(())
}

/// **The AV mux policy's one number**: the picture's own length in seconds, `frame_count / fps`,
/// rendered to the microsecond for ffmpeg's `-t`. The picture is the exact quantity; the soundtrack
/// and the container are fitted to it.
///
/// Extracted (sc-19549) so the two muxing call sites cannot drift into two spellings of the same
/// rule: the generation mux [`audio_mux_args`] and the SeedVR2 upscale mux
/// `seedvr2::seedvr2_audio_mux_args` (cfg-gated to the lanes that ship it, so it is named rather
/// than linked). Their argument vectors legitimately differ — the upscale
/// maps a source clip's optional audio and writes `+faststart` in the same pass — but the BOUND is
/// one policy and is computed in exactly one place. The rationale, and the measurements behind the
/// choice of `-t` over `-shortest` and over no flag at all, live on [`audio_mux_args`].
///
/// `fps.max(1)` mirrors `encode_inner`'s own clamp rather than dividing by zero.
pub(crate) fn picture_bound_seconds(frame_count: usize, fps: u32) -> String {
    let seconds = frame_count as f64 / f64::from(fps.max(1));
    format!("{seconds:.6}")
}

/// The step-2 mux command: copy the encoded picture, encode the WAV as AAC, and **bound the file at
/// the picture's own length** — [`picture_bound_seconds`].
///
/// # Why `-t` and not `-shortest` (sc-19425)
///
/// `-shortest` makes the SOUNDTRACK a candidate for deciding the clip's length, and when it wins it
/// does so by **discarding video frames** — silently, and out of proportion to the mismatch.
/// Measured with the bundled ffmpeg 7.1 on this exact two-step command, at 24 fps, MiniMax-H3's 14
/// legal frame counts, with a soundtrack already fitted to the picture to within a third of one
/// sample (`round(frames / 24 · 32000)` samples per channel — the engine's own mux policy):
///
/// | frames | `-shortest` | no flag | `-t frames/fps` |
/// |---|---|---|---|
/// | 124 | **121** | 124 | 124 |
/// | 175 | **173** | 175 | 175 |
/// | 226 | **225** | 226 | 226 |
/// | 277 | **275** | 277 | 277 |
/// | 328 | **327** | 328 | 328 |
/// | the other 9 | exact | exact | exact |
///
/// Five of the fourteen lost picture — up to three frames for a 10 µs audio deficit — while the
/// container went on advertising the AUDIO's duration (5.17 s for a file holding 121 frames = 5.04 s
/// of picture). That is precisely "the container claims a duration the file does not have", the
/// defect sc-12371 measured `EncodedClip` for, reappearing one layer down where measuring the
/// `DecodedVideo` cannot see it.
///
/// Dropping the flag outright fixes the frame loss but gives up what it was for: an audio track
/// LONGER than the picture then extends the file (measured: a 2× track produced a 16.0 s container
/// around 8.0 s of picture). `-t` is the only one of the three that is right in both directions, and
/// it is the container-level spelling of the same rule the MiniMax-H3 engine applies to the
/// waveform: **the picture is the exact quantity, so the picture is what everything else is fitted
/// to.** It is a strict improvement for LTX too, whose vocoder output is not length-matched to the
/// frame count at all and which today loses picture whenever that output lands short.
///
/// # Why `-t` cannot do to the picture what `-shortest` does
///
/// MEASURED: under `-c:v copy` the bound carries a consistent **two frames of slack at the tail** —
/// `-t 4.0` on a 24 fps clip keeps 98 frames, not the 96 the arithmetic suggests, and `-t 2.5` keeps
/// 62, not 60. (Consistent with the bound being applied to decode timestamps, which lag presentation
/// by libx264's default B-frame reorder delay; the two-frame figure is measured, that attribution is
/// not.) So the bound is inclusive at its own end and errs toward keeping picture, which is the
/// direction that matters here: at
/// `frame_count / fps` there is nothing past it to keep, and a bound short by a microsecond of
/// `{:.6}` rounding is nowhere near a frame interval (41 667 µs at 24 fps) of the last frame's
/// presentation time at `(frame_count - 1) / fps`. Trimming a long soundtrack is unaffected — the
/// audio is re-encoded, not copied (measured: a 2× track lands at 5.17 s, not 10.33 s).
fn audio_mux_args(
    video: &Path,
    wav: &Path,
    out: &Path,
    frame_count: usize,
    fps: u32,
) -> Vec<String> {
    vec![
        "ffmpeg".to_owned(),
        "-nostdin".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        video.to_string_lossy().into_owned(),
        "-i".to_owned(),
        wav.to_string_lossy().into_owned(),
        "-c:v".to_owned(),
        "copy".to_owned(),
        "-c:a".to_owned(),
        "aac".to_owned(),
        "-t".to_owned(),
        picture_bound_seconds(frame_count, fps),
        // Explicit, though it is also ffmpeg's default for a multi-input command: the
        // container metadata — including the sc-15956 workflow tag written in step 1 —
        // comes from the VIDEO, never from the WAV. Stated because "the default happens to
        // be right" is not a property anyone maintains, and the step below it depends on
        // this one having carried the tag through.
        "-map_metadata".to_owned(),
        "0".to_owned(),
        out.to_string_lossy().into_owned(),
    ]
}

/// Write f32 PCM to a canonical 16-bit WAV. Signals already within `[-1, 1]` retain their original
/// amplitude; only over-range input is peak-normalized to prevent clipping. `pub(crate)` so the
/// pure-audio job path reuses it (sc-13404).
pub(crate) fn write_wav_pcm16(audio: &AudioTrack, path: &Path) -> WorkerResult<()> {
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |max, &sample| max.max(sample.abs()));
    let scale = i16::MAX as f32 / peak.max(1.0);
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
    /// Whether the mp4 carries an AAC soundtrack (sc-19577).
    ///
    /// `decoded.audio.is_some()` is the ONE condition `encode_inner` branches on to run the mux, so
    /// this is the muxer's own predicate rather than a claim about the model: a MiniMax-H3 t2va job
    /// whose pipeline returned no audio records `false` here, and an LTX render that did record
    /// `true`. Reading it from the model id — the obvious shortcut for the app's first joint
    /// audio+video family — would be a per-family hardcode that is wrong for exactly the renders a
    /// user would most want to check.
    ///
    /// Measured here rather than after `encode_media` because `decoded` is MOVED into the encoder;
    /// this is the last point at which the question can be asked at all.
    has_audio: bool,
}

impl EncodedClip {
    /// Measure the clip that is about to be written.
    fn measure(decoded: &DecodedVideo) -> Self {
        Self {
            frames: decoded.frames.len(),
            fps: decoded.fps.max(1),
            has_audio: decoded.audio.is_some(),
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
        // sc-19577 — whether this file has a soundtrack, measured off what was actually MUXED. It
        // lands in the asset's `file` block beside the other measurements and is the only thing any
        // audio indicator in the UI reads. Emitted unconditionally (never omitted for `false`), so an
        // ABSENT key means "recorded before sc-19577" and a `false` means "measured, and silent" —
        // two different facts that a presence check alone would conflate.
        "hasAudio": clip.has_audio,
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
        // sc-17160: the audio references join the list-valued ids for exactly the sc-12345 reason
        // — they are a top-level payload field, so the `advanced.clone()` every `*_raw_settings`
        // builder starts with does not carry them, and a replay that omits them re-runs a
        // multi-modal reference job with its audio conditioning silently missing.
        "referenceAudioAssetIds": request.reference_audio_asset_ids,
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

// The shared standalone-audio-reference resolver (sc-17160 / sc-18650). Not an engine family: it is
// cross-model and cross-platform. It is a module of its own so the shared parent does not re-absorb
// a self-contained media pipeline — the property
// `video_jobs_remains_split_into_real_engine_modules` bounds, and which sc-18650's ffmpeg
// normalization would otherwise have pushed past its line budget.
pub(crate) mod reference_audio;
pub(crate) use reference_audio::resolve_reference_audio_conditioning;
pub(crate) mod seedvr2;
pub(crate) mod wan;
#[cfg(target_os = "macos")]
use wan::{generate_wan, wan_available, wan_engine_id, wan_raw_settings, WAN_ADAPTER};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) mod bernini;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
// `pub(crate)` since sc-18814: the shared video memory gate reads this lane's VRAM budget from
// `candle_video_vram_budget` — the same figure `wan_video_fit_error` / `svd_fit_error` are handed
// — rather than probing the card a second way.
pub(crate) mod candle;
#[cfg(target_os = "macos")]
use bernini::{
    bernini_available, bernini_engine_id, bernini_raw_settings, generate_bernini, BERNINI_ADAPTER,
};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use bernini::{
    bernini_engine_id, bernini_raw_settings, generate_candle_bernini, CANDLE_BERNINI_ADAPTER,
};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle::{
    generate_candle_scail2, generate_candle_scail2_replace, generate_candle_video,
    generate_candle_wan_comfyui, generate_candle_wan_vace, generate_candle_wan_vace_extend_bridge,
    generate_candle_wan_vace_fun, is_candle_video_engine, wan_comfyui_claims,
    CANDLE_SCAIL2_ADAPTER, CANDLE_WAN_VACE_ADAPTER, CANDLE_WAN_VACE_FUN_ADAPTER,
};
mod scail2;
#[cfg(target_os = "macos")]
use scail2::{
    generate_scail2, generate_scail2_replace, scail2_available, scail2_engine_id,
    scail2_raw_settings, SCAIL2_ADAPTER,
};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use scail2::{scail2_engine_id, scail2_raw_settings};
mod krea_realtime;
#[cfg(target_os = "macos")]
use krea_realtime::{
    generate_krea_realtime, krea_realtime_available, krea_realtime_engine_id, KREA_REALTIME_ADAPTER,
};
pub(crate) mod ltx;
#[cfg(target_os = "macos")]
use ltx::{generate_ltx, ltx_available, ltx_engine_id, ltx_raw_settings, LTX_ADAPTER};
pub use ltx::{text_encoder_options_for_adapter, TextEncoderOption};
mod mochi;
#[cfg(target_os = "macos")]
use mochi::{generate_mochi, mochi_available, mochi_engine_id, MOCHI_ADAPTER};
pub(crate) mod minimax_h3;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use minimax_h3::minimax_h3_engine_id;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use minimax_h3::{generate_candle_minimax_h3, CANDLE_MINIMAX_H3_ADAPTER};
#[cfg(target_os = "macos")]
use minimax_h3::{generate_minimax_h3, minimax_h3_available, MINIMAX_H3_ADAPTER};
mod svd;
#[cfg(target_os = "macos")]
use svd::{generate_svd, svd_available, svd_engine_id, svd_raw_settings, SVD_ADAPTER};
mod vace;
#[cfg(target_os = "macos")]
use vace::{
    generate_wan_vace, generate_wan_vace_extend_bridge, generate_wan_vace_fun,
    resolve_wan_vace_model_dir, wan_vace_extend_raw_settings, wan_vace_raw_settings,
    WAN_VACE_ADAPTER, WAN_VACE_FUN_ADAPTER,
};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use vace::{wan_vace_extend_raw_settings, wan_vace_raw_settings};

/// Resolve the inference generator descriptors the production dispatch can load for one routed
/// video model/mode. This is the matching-platform mapping source consumed by the capability-facts
/// dumper; it deliberately reuses the same family resolvers as [`resolve_video_route`] and
/// [`resolve_candle_video_route`] instead of maintaining a second model table.
#[cfg(target_os = "macos")]
pub(crate) fn runtime_descriptor_engine_ids(model: &str, mode: &str) -> Vec<&'static str> {
    if model == "wan_2_2_vace_fun_14b" {
        return if mode == "replace_person" {
            vec!["wan2_2_vace_fun_14b"]
        } else {
            Vec::new()
        };
    }
    if mode == "replace_person" {
        return if let Some(engine_id) = scail2_engine_id(model) {
            vec![engine_id]
        } else if let Some(engine_id) = ltx_engine_id(model) {
            vec![engine_id]
        } else {
            vec!["wan_vace"]
        };
    }
    if matches!(mode, "extend_clip" | "video_bridge")
        && wan_engine_id(model) == Some("wan2_2_ti2v_5b")
    {
        // Production prefers the ControlClip VACE engine and falls back to the TI2V keyframe path
        // when that snapshot is not provisioned. Both descriptors therefore govern the route.
        return vec!["wan_vace", "wan2_2_ti2v_5b"];
    }
    wan_engine_id(model)
        .or_else(|| ltx_engine_id(model))
        .or_else(|| svd_engine_id(model))
        .or_else(|| bernini_engine_id(model))
        .or_else(|| scail2_engine_id(model))
        .or_else(|| krea_realtime_engine_id(model))
        .or_else(|| mochi_engine_id(model))
        // MiniMax-H3 (epic 17137 / sc-19508) — the same tail arm `resolve_video_route_with` carries.
        // Unfiltered by `minimax_h3_ready` exactly as every resolver above is unfiltered by its
        // `*_available` sibling: this function answers "which inference generator does dispatch
        // NAME for this route", not "is it loadable on this machine right now". Omitting it made
        // the capability-facts dumper report "worker dispatch resolves no inference engine" for a
        // route whose dispatch arm is right there — main added this derivation while the epic added
        // the family, and neither side's diff touched the other's lines.
        .or_else(|| minimax_h3_engine_id(model))
        .into_iter()
        .collect()
}

/// Candle sibling of [`runtime_descriptor_engine_ids`], sourced from the actual Candle dispatch
/// ladder and its `candle_video_engine_id` registry join.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn runtime_descriptor_engine_ids(model: &str, mode: &str) -> Vec<&'static str> {
    // SC-18902 withdrew Eros from Candle after its exact-head CUDA acceptance produced unusable
    // noise. Keep capability facts on the same terminal-refusal truth as production dispatch:
    // generic replacement/extension fallbacks must never advertise Wan-VACE for this stable id.
    if model == "ltx_2_3_eros" {
        return Vec::new();
    }
    if model == "wan_2_2_vace_fun_14b" {
        return if mode == "replace_person" {
            vec!["wan2_2_vace_fun_14b"]
        } else {
            Vec::new()
        };
    }
    if mode == "replace_person" {
        return vec![scail2_engine_id(model)
            .or_else(|| {
                candle::candle_video_engine_id(model)
                    .filter(|engine_id| *engine_id == "ltx_2_3_distilled")
            })
            .unwrap_or("wan_vace")];
    }
    if mode == "animate_character" {
        return scail2_engine_id(model).into_iter().collect();
    }
    if matches!(mode, "extend_clip" | "video_bridge") {
        return vec![candle::candle_video_engine_id(model)
            .filter(|id| *id == "ltx_2_3_distilled")
            .unwrap_or("wan_vace")];
    }
    bernini_engine_id(model)
        .or_else(|| candle::candle_video_engine_id(model))
        .or_else(|| minimax_h3_engine_id(model))
        .into_iter()
        .collect()
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
        // Last resort only — see `image_jobs::lora_weight`; matches the API/web default.
        .unwrap_or(1.0) as f32
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

/// Assemble SCAIL-2 `animate_character` conditioning from already-loaded character images +
/// driving frames. Each reference is independently segmented into its paired color mask; the
/// backend-specific SAM3 module and background convention stay at the call site while this
/// orchestration remains shared between MLX and Candle.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn assemble_scail2_animate_conditioning<FR, FD>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    segment_reference: FR,
    segment_driving: FD,
) -> WorkerResult<Vec<Conditioning>>
where
    FR: FnOnce(gen_core::CancelFlag) -> WorkerResult<Vec<(Image, Image)>> + Send + 'static,
    FD: FnOnce(gen_core::CancelFlag) -> WorkerResult<(Vec<Image>, Vec<Image>)> + Send + 'static,
{
    let reference_pairs = scail2::scail2_segment_blocking(
        api,
        settings,
        job_id,
        "scail2 reference segment task",
        segment_reference,
    )
    .await?;
    let (driving, driving_mask) = scail2::scail2_segment_blocking(
        api,
        settings,
        job_id,
        "scail2 driving segment task",
        segment_driving,
    )
    .await?;
    scail2::scail2_animate_conditioning(reference_pairs, driving, driving_mask)
}

// `pub(crate)` so `media_jobs`' own ffmpeg-backed tests can reuse `tests::ffmpeg_reachable` —
// the ONE place that turns "no ffmpeg on this lane" into an assert when the lane declared one
// (sc-19549). A second copy of that predicate is a second way for a lane to silently skip.
#[cfg(test)]
pub(crate) mod tests;
