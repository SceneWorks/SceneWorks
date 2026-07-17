//! The video-generation job request (epic 3018, sc-3033).
//!
//! Parses a job `payload` into a typed request so the native MLX/candle Rust
//! worker reads the same payload the UI already sends (the parse was ported from
//! the retired Python worker's `video_request_from_job`). The
//! `advanced` and `model_manifest_entry` maps pass through verbatim (they carry
//! per-model knobs like steps/guidanceScale/imageConditioningStrength and the
//! resolved manifest entry), so adding a model needs no DTO change.
//!
//! The advanced-mode asset ids (`last_frame` / `source_clip` / `bridge_right_clip`
//! / `person_track`) are parsed and carried so the asset fact + routing layer see
//! them. The MLX video path now consumes them for the cutover modes: `first_last_frame`
//! (two keyframes, sc-3520) and `replace_person` (the `source_clip` + `person_track` +
//! character refs → native Wan-VACE, sc-3521). `extend_clip` / `video_bridge` still stay
//! on the Python torch worker for now (sc-3522). The MLX-vs-Python routing decision is
//! sc-3036.

use serde_json::Value;

use crate::contracts::JsonObject;
use crate::payload_util::{
    array_or_empty, clamped_u32, declared_resolution, nonempty_string_or, object_or_empty,
    optional_i64, optional_id, parse_u32, string_list,
};

/// Defaults matching the Python `video_request_from_job` (`.get(key, default)`).
const DEFAULT_MODE: &str = "image_to_video";
const DEFAULT_MODEL: &str = "ltx_2_3";
const DEFAULT_QUALITY: &str = "balanced";
const DEFAULT_REPLACEMENT_MODE: &str = "face_only";

/// A typed video-generation request, parsed from a job payload. One job produces a
/// single video asset (unlike images, which batch `count`).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoRequest {
    pub project_id: String,
    pub mode: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub model: String,
    /// Clip length in seconds, clamped to 1.0..=30.0 (Python `safe_float`).
    ///
    /// That blanket 30 is a payload-sanity bound, NOT the model's limit: each video model
    /// declares its own `limits.hardMaxDuration` (4–15 s), enforced as a *rejection* by
    /// [`duration_limit_error`] at the API and worker gates rather than clamped here — see
    /// that function for why this parse is the wrong home for it (sc-12297).
    pub duration: f32,
    /// Frames per second, clamped to 1..=60 (Python `safe_int`).
    ///
    /// An **omitted** fps resolves to the model's declared `defaults.fps` rather than the blanket
    /// 25 — see [`resolve_fps`]. That blanket is out of menu for 7 of the 10 shipped video models,
    /// so it silently rendered them off-spec (mochi at 25 instead of 30 plays a 30 fps prior 20%
    /// slow) and would make [`fps_limit_error`] reject requests that name no fps at all.
    ///
    /// A value the caller *did* name is clamped here but never rewritten onto the model's menu:
    /// `limits.fps` is enforced as a *rejection* by [`fps_limit_error`] at the API and worker
    /// gates, because fps multiplies into `frames = duration × fps` and snapping it would silently
    /// change the clip length asked for (sc-12347).
    pub fps: u32,
    /// Output dimensions: clamped to 256..=1920, floored to the model's dimension
    /// granularity — `model_manifest_entry.limits.requiresDimensionsMultipleOf`, default
    /// 32 (Python `normalized_dimensions`) — then fitted into the model's area cap,
    /// `limits.maxPixels`, if it declares one (no cap when absent). Defaults 768x512.
    ///
    /// Each video model declares the stride its engine actually needs, so the advertised
    /// buckets survive the floor: 16 for mochi (sc-11993) and for bernini / the Wan A14B
    /// trio, 64 for ltx and svd. Several engines additionally cap **area** — 921,600 px for the
    /// 14B family (bernini / the Wan A14B trio / scail2) and 901,120 px for the TI2V-5B, each
    /// its own upstream budget (sc-12308). `maxPixels` is what keeps a raw over-cap request
    /// (retry / MCP / preset replay, none of which pass through the UI dropdown) from reaching
    /// them (sc-12294).
    pub width: u32,
    pub height: u32,
    pub quality: String,
    /// Base seed; `None` means derive deterministically from the prompt at run time.
    pub seed: Option<i64>,
    /// LoRA specs, passed through verbatim (shape resolved per family).
    pub loras: Vec<Value>,
    pub character_id: Option<String>,
    pub character_look_id: Option<String>,
    /// Image→video conditioning source (consumed by the MLX I2V path).
    pub source_asset_id: Option<String>,
    /// How the starting image is fitted to the output `width`×`height` before
    /// conditioning — one of crop/pad/outpaint/stretch, default crop (sc-6139).
    /// Mirrors `ImageRequest::fit_mode`; the I2V / first-last resolve paths pass it
    /// to the same `fit_engine_image` helper the image-edit lane uses so a square
    /// source no longer silently stretches into an off-aspect output.
    pub fit_mode: String,
    // --- Advanced-mode fields: parsed + carried, but Python-routed (sc-3036). ---
    pub person_track_id: Option<String>,
    pub replacement_mode: String,
    pub last_frame_asset_id: Option<String>,
    pub source_clip_asset_id: Option<String>,
    /// Multiple source clips for Bernini's multi-source-video edit mode
    /// (`multi_video_to_video` / mv2v, sc-5425). `generate_bernini` pushes one
    /// `Conditioning::VideoClip` per clip; the single-clip modes use
    /// `source_clip_asset_id` instead.
    pub source_clip_asset_ids: Vec<String>,
    pub bridge_right_clip_asset_id: Option<String>,
    /// Subject reference images for Bernini's reference-driven video modes
    /// (`reference_to_video` / `reference_video_to_video` / `ads2v`, sc-4703 /
    /// sc-5425). Consumed by the MLX `generate_bernini` path as the planner's
    /// `MultiReference` conditioning.
    pub reference_asset_ids: Vec<String>,
    /// The reference *video* slot for Bernini's `ads2v` mode (sc-5425): a second
    /// source video distinct from `source_clip_asset_id`. `generate_bernini` pushes
    /// it as a second `Conditioning::VideoClip` after the source clip.
    pub reference_clip_asset_id: Option<String>,
    /// Per-model advanced knobs (steps, guidanceScale, imageConditioningStrength,
    /// timelineContext, …), passed through.
    pub advanced: JsonObject,
    /// The resolved builtin+user model manifest entry (repo/quant/limits/…).
    pub model_manifest_entry: JsonObject,
}

impl VideoRequest {
    /// Parse a job payload (the `payload` object of a `JobSnapshot`). Infallible:
    /// missing fields fall back to the Python defaults and `project_id` may be empty
    /// — the caller validates it is present (the worker rejects an empty project id).
    pub fn from_payload(payload: &JsonObject) -> Self {
        let model_manifest_entry = object_or_empty(payload, "modelManifestEntry");
        let (width, height) = normalized_dimensions(
            payload.get("width"),
            payload.get("height"),
            &model_manifest_entry,
        );
        Self {
            project_id: nonempty_string_or(payload, "projectId", ""),
            mode: nonempty_string_or(payload, "mode", DEFAULT_MODE),
            prompt: nonempty_string_or(payload, "prompt", ""),
            negative_prompt: nonempty_string_or(payload, "negativePrompt", ""),
            model: nonempty_string_or(payload, "model", DEFAULT_MODEL),
            duration: resolve_duration(payload.get("duration"), &model_manifest_entry),
            fps: resolve_fps(payload.get("fps"), &model_manifest_entry),
            width,
            height,
            quality: nonempty_string_or(payload, "quality", DEFAULT_QUALITY),
            seed: optional_i64(payload, "seed"),
            loras: array_or_empty(payload, "loras"),
            character_id: optional_id(payload, "characterId"),
            character_look_id: optional_id(payload, "characterLookId"),
            source_asset_id: optional_id(payload, "sourceAssetId"),
            fit_mode: crate::image_request::normalize_fit_mode(
                payload.get("fitMode").and_then(Value::as_str),
            ),
            person_track_id: optional_id(payload, "personTrackId"),
            replacement_mode: nonempty_string_or(
                payload,
                "replacementMode",
                DEFAULT_REPLACEMENT_MODE,
            ),
            last_frame_asset_id: optional_id(payload, "lastFrameAssetId"),
            source_clip_asset_id: optional_id(payload, "sourceClipAssetId"),
            source_clip_asset_ids: string_list(payload, "sourceClipAssetIds"),
            bridge_right_clip_asset_id: optional_id(payload, "bridgeRightClipAssetId"),
            reference_asset_ids: string_list(payload, "referenceAssetIds"),
            reference_clip_asset_id: optional_id(payload, "referenceClipAssetId"),
            advanced: object_or_empty(payload, "advanced"),
            model_manifest_entry,
        }
    }

    /// Raw frame count from `duration * fps` (Python `round(duration * fps)`), at
    /// least 1. The per-model snap ([`frame_count`](Self::frame_count)) rounds this
    /// to the model's temporal stride.
    pub fn raw_frame_count(&self) -> u32 {
        ((self.duration * self.fps as f32).round() as i64).max(1) as u32
    }

    /// The frame count the model will actually produce: [`raw_frame_count`](Self::raw_frame_count)
    /// snapped to the model's temporal stride by the shared [`video_frame_count`] ladder — the SAME
    /// ladder the worker's generation arms resolve, so what the asset sidecar records can never
    /// drift from what the engine renders (sc-12371). The engine's `validate()` stays authoritative
    /// for the real models (sc-3034 / sc-3035); this is the pre-snap.
    pub fn frame_count(&self) -> u32 {
        video_frame_count(&self.model, self.raw_frame_count())
    }
}

/// The frame count `model` will actually render for `raw_frames` requested, coerced onto that
/// engine's temporal stride: LTX `8k+1`, **Mochi `6k+1`**, everything else (Wan and the families
/// built on it) `4k+1`.
///
/// THE ONE LADDER (sc-12371). It has two kinds of caller by design: [`VideoRequest::frame_count`],
/// which is what every asset sidecar records as `frameCount`, and the worker's generation arms,
/// which is what the engine is actually handed. Those were two independent implementations with
/// DIFFERENT fallback arms — the worker's fell through to Wan, this one kept the raw count — and the
/// gap was not theoretical: every arm that drives a Wan engine under a non-Wan model id rendered
/// `wan_frame_count(raw)` (149 at the 6 s x 25 fps default) while its sidecar wrote the unsnapped
/// raw (150). `bernini`, `scail2_14b` and the in-place ComfyUI `external_base_*` ids all did this,
/// on both lanes, and nothing failed loudly — the clip was simply one frame shorter than the asset
/// claimed. A frame count is invisible in the output when it is wrong (epic 1788).
///
/// THE FALLBACK IS WAN, DELIBERATELY — it is not the "unknown model" arm it reads as. Every video
/// model that reaches a real generation arm is a Wan-family renderer unless it is LTX or Mochi
/// (the explicit arms above): `bernini` is a Wan2.2-T2V-A14B renderer, `scail2_14b` a Wan2.1-14B
/// I2V one, `external_base_*` an in-place ComfyUI Wan2.2, and Wan-VACE pins the same stride. That
/// is why an `is_wan_model` arm here would be a bug, not a tidy-up: the ids above are Wan engines
/// that the predicate does not match. SVD is the one video engine off this lattice and it never
/// asks — it takes an explicit `numFrames` burst knob on both its engine input and its sidecar.
///
/// The per-family arms are what sc-11992 bought: the stride used to be an inline
/// `if is_ltx { ltx } else { wan }` binary repeated in each generation path, so Mochi silently
/// inherited the WAN stride. Mochi's AsymmVAE is 6x temporal and its `validate_request` hard-rejects
/// anything but `1 + 6k`, so EVERY default-duration Mochi job died on engine validation.
pub fn video_frame_count(model: &str, raw_frames: u32) -> u32 {
    if is_ltx_model(model) {
        ltx_frame_count(raw_frames)
    } else if is_mochi_model(model) {
        mochi_frame_count(raw_frames)
    } else {
        wan_frame_count(raw_frames)
    }
}

/// LTX-2.3 temporal stride: frames snap to the nearest `8k + 1`, minimum 9. Direct
/// port of the Python `ltx_frame_count`.
pub fn ltx_frame_count(raw_frames: u32) -> u32 {
    let frame_count = raw_frames.max(9);
    let lower = frame_count - ((frame_count - 1) % 8);
    let upper = lower + 8;
    if lower < 9 {
        return upper;
    }
    let lower_delta = frame_count - lower;
    let upper_delta = upper - frame_count;
    if lower_delta <= upper_delta {
        lower
    } else {
        upper
    }
}

/// Wan2.2 temporal stride: the VAE math `t_lat = (frames − 1) / 4 + 1` requires
/// `frames ≡ 1 (mod 4)`. Frames floor to the largest `4k + 1` at or below `raw`,
/// with a 5-frame minimum — the Python worker's `_run_wan_mlx`
/// `max(5, raw - ((raw - 1) % 4))`.
pub fn wan_frame_count(raw_frames: u32) -> u32 {
    let raw = raw_frames.max(1);
    (raw - ((raw - 1) % 4)).max(5)
}

/// Mochi's 7-frame floor: `6·1 + 1`, the smallest on-lattice count that yields more than
/// one latent frame (`lf = 1 + (frames − 1) / 6`). Mirrors LTX's 9 (`8·1 + 1`) and Wan's
/// 5 (`4·1 + 1`) — one temporal stride above a single still frame.
const MOCHI_MIN_FRAMES: u32 = 7;

/// Mochi 1 temporal stride: the AsymmVAE's 6× temporal ratio makes `t_lat` equal to
/// `1 + (frames − 1) / 6`, so the engine's `validate_request` hard-rejects anything but
/// `frames ≡ 1 (mod 6)`. Frames snap to the NEAREST `6k + 1` (ties → lower), minimum
/// [`MOCHI_MIN_FRAMES`] — the LTX rule, deliberately NOT Wan's floor: flooring the shipped
/// 5s @ 30fps default (150) would give 145, which the runtime happily accepts
/// (`145 % 6 == 1`) while silently rendering 4.83s instead of the requested 5s. Nearest
/// gives 151 (`1 + 6·25`), the frame count the manifest documents for the 5s default.
pub fn mochi_frame_count(raw_frames: u32) -> u32 {
    let frame_count = raw_frames.max(MOCHI_MIN_FRAMES);
    // `frame_count >= 7` ⇒ `lower >= 7`, so the floor needs no second guard.
    let lower = frame_count - ((frame_count - 1) % 6);
    let upper = lower + 6;
    if frame_count - lower <= upper - frame_count {
        lower
    } else {
        upper
    }
}

/// Whether `model` is an LTX-2.3 family id (`ltx_2_3`, `ltx_2_3_eros`, …).
pub fn is_ltx_model(model: &str) -> bool {
    model.starts_with("ltx")
}

/// Whether `model` is a Wan2.2 family id (`wan_2_2`, `wan_2_2_t2v_14b`, …).
pub fn is_wan_model(model: &str) -> bool {
    model.starts_with("wan")
}

/// Whether `model` is a Mochi 1 family id (`mochi_1`, …). Epic 1788 / sc-11993.
pub fn is_mochi_model(model: &str) -> bool {
    model.starts_with("mochi")
}

/// `limits.hardMaxDuration` (seconds) from a resolved manifest entry, or `None` for **no cap**.
///
/// The second constraint key core reads, after B3's [`dimension_multiple_of`] (sc-11993). It had
/// ten declarations and zero readers — in Rust *and* in the web — for as long as it has existed
/// (sc-12297): the word "hard" promised an enforcement that was never written, so every video
/// model accepted any duration the blanket `1..=30` clamp allowed.
///
/// Absent ⇒ no cap, so a model that declares nothing is byte-identical to before this key had a
/// reader. Same never-block-without-evidence posture as `mlx_fit_gate`'s `fit_decision`, and the
/// same typo'd-manifest tolerance as [`max_pixels_of`]: a non-positive or non-finite cap is not a
/// constraint anyone could satisfy (it would reject *every* request, including the model's own
/// `defaults.duration`), so it falls back to no cap rather than bricking the model.
pub fn hard_max_duration(model_manifest_entry: &JsonObject) -> Option<f32> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("hardMaxDuration"))
        .and_then(Value::as_f64)
        .map(|cap| cap as f32)
        .filter(|cap| cap.is_finite() && *cap > 0.0)
}

/// The actionable rejection for a `duration` past the model's declared [`hard_max_duration`], or
/// `None` to admit. The pure decision, so both enqueue gates assert on one message (sc-12297).
///
/// **Reject, never clamp.** Quietly rewriting a 30 s request into a 5 s render is the same
/// silent-coercion bug class B3 exists to fix (the invisible 848→832 dimension rewrite, sc-11993 /
/// sc-12294): duration is a creative choice, and returning a sixth of the clip the caller asked for
/// — with no error — is worse than refusing. This is also why the cap has no home in
/// [`VideoRequest::from_payload`], which is infallible by contract and has no error surface; the
/// rejection lives where a rejection can actually happen.
///
/// Message follows the house convention (`mlx_fit_gate::too_big_error`): name the model, state
/// what was asked and what the cap is, and give the lever. Pinned by
/// `duration_limit_error_names_the_model_the_request_and_the_cap`.
pub fn duration_limit_error(
    model: &str,
    duration: f32,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    let cap = hard_max_duration(model_manifest_entry)?;
    (duration > cap).then(|| {
        format!(
            "{model} renders clips up to {cap}s, but this request asks for {duration}s. Shorten \
             the clip to {cap}s or less, or choose a model that renders longer clips."
        )
    })
}

/// The clip length used when neither the caller nor the model's manifest entry names one — the
/// historical blanket, kept for models that declare no `defaults.duration`.
const DEFAULT_DURATION: f32 = 6.0;

/// The payload-sanity bounds on duration (the Python `safe_float`). NOT a model's limit: that is
/// [`hard_max_duration`].
const MIN_DURATION: f32 = 1.0;
const MAX_DURATION: f32 = 30.0;

/// `defaults.duration` (seconds) from a resolved manifest entry, or `None` when the model declares
/// none.
///
/// Dead in exactly the way [`default_fps`] was, with a worse consequence: sc-12297 began enforcing
/// [`hard_max_duration`] without ever comparing the API's **own** blanket [`DEFAULT_DURATION`] to
/// the caps it was now enforcing. 6.0 is past the cap of **7 of the 10** shipped video models, and
/// the DTO serializes it unconditionally, so a payload that named NO duration was rejected with
/// "this request asks for 6s" — naming a value the caller never set, and offering a lever
/// ("shorten the clip") for a field they never touched (sc-12400).
///
/// Same posture as [`default_fps`]: a default outside [`MIN_DURATION`]`..=`[`MAX_DURATION`], or
/// non-finite, is not a length anyone meant, so it falls back to the blanket rather than
/// propagating a degenerate value into `frames = duration × fps`.
pub fn default_duration(model_manifest_entry: &JsonObject) -> Option<f32> {
    model_manifest_entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("duration"))
        .and_then(Value::as_f64)
        .map(|duration| duration as f32)
        .filter(|duration| duration.is_finite() && (MIN_DURATION..=MAX_DURATION).contains(duration))
}

/// The clip length a payload actually renders at: the caller's value clamped to the sanity bounds,
/// or — when the caller named none — the model's declared [`default_duration`], falling back to
/// [`DEFAULT_DURATION`].
///
/// The duration twin of [`resolve_fps`], and the fix for the regression [`default_duration`]
/// documents. **Resolution, not rejection**: a value the caller *did* name is clamped and then
/// judged by [`duration_limit_error`], never quietly shortened onto the cap.
///
/// Reads through [`safe_float`], so the `safe_float` contract is unchanged: a numeric **string**
/// (`"4.5"`) is a value the caller named. An unreadable / `null` duration is treated as absent — a
/// value core cannot read is not a value the caller named.
pub fn resolve_duration(requested: Option<&Value>, model_manifest_entry: &JsonObject) -> f32 {
    let fallback = default_duration(model_manifest_entry).unwrap_or(DEFAULT_DURATION);
    safe_float(requested, fallback, MIN_DURATION, MAX_DURATION)
}

/// The frame rate used when neither the caller nor the model's manifest entry names one — the
/// historical blanket default, kept for models that declare no `defaults.fps` so an undeclared
/// model is byte-identical to before [`default_fps`] had a reader.
const DEFAULT_FPS: u32 = 25;

/// The payload-sanity bounds on fps (the Python `safe_int`). NOT a model's limit: that is
/// [`allowed_fps`]. Values outside this are a nonsense payload rather than an off-menu request,
/// so they clamp here and the model's menu judges the result.
const MIN_FPS: u32 = 1;
const MAX_FPS: u32 = 60;

/// `defaults.fps` from a resolved manifest entry, or `None` when the model declares none.
///
/// The third dead manifest key epic 12334 revives, and the one that makes [`fps_limit_error`]
/// shippable at all. `dto.rs`'s `default_video_fps` is a blanket **25** that no model chose, and
/// the MCP server omits `fps` entirely when its caller does — so "the caller said nothing" has
/// always resolved to 25 regardless of what the model declares. That is out of menu for **7 of 10**
/// shipped video models (every `[16]` model, `wan_2_2`'s `[16, 24]`, and mochi's `[30]`), which
/// means an fps allowlist alone would reject the API's *own default* on a request that merely
/// omits the field (sc-12347).
///
/// It is also a live bug on its own, with no odd input required: an MCP `generate_video(model =
/// "mochi_1", duration = 5)` resolves to 25 fps, not mochi's declared 30, so `5 × 25 = 125` snaps
/// to 127 frames — `127 % 6 == 1`, so the engine accepts it — and a 30 fps motion prior plays back
/// 20% slow. Exactly the harm mochi's own manifest comment predicts. The web never had this
/// problem because `VideoStudio.jsx:223` reads `defaults.fps`; only the backend ignored it.
///
/// Same never-invent-a-constraint-from-a-typo posture as [`hard_max_duration`] / [`max_pixels_of`]:
/// a default outside [`MIN_FPS`]`..=`[`MAX_FPS`] is not a frame rate anyone could have meant, so it
/// falls back to [`DEFAULT_FPS`] rather than propagating a degenerate value into the frame count.
pub fn default_fps(model_manifest_entry: &JsonObject) -> Option<u32> {
    model_manifest_entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("fps"))
        .and_then(Value::as_u64)
        .filter(|fps| (u64::from(MIN_FPS)..=u64::from(MAX_FPS)).contains(fps))
        .map(|fps| fps as u32)
}

/// `limits.fps` — the discrete set of frame rates a model advertises — or `None` for **no menu**,
/// meaning any fps the blanket [`MIN_FPS`]`..=`[`MAX_FPS`] bound allows.
///
/// Unlike [`hard_max_duration`]'s scalar ceiling this is an **allowlist**, so the comparison is
/// membership rather than `>`. It had ten declarations and zero *Rust* readers (sc-12347) while
/// the web bounded the picker with it at `VideoStudio.jsx:599/912` and `PresetManagerScreen.jsx`
/// already refused to *save* an off-menu fps via `checkInMenu` — so the product treated this key as
/// binding everywhere except the backend that actually renders.
///
/// Absent ⇒ no menu, so a model that declares nothing is byte-identical to before this key had a
/// reader. An array with no usable entry (empty, or every value non-integer / out of the sanity
/// bounds) is likewise **no menu**: a menu nothing can satisfy would reject every request including
/// the model's own `defaults.fps`, which is a typo'd manifest bricking a model rather than a
/// constraint. Same fallback as an unhonorable `hardMaxDuration` or `maxPixels`.
pub fn allowed_fps(model_manifest_entry: &JsonObject) -> Option<Vec<u32>> {
    let menu: Vec<u32> = model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("fps"))
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_u64)
        .filter(|fps| (u64::from(MIN_FPS)..=u64::from(MAX_FPS)).contains(fps))
        .map(|fps| fps as u32)
        .collect();
    (!menu.is_empty()).then_some(menu)
}

/// The fps a payload actually renders at: the caller's value clamped to the sanity bounds, or —
/// when the caller named none — the model's declared [`default_fps`], falling back to the blanket
/// [`DEFAULT_FPS`].
///
/// **Resolution, not rejection**, which is why it belongs in [`VideoRequest::from_payload`] (whose
/// infallibility [`fps_limit_error`] must not disturb) next to [`normalized_dimensions`] — the
/// established precedent for consulting the manifest entry during the parse. Filling an omitted
/// field from the model's own declaration is not coercion: nothing the caller asked for is being
/// overridden, and the alternative is the blanket 25 that silently renders 7 of 10 models off-spec.
/// A value the caller *did* name is never rewritten — it is clamped and then judged by
/// [`fps_limit_error`], so an off-menu fps is refused rather than quietly snapped onto the menu
/// (the reject-never-clamp rule [`duration_limit_error`] states).
///
/// Reads through [`parse_u32`], so the `safe_int` contract is unchanged: a numeric **string**
/// (`"30"`) is a value the caller named, not an omission. An unparseable / `null` fps IS treated as
/// absent — a value core cannot read is not a value the caller named. The API never produces one
/// (serde types the field), so that only governs hand-written worker payloads.
pub fn resolve_fps(requested: Option<&Value>, model_manifest_entry: &JsonObject) -> u32 {
    parse_u32(requested)
        .unwrap_or_else(|| default_fps(model_manifest_entry).unwrap_or(DEFAULT_FPS))
        .clamp(MIN_FPS, MAX_FPS)
}

/// The geometry used when neither the caller nor the model's manifest entry names one — the
/// historical blanket, kept for models that declare no `defaults.resolution`.
const DEFAULT_WIDTH: u32 = 768;
const DEFAULT_HEIGHT: u32 = 512;

/// The payload-sanity bounds on either dimension (the Python `normalized_dimensions` clamp). NOT a
/// model's limit: the stride is [`dimension_multiple_of`] and the area cap is [`max_pixels_of`].
const MIN_DIMENSION: u32 = 256;
const MAX_DIMENSION: u32 = 1920;

/// `defaults.resolution` (`"848x480"`) from a resolved manifest entry as `(width, height)`, or
/// `None` when the model declares none or declares one this parse cannot honor.
///
/// The **third** instance of the dead-`defaults.*` key [`default_fps`] and [`default_duration`]
/// document, and the one with no error surface at all: dimensions are *coerced* by
/// [`normalized_dimensions`], never rejected, so this failure is silent by construction rather than
/// a 400 someone eventually reports.
///
/// The blanket 768×512 is not merely un-declared for **8 of the 10** shipped video models — it is
/// not even in their `limits.resolutions`, i.e. it is a geometry the model never advertised at any
/// point. mochi_1 is the sharp case: its declared (and only trained) bucket is 848×480, so an MCP
/// `generate_video(model = "mochi_1", prompt = …)` that names no size rendered at 768×512 — a
/// geometry the model was never trained on — while the web, which reads `defaults.resolution` at
/// `VideoStudio.jsx:234`, rendered 848×480 from the same request (sc-12400).
///
/// Same never-invent-from-a-typo posture as its siblings: anything that is not `W x H` with both
/// dimensions inside the clamp range falls back to the blanket rather than emitting a degenerate
/// size. The value is deliberately NOT snapped to the model's stride here —
/// [`normalized_dimensions`] does that for declared and blanket geometry alike, so a model whose
/// advertised default is off its own lattice is floored exactly as a caller's request would be.
pub fn default_resolution(model_manifest_entry: &JsonObject) -> Option<(u32, u32)> {
    declared_resolution(model_manifest_entry, MIN_DIMENSION, MAX_DIMENSION)
}

/// The advertised frame rates as a human list: `30`, `16 or 24`, `6, 7, 8, 10, 12, or 25`.
fn humanized_fps_menu(menu: &[u32]) -> String {
    match menu {
        [only] => only.to_string(),
        [first, second] => format!("{first} or {second}"),
        [rest @ .., last] => format!(
            "{}, or {last}",
            rest.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        [] => String::new(),
    }
}

/// The actionable rejection for an `fps` outside the model's declared [`allowed_fps`], or `None` to
/// admit. The pure decision, so both enqueue gates assert on one message (sc-12347).
///
/// **Reject, never clamp**, for [`duration_limit_error`]'s reason and one of its own: fps is the
/// other multiplier in `frames = duration × fps`, so silently snapping 60 onto mochi's 30 would
/// halve the clip the caller asked for. sc-12297's duration cap does not bound frames — a *legally
/// 5-second* mochi request at 60 fps is 301 frames, double the shipped default's 151, and
/// `301 % 6 == 1` clears the engine's own check.
///
/// The gate is **uniform**, deliberately. svd's fps is pacing-only (its manifest says duration only
/// sets playback pacing), so an off-menu fps there does not multiply into length and is a weaker
/// harm than mochi's. Keying the gate on "does fps drive frame count" would hardcode family
/// knowledge into the gate — the exact anti-pattern epic 12334 exists to remove. The manifest is
/// the seam: a model that renders at more rates declares them, and one that renders at any rate
/// omits the key.
///
/// ⚠️ This bounds frames only as far as the *contract* goes — with sc-12297 it pins mochi at
/// `hardMaxDuration × max(limits.fps) = 5 × 30 → 151` frames, which is still ~4× past the ~36-frame
/// correctness ceiling MLX's `i32::MAX` element limit imposes (sc-12349). Honoring what a model
/// advertises is not a safety gate; do not read it as one.
///
/// Message follows the house convention (`mlx_fit_gate::too_big_error`): name the model, state what
/// was asked and what is allowed, and give the lever. Pinned by
/// `fps_limit_error_names_the_model_the_request_and_the_menu`.
pub fn fps_limit_error(model: &str, fps: u32, model_manifest_entry: &JsonObject) -> Option<String> {
    let menu = allowed_fps(model_manifest_entry)?;
    (!menu.contains(&fps)).then(|| {
        let allowed = humanized_fps_menu(&menu);
        format!(
            "{model} renders at {allowed} fps, but this request asks for {fps} fps. Set fps to \
             {allowed}, or choose a model that renders at {fps} fps."
        )
    })
}

/// The dimension granularity applied when a model's manifest entry does not declare
/// `limits.requiresDimensionsMultipleOf` — the historical blanket floor, kept as the
/// default so every already-shipped model's geometry is byte-for-byte unchanged.
const DEFAULT_DIMENSION_MULTIPLE: u32 = 32;

/// Clamp + floor dimensions like the Python `normalized_dimensions`: parse (default
/// 768x512), clamp to 256..=1920, floor to the model's dimension multiple, floor of 256.
///
/// The multiple comes from the resolved manifest entry's
/// `limits.requiresDimensionsMultipleOf`, defaulting to [`DEFAULT_DIMENSION_MULTIPLE`].
/// A blanket 32 silently rewrote Mochi's native (and only trained) 848x480 bucket to
/// 832x480 — `848 % 32 == 16` — and the rewrite was invisible because `832 % 16 == 0`
/// satisfies the engine's own ÷16 check (sc-11993). Models opt in by *declaring* the
/// limit; the rest keep 32.
///
/// sc-12294 then swept every video model against its engine's real constraint, and the
/// answer differed per model — the floor is not one number:
///
/// * **16** — bernini and the Wan A14B trio (t2v / i2v / vace-fun) ride a z16 VAE:
///   `patch 2 × vae_stride 8`. The constraint is the engines' own divisibility check —
///   candle `SIZE_MULTIPLE_14B = 16`, enforced via `is_multiple_of`, and mlx's `align_dim`
///   rounding to `patch · vae_stride`. They now declare it, so bernini's own default
///   848x480 stops silently rendering as 832x480. `720 = 45·16` is ON this lattice, so these
///   models advertise **true 720p** (`1280x720`) — see the area note below.
/// * **32** — the dense Wan TI2V-5B (z48 vae22, `vae_stride 16` → candle `SIZE_MULTIPLE =
///   32`) and scail2 (hardcoded `DIM_ALIGN`) genuinely need it. There the *advertisement*
///   was wrong, so their 720p buckets were corrected to the 704 they always rendered (`704 =
///   32·22`; 720 is off a 32 lattice); both keep this default.
/// * **64** — ltx (`validate_request` hard-errors below it: stage-1 runs at `//2//32`) and
///   svd (`SIZE_ALIGN = VAE_SCALE * 8`). A declared 32 was too *loose* for these: it could
///   floor onto a ÷32-but-not-÷64 size the engine then rejects.
///
/// (The descriptors' `min_size` corroborates each stride by convention, but it is NOT the
/// evidence: gen-core checks `req.width < self.min_size` — a lower *bound*, not a lattice.
/// The divisibility constants above are the actual constraint.)
///
/// So a declared value can be looser OR stricter than the default — it is the engine's
/// truth, not a blanket policy.
///
/// The stride is only half the geometry. Some engines also cap **area**, and the stride
/// floor does not imply the cap: a finer stride makes MORE over-cap sizes reachable, not
/// fewer. Models whose engine caps declare `limits.maxPixels`, and the result is fitted
/// into it by [`fit_to_max_pixels`]. A model that declares nothing is uncapped (sc-12294).
fn normalized_dimensions(
    width: Option<&Value>,
    height: Option<&Value>,
    model_manifest_entry: &JsonObject,
) -> (u32, u32) {
    let multiple = dimension_multiple_of(model_manifest_entry);
    // An omitted side takes the model's declared `defaults.resolution`, not the blanket 768×512 —
    // which 8 of the 10 shipped video models do not advertise anywhere, mochi_1's 848×480 native
    // bucket being the case that matters (sc-12400). Each side falls back independently, exactly as
    // the two blanket constants did, so a caller naming only one dimension is unchanged in shape.
    let (default_width, default_height) =
        default_resolution(model_manifest_entry).unwrap_or((DEFAULT_WIDTH, DEFAULT_HEIGHT));
    let w = floor_to_multiple(
        clamped_u32(width, default_width, MIN_DIMENSION, MAX_DIMENSION),
        multiple,
    );
    let h = floor_to_multiple(
        clamped_u32(height, default_height, MIN_DIMENSION, MAX_DIMENSION),
        multiple,
    );
    match max_pixels_of(model_manifest_entry) {
        Some(cap) if (w as u64) * (h as u64) > cap => fit_to_max_pixels(w, h, multiple, cap),
        _ => (w, h),
    }
}

/// The lower bound a [`max_pixels_of`] cap must clear to be honorable: [`floor_to_multiple`]
/// hard-floors every dimension at 256, so a cap under 256×256 could never be satisfied.
const MIN_HONORABLE_MAX_PIXELS: u64 = 256 * 256;

/// `limits.maxPixels` from a resolved manifest entry, or `None` for **no cap** — the
/// default-absent behavior, so every model that does not declare one is byte-identical to
/// before this key existed (sc-12294).
///
/// A cap is declared only where the *engine* enforces one, and it is deliberately not a
/// blanket: ltx / svd / mochi have no area check in either backend, so they get no key.
/// Where the two backends disagree the manifest carries the STRICTER truth, because one
/// manifest serves both — see the per-model notes in `builtin.models.jsonc`.
///
/// Same infallibility posture as [`dimension_multiple_of`]: `from_payload` must survive a
/// typo'd manifest, so a cap the geometry could never honor (zero / negative / non-integer /
/// below the 256×256 floor) falls back to *no cap* rather than emitting a degenerate size.
/// That matches the default-absent behavior instead of inventing a constraint from a typo.
fn max_pixels_of(model_manifest_entry: &JsonObject) -> Option<u64> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("maxPixels"))
        .and_then(Value::as_u64)
        .filter(|cap| *cap >= MIN_HONORABLE_MAX_PIXELS)
}

/// The largest `(width, height)` inside `max_pixels` that preserves the input aspect ratio and
/// stays on the `multiple` lattice.
///
/// Keeps the shape of `generate_wan.py`'s `_best_output_size`: derive the ideal `(ow, oh)` from
/// `√(max_area·ratio)`, then try width-first and height-first alignment and keep whichever distorts
/// the ratio less.
///
/// **This is app-layer normalization, not a mirror of engine code (sc-12308).** It was originally
/// ported from mlx's `best_output_size` so the app would agree with the backend that *silently
/// rescaled*. Neither backend does that any more — mlx now rejects an over-cap request exactly as
/// candle always did — so this function's job is to normalize a genuinely over-cap **custom** request
/// into a legal geometry before it ever reaches an engine. It must never fire for an advertised
/// bucket: with each model's cap corrected to its own engine's real budget, every bucket is a fixed
/// point (pinned by `shipped_manifest_matches_each_engines_real_geometry`).
///
/// The `.max(d)` clamps guard against a degenerate area flooring a dimension to 0 (and a subsequent
/// `area / 0` → NaN ratio). They cannot fire for any *shipped* cap, and are kept rather than trimmed.
///
/// The bound comes from the shipped caps, NOT from [`max_pixels_of`]'s filter — that distinction
/// matters if a future cap is added. [`MIN_HONORABLE_MAX_PIXELS`] only guarantees a *square*
/// 256×256 fits; it says nothing about a non-square fit, which drives the minor dimension well
/// below 256 (a cap of exactly 65,536 — which passes the filter — fits 1920×256 to `(688, 80)` at
/// stride 16, violating [`normalized_dimensions`]'s own 256 floor and every engine's `min_size`).
/// What actually rules the clamps out is the reachable minimum: [`normalized_dimensions`] clamps
/// both inputs to `[256, 1920]`, so the aspect ratio cannot exceed `1920/256 = 7.5`, and at the
/// SMALLEST cap the manifest declares (the 5B's 901,120) the smaller ideal dimension is
/// `√(901120/7.5) ≈ 346` — comfortably above every stride, so nothing floors to 0. (The 14B family's
/// 921,600 is larger, so it is slacker still.) A cap below ~491,520 would put an extreme-ratio
/// request back into clamp territory; the manifest's caps are test-pinned.
fn fit_to_max_pixels(width: u32, height: u32, multiple: u32, max_pixels: u64) -> (u32, u32) {
    let (w, h, d) = (width as f64, height as f64, multiple as f64);
    let area = max_pixels as f64;
    let ratio = w / h;
    let ow = (area * ratio).sqrt();
    let oh = area / ow;

    // Option 1: align width first, derive height from the remaining area.
    let ow1 = ((ow / d).floor() * d).max(d);
    let oh1 = ((area / ow1 / d).floor() * d).max(d);
    let ratio1 = ow1 / oh1;

    // Option 2: align height first, derive width.
    let oh2 = ((oh / d).floor() * d).max(d);
    let ow2 = ((area / oh2 / d).floor() * d).max(d);
    let ratio2 = ow2 / oh2;

    let dist1 = (ratio / ratio1).max(ratio1 / ratio);
    let dist2 = (ratio / ratio2).max(ratio2 / ratio);
    if dist1 < dist2 {
        (ow1 as u32, oh1 as u32)
    } else {
        (ow2 as u32, oh2 as u32)
    }
}

/// `limits.requiresDimensionsMultipleOf` from a resolved manifest entry, else
/// [`DEFAULT_DIMENSION_MULTIPLE`]. Read as a plain `u64` (the manifest is machine-authored
/// — same reader as the `engines.rs` manifest checks). `from_payload` is infallible and
/// must survive a typo'd manifest entry, so anything the geometry can't honor falls back
/// to the 32 default rather than panicking or silently emitting an off-lattice dimension.
///
/// Accepted: a positive power of two that divides 256 (1/2/4/8/16/32/64/128/256 — Mochi's
/// declared 16 and every shipped model's 32 are both in the set). Everything else falls
/// back: zero / negative / non-integer (which would panic on `% 0` or fail `as_u64`), and
/// — less obviously — any multiple that does not divide 256. [`floor_to_multiple`] clamps
/// its result up to a hard floor of 256, and that `.max(256)` rescue is only on-lattice
/// when the multiple divides 256. A typo'd `1024` would otherwise yield 256x256 with
/// `256 % 1024 == 256 != 0`, breaking the multiple invariant with no fallback firing; so
/// would an in-range-looking `96` (`256 % 96 == 64`). Divisibility is the real constraint
/// here, not magnitude — a bare upper bound would let `96` through.
fn dimension_multiple_of(model_manifest_entry: &JsonObject) -> u32 {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("requiresDimensionsMultipleOf"))
        .and_then(Value::as_u64)
        .and_then(|multiple| u32::try_from(multiple).ok())
        .filter(|multiple| *multiple > 0 && 256 % *multiple == 0)
        .unwrap_or(DEFAULT_DIMENSION_MULTIPLE)
}

fn floor_to_multiple(value: u32, multiple: u32) -> u32 {
    (value - (value % multiple)).max(256)
}

/// Parse a float (JSON number or numeric string), clamp to `[min, max]`, default
/// when absent/unparseable — the `safe_float` contract. Video-specific (image has no
/// float payload field), so it stays local rather than moving into `payload_util`.
fn safe_float(value: Option<&Value>, default: f32, min: f32, max: f32) -> f32 {
    value
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn defaults_when_payload_is_minimal() {
        let request = VideoRequest::from_payload(&payload(json!({ "projectId": "proj_1" })));
        assert_eq!(request.project_id, "proj_1");
        assert_eq!(request.mode, "image_to_video");
        assert_eq!(request.model, "ltx_2_3");
        assert_eq!(request.duration, 6.0);
        assert_eq!(request.fps, 25);
        assert_eq!(request.width, 768);
        assert_eq!(request.height, 512);
        assert_eq!(request.quality, "balanced");
        assert_eq!(request.replacement_mode, "face_only");
        assert!(request.seed.is_none());
        assert!(request.loras.is_empty());
        assert!(request.advanced.is_empty());
        assert!(request.source_asset_id.is_none());
        // sc-6139: starting-image fit defaults to crop (never stretch), like images.
        assert_eq!(request.fit_mode, "crop");
    }

    #[test]
    fn normalizes_fit_mode_like_images() {
        let pad = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "mode": "image_to_video", "fitMode": "pad"
        })));
        assert_eq!(pad.fit_mode, "pad");

        // Case-insensitive, same normalizer as ImageRequest.
        let stretch = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "fitMode": "Stretch"
        })));
        assert_eq!(stretch.fit_mode, "stretch");

        // Unknown / blank coerce back to the crop default.
        let bogus = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "fitMode": "nonsense"
        })));
        assert_eq!(bogus.fit_mode, "crop");
    }

    #[test]
    fn clamps_duration_fps_and_dimensions() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "duration": 99, "fps": 999, "width": 99999, "height": 100
        })));
        assert_eq!(request.duration, 30.0);
        assert_eq!(request.fps, 60);
        // 1920 is already a multiple of 32; 100 clamps to 256.
        assert_eq!(request.width, 1920);
        assert_eq!(request.height, 256);

        let low = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "duration": 0, "fps": 0
        })));
        assert_eq!(low.duration, 1.0);
        assert_eq!(low.fps, 1);
    }

    #[test]
    fn floors_dimensions_to_multiple_of_32_by_default() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "width": 1000, "height": 543
        })));
        // 1000 -> 992 (31*32), 543 -> 512 (16*32).
        assert_eq!(request.width, 992);
        assert_eq!(request.height, 512);

        // A manifest entry that does NOT declare the limit keeps the 32 default. This is
        // still the shipped case for the two models whose engines genuinely need 32 —
        // wan_2_2 (TI2V-5B) and scail2_14b — so the default path stays live and covered.
        // Only models that *declare* `requiresDimensionsMultipleOf` move off it.
        let undeclared = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "wan_2_2", "width": 848, "height": 480,
            "modelManifestEntry": { "family": "wan-video", "limits": { "resolutions": ["832x480"] } }
        })));
        assert_eq!((undeclared.width, undeclared.height), (832, 480));
    }

    /// sc-12305 — the silent failure that justifies rejecting a generation job on the
    /// generic `POST /api/v1/jobs`.
    ///
    /// That route enqueues `type` + payload verbatim, resolving no manifest entry (only the
    /// typed `/api/v1/video/jobs` and `/api/v1/image/jobs` do). With the entry absent,
    /// `dimension_multiple_of` misses `requiresDimensionsMultipleOf` and falls back to 32,
    /// so Mochi's native — and only trained — 848x480 bucket renders as 832x480. The engine
    /// never catches it: `832 % 16 == 0` satisfies its own ÷16 check, so there is no error,
    /// just off-bucket inference.
    ///
    /// This pins the *reason* the API guard exists. It is deliberately asserted here, at the
    /// geometry, rather than only at the route: if a future change ever makes an absent entry
    /// safe (a distinguishable absent-vs-unknown signal — sc-12304), this test goes RED and
    /// the guard can be revisited on evidence instead of belief.
    #[test]
    fn mochi_without_manifest_entry_silently_loses_its_native_bucket() {
        let no_entry = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 848, "height": 480,
        })));
        assert_eq!(
            (no_entry.width, no_entry.height),
            (832, 480),
            "an absent modelManifestEntry must still be the ÷32 fallback — if this changed, \
             the sc-12305 API guard's premise changed with it"
        );
        // Silent, not loud: the engine's own divisibility check passes on the rewritten size.
        assert_eq!(no_entry.width % 16, 0);

        // The same payload through a typed route — which resolves the entry — keeps 848.
        let with_entry = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 848, "height": 480,
            "modelManifestEntry": { "family": "mochi", "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((with_entry.width, with_entry.height), (848, 480));
    }

    #[test]
    fn honors_manifest_dimension_multiple_so_mochi_keeps_848() {
        // 848 % 32 == 16, so the blanket 32 floor silently rewrote Mochi's ONLY trained
        // bucket to 832x480 — and 832 % 16 == 0, so the engine's ÷16 check passed and the
        // off-bucket render went unnoticed. The manifest's `requiresDimensionsMultipleOf: 16`
        // now preserves it.
        let mochi = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 848, "height": 480,
            "modelManifestEntry": { "family": "mochi", "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((mochi.width, mochi.height), (848, 480));

        // The portrait bucket too.
        let portrait = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 480, "height": 848,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((portrait.width, portrait.height), (480, 848));

        // An explicitly declared 32 is identical to the default path.
        let declared_32 = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "scail2_14b", "width": 1000, "height": 543,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 32 } }
        })));
        assert_eq!((declared_32.width, declared_32.height), (992, 512));

        // 16 is a smaller floor, not a bypass: off-lattice input still floors.
        let odd = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 855, "height": 489,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((odd.width, odd.height), (848, 480));
    }

    /// sc-12294: the floor is per-engine, and a declared value can be looser OR stricter
    /// than the 32 default. Each case below mirrors the stride derived from that engine's
    /// own constraint, so the advertised bucket is what actually renders.
    #[test]
    fn declared_multiple_honors_each_engines_true_stride() {
        // --- 16: bernini renders through a Wan2.2-T2V-A14B snapshot (patch 2 × vae_stride
        // 8). Its OWN default bucket, 848x480, used to floor to 832x480 under the blanket
        // 32 — silently, because 832 % 16 == 0 passed the engine's own check.
        let bernini = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "bernini", "width": 848, "height": 480,
            "modelManifestEntry": { "family": "bernini", "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((bernini.width, bernini.height), (848, 480));

        // The grid-16 14B family (bernini + the Wan trio) renders TRUE 720p: `720 = 45·16` is on
        // its lattice and `1280×720 = 921,600` is exactly its area cap (upstream's own
        // `MAX_AREA_CONFIGS` for t2v/i2v-A14B and vace-14B). It must survive the ÷16 floor AND the
        // area fit untouched. This bucket was previously advertised at 1280x704 — the TI2V-5B's
        // geometry — because the cap wrongly carried the 5B's 901,120 (sc-12308).
        for model in [
            "bernini",
            "wan_2_2_t2v_14b",
            "wan_2_2_i2v_14b",
            "wan_2_2_vace_fun_14b",
        ] {
            let req = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": model, "width": 1280, "height": 720,
                "modelManifestEntry": { "limits": {
                    "requiresDimensionsMultipleOf": 16, "maxPixels": 921_600
                } }
            })));
            assert_eq!(
                (req.width, req.height),
                (1280, 720),
                "{model} must render its canonical 720p as asked, not refit it"
            );
            assert!(
                (req.width as usize) * (req.height as usize) <= 1280 * 720,
                "{model} bucket must fit MAX_AREA_14B"
            );

            // Every A14B advertised bucket is ÷32, so the at-cap check above holds under
            // either floor. What the declared 16 actually buys is a CUSTOM dim below the
            // cap: 848 survives at ÷16 and would be cut to 832 at ÷32. This is the
            // assertion that makes the declaration load-bearing for the Wan trio.
            let custom = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": model, "width": 848, "height": 480,
                "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
            })));
            assert_eq!(
                (custom.width, custom.height),
                (848, 480),
                "{model} declares 16, so a custom 848x480 must not be cut to 832"
            );
        }

        // --- 64: ltx hard-errors below ÷64 (stage-1 runs at //2//32) and svd's UNet needs
        // VAE 8× × 8×. A declared 32 was too LOOSE — 736 is ÷32 but the engine rejects it.
        // 64 floors it onto the lattice the engine actually accepts.
        for model in ["ltx_2_3", "ltx_2_3_eros", "svd"] {
            let req = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": model, "width": 1280, "height": 736,
                "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 64 } }
            })));
            assert_eq!(
                (req.width, req.height),
                (1280, 704),
                "{model} must floor 736 -> 704"
            );
            assert_eq!(req.height % 64, 0, "{model} must land on the ÷64 lattice");
        }

        // Each model's advertised buckets are fixed points under its own stride — that is
        // the whole invariant sc-12294 restores: what is advertised is what renders.
        for (model, multiple, buckets) in [
            (
                "bernini",
                16,
                vec![(848, 480), (480, 848), (1280, 704), (704, 1280)],
            ),
            (
                "wan_2_2_t2v_14b",
                16,
                vec![(832, 480), (480, 832), (1280, 704), (704, 1280)],
            ),
            (
                "wan_2_2_i2v_14b",
                16,
                vec![(832, 480), (480, 832), (1280, 704), (704, 1280)],
            ),
            (
                "wan_2_2_vace_fun_14b",
                16,
                vec![(832, 480), (480, 832), (1280, 704), (704, 1280)],
            ),
            (
                "ltx_2_3",
                64,
                vec![(768, 512), (512, 768), (640, 640), (1280, 704), (704, 1280)],
            ),
            ("svd", 64, vec![(1024, 576), (576, 1024)]),
            // Genuinely-32 engines: the advertisement was corrected to the 704 they render.
            ("wan_2_2", 32, vec![(832, 480), (1280, 704), (704, 1280)]),
            (
                "scail2_14b",
                32,
                vec![(832, 480), (480, 832), (1280, 704), (704, 1280)],
            ),
        ] {
            for (w, h) in buckets {
                let req = VideoRequest::from_payload(&payload(json!({
                    "projectId": "p", "model": model, "width": w, "height": h,
                    "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": multiple } }
                })));
                assert_eq!(
                    (req.width, req.height),
                    (w, h),
                    "{model} advertises {w}x{h}; it must render {w}x{h}"
                );
            }
        }
    }

    /// Every video model in the SHIPPED `builtin.models.jsonc`, keyed by id, with the geometry
    /// its ENGINE actually enforces — derived from the runtime source, NOT read back from the
    /// manifest. That independence is the whole point: this table is the expectation, the
    /// manifest is the thing under test, so a manifest edit that drifts from engine truth
    /// fails here instead of shipping.
    ///
    /// `multiple` — the divisibility the engine checks (`is_multiple_of` on candle,
    /// `align_dim` rounding on mlx). `None` = declares nothing, inherits the 32 default.
    /// `max_pixels` — the engine's area cap; `None` = neither backend caps, so the manifest
    /// must NOT invent one.
    ///
    /// **The backends now agree** — sc-12308 reconciled them, so the table is simply each
    /// engine's cap rather than the stricter of two answers. Both reject an over-cap request
    /// (candle `wan14b.rs:645` / `model_vace.rs:298` / `candle-gen-bernini/src/config.rs:164` /
    /// `candle-gen-scail2/src/pipeline.rs:294` / `lib.rs`'s new `MAX_AREA_5B` check; mlx
    /// `validate_impl` → `reject_over_area`), at these values.
    ///
    /// Before that they disagreed three ways on one manifest entry, and the table had to carry
    /// the strictest: mlx T2V/VACE/bernini/scail2 were **uncapped**, mlx I2V/5B **silently
    /// refit** (1280×720 → 1264×704, off every advertised bucket), and candle **hard-errored**
    /// — except candle's own 5B, which had no area check at all and ran to an opaque OOM. The
    /// values themselves were wrong too: the whole 14B family carried the TI2V-5B's 901,120.
    /// * `ltx_2_3` / `ltx_2_3_eros` / `svd` / `mochi_1` — no `maxPixels`-expressible area cap in
    ///   either backend, so no cap is declared. Not literally "no checks": candle-LTX caps
    ///   **latent tokens** (`candle-gen-ltx/src/lib.rs:454`: `t_lat·h_lat·w_lat > 131_072`), which
    ///   is proportional to `frames × w × h`. That is a frames×area constraint and therefore
    ///   outside `maxPixels`' scope — `maxPixels` is a pure per-frame area budget with no frame
    ///   term, so no value of it could express this cap without either under- or over-constraining
    ///   at some duration. It is also unreachable within the advertised duration/fps limits (the
    ///   worst advertised case, 15s at 30fps and 1280×704, is ~25k of the 131,072 tokens), and
    ///   candle-LTX registers `ltx_2_3_distilled` — a different id from these entries. Frame-count
    ///   shaping is the sibling lever (cf. mochi's 6k+1 rule, sc-11993); if an LTX request ever
    ///   needs to honor this cap it belongs there, not in `maxPixels`.
    const ENGINE_GEOMETRY: &[(&str, Option<u32>, Option<u64>)] = &[
        ("ltx_2_3", Some(64), None),
        ("ltx_2_3_eros", Some(64), None),
        ("svd", Some(64), None),
        // The 5B keeps 901,120 — upstream gives `ti2v-5B` exactly `1280*704` / `704*1280`, and its
        // z48 VAE's 32-px grid is why 704, not 720, is its real 720p.
        ("wan_2_2", None, Some(901_120)),
        // The 14B family carries upstream's own `1280*720` = 921,600 budget (sc-12308). It was
        // 901,120 here only because the engine constant had borrowed the 5B's number.
        ("wan_2_2_t2v_14b", Some(16), Some(921_600)),
        ("wan_2_2_i2v_14b", Some(16), Some(921_600)),
        ("wan_2_2_vace_fun_14b", Some(16), Some(921_600)),
        ("bernini", Some(16), Some(921_600)),
        // SCAIL-2 shares the 14B cap, but its buckets stay 704-tall for an unrelated reason: its
        // own `DIM_ALIGN = 32` tiling grid (mlx-gen-scail2/src/generate.rs:73) — NOT the VAE stride
        // — so 720 is off ITS lattice even though the area now fits.
        ("scail2_14b", None, Some(921_600)),
    ];

    /// The `models` array of the SHIPPED manifest — the exact bytes the app embeds and seeds.
    fn builtin_video_models() -> Vec<Value> {
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw)).expect("parses as JSON");
        manifest
            .get("models")
            .and_then(Value::as_array)
            .expect("models array")
            .iter()
            .filter(|m| m.get("type").and_then(Value::as_str) == Some("video"))
            .cloned()
            .collect()
    }

    /// sc-12294 — the invariant this story exists to establish, asserted against the REAL
    /// manifest bytes rather than inline literals: **what a video model advertises is what it
    /// renders, and everything it advertises is legal on its engine.**
    ///
    /// The earlier revision of this suite hardcoded the manifest values it claimed to verify
    /// (`"limits": { "requiresDimensionsMultipleOf": 16 }` written straight into the payload),
    /// so it only re-tested `normalized_dimensions`' arithmetic — reverting bernini's manifest
    /// declaration to 32 (restoring the original sc-12294 bug) left the whole suite GREEN.
    /// This test loads the shipped manifest and checks it against [`ENGINE_GEOMETRY`], so that
    /// mutation is RED.
    #[test]
    fn shipped_manifest_matches_each_engines_real_geometry() {
        let models = builtin_video_models();
        assert_eq!(
            models.len(),
            ENGINE_GEOMETRY.len(),
            "a video model was added/removed; derive its stride + area cap from its engine \
             source and add it to ENGINE_GEOMETRY — do not blanket-apply"
        );

        for (id, want_multiple, want_max_pixels) in ENGINE_GEOMETRY {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"))
                .as_object()
                .expect("model entry object");
            let limits = entry
                .get("limits")
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("{id} has limits"));

            // 1. The manifest declares exactly the stride the engine enforces.
            let declared = limits
                .get("requiresDimensionsMultipleOf")
                .and_then(Value::as_u64)
                .map(|m| m as u32);
            assert_eq!(
                declared, *want_multiple,
                "{id}: manifest stride disagrees with its engine's divisibility check"
            );
            // ...and the effective floor is what the geometry will actually apply.
            assert_eq!(
                dimension_multiple_of(entry),
                want_multiple.unwrap_or(DEFAULT_DIMENSION_MULTIPLE),
                "{id}: effective dimension multiple"
            );

            // 2. The manifest declares an area cap exactly where an engine has one. A model
            //    with no engine cap must not invent one.
            assert_eq!(
                limits.get("maxPixels").and_then(Value::as_u64),
                *want_max_pixels,
                "{id}: manifest area cap disagrees with its engine's max-area check"
            );
            assert_eq!(
                max_pixels_of(entry),
                *want_max_pixels,
                "{id}: effective max pixels"
            );

            // 3. Every advertised bucket AND the shipped default is a FIXED POINT of the
            //    geometry — advertise 848x480 and 848x480 is what renders — and fits the cap.
            let buckets = limits
                .get("resolutions")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("{id} advertises resolutions"));
            let default_res = entry
                .get("defaults")
                .and_then(Value::as_object)
                .and_then(|d| d.get("resolution"))
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{id} has a default resolution"));
            assert!(
                buckets.iter().any(|b| b.as_str() == Some(default_res)),
                "{id}: default {default_res} is not one of its advertised buckets"
            );

            for advertised in buckets
                .iter()
                .map(|b| b.as_str().expect("resolution is a string"))
                .chain(std::iter::once(default_res))
            {
                let (w, h) = advertised
                    .split_once('x')
                    .map(|(w, h)| (w.parse::<u32>().unwrap(), h.parse::<u32>().unwrap()))
                    .unwrap_or_else(|| panic!("{id}: malformed resolution {advertised}"));

                let request = VideoRequest::from_payload(&payload(json!({
                    "projectId": "p", "model": *id, "width": w, "height": h,
                    "modelManifestEntry": Value::Object(entry.clone()),
                })));
                assert_eq!(
                    (request.width, request.height),
                    (w, h),
                    "{id} advertises {advertised}; it must render {advertised}"
                );
                if let Some(cap) = want_max_pixels {
                    assert!(
                        (w as u64) * (h as u64) <= *cap,
                        "{id}: advertised {advertised} exceeds its engine's area cap {cap}"
                    );
                }
            }
        }
    }

    /// sc-12297 — the same invariant as [`shipped_manifest_matches_each_engines_real_geometry`],
    /// on the temporal axis: **what a video model advertises is what it will render.** Asserted
    /// against the REAL manifest bytes, because `hardMaxDuration` spent its whole existence with
    /// ten declarations and zero readers — a suite that hardcoded the caps it claimed to check
    /// would re-test `duration_limit_error`'s `>` and notice nothing.
    ///
    /// The table is derived from the shipped `limits.durations` dropdown, not copied from
    /// `hardMaxDuration` — that independence is what makes the `cap == max(durations)` assertion
    /// below a real check rather than a tautology.
    const DURATION_LIMITS: &[(&str, f32)] = &[
        ("ltx_2_3", 15.0),
        ("ltx_2_3_eros", 15.0),
        ("svd", 4.0),
        ("wan_2_2", 8.0),
        ("wan_2_2_t2v_14b", 5.0),
        ("wan_2_2_i2v_14b", 5.0),
        ("wan_2_2_vace_fun_14b", 5.0),
        ("bernini", 5.0),
        ("scail2_14b", 5.0),
    ];

    #[test]
    fn shipped_manifest_duration_caps_admit_everything_they_advertise() {
        let models = builtin_video_models();
        assert_eq!(
            models.len(),
            DURATION_LIMITS.len(),
            "a video model was added/removed; declare its hardMaxDuration and add it to \
             DURATION_LIMITS — an undeclared cap is silently NO cap"
        );

        for (id, want_cap) in DURATION_LIMITS {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"))
                .as_object()
                .expect("model entry object");
            let limits = entry
                .get("limits")
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("{id} has limits"));

            // 1. Every video model DECLARES a cap, and core reads exactly it. `None` here is the
            //    sc-12297 bug itself — an absent key means "no cap", which is how a 30 s request
            //    reached every one of these models.
            assert_eq!(
                hard_max_duration(entry),
                Some(*want_cap),
                "{id}: effective hard max duration"
            );

            let buckets: Vec<f64> = limits
                .get("durations")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("{id} advertises durations"))
                .iter()
                .map(|d| {
                    d.as_f64()
                        .unwrap_or_else(|| panic!("{id}: non-numeric duration"))
                })
                .collect();
            let default_duration = entry
                .get("defaults")
                .and_then(Value::as_object)
                .and_then(|d| d.get("duration"))
                .and_then(Value::as_f64)
                .unwrap_or_else(|| panic!("{id} has a default duration"));

            // 2. THE UI-PARITY INVARIANT, and the reason this change is safe to ship: the cap is
            //    exactly the top of the advertised dropdown, for all ten models. The web UI bounds
            //    duration via `limits.durations` (VideoStudio.jsx:894), so enforcing the cap cannot
            //    reject a single request the studio is capable of sending — it rejects only what
            //    the raw API / MCP / preset-replay paths send past the advertisement. A model whose
            //    cap sat BELOW its own dropdown would start rejecting its own UI, which is what
            //    this assertion exists to catch before it ships.
            let advertised_max = buckets.iter().copied().fold(f64::MIN, f64::max);
            assert_eq!(
                f64::from(*want_cap),
                advertised_max,
                "{id}: hardMaxDuration must equal the top of its advertised durations — a lower \
                 cap rejects the model's own dropdown; a higher one advertises less than it allows"
            );

            // 3. FIXED POINT: every advertised bucket and the shipped default are ADMITTED.
            //    Advertise 5s and 5s renders — the cap is `>`, so an at-cap request passes.
            for advertised in buckets
                .iter()
                .copied()
                .chain(std::iter::once(default_duration))
            {
                assert_eq!(
                    duration_limit_error(id, advertised as f32, entry),
                    None,
                    "{id} advertises {advertised}s; the cap must admit it"
                );
            }

            // 4. MUTATION CHECK: the cap actually rejects something. Without this, a cap read as
            //    `None` (the shipped bug) or a neutered `>` passes every assertion above.
            let over = *want_cap + 0.5;
            let message = duration_limit_error(id, over, entry).unwrap_or_else(|| {
                panic!("{id}: {over}s is past the {want_cap}s cap and must be rejected")
            });
            assert!(message.contains(id), "{id}: names the model: {message}");
        }
    }

    /// The house message convention (`mlx_fit_gate::too_big_error_names_model_budget_and_optional_staged`):
    /// name the model, state what was asked and what the limit is, and give the lever. A caller
    /// that hits this has no other signal — the request simply stops — so the message IS the UX.
    #[test]
    fn duration_limit_error_names_the_model_the_request_and_the_cap() {
        let mochi = payload(json!({ "limits": { "hardMaxDuration": 5 } }));
        let message =
            duration_limit_error("mochi_1", 30.0, &mochi).expect("30s is past the 5s cap");
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(message.contains("5s"), "states the cap: {message}");
        assert!(message.contains("30s"), "states what was asked: {message}");
        assert!(message.contains("Shorten"), "gives the lever: {message}");

        // At-cap and under-cap admit: the bound is `>`, matching every advertised bucket being a
        // legal request (`shipped_manifest_duration_caps_admit_everything_they_advertise`).
        assert_eq!(duration_limit_error("mochi_1", 5.0, &mochi), None);
        assert_eq!(duration_limit_error("mochi_1", 1.0, &mochi), None);
        assert!(duration_limit_error("mochi_1", 5.5, &mochi).is_some());
    }

    /// A model that declares no cap is UNCONSTRAINED — the default-absent behavior that keeps
    /// every non-declaring model (a user-manifest entry, the stub lane, an uncatalogued id whose
    /// entry resolves to `{}`) byte-identical to before the key had a reader.
    ///
    /// Also the infallibility contract: a cap nobody could satisfy falls back to NO cap rather
    /// than bricking the model. A typo'd `0` or `-1` would otherwise reject every request the
    /// model has — including its own `defaults.duration` — turning a manifest typo into a
    /// completely dead model instead of an unenforced limit.
    #[test]
    fn absent_or_unhonorable_hard_max_duration_means_no_cap() {
        for empty in [json!({}), json!({ "limits": {} })] {
            let entry = payload(empty.clone());
            assert_eq!(hard_max_duration(&entry), None, "no cap declared: {empty}");
            assert_eq!(duration_limit_error("any_model", 30.0, &entry), None);
        }

        for bad in [json!(0), json!(-1), json!("5"), json!(null), json!([5])] {
            let entry = payload(json!({ "limits": { "hardMaxDuration": bad } }));
            assert_eq!(hard_max_duration(&entry), None, "unhonorable cap {bad}");
            assert_eq!(
                duration_limit_error("any_model", 30.0, &entry),
                None,
                "an unsatisfiable cap must not brick the model: {bad}"
            );
        }

        // A fractional cap IS honorable — nothing about the key requires an integer.
        let fractional = payload(json!({ "limits": { "hardMaxDuration": 2.5 } }));
        assert_eq!(hard_max_duration(&fractional), Some(2.5));
        assert!(duration_limit_error("m", 3.0, &fractional).is_some());
        assert_eq!(duration_limit_error("m", 2.5, &fractional), None);
    }

    /// sc-12400 — **the invariant whose absence let sc-12297 ship a regression**, stated in the
    /// general form so it covers every axis at once: *the value the API resolves for an OMITTED
    /// field must be admitted by the gate the API then applies to it.*
    ///
    /// sc-12297's own conformance test asserts `duration_limit_error(id, defaults.duration) == None`
    /// — the manifest's internal fixed point — and that passed. What nothing checked was the
    /// **DTO's blanket** against the caps: 6.0 is past the cap of 7 of the 10 shipped models, and
    /// the DTO serialized it unconditionally, so a payload naming NO duration was refused for
    /// "asking for 6s". The manifest was self-consistent; the API's default simply was not part of
    /// the test.
    ///
    /// So this asserts the resolution ITSELF (`resolve_*(None, entry)`), not a manifest value — the
    /// only form that catches a blanket the model rejects.
    #[test]
    fn what_the_api_resolves_for_an_omitted_field_is_admitted_by_that_models_own_gate() {
        let models = builtin_video_models();

        for (id, ..) in FPS_MENUS {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present"))
                .as_object()
                .expect("model entry object");

            // A payload naming NOTHING — the MCP `generate_video(model, prompt)` shape.
            let bare = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": id,
                "modelManifestEntry": Value::Object(entry.clone()),
            })));

            assert_eq!(
                duration_limit_error(id, bare.duration, entry),
                None,
                "{id}: the duration resolved for an omitted field ({}s) must be admitted by the \
                 model's own cap — this is sc-12400: the blanket {DEFAULT_DURATION}s was past 7 of \
                 10 caps, so the API rejected requests that named no duration at all",
                bare.duration
            );
            assert_eq!(
                fps_limit_error(id, bare.fps, entry),
                None,
                "{id}: the fps resolved for an omitted field ({}) must be on the model's own menu",
                bare.fps
            );

            // The geometry axis has no gate to be admitted BY — dimensions coerce rather than
            // reject — so the invariant takes its strongest available form: what a size-less
            // request renders must be the model's own advertised default, snapped to its own
            // stride. That is what the web produces from the same request (`VideoStudio.jsx:234`),
            // and the blanket 768x512 is not even in `limits.resolutions` for 8 of these 10.
            //
            // The expectation comes from DEFAULT_RESOLUTIONS, NOT from `default_resolution` — see
            // that table's comment: deriving it from the function under test made this green
            // through the very bug it checks.
            let &(_, declared_w, declared_h) = DEFAULT_RESOLUTIONS
                .iter()
                .find(|(model, ..)| model == id)
                .unwrap_or_else(|| panic!("{id} listed in DEFAULT_RESOLUTIONS"));
            assert_eq!(
                default_resolution(entry),
                Some((declared_w, declared_h)),
                "{id}: core must read the resolution the manifest actually declares"
            );

            let (want_w, want_h) = (declared_w, declared_h);
            let multiple = dimension_multiple_of(entry);
            let (want_w, want_h) = (
                floor_to_multiple(want_w, multiple),
                floor_to_multiple(want_h, multiple),
            );
            let (want_w, want_h) = match max_pixels_of(entry) {
                Some(cap) if u64::from(want_w) * u64::from(want_h) > cap => {
                    fit_to_max_pixels(want_w, want_h, multiple, cap)
                }
                _ => (want_w, want_h),
            };
            assert_eq!(
                (bare.width, bare.height),
                (want_w, want_h),
                "{id}: a size-less request must render the model's declared defaults.resolution, \
                 not the blanket {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}"
            );

            // MUTATION CHECK: the assertions above are only meaningful where the blanket would have
            // FAILED. Pin that this model is one of those, or that it is a known survivor.
            let cap = hard_max_duration(entry).unwrap_or_else(|| panic!("{id} declares a cap"));
            if DEFAULT_DURATION > cap {
                assert!(
                    duration_limit_error(id, DEFAULT_DURATION, entry).is_some(),
                    "{id}: fixture assumes the blanket is over-cap here"
                );
                assert!(
                    bare.duration < DEFAULT_DURATION,
                    "{id}: resolution must have MOVED off the blanket, not merely passed"
                );
            }
        }

        // The blanket duration was over-cap for a MAJORITY of shipped models — the same 7-of-10
        // shape as the fps blanket, and the reason this is a generalized invariant rather than two
        // coincidences. Pinned as a number so reintroducing a blanket default fails here.
        let over_cap: Vec<&str> = FPS_MENUS
            .iter()
            .filter_map(|(id, ..)| {
                let entry = models
                    .iter()
                    .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))?
                    .as_object()?;
                (hard_max_duration(entry)? < DEFAULT_DURATION).then_some(*id)
            })
            .collect();
        assert_eq!(
            over_cap.len(),
            6,
            "the blanket {DEFAULT_DURATION}s is past the cap of these models: {over_cap:?} — if this \
             count moved, re-read why resolve_duration consults defaults.duration before changing it"
        );

        // …and the geometry blanket is UNADVERTISED — not merely un-declared — for 8 of the 10:
        // 768x512 is absent from their `limits.resolutions` entirely, so the API was rendering a
        // size the model never offered anywhere. Pinned so reintroducing a blanket fails here.
        let unadvertised: Vec<&str> = FPS_MENUS
            .iter()
            .filter_map(|(id, ..)| {
                let entry = models
                    .iter()
                    .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))?
                    .as_object()?;
                let blanket = format!("{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}");
                let advertises = entry
                    .get("limits")?
                    .as_object()?
                    .get("resolutions")?
                    .as_array()?
                    .iter()
                    .any(|r| r.as_str() == Some(blanket.as_str()));
                (!advertises).then_some(*id)
            })
            .collect();
        assert_eq!(
            unadvertised.len(),
            7,
            "the blanket {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} is not advertised by these models: \
             {unadvertised:?}"
        );
    }

    /// [`resolve_duration`] — the duration twin of
    /// `omitted_fps_resolves_to_the_models_declared_default`.
    #[test]
    fn omitted_duration_resolves_to_the_models_declared_default() {
        let mochi = payload(json!({
            "limits": { "hardMaxDuration": 5 }, "defaults": { "duration": 5 }
        }));

        // Omitted / unreadable ⇒ the model's declared default, NOT the blanket 6.0 — which this
        // model's own cap REJECTS, which is the entire bug.
        assert_eq!(resolve_duration(None, &mochi), 5.0);
        assert_eq!(resolve_duration(Some(&json!(null)), &mochi), 5.0);
        assert_eq!(resolve_duration(Some(&json!("nonsense")), &mochi), 5.0);
        assert_eq!(
            duration_limit_error("mochi_1", resolve_duration(None, &mochi), &mochi),
            None,
            "the resolved default must be admitted by the cap that resolved it"
        );
        assert!(
            duration_limit_error("mochi_1", DEFAULT_DURATION, &mochi).is_some(),
            "…and the blanket would NOT have been — sc-12400"
        );

        // A numeric STRING is a value the caller NAMED (the `safe_float` contract). Pinned with 3,
        // which is neither the blanket (6) nor this model's default (5), so falling back to either
        // is RED.
        assert_eq!(resolve_duration(Some(&json!("3")), &mochi), 3.0);
        assert_eq!(resolve_duration(Some(&json!(4.5)), &mochi), 4.5);

        // Named ⇒ honored verbatim, even past the cap. The gate refuses it; resolution does not
        // quietly shorten it — reject, never clamp.
        assert_eq!(resolve_duration(Some(&json!(30)), &mochi), 30.0);
        assert!(duration_limit_error("mochi_1", 30.0, &mochi).is_some());

        // Named-but-nonsense clamps to the sanity bounds (unchanged), then the cap judges.
        assert_eq!(resolve_duration(Some(&json!(999)), &mochi), MAX_DURATION);
        assert_eq!(resolve_duration(Some(&json!(0)), &mochi), MIN_DURATION);

        // No declared default ⇒ the blanket, so a model that declares nothing is unchanged.
        let bare = payload(json!({}));
        assert_eq!(resolve_duration(None, &bare), DEFAULT_DURATION);
        assert_eq!(default_duration(&bare), None);

        // An unhonorable default is not a length anyone meant ⇒ the blanket.
        for bad in [json!(0), json!(-1), json!(31), json!("5"), json!(null)] {
            let entry = payload(json!({ "defaults": { "duration": bad } }));
            assert_eq!(default_duration(&entry), None, "unhonorable default {bad}");
            assert_eq!(resolve_duration(None, &entry), DEFAULT_DURATION);
        }
    }

    /// sc-12347 — the fps axis of the same invariant, derived from the shipped `limits.fps` menus
    /// and the `defaults.fps` each model declares. `THE_BLANKET_FPS` is the value `dto.rs` used to
    /// default to; it is listed here so assertion 5 can prove *why* [`resolve_fps`] must consult
    /// the manifest, rather than that claim living only in a comment.
    const FPS_MENUS: &[(&str, &[u32], u32)] = &[
        ("ltx_2_3", &[24, 25, 30], 25),
        ("ltx_2_3_eros", &[24, 25, 30], 25),
        ("svd", &[6, 7, 8, 10, 12, 25], 7),
        ("wan_2_2", &[16, 24], 24),
        ("wan_2_2_t2v_14b", &[16], 16),
        ("wan_2_2_i2v_14b", &[16], 16),
        ("wan_2_2_vace_fun_14b", &[16], 16),
        ("bernini", &[16], 16),
        ("scail2_14b", &[16], 16),
    ];

    /// The fps the API blanket-defaulted to before sc-12347, kept as a named constant because the
    /// point of assertion 5 is that this number is *wrong for most models* — deleting the constant
    /// would delete the regression guard.
    const THE_BLANKET_FPS: u32 = 25;

    /// Each shipped model's declared `defaults.resolution`, TRANSCRIBED from the manifest — and
    /// deliberately independent of [`default_resolution`], which is the function under test.
    ///
    /// Deriving the expectation by calling `default_resolution` instead made the geometry assertion
    /// a tautology (`f(x) == f(x)`): stubbing the function to `None` moved the expected value AND
    /// the actual value to the blanket together, so the test stayed GREEN through the exact bug it
    /// exists to catch. Only the API's end-to-end test — which hardcodes 848x480 — was red. This
    /// table is what makes the core half a real check.
    const DEFAULT_RESOLUTIONS: &[(&str, u32, u32)] = &[
        ("ltx_2_3", 768, 512),
        ("ltx_2_3_eros", 768, 512),
        ("svd", 1024, 576),
        ("wan_2_2", 832, 480),
        // 720, not 704: sc-12308 (#1581) restored TRUE 720p to the A14B pair by lifting maxPixels
        // to the 14B's real 921,600 (sc-12294 had walked the manifest down to the 5B's 901,120).
        // This table caught that drift in CI when main moved under this branch — which is the
        // table's whole purpose: transcribed from the manifest, so a manifest change forces a
        // conscious update here instead of passing silently.
        ("wan_2_2_t2v_14b", 1280, 720),
        ("wan_2_2_i2v_14b", 1280, 720),
        ("wan_2_2_vace_fun_14b", 832, 480),
        ("bernini", 848, 480),
        ("scail2_14b", 832, 480),
    ];

    /// sc-12347 — the temporal invariant's other half: **what a video model advertises is what it
    /// renders**, on the axis that actually multiplies into frame count.
    ///
    /// Same four-part skeleton as `shipped_manifest_duration_caps_admit_everything_they_advertise`,
    /// against the REAL manifest bytes for the same reason: `limits.fps` spent its whole existence
    /// with ten declarations and zero Rust readers, so a suite that hardcoded the menus it claims to
    /// check would re-test `Vec::contains` and notice nothing.
    #[test]
    fn shipped_manifest_fps_menus_admit_everything_they_advertise() {
        let models = builtin_video_models();
        assert_eq!(
            models.len(),
            FPS_MENUS.len(),
            "a video model was added/removed; declare its limits.fps + defaults.fps and add it to \
             FPS_MENUS — an undeclared menu is silently NO menu"
        );

        for (id, want_menu, want_default) in FPS_MENUS {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"))
                .as_object()
                .expect("model entry object");

            // 1. Every video model DECLARES a menu, and core reads exactly it. `None` here is the
            //    sc-12347 bug itself — an absent menu means "any fps", which is how a 60 fps
            //    request reached a model that advertises only 30.
            assert_eq!(
                allowed_fps(entry),
                Some(want_menu.to_vec()),
                "{id}: effective fps menu"
            );

            // 2. FIXED POINT: the model's own declared default is on its own menu. This is the
            //    assertion the story asks for, and it is what makes `resolve_fps` safe — resolving
            //    an omitted fps from `defaults.fps` must never produce a value this gate refuses.
            assert_eq!(
                default_fps(entry),
                Some(*want_default),
                "{id}: effective default fps"
            );
            assert!(
                want_menu.contains(want_default),
                "{id}: defaults.fps {want_default} must be on its own limits.fps menu {want_menu:?}"
            );

            // 3. Every advertised rate is ADMITTED — the menu IS the web's dropdown source
            //    (`VideoStudio.jsx:912`), so anything the picker can emit must pass the gate.
            for advertised in want_menu.iter().copied().chain([*want_default]) {
                assert_eq!(
                    fps_limit_error(id, advertised, entry),
                    None,
                    "{id} advertises {advertised} fps; the menu must admit it"
                );
            }

            // 4. MUTATION CHECK: the menu actually rejects something. Assertions 1-3 all pass if
            //    `allowed_fps` returns `None` (the shipped bug) or the membership test is neutered.
            //    61 is past the blanket sanity bound, so it is off EVERY shipped menu.
            let off_menu = 61;
            assert!(
                !want_menu.contains(&off_menu),
                "{id}: fixture assumes {off_menu} is off-menu"
            );
            let message = fps_limit_error(id, off_menu, entry)
                .unwrap_or_else(|| panic!("{id}: {off_menu} fps is off-menu and must be rejected"));
            assert!(message.contains(id), "{id}: names the model: {message}");

            // 5. THE REASON `resolve_fps` READS THE MANIFEST: a payload that names NO fps must
            //    resolve to a rate this model's own menu admits. Enforcing `limits.fps` while
            //    defaulting to a blanket the menu rejects would 400 the API's own default — which
            //    is exactly what the pre-sc-12347 blanket did on 7 of these 10 models.
            let resolved = resolve_fps(None, entry);
            assert_eq!(
                resolved, *want_default,
                "{id}: omitted fps takes the model's default"
            );
            assert_eq!(
                fps_limit_error(id, resolved, entry),
                None,
                "{id}: an fps-less payload must be ADMITTED, not rejected"
            );
        }

        // The blanket the DTO used to apply is off-menu for a MAJORITY of shipped models — the
        // finding that made "enforce limits.fps" unshippable on its own. Pinned as a number rather
        // than prose so re-introducing a blanket default fails here.
        let rejecting_the_blanket: Vec<&str> = FPS_MENUS
            .iter()
            .filter(|(_, menu, _)| !menu.contains(&THE_BLANKET_FPS))
            .map(|(id, _, _)| *id)
            .collect();
        assert_eq!(
            rejecting_the_blanket.len(),
            6,
            "the blanket {THE_BLANKET_FPS} fps is off-menu for these models: {rejecting_the_blanket:?} \
             — if this count moved, re-read why resolve_fps consults defaults.fps before changing it"
        );
    }

    /// sc-12347 + sc-12297 together — **the frame ceiling the story's Option B would have built
    /// explicitly, obtained for free as a derived property.**
    ///
    /// Neither gate bounds frames alone: the duration cap admits 5s @ 60fps (301 mochi frames), and
    /// the fps menu admits 30fps @ 30s (901). Together they pin `frames ≤ hardMaxDuration ×
    /// max(limits.fps)` for every shipped model, which is exactly B's ceiling — so A subsumes B
    /// rather than trading against it.
    ///
    /// ⚠️ This is a CONTRACT bound, NOT a safety gate. Mochi's 151 is still ~4× past the ~36-frame
    /// correctness ceiling that MLX's `i32::MAX` element limit imposes (sc-12349), and no amount of
    /// honoring the manifest closes that — see `mlx-gen-mochi`'s `MAX_TENSOR_ELEMS`.
    #[test]
    fn the_two_gates_together_bound_frames_at_the_advertised_ceiling() {
        let models = builtin_video_models();

        for (id, menu, _) in FPS_MENUS {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present"))
                .as_object()
                .expect("model entry object");
            let cap = hard_max_duration(entry).unwrap_or_else(|| panic!("{id} declares a cap"));
            let fastest = menu.iter().copied().max().expect("non-empty menu");
            let ceiling = (cap * fastest as f32).round() as u32;

            // Anything BOTH gates admit lands at or under the derived ceiling.
            for fps in menu.iter() {
                let request = VideoRequest::from_payload(&payload(json!({
                    "projectId": "p", "model": id, "duration": cap, "fps": fps,
                    "modelManifestEntry": Value::Object(entry.clone()),
                })));
                assert_eq!(duration_limit_error(id, request.duration, entry), None);
                assert_eq!(fps_limit_error(id, request.fps, entry), None);
                assert!(
                    request.raw_frame_count() <= ceiling,
                    "{id}: {cap}s @ {fps}fps = {} raw frames, past the advertised ceiling {ceiling}",
                    request.raw_frame_count()
                );
            }

            // MUTATION CHECK: the ceiling binds because BOTH gates hold. Drop either one and the
            // frame count blows past it — which is the whole argument for this story existing
            // alongside sc-12297.
            let over_fps = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": id, "duration": cap, "fps": 60,
                "modelManifestEntry": Value::Object(entry.clone()),
            })));
            if fastest < 60 {
                assert!(
                    over_fps.raw_frame_count() > ceiling,
                    "{id}: 60fps must exceed the ceiling for this test to mean anything"
                );
                assert!(
                    fps_limit_error(id, over_fps.fps, entry).is_some(),
                    "{id}: only the FPS gate refuses {cap}s @ 60fps — the duration cap admits it"
                );
                assert_eq!(
                    duration_limit_error(id, over_fps.duration, entry),
                    None,
                    "{id}: the duration cap admits this request; it is legally {cap}s"
                );
            }
        }

        // The concrete case from the story: mochi_1, a *legally 5-second* request at 60fps.
        let mochi = payload(json!({
            "limits": { "hardMaxDuration": 5, "fps": [30] }, "defaults": { "fps": 30 }
        }));
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "duration": 5, "fps": 60,
            "modelManifestEntry": Value::Object(mochi.clone()),
        })));
        assert_eq!(request.raw_frame_count(), 300);
        assert_eq!(
            request.frame_count(),
            301,
            "301 % 6 == 1, so the engine accepts it"
        );
        assert_eq!(
            duration_limit_error("mochi_1", request.duration, &mochi),
            None,
            "the request IS 5 seconds — sc-12297's cap is doing exactly what it says"
        );
        assert!(
            fps_limit_error("mochi_1", request.fps, &mochi).is_some(),
            "301 frames from a legal 5s request is refused only by the fps menu"
        );

        // …and the shipped default is the documented 151.
        let default_request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "duration": 5,
            "modelManifestEntry": Value::Object(mochi),
        })));
        assert_eq!(
            default_request.fps, 30,
            "omitted fps takes mochi's declared 30"
        );
        assert_eq!(default_request.frame_count(), 151);
    }

    /// The house message convention: name the model, state what was asked and what is allowed, and
    /// give the lever. A caller that hits this has no other signal — the request simply stops.
    #[test]
    fn fps_limit_error_names_the_model_the_request_and_the_menu() {
        let mochi = payload(json!({ "limits": { "fps": [30] } }));
        let message = fps_limit_error("mochi_1", 60, &mochi).expect("60 is off mochi's [30] menu");
        assert!(message.contains("mochi_1"), "names the model: {message}");
        assert!(
            message.contains("30 fps"),
            "states what is allowed: {message}"
        );
        assert!(
            message.contains("60 fps"),
            "states what was asked: {message}"
        );
        assert!(message.contains("Set fps to"), "gives the lever: {message}");

        // Every advertised rate admits; anything else is refused — membership, not a range. 24 and
        // 30 bracket 25, so a `>=`/`<=` mutation of the check cannot pass this.
        let ltx = payload(json!({ "limits": { "fps": [24, 25, 30] } }));
        for advertised in [24, 25, 30] {
            assert_eq!(fps_limit_error("ltx_2_3", advertised, &ltx), None);
        }
        for off_menu in [23, 26, 29, 31] {
            assert!(
                fps_limit_error("ltx_2_3", off_menu, &ltx).is_some(),
                "{off_menu} is between/around advertised rates but is not one of them"
            );
        }

        // The menu is humanized per arity: one value, two values, and a long list each read as
        // English rather than as a debug-printed Vec.
        assert!(
            fps_limit_error("m", 1, &payload(json!({ "limits": { "fps": [30] } })))
                .expect("off-menu")
                .contains("renders at 30 fps")
        );
        assert!(
            fps_limit_error("m", 1, &payload(json!({ "limits": { "fps": [16, 24] } })))
                .expect("off-menu")
                .contains("renders at 16 or 24 fps")
        );
        assert!(fps_limit_error(
            "m",
            1,
            &payload(json!({ "limits": { "fps": [6, 7, 8, 10, 12, 25] } }))
        )
        .expect("off-menu")
        .contains("renders at 6, 7, 8, 10, 12, or 25 fps"));
    }

    /// A model that declares no menu is UNCONSTRAINED — the default-absent behavior that keeps
    /// every non-declaring model (a user-manifest entry, the stub lane, an uncatalogued id whose
    /// entry resolves to `{}`) byte-identical to before the key had a reader.
    ///
    /// Also the infallibility contract: a menu nobody could satisfy falls back to NO menu rather
    /// than bricking the model. An empty array, or one whose every entry is unusable, would
    /// otherwise reject every request the model has — including its own `defaults.fps`.
    #[test]
    fn absent_or_unhonorable_limits_fps_means_no_menu() {
        for empty in [json!({}), json!({ "limits": {} })] {
            let entry = payload(empty.clone());
            assert_eq!(allowed_fps(&entry), None, "no menu declared: {empty}");
            assert_eq!(fps_limit_error("any_model", 60, &entry), None);
        }

        for bad in [
            json!([]),
            json!(30),
            json!("30"),
            json!(null),
            json!({ "min": 30 }),
        ] {
            let entry = payload(json!({ "limits": { "fps": bad } }));
            assert_eq!(allowed_fps(&entry), None, "unhonorable menu {bad}");
            assert_eq!(
                fps_limit_error("any_model", 60, &entry),
                None,
                "an unsatisfiable menu must not brick the model: {bad}"
            );
        }

        // A menu whose every entry is out of the sanity bounds is no menu at all — but a menu with
        // ONE usable entry keeps that entry and drops the junk, rather than falling back to
        // unconstrained. Dropping junk is not the same decision as having no menu.
        let all_junk = payload(json!({ "limits": { "fps": [0, 61, "30", null] } }));
        assert_eq!(allowed_fps(&all_junk), None);
        assert_eq!(fps_limit_error("m", 60, &all_junk), None);

        let partly_junk = payload(json!({ "limits": { "fps": [0, 30, 999] } }));
        assert_eq!(allowed_fps(&partly_junk), Some(vec![30]));
        assert_eq!(fps_limit_error("m", 30, &partly_junk), None);
        assert!(fps_limit_error("m", 24, &partly_junk).is_some());
    }

    /// [`resolve_fps`] — resolution, not coercion. An omitted fps takes the model's declared
    /// default; a NAMED fps is never rewritten onto the menu, only clamped to the sanity bounds and
    /// then judged by [`fps_limit_error`].
    ///
    /// The named-value half is the reject-never-clamp rule: silently snapping a 60 fps request onto
    /// mochi's 30 would halve the clip the caller asked for, which is the same silent-coercion class
    /// as the 848→832 rewrite.
    #[test]
    fn omitted_fps_resolves_to_the_models_declared_default() {
        let mochi = payload(json!({ "limits": { "fps": [30] }, "defaults": { "fps": 30 } }));

        // Omitted / unreadable ⇒ the model's declared default, NOT the blanket 25. Asserted
        // against a menu whose default (30) differs from the blanket, so a `resolve_fps` that
        // ignored the manifest would be RED here rather than coincidentally right.
        assert_eq!(resolve_fps(None, &mochi), 30);
        assert_eq!(resolve_fps(Some(&json!(null)), &mochi), 30);
        assert_eq!(resolve_fps(Some(&json!("nonsense")), &mochi), 30);

        // A numeric STRING is a value the caller NAMED, not an omission — the pre-existing
        // `safe_int` contract (`reads_numeric_strings_and_carries_advanced_fields`). Pinned with 24,
        // which is neither the blanket nor this model's default, so falling back to either is RED.
        assert_eq!(resolve_fps(Some(&json!("24")), &mochi), 24);
        assert!(
            fps_limit_error("mochi_1", 24, &mochi).is_some(),
            "and it is then judged by the menu like any named value"
        );

        // Named ⇒ honored verbatim, even when off-menu. The gate refuses it; resolution does not
        // quietly fix it up.
        assert_eq!(resolve_fps(Some(&json!(60)), &mochi), 60);
        assert!(fps_limit_error("mochi_1", 60, &mochi).is_some());

        // Named-but-nonsense clamps to the sanity bounds (unchanged behavior), then the menu judges.
        assert_eq!(resolve_fps(Some(&json!(999)), &mochi), 60);
        assert_eq!(resolve_fps(Some(&json!(0)), &mochi), 1);

        // No declared default ⇒ the blanket, so a model that declares nothing is unchanged.
        let bare = payload(json!({}));
        assert_eq!(resolve_fps(None, &bare), THE_BLANKET_FPS);
        assert_eq!(default_fps(&bare), None);

        // An unhonorable default is not a rate anyone meant ⇒ the blanket, rather than propagating
        // a degenerate value into `frames = duration × fps`.
        for bad in [json!(0), json!(61), json!(-1), json!("30"), json!(null)] {
            let entry = payload(json!({ "defaults": { "fps": bad } }));
            assert_eq!(default_fps(&entry), None, "unhonorable default {bad}");
            assert_eq!(resolve_fps(None, &entry), THE_BLANKET_FPS);
        }

        // The live bug this closes, end to end: mochi at the blanket renders a 30fps prior 20% slow.
        let before = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "duration": 5,
            "modelManifestEntry": json!({ "limits": { "fps": [30] } }),
        })));
        assert_eq!(
            before.fps, THE_BLANKET_FPS,
            "no declared default ⇒ the old behavior"
        );
        assert_eq!(
            before.frame_count(),
            127,
            "5 × 25 = 125 → 127, and 127 % 6 == 1"
        );

        let after = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "duration": 5,
            "modelManifestEntry": Value::Object(mochi),
        })));
        assert_eq!(after.fps, 30, "the declared default is honored");
        assert_eq!(
            after.frame_count(),
            151,
            "the frame count the manifest documents"
        );
    }

    /// sc-12294 BLOCKER — the case the dropdown never produces but every raw path does.
    ///
    /// `1280x720` is the shipped `defaults.resolution` of the A14B T2V/I2V entries, so it is
    /// the single most common stored value. On `main` a blanket ÷32 floored it to 1280x704 by
    /// accident. Declaring ÷16 removes that accidental protection — so each model's cap must be
    /// enforced, or job retry / MCP / `POST /api/v1/jobs` / preset replay would hand an engine a
    /// geometry it rejects.
    ///
    /// sc-12308 changed what "legal" means per model, and this test spans both answers: the ÷16
    /// 14B family now renders 1280x720 as-is (it is exactly at their 921,600 cap), while the ÷32
    /// models (the 5B, scail2) still floor it to 1280x704 on their stride. Same stored value,
    /// two legal outcomes — which is the point of judging against [`ENGINE_GEOMETRY`] per model.
    ///
    /// Driven through the REAL manifest entries, but judged against [`ENGINE_GEOMETRY`] — the
    /// engine's truth, never the manifest's own claim. Reading the cap back out of the entry
    /// would make this vacuous exactly when it matters: delete a `maxPixels` and the area
    /// assertion would simply stop running. Judged independently, a deleted cap is RED here.
    #[test]
    fn stored_720p_is_legal_on_every_video_model() {
        for model in builtin_video_models() {
            let entry = model.as_object().expect("model entry object");
            let id = entry.get("id").and_then(Value::as_str).expect("id");
            let (_, want_multiple, want_max_pixels) = ENGINE_GEOMETRY
                .iter()
                .find(|(known, _, _)| known == &id)
                .unwrap_or_else(|| panic!("{id} covered by ENGINE_GEOMETRY"));

            let request = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": id, "width": 1280, "height": 720,
                "modelManifestEntry": model.clone(),
            })));

            let multiple = want_multiple.unwrap_or(DEFAULT_DIMENSION_MULTIPLE);
            assert_eq!(
                (request.width % multiple, request.height % multiple),
                (0, 0),
                "{id}: stored 1280x720 -> {}x{} is off the ÷{multiple} lattice its engine checks",
                request.width,
                request.height
            );
            if let Some(cap) = want_max_pixels {
                assert!(
                    (request.width as u64) * (request.height as u64) <= *cap,
                    "{id}: stored 1280x720 -> {}x{} = {} px exceeds the engine cap {cap} — \
                     both backends would now reject it",
                    request.width,
                    request.height,
                    (request.width as u64) * (request.height as u64)
                );
            }
        }
    }

    /// The fit normalizes a genuinely over-cap CUSTOM request into a legal geometry, keeping the
    /// shape of the reference's `_best_output_size` (aspect-preserving, grid-aligned, shrink-only).
    ///
    /// **sc-12308 reframed why this exists.** It used to be justified as mirroring mlx's
    /// `best_output_size` so the app "agreed with the backend that silently rescales". Neither
    /// backend rescales any more — both reject over-cap geometry — so this is now *app-layer
    /// normalization*: it is what keeps a custom over-cap request from reaching an engine that would
    /// reject it. It is no longer a mirror of engine code, so it is pinned on its own contract.
    ///
    /// Crucially it must NOT fire for anything advertised: `shipped_manifest_matches_each_engines_real_geometry`
    /// pins every bucket as a fixed point, and the 14B family's `1280x720` is exactly at its cap.
    #[test]
    fn area_fit_normalizes_only_genuinely_over_cap_requests() {
        // 14B family: the canonical 720p sits AT the cap (the check is `>`), so it is untouched.
        // This is the sc-12308 regression — it used to be refit to 1264x704, off every bucket.
        assert_eq!(1280 * 720, 921_600);
        assert_eq!(fit_to_max_pixels(1280, 720, 16, 921_600), (1280, 720));
        let at_cap = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "wan_2_2_i2v_14b", "width": 1280, "height": 720,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16, "maxPixels": 921_600 } }
        })));
        assert_eq!(
            (at_cap.width, at_cap.height),
            (1280, 720),
            "the 14B family's canonical 720p must survive normalization untouched"
        );

        // A genuinely over-cap custom request IS normalized: aspect-preserving, grid-aligned,
        // shrink-only, and inside the cap.
        let (w, h) = fit_to_max_pixels(1280, 1024, 16, 921_600);
        assert!((w as u64) * (h as u64) <= 921_600, "fits the cap");
        assert_eq!((w % 16, h % 16), (0, 0), "lands on the declared lattice");
        assert!(w <= 1280 && h <= 1024, "never upscales");
        let (orig, fitted) = (1280.0 / 1024.0, w as f64 / h as f64);
        assert!(
            (orig - fitted).abs() / orig < 0.05,
            "aspect preserved within 5%: {orig} vs {fitted}"
        );

        // The 5B keeps its own smaller budget, so ITS 720p is 704-tall and also a fixed point.
        assert_eq!(1280 * 704, 901_120);
        assert_eq!(fit_to_max_pixels(1280, 704, 32, 901_120), (1280, 704));
    }

    /// A model that declares no `maxPixels` is UNCONSTRAINED — the default-absent behavior that
    /// keeps every non-declaring model byte-identical to before the key existed. Also covers the
    /// infallibility contract: a typo'd cap the geometry could never honor falls back to no cap
    /// rather than emitting a degenerate size.
    #[test]
    fn absent_or_unhonorable_max_pixels_means_no_cap() {
        // No `maxPixels` → 1280x720 floors on stride alone and is NOT area-fitted.
        let uncapped = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 1280, "height": 720,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((uncapped.width, uncapped.height), (1280, 720));
        assert!((uncapped.width as u64) * (uncapped.height as u64) > 901_120);

        // Same entry + a cap → fitted. This is what makes the assertion above meaningful.
        let capped = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 1280, "height": 720,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16, "maxPixels": 901_120 } }
        })));
        assert_eq!((capped.width, capped.height), (1264, 704));

        // Unhonorable caps fall back to NO cap (never a degenerate size): the 256 floor means
        // nothing under 256×256 is satisfiable.
        for bad in [
            json!(0),
            json!(-1),
            json!(1.5),
            json!("901120"),
            json!(65_535),
        ] {
            let entry = payload(json!({ "limits": { "maxPixels": bad } }));
            assert_eq!(max_pixels_of(&entry), None, "unhonorable maxPixels {bad}");
        }
        // The smallest honorable cap is exactly the 256x256 floor.
        let floor = payload(json!({ "limits": { "maxPixels": 65_536 } }));
        assert_eq!(max_pixels_of(&floor), Some(65_536));
    }

    #[test]
    fn malformed_dimension_multiple_falls_back_to_32_without_panicking() {
        // `value % 0` panics; `from_payload` is infallible and must never crash on a
        // typo'd/hostile manifest entry. Non-`u64` shapes fall back to the 32 default
        // (the manifest is machine-authored — `as_u64`, matching the engines.rs reader).
        for bogus in [
            json!(0),
            json!(-16),
            json!("16"),
            json!(null),
            json!({}),
            json!(1.5),
        ] {
            let request = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "width": 1000, "height": 543,
                "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": bogus } }
            })));
            assert_eq!((request.width, request.height), (992, 512), "bogus={bogus}");
        }

        // `limits` present but not an object → default.
        let odd = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "width": 1000, "modelManifestEntry": { "limits": 7 }
        })));
        assert_eq!(odd.width, 992);
    }

    #[test]
    fn dimension_multiple_that_cannot_divide_256_falls_back_to_32() {
        // `floor_to_multiple` clamps up to a hard floor of 256, so a multiple that does
        // not divide 256 breaks the very invariant it declares: without this guard,
        // `1024` yielded 256x256 (`256 % 1024 == 256`) and nothing fell back. Magnitude
        // is not the tell — `96` is small and still off-lattice (`256 % 96 == 64`).
        for bogus in [
            json!(1024),
            json!(4096),
            json!(100_000),
            json!(96),
            json!(48),
        ] {
            let request = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "width": 1000, "height": 543,
                "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": bogus } }
            })));
            assert_eq!(
                (request.width, request.height),
                (992, 512),
                "bogus={bogus} must fall back to the 32 default"
            );
        }

        // The accepted set — every power of two dividing 256 — is honored, and the
        // resulting geometry is always on-lattice.
        for good in [1u32, 2, 4, 8, 16, 32, 64, 128, 256] {
            let request = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "width": 1000, "height": 543,
                "modelManifestEntry": {
                    "limits": { "requiresDimensionsMultipleOf": good }
                }
            })));
            assert_eq!(request.width % good, 0, "width off-lattice for {good}");
            assert_eq!(request.height % good, 0, "height off-lattice for {good}");
        }
    }

    #[test]
    fn reads_numeric_strings_and_carries_advanced_fields() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p",
            "duration": "4.5",
            "fps": "30",
            "seed": "42",
            "sourceAssetId": "asset_src",
            "personTrackId": "track_1",
            "lastFrameAssetId": "asset_last",
            "advanced": { "guidanceScale": 4.0, "steps": 20 },
            "modelManifestEntry": { "family": "ltx-video", "repo": "x" },
            "loras": [{ "path": "a.safetensors", "weight": 0.8 }]
        })));
        assert_eq!(request.duration, 4.5);
        assert_eq!(request.fps, 30);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.source_asset_id.as_deref(), Some("asset_src"));
        assert_eq!(request.person_track_id.as_deref(), Some("track_1"));
        assert_eq!(request.last_frame_asset_id.as_deref(), Some("asset_last"));
        assert_eq!(request.advanced.get("steps"), Some(&json!(20)));
        assert_eq!(
            request.model_manifest_entry.get("family"),
            Some(&json!("ltx-video"))
        );
        assert_eq!(request.loras.len(), 1);
    }

    #[test]
    fn parses_reference_asset_ids_dropping_blanks() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p",
            "mode": "reference_video_to_video",
            "sourceClipAssetId": "clip-a",
            "referenceAssetIds": ["ref-1", "  ", "ref-2", 42]
        })));
        assert_eq!(request.mode, "reference_video_to_video");
        assert_eq!(request.source_clip_asset_id.as_deref(), Some("clip-a"));
        assert_eq!(request.reference_asset_ids, vec!["ref-1", "ref-2"]);

        // Absent → empty, never a panic.
        let bare = VideoRequest::from_payload(&payload(json!({ "projectId": "p" })));
        assert!(bare.reference_asset_ids.is_empty());
    }

    #[test]
    fn parses_mv2v_and_ads2v_multi_source_fields() {
        // mv2v: multiple source clips, blanks/non-strings dropped.
        let mv2v = VideoRequest::from_payload(&payload(json!({
            "projectId": "p",
            "mode": "multi_video_to_video",
            "sourceClipAssetIds": ["clip-a", "  ", "clip-b", 7]
        })));
        assert_eq!(mv2v.source_clip_asset_ids, vec!["clip-a", "clip-b"]);

        // ads2v: source clip + reference clip + reference images.
        let ads2v = VideoRequest::from_payload(&payload(json!({
            "projectId": "p",
            "mode": "ads2v",
            "sourceClipAssetId": "clip-src",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["ref-1"]
        })));
        assert_eq!(ads2v.source_clip_asset_id.as_deref(), Some("clip-src"));
        assert_eq!(ads2v.reference_clip_asset_id.as_deref(), Some("clip-ref"));
        assert_eq!(ads2v.reference_asset_ids, vec!["ref-1"]);

        // Absent → empty / None, never a panic.
        let bare = VideoRequest::from_payload(&payload(json!({ "projectId": "p" })));
        assert!(bare.source_clip_asset_ids.is_empty());
        assert!(bare.reference_clip_asset_id.is_none());
    }

    #[test]
    fn raw_frame_count_rounds_duration_times_fps() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "stub", "duration": 2.0, "fps": 24
        })));
        assert_eq!(request.raw_frame_count(), 48);
        // An id off the named families takes the ladder's WAN fallback, NOT the raw count
        // (sc-12371 — see `video_frame_count`: that arm is where the real Wan engines carrying
        // non-Wan ids live, so it cannot be the raw arm it used to be). For a genuinely unknown id
        // like this one the only thing it decides is the procedural stub's placeholder length, and
        // that stays self-consistent either way: `generate_stub_video` renders `frame_count()`
        // frames and `stub_raw_settings` records the very same call.
        assert_eq!(request.frame_count(), wan_frame_count(48));
        assert_eq!(request.frame_count(), 45);
    }

    #[test]
    fn ltx_frame_count_snaps_to_8k_plus_1() {
        // Exact 8k+1 values are unchanged.
        assert_eq!(ltx_frame_count(9), 9);
        assert_eq!(ltx_frame_count(17), 17);
        assert_eq!(ltx_frame_count(121), 121);
        // Below the floor snaps up to 9.
        assert_eq!(ltx_frame_count(1), 9);
        assert_eq!(ltx_frame_count(8), 9);
        // Nearest wins; ties go to the lower 8k+1.
        assert_eq!(ltx_frame_count(13), 9); // |13-9|=4 == |17-13|=4 -> lower
        assert_eq!(ltx_frame_count(14), 17); // closer to 17
                                             // A real default: 6s * 25fps = 150 -> nearest of 145 / 153 -> 153.
        assert_eq!(ltx_frame_count(150), 153);
    }

    #[test]
    fn wan_frame_count_floors_to_4n_plus_1_min_5() {
        // Exact 1+4k values >= 5 are unchanged.
        assert_eq!(wan_frame_count(5), 5);
        assert_eq!(wan_frame_count(81), 81);
        // Floors to the 1+4k at or below raw (Python `raw - ((raw-1)%4)`).
        assert_eq!(wan_frame_count(48), 45); // 48-(47%4=3)
        assert_eq!(wan_frame_count(83), 81); // 83-(82%4=2)
        assert_eq!(wan_frame_count(8), 5); // 8-(7%4=3)
                                           // 5-frame floor for tiny counts.
        assert_eq!(wan_frame_count(1), 5);
        assert_eq!(wan_frame_count(4), 5);
    }

    #[test]
    fn mochi_frame_count_snaps_to_6k_plus_1_min_7() {
        // Exact 6k+1 values >= 7 are unchanged.
        assert_eq!(mochi_frame_count(7), 7);
        assert_eq!(mochi_frame_count(19), 19);
        assert_eq!(mochi_frame_count(151), 151);
        // Below the floor snaps up to 7 (6·1+1 — the smallest on-lattice count with more
        // than one latent frame, mirroring LTX's 9 = 8·1+1 and Wan's 5 = 4·1+1).
        assert_eq!(mochi_frame_count(1), 7);
        assert_eq!(mochi_frame_count(6), 7);
        // Nearest wins; ties go to the lower 6k+1 (the LTX rule).
        assert_eq!(mochi_frame_count(10), 7); // |10-7|=3 == |13-10|=3 -> lower
        assert_eq!(mochi_frame_count(11), 13); // closer to 13
                                               // The shipped default: 5s * 30fps = 150 -> 151 (1 + 6·25). Wan's FLOOR rule would
                                               // give 145 — accepted by the runtime (145 % 6 == 1) but silently 4.83s, not 5s.
        assert_eq!(mochi_frame_count(150), 151);
        assert_ne!(mochi_frame_count(150), wan_frame_count(150));
        // Every output lands on the runtime's lattice (`validate_request`: frames % 6 == 1)
        // and never rounds down below the 7-frame floor.
        for raw in 1..=400 {
            let frames = mochi_frame_count(raw);
            assert_eq!(
                frames % 6,
                1,
                "raw={raw} -> {frames} is off the 6k+1 lattice"
            );
            assert!(frames >= 7, "raw={raw} -> {frames} below the floor");
        }
    }

    #[test]
    fn frame_count_dispatches_by_model_family() {
        let ltx = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "ltx_2_3", "duration": 6.0, "fps": 25
        })));
        assert_eq!(ltx.frame_count(), ltx_frame_count(150));

        let wan = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 3.0, "fps": 16
        })));
        assert_eq!(wan.frame_count(), wan_frame_count(48));

        // Mochi's shipped default (5s @ 30fps, the manifest's only fps): 150 -> 151.
        let mochi = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "duration": 5.0, "fps": 30
        })));
        assert_eq!(mochi.raw_frame_count(), 150);
        assert_eq!(mochi.frame_count(), 151);
        assert_eq!(mochi.frame_count(), mochi_frame_count(150));

        assert!(is_ltx_model("ltx_2_3_eros"));
        assert!(is_wan_model("wan_2_2"));
        assert!(!is_ltx_model("wan_2_2"));
        // Mochi is its own family — it must not fall through to the Wan floor.
        assert!(is_mochi_model("mochi_1"));
        // Prefix, not an exact id: a future `mochi_1_preview`/`mochi_2` must keep hitting
        // the 6k+1 lattice. Exact-matching `mochi_1` would drop it into `video_frame_count`'s
        // fallback arm -> the WAN 4k+1 stride -> off-lattice frames -> engine hard-reject.
        assert!(is_mochi_model("mochi_1_preview"));
        assert!(is_mochi_model("mochi_2"));
        assert!(!is_mochi_model("wan_2_2"));
        assert!(!is_wan_model("mochi_1"));
        assert!(!is_ltx_model("mochi_1"));
    }

    /// sc-12371: the ladder's fallback arm is the WAN stride, and that is load-bearing — NOT the
    /// "unknown model" default it reads as. `bernini`, `scail2_14b` and the in-place ComfyUI
    /// `external_base_*` ids are all Wan engines that `is_wan_model` does not match, and their
    /// generation arms hand the engine `wan_frame_count(raw)`. Reverting this arm to the raw count
    /// (its pre-sc-12371 shape) makes each one's sidecar record a clip length the rendered file does
    /// not have.
    ///
    /// DISCRIMINATING BY CONSTRUCTION: 150 is deliberately OFF the 4k+1 lattice
    /// (`wan_frame_count(150) == 149`), so a raw fallback is RED here. Probing an already-on-lattice
    /// count (149, 81, ...) would agree with BOTH fallbacks and pin nothing.
    #[test]
    fn video_frame_count_fallback_is_the_wan_stride_not_the_raw_count() {
        assert_ne!(
            wan_frame_count(150),
            150,
            "the probe must discriminate: 150 has to be OFF the 4k+1 lattice or this test \
             passes with the fallback reverted to `raw`"
        );
        for model in ["bernini", "scail2_14b", "external_base_wan22_comfyui"] {
            assert_eq!(
                video_frame_count(model, 150),
                wan_frame_count(150),
                "{model} drives a Wan engine, so the ladder must return the Wan stride its \
                 generation arm hands the engine — not the unsnapped raw count"
            );
        }
        // The named families keep their own strides — the fallback must not swallow them.
        assert_eq!(video_frame_count("ltx_2_3", 150), ltx_frame_count(150));
        assert_eq!(video_frame_count("mochi_1", 150), mochi_frame_count(150));
        assert_eq!(video_frame_count("wan_2_2", 150), wan_frame_count(150));
        // ...and those three strides are mutually distinct at 150, so the arms above are really
        // dispatching rather than all collapsing onto one lattice.
        assert_eq!((ltx_frame_count(150), mochi_frame_count(150)), (153, 151));
    }

    /// sc-12371: `frame_count()` — what every asset sidecar records — must BE the shared ladder, so
    /// it cannot drift from what the worker's arms hand the engine. Pinned on the Wan-engine ids
    /// whose prefixes the family predicates do not match, which is exactly where the two
    /// implementations used to disagree (149 rendered vs 150 recorded).
    #[test]
    fn frame_count_is_the_shared_ladder_for_wan_engines_off_the_prefix() {
        for model in ["bernini", "scail2_14b", "external_base_wan22_comfyui"] {
            let request = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": model, "duration": 6.0, "fps": 25
            })));
            assert_eq!(request.raw_frame_count(), 150);
            assert_eq!(
                request.frame_count(),
                video_frame_count(model, 150),
                "{model}: the sidecar's frame count must resolve through the shared ladder"
            );
            assert_eq!(
                request.frame_count(),
                149,
                "{model}: the Wan engine renders 149 at the 6s x 25fps default, so the asset \
                 must not claim the unsnapped 150"
            );
        }
    }
}
