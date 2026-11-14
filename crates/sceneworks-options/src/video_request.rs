use crate::contracts::JsonObject;
use crate::limits::humanized_number_menu;
use crate::payload_util::{declared_resolution, parse_u32};
use serde::{Deserialize, Serialize};
use serde_json::Value;
pub const VIDEO_JOB_MODES: &[&str] = &[
    "image_to_video",
    "text_to_video",
    "first_last_frame",
    "extend_clip",
    "video_bridge",
    "replace_person",
    // Bernini editing / reference-driven video modes (sc-4703).
    "video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    // Bernini multi-source-video modes (sc-5425): mv2v (multiple source clips)
    // and ads2v (source video + reference video + reference images).
    "multi_video_to_video",
    "ads2v",
    // SCAIL-2 standalone character animation (sc-5448 / sc-5449, epic 5439): reference
    // character image + driving video → animated clip. It was wired end-to-end — catalog
    // `capabilities`, `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible`, the candle claim gate,
    // the worker's `generate_scail2` — and offered in the Video Studio, but never added
    // HERE, so every submission 400'd on "Unsupported video mode" and the mode was
    // unreachable from the moment it shipped (GH #2074).
    "animate_character",
];
pub const MAX_SCAIL2_REFERENCE_CHARACTERS: usize = 6;
pub const MINIMAX_H3_MIN_DURATION: f32 =
    MINIMAX_H3_LEGAL_FRAME_COUNTS[0] as f32 / MINIMAX_H3_FPS as f32;
pub fn hard_max_duration(model_manifest_entry: &JsonObject) -> Option<f32> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("hardMaxDuration"))
        .and_then(Value::as_f64)
        .map(|cap| cap as f32)
        .filter(|cap| cap.is_finite() && *cap > 0.0)
}

/// `limits.hardMinDuration` (seconds) from a resolved manifest entry, or `None` for **no floor**.
///
/// The mirror of [`hard_max_duration`], added by sc-17158 because a family can have a floor its own
/// engine refuses below rather than pads up to. MiniMax-H3's shortest renderable clip is
/// [`MINIMAX_H3_MIN_DURATION`] (124 frames at 24 fps); a 3 s request is not a short clip there, it
/// is an unrenderable one, and silently lengthening it to 5.1667 s is the same silent-coercion bug
/// class [`duration_limit_error`] exists to refuse at the other end.
///
/// Absent ⇒ no floor, so every model that declares nothing is byte-identical to before this key had
/// a reader — the same never-block-without-evidence posture, and the same typo'd-manifest tolerance:
/// a non-positive or non-finite floor is not a constraint anyone could satisfy.
pub fn hard_min_duration(model_manifest_entry: &JsonObject) -> Option<f32> {
    model_manifest_entry
        .get("limits")
        .and_then(Value::as_object)
        .and_then(|limits| limits.get("hardMinDuration"))
        .and_then(Value::as_f64)
        .map(|floor| floor as f32)
        .filter(|floor| floor.is_finite() && *floor > 0.0)
}

/// The actionable rejection for a `duration` outside the model's declared
/// [`hard_min_duration`]..=[`hard_max_duration`] window, or `None` to admit. The pure decision, so
/// both enqueue gates assert on one message (sc-12297; floor added by sc-17158).
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
    // Manifest bounds are declared to four decimals (`hardMinDuration: 5.1667` for MiniMax-H3's
    // 124-frame floor, really 124/24 = 5.16666… s). Compare at that precision, or the exact floor
    // a client derives from frames (5.1666665 as f32) is refused as "under" its own minimum.
    let declared = |seconds: f32| (f64::from(seconds) * 10_000.0).round();
    if let Some(floor) = hard_min_duration(model_manifest_entry) {
        if declared(duration) < declared(floor) {
            return Some(format!(
                "{model} renders clips of at least {floor}s, but this request asks for \
                 {duration}s. Lengthen the clip to {floor}s or more, or choose a model that \
                 renders shorter clips."
            ));
        }
    }
    let cap = hard_max_duration(model_manifest_entry)?;
    (declared(duration) > declared(cap)).then(|| {
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
pub const DEFAULT_WIDTH: u32 = 768;
pub const DEFAULT_HEIGHT: u32 = 512;

/// The payload-sanity bounds on either dimension (the Python `normalized_dimensions` clamp). NOT a
/// model's limit: the stride is [`dimension_multiple_of`] and the area cap is [`max_pixels_of`].
pub const MIN_DIMENSION: u32 = 256;
pub const MAX_DIMENSION: u32 = 1920;

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
        let allowed = humanized_number_menu(&menu);
        format!(
            "{model} renders at {allowed} fps, but this request asks for {fps} fps. Set fps to \
             {allowed}, or choose a model that renders at {fps} fps."
        )
    })
}

/// The reference-image cap applied when a model declares no `limits.maxReferenceAssets` —
/// the historical blanket, so every already-shipped video model keeps refusing a 9th
/// reference exactly as it did before sc-17160 raised the API's payload-sanity ceiling.
pub const DEFAULT_MAX_REFERENCE_ASSETS: usize = 8;
/// The source-clip cap applied when a model declares no `limits.maxSourceClipAssets` —
/// the historical blanket, unchanged for every already-shipped video model.
pub const DEFAULT_MAX_SOURCE_CLIP_ASSETS: usize = 8;
/// The reference-**audio** cap applied when a model declares no
/// `limits.maxReferenceAudioAssets`.
///
/// **Zero, deliberately.** `referenceAudioAssetIds` is new surface (sc-17160) that no video
/// model shipped before it could consume, and no engine silently ignores conditioning it was
/// handed — it renders unconditioned, which is invisible in the output (the epic-1788 class).
/// Defaulting to 0 means a model has to *declare* that it takes audio references before one
/// can reach it, so the new field is inert for every existing family rather than accepted-and-
/// dropped.
pub const DEFAULT_MAX_REFERENCE_AUDIO_ASSETS: usize = 0;

/// The per-model reference-media caps resolved from a manifest entry's `limits`.
///
/// Four independent facts, because a model's reference surface is not one number: MiniMax-H3's
/// Ref2VA takes ≤9 images, ≤3 video clips and ≤3 audio clips but ≤12 **files in total**, so the
/// per-list caps do not multiply out to the real ceiling and the combined cap is not derivable
/// from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceCaps {
    /// `limits.maxReferenceAssets` — still-image references (`referenceAssetIds`).
    pub images: usize,
    /// `limits.maxSourceClipAssets` — source/reference video clips (`sourceClipAssetIds`).
    pub clips: usize,
    /// `limits.maxReferenceAudioAssets` — audio references (`referenceAudioAssetIds`).
    pub audio: usize,
    /// `limits.maxCombinedReferenceAssets` — the ceiling on images + clips + audio together, or
    /// `None` when the model declares none. Absent ⇒ no combined ceiling, so a model that
    /// declares nothing behaves exactly as it did before this key had a reader.
    pub combined: Option<usize>,
}

impl Default for ReferenceCaps {
    fn default() -> Self {
        Self {
            images: DEFAULT_MAX_REFERENCE_ASSETS,
            clips: DEFAULT_MAX_SOURCE_CLIP_ASSETS,
            audio: DEFAULT_MAX_REFERENCE_AUDIO_ASSETS,
            combined: None,
        }
    }
}

/// [`ReferenceCaps`] for a resolved manifest entry, falling back to the pre-sc-17160 defaults
/// for every key the model does not declare.
///
/// Same never-block-without-evidence posture as [`hard_max_duration`]: a declared cap that is
/// not a non-negative integer is not a constraint anyone could satisfy, so it falls back to the
/// default rather than bricking the model.
pub fn reference_caps(model_manifest_entry: &JsonObject) -> ReferenceCaps {
    let limits = model_manifest_entry
        .get("limits")
        .and_then(Value::as_object);
    let cap = |key: &str| -> Option<usize> {
        limits
            .and_then(|limits| limits.get(key))
            .and_then(Value::as_u64)
            .and_then(|cap| usize::try_from(cap).ok())
    };
    let defaults = ReferenceCaps::default();
    ReferenceCaps {
        images: cap("maxReferenceAssets").unwrap_or(defaults.images),
        clips: cap("maxSourceClipAssets").unwrap_or(defaults.clips),
        audio: cap("maxReferenceAudioAssets").unwrap_or(defaults.audio),
        combined: cap("maxCombinedReferenceAssets"),
    }
}

/// The actionable rejection for a request whose reference media exceeds what the model declares,
/// or `None` to admit. The pure decision, so the API enqueue gate and the worker preflight assert
/// on one message — the same shape as [`duration_limit_error`] / [`fps_limit_error`] (sc-17160).
///
/// **Per-model, not a global bump.** MiniMax-H3 accepts 9 images where every family before it
/// accepted 8; raising the shared API constant alone would have handed every other video model a
/// 9th reference its engine was never asked to encode. The manifest is the seam epic 12334
/// established for exactly this: a model that takes more declares more, and one that declares
/// nothing keeps the historical [`DEFAULT_MAX_REFERENCE_ASSETS`].
///
/// Per-list caps are checked before the combined one so the message names the list actually over
/// budget; the combined cap is the only one that can reject a request in which every individual
/// list fits.
///
/// Message follows the house convention (`mlx_fit_gate::too_big_error`): name the model, state
/// what was asked and what the cap is, and give the lever.
/// What a `reference_to_video` reference set is, judged against the ONE rule every layer must
/// agree on (sc-19574).
///
/// Four layers used to hold three opinions about an audio-only set: the Video Studio enabled
/// Generate for it, `validate_video_job` accepted it, `sceneworks-mcp` submitted it, and the worker
/// refused it. A user could assemble a request the product presented as valid and then be told no.
///
/// The engine was right, and it is not a judgement call: diffusers `MiniMaxH3` (upstream PR #14355,
/// `0.40.0.dev0 @ 7564fb01`) documents `MiniMaxH3AudioReference` as "a voice or music reference: at
/// most 3 per request, and never on its own — an audio reference has to be paired with at least one
/// image or video reference. It never reaches the conditioner and is encoded by the audio VAE
/// alone", and enforces it in `before_encoder.py`:
///
/// ```text
/// if set(kinds) == {"audio"}:
///     raise ValueError("An audio reference has to be paired with at least one image or video
///                       reference and cannot be used on its own.")
/// ```
///
/// So an audio-only set leaves the VISUAL stream unconditioned, and the render it would produce is
/// not the one asked for.
///
/// It lives HERE, as one predicate the API, the MCP tool and the worker all call, because
/// restating the rule in each of them is exactly how they came to disagree. Each layer still words
/// its own message — an agent's error, a user's error and an engine's error are different
/// sentences — but none of them decides the rule.
///
/// It judges MEDIA KINDS, not model ids: "audio conditions the soundtrack, the visual stream needs
/// a visual reference" is a fact about which lists carry pixels. Which of the three lists a given
/// model takes at all, and how many, stays with the model as its declared
/// `limits.max*ReferenceAssets` ([`reference_limit_error`]) — a model that takes no audio refuses
/// it there, one list up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSetVerdict {
    /// At least one image or video-clip reference: the visual conditioner has something to read.
    Conditionable,
    /// No references at all.
    Empty,
    /// Audio references only — the shape upstream raises on.
    AudioOnly,
}

/// Classify a `reference_to_video` reference set. See [`ReferenceSetVerdict`].
pub fn classify_reference_set(images: usize, clips: usize, audio: usize) -> ReferenceSetVerdict {
    if images > 0 || clips > 0 {
        return ReferenceSetVerdict::Conditionable;
    }
    if audio > 0 {
        return ReferenceSetVerdict::AudioOnly;
    }
    ReferenceSetVerdict::Empty
}

pub fn reference_limit_error(
    model: &str,
    images: usize,
    clips: usize,
    audio: usize,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    let caps = reference_caps(model_manifest_entry);
    for (count, cap, field, noun) in [
        (images, caps.images, "referenceAssetIds", "reference images"),
        (clips, caps.clips, "sourceClipAssetIds", "source clips"),
        (
            audio,
            caps.audio,
            "referenceAudioAssetIds",
            "audio references",
        ),
    ] {
        if count > cap {
            return Some(if cap == 0 {
                format!(
                    "{model} takes no {noun}, but this request supplies {count}. Remove \
                     {field}, or choose a model that conditions on {noun}."
                )
            } else {
                format!(
                    "{model} takes up to {cap} {noun}, but this request supplies {count}. \
                     Reduce {field} to {cap} or fewer, or choose a model that takes more."
                )
            });
        }
    }
    let combined = caps.combined?;
    let total = images + clips + audio;
    (total > combined).then(|| {
        format!(
            "{model} takes up to {combined} reference files in total, but this request supplies \
             {total} ({images} reference images + {clips} source clips + {audio} audio \
             references). Remove {} of them.",
            total - combined
        )
    })
}

/// The dimension granularity applied when a model's manifest entry does not declare
/// `limits.requiresDimensionsMultipleOf` — the historical blanket floor, kept as the
/// default so every already-shipped model's geometry is byte-for-byte unchanged.
pub const DEFAULT_DIMENSION_MULTIPLE: u32 = 32;

fn safe_float(value: Option<&Value>, default: f32, min: f32, max: f32) -> f32 {
    parse_safe_float(value).unwrap_or(default).clamp(min, max)
}
fn parse_safe_float(value: Option<&Value>) -> Option<f32> {
    value
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
}
pub const MINIMAX_H3_FPS: u32 = 24;
pub const MINIMAX_H3_LEGAL_FRAME_COUNTS: [u32; 14] = [
    124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345,
];

pub const LTX25_VAE_DECODER_CONV: &str = "conv";
pub const LTX25_VAE_DECODER_DIFFUSION: &str = "diffusion";

/// LTX-2.5 exposes at most two temporal x2 refinement rounds.
pub const LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS: u32 = 2;

/// The admitted range and default for `advanced.referenceImageShortEdge` — the short edge a
/// MiniMax-H3 image reference is encoded at, in pixels (sc-23402).
///
/// These mirror gen-core's `REFERENCE_IMAGE_SHORT_EDGE_{MIN,MAX,DEFAULT}` and exist here for the
/// same reason [`VideoAutoDurationRange`] does: this crate deliberately has no gen-core dependency,
/// so the API and the film harness can judge and record the value without taking a provider
/// dependency. The default is upstream's own `reference_image_short_edge` pipeline value, and the
/// engine resolves it with `sceneworks_gen_core::effective_reference_image_short_edge` — the same
/// rule [`effective_reference_image_short_edge`] applies on this side of the boundary.
///
/// The knob only ever goes DOWNWARD from the default: it sizes the REFERENCE, never the render, so
/// lowering it buys reference token count (roughly quadratic in the short edge) at the cost of
/// reference detail. An out-of-range value is refused, never clamped — a clamp would change the
/// token budget the caller believes it asked for.
pub const REFERENCE_IMAGE_SHORT_EDGE_MIN: u32 = 1024;
/// Largest admitted reference-image short edge; see [`REFERENCE_IMAGE_SHORT_EDGE_MIN`].
pub const REFERENCE_IMAGE_SHORT_EDGE_MAX: u32 = 2048;
/// The reference-image short edge when a request names none; see [`REFERENCE_IMAGE_SHORT_EDGE_MIN`].
pub const REFERENCE_IMAGE_SHORT_EDGE_DEFAULT: u32 = 2048;

/// The EFFECTIVE reference-image short edge for a requested value: the value itself, or
/// [`REFERENCE_IMAGE_SHORT_EDGE_DEFAULT`] when the request names none (sc-23402).
///
/// The local twin of gen-core's `effective_reference_image_short_edge`, applied here only because
/// this crate has no gen-core dependency (see [`REFERENCE_IMAGE_SHORT_EDGE_MIN`]). Anything that
/// RECORDS the value — the film harness's attempt record — resolves it through this function so the
/// number written down cannot drift from the number the engine rendered at.
pub fn effective_reference_image_short_edge(requested: Option<u32>) -> u32 {
    requested.unwrap_or(REFERENCE_IMAGE_SHORT_EDGE_DEFAULT)
}

/// Parse `advanced.referenceImageShortEdge` (sc-23402). Omission is the engine's own default.
///
/// Refused, never clamped, naming the field, the value and the range — the same refusal gen-core's
/// `validate_reference_image_short_edge` gives at the port's `validate`, given here so a job is
/// refused before it costs a 53 GB text-encoder load.
pub fn requested_reference_image_short_edge(advanced: &JsonObject) -> Result<Option<u32>, String> {
    let Some(value) = advanced.get("referenceImageShortEdge") else {
        return Ok(None);
    };
    let edge = value.as_u64().ok_or_else(|| {
        format!(
            "advanced.referenceImageShortEdge must be an integer from \
             {REFERENCE_IMAGE_SHORT_EDGE_MIN} to {REFERENCE_IMAGE_SHORT_EDGE_MAX}"
        )
    })?;
    if !(u64::from(REFERENCE_IMAGE_SHORT_EDGE_MIN)..=u64::from(REFERENCE_IMAGE_SHORT_EDGE_MAX))
        .contains(&edge)
    {
        return Err(format!(
            "advanced.referenceImageShortEdge must be from {REFERENCE_IMAGE_SHORT_EDGE_MIN} to \
             {REFERENCE_IMAGE_SHORT_EDGE_MAX}, got {edge}"
        ));
    }
    Ok(Some(edge as u32))
}

/// The request key for MiniMax-H3's image-reference resize policy (sc-26691), riding `advanced`
/// beside [`requested_reference_image_short_edge`]'s `referenceImageShortEdge`.
pub const REFERENCE_IMAGE_RESIZE_MODE_KEY: &str = "referenceImageResizeMode";

/// How a MiniMax-H3 image reference is resized before it is encoded (sc-26691) — the local twin of
/// gen-core's `ReferenceImageResizeMode`, kept here for the same reason
/// [`REFERENCE_IMAGE_SHORT_EDGE_MIN`] is: this crate has no gen-core dependency, yet the API, the
/// film harness and the replay boundary all judge and record the value.
///
/// * `match` — scale the reference to the target canvas's pixel area, never upscaling. The engine's
///   default for a request that names none.
/// * `max` — only downsize a short edge above `referenceImageShortEdge`.
/// * `diffusers` — force the short edge to `referenceImageShortEdge`, upscaling included. This is
///   the ONLY behaviour the engine had before sc-26691, which is why a record written before the
///   field existed replays as `diffusers` ([`REFERENCE_IMAGE_RESIZE_MODE_LEGACY`]).
///
/// The short-edge knob is read by `max` / `diffusers` only; under `match` it is inert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceImageResizeMode {
    #[default]
    Match,
    Max,
    Diffusers,
}

impl ReferenceImageResizeMode {
    /// Every admitted mode, in wire order.
    pub const ALL: [Self; 3] = [Self::Match, Self::Max, Self::Diffusers];

    /// The wire spelling (`match` / `max` / `diffusers`), identical to gen-core's serde form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Max => "max",
            Self::Diffusers => "diffusers",
        }
    }

    /// Parse the wire spelling; exact and lowercase, as gen-core's serde is.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.as_str() == value)
    }
}

/// The resize mode a request that names none resolves to: gen-core's own `None ⇒ Match`.
pub const REFERENCE_IMAGE_RESIZE_MODE_DEFAULT: ReferenceImageResizeMode =
    ReferenceImageResizeMode::Match;

/// The resize mode a record written BEFORE sc-26691 was rendered with. Such a record carries no
/// `referenceImageResizeMode`, and its reference was encoded the only way the engine then knew —
/// forced to the short edge — so a replay of it must ask for that, not today's default.
pub const REFERENCE_IMAGE_RESIZE_MODE_LEGACY: ReferenceImageResizeMode =
    ReferenceImageResizeMode::Diffusers;

/// The EFFECTIVE resize mode for a requested value — the local twin of gen-core's
/// `effective_reference_image_resize_mode` (`None` ⇒ [`REFERENCE_IMAGE_RESIZE_MODE_DEFAULT`]).
/// Anything that RECORDS the mode resolves it through here so the record cannot drift from the
/// engine.
pub fn effective_reference_image_resize_mode(
    requested: Option<ReferenceImageResizeMode>,
) -> ReferenceImageResizeMode {
    requested.unwrap_or(REFERENCE_IMAGE_RESIZE_MODE_DEFAULT)
}

/// Parse `advanced.referenceImageResizeMode` (sc-26691). Omission is `Ok(None)` — the engine's own
/// default. Refused, naming the field and the admitted spellings, so a typo costs a payload parse
/// rather than a 53 GB text-encoder load.
pub fn requested_reference_image_resize_mode(
    advanced: &JsonObject,
) -> Result<Option<ReferenceImageResizeMode>, String> {
    let Some(value) = advanced.get(REFERENCE_IMAGE_RESIZE_MODE_KEY) else {
        return Ok(None);
    };
    value
        .as_str()
        .and_then(ReferenceImageResizeMode::parse)
        .map(Some)
        .ok_or_else(|| {
            format!(
                "advanced.{REFERENCE_IMAGE_RESIZE_MODE_KEY} must be one of \"match\", \"max\" or \
                 \"diffusers\", got {value}"
            )
        })
}

/// Whether a job payload renders through the MiniMax-H3 engine (sc-26691): a catalog
/// `minimax_h3*` id, or an imported H3 checkpoint, whose own id carries no such prefix but whose
/// resolved `modelManifestEntry` is of the `minimax-h3` family.
pub fn payload_is_minimax_h3(payload: &JsonObject) -> bool {
    is_minimax_h3_model(&payload_model_id(payload))
        || payload
            .get("modelManifestEntry")
            .and_then(|entry| entry.get("family"))
            .and_then(Value::as_str)
            == Some("minimax-h3")
}

/// Validate and STAMP the resize mode on a video job payload at a creation boundary (sc-26691).
///
/// A MiniMax-H3 payload leaves this function carrying an explicit, valid
/// `advanced.referenceImageResizeMode`: the caller's own value, or `absent` when it named none. The
/// stamp is what makes the persisted job self-describing — a job written from now on always says
/// which policy it ran, so the one record that can lack the key is a record written before the key
/// existed, and [`replay_reference_image_resize_mode_default`] can read that absence as "legacy".
///
/// Fresh requests pass [`REFERENCE_IMAGE_RESIZE_MODE_DEFAULT`]; replays pass what
/// [`replay_reference_image_resize_mode_default`] decides. Every other family is refused the key:
/// it has no reference resize to apply it to, and a visible-but-inert control is worse than a
/// refusal.
pub fn stamp_reference_image_resize_mode(
    payload: &mut JsonObject,
    absent: ReferenceImageResizeMode,
) -> Result<(), String> {
    let model = payload_model_id(payload);
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let requested = match advanced {
        Some(advanced) => requested_reference_image_resize_mode(advanced)?,
        None => None,
    };
    if !payload_is_minimax_h3(payload) {
        if requested.is_some() {
            return Err(format!(
                "advanced.{REFERENCE_IMAGE_RESIZE_MODE_KEY} is a MiniMax-H3 reference control and \
                 is not supported by {model}"
            ));
        }
        return Ok(());
    }
    if requested.is_some() {
        return Ok(());
    }
    let advanced = payload
        .entry("advanced")
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !advanced.is_object() {
        // `VideoRequest::from_payload` reads a non-object `advanced` as empty; keep that reading
        // rather than replacing a caller value this function was not asked to judge.
        return Ok(());
    }
    if let Some(advanced) = advanced.as_object_mut() {
        advanced.insert(
            REFERENCE_IMAGE_RESIZE_MODE_KEY.to_owned(),
            Value::String(absent.as_str().to_owned()),
        );
    }
    Ok(())
}

/// Stamp the resize mode a job is about to RENDER with, at the moment a worker claims it
/// (sc-26691). Returns whether `payload` changed.
///
/// Creation stamps every new MiniMax-H3 job, but a job queued before sc-26691 reaches the claim
/// without the key, and the worker that claims it now renders the engine default (`match`). The
/// claim is the one execution-time boundary every render crosses: stamping the RESOLVED mode there
/// makes the persisted job say what actually ran, so an unstamped H3 record can only be one that
/// was claimed — and rendered — before the upgrade. A malformed value is left for the worker to
/// refuse by name; another family is untouched.
pub fn stamp_claimed_reference_image_resize_mode(payload: &mut JsonObject) -> bool {
    if !payload_is_minimax_h3(payload) {
        return false;
    }
    let advanced = payload
        .entry("advanced")
        .or_insert_with(|| Value::Object(JsonObject::new()));
    let Some(advanced) = advanced.as_object_mut() else {
        return false;
    };
    if advanced.contains_key(REFERENCE_IMAGE_RESIZE_MODE_KEY) {
        return false;
    }
    advanced.insert(
        REFERENCE_IMAGE_RESIZE_MODE_KEY.to_owned(),
        Value::String(
            effective_reference_image_resize_mode(None)
                .as_str()
                .to_owned(),
        ),
    );
    true
}

/// The resize mode a REPLAY (retry / duplicate) stamps when the merged payload names none
/// (sc-26691).
///
/// `persisted` is the job payload as it was recorded, and `persisted_rendered` whether that job
/// actually rendered (completed). Every creation boundary and every claim since sc-26691 stamps the
/// key ([`stamp_reference_image_resize_mode`], [`stamp_claimed_reference_image_resize_mode`]), so a
/// MiniMax-H3 record that lacks it AND rendered was rendered before the upgrade, under the forced
/// short edge, and replays as [`REFERENCE_IMAGE_RESIZE_MODE_LEGACY`]. An unstamped record that never
/// rendered (failed or canceled before any claim stamped it) produced nothing to reproduce, so its
/// replay is a new request and takes today's default — as does a record that carried a mode the
/// changes then dropped, or a replay that switched INTO the family from another model.
pub fn replay_reference_image_resize_mode_default(
    persisted: &JsonObject,
    persisted_rendered: bool,
) -> ReferenceImageResizeMode {
    let unstamped_h3 = payload_is_minimax_h3(persisted)
        && !persisted
            .get("advanced")
            .and_then(Value::as_object)
            .is_some_and(|advanced| advanced.contains_key(REFERENCE_IMAGE_RESIZE_MODE_KEY));
    if unstamped_h3 && persisted_rendered {
        REFERENCE_IMAGE_RESIZE_MODE_LEGACY
    } else {
        REFERENCE_IMAGE_RESIZE_MODE_DEFAULT
    }
}

/// The REPLAY (retry / duplicate) boundary for the resize mode (sc-26691): `merged` is the
/// persisted payload with `changes` shallow-merged over it.
///
/// A mode the record carried is INHERITED, not authored, so a replay that moves off MiniMax-H3
/// without restating `advanced` drops it (another family has nothing to apply it to — the same
/// reset the MOSS controls get) rather than being refused for a key the caller never sent. What
/// remains is then stamped exactly as at creation, with the absent default
/// [`replay_reference_image_resize_mode_default`] decides.
pub fn stamp_replayed_reference_image_resize_mode(
    merged: &mut JsonObject,
    persisted: &JsonObject,
    persisted_rendered: bool,
    changes: &JsonObject,
) -> Result<(), String> {
    if !payload_is_minimax_h3(merged) && !changes.contains_key("advanced") {
        if let Some(advanced) = merged.get_mut("advanced").and_then(Value::as_object_mut) {
            advanced.remove(REFERENCE_IMAGE_RESIZE_MODE_KEY);
        }
    }
    stamp_reference_image_resize_mode(
        merged,
        replay_reference_image_resize_mode_default(persisted, persisted_rendered),
    )
}

// Validated seconds window for LTX-2.5's opt-in duration predictor.
///
/// This lives in core rather than gen-core so the API can reject malformed requests before a job
/// exists without taking a provider dependency. The worker converts it to gen-core's identically
/// shaped `AutoDurationRange` only after the model-specific capability gate has passed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoAutoDurationRange {
    pub min_seconds: f32,
    pub max_seconds: f32,
}

/// Parse the Video Studio's scalar auto-duration keys. Omission and an explicit `false` are both
/// off; opting in requires a finite, positive, ordered min/max window.
pub fn requested_auto_duration(
    advanced: &JsonObject,
) -> Result<Option<VideoAutoDurationRange>, String> {
    let Some(enabled) = advanced.get("autoDuration") else {
        return Ok(None);
    };
    let Some(enabled) = enabled.as_bool() else {
        return Err("advanced.autoDuration must be a boolean when present".to_owned());
    };
    if !enabled {
        return Ok(None);
    }
    let number = |key: &str| {
        advanced
            .get(key)
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                format!("advanced.{key} must be a finite number when auto-duration is enabled")
            })
    };
    let min_seconds = number("autoDurationMinSeconds")?;
    let max_seconds = number("autoDurationMaxSeconds")?;
    if min_seconds <= 0.0 || max_seconds <= 0.0 {
        return Err("advanced auto-duration min/max seconds must both be positive".to_owned());
    }
    if min_seconds > max_seconds {
        return Err(format!(
            "advanced.autoDurationMinSeconds {min_seconds} exceeds advanced.autoDurationMaxSeconds {max_seconds}"
        ));
    }
    Ok(Some(VideoAutoDurationRange {
        min_seconds,
        max_seconds,
    }))
}

/// Parse LTX-2.5's decoder selector. Omission is the fast Conv decoder product default.
pub fn requested_ltx25_vae_decoder(advanced: &JsonObject) -> Result<&str, String> {
    match advanced.get("vaeDecoder") {
        None => Ok(LTX25_VAE_DECODER_CONV),
        Some(Value::String(value))
            if matches!(
                value.as_str(),
                LTX25_VAE_DECODER_CONV | LTX25_VAE_DECODER_DIFFUSION
            ) =>
        {
            Ok(value)
        }
        Some(_) => {
            Err("advanced.vaeDecoder must be \"conv\" or \"diffusion\" when present".to_owned())
        }
    }
}

/// Parse LTX-2.5's temporal-refinement count. Omission and zero are both off.
pub fn requested_temporal_upsample_rounds(advanced: &JsonObject) -> Result<u32, String> {
    let Some(value) = advanced.get("temporalUpsampleRounds") else {
        return Ok(0);
    };
    let rounds = value.as_u64().ok_or_else(|| {
        format!(
            "advanced.temporalUpsampleRounds must be an integer from 0 to {LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS}"
        )
    })?;
    if rounds > u64::from(LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS) {
        return Err(format!(
            "advanced.temporalUpsampleRounds must be from 0 to {LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS}, got {rounds}"
        ));
    }
    Ok(rounds as u32)
}

pub fn is_minimax_h3_model(model: &str) -> bool {
    model.starts_with("minimax_h3")
}

pub fn payload_model_id(payload: &JsonObject) -> String {
    crate::payload_util::nonempty_string_or(payload, "model", "ltx_2_3")
}
