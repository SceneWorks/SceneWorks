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
    array_or_empty, clamped_u32, nonempty_string_or, object_or_empty, optional_i64, optional_id,
    string_list,
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
    pub duration: f32,
    /// Frames per second, clamped to 1..=60 (Python `safe_int`).
    pub fps: u32,
    /// Output dimensions: clamped to 256..=1920, floored to the model's dimension
    /// granularity — `model_manifest_entry.limits.requiresDimensionsMultipleOf`, default
    /// 32 (Python `normalized_dimensions`) — then fitted into the model's area cap,
    /// `limits.maxPixels`, if it declares one (no cap when absent). Defaults 768x512.
    ///
    /// Each video model declares the stride its engine actually needs, so the advertised
    /// buckets survive the floor: 16 for mochi (sc-11993) and for bernini / the Wan A14B
    /// trio, 64 for ltx and svd. The A14B-family engines additionally cap area at 901,120
    /// px; `maxPixels` is what keeps a raw over-cap request (retry / MCP / preset replay,
    /// none of which pass through the UI dropdown) from reaching them (sc-12294).
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
            duration: safe_float(payload.get("duration"), 6.0, 1.0, 30.0),
            fps: clamped_u32(payload.get("fps"), 25, 1, 60),
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

    /// The frame count the model will actually produce: the raw `duration * fps`
    /// snapped to the model's temporal constraint (LTX `8k + 1`, Wan `4n + 1`,
    /// Mochi `6k + 1`). Unknown / stub models keep the raw count. The engine's
    /// `validate()` is authoritative for the real models (sc-3034 / sc-3035); this
    /// mirrors the Python worker's pre-snap so the stub clip length and the UI
    /// estimate agree.
    pub fn frame_count(&self) -> u32 {
        let raw = self.raw_frame_count();
        if is_ltx_model(&self.model) {
            ltx_frame_count(raw)
        } else if is_wan_model(&self.model) {
            wan_frame_count(raw)
        } else if is_mochi_model(&self.model) {
            mochi_frame_count(raw)
        } else {
            raw
        }
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
///   848x480 stops silently rendering as 832x480. Their 720p buckets still move to 704, but
///   for a different reason: the A14B **area** cap (704×1280 = 901,120 px), which candle
///   hard-errors on and mlx i2v silently rescales for — not the stride.
/// * **32** — the dense Wan TI2V-5B (z48 vae22, `vae_stride 16` → candle `SIZE_MULTIPLE =
///   32`) and scail2 (hardcoded `DIM_ALIGN`) genuinely need it. There the *advertisement*
///   was wrong, so their 720p buckets were corrected to the 704 they always rendered; both
///   keep this default.
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
    let w = floor_to_multiple(clamped_u32(width, 768, 256, 1920), multiple);
    let h = floor_to_multiple(clamped_u32(height, 512, 256, 1920), multiple);
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
/// A direct port of the mlx engine's `best_output_size` (`mlx-gen-wan/src/pipeline.rs:94`,
/// itself a port of `generate_wan.py`'s `_best_output_size`): derive the ideal `(ow, oh)` from
/// `√(max_area·ratio)`, then try width-first and height-first alignment and keep whichever
/// distorts the ratio less. Porting it rather than inventing a fit is the point of the story —
/// mlx I2V/TI2V silently rescale through this exact function, so matching it is what makes the
/// app's normalization agree with the backend that rescales, while keeping the result inside the
/// cap that the candle backend hard-errors on. One geometry, both backends (sc-12294).
///
/// The `.max(d)` clamps mirror the engine's own F-030 guard against a degenerate area flooring a
/// dimension to 0 (and a subsequent `area / 0` → NaN ratio). They cannot fire for any *shipped*
/// cap, and the port is kept faithful rather than trimmed.
///
/// The bound comes from the shipped caps, NOT from [`max_pixels_of`]'s filter — that distinction
/// matters if a future cap is added. [`MIN_HONORABLE_MAX_PIXELS`] only guarantees a *square*
/// 256×256 fits; it says nothing about a non-square fit, which drives the minor dimension well
/// below 256 (a cap of exactly 65,536 — which passes the filter — fits 1920×256 to `(688, 80)` at
/// stride 16, violating [`normalized_dimensions`]'s own 256 floor and every engine's `min_size`).
/// What actually rules the clamps out is the reachable minimum: [`normalized_dimensions`] clamps
/// both inputs to `[256, 1920]`, so the aspect ratio cannot exceed `1920/256 = 7.5`, and at the
/// only cap the manifest declares (901,120) the smaller ideal dimension is `√(901120/7.5) ≈ 346`
/// — comfortably above every stride, so nothing floors to 0. A cap below ~491,520 would put an
/// extreme-ratio request back into clamp territory; the manifest's caps are test-pinned.
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

        // A finer floor is NOT a licence to advertise 720p. The A14B family (bernini + the
        // Wan trio) is capped by AREA, not stride: candle hard-errors above MAX_AREA_14B
        // (704×1280 = 901,120 px) and mlx i2v silently rescales 1280×720 → 1264×704. So
        // 1280x704 — exactly AT the cap — is the real 1280-wide bucket, and it must survive
        // the ÷16 floor untouched.
        for model in [
            "bernini",
            "wan_2_2_t2v_14b",
            "wan_2_2_i2v_14b",
            "wan_2_2_vace_fun_14b",
        ] {
            let req = VideoRequest::from_payload(&payload(json!({
                "projectId": "p", "model": model, "width": 1280, "height": 704,
                "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
            })));
            assert_eq!(
                (req.width, req.height),
                (1280, 704),
                "{model} must keep its at-cap 1280x704 bucket"
            );
            assert!(
                (req.width as usize) * (req.height as usize) <= 704 * 1280,
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
    /// Where the backends disagree the table carries the STRICTER side, because one manifest
    /// serves both:
    /// * `wan_2_2_t2v_14b`  — candle errors (`wan14b.rs:645`); mlx `max_area: 0`, uncapped.
    /// * `wan_2_2_vace_fun_14b` — candle errors (`model_vace.rs:298`); mlx uncapped (`:107`).
    /// * `bernini`  — candle errors (`candle-gen-bernini/src/config.rs:164`); mlx calls
    ///   `align_dim` directly and never caps.
    /// * `scail2_14b` — candle errors (`candle-gen-scail2/src/pipeline.rs:294`); mlx uncapped.
    /// * `wan_2_2` (TI2V-5B) — the reverse: mlx caps + silently rescales
    ///   (`config.rs:293`); candle's 5B `validate` (`lib.rs:328`) has NO area check.
    /// * `wan_2_2_i2v_14b` — the only agreement, and even there candle errors while mlx
    ///   rescales.
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
        ("mochi_1", Some(16), None),
        ("wan_2_2", None, Some(901_120)),
        ("wan_2_2_t2v_14b", Some(16), Some(901_120)),
        ("wan_2_2_i2v_14b", Some(16), Some(901_120)),
        ("wan_2_2_vace_fun_14b", Some(16), Some(901_120)),
        ("bernini", Some(16), Some(901_120)),
        ("scail2_14b", None, Some(901_120)),
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

    /// sc-12294 BLOCKER — the case the dropdown never produces but every raw path does.
    ///
    /// `1280x720` was the shipped `defaults.resolution` of the A14B T2V/I2V entries, so it is
    /// the single most common stored value. On `main` a blanket ÷32 floored it to 1280x704 (=
    /// the cap exactly) by accident. Declaring ÷16 removes that accidental protection — so the
    /// cap must be enforced, or job retry / MCP / `POST /api/v1/jobs` / preset replay would
    /// hand candle a hard error and mlx a silent rescale.
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
                     candle would hard-error and mlx would silently rescale",
                    request.width,
                    request.height,
                    (request.width as u64) * (request.height as u64)
                );
            }
        }
    }

    /// The fit is a port of mlx's `best_output_size`, and this pins it to the value the
    /// engine's OWN test pins (`mlx-gen-wan/src/pipeline.rs:1510`): `best_output_size(1280,
    /// 720, 16, 16, 704*1280) == (1264, 704)`. If the two ever diverge, the app stops agreeing
    /// with the backend that silently rescales — which is the whole reason to port rather than
    /// invent a fit.
    #[test]
    fn area_fit_matches_the_mlx_engines_own_pinned_value() {
        assert_eq!(fit_to_max_pixels(1280, 720, 16, 901_120), (1264, 704));

        // Aspect-preserving and grid-aligned, and it only ever shrinks.
        let (w, h) = fit_to_max_pixels(1280, 720, 16, 901_120);
        assert!((w as u64) * (h as u64) <= 901_120, "fits the cap");
        assert_eq!((w % 16, h % 16), (0, 0), "lands on the declared lattice");
        assert!(w <= 1280 && h <= 720, "never upscales");
        let (orig, fitted) = (1280.0 / 720.0, w as f64 / h as f64);
        assert!(
            (orig - fitted).abs() / orig < 0.05,
            "aspect preserved within 5%: {orig} vs {fitted}"
        );

        // An at-cap request is untouched: the engines check `>`, so 901,120 exactly passes.
        assert_eq!(1280 * 704, 901_120);
        let at_cap = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "wan_2_2_i2v_14b", "width": 1280, "height": 704,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16, "maxPixels": 901_120 } }
        })));
        assert_eq!((at_cap.width, at_cap.height), (1280, 704));
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
        // Unknown model keeps the raw count.
        assert_eq!(request.frame_count(), 48);
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
        // the 6k+1 lattice. Exact-matching `mochi_1` would drop it into `frame_count`'s
        // `else { raw }` arm -> off-lattice frames -> engine hard-reject.
        assert!(is_mochi_model("mochi_1_preview"));
        assert!(is_mochi_model("mochi_2"));
        assert!(!is_mochi_model("wan_2_2"));
        assert!(!is_wan_model("mochi_1"));
        assert!(!is_ltx_model("mochi_1"));
    }
}
