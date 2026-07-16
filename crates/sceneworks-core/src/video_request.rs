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
    /// Output dimensions: clamped to 256..=1920, then floored to the model's dimension
    /// granularity — `model_manifest_entry.limits.requiresDimensionsMultipleOf`, default
    /// 32 (Python `normalized_dimensions`). Defaults 768x512. Mochi declares 16 so its
    /// native 848x480 bucket survives the floor (sc-11993).
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
/// limit; the rest keep 32 (sc-12294 tracks the shipped 848/720 buckets that still
/// floor, pending a decision — changing them retroactively would move the geometry of
/// already-reproducible recipes).
fn normalized_dimensions(
    width: Option<&Value>,
    height: Option<&Value>,
    model_manifest_entry: &JsonObject,
) -> (u32, u32) {
    let multiple = dimension_multiple_of(model_manifest_entry);
    let w = floor_to_multiple(clamped_u32(width, 768, 256, 1920), multiple);
    let h = floor_to_multiple(clamped_u32(height, 512, 256, 1920), multiple);
    (w, h)
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

        // A manifest entry that does NOT declare the limit keeps the 32 default — the
        // shipped case for every video model except ltx (declares 32) and mochi (16).
        // sc-12294 tracks bernini's own 848x480 bucket still flooring to 832x480 pending
        // a product decision; B3 deliberately does NOT change it. Only models that
        // *declare* `requiresDimensionsMultipleOf` opt out of the 32 floor.
        let bernini = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "bernini", "width": 848, "height": 480,
            "modelManifestEntry": { "family": "bernini", "limits": { "resolutions": ["848x480"] } }
        })));
        assert_eq!((bernini.width, bernini.height), (832, 480));
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

        // A declared 32 (ltx) is identical to the default path.
        let ltx = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "ltx_2_3", "width": 1000, "height": 543,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 32 } }
        })));
        assert_eq!((ltx.width, ltx.height), (992, 512));

        // 16 is a smaller floor, not a bypass: off-lattice input still floors.
        let odd = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "mochi_1", "width": 855, "height": 489,
            "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } }
        })));
        assert_eq!((odd.width, odd.height), (848, 480));
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
