//! The image-generation job request (epic 3018, sc-3020).
//!
//! Parses a job `payload` into a typed request so the native MLX/candle Rust
//! worker reads the same payload the UI already sends (the parse was ported from
//! the retired Python worker's `image_request_from_job`). The `advanced`
//! and `model_manifest_entry` maps pass through verbatim (they carry per-family knobs
//! like steps/guidanceScale/mlxQuantize/poses/angleSet/controlScale/referenceStrength
//! and the resolved model manifest entry), so adding a family needs no DTO change.
//!
//! **Geometry, deliberately un-normalized (sc-12384).** Unlike the video request
//! ([`crate::video_request`], which floors dims to `limits.requiresDimensionsMultipleOf`
//! and fits them into `limits.maxPixels` via `normalized_dimensions`), the image lane
//! applies NO stride/area refit — `width`/`height` are only sanity-clamped to
//! `256..=4096` and handed to the engine as-is. The engine is the single authority on
//! image geometry: it rejects an illegal size loudly rather than the app silently
//! substituting a different one (the failure mode a refit twin would reintroduce — cf.
//! the per-side clamp `image_jobs::sensenova::sensenova_dim` was doing to over-cap
//! SenseNova buckets before sc-12384 trimmed them). What keeps that contract honest is
//! not a runtime normalize but a build-time guard: every image model's advertised
//! `limits.resolutions` + default must fit its PINNED engine's `[min_size, max_size]`
//! envelope, asserted by `sceneworks_worker::engines`'s
//! `shipped_image_geometry_is_within_the_pinned_engine_envelope`. So the manifest may
//! only advertise buckets the engine honors, and no app-layer refit is needed or wanted.

use serde_json::Value;

use crate::contracts::{ImageUpscaleRequest, JsonObject};
use crate::payload_util::{
    array_or_empty, clamped_u32, declared_resolution, int_array, nonempty_string_or,
    object_or_empty, optional_i64, optional_id, string_list, string_or,
};

/// Default model when the payload omits one (matches the Python worker).
const DEFAULT_MODEL: &str = "z_image_turbo";
const DEFAULT_MODE: &str = "text_to_image";
const DEFAULT_STYLE_PRESET: &str = "cinematic";
/// Default fit mode (epic 2551): never distort, cover the frame. Shared with the
/// video request (sc-6139) so image- and video-conditioned sources normalize identically.
pub(crate) const DEFAULT_FIT_MODE: &str = "crop";
pub(crate) const FIT_MODES: [&str; 4] = ["crop", "pad", "outpaint", "stretch"];

/// The square used when neither the caller nor the model's manifest entry names a size — the
/// historical blanket, kept for models that declare no `defaults.resolution`.
const DEFAULT_DIMENSION: u32 = 1024;

/// The batch size used when neither the caller nor the model's manifest entry names one — the
/// historical blanket, kept for models that declare no `defaults.count`.
const DEFAULT_COUNT: u32 = 4;

/// The payload-sanity bounds on batch size. NOT a model's limit: that is `limits.count`, which is
/// still unread (sc-12335 — a menu, and a separate question from this default).
const MIN_COUNT: u32 = 1;
const MAX_COUNT: u32 = 8;

/// `defaults.count` from a resolved manifest entry, or `None` for the blanket [`DEFAULT_COUNT`].
///
/// The fifth and last dead `defaults.*` key, and the widest: **29 of the 45** image models declare
/// `defaults.count: 1` while the API blanket-defaulted to **4**, so a caller that named no count got
/// four images from a model that asks for one.
///
/// It matters most on exactly the family sc-12427 just moved: `sensenova_u1_8b` declares
/// `2048x2048` **and** `count: 1`. Honoring only the resolution took a bare
/// `generate_image(model = "sensenova_u1_8b", prompt = …)` from `4 x 1024²` to `4 x 2048²` — **4x
/// the pixels of what shipped before**, where the model's own declaration is `1 x 2048²`, i.e. the
/// original cost at the correct geometry. The two keys are one intent; reading half of it made the
/// bare call more expensive rather than more correct.
///
/// Zero web blast radius, for the same reason as every other key here: the studio always SENDS a
/// count (`ImageStudio.jsx:1950`), so it is a caller that names a value and never reaches this
/// path. That the studio *seeds* its own control with a hardcoded 4 rather than the model's
/// declaration is a separate, user-visible question — it belongs to `limits.count` / sc-12335, not
/// to this default.
///
/// Same never-invent-from-a-typo posture as its siblings: a declared count outside
/// [`MIN_COUNT`]`..=`[`MAX_COUNT`] is not a batch anyone meant, so it falls back to the blanket.
pub fn default_count(model_manifest_entry: &JsonObject) -> Option<u32> {
    model_manifest_entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("count"))
        .and_then(Value::as_u64)
        .filter(|count| (u64::from(MIN_COUNT)..=u64::from(MAX_COUNT)).contains(count))
        .map(|count| count as u32)
}

/// The payload-sanity bounds on either dimension (the Python `safe_int` clamp). Deliberately wider
/// than the video lane's 1920 ceiling: image models legitimately render at 2048 (the sensenova U1
/// family declares `2048x2048`), which is why [`default_resolution`] takes its bounds rather than
/// sharing one envelope across lanes.
const MIN_DIMENSION: u32 = 256;
const MAX_DIMENSION: u32 = 4096;

/// `defaults.resolution` from a resolved manifest entry as `(width, height)`, or `None` for the
/// blanket [`DEFAULT_DIMENSION`] square.
///
/// The image half of the dead-`defaults.*` sweep (sc-12400 closed the video lane's fps / duration /
/// resolution). The web honors this key (`ImageStudio.jsx:215`); Rust did not, so a raw API / MCP
/// caller that named no size rendered a blanket 1024x1024 for **5 of the 45** image models that
/// declare something else — the four `sensenova_u1_8b` variants at `2048x2048` (HALF the declared
/// resolution, on the text/infographic family where pixels are the point) and `chroma1_flash` at
/// `768x768`.
///
/// Same never-invent-from-a-typo posture as its video twin; see [`declared_resolution`].
pub fn default_resolution(model_manifest_entry: &JsonObject) -> Option<(u32, u32)> {
    declared_resolution(model_manifest_entry, MIN_DIMENSION, MAX_DIMENSION)
}

/// A typed image-generation request, parsed from a job payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRequest {
    pub project_id: String,
    pub mode: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub model: String,
    /// Number of images, clamped to 1..=8.
    pub count: u32,
    /// Base seed (per-image seeds in `seeds` take precedence).
    pub seed: Option<i64>,
    /// Explicit per-image seeds.
    pub seeds: Vec<i64>,
    /// Output dimensions, clamped to 256..=4096.
    pub width: u32,
    pub height: u32,
    pub style_preset: String,
    /// LoRA specs, passed through verbatim (shape resolved per family).
    pub loras: Vec<Value>,
    pub character_id: Option<String>,
    pub character_look_id: Option<String>,
    pub source_asset_id: Option<String>,
    pub reference_asset_id: Option<String>,
    /// Multiple reference images for a multi-reference edit (sc-6211). Plural companion to the
    /// singular `reference_asset_id`; the FLUX.2-dev multi-image Image-Studio picker sends these.
    /// Empty for every single-reference / character flow (which keeps using `reference_asset_id`).
    pub reference_asset_ids: Vec<String>,
    pub mask_asset_id: Option<String>,
    /// One of crop/pad/outpaint/stretch (default crop).
    pub fit_mode: String,
    /// The resolved model manifest entry (repo/quant/limits/…), passed through.
    pub model_manifest_entry: JsonObject,
    /// Per-family advanced knobs (steps, guidanceScale, mlxQuantize, …), passed through.
    pub advanced: JsonObject,
    /// Image Studio "Upscale" toggle (sc-8091): when `enabled`, the worker upscales each
    /// generated image with `engine` (`seedvr2` / `real-esrgan`) at `factor` and writes a
    /// second "(Nx upscaled)" asset — mirroring the Python worker. Disabled when omitted.
    pub upscale: ImageUpscaleRequest,
}

impl ImageRequest {
    /// Parse a job payload (the `payload` object of a `JobSnapshot`). Infallible:
    /// missing fields fall back to the Python defaults and `project_id` may be empty
    /// — the caller validates it is present (the worker rejects an empty project id).
    pub fn from_payload(payload: &JsonObject) -> Self {
        // Hoisted above the struct literal because the geometry now depends on it — the same shape
        // `VideoRequest::from_payload` uses, and the reason the field is read here rather than in
        // field order below.
        let model_manifest_entry = object_or_empty(payload, "modelManifestEntry");
        let (default_width, default_height) = default_resolution(&model_manifest_entry)
            .unwrap_or((DEFAULT_DIMENSION, DEFAULT_DIMENSION));
        let default_count = default_count(&model_manifest_entry).unwrap_or(DEFAULT_COUNT);
        Self {
            project_id: string_or(payload, "projectId", ""),
            mode: nonempty_string_or(payload, "mode", DEFAULT_MODE),
            prompt: string_or(payload, "prompt", ""),
            negative_prompt: string_or(payload, "negativePrompt", ""),
            model: nonempty_string_or(payload, "model", DEFAULT_MODEL),
            count: clamped_u32(payload.get("count"), default_count, MIN_COUNT, MAX_COUNT),
            seed: optional_i64(payload, "seed"),
            seeds: int_array(payload, "seeds"),
            width: clamped_u32(
                payload.get("width"),
                default_width,
                MIN_DIMENSION,
                MAX_DIMENSION,
            ),
            height: clamped_u32(
                payload.get("height"),
                default_height,
                MIN_DIMENSION,
                MAX_DIMENSION,
            ),
            style_preset: nonempty_string_or(payload, "stylePreset", DEFAULT_STYLE_PRESET),
            loras: array_or_empty(payload, "loras"),
            character_id: optional_id(payload, "characterId"),
            character_look_id: optional_id(payload, "characterLookId"),
            source_asset_id: optional_id(payload, "sourceAssetId"),
            reference_asset_id: optional_id(payload, "referenceAssetId"),
            reference_asset_ids: string_list(payload, "referenceAssetIds"),
            mask_asset_id: optional_id(payload, "maskAssetId"),
            fit_mode: normalize_fit_mode(payload.get("fitMode").and_then(Value::as_str)),
            model_manifest_entry,
            advanced: object_or_empty(payload, "advanced"),
            upscale: parse_upscale(payload),
        }
    }

    /// The resolved seed for image `index`: an explicit per-image seed wins, else the
    /// base seed offset by the index (so a multi-image batch from one seed differs),
    /// else `None` (the generator picks a random seed and records it).
    pub fn seed_for(&self, index: usize) -> Option<i64> {
        if let Some(seed) = self.seeds.get(index) {
            return Some(*seed);
        }
        self.seed.map(|base| base.wrapping_add(index as i64))
    }
}

/// Parse the optional `upscale` object (the Image Studio "Upscale" toggle). A missing or
/// malformed value falls back to the disabled default, so a bad payload never aborts a
/// generation — it just skips upscaling.
fn parse_upscale(payload: &JsonObject) -> ImageUpscaleRequest {
    payload
        .get("upscale")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

pub(crate) fn normalize_fit_mode(value: Option<&str>) -> String {
    let normalized = value
        .unwrap_or(DEFAULT_FIT_MODE)
        .trim()
        .to_ascii_lowercase();
    if FIT_MODES.contains(&normalized.as_str()) {
        normalized
    } else {
        DEFAULT_FIT_MODE.to_owned()
    }
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
        let request = ImageRequest::from_payload(&payload(json!({ "projectId": "proj_1" })));
        assert_eq!(request.project_id, "proj_1");
        assert_eq!(request.mode, "text_to_image");
        assert_eq!(request.model, "z_image_turbo");
        assert_eq!(request.count, 4);
        assert_eq!(request.width, 1024);
        assert_eq!(request.height, 1024);
        assert_eq!(request.style_preset, "cinematic");
        assert_eq!(request.fit_mode, "crop");
        assert_eq!(request.prompt, "");
        assert!(request.seed.is_none());
        assert!(request.seeds.is_empty());
        assert!(request.loras.is_empty());
        assert!(request.advanced.is_empty());
        // No `upscale` object ⇒ the disabled default (the Image Studio toggle is off).
        assert!(request.upscale.is_disabled());
    }

    #[test]
    fn parses_inline_upscale_request() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p",
            "upscale": { "enabled": true, "engine": "seedvr2", "factor": 4, "softness": 0.5 }
        })));
        assert!(request.upscale.enabled);
        assert_eq!(request.upscale.engine, "seedvr2");
        assert_eq!(request.upscale.factor, 4);
        assert!((request.upscale.softness() - 0.5).abs() < 1e-6);

        // A malformed `upscale` value never aborts parsing — it falls back to disabled.
        let bad = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "upscale": "yes please"
        })));
        assert!(bad.upscale.is_disabled());
    }

    #[test]
    fn clamps_count_and_dimensions() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "count": 99, "width": 10, "height": 99999
        })));
        assert_eq!(request.count, 8);
        assert_eq!(request.width, 256);
        assert_eq!(request.height, 4096);

        let low = ImageRequest::from_payload(&payload(json!({ "projectId": "p", "count": 0 })));
        assert_eq!(low.count, 1);
    }

    #[test]
    fn reads_numeric_strings_and_seeds() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "count": "3", "seed": "42", "seeds": [1, "2", null, 3]
        })));
        assert_eq!(request.count, 3);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.seeds, vec![1, 2, 3]);
    }

    #[test]
    fn seed_for_prefers_explicit_then_offsets_base() {
        let explicit = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "seed": 100, "seeds": [7, 8]
        })));
        assert_eq!(explicit.seed_for(0), Some(7));
        assert_eq!(explicit.seed_for(1), Some(8));
        // No explicit seed at index 2 -> base + index.
        assert_eq!(explicit.seed_for(2), Some(102));

        let none = ImageRequest::from_payload(&payload(json!({ "projectId": "p" })));
        assert_eq!(none.seed_for(0), None);
    }

    #[test]
    fn normalizes_fit_mode_and_passes_through_maps() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p",
            "fitMode": "OUTPAINT",
            "advanced": { "steps": 8, "mlxQuantize": 8 },
            "modelManifestEntry": { "family": "z-image", "repo": "x" },
            "loras": [{ "path": "a.safetensors", "weight": 0.8 }]
        })));
        assert_eq!(request.fit_mode, "outpaint");
        assert_eq!(request.advanced.get("steps"), Some(&json!(8)));
        assert_eq!(
            request.model_manifest_entry.get("family"),
            Some(&json!("z-image"))
        );
        assert_eq!(request.loras.len(), 1);

        let bogus =
            ImageRequest::from_payload(&payload(json!({ "projectId": "p", "fitMode": "weird" })));
        assert_eq!(bogus.fit_mode, "crop");
    }
}
