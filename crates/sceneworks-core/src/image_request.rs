//! The image-generation job request (epic 3018, sc-3020).
//!
//! Parses a job `payload` into a typed request so the native MLX/candle Rust
//! worker reads the same payload the UI already sends (the parse was ported from
//! the retired Python worker's `image_request_from_job`). The `advanced`
//! and `model_manifest_entry` maps pass through verbatim (they carry per-family knobs
//! like steps/guidanceScale/mlxQuantize/poses/angleSet/controlScale/referenceStrength
//! and the resolved model manifest entry), so adding a family needs no DTO change.
//!
//! Geometry is validated at API enqueue and worker dispatch by [`validate_image_request_payload`].
//! Accepted dimensions are preserved verbatim, including declared floors below 256 pixels.
//! The engine remains authoritative for geometry beyond the declared product envelope.

pub mod flux3;

use serde_json::Value;

use crate::contracts::{HiresFixRequest, ImageUpscaleRequest, JsonObject};
use crate::payload_util::{
    array_or_empty, int_array, nonempty_string_or, object_or_empty, optional_i64, optional_id,
    string_list, string_or,
};

/// Default model when the payload omits one (matches the Python worker).
const DEFAULT_MODEL: &str = "z_image_turbo";
const DEFAULT_MODE: &str = "text_to_image";
/// Default `stylePreset` when the payload omits one (matches the Python worker, and the
/// rust-api DTO's `default_style_preset`).
///
/// Public because it is the value that means "no preset ran": the API overwrites this field with a
/// recipe-preset id when one does, so anything still holding this literal names nothing.
/// [`crate::workflow_resolution::INERT_STYLE_PRESETS`] reads it from here rather than repeating the
/// literal, because every generated envelope carries it — misreading it as an unresolved preset
/// would mark every shared image unreplayable.
pub const DEFAULT_STYLE_PRESET: &str = "cinematic";
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
/// `pub(crate)` so `workflow_share`'s pose budget can be derived from it rather than restate it:
/// a pose set is the pose lane's batch size (one image per pose), so this is the ceiling it is
/// measured against.
pub use sceneworks_options::image_request::MAX_COUNT;

/// The historical ceiling on `advanced.steps` the generic image lane forwards (raised to a model's
/// own default where that is higher). The worker clamps above it; Image Studio's Steps input
/// mirrors it (`BASE_MAX_STEPS`), and Supra2's request boundary refuses above it (sc-24215).
pub use sceneworks_options::image_request::BASE_MAX_IMAGE_STEPS;

/// The maximum number of strict poses one image-generation job may render.
///
/// A pose set is not ordinary batch metadata: every entry becomes a separate generated image and
/// therefore a separate output artifact. Keep that fan-out finite at eight ordinary maximum-size
/// batches. The resulting 64-pose ceiling still admits the complete shipped 46-pose library plus
/// 18 user-created Key Point Library entries in one run, while preventing an accidentally huge
/// library selection from becoming an effectively unbounded job.
///
/// This is the product/request contract. The workflow-share envelope derives its entry cap from
/// this value, and the web's mechanically pinned twin is checked by a cross-file drift test.
pub use sceneworks_options::MAX_JOB_POSES;

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
pub use sceneworks_options::image_request::default_count;

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
pub use sceneworks_options::image_request::default_resolution;

/// A typed image-generation request, parsed from a job payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRequest {
    pub project_id: String,
    pub mode: String,
    pub prompt: String,
    /// A NATIVE prompt batch (sc-25681, `image.maxPromptBatchImages`): the prompts one engine call
    /// renders together, verbatim. Empty for an ordinary request. When set, `prompt` is `prompts[0]`
    /// (the API enforces it) and image `k` renders `prompts[k % prompts.len()]` — see
    /// [`Self::image_prompt`].
    pub prompts: Vec<String>,
    pub negative_prompt: String,
    pub model: String,
    /// Number of images; validated to 1..=8 before enqueue/load.
    pub count: u32,
    /// Base seed (per-image seeds in `seeds` take precedence).
    pub seed: Option<i64>,
    /// Explicit per-image seeds.
    pub seeds: Vec<i64>,
    /// Output dimensions, preserved verbatim after model-envelope validation.
    pub width: u32,
    pub height: u32,
    pub style_preset: String,
    /// The Style-catalog id the caller selected (sc-13134), a group id or a sub-style id, or
    /// `None`. Dedicated field over the dead `style_preset`: the API resolves it to the catalog
    /// style text and folds it into `prompt` (the `Subject:`/`Style:` splice) before the
    /// worker sees the payload, so the worker mainly records it for lineage/replay. The web sends
    /// the already-composed prompt and omits this top-level id, so a re-parse never double-folds.
    pub style_id: Option<String>,
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
    /// Family-local raw carrier validation and authored dimension intent, captured before tolerant
    /// parsing/defaults can erase malformed slots or make an omitted dimension look explicit.
    pub flux3_reference_request: Option<Result<flux3::ReferenceRequest, String>>,
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
    /// Optional A1111-style two-pass high-resolution refinement. Disabled when omitted.
    pub hires_fix: HiresFixRequest,
}

impl ImageRequest {
    /// Explicit quantization bits (or the verified portable binding's frozen tier), before
    /// model-specific defaults or tier validation.
    pub fn quant_bits(&self) -> Option<i64> {
        crate::payload_util::request_quant_bits(&self.advanced)
    }

    /// The prompt image `index` of this job rendered: `prompt` for an ordinary request, and
    /// `prompts[index % prompts.len()]` for a native prompt batch (prompt-minor order — the engine
    /// renders the whole prompt list once per `count` repetition).
    pub fn image_prompt(&self, index: usize) -> &str {
        if self.prompts.is_empty() {
            &self.prompt
        } else {
            // MOSS repeats each prompt's images together (prompt-major); other providers use
            // the historical prompt-minor convention.
            let prompt_index = if crate::moss_image::is_moss(&self.model) {
                index / self.count.max(1) as usize
            } else {
                index % self.prompts.len()
            };
            &self.prompts[prompt_index % self.prompts.len()]
        }
    }

    /// The images this request produces: `count`, or `prompts × count` for a native prompt batch.
    pub fn image_count(&self) -> u32 {
        let embedding_batch =
            if crate::moss_image::is_moss(&self.model) && self.mode == "edit_image" {
                self.advanced
                    .get(crate::moss_image_advanced::KEY)
                    .and_then(|v| v.get("positive"))
                    .and_then(|v| v.get("values"))
                    .and_then(|v| v.get("metadata"))
                    .and_then(|v| v.get("shape"))
                    .and_then(|v| v.get(0))
                    .and_then(Value::as_u64)
                    .filter(|n| *n <= 8)
                    .unwrap_or(1) as usize
            } else {
                1
            };
        self.count.saturating_mul(
            u32::try_from(self.prompts.len().max(embedding_batch).max(1)).unwrap_or(u32::MAX),
        )
    }

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
            prompts: payload
                .get(crate::image_sampling_controls::PROMPTS_KEY)
                .and_then(Value::as_array)
                .map(|prompts| {
                    prompts
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            negative_prompt: string_or(payload, "negativePrompt", ""),
            model: nonempty_string_or(payload, "model", DEFAULT_MODEL),
            count: unsigned_or(payload.get("count"), default_count),
            seed: optional_i64(payload, "seed"),
            seeds: int_array(payload, "seeds"),
            width: unsigned_or(payload.get("width"), default_width),
            height: unsigned_or(payload.get("height"), default_height),
            style_preset: nonempty_string_or(payload, "stylePreset", DEFAULT_STYLE_PRESET),
            style_id: optional_id(payload, "styleId"),
            loras: array_or_empty(payload, "loras"),
            character_id: optional_id(payload, "characterId"),
            character_look_id: optional_id(payload, "characterLookId"),
            source_asset_id: optional_id(payload, "sourceAssetId"),
            reference_asset_id: optional_id(payload, "referenceAssetId"),
            reference_asset_ids: string_list(payload, "referenceAssetIds"),
            flux3_reference_request: flux3::is_model(&nonempty_string_or(
                payload,
                "model",
                DEFAULT_MODEL,
            ))
            .then(|| flux3::parse(payload)),
            mask_asset_id: optional_id(payload, "maskAssetId"),
            fit_mode: normalize_fit_mode(payload.get("fitMode").and_then(Value::as_str)),
            model_manifest_entry,
            advanced: object_or_empty(payload, "advanced"),
            upscale: parse_upscale(payload),
            hires_fix: parse_hires_fix(payload),
        }
    }
}

/// Parse the optional `upscale` object (the Image Studio "Upscale" toggle). A missing or
/// malformed value falls back to the disabled default for tolerant legacy reads. Execution
/// boundaries call `validate_image_request_payload` first and reject malformed controls.
fn parse_upscale(payload: &JsonObject) -> ImageUpscaleRequest {
    payload
        .get("upscale")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

/// Parse the optional `hiresFix` object. As with `upscale`, malformed data falls back to the
/// disabled default so older and hand-authored payloads remain compatibility-safe.
fn parse_hires_fix(payload: &JsonObject) -> HiresFixRequest {
    payload
        .get("hiresFix")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

/// The shared Hires.fix preflight gate (sc-26644): `None` admits, `Some(message)` refuses with a
/// caller-facing message. Hires.fix is a plain text-to-image second pass: existing edit/control
/// inputs, PiD, multi-phase sampling, and the separate post-generation upscaler are intentionally
/// mutually exclusive rather than being silently dropped by a provider.
///
/// Lives in core so the two boundaries that must agree cannot drift: the worker re-applies it
/// before any model load, and the API applies it at image-job enqueue (the typed route used to
/// drop the web's top-level `hiresFix` entirely, so a request for a model that should refuse it
/// returned 201 and the worker-side refusal never fired). `family` is resolved by the caller —
/// the worker falls back to the generators registry on macOS, the API reads the canonicalized
/// manifest entry.
pub fn hires_fix_request_error(request: &ImageRequest, family: &str) -> Option<String> {
    let mut entry = request.model_manifest_entry.clone();
    if !model_supports_hires_fix(request) {
        entry
            .entry("image".to_owned())
            .or_insert_with(|| serde_json::json!({}))["supportsHiresFix"] = Value::Bool(false);
    }
    sceneworks_options::image_request::hires_fix_request_error(
        &sceneworks_options::image_request::HiresFixInput {
            hires_fix: &request.hires_fix,
            upscale: &request.upscale,
            mode: &request.mode,
            model: &request.model,
            model_manifest_entry: &entry,
            source_asset_id: &request.source_asset_id,
            reference_asset_id: &request.reference_asset_id,
            reference_asset_ids: &request.reference_asset_ids,
            mask_asset_id: &request.mask_asset_id,
            character_id: &request.character_id,
            character_look_id: &request.character_look_id,
            advanced: &request.advanced,
            width: request.width,
            height: request.height,
        },
        family,
    )
}

/// The checkpoint's `image.supportsHiresFix` product fact (absent means true). A builtin reads it
/// from its compiled typed spec, which author choices cannot change, so overriding the model's
/// defaults in `user.models.jsonc` can never re-enable a pass the checkpoint cannot run. Any other
/// model reads the manifest entry the request carries.
fn model_supports_hires_fix(request: &ImageRequest) -> bool {
    match crate::registry::builtin_model_spec(&request.model) {
        Some(spec) => spec.product.supports_hires_fix(),
        None => crate::registry::supports_hires_fix(request.model_manifest_entry.get("image")),
    }
}

/// The post-first-pass target side: `dimension × upscale_by`, rounded. Shared by the enqueue
/// gate above and the worker's HiresFixPlan.
pub use sceneworks_options::image_request::hires_fix_target_dimension;

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

/// The ORDERED condition-image asset ids of a one-ordered-list image edit, or `None` when a carrier
/// is malformed. Re-exported from the router so the API validates the EXACT list the worker will
/// render, in the exact order — see `sceneworks_core::jobs_store::qwen_image_2_1_reference_ids` for
/// why the order is semantic rather than bookkeeping.
pub use crate::jobs_store::qwen_image_2_1_reference_ids as ordered_image_reference_ids;

/// The family-specific flattened list. FLUX.3 retains duplicate slots and refuses masks; the
/// historical Qwen helper deliberately deduplicates and treats a mask as an ordinary reference.
/// These policies must not be shared merely because both engines consume an ordered list.
pub use sceneworks_options::image_request::ordered_image_reference_ids_for_model;

/// Reject an `edit_image` / `character_image` request for a model whose route is ONE ordered
/// conditioning list, or `None` when the request is admissible (sc-24110).
///
/// The SHAPE half only. **Counting is not done here** — the cap lives in
/// `limits.maxReferenceAssets` and is enforced once, by
/// [`crate::image_limits::image_reference_limit_error`], so there is exactly one place that knows
/// what the ceiling is. What is left is everything else an ordered list can be wrong about:
///
///   1. **no references at all.** The conditioned modes ARE the conditioning call; with nothing to
///      condition on there is no request to make. A 400 rather than a silent fall-back to
///      text-to-image, which would render something the caller did not ask for.
///   2. **a per-reference strength.** Upstream's condition images carry no strength and the engine
///      refuses anything but unset-or-1.0; this says so at enqueue in the engine's own terms.
///   3. **a malformed carrier**, which fails closed.
///
/// `None` for every model that declares no `limits.maxReferenceAssets` — the same declaration that
/// arms the cap arms this, so the two halves can never be armed separately and the image models
/// neither story touches are byte-for-byte unchanged.
///
/// Per-reference GEOMETRY and FORMAT are not judged here either — that needs the asset store, so it
/// lives beside the call site (`apps/rust-api`'s `validate_ordered_image_references`).
pub use sceneworks_options::image_request::ordered_image_reference_error;

/// The strength half of [`ordered_image_reference_error`], split out so the four carriers the UI
/// and the API have historically used for this one quantity are read in one place.
///
/// The refusal carries the ENGINE's reason, not a SceneWorks-flavored restatement of it: a
/// condition image on this route is composed at full weight because upstream's pipeline has no
/// strength input at all — there is nothing to turn down. A value of exactly 1.0 is accepted (it is
/// what "full weight" spells), and so is an absent one.
pub use sceneworks_options::image_request::ordered_image_reference_strength_error;

/// The `advanced.quantTier` label of the publisher FP8 rowwise package (sc-26241): the inference
/// `Quant::Fp8Rowwise` tier, wire key `fp8r`. It is a tier IDENTITY, never a point on the
/// `mlxQuantize` bits ladder — aliasing it to q8 would load different numerics than the request
/// named.
pub use sceneworks_options::image_request::{
    flux3_requested_tier, image_quant_tier_refusal, requests_fp8r_tier, Flux3Tier,
    FLUX3_PRECISION_CARRIERS, FLUX3_TIERED_MODEL_IDS, FP8R_QUANT_TIER,
};
/// Refuse a FLUX 3 package absent from the authoritative catalog. Tier routing is a native
/// capability, not proof that an artifact has verified pins. In particular, base q4/q8 cannot
/// borrow distilled conversion hashes or silently fall through to the default dense row.
/// The API applies this after preset/model canonicalization; workers independently verify the
/// selected package and its conversion pins before loading it.
pub fn flux3_catalog_tier_refusal(
    model: &str,
    advanced: &JsonObject,
    manifest: &Value,
) -> Option<String> {
    if !FLUX3_TIERED_MODEL_IDS.contains(&model) {
        return None;
    }
    let tier = match flux3_requested_tier(advanced) {
        Ok(tier) => tier,
        Err(reason) => return Some(format!("{model}: {reason}")),
    };
    let declared =
        manifest
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(|rows| {
                rows.iter().any(|row| {
            row.get("coRequisite").and_then(Value::as_bool) != Some(true)
                && row.get("variant").and_then(Value::as_str) == Some(tier.key())
                && !crate::model_artifacts::artifact_selection::is_pending_artifact_download(row)
                && (!matches!(tier, Flux3Tier::Q4 | Flux3Tier::Q8)
                    || crate::model_artifacts::artifact_selection::local_derivation(row).is_some())
        })
            });
    if declared {
        return None;
    }
    let evidence = if model == "flux3_klein_base" && matches!(tier, Flux3Tier::Q4 | Flux3Tier::Q8) {
        " Verified base conversion output size and SHA-256 pins are required before installation \
         can be offered. Distilled pins cannot be reused."
    } else {
        " Install a catalog-declared package with verified artifact pins."
    };
    Some(format!("{model}: the {} package has no verified catalog artifact.{evidence} No dense fallback is permitted.", tier.key()))
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
        assert!(request.style_id.is_none());
        assert_eq!(request.fit_mode, "crop");
        assert_eq!(request.prompt, "");
        assert!(request.seed.is_none());
        assert!(request.seeds.is_empty());
        assert!(request.loras.is_empty());
        assert!(request.advanced.is_empty());
        // No `upscale` object ⇒ the disabled default (the Image Studio toggle is off).
        assert!(request.upscale.is_disabled());
        assert!(request.hires_fix.is_disabled());
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
    fn parses_hires_fix_request_and_malformed_payload_defaults_off() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p",
            "hiresFix": {
                "enabled": true,
                "steps": 18,
                "denoisingStrength": 0.6,
                "upscaleBy": 2.5,
                "cfgScale": 5.5
            }
        })));
        assert!(request.hires_fix.enabled);
        assert_eq!(request.hires_fix.steps, 18);
        assert_eq!(request.hires_fix.effective_denoising_strength(), 0.6);
        assert_eq!(request.hires_fix.effective_upscale_by(), 2.5);
        assert_eq!(request.hires_fix.cfg_scale, Some(5.5));

        let bad = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "hiresFix": "yes please"
        })));
        assert!(bad.hires_fix.is_disabled());
    }

    /// RealVisXL Lightning's Hires.fix exclusion is its typed `image.supportsHiresFix: false`
    /// fact, which author choices cannot change: overriding its defaults (or trying to override
    /// the fact itself) in user.models.jsonc still refuses Hires.fix, while a model without the
    /// fact is unaffected.
    #[test]
    fn hires_fix_refusal_follows_the_immutable_supports_hires_fix_fact() {
        let hires = |model: &str, entry: Value| {
            ImageRequest::from_payload(&payload(json!({
                "projectId": "p",
                "model": model,
                "mode": "text_to_image",
                "width": 1024,
                "height": 1024,
                "modelManifestEntry": entry,
                "hiresFix": { "enabled": true, "upscaleBy": 1.5 }
            })))
        };
        let lightning = crate::registry::builtin_model_spec("realvisxl_lightning").unwrap();
        assert!(!lightning.product.supports_hires_fix());
        let user = json!({
            "defaults": { "steps": 4 },
            "image": { "supportsHiresFix": true },
            "ui": { "label": "My Lightning" }
        });
        let projected = lightning.project(Some(&user));
        assert_eq!(
            projected["defaults"]["steps"],
            json!(4),
            "the author choice applies"
        );
        assert_eq!(
            projected["image"]["supportsHiresFix"],
            json!(false),
            "an author choice cannot change the checkpoint fact"
        );
        let refusal = hires_fix_request_error(&hires("realvisxl_lightning", projected), "sdxl")
            .expect("Hires.fix is refused for RealVisXL Lightning");
        assert!(refusal.contains("Lightning"), "{refusal}");
        // The builtin's compiled fact wins even over a carried entry that omits it.
        assert!(
            hires_fix_request_error(&hires("realvisxl_lightning", json!({})), "sdxl").is_some()
        );

        let realvisxl = crate::registry::builtin_model_spec("realvisxl").unwrap();
        assert!(realvisxl.product.supports_hires_fix());
        assert_eq!(
            hires_fix_request_error(&hires("realvisxl", realvisxl.project(None)), "sdxl"),
            None
        );
        // A non-builtin entry carries its own fact.
        assert!(hires_fix_request_error(
            &hires(
                "user_sdxl",
                json!({ "image": { "supportsHiresFix": false } })
            ),
            "sdxl"
        )
        .is_some());
        assert_eq!(
            hires_fix_request_error(&hires("user_sdxl", json!({})), "sdxl"),
            None
        );
    }

    #[test]
    fn parses_style_id_when_present() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "styleId": "ghibli-style"
        })));
        assert_eq!(request.style_id.as_deref(), Some("ghibli-style"));

        // A blank id is treated as absent (optional_id trims/filters empties).
        let blank =
            ImageRequest::from_payload(&payload(json!({ "projectId": "p", "styleId": "  " })));
        assert!(blank.style_id.is_none());
    }

    #[test]
    fn preserves_raw_count_and_dimensions_for_explicit_boundary_validation() {
        let raw = payload(json!({"projectId":"p","count":99,"width":10,"height":99999}));
        let request = ImageRequest::from_payload(&raw);
        assert_eq!(
            (request.count, request.width, request.height),
            (99, 10, 99999)
        );
        assert!(validate_image_request_payload(&raw, &JsonObject::new()).is_err());
        let low = payload(json!({"projectId":"p","count":0}));
        assert_eq!(ImageRequest::from_payload(&low).count, 0);
        assert!(validate_image_request_payload(&low, &JsonObject::new()).is_err());
    }

    /// sc-25681: a model's declared `limits.minDimension` below the 256 sanity floor is honoured
    /// (the API admits it, so the worker must not re-size it); a declared floor above 256, or none,
    /// is enforced by the shared boundary, without resizing the parsed value.
    ///
    /// *Mutation that reds this:* clamp to the constant `MIN_DIMENSION` again.
    #[test]
    fn a_declared_floor_below_the_sanity_floor_reaches_the_request() {
        let request = |entry: Value, side: u32| {
            ImageRequest::from_payload(&payload(json!({
                "projectId": "p", "width": side, "height": side, "modelManifestEntry": entry
            })))
        };
        let iris = json!({ "limits": { "minDimension": 16, "maxDimension": 2048 } });
        assert_eq!(request(iris.clone(), 64).width, 64);
        assert_eq!(
            request(iris, 8).height,
            8,
            "parser preserves intent; the shared boundary rejects below-floor requests"
        );
        let high = json!({ "limits": { "minDimension": 512 } });
        assert_eq!(
            request(high, 300).width,
            300,
            "the sanity floor never rises"
        );
        assert_eq!(request(json!({}), 64).width, 64);
    }

    /// sc-25681: a native prompt batch maps image `k` to `prompts[k % len]` (prompt-minor) and
    /// produces `prompts × count` images; an ordinary request is unchanged.
    ///
    /// *Mutation that reds this:* index `prompts[k / count]` (prompt-major).
    #[test]
    fn a_prompt_batch_is_prompt_minor() {
        let batch = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "prompt": "a", "prompts": ["a", "b", "c"], "count": 2
        })));
        assert_eq!(batch.image_count(), 6);
        let order: Vec<&str> = (0..6).map(|k| batch.image_prompt(k)).collect();
        assert_eq!(order, vec!["a", "b", "c", "a", "b", "c"]);
        let plain = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "prompt": "solo", "count": 3
        })));
        assert!(plain.prompts.is_empty());
        assert_eq!(plain.image_count(), 3);
        assert_eq!(plain.image_prompt(2), "solo");
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

    /// sc-24114: `image_to_image` is the declared single-reference face of the ordered-list call,
    /// so it is judged exactly like the edit modes — no references is a 400 — and a strength is
    /// refused on EVERY mode that carries references, because the router claims a non-empty list
    /// mode-independently (a `text_to_image` + `referenceAssetId` request is the same engine call).
    #[test]
    fn ordered_reference_shape_gate_covers_image_to_image_and_every_referenced_mode() {
        let entry = payload(json!({ "limits": { "maxReferenceAssets": 10 } }));
        let check =
            |body: Value| ordered_image_reference_error("qwen_image_2_1", &payload(body), &entry);

        let empty = check(json!({ "mode": "image_to_image" }));
        assert!(
            empty
                .as_deref()
                .is_some_and(|m| m.contains("supplies none")),
            "{empty:?}"
        );
        assert_eq!(
            check(json!({ "mode": "image_to_image", "referenceAssetId": "r" })),
            None
        );
        for mode in ["image_to_image", "text_to_image", "image_generation"] {
            let refused = check(json!({
                "mode": mode, "referenceAssetId": "r", "advanced": { "strength": 0.4 }
            }));
            assert!(
                refused
                    .as_deref()
                    .is_some_and(|m| m.contains("no strength")),
                "{mode}: {refused:?}"
            );
        }
        // A plain text-to-image request is untouched.
        assert_eq!(check(json!({ "mode": "text_to_image" })), None);
        assert_eq!(check(json!({ "advanced": { "strength": 0.4 } })), None);
    }
    #[test]
    fn flux3_requested_tier_accepts_each_package_spelling_and_refuses_conflicts() {
        let advanced = |value: Value| value.as_object().cloned().unwrap();
        for (value, tier) in [
            (json!({}), Flux3Tier::Bf16),
            (
                json!({"quantTier": null, "mlxQuantize": null}),
                Flux3Tier::Bf16,
            ),
            (
                json!({"mlxQuantize": 0, "mlxQuantizeExplicit": true}),
                Flux3Tier::Bf16,
            ),
            (json!({"quantTier": " BF16 "}), Flux3Tier::Bf16),
            (json!({"mlxQuantize": 4}), Flux3Tier::Q4),
            (json!({"mlxQuantize": "8"}), Flux3Tier::Q8),
            (json!({"quantTier": "q8", "mlxQuantize": 8}), Flux3Tier::Q8),
            (json!({"quantTier": "fp8r"}), Flux3Tier::Fp8Rowwise),
            (json!({"quantTier": " FP8R "}), Flux3Tier::Fp8Rowwise),
            (
                json!({"quantTier": "nvfp4", "mlxQuantize": null}),
                Flux3Tier::Nvfp4,
            ),
            (json!({"convRot": false}), Flux3Tier::Bf16),
        ] {
            assert_eq!(
                flux3_requested_tier(&advanced(value.clone())),
                Ok(tier),
                "{value}"
            );
        }
        // An absent tier is the dense package, never the app-wide q8 default.
        assert_ne!(
            flux3_requested_tier(&advanced(json!({}))),
            Ok(Flux3Tier::Q8)
        );
        for value in [
            json!({"mlxQuantize": 6}),
            json!({"mlxQuantize": "x"}),
            json!({"quantTier": "int8-convrot"}),
            json!({"quantTier": 8}),
            json!({"convRot": true}),
            json!({"quantTier": "fp8r", "mlxQuantize": 8}),
            json!({"quantTier": "nvfp4", "mlxQuantize": 4}),
            json!({"quantTier": "q4", "mlxQuantize": 8}),
        ] {
            assert!(
                flux3_requested_tier(&advanced(value.clone())).is_err(),
                "{value}"
            );
        }
    }

    #[test]
    fn flux3_tiers_name_their_lane_marker_and_recipe_bits_by_identity() {
        assert_eq!(Flux3Tier::Fp8Rowwise.key(), "fp8r");
        assert_eq!(Flux3Tier::Fp8Rowwise.recipe_bits(), None);
        assert_eq!(Flux3Tier::Nvfp4.recipe_bits(), None);
        assert_eq!(Flux3Tier::Q4.recipe_bits(), Some(4));
        assert_eq!(Flux3Tier::Q8.recipe_bits(), Some(8));
        assert_eq!(
            Flux3Tier::Fp8Rowwise.required_worker_capability(),
            Some("fp8r")
        );
        assert_eq!(
            Flux3Tier::Nvfp4.required_worker_capability(),
            Some("flux3_nvfp4")
        );
        assert_eq!(Flux3Tier::Q8.required_worker_capability(), None);
        let mlx: Vec<_> = Flux3Tier::ALL
            .into_iter()
            .filter(|t| t.served_by_mlx())
            .collect();
        assert_eq!(mlx, [Flux3Tier::Bf16, Flux3Tier::Q4, Flux3Tier::Q8]);
        assert!(Flux3Tier::ALL.into_iter().all(Flux3Tier::served_by_candle));
        for tier in Flux3Tier::ALL {
            assert_eq!(Flux3Tier::from_key(tier.key()), Some(tier));
        }
    }

    #[test]
    fn flux3_catalog_requires_a_real_row_for_the_requested_package() {
        let advanced = |tier| payload(json!({"quantTier":tier}));
        let dense_only = json!({"downloads":[{"variant":"bf16","default":true}]});
        assert!(
            flux3_catalog_tier_refusal("flux3_klein_base", &advanced("bf16"), &dense_only)
                .is_none()
        );
        for tier in ["q4", "q8"] {
            let reason =
                flux3_catalog_tier_refusal("flux3_klein_base", &advanced(tier), &dense_only)
                    .unwrap();
            assert!(
                reason.contains(tier) && reason.contains("SHA-256") && !reason.contains("403"),
                "{reason}"
            );
            assert!(reason.contains("No dense fallback"), "{reason}");
            let malformed = json!({"downloads":[{"variant":tier,"localDerivation":{}}]});
            assert!(
                flux3_catalog_tier_refusal("flux3_klein_base", &advanced(tier), &malformed)
                    .is_some()
            );
            // SYNTHETIC pins test the shared path only, never production base weight evidence.
            let pinned = json!({"downloads":[{"variant":tier,"localDerivation":{
                "fromVariant":"bf16","conversion":"fixture-only","weightsFile":"model.safetensors",
                "weightsBytes":1,"weightsSha256":"a".repeat(64)
            }}]});
            assert!(
                flux3_catalog_tier_refusal("flux3_klein_base", &advanced(tier), &pinned).is_none()
            );
        }
        for tier in ["fp8r", "nvfp4"] {
            assert!(
                flux3_catalog_tier_refusal("flux3_klein_base", &advanced(tier), &dense_only)
                    .is_some()
            );
            let declared = json!({"downloads":[{"variant":tier}]});
            assert!(
                flux3_catalog_tier_refusal("flux3_klein_base", &advanced(tier), &declared)
                    .is_none()
            );
        }
    }

    #[test]
    fn image_quant_tier_refusal_is_flux3_shaped_and_fp8r_is_flux3_only() {
        let advanced = |value: Value| value.as_object().cloned().unwrap();
        assert_eq!(
            image_quant_tier_refusal("z_image", &advanced(json!({}))),
            None
        );
        assert_eq!(
            image_quant_tier_refusal("z_image", &advanced(json!({"quantTier": "nvfp4"}))),
            None
        );
        assert_eq!(
            image_quant_tier_refusal("flux3_klein", &advanced(json!({"quantTier": "fp8r"}))),
            None
        );
        assert_eq!(
            image_quant_tier_refusal("flux3_klein_base", &advanced(json!({"quantTier": "fp8r"}))),
            None
        );
        for model in ["flux2_klein_9b", "krea_2_turbo", "z_image"] {
            let reason = image_quant_tier_refusal(model, &advanced(json!({"quantTier": "fp8r"})))
                .expect("non-FLUX3 fp8r must be refused");
            assert!(reason.contains(model), "{reason}");
        }
        let reason = image_quant_tier_refusal(
            "flux3_klein",
            &advanced(json!({"quantTier": "fp8r", "mlxQuantize": 8})),
        )
        .expect("conflict refused");
        assert!(reason.contains("flux3_klein"), "{reason}");
    }
}

pub use sceneworks_options::image_request::unsigned_or;

/// Count validation shared by typed HTTP requests and raw worker payloads.
pub use sceneworks_options::image_request::image_count_error;

/// Validate raw queued intent before tolerant parsing can hide malformed fields. Uses the canonical
/// manifest's existing per-model envelope and FLUX.3's reference-derived geometry contract.
pub use sceneworks_options::image_request::validate_image_request_payload;

#[cfg(test)]
mod request_validation_tests {
    use super::*;
    use serde_json::json;
    fn check(value: Value, entry: Value) -> Result<(), String> {
        validate_image_request_payload(value.as_object().unwrap(), entry.as_object().unwrap())
    }
    #[test]
    fn rejects_bad_counts_dimensions_and_upscale_without_clamping() {
        for value in [
            json!({"count":0}),
            json!({"count":9}),
            json!({"count":"2"}),
            json!({"width":0}),
            json!({"width":4097}),
            json!({"height":-1}),
            json!({"width":1.5}),
            json!({"upscale":{"enabled":true,"factor":3}}),
        ] {
            assert!(check(value, json!({})).is_err());
        }
        assert!(check(
            json!({"width":1024,"height":768,"count":8,"upscale":{"enabled":true,"factor":4}}),
            json!({})
        )
        .is_ok());
    }
    #[test]
    fn declared_small_geometry_is_preserved_and_enforced() {
        let entry = json!({"limits":{"minDimension":16,"maxDimension":2048,"requiresDimensionsMultipleOf":16}});
        let payload = json!({"width":16,"height":32,"count":1,"modelManifestEntry":entry});
        assert!(check(payload.clone(), entry.clone()).is_ok());
        let parsed = ImageRequest::from_payload(payload.as_object().unwrap());
        assert_eq!((parsed.width, parsed.height), (16, 32));
        assert!(check(json!({"width":17,"height":32}), entry.clone()).is_err());
        assert!(check(json!({"width":16,"height":32}), json!({})).is_err());
        assert!(check(json!({"width":2049,"height":32}), entry).is_err());
    }
    #[test]
    fn flux3_reference_floor_and_omitted_sides_keep_provider_contract() {
        assert!(check(
            json!({"model":"flux3_klein","referenceAssetId":"a","width":16}),
            json!({"limits":{"minDimension":64}})
        )
        .is_ok());
        assert!(check(
            json!({"model":"flux3_klein","referenceAssetId":"a","width":17}),
            json!({})
        )
        .is_err());
    }
}
