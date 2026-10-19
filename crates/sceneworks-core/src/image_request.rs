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

pub mod flux3;

use serde_json::Value;

use crate::contracts::{HiresFixRequest, ImageUpscaleRequest, JsonObject};
use crate::payload_util::{
    array_or_empty, clamped_u32, declared_resolution, int_array, nonempty_string_or,
    object_or_empty, optional_i64, optional_id, string_list, string_or,
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
const MIN_COUNT: u32 = 1;
/// `pub(crate)` so `workflow_share`'s pose budget can be derived from it rather than restate it:
/// a pose set is the pose lane's batch size (one image per pose), so this is the ceiling it is
/// measured against.
pub(crate) const MAX_COUNT: u32 = 8;

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
pub const MAX_JOB_POSES: usize = 8 * MAX_COUNT as usize;

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
    /// A NATIVE prompt batch (sc-25681, `image.maxPromptBatchImages`): the prompts one engine call
    /// renders together, verbatim. Empty for an ordinary request. When set, `prompt` is `prompts[0]`
    /// (the API enforces it) and image `k` renders `prompts[k % prompts.len()]` — see
    /// [`Self::image_prompt`].
    pub prompts: Vec<String>,
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
    /// The prompt image `index` of this job rendered: `prompt` for an ordinary request, and
    /// `prompts[index % prompts.len()]` for a native prompt batch (prompt-minor order — the engine
    /// renders the whole prompt list once per `count` repetition).
    pub fn image_prompt(&self, index: usize) -> &str {
        if self.prompts.is_empty() {
            &self.prompt
        } else {
            &self.prompts[index % self.prompts.len()]
        }
    }

    /// The images this request produces: `count`, or `prompts × count` for a native prompt batch.
    pub fn image_count(&self) -> u32 {
        self.count
            .saturating_mul(u32::try_from(self.prompts.len().max(1)).unwrap_or(u32::MAX))
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
        // The sanity floor yields to a model's DECLARED `limits.minDimension` below it (sc-25681):
        // the API admits a declared envelope down to its own floor (Iris-3B renders from 16 px,
        // Qwen-Image 2.1 from 32), and the blanket 256 here silently re-sized those requests on the
        // worker — an accepted size that never reached the engine. Every model that declares no
        // floor (or one at/above 256) keeps the historical clamp.
        let min_dimension = crate::video_request::image_dimension_envelope(&model_manifest_entry)
            .min
            .map_or(MIN_DIMENSION, |declared| declared.min(MIN_DIMENSION));
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
            count: clamped_u32(payload.get("count"), default_count, MIN_COUNT, MAX_COUNT),
            seed: optional_i64(payload, "seed"),
            seeds: int_array(payload, "seeds"),
            width: clamped_u32(
                payload.get("width"),
                default_width,
                min_dimension,
                MAX_DIMENSION,
            ),
            height: clamped_u32(
                payload.get("height"),
                default_height,
                min_dimension,
                MAX_DIMENSION,
            ),
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
/// malformed value falls back to the disabled default, so a bad payload never aborts a
/// generation — it just skips upscaling.
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
pub fn ordered_image_reference_ids_for_model(
    model: &str,
    payload: &JsonObject,
) -> Option<Vec<String>> {
    if flux3::is_model(model) {
        flux3::reference_ids(payload).ok()
    } else {
        ordered_image_reference_ids(payload)
    }
}

/// Reject an `edit_image` / `character_image` request for a model whose route is ONE ordered
/// conditioning list, or `None` when the request is admissible (sc-24110).
///
/// The SHAPE half only. **Counting is not done here** — the cap lives in
/// `limits.maxReferenceAssets` and is enforced once, by
/// [`crate::video_request::image_reference_limit_error`], so there is exactly one place that knows
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
pub fn ordered_image_reference_error(
    model: &str,
    payload: &JsonObject,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    if flux3::is_model(model) {
        return flux3::parse(payload).err();
    }
    let cap = crate::video_request::image_max_reference_assets(model_manifest_entry)?;
    let mode = payload.get("mode").and_then(Value::as_str).unwrap_or("");
    // sc-24114: `image_to_image` is the declared single-reference face of the same call, so it is
    // a CONDITIONED mode exactly like the edit modes (it needs at least one reference).
    let conditioned_mode = matches!(mode, "edit_image" | "character_image" | "image_to_image");
    let Some(ids) = ordered_image_reference_ids(payload) else {
        return Some(format!(
            "{model}: one of sourceAssetId / maskAssetId / referenceAssetId / referenceAssetIds is \
             malformed — each must be a non-blank asset id string, and referenceAssetIds must be an \
             array of them."
        ));
    };
    if ids.is_empty() {
        if !conditioned_mode {
            // A plain text-to-image request: nothing to condition on, nothing to judge.
            return None;
        }
        return Some(format!(
            "{model} conditions on an ordered list of 1 to {cap} reference images, but this \
             {mode} request supplies none. Add at least one reference, or send a text-to-image \
             request instead."
        ));
    }
    // A non-empty list IS the conditioning call whatever the mode says (the router claims it
    // mode-independently), so a strength is refused on every mode that carries references.
    ordered_image_reference_strength_error(model, payload)
}

/// The strength half of [`ordered_image_reference_error`], split out so the four carriers the UI
/// and the API have historically used for this one quantity are read in one place.
///
/// The refusal carries the ENGINE's reason, not a SceneWorks-flavored restatement of it: a
/// condition image on this route is composed at full weight because upstream's pipeline has no
/// strength input at all — there is nothing to turn down. A value of exactly 1.0 is accepted (it is
/// what "full weight" spells), and so is an absent one.
fn ordered_image_reference_strength_error(model: &str, payload: &JsonObject) -> Option<String> {
    let advanced = payload.get("advanced").and_then(Value::as_object);
    for key in ["strength", "referenceStrength"] {
        let Some(value) = payload
            .get(key)
            .or_else(|| advanced.and_then(|advanced| advanced.get(key)))
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        let Some(strength) = value.as_f64() else {
            return Some(format!("{model}: `{key}` must be a number, or absent."));
        };
        if (strength - 1.0).abs() > f64::EPSILON {
            return Some(format!(
                "{model} conditions on a reference at full weight — upstream's condition images \
                 have no strength, so `{key}: {strength}` has nothing to apply to. Drop the field \
                 (or send 1.0)."
            ));
        }
    }
    None
}

/// The `advanced.quantTier` label of the publisher FP8 rowwise package (sc-26241): the inference
/// `Quant::Fp8Rowwise` tier, wire key `fp8r`. It is a tier IDENTITY, never a point on the
/// `mlxQuantize` bits ladder — aliasing it to q8 would load different numerics than the request
/// named.
pub const FP8R_QUANT_TIER: &str = "fp8r";

/// The image models whose precision is one of the [`Flux3Tier`] packages (sc-26241). FLUX 3 Image
/// Klein (distilled) is the only catalog FLUX 3 model; it is also the only model that ships a
/// publisher FP8R package, so `quantTier: "fp8r"` anywhere else is refused rather than run at some
/// other precision.
pub const FLUX3_TIERED_MODEL_IDS: &[&str] = &["flux3_klein", "flux3_klein_base"];

/// The `advanced` keys that select a FLUX 3 precision. A consumer that judges the rest of the
/// request surface strips exactly these once [`flux3_requested_tier`] has accepted them.
pub const FLUX3_PRECISION_CARRIERS: &[&str] =
    &["quantTier", "mlxQuantize", "mlxQuantizeExplicit", "convRot"];

/// One FLUX 3 Image Klein weight package (sc-26241). Every tier is a DIFFERENT artifact, never a
/// load-time transform of another:
///
/// * `bf16` — the dense package at the pinned snapshot root (both backends);
/// * `q4` / `q8` — native packages converted on this machine from that dense package
///   (`localDerivation`): Candle GGML Q4_1 / Q8_0 off-Mac, MLX affine W4/W8 (group 64) on macOS;
/// * `fp8r` / `nvfp4` — the publisher packages at `variants/<tier>` inside the same snapshot, loaded
///   as shipped by the Candle provider on CUDA hardware that clears their device floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flux3Tier {
    Bf16,
    Q4,
    Q8,
    Fp8Rowwise,
    Nvfp4,
}

impl Flux3Tier {
    pub const ALL: [Self; 5] = [
        Self::Bf16,
        Self::Q4,
        Self::Q8,
        Self::Fp8Rowwise,
        Self::Nvfp4,
    ];

    /// The tier key every consumer names it by (manifest `variant`, fit gate, telemetry).
    pub const fn key(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Fp8Rowwise => FP8R_QUANT_TIER,
            Self::Nvfp4 => "nvfp4",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.key() == key)
    }

    /// Whether the native MLX provider has this package (its descriptor advertises q4/q8 only).
    pub const fn served_by_mlx(self) -> bool {
        matches!(self, Self::Bf16 | Self::Q4 | Self::Q8)
    }

    /// Whether the native Candle provider has this package (q4/q8 on every build; fp8r/nvfp4 on
    /// CUDA, gated per worker by [`Self::required_worker_capability`]).
    pub const fn served_by_candle(self) -> bool {
        true
    }

    /// The `mlxQuantize` bit width the recipe records, when the tier has an honest one.
    pub const fn recipe_bits(self) -> Option<i64> {
        match self {
            Self::Q4 => Some(4),
            Self::Q8 => Some(8),
            Self::Bf16 | Self::Fp8Rowwise | Self::Nvfp4 => None,
        }
    }

    /// The device-floor marker a Candle worker must advertise to be handed this tier: FP8R needs
    /// sm_89+ and NVFP4 sm_120+ (the inference `DitFormat::check_device` floors). `None` for the
    /// tiers every Candle device runs.
    pub const fn required_worker_capability(self) -> Option<&'static str> {
        match self {
            Self::Fp8Rowwise => Some(FP8R_QUANT_TIER),
            Self::Nvfp4 => Some("flux3_nvfp4"),
            Self::Bf16 | Self::Q4 | Self::Q8 => None,
        }
    }

    /// A publisher package shipped inside the pinned snapshot at `variants/<key>`.
    pub const fn is_publisher_variant(self) -> bool {
        matches!(self, Self::Fp8Rowwise | Self::Nvfp4)
    }

    /// A package converted locally from the dense one (a `localDerivation` catalog row).
    pub const fn is_local_derivation(self) -> bool {
        matches!(self, Self::Q4 | Self::Q8)
    }
}

/// Whether `advanced.quantTier` names the FP8R tier (trimmed, ASCII case-insensitive — the same
/// normalization every other named-tier reader applies).
pub fn requests_fp8r_tier(advanced: &JsonObject) -> bool {
    advanced
        .get("quantTier")
        .and_then(Value::as_str)
        .is_some_and(|tier| tier.trim().eq_ignore_ascii_case(FP8R_QUANT_TIER))
}

/// The FLUX 3 package a request's precision carriers name. Both public spellings are accepted —
/// the named `quantTier` and the studio's `mlxQuantize` bit width (`0` dense, `4`, `8`) — and they
/// must agree when both are present. Absent carriers mean the dense `bf16` package, never the
/// app-wide `q8` default: a FLUX 3 tier is an installed artifact the request chooses.
pub fn flux3_requested_tier(advanced: &JsonObject) -> Result<Flux3Tier, String> {
    let named = match advanced.get("quantTier") {
        None | Some(Value::Null) => None,
        Some(Value::String(tier)) if tier.trim().is_empty() => None,
        Some(Value::String(tier)) => {
            let key = tier.trim().to_ascii_lowercase();
            Some(match key.as_str() {
                "dense" => Flux3Tier::Bf16,
                other => Flux3Tier::from_key(other).ok_or_else(|| {
                    format!(
                        "advanced.quantTier {tier:?} is not a FLUX 3 Image Klein package; the \
                         packages are bf16, q4, q8, fp8r and nvfp4"
                    )
                })?,
            })
        }
        Some(other) => {
            return Err(format!(
                "advanced.quantTier must be a tier name, not {other}"
            ))
        }
    };
    let bits = match advanced.get("mlxQuantize") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let bits = value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
                .ok_or_else(|| format!("advanced.mlxQuantize must be an integer, not {value}"))?;
            Some(match bits {
                bits if bits <= 0 => Flux3Tier::Bf16,
                4 => Flux3Tier::Q4,
                8 => Flux3Tier::Q8,
                other => {
                    return Err(format!(
                        "advanced.mlxQuantize {other} is not a FLUX 3 Image Klein package; the bit \
                         tiers are 0 (bf16), 4 and 8"
                    ))
                }
            })
        }
    };
    if advanced
        .get("convRot")
        .is_some_and(|convrot| !matches!(convrot, Value::Null | Value::Bool(false)))
    {
        return Err("advanced.convRot is not a FLUX 3 Image Klein package".to_owned());
    }
    match (named, bits) {
        (None, None) => Ok(Flux3Tier::Bf16),
        (Some(tier), None) | (None, Some(tier)) => Ok(tier),
        (Some(named), Some(bits)) if named == bits => Ok(named),
        (Some(named), Some(bits)) => Err(format!(
            "advanced.quantTier {:?} and advanced.mlxQuantize ({}) name different FLUX 3 \
             packages; send one",
            named.key(),
            bits.key()
        )),
    }
}

/// The request-shape rule for named precision tiers, shared by the API (typed 400), routing (the
/// candle gap wording) and the worker trust boundary (raw queue writes, stored jobs). `None` =
/// acceptable; `Some(reason)` = refuse.
///
/// * A [`FLUX3_TIERED_MODEL_IDS`] model must name exactly one of its packages
///   ([`flux3_requested_tier`]).
/// * Only those models ship a publisher FP8R package.
///
/// Backend capability (MLX serves bf16/q4/q8; FP8R/NVFP4 need a CUDA worker past their device
/// floor) is NOT judged here: routing owns which lane and worker may run the tier.
pub fn image_quant_tier_refusal(model: &str, advanced: &JsonObject) -> Option<String> {
    if FLUX3_TIERED_MODEL_IDS.contains(&model) {
        return flux3_requested_tier(advanced)
            .err()
            .map(|reason| format!("{model}: {reason}"));
    }
    requests_fp8r_tier(advanced).then(|| {
        format!(
            "advanced.quantTier \"{FP8R_QUANT_TIER}\" selects a publisher FP8 rowwise package, and \
             {model} ships none (only {FLUX3_TIERED_MODEL_IDS:?} do)"
        )
    })
}

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
        " Base repository file access is blocked (HTTP 403); verified base conversion output size \
         and SHA-256 pins are required before installation can be offered. Distilled pins cannot \
         be reused."
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

    /// sc-25681: a model's declared `limits.minDimension` below the 256 sanity floor is honoured
    /// (the API admits it, so the worker must not re-size it); a declared floor above 256, or none,
    /// keeps the historical clamp.
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
            16,
            "still clamped to the declared floor"
        );
        let high = json!({ "limits": { "minDimension": 512 } });
        assert_eq!(
            request(high, 300).width,
            300,
            "the sanity floor never rises"
        );
        assert_eq!(request(json!({}), 64).width, 256);
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
                reason.contains(tier) && reason.contains("SHA-256") && reason.contains("403"),
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
