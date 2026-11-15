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
use crate::resolved_options::{BindingError, BindingInput, Bound, Origin};

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

/// The two caller-selectable PiD checkpoint fields; filename confinement remains a native
/// artifact-selection check after binding.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PidCheckpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// A nested control-weight field retains the shape distinction needed by tolerant legacy
/// routes and by SDXL's strict refusal. Malformed content is never used as a path or revision.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ControlWeightField<T> {
    #[default]
    Absent,
    Null,
    Valid(T),
    Invalid,
}

impl<T> ControlWeightField<T> {
    pub fn valid(&self) -> Option<&T> {
        match self {
            Self::Valid(value) => Some(value),
            _ => None,
        }
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid)
    }
}

impl ControlWeightField<String> {
    pub fn valid_trimmed(&self) -> Option<&str> {
        self.valid()
            .map(String::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
    }
}

/// Typed view of the resolved control-weight carrier. Parsing never grants trust to `path`,
/// `revision`, or `_catalogAuthorized`; native route checks still confine and authorize them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlWeightsFields {
    pub repo: ControlWeightField<String>,
    pub filename: ControlWeightField<String>,
    pub path: ControlWeightField<String>,
    pub revision: ControlWeightField<String>,
    pub overlay_id: ControlWeightField<String>,
    pub catalog_authorized: ControlWeightField<bool>,
}

impl<'de> serde::Deserialize<'de> for ControlWeightsFields {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let map = JsonObject::deserialize(deserializer)?;
        fn text(map: &JsonObject, key: &str) -> ControlWeightField<String> {
            match map.get(key) {
                None => ControlWeightField::Absent,
                Some(Value::Null) => ControlWeightField::Null,
                Some(Value::String(value)) => ControlWeightField::Valid(value.clone()),
                Some(_) => ControlWeightField::Invalid,
            }
        }
        let catalog_authorized = match map.get("_catalogAuthorized") {
            None => ControlWeightField::Absent,
            Some(Value::Null) => ControlWeightField::Null,
            Some(Value::Bool(value)) => ControlWeightField::Valid(*value),
            Some(_) => ControlWeightField::Invalid,
        };
        Ok(Self {
            repo: text(&map, "repo"),
            filename: text(&map, "filename"),
            path: text(&map, "path"),
            revision: text(&map, "revision"),
            overlay_id: text(&map, "overlayId"),
            catalog_authorized,
        })
    }
}

impl serde::Serialize for ControlWeightsFields {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = JsonObject::new();
        fn put<T: serde::Serialize>(
            map: &mut JsonObject,
            key: &str,
            field: &ControlWeightField<T>,
        ) {
            match field {
                ControlWeightField::Valid(value) => {
                    map.insert(
                        key.to_owned(),
                        serde_json::to_value(value).expect("typed field serializes"),
                    );
                }
                ControlWeightField::Null => {
                    map.insert(key.to_owned(), Value::Null);
                }
                ControlWeightField::Absent | ControlWeightField::Invalid => {}
            }
        }
        put(&mut map, "repo", &self.repo);
        put(&mut map, "filename", &self.filename);
        put(&mut map, "path", &self.path);
        put(&mut map, "revision", &self.revision);
        put(&mut map, "overlayId", &self.overlay_id);
        put(&mut map, "_catalogAuthorized", &self.catalog_authorized);
        serde::Serialize::serialize(&map, serializer)
    }
}

/// Image options lowered once from the shared resolver's canonical effective request.
/// Each value retains whether the caller supplied it or the resolver applied a default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageAdvanced {
    pub sampler: Option<Bound<String>>,
    pub scheduler: Option<Bound<String>>,
    pub scheduler_shift: Option<Bound<f64>>,
    /// SenseNova interleave reads the legacy alias itself, independently of schedulerShift.
    pub interleave_timestep_shift: Option<Bound<f64>>,
    pub steps: Option<Bound<i64>>,
    pub guidance_scale: Option<Bound<f64>>,
    pub guidance_method: Option<Bound<String>>,
    pub cfg_interval: Option<Bound<Vec<f64>>>,
    pub caption_overflow: Option<Bound<String>>,
    pub num_inference_steps: Option<Bound<i64>>,
    pub enhance_prompt: Option<Bound<bool>>,
    pub enhance_max_tokens: Option<Bound<i64>>,
    pub enhance_temperature: Option<Bound<f64>>,
    pub decoder: Option<Bound<String>>,
    pub use_pid: Option<Bound<bool>>,
    pub pid_target: Option<Bound<String>>,
    pub pid_checkpoint: Option<Bound<PidCheckpoint>>,
    pub pid_gemma: Option<Bound<String>>,
    pub ip_adapter_scale: Option<Bound<f64>>,
    pub ip_adapter_repo: Option<Bound<String>>,
    pub controlnet_conditioning_scale: Option<Bound<f64>>,
    pub true_cfg_scale: Option<Bound<f64>>,
    pub strength: Option<Bound<f64>>,
    pub reference_strength: Option<Bound<f64>>,
    pub denoising_strength: Option<Bound<f64>>,
    pub image_strength: Option<Bound<f64>>,
    pub ip_adapter_scale_legacy: Option<Bound<f64>>,
    pub text_style_gain: Option<Bound<f64>>,
    pub view_angle: Option<Bound<String>>,
    pub view_angles: Option<Bound<Vec<String>>>,
    pub poses: Option<Bound<Vec<Value>>>,
    pub face_restore: Option<Bound<bool>>,
    pub control_mode: Option<Bound<String>>,
    pub control_image: Option<Bound<String>>,
    pub control_weights: Option<Bound<ControlWeightsFields>>,
    pub control_scale: Option<Bound<f64>>,
    pub open_pose_scale: Option<Bound<f64>>,
    pub cn_scale: Option<Bound<f64>>,
    pub tile: Option<Bound<i64>>,
    pub overlap: Option<Bound<i64>>,
    pub prompt: Option<Bound<String>>,
    pub negative_prompt: Option<Bound<String>>,
    pub tile_control_net_repo: Option<Bound<String>>,
    pub id_weight: Option<Bound<f64>>,
    pub timestep_to_start_cfg: Option<Bound<i64>>,
    pub angle_set: Option<Bound<bool>>,
    pub keypoint_collection_id: Option<Bound<String>>,
    pub system_message: Option<Bound<String>>,
    pub image_guidance_scale: Option<Bound<f64>>,
    pub max_images: Option<Bound<i64>>,
    pub max_new_tokens: Option<Bound<i64>>,
    pub think_mode: Option<Bound<bool>>,
    pub phases: Option<Bound<Vec<Value>>>,
    pub flash_attn: Option<Bound<bool>>,
    /// Outer `None` means absent; inner `None` means an authored null that suppresses
    /// fallback to the verified portable bound tier.
    pub mlx_quantize: Option<Bound<Option<i64>>>,
    pub mlx_quantize_explicit: Option<Bound<bool>>,
    pub conv_rot: Option<Bound<bool>>,
    pub quant_tier: Option<Bound<String>>,
    pub quant: Option<Bound<String>>,
    pub model_path: Option<Bound<String>>,
    pub text_encoder_model: Option<Bound<String>>,
    pub transparent_background: Option<Bound<bool>>,
    pub flux3_layout: Option<Bound<JsonObject>>,
    pub flux3_reference_ids: Option<Bound<Vec<String>>>,
    pub moss_image: Option<Bound<gen_core::moss_image::MossImageOptions>>,
    pub moss_image_edit: Option<Bound<crate::moss_image::EditOptions>>,
    pub moss_image_advanced: Option<Bound<crate::moss_image_advanced::EditAdvanced>>,
}

impl ImageAdvanced {
    /// Serialize the resolved typed fields at the provider and workflow wire boundary.
    /// This never reads the authored option bag and does not parse a serialized value back.
    pub fn provider_wire(&self) -> JsonObject {
        let mut wire = JsonObject::new();
        macro_rules! put {
            ($field:ident, $key:literal) => {
                if let Some(value) = &self.$field {
                    wire.insert(
                        $key.to_owned(),
                        serde_json::to_value(&value.value).expect("bound image option serializes"),
                    );
                }
            };
        }
        put!(sampler, "sampler");
        put!(scheduler, "scheduler");
        put!(scheduler_shift, "schedulerShift");
        put!(interleave_timestep_shift, "timestepShift");
        put!(steps, "steps");
        put!(guidance_scale, "guidanceScale");
        put!(guidance_method, "guidanceMethod");
        put!(cfg_interval, "cfgInterval");
        put!(caption_overflow, "captionOverflow");
        put!(num_inference_steps, "numInferenceSteps");
        put!(enhance_prompt, "enhancePrompt");
        put!(enhance_max_tokens, "enhanceMaxTokens");
        put!(enhance_temperature, "enhanceTemperature");
        put!(decoder, "decoder");
        put!(use_pid, "usePid");
        put!(pid_target, "pidTarget");
        put!(pid_checkpoint, "pidCheckpoint");
        put!(pid_gemma, "pidGemma");
        put!(ip_adapter_scale, "ipAdapterScale");
        put!(ip_adapter_repo, "ipAdapterRepo");
        put!(controlnet_conditioning_scale, "controlnetConditioningScale");
        put!(true_cfg_scale, "trueCfgScale");
        put!(strength, "strength");
        put!(reference_strength, "referenceStrength");
        put!(denoising_strength, "denoisingStrength");
        put!(image_strength, "imageStrength");
        put!(ip_adapter_scale_legacy, "ip_adapter_scale");
        put!(text_style_gain, "textStyleGain");
        put!(view_angle, "viewAngle");
        put!(view_angles, "viewAngles");
        put!(poses, "poses");
        put!(face_restore, "faceRestore");
        put!(control_mode, "controlMode");
        put!(control_image, "controlImage");
        put!(control_weights, "controlWeights");
        put!(control_scale, "controlScale");
        put!(open_pose_scale, "openPoseScale");
        put!(cn_scale, "cnScale");
        put!(tile, "tile");
        put!(overlap, "overlap");
        put!(prompt, "prompt");
        put!(negative_prompt, "negativePrompt");
        put!(tile_control_net_repo, "tileControlNetRepo");
        put!(id_weight, "idWeight");
        put!(timestep_to_start_cfg, "timestepToStartCfg");
        put!(angle_set, "angleSet");
        put!(keypoint_collection_id, "keypointCollectionId");
        put!(system_message, "systemMessage");
        put!(image_guidance_scale, "imageGuidanceScale");
        put!(max_images, "maxImages");
        put!(max_new_tokens, "maxNewTokens");
        put!(think_mode, "thinkMode");
        put!(phases, "phases");
        put!(flash_attn, "flashAttn");
        put!(mlx_quantize, "mlxQuantize");
        put!(mlx_quantize_explicit, "mlxQuantizeExplicit");
        put!(conv_rot, "convRot");
        put!(quant_tier, "quantTier");
        put!(quant, "quant");
        put!(model_path, "modelPath");
        put!(text_encoder_model, "textEncoderModel");
        put!(transparent_background, "transparentBackground");
        put!(flux3_layout, "flux3_layout");
        put!(flux3_reference_ids, "flux3_reference_ids");
        put!(moss_image, "mossImage");
        put!(moss_image_edit, "mossImageEdit");
        put!(moss_image_advanced, "mossImageAdvanced");
        wire
    }

    pub fn bind(input: &BindingInput<'_>) -> Result<Self, BindingError> {
        Ok(Self {
            sampler: input.text("advanced", "sampler")?,
            scheduler: input.text("advanced", "scheduler")?,
            scheduler_shift: input.number("advanced", "schedulerShift")?,
            interleave_timestep_shift: input.number_exact_alias("advanced", "timestepShift")?,
            steps: input.integer("advanced", "steps")?,
            guidance_scale: input.number("advanced", "guidanceScale")?,
            guidance_method: input.text("advanced", "guidanceMethod")?,
            cfg_interval: input.structured("advanced", "cfgInterval")?,
            caption_overflow: input.text("advanced", "captionOverflow")?,
            num_inference_steps: input.integer("advanced", "numInferenceSteps")?,
            enhance_prompt: input.boolean("advanced", "enhancePrompt")?,
            enhance_max_tokens: input.integer("advanced", "enhanceMaxTokens")?,
            enhance_temperature: input.number("advanced", "enhanceTemperature")?,
            decoder: input.text("advanced", "decoder")?,
            use_pid: input.boolean("advanced", "usePid")?,
            pid_target: input.text("advanced", "pidTarget")?,
            pid_checkpoint: input.structured("advanced", "pidCheckpoint")?,
            pid_gemma: input.text("advanced", "pidGemma")?,
            ip_adapter_scale: input.number("advanced", "ipAdapterScale")?,
            ip_adapter_repo: input.text("advanced", "ipAdapterRepo")?,
            controlnet_conditioning_scale: input
                .number("advanced", "controlnetConditioningScale")?,
            true_cfg_scale: input.number("advanced", "trueCfgScale")?,
            strength: input.number("advanced", "strength")?,
            reference_strength: input.number("advanced", "referenceStrength")?,
            denoising_strength: input.number("advanced", "denoisingStrength")?,
            image_strength: input.number("advanced", "imageStrength")?,
            ip_adapter_scale_legacy: input.number("advanced", "ip_adapter_scale")?,
            text_style_gain: input.number("advanced", "textStyleGain")?,
            view_angle: input.text("advanced", "viewAngle")?,
            view_angles: input.structured("advanced", "viewAngles")?,
            poses: input.structured("advanced", "poses")?,
            face_restore: input.boolean("advanced", "faceRestore")?,
            control_mode: input.text("advanced", "controlMode")?,
            control_image: input.text("advanced", "controlImage")?,
            control_weights: input.structured("advanced", "controlWeights")?,
            control_scale: input.number("advanced", "controlScale")?,
            open_pose_scale: input.number("advanced", "openPoseScale")?,
            cn_scale: input.number("advanced", "cnScale")?,
            tile: input.integer("advanced", "tile")?,
            overlap: input.integer("advanced", "overlap")?,
            prompt: input.text("advanced", "prompt")?,
            negative_prompt: input.text("advanced", "negativePrompt")?,
            tile_control_net_repo: input.text("advanced", "tileControlNetRepo")?,
            id_weight: input.number("advanced", "idWeight")?,
            timestep_to_start_cfg: input.integer("advanced", "timestepToStartCfg")?,
            angle_set: input.boolean("advanced", "angleSet")?,
            keypoint_collection_id: input.text("advanced", "keypointCollectionId")?,
            system_message: input.text("advanced", "systemMessage")?,
            image_guidance_scale: input.number("advanced", "imageGuidanceScale")?,
            max_images: input.integer("advanced", "maxImages")?,
            max_new_tokens: input.integer("advanced", "maxNewTokens")?,
            think_mode: input.boolean("advanced", "thinkMode")?,
            phases: input.structured("advanced", "phases")?,
            flash_attn: input.boolean("advanced", "flashAttn")?,
            mlx_quantize: input.nullable_integer("advanced", "mlxQuantize")?,
            mlx_quantize_explicit: input.boolean("advanced", "mlxQuantizeExplicit")?,
            conv_rot: input.boolean("advanced", "convRot")?,
            quant_tier: input.text("advanced", "quantTier")?,
            quant: input.text("advanced", "quant")?,
            model_path: input.text("advanced", "modelPath")?,
            text_encoder_model: input.text("advanced", "textEncoderModel")?,
            transparent_background: input.boolean("advanced", "transparentBackground")?,
            flux3_layout: input.structured("advanced", "flux3_layout")?,
            flux3_reference_ids: input.structured("advanced", "flux3_reference_ids")?,
            moss_image: input.structured("advanced", "mossImage")?,
            moss_image_edit: input.structured("advanced", "mossImageEdit")?,
            moss_image_advanced: input.structured("advanced", "mossImageAdvanced")?,
        })
    }
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
    /// Canonical effective options for execution. `None` denotes a legacy authoring/parser
    /// request that has not crossed the shared resolver boundary.
    pub resolved_advanced: Option<ImageAdvanced>,
    /// Worker-verified local sources. The API never binds client-authored path claims.
    pub portable_aux: crate::portable_aux_bindings::PortableAuxBindings,
    /// Image Studio "Upscale" toggle (sc-8091): when `enabled`, the worker upscales each
    /// generated image with `engine` (`seedvr2` / `real-esrgan`) at `factor` and writes a
    /// second "(Nx upscaled)" asset — mirroring the Python worker. Disabled when omitted.
    pub upscale: ImageUpscaleRequest,
    /// Optional A1111-style two-pass high-resolution refinement. Disabled when omitted.
    pub hires_fix: HiresFixRequest,
}

impl ImageRequest {
    pub fn resolved_flux3_tier(&self) -> Result<Flux3Tier, String> {
        flux3_requested_tier_resolved(
            self.resolved_advanced
                .as_ref()
                .expect("image options must be resolved"),
        )
    }

    pub fn effective_advanced_wire(&self) -> JsonObject {
        self.resolved_advanced
            .as_ref()
            .expect("image options must be resolved")
            .provider_wire()
    }

    /// Access canonical scalar options only after `from_resolved`. A missing binding is a
    /// programming error at an execution boundary, not a reason to reinterpret authored JSON.
    pub fn advanced_bool(&self, key: &'static str) -> Option<bool> {
        let options = self
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved");
        let value = match key {
            "enhancePrompt" => &options.enhance_prompt,
            "usePid" => &options.use_pid,
            "faceRestore" => &options.face_restore,
            "angleSet" => &options.angle_set,
            "thinkMode" => &options.think_mode,
            "flashAttn" => &options.flash_attn,
            "mlxQuantizeExplicit" => &options.mlx_quantize_explicit,
            "convRot" => &options.conv_rot,
            "transparentBackground" => &options.transparent_background,
            _ => panic!("undeclared image boolean accessor: {key}"),
        };
        value.as_ref().map(|bound| bound.value)
    }

    pub fn advanced_i64(&self, key: &'static str) -> Option<i64> {
        let options = self
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved");
        if key == "mlxQuantize" {
            return options.mlx_quantize.as_ref().and_then(|bound| bound.value);
        }
        let value = match key {
            "steps" => &options.steps,
            "enhanceMaxTokens" => &options.enhance_max_tokens,
            "numInferenceSteps" => &options.num_inference_steps,
            "tile" => &options.tile,
            "overlap" => &options.overlap,
            "timestepToStartCfg" => &options.timestep_to_start_cfg,
            "maxImages" => &options.max_images,
            "maxNewTokens" => &options.max_new_tokens,
            _ => panic!("undeclared image integer accessor: {key}"),
        };
        value.as_ref().map(|bound| bound.value)
    }

    pub fn advanced_f64(&self, key: &'static str) -> Option<f64> {
        self.advanced_f64_bound(key).map(|bound| bound.value)
    }

    /// Use an authored override ahead of another option's materialized model default.
    pub fn authored_advanced_f64(&self, key: &'static str) -> Option<f64> {
        self.advanced_f64_bound(key)
            .filter(|bound| matches!(bound.origin, Origin::Authored { .. }))
            .map(|bound| bound.value)
    }

    fn advanced_f64_bound(&self, key: &'static str) -> Option<&Bound<f64>> {
        let options = self
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved");
        let value = match key {
            "schedulerShift" | "timestepShift" => &options.scheduler_shift,
            "guidanceScale" => &options.guidance_scale,
            "enhanceTemperature" => &options.enhance_temperature,
            "ipAdapterScale" => &options.ip_adapter_scale,
            "controlnetConditioningScale" => &options.controlnet_conditioning_scale,
            "trueCfgScale" => &options.true_cfg_scale,
            "strength" => &options.strength,
            "referenceStrength" => &options.reference_strength,
            "denoisingStrength" => &options.denoising_strength,
            "imageStrength" => &options.image_strength,
            "ip_adapter_scale" => &options.ip_adapter_scale_legacy,
            "textStyleGain" => &options.text_style_gain,
            "controlScale" => &options.control_scale,
            "openPoseScale" => &options.open_pose_scale,
            "cnScale" => &options.cn_scale,
            "idWeight" => &options.id_weight,
            "imageGuidanceScale" => &options.image_guidance_scale,
            _ => panic!("undeclared image number accessor: {key}"),
        };
        value.as_ref()
    }

    pub fn advanced_text(&self, key: &'static str) -> Option<&str> {
        let options = self
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved");
        let value = match key {
            "sampler" => &options.sampler,
            "scheduler" => &options.scheduler,
            "guidanceMethod" => &options.guidance_method,
            "captionOverflow" => &options.caption_overflow,
            "decoder" => &options.decoder,
            "pidTarget" => &options.pid_target,
            "pidGemma" => &options.pid_gemma,
            "ipAdapterRepo" => &options.ip_adapter_repo,
            "viewAngle" => &options.view_angle,
            "controlMode" => &options.control_mode,
            "controlImage" => &options.control_image,
            "prompt" => &options.prompt,
            "negativePrompt" => &options.negative_prompt,
            "tileControlNetRepo" => &options.tile_control_net_repo,
            "keypointCollectionId" => &options.keypoint_collection_id,
            "systemMessage" => &options.system_message,
            "quantTier" => &options.quant_tier,
            "quant" => &options.quant,
            "modelPath" => &options.model_path,
            "textEncoderModel" => &options.text_encoder_model,
            _ => panic!("undeclared image text accessor: {key}"),
        };
        value.as_ref().map(|bound| bound.value.as_str())
    }

    /// Bind a request after portable validation has produced `effective`. The legacy parser
    /// remains available to authoring and persistence callers; execution uses these typed fields.
    pub fn from_resolved(
        authored: &JsonObject,
        effective: &JsonObject,
    ) -> Result<Self, BindingError> {
        let mut request = Self::from_payload_internal(effective, false);
        request.resolved_advanced = Some(ImageAdvanced::bind(&BindingInput::new(
            authored, effective,
        ))?);
        if flux3::is_model(&request.model) {
            request.flux3_reference_request = Some(flux3::parse_resolved(
                effective,
                request.resolved_advanced.as_ref().expect("bound above"),
            ));
        }
        // Keep exactly what the caller authored for upgrade and persistence. Execution reads
        // only `resolved_advanced` and other typed fields from the effective request.
        request.advanced = object_or_empty(authored, "advanced");
        Ok(request)
    }

    /// Legacy authoring/persistence reader. Execution uses [`Self::resolved_quant_bits`].
    /// Explicit quantization bits (or the verified portable binding's frozen tier), before
    /// model-specific defaults or tier validation.
    pub fn quant_bits(&self) -> Option<i64> {
        crate::payload_util::request_quant_bits(&self.advanced)
    }

    /// Precision for native execution, including the verified portable tier only when the
    /// caller left `mlxQuantize` absent. Authored null deliberately suppresses that fallback.
    pub fn resolved_quant_bits(&self) -> Option<i64> {
        let options = self
            .resolved_advanced
            .as_ref()
            .expect("image options must be resolved");
        crate::payload_util::request_quant_bits_typed(
            options.mlx_quantize.as_ref(),
            self.portable_aux.bound_tier(),
        )
    }

    /// Probe one exact artifact tier without rewriting the authored request or its verified
    /// source binding. The probe is transient and cannot be persisted as user intent.
    pub fn with_probe_quant_bits(&self, bits: i64) -> Self {
        let mut probe = self.clone();
        probe
            .resolved_advanced
            .as_mut()
            .expect("image options must be resolved")
            .mlx_quantize = Some(Bound {
            value: Some(bits),
            origin: crate::resolved_options::Origin::Internal("artifact tier probe"),
        });
        probe
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
                self.resolved_advanced
                    .as_ref()
                    .and_then(|options| options.moss_image_advanced.as_ref())
                    .and_then(|bound| bound.value.positive.as_ref())
                    .and_then(|positive| positive.values.metadata.as_ref())
                    .and_then(|metadata| metadata.shape.first())
                    .copied()
                    .filter(|count| *count <= 8)
                    .unwrap_or(1)
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
        Self::from_payload_internal(payload, true)
    }

    fn from_payload_internal(payload: &JsonObject, parse_flux3: bool) -> Self {
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
            flux3_reference_request: (parse_flux3
                && flux3::is_model(&nonempty_string_or(payload, "model", DEFAULT_MODEL)))
            .then(|| flux3::parse(payload)),
            mask_asset_id: optional_id(payload, "maskAssetId"),
            fit_mode: normalize_fit_mode(payload.get("fitMode").and_then(Value::as_str)),
            model_manifest_entry,
            advanced: object_or_empty(payload, "advanced"),
            resolved_advanced: None,
            portable_aux: Default::default(),
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
    if request.hires_fix.is_disabled() {
        return None;
    }
    if request.mode != "text_to_image" {
        return Some("Hires.fix is only available for text-to-image generation.".to_owned());
    }
    if request.upscale.enabled {
        return Some(
            "Hires.fix and the post-generation Upscale option are mutually exclusive.".to_owned(),
        );
    }
    let options = request
        .resolved_advanced
        .as_ref()
        .expect("Hires.fix execution options must be resolved");
    let advertises_img2img = request
        .model_manifest_entry
        .get("ui")
        .and_then(|ui| ui.get("img2img"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if family != "sdxl" && !advertises_img2img {
        return Some(format!(
            "Hires.fix requires an img2img-capable model; '{}' does not advertise that capability.",
            request.model
        ));
    }
    let has_edit_or_control_input = request.source_asset_id.is_some()
        || request.reference_asset_id.is_some()
        || !request.reference_asset_ids.is_empty()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || options
            .poses
            .as_ref()
            .is_some_and(|poses| !poses.value.is_empty())
        || options
            .control_image
            .as_ref()
            .is_some_and(|id| !id.value.trim().is_empty());
    if has_edit_or_control_input {
        return Some(
            "Hires.fix cannot be combined with image-edit, reference, character, mask, or strict-control inputs."
                .to_owned(),
        );
    }
    if options.use_pid.as_ref().is_some_and(|value| value.value) {
        return Some("Hires.fix and PiD super-resolution are mutually exclusive.".to_owned());
    }
    if options
        .phases
        .as_ref()
        .is_some_and(|phases| !phases.value.is_empty())
    {
        return Some("Hires.fix cannot be combined with multi-phase sampling.".to_owned());
    }
    let acceleration_sampler = options
        .sampler
        .as_ref()
        .map(|sampler| sampler.value.trim())
        .map(str::to_ascii_lowercase)
        .is_some_and(|sampler| matches!(sampler.as_str(), "lightning" | "lcm" | "hyper"));
    if !model_supports_hires_fix(request) || acceleration_sampler {
        return Some(
            "Hires.fix is not supported by Lightning, LCM, or Hyper acceleration samplers because they do not accept the required img2img second pass."
                .to_owned(),
        );
    }
    let upscale_by = request.hires_fix.effective_upscale_by();
    let target_width = hires_fix_target_dimension(request.width, upscale_by);
    let target_height = hires_fix_target_dimension(request.height, upscale_by);
    if target_width > 4096 || target_height > 4096 {
        return Some(format!(
            "Hires.fix target {}x{} exceeds the 4096px image limit; lower the base resolution or Upscale by value.",
            target_width, target_height
        ));
    }
    None
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

pub use sceneworks_options::image_request::{
    ordered_image_reference_error, ordered_image_reference_ids_for_model,
};

pub use sceneworks_options::image_request::{
    flux3_requested_tier, requests_fp8r_tier, Flux3Tier, FLUX3_PRECISION_CARRIERS,
    FLUX3_TIERED_MODEL_IDS, FP8R_QUANT_TIER,
};

pub fn flux3_requested_tier_resolved(options: &ImageAdvanced) -> Result<Flux3Tier, String> {
    let named = options
        .quant_tier
        .as_ref()
        .map(|bound| bound.value.trim())
        .filter(|tier| !tier.is_empty())
        .map(|tier| {
            let key = tier.to_ascii_lowercase();
            match key.as_str() {
                "dense" => Ok(Flux3Tier::Bf16),
                other => Flux3Tier::from_key(other).ok_or_else(|| {
                    format!(
                        "advanced.quantTier {tier:?} is not a FLUX 3 Image Klein package; the \
                         packages are bf16, q4, q8, fp8r and nvfp4"
                    )
                }),
            }
        })
        .transpose()?;
    let bits = options
        .mlx_quantize
        .as_ref()
        .and_then(|bound| bound.value)
        .map(|bits| match bits {
            bits if bits <= 0 => Ok(Flux3Tier::Bf16),
            4 => Ok(Flux3Tier::Q4),
            8 => Ok(Flux3Tier::Q8),
            other => Err(format!(
                "advanced.mlxQuantize {other} is not a FLUX 3 Image Klein package; the bit \
                 tiers are 0 (bf16), 4 and 8"
            )),
        })
        .transpose()?;
    if options.conv_rot.as_ref().is_some_and(|bound| bound.value) {
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

pub use sceneworks_options::image_request::image_quant_tier_refusal;

pub fn image_quant_tier_refusal_resolved(model: &str, options: &ImageAdvanced) -> Option<String> {
    if FLUX3_TIERED_MODEL_IDS.contains(&model) {
        return flux3_requested_tier_resolved(options)
            .err()
            .map(|reason| format!("{model}: {reason}"));
    }
    options
        .quant_tier
        .as_ref()
        .is_some_and(|tier| tier.value.trim().eq_ignore_ascii_case(FP8R_QUANT_TIER))
        .then(|| {
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
    flux3_catalog_tier_refusal_for_tier(model, tier, manifest)
}

pub fn flux3_catalog_tier_refusal_resolved(
    model: &str,
    options: &ImageAdvanced,
    manifest: &Value,
) -> Option<String> {
    if !FLUX3_TIERED_MODEL_IDS.contains(&model) {
        return None;
    }
    let tier = match flux3_requested_tier_resolved(options) {
        Ok(tier) => tier,
        Err(reason) => return Some(format!("{model}: {reason}")),
    };
    flux3_catalog_tier_refusal_for_tier(model, tier, manifest)
}

fn flux3_catalog_tier_refusal_for_tier(
    model: &str,
    tier: Flux3Tier,
    manifest: &Value,
) -> Option<String> {
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
    fn resolved_image_options_preserve_effective_values_and_authored_origin() {
        let authored = payload(json!({
            "projectId": "p",
            "advanced": {"steps": "8", "timestepShift": "2.5", "enhancePrompt": true}
        }));
        let effective = payload(json!({
            "projectId": "p",
            "advanced": {"steps": 8, "schedulerShift": 2.5, "enhancePrompt": true, "decoder": "native"}
        }));
        let request = ImageRequest::from_resolved(&authored, &effective).unwrap();
        assert_eq!(request.advanced.get("steps"), Some(&json!("8")));
        assert_eq!(request.advanced_i64("steps"), Some(8));
        assert_eq!(request.advanced_f64("timestepShift"), Some(2.5));
        assert_eq!(request.advanced_bool("enhancePrompt"), Some(true));
        assert_eq!(request.advanced_text("decoder"), Some("native"));
        assert_eq!(
            request.effective_advanced_wire(),
            effective
                .get("advanced")
                .unwrap()
                .as_object()
                .unwrap()
                .clone()
        );
        let options = request.resolved_advanced.unwrap();
        assert_eq!(
            options.scheduler_shift.unwrap().origin,
            crate::resolved_options::Origin::Authored {
                spelling: "timestepShift"
            }
        );
        assert!(matches!(
            options.decoder.unwrap().origin,
            crate::resolved_options::Origin::Default(_)
        ));
    }

    #[test]
    fn resolved_image_options_reject_noncanonical_effective_numbers() {
        let payload = payload(json!({"advanced": {"steps": "8"}}));
        assert!(matches!(
            ImageRequest::from_resolved(&payload, &payload),
            Err(BindingError::InvalidEffectiveValue { .. })
        ));
    }

    #[test]
    fn resolved_image_options_preserve_explicit_null_quantization() {
        let authored = payload(json!({"advanced": {"mlxQuantize": null}}));
        let request = ImageRequest::from_resolved(&authored, &authored).unwrap();
        assert_eq!(request.advanced_i64("mlxQuantize"), None);
        assert_eq!(request.resolved_quant_bits(), None);
        assert_eq!(
            request.effective_advanced_wire().get("mlxQuantize"),
            Some(&Value::Null)
        );
        assert!(matches!(
            request
                .resolved_advanced
                .unwrap()
                .mlx_quantize
                .unwrap()
                .origin,
            crate::resolved_options::Origin::Authored {
                spelling: "mlxQuantize"
            }
        ));
        let absent = payload(json!({"advanced": {}}));
        assert!(ImageRequest::from_resolved(&absent, &absent)
            .unwrap()
            .resolved_advanced
            .unwrap()
            .mlx_quantize
            .is_none());
    }

    #[test]
    fn verified_bound_tier_fills_only_absent_quantization() {
        let trusted = payload(json!({"advanced": {
            "__portableFrozenSources": true,
            "__portableBoundTier": "q4"
        }}));
        let portable_aux =
            crate::portable_aux_bindings::PortableAuxBindings::from_trusted_dispatch(&trusted)
                .unwrap();
        let empty = payload(json!({"advanced": {}}));
        let mut request = ImageRequest::from_resolved(&empty, &empty).unwrap();
        request.portable_aux = portable_aux.clone();
        assert_eq!(request.resolved_quant_bits(), Some(4));

        let explicit = payload(json!({"advanced": {"mlxQuantize": 8}}));
        let mut request = ImageRequest::from_resolved(&explicit, &explicit).unwrap();
        request.portable_aux = portable_aux.clone();
        assert_eq!(request.resolved_quant_bits(), Some(8));

        let null = payload(json!({"advanced": {"mlxQuantize": null}}));
        let mut request = ImageRequest::from_resolved(&null, &null).unwrap();
        request.portable_aux = portable_aux;
        assert_eq!(request.resolved_quant_bits(), None);
    }

    #[test]
    fn artifact_probe_changes_only_typed_quantization() {
        let payload = payload(json!({"advanced": {"mlxQuantize": null}}));
        let request = ImageRequest::from_resolved(&payload, &payload).unwrap();
        let probe = request.with_probe_quant_bits(4);
        assert_eq!(probe.resolved_quant_bits(), Some(4));
        assert_eq!(request.resolved_quant_bits(), None);
        assert_eq!(probe.advanced, request.advanced);
        assert_eq!(
            probe.effective_advanced_wire().get("mlxQuantize"),
            Some(&json!(4))
        );
        assert_eq!(
            probe
                .resolved_advanced
                .unwrap()
                .mlx_quantize
                .unwrap()
                .origin,
            crate::resolved_options::Origin::Internal("artifact tier probe")
        );
    }

    #[test]
    fn typed_flux3_tier_preserves_named_bits_and_conflict() {
        for (advanced, expected) in [
            (json!({}), Ok(Flux3Tier::Bf16)),
            (json!({"quantTier":"Q4"}), Ok(Flux3Tier::Q4)),
            (json!({"mlxQuantize":8}), Ok(Flux3Tier::Q8)),
            (
                json!({"quantTier":"dense","mlxQuantize":0}),
                Ok(Flux3Tier::Bf16),
            ),
            (json!({"quantTier":"q4","mlxQuantize":8}), Err(())),
            (json!({"convRot":true}), Err(())),
        ] {
            let authored = payload(json!({"advanced": advanced}));
            let request = ImageRequest::from_resolved(&authored, &authored).unwrap();
            assert_eq!(request.resolved_flux3_tier().map_err(|_| ()), expected);
            assert_eq!(
                request.resolved_flux3_tier(),
                flux3_requested_tier(&request.advanced)
            );
        }
    }

    #[test]
    fn flux3_reference_request_uses_bound_advanced_fields() {
        let valid = payload(json!({
            "model": "flux3_klein", "mode": "edit_image",
            "referenceAssetId": "asset-a",
            "advanced": {"flux3_reference_ids": ["slot-a"]}
        }));
        let request = ImageRequest::from_resolved(&valid, &valid).unwrap();
        let references = request.flux3_reference_request.unwrap().unwrap();
        assert_eq!(references.asset_ids, vec!["asset-a"]);
        assert_eq!(references.reference_ids, Some(vec!["slot-a".to_owned()]));

        let unsupported = payload(json!({
            "model": "flux3_klein", "referenceAssetId": "asset-a",
            "advanced": {"strength": 1.0}
        }));
        let request = ImageRequest::from_resolved(&unsupported, &unsupported).unwrap();
        assert!(request
            .flux3_reference_request
            .unwrap()
            .unwrap_err()
            .contains("strength"));

        let authored =
            payload(json!({"model":"flux3_klein","referenceAssetId":"asset-a","advanced":{}}));
        let effective = payload(
            json!({"model":"flux3_klein","referenceAssetId":"asset-a","advanced":{"strength":0.5}}),
        );
        assert!(ImageRequest::from_resolved(&authored, &effective)
            .unwrap()
            .flux3_reference_request
            .unwrap()
            .is_ok());
    }

    #[test]
    fn pid_checkpoint_binds_named_fields() {
        let payload = payload(json!({"advanced": {"pidCheckpoint": {
            "repo": "SceneWorks/pid-flux", "filename": "student.safetensors"
        }}}));
        let request = ImageRequest::from_resolved(&payload, &payload).unwrap();
        let checkpoint = &request
            .resolved_advanced
            .unwrap()
            .pid_checkpoint
            .unwrap()
            .value;
        assert_eq!(checkpoint.repo.as_deref(), Some("SceneWorks/pid-flux"));
        assert_eq!(checkpoint.filename.as_deref(), Some("student.safetensors"));
    }

    #[test]
    fn control_weights_preserve_nested_shape_for_route_specific_checks() {
        let payload = payload(json!({"advanced": {"controlWeights": {
            "repo": 7, "filename": " ../escape ", "overlayId": " depth ",
            "_catalogAuthorized": "yes", "revision": null
        }}}));
        let request = ImageRequest::from_resolved(&payload, &payload).unwrap();
        let weights = &request
            .resolved_advanced
            .as_ref()
            .unwrap()
            .control_weights
            .as_ref()
            .unwrap()
            .value;
        assert!(weights.repo.is_invalid());
        assert_eq!(weights.filename.valid_trimmed(), Some("../escape"));
        assert_eq!(weights.overlay_id.valid_trimmed(), Some("depth"));
        assert!(weights.catalog_authorized.is_invalid());
        assert_eq!(weights.revision, ControlWeightField::Null);
        assert_eq!(
            request
                .effective_advanced_wire()
                .get("controlWeights")
                .unwrap(),
            &json!({"filename":" ../escape ","overlayId":" depth ","revision":null})
        );
    }

    #[test]
    fn moss_embedding_batch_count_uses_bound_metadata() {
        let request = payload(json!({
            "model": "moss_image_sft",
            "mode": "edit_image",
            "count": 2,
            "advanced": {"mossImageAdvanced": {"positive": {"values": {
                "assetId": "embedding-1", "sha256": "abc",
                "metadata": {"dtype": "f32", "shape": [3, 64], "byteLength": 768, "sha256": "abc"}
            }}}}
        }));
        let bound = ImageRequest::from_resolved(&request, &request).unwrap();
        assert_eq!(bound.image_count(), 6);
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
            let payload = payload(json!({
                "projectId": "p",
                "model": model,
                "mode": "text_to_image",
                "width": 1024,
                "height": 1024,
                "modelManifestEntry": entry,
                "hiresFix": { "enabled": true, "upscaleBy": 1.5 }
            }));
            ImageRequest::from_resolved(&payload, &payload).unwrap()
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

pub use sceneworks_options::image_request::{
    image_count_error, unsigned_or, validate_image_request_payload,
};

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
