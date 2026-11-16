//! Portable training policy. Native APIs and workers reexport this owner; installation,
//! provider selection, dataset files, authorization and output confinement stay native.
use crate::contracts::{ContractNumber, ExtraFields, JsonObject};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
pub mod flux3;
pub mod iris;
pub const MAX_REFERENCE_IMAGES_LIMIT: &str = "maxReferenceImages";
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $($variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[non_exhaustive]
        #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub enum $name {
            $($variant,)+
            Unknown(String),
        }


        impl $name {
            /// Wire values declared by this enum, excluding forward-compatible unknown values.
            pub const fn known_values() -> &'static [&'static str] {
                &[$($value,)+]
            }

            pub const fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $value,)+
                    Self::Unknown(value) => value.as_str(),
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Ok(match value.as_str() {
                    $($value => Self::$variant,)+
                    _ => Self::Unknown(value),
                })
            }
        }
    };
}

/// Learning-rate schedulers both worker kernels honor. Submit-time validation
/// rejects any `advanced.lrScheduler` outside this set so a non-constant choice
/// is never accepted as no-op metadata. This is the *learning-rate* scheduler;
/// the flow-matching noise scheduler is configured separately via
/// `advanced.timestepType`/`timestepBias`.
pub const SUPPORTED_LR_SCHEDULERS: [&str; 3] = ["constant", "linear", "cosine"];

/// `advanced` key of the relative-mode **weight noising** strength (epic 2123, sc-24826): after
/// every optimizer update the native trainer perturbs each adapter tensor by
/// `N(0,1) · sigma · rms(w)`. Absent or `0` is off.
pub const WEIGHT_NOISE_SIGMA_KEY: &str = "weightNoiseSigma";
/// Inclusive upper bound submit-time validation accepts for [`WEIGHT_NOISE_SIGMA_KEY`]. The web
/// form enforces the identical bound (`weightNoiseSigmaMax` in
/// `apps/web/src/training/trainingConfig.js`; a parity test pins the two together, epic 2123 E6).
/// The upstream suggested strength is [`WEIGHT_NOISE_SIGMA_SUGGESTED`]; 0.1 is eight times that —
/// far into "destroys the adapter" territory, so anything above it is a typo, not a choice.
pub const WEIGHT_NOISE_SIGMA_MAX: f64 = 0.1;
/// The upstream (ai-toolkit-perceptual) suggested weight-noise strength when enabled.
pub const WEIGHT_NOISE_SIGMA_SUGGESTED: f64 = 0.0125;
/// Target `limits` flag: `true` when this target's native trainer honors
/// [`WEIGHT_NOISE_SIGMA_KEY`] (its `TrainerDescriptor::techniques.weight_noise`). Absent = no. Since
/// sc-24827 every LoRA/LoKr trainer declares it on BOTH backends (MLX and Candle), so the builtin
/// value is the truth on every platform; the Krea control-branch target (a full-weight ControlNet
/// branch, no adapter) does not carry it. The web form shows the weight-noise toggle only where this
/// is `true`, and submit-time validation refuses a non-zero sigma elsewhere. A worker test pins the
/// flag to the linked trainer descriptors on each platform, so it cannot drift from the engine.
pub const WEIGHT_NOISE_SUPPORT_LIMIT: &str = "supportsWeightNoise";
/// Target `limits` flag for [`GRADIENT_NOISE_ETA_KEY`] (`TrainerDescriptor::techniques
/// .gradient_noise`, sc-24827) — same contract, gating, and drift test as
/// [`WEIGHT_NOISE_SUPPORT_LIMIT`].
pub const GRADIENT_NOISE_SUPPORT_LIMIT: &str = "supportsGradientNoise";

/// Epic 2123 JSON-null policy: an explicit JSON `null` on any technique key of `advanced` (weight
/// and gradient noise, resolution buckets, subject-masked loss, and the depth / body / face /
/// latent-perceptual losses) means exactly what an absent key means — off, or the knob's default.
/// Every technique parser (submit-time validation and the worker's preflight alike) reads its keys
/// through this one accessor, so the policy cannot differ between keys or between API and worker.
/// Resolve the MiniMax-H3-Image training base tier before a native training plan is built.
pub fn minimax_h3_image_base_tier(advanced: &JsonObject) -> Result<Option<&'static str>, String> {
    const H3_BASE_TIER_KEY: &str = "baseTier";
    const TIERS: [&str; 3] = ["bf16", "q8", "q4"];
    let Some(value) = advanced.get(H3_BASE_TIER_KEY) else {
        return Ok(None);
    };
    let Some(text) = value.as_str() else {
        return Err(format!(
            "{H3_BASE_TIER_KEY} must be a string (auto, bf16, q8 or q4)."
        ));
    };
    let text = text.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    TIERS
        .into_iter()
        .find(|tier| tier.eq_ignore_ascii_case(text))
        .map(Some)
        .ok_or_else(|| {
            format!("{H3_BASE_TIER_KEY} {text:?} is not a MiniMax-H3-Image tier (use auto, bf16, q8 or q4).")
        })
}

pub fn technique_value<'a>(advanced: &'a JsonObject, key: &str) -> Option<&'a Value> {
    advanced.get(key).filter(|value| !value.is_null())
}

/// The field-error message for an aux loss's inverted noise window (`min_t > max_t`). When
/// `max_key` is unset the message says so and names its default `max_t` — sc-2124 lowered the
/// decoded-x0 losses' default to `0.5`, so a carried-over `*MinT` above it is now refused with
/// the reason spelled out instead of only the numbers.
pub(crate) fn inverted_window_message(
    advanced: &JsonObject,
    (min_key, min_t): (&str, f64),
    (max_key, max_t): (&str, f64),
) -> String {
    if technique_value(advanced, max_key).is_some() {
        format!("{max_key} ({max_t}) must be at least {min_key} ({min_t}).")
    } else {
        format!(
            "{max_key} must be at least {min_key} ({min_t}): {max_key} is unset and its default \
             is now {max_t}. Set {max_key} explicitly or lower {min_key}."
        )
    }
}

/// Every epic 2123 technique key of `advanced` that [`technique_value`] governs.
const BASIC_TECHNIQUE_KEYS: &[&str] = &[
    WEIGHT_NOISE_SIGMA_KEY,
    GRADIENT_NOISE_ETA_KEY,
    GRADIENT_NOISE_GAMMA_KEY,
    RESOLUTION_BUCKETS_KEY,
    SUBJECT_MASK_LOSS_KEY,
    SUBJECT_MASK_BACKGROUND_WEIGHT_KEY,
    SUBJECT_MASK_SUBJECT_WEIGHT_KEY,
];
const TECHNIQUE_KEY_GROUPS: &[&[&str]] = &[
    BASIC_TECHNIQUE_KEYS,
    depth_anchoring::TECHNIQUE_KEYS,
    body_losses::TECHNIQUE_KEYS,
    face_losses::TECHNIQUE_KEYS,
    latent_perceptual::TECHNIQUE_KEYS,
];

const fn technique_key_count() -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < TECHNIQUE_KEY_GROUPS.len() {
        count += TECHNIQUE_KEY_GROUPS[i].len();
        i += 1;
    }
    count
}

const fn technique_keys() -> [&'static str; technique_key_count()] {
    let mut keys = [""; technique_key_count()];
    let mut index = 0;
    let mut group = 0;
    while group < TECHNIQUE_KEY_GROUPS.len() {
        let mut key = 0;
        while key < TECHNIQUE_KEY_GROUPS[group].len() {
            keys[index] = TECHNIQUE_KEY_GROUPS[group][key];
            index += 1;
            key += 1;
        }
        group += 1;
    }
    keys
}

/// Every technique key, derived from the owning modules in stable parser order.
pub const TECHNIQUE_KEYS: [&str; technique_key_count()] = technique_keys();

fn target_limit_flag(target: &TrainingTarget, flag: &str) -> bool {
    target.limits.get(flag).and_then(Value::as_bool) == Some(true)
}

/// Whether `target` advertises weight-noise support.
pub fn target_supports_weight_noise(target: &TrainingTarget) -> bool {
    target_limit_flag(target, WEIGHT_NOISE_SUPPORT_LIMIT)
}

/// Whether `target` advertises gradient-noise support.
pub fn target_supports_gradient_noise(target: &TrainingTarget) -> bool {
    target_limit_flag(target, GRADIENT_NOISE_SUPPORT_LIMIT)
}

/// Target `limits` flag: `true` when this target's native trainer on the serving platform honors
/// [`RESOLUTION_BUCKETS_KEY`] (its `TrainerDescriptor::techniques.resolution_buckets`, sc-2127).
/// Absent = no. The same mechanism as [`WEIGHT_NOISE_SUPPORT_LIMIT`]: the web form shows the
/// buckets toggle only where this is `true`, submit-time validation refuses a bucket list
/// elsewhere, and a worker test pins the flag to the linked trainer descriptors per platform.
pub const RESOLUTION_BUCKETS_SUPPORT_LIMIT: &str = "supportsResolutionBuckets";

/// Builtin targets whose trainer does NOT honor resolution buckets on any platform: LTX-2.5 trains
/// on prepared latent packs whose geometry is fixed by the bundle, so there is no spatial edge to
/// bucket; the Iris-3B trainers (generation, depth, restoration) declare no epic 2123 technique —
/// they bucket by their own `shape_policy` / `aspect_ratio_bucket` options instead. Every other
/// builtin target advertises [`RESOLUTION_BUCKETS_SUPPORT_LIMIT`].
#[doc(hidden)]
pub const RESOLUTION_BUCKETS_UNSUPPORTED_TARGETS: [&str; 6] = [
    "flux3_klein_lora",
    "flux3_klein_edit_lora",
    "ltx_2_5_video_lora",
    "iris_3b_generation",
    "iris_3b_depth",
    "iris_3b_restore",
];

/// Whether `target` advertises resolution-bucket support.
pub fn target_supports_resolution_buckets(target: &TrainingTarget) -> bool {
    target_limit_flag(target, RESOLUTION_BUCKETS_SUPPORT_LIMIT)
}

/// `advanced` key of the **multi-resolution buckets** list (epic 2123, sc-2127): an array of
/// `{ "resolution": <px>, "repeats": <n> }` rows. Each dataset image is trained at every listed
/// resolution, `repeats` times per epoch per row (ai-toolkit's `resolution` + `num_repeats` lists),
/// so rows 512/768/1024 with repeats 16/4/1 train on a 16:4:1 per-image mix. Absent is off (one
/// bucket at `resolution`, today's behaviour); present-but-empty is a field error.
pub const RESOLUTION_BUCKETS_KEY: &str = "resolutionBuckets";
/// Most bucket rows submit-time validation accepts — the engine's own cap
/// (`gen_core::MAX_RESOLUTION_BUCKETS`; the worker pins the two together). The web form enforces the
/// same bound (`resolutionBucketsMax` in `apps/web/src/training/trainingConfig.js`, parity-tested).
pub const RESOLUTION_BUCKETS_MAX: usize = 8;
/// Largest per-bucket repeat count submit-time validation accepts (web `resolutionBucketRepeatsMax`).
/// Repeats are a per-image mix ratio; 100:1 is already far past any useful skew.
pub const RESOLUTION_BUCKET_REPEATS_MAX: u64 = 100;
/// Every bucket resolution must be a multiple of this (web `resolutionBucketStride`): the native
/// trainers floor each training edge to a multiple of 32 so the latent grid tiles cleanly, so any
/// other value would silently train at a different size than the one requested.
pub const RESOLUTION_BUCKET_STRIDE: u64 = 32;

/// One validated `advanced.resolutionBuckets` row (see [`RESOLUTION_BUCKETS_KEY`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolutionBucketSpec {
    pub resolution: u32,
    pub repeats: u32,
}

/// A rectangular aspect-ratio bucket accepted alongside the established square form.
/// Row order remains meaningful: the native resolver preserves it for tie breaking and schedule
/// provenance, and portrait geometry is distinct from landscape geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectRatioBucketSpec {
    pub width: u32,
    pub height: u32,
    pub repeats: u32,
}

/// The product bucket wire contract. Legacy square rows retain their existing all-items-at-each-row
/// semantics; rectangular rows are an additive native contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionBucketConfig {
    Legacy(Vec<ResolutionBucketSpec>),
    AspectRatio(Vec<AspectRatioBucketSpec>),
}

fn bucket_repeats(row: &serde_json::Map<String, Value>, n: usize) -> Result<u32, String> {
    row.get("repeats")
        .and_then(Value::as_u64)
        .filter(|value| (1..=RESOLUTION_BUCKET_REPEATS_MAX).contains(value))
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!(
            "{RESOLUTION_BUCKETS_KEY} row {n}: repeats must be a whole number between 1 and {RESOLUTION_BUCKET_REPEATS_MAX}."
        ))
}

fn bucket_dimension(
    row: &serde_json::Map<String, Value>,
    key: &str,
    n: usize,
) -> Result<u32, String> {
    row.get(key)
        .and_then(Value::as_u64)
        .filter(|value| *value > 0 && *value % RESOLUTION_BUCKET_STRIDE == 0)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!(
            "{RESOLUTION_BUCKETS_KEY} row {n}: {key} must be a positive whole number of pixels divisible by {RESOLUTION_BUCKET_STRIDE}."
        ))
}

/// Parse either the legacy square `[{resolution,repeats}]` list or the additive rectangular
/// `[{width,height,repeats}]` list. Input may use exactly one form: mixed, incomplete, fractional,
/// duplicate-geometry and present-empty lists refuse rather than being normalized into an altered
/// training plan.
pub fn parse_resolution_bucket_config(
    advanced: &serde_json::Map<String, Value>,
) -> Result<Option<ResolutionBucketConfig>, String> {
    let Some(value) = technique_value(advanced, RESOLUTION_BUCKETS_KEY) else {
        return Ok(None);
    };
    let rows = value.as_array().ok_or_else(|| {
        format!("{RESOLUTION_BUCKETS_KEY} must be a list of square or rectangular bucket rows.")
    })?;
    if rows.is_empty() {
        return Err(format!(
            "{RESOLUTION_BUCKETS_KEY} must list at least one bucket (omit it to disable bucketing)."
        ));
    }
    if rows.len() > RESOLUTION_BUCKETS_MAX {
        return Err(format!(
            "{RESOLUTION_BUCKETS_KEY} has {} buckets; at most {RESOLUTION_BUCKETS_MAX} are allowed.",
            rows.len()
        ));
    }

    let mut legacy = Vec::with_capacity(rows.len());
    let mut aspect = Vec::with_capacity(rows.len());
    let mut form: Option<&str> = None;
    for (index, value) in rows.iter().enumerate() {
        let n = index + 1;
        let row = value
            .as_object()
            .ok_or_else(|| format!("{RESOLUTION_BUCKETS_KEY} row {n} must be an object."))?;
        let has_resolution = row.contains_key("resolution");
        let has_width = row.contains_key("width");
        let has_height = row.contains_key("height");
        let row_form = match (has_resolution, has_width, has_height) {
            (true, false, false) => "legacy",
            (false, true, true) => "aspect",
            _ => return Err(format!(
                "{RESOLUTION_BUCKETS_KEY} row {n} must use exactly {{resolution, repeats}} or {{width, height, repeats}}."
            )),
        };
        if form.is_some_and(|previous| previous != row_form) {
            return Err(format!(
                "{RESOLUTION_BUCKETS_KEY} must not mix square and rectangular bucket rows."
            ));
        }
        form = Some(row_form);
        let repeats = bucket_repeats(row, n)?;
        match row_form {
            "legacy" => {
                if row.len() != 2 || !row.contains_key("repeats") {
                    return Err(format!(
                        "{RESOLUTION_BUCKETS_KEY} row {n} must contain only resolution and repeats."
                    ));
                }
                let resolution = bucket_dimension(row, "resolution", n)?;
                if legacy
                    .iter()
                    .any(|bucket: &ResolutionBucketSpec| bucket.resolution == resolution)
                {
                    return Err(format!(
                        "{RESOLUTION_BUCKETS_KEY} lists resolution {resolution} more than once."
                    ));
                }
                legacy.push(ResolutionBucketSpec {
                    resolution,
                    repeats,
                });
            }
            "aspect" => {
                if row.len() != 3 || !row.contains_key("repeats") {
                    return Err(format!(
                        "{RESOLUTION_BUCKETS_KEY} row {n} must contain only width, height and repeats."
                    ));
                }
                let width = bucket_dimension(row, "width", n)?;
                let height = bucket_dimension(row, "height", n)?;
                if aspect.iter().any(|bucket: &AspectRatioBucketSpec| {
                    bucket.width == width && bucket.height == height
                }) {
                    return Err(format!(
                        "{RESOLUTION_BUCKETS_KEY} lists geometry {width}x{height} more than once."
                    ));
                }
                aspect.push(AspectRatioBucketSpec {
                    width,
                    height,
                    repeats,
                });
            }
            _ => unreachable!("row form is classified above"),
        }
    }
    Ok(Some(match form {
        Some("legacy") => ResolutionBucketConfig::Legacy(legacy),
        Some("aspect") => ResolutionBucketConfig::AspectRatio(aspect),
        _ => unreachable!("non-empty rows have a form"),
    }))
}

/// Backwards-compatible reader for native square-bucket consumers. New rectangular consumers use
/// [`parse_resolution_bucket_config`] so they receive width and height without lossy projection.
pub fn parse_resolution_buckets(
    advanced: &serde_json::Map<String, Value>,
) -> Result<Option<Vec<ResolutionBucketSpec>>, String> {
    match parse_resolution_bucket_config(advanced)? {
        None => Ok(None),
        Some(ResolutionBucketConfig::Legacy(rows)) => Ok(Some(rows)),
        Some(ResolutionBucketConfig::AspectRatio(_)) => Err(format!(
            "{RESOLUTION_BUCKETS_KEY} rectangular rows require a native aspect-ratio bucket trainer."
        )),
    }
}

/// The largest training resolution a config trains at: the largest
/// [`RESOLUTION_BUCKETS_KEY`] row when buckets are set (they replace `resolution`), else
/// `resolution`. Memory admission sizes for this (epic 2123 E7). A malformed bucket list falls back
/// to `resolution` here only because plan validation refuses it before any job is queued.
pub fn training_max_resolution(config: &TrainingConfig) -> u32 {
    match parse_training_bucket_config(&config.advanced) {
        Ok(Some(ResolutionBucketConfig::Legacy(buckets))) => buckets
            .iter()
            .map(|bucket| bucket.resolution)
            .max()
            .unwrap_or(config.resolution),
        Ok(Some(ResolutionBucketConfig::AspectRatio(buckets))) => buckets
            .iter()
            .flat_map(|bucket| [bucket.width, bucket.height])
            .max()
            .unwrap_or(config.resolution),
        _ => config.resolution,
    }
}

/// `advanced` key of the annealed **gradient noise** initial scale `eta` (epic 2123, sc-24827):
/// on every optimizer update `t` the native trainer adds `N(0,1) · eta / (1 + t)^gamma` to each
/// adapter gradient (after the norm clip, before the step). Absent or `0` is off.
pub const GRADIENT_NOISE_ETA_KEY: &str = "gradientNoiseEta";
/// `advanced` key of the gradient-noise annealing exponent `gamma` (absent ⇒
/// [`GRADIENT_NOISE_GAMMA_DEFAULT`]; only meaningful while eta is on).
pub const GRADIENT_NOISE_GAMMA_KEY: &str = "gradientNoiseGamma";
/// Inclusive upper bound for [`GRADIENT_NOISE_ETA_KEY`]. The web form enforces the identical bound
/// (`gradientNoiseEtaMax` in `apps/web/src/training/trainingConfig.js`; a parity test pins them,
/// epic 2123 E6). Ten times the upstream suggestion [`GRADIENT_NOISE_ETA_SUGGESTED`]: the noise is
/// per element, so even 0.1 already dwarfs a unit-norm-clipped gradient on any real adapter.
pub const GRADIENT_NOISE_ETA_MAX: f64 = 0.1;
/// The upstream (ai-toolkit-perceptual `neelakantan` mode) suggested eta when enabled.
pub const GRADIENT_NOISE_ETA_SUGGESTED: f64 = 0.01;
/// Inclusive upper bound for [`GRADIENT_NOISE_GAMMA_KEY`] (web parity: `gradientNoiseGammaMax`).
/// The annealed-noise analysis (Neelakantan et al. 2015, after Welling & Teh) needs `gamma <= 1`;
/// larger exponents switch the noise off within a handful of updates.
pub const GRADIENT_NOISE_GAMMA_MAX: f64 = 1.0;
/// Default annealing exponent (the paper's and upstream's 0.55).
pub const GRADIENT_NOISE_GAMMA_DEFAULT: f64 = 0.55;

/// Body losses (epic 2123, sc-24832): ViTPose proportion, HybrIK shape and Sapiens normal losses —
/// keys, bounds, the shared strict parser, support flags and auxiliary catalog models.
pub mod body_losses;

/// Depth anchoring (epic 2123, sc-2125): keys, bounds, the shared strict parser and the auxiliary
/// catalog models.
pub mod depth_anchoring;

/// Face losses (epic 2123, sc-24831): the ArcFace identity loss and the FaceMesh landmark loss —
/// keys, bounds, the shared strict parsers, support flags and auxiliary catalog models.
pub mod face_losses;

/// The latent-space perceptual losses (epic 2123, sc-24833): VAE anchor + E-LatentLPIPS keys,
/// bounds, the shared strict parser, support flags and auxiliary catalog models.
pub mod latent_perceptual;

/// The de-distill training adapter (sc-25213): keys, the strict version→catalog-model parser, the
/// support flag and the cataloged adapter files.
pub mod training_adapter;

/// Target `limits` flag: `true` when this target's native trainer honors subject-masked loss
/// ([`SUBJECT_MASK_LOSS_KEY`], its `TrainerDescriptor::techniques.subject_mask_loss`). Absent = no.
/// Every LoRA trainer declares it on both MLX and Candle except LTX-2.5 (prepared latent bundles)
/// and the Krea ControlNet branch, so the flag is static per target and needs no Candle
/// projection; a worker test pins it to the linked trainer descriptors on each platform. The web
/// form shows the toggle only where this is `true`, and submit-time validation refuses mask loss
/// elsewhere with a `subjectMaskLoss` field error.
pub const SUBJECT_MASK_LOSS_SUPPORT_LIMIT: &str = "supportsSubjectMaskLoss";

/// Whether `target` advertises subject-masked-loss support.
pub fn target_supports_subject_mask_loss(target: &TrainingTarget) -> bool {
    target_limit_flag(target, SUBJECT_MASK_LOSS_SUPPORT_LIMIT)
}

/// `advanced` key that turns on **subject-masked loss weighting** (epic 2123, sc-24828): each
/// image's subject mask (`<dataset>/masks/<content hash>.png`, see
/// the native subject-mask store) weights the native trainer's per-element loss. A boolean;
/// absent or `false` is off.
pub const SUBJECT_MASK_LOSS_KEY: &str = "subjectMaskLoss";
/// `advanced` key of the loss weight of a pure-background latent cell, in
/// `SUBJECT_MASK_BACKGROUND_WEIGHT_RANGE` (absent ⇒ [`SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT`]).
pub const SUBJECT_MASK_BACKGROUND_WEIGHT_KEY: &str = "subjectMaskBackgroundWeight";
/// `advanced` key of the loss weight of a pure-subject latent cell, `> 0` and at most
/// [`SUBJECT_MASK_WEIGHT_MAX`] (absent ⇒ [`SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT`]).
pub const SUBJECT_MASK_SUBJECT_WEIGHT_KEY: &str = "subjectMaskSubjectWeight";
/// Upper bound of both mask weights — the engine's `SubjectMaskLoss` bounds. The web form uses the
/// identical bound (`subjectMaskWeightMax` in `apps/web/src/training/trainingConfig.js`; a parity
/// test pins the two together, epic 2123 E6). The background weight may be `0` (drop the
/// background); the subject weight must be `> 0`.
pub const SUBJECT_MASK_WEIGHT_MAX: f64 = 1.0;
/// Default background weight when masked loss is on: keep a little background signal so the
/// adapter does not learn arbitrary backgrounds, while the subject dominates.
pub const SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT: f64 = 0.1;
/// Default subject weight when masked loss is on.
pub const SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT: f64 = 1.0;

string_enum! {
    /// Output modality of a training target. `Image` is the first production
    /// target; `Video` and `Audio` are reserved so the contract stays generic.
    pub enum TrainingModality {
        Image => "image",
        Video => "video",
        Audio => "audio",
    }
}

string_enum! {
    /// What a training run produces. `Lora` is a LoRA/LoKr adapter; `ControlBranch` is a
    /// ControlNet control branch (an "overlay" the engine writes, distinct from a LoRA — it
    /// injects a conditioning residual rather than reparameterizing weights). ControlNet
    /// training (epic 10159, the `krea_control` kernel) produces the latter. `BaseCheckpoint` is a
    /// FULL base fine-tune (sc-14056/sc-15036): every backbone weight retrained, emitted as a
    /// diffusers-shaped transformer component directory (~8 GB) rather than an adapter file — so it
    /// belongs in the **model catalog**, not the LoRA registry.
    ///
    /// Unlike the other two, `BaseCheckpoint` is a per-**job** property, not a per-target one: the
    /// Mage target advertises `networkTypes: ["lora", "lokr", "full"]`, so the very same target
    /// produces an adapter or a base checkpoint depending on `advanced.networkType`. See
    /// [`training_output_kind`] — the single discriminator both the plan builder and the API use.
    pub enum TrainingOutputKind {
        Lora => "lora",
        ControlBranch => "control_branch",
        BaseCheckpoint => "base_checkpoint",
    }
}

/// The `advanced.networkType` value that selects a FULL base fine-tune (sc-14056). Every other
/// value (`lora`, `lokr`, absent, unrecognized) is an adapter run.
pub const NETWORK_TYPE_FULL: &str = "full";

/// Whether a training config's `advanced.networkType` selects a FULL base fine-tune.
///
/// The ONE discriminator, so the plan's declared `outputKind`, the API's output-location fork, and
/// the worker's `full_finetune` engine flag can never disagree about what a run produces. Trimmed
/// and case-insensitive, matching the worker's `map_training_config`.
pub fn config_is_full_finetune(config: &TrainingConfig) -> bool {
    config
        .advanced
        .get("networkType")
        .and_then(Value::as_str)
        .is_some_and(|value| value.trim().eq_ignore_ascii_case(NETWORK_TYPE_FULL))
}

/// What a run against `target` with `config` actually produces.
///
/// A target declares the kind it produces by default, but a target that offers `"full"` among its
/// `limits.networkTypes` produces a [`TrainingOutputKind::BaseCheckpoint`] for a full run —
/// derived here rather than copied off the target, so there is one discriminator and not two. A
/// control-branch target is never a full fine-tune (no control kernel offers `"full"`), so its kind
/// is returned unchanged.
pub fn training_output_kind(
    target_kind: &TrainingOutputKind,
    config: &TrainingConfig,
) -> TrainingOutputKind {
    if matches!(target_kind, TrainingOutputKind::Lora) && config_is_full_finetune(config) {
        return TrainingOutputKind::BaseCheckpoint;
    }
    target_kind.clone()
}

/// A registered training target: the combination of a base model, output kind,
/// and execution kernel, plus the defaults and bounds the UI builds on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingTarget {
    /// Stable target id, e.g. `z_image_turbo_lora`.
    pub id: String,
    pub name: String,
    pub modality: TrainingModality,
    pub output_kind: TrainingOutputKind,
    /// SceneWorks LoRA/model family, e.g. `z-image`. Drives downstream
    /// generation-side compatibility of the produced LoRA.
    pub family: String,
    /// Manifest model id this target trains against.
    pub base_model: String,
    /// Optional source repository for the base model weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_repo: Option<String>,
    /// Identifier of the native execution kernel that runs this target.
    pub kernel: String,
    /// Visible (simple-panel) config defaults for this target.
    pub defaults: TrainingConfig,
    /// Bounds and choices for advanced fields; free-form to stay generic.
    pub limits: JsonObject,
    /// Presentation hints (labels, descriptions); free-form.
    pub ui: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingConfig {
    /// LoRA network rank (dimension).
    pub rank: u32,
    /// LoRA alpha scaling factor. This stays a JSON number so MOSS can use a
    /// finite positive fractional scale while legacy targets retain their own
    /// explicitly advertised integral policies.
    pub alpha: ContractNumber,
    pub learning_rate: ContractNumber,
    /// Total training steps.
    pub steps: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    /// Square training resolution edge in pixels (e.g. `1024`). Aspect-ratio
    /// bucketing details, when used, live in `advanced`.
    pub resolution: u32,
    /// Checkpoint cadence, in steps.
    pub save_every: u32,
    /// MOSS uses a literal nonnegative seed, including zero. Its Studio Randomize
    /// action persists a concrete value; there is no negative/None Auto sentinel.
    pub seed: i64,
    /// Optimizer name, kept a free string to stay engine-agnostic.
    pub optimizer: String,
    /// Trigger word baked into captions and surfaced on the output LoRA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_word: Option<String>,
    /// Advanced, collapsed-by-default fields. Free-form by design. Preview-sample
    /// knobs live here: `sampleEvery` (cadence in steps; 0 disables), `sampleSteps`,
    /// `sampleGuidanceScale`, `sampleCount` (images per step; default 4), and
    /// `samplePrompts` (string array; the UI prefills trigger-derived defaults).
    /// All native backends cap the prompt pool at `sampleCount`
    /// (one preview per prompt, truncated — never padded).
    pub advanced: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Error produced when a `LoraTrainingRequest` cannot be resolved into a valid
/// `TrainingPlan`. These map to client errors: the request is structurally
/// fine but the dataset or config cannot produce a runnable plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingPlanError {
    /// The dataset has no items to train on.
    EmptyDataset,
    /// A hyperparameter is out of range; carries a human-facing reason.
    InvalidConfig(String),
    /// A value falls outside the target capability that was advertised to the
    /// client. Kept structured so the API can return a field-specific error
    /// without scraping a human-facing sentence.
    TargetLimit(TrainingTargetLimitError),
    /// A config field holds an invalid value (wrong type, out of range, or an incompatible
    /// combination). Names the request `field` so the API returns a field-level error.
    InvalidField { field: String, message: String },
    /// The dataset's shape does not fit the target — an instruction-edit dataset (items carrying
    /// ordered reference images, sc-24161) on a target that cannot train edits, a mixed or
    /// over-cap edit dataset, or a plain dataset on an edit-only target. Human-facing reason.
    InvalidDataset(String),
}

/// A target-advertised limit rejected while normalizing a training request.
///
/// These errors deliberately name the request field and the advertised bound.
/// A target catalog is a capability contract, not presentation-only metadata:
/// an unknown numeric entry fails closed rather than becoming an unenforced UI
/// hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingTargetLimitError {
    BelowMinimum {
        field: String,
        value: u64,
        minimum: u64,
    },
    AboveMaximum {
        field: String,
        value: u64,
        maximum: u64,
    },
    UnsupportedValue {
        field: String,
        value: String,
        allowed: Vec<String>,
    },
    UnsupportedNumericValue {
        field: String,
        value: u64,
        allowed: Vec<u64>,
    },
    InvalidAdvertisedLimit {
        field: String,
        detail: String,
    },
    UnsupportedNumericLimit {
        field: String,
    },
    /// A micro-step quantity that must be a whole number of gradient-accumulation windows
    /// (Iris: checkpoints, previews and the warmup land on optimizer-step boundaries).
    NotMultipleOf {
        field: String,
        value: u64,
        multiple: u64,
    },
}

impl std::fmt::Display for TrainingTargetLimitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BelowMinimum {
                field,
                value,
                minimum,
            } => write!(formatter, "{field} ({value}) must be at least {minimum}."),
            Self::AboveMaximum {
                field,
                value,
                maximum,
            } => write!(formatter, "{field} ({value}) must be at most {maximum}."),
            Self::UnsupportedValue {
                field,
                value,
                allowed,
            } => write!(
                formatter,
                "Unsupported {field} '{value}'. Allowed values: {}.",
                allowed.join(", ")
            ),
            Self::UnsupportedNumericValue {
                field,
                value,
                allowed,
            } => write!(
                formatter,
                "Unsupported {field} {value}. Allowed values: {}.",
                allowed
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::InvalidAdvertisedLimit { field, detail } => {
                write!(
                    formatter,
                    "Training target limit '{field}' is invalid: {detail}."
                )
            }
            Self::UnsupportedNumericLimit { field } => write!(
                formatter,
                "Training target advertises unsupported numeric limit '{field}'."
            ),
            Self::NotMultipleOf {
                field,
                value,
                multiple,
            } => write!(
                formatter,
                "{field} ({value}) is counted in micro-steps and must be a multiple of the \
                 gradient accumulation ({multiple})."
            ),
        }
    }
}

impl std::fmt::Display for TrainingPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDataset => {
                formatter.write_str("Training dataset has no items. Add at least one image.")
            }
            Self::InvalidConfig(detail) | Self::InvalidDataset(detail) => {
                formatter.write_str(detail)
            }
            Self::TargetLimit(error) => error.fmt(formatter),
            Self::InvalidField { message, .. } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for TrainingPlanError {}

pub fn validate_training_config_for_target(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    // Legacy task plans may omit networkType. Resolve the same objective default used by
    // execution before support checks inspect the config; keep the authored object intact.
    let mut effective_config;
    let config = if iris::is_iris_training_kernel(&target.kernel) {
        effective_config = config.clone();
        let network = network_type_for_model(&config.advanced, &target.kernel, &target.base_model)
            .map_err(|error| TrainingPlanError::InvalidConfig(error.detail))?;
        effective_config
            .advanced
            .insert("networkType".into(), json!(network));
        &effective_config
    } else {
        config
    };
    if flux3::KERNELS.contains(&target.kernel.as_str()) {
        flux3::validate(target, config)?;
    }
    validate_advertised_numeric_limits(target, config)?;
    validate_advertised_optimizer_limit(target, config)?;
    validate_training_config(config)?;
    if matches!(
        target.kernel.as_str(),
        "minimax_h3_image_lora" | "minimax_h3_image_edit_lora"
    ) {
        minimax_h3_image_base_tier(&config.advanced).map_err(|message| {
            TrainingPlanError::InvalidField {
                field: "baseTier".to_owned(),
                message,
            }
        })?;
    } else if config.advanced.contains_key("baseTier") {
        return Err(TrainingPlanError::InvalidField {
            field: "baseTier".to_owned(),
            message: "baseTier is supported only by MiniMax-H3-Image training targets.".to_owned(),
        });
    }
    if target.kernel == "moss_image_lora" {
        moss_training_seed(config.seed)?;
        validate_moss_training_field_shapes(&config.advanced)?;
    }
    validate_advertised_advanced_choices(target, config)?;
    if iris::is_iris_training_kernel(&target.kernel) {
        iris::validate_iris_training_config(target, config)?;
    }
    validate_resolution_buckets_for_target(target, config)?;
    validate_technique_support(target, config)?;
    depth_anchoring::validate_support(target, config)?;
    body_losses::validate_support(target, config)?;
    face_losses::validate_support(target, config)?;
    latent_perceptual::validate_support(target, config)?;
    training_adapter::validate_support(target, config)?;
    validate_subject_mask_loss_support(target, config)?;
    let network_type = match config.advanced.get("networkType") {
        None => "lora",
        Some(Value::String(value)) if value.trim().is_empty() => "lora",
        Some(Value::String(value)) => value.trim(),
        Some(_) => {
            return Err(TrainingPlanError::InvalidConfig(
                "networkType must be a string.".to_owned(),
            ));
        }
    };
    if let Some(supported) = target.limits.get("networkTypes").and_then(Value::as_array) {
        let admitted = supported.iter().any(|value| {
            value
                .as_str()
                .is_some_and(|value| value.eq_ignore_ascii_case(network_type))
        });
        if !admitted {
            let supported = supported
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(TrainingPlanError::InvalidConfig(format!(
                "Training target '{}' does not support networkType '{network_type}'. Supported network types: {supported}.",
                target.id
            )));
        }
    }
    if config_is_full_finetune(config) {
        // Platform-effective API catalogs may publish stricter full-tune execution requirements.
        // The core builtins retain MLX defaults; only the cloned Candle target carries this marker.
        let requirements = target
            .defaults
            .advanced
            .get("fullFinetuneConfig")
            .and_then(Value::as_object);
        if let Some(required_precision) = requirements
            .and_then(|values| values.get("mixedPrecision"))
            .and_then(Value::as_str)
        {
            let precision = config
                .advanced
                .get("mixedPrecision")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !precision.eq_ignore_ascii_case(required_precision) {
                return Err(TrainingPlanError::InvalidConfig(format!(
                    "{} full base fine-tuning requires mixedPrecision={required_precision}.",
                    target.name
                )));
            }
        }
        if requirements
            .and_then(|values| values.get("gradientCheckpointing"))
            .and_then(Value::as_bool)
            == Some(false)
            && config
                .advanced
                .get("gradientCheckpointing")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        {
            return Err(TrainingPlanError::InvalidConfig(format!(
                "{} full base fine-tuning does not support gradient checkpointing.",
                target.name
            )));
        }
    }
    validate_config_shape(
        &serde_json::to_value(config)
            .expect("training config serializes")
            .as_object()
            .expect("config object")
            .clone(),
    )
    .map_err(|error| TrainingPlanError::InvalidField {
        field: error.field,
        message: error.detail,
    })?;
    Ok(())
}

/// Enforce every numeric capability currently advertised in a target's `limits`
/// bag. Adding a new numeric limit without adding its request-field mapping is
/// intentionally rejected: otherwise a provider could publish a bound that the
/// shared request boundary silently ignores.
fn validate_advertised_numeric_limits(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    for (field, advertised) in &target.limits {
        if !contains_json_number(advertised) {
            continue;
        }
        // A dataset-shape limit, not a request field: the edit-pair reference cap (sc-24161) is
        // enforced against the dataset by `validate_dataset_shape_for_target`. It must still be a
        // non-negative integer, so a malformed advertisement is refused rather than read as 0.
        if field == MAX_REFERENCE_IMAGES_LIMIT {
            if advertised
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .is_none()
            {
                return Err(invalid_advertised_limit(
                    field,
                    "the reference-image cap must be a non-negative integer",
                ));
            }
            continue;
        }
        let values = advertised.as_array().ok_or_else(|| {
            invalid_advertised_limit(field, "numeric limits must be an array of integer bounds")
        })?;

        match field.as_str() {
            "rank" if !(target.kernel == "mage_flow_lora" && config_is_full_finetune(config)) => {
                validate_advertised_numeric_range(field, values, u64::from(config.rank))?
            }
            "rank" => {}
            // Legacy target limits intentionally describe integral alpha. MOSS has no
            // `alpha` numeric range so its finite fractional alpha passes through its
            // native f32 contract; another target must explicitly opt into fractions.
            "alpha" => validate_advertised_numeric_range(
                field,
                values,
                config.alpha.as_u64().ok_or_else(|| {
                    TrainingPlanError::InvalidConfig(
                        "alpha must be a whole number for this training target.".to_owned(),
                    )
                })?,
            )?,
            "steps" => validate_advertised_numeric_range(field, values, u64::from(config.steps))?,
            "batchSize" => {
                validate_advertised_numeric_range(field, values, u64::from(config.batch_size))?
            }
            "resolutions" => validate_advertised_numeric_choices(
                field,
                "resolution",
                values,
                u64::from(config.resolution),
            )?,
            // Preview images per cadence (`advanced.sampleCount`, default 4 — the worker's
            // `DEFAULT_SAMPLE_COUNT`). A trainer that renders fewer advertises its cap here.
            "sampleCount" => validate_advertised_numeric_range(
                field,
                values,
                match config.advanced.get("sampleCount") {
                    None | Some(Value::Null) => 4,
                    Some(value) => value.as_u64().ok_or_else(|| {
                        TrainingPlanError::InvalidConfig(
                            "sampleCount must be a non-negative integer.".to_owned(),
                        )
                    })?,
                },
            )?,
            _ => {
                return Err(TrainingPlanError::TargetLimit(
                    TrainingTargetLimitError::UnsupportedNumericLimit {
                        field: field.clone(),
                    },
                ));
            }
        }
    }
    Ok(())
}

fn contains_json_number(value: &Value) -> bool {
    match value {
        Value::Number(_) => true,
        Value::Array(values) => values.iter().any(contains_json_number),
        Value::Object(values) => values.values().any(contains_json_number),
        Value::Null | Value::Bool(_) | Value::String(_) => false,
    }
}

fn validate_advertised_numeric_range(
    field: &str,
    values: &[Value],
    value: u64,
) -> Result<(), TrainingPlanError> {
    let bounds = numeric_limit_values(field, values)?;
    let [minimum, maximum] = bounds.as_slice() else {
        return Err(invalid_advertised_limit(
            field,
            "a numeric range must contain exactly two integer bounds",
        ));
    };
    if minimum > maximum {
        return Err(invalid_advertised_limit(
            field,
            "the minimum must not exceed the maximum",
        ));
    }
    if value < *minimum {
        return Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::BelowMinimum {
                field: field.to_owned(),
                value,
                minimum: *minimum,
            },
        ));
    }
    if value > *maximum {
        return Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::AboveMaximum {
                field: field.to_owned(),
                value,
                maximum: *maximum,
            },
        ));
    }
    Ok(())
}

fn validate_advertised_numeric_choices(
    advertised_field: &str,
    request_field: &str,
    values: &[Value],
    value: u64,
) -> Result<(), TrainingPlanError> {
    let allowed = numeric_limit_values(advertised_field, values)?;
    if allowed.is_empty() {
        return Err(invalid_advertised_limit(
            advertised_field,
            "a numeric choice list must not be empty",
        ));
    }
    if !allowed.contains(&value) {
        return Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedNumericValue {
                field: request_field.to_owned(),
                value,
                allowed,
            },
        ));
    }
    Ok(())
}

fn numeric_limit_values(field: &str, values: &[Value]) -> Result<Vec<u64>, TrainingPlanError> {
    values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                invalid_advertised_limit(field, "numeric bounds must be non-negative integers")
            })
        })
        .collect()
}

fn invalid_advertised_limit(field: &str, detail: &str) -> TrainingPlanError {
    TrainingPlanError::TargetLimit(TrainingTargetLimitError::InvalidAdvertisedLimit {
        field: field.to_owned(),
        detail: detail.to_owned(),
    })
}

fn validate_advertised_optimizer_limit(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    let Some(values) = target.limits.get("optimizers") else {
        return Ok(());
    };
    let values = values.as_array().ok_or_else(|| {
        invalid_advertised_limit("optimizers", "an optimizer choice list must be an array")
    })?;
    let allowed = values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                invalid_advertised_limit("optimizers", "optimizer choices must be strings")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if allowed.is_empty() {
        return Err(invalid_advertised_limit(
            "optimizers",
            "an optimizer choice list must not be empty",
        ));
    }
    if !allowed.iter().any(|value| value == &config.optimizer) {
        return Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedValue {
                field: "optimizer".to_owned(),
                value: config.optimizer.clone(),
                allowed,
            },
        ));
    }
    Ok(())
}

/// Enforce the advanced-bag choice lists a target advertises (`lrSchedulers`, `timestepTypes`,
/// `timestepBiases`, `lossTypes`, `mixedPrecisions`) and a pinned-off `gradientCheckpointing`. A
/// target that advertises a list is promising the Studio exactly those choices, so a request
/// outside it is refused at submit rather than travelling to a trainer that would refuse it after
/// load (or, worse, ignore it). Targets that advertise no list keep their established behaviour.
fn validate_advertised_advanced_choices(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    for (limit, field) in [
        ("lrSchedulers", "lrScheduler"),
        ("timestepTypes", "timestepType"),
        ("timestepBiases", "timestepBias"),
        ("lossTypes", "lossType"),
        ("mixedPrecisions", "mixedPrecision"),
    ] {
        let Some(advertised) = target.limits.get(limit) else {
            continue;
        };
        let allowed = advertised
            .as_array()
            .and_then(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().map(str::to_owned))
                    .collect::<Option<Vec<_>>>()
            })
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                invalid_advertised_limit(limit, "a choice list must be non-empty strings")
            })?;
        let Some(requested) = config.advanced.get(field) else {
            continue;
        };
        let requested = requested.as_str().ok_or_else(|| {
            TrainingPlanError::InvalidConfig(format!("{field} must be a string."))
        })?;
        let normalized = requested.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if !allowed
            .iter()
            .any(|value| value.eq_ignore_ascii_case(&normalized))
        {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: field.to_owned(),
                    value: requested.to_owned(),
                    allowed,
                },
            ));
        }
    }
    if target.limits.get("gradientCheckpointing") == Some(&Value::Bool(false))
        && config.advanced.get("gradientCheckpointing") == Some(&Value::Bool(true))
    {
        return Err(TrainingPlanError::InvalidConfig(format!(
            "Training target '{}' does not implement gradient checkpointing.",
            target.id
        )));
    }
    Ok(())
}

/// Validate MOSS's JSON controls before submission or native request lowering.
/// Missing controls retain the worker defaults; present values must match the native
/// bool/string-array/i32 contract. The native options bag accepts only its deterministic
/// posterior policy (an empty or absent bag also selects that policy).
pub fn validate_moss_training_field_shapes(advanced: &JsonObject) -> Result<(), TrainingPlanError> {
    let invalid = |field: &str, message: &str| TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message: format!("MOSS {field} {message}"),
    };
    if let Some(options) = advanced.get("modelOptions") {
        let options = options
            .as_object()
            .ok_or_else(|| invalid("modelOptions", "must be an object"))?;
        if options
            .iter()
            .any(|(k, v)| k != "vae_posterior" || v.as_str() != Some("posterior_mean_v1"))
        {
            return Err(invalid(
                "modelOptions",
                "accepts only vae_posterior=posterior_mean_v1",
            ));
        }
    }
    if let Some(modules) = advanced.get("loraTargetModules") {
        let modules = modules
            .as_array()
            .ok_or_else(|| invalid("loraTargetModules", "must be an array"))?;
        if modules.iter().any(|module| !module.is_string()) {
            return Err(invalid("loraTargetModules", "must contain only strings"));
        }
    }
    if let Some(factor) = advanced.get("decomposeFactor") {
        let factor = factor
            .as_i64()
            .ok_or_else(|| invalid("decomposeFactor", "must be an integer"))?;
        if factor != -1 && (factor <= 0 || i32::try_from(factor).is_err()) {
            return Err(invalid("decomposeFactor", "must be -1 or a positive i32"));
        }
    }
    for key in ["cacheLatents", "cacheTextEmbeddings"] {
        if advanced.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(invalid(key, "must be true or false"));
        }
    }
    Ok(())
}

/// Checked MOSS literal seed lowering. Zero remains zero; negatives are not an
/// implicit Auto request. Other training families retain their existing policies.
pub fn moss_training_seed(seed: i64) -> Result<u64, TrainingPlanError> {
    u64::try_from(seed).map_err(|_| TrainingPlanError::InvalidField {
        field: "seed".to_owned(),
        message: "MOSS seed must be a literal nonnegative whole number, including zero.".to_owned(),
    })
}

/// Convert a product alpha JSON number into the native f32 contract without letting a
/// mathematically positive f64 overflow to infinity or underflow to zero at the boundary.
pub fn training_alpha_f32(alpha: &ContractNumber) -> Result<f32, TrainingPlanError> {
    let value = alpha.as_f64().unwrap_or(f64::NAN);
    if !(value.is_finite() && value > 0.0) {
        return Err(TrainingPlanError::InvalidConfig(
            "alpha must be a positive finite number.".to_owned(),
        ));
    }
    let native = value as f32;
    if !(native.is_finite() && native > 0.0) {
        return Err(TrainingPlanError::InvalidConfig(
            "alpha must be representable as a positive finite f32.".to_owned(),
        ));
    }
    Ok(native)
}

fn validate_training_config(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let positive = |value: u32, field: &str| {
        if value == 0 {
            Err(TrainingPlanError::InvalidConfig(format!(
                "{field} must be at least 1."
            )))
        } else {
            Ok(())
        }
    };
    if !config_is_full_finetune(config) {
        positive(config.rank, "rank")?;
    }
    training_alpha_f32(&config.alpha)?;
    positive(config.steps, "steps")?;
    positive(config.resolution, "resolution")?;
    positive(config.batch_size, "batchSize")?;
    positive(config.gradient_accumulation, "gradientAccumulation")?;
    let learning_rate = config.learning_rate.as_f64().unwrap_or(f64::NAN);
    if !(learning_rate.is_finite() && learning_rate > 0.0) {
        return Err(TrainingPlanError::InvalidConfig(
            "learningRate must be a positive finite number.".to_owned(),
        ));
    }
    validate_lr_scheduler(config)?;
    validate_weight_noise(config)?;
    validate_gradient_noise(config)?;
    parse_training_bucket_config(&config.advanced).map_err(|message| {
        TrainingPlanError::InvalidField {
            field: RESOLUTION_BUCKETS_KEY.to_owned(),
            message,
        }
    })?;
    depth_anchoring::validate(config)?;
    body_losses::validate(config)?;
    face_losses::validate(config)?;
    latent_perceptual::validate(config)?;
    validate_subject_mask_loss(config)?;
    Ok(())
}

/// Which technique of `advanced` reads the per-image subject masks (sc-2124): `Some(field)` naming
/// the field that turned it on — [`SUBJECT_MASK_LOSS_KEY`] for subject-masked loss, else
/// `normalRestrictToSubject` for the subject-restricted normal loss (sc-24832) — or `None` when the
/// run reads no mask. The API (which has missing masks generated before training) and the worker
/// (which refuses a run still missing one) share this rule.
pub fn subject_mask_consumer(
    advanced: &JsonObject,
) -> Result<Option<&'static str>, TrainingPlanError> {
    if subject_mask_loss_weights(advanced)?.is_some() {
        return Ok(Some(SUBJECT_MASK_LOSS_KEY));
    }
    let body = body_losses::body_loss_settings(advanced)?;
    Ok(subject_mask_consumer_resolved(None, &body))
}

pub fn subject_mask_consumer_resolved(
    weights: Option<(f64, f64)>,
    body: &body_losses::BodyLossSettings,
) -> Option<&'static str> {
    if weights.is_some() {
        return Some(SUBJECT_MASK_LOSS_KEY);
    }
    (body.normal.is_some() && body.normal_restrict_to_subject)
        .then_some(body_losses::NORMAL_RESTRICT_TO_SUBJECT_KEY)
}

/// The resolved subject-masked-loss weights `(background, subject)` of `config`'s `advanced` bag
/// (epic 2123, sc-24828): `Ok(None)` when [`SUBJECT_MASK_LOSS_KEY`] is absent or `false`; else
/// each weight (defaulted when absent) range-checked. Every failure is a
/// [`TrainingPlanError::InvalidField`] naming the field. Shared by submit-time validation and the
/// worker, so the two cannot disagree on defaults or bounds.
pub fn subject_mask_loss_weights(
    advanced: &JsonObject,
) -> Result<Option<(f64, f64)>, TrainingPlanError> {
    let field_error = |field: &str, message: String| TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message,
    };
    let enabled = match technique_value(advanced, SUBJECT_MASK_LOSS_KEY) {
        None => false,
        Some(Value::Bool(on)) => *on,
        Some(_) => {
            return Err(field_error(
                SUBJECT_MASK_LOSS_KEY,
                format!("{SUBJECT_MASK_LOSS_KEY} must be true or false."),
            ))
        }
    };
    let weight = |key: &str, default: f64, zero_ok: bool| -> Result<f64, TrainingPlanError> {
        let Some(value) = technique_value(advanced, key) else {
            return Ok(default);
        };
        let weight = value
            .as_f64()
            .filter(|weight| weight.is_finite())
            .ok_or_else(|| field_error(key, format!("{key} must be a number.")))?;
        let in_range = if zero_ok {
            (0.0..=SUBJECT_MASK_WEIGHT_MAX).contains(&weight)
        } else {
            weight > 0.0 && weight <= SUBJECT_MASK_WEIGHT_MAX
        };
        if !in_range {
            let range = if zero_ok {
                format!("between 0 and {SUBJECT_MASK_WEIGHT_MAX}")
            } else {
                format!("greater than 0 and at most {SUBJECT_MASK_WEIGHT_MAX}")
            };
            return Err(field_error(
                key,
                format!("{key} ({weight}) must be {range}."),
            ));
        }
        Ok(weight)
    };
    // The weights are range-checked even while the toggle is off, so a stored out-of-range value
    // never lies dormant until someone flips the switch.
    let background = weight(
        SUBJECT_MASK_BACKGROUND_WEIGHT_KEY,
        SUBJECT_MASK_BACKGROUND_WEIGHT_DEFAULT,
        true,
    )?;
    let subject = weight(
        SUBJECT_MASK_SUBJECT_WEIGHT_KEY,
        SUBJECT_MASK_SUBJECT_WEIGHT_DEFAULT,
        false,
    )?;
    Ok(enabled.then_some((background, subject)))
}

/// Refuses subject-masked loss on a target that does not advertise
/// [`SUBJECT_MASK_LOSS_SUPPORT_LIMIT`] — a `subjectMaskLoss` field error at submit time instead of a
/// refusal after the job is queued (for a control target, after its condition render). Runs after
/// [`validate_subject_mask_loss`], so the keys are already well-formed here.
fn validate_subject_mask_loss_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    if subject_mask_loss_weights(&config.advanced)?.is_some()
        && !target_supports_subject_mask_loss(target)
    {
        return Err(TrainingPlanError::InvalidField {
            field: SUBJECT_MASK_LOSS_KEY.to_owned(),
            message: format!(
                "{} does not support subject-masked loss ({SUBJECT_MASK_LOSS_KEY}).",
                target.name
            ),
        });
    }
    Ok(())
}

/// Validates the subject-masked-loss keys (epic 2123 E6) — see [`subject_mask_loss_weights`].
fn validate_subject_mask_loss(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    subject_mask_loss_weights(&config.advanced).map(|_| ())
}

/// Target half of the `advanced.resolutionBuckets` validation. Legacy square rows retain the
/// target's advertised-resolution menu; rectangular rows intentionally have no invented product
/// maximum beyond the shared positive/u32/stride parser.
fn validate_resolution_buckets_for_target(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    let buckets = parse_training_bucket_config(&config.advanced).map_err(|message| {
        TrainingPlanError::InvalidField {
            field: RESOLUTION_BUCKETS_KEY.to_owned(),
            message,
        }
    })?;
    let Some(buckets) = buckets else {
        return Ok(());
    };
    // The target's trainer on this platform must declare buckets: reject before queueing instead of
    // allowing a plan to reach native preflight and fail after the product has accepted it.
    if !target_supports_resolution_buckets(target) {
        return Err(TrainingPlanError::InvalidField {
            field: RESOLUTION_BUCKETS_KEY.to_owned(),
            message: format!(
                "{} does not support multi-resolution buckets ({RESOLUTION_BUCKETS_KEY}) on this platform.",
                target.name
            ),
        });
    }
    let ResolutionBucketConfig::Legacy(buckets) = buckets else {
        if target.kernel != "moss_image_lora" {
            return Err(TrainingPlanError::InvalidField {
                field: RESOLUTION_BUCKETS_KEY.to_owned(),
                message: format!(
                    "{} supports square resolution buckets only; width/height buckets require a MOSS training target.",
                    target.name
                ),
            });
        }
        return Ok(());
    };
    let Some(allowed) = target.limits.get("resolutions").and_then(Value::as_array) else {
        return Ok(());
    };
    let allowed = numeric_limit_values("resolutions", allowed)?;
    if let Some(bucket) = buckets
        .iter()
        .find(|bucket| !allowed.contains(&u64::from(bucket.resolution)))
    {
        let allowed = allowed
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(TrainingPlanError::InvalidField {
            field: RESOLUTION_BUCKETS_KEY.to_owned(),
            message: format!(
                "{RESOLUTION_BUCKETS_KEY}: {} does not train at {}px. Supported resolutions: {allowed}.",
                target.name, bucket.resolution
            ),
        });
    }
    Ok(())
}

/// Refuses a non-zero `advanced.weightNoiseSigma` / `gradientNoiseEta` on a target that does not
/// advertise [`WEIGHT_NOISE_SUPPORT_LIMIT`] / [`GRADIENT_NOISE_SUPPORT_LIMIT`] — a field error at
/// submit time instead of a refusal after the job is queued. Runs after [`validate_weight_noise`]
/// and [`validate_gradient_noise`], so the values are already valid numbers here.
fn validate_technique_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    for (key, supported, technique) in [
        (
            WEIGHT_NOISE_SIGMA_KEY,
            target_supports_weight_noise(target),
            "weight noise",
        ),
        (
            GRADIENT_NOISE_ETA_KEY,
            target_supports_gradient_noise(target),
            "gradient noise",
        ),
    ] {
        let value = technique_value(&config.advanced, key)
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if value > 0.0 && !supported {
            return Err(TrainingPlanError::InvalidField {
                field: key.to_owned(),
                message: format!(
                    "{} does not support {technique} ({key}) on this platform.",
                    target.name
                ),
            });
        }
    }
    Ok(())
}

/// Validates `advanced.weightNoiseSigma` (epic 2123 weight noising): when present it must be a
/// finite number in `0..=`[`WEIGHT_NOISE_SIGMA_MAX`], and a non-zero value cannot be combined with
/// a full base fine-tune — weight noise perturbs adapter factors only (E5). Each failure is a
/// [`TrainingPlanError::InvalidField`] naming the field.
fn validate_weight_noise(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let Some(value) = technique_value(&config.advanced, WEIGHT_NOISE_SIGMA_KEY) else {
        return Ok(());
    };
    let field_error = |message: String| TrainingPlanError::InvalidField {
        field: WEIGHT_NOISE_SIGMA_KEY.to_owned(),
        message,
    };
    let sigma = value
        .as_f64()
        .filter(|sigma| sigma.is_finite())
        .ok_or_else(|| field_error(format!("{WEIGHT_NOISE_SIGMA_KEY} must be a number.")))?;
    if !(0.0..=WEIGHT_NOISE_SIGMA_MAX).contains(&sigma) {
        return Err(field_error(format!(
            "{WEIGHT_NOISE_SIGMA_KEY} ({sigma}) must be between 0 and {WEIGHT_NOISE_SIGMA_MAX}."
        )));
    }
    if sigma > 0.0 && config_is_full_finetune(config) {
        return Err(field_error(format!(
            "{WEIGHT_NOISE_SIGMA_KEY} perturbs adapter weights only and cannot be combined with \
             networkType '{NETWORK_TYPE_FULL}'."
        )));
    }
    Ok(())
}

/// Reads an optional finite number in `0..=max` from `advanced[key]` (absent ⇒ `None`), as a
/// field-level [`TrainingPlanError::InvalidField`] on any other value.
fn advanced_bounded_number(
    config: &TrainingConfig,
    key: &str,
    max: f64,
) -> Result<Option<f64>, TrainingPlanError> {
    if technique_value(&config.advanced, key).is_none() {
        return Ok(None);
    }
    preflight_bounded_f64(&config.advanced, key, max, 0.0)
        .map(Some)
        .map_err(|error| TrainingPlanError::InvalidField {
            field: error.field,
            message: error.detail,
        })
}

/// Validates `advanced.gradientNoiseEta` / `gradientNoiseGamma` (epic 2123 gradient noise,
/// sc-24827): each, when present, a finite number within its bound ([`GRADIENT_NOISE_ETA_MAX`],
/// [`GRADIENT_NOISE_GAMMA_MAX`]); a non-zero eta cannot be combined with a full base fine-tune —
/// the noise perturbs adapter gradients only (E5). Each failure names its field.
fn validate_gradient_noise(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let eta = advanced_bounded_number(config, GRADIENT_NOISE_ETA_KEY, GRADIENT_NOISE_ETA_MAX)?;
    advanced_bounded_number(config, GRADIENT_NOISE_GAMMA_KEY, GRADIENT_NOISE_GAMMA_MAX)?;
    if eta.is_some_and(|eta| eta > 0.0) && config_is_full_finetune(config) {
        return Err(TrainingPlanError::InvalidField {
            field: GRADIENT_NOISE_ETA_KEY.to_owned(),
            message: format!(
                "{GRADIENT_NOISE_ETA_KEY} perturbs adapter gradients only and cannot be combined \
                 with networkType '{NETWORK_TYPE_FULL}'."
            ),
        });
    }
    Ok(())
}

/// Validates the learning-rate scheduler knobs in `advanced`. The scheduler name
/// must be one of [`SUPPORTED_LR_SCHEDULERS`] (so a non-constant choice is never
/// silently ignored), and an optional `lrWarmupSteps` ramp must be a non-negative
/// integer shorter than the run. Absent keys keep the constant-LR default.
fn validate_lr_scheduler(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    if let Some(value) = config.advanced.get("lrScheduler") {
        let name = value.as_str().ok_or_else(|| {
            TrainingPlanError::InvalidConfig("lrScheduler must be a string.".to_owned())
        })?;
        let normalized = name.trim().to_lowercase();
        if !SUPPORTED_LR_SCHEDULERS.contains(&normalized.as_str()) {
            return Err(TrainingPlanError::InvalidConfig(format!(
                "Unsupported lrScheduler '{name}'. Supported schedulers: {}.",
                SUPPORTED_LR_SCHEDULERS.join(", ")
            )));
        }
    }
    if let Some(value) = config.advanced.get("lrWarmupSteps") {
        let warmup = value
            .as_u64()
            .or_else(|| match value.as_f64() {
                Some(number) if number.is_finite() && number >= 0.0 && number.fract() == 0.0 => {
                    Some(number as u64)
                }
                _ => None,
            })
            .ok_or_else(|| {
                TrainingPlanError::InvalidConfig(
                    "lrWarmupSteps must be a non-negative integer.".to_owned(),
                )
            })?;
        if warmup >= u64::from(config.steps) {
            return Err(TrainingPlanError::InvalidConfig(format!(
                "lrWarmupSteps ({warmup}) must be less than steps ({}).",
                config.steps
            )));
        }
    }
    Ok(())
}

/// Dataset facts read by the pure Iris validators. This is not a persisted training carrier.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDataset {
    pub items: Vec<TrainingDatasetItem>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetItem {
    pub display_name: String,
    pub caption: TrainingCaption,
    #[serde(flatten)]
    pub extra: ExtraFields,
}
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingCaption {
    pub text: String,
}

fn object(value: Value) -> JsonObject {
    value.as_object().cloned().unwrap_or_default()
}
fn is_safe_relative_path(relative: &str) -> bool {
    use std::path::{Component, Path};
    !relative.trim().is_empty()
        && !relative.bytes().any(|byte| byte.is_ascii_control())
        && !relative.contains(['\\', ':'])
        && Path::new(relative)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

use crate::request_validation::ValidationRefusal;

/// Submission carriers whose config policy belongs to this module.
pub fn handles(kind: &str) -> bool {
    matches!(kind, "training" | "lora_train" | "control_training")
}

fn refusal(field: impl Into<String>, detail: impl Into<String>) -> ValidationRefusal {
    let mut error = ValidationRefusal::new(field, detail);
    error.code = "training_field_error";
    error.context["field"] = json!(error.field);
    error
}

/// Preserve existing target-limit identities and structured bounds for all transports.
pub fn policy_refusal(error: TrainingPlanError) -> ValidationRefusal {
    let detail = error.to_string();
    match error {
        TrainingPlanError::TargetLimit(limit) => {
            let context = serde_json::to_value(limit).expect("training limit is serializable");
            let field = context
                .get("field")
                .and_then(Value::as_str)
                .unwrap_or("config");
            let mut error = refusal(field, detail);
            error.code = "training_target_limit";
            for (key, value) in context.as_object().expect("tagged limit object") {
                error.context[key] = value.clone();
            }
            error
        }
        TrainingPlanError::InvalidField { field, .. } => refusal(field, detail),
        _ => {
            let field = crate::training::SPECS
                .iter()
                .filter(|spec| detail.contains(spec.key))
                .max_by_key(|spec| spec.key.len())
                .map_or("config", |spec| spec.key);
            refusal(field, detail)
        }
    }
}

/// Legacy worker compatibility coercions. They never mutate the authored carrier.
pub fn preflight_string(
    advanced: &JsonObject,
    key: &str,
    default: &str,
) -> Result<String, ValidationRefusal> {
    match advanced.get(key) {
        None => Ok(default.to_owned()),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.trim().to_owned()),
        Some(Value::String(_)) => Ok(default.to_owned()),
        Some(_) => Err(refusal(
            key,
            format!("Training config field '{key}' must be a string."),
        )),
    }
}
pub fn preflight_bool(
    advanced: &JsonObject,
    key: &str,
    default: bool,
) -> Result<bool, ValidationRefusal> {
    match advanced.get(key) {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::String(value)) if value.trim().eq_ignore_ascii_case("true") => Ok(true),
        Some(Value::String(value)) if value.trim().eq_ignore_ascii_case("false") => Ok(false),
        Some(_) => Err(refusal(
            key,
            format!("Training config field '{key}' must be a boolean."),
        )),
    }
}
pub fn preflight_u32(
    advanced: &JsonObject,
    key: &str,
    default: u32,
) -> Result<u32, ValidationRefusal> {
    match advanced.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .or_else(|| value.as_str()?.trim().parse().ok())
            .ok_or_else(|| {
                refusal(
                    key,
                    format!("Training config field '{key}' must be a non-negative integer."),
                )
            }),
    }
}
pub fn preflight_i32(
    advanced: &JsonObject,
    key: &str,
    default: i32,
) -> Result<i32, ValidationRefusal> {
    match advanced.get(key) {
        None => Ok(default),
        Some(value) => crate::payload_util::json_integer(value)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| {
                refusal(
                    key,
                    format!("Training config field '{key}' must be a signed 32-bit integer."),
                )
            }),
    }
}
pub fn preflight_f32(
    advanced: &JsonObject,
    key: &str,
    default: f32,
) -> Result<f32, ValidationRefusal> {
    let Some(value) = advanced.get(key) else {
        return Ok(default);
    };
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .and_then(|value| {
            let converted = value as f32;
            (converted.is_finite() && (value == 0.0 || converted != 0.0)).then_some(converted)
        })
        .ok_or_else(|| {
            refusal(
                key,
                format!("Training config field '{key}' must be a finite f32 number."),
            )
        })
}

fn config_carrier<'a>(
    kind: &str,
    payload: &'a JsonObject,
) -> Result<(&'a JsonObject, &'static str), ValidationRefusal> {
    if kind == "training" {
        return Ok((payload, ""));
    }
    payload
        .get("plan")
        .and_then(Value::as_object)
        .map(|plan| (plan, "plan."))
        .ok_or_else(|| {
            refusal(
                "plan",
                "Training job payload requires a resolved plan object.",
            )
        })
}

/// Validate every registered authored training value before tolerant parsers or DTOs can erase it.
/// Omission remains omission; explicit technique nulls keep their historical off/default meaning.
pub fn validate_shape(kind: &str, payload: &JsonObject) -> Result<(), ValidationRefusal> {
    if !handles(kind) {
        return Ok(());
    }
    let (carrier, prefix) = config_carrier(kind, payload)?;
    let config = carrier
        .get("config")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            refusal(
                format!("{prefix}config"),
                "Training config must be an object.",
            )
        })?;
    validate_config_shape(config).map_err(|mut error| {
        error.field = error.field.trim_start_matches("advanced.").to_owned();
        error.context["field"] = json!(error.field);
        error
    })?;
    if kind == "training" {
        for key in ["targetId", "datasetId", "outputName"] {
            if !carrier
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|v| !v.trim().is_empty())
            {
                return Err(refusal(
                    key,
                    if key == "outputName" {
                        "Training output name is required.".to_owned()
                    } else {
                        format!("{key} must be a nonempty string.")
                    },
                ));
            }
        }
        for key in ["dryRun", "licenseAcknowledged"] {
            if carrier.get(key).is_some_and(|value| !value.is_boolean()) {
                return Err(refusal(key, format!("{key} must be true or false.")));
            }
        }
        for key in ["datasetVersion", "presetVersion"] {
            if carrier
                .get(key)
                .filter(|v| !v.is_null())
                .is_some_and(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()).is_none())
            {
                return Err(refusal(
                    key,
                    format!("{key} must be a non-negative u32 integer."),
                ));
            }
        }
        if carrier
            .get("presetId")
            .filter(|v| !v.is_null())
            .is_some_and(|v| !v.is_string())
        {
            return Err(refusal("presetId", "presetId must be a string."));
        }
        if carrier.get("presetVersion").is_some_and(|v| !v.is_null())
            && !carrier.get("presetId").is_some_and(Value::is_string)
        {
            return Err(refusal(
                "presetVersion",
                "presetVersion requires a matching presetId.",
            ));
        }
        for key in ["outputScope", "requestedGpu", "qualityPreset"] {
            if carrier.get(key).is_some_and(|v| !v.is_string()) {
                return Err(refusal(key, format!("{key} must be a string.")));
            }
        }
    }
    Ok(())
}

/// Config-only entry also guards typed native callers before permissive engine lowering.
pub fn validate_config_shape(config: &JsonObject) -> Result<(), ValidationRefusal> {
    serde_path_to_error::deserialize::<_, TrainingConfig>(Value::Object(config.clone()))
        .map_err(|error| refusal(error.path().to_string(), error.inner().to_string()))?;
    let root = json!({"config": config});
    for spec in crate::training::SPECS
        .iter()
        .filter(|spec| spec.location.starts_with("training.config.advanced"))
    {
        let relative = spec
            .location
            .strip_prefix("training.")
            .expect("training location");
        visit_registered(
            &root,
            &relative.split('.').collect::<Vec<_>>(),
            "",
            &mut |parent, path| {
                let Some(value) = parent.get(spec.key) else {
                    return Ok(());
                };
                let field = format!("{path}.{}", spec.key)
                    .trim_start_matches("config.advanced.")
                    .to_owned();
                if value.is_null()
                    && (TECHNIQUE_KEYS.contains(&spec.key)
                        || matches!(
                            spec.key,
                            "memoryLimitBytes"
                                | "memoryLimitFraction"
                                | "trainingAdapterRepo"
                                | "trainingAdapterVersion"
                                | "iris"
                                | "iris_restore"
                                | "aspectRatioBuckets"
                        )
                        || relative.starts_with("config.advanced.iris"))
                {
                    return Ok(());
                }
                // Common native compatibility keys accept the exact same string coercions here and
                // during engine mapping. Family validators below can intentionally be stricter.
                if relative == "config.advanced" {
                    let coercion = match spec.key {
                        "gradientCheckpointing"
                        | "resume"
                        | "cacheLatents"
                        | "cacheTextEmbeddings" => {
                            Some(preflight_bool(parent, spec.key, false).map(|_| ()))
                        }
                        "lrWarmupSteps" | "sampleEvery" | "sampleSteps" | "sampleCount" => {
                            Some(preflight_u32(parent, spec.key, 0).map(|_| ()))
                        }
                        "decomposeFactor" => Some(preflight_i32(parent, spec.key, -1).map(|_| ())),
                        "weightDecay" | "sampleGuidanceScale" | "memoryLimitFraction" => {
                            Some(preflight_f32(parent, spec.key, 0.0).map(|_| ()))
                        }
                        "memoryLimitBytes" => Some(
                            value
                                .as_u64()
                                .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
                                .filter(|v| *v > 0)
                                .map(|_| ())
                                .ok_or_else(|| {
                                    refusal(&field, "memoryLimitBytes must be a positive u64.")
                                }),
                        ),
                        _ => None,
                    };
                    if let Some(result) = coercion {
                        return result.map_err(|mut error| {
                            error.field = field.clone();
                            error
                        });
                    }
                }
                let valid = match spec.shape {
                    crate::Shape::Bool => value.is_boolean(),
                    crate::Shape::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
                    crate::Shape::Number => value.as_f64().is_some_and(f64::is_finite),
                    crate::Shape::Text => value.is_string(),
                    crate::Shape::Strings => value
                        .as_array()
                        .is_some_and(|v| v.iter().all(Value::is_string)),
                    crate::Shape::Numbers => value
                        .as_array()
                        .is_some_and(|v| v.iter().all(|n| n.as_f64().is_some_and(f64::is_finite))),
                    crate::Shape::Object(_) => value.is_object(),
                    crate::Shape::List(_) => value.is_array(),
                    crate::Shape::OneOf("boolean or string list") => {
                        value.is_boolean()
                            || value
                                .as_array()
                                .is_some_and(|v| v.iter().all(Value::is_string))
                    }
                    crate::Shape::OneOf(_) => value.is_string(),
                };
                if valid {
                    Ok(())
                } else {
                    Err(refusal(
                        field,
                        format!("{} has the wrong JSON type for {:?}.", spec.key, spec.shape),
                    ))
                }
            },
        )?;
    }
    if let Some(advanced) = config.get("advanced").and_then(Value::as_object) {
        if let Some(value) = technique_value(advanced, "memoryLimitFraction") {
            let fraction = value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse::<f64>().ok());
            if !fraction.is_some_and(|n| n.is_finite() && n > 0.0 && n <= 1.0 && n as f32 > 0.0) {
                return Err(refusal(
                    "advanced.memoryLimitFraction",
                    "memoryLimitFraction must be a positive finite f32 fraction at most 1.",
                ));
            }
        }
        training_output_scope(advanced)?;
    }
    Ok(())
}

fn visit_registered(
    value: &Value,
    parts: &[&str],
    path: &str,
    visit: &mut impl FnMut(&JsonObject, &str) -> Result<(), ValidationRefusal>,
) -> Result<(), ValidationRefusal> {
    let Some((part, rest)) = parts.split_first() else {
        return value.as_object().map_or(Ok(()), |value| visit(value, path));
    };
    if let Some(key) = part.strip_suffix("[]") {
        if let Some(values) = value.get(key).and_then(Value::as_array) {
            for (index, value) in values.iter().enumerate() {
                visit_registered(value, rest, &format!("{path}.{key}[{index}]"), visit)?;
            }
        }
    } else if let Some(value) = value.get(part) {
        visit_registered(
            value,
            rest,
            format!("{path}.{part}").trim_start_matches('.'),
            visit,
        )?;
    }
    Ok(())
}

/// Omission and blanks retain the historical project-scoped output default.
pub fn training_output_scope(advanced: &JsonObject) -> Result<String, ValidationRefusal> {
    let scope = preflight_string(advanced, "outputScope", "project")?;
    match scope.as_str() {
        "project" | "global" => Ok(scope),
        _ => Err(refusal(
            "outputScope",
            format!("Unsupported training outputScope: {scope}. Use project or global."),
        )),
    }
}

/// Static target/config policy. Facts must come from the effective native catalog, not the plan.
pub fn validate(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<(), ValidationRefusal> {
    validate_shape(kind, payload)?;
    let (carrier, _) = config_carrier(kind, payload)?;
    let target = resolve_target(kind, carrier, entry)?;
    let config: TrainingConfig = serde_json::from_value(carrier["config"].clone())
        .map_err(|error| refusal("config", error.to_string()))?;
    validate_training_config_for_target(&target, &config).map_err(policy_refusal)?;
    if kind == "control_training" && target.kernel != "krea_control" {
        return Err(refusal(
            "target.kernel",
            format!(
                "control_training expects a 'krea_control' plan, got kernel '{}'.",
                target.kernel
            ),
        ));
    }
    if kind != "training" {
        if let Some(submitted) = carrier.get("target").and_then(Value::as_object) {
            for (key, expected) in [
                ("targetId", target.id.as_str()),
                ("kernel", target.kernel.as_str()),
                ("baseModel", target.base_model.as_str()),
            ] {
                if submitted
                    .get(key)
                    .and_then(Value::as_str)
                    .is_some_and(|value| value != expected)
                {
                    return Err(refusal(format!("target.{key}"), format!("Training plan target.{key} does not match the authoritative target '{0}'.", target.id)));
                }
            }
        }
    }

    if kind == "training"
        && flux3::KERNELS.contains(&target.kernel.as_str())
        && config.advanced.get("resume").and_then(Value::as_bool) == Some(true)
    {
        return Err(refusal("resume", "Resume a stopped FLUX.3 run using its queue entry; a new submit cannot select a previous checkpoint identity."));
    }
    if kind == "training"
        && target.output_kind == TrainingOutputKind::ControlBranch
        && carrier
            .get("dryRun")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    {
        return Err(refusal("dryRun", "ControlNet training runs the full render + train pipeline and does not support a dry run; submit with dryRun=false."));
    }
    Ok(())
}

fn resolve_target(
    kind: &str,
    carrier: &JsonObject,
    entry: &JsonObject,
) -> Result<TrainingTarget, ValidationRefusal> {
    let matches = |target: &Value| {
        if kind == "training" {
            target.get("id") == carrier.get("targetId")
        } else {
            let submitted = carrier.get("target");
            target.get("id").is_some()
                && target.get("id")
                    == submitted.and_then(|v| v.get("targetId").or_else(|| v.get("id")))
                || (target.get("kernel").is_some()
                    && target.get("kernel") == submitted.and_then(|v| v.get("kernel"))
                    && target.get("baseModel") == submitted.and_then(|v| v.get("baseModel")))
        }
    };
    let direct = Value::Object(entry.clone());
    let target = [
        Some(&direct),
        entry.get("trainingTarget"),
        entry.get("target"),
    ]
    .into_iter()
    .flatten()
    .find(|value| value.get("defaults").is_some() && matches(value))
    .or_else(|| {
        entry
            .get("trainingTargets")
            .or_else(|| entry.get("targets"))
            .and_then(|v| v.as_array().or_else(|| v.get("targets")?.as_array()))
            .and_then(|targets| targets.iter().find(|value| matches(value)))
    })
    .ok_or_else(|| {
        refusal(
            "targetId",
            "Training validation requires the matching authoritative training target facts.",
        )
    })?;
    serde_json::from_value(target.clone()).map_err(|error| {
        refusal(
            "targetId",
            format!("Invalid training target facts: {error}"),
        )
    })
}

/// How many ordered reference images one instruction-edit training item may carry for `target`
/// (`limits.maxReferenceImages`), or `0` when the target cannot train on edit pairs at all — every
/// target before sc-24161. For Qwen Image 2.1 this mirrors the base model's own
/// `limits.maxReferenceAssets` (10) and the engine trainer's `max_reference_images`; the contract
/// tests pin all three together so the cap is never a second, drifting number.
pub fn training_target_max_reference_images(target: &TrainingTarget) -> u32 {
    target
        .limits
        .get(MAX_REFERENCE_IMAGES_LIMIT)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

/// Whether `target` trains on instruction-edit pairs (its reference cap is non-zero, sc-24161).
pub fn training_target_trains_edit_pairs(target: &TrainingTarget) -> bool {
    training_target_max_reference_images(target) > 0
}

pub fn apply_candle_training_defaults(kernel: &str, config: &mut TrainingConfig) {
    match kernel {
        // These Candle trainers are intentionally dense and do not render previews. MLX retains
        // its established checkpointing/preview defaults in the core catalog and macOS response.
        "kolors_lora" | "sd3_lora" | "mage_flow_lora" | "anima_lora" => {
            config
                .advanced
                .insert("gradientCheckpointing".to_owned(), json!(false));
            config.advanced.insert("sampleEvery".to_owned(), json!(0));
            if kernel == "mage_flow_lora" {
                config.advanced.insert(
                    "fullFinetuneConfig".to_owned(),
                    json!({
                        "mixedPrecision": "f32",
                        "gradientCheckpointing": false
                    }),
                );
            }
        }
        // The Wan single-DiT and I2V trainers do not preview. Both A14B variants require
        // checkpointing on Candle; TI2V-5B remains a negative and honors its existing value.
        "wan_lora" => {
            config.advanced.insert("sampleEvery".to_owned(), json!(0));
        }
        "wan_moe_lora" => {
            config
                .advanced
                .insert("gradientCheckpointing".to_owned(), json!(true));
            config.advanced.insert("sampleEvery".to_owned(), json!(0));
        }
        _ => {}
    }
}

pub fn preflight_bounded_f64(
    advanced: &JsonObject,
    key: &str,
    max: f64,
    default: f64,
) -> Result<f64, ValidationRefusal> {
    let Some(value) = technique_value(advanced, key) else {
        return Ok(default);
    };
    value
        .as_f64()
        .filter(|number| {
            number.is_finite()
                && (0.0..=max).contains(number)
                && (*number == 0.0 || (*number as f32).is_finite() && *number as f32 != 0.0)
        })
        .ok_or_else(|| {
            refusal(
                key,
                format!("Training config field '{key}' must be a number between 0 and {max}."),
            )
        })
}

/// Generic native lowering defaults. The authored advanced object remains a separate snapshot.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedTrainingAdvanced {
    pub weight_decay: f32,
    pub lr_scheduler: String,
    pub lr_warmup_steps: u32,
    pub network_type: String,
    #[serde(skip_serializing)]
    pub full_finetune: bool,
    pub decompose_factor: i32,
    pub lora_target_modules: Vec<String>,
    pub timestep_type: String,
    pub timestep_bias: String,
    pub loss_type: String,
    #[serde(rename = "mixedPrecision")]
    pub train_dtype: String,
    pub gradient_checkpointing: bool,
    pub sample_every: u32,
    pub sample_steps: u32,
    pub sample_guidance_scale: f32,
    pub sample_count: u32,
    pub sample_prompts: Vec<String>,
    pub resume: bool,
    pub weight_noise_sigma: f32,
    pub gradient_noise_eta: f32,
    pub gradient_noise_gamma: f32,
    pub cache_latents: bool,
    pub cache_text_embeddings: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit_fraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_type: Option<String>,
    pub output_scope: String,
    pub requested_gpu: String,
}

/// Preserve the existing generic bf16/f32 lowering.
pub fn normalize_train_dtype(value: &str) -> String {
    if value.trim().eq_ignore_ascii_case("bf16") {
        "bf16"
    } else {
        "f32"
    }
    .to_owned()
}

/// Iris's trainer supports fp16 with dynamic loss scaling; other kernels keep the generic rule.
pub fn train_dtype_for_kernel(kernel: &str, value: &str) -> String {
    if iris::is_iris_training_kernel(kernel) && value.trim().eq_ignore_ascii_case("fp16") {
        "fp16".to_owned()
    } else {
        normalize_train_dtype(value)
    }
}

/// Resolve the same typed execution controls when a native caller has a bare config and kernel.
pub fn resolve_advanced_for_kernel(
    advanced: &JsonObject,
    kernel: &str,
) -> Result<ResolvedTrainingAdvanced, ValidationRefusal> {
    let mut resolved_advanced = resolve_advanced(advanced)?;
    resolved_advanced.train_dtype = train_dtype_for_kernel(
        kernel,
        &preflight_string(advanced, "mixedPrecision", "bf16")?,
    );
    if kernel == "moss_image_lora" {
        resolved_advanced.timestep_type =
            moss_timestep_type(advanced, resolved_advanced.timestep_type);
    }
    Ok(resolved_advanced)
}

/// Iris task objectives default to full training; generation and other models default to LoRA.
pub fn network_type_for_model(
    advanced: &JsonObject,
    kernel: &str,
    base_model: &str,
) -> Result<String, ValidationRefusal> {
    let default = if iris::is_iris_training_kernel(kernel)
        && matches!(
            iris::IrisObjective::from_base_model(base_model),
            Some(iris::IrisObjective::Depth | iris::IrisObjective::Restoration)
        ) {
        "full"
    } else {
        "lora"
    };
    preflight_string(advanced, "networkType", default).map(|value| value.to_ascii_lowercase())
}

pub fn resolve_advanced_for_model(
    advanced: &JsonObject,
    kernel: &str,
    base_model: &str,
) -> Result<ResolvedTrainingAdvanced, ValidationRefusal> {
    // FLUX.3's native trainer has a source profile whose omitted advanced controls must be
    // resolved before the generic engine fields are mapped. Keep the same default owner for API,
    // worker, and preview callers rather than seeding a second copy at worker dispatch.
    let seeded;
    let advanced = if flux3::KERNELS.contains(&kernel) {
        seeded = {
            let mut defaults = flux3::target(kernel == flux3::KERNELS[1]).defaults.advanced;
            defaults.extend(advanced.clone());
            defaults
        };
        &seeded
    } else {
        advanced
    };
    let mut result = resolve_advanced_for_kernel(advanced, kernel)?;
    result.network_type = network_type_for_model(advanced, kernel, base_model)?;
    result.full_finetune = result.network_type == "full";
    Ok(result)
}

/// Resolve legacy omissions and common scalar coercions once for previews and native mapping.
pub fn resolve_advanced(
    advanced: &JsonObject,
) -> Result<ResolvedTrainingAdvanced, ValidationRefusal> {
    let strings = |key: &str| -> Result<Vec<String>, ValidationRefusal> {
        let Some(value) = advanced.get(key) else {
            return Ok(Vec::new());
        };
        value
            .as_array()
            .and_then(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().map(str::to_owned))
                    .collect()
            })
            .ok_or_else(|| refusal(key, format!("{key} must contain only strings.")))
    };
    let sample_count = preflight_u32(advanced, "sampleCount", 4)?;
    let sample_prompts = strings("samplePrompts")?
        .into_iter()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .take(sample_count as usize)
        .collect();
    let memory_limit_bytes = technique_value(advanced, "memoryLimitBytes")
        .map(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    refusal(
                        "memoryLimitBytes",
                        "memoryLimitBytes must be a positive u64.",
                    )
                })
        })
        .transpose()?;
    let memory_limit_fraction = technique_value(advanced, "memoryLimitFraction")
        .map(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0 && *v as f32 > 0.0)
                .map(|v| v as f32)
                .ok_or_else(|| {
                    refusal(
                        "memoryLimitFraction",
                        "memoryLimitFraction must be a positive finite f32 fraction at most 1.",
                    )
                })
        })
        .transpose()?;
    let network_type = preflight_string(advanced, "networkType", "lora")?;
    let control_type = advanced
        .get("controlType")
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| refusal("controlType", "controlType must be a string."))
        })
        .transpose()?;
    Ok(ResolvedTrainingAdvanced {
        weight_decay: preflight_f32(advanced, "weightDecay", 0.0)?,
        lr_scheduler: preflight_string(advanced, "lrScheduler", "constant")?,
        lr_warmup_steps: preflight_u32(advanced, "lrWarmupSteps", 0)?,
        full_finetune: network_type.eq_ignore_ascii_case("full"),
        network_type,
        decompose_factor: preflight_i32(advanced, "decomposeFactor", -1)?,
        lora_target_modules: strings("loraTargetModules")?,
        timestep_type: preflight_string(advanced, "timestepType", "sigmoid")?,
        timestep_bias: preflight_string(advanced, "timestepBias", "balanced")?,
        loss_type: preflight_string(advanced, "lossType", "mse")?,
        train_dtype: normalize_train_dtype(&preflight_string(advanced, "mixedPrecision", "bf16")?),
        gradient_checkpointing: preflight_bool(advanced, "gradientCheckpointing", false)?,
        sample_every: preflight_u32(advanced, "sampleEvery", 0)?,
        sample_steps: preflight_u32(advanced, "sampleSteps", 20)?,
        sample_guidance_scale: preflight_f32(advanced, "sampleGuidanceScale", 1.0)?,
        sample_count,
        sample_prompts,
        resume: preflight_bool(advanced, "resume", false)?,
        weight_noise_sigma: preflight_bounded_f64(
            advanced,
            WEIGHT_NOISE_SIGMA_KEY,
            WEIGHT_NOISE_SIGMA_MAX,
            0.0,
        )? as f32,
        gradient_noise_eta: preflight_bounded_f64(
            advanced,
            GRADIENT_NOISE_ETA_KEY,
            GRADIENT_NOISE_ETA_MAX,
            0.0,
        )? as f32,
        gradient_noise_gamma: preflight_bounded_f64(
            advanced,
            GRADIENT_NOISE_GAMMA_KEY,
            GRADIENT_NOISE_GAMMA_MAX,
            GRADIENT_NOISE_GAMMA_DEFAULT,
        )? as f32,
        cache_latents: preflight_bool(advanced, "cacheLatents", true)?,
        cache_text_embeddings: preflight_bool(advanced, "cacheTextEmbeddings", true)?,
        memory_limit_bytes,
        memory_limit_fraction,
        control_type,
        output_scope: training_output_scope(advanced)?,
        requested_gpu: preflight_string(advanced, "requestedGpu", "auto")?,
    })
}

/// Return an effective copy while retaining authored omission and provenance in the input.
pub fn resolve_payload(
    kind: &str,
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<JsonObject, ValidationRefusal> {
    validate(kind, payload, entry)?;
    let (carrier, _) = config_carrier(kind, payload)?;
    let target = resolve_target(kind, carrier, entry)?;
    let advanced = carrier["config"]["advanced"]
        .as_object()
        .expect("validated advanced object");
    let resolved_advanced =
        resolve_advanced_for_model(advanced, &target.kernel, &target.base_model)?;
    let resolved =
        serde_json::to_value(resolved_advanced).expect("resolved training config serializes");
    let mut result = Value::Object(payload.clone());
    let effective = if kind == "training" {
        &mut result["config"]["advanced"]
    } else {
        &mut result["plan"]["config"]["advanced"]
    };
    effective
        .as_object_mut()
        .expect("validated advanced object")
        .extend(resolved.as_object().expect("resolved object").clone());
    if target.kernel == "moss_image_lora" {
        effective["modelOptions"] = Value::Object(moss_model_options(advanced));
    }
    if kind == "training" && !payload.contains_key("dryRun") {
        result["dryRun"] = json!(true);
    }
    Ok(result.as_object().expect("request object").clone())
}

/// Preset existence is catalog-owned; this portable rule owns its target/version binding.
pub fn validate_preset_binding(
    target_id: &str,
    preset_id: &str,
    preset_target_id: &str,
    preset_version: u32,
    requested_version: Option<u32>,
) -> Result<(), ValidationRefusal> {
    if preset_target_id != target_id {
        return Err(refusal("presetId", format!("Training preset '{preset_id}' targets '{preset_target_id}', but the request targets '{target_id}'.")));
    }
    if let Some(requested_version) = requested_version {
        if requested_version != preset_version {
            return Err(refusal("presetVersion", format!("Training preset '{preset_id}' is version {preset_version}, but the request pinned version {requested_version}.")));
        }
    }
    Ok(())
}

/// MOSS's portable posterior and sampler defaults; authored values are validated separately.
pub fn moss_model_options(advanced: &JsonObject) -> JsonObject {
    let mut options = advanced
        .get("modelOptions")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    options
        .entry("vae_posterior".to_owned())
        .or_insert_with(|| json!("posterior_mean_v1"));
    options
}

/// The native trainer ABI keeps authored provider options, except for two family contracts:
/// MOSS takes its validated posterior bag, and FLUX.3 keeps only its source-profile namespace so
/// generic retry controls cannot change a checkpoint identity hash.
pub fn training_model_options_wire(advanced: &JsonObject, kernel: &str) -> JsonObject {
    if kernel == "moss_image_lora" {
        return moss_model_options(advanced);
    }
    if flux3::KERNELS.contains(&kernel) {
        let mut seeded = flux3::target(kernel == flux3::KERNELS[1]).defaults.advanced;
        seeded.extend(advanced.clone());
        return seeded
            .into_iter()
            .filter(|(key, _)| key == "flux3")
            .collect();
    }
    advanced.clone()
}
pub fn moss_timestep_type(advanced: &JsonObject, mapped: String) -> String {
    if advanced
        .get("timestepType")
        .and_then(Value::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        "moss_stage3_v1".to_owned()
    } else {
        mapped
    }
}

/// Resolve the established square/rectangular carrier and the explicit rectangular alias together.
pub fn parse_training_bucket_config(
    advanced: &JsonObject,
) -> Result<Option<ResolutionBucketConfig>, String> {
    let Some(aspect) = technique_value(advanced, "aspectRatioBuckets") else {
        return parse_resolution_bucket_config(advanced);
    };
    if technique_value(advanced, RESOLUTION_BUCKETS_KEY).is_some() {
        return Err("resolutionBuckets and aspectRatioBuckets cannot both be present.".to_owned());
    }
    let mut canonical = advanced.clone();
    canonical.insert(RESOLUTION_BUCKETS_KEY.to_owned(), aspect.clone());
    let parsed = parse_resolution_bucket_config(&canonical)?;
    if matches!(parsed, Some(ResolutionBucketConfig::Legacy(_))) {
        return Err("aspectRatioBuckets requires width/height/repeats rows.".to_owned());
    }
    Ok(parsed)
}
