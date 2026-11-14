//! Pure image request bounds and ordered reference policy.
pub mod flux3;
use crate::contracts::{ImageUpscaleRequest, JsonObject};
use crate::payload_util::declared_resolution;
use serde_json::Value;
const DEFAULT_MODEL: &str = "z_image_turbo";
const DEFAULT_COUNT: u32 = 4;
const DEFAULT_DIMENSION: u32 = 1024;
const MIN_COUNT: u32 = 1;
pub const MAX_COUNT: u32 = 8;
pub const BASE_MAX_IMAGE_STEPS: u32 = 80;
const MIN_DIMENSION: u32 = 256;
const MAX_DIMENSION: u32 = 4096;
pub fn default_count(model_manifest_entry: &JsonObject) -> Option<u32> {
    model_manifest_entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("count"))
        .and_then(Value::as_u64)
        .filter(|count| (u64::from(MIN_COUNT)..=u64::from(MAX_COUNT)).contains(count))
        .map(|count| count as u32)
}

pub fn default_resolution(model_manifest_entry: &JsonObject) -> Option<(u32, u32)> {
    declared_resolution(model_manifest_entry, MIN_DIMENSION, MAX_DIMENSION)
}

pub fn unsigned_or(value: Option<&Value>, default: u32) -> u32 {
    value
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(default)
}

pub fn image_count_error(count: u32) -> Option<String> {
    (!(MIN_COUNT..=MAX_COUNT).contains(&count)).then(|| "count must be between 1 and 8".to_owned())
}

pub fn validate_image_request_payload(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<(), String> {
    for key in ["count", "width", "height"] {
        if let Some(value) = payload.get(key).filter(|v| !v.is_null()) {
            if value.as_u64().and_then(|n| u32::try_from(n).ok()).is_none() {
                return Err(format!("{key} must be an unsigned integer"));
            }
        }
    }
    let count = unsigned_or(
        payload.get("count"),
        default_count(entry).unwrap_or(DEFAULT_COUNT),
    );
    if let Some(error) = image_count_error(count) {
        return Err(error);
    }
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MODEL);
    let reference_derived = flux3::is_model(model) && !flux3::parse(payload)?.asset_ids.is_empty();
    if !reference_derived {
        let (width, height) =
            default_resolution(entry).unwrap_or((DEFAULT_DIMENSION, DEFAULT_DIMENSION));
        if let Some(error) = crate::image_limits::image_dimension_error(
            model,
            unsigned_or(payload.get("width"), width),
            unsigned_or(payload.get("height"), height),
            entry,
            MIN_DIMENSION,
            MAX_DIMENSION,
        ) {
            return Err(error);
        }
    }
    let upscale = match payload.get("upscale").filter(|v| !v.is_null()) {
        None => ImageUpscaleRequest::default(),
        Some(value) => serde_json::from_value::<ImageUpscaleRequest>(value.clone())
            .map_err(|error| format!("upscale: {error}"))?,
    };
    upscale.validate()
}

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

pub fn ordered_image_reference_error(
    model: &str,
    payload: &JsonObject,
    model_manifest_entry: &JsonObject,
) -> Option<String> {
    if flux3::is_model(model) {
        return flux3::parse(payload).err();
    }
    if payload
        .get("advanced")
        .and_then(Value::as_object)
        .is_some_and(|advanced| {
            ["flux3_layout", "flux3_reference_ids"]
                .iter()
                .any(|key| advanced.get(*key).is_some_and(|value| !value.is_null()))
        })
    {
        return Some(format!("{model}: FLUX.3 layout and attachment-instance controls are unsupported; remove them or select FLUX.3."));
    }
    let cap = crate::image_limits::image_max_reference_assets(model_manifest_entry)?;
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

pub fn ordered_image_reference_strength_error(model: &str, payload: &JsonObject) -> Option<String> {
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
pub fn ordered_image_reference_ids(payload: &JsonObject) -> Option<Vec<String>> {
    fn scalar<'a>(payload: &'a JsonObject, key: &str) -> Option<Option<&'a str>> {
        match payload.get(key) {
            None | Some(Value::Null) => Some(None),
            Some(Value::String(value)) => Some(Some(value.trim()).filter(|id| !id.is_empty())),
            Some(_) => None,
        }
    }

    let mut ids: Vec<String> = Vec::new();
    for key in ["sourceAssetId", "maskAssetId"] {
        if let Some(id) = scalar(payload, key)? {
            ids.push(id.to_owned());
        }
    }
    match payload.get("referenceAssetIds") {
        None | Some(Value::Null) => {}
        Some(Value::Array(values)) => {
            for value in values {
                let id = value.as_str()?.trim();
                if id.is_empty() {
                    return None;
                }
                ids.push(id.to_owned());
            }
        }
        Some(_) => return None,
    }
    if let Some(id) = scalar(payload, "referenceAssetId")? {
        ids.push(id.to_owned());
    }
    // DEDUPE by asset id, keeping the FIRST occurrence — and this is a correctness fix, not tidying.
    //
    // The web's `editReferenceIds` leads `referenceAssetIds` with the working image while
    // `buildEditJobBody` also sets `sourceAssetId`, so the ordinary Image-Editor payload names the
    // same asset twice. On this engine every entry occupies one of the model's slots AND gets its
    // own number in the prompt template, so sending it twice silently costs a slot and renumbers
    // every reference after it — a duplicate is not a harmless repeat, it changes the render and
    // the count the cap is measured against. First occurrence wins because the earlier slot is the
    // one the prompt refers to.
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.retain(|id| seen.insert(id.clone()));
    Some(ids)
}
pub struct HiresFixInput<'a> {
    pub hires_fix: &'a crate::contracts::HiresFixRequest,
    pub upscale: &'a crate::contracts::ImageUpscaleRequest,
    pub mode: &'a str,
    pub model: &'a str,
    pub model_manifest_entry: &'a JsonObject,
    pub source_asset_id: &'a Option<String>,
    pub reference_asset_id: &'a Option<String>,
    pub reference_asset_ids: &'a [String],
    pub mask_asset_id: &'a Option<String>,
    pub character_id: &'a Option<String>,
    pub character_look_id: &'a Option<String>,
    pub advanced: &'a JsonObject,
    pub width: u32,
    pub height: u32,
}
pub fn hires_fix_request_error(request: &HiresFixInput<'_>, family: &str) -> Option<String> {
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
        || request
            .advanced
            .get("poses")
            .and_then(Value::as_array)
            .is_some_and(|poses| !poses.is_empty())
        || request
            .advanced
            .get("controlImage")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    if has_edit_or_control_input {
        return Some(
            "Hires.fix cannot be combined with image-edit, reference, character, mask, or strict-control inputs."
                .to_owned(),
        );
    }
    if request
        .advanced
        .get("usePid")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("Hires.fix and PiD super-resolution are mutually exclusive.".to_owned());
    }
    if request
        .advanced
        .get("phases")
        .and_then(Value::as_array)
        .is_some_and(|phases| !phases.is_empty())
    {
        return Some("Hires.fix cannot be combined with multi-phase sampling.".to_owned());
    }
    let acceleration_sampler = request
        .advanced
        .get("sampler")
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .is_some_and(|sampler| matches!(sampler.as_str(), "lightning" | "lcm" | "hyper"));
    if request
        .model_manifest_entry
        .get("image")
        .and_then(|v| v.get("supportsHiresFix"))
        == Some(&Value::Bool(false))
        || acceleration_sampler
    {
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
pub fn hires_fix_target_dimension(dimension: u32, upscale_by: f32) -> u32 {
    (dimension as f64 * upscale_by as f64).round() as u32
}

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
