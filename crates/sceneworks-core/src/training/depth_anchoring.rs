//! **Depth anchoring** (epic 2123, sc-2125) — the product-side contract of the native trainer's
//! decoded-x0 Depth-Anything-V2 anchoring loss: the `advanced` keys, their bounds (the web form
//! enforces the identical bounds — a parity test pins them, E6), the one strict parser the API
//! (submit-time field errors) and the worker (preflight + engine mapping) share, and the auxiliary
//! catalog models the technique needs installed.

use serde_json::Value;

use super::{technique_value, TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// `advanced` key of the depth-anchoring loss weight. Absent or `0` is off.
pub const DEPTH_ANCHORING_WEIGHT_KEY: &str = "depthAnchoringWeight";
/// `advanced` key of the Depth-Anything-V2 size (`small` / `base` / `large`; default `small`).
pub const DEPTH_ANCHORING_MODEL_KEY: &str = "depthAnchoringModel";
/// `advanced` key of the inclusive lower noise-level bound (`0` = clean … `1` = pure noise).
pub const DEPTH_ANCHORING_MIN_T_KEY: &str = "depthAnchoringMinT";
/// `advanced` key of the inclusive upper noise-level bound.
pub const DEPTH_ANCHORING_MAX_T_KEY: &str = "depthAnchoringMaxT";
/// `advanced` key of the alternation period: `1` adds the depth loss to the diffusion loss every
/// step; `n ≥ 2` makes every n-th step a depth-only step (the diffusion loss contributes zero).
pub const DEPTH_ANCHORING_EVERY_KEY: &str = "depthAnchoringEvery";

/// Inclusive upper bound of [`DEPTH_ANCHORING_WEIGHT_KEY`] (web: `depthAnchoringWeightMax`).
/// Upstream calibrates `0.1` for DA2-Small and warns that larger values over-smooth; ten times
/// that is a typo, not a choice.
pub const DEPTH_ANCHORING_WEIGHT_MAX: f64 = 1.0;
/// The upstream (ai-toolkit-perceptual) suggested weight for DA2-Small (web:
/// `depthAnchoringWeightSuggested`).
pub const DEPTH_ANCHORING_WEIGHT_SUGGESTED: f64 = 0.1;
/// Inclusive upper bound of [`DEPTH_ANCHORING_EVERY_KEY`] (web: `depthAnchoringEveryMax`).
pub const DEPTH_ANCHORING_EVERY_MAX: u64 = 16;
/// Default alternation period: strict diffusion / depth alternation (upstream default).
pub const DEPTH_ANCHORING_EVERY_DEFAULT: u32 = 2;
/// The accepted [`DEPTH_ANCHORING_MODEL_KEY`] values (web: `depthAnchoringModelOptions`).
pub const DEPTH_ANCHORING_MODELS: [&str; 3] = ["small", "base", "large"];

/// Target `limits` flag: `true` when this target's native trainer on the serving platform honors
/// depth anchoring (its `TrainerDescriptor::techniques.depth_anchoring`). Absent = no. The same
/// gating as weight noise ([`super::WEIGHT_NOISE_SUPPORT_LIMIT`]): every LoRA trainer declares it
/// on MLX and Candle alike except the Krea ControlNet branch (sc-24830), so the builtin
/// catalog flag is static per target; the web form shows the toggle only where it is `true`,
/// submit-time validation refuses a non-zero weight elsewhere, and a worker test pins the flag to
/// the linked trainer descriptors on each platform.
pub const DEPTH_ANCHORING_SUPPORT_LIMIT: &str = "supportsDepthAnchoring";

/// Whether `target` (as projected for the serving platform) advertises depth-anchoring support.
pub fn target_supports_depth_anchoring(target: &TrainingTarget) -> bool {
    target
        .limits
        .get(DEPTH_ANCHORING_SUPPORT_LIMIT)
        .and_then(Value::as_bool)
        == Some(true)
}

/// The LTX-2.5 workflows (`advanced.ltxWorkflow`, web: `depthAnchoringNoVideoLtxWorkflows`) that
/// generate no video stream — audio-only, or video as frozen conditioning — so there is no x0 video
/// latent to decode; the engine's LTX-2.5 trainers refuse depth anchoring for them (sc-24830).
pub const DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS: [&str; 6] = [
    "v2a_lora",
    "t2a_lora",
    "audio_extend_lora",
    "audio_inpainting_lora",
    "audio_suffix_lora",
    "a2a_ic_lora",
];

/// Why depth anchoring cannot run for a `base_model` target + config even though the target
/// advertises it, or `None`: a full base fine-tune (the engine trains aux losses through the adapter
/// step only), or an LTX-2.5 workflow that generates no video. Mirrors the engine's typed refusals.
pub fn depth_anchoring_combination_refusal(
    base_model: &str,
    config: &TrainingConfig,
) -> Option<String> {
    if super::config_is_full_finetune(config) {
        return Some(
            "Depth anchoring trains a LoRA/LoKr adapter only, not a full fine-tune.".to_owned(),
        );
    }
    if base_model == "ltx_2_5" {
        let workflow = config
            .advanced
            .get("ltxWorkflow")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS.contains(&workflow) {
            return Some(format!(
                "Depth anchoring needs a generated video stream; the LTX-2.5 workflow '{workflow}' \
                 generates none."
            ));
        }
    }
    None
}

/// Refuses an enabled depth-anchoring request on a target that does not advertise
/// [`DEPTH_ANCHORING_SUPPORT_LIMIT`], or on a combination the engine refuses
/// ([`depth_anchoring_combination_refusal`]) — a `depthAnchoringWeight` field error at submit time
/// instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    if depth_anchoring_settings(&config.advanced)?.is_none() {
        return Ok(());
    }
    if !target_supports_depth_anchoring(target) {
        return Err(field_error(
            DEPTH_ANCHORING_WEIGHT_KEY,
            format!(
                "{} does not support depth anchoring ({DEPTH_ANCHORING_WEIGHT_KEY}) on this \
                 platform.",
                target.name
            ),
        ));
    }
    if let Some(reason) = depth_anchoring_combination_refusal(&target.base_model, config) {
        return Err(field_error(DEPTH_ANCHORING_WEIGHT_KEY, reason));
    }
    Ok(())
}

/// An auxiliary training-time model a technique needs installed — a `componentOnly` catalog entry
/// in `config/manifests/builtin.models.jsonc` (a parity test pins repo / revision / file to it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxTrainingModel {
    /// Catalog model id.
    pub id: &'static str,
    /// Human label used in refusal messages.
    pub label: &'static str,
    /// Hugging Face repo.
    pub repo: &'static str,
    /// Pinned revision.
    pub revision: &'static str,
    /// The weight file the trainer loads from the snapshot directory.
    pub file: &'static str,
}

/// TAEF1 — the tiny differentiable decoder for the FLUX.1 16-channel latent family (Z-Image's VAE),
/// through which depth anchoring decodes the model's x0 prediction.
pub const TAEF1_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taef1",
    label: "TAEF1 tiny decoder",
    repo: "madebyollin/taef1",
    revision: "b1b2d00e9e440cfbf3dedb34266864da86016ceb",
    file: "diffusion_pytorch_model.safetensors",
};

/// TAESDXL — the tiny decoder for the SDXL 4-channel latent family (SDXL / Illustrious, Kolors).
pub const TAESDXL_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taesdxl",
    label: "TAESDXL tiny decoder",
    repo: "madebyollin/taesdxl",
    revision: "b20258aaef75ef61e659c1e0f14f251cf0ad153e",
    file: "diffusion_pytorch_model.safetensors",
};

/// TAESD3 — the tiny decoder for the SD3 16-channel latent family (SD3.5 Large / Medium).
pub const TAESD3_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taesd3",
    label: "TAESD3 tiny decoder",
    repo: "madebyollin/taesd3",
    revision: "d58dcaccd2b36fcb7a6b9e93c1cc507acab5a778",
    file: "diffusion_pytorch_model.safetensors",
};

/// TAEF2 — the tiny decoder for the FLUX.2 32-channel latent family (Lens).
pub const TAEF2_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taef2",
    label: "TAEF2 tiny decoder",
    repo: "madebyollin/taef2",
    revision: "bd244ebfe4398c84fbf312a2f1c868c676b500e3",
    file: "taef2.safetensors",
};

/// TAEW2.1 (TAEHV) — the tiny decoder for the Wan 2.1 16-channel latent family, which Wan 2.2
/// A14B and the Qwen-Image VAE (Krea 2, Anima) share; run per frame. madebyollin publishes it on
/// GitHub only, so the catalog installs Kijai's Hugging Face re-upload.
pub const TAEW2_1_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taew2_1",
    label: "TAEW2.1 tiny decoder",
    repo: "Kijai/WanVideo_comfy",
    revision: "8260d429d19fd7a72304cad059160b95d843913f",
    file: "taew2_1.safetensors",
};

/// TAEW2.2 (TAEHV) — the tiny decoder for the Wan 2.2 TI2V-5B 48-channel latent family.
pub const TAEW2_2_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taew2_2",
    label: "TAEW2.2 tiny decoder",
    repo: "Kijai/WanVideo_comfy",
    revision: "8260d429d19fd7a72304cad059160b95d843913f",
    file: "taew2_2.safetensors",
};

/// TAELTX2.3 (TAEHV) — the tiny decoder for the LTX-2.3 / LTX-2.5 128-channel latent family.
pub const TAELTX2_3_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "taeltx2_3",
    label: "TAELTX2.3 tiny decoder",
    repo: "Kijai/LTX2.3_comfy",
    revision: "6d980fde0d330f2fed6ff8dfdfddb06d88a004e5",
    file: "vae/taeltx2_3.safetensors",
};

/// Every cataloged tiny x0 decoder (a parity test pins each to its `builtin.models.jsonc` entry).
pub const X0_DECODER_MODELS: [AuxTrainingModel; 7] = [
    TAEF1_MODEL,
    TAESDXL_MODEL,
    TAESD3_MODEL,
    TAEF2_MODEL,
    TAEW2_1_MODEL,
    TAEW2_2_MODEL,
    TAELTX2_3_MODEL,
];

/// Depth-Anything-V2 checkpoints by size (`DEPTH_ANCHORING_MODELS` order).
pub const DEPTH_ANYTHING_V2_MODELS: [AuxTrainingModel; 3] = [
    AuxTrainingModel {
        id: "depth_anything_v2_small",
        label: "Depth Anything V2 Small",
        repo: "depth-anything/Depth-Anything-V2-Small-hf",
        revision: "5426e4f0f36572d16453bbda7a8389317b1bef99",
        file: "model.safetensors",
    },
    AuxTrainingModel {
        id: "depth_anything_v2_base",
        label: "Depth Anything V2 Base",
        repo: "depth-anything/Depth-Anything-V2-Base-hf",
        revision: "b1958afc87fb45a9e3746cb387596094de553ed8",
        file: "model.safetensors",
    },
    AuxTrainingModel {
        id: "depth_anything_v2_large",
        label: "Depth Anything V2 Large",
        repo: "depth-anything/Depth-Anything-V2-Large-hf",
        revision: "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
        file: "model.safetensors",
    },
];

/// The Depth-Anything-V2 catalog model for a validated size string.
pub fn depth_anything_v2_model(size: &str) -> Option<&'static AuxTrainingModel> {
    DEPTH_ANCHORING_MODELS
        .iter()
        .position(|s| *s == size)
        .map(|i| &DEPTH_ANYTHING_V2_MODELS[i])
}

/// Where a native trainer's decoded-x0 perceptual losses get their x0 decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X0DecoderSource {
    /// A cataloged tiny decoder for the trainer's latent family (installed through the Model
    /// Manager; its snapshot directory is handed to the trainer).
    Catalog(&'static AuxTrainingModel),
    /// No tiny decoder matches the family, so the trainer decodes through its own full VAE (from the
    /// base model it already loads — Mage-Flow's one-step Mage-VAE). Nothing extra to install.
    BaseModelVae,
}

/// The x0 decoder for a native trainer's latent family (keyed by the engine trainer id, sc-24830).
/// `None` for a trainer that cannot run a decoded-x0 perceptual loss (the Krea ControlNet branch).
pub fn x0_decoder_for_trainer(trainer_id: &str) -> Option<X0DecoderSource> {
    use X0DecoderSource::{BaseModelVae, Catalog};
    Some(match trainer_id {
        "z_image_turbo" => Catalog(&TAEF1_MODEL),
        "sdxl" | "kolors" => Catalog(&TAESDXL_MODEL),
        "sd3_5_large" | "sd3_5_medium" => Catalog(&TAESD3_MODEL),
        "lens" => Catalog(&TAEF2_MODEL),
        "krea_2_raw" | "anima_base" | "wan2_2_t2v_14b" | "wan2_2_i2v_14b" => {
            Catalog(&TAEW2_1_MODEL)
        }
        "wan2_2_ti2v_5b" => Catalog(&TAEW2_2_MODEL),
        // LTX-2.5 shares the LTX-2.3 latent space (upstream lists TAELTX2.3 for both); its
        // prepared bundles carry the clean video latent the reference decodes.
        "ltx_2_3" | "ltx_2_5" | "ltx_2_5_distilled" => Catalog(&TAELTX2_3_MODEL),
        "mage_flow_base" => BaseModelVae,
        _ => return None,
    })
}

/// Target `limits` key naming the x0 decoder the target's decoded-x0 perceptual losses (depth
/// anchoring, body and face losses, VAE anchor) decode through: `{ "label": <decoder>, "install":
/// <bool> }` — `install` is `true` for a cataloged tiny decoder the user installs from the Models
/// screen, `false` when the trainer decodes through its base model's own VAE. Absent for a target
/// with no x0 decoder. Derived from [`x0_decoder_for_trainer`] (via the target's trainer identity),
/// never written by hand, so the web form's help text names the decoder the worker really loads.
pub const X0_DECODER_LIMIT: &str = "x0Decoder";

/// The [`X0_DECODER_LIMIT`] value for an x0 decoder source.
pub fn x0_decoder_limit(source: X0DecoderSource) -> Value {
    match source {
        X0DecoderSource::Catalog(model) => {
            serde_json::json!({ "label": model.label, "install": true })
        }
        X0DecoderSource::BaseModelVae => {
            serde_json::json!({ "label": "the base model's own VAE", "install": false })
        }
    }
}

/// A validated, **enabled** depth-anchoring request.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnchoringSettings {
    pub weight: f64,
    /// One of [`DEPTH_ANCHORING_MODELS`].
    pub model: &'static str,
    pub min_t: f64,
    pub max_t: f64,
    pub every: u32,
}

fn field_error(field: &str, message: String) -> TrainingPlanError {
    TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message,
    }
}

fn unit_interval(advanced: &JsonObject, key: &str, default: f64) -> Result<f64, TrainingPlanError> {
    match technique_value(advanced, key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|t| t.is_finite() && (0.0..=1.0).contains(t))
            .ok_or_else(|| field_error(key, format!("{key} must be a number between 0 and 1."))),
    }
}

/// Strictly parse the depth-anchoring keys of a plan's `advanced` bag. Every present key is
/// validated (type and range) even when the technique is off; `Ok(None)` means off (weight absent
/// or `0`). Each failure is a [`TrainingPlanError::InvalidField`] naming the offending key.
pub fn depth_anchoring_settings(
    advanced: &JsonObject,
) -> Result<Option<DepthAnchoringSettings>, TrainingPlanError> {
    let weight = match technique_value(advanced, DEPTH_ANCHORING_WEIGHT_KEY) {
        None => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|w| w.is_finite() && (0.0..=DEPTH_ANCHORING_WEIGHT_MAX).contains(w))
            .ok_or_else(|| {
                field_error(
                    DEPTH_ANCHORING_WEIGHT_KEY,
                    format!(
                        "{DEPTH_ANCHORING_WEIGHT_KEY} must be a number between 0 and \
                         {DEPTH_ANCHORING_WEIGHT_MAX}."
                    ),
                )
            })?,
    };
    let model = match technique_value(advanced, DEPTH_ANCHORING_MODEL_KEY) {
        None => DEPTH_ANCHORING_MODELS[0],
        Some(value) => value
            .as_str()
            .map(|s| s.trim().to_ascii_lowercase())
            .and_then(|s| DEPTH_ANCHORING_MODELS.iter().copied().find(|m| *m == s))
            .ok_or_else(|| {
                field_error(
                    DEPTH_ANCHORING_MODEL_KEY,
                    format!(
                        "{DEPTH_ANCHORING_MODEL_KEY} must be one of {}.",
                        DEPTH_ANCHORING_MODELS.join(", ")
                    ),
                )
            })?,
    };
    let min_t = unit_interval(advanced, DEPTH_ANCHORING_MIN_T_KEY, 0.0)?;
    let max_t = unit_interval(advanced, DEPTH_ANCHORING_MAX_T_KEY, 1.0)?;
    if min_t > max_t {
        return Err(field_error(
            DEPTH_ANCHORING_MAX_T_KEY,
            format!(
                "{DEPTH_ANCHORING_MAX_T_KEY} ({max_t}) must be at least \
                 {DEPTH_ANCHORING_MIN_T_KEY} ({min_t})."
            ),
        ));
    }
    let every = match technique_value(advanced, DEPTH_ANCHORING_EVERY_KEY) {
        None => DEPTH_ANCHORING_EVERY_DEFAULT,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=DEPTH_ANCHORING_EVERY_MAX).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| {
                field_error(
                    DEPTH_ANCHORING_EVERY_KEY,
                    format!(
                        "{DEPTH_ANCHORING_EVERY_KEY} must be a whole number between 1 and \
                         {DEPTH_ANCHORING_EVERY_MAX}."
                    ),
                )
            })?,
    };
    if weight == 0.0 {
        return Ok(None);
    }
    Ok(Some(DepthAnchoringSettings {
        weight,
        model,
        min_t,
        max_t,
        every,
    }))
}

/// Submit-time validation (called from the plan's config validation).
pub(super) fn validate(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    depth_anchoring_settings(&config.advanced).map(|_| ())
}
