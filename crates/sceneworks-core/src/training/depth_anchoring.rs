//! **Depth anchoring** (epic 2123, sc-2125) — the product-side contract of the native trainer's
//! decoded-x0 Depth-Anything-V2 anchoring loss: the `advanced` keys, their bounds (the web form
//! enforces the identical bounds — a parity test pins them, E6), the one strict parser the API
//! (submit-time field errors) and the worker (preflight + engine mapping) share, and the auxiliary
//! catalog models the technique needs installed.

use serde_json::Value;

use super::{TrainingConfig, TrainingPlanError, TrainingTarget};
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
/// mechanism as weight noise ([`super::WEIGHT_NOISE_SUPPORT_LIMIT`]): the builtin catalog carries
/// the MLX truth, [`super::project_candle_training_limits`] removes it for Candle, the web form
/// shows the toggle only where it is `true`, submit-time validation refuses a non-zero weight
/// elsewhere, and a worker test pins the flag to the linked trainer descriptors.
pub const DEPTH_ANCHORING_SUPPORT_LIMIT: &str = "supportsDepthAnchoring";

/// Whether `target` (as projected for the serving platform) advertises depth-anchoring support.
pub fn target_supports_depth_anchoring(target: &TrainingTarget) -> bool {
    target
        .limits
        .get(DEPTH_ANCHORING_SUPPORT_LIMIT)
        .and_then(Value::as_bool)
        == Some(true)
}

/// Refuses an enabled depth-anchoring request on a target that does not advertise
/// [`DEPTH_ANCHORING_SUPPORT_LIMIT`] — a `depthAnchoringWeight` field error at submit time instead
/// of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    if depth_anchoring_settings(&config.advanced)?.is_some()
        && !target_supports_depth_anchoring(target)
    {
        return Err(field_error(
            DEPTH_ANCHORING_WEIGHT_KEY,
            format!(
                "{} does not support depth anchoring ({DEPTH_ANCHORING_WEIGHT_KEY}) on this \
                 platform.",
                target.name
            ),
        ));
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

/// The tiny x0 decoder for a native trainer's latent family (keyed by the engine trainer id), if
/// one is cataloged. Only the FLUX.1-latent Z-Image trainer has one today (TAEF1); a trainer
/// without one cannot run a decoded-x0 perceptual loss.
pub fn x0_decoder_for_trainer(trainer_id: &str) -> Option<&'static AuxTrainingModel> {
    match trainer_id {
        "z_image_turbo" => Some(&TAEF1_MODEL),
        _ => None,
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
    match advanced.get(key) {
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
    let weight = match advanced.get(DEPTH_ANCHORING_WEIGHT_KEY) {
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
    let model = match advanced.get(DEPTH_ANCHORING_MODEL_KEY) {
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
    let every = match advanced.get(DEPTH_ANCHORING_EVERY_KEY) {
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
