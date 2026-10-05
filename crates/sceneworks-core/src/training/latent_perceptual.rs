//! **Latent-space perceptual losses** (epic 2123, sc-24833) — the product-side contract of the
//! native trainer's VAE perceptual anchor and E-LatentLPIPS losses: the `advanced` keys, their
//! bounds (the web form enforces the identical bounds — a parity test pins them, E6), the one
//! strict parser the API (submit-time field errors) and the worker (preflight + engine mapping)
//! share, the per-target support flags, and the auxiliary catalog models each technique loads.
//!
//! - **VAE anchor** (`vaeAnchor*`): the model's x0 prediction is decoded with the family's tiny
//!   decoder and re-encoded by a frozen FLUX.2 VAE encoder; its multi-scale encoder features are
//!   compared (per-level `1 − cos`) with the training image's, cached once per image. Needs the
//!   family's tiny x0 decoder ([`super::depth_anchoring::x0_decoder_for_trainer`]) and the FLUX.2
//!   VAE ([`FLUX2_VAE_MODEL`]).
//! - **E-LatentLPIPS** (`latentLpips*`): a learned perceptual metric run directly on the x0
//!   latent, with the weights calibrated for the trainer's latent family
//!   ([`latent_lpips_model_for_trainer`]); a family with no published weights cannot use it.
//!
//! Both default to upstream's (ai-toolkit-perceptual) schedule once given a weight: noise window
//! `[0, 0.5]`, added to the diffusion loss on every in-window step (`every = 1`).

use serde_json::Value;

use super::depth_anchoring::AuxTrainingModel;
use super::{TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// One latent-perceptual technique's product contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentLossSpec {
    /// Human name used in messages (e.g. `"the VAE anchor loss"`).
    pub label: &'static str,
    /// `advanced` key of the loss weight. Absent or `0` is off.
    pub weight_key: &'static str,
    /// `advanced` key of the inclusive lower noise-level bound (`0` = clean … `1` = pure noise).
    pub min_t_key: &'static str,
    /// `advanced` key of the inclusive upper noise-level bound.
    pub max_t_key: &'static str,
    /// `advanced` key of the alternation period: `1` adds the loss to the diffusion loss every
    /// step; `n ≥ 2` makes every n-th step a loss-only step.
    pub every_key: &'static str,
    /// Target `limits` flag: `true` when the target's native trainer on the serving platform
    /// honors the technique (its `TrainerDescriptor::techniques` flag). Absent = no.
    pub support_limit: &'static str,
}

/// Inclusive upper bound of either loss weight (web: `vaeAnchorWeightMax` / `latentLpipsWeightMax`).
/// Both losses are O(1) at their optimum-free start (cosine terms ≤ 2, LPIPS ≈ 0.1–1); ten times the
/// upstream test weight is a typo, not a choice.
pub const LATENT_LOSS_WEIGHT_MAX: f64 = 10.0;
/// The weight the form seeds when a loss is switched on (web: `vaeAnchorWeightSuggested` /
/// `latentLpipsWeightSuggested`): upstream's VAE-anchor test weight `1.0`; E-LatentLPIPS is a
/// standalone training objective in its paper (weight `1`).
pub const LATENT_LOSS_WEIGHT_SUGGESTED: f64 = 1.0;
/// Default upper noise-level bound (upstream `*_loss_max_t = 0.5`).
pub const LATENT_LOSS_MAX_T_DEFAULT: f64 = 0.5;
/// Default alternation period: additive on every step (upstream adds both losses).
pub const LATENT_LOSS_EVERY_DEFAULT: u32 = 1;
/// Inclusive upper bound of either alternation period.
pub const LATENT_LOSS_EVERY_MAX: u64 = 16;

/// The VAE perceptual anchor.
pub const VAE_ANCHOR: LatentLossSpec = LatentLossSpec {
    label: "the VAE anchor loss",
    weight_key: "vaeAnchorWeight",
    min_t_key: "vaeAnchorMinT",
    max_t_key: "vaeAnchorMaxT",
    every_key: "vaeAnchorEvery",
    support_limit: "supportsVaeAnchorLoss",
};

/// E-LatentLPIPS.
pub const LATENT_LPIPS: LatentLossSpec = LatentLossSpec {
    label: "the E-LatentLPIPS loss",
    weight_key: "latentLpipsWeight",
    min_t_key: "latentLpipsMinT",
    max_t_key: "latentLpipsMaxT",
    every_key: "latentLpipsEvery",
    support_limit: "supportsLatentLpipsLoss",
};

/// Both techniques, in a stable order.
pub const LATENT_LOSSES: [LatentLossSpec; 2] = [VAE_ANCHOR, LATENT_LPIPS];

/// Builtin targets whose **MLX** trainer declares the VAE anchor (`techniques.vae_anchor_loss`):
/// every trainer wired through the shared aux-loss builder with an x0 decoder — Z-Image, SDXL /
/// Illustrious, Kolors, SD3.5, Lens and Mage-Flow. A worker test pins this to the descriptors.
pub const VAE_ANCHOR_TARGETS: [&str; 9] = [
    "z_image_turbo_lora",
    "sdxl_lora",
    "illustrious_xl_v1_lora",
    "illustrious_xl_v2_lora",
    "kolors_lora",
    "sd3_5_large_lora",
    "sd3_5_medium_lora",
    "lens_turbo_lora",
    "mage_flow_base_lora",
];

/// Builtin targets whose **MLX** trainer declares E-LatentLPIPS (`techniques.latent_lpips_loss`):
/// the trainers whose latent family has published weights ([`latent_lpips_model_for_trainer`]).
pub const LATENT_LPIPS_TARGETS: [&str; 7] = [
    "z_image_turbo_lora",
    "sdxl_lora",
    "illustrious_xl_v1_lora",
    "illustrious_xl_v2_lora",
    "kolors_lora",
    "sd3_5_large_lora",
    "sd3_5_medium_lora",
];

/// Builtin targets whose **Candle** trainer declares the VAE anchor (Mage-Flow, through the shared
/// builder with its full Mage-VAE decoder). No Candle trainer declares E-LatentLPIPS yet.
pub const CANDLE_VAE_ANCHOR_TARGETS: [&str; 1] = ["mage_flow_base_lora"];

/// Insert the MLX-truth latent-perceptual support flags into a builtin target's `limits`.
pub(super) fn insert_mlx_limits(target: &mut TrainingTarget) {
    for (spec, targets) in [
        (&VAE_ANCHOR, &VAE_ANCHOR_TARGETS[..]),
        (&LATENT_LPIPS, &LATENT_LPIPS_TARGETS[..]),
    ] {
        if targets.contains(&target.id.as_str()) {
            target
                .limits
                .insert(spec.support_limit.to_owned(), Value::Bool(true));
        }
    }
}

/// Project the latent-perceptual flags of a builtin (MLX-truth) target onto the Candle backend.
pub(super) fn project_candle_limits(target: &mut TrainingTarget) {
    if !CANDLE_VAE_ANCHOR_TARGETS.contains(&target.id.as_str()) {
        target.limits.remove(VAE_ANCHOR.support_limit);
    }
    target.limits.remove(LATENT_LPIPS.support_limit);
}

/// The FLUX.2 VAE whose encoder the VAE anchor runs — the VAE inside the existing
/// `SceneWorks/flux2-dev-mlx` catalog repo (identical float VAE in every tier); the `flux2_vae`
/// component entry downloads just this directory, and any installed FLUX.2 [dev] tier satisfies it
/// too (see [`FLUX2_VAE_TIER_DIRS`]).
pub const FLUX2_VAE_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "flux2_vae",
    label: "FLUX.2 VAE",
    repo: "SceneWorks/flux2-dev-mlx",
    revision: "2868b1461b2b6e6e05d84e52534df3632b4c7d5d",
    file: "bf16/vae/diffusion_pytorch_model.safetensors",
};

/// The snapshot sub-directories that hold the FLUX.2 VAE, in preference order (each tier of the
/// repo ships the same float VAE; the component entry installs the first).
pub const FLUX2_VAE_TIER_DIRS: [&str; 3] = ["bf16/vae", "q8/vae", "q4/vae"];

/// The published E-LatentLPIPS weights (`Mingguksky/elatentlpips`) for the latent families a
/// native trainer uses: SDXL (SDXL / Illustrious / Kolors), SD3 (SD3.5), FLUX.1 (Z-Image).
pub const ELATENTLPIPS_MODELS: [AuxTrainingModel; 3] = [
    AuxTrainingModel {
        id: "elatentlpips_sdxl",
        label: "E-LatentLPIPS (SDXL latents)",
        repo: "Mingguksky/elatentlpips",
        revision: "0ff7f3f693029324916ed2a392a20694e5f219f3",
        file: "elatentlpips_ckpt/sdxl_latest_vgg16_tuned.pth",
    },
    AuxTrainingModel {
        id: "elatentlpips_sd3",
        label: "E-LatentLPIPS (SD3 latents)",
        repo: "Mingguksky/elatentlpips",
        revision: "0ff7f3f693029324916ed2a392a20694e5f219f3",
        file: "elatentlpips_ckpt/sd3_latest_vgg16_tuned.pth",
    },
    AuxTrainingModel {
        id: "elatentlpips_flux",
        label: "E-LatentLPIPS (FLUX.1 latents)",
        repo: "Mingguksky/elatentlpips",
        revision: "0ff7f3f693029324916ed2a392a20694e5f219f3",
        file: "elatentlpips_ckpt/flux_latest_vgg16_tuned.pth",
    },
];

/// The E-LatentLPIPS weights matching a native trainer's latent family (keyed by the engine
/// trainer id), or `None` when no published weights exist for it (FLUX.2, Qwen-Image / Wan, LTX
/// latents) — such a trainer cannot run E-LatentLPIPS.
pub fn latent_lpips_model_for_trainer(trainer_id: &str) -> Option<&'static AuxTrainingModel> {
    match trainer_id {
        "sdxl" | "kolors" => Some(&ELATENTLPIPS_MODELS[0]),
        "sd3_5_large" | "sd3_5_medium" => Some(&ELATENTLPIPS_MODELS[1]),
        "z_image_turbo" => Some(&ELATENTLPIPS_MODELS[2]),
        _ => None,
    }
}

/// A validated, **enabled** latent-perceptual loss request.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentLossSettings {
    pub weight: f64,
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

/// Strictly parse `spec`'s keys of a plan's `advanced` bag. Every present key is validated (type
/// and range) even when the technique is off; `Ok(None)` means off (weight absent or `0`). Each
/// failure is a [`TrainingPlanError::InvalidField`] naming the offending key.
pub fn latent_loss_settings(
    spec: &LatentLossSpec,
    advanced: &JsonObject,
) -> Result<Option<LatentLossSettings>, TrainingPlanError> {
    let weight = match advanced.get(spec.weight_key) {
        None => 0.0,
        Some(value) => value
            .as_f64()
            .filter(|w| w.is_finite() && (0.0..=LATENT_LOSS_WEIGHT_MAX).contains(w))
            .ok_or_else(|| {
                field_error(
                    spec.weight_key,
                    format!(
                        "{} must be a number between 0 and {LATENT_LOSS_WEIGHT_MAX}.",
                        spec.weight_key
                    ),
                )
            })?,
    };
    let min_t = unit_interval(advanced, spec.min_t_key, 0.0)?;
    let max_t = unit_interval(advanced, spec.max_t_key, LATENT_LOSS_MAX_T_DEFAULT)?;
    if min_t > max_t {
        return Err(field_error(
            spec.max_t_key,
            format!(
                "{} ({max_t}) must be at least {} ({min_t}).",
                spec.max_t_key, spec.min_t_key
            ),
        ));
    }
    let every = match advanced.get(spec.every_key) {
        None => LATENT_LOSS_EVERY_DEFAULT,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=LATENT_LOSS_EVERY_MAX).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| {
                field_error(
                    spec.every_key,
                    format!(
                        "{} must be a whole number between 1 and {LATENT_LOSS_EVERY_MAX}.",
                        spec.every_key
                    ),
                )
            })?,
    };
    if weight == 0.0 {
        return Ok(None);
    }
    Ok(Some(LatentLossSettings {
        weight,
        min_t,
        max_t,
        every,
    }))
}

/// Whether `target` (as projected for the serving platform) advertises `spec`'s support flag.
pub fn target_supports(spec: &LatentLossSpec, target: &TrainingTarget) -> bool {
    target
        .limits
        .get(spec.support_limit)
        .and_then(Value::as_bool)
        == Some(true)
}

/// Submit-time validation of both techniques' keys (called from the plan's config validation).
pub(super) fn validate(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    for spec in &LATENT_LOSSES {
        latent_loss_settings(spec, &config.advanced)?;
    }
    Ok(())
}

/// Refuses an enabled technique on a target that does not advertise it — a field error on the
/// weight key at submit time instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    for spec in &LATENT_LOSSES {
        if latent_loss_settings(spec, &config.advanced)?.is_some() && !target_supports(spec, target)
        {
            return Err(field_error(
                spec.weight_key,
                format!(
                    "{} does not support {} ({}) on this platform.",
                    target.name, spec.label, spec.weight_key
                ),
            ));
        }
    }
    Ok(())
}
