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
use super::{technique_value, TrainingConfig, TrainingPlanError, TrainingTarget};
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

/// Builtin targets whose trainer declares the VAE anchor (`techniques.vae_anchor_loss`) — on both
/// platforms (technique flags are static per target, sc-24830): every target whose trainer has an x0
/// decoder ([`super::depth_anchoring::x0_decoder_for_trainer`]) and drives the shared aux-loss
/// builder, i.e. exactly the depth-anchoring targets. A worker test pins this to the descriptors.
pub const VAE_ANCHOR_TARGETS: [&str; 16] = [
    "z_image_turbo_lora",
    "sdxl_lora",
    "illustrious_xl_v1_lora",
    "illustrious_xl_v2_lora",
    "kolors_lora",
    "sd3_5_large_lora",
    "sd3_5_medium_lora",
    "lens_turbo_lora",
    "krea_2_raw_lora",
    "anima_base_lora",
    "wan_lora",
    "wan_t2v_14b_lora",
    "wan_i2v_14b_lora",
    "ltx_video_lora",
    "ltx_2_5_video_lora",
    "mage_flow_base_lora",
];

/// Builtin targets whose trainer declares E-LatentLPIPS (`techniques.latent_lpips_loss`) on both
/// platforms: the trainers whose latent family has published weights
/// ([`latent_lpips_model_for_trainer`]). No decoder is involved.
pub const LATENT_LPIPS_TARGETS: [&str; 7] = [
    "z_image_turbo_lora",
    "sdxl_lora",
    "illustrious_xl_v1_lora",
    "illustrious_xl_v2_lora",
    "kolors_lora",
    "sd3_5_large_lora",
    "sd3_5_medium_lora",
];

/// Insert the latent-perceptual support flags into a builtin target's `limits`.
pub(super) fn insert_limits(target: &mut TrainingTarget) {
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

/// Why `spec` cannot run for a `base_model` target + config even though the target advertises
/// it, or `None` — mirroring the engine's typed refusals: a full base fine-tune (aux losses train
/// through the adapter step only), and — for the decoded-x0 VAE anchor — an LTX-2.5 workflow that
/// generates no video (no x0 video latent to decode;
/// [`super::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS`]).
pub fn latent_loss_combination_refusal(
    spec: &LatentLossSpec,
    base_model: &str,
    config: &TrainingConfig,
) -> Option<String> {
    let label = capitalized(spec.label);
    if super::config_is_full_finetune(config) {
        return Some(format!(
            "{label} trains a LoRA/LoKr adapter only, not a full fine-tune."
        ));
    }
    if spec.weight_key == VAE_ANCHOR.weight_key && base_model == "ltx_2_5" {
        let workflow = config
            .advanced
            .get("ltxWorkflow")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if super::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS.contains(&workflow) {
            return Some(format!(
                "{label} decodes the generated video stream; the LTX-2.5 workflow '{workflow}' \
                 generates none."
            ));
        }
    }
    None
}

fn capitalized(label: &str) -> String {
    let mut c = label.chars();
    match c.next() {
        Some(first) => first.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
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
    match technique_value(advanced, key) {
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
    let weight = match technique_value(advanced, spec.weight_key) {
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
    let every = match technique_value(advanced, spec.every_key) {
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

/// Refuses an enabled technique on a target that does not advertise it, or on a combination the
/// engine refuses ([`latent_loss_combination_refusal`]) — a field error on the weight key at submit
/// time instead of a refusal after the job is queued.
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
        if latent_loss_settings(spec, &config.advanced)?.is_some() {
            if let Some(reason) = latent_loss_combination_refusal(spec, &target.base_model, config)
            {
                return Err(field_error(spec.weight_key, reason));
            }
        }
    }
    Ok(())
}
