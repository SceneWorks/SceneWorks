//! **Face losses** (epic 2123, sc-24831) — the product-side contract of the native trainer's two
//! decoded-x0 face losses: the ArcFace **identity loss** and the MediaPipe FaceMesh **landmark
//! loss**. Like depth anchoring ([`super::depth_anchoring`]): the `advanced` keys, their bounds
//! (the web form enforces the identical bounds — a parity test pins them, E6), the one strict
//! parser the API (submit-time field errors) and the worker (preflight + engine mapping) share,
//! the per-target support flags, and the auxiliary catalog models the losses load.
//!
//! The identity loss reuses the shipped `instantid_face_stack` bundle (SCRFD-10g detector + the
//! antelopev2 `glintr100` iresnet100 ArcFace) rather than upstream ai-toolkit-perceptual's
//! buffalo_l `w600k_r50` (iresnet50): it is the ArcFace SceneWorks already installs and pins, the
//! loss shape is identical (512-d embedding cosine), and the native loader reads any IResNet depth
//! from the checkpoint keys, so a converted `w600k_r50` would load unchanged.
//!
//! The landmark loss runs MediaPipe's FaceMesh-v2 from [`FACEMESH_V2_MODEL`] — upstream's
//! `py-feat/mp_facemesh_v2` onnx2torch checkpoint lowered to the native fx-program format (inference
//! `crates/media/mlx-gen/tools/convert_mp_facemesh_v2.py`) and rehosted at a pinned revision.

use serde_json::Value;

use super::depth_anchoring::AuxTrainingModel;
use super::{technique_value, TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// `advanced` key of the identity-loss weight. Absent or `0` is off.
pub const IDENTITY_LOSS_WEIGHT_KEY: &str = "identityLossWeight";
/// `advanced` key of the identity loss's inclusive lower noise-level bound.
pub const IDENTITY_LOSS_MIN_T_KEY: &str = "identityLossMinT";
/// `advanced` key of the identity loss's inclusive upper noise-level bound.
pub const IDENTITY_LOSS_MAX_T_KEY: &str = "identityLossMaxT";
/// `advanced` key of the identity loss's alternation period (`1` = added every step; `n ≥ 2` =
/// every n-th step is identity-only).
pub const IDENTITY_LOSS_EVERY_KEY: &str = "identityLossEvery";
/// `advanced` key of the identity gate: a step whose live cosine to the reference is at or below
/// it contributes no identity loss (no push on a hallucinated non-face).
pub const IDENTITY_LOSS_MIN_COS_KEY: &str = "identityLossMinCos";
/// `advanced` key of the reference embedding mode (`dataset_average` / `per_image`).
pub const IDENTITY_LOSS_REFERENCE_KEY: &str = "identityLossReference";

/// `advanced` key of the face-landmark loss weight. Absent or `0` is off.
pub const FACE_LANDMARK_LOSS_WEIGHT_KEY: &str = "faceLandmarkLossWeight";
/// `advanced` key of the landmark loss's inclusive lower noise-level bound.
pub const FACE_LANDMARK_LOSS_MIN_T_KEY: &str = "faceLandmarkLossMinT";
/// `advanced` key of the landmark loss's inclusive upper noise-level bound.
pub const FACE_LANDMARK_LOSS_MAX_T_KEY: &str = "faceLandmarkLossMaxT";
/// `advanced` key of the landmark loss's alternation period.
pub const FACE_LANDMARK_LOSS_EVERY_KEY: &str = "faceLandmarkLossEvery";

/// Inclusive upper bound of both face-loss weights (web: `faceLossWeightMax`). Upstream suggests
/// starting at 0.01–0.1.
pub const FACE_LOSS_WEIGHT_MAX: f64 = 1.0;
/// The weight the toggles seed (upstream's documented starting point; web:
/// `faceLossWeightSuggested`).
pub const FACE_LOSS_WEIGHT_SUGGESTED: f64 = 0.1;
/// Inclusive upper bound of both alternation periods (web: `faceLossEveryMax`).
pub const FACE_LOSS_EVERY_MAX: u64 = 16;
/// Default alternation period: strict diffusion / aux alternation (web: `faceLossEveryDefault`).
pub const FACE_LOSS_EVERY_DEFAULT: u32 = 2;
/// Default identity gate (upstream `identity_loss_min_cos`; web: `identityLossMinCosDefault`).
pub const IDENTITY_LOSS_MIN_COS_DEFAULT: f64 = 0.2;
/// The accepted [`IDENTITY_LOSS_REFERENCE_KEY`] values, default first (web:
/// `identityLossReferenceOptions`).
pub const IDENTITY_LOSS_REFERENCES: [&str; 2] = ["dataset_average", "per_image"];

/// Target `limits` flag: `true` when this target's native trainer on the serving platform honors
/// the identity loss (`TrainerDescriptor::techniques.identity_loss`). Same mechanism as
/// [`super::depth_anchoring::DEPTH_ANCHORING_SUPPORT_LIMIT`]: static per target and identical on
/// both platforms — set wherever the target's trainer resolves an x0 decoder and its descriptor
/// declares the technique (a worker test pins it to the active platform's descriptors); the web form
/// shows the control only where it is `true` and submit-time validation refuses a non-zero weight
/// elsewhere.
pub const IDENTITY_LOSS_SUPPORT_LIMIT: &str = "supportsIdentityLoss";
/// Target `limits` flag for the face-landmark loss — the same mechanism as
/// [`IDENTITY_LOSS_SUPPORT_LIMIT`] (`TrainerDescriptor::techniques.face_landmark_loss`).
pub const FACE_LANDMARK_LOSS_SUPPORT_LIMIT: &str = "supportsFaceLandmarkLoss";

fn flag(target: &TrainingTarget, key: &str) -> bool {
    target.limits.get(key).and_then(Value::as_bool) == Some(true)
}

/// Whether `target` (as projected for the serving platform) advertises the identity loss.
pub fn target_supports_identity_loss(target: &TrainingTarget) -> bool {
    flag(target, IDENTITY_LOSS_SUPPORT_LIMIT)
}

/// Whether `target` (as projected for the serving platform) advertises the face-landmark loss.
pub fn target_supports_face_landmark_loss(target: &TrainingTarget) -> bool {
    flag(target, FACE_LANDMARK_LOSS_SUPPORT_LIMIT)
}

/// The SCRFD-10g detector of the shipped `instantid_face_stack` bundle (both face losses detect the
/// reference face with it).
pub const FACE_STACK_SCRFD: AuxTrainingModel = AuxTrainingModel {
    id: "instantid_face_stack",
    label: "InstantID face analysis stack",
    repo: "SceneWorks/instantid-mlx",
    revision: "bca0cacf8e5e04529bb2b326a521361b02be84fd",
    file: "scrfd_10g.safetensors",
};

/// The glintr100 ArcFace of the same bundle (the identity loss's embedder).
pub const FACE_STACK_ARCFACE: AuxTrainingModel = AuxTrainingModel {
    file: "arcface_iresnet100.safetensors",
    ..FACE_STACK_SCRFD
};

/// The converted MediaPipe FaceMesh-v2 landmark detector (the landmark loss's model): upstream
/// `py-feat/mp_facemesh_v2` @ `39eb85054cf76fe0f57b7e12d6765ae89d89f2b5` (Apache-2.0), lowered by
/// `convert_mp_facemesh_v2.py` (self-check max abs 0.0 vs the torch module).
pub const FACEMESH_V2_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "mp_facemesh_v2",
    label: "MediaPipe FaceMesh v2",
    repo: "SceneWorks/mp-facemesh-v2",
    revision: "4e65de94fd007bde5eccd557bd83f067cbd3d596",
    file: "face_landmarks_detector.safetensors",
};

/// A validated, **enabled** aux-loss schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceLossSchedule {
    pub weight: f64,
    pub min_t: f64,
    pub max_t: f64,
    pub every: u32,
}

/// A validated, **enabled** identity-loss request.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityLossSettings {
    pub schedule: FaceLossSchedule,
    pub min_cos: f64,
    /// One of [`IDENTITY_LOSS_REFERENCES`].
    pub reference: &'static str,
}

fn field_error(field: &str, message: String) -> TrainingPlanError {
    TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message,
    }
}

fn bounded(
    advanced: &JsonObject,
    key: &str,
    (lo, hi): (f64, f64),
    default: f64,
) -> Result<f64, TrainingPlanError> {
    match technique_value(advanced, key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|v| v.is_finite() && (lo..=hi).contains(v))
            .ok_or_else(|| {
                field_error(
                    key,
                    format!("{key} must be a number between {lo} and {hi}."),
                )
            }),
    }
}

/// Strictly parse one loss's weight / window / period keys. Every present key is validated even
/// when the loss is off; `Ok(None)` means off (weight absent or `0`).
fn schedule(
    advanced: &JsonObject,
    [weight_key, min_key, max_key, every_key]: [&str; 4],
) -> Result<Option<FaceLossSchedule>, TrainingPlanError> {
    let weight = bounded(advanced, weight_key, (0.0, FACE_LOSS_WEIGHT_MAX), 0.0)?;
    let min_t = bounded(advanced, min_key, (0.0, 1.0), 0.0)?;
    let max_t = bounded(advanced, max_key, (0.0, 1.0), 1.0)?;
    if min_t > max_t {
        return Err(field_error(
            max_key,
            format!("{max_key} ({max_t}) must be at least {min_key} ({min_t})."),
        ));
    }
    let every = match technique_value(advanced, every_key) {
        None => FACE_LOSS_EVERY_DEFAULT,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=FACE_LOSS_EVERY_MAX).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| {
                field_error(
                    every_key,
                    format!(
                        "{every_key} must be a whole number between 1 and {FACE_LOSS_EVERY_MAX}."
                    ),
                )
            })?,
    };
    Ok((weight > 0.0).then_some(FaceLossSchedule {
        weight,
        min_t,
        max_t,
        every,
    }))
}

/// Strictly parse the identity-loss keys; `Ok(None)` = off. Failures are field errors naming the
/// offending key.
pub fn identity_loss_settings(
    advanced: &JsonObject,
) -> Result<Option<IdentityLossSettings>, TrainingPlanError> {
    let schedule = schedule(
        advanced,
        [
            IDENTITY_LOSS_WEIGHT_KEY,
            IDENTITY_LOSS_MIN_T_KEY,
            IDENTITY_LOSS_MAX_T_KEY,
            IDENTITY_LOSS_EVERY_KEY,
        ],
    )?;
    let min_cos = bounded(
        advanced,
        IDENTITY_LOSS_MIN_COS_KEY,
        (-1.0, 1.0),
        IDENTITY_LOSS_MIN_COS_DEFAULT,
    )?;
    let reference = match technique_value(advanced, IDENTITY_LOSS_REFERENCE_KEY) {
        None => IDENTITY_LOSS_REFERENCES[0],
        Some(value) => value
            .as_str()
            .map(|s| s.trim().to_ascii_lowercase())
            .and_then(|s| IDENTITY_LOSS_REFERENCES.iter().copied().find(|m| *m == s))
            .ok_or_else(|| {
                field_error(
                    IDENTITY_LOSS_REFERENCE_KEY,
                    format!(
                        "{IDENTITY_LOSS_REFERENCE_KEY} must be one of {}.",
                        IDENTITY_LOSS_REFERENCES.join(", ")
                    ),
                )
            })?,
    };
    Ok(schedule.map(|schedule| IdentityLossSettings {
        schedule,
        min_cos,
        reference,
    }))
}

/// Strictly parse the face-landmark keys; `Ok(None)` = off.
pub fn face_landmark_loss_settings(
    advanced: &JsonObject,
) -> Result<Option<FaceLossSchedule>, TrainingPlanError> {
    schedule(
        advanced,
        [
            FACE_LANDMARK_LOSS_WEIGHT_KEY,
            FACE_LANDMARK_LOSS_MIN_T_KEY,
            FACE_LANDMARK_LOSS_MAX_T_KEY,
            FACE_LANDMARK_LOSS_EVERY_KEY,
        ],
    )
}

/// Submit-time validation (called from the plan's config validation).
pub(super) fn validate(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    identity_loss_settings(&config.advanced)?;
    face_landmark_loss_settings(&config.advanced)?;
    Ok(())
}

/// Refuses an enabled face loss on a target that does not advertise it — a field error at submit
/// time instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    if identity_loss_settings(&config.advanced)?.is_some() {
        if !target_supports_identity_loss(target) {
            return Err(field_error(
                IDENTITY_LOSS_WEIGHT_KEY,
                format!(
                    "{} does not support the identity loss ({IDENTITY_LOSS_WEIGHT_KEY}) on this \
                     platform.",
                    target.name
                ),
            ));
        }
        if let Some(reason) =
            face_loss_combination_refusal(&target.base_model, config, "The identity loss")
        {
            return Err(field_error(IDENTITY_LOSS_WEIGHT_KEY, reason));
        }
    }
    if face_landmark_loss_settings(&config.advanced)?.is_some() {
        if !target_supports_face_landmark_loss(target) {
            return Err(field_error(
                FACE_LANDMARK_LOSS_WEIGHT_KEY,
                format!(
                    "{} does not support the face-landmark loss ({FACE_LANDMARK_LOSS_WEIGHT_KEY}) \
                     on this platform.",
                    target.name
                ),
            ));
        }
        if let Some(reason) =
            face_loss_combination_refusal(&target.base_model, config, "The face-landmark loss")
        {
            return Err(field_error(FACE_LANDMARK_LOSS_WEIGHT_KEY, reason));
        }
    }
    Ok(())
}

/// Why a face loss (`label`) cannot run for a `base_model` target + config even though the target
/// advertises it, or `None` — the same refusals the engine raises for every decoded-x0 loss and
/// [`super::depth_anchoring::depth_anchoring_combination_refusal`] mirrors for depth: a full base
/// fine-tune (aux losses train through the adapter step only), or an LTX-2.5 workflow that generates
/// no video ([`super::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS`]).
pub fn face_loss_combination_refusal(
    base_model: &str,
    config: &TrainingConfig,
    label: &str,
) -> Option<String> {
    if super::config_is_full_finetune(config) {
        return Some(format!(
            "{label} trains a LoRA/LoKr adapter only, not a full fine-tune."
        ));
    }
    if base_model == "ltx_2_5" {
        let workflow = config
            .advanced
            .get("ltxWorkflow")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if super::depth_anchoring::DEPTH_ANCHORING_NO_VIDEO_LTX_WORKFLOWS.contains(&workflow) {
            return Some(format!(
                "{label} needs a generated video stream; the LTX-2.5 workflow '{workflow}' \
                 generates none."
            ));
        }
    }
    None
}
