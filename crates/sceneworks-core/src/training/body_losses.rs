//! **Body losses** (epic 2123, sc-24832) — the product-side contract of the native trainer's three
//! decoded-x0 body perceptual losses: ViTPose+ bone-length **proportions**, HybrIK SMPL-beta
//! **shape** and Sapiens surface **normals**. The `advanced` keys, their bounds (the web form
//! enforces the identical bounds — a parity test pins them, E6), the one strict parser the API
//! (submit-time field errors) and the worker (preflight + engine mapping) share, the per-target
//! support flags, and the auxiliary catalog models each loss needs installed.
//!
//! All three losses share one reference-time person detector (ViTPose), so the ViTPose checkpoint
//! is needed whenever any of them is on.

use serde_json::Value;

use super::depth_anchoring::AuxTrainingModel;
use super::{TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// One of the three body losses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyLoss {
    /// ViTPose bone-length-ratio loss.
    Proportion,
    /// HybrIK SMPL-beta loss.
    Shape,
    /// Sapiens surface-normal loss.
    Normal,
}

impl BodyLoss {
    /// Every body loss, in engine arm order.
    pub const ALL: [BodyLoss; 3] = [BodyLoss::Proportion, BodyLoss::Shape, BodyLoss::Normal];

    /// The `advanced` key prefix (`bodyProportion…`, `bodyShape…`, `normal…`).
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Proportion => "bodyProportion",
            Self::Shape => "bodyShape",
            Self::Normal => "normal",
        }
    }

    /// Human name for messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Proportion => "body proportion loss",
            Self::Shape => "body shape loss",
            Self::Normal => "normal loss",
        }
    }

    /// The target `limits` flag advertising this loss (its trainer descriptor's technique flag on
    /// the serving platform). Absent = no.
    pub fn support_limit(self) -> &'static str {
        match self {
            Self::Proportion => BODY_PROPORTION_SUPPORT_LIMIT,
            Self::Shape => BODY_SHAPE_SUPPORT_LIMIT,
            Self::Normal => NORMAL_SUPPORT_LIMIT,
        }
    }

    /// The `advanced` key of this loss's weight (absent or `0` = off).
    pub fn weight_key(self) -> String {
        format!("{}Weight", self.prefix())
    }

    /// The default noise-level window (upstream: proportion `[0, 1]`, shape and normal
    /// `[0.4, 0.8]`).
    pub fn default_window(self) -> (f64, f64) {
        match self {
            Self::Proportion => (0.0, 1.0),
            Self::Shape | Self::Normal => (0.4, 0.8),
        }
    }

    /// The frozen model this loss needs installed besides ViTPose — `None` for the proportion
    /// loss (ViTPose is its model).
    pub fn own_model(self) -> Option<&'static AuxTrainingModel> {
        match self {
            Self::Proportion => None,
            Self::Shape => HYBRIK_MODEL,
            Self::Normal => SAPIENS_NORMAL_MODEL,
        }
    }

    /// Whether every checkpoint this loss loads is in the shipped catalog (ViTPose plus its own
    /// model). A loss whose weights are not cataloged cannot be installed, so no target
    /// advertises it.
    pub fn weights_cataloged(self) -> bool {
        match self {
            Self::Proportion => true,
            Self::Shape | Self::Normal => self.own_model().is_some(),
        }
    }
}

/// Target `limits` flag of the proportion loss (`techniques.body_proportion_loss`).
pub const BODY_PROPORTION_SUPPORT_LIMIT: &str = "supportsBodyProportionLoss";
/// Target `limits` flag of the shape loss (`techniques.body_shape_loss`).
pub const BODY_SHAPE_SUPPORT_LIMIT: &str = "supportsBodyShapeLoss";
/// Target `limits` flag of the normal loss (`techniques.normal_loss`).
pub const NORMAL_SUPPORT_LIMIT: &str = "supportsNormalLoss";

/// `advanced` key: add the two head ratios to the proportion loss (bool, default `false`).
pub const BODY_PROPORTION_INCLUDE_HEAD_KEY: &str = "bodyProportionIncludeHead";
/// `advanced` key: the shape loss only counts while the live betas' cosine to the reference
/// exceeds this (default [`BODY_SHAPE_MIN_COS_DEFAULT`]).
pub const BODY_SHAPE_MIN_COS_KEY: &str = "bodyShapeMinCos";
/// `advanced` key: average the normal loss over the item's subject mask only (bool, default
/// `false`; needs every image's subject mask).
pub const NORMAL_RESTRICT_TO_SUBJECT_KEY: &str = "normalRestrictToSubject";

/// Inclusive upper bound of every body-loss weight (web: `bodyLossWeightMax`). Upstream suggests
/// 0.1–0.2 for the proportion loss; ten times its suggestion is a typo, not a choice.
pub const BODY_LOSS_WEIGHT_MAX: f64 = 1.0;
/// The weight the toggle seeds (web: `bodyLossWeightSuggested`) — upstream's "try 0.1–0.2 for
/// full-body datasets" for the proportion loss; upstream ships no suggestion for shape/normal, so
/// the same conservative value.
pub const BODY_LOSS_WEIGHT_SUGGESTED: f64 = 0.1;
/// Inclusive upper bound of every body-loss alternation period (web: `bodyLossEveryMax`).
pub const BODY_LOSS_EVERY_MAX: u64 = 16;
/// Default alternation period: strict diffusion / aux alternation (upstream default).
pub const BODY_LOSS_EVERY_DEFAULT: u32 = 2;
/// Upstream `body_shape_loss_min_cos` (web: `bodyShapeMinCosDefault`).
pub const BODY_SHAPE_MIN_COS_DEFAULT: f64 = 0.2;

/// ViTPose+ base — the body-proportion encoder and every body loss's person detector.
pub const VITPOSE_PLUS_BASE_MODEL: AuxTrainingModel = AuxTrainingModel {
    id: "vitpose_plus_base",
    label: "ViTPose+ Base",
    repo: "usyd-community/vitpose-plus-base",
    revision: "92be54d7a29e42fad47b6e2ca01dd9e685a61e0d",
    file: "model.safetensors",
};

/// HybrIK ResNet-34 shape encoder. Upstream ships it only as a Google-Drive `.pth`; it is
/// cataloged once its safetensors re-host (`scripts/reference/body_losses_reference.py --convert`
/// in the inference repo) is published, until then `None` and the shape loss is advertised by no
/// target.
pub const HYBRIK_MODEL: Option<&AuxTrainingModel> = None;
/// Sapiens-0.3B normal estimator. Upstream (`facebook/sapiens-normal-0.3b`) ships only a pickled
/// `.pth`; cataloged once its safetensors re-host is published (as [`HYBRIK_MODEL`]).
pub const SAPIENS_NORMAL_MODEL: Option<&AuxTrainingModel> = None;

/// Whether `target` (as projected for the serving platform) advertises `loss`.
pub fn target_supports(target: &TrainingTarget, loss: BodyLoss) -> bool {
    target
        .limits
        .get(loss.support_limit())
        .and_then(Value::as_bool)
        == Some(true)
}

/// One validated, **enabled** body loss.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyLossSchedule {
    pub weight: f64,
    pub min_t: f64,
    pub max_t: f64,
    pub every: u32,
}

/// The validated body-loss request (each `None` = off).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyLossSettings {
    pub proportion: Option<BodyLossSchedule>,
    pub include_head: bool,
    pub shape: Option<BodyLossSchedule>,
    pub shape_min_cos: f64,
    pub normal: Option<BodyLossSchedule>,
    pub normal_restrict_to_subject: bool,
}

impl BodyLossSettings {
    /// The schedule of `loss` (`None` = off).
    pub fn schedule(&self, loss: BodyLoss) -> Option<&BodyLossSchedule> {
        match loss {
            BodyLoss::Proportion => self.proportion.as_ref(),
            BodyLoss::Shape => self.shape.as_ref(),
            BodyLoss::Normal => self.normal.as_ref(),
        }
    }

    /// Whether any body loss is on.
    pub fn any_enabled(&self) -> bool {
        BodyLoss::ALL.iter().any(|&l| self.schedule(l).is_some())
    }
}

fn field_error(field: &str, message: String) -> TrainingPlanError {
    TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message,
    }
}

fn number_in(
    advanced: &JsonObject,
    key: &str,
    default: f64,
    lo: f64,
    hi: f64,
) -> Result<f64, TrainingPlanError> {
    match advanced.get(key) {
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

fn flag(advanced: &JsonObject, key: &str) -> Result<bool, TrainingPlanError> {
    match advanced.get(key) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| field_error(key, format!("{key} must be true or false."))),
    }
}

fn schedule(
    advanced: &JsonObject,
    loss: BodyLoss,
) -> Result<Option<BodyLossSchedule>, TrainingPlanError> {
    let p = loss.prefix();
    let weight = number_in(advanced, &loss.weight_key(), 0.0, 0.0, BODY_LOSS_WEIGHT_MAX)?;
    let (min_default, max_default) = loss.default_window();
    let min_key = format!("{p}MinT");
    let max_key = format!("{p}MaxT");
    let min_t = number_in(advanced, &min_key, min_default, 0.0, 1.0)?;
    let max_t = number_in(advanced, &max_key, max_default, 0.0, 1.0)?;
    if min_t > max_t {
        return Err(field_error(
            &max_key,
            format!("{max_key} ({max_t}) must be at least {min_key} ({min_t})."),
        ));
    }
    let every_key = format!("{p}Every");
    let every = match advanced.get(&every_key) {
        None => BODY_LOSS_EVERY_DEFAULT,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=BODY_LOSS_EVERY_MAX).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| {
                field_error(
                    &every_key,
                    format!(
                        "{every_key} must be a whole number between 1 and {BODY_LOSS_EVERY_MAX}."
                    ),
                )
            })?,
    };
    Ok((weight > 0.0).then_some(BodyLossSchedule {
        weight,
        min_t,
        max_t,
        every,
    }))
}

/// Strictly parse the body-loss keys of a plan's `advanced` bag. Every present key is validated
/// (type and range) even when its loss is off. Each failure is a
/// [`TrainingPlanError::InvalidField`] naming the offending key.
pub fn body_loss_settings(advanced: &JsonObject) -> Result<BodyLossSettings, TrainingPlanError> {
    Ok(BodyLossSettings {
        proportion: schedule(advanced, BodyLoss::Proportion)?,
        include_head: flag(advanced, BODY_PROPORTION_INCLUDE_HEAD_KEY)?,
        shape: schedule(advanced, BodyLoss::Shape)?,
        shape_min_cos: number_in(
            advanced,
            BODY_SHAPE_MIN_COS_KEY,
            BODY_SHAPE_MIN_COS_DEFAULT,
            -1.0,
            1.0,
        )?,
        normal: schedule(advanced, BodyLoss::Normal)?,
        normal_restrict_to_subject: flag(advanced, NORMAL_RESTRICT_TO_SUBJECT_KEY)?,
    })
}

/// Submit-time validation (called from the plan's config validation).
pub(super) fn validate(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    body_loss_settings(&config.advanced).map(|_| ())
}

/// Refuses an enabled body loss on a target that does not advertise it — a `<loss>Weight` field
/// error at submit time instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    let settings = body_loss_settings(&config.advanced)?;
    for loss in BodyLoss::ALL {
        if settings.schedule(loss).is_some() && !target_supports(target, loss) {
            let key = loss.weight_key();
            return Err(field_error(
                &key,
                format!(
                    "{} does not support the {} ({key}) on this platform.",
                    target.name,
                    loss.label()
                ),
            ));
        }
    }
    Ok(())
}
