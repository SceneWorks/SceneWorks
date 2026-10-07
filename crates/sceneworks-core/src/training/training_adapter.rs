//! **Training adapter** (sc-25213) — the product-side contract of the frozen de-distill LoRA the
//! native Z-Image-Turbo trainer applies to the step-distilled base for training only (ai-toolkit's
//! `assistant_lora_path`): the `advanced` keys, the one strict parser the API (submit-time field
//! errors) and the worker (engine mapping) share, the per-target support flag, and the catalog
//! models each version resolves to.
//!
//! The Z-Image-Turbo presets set `trainingAdapterRepo` to ostris' `zimage_turbo_training_adapter`
//! and `trainingAdapterVersion` to `v2-default`; the Training Studio offers `v1` / `v2`. The worker
//! resolves the version to its installed `.safetensors` and hands the trainer the file
//! (`gen_core::TrainingConfig::training_adapter`), which trains against base + adapter and renders
//! previews / saves the LoRA without it.

use serde_json::Value;

use super::depth_anchoring::AuxTrainingModel;
use super::{technique_value, TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// `advanced` key naming the training adapter's Hugging Face repo. Absent or blank is off.
pub const TRAINING_ADAPTER_REPO_KEY: &str = "trainingAdapterRepo";
/// `advanced` key selecting the adapter version (`v1`, `v2`, or the presets' `v2-default`; blank
/// means the default, v2).
pub const TRAINING_ADAPTER_VERSION_KEY: &str = "trainingAdapterVersion";
/// Target `limits` flag: `true` when this target's native trainer honors the training adapter (its
/// `TrainerDescriptor::techniques.training_adapter`) — the Z-Image-Turbo trainer on MLX and Candle.
/// Absent = no; a worker test pins it to the linked trainer descriptors on each platform.
pub const TRAINING_ADAPTER_SUPPORT_LIMIT: &str = "supportsTrainingAdapter";

/// The one training-adapter repo the product ships (ostris' Z-Image-Turbo de-distill adapter).
pub const ZIMAGE_TURBO_TRAINING_ADAPTER_REPO: &str = "ostris/zimage_turbo_training_adapter";
/// The pinned revision both versions are cataloged at.
const ZIMAGE_TURBO_TRAINING_ADAPTER_REVISION: &str = "654cd1bf8b3589d9442cb1c5e35221879e1af4f3";

/// v1 of the de-distill adapter (rank 32, ~170 MB) — the stable one.
pub const ZIMAGE_TURBO_TRAINING_ADAPTER_V1: AuxTrainingModel = AuxTrainingModel {
    id: "zimage_turbo_training_adapter_v1",
    label: "Z-Image Turbo de-distill training adapter v1",
    repo: ZIMAGE_TURBO_TRAINING_ADAPTER_REPO,
    revision: ZIMAGE_TURBO_TRAINING_ADAPTER_REVISION,
    file: "zimage_turbo_training_adapter_v1.safetensors",
};

/// v2 of the de-distill adapter (rank 64, ~340 MB) — the presets' default.
pub const ZIMAGE_TURBO_TRAINING_ADAPTER_V2: AuxTrainingModel = AuxTrainingModel {
    id: "zimage_turbo_training_adapter_v2",
    label: "Z-Image Turbo de-distill training adapter v2",
    repo: ZIMAGE_TURBO_TRAINING_ADAPTER_REPO,
    revision: ZIMAGE_TURBO_TRAINING_ADAPTER_REVISION,
    file: "zimage_turbo_training_adapter_v2.safetensors",
};

/// Every cataloged training adapter (a parity test pins each to its `builtin.models.jsonc` entry).
pub const TRAINING_ADAPTER_MODELS: [AuxTrainingModel; 2] = [
    ZIMAGE_TURBO_TRAINING_ADAPTER_V1,
    ZIMAGE_TURBO_TRAINING_ADAPTER_V2,
];

fn field_error(field: &str, message: String) -> TrainingPlanError {
    TrainingPlanError::InvalidField {
        field: field.to_owned(),
        message,
    }
}

/// A key's string value, trimmed (`None` when absent, JSON `null`, or blank). A non-string value is
/// a field error naming the key.
fn string_value<'a>(
    advanced: &'a JsonObject,
    key: &str,
) -> Result<Option<&'a str>, TrainingPlanError> {
    match technique_value(advanced, key) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.trim()).filter(|value| !value.is_empty())),
        Some(_) => Err(field_error(key, format!("{key} must be a string."))),
    }
}

/// The catalog model a plan's training-adapter keys select, or `None` when off. Strict: a repo
/// other than [`ZIMAGE_TURBO_TRAINING_ADAPTER_REPO`], a version other than `v1` / `v2` /
/// `v2-default` (case-insensitive; blank = v2), or a version without a repo is a field error.
pub fn training_adapter_model(
    advanced: &JsonObject,
) -> Result<Option<&'static AuxTrainingModel>, TrainingPlanError> {
    let repo = string_value(advanced, TRAINING_ADAPTER_REPO_KEY)?;
    let version = string_value(advanced, TRAINING_ADAPTER_VERSION_KEY)?;
    let Some(repo) = repo else {
        return match version {
            None => Ok(None),
            Some(version) => Err(field_error(
                TRAINING_ADAPTER_VERSION_KEY,
                format!(
                    "{TRAINING_ADAPTER_VERSION_KEY} '{version}' needs a \
                     {TRAINING_ADAPTER_REPO_KEY}."
                ),
            )),
        };
    };
    if repo != ZIMAGE_TURBO_TRAINING_ADAPTER_REPO {
        return Err(field_error(
            TRAINING_ADAPTER_REPO_KEY,
            format!(
                "Unknown training adapter '{repo}'. Supported: {ZIMAGE_TURBO_TRAINING_ADAPTER_REPO}."
            ),
        ));
    }
    match version.map(str::to_ascii_lowercase).as_deref() {
        Some("v1") => Ok(Some(&ZIMAGE_TURBO_TRAINING_ADAPTER_V1)),
        None | Some("v2") | Some("v2-default") => Ok(Some(&ZIMAGE_TURBO_TRAINING_ADAPTER_V2)),
        Some(_) => Err(field_error(
            TRAINING_ADAPTER_VERSION_KEY,
            format!(
                "Unknown {TRAINING_ADAPTER_VERSION_KEY} '{}'. Supported: v1, v2.",
                version.unwrap_or_default()
            ),
        )),
    }
}

/// Whether `target` advertises training-adapter support.
pub fn target_supports_training_adapter(target: &TrainingTarget) -> bool {
    target
        .limits
        .get(TRAINING_ADAPTER_SUPPORT_LIMIT)
        .and_then(Value::as_bool)
        == Some(true)
}

/// Refuses malformed training-adapter keys, and a training adapter on a target that does not
/// advertise it — a field error at submit time instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    if training_adapter_model(&config.advanced)?.is_some()
        && !target_supports_training_adapter(target)
    {
        return Err(field_error(
            TRAINING_ADAPTER_REPO_KEY,
            format!(
                "{} does not support a training adapter ({TRAINING_ADAPTER_REPO_KEY}).",
                target.name
            ),
        ));
    }
    Ok(())
}
