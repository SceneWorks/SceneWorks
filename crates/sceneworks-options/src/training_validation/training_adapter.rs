//! **Training adapter** (sc-25213, SW-F369) — the product-side contract of the frozen de-distill
//! LoRA a native trainer applies to a step-distilled base for training only (ai-toolkit's
//! `assistant_lora_path`): the `advanced` keys, the one strict parser the API (submit-time field
//! errors) and the worker (engine mapping) share, the per-target support flag, and the catalog
//! models each version resolves to.
//!
//! Each adapter is bound to the targets of the distilled base it was trained for:
//!
//! * ostris' `zimage_turbo_training_adapter` for Z-Image-Turbo. Its presets set
//!   `trainingAdapterVersion` to `v2-default`; the Training Studio offers `v1` / `v2`.
//! * ostris' `minimax_h3_training_adapter` image adapter for the 6-step MiniMax-H3-Image
//!   (text-to-image and edit LoRA targets) — the assistant LoRA ai-toolkit's own
//!   `minimax_h3_image` training applies, and the presets' default. One version, `v1`.
//! * ShootTheSound's `Fizgig-MiniMax-H3-Image-Training-Adapter` for the same targets, kept
//!   selectable (`v1`) for runs and presets that name it.
//!
//! Both MiniMax-H3-Image adapters are Model Derivatives of MiniMax-H3-Image, so their catalog
//! entries carry that model's licence gate.
//!
//! The worker resolves the version to its installed `.safetensors` and hands the trainer the file
//! (`gen_core::TrainingConfig::training_adapter`), which trains against base + adapter (strength
//! 1.0) and renders previews / saves the LoRA without it.

use serde_json::Value;

use super::depth_anchoring::AuxTrainingModel;
use super::{technique_value, TrainingConfig, TrainingPlanError, TrainingTarget};
use crate::contracts::JsonObject;

/// `advanced` key naming the training adapter's Hugging Face repo. Absent or blank is off.
pub const TRAINING_ADAPTER_REPO_KEY: &str = "trainingAdapterRepo";
/// `advanced` key selecting the adapter version (one of the repo's
/// [`TrainingAdapterFamily::versions`] or a [`TrainingAdapterFamily::default_aliases`] entry; blank
/// means the repo's default).
pub const TRAINING_ADAPTER_VERSION_KEY: &str = "trainingAdapterVersion";
/// Target `limits` flag: `true` when this target's native trainer honors the training adapter (its
/// `TrainerDescriptor::techniques.training_adapter`) — the Z-Image-Turbo trainer on MLX and Candle,
/// and the MiniMax-H3-Image trainer. Absent = no; a worker test pins it to the linked trainer
/// descriptors on each platform.
pub const TRAINING_ADAPTER_SUPPORT_LIMIT: &str = "supportsTrainingAdapter";

/// ostris' Z-Image-Turbo de-distill adapter repo.
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

/// ShootTheSound's MiniMax-H3-Image de-distill adapter repo (SW-F369).
pub const MINIMAX_H3_IMAGE_TRAINING_ADAPTER_REPO: &str =
    "ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter";
/// The pinned revision the adapter is cataloged at.
const MINIMAX_H3_IMAGE_TRAINING_ADAPTER_REVISION: &str = "f1a4547596faeb98bae568dd28da296c49fbe0a1";

/// The Fizgig MiniMax-H3-Image de-distill adapter (rank 16, kohya keys, ~149 MB). Kept on at
/// strength 1.0 while training and off at inference, per its model card.
pub const MINIMAX_H3_IMAGE_TRAINING_ADAPTER_V1: AuxTrainingModel = AuxTrainingModel {
    id: "minimax_h3_image_training_adapter_v1",
    label: "MiniMax-H3-Image de-distill training adapter v1",
    repo: MINIMAX_H3_IMAGE_TRAINING_ADAPTER_REPO,
    revision: MINIMAX_H3_IMAGE_TRAINING_ADAPTER_REVISION,
    file: "fizgig_minimax_h3_image_training_adapter.safetensors",
};

/// ostris' MiniMax-H3 training-adapter repo: one assistant LoRA per H3 architecture (video, image,
/// ref2va, FastH3, P-H3-Image). Only its image file is cataloged.
pub const MINIMAX_H3_OSTRIS_TRAINING_ADAPTER_REPO: &str = "ostris/minimax_h3_training_adapter";
/// The pinned revision the image adapter is cataloged at.
const MINIMAX_H3_OSTRIS_TRAINING_ADAPTER_REVISION: &str =
    "e4e119caa391053286145e48ebbaf0e0c00293d3";

/// ostris' MiniMax-H3-Image assistant LoRA (rank 16 over the attention and MLP of the 50 DiT
/// blocks and the 2 token-refiner blocks, ComfyUI keys, ~155 MB) — the one ai-toolkit's
/// `minimax_h3_image` architecture trains with (`assistant_lora_path`), and the presets' default.
/// On at strength 1.0 while training, off at inference.
pub const MINIMAX_H3_IMAGE_OSTRIS_TRAINING_ADAPTER_V1: AuxTrainingModel = AuxTrainingModel {
    id: "minimax_h3_image_ostris_training_adapter_v1",
    label: "MiniMax-H3-Image de-distill training adapter v1 (ostris / ai-toolkit)",
    repo: MINIMAX_H3_OSTRIS_TRAINING_ADAPTER_REPO,
    revision: MINIMAX_H3_OSTRIS_TRAINING_ADAPTER_REVISION,
    file: "minimax_h3_image_training_adapter_v1.safetensors",
};

/// Every cataloged training adapter (a parity test pins each to its `builtin.models.jsonc` entry).
pub const TRAINING_ADAPTER_MODELS: [AuxTrainingModel; 4] = [
    ZIMAGE_TURBO_TRAINING_ADAPTER_V1,
    ZIMAGE_TURBO_TRAINING_ADAPTER_V2,
    MINIMAX_H3_IMAGE_TRAINING_ADAPTER_V1,
    MINIMAX_H3_IMAGE_OSTRIS_TRAINING_ADAPTER_V1,
];

/// One shipped training-adapter repo: the versions it offers, the version a blank selection means,
/// and the training targets whose base it was distilled against. An adapter on any other target is
/// refused at submit — a de-distill LoRA for one base is noise on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingAdapterFamily {
    /// Hugging Face repo — the `trainingAdapterRepo` value that selects this family.
    pub repo: &'static str,
    /// `(version, model)` pairs the version key may name (matched case-insensitively).
    pub versions: &'static [(&'static str, AuxTrainingModel)],
    /// The version a blank / absent `trainingAdapterVersion` selects.
    pub default_version: &'static str,
    /// Legacy spellings that also select [`Self::default_version`] (Z-Image's `v2-default`).
    pub default_aliases: &'static [&'static str],
    /// Training target ids this adapter applies to.
    pub target_ids: &'static [&'static str],
}

/// Every shipped training-adapter repo.
pub static TRAINING_ADAPTER_FAMILIES: [TrainingAdapterFamily; 3] = [
    TrainingAdapterFamily {
        repo: ZIMAGE_TURBO_TRAINING_ADAPTER_REPO,
        versions: &[
            ("v1", ZIMAGE_TURBO_TRAINING_ADAPTER_V1),
            ("v2", ZIMAGE_TURBO_TRAINING_ADAPTER_V2),
        ],
        default_version: "v2",
        default_aliases: &["v2-default"],
        target_ids: &["z_image_turbo_lora"],
    },
    TrainingAdapterFamily {
        repo: MINIMAX_H3_IMAGE_TRAINING_ADAPTER_REPO,
        versions: &[("v1", MINIMAX_H3_IMAGE_TRAINING_ADAPTER_V1)],
        default_version: "v1",
        default_aliases: &[],
        target_ids: &["minimax_h3_image_lora", "minimax_h3_image_edit_lora"],
    },
    TrainingAdapterFamily {
        repo: MINIMAX_H3_OSTRIS_TRAINING_ADAPTER_REPO,
        versions: &[("v1", MINIMAX_H3_IMAGE_OSTRIS_TRAINING_ADAPTER_V1)],
        default_version: "v1",
        default_aliases: &[],
        target_ids: &["minimax_h3_image_lora", "minimax_h3_image_edit_lora"],
    },
];

/// Every training target some shipped adapter applies to — the targets whose catalog row carries
/// [`TRAINING_ADAPTER_SUPPORT_LIMIT`] (a test pins this to the union of the families' targets).
pub const TRAINING_ADAPTER_TARGETS: [&str; 3] = [
    "z_image_turbo_lora",
    "minimax_h3_image_lora",
    "minimax_h3_image_edit_lora",
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

/// A selected adapter: the family its repo names and the catalog model its version resolves to.
pub type TrainingAdapterSelection = (&'static TrainingAdapterFamily, &'static AuxTrainingModel);

/// The adapter family and catalog model a plan's training-adapter keys select, or `None` when off.
/// Strict: a repo outside [`TRAINING_ADAPTER_FAMILIES`], a version the repo does not offer
/// (case-insensitive; blank = the repo's default), or a version without a repo is a field error.
pub fn training_adapter_selection(
    advanced: &JsonObject,
) -> Result<Option<TrainingAdapterSelection>, TrainingPlanError> {
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
    let Some(family) = TRAINING_ADAPTER_FAMILIES
        .iter()
        .find(|family| family.repo == repo)
    else {
        let supported = TRAINING_ADAPTER_FAMILIES
            .iter()
            .map(|family| family.repo)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(field_error(
            TRAINING_ADAPTER_REPO_KEY,
            format!("Unknown training adapter '{repo}'. Supported: {supported}."),
        ));
    };
    let wanted = match version.map(str::to_ascii_lowercase) {
        None => family.default_version.to_owned(),
        Some(version) if family.default_aliases.contains(&version.as_str()) => {
            family.default_version.to_owned()
        }
        Some(version) => version,
    };
    match family.versions.iter().find(|(name, _)| *name == wanted) {
        Some((_, model)) => Ok(Some((family, model))),
        None => {
            let supported = family
                .versions
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ");
            Err(field_error(
                TRAINING_ADAPTER_VERSION_KEY,
                format!(
                    "Unknown {TRAINING_ADAPTER_VERSION_KEY} '{}' for {repo}. Supported: \
                     {supported}.",
                    version.unwrap_or_default()
                ),
            ))
        }
    }
}

/// The catalog model a plan's training-adapter keys select, or `None` when off. See
/// [`training_adapter_selection`] for the strict parse.
pub fn training_adapter_model(
    advanced: &JsonObject,
) -> Result<Option<&'static AuxTrainingModel>, TrainingPlanError> {
    Ok(training_adapter_selection(advanced)?.map(|(_, model)| model))
}

/// Whether `target` advertises training-adapter support.
pub fn target_supports_training_adapter(target: &TrainingTarget) -> bool {
    target
        .limits
        .get(TRAINING_ADAPTER_SUPPORT_LIMIT)
        .and_then(Value::as_bool)
        == Some(true)
}

/// Refuses malformed training-adapter keys, a training adapter on a target that does not advertise
/// it, and an adapter distilled for a different base than the target's — a field error at submit
/// time instead of a refusal after the job is queued.
pub(super) fn validate_support(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    let Some((family, _)) = training_adapter_selection(&config.advanced)? else {
        return Ok(());
    };
    if !target_supports_training_adapter(target) {
        return Err(field_error(
            TRAINING_ADAPTER_REPO_KEY,
            format!(
                "{} does not support a training adapter ({TRAINING_ADAPTER_REPO_KEY}).",
                target.name
            ),
        ));
    }
    if !family.target_ids.contains(&target.id.as_str()) {
        return Err(field_error(
            TRAINING_ADAPTER_REPO_KEY,
            format!(
                "Training adapter '{}' is not for {}: it de-distills a different base model.",
                family.repo, target.name
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The support-flag target list is exactly the union of the families' targets, every family
    /// version is cataloged, and every family's default resolves to one of its own versions.
    #[test]
    fn families_are_self_consistent() {
        let mut union: Vec<&str> = TRAINING_ADAPTER_FAMILIES
            .iter()
            .flat_map(|family| family.target_ids.iter().copied())
            .collect();
        union.sort_unstable();
        // Two adapters (ostris', Fizgig's) share the MiniMax-H3-Image targets.
        union.dedup();
        let mut targets = TRAINING_ADAPTER_TARGETS.to_vec();
        targets.sort_unstable();
        assert_eq!(union, targets);
        let mut models: Vec<&str> = TRAINING_ADAPTER_FAMILIES
            .iter()
            .flat_map(|family| family.versions.iter().map(|(_, model)| model.id))
            .collect();
        models.sort_unstable();
        let mut cataloged: Vec<&str> = TRAINING_ADAPTER_MODELS.iter().map(|m| m.id).collect();
        cataloged.sort_unstable();
        assert_eq!(models, cataloged);
        for family in &TRAINING_ADAPTER_FAMILIES {
            assert!(family
                .versions
                .iter()
                .any(|(name, _)| *name == family.default_version));
            for (_, model) in family.versions {
                assert_eq!(model.repo, family.repo);
            }
        }
    }
}
