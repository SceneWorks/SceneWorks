//! Adapter selection shared by image/video and their attribution boundaries.
use crate::image_jobs::{declared_adapter_file, lora_path};
use crate::{Settings, WorkerError, WorkerResult};
use serde_json::Value;
use std::path::PathBuf;

pub(crate) fn resolve_adapter_file(
    lora: &Value,
    settings: &Settings,
    model: &str,
) -> WorkerResult<PathBuf> {
    let candidate = selected_adapter_path(lora, settings, model)?;
    crate::normalize_app_managed_lora_path(settings, &candidate)
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn resolve_prepared_adapter_file(
    lora: &Value,
    settings: &Settings,
    model: &str,
) -> WorkerResult<gen_core::PinnedWeightsFile> {
    let candidate = selected_adapter_path(lora, settings, model)?;
    crate::paths::pin_app_managed_model_file(settings, &candidate, "LoRA file")
}

fn selected_adapter_path(lora: &Value, settings: &Settings, model: &str) -> WorkerResult<PathBuf> {
    let raw = lora_path(lora)
        .ok_or_else(|| WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned()))?;
    let confined = crate::normalize_app_managed_lora_path(settings, &raw)?;
    let candidate = sceneworks_core::lora_family::resolve_adapter_for_model(
        &confined,
        declared_adapter_file(lora),
        model,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "LoRA selected file is missing, invalid or outside its adapter root: {}",
            confined.display()
        ))
    })?;
    // Legacy direct-file routes have always owned the original lexical entry, so a symlink
    // retarget between selection and dispatch must still invalidate their prepared token.
    // Confinement and shared selection above remain mandatory before retaining that entry.
    let candidate = if !matches!(model, "flux3_klein" | "flux3_klein_base") && !confined.is_dir() {
        raw
    } else {
        candidate
    };
    Ok(candidate)
}

pub(crate) fn lora_weight(lora: &Value) -> f64 {
    lora.get("weight")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        // Last resort only — the API always stamps a weight onto the spec. Matches the API/web
        // default (`DEFAULT_LORA_WEIGHT`) so a weightless payload can't apply at a scale no
        // surface would have shown.
        .unwrap_or(1.0)
}

/// Repeated selections must retain one identity, never hide a changed file behind an older token.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn retain_adapter_pin(
    pins: &mut std::collections::BTreeMap<PathBuf, gen_core::PinnedWeightsFile>,
    pin: gen_core::PinnedWeightsFile,
) -> WorkerResult<&gen_core::PinnedWeightsFile> {
    use std::collections::btree_map::Entry;
    match pins.entry(pin.loader_path().to_path_buf()) {
        Entry::Vacant(entry) => Ok(entry.insert(pin)),
        Entry::Occupied(entry) => {
            if entry.get() != &pin {
                return Err(WorkerError::InvalidPayload(
                    "Adapter source changed between repeated selections".into(),
                ));
            }
            Ok(entry.into_mut())
        }
    }
}
