//! Resolve the selected Loop-9B tier from the same server-owned manifest as portable admission.
use crate::{JsonObject, Settings, WorkerError, WorkerResult};
use sceneworks_core::loop_utility;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// All Loop utility paths use the same vision-capable provider and resident snapshot. Only JSON
/// tasks apply a constraint to the decoding request itself.
pub(crate) fn requirements() -> gen_core::core_llm::ModelRequirements {
    use gen_core::core_llm::{Constraint, ModelRequirements};
    ModelRequirements::default()
        .with_vision()
        .with_constraint(Constraint::Json)
}

pub(crate) fn weights_dir(settings: &Settings, payload: &JsonObject) -> WorkerResult<PathBuf> {
    let tier = loop_utility::tier(payload.get("variant").and_then(Value::as_str))
        .map_err(WorkerError::InvalidPayload)?;
    let missing = || {
        WorkerError::InvalidPayload(format!(
        "The {tier} tier of {} is unavailable or incomplete. Download that exact tier in the Model Manager.",
        loop_utility::MODEL_NAME
    ))
    };
    let entry = payload
        .get("modelManifestEntries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(loop_utility::MODEL_ID))
        .ok_or_else(missing)?;
    let row = loop_utility::download(entry, std::env::consts::OS, tier).ok_or_else(missing)?;
    let repo = row
        .get("repo")
        .and_then(Value::as_str)
        .ok_or_else(missing)?;
    let revision = row
        .get("revision")
        .and_then(Value::as_str)
        .ok_or_else(missing)?;
    let snapshot = crate::huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
        .ok_or_else(missing)?;
    let weights = weights_directory(row, &snapshot).ok_or_else(missing)?;

    let backend = if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    };
    let Some(needed) = entry
        .get(backend)
        .and_then(|lane| lane.get("minMemoryGbByTier"))
        .and_then(|tiers| tiers.get(tier))
        .and_then(Value::as_f64)
    else {
        return Ok(weights);
    };
    #[cfg(target_os = "macos")]
    let available = crate::fit_gate::unified_total_gb();
    #[cfg(not(target_os = "macos"))]
    let available = crate::vram_gate::live_capped_budget_fresh_blocking(&settings.gpu_id)
        .map(|budget| budget.free_gb);
    if let crate::fit_gate::FitDecision::TooBig {
        needed_gb,
        available_gb,
    } = crate::fit_gate::fit_decision(Some(needed), available)
    {
        return Err(WorkerError::InvalidPayload(format!(
            "{} {tier} needs {needed_gb:.1} GB of memory, but this worker has {available_gb:.1} GB available. Select a smaller installed tier.",
            loop_utility::MODEL_NAME
        )));
    }
    Ok(weights)
}

fn weights_directory(row: &Value, snapshot: &Path) -> Option<PathBuf> {
    let files: Vec<_> = row
        .get("files")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?;
    let mut configs = files.iter().copied().filter(|file| {
        Path::new(file).file_name().and_then(|name| name.to_str()) == Some("config.json")
    });
    let config = configs.next()?;
    if configs.next().is_some() {
        return None;
    }
    let weights = crate::paths::safe_join(snapshot, config)
        .ok()?
        .parent()?
        .to_path_buf();
    for file in files {
        let file_path = crate::paths::safe_join(snapshot, file).ok()?;
        if file_path.file_name()?.to_str()? == "model-*.safetensors" {
            let directory = file_path.parent()?;
            let index =
                std::fs::read_to_string(directory.join("model.safetensors.index.json")).ok()?;
            let index: Value = serde_json::from_str(&index).ok()?;
            let shards = index.get("weight_map")?.as_object()?;
            if shards.is_empty()
                || !shards.values().all(|shard| {
                    shard
                        .as_str()
                        .and_then(|file| crate::paths::safe_join(directory, file).ok())
                        .is_some_and(|file| file.is_file())
                })
            {
                return None;
            }
        } else if !file_path.is_file() {
            return None;
        }
    }
    Some(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tier_folder_requires_every_file_in_its_own_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let row = json!({"files": ["q4/config.json", "q4/model.safetensors"]});
        let selected = temp.path().join("q4");
        std::fs::create_dir(&selected).unwrap();
        std::fs::write(selected.join("config.json"), b"{}").unwrap();
        std::fs::write(temp.path().join("model.safetensors"), b"other tier").unwrap();
        assert_eq!(weights_directory(&row, temp.path()), None);
        std::fs::write(selected.join("model.safetensors"), b"fixture").unwrap();
        assert_eq!(weights_directory(&row, temp.path()), Some(selected));
    }
}
