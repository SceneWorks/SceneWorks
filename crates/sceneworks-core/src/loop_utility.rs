//! Shared identity and artifact selection for the Loop-9B utility.
use serde_json::Value;
use std::path::Path;

pub const MODEL_ID: &str = "vision_sensenova_loop_9b";
pub const MODEL_REPO: &str = "sensenova/SenseNova-Loop-9B";
pub const MODEL_NAME: &str = "SenseNova-Loop-9B";
pub const CAPTIONER: &str = "sensenova_loop";
pub const TIERS: [&str; 3] = ["bf16", "q8", "q4"];

pub fn tier(value: Option<&str>) -> Result<&'static str, String> {
    TIERS
        .into_iter()
        .find(|tier| Some(*tier) == value.map(str::trim))
        .ok_or_else(|| format!("Select a {MODEL_NAME} tier: {}.", TIERS.join(", ")))
}

/// A tier is available only when exactly one platform-compatible primary download declares it.
pub fn download<'a>(entry: &'a Value, platform: &str, tier: &str) -> Option<&'a Value> {
    use crate::model_artifacts::artifact_selection::{
        is_co_requisite_download, is_supported_model_download,
    };
    if entry.get("id")?.as_str()? != MODEL_ID || !TIERS.contains(&tier) {
        return None;
    }
    let mut rows = entry.get("downloads")?.as_array()?.iter().filter(|row| {
        is_supported_model_download(row)
            && !is_co_requisite_download(row)
            && row.get("variant").and_then(Value::as_str) == Some(tier)
            && row
                .get("platforms")
                .and_then(Value::as_array)
                .is_none_or(|platforms| {
                    platforms.iter().any(|item| item.as_str() == Some(platform))
                })
    });
    let row = rows.next()?;
    if rows.next().is_some() {
        return None;
    }
    let revision = row.get("revision")?.as_str()?;
    if revision == "0000000000000000000000000000000000000000"
        || crate::model_artifacts::validate_immutable_revision(revision).is_err()
        || enumerated_files(row).is_none()
    {
        return None;
    }
    Some(row)
}

/// Loop artifacts list every file. A glob or unsafe selector cannot establish completeness.
pub fn enumerated_files(row: &Value) -> Option<Vec<&str>> {
    let files: Vec<_> = row
        .get("files")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<_>>()?;
    (!files.is_empty()
        && files.iter().all(|path| {
            !path.contains(['*', '?', '[', ']'])
                && crate::safe_path::relative_normal_components(path).is_ok()
        }))
    .then_some(files)
}

/// Stat the exact pinned snapshot's declared files. Portable admission and the worker source
/// guard authenticate their bytes against the publisher ledger before execution.
pub fn missing_snapshot_files(row: &Value, snapshot: &Path) -> Option<Vec<String>> {
    Some(
        enumerated_files(row)?
            .into_iter()
            .filter(|file| {
                !snapshot
                    .join(file)
                    .metadata()
                    .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
            })
            .map(str::to_owned)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn loop_variant_selection_requires_one_published_exact_platform_tier() {
        let entry = json!({
            "id": MODEL_ID,
            "downloads": [
                {"provider": "huggingface", "repo": MODEL_REPO, "variant": "bf16",
                 "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                 "files": ["config.json"], "platforms": ["macos"]},
                {"provider": "huggingface", "repo": "SceneWorks/fixture-loop-mlx", "variant": "q4",
                 "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                 "files": ["q4/config.json"], "platforms": ["macos"]},
                {"provider": "huggingface", "repo": "SceneWorks/fixture-loop-cuda", "variant": "q4",
                 "revision": "0000000000000000000000000000000000000000",
                 "files": ["q4/config.json"], "platforms": ["windows", "linux"]}
            ]
        });
        assert_eq!(tier(Some("q4")), Ok("q4"));
        assert!(tier(None).is_err());
        assert!(tier(Some("q2")).is_err());
        assert_eq!(
            download(&entry, "macos", "q4").unwrap()["repo"],
            "SceneWorks/fixture-loop-mlx"
        );
        assert!(download(&entry, "macos", "q8").is_none());
        assert!(download(&entry, "linux", "q4").is_none());
        assert!(download(&entry, "windows", "bf16").is_none());
    }
}
