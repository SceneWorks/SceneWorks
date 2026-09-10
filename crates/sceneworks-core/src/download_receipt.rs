//! Shared installation-receipt identity and write coordination.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Component, Path};

/// Serialize receipt read/modify/write across discovery and the download worker.
pub fn lock(managed: &Path) -> io::Result<crate::file_lock::FileLock> {
    crate::file_lock::FileLock::exclusive(
        OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(managed.join(".sceneworks-download-receipt.lock"))?,
    )
}

/// Publish a complete receipt while holding [`lock`].
pub fn write(marker: &Path, receipt: &Value) -> io::Result<()> {
    crate::store_util::atomic_write(marker, &serde_json::to_vec_pretty(receipt)?)
        .map_err(io::Error::other)
}

/// Missing, JSON null, and blank strings all represent an unrecorded revision.
pub fn revision_missing(receipt: &Value) -> bool {
    match receipt.get("snapshotRevision") {
        None | Some(Value::Null) => true,
        Some(Value::String(value)) => value.trim().is_empty(),
        _ => false,
    }
}

/// Recover identity only from one immutable snapshot holding every recorded file. Existing
/// tree stamps must still match; this never establishes or replaces an integrity baseline.
pub fn recover_revision(receipt: &Value, repo_root: &Path) -> Option<String> {
    if !revision_missing(receipt) {
        return None;
    }
    let files = receipt
        .get("resolvedFiles")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()?;
    if files.is_empty()
        || files.iter().any(|file| {
            file.is_empty()
                || file.contains('\\')
                || Path::new(file)
                    .components()
                    .any(|part| !matches!(part, Component::Normal(_)))
        })
    {
        return None;
    }
    let entries = fs::read_dir(repo_root.join("snapshots")).ok()?;
    let mut matching = Vec::new();
    for entry in entries {
        let entry = entry.ok()?;
        let revision = entry.file_name().into_string().ok()?;
        if crate::model_artifacts::validate_immutable_revision(&revision).is_err()
            || !entry.file_type().ok()?.is_dir()
        {
            continue;
        }
        let snapshot = entry.path();
        if files.iter().all(|file| snapshot.join(file).is_file()) {
            matching.push((revision, snapshot));
        }
    }
    let [(revision, snapshot)] = matching.as_slice() else {
        return None;
    };
    // Promotion copies only resolvedFiles. A complete external tree is not enough if an older
    // receipt omitted a shard: pinning that list would advertise a local bundle that cannot load.
    for name in files
        .iter()
        .filter(|name| name.ends_with(crate::safetensors::SAFETENSORS_INDEX_SUFFIX))
    {
        let index: Value = serde_json::from_slice(&fs::read(snapshot.join(name)).ok()?).ok()?;
        let weights = index.get("weight_map")?.as_object()?;
        if weights.is_empty() {
            return None;
        }
        let parent = Path::new(name).parent()?;
        for shard in weights.values() {
            let shard = parent.join(shard.as_str()?);
            if !files.iter().any(|file| Path::new(file) == shard) {
                return None;
            }
        }
    }
    if let Some(expected) = receipt.get("artifactTreeStamp").filter(|v| !v.is_null()) {
        if expected.as_str()? != resolved_files_tree_stamp(snapshot, &files).ok()? {
            return None;
        }
    }
    Some(revision.clone())
}

pub fn update_metadata_stamp(digest: &mut Sha256, metadata: &std::fs::Metadata) {
    digest.update(metadata.len().to_le_bytes());
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
    digest.update(
        modified
            .map_or(0, |duration| duration.as_secs())
            .to_le_bytes(),
    );
    digest.update(
        modified
            .map_or(0, |duration| duration.subsec_nanos())
            .to_le_bytes(),
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        digest.update(metadata.dev().to_le_bytes());
        digest.update(metadata.ino().to_le_bytes());
        digest.update(metadata.ctime().to_le_bytes());
        digest.update(metadata.ctime_nsec().to_le_bytes());
    }
}

pub fn resolved_files_tree_stamp(
    root: &Path,
    files: &[impl AsRef<str>],
) -> std::io::Result<String> {
    let mut names = files.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    names.sort_unstable();
    let mut digest = Sha256::new();
    for name in names {
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("artifact receipt contains unsafe resolved file {name:?}"),
            ));
        }
        let path = root.join(relative);
        let metadata = std::fs::symlink_metadata(&path)?;
        digest.update(name.as_bytes());
        digest.update([0]);
        update_metadata_stamp(&mut digest, &metadata);
        if metadata.file_type().is_symlink() {
            digest.update(std::fs::read_link(&path)?.to_string_lossy().as_bytes());
            digest.update(b"followed-target");
            update_metadata_stamp(&mut digest, &std::fs::metadata(&path)?);
        }
        digest.update([0xff]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    const REV: &str = "1111111111111111111111111111111111111111";
    const OTHER: &str = "2222222222222222222222222222222222222222";

    fn seed(root: &Path, revision: &str) -> std::path::PathBuf {
        let snapshot = root.join("snapshots").join(revision);
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
        snapshot
    }

    #[test]
    fn recovers_all_missing_forms_but_preserves_recorded_identity() {
        let temp = tempfile::tempdir().unwrap();
        seed(temp.path(), REV);
        for revision in [None, Some(Value::Null), Some(json!("")), Some(json!("  "))] {
            let mut receipt = json!({"resolvedFiles":["model.safetensors"]});
            if let Some(revision) = revision {
                receipt["snapshotRevision"] = revision;
            }
            assert_eq!(
                recover_revision(&receipt, temp.path()).as_deref(),
                Some(REV)
            );
        }
        for revision in [json!(OTHER), json!("main"), json!(false)] {
            let receipt =
                json!({"resolvedFiles":["model.safetensors"], "snapshotRevision":revision});
            assert_eq!(recover_revision(&receipt, temp.path()), None);
        }
    }

    #[test]
    fn unavailable_incomplete_unsafe_and_ambiguous_snapshots_stay_unresolved() {
        let temp = tempfile::tempdir().unwrap();
        let receipt = json!({"resolvedFiles":["model.safetensors"]});
        assert_eq!(recover_revision(&receipt, temp.path()), None);
        seed(temp.path(), "main");
        assert_eq!(recover_revision(&receipt, temp.path()), None);
        seed(temp.path(), REV);
        for files in [
            json!([]),
            json!(["missing"]),
            json!(["../model.safetensors"]),
            json!(["/model.safetensors"]),
            json!(["..\\model.safetensors"]),
            json!([null]),
            json!([""]),
        ] {
            assert_eq!(
                recover_revision(&json!({"resolvedFiles":files}), temp.path()),
                None
            );
        }
        seed(temp.path(), OTHER);
        assert_eq!(recover_revision(&receipt, temp.path()), None);
    }

    #[test]
    fn every_indexed_shard_must_be_recorded_even_when_all_exist_externally() {
        let temp = tempfile::tempdir().unwrap();
        let snapshot = seed(temp.path(), REV);
        fs::write(snapshot.join("other.safetensors"), b"other weights").unwrap();
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model.safetensors","b":"other.safetensors"}}"#,
        )
        .unwrap();
        let mut receipt =
            json!({"resolvedFiles":["model.safetensors.index.json", "model.safetensors"]});
        assert_eq!(recover_revision(&receipt, temp.path()), None);
        receipt["resolvedFiles"]
            .as_array_mut()
            .unwrap()
            .push(json!("other.safetensors"));
        assert_eq!(
            recover_revision(&receipt, temp.path()).as_deref(),
            Some(REV)
        );
    }

    #[test]
    fn existing_integrity_stamp_must_match_and_is_not_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let snapshot = seed(temp.path(), REV);
        let stamp = resolved_files_tree_stamp(&snapshot, &["model.safetensors"]).unwrap();
        let receipt = json!({"resolvedFiles":["model.safetensors"], "artifactTreeStamp":stamp});
        assert_eq!(
            recover_revision(&receipt, temp.path()).as_deref(),
            Some(REV)
        );
        fs::write(snapshot.join("model.safetensors"), b"changed weights").unwrap();
        assert_eq!(recover_revision(&receipt, temp.path()), None);
        assert_eq!(receipt["artifactTreeStamp"], stamp);
    }
}
