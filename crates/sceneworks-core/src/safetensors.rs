//! Bounded, allocation-light validation for installed safetensors model files.
//!
//! These helpers are intentionally in `sceneworks-core` so API readiness checks and worker-side
//! resolution cannot disagree about whether the same on-disk snapshot is usable.

use serde_json::Value;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Component, Path};

const MAX_HEADER_BYTES: usize = 100_000_000;

fn dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" | "F64" => Some(8),
        _ => None,
    }
}

/// Validate a safetensors file without reading its potentially multi-gigabyte tensor body.
///
/// The bounded JSON header must describe at least one tensor; tensor shapes, dtypes, and byte ranges
/// must agree; and the ranges must cover the data section contiguously. This catches placeholders,
/// truncated files, malformed headers, overlapping tensors, and headers whose claimed shape does not
/// match the stored byte range.
pub fn file_is_structurally_valid(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(file_len) = file.metadata().map(|metadata| metadata.len()) else {
        return false;
    };
    let mut header_len_bytes = [0_u8; 8];
    if file.read_exact(&mut header_len_bytes).is_err() {
        return false;
    }
    let Ok(header_len) = usize::try_from(u64::from_le_bytes(header_len_bytes)) else {
        return false;
    };
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return false;
    }
    let Some(header_and_prefix) = (header_len as u64).checked_add(8) else {
        return false;
    };
    if header_and_prefix > file_len {
        return false;
    }
    let mut header = vec![0_u8; header_len];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let Ok(Value::Object(entries)) = serde_json::from_slice::<Value>(&header) else {
        return false;
    };

    let data_len = file_len - header_and_prefix;
    let mut ranges = Vec::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            if !entry.is_object() {
                return false;
            }
            continue;
        }
        if name.is_empty() {
            return false;
        }
        let Some(entry) = entry.as_object() else {
            return false;
        };
        let Some(element_size) = entry
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(dtype_size)
        else {
            return false;
        };
        let Some(shape) = entry.get("shape").and_then(Value::as_array) else {
            return false;
        };
        let Some(element_count) = shape.iter().try_fold(1_u64, |count, dimension| {
            count.checked_mul(dimension.as_u64()?)
        }) else {
            return false;
        };
        let Some(offsets) = entry.get("data_offsets").and_then(Value::as_array) else {
            return false;
        };
        let [start, end] = offsets.as_slice() else {
            return false;
        };
        let (Some(start), Some(end)) = (start.as_u64(), end.as_u64()) else {
            return false;
        };
        let Some(expected_bytes) = element_count.checked_mul(element_size) else {
            return false;
        };
        if start > end || end > data_len || end - start != expected_bytes {
            return false;
        }
        ranges.push((start, end));
    }
    if ranges.is_empty() {
        return false;
    }
    ranges.sort_unstable();
    let mut next_offset = 0_u64;
    for (start, end) in ranges {
        if start != next_offset {
            return false;
        }
        next_offset = end;
    }
    next_offset == data_len
}

fn safe_relative_shard_path(shard: &str) -> bool {
    if shard.trim() != shard
        || shard.is_empty()
        || shard.starts_with(['/', '\\'])
        || shard.contains('\\')
        || shard.contains(':')
    {
        return false;
    }
    Path::new(shard)
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
}

/// Validate every unique shard named by a safetensors index under `dir`.
///
/// Index entries are constrained to safe relative paths before joining them to `dir`; absolute,
/// drive-qualified, backslash-qualified, and `..` traversal paths are rejected even when they happen
/// to name an existing valid file outside the snapshot.
pub fn indexed_files_are_structurally_valid(dir: &Path, index_file: &Path) -> bool {
    let Ok(index_raw) = std::fs::read_to_string(index_file) else {
        return false;
    };
    let Ok(index) = serde_json::from_str::<Value>(&index_raw) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return false;
    };
    let Some(shards) = weight_map
        .values()
        .map(|value| value.as_str().map(str::to_owned))
        .collect::<Option<BTreeSet<String>>>()
    else {
        return false;
    };
    !shards.is_empty()
        && shards.iter().all(|shard| {
            safe_relative_shard_path(shard)
                && file_is_structurally_valid(&dir.join(Path::new(shard)))
        })
}

/// The suffix that identifies a sharded-safetensors index (`model.safetensors.index.json`,
/// `diffusion_pytorch_model.safetensors.index.json`, …).
pub const SAFETENSORS_INDEX_SUFFIX: &str = ".safetensors.index.json";

/// Whether `path` names a sharded-safetensors index by filename.
pub fn is_safetensors_index_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(SAFETENSORS_INDEX_SUFFIX))
}

/// Shards a `*.safetensors.index.json` names in `weight_map` that are NOT on disk beside it.
///
/// **Existence only.** This stats each distinct shard and never opens one — no header parse, no
/// tensor read — because it runs on the install/availability lanes where a hashing or header-reading
/// check would be a measurable per-model cost (sc-20526). Structural validity of the shards
/// themselves stays with [`indexed_files_are_structurally_valid`], which the bespoke
/// family-completeness predicates use.
///
/// An index that cannot be read, does not parse, or carries no usable `weight_map` is itself
/// reported missing: a `*.safetensors.index.json` that is not a usable index is a torn install, not
/// an unrelated file. Callers must therefore only pass paths that pass
/// [`is_safetensors_index_path`].
///
/// Index entries are constrained to safe relative paths before joining them to `dir`, so an absolute,
/// drive-qualified, backslash-qualified, or `..`-traversing entry can never be satisfied by a file
/// outside the snapshot.
pub fn missing_indexed_shards(dir: &Path, index_file: &Path) -> Vec<String> {
    let unusable = || {
        vec![index_file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(SAFETENSORS_INDEX_SUFFIX)
            .to_owned()]
    };
    let Ok(index_raw) = std::fs::read_to_string(index_file) else {
        return unusable();
    };
    let Ok(index) = serde_json::from_str::<Value>(&index_raw) else {
        return unusable();
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return unusable();
    };
    let Some(shards) = weight_map
        .values()
        .map(|value| value.as_str().map(str::to_owned))
        .collect::<Option<BTreeSet<String>>>()
    else {
        return unusable();
    };
    if shards.is_empty() {
        return unusable();
    }
    shards
        .into_iter()
        .filter(|shard| !safe_relative_shard_path(shard) || !dir.join(Path::new(shard)).is_file())
        .collect()
}

/// Whether every distinct shard a `*.safetensors.index.json` names exists beside it.
///
/// See [`missing_indexed_shards`] for the cost contract (stat only).
pub fn indexed_shards_are_present(dir: &Path, index_file: &Path) -> bool {
    missing_indexed_shards(dir, index_file).is_empty()
}

fn non_empty_json_object(
    path: &Path,
    predicate: impl FnOnce(&serde_json::Map<String, Value>) -> bool,
) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok())
        .and_then(|value| value.as_object().cloned())
        .is_some_and(|object| !object.is_empty() && predicate(&object))
}

/// Whether `dir` is a complete Gemma text-encoder snapshot usable by both API and worker.
pub fn gemma_text_encoder_dir_is_complete(dir: &Path) -> bool {
    if !non_empty_json_object(&dir.join("config.json"), |_| true)
        || !non_empty_json_object(&dir.join("tokenizer.json"), |object| {
            object.get("model").is_some_and(Value::is_object)
        })
    {
        return false;
    }
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        indexed_files_are_structurally_valid(dir, &index)
    } else {
        file_is_structurally_valid(&dir.join("model.safetensors"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tiny_safetensors(path: &Path) {
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.push(7);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_gemma_scaffold(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), br#"{"model_type":"gemma3_text"}"#).unwrap();
        std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
    }

    #[test]
    fn structural_validation_rejects_truncated_and_inconsistent_tensors() {
        let temp = tempfile::tempdir().unwrap();
        let valid = temp.path().join("valid.safetensors");
        write_tiny_safetensors(&valid);
        assert!(file_is_structurally_valid(&valid));

        std::fs::write(temp.path().join("placeholder.safetensors"), b"x").unwrap();
        assert!(!file_is_structurally_valid(
            &temp.path().join("placeholder.safetensors")
        ));

        let bad_header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,1]}}"#;
        let mut bad = (bad_header.len() as u64).to_le_bytes().to_vec();
        bad.extend_from_slice(bad_header);
        bad.push(0);
        std::fs::write(temp.path().join("bad-shape.safetensors"), bad).unwrap();
        assert!(!file_is_structurally_valid(
            &temp.path().join("bad-shape.safetensors")
        ));
    }

    #[test]
    fn gemma_index_rejects_absolute_and_traversal_shards() {
        let temp = tempfile::tempdir().unwrap();
        let gemma = temp.path().join("gemma");
        write_gemma_scaffold(&gemma);
        let shard = gemma.join("model-00001-of-00001.safetensors");
        write_tiny_safetensors(&shard);

        for unsafe_path in [
            "../outside.safetensors",
            "/outside.safetensors",
            r"C:\outside.safetensors",
            r"..\outside.safetensors",
        ] {
            std::fs::write(
                gemma.join("model.safetensors.index.json"),
                serde_json::json!({"weight_map":{"weight":unsafe_path}}).to_string(),
            )
            .unwrap();
            assert!(
                !gemma_text_encoder_dir_is_complete(&gemma),
                "unsafe shard path must be rejected: {unsafe_path}"
            );
        }

        std::fs::write(
            gemma.join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        assert!(gemma_text_encoder_dir_is_complete(&gemma));

        std::fs::write(&shard, b"not safetensors").unwrap();
        assert!(
            !gemma_text_encoder_dir_is_complete(&gemma),
            "every indexed shard must be structurally valid"
        );
    }

    /// sc-20526: the lens_turbo bf16 shape — an index naming three shards with only the LAST one on
    /// disk. `model.embed_tokens.weight` lives in shard 1, so the load dies with "cannot find tensor"
    /// even though the component directory holds a real `.safetensors`.
    #[test]
    fn missing_indexed_shards_names_every_absent_shard() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("text_encoder");
        std::fs::create_dir_all(&dir).unwrap();
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(
            &index,
            serde_json::json!({"weight_map":{
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00002-of-00003.safetensors",
                "lm_head.weight": "model-00003-of-00003.safetensors",
            }})
            .to_string(),
        )
        .unwrap();
        write_tiny_safetensors(&dir.join("model-00003-of-00003.safetensors"));

        assert_eq!(
            missing_indexed_shards(&dir, &index),
            vec![
                "model-00001-of-00003.safetensors".to_owned(),
                "model-00002-of-00003.safetensors".to_owned(),
            ],
            "the two undownloaded shards must be reported"
        );
        assert!(!indexed_shards_are_present(&dir, &index));

        write_tiny_safetensors(&dir.join("model-00001-of-00003.safetensors"));
        write_tiny_safetensors(&dir.join("model-00002-of-00003.safetensors"));
        assert!(
            indexed_shards_are_present(&dir, &index),
            "a genuinely complete shard set must pass"
        );
    }

    /// Existence only: a shard that exists but is a truncated placeholder still satisfies this check.
    /// The whole point is that the install/availability lanes never open a shard (sc-20526).
    #[test]
    fn indexed_shard_presence_never_opens_a_shard() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"a":"diffusion_pytorch_model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("diffusion_pytorch_model-00001-of-00001.safetensors"),
            b"x",
        )
        .unwrap();
        assert!(indexed_shards_are_present(dir, &index));
        assert!(
            !indexed_files_are_structurally_valid(dir, &index),
            "the structural check is the separate, costlier contract"
        );
    }

    /// An unreadable/garbage/weight_map-less `*.safetensors.index.json` is a torn install, not an
    /// unrelated file — it must report missing rather than pass.
    #[test]
    fn unusable_index_reports_itself_missing() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        let index = dir.join("model.safetensors.index.json");

        assert_eq!(
            missing_indexed_shards(dir, &index),
            vec!["model.safetensors.index.json".to_owned()],
            "absent index"
        );
        std::fs::write(&index, b"{not json").unwrap();
        assert_eq!(
            missing_indexed_shards(dir, &index).len(),
            1,
            "garbage index"
        );
        std::fs::write(&index, br#"{"metadata":{}}"#).unwrap();
        assert_eq!(
            missing_indexed_shards(dir, &index).len(),
            1,
            "no weight_map"
        );
        std::fs::write(&index, br#"{"weight_map":{}}"#).unwrap();
        assert_eq!(
            missing_indexed_shards(dir, &index).len(),
            1,
            "empty weight_map"
        );
    }

    #[test]
    fn index_paths_are_recognised_by_suffix() {
        assert!(is_safetensors_index_path(Path::new(
            "bf16/text_encoder/model.safetensors.index.json"
        )));
        assert!(is_safetensors_index_path(Path::new(
            "diffusion_pytorch_model.safetensors.index.json"
        )));
        assert!(!is_safetensors_index_path(Path::new("model_index.json")));
        assert!(!is_safetensors_index_path(Path::new(
            "model-00001-of-00003.safetensors"
        )));
    }

    /// A traversal entry can never be satisfied by a file that really exists outside the snapshot.
    #[test]
    fn unsafe_shard_entries_are_reported_missing() {
        let temp = tempfile::tempdir().unwrap();
        write_tiny_safetensors(&temp.path().join("outside.safetensors"));
        let dir = temp.path().join("component");
        std::fs::create_dir_all(&dir).unwrap();
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
        )
        .unwrap();
        assert_eq!(
            missing_indexed_shards(&dir, &index),
            vec!["../outside.safetensors".to_owned()]
        );
    }
}
