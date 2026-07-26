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
}
