//! Bounded, allocation-light validation for installed safetensors model files.
//!
//! These helpers are intentionally in `sceneworks-core` so API readiness checks and worker-side
//! resolution cannot disagree about whether the same on-disk snapshot is usable.

use serde_json::Value;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Component, Path};

/// General model/checkpoint descriptor bound (the safetensors format limit).
pub const MAX_HEADER_BYTES: u64 = 100_000_000;
/// Adapter imports and prepared training bundles intentionally use a smaller descriptor bound.
pub const MAX_ADAPTER_HEADER_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub enum HeaderJsonPolicy {
    /// Preserve metadata discovery/import's serde JSON (last duplicate key wins) contract.
    Standard,
    /// Runnable checkpoint inspection rejects ambiguous descriptors, including nested duplicates.
    RejectDuplicateKeys,
}

#[derive(Debug)]
pub enum HeaderError {
    Io(std::io::Error),
    Truncated,
    InvalidLength(u64),
    Malformed(String),
    DuplicateKey(String),
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Truncated => f.write_str("safetensors header is truncated"),
            Self::InvalidLength(length) => {
                write!(f, "safetensors header length {length} is outside its bound")
            }
            Self::Malformed(error) | Self::DuplicateKey(error) => {
                write!(f, "invalid safetensors descriptor: {error}")
            }
        }
    }
}
impl std::error::Error for HeaderError {}

/// A decoded descriptor. Tensor-body validation is deliberately separate: discovery/import and
/// runnable checkpoint inspection have different dtype, metadata and payload-range contracts.
pub struct Header {
    pub value: Value,
    pub data_start: u64,
    pub file_len: u64,
}

/// Read only the bounded descriptor from an already-open stream. The supplied file length also
/// permits callers inspecting a bounded prefix to validate against the complete artifact size.
pub fn read_header(
    reader: &mut impl Read,
    file_len: u64,
    max_header_bytes: u64,
    json_policy: HeaderJsonPolicy,
) -> Result<Header, HeaderError> {
    let mut length = [0_u8; 8];
    reader
        .read_exact(&mut length)
        .map_err(|_| HeaderError::Truncated)?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > max_header_bytes {
        return Err(HeaderError::InvalidLength(header_len));
    }
    let data_start = header_len
        .checked_add(8)
        .filter(|end| *end <= file_len)
        .ok_or(HeaderError::Truncated)?;
    let header_len =
        usize::try_from(header_len).map_err(|_| HeaderError::InvalidLength(header_len))?;
    let mut raw = vec![0_u8; header_len];
    reader
        .read_exact(&mut raw)
        .map_err(|_| HeaderError::Truncated)?;
    if matches!(json_policy, HeaderJsonPolicy::RejectDuplicateKeys) {
        let text =
            std::str::from_utf8(&raw).map_err(|error| HeaderError::Malformed(error.to_string()))?;
        crate::jsonc::reject_duplicate_keys(text).map_err(|error| {
            if error.contains("duplicate object key") {
                HeaderError::DuplicateKey(error)
            } else {
                HeaderError::Malformed(error)
            }
        })?;
    }
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| HeaderError::Malformed(error.to_string()))?;
    if !value.is_object() {
        return Err(HeaderError::Malformed(
            "header must be a JSON object".to_owned(),
        ));
    }
    Ok(Header {
        value,
        data_start,
        file_len,
    })
}

/// Open a safetensors artifact and read its descriptor without opening a second metadata handle.
pub fn read_file_header(
    path: &Path,
    max_header_bytes: u64,
    json_policy: HeaderJsonPolicy,
) -> Result<Header, HeaderError> {
    let mut file = std::fs::File::open(path).map_err(HeaderError::Io)?;
    let file_len = file.metadata().map_err(HeaderError::Io)?.len();
    read_header(&mut file, file_len, max_header_bytes, json_policy)
}

// The pinned safetensors 0.7 descriptor contract includes sub-byte and complex dtypes.
// Keep the FNUZ aliases already recognized by checkpoint inspection as well.
pub(crate) fn dtype_bits(dtype: &str) -> Option<u64> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "U16" | "I16" | "F16" | "BF16" => Some(16),
        "U32" | "I32" | "F32" => Some(32),
        "U64" | "I64" | "F64" | "C64" => Some(64),
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
    let Ok(header) = read_file_header(path, MAX_HEADER_BYTES, HeaderJsonPolicy::Standard) else {
        return false;
    };
    let entries = header
        .value
        .as_object()
        .expect("shared reader requires an object");
    let data_len = header.file_len - header.data_start;
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
        let Some(bits_per_element) = entry
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(dtype_bits)
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
        let Some(expected_bits) = element_count.checked_mul(bits_per_element) else {
            return false;
        };
        if expected_bits % 8 != 0
            || start > end
            || end > data_len
            || end - start != expected_bits / 8
        {
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

    fn descriptor(raw: &[u8]) -> Vec<u8> {
        let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(raw);
        bytes
    }

    #[test]
    fn shared_header_bounds_truncation_shape_and_json_policy() {
        let valid = descriptor(br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#);
        let mut file = valid.clone();
        file.push(7);
        let mut cursor = std::io::Cursor::new(&file);
        let header = read_header(
            &mut cursor,
            file.len() as u64,
            MAX_HEADER_BYTES,
            HeaderJsonPolicy::Standard,
        )
        .unwrap();
        assert_eq!(
            cursor.position(),
            valid.len() as u64,
            "tensor payload is never read"
        );
        assert_eq!(header.data_start, valid.len() as u64);
        assert!(
            read_header(
                &mut file.as_slice(),
                file.len() as u64,
                (valid.len() - 8) as u64,
                HeaderJsonPolicy::Standard
            )
            .is_ok(),
            "the exact bound is inclusive"
        );
        assert!(matches!(
            read_header(
                &mut file.as_slice(),
                file.len() as u64,
                1,
                HeaderJsonPolicy::Standard
            ),
            Err(HeaderError::InvalidLength(_))
        ));
        for raw in [b"".as_slice(), b"[]", b"null", b"{broken"] {
            let bytes = descriptor(raw);
            assert!(read_header(
                &mut bytes.as_slice(),
                bytes.len() as u64,
                MAX_HEADER_BYTES,
                HeaderJsonPolicy::Standard
            )
            .is_err());
        }
        for bytes in [
            vec![0; 7],
            u64::MAX.to_le_bytes().to_vec(),
            descriptor(b"{}")[..9].to_vec(),
        ] {
            assert!(read_header(
                &mut bytes.as_slice(),
                bytes.len() as u64,
                MAX_HEADER_BYTES,
                HeaderJsonPolicy::Standard
            )
            .is_err());
        }
        let overflow = u64::MAX.to_le_bytes();
        assert!(matches!(
            read_header(
                &mut overflow.as_slice(),
                u64::MAX,
                u64::MAX,
                HeaderJsonPolicy::Standard
            ),
            Err(HeaderError::Truncated)
        ));
        let duplicate = descriptor(br#"{"__metadata__":{"format":"pt","format":"np"}}"#);
        assert!(read_header(
            &mut duplicate.as_slice(),
            duplicate.len() as u64,
            MAX_HEADER_BYTES,
            HeaderJsonPolicy::Standard
        )
        .is_ok());
        assert!(matches!(
            read_header(
                &mut duplicate.as_slice(),
                duplicate.len() as u64,
                MAX_HEADER_BYTES,
                HeaderJsonPolicy::RejectDuplicateKeys
            ),
            Err(HeaderError::DuplicateKey(_))
        ));
    }

    #[test]
    fn structural_validation_covers_scalar_empty_and_float8_descriptors() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("metadata-only-fixture.safetensors");
        // These fixture bytes are never executed by an inference backend.
        for (dtype, shape, size) in [
            ("BF16", serde_json::json!([]), 2),
            ("F8_E8M0", serde_json::json!([2]), 2),
            ("F32", serde_json::json!([0]), 0),
            ("F4", serde_json::json!([2]), 1),
            ("F6_E2M3", serde_json::json!([4]), 3),
            ("F6_E3M2", serde_json::json!([4]), 3),
            ("C64", serde_json::json!([]), 8),
        ] {
            let raw =
                serde_json::json!({"weight":{"dtype":dtype,"shape":shape,"data_offsets":[0,size]}})
                    .to_string();
            let mut bytes = descriptor(raw.as_bytes());
            bytes.resize(bytes.len() + size, 0);
            std::fs::write(&path, bytes).unwrap();
            assert!(file_is_structurally_valid(&path), "{dtype} descriptor");
        }
        for raw in [
            r#"{"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#,
            r#"{"a":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        ] {
            let mut bytes = descriptor(raw.as_bytes());
            bytes.extend_from_slice(&[0, 0]);
            std::fs::write(&path, bytes).unwrap();
            assert!(!file_is_structurally_valid(&path));
        }
    }

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
