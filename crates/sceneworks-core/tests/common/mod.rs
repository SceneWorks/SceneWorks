//! Descriptor-scale/filesystem fixtures shared by core integration binaries.
// Each binary consumes a different subset of this private module.
#![allow(dead_code)]

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

pub fn fixture_path(relative_path: &str) -> PathBuf {
    repo_root()
        .join("tests")
        .join("fixtures")
        .join("rust_migration_contracts")
        .join(relative_path)
}

pub fn write_safetensors(path: &Path, entries: &[(&str, &str)], metadata: Option<Value>, fill: u8) {
    let mut header = Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".to_owned(), metadata);
    }
    let mut offset = 0_u64;
    for (name, dtype) in entries {
        let width = match *dtype {
            "F16" | "BF16" => 2,
            "F32" => 4,
            _ => 1,
        };
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + width]}),
        );
        offset += width;
    }
    let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, fill);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

/// A minimal single-file Krea 2 native DiT: the `txtfusion.` marker the family detector keys on,
/// every tensor dense bf16, bytes deterministic. Same shape the plan-store suite uses, so the two
/// suites compile the same checkpoint through the two ownership modes.
pub fn write_krea_native_file(path: &Path, fill: u8) {
    write_safetensors(
        path,
        &[
            ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
            ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
            ("model.diffusion_model.first.weight", "BF16"),
        ],
        None,
        fill,
    );
}

/// Path → (len, modified) for every file under `root`, so a test can prove a whole tree is
/// byte-for-byte and timestamp-for-timestamp untouched.
pub fn tree_snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, std::time::SystemTime, Vec<u8>)> {
    let mut snapshot = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).unwrap();
            if metadata.is_dir() {
                stack.push(path);
                continue;
            }
            snapshot.insert(
                path.strip_prefix(root).unwrap().to_path_buf(),
                (
                    metadata.len(),
                    metadata.modified().unwrap(),
                    fs::read(&path).unwrap_or_default(),
                ),
            );
        }
    }
    snapshot
}

/// Read fixtures through the actual bounded header owner; no local header parser.
pub fn safetensors_header(path: &Path) -> sceneworks_core::safetensors::Header {
    use sceneworks_core::safetensors::{read_file_header, HeaderJsonPolicy, MAX_HEADER_BYTES};
    read_file_header(path, MAX_HEADER_BYTES, HeaderJsonPolicy::Standard)
        .expect("fixture header reads")
}
