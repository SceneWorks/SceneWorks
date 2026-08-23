use std::fs;
use std::path::{Path, PathBuf};

use sceneworks_core::checkpoint_import::{ManagedProvenanceV1, SourceLocatorV1};
use sceneworks_core::checkpoint_inspector::{
    discover_checkpoint, inspect_checkpoint, inspect_checkpoint_with_hook,
    CheckpointDiagnosticCodeV1, CheckpointInspectionEventV1, CheckpointInspectionRequestV1,
    CheckpointLayoutV1,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn linked_request(root: &Path, relative_path: &str) -> CheckpointInspectionRequestV1 {
    CheckpointInspectionRequestV1::linked(
        "community-checkpoint",
        root,
        relative_path,
        "comfy-primary",
    )
    .expect("valid request")
}

fn write_safetensors(path: &Path, entries: &[(&str, &str)], metadata: Option<Value>) {
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
            json!({
                "dtype": dtype,
                "shape": [1],
                "data_offsets": [offset, offset + width],
            }),
        );
        offset += width;
    }
    let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, 0x5a);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn qwen_transformer_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("model.diffusion_model.img_in.weight", "F8_E4M3"),
        (
            "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
            "F8_E4M3",
        ),
        (
            "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
            "F8_E4M3",
        ),
        (
            "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
            "F8_E4M3",
        ),
        (
            "model.diffusion_model.transformer_blocks.0.txt_mlp.net.0.proj.weight",
            "F8_E4M3",
        ),
        (
            "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
            "F8_E4M3",
        ),
    ]
}

fn sdxl_fused_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "model.diffusion_model.input_blocks.7.0.out_layers.3.weight",
            "F16",
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
            "F16",
        ),
        (
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "F16",
        ),
        (
            "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight",
            "F16",
        ),
        ("first_stage_model.encoder.conv_in.weight", "F16"),
        ("first_stage_model.decoder.conv_out.weight", "F16"),
    ]
}

fn vae_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("encoder.conv_in.weight", "F32"),
        ("encoder.down.0.block.0.conv1.weight", "F32"),
        ("encoder.mid.attn_1.q.weight", "F32"),
        ("decoder.conv_out.weight", "F32"),
        ("decoder.up.0.block.0.conv1.weight", "F32"),
        ("decoder.mid.attn_1.q.weight", "F32"),
    ]
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend(value.to_le_bytes());
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend(value.as_bytes());
}

fn write_tiny_gguf(path: &Path, architecture: &str) -> usize {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 1); // tensor count
    push_u64(&mut bytes, 2); // metadata count

    push_gguf_string(&mut bytes, "general.architecture");
    push_u32(&mut bytes, 8); // string
    push_gguf_string(&mut bytes, architecture);
    push_gguf_string(&mut bytes, "general.alignment");
    push_u32(&mut bytes, 4); // u32
    push_u32(&mut bytes, 32);

    push_gguf_string(&mut bytes, "model.weight");
    push_u32(&mut bytes, 1); // dimensions
    push_u64(&mut bytes, 1);
    push_u32(&mut bytes, 0); // F32
    let tensor_offset_position = bytes.len();
    push_u64(&mut bytes, 0); // relative data offset

    let aligned = bytes.len().div_ceil(32) * 32;
    bytes.resize(aligned, 0);
    bytes.extend(1_f32.to_le_bytes());
    fs::write(path, bytes).unwrap();
    tensor_offset_position
}

fn write_duplicate_metadata_gguf(path: &Path) {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, 2);
    for value in ["flux", "qwen"] {
        push_gguf_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8);
        push_gguf_string(&mut bytes, value);
    }
    fs::write(path, bytes).unwrap();
}

fn write_quantized_gguf(
    path: &Path,
    first_dimension: u64,
    second_dimension: u64,
    tensor_type: u32,
    block_bytes: usize,
) {
    write_quantized_gguf_fixture(
        path,
        3,
        32,
        Some(2),
        first_dimension,
        second_dimension,
        tensor_type,
        block_bytes,
        "model.weight",
    );
}

#[derive(Clone, Copy)]
struct GgufFixtureOffsets {
    alignment: usize,
    quantization_version: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn write_quantized_gguf_fixture(
    path: &Path,
    version: u32,
    alignment: u32,
    quantization_version: Option<u32>,
    first_dimension: u64,
    second_dimension: u64,
    tensor_type: u32,
    block_bytes: usize,
    tensor_name: &str,
) -> GgufFixtureOffsets {
    let mut bytes = b"GGUF".to_vec();
    push_u32(&mut bytes, version);
    push_u64(&mut bytes, 1); // tensor count
    push_u64(
        &mut bytes,
        if quantization_version.is_some() { 3 } else { 2 },
    );

    push_gguf_string(&mut bytes, "general.architecture");
    push_u32(&mut bytes, 8); // string
    push_gguf_string(&mut bytes, "flux");
    push_gguf_string(&mut bytes, "general.alignment");
    push_u32(&mut bytes, 4); // u32
    let alignment_offset = bytes.len();
    push_u32(&mut bytes, alignment);
    let quantization_version_offset = quantization_version.map(|version| {
        push_gguf_string(&mut bytes, "general.quantization_version");
        push_u32(&mut bytes, 4); // u32
        let offset = bytes.len();
        push_u32(&mut bytes, version);
        offset
    });

    push_gguf_string(&mut bytes, tensor_name);
    push_u32(&mut bytes, 2); // dimensions
    push_u64(&mut bytes, first_dimension);
    push_u64(&mut bytes, second_dimension);
    push_u32(&mut bytes, tensor_type);
    push_u64(&mut bytes, 0); // relative data offset

    let alignment = alignment as usize;
    let aligned = bytes.len() + (alignment - (bytes.len() % alignment)) % alignment;
    bytes.resize(aligned, 0);
    bytes.resize(bytes.len() + block_bytes, 0x5a);
    fs::write(path, bytes).unwrap();
    GgufFixtureOffsets {
        alignment: alignment_offset,
        quantization_version: quantization_version_offset,
    }
}

fn write_raw_safetensors(path: &Path, header: &[u8], body: &[u8]) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(body);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_safetensors_index(path: &Path, mappings: &[(&str, &str)]) {
    let weight_map = mappings
        .iter()
        .map(|(tensor, shard)| ((*tensor).to_owned(), json!(shard)))
        .collect::<Map<String, Value>>();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
}

fn overwrite_u32(path: &Path, offset: usize, value: u32) {
    let mut bytes = fs::read(path).unwrap();
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    fs::write(path, bytes).unwrap();
}

fn replace_bytes(path: &Path, needle: &[u8], replacement: &[u8]) {
    assert_eq!(needle.len(), replacement.len());
    let mut bytes = fs::read(path).unwrap();
    let offset = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("fixture contains bytes to replace");
    bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
    fs::write(path, bytes).unwrap();
}

fn sha256_file(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn restore_modified_time(path: &Path, modified: std::time::SystemTime) {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
}

#[cfg(unix)]
fn symlink_file(target: &Path, link: &Path) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[cfg(windows)]
fn symlink_file(target: &Path, link: &Path) {
    std::os::windows::fs::symlink_file(target, link).unwrap();
}

#[cfg(windows)]
struct WindowsDirectoryJunction(PathBuf);

#[cfg(windows)]
impl WindowsDirectoryJunction {
    fn create(path: &Path, target: &Path) -> Self {
        let output = std::process::Command::new("cmd")
            .args(["/D", "/C", "mklink", "/J"])
            .arg(path)
            .arg(target)
            .output()
            .expect("launch cmd.exe to create checkpoint fixture junction");
        assert!(
            output.status.success(),
            "mklink /J failed with {}\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        use std::os::windows::fs::MetadataExt as _;
        assert_ne!(
            fs::symlink_metadata(path).unwrap().file_attributes() & 0x400,
            0,
            "mklink /J fixture must be a reparse point"
        );
        Self(path.to_owned())
    }
}

#[cfg(windows)]
impl Drop for WindowsDirectoryJunction {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.0);
    }
}

fn has_code(
    result: &sceneworks_core::checkpoint_inspector::CheckpointInspectionV1,
    code: CheckpointDiagnosticCodeV1,
) -> bool {
    result.diagnostics.iter().any(|item| item.code == code)
}

#[test]
fn safetensors_single_file_and_fused_checkpoint_share_the_inventory_contract() {
    let temp = TempDir::new().unwrap();
    write_safetensors(
        &temp.path().join("qwen.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    write_safetensors(
        &temp.path().join("sdxl.safetensors"),
        &sdxl_fused_entries(),
        Some(json!({"format": "pt"})),
    );

    let single = inspect_checkpoint(&linked_request(temp.path(), "qwen.safetensors"));
    let fused = inspect_checkpoint(&linked_request(temp.path(), "sdxl.safetensors"));

    assert!(single.is_runnable(), "{:?}", single.diagnostics);
    assert!(fused.is_runnable(), "{:?}", fused.diagnostics);
    assert_eq!(single.layout, Some(CheckpointLayoutV1::SingleFile));
    assert_eq!(fused.layout, Some(CheckpointLayoutV1::FusedCheckpoint));
    assert_eq!(
        single.inventory.schema_version,
        fused.inventory.schema_version
    );
    assert_eq!(single.inventory.records.len(), 1);
    assert_eq!(fused.inventory.records.len(), 1);
    assert_eq!(single.plans.len(), 1);
    assert_eq!(fused.plans.len(), 1);
}

#[test]
fn gguf_and_component_directory_produce_the_same_inventory_shape() {
    let temp = TempDir::new().unwrap();
    write_tiny_gguf(&temp.path().join("model.gguf"), "flux");

    let directory = temp.path().join("diffusers");
    write_safetensors(
        &directory.join("transformer/model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    write_safetensors(
        &directory.join("vae/model.safetensors"),
        &vae_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::write(
        directory.join("model_index.json"),
        br#"{"_class_name":"QwenImagePipeline","transformer":["diffusers","QwenImageTransformer2DModel"],"vae":["diffusers","AutoencoderKL"]}"#,
    )
    .unwrap();
    fs::write(
        directory.join("transformer/config.json"),
        br#"{"_class_name":"QwenImageTransformer2DModel"}"#,
    )
    .unwrap();
    fs::write(
        directory.join("vae/config.json"),
        br#"{"_class_name":"AutoencoderKL"}"#,
    )
    .unwrap();

    let gguf = inspect_checkpoint(&linked_request(temp.path(), "model.gguf"));
    let components = inspect_checkpoint(&linked_request(temp.path(), "diffusers"));

    assert!(gguf.is_runnable(), "{:?}", gguf.diagnostics);
    assert!(components.is_runnable(), "{:?}", components.diagnostics);
    assert_eq!(gguf.layout, Some(CheckpointLayoutV1::SingleFile));
    assert_eq!(
        components.layout,
        Some(CheckpointLayoutV1::ComponentDirectory)
    );
    assert_eq!(gguf.inventory.records.len(), 1);
    assert_eq!(components.inventory.records.len(), 1);
    assert!(components
        .evidence
        .iter()
        .any(|item| item.relative_path.ends_with("model_index.json")));
    assert!(components
        .evidence
        .iter()
        .any(|item| item.role.as_deref() == Some("vae")));
}

#[test]
fn discovery_is_header_only_but_runnable_validation_checks_declared_ranges() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("truncated.safetensors");
    write_safetensors(
        &path,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();

    let request = linked_request(temp.path(), "truncated.safetensors");
    let discovery = discover_checkpoint(&request);
    assert_eq!(discovery.candidates.len(), 1);
    assert!(!discovery
        .diagnostics
        .iter()
        .any(|item| item.code == CheckpointDiagnosticCodeV1::TruncatedData));

    let inspected = inspect_checkpoint(&request);
    assert!(has_code(
        &inspected,
        CheckpointDiagnosticCodeV1::TruncatedData
    ));
    assert!(inspected.inventory.records.is_empty());
}

#[test]
fn duplicate_keys_and_malformed_metadata_are_typed_and_actionable() {
    let temp = TempDir::new().unwrap();
    let duplicate_header = br#"{"model.diffusion_model.img_in.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"model.diffusion_model.img_in.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let mut duplicate = (duplicate_header.len() as u64).to_le_bytes().to_vec();
    duplicate.extend(duplicate_header);
    duplicate.push(1);
    fs::write(temp.path().join("duplicate.safetensors"), duplicate).unwrap();

    write_safetensors(
        &temp.path().join("bad-metadata.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": 7})),
    );

    let duplicate = inspect_checkpoint(&linked_request(temp.path(), "duplicate.safetensors"));
    let malformed = inspect_checkpoint(&linked_request(temp.path(), "bad-metadata.safetensors"));
    assert!(has_code(
        &duplicate,
        CheckpointDiagnosticCodeV1::DuplicateKey
    ));
    assert!(duplicate
        .diagnostics
        .iter()
        .any(|item| item.message.contains("model.diffusion_model.img_in.weight")));
    assert!(has_code(
        &malformed,
        CheckpointDiagnosticCodeV1::MalformedMetadata
    ));
    assert!(malformed
        .diagnostics
        .iter()
        .any(|item| item.message.contains("string values")));
}

#[test]
fn missing_index_shards_and_ambiguous_component_roles_are_diagnostics() {
    let temp = TempDir::new().unwrap();
    let indexed = temp.path().join("indexed");
    write_safetensors(
        &indexed.join("model-00001-of-00002.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::write(
        indexed.join("model.safetensors.index.json"),
        br#"{"metadata":{},"weight_map":{"one":"model-00001-of-00002.safetensors","two":"model-00002-of-00002.safetensors"}}"#,
    )
    .unwrap();

    let conflicting = temp.path().join("conflicting");
    write_safetensors(
        &conflicting.join("vae/not-a-vae.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );

    let missing = inspect_checkpoint(&linked_request(temp.path(), "indexed"));
    let ambiguous = inspect_checkpoint(&linked_request(temp.path(), "conflicting"));
    assert!(has_code(
        &missing,
        CheckpointDiagnosticCodeV1::MissingSidecar
    ));
    assert!(missing
        .diagnostics
        .iter()
        .any(|item| item.message.contains("model-00002-of-00002.safetensors")));
    assert!(has_code(
        &ambiguous,
        CheckpointDiagnosticCodeV1::AmbiguousComponentRole
    ));
    assert!(ambiguous
        .diagnostics
        .iter()
        .any(|item| { item.message.contains("vae") && item.message.contains("transformer") }));
}

#[test]
fn unchanged_bytes_are_deterministic_and_changed_bytes_rotate_the_fingerprint() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("model.safetensors");
    write_safetensors(
        &path,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let request = linked_request(temp.path(), "model.safetensors");

    let first = inspect_checkpoint(&request);
    let second = inspect_checkpoint(&request);
    assert!(first.is_runnable(), "{:?}", first.diagnostics);
    assert_eq!(first.fingerprint, second.fingerprint);
    assert_eq!(
        first.inventory.canonical_json().unwrap(),
        second.inventory.canonical_json().unwrap()
    );
    assert_eq!(
        first.plans[0].canonical_json().unwrap(),
        second.plans[0].canonical_json().unwrap()
    );

    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    fs::write(&path, bytes).unwrap();
    let changed = inspect_checkpoint(&request);
    assert!(changed.is_runnable(), "{:?}", changed.diagnostics);
    assert_ne!(first.fingerprint, changed.fingerprint);
    assert_ne!(
        first.inventory.canonical_json().unwrap(),
        changed.inventory.canonical_json().unwrap()
    );
}

#[test]
fn descriptor_duplicate_keys_and_invalid_gguf_ranges_fail_closed() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("descriptor");
    write_safetensors(
        &directory.join("transformer/model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::write(
        directory.join("transformer/config.json"),
        br#"{"_class_name":"QwenImageTransformer2DModel","_class_name":"Other"}"#,
    )
    .unwrap();

    let tensor_offset_position = write_tiny_gguf(&temp.path().join("bad.gguf"), "flux");
    let bad_path = temp.path().join("bad.gguf");
    let mut bytes = fs::read(&bad_path).unwrap();
    bytes[tensor_offset_position..tensor_offset_position + 8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&bad_path, bytes).unwrap();

    let descriptor = inspect_checkpoint(&linked_request(temp.path(), "descriptor"));
    let gguf = inspect_checkpoint(&linked_request(temp.path(), "bad.gguf"));
    assert!(has_code(
        &descriptor,
        CheckpointDiagnosticCodeV1::DuplicateKey
    ));
    assert!(has_code(
        &gguf,
        CheckpointDiagnosticCodeV1::InvalidTensorRange
    ));
}

#[test]
fn path_confinement_errors_are_typed_and_do_not_read_outside_the_root() {
    let temp = TempDir::new().unwrap();
    let request = CheckpointInspectionRequestV1::linked(
        "escape",
        temp.path(),
        PathBuf::from("../outside.safetensors"),
        "comfy-primary",
    );
    assert!(request.is_err());
}

#[test]
fn gguf_component_path_supplies_role_without_overriding_embedded_architecture() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("comfy");
    fs::create_dir_all(directory.join("unet")).unwrap();
    write_tiny_gguf(&directory.join("unet/model.gguf"), "flux");

    let result = inspect_checkpoint(&linked_request(temp.path(), "comfy"));
    assert!(result.is_runnable(), "{:?}", result.diagnostics);
    assert_eq!(result.layout, Some(CheckpointLayoutV1::ComponentDirectory));
    assert_eq!(result.evidence[0].role.as_deref(), Some("transformer"));
    assert_eq!(result.evidence[0].family.as_deref(), Some("flux"));
}

#[test]
fn managed_sources_compile_managed_locators_for_every_artifact() {
    let temp = TempDir::new().unwrap();
    write_safetensors(
        &temp.path().join("model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let request = CheckpointInspectionRequestV1::managed(
        "managed-community-model",
        temp.path(),
        "model.safetensors",
        "install-42",
        ManagedProvenanceV1 {
            source: "civitai".to_owned(),
            reference: Some("model-version-7".to_owned()),
            ..ManagedProvenanceV1::default()
        },
    )
    .unwrap();

    let result = inspect_checkpoint(&request);
    assert!(result.is_runnable(), "{:?}", result.diagnostics);
    assert!(result.plans[0]
        .layers
        .iter()
        .all(|layer| matches!(layer.source, SourceLocatorV1::Managed { .. })));
}

#[test]
fn descriptor_byte_changes_rotate_directory_fingerprint_and_plan_identity() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("diffusers");
    write_safetensors(
        &directory.join("transformer/model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let config = directory.join("transformer/config.json");
    fs::write(
        &config,
        br#"{"_class_name":"QwenImageTransformer2DModel","revision":"one"}"#,
    )
    .unwrap();
    let request = linked_request(temp.path(), "diffusers");
    let first = inspect_checkpoint(&request);
    fs::write(
        &config,
        br#"{"_class_name":"QwenImageTransformer2DModel","revision":"two"}"#,
    )
    .unwrap();
    let second = inspect_checkpoint(&request);

    assert!(first.is_runnable(), "{:?}", first.diagnostics);
    assert!(second.is_runnable(), "{:?}", second.diagnostics);
    assert_ne!(first.fingerprint, second.fingerprint);
    assert_ne!(first.plans[0].plan_id, second.plans[0].plan_id);
}

#[test]
fn duplicate_gguf_metadata_and_unsafe_index_paths_fail_closed() {
    let temp = TempDir::new().unwrap();
    write_duplicate_metadata_gguf(&temp.path().join("duplicate.gguf"));

    let indexed = temp.path().join("indexed");
    write_safetensors(
        &indexed.join("model-00001-of-00001.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::write(
        indexed.join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .unwrap();

    let duplicate = inspect_checkpoint(&linked_request(temp.path(), "duplicate.gguf"));
    let unsafe_index = inspect_checkpoint(&linked_request(temp.path(), "indexed"));
    assert!(has_code(
        &duplicate,
        CheckpointDiagnosticCodeV1::DuplicateKey
    ));
    assert!(has_code(
        &unsafe_index,
        CheckpointDiagnosticCodeV1::PathEscapesRoot
    ));
    assert!(unsafe_index.inventory.records.is_empty());
}

#[test]
fn safetensors_indices_are_bijections_over_the_actual_shard_tensor_tables() {
    let temp = TempDir::new().unwrap();
    let shard_name = "model-00001-of-00001.safetensors";

    let valid = temp.path().join("valid/transformer");
    let qwen_entries = qwen_transformer_entries();
    write_safetensors(
        &valid.join(shard_name),
        &qwen_entries,
        Some(json!({"format": "pt"})),
    );
    let valid_mappings = qwen_entries
        .iter()
        .map(|(tensor, _)| (*tensor, shard_name))
        .collect::<Vec<_>>();
    write_safetensors_index(&valid.join("model.safetensors.index.json"), &valid_mappings);
    let valid = inspect_checkpoint(&linked_request(temp.path(), "valid"));
    assert!(valid.is_runnable(), "{:?}", valid.diagnostics);
    let shard_evidence = valid
        .evidence
        .iter()
        .find(|item| item.relative_path.ends_with(shard_name))
        .expect("shard evidence");
    let mut expected_tensor_names = qwen_entries
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .collect::<Vec<_>>();
    expected_tensor_names.sort();
    assert_eq!(shard_evidence.tensor_names, expected_tensor_names);

    let bogus = temp.path().join("bogus/transformer");
    write_safetensors(&bogus.join(shard_name), &[("actual.weight", "F16")], None);
    write_safetensors_index(
        &bogus.join("model.safetensors.index.json"),
        &[("bogus.weight", shard_name)],
    );

    let incomplete = temp.path().join("incomplete/transformer");
    write_safetensors(
        &incomplete.join(shard_name),
        &[("actual.one", "F16"), ("actual.two", "F16")],
        None,
    );
    write_safetensors_index(
        &incomplete.join("model.safetensors.index.json"),
        &[("actual.one", shard_name)],
    );

    let duplicate = temp.path().join("duplicate/transformer");
    let shard_one = "model-00001-of-00002.safetensors";
    let shard_two = "model-00002-of-00002.safetensors";
    write_safetensors(
        &duplicate.join(shard_one),
        &[("shared.weight", "F16")],
        None,
    );
    write_safetensors(
        &duplicate.join(shard_two),
        &[("shared.weight", "F16"), ("other.weight", "F16")],
        None,
    );
    write_safetensors_index(
        &duplicate.join("model.safetensors.index.json"),
        &[("shared.weight", shard_one), ("other.weight", shard_two)],
    );

    let bogus = inspect_checkpoint(&linked_request(temp.path(), "bogus"));
    let incomplete = inspect_checkpoint(&linked_request(temp.path(), "incomplete"));
    let duplicate = inspect_checkpoint(&linked_request(temp.path(), "duplicate"));
    for result in [&bogus, &incomplete, &duplicate] {
        assert!(has_code(
            result,
            CheckpointDiagnosticCodeV1::IndexTensorMismatch
        ));
        assert!(result.inventory.records.is_empty());
    }
    assert!(bogus.diagnostics.iter().any(|item| item
        .message
        .contains("does not exist in its declared shard")));
    assert!(incomplete
        .diagnostics
        .iter()
        .any(|item| item.message.contains("is missing from weight_map")));
    assert!(duplicate
        .diagnostics
        .iter()
        .any(|item| item.message.contains("exists in multiple indexed shards")));
}

#[test]
fn same_size_mutation_between_complete_passes_is_typed_and_uses_verified_bytes() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("mutable");
    let transformer = directory.join("transformer/model.safetensors");
    write_safetensors(
        &transformer,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    write_safetensors(
        &directory.join("vae/model.safetensors"),
        &vae_entries(),
        Some(json!({"format": "pt"})),
    );
    let original_size = fs::metadata(&transformer).unwrap().len();
    let request = linked_request(temp.path(), "mutable");
    let changed = inspect_checkpoint_with_hook(&request, |event| {
        if event == CheckpointInspectionEventV1::FirstExactBytePassComplete {
            replace_bytes(&transformer, &[0x5a], &[0x5b]);
        }
    });
    assert_eq!(fs::metadata(&transformer).unwrap().len(), original_size);
    assert!(has_code(
        &changed,
        CheckpointDiagnosticCodeV1::SourceChangedDuringInspection
    ));
    assert!(!changed.is_runnable());
    assert!(changed.inventory.records.is_empty());

    let stable = inspect_checkpoint(&request);
    assert!(stable.is_runnable(), "{:?}", stable.diagnostics);
    assert_eq!(changed.fingerprint, stable.fingerprint);
    assert_eq!(changed.evidence, stable.evidence);
}

#[test]
fn mutation_after_the_second_pass_is_caught_by_final_identity_revalidation() {
    let temp = TempDir::new().unwrap();
    let artifact = temp.path().join("mutable.safetensors");
    write_safetensors(
        &artifact,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let original_size = fs::metadata(&artifact).unwrap().len();
    let changed = inspect_checkpoint_with_hook(
        &linked_request(temp.path(), "mutable.safetensors"),
        |event| {
            if event == CheckpointInspectionEventV1::SecondExactBytePassComplete {
                replace_bytes(&artifact, &[0x5a], &[0x5b]);
            }
        },
    );
    assert_eq!(fs::metadata(&artifact).unwrap().len(), original_size);
    assert!(has_code(
        &changed,
        CheckpointDiagnosticCodeV1::SourceChangedDuringInspection
    ));
    assert!(!changed.is_runnable());
    assert!(changed.inventory.records.is_empty());
}

#[test]
fn timestamp_restored_same_size_mutation_uses_authoritative_final_bytes() {
    let temp = TempDir::new().unwrap();
    let artifact = temp.path().join("timestamp-restored.safetensors");
    write_safetensors(
        &artifact,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let original_size = fs::metadata(&artifact).unwrap().len();
    let original_modified = fs::metadata(&artifact).unwrap().modified().unwrap();
    let changed = inspect_checkpoint_with_hook(
        &linked_request(temp.path(), "timestamp-restored.safetensors"),
        |event| {
            if event == CheckpointInspectionEventV1::SecondExactBytePassComplete {
                replace_bytes(&artifact, &[0x5a], &[0x5b]);
                restore_modified_time(&artifact, original_modified);
            }
        },
    );

    assert_eq!(fs::metadata(&artifact).unwrap().len(), original_size);
    assert_eq!(
        fs::metadata(&artifact).unwrap().modified().unwrap(),
        original_modified,
        "the regression must defeat metadata-only revalidation deterministically"
    );
    assert!(has_code(
        &changed,
        CheckpointDiagnosticCodeV1::SourceChangedDuringInspection
    ));
    assert!(!changed.is_runnable());
    assert!(changed.inventory.records.is_empty());
    assert_eq!(changed.evidence.len(), 1);
    assert_eq!(
        changed.evidence[0].sha256,
        sha256_file(&artifact),
        "even a rejected inspection returns only evidence derived from the final verified bytes"
    );

    let stable = inspect_checkpoint(&linked_request(
        temp.path(),
        "timestamp-restored.safetensors",
    ));
    assert!(stable.is_runnable(), "{:?}", stable.diagnostics);
    assert_eq!(changed.evidence, stable.evidence);
    assert_eq!(changed.fingerprint, stable.fingerprint);
}

#[test]
fn discovery_limits_are_typed_instead_of_silently_truncating_the_inventory() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("too-many");
    fs::create_dir_all(&directory).unwrap();
    for index in 0..4200 {
        fs::create_dir(directory.join(format!("empty-{index:04}"))).unwrap();
    }
    let discovery = discover_checkpoint(&linked_request(temp.path(), "too-many"));
    assert!(discovery
        .diagnostics
        .iter()
        .any(|item| item.code == CheckpointDiagnosticCodeV1::DiscoveryLimitExceeded));
}

#[test]
fn aggregate_tensor_name_evidence_budget_fails_closed_across_artifacts() {
    let temp = TempDir::new().unwrap();
    let directory = temp.path().join("aggregate-evidence/transformer");
    let left_names = (0..40_000)
        .map(|index| format!("left.{index:05}.weight"))
        .collect::<Vec<_>>();
    let right_names = (0..40_000)
        .map(|index| format!("right.{index:05}.weight"))
        .collect::<Vec<_>>();
    let mut left_entries = qwen_transformer_entries();
    left_entries.extend(left_names.iter().map(|name| (name.as_str(), "U8")));
    let right_entries = right_names
        .iter()
        .map(|name| (name.as_str(), "U8"))
        .collect::<Vec<_>>();
    write_safetensors(&directory.join("left.safetensors"), &left_entries, None);
    write_safetensors(&directory.join("right.safetensors"), &right_entries, None);

    let inspected = inspect_checkpoint(&linked_request(temp.path(), "aggregate-evidence"));
    let budget_diagnostics = inspected
        .diagnostics
        .iter()
        .filter(|item| item.code == CheckpointDiagnosticCodeV1::DiscoveryLimitExceeded)
        .collect::<Vec<_>>();
    assert_eq!(budget_diagnostics.len(), 1, "{:?}", inspected.diagnostics);
    assert_eq!(
        budget_diagnostics[0].message,
        "checkpoint inspection exceeded the aggregate evidence budget of 65536 tensor names, 33554432 tensor-name UTF-8 bytes, or 67108864 total evidence UTF-8 bytes"
    );
    assert!(!inspected.is_runnable());
    assert!(inspected.inventory.records.is_empty());
}

#[test]
fn family_evidence_conflicts_and_declared_tensor_geometry_fail_closed() {
    let temp = TempDir::new().unwrap();
    let conflict = temp.path().join("family-conflict");
    write_raw_safetensors(
        &conflict.join("transformer/model.safetensors"),
        br#"{"opaque.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0x5a],
    );
    fs::write(
        conflict.join("model_index.json"),
        br#"{"_class_name":"FluxTransformer2DModel","architectures":["QwenImageTransformer2DModel"]}"#,
    )
    .unwrap();
    write_tiny_gguf(&temp.path().join("invalid-family.gguf"), "FLUX");

    write_raw_safetensors(
        &temp.path().join("overlap.safetensors"),
        br#"{"model.diffusion_model.img_in.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"model.diffusion_model.transformer_blocks.0.attn.to_q.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0x5a],
    );
    write_raw_safetensors(
        &temp.path().join("shape-mismatch.safetensors"),
        br#"{"model.diffusion_model.img_in.weight":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}}"#,
        &[0x5a; 2],
    );
    write_quantized_gguf(&temp.path().join("bad-quant.gguf"), 1, 32, 2, 18); // Q4_0: total count is divisible by 32, but the first dimension is not.

    let family = inspect_checkpoint(&linked_request(temp.path(), "family-conflict"));
    let invalid_family = inspect_checkpoint(&linked_request(temp.path(), "invalid-family.gguf"));
    let overlap = inspect_checkpoint(&linked_request(temp.path(), "overlap.safetensors"));
    let mismatch = inspect_checkpoint(&linked_request(temp.path(), "shape-mismatch.safetensors"));
    let quant = inspect_checkpoint(&linked_request(temp.path(), "bad-quant.gguf"));

    let family_diagnostics = family
        .diagnostics
        .iter()
        .filter(|item| {
            matches!(
                item.code,
                CheckpointDiagnosticCodeV1::FamilyDialectConflict
                    | CheckpointDiagnosticCodeV1::MissingFamilyEvidence
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(family_diagnostics.len(), 1, "{:?}", family.diagnostics);
    assert_eq!(
        family_diagnostics[0].code,
        CheckpointDiagnosticCodeV1::FamilyDialectConflict
    );
    assert_eq!(
        family_diagnostics[0].message,
        "JSON descriptor contains conflicting model-family evidence: flux, qwen-image"
    );
    let invalid_family_diagnostics = invalid_family
        .diagnostics
        .iter()
        .filter(|item| {
            matches!(
                item.code,
                CheckpointDiagnosticCodeV1::FamilyDialectConflict
                    | CheckpointDiagnosticCodeV1::MissingFamilyEvidence
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        invalid_family_diagnostics.len(),
        1,
        "{:?}",
        invalid_family.diagnostics
    );
    assert_eq!(
        invalid_family_diagnostics[0].code,
        CheckpointDiagnosticCodeV1::MissingFamilyEvidence
    );
    assert_eq!(
        invalid_family_diagnostics[0].message,
        "GGUF general.architecture must be a non-empty lowercase ASCII [a-z0-9]+ string"
    );
    for result in [&overlap, &mismatch, &quant] {
        assert!(has_code(
            result,
            CheckpointDiagnosticCodeV1::InvalidTensorRange
        ));
        assert!(result.inventory.records.is_empty());
    }
    assert!(quant
        .diagnostics
        .iter()
        .any(|item| item.message.contains("first dimension 1")));
}

#[test]
fn current_gguf_types_and_truncated_descriptors_receive_exact_validation() {
    let temp = TempDir::new().unwrap();
    write_quantized_gguf(&temp.path().join("mxfp4.gguf"), 32, 1, 39, 17);
    let mut truncated = b"GGUF".to_vec();
    push_u32(&mut truncated, 3);
    fs::write(temp.path().join("truncated.gguf"), truncated).unwrap();

    let modern = inspect_checkpoint(&linked_request(temp.path(), "mxfp4.gguf"));
    let truncated = inspect_checkpoint(&linked_request(temp.path(), "truncated.gguf"));

    assert!(modern.is_runnable(), "{:?}", modern.diagnostics);
    assert_eq!(modern.evidence[0].declared_tensor_bytes, Some(17));
    assert!(has_code(
        &truncated,
        CheckpointDiagnosticCodeV1::TruncatedHeader
    ));
    assert!(!has_code(
        &truncated,
        CheckpointDiagnosticCodeV1::HeaderTooLarge
    ));
}

#[test]
fn gguf_v2_alignment_names_and_quantization_metadata_follow_the_current_spec() {
    let temp = TempDir::new().unwrap();
    write_quantized_gguf_fixture(
        &temp.path().join("valid-v2-alignment-24.gguf"),
        2,
        24,
        Some(2),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    write_quantized_gguf_fixture(
        &temp.path().join("bad-alignment.gguf"),
        3,
        1,
        Some(2),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    write_quantized_gguf_fixture(
        &temp.path().join("missing-quantization-version.gguf"),
        3,
        32,
        None,
        32,
        1,
        39,
        17,
        "model.weight",
    );
    write_quantized_gguf_fixture(
        &temp.path().join("zero-quantization-version.gguf"),
        3,
        32,
        Some(0),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    write_quantized_gguf_fixture(
        &temp.path().join("future-quantization-version.gguf"),
        3,
        32,
        Some(3),
        32,
        1,
        39,
        17,
        "model.weight",
    );

    let wrong_alignment_type = temp.path().join("wrong-alignment-type.gguf");
    let wrong_type_offsets = write_quantized_gguf_fixture(
        &wrong_alignment_type,
        3,
        32,
        Some(2),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    overwrite_u32(&wrong_alignment_type, wrong_type_offsets.alignment - 4, 5); // i32

    let invalid_key = temp.path().join("invalid-metadata-key.gguf");
    write_quantized_gguf_fixture(&invalid_key, 3, 32, Some(2), 32, 1, 39, 17, "model.weight");
    replace_bytes(&invalid_key, b"general.alignment", b"General.alignment");

    let long_tensor_name = "x".repeat(65);
    write_quantized_gguf_fixture(
        &temp.path().join("long-tensor-name.gguf"),
        3,
        32,
        Some(2),
        32,
        1,
        39,
        17,
        &long_tensor_name,
    );

    let valid = inspect_checkpoint(&linked_request(temp.path(), "valid-v2-alignment-24.gguf"));
    assert!(valid.is_runnable(), "{:?}", valid.diagnostics);
    assert_eq!(valid.evidence[0].declared_tensor_bytes, Some(17));
    for path in [
        "bad-alignment.gguf",
        "missing-quantization-version.gguf",
        "zero-quantization-version.gguf",
        "future-quantization-version.gguf",
        "wrong-alignment-type.gguf",
        "invalid-metadata-key.gguf",
        "long-tensor-name.gguf",
    ] {
        let result = inspect_checkpoint(&linked_request(temp.path(), path));
        assert!(
            has_code(&result, CheckpointDiagnosticCodeV1::MalformedMetadata),
            "{path}: {:?}",
            result.diagnostics
        );
        assert!(result.inventory.records.is_empty(), "{path}");
    }
}

#[test]
fn gguf_metadata_mutations_between_exact_byte_passes_are_never_runnable() {
    let temp = TempDir::new().unwrap();
    let alignment_path = temp.path().join("alignment.gguf");
    let alignment_offsets = write_quantized_gguf_fixture(
        &alignment_path,
        3,
        24,
        Some(2),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    let alignment =
        inspect_checkpoint_with_hook(&linked_request(temp.path(), "alignment.gguf"), |event| {
            if event == CheckpointInspectionEventV1::FirstExactBytePassComplete {
                overwrite_u32(&alignment_path, alignment_offsets.alignment, 32);
            }
        });
    assert!(has_code(
        &alignment,
        CheckpointDiagnosticCodeV1::SourceChangedDuringInspection
    ));
    assert!(!alignment.is_runnable());

    let quantization_path = temp.path().join("quantization.gguf");
    let quantization_offsets = write_quantized_gguf_fixture(
        &quantization_path,
        3,
        32,
        Some(2),
        32,
        1,
        39,
        17,
        "model.weight",
    );
    let quantization_offset = quantization_offsets
        .quantization_version
        .expect("quantization version offset");
    let quantization =
        inspect_checkpoint_with_hook(&linked_request(temp.path(), "quantization.gguf"), |event| {
            if event == CheckpointInspectionEventV1::FirstExactBytePassComplete {
                overwrite_u32(&quantization_path, quantization_offset, 1);
            }
        });
    assert!(has_code(
        &quantization,
        CheckpointDiagnosticCodeV1::SourceChangedDuringInspection
    ));
    assert!(!quantization.is_runnable());
}

#[test]
fn oversized_safetensors_headers_preserve_the_discovery_diagnostic() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("oversized.safetensors"),
        100_000_001_u64.to_le_bytes(),
    )
    .unwrap();

    let discovery = discover_checkpoint(&linked_request(temp.path(), "oversized.safetensors"));
    let inspected = inspect_checkpoint(&linked_request(temp.path(), "oversized.safetensors"));
    for diagnostics in [&discovery.diagnostics, &inspected.diagnostics] {
        assert!(diagnostics
            .iter()
            .any(|item| item.code == CheckpointDiagnosticCodeV1::HeaderTooLarge));
        assert!(!diagnostics
            .iter()
            .any(|item| item.code == CheckpointDiagnosticCodeV1::MalformedMetadata));
    }
}

#[test]
fn canonical_link_confinement_accepts_internal_and_rejects_escaping_reparse_targets() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("root");
    let internal_target = root.join("shared/model.safetensors");
    write_safetensors(
        &internal_target,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let internal_dir = root.join("internal/transformer");
    fs::create_dir_all(&internal_dir).unwrap();
    symlink_file(&internal_target, &internal_dir.join("model.safetensors"));

    let external_target = temp.path().join("outside.safetensors");
    write_safetensors(
        &external_target,
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let escaping_dir = root.join("escaping/transformer");
    fs::create_dir_all(&escaping_dir).unwrap();
    symlink_file(&external_target, &escaping_dir.join("model.safetensors"));

    let internal = inspect_checkpoint(&linked_request(&root, "internal"));
    let escaping = inspect_checkpoint(&linked_request(&root, "escaping"));
    assert!(internal.is_runnable(), "{:?}", internal.diagnostics);
    assert!(has_code(
        &escaping,
        CheckpointDiagnosticCodeV1::PathEscapesRoot
    ));
    assert!(escaping.inventory.records.is_empty());
}

#[cfg(unix)]
#[test]
fn unix_directory_symlinks_are_confined_on_linux_and_wsl() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("root");
    let internal_target = root.join("shared/transformer");
    write_safetensors(
        &internal_target.join("model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::create_dir_all(root.join("internal")).unwrap();
    std::os::unix::fs::symlink(&internal_target, root.join("internal/transformer")).unwrap();

    let external_target = temp.path().join("outside/transformer");
    write_safetensors(
        &external_target.join("model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::create_dir_all(root.join("escaping")).unwrap();
    std::os::unix::fs::symlink(&external_target, root.join("escaping/transformer")).unwrap();

    let internal = inspect_checkpoint(&linked_request(&root, "internal"));
    let escaping = inspect_checkpoint(&linked_request(&root, "escaping"));
    assert!(internal.is_runnable(), "{:?}", internal.diagnostics);
    assert!(has_code(
        &escaping,
        CheckpointDiagnosticCodeV1::PathEscapesRoot
    ));
    assert!(escaping.inventory.records.is_empty());
}

#[cfg(windows)]
#[test]
fn windows_directory_junctions_are_confined_as_reparse_points() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("root");
    let internal_target = root.join("shared");
    write_safetensors(
        &internal_target.join("transformer/model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::create_dir_all(&root).unwrap();
    let _internal = WindowsDirectoryJunction::create(&root.join("internal"), &internal_target);

    let external_target = temp.path().join("outside");
    write_safetensors(
        &external_target.join("transformer/model.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    let _escaping = WindowsDirectoryJunction::create(&root.join("escaping"), &external_target);

    let internal = inspect_checkpoint(&linked_request(&root, "internal"));
    let escaping = inspect_checkpoint(&linked_request(&root, "escaping"));
    assert!(internal.is_runnable(), "{:?}", internal.diagnostics);
    assert!(has_code(
        &escaping,
        CheckpointDiagnosticCodeV1::PathEscapesRoot
    ));
    assert!(escaping.inventory.records.is_empty());
}

#[test]
fn descriptor_family_fallback_and_indexed_hidden_shards_are_fail_closed() {
    let temp = TempDir::new().unwrap();
    let descriptor_family = temp.path().join("descriptor-family");
    write_raw_safetensors(
        &descriptor_family.join("transformer/model.safetensors"),
        br#"{"opaque.weight":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#,
        &[0x5a],
    );
    fs::write(
        descriptor_family.join("transformer/config.json"),
        br#"{"_class_name":"FluxTransformer2DModel"}"#,
    )
    .unwrap();

    let hidden = temp.path().join("hidden-index");
    write_safetensors(
        &hidden.join(".model-00001-of-00001.safetensors"),
        &qwen_transformer_entries(),
        Some(json!({"format": "pt"})),
    );
    fs::write(
        hidden.join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":".model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();
    write_raw_safetensors(
        &temp.path().join("misaligned-f4.safetensors"),
        br#"{"opaque.weight":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#,
        &[0x5a],
    );

    let fallback = inspect_checkpoint(&linked_request(temp.path(), "descriptor-family"));
    let hidden = inspect_checkpoint(&linked_request(temp.path(), "hidden-index"));
    let misaligned = inspect_checkpoint(&linked_request(temp.path(), "misaligned-f4.safetensors"));

    assert!(fallback.is_runnable(), "{:?}", fallback.diagnostics);
    assert_eq!(fallback.plans[0].family, "flux");
    assert!(has_code(
        &hidden,
        CheckpointDiagnosticCodeV1::MissingSidecar
    ));
    assert!(hidden.diagnostics.iter().any(|item| item
        .message
        .contains("not discovered as an importable weight artifact")));
    assert!(has_code(
        &misaligned,
        CheckpointDiagnosticCodeV1::InvalidTensorRange
    ));
    assert!(misaligned
        .diagnostics
        .iter()
        .any(|item| item.message.contains("not byte-aligned")));
}
