use super::*;
#[cfg(target_os = "macos")]
use crate::tests::support::isolate_hf_cache;

#[test]
fn catalog_field_is_path_free_and_default_can_be_omitted() {
    let mut model = json!({ "id": "future_image" });
    set_runtime_text_encoder_options(
        &mut model,
        vec![
            json!({
                "id": "default",
                "label": "Model encoder (default)",
                "description": "Uses the bundled encoder.",
                "isDefault": true
            }),
            json!({
                "id": "text_encoder_0123456789abcdef0123456789abcdef",
                "label": "Alternate",
                "description": "Contract validated.",
                "isDefault": false
            }),
        ],
    )
    .unwrap();
    let serialized = serde_json::to_string(&model["textEncoderOptions"]).unwrap();
    for forbidden in ["path", "repo", "revision", "files", "download"] {
        assert!(!serialized.contains(forbidden), "{forbidden}");
    }

    set_runtime_text_encoder_options(&mut model, Vec::new()).unwrap();
    assert!(model.get("textEncoderOptions").is_none());
}

#[test]
fn embedded_builtin_lookup_preserves_entries_and_isolates_returned_clones() {
    let predicate = |entry: &Value| entry["id"] == "moss_image_base";
    let expected = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .unwrap()["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| predicate(entry))
        .unwrap()
        .clone();
    let mut first = embedded_builtin_catalog_entry(predicate).unwrap().unwrap();
    assert_eq!(
        first, expected,
        "the complete cross-platform entry is retained"
    );
    first["downloads"] = json!([]);
    assert_eq!(
        embedded_builtin_catalog_entry(predicate).unwrap(),
        Some(expected)
    );
    assert!(embedded_builtin_catalog_entry(|_| false).unwrap().is_none());
}

#[test]
fn install_health_uses_full_core_safetensors_validation() {
    let temp = tempfile::tempdir().unwrap();
    let component = temp.path().join("transformer");
    std::fs::create_dir(&component).unwrap();
    let path = component.join("model.safetensors");
    write_tiny_safetensors(&path);
    assert!(diffusers_component_safetensors_are_valid(
        temp.path(),
        "transformer"
    ));
    for raw in [
        br#"{}"#.as_slice(),
        br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,1]}}"#,
    ] {
        let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(raw);
        bytes.push(0);
        std::fs::write(&path, bytes).unwrap();
        assert!(!diffusers_component_safetensors_are_valid(
            temp.path(),
            "transformer"
        ));
    }
}

fn write_tiny_safetensors(path: &FsPath) {
    let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.push(0);
    std::fs::write(path, bytes).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn legacy_ltx_fallback_refreshes_when_alternate_becomes_valid_or_corrupt() {
    let _env = isolate_hf_cache();
    let data_dir = tempfile::tempdir().unwrap();
    let repo = sceneworks_core::hf_home::huggingface_repo_cache_path(
        data_dir.path(),
        "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit",
    )
    .unwrap();
    let snapshot = repo
        .join("snapshots")
        .join("0123456789abcdef0123456789abcdef01234567");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(
        snapshot.join("config.json"),
        br#"{"model_type":"gemma3_text"}"#,
    )
    .unwrap();
    std::fs::write(
        snapshot.join("tokenizer.json"),
        br#"{"model":{"type":"BPE"}}"#,
    )
    .unwrap();
    write_tiny_safetensors(&snapshot.join("model.safetensors"));

    let mut model = json!({ "id": "ltx_2_3", "adapter": "ltx_video" });
    let catalog = vec![model.clone()];
    apply_runtime_text_encoder_options(&mut model, &catalog, data_dir.path(), &[]).unwrap();
    assert_eq!(model["textEncoderOptions"].as_array().unwrap().len(), 2);

    std::fs::write(snapshot.join("model.safetensors"), b"truncated").unwrap();
    apply_runtime_text_encoder_options(&mut model, &catalog, data_dir.path(), &[]).unwrap();
    let options = model["textEncoderOptions"].as_array().unwrap();
    assert_eq!(options.len(), 1);
    assert_eq!(options[0]["id"], "default");

    std::fs::remove_dir_all(&repo).unwrap();
    apply_runtime_text_encoder_options(&mut model, &catalog, data_dir.path(), &[]).unwrap();
    assert_eq!(model["textEncoderOptions"].as_array().unwrap().len(), 1);
}
