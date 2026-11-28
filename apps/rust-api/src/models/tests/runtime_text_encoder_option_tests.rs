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

/// The whole-catalog encoder pass must leave `GET /models` byte-for-byte what the per-model
/// computation produced: same rows, same keys in the same insertion order, same option lists.
/// The reference below is the pre-batching loop verbatim, run against the same snapshot.
#[tokio::test]
async fn sized_catalog_is_byte_identical_to_the_per_model_encoder_computation() {
    let _env = crate::tests::support::isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    // The real embedded builtin catalog (download contexts, variants, commercial-use pointers,
    // memory guidance), so every field the loop writes is exercised.
    let manifests = temp.path().join("config/manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        serde_json::to_vec(
            &sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    crate::tests::support::write_empty_sibling_manifests(&manifests);
    let (_, state) =
        crate::create_app_with_state(crate::tests::support::test_settings(&temp)).unwrap();
    *state
        .test_hooks
        .model_size_estimate_disabled_override
        .lock() = Some(true);
    // A row whose seeds point at a staged directory, so source collection has real work to do.
    let staged = state.settings.data_dir.join("staged-encoder");
    std::fs::create_dir_all(staged.join("text_encoder")).unwrap();
    write_tiny_safetensors(&staged.join("text_encoder/model.safetensors"));
    std::fs::write(staged.join("text_encoder/config.json"), b"{}").unwrap();
    let mut snapshot = model_catalog_snapshot(&state)
        .await
        .unwrap()
        .as_ref()
        .clone();
    assert!(
        snapshot.len() > 50,
        "the builtin catalog seeds the snapshot"
    );
    snapshot.push(json!({
        "id": "staged_row",
        "name": "Staged",
        "type": "image",
        "catalogScope": "imported",
        "installedPath": staged,
        "adapter": "ltx_video",
        "variants": [],
    }));
    {
        let mut cache = state.model_catalog_cache.state.lock();
        cache.snapshot = Some((
            cache.generation,
            crate::generation::enqueue_backend(&state),
            Arc::new(snapshot),
        ));
    }

    let batched = model_catalog_sized(&state).await.unwrap();

    let size_estimates = estimate_current_model_catalog_sizes(&state).await.unwrap();
    let mut reference = model_catalog_snapshot(&state)
        .await
        .unwrap()
        .as_ref()
        .clone();
    crate::model_sources::refresh_live_external_availability(
        &state.settings.data_dir,
        &mut reference,
    )
    .await
    .unwrap();
    let selection_catalog = reference.clone();
    for model in &mut reference {
        let context = model_download_context(model).unwrap();
        let live_estimate = context.as_ref().and_then(|context| {
            size_estimates
                .get(&(context.repo.clone(), context.files.clone()))
                .copied()
                .flatten()
        });
        apply_model_catalog_size_fields(model, context.as_ref(), live_estimate).unwrap();
        apply_runtime_text_encoder_options(
            model,
            &selection_catalog,
            &state.settings.data_dir,
            &state.settings.external_model_roots,
        )
        .unwrap();
        annotate_commercial_use_alternatives(model, &selection_catalog);
    }

    assert_eq!(batched.len(), reference.len());
    assert_eq!(
        serde_json::to_vec(&batched).unwrap(),
        serde_json::to_vec(&reference).unwrap(),
        "the batched encoder pass changed the public catalog bytes"
    );
    // A second, memo-warm request is identical too.
    assert_eq!(
        serde_json::to_vec(&model_catalog_sized(&state).await.unwrap()).unwrap(),
        serde_json::to_vec(&reference).unwrap(),
    );
}
