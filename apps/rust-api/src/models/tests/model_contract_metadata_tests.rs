use super::*;
use sceneworks_core::model_artifacts::external_library::{
    ExternalArtifactRequirement, ExternalLibraryBindingStore, ModelResolution,
};

#[test]
fn import_sources_store_normalized_repo_and_provenance_for_both_wire_shapes() {
    for (body, expected_reference) in [
        (json!({"repo":" Owner/_Model.git "}), "Owner/_Model.git"),
        (
            json!({"source":{"kind":"huggingFace", "repo":" Owner/_Model.git ", "revision":"main"}}),
            "Owner/_Model.git@main",
        ),
    ] {
        let mut payload: ModelImportRequest = serde_json::from_value(body).unwrap();
        let (provenance, _) = normalize_model_import_source(&mut payload).unwrap();
        assert_eq!(payload.repo.as_deref(), Some("Owner/_Model.git"));
        assert_eq!(provenance.reference.as_deref(), Some(expected_reference));
    }
    for repo in [
        "a/b/c",
        "../name",
        "owner/model--bad",
        "owner/bad..name",
        "owner/model \n name",
    ] {
        assert!(validate_huggingface_repo(repo).is_err(), "{repo:?}");
    }
}

#[tokio::test]
async fn selected_lookup_refreshes_only_the_selected_current_row_and_keeps_snapshot_immutable() {
    let _env = crate::tests::support::isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let (_, state) =
        crate::create_app_with_state(crate::tests::support::test_settings(&temp)).unwrap();
    let library = temp.path().join("external-library");
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let source = library
        .join("models--owner--model")
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("model.safetensors"), b"metadata fixture").unwrap();
    let requirement = ExternalArtifactRequirement {
        repository: "owner/model".to_owned(),
        revision: Some(revision.to_owned()),
        variant: "default".to_owned(),
        files: vec![PathBuf::from("model.safetensors")],
        is_primary: true,
    };
    let store = ExternalLibraryBindingStore::new(&state.settings.data_dir).unwrap();
    let (binding, _) = store
        .bind_or_probe_validated(&library, std::slice::from_ref(&requirement))
        .unwrap();
    let resolution =
        ModelResolution::external_ready(library.clone(), binding, vec![requirement]).unwrap();
    let selected = json!({"id":"selected", "modelResolution":resolution});
    // A corrupt unselected carrier must neither be refreshed nor fail the selected lookup.
    let snapshot = Arc::new(vec![
        json!({"id":"other", "modelResolution":{"invalid":true}}),
        selected.clone(),
    ]);
    {
        let mut cache = state.model_catalog_cache.state.lock();
        cache.snapshot = Some((
            cache.generation,
            crate::generation::enqueue_backend(&state),
            snapshot.clone(),
        ));
    }
    let detached = temp.path().join("detached");
    std::fs::rename(&library, &detached).unwrap();
    let offline = require_model_by_id(&state, "selected").await.unwrap();
    assert_eq!(
        offline["modelAvailability"],
        "installed_external_unavailable"
    );
    std::fs::rename(detached, library).unwrap();
    let mut online = find_model_by_id(&state, "selected").await.unwrap().unwrap();
    assert_eq!(online["modelAvailability"], "external_ready");
    online["callerMutation"] = json!(true);
    assert_eq!(snapshot[1], selected);
    let current = model_catalog_snapshot(&state).await.unwrap();
    assert!(Arc::ptr_eq(&snapshot, &current));
    assert!(find_model_by_id(&state, "missing").await.unwrap().is_none());
    let missing = require_model_by_id(&state, "missing").await.unwrap_err();
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.detail, "Model not found");
}
