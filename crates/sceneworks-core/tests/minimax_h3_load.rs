use gen_core::{LoadSpec, WeightsSource};
use sceneworks_core::minimax_h3_load::PreparedH3Load;
use std::path::Path;

fn fixture(root: &Path) -> LoadSpec {
    for component in [
        "transformer",
        "transformer_ref",
        "text_encoder",
        "vae",
        "audio_vae",
        "tokenizer",
        "FL2VA/audio_vae",
    ] {
        let dir = root.join(component);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{\"nonDefault\":7}").unwrap();
        std::fs::write(dir.join("model.safetensors"), component.as_bytes()).unwrap();
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()));
    spec.components.insert(
        "transformer".into(),
        WeightsSource::Dir(root.join("transformer")),
    );
    spec.components.insert(
        "text_encoder".into(),
        WeightsSource::Dir(root.join("text_encoder")),
    );
    spec
}

#[test]
fn imported_h3_prepares_both_tasks_shared_configs_and_exact_adapter_files() {
    let temp = tempfile::tempdir().unwrap();
    let mut spec = fixture(temp.path());
    let adapter = temp.path().join("style.safetensors");
    std::fs::write(&adapter, b"non-default adapter").unwrap();
    spec.adapters.push(gen_core::AdapterSpec::new(
        adapter.clone(),
        0.375,
        gen_core::AdapterKind::Lokr,
    ));
    let receipt = PreparedH3Load::prepare(&mut spec).unwrap();
    assert_eq!(receipt.files().len(), 15);
    assert!(spec.prepared_file_pins().is_finalized());
    assert_eq!(
        spec.components
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["text_encoder", "transformer"],
        "preparation must not invent an unsupported transformer_ref native component"
    );
    assert!(spec
        .prepared_file_pins()
        .get(&std::path::absolute(temp.path().join("transformer_ref/model.safetensors")).unwrap())
        .is_some());
    assert!(spec
        .prepared_file_pins()
        .get(&std::path::absolute(temp.path().join("FL2VA/audio_vae/config.json")).unwrap())
        .is_some());
    receipt.validate(&spec).unwrap();
    let records = receipt.prepared_sources_json();
    assert_eq!(records.as_array().unwrap().len(), 15);
    assert!(spec
        .prepared_file_pins()
        .get(&std::path::absolute(&adapter).unwrap())
        .is_some());
    assert!(records
        .as_array()
        .unwrap()
        .iter()
        .all(|row| row["sha256"].as_str().unwrap().len() == 64));
}

#[test]
fn every_consumed_component_and_table_changes_prepared_cache_identity() {
    for component in [
        "transformer",
        "transformer_ref",
        "text_encoder",
        "vae",
        "audio_vae",
        "tokenizer",
        "FL2VA/audio_vae",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut before = fixture(temp.path());
        let receipt = PreparedH3Load::prepare(&mut before).unwrap();
        std::fs::write(
            temp.path().join(component).join("model.safetensors"),
            b"different content",
        )
        .unwrap();
        assert!(
            receipt.validate(&before).is_err(),
            "{component} stale receipt must fail"
        );
        let mut after = fixture_spec(temp.path());
        PreparedH3Load::prepare(&mut after).unwrap();
        assert_ne!(
            before.prepared_file_pins(),
            after.prepared_file_pins(),
            "{component} must miss the warm key"
        );
    }
}

fn fixture_spec(root: &Path) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()));
    spec.components.insert(
        "transformer".into(),
        WeightsSource::Dir(root.join("transformer")),
    );
    spec.components.insert(
        "text_encoder".into(),
        WeightsSource::Dir(root.join("text_encoder")),
    );
    spec
}

#[test]
fn directory_membership_rejects_added_shards_and_removed_configs_without_rehashing() {
    for remove in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let mut spec = fixture(temp.path());
        let receipt = PreparedH3Load::prepare(&mut spec).unwrap();
        if remove {
            std::fs::remove_file(temp.path().join("tokenizer/config.json")).unwrap();
        } else {
            std::fs::write(
                temp.path().join("transformer/unplanned.safetensors"),
                b"foreign",
            )
            .unwrap();
        }
        assert!(receipt.validate(&spec).is_err());
    }
}

#[test]
fn staged_receipt_clone_and_drop_preserve_seals_and_never_remove_owned_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let mut spec = fixture(temp.path());
    let receipt = PreparedH3Load::prepare(&mut spec).unwrap();
    let staged = spec
        .clone()
        .with_offload_policy(gen_core::OffloadPolicy::Sequential)
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    receipt.validate(&staged).unwrap();
    assert_eq!(spec.prepared_file_pins(), staged.prepared_file_pins());
    drop(staged);
    drop(receipt);
    assert_eq!(
        std::fs::read(temp.path().join("transformer/model.safetensors")).unwrap(),
        b"transformer"
    );
    spec.validate_prepared_file_pins().unwrap();
}

#[test]
fn changed_schedule_config_cannot_use_the_old_prepared_load() {
    let temp = tempfile::tempdir().unwrap();
    let mut spec = fixture(temp.path());
    let receipt = PreparedH3Load::prepare(&mut spec).unwrap();
    std::fs::write(
        temp.path().join("transformer/config.json"),
        b"{\"schedule\":17}",
    )
    .unwrap();
    assert!(receipt.validate(&spec).is_err());
    let mut after = fixture_spec(temp.path());
    PreparedH3Load::prepare(&mut after).unwrap();
    assert_ne!(spec.prepared_file_pins(), after.prepared_file_pins());
}

#[test]
fn preparation_cannot_reauthorize_changed_source_or_added_unplanned_shard() {
    for add in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let mut spec = fixture(temp.path());
        let source = temp.path().join("transformer/model.safetensors");
        let pin = gen_core::PinnedWeightsFile::pin(&source).unwrap();
        let authority = sceneworks_core::minimax_h3_load::H3SourceAuthority::new(
            vec![pin],
            temp.path().join("transformer"),
            ["model.safetensors".to_owned()].into_iter().collect(),
        )
        .unwrap();
        if add {
            std::fs::write(
                temp.path().join("transformer/unplanned.safetensors"),
                b"foreign",
            )
            .unwrap();
        } else {
            std::fs::write(&source, b"replacement not in the approved plan").unwrap();
        }
        assert!(PreparedH3Load::prepare_authorized(&mut spec, Some(authority)).is_err());
    }
}

#[test]
fn split_roots_keep_reference_seals_without_an_unsupported_native_slot() {
    let shared = tempfile::tempdir().unwrap();
    let backbone = tempfile::tempdir().unwrap();
    let mut spec = fixture(shared.path());
    fixture(backbone.path());
    spec.components.insert(
        "transformer".into(),
        WeightsSource::Dir(backbone.path().join("transformer")),
    );
    let receipt = PreparedH3Load::prepare(&mut spec).unwrap();
    assert!(!spec.components.contains_key("transformer_ref"));
    assert!(receipt.files().iter().any(|pin| pin.loader_path()
        == std::path::absolute(backbone.path().join("transformer_ref/model.safetensors"))
            .unwrap()));
    assert!(spec
        .prepared_file_pins()
        .get(
            &std::path::absolute(backbone.path().join("transformer_ref/model.safetensors"))
                .unwrap()
        )
        .is_none());
    receipt.validate(&spec).unwrap();
    std::fs::write(
        backbone.path().join("transformer_ref/model.safetensors"),
        b"changed companion",
    )
    .unwrap();
    assert!(receipt.validate(&spec).is_err());
}

#[test]
fn blocking_reader_retains_lease_after_async_owner_is_dropped() {
    let temp = tempfile::tempdir().unwrap();
    fixture(temp.path());
    let lock_path = temp.path().join("source.lock");
    let open = || {
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap()
    };
    let source = sceneworks_core::file_lock::FileLock::shared(open()).unwrap();
    let authority = sceneworks_core::minimax_h3_load::H3SourceAuthority::new(
        vec![
            gen_core::PinnedWeightsFile::pin(temp.path().join("transformer/model.safetensors"))
                .unwrap(),
        ],
        temp.path().join("transformer"),
        ["model.safetensors".to_owned()].into_iter().collect(),
    )
    .unwrap()
    .with_leases(source, None);
    let reader = authority.clone();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        reader.validate().unwrap();
        started_tx.send(()).unwrap();
        finish_rx.recv().unwrap();
        reader.validate().unwrap();
        drop(reader);
    });
    started_rx.recv().unwrap();
    drop(authority); // Async request aborted, blocking reader still running.
    assert!(sceneworks_core::file_lock::FileLock::try_exclusive(open()).is_err());
    finish_tx.send(()).unwrap();
    thread.join().unwrap();
    assert!(sceneworks_core::file_lock::FileLock::try_exclusive(open()).is_ok());
}
