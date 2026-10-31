//! Dense MiniMax-H3 fine-tune import identification and planning (sc-26680, epic 26679).
//!
//! Drives the REAL header classifier, import gate, inspector and plan store over a synthetic H3
//! transformer component directory — fixture-scale bytes, the published key grammar — and asserts
//! that each ownership mode persists a plan the shared assembly validator binds.

use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};

use sceneworks_core::base_weights::{
    classify_base_header, detect_base_weight_file, import_detection_supported,
    imported_model_primary_weight_file, is_minimax_h3_transformer_dir, BaseWeightDetection,
    ComponentRole, QuantFormat,
};
use sceneworks_core::checkpoint_import::ManagedProvenanceV1;
use sceneworks_core::checkpoint_ingest::ManagedIngest;
use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
use sceneworks_core::minimax_h3_import::{MiniMaxH3ImportAssembly, MINIMAX_H3_IMPORT_FAMILY};

/// The published DiT key grammar, trimmed to one block: the refiner + audio head that identify the
/// family, the per-block `adaln_proj.linear` that Ideogram's marker would otherwise claim, and the
/// mixed-precision f32 heads beside the bf16 stack.
const H3_DIT_KEYS: &[(&str, &str)] = &[
    ("transformer_blocks.0.adaln_proj.linear.weight", "BF16"),
    ("transformer_blocks.0.attn.to_qkv.weight", "BF16"),
    ("transformer_blocks.0.ff.net.0.proj.weight", "BF16"),
    ("token_refiner.refiner_blocks.0.attn.to_qkv.weight", "BF16"),
    ("token_refiner.final_norm.weight", "BF16"),
    ("audio_proj_in.weight", "F32"),
    ("audio_proj_out.weight", "F32"),
    ("context_embedder.weight", "F32"),
    ("proj_in.weight", "BF16"),
    ("proj_out.weight", "BF16"),
];

fn write_safetensors(path: &Path, entries: &[(&str, &str)], fill: u8) {
    let mut header = Map::new();
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

fn write_h3_dit_dir(dir: &Path) {
    let mut config = sceneworks_core::minimax_h3_codec::H3Config::published().0;
    for (key, value) in [
        ("num_layers", 1),
        ("num_refiner_layers", 1),
        ("num_attention_heads", 2),
        ("attention_head_dim", 6),
        ("hidden_size", 8),
        ("ffn_dim", 8),
        ("in_channels", 2),
        ("audio_in_channels", 2),
        ("text_dim", 8),
        ("freq_dim", 4),
        ("time_embed_hidden_dim", 8),
        ("time_embed_dim", 4),
        ("rope_freq_dim", 1),
    ] {
        config[key] = json!(value);
    }
    let shapes = sceneworks_core::minimax_h3_codec::H3Config(config.clone())
        .shapes()
        .unwrap();
    let mut header = Map::new();
    let mut offset = 0_u64;
    for (name, shape) in shapes {
        let size = shape.iter().product::<usize>() as u64 * 2;
        header.insert(
            name,
            json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,offset+size]}),
        );
        offset += size;
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, 0x3a);
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("diffusion_pytorch_model.safetensors"), bytes).unwrap();
    fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

fn header_of(entries: &[(&str, &str)]) -> Value {
    let mut map = Map::new();
    for (name, dtype) in entries {
        map.insert((*name).to_owned(), json!({ "dtype": dtype }));
    }
    Value::Object(map)
}

#[test]
fn an_h3_dit_is_identified_by_its_tensors_not_by_its_name() {
    let BaseWeightDetection::Recognized(verdict) = classify_base_header(&header_of(H3_DIT_KEYS))
    else {
        panic!("an H3 DiT header must be recognized");
    };
    assert_eq!(verdict.family.as_deref(), Some(MINIMAX_H3_IMPORT_FAMILY));
    assert_eq!(verdict.component, ComponentRole::Transformer);
    assert_eq!(verdict.quant, QuantFormat::Bf16);

    // The family is a property of the bytes: an arbitrary file name changes nothing.
    let dir = tempfile::tempdir().unwrap();
    let renamed = dir.path().join("singularity-merge-final.safetensors");
    write_safetensors(&renamed, H3_DIT_KEYS, 0x11);
    let BaseWeightDetection::Recognized(renamed_verdict) =
        detect_base_weight_file(&renamed).unwrap()
    else {
        panic!("renamed H3 DiT must still be recognized");
    };
    assert_eq!(renamed_verdict.family.as_deref(), Some("minimax-h3"));

    // Without the refiner + audio head, the shared `adaln_proj.` marker is Ideogram's — which is
    // exactly the misclassification the H3 arm's ordering removes.
    let without_markers: Vec<_> = H3_DIT_KEYS
        .iter()
        .copied()
        .filter(|(name, _)| !name.starts_with("token_refiner") && !name.starts_with("audio_proj"))
        .collect();
    let BaseWeightDetection::Recognized(other) = classify_base_header(&header_of(&without_markers))
    else {
        panic!("still a transformer");
    };
    assert_eq!(other.family.as_deref(), Some("ideogram"));
}

#[test]
fn the_import_gate_admits_only_a_complete_dense_h3_component_directory() {
    let dir = tempfile::tempdir().unwrap();
    let h3 = dir.path().join("h3-finetune");
    write_h3_dit_dir(&h3);
    assert!(is_minimax_h3_transformer_dir(&h3));
    let weights = imported_model_primary_weight_file(&h3).expect("the DiT is the primary file");
    assert_eq!(
        weights.file_name().and_then(|name| name.to_str()),
        Some("diffusion_pytorch_model.safetensors")
    );
    import_detection_supported(&detect_base_weight_file(&weights).unwrap())
        .expect("a dense bf16 H3 DiT is importable");

    // Every dense precision is admitted; a packed tier is not.
    for (dtype, admitted) in [("F16", true), ("F32", true), ("U8", false)] {
        let entries: Vec<_> = H3_DIT_KEYS.iter().map(|(name, _)| (*name, dtype)).collect();
        let verdict = classify_base_header(&header_of(&entries));
        assert_eq!(
            import_detection_supported(&verdict).is_ok(),
            admitted,
            "{dtype}: {verdict:?}"
        );
    }

    // A config naming another class is not an H3 component directory.
    let wrong_class = dir.path().join("wrong-class");
    write_h3_dit_dir(&wrong_class);
    fs::write(
        wrong_class.join("config.json"),
        json!({ "_class_name": "WanTransformer3DModel" }).to_string(),
    )
    .unwrap();
    assert!(!is_minimax_h3_transformer_dir(&wrong_class));

    // A second weight file would be mapped by the provider without being planned or bound.
    let stray = dir.path().join("stray-shard");
    write_h3_dit_dir(&stray);
    write_safetensors(&stray.join("extra.safetensors"), H3_DIT_KEYS, 0x01);
    assert!(!is_minimax_h3_transformer_dir(&stray));
    // `imported_model_primary_weight_file` still names the diffusers weight file here (the
    // Mage-Flow directory shape shares the file names), so the H3 directory predicate above is the
    // gate the importer and the API must — and do — apply on top of it.

    // No config at all: the provider cannot derive the geometry.
    let no_config = dir.path().join("no-config");
    write_h3_dit_dir(&no_config);
    fs::remove_file(no_config.join("config.json")).unwrap();
    assert!(!is_minimax_h3_transformer_dir(&no_config));
}

fn provenance() -> ManagedProvenanceV1 {
    ManagedProvenanceV1 {
        source: "local".to_owned(),
        ..ManagedProvenanceV1::default()
    }
}

#[test]
fn managed_and_linked_imports_persist_a_plan_the_h3_assembly_binds() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let library_dir = fs::canonicalize(library.path()).unwrap();
    let source = library_dir.join("my-h3");
    write_h3_dit_dir(&source);
    let store = CheckpointPlanStore::open(data.path());

    // MANAGED: the transactional ingest the model-import job runs.
    let ingest = ManagedIngest::begin(&store, "my_h3", provenance()).unwrap();
    for file in ["diffusion_pytorch_model.safetensors", "config.json"] {
        ingest.stage_copy_file(&source.join(file), file).unwrap();
    }
    let install = ingest
        .finalize("diffusion_pytorch_model.safetensors", None)
        .expect("a dense H3 DiT commits");
    assert!(is_minimax_h3_transformer_dir(&install.install_path));
    let resolved = store
        .resolve(&install.checkpoint_id)
        .expect("the persisted plan resolves");
    assert_eq!(resolved.family(), MINIMAX_H3_IMPORT_FAMILY);
    let managed = MiniMaxH3ImportAssembly::from_declared_plan(
        &resolved.plan,
        serde_json::json!({"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"})
            .as_object()
            .unwrap(),
    )
    .expect("complete assembly");
    assert_eq!(
        resolved
            .plan
            .layers
            .iter()
            .find(|layer| layer.layer_id == managed.transformer_layer_id)
            .map(|layer| layer.role.as_str()),
        Some("transformer")
    );

    // LINKED: the same directory referenced in place. Its plan also carries the config descriptor,
    // and the assembly binds it as part of the DiT component rather than refusing it.
    let root = store.approve_root(&library_dir).unwrap();
    let compiled = store
        .compile_linked(&root.root_id, "my-h3")
        .expect("the linked H3 directory compiles");
    assert_eq!(compiled.plan.family, MINIMAX_H3_IMPORT_FAMILY);
    let linked = MiniMaxH3ImportAssembly::from_declared_plan(
        &compiled.plan,
        serde_json::json!({"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"})
            .as_object()
            .unwrap(),
    )
    .expect("complete assembly");
    assert_eq!(linked.descriptor_layer_ids.len(), 1, "{linked:?}");
}

#[test]
fn h3_interruption_retry_drift_relink_duplicates_and_leases_preserve_ownership() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("custom");
    write_h3_dit_dir(&source);
    let original = fs::read(source.join("diffusion_pytorch_model.safetensors")).unwrap();
    let store = CheckpointPlanStore::open(data.path());
    let root = store.approve_root(library.path()).unwrap();
    let linked = store.compile_linked(&root.root_id, "custom").unwrap();
    // Drop an interrupted transfer: neither published state nor partial install can survive.
    {
        let ingest = ManagedIngest::begin(&store, "retry_h3", provenance()).unwrap();
        ingest
            .stage_copy_file(&source.join("config.json"), "config.json")
            .unwrap();
    }
    assert!(!store.install_dir("retry_h3").unwrap().exists());
    assert!(store.record("managed/retry_h3").is_err());
    let ingest = ManagedIngest::begin(&store, "retry_h3", provenance()).unwrap();
    for file in ["config.json", "diffusion_pytorch_model.safetensors"] {
        ingest.stage_copy_file(&source.join(file), file).unwrap();
    }
    let managed = ingest
        .finalize("diffusion_pytorch_model.safetensors", None)
        .unwrap();
    assert_eq!(
        managed.duplicate_checkpoint_ids(),
        std::slice::from_ref(&linked.checkpoint_id)
    );
    // Shared read leases admit concurrent renders; unrelated sources do not contend.
    let lease = store.lease_source(&managed.checkpoint_id).unwrap();
    let second = store.lease_source(&managed.checkpoint_id).unwrap();
    let unrelated = store.lock_source_for_removal("managed/unrelated").unwrap();
    assert!(store
        .remove_managed("retry_h3")
        .unwrap_err()
        .to_string()
        .contains("in use"));
    assert_eq!(
        fs::read(
            managed
                .install_path
                .join("diffusion_pytorch_model.safetensors")
        )
        .unwrap(),
        original
    );
    assert!(store.resolve(&managed.checkpoint_id).is_ok());
    drop((lease, second, unrelated));
    assert!(store.remove_managed("retry_h3").unwrap());
    assert_eq!(
        fs::read(source.join("diffusion_pytorch_model.safetensors")).unwrap(),
        original
    );
    // Drift cannot remain ready, even if the container/header and file length are unchanged.
    let mut changed = original.clone();
    *changed.last_mut().unwrap() ^= 1;
    fs::write(source.join("diffusion_pytorch_model.safetensors"), &changed).unwrap();
    assert_eq!(
        store.resolve(&linked.checkpoint_id).unwrap_err().code(),
        "source-drifted"
    );
    let rescanned = store.compile_linked(&root.root_id, "custom").unwrap();
    assert_ne!(rescanned.plan.plan_id, linked.plan.plan_id);
    let moved = library.path().join("relocated");
    fs::create_dir(&moved).unwrap();
    fs::rename(&source, moved.join("custom")).unwrap();
    store.relink_root(&root.root_id, &moved).unwrap();
    assert!(store.resolve(&linked.checkpoint_id).is_ok());
    let lease = store.lease_source(&linked.checkpoint_id).unwrap();
    assert!(store.remove_root(&root.root_id).is_err());
    assert!(store.record(&linked.checkpoint_id).is_ok());
    drop(lease);
    store.remove_root(&root.root_id).unwrap();
    assert_eq!(
        fs::read(moved.join("custom/diffusion_pytorch_model.safetensors")).unwrap(),
        changed
    );
}

#[test]
fn compatible_base_downloads_never_replace_imported_task_partitions() {
    use sceneworks_core::minimax_h3_import::import_component_downloads;
    let base = |id: &str, companion: &str| {
        json!({
            "id":id,"family":"minimax-h3","downloads":[
                {"provider":"huggingface","repo":"fixture/tiers","variant":"q4","files":["q4/primary/*"]},
                {"provider":"huggingface","repo":"fixture/tiers","coRequisite":true,"variant":"q4","files":[format!("q4/{companion}/*")]},
                {"provider":"huggingface","repo":"fixture/tiers","coRequisite":true,"variant":"q8","componentId":"text_encoder","files":["q8/text_encoder/*"]},
                {"provider":"huggingface","repo":"fixture/tiers","coRequisite":true,"variant":"q4","componentId":"text_encoder","files":["q4/text_encoder/*"]},
                {"provider":"huggingface","repo":"fixture/shared","coRequisite":true,"componentId":"tokenizer","files":["tokenizer/*"]},
                {"provider":"huggingface","repo":"fixture/optional","coRequisite":true,"required":"soft","files":["predictor.bin"]}
            ]
        })
    };
    for (partition, id, companion) in [
        ("fl2va", "minimax_h3", "transformer_ref"),
        ("ref2va", "minimax_h3_ref", "transformer"),
        ("hybrid", "minimax_h3", "transformer_ref"),
    ] {
        let entry = json!({"baseModel":id,"minimaxH3Partition":partition});
        let rows = import_component_downloads(
            entry.as_object().unwrap(),
            &base(id, companion),
            Some("q4"),
            "macos",
        )
        .unwrap();
        assert_eq!(rows.len(), if partition == "hybrid" { 2 } else { 3 });
        let encoded = json!(rows).to_string();
        assert!(!encoded.contains("primary"));
        assert!(!encoded.contains("q8"));
        assert!(!encoded.contains("predictor"));
        assert_eq!(encoded.contains(companion), partition != "hybrid");
        assert!(
            import_component_downloads(
                entry.as_object().unwrap(),
                &base(id, companion),
                Some("q8"),
                "macos"
            )
            .is_err(),
            "a stray encoder row cannot establish a compatible base tier or omit its companion"
        );
    }
}

#[test]
fn shared_h3_component_globs_cannot_hide_incomplete_weights_or_constructor_files() {
    use sceneworks_core::minimax_h3_import::import_component_files_complete;
    let temp = tempfile::tempdir().unwrap();
    let put = |relative: &str, bytes: &[u8]| {
        let path = temp.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    };
    let row = |id: &str, subdir: &str| json!({"componentId":id,"subdir":subdir,"files":[format!("{subdir}/*")]});
    let encoder = row("text_encoder", "q4/text_encoder");
    put("q4/text_encoder/config.json", b"{}");
    put(
        "q4/text_encoder/model.safetensors.index.json",
        br#"{"weight_map":{"a":"part1.safetensors","b":"part2.safetensors"}}"#,
    );
    put("q4/text_encoder/part1.safetensors", b"one");
    assert!(!import_component_files_complete(temp.path(), &encoder));
    put("q4/text_encoder/part2.safetensors", b"two");
    assert!(import_component_files_complete(temp.path(), &encoder));
    put(
        "q4/text_encoder/model.safetensors.index.json",
        br#"{"weight_map":{"a":"../foreign.safetensors"}}"#,
    );
    put("q4/foreign.safetensors", b"foreign");
    assert!(!import_component_files_complete(temp.path(), &encoder));
    for (id, subdir) in [("video_vae", "vae"), ("audio_vae", "audio_vae")] {
        let component = row(id, subdir);
        put(&format!("{subdir}/config.json"), b"{}");
        assert!(!import_component_files_complete(temp.path(), &component));
        put(&format!("{subdir}/weights.safetensors"), b"weights");
        assert!(import_component_files_complete(temp.path(), &component));
    }
    let tokenizer = row("tokenizer", "tokenizer");
    put("tokenizer/tokenizer.json", b"{}");
    assert!(!import_component_files_complete(temp.path(), &tokenizer));
    put("tokenizer/tokenizer_config.json", b"{}");
    assert!(import_component_files_complete(temp.path(), &tokenizer));
    let constructor = json!({"componentId":"fl2va_audio_vae_config","subdir":"FL2VA/audio_vae","files":["FL2VA/audio_vae/config.json","FL2VA/audio_vae/config.yaml","FL2VA/audio_vae/metadata.json"]});
    put("FL2VA/audio_vae/config.json", b"{}");
    put("FL2VA/audio_vae/config.yaml", b"fixture: true");
    assert!(!import_component_files_complete(temp.path(), &constructor));
    put("FL2VA/audio_vae/metadata.json", b"{}");
    assert!(import_component_files_complete(temp.path(), &constructor));
}

fn base_config_context(config: Value) -> std::collections::BTreeMap<String, Value> {
    [("transformerConfig".into(), sceneworks_core::minimax_h3_import::config_binding(config,
        json!({"kind":"selected-base","baseModel":"minimax_h3","repo":"fixture/h3","revision":"1111111111111111111111111111111111111111","path":"bf16/transformer/config.json","componentTier":"bf16"})))].into()
}

#[test]
fn weight_only_config_is_immutable_semantic_context_across_ownership_restart_and_normalization() {
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    write_h3_dit_dir(library.path());
    let mut config: Value =
        serde_json::from_slice(&fs::read(library.path().join("config.json")).unwrap()).unwrap();
    config["rope_theta"] = json!(123456.0);
    config["future_provider_field"] = json!({"schedule":[0.125, 0.375],"precision":"trained-bf16"});
    fs::remove_file(library.path().join("config.json")).unwrap();
    let file = "diffusion_pytorch_model.safetensors";
    let original = fs::read(library.path().join(file)).unwrap();
    let store = CheckpointPlanStore::open(data.path());
    let root = store.approve_root(library.path()).unwrap();
    assert!(store.compile_linked(&root.root_id, file).is_err());
    let mut context = base_config_context(config.clone());
    context.insert("sourceEncoding".into(), json!("convrot_int8"));
    let linked = store
        .compile_linked_with_context(&root.root_id, file, context.clone())
        .unwrap();
    assert!(
        !linked.plan.context.contains_key("sourceEncoding"),
        "precision is derived from actual tensors"
    );
    context.insert("sourceEncoding".into(), json!("unknown_future_codec"));
    assert!(store
        .compile_linked_with_context(&root.root_id, file, context)
        .is_err());
    let ingest = ManagedIngest::begin(&store, "bound_h3", provenance()).unwrap();
    ingest
        .stage_copy_file(&library.path().join(file), file)
        .unwrap();
    let managed = ingest
        .finalize_with_context(file, None, base_config_context(config.clone()))
        .unwrap();
    assert_eq!(
        linked.plan.semantic_digest().unwrap(),
        managed.compiled.plan.semantic_digest().unwrap()
    );
    drop(store);
    let store = CheckpointPlanStore::open(data.path());
    for id in [&linked.checkpoint_id, &managed.checkpoint_id] {
        let resolved = store.resolve(id).unwrap();
        let bound =
            sceneworks_core::minimax_h3_import::bound_config(&resolved.plan.context).unwrap();
        let inventory = H3SourceInventory::from_resolved_with_config(&resolved, bound).unwrap();
        assert_eq!(inventory.config.0, config);
        let derivatives = CheckpointDerivativeStore::open(data.path()).unwrap();
        let request = inventory.derivative_request();
        derivatives
            .ensure(id, &request, |out| inventory.produce(out))
            .unwrap();
        let key = derivatives.cache_key(&resolved, &request).unwrap();
        let output = derivatives.cache().bundle_path(&key).unwrap();
        let normalized: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(normalized, config);
    }
    // A rescan reuses the persisted binding without access to a foreign base file.
    let rescanned = store.compile_linked(&root.root_id, file).unwrap();
    assert_eq!(
        rescanned.plan.semantic_digest().unwrap(),
        linked.plan.semantic_digest().unwrap()
    );
    let mut changed = config.clone();
    changed["future_provider_field"]["schedule"] = json!([0.25]);
    let changed_plan = store
        .compile_linked_with_context(&root.root_id, file, base_config_context(changed))
        .unwrap();
    assert_ne!(
        linked.plan.semantic_digest().unwrap(),
        changed_plan.plan.semantic_digest().unwrap()
    );
    let mut wrong = config.clone();
    wrong["hidden_size"] = json!(16);
    assert!(store
        .compile_linked_with_context(&root.root_id, file, base_config_context(wrong))
        .is_err());
    assert_eq!(fs::read(library.path().join(file)).unwrap(), original);
}

#[test]
fn source_configuration_wins_and_conflicting_source_claims_refuse() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("source");
    write_h3_dit_dir(&source);
    let own: Value =
        serde_json::from_slice(&fs::read(source.join("config.json")).unwrap()).unwrap();
    let mut fallback = own.clone();
    fallback["rope_theta"] = json!(987654.0);
    let store = CheckpointPlanStore::open(data.path());
    let root = store.approve_root(library.path()).unwrap();
    let compiled = store
        .compile_linked_with_context(
            &root.root_id,
            "source",
            base_config_context(fallback.clone()),
        )
        .unwrap();
    assert_eq!(compiled.plan.context["transformerConfig"]["config"], own);
    assert_eq!(
        compiled.plan.context["transformerConfig"]["provenance"]["kind"],
        "source"
    );
    let file = source.join("diffusion_pytorch_model.safetensors");
    let bytes = fs::read(&file).unwrap();
    let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    header["__metadata__"] = json!({"h3_config":fallback.to_string()});
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut conflict = (encoded.len() as u64).to_le_bytes().to_vec();
    conflict.extend(encoded);
    conflict.extend_from_slice(&bytes[8 + length..]);
    fs::write(&file, conflict).unwrap();
    assert!(store.compile_linked(&root.root_id, "source").is_err());
}

#[test]
fn bound_architecture_rejects_actual_resident_interface_mismatches_not_trained_geometry() {
    use sceneworks_core::minimax_h3_import::import_component_configuration_compatible;
    let snapshot = tempfile::tempdir().unwrap();
    let config = json!({"text_dim":8,"in_channels":2,"audio_in_channels":4,"hidden_size":128,"rope_theta":123456.0,"trained_schedule":[0.125]});
    for (component, directory, document, field) in [
        (
            "text_encoder",
            "q4/text_encoder",
            json!({"text_config":{"hidden_size":8}}),
            "text_dim",
        ),
        (
            "video_vae",
            "vae",
            json!({"latent_channels":2}),
            "in_channels",
        ),
        (
            "audio_vae",
            "audio_vae",
            json!({"latent_channels":4}),
            "audio_in_channels",
        ),
    ] {
        fs::create_dir_all(snapshot.path().join(directory)).unwrap();
        fs::write(
            snapshot.path().join(directory).join("config.json"),
            document.to_string(),
        )
        .unwrap();
        let row =
            json!({"componentId":component,"subdir":directory,"files":[format!("{directory}/*")]});
        import_component_configuration_compatible(snapshot.path(), &row, &config).unwrap();
        let mut incompatible = config.clone();
        incompatible[field] = json!(999);
        assert!(
            import_component_configuration_compatible(snapshot.path(), &row, &incompatible)
                .unwrap_err()
                .contains("incompatible")
        );
        fs::write(snapshot.path().join(directory).join("config.json"), "{}").unwrap();
        assert!(import_component_configuration_compatible(snapshot.path(), &row, &config).is_err());
    }
}
