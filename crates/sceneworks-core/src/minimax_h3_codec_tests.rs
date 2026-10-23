use super::*;
use crate::checkpoint_derivative::{CheckpointDerivativeOutcomeV1, CheckpointDerivativeStore};
use crate::checkpoint_plan_store::CheckpointPlanStore;

fn config() -> H3Config {
    let mut config = H3Config::published();
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
        config.0[key] = json!(value);
    }
    config
}

fn native_key(name: &str) -> String {
    let mut name = name
        .replace("transformer_blocks.", "blocks.")
        .replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
        .replace(".attn.norm_q.", ".attn.q_norm.")
        .replace(".attn.norm_k.", ".attn.k_norm.")
        .replace(".attn.to_out.0.", ".attn.out_proj.")
        .replace(".ff.net.0.proj.", ".mlp.fc1.")
        .replace(".ff.net.2.", ".mlp.fc2.");
    for (canonical, raw) in [
        ("proj_in.", "video_patch_proj."),
        ("audio_proj_in.", "audio_patch_proj."),
        ("context_embedder.", "condition_proj."),
        ("time_embedder.linear_1.", "time_embedder.proj_in."),
        ("time_embedder.linear_2.", "time_embedder.proj_out."),
        ("norm_out.norm.", "final_layer.norm."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("proj_out.", "final_layer.video_out."),
        ("audio_proj_out.", "final_layer.audio_out."),
    ] {
        if let Some(rest) = name.strip_prefix(canonical) {
            name = format!("{raw}{rest}");
            break;
        }
    }
    name
}

type Entries = BTreeMap<String, (Vec<usize>, String, Vec<u8>)>;

fn canonical(dtype: &str) -> Entries {
    canonical_for(&config(), dtype)
}

fn canonical_for(config: &H3Config, dtype: &str) -> Entries {
    config
        .shapes()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let mut payload = Vec::new();
            for element in 0..shape.iter().product::<usize>() {
                // Distinct actual tensor values make a wrong permutation discriminating.
                let v = (i * 1000 + element) as f32 / 32.0;
                match dtype {
                    "F32" => payload.extend(v.to_le_bytes()),
                    "BF16" => payload.extend(((v.to_bits() >> 16) as u16).to_le_bytes()),
                    "F16" => payload.extend(((element % 100) as u16 + 0x3c00).to_le_bytes()),
                    _ => panic!(),
                }
            }
            (name, (shape, dtype.to_owned(), payload))
        })
        .collect()
}

fn native(canonical: &Entries, layout: H3QkvLayout) -> Entries {
    let mut entries = BTreeMap::new();
    for (key, (shape, dtype, bytes)) in canonical {
        if key.ends_with(".attn.to_k.weight") || key.ends_with(".attn.to_v.weight") {
            continue;
        }
        let mut key = native_key(key);
        let mut shape = shape.clone();
        let payload = if let Some(prefix) = key.strip_suffix(".attn.to_q.weight") {
            let canonical_prefix = if prefix.starts_with("blocks.") {
                prefix.replacen("blocks.", "transformer_blocks.", 1)
            } else {
                prefix.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
            };
            let q = &canonical[&format!("{canonical_prefix}.attn.to_q.weight")].2;
            let k = &canonical[&format!("{canonical_prefix}.attn.to_k.weight")].2;
            let v = &canonical[&format!("{canonical_prefix}.attn.to_v.weight")].2;
            let mut raw = Vec::new();
            if layout == H3QkvLayout::Interleaved {
                let head_bytes = q.len() / 2;
                for head in 0..2 {
                    for projection in [q, k, v] {
                        raw.extend(&projection[head * head_bytes..(head + 1) * head_bytes]);
                    }
                }
            } else {
                for projection in [q, k, v] {
                    raw.extend(projection);
                }
            }
            shape[0] *= 3;
            key = format!("{prefix}.attn.qkv_proj.weight");
            raw
        } else if key.ends_with(".mlp.fc1.weight") {
            [&bytes[bytes.len() / 2..], &bytes[..bytes.len() / 2]].concat()
        } else {
            bytes.clone()
        };
        entries.insert(
            format!("model.diffusion_model.{key}"),
            (shape, dtype.clone(), payload),
        );
    }
    let rope = config().0["rope_freq_dim"].as_u64().unwrap() as usize;
    let theta = config().0["rope_theta"].as_f64().unwrap() as f32;
    let bytes: Vec<u8> = (0..rope)
        .flat_map(|i| theta.powf(-(i as f32 / rope as f32)).to_le_bytes())
        .collect();
    entries.insert(
        "model.diffusion_model.rope.inv_freq".into(),
        (vec![rope], "F32".into(), bytes),
    );
    entries
}

fn write_file(path: &Path, entries: &Entries, meta: Option<Value>) {
    let mut header = Map::new();
    if let Some(meta) = meta {
        header.insert("__metadata__".into(), meta);
    }
    let mut payload = Vec::new();
    for (key, (shape, dtype, bytes)) in entries {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            key.clone(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend(header);
    output.extend(payload);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, output).unwrap();
}

fn read_file(path: &Path) -> Entries {
    let bytes = std::fs::read(path).unwrap();
    let body = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = read_safetensors_header(path).unwrap();
    header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| key.as_str() != "__metadata__")
        .map(|(key, v)| {
            let shape = serde_json::from_value(v["shape"].clone()).unwrap();
            (
                key.clone(),
                (
                    shape,
                    v["dtype"].as_str().unwrap().to_owned(),
                    bytes[body + v["data_offsets"][0].as_u64().unwrap() as usize
                        ..body + v["data_offsets"][1].as_u64().unwrap() as usize]
                        .to_vec(),
                ),
            )
        })
        .collect()
}

fn native_source(dir: &Path, dtype: &str, layout: H3QkvLayout) -> (PathBuf, Entries) {
    let canonical = canonical(dtype);
    let file = dir.join("completely-arbitrary-user-merge.safetensors");
    write_file(
        &file,
        &native(&canonical, layout),
        Some(
            json!({"h3_config":config().0.to_string(),"h3_qkv_layout":if layout == H3QkvLayout::Interleaved {"interleaved"} else {"contiguous"}}),
        ),
    );
    (file, canonical)
}

#[test]
fn pinned_provider_inventory_has_638_tensors_and_all_shapes() {
    let shapes = H3Config::published().shapes().unwrap();
    assert_eq!(shapes.len(), 638);
    assert_eq!(
        shapes["transformer_blocks.0.adaln_proj.linear.weight"],
        [96_768, 2_688]
    );
    assert_eq!(shapes["norm_out.linear.weight"], [10_752, 2_688]);
    assert_eq!(
        shapes["transformer_blocks.49.attn.to_q.weight"],
        [7_168, 5_376]
    );
}

#[test]
fn actual_native_bytes_normalize_through_persisted_plan_and_shared_cache_in_all_precisions() {
    for dtype in ["F16", "BF16", "F32"] {
        for layout in [H3QkvLayout::Interleaved, H3QkvLayout::Contiguous] {
            let data = tempfile::tempdir().unwrap();
            let library = tempfile::tempdir().unwrap();
            let (file, expected) = native_source(library.path(), dtype, layout);
            let plans = CheckpointPlanStore::open(data.path());
            let root = plans.approve_root(library.path()).unwrap();
            let compiled = plans
                .compile_linked(&root.root_id, file.file_name().unwrap().to_str().unwrap())
                .unwrap();
            let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
            let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
            assert!(inventory.native);
            let store = CheckpointDerivativeStore::open(data.path()).unwrap();
            let request = inventory.derivative_request();
            assert!(matches!(
                store
                    .ensure(&compiled.checkpoint_id, &request, |out| inventory
                        .produce(out))
                    .unwrap(),
                CheckpointDerivativeOutcomeV1::Produced(_)
            ));
            let key = store.cache_key(&resolved, &request).unwrap();
            let output = store.cache().bundle_path(&key).unwrap();
            let actual = read_file(&output.join(inventory.shard_name(0)));
            // Complete tensor-valued equality against independently constructed canonical arrays:
            // QKV interleaved/contiguous are different in source bytes; FFN gate swap is not identity.
            assert_eq!(actual, expected, "{dtype} {layout:?}");
            assert_eq!(
                serde_json::from_reader::<_, Value>(
                    File::open(output.join(H3_CONFIG_FILE)).unwrap()
                )
                .unwrap(),
                config().0
            );
            assert!(store.acquire(&resolved, &request).unwrap().is_some());
            assert!(matches!(
                store
                    .ensure(&compiled.checkpoint_id, &request, |_| panic!(
                        "must reuse cache"
                    ))
                    .unwrap(),
                CheckpointDerivativeOutcomeV1::AlreadyPresent(_)
            ));
            assert_eq!(
                read_file(&file),
                native(&expected, layout),
                "normalization never mutates source"
            );
        }
    }
}

fn sharded(dir: &Path) -> Vec<(String, PathBuf)> {
    let entries = canonical("F32");
    let mut a = Entries::new();
    let mut b = Entries::new();
    let mut index = Map::new();
    for (i, (name, tensor)) in entries.into_iter().enumerate() {
        let shard = if i % 2 == 0 {
            "renamed-a.safetensors"
        } else {
            "renamed-b.safetensors"
        };
        index.insert(name.clone(), json!(shard));
        if i % 2 == 0 {
            a.insert(name, tensor);
        } else {
            b.insert(name, tensor);
        }
    }
    write_file(&dir.join("renamed-a.safetensors"), &a, None);
    write_file(&dir.join("renamed-b.safetensors"), &b, None);
    std::fs::write(dir.join(H3_CONFIG_FILE), config().0.to_string()).unwrap();
    std::fs::write(
        dir.join(H3_INDEX_FILE),
        json!({"weight_map":index}).to_string(),
    )
    .unwrap();
    [
        "renamed-a.safetensors",
        "renamed-b.safetensors",
        H3_CONFIG_FILE,
        H3_INDEX_FILE,
    ]
    .into_iter()
    .map(|n| (n.into(), dir.join(n)))
    .collect()
}

#[test]
fn renamed_shards_assemble_exact_inventory_through_plan_store() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("arbitrary-compatible-checkpoint");
    std::fs::create_dir_all(&source).unwrap();
    let files = sharded(&source);
    let inventory = H3SourceInventory::from_files(&files).unwrap();
    assert!(!inventory.native);
    let store = CheckpointPlanStore::open(data.path());
    let root = store.approve_root(library.path()).unwrap();
    let compiled = store
        .compile_linked(&root.root_id, "arbitrary-compatible-checkpoint")
        .unwrap();
    assert_eq!(compiled.plan.family, "minimax-h3");
    assert_eq!(
        compiled
            .plan
            .layers
            .iter()
            .filter(|l| l.role == "transformer")
            .count(),
        2
    );
    let resolved = store.resolve(&compiled.checkpoint_id).unwrap();
    assert_eq!(
        H3SourceInventory::from_resolved(&resolved)
            .unwrap()
            .tensors
            .len(),
        config().shapes().unwrap().len()
    );
}

#[test]
fn incomplete_duplicate_conflicting_and_unconsumed_tensors_name_exact_faults() {
    let dir = tempfile::tempdir().unwrap();
    let (path, expected) = native_source(dir.path(), "F32", H3QkvLayout::Interleaved);
    let files = vec![(
        path.file_name().unwrap().to_string_lossy().into_owned(),
        path.clone(),
    )];
    let mut entries = native(&expected, H3QkvLayout::Interleaved);
    let missing = "model.diffusion_model.blocks.0.norm1.weight";
    let restore = entries.remove(missing).unwrap();
    let meta = Some(json!({"h3_config":config().0.to_string()}));
    write_file(&path, &entries, meta.clone());
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("missing 1 tensors: transformer_blocks.0.norm1.weight"));
    entries.insert(missing.into(), restore);
    entries.insert(
        "model.diffusion_model.blocks.0.unknown_future_quant.weight_scale".into(),
        (vec![1], "F32".into(), vec![0; 4]),
    );
    write_file(&path, &entries, meta.clone());
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("unknown_future_quant"));
    entries.remove("model.diffusion_model.blocks.0.unknown_future_quant.weight_scale");
    entries.insert(
        "model.diffusion_model.blocks.0.attn.to_q.weight".into(),
        expected["transformer_blocks.0.attn.to_q.weight"].clone(),
    );
    write_file(&path, &entries, meta.clone());
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("conflicting H3 canonical tensor"));
    entries.remove("model.diffusion_model.blocks.0.attn.to_q.weight");
    let qkv = entries
        .get_mut("model.diffusion_model.blocks.0.attn.qkv_proj.weight")
        .unwrap();
    qkv.0[0] -= 1;
    qkv.2.truncate(qkv.0.iter().product::<usize>() * 4);
    write_file(&path, &entries, meta);
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("fused QKV"));
}

#[test]
fn shard_index_mismatch_duplicate_and_missing_are_precise() {
    let dir = tempfile::tempdir().unwrap();
    let files = sharded(dir.path());
    let mut index: Value =
        serde_json::from_reader(File::open(dir.path().join(H3_INDEX_FILE)).unwrap()).unwrap();
    let key = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    index["weight_map"][&key] = json!("absent.safetensors");
    std::fs::write(dir.path().join(H3_INDEX_FILE), index.to_string()).unwrap();
    let err = H3SourceInventory::from_files(&files).unwrap_err();
    assert!(err.contains(&key) && err.contains("absent.safetensors"));
    let files = sharded(dir.path());
    let mut b = read_file(&dir.path().join("renamed-b.safetensors"));
    let a = read_file(&dir.path().join("renamed-a.safetensors"));
    let (key, value) = a.first_key_value().unwrap();
    b.insert(key.clone(), value.clone());
    write_file(&dir.path().join("renamed-b.safetensors"), &b, None);
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("duplicate H3 tensor"));
    let files = sharded(dir.path());
    std::fs::remove_file(dir.path().join("renamed-b.safetensors")).unwrap();
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("invalid MiniMax-H3 safetensors"));
}

#[test]
fn config_conflicts_future_formats_and_wrong_shapes_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let (file, _) = native_source(dir.path(), "BF16", H3QkvLayout::Interleaved);
    let mut alternate = config().0;
    alternate["rope_theta"] = json!(20000.0);
    std::fs::write(dir.path().join(H3_CONFIG_FILE), alternate.to_string()).unwrap();
    let files = vec![
        (
            file.file_name().unwrap().to_string_lossy().into_owned(),
            file,
        ),
        (H3_CONFIG_FILE.into(), dir.path().join(H3_CONFIG_FILE)),
    ];
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("conflicting"));
    let mut future = config();
    future.0["quantization"] = json!({"format":"w4a8"});
    assert!(future
        .shapes()
        .unwrap_err()
        .contains("stock packed-tier quantization policy"));
    let mut missing = config();
    missing.0.as_object_mut().unwrap().remove("rope_theta");
    assert!(missing.shapes().unwrap_err().contains("rope_theta"));
}

#[test]
fn native_rope_buffer_must_match_bound_config() {
    let dir = tempfile::tempdir().unwrap();
    let (file, expected) = native_source(dir.path(), "F32", H3QkvLayout::Interleaved);
    let mut entries = native(&expected, H3QkvLayout::Interleaved);
    entries
        .get_mut("model.diffusion_model.rope.inv_freq")
        .unwrap()
        .2 = 2f32.to_le_bytes().to_vec();
    write_file(
        &file,
        &entries,
        Some(json!({"h3_config":config().0.to_string()})),
    );
    assert!(
        H3SourceInventory::from_files(&[("arbitrary.safetensors".into(), file)])
            .unwrap_err()
            .contains("rope.inv_freq[0] conflicts")
    );
}

#[test]
fn arbitrary_sidecar_and_incomplete_shards_keep_admission_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    let files = sharded(dir.path());
    std::fs::remove_file(dir.path().join("renamed-b.safetensors")).unwrap();
    let primary = crate::base_weights::minimax_h3_import_primary(dir.path()).unwrap();
    let detection =
        crate::base_weights::detect_import_source_weights(dir.path(), &primary).unwrap();
    assert!(crate::base_weights::import_detection_supported(&detection)
        .unwrap_err()
        .contains("missing from shard"));
    for (_, path) in files {
        let _ = std::fs::remove_file(path);
    }
    let (file, _) = native_source(dir.path(), "F32", H3QkvLayout::Interleaved);
    std::fs::write(dir.path().join(H3_CONFIG_FILE), config().0.to_string()).unwrap();
    assert_eq!(
        crate::base_weights::minimax_h3_import_primary(dir.path()),
        Some(file.clone())
    );
    crate::base_weights::import_detection_supported(
        &crate::base_weights::detect_import_source_weights(dir.path(), &file).unwrap(),
    )
    .unwrap();
}

#[test]
fn canonical_single_file_metadata_and_namespace_are_normalized() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("whatever.safetensors");
    let expected = canonical("F32");
    let namespaced = expected
        .iter()
        .map(|(name, tensor)| (format!("model.diffusion_model.{name}"), tensor.clone()))
        .collect();
    write_file(
        &file,
        &namespaced,
        Some(json!({"h3_config":config().0.to_string()})),
    );
    let inventory =
        H3SourceInventory::from_files(&[("whatever.safetensors".into(), file)]).unwrap();
    assert!(!inventory.native);
    assert!(inventory.requires_normalization);
    assert_eq!(
        inventory.tensors.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
}

#[test]
fn managed_h3_config_is_bound_and_mutation_refuses() {
    let data = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    let files = sharded(source.path());
    let plans = CheckpointPlanStore::open(data.path());
    let ingest = crate::checkpoint_ingest::ManagedIngest::begin(
        &plans,
        "bound_h3",
        crate::checkpoint_import::ManagedProvenanceV1 {
            source: "local".into(),
            ..Default::default()
        },
    )
    .unwrap();
    for (name, path) in &files {
        ingest.stage_copy_file(path, name).unwrap();
    }
    let install = ingest.finalize("renamed-a.safetensors", None).unwrap();
    assert_eq!(install.compiled.plan.layers.len(), 4);
    let mut changed = config().0;
    changed["rope_theta"] = json!(20000.0);
    std::fs::write(
        install.install_path.join(H3_CONFIG_FILE),
        changed.to_string(),
    )
    .unwrap();
    assert!(plans.resolve(&install.checkpoint_id).is_err());
}

#[test]
fn oversized_descriptors_refuse_before_json_allocation_in_admission_and_codec() {
    for descriptor in [H3_CONFIG_FILE, H3_INDEX_FILE] {
        let dir = tempfile::tempdir().unwrap();
        let files = sharded(dir.path());
        // Sparse 1 GiB descriptor: metadata alone must reject it, without allocating/reading 1 GiB.
        let file = File::create(dir.path().join(descriptor)).unwrap();
        file.set_len(1 << 30).unwrap();
        let error = H3SourceInventory::from_files(&files).unwrap_err();
        assert!(
            error.contains(descriptor) && error.contains("16777216 byte descriptor limit"),
            "{error}"
        );
        let primary = crate::base_weights::minimax_h3_import_primary(dir.path()).unwrap();
        let detection =
            crate::base_weights::detect_import_source_weights(dir.path(), &primary).unwrap();
        let error = crate::base_weights::import_detection_supported(&detection).unwrap_err();
        assert!(
            error.contains(descriptor) && error.contains("descriptor limit"),
            "{error}"
        );
    }
}

#[test]
fn uppercase_shards_normalize_to_provider_enumerable_names_with_exact_values() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("mixed-case");
    std::fs::create_dir_all(&source).unwrap();
    sharded(&source);
    let mut index: Value =
        serde_json::from_reader(File::open(source.join(H3_INDEX_FILE)).unwrap()).unwrap();
    for (old, new) in [
        ("renamed-a.safetensors", "part-A.SAFETENSORS"),
        ("renamed-b.safetensors", "part-B.SafeTensors"),
    ] {
        std::fs::rename(source.join(old), source.join(new)).unwrap();
        for shard in index["weight_map"].as_object_mut().unwrap().values_mut() {
            if shard.as_str() == Some(old) {
                *shard = json!(new);
            }
        }
    }
    std::fs::write(source.join(H3_INDEX_FILE), index.to_string()).unwrap();
    let plans = CheckpointPlanStore::open(data.path());
    let root = plans.approve_root(library.path()).unwrap();
    let compiled = plans.compile_linked(&root.root_id, "mixed-case").unwrap();
    let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    assert!(!inventory.native);
    assert!(inventory.requires_normalization);
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&compiled.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let output = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let mut actual = Entries::new();
    for i in 0..2 {
        let name = inventory.shard_name(i);
        assert!(name.ends_with(".safetensors"));
        actual.extend(read_file(&output.join(name)));
    }
    assert_eq!(actual, canonical("F32"));
}

fn convrot_fixture(layout: H3QkvLayout) -> (H3Config, Entries, Entries) {
    let mut cfg = config();
    cfg.0["hidden_size"] = json!(256);
    cfg.0["attention_head_dim"] = json!(128);
    cfg.0["ffn_dim"] = json!(256);
    cfg.0["time_embed_dim"] = json!(64);
    let mut expected = canonical_for(&cfg, "F16");
    let mut scales = Entries::new();
    for (index, (name, (shape, dtype, bytes))) in expected.iter_mut().enumerate() {
        if name.starts_with("transformer_blocks.")
            && name.ends_with(".weight")
            && (name.contains(".attn.to_")
                || name.contains(".ff.")
                || name.contains(".adaln_proj.linear."))
        {
            *dtype = "I8".into();
            // Break byte-period/input-width alignment so a row or half swap changes the codes.
            *bytes = (0..shape.iter().product())
                .map(|i| {
                    let row = i / shape[1];
                    ((i * 17 + row * 29 + row / 251 * 37 + index) % 256) as u8
                })
                .collect();
            scales.insert(
                name.clone(),
                (
                    vec![shape[0], 1],
                    "F32".into(),
                    (0..shape[0])
                        .flat_map(|row| ((index * 100 + row + 1) as f32 / 4096.0).to_le_bytes())
                        .collect(),
                ),
            );
        }
    }
    let mut raw = native(&expected, layout);
    // Independently build raw scales through the same source exporter QKV/gate layout as the
    // original float checkpoint, then retain canonical byte arrays as the expected consumer data.
    let mut all_scales = canonical_for(&cfg, "F32");
    for (key, value) in &scales {
        all_scales.insert(key.clone(), value.clone());
    }
    let raw_scales = native(&all_scales, layout);
    let quantized: Vec<_> = raw
        .iter()
        .filter(|(_, (_, dtype, _))| dtype == "I8")
        .map(|(name, (shape, _, _))| (name.clone(), shape.clone()))
        .collect();
    for (name, shape) in quantized {
        let base = name.strip_suffix(".weight").unwrap();
        raw.insert(format!("{base}.weight_scale"), raw_scales[&name].clone());
        let group = if shape[1] == 64 { 64 } else { 256 };
        let payload = json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":group})
            .to_string()
            .into_bytes();
        raw.insert(
            format!("{base}.comfy_quant"),
            (vec![payload.len()], "U8".into(), payload),
        );
    }
    for (name, value) in scales {
        let base = name.strip_suffix(".weight").unwrap();
        let input = expected[&name].0[1];
        let group = if input == 64 { 64 } else { 256 };
        let payload = json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":group})
            .to_string()
            .into_bytes();
        expected.insert(format!("{base}.weight_scale"), value);
        expected.insert(
            format!("{base}.comfy_quant"),
            (vec![payload.len()], "U8".into(), payload),
        );
    }
    (cfg, raw, expected)
}

#[test]
fn convrot_groups64_and256_preserve_codes_scales_and_descriptors_through_real_cache() {
    for layout in [H3QkvLayout::Interleaved, H3QkvLayout::Contiguous] {
        let (cfg, raw, expected) = convrot_fixture(layout);
        let data = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();
        let file = library.path().join("user-checkpoint.safetensors");
        write_file(
            &file,
            &raw,
            Some(
                json!({"h3_config":cfg.0.to_string(),"h3_qkv_layout":if layout == H3QkvLayout::Interleaved { "interleaved" } else { "contiguous" }}),
            ),
        );
        let plans = CheckpointPlanStore::open(data.path());
        let root = plans.approve_root(library.path()).unwrap();
        let compiled = plans
            .compile_linked(&root.root_id, "user-checkpoint.safetensors")
            .unwrap();
        let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
        let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
        assert!(inventory.convrot_int8);
        assert_eq!(
            inventory.derivative_request().codec_id,
            "convrot-int8-native"
        );
        let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
        let request = inventory.derivative_request();
        cache
            .ensure(&compiled.checkpoint_id, &request, |out| {
                inventory.produce(out)
            })
            .unwrap();
        let output = cache
            .cache()
            .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
            .unwrap();
        assert_eq!(read_file(&output.join(inventory.shard_name(0))), expected);
        // Canonical output is itself accepted with the exact descriptor/scale geometry.
        H3SourceInventory::from_files(&[
            (H3_CONFIG_FILE.into(), output.join(H3_CONFIG_FILE)),
            (
                inventory.shard_name(0),
                output.join(inventory.shard_name(0)),
            ),
        ])
        .unwrap();
    }
}

#[test]
fn convrot_malformed_descriptors_scales_and_signed_storage_refuse_before_copy() {
    let (cfg, original, _) = convrot_fixture(H3QkvLayout::Interleaved);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("arbitrary.safetensors");
    let descriptor = "model.diffusion_model.blocks.0.attn.qkv_proj.comfy_quant";
    let scale = "model.diffusion_model.blocks.0.attn.qkv_proj.weight_scale";
    for fault in [
        "unknown-field",
        "group",
        "zero-scale",
        "nan-scale",
        "missing-scale",
        "unsigned",
        "orphan",
    ] {
        let mut raw = original.clone();
        match fault {
            "unknown-field" | "group" => {
                let mut value: Value = serde_json::from_slice(&raw[descriptor].2).unwrap();
                if fault == "group" {
                    value["convrot_groupsize"] = json!(32);
                } else {
                    value["per_row"] = json!(true);
                }
                let bytes = value.to_string().into_bytes();
                raw.insert(descriptor.into(), (vec![bytes.len()], "U8".into(), bytes));
            }
            "zero-scale" | "nan-scale" => raw.get_mut(scale).unwrap().2[..4].copy_from_slice(
                &if fault == "zero-scale" {
                    0f32
                } else {
                    f32::NAN
                }
                .to_le_bytes(),
            ),
            "missing-scale" => {
                raw.remove(scale);
            }
            "unsigned" => {
                raw.get_mut("model.diffusion_model.blocks.0.attn.qkv_proj.weight")
                    .unwrap()
                    .1 = "U8".into()
            }
            "orphan" => {
                raw.remove(descriptor);
            }
            _ => unreachable!(),
        }
        write_file(&file, &raw, Some(json!({"h3_config":cfg.0.to_string()})));
        assert!(
            H3SourceInventory::from_files(&[("arbitrary.safetensors".into(), file.clone())])
                .is_err(),
            "{fault}"
        );
    }
}

#[test]
fn weight_only_requires_explicit_config_and_owned_source_configuration_wins() {
    let dir = tempfile::tempdir().unwrap();
    let (file, expected) = native_source(dir.path(), "BF16", H3QkvLayout::Interleaved);
    let files = vec![("user.safetensors".into(), file.clone())];
    let mut fallback = config();
    fallback.0["rope_theta"] = json!(20000.0);
    let owned = H3SourceInventory::from_files_with_config(&files, Some(fallback)).unwrap();
    assert_eq!(owned.config.0, config().0);
    write_file(&file, &native(&expected, H3QkvLayout::Interleaved), None);
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("no config.json or embedded h3_config"));
    let mut explicit = config();
    explicit.0["provider_specific_schedule"] =
        json!({"knots":[0.0,0.125,1.0],"precision":"preserve-me"});
    let inventory =
        H3SourceInventory::from_files_with_config(&files, Some(explicit.clone())).unwrap();
    assert_eq!(inventory.config.0, explicit.0);
}

#[test]
fn convrot_codes_scales_and_descriptors_can_live_in_separate_sealed_shards() {
    let (cfg, _, expected) = convrot_fixture(H3QkvLayout::Interleaved);
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("user-shards");
    std::fs::create_dir_all(&source).unwrap();
    let names = [
        "codes.safetensors",
        "scales.safetensors",
        "descriptors.safetensors",
    ];
    let mut parts: [Entries; 3] = std::array::from_fn(|_| Entries::new());
    let mut index = Map::new();
    for (key, tensor) in &expected {
        let part = if key.ends_with(".comfy_quant") {
            2
        } else if key.ends_with(".weight_scale") {
            1
        } else {
            0
        };
        parts[part].insert(key.clone(), tensor.clone());
        index.insert(key.clone(), json!(names[part]));
    }
    for (name, part) in names.iter().zip(&parts) {
        write_file(&source.join(name), part, None);
    }
    std::fs::write(source.join(H3_CONFIG_FILE), cfg.0.to_string()).unwrap();
    std::fs::write(
        source.join(H3_INDEX_FILE),
        json!({"weight_map":index}).to_string(),
    )
    .unwrap();
    let plans = CheckpointPlanStore::open(data.path());
    let root = plans.approve_root(library.path()).unwrap();
    let compiled = plans.compile_linked(&root.root_id, "user-shards").unwrap();
    let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
    assert_eq!(resolved.plan.context["sourceEncoding"], "convrot_int8");
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    assert!(!inventory.native);
    assert!(inventory.convrot_int8);
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&compiled.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let _lease = cache.acquire(&resolved, &request).unwrap().unwrap();
    let output = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let mut actual = Entries::new();
    for i in 0..3 {
        actual.extend(read_file(&output.join(inventory.shard_name(i))));
    }
    assert_eq!(actual, expected);
    parts[1].values_mut().next().unwrap().2[..4].copy_from_slice(&0f32.to_le_bytes());
    write_file(&source.join(names[1]), &parts[1], None);
    assert!(
        plans.resolve(&compiled.checkpoint_id).is_err(),
        "cross-shard scale drift invalidates the source identity"
    );
    let files: Vec<_> = names
        .into_iter()
        .chain([H3_CONFIG_FILE, H3_INDEX_FILE])
        .map(|name| (name.to_owned(), source.join(name)))
        .collect();
    assert!(H3SourceInventory::from_files(&files)
        .unwrap_err()
        .contains("finite positive F32 scales"));
}
