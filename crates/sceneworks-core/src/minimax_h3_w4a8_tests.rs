//! Synthetic actual-format S6 fixtures; no model downloads or device execution.
use super::*;

fn w4a8_fixture(layout: H3QkvLayout, pruned: bool) -> (H3Config, Entries) {
    w4a8_fixture_depth(layout, pruned, 1, 1)
}

fn w4a8_fixture_depth(
    layout: H3QkvLayout,
    pruned: bool,
    layers: usize,
    refiners: usize,
) -> (H3Config, Entries) {
    let mut cfg = config();
    cfg.0["num_layers"] = json!(layers);
    cfg.0["num_refiner_layers"] = json!(refiners);
    for (key, value) in [
        ("hidden_size", 256),
        ("attention_head_dim", 128),
        ("ffn_dim", 256),
    ] {
        cfg.0[key] = json!(value);
    }
    let mut canonical = canonical_for(&cfg, "BF16");
    if pruned {
        let shapes = cfg.pruned_shapes(5, 8).unwrap();
        canonical.retain(|key, _| shapes.contains_key(key));
        for (seed, (key, shape)) in shapes.into_iter().enumerate() {
            if canonical.get(&key).is_none_or(|v| v.0 != shape) {
                let dtype = if key == TABLE_KEY { "F32" } else { "F16" };
                canonical.insert(
                    key,
                    (
                        shape.clone(),
                        dtype.into(),
                        stored_bytes(dtype, seed, shape.iter().product()),
                    ),
                );
            }
        }
    }
    let mut raw = native(&canonical, layout);
    let projections: Vec<_> = raw
        .keys()
        .filter(|key| {
            key.ends_with(".weight")
                && (key.contains(".attn.qkv_proj.")
                    || key.contains(".attn.out_proj.")
                    || key.contains(".mlp."))
        })
        .cloned()
        .collect();
    for (seed, key) in projections.into_iter().enumerate() {
        let base = key.strip_suffix(".weight").unwrap();
        let shape = raw[&key].0.clone();
        let (n, k) = (shape[0], shape[1]);
        let descriptor = if key.contains("token_refiner.") {
            raw.insert(
                key.clone(),
                (
                    shape,
                    "I8".into(),
                    (0..n * k)
                        .map(|i| (i * 17 + i / k * 13 + seed) as u8)
                        .collect(),
                ),
            );
            raw.insert(
                format!("{base}.weight_scale"),
                (
                    vec![n, 1],
                    "F32".into(),
                    (0..n)
                        .flat_map(|i| ((i + 1) as f32 / 1024.0).to_le_bytes())
                        .collect(),
                ),
            );
            json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256})
        } else {
            raw.insert(
                key.clone(),
                (
                    vec![n, k / 2],
                    "I8".into(),
                    (0..n * k / 2)
                        .map(|i| (i * 19 + i / (k / 2) * 23 + seed) as u8)
                        .collect(),
                ),
            );
            raw.insert(
                format!("{base}.weight_s_rel"),
                (
                    vec![n, k / 16],
                    "F8_E4M3".into(),
                    (0..n * k / 16).map(|i| (i % 0x7f) as u8).collect(),
                ),
            );
            raw.insert(
                format!("{base}.weight_s_channel"),
                (
                    vec![n],
                    "F32".into(),
                    (0..n)
                        .flat_map(|i| ((i + 1) as f32 / 4096.0).to_le_bytes())
                        .collect(),
                ),
            );
            raw.insert(
                format!("{base}.weight_codebook"),
                (
                    vec![16],
                    "F32".into(),
                    (0..16)
                        .flat_map(|i| (i as f32 - 7.5).to_le_bytes())
                        .collect(),
                ),
            );
            json!({"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256})
        };
        let bytes = descriptor.to_string().into_bytes();
        raw.insert(
            format!("{base}.comfy_quant"),
            (vec![bytes.len()], "U8".into(), bytes),
        );
    }
    (cfg, raw)
}

#[test]
fn mixed_raw_normalization_preserves_packing_dtypes_and_row_bindings() {
    for layout in [H3QkvLayout::Interleaved, H3QkvLayout::Contiguous] {
        for pruned in [false, true] {
            let (cfg, raw) = w4a8_fixture(layout, pruned);
            let data = tempfile::tempdir().unwrap();
            let library = tempfile::tempdir().unwrap();
            let source = library.path().join("renamed.safetensors");
            write_file(
                &source,
                &raw,
                Some(
                    json!({"h3_config":cfg.0.to_string(),"h3_qkv_layout":if layout==H3QkvLayout::Interleaved {"interleaved"} else {"contiguous"}}),
                ),
            );
            let plans = CheckpointPlanStore::open(data.path());
            let root = plans.approve_root(library.path()).unwrap();
            let compiled = plans
                .compile_linked(&root.root_id, "renamed.safetensors")
                .unwrap();
            let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
            assert_eq!(resolved.plan.context["sourceEncoding"], "asym_w4a8_int8");
            let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
            assert!(inventory.w4a8_int8 && inventory.convrot_int8);
            assert_eq!(inventory.pruning.is_some(), pruned);
            let request = inventory.derivative_request();
            assert_eq!(request.codec_id, "w4a8-int8-native");
            let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
            cache
                .ensure(&compiled.checkpoint_id, &request, |out| {
                    inventory.produce(out)
                })
                .unwrap();
            let output = cache
                .cache()
                .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
                .unwrap();
            let actual = read_file(&output.join(inventory.shard_name(0)));
            // Independent row oracle: all packed operands must follow exactly the same QKV move.
            for suffix in ["weight", "weight_s_rel", "weight_s_channel"] {
                let original =
                    &raw[&format!("model.diffusion_model.blocks.0.attn.qkv_proj.{suffix}")];
                let row_bytes = original.2.len() / 768;
                for (j, leaf) in ["to_q", "to_k", "to_v"].into_iter().enumerate() {
                    let mut expected = Vec::new();
                    for row in 0..256 {
                        let source_row = if layout == H3QkvLayout::Interleaved {
                            (row / 128 * 3 + j) * 128 + row % 128
                        } else {
                            j * 256 + row
                        };
                        expected.extend_from_slice(
                            &original.2[source_row * row_bytes..(source_row + 1) * row_bytes],
                        );
                    }
                    let mapped = &actual[&format!("transformer_blocks.0.attn.{leaf}.{suffix}")];
                    assert_eq!(mapped.1, original.1);
                    assert_eq!(mapped.2, expected);
                }
            }
            for suffix in ["weight", "weight_s_rel", "weight_s_channel"] {
                let original = &raw[&format!("model.diffusion_model.blocks.0.mlp.fc1.{suffix}")].2;
                assert_eq!(
                    actual[&format!("transformer_blocks.0.ff.net.0.proj.{suffix}")].2,
                    [
                        &original[original.len() / 2..],
                        &original[..original.len() / 2]
                    ]
                    .concat()
                );
            }
            assert_eq!(
                actual["transformer_blocks.0.attn.to_q.weight_codebook"],
                raw["model.diffusion_model.blocks.0.attn.qkv_proj.weight_codebook"]
            );
            let canonical = H3SourceInventory::from_files(&[
                (H3_CONFIG_FILE.into(), output.join(H3_CONFIG_FILE)),
                (
                    inventory.shard_name(0),
                    output.join(inventory.shard_name(0)),
                ),
            ])
            .unwrap();
            assert!(canonical.w4a8_int8 && canonical.convrot_int8);
            assert_eq!(canonical.pruning, inventory.pruning);
        }
    }
}

#[test]
fn cross_shard_managed_and_linked_imports_keep_native_format_and_source_identity() {
    let (cfg, raw) = w4a8_fixture(H3QkvLayout::Interleaved, true);
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();
    let raw_path = library.path().join("unrelated-name.safetensors");
    write_file(
        &raw_path,
        &raw,
        Some(json!({"h3_config":cfg.0.to_string()})),
    );
    let plans = CheckpointPlanStore::open(data.path());
    let root = plans.approve_root(library.path()).unwrap();
    let compiled = plans
        .compile_linked(&root.root_id, "unrelated-name.safetensors")
        .unwrap();
    let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&compiled.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let normalized = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let canonical = read_file(&normalized.join(inventory.shard_name(0)));
    let source = library.path().join("arbitrary-shards");
    std::fs::create_dir_all(&source).unwrap();
    let names = [
        "codes.safetensors",
        "scales.safetensors",
        "descriptors.safetensors",
    ];
    let mut parts = [Entries::new(), Entries::new(), Entries::new()];
    let mut index = Map::new();
    for (key, value) in &canonical {
        let part = if key.ends_with(".comfy_quant") {
            2
        } else if key.ends_with(".weight_s_rel")
            || key.ends_with(".weight_s_channel")
            || key.ends_with(".weight_codebook")
            || key.ends_with(".weight_scale")
        {
            1
        } else {
            0
        };
        parts[part].insert(key.clone(), value.clone());
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
    let linked = plans
        .compile_linked(&root.root_id, "arbitrary-shards")
        .unwrap();
    let linked_resolved = plans.resolve(&linked.checkpoint_id).unwrap();
    let linked_inventory = H3SourceInventory::from_resolved(&linked_resolved).unwrap();
    let ingest = crate::checkpoint_ingest::ManagedIngest::begin(
        &plans,
        "packed_user",
        crate::checkpoint_import::ManagedProvenanceV1 {
            source: "local".into(),
            ..Default::default()
        },
    )
    .unwrap();
    for name in names.into_iter().chain([H3_CONFIG_FILE, H3_INDEX_FILE]) {
        ingest.stage_copy_file(&source.join(name), name).unwrap();
    }
    let managed = ingest.finalize(names[0], None).unwrap();
    let managed_resolved = plans.resolve(&managed.compiled.checkpoint_id).unwrap();
    let managed_inventory = H3SourceInventory::from_resolved(&managed_resolved).unwrap();
    assert_eq!(linked_inventory.source_encoding(), Some("asym_w4a8_int8"));
    assert_eq!(
        managed_inventory.source_encoding(),
        linked_inventory.source_encoding()
    );
    assert_eq!(managed_inventory.pruning, inventory.pruning);
    assert_eq!(linked_inventory.pruning, inventory.pruning);
    assert_eq!(
        linked_inventory.derivative_request().codec_id,
        "w4a8-int8-native"
    );
    let entry = json!({"minimaxH3Partition":"ref2va","baseModel":"minimax_h3_ref"})
        .as_object()
        .unwrap()
        .clone();
    let mut assembly =
        crate::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(&linked.plan, &entry)
            .unwrap();
    crate::minimax_h3_import::verify_imported_pruning(&assembly, &linked_inventory).unwrap();
    assembly.source_encoding = Some("convrot_int8".into());
    assert!(
        crate::minimax_h3_import::verify_imported_pruning(&assembly, &linked_inventory)
            .unwrap_err()
            .contains("encoding")
    );
    // Same shape, different codebook/scales/packing must change plan/cache identity, not just
    // the pruning identity (the AdaLN table is deliberately unchanged).
    let first_key = cache
        .cache_key(&linked_resolved, &linked_inventory.derivative_request())
        .unwrap();
    let codebook = parts[1]
        .iter_mut()
        .find(|(k, _)| k.ends_with(".weight_codebook"))
        .unwrap()
        .1;
    codebook.2[..4].copy_from_slice(&(-6.25f32).to_le_bytes());
    write_file(&source.join(names[1]), &parts[1], None);
    assert!(plans.resolve(&linked.checkpoint_id).is_err());
    assert!(
        plans.resolve(&managed.compiled.checkpoint_id).is_ok(),
        "managed copy is independent of linked source drift"
    );
    let changed = plans
        .compile_linked(&root.root_id, "arbitrary-shards")
        .unwrap();
    let changed_resolved = plans.resolve(&changed.checkpoint_id).unwrap();
    let changed_inventory = H3SourceInventory::from_resolved(&changed_resolved).unwrap();
    assert_eq!(changed_inventory.pruning, linked_inventory.pruning);
    assert_ne!(
        cache
            .cache_key(&changed_resolved, &changed_inventory.derivative_request())
            .unwrap(),
        first_key
    );
}

#[test]
fn published_depth_mixed_inventory_uses_every_trunk_and_refiner_projection() {
    // Match the source's 50 trunk / 2 refiner mechanism at bounded channel width: 200 packed
    // raw projections and eight ConvRot projections. Canonical QKV splits add two per block.
    let (layers, refiners) = (50, 2);
    let (cfg, raw) = w4a8_fixture_depth(H3QkvLayout::Interleaved, true, layers, refiners);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not-a-model-name.safetensors");
    write_file(&file, &raw, Some(json!({"h3_config":cfg.0.to_string()})));
    let inventory =
        H3SourceInventory::from_files(&[("not-a-model-name.safetensors".into(), file)]).unwrap();
    assert_eq!(
        raw.keys().filter(|k| k.ends_with(".weight_s_rel")).count(),
        4 * layers
    );
    assert_eq!(
        raw.keys().filter(|k| k.ends_with(".weight_scale")).count(),
        4 * refiners
    );
    assert_eq!(
        inventory
            .tensors
            .keys()
            .filter(|k| k.ends_with(".weight_s_rel"))
            .count(),
        6 * layers
    );
    assert_eq!(
        inventory
            .tensors
            .keys()
            .filter(|k| k.ends_with(".weight_scale"))
            .count(),
        6 * refiners
    );
    assert!(inventory.w4a8_int8 && inventory.convrot_int8 && inventory.pruning.is_some());
}

#[test]
fn malformed_or_future_operands_refuse_before_publication() {
    let (cfg, original) = w4a8_fixture(H3QkvLayout::Interleaved, true);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("anything.safetensors");
    let base = "model.diffusion_model.blocks.0.attn.qkv_proj";
    for fault in [
        "missing-codebook",
        "missing-relative",
        "missing-channel",
        "nan-table",
        "nan-codebook",
        "negative-relative",
        "nan-relative",
        "zero-channel",
        "bad-relative-dtype",
        "bad-packing",
        "future-format",
        "future-field",
        "missing-table",
    ] {
        let mut raw = original.clone();
        match fault {
            "missing-codebook" => {
                raw.remove(&format!("{base}.weight_codebook"));
            }
            "missing-relative" => {
                raw.remove(&format!("{base}.weight_s_rel"));
            }
            "missing-channel" => {
                raw.remove(&format!("{base}.weight_s_channel"));
            }
            "nan-table" => raw
                .get_mut(&format!("model.diffusion_model.{TABLE_KEY}"))
                .unwrap()
                .2[..4]
                .copy_from_slice(&f32::NAN.to_le_bytes()),
            "nan-codebook" => raw.get_mut(&format!("{base}.weight_codebook")).unwrap().2[..4]
                .copy_from_slice(&f32::NAN.to_le_bytes()),
            "negative-relative" => {
                raw.get_mut(&format!("{base}.weight_s_rel")).unwrap().2[0] = 0xb8
            }
            "nan-relative" => raw.get_mut(&format!("{base}.weight_s_rel")).unwrap().2[0] = 0x7f,
            "zero-channel" => raw.get_mut(&format!("{base}.weight_s_channel")).unwrap().2[..4]
                .copy_from_slice(&0f32.to_le_bytes()),
            "bad-relative-dtype" => {
                raw.get_mut(&format!("{base}.weight_s_rel")).unwrap().1 = "U8".into()
            }
            "bad-packing" => raw.get_mut(&format!("{base}.weight")).unwrap().1 = "U8".into(),
            "future-format" | "future-field" => {
                let bytes = if fault == "future-format" {
                    br#"{"format":"asym_w4a8_future","group_size":16,"convrot_groupsize":256}"#
                        .to_vec()
                } else {
                    br#"{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256,"future":true}"#.to_vec()
                };
                raw.insert(
                    format!("{base}.comfy_quant"),
                    (vec![bytes.len()], "U8".into(), bytes),
                );
            }
            "missing-table" => {
                raw.remove(&format!("model.diffusion_model.{TABLE_KEY}"));
            }
            _ => unreachable!(),
        }
        write_file(&file, &raw, Some(json!({"h3_config":cfg.0.to_string()})));
        assert!(
            H3SourceInventory::from_files(&[("anything.safetensors".into(), file.clone())])
                .is_err(),
            "accepted {fault}"
        );
    }
}
