use super::*;
use crate::base_weights::QuantFormat;

const QKV: &str = "model.diffusion_model.blocks.0.attn.qkv_proj";
const FF: &str = "model.diffusion_model.blocks.0.mlp.fc1";

fn codes(row: usize, col: usize) -> u8 {
    ((row * 13 + col * 7 + 1) % 251) as u8
}
fn exponent(row: usize, block: usize) -> u8 {
    ((row * 11 + block * 19) % 250) as u8
}

fn fixture(format: QuantFormat, layout: H3QkvLayout, table: bool) -> (Entries, Value) {
    let mut cfg = config();
    cfg.0["hidden_size"] = json!(144);
    cfg.0["attention_head_dim"] = json!(25);
    cfg.0["ffn_dim"] = json!(73);
    let mut raw = native(&canonical_for(&cfg, "F16"), layout);
    let mut metadata = source_meta(&cfg, layout);
    let mut declarations = Map::new();
    for base in [QKV, FF] {
        let key = format!("{base}.weight");
        let dims = raw[&key].0.clone();
        let logical = [dims[0], dims[1]];
        let mx = format == QuantFormat::Mxfp8;
        let stored = if mx {
            gen_core::mxfp8_padded_shape(logical)
        } else {
            logical
        };
        let dtype = if format == QuantFormat::Fp8E5M2Scalar {
            "F8_E5M2"
        } else {
            "F8_E4M3"
        };
        let data = (0..stored[0])
            .flat_map(|r| (0..stored[1]).map(move |c| codes(r, c)))
            .collect();
        raw.insert(key, (stored.to_vec(), dtype.into(), data));
        let desc = match format {
            QuantFormat::Mxfp8 => {
                let scale_shape = gen_core::mxfp8_scale_shape(stored);
                let mut scale = vec![127; scale_shape.iter().product()];
                for r in 0..stored[0] {
                    for b in 0..stored[1] / 32 {
                        scale[gen_core::mxfp8_swizzled_scale_index(stored, r, b)] = exponent(r, b);
                    }
                }
                raw.insert(
                    format!("{base}.weight_scale"),
                    (scale_shape.to_vec(), "U8".into(), scale),
                );
                json!({"format":"mxfp8","orig_shape":dims,"orig_dtype":"bfloat16","group_size":32})
            }
            QuantFormat::Fp8E4M3Rowwise => {
                let scales = (0..logical[0])
                    .flat_map(|r| ((r + 1) as f32 / 100.).to_le_bytes())
                    .collect();
                raw.insert(
                    format!("{base}.weight_scale"),
                    (vec![logical[0]], "F32".into(), scales),
                );
                json!({"format":"fp8r"})
            }
            QuantFormat::Fp8E4M3Scalar | QuantFormat::Fp8E5M2Scalar => {
                raw.insert(
                    format!("{base}.weight_scale"),
                    (vec![], "F32".into(), 2.25f32.to_le_bytes().into()),
                );
                json!({"format":if dtype=="F8_E5M2" {"float8_e5m2"} else {"float8_e4m3fn"},"full_precision_matrix_mult":true})
            }
            _ => unreachable!(),
        };
        if format != QuantFormat::Fp8E4M3Rowwise {
            raw.insert(
                format!("{base}.input_scale"),
                (vec![], "F32".into(), 0.75f32.to_le_bytes().into()),
            );
        }
        if table {
            declarations.insert(base.into(), desc);
        } else {
            let bytes = serde_json::to_vec(&desc).unwrap();
            raw.insert(
                format!("{base}.comfy_quant"),
                (vec![bytes.len()], "U8".into(), bytes),
            );
        }
    }
    if table {
        metadata["_quantization_metadata"] = json!(json!({"layers":declarations}).to_string());
    }
    (raw, metadata)
}

#[test]
fn normalizes_encoded_qkv_and_swiglu_with_the_source_codec_and_row_order() {
    for format in [
        QuantFormat::Fp8E4M3Scalar,
        QuantFormat::Fp8E5M2Scalar,
        QuantFormat::Fp8E4M3Rowwise,
        QuantFormat::Mxfp8,
    ] {
        for layout in [H3QkvLayout::Contiguous, H3QkvLayout::Interleaved] {
            let (raw, meta) = fixture(format, layout, layout == H3QkvLayout::Interleaved);
            let library = tempfile::tempdir().unwrap();
            let data = tempfile::tempdir().unwrap();
            let file = library.path().join("h3.safetensors");
            write_file(&file, &raw, Some(meta));
            assert!(crate::base_weights::is_minimax_h3_single_file(&file));
            let detection =
                crate::base_weights::detect_import_source_weights(&file, &file).unwrap();
            assert!(
                matches!(&detection,crate::base_weights::BaseWeightDetection::Recognized(verdict) if verdict.quant==format)
            );
            crate::base_weights::import_detection_supported(&detection).unwrap();
            let plans = CheckpointPlanStore::open(data.path());
            let root = plans.approve_root(library.path()).unwrap();
            let compiled = plans
                .compile_linked(&root.root_id, "h3.safetensors")
                .unwrap();
            let resolved = plans.resolve(&compiled.checkpoint_id).unwrap();
            let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
            assert_eq!(resolved.plan.context["sourceEncoding"], format.as_str());
            assert_eq!(inventory.source_encoding(), Some(format.as_str()));
            assert!(!inventory.convrot_int8 && !inventory.w4a8_int8);
            let request = inventory.derivative_request();
            assert_eq!(Some(request.codec_id.as_str()), format.source_codec_id());
            let store = CheckpointDerivativeStore::open(data.path()).unwrap();
            store
                .ensure(&compiled.checkpoint_id, &request, |out| {
                    inventory.produce(out)
                })
                .unwrap();
            let dir = store
                .cache()
                .bundle_path(&store.cache_key(&resolved, &request).unwrap())
                .unwrap();
            let shard = inventory.shard_name(0);
            let output = read_file(&dir.join(&shard));
            let detected =
                crate::base_weights::detect_import_source_weights(&dir, &dir.join(&shard)).unwrap();
            assert!(
                matches!(detected,crate::base_weights::BaseWeightDetection::Recognized(verdict) if verdict.quant==format)
            );
            if format == QuantFormat::Fp8E5M2Scalar && layout == H3QkvLayout::Contiguous {
                // Keep weights and their calibrated operands on different shards. The
                // primary file alone contains no descriptor or scale authority.
                let sharded = tempfile::tempdir().unwrap();
                let root = sharded.path();
                let mut index = Map::new();
                for (take_companions, filename) in [
                    (false, "weights.safetensors"),
                    (true, "companions.safetensors"),
                ] {
                    let entries: Entries = output
                        .iter()
                        .filter(|(name, _)| {
                            let companion = name.ends_with(".weight_scale")
                                || name.ends_with(".input_scale")
                                || name.ends_with(".comfy_quant");
                            companion == take_companions
                        })
                        .map(|(name, entry)| {
                            index.insert(name.clone(), json!(filename));
                            (name.clone(), entry.clone())
                        })
                        .collect();
                    write_file(&root.join(filename), &entries, None);
                }
                std::fs::copy(dir.join(H3_CONFIG_FILE), root.join(H3_CONFIG_FILE)).unwrap();
                std::fs::write(
                    root.join(H3_INDEX_FILE),
                    serde_json::to_vec(&json!({"weight_map":index})).unwrap(),
                )
                .unwrap();
                let detection = crate::base_weights::detect_import_source_weights(
                    root,
                    &root.join("weights.safetensors"),
                )
                .unwrap();
                assert!(
                    matches!(detection,crate::base_weights::BaseWeightDetection::Recognized(v) if v.quant==format)
                );
            }
            let normalized = H3SourceInventory::from_files(&[
                (H3_CONFIG_FILE.into(), dir.join(H3_CONFIG_FILE)),
                (H3_INDEX_FILE.into(), dir.join(H3_INDEX_FILE)),
                (shard.clone(), dir.join(shard)),
            ])
            .unwrap();
            assert_eq!(normalized.source_encoding(), inventory.source_encoding());
            assert!(
                !normalized.requires_normalization,
                "canonical FP8 shards can load in place"
            );
            for (part, leaf) in ["attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj"]
                .into_iter()
                .enumerate()
            {
                let dest = format!("transformer_blocks.0.{leaf}");
                let rows = if part == 3 { 146 } else { 50 };
                let src_row = |r: usize| {
                    if part == 3 {
                        (r + 73) % 146
                    } else if layout == H3QkvLayout::Contiguous {
                        part * 50 + r
                    } else {
                        (r / 25 * 3 + part) * 25 + r % 25
                    }
                };
                let mx = format == QuantFormat::Mxfp8;
                let stored = if mx {
                    gen_core::mxfp8_padded_shape([rows, 144])
                } else {
                    [rows, 144]
                };
                let (shape, _, bytes) = &output[&format!("{dest}.weight")];
                assert_eq!(shape, &stored.to_vec());
                for r in 0..rows {
                    for c in 0..stored[1] {
                        assert_eq!(bytes[r * stored[1] + c], codes(src_row(r), c));
                    }
                }
                if mx {
                    assert!(bytes[rows * stored[1]..].iter().all(|&b| b == 0));
                }
                let (_, _, scales) = &output[&format!("{dest}.weight_scale")];
                if mx {
                    for r in 0..rows {
                        for b in 0..stored[1] / 32 {
                            assert_eq!(
                                scales[gen_core::mxfp8_swizzled_scale_index(stored, r, b)],
                                exponent(src_row(r), b)
                            );
                        }
                    }
                    let desc: Value =
                        serde_json::from_slice(&output[&format!("{dest}.comfy_quant")].2).unwrap();
                    assert_eq!(desc["orig_shape"], json!([rows, 144]));
                    assert_eq!(desc["orig_dtype"], "bfloat16");
                    assert_eq!(desc["group_size"], 32);
                } else if format == QuantFormat::Fp8E4M3Rowwise {
                    let expected: Vec<_> = (0..rows)
                        .flat_map(|r| ((src_row(r) + 1) as f32 / 100.).to_le_bytes())
                        .collect();
                    assert_eq!(scales, &expected);
                } else {
                    assert_eq!(scales, &2.25f32.to_le_bytes());
                }
                if format != QuantFormat::Fp8E4M3Rowwise {
                    assert_eq!(
                        output[&format!("{dest}.input_scale")].2,
                        0.75f32.to_le_bytes()
                    );
                }
            }
        }
    }
}

#[test]
fn rejects_mixed_fp8_conventions_and_forged_h3_source_stamps() {
    let (mut raw, meta) = fixture(QuantFormat::Fp8E4M3Scalar, H3QkvLayout::Contiguous, false);
    raw.get_mut(&format!("{FF}.weight")).unwrap().1 = "F8_E5M2".into();
    let bytes = br#"{"format":"float8_e5m2"}"#.to_vec();
    raw.insert(
        format!("{FF}.comfy_quant"),
        (vec![bytes.len()], "U8".into(), bytes),
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h3.safetensors");
    write_file(&path, &raw, Some(meta));
    let err = H3SourceInventory::from_files(&[("h3.safetensors".into(), path)]).unwrap_err();
    assert!(err.contains("mixed H3 FP8"), "{err}");
    let forged = json!({"family":"minimax-h3","importQuantFormat":"fp8_e4m3_scalar"});
    assert!(crate::jobs_store::imported_entry_source_codec(forged.as_object().unwrap()).is_none());
}

#[test]
fn descriptorless_e5_and_legacy_scale_spelling_preserve_their_physical_contract() {
    let layout = H3QkvLayout::Interleaved;
    let (mut raw, meta) = fixture(QuantFormat::Fp8E5M2Scalar, layout, false);
    for base in [QKV, FF] {
        raw.remove(&format!("{base}.comfy_quant"));
        raw.remove(&format!("{base}.input_scale"));
        raw.remove(&format!("{base}.weight_scale"));
    }
    raw.insert(
        format!("{FF}.scale_weight"),
        (vec![1], "F16".into(), 0x4100u16.to_le_bytes().into()),
    );
    raw.insert(
        "scaled_fp8".into(),
        (vec![1], "F32".into(), 1f32.to_le_bytes().into()),
    );
    let library = tempfile::tempdir().unwrap();
    let file = library.path().join("unknown-name.safetensors");
    write_file(&file, &raw, Some(meta));
    let data = tempfile::tempdir().unwrap();
    let store = CheckpointPlanStore::open(data.path());
    let root = store.approve_root(library.path()).unwrap();
    let plan = store
        .compile_linked(&root.root_id, "unknown-name.safetensors")
        .unwrap();
    let resolved = store.resolve(&plan.checkpoint_id).unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    let request = inventory.derivative_request();
    let derivatives = CheckpointDerivativeStore::open(data.path()).unwrap();
    derivatives
        .ensure(&plan.checkpoint_id, &request, |out| inventory.produce(out))
        .unwrap();
    let dir = derivatives
        .cache()
        .bundle_path(&derivatives.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let raw = read_file(&dir.join(inventory.shard_name(0)));
    assert_eq!(
        raw["transformer_blocks.0.ff.net.0.proj.scale_weight"],
        (vec![1], "F16".into(), 0x4100u16.to_le_bytes().into())
    );
    assert_eq!(raw["scaled_fp8"].2, 1f32.to_le_bytes());
    assert!(!raw.contains_key("transformer_blocks.0.attn.to_q.weight_scale"));
    assert!(!raw.contains_key("transformer_blocks.0.attn.to_q.comfy_quant"));
    assert_eq!(inventory.source_encoding(), Some("fp8_e5m2_scalar"));
}
