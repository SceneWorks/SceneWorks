//! Synthetic dense MiniMax-H3 DiT import fixtures (sc-26680), shared by the model-import job tests
//! and the video arm tests so both drive the SAME artifact: the published DiT key grammar at
//! fixture scale, inside a diffusers transformer component directory.

use std::path::{Path, PathBuf};

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda as platform_runtime;
#[cfg(target_os = "macos")]
use runtime_macos as platform_runtime;
use serde_json::{json, Map, Value};

/// Write a safetensors file declaring `entries` (one element each) filled with `fill`.
pub fn write_safetensors(path: &Path, entries: &[(&str, &str)], fill: u8) {
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
    let encoded = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, fill);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("dir creates");
    std::fs::write(path, bytes).expect("fixture writes");
}

/// A dense H3 transformer component directory at `dir`: the `MiniMaxH3Transformer3DModel` config
/// beside exactly one `diffusion_pytorch_model.safetensors`.
pub fn write_minimax_h3_dit_dir(dir: &Path) {
    let config = h3_fixture_config();
    let shapes = sceneworks_core::minimax_h3_codec::H3Config(config.clone())
        .shapes()
        .unwrap();
    let mut header = Map::new();
    let mut offset = 0_u64;
    for (name, shape) in shapes {
        let dtype = if name.starts_with("audio_") {
            "F32"
        } else {
            "BF16"
        };
        let size = shape.iter().product::<usize>() as u64 * if dtype == "F32" { 4 } else { 2 };
        header.insert(
            name,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+size]}),
        );
        offset += size;
    }
    let encoded = serde_json::to_vec(&header).unwrap();
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.resize(bytes.len() + offset as usize, 0x3a);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), bytes).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

pub fn h3_fixture_config() -> Value {
    let mut config = sceneworks_core::minimax_h3_codec::H3Config::published().0;
    for (key, value) in [
        ("num_layers", 1),
        ("num_refiner_layers", 1),
        ("num_attention_heads", 2),
        ("attention_head_dim", 6),
        ("hidden_size", 8),
        ("ffn_dim", 8),
        ("in_channels", 24),
        ("audio_in_channels", 32),
        ("text_dim", 5120),
        ("freq_dim", 4),
        ("time_embed_hidden_dim", 8),
        ("time_embed_dim", 4),
        ("rope_freq_dim", 1),
    ] {
        config[key] = json!(value);
    }
    config
}

/// A committed managed H3 import and the catalog entry the import job stamps for it.
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
pub struct ManagedMiniMaxH3Import {
    pub entry: Map<String, Value>,
    pub install_dir: PathBuf,
    pub checkpoint_id: String,
    pub plan_id: String,
    pub semantic_digest: String,
    _cache_root: Option<sceneworks_core::hf_home::test_support::CacheRootGuard>,
}

/// Commit a managed H3 DiT import into `data_dir` through the REAL transactional ingest and build
/// its entry with the SAME stamp the model-import job applies (`run_model_import_job`; the job test
/// `managed_h3_dit_import_stamps_a_selectable_video_entry` pins the job to this shape).
#[cfg_attr(
    not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )),
    allow(dead_code)
)]
pub fn stage_managed_minimax_h3_import(data_dir: &Path, model_id: &str) -> ManagedMiniMaxH3Import {
    stage_managed_minimax_h3_import_layout(data_dir, model_id, "dense")
}

pub fn stage_managed_minimax_h3_import_layout(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
) -> ManagedMiniMaxH3Import {
    stage_managed_minimax_h3_import_source(data_dir, model_id, layout, "fl2va")
}

pub fn stage_managed_minimax_h3_import_source(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
    partition: &str,
) -> ManagedMiniMaxH3Import {
    stage_managed_minimax_h3_import_source_with_tier(data_dir, model_id, layout, partition, "q4")
}

/// Select resident precision BEFORE the fixture's immutable plan is compiled, just as the import
/// API does. Editing an entry's tier after commitment is deliberately not a legitimate import.
pub fn stage_managed_minimax_h3_import_source_with_tier(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
    partition: &str,
    component_tier: &str,
) -> ManagedMiniMaxH3Import {
    assert!(matches!(component_tier, "q4" | "q8" | "bf16"));
    let cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new_if_unregistered(
        data_dir.to_owned(),
        data_dir.join("cache/fixture-hub"),
    );
    stage_component_snapshots(data_dir);
    let source = data_dir.join("models/incoming").join(model_id);
    write_minimax_h3_dit_dir(&source);
    let mut files = vec![
        "diffusion_pytorch_model.safetensors".to_owned(),
        "config.json".to_owned(),
    ];
    if matches!(
        layout,
        "native" | "convrot" | "pruned" | "pruned-other" | "w4a8"
    ) {
        let pruned = matches!(layout, "pruned" | "pruned-other" | "w4a8");
        let mut config = h3_fixture_config();
        if matches!(layout, "convrot" | "pruned" | "pruned-other" | "w4a8") {
            config["hidden_size"] = json!(256);
            config["attention_head_dim"] = json!(128);
            config["ffn_dim"] = json!(256);
            config["time_embed_dim"] = json!(64);
        }
        let source_config = sceneworks_core::minimax_h3_codec::H3Config(config.clone());
        let shapes = if pruned {
            source_config.pruned_shapes(1025, 8).unwrap()
        } else {
            source_config.shapes().unwrap()
        };
        let mut header = Map::new();
        header.insert(
            "__metadata__".into(),
            json!({"h3_config":config.to_string(),"h3_qkv_layout":"interleaved"}),
        );
        let mut payload = Vec::new();
        for (canonical, mut shape) in shapes {
            let table = canonical == "adaln_t_table";
            let reduced = pruned
                && (canonical.contains(".adaln_proj.linear.")
                    || canonical.starts_with("norm_out.linear."));
            let reduced_weight = reduced && canonical.ends_with(".weight");
            let folded_bias = reduced && canonical.ends_with(".bias");
            if canonical.ends_with(".attn.to_k.weight") || canonical.ends_with(".attn.to_v.weight")
            {
                continue;
            }
            let mut key = canonical
                .replace("transformer_blocks.", "blocks.")
                .replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
                .replace(".attn.to_out.0.", ".attn.out_proj.")
                .replace(".attn.norm_q.", ".attn.q_norm.")
                .replace(".attn.norm_k.", ".attn.k_norm.")
                .replace(".ff.net.0.proj.", ".mlp.fc1.")
                .replace(".ff.net.2.", ".mlp.fc2.");
            if key.ends_with(".attn.to_q.weight") {
                key = key.replace(".attn.to_q.weight", ".attn.qkv_proj.weight");
                shape[0] *= 3;
            }
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
                if let Some(rest) = key.strip_prefix(canonical) {
                    key = format!("{raw}{rest}");
                    break;
                }
            }
            let quantized = matches!(layout, "convrot" | "pruned" | "pruned-other" | "w4a8")
                && (key.starts_with("blocks.")
                    || (layout == "w4a8" && key.starts_with("token_refiner.blocks.")))
                && shape.len() == 2
                && key.ends_with(".weight")
                && (key.contains(".attn.")
                    || key.contains(".mlp.")
                    || (!pruned && key.contains(".adaln_proj.linear.")));
            let w4a8 = layout == "w4a8" && quantized && key.starts_with("blocks.");
            if w4a8 {
                shape[1] /= 2;
            }
            let dtype = if quantized {
                "I8"
            } else if reduced_weight {
                "BF16"
            } else if folded_bias {
                "F16"
            } else {
                "F32"
            };
            let offset = payload.len();
            if table {
                for row in 0..1025 {
                    for column in 0..8 {
                        let mut value = ((row as f32 / 1024.0) - 0.5) * (column + 1) as f32 / 32.0;
                        if layout == "pruned-other" && row == 0 && column == 3 {
                            value += 0.125;
                        }
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                }
            } else if reduced_weight {
                for row in 0..shape[0] {
                    for column in 0..shape[1] {
                        let value = ((row % 7 + 1) * (column + 1)) as f32 / 512.0;
                        payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                    }
                }
            } else if folded_bias {
                for row in 0..shape[0] {
                    let bits = [0xac00u16, 0xa800, 0, 0x2800, 0x2c00][row % 5];
                    payload.extend_from_slice(&bits.to_le_bytes());
                }
            } else {
                let size = shape.iter().product::<usize>() * if quantized { 1 } else { 4 };
                payload.resize(offset + size, if quantized { 0 } else { 0x3a });
                if quantized {
                    for row in 0..shape[0] {
                        payload[offset + row * shape[1]] = 58;
                    }
                }
            }
            header.insert(
                key.clone(),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,payload.len()]}),
            );
            if w4a8 {
                let base = key.strip_suffix(".weight").unwrap();
                let offset = payload.len();
                payload.resize(offset + shape[0] * shape[1] / 8, 0x38);
                header.insert(format!("{base}.weight_s_rel"), json!({"dtype":"F8_E4M3","shape":[shape[0],shape[1]/8],"data_offsets":[offset,payload.len()]}));
                let offset = payload.len();
                for _ in 0..shape[0] {
                    payload.extend(0.125f32.to_le_bytes());
                }
                header.insert(
                    format!("{base}.weight_s_channel"),
                    json!({"dtype":"F32","shape":[shape[0]],"data_offsets":[offset,payload.len()]}),
                );
                let offset = payload.len();
                for i in 0..16 {
                    payload.extend((i as f32 - 7.5).to_le_bytes());
                }
                header.insert(
                    format!("{base}.weight_codebook"),
                    json!({"dtype":"F32","shape":[16],"data_offsets":[offset,payload.len()]}),
                );
                let descriptor =
                    json!({"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256})
                        .to_string()
                        .into_bytes();
                let offset = payload.len();
                payload.extend_from_slice(&descriptor);
                header.insert(format!("{base}.comfy_quant"), json!({"dtype":"U8","shape":[descriptor.len()],"data_offsets":[offset,payload.len()]}));
            } else if quantized {
                let base = key.strip_suffix(".weight").unwrap();
                let offset = payload.len();
                for _ in 0..shape[0] {
                    payload.extend(0.125f32.to_le_bytes());
                }
                header.insert(format!("{base}.weight_scale"), json!({"dtype":"F32","shape":[shape[0],1],"data_offsets":[offset,payload.len()]}));
                let descriptor = json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":if shape[1] == 64 {64} else {256}}).to_string().into_bytes();
                let offset = payload.len();
                payload.extend_from_slice(&descriptor);
                header.insert(format!("{base}.comfy_quant"), json!({"dtype":"U8","shape":[descriptor.len()],"data_offsets":[offset,payload.len()]}));
            }
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.extend(payload);
        std::fs::write(source.join("renamed-user-checkpoint.safetensors"), bytes).unwrap();
        files = vec!["renamed-user-checkpoint.safetensors".into()];
    } else if layout == "sharded" {
        // Two real tensor-valued files plus an exact index; the provider must get BOTH.
        let file = source.join("diffusion_pytorch_model.safetensors");
        let bytes = std::fs::read(&file).unwrap();
        let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let original: Map<String, Value> = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
        let mut index = Map::new();
        files = vec![
            "config.json".into(),
            "diffusion_pytorch_model.safetensors.index.json".into(),
        ];
        for part in 0..2 {
            let name = format!("renamed-part-{part}.safetensors");
            let mut header = Map::new();
            let mut payload = Vec::new();
            for (i, (key, value)) in original.iter().enumerate() {
                if i % 2 != part {
                    continue;
                }
                let start = value["data_offsets"][0].as_u64().unwrap() as usize;
                let end = value["data_offsets"][1].as_u64().unwrap() as usize;
                let offset = payload.len();
                payload.extend_from_slice(&bytes[8 + len + start..8 + len + end]);
                let mut value = value.clone();
                value["data_offsets"] = json!([offset, payload.len()]);
                header.insert(key.clone(), value);
                index.insert(key.clone(), json!(name));
            }
            let encoded = serde_json::to_vec(&header).unwrap();
            let mut output = (encoded.len() as u64).to_le_bytes().to_vec();
            output.extend(encoded);
            output.extend(payload);
            std::fs::write(source.join(&name), output).unwrap();
            files.push(name);
        }
        std::fs::write(
            source.join("diffusion_pytorch_model.safetensors.index.json"),
            json!({"weight_map":index}).to_string(),
        )
        .unwrap();
        std::fs::remove_file(file).unwrap();
    }
    let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir);
    let ingest = sceneworks_core::checkpoint_ingest::ManagedIngest::begin(
        &store,
        model_id,
        sceneworks_core::checkpoint_import::ManagedProvenanceV1 {
            source: "local".to_owned(),
            ..Default::default()
        },
    )
    .expect("ingest begins");
    for file in &files {
        ingest
            .stage_copy_file(&source.join(file), file)
            .expect("stages");
    }
    let declaration = json!({"family":"minimax-h3","baseModel":if partition == "ref2va" {"minimax_h3_ref"} else {"minimax_h3"},
        "minimaxH3Partition":partition,"importComponentTier":component_tier,"downloads":component_downloads(component_tier,partition)});
    let context = captured_component_context(data_dir, declaration.as_object().unwrap());
    let install = ingest
        .finalize_with_context(
            if matches!(
                layout,
                "native" | "convrot" | "pruned" | "pruned-other" | "w4a8"
            ) {
                "renamed-user-checkpoint.safetensors"
            } else if layout == "sharded" {
                "renamed-part-0.safetensors"
            } else {
                "diffusion_pytorch_model.safetensors"
            },
            None,
            context,
        )
        .expect("a supported H3 DiT commits");
    let semantic_digest = install.compiled.record.plan.semantic_digest.clone();
    let mut entry = json!({
        "id": model_id,
        "name": "Imported H3 fine-tune",
        "type": "image",
        "family": "minimax-h3",
        "baseModel": if partition == "ref2va" { "minimax_h3_ref" } else { "minimax_h3" },
        "minimaxH3Partition": partition,
        "importComponentTier": component_tier,
        "downloads": component_downloads(component_tier, partition),
        "importPlan": {
            "checkpointId": install.checkpoint_id,
            "planId": install.compiled.plan.plan_id,
            "semanticDigest": semantic_digest,
        },
        "paths": { "model": install.install_path.display().to_string() },
    })
    .as_object()
    .cloned()
    .expect("object");
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
        &install.compiled.plan,
        &entry,
    )
    .expect("explicit complete assembly");
    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    sceneworks_core::lora_family::apply_model_manifest_defaults(
        &mut entry,
        "video",
        Some("minimax-h3"),
    );
    ManagedMiniMaxH3Import {
        entry,
        install_dir: install.install_path.clone(),
        checkpoint_id: install.checkpoint_id.clone(),
        plan_id: install.compiled.plan.plan_id.clone(),
        semantic_digest,
        _cache_root: cache_root,
    }
}

pub const SHARED_REVISION: &str = "1111111111111111111111111111111111111111";
pub const TIER_REVISION: &str = "2222222222222222222222222222222222222222";

/// The same source-content capture fresh managed/linked imports perform. Use only at explicit
/// fixture construction/recompilation, never in a render helper (which would bless drift).
pub fn captured_component_context(
    data_dir: &Path,
    entry: &Map<String, Value>,
) -> std::collections::BTreeMap<String, Value> {
    use sceneworks_core::model_artifacts::provenance::{
        capture_h3_component_closure_with_check, import_declaration_context, MODEL_DECLARATION_KEY,
    };
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    assert!(
        library.root().starts_with(data_dir.parent().unwrap()),
        "component capture fixtures require a private per-data cache guard"
    );
    // Project the actual typed import declaration, not later catalog-derived codec annotations.
    // Source quantization is independently derived and bound by the checkpoint inspector.
    let declaration = json!({"family":entry.get("family"),"baseModel":entry.get("baseModel"),
        "minimaxH3Partition":entry.get("minimaxH3Partition"),"importComponentTier":entry.get("importComponentTier"),
        "downloads":entry.get("downloads")});
    let declaration = declaration.as_object().unwrap();
    let mut context =
        import_declaration_context(declaration).expect("fixture declaration is valid");
    context
        .get_mut(MODEL_DECLARATION_KEY)
        .expect("fixture declaration context")["componentClosure"] =
        capture_h3_component_closure_with_check(data_dir, declaration, &|| Ok(()))
            .expect("complete real fixture component closure");
    context
}

/// The ACTUAL snapshot roots an import binds its shared components against: the private
/// source-library copies [`stage_component_snapshots`] materialized inside the data dir, not the
/// operator tier/base scratch trees a test happened to stage first. Mutating or asserting against
/// any other root no longer describes the committed components.
pub fn shared_component_snapshot(data_dir: &Path) -> PathBuf {
    component_snapshot(data_dir, "MiniMaxAI/MiniMax-H3", SHARED_REVISION)
}

/// The tiered component snapshot root (the `SceneWorks/minimax-h3-mlx` rehost) an import binds.
pub fn tier_component_snapshot(data_dir: &Path) -> PathBuf {
    component_snapshot(data_dir, "SceneWorks/minimax-h3-mlx", TIER_REVISION)
}

fn component_snapshot(data_dir: &Path, repo: &str, revision: &str) -> PathBuf {
    sceneworks_core::hf_home::model_source_library(data_dir)
        .repository_root(repo)
        .expect("fixture component repository")
        .join("snapshots")
        .join(revision)
}

/// Reset the private bound component snapshots to their committed fixture content. A test that
/// mutates or removes a bound component restores the exact bytes the import's frozen closure was
/// captured against before its next case, so later cases never inherit earlier mutations.
pub fn restage_component_snapshots(data_dir: &Path) {
    assert!(
        sceneworks_core::hf_home::model_source_library(data_dir)
            .root()
            .starts_with(data_dir.parent().unwrap()),
        "fixture reset must never address a real/shared HF cache"
    );
    for repo in ["MiniMaxAI/MiniMax-H3", "SceneWorks/minimax-h3-mlx"] {
        let root = sceneworks_core::hf_home::model_source_library(data_dir)
            .repository_root(repo)
            .expect("fixture component repository");
        if root.exists() {
            std::fs::remove_dir_all(&root).expect("fixture snapshot reset removes the repo tree");
        }
    }
    stage_component_snapshots(data_dir);
}

/// Populate only synthetic source-library snapshots selected by the fixture's private data root.
/// Callers retain their scoped CacheRootGuard; this helper never reads network or model payloads.
pub fn stage_component_snapshots(data_dir: &Path) {
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    assert!(
        library.root().starts_with(data_dir.parent().unwrap()),
        "fixture component snapshots require a private data-root cache guard"
    );
    let base = library
        .repository_root("MiniMaxAI/MiniMax-H3")
        .unwrap()
        .join("snapshots")
        .join(SHARED_REVISION);
    let tiers = library
        .repository_root("SceneWorks/minimax-h3-mlx")
        .unwrap()
        .join("snapshots")
        .join(TIER_REVISION);
    fn copy_missing(source: &Path, target: &Path) {
        std::fs::create_dir_all(target).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let destination = target.join(entry.file_name());
            if entry.path().is_dir() {
                copy_missing(&entry.path(), &destination);
            } else if !destination.exists() {
                std::fs::copy(entry.path(), destination).unwrap();
            }
        }
    }
    let scratch = data_dir.join("cache/synthetic-component-staging");
    let scratch_base = scratch.join("base");
    let scratch_tiers = scratch.join("tiers");
    for tier in ["q4", "q8", "bf16"] {
        let encoder = if tier == "bf16" {
            scratch_base.join("text_encoder")
        } else {
            scratch_tiers.join(tier).join("text_encoder")
        };
        write_shared_configs(&scratch_base, &encoder);
        write_companion(&scratch_tiers.join(tier).join("transformer_ref"));
        write_companion(&scratch_tiers.join(tier).join("transformer"));
    }
    copy_missing(&scratch_base, &base);
    copy_missing(&scratch_tiers, &tiers);
}

pub fn component_downloads(tier: &str, partition: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    for (id, dir) in [
        ("tokenizer", "tokenizer"),
        ("video_vae", "vae"),
        ("audio_vae", "audio_vae"),
        ("fl2va_audio_vae_config", "FL2VA/audio_vae"),
    ] {
        rows.push(json!({"provider":"huggingface","repo":"MiniMaxAI/MiniMax-H3","revision":SHARED_REVISION,"coRequisite":true,"componentId":id,"subdir":dir,"files":if id == "fl2va_audio_vae_config" { json!(["FL2VA/audio_vae/config.json","FL2VA/audio_vae/config.yaml","FL2VA/audio_vae/metadata.json"]) } else { json!([format!("{dir}/*")]) }}));
    }
    let encoder = if tier == "bf16" {
        "text_encoder".to_owned()
    } else {
        format!("{tier}/text_encoder")
    };
    rows.push(json!({"provider":"huggingface","repo":if tier == "bf16" { "MiniMaxAI/MiniMax-H3" } else { "SceneWorks/minimax-h3-mlx" },"revision":if tier == "bf16" {SHARED_REVISION} else {TIER_REVISION},"coRequisite":true,"variant":tier,"componentId":"text_encoder","subdir":encoder,"files":[format!("{encoder}/*")]}));
    if partition != "hybrid" {
        let companion = if partition == "ref2va" {
            "transformer"
        } else {
            "transformer_ref"
        };
        let dir = format!("{tier}/{companion}");
        rows.push(json!({"provider":"huggingface","repo":"SceneWorks/minimax-h3-mlx","revision":TIER_REVISION,"coRequisite":true,"variant":tier,"subdir":dir,"files":[format!("{dir}/*")]}));
    }
    rows
}

pub fn write_companion(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    write_safetensors(&dir.join("fixture.safetensors"), &[("weight", "F16")], 0x2a);
    std::fs::write(
        dir.join("diffusion_pytorch_model.safetensors.index.json"),
        json!({"weight_map":{"weight":"fixture.safetensors"}}).to_string(),
    )
    .unwrap();
}

#[cfg(all(test, target_os = "macos"))]
pub fn install_fixture_snapshots(hub: &Path, tiers: &Path, base: &Path) {
    for (repo, revision, source) in [
        ("models--MiniMaxAI--MiniMax-H3", SHARED_REVISION, base),
        ("models--SceneWorks--minimax-h3-mlx", TIER_REVISION, tiers),
    ] {
        let snapshot = hub.join(repo).join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        for name in [
            "q4",
            "q8",
            "bf16",
            "tokenizer",
            "vae",
            "audio_vae",
            "FL2VA",
            "text_encoder",
        ] {
            let target = source.join(name);
            let link = snapshot.join(name);
            if target.exists() && std::fs::symlink_metadata(&link).is_err() {
                std::os::unix::fs::symlink(target, link).unwrap();
            }
        }
    }
}

/// Portable constructor metadata for import-only CPU tests; no native provider is linked here.
#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub fn write_shared_configs(root: &Path, encoder: &Path) {
    let put = |path: PathBuf, value: Value| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, value.to_string()).unwrap();
    };
    put(
        encoder.join("config.json"),
        json!({"text_config":{"hidden_size":5120}}),
    );
    put(
        encoder.join("model.safetensors.index.json"),
        json!({"weight_map":{"weight":"fixture.safetensors"}}),
    );
    write_safetensors(
        &encoder.join("fixture.safetensors"),
        &[("weight", "F16")],
        0x2a,
    );
    for (dir, channels) in [("vae", 24), ("audio_vae", 32)] {
        put(
            root.join(dir).join("config.json"),
            json!({"latent_channels":channels}),
        );
        write_safetensors(
            &root.join(dir).join("fixture.safetensors"),
            &[("weight", "F16")],
            0x2b,
        );
    }
    put(
        root.join("tokenizer/tokenizer.json"),
        json!({"model":{"type":"BPE","vocab":{},"merges":[]}}),
    );
    put(root.join("tokenizer/tokenizer_config.json"), json!({}));
    for name in ["config.json", "metadata.json"] {
        put(root.join("FL2VA/audio_vae").join(name), json!({}));
    }
    std::fs::write(
        root.join("FL2VA/audio_vae/config.yaml"),
        "model_config: {}\n",
    )
    .unwrap();
}

/// Constructor documents at fixture scale; no model weights, downloads, or tensor runtime.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub fn write_shared_configs(root: &Path, encoder: &Path) {
    let put = |path: PathBuf, value: Value| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, value.to_string()).unwrap();
    };
    put(
        encoder.join("config.json"),
        json!({
            "model_type":"qwen3_vl", "image_token_id":151655, "video_token_id":151656,
            "vision_start_token_id":151652, "vision_end_token_id":151653,
            "text_config":{"model_type":"qwen3_vl_text", "hidden_size":5120,
                "num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,
                "head_dim":128,"intermediate_size":25600,"rms_norm_eps":0.000001,
                "rope_theta":5000000,"vocab_size":151936,
                "rope_scaling":{"mrope_interleaved":true,"mrope_section":[24,20,20]}}
        }),
    );
    put(
        encoder.join("model.safetensors.index.json"),
        json!({"weight_map":{"weight":"fixture.safetensors"}}),
    );
    write_safetensors(
        &encoder.join("fixture.safetensors"),
        &[("weight", "F16")],
        0x2a,
    );
    write_safetensors(
        &root.join("vae/fixture.safetensors"),
        &[("weight", "F16")],
        0x2b,
    );
    write_safetensors(
        &root.join("audio_vae/fixture.safetensors"),
        &[("weight", "F16")],
        0x2c,
    );
    let video = platform_runtime::providers::minimax_h3::config::MiniMaxH3VaeConfig::default();
    put(
        root.join("vae/config.json"),
        json!({
            "latent_channels":video.latent_channels,"out_channels":video.out_channels,
            "decoder_num_layers":video.num_layers,"decoder_num_attention_heads":video.num_heads,
            "decoder_attention_head_dim":video.head_dim,"decoder_num_register_tokens":video.num_register_tokens,
            "decoder_ffn_mult":video.ffn_mult,"decoder_rope_theta":video.rope_theta,
            "decoder_rope_dim_ratio":video.rope_dim_ratio,"decoder_norm_eps":video.norm_eps,
            "clip_length":video.clip_length,"token_drop":video.token_drop,
            "latents_mean":video.latents_mean,"latents_std":video.latents_std,
            "in_channels":video.in_channels,"block_out_channels":video.block_out_channels,
            "layers_per_block":video.layers_per_block,"spatial_downsample_factors":video.spatial_downsample_factors,
            "temporal_downsample_factors":video.temporal_downsample_factors,"norm_num_groups":video.norm_num_groups,
            "norm_eps":video.encoder_norm_eps
        }),
    );
    let source = json!({"_class_name":"MiniMaxH3AudioVAE","output_channel":2,"sample_rate":32000,
        "source_config_path":"config.yaml","source_metadata_path":"metadata.json",
        "source_safetensors_path":"model.safetensors","latent_channels":32,
        "latents_mean":vec![0.0;32],"latents_std":vec![1.0;32]});
    put(root.join("FL2VA/audio_vae/config.json"), source);
    std::fs::write(root.join("FL2VA/audio_vae/config.yaml"), "model_config:\n  sr: 32000\n  decoder_dim: 1024\n  audio_channel: 1\n  vae_latent_channels: 32\n").unwrap();
    put(
        root.join("FL2VA/audio_vae/metadata.json"),
        json!({"metadata":{"kwargs":{
            "attn_proj":true,"decoder_dim":1024,"decoder_rates":[5,5,2,2,2,2,2],"decoder_type":"bigvgan",
            "encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_dim":2048,"sample_rate":32000,"vae_latent_channels":32
        }}}),
    );
    put(
        root.join("audio_vae/config.json"),
        json!({"decoder_dim":1024,
        "decoder_kernel_sizes":[9,9,4,4,4,4,4],"decoder_rates":[5,5,2,2,2,2,2],
        "encoder_dim":64,"encoder_rates":[2,4,4,5,5],"latent_channels":32,"latent_dim":2048,
        "num_attention_heads":8,"resblock_dilation_sizes":[[1,3,5],[1,3,5],[1,3,5]],
        "resblock_kernel_sizes":[3,7,11],"sampling_rate":32000,"latents_mean":vec![0.0;32],"latents_std":vec![1.0;32]}),
    );
    put(
        root.join("tokenizer/tokenizer.json"),
        json!({"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[
            {"content":"<|vision_start|>","id":151652},{"content":"<|vision_end|>","id":151653},
            {"content":"<|image_pad|>","id":151655},{"content":"<|video_pad|>","id":151656}
        ]}),
    );
    put(
        root.join("tokenizer/tokenizer_config.json"),
        json!({"additional_special_tokens":platform_runtime::providers::minimax_h3::text_encoder::MINIMAX_ADDED_SPECIALS}),
    );
}

#[test]
fn convrot_managed_import_binds_declared_assembly_and_canonical_native_operands() {
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;
    use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
    use sceneworks_core::checkpoint_weight_facts::{
        CheckpointWeightFactsV1, MaterializationUnavailable, NativeExecutionCapabilityFact,
        SourceBindingFact, CONVROT_INT8_CODEC_ID,
    };
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let data = tempfile::tempdir().unwrap();
    let imported =
        stage_managed_minimax_h3_import_source(data.path(), "user-int8", "convrot", "ref2va");
    let plans = CheckpointPlanStore::open(data.path());
    let resolved = plans.resolve(&imported.checkpoint_id).unwrap();
    let assembly =
        sceneworks_core::minimax_h3_import::verify_entry_assembly(&imported.entry, &resolved.plan)
            .unwrap();
    assert_eq!(assembly.partition.as_str(), "ref2va");
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    assert!(inventory.convrot_int8);
    assert_eq!(imported.entry["importQuantFormat"], "convrot_int8");
    let codec = sceneworks_core::jobs_store::imported_entry_source_codec(&imported.entry).unwrap();
    assert_eq!(codec, CONVROT_INT8_CODEC_ID);
    let source = &resolved.layers[0].path;
    let facts = CheckpointWeightFactsV1::declared_source(
        Some(SourceBindingFact::new(
            source.file_name().and_then(|name| name.to_str()),
            std::fs::metadata(source).unwrap().len(),
        )),
        NativeExecutionCapabilityFact::dense_only(),
        codec,
        MaterializationUnavailable::NoRuntimeReceipt,
    )
    .unwrap();
    let persisted = serde_json::to_string(&facts).unwrap();
    let replayed: CheckpointWeightFactsV1 = serde_json::from_str(&persisted).unwrap();
    assert_eq!(replayed, facts);
    assert_eq!(
        replayed.executes_natively(codec),
        None,
        "CPU source proof never fabricates execution evidence"
    );
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&imported.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let _lease = cache.acquire(&resolved, &request).unwrap().unwrap();
    assert_eq!(request.codec_id, "convrot-int8-native");
}

/// Synthetic managed import -> sealed derivative -> H3 native projection. Executing Kitchen
/// f8b7e97's regular H4/Kronecker `_rotate_activation` on CPU with x=e3, codes=58*e0 and
/// scale=.125 yields BF16 -0.453125 (group256), -0.90625 (group64); no rotation yields zero
/// and a Sylvester first column has the wrong sign. No model checkpoint is downloaded.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires coordinator-reserved local Metal token; tiny synthetic projections only"]
fn convrot_import_native_mlx_projection_smoke() {
    use platform_runtime::providers::minimax_h3 as h3;
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;
    use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let data = tempfile::tempdir().unwrap();
    let imported =
        stage_managed_minimax_h3_import_source(data.path(), "native-int8", "convrot", "ref2va");
    let resolved = CheckpointPlanStore::open(data.path())
        .resolve(&imported.checkpoint_id)
        .unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&imported.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let _lease = cache.acquire(&resolved, &request).unwrap().unwrap();
    let output = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let weights = platform_runtime::media::weights::Weights::from_file(
        output.join("diffusion_pytorch_model-00001-of-00001.safetensors"),
    )
    .unwrap();
    for (prefix, input, rows, want) in [
        ("transformer_blocks.0.attn.to_q", 256, 256, -0.453125),
        ("transformer_blocks.0.adaln_proj.linear", 64, 4608, -0.90625),
    ] {
        let projection = h3::quant::lin(&weights, prefix, false).unwrap();
        let native = projection.convrot_base().unwrap();
        assert_eq!(native.descriptor().rotation_group, input);
        assert_eq!(native.descriptor().out_features, rows);
        assert!(projection.dense_weight().is_none());
        let mut activation = vec![0f32; input];
        activation[3] = 1.0;
        let x = mlx_rs::Array::from_slice(&activation, &[1, input as i32])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let output = projection
            .forward(&x)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        output.eval().unwrap();
        assert_eq!(output.shape(), &[1, rows as i32]);
        assert!(output.as_slice::<f32>().iter().all(|&value| value == want));
        println!("{prefix}: native ConvRot group{input}, all outputs={want}, no dense base");
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "requires coordinator-reserved CUDA runner; tiny synthetic projections only"]
fn convrot_import_native_cuda_projection_smoke() {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use platform_runtime::providers::minimax_h3 as h3;
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;
    use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let data = tempfile::tempdir().unwrap();
    let imported =
        stage_managed_minimax_h3_import_source(data.path(), "native-int8", "convrot", "ref2va");
    let resolved = CheckpointPlanStore::open(data.path())
        .resolve(&imported.checkpoint_id)
        .unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    let cache = CheckpointDerivativeStore::open(data.path()).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&imported.checkpoint_id, &request, |out| {
            inventory.produce(out)
        })
        .unwrap();
    let _lease = cache.acquire(&resolved, &request).unwrap().unwrap();
    let output = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let device = Device::new_cuda(0).expect("CUDA fixture must execute, never skip-as-pass");
    // Match MiniMaxH3Dit::load_from_dir: preserve I8 codes and F32 row scales through
    // the validated native reader, not Candle's generic safetensors dtype loader.
    let weights = candle_gen::Weights::from_files_native_int8(
        &[output.join("diffusion_pytorch_model-00001-of-00001.safetensors")],
        &device,
    )
    .unwrap();
    for (prefix, input, rows, want) in [
        ("transformer_blocks.0.attn.to_q", 256, 256, -0.453125),
        ("transformer_blocks.0.adaln_proj.linear", 64, 4608, -0.90625),
    ] {
        let loaded = h3::quant::lin(&weights, h3::quant::DIT, prefix, false, DType::BF16).unwrap();
        assert_eq!(loaded.base_bytes, rows * (input + 4));
        let mut activation = vec![0f32; input];
        activation[3] = 1.0;
        let x = Tensor::from_vec(activation, (1, input), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let values = loaded
            .linear
            .forward_upcast(&x)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(values.len(), rows);
        assert!(values.iter().all(|&value| value == want));
        println!(
            "{prefix}: native ConvRot group{input}, all outputs={want}, {} base bytes",
            loaded.base_bytes
        );
    }
}

#[cfg(test)]
struct PrunedManagedArtifact {
    imported: ManagedMiniMaxH3Import,
    binding: sceneworks_core::minimax_h3_codec::H3PruningBinding,
    output: PathBuf,
    _lease: sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheLease,
}

#[cfg(test)]
fn stage_sealed_pruned_artifact(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
) -> PrunedManagedArtifact {
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;
    use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let imported = stage_managed_minimax_h3_import_source(data_dir, model_id, layout, "ref2va");
    let resolved = CheckpointPlanStore::open(data_dir)
        .resolve(&imported.checkpoint_id)
        .unwrap();
    let assembly =
        sceneworks_core::minimax_h3_import::verify_entry_assembly(&imported.entry, &resolved.plan)
            .unwrap();
    let inventory = H3SourceInventory::from_resolved(&resolved).unwrap();
    sceneworks_core::minimax_h3_import::verify_imported_pruning(&assembly, &inventory).unwrap();
    let binding = inventory
        .pruning
        .clone()
        .expect("source supplies the complete reduced topology");
    assert_eq!(assembly.pruning.as_ref(), Some(&binding));
    assert!(
        inventory.convrot_int8,
        "body ConvRot remains the actual numerical codec"
    );
    assert!(inventory.requires_normalization);
    let cache = CheckpointDerivativeStore::open(data_dir).unwrap();
    let request = inventory.derivative_request();
    cache
        .ensure(&imported.checkpoint_id, &request, |outputs| {
            inventory.produce(outputs)
        })
        .unwrap();
    let lease = cache.acquire(&resolved, &request).unwrap().unwrap();
    let output = cache
        .cache()
        .bundle_path(&cache.cache_key(&resolved, &request).unwrap())
        .unwrap();
    let files = request
        .outputs
        .iter()
        .map(|name| (name.clone(), output.join(name)))
        .collect::<Vec<_>>();
    let normalized = H3SourceInventory::from_files(&files).unwrap();
    assert_eq!(
        normalized.pruning.as_ref(),
        Some(&binding),
        "normalization preserves the complete engine pruning identity"
    );
    assert_eq!(normalized.config.0["num_layers"], 1);
    assert_eq!(normalized.config.0["hidden_size"], 256);
    assert_eq!(
        normalized.config.0["time_embed_dim"], 64,
        "source architecture is not rewritten to rank8 or stock geometry"
    );
    PrunedManagedArtifact {
        imported,
        binding,
        output,
        _lease: lease,
    }
}

#[test]
fn pruned_managed_import_seals_source_precision_and_whole_basis_identity() {
    use gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype};
    use sceneworks_core::checkpoint_plan_store::CheckpointPlanStore;
    use sceneworks_core::minimax_h3_codec::H3SourceInventory;
    let data = tempfile::tempdir().unwrap();
    let first = stage_sealed_pruned_artifact(data.path(), "pruned-source", "pruned");
    let other = stage_sealed_pruned_artifact(data.path(), "different-basis", "pruned-other");
    assert_eq!(first.imported.entry["importQuantFormat"], "convrot_int8");
    assert_eq!(first.binding.grid, 1025);
    assert_eq!(first.binding.rank, 8);
    assert_ne!(first.binding.identity, other.binding.identity);
    assert_ne!(
        first.output, other.output,
        "a foreign basis cannot reuse the derivative/provider directory"
    );
    assert!(first.binding.source_bytes < first.binding.compute_bytes);
    let shard = first
        .output
        .join("diffusion_pytorch_model-00001-of-00001.safetensors");
    let headers = safetensors_path_tensor_headers(&shard).unwrap();
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    {
        use platform_runtime::providers::minimax_h3 as h3;
        let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(data.path().into()))
            .with_component(
                "transformer",
                gen_core::WeightsSource::Dir(first.output.clone()),
            );
        let contract = h3::memory_strategy::contract_for(&spec).unwrap();
        let rows = gen_core_h3::pruned_adaln::SourcePruneSizing::conditioned_timestep_bound(
            h3::model::MAX_STEPS as usize,
        )
        .unwrap() as u64;
        // Independent fixture shape/dtype formula: one 256-wide block (18 modulation groups),
        // final shift/scale, rank-eight coordinates, all retained at F32. This is a test of the
        // provider declaration, not a second worker estimator or a stock-geometry substitution.
        let retained = rows * (256 * (18 + 2) + 8) * 4;
        // Independent fixture inventory: file framing/descriptors are not resident tensors.
        // The ordinary body norms and dense refiner materialize at BF16; native codes/scales/
        // biases and pruned operands retain their source width before the pruning correction.
        let (hidden, head, attention, ffn) = (256_usize, 128_usize, 256_usize, 256_usize);
        let mut cast_shapes = vec![(
            String::from("token_refiner.final_norm.weight"),
            vec![hidden],
        )];
        for container in ["transformer_blocks.0", "token_refiner.refiner_blocks.0"] {
            for norm in ["norm1", "norm2"] {
                cast_shapes.push((format!("{container}.{norm}.weight"), vec![hidden]));
            }
            for norm in ["norm_q", "norm_k"] {
                cast_shapes.push((format!("{container}.attn.{norm}.weight"), vec![head]));
            }
        }
        for projection in ["to_q", "to_k", "to_v"] {
            cast_shapes.push((
                format!("token_refiner.refiner_blocks.0.attn.{projection}.weight"),
                vec![attention, hidden],
            ));
        }
        cast_shapes.extend([
            (
                String::from("token_refiner.refiner_blocks.0.attn.to_out.0.weight"),
                vec![hidden, attention],
            ),
            (
                String::from("token_refiner.refiner_blocks.0.ff.net.0.proj.weight"),
                vec![2 * ffn, hidden],
            ),
            (
                String::from("token_refiner.refiner_blocks.0.ff.net.2.weight"),
                vec![hidden, ffn],
            ),
        ]);
        let mut cast_savings = 0_u64;
        for (name, shape) in cast_shapes {
            let header = headers
                .iter()
                .find(|header| header.name == name)
                .expect(&name);
            assert_eq!(header.dtype, Dtype::F32, "{name}");
            assert_eq!(header.shape, shape, "{name}");
            let elements = shape.iter().product::<usize>() as u64;
            assert_eq!(header.data_bytes, elements * 4, "{name}");
            cast_savings += elements * (4 - 2);
        }
        let payload_bytes = headers
            .iter()
            .filter(|header| !header.name.ends_with(".comfy_quant"))
            .map(|header| header.data_bytes)
            .sum::<u64>();
        let descriptors = gen_core::weightsmeta::read_safetensors_path_tensor_payloads(
            &shard,
            |header| header.name.ends_with(".comfy_quant"),
            4096,
        )
        .unwrap();
        let groups = descriptors
            .values()
            .map(|bytes| {
                let descriptor: Value = serde_json::from_slice(bytes).unwrap();
                assert_eq!(descriptor["format"], "int8_tensorwise");
                assert_eq!(descriptor["convrot"], true);
                descriptor["convrot_groupsize"].as_u64().unwrap()
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(groups, std::collections::BTreeSet::from([256]));
        let rotations = groups.iter().map(|group| 4 * group * group).sum::<u64>();
        // Candle owns one shared cuBLASLt workspace; MLX does not. Constructor overlap is
        // a separate peak and must not be added to this steady resident inventory.
        let backend_workspace = if cfg!(target_os = "macos") {
            0
        } else {
            32 * 1024 * 1024
        };
        let corrected = payload_bytes - cast_savings + rotations + backend_workspace
            - first.binding.source_bytes
            + first.binding.compute_bytes
            + retained;
        assert_eq!(contract.asset_facts.transformer_bytes, corrected);
        assert_eq!(
            contract.resident_components()[0].resident_bytes,
            first.binding.compute_bytes + retained
        );
        assert_eq!(
            contract.evicted_component_bytes(),
            0,
            "the actual imported pruned operands are retained, not fictitiously evicted"
        );
        assert_eq!(contract.steady_state_transformer_bytes(), corrected);
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
    }
    for (key, dtype, shape) in [
        ("adaln_t_table", Dtype::F32, vec![1025, 8]),
        (
            "transformer_blocks.0.adaln_proj.linear.weight",
            Dtype::BF16,
            vec![4608, 8],
        ),
        (
            "transformer_blocks.0.adaln_proj.linear.bias",
            Dtype::F16,
            vec![4608],
        ),
        ("norm_out.linear.weight", Dtype::BF16, vec![512, 8]),
        ("norm_out.linear.bias", Dtype::F16, vec![512]),
    ] {
        let header = headers.iter().find(|header| header.name == key).expect(key);
        assert_eq!(header.dtype, dtype, "{key}");
        assert_eq!(header.shape, shape, "{key}");
    }
    assert!(!headers
        .iter()
        .any(|header| header.name.starts_with("time_embedder.")));
    assert!(!headers
        .iter()
        .any(|header| header.name.contains("adaln_proj.linear.comfy_quant")));
    let store = CheckpointPlanStore::open(data.path());
    let first_plan = store.resolve(&first.imported.checkpoint_id).unwrap();
    let other_plan = store.resolve(&other.imported.checkpoint_id).unwrap();
    let first_assembly = sceneworks_core::minimax_h3_import::verify_entry_assembly(
        &first.imported.entry,
        &first_plan.plan,
    )
    .unwrap();
    let foreign = H3SourceInventory::from_resolved(&other_plan).unwrap();
    assert!(
        sceneworks_core::minimax_h3_import::verify_imported_pruning(&first_assembly, &foreign)
            .is_err()
    );
    let mut forged = first.imported.entry.clone();
    forged["importPlan"]["assembly"]["adalnPruning"]["identity"] = json!(other.binding.identity);
    assert!(
        sceneworks_core::minimax_h3_import::verify_entry_assembly(&forged, &first_plan.plan)
            .is_err()
    );
}

#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
fn pruned_import_oracle() -> Value {
    serde_json::from_str(include_str!(
        "../tests/fixtures/minimax_h3_pruned_import.json"
    ))
    .unwrap()
}

#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
fn fixture_floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
fn assert_pruned_numeric(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "{label}[{index}] must be finite"
        );
        assert!(
            (got - want).abs() <= tolerance,
            "{label}[{index}]={got}, expected {want}, abs bound {tolerance}"
        );
    }
}

#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
fn assert_pruned_period(actual: &[f32], case: &Value, rows: usize, tolerance: f32, label: &str) {
    assert_eq!(actual.len(), rows, "{label}");
    let period = fixture_floats(&case["projection_period"]);
    assert_eq!(period.len(), 35);
    for (row, &value) in actual.iter().enumerate() {
        let expected = period[row % 35];
        assert!(value.is_finite() && expected.is_finite(), "{label}[{row}]");
        assert!(
            (value - expected).abs() <= tolerance,
            "{label}[{row}]={value}, expected {expected}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires explicit native Metal fixture workflow; never run in ordinary CI"]
fn w4a8_import_native_mlx_projection_smoke() {
    use mlx_rs::{Array, Dtype};
    use platform_runtime::providers::minimax_h3 as h3;
    let data = tempfile::tempdir().unwrap();
    let artifact = stage_sealed_pruned_artifact(data.path(), "packed-native-mlx", "w4a8");
    assert_eq!(
        artifact.imported.entry["importQuantFormat"],
        "asym_w4a8_int8"
    );
    let mut weights = platform_runtime::media::weights::Weights::from_file(
        artifact
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let cfg = h3::MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(artifact.output.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut model = h3::MiniMaxH3Dit::from_weights(&mut weights, &cfg, Dtype::Bfloat16).unwrap();
    model
        .ensure_pruning_identity(&artifact.binding.engine_identity().unwrap())
        .unwrap();
    model.verify_head_sources().unwrap();
    model.verify_adaln_sources().unwrap();
    model.verify_body_sources().unwrap();
    for (base, packed) in [
        ("transformer_blocks.0.attn.to_q", true),
        ("transformer_blocks.0.ff.net.0.proj", true),
        ("token_refiner.refiner_blocks.0.attn.to_q", false),
    ] {
        let linear = h3::quant::lin(&weights, base, false).unwrap();
        assert_eq!(linear.w4a8_base().is_some(), packed);
        assert_eq!(linear.convrot_base().is_some(), !packed);
        if let Some(q) = linear.w4a8_base() {
            let rows = if base.contains(".ff.") { 512 } else { 256 };
            assert_eq!(
                q.nbytes(),
                rows * (128 + 16 + 4) + 64,
                "only packed original operands are resident"
            );
        }
        let mut values = vec![0f32; 256];
        values[3] = 1.0;
        let x = Array::from_slice(&values, &[1, 256])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let out = linear
            .forward_upcast(&x)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        out.eval().unwrap();
        assert!(out.as_slice::<f32>().iter().all(|v| v.is_finite()));
        assert!(out.as_slice::<f32>().iter().any(|v| *v != 0.0));
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "requires explicit native CUDA fixture workflow; never run in ordinary CI"]
fn w4a8_import_native_cuda_projection_smoke() {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use platform_runtime::providers::minimax_h3 as h3;
    let device = Device::new_cuda(0).expect("CUDA fixture must execute, never skip-as-pass");
    let data = tempfile::tempdir().unwrap();
    let artifact = stage_sealed_pruned_artifact(data.path(), "packed-native-cuda", "w4a8");
    assert_eq!(
        artifact.imported.entry["importQuantFormat"],
        "asym_w4a8_int8"
    );
    let weights = candle_gen::Weights::from_files_native_int8(
        &[artifact
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors")],
        &device,
    )
    .unwrap();
    let cfg = h3::MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(artifact.output.join("config.json")).unwrap(),
    )
    .unwrap();
    let model = h3::MiniMaxH3Dit::from_weights(&weights, &cfg, &device, DType::BF16).unwrap();
    model
        .ensure_pruning_identity(&artifact.binding.engine_identity().unwrap())
        .unwrap();
    for (base, packed) in [
        ("transformer_blocks.0.attn.to_q", true),
        ("transformer_blocks.0.ff.net.0.proj", true),
        ("token_refiner.refiner_blocks.0.attn.to_q", false),
    ] {
        let loaded = h3::quant::lin(&weights, h3::quant::DIT, base, false, DType::BF16).unwrap();
        assert_eq!(loaded.linear.is_w4a8(), packed);
        assert_eq!(loaded.linear.is_convrot_int8(), !packed);
        if packed {
            let rows = if base.contains(".ff.") { 512 } else { 256 };
            assert_eq!(
                loaded.base_bytes,
                rows * (128 + 16 + 4) + 64,
                "only packed original operands are resident"
            );
        }
        let mut values = vec![0f32; 256];
        values[3] = 1.0;
        let x = Tensor::from_vec(values, (1, 256), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = loaded
            .linear
            .forward_upcast(&x)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| *v != 0.0));
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires native Metal; managed synthetic source only, never skip-as-pass"]
fn pruned_import_native_mlx_projection_smoke() {
    use h3::dit::model::{BlockModulation, PackedForward};
    use mlx_rs::{Array, Dtype};
    use platform_runtime::providers::minimax_h3 as h3;
    let data = tempfile::tempdir().unwrap();
    let artifact = stage_sealed_pruned_artifact(data.path(), "pruned-native-mlx", "pruned");
    let foreign = stage_sealed_pruned_artifact(data.path(), "foreign-pruned-mlx", "pruned-other");
    let mut weights = platform_runtime::media::weights::Weights::from_file(
        artifact
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let cfg = h3::MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(artifact.output.join("config.json")).unwrap(),
    )
    .unwrap();
    let expected = artifact.binding.engine_identity().unwrap();
    let runtime = h3::detect(&weights, &cfg).unwrap().unwrap();
    assert_eq!(runtime.identity(), expected);
    assert_eq!(runtime.nbytes() as u64, artifact.binding.compute_bytes);
    let mut model = h3::MiniMaxH3Dit::from_weights(&mut weights, &cfg, Dtype::Bfloat16).unwrap();
    model.ensure_pruning_identity(&expected).unwrap();
    model.verify_head_sources().unwrap();
    model.verify_adaln_sources().unwrap();
    model.verify_body_sources().unwrap();
    let oracle = pruned_import_oracle();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let timestep = case["timestep"].as_f64().unwrap() as f32;
        let coords = model.embed_timesteps(&[timestep]).unwrap();
        coords.eval().unwrap();
        assert_pruned_numeric(
            coords.as_slice::<f32>(),
            &fixture_floats(&case["coordinates"]),
            2e-6,
            "MLX imported table coordinates",
        );
        let block = runtime
            .block_projection(0)
            .unwrap()
            .projected(&coords)
            .unwrap();
        block.eval().unwrap();
        assert_pruned_period(
            block.as_slice::<f32>(),
            case,
            4608,
            2e-5,
            "MLX imported reduced block",
        );
        let final_projection = runtime.final_projection().projected(&coords).unwrap();
        final_projection.eval().unwrap();
        assert_pruned_period(
            final_projection.as_slice::<f32>(),
            case,
            512,
            2e-5,
            "MLX imported reduced final",
        );
    }
    let body = h3::quant::lin(&weights, "transformer_blocks.0.attn.to_q", false).unwrap();
    assert_eq!(
        body.convrot_base().unwrap().descriptor().rotation_group,
        256
    );
    assert!(body.dense_weight().is_none());
    let mut input = vec![0.0f32; 256];
    input[3] = 1.0;
    let body_out = body
        .forward(
            &Array::from_slice(&input, &[1, 256])
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        )
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    body_out.eval().unwrap();
    assert!(body_out
        .as_slice::<f32>()
        .iter()
        .all(|&value| value == -0.453125));
    let times = h3::PackedLayout::row_timesteps(0.23, 0.7);
    assert_eq!(times, [0.23, 0.7, 0.999, 1.0]);
    let schedule = h3::TimestepSchedule::new(vec![times.to_vec()]).unwrap();
    let coords = model.embed_timesteps(&times).unwrap();
    let norm = runtime.final_projection().modulation(&coords).unwrap();
    let cache = runtime.precompute(schedule.clone(), 1).unwrap();
    assert!(cache.is_current_for_pruned(&schedule, &expected));
    assert!(!cache.is_current_for(&schedule));
    let video = Array::from_slice(
        &vec![0.01f32; 2 * cfg.video_patch_dim() as usize],
        &[1, 2, cfg.video_patch_dim()],
    );
    let audio = Array::from_slice(
        &vec![0.02f32; 2 * cfg.audio_in_channels as usize],
        &[1, 2, cfg.audio_in_channels],
    );
    let text = Array::from_slice(
        &vec![0.03f32; cfg.hidden_size as usize],
        &[1, 1, cfg.hidden_size],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let indices = Array::from_slice(&[1i32, 6, 11, 5, 0], &[5]);
    let timesteps = Array::from_slice(&[0i32, 2, 3, 1, 0], &[5]);
    let rope = h3::MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(&Array::from_slice(&[0.0f32; 15], &[5, 3]))
        .unwrap();
    let packed = PackedForward {
        video_rows: &video,
        audio_rows: &audio,
        text_rows: &text,
        adaln_indices: &indices,
        timestep_indices: &timesteps,
        tables: &tables,
        text_indices: &[0],
        video_indices: &[1, 4],
        audio_indices: &[2, 3],
    };
    let cached = model
        .forward_packed(&packed, BlockModulation::Cached(&cache), &norm)
        .unwrap();
    let resident = model
        .forward_packed(&packed, BlockModulation::Temb(&coords), &norm)
        .unwrap();
    for (actual, reference) in [(&resident.0, &cached.0), (&resident.1, &cached.1)] {
        actual.eval().unwrap();
        reference.eval().unwrap();
        assert_pruned_numeric(
            actual.as_slice::<f32>(),
            reference.as_slice::<f32>(),
            2e-5,
            "MLX imported resident/cache full forward",
        );
    }
    let foreign_weights = platform_runtime::media::weights::Weights::from_file(
        foreign
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let foreign_runtime = h3::detect(&foreign_weights, &cfg).unwrap().unwrap();
    let foreign_cache = foreign_runtime.precompute(schedule, 1).unwrap();
    assert!(model
        .ensure_pruning_identity(&foreign.binding.engine_identity().unwrap())
        .is_err());
    let error = model
        .forward_packed(&packed, BlockModulation::Cached(&foreign_cache), &norm)
        .unwrap_err()
        .to_string();
    assert!(error.contains("pruning identity"), "{error}");
    println!("managed pruned import MLX: grid1025/rank8, stored BF16/F16, nine pinned numeric cases, native ConvRot group256=-0.453125, resident/cache video+audio forward, foreign basis refused; identity={}", artifact.binding.identity);
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "requires native CUDA; managed synthetic source only, never skip-as-pass"]
fn pruned_import_native_cuda_projection_smoke() {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use h3::dit::model::{BlockModulation, PackedForward};
    use platform_runtime::providers::minimax_h3 as h3;
    let device = Device::new_cuda(0).expect("CUDA fixture must execute, never skip-as-pass");
    let data = tempfile::tempdir().unwrap();
    let artifact = stage_sealed_pruned_artifact(data.path(), "pruned-native-cuda", "pruned");
    let foreign = stage_sealed_pruned_artifact(data.path(), "foreign-pruned-cuda", "pruned-other");
    let weights = candle_gen::Weights::from_files_native_int8(
        &[artifact
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors")],
        &device,
    )
    .unwrap();
    let cfg = h3::MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(artifact.output.join("config.json")).unwrap(),
    )
    .unwrap();
    let model = h3::MiniMaxH3Dit::from_weights(&weights, &cfg, &device, DType::BF16).unwrap();
    let expected = artifact.binding.engine_identity().unwrap();
    model.ensure_pruning_identity(&expected).unwrap();
    let curve = model.curve().unwrap();
    assert_eq!(curve.identity(), expected);
    assert_eq!(curve.nbytes() as u64, artifact.binding.compute_bytes);
    let flat = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let oracle = pruned_import_oracle();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let coords = model
            .curve_coordinates(&[case["timestep"].as_f64().unwrap() as f32])
            .unwrap();
        assert_pruned_numeric(
            &flat(&coords),
            &fixture_floats(&case["coordinates"]),
            2e-6,
            "CUDA imported table coordinates",
        );
        let block = curve.block_projection(0).unwrap().forward(&coords).unwrap();
        assert_pruned_period(
            &flat(&block),
            case,
            4608,
            2e-5,
            "CUDA imported reduced block",
        );
        let final_projection = curve
            .norm_out_projection()
            .unwrap()
            .forward(&coords)
            .unwrap();
        assert_pruned_period(
            &flat(&final_projection),
            case,
            512,
            2e-5,
            "CUDA imported reduced final",
        );
    }
    let body = h3::quant::lin(
        &weights,
        h3::quant::DIT,
        "transformer_blocks.0.attn.to_q",
        false,
        DType::BF16,
    )
    .unwrap();
    assert!(body.linear.is_convrot_int8());
    let mut input = vec![0.0f32; 256];
    input[3] = 1.0;
    let body_out = body
        .linear
        .forward_upcast(
            &Tensor::from_vec(input, (1, 256), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        )
        .unwrap();
    assert!(flat(&body_out).iter().all(|&value| value == -0.453125));
    let blocks = (0..cfg.num_layers)
        .map(|layer| {
            h3::DitBlock::from_weights_curve(
                &weights,
                &format!("transformer_blocks.{layer}"),
                &cfg,
                DType::BF16,
                curve.block_projection(layer).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let times = h3::PackedLayout::row_timesteps(0.23, 0.7);
    assert_eq!(times, [0.23, 0.7, 0.999, 1.0]);
    let schedule = h3::TimestepSchedule::new(vec![times.to_vec()]).unwrap();
    let coords = model.curve_coordinates(&times).unwrap();
    let (cache, norm) = curve.precompute(&blocks, &schedule).unwrap();
    assert!(cache.is_current_for_pruning(&schedule, expected));
    assert!(!cache.is_current_for(&schedule));
    let video = Tensor::from_vec(
        vec![0.01f32; 2 * cfg.video_patch_dim()],
        (1, 2, cfg.video_patch_dim()),
        &device,
    )
    .unwrap();
    let audio = Tensor::from_vec(
        vec![0.02f32; 2 * cfg.audio_in_channels],
        (1, 2, cfg.audio_in_channels),
        &device,
    )
    .unwrap();
    let text = Tensor::from_vec(
        vec![0.03f32; cfg.hidden_size],
        (1, 1, cfg.hidden_size),
        &device,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let indices = Tensor::from_vec(vec![1u32, 6, 11, 5, 0], (5,), &device).unwrap();
    let timesteps = Tensor::from_vec(vec![0u32, 2, 3, 1, 0], (5,), &device).unwrap();
    let rope = h3::MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope
        .tables(&Tensor::zeros((5, 3), DType::F32, &device).unwrap())
        .unwrap();
    let packed = PackedForward {
        video_rows: &video,
        audio_rows: &audio,
        text_rows: &text,
        adaln_indices: &indices,
        timestep_indices: &timesteps,
        tables: &tables,
        text_indices: &[0],
        video_indices: &[1, 4],
        audio_indices: &[2, 3],
    };
    let cached = model
        .forward_packed(&packed, BlockModulation::Cached(&cache), &norm)
        .unwrap();
    let resident = model
        .forward_packed(&packed, BlockModulation::CurveCoords(&coords), &norm)
        .unwrap();
    for (actual, reference) in [(&resident.0, &cached.0), (&resident.1, &cached.1)] {
        assert_pruned_numeric(
            &flat(actual),
            &flat(reference),
            2e-5,
            "CUDA imported resident/cache full forward",
        );
    }
    let foreign_weights = candle_gen::Weights::from_files_native_int8(
        &[foreign
            .output
            .join("diffusion_pytorch_model-00001-of-00001.safetensors")],
        &device,
    )
    .unwrap();
    let foreign_model =
        h3::MiniMaxH3Dit::from_weights(&foreign_weights, &cfg, &device, DType::BF16).unwrap();
    let foreign_curve = foreign_model.curve().unwrap();
    let foreign_blocks = (0..cfg.num_layers)
        .map(|layer| {
            h3::DitBlock::from_weights_curve(
                &foreign_weights,
                &format!("transformer_blocks.{layer}"),
                &cfg,
                DType::BF16,
                foreign_curve.block_projection(layer).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (foreign_cache, _) = foreign_curve
        .precompute(&foreign_blocks, &schedule)
        .unwrap();
    assert!(model
        .ensure_pruning_identity(&foreign.binding.engine_identity().unwrap())
        .is_err());
    let error = model
        .forward_packed(&packed, BlockModulation::Cached(&foreign_cache), &norm)
        .unwrap_err()
        .to_string();
    assert!(error.contains("pruning identity"), "{error}");
    println!("managed pruned import CUDA: grid1025/rank8, stored BF16/F16, nine pinned numeric cases, native ConvRot group256=-0.453125, resident/cache video+audio forward, foreign basis refused; identity={}", artifact.binding.identity);
}
