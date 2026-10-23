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
pub(crate) fn write_safetensors(path: &Path, entries: &[(&str, &str)], fill: u8) {
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
pub(crate) fn write_minimax_h3_dit_dir(dir: &Path) {
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

pub(crate) fn h3_fixture_config() -> Value {
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
pub(crate) struct ManagedMiniMaxH3Import {
    pub(crate) entry: Map<String, Value>,
    pub(crate) install_dir: PathBuf,
    pub(crate) checkpoint_id: String,
    pub(crate) plan_id: String,
    pub(crate) semantic_digest: String,
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
pub(crate) fn stage_managed_minimax_h3_import(
    data_dir: &Path,
    model_id: &str,
) -> ManagedMiniMaxH3Import {
    stage_managed_minimax_h3_import_layout(data_dir, model_id, "dense")
}

pub(crate) fn stage_managed_minimax_h3_import_layout(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
) -> ManagedMiniMaxH3Import {
    stage_managed_minimax_h3_import_source(data_dir, model_id, layout, "fl2va")
}

pub(crate) fn stage_managed_minimax_h3_import_source(
    data_dir: &Path,
    model_id: &str,
    layout: &str,
    partition: &str,
) -> ManagedMiniMaxH3Import {
    let source = data_dir.join("models/incoming").join(model_id);
    write_minimax_h3_dit_dir(&source);
    let mut files = vec![
        "diffusion_pytorch_model.safetensors".to_owned(),
        "config.json".to_owned(),
    ];
    if layout == "native" || layout == "convrot" {
        let mut config = h3_fixture_config();
        if layout == "convrot" {
            config["hidden_size"] = json!(256);
            config["attention_head_dim"] = json!(128);
            config["ffn_dim"] = json!(256);
            config["time_embed_dim"] = json!(64);
        }
        let shapes = sceneworks_core::minimax_h3_codec::H3Config(config.clone())
            .shapes()
            .unwrap();
        let mut header = Map::new();
        header.insert(
            "__metadata__".into(),
            json!({"h3_config":config.to_string(),"h3_qkv_layout":"interleaved"}),
        );
        let mut payload = Vec::new();
        for (canonical, mut shape) in shapes {
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
            let quantized = layout == "convrot"
                && key.starts_with("blocks.")
                && shape.len() == 2
                && key.ends_with(".weight")
                && (key.contains(".attn.")
                    || key.contains(".mlp.")
                    || key.contains(".adaln_proj.linear."));
            let offset = payload.len();
            let size = shape.iter().product::<usize>() * if quantized { 1 } else { 4 };
            payload.resize(offset + size, if quantized { 0 } else { 0x3a });
            if quantized {
                for row in 0..shape[0] {
                    payload[offset + row * shape[1]] = 58;
                }
            }
            header.insert(key.clone(), json!({"dtype":if quantized {"I8"} else {"F32"},"shape":shape,"data_offsets":[offset,payload.len()]}));
            if quantized {
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
    let install = ingest
        .finalize(
            if layout == "native" || layout == "convrot" {
                "renamed-user-checkpoint.safetensors"
            } else if layout == "sharded" {
                "renamed-part-0.safetensors"
            } else {
                "diffusion_pytorch_model.safetensors"
            },
            None,
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
        "importComponentTier": "q4",
        "downloads": component_downloads("q4", partition),
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
    }
}

pub(crate) const SHARED_REVISION: &str = "1111111111111111111111111111111111111111";
pub(crate) const TIER_REVISION: &str = "2222222222222222222222222222222222222222";

pub(crate) fn component_downloads(tier: &str, partition: &str) -> Vec<Value> {
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

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn write_companion(dir: &Path) {
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
pub(crate) fn install_fixture_snapshots(hub: &Path, tiers: &Path, base: &Path) {
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

/// Constructor documents at fixture scale; no model weights, downloads, or tensor runtime.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn write_shared_configs(root: &Path, encoder: &Path) {
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
    let weights = candle_gen::Weights::from_files(
        &[output.join("diffusion_pytorch_model-00001-of-00001.safetensors")],
        &device,
        DType::BF16,
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
