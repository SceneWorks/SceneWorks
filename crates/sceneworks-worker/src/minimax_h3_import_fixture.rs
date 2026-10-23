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
    if layout == "native" {
        let config = h3_fixture_config();
        let shapes = sceneworks_core::minimax_h3_codec::H3Config(config.clone())
            .shapes()
            .unwrap();
        let mut header = Map::new();
        header.insert(
            "__metadata__".into(),
            json!({"h3_config":config.to_string(),"h3_qkv_layout":"interleaved"}),
        );
        let mut offset = 0u64;
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
            let size = shape.iter().product::<usize>() as u64 * 4;
            header.insert(
                key,
                json!({"dtype":"F32","shape":shape,"data_offsets":[offset,offset+size]}),
            );
            offset += size;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.resize(bytes.len() + offset as usize, 0x3a);
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
            if layout == "native" {
                "renamed-user-checkpoint.safetensors"
            } else if layout == "sharded" {
                "renamed-part-0.safetensors"
            } else {
                "diffusion_pytorch_model.safetensors"
            },
            None,
        )
        .expect("a dense H3 DiT commits");
    let semantic_digest = install.compiled.record.plan.semantic_digest.clone();
    let mut entry = json!({
        "id": model_id,
        "name": "Imported H3 fine-tune",
        "type": "image",
        "family": "minimax-h3",
        "baseModel": if partition == "ref2va" { "minimax_h3_ref" } else { "minimax_h3" },
        "minimaxH3Partition": partition,
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
        json!({"weight_map":{}}),
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
