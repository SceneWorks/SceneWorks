//! Synthetic dense MiniMax-H3 DiT import fixtures (sc-26680), shared by the model-import job tests
//! and the video arm tests so both drive the SAME artifact: the published DiT key grammar at
//! fixture scale, inside a diffusers transformer component directory.

use std::path::{Path, PathBuf};

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda as platform_runtime;
#[cfg(target_os = "macos")]
use runtime_macos as platform_runtime;
use serde_json::{json, Map, Value};

/// One block of the published DiT grammar: the refiner + joint-audio head that identify the family
/// by content, the per-block `adaln_proj.linear`, and the mixed-precision f32 heads.
const H3_DIT_KEYS: &[(&str, &str)] = &[
    ("transformer_blocks.0.adaln_proj.linear.weight", "BF16"),
    ("transformer_blocks.0.attn.to_qkv.weight", "BF16"),
    ("token_refiner.refiner_blocks.0.attn.to_qkv.weight", "BF16"),
    ("token_refiner.final_norm.weight", "BF16"),
    ("audio_proj_in.weight", "F32"),
    ("audio_proj_out.weight", "F32"),
    ("context_embedder.weight", "F32"),
    ("proj_in.weight", "BF16"),
    ("proj_out.weight", "BF16"),
];

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
    write_safetensors(
        &dir.join(sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_WEIGHTS_FILE),
        H3_DIT_KEYS,
        0x3a,
    );
    std::fs::write(
        dir.join(sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_CONFIG_FILE),
        json!({
            "_class_name": sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_CLASS,
            "num_layers": 1
        })
        .to_string(),
    )
    .expect("config writes");
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
    stage_managed_minimax_h3_import_with_partition(data_dir, model_id, "fl2va")
}

pub(crate) fn stage_managed_minimax_h3_import_with_partition(
    data_dir: &Path,
    model_id: &str,
    partition: &str,
) -> ManagedMiniMaxH3Import {
    let source = data_dir.join("models/incoming").join(model_id);
    write_minimax_h3_dit_dir(&source);
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
    for file in [
        sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_WEIGHTS_FILE,
        sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_CONFIG_FILE,
    ] {
        ingest
            .stage_copy_file(&source.join(file), file)
            .expect("stages");
    }
    let install = ingest
        .finalize(
            sceneworks_core::base_weights::MINIMAX_H3_TRANSFORMER_WEIGHTS_FILE,
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
