//! Shared Iris task-export weight fixture for this crate's tests.
//!
//! `gen_core::iris` export resolution (`TaskExport::from_dir`, `RestorationResources::from_dir`)
//! reads the `model.safetensors` header for the trained-export `irisTask` stamp and refuses a
//! header it cannot parse, so a staged install's weights must be a real safetensors file. The stub
//! loaders the tests inject never read tensors, so a header-only file is enough.

// Not every target/feature combination compiles every caller.
#![allow(dead_code)]

use gen_core::iris::{IrisTask, ADAPTER_TASK_KEY};

/// A valid, tensor-free `model.safetensors` stamped `irisTask=<task>` in its `__metadata__` (the
/// stamp a trained export of that task carries; a matching stamp resolves, as an unstamped release
/// export does).
pub(crate) fn task_weights(task: IrisTask) -> Vec<u8> {
    let mut header = serde_json::json!({ "__metadata__": { ADAPTER_TASK_KEY: task.name() } })
        .to_string()
        .into_bytes();
    // The safetensors format pads the header with spaces to an 8-byte boundary.
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes
}

/// The released `speridlabs/iris-3b` backbone `config.yaml` (Apache-2.0), so tests that stage an
/// install run the engine's own plan resolution against the real architecture (sc-25689).
pub(crate) const RELEASE_CONFIG_YAML: &str = include_str!("../tests/fixtures/iris/config.yaml");

/// A valid safetensors file holding zero-filled tensors: `(name, dtype, shape)`, with `dtype` one
/// of the safetensors spellings `"F32"` / `"BF16"`.
pub(crate) fn zero_tensors(tensors: &[(&str, &str, &[usize])]) -> Vec<u8> {
    let (mut bytes, data) = tensor_file_head(tensors);
    bytes.resize(bytes.len() + data, 0);
    bytes
}

/// [`zero_tensors`] written as a SPARSE file: the header is real and the data region is a hole, so
/// a multi-GiB tensor costs no disk and no memory. For the tests that need a component whose
/// PRICED bytes are large enough to move a memory-strategy selection (the pricing reads headers).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn write_sparse_zero_tensors(
    path: &std::path::Path,
    tensors: &[(&str, &str, &[usize])],
) {
    let (head, data) = tensor_file_head(tensors);
    std::fs::write(path, &head).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_len((head.len() + data) as u64).unwrap();
}

/// The length prefix and header of a safetensors file holding `tensors`, and its data length.
fn tensor_file_head(tensors: &[(&str, &str, &[usize])]) -> (Vec<u8>, usize) {
    let mut header = serde_json::Map::new();
    let mut offset = 0_usize;
    for (name, dtype, shape) in tensors {
        let width = match *dtype {
            "F32" => 4,
            "BF16" => 2,
            other => panic!("unsupported fixture dtype {other}"),
        };
        let bytes = shape.iter().product::<usize>() * width;
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let mut header = serde_json::Value::Object(header).to_string().into_bytes();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    (bytes, offset)
}

/// A generation backbone `model.safetensors` both Iris engines can PRICE (sc-25690). Since the
/// memory-strategy contracts landed, admission reads the backbone's tensor headers before any
/// load, so a staged install needs a real header. One projection matrix and one norm gain: the two
/// width classes the loaders hold (the compute dtype, and f32 as stored).
pub(crate) fn priceable_backbone_weights() -> Vec<u8> {
    zero_tensors(&[
        ("blocks.0.attn.qkv.weight", "F32", &[4, 4]),
        ("blocks.0.norm.weight", "F32", &[4]),
    ])
}

/// The one language-tower tensor of [`priceable_text_encoder_files`], and the shard holding it.
pub(crate) const TEXT_ENCODER_LANGUAGE_TENSOR: &str = "model.language_model.embed_tokens.weight";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) const TEXT_ENCODER_LANGUAGE_SHARD: &str = "model-00001-of-00002.safetensors";

/// The files of a Qwen3-VL text-encoder snapshot both Iris engines can PRICE: a miniature tower
/// config (the shape of the engines' own tiny snapshot), the shard index, and two header-valid
/// shards — one language-tower tensor (resident) and one vision-tower tensor (never materialized).
///
/// The MLP width is 64, not the engines' miniature 48 (sc-26266): the MLX tower packs every
/// projection at ONE affine group size, so a q8 / q4 contract exists only for a tower whose
/// widths a group (32 or 64) divides. 48 has none, and the pre-load contract refuses the tier.
pub(crate) fn priceable_text_encoder_files() -> Vec<(&'static str, Vec<u8>)> {
    let config = serde_json::json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "dtype": "float32",
        "image_token_id": 151655,
        "model_type": "qwen3_vl",
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 64,
            "max_position_embeddings": 4096,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 4,
            "num_hidden_layers": 36,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [4, 2, 2],
                "rope_type": "default"
            },
            "rope_theta": 5000000.0,
            "tie_word_embeddings": true,
            "use_cache": true,
            "vocab_size": 25
        },
        "tie_word_embeddings": true,
        "transformers_version": "4.57.1",
        "video_token_id": 151656,
        "vision_config": {
            "deepstack_visual_indexes": [0],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "model_type": "qwen3_vl",
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652
    });
    let language = TEXT_ENCODER_LANGUAGE_TENSOR;
    let vision = "model.visual.patch_embed.proj.weight";
    let index = serde_json::json!({
        "metadata": { "format": "pt" },
        "weight_map": {
            language: "model-00001-of-00002.safetensors",
            vision: "model-00002-of-00002.safetensors",
        },
    });
    vec![
        ("config.json", config.to_string().into_bytes()),
        ("generation_config.json", b"{}".to_vec()),
        ("tokenizer.json", b"{}".to_vec()),
        ("tokenizer_config.json", b"{}".to_vec()),
        (
            "model.safetensors.index.json",
            index.to_string().into_bytes(),
        ),
        (
            TEXT_ENCODER_LANGUAGE_SHARD,
            zero_tensors(&[(language, "BF16", &[25, 32])]),
        ),
        (
            "model-00002-of-00002.safetensors",
            zero_tensors(&[(vision, "BF16", &[16, 4])]),
        ),
    ]
}
