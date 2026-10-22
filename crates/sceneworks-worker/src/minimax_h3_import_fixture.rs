//! Synthetic dense MiniMax-H3 DiT import fixtures (sc-26680), shared by the model-import job tests
//! and the video arm tests so both drive the SAME artifact: the published DiT key grammar at
//! fixture scale, inside a diffusers transformer component directory.

use std::path::{Path, PathBuf};

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
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_plan(
        &install.compiled.plan,
    )
    .expect("complete assembly");
    let semantic_digest = install.compiled.record.plan.semantic_digest.clone();
    let mut entry = json!({
        "id": model_id,
        "name": "Imported H3 fine-tune",
        "type": "image",
        "family": "minimax-h3",
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
