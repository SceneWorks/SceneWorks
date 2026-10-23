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
    ComponentRole, QuantFormat, MINIMAX_H3_TRANSFORMER_CLASS,
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
    write_safetensors(
        &dir.join("diffusion_pytorch_model.safetensors"),
        H3_DIT_KEYS,
        0x3a,
    );
    fs::write(
        dir.join("config.json"),
        json!({ "_class_name": MINIMAX_H3_TRANSFORMER_CLASS, "num_layers": 1 }).to_string(),
    )
    .unwrap();
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
