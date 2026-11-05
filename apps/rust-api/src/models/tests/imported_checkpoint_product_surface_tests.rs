use super::*;

/// A plan-backed imported MiniMax-H3 entry. `partition` / `base_model` are the DECLARATION the
/// contract validates — deliberately parameterized, because the whole point of the surface is
/// that it reports what was declared rather than what the shapes happen to look like.
fn imported_h3_entry(partition: &str, base_model: &str) -> JsonObject {
    json!({
        "id": "my_singularity_tune",
        "name": "My Singularity tune",
        "type": "video",
        "family": "minimax-h3",
        "catalogScope": "user",
        "importSourceShape": "transformer_directory",
        "baseModel": base_model,
        "minimaxH3Partition": partition,
        "importComponentTier": "q4",
        "importPlan": {
            "checkpointId": "managed/install-42",
            "planId": "checkpoint-plan-7",
            "assembly": {
                "schemaVersion": 1,
                "family": "minimax-h3",
                "partition": partition,
                "components": {
                    "transformer": {
                        "source": "plan",
                        "layerId": "artifact:diffusion_pytorch_model.safetensors",
                        "layerIds": ["artifact:diffusion_pytorch_model.safetensors"],
                        "directory": "",
                        "descriptorLayerIds": []
                    }
                }
            }
        },
        "capabilities": ["reference_to_video"],
        "macSupport": {"supported": true},
        "candleSupport": {"supported": true},
    })
    .as_object()
    .expect("fixture is an object")
    .clone()
}

/// A data dir that cannot exist, so every component-presence probe answers ABSENT without the
/// test reading or writing anything on this host.
/// Returned by reference: `Path` is unsized, so returning it by value is a compile error, and
/// `apply_imported_checkpoint_product_surface` takes `&FsPath` anyway.
fn empty_data_dir() -> &'static FsPath {
    std::path::Path::new("/nonexistent-sc26687-product-surface")
}

fn project(mut entry: JsonObject) -> JsonObject {
    apply_imported_checkpoint_product_surface(&mut entry, empty_data_dir(), None);
    entry
}

/// E6/E8/E9: readiness is meaningful only for a committed source AND its captured residents.
/// Keep the real transactional fixture and its private cache guard alive during projection.
struct CompiledImport {
    entry: JsonObject,
    data_dir: std::path::PathBuf,
    _import: sceneworks_worker::minimax_h3_import_fixture::ManagedMiniMaxH3Import,
    _temp: tempfile::TempDir,
}

fn real_imported_h3_entry(partition: &str) -> CompiledImport {
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("data");
    let imported =
        sceneworks_worker::minimax_h3_import_fixture::stage_managed_minimax_h3_import_source(
            &data_dir,
            "my_singularity_tune",
            "dense",
            partition,
        );
    crate::tests::support::assert_h3_component_authority_required(&data_dir, &imported.entry);
    let mut entry = imported.entry.clone();
    entry.insert("catalogScope".into(), json!("user"));
    entry.insert("macSupport".into(), json!({"supported":true}));
    entry.insert("candleSupport".into(), json!({"supported":true}));
    CompiledImport {
        entry,
        data_dir,
        _import: imported,
        _temp: temp,
    }
}

fn project_at(mut entry: JsonObject, data_dir: &FsPath) -> JsonObject {
    apply_imported_checkpoint_product_surface(&mut entry, data_dir, None);
    entry
}

fn operations(entry: &JsonObject) -> Vec<String> {
    entry["supportedOperations"]
        .as_array()
        .expect("supportedOperations is an array")
        .iter()
        .map(|value| value.as_str().expect("operation is a string").to_owned())
        .collect()
}

#[test]
fn ref2va_declares_only_the_reference_operation() {
    let entry = project(imported_h3_entry("ref2va", "minimax_h3_ref"));
    assert_eq!(operations(&entry), ["reference_to_video"]);
    assert_eq!(entry["importedSource"]["sourceVariant"], json!("ref2va"));
    assert_eq!(
        entry["importedSource"]["declaredPartition"],
        json!("ref2va")
    );
    assert_eq!(
        entry["importedSource"]["baseModel"],
        json!("minimax_h3_ref")
    );
    assert_eq!(entry["importedSource"]["ownershipMode"], json!("managed"));
    assert_eq!(entry["importedSource"]["componentTier"], json!("q4"));
}

#[test]
fn fl2va_declares_the_base_operations_and_not_the_reference_one() {
    let entry = project(imported_h3_entry("fl2va", "minimax_h3"));
    assert_eq!(
        operations(&entry),
        ["text_to_video", "image_to_video", "first_last_frame"]
    );
}

#[test]
fn hybrid_declares_every_operation_the_checkpoint_can_serve() {
    let entry = project(imported_h3_entry("hybrid", "minimax_h3"));
    assert_eq!(
        operations(&entry),
        [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "reference_to_video"
        ]
    );
}

#[test]
fn supported_backends_follow_the_lane_verdicts_the_scheduler_uses() {
    let mut entry = imported_h3_entry("ref2va", "minimax_h3_ref");
    entry.insert("macSupport".to_owned(), json!({"supported": false}));
    let entry = project(entry);
    assert_eq!(entry["supportedBackends"], json!(["candle-cuda"]));
}

#[test]
fn an_entry_no_lane_has_judged_omits_the_backend_verdict() {
    let mut entry = imported_h3_entry("ref2va", "minimax_h3_ref");
    entry.remove("macSupport");
    entry.remove("candleSupport");
    let entry = project(entry);
    assert!(
        entry.get("supportedBackends").is_none(),
        "nobody judged this entry, so the surface must stay silent rather than refuse it"
    );
}

#[test]
fn a_complete_assembly_is_ready_and_offers_no_repair() {
    let CompiledImport {
        entry,
        data_dir,
        _import,
        _temp,
    } = real_imported_h3_entry("fl2va");
    let resident_count = entry["downloads"].as_array().unwrap().len();
    let entry = project_at(entry, &data_dir);
    let components = entry["componentReadiness"]["components"]
        .as_array()
        .expect("components");
    assert_eq!(
        components.len(),
        1 + resident_count,
        "the imported DiT AND every frozen resident are bound: {components:?}"
    );
    assert!(
        components
            .iter()
            .all(|component| component["present"] == true),
        "{components:?}"
    );
    assert_eq!(components[0]["componentId"], json!("transformer"));
    assert_eq!(
        components[0]["present"],
        json!(true),
        "the plan-backed source reads present on its own resolution, nothing aggregates it"
    );
    assert_eq!(entry["componentReadiness"]["ready"], json!(true));
    assert_eq!(entry["repairActions"], json!([]));
}

#[test]
fn a_missing_resident_component_blocks_readiness_and_names_the_download() {
    let CompiledImport {
        entry,
        data_dir,
        _import,
        _temp,
    } = real_imported_h3_entry("fl2va");
    // Remove actual committed bytes, never edit downloads after commitment to bless a new row.
    let audio = sceneworks_worker::minimax_h3_import_fixture::shared_component_snapshot(&data_dir)
        .join("audio_vae/fixture.safetensors");
    assert!(audio.starts_with(&data_dir));
    std::fs::remove_file(&audio).unwrap();
    let entry = project_at(entry, &data_dir);
    assert_eq!(
        entry["componentReadiness"]["ready"],
        json!(false),
        "a half-provisioned assembly must never look ready"
    );
    let components = entry["componentReadiness"]["components"]
        .as_array()
        .expect("components");
    assert_eq!(
        components
            .iter()
            .find(|component| component["source"] == json!("plan"))
            .expect("imported DiT row")["present"],
        json!(true),
        "a missing resident component must not demote the imported source: {components:?}"
    );
    let repairs = entry["repairActions"].as_array().expect("repairActions");
    assert!(
        repairs.iter().any(|repair| {
            repair["id"] == json!("install-component:audio_vae")
                && repair["kind"] == json!("download")
                && repair["componentId"] == json!("audio_vae")
                && repair["variant"] == json!("q4")
                && repair["label"].as_str().is_some_and(|label| {
                    label.contains("audio_vae") && label.contains("MiniMaxAI/MiniMax-H3")
                })
        }),
        "{repairs:?}"
    );
}

#[test]
fn an_incompatible_base_declaration_offers_the_compatible_base_repair() {
    // The entry declares ref2va but names the FL2VA base: shapes cannot settle that, and no
    // download fixes it either.
    let entry = project(imported_h3_entry("ref2va", "minimax_h3"));
    assert_eq!(entry["componentReadiness"]["ready"], json!(false));
    assert_eq!(entry["importedSource"]["declaredPartition"], Value::Null);
    let repairs = entry["repairActions"].as_array().expect("repairActions");
    assert!(
        repairs
            .iter()
            .any(|repair| repair["id"] == json!("declare-compatible-base")),
        "{repairs:?}"
    );
}

#[test]
fn a_builtin_entry_carries_no_imported_product_surface() {
    let mut entry = imported_h3_entry("ref2va", "minimax_h3_ref");
    entry.insert("catalogScope".to_owned(), json!("builtin"));
    let entry = project(entry);
    for key in [
        "importedSource",
        "supportedOperations",
        "supportedBackends",
        "componentReadiness",
        "repairActions",
    ] {
        assert!(entry.get(key).is_none(), "builtin entry gained {key}");
    }
}
