use super::*;

fn assert_matches_request_validator(entry: &JsonObject, expected: bool) {
    assert_eq!(entry["supportsLastOnlyImageToVideo"], json!(expected));
    let payload = json!({"mode": "image_to_video", "lastFrameAssetId": "last-frame"});
    assert_eq!(
        crate::validate_video_reference_asset_ids_payload(
            payload.as_object().unwrap(),
            &Value::Object(entry.clone()),
        )
        .is_ok(),
        expected,
        "projected last-only fact must match actual request admission for {entry:?}"
    );
}

#[test]
fn imported_partitions_require_the_verified_plan_shape() {
    use sceneworks_core::minimax_h3_import::{
        MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION, MINIMAX_H3_IMPORT_FAMILY,
        MINIMAX_H3_IMPORT_SOURCE_SHAPE,
    };
    for (partition, base, expected) in [
        ("fl2va", "minimax_h3", true),
        ("hybrid", "minimax_h3", true),
        ("ref2va", "minimax_h3_ref", false),
    ] {
        let mut entry = json!({
            "id": "custom", "family": MINIMAX_H3_IMPORT_FAMILY,
            "importSourceShape": MINIMAX_H3_IMPORT_SOURCE_SHAPE,
            "baseModel": base, "minimaxH3Partition": partition,
            "importPlan": {"checkpointId": "managed/custom", "assembly": {
                "schemaVersion": MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION, "partition": partition
            }}
        })
        .as_object()
        .unwrap()
        .clone();
        apply_last_only_video_capability(&mut entry);
        assert_matches_request_validator(&entry, expected);
        entry["importPlan"]["assembly"]["partition"] = json!("forged-partition");
        apply_last_only_video_capability(&mut entry);
        assert_matches_request_validator(&entry, false);
        entry.remove("importPlan");
        apply_last_only_video_capability(&mut entry);
        assert_matches_request_validator(&entry, false);
    }
}

#[test]
fn stock_and_unknown_entries_match_the_validator() {
    for (id, expected) in [
        ("minimax_h3", true),
        ("minimax_h3_ref", false),
        ("unknown", false),
    ] {
        let mut entry = json!({"id": id}).as_object().unwrap().clone();
        apply_last_only_video_capability(&mut entry);
        assert_matches_request_validator(&entry, expected);
    }
}

/// SW-7032: the MiniMax-H3 turbo surface the web shows is the server's own H3 predicate, served.
#[test]
fn the_minimax_h3_turbo_surface_is_the_servers_predicate() {
    for (id, expected) in [
        ("minimax_h3", true),
        ("minimax_h3_ref", true),
        ("bernini", false),
        ("community_h3", false),
    ] {
        let mut entry = json!({"id": id, "family": "minimax-h3"})
            .as_object()
            .unwrap()
            .clone();
        apply_last_only_video_capability(&mut entry);
        assert_eq!(entry["supportsMinimaxH3Turbo"], json!(expected), "{id}");
        assert_eq!(
            expected,
            sceneworks_core::video_request::is_minimax_h3_model(id),
            "{id}"
        );
    }
}

/// SW-7032: the catalog's size enrichment (the last projection step, after the variant, convert-tier
/// and size fields) serves the core memory guidance for the entry as projected.
#[test]
fn size_enrichment_serves_the_core_memory_guidance() {
    let mut model = sceneworks_core::builtin_manifests::builtin_model_entry("krea_realtime_14b")
        .map(|entry| Value::Object(entry.clone()))
        .expect("shipped entry");
    let data_dir = tempfile::tempdir().unwrap();
    apply_variant_fields(model.as_object_mut().unwrap(), data_dir.path());
    let context = model_download_context(&model).unwrap();
    apply_model_catalog_size_fields(&mut model, context.as_ref(), Some(1))
        .expect("size enrichment");
    let served = model[sceneworks_core::memory_guidance::FIELD].clone();
    assert_eq!(
        served,
        sceneworks_core::memory_guidance::model_memory_guidance(&model)
    );
    assert_eq!(
        served["mlx"]["suggestion"][0]["tier"],
        json!("bf16"),
        "fidelity order is served"
    );
    assert!(served["mlx"]["tiers"]["q4"]["minHostGb"].is_number());
}
