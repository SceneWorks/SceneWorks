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
