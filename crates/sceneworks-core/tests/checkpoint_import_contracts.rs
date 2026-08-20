use std::{fs, path::PathBuf};

use sceneworks_core::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointInventoryV1, ImportLayerV1, ImportPlanReferenceV1,
    ImportPlanSummaryV1, ImportPlanV1, ManagedProvenanceV1, SourceLocatorV1,
    CHECKPOINT_IMPORT_CONTRACT_VERSION,
};
use serde_json::{json, Value};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
type JsonParser = fn(Value) -> Result<(), serde_json::Error>;

fn linked(root_id: &str, path: &str) -> SourceLocatorV1 {
    SourceLocatorV1::linked(root_id, path, DIGEST).expect("valid linked locator")
}

fn managed(install_id: &str, path: &str) -> SourceLocatorV1 {
    SourceLocatorV1::managed(
        install_id,
        path,
        DIGEST,
        ManagedProvenanceV1 {
            source: "huggingface".to_owned(),
            reference: Some("org/model@rev".to_owned()),
        },
    )
    .expect("valid managed locator")
}

fn plan(source: SourceLocatorV1) -> ImportPlanV1 {
    ImportPlanV1::new(
        "plan-flux-dev",
        "flux",
        vec![ImportLayerV1 {
            layer_id: "transformer".to_owned(),
            role: "transformer".to_owned(),
            target_path: "transformer/model.safetensors".to_owned(),
            source,
        }],
    )
    .expect("valid import plan")
}

#[test]
fn linked_and_managed_copies_have_the_same_semantic_plan_but_different_bindings() {
    let linked_plan = plan(linked("external-models", "flux/model.safetensors"));
    let managed_plan = plan(managed("installed-flux", "weights/model.safetensors"));

    assert_eq!(
        linked_plan.layers[0].source.semantic_identity().unwrap(),
        managed_plan.layers[0].source.semantic_identity().unwrap(),
        "equal source bytes must be locator-independent"
    );
    assert_eq!(
        linked_plan.semantic_digest().unwrap(),
        managed_plan.semantic_digest().unwrap()
    );
    assert_ne!(
        linked_plan.source_binding_identity().unwrap(),
        managed_plan.source_binding_identity().unwrap()
    );

    let linked_reference = linked_plan.plan_reference().unwrap();
    let managed_reference = managed_plan.plan_reference().unwrap();
    assert_eq!(
        linked_reference.semantic_digest,
        managed_reference.semantic_digest
    );
    assert_ne!(
        linked_reference.source_binding_identity,
        managed_reference.source_binding_identity
    );
    assert_eq!(
        linked_plan.semantic_digest().unwrap(),
        "sha256:d224a8484eaa43937759d190ff91a6de912585505b459602a61e2150d914bef5"
    );
    assert_eq!(
        linked_plan.source_binding_identity().unwrap(),
        "sha256:402ad8fc8e320fc95e242e07ca1e86194489ef793bc147ea5d331a3c9cf3afd7"
    );
    assert_eq!(
        managed_plan.source_binding_identity().unwrap(),
        "sha256:dcb8d85fd837eac9b4e776b4e6a992b6c8a60cb807d40f9bfb20caff58649122"
    );
}

#[test]
fn canonical_serialization_and_domain_separated_hashes_are_deterministic() {
    let first = ImportPlanV1::new(
        "plan-flux-dev",
        "flux",
        vec![
            ImportLayerV1 {
                layer_id: "vae".to_owned(),
                role: "vae".to_owned(),
                target_path: "vae/model.safetensors".to_owned(),
                source: linked("root", "vae/model.safetensors"),
            },
            ImportLayerV1 {
                layer_id: "transformer".to_owned(),
                role: "transformer".to_owned(),
                target_path: "transformer/model.safetensors".to_owned(),
                source: linked("root", "transformer/model.safetensors"),
            },
        ],
    )
    .expect("valid plan");
    let second = ImportPlanV1::new(
        "plan-flux-dev",
        "flux",
        vec![
            ImportLayerV1 {
                layer_id: "transformer".to_owned(),
                role: "transformer".to_owned(),
                target_path: "transformer/model.safetensors".to_owned(),
                source: linked("root", "transformer/model.safetensors"),
            },
            ImportLayerV1 {
                layer_id: "vae".to_owned(),
                role: "vae".to_owned(),
                target_path: "vae/model.safetensors".to_owned(),
                source: linked("root", "vae/model.safetensors"),
            },
        ],
    )
    .expect("valid plan");

    assert_eq!(
        first.canonical_json().unwrap(),
        second.canonical_json().unwrap()
    );
    assert_eq!(
        first.semantic_digest().unwrap(),
        second.semantic_digest().unwrap()
    );
    assert_eq!(
        first.source_binding_identity().unwrap(),
        second.source_binding_identity().unwrap()
    );
    assert_ne!(
        first.semantic_digest().unwrap(),
        first.source_binding_identity().unwrap()
    );
}

#[test]
fn catalog_records_hold_only_a_reference_and_compact_summary() {
    let import_plan = plan(linked("external", "flux/model.safetensors"));
    let record = CheckpointCatalogRecordV1::from_plan("checkpoint-flux-dev", &import_plan).unwrap();
    let inventory = CheckpointInventoryV1::new(vec![record.clone()]).expect("valid inventory");
    let encoded = serde_json::to_value(&inventory).expect("serializes");

    assert_eq!(encoded["records"][0]["schemaVersion"], 1);
    assert_eq!(encoded["records"][0]["plan"]["planId"], "plan-flux-dev");
    assert_eq!(encoded["records"][0]["summary"]["layerCount"], 1);
    assert!(encoded["records"][0].get("layers").is_none());
    assert!(encoded["records"][0]["summary"].get("layers").is_none());
    assert_eq!(
        CheckpointInventoryV1::new(vec![record])
            .unwrap()
            .canonical_json()
            .unwrap(),
        inventory.canonical_json().unwrap()
    );
}

#[test]
fn serde_rejects_unknown_versions_invalid_paths_hashes_and_unknown_fields() {
    let bad_version = json!({"kind":"linked","schemaVersion":2,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST});
    let error = serde_json::from_value::<SourceLocatorV1>(bad_version)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recompile/rescan required"), "{error}");

    for relative_path in [
        "/absolute",
        "../escape",
        "safe/../escape",
        "safe//model",
        "safe\\model",
        "C:/escape",
        ".",
    ] {
        let value = json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":relative_path,"fingerprint":DIGEST});
        assert!(
            serde_json::from_value::<SourceLocatorV1>(value).is_err(),
            "{relative_path}"
        );
    }
    let invalid_hash = json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST.to_uppercase()});
    assert!(serde_json::from_value::<SourceLocatorV1>(invalid_hash).is_err());
    let unknown_field = json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST,"extra":true});
    assert!(serde_json::from_value::<SourceLocatorV1>(unknown_field).is_err());
}

#[test]
fn every_versioned_contract_fails_closed_during_deserialization() {
    let import_plan = plan(linked("root", "model.safetensors"));
    let record = CheckpointCatalogRecordV1::from_plan("checkpoint", &import_plan).unwrap();
    let inventory = CheckpointInventoryV1::new(vec![record.clone()]).unwrap();
    let values: Vec<Value> = vec![
        serde_json::to_value(linked("root", "model.safetensors")).unwrap(),
        serde_json::to_value(&import_plan).unwrap(),
        serde_json::to_value(import_plan.plan_reference().unwrap()).unwrap(),
        serde_json::to_value(import_plan.summary().unwrap()).unwrap(),
        serde_json::to_value(&record).unwrap(),
        serde_json::to_value(&inventory).unwrap(),
    ];
    let parsers: [fn(Value) -> Result<(), serde_json::Error>; 6] = [
        |value| serde_json::from_value::<SourceLocatorV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanReferenceV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanSummaryV1>(value).map(|_| ()),
        |value| serde_json::from_value::<CheckpointCatalogRecordV1>(value).map(|_| ()),
        |value| serde_json::from_value::<CheckpointInventoryV1>(value).map(|_| ()),
    ];
    for (mut value, parse) in values.into_iter().zip(parsers) {
        value["schemaVersion"] = json!(CHECKPOINT_IMPORT_CONTRACT_VERSION + 1);
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("recompile/rescan required"), "{error}");
    }
}

#[test]
fn future_version_envelopes_win_over_future_shape_errors() {
    let cases: Vec<(Value, JsonParser)> = vec![
        (
            json!({"kind":"future_locator","schemaVersion":2,"newLocatorField":true}),
            |value| serde_json::from_value::<SourceLocatorV1>(value).map(|_| ()),
        ),
        (json!({"schemaVersion":2,"futurePlanField":true}), |value| {
            serde_json::from_value::<ImportPlanV1>(value).map(|_| ())
        }),
        (
            json!({"schemaVersion":2,"futureReferenceField":true}),
            |value| serde_json::from_value::<ImportPlanReferenceV1>(value).map(|_| ()),
        ),
        (
            json!({"schemaVersion":2,"futureSummaryField":true}),
            |value| serde_json::from_value::<ImportPlanSummaryV1>(value).map(|_| ()),
        ),
        (
            json!({"schemaVersion":2,"futureRecordField":true}),
            |value| serde_json::from_value::<CheckpointCatalogRecordV1>(value).map(|_| ()),
        ),
        (
            json!({"schemaVersion":2,"futureInventoryField":true}),
            |value| serde_json::from_value::<CheckpointInventoryV1>(value).map(|_| ()),
        ),
    ];
    for (value, parse) in cases {
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("recompile/rescan required"), "{error}");
    }

    let future_nested_locator = json!({
        "schemaVersion": 1, "planId": "plan", "family": "family",
        "layers": [{
            "layerId": "layer", "role": "role", "targetPath": "model.safetensors",
            "source": {"kind": "future_locator", "schemaVersion": 2, "futureLocatorField": true}
        }]
    });
    let error = serde_json::from_value::<ImportPlanV1>(future_nested_locator)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recompile/rescan required"), "{error}");

    let future_nested_plan = json!({
        "schemaVersion":1,"checkpointId":"checkpoint",
        "plan":{"schemaVersion":2,"futureReferenceField":true},
        "summary":{"schemaVersion":2,"futureSummaryField":true}
    });
    let error = serde_json::from_value::<CheckpointCatalogRecordV1>(future_nested_plan)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recompile/rescan required"), "{error}");
}

#[test]
fn raw_json_rejects_duplicate_keys_for_every_versioned_contract() {
    macro_rules! assert_duplicate_schema_version {
        ($contract:ty, $document:expr, $key:literal) => {{
            let document = $document;
            let error = serde_json::from_str::<$contract>(&document)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("duplicate object key `{}`", $key)),
                "{error}"
            );
        }};
    }
    macro_rules! assert_duplicate_body_field {
        ($contract:ty, $document:expr) => {{
            let document = $document;
            let error = serde_json::from_str::<$contract>(&document)
                .unwrap_err()
                .to_string();
            assert!(error.contains("duplicate"), "{error}");
        }};
    }

    // A future version followed by v1 must never be collapsed to v1 by a
    // map-based intermediate decoder.
    assert_duplicate_schema_version!(
        SourceLocatorV1,
        format!(
            r#"{{"kind":"linked","schemaVersion":2,"schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
        ),
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        ImportPlanV1,
        format!(
            r#"{{"schemaVersion":2,"schemaVersion":1,"planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{{"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}}}]}}"#
        ),
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        ImportPlanReferenceV1,
        format!(
            r#"{{"schemaVersion":2,"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}}"#
        ),
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        ImportPlanSummaryV1,
        format!(
            r#"{{"schemaVersion":2,"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}"#
        ),
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        CheckpointInventoryV1,
        r#"{"schemaVersion":2,"schemaVersion":1,"records":[]}"#,
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        CheckpointCatalogRecordV1,
        r#"{"schemaVersion":2,"schemaVersion":1,"checkpointId":"checkpoint"}"#,
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        CheckpointCatalogRecordV1,
        format!(
            r#"{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{{"schemaVersion":2,"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}"#
        ),
        "schemaVersion"
    );
    assert_duplicate_schema_version!(
        CheckpointInventoryV1,
        format!(
            r#"{{"schemaVersion":1,"records":[{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{{"schemaVersion":2,"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}]}}"#
        ),
        "schemaVersion"
    );

    // Reject duplicate fields in the v1 body too, including nested versioned
    // values reached through a plan and through an otherwise unversioned record.
    assert_duplicate_body_field!(
        SourceLocatorV1,
        format!(
            r#"{{"kind":"linked","schemaVersion":1,"rootId":"first","rootId":"second","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
        )
    );
    assert_duplicate_body_field!(
        ImportPlanV1,
        format!(
            r#"{{"schemaVersion":1,"planId":"first","planId":"second","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{{"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}}}]}}"#
        )
    );
    assert_duplicate_body_field!(
        ImportPlanReferenceV1,
        format!(
            r#"{{"schemaVersion":1,"planId":"first","planId":"second","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}}"#
        )
    );
    assert_duplicate_body_field!(
        ImportPlanSummaryV1,
        format!(
            r#"{{"schemaVersion":1,"family":"first","family":"second","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}"#
        )
    );
    assert_duplicate_body_field!(
        CheckpointInventoryV1,
        r#"{"schemaVersion":1,"records":[],"records":[]}"#
    );

    assert_duplicate_body_field!(
        CheckpointCatalogRecordV1,
        format!(
            r#"{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{{"schemaVersion":1,"planId":"first","planId":"second","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}"#
        )
    );
}

#[test]
fn future_versions_precede_duplicate_body_diagnostics() {
    macro_rules! assert_recompile_rescan_required {
        ($contract:ty, $document:expr) => {{
            let document = $document;
            let error = serde_json::from_str::<$contract>(&document)
                .unwrap_err()
                .to_string();
            assert!(error.contains("recompile/rescan required"), "{error}");
        }};
    }

    assert_recompile_rescan_required!(
        SourceLocatorV1,
        format!(r#"{{"kind":"future","schemaVersion":2,"rootId":"first","rootId":"second"}}"#)
    );
    assert_recompile_rescan_required!(
        ImportPlanV1,
        r#"{"schemaVersion":2,"planId":"first","planId":"second","futurePlanField":true}"#
    );
    assert_recompile_rescan_required!(
        ImportPlanReferenceV1,
        r#"{"schemaVersion":2,"planId":"first","planId":"second","futureReferenceField":true}"#
    );
    assert_recompile_rescan_required!(
        ImportPlanSummaryV1,
        r#"{"schemaVersion":2,"family":"first","family":"second","futureSummaryField":true}"#
    );
    assert_recompile_rescan_required!(
        CheckpointInventoryV1,
        r#"{"schemaVersion":2,"records":[],"records":[],"futureInventoryField":true}"#
    );

    // The v1 outer document must stream each nested value losslessly, so its
    // nested future version gets the same precedence over its duplicate body.
    assert_recompile_rescan_required!(
        ImportPlanV1,
        r#"{"schemaVersion":1,"planId":"plan","family":"family","layers":[{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{"kind":"future","schemaVersion":2,"rootId":"first","rootId":"second"}}]}"#
    );
    assert_recompile_rescan_required!(
        CheckpointCatalogRecordV1,
        r#"{"schemaVersion":1,"checkpointId":"checkpoint","plan":{"schemaVersion":2,"planId":"first","planId":"second","futureReferenceField":true},"summary":{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}"#
    );
    assert_recompile_rescan_required!(
        CheckpointInventoryV1,
        r#"{"schemaVersion":1,"records":[{"schemaVersion":1,"checkpointId":"checkpoint","plan":{"schemaVersion":2,"planId":"first","planId":"second","futureReferenceField":true},"summary":{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}]}"#
    );
}

#[test]
fn recursive_preflight_is_order_independent_across_nested_surfaces() {
    macro_rules! assert_recompile {
        ($contract:ty, $document:expr) => {{
            let error = serde_json::from_str::<$contract>(&$document)
                .unwrap_err()
                .to_string();
            assert!(error.contains("schema version 2"), "{error}");
            assert!(error.contains("recompile/rescan required"), "{error}");
            assert!(!error.contains("`planId`"), "{error}");
            assert!(!error.contains("`checkpointId`"), "{error}");
            assert!(!error.contains("`records`"), "{error}");
        }};
    }

    let plan_permutations = [
        r#"{"schemaVersion":1,"planId":"first","planId":"second","family":"family","layers":[{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{"kind":"future","schemaVersion":2}}]}"#.to_owned(),
        r#"{"layers":[{"source":{"schemaVersion":2,"kind":"future"},"targetPath":"model.safetensors","role":"role","layerId":"layer"}],"family":"family","planId":"first","planId":"second","schemaVersion":1}"#.to_owned(),
    ];
    for document in plan_permutations {
        assert_recompile!(ImportPlanV1, document);
    }

    let duplicate_body_layer = format!(
        r#"{{"layerId":"first","layerId":"second","role":"role","targetPath":"model.safetensors","source":{{"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}}}"#
    );
    let future_locator_layer = r#"{"layerId":"future","role":"role","targetPath":"future.safetensors","source":{"kind":"future","schemaVersion":2}}"#;
    for document in [
        format!(
            r#"{{"schemaVersion":1,"planId":"plan","family":"family","layers":[{duplicate_body_layer},{future_locator_layer}]}}"#
        ),
        format!(
            r#"{{"schemaVersion":1,"planId":"plan","family":"family","layers":[{future_locator_layer},{duplicate_body_layer}]}}"#
        ),
    ] {
        assert_recompile!(ImportPlanV1, document);
    }

    let valid_reference = format!(
        r#"{{"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}}"#
    );
    let catalog_permutations = [
        format!(
            r#"{{"schemaVersion":1,"checkpointId":"first","checkpointId":"second","plan":{valid_reference},"summary":{{"schemaVersion":2,"futureSummaryField":true}}}}"#
        ),
        format!(
            r#"{{"summary":{{"futureSummaryField":true,"schemaVersion":2}},"plan":{valid_reference},"checkpointId":"first","checkpointId":"second","schemaVersion":1}}"#
        ),
    ];
    for document in catalog_permutations {
        assert_recompile!(CheckpointCatalogRecordV1, document);
    }

    let valid_record = format!(
        r#"{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{valid_reference},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}"#
    );
    let inventory_permutations = [
        format!(
            r#"{{"schemaVersion":1,"records":[{valid_record}],"records":[{{"schemaVersion":2,"futureRecordField":true}}]}}"#
        ),
        format!(
            r#"{{"records":[{{"futureRecordField":true,"schemaVersion":2}}],"records":[{valid_record}],"schemaVersion":1}}"#
        ),
    ];
    for document in inventory_permutations {
        assert_recompile!(CheckpointInventoryV1, document);
    }

    let duplicate_body_record = format!(
        r#"{{"schemaVersion":1,"checkpointId":"first","checkpointId":"second","plan":{valid_reference},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}"#
    );
    let future_record = r#"{"schemaVersion":2,"futureRecordField":true}"#;
    for document in [
        format!(r#"{{"schemaVersion":1,"records":[{duplicate_body_record},{future_record}]}}"#),
        format!(r#"{{"schemaVersion":1,"records":[{future_record},{duplicate_body_record}]}}"#),
    ] {
        assert_recompile!(CheckpointInventoryV1, document);
    }
}

#[test]
fn nested_duplicate_schema_versions_precede_earlier_body_duplicates() {
    macro_rules! assert_duplicate_version {
        ($contract:ty, $document:expr) => {{
            let error = serde_json::from_str::<$contract>(&$document)
                .unwrap_err()
                .to_string();
            assert!(
                error.starts_with(
                    "checkpoint-import JSON contains duplicate object key `schemaVersion`"
                ),
                "{error}"
            );
            assert!(!error.contains("`planId`"), "{error}");
            assert!(!error.contains("`checkpointId`"), "{error}");
        }};
    }

    assert_duplicate_version!(
        ImportPlanV1,
        format!(
            r#"{{"schemaVersion":1,"planId":"first","planId":"second","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{{"kind":"linked","schemaVersion":"invalid","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}}}]}}"#
        )
    );
    assert_duplicate_version!(
        CheckpointCatalogRecordV1,
        format!(
            r#"{{"schemaVersion":1,"checkpointId":"first","checkpointId":"second","plan":{{"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}},"summary":{{"schemaVersion":1,"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}"#
        )
    );
    assert_duplicate_version!(
        CheckpointInventoryV1,
        format!(
            r#"{{"schemaVersion":1,"records":[{{"schemaVersion":1,"checkpointId":"first","checkpointId":"second","plan":{{"schemaVersion":2,"schemaVersion":1,"planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}"}},"summary":{{"schemaVersion":1,"family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}"}}}}]}}"#
        )
    );
}

#[test]
fn future_outer_envelopes_precede_all_nested_and_body_duplicates() {
    macro_rules! assert_outer_future {
        ($contract:ty, $document:expr) => {{
            let error = serde_json::from_str::<$contract>(&$document)
                .unwrap_err()
                .to_string();
            assert!(error.contains("schema version 2"), "{error}");
            assert!(error.contains("recompile/rescan required"), "{error}");
            assert!(!error.contains("duplicate object key"), "{error}");
        }};
    }

    assert_outer_future!(
        SourceLocatorV1,
        r#"{"kind":"future","schemaVersion":2,"rootId":"a","rootId":"b"}"#
    );
    assert_outer_future!(
        ImportPlanReferenceV1,
        r#"{"schemaVersion":2,"planId":"a","planId":"b"}"#
    );
    assert_outer_future!(
        ImportPlanSummaryV1,
        r#"{"schemaVersion":2,"family":"a","family":"b"}"#
    );
    assert_outer_future!(
        ImportPlanV1,
        r#"{"schemaVersion":2,"planId":"a","planId":"b","layers":[{"source":{"schemaVersion":1,"schemaVersion":2}}]}"#
    );
    assert_outer_future!(
        CheckpointCatalogRecordV1,
        r#"{"schemaVersion":2,"checkpointId":"a","checkpointId":"b","plan":{"schemaVersion":1,"schemaVersion":2}}"#
    );
    assert_outer_future!(
        CheckpointInventoryV1,
        r#"{"schemaVersion":2,"records":[],"records":[{"schemaVersion":1,"schemaVersion":2}]}"#
    );
}

#[test]
fn invalid_manual_plans_cannot_emit_identities_or_catalog_records() {
    let valid_plan = plan(linked("root", "model.safetensors"));
    let valid_semantic = valid_plan.semantic_digest().unwrap();
    let valid_binding = valid_plan.source_binding_identity().unwrap();

    let mut future_plan = valid_plan.clone();
    future_plan.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    for error in [
        future_plan.semantic_digest().unwrap_err(),
        future_plan.source_binding_identity().unwrap_err(),
        future_plan.plan_reference().unwrap_err(),
        future_plan.summary().unwrap_err(),
        CheckpointCatalogRecordV1::from_plan("future", &future_plan).unwrap_err(),
    ] {
        let error = error.to_string();
        assert!(error.contains("recompile/rescan required"), "{error}");
    }
    assert!(!valid_semantic.is_empty());
    assert!(!valid_binding.is_empty());

    let mut invalid_locator_plan = valid_plan;
    match &mut invalid_locator_plan.layers[0].source {
        SourceLocatorV1::Linked { schema_version, .. }
        | SourceLocatorV1::Managed { schema_version, .. } => {
            *schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
        }
    }
    assert!(invalid_locator_plan.semantic_digest().is_err());
    assert!(invalid_locator_plan.source_binding_identity().is_err());
    let error = CheckpointCatalogRecordV1::from_plan("invalid-locator", &invalid_locator_plan)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recompile/rescan required"), "{error}");
}

#[test]
fn ordinary_serde_publication_is_validation_gated_for_every_versioned_surface() {
    fn assert_recompile<T: serde::Serialize>(value: &T) {
        let error = serde_json::to_string(value).unwrap_err().to_string();
        assert!(error.contains("recompile/rescan required"), "{error}");
    }

    let valid_plan = plan(linked("root", "model.safetensors"));

    let mut future_locator = linked("root", "model.safetensors");
    match &mut future_locator {
        SourceLocatorV1::Linked { schema_version, .. }
        | SourceLocatorV1::Managed { schema_version, .. } => {
            *schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
        }
    }
    assert_recompile(&future_locator);

    let mut future_plan = valid_plan.clone();
    future_plan.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    assert_recompile(&future_plan);

    let mut future_reference = valid_plan.plan_reference().unwrap();
    future_reference.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    assert_recompile(&future_reference);

    let mut future_summary = valid_plan.summary().unwrap();
    future_summary.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    assert_recompile(&future_summary);

    let valid_record = CheckpointCatalogRecordV1::from_plan("checkpoint", &valid_plan).unwrap();
    let mut future_record = valid_record.clone();
    future_record.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    assert_recompile(&future_record);

    let mut future_inventory = CheckpointInventoryV1::new(vec![valid_record]).unwrap();
    future_inventory.schema_version = CHECKPOINT_IMPORT_CONTRACT_VERSION + 1;
    assert_recompile(&future_inventory);

    let mut nested_future_locator = valid_plan;
    nested_future_locator.layers[0].source = future_locator;
    assert_recompile(&nested_future_locator);
}

#[test]
fn catalog_record_recomputes_every_loaded_plan_claim() {
    let import_plan = plan(linked("root", "model.safetensors"));
    let record = CheckpointCatalogRecordV1::from_plan("checkpoint", &import_plan).unwrap();
    record.validate_loaded_plan(&import_plan).unwrap();

    let mut wrong_id = record.clone();
    wrong_id.plan.plan_id = "other-plan".to_owned();
    assert!(wrong_id.validate_loaded_plan(&import_plan).is_err());

    let forged_identity = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    let mut wrong_semantic = record.clone();
    wrong_semantic.plan.semantic_digest = forged_identity.to_owned();
    wrong_semantic.summary.semantic_digest = forged_identity.to_owned();
    assert!(wrong_semantic.validate_loaded_plan(&import_plan).is_err());

    let mut wrong_binding = record.clone();
    wrong_binding.plan.source_binding_identity = forged_identity.to_owned();
    assert!(wrong_binding.validate_loaded_plan(&import_plan).is_err());

    let mut wrong_family = record.clone();
    wrong_family.summary.family = "other-family".to_owned();
    assert!(wrong_family.validate_loaded_plan(&import_plan).is_err());

    let mut wrong_count = record.clone();
    wrong_count.summary.layer_count = 2;
    wrong_count.summary.layer_roles.push("vae".to_owned());
    assert!(wrong_count.validate_loaded_plan(&import_plan).is_err());

    let mut wrong_roles = record;
    wrong_roles.summary.layer_roles = vec!["other-role".to_owned()];
    assert!(wrong_roles.validate_loaded_plan(&import_plan).is_err());
}

#[test]
fn inventory_rejects_ambiguous_duplicate_plan_ids() {
    let import_plan = plan(linked("root", "model.safetensors"));
    let first = CheckpointCatalogRecordV1::from_plan("checkpoint-a", &import_plan).unwrap();
    let second = CheckpointCatalogRecordV1::from_plan("checkpoint-b", &import_plan).unwrap();
    let error = CheckpointInventoryV1::new(vec![first, second])
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate import plan ids"), "{error}");
}

#[test]
fn checked_layer_count_prevents_v1_truncation() {
    assert_eq!(ImportPlanV1::checked_layer_count(1).unwrap(), 1);
    if usize::BITS > 32 {
        let error = ImportPlanV1::checked_layer_count((u32::MAX as usize) + 1)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("more layers than v1 can represent"),
            "{error}"
        );
    }
}

#[test]
fn deserialization_rejects_noncanonical_plan_and_mismatched_catalog_digest() {
    let import_plan = plan(linked("root", "model.safetensors"));
    let mut plan_value = serde_json::to_value(&import_plan).unwrap();
    plan_value["layers"] = json!([
        {"layerId":"z","role":"vae","targetPath":"vae/model.safetensors","source":serde_json::to_value(linked("root", "vae/model.safetensors")).unwrap()},
        {"layerId":"a","role":"transformer","targetPath":"transformer/model.safetensors","source":serde_json::to_value(linked("root", "transformer/model.safetensors")).unwrap()}
    ]);
    assert!(serde_json::from_value::<ImportPlanV1>(plan_value).is_err());

    let mut record_value = serde_json::to_value(
        CheckpointCatalogRecordV1::from_plan("checkpoint", &import_plan).unwrap(),
    )
    .unwrap();
    record_value["summary"]["semanticDigest"] =
        json!("sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    assert!(serde_json::from_value::<CheckpointCatalogRecordV1>(record_value).is_err());
}

#[test]
fn published_schema_covers_every_contract_and_uses_strict_versioned_variants() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(path).expect("read schema"))
        .expect("schema JSON parses");
    let defs = schema["$defs"].as_object().expect("schema definitions");
    for name in [
        "nonBlankText",
        "sourceLocator",
        "importPlan",
        "planReference",
        "planSummary",
        "catalogRecord",
        "checkpointInventory",
    ] {
        assert!(defs.contains_key(name), "missing {name} schema");
    }
    let variants = defs["sourceLocator"]["oneOf"]
        .as_array()
        .expect("source variants");
    assert_eq!(variants.len(), 2);
    for variant in variants {
        assert_eq!(variant["additionalProperties"], false);
        assert_eq!(variant["properties"]["schemaVersion"]["const"], 1);
    }
    assert_eq!(defs["sha256"]["pattern"], "^[0-9a-f]{64}$");
    assert_eq!(defs["nonBlankText"]["pattern"], "(?=.*\\S)");
    assert_eq!(defs["relativePath"]["pattern"], "^(?=.*\\S)(?!/)(?!.*\\\\)(?!.*:)(?!.*//)(?!.*(?:^|/)\\.{1,2}(?:/|$))(?!.*(?:^|/)\\.{1,2}$)(?!.*\\/$)[^\\x00-\\x1F\\x7F]+$");
    assert_eq!(
        defs["planSummary"]["properties"]["layerCount"]["maximum"],
        4_294_967_295_u64
    );
    assert_eq!(
        defs["importPlan"]["properties"]["layers"]["uniqueItems"],
        true
    );
    assert_eq!(
        defs["importPlan"]["properties"]["layers"]["maxItems"],
        4_294_967_295_u64
    );
    assert_eq!(
        defs["checkpointInventory"]["properties"]["records"]["uniqueItems"],
        true
    );
    assert_eq!(
        defs["catalogRecord"]["properties"]["schemaVersion"]["const"],
        1
    );
    assert!(defs["catalogRecord"]["required"]
        .as_array()
        .unwrap()
        .contains(&json!("schemaVersion")));
}

#[test]
fn published_schema_and_serde_reject_the_same_fail_closed_edge_cases() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(path).expect("read schema"))
        .expect("schema JSON parses");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");

    let valid_locator = serde_json::to_value(linked("root", "models/model.safetensors")).unwrap();
    assert!(validator.is_valid(&valid_locator), "valid locator");
    assert!(serde_json::from_value::<SourceLocatorV1>(valid_locator).is_ok());

    let explicit_null_reference = json!({
        "kind": "managed", "schemaVersion": 1, "installId": "install",
        "relativePath": "models/model.safetensors", "sha256": DIGEST,
        "provenance": {"source": "huggingface", "reference": null}
    });
    assert!(validator.is_valid(&explicit_null_reference));
    assert!(serde_json::from_value::<SourceLocatorV1>(explicit_null_reference).is_ok());

    let linked_with_managed_ownership = json!({
        "kind": "linked", "schemaVersion": 1, "rootId": "root", "installId": "install",
        "relativePath": "models/model.safetensors", "fingerprint": DIGEST
    });
    assert!(!validator.is_valid(&linked_with_managed_ownership));
    assert!(serde_json::from_value::<SourceLocatorV1>(linked_with_managed_ownership).is_err());

    let managed_with_linked_ownership = json!({
        "kind": "managed", "schemaVersion": 1, "installId": "install", "rootId": "root",
        "relativePath": "models/model.safetensors", "sha256": DIGEST,
        "provenance": {"source": "huggingface"}
    });
    assert!(!validator.is_valid(&managed_with_linked_ownership));
    assert!(serde_json::from_value::<SourceLocatorV1>(managed_with_linked_ownership).is_err());

    let trailing_separator = json!({
        "kind": "linked", "schemaVersion": 1, "rootId": "root",
        "relativePath": "models/", "fingerprint": DIGEST
    });
    assert!(!validator.is_valid(&trailing_separator));
    assert!(serde_json::from_value::<SourceLocatorV1>(trailing_separator).is_err());

    let whitespace_identifier = json!({
        "kind": "linked", "schemaVersion": 1, "rootId": "   ",
        "relativePath": "models/model.safetensors", "fingerprint": DIGEST
    });
    assert!(!validator.is_valid(&whitespace_identifier));
    assert!(serde_json::from_value::<SourceLocatorV1>(whitespace_identifier).is_err());

    let mut oversized_summary =
        serde_json::to_value(plan(linked("root", "model.safetensors")).summary().unwrap()).unwrap();
    oversized_summary["layerCount"] = json!(4_294_967_296_u64);
    assert!(!validator.is_valid(&oversized_summary));
    assert!(serde_json::from_value::<ImportPlanSummaryV1>(oversized_summary).is_err());
}

#[test]
fn schema_requires_the_published_serde_semantic_validation_step() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(path).expect("read schema"))
        .expect("schema JSON parses");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");
    assert!(schema["$defs"]["checkpointInventory"]["$comment"]
        .as_str()
        .expect("published semantic validation rule")
        .contains("CheckpointInventoryV1"));

    let import_plan = plan(linked("root", "model.safetensors"));
    let record = CheckpointCatalogRecordV1::from_plan("checkpoint", &import_plan).unwrap();
    let mut exact_duplicate_layers = serde_json::to_value(&import_plan).unwrap();
    exact_duplicate_layers["layers"] = json!([
        exact_duplicate_layers["layers"][0].clone(),
        exact_duplicate_layers["layers"][0].clone()
    ]);
    assert!(!validator.is_valid(&exact_duplicate_layers));
    assert!(serde_json::from_value::<ImportPlanV1>(exact_duplicate_layers).is_err());

    let mut exact_duplicate_records =
        serde_json::to_value(CheckpointInventoryV1::new(vec![record.clone()]).unwrap()).unwrap();
    exact_duplicate_records["records"] = json!([
        exact_duplicate_records["records"][0].clone(),
        exact_duplicate_records["records"][0].clone()
    ]);
    assert!(!validator.is_valid(&exact_duplicate_records));
    assert!(serde_json::from_value::<CheckpointInventoryV1>(exact_duplicate_records).is_err());

    let mut keyed_duplicate_records =
        serde_json::to_value(CheckpointInventoryV1::new(vec![record]).unwrap()).unwrap();
    let mut rebound = keyed_duplicate_records["records"][0].clone();
    rebound["plan"]["sourceBindingIdentity"] =
        json!("sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    keyed_duplicate_records["records"] =
        json!([keyed_duplicate_records["records"][0].clone(), rebound]);
    assert!(
        validator.is_valid(&keyed_duplicate_records),
        "schema cannot express unique checkpointId"
    );
    assert!(serde_json::from_value::<CheckpointInventoryV1>(keyed_duplicate_records).is_err());

    let mut mismatched_summary = serde_json::to_value(
        CheckpointCatalogRecordV1::from_plan("checkpoint", &import_plan).unwrap(),
    )
    .unwrap();
    mismatched_summary["summary"]["semanticDigest"] =
        json!("sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    assert!(
        validator.is_valid(&mismatched_summary),
        "schema cannot compare sibling values"
    );
    assert!(serde_json::from_value::<CheckpointCatalogRecordV1>(mismatched_summary).is_err());
}
