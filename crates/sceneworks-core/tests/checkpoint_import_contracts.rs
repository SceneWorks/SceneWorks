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
        linked_plan.layers[0].source.semantic_identity(),
        managed_plan.layers[0].source.semantic_identity(),
        "equal source bytes must be locator-independent"
    );
    assert_eq!(
        linked_plan.semantic_digest(),
        managed_plan.semantic_digest()
    );
    assert_ne!(
        linked_plan.source_binding_identity(),
        managed_plan.source_binding_identity()
    );

    let linked_reference = linked_plan.plan_reference();
    let managed_reference = managed_plan.plan_reference();
    assert_eq!(
        linked_reference.semantic_digest,
        managed_reference.semantic_digest
    );
    assert_ne!(
        linked_reference.source_binding_identity,
        managed_reference.source_binding_identity
    );
    assert_eq!(
        linked_plan.semantic_digest(),
        "sha256:d224a8484eaa43937759d190ff91a6de912585505b459602a61e2150d914bef5"
    );
    assert_eq!(
        linked_plan.source_binding_identity(),
        "sha256:402ad8fc8e320fc95e242e07ca1e86194489ef793bc147ea5d331a3c9cf3afd7"
    );
    assert_eq!(
        managed_plan.source_binding_identity(),
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
    assert_eq!(first.semantic_digest(), second.semantic_digest());
    assert_eq!(
        first.source_binding_identity(),
        second.source_binding_identity()
    );
    assert_ne!(first.semantic_digest(), first.source_binding_identity());
}

#[test]
fn catalog_records_hold_only_a_reference_and_compact_summary() {
    let import_plan = plan(linked("external", "flux/model.safetensors"));
    let record = CheckpointCatalogRecordV1::from_plan("checkpoint-flux-dev", &import_plan).unwrap();
    let inventory = CheckpointInventoryV1::new(vec![record.clone()]).expect("valid inventory");
    let encoded = serde_json::to_value(&inventory).expect("serializes");

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
    let inventory = CheckpointInventoryV1::new(vec![record]).unwrap();
    let values: Vec<Value> = vec![
        serde_json::to_value(linked("root", "model.safetensors")).unwrap(),
        serde_json::to_value(&import_plan).unwrap(),
        serde_json::to_value(import_plan.plan_reference()).unwrap(),
        serde_json::to_value(import_plan.summary().unwrap()).unwrap(),
        serde_json::to_value(&inventory).unwrap(),
    ];
    let parsers: [fn(Value) -> Result<(), serde_json::Error>; 5] = [
        |value| serde_json::from_value::<SourceLocatorV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanReferenceV1>(value).map(|_| ()),
        |value| serde_json::from_value::<ImportPlanSummaryV1>(value).map(|_| ()),
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
        "checkpointId":"checkpoint",
        "plan":{"schemaVersion":2,"futureReferenceField":true},
        "summary":{"schemaVersion":2,"futureSummaryField":true}
    });
    let error = serde_json::from_value::<CheckpointCatalogRecordV1>(future_nested_plan)
        .unwrap_err()
        .to_string();
    assert!(error.contains("recompile/rescan required"), "{error}");
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
