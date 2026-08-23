use std::{fs, path::PathBuf};

use sceneworks_core::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointInventoryV1, ImportLayerV1, ImportPlanReferenceV1,
    ImportPlanSummaryV1, ImportPlanV1, ManagedProvenanceV1, SourceLocatorV1,
    CHECKPOINT_IMPORT_CONTRACT_VERSION,
};
use serde_json::{json, Value};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
type JsonParser = fn(Value) -> Result<(), serde_json::Error>;
type RawParser = fn(&str) -> Result<(), serde_json::Error>;

#[derive(Clone, Copy, Debug)]
enum VersionSurface {
    Locator,
    Plan,
    Reference,
    Summary,
    Record,
    Inventory,
}

#[derive(Debug)]
struct RawContractCase {
    label: &'static str,
    document: String,
    parse: RawParser,
}

fn parse_locator(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<SourceLocatorV1>(document).map(|_| ())
}

fn parse_plan(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<ImportPlanV1>(document).map(|_| ())
}

fn parse_reference(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<ImportPlanReferenceV1>(document).map(|_| ())
}

fn parse_summary(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<ImportPlanSummaryV1>(document).map(|_| ())
}

fn parse_record(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<CheckpointCatalogRecordV1>(document).map(|_| ())
}

fn parse_inventory(document: &str) -> Result<(), serde_json::Error> {
    serde_json::from_str::<CheckpointInventoryV1>(document).map(|_| ())
}

fn versioned_object(surface: VersionSurface, version: &str, reverse: bool) -> String {
    let version_field = format!(r#""schemaVersion":{version}"#);
    let body = match surface {
        VersionSurface::Locator => format!(
            r#""kind":"linked","rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}""#
        ),
        VersionSurface::Plan => format!(
            r#""planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{}}}]"#,
            versioned_object(VersionSurface::Locator, "1", false)
        ),
        VersionSurface::Reference => format!(
            r#""planId":"plan","semanticDigest":"sha256:{DIGEST}","sourceBindingIdentity":"sha256:{DIGEST}""#
        ),
        VersionSurface::Summary => format!(
            r#""family":"family","layerCount":1,"layerRoles":["role"],"semanticDigest":"sha256:{DIGEST}""#
        ),
        VersionSurface::Record => format!(
            r#""checkpointId":"checkpoint","plan":{},"summary":{}"#,
            versioned_object(VersionSurface::Reference, "1", false),
            versioned_object(VersionSurface::Summary, "1", false)
        ),
        VersionSurface::Inventory => r#""records":[]"#.to_owned(),
    };
    if reverse {
        format!("{{{body},{version_field}}}")
    } else {
        format!("{{{version_field},{body}}}")
    }
}

fn managed_locator_object(version: &str, reverse: bool) -> String {
    let version_field = format!(r#""schemaVersion":{version}"#);
    let body = format!(
        r#""kind":"managed","installId":"install","relativePath":"model.safetensors","sha256":"{DIGEST}","provenance":{{"source":"huggingface"}}"#
    );
    if reverse {
        format!("{{{body},{version_field}}}")
    } else {
        format!("{{{version_field},{body}}}")
    }
}

fn versioned_fields(surface: VersionSurface, version: &str) -> Vec<String> {
    let schema = format!(r#""schemaVersion":{version}"#);
    match surface {
        VersionSurface::Locator => vec![
            schema,
            r#""kind":"linked""#.to_owned(),
            r#""rootId":"root""#.to_owned(),
            r#""relativePath":"model.safetensors""#.to_owned(),
            format!(r#""fingerprint":"{DIGEST}""#),
        ],
        VersionSurface::Plan => vec![
            schema,
            r#""planId":"plan""#.to_owned(),
            r#""family":"family""#.to_owned(),
            format!(
                r#""layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{}}}]"#,
                versioned_object(VersionSurface::Locator, "1", false)
            ),
        ],
        VersionSurface::Reference => vec![
            schema,
            r#""planId":"plan""#.to_owned(),
            format!(r#""semanticDigest":"sha256:{DIGEST}""#),
            format!(r#""sourceBindingIdentity":"sha256:{DIGEST}""#),
        ],
        VersionSurface::Summary => vec![
            schema,
            r#""family":"family""#.to_owned(),
            r#""layerCount":1"#.to_owned(),
            r#""layerRoles":["role"]"#.to_owned(),
            format!(r#""semanticDigest":"sha256:{DIGEST}""#),
        ],
        VersionSurface::Record => vec![
            schema,
            r#""checkpointId":"checkpoint""#.to_owned(),
            format!(
                r#""plan":{}"#,
                versioned_object(VersionSurface::Reference, "1", false)
            ),
            format!(
                r#""summary":{}"#,
                versioned_object(VersionSurface::Summary, "1", false)
            ),
        ],
        VersionSurface::Inventory => vec![schema, r#""records":[]"#.to_owned()],
    }
}

fn managed_locator_fields(version: &str) -> Vec<String> {
    vec![
        format!(r#""schemaVersion":{version}"#),
        r#""kind":"managed""#.to_owned(),
        r#""installId":"install""#.to_owned(),
        r#""relativePath":"model.safetensors""#.to_owned(),
        format!(r#""sha256":"{DIGEST}""#),
        r#""provenance":{"source":"huggingface"}"#.to_owned(),
    ]
}

fn object_from_fields(fields: &[String]) -> String {
    format!("{{{}}}", fields.join(","))
}

fn object_field_permutations(mut fields: Vec<String>) -> Vec<String> {
    fn visit(fields: &mut [String], index: usize, output: &mut Vec<String>) {
        if index == fields.len() {
            output.push(object_from_fields(fields));
            return;
        }
        for candidate in index..fields.len() {
            fields.swap(index, candidate);
            visit(fields, index + 1, output);
            fields.swap(index, candidate);
        }
    }

    let mut output = Vec::new();
    visit(&mut fields, 0, &mut output);
    output
}

fn next_xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn generated_json(state: &mut u64, depth: usize) -> String {
    let choice_limit = if depth >= 3 { 4 } else { 7 };
    match (next_xorshift(state) as usize) % choice_limit {
        0 => {
            const NUMBERS: [&str; 10] = [
                "0",
                "-0",
                "1.25",
                "0.10e1",
                "4294967295",
                "4294967296",
                "1e400",
                "-1e400",
                "2E+17",
                "10e-1",
            ];
            NUMBERS[(next_xorshift(state) as usize) % NUMBERS.len()].to_owned()
        }
        1 => format!(r#""text-{:016x}-r\u006fot\/path""#, next_xorshift(state)),
        2 => ["true", "false", "null"][(next_xorshift(state) as usize) % 3].to_owned(),
        3 => format!("{}", next_xorshift(state) % 1_000_000),
        4 => {
            let length = 1 + (next_xorshift(state) as usize % 4);
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(generated_json(state, depth + 1));
            }
            format!("[{}]", values.join(","))
        }
        5 => {
            let length = 1 + (next_xorshift(state) as usize % 4);
            let mut fields = Vec::with_capacity(length);
            for index in 0..length {
                let suffix = next_xorshift(state);
                fields.push(format!(
                    r#""k{depth}-{index}-{suffix:016x}":{}"#,
                    generated_json(state, depth + 1)
                ));
            }
            object_from_fields(&fields)
        }
        _ => format!(
            r#"{{"nested":[{},{{"leaf":{}}}]}}"#,
            generated_json(state, depth + 1),
            generated_json(state, depth + 1)
        ),
    }
}

fn raw_occurrences(surface: VersionSurface, envelope: &str) -> Vec<RawContractCase> {
    let valid_reference = versioned_object(VersionSurface::Reference, "1", false);
    let valid_summary = versioned_object(VersionSurface::Summary, "1", false);
    let direct_parse = match surface {
        VersionSurface::Locator => parse_locator as RawParser,
        VersionSurface::Plan => parse_plan,
        VersionSurface::Reference => parse_reference,
        VersionSurface::Summary => parse_summary,
        VersionSurface::Record => parse_record,
        VersionSurface::Inventory => parse_inventory,
    };
    let mut cases = vec![RawContractCase {
        label: "direct",
        document: envelope.to_owned(),
        parse: direct_parse,
    }];
    match surface {
        VersionSurface::Locator => cases.push(RawContractCase {
            label: "plan.layers[*].source",
            document: format!(
                r#"{{"schemaVersion":1,"planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{envelope}}}]}}"#
            ),
            parse: parse_plan,
        }),
        VersionSurface::Reference => {
            let record = format!(
                r#"{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{envelope},"summary":{valid_summary}}}"#
            );
            cases.push(RawContractCase {
                label: "catalog.plan",
                document: record.clone(),
                parse: parse_record,
            });
            cases.push(RawContractCase {
                label: "inventory.records[*].plan",
                document: format!(r#"{{"schemaVersion":1,"records":[{record}]}}"#),
                parse: parse_inventory,
            });
        }
        VersionSurface::Summary => {
            let record = format!(
                r#"{{"schemaVersion":1,"checkpointId":"checkpoint","plan":{valid_reference},"summary":{envelope}}}"#
            );
            cases.push(RawContractCase {
                label: "catalog.summary",
                document: record.clone(),
                parse: parse_record,
            });
            cases.push(RawContractCase {
                label: "inventory.records[*].summary",
                document: format!(r#"{{"schemaVersion":1,"records":[{record}]}}"#),
                parse: parse_inventory,
            });
        }
        VersionSurface::Record => cases.push(RawContractCase {
            label: "inventory.records[*]",
            document: format!(r#"{{"schemaVersion":1,"records":[{envelope}]}}"#),
            parse: parse_inventory,
        }),
        VersionSurface::Plan | VersionSurface::Inventory => {}
    }
    cases
}

fn replace_schema_version_key(document: &str) -> String {
    document.replacen("\"schemaVersion\"", "\"\\u0073chemaVersion\"", 1)
}

fn add_future_payload(envelope: &str, payload: &str, reverse: bool) -> String {
    let body = envelope
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .expect("object fixture");
    if reverse {
        format!(r#"{{"futurePayload":{payload},{body}}}"#)
    } else {
        format!(r#"{{{body},"futurePayload":{payload}}}"#)
    }
}

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
            ..ManagedProvenanceV1::default()
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
    // sc-20636: the semantic digest excludes `planId`. The inspector derives a plan id from the
    // checkpoint id, which encodes ownership, so a plan id in the semantic form made two copies of
    // identical bytes — one linked, one managed — carry DIFFERENT "locator-independent" digests.
    // Pinned as a differently-named plan over the same layers so a reintroduction reds here.
    let renamed_plan = ImportPlanV1::new(
        "checkpoint-plan-deadbeef",
        &linked_plan.family,
        linked_plan.layers.clone(),
    )
    .expect("valid import plan");
    assert_eq!(
        renamed_plan.semantic_digest().unwrap(),
        linked_plan.semantic_digest().unwrap(),
        "the plan id is a document name, not content, and must not reach the semantic digest"
    );
    assert_ne!(
        renamed_plan.source_binding_identity().unwrap(),
        linked_plan.source_binding_identity().unwrap(),
        "the source binding still binds the exact document"
    );
    assert_eq!(
        linked_plan.semantic_digest().unwrap(),
        "sha256:11406d061fa8c81e6bac300bd946a3990b89556878d8fd5d66a15cc2e96c2f52"
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

    let mut wrong_reference_semantic = record.clone();
    wrong_reference_semantic.plan.semantic_digest = forged_identity.to_owned();
    assert!(wrong_reference_semantic
        .validate_loaded_plan(&import_plan)
        .is_err());

    let mut wrong_summary_semantic = record.clone();
    wrong_summary_semantic.summary.semantic_digest = forged_identity.to_owned();
    assert!(wrong_summary_semantic
        .validate_loaded_plan(&import_plan)
        .is_err());

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
    assert_eq!(
        ImportPlanV1::checked_layer_count(u32::MAX as usize).unwrap(),
        u32::MAX
    );
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
    assert_eq!(
        defs["nonBlankText"]["pattern"],
        "(?=.*[^\\u0009-\\u000D\\u0020\\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000])"
    );
    assert_eq!(defs["relativePath"]["pattern"], "^(?=[\\s\\S]*[^\\u0009-\\u000D\\u0020\\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000])(?!/)(?![\\s\\S]*\\\\)(?![\\s\\S]*:)(?![\\s\\S]*//)(?![\\s\\S]*(?:^|/)\\.{1,2}(?:/|$))(?![\\s\\S]*(?:^|/)\\.{1,2}$)(?![\\s\\S]*\\/$)[^\\x00-\\x1F\\x7F]+$");
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

#[test]
fn frozen_exact_u32_lexemes_cover_every_versioned_occurrence_and_field_order() {
    let surfaces = [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ];
    let accepted = ["1", "1.0", "1e0", "1E+0", "10e-1", "0.10e1"];
    let unsupported = [
        ("0", 0_u32),
        ("0.0", 0),
        ("2", 2),
        ("2.0", 2),
        ("2e0", 2),
        ("4294967295", u32::MAX),
    ];
    let invalid = [
        "null",
        "true",
        "\"1\"",
        "[]",
        "{}",
        "1.5",
        "1e-1",
        "-0",
        "-1",
        "-1e0",
        "4294967296",
        "1e400",
        "-1e400",
    ];

    for surface in surfaces {
        for reverse in [false, true] {
            for lexeme in accepted {
                let envelope = versioned_object(surface, lexeme, reverse);
                for case in raw_occurrences(surface, &envelope) {
                    let result = (case.parse)(&case.document);
                    assert!(
                        result.is_ok(),
                        "{surface:?} {} reverse={reverse} lexeme={lexeme}: {result:?}",
                        case.label
                    );
                }
            }
            for (lexeme, expected) in unsupported {
                let envelope = versioned_object(surface, lexeme, reverse);
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    let expected = format!("schema version {expected} is unsupported");
                    assert!(
                        error.contains(&expected) && error.contains("recompile/rescan required"),
                        "{surface:?} {} reverse={reverse} lexeme={lexeme}: {error}",
                        case.label
                    );
                }
            }
            for lexeme in invalid {
                let envelope = versioned_object(surface, lexeme, reverse);
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    assert!(
                        error.starts_with("checkpoint-import schemaVersion must be a u32"),
                        "{surface:?} {} reverse={reverse} lexeme={lexeme}: {error}",
                        case.label
                    );
                }
            }
        }
    }
}

#[test]
fn managed_locator_variant_has_the_same_lexical_and_nested_version_contract() {
    for reverse in [false, true] {
        for lexeme in ["1", "1.0", "1e0", "1E+0", "10e-1", "0.10e1"] {
            let envelope = managed_locator_object(lexeme, reverse);
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let result = (case.parse)(&case.document);
                assert!(result.is_ok(), "{} {lexeme}: {result:?}", case.label);
            }
        }
        for (lexeme, expected) in [("0.0", 0_u32), ("2e0", 2), ("4294967295", u32::MAX)] {
            let envelope = managed_locator_object(lexeme, reverse);
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let error = (case.parse)(&case.document).unwrap_err().to_string();
                assert!(
                    error.contains(&format!("schema version {expected} is unsupported")),
                    "{error}"
                );
            }
        }
        for lexeme in ["null", "1.5", "-0", "4294967296", "1e400", "-1e400"] {
            let envelope = managed_locator_object(lexeme, reverse);
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let error = (case.parse)(&case.document).unwrap_err().to_string();
                assert!(
                    error.starts_with("checkpoint-import schemaVersion must be a u32"),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn missing_versions_and_all_sibling_field_permutations_cover_every_occurrence() {
    let accepted = ["1", "1.0", "1e0", "1E+0", "10e-1", "0.10e1"];
    let unsupported = [
        ("0", 0_u32),
        ("0.0", 0),
        ("2", 2),
        ("2.0", 2),
        ("2e0", 2),
        ("4294967295", u32::MAX),
    ];
    let invalid = [
        "null",
        "true",
        "\"1\"",
        "[]",
        "{}",
        "1.5",
        "1e-1",
        "-0",
        "-1",
        "-1e0",
        "4294967296",
        "1e400",
        "-1e400",
    ];

    for surface in [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ] {
        for lexeme in accepted {
            for envelope in object_field_permutations(versioned_fields(surface, lexeme)) {
                for case in raw_occurrences(surface, &envelope) {
                    let result = (case.parse)(&case.document);
                    assert!(
                        result.is_ok(),
                        "{surface:?} {} lexeme={lexeme}: {result:?}",
                        case.label
                    );
                }
            }
        }
        for (lexeme, expected) in unsupported {
            for envelope in object_field_permutations(versioned_fields(surface, lexeme)) {
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    assert!(
                        error.contains(&format!("schema version {expected} is unsupported")),
                        "{surface:?} {} lexeme={lexeme}: {error}",
                        case.label
                    );
                }
            }
        }
        for lexeme in invalid {
            for envelope in object_field_permutations(versioned_fields(surface, lexeme)) {
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    assert!(
                        error.starts_with("checkpoint-import schemaVersion must be a u32"),
                        "{surface:?} {} lexeme={lexeme}: {error}",
                        case.label
                    );
                }
            }
        }
        let mut missing_fields = versioned_fields(surface, "1");
        missing_fields.remove(0);
        for envelope in object_field_permutations(missing_fields) {
            for case in raw_occurrences(surface, &envelope) {
                let error = (case.parse)(&case.document).unwrap_err().to_string();
                assert!(
                    error.starts_with("checkpoint-import schemaVersion must be a u32"),
                    "{surface:?} missing {}: {error}",
                    case.label
                );
            }
        }
    }

    for lexeme in accepted {
        for envelope in object_field_permutations(managed_locator_fields(lexeme)) {
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let result = (case.parse)(&case.document);
                assert!(
                    result.is_ok(),
                    "managed {} lexeme={lexeme}: {result:?}",
                    case.label
                );
            }
        }
    }
    for (lexeme, expected) in unsupported {
        for envelope in object_field_permutations(managed_locator_fields(lexeme)) {
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let error = (case.parse)(&case.document).unwrap_err().to_string();
                assert!(
                    error.contains(&format!("schema version {expected} is unsupported")),
                    "managed {} lexeme={lexeme}: {error}",
                    case.label
                );
            }
        }
    }
    for lexeme in invalid {
        for envelope in object_field_permutations(managed_locator_fields(lexeme)) {
            for case in raw_occurrences(VersionSurface::Locator, &envelope) {
                let error = (case.parse)(&case.document).unwrap_err().to_string();
                assert!(
                    error.starts_with("checkpoint-import schemaVersion must be a u32"),
                    "managed {} lexeme={lexeme}: {error}",
                    case.label
                );
            }
        }
    }
    let mut missing_managed_fields = managed_locator_fields("1");
    missing_managed_fields.remove(0);
    for envelope in object_field_permutations(missing_managed_fields) {
        for case in raw_occurrences(VersionSurface::Locator, &envelope) {
            let error = (case.parse)(&case.document).unwrap_err().to_string();
            assert!(
                error.starts_with("checkpoint-import schemaVersion must be a u32"),
                "managed missing {}: {error}",
                case.label
            );
        }
    }
}

#[test]
fn duplicate_schema_version_forms_are_order_independent_at_every_occurrence() {
    let version_pairs = [
        [r#""schemaVersion":2"#, r#""schemaVersion":1"#],
        [r#""schemaVersion":1"#, r#""schemaVersion":2"#],
        [r#""schemaVersion":1"#, r#""schemaVersion":1"#],
        [r#""schemaVersion":1"#, r#""\u0073chemaVersion":1"#],
    ];
    for surface in [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ] {
        for reverse in [false, true] {
            for pair in version_pairs {
                let mut fields = versioned_fields(surface, "1");
                fields.remove(0);
                if reverse {
                    fields.extend(pair.into_iter().map(str::to_owned));
                } else {
                    fields.splice(0..0, pair.into_iter().map(str::to_owned));
                }
                let envelope = object_from_fields(&fields);
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    assert!(
                        error.contains("duplicate object key `schemaVersion`"),
                        "{error}"
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_or_missing_current_envelopes_precede_nested_future_versions() {
    let future_locator = r#"{"kind":"future","schemaVersion":2,"future":1e400}"#;
    let future_reference = r#"{"schemaVersion":2,"future":1e400}"#;
    let future_record = r#"{"schemaVersion":2,"future":1e400}"#;
    let documents = [
        (
            format!(
                r#"{{"planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{future_locator}}}]}}"#
            ),
            parse_plan as RawParser,
        ),
        (
            format!(
                r#"{{"schemaVersion":"bad","planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"model.safetensors","source":{future_locator}}}]}}"#
            ),
            parse_plan,
        ),
        (
            format!(
                r#"{{"checkpointId":"checkpoint","plan":{future_reference},"summary":{}}}"#,
                versioned_object(VersionSurface::Summary, "1", false)
            ),
            parse_record,
        ),
        (
            format!(
                r#"{{"schemaVersion":"bad","checkpointId":"checkpoint","plan":{future_reference},"summary":{}}}"#,
                versioned_object(VersionSurface::Summary, "1", false)
            ),
            parse_record,
        ),
        (
            format!(r#"{{"records":[{future_record}]}}"#),
            parse_inventory,
        ),
        (
            format!(r#"{{"schemaVersion":"bad","records":[{future_record}]}}"#),
            parse_inventory,
        ),
    ];
    for (document, parse) in documents {
        let error = parse(&document).unwrap_err().to_string();
        assert!(
            error.starts_with("checkpoint-import schemaVersion must be a u32"),
            "{error}"
        );
        assert!(!error.contains("recompile/rescan required"), "{error}");
    }
}

#[test]
fn future_envelopes_skip_extreme_and_nested_body_numbers_at_every_occurrence() {
    let surfaces = [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ];
    let payloads = [
        "1e400",
        "-1e400",
        r#"{"array":[1e400,{"deeper":[-1e400,0.10e1]}]}"#,
    ];
    for surface in surfaces {
        for reverse in [false, true] {
            for payload in payloads {
                let envelope =
                    add_future_payload(&versioned_object(surface, "2", reverse), payload, reverse);
                for case in raw_occurrences(surface, &envelope) {
                    let error = (case.parse)(&case.document).unwrap_err().to_string();
                    assert!(
                        error.contains("schema version 2 is unsupported")
                            && error.contains("recompile/rescan required"),
                        "{surface:?} {} reverse={reverse} payload={payload}: {error}",
                        case.label
                    );
                    assert!(!error.contains("number out of range"), "{error}");
                }
            }
        }
    }
}

#[test]
fn escaped_keys_values_paths_and_unicode_are_decoded_before_contract_decisions() {
    for surface in [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ] {
        let escaped = replace_schema_version_key(&versioned_object(surface, "2", false));
        for case in raw_occurrences(surface, &escaped) {
            let error = (case.parse)(&case.document).unwrap_err().to_string();
            assert!(error.contains("schema version 2 is unsupported"), "{error}");
        }

        let duplicate = versioned_object(surface, "1", false).replacen(
            r#""schemaVersion":1"#,
            r#""schemaVersion":2,"\u0073chemaVersion":1"#,
            1,
        );
        for case in raw_occurrences(surface, &duplicate) {
            let error = (case.parse)(&case.document).unwrap_err().to_string();
            assert!(
                error.contains("duplicate object key `schemaVersion`"),
                "{error}"
            );
        }
    }

    let ordinary_document = format!(
        r#"{{"schemaVersion":1,"planId":"plan","family":"family","layers":[{{"layerId":"layer","role":"role","targetPath":"models/file.safetensors","source":{{"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"models/file.safetensors","fingerprint":"{DIGEST}"}}}}]}}"#
    );
    assert!(parse_plan(&ordinary_document).is_ok());
    let escaped_document = format!(
        r#"{{"schemaVersion":1,"planId":"pl\u0061n","family":"f\u0061mily","layers":[{{"layerId":"l\u0061yer","role":"r\u006fle","targetPath":"models\/file.safetensors","source":{{"kind":"linked","schemaVersion":1,"rootId":"r\u006fot","relativePath":"models\/file.safetensors","fingerprint":"{DIGEST}"}}}}]}}"#
    );
    let escaped: ImportPlanV1 = serde_json::from_str(&escaped_document).unwrap();
    let ordinary: ImportPlanV1 = serde_json::from_str(&ordinary_document).unwrap();
    assert_eq!(escaped, ordinary);
    assert_eq!(
        escaped.canonical_json().unwrap(),
        ordinary.canonical_json().unwrap()
    );
    assert_eq!(
        escaped.semantic_digest().unwrap(),
        ordinary.semantic_digest().unwrap()
    );
    assert_eq!(
        escaped.source_binding_identity().unwrap(),
        ordinary.source_binding_identity().unwrap()
    );

    let unicode: SourceLocatorV1 = serde_json::from_str(&format!(
        r#"{{"kind":"linked","schemaVersion":1,"rootId":"root-\uD83D\uDE80","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
    ))
    .unwrap();
    assert!(unicode.canonical_json().unwrap().contains('🚀'));

    for invalid_path in [r#"safe\\model"#, r#"safe\u0000model"#] {
        let document = format!(
            r#"{{"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"{invalid_path}","fingerprint":"{DIGEST}"}}"#
        );
        let error = serde_json::from_str::<SourceLocatorV1>(&document)
            .unwrap_err()
            .to_string();
        assert!(error.contains("portable confined relative path"), "{error}");
    }
}

#[test]
fn json_syntax_trailing_content_and_depth_boundaries_never_admit_documents() {
    for invalid_version in ["01", "+1", ".1", "1.", "1e", "NaN", "Infinity"] {
        let document = format!(r#"{{"schemaVersion":{invalid_version},"records":[]}}"#);
        let error = parse_inventory(&document).unwrap_err().to_string();
        assert!(
            !error.starts_with("checkpoint-import schemaVersion must be a u32"),
            "invalid JSON number grammar leaked into contract diagnostics: {error}"
        );
    }
    for malformed in [
        r#"{"schemaVersion":1,"records":[]"#,
        r#"{"schemaVersion":1,"records":[],}"#,
        r#"{"schemaVersion":1,"records":[]]"#,
    ] {
        assert!(parse_inventory(malformed).is_err(), "{malformed}");
    }
    assert!(parse_inventory(r#"{"schemaVersion":1,"records":[]} true"#).is_err());
    assert!(parse_inventory(r#"{"schemaVersion":1,"records":[]}   "#).is_ok());
    assert!(parse_inventory(r#"{"schemaVersion":2,"future":1e400} true"#).is_err());

    let nested_within_bound = format!("{}0{}", "[".repeat(40), "]".repeat(40));
    let within_bound = format!(r#"{{"schemaVersion":2,"future":{nested_within_bound} }}"#);
    let error = parse_inventory(&within_bound).unwrap_err().to_string();
    assert!(error.contains("schema version 2 is unsupported"), "{error}");

    let nested_beyond_bound = format!("{}0{}", "[".repeat(200), "]".repeat(200));
    let beyond_bound = format!(r#"{{"schemaVersion":2,"future":{nested_beyond_bound} }}"#);
    let result = std::panic::catch_unwind(|| parse_inventory(&beyond_bound));
    assert!(result.is_ok(), "depth handling panicked");
    assert!(result.unwrap().is_err(), "depth boundary admitted document");

    let lone_surrogate = format!(
        r#"{{"kind":"linked","schemaVersion":1,"rootId":"\uD800","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
    );
    assert!(parse_locator(&lone_surrogate).is_err());
    let invalid_escape = format!(
        r#"{{"kind":"linked","schemaVersion":1,"rootId":"\q","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
    );
    assert!(parse_locator(&invalid_escape).is_err());
    let invalid_utf8 = [
        b'{', b'"', b's', b'c', b'h', b'e', b'm', b'a', b'V', b'e', b'r', b's', b'i', b'o', b'n',
        b'"', b':', b'1', b',', b'"', b'r', b'e', b'c', b'o', b'r', b'd', b's', b'"', b':', b'[',
        0xff, b']', b'}',
    ];
    assert!(serde_json::from_slice::<CheckpointInventoryV1>(&invalid_utf8).is_err());
}

#[test]
fn fixed_seed_generated_body_property_matrix_is_bounded_and_order_independent() {
    let surfaces = [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ];
    let mut state = 0x8a5c_d789_635d_2dff_u64;
    let mut unique_bodies = std::collections::BTreeSet::new();
    let mut defect_counts = [0_usize; 4];
    let mut attempts = 0_usize;
    while unique_bodies.len() < 1_000 {
        attempts += 1;
        assert!(
            attempts <= 10_000,
            "generator could not produce 1,000 unique bodies"
        );
        let generated = generated_json(&mut state, 0);
        if !unique_bodies.insert(generated.clone()) {
            continue;
        }
        let iteration = unique_bodies.len() - 1;
        let payload = format!(r#"{{"generated":{generated}}}"#);
        let surface = surfaces[(next_xorshift(&mut state) as usize) % surfaces.len()];

        let defect = (iteration % 4) as u8;
        defect_counts[usize::from(defect)] += 1;
        let mut fields = versioned_fields(surface, "1");
        let expected = match defect {
            0 => {
                fields[0] = r#""schemaVersion":2e0"#.to_owned();
                fields.push(format!(r#""futureBody":{payload}"#));
                "schema version 2 is unsupported"
            }
            1 => {
                fields.push(r#""\u0073chemaVersion":2"#.to_owned());
                fields.push(format!(r#""duplicateVersionBody":{payload}"#));
                "duplicate object key `schemaVersion`"
            }
            2 => {
                const INVALID_VERSIONS: [&str; 5] = ["\"bad\"", "-0", "1.5", "1e400", "{}"];
                let invalid =
                    INVALID_VERSIONS[(next_xorshift(&mut state) as usize) % INVALID_VERSIONS.len()];
                fields[0] = format!(r#""schemaVersion":{invalid}"#);
                fields.push(format!(r#""invalidVersionBody":{payload}"#));
                "schemaVersion must be a u32"
            }
            _ => {
                fields.push(format!(r#""generatedBody":{payload}"#));
                fields.push(format!(
                    r#""generatedBody":{}"#,
                    generated_json(&mut state, 0)
                ));
                "duplicate object key `generatedBody`"
            }
        };
        if next_xorshift(&mut state) & 1 != 0 {
            fields[0] = fields[0].replacen("schemaVersion", r#"\u0073chemaVersion"#, 1);
        }
        let rotation = (next_xorshift(&mut state) as usize) % fields.len();
        fields.rotate_left(rotation);
        if next_xorshift(&mut state) & 1 != 0 {
            fields.reverse();
        }
        let envelope = object_from_fields(&fields);
        for case in raw_occurrences(surface, &envelope) {
            let error = (case.parse)(&case.document).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "iteration={iteration} defect={defect} {surface:?} {}: expected {expected}: {error}",
                case.label
            );
        }
    }
    assert_eq!(unique_bodies.len(), 1_000);
    assert!(
        defect_counts.iter().all(|&count| count >= 200),
        "{defect_counts:?}"
    );
}

#[test]
fn sibling_diagnostic_precedence_is_permutation_invariant() {
    let valid_reference = versioned_object(VersionSurface::Reference, "1", false);
    let valid_summary = versioned_object(VersionSurface::Summary, "1", false);
    let future = r#"{"schemaVersion":2,"future":1e400}"#;
    let duplicate_version = format!(
        r#"{{"schemaVersion":1,"\u0073chemaVersion":1,"checkpointId":"dup-version","plan":{valid_reference},"summary":{valid_summary}}}"#
    );
    let invalid_version = format!(
        r#"{{"schemaVersion":"bad","checkpointId":"invalid","plan":{valid_reference},"summary":{valid_summary}}}"#
    );
    let duplicate_body = format!(
        r#"{{"schemaVersion":1,"checkpointId":"a","checkpointId":"b","plan":{valid_reference},"summary":{valid_summary}}}"#
    );

    let pairs = [
        (
            duplicate_version.as_str(),
            future,
            "duplicate object key `schemaVersion`",
        ),
        (
            duplicate_version.as_str(),
            invalid_version.as_str(),
            "duplicate object key `schemaVersion`",
        ),
        (
            duplicate_version.as_str(),
            duplicate_body.as_str(),
            "duplicate object key `schemaVersion`",
        ),
        (
            invalid_version.as_str(),
            future,
            "schema version 2 is unsupported",
        ),
        (
            invalid_version.as_str(),
            duplicate_body.as_str(),
            "schemaVersion must be a u32",
        ),
        (
            future,
            duplicate_body.as_str(),
            "schema version 2 is unsupported",
        ),
    ];
    for (left, right, expected) in pairs {
        for records in [format!("{left},{right}"), format!("{right},{left}")] {
            let document = format!(r#"{{"schemaVersion":1,"records":[{records}]}}"#);
            let error = parse_inventory(&document).unwrap_err().to_string();
            assert!(error.contains(expected), "expected {expected}: {error}");
        }
    }

    let outer_invalid_nested_future = format!(r#"{{"schemaVersion":"bad","records":[{future}]}}"#);
    let error = parse_inventory(&outer_invalid_nested_future)
        .unwrap_err()
        .to_string();
    assert!(
        error.starts_with("checkpoint-import schemaVersion must be a u32"),
        "{error}"
    );

    let unknown_future =
        r#"{"schemaVersion":1,"records":[],"unknown":{"schemaVersion":2,"future":1e400}}"#;
    let error = parse_inventory(unknown_future).unwrap_err().to_string();
    assert!(!error.contains("recompile/rescan required"), "{error}");

    let duplicate_layer = format!(
        r#"{{"schemaVersion":1,"planId":"plan","family":"family","layers":[{{"layerId":"a","layerId":"b","role":"role","targetPath":"model.safetensors","source":{}}}]}}"#,
        versioned_object(VersionSurface::Locator, "1", false)
    );
    let error = parse_plan(&duplicate_layer).unwrap_err().to_string();
    assert!(error.contains("duplicate object key `layerId`"), "{error}");

    let duplicate_provenance = format!(
        r#"{{"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":"{DIGEST}","provenance":{{"source":"a","source":"b"}}}}"#
    );
    let error = parse_locator(&duplicate_provenance)
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate object key `source`"), "{error}");

    let lexical_duplicate = format!(
        r#"{{"kind":"linked","schemaVersion":1,"z":1,"z":2,"a":1,"a":2,"rootId":"root","relativePath":"model.safetensors","fingerprint":"{DIGEST}"}}"#
    );
    let error = parse_locator(&lexical_duplicate).unwrap_err().to_string();
    assert!(error.contains("duplicate object key `a`"), "{error}");
}

#[test]
fn layer_count_uses_the_same_exact_u32_decoder_as_schema_versions() {
    for lexeme in ["1", "1.0", "1e0", "1E+0", "10e-1", "0.10e1"] {
        let summary = versioned_object(VersionSurface::Summary, "1", false).replacen(
            r#""layerCount":1"#,
            &format!(r#""layerCount":{lexeme}"#),
            1,
        );
        for case in raw_occurrences(VersionSurface::Summary, &summary) {
            let result = (case.parse)(&case.document);
            assert!(result.is_ok(), "{} {lexeme}: {result:?}", case.label);
        }
    }
    for lexeme in [
        "0",
        "0.0",
        "1.5",
        "1e-1",
        "-0",
        "-1",
        "4294967296",
        "1e400",
        "-1e400",
    ] {
        let summary = versioned_object(VersionSurface::Summary, "1", false).replacen(
            r#""layerCount":1"#,
            &format!(r#""layerCount":{lexeme}"#),
            1,
        );
        for case in raw_occurrences(VersionSurface::Summary, &summary) {
            let error = (case.parse)(&case.document).unwrap_err().to_string();
            assert!(
                error.contains("layerCount") || error.contains("layer count"),
                "{} {lexeme}: {error}",
                case.label
            );
        }
    }
}

#[test]
fn schema_and_serde_reject_path_separators_hidden_by_ecmascript_line_separators() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(path).expect("read schema"))
        .expect("schema JSON parses");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");

    for relative_path in ["a\u{2028}b", "a\u{2029}b"] {
        for value in [
            json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":relative_path,"fingerprint":DIGEST}),
            json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":relative_path,"sha256":DIGEST,"provenance":{"source":"huggingface"}}),
        ] {
            assert!(
                validator.is_valid(&value),
                "schema rejected ordinary path: {value}"
            );
            assert!(
                serde_json::from_value::<SourceLocatorV1>(value.clone()).is_ok(),
                "serde rejected ordinary path: {value}"
            );
        }
    }

    for (label, relative_path) in [
        ("U+2028 slash", "a\u{2028}//b"),
        ("U+2028 colon", "a\u{2028}:b"),
        ("U+2028 trailing separator", "a\u{2028}x/"),
        ("U+2029 slash", "a\u{2029}//b"),
        ("U+2029 colon", "a\u{2029}:b"),
        ("U+2029 trailing separator", "a\u{2029}x/"),
    ] {
        for value in [
            json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":relative_path,"fingerprint":DIGEST}),
            json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":relative_path,"sha256":DIGEST,"provenance":{"source":"huggingface"}}),
        ] {
            assert!(
                !validator.is_valid(&value),
                "schema admitted {label}: {value}"
            );
            assert!(
                serde_json::from_value::<SourceLocatorV1>(value.clone()).is_err(),
                "serde admitted {label}: {value}"
            );
        }
    }
}

#[test]
fn schema_expressible_rules_are_bidirectionally_aligned_and_semantic_gaps_are_explicit() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(path).expect("read schema"))
        .expect("schema JSON parses");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");
    let defs = schema["$defs"].as_object().unwrap();
    assert_eq!(
        defs["planSummary"]["properties"]["layerRoles"]["maxItems"],
        4_294_967_295_u64
    );

    let aligned_locator_mutations = [
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST,"unknown":true}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"/absolute","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"safe//file","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"safe/../file","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"safe:","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"safe/","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"safe\u{7f}file","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"\u{85}","relativePath":"model.safetensors","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"\u{feff}","relativePath":"model.safetensors","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"\u{85}","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"\u{feff}","fingerprint":DIGEST}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":DIGEST.to_uppercase()}),
        json!({"kind":"linked","schemaVersion":1,"rootId":"root","relativePath":"model.safetensors","fingerprint":"0".repeat(63)}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":"huggingface"}}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":"huggingface","reference":null}}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":"huggingface","reference":"org/model"}}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":" "}}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":"huggingface","reference":" "}}),
        json!({"kind":"managed","schemaVersion":1,"installId":"install","relativePath":"model.safetensors","sha256":DIGEST,"provenance":{"source":"huggingface","extra":true}}),
    ];
    for value in aligned_locator_mutations {
        assert_eq!(
            validator.is_valid(&value),
            serde_json::from_value::<SourceLocatorV1>(value.clone()).is_ok(),
            "schema/serde mismatch for {value}"
        );
    }

    let managed_without_reference: SourceLocatorV1 = serde_json::from_value(json!({
        "kind":"managed","schemaVersion":1,"installId":"install",
        "relativePath":"model.safetensors","sha256":DIGEST,
        "provenance":{"source":"huggingface"}
    }))
    .unwrap();
    let managed_null_reference: SourceLocatorV1 = serde_json::from_value(json!({
        "kind":"managed","schemaVersion":1,"installId":"install",
        "relativePath":"model.safetensors","sha256":DIGEST,
        "provenance":{"source":"huggingface","reference":null}
    }))
    .unwrap();
    assert_eq!(managed_without_reference, managed_null_reference);
    assert!(!managed_null_reference
        .canonical_json()
        .unwrap()
        .contains("reference"));

    let strict_plan = plan(managed("install", "model.safetensors"));
    let strict_record = CheckpointCatalogRecordV1::from_plan("checkpoint", &strict_plan).unwrap();
    let mut unknown_plan = serde_json::to_value(&strict_plan).unwrap();
    unknown_plan["unknown"] = json!(true);
    let mut unknown_layer = serde_json::to_value(&strict_plan).unwrap();
    unknown_layer["layers"][0]["unknown"] = json!(true);
    let mut unknown_locator = serde_json::to_value(&strict_plan).unwrap();
    unknown_locator["layers"][0]["source"]["unknown"] = json!(true);
    let mut unknown_provenance = serde_json::to_value(&strict_plan).unwrap();
    unknown_provenance["layers"][0]["source"]["provenance"]["unknown"] = json!(true);
    let mut unknown_reference = serde_json::to_value(&strict_record).unwrap();
    unknown_reference["plan"]["unknown"] = json!(true);
    let mut unknown_summary = serde_json::to_value(&strict_record).unwrap();
    unknown_summary["summary"]["unknown"] = json!(true);
    let mut unknown_record = serde_json::to_value(&strict_record).unwrap();
    unknown_record["unknown"] = json!(true);
    let mut unknown_nested_record =
        serde_json::to_value(CheckpointInventoryV1::new(vec![strict_record]).unwrap()).unwrap();
    unknown_nested_record["records"][0]["unknown"] = json!(true);
    let mut unknown_inventory = unknown_nested_record.clone();
    unknown_inventory["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("unknown");
    unknown_inventory["unknown"] = json!(true);
    let strict_unknowns: Vec<(Value, JsonParser)> = vec![
        (unknown_plan, |value| {
            serde_json::from_value::<ImportPlanV1>(value).map(|_| ())
        }),
        (unknown_layer, |value| {
            serde_json::from_value::<ImportPlanV1>(value).map(|_| ())
        }),
        (unknown_locator, |value| {
            serde_json::from_value::<ImportPlanV1>(value).map(|_| ())
        }),
        (unknown_provenance, |value| {
            serde_json::from_value::<ImportPlanV1>(value).map(|_| ())
        }),
        (unknown_reference, |value| {
            serde_json::from_value::<CheckpointCatalogRecordV1>(value).map(|_| ())
        }),
        (unknown_summary, |value| {
            serde_json::from_value::<CheckpointCatalogRecordV1>(value).map(|_| ())
        }),
        (unknown_record, |value| {
            serde_json::from_value::<CheckpointCatalogRecordV1>(value).map(|_| ())
        }),
        (unknown_nested_record, |value| {
            serde_json::from_value::<CheckpointInventoryV1>(value).map(|_| ())
        }),
        (unknown_inventory, |value| {
            serde_json::from_value::<CheckpointInventoryV1>(value).map(|_| ())
        }),
    ];
    for (value, parse) in strict_unknowns {
        assert!(
            !validator.is_valid(&value),
            "schema admitted unknown field: {value}"
        );
        assert!(
            parse(value.clone()).is_err(),
            "serde admitted unknown field: {value}"
        );
    }

    let import_plan = ImportPlanV1::new(
        "plan",
        "family",
        vec![
            ImportLayerV1 {
                layer_id: "a".to_owned(),
                role: "a-role".to_owned(),
                target_path: "a.safetensors".to_owned(),
                source: linked("root", "a.safetensors"),
            },
            ImportLayerV1 {
                layer_id: "b".to_owned(),
                role: "b-role".to_owned(),
                target_path: "b.safetensors".to_owned(),
                source: linked("root", "b.safetensors"),
            },
        ],
    )
    .unwrap();
    let mut unsorted_layers = serde_json::to_value(&import_plan).unwrap();
    unsorted_layers["layers"].as_array_mut().unwrap().reverse();
    assert!(validator.is_valid(&unsorted_layers));
    assert!(serde_json::from_value::<ImportPlanV1>(unsorted_layers).is_err());

    let mut keyed_duplicate_layers = serde_json::to_value(&import_plan).unwrap();
    let mut second = keyed_duplicate_layers["layers"][0].clone();
    second["role"] = json!("different-role");
    keyed_duplicate_layers["layers"] = json!([keyed_duplicate_layers["layers"][0].clone(), second]);
    assert!(validator.is_valid(&keyed_duplicate_layers));
    assert!(serde_json::from_value::<ImportPlanV1>(keyed_duplicate_layers).is_err());

    let mut count_mismatch = serde_json::to_value(import_plan.summary().unwrap()).unwrap();
    count_mismatch["layerCount"] = json!(1);
    assert!(validator.is_valid(&count_mismatch));
    assert!(serde_json::from_value::<ImportPlanSummaryV1>(count_mismatch).is_err());

    let first = CheckpointCatalogRecordV1::from_plan("a", &import_plan).unwrap();
    let mut second = first.clone();
    second.checkpoint_id = "b".to_owned();
    second.plan.plan_id = first.plan.plan_id.clone();
    let keyed_plan_duplicate = json!({"schemaVersion":1,"records":[
        serde_json::to_value(&first).unwrap(), serde_json::to_value(&second).unwrap()
    ]});
    assert!(validator.is_valid(&keyed_plan_duplicate));
    assert!(serde_json::from_value::<CheckpointInventoryV1>(keyed_plan_duplicate).is_err());

    let readme = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/schemas/README.md"),
    )
    .unwrap();
    for documented in [
        "keyed uniqueness",
        "sorted layers",
        "summary counts",
        "reference/summary digest agreement",
    ] {
        assert!(
            readme.contains(documented),
            "missing semantic-only rule: {documented}"
        );
    }
}

#[test]
fn constructors_and_every_publication_surface_reject_manual_invalid_state() {
    fn serialization_error<T: serde::Serialize>(value: &T) -> String {
        serde_json::to_string(value).unwrap_err().to_string()
    }

    for invalid_version in [0, 2] {
        let mut locator = linked("root", "model.safetensors");
        match &mut locator {
            SourceLocatorV1::Linked { schema_version, .. }
            | SourceLocatorV1::Managed { schema_version, .. } => *schema_version = invalid_version,
        }
        assert!(locator.validate().is_err());
        assert!(locator.canonical_json().is_err());
        assert!(locator.content_digest().is_err());
        assert!(locator.semantic_identity().is_err());
        assert!(locator.source_binding_identity().is_err());
        assert!(!serialization_error(&locator).is_empty());

        let mut managed_locator = managed("install", "model.safetensors");
        match &mut managed_locator {
            SourceLocatorV1::Linked { schema_version, .. }
            | SourceLocatorV1::Managed { schema_version, .. } => *schema_version = invalid_version,
        }
        assert!(managed_locator.validate().is_err());
        assert!(managed_locator.canonical_json().is_err());
        assert!(managed_locator.content_digest().is_err());
        assert!(managed_locator.semantic_identity().is_err());
        assert!(managed_locator.source_binding_identity().is_err());
        assert!(!serialization_error(&managed_locator).is_empty());

        let mut import_plan = plan(linked("root", "model.safetensors"));
        import_plan.schema_version = invalid_version;
        assert!(import_plan.validate().is_err());
        assert!(import_plan.canonical_json().is_err());
        assert!(import_plan.semantic_digest().is_err());
        assert!(import_plan.source_binding_identity().is_err());
        assert!(import_plan.plan_reference().is_err());
        assert!(import_plan.summary().is_err());
        assert!(!serialization_error(&import_plan).is_empty());

        let valid_plan = plan(linked("root", "model.safetensors"));
        let mut reference = valid_plan.plan_reference().unwrap();
        reference.schema_version = invalid_version;
        assert!(reference.validate().is_err());
        assert!(reference.canonical_json().is_err());
        assert!(!serialization_error(&reference).is_empty());

        let mut summary = valid_plan.summary().unwrap();
        summary.schema_version = invalid_version;
        assert!(summary.validate().is_err());
        assert!(summary.canonical_json().is_err());
        assert!(!serialization_error(&summary).is_empty());

        let valid_record = CheckpointCatalogRecordV1::from_plan("checkpoint", &valid_plan).unwrap();
        let mut record = valid_record.clone();
        record.schema_version = invalid_version;
        assert!(record.validate().is_err());
        assert!(record.canonical_json().is_err());
        assert!(!serialization_error(&record).is_empty());

        let mut nested_reference = valid_record.clone();
        nested_reference.plan.schema_version = invalid_version;
        assert!(nested_reference.validate().is_err());
        assert!(nested_reference.canonical_json().is_err());
        assert!(!serialization_error(&nested_reference).is_empty());
        let nested_reference_inventory = CheckpointInventoryV1 {
            schema_version: 1,
            records: vec![nested_reference],
        };
        assert!(nested_reference_inventory.validate().is_err());
        assert!(nested_reference_inventory.canonical_json().is_err());
        assert!(!serialization_error(&nested_reference_inventory).is_empty());

        let mut nested_summary = valid_record.clone();
        nested_summary.summary.schema_version = invalid_version;
        assert!(nested_summary.validate().is_err());
        assert!(nested_summary.canonical_json().is_err());
        assert!(!serialization_error(&nested_summary).is_empty());

        let mut inventory = CheckpointInventoryV1::new(vec![valid_record]).unwrap();
        inventory.schema_version = invalid_version;
        assert!(inventory.validate().is_err());
        assert!(inventory.canonical_json().is_err());
        assert!(!serialization_error(&inventory).is_empty());

        let nested_inventory = CheckpointInventoryV1 {
            schema_version: 1,
            records: vec![nested_summary],
        };
        assert!(nested_inventory.validate().is_err());
        assert!(nested_inventory.canonical_json().is_err());
        assert!(!serialization_error(&nested_inventory).is_empty());
    }

    let mut nested_locator = plan(linked("root", "model.safetensors"));
    match &mut nested_locator.layers[0].source {
        SourceLocatorV1::Linked { schema_version, .. }
        | SourceLocatorV1::Managed { schema_version, .. } => *schema_version = 2,
    }
    assert!(nested_locator.validate().is_err());
    assert!(nested_locator.canonical_json().is_err());
    assert!(nested_locator.semantic_digest().is_err());
    assert!(nested_locator.source_binding_identity().is_err());
    assert!(!serialization_error(&nested_locator).is_empty());

    let invalid_provenance = ManagedProvenanceV1 {
        source: " ".to_owned(),
        reference: None,
        ..ManagedProvenanceV1::default()
    };
    assert!(invalid_provenance.validate().is_err());
    assert!(!serialization_error(&invalid_provenance).is_empty());
    let invalid_layer = ImportLayerV1 {
        layer_id: " ".to_owned(),
        role: "role".to_owned(),
        target_path: "model.safetensors".to_owned(),
        source: linked("root", "model.safetensors"),
    };
    assert!(invalid_layer.validate().is_err());
    assert!(!serialization_error(&invalid_layer).is_empty());
}

#[test]
fn deterministic_publication_round_trips_and_identity_domains_cover_manual_values() {
    let plan_a = ImportPlanV1::new(
        "plan-a",
        "family",
        vec![ImportLayerV1 {
            layer_id: "layer".to_owned(),
            role: "role".to_owned(),
            target_path: "model.safetensors".to_owned(),
            source: linked("root-a", "model.safetensors"),
        }],
    )
    .unwrap();
    let plan_b = ImportPlanV1::new(
        "plan-b",
        "family",
        vec![ImportLayerV1 {
            layer_id: "layer".to_owned(),
            role: "role".to_owned(),
            target_path: "model.safetensors".to_owned(),
            source: linked("root-b", "model.safetensors"),
        }],
    )
    .unwrap();
    let record_a = CheckpointCatalogRecordV1::from_plan("a", &plan_a).unwrap();
    let record_b = CheckpointCatalogRecordV1::from_plan("b", &plan_b).unwrap();
    let manually_reversed = CheckpointInventoryV1 {
        schema_version: 1,
        records: vec![record_b.clone(), record_a.clone()],
    };
    manually_reversed.validate().unwrap();
    let ordinary = serde_json::to_string(&manually_reversed).unwrap();
    let canonical = manually_reversed.canonical_json().unwrap();
    assert_eq!(ordinary, canonical);
    assert!(
        ordinary.find(r#""checkpointId":"a""#).unwrap()
            < ordinary.find(r#""checkpointId":"b""#).unwrap()
    );

    let inventory = CheckpointInventoryV1::new(vec![record_b, record_a]).unwrap();
    let valid_documents = [
        linked("root", "model.safetensors")
            .canonical_json()
            .unwrap(),
        plan_a.canonical_json().unwrap(),
        plan_a.plan_reference().unwrap().canonical_json().unwrap(),
        plan_a.summary().unwrap().canonical_json().unwrap(),
        CheckpointCatalogRecordV1::from_plan("checkpoint", &plan_a)
            .unwrap()
            .canonical_json()
            .unwrap(),
        inventory.canonical_json().unwrap(),
    ];
    let schema_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/schemas/checkpoint-import.schema.json");
    let schema: Value = serde_json::from_str(&fs::read_to_string(schema_path).unwrap()).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    for (surface, document) in [
        VersionSurface::Locator,
        VersionSurface::Plan,
        VersionSurface::Reference,
        VersionSurface::Summary,
        VersionSurface::Record,
        VersionSurface::Inventory,
    ]
    .into_iter()
    .zip(valid_documents)
    {
        let parsed_value: Value = serde_json::from_str(&document).unwrap();
        assert!(validator.is_valid(&parsed_value), "{surface:?}: {document}");
        let round_trip = match surface {
            VersionSurface::Locator => serde_json::from_str::<SourceLocatorV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
            VersionSurface::Plan => serde_json::from_str::<ImportPlanV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
            VersionSurface::Reference => serde_json::from_str::<ImportPlanReferenceV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
            VersionSurface::Summary => serde_json::from_str::<ImportPlanSummaryV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
            VersionSurface::Record => serde_json::from_str::<CheckpointCatalogRecordV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
            VersionSurface::Inventory => serde_json::from_str::<CheckpointInventoryV1>(&document)
                .unwrap()
                .canonical_json()
                .unwrap(),
        };
        assert_eq!(round_trip, document, "{surface:?}");
    }

    let locator_semantic = plan_a.layers[0].source.semantic_identity().unwrap();
    let locator_binding = plan_a.layers[0].source.source_binding_identity().unwrap();
    let plan_semantic = plan_a.semantic_digest().unwrap();
    let plan_binding = plan_a.source_binding_identity().unwrap();
    let domains = [
        locator_semantic,
        locator_binding,
        plan_semantic,
        plan_binding,
    ];
    assert_eq!(
        domains
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4
    );

    let rebound = ImportPlanV1::new(
        plan_a.plan_id.clone(),
        plan_a.family.clone(),
        vec![ImportLayerV1 {
            source: managed("install", "different/model.safetensors"),
            ..plan_a.layers[0].clone()
        }],
    )
    .unwrap();
    assert_eq!(
        plan_a.semantic_digest().unwrap(),
        rebound.semantic_digest().unwrap()
    );
    assert_ne!(
        plan_a.source_binding_identity().unwrap(),
        rebound.source_binding_identity().unwrap()
    );
    let old_record = CheckpointCatalogRecordV1::from_plan("checkpoint", &plan_a).unwrap();
    assert!(old_record.validate_loaded_plan(&rebound).is_err());

    let managed_a = plan(managed("install", "model.safetensors"));
    let mut managed_b = managed_a.clone();
    if let SourceLocatorV1::Managed { provenance, .. } = &mut managed_b.layers[0].source {
        provenance.reference = Some("different/reference".to_owned());
    }
    assert_eq!(
        managed_a.semantic_digest().unwrap(),
        managed_b.semantic_digest().unwrap()
    );
    assert_ne!(
        managed_a.source_binding_identity().unwrap(),
        managed_b.source_binding_identity().unwrap()
    );

    let mut changed_content = plan_a.clone();
    match &mut changed_content.layers[0].source {
        SourceLocatorV1::Linked { fingerprint, .. } => *fingerprint = "f".repeat(64),
        SourceLocatorV1::Managed { sha256, .. } => *sha256 = "f".repeat(64),
    }
    assert_ne!(
        plan_a.semantic_digest().unwrap(),
        changed_content.semantic_digest().unwrap()
    );
    assert_ne!(
        plan_a.source_binding_identity().unwrap(),
        changed_content.source_binding_identity().unwrap()
    );
    let mut changed_logical = plan_a.clone();
    changed_logical.layers[0].role = "different".to_owned();
    assert_ne!(
        plan_a.semantic_digest().unwrap(),
        changed_logical.semantic_digest().unwrap()
    );
    assert_ne!(
        plan_a.source_binding_identity().unwrap(),
        changed_logical.source_binding_identity().unwrap()
    );

    let composed = SourceLocatorV1::linked("café", "model.safetensors", DIGEST).unwrap();
    let decomposed = SourceLocatorV1::linked("café", "model.safetensors", DIGEST).unwrap();
    assert_ne!(
        composed.canonical_json().unwrap(),
        decomposed.canonical_json().unwrap()
    );
    assert_ne!(
        composed.source_binding_identity().unwrap(),
        decomposed.source_binding_identity().unwrap()
    );
}
