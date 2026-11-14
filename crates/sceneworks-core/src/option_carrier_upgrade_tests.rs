// The archived production module resolves its original crate-owned imports through these exports.
#[allow(dead_code)]
#[path = "../tests/fixtures/sw7037/workflow_share_v1.rs"]
mod old_reader;
use crate::workflow_share::{build_workflow_share_from, parse_workflow_share, WorkflowAssetFacts};
use serde_json::{json, Value};

fn share(advanced: Value) -> Value {
    let facts = WorkflowAssetFacts {
        mode: "text_to_image".into(),
        model: "z_image_turbo".into(),
        prompt: "fox".into(),
        negative_prompt: String::new(),
        seed: 7,
        width: Some(1024),
        height: Some(1024),
    };
    serde_json::to_value(build_workflow_share_from(
        &facts,
        json!({"advanced":advanced}).as_object().unwrap(),
    ))
    .unwrap()
}
#[test]
fn released_reader_accepts_new_ordinary_shares_but_refuses_active_v2() {
    let ordinary = share(json!({"steps":9,"sampler":"euler"}));
    assert_eq!(ordinary["schemaVersion"], 1);
    assert!(old_reader::parse_workflow_share(&ordinary).is_ok());
    assert!(parse_workflow_share(&ordinary).is_ok());
    let active = share(json!({"lowResStart":{"version":1,"k":4,"scale":0.5}}));
    assert_eq!(active["schemaVersion"], 2);
    assert_eq!(
        active["advanced"]["lowResStart"],
        json!({"version":1,"k":4,"scale":0.5})
    );
    assert!(matches!(
        old_reader::parse_workflow_share(&active),
        Err(old_reader::WorkflowShareError::UnsupportedSchemaVersion {
            file: 2,
            supported: 1
        })
    ));
    assert!(parse_workflow_share(&active).is_ok());
    // Demonstrate why the outer gate is necessary with the real old reader.
    let mut falsely_v1 = active.clone();
    falsely_v1["schemaVersion"] = json!(1);
    assert!(!old_reader::parse_workflow_share(&falsely_v1)
        .unwrap()
        .advanced
        .contains_key("lowResStart"));
    assert!(parse_workflow_share(&falsely_v1).is_err());
}
#[test]
fn future_nested_identity_refuses_and_does_not_modify_input() {
    let mut active = share(json!({"lowResStart":{"version":1,"k":4,"scale":0.5}}));
    active["advanced"]["lowResStart"]["version"] = json!(9);
    let original = active.clone();
    assert!(parse_workflow_share(&active).is_err());
    assert_eq!(active, original);
}
#[test]
fn disabled_low_res_keeps_legacy_envelope() {
    for disabled in [Value::Null, json!(false)] {
        let value = share(json!({"lowResStart":disabled}));
        assert_eq!(value["schemaVersion"], 1);
        assert!(old_reader::parse_workflow_share(&value).is_ok());
    }
}

#[test]
fn reading_a_future_recipe_refuses_without_rewriting_the_sidecar() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../sceneworks-options/tests/fixtures/historical-builders.json"
    ))
    .unwrap();
    let mut recipe = corpus["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "historical-recipe-sidecar")
        .unwrap()["value"]
        .clone();
    recipe["optionsSchemaVersion"] = json!(9);
    let original = serde_json::to_string_pretty(&recipe).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("recipe.json");
    std::fs::write(&file, &original).unwrap();
    assert!(matches!(
        crate::store_util::read_json(&file),
        Err(crate::project_store::ProjectStoreError::FieldInvalid {
            code: "UnsupportedSchemaVersion",
            ..
        })
    ));
    assert_eq!(std::fs::read_to_string(file).unwrap(), original);
}

#[test]
fn indexed_history_stays_visible_with_a_typed_upgrade_refusal() {
    let asset = json!({"id":"historic","recipe":{"optionsSchemaVersion":9,"prompt":"keep me"},"provenance":{"checkpoint":"legacy"}});
    let mut cache = crate::asset_index::GenerationSetCache::default();
    let visible = crate::asset_index::hydrate_indexed_asset_cached(
        "project",
        std::path::Path::new("unused"),
        asset.clone(),
        &mut cache,
    )
    .unwrap();
    assert_eq!(visible["recipe"], asset["recipe"]);
    assert_eq!(visible["provenance"], asset["provenance"]);
    assert_eq!(
        visible["optionsUpgradeError"]["code"],
        "UnsupportedSchemaVersion"
    );
}

#[test]
fn an_unknown_v2_recipe_cannot_be_silently_dropped() {
    let mut envelope = share(json!({"lowResStart":{"version":1,"k":4,"scale":0.5}}));
    envelope["advanced"]["futureRecipe"] = json!({"version":1});
    let original = envelope.clone();
    assert!(parse_workflow_share(&envelope).is_err());
    assert_eq!(envelope, original);
}
