use sceneworks_options::upgrade::{metadata, upgrade, Carrier};
use serde_json::{json, Value};

fn carrier(name: &str) -> Carrier {
    match name {
        "request" => Carrier::Request,
        "job" => Carrier::Job,
        "recipe" => Carrier::Recipe,
        "studioSettings" => Carrier::StudioSettings,
        "simpleSettings" => Carrier::SimpleSettings,
        "filmPlan" => Carrier::FilmPlan,
        "trainingPlan" => Carrier::TrainingPlan,
        _ => panic!("unknown fixture carrier {name}"),
    }
}
#[test]
fn historical_outputs_preserve_all_values_and_omissions() {
    let corpus: Value =
        serde_json::from_str(include_str!("fixtures/historical-builders.json")).unwrap();
    for row in corpus["rows"].as_array().unwrap() {
        let before = row["value"].clone();
        let kind = carrier(row["carrier"].as_str().unwrap());
        let after = upgrade(&before, kind).unwrap_or_else(|e| panic!("{}: {e}", row["name"]));
        assert_eq!(
            after, before,
            "{} changed authored/omitted values",
            row["name"]
        );
        assert_eq!(upgrade(&after, kind).unwrap(), after);
    }
}
#[test]
fn aliases_are_canonicalized_without_erasing_originals_or_stamps() {
    let before = json!({"payload":{"ip_adapter_scale":0,"advanced":{"timestepShift":"3","memoryStrategy":{"rung":2},"moss_image_execution_report":{"provider":"moss"}}},"provenance":{"repo":"kept","modelRepo":"also kept"},"telemetry":{"seconds":0}});
    let after = upgrade(&before, Carrier::Job).unwrap();
    assert_eq!(after["payload"]["ipAdapterScale"], 0);
    assert_eq!(after["payload"]["advanced"]["schedulerShift"], "3");
    assert_eq!(after["payload"]["advanced"]["timestepShift"], "3");
    assert_eq!(
        after["payload"]["advanced"]["memoryStrategy"],
        before["payload"]["advanced"]["memoryStrategy"]
    );
    assert_eq!(after["telemetry"], before["telemetry"]);
    assert_eq!(after["provenance"], before["provenance"]);
    assert!(after["payload"].get("seed").is_none());
}
#[test]
fn refusals_are_typed_and_leave_originals_untouched() {
    for (before, kind, code) in [
        (
            json!({"advanced":{"schedulerShift":3,"timestepShift":4}}),
            Carrier::Request,
            "ConflictingAliases",
        ),
        (
            json!({"optionsSchemaVersion":2}),
            Carrier::StudioSettings,
            "UnsupportedSchemaVersion",
        ),
        (
            json!({"schemaVersion":4}),
            Carrier::FilmPlan,
            "UnsupportedSchemaVersion",
        ),
        (
            json!({"schemaVersion":2}),
            Carrier::TrainingPlan,
            "UnsupportedSchemaVersion",
        ),
        (
            json!({"schemaVersion":3}),
            Carrier::ShareEnvelope,
            "UnsupportedSchemaVersion",
        ),
        (
            json!({"optionsSchemaVersion":null}),
            Carrier::Request,
            "InvalidSchemaVersion",
        ),
        (
            json!({"plan":{"schemaVersion":1,"planVersion":2,"target":{},"dataset":{}}}),
            Carrier::Request,
            "UnsupportedSchemaVersion",
        ),
    ] {
        let original = before.clone();
        assert_eq!(upgrade(&before, kind).unwrap_err().code, code);
        assert_eq!(before, original);
    }
}
#[test]
fn compiled_plan_revision_is_not_a_training_kernel_version() {
    assert!(upgrade(
        &json!({"schemaVersion":7,"planVersion":21}),
        Carrier::CompiledFilmPlan
    )
    .is_ok());
}
#[test]
fn browser_metadata_is_generated_from_the_native_registry() {
    let browser: Value = serde_json::from_str(include_str!(
        "../../../apps/web/src/optionUpgradeMetadata.json"
    ))
    .unwrap();
    assert_eq!(browser,metadata(),"cargo run -p sceneworks-options --example upgrade_metadata > apps/web/src/optionUpgradeMetadata.json");
}

#[test]
fn fresh_generation_strips_machine_decisions_but_retains_source_identity() {
    use sceneworks_options::upgrade::strip_machine_decisions;
    let original = json!({"seed":123,"expectedImportedModelRef":{"checkpointId":"old-source"},"advanced":{"memoryStrategy":{"rung":1},"memoryEvidenceRevision":2,"moss_image_execution_report":{"seed":123},"steps":9,"sampler":"euler"}});
    let mut fresh = original.as_object().unwrap().clone();
    strip_machine_decisions(&mut fresh);
    assert_eq!(fresh["advanced"], json!({"steps":9,"sampler":"euler"}));
    assert_eq!(
        fresh["expectedImportedModelRef"],
        original["expectedImportedModelRef"]
    );
    assert!(original["advanced"].get("memoryStrategy").is_some());
}

#[test]
fn nested_recipe_versions_refuse_before_any_consumer_can_ignore_them() {
    let input = json!({"rawAdapterSettings":{"lowResStart":{"version":9,"k":4,"scale":0.5}}});
    assert_eq!(
        upgrade(&input, Carrier::Recipe).unwrap_err().code,
        "UnsupportedSchemaVersion"
    );
}

#[test]
fn top_level_alias_retains_authored_value_and_original_spelling() {
    let before = json!({"ip_adapter_scale":0.25});
    let after = upgrade(&before, Carrier::Request).unwrap();
    assert_eq!(
        after,
        json!({"ipAdapterScale":0.25,"ip_adapter_scale":0.25})
    );
    assert_eq!(before, json!({"ip_adapter_scale":0.25}));
}
