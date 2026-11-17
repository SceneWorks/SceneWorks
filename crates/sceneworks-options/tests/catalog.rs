use sceneworks_options::{
    by_id, by_wire, registry, DefaultRule, IntentClass, OptionContext, Predicate,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[test]
fn declarations_have_unambiguous_ids_and_wire_keys() {
    let mut ids = BTreeSet::new();
    let mut wires = BTreeSet::new();
    for spec in registry() {
        assert!(ids.insert(spec.id), "duplicate id {}", spec.id);
        assert!(!spec.id.is_empty() && !spec.key.is_empty() && !spec.binding.is_empty());
        assert!(!spec.ui.label.is_empty() && !spec.ui.renderer.is_empty());
        for name in std::iter::once(spec.key).chain(spec.aliases.iter().copied()) {
            assert!(
                wires.insert((spec.location, name)),
                "duplicate wire {}.{name}",
                spec.location
            );
            assert_eq!(by_wire(spec.location, name).unwrap().id, spec.id);
        }
        if let DefaultRule::Literal(literal) = spec.default {
            let parsed = serde_json::from_str::<Value>(literal);
            assert!(parsed.is_ok(), "{} has non-JSON literal {literal}", spec.id);
        }
        assert_eq!(by_id(spec.id), Some(spec));
    }
}

#[test]
fn metadata_is_serializable_and_preserves_alias_omission_and_context() {
    let serialized = serde_json::to_value(registry().collect::<Vec<_>>()).unwrap();
    assert!(serialized.is_array());
    let shift = by_wire("advanced", "timestepShift").unwrap();
    assert_eq!(shift.id, "advanced.schedulerShift");
    assert_eq!(shift.read(&json!({})), None);
    let explicit_null = json!({"schedulerShift":null, "timestepShift":3.0});
    assert_eq!(
        shift.read(&explicit_null),
        Some(("schedulerShift", &Value::Null))
    );
    let explicit_default = json!({"timestepShift":3});
    assert_eq!(
        shift.read(&explicit_default),
        Some(("timestepShift", &json!(3)))
    );
    let stamp = by_wire("advanced", "__portableFrozenSources").unwrap();
    assert_eq!(stamp.intent, IntentClass::Context);
    assert!(!stamp.user_settable);
    assert_eq!(
        by_wire("film.model.advanced", "referenceImageShortEdge")
            .unwrap()
            .id,
        "advanced.referenceImageShortEdge"
    );
}

#[test]
fn capability_predicates_distinguish_absence_false_and_selected_route() {
    let model = json!({"type":"image","family":"moss_image","capabilities":["edit_image"],"image":{"supportsCfgInterval":false}});
    let features = json!({"id":"moss_image","supports_lora":true,"samplers":["euler"]});
    let target =
        json!({"id":"moss_image_lora","kernel":"moss_lora","limits":{"supportsWeightNoise":false}});
    let context = OptionContext {
        operation: "text_to_image",
        model: &model,
        features: &features,
        target: &target,
    };
    assert!(Predicate::Capability("edit_image").matches(&context));
    assert!(Predicate::Feature("supports_lora").matches(&context));
    assert!(Predicate::Feature("samplers").matches(&context));
    assert!(!Predicate::Feature("unknown").matches(&context));
    assert!(!Predicate::ModelField("image.supportsCfgInterval").matches(&context));
    assert!(!Predicate::TrainingTarget("limits.supportsWeightNoise").matches(&context));
    assert!(Predicate::TrainingKernel("moss_lora").matches(&context));
    assert!(Predicate::TrainingId("moss_image_lora").matches(&context));
}

#[test]
fn caption_shapes_and_content_intents_remain_explicit() {
    use sceneworks_options::Shape;
    let root = "advanced.structuredPrompt.caption";
    assert_eq!(
        by_wire(root, "high_level_description").unwrap().shape,
        Shape::Text
    );
    assert!(by_wire(root, "compositional_deconstruction").is_some());
    let style = "advanced.structuredPrompt.caption.style_description";
    let element = "advanced.structuredPrompt.caption.compositional_deconstruction.elements[]";
    for (location, maximum) in [(style, "16"), (element, "5")] {
        let palette = by_wire(location, "color_palette").unwrap();
        assert_eq!(palette.shape, Shape::Strings);
        assert!(palette
            .limits
            .iter()
            .any(|limit| limit.contains(maximum) && limit.contains("^#[0-9A-F]{6}$")));
    }
    let bbox = by_wire(element, "bbox").unwrap();
    assert!(bbox.limits.iter().any(|limit| limit.contains("1000")
        && limit.contains("four integers")
        && limit.contains("ymin <= ymax")));
    for (key, other) in [("photo", "art_style"), ("art_style", "photo")] {
        assert!(by_wire(style, key)
            .unwrap()
            .effects
            .iter()
            .any(|effect| effect.action == "exclude" && effect.target.ends_with(other)));
    }
    for id in [
        "characterTest.lookId",
        "promptRefine.modelId",
        "promptRefine.workflow",
        "image.characterId",
        "image.characterLookId",
        "video.characterId",
        "video.characterLookId",
        "video.personTrackId",
    ] {
        assert_eq!(by_id(id).unwrap().intent, IntentClass::WhatToMake, "{id}");
    }
}

#[test]
fn yue2_job_kind_predicates_use_request_context() {
    let model = json!({"family":"yue2", "type":"audio"});
    let context = OptionContext {
        operation: "plan",
        model: &model,
        features: &Value::Null,
        target: &Value::Null,
    };
    assert!(by_id("yue2.seed").unwrap().applies_when.matches(&context));
    assert!(!by_id("yue2.steps").unwrap().applies_when.matches(&context));
    let rendering = OptionContext {
        operation: "renderVersion",
        ..context
    };
    assert!(!by_id("yue2.seed").unwrap().applies_when.matches(&rendering));
    assert!(by_id("yue2.steps")
        .unwrap()
        .applies_when
        .matches(&rendering));
}

#[test]
fn timeline_context_fields_are_typed_context() {
    use sceneworks_options::Shape;
    for (key, shape) in [
        ("action", Shape::OneOf("extend | replace | bridge")),
        ("timelineId", Shape::Text),
        ("timelineName", Shape::Text),
        ("itemId", Shape::Text),
        ("trackId", Shape::Text),
        ("sourceAssetId", Shape::Text),
        ("sourceTimestamp", Shape::Number),
        ("timelineStart", Shape::Number),
        ("timelineEnd", Shape::Number),
    ] {
        let spec = by_wire("advanced.timelineContext", key).unwrap();
        assert_eq!(spec.shape, shape);
        assert_eq!(spec.intent, IntentClass::Context);
        assert_eq!(spec.default, DefaultRule::Absent);
    }
}
