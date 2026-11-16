use sceneworks_core::{
    options::{by_wire, registry, IntentClass},
    training::TECHNIQUE_KEYS,
    workflow_share::ADVANCED_KEY_RULES,
};

#[test]
fn shared_advanced_training_and_film_options_all_have_declarations() {
    for rule in ADVANCED_KEY_RULES {
        assert!(
            by_wire("advanced", rule.key).is_some(),
            "unregistered share key {}",
            rule.key
        );
    }
    for key in TECHNIQUE_KEYS {
        assert!(
            by_wire("training.config.advanced", key).is_some(),
            "unregistered technique {key}"
        );
    }
    for key in [
        "referenceImageShortEdge",
        "referenceImageResizeMode",
        "steps",
    ] {
        assert!(
            by_wire("film.model.advanced", key).is_some(),
            "unregistered film option {key}"
        );
    }
}

#[test]
fn portable_worker_stamps_are_context_and_never_user_options() {
    for option in registry().filter(|option| option.key.starts_with("__portable")) {
        assert_eq!(option.intent, IntentClass::Context);
        assert!(!option.user_settable);
    }
    for key in [
        "__portableAuxPaths",
        "__portableBoundTier",
        "__portableFrozenSources",
        "__portableGemmaFiles",
    ] {
        assert!(by_wire("advanced", key).is_some());
    }
}

#[test]
fn option_applicability_reads_selected_model_and_backend_declarations() {
    use sceneworks_core::{options_context, registry::builtin_model_spec};
    let spec = builtin_model_spec("ltx_2_5").unwrap();
    let mlx = options_context::for_model(spec, "mlx", "text_to_video", Some("ltx_2_5"));
    let candle =
        options_context::for_model(spec, "candle", "text_to_video", Some("ltx_2_5_distilled"));
    let option = by_wire("advanced", "autoDuration").unwrap();
    assert!(option.applies_when.matches(&mlx.context()));
    assert!(option.applies_when.matches(&candle.context()));
    let absent = options_context::for_model(spec, "unavailable", "text_to_video", None);
    assert!(!option.applies_when.matches(&absent.context()));
    let image = builtin_model_spec("flux_dev").unwrap();
    let image_facts = options_context::for_model(image, "mlx", "text_to_image", None);
    assert!(!option.applies_when.matches(&image_facts.context()));
    assert!(by_wire("advanced", "usePid")
        .unwrap()
        .applies_when
        .matches(&image_facts.context()));
}

#[test]
fn guidance_opt_out_and_phase_compatibility_preserve_existing_controls() {
    use sceneworks_core::{options_context, registry::builtin_model_spec};
    let guidance = by_wire("advanced", "guidanceScale").unwrap();
    for id in ["krea_2_raw", "instantid_realvisxl"] {
        let spec = builtin_model_spec(id).unwrap();
        let mut facts = options_context::for_model(spec, "mlx", "text_to_image", None);
        assert!(guidance.applies_when.matches(&facts.context()), "{id}");
        facts.model["image"]["supportsGuidance"] = serde_json::json!(false);
        assert!(
            !guidance.applies_when.matches(&facts.context()),
            "{id} opt-out"
        );
    }
    let phases = by_wire("advanced", "phases").unwrap();
    for (id, expected) in [("krea_2_raw", true), ("krea_2_turbo", false)] {
        let spec = builtin_model_spec(id).unwrap();
        let facts = options_context::for_model(spec, "mlx", "text_to_image", None);
        assert_eq!(
            phases.applies_when.matches(&facts.context()),
            expected,
            "{id}"
        );
    }
    for target in [
        "advanced.steps",
        "advanced.guidanceScale",
        "advanced.usePid",
        "image.hiresFix",
        "image.sourceAssetId",
        "image.referenceAssetIds",
        "advanced.poses",
    ] {
        assert!(
            phases.effects.iter().any(|effect| effect.target == target),
            "{target}"
        );
    }
}

#[test]
fn image_capabilities_enable_vqa_interleave_and_inpaint_requests() {
    use sceneworks_core::{options_context, registry::builtin_model_spec};
    let sense = options_context::for_model(
        builtin_model_spec("sensenova_u1_8b").unwrap(),
        "mlx",
        "vqa",
        None,
    );
    assert_eq!(sense.model["type"], "image");
    for option in registry().filter(|option| matches!(option.location, "vqa" | "interleave")) {
        assert!(
            option.applies_when.matches(&sense.context()),
            "{}",
            option.id
        );
    }
    let sdxl = options_context::for_model(
        builtin_model_spec("sdxl").unwrap(),
        "mlx",
        "edit_image",
        None,
    );
    assert!(by_wire("image", "maskAssetId")
        .unwrap()
        .applies_when
        .matches(&sdxl.context()));
    assert!(!by_wire("image", "maskAssetId")
        .unwrap()
        .applies_when
        .matches(&sense.context()));
    assert!(!by_wire("vqa", "question")
        .unwrap()
        .applies_when
        .matches(&sdxl.context()));
}

#[test]
fn hires_fix_respects_checkpoint_opt_out_for_parent_and_children() {
    use sceneworks_core::{options_context, registry::builtin_model_spec};
    for (id, supported) in [("realvisxl_lightning", false), ("realvisxl", true)] {
        let facts = options_context::for_model(
            builtin_model_spec(id).unwrap(),
            "mlx",
            "text_to_image",
            None,
        );
        if supported {
            assert!(facts.model["image"].get("supportsHiresFix").is_none());
        } else {
            assert_eq!(facts.model["image"]["supportsHiresFix"], false);
        }
        for option in registry()
            .filter(|option| option.id == "image.hiresFix" || option.location == "image.hiresFix")
        {
            assert_eq!(
                option.applies_when.matches(&facts.context()),
                supported,
                "{id}: {}",
                option.id
            );
        }
    }
}
