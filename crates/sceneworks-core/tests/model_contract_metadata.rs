//! Pure metadata/filesystem regressions for SW-F327 F-061/064/065/066.
use sceneworks_core::{builtin_manifests, film_plan, lora_family, model_family};
use serde_json::json;

#[test]
fn film_and_enqueue_use_the_same_canonical_one_directional_family_gate() {
    for row in builtin_manifests::builtin_model_entries() {
        let Some(model) = row.as_object() else {
            continue;
        };
        let id = model["id"].as_str().unwrap();
        let accepted = lora_family::accepted_model_lora_families(model);
        for family in [
            "flux",
            "flux2",
            "wan-video",
            "krea_2",
            "krea-2",
            "minimax-h3",
            "qwen-image",
            "qwen-image-2-1",
        ] {
            let lora = film_plan::PlanLoraEntry {
                id: "fixture".to_owned(),
                name: "fixture".to_owned(),
                family: family.to_owned(),
                default_weight: 1.0,
                model_ids: Vec::new(),
            };
            assert_eq!(
                film_plan::lora_family_matches_model(&lora, id),
                accepted.contains(&model_family::canonical(family)),
                "{id}/{family}"
            );
        }
    }
    assert!(lora_family::accepted_lora_families("chroma").contains(&"flux".to_owned()));
    assert!(!lora_family::accepted_lora_families("flux").contains(&"chroma".to_owned()));
}

#[test]
fn manifest_gates_preserve_every_existing_generation_and_size_pair() {
    let ids = [
        "ltx_2_3",
        "ltx_2_3_eros",
        "ltx_2_5",
        "wan_2_2",
        "wan_2_2_t2v_14b",
        "wan_2_2_i2v_14b",
        "wan_2_2_vace_fun_14b",
        "scail2_14b",
        "krea_realtime_14b",
    ];
    for model in ids {
        let row = builtin_manifests::builtin_model_entry(model).unwrap();
        let family = row["family"].as_str().unwrap();
        for base in ids {
            let old = base == model
                || match family {
                    "ltx-video" => {
                        ["ltx_2_3", "ltx_2_3_eros"].contains(&model)
                            && ["ltx_2_3", "ltx_2_3_eros"].contains(&base)
                    }
                    "scail2" | "krea-realtime" => {
                        model.ends_with("_14b")
                            && base.ends_with("_14b")
                            && (family == "scail2" || !base.split('_').any(|part| part == "i2v"))
                    }
                    _ => false,
                };
            assert_eq!(
                lora_family::base_model_satisfies_gate(family, model, base),
                old,
                "{model}/{base}"
            );
        }
    }
    assert!(lora_family::base_model_satisfies_gate(
        "krea-realtime",
        "custom_14b",
        "custom_t2v_14b"
    ));
    assert!(!lora_family::base_model_satisfies_gate(
        "krea-realtime",
        "custom_14b",
        "custom_i2v_14b"
    ));
}

#[test]
fn supplied_custom_generations_override_ids_and_keep_sibling_generations_separate() {
    let host = json!({"family":"ltx-video", "loraCompatibility":{"generation":"2.3"}});
    let base = json!({"family":"ltx-video", "loraCompatibility":{"generation":"2.3"}});
    let different = json!({"family":"ltx-video", "loraCompatibility":{"generation":"2.5"}});
    assert!(lora_family::base_model_satisfies_gate_with_manifest(
        "ltx-video",
        "custom_host",
        "custom_base",
        host.as_object(),
        base.as_object()
    ));
    assert!(!lora_family::base_model_satisfies_gate_with_manifest(
        "ltx-video",
        "custom_host",
        "custom_base",
        host.as_object(),
        different.as_object()
    ));
    assert!(lora_family::base_model_satisfies_gate_with_manifest(
        "ltx-video",
        "ltx_2_3",
        "ltx_2_3_eros",
        json!({}).as_object(),
        None
    ));
    let legacy = json!({"id":"old_custom", "family":"ltx-video", "baseModel":"ltx_2_3"});
    assert!(lora_family::base_model_satisfies_gate_with_manifest(
        "ltx-video",
        "old_custom",
        "ltx_2_3_eros",
        legacy.as_object(),
        None
    ));
}

#[test]
fn namespace_aliases_keep_generation_and_family_specific_evidence() {
    for (alias, family) in [
        ("krea2", "krea_2"),
        ("KREA-2", "krea_2"),
        ("QWEN_IMAGE_2_1", "qwen-image-2-1"),
        ("moss_image", "moss-image"),
    ] {
        let header = json!({"__metadata__":{"family":alias}});
        assert_eq!(
            lora_family::detect_lora_family(&header).as_deref(),
            Some(family)
        );
        assert_eq!(model_family::canonical(alias), family);
    }
    for (class, family) in [
        ("FluxPipeline", "flux"),
        ("ChromaPipeline", "chroma"),
        ("WanI2VPipeline", "wan-video"),
        ("LTXVideoPipeline", "ltx-video"),
        ("StableDiffusionXLPipeline", "sdxl"),
    ] {
        assert_eq!(
            lora_family::diffusers_class_name_to_family(class).as_deref(),
            Some(family)
        );
    }
    let qwen = json!({"__metadata__":{"family":"qwen-image"}});
    let qwen_21 = json!({"__metadata__":{"family":"qwen-image-2-1"}});
    assert_ne!(
        lora_family::detect_lora_family(&qwen),
        lora_family::detect_lora_family(&qwen_21)
    );
}

#[test]
fn film_known_empty_support_is_refused_while_unknown_catalog_ids_remain_permissive() {
    let lora = film_plan::PlanLoraEntry {
        id: "fixture".into(),
        name: "fixture".into(),
        family: "qwen-image".into(),
        model_ids: Vec::new(),
        default_weight: 1.0,
    };
    assert!(!film_plan::lora_family_matches_model(
        &lora,
        "qwen_image_2_1_pe_t2i"
    ));
    assert!(!film_plan::lora_family_matches_model(&lora, "real_esrgan"));
    assert!(film_plan::lora_family_matches_model(
        &lora,
        "user_custom_model"
    ));
}

#[test]
fn explicit_family_metadata_preserves_identity_without_manufacturing_loader_support() {
    use sceneworks_core::base_weights::{
        import_supported, BaseWeightVerdict, ComponentRole, QuantFormat,
    };
    for family in ["qwen-image-2-1", "krea-realtime"] {
        for key in [
            "baseModel",
            "ss_base_model_version",
            "modelspec.architecture",
        ] {
            let header = json!({"__metadata__":{(key):family}});
            assert_eq!(
                lora_family::detect_lora_family(&header).as_deref(),
                Some(family),
                "{key}/{family}"
            );
        }
        let adapter = json!({"id":"fixture", "family":family});
        assert!(lora_family::validate_lora_compatibility(
            std::slice::from_ref(&adapter),
            Some(family),
            "fixture",
            None
        )
        .is_ok());
        let wrong_host = if family == "qwen-image-2-1" {
            "qwen-image"
        } else {
            "wan-video"
        };
        assert!(lora_family::validate_lora_compatibility(
            &[adapter],
            Some(wrong_host),
            "fixture",
            None
        )
        .is_err());
        let verdict = BaseWeightVerdict {
            family: Some(family.to_owned()),
            component: ComponentRole::Transformer,
            quant: QuantFormat::Bf16,
        };
        assert!(
            import_supported(&verdict).is_err(),
            "identity recognition must not authorize a new {family} loader"
        );
    }
    assert!(lora_family::validate_lora_compatibility(
        &[json!({"family":"wan-video", "baseModel":"wan_2_2_t2v_14b"})],
        Some("krea-realtime"),
        "krea_realtime",
        Some("krea_realtime_14b")
    )
    .is_ok());
}

#[test]
fn artifact_identity_stores_the_parser_normalized_repository() {
    let identity = sceneworks_core::model_artifacts::ArtifactIdentity::pinned(
        " Owner/_Model.git ",
        "0123456789abcdef0123456789abcdef01234567",
        "default",
    )
    .unwrap();
    assert_eq!(identity.repository, "Owner/_Model.git");
    assert!(identity.fingerprint.starts_with("Owner/_Model.git@"));
}

#[test]
fn exact_lora_metadata_outweighs_earlier_broad_repository_heuristics() {
    let header = json!({"__metadata__":{"baseModel":"community/Krea-style-adapter", "modelspec.architecture":"krea-realtime"}});
    assert_eq!(
        lora_family::detect_lora_family(&header).as_deref(),
        Some("krea-realtime")
    );
}
