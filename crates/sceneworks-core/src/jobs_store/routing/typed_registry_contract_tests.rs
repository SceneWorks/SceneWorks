//! Cross-check typed native video mappings against the actual claim predicate. This replaces the
//! generated capability matrix's video join guard without digesting source files.
use super::{
    candle::video_mode_is_candle_eligible, catalog::VIDEO_UI_MODES,
    mlx::video_mode_is_mlx_eligible, video_mode_conditioning_requirements,
};
use crate::registry::builtin_model_spec;
use std::collections::BTreeSet;

fn image_claim(model: &str, fields: serde_json::Value, backend: &str) -> bool {
    let mut payload = fields.as_object().expect("image fields").clone();
    payload.insert("model".into(), serde_json::json!(model));
    match backend {
        "mlx" => super::mlx::image_request_mlx_eligible(model, &payload),
        "candle" => {
            let mut job = super::canonical_video_route_probe("wan_2_2", "text_to_video")
                .expect("structural job fixture");
            job.job_type =
                if payload.get("mode").and_then(serde_json::Value::as_str) == Some("edit_image") {
                    crate::contracts::JobType::ImageEdit
                } else {
                    crate::contracts::JobType::ImageGenerate
                };
            job.payload = payload;
            super::candle::image_job_is_candle_eligible(&job)
        }
        _ => panic!("unsupported backend"),
    }
}

#[test]
fn conditioned_image_routes_keep_their_real_request_shapes() {
    use serde_json::json;
    for (model, fields) in [
        (
            "qwen_image_edit_2511",
            json!({"mode":"character_image","referenceAssetIds":["a","b"]}),
        ),
        (
            "qwen_image_edit_2511_lightning",
            json!({"mode":"character_image","referenceAssetIds":["a","b"]}),
        ),
        (
            "sensenova_u1_8b_infographic_v3_fast",
            json!({"mode":"character_image","referenceAssetIds":["a","b"]}),
        ),
        (
            "flux2_dev",
            json!({"mode":"character_image","referenceAssetIds":["a","b"]}),
        ),
        (
            "z_image_edit",
            json!({"mode":"edit_image","sourceAssetId":"source"}),
        ),
        (
            "instantid_realvisxl",
            json!({"mode":"character_image","referenceAssetId":"face"}),
        ),
        (
            "pulid_flux_dev",
            json!({"mode":"character_image","referenceAssetId":"face"}),
        ),
    ] {
        for backend in ["mlx", "candle"] {
            assert!(
                image_claim(model, fields.clone(), backend),
                "{model}/{backend} must claim its actual conditioned request"
            );
        }
    }

    // The Qwen Edit adapter travels with the same conditioned request on both lanes.
    let qwen_adapter = json!({
        "mode":"character_image", "referenceAssetId":"face",
        "loras":[{"id":"user-adapter","networkType":"lora"}]
    });
    for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
        for backend in ["mlx", "candle"] {
            assert!(image_claim(model, qwen_adapter.clone(), backend));
        }
    }

    // An empty reference is not a conditioned operation on either backend.
    for backend in ["mlx", "candle"] {
        assert!(!image_claim(
            "qwen_image_edit_2511",
            json!({"mode":"character_image","referenceAssetId":" "}),
            backend
        ));
    }
}

#[test]
fn both_native_lanes_keep_shipped_text_and_control_claims() {
    use serde_json::json;
    for model in [
        "flux2_klein_9b",
        "flux2_dev",
        "sd3_5_large",
        "sd3_5_medium",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ] {
        let spec = builtin_model_spec(model).expect("shipped image spec");
        assert!(
            spec.routes.contains_key("mlx"),
            "{model} has an MLX native route"
        );
        assert!(
            spec.routes.contains_key("candle"),
            "{model} has a Candle native route"
        );
        for backend in ["mlx", "candle"] {
            assert!(
                image_claim(model, json!({"mode":"text_to_image"}), backend),
                "{model}/{backend} must claim its public text request"
            );
        }
    }

    // The strict SDXL OpenPose lane is exactly these five product ids. Check the production
    // Candle lane for every shipped model rather than relying on a copied allowlist alone.
    let exact = [
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ];
    let manifest: serde_json::Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(
        crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1,
    ))
    .unwrap();
    let actual: BTreeSet<_> = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .filter(|model| {
            let mut job = super::canonical_video_route_probe("wan_2_2", "text_to_video").unwrap();
            job.job_type = crate::contracts::JobType::ImageGenerate;
            job.payload = json!({
                "model": model,
                "mode": "text_to_image",
                "prompt": "probe",
                "advanced": {"poses":[{}]}
            })
            .as_object()
            .unwrap()
            .clone();
            super::candle::image_job_candle_lane(&job)
                == Some(super::candle::CandleImageLane::SdxlControl)
        })
        .collect();
    assert_eq!(actual, exact.into_iter().collect());
    for model in manifest["models"].as_array().unwrap() {
        assert!(
            !model["capabilities"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|operation| operation == "style_variations"),
            "{} must not advertise the retired style mode",
            model["id"]
        );
    }
    for model in exact {
        for backend in ["mlx", "candle"] {
            assert!(image_claim(
                model,
                json!({"mode":"text_to_image","advanced":{"poses":[{}]}}),
                backend
            ));
        }
    }
}

#[test]
fn guidance_and_adapter_axes_remain_backend_specific() {
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        let spec = builtin_model_spec(model).unwrap();
        let mlx = spec.routes.get("mlx").unwrap();
        let candle = spec.routes.get("candle").unwrap();
        assert!(mlx
            .supported_guidance_methods
            .iter()
            .any(|value| value == "cfg_pp"));
        assert!(!candle
            .supported_guidance_methods
            .iter()
            .any(|value| value == "cfg_pp"));
        for backend in ["mlx", "candle"] {
            assert!(image_claim(
                model,
                serde_json::json!({"mode":"text_to_image","advanced":{"guidanceMethod":"cfg"}}),
                backend
            ));
        }
    }
    for model in ["bernini", "bernini_image"] {
        let spec = builtin_model_spec(model).unwrap();
        assert!(
            spec.routes
                .get("mlx")
                .is_none_or(|route| !route.supports_lora && !route.supports_lokr),
            "{model} must not advertise native MLX adapter execution"
        );
        let candle_adapters = spec.routes.get("candle").into_iter().chain(
            spec.video_routes
                .get("candle")
                .into_iter()
                .flatten()
                .flat_map(|route| &route.engines),
        );
        assert!(
            candle_adapters
                .into_iter()
                .any(|route| route.supports_lora && route.supports_lokr),
            "{model} retains its Candle-only adapters"
        );
    }
}

#[test]
fn resolved_control_routes_and_retired_scail_conditioning_keep_their_owners() {
    use serde_json::json;
    for model in ["sdxl", "realvisxl", "realvisxl_lightning"] {
        let spec = builtin_model_spec(model).unwrap();
        assert!(spec.routes.contains_key("mlx"), "{model} MLX control owner");
        assert!(
            spec.routes.contains_key("candle"),
            "{model} Candle control owner"
        );
        for backend in ["mlx", "candle"] {
            assert!(image_claim(
                model,
                json!({"mode":"text_to_image","advanced":{"poses":[{}]}}),
                backend
            ));
        }
    }
    let scail = builtin_model_spec("scail2_14b").unwrap();
    for backend in ["mlx", "candle"] {
        for route in scail.video_routes.get(backend).into_iter().flatten() {
            assert!(
                route.engines.iter().all(|engine| !engine
                    .conditioning
                    .iter()
                    .any(|kind| kind == "multiReference")),
                "SCAIL-2 retired multiReference on {backend}/{}",
                route.mode
            );
        }
    }
    assert!(
        crate::registry::approved_backend_exceptions()
            .iter()
            .all(|record| record.id != "epic-8588-conditioning-sequencing"),
        "resolved conditioning must have no stale exception"
    );
}

#[test]
fn former_audit_regressions_follow_typed_image_and_video_owners() {
    use serde_json::json;
    let flux = builtin_model_spec("flux2_dev").unwrap();
    for backend in ["mlx", "candle"] {
        assert!(
            flux.routes[backend].supports_prompt_enhancement,
            "FLUX.2 prompt enhancement on {backend}"
        );
        assert!(image_claim(
            "sana_1600m",
            json!({"mode":"text_to_image","referenceAssetId":"base"}),
            backend
        ));
        assert!(image_claim(
            "sana_sprint_1600m",
            json!({"mode":"text_to_image","referenceAssetId":"base"}),
            backend
        ));
        assert!(image_claim(
            "mage_flow_edit",
            json!({"mode":"edit_image","sourceAssetId":"base","referenceAssetIds":["ref"]}),
            backend
        ));
        let mage = builtin_model_spec("mage_flow_edit").unwrap();
        for quant in ["q4", "q8"] {
            assert!(
                mage.routes[backend]
                    .supported_quants
                    .iter()
                    .any(|q| q == quant),
                "Mage Edit {backend}/{quant}"
            );
        }
        for kind in ["reference", "multiReference"] {
            assert!(mage.routes[backend]
                .conditioning
                .iter()
                .any(|value| value == kind));
        }
        for mode in ["extend_clip", "video_bridge", "replace_person"] {
            assert!(super::video_backend_mode_supported(backend, "ltx_2_3", mode).unwrap());
            assert!(super::video_backend_mode_supported(backend, "ltx_2_5", mode).unwrap());
            assert_eq!(
                super::video_backend_mode_supported(backend, "ltx_2_3_eros", mode).unwrap(),
                backend == "mlx",
                "Eros withdrawal on {backend}/{mode}"
            );
        }
        assert!(super::video_backend_mode_supported(
            backend,
            "wan_2_2_vace_fun_14b",
            "replace_person"
        )
        .unwrap());
    }
    let krea = builtin_model_spec("krea_2_turbo").unwrap().project(None);
    let convrot = krea["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["variant"] == "int8-convrot")
        .unwrap();
    assert_eq!(convrot["platforms"], json!(["windows", "linux"]));
    let mlx_caps: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../config/engine-capabilities/runtime/capabilities.mlx.json"
    ))
    .unwrap();
    let candle_caps: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../config/engine-capabilities/runtime/capabilities.candle.json"
    ))
    .unwrap();
    assert!(!mlx_caps["workerCapabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "int8_convrot"));
    assert!(candle_caps["workerCapabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "int8_convrot"));

    let scail = builtin_model_spec("scail2_14b").unwrap();
    for backend in ["mlx", "candle"] {
        assert!(scail.video_routes[backend].iter().any(|route| {
            route.mode == "animate_character"
                && route
                    .engines
                    .iter()
                    .any(|engine| engine.supports_lora && engine.supports_lokr)
        }));
    }
    let vace = builtin_model_spec("wan_2_2_vace_fun_14b").unwrap();
    for backend in ["mlx", "candle"] {
        let replace = vace.video_routes[backend]
            .iter()
            .find(|route| route.mode == "replace_person")
            .unwrap();
        assert!(replace.engines.iter().any(|engine| engine.supports_lora));
        assert!(replace.engines.iter().any(|engine| engine.supports_lokr));
        for kind in ["controlClip", "reference"] {
            assert!(replace
                .engines
                .iter()
                .any(|engine| engine.conditioning.iter().any(|value| value == kind)));
        }
        for quant in ["q4", "q8"] {
            assert_eq!(
                replace
                    .engines
                    .iter()
                    .any(|engine| engine.supported_quants.iter().any(|value| value == quant)),
                backend == "mlx",
                "VACE-Fun {backend}/{quant}"
            );
        }
    }
    let mage_target = crate::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == "mage_flow_base_lora")
        .unwrap();
    assert!(mage_target.limits["networkTypes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|kind| kind == "full"));
    let chatterbox = builtin_model_spec("chatterbox_tts").unwrap();
    let candle_audio = chatterbox.provider_routes["candle"]
        .iter()
        .filter_map(|provider| provider.generator.as_ref())
        .find(|generator| generator.id == "chatterbox_tts")
        .unwrap();
    for kind in ["referenceAudio", "voiceEmbedding"] {
        assert!(candle_audio.conditioning.iter().any(|value| value == kind));
    }
    assert!(!chatterbox
        .provider_routes
        .get("mlx")
        .into_iter()
        .flatten()
        .filter_map(|provider| provider.generator.as_ref())
        .any(|generator| generator.id == "chatterbox_tts"));
}

#[test]
fn former_audit_upscale_examples_use_live_dispatch_predicates() {
    use crate::contracts::JobType;
    for engine in ["aura-sr", "real-esrgan", "seedvr2"] {
        let mut job = super::canonical_video_route_probe("wan_2_2", "text_to_video").unwrap();
        job.job_type = JobType::ImageUpscale;
        job.payload = serde_json::json!({"engine":engine})
            .as_object()
            .unwrap()
            .clone();
        for eligible in [
            super::mlx::upscale_job_is_mlx_eligible(&job),
            super::candle::upscale_job_is_candle_eligible(&job),
        ] {
            assert_eq!(eligible, engine != "aura-sr", "{engine} dispatch");
        }
    }
}

#[test]
fn imported_image_defaults_join_live_family_routes() {
    use super::catalog::imported_provider_routes;
    use serde_json::json;
    let supported: BTreeSet<_> = crate::base_weights::IMPORT_SUPPORTED_FAMILIES
        .iter()
        .copied()
        .collect();
    assert_eq!(supported, ["flux2", "krea_2", "mage-flow", "sdxl"].into());
    for (family, expected_modes) in [
        ("krea_2", json!(["text_to_image", "edit_image"])),
        ("sdxl", json!(["text_to_image"])),
        ("mage-flow", json!(["text_to_image"])),
        ("flux2", json!(["text_to_image"])),
    ] {
        let mut entry = serde_json::Map::new();
        crate::lora_family::apply_model_manifest_defaults(&mut entry, "image", Some(family));
        assert_eq!(
            entry.get("capabilities"),
            Some(&expected_modes),
            "{family} default operation claim"
        );
        for backend in ["mlx", "candle"] {
            for route in imported_provider_routes(backend, family)
                .filter(|route| route.operation == "generate")
            {
                let mut imported = entry.clone();
                imported.insert("id".into(), json!(format!("user_{family}")));
                imported.insert("family".into(), json!(family));
                imported.insert("importSourceShape".into(), json!(route.source));
                imported.insert("paths".into(), json!({"model":"/tmp/registered-import"}));
                let payload = json!({
                    "model": format!("user_{family}"),
                    "mode": "text_to_image",
                    "modelManifestEntry": imported
                });
                let payload = payload.as_object().unwrap();
                assert!(
                    super::catalog::imported_backend_declares_route(payload, backend),
                    "{family}/{backend}/{} native declaration",
                    route.source
                );
                assert!(
                    super::catalog::imported_image_request_provider_eligible(
                        &format!("user_{family}"),
                        payload,
                        backend
                    ),
                    "{family}/{backend}/{} production route",
                    route.source
                );
            }
        }
    }
}

#[test]
fn video_advanced_job_types_use_claimable_modes() {
    use crate::contracts::JobType;
    for (mode, job_type) in [
        ("extend_clip", JobType::VideoExtend),
        ("video_bridge", JobType::VideoBridge),
        ("replace_person", JobType::PersonReplace),
    ] {
        let request = super::canonical_video_route_probe("ltx_2_5", mode).unwrap();
        assert_eq!(request.job_type, job_type, "{mode} request job type");
        for backend in ["mlx", "candle"] {
            assert!(
                super::video_backend_mode_supported(backend, "ltx_2_5", mode).unwrap(),
                "{backend} has a native representative for {mode}"
            );
        }
    }
    for backend in ["mlx", "candle"] {
        assert!(
            !super::video_backend_mode_supported(backend, "wan_2_2_i2v_14b", "first_last_frame")
                .unwrap(),
            "{backend} must reject a withdrawn keyframe route"
        );
    }
}

#[test]
fn every_shipped_video_mode_with_a_native_mapping_is_claimable() {
    let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("shipped model manifest");
    let manifest: serde_json::Value =
        serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw))
            .expect("shipped model manifest parses");
    let models = manifest["models"].as_array().expect("shipped models");
    for model in models.iter().filter(|model| model["type"] == "video") {
        let id = model["id"].as_str().expect("model id");
        let spec = builtin_model_spec(id).expect("every shipped video has a typed spec");
        for (backend, eligible) in [
            ("mlx", video_mode_is_mlx_eligible as fn(&str, &str) -> bool),
            (
                "candle",
                video_mode_is_candle_eligible as fn(&str, &str) -> bool,
            ),
        ] {
            let mapped: BTreeSet<_> = spec
                .video_routes
                .get(backend)
                .into_iter()
                .flatten()
                .map(|route| route.mode.as_str())
                .collect();
            for route in spec.video_routes.get(backend).into_iter().flatten() {
                assert!(
                    !route.engines.is_empty(),
                    "{id}/{backend}/{} maps no engine",
                    route.mode
                );
                let conditioning: BTreeSet<_> = route
                    .engines
                    .iter()
                    .flat_map(|engine| engine.conditioning.iter().map(String::as_str))
                    .collect();
                for group in video_mode_conditioning_requirements(&route.mode) {
                    assert!(
                        group.iter().any(|kind| conditioning.contains(kind)),
                        "{id}/{backend}/{} lacks required conditioning {group:?}",
                        route.mode
                    );
                }
            }
            for mode in VIDEO_UI_MODES {
                assert_eq!(
                    mapped.contains(mode),
                    eligible(id, mode),
                    "{id}/{backend}/{mode} native mapping and production claim differ"
                );
            }
        }
    }
}

#[test]
fn imported_provider_operations_keep_exact_source_and_conditioning_claims() {
    use super::catalog::{
        imported_backend_declares_route, imported_image_request_provider_eligible,
    };
    let mut sources = BTreeSet::new();
    for backend in ["mlx", "candle"] {
        let raw = if backend == "mlx" {
            include_str!("../../../../../config/engine-capabilities/capabilities.mlx.json")
        } else {
            include_str!("../../../../../config/engine-capabilities/capabilities.candle.json")
        };
        let facts: serde_json::Value = serde_json::from_str(raw).unwrap();
        for route in facts["imports"].as_array().unwrap() {
            sources.insert((
                backend,
                route["family"].as_str().unwrap().to_owned(),
                route["source"].as_str().unwrap().to_owned(),
            ));
        }
    }
    for (backend, family, source) in sources {
        for operation in ["generate", "edit", "multi_phase", "pose"] {
            let mut payload = serde_json::json!({
                "model": "user_registry_probe",
                "modelManifestEntry": {
                    "id": "user_registry_probe",
                    "family": family,
                    "importSourceShape": source,
                    "paths": { "model": "/tmp/imported-registry-probe" }
                }
            });
            match operation {
                "edit" => {
                    payload["mode"] = serde_json::json!("edit_image");
                    payload["sourceAssetId"] = serde_json::json!("source");
                }
                "multi_phase" => payload["advanced"] = serde_json::json!({ "phases": [{}] }),
                "pose" => payload["advanced"] = serde_json::json!({ "poses": [{}] }),
                _ => {}
            }
            let request = payload.as_object().unwrap();
            assert_eq!(
                imported_backend_declares_route(request, backend),
                imported_image_request_provider_eligible("user_registry_probe", request, backend),
                "{backend}/{family}/{source}/{operation} descriptor declaration and production claim differ",
            );
        }
    }
}
