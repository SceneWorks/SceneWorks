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

// ---------------------------------------------------------------------------------------------
// The reverse direction of the approved-exception check.
//
// `registry::exceptions` proves every APPROVED cell is still MLX-only. This proves the converse:
// every cell that IS MLX-only today is approved, so a new unapproved gap (a Candle route withdrawn
// in `route-policy.json`, a Candle descriptor losing a tier, a mode or an adapter) fails here
// instead of shipping silently. The cell semantics are the retired matrix's: a backend serves a
// cell when the typed native route (`ModelSpec::routes` / `video_routes`, the compiled descriptor
// facts) declares the axis value, or a Candle request lands on a bespoke production lane, AND the
// production claim gate (`worker_supports_job` against that backend's checked-in worker
// capabilities) admits the canonical request that carries it. No source text is read.
// ---------------------------------------------------------------------------------------------

mod residual {
    use crate::contracts::{
        ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage, WorkerCapability,
        WorkerSnapshot, WorkerStatus,
    };
    use crate::registry::{GeneratorRoute, ModelSpec};
    use serde_json::{json, Map, Value};
    use std::collections::{BTreeMap, BTreeSet};

    pub(super) const BACKENDS: [&str; 2] = ["mlx", "candle"];

    /// `cell path -> (served on MLX, served on Candle)`.
    pub(super) type Cells = BTreeMap<String, (bool, bool)>;

    fn worker(backend: &str) -> WorkerSnapshot {
        let raw = match backend {
            "mlx" => include_str!(
                "../../../../../config/engine-capabilities/runtime/capabilities.mlx.json"
            ),
            _ => include_str!(
                "../../../../../config/engine-capabilities/runtime/capabilities.candle.json"
            ),
        };
        let facts: Value = serde_json::from_str(raw).expect("runtime worker facts");
        let capabilities = facts["workerCapabilities"]
            .as_array()
            .expect("worker capabilities")
            .iter()
            .map(|value| {
                serde_json::from_value::<WorkerCapability>(value.clone())
                    .expect("known worker capability")
            })
            .collect();
        WorkerSnapshot {
            id: format!("residual-{backend}"),
            platform: None,
            gpu_id: if backend == "mlx" { "mlx" } else { "0" }.to_owned(),
            gpu_name: None,
            status: WorkerStatus::Idle,
            current_job_id: None,
            capabilities,
            loaded_models: Vec::new(),
            utilization: None,
            status_reason: None,
            registered_at: String::new(),
            last_seen_at: String::new(),
            extra: BTreeMap::new(),
        }
    }

    fn job(job_type: JobType, model: &str, payload: Value) -> JobSnapshot {
        let mut payload = payload.as_object().cloned().expect("object payload");
        payload.insert("model".into(), json!(model));
        payload
            .entry("prompt".to_owned())
            .or_insert_with(|| json!("probe"));
        JobSnapshot {
            id: "residual-probe".to_owned(),
            job_type,
            status: JobStatus::Queued,
            project_id: None,
            project_name: None,
            payload,
            result: Map::new(),
            requested_gpu: "auto".to_owned(),
            assigned_gpu: None,
            worker_id: None,
            progress: ContractNumber::from(0_u64),
            stage: ProgressStage::Queued,
            message: String::new(),
            error: None,
            eta_seconds: None,
            elapsed_seconds: None,
            attempts: 0,
            source_job_id: None,
            duplicate_of_job_id: None,
            cancel_requested: false,
            created_at: String::new(),
            updated_at: String::new(),
            started_at: None,
            completed_at: None,
            canceled_at: None,
            last_heartbeat_at: None,
            peak_gpu_memory_pct: None,
            peak_gpu_load_pct: None,
            backend: None,
            title: None,
            extra: BTreeMap::new(),
        }
    }

    /// The two production lanes, as the claim gate sees them.
    pub(super) struct Lanes {
        workers: BTreeMap<&'static str, WorkerSnapshot>,
    }

    impl Lanes {
        pub(super) fn new() -> Self {
            Self {
                workers: BACKENDS.iter().map(|b| (*b, worker(b))).collect(),
            }
        }

        fn claims(&self, backend: &str, job: &JobSnapshot) -> bool {
            crate::jobs_store::worker_supports_job(&self.workers[backend], job)
        }
    }

    fn field<'a>(spec: &'a ModelSpec, key: &str) -> Option<&'a Value> {
        spec.product.fields.get(key)
    }

    fn ui_flag(spec: &ModelSpec, key: &str) -> bool {
        field(spec, "ui")
            .and_then(|ui| ui.get(key))
            .and_then(Value::as_bool)
            == Some(true)
    }

    fn evaluated_operations(spec: &ModelSpec) -> Vec<String> {
        let mut operations: BTreeSet<String> = field(spec, "capabilities")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        if ui_flag(spec, "img2img") {
            operations.insert("image_to_image".into());
        }
        if ui_flag(spec, "promptEnhance") {
            operations.insert("prompt_enhancement".into());
        }
        operations.into_iter().collect()
    }

    /// The production request for one image operation (the API's canonical shape).
    fn operation_request(spec: &ModelSpec, operation: &str) -> Option<(JobType, Value)> {
        let id = spec.product.id.as_str();
        Some(match operation {
            "text_to_image" => (JobType::ImageGenerate, json!({ "mode": operation })),
            "edit_image" => (
                JobType::ImageEdit,
                json!({ "mode": operation, "sourceAssetId": "probe" }),
            ),
            "image_to_image" if id == "z_image_edit" => (
                JobType::ImageEdit,
                json!({ "mode": "edit_image", "sourceAssetId": "probe" }),
            ),
            "image_to_image" if id.starts_with("flux2_") => (
                JobType::ImageGenerate,
                json!({ "mode": operation, "referenceAssetId": "probe" }),
            ),
            "image_to_image" => (
                JobType::ImageGenerate,
                json!({ "mode": "text_to_image", "referenceAssetId": "probe" }),
            ),
            "character_image" => (
                JobType::ImageGenerate,
                json!({ "mode": operation, "referenceAssetId": "probe" }),
            ),
            "vqa" => (
                JobType::ImageVqa,
                json!({ "sourceAssetId": "probe", "question": "probe" }),
            ),
            "interleave" => (
                JobType::ImageInterleave,
                json!({ "sourceAssetIds": ["probe"] }),
            ),
            "image_inpaint" => (
                JobType::ImageEdit,
                json!({ "mode": "edit_image", "sourceAssetId": "probe", "maskAssetId": "probe-mask" }),
            ),
            "image_detail" => (
                JobType::ImageDetail,
                json!({ "sourceAssetId": "probe", "engine": "controlnet-tile" }),
            ),
            "prompt_enhancement" => (
                JobType::ImageGenerate,
                json!({ "mode": "text_to_image", "advanced": { "enhancePrompt": true } }),
            ),
            _ => return None,
        })
    }

    fn canonical_image_request(spec: &ModelSpec) -> Option<(JobType, Value)> {
        let operations = evaluated_operations(spec);
        [
            "text_to_image",
            "edit_image",
            "character_image",
            "image_to_image",
            "image_inpaint",
            "image_detail",
            "vqa",
            "interleave",
        ]
        .into_iter()
        .find(|op| operations.iter().any(|item| item == op))
        .and_then(|op| operation_request(spec, op))
    }

    /// A Candle request that lands on a bespoke production lane is served by that lane even when
    /// the shared generator descriptor does not spell the axis: the generic txt2img lane refuses
    /// every conditioning carrier, so reaching a bespoke lane IS that lane's claim.
    fn candle_bespoke_lane(job: &JobSnapshot) -> bool {
        use super::super::candle::{image_job_candle_lane, CandleImageLane};
        !matches!(
            image_job_candle_lane(job),
            None | Some(CandleImageLane::TextToImage)
        )
    }

    fn image_route<'a>(spec: &'a ModelSpec, backend: &str) -> Option<&'a GeneratorRoute> {
        spec.routes.get(backend)
    }

    /// A backend serves an adapter cell when its typed descriptor declares it, or (Candle) the
    /// request lands on a bespoke lane, and the production gate claims the request.
    fn served(
        lanes: &Lanes,
        spec: &ModelSpec,
        backend: &str,
        job: &JobSnapshot,
        descriptor: impl Fn(&GeneratorRoute) -> bool,
    ) -> bool {
        let declared = image_route(spec, backend).is_some_and(&descriptor)
            || (backend == "candle" && candle_bespoke_lane(job));
        declared && lanes.claims(backend, job)
    }

    /// Operations and conditioning shapes may also be served by a bespoke production lane the
    /// shared generator descriptor does not spell (identity adapters, edit lanes). On Candle the
    /// router names that lane. The MLX router has no lane enum but refuses every request shape it
    /// has no lane for, so its admission of the carrier-bearing request is the lane's claim, with
    /// one exception: MLX admits a pose on a pose-as-reference model by folding it into the
    /// reference, so a strict `control` cell on MLX also needs the product to declare a pose
    /// control surface (`ui.controlModes`).
    fn served_by_lane(
        lanes: &Lanes,
        spec: &ModelSpec,
        backend: &str,
        job: &JobSnapshot,
        shape: Option<&str>,
        descriptor: impl Fn(&GeneratorRoute) -> bool,
    ) -> bool {
        let mlx_lane = backend == "mlx"
            && (shape != Some("control")
                || field(spec, "ui")
                    .and_then(|ui| ui.get("controlModes"))
                    .and_then(Value::as_array)
                    .is_some_and(|modes| modes.iter().any(|mode| mode == "pose")));
        let declared = image_route(spec, backend).is_some_and(&descriptor)
            || mlx_lane
            || (backend == "candle" && candle_bespoke_lane(job));
        declared && lanes.claims(backend, job)
    }

    fn artifact_tier(spec: &ModelSpec, tier: &str, backend: &str) -> bool {
        field(spec, "downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|row| row.get("variant").and_then(Value::as_str) == Some(tier))
            .any(|row| {
                let platforms: Vec<&str> = row
                    .get("platforms")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect();
                platforms.is_empty()
                    || platforms.iter().any(|platform| {
                        if backend == "mlx" {
                            *platform == "macos"
                        } else {
                            matches!(*platform, "windows" | "linux")
                        }
                    })
            })
    }

    fn precision_advanced(tier: &str) -> Option<Option<Value>> {
        Some(match tier {
            "bf16" => None,
            "q4" => Some(json!({ "mlxQuantize": 4 })),
            "q8" => Some(json!({ "mlxQuantize": 8 })),
            "nvfp4" => Some(json!({ "quantTier": "nvfp4" })),
            crate::image_request::FP8R_QUANT_TIER => {
                Some(json!({ "quantTier": crate::image_request::FP8R_QUANT_TIER }))
            }
            "int8-convrot" => Some(json!({ "convRot": true })),
            _ => return None,
        })
    }

    fn precision_tiers(spec: &ModelSpec, routes: &[&GeneratorRoute]) -> BTreeSet<String> {
        let mut tiers: BTreeSet<String> = field(spec, "downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|row| row.get("variant").and_then(Value::as_str))
            .filter(|variant| !variant.is_empty() && *variant != "training")
            .map(str::to_owned)
            .collect();
        for route in routes {
            if matches!(route.modality.as_str(), "image" | "video" | "both") {
                tiers.insert("bf16".into());
            }
            tiers.extend(route.supported_quants.iter().cloned());
        }
        tiers
    }

    fn guidance_methods(value: Option<&Value>) -> Option<BTreeSet<String>> {
        value.and_then(Value::as_array).map(|methods| {
            methods
                .iter()
                .filter_map(Value::as_str)
                .filter(|m| !m.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
    }

    fn manifest_guidance(spec: &ModelSpec, backend: Option<&str>) -> Option<BTreeSet<String>> {
        match backend {
            None => guidance_methods(field(spec, "limits").and_then(|l| l.get("guidanceMethods"))),
            Some(b) => guidance_methods(
                field(spec, b)
                    .and_then(|v| v.get("limits"))
                    .and_then(|l| l.get("guidanceMethods")),
            ),
        }
    }

    fn conditioning_request(
        spec: &ModelSpec,
        shape: &str,
        job_type: JobType,
        mut payload: Value,
        operations: &[String],
    ) -> Option<(JobType, Value)> {
        let id = spec.product.id.as_str();
        let has = |op: &str| operations.iter().any(|item| item == op);
        let mut kind = job_type;
        match shape {
            "reference" if id == "bernini_image" => return operation_request(spec, "edit_image"),
            "reference"
                if (id.starts_with("sensenova_")
                    || id.starts_with("qwen_image_edit")
                    || id.starts_with("flux2_"))
                    && has("character_image") =>
            {
                return operation_request(spec, "character_image")
            }
            "reference" | "referenceRgba" => {
                if kind != JobType::ImageEdit {
                    payload["referenceAssetId"] = json!("probe");
                }
            }
            "multiReference" | "reduxRefs" => {
                if has("character_image") {
                    let (k, mut character) = operation_request(spec, "character_image")?;
                    character.as_object_mut()?.remove("referenceAssetId");
                    character["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                    kind = k;
                    payload = character;
                } else if has("edit_image") {
                    let (k, mut edit) = operation_request(spec, "edit_image")?;
                    edit["sourceAssetId"] = json!("probe-source");
                    edit["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                    kind = k;
                    payload = edit;
                } else {
                    payload["referenceAssetIds"] = json!(["probe-a", "probe-b"]);
                }
            }
            "control" => payload["advanced"] = json!({ "poses": [{}] }),
            "depth" => payload["advanced"] = json!({ "depthAssetId": "probe" }),
            "mask" => {
                let (k, mut edit) = operation_request(spec, "edit_image")?;
                edit["maskAssetId"] = json!("probe-mask");
                kind = k;
                payload = edit;
            }
            "conversationHistory" => {
                kind = JobType::ImageInterleave;
                payload = json!({ "conversationHistory": [{ "role": "user", "text": "probe" }] });
            }
            _ => return None,
        }
        Some((kind, payload))
    }

    fn image_cells(lanes: &Lanes, spec: &ModelSpec, out: &mut Cells) {
        let id = spec.product.id.clone();
        let mut push = |axis: &str, value: &str, mlx: bool, candle: bool| {
            out.insert(format!("models/{id}/{axis}/{value}"), (mlx, candle));
        };
        let operations = evaluated_operations(spec);
        for operation in &operations {
            let Some((job_type, payload)) = operation_request(spec, operation) else {
                continue;
            };
            let probe = job(job_type, &spec.product.id, payload);
            let [mlx, candle] = BACKENDS.map(|b| {
                if operation == "prompt_enhancement" {
                    image_route(spec, b).is_some_and(|r| r.supports_prompt_enhancement)
                        && lanes.claims(b, &probe)
                } else {
                    served_by_lane(lanes, spec, b, &probe, None, |_| true)
                }
            });
            push("operationAndMode", operation, mlx, candle);
        }
        let Some((job_type, base)) = canonical_image_request(spec) else {
            return;
        };
        if field(spec, "loraCompatibility").is_some_and(|v| !v.is_null()) {
            for adapter in ["lora", "lokr"] {
                let mut payload = base.clone();
                payload["loras"] = json!([{ "id": "probe", "networkType": adapter }]);
                let probe = job(job_type.clone(), &spec.product.id, payload);
                let [mlx, candle] = BACKENDS.map(|b| {
                    served(lanes, spec, b, &probe, |r| match adapter {
                        "lora" => r.supports_lora,
                        _ => r.supports_lokr,
                    })
                });
                push("userAdapters", adapter, mlx, candle);
            }
        }
        let mut shapes: BTreeSet<String> = BACKENDS
            .iter()
            .filter_map(|b| image_route(spec, b))
            .flat_map(|r| r.conditioning.iter().cloned())
            .collect();
        if ui_flag(spec, "img2img") {
            shapes.insert("reference".into());
        }
        if ui_flag(spec, "multiReference") || ui_flag(spec, "sourceWithMultiReference") {
            shapes.insert("multiReference".into());
        }
        if ui_flag(spec, "poseLibrary") {
            shapes.insert("control".into());
        }
        // A character request is a reference request: its face or subject travels in the
        // reference carrier.
        if operations.iter().any(|op| op == "character_image") {
            shapes.insert("reference".into());
        }
        for shape in &shapes {
            let Some((kind, payload)) =
                conditioning_request(spec, shape, job_type.clone(), base.clone(), &operations)
            else {
                continue;
            };
            let probe = job(kind, &spec.product.id, payload);
            let [mlx, candle] = BACKENDS.map(|b| {
                served_by_lane(lanes, spec, b, &probe, Some(shape), |r| {
                    r.conditioning.iter().any(|k| k == shape)
                })
            });
            push("conditioningShape", shape, mlx, candle);
        }
        let routes: Vec<&GeneratorRoute> = BACKENDS
            .iter()
            .filter_map(|b| image_route(spec, b))
            .collect();
        for tier in precision_tiers(spec, &routes) {
            let Some(advanced) = precision_advanced(&tier) else {
                continue;
            };
            let mut payload = base.clone();
            if let Some(advanced) = advanced {
                payload["advanced"] = advanced;
            }
            let probe = job(job_type.clone(), &spec.product.id, payload);
            let [mlx, candle] = BACKENDS.map(|b| {
                let declared = image_route(spec, b)
                    .is_some_and(|r| tier == "bf16" || r.supported_quants.contains(&tier))
                    || artifact_tier(spec, &tier, b);
                declared && lanes.claims(b, &probe)
            });
            push("precisionTier", &tier, mlx, candle);
        }
        let mut methods = manifest_guidance(spec, None).unwrap_or_default();
        for b in BACKENDS {
            methods.extend(manifest_guidance(spec, Some(b)).unwrap_or_default());
        }
        for method in &methods {
            let mut payload = json!({ "mode": "text_to_image" });
            if method != "cfg" {
                payload["advanced"] = json!({ "guidanceMethod": method });
            }
            let probe = job(JobType::ImageGenerate, &spec.product.id, payload);
            let [mlx, candle] = BACKENDS.map(|b| {
                let listed = manifest_guidance(spec, Some(b))
                    .or_else(|| manifest_guidance(spec, None))
                    .map_or(method == "cfg", |set| set.contains(method));
                listed
                    && lanes.claims(b, &probe)
                    && image_route(spec, b).is_some_and(|r| {
                        method == "cfg" || r.supported_guidance_methods.contains(method)
                    })
            });
            push("guidanceMethod", method, mlx, candle);
        }
    }

    fn video_engines<'a>(
        spec: &'a ModelSpec,
        backend: &str,
        mode: &str,
    ) -> Vec<&'a GeneratorRoute> {
        spec.video_routes
            .get(backend)
            .into_iter()
            .flatten()
            .filter(|route| route.mode == mode)
            .flat_map(|route| &route.engines)
            .collect()
    }

    fn video_probe(model: &str, mode: &str) -> JobSnapshot {
        super::super::canonical_video_route_probe(model, mode).expect("canonical video probe")
    }

    fn video_cells(lanes: &Lanes, spec: &ModelSpec, out: &mut Cells) {
        use super::super::catalog::VIDEO_UI_MODES;
        use super::super::video_mode_conditioning_requirements;
        let id = spec.product.id.as_str();
        let mut push = |axis: &str, value: &str, mlx: bool, candle: bool| {
            out.insert(format!("models/{id}/{axis}/{value}"), (mlx, candle));
        };
        for mode in VIDEO_UI_MODES {
            let probe = video_probe(id, mode);
            let [mlx, candle] = BACKENDS
                .map(|b| !video_engines(spec, b, mode).is_empty() && lanes.claims(b, &probe));
            push("operationAndMode", mode, mlx, candle);
        }
        for adapter in ["lora", "lokr"] {
            let [mlx, candle] = BACKENDS.map(|b| {
                VIDEO_UI_MODES.iter().any(|mode| {
                    let declared = video_engines(spec, b, mode).iter().any(|e| match adapter {
                        "lora" => e.supports_lora,
                        _ => e.supports_lokr,
                    });
                    let mut probe = video_probe(id, mode);
                    probe.payload.insert(
                        "loras".into(),
                        json!([{ "id": "probe", "networkType": adapter }]),
                    );
                    declared && lanes.claims(b, &probe)
                })
            });
            push("userAdapters", adapter, mlx, candle);
        }
        let all: Vec<&GeneratorRoute> = BACKENDS
            .iter()
            .flat_map(|b| spec.video_routes.get(*b).into_iter().flatten())
            .flat_map(|route| &route.engines)
            .collect();
        let shapes: BTreeSet<&str> = all
            .iter()
            .flat_map(|e| e.conditioning.iter().map(String::as_str))
            .collect();
        for shape in shapes {
            let requirement = match shape {
                "reduxRefs" | "referenceAudio" | "referenceVideo" => "multiReference",
                other => other,
            };
            let modes: Vec<&str> = VIDEO_UI_MODES
                .iter()
                .copied()
                .filter(|mode| {
                    video_mode_conditioning_requirements(mode)
                        .iter()
                        .any(|alternatives| alternatives.contains(&requirement))
                        || (shape == "mask" && *mode == "replace_person")
                })
                .collect();
            let [mlx, candle] = BACKENDS.map(|b| {
                modes.iter().any(|mode| {
                    let engines = video_engines(spec, b, mode);
                    let declared = engines
                        .iter()
                        .any(|e| e.conditioning.iter().any(|k| k == shape));
                    let shadowed = shape == "reference"
                        && *mode == "reference_to_video"
                        && engines
                            .iter()
                            .any(|e| e.conditioning.iter().any(|k| k == "multiReference"));
                    declared && !shadowed && lanes.claims(b, &video_probe(id, mode))
                })
            });
            push("conditioningShape", shape, mlx, candle);
        }
        // Precision is probed on the first mapped, claimable mode, as a request for the model's
        // canonical operation would be.
        let Some(mode) = VIDEO_UI_MODES.iter().copied().find(|mode| {
            BACKENDS.iter().any(|b| {
                !video_engines(spec, b, mode).is_empty() && lanes.claims(b, &video_probe(id, mode))
            })
        }) else {
            return;
        };
        for tier in precision_tiers(spec, &all) {
            let Some(advanced) = precision_advanced(&tier) else {
                continue;
            };
            let mut probe = video_probe(id, mode);
            if let Some(advanced) = advanced {
                probe.payload.insert("advanced".into(), advanced);
            }
            let [mlx, candle] = BACKENDS.map(|b| {
                let declared = video_engines(spec, b, mode)
                    .iter()
                    .any(|e| tier == "bf16" || e.supported_quants.contains(&tier))
                    || artifact_tier(spec, &tier, b);
                declared && lanes.claims(b, &probe)
            });
            push("precisionTier", &tier, mlx, candle);
        }
    }

    /// Every MLX-only utility request: the representative request of each model-independent GPU
    /// job type, each upscale engine, and each vector mode a shipped model advertises.
    pub(super) fn gpu_job_cells(lanes: &Lanes) -> Cells {
        let mut probes: Vec<(String, JobSnapshot)> = [
            (
                "person_detect",
                JobType::PersonDetect,
                "person_detector",
                json!({ "sourceAssetId": "probe" }),
            ),
            (
                "person_track",
                JobType::PersonTrack,
                "person_detector",
                json!({ "sourceAssetId": "probe" }),
            ),
            (
                "pose_detect",
                JobType::PoseDetect,
                "dwpose_pose_detector",
                json!({ "sourceAssetId": "probe" }),
            ),
            (
                "kps_extract",
                JobType::KpsExtract,
                "person_detector",
                json!({ "sourceAssetId": "probe" }),
            ),
            (
                "image_segment",
                JobType::ImageSegment,
                "sam3_person_segment",
                json!({ "sourceAssetId": "probe", "box": [0, 0, 1, 1] }),
            ),
            (
                "image_depth",
                JobType::ImageDepth,
                "iris_3b_depth",
                json!({ "sourceAssetId": "probe", "resolution": { "mode": "capped", "maxSide": 1024 } }),
            ),
            (
                "video_upscale",
                JobType::VideoUpscale,
                "seedvr2_upscaler",
                json!({ "engine": "seedvr2", "sourceAssetId": "probe" }),
            ),
            (
                "training_caption",
                JobType::TrainingCaption,
                "joycaption_beta_one",
                json!({ "captioner": "joy_caption", "sourceAssetId": "probe" }),
            ),
            (
                "dataset_analysis",
                JobType::DatasetAnalysis,
                "clip_vit_l14",
                json!({ "datasetId": "probe" }),
            ),
            (
                "catalog_analysis",
                JobType::CatalogAnalysis,
                "clip_vit_l14",
                json!({ "catalogId": "probe" }),
            ),
            (
                "dataset_upscale",
                JobType::DatasetUpscale,
                "real_esrgan",
                json!({ "datasetId": "probe", "engine": "real-esrgan" }),
            ),
            (
                "dataset_face_analysis",
                JobType::DatasetFaceAnalysis,
                "person_detector",
                json!({ "datasetId": "probe" }),
            ),
            (
                "dataset_subject_mask",
                JobType::DatasetSubjectMask,
                "sam3_person_segment",
                json!({ "datasetId": "probe" }),
            ),
            (
                "face_likeness_compare",
                JobType::FaceLikenessCompare,
                "person_detector",
                json!({ "sourceAssetId": "probe", "referenceAssetId": "probe-ref" }),
            ),
            (
                "prompt_refine",
                JobType::PromptRefine,
                "prompt_refine_anubis_8b",
                json!({ "prompt": "probe" }),
            ),
        ]
        .into_iter()
        .map(|(name, kind, model, payload)| {
            (
                format!("gpuJobTypes/{name}/representative_request"),
                job(kind, model, payload),
            )
        })
        .collect();
        for engine in ["real-esrgan", "seedvr2", "aura-sr"] {
            probes.push((
                format!("gpuJobTypes/image_upscale/engine:{engine}"),
                job(
                    JobType::ImageUpscale,
                    engine,
                    json!({ "engine": engine, "sourceAssetId": "probe" }),
                ),
            ));
        }
        for spec in crate::registry::builtin_model_specs() {
            for (mode, payload) in [
                (
                    "image_to_svg",
                    json!({ "mode": "image_to_svg", "sourceAssetId": "probe" }),
                ),
                ("text_to_svg", json!({ "mode": "text_to_svg" })),
            ] {
                if evaluated_operations(spec).iter().any(|op| op == mode) {
                    probes.push((
                        format!("gpuJobTypes/vector_generate/{mode}"),
                        job(JobType::VectorGenerate, &spec.product.id, payload),
                    ));
                }
            }
        }
        probes
            .into_iter()
            .map(|(cell, probe)| {
                (
                    cell,
                    (lanes.claims("mlx", &probe), lanes.claims("candle", &probe)),
                )
            })
            .collect()
    }

    /// Every MLX-only training cell: a shipped target and network type the MLX trainer export
    /// serves and the MLX lane claims, but Candle does not.
    pub(super) fn training_cells(lanes: &Lanes) -> Cells {
        use crate::registry::{native_trainer_serves, offered_network_types};
        let mut out = Cells::new();
        for target in crate::training::builtin_training_targets().targets {
            for kind in offered_network_types(&target) {
                let job_type = if kind == "control" {
                    JobType::ControlTraining
                } else {
                    JobType::LoraTrain
                };
                let mut probe = job(
                    job_type,
                    "",
                    json!({
                        "dryRun": false,
                        "plan": {
                            "target": {
                                "targetId": target.id,
                                "kernel": target.kernel,
                                "baseModel": target.base_model
                            },
                            "config": { "advanced": { "networkType": kind } }
                        }
                    }),
                );
                probe.payload.remove("model");
                probe.payload.remove("prompt");
                let [mlx, candle] = BACKENDS
                    .map(|b| native_trainer_serves(b, &target, &kind) && lanes.claims(b, &probe));
                out.insert(
                    format!("trainingKernels/{}/{kind}", target.id),
                    (mlx, candle),
                );
            }
        }
        out
    }

    /// Every cell of the given generators as `(mlx, candle)`, from typed specs and production
    /// claims.
    pub(super) fn model_cells<'a>(
        lanes: &Lanes,
        specs: impl Iterator<Item = &'a ModelSpec>,
    ) -> Cells {
        let mut out = Cells::new();
        for spec in specs {
            match spec.product.model_type.as_str() {
                "image" => image_cells(lanes, spec, &mut out),
                "video" => video_cells(lanes, spec, &mut out),
                _ => {}
            }
            let preview = |b: &str| spec.preview_by_backend.get(b).copied() == Some(true);
            out.insert(
                format!("models/{}/preview", spec.product.id),
                (preview("mlx"), preview("candle")),
            );
        }
        out
    }

    /// Every cell the shipped catalog, trainer exports and utility lanes define.
    pub(super) fn all_cells(lanes: &Lanes) -> Cells {
        let mut cells = model_cells(lanes, crate::registry::builtin_model_specs());
        cells.extend(training_cells(lanes));
        cells.extend(gpu_job_cells(lanes));
        cells
    }
}

#[test]
fn every_mlx_only_cell_is_an_approved_exception() {
    let lanes = residual::Lanes::new();
    let residual: BTreeSet<String> = residual::all_cells(&lanes)
        .into_iter()
        .filter(|(_, (mlx, candle))| *mlx && !*candle)
        .map(|(cell, _)| cell)
        .collect();
    let approved: BTreeSet<String> = crate::registry::approved_backend_exceptions()
        .iter()
        .flat_map(|record| record.cells.iter().cloned())
        .collect();
    let unapproved: Vec<_> = residual.difference(&approved).collect();
    let no_longer_residual: Vec<_> = approved.difference(&residual).collect();
    assert!(
        unapproved.is_empty() && no_longer_residual.is_empty(),
        "the approved exception register must equal the typed MLX-only residual set.\n\
         MLX-only with no approved product decision: {unapproved:#?}\n\
         approved but no longer MLX-only: {no_longer_residual:#?}"
    );
}

/// The two-lane answers the retired matrix's regression tests pinned, re-asserted on the typed cell
/// evaluator above (typed descriptors joined with the production claim gate). Each row names the
/// retired test it came from in `docs/.../sw7032-matrix-test-disposition.tsv`.
#[test]
fn former_matrix_cells_keep_their_two_lane_answers() {
    const BOTH: (bool, bool) = (true, true);
    const NEITHER: (bool, bool) = (false, false);
    const MLX_ONLY: (bool, bool) = (true, false);
    const CANDLE_ONLY: (bool, bool) = (false, true);
    let mut expected: Vec<(String, (bool, bool))> = Vec::new();
    let mut add = |cell: String, answer: (bool, bool)| expected.push((cell, answer));

    // qwen_image_2_1_adapters_are_served_on_both_registered_lanes
    for adapter in ["lora", "lokr"] {
        add(
            format!("models/qwen_image_2_1/userAdapters/{adapter}"),
            BOTH,
        );
    }
    // sc19059_resolved_candle_conditioning_is_not_left_in_the_residual_register
    for model in ["sdxl", "realvisxl", "realvisxl_lightning"] {
        add(format!("models/{model}/conditioningShape/control"), BOTH);
    }
    // sc18476_conditioning_cells_match_production_routes
    for model in [
        "sensenova_u1_8b",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3_fast",
        "qwen_image_edit_2511",
        "qwen_image_edit_2511_lightning",
        "flux2_dev",
    ] {
        for shape in ["reference", "multiReference"] {
            add(format!("models/{model}/conditioningShape/{shape}"), BOTH);
        }
    }
    for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
        for operation in ["edit_image", "character_image"] {
            add(format!("models/{model}/operationAndMode/{operation}"), BOTH);
        }
        // sc18477_bespoke_qwen_edit_adapter_cells_match_production_routes
        for adapter in ["lora", "lokr"] {
            add(format!("models/{model}/userAdapters/{adapter}"), BOTH);
        }
    }
    // sc18481_existing_candle_image_lanes_are_not_descriptor_false_negatives
    for (model, shape) in [
        ("bernini_image", "reference"),
        ("flux_dev", "reference"),
        ("flux_schnell", "reference"),
        ("sdxl", "reference"),
        ("sdxl", "mask"),
        ("realvisxl", "reference"),
        ("realvisxl", "mask"),
        ("illustrious_xl_v1", "reference"),
        ("illustrious_xl_v1", "mask"),
        ("illustrious_xl_v2", "reference"),
        ("illustrious_xl_v2", "mask"),
        ("kolors", "control"),
    ] {
        add(format!("models/{model}/conditioningShape/{shape}"), BOTH);
    }
    // sc18481_z_image_edit_and_removed_no_op_style_claims_use_real_product_shapes
    add(
        "models/z_image_edit/operationAndMode/image_to_image".into(),
        BOTH,
    );
    add(
        "models/wan_2_2_i2v_14b/operationAndMode/first_last_frame".into(),
        NEITHER,
    );
    // sc18481_cross_platform_catalog_rows_retain_both_native_text_routes
    for model in [
        "flux2_klein_9b",
        "flux2_dev",
        "sd3_5_large",
        "sd3_5_medium",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ] {
        add(
            format!("models/{model}/operationAndMode/text_to_image"),
            BOTH,
        );
    }
    // sc18481_bespoke_identity_routes_are_derived_from_the_live_lane_tables
    for model in ["instantid_realvisxl", "pulid_flux_dev"] {
        add(
            format!("models/{model}/operationAndMode/character_image"),
            BOTH,
        );
        add(format!("models/{model}/conditioningShape/reference"), BOTH);
    }
    add("models/pulid_flux_dev/preview".into(), BOTH);
    // sc20739_strict_pose_control_uses_only_real_product_lanes
    for model in [
        "z_image_turbo",
        "z_image",
        "qwen_image",
        "flux_dev",
        "krea_2_turbo",
        "flux2_dev",
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        add(format!("models/{model}/conditioningShape/control"), BOTH);
    }
    for model in [
        "qwen_image_edit_2511_lightning",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        add(format!("models/{model}/conditioningShape/control"), NEITHER);
    }
    // sc18481_cfg_pp_has_an_exact_guidance_axis_and_live_specialist_owner
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        add(format!("models/{model}/guidanceMethod/cfg"), BOTH);
        add(format!("models/{model}/guidanceMethod/cfg_pp"), MLX_ONLY);
    }
    // sc18481_bernini_preserves_candle_only_adapters_without_global_ui_exposure
    for model in ["bernini", "bernini_image"] {
        for adapter in ["lora", "lokr"] {
            add(
                format!("models/{model}/userAdapters/{adapter}"),
                CANDLE_ONLY,
            );
        }
    }
    // advanced_video_gpu_rows_use_models_that_advertise_each_operation +
    // audit_regressions_are_derived_from_production_truth
    add(
        "models/flux2_dev/operationAndMode/prompt_enhancement".into(),
        BOTH,
    );
    for model in ["ltx_2_3", "ltx_2_3_eros", "ltx_2_5"] {
        for mode in ["extend_clip", "video_bridge", "replace_person"] {
            add(
                format!("models/{model}/operationAndMode/{mode}"),
                if model == "ltx_2_3_eros" {
                    MLX_ONLY
                } else {
                    BOTH
                },
            );
        }
    }
    add(
        "models/wan_2_2_vace_fun_14b/operationAndMode/replace_person".into(),
        BOTH,
    );
    for shape in ["controlClip", "reference"] {
        add(
            format!("models/wan_2_2_vace_fun_14b/conditioningShape/{shape}"),
            BOTH,
        );
    }
    for adapter in ["lora", "lokr"] {
        add(
            format!("models/wan_2_2_vace_fun_14b/userAdapters/{adapter}"),
            BOTH,
        );
        add(format!("models/scail2_14b/userAdapters/{adapter}"), BOTH);
    }
    add(
        "models/wan_2_2_vace_fun_14b/precisionTier/bf16".into(),
        BOTH,
    );
    for tier in ["q4", "q8"] {
        add(
            format!("models/wan_2_2_vace_fun_14b/precisionTier/{tier}"),
            MLX_ONLY,
        );
    }
    for (shape, answer) in [
        ("reference", NEITHER),
        ("multiReference", BOTH),
        ("videoClip", BOTH),
    ] {
        add(format!("models/bernini/conditioningShape/{shape}"), answer);
    }
    for model in [
        "bernini",
        "bernini_image",
        "ltx_2_3",
        "sana_1600m",
        "sana_sprint_1600m",
        "scail2_14b",
    ] {
        add(format!("models/{model}/precisionTier/bf16"), BOTH);
    }
    for model in ["bernini", "bernini_image"] {
        for tier in ["q4", "q8"] {
            add(format!("models/{model}/precisionTier/{tier}"), BOTH);
        }
    }
    add(
        "models/krea_2_turbo/precisionTier/int8-convrot".into(),
        CANDLE_ONLY,
    );
    for tier in ["q4", "q8"] {
        add(format!("models/mage_flow_edit/precisionTier/{tier}"), BOTH);
    }
    for shape in ["reference", "multiReference"] {
        add(
            format!("models/mage_flow_edit/conditioningShape/{shape}"),
            BOTH,
        );
    }
    for model in ["sana_1600m", "sana_sprint_1600m"] {
        add(
            format!("models/{model}/operationAndMode/image_to_image"),
            BOTH,
        );
        add(format!("models/{model}/conditioningShape/reference"), BOTH);
    }
    add("trainingKernels/mage_flow_base_lora/full".into(), BOTH);
    add("gpuJobTypes/image_upscale/engine:aura-sr".into(), NEITHER);
    for engine in ["real-esrgan", "seedvr2"] {
        add(format!("gpuJobTypes/image_upscale/engine:{engine}"), BOTH);
    }
    // qwen_image_2_1_training_target_is_supported_on_both_backends_at_the_pin
    for target in ["qwen_image_2_1_lora", "qwen_image_2_1_edit_lora"] {
        for kind in ["lora", "lokr"] {
            add(format!("trainingKernels/{target}/{kind}"), BOTH);
        }
    }
    // terminal_vector_install_is_projected_from_native_worker_capabilities
    add("gpuJobTypes/vector_generate/image_to_svg".into(), BOTH);

    let lanes = residual::Lanes::new();
    let cells = residual::all_cells(&lanes);
    // InstantID's pose drives its landmark ControlNet on both lanes; the product declares no
    // `controlModes` for it, so the cell is asserted through the production lanes directly.
    let pose = serde_json::json!({"mode":"character_image","referenceAssetId":"face","advanced":{"poses":[{}]}});
    for backend in ["mlx", "candle"] {
        assert!(
            image_claim("instantid_realvisxl", pose.clone(), backend),
            "InstantID strict pose on {backend}"
        );
    }
    let mut instantid = super::canonical_video_route_probe("wan_2_2", "text_to_video").unwrap();
    instantid.job_type = crate::contracts::JobType::ImageGenerate;
    let mut payload = pose.as_object().unwrap().clone();
    payload.insert("model".into(), serde_json::json!("instantid_realvisxl"));
    instantid.payload = payload;
    assert_eq!(
        super::candle::image_job_candle_lane(&instantid),
        Some(super::candle::CandleImageLane::InstantId)
    );
    // bespoke_conditioning_authority_is_exact_and_route_backed: the plural character request
    // reaches the bespoke Qwen-Edit lane only for the edit model, never for its text-to-image
    // sibling, and a blank reference is not a conditioned request on either FLUX.2 lane.
    let candle_lane = |model: &str, fields: serde_json::Value| {
        let mut job = super::canonical_video_route_probe("wan_2_2", "text_to_video").unwrap();
        job.job_type = crate::contracts::JobType::ImageGenerate;
        let mut payload = fields.as_object().unwrap().clone();
        payload.insert("model".into(), serde_json::json!(model));
        job.payload = payload;
        super::candle::image_job_candle_lane(&job)
    };
    let plural = serde_json::json!({"mode":"character_image","referenceAssetIds":["a","b"]});
    assert_eq!(
        candle_lane("qwen_image_edit_2511", plural.clone()),
        Some(super::candle::CandleImageLane::QwenEdit)
    );
    assert_ne!(
        candle_lane("qwen_image", plural.clone()),
        Some(super::candle::CandleImageLane::QwenEdit)
    );
    assert!(!image_claim("qwen_image", plural, "candle"));
    for backend in ["mlx", "candle"] {
        assert!(!image_claim(
            "flux2_dev",
            serde_json::json!({"mode":"character_image","referenceAssetId":" "}),
            backend
        ));
    }
    // sc18481_supported_import_defaults_are_provenanced_and_use_live_family_routes: per family,
    // which lanes serve an imported checkpoint, its conditioning, and its adapters. A source shape
    // registered on either lane is probed on both, so a lane that registers no route refuses.
    let imported = |family: &str, backend: &str, extra: serde_json::Value| {
        let sources: BTreeSet<String> = ["mlx", "candle"]
            .into_iter()
            .flat_map(|lane| super::catalog::imported_provider_routes(lane, family))
            .filter(|route| route.operation == "generate")
            .map(|route| route.source.clone())
            .collect();
        assert!(!sources.is_empty(), "{family} registers an imported route");
        sources.into_iter().any(|source| {
            let mut entry = serde_json::Map::new();
            crate::lora_family::apply_model_manifest_defaults(&mut entry, "image", Some(family));
            entry.insert("id".into(), serde_json::json!("user_import_probe"));
            entry.insert("family".into(), serde_json::json!(family));
            entry.insert("importSourceShape".into(), serde_json::json!(source));
            entry.insert(
                "paths".into(),
                serde_json::json!({"model":"/tmp/registered-import"}),
            );
            let mut payload = serde_json::json!({
                "model": "user_import_probe",
                "mode": "text_to_image",
                "modelManifestEntry": entry
            });
            for (key, value) in extra.as_object().unwrap() {
                payload[key] = value.clone();
            }
            super::catalog::imported_image_request_provider_eligible(
                "user_import_probe",
                payload.as_object().unwrap(),
                backend,
            )
        })
    };
    for (family, serves) in [
        ("krea_2", BOTH),
        ("sdxl", BOTH),
        ("mage-flow", MLX_ONLY),
        ("flux2", CANDLE_ONLY),
    ] {
        let answer = |extra: serde_json::Value| {
            (
                imported(family, "mlx", extra.clone()),
                imported(family, "candle", extra),
            )
        };
        assert_eq!(
            answer(serde_json::json!({})),
            serves,
            "imported {family} text"
        );
        let adapters = if matches!(family, "krea_2" | "sdxl") {
            BOTH
        } else {
            (serves.0 && family == "flux2", serves.1 && family == "flux2")
        };
        for adapter in ["lora", "lokr"] {
            assert_eq!(
                answer(serde_json::json!({"loras":[{"id":"probe","networkType":adapter}]})),
                adapters,
                "imported {family}/{adapter}"
            );
        }
    }
    for fields in [
        serde_json::json!({"mode":"edit_image","sourceAssetId":"probe"}),
        serde_json::json!({"mode":"edit_image","sourceAssetId":"probe","referenceAssetIds":["a","b"]}),
    ] {
        for backend in ["mlx", "candle"] {
            assert!(
                imported("krea_2", backend, fields.clone()),
                "imported krea_2 conditioned edit on {backend}: {fields}"
            );
        }
    }
    let wrong: Vec<String> = expected
        .iter()
        .filter_map(|(cell, answer)| {
            let actual = cells.get(cell);
            (actual != Some(answer)).then(|| format!("{cell}: expected {answer:?}, got {actual:?}"))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "former matrix cells changed their two-lane answer:\n  {}",
        wrong.join("\n  ")
    );
}

/// Every operation a shipped image model advertises has a canonical production request, so its
/// two-lane answer is evaluated rather than silently skipped.
#[test]
fn every_advertised_image_operation_has_an_evaluated_cell() {
    let lanes = residual::Lanes::new();
    let cells = residual::model_cells(&lanes, crate::registry::builtin_model_specs());
    let mut checked = 0;
    for spec in
        crate::registry::builtin_model_specs().filter(|spec| spec.product.model_type == "image")
    {
        for operation in spec
            .product
            .fields
            .get("capabilities")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
        {
            let cell = format!("models/{}/operationAndMode/{operation}", spec.product.id);
            assert!(
                cells.contains_key(&cell),
                "{cell} has no canonical production request"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "only {checked} advertised image operations were evaluated"
    );
}

/// The catalog's MLX verdict for a builtin is the claim gate's: every shipped image and video
/// builtin carries a route-policy row, and its `macSupport.supported` equals that row's MLX
/// admission. A builtin with no row would otherwise read one way in the catalog and be refused by
/// the router, leaving its job queued unclaimed.
#[test]
fn builtin_mac_support_is_exactly_the_route_policy_admission() {
    let mut checked = 0;
    for spec in crate::registry::builtin_model_specs() {
        let kind = spec.product.model_type.as_str();
        if !matches!(kind, "image" | "video") {
            continue;
        }
        let policy = spec
            .routing
            .as_ref()
            .unwrap_or_else(|| panic!("{} has no route-policy row", spec.product.id));
        assert_eq!(
            super::catalog::model_mac_support(&spec.product.id, kind, Some(&spec.product.family))
                .supported,
            policy.mlx_routed(),
            "{} catalog MLX verdict differs from its route policy",
            spec.product.id
        );
        checked += 1;
    }
    assert!(
        checked > 60,
        "only {checked} image/video builtins were checked"
    );
}
