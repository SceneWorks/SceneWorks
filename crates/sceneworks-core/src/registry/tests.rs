use super::*;
use serde_json::json;
const PRODUCTS: &str = include_str!("../../../../config/manifests/builtin.models.jsonc");
fn product_json() -> String {
    crate::jsonc::strip_jsonc_comments(PRODUCTS)
}
const EVIDENCE: &str = include_str!("../../../../config/manifests/registry/evidence.json");
const POLICY: &str = include_str!("../../../../config/manifests/registry/route-policy.json");
const PREVIEW: &str = include_str!("../../../../config/manifests/builtin.preview-support.jsonc");
fn preview_json() -> String {
    crate::jsonc::strip_jsonc_comments(PREVIEW)
}
const MLX: &str =
    include_str!("../../../../config/engine-capabilities/runtime/capabilities.mlx.json");
const CANDLE: &str =
    include_str!("../../../../config/engine-capabilities/runtime/capabilities.candle.json");

#[test]
fn every_builtin_memory_fact_is_projected_from_validated_evidence() {
    let raw: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(include_str!(
        "../../../../config/manifests/builtin.models.jsonc"
    )))
    .unwrap();
    assert_eq!(raw["schemaVersion"], 2);
    super::evidence::validate_evidence_document(EVIDENCE, &raw).unwrap();
    let effective = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
    assert_eq!(raw["models"].as_array().unwrap().len(), 143);
    assert_eq!(effective["models"].as_array().unwrap().len(), 143);
    let evidence: Value = serde_json::from_str(EVIDENCE).unwrap();
    let mut mlx_count = 0;
    let mut candle_count = 0;
    for (original, projected) in raw["models"]
        .as_array()
        .unwrap()
        .iter()
        .zip(effective["models"].as_array().unwrap())
    {
        assert_eq!(original["id"], projected["id"]);
        let id = original["id"].as_str().unwrap();
        let mut once = original.clone();
        project_builtin_memory_evidence(&mut once).unwrap();
        let mut twice = once.clone();
        project_builtin_memory_evidence(&mut twice).unwrap();
        assert_eq!(once, twice, "{id}: evidence projection must be idempotent");
        if once != *projected {
            let differing: Vec<_> = once
                .as_object()
                .unwrap()
                .keys()
                .filter(|key| once[*key] != projected[*key])
                .cloned()
                .collect();
            panic!("{id}: effective model differs from evidence projection in {differing:?}");
        }
        for backend in ["mlx", "candle"] {
            if original.get(backend).is_some() {
                if backend == "mlx" {
                    mlx_count += 1;
                } else {
                    candle_count += 1;
                }
            }
            if let Some(facts) = evidence[id].get(backend).and_then(Value::as_object) {
                for (key, value) in facts {
                    assert!(
                        original[backend].get(key).is_none(),
                        "{id}/{backend}.{key} leaked into product"
                    );
                    assert_eq!(&projected[backend][key], value);
                }
            }
        }
        for row in original["downloads"].as_array().unwrap_or(&Vec::new()) {
            if let Some(footprint) = row.get("footprint") {
                for key in ["residentMemoryBytes", "peakMemoryBytes", "measuredPixels"] {
                    assert!(
                        footprint.get(key).is_none(),
                        "{id} product download has {key}"
                    );
                }
            }
        }
    }
    assert_eq!((mlx_count, candle_count), (75, 86));
    let mut malformed = evidence.clone();
    malformed["flux_schnell"]["mlx"]["unknownMemoryKey"] = json!(1);
    assert!(super::evidence::validate_evidence_document(&malformed.to_string(), &raw).is_err());
    let mut ambiguous = evidence.clone();
    let rows = ambiguous["flux3_klein"]["downloads"]
        .as_array_mut()
        .unwrap();
    let row = rows
        .iter_mut()
        .find(|row| row.get("platforms").is_some())
        .unwrap();
    row.as_object_mut().unwrap().remove("platforms");
    assert!(super::evidence::validate_evidence_document(&ambiguous.to_string(), &raw).is_err());
}

#[test]
fn both_shipping_routes_and_the_v1_compatibility_projection_are_preserved() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    assert_eq!(spec.routes["mlx"].id, "flux1_schnell");
    assert_eq!(spec.routes["mlx"].conditioning, ["reference"]);
    assert_eq!(spec.routes["mlx"].supported_quants, ["q4", "q8"]);
    assert!(spec.routes["candle"].supported_quants.is_empty());
    assert!(spec.routes["candle"].conditioning.is_empty());
    let effective = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
    let legacy = effective["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == "flux_schnell")
        .unwrap();
    assert_eq!(
        &spec.project(None),
        legacy,
        "effective readers must see the same public facts and exact calibration identities"
    );
    assert_eq!(
        crate::builtin_manifests::builtin_model_entry("flux_schnell"),
        legacy.as_object()
    );
    assert!(builtin_model_spec("flux_dev").is_some());
}

#[test]
fn author_can_override_product_choices_but_cannot_replace_native_or_evidence_authority() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    let overlay = json!({
        "name":"Family favorite","hidden":true,"ui":{"label":"Custom","pid":null,"promptGuide":{"path":"/tmp"}},
        "defaults":{"steps":8,"guidanceScale":99,"count":999,"sampler":"heun","resolution":"1024x1024"},
        "id":"alias","family":"sdxl","adapter":"external","downloads":[],"gated":true,"paths":{"model":"/tmp"},
        "limits":{"samplers":[]},"image":{"supportsGuidance":true},"mlx":{"minMemoryGb":0},"candle":{"calibrations":[]},
        "modelPath":"/tmp", "installed":true,"catalogScope":"user"
    });
    let mut expected = spec.project(None);
    expected["name"] = json!("Family favorite");
    expected["hidden"] = json!(true);
    expected["ui"]["label"] = json!("Custom");
    expected["defaults"]["steps"] = json!(8);
    expected["defaults"]["sampler"] = json!("heun");
    assert_eq!(spec.project(Some(&overlay)), expected);
}

#[test]
fn descriptor_contents_change_the_spec_while_provenance_does_not() {
    let mut mlx: Value = serde_json::from_str(MLX).unwrap();
    let build = |mlx: &Value| {
        assemble::assemble(
            &product_json(),
            EVIDENCE,
            POLICY,
            &preview_json(),
            &[("mlx", &mlx.to_string()), ("candle", CANDLE)],
        )
        .unwrap()
    };
    let before = build(&mlx);
    mlx["generatedFrom"]["inferenceRevision"] = json!("unrelated-pin");
    assert_eq!(
        serde_json::to_value(&before).unwrap(),
        serde_json::to_value(build(&mlx)).unwrap()
    );
    let desc = mlx["snapshot"]["generator_capabilities"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["id"] == "flux1_schnell")
        .unwrap();
    desc["supports_preview"] = json!(false);
    desc["samplers"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry != "heun");
    let changed = build(&mlx);
    let spec = &changed["flux_schnell"];
    assert!(!spec.routes["mlx"].supports_preview);
    assert!(spec.routes["candle"].supports_preview);
    assert!(!spec.product.fields["limits"]["samplers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|menu| menu == "heun"));
    let mut evidence: Value = serde_json::from_str(EVIDENCE).unwrap();
    evidence["flux_schnell"]["candle"]["calibrations"][0]["provider"] = json!("other");
    assert!(
        assemble::assemble(
            &product_json(),
            &evidence.to_string(),
            POLICY,
            &preview_json(),
            &[("mlx", MLX), ("candle", CANDLE)]
        )
        .is_ok(),
        "existing calibration identities are evidence, not a primary-route alias"
    );
}

#[test]
fn shipped_snapshot_matches_the_build_time_join() {
    let shipped = include_str!("../../../../config/manifests/registry/specs.json").trim_end();
    let compiled = include_str!(concat!(env!("OUT_DIR"), "/model-registry.json"));
    assert_eq!(
        compiled, shipped,
        "refresh with cargo run -q -p sceneworks-core --bin dump-model-registry"
    );
}

#[test]
fn user_menus_only_narrow_native_product_choices_and_reconcile_defaults() {
    let spec = builtin_model_spec("flux_schnell").unwrap();
    let row = spec.project(Some(&json!({
        "limits": {"resolutions": ["768x768"], "count": [1, 2], "samplers": ["unshipped"], "schedulers": ["karras"]},
        "defaults": {"resolution": "1024x1024", "count": 2, "sampler": "unshipped"}
    })));
    assert_eq!(row["defaults"]["resolution"], "768x768");
    assert_eq!(row["defaults"]["count"], 2);
    assert_eq!(row["defaults"]["scheduler"], "karras");
    assert_eq!(
        row["limits"]["samplers"],
        spec.project(None)["limits"]["samplers"]
    );
    assert!(row["defaults"].get("sampler").is_none());
}

#[test]
fn mutable_distribution_revision_cannot_enter_the_static_catalog() {
    let mut products: Value = serde_json::from_str(&product_json()).unwrap();
    let flux = products["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["id"] == "flux_schnell")
        .unwrap();
    flux["downloads"][0]["revision"] = json!("main");
    let result = assemble::assemble(
        &products.to_string(),
        EVIDENCE,
        POLICY,
        &preview_json(),
        &[("mlx", MLX), ("candle", CANDLE)],
    );
    assert!(result.unwrap_err().contains("valid download"));
}

#[test]
fn all_frozen_builtin_rows_have_exact_typed_product_and_evidence_projection() {
    let effective = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
    let rows = effective["models"].as_array().unwrap();
    assert_eq!(rows.len(), 143);
    assert_eq!(super::BUILTINS.len(), rows.len());
    for row in rows {
        let id = row["id"].as_str().unwrap();
        let spec = builtin_model_spec(id).unwrap_or_else(|| panic!("missing typed spec for {id}"));
        let projected = spec.project(None);
        if projected != *row {
            let fields: Vec<_> = row
                .as_object()
                .unwrap()
                .keys()
                .filter(|field| row[*field] != projected[*field])
                .collect();
            let candle: Vec<_> = row["candle"]
                .as_object()
                .into_iter()
                .flat_map(|fields| fields.keys())
                .filter(|field| row["candle"][*field] != projected["candle"][*field])
                .collect();
            panic!("{id}: public product/evidence drift in {fields:?}, candle {candle:?}");
        }
    }
}

#[test]
fn compiled_spec_evidence_is_the_projection_authority() {
    let mut spec = builtin_model_spec("flux_schnell").unwrap().clone();
    spec.evidence.get_mut("candle").unwrap().min_memory_gb = Some(99);
    assert_eq!(spec.project(None)["candle"]["minMemoryGb"], 99);
    let footprint = spec.download_evidence[0]["footprint"]
        .as_object_mut()
        .unwrap();
    footprint.insert("residentMemoryBytes".to_owned(), json!(12345));
    let projected = spec.project(None);
    assert!(projected["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .any(|download| download["footprint"]["residentMemoryBytes"] == 12345));
}

#[test]
fn all_public_generator_and_provider_families_have_native_projections() {
    let models = &super::BUILTINS;
    let count = |model_type: &str, has_route: fn(&ModelSpec) -> bool| {
        models
            .values()
            .filter(|spec| spec.product.model_type == model_type && has_route(spec))
            .count()
    };
    assert_eq!(count("image", |spec| !spec.routes.is_empty()), 59);
    assert_eq!(count("video", |spec| !spec.video_routes.is_empty()), 13);
    assert_eq!(count("audio", |spec| !spec.provider_routes.is_empty()), 15);
    assert_eq!(count("vector", |spec| !spec.provider_routes.is_empty()), 2);
    assert_eq!(
        models["starvector_1b"].provider_routes["mlx"][0].id,
        "mlx-starvector-1b"
    );
    assert_eq!(
        models["starvector_1b"].provider_routes["candle"][0].id,
        "candle-starvector-1b"
    );
    assert_eq!(
        models["ltx_2_5"].video_routes["mlx"]
            .iter()
            .map(|route| route.mode.as_str())
            .collect::<Vec<_>>(),
        models["ltx_2_5"].video_routes["candle"]
            .iter()
            .map(|route| route.mode.as_str())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn yue2_tiers_require_the_candle_audio_export() {
    let spec = builtin_model_spec("yue2").expect("shipped audio model");
    let tiers: std::collections::BTreeSet<_> = spec.product.fields["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["variant"].as_str())
        .collect();
    assert_eq!(tiers, ["bf16", "q4", "q8"].into_iter().collect());
    let candle_audio: Value = serde_json::from_str(include_str!(
        "../../../../config/engine-capabilities/audio/capabilities.candle.json"
    ))
    .unwrap();
    assert_eq!(candle_audio["backend"], "candle");
    assert!(candle_audio["engines"]
        .as_array()
        .unwrap()
        .iter()
        .any(|row| row["id"] == "yue2" && row["modality"] == "audio"));
    assert!(spec.provider_routes.contains_key("candle"));
    // The generic runtime export also mentions YuE2 on MLX. The typed public projection
    // correctly withholds that provider because only Candle registers the audio engine.
    assert!(!spec.provider_routes.contains_key("mlx"));
}

#[test]
fn starvector_install_exposes_only_image_to_svg_on_both_native_lanes() {
    for id in ["starvector_1b", "starvector_8b"] {
        let spec = builtin_model_spec(id).expect("shipped vector model");
        assert_eq!(spec.product.model_type, "vector");
        assert_eq!(spec.product.fields["capabilities"], json!(["image_to_svg"]));
        for backend in ["mlx", "candle"] {
            assert_eq!(spec.provider_routes[backend].len(), 1, "{id}/{backend}");
            assert!(spec.provider_routes[backend][0].id.starts_with(backend));
        }
    }
}

#[test]
fn pid_decoder_overlays_are_installable_without_standalone_provider_routes() {
    for id in ["pid_qwenimage", "pid_flux", "pid_flux2", "pid_sdxl"] {
        let spec = builtin_model_spec(id).expect("shipped decoder overlay");
        assert_eq!(spec.product.model_type, "utility");
        assert!(
            spec.provider_routes.is_empty(),
            "{id} is loaded by its owning image model"
        );
        assert!(spec.routes.is_empty(), "{id} is not an image generator");
    }
}

#[test]
fn router_policy_and_preview_feature_export_keep_the_frozen_population() {
    let policy = super::policy::parse(POLICY).unwrap();
    assert_eq!((policy.image.len(), policy.video.len()), (63, 14));
    let aliases: Vec<_> = policy
        .image
        .iter()
        .filter(|row| row.legacy_alias)
        .map(|row| row.id.as_str())
        .chain(
            policy
                .video
                .iter()
                .filter(|row| row.legacy_alias)
                .map(|row| row.id.as_str()),
        )
        .collect();
    assert_eq!(
        aliases,
        ["qwen_image_edit", "qwen_image_edit_2509", "mochi_1"]
    );
    for id in policy
        .image
        .iter()
        .filter(|row| !row.legacy_alias)
        .map(|row| row.id.as_str())
        .chain(
            policy
                .video
                .iter()
                .filter(|row| !row.legacy_alias)
                .map(|row| row.id.as_str()),
        )
    {
        assert!(builtin_model_spec(id).unwrap().routing.is_some(), "{id}");
    }

    let source: Value = serde_json::from_str(&preview_json()).unwrap();
    let preview = source["models"].as_object().unwrap();
    assert_eq!(preview.len(), 83);
    for (id, lanes) in preview {
        let exported = super::preview_feature(id).unwrap_or_else(|| panic!("missing {id} preview"));
        assert_eq!(
            serde_json::to_value(exported).unwrap(),
            *lanes,
            "{id} preview changed"
        );
        if let Some(spec) = builtin_model_spec(id) {
            for (backend, route) in &spec.routes {
                if let Some(advertised) = exported.get(backend) {
                    assert_eq!(*advertised, route.supports_preview, "{id}/{backend}");
                }
            }
        }
    }
    assert_eq!(super::LEGACY_PREVIEW.len(), 8);
}
