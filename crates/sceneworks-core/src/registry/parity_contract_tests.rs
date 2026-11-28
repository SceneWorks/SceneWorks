//! Regression checks at the compiled registry boundary against canonical `sga.capabilities/2`
//! exports (`SW-6983`), without a second capability matrix.
use super::BUILTINS;
use crate::sga_capabilities::{CapabilitiesExportV2, ExportedRouteV2};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn exported_route_to_generator_json(route: &ExportedRouteV2) -> Value {
    let mut features = serde_json::Map::new();
    for feature in &route.features {
        if feature.feature_id == "low_res_start" && feature.applicable {
            features.insert(
                feature.feature_id.clone(),
                serde_json::to_value(feature).unwrap(),
            );
        }
    }
    let mut obj = serde_json::Map::new();
    obj.insert("id".into(), json!(route.route_id));
    obj.insert("family".into(), json!(route.family));
    obj.insert("backend".into(), json!(route.backend));
    obj.insert("modality".into(), json!(route.modality));
    obj.insert("conditioning".into(), json!(route.conditioning.kinds));
    obj.insert(
        "supported_quants".into(),
        json!(route.quant.supported_tiers),
    );
    obj.insert("supports_lora".into(), json!(route.adapters.supports_lora));
    obj.insert("supports_lokr".into(), json!(route.adapters.supports_lokr));
    obj.insert(
        "supports_preview".into(),
        json!(route.capabilities.supports_preview),
    );
    obj.insert(
        "supports_prompt_enhancement".into(),
        json!(route.capabilities.supports_prompt_enhancement),
    );
    obj.insert("samplers".into(), json!(route.solvers));
    obj.insert("schedulers".into(), json!(route.schedules));
    obj.insert(
        "supported_guidance_methods".into(),
        json!(route.guidance.methods),
    );
    if let Some(pag) = &route.guidance.pag {
        obj.insert("pag".into(), serde_json::to_value(pag).unwrap());
    }
    if !features.is_empty() {
        obj.insert("features".into(), Value::Object(features));
    }
    Value::Object(obj)
}

fn native_generators() -> BTreeMap<(String, String), Value> {
    let mut routes = BTreeMap::new();
    for (backend, export) in [
        ("mlx", crate::sga_capabilities::macos_mlx()),
        ("candle", crate::sga_capabilities::candle_cuda()),
    ] {
        let export: &CapabilitiesExportV2 = export;
        assert_eq!(export.backend, backend);
        for route in export.routes.values().chain(export.audio_routes.values()) {
            // The MLX export carries Candle audio routes for host discovery.
            // Only routes tagged for this backend can enter its compiled route.
            if route.backend != backend {
                continue;
            }
            assert!(routes
                .insert(
                    (backend.into(), route.route_id.clone()),
                    exported_route_to_generator_json(route)
                )
                .is_none());
        }
    }
    routes
}

fn assert_native_route(
    route: &impl serde::Serialize,
    backend: &str,
    native: &BTreeMap<(String, String), Value>,
) {
    let actual = serde_json::to_value(route).unwrap();
    let id = actual["id"].as_str().unwrap();
    assert_eq!(actual["backend"], backend);
    assert_eq!(
        native.get(&(backend.into(), id.into())),
        Some(&actual),
        "descriptor axis drift in {backend}/{id}"
    );
}

#[test]
fn every_shipped_generator_route_retains_all_native_descriptor_axes() {
    let native = native_generators();
    let mut image_routes = 0;
    let mut video_engines = 0;
    let mut provider_generators = 0;
    for (id, spec) in &*BUILTINS {
        assert_eq!(&spec.product.id, id);
        for (backend, route) in &spec.routes {
            assert_native_route(route, backend, &native);
            image_routes += 1;
        }
        for (backend, modes) in &spec.video_routes {
            let mut seen = BTreeSet::new();
            for mode in modes {
                assert!(
                    seen.insert(&mode.mode),
                    "duplicate {id}/{backend}/{} mode",
                    mode.mode
                );
                assert!(
                    !mode.engines.is_empty(),
                    "empty {id}/{backend}/{} mode",
                    mode.mode
                );
                for engine in &mode.engines {
                    assert_native_route(engine, backend, &native);
                    video_engines += 1;
                }
            }
        }
        for (backend, providers) in &spec.provider_routes {
            for provider in providers {
                if let Some(generator) = &provider.generator {
                    assert_native_route(generator, backend, &native);
                    provider_generators += 1;
                }
            }
        }
    }
    assert!(image_routes > 50 && video_engines > 10 && provider_generators > 0);
}

#[test]
fn public_video_operations_are_backed_by_compiled_native_modes() {
    let mut video_models = 0;
    for spec in BUILTINS
        .values()
        .filter(|spec| spec.product.model_type == "video")
    {
        video_models += 1;
        let capabilities = spec.product.fields["capabilities"].as_array().unwrap();
        assert!(
            !capabilities.is_empty(),
            "{} has no operations",
            spec.product.id
        );
        for capability in capabilities {
            let operation = capability.as_str().unwrap();
            assert!(
                spec.video_routes
                    .values()
                    .flatten()
                    .any(|mode| mode.mode == operation),
                "{}/{} has no native video mode",
                spec.product.id,
                operation
            );
        }
    }
    assert!(video_models > 10);
}
