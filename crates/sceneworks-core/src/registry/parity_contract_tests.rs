//! Regression checks at the compiled registry boundary, without a second capability matrix.
use super::BUILTINS;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn native_generators() -> BTreeMap<(String, String), Value> {
    let mut routes = BTreeMap::new();
    for (backend, source) in [
        (
            "mlx",
            include_str!("../../../../config/engine-capabilities/runtime/capabilities.mlx.json"),
        ),
        (
            "candle",
            include_str!("../../../../config/engine-capabilities/runtime/capabilities.candle.json"),
        ),
    ] {
        let dump: Value = serde_json::from_str(source).unwrap();
        assert_eq!(dump["snapshot"]["backend"], backend);
        for route in dump["snapshot"]["generator_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .chain(
                dump["snapshot"]["audio_generator_capabilities"]
                    .as_array()
                    .unwrap(),
            )
        {
            let id = route["id"].as_str().unwrap();
            // The MLX dump carries Candle audio inventory for host discovery.
            // Only descriptors tagged for this backend can enter its compiled route.
            if route["backend"] != backend {
                continue;
            }
            assert!(routes
                .insert((backend.into(), id.into()), route.clone())
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
