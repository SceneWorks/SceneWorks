//! Cross-check typed native video mappings against the actual claim predicate. This replaces the
//! generated capability matrix's video join guard without digesting source files.
use super::{
    candle::video_mode_is_candle_eligible, catalog::VIDEO_UI_MODES,
    mlx::video_mode_is_mlx_eligible, video_mode_conditioning_requirements,
};
use crate::registry::builtin_model_spec;
use std::collections::BTreeSet;

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
