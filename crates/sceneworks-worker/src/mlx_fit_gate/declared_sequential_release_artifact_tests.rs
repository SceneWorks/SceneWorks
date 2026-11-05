use super::*;
use crate::memory_route_registry::{MemoryRouteMode, MemoryRouteRequestContext};

fn no_staging(admission: &MlxRequestAdmission) -> bool {
    matches!(admission, MlxRequestAdmission::Admitted(evaluation)
            if !evaluation.memory.stage_residency && !evaluation.memory.stream_transformer_blocks)
}

#[test]
#[ignore = "requires installed flux2_klein_9b bf16, converted flux2_klein_9b_true_v2 and flux2_dev bf16 on a 128 GiB Mac"]
fn declared_sequential_release_matches_the_request_admission() {
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    ))
    .unwrap();
    let home = PathBuf::from(std::env::var_os("HOME").unwrap());
    let hub = home.join(".cache/huggingface/hub");
    let cases = [
            (
                "flux2_klein_9b",
                "flux2_klein_9b",
                hub.join("models--SceneWorks--flux2-klein-9b-mlx/snapshots/acf05e8d5103838baba6a5e32dc91d6997a56023/bf16"),
                true,
            ),
            (
                "flux2_klein_9b",
                "flux2_klein_9b_true_v2",
                home.join("SceneWorks/data/models/mlx/flux2_klein_9b_true_v2"),
                true,
            ),
            // sc-23187: Dev's resident pipeline does not fit the request gate, so it keeps staging.
            (
                "flux2_dev",
                "flux2_dev",
                hub.join("models--SceneWorks--flux2-dev-mlx/snapshots/2868b1461b2b6e6e05d84e52534df3632b4c7d5d/bf16"),
                false,
            ),
        ];
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for (engine, route, dir, expect_release) in cases {
        assert!(dir.is_dir(), "missing {}", dir.display());
        let model = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == route)
            .unwrap()
            .as_object()
            .unwrap();
        let mode = MemoryRouteMode::TextToImage;
        let context = MemoryRouteRequestContext {
            mode,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
        };
        let supports_sequential = crate::inference_runtime::media_descriptor(engine)
            .unwrap()
            .capabilities
            .supports_sequential_offload;
        let before = LoadSpec::new(WeightsSource::Dir(dir.clone())).with_resolved_route(route);
        let declared = crate::image_jobs::prepare_mlx_load_policy(
            engine,
            Some("bf16"),
            Some(mode),
            model,
            before.clone(),
            context,
            true,
            supports_sequential,
        )
        .unwrap();
        let budget = runtime
            .block_on(crate::generator_cache::mlx_tier_budget(engine))
            .unwrap();
        let admit = |candidate: &LoadSpec| {
            let plan =
                MlxRequestPlan::for_spec_and_manifest(engine, route, candidate, Some(model), None)
                    .with_resolved_artifact_tier(Some("bf16"))
                    .unwrap();
            let inputs = MlxRequestInputs {
                native_batch: 1,
                width: 1024,
                height: 1024,
                count: 2,
                mode: "text_to_image".to_owned(),
                overlay: provider_overlay_for_load_spec(engine, candidate, None),
                adapter_count: 0,
                has_reference: false,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                conditioning_windows: None,
                negative_branch: false,
                prefix_kv_cache_eligible: false,
                reference_resources: None,
            };
            preflight_request(candidate, &plan, &inputs, budget).unwrap()
        };
        let alternative =
            crate::image_jobs::declared_sequential_resident_alternative(engine, &before, &declared);
        let released = alternative.as_ref().is_some_and(|resident| {
            no_staging(&admit(&declared))
                && no_staging(&admit(resident))
                && preflight_load_rejection(engine, resident).is_none()
        });
        eprintln!(
            "[sc-24806] {route}: declared offload={:?} alternative={} released={released}",
            declared.offload_policy,
            alternative.is_some()
        );
        assert_eq!(
            declared.offload_policy,
            OffloadPolicy::Sequential,
            "{route}"
        );
        assert_eq!(released, expect_release, "{route}");
    }
}
