// Imported dense MiniMax-H3 DiT (sc-26680, epic 26679): import → plan → worker → provider seam.
//
// Each test commits a REAL managed import through the plan store, builds the entry the import job
// stamps (`minimax_h3_import_fixture`, pinned to the job by
// `tests::lora_and_downloads::managed_h3_dit_import_stamps_a_selectable_video_entry`), and drives the
// REAL arm against a stub loader, asserting on the `LoadSpec` that reached the provider. The
// controls remove exactly one thing — the binding, or the imported id — and must stop the arm before
// any load.
#![cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]

use super::*;

/// A text-to-video request for the imported entry under `model`.
fn imported_minimax_h3_request(
    model: &str,
    entry: &serde_json::Map<String, Value>,
) -> VideoRequest {
    request(json!({
        "projectId": "p",
        "model": model,
        "mode": "text_to_video",
        "prompt": "a courier crosses a rain-lit square",
        "duration": 6.0,
        "fps": 24,
        "modelManifestEntry": entry,
    }))
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn without_assembly(entry: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let mut unbound = entry.clone();
    unbound["importPlan"]
        .as_object_mut()
        .expect("importPlan object")
        .remove("assembly");
    unbound
}

#[cfg(target_os = "macos")]
#[test]
fn an_imported_h3_dit_reaches_the_mlx_provider_as_its_transformer_with_its_identity() {
    let tiers = minimax_h3_tier_root("mm_imp_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_imp_base_");
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        &minimax_h3_data_dir(tiers.path()),
        "singularity_fl2va_dense",
    );
    let request = imported_minimax_h3_request("singularity_fl2va_dense", &imported.entry);
    let probe = ArmProbe::default();

    let (_decoded, raw) = drive_minimax_h3_arm(tiers.path(), base.path(), &probe, &request)
        .expect("an imported dense DiT with a resident install renders");

    let spec = probe.spec.lock().unwrap().clone().expect("a load ran");
    let dir = |source: Option<&WeightsSource>| match source {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        other => panic!("expected a directory source, got {other:?}"),
    };
    let transformer = dir(spec.components.get("transformer"));
    assert_eq!(
        canonical(&transformer),
        canonical(&imported.install_dir),
        "the provider's transformer component must BE the plan's DiT directory"
    );
    assert_ne!(
        canonical(&transformer),
        canonical(&tiers.path().join("q4").join("transformer")),
        "the resident builtin DiT must not be substituted for the import"
    );
    // The companion the binding records is real at the sibling the provider probes.
    assert_eq!(
        canonical(&transformer.with_file_name("transformer_ref")),
        canonical(&tiers.path().join("q4").join("transformer_ref"))
    );
    assert_eq!(dir(Some(&spec.weights)), base.path().to_path_buf());
    assert_eq!(
        dir(spec.components.get("text_encoder")),
        tiers.path().join("q4").join("text_encoder"),
        "the resident text encoder is reused"
    );
    assert_eq!(spec.quantize, None, "the imported DiT is dense");
    // The base partition's lattice, not the Wan stride.
    assert_eq!(probe.engine_frames(), 158);

    // Identity: the imported id, its plan and its digest — no builtin id anywhere.
    assert_eq!(raw["model"], json!("singularity_fl2va_dense"));
    assert_eq!(raw["minimaxH3Tier"], json!("imported"));
    assert_eq!(
        raw["importPlan"]["checkpointId"],
        json!(imported.checkpoint_id)
    );
    assert_eq!(raw["importPlan"]["planId"], json!(imported.plan_id));
    assert_eq!(
        raw["importPlan"]["semanticDigest"],
        json!(imported.semantic_digest)
    );
    assert_eq!(
        raw["importPlan"]["assembly"],
        imported.entry["importPlan"]["assembly"]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn an_imported_h3_prefixed_id_routes_and_loads_its_plan_weights() {
    let tiers = minimax_h3_tier_root("mm_prefix_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_prefix_base_");
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        &minimax_h3_data_dir(tiers.path()),
        "minimax_h3_custom",
    );
    let request = imported_minimax_h3_request("minimax_h3_custom", &imported.entry);
    let settings = Settings {
        data_dir: minimax_h3_data_dir(tiers.path()),
        ..offline_settings()
    };
    assert_eq!(
        resolve_video_route(&request, &settings),
        VideoRoute::MiniMaxH3("minimax_h3")
    );
    let probe = ArmProbe::default();
    let (_, raw) = drive_minimax_h3_arm(tiers.path(), base.path(), &probe, &request)
        .expect("a user id sharing the builtin prefix is still an imported checkpoint");
    let spec = probe
        .spec
        .lock()
        .unwrap()
        .clone()
        .expect("a provider load ran");
    let Some(WeightsSource::Dir(transformer)) = spec.components.get("transformer") else {
        panic!("the imported transformer must be staged");
    };
    assert_eq!(canonical(transformer), canonical(&imported.install_dir));
    assert_eq!(raw["model"], json!("minimax_h3_custom"));
    assert_eq!(
        raw["importPlan"]["checkpointId"],
        json!(imported.checkpoint_id)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn an_imported_h3_entry_without_its_binding_or_under_a_builtin_id_never_loads() {
    let tiers = minimax_h3_tier_root("mm_imp_neg_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_imp_neg_base_");
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        &minimax_h3_data_dir(tiers.path()),
        "singularity_fl2va_dense",
    );

    let probe = ArmProbe::default();
    let error = drive_minimax_h3_arm(
        tiers.path(),
        base.path(),
        &probe,
        &imported_minimax_h3_request(
            "singularity_fl2va_dense",
            &without_assembly(&imported.entry),
        ),
    )
    .map(|(_d, raw)| raw)
    .expect_err("an unbound import must be refused");
    assert!(
        error
            .to_string()
            .contains("[checkpoint-plan:ambiguous-component]"),
        "{error}"
    );
    assert!(
        probe.spec.lock().unwrap().is_none(),
        "refused before any load"
    );

    let probe = ArmProbe::default();
    let error = drive_minimax_h3_arm(
        tiers.path(),
        base.path(),
        &probe,
        &imported_minimax_h3_request("minimax_h3", &imported.entry),
    )
    .map(|(_d, raw)| raw)
    .expect_err("the import must not render under the builtin id");
    assert!(error.to_string().contains("builtin id"), "{error}");
    assert!(
        probe.spec.lock().unwrap().is_none(),
        "refused before any load"
    );

    // The macOS router sends the imported entry to this arm, never the stub or a builtin arm.
    let settings = Settings {
        data_dir: minimax_h3_data_dir(tiers.path()),
        ..offline_settings()
    };
    assert_eq!(
        resolve_video_route(
            &imported_minimax_h3_request("singularity_fl2va_dense", &imported.entry),
            &settings
        ),
        VideoRoute::MiniMaxH3("minimax_h3")
    );
    let mut wan = imported.entry.clone();
    wan.insert("family".to_owned(), json!("wan-video"));
    assert_eq!(
        resolve_video_route(
            &imported_minimax_h3_request("singularity_fl2va_dense", &wan),
            &settings
        ),
        VideoRoute::NoPlanLane,
        "a plan-backed entry with no MLX lane is refused, never stubbed"
    );
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn drive_candle_imported_minimax_h3(
    settings: &Settings,
    probe: &ArmProbe,
    request: &VideoRequest,
) -> WorkerResult<Value> {
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-candle-imported-h3", "type": "video_generate", "status": "running",
        "projectId": null, "projectName": null, "payload": {}, "result": {},
        "requestedGpu": "auto", "assignedGpu": null, "workerId": "test-worker", "progress": 0.0,
        "stage": "queued", "message": "queued", "error": null, "etaSeconds": null,
        "elapsedSeconds": null, "attempts": 1, "sourceJobId": null, "duplicateOfJobId": null,
        "cancelRequested": false, "createdAt": "2026-10-22T00:00:00Z",
        "updatedAt": "2026-10-22T00:00:00Z", "startedAt": null, "completedAt": null,
        "canceledAt": null, "lastHeartbeatAt": null
    }))
    .expect("job snapshot");
    let project_path = settings.data_dir.join("unused-project");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(
            crate::video_jobs::minimax_h3::generate_candle_minimax_h3_using(
                &ApiClient::new(settings),
                settings,
                &job,
                request,
                &project_path,
                "minimax_h3",
                "candle",
                probe.loader(),
            ),
        )
        .map(|(_decoded, raw)| raw)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn an_imported_h3_dit_reaches_the_candle_provider_as_its_transformer_with_its_identity() {
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    const REVISION: &str = "1111111111111111111111111111111111111111";
    let data = tempfile::Builder::new()
        .prefix("sw_candle_imported_h3_")
        .tempdir()
        .expect("temp dir");
    let snapshot = |repo: &str| {
        let root = huggingface_repo_cache_path(data.path(), repo)
            .expect("repo cache path")
            .join("snapshots")
            .join(REVISION);
        std::fs::create_dir_all(&root).expect("snapshot dir");
        root
    };
    let upstream = snapshot("MiniMaxAI/MiniMax-H3");
    let rehost = snapshot("SceneWorks/minimax-h3-mlx");
    for component in [
        upstream.join("vae"),
        upstream.join("audio_vae"),
        upstream.join("text_encoder"),
        rehost.join("q4").join("text_encoder"),
        rehost.join("q4").join("transformer"),
    ] {
        std::fs::create_dir_all(&component).expect("component");
        std::fs::write(component.join("config.json"), b"{}").expect("config");
    }
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        data.path(),
        "singularity_fl2va_dense",
    );
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        backend_candle_enabled: true,
        ..offline_settings()
    };

    let request = imported_minimax_h3_request("singularity_fl2va_dense", &imported.entry);
    assert!(
        resolve_candle_video_route_for_test(&request, &settings).is_ok(),
        "the candle router claims the imported entry"
    );
    let probe = ArmProbe::default();
    let raw = drive_candle_imported_minimax_h3(&settings, &probe, &request)
        .expect("an imported dense DiT with a resident install renders");
    let spec = probe.spec.lock().unwrap().clone().expect("a load ran");
    let Some(WeightsSource::Dir(transformer)) = spec.components.get("transformer") else {
        panic!("the imported DiT must ride components[\"transformer\"]");
    };
    assert_eq!(canonical(transformer), canonical(&imported.install_dir));
    assert_ne!(
        canonical(transformer),
        canonical(&rehost.join("q4").join("transformer"))
    );
    assert!(matches!(&spec.weights, WeightsSource::Dir(root) if *root == upstream));
    assert_eq!(spec.quantize, None);
    assert_eq!(raw["model"], json!("singularity_fl2va_dense"));
    assert_eq!(
        raw["importPlan"]["checkpointId"],
        json!(imported.checkpoint_id)
    );
    assert_eq!(raw["importPlan"]["planId"], json!(imported.plan_id));
    assert_eq!(
        raw["importPlan"]["semanticDigest"],
        json!(imported.semantic_digest)
    );

    // Controls: the binding removed, or the builtin id substituted, never reach the loader.
    let unbound = without_assembly(&imported.entry);
    for (model, entry, needle) in [
        ("singularity_fl2va_dense", &unbound, "ambiguous-component"),
        ("minimax_h3", &imported.entry, "builtin id"),
    ] {
        let probe = ArmProbe::default();
        let error = drive_candle_imported_minimax_h3(
            &settings,
            &probe,
            &imported_minimax_h3_request(model, entry),
        )
        .expect_err("refused");
        assert!(error.to_string().contains(needle), "{model}: {error}");
        assert!(
            probe.spec.lock().unwrap().is_none(),
            "{model}: refused before any load"
        );
    }
}
