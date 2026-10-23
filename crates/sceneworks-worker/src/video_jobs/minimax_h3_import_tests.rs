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
    crate::minimax_h3_import_fixture::write_shared_configs(
        base.path(),
        &tiers.path().join("q4/text_encoder"),
    );
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
    crate::minimax_h3_import_fixture::write_shared_configs(
        base.path(),
        &tiers.path().join("q4/text_encoder"),
    );
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
    crate::minimax_h3_import_fixture::write_shared_configs(
        base.path(),
        &tiers.path().join("q4/text_encoder"),
    );
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
    crate::minimax_h3_import_fixture::write_shared_configs(
        &upstream,
        &rehost.join("q4/text_encoder"),
    );
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

#[cfg(target_os = "macos")]
#[test]
fn imported_declared_partitions_keep_their_weights_for_every_operation() {
    for partition in ["fl2va", "ref2va", "hybrid"] {
        let companions: &[&str] = match partition {
            "fl2va" => &["transformer_ref"],
            "ref2va" => &["transformer"],
            _ => &[],
        };
        let tiers = minimax_h3_tier_root("mm_contract_", &["q4"], companions);
        let base = minimax_h3_base_root("mm_contract_base_");
        crate::minimax_h3_import_fixture::write_shared_configs(
            base.path(),
            &tiers.path().join("q4/text_encoder"),
        );
        let data = minimax_h3_data_dir(tiers.path());
        let imported =
            crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_with_partition(
                &data,
                "minimax_h3_custom",
                partition,
            );
        let (project_id, project_path, image_id) = stage_minimax_h3_reference_project(&data);
        // The provider cache deliberately reuses one generator across this checkpoint's modes.
        // Keep one positive probe attached to that generator, not a fresh uncalled loader per mode.
        let positive_probe = ArmProbe {
            audio: Some(gen_core::AudioTrack {
                samples: vec![0.125; 32],
                sample_rate: 32000,
                channels: 2,
                stems: Vec::new(),
            }),
            ..Default::default()
        };
        for mode in [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "reference_to_video",
        ] {
            let mut payload = json!({"projectId":project_id,"model":"minimax_h3_custom","modelManifestEntry":imported.entry,
                "mode":mode,"prompt":"a courier","duration":6.0,"fps":24});
            match mode {
                "image_to_video" => payload["sourceAssetId"] = json!(image_id),
                "first_last_frame" => {
                    payload["sourceAssetId"] = json!(image_id);
                    payload["lastFrameAssetId"] = json!(image_id);
                }
                "reference_to_video" => payload["referenceAssetIds"] = json!([image_id]),
                _ => {}
            }
            let request = request(payload);
            let supported =
                partition == "hybrid" || (partition == "ref2va") == (mode == "reference_to_video");
            let refused_probe = ArmProbe::default();
            let probe = if supported {
                &positive_probe
            } else {
                &refused_probe
            };
            let result = drive_minimax_h3_arm_in_project(
                tiers.path(),
                base.path(),
                probe,
                &request,
                &project_path,
            );
            if !supported {
                let error = result.map(|(_, raw)| raw).unwrap_err();
                assert!(
                    error.to_string().contains("unsupported-operation"),
                    "{partition}/{mode}: {error}"
                );
                assert!(probe.spec.lock().unwrap().is_none());
                continue;
            }
            let (decoded, raw) =
                result.unwrap_or_else(|error| panic!("{partition}/{mode}: {error}"));
            assert_eq!(
                decoded.audio.unwrap().samples,
                vec![0.125; 32],
                "synchronized output audio is retained"
            );
            let spec = probe
                .spec
                .lock()
                .unwrap()
                .clone()
                .expect("native load reached");
            let Some(WeightsSource::Dir(transformer)) = spec.components.get("transformer") else {
                panic!("staged transformer");
            };
            let selected = if mode == "reference_to_video" {
                transformer.with_file_name("transformer_ref")
            } else {
                transformer.clone()
            };
            assert_eq!(
                canonical(&selected),
                canonical(&imported.install_dir),
                "{partition}/{mode} must denoise on the imported checkpoint"
            );
            assert_eq!(raw["model"], "minimax_h3_custom");
            assert_eq!(
                raw["importPlan"]["assembly"],
                imported.entry["importPlan"]["assembly"]
            );
            if partition == "hybrid" {
                assert_eq!(
                    canonical(transformer),
                    canonical(&transformer.with_file_name("transformer_ref"))
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn incomplete_or_incompatible_imported_shared_components_never_reach_provider_load() {
    let tiers = minimax_h3_tier_root(
        "mm_components_",
        &["q4"],
        &["transformer", "transformer_ref"],
    );
    let base = minimax_h3_base_root("mm_components_base_");
    let encoder = tiers.path().join("q4/text_encoder");
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        &minimax_h3_data_dir(tiers.path()),
        "component_test",
    );
    for file in [
        encoder.join("config.json"),
        base.path().join("vae/config.json"),
        base.path().join("audio_vae/config.json"),
        base.path().join("FL2VA/audio_vae/metadata.json"),
        base.path().join("tokenizer/tokenizer.json"),
        base.path().join("tokenizer/tokenizer_config.json"),
    ] {
        for missing in [true, false] {
            crate::minimax_h3_import_fixture::write_shared_configs(base.path(), &encoder);
            if missing {
                std::fs::remove_file(&file).unwrap();
            } else {
                std::fs::write(&file, b"{}").unwrap();
            }
            let probe = ArmProbe::default();
            let result = drive_minimax_h3_arm(
                tiers.path(),
                base.path(),
                &probe,
                &imported_minimax_h3_request("component_test", &imported.entry),
            );
            assert!(
                result.is_err(),
                "{} must refuse, missing={missing}",
                file.display()
            );
            assert!(
                probe.spec.lock().unwrap().is_none(),
                "incomplete assembly never reaches load"
            );
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_imported_partition_staging_matches_the_task_selected_sibling() {
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    let data = tempfile::tempdir().unwrap();
    let snapshot = |repo: &str| {
        let root = huggingface_repo_cache_path(data.path(), repo)
            .unwrap()
            .join("snapshots")
            .join("1111111111111111111111111111111111111111");
        std::fs::create_dir_all(&root).unwrap();
        root
    };
    let root = snapshot("MiniMaxAI/MiniMax-H3");
    let rehost = snapshot("SceneWorks/minimax-h3-mlx");
    std::fs::create_dir_all(rehost.join("q4/transformer")).unwrap();
    std::fs::write(rehost.join("q4/transformer/config.json"), b"{}").unwrap();
    crate::minimax_h3_import_fixture::write_shared_configs(&root, &rehost.join("q4/text_encoder"));
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        backend_candle_enabled: true,
        ..offline_settings()
    };
    for partition in ["fl2va", "ref2va", "hybrid"] {
        let imported =
            crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_with_partition(
                data.path(),
                &format!("candle_{partition}"),
                partition,
            );
        for references in [false, true] {
            let mut request =
                imported_minimax_h3_request(&format!("candle_{partition}"), &imported.entry);
            if references {
                request.mode = "reference_to_video".to_owned();
            }
            let dit =
                crate::video_jobs::minimax_h3::resolve_imported_minimax_h3_dit(&request, &settings)
                    .unwrap();
            let supported = partition == "hybrid" || (partition == "ref2va") == references;
            if !supported {
                continue;
            }
            let (load, _) = crate::video_jobs::minimax_h3::resolve_candle_imported_minimax_h3_load(
                &settings, &request, &dit,
            )
            .unwrap();
            let transformer = load.dit_dir.unwrap();
            let selected = if references {
                transformer.with_file_name("transformer_ref")
            } else {
                transformer
            };
            // Windows uses snapshot-style hardlinks; compare the exact bytes instead of paths.
            assert_eq!(
                std::fs::read(selected.join("diffusion_pytorch_model.safetensors")).unwrap(),
                std::fs::read(
                    imported
                        .install_dir
                        .join("diffusion_pytorch_model.safetensors")
                )
                .unwrap()
            );
            assert_eq!(load.quant, None);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn imported_companion_config_mismatches_are_refused_not_defaulted() {
    let tiers = minimax_h3_tier_root("mm_mismatch_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_mismatch_base_");
    let encoder = tiers.path().join("q4/text_encoder");
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        &minimax_h3_data_dir(tiers.path()),
        "mismatch_test",
    );
    for (file, pointer, value) in [
        (
            encoder.join("config.json"),
            "/text_config/hidden_size",
            json!(4096),
        ),
        (
            base.path().join("vae/config.json"),
            "/temporal_downsample_factors",
            json!([1, 1, 1, 1, 1, 1]),
        ),
        (
            base.path().join("audio_vae/config.json"),
            "/sampling_rate",
            json!(16000),
        ),
        (
            base.path().join("tokenizer/tokenizer.json"),
            "/added_tokens/0/id",
            json!(42),
        ),
        (
            base.path().join("tokenizer/tokenizer_config.json"),
            "/additional_special_tokens/0",
            json!("</d>"),
        ),
    ] {
        crate::minimax_h3_import_fixture::write_shared_configs(base.path(), &encoder);
        let mut config: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        *config.pointer_mut(pointer).unwrap() = value;
        std::fs::write(&file, config.to_string()).unwrap();
        let probe = ArmProbe::default();
        let error = drive_minimax_h3_arm(
            tiers.path(),
            base.path(),
            &probe,
            &imported_minimax_h3_request("mismatch_test", &imported.entry),
        )
        .map(|(_, raw)| raw)
        .unwrap_err();
        assert!(
            error.to_string().contains("incompatible-component"),
            "{}: {error}",
            file.display()
        );
        assert!(probe.spec.lock().unwrap().is_none());
    }
}
