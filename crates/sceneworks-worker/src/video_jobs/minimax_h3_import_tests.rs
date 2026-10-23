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
use crate::video_jobs::minimax_h3::with_imported_h3_source_seals;

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

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
/// Budget for a stub-loader request with no adapters or sparse workspace: exactly the existing
/// measured dense floor plus the production reserve. This is fixture input, not device evidence.
fn imported_h3_fixture_budget(
    entry: &serde_json::Map<String, Value>,
) -> crate::vram_gate::VramBudget {
    let free_gb = entry["candle"]["vramGbByTier"]["bf16"]
        .as_f64()
        .expect("fixture declares the dense admission floor")
        + crate::vram_gate::HEADROOM_GB;
    crate::vram_gate::VramBudget {
        free_gb,
        total_gb: free_gb,
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn imported_h3_admission_requires_a_known_sufficient_budget() {
    let data = tempfile::tempdir().unwrap();
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
        data.path(),
        "admission_fixture",
    );
    let sufficient = imported_h3_fixture_budget(&imported.entry);
    let gate = |budget| {
        crate::vram_gate::minimax_h3_fit_error_with_sparse(
            &imported.entry,
            "bf16",
            0,
            "fixture-only",
            budget,
            None,
        )
    };
    let missing = gate(None).expect("missing probe must refuse").to_string();
    assert!(
        missing.contains("could not read free GPU VRAM"),
        "{missing}"
    );
    let insufficient = gate(Some(crate::vram_gate::VramBudget {
        free_gb: 0.0,
        ..sufficient
    }))
    .expect("insufficient budget must refuse")
    .to_string();
    assert!(
        insufficient.contains("needs ~") && insufficient.contains("free GPU VRAM"),
        "{insufficient}"
    );
    assert!(
        gate(Some(sufficient)).is_none(),
        "the exact measured floor and reserve must fit"
    );
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
    assert_eq!(
        canonical(&dir(Some(&spec.weights)).join("vae")),
        canonical(&base.path().join("vae"))
    );
    assert_eq!(
        canonical(&dir(spec.components.get("text_encoder"))),
        canonical(&tiers.path().join("q4").join("text_encoder")),
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
fn native_and_sharded_plans_reach_the_real_h3_provider_load_seam() {
    for layout in ["native", "sharded"] {
        let tiers =
            minimax_h3_tier_root("mm_layout_", &["q4"], &["transformer", "transformer_ref"]);
        let base = minimax_h3_base_root("mm_layout_base_");
        crate::minimax_h3_import_fixture::write_shared_configs(
            base.path(),
            &tiers.path().join("q4/text_encoder"),
        );
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_layout(
            &minimax_h3_data_dir(tiers.path()),
            "arbitrarily_renamed_h3",
            layout,
        );
        let request = imported_minimax_h3_request("arbitrarily_renamed_h3", &imported.entry);
        let probe = ArmProbe::default();
        let (_, raw) = drive_minimax_h3_arm(tiers.path(), base.path(), &probe, &request)
            .expect("layout must load");
        let spec = probe
            .spec
            .lock()
            .unwrap()
            .clone()
            .expect("provider received LoadSpec");
        let WeightsSource::Dir(directory) = spec.components.get("transformer").unwrap() else {
            panic!("provider requires component directory")
        };
        let mut weights = 0;
        let mut files = Vec::new();
        for entry in std::fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "safetensors") {
                weights += 1;
            }
            files.push((entry.file_name().to_string_lossy().into_owned(), path));
        }
        let canonical_inventory =
            sceneworks_core::minimax_h3_codec::H3SourceInventory::from_files(&files)
                .expect("provider input is complete canonical tensors");
        assert!(
            !canonical_inventory.native,
            "native source was actually normalized, not merely recognized"
        );
        assert_eq!(weights, if layout == "native" { 1 } else { 2 });
        assert_eq!(raw["model"], json!("arbitrarily_renamed_h3"));
        assert_eq!(
            raw["importPlan"]["checkpointId"],
            json!(imported.checkpoint_id)
        );
        assert_eq!(
            raw["importPlan"]["semanticDigest"],
            json!(imported.semantic_digest)
        );
        assert_eq!(
            raw["importPlan"]["assembly"],
            imported.entry["importPlan"]["assembly"]
        );
        if layout == "native" {
            assert_ne!(canonical(directory), canonical(&imported.install_dir));
        } else {
            assert_eq!(canonical(directory), canonical(&imported.install_dir));
        }
    }
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
    budget: Option<crate::vram_gate::VramBudget>,
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
                std::future::ready(budget),
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
    use crate::minimax_h3_import_fixture::{SHARED_REVISION, TIER_REVISION};
    let data = tempfile::Builder::new()
        .prefix("sw_candle_imported_h3_")
        .tempdir()
        .expect("temp dir");
    let snapshot = |repo: &str| {
        let root = huggingface_repo_cache_path(data.path(), repo)
            .expect("repo cache path")
            .join("snapshots")
            .join(if repo == "SceneWorks/minimax-h3-mlx" {
                TIER_REVISION
            } else {
                SHARED_REVISION
            });
        std::fs::create_dir_all(&root).expect("snapshot dir");
        root
    };
    let upstream = snapshot("MiniMaxAI/MiniMax-H3");
    let rehost = snapshot("SceneWorks/minimax-h3-mlx");
    crate::minimax_h3_import_fixture::write_companion(&rehost.join("q4/transformer_ref"));
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
    let raw = drive_candle_imported_minimax_h3(
        &settings,
        &probe,
        &request,
        Some(imported_h3_fixture_budget(&imported.entry)),
    )
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

    // The real admission gate still rejects absent/insufficient budgets before invoking the loader.
    for (budget, needle) in [
        (None, "could not read free GPU VRAM"),
        (
            Some(crate::vram_gate::VramBudget {
                free_gb: 0.0,
                ..imported_h3_fixture_budget(&imported.entry)
            }),
            "needs ~",
        ),
    ] {
        let rejected = ArmProbe::default();
        let error = drive_candle_imported_minimax_h3(&settings, &rejected, &request, budget)
            .expect_err("the real gate must refuse before the stub loader");
        assert!(error.to_string().contains(needle), "{error}");
        assert!(
            rejected.spec.lock().unwrap().is_none(),
            "refused before provider loading"
        );
    }

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
            Some(imported_h3_fixture_budget(&imported.entry)),
        )
        .expect_err("refused");
        assert!(error.to_string().contains(needle), "{model}: {error}");
        assert!(
            probe.spec.lock().unwrap().is_none(),
            "{model}: refused before any load"
        );
    }
}

#[test]
fn source_mutation_before_or_during_normalization_cannot_publish() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.safetensors");
    std::fs::write(&source, b"original").unwrap();
    let pin = gen_core::PinnedWeightsFile::pin(&source).unwrap();
    assert!(with_imported_h3_source_seals(std::slice::from_ref(&pin), || Ok(())).is_ok());
    let error = with_imported_h3_source_seals(std::slice::from_ref(&pin), || {
        std::fs::write(&source, b"mutated!").unwrap();
        Ok(())
    })
    .unwrap_err();
    assert!(error.contains("seal mismatch"), "{error}");
    assert!(with_imported_h3_source_seals::<()>(&[pin], || panic!(
        "stale source must refuse before read"
    ))
    .is_err());
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
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_source(
            &data,
            "minimax_h3_custom",
            "dense",
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
fn linked_mixed_case_shards_reach_the_provider_as_lowercase_canonical_files() {
    let tiers = minimax_h3_tier_root("mm_upper_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_upper_base_");
    crate::minimax_h3_import_fixture::write_shared_configs(
        base.path(),
        &tiers.path().join("q4/text_encoder"),
    );
    let data = minimax_h3_data_dir(tiers.path());
    let original = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_layout(
        &data,
        "uppercase_h3",
        "sharded",
    );
    let library = tempfile::tempdir().unwrap();
    let source = library.path().join("linked-h3");
    std::fs::create_dir_all(&source).unwrap();
    for file in std::fs::read_dir(&original.install_dir).unwrap() {
        let file = file.unwrap();
        std::fs::copy(file.path(), source.join(file.file_name())).unwrap();
    }
    let index_file = source.join("diffusion_pytorch_model.safetensors.index.json");
    let mut index: Value =
        serde_json::from_reader(std::fs::File::open(&index_file).unwrap()).unwrap();
    for (old, new) in [
        ("renamed-part-0.safetensors", "part-0.SAFETENSORS"),
        ("renamed-part-1.safetensors", "part-1.SafeTensors"),
    ] {
        std::fs::rename(source.join(old), source.join(new)).unwrap();
        for shard in index["weight_map"].as_object_mut().unwrap().values_mut() {
            if shard.as_str() == Some(old) {
                *shard = json!(new);
            }
        }
    }
    std::fs::write(&index_file, index.to_string()).unwrap();
    let plans = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data);
    let root = plans.approve_root(library.path()).unwrap();
    let compiled = plans.compile_linked(&root.root_id, "linked-h3").unwrap();
    let mut entry = original.entry.clone();
    entry.insert("importPlan".into(),json!({"checkpointId":compiled.checkpoint_id,"planId":compiled.plan.plan_id,"semanticDigest":compiled.record.plan.semantic_digest}));
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
        &compiled.plan,
        &entry,
    )
    .unwrap();
    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    let request = imported_minimax_h3_request("uppercase_h3", &entry);
    let probe = ArmProbe::default();
    let (_, raw) = drive_minimax_h3_arm(tiers.path(), base.path(), &probe, &request).unwrap();
    let spec = probe.spec.lock().unwrap().clone().unwrap();
    let WeightsSource::Dir(directory) = &spec.components["transformer"] else {
        panic!("expected canonical directory")
    };
    let files: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                entry.path(),
            )
        })
        .collect();
    assert_eq!(
        files
            .iter()
            .filter(|(name, _)| name.ends_with(".safetensors"))
            .count(),
        2
    );
    assert!(
        !sceneworks_core::minimax_h3_codec::H3SourceInventory::from_files(&files)
            .unwrap()
            .requires_normalization
    );
    assert_eq!(
        raw["importPlan"]["checkpointId"],
        json!(compiled.checkpoint_id)
    );
    assert_ne!(canonical(directory), canonical(&source));
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
            .join(if repo == "SceneWorks/minimax-h3-mlx" {
                crate::minimax_h3_import_fixture::TIER_REVISION
            } else {
                crate::minimax_h3_import_fixture::SHARED_REVISION
            });
        std::fs::create_dir_all(&root).unwrap();
        root
    };
    let root = snapshot("MiniMaxAI/MiniMax-H3");
    let rehost = snapshot("SceneWorks/minimax-h3-mlx");
    std::fs::create_dir_all(rehost.join("q4/transformer")).unwrap();
    std::fs::write(rehost.join("q4/transformer/config.json"), b"{}").unwrap();
    crate::minimax_h3_import_fixture::write_companion(&rehost.join("q4/transformer"));
    crate::minimax_h3_import_fixture::write_companion(&rehost.join("q4/transformer_ref"));
    crate::minimax_h3_import_fixture::write_shared_configs(&root, &rehost.join("q4/text_encoder"));
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        backend_candle_enabled: true,
        ..offline_settings()
    };
    for partition in ["fl2va", "ref2va", "hybrid"] {
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_source(
            data.path(),
            &format!("candle_{partition}"),
            "dense",
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

#[cfg(target_os = "macos")]
#[test]
fn native_and_sharded_managed_and_linked_sources_preserve_explicit_partition_operations() {
    for layout in ["native", "sharded"] {
        for partition in ["fl2va", "ref2va", "hybrid"] {
            for linked in [false, true] {
                let companions: &[&str] = match partition {
                    "fl2va" => &["transformer_ref"],
                    "ref2va" => &["transformer"],
                    _ => &[],
                };
                let tiers = minimax_h3_tier_root("mm_combined_", &["q4"], companions);
                let base = minimax_h3_base_root("mm_combined_base_");
                crate::minimax_h3_import_fixture::write_shared_configs(
                    base.path(),
                    &tiers.path().join("q4/text_encoder"),
                );
                let data = minimax_h3_data_dir(tiers.path());
                let id = format!("combined_{layout}_{partition}_{linked}");
                let imported =
                    crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_source(
                        &data, &id, layout, partition,
                    );
                let mut entry = imported.entry.clone();
                let library = tempfile::tempdir().unwrap();
                if linked {
                    let source = library.path().join("arbitrary-checkpoint");
                    std::fs::create_dir_all(&source).unwrap();
                    for file in std::fs::read_dir(&imported.install_dir).unwrap() {
                        let file = file.unwrap();
                        std::fs::copy(file.path(), source.join(file.file_name())).unwrap();
                    }
                    let plans =
                        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data);
                    let root = plans.approve_root(library.path()).unwrap();
                    let compiled = plans
                        .compile_linked(&root.root_id, "arbitrary-checkpoint")
                        .unwrap();
                    entry.remove("paths");
                    entry.insert("importPlan".into(),json!({"checkpointId":compiled.checkpoint_id,"planId":compiled.plan.plan_id,"semanticDigest":compiled.record.plan.semantic_digest}));
                    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(&compiled.plan,&entry).unwrap();
                    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(
                        &mut entry, &assembly,
                    );
                }
                let (project_id, project_path, image_id) =
                    stage_minimax_h3_reference_project(&data);
                let positive = ArmProbe::default();
                let mut normalized = None;
                for mode in [
                    "text_to_video",
                    "image_to_video",
                    "first_last_frame",
                    "reference_to_video",
                ] {
                    let mut payload = json!({"projectId":project_id,"model":id,"modelManifestEntry":entry,"mode":mode,"prompt":"a courier","duration":6.0,"fps":24});
                    match mode {
                        "image_to_video" => payload["sourceAssetId"] = json!(image_id),
                        "first_last_frame" => {
                            payload["sourceAssetId"] = json!(image_id);
                            payload["lastFrameAssetId"] = json!(image_id);
                        }
                        "reference_to_video" => payload["referenceAssetIds"] = json!([image_id]),
                        _ => {}
                    }
                    let supported = partition == "hybrid"
                        || (partition == "ref2va") == (mode == "reference_to_video");
                    let rejected = ArmProbe::default();
                    let probe = if supported { &positive } else { &rejected };
                    let result = drive_minimax_h3_arm_in_project(
                        tiers.path(),
                        base.path(),
                        probe,
                        &request(payload),
                        &project_path,
                    );
                    if !supported {
                        assert!(result
                            .map(|(_, raw)| raw)
                            .unwrap_err()
                            .to_string()
                            .contains("unsupported-operation"));
                        assert!(probe.spec.lock().unwrap().is_none());
                        continue;
                    }
                    let (_, raw) = result.unwrap_or_else(|error| {
                        panic!("{layout}/{partition}/linked={linked}/{mode}: {error}")
                    });
                    let spec = probe
                        .spec
                        .lock()
                        .unwrap()
                        .clone()
                        .expect("provider load seam reached");
                    let WeightsSource::Dir(base_dit) = &spec.components["transformer"] else {
                        panic!("canonical component required")
                    };
                    let selected = if mode == "reference_to_video" {
                        base_dit.with_file_name("transformer_ref")
                    } else {
                        base_dit.clone()
                    };
                    let selected = canonical(&selected);
                    if let Some(expected) = &normalized {
                        assert_eq!(
                            &selected, expected,
                            "every supported operation uses the SAME imported/normalized bytes"
                        );
                    } else {
                        normalized = Some(selected.clone());
                    }
                    let files: Vec<_> = std::fs::read_dir(&selected)
                        .unwrap()
                        .map(|e| {
                            let e = e.unwrap();
                            (e.file_name().to_string_lossy().into_owned(), e.path())
                        })
                        .collect();
                    let inventory =
                        sceneworks_core::minimax_h3_codec::H3SourceInventory::from_files(&files)
                            .unwrap();
                    assert!(!inventory.native && !inventory.requires_normalization);
                    assert_eq!(raw["model"], json!(id));
                    assert_eq!(
                        raw["importPlan"]["assembly"],
                        entry["importPlan"]["assembly"]
                    );
                    assert_eq!(
                        raw["importPlan"]["checkpointId"],
                        entry["importPlan"]["checkpointId"]
                    );
                }
                // Identical tensor geometry does not supply an absent semantic declaration.
                let mut ambiguous = entry.clone();
                ambiguous.remove("minimaxH3Partition");
                let probe = ArmProbe::default();
                let error = drive_minimax_h3_arm(
                    tiers.path(),
                    base.path(),
                    &probe,
                    &imported_minimax_h3_request(&id, &ambiguous),
                )
                .map(|(_, raw)| raw)
                .unwrap_err();
                assert!(error.to_string().contains("minimaxH3Partition"), "{error}");
                assert!(probe.spec.lock().unwrap().is_none());
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn an_explicit_h3_declaration_cannot_relabel_a_linked_non_h3_source() {
    let tiers = minimax_h3_tier_root("mm_nonh3_", &["q4"], &["transformer", "transformer_ref"]);
    let base = minimax_h3_base_root("mm_nonh3_base_");
    let data = minimax_h3_data_dir(tiers.path());
    let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_source(
        &data,
        "declared_h3",
        "native",
        "hybrid",
    );
    let library = tempfile::tempdir().unwrap();
    crate::minimax_h3_import_fixture::write_safetensors(
        &library.path().join("not-h3.safetensors"),
        &[("model.diffusion_model.txtfusion.projector.weight", "BF16")],
        0x3a,
    );
    let plans = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data);
    let root = plans.approve_root(library.path()).unwrap();
    let compiled = plans
        .compile_linked(&root.root_id, "not-h3.safetensors")
        .unwrap();
    assert_eq!(compiled.plan.family, "krea_2");
    let mut entry = imported.entry;
    entry.remove("paths");
    entry.insert("importPlan".into(),json!({"checkpointId":compiled.checkpoint_id,"planId":compiled.plan.plan_id,"semanticDigest":compiled.record.plan.semantic_digest,"assembly":{"partition":"hybrid"}}));
    let probe = ArmProbe::default();
    let error = drive_minimax_h3_arm(
        tiers.path(),
        base.path(),
        &probe,
        &imported_minimax_h3_request("declared_h3", &entry),
    )
    .map(|(_, raw)| raw)
    .unwrap_err();
    assert!(error.to_string().contains("krea_2"), "{error}");
    assert!(probe.spec.lock().unwrap().is_none());
}

#[test]
fn provider_resolution_holds_dense_and_native_source_leases_until_release() {
    for layout in ["dense", "native", "sharded", "convrot"] {
        let data = tempfile::tempdir().unwrap();
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_layout(
            data.path(),
            "leased_h3",
            layout,
        );
        let settings = Settings {
            data_dir: data.path().to_path_buf(),
            ..offline_settings()
        };
        let request = imported_minimax_h3_request("leased_h3", &imported.entry);
        let provider =
            crate::video_jobs::minimax_h3::resolve_imported_minimax_h3_dit(&request, &settings)
                .unwrap();
        let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data.path());
        let refusal = store.remove_managed("leased_h3").unwrap_err();
        assert!(
            refusal.to_string().contains("in use"),
            "{layout}: {refusal}"
        );
        assert!(imported.install_dir.is_dir());
        assert!(store.resolve(&imported.checkpoint_id).is_ok());
        drop(provider);
        sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore::open(data.path())
            .unwrap()
            .remove_managed_install("leased_h3")
            .unwrap();
        assert!(!imported.install_dir.exists());
    }
}

#[test]
fn provider_stages_the_persisted_weight_only_config_and_source_owned_config_wins() {
    for source_owned in [false, true] {
        let data = tempfile::tempdir().unwrap();
        let mut imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import_layout(
            data.path(),
            "configured_h3",
            "dense",
        );
        let config_path = imported.install_dir.join("config.json");
        let own: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        let mut fallback = own.clone();
        fallback["rope_theta"] = json!(123456.0);
        fallback["provider_extension"] = json!({"trained_schedule":[0.125,0.75]});
        if !source_owned {
            std::fs::remove_file(&config_path).unwrap();
        }
        let context = [("transformerConfig".into(), sceneworks_core::minimax_h3_import::config_binding(fallback.clone(),
            json!({"kind":"selected-base","baseModel":"minimax_h3","repo":"fixture/config","revision":"1111111111111111111111111111111111111111","path":"bf16/transformer/config.json","componentTier":"bf16"})))].into();
        let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data.path());
        let compiled = store
            .compile_managed_with_context(
                "configured_h3",
                "diffusion_pytorch_model.safetensors",
                sceneworks_core::checkpoint_import::ManagedProvenanceV1 {
                    source: "fixture".into(),
                    ..Default::default()
                },
                context,
            )
            .unwrap();
        imported.entry["importPlan"] = json!({"checkpointId":compiled.checkpoint_id,"planId":compiled.plan.plan_id,"semanticDigest":compiled.plan.semantic_digest().unwrap()});
        let assembly =
            sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
                &compiled.plan,
                &imported.entry,
            )
            .unwrap();
        sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(
            &mut imported.entry,
            &assembly,
        );
        let settings = Settings {
            data_dir: data.path().to_path_buf(),
            ..offline_settings()
        };
        let request = imported_minimax_h3_request("configured_h3", &imported.entry);
        let provider =
            crate::video_jobs::minimax_h3::resolve_imported_minimax_h3_dit(&request, &settings)
                .unwrap();
        let actual: Value = serde_json::from_slice(
            &std::fs::read(provider.transformer_dir.join("config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(actual, if source_owned { own } else { fallback });
        assert!(
            store.remove_managed("configured_h3").is_err(),
            "configuration binding retains the source lease"
        );
    }
}

fn drive_bound_component_import(
    settings: &Settings,
    request: &VideoRequest,
    probe: &ArmProbe,
) -> WorkerResult<Value> {
    #[cfg(target_os = "macos")]
    {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(crate::video_jobs::minimax_h3::generate_minimax_h3_using(
                &ApiClient::new(settings),
                settings,
                &minimax_h3_job_snapshot(),
                request,
                &settings.data_dir.join("unused-project"),
                "minimax_h3",
                "mlx",
                probe.loader(),
            ))
            .map(|(_, raw)| raw)
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        drive_candle_imported_minimax_h3(
            settings,
            probe,
            request,
            Some(imported_h3_fixture_budget(&request.model_manifest_entry)),
        )
    }
}

#[test]
fn bound_shared_tier_and_exact_snapshots_reach_native_loadspec_despite_distracting_main() {
    use crate::minimax_h3_import_fixture::{
        component_downloads, write_companion, write_shared_configs, SHARED_REVISION, TIER_REVISION,
    };
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ]);
    for (tier, main_complete) in [("bf16", false), ("bf16", true), ("q4", false), ("q4", true)] {
        let data = tempfile::tempdir().unwrap();
        let snapshot = |repo: &str, revision: &str| {
            let cache = huggingface_repo_cache_path(data.path(), repo).unwrap();
            let dir = cache.join("snapshots").join(revision);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::create_dir_all(cache.join("refs")).unwrap();
            std::fs::write(
                cache.join("refs/main"),
                "3333333333333333333333333333333333333333",
            )
            .unwrap();
            dir
        };
        let base_a = snapshot("MiniMaxAI/MiniMax-H3", SHARED_REVISION);
        let tier_a = snapshot("SceneWorks/minimax-h3-mlx", TIER_REVISION);
        let base_b = snapshot(
            "MiniMaxAI/MiniMax-H3",
            "3333333333333333333333333333333333333333",
        );
        let tier_b = snapshot(
            "SceneWorks/minimax-h3-mlx",
            "3333333333333333333333333333333333333333",
        );
        let encoder_a = if tier == "bf16" {
            base_a.join("text_encoder")
        } else {
            tier_a.join(format!("{tier}/text_encoder"))
        };
        write_shared_configs(&base_a, &encoder_a);
        write_companion(&tier_a.join(format!("{tier}/transformer_ref")));
        std::fs::write(base_b.join("partial.txt"), b"mutable main is partial").unwrap();
        std::fs::write(tier_b.join("partial.txt"), b"mutable main is partial").unwrap();
        let imported = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
            data.path(),
            "bound_h3",
        );
        let mut entry = imported.entry.clone();
        entry.insert("importComponentTier".into(), json!(tier));
        entry.insert(
            "downloads".into(),
            json!(component_downloads(tier, "fl2va")),
        );
        let settings = Settings {
            data_dir: data.path().to_path_buf(),
            backend_candle_enabled: true,
            ..offline_settings()
        };
        if main_complete {
            let encoder_b = if tier == "bf16" {
                base_b.join("text_encoder")
            } else {
                tier_b.join(format!("{tier}/text_encoder"))
            };
            write_shared_configs(&base_b, &encoder_b);
            write_companion(&tier_b.join(format!("{tier}/transformer_ref")));
            crate::minimax_h3_import_fixture::write_safetensors(
                &base_b.join("vae/fixture.safetensors"),
                &[("weight", "F16")],
                0x7f,
            );
        }
        let probe = ArmProbe::default();
        for explicit in [None, Some(if tier == "bf16" { 0 } else { 4 })] {
            let mut request = imported_minimax_h3_request("bound_h3", &entry);
            if let Some(bits) = explicit {
                request.advanced.insert("mlxQuantize".into(), json!(bits));
            }
            let raw = drive_bound_component_import(&settings,&request,&probe).unwrap_or_else(|error| panic!("tier={tier}, explicit={explicit:?}, main_complete={main_complete}: {error}"));
            let spec = probe.spec.lock().unwrap().clone().unwrap();
            assert!(
                matches!(&spec.weights,WeightsSource::Dir(root) if canonical(root)==canonical(&base_a))
            );
            assert!(
                matches!(spec.components.get("text_encoder"),Some(WeightsSource::Dir(dir)) if canonical(dir)==canonical(&encoder_a))
            );
            let Some(WeightsSource::Dir(transformer)) = spec.components.get("transformer") else {
                panic!("missing imported transformer")
            };
            assert_eq!(canonical(transformer), canonical(&imported.install_dir));
            #[cfg(target_os = "macos")]
            assert_eq!(
                canonical(&transformer.with_file_name("transformer_ref")),
                canonical(&tier_a.join(format!("{tier}/transformer_ref")))
            );
            assert_eq!(
                spec.quantize, None,
                "shared tier cannot rewrite imported dense precision"
            );
            assert_eq!(raw["minimaxH3SharedTier"], tier);
            assert_eq!(
                raw["minimaxH3ComponentBindings"]["downloads"],
                entry["downloads"]
            );
        }
        let mut different = imported_minimax_h3_request("bound_h3", &entry);
        different.advanced.insert("mlxQuantize".into(), json!(8));
        let probe = ArmProbe::default();
        let error = drive_bound_component_import(&settings, &different, &probe).unwrap_err();
        assert!(
            error.to_string().contains("no component substitution"),
            "{error}"
        );
        assert!(probe.spec.lock().unwrap().is_none());
        std::fs::remove_file(base_a.join("vae/config.json")).unwrap();
        let probe = ArmProbe::default();
        assert!(
            drive_bound_component_import(
                &settings,
                &imported_minimax_h3_request("bound_h3", &entry),
                &probe
            )
            .is_err(),
            "a complete mutable main cannot rescue a torn bound snapshot"
        );
        assert!(probe.spec.lock().unwrap().is_none());
    }
}
