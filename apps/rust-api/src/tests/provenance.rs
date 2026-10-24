//! CPU-only real persistence/authority boundaries. No model loads, downloads, network, or GPU.
use super::support::*;
use sceneworks_core::checkpoint_import::{
    CheckpointCatalogRecordV1, CheckpointContainerV1, CheckpointInventoryV1, ImportLayerV1,
    ImportPlanV1, SourceLocatorV1,
};
use sceneworks_core::model_artifacts::provenance::{
    checkpoint_model_provenance, import_declaration_context,
};

#[tokio::test]
async fn inherited_component_acknowledgment_cannot_be_cleared_by_fine_tune_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let (_app, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let mut entry = json!({"id":"my-h3","family":"minimax-h3","baseModel":"minimax_h3_ref",
        "minimaxH3Partition":"ref2va","license":"MIT","requiresLicenseAcknowledgment":false,
        "inheritedLicenseAcknowledgments":[]})
    .as_object()
    .unwrap()
    .clone();
    let error =
        crate::models::ensure_inherited_component_licenses_acknowledged(&state, &entry, false)
            .await
            .unwrap_err();
    assert_eq!(error.status, StatusCode::FORBIDDEN);
    assert_eq!(error.code, Some("license_acknowledgment_required"));
    crate::models::ensure_inherited_component_licenses_acknowledged(&state, &entry, true)
        .await
        .unwrap();
    // False author flags and an empty author-supplied inherited set do not bypass the trusted
    // component set; an independently stored current acknowledgment is what clears the gate.
    entry.insert("licenseNotice".into(), json!("Author replacement"));
    crate::models::ensure_inherited_component_licenses_acknowledged(&state, &entry, false)
        .await
        .unwrap();
    let path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&state.settings.config_dir);
    sceneworks_core::license_acknowledgments::revoke_acknowledgment(&path, "minimax_h3").unwrap();
    let error =
        crate::models::ensure_inherited_component_licenses_acknowledged(&state, &entry, false)
            .await
            .unwrap_err();
    assert_eq!(error.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn provenance_api_handoff_persists_all_asset_modes_and_survives_restart_and_edits() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let data_dir = settings.data_dir.clone();
    let (_app, state) = create_app_with_state(settings.clone()).unwrap();
    let project = state
        .project_store
        .create_project("Provenance Smoke")
        .unwrap();
    let project_path = data_dir.join("projects/provenance-smoke.sceneworks");
    let mut plan = ImportPlanV1::new(
        format!("checkpoint-plan-{}", "1".repeat(64)),
        "ltx2",
        vec![ImportLayerV1 {
            layer_id: "dit".into(),
            role: "transformer".into(),
            target_path: "model.safetensors".into(),
            container: CheckpointContainerV1::Safetensors,
            source: SourceLocatorV1::linked("root-fixture", "model.safetensors", "a".repeat(64))
                .unwrap(),
        }],
    )
    .unwrap();
    plan.context = import_declaration_context(
        json!({"family":"ltx2","baseModel":"declared-base"})
            .as_object()
            .unwrap(),
    )
    .unwrap();
    let checkpoint_id = "linked/root-fixture/model.safetensors";
    let record = CheckpointCatalogRecordV1::from_plan(checkpoint_id, &plan).unwrap();
    let plans_dir = data_dir.join("checkpoints/plans");
    std::fs::create_dir_all(&plans_dir).unwrap();
    std::fs::write(
        plans_dir.join(format!("{}.json", plan.plan_id)),
        plan.canonical_json().unwrap(),
    )
    .unwrap();
    std::fs::write(
        data_dir.join("checkpoints/inventory.json"),
        CheckpointInventoryV1::new(vec![record.clone()])
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    let trusted = checkpoint_model_provenance(&plan, "my-import").unwrap();
    let entry = json!({"id":"my-import","family":"ltx2","baseModel":"declared-base","importPlan":{
        "checkpointId":checkpoint_id,"planId":plan.plan_id,"semanticDigest":record.plan.semantic_digest}});
    let job = state
        .jobs_store
        .create_job_with_model_provenance(
            crate::CreateJob {
                job_type: crate::JobType::ImageGenerate,
                project_id: Some(project.id.clone()),
                project_name: None,
                payload: json!({"model":"my-import","modelManifestEntry":entry})
                    .as_object()
                    .unwrap()
                    .clone(),
                requested_gpu: "auto".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            },
            Some(&trusted),
        )
        .unwrap();
    let modes = [
        ("image", "image/png", "png"),
        ("video", "video/mp4", "mp4"),
        ("audio", "audio/wav", "wav"),
        ("document", "text/plain", "txt"),
        ("vector", "image/svg+xml", "svg"),
        ("depth", "application/octet-stream", "npy"),
    ];
    let mut facts = Vec::new();
    for (kind, mime, extension) in modes {
        let media = format!("assets/{kind}/fixture.{extension}");
        let path = project_path.join(&media);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            if kind == "image" {
                PNG_32X32
            } else {
                b"synthetic CPU persistence fixture"
            },
        )
        .unwrap();
        facts.push(
            json!({"assetId":format!("asset_{kind}"),"mediaPath":media,"mimeType":mime,
            "type":kind,"model":"my-import","mode":"fixture","adapter":"cpu-boundary-fixture",
            "createdAt":"2026-10-23T00:00:00Z","displayName":format!("{kind} provenance fixture"),"width":32,"height":32,
            "modelProvenance":{"forged":true},"extra":{"modelProvenance":{"forged":true}}}),
        );
    }
    let mut result = json!({"generationSetId":"genset_provenance","assetWrites":facts})
        .as_object()
        .unwrap()
        .clone();
    let mut replay = result.clone();
    crate::jobs::persist_reported_assets(&state, &job.id, &mut result)
        .await
        .unwrap();
    assert_eq!(result["assets"].as_array().unwrap().len(), 6);
    for asset in result["assets"].as_array().unwrap() {
        assert_eq!(asset["modelProvenance"], trusted);
        assert_eq!(asset["recipe"]["modelProvenance"], trusted);
        assert!(asset["extra"].get("modelProvenance").is_none());
    }
    assert_eq!(
        state.jobs_store.model_provenance(&job.id).unwrap(),
        Some(trusted.clone())
    );
    // The second handoff is recovery/idempotency: the source catalog can be gone, but cannot
    // rewrite the already-frozen job identity. This path uses the durable commitment, not metadata.
    std::fs::remove_file(data_dir.join("checkpoints/inventory.json")).unwrap();
    let (app_after_restart, restarted) = create_app_with_state(settings).unwrap();
    crate::jobs::persist_reported_assets(&restarted, &job.id, &mut replay)
        .await
        .unwrap();
    assert_eq!(replay["assets"][0]["modelProvenance"], trusted);
    let (_, asset) = request(
        app_after_restart,
        "GET",
        &format!("/api/v1/projects/{}/assets/asset_image", project.id),
        Value::Null,
    )
    .await;
    assert_eq!(asset["modelProvenance"], trusted);
    assert_eq!(
        restarted.jobs_store.model_provenance(&job.id).unwrap(),
        Some(trusted.clone())
    );
    let edited = temp.path().join("edited.png");
    std::fs::write(&edited, PNG_32X32).unwrap();
    let edited = restarted
        .project_store
        .import_asset(
            &project.id,
            crate::UploadAsset {
                filename: "edited.png".into(),
                content_type: Some("image/png".into()),
                source_path: edited,
                source_asset_id: Some("asset_image".into()),
                provenance: Some(json!({"modelProvenance":{"forged":true},"editor":"crop"})),
            },
        )
        .unwrap();
    assert!(edited["extra"].get("modelProvenance").is_none());
    assert_eq!(edited["lineage"]["modelProvenance"], json!([trusted]));
    println!("SC26690_CPU_API_PROVENANCE_SMOKE: 6 asset modes persisted; worker claims replaced; DB/sidecar/index+HTTP read after restart preserved; edit ancestry retained");
}

fn authority_plan_fixture(settings: &Settings) -> (Value, Value) {
    let mut plan = ImportPlanV1::new(
        format!("checkpoint-plan-{}", "2".repeat(64)),
        "sdxl",
        vec![ImportLayerV1 {
            layer_id: "checkpoint".into(),
            role: "checkpoint".into(),
            target_path: "model.safetensors".into(),
            container: CheckpointContainerV1::Safetensors,
            source: SourceLocatorV1::linked("root-authority", "model.safetensors", "b".repeat(64))
                .unwrap(),
        }],
    )
    .unwrap();
    plan.context = import_declaration_context(
        json!({"family":"sdxl","baseModel":"realvisxl"})
            .as_object()
            .unwrap(),
    )
    .unwrap();
    let checkpoint_id = "linked/root-authority/model.safetensors";
    let record = CheckpointCatalogRecordV1::from_plan(checkpoint_id, &plan).unwrap();
    let plans = settings.data_dir.join("checkpoints/plans");
    std::fs::create_dir_all(&plans).unwrap();
    std::fs::write(
        plans.join(format!("{}.json", plan.plan_id)),
        plan.canonical_json().unwrap(),
    )
    .unwrap();
    std::fs::write(
        settings.data_dir.join("checkpoints/inventory.json"),
        CheckpointInventoryV1::new(vec![record.clone()])
            .unwrap()
            .canonical_json()
            .unwrap(),
    )
    .unwrap();
    let entry = json!({"id":"actual_import","name":"Actual imported checkpoint","family":"sdxl","type":"image","baseModel":"realvisxl","importSourceShape":"fused_checkpoint","downloads":[],
        "importPlan":{"checkpointId":checkpoint_id,"planId":plan.plan_id,"semanticDigest":record.plan.semantic_digest}});
    (
        entry,
        checkpoint_model_provenance(&plan, "actual_import").unwrap(),
    )
}

async fn persist_authority_output(
    state: &crate::AppState,
    project_id: &str,
    job_id: &str,
    asset_id: &str,
    kind: &str,
    legacy_claim: bool,
) -> Value {
    let project = state.project_store.get_project(project_id).unwrap();
    let (extension, mime) = match kind {
        "document" => ("txt", "text/plain"),
        "depth" => ("npy", "application/octet-stream"),
        _ => ("png", "image/png"),
    };
    let media = format!("assets/{kind}/{asset_id}.{extension}");
    let path = std::path::Path::new(&project.path).join(&media);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, PNG_32X32).unwrap();
    let mut fact = json!({"assetId":asset_id,"mediaPath":media,"mimeType":mime,"type":kind,
        "displayName":"Authority boundary fixture","createdAt":"2026-10-23T00:00:00Z","mode":"authority_fixture","model":"worker-model","adapter":"cpu-boundary",
        "modelProvenance":{"forged":true},"extra":{"modelProvenance":{"forged":true}}});
    if legacy_claim {
        fact["rawAdapterSettings"] =
            json!({"checkpointDerivative":{"cacheKey":"not-an-authenticated-key"}});
    }
    let mut result = json!({"generationSetId":"genset_authority","assetWrites":[fact]})
        .as_object()
        .unwrap()
        .clone();
    crate::jobs::persist_reported_assets(state, job_id, &mut result)
        .await
        .unwrap();
    result["assets"][0].clone()
}

#[tokio::test]
async fn raw_http_carriers_cannot_freeze_another_import_and_replays_keep_execution_canonical() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let (entry, trusted) = authority_plan_fixture(&settings);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::write(manifests.join("builtin.models.jsonc"), json!({"schemaVersion":1,"models":[
        {"id":"sensenova_u1_8b","name":"SenseNova","family":"sensenova-u1","type":"image","downloads":[]},
        {"id":"iris_3b_depth","name":"Iris depth","family":"iris","type":"utility","downloads":[]},
        {"id":"real_esrgan","name":"Real ESRGAN","family":"realesrgan","type":"utility","downloads":[]},entry]}).to_string()).unwrap();
    let (app, state) = create_app_with_state(settings).unwrap();
    let project = state.project_store.create_project("Raw authority").unwrap();
    let mut forged = entry.clone();
    forged["id"] = json!("forged-owner");
    let cases = [
        (
            "image_interleave",
            Some("sensenova_u1_8b"),
            "sensenova_u1_8b",
            "image",
        ),
        (
            "image_vqa",
            Some("sensenova_u1_8b"),
            "sensenova_u1_8b",
            "document",
        ),
        ("image_depth", None, "iris_3b_depth", "depth"),
        (
            "image_upscale",
            Some("RealESRGAN_x4plus"),
            "real_esrgan",
            "image",
        ),
        (
            "image_detail",
            Some("actual_import"),
            "actual_import",
            "image",
        ),
    ];
    for (index, (job_type, model, actual_model, output_type)) in cases.into_iter().enumerate() {
        let mut payload = json!({"projectId":project.id,"sourceAssetId":"source_fixture",
            "modelManifestEntry":forged,"modelManifestEntries":[forged],"baseModelManifestEntry":forged,"modelProvenance":{"forged":true}});
        if let Some(model) = model {
            payload["model"] = json!(model);
        }
        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({"type":job_type,"projectId":project.id,"requestedGpu":"auto","payload":payload}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{job_type}: {created}");
        let id = created["id"].as_str().unwrap();
        let queued = state.jobs_store.get_job(id).unwrap();
        assert_eq!(
            created["revision"], queued.extra["revision"],
            "admission and the durable queue must publish the same snapshot"
        );
        assert_eq!(
            queued.payload["modelManifestEntry"]["id"], actual_model,
            "the worker must receive the actual server selection"
        );
        assert!(queued.payload.get("baseModelManifestEntry").is_none());
        assert!(queued.payload.get("modelProvenance").is_none());
        if let Some(entries) = queued
            .payload
            .get("modelManifestEntries")
            .and_then(Value::as_array)
        {
            assert!(
                entries.iter().all(|entry| entry["id"] == actual_model),
                "plural execution carriers must also be regenerated, never retain the forgery"
            );
        }
        let asset = persist_authority_output(
            &state,
            &project.id,
            id,
            &format!("asset_raw_{index}"),
            output_type,
            false,
        )
        .await;
        if actual_model == "actual_import" {
            assert_eq!(
                state.jobs_store.model_provenance(id).unwrap(),
                Some(trusted.clone())
            );
            assert_eq!(asset["modelProvenance"], trusted);
        } else {
            assert_eq!(state.jobs_store.model_provenance(id).unwrap(), None);
            assert!(asset.get("modelProvenance").is_none());
            assert!(asset["recipe"].get("modelProvenance").is_none());
        }
        let (status, replay) = request(app.clone(), "POST", &format!("/api/v1/jobs/{id}/duplicate"), json!({"payloadChanges":{
            "modelManifestEntry":forged,"modelManifestEntries":[forged],"baseModelManifestEntry":forged,"modelProvenance":{"forged":true}}})).await;
        assert_eq!(status, StatusCode::CREATED, "{job_type}: {replay}");
        assert_eq!(replay["payload"]["modelManifestEntry"]["id"], actual_model);
        assert!(!replay["payload"]["baseModelManifestEntry"].is_object());
        if let Some(entries) = replay["payload"]["modelManifestEntries"].as_array() {
            assert!(
                entries.iter().all(|entry| entry["id"] == actual_model),
                "replay must reauthorize plural carriers as well as its primary"
            );
        }
        let replay_id = replay["id"].as_str().unwrap();
        assert_eq!(
            replay["revision"],
            state.jobs_store.get_job(replay_id).unwrap().extra["revision"]
        );
        let expected = (actual_model == "actual_import").then(|| trusted.clone());
        assert_eq!(
            state.jobs_store.model_provenance(replay_id).unwrap(),
            expected
        );
    }
    println!("SC26690_RAW_HTTP_AUTHORITY_PROOF: all five raw image routes and duplicate boundaries delivered server-selected execution carriers; only the actual imported model froze its own source");
}

#[tokio::test]
async fn unfrozen_legacy_carriers_remain_unknown_without_terminal_persistence_retries() {
    let temp = tempfile::tempdir().unwrap();
    let (_app, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let project = state
        .project_store
        .create_project("Unknown legacy authority")
        .unwrap();
    let job = state
        .jobs_store
        .create_job(crate::CreateJob {
            job_type: crate::JobType::ImageInterleave,
            project_id: Some(project.id.clone()),
            project_name: None,
            payload: json!({"model":"sensenova_u1_8b","modelManifestEntry":{
            "id":"forged-owner","importPlan":{"checkpointId":"missing-plan","planId":"missing"}}})
            .as_object()
            .unwrap()
            .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    for _ in 0..2 {
        let asset = persist_authority_output(
            &state,
            &project.id,
            &job.id,
            "asset_unknown_legacy",
            "image",
            true,
        )
        .await;
        assert!(asset.get("modelProvenance").is_none());
        assert!(asset["recipe"].get("modelProvenance").is_none());
        assert!(asset["recipe"]["rawAdapterSettings"]
            .get("checkpointDerivative")
            .is_none());
        assert_eq!(state.jobs_store.model_provenance(&job.id).unwrap(), None);
    }
}

#[tokio::test]
async fn native_worker_sidecars_retain_clip_provenance_in_frames_films_and_job_results() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let (_, trusted) = authority_plan_fixture(&settings);
    let (_app, state) = create_app_with_state(settings).unwrap();
    let project = state
        .project_store
        .create_project("Native ancestry")
        .unwrap();
    let project_path = std::path::Path::new(&project.path);
    let job = state
        .jobs_store
        .create_job_with_model_provenance(
            crate::CreateJob {
                job_type: crate::JobType::VideoGenerate,
                project_id: Some(project.id.clone()),
                project_name: None,
                payload: serde_json::Map::new(),
                requested_gpu: "auto".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            },
            Some(&trusted),
        )
        .unwrap();
    persist_authority_output(
        &state,
        &project.id,
        &job.id,
        "asset_source_clip",
        "video",
        false,
    )
    .await;
    let original = state
        .project_store
        .get_asset(&project.id, "asset_source_clip")
        .unwrap();
    let mut prior = trusted.clone();
    prior["modelId"] = json!("historical-import");
    for (id, kind, mode, parents) in [
        (
            "native_frame",
            "image",
            "frame_extract",
            json!(["asset_source_clip", "asset_source_clip"]),
        ),
        (
            "native_film",
            "video",
            "timeline_render",
            json!(["native_frame", "asset_source_clip"]),
        ),
    ] {
        let extension = if kind == "video" { "mp4" } else { "png" };
        let media = format!("assets/native/{id}.{extension}");
        let path = project_path.join(&media);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, PNG_32X32).unwrap();
        let sidecar = path.with_extension("sceneworks.json");
        let mut native = json!({"schemaVersion":1,"id":id,"projectId":project.id,"generationSetId":null,"type":kind,
            "displayName":"Native derived media","createdAt":"2026-10-23T00:00:00Z",
            "file":{"path":media,"mimeType":if kind=="video" {"video/mp4"} else {"image/png"},"width":32,"height":32,"duration":null,"fps":null},
            "status":{"favorite":false,"rating":0,"rejected":false,"trashed":false},
            "recipe":{"mode":mode,"model":"timeline-frame-extract","adapter":"ffmpeg","prompt":"Derived output","negativePrompt":"","seed":0,"loras":[],"stylePreset":"none","normalizedSettings":{},"rawAdapterSettings":{}},
            "lineage":{"parents":parents,"sourceAssetId":"asset_source_clip","sourceTimestamp":0.25,"jobId":"job_native_fixture"}});
        if kind == "image" {
            native["lineage"]["modelProvenance"] = json!([prior.clone()]);
        }
        let native = state
            .project_store
            .persist_native_asset_sidecar(&project.id, &sidecar, native)
            .unwrap();
        let inherited = native["lineage"]["modelProvenance"].as_array().unwrap();
        assert_eq!(
            inherited.iter().filter(|value| *value == &trusted).count(),
            1
        );
        assert!(
            inherited.contains(&prior),
            "existing trusted ancestry must not be replaced"
        );
        let repeated = state
            .project_store
            .persist_native_asset_sidecar(&project.id, &sidecar, native.clone())
            .unwrap();
        assert_eq!(
            repeated["lineage"], native["lineage"],
            "native registration is idempotent"
        );
        let reloaded = state.project_store.get_asset(&project.id, id).unwrap();
        let disk: Value = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        assert_eq!(reloaded["lineage"], native["lineage"]);
        assert_eq!(disk["lineage"], native["lineage"]);
        // Real native workers return assets, not assetWrites; the API does not subsequently repair
        // them. The returned envelope must therefore already match the sidecar/index ancestry.
        let mut native_result = json!({"assets":[native],"assetIds":[id]})
            .as_object()
            .unwrap()
            .clone();
        crate::jobs::persist_reported_assets(&state, "job_native_fixture", &mut native_result)
            .await
            .unwrap();
        assert_eq!(native_result["assets"][0]["lineage"], reloaded["lineage"]);
    }
    assert_eq!(
        state
            .project_store
            .get_asset(&project.id, "asset_source_clip")
            .unwrap(),
        original
    );
    println!("SC26690_NATIVE_ANCESTRY_PROOF: provenance-bound clip -> native frame -> native film; returned job assets, disk sidecars and index agree, ancestry deduplicated, parent unchanged");
}

#[tokio::test]
async fn typed_http_admission_freezes_catalog_authority_before_asset_persistence() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let (entry, trusted) = authority_plan_fixture(&settings);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        json!({"schemaVersion":1,"models":[entry]}).to_string(),
    )
    .unwrap();
    let (app, state) = create_app_with_state(settings).unwrap();
    let project = state
        .project_store
        .create_project("Typed authority")
        .unwrap();
    let (status, created) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({"projectId":project.id,
        "model":"actual_import","prompt":"A typed provenance fixture"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().unwrap();
    let queued = state.jobs_store.get_job(id).unwrap();
    assert_eq!(created["revision"], queued.extra["revision"]);
    assert_eq!(queued.payload["modelManifestEntry"]["id"], "actual_import");
    assert_eq!(
        state.jobs_store.model_provenance(id).unwrap(),
        Some(trusted.clone())
    );
    // Completion must retain admitted authority even if the source catalog changes/disappears.
    std::fs::remove_file(state.settings.data_dir.join("checkpoints/inventory.json")).unwrap();
    let asset = persist_authority_output(
        &state,
        &project.id,
        id,
        "asset_typed_authority",
        "image",
        false,
    )
    .await;
    assert_eq!(asset["modelProvenance"], trusted);
    assert_eq!(asset["recipe"]["modelProvenance"], trusted);
    assert_eq!(
        state
            .project_store
            .get_asset(&project.id, "asset_typed_authority")
            .unwrap()["modelProvenance"],
        trusted
    );
    println!("SC26690_TYPED_HTTP_AUTHORITY_PROOF: typed image route froze its own catalog checkpoint before dispatch; later asset/index/recipe retained that binding without the source inventory");
}

#[tokio::test]
async fn shipped_h3_rows_keep_their_stock_policy_but_cannot_hide_a_plan_binding() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    let rows: Vec<_> = ["minimax_h3", "minimax_h3_ref"]
        .into_iter()
        .map(|id| {
            Value::Object(
                sceneworks_core::builtin_manifests::builtin_model_entry(id)
                    .unwrap()
                    .clone(),
            )
        })
        .collect();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        json!({"schemaVersion":1,"models":rows}).to_string(),
    )
    .unwrap();
    let (app, state) = create_app_with_state(settings).unwrap();
    for id in ["minimax_h3", "minimax_h3_ref"] {
        let entry = sceneworks_core::builtin_manifests::builtin_model_entry(id).unwrap();
        crate::models::ensure_inherited_component_licenses_acknowledged(&state, entry, false)
            .await
            .unwrap();
        // Own stock download terms remain enforced independently of this import-only gate.
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/models/{id}/download"),
            json!({"requestedGpu":"auto"}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{error}");
        assert_eq!(error["code"], "license_acknowledgment_required");
        let mut draft = entry.clone();
        draft.insert("baseModel".into(), json!(id));
        draft.insert(
            "minimaxH3Partition".into(),
            json!(if id == "minimax_h3_ref" {
                "ref2va"
            } else {
                "fl2va"
            }),
        );
        draft.insert("requiresLicenseAcknowledgment".into(), json!(false));
        draft.insert("inheritedLicenseAcknowledgments".into(), json!([]));
        draft.insert("trustedStock".into(), json!(true));
        let error =
            crate::models::ensure_inherited_component_licenses_acknowledged(&state, &draft, false)
                .await
                .unwrap_err();
        assert_eq!(
            error.status,
            StatusCode::FORBIDDEN,
            "declared imports cannot claim a stock exemption"
        );
        draft.insert("id".into(), json!("community_h3_import"));
        assert_eq!(
            crate::models::ensure_inherited_component_licenses_acknowledged(&state, &draft, false)
                .await
                .unwrap_err()
                .status,
            StatusCode::FORBIDDEN
        );
        let mut bound = entry.clone();
        bound.insert(
            "importPlan".into(),
            json!({"checkpointId":"unregistered-stock-shadow"}),
        );
        assert!(
            crate::models::ensure_inherited_component_licenses_acknowledged(&state, &bound, false)
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn ordinary_custom_h3_catalogs_keep_own_terms_without_invented_import_lineage() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    let mut rows = Vec::new();
    for (id, source) in [
        (
            "custom_h3_hf",
            json!({"provider":"huggingface","repo":"community/complete-h3"}),
        ),
        (
            "custom_h3_local",
            json!({"provider":"local","path":"models/custom/full-h3"}),
        ),
    ] {
        let mut entry = sceneworks_core::builtin_manifests::builtin_model_entry("minimax_h3")
            .unwrap()
            .clone();
        entry.insert("id".into(), json!(id));
        entry.insert("source".into(), source);
        entry.insert("runtimeQuantTiers".into(), json!(["q4", "q8", "bf16"]));
        entry.insert("quantization".into(), json!({"bits":8,"group_size":64}));
        entry.insert(
            "paths".into(),
            json!({"model":temp.path().join("ordinary-catalog/full-h3")}),
        );
        entry.insert("inheritedLicenseAcknowledgments".into(), json!([]));
        rows.push(Value::Object(entry));
    }
    std::fs::write(
        manifests.join("user.models.jsonc"),
        json!({"schemaVersion":1,"models":rows}).to_string(),
    )
    .unwrap();
    let (app, state) = create_app_with_state(settings).unwrap();
    for row in rows {
        let entry = row.as_object().unwrap();
        crate::models::ensure_inherited_component_licenses_acknowledged(&state, entry, false)
            .await
            .unwrap();
        assert!(
            sceneworks_core::model_artifacts::provenance::resolved_model_provenance(
                &state.settings.data_dir,
                entry
            )
            .unwrap()
            .is_none()
        );
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/models/{}/download", entry["id"].as_str().unwrap()),
            json!({"requestedGpu":"auto"}),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{error}");
        assert_eq!(error["code"], "license_acknowledgment_required");
        for (key, value) in [
            ("importPlan", json!({"checkpointId":"missing-custom-plan"})),
            ("baseModel", json!("minimax_h3")),
            ("minimaxH3Partition", json!("fl2va")),
            ("importSourceShape", json!("transformer_directory")),
            (
                "inheritedLicenseAcknowledgments",
                json!([{"id":"minimax_h3"}]),
            ),
            (
                "source",
                json!({"provider":"local","path":"models/imports/custom-h3"}),
            ),
        ] {
            let mut claimed_import = entry.clone();
            claimed_import.insert(key.into(), value);
            claimed_import.insert("trustedStock".into(), json!(true));
            claimed_import.insert("requiresLicenseAcknowledgment".into(), json!(false));
            assert!(
                crate::models::ensure_inherited_component_licenses_acknowledged(
                    &state,
                    &claimed_import,
                    false
                )
                .await
                .is_err(),
                "positive {key} evidence cannot be erased by false license/stock flags"
            );
        }
    }
}
