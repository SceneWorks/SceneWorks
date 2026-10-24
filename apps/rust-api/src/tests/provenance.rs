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
        .create_job(crate::CreateJob {
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
        })
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
