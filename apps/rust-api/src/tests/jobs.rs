//! rust-api jobs tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

#[test]
fn terminal_model_lifecycle_jobs_invalidate_catalog_snapshots() {
    for job_type in [
        crate::JobType::ModelDownload,
        crate::JobType::ModelImport,
        crate::JobType::ModelConvert,
    ] {
        for status in [
            crate::JobStatus::Completed,
            crate::JobStatus::Failed,
            crate::JobStatus::Canceled,
            crate::JobStatus::Interrupted,
        ] {
            assert!(
                crate::jobs::terminal_model_job_changes_catalog(&job_type, &status),
                "{job_type:?} {status:?} may change install state"
            );
        }
        assert!(!crate::jobs::terminal_model_job_changes_catalog(
            &job_type,
            &crate::JobStatus::Running
        ));
    }
    assert!(!crate::jobs::terminal_model_job_changes_catalog(
        &crate::JobType::ImageGenerate,
        &crate::JobStatus::Completed
    ));
}

#[tokio::test]
async fn worker_termination_refreshes_warm_model_catalog_after_artifact_mutation() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    single_model_manifest(&config_dir, "crashed_model", "crashed/model");
    let (app, state) =
        create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
    *state.model_size_estimate_disabled_override.lock() = Some(true);
    crate::test_reset_catalog_build_counters();

    let (status, warm) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(warm[0]["installState"], "missing");
    assert_eq!(crate::test_model_catalog_builds(), 1);

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "download-worker",
            "gpuId": "cpu",
            "gpuName": null,
            "capabilities": ["model_download"],
            "loadedModels": []
        }),
    )
    .await;
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "model_download",
            "payload": { "modelId": "crashed_model" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = created["id"].as_str().expect("job id is string");
    let (status, claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "download-worker" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claim["job"]["id"], job_id);

    let marker_dir = temp_dir.path().join("data/models/crashed__model");
    std::fs::create_dir_all(&marker_dir).expect("model marker dir creates");
    std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("model marker writes");

    let (status, failed) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/download-worker/terminated",
        json!({ "signal": 9, "exitCode": null }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(failed["status"], "failed");

    let (status, refreshed) = request(app, "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(refreshed[0]["installState"], "installed");
    assert_eq!(
        crate::test_model_catalog_builds(),
        2,
        "worker termination must invalidate a warm install-state snapshot"
    );
}

#[tokio::test]
async fn stale_worker_sweep_refreshes_warm_model_catalog_after_artifact_mutation() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    single_model_manifest(&config_dir, "stale_model", "stale/model");
    let mut settings = test_settings(&temp_dir);
    settings.worker_timeout_seconds = 1;
    let jobs_db_path = settings.jobs_db_path.clone();
    let (app, state) = create_app_with_state(settings).expect("app and state create");
    *state.model_size_estimate_disabled_override.lock() = Some(true);
    crate::test_reset_catalog_build_counters();

    let (status, warm) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(warm[0]["installState"], "missing");
    assert_eq!(crate::test_model_catalog_builds(), 1);

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "import-worker",
            "gpuId": "cpu",
            "gpuName": null,
            "capabilities": ["model_import"],
            "loadedModels": []
        }),
    )
    .await;
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "model_import",
            "payload": { "modelId": "stale_model" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = created["id"].as_str().expect("job id is string");
    let (status, claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "import-worker" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claim["job"]["id"], job_id);

    let marker_dir = temp_dir.path().join("data/models/stale__model");
    std::fs::create_dir_all(&marker_dir).expect("model marker dir creates");
    std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
        .expect("model marker writes");

    let connection = rusqlite::Connection::open(jobs_db_path).expect("jobs db opens");
    let updated = connection
        .execute(
            "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = ?1",
            rusqlite::params!["import-worker"],
        )
        .expect("worker timestamp ages");
    assert_eq!(updated, 1);
    drop(connection);

    let (status, _) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    let (status, interrupted) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(interrupted["status"], "interrupted");

    let (status, refreshed) = request(app, "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(refreshed[0]["installState"], "installed");
    assert_eq!(
        crate::test_model_catalog_builds(),
        2,
        "stale-worker interruption must invalidate a warm install-state snapshot"
    );
}

/// F-003 / sc-11159: a path-traversal `model` id is rejected at the POST boundary for BOTH
/// the image and video enqueue lanes (before any job is created), closing the remote
/// arbitrary-write primitive the worker filename builders would otherwise expose.
#[tokio::test]
async fn image_and_video_jobs_reject_path_unsafe_model_id() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (endpoint, mode) in [
        ("/api/v1/image/jobs", "text_to_image"),
        ("/api/v1/video/jobs", "text_to_video"),
    ] {
        for evil in ["../../../../etc/passwd", "..\\..\\evil", "/abs/pwn", "a/b"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                endpoint,
                json!({
                    "projectId": "project-1",
                    "mode": mode,
                    "prompt": "a fox",
                    "model": evil,
                }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{endpoint} {evil:?}");
            assert!(
                body["detail"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("model must be a plain model id"),
                "{endpoint} {evil:?}: unexpected error {body}"
            );
        }
    }
}

/// sc-12305: the generic `POST /api/v1/jobs` enqueues `type` + payload verbatim — no
/// manifest resolution — so a generation job through that door carries no
/// `modelManifestEntry` and silently renders off-bucket (see the
/// `mochi_without_manifest_entry_*` test in `video_request.rs` for the exact geometry).
/// Every job type whose typed route injects an entry must be rejected here, pointed at
/// that route. Covers image as well as video: `image_request.rs` reads the entry the same way.
#[tokio::test]
async fn generic_jobs_route_rejects_generation_types_with_their_typed_route() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (job_type, typed_route) in [
        ("image_generate", "/api/v1/image/jobs"),
        ("image_edit", "/api/v1/image/jobs"),
        ("video_generate", "/api/v1/video/jobs"),
        ("video_extend", "/api/v1/video/jobs"),
        ("video_bridge", "/api/v1/video/jobs"),
        ("person_replace", "/api/v1/video/jobs"),
        // Vector Studio is a typed `vector_generate` capability. `image_to_svg` is a payload
        // mode owned by its dedicated fixture route, never a generic job type.
        ("vector_generate", "/api/v1/image/vectorize/jobs"),
        // Audio Studio (sc-13404): the audio route injects the model's manifest entry too, so an
        // `audio_generate` job enqueued raw through the generic route must be rejected the same way.
        ("audio_generate", "/api/v1/audio/jobs"),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "projectId": "project-1",
                "requestedGpu": "auto",
                "payload": { "model": "mochi_1", "prompt": "a fox", "width": 848, "height": 480 },
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{job_type} must be rejected"
        );
        let detail = body["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains(typed_route),
            "{job_type}: error must name {typed_route}, got {body}"
        );
    }
}

fn write_vector_test_manifest(config_dir: &std::path::Path, capabilities: &[&str]) {
    write_vector_test_manifest_with_provider_state(config_dir, capabilities, true);
}

fn write_vector_test_manifest_with_provider_state(
    config_dir: &std::path::Path,
    capabilities: &[&str],
    provider_available: bool,
) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    let provider = |id| {
        if provider_available {
            json!({ "id": id, "available": true })
        } else {
            json!({ "id": id, "available": false, "reason": "pending_terminal_inference_pin" })
        }
    };
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "models": [{
                "id": "starvector_test",
                "name": "StarVector test",
                "type": "vector",
                "family": "starvector",
                "adapter": "starvector",
                "capabilities": capabilities,
                "vector": { "providers": {
                    "mlx": provider("mlx-starvector-1b"),
                    "candle": provider("candle-starvector-1b")
                } },
                "downloads": [{
                    "provider": "huggingface",
                    "repo": "SceneWorks/starvector-test",
                    "revision": "2222222222222222222222222222222222222222",
                    "files": ["config.json", "model.safetensors"]
                }],
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("builtin models write");
    write_empty_sibling_manifests(config_dir);
    let installed = config_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("test root")
        .join("data/models/SceneWorks__starvector-test");
    std::fs::create_dir_all(&installed).expect("test vector install dir creates");
    std::fs::write(installed.join(".sceneworks-download-complete.json"), b"{}")
        .expect("test vector receipt writes");
    std::fs::write(installed.join("config.json"), b"{}").expect("test config writes");
    std::fs::write(installed.join("model.safetensors"), b"weights").expect("test weights write");
}

fn write_vector_workflow_test_manifest(
    config_dir: &std::path::Path,
    raster_revision: &str,
    vector_revision: &str,
) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        serde_json::to_vec_pretty(&json!({
            "schemaVersion": 1,
            "models": [
                {
                    "id": "flux_schnell",
                    "name": "Raster test",
                    "type": "image",
                    "family": "flux",
                    "adapter": "flux_diffusers",
                    "capabilities": ["text_to_image"],
                    "defaults": { "count": 1, "resolution": { "width": 512, "height": 512 } },
                    "downloads": [{
                        "provider": "huggingface",
                        "repo": "SceneWorks/raster-workflow-test",
                        "revision": raster_revision,
                        "files": ["config.json", "model.safetensors"]
                    }]
                },
                {
                    "id": "starvector_test",
                    "name": "Vector test",
                    "type": "vector",
                    "family": "starvector",
                    "adapter": "starvector",
                    "capabilities": ["image_to_svg"],
                    "vector": {
                        "acceptsTextGuidance": false,
                        "providers": {
                            "mlx": { "id": "mlx-starvector-test", "available": true },
                            "candle": { "id": "candle-starvector-test", "available": true }
                        }
                    },
                    "downloads": [{
                        "provider": "huggingface",
                        "repo": "SceneWorks/vector-workflow-test",
                        "revision": vector_revision,
                        "files": ["config.json", "model.safetensors"]
                    }]
                }
            ]
        }))
        .expect("manifest serializes"),
    )
    .expect("builtin models write");
    write_empty_sibling_manifests(config_dir);
    let root = config_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("test root");
    for repo in [
        "SceneWorks__raster-workflow-test",
        "SceneWorks__vector-workflow-test",
    ] {
        let installed = root.join("data/models").join(repo);
        std::fs::create_dir_all(&installed).expect("workflow install dir creates");
        std::fs::write(installed.join(".sceneworks-download-complete.json"), b"{}")
            .expect("workflow receipt writes");
        std::fs::write(installed.join("config.json"), b"{}").expect("config writes");
        std::fs::write(installed.join("model.safetensors"), b"weights").expect("weights write");
    }
}

#[test]
fn prompt_vector_revision_identity_rejects_mutable_missing_and_conflicting_primaries() {
    let revision = "1111111111111111111111111111111111111111";
    assert_eq!(
        crate::generation::authoritative_workflow_revision(&json!({
            "downloads": [{ "revision": revision }]
        }))
        .expect("one immutable revision"),
        revision
    );
    for downloads in [
        json!([]),
        json!([{ "revision": "main" }]),
        json!([{ "revision": revision }, { "revision": "main" }]),
        json!([
            { "revision": revision },
            { "revision": "2222222222222222222222222222222222222222" }
        ]),
    ] {
        let error = crate::generation::authoritative_workflow_revision(&json!({
            "downloads": downloads
        }))
        .expect_err("ambiguous identity refuses");
        assert_eq!(error.code, Some("vector_workflow_artifact_ambiguous"));
    }
}

#[test]
fn prompt_vector_intermediate_ownership_is_server_authored_and_worker_facts_are_replaced() {
    let public_request: crate::dto::ImageJobRequest = serde_json::from_value(json!({
        "projectId": "project-1",
        "prompt": "ordinary image",
        "workflowParentId": "job_forged",
        "workflowId": "vwf_forged"
    }))
    .expect("image request parses");
    assert!(public_request.workflow_parent_id.is_none());
    assert!(public_request.workflow_id.is_none());

    let payload = json!({
        "workflowParentId": "job_parent1",
        "workflowId": "vwf_workflow1"
    })
    .as_object()
    .expect("payload object")
    .clone();
    let mut writes = vec![json!({
        "assetId": "asset-1",
        "vectorWorkflowOwnership": {
            "workflowId": "vwf_worker_forgery",
            "parentJobId": "job_worker_forgery"
        }
    })];
    crate::jobs::stamp_vector_workflow_asset_writes(
        &crate::JobType::ImageGenerate,
        &payload,
        "job_child1",
        &mut writes,
    );
    assert_eq!(
        writes[0]["vectorWorkflowOwnership"],
        json!({
            "role": "retained_intermediate",
            "publication": "unpublished",
            "workflowId": "vwf_workflow1",
            "parentJobId": "job_parent1",
            "childJobId": "job_child1",
            "hidden": true,
        })
    );

    crate::jobs::stamp_vector_workflow_asset_writes(
        &crate::JobType::VideoGenerate,
        &payload,
        "job_child1",
        &mut writes,
    );
    assert!(writes[0].get("vectorWorkflowOwnership").is_none());
}

#[tokio::test]
async fn prompt_vector_workflow_persists_a_nonclaimable_parent_and_cancel_cascades() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_vector_workflow_test_manifest(
        &temp_dir.path().join("config/manifests"),
        "1111111111111111111111111111111111111111",
        "2222222222222222222222222222222222222222",
    );
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Prompt vectors" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, parent) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/prompt/jobs",
        json!({
            "projectId": project_id,
            "prompt": "a geometric fox mark",
            "negativePrompt": "photographic texture",
            "rasterModel": "flux_schnell",
            "vectorModel": "starvector_test",
            "seed": 17,
            "detailBudget": {
                "maxNewTokens": 2048,
                "maxSvgBytes": 131072,
                "maxWallTimeMs": 60000
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{parent}");
    assert_eq!(parent["type"], "vector_generate");
    assert_eq!(parent["status"], "pending_workflow");
    assert_eq!(parent["stage"], "pending_workflow");
    assert!(parent["payload"]["sourceAssetId"].is_null());
    let workflow = &parent["payload"]["workflow"];
    assert_eq!(workflow["kind"], "create_from_prompt");
    assert_eq!(workflow["disclosure"], "raster_to_vector");
    assert_eq!(
        workflow["intermediateVisibility"],
        "hidden_retained_on_success"
    );
    assert_eq!(
        workflow["rasterStage"]["revision"],
        "1111111111111111111111111111111111111111"
    );
    assert_eq!(
        workflow["vectorStage"]["revision"],
        "2222222222222222222222222222222222222222"
    );
    assert_eq!(workflow["vectorStage"]["mode"], "image_to_svg");
    let parent_id = parent["id"].as_str().expect("parent id");
    let child_id = workflow["childJobId"].as_str().expect("child id");

    let (_, child) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{child_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(child["type"], "image_generate");
    assert_eq!(child["status"], "queued");
    assert_eq!(child["payload"]["workflowParentId"], parent_id);
    assert_eq!(child["payload"]["workflowId"], workflow["id"]);

    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{parent_id}/cancel"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");
    let (_, child) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{child_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(child["status"], "canceled");
    let (_, assets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    assert!(assets.as_array().expect("assets").is_empty());

    let (_, bulk_parent) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/prompt/jobs",
        json!({
            "projectId": project_id,
            "prompt": "a monoline heron",
            "rasterModel": "flux_schnell",
            "vectorModel": "starvector_test"
        }),
    )
    .await;
    let bulk_parent_id = bulk_parent["id"].as_str().expect("bulk parent id");
    let bulk_child_id = bulk_parent["payload"]["workflow"]["childJobId"]
        .as_str()
        .expect("bulk child id");
    let (status, bulk) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/cancel-pending",
        json!({ "projectId": project_id }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{bulk}");
    assert_eq!(bulk["canceled"], 2);
    for id in [bulk_parent_id, bulk_child_id] {
        let (_, canceled) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{id}"),
            Value::Null,
        )
        .await;
        assert_eq!(canceled["status"], "canceled");
    }
}

#[tokio::test]
async fn prompt_vector_replay_creates_both_stages_anew_and_revision_drift_is_typed() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_vector_workflow_test_manifest(
        &temp_dir.path().join("config/manifests"),
        "1111111111111111111111111111111111111111",
        "2222222222222222222222222222222222222222",
    );
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Prompt replay" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let body = json!({
        "projectId": project_id,
        "prompt": "a single-line owl",
        "rasterModel": "flux_schnell",
        "vectorModel": "starvector_test",
        "seed": 23
    });
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/prompt/jobs",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let original_id = original["id"].as_str().expect("original id");
    let original_child = original["payload"]["workflow"]["childJobId"]
        .as_str()
        .expect("original child");

    let (status, replay) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{original_id}/retry"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replay}");
    assert_eq!(replay["sourceJobId"], original_id);
    assert_eq!(replay["attempts"], 2);
    assert_ne!(replay["id"], original["id"]);
    assert_ne!(
        replay["payload"]["workflow"]["id"],
        original["payload"]["workflow"]["id"]
    );
    assert_ne!(replay["payload"]["workflow"]["childJobId"], original_child);
    assert_eq!(
        replay["payload"]["workflow"]["rasterStage"]["revision"],
        original["payload"]["workflow"]["rasterStage"]["revision"]
    );
    assert_eq!(
        replay["payload"]["workflow"]["vectorStage"]["revision"],
        original["payload"]["workflow"]["vectorStage"]["revision"]
    );

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(before.as_array().expect("jobs").len(), 4);
    let mut drifted = body;
    drifted["expectedRasterRevision"] = json!("3333333333333333333333333333333333333333");
    drifted["expectedVectorRevision"] = json!("2222222222222222222222222222222222222222");
    let (status, drift) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/prompt/jobs",
        drifted,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{drift}");
    assert_eq!(drift["code"], "vector_workflow_revision_drift");
    assert_eq!(drift["context"]["stage"], "raster");
    let (_, after) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(after.as_array().expect("jobs").len(), 4);
}

#[tokio::test]
async fn vector_route_reports_typed_unavailable_backend_before_enqueue() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_vector_test_manifest_with_provider_state(
        &temp_dir.path().join("config/manifests"),
        &["image_to_svg"],
        false,
    );
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Vector" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (_, source) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "source.png",
        "image/png",
        b"png-bytes",
    )
    .await;
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/jobs",
        json!({
            "projectId": project_id,
            "mode": "image_to_svg",
            "model": "starvector_test",
            "sourceAssetId": source["id"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["code"], "vector_backend_unavailable");
    assert_eq!(body["context"]["reason"], "pending_terminal_inference_pin");
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().expect("jobs").is_empty());
}

#[test]
fn builtin_starvector_manifests_are_exact_native_image_to_svg_closures() {
    let cases = [
        (
            "starvector_1b",
            "1b",
            "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
            5_147_481_592u64,
            5_142_705_320u64,
            true,
            None,
            json!([
                "README.md",
                "added_tokens.json",
                "config.json",
                "merges.txt",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json"
            ]),
        ),
        (
            "starvector_8b",
            "8b",
            "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
            15_015_835_105u64,
            15_014_294_040u64,
            true,
            None,
            json!([
                "README.md",
                "added_tokens.json",
                "config.json",
                "merges.txt",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "processor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json"
            ]),
        ),
    ];

    for (
        id,
        tier,
        revision,
        closure_bytes,
        static_floor_bytes,
        provider_available,
        provider_reason,
        files,
    ) in cases
    {
        let model = crate::models::embedded_builtin_catalog_entry(|entry| {
            entry.get("id").and_then(Value::as_str) == Some(id)
        })
        .expect("embedded manifest parses")
        .unwrap_or_else(|| panic!("{id} entry exists"));
        assert_eq!(model["type"], "vector");
        assert_eq!(model["capabilities"], json!(["image_to_svg"]));
        assert_eq!(model["adapter"], "starvector");
        assert_eq!(model["vector"]["acceptsTextGuidance"], false);
        assert!(model["capabilities"]
            .as_array()
            .expect("capabilities")
            .iter()
            .all(|capability| capability != "text_to_svg"));
        for backend in ["mlx", "candle"] {
            assert_eq!(
                model["vector"]["providers"][backend]["id"],
                format!("{backend}-starvector-{tier}")
            );
            assert_eq!(
                model["vector"]["providers"][backend]["available"],
                provider_available
            );
            assert_eq!(
                model["vector"]["providers"][backend]
                    .get("reason")
                    .and_then(Value::as_str),
                provider_reason
            );
        }
        assert_eq!(model["vector"]["deviceAdmission"]["schemaVersion"], 1);
        assert_eq!(
            model["vector"]["deviceAdmission"]["basis"],
            "exact_safetensors_bytes"
        );
        assert_eq!(model["vector"]["deviceAdmission"]["measured"], false);
        assert_eq!(
            model["vector"]["deviceAdmission"]["staticWeightFloorBytes"],
            static_floor_bytes
        );
        if id == "starvector_8b" {
            let candidate = &model["vector"]["deviceAdmission"]["terminalCandidate"];
            assert_eq!(
                candidate["inferenceRevision"],
                "53a0ef89525e1d1f7202d4932e9cccc4388e9229"
            );
            assert_eq!(
                candidate["corpusSha256"],
                "757370c4eed38a52a29ac80c258fdedd7e437ab891637bcb1c916aa608bf32b5"
            );
            assert_eq!(
                candidate["productionClosure"]["sha256"],
                "ae95f72bb5265aceccc5d69c8e379e6e267d7215bc5015e65a29f5f5e3d8e64e"
            );
            assert_eq!(
                candidate["productionClosure"]["entries"]
                    .as_array()
                    .expect("production closure entries")
                    .len(),
                28
            );
            assert_eq!(
                candidate["supportedDevices"]["mlx"],
                json!([{
                    "deviceClass": "apple_unified_memory",
                    "totalBytes": 137_438_953_472u64
                }])
            );
            assert_eq!(
                candidate["supportedDevices"]["candle"],
                json!([{
                    "deviceClass": "nvidia_dedicated_vram",
                    "deviceName": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                    "totalBytes": 102_641_958_912u64
                }])
            );
        }

        let download = &model["downloads"][0];
        assert_eq!(
            download["repo"],
            format!("starvector/starvector-{tier}-im2svg")
        );
        assert_eq!(download["revision"], revision);
        assert_eq!(download["estimatedSizeBytes"], closure_bytes);
        assert_eq!(download["footprint"]["diskSizeBytes"], closure_bytes);
        assert_eq!(download["files"], files);
        assert!(download["files"]
            .as_array()
            .expect("files")
            .iter()
            .all(|file| !file.as_str().unwrap_or_default().ends_with(".py")));
        assert_eq!(
            model["licenseUrl"],
            format!("https://huggingface.co/starvector/starvector-{tier}-im2svg/tree/{revision}")
        );
        assert!(download["files"]
            .as_array()
            .expect("files")
            .contains(&json!("README.md")));
        assert_eq!(
            model["ui"]["promptGuide"]["path"],
            format!("/prompt-guides/starvector-{tier}.md")
        );
    }
}

#[test]
fn builtin_starvector_license_provenance_covers_both_immutable_model_cards() {
    let licenses: Value =
        serde_json::from_str(include_str!("../../../desktop/licenses/manifest.json"))
            .expect("desktop license manifest parses");
    let component = licenses["components"]
        .as_array()
        .expect("license components")
        .iter()
        .find(|component| component["id"] == "starvector-1b")
        .expect("StarVector license component");
    assert_eq!(
        component["models"],
        json!(["starvector_1b", "starvector_8b"])
    );
    assert_eq!(component["license"], "Apache-2.0");
    let usage = component["usage"].as_str().expect("usage");
    for revision in [
        "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
        "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
    ] {
        assert!(usage.contains(revision));
    }
    assert!(usage.contains("model card"));
    assert!(usage.contains("never executes"));
    assert!(usage.contains("no separate NOTICE"));

    let provenance = include_str!("../../../desktop/licenses/starvector-1b/README.md");
    assert!(provenance.contains("starvector/starvector-1b-im2svg@380ab95"));
    assert!(provenance.contains("starvector/starvector-8b-im2svg@518beea"));
    assert!(provenance.contains("model card"));
    assert!(provenance.contains("excludes both repositories' Python modules"));
}

#[tokio::test]
async fn vector_route_reports_typed_missing_and_incomplete_model_manager_recovery() {
    let _env = isolate_hf_cache();
    for (cache_state, expected_reason) in [
        ("missing", "model_missing"),
        ("incomplete", "model_incomplete"),
    ] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        write_vector_test_manifest(&config_dir, &["image_to_svg"]);
        let installed = temp_dir
            .path()
            .join("data/models/SceneWorks__starvector-test");
        std::fs::remove_file(installed.join(".sceneworks-download-complete.json"))
            .expect("remove complete marker");
        if cache_state == "incomplete" {
            std::fs::remove_dir_all(&installed).expect("remove managed install");
            let snapshot = temp_dir.path().join(
                "data/cache/huggingface/hub/models--SceneWorks--starvector-test/snapshots/abc123",
            );
            std::fs::create_dir_all(&snapshot).expect("partial HF snapshot creates");
            std::fs::write(snapshot.join("config.json"), "{}").expect("partial file writes");
        } else {
            std::fs::remove_dir_all(&installed).expect("missing install removes directory");
        }
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Vector" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id");
        let (_, source) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "source.png",
            "image/png",
            b"png-bytes",
        )
        .await;
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/image/vectorize/jobs",
            json!({
                "projectId": project_id,
                "mode": "image_to_svg",
                "model": "starvector_test",
                "sourceAssetId": source["id"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{cache_state}: {body}");
        assert_eq!(body["code"], "vector_model_unavailable");
        assert_eq!(body["context"]["reason"], expected_reason);
        assert_eq!(body["context"]["downloadable"], true);
        assert!(body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("Model Manager"));
        let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
        assert!(jobs.as_array().expect("jobs").is_empty());
    }
}

#[tokio::test]
async fn vector_route_validates_and_stamps_typed_image_to_svg_request() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_vector_test_manifest(
        &temp_dir.path().join("config/manifests"),
        &["image_to_svg", "text_to_svg"],
    );
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Vector Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let (status, source) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "source.png",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let source_asset_id = source["id"].as_str().expect("source id");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/jobs",
        json!({
            "projectId": project_id,
            "projectName": "Vector Project",
            "mode": "image_to_svg",
            "model": "starvector_test",
            "sourceAssetId": source_asset_id,
            "prompt": "keep the silhouette",
            "sampling": { "temperature": 0.1, "topP": 0.95, "seed": 42 },
            "detailBudget": { "maxNewTokens": 2048, "maxSvgBytes": 131072, "maxWallTimeMs": 90000 }
        }),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["type"], "vector_generate");
    assert_eq!(created["requestedGpu"], "auto");
    assert_eq!(created["payload"]["mode"], "image_to_svg");
    assert_eq!(created["payload"]["model"], "starvector_test");
    assert_eq!(created["payload"]["sourceAssetId"], source_asset_id);
    assert_eq!(created["payload"]["sampling"]["seed"], 42);
    assert_eq!(created["payload"]["detailBudget"]["maxSvgBytes"], 131072);
    assert_eq!(
        created["payload"]["modelManifestEntry"]["adapter"],
        "starvector"
    );
    assert_eq!(
        created["payload"]["modelManifestEntry"]["capabilities"],
        json!(["image_to_svg", "text_to_svg"])
    );
    assert!(created["payload"].get("fixtureSvg").is_none());

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "text-only-vector-worker",
            "gpuId": "test-gpu-1",
            "gpuName": "Test GPU",
            "capabilities": ["gpu", "vector_text_to_svg"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, wrong_claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "text-only-vector-worker" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(wrong_claim["job"].is_null());
    claim_job_as_worker(
        &app,
        created["id"].as_str().expect("job id"),
        "image-vector-worker",
        &["gpu", "vector_image_to_svg"],
    )
    .await;

    let (status, text_created) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vectorize/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_svg",
            "model": "starvector_test",
            "prompt": "a minimal geometric fox"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(text_created["payload"]["mode"], "text_to_svg");
    assert!(text_created["payload"]["sourceAssetId"].is_null());
    let (status, text_claim) = request(
        app,
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "text-only-vector-worker" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(text_claim["job"]["id"], text_created["id"]);
}

#[tokio::test]
async fn vector_route_rejects_bad_source_ownership_media_and_model_capability_before_enqueue() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_vector_test_manifest(&temp_dir.path().join("config/manifests"), &["image_to_svg"]);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project_a) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Project A" }),
    )
    .await;
    let (_, project_b) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Project B" }),
    )
    .await;
    let project_a_id = project_a["id"].as_str().expect("project A id");
    let project_b_id = project_b["id"].as_str().expect("project B id");
    let (_, raster) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_a_id}/assets"),
        "source.png",
        "image/png",
        b"png-bytes",
    )
    .await;
    let raster_id = raster["id"].as_str().expect("raster id");
    let (status, video) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_a_id}/assets"),
        "source.mp4",
        "video/mp4",
        b"mp4-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let video_id = video["id"].as_str().expect("video id");

    for (body, expected_status) in [
        (
            json!({
                "projectId": project_a_id,
                "mode": "image_to_svg",
                "model": "starvector_test"
            }),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({
                "projectId": project_a_id,
                "mode": "image_to_svg",
                "model": "starvector_test",
                "sourceAssetId": video_id
            }),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({
                "projectId": project_b_id,
                "mode": "image_to_svg",
                "model": "starvector_test",
                "sourceAssetId": raster_id
            }),
            StatusCode::NOT_FOUND,
        ),
        (
            json!({
                "projectId": project_a_id,
                "mode": "text_to_svg",
                "model": "starvector_test",
                "prompt": "a mark"
            }),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({
                "projectId": project_a_id,
                "mode": "image_to_svg",
                "model": "starvector_test",
                "sourceAssetId": raster_id,
                "fixtureSvg": "<svg/>"
            }),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let (status, _) = request(app.clone(), "POST", "/api/v1/image/vectorize/jobs", body).await;
        assert_eq!(status, expected_status);
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().expect("jobs array").len(), 0);
}

/// The other half of the guard: the job types the generic route legitimately serves keep
/// working. `image_upscale` / `image_detail` are the real web callers (batch ops), and
/// neither has a typed door — so the sc-12305 rejection must not touch them.
///
/// sc-19708: these payloads are no longer forwarded byte-identical — the model-source seam
/// attaches the fixed utility/default model's catalog entry (the sc-18480 enrichment pattern),
/// because the worker's pre-loader guard fails closed on a model-backed job with no carriers.
/// The client's own fields must still pass through untouched.
#[tokio::test]
async fn generic_jobs_route_still_serves_non_generation_types() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (job_type, expected_model) in [
        ("image_upscale", "real_esrgan"),
        ("image_detail", "realvisxl"),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "requestedGpu": "auto",
                "payload": { "sourceAssetId": "asset-1" },
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{job_type} must still enqueue: {body}"
        );
        assert_eq!(
            body["payload"]["sourceAssetId"],
            json!("asset-1"),
            "the client's own payload fields must pass through unchanged"
        );
        let stamped = body["payload"]["modelManifestEntries"]
            .as_array()
            .unwrap_or_else(|| panic!("{job_type} must carry its fixed model entry, got {body}"))
            .iter()
            .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            stamped,
            vec![expected_model],
            "{job_type} must carry exactly its fixed utility/default model"
        );
    }
}

/// sc-18480: Batch Detail keeps its established raw `/api/v1/jobs` contract, but the Candle SDXL
/// provider needs the selected model's three descriptor-owned co-requisites. Start with the exact
/// client shape (no `modelManifestEntry`) and prove the API enriches the persisted worker payload
/// from the shipped catalog. `model_jobs::sdxl_co_requisites_resolve_all_three_from_every_live_*`
/// proves this same three-id entry resolves to installed paths at the worker seam.
#[tokio::test]
async fn raw_batch_detail_injects_authoritative_sdxl_components_for_the_worker() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .expect("shipped model manifest writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Batch Detail",
            "requestedGpu": "auto",
            "payload": {
                "projectId": "project-1",
                "sourceAssetId": "asset-1",
                "model": "realvisxl",
                "displayName": "portrait.png",
                "advanced": { "strength": 0.55, "cnScale": 0.7 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "detail job enqueues: {job}");

    let entry = &job["payload"]["modelManifestEntry"];
    assert_eq!(entry["id"], "realvisxl");
    assert_eq!(entry["family"], "sdxl");
    assert_eq!(entry["type"], "image");
    #[cfg(not(target_os = "macos"))]
    assert_eq!(
        job["payload"]["advanced"]["mlxQuantize"],
        json!(0),
        "the Candle route must persist its supported dense-bf16 tier instead of inheriting q4"
    );
    let component_ids: std::collections::BTreeSet<&str> = entry["downloads"]
        .as_array()
        .expect("authoritative downloads array")
        .iter()
        .filter(|download| download["coRequisite"] == json!(true))
        .filter_map(|download| download["componentId"].as_str())
        .collect();
    assert_eq!(
        component_ids,
        std::collections::BTreeSet::from([
            "tokenizer_clip_l",
            "tokenizer_clip_bigg",
            "vae_fp16_fix",
        ]),
        "the raw job must reach Candle with every component its SDXL descriptor requires"
    );

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "candle-detail-worker",
            "gpuId": "0",
            "gpuName": "Candle GPU",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, claimed) = request(
        app,
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "candle-detail-worker" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "detail dispatches: {claimed}");
    assert_eq!(claimed["job"]["id"], job["id"]);
}

/// The OpenPose co-requisite belongs to Model Manager install/repair authority, not the generic
/// image worker payload. A normal no-pose SDXL request must carry only the provider's unconditional
/// hard components (three on Candle; none for the self-contained MLX package), and retry/duplicate
/// must preserve that projection at their shared canonicalization boundary.
#[tokio::test]
async fn ordinary_sdxl_txt2img_never_forwards_the_soft_openpose_component() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .expect("shipped model manifest writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "model": "realvisxl",
            "count": 1,
            "advanced": { "steps": 20 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "txt2img enqueues: {original}");
    let job_id = original["id"].as_str().expect("job id");
    let mut boundary_jobs = vec![("create", original.clone())];
    for operation in ["retry", "duplicate"] {
        let (status, replay) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({ "payloadChanges": { "advanced": { "steps": 21 } } }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {replay}");
        boundary_jobs.push((operation, replay));
    }

    for (boundary, job) in boundary_jobs {
        assert!(
            job["payload"]["advanced"].get("poses").is_none(),
            "{boundary}: the ordinary request must remain a no-pose job"
        );
        let co_requisites: Vec<&Value> = job["payload"]["modelManifestEntry"]["downloads"]
            .as_array()
            .expect("authoritative downloads array")
            .iter()
            .filter(|download| download["coRequisite"] == json!(true))
            .collect();
        let component_ids: std::collections::BTreeSet<&str> = co_requisites
            .iter()
            .filter_map(|download| download["componentId"].as_str())
            .collect();
        assert!(
            !component_ids.contains("controlnet_openpose"),
            "{boundary}: a no-pose job must not forward the soft OpenPose component: {component_ids:?}"
        );
        let hard_sdxl_components = std::collections::BTreeSet::from([
            "tokenizer_clip_l",
            "tokenizer_clip_bigg",
            "vae_fp16_fix",
        ]);
        assert_eq!(
            component_ids, hard_sdxl_components,
            "{boundary}: the authoritative payload carries only the exact hard SDXL metadata rows"
        );
        for download in &co_requisites {
            let platforms = download["platforms"]
                .as_array()
                .expect("hard SDXL co-requisites declare their platforms")
                .iter()
                .filter_map(Value::as_str)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(
                platforms,
                std::collections::BTreeSet::from(["windows", "linux"]),
                "{boundary}: carried hard SDXL metadata is Candle-only: {download}"
            );
        }
        #[cfg(target_os = "macos")]
        let target_platform = "macos";
        #[cfg(target_os = "windows")]
        let target_platform = "windows";
        #[cfg(target_os = "linux")]
        let target_platform = "linux";
        let applicable_component_ids = co_requisites
            .iter()
            .filter(|download| {
                download["platforms"].as_array().is_some_and(|platforms| {
                    platforms
                        .iter()
                        .any(|platform| platform.as_str() == Some(target_platform))
                })
            })
            .filter_map(|download| download["componentId"].as_str())
            .collect::<std::collections::BTreeSet<_>>();
        #[cfg(target_os = "macos")]
        assert!(
            applicable_component_ids.is_empty(),
            "{boundary}: carried Candle metadata is inapplicable to self-contained MLX SDXL: \
             {applicable_component_ids:?}"
        );
        #[cfg(not(target_os = "macos"))]
        assert_eq!(
            applicable_component_ids, hard_sdxl_components,
            "{boundary}: Candle receives exactly its three descriptor-required components"
        );
    }
}

#[tokio::test]
async fn raw_batch_detail_overwrites_untrusted_client_manifest_metadata() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .expect("shipped model manifest writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app,
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "requestedGpu": "auto",
            "payload": {
                "projectId": "project-1",
                "sourceAssetId": "asset-1",
                "model": "realvisxl",
                "advanced": { "strength": 0.55, "cnScale": 0.7 },
                "modelManifestEntry": {
                    "id": "client-spoof",
                    "family": "sdxl",
                    "downloads": [{
                        "coRequisite": true,
                        "componentId": "vae_fp16_fix",
                        "repo": "untrusted/arbitrary-repo",
                        "files": ["arbitrary.safetensors"]
                    }]
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "detail job enqueues: {job}");
    let entry = &job["payload"]["modelManifestEntry"];
    assert_eq!(entry["id"], "realvisxl");
    assert_eq!(entry["family"], "sdxl");
    assert!(
        !entry.to_string().contains("untrusted/arbitrary-repo"),
        "client manifest metadata must be replaced, never merged or trusted: {entry}"
    );
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn raw_batch_detail_rejects_every_explicit_packed_tier_carrier() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_shipped_image_model_manifests(temp_dir.path());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for advanced in [
        json!({ "mlxQuantize": 4 }),
        json!({ "mlxQuantize": 8 }),
        json!({ "convRot": true }),
        json!({ "quantTier": "nvfp4" }),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": "project-1",
                "requestedGpu": "auto",
                "payload": {
                    "projectId": "project-1",
                    "sourceAssetId": "asset-1",
                    "model": "realvisxl",
                    "advanced": advanced
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{advanced}: {body}");
        assert!(
            body["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("dense bf16")),
            "{advanced}: {body}"
        );
    }

    // A legacy no-model payload remains unmodified only while it carries no explicit packed
    // selection. Otherwise ImageRequest's fallback model would turn this into a hidden packed
    // RealVisXL request and bypass the route-owned Candle admission contract.
    for advanced in [
        json!({ "mlxQuantize": 4 }),
        json!({ "convRot": true }),
        json!({ "quantTier": "nvfp4" }),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "requestedGpu": "auto",
                "payload": { "advanced": advanced }
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "no-model {advanced}: {body}"
        );
        assert!(body["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("dense bf16")));
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().expect("jobs array").is_empty());
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn retry_and_duplicate_recanonicalize_batch_detail_manifest_and_dense_tier() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_shipped_image_model_manifests(temp_dir.path());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "requestedGpu": "auto",
            "payload": {
                "projectId": "project-1",
                "sourceAssetId": "asset-1",
                "model": "realvisxl",
                "advanced": { "strength": 0.55 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id");

    for operation in ["retry", "duplicate"] {
        let (status, replay) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "modelManifestEntry": {
                        "id": "client-spoof",
                        "family": "krea_2",
                        "modelPath": "C:/attacker/checkpoint.safetensors"
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {replay}");
        assert_eq!(replay["payload"]["modelManifestEntry"]["id"], "realvisxl");
        assert_eq!(replay["payload"]["modelManifestEntry"]["family"], "sdxl");
        assert_eq!(replay["payload"]["advanced"]["mlxQuantize"], 0);
        assert!(
            !replay["payload"].to_string().contains("client-spoof"),
            "{operation} must overwrite spoofed metadata: {replay}"
        );
    }

    for operation in ["retry", "duplicate"] {
        for advanced in [
            json!({ "mlxQuantize": 4 }),
            json!({ "convRot": true }),
            json!({ "quantTier": "nvfp4" }),
        ] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "advanced": advanced } }),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{operation} {advanced}: {body}"
            );
            assert!(body["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("dense bf16")));
        }
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs.as_array().map(Vec::len),
        Some(3),
        "only the original plus two canonical spoof replays may persist"
    );
}

#[tokio::test]
async fn retry_and_duplicate_recanonicalize_imported_generate_and_edit_manifests() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_imported_image_model_manifests(temp_dir.path());
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Imported image replay" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    for (mode, source) in [
        ("text_to_image", Value::Null),
        ("edit_image", json!("source-asset")),
    ] {
        let mut body = json!({
            "projectId": project_id,
            "mode": mode,
            "prompt": "a fox",
            "model": "imported_krea",
            "count": 1
        });
        if !source.is_null() {
            body["sourceAssetId"] = source;
        }
        let (status, original) = request(app.clone(), "POST", "/api/v1/image/jobs", body).await;
        assert_eq!(status, StatusCode::CREATED, "mode={mode}: {original}");
        assert_eq!(
            original["payload"]["modelManifestEntry"]["id"],
            "imported_krea"
        );
        let job_id = original["id"].as_str().expect("job id");

        for operation in ["retry", "duplicate"] {
            let (status, replay) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({
                    "payloadChanges": {
                        "modelManifestEntry": {
                            "id": "client-spoof",
                            "family": "sdxl",
                            "paths": { "model": "C:/attacker/other-model" }
                        }
                    }
                }),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{mode} {operation}: {replay}");
            let entry = &replay["payload"]["modelManifestEntry"];
            assert_eq!(entry["id"], "imported_krea");
            assert_eq!(entry["family"], "krea_2");
            assert!(
                entry["paths"]["model"]
                    .as_str()
                    .is_some_and(|path| path.contains("imported_krea")),
                "the authoritative imported install path must survive: {entry}"
            );
            assert!(
                !entry.to_string().contains("attacker"),
                "{mode} {operation} must replace the spoofed path: {entry}"
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn write_shipped_image_model_manifests(root: &std::path::Path) {
    let manifest_dir = root.join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .expect("shipped model manifest writes");
    write_empty_sibling_manifests(&manifest_dir);
}

fn write_imported_image_model_manifests(root: &std::path::Path) {
    let manifest_dir = root.join("config/manifests");
    let install_dir = root.join("data/models/imports/imported_krea");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::create_dir_all(&install_dir).expect("imported install dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin models write");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        format!(
            r#"{{
                "schemaVersion": 1,
                "models": [{{
                    "id": "imported_krea",
                    "name": "Imported Krea",
                    "type": "image",
                    "family": "krea_2",
                    "importSourceShape": "transformer_file",
                    "capabilities": ["text_to_image", "edit_image"],
                    "paths": {{ "model": "{}" }},
                    "defaults": {{ "count": 1, "resolution": "1024x1024" }},
                    "limits": {{}},
                    "loraCompatibility": {{ "families": ["krea_2"] }}
                }}]
            }}"#,
            install_dir.display().to_string().replace('\\', "\\\\")
        ),
    )
    .expect("user models write");
    for (name, key) in [
        ("builtin.loras.jsonc", "loras"),
        ("user.loras.jsonc", "loras"),
        ("builtin.recipe-presets.jsonc", "presets"),
        ("user.recipe-presets.jsonc", "presets"),
    ] {
        std::fs::write(
            manifest_dir.join(name),
            format!(r#"{{ "schemaVersion": 1, "{key}": [] }}"#),
        )
        .expect("empty sibling manifest writes");
    }
}

/// sc-13617 / F-055: raw jobs whose `model` reaches worker-side model path resolution must
/// reject traversal before a job row exists. This table is the API-side inventory for that key.
#[tokio::test]
async fn generic_model_backed_jobs_reject_path_unsafe_model_before_create() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for job_type in ["image_upscale", "image_detail", "prompt_refine"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "requestedGpu": "auto",
                "payload": { "model": "../../outside" },
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{job_type}: {body}");
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().expect("jobs array").is_empty());
}

/// sc-13617 / F-055: raw utility jobs whose `modelId` reaches worker-side model path
/// construction must reject traversal before persistence.
#[tokio::test]
async fn generic_model_utility_jobs_reject_path_unsafe_model_id_before_create() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for job_type in ["model_download", "model_import", "model_convert"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": job_type,
                "requestedGpu": "auto",
                "payload": { "modelId": "../../outside" },
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{job_type}: {body}");
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().expect("jobs array").is_empty());
}

#[tokio::test]
async fn raw_model_convert_rejects_unrecoverable_output_before_enqueue() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let conversion_root = settings.data_dir.join("models/mlx");
    let app = create_app(settings).expect("app creates");

    let request_convert = |output_dir: &std::path::Path| {
        json!({
            "type": "model_convert",
            "requestedGpu": "auto",
            "payload": {
                "modelId": "model-1",
                "sourceRepo": "owner/source",
                "outputDir": output_dir.display().to_string(),
            },
        })
    };

    let (status, accepted) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        request_convert(&conversion_root.join(".foo.finalize-backup-123")),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "backup-looking model names remain valid in the disjoint recovery design: {accepted}"
    );

    for (output_dir, expected) in [
        (conversion_root.join("nested/model-2"), "direct child"),
        (
            temp_dir.path().join("data/models/custom-model"),
            "direct child",
        ),
        (
            conversion_root.join(".sceneworks-finalize-backups"),
            "reserved",
        ),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            request_convert(&output_dir),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{output_dir:?}: {body}");
        assert!(
            body["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains(expected)),
            "{output_dir:?}: expected {expected:?} in {body}"
        );
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs.as_array().expect("jobs array").len(),
        1,
        "invalid conversion targets must fail before persistence"
    );
}

/// Write a two-variant convert-at-install family that SHARES one source repo — the Anima shape: each
/// card names its own `mlx.convertSourceFile` inside `owner/shared`, and the per-variant downloads are
/// serialized, so one variant's weights land while the other's are still streaming.
fn shared_repo_convert_manifest(config_dir: &std::path::Path) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    let models = ["alpha", "beta"]
        .into_iter()
        .map(|variant| {
            json!({
                "id": format!("fixture_{variant}"),
                "name": format!("Fixture {variant}"),
                "type": "image",
                "family": "fixture",
                "downloads": [{
                    "provider": "huggingface",
                    "repo": "owner/shared",
                    "files": [format!("split_files/diffusion_models/{variant}.safetensors")]
                }],
                "mlx": {
                    "requiresConversion": true,
                    "converter": "fixture_quant",
                    "convertSourceRepo": "owner/shared",
                    "convertSourceFile": format!("split_files/diffusion_models/{variant}.safetensors")
                }
            })
        })
        .collect::<Vec<_>>();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        serde_json::to_vec(&json!({ "schemaVersion": 1, "models": models }))
            .expect("manifest serializes"),
    )
    .expect("builtin models writes");
    write_empty_sibling_manifests(config_dir);
}

/// Seed `file` into the HF cache snapshot for `repo` under `data_dir`, the way a completed download
/// leaves it (`refs/main` → `snapshots/<rev>/<file>`).
fn seed_snapshot_file(data_dir: &std::path::Path, repo: &str, file: &str) {
    let revision = "a".repeat(40);
    let repo_dir = huggingface_repo_cache_path(data_dir, repo).expect("repo cache path");
    let snapshot = repo_dir.join("snapshots").join(&revision);
    let path = snapshot.join(file);
    std::fs::create_dir_all(path.parent().expect("snapshot parent")).expect("snapshot dir creates");
    std::fs::write(&path, b"weights").expect("source file writes");
    std::fs::create_dir_all(repo_dir.join("refs")).expect("refs dir creates");
    std::fs::write(repo_dir.join("refs").join("main"), &revision).expect("refs/main writes");
}

/// Michael, on-device (Anima 2B Turbo): the three Anima variants share `circlestone-labs/Anima` and
/// download one at a time, so the sibling cards flipped to *downloaded* the moment THEIR files landed.
/// Converting the variant whose 4 GB DiT was still streaming enqueued a job that the worker failed
/// instantly with a bare "Anima source DiT is missing." — indistinguishable from a real defect. The
/// convert request boundary must refuse it while it can still explain why, and must NOT block the
/// sibling whose weights ARE on disk.
#[tokio::test]
async fn model_convert_is_refused_while_its_source_weights_are_still_downloading() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let _env = isolate_hf_cache();
    shared_repo_convert_manifest(&temp_dir.path().join("config/manifests"));
    let settings = test_settings(&temp_dir);
    let data_dir = settings.data_dir.clone();
    let app = create_app(settings).expect("app creates");

    // Nothing cached at all: the source repo, not just one variant's file, is missing.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/fixture_alpha/convert",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["detail"].as_str().is_some_and(
            |detail| detail.contains("owner/shared") && detail.contains("not downloaded")
        ),
        "an uncached source repo must say so: {body}"
    );

    // Alpha's weights land; beta's are still streaming (the Anima situation).
    seed_snapshot_file(
        &data_dir,
        "owner/shared",
        "split_files/diffusion_models/alpha.safetensors",
    );
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/fixture_beta/convert",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["detail"].as_str().is_some_and(|detail| detail
            .contains("split_files/diffusion_models/beta.safetensors")
            && detail.contains("has not finished downloading")),
        "a shared-repo sibling's missing file must name the file, not the repo: {body}"
    );

    // The variant whose weights ARE cached converts — the gate refuses the unready one only.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/fixture_alpha/convert",
        json!({}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a cached source must still convert: {body}"
    );

    // An in-flight download for the model itself is reported as such, with its progress, even once
    // the file exists (a re-download of the same variant).
    let (status, download) = request(
        app.clone(),
        "POST",
        "/api/v1/models/fixture_alpha/download",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{download}");
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/models/fixture_alpha/convert",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("still downloading")),
        "an in-flight download for this model must be named as the reason: {body}"
    );
}

/// The route-derived output path is a trust-boundary input too. Confinement must run before
/// `model_catalog` performs its filesystem/cache sweep, so an invalid path-shaped ID wins over the
/// catalog's otherwise-observable "Model not found" response.
#[tokio::test]
async fn typed_model_convert_confines_route_id_before_catalog_lookup() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for (encoded_model_id, expected) in [
        ("%2E%2E", "direct child"),
        (".sceneworks-finalize-backups", "reserved"),
    ] {
        let endpoint = format!("/api/v1/models/{encoded_model_id}/convert");
        let (status, body) = request(app.clone(), "POST", &endpoint, json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{endpoint}: {body}");
        let detail = body["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains(expected),
            "{endpoint}: confinement must precede catalog/404 work, got {body}"
        );
        assert!(
            !detail.contains("Model not found"),
            "{endpoint}: invalid route ID must not fall through to catalog lookup"
        );
    }
}

/// sc-13617 / F-011: retry and duplicate merge payloadChanges into an existing generation
/// payload. Validate the merged model before either operation can persist a new job.
#[tokio::test]
async fn retry_and_duplicate_reject_path_unsafe_merged_generation_model_before_create() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, audio_job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "Safe original",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{audio_job}");

    let (status, vqa_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/vqa/jobs",
        json!({
            "projectId": project_id,
            "sourceAssetId": "asset-1",
            "question": "What is shown?",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{vqa_job}");

    let (status, interleave_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/interleave/jobs",
        json!({
            "projectId": project_id,
            "prompt": "Create a safe image",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{interleave_job}");

    for job in [&audio_job, &vqa_job, &interleave_job] {
        let job_id = job["id"].as_str().expect("job id");
        for operation in ["retry", "duplicate"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "model": "../../outside" } }),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{} {operation}: {body}",
                job["type"]
            );
        }
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().expect("jobs array").len(), 3);
}

/// sc-13639: retry/duplicate are alternate image-job creation boundaries. Their shallow
/// `payloadChanges` merge must not let a caller manufacture the catalog-only authorization bit,
/// a hosted tuple, or an app-managed local path that the original typed image route would reject.
#[tokio::test]
async fn retry_and_duplicate_reauthorize_merged_control_weights_before_create() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Control Retry Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id");

    let forged_hosted = json!({
        "advanced": {
            "controlWeights": {
                "overlayId": "forged-overlay",
                "_catalogAuthorized": true,
                "repo": "attacker/weights",
                "filename": "payload.safetensors",
                "revision": "0123456789abcdef0123456789abcdef01234567"
            }
        }
    });
    let forged_path = json!({
        "advanced": {
            "controlWeights": {
                "path": temp_dir.path()
                    .join("data/models/krea_2/control.safetensors")
                    .display()
                    .to_string(),
                "_catalogAuthorized": true
            }
        }
    });
    for operation in ["retry", "duplicate"] {
        for injection in [&forged_hosted, &forged_path] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": injection }),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{operation} must reject {injection}: {body}"
            );
        }
    }

    // The one shipped SDXL OpenPose tuple is accepted at every alternate create boundary, while
    // caller-forged authorization/revision fields are removed and never persisted.
    for operation in ["retry", "duplicate"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "advanced": {
                        "controlWeights": {
                            "repo": "xinsir/controlnet-openpose-sdxl-1.0",
                            "filename": "diffusion_pytorch_model.safetensors",
                            "revision": "0123456789abcdef0123456789abcdef01234567",
                            "_catalogAuthorized": true
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {body}");
        let weights = body["payload"]["advanced"]["controlWeights"]
            .as_object()
            .expect("canonical controlWeights");
        assert_eq!(
            weights.get("repo").and_then(Value::as_str),
            Some("xinsir/controlnet-openpose-sdxl-1.0")
        );
        assert_eq!(
            weights.get("filename").and_then(Value::as_str),
            Some("diffusion_pytorch_model.safetensors")
        );
        assert!(!weights.contains_key("revision"));
        assert!(!weights.contains_key("_catalogAuthorized"));
    }

    // Legitimate operations retain their existing success semantics after the merged payload is
    // canonicalized; rejected injections above must not have persisted hidden jobs.
    for operation in ["retry", "duplicate"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({ "payloadChanges": { "prompt": format!("safe {operation}") } }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {body}");
        assert_eq!(
            body["payload"]["prompt"],
            format!("safe {operation}"),
            "{operation} must persist the clean merged payload"
        );
    }
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs.as_array().expect("jobs array").len(),
        5,
        "only the original, two shipped-control operations, and two clean operations may persist"
    );
}

/// SC-18314: the browser authors only an opaque encoder id. Every image-create boundary must
/// discard caller/persisted resolution metadata, resolve the id against current server state, and
/// reject an id that has disappeared instead of silently substituting the model encoder.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn image_create_retry_and_duplicate_resolve_text_encoder_fresh_and_fail_closed() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    single_model_manifest(&manifest_dir, "krea_2_turbo", "SceneWorks/krea-2-turbo-mlx");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let stale_id = "text_encoder_ffffffffffffffffffffffffffffffff";
    let base_payload = json!({
        "projectId": "project-1",
        "mode": "text_to_image",
        "prompt": "mist over hills",
        "model": "krea_2_turbo",
        "count": 1,
        "width": 1024,
        "height": 1024
    });
    let mut stale_create = base_payload.clone();
    stale_create["advanced"] = json!({ "textEncoderModel": stale_id });
    // A typed create must ignore any client attempt to carry the private resolution and reject the
    // unavailable authored id from a fresh catalog lookup.
    stale_create["modelManifestEntry"] = json!({
        "resolvedTextEncoder": {
            "selectionId": stale_id,
            "sourceKind": "directory",
            "path": temp_dir.path().join("attacker-selected")
        }
    });
    let (status, body) = request(app.clone(), "POST", "/api/v1/image/jobs", stale_create).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("is unavailable")));

    let (status, original) = request(app.clone(), "POST", "/api/v1/image/jobs", base_payload).await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    assert!(original["payload"]["modelManifestEntry"]
        .get("resolvedTextEncoder")
        .is_none());
    let job_id = original["id"].as_str().expect("job id");

    for operation in ["retry", "duplicate"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "advanced": { "textEncoderModel": stale_id },
                    "modelManifestEntry": {
                        "resolvedTextEncoder": {
                            "selectionId": stale_id,
                            "sourceKind": "directory",
                            "path": temp_dir.path().join("attacker-selected")
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {body}");
        assert!(body["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("is unavailable")));

        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "modelManifestEntry": {
                        "resolvedTextEncoder": {
                            "selectionId": stale_id,
                            "sourceKind": "directory",
                            "path": temp_dir.path().join("attacker-selected")
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {body}");
        assert!(body["payload"]["modelManifestEntry"]
            .get("resolvedTextEncoder")
            .is_none());
    }
}

/// SC-18314: server resolution is worker-private. Exercise the raw queue primitive with the exact
/// typed-image metadata shape so every generic job projection is covered without depending on a
/// platform provider fixture. The raw store and `/jobs/claim` must retain the resolution; every
/// browser-visible HTTP/SSE shape must retain only the authored opaque id.
#[tokio::test]
async fn public_job_boundaries_hide_selected_text_encoder_path_but_worker_claim_retains_it() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) =
        create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
    let selected = temp_dir.path().join("server/private/selected.safetensors");
    let selected_parent = selected.parent().expect("selected parent").to_path_buf();
    let distinct_server_root = temp_dir.path().join("other/managed/models");
    let distinct_canonical_target = temp_dir.path().join("outside/canonical/encoder.bin");
    let selection_id = "text_encoder_0123456789abcdef0123456789abcdef";
    let job_payload = json!({
        "prompt": "/imagine a lake",
        "installedPath": "/public/models/base.safetensors",
        "sourcePath": "/public/loras/style.safetensors",
        "selectedEcho": selected.display().to_string(),
        "advanced": { "textEncoderModel": selection_id },
        "modelManifestEntry": {
            "resolvedTextEncoder": {
                "selectionId": selection_id,
                "sourceKind": "file",
                "path": selected
            }
        }
    });
    let create_request = json!({
        "type": "image_detail",
        "projectId": "project-1",
        "projectName": "Project 1",
        "payload": job_payload,
        "requestedGpu": "auto"
    });
    let assert_public = |surface: &str, value: &Value| {
        let encoded = value.to_string();
        assert!(
            !encoded.contains("resolvedTextEncoder"),
            "{surface} exposed server-private resolution: {value}"
        );
        assert!(
            !encoded.contains(selected.to_string_lossy().as_ref()),
            "{surface} exposed selected filesystem path: {value}"
        );
        assert!(
            !encoded.contains(selected_parent.to_string_lossy().as_ref()),
            "{surface} exposed selected filesystem prefix: {value}"
        );
        assert!(
            !encoded.contains(distinct_server_root.to_string_lossy().as_ref()),
            "{surface} exposed an allowed model root: {value}"
        );
        assert!(
            !encoded.contains(distinct_canonical_target.to_string_lossy().as_ref()),
            "{surface} exposed a distinct canonical target: {value}"
        );
    };

    let mut events = state.events.subscribe();
    let (status, created) =
        request(app.clone(), "POST", "/api/v1/jobs", create_request.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_public("create response", &created);
    assert_eq!(
        created["payload"]["advanced"]["textEncoderModel"],
        selection_id
    );
    assert_eq!(created["payload"]["prompt"], "/imagine a lake");
    assert_eq!(
        created["payload"]["installedPath"],
        "/public/models/base.safetensors"
    );
    assert_eq!(
        created["payload"]["sourcePath"],
        "/public/loras/style.safetensors"
    );
    assert_eq!(
        created["payload"]["selectedEcho"],
        "[selected text encoder]"
    );
    let job_id = created["id"].as_str().expect("job id").to_owned();
    let raw = state.jobs_store.get_job(&job_id).expect("raw job reads");
    assert_eq!(
        raw.payload["modelManifestEntry"]["resolvedTextEncoder"]["path"],
        selected.display().to_string(),
        "public projection must not mutate the worker-owned stored row"
    );
    assert_eq!(
        raw.payload["selectedEcho"],
        selected.display().to_string(),
        "the raw worker payload must retain an exact selected-path echo"
    );
    let mut raw_with_extra = raw.clone();
    raw_with_extra.status = sceneworks_core::contracts::JobStatus::Failed;
    raw_with_extra.extra.insert(
        "partialAssetPath".to_owned(),
        Value::String("/public/outputs/partial.png".to_owned()),
    );
    let projected_extra =
        serde_json::to_value(crate::public_job_snapshot(raw_with_extra)).expect("job serializes");
    assert_eq!(
        projected_extra["partialAssetPath"], "/public/outputs/partial.png",
        "unrelated partial output paths remain public contract data"
    );

    for expected in ["job.updated", "queue.updated"] {
        let event = tokio::time::timeout(Duration::from_secs(1), events.next())
            .await
            .expect("create event arrives")
            .expect("event stream remains open");
        assert_eq!(event.event, expected);
        assert_public(
            &format!("live {expected}"),
            &serde_json::from_str(&event.data).expect("event data parses"),
        );
    }

    let (status, listed) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_public("list response", &listed);
    let (status, fetched) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_public("get response", &fetched);
    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_public("queue response", &queue);

    let (status, reconnect) = request_sse_prefix(app.clone(), "/api/v1/jobs/events", 3).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reconnect[1].0, "jobs.snapshot");
    assert_public("reconnect jobs.snapshot", &reconnect[1].1);
    assert_eq!(reconnect[2].0, "queue.updated");
    assert_public("reconnect queue.updated", &reconnect[2].1);

    for operation in ["retry", "duplicate"] {
        let (status, response) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({ "payloadChanges": { "prompt": format!("safe {operation}") } }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {response}");
        assert_public(&format!("{operation} response"), &response);
    }

    let (status, canceled_one) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{canceled_one}");
    assert_public("single cancel response", &canceled_one);
    let (status, cleared_one) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{cleared_one}");
    assert_public("single clear response", &cleared_one);

    let (status, canceled) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/cancel-pending",
        json!({ "projectId": "project-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{canceled}");
    assert_public("cancel-pending response", &canceled);

    // Seed a fresh row for the one private boundary: a compatible worker claim. The raw payload
    // must survive public projection intact so the worker can validate and prepare its receipt.
    let (status, worker_job) =
        request(app.clone(), "POST", "/api/v1/jobs", create_request.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{worker_job}");
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "Test GPU",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{claimed}");
    assert_eq!(
        claimed["job"]["payload"]["modelManifestEntry"]["resolvedTextEncoder"]["path"],
        selected.display().to_string(),
        "the worker claim must retain the server-private exact source"
    );
    let claimed_id = claimed["job"]["id"].as_str().expect("claimed id");
    while tokio::time::timeout(Duration::from_millis(25), events.next())
        .await
        .is_ok()
    {}
    let (status, progress) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{claimed_id}/progress"),
        json!({
            "status": "running",
            "stage": "running",
            "progress": 0.5,
            "message": "halfway",
            "workerId": "worker-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{progress}");
    assert_public("progress response", &progress);
    let progress_event = tokio::time::timeout(Duration::from_secs(1), events.next())
        .await
        .expect("progress job.updated arrives")
        .expect("event stream remains open");
    assert_eq!(progress_event.event, "job.updated");
    assert_public(
        "progress job.updated",
        &serde_json::from_str(&progress_event.data).expect("progress event parses"),
    );
    let progress_queue = tokio::time::timeout(Duration::from_secs(1), events.next())
        .await
        .expect("progress queue.updated arrives")
        .expect("event stream remains open");
    assert_eq!(progress_queue.event, "queue.updated");
    assert_public(
        "progress queue.updated",
        &serde_json::from_str(&progress_queue.data).expect("progress queue event parses"),
    );

    let private_error = format!(
        "Selected text encoder must be inside an app-managed directory ({}, {}). Pinned target changed from {} to {}",
        selected_parent.display(),
        distinct_server_root.display(),
        selected.display(),
        distinct_canonical_target.display()
    );
    let private_result = json!({
        "partialAssetPath": "/public/outputs/partial.png",
        "selectedReceipt": selected.display().to_string()
    });
    let (status, failed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{claimed_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 1,
            "message": format!("Selected encoder failed at {}", selected.display()),
            "error": private_error,
            "result": private_result,
            "workerId": "worker-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{failed}");
    assert_public("failed progress response", &failed);
    assert!(failed["error"]
        .as_str()
        .is_some_and(|error| error.contains("Selected text encoder must be inside")));
    assert!(failed["error"]
        .as_str()
        .is_some_and(|error| error.contains("[selected text encoder]")));
    assert_eq!(
        failed["result"]["partialAssetPath"],
        "/public/outputs/partial.png"
    );
    assert_eq!(
        failed["result"]["selectedReceipt"],
        "[selected text encoder]"
    );
    let raw_failed = state
        .jobs_store
        .get_job(claimed_id)
        .expect("raw failure reads");
    assert_eq!(raw_failed.error.as_deref(), Some(private_error.as_str()));
    assert!(raw_failed
        .message
        .contains(selected.to_string_lossy().as_ref()));
    assert_eq!(
        raw_failed.result["selectedReceipt"],
        selected.display().to_string()
    );
    let failed_event = tokio::time::timeout(Duration::from_secs(1), events.next())
        .await
        .expect("failed job.updated arrives")
        .expect("event stream remains open");
    assert_eq!(failed_event.event, "job.updated");
    assert_public(
        "failed job.updated",
        &serde_json::from_str(&failed_event.data).expect("failed event parses"),
    );
    let (status, failed_get) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{claimed_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_public("failed get response", &failed_get);
    let (status, failed_list) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_public("failed list response", &failed_list);
    let (status, failed_reconnect) =
        request_sse_prefix(app.clone(), "/api/v1/jobs/events", 3).await;
    assert_eq!(status, StatusCode::OK);
    assert_public("failed reconnect jobs.snapshot", &failed_reconnect[1].1);
    assert_public("failed reconnect queue.updated", &failed_reconnect[2].1);
    let (status, cleared_terminal) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{claimed_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{cleared_terminal}");
    assert_public("terminal clear response", &cleared_terminal);

    // The supervisor crash report wraps its snapshot in `Option<JobSnapshot>` rather than using
    // the ordinary progress response. Claim one final raw row so that container boundary is also
    // proven public while the persisted worker payload remains exact.
    let (status, termination_job) =
        request(app.clone(), "POST", "/api/v1/jobs", create_request).await;
    assert_eq!(status, StatusCode::CREATED, "{termination_job}");
    let termination_job_id = termination_job["id"]
        .as_str()
        .expect("termination job id")
        .to_owned();
    let (status, termination_claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{termination_claim}");
    assert_eq!(termination_claim["job"]["id"], termination_job_id);
    let (status, terminated) = request(
        app,
        "POST",
        "/api/v1/workers/worker-1/terminated",
        json!({ "signal": 9, "exitCode": null }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{terminated}");
    assert_eq!(terminated["id"], termination_job_id);
    assert_public("worker-terminated response", &terminated);
    let raw_terminated = state
        .jobs_store
        .get_job(&termination_job_id)
        .expect("raw terminated job reads");
    assert_eq!(
        raw_terminated.payload["modelManifestEntry"]["resolvedTextEncoder"]["path"],
        selected.display().to_string(),
        "worker-termination projection must not mutate the stored worker payload"
    );
}

#[test]
fn selected_encoder_root_file_parent_is_never_a_universal_redaction_prefix() {
    #[cfg(unix)]
    let selected = std::path::Path::new("/selected.safetensors");
    #[cfg(windows)]
    let selected = std::path::Path::new(r"C:\selected.safetensors");

    let cases = [
        (
            "https://example.com/models/help",
            "https://example.com/models/help",
        ),
        (
            "keep this/that slash-bearing prose",
            "keep this/that slash-bearing prose",
        ),
        ("/", "/"),
        ("root: / and keep this", "root: / and keep this"),
        ("root: /; keep this", "root: /; keep this"),
        (r#"root: "/" and keep this"#, r#"root: "/" and keep this"#),
        (
            "/another/private/models/escaped.safetensors",
            "[selected text encoder]",
        ),
    ];
    for (input, expected) in cases {
        let mut actual = input.to_owned();
        crate::redact_private_text_encoder_diagnostic(&mut actual);
        assert_eq!(actual, expected, "input: {input}");
    }

    let mut selected = selected.display().to_string();
    crate::redact_private_text_encoder_diagnostic(&mut selected);
    assert_eq!(selected, "[selected text encoder]");
}

#[cfg(unix)]
#[test]
fn selected_encoder_symlink_canonical_target_is_redacted_without_using_root() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let lexical_parent = temp_dir.path().join("managed/text-encoders");
    let escaped_parent = temp_dir.path().join("outside-root");
    std::fs::create_dir_all(&lexical_parent).expect("lexical parent creates");
    std::fs::create_dir_all(&escaped_parent).expect("escaped parent creates");
    let canonical_target = escaped_parent.join("target.safetensors");
    std::fs::write(&canonical_target, b"sentinel").expect("target writes");
    let lexical = lexical_parent.join("selected.safetensors");
    std::os::unix::fs::symlink(&canonical_target, &lexical).expect("symlink creates");

    let mut diagnostic = format!(
        "Selected text encoder resolved from {} to {}; allowed roots: {}",
        lexical.display(),
        canonical_target.display(),
        temp_dir.path().join("another/model/root").display()
    );
    crate::redact_private_text_encoder_diagnostic(&mut diagnostic);
    assert!(!diagnostic.contains(temp_dir.path().to_string_lossy().as_ref()));
    assert!(diagnostic.contains("Selected text encoder resolved from"));
    assert!(diagnostic.contains("[selected text encoder]"));
}

#[test]
fn selected_encoder_path_token_scrubber_handles_wrappers_and_preserves_web_urls() {
    let cases = [
        ("source:/private/model", "source:[selected text encoder]"),
        ("`/private/model with spaces`", "[selected text encoder]"),
        ("</private/model>", "[selected text encoder]"),
        (
            r"source=C:\Models\private\model",
            "source=[selected text encoder]",
        ),
        (
            r"source=\\server\share\private\model",
            "source=[selected text encoder]",
        ),
        ("file:///private/model", "[selected text encoder]"),
        (
            "https://example.com/private/model",
            "https://example.com/private/model",
        ),
        (
            "//cdn.example.com/private/model",
            "//cdn.example.com/private/model",
        ),
        ("relative/model", "relative/model"),
        ("./relative/model", "./relative/model"),
        ("../relative/model", "../relative/model"),
        ("~/relative/model", "~/relative/model"),
        ("${HOME}/relative/model", "${HOME}/relative/model"),
        (
            "/Volumes/External Models/encoder.safetensors changed",
            "[selected text encoder]",
        ),
        (
            "at /tmp/private see https://example.com/public",
            "at [selected text encoder]",
        ),
        ("/💾", "[selected text encoder]"),
        ("/_", "[selected text encoder]"),
        ("/...", "[selected text encoder]"),
        ("/", "/"),
        (r"C:\", r"C:\"),
    ];

    for (input, expected) in cases {
        let mut actual = input.to_owned();
        crate::redact_private_text_encoder_diagnostic(&mut actual);
        assert_eq!(actual, expected, "input: {input}");
    }
}

#[test]
fn selected_encoder_exact_scrub_obeys_file_and_directory_component_boundaries() {
    let file = crate::private_text_encoder_path_spellings("/models/x.safetensors", Some("file"));
    let windows_file =
        crate::private_text_encoder_path_spellings("C:/Models/X.safetensors", Some("file"));
    let windows_unc = crate::private_text_encoder_path_spellings(
        r"\\Server\Share\Encoder.safetensors",
        Some("file"),
    );
    let encoded_file = crate::private_text_encoder_path_spellings(
        "/Volumes/External Models/x.safetensors",
        Some("file"),
    );
    let unix_backslash = crate::private_text_encoder_path_spellings(r"/models/a\b", Some("file"));
    let directory =
        crate::private_text_encoder_path_spellings("/models/encoder", Some("directory"));
    let trailing_directory =
        crate::private_text_encoder_path_spellings("/models/encoder/", Some("directory"));
    let cases = [
        (&file, "/models/x.safetensors", "[selected text encoder]"),
        (
            &file,
            "file:///models/x.safetensors",
            "[selected text encoder]",
        ),
        (
            &windows_file,
            "file:///C:/Models/X.safetensors",
            "[selected text encoder]",
        ),
        (
            &windows_file,
            "file:/c:/models/x.safetensors",
            "[selected text encoder]",
        ),
        (
            &windows_file,
            r"c:\models\x.safetensors",
            "[selected text encoder]",
        ),
        (
            &windows_unc,
            "//server/share/encoder.safetensors",
            "[selected text encoder]",
        ),
        (
            &encoded_file,
            "file:///Volumes/External%20Models/x.safetensors",
            "[selected text encoder]",
        ),
        (
            &encoded_file,
            "file:///Volumes/%e2%98%83/x.safetensors",
            "[selected text encoder]",
        ),
        (
            &encoded_file,
            "/Volumes/External%20Models/x.safetensors",
            "/Volumes/External%20Models/x.safetensors",
        ),
        (
            &file,
            "/models/x.safetensors.backup",
            "/models/x.safetensors.backup",
        ),
        (&file, "/models/x.safetensors.", "[selected text encoder]."),
        (&file, "/models/x.safetensors!", "[selected text encoder]!"),
        (
            &file,
            "xhttp:///models/x.safetensors",
            "xhttp://[selected text encoder]",
        ),
        (
            &directory,
            "/models/encoder changed",
            "[selected text encoder] changed",
        ),
        (
            &directory,
            "/models/encoder/shard.safetensors",
            "[selected text encoder]/shard.safetensors",
        ),
        (
            &trailing_directory,
            "/models/encoder/shard.safetensors",
            "[selected text encoder]/shard.safetensors",
        ),
        (&directory, "/models/encoder-v2", "/models/encoder-v2"),
        (
            &directory,
            "/backup/models/encoder",
            "/backup/models/encoder",
        ),
        (
            &directory,
            "https://example.com/models/encoder",
            "https://example.com/models/encoder",
        ),
        (&unix_backslash, "/models/a/b", "/models/a/b"),
    ];

    for (spellings, input, expected) in cases {
        let mut actual = input.to_owned();
        crate::redact_selected_text_encoder_paths(&mut actual, spellings);
        assert_eq!(actual, expected, "input: {input}");
    }

    let unknown_directory = crate::private_text_encoder_path_spellings("/models/unknown", None);
    let mut unknown_descendant = "/models/unknown/shard.safetensors".to_owned();
    crate::redact_selected_text_encoder_paths(&mut unknown_descendant, &unknown_directory);
    assert_eq!(
        unknown_descendant,
        "[selected text encoder]/shard.safetensors"
    );

    let posix_double_slash =
        crate::private_text_encoder_path_spellings("//mnt/Encoder", Some("file"));
    let mut case_distinct_posix = "//mnt/encoder".to_owned();
    crate::redact_selected_text_encoder_paths(&mut case_distinct_posix, &posix_double_slash);
    assert_eq!(case_distinct_posix, "//mnt/encoder");
}

#[test]
fn serialize_job_lora_carries_network_type_to_payload() {
    // A trained LoKr adapter records networkType (epic 2193); the generation
    // payload must carry it so the worker can route LoKr off the MLX backend
    // without opening the file.
    let lora = json!({
        "id": "char",
        "family": "sdxl",
        "networkType": "lokr",
        "source": { "provider": "training" },
    });
    let payload = serialize_job_lora(&lora, &json!({}), "char");
    assert_eq!(
        payload.get("networkType").and_then(Value::as_str),
        Some("lokr")
    );

    // A plain LoRA without the field stays absent/null (treated as lora downstream).
    let plain = serialize_job_lora(&json!({ "id": "x", "family": "sdxl" }), &json!({}), "x");
    assert!(plain.get("networkType").map(Value::is_null).unwrap_or(true));
}

#[test]
fn serialize_job_lora_carries_accelerator_role_to_payload() {
    // The Krea 2 turbo accelerator LoRA (sc-13882) records `role: accelerator`; the generation payload
    // must carry it so the worker can switch a Krea 2 Raw t2i job to the turbo sampling regime (epic
    // 13879 S3, sc-13883) — the sampling-regime sibling of `conditioningRole`.
    let lora = json!({
        "id": "krea2_turbo_accel",
        "family": "krea_2",
        "role": "accelerator",
        "source": { "provider": "huggingface" },
    });
    let payload = serialize_job_lora(&lora, &json!({}), "krea2_turbo_accel");
    assert_eq!(
        payload.get("role").and_then(Value::as_str),
        Some("accelerator")
    );

    // A plain LoRA without the field stays absent/null (a plain additive residual downstream).
    let plain = serialize_job_lora(&json!({ "id": "x", "family": "krea_2" }), &json!({}), "x");
    assert!(plain.get("role").map(Value::is_null).unwrap_or(true));
}

#[test]
fn person_readiness_reflects_live_worker_capabilities() {
    let workers = vec![
        readiness_worker(
            "gpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetect,
                WorkerCapability::PersonTrack,
                WorkerCapability::PersonReplace,
            ],
        ),
        readiness_worker(
            "cpu",
            WorkerStatus::Idle,
            vec![
                WorkerCapability::PersonDetectPreview,
                WorkerCapability::PersonTrackPreview,
            ],
        ),
        // Segment capability exists only on an offline worker -> not ready.
        readiness_worker(
            "dead",
            WorkerStatus::Offline,
            vec![WorkerCapability::PersonSegment],
        ),
    ];
    let readiness = person_readiness_from_workers(&workers);
    assert_eq!(readiness["detect"]["ready"], json!(true));
    assert_eq!(readiness["detect"]["capability"], json!("person_detect"));
    assert_eq!(readiness["track"]["ready"], json!(true));
    assert_eq!(readiness["replace"]["ready"], json!(true));
    assert_eq!(readiness["detectPreview"]["ready"], json!(true));
    assert_eq!(readiness["segment"]["ready"], json!(false));
}

/// sc-16260: an `unhealthy` worker is heartbeating, so it is NOT offline — but it has withdrawn
/// every capability it serves and will claim nothing. Readiness must exclude it, or the UI ungates
/// Replace Person on a host whose GPU cannot be initialized.
///
/// Pinned against a worker that still advertises the capability, which is what makes this a real
/// gate rather than a restatement of the capability check: the SAME advertisement reads ready when
/// idle and not-ready when unhealthy.
#[test]
fn person_readiness_excludes_an_unhealthy_worker() {
    let advertised = vec![
        WorkerCapability::PersonDetect,
        WorkerCapability::PersonTrack,
    ];

    let idle = vec![readiness_worker(
        "gpu",
        WorkerStatus::Idle,
        advertised.clone(),
    )];
    assert_eq!(
        person_readiness_from_workers(&idle)["detect"]["ready"],
        json!(true),
        "control: the identical advertisement must read ready while the worker is idle"
    );

    let unhealthy = vec![readiness_worker("gpu", WorkerStatus::Unhealthy, advertised)];
    let readiness = person_readiness_from_workers(&unhealthy);
    assert_eq!(
        readiness["detect"]["ready"],
        json!(false),
        "an unhealthy worker cannot run person detection, whatever its registration still says"
    );
    assert_eq!(readiness["track"]["ready"], json!(false));
}

#[tokio::test]
async fn create_image_job_rejects_over_length_negative_prompt() {
    // sc-8884 (F-082): negativePrompt now shares the prompt char cap.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "negativePrompt": "n".repeat(4001),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("negativePrompt")));
}

#[tokio::test]
async fn create_image_job_rejects_oversized_advanced_object() {
    // sc-8884 (F-082): the free-form `advanced` bag is bounded by serialized size.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "advanced": { "blob": "a".repeat(64 * 1024 + 1) },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("advanced")));
}

#[tokio::test]
async fn image_prompt_enhancement_is_typed_bounded_and_route_scoped() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let body = |model: &str, advanced: Value| {
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "model": model,
            "prompt": "mist over hills",
            "count": 1,
            "advanced": advanced,
        })
    };

    let (status, initial) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        body(
            "flux2_dev",
            json!({
                "enhancePrompt": true,
                "enhanceTemperature": 0.2,
                "enhanceMaxTokens": 2048,
            }),
        ),
    )
    .await;
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    assert_eq!(status, StatusCode::CREATED, "{initial}");
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{initial}");
        assert!(initial["detail"]
            .as_str()
            .is_some_and(|value| value.contains("native MLX or Candle")));
    }

    for (advanced, detail) in [
        (json!({ "enhancePrompt": "yes" }), "must be a boolean"),
        (
            json!({ "enhancePrompt": true, "enhanceTemperature": 2.01 }),
            "must be between 0 and 2",
        ),
        (
            json!({ "enhancePrompt": true, "enhanceMaxTokens": 2049 }),
            "must be between 1 and 2048",
        ),
        (
            json!({ "enhancePrompt": false, "enhanceMaxTokens": 64 }),
            "requires advanced.enhancePrompt=true",
        ),
        (
            json!({
                "enhancePrompt": true,
                "promptEnhancement": { "outcome": "enhanced" },
            }),
            "worker-owned",
        ),
    ] {
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            body("flux2_dev", advanced),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(
            error["detail"]
                .as_str()
                .is_some_and(|value| value.contains(detail)),
            "{error}"
        );
    }

    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        body("flux2_klein_9b", json!({ "enhancePrompt": true })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error["detail"]
        .as_str()
        .is_some_and(|value| value.contains("FLUX.2-Klein")));

    for strict_control in [
        json!({ "poses": [{ "id": "pose-1" }] }),
        json!({ "controlWeights": { "overlayId": "flux2-depth" } }),
        json!({ "controlImage": "asset-1" }),
        json!({ "controlMode": "depth" }),
    ] {
        let mut advanced = strict_control.as_object().unwrap().clone();
        advanced.insert("enhancePrompt".to_owned(), json!(true));
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            body("flux2_dev", Value::Object(advanced)),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"]
            .as_str()
            .is_some_and(|value| value.contains("strict control")));
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        // Retry and duplicate validate the exact shallow-merged payload they will enqueue. A valid
        // dev job therefore cannot be replayed as Klein while retaining its enhancement request.
        let job_id = initial["id"].as_str().expect("created job id");
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "model": "flux2_klein_9b" } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(error["detail"]
                .as_str()
                .is_some_and(|value| value.contains("FLUX.2-Klein")));
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        // Candle owns exactly the native base + bespoke edit routes. Character/style modes and
        // legacy reference aliases must never reach the generic base renderer and drop their input.
        for mode in ["character_image", "style_variations"] {
            let mut payload = body("flux2_dev", json!({ "enhancePrompt": true }));
            payload["mode"] = json!(mode);
            payload["referenceAssetId"] = json!("reference-1");
            let (status, error) = request(app.clone(), "POST", "/api/v1/image/jobs", payload).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "mode={mode}: {error}");
            assert!(error["detail"]
                .as_str()
                .is_some_and(|value| value.contains("Candle supports only")));
        }

        for carrier in [
            json!({ "sourceAssetId": "source-1" }),
            json!({ "referenceAssetId": "reference-1" }),
            json!({ "referenceAssetIds": ["reference-1"] }),
        ] {
            let mut payload = body("flux2_dev", json!({ "enhancePrompt": true }));
            payload
                .as_object_mut()
                .unwrap()
                .extend(carrier.as_object().unwrap().clone());
            let (status, error) = request(app.clone(), "POST", "/api/v1/image/jobs", payload).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
            assert!(error["detail"]
                .as_str()
                .is_some_and(|value| value.contains("cannot include source or reference")));
        }

        let mut missing_edit_input = body("flux2_dev", json!({ "enhancePrompt": true }));
        missing_edit_input["mode"] = json!("edit_image");
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            missing_edit_input,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"]
            .as_str()
            .is_some_and(|value| value.contains("requires a source or reference")));

        let mut valid_edit = body("flux2_dev", json!({ "enhancePrompt": true }));
        valid_edit["mode"] = json!("edit_image");
        valid_edit["sourceAssetId"] = json!("source-1");
        let (status, edit) = request(app.clone(), "POST", "/api/v1/image/jobs", valid_edit).await;
        assert_eq!(status, StatusCode::CREATED, "{edit}");

        let job_id = initial["id"].as_str().expect("created job id");
        for operation in ["retry", "duplicate"] {
            for mode in [
                "character_image",
                "style_variations",
                "reference",
                "image_to_image",
            ] {
                let (status, error) = request(
                    app.clone(),
                    "POST",
                    &format!("/api/v1/jobs/{job_id}/{operation}"),
                    json!({
                        "payloadChanges": {
                            "mode": mode,
                            "referenceAssetId": "reference-1"
                        }
                    }),
                )
                .await;
                assert_eq!(
                    status,
                    StatusCode::BAD_REQUEST,
                    "{operation} mode={mode}: {error}"
                );
                assert!(error["detail"]
                    .as_str()
                    .is_some_and(|value| value.contains("Candle supports only")));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // MLX legitimately owns all four surfaced modes, provided every edit-like mode carries the
        // reference that selects its native edit route.
        for (mode, carrier) in [
            ("edit_image", json!({ "sourceAssetId": "source-1" })),
            (
                "character_image",
                json!({ "referenceAssetId": "reference-1" }),
            ),
            (
                "style_variations",
                json!({ "referenceAssetIds": ["reference-1"] }),
            ),
        ] {
            let mut payload = body("flux2_dev", json!({ "enhancePrompt": true }));
            payload["mode"] = json!(mode);
            payload
                .as_object_mut()
                .unwrap()
                .extend(carrier.as_object().unwrap().clone());
            let (status, created) =
                request(app.clone(), "POST", "/api/v1/image/jobs", payload).await;
            assert_eq!(status, StatusCode::CREATED, "mode={mode}: {created}");
        }
    }

    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        // A backendless API cannot enqueue enhancement directly or resurrect it through a replay.
        let (status, plain) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            body("flux2_dev", json!({})),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{plain}");
        let job_id = plain["id"].as_str().expect("plain job id");
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "advanced": { "enhancePrompt": true } } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(error["detail"]
                .as_str()
                .is_some_and(|value| value.contains("native MLX or Candle")));
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
async fn post_preset_prompt_enhancement_uses_the_resolved_candle_route() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "z_image_turbo",
              "name": "Z-Image",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image", "character_image", "style_variations"],
              "downloads": [], "paths": {}, "defaults": {}, "limits": {}, "ui": {}
            },
            {
              "id": "flux2_dev",
              "name": "FLUX.2 Dev",
              "family": "flux2",
              "type": "image",
              "adapter": "flux2_diffusers",
              "capabilities": ["text_to_image", "edit_image", "character_image", "style_variations"],
              "downloads": [], "paths": {}, "defaults": {}, "limits": {}, "ui": { "promptEnhance": true }
            }
          ]
        }
        "#,
    )
    .expect("builtin models write");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models write");
    for name in ["builtin.loras.jsonc", "user.loras.jsonc"] {
        std::fs::write(
            manifest_dir.join(name),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("lora manifest writes");
    }
    std::fs::write(
        manifest_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [{
            "id": "resolved_flux2_dev",
            "name": "Resolved FLUX.2 Dev",
            "workflow": "text_to_image",
            "model": "flux2_dev",
            "loras": []
          }]
        }
        "#,
    )
    .expect("builtin presets write");
    std::fs::write(
        manifest_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets write");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Prompt enhancement preset" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": "z_image_turbo",
            "mode": "character_image",
            "prompt": "mist over hills",
            "referenceAssetId": "reference-1",
            "recipePresetId": "resolved_flux2_dev",
            "advanced": { "enhancePrompt": true }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error["detail"]
            .as_str()
            .is_some_and(|value| value.contains("Candle supports only")),
        "the post-preset FLUX.2-dev model must be checked against Candle's actual route: {error}"
    );
}

#[tokio::test]
async fn create_image_job_enforces_the_pose_output_ceiling() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let poses: Vec<Value> = (0..sceneworks_core::image_request::MAX_JOB_POSES)
        .map(|index| json!({ "id": format!("pose-{index}"), "keypoints": [] }))
        .collect();
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "advanced": { "poses": poses },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");

    let mut over = poses;
    over.push(json!({ "id": "one-too-many", "keypoints": [] }));
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "advanced": { "poses": over },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(
        error["detail"],
        "advanced.poses must contain at most 64 entries; each pose renders one image"
    );
}

#[tokio::test]
async fn candle_required_builtin_krea_keeps_builtin_scope_and_queues() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    single_model_manifest(&manifest_dir, "krea_2_turbo", "SceneWorks/krea-2-turbo-mlx");
    let mut settings = test_settings(&temp_dir);
    settings.candle_required = true;
    let app = create_app(settings).expect("app creates");

    let (status, created) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "model": "krea_2_turbo",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(
        created["payload"]["modelManifestEntry"]["catalogScope"],
        json!("builtin"),
        "the worker-facing merged manifest must preserve builtin scope"
    );
}

#[tokio::test]
async fn candle_required_rejects_unsupported_import_before_creating_a_job() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&manifest_dir);
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin manifest writes");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [{
            "id": "user_krea",
            "name": "User Krea",
            "type": "image",
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": "/probe/user-krea.safetensors" }
          }]
        }"#,
    )
    .expect("user manifest writes");
    let mut settings = test_settings(&temp_dir);
    settings.candle_required = true;
    let jobs_db_path = settings.jobs_db_path.clone();
    let app = create_app(settings).expect("app creates");

    for advanced in [
        json!({ "poses": [{ "id": "pose-1", "keypoints": [] }] }),
        json!({ "controlImage": "control-1" }),
        json!({ "controlMode": "pose" }),
        json!({
            "phases": [{ "steps": 4 }],
            "controlImage": "control-1"
        }),
        json!({
            "poses": [{ "id": "pose-1", "keypoints": [] }],
            "controlMode": "canny"
        }),
    ] {
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "model": "user_krea",
                "mode": "text_to_image",
                "prompt": "mist over hills",
                "count": 1,
                "advanced": advanced,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("candle_unsupported")));
    }

    let connection = rusqlite::Connection::open(jobs_db_path).expect("jobs db opens");
    let count: i64 = connection
        .query_row("select count(*) from jobs", [], |row| row.get(0))
        .expect("job count reads");
    assert_eq!(count, 0, "a preflight refusal must not create a queued job");
}

/// The pinned soft VAE donor the `wan_2_1_vae` decoder option depends on, mirroring the shipped
/// `qwen_image` co-requisite row. Written into a test manifest so a tempdir catalog can advertise a
/// genuinely SELECTABLE decoder option rather than only the "not installed" refusal.
const DECODER_DONOR_REPO: &str = "SceneWorks/krea-realtime-14b-mlx";
const DECODER_DONOR_REVISION: &str = "e68e9a3d98187fdf6936838ffcf6df5aa48d6626";
const DECODER_DONOR_FILE: &str = "q4/vae.safetensors";

/// A catalog holding the real `qwen_image` id — the id the checked-in engine decoder facts key on,
/// so `decoders.byBackend` is stamped onto the resolved entry — plus that row's pinned soft VAE
/// donor, with the donor's exact snapshot file seeded under `data_dir` so the descriptor-derived
/// MLX option resolves `available: true`.
///
/// Installing the donor is what makes the backend selection observable: with it absent, both lanes
/// refuse (candle for "no such option", MLX for "not installed") and the two are indistinguishable.
fn write_decoder_capable_catalog(temp_dir: &tempfile::TempDir) {
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&manifest_dir);
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        format!(
            r#"{{ "schemaVersion": 1, "models": [{{
                "id": "qwen_image", "name": "Qwen Image", "type": "image", "family": "test",
                "downloads": [
                  {{ "provider": "huggingface", "repo": "SceneWorks/qwen-image-mlx" }},
                  {{
                    "provider": "huggingface",
                    "repo": "{DECODER_DONOR_REPO}",
                    "revision": "{DECODER_DONOR_REVISION}",
                    "coRequisite": true,
                    "required": "soft",
                    "componentId": "vae",
                    "files": ["{DECODER_DONOR_FILE}"],
                    "estimatedSizeBytes": 507591212
                  }}
                ]
            }}] }}"#
        ),
    )
    .expect("builtin models writes");

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let repo_cache =
        huggingface_repo_cache_path(&data_dir, DECODER_DONOR_REPO).expect("donor cache path");
    let snapshot = repo_cache.join("snapshots").join(DECODER_DONOR_REVISION);
    let donor = snapshot.join(DECODER_DONOR_FILE);
    std::fs::create_dir_all(donor.parent().expect("donor parent")).expect("donor dir creates");
    std::fs::write(&donor, b"pinned donor").expect("donor writes");
}

fn decoder_image_job(decoder: Value) -> Value {
    json!({
        "projectId": "project-1",
        "model": "qwen_image",
        "mode": "text_to_image",
        "prompt": "mist over hills",
        "count": 1,
        "advanced": { "decoder": decoder },
    })
}

/// sc-18420: the alternate-decoder gate must consult the option list of the backend this API
/// instance actually routes to. It derived that with a bare `cfg!(target_os = "macos")`, so under
/// `SCENEWORKS_CANDLE_REQUIRED` on macOS — a real, supported mode — it validated `advanced.decoder`
/// against MLX's list while the job executed on Candle: an MLX-only decoder was admitted and then
/// failed on the worker, and a Candle-valid one would have been 400'd.
///
/// Both directions of the same catalog row and the same request: only `candle_required` differs.
#[tokio::test]
async fn candle_required_moves_the_decoder_gate_onto_the_candle_option_list() {
    let _env = isolate_hf_cache();

    // Candle lane: the shipped facts declare no candle decoder for this row, so the MLX-only
    // selection must be refused at enqueue rather than deferred to a worker that cannot run it.
    let candle_dir = tempfile::tempdir().expect("temp dir creates");
    write_decoder_capable_catalog(&candle_dir);
    let mut candle_settings = test_settings(&candle_dir);
    candle_settings.candle_required = true;
    let candle_app = create_app(candle_settings).expect("app creates");
    let (status, error) = request(
        candle_app,
        "POST",
        "/api/v1/image/jobs",
        decoder_image_job(json!("wan_2_1_vae")),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    let detail = error["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("is not compatible with qwen_image"),
        "candle_required must reject an MLX-only decoder as incompatible, not merely uninstalled. \
         NOTE: this exact branch is coupled to the SHIPPED decoder facts declaring ZERO candle \
         options for qwen_image — if `capabilities.candle.json` ever gains a decoderOption for this \
         row, the correct refusal becomes 'is not installed' (or the selection becomes valid) and \
         this assertion must be re-stated against the new facts, NOT deleted. The backend-selection \
         claim itself lives in the platform-free unit test \
         `the_gate_consults_only_the_executing_backends_option_list`: {error}"
    );

    // Native lane: the same request against the same row, with candle_required off. On macOS the
    // MLX option is installed and selectable, so it is accepted — which is what proves the
    // refusal above came from the backend swap and not from the option being unusable.
    let native_dir = tempfile::tempdir().expect("temp dir creates");
    write_decoder_capable_catalog(&native_dir);
    let native_app = create_app(test_settings(&native_dir)).expect("app creates");
    let (status, body) = request(
        native_app,
        "POST",
        "/api/v1/image/jobs",
        decoder_image_job(json!("wan_2_1_vae")),
    )
    .await;
    if cfg!(target_os = "macos") {
        assert_eq!(status, StatusCode::CREATED, "{body}");
        assert_eq!(
            body["payload"]["advanced"]["decoder"], "wan_2_1_vae",
            "the accepted selection must reach the worker verbatim"
        );
    } else {
        // Candle everywhere else regardless of the setting — same refusal as above.
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("is not compatible with qwen_image")));
    }
}

/// sc-18420: retry/duplicate re-run the text-encoder gate but skipped the decoder gate entirely,
/// and `payloadChanges` is a SHALLOW merge — `advanced` arrives replaced wholesale. A retry could
/// therefore enqueue exactly the decoder shapes the create path 400s.
#[tokio::test]
async fn retry_and_duplicate_gate_a_merged_decoder_selection_the_create_path_refuses() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_decoder_capable_catalog(&temp_dir);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // The original job selects no decoder, so it is admitted on either lane.
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        decoder_image_job(Value::Null),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for (label, advanced) in [
        // Mutually exclusive with usePid — refused on every lane, before any list lookup.
        (
            "exactly one decoder",
            json!({ "decoder": "wan_2_1_vae", "usePid": true }),
        ),
        // Not an option on any backend for this row.
        (
            "is not compatible with qwen_image",
            json!({ "decoder": "no_such_decoder" }),
        ),
        // The typed shape guard the create path applies to the same field.
        (
            "advanced.decoder must be a decoder id string",
            json!({ "decoder": 7 }),
        ),
    ] {
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "advanced": advanced } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(
                error["detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains(label)),
                "{operation} must fail with the create path's own refusal ({label}): {error}"
            );
        }
    }

    // A refused retry/duplicate must not have enqueued anything, and the boundary must still admit
    // the selection the create path admits — the gate is not a blanket refusal of `advanced`.
    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused retry/duplicate must not persist a job"
    );

    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({ "payloadChanges": { "advanced": { "decoder": "wan_2_1_vae" } } }),
    )
    .await;
    if cfg!(target_os = "macos") {
        assert_eq!(status, StatusCode::CREATED, "{replayed}");
        assert_eq!(replayed["payload"]["advanced"]["decoder"], "wan_2_1_vae");
    } else {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{replayed}");
    }
}

/// sc-18420, video half: `create_video_job` runs the same decoder gate, and the merged
/// retry/duplicate boundary skipped it there too. No video provider advertises an alternate
/// decoder, so every selection must fail closed at enqueue rather than reach a worker that has no
/// such option.
#[tokio::test]
async fn retry_and_duplicate_gate_a_merged_video_decoder_selection() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "a drone shot",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for (label, advanced) in [
        (
            "is not compatible with",
            json!({ "decoder": "wan_2_1_vae" }),
        ),
        (
            "exactly one decoder",
            json!({ "decoder": "wan_2_1_vae", "usePid": true }),
        ),
    ] {
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "advanced": advanced } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(
                error["detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains(label)),
                "{operation} must reproduce the video create path's refusal ({label}): {error}"
            );
        }
    }

    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused retry/duplicate must not persist a job"
    );

    // A replay that selects no decoder is untouched by the new gate.
    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({ "payloadChanges": { "prompt": "a slower drone shot" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
}

/// One image model and one video model, each declaring its own LoRA family, plus the four adapters
/// the retry/duplicate LoRA-gate tests need: a wrong-family one, a right-family-but-absent-on-disk
/// one, and a compatible installed one per lane. Mirrors the fixture
/// `generation_job_routes_reject_incompatible_loras` uses on the create path, so both boundaries are
/// asserted against the SAME catalog and therefore the same refusal strings.
fn write_lora_gate_catalog(temp_dir: &tempfile::TempDir) {
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [
            {
              "id": "gate_image",
              "name": "Gate Image",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image", "edit_image", "character_image"],
              "downloads": [], "paths": {}, "defaults": {}, "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": {}
            },
            {
              // A REAL routed id (sc-19570's claimability gate is id-keyed against the routing
              // catalog, so a synthetic id would 400 at create before the LoRA gate under test
              // ever runs). Only the id is real; every other field is this fixture's own.
              "id": "ltx_2_3",
              "name": "Gate Video",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video", "image_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
              "downloads": [], "paths": {}, "defaults": {}, "limits": {},
              "loraCompatibility": { "families": ["ltx-video"] },
              "ui": {}
            }
          ]
        }"#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{
          "schemaVersion": 1,
          "loras": [
            {
              "id": "qwen_style", "name": "Qwen Style", "family": "qwen-image",
              "triggerWords": [], "compatibility": { "families": ["qwen-image"] },
              "source": { "provider": "local", "path": "loras/qwen.safetensors" }
            },
            {
              "id": "deleted_style", "name": "Deleted Style", "family": "z-image",
              "triggerWords": [], "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/deleted.safetensors" }
            },
            {
              "id": "good_style", "name": "Good Style", "family": "z-image",
              "triggerWords": [], "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/good.safetensors" }
            },
            {
              "id": "motion_style", "name": "Motion Style", "family": "ltx-video",
              "triggerWords": [], "compatibility": { "families": ["ltx-video"] },
              "source": { "provider": "local", "path": "loras/motion.safetensors" }
            }
          ]
        }"#,
    )
    .expect("user loras writes");
    for file in ["builtin.recipe-presets.jsonc", "user.recipe-presets.jsonc"] {
        std::fs::write(
            config_dir.join(file),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("preset manifest writes");
    }
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    // `deleted.safetensors` is deliberately NOT written: a catalog-backed adapter whose file is gone
    // resolves to a non-installed state, which is the create path's "is not installed" refusal.
    for name in ["qwen.safetensors", "good.safetensors", "motion.safetensors"] {
        write_test_safetensors(&lora_dir.join(name));
    }
}

/// sc-18420: `validate_job_lora_compatibility_with` runs at BOTH create boundaries
/// (`create_image_job`, `create_video_job`) and at neither retry/duplicate one. `loras` is a
/// TOP-LEVEL key, so the shallow `payload_changes` merge replaces the whole array — a retry could
/// swap a validated adapter set for a wrong-family, uninstalled, or entirely unknown one and enqueue
/// it, deferring the failure to a worker that cannot load the file.
///
/// Every refusal string here is the create path's own, asserted verbatim against the same catalog.
#[tokio::test]
async fn retry_and_duplicate_gate_a_merged_lora_set_the_create_path_refuses() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_lora_gate_catalog(&temp_dir);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    // The LoRA catalog is project-scoped, so the gate needs a real project to resolve against.
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "LoRA Gate" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Baseline: created with a COMPATIBLE adapter, so the refusals below are about the merged set
    // and not about the route rejecting adapters at all.
    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "model": "gate_image",
            "count": 1,
            "loras": [{ "id": "good_style" }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for (expected, loras) in [
        (
            "LoRA qwen_style is not compatible with model gate_image",
            json!([{ "id": "qwen_style" }]),
        ),
        (
            "LoRA is not installed: deleted_style",
            json!([{ "id": "deleted_style" }]),
        ),
        (
            "LoRA not found: no_such_lora",
            json!([{ "id": "no_such_lora" }]),
        ),
        // A mixed set must be refused for its bad member, not silently pruned to the good one.
        (
            "LoRA qwen_style is not compatible with model gate_image",
            json!([{ "id": "good_style" }, { "id": "qwen_style" }]),
        ),
    ] {
        // The create path's own verdict on the identical set, so the two boundaries are compared
        // rather than the retry refusal being asserted in isolation.
        let (create_status, create_error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_image",
                "prompt": "mist over hills",
                "model": "gate_image",
                "count": 1,
                "loras": loras,
            }),
        )
        .await;
        assert_eq!(create_status, StatusCode::BAD_REQUEST, "{create_error}");
        assert_eq!(create_error["detail"], expected);

        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "loras": loras } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert_eq!(
                error["detail"], expected,
                "{operation} must reproduce the create path's LoRA refusal verbatim"
            );
        }
    }

    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused retry/duplicate must not persist a job"
    );

    // The gate must still admit — and NORMALIZE — a set the create path accepts, so the canonical
    // object this boundary returns carries the hydrated catalog spec rather than the bare id.
    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({ "payloadChanges": { "loras": [{ "id": "good_style" }] } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
    assert_eq!(replayed["payload"]["loras"][0]["id"], "good_style");
    assert!(
        replayed["payload"]["loras"][0].get("name").is_some(),
        "the admitted set must be the normalized catalog spec, not the caller's bare id: {replayed}"
    );
}

/// sc-18420: the retry/duplicate LoRA gate must honour the ORIGINAL job's inline-LoRA provenance.
///
/// `characters.rs`'s test-job route creates `image_generate` jobs with `allow_inline_loras = true`,
/// and a character's adapters are inline links (`character_lora_<hex>`, `category: "character"`,
/// path-bearing) that `character_store::attach_lora` registers in NO catalog. Mirroring the gate
/// with a hard-coded `false` refused that persisted set with "LoRA not found" — breaking even a
/// retry with EMPTY `payloadChanges`.
///
/// Both directions, because the obvious fix opens a hole: `characterId` and `mode` are
/// caller-settable (`ImageJobRequest` exposes both), so permission must come from the persisted
/// LINK SHAPE, which only the character route can have put there. An ordinary image job wearing
/// those markers must NOT be able to smuggle an inline path-bearing adapter through
/// `payloadChanges`.
#[tokio::test]
async fn retry_honours_character_inline_lora_provenance_without_letting_others_borrow_it() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_lora_gate_catalog(&temp_dir);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Character Retry" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let (_, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    let character_id = character["id"].as_str().expect("character id").to_owned();
    let source = temp_dir.path().join("data/loras/character.safetensors");
    write_test_safetensors(&source);
    let (status, attached) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Character Style",
            "sourcePath": source.display().to_string(),
            // Compatible with `gate_image` so the test-job itself is admitted; the family gate is
            // pinned elsewhere and is not what this test is about.
            "compatibility": { "families": ["z-image"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{attached}");

    let (status, character_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "model": "gate_image" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{character_job}");
    let character_job_id = character_job["id"].as_str().expect("job id").to_owned();
    let inline_lora = character_job["payload"]["loras"][0].clone();
    assert_eq!(
        inline_lora["category"], "character",
        "the fixture must really be an inline character link, or this test proves nothing: \
         {character_job}"
    );
    let inline_lora_id = inline_lora["id"].as_str().expect("link id").to_owned();
    assert!(
        inline_lora_id.starts_with("character_lora_"),
        "got: {inline_lora_id}"
    );

    // DIRECTION 1: the character job's own inline set survives retry AND duplicate, including the
    // no-op replay that the hard-coded `false` broke.
    for operation in ["retry", "duplicate"] {
        let (status, replayed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{character_job_id}/{operation}"),
            json!({ "payloadChanges": {} }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{operation} of a character test job must not refuse its own inline links: {replayed}"
        );
        assert_eq!(
            replayed["payload"]["loras"][0]["id"], inline_lora_id,
            "{operation} must preserve the character link"
        );
    }
    // And a real change alongside them still works.
    let (status, replayed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{character_job_id}/retry"),
        json!({ "payloadChanges": { "prompt": "portrait, side light" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
    assert_eq!(replayed["payload"]["prompt"], "portrait, side light");

    // The permit is PER-ADAPTER, not a blanket flag on the merged array: a genuine character job may
    // replay its own links, but may not use that standing to introduce a foreign inline adapter.
    let foreign = temp_dir.path().join("data/loras/foreign.safetensors");
    write_test_safetensors(&foreign);
    for (label, loras) in [
        // Swap the persisted set for an entirely foreign inline adapter.
        (
            "LoRA not found: character_lora_foreign",
            json!([{
                "id": "character_lora_foreign",
                "name": "Foreign",
                "category": "character",
                "sourcePath": foreign.display().to_string(),
                "compatibility": { "families": ["z-image"] }
            }]),
        ),
        // Keep the persisted link AND smuggle a foreign one alongside it: the permit must not
        // launder its companion.
        (
            "LoRA not found: character_lora_foreign",
            json!([
                inline_lora.clone(),
                {
                    "id": "character_lora_foreign",
                    "name": "Foreign",
                    "category": "character",
                    "sourcePath": foreign.display().to_string(),
                    "compatibility": { "families": ["z-image"] }
                }
            ]),
        ),
        // Replay the persisted link's OWN id with a REDIRECTED path — the case an id-only match
        // would wave through, carrying an arbitrary file into the enqueued payload.
        (
            &format!("LoRA not found: {inline_lora_id}"),
            json!([{
                "id": inline_lora_id,
                "name": "Character Style",
                "category": "character",
                "sourcePath": foreign.display().to_string(),
                "compatibility": { "families": ["z-image"] }
            }]),
        ),
    ] {
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{character_job_id}/{operation}"),
                json!({ "payloadChanges": { "loras": loras } }),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{operation} must not extend the character permit beyond the PERSISTED links: \
                 {error}"
            );
            assert_eq!(error["detail"], label);
        }
    }

    // A CATALOG adapter added alongside the persisted inline set is admitted — the narrowing must
    // not turn a character job into one that can never gain an adapter — and both survive, in order.
    let (status, widened) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{character_job_id}/duplicate"),
        json!({
            "payloadChanges": {
                "loras": [inline_lora.clone(), { "id": "good_style" }]
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{widened}");
    assert_eq!(widened["payload"]["loras"][0]["id"], inline_lora_id);
    assert_eq!(widened["payload"]["loras"][1]["id"], "good_style");
    assert!(
        widened["payload"]["loras"][1].get("name").is_some(),
        "the added catalog adapter must be hydrated from the catalog, not passed through inline: \
         {widened}"
    );

    // DIRECTION 2: an ORDINARY image job that wears both caller-settable markers must not borrow
    // that permission. Create is happy to make it — the LoRA gate no-ops on an empty set — which is
    // exactly why provenance cannot be read from the markers.
    let smuggler = temp_dir.path().join("data/loras/smuggled.safetensors");
    write_test_safetensors(&smuggler);
    let (status, ordinary) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "character_image",
            "characterId": "character_not_really_mine",
            "prompt": "mist over hills",
            "model": "gate_image",
            "count": 1,
            "referenceAssetIds": [],
            "loras": [],
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the marker-wearing decoy must be creatable for the smuggle attempt to be meaningful: \
         {ordinary}"
    );
    let ordinary_id = ordinary["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for operation in ["retry", "duplicate"] {
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{ordinary_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "loras": [{
                        "id": "character_lora_forged",
                        "name": "Forged",
                        "category": "character",
                        "sourcePath": smuggler.display().to_string(),
                        "compatibility": { "families": ["z-image"] }
                    }]
                }
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{operation} must not accept an inline adapter for a job whose PERSISTED set had none: \
             {error}"
        );
        assert_eq!(error["detail"], "LoRA not found: character_lora_forged");
    }

    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused smuggle must not persist a job"
    );

    // The decoy still validates a CATALOG adapter normally, so the refusal above is about inline
    // permission and not about the job being unable to take adapters at all.
    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{ordinary_id}/duplicate"),
        json!({ "payloadChanges": { "loras": [{ "id": "good_style" }] } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
    assert_eq!(replayed["payload"]["loras"][0]["id"], "good_style");
}

/// The video half of the same LoRA bypass — `create_video_job` runs the identical gate.
#[tokio::test]
async fn retry_and_duplicate_gate_a_merged_video_lora_set_the_create_path_refuses() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_lora_gate_catalog(&temp_dir);
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    // The LoRA catalog is project-scoped, so the gate needs a real project to resolve against.
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "LoRA Gate" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a drone shot",
            "model": "ltx_2_3",
            "loras": [{ "id": "motion_style" }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{original}");
    let job_id = original["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for (expected, loras) in [
        (
            "LoRA qwen_style is not compatible with model ltx_2_3",
            json!([{ "id": "qwen_style" }]),
        ),
        // A z-image adapter is installed and well-formed, just wrong for THIS lane's model.
        (
            "LoRA good_style is not compatible with model ltx_2_3",
            json!([{ "id": "good_style" }]),
        ),
        (
            "LoRA not found: no_such_lora",
            json!([{ "id": "no_such_lora" }]),
        ),
    ] {
        let (create_status, create_error) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_video",
                "prompt": "a drone shot",
                "model": "ltx_2_3",
                "loras": loras,
            }),
        )
        .await;
        assert_eq!(create_status, StatusCode::BAD_REQUEST, "{create_error}");
        assert_eq!(create_error["detail"], expected);

        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "loras": loras } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert_eq!(
                error["detail"], expected,
                "{operation} must reproduce the video create path's LoRA refusal verbatim"
            );
        }
    }

    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused retry/duplicate must not persist a job"
    );

    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({ "payloadChanges": { "loras": [{ "id": "motion_style" }] } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
    assert_eq!(replayed["payload"]["loras"][0]["id"], "motion_style");
}

/// sc-18420: the same retry/duplicate bypass for the imported-submission gate. The create path
/// refuses an imported request shape the resolved provider registration cannot execute; the merged
/// boundary never ran that check, so a retry could swap `advanced` for one carrying a shape the
/// backend has no route for.
#[tokio::test]
async fn retry_and_duplicate_gate_a_merged_imported_shape_the_create_path_refuses() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&manifest_dir);
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin manifest writes");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [{
            "id": "user_krea",
            "name": "User Krea",
            "type": "image",
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": "/probe/user-krea.safetensors" }
          }]
        }"#,
    )
    .expect("user manifest writes");
    let mut settings = test_settings(&temp_dir);
    // Candle declares krea_2/transformer_file for `generate` but NOT for `pose`, so the plain
    // request is admitted and the pose replay is the shape with no route.
    settings.candle_required = true;
    let app = create_app(settings).expect("app creates");

    let (status, original) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "model": "user_krea",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the plain imported generate shape has a candle route: {original}"
    );
    let job_id = original["id"].as_str().expect("job id").to_owned();

    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let before = before.as_array().expect("jobs array").len();

    for advanced in [
        json!({ "poses": [{ "id": "pose-1", "keypoints": [] }] }),
        json!({ "controlImage": "control-1" }),
        json!({ "controlMode": "pose" }),
    ] {
        for operation in ["retry", "duplicate"] {
            let (status, error) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": { "advanced": advanced } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(
                error["detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("candle_unsupported")),
                "{operation} must reproduce the create path's imported refusal: {error}"
            );
        }
    }

    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        after.as_array().expect("jobs array").len(),
        before,
        "a refused retry/duplicate must not persist a job"
    );

    // The gate must still pass a merged shape the create path accepts — a prompt-only replay keeps
    // the admitted generate operation.
    let (status, replayed) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/duplicate"),
        json!({ "payloadChanges": { "prompt": "fog over hills" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replayed}");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn native_imported_control_requires_pose_but_preserves_krea_pose_user_map() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&manifest_dir);
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin manifest writes");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [{
            "id": "user_krea",
            "name": "User Krea",
            "type": "image",
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": "/probe/user-krea.safetensors" }
          }]
        }"#,
    )
    .expect("user manifest writes");
    let settings = test_settings(&temp_dir);
    let jobs_db_path = settings.jobs_db_path.clone();
    let app = create_app(settings).expect("app creates");

    for advanced in [
        json!({ "controlImage": "control-1" }),
        json!({ "controlMode": "pose" }),
        json!({
            "phases": [{ "steps": 4 }],
            "controlImage": "control-1"
        }),
        json!({
            "poses": [{ "id": "pose-1", "keypoints": [] }],
            "controlMode": "canny"
        }),
    ] {
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "model": "user_krea",
                "mode": "text_to_image",
                "prompt": "mist over hills",
                "count": 1,
                "advanced": advanced,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("imported_control_unsupported")));
    }

    let (status, created) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "model": "user_krea",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "advanced": {
                "poses": [{ "id": "pose-1", "keypoints": [] }],
                "controlImage": "control-1",
                "controlMode": "pose"
            },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["payload"]["advanced"]["controlImage"], "control-1");

    let connection = rusqlite::Connection::open(jobs_db_path).expect("jobs db opens");
    let count: i64 = connection
        .query_row("select count(*) from jobs", [], |row| row.get(0))
        .expect("job count reads");
    assert_eq!(count, 1, "only the supported imported Pose request queues");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn native_imported_mage_queues_only_the_exact_registered_generate_shape() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let manifest_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&manifest_dir).expect("manifest dir creates");
    write_empty_sibling_manifests(&manifest_dir);
    std::fs::write(
        manifest_dir.join("builtin.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("builtin manifest writes");
    std::fs::write(
        manifest_dir.join("user.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [{
            "id": "finetune_mage",
            "name": "Fine-tuned Mage",
            "type": "image",
            "family": "mage-flow",
            "importSourceShape": "transformer_directory",
            "paths": { "model": "/probe/finetune-mage" }
          }]
        }"#,
    )
    .expect("user manifest writes");
    let settings = test_settings(&temp_dir);
    let jobs_db_path = settings.jobs_db_path.clone();
    let app = create_app(settings).expect("app creates");

    for (label, extra) in [
        (
            "edit",
            json!({ "mode": "edit_image", "sourceAssetId": "source-1" }),
        ),
        ("reference", json!({ "referenceAssetId": "reference-1" })),
        (
            "multi-phase",
            json!({ "advanced": { "phases": [{ "steps": 4 }] } }),
        ),
        (
            "unsupported quant tier",
            json!({ "advanced": { "quantTier": "nvfp4" } }),
        ),
    ] {
        let mut payload = json!({
            "projectId": "project-1",
            "model": "finetune_mage",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
        });
        payload
            .as_object_mut()
            .expect("request object")
            .extend(extra.as_object().expect("extra object").clone());
        let (status, error) = request(app.clone(), "POST", "/api/v1/image/jobs", payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {error}");
        assert!(
            error["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("imported_unsupported")),
            "{label} must fail at exact imported-provider admission: {error}"
        );
    }

    let (status, created) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "model": "finetune_mage",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");

    let connection = rusqlite::Connection::open(jobs_db_path).expect("jobs db opens");
    let count: i64 = connection
        .query_row("select count(*) from jobs", [], |row| row.get(0))
        .expect("job count reads");
    assert_eq!(count, 1, "only the exact registered generate shape queues");
}

/// Legacy over-limit payloads stay inspectable, but replaying them would create new unbounded work.
/// Retry and duplicate therefore reject until the caller reduces `advanced.poses` to the current
/// product ceiling. This makes the compatibility policy executable instead of an incidental side
/// effect of which creation route happened to run.
#[tokio::test]
async fn legacy_over_limit_pose_job_is_readable_but_cannot_be_replayed() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let jobs_db_path = settings.jobs_db_path.clone();
    let app = create_app(settings).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "legacy pose set",
            "count": 1,
            "advanced": { "poses": [{ "id": "pose-0", "keypoints": [] }] },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let job_id = created["id"].as_str().expect("job id");
    let mut legacy_payload = created["payload"]
        .as_object()
        .cloned()
        .expect("stored payload object");
    let legacy_poses: Vec<Value> = (0..=sceneworks_core::image_request::MAX_JOB_POSES)
        .map(|index| json!({ "id": format!("legacy-pose-{index}"), "keypoints": [] }))
        .collect();
    legacy_payload.insert("advanced".to_owned(), json!({ "poses": legacy_poses }));

    let connection = rusqlite::Connection::open(jobs_db_path).expect("jobs db opens");
    let updated = connection
        .execute(
            "update jobs set payload_json = ?1 where id = ?2",
            rusqlite::params![Value::Object(legacy_payload).to_string(), job_id],
        )
        .expect("legacy payload writes");
    assert_eq!(updated, 1);
    drop(connection);

    let (status, stored) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    assert_eq!(
        stored["payload"]["advanced"]["poses"]
            .as_array()
            .map(Vec::len),
        Some(65)
    );

    for operation in ["retry", "duplicate"] {
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/{operation}"),
            json!({ "payloadChanges": {} }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert_eq!(
            error["detail"],
            "advanced.poses must contain at most 64 entries; each pose renders one image"
        );
    }

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().map(Vec::len), Some(1));
}

#[tokio::test]
async fn create_video_job_rejects_over_length_negative_prompt() {
    // sc-8884 (F-082): the negative-prompt cap is shared by the video validator too.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, error) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "a drone shot",
            "negativePrompt": "n".repeat(4001),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("negativePrompt")));
}

/// Seed a minimal audio + non-audio manifest into a test config dir so
/// `resolve_model_manifest_entry` resolves a real `type: audio` entry for the audio route.
fn write_audio_manifest(config_dir: &std::path::Path) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "kokoro_82m",
              "name": "Kokoro 82M",
              "family": "kokoro",
              "type": "audio",
              "audio": {
                "voices": [{ "id": "af_heart", "language": "en-US" }, { "id": "bm_george", "language": "en-GB" }],
                "languages": ["en-US", "en-GB"],
                "sampleRates": [24000],
                "maxDurationSecs": 30
              },
              "downloads": [
                { "provider": "huggingface", "repo": "hexgrad/Kokoro-82M", "files": ["config.json", "kokoro-v1_0.pth", "voices/*"] }
              ],
              "paths": { "model": "${HF_CACHE}/hexgrad/Kokoro-82M" },
              "ui": { "label": "Kokoro 82M" }
            },
            {
              "id": "moss_sfx_v2",
              "name": "MOSS SoundEffect v2 (SFX)",
              "family": "moss_soundeffect",
              "type": "audio",
              "audio": {
                "languages": ["en", "zh"],
                "sampleRates": [48000],
                "maxDurationSecs": 30
              },
              "downloads": [
                { "provider": "huggingface", "repo": "OpenMOSS-Team/MOSS-SoundEffect-v2.0", "files": ["model_index.json"] }
              ],
              "paths": { "model": "${HF_CACHE}/OpenMOSS-Team/MOSS-SoundEffect-v2.0" },
              "ui": { "label": "MOSS SoundEffect v2" }
            },
            {
              "id": "acestep_v15_turbo",
              "name": "ACE-Step v1.5 XL Turbo (Music)",
              "family": "acestep",
              "type": "audio",
              "audio": {
                "languages": ["en", "zh"],
                "sampleRates": [48000],
                "maxDurationSecs": 600,
                "editModes": ["inpaint", "repaint", "extend"],
                "conditioning": ["AudioEdit"]
              },
              "downloads": [
                { "provider": "huggingface", "repo": "ACE-Step/acestep-v15-xl-turbo-diffusers", "files": ["model_index.json"] }
              ],
              "paths": { "model": "${HF_CACHE}/ACE-Step/acestep-v15-xl-turbo-diffusers" },
              "ui": { "label": "ACE-Step v1.5 XL Turbo" }
            },
            {
              "id": "openvoice_v2",
              "name": "OpenVoice V2 (Voice Conversion)",
              "family": "openvoice",
              "type": "audio",
              "audio": {
                "sampleRates": [22050],
                "conditioning": ["ReferenceAudio"]
              },
              "downloads": [
                { "provider": "huggingface", "repo": "myshell-ai/OpenVoiceV2", "files": ["converter/config.json", "converter/checkpoint.pth"] }
              ],
              "paths": { "model": "${HF_CACHE}/myshell-ai/OpenVoiceV2" },
              "ui": { "label": "OpenVoice V2" }
            },
            {
              "id": "moss_ttsd_v05",
              "name": "MOSS-TTSD v0.5 (Multi-Speaker Dialogue)",
              "family": "moss_ttsd",
              "type": "audio",
              "audio": {
                "languages": ["zh", "en"],
                "sampleRates": [24000],
                "maxDurationSecs": 300,
                "supportsMultiSpeaker": true,
                "maxSpeakers": 2,
                "supportsStreaming": false
              },
              "downloads": [
                { "provider": "huggingface", "repo": "OpenMOSS-Team/MOSS-TTSD-v0.5", "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] },
                { "provider": "huggingface", "repo": "OpenMOSS-Team/XY_Tokenizer_TTSD_V0", "revision": "c83433728e698ed0698e88cb5096bc221fb8f8c5", "coRequisite": true, "files": ["xy_tokenizer.ckpt"] }
              ],
              "paths": { "model": "${HF_CACHE}/OpenMOSS-Team/MOSS-TTSD-v0.5" },
              "ui": { "label": "MOSS-TTSD v0.5 (Multi-Speaker)" }
            },
            {
              "id": "not-audio-img",
              "name": "Not Audio",
              "family": "z_image",
              "type": "image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/not-audio", "files": ["*.safetensors"] }
              ],
              "paths": {},
              "ui": { "label": "Not Audio" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
}

/// The audio job path (sc-13404): a well-formed `POST /api/v1/audio/jobs` maps the request into an
/// `audio_generate` job whose payload carries the audio knobs (voice / language / targetDurationSecs
/// / seed) verbatim and the resolved `type: audio` manifest entry — the audio twin of how the video
/// route injects `modelManifestEntry`.
#[tokio::test]
async fn create_audio_job_maps_request_to_audio_generate_payload() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "Hello from SceneWorks audio.",
            "voice": "bm_george",
            "language": "en-GB",
            "targetDurationSecs": 4.0,
            "seed": 7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "kokoro_82m");
    assert_eq!(payload["prompt"], "Hello from SceneWorks audio.");
    assert_eq!(payload["voice"], "bm_george");
    assert_eq!(payload["language"], "en-GB");
    assert_eq!(payload["targetDurationSecs"], 4.0);
    assert_eq!(payload["seed"], 7);
    // The resolved manifest entry must travel with the job (the worker resolves weights from it),
    // and it must be THIS model's `type: audio` entry — not `{}` (which would slip a non-audio job
    // through) — carrying the HF download repo the worker resolves the Kokoro snapshot from.
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "kokoro_82m");
    assert_eq!(entry["type"], "audio");
    assert_eq!(entry["downloads"][0]["repo"], "hexgrad/Kokoro-82M");
    // requestedGpu is stripped from the payload (it rides the job envelope), mirroring the video route.
    assert!(payload.get("requestedGpu").is_none());
}

/// The multi-speaker path (MOSS-TTSD, sc-13676): a well-formed dialogue `POST /api/v1/audio/jobs`
/// carries the segmented `script` (each turn's text + speaker) through to the worker payload verbatim,
/// with an EMPTY prompt accepted (the script carries the text), and injects the resolved MOSS-TTSD
/// `type: audio` manifest entry so the worker resolves both the AR + codec snapshots.
#[tokio::test]
async fn create_audio_job_maps_the_multi_speaker_script() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dialogue Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_ttsd_v05",
            // Empty prompt is accepted because a non-empty multi-speaker script carries the text.
            "prompt": "",
            "script": [
                { "text": "Hello, how are you today?", "speaker": "S1" },
                { "text": "I'm doing great, thanks for asking!", "speaker": "S2" },
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "moss_ttsd_v05");
    // The segmented dialogue travels verbatim to the worker as `script` (camelCase round-trip).
    let script = payload["script"]
        .as_array()
        .expect("script array in payload");
    assert_eq!(script.len(), 2);
    assert_eq!(script[0]["text"], "Hello, how are you today?");
    assert_eq!(script[0]["speaker"], "S1");
    assert_eq!(script[1]["speaker"], "S2");
    // The MOSS-TTSD manifest entry travels so the worker resolves the AR + codec co-requisite.
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "moss_ttsd_v05");
    assert_eq!(entry["type"], "audio");
    assert_eq!(entry["audio"]["supportsMultiSpeaker"], true);
    assert_eq!(entry["audio"]["maxSpeakers"], 2);
}

/// A multi-speaker script naming >1 turn but with a whitespace-only prompt AND no script is still
/// rejected — the relaxed prompt guard only accepts an empty prompt when a real script is present.
#[tokio::test]
async fn create_audio_job_rejects_empty_prompt_and_empty_script() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Dialogue Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // Empty prompt + empty script array → rejected (a script with no segments is not content).
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({ "projectId": project_id, "model": "moss_ttsd_v05", "prompt": "", "script": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // A script segment with empty text → rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_ttsd_v05",
            "prompt": "",
            "script": [{ "text": "   ", "speaker": "S1" }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

/// The Sound FX path (MOSS-SoundEffect, sc-13409): a well-formed SFX `POST /api/v1/audio/jobs`
/// carries the diffusion sampling knobs (`guidance` = CFG scale, `steps`) through to the worker
/// payload verbatim alongside the shared audio knobs, and injects the resolved MOSS `type: audio`
/// manifest entry — so the worker can map guidance/steps onto the top-level GenerationRequest. No
/// voice is sent (MOSS advertises no voice surface).
#[tokio::test]
async fn create_audio_job_maps_sfx_sampling_knobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SFX Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_sfx_v2",
            "prompt": "a heavy wooden door creaking open",
            "language": "en",
            "targetDurationSecs": 3.0,
            "guidance": 6.5,
            "steps": 60,
            "seed": 11,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "moss_sfx_v2");
    assert_eq!(payload["prompt"], "a heavy wooden door creaking open");
    assert_eq!(payload["language"], "en");
    assert_eq!(payload["targetDurationSecs"], 3.0);
    // The SFX sampling knobs ride through to the worker payload (which maps them onto the top-level
    // GenerationRequest's guidance/steps — not AudioParams).
    assert_eq!(payload["guidance"], 6.5);
    assert_eq!(payload["steps"], 60);
    assert_eq!(payload["seed"], 11);
    // No voice on an SFX request; the MOSS manifest entry travels for weight resolution.
    assert!(payload.get("voice").is_none());
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "moss_sfx_v2");
    assert_eq!(entry["type"], "audio");
    assert_eq!(
        entry["downloads"][0]["repo"],
        "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
    );
}

/// The Voice Clone path (OpenVoice V2 conversion chain, sc-13411 C4): a `POST /api/v1/audio/jobs` that
/// names a converter model + `referenceAudioAssetId` carries the reference + match strength through to
/// the worker payload AND injects a SECOND manifest entry — the base TTS model (`baseModelManifestEntry`)
/// — so the worker resolves both snapshots. The converter's own entry rides as `modelManifestEntry`.
#[tokio::test]
async fn create_audio_job_injects_base_model_entry_for_voice_clone() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Voice Clone Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "openvoice_v2",
            "prompt": "Clone this into my reference voice.",
            "referenceAudioAssetId": "ref-voice-1",
            "matchStrength": 0.5,
            "seed": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "openvoice_v2");
    assert_eq!(payload["referenceAudioAssetId"], "ref-voice-1");
    assert_eq!(payload["matchStrength"], 0.5);
    // The selected converter's entry rides as modelManifestEntry (weights resolution).
    assert_eq!(payload["modelManifestEntry"]["id"], "openvoice_v2");
    assert_eq!(
        payload["modelManifestEntry"]["downloads"][0]["repo"],
        "myshell-ai/OpenVoiceV2"
    );
    // ...and the base TTS (Kokoro) entry is injected so the worker resolves the base generator too.
    let base = &payload["baseModelManifestEntry"];
    assert_eq!(base["id"], "kokoro_82m");
    assert_eq!(base["type"], "audio");
    assert_eq!(base["downloads"][0]["repo"], "hexgrad/Kokoro-82M");
}

/// A non-voice-clone audio job (no reference) carries NO `baseModelManifestEntry` — the base-model
/// resolution is scoped to the voice-clone path so ordinary Speech/SFX/Music jobs are untouched.
#[tokio::test]
async fn create_audio_job_omits_base_model_entry_without_a_reference() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Plain Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "Just plain speech.",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("baseModelManifestEntry").is_none());
}

/// The Voice Clone match-strength floor (sc-13411 C4): the API blanket-bounds `matchStrength` to
/// 0..=1 (the converter re-checks it), so an out-of-range value is a 400 rather than reaching the worker.
#[tokio::test]
async fn create_audio_job_rejects_out_of_range_match_strength() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Bad Strength Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, _body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "openvoice_v2",
            "prompt": "over-driven strength",
            "referenceAudioAssetId": "ref-voice-1",
            "matchStrength": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// The audio validator rejects out-of-blanket sampling knobs up front (sc-13409) — the API blanket
/// (guidance 0..=100, steps 1..=10000); the per-model range (MOSS: guidance 1..=20, steps ≤1000) is
/// the generator's own `validate` at generate time.
#[tokio::test]
async fn create_audio_job_rejects_nonsense_sampling_knobs() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SFX Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "moss_sfx_v2",
            "prompt": "thunder",
            "steps": 0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("steps")));
}

/// The Music path (ACE-Step, sc-13410): a well-formed music `POST /api/v1/audio/jobs` carries the
/// describe-the-music sub-block (bpm / musicalKey / lyrics) + steps AND the extend/edit source band
/// (sourceAudioAssetId / editMode / editRegion* / editStrength) through to the worker payload verbatim,
/// alongside the resolved ACE-Step `type: audio` manifest entry — so the worker maps them onto
/// AudioParams + a Conditioning::AudioEdit.
#[tokio::test]
async fn create_audio_job_maps_music_and_edit_fields() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "language": "en",
            "targetDurationSecs": 8.0,
            "steps": 8,
            "bpm": 92.0,
            "musicalKey": "C minor",
            "lyrics": "[verse] la la la",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "extend",
            "editRegionEndSecs": 20.0,
            "editStrength": 0.5,
            "seed": 5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "audio_generate");
    let payload = &job["payload"];
    assert_eq!(payload["model"], "acestep_v15_turbo");
    // Describe-the-music sub-block (rides AudioParams music fields).
    assert_eq!(payload["bpm"], 92.0);
    assert_eq!(payload["musicalKey"], "C minor");
    assert_eq!(payload["lyrics"], "[verse] la la la");
    assert_eq!(payload["steps"], 8);
    // Extend/edit source band (rides a Conditioning::AudioEdit).
    assert_eq!(payload["sourceAudioAssetId"], "audio-src-1");
    assert_eq!(payload["editMode"], "extend");
    assert_eq!(payload["editRegionEndSecs"], 20.0);
    assert_eq!(payload["editStrength"], 0.5);
    let entry = &payload["modelManifestEntry"];
    assert_eq!(entry["id"], "acestep_v15_turbo");
    assert_eq!(entry["type"], "audio");
    assert_eq!(
        entry["downloads"][0]["repo"],
        "ACE-Step/acestep-v15-xl-turbo-diffusers"
    );
}

/// A half-specified edit (a source track without an edit mode, or vice versa) is malformed — one names
/// WHAT to edit, the other HOW — so the validator rejects it up front rather than silently dropping the
/// edit (sc-13410).
#[tokio::test]
async fn create_audio_job_rejects_half_specified_edit() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            // A source with no editMode → rejected as a malformed pair.
            "sourceAudioAssetId": "audio-src-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("editMode")));
}

/// The Music validator sanity-bounds the describe-the-music + edit fields up front (sc-13410): a
/// non-positive BPM, an unknown edit mode, and a mis-ordered region are all 400s before the job is
/// enqueued (the per-model gate — mode ∈ advertised editModes, region inside the clip — is the
/// generator's own `validate`).
#[tokio::test]
async fn create_audio_job_rejects_nonsense_music_fields() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A non-positive BPM is rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "bpm": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"].as_str().is_some_and(|d| d.contains("bpm")));

    // An unknown edit mode token is rejected up front.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "morph",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editMode")));

    // A region whose end is at/below its start is rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "inpaint",
            "editRegionStartSecs": 5.0,
            "editRegionEndSecs": 2.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editRegionEndSecs")));
}

/// The editMode token is case-insensitive at the API, matching the worker's own case handling
/// (`edit_mode.map(|m| m.to_lowercase())` at deserialize, then `parse_audio_edit_mode`). A mixed-case
/// KNOWN token (`"Extend"`) is accepted and forwarded verbatim (the worker lowercases it), while a
/// mixed-case UNKNOWN token (`"Morph"`) is still rejected — lowercasing widens casing, not the mode
/// set (sc-13410, worker-parity fix).
#[tokio::test]
async fn create_audio_job_accepts_case_insensitive_edit_mode() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Music Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A mixed-case KNOWN token is accepted (parity with the worker) and forwarded verbatim.
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "gentle lofi piano loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "Extend",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["editMode"], "Extend");

    // A mixed-case UNKNOWN token is still rejected — casing widened, the mode set did not.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "acestep_v15_turbo",
            "prompt": "loop",
            "sourceAudioAssetId": "audio-src-1",
            "editMode": "Morph",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|d| d.contains("editMode")));
}

/// The audio route is a door for `type: audio` models only: an image/video model posted here is
/// rejected up front rather than failing deep in the worker's audio lane (sc-13404).
#[tokio::test]
async fn create_audio_job_rejects_non_audio_model() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A path-safe but non-audio model id resolves to a `type: image` entry → rejected.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "not-audio-img",
            "prompt": "this is not audio",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("audio")));

    // An unknown model resolves to `{}` (no type) and is rejected for the same reason.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "definitely-not-real",
            "prompt": "hello",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// The audio validator bounds the script prompt exactly as the image/video validators do (sc-13404).
#[tokio::test]
async fn create_audio_job_rejects_empty_prompt() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_audio_manifest(&temp_dir.path().join("config/manifests"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Audio Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/audio/jobs",
        json!({
            "projectId": project_id,
            "model": "kokoro_82m",
            "prompt": "   ",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("prompt")));
}

// The queue-lifecycle tests below drive `POST /api/v1/jobs` — claim, cancel, retry,
// progress, clear, stale sweep. They need *a* GPU-routed claimable job and do not care
// which; they use `image_detail` (a real caller of this route — the web batch ops post it)
// rather than `image_generate`, which sc-12305 moved behind its typed route so the model's
// manifest entry is always resolved. The worker `capabilities` match by job type
// (`required_capability`), so the two move together.

#[tokio::test]
async fn worker_can_register_claim_and_complete_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"]["id"], created["id"]);
    assert_eq!(claimed["job"]["status"], "preparing");

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Done",
            "workerId": "worker-1",
            "result": { "assetIds": ["asset-1"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["result"], json!({ "assetIds": ["asset-1"] }));

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["completed"], 1);
    assert_eq!(queue["workers"][0]["status"], "idle");
}

/// sc-13842: the audio streaming pump may have a Running progress POST already in flight when the
/// cancel watcher commits terminal Canceled. Hold that nonterminal report immediately before the
/// authoritative store transition, let Canceled win, then prove the delayed report receives 409
/// and cannot resurrect the terminal row.
#[tokio::test]
async fn in_flight_running_progress_cannot_resurrect_a_canceled_job() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) =
        create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "payload": { "prompt": "streaming race" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = created["id"].as_str().expect("job id").to_owned();
    claim_job_as_worker(&app, &job_id, "stream-worker", &["image_detail"]).await;

    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
    *state.progress_before_accept_once.lock() = Some(barrier.clone());
    let progress_app = app.clone();
    let progress_job_id = job_id.clone();
    let progress = tokio::spawn(async move {
        request(
            progress_app,
            "POST",
            &format!("/api/v1/jobs/{progress_job_id}/progress"),
            json!({
                "status": "running",
                "stage": "generating",
                "progress": 0.5,
                "message": "Streaming audio…",
                "workerId": "stream-worker"
            }),
        )
        .await
    });

    barrier.wait().await;
    let (cancel_status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "canceled",
            "stage": "canceled",
            "progress": 1,
            "message": "Canceled by user.",
            "workerId": "stream-worker"
        }),
    )
    .await;
    assert_eq!(cancel_status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");
    barrier.wait().await;

    let (progress_status, _) = progress.await.expect("progress request joins");
    assert_eq!(
        progress_status,
        StatusCode::CONFLICT,
        "a nonterminal progress report accepted after cancellation must receive 409"
    );
    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "canceled");
}

#[tokio::test]
async fn authenticated_lan_caller_cannot_mutate_an_unclaimed_job_without_worker_id() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let auth = [("x-sceneworks-token", "secret-token")];
    let peer = "192.168.1.44:50123";

    let (status, created) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
        peer,
        &auth,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let job_id = created["id"].as_str().expect("job id");

    let (status, rejected) = request_with_peer_headers(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "ownerless completion"
        }),
        peer,
        &auth,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(rejected["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("no longer owns")));

    let (status, unchanged) = request_with_peer_headers(
        app,
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
        peer,
        &auth,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unchanged["status"], "queued");
    assert!(unchanged["result"]
        .as_object()
        .is_some_and(serde_json::Map::is_empty));
}

#[tokio::test]
async fn progress_ticks_only_republish_queue_on_status_change() {
    // sc-4203 (F-API-5): a pure progress tick (status unchanged) must not trigger the
    // full queue-summary recompute + queue.updated broadcast; a status transition must.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    // Move the job into `running` (a transition from `preparing`).
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.2, "message": "step", "workerId": "worker-1" }),
    )
    .await;

    // Subscribe AFTER the transition so we only observe the next ticks' events.
    let mut events = state.events.subscribe();

    // A pure progress tick (running -> running): job.updated, but NOT queue.updated.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "running", "stage": "running", "progress": 0.6, "message": "step", "workerId": "worker-1" }),
    )
    .await;
    let tick_events = drain_event_names(&mut events).await;
    assert!(
        tick_events.iter().any(|name| name == "job.updated"),
        "a progress tick still emits job.updated: {tick_events:?}"
    );
    assert!(
        !tick_events.iter().any(|name| name == "queue.updated"),
        "a pure progress tick must not republish the queue: {tick_events:?}"
    );

    // A status transition (running -> completed) republishes the queue.
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({ "status": "completed", "stage": "completed", "progress": 1, "message": "done", "workerId": "worker-1" }),
    )
    .await;
    let done_events = drain_event_names(&mut events).await;
    assert!(
        done_events.iter().any(|name| name == "queue.updated"),
        "a status transition must republish the queue: {done_events:?}"
    );
}

#[tokio::test]
async fn cleared_terminal_noop_retry_is_owner_checked_and_emits_no_events() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    let terminal_payload = json!({
        "status": "completed",
        "stage": "completed",
        "progress": 1,
        "message": "done",
        "result": {},
        "workerId": "worker-1"
    });
    let (status, completed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        terminal_payload.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(completed["status"], "completed");
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let mut events = state.events.subscribe();
    let (status, retry) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        terminal_payload,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(retry["status"], "completed");
    assert!(
        drain_event_names(&mut events).await.is_empty(),
        "an unapplied, unaugmented same-terminal retry must publish no event"
    );

    let (status, rejected) = request(
        app,
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "stale retry",
            "workerId": "worker-2"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(rejected["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("no longer owns")));
    assert!(
        drain_event_names(&mut events).await.is_empty(),
        "an unauthorized same-terminal retry must publish no event"
    );
}

#[tokio::test]
async fn canceling_queued_job_finishes_without_worker_acknowledgement() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "mist over hills" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let job_id = created["id"].as_str().expect("job id is string");
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");
    assert_eq!(canceled["stage"], "canceled");
    assert_eq!(canceled["progress"], 1.0);
    assert_eq!(canceled["cancelRequested"], true);
    assert_eq!(canceled["message"], "Canceled before a worker started.");
    assert!(canceled["canceledAt"].is_string());
    assert!(canceled["completedAt"].is_string());
    assert_eq!(canceled["workerId"], Value::Null);

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["canceled"], 1);
    assert_eq!(queue["counts"]["queued"], 0);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, claimed) = request(
        app,
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"], Value::Null);
}

#[tokio::test]
async fn clear_jobs_soft_hides_terminal_items_from_the_queue() {
    // sc-12231 / issue #1556: POST /api/v1/jobs/clear drops every terminal job from
    // the queue list + counts, returns the cleared ids, and leaves active jobs alone.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // A queued job we cancel (terminal) + a queued job left active.
    let (_, terminal) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "done" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let terminal_id = terminal["id"].as_str().expect("job id").to_owned();
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/cancel"),
        Value::Null,
    )
    .await;

    let (_, active) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "wait" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let active_id = active["id"].as_str().expect("job id").to_owned();

    // Both are listed before clearing.
    let (_, before) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(before.as_array().expect("jobs array").len(), 2);

    // Clear (empty body == all projects). Reports the one terminal job by id.
    let (status, cleared) = request(app.clone(), "POST", "/api/v1/jobs/clear", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["cleared"], 1);
    assert_eq!(cleared["clearedIds"], json!([terminal_id]));

    // The queue now lists only the still-active job.
    let (_, after) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let ids: Vec<&str> = after
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(ids, vec![active_id.as_str()]);

    // Status counts drop the canceled job; the queued one remains.
    let (_, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(queue["counts"]["canceled"], 0);
    assert_eq!(queue["counts"]["queued"], 1);
}

#[tokio::test]
async fn clear_jobs_scopes_to_the_requested_project() {
    // sc-12231: the clear honors the body's projectId so clearing one workspace's
    // completed items never touches another's.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let mut ids = Vec::new();
    for project in ["project-a", "project-b"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": project,
                "projectName": project,
                "payload": { "prompt": "done" },
                "requestedGpu": "auto"
            }),
        )
        .await;
        let id = job["id"].as_str().expect("job id").to_owned();
        request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{id}/cancel"),
            Value::Null,
        )
        .await;
        ids.push(id);
    }

    // Clear only project-a.
    let (status, cleared) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/clear",
        json!({ "projectId": "project-a" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["clearedIds"], json!([ids[0]]));

    // project-a is empty; project-b's canceled job is untouched.
    let (_, a_jobs) = request(
        app.clone(),
        "GET",
        "/api/v1/jobs?projectId=project-a",
        Value::Null,
    )
    .await;
    assert!(a_jobs.as_array().expect("jobs array").is_empty());
    let (_, b_jobs) = request(app, "GET", "/api/v1/jobs?projectId=project-b", Value::Null).await;
    let b_ids: Vec<&str> = b_jobs
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(b_ids, vec![ids[1].as_str()]);
}

#[tokio::test]
async fn cancel_pending_jobs_cancels_every_queued_item_but_not_active_ones() {
    // sc-13448: POST /api/v1/jobs/cancel-pending flips every pending (queued) job to
    // terminal `canceled` in one call, returns the updated snapshots, and leaves a
    // worker-owned (active) job running.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Register a worker + create three jobs, then claim one so it is active.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let mut ids = Vec::new();
    for prompt in ["one", "two", "three"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": prompt },
                "requestedGpu": "auto"
            }),
        )
        .await;
        ids.push(job["id"].as_str().expect("job id").to_owned());
    }

    // Claim the oldest job — it becomes active (Preparing) and worker-owned.
    let (_, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    let active_id = claimed["job"]["id"]
        .as_str()
        .expect("claimed id")
        .to_owned();
    assert_eq!(claimed["job"]["status"], "preparing");

    // Cancel all pending (empty body == all projects). The two still-queued jobs are
    // canceled; the active job is not.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/cancel-pending",
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["canceled"], 2);
    let canceled_jobs = canceled["jobs"].as_array().expect("jobs array");
    assert_eq!(canceled_jobs.len(), 2);
    for job in canceled_jobs {
        assert_eq!(job["status"], "canceled");
        assert_eq!(job["stage"], "canceled");
        assert_eq!(job["cancelRequested"], true);
        assert_eq!(job["message"], "Canceled before a worker started.");
        assert!(job["canceledAt"].is_string());
        assert_ne!(job["id"], json!(active_id));
    }

    // The active job is untouched: still preparing, no cancel requested.
    let (_, active) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{active_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(active["status"], "preparing");
    assert_eq!(active["cancelRequested"], false);

    // Status counts: two canceled, none queued, the active one still in flight.
    let (_, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(queue["counts"]["canceled"], 2);
    assert_eq!(queue["counts"]["queued"], 0);

    // A second sweep cancels nothing — no pending jobs remain.
    let (status, again) = request(app, "POST", "/api/v1/jobs/cancel-pending", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(again["canceled"], 0);
}

#[tokio::test]
async fn prioritize_jobs_moves_selected_pending_work_ahead_for_the_next_claim() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-priority",
            "gpuId": "gpu-0",
            "gpuName": "GPU 0",
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (_, first) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "payload": { "prompt": "first" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let (_, selected) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "payload": { "prompt": "selected" },
            "requestedGpu": "auto"
        }),
    )
    .await;

    let selected_id = selected["id"].as_str().expect("selected id");
    let (status, prioritized) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/prioritize",
        json!({ "jobIds": [selected_id] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(prioritized["prioritized"], 1);
    assert_eq!(prioritized["jobs"][0]["id"], selected["id"]);
    assert!(prioritized["jobs"][0]["queueRank"]
        .as_i64()
        .is_some_and(|rank| rank > 0));

    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-priority" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(claimed["job"]["id"], selected["id"]);
    let (_, first_after) = request(
        app,
        "GET",
        &format!("/api/v1/jobs/{}", first["id"].as_str().expect("first id")),
        Value::Null,
    )
    .await;
    assert_eq!(first_after["status"], "queued");
}

#[tokio::test]
async fn clear_single_job_soft_hides_only_that_terminal_job() {
    // sc-12231 / issue #1556: POST /api/v1/jobs/:id/clear (the per-card ×) drops one
    // terminal job from the queue and leaves its siblings alone.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // Two queued jobs; cancel one so it is terminal, leave the other active.
    let mut ids = Vec::new();
    for prompt in ["done", "wait"] {
        let (_, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_detail",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": prompt },
                "requestedGpu": "auto"
            }),
        )
        .await;
        ids.push(job["id"].as_str().expect("job id").to_owned());
    }
    let (terminal_id, active_id) = (ids[0].clone(), ids[1].clone());
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/cancel"),
        Value::Null,
    )
    .await;

    // Clear just the terminal one.
    let (status, cleared) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{terminal_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["id"], terminal_id);

    // Only the still-active job remains in the queue.
    let (_, after) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    let remaining: Vec<&str> = after
        .as_array()
        .expect("jobs array")
        .iter()
        .filter_map(|job| job["id"].as_str())
        .collect();
    assert_eq!(remaining, vec![active_id.as_str()]);
}

#[tokio::test]
async fn clear_publishes_a_live_tombstone_and_retains_it_for_reconnect() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");
    let (_, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "done" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = job["id"].as_str().expect("job id").to_owned();
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/cancel"),
        Value::Null,
    )
    .await;

    let mut events = state.events.subscribe();
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let names = drain_event_names(&mut events).await;
    assert!(
        names.iter().any(|name| name == "jobs.cleared"),
        "connected peers need an explicit clear tombstone: {names:?}"
    );

    let (status, ticket) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/events/ticket",
        json!({ "knownTerminalJobIds": [job_id] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ticket = ticket["ticket"].as_str().expect("event ticket");
    let (status, reconnect) =
        request_sse_prefix(app, &format!("/api/v1/jobs/events?ticket={ticket}"), 3).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        reconnect[1].1["jobs"]
            .as_array()
            .expect("snapshot jobs")
            .iter()
            .all(|job| job["id"] != json!(job_id)),
        "a requested soft-hidden row must not be composed back into the snapshot"
    );
    assert!(
        reconnect[1].1["clearedJobIds"]
            .as_array()
            .expect("clear tombstones array")
            .iter()
            .any(|id| id == &json!(job_id)),
        "a peer reconnecting after the live event must receive the persistent tombstone"
    );
}

#[tokio::test]
async fn clear_single_job_rejects_a_non_terminal_job() {
    // sc-12231: clearing an active (queued) job is a 400 — the × only appears on
    // terminal cards, and the server refuses to soft-hide a live job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_detail",
            "projectId": "project-1",
            "projectName": "Project 1",
            "payload": { "prompt": "wait" },
            "requestedGpu": "auto"
        }),
    )
    .await;
    let job_id = job["id"].as_str().expect("job id").to_owned();

    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/clear"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // The job is untouched — still listed.
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().expect("jobs array").len(), 1);
}

#[tokio::test]
async fn image_job_route_threads_upscale_contract_when_enabled() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "z_image_turbo",
              "name": "Z-Image-Turbo",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image"],
              "downloads": [],
              "paths": {},
              "resources": {
                "imageUpscalers": {
                  "real-esrgan": {
                    "x2": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x2plus.pth" },
                    "x4": { "repo": "nateraw/real-esrgan", "file": "RealESRGAN_x4plus.pth" }
                  }
                }
              },
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": [], "types": [] },
              "ui": {}
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let base_request = json!({
        "projectId": "project-1",
        "mode": "text_to_image",
        "prompt": "mist over hills",
        "count": 1,
        "seed": 123
    });
    let (status, base_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        base_request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(base_job["payload"].get("upscale").is_none());
    assert_eq!(
        base_job["payload"]["modelManifestEntry"]["resources"]["imageUpscalers"]["real-esrgan"]
            ["x4"]["file"],
        json!("RealESRGAN_x4plus.pth")
    );

    let mut disabled_request = base_request.clone();
    disabled_request["upscale"] = json!({ "enabled": false, "factor": 4, "engine": "real-esrgan" });
    let (status, disabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", disabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(disabled_job["payload"], base_job["payload"]);

    let mut enabled_request = base_request;
    enabled_request["upscale"] = json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" });
    let (status, enabled_job) =
        request(app.clone(), "POST", "/api/v1/image/jobs", enabled_request).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        enabled_job["payload"]["upscale"],
        json!({ "enabled": true, "factor": 4, "engine": "real-esrgan" })
    );

    let (status, error) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 1,
            "seed": 123,
            "upscale": { "enabled": true, "factor": 3, "engine": "real-esrgan" }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["detail"], "upscale.factor must be 2 or 4");
}

#[tokio::test]
async fn image_job_route_threads_reference_asset_ids() {
    // sc-6358 / sc-6107 regression guard: the multi-reference edit picker sends a top-level
    // `referenceAssetIds` array. The typed ImageJobRequest must carry the plural list through to the
    // job payload — without the field, serde drops the unknown key on deserialize and `to_json_object`
    // never forwards it, so the worker's `flux2_edit_reference_ids` never sees the references and the
    // FLUX.2 multi-reference edit silently no-ops (the original sc-6211 defect).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, edit_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "in the style of the references",
            "sourceAssetId": "work-scratch",
            "referenceAssetIds": ["work-scratch", "ref-a", "ref-b"],
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(edit_job["type"], "image_edit");
    assert_eq!(
        edit_job["payload"]["referenceAssetIds"],
        json!(["work-scratch", "ref-a", "ref-b"])
    );

    // A request that doesn't send the list still serializes a present (empty) array — the worker's
    // `string_list` treats missing/empty identically, so this never surprises a single-reference edit.
    let (status, plain_job) = request(
        app,
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make it dusk",
            "sourceAssetId": "asset-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(plain_job["payload"]["referenceAssetIds"], json!([]));
}

#[tokio::test]
async fn ideogram_plain_text_job_returns_immediately_in_pending_caption() {
    // sc-9120: a direct/headless plain-text Ideogram 4 job returns 201 IMMEDIATELY in the non-claimable
    // `pending_caption` status — the POST no longer waits on the magic-prompt expansion at all. A
    // background watcher then runs the same separate expansion the web runs, rewrites the prompt to the
    // rich JSON caption, and promotes the job to `queued`, so the worker only ever sees it once queued.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    // The POST is NOT spawned/awaited concurrently — it must return on its own, promptly.
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "count": 1,
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["type"], "image_generate");
    assert_eq!(
        image_job["status"], "pending_caption",
        "the POST must return immediately in pending_caption, not wait on the caption"
    );
    // Still the ORIGINAL prompt at this point — the rewrite happens on the async promotion.
    assert_eq!(
        image_job["payload"]["prompt"],
        "a red fox in a snowy forest"
    );
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // The background watcher enqueues the magic-prompt expansion carrying the plain prompt, the
    // magic_prompt task, and the derived aspect ratio.
    let refine_id = wait_for_prompt_refine_job(&app).await;
    let (status, refine_job) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{refine_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(refine_job["type"], "prompt_refine");
    assert_eq!(refine_job["payload"]["task"], "magic_prompt");
    assert_eq!(
        refine_job["payload"]["prompt"],
        "a red fox in a snowy forest"
    );
    assert_eq!(refine_job["payload"]["aspectRatio"], "1:1");

    // Complete the expansion with a rich caption through a real worker claim.
    let caption = r#"{"high_level_description": "a red fox", "compositional_deconstruction": {"background": "a snowy forest at golden hour", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;

    // The watcher now promotes the image job to `queued` with the rich caption as its prompt.
    let promoted = wait_for_job_out_of_pending_caption(&app, &image_job_id).await;
    assert_eq!(promoted["status"], "queued");
    assert_eq!(promoted["payload"]["model"], "ideogram_4");
    assert_eq!(promoted["payload"]["prompt"], caption);
}

#[tokio::test]
async fn ideogram_plain_text_job_degrades_to_original_prompt_when_expansion_fails() {
    // sc-9120 graceful degradation: if the magic_prompt expansion fails (e.g. the refiner is not
    // staged), the background watcher still promotes the image job to `queued` with the ORIGINAL prompt
    // — the worker's format-guard + placeholder reseed net (sc-6501) remains the fallback, so the job
    // is never stranded in pending_caption and a render is always produced.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    let refine_id = wait_for_prompt_refine_job(&app).await;
    const WORKER_ID: &str = "test-prompt-refine-worker";
    claim_job_as_worker(&app, &refine_id, WORKER_ID, &["gpu", "prompt_refine"]).await;
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{refine_id}/progress"),
        json!({
            "status": "failed",
            "stage": "failed",
            "progress": 0,
            "message": "Expansion failed.",
            "error": "prompt-refine model not staged",
            "workerId": WORKER_ID
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let promoted = wait_for_job_out_of_pending_caption(&app, &image_job_id).await;
    assert_eq!(promoted["status"], "queued");
    assert_eq!(promoted["type"], "image_generate");
    assert_eq!(promoted["payload"]["prompt"], "a red fox in a snowy forest");
}

#[tokio::test]
async fn ideogram_plain_text_job_degrades_on_invalid_caption_after_bounded_resamples() {
    // sc-9120: the expansion runs in a BACKGROUND task (no HTTP connection held), so a completed-but-
    // invalid caption may be re-sampled a small, bounded number of times (MAX_CAPTION_ATTEMPTS). When
    // every attempt returns prose (not a caption), the watcher degrades the image job to `queued` with
    // the ORIGINAL prompt (the worker's reseed net recovers it). The re-sample budget is small and
    // bounded, so an impatient client's retries can't stack unbounded magic-prompt jobs.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // Feed every re-sample a prose (non-caption) reply until the watcher exhausts its budget and
    // degrades. Completing each refine job unblocks the next attempt (a fresh refine job).
    let mut previous: Option<String> = None;
    loop {
        let job =
            wait_for_job_out_of_pending_caption_or_refine(&app, &image_job_id, previous.as_deref())
                .await;
        match job {
            PendingOrRefine::Promoted(promoted) => {
                // Degraded to the original prompt once the budget was exhausted.
                assert_eq!(promoted["status"], "queued");
                assert_eq!(promoted["payload"]["prompt"], "a red fox in a snowy forest");
                break;
            }
            PendingOrRefine::Refine(refine_id) => {
                complete_prompt_refine_job(
                    &app,
                    &refine_id,
                    json!({ "refinedPrompt": "just a fox, nothing structured" }),
                )
                .await;
                previous = Some(refine_id);
            }
        }
    }
}

#[tokio::test]
async fn pending_caption_ideogram_job_is_cancelable() {
    // sc-9120: a pending_caption job must be cancelable — it goes straight to `canceled` (no worker to
    // acknowledge), and a subsequent caption promotion does NOT resurrect it (the guarded UPDATE only
    // matches a still-pending row).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "a red fox in a snowy forest",
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["status"], "pending_caption");
    let image_job_id = image_job["id"].as_str().expect("job id").to_owned();

    // Cancel while still pending — it terminates immediately.
    let (status, canceled) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{image_job_id}/cancel"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(canceled["status"], "canceled");

    // Even if the background watcher's expansion later completes and it tries to promote, the canceled
    // job must NOT flip back to queued. Complete the refine job it enqueued and confirm the image job
    // stays canceled.
    let refine_id = wait_for_prompt_refine_job(&app).await;
    let caption =
        r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;
    // Give the watcher a moment to attempt (and no-op) the promotion.
    for _ in 0..25 {
        let (_, job) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/jobs/{image_job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(
            job["status"], "canceled",
            "a canceled pending_caption job must never be resurrected by a late promotion"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn concurrent_ideogram_captions_share_one_refine_job() {
    // sc-9120: two identical plain-text Ideogram jobs (an impatient client re-POSTing) must reuse ONE
    // in-flight magic-prompt refine job rather than stacking a fresh one each time. Both image jobs land
    // in pending_caption; the second caption watcher reuses the first's refine job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let post = |app: axum::Router| async move {
        request(
            app,
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "mode": "text_to_image",
                "prompt": "a red fox in a snowy forest",
                "model": "ideogram_4",
                "seed": 7
            }),
        )
        .await
    };

    let (status_a, job_a) = post(app.clone()).await;
    assert_eq!(status_a, StatusCode::CREATED);
    assert_eq!(job_a["status"], "pending_caption");
    // Wait for the first refine job to be in flight before the second POST, so the reuse path is hit
    // deterministically.
    let refine_id = wait_for_prompt_refine_job(&app).await;

    let (status_b, job_b) = post(app.clone()).await;
    assert_eq!(status_b, StatusCode::CREATED);
    assert_eq!(job_b["status"], "pending_caption");
    assert_ne!(job_a["id"], job_b["id"], "two distinct image jobs");

    // Let the second watcher run its reuse lookup, then assert exactly one refine job exists.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    let refine_ids: Vec<String> = jobs
        .as_array()
        .expect("jobs is an array")
        .iter()
        .filter(|job| job["type"] == "prompt_refine")
        .filter_map(|job| job["id"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(
        refine_ids,
        vec![refine_id.clone()],
        "the second identical caption must reuse the in-flight refine job, not stack a new one"
    );

    // Completing the shared refine job promotes BOTH image jobs to queued with the rich caption.
    let caption =
        r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": []}}"#;
    complete_prompt_refine_job(&app, &refine_id, json!({ "refinedPrompt": caption })).await;
    let a = wait_for_job_out_of_pending_caption(&app, job_a["id"].as_str().unwrap()).await;
    let b = wait_for_job_out_of_pending_caption(&app, job_b["id"].as_str().unwrap()).await;
    assert_eq!(a["status"], "queued");
    assert_eq!(a["payload"]["prompt"], caption);
    assert_eq!(b["status"], "queued");
    assert_eq!(b["payload"]["prompt"], caption);
}

#[tokio::test]
async fn ideogram_caption_prompt_dispatches_without_expansion() {
    // sc-6519: an already-structured caption (the normal web submit) is never re-expanded — no
    // magic_prompt job is enqueued and the job dispatches immediately with the caption unchanged.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let caption =
        r#"{"compositional_deconstruction": {"background": "a beach at sunset", "elements": []}}"#;
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": caption,
            "model": "ideogram_4",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["prompt"], caption);

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "an already-structured caption must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn ideogram_edit_job_skips_auto_caption() {
    // sc-6519: an Ideogram 4 EDIT job conditions on a source image and its prompt is an edit
    // instruction, not a scene to caption — the auto-caption must not rewrite it, and no magic_prompt
    // job is enqueued.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make the sky purple",
            "model": "ideogram_4",
            "sourceAssetId": "asset-1",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "image_edit");
    assert_eq!(job["payload"]["prompt"], "make the sky purple");

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "an Ideogram edit job must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn non_ideogram_image_job_skips_auto_caption() {
    // sc-6519: the auto-caption is gated to the Ideogram 4 models — a plain prompt for any other
    // model dispatches immediately, unchanged, with no magic_prompt expansion job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "model": "flux_dev",
            "seed": 7
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["prompt"], "mist over hills");

    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert!(
        jobs.as_array()
            .expect("jobs is an array")
            .iter()
            .all(|job| job["type"] != "prompt_refine"),
        "a non-Ideogram job must not enqueue a magic_prompt job"
    );
}

#[tokio::test]
async fn image_caption_refine_job_resolves_asset_to_confined_image_path() {
    // epic 8102 / sc-8108: the reference-image → JSON caption flow POSTs `task: "image_caption"` with a
    // project `sourceAssetId` (+ `projectId`) and the vision model's repo. The handler resolves the
    // asset to an absolute on-disk `imagePath` (inside the project dir), forwards the model verbatim,
    // and enqueues a `prompt_refine` job carrying that image-caption payload (no text prompt required).
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Caption Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Reference.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let rel_path = asset["file"]["path"]
        .as_str()
        .expect("file path")
        .to_owned();

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_caption",
            "sourceAssetId": asset_id,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "prompt_refine");
    assert_eq!(job["payload"]["task"], "image_caption");
    assert_eq!(
        job["payload"]["model"],
        "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
    );
    // No text prompt is required for an image-caption job.
    assert!(job["payload"].get("prompt").is_none());
    // The resolved imagePath is the asset's absolute on-disk location inside the project dir.
    // Compare as paths: the handler joins component-wise (native separators), while `rel_path`
    // keeps the asset record's literal `/`, so a string comparison breaks on Windows (sc-8967).
    let expected = project_path.join(&rel_path);
    assert_eq!(
        std::path::Path::new(job["payload"]["imagePath"].as_str().unwrap()),
        expected
    );
}

#[tokio::test]
async fn image_caption_refine_job_requires_source_asset_and_project() {
    // sc-8108: the image-caption task is driven by a reference asset, so it must reject a request that
    // omits the `sourceAssetId` or `projectId` rather than enqueue an unresolvable job.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_caption", "projectId": "project-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_caption", "sourceAssetId": "asset-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_describe_refine_job_resolves_asset_and_forwards_caption_style() {
    // epic 8203 / sc-8206: the reference-image → plain-text DESCRIBE flow POSTs `task: "image_describe"`
    // with a project `sourceAssetId` (+ `projectId`), the vision model's repo, and an optional
    // `captionStyle`. The handler resolves the asset to a confined on-disk `imagePath` (same path as the
    // caption flow) and forwards `model` + `captionStyle` verbatim, with no text prompt required.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Describe Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "Reference.PNG",
        "image/png",
        b"png-bytes",
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let rel_path = asset["file"]["path"]
        .as_str()
        .expect("file path")
        .to_owned();

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetId": asset_id,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
            "captionStyle": "tags"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "prompt_refine");
    assert_eq!(job["payload"]["task"], "image_describe");
    assert_eq!(job["payload"]["captionStyle"], "tags");
    assert!(job["payload"].get("prompt").is_none());
    // Path (not string) comparison: separator-agnostic on Windows (sc-8967, same as the caption test).
    let expected = project_path.join(&rel_path);
    assert_eq!(
        std::path::Path::new(job["payload"]["imagePath"].as_str().unwrap()),
        expected
    );
}

#[tokio::test]
async fn image_describe_refine_job_requires_source_asset_and_project() {
    // sc-8206: the describe task is image-driven like image_caption, so it must reject a request missing
    // the `sourceAssetId` or `projectId`.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_describe", "projectId": "project-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({ "task": "image_describe", "sourceAssetId": "asset-1" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn mood_board_refine_job_resolves_multiple_assets_to_image_paths() {
    // epic 8588 / sc-8595: a "mood board" describe POSTs `sourceAssetIds` (plural). The handler resolves
    // EACH id to a confined on-disk path, in order, and forwards them as the worker's `imagePaths` array
    // (no scalar `imagePath`), so the vision model synthesizes ONE prompt from the shared aesthetic.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Mood Board Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

    let mut asset_ids = Vec::new();
    let mut rel_paths = Vec::new();
    for name in ["First.png", "Second.png"] {
        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            name,
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        asset_ids.push(asset["id"].as_str().expect("asset id").to_owned());
        rel_paths.push(
            asset["file"]["path"]
                .as_str()
                .expect("file path")
                .to_owned(),
        );
    }

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetIds": asset_ids,
            "projectId": project_id,
            "model": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["task"], "image_describe");
    // The plural array is emitted; the scalar single-image key is NOT.
    assert!(job["payload"].get("imagePath").is_none());
    let paths = job["payload"]["imagePaths"]
        .as_array()
        .expect("imagePaths array");
    assert_eq!(paths.len(), 2, "both references resolved");
    for (path, rel) in paths.iter().zip(rel_paths.iter()) {
        // Path (not string) comparison: separator-agnostic on Windows (sc-8967).
        let expected = project_path.join(rel);
        assert_eq!(std::path::Path::new(path.as_str().unwrap()), expected);
    }
}

#[tokio::test]
async fn mood_board_refine_job_rejects_more_than_the_cap() {
    // sc-8595: the server-side ceiling (MAX_MOOD_BOARD_IMAGES) is authoritative — a board over the cap is
    // rejected with 400 before any asset resolution, so a runaway list cannot exhaust the vision runtime.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let too_many: Vec<String> = (0..(crate::prompts::MAX_MOOD_BOARD_IMAGES + 1))
        .map(|i| format!("asset-{i}"))
        .collect();
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        json!({
            "task": "image_describe",
            "sourceAssetIds": too_many,
            "projectId": "project-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_and_video_job_routes_normalize_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "projectName": "Project 1",
            "mode": "text_to_image",
            "prompt": "mist over hills",
            "count": 2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["type"], "image_generate");
    assert_eq!(image_job["projectId"], "project-1");
    assert!(image_job["payload"].get("requestedGpu").is_none());
    assert_eq!(image_job["payload"]["seed"], Value::Null);
    assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 2);

    let (status, edit_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": "project-1",
            "mode": "edit_image",
            "prompt": "make it dusk",
            "sourceAssetId": "asset-1",
            "seed": 42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(edit_job["type"], "image_edit");
    assert!(edit_job["payload"].get("seeds").is_none());

    let (status, wide_seed_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": " ",
            "mode": "text_to_image",
            "prompt": "space project id stays Python-compatible",
            "seed": -42
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(wide_seed_job["payload"]["projectId"], " ");
    assert_eq!(wide_seed_job["payload"]["seed"], -42);

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            // Names `wan_2_2` rather than riding `default_video_model()`, which is `ltx_2_3`.
            // LTX's replacement provider is the IC-LoRA keyframe-append path, so
            // `video_request_is_mlx_eligible` and `ltx_replace_candle_eligible` both require
            // `loras_contain_ltx_ic_lora`; an adapter-free LTX `replace_person` is claimed by NO
            // lane, and sc-19504's enqueue gate correctly 400s it rather than letting it sit
            // `queued` forever. Supplying an adapter instead would need a seeded LoRA catalog and
            // an on-disk weight file, dragging catalog resolution into a test that exists to prove
            // payload NORMALIZATION. `wan_2_2` is claimable adapter-free on both lanes (native
            // Wan-VACE), so the shape below is unchanged and the assertions stay on topic. The
            // server-default video model has its own dedicated coverage further down this file.
            "model": "wan_2_2",
            "mode": "replace_person",
            "prompt": "hero walks through rain",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(video_job["type"], "person_replace");
    assert!(video_job["payload"].get("requestedGpu").is_none());

    let (status, integer_duration_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "integer duration stays an integer",
            "duration": 6
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(integer_duration_job["payload"]["duration"], 6);

    let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 5);
}

/// THE regression test the cap change required: a video model that declares none of the four new
/// `limits` keys keeps the exact reference budget it had before sc-17160.
///
/// The shape that matters is 8 images + 8 clips — SIXTEEN reference files on one request, which
/// this route accepts today because the pre-story caps were per-list only. Introducing the combined
/// cap as a payload-sanity BLANKET of 12 (the reading the story's wording invites) would have
/// refused it, silently narrowing every already-shipped video model. That is why the combined cap
/// is per-model declaration only, and why this test asserts an ACCEPT rather than a reject: the
/// rejecting tests would all still pass with the regression in place.
#[tokio::test]
async fn existing_video_models_keep_their_pre_sc_17160_reference_budget() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Budget" }),
    )
    .await;

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit",
            "sourceClipAssetId": "clip-src",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"],
            "sourceClipAssetIds": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "16 reference files was legal before this story and must stay legal: {job}"
    );
    assert_eq!(
        job["payload"]["referenceAssetIds"].as_array().map(Vec::len),
        Some(8)
    );
    assert_eq!(
        job["payload"]["sourceClipAssetIds"]
            .as_array()
            .map(Vec::len),
        Some(8)
    );
}

/// The story's acceptance pair at the ROUTE, against a model that declares the Ref2VA reference
/// surface: 9 images + 3 clips + 3 audio is 15 reference files and must be refused; 9 + 2 + 1 is
/// 12 and must be accepted, with all three lists on the enqueued payload.
///
/// Driven off a SEEDED manifest rather than a shipped model on purpose. The caps are per-model
/// (`limits.maxReferenceAssets` / `maxSourceClipAssets` / `maxReferenceAudioAssets` /
/// `maxCombinedReferenceAssets`), so the only honest way to exercise the accepting side today is a
/// model that declares them — MiniMax-H3's own manifest entry is sc-17158. That separation is the
/// point of the design: this test needs no engine and no weights to pin the contract.
///
/// The seeded entry carries a REAL routed id (sc-19504): the enqueue no-lane gate refuses a model
/// no lane can claim, so the old synthetic `ref2va_probe` would now 400 before reaching the caps.
/// The entry is still entirely this test's own — its four caps are the fixture's, not the shipped
/// manifest's — so the assertions are unchanged; only the id is one the routing tables recognise.
///
/// The combined budget has NO payload-sanity blanket, only this per-model declaration — see the
/// note in `validate_video_job`. 12 is MiniMax-H3's number and today's per-list caps already admit
/// a 16-file request (8 images + 8 clips), so a blanket 12 would have narrowed every existing
/// video model. `existing_video_models_keep_their_pre_sc_17160_reference_budget` holds that line.
#[tokio::test]
async fn ref2va_reference_caps_refuse_fifteen_files_and_admit_twelve() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "minimax_h3_ref",
              "name": "Ref2VA Probe",
              "family": "minimax-h3",
              "type": "video",
              "adapter": "stub",
              "capabilities": ["text_to_video", "reference_to_video"],
              "downloads": [],
              "paths": {},
              "defaults": {},
              "limits": {
                "maxReferenceAssets": 9,
                "maxSourceClipAssets": 3,
                "maxReferenceAudioAssets": 3,
                "maxCombinedReferenceAssets": 12
              },
              "ui": {}
            },
            {
              "id": "legacy_probe",
              "name": "Legacy Probe",
              "family": "ltx-video",
              "type": "video",
              "adapter": "stub",
              "capabilities": ["text_to_video", "reference_to_video"],
              "downloads": [],
              "paths": {},
              "defaults": {},
              "limits": {},
              "ui": {}
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Ref2VA" }),
    )
    .await;

    let submit = |model: &str, images: usize, clips: usize, audio: usize| {
        let app = app.clone();
        let body = json!({
            "projectId": "project-1",
            "model": model,
            "mode": "reference_to_video",
            "prompt": "the subject speaks",
            "referenceAssetIds": (0..images).map(|i| format!("img-{i}")).collect::<Vec<_>>(),
            "sourceClipAssetIds": (0..clips).map(|i| format!("clip-{i}")).collect::<Vec<_>>(),
            "referenceAudioAssetIds": (0..audio).map(|i| format!("aud-{i}")).collect::<Vec<_>>(),
        });
        async move { request(app, "POST", "/api/v1/video/jobs", body).await }
    };

    // 9 + 3 + 3 = 15 > 12. REFUSED, and the message decomposes the total so the caller knows how
    // much to cut without counting three lists themselves. Nothing but the combined budget can
    // decide this one: 9 <= 9, 3 <= 3 and 3 <= 3 all pass their own caps.
    let (status, over) = submit("minimax_h3_ref", 9, 3, 3).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        over["detail"],
        "minimax_h3_ref takes up to 12 reference files in total, but this request supplies 15 \
         (9 reference images + 3 source clips + 3 audio references). Remove 3 of them."
    );

    // 9 + 2 + 1 = 12. ACCEPTED, at the cap, and every list reaches the enqueued payload. These
    // four assertions were withheld together with the route while
    // `minimax_h3_ref`/`reference_to_video` had no claiming lane; sc-18650 restored the MLX route
    // and them with it.
    let (status, at_cap) = submit("minimax_h3_ref", 9, 2, 1).await;
    assert_eq!(status, StatusCode::CREATED, "{at_cap}");
    assert_eq!(at_cap["type"], "video_generate");
    assert_eq!(
        at_cap["payload"]["referenceAssetIds"]
            .as_array()
            .map(Vec::len),
        Some(9)
    );
    assert_eq!(
        at_cap["payload"]["sourceClipAssetIds"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        at_cap["payload"]["referenceAudioAssetIds"],
        json!(["aud-0"]),
        "the audio references must reach the worker verbatim, not just validate"
    );

    // A shape that clears the blanket but not what THIS model declares: 4 clips against its
    // declared 3, only 8 files in total. Refused by the per-model gate, naming the model.
    let (status, per_model) = submit("minimax_h3_ref", 4, 4, 0).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        per_model["detail"],
        "minimax_h3_ref takes up to 3 source clips, but this request supplies 4. Reduce \
         sourceClipAssetIds to 3 or fewer, or choose a model that takes more."
    );

    // The SAME 9 + 2 + 1 request against a model that declares nothing is refused — twice over,
    // and the image cap is what it trips first. This is the per-family shape doing its job: the
    // declaration travels with the model, not with the API constant.
    let (status, legacy) = submit("legacy_probe", 9, 2, 1).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        legacy["detail"],
        "legacy_probe takes up to 8 reference images, but this request supplies 9. Reduce \
         referenceAssetIds to 8 or fewer, or choose a model that takes more."
    );
}

#[tokio::test]
async fn bernini_video_modes_validate_required_media() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Bernini" }),
    )
    .await;

    // video_to_video without a source clip is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_to_video without reference images is rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // reference_video_to_video needs BOTH a source clip and references.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Blank reference ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Reference id lists are bounded before the worker has to encode them.
    //
    // THE REGRESSION ASSERTION FOR sc-17160. This 9-reference request 400'd before that story and
    // must keep 400ing after it, even though the API's payload-sanity blanket was raised from 8 to
    // 9 for MiniMax-H3 — bernini declares no `limits.maxReferenceAssets`, so the per-model gate in
    // `create_video_job` holds it at the historical 8. The status alone would pass for the wrong
    // reason (any 400 satisfies it), so the message is asserted too: it has to be bernini's own cap
    // talking, not the blanket.
    let (status, over_cap) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1", "ref-2", "ref-3", "ref-4", "ref-5", "ref-6", "ref-7", "ref-8", "ref-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        over_cap["detail"],
        "bernini takes up to 8 reference images, but this request supplies 9. Reduce \
         referenceAssetIds to 8 or fewer, or choose a model that takes more."
    );

    // 8 is still admitted — the cap moved for nobody, in either direction.
    let (status, at_cap) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1", "ref-2", "ref-3", "ref-4", "ref-5", "ref-6", "ref-7", "ref-8"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        at_cap["payload"]["referenceAssetIds"]
            .as_array()
            .map(Vec::len),
        Some(8)
    );
    // And the new list is present-but-empty on a model that has nothing to do with it, so a replay
    // reader never has to tell "absent" from "empty" (sc-12345).
    assert_eq!(at_cap["payload"]["referenceAudioAssetIds"], json!([]));

    // The new audio list is INERT for every already-shipped video model: bernini declares no
    // `limits.maxReferenceAudioAssets`, which defaults to 0, so a single audio reference is
    // refused rather than accepted-and-silently-dropped by an engine that cannot consume it.
    let (status, audio_refused) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1"],
            "referenceAudioAssetIds": ["voice-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        audio_refused["detail"],
        "bernini takes no audio references, but this request supplies 1. Remove \
         referenceAudioAssetIds, or choose a model that conditions on audio references."
    );

    // Blank audio ids are rejected exactly as blank image and clip ids are, and by the blanket —
    // so the refusal does not depend on the model declaring anything.
    let (status, blank_audio) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_to_video",
            "prompt": "the subject dances",
            "referenceAssetIds": ["ref-1"],
            "referenceAudioAssetIds": ["  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        blank_audio["detail"],
        "referenceAudioAssetIds must not contain blank ids"
    );

    // A complete video_to_video request creates a base video_generate job that
    // carries the source clip.
    let (status, v2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "video_to_video",
            "prompt": "make it golden hour",
            "sourceClipAssetId": "clip-a"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(v2v_job["type"], "video_generate");
    assert_eq!(v2v_job["payload"]["mode"], "video_to_video");
    assert_eq!(v2v_job["payload"]["sourceClipAssetId"], "clip-a");

    // A complete reference_video_to_video request carries both the clip and the refs.
    let (status, rv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "reference_video_to_video",
            "prompt": "swap the subject",
            "sourceClipAssetId": "clip-a",
            "referenceAssetIds": ["ref-1", "ref-2"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(rv2v_job["type"], "video_generate");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][0], "ref-1");
    assert_eq!(rv2v_job["payload"]["referenceAssetIds"][1], "ref-2");

    // multi_video_to_video (sc-5425) needs at least two source clips.
    for clips in [json!([]), json!(["clip-a"])] {
        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "model": "bernini",
                "mode": "multi_video_to_video",
                "prompt": "blend the takes",
                "sourceClipAssetIds": clips
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // Blank source-clip ids are rejected.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "  "]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Source clip lists are bounded before worker-side video conditioning.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-1", "clip-2", "clip-3", "clip-4", "clip-5", "clip-6", "clip-7", "clip-8", "clip-9"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A complete multi_video_to_video request carries the clip array.
    let (status, mv2v_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "multi_video_to_video",
            "prompt": "blend the takes",
            "sourceClipAssetIds": ["clip-a", "clip-b"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(mv2v_job["type"], "video_generate");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][0], "clip-a");
    assert_eq!(mv2v_job["payload"]["sourceClipAssetIds"][1], "clip-b");

    // ads2v (sc-5425) needs a source clip, a reference video, AND >=1 reference image.
    let ads2v_incomplete = [
        json!({ "referenceClipAssetId": "clip-ref", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceAssetIds": ["ref-1"] }),
        json!({ "sourceClipAssetId": "clip-src", "referenceClipAssetId": "clip-ref" }),
    ];
    for extra in ads2v_incomplete {
        let mut body = json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip"
        });
        let object = body.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        let (status, _) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // A complete ads2v request carries the source clip, reference video, and references.
    let (status, ads2v_job) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "bernini",
            "mode": "ads2v",
            "prompt": "drive the edit with the reference clip",
            "sourceClipAssetId": "clip-src",
            "referenceClipAssetId": "clip-ref",
            "referenceAssetIds": ["ref-1"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(ads2v_job["type"], "video_generate");
    assert_eq!(ads2v_job["payload"]["sourceClipAssetId"], "clip-src");
    assert_eq!(ads2v_job["payload"]["referenceClipAssetId"], "clip-ref");
    assert_eq!(ads2v_job["payload"]["referenceAssetIds"][0], "ref-1");
}

/// SCAIL-2 standalone character animation reaches the queue (GH #2074). `animate_character` was
/// wired everywhere BUT this route's mode allow-list — catalog `capabilities`, `VIDEO_UI_MODES`,
/// `video_mode_is_mlx_eligible`, the candle claim gate, the worker's `generate_scail2`, and the
/// Video Studio mode picker all shipped it (sc-5448 / sc-5449), so the studio offered a mode the
/// API answered with 400 "Unsupported video mode". The declaration being correct in every OTHER
/// layer is exactly why nothing caught it; this test pins REACHABILITY, not declaration.
///
/// It also pins the required-media contract, which mirrors the worker's
/// `resolve_scail2_conditioning`: a driving clip plus a character image (`referenceAssetIds[0]`
/// preferred, `sourceAssetId` accepted) — both hard engine inputs, so an incomplete request is
/// refused at enqueue instead of failing minutes into the render.
#[tokio::test]
async fn scail2_animate_character_reaches_the_queue_and_validates_media() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SCAIL-2" }),
    )
    .await;

    // The regression itself: a complete request is ACCEPTED, not "Unsupported video mode".
    let (status, animate_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving",
            "referenceAssetIds": ["ref-character"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The base video type, which is what `video_job_is_mlx_eligible` gates on for this mode.
    assert_eq!(animate_job["type"], "video_generate");
    assert_eq!(animate_job["payload"]["mode"], "animate_character");
    assert_eq!(animate_job["payload"]["sourceClipAssetId"], "clip-driving");
    assert_eq!(
        animate_job["payload"]["referenceAssetIds"][0],
        "ref-character"
    );

    // The worker also accepts the i2v `sourceAssetId` as the character.
    let (status, source_asset_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving",
            "sourceAssetId": "ref-character"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        source_asset_job["payload"]["sourceAssetId"],
        "ref-character"
    );

    // No driving video, and no character image: each is refused on its own.
    let incomplete = [
        json!({ "referenceAssetIds": ["ref-character"] }),
        json!({ "sourceClipAssetId": "clip-driving" }),
    ];
    for extra in incomplete {
        let mut body = json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances"
        });
        let object = body.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        let (status, _) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // An unknown mode still 400s — the allow-list widened by exactly one entry.
    let (status, _) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_creature",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving",
            "referenceAssetIds": ["ref-character"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

fn write_scail2_reference_test_manifest(config_dir: &std::path::Path, multi_reference: bool) {
    std::fs::create_dir_all(config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "models": [{
                "id": "scail2_14b",
                "name": "SCAIL-2 14B",
                "family": "scail2",
                "type": "video",
                "capabilities": ["image_to_video"],
                "ui": { "scail2MultiReference": multi_reference }
            }]
        }))
        .expect("multi-reference manifest serializes"),
    )
    .expect("multi-reference manifest writes");
    write_empty_sibling_manifests(config_dir);
}

/// The paired SCAIL-2 provider has six source-position slots: the first character plus five
/// ordered extras. The API owns the public boundary, while the worker keeps the same backstop for
/// retry/legacy payloads; neither is permitted to silently retain only the first six. The source
/// implementation must also stay behind the resolved descriptor gate until the final inference
/// pin carries the paired layout. This pins that gate and the canonical empty-array payload shape
/// used by recipes, replay, retry, and duplicate.
#[tokio::test]
async fn scail2_animate_character_preserves_ordered_multi_references_and_rejects_seven() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SCAIL-2 multi-reference" }),
    )
    .await;

    let base = || {
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving"
        })
    };

    // 0 references without the legacy source fallback remains incomplete. The one-reference
    // path remains available while the current engine descriptor has not yet proven the paired
    // layout.
    let (status, _) = request(app.clone(), "POST", "/api/v1/video/jobs", base()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let mut padded_model = base();
    padded_model
        .as_object_mut()
        .unwrap()
        .insert("model".to_owned(), json!(" scail2_14b "));
    padded_model
        .as_object_mut()
        .unwrap()
        .insert("referenceAssetIds".to_owned(), json!(["character-primary"]));
    let (status, error) = request(app.clone(), "POST", "/api/v1/video/jobs", padded_model).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error["detail"].as_str().is_some_and(
            |detail| detail.contains("model must not contain leading or trailing whitespace")
        ),
        "submit must reject a padded model id rather than letting the worker trim it: {error}"
    );
    let mut padded_reference = base();
    padded_reference.as_object_mut().unwrap().insert(
        "referenceAssetIds".to_owned(),
        json!([" character-primary "]),
    );
    let (status, error) =
        request(app.clone(), "POST", "/api/v1/video/jobs", padded_reference).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("leading or trailing whitespace")),
        "submit must reject a padded reference id rather than letting the worker trim it: {error}"
    );
    let single_reference = vec!["character-0"];
    let mut single = base();
    single.as_object_mut().unwrap().insert(
        "referenceAssetIds".to_owned(),
        json!(single_reference.clone()),
    );
    let (status, job) = request(app.clone(), "POST", "/api/v1/video/jobs", single).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["payload"]["referenceAssetIds"], json!(single_reference));

    // A direct API caller is held to the same honest gate as the Studio. This prevents the
    // source-ready worker path from making today's older inference pin claim multi-reference
    // support before the descriptor flag is shipped with the final paired pin.
    for count in [2usize, 6] {
        let references: Vec<String> = (0..count)
            .map(|index| format!("character-{index}"))
            .collect();
        let mut body = base();
        body.as_object_mut()
            .unwrap()
            .insert("referenceAssetIds".to_owned(), json!(references));
        let (status, error) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            error
                .to_string()
                .contains("multi-reference is unavailable until the paired inference descriptor"),
            "{count} references must stay behind the current descriptor gate, got: {error}"
        );
    }

    // A sourceAssetId remains the old single-reference fallback. Whether referenceAssetIds was
    // absent or explicitly [], the stored/replayable shape must canonicalize to [] rather than
    // accidentally serializing null or dropping the field on a retry/duplicate.
    for reference_ids in [None, Some(json!([]))] {
        let mut body = base();
        let object = body.as_object_mut().unwrap();
        object.insert("sourceAssetId".to_owned(), json!("legacy-reference"));
        if let Some(reference_ids) = reference_ids {
            object.insert("referenceAssetIds".to_owned(), reference_ids);
        }
        let (status, job) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(job["payload"]["referenceAssetIds"], json!([]));
    }

    let mut seven = base();
    seven.as_object_mut().unwrap().insert(
        "referenceAssetIds".to_owned(),
        json!([
            "character-0",
            "character-1",
            "character-2",
            "character-3",
            "character-4",
            "character-5",
            "character-6"
        ]),
    );
    let (status, error) = request(app, "POST", "/api/v1/video/jobs", seven).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error.to_string().contains("at most 6 reference characters"),
        "seven references must be rejected explicitly, got: {error}"
    );

    // The exact same endpoint becomes source-ready when the resolved, server-owned manifest
    // carries the paired-descriptor gate. Use an isolated miniature catalog rather than changing
    // the real builtin manifest before inference-main and terminal evidence exist.
    let enabled_temp_dir = tempfile::tempdir().expect("enabled temp dir creates");
    let config_dir = enabled_temp_dir.path().join("config/manifests");
    write_scail2_reference_test_manifest(&config_dir, true);
    let enabled_app = create_app(test_settings(&enabled_temp_dir)).expect("enabled app creates");
    request(
        enabled_app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SCAIL-2 paired descriptor" }),
    )
    .await;
    for count in [2usize, 6] {
        let references: Vec<String> = (0..count)
            .map(|index| format!("character-{index}"))
            .collect();
        let mut body = base();
        body.as_object_mut()
            .unwrap()
            .insert("referenceAssetIds".to_owned(), json!(references.clone()));
        let (status, job) = request(enabled_app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{count} references must be accepted"
        );
        assert_eq!(
            job["payload"]["referenceAssetIds"],
            json!(references),
            "{count} references must stay ordered for the paired worker contract"
        );
    }
}

/// Retry and duplicate rebuild `modelManifestEntry` from current server state, so the exact merged
/// reference array must be re-authorized against that rebuilt entry before either operation creates
/// work. This protects both directions of the descriptor transition: a current-pin replay cannot
/// bypass the missing flag, while an enabled pin keeps every accepted id in caller order.
#[tokio::test]
async fn retry_and_duplicate_strictly_validate_scail2_multi_reference_replay() {
    let current_temp_dir = tempfile::tempdir().expect("current temp dir creates");
    let current_app = create_app(test_settings(&current_temp_dir)).expect("current app creates");
    request(
        current_app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SCAIL-2 current descriptor replay" }),
    )
    .await;
    let (status, current_original) = request(
        current_app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving",
            "referenceAssetIds": ["character-primary"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{current_original}");
    let current_job_id = current_original["id"].as_str().expect("current job id");

    for operation in ["retry", "duplicate"] {
        let (status, error) = request(
            current_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{current_job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "model": " scail2_14b ",
                    "referenceAssetIds": ["character-primary", "character-secondary"]
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert!(
            error["detail"]
                .as_str()
                .is_some_and(|detail| detail
                    .contains("model must not contain leading or trailing whitespace")),
            "{operation} must reject the padded-model capability bypass: {error}"
        );

        let (status, error) = request(
            current_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{current_job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "mode": " animate_character ",
                    "referenceAssetIds": ["character-primary", "character-secondary"]
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert!(
            error["detail"]
                .as_str()
                .is_some_and(|detail| detail
                    .contains("mode must not contain leading or trailing whitespace")),
            "{operation} must reject the padded-mode capability bypass: {error}"
        );

        let (status, error) = request(
            current_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{current_job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "referenceAssetIds": [" character-primary "]
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert!(
            error["detail"].as_str().is_some_and(|detail| detail
                .contains("referenceAssetIds must not contain leading or trailing whitespace")),
            "{operation} must reject a padded id rather than letting VideoRequest trim it: {error}"
        );

        let (status, error) = request(
            current_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{current_job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "referenceAssetIds": ["character-primary", "character-secondary"]
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert!(
            error["detail"].as_str().is_some_and(|detail| detail
                .contains("multi-reference is unavailable until the paired inference descriptor")),
            "{operation} must enforce the current-pin refusal: {error}"
        );

        let (status, error) = request(
            current_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{current_job_id}/{operation}"),
            json!({
                "payloadChanges": {
                    "referenceAssetIds": ["r0", "r1", "r2", "r3", "r4", "r5", "r6"]
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
        assert!(
            error["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("at most 6 reference characters")),
            "{operation} must reject seven before descriptor dispatch: {error}"
        );

        for (malformed, expected) in [
            (json!("not-an-array"), "must be an array of string ids"),
            (
                json!(["character-primary", 7]),
                "must contain only string ids",
            ),
            (
                json!(["character-primary", "  "]),
                "must not contain blank ids",
            ),
        ] {
            let (status, error) = request(
                current_app.clone(),
                "POST",
                &format!("/api/v1/jobs/{current_job_id}/{operation}"),
                json!({ "payloadChanges": { "referenceAssetIds": malformed } }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {error}");
            assert!(
                error["detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains(expected)),
                "{operation} must reject malformed referenceAssetIds ({expected}): {error}"
            );
        }
    }

    let enabled_temp_dir = tempfile::tempdir().expect("enabled temp dir creates");
    write_scail2_reference_test_manifest(&enabled_temp_dir.path().join("config/manifests"), true);
    let enabled_app = create_app(test_settings(&enabled_temp_dir)).expect("enabled app creates");
    request(
        enabled_app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "SCAIL-2 enabled descriptor replay" }),
    )
    .await;
    let (status, enabled_original) = request(
        enabled_app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "model": "scail2_14b",
            "mode": "animate_character",
            "prompt": "the character dances",
            "sourceClipAssetId": "clip-driving",
            "referenceAssetIds": ["character-primary"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{enabled_original}");
    let enabled_job_id = enabled_original["id"].as_str().expect("enabled job id");
    let ordered = json!([
        "character-4",
        "character-1",
        "character-5",
        "character-0",
        "character-3",
        "character-2"
    ]);
    for operation in ["retry", "duplicate"] {
        let (status, replayed) = request(
            enabled_app.clone(),
            "POST",
            &format!("/api/v1/jobs/{enabled_job_id}/{operation}"),
            json!({ "payloadChanges": { "referenceAssetIds": ordered.clone() } }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {replayed}");
        assert_eq!(
            replayed["payload"]["referenceAssetIds"], ordered,
            "{operation} must preserve the exact six-reference order"
        );
    }
}

#[tokio::test]
async fn person_tracking_routes_match_contracts() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Tracking Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
    std::fs::write(
        project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": project_id,
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {}
        }))
        .expect("json"),
    )
    .expect("track sidecar writes");

    let (status, tracks) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tracks[0]["id"], "track_1");
    assert_eq!(
        tracks[0]["path"],
        "person-tracks/track_1.sceneworks.person-track.json"
    );

    let (status, track) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/person-tracks/track_1"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(track["name"], "Hero");

    let (status, detection_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/detections"),
        json!({ "sourceAssetId": "asset-video", "sourceTimestamp": 1.25 }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(detection_job["type"], "person_detect");
    assert_eq!(detection_job["payload"]["sourceTimestamp"], 1.25);
    assert!(detection_job["projectName"]
        .as_str()
        .is_some_and(|value| value.starts_with("tracking")));

    let detection = json!({
        "id": "person_1",
        "box": { "x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6 }
    });
    let (status, track_job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/person-tracks/jobs"),
        json!({
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "detection": detection,
            "trackName": "Hero"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(track_job["type"], "person_track");
    assert_eq!(track_job["payload"]["trackName"], "Hero");

    for invalid_path in [
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E"),
        format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E%2Fescape"),
        format!("/api/v1/projects/{project_id}/person-tracks/track~bad"),
    ] {
        let (status, error) = request(app.clone(), "GET", &invalid_path, Value::Null).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid person track ID");
    }

    let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(queue["counts"]["queued"], 2);
}

#[tokio::test]
async fn generation_job_routes_reject_incompatible_loras() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                },
                {
                  "id": "ltx_2_3",
                  "name": "LTX",
                  "family": "ltx-video",
                  "type": "video",
                  "adapter": "ltx_video",
                  "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["ltx-video"] },
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                }
              ]
            }
            "#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "bad_qwen",
                  "name": "Bad Qwen",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "loras": [{ "id": "qwen_style" }]
                }
              ]
            }
            "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("qwen.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Compatibility" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (status, image_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        image_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    let (status, unknown_model_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "missing_model",
            "loras": [{ "id": "qwen_style" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        unknown_model_error["detail"],
        "Model missing_model not found; cannot verify LoRA compatibility"
    );

    let (status, preset_error) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "prompt": "mist",
            "model": "z_image_turbo",
            "recipePresetId": "bad_qwen"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        preset_error["detail"],
        "LoRA qwen_style is not compatible with model z_image_turbo"
    );

    for (mode, extra) in [
        ("image_to_video", json!({ "sourceAssetId": "asset-image" })),
        ("text_to_video", json!({})),
        (
            "first_last_frame",
            json!({ "sourceAssetId": "asset-first", "lastFrameAssetId": "asset-last" }),
        ),
        ("extend_clip", json!({ "sourceClipAssetId": "asset-video" })),
        (
            "video_bridge",
            json!({ "sourceClipAssetId": "asset-left", "bridgeRightClipAssetId": "asset-right" }),
        ),
        (
            "replace_person",
            json!({ "sourceClipAssetId": "asset-video", "personTrackId": "track-1", "characterId": "character-1" }),
        ),
    ] {
        let mut payload = json!({
            "projectId": project_id,
            "mode": mode,
            "prompt": "motion",
            "model": "ltx_2_3",
            "loras": [{ "id": "qwen_style" }]
        });
        payload
            .as_object_mut()
            .expect("video payload object")
            .extend(extra.as_object().expect("extra payload object").clone());
        let (status, video_error) =
            request(app.clone(), "POST", "/api/v1/video/jobs", payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mode}");
        assert_eq!(
            video_error["detail"],
            "LoRA qwen_style is not compatible with model ltx_2_3"
        );
    }

    let (_, character) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters"),
        json!({ "name": "Mira", "type": "person" }),
    )
    .await;
    let character_id = character["id"].as_str().expect("character id");
    let character_lora = temp_dir
        .path()
        .join("data/loras/character-qwen.safetensors");
    write_test_safetensors(&character_lora);
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
        json!({
            "name": "Character Qwen",
            "sourcePath": character_lora.display().to_string(),
            "compatibility": { "families": ["qwen-image"] }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, character_error) = request(
        app,
        "POST",
        &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
        json!({ "prompt": "portrait", "model": "z_image_turbo" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(character_error["detail"]
        .as_str()
        .unwrap()
        .contains("is not compatible with model z_image_turbo"));
}

/// The seeded video model carries a REAL routed id (`wan_2_2`, sc-19504) rather than the old
/// synthetic `vid-model`: the enqueue no-lane gate refuses a model no lane can claim, so a
/// synthetic id now 400s before preset expansion can be observed. The entry is still the fixture's
/// own — its defaults, limits and repo are written here — so nothing this test pins has changed.
#[tokio::test]
async fn video_jobs_expand_recipe_presets_server_side() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "wan_2_2",
              "name": "Vid Model",
              "family": "wan-video",
              "type": "video",
              "adapter": "wan_video",
              "capabilities": ["text_to_video", "image_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["wan-video"] },
              "ui": { "label": "Vid" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "motion-lora",
              "name": "Motion LoRA",
              "family": "wan-video",
              "triggerWords": ["motion"],
              "compatibility": { "families": ["wan-video"] },
              "source": { "provider": "local", "path": "loras/motion.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_motion",
              "name": "Dream Motion",
              "workflow": "text_to_video",
              "model": "wan_2_2",
              "defaults": { "duration": 8, "fps": 30, "resolution": "1280x720", "quality": "best", "negativePrompt": "jitter" },
              "prompt": { "prefix": "cinematic", "suffix": "smooth camera motion" },
              "loras": [{ "id": "motion-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("motion.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Video Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "wan_2_2",
            // Client render settings that DIFFER from the preset's declared
            // defaults — the studio seeds the form from the preset but the user
            // is free to override, so these submitted values must win.
            "duration": 10,
            "fps": 24,
            "width": 640,
            "height": 640,
            "quality": "fast",
            "negativePrompt": "client jitter",
            "recipePresetId": "dream_motion"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // Prompt prefix/suffix are folded in server-side around the raw client
    // prompt — the regression that motivated this path.
    assert_eq!(
        video_job["payload"]["prompt"],
        "cinematic, a fox runs, smooth camera motion"
    );
    // Render settings are client-owned and overrideable: the submitted values
    // win over the preset's declared defaults (8 / 30 / 1280x720 / best / jitter).
    assert_eq!(video_job["payload"]["duration"], 10);
    assert_eq!(video_job["payload"]["fps"], 24);
    assert_eq!(video_job["payload"]["width"], 640);
    assert_eq!(video_job["payload"]["height"], 640);
    assert_eq!(video_job["payload"]["quality"], "fast");
    assert_eq!(video_job["payload"]["negativePrompt"], "client jitter");
    // Preset LoRA merged in (client sent none) and stamped under advanced.
    assert_eq!(video_job["payload"]["loras"][0]["id"], "motion-lora");
    assert_eq!(
        video_job["payload"]["advanced"]["recipePresetId"],
        "dream_motion"
    );

    // sc-10520: submitting the job stamped lastUsedAt into the usage side store, and it
    // surfaces on the catalog read even though dream_motion is a read-only BUILTIN preset
    // (its own manifest can't be rewritten). The store lives beside the manifests.
    assert!(
        config_dir.join("recipe-preset-usage.json").is_file(),
        "job submit should create the recipe-preset usage store"
    );
    let (status, presets) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/recipe-presets?projectId={project_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let dream = presets
        .as_array()
        .expect("presets list")
        .iter()
        .find(|preset| preset["id"] == "dream_motion")
        .expect("dream_motion present");
    assert_eq!(dream["scope"], "builtin");
    assert!(
        dream["lastUsedAt"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "builtin preset should carry lastUsedAt after use, got {:?}",
        dream["lastUsedAt"]
    );
}

#[tokio::test]
async fn preset_overridden_video_model_carries_its_own_manifest_entry() {
    // sc-12300: a client that OMITS `model` (MCP's submit_video_job documents its `model`
    // param as "Omit for the server default") gets `default_video_model()` from serde, which
    // is exactly the gate `apply_recipe_preset_to_video_payload` uses to let the preset's
    // model win. The preset then overwrites job_payload["model"] — but the manifest entry
    // used to be resolved from the DTO's pre-override `payload.model`, so the job was
    // enqueued carrying the OVERRIDDEN model id alongside the DEFAULT model's entry.
    //
    // Both halves of that mismatch are asserted, because they fail in opposite ways:
    //   - `repo`   — the LOUD failure: the worker reaches for the wrong model's weights.
    //   - `limits.requiresDimensionsMultipleOf` — the SILENT one: sceneworks-core's
    //     normalized_dimensions honors this (sc-11993). A 16-multiple model handed a
    //     32-declaring entry silently renders off-bucket (Mochi's native 848x480 -> 832x480).
    // Pinning only `repo` would let the silent geometry bug regress undetected.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // `default-vid` is BUILT FROM default_video_model() rather than hardcoding its id. That
    // literal is what the preset-override gate compares against, so a hardcoded fixture would
    // quietly stop modelling *the default's* entry if the default ever changed: the pre-fix
    // failure would degrade from "carries the DEFAULT's entry" to "carries {}" — still red,
    // but no longer demonstrating the documented defect. Its 32 / owner/default mirror the
    // real default video manifest; the override entry mirrors the real mochi_1's 16 / distinct
    // repo. Both fixture ids are REAL routed video models (sc-19504): the enqueue no-lane gate
    // refuses a model no lane can claim, so a synthetic id would now 400 before this assertion.
    // Each keeps its own seeded limits, so what the test pins is unchanged.
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 32 },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "mochi_1",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/mochi_1", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 16 },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "mochi_1",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Preset Override Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, video_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            // `model` deliberately OMITTED — this is the trigger. Sending it explicitly
            // closes the gate and the bug never fires.
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The preset's model won over the omitted-model default...
    assert_eq!(video_job["payload"]["model"], "mochi_1");
    // ...and the entry travelling with it must describe THAT model, not the default's.
    let entry = &video_job["payload"]["modelManifestEntry"];
    assert_eq!(
        entry["id"], "mochi_1",
        "manifest entry should be resolved from the post-override model"
    );
    assert_eq!(
        entry["downloads"][0]["repo"], "owner/mochi_1",
        "wrong repo => the worker fetches the wrong model's weights (loud failure)"
    );
    assert_eq!(
        entry["limits"]["requiresDimensionsMultipleOf"], 16,
        "wrong dimension floor => silently renders off-bucket geometry (sc-11993)"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn mac_only_video_is_rejected_before_enqueue_on_direct_preset_and_replay_routes() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Base LTX",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [{ "provider": "huggingface", "repo": "owner/base", "files": ["*.safetensors"], "default": true }],
              "paths": {}, "defaults": {}, "limits": {}, "ui": { "label": "Base LTX" }
            },
            {
              "id": "ltx_2_3_eros",
              "name": "LTX Eros",
              "family": "ltx-video",
              "type": "video",
              "macOnly": true,
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [{ "provider": "huggingface", "repo": "owner/eros", "files": ["*.safetensors"], "default": true }],
              "paths": {}, "defaults": {}, "limits": {}, "ui": { "label": "LTX Eros" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{
          "schemaVersion": 1,
          "models": [{
            "id": "ltx_2_3_eros",
            "macOnly": false,
            "downloadable": true,
            "usable": true
          }]
        }"#,
    )
    .expect("user override attempt writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [{
            "id": "eros_preset",
            "name": "Eros Preset",
            "workflow": "text_to_video",
            "model": "ltx_2_3_eros",
            "defaults": {},
            "prompt": { "prefix": "cinematic" }
          }]
        }
        "#,
    )
    .expect("builtin presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user presets writes");

    let (app, state) =
        create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
    *state.video_platform_override.lock() = Some("windows");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Platform Withdrawal Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    for payload in [
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox",
            "model": "ltx_2_3_eros"
        }),
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox",
            "recipePresetId": "eros_preset"
        }),
    ] {
        let (status, body) = request(app.clone(), "POST", "/api/v1/video/jobs", payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("only on macOS"));
    }
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().expect("jobs array").is_empty());

    // Base LTX stays cross-platform, while the same Eros route remains valid on macOS.
    let (status, base_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox",
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{base_job}");
    *state.video_platform_override.lock() = Some("macos");
    let (status, eros_job) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox",
            "model": "ltx_2_3_eros"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{eros_job}");

    // A legacy Eros job and a model-changing replay are both rejected before a new row is queued.
    *state.video_platform_override.lock() = Some("linux");
    for (job_id, payload_changes) in [
        (eros_job["id"].as_str().unwrap(), json!({})),
        (
            base_job["id"].as_str().unwrap(),
            json!({ "model": "ltx_2_3_eros" }),
        ),
    ] {
        for operation in ["retry", "duplicate"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{job_id}/{operation}"),
                json!({ "payloadChanges": payload_changes }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {body}");
            assert!(body["detail"]
                .as_str()
                .unwrap_or_default()
                .contains("only on macOS"));
        }
    }
    let (_, jobs) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs.as_array().expect("jobs array").len(),
        2,
        "only the explicitly valid base and macOS Eros jobs may exist"
    );
}

#[tokio::test]
async fn preset_overridden_image_model_carries_its_own_manifest_entry() {
    // sc-12300: create_image_job has the identical ordering shape as create_video_job —
    // apply_recipe_preset_to_image_payload may overwrite job_payload["model"] (gated on the
    // omitted-model default_image_model()), and the entry was likewise resolved from the
    // pre-override DTO. Same defect, same function family, so it is pinned the same way.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // Built FROM default_image_model() — the id the override gate compares against — for the
    // same durability reason as the video fixture above.
    let default_image_model = crate::defaults::default_image_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_IMAGE_MODEL__",
              "name": "Default Img",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 32 },
              "ui": { "label": "Default Img" }
            },
            {
              "id": "preset-img",
              "name": "Preset Img",
              "family": "flux",
              "type": "image",
              "adapter": "flux",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/preset-img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "requiresDimensionsMultipleOf": 16 },
              "ui": { "label": "Preset Img" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_IMAGE_MODEL__", &default_image_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "img_override",
              "name": "Img Override",
              "workflow": "text_to_image",
              "model": "preset-img",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Img Override Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox runs",
            // `model` deliberately OMITTED — the trigger.
            "recipePresetId": "img_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(image_job["payload"]["model"], "preset-img");
    let entry = &image_job["payload"]["modelManifestEntry"];
    assert_eq!(
        entry["id"], "preset-img",
        "manifest entry should be resolved from the post-override model"
    );
    assert_eq!(
        entry["downloads"][0]["repo"], "owner/preset-img",
        "wrong repo => the worker fetches the wrong model's weights (loud failure)"
    );
    assert_eq!(
        entry["limits"]["requiresDimensionsMultipleOf"], 16,
        "wrong dimension floor => silently renders off-bucket geometry"
    );
}

#[tokio::test]
async fn preset_image_job_builds_each_catalog_once() {
    // sc-8819 (F-017): a preset-backed POST /image/jobs fans out into recipe_preset_catalog,
    // merge_preset_loras_into_payload, and validate_job_lora_compatibility. Before the fix
    // each re-assembled model_catalog/lora_catalog from scratch, re-running the whole
    // per-model/per-LoRA filesystem install-state probe sweep 2× each. The request-scoped
    // JobCatalogSnapshot threads one snapshot through those seams so each catalog is built
    // exactly once per job-create — assert that here.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024", "negativePrompt": "blur" },
              "prompt": { "prefix": "cinematic", "suffix": "high detail" },
              "loras": [{ "id": "style-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Image Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Reset the probe counters immediately before the job-create so project setup /
    // seeding above doesn't count against it.
    crate::test_reset_catalog_build_counters();
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "job created: {image_job:?}");

    // Correctness preserved: preset prompt folded in, preset LoRA merged, and it validated.
    assert_eq!(
        image_job["payload"]["prompt"],
        "cinematic, a fox, high detail"
    );
    assert_eq!(image_job["payload"]["loras"][0]["id"], "style-lora");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );

    // The heart of sc-8819: each catalog's full FS-probe assembly ran exactly once for the
    // whole request, not 2–3× as before the snapshot was threaded through.
    assert_eq!(
        crate::test_model_catalog_builds(),
        1,
        "model catalog should be assembled once per preset job-create"
    );
    assert_eq!(
        crate::test_lora_catalog_builds(),
        1,
        "lora catalog should be assembled once per preset job-create"
    );
}

#[tokio::test]
async fn preset_image_job_skips_server_lora_merge_when_client_resolved() {
    // The web studio seeds a selected preset's LoRAs into the visible picker and sends them in
    // `loras`, so it — not the server — is authoritative for which preset LoRAs apply. When it
    // also sends presetLorasResolvedClientSide, the server must NOT re-merge the preset's LoRAs:
    // that is what lets a user REMOVE a preset LoRA (send it absent) and have the removal stick,
    // instead of the server silently adding it back. Headless clients that omit the flag keep the
    // server-side merge (covered by preset_image_job_builds_each_catalog_once).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024" },
              "prompt": { "prefix": "cinematic" },
              "loras": [{ "id": "style-lora", "weight": 0.5 }]
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");
    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    write_test_safetensors(&lora_dir.join("style.safetensors"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Resolved Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // The client selected the preset (recipePresetId) but removed its only LoRA in the picker, so
    // it sends an empty `loras` plus the client-resolved flag.
    let (status, image_job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style",
            "presetLorasResolvedClientSide": true,
            "loras": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "job created: {image_job:?}");

    // The preset prompt is still folded in and the preset id is stamped, but the server left the
    // client's (empty) `loras` untouched — style-lora was NOT re-added.
    assert_eq!(image_job["payload"]["prompt"], "cinematic, a fox");
    assert_eq!(
        image_job["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );
    assert_eq!(
        image_job["payload"]["loras"],
        json!([]),
        "client-resolved preset LoRAs must not be re-merged by the server"
    );
}

#[tokio::test]
async fn preset_image_job_skips_server_prompt_fold_when_client_resolved() {
    // General presets stack (epic 11949): the studio composes the full preset-stack prompt
    // client-side because the server only knows how to fold ONE recipePresetId's prefix/suffix.
    // When the studio sends presetPromptResolvedClientSide, the server must take `prompt` verbatim
    // and NOT re-apply preset_prompt — otherwise the base preset's prefix would be folded twice.
    // Headless clients that omit the flag keep the server-side fold (asserted below as baseline).
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "dream_style",
              "name": "Dream Style",
              "workflow": "text_to_image",
              "model": "img-model",
              "defaults": { "count": 1, "resolution": "1024x1024" },
              "prompt": { "prefix": "cinematic" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Prompt Preset Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Baseline: no flag → server folds the preset prefix into the prompt (today's behavior).
    let (status, folded) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "baseline job created: {folded:?}"
    );
    assert_eq!(folded["payload"]["prompt"], "cinematic, a fox");

    // Client-authoritative: the studio already composed "cinematic, a fox" and sends the flag,
    // so the server must NOT fold the prefix again (would yield "cinematic, cinematic, a fox").
    let (status, verbatim) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "cinematic, a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "recipePresetId": "dream_style",
            "presetPromptResolvedClientSide": true
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "client-resolved job created: {verbatim:?}"
    );
    assert_eq!(
        verbatim["payload"]["prompt"], "cinematic, a fox",
        "client-resolved prompt must be taken verbatim, not re-folded"
    );
    // The preset id is still stamped for usage tracking.
    assert_eq!(
        verbatim["payload"]["advanced"]["recipePresetId"],
        "dream_style"
    );
}

#[tokio::test]
async fn generation_routes_reject_invalid_payloads() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({ "projectId": "project-1", "prompt": "x".repeat(4001) }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = request(
        app,
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": "project-1",
            "mode": "image_to_video",
            "prompt": "missing source image"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn frame_extract_rejects_non_finite_playhead() {
    let result = crate::validate_frame_extract(&crate::FrameExtractRequest {
        playhead_seconds: f64::NAN,
        intended_use: "reuse".to_owned(),
        requested_gpu: "auto".to_owned(),
    });

    assert!(result.is_err());
}

#[test]
fn image_dimension_cap_covers_sensenova_buckets() {
    // Raised so SenseNova-U1's true trained buckets (largest 3456) pass.
    assert_eq!(crate::MAX_IMAGE_DIMENSION, 4096);
    assert!(crate::validate_dimension(2720, "width", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(3456, "height", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(4096, "width", crate::MAX_IMAGE_DIMENSION).is_ok());
    assert!(crate::validate_dimension(4097, "width", crate::MAX_IMAGE_DIMENSION).is_err());
    assert!(crate::validate_dimension(255, "width", crate::MAX_IMAGE_DIMENSION).is_err());
}

#[test]
fn vqa_job_validation_requires_question_and_asset() {
    let base = crate::VqaJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        source_asset_id: "asset_1".to_owned(),
        question: "What is in this image?".to_owned(),
        model: "sensenova_u1_8b".to_owned(),
        max_new_tokens: 256,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(crate::validate_vqa_job(&base).is_ok());

    // The UI's length presets are all valid.
    for tokens in [256u32, 512, 1024] {
        let mut request = base.clone();
        request.max_new_tokens = tokens;
        assert!(crate::validate_vqa_job(&request).is_ok());
    }

    let mut blank_question = base.clone();
    blank_question.question = "   ".to_owned();
    assert!(crate::validate_vqa_job(&blank_question).is_err());

    let mut missing_asset = base.clone();
    missing_asset.source_asset_id = String::new();
    assert!(crate::validate_vqa_job(&missing_asset).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(crate::validate_vqa_job(&missing_project).is_err());

    let mut too_many_tokens = base.clone();
    too_many_tokens.max_new_tokens = 4096;
    assert!(crate::validate_vqa_job(&too_many_tokens).is_err());
}

#[test]
fn interleave_job_validation_bounds_prompt_images_and_assets() {
    let base = crate::InterleaveJobRequest {
        project_id: "project-1".to_owned(),
        project_name: None,
        prompt: "A short illustrated guide to brewing tea".to_owned(),
        source_asset_ids: Vec::new(),
        model: "sensenova_u1_8b".to_owned(),
        max_images: 6,
        width: 1024,
        height: 1024,
        seed: None,
        requested_gpu: "auto".to_owned(),
        advanced: serde_json::Map::new(),
    };
    assert!(crate::validate_interleave_job(&base).is_ok());

    // Optional input images (it2i) are allowed.
    let mut with_sources = base.clone();
    with_sources.source_asset_ids = vec!["asset_1".to_owned(), "asset_2".to_owned()];
    assert!(crate::validate_interleave_job(&with_sources).is_ok());

    let mut blank_prompt = base.clone();
    blank_prompt.prompt = "   ".to_owned();
    assert!(crate::validate_interleave_job(&blank_prompt).is_err());

    let mut missing_project = base.clone();
    missing_project.project_id = String::new();
    assert!(crate::validate_interleave_job(&missing_project).is_err());

    let mut zero_images = base.clone();
    zero_images.max_images = 0;
    assert!(crate::validate_interleave_job(&zero_images).is_err());

    let mut too_many_images = base.clone();
    too_many_images.max_images = 11;
    assert!(crate::validate_interleave_job(&too_many_images).is_err());

    let mut blank_asset = base.clone();
    blank_asset.source_asset_ids = vec!["  ".to_owned()];
    assert!(crate::validate_interleave_job(&blank_asset).is_err());

    let mut tiny = base.clone();
    tiny.width = 64;
    assert!(crate::validate_interleave_job(&tiny).is_err());
}

#[tokio::test]
async fn worker_heartbeat_interrupts_previous_active_job_through_http() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    let job_id = created["id"].as_str().expect("job id is string");
    // The owning worker reports at least one heartbeat for the job, so a
    // later idle heartbeat is a genuine restart (not a claim race) and must
    // reclaim the now-orphaned active job.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "busy", "currentJobId": job_id, "loadedModels": [] }),
    )
    .await;

    let (status, worker) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(worker["currentJobId"], Value::Null);

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
    assert_eq!(job["workerId"], Value::Null);
}

#[tokio::test]
async fn worker_heartbeat_broadcasts_job_updated_for_the_job_it_interrupts() {
    // sc-18182: sc-8186 gave the time-based stale sweep a `job.updated` broadcast, but the
    // OTHER path that terminates a job behind the client's back — the idle heartbeat of a
    // RESTARTED worker reclaiming its previous incarnation's job — kept publishing only
    // `worker.updated`. The web client is SSE-driven and does not poll jobs, so that job went
    // terminal in the DB while the studio showed "generating" forever. This is the exact
    // sequence a Mac MLX worker produces after being OOM-killed mid-render.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;
    // Establish the job's first heartbeat, so the idle heartbeat below is treated as a
    // genuine restart rather than a claim race.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "busy", "currentJobId": job_id, "loadedModels": [] }),
    )
    .await;

    // Subscribe only now, so we observe the restart heartbeat's events and nothing before it.
    let mut events = state.events.subscribe();
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let restart_events = drain_event_names(&mut events).await;
    assert!(
        restart_events.iter().any(|name| name == "job.updated"),
        "a restart heartbeat must broadcast job.updated for the job it interrupts, \
         or an SSE-only client never leaves the generating state: {restart_events:?}"
    );
    assert!(
        restart_events.iter().any(|name| name == "queue.updated"),
        "the queue must be refreshed too, so the job leaves the active queue view: \
         {restart_events:?}"
    );

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
}

#[tokio::test]
async fn worker_heartbeat_without_an_interrupt_broadcasts_no_job_event() {
    // Guard the other side of sc-18182: the ordinary busy heartbeat that every worker sends
    // every few seconds must NOT publish job.updated/queue.updated. Getting this wrong would
    // put the whole queue on the SSE bus at heartbeat cadence.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, state) = create_app_with_state(test_settings(&temp_dir)).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    let mut events = state.events.subscribe();
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/worker-1/heartbeat",
        json!({ "status": "busy", "currentJobId": job_id, "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let heartbeat_events = drain_event_names(&mut events).await;
    assert!(
        !heartbeat_events.iter().any(|name| name == "job.updated"),
        "a routine busy heartbeat interrupts nothing and must not publish job.updated: \
         {heartbeat_events:?}"
    );
}

#[tokio::test]
async fn stale_sweep_broadcasts_job_updated_for_interrupted_jobs() {
    // sc-8186: the heartbeat stale-sweep marks an in-flight job `interrupted` in the DB, but
    // (unlike a worker-reported terminal status) emitted no per-job event — so a live client's
    // job card, driven by `job.updated`, showed its last running state forever. The sweep must
    // now broadcast `job.updated` for each job it interrupts.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    // Smallest timeout the store honors (clamped to >=1s); we sleep just past it to go stale.
    settings.worker_timeout_seconds = 1;
    let (app, state) = create_app_with_state(settings).expect("app creates");

    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    // Let the worker's last_seen age past the (1s) timeout so the next sweep interrupts its job,
    // then subscribe so we only observe the sweep's events. last_seen is stored at second
    // granularity and the cutoff is `now - 1s`, so we sleep just over 2s to clear the boundary.
    tokio::time::sleep(Duration::from_millis(2_100)).await;
    let mut events = state.events.subscribe();

    // Any endpoint that runs `queue_summary_snapshot` triggers the sweep; GET /queue is the
    // canonical one.
    let (status, _) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
    assert_eq!(status, StatusCode::OK);

    let sweep_events = drain_event_names(&mut events).await;
    assert!(
        sweep_events.iter().any(|name| name == "job.updated"),
        "the stale-sweep must broadcast job.updated for the interrupted job: {sweep_events:?}"
    );

    let (status, job) = request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
}

#[tokio::test]
async fn claim_sweeps_stale_jobs_once_and_still_refreshes_the_queue() {
    // sc-8889 / F-087: claim_job runs mark_stale_workers_interrupted in its own
    // transaction, then refreshes the queue via publish_queue_skip_sweep — which
    // no longer sweeps a SECOND time. This pins that dropping the duplicate sweep
    // did not regress the claim path: a claim still reaps a stale worker's job to
    // `interrupted` and still emits queue.updated.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.worker_timeout_seconds = 1;
    let (app, state) = create_app_with_state(settings).expect("app creates");

    // worker-1 claims a job, then goes stale.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-1",
            "gpuId": "gpu-0",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    let stale_job_id = created["id"].as_str().expect("job id is string").to_owned();
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-1" }),
    )
    .await;

    // A fresh worker registers plus a second queued job so worker-2's claim
    // actually returns work (response.is_some -> the queue refresh fires). Age
    // worker-1 past the 1s timeout so the next claim's sweep reaps its job.
    request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-2",
            "gpuId": "gpu-1",
            "gpuName": null,
            "capabilities": ["image_detail"],
            "loadedModels": []
        }),
    )
    .await;
    request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({ "type": "image_detail", "payload": {}, "requestedGpu": "auto" }),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(2_100)).await;
    let mut events = state.events.subscribe();

    // worker-2 claims the second job. claim_job runs its own stale sweep
    // (interrupting worker-1's job) and refreshes the queue via the skip-sweep
    // path — without sweeping a second time.
    let (status, claim) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-2" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        claim["job"].is_object(),
        "worker-2 claims the second queued job: {claim}"
    );

    let claim_events = drain_event_names(&mut events).await;
    assert!(
        claim_events.iter().any(|name| name == "queue.updated"),
        "a claim that returns work still refreshes the queue: {claim_events:?}"
    );

    // The stale job was reaped exactly by the claim's own single sweep.
    let (status, job) = request(
        app,
        "GET",
        &format!("/api/v1/jobs/{stale_job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(job["status"], "interrupted");
    assert_eq!(job["workerId"], Value::Null);
}

/// sc-12297: `limits.hardMaxDuration` is enforced at enqueue, and — the part that makes WHERE it
/// lives load-bearing — against the POST-PRESET model's cap.
///
/// The fixture is built so the two plausible homes for this check disagree:
///   * default video model — cap 15 (generous)
///   * `mochi_1`             — cap  5 (strict)
///
/// The request omits `model` (so the preset's model wins, per sc-12300) and asks for 10s. Gating
/// on the DTO's `payload.model` — i.e. inside `validate_video_job`, the intuitive home, which runs
/// BEFORE `apply_recipe_preset_to_video_payload` — reads the DEFAULT's cap of 15, admits 10s, and
/// enqueues a job the strict model can't render. Only a gate placed after preset expansion AND
/// manifest resolution sees the 5 that actually applies. That is the whole reason this check is not
/// in `validate_video_job`, and this test is what pins it there.
#[tokio::test]
async fn video_duration_past_the_post_preset_models_hard_cap_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "hardMaxDuration": 15 },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "mochi_1",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/mochi_1", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": { "hardMaxDuration": 5 },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "mochi_1",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Duration Cap Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // 10s: legal for the default (15) but past the preset-resolved model's 5. `model` omitted so
    // the preset's model wins — the sc-12300 shape.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 10,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "10s past mochi_1's 5s cap must be refused at enqueue, not silently clamped: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("mochi_1"),
        "names the model whose cap applied — NOT the default's: {detail}"
    );
    assert!(detail.contains("5s"), "states the cap: {detail}");
    assert!(detail.contains("10s"), "states what was asked: {detail}");

    // At-cap admits: 5s is exactly the cap, and the bound is `>`. This is what keeps the
    // assertion above from passing for a gate that simply rejects everything.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 5,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "5s is at the cap: {body}");
    assert_eq!(body["payload"]["model"], "mochi_1");

    // ...and the SAME 10s request against the default model (cap 15) is admitted, proving the
    // rejection above came from the per-model cap rather than a blanket duration bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "duration": 10,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "10s is within the default model's 15s cap: {body}"
    );
}

/// sc-12347: `limits.fps` is enforced at enqueue against the POST-PRESET model's menu, and an
/// OMITTED fps resolves to that model's declared `defaults.fps` rather than a blanket.
///
/// The fixture makes both halves load-bearing by having the two models' menus disagree:
///   * default video model — `[24, 25, 30]`, default 25 (permissive)
///   * `mochi_1`             — `[30]`, default 30 (strict)
///
/// `fps: 25` is the discriminator. It is on the default's menu and off `mochi_1`'s, so a gate
/// reading the DTO's stale `payload.model` — i.e. inside `validate_video_job`, before
/// `apply_recipe_preset_to_video_payload` — admits it and enqueues a job the strict model does not
/// advertise. 25 is also the blanket the DTO used to default to, which is why the omitted-fps case
/// below is the regression this story nearly shipped rather than a nicety.
#[tokio::test]
async fn video_fps_outside_the_post_preset_models_menu_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "fps": 25 },
              "limits": { "fps": [24, 25, 30] },
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "mochi_1",
              "name": "Preset Vid",
              "family": "mochi",
              "type": "video",
              "adapter": "mochi_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/mochi_1", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "fps": 30 },
              "limits": { "fps": [30] },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "mochi_1",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Fps Menu Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // 25fps: on the DEFAULT's menu, off the preset-resolved model's. `model` omitted so the preset's
    // model wins — the sc-12300 shape. A gate reading the stale DTO model admits this.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 25,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "25fps is off mochi_1's [30] menu and must be refused at enqueue, not snapped: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("mochi_1"),
        "names the model whose menu applied — NOT the default's: {detail}"
    );
    assert!(
        detail.contains("30 fps"),
        "states what is allowed: {detail}"
    );
    assert!(detail.contains("25 fps"), "states what was asked: {detail}");

    // THE REGRESSION THIS STORY NEARLY SHIPPED: omitting fps must be ADMITTED, and must resolve to
    // the post-preset model's declared 30 — not the blanket 25 (which the assertion above proves is
    // rejected), and not the DEFAULT model's 25. Both wrong answers are 25, so this pins the
    // resolution AND that it is keyed off the post-preset model.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a request naming no fps must not be refused by the menu: {body}"
    );
    assert_eq!(body["payload"]["model"], "mochi_1");
    assert_eq!(
        body["payload"]["fps"], 30,
        "the enqueued payload records the model's declared rate, not the blanket 25: {body}"
    );

    // An advertised rate admits — keeps the rejection above from passing for a gate that refuses
    // everything.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 30,
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "30 is what mochi_1 advertises: {body}"
    );
    assert_eq!(body["payload"]["fps"], 30);

    // ...and the SAME 25fps request against the default model IS admitted, proving the rejection
    // came from the per-model menu rather than a blanket fps bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 25,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "25 is on the default model's [24, 25, 30] menu: {body}"
    );
    assert_eq!(body["payload"]["fps"], 25);

    // The blanket payload-sanity bound still applies to a NAMED fps, and still comes from
    // `validate_video_job` — a different message than the per-model menu's.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "fps": 240,
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "240 is past the sanity bound: {body}"
    );
    assert!(body["detail"]
        .as_str()
        .unwrap_or_default()
        .contains("fps must be between 1 and 60"));
}

/// sc-19426: `limits.hardMinSteps` is enforced at enqueue against the POST-PRESET model's floor,
/// and — the part that makes the key worth existing — a below-floor request is REFUSED rather than
/// raised onto the floor.
///
/// The fixture makes the two plausible homes disagree, exactly as the duration and fps tests above
/// do:
///   * default video model — no floor at all (1 step is fine)
///   * `mochi_1`             — floor 2, the MiniMax-H3 shape
///
/// `advanced.steps: 1` is the discriminator. A gate reading the DTO's stale `payload.model` — i.e.
/// inside `validate_video_job`, which runs BEFORE `apply_recipe_preset_to_video_payload` — sees the
/// default's absent floor, admits it, and enqueues a job whose scheduler cannot build a schedule
/// from a single sigma grid point.
#[tokio::test]
async fn video_steps_under_the_post_preset_models_hard_floor_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "steps": 8 },
              "limits": {},
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "mochi_1",
              "name": "Preset Vid",
              "family": "minimax-h3",
              "type": "video",
              "adapter": "minimax_h3",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/mochi_1", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "steps": 50 },
              "limits": { "hardMinSteps": 2 },
              "ui": { "label": "Preset Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "mochi_1",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Step Floor Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // 1 step: fine for the default (no floor) but under the preset-resolved model's 2. `model`
    // omitted so the preset's model wins — the sc-12300 shape.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 1 },
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "1 step under mochi_1's 2-step floor must be refused at enqueue, not raised: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("mochi_1"),
        "names the model whose floor applied — NOT the default's: {detail}"
    );
    assert!(
        detail.contains("at least 2 sampling steps"),
        "states the floor: {detail}"
    );
    assert!(
        detail.contains("asks for 1."),
        "states what was asked: {detail}"
    );

    // At-floor admits: 2 is exactly the floor, and the bound is `<`. This is what keeps the
    // assertion above from passing for a gate that simply rejects everything.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 2 },
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "2 is at the floor: {body}");
    assert_eq!(body["payload"]["model"], "mochi_1");
    assert_eq!(
        body["payload"]["advanced"]["steps"], 2,
        "the admitted count travels VERBATIM — the gate refuses, it never rewrites"
    );

    // A request naming NO steps is admitted even on the floored model: `advanced` is a passthrough
    // map with no blanket step count, so an omitted `steps` means the engine picks. Refusing it
    // would be the sc-12400 regression on a new axis.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a steps-less payload must not be refused by the floor: {body}"
    );

    // ...and the SAME 1-step request against the default model (no floor) is admitted, proving the
    // rejection above came from the per-model floor rather than a blanket step bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 1 },
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the default model declares no floor, so 1 step is its own business: {body}"
    );
}

/// sc-19502: `limits.steps` — the EXACT-value menu — is enforced at enqueue, and the half a floor
/// could never express is enforced too: an OVER-menu count is refused.
///
/// This is the reachability half of the story. `limits.steps: [8]` is only a real constraint if a
/// real HTTP request trips it, and `advanced.steps: 30` is the exact payload the story names: it
/// used to be accepted here, then 400 late from the candle engine after dispatch, or — on mlx — be
/// accepted and silently rendered at 8, a control that visibly did nothing.
///
/// The fixture makes the two plausible homes disagree the same way the floor test above does:
///   * default video model — no menu at all (30 steps is its own business)
///   * `mochi_1`           — menu `[8]`, the distilled shape
///
/// Both ids are REAL routed video models, and `mochi_1` is the same id the floor test above uses.
/// The menu model cannot be a made-up id: the sc-19504 no-lane gate runs last on the enqueue path
/// and refuses any video request no backend's claim predicate accepts, so a synthetic id 400s for
/// "no backend implements it" and the ON-menu `201` arm below — the one that keeps the rejections
/// above from passing for a gate that simply refuses every step count — can never be reached.
/// `mochi_1` is `video_mlx_routed` + `candle_video_routed` and serves `text_to_video` only, so both
/// lanes claim it on every platform. The menu itself is fixture-declared, which is the point: this
/// asserts the ENFORCEMENT of `limits.steps`, not any particular model's declaration of it.
///
/// It was `ltx_2_3_eros` until the 2026-08-19 main sync. That id is now a product WITHDRAWAL
/// off-Mac (sc-18902 removed its candle route; `video_model_withdrawn_on_platform` names it
/// literally), so on `parity-rust` the platform gate 400'd first with "available only on macOS" and
/// the catalog read below could not find the row at all — a fixture id that had quietly become
/// platform-dependent. Nothing about the step-menu contract changed; only the id it is asserted on.
#[tokio::test]
async fn video_steps_off_the_post_preset_models_exact_menu_is_rejected() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    let default_video_model = crate::defaults::default_video_model();
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "__DEFAULT_VIDEO_MODEL__",
              "name": "Default Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/default-vid", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "steps": 8 },
              "limits": {},
              "ui": { "label": "Default Vid" }
            },
            {
              "id": "mochi_1",
              "name": "Distilled Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/mochi_1", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": { "steps": 8 },
              "limits": { "steps": [8] },
              "ui": { "label": "Distilled Vid" }
            }
          ]
        }
        "#
        .replace("__DEFAULT_VIDEO_MODEL__", &default_video_model),
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    std::fs::write(
        config_dir.join("builtin.recipe-presets.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "presets": [
            {
              "id": "preset_override",
              "name": "Preset Override",
              "workflow": "text_to_video",
              "model": "mochi_1",
              "defaults": {},
              "prompt": { "prefix": "cinematic", "suffix": "smooth" }
            }
          ]
        }
        "#,
    )
    .expect("builtin recipe presets writes");
    std::fs::write(
        config_dir.join("user.recipe-presets.jsonc"),
        r#"{ "schemaVersion": 1, "presets": [] }"#,
    )
    .expect("user recipe presets writes");

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Step Menu Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // THE CASE THE STORY NAMES. 30 steps: over the menu, so a FLOOR-shaped key would have admitted
    // it. `model` omitted so the preset's model wins — the sc-12300 shape.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 30 },
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "30 steps is off mochi_1's exact menu and must be refused at enqueue: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("mochi_1"),
        "names the model whose menu applied — NOT the default's: {detail}"
    );
    assert!(
        detail.contains("fixed 8-step schedule"),
        "states the legal value: {detail}"
    );
    assert!(
        detail.contains("asks for 30 steps"),
        "states what was asked: {detail}"
    );

    // Under-menu is refused too, so the menu is not secretly a ceiling.
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 4 },
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "4 is off the menu as well");

    // ON the menu admits, and travels VERBATIM. This is what keeps the assertions above from
    // passing for a gate that simply rejects every step count.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 8 },
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "8 is the menu: {body}");
    assert_eq!(body["payload"]["model"], "mochi_1");
    assert_eq!(
        body["payload"]["advanced"]["steps"], 8,
        "the admitted count travels VERBATIM — the gate refuses, it never rewrites"
    );

    // A request naming NO steps is admitted: `advanced` is a passthrough map, so an omitted `steps`
    // means the engine runs its baked schedule. Refusing it would make every distilled model
    // unusable without the caller knowing its magic number.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "recipePresetId": "preset_override"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a steps-less payload must not be refused by the menu: {body}"
    );

    // ...and the SAME 30-step request against the default model (no menu) is admitted, proving the
    // rejection above came from the per-model menu rather than a blanket step bound.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "advanced": { "steps": 30 },
            "model": default_video_model
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the default model declares no menu, so 30 steps is its own business: {body}"
    );

    // REACHABILITY FOR THE UI HALF. Video Studio pins its Steps control off `limits.steps`, which
    // only works if the catalog endpoint actually serializes the key — a `limits` block that
    // allowlisted its contents would leave the control free while the gate above still refused, i.e.
    // the UI/gate desync this story exists to remove. Asserted on the wire rather than by reading
    // the serializer.
    let (status, catalog) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "catalog lists: {catalog}");
    let listed = catalog
        .as_array()
        .expect("catalog is an array")
        .iter()
        .find(|model| model["id"] == "mochi_1")
        .expect("mochi_1 is listed");
    assert_eq!(
        listed["limits"]["steps"],
        json!([8]),
        "the catalog must carry limits.steps through to the studio: {listed}"
    );
}

/// sc-12400 — the regression sc-12297 shipped: a request that names **no duration at all** was
/// rejected for "asking for 6s", on 7 of the 10 shipped video models.
///
/// Against the REAL manifest, because the whole bug is that the DTO's blanket 6.0 was never compared
/// to the caps sc-12297 began enforcing — a fixture would let me pick a cap that hides it.
///
/// This is the exact payload the MCP `generate_video` tool sends when its caller names only a model
/// and a prompt (`server.rs` omits every `None` optional), so it is the natural non-UI call, not an
/// edge case. Before the fix: `400 mochi_1 renders clips up to 5s, but this request asks for 6s` —
/// naming a value the caller never set, with a lever for a field they never touched.
#[tokio::test]
async fn a_video_request_naming_no_duration_is_admitted_at_the_models_own_default() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .expect("builtin manifests seed");

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Duration Default Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // Every shipped video model whose cap is UNDER the old blanket 6.0 — i.e. every model this
    // regression bricked — plus ltx_2_3, whose cap of 15 admitted 6.0 and which therefore stayed
    // green throughout. Listing the survivor alongside the victims is what makes this a per-model
    // assertion rather than "the route works".
    //
    // Each row now carries a mode the model actually SERVES, plus that mode's required media
    // (sc-19504). It used to submit `text_to_video` for all seven, which three of them do not do at
    // all — `scail2_14b` is animate/replace only, `wan_2_2_vace_fun_14b` is replace only, `svd` is
    // image-conditioned only — so those three rows were asserting a 201 on a job NO lane would have
    // claimed, i.e. one that would have sat queued forever. The enqueue no-lane gate refuses that
    // shape now, and the fix is to drive each model where it lives: the duration default this test
    // is about is resolved from the manifest entry and is mode-independent, so nothing is weakened.
    for (model, mode, media, want_duration) in [
        ("bernini", "text_to_video", json!({}), 5.0),
        (
            "scail2_14b",
            "animate_character",
            json!({ "sourceClipAssetId": "clip-1", "referenceAssetIds": ["img-1"] }),
            5.0,
        ),
        ("wan_2_2_t2v_14b", "text_to_video", json!({}), 5.0),
        (
            "wan_2_2_i2v_14b",
            "image_to_video",
            json!({ "sourceAssetId": "img-1" }),
            5.0,
        ),
        (
            "wan_2_2_vace_fun_14b",
            "replace_person",
            json!({
                "sourceClipAssetId": "clip-1",
                "personTrackId": "track-1",
                "characterId": "character-1"
            }),
            5.0,
        ),
        (
            "svd",
            "image_to_video",
            json!({ "sourceAssetId": "img-1" }),
            4.0,
        ),
        ("ltx_2_3", "text_to_video", json!({}), 6.0),
    ] {
        let mut request_body = json!({
            "projectId": project_id,
            "mode": mode,
            "prompt": "a fox runs",
            "model": model
        });
        request_body
            .as_object_mut()
            .expect("body object")
            .extend(media.as_object().expect("media object").clone());
        let (status, body) = request(app.clone(), "POST", "/api/v1/video/jobs", request_body).await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{model}: a request naming no duration must be admitted, not 400'd for a value it \
             never set: {body}"
        );
        assert_eq!(
            body["payload"]["duration"], want_duration,
            "{model}: the enqueued payload records the model's declared defaults.duration, not the \
             blanket 6.0: {body}"
        );
    }

    // A NAMED over-cap duration is still refused — the cap is intact; only the phantom 6.0 is gone.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "bernini",
            "duration": 30
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "sc-12297's cap must still refuse a duration the caller actually asked for: {body}"
    );
    assert!(body["detail"]
        .as_str()
        .unwrap_or_default()
        .contains("asks for 30s"));

    // The GEOMETRY axis of the same bug — the third dead `defaults.*` key. No 400 to observe here
    // (dimensions coerce), so the observable is the enqueued size itself: `bernini` must take its
    // declared 848x480 native bucket (stride 16), NOT the blanket 768x512 it never advertises.
    // `wan_2_2_t2v_14b` (1280x720) and `ltx_2_3` (768x512) come along to prove this reads the
    // per-model value rather than one hardcoded pair.
    for (model, want_w, want_h) in [
        ("bernini", 848, 480),
        // True 720p since sc-12308 (#1581) lifted the A14B area cap to its real 921,600.
        ("wan_2_2_t2v_14b", 1280, 720),
        ("ltx_2_3", 768, 512),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_video",
                "prompt": "a fox runs",
                "model": model
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{model}: {body}");
        assert_eq!(
            (
                body["payload"]["width"].as_u64(),
                body["payload"]["height"].as_u64()
            ),
            (Some(want_w), Some(want_h)),
            "{model}: a size-less request must enqueue the model's declared defaults.resolution: \
             {body}"
        );
    }

    // A NAMED size is still honored verbatim — resolution fills a gap, it does not override.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_video",
            "prompt": "a fox runs",
            "model": "bernini",
            "width": 640,
            "height": 384
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(
        (
            body["payload"]["width"].as_u64(),
            body["payload"]["height"].as_u64()
        ),
        (Some(640), Some(384)),
        "a caller's own size must not be replaced by the model's default: {body}"
    );
}

/// sc-12400 (image half): a request that names no size renders the MODEL's declared
/// `defaults.resolution`, not a blanket 1024 square.
///
/// The image twin of `a_video_request_naming_no_duration_is_admitted_at_the_models_own_default`,
/// and against the REAL manifest for the same reason: the bug is precisely that the DTO's blanket
/// was never compared to what each model declares, so a fixture would let me pick the values that
/// hide it.
///
/// No 400 to observe on this axis — geometry coerces — so the observable IS the enqueued size. The
/// expectations are transcribed from the manifest, never derived from `default_resolution`: routing
/// them through the function under test is what made the video half's first cut a tautology that
/// survived deleting the feature.
#[tokio::test]
async fn an_image_request_naming_no_size_renders_the_models_own_declared_resolution() {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .expect("builtin manifests seed");

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Image Default Resolution Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    // A matrix that pins each axis INDEPENDENTLY. `defaults.resolution` and `defaults.count` are
    // read by separate code, so a fixture where the two always agree cannot tell a working pair
    // from one reader quietly carrying the other.
    //
    // Every id here is verified to exist. The first cut used `flux1_schnell`, which is NOT a model
    // (it is `flux_schnell`): it resolved to `{}`, took the unknown-model blanket, and passed —
    // green, while testing nothing about a declared default. **A typo'd id is a silent no-op in
    // this test by construction**, which is exactly why the control row must be a REAL model whose
    // declarations happen to equal the blankets.
    for (model, want_w, want_h, want_count) in [
        // BOTH axes discriminate. 2048x2048 + count 1: the blankets rendered this at HALF
        // resolution and FOUR times over, on the text/infographic family where pixels are the
        // entire point — and honoring only the size made the bare call 4x MORE expensive.
        ("sensenova_u1_8b", 2048, 2048, 1),
        ("sensenova_u1_8b_fast", 2048, 2048, 1),
        // COUNT discriminates, resolution coincides — red if the count reader is dropped, green if
        // only the size reader works.
        ("z_image", 1024, 1024, 1),
        // RESOLUTION discriminates, count coincides — the mirror image.
        ("chroma1_flash", 768, 768, 4),
        // NEITHER discriminates: a real model whose declarations equal both blankets. Keeps the
        // rows above from passing for a reader that returns some other model's values.
        ("z_image_turbo", 1024, 1024, 4),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "mode": "text_to_image",
                "prompt": "a fox",
                "model": model
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{model}: {body}");
        assert_eq!(
            (
                body["payload"]["width"].as_u64(),
                body["payload"]["height"].as_u64()
            ),
            (Some(want_w), Some(want_h)),
            "{model}: a size-less request must enqueue the model's declared defaults.resolution: \
             {body}"
        );
        assert_eq!(
            body["payload"]["count"].as_u64(),
            Some(want_count),
            "{model}: a count-less request must enqueue the model's declared defaults.count, not \
             the blanket 4: {body}"
        );
        // The seed batch is generated from the RESOLVED count, so it must agree — otherwise a
        // count-1 model would still carry four seeds and the payload would contradict itself.
        assert_eq!(
            body["payload"]["seeds"].as_array().map(Vec::len),
            Some(want_count as usize),
            "{model}: one seed per image actually rendered: {body}"
        );
    }

    // A NAMED size is honored verbatim — resolution fills a gap, it never overrides.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "sensenova_u1_8b",
            "width": 512,
            "height": 512
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(
        (
            body["payload"]["width"].as_u64(),
            body["payload"]["height"].as_u64()
        ),
        (Some(512), Some(512)),
        "a caller's own size must not be replaced by the model's default: {body}"
    );
}

// --- sc-13134: server-side Style fold (headless/MCP parity with the web composer) -----------

/// Write the minimal builtin manifests a Style-fold image job needs into `<config>/manifests`:
/// a single image model, empty user models, and a small Style catalog carrying one group with one
/// sub-style. Returns nothing — the caller builds the app from the same temp dir.
fn write_style_test_manifests(config_dir: &std::path::Path) {
    let manifests = config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir creates");
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "img-model",
              "name": "Img Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/img", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Img" }
            },
            {
              "id": "tag-model",
              "name": "Tag Model",
              "family": "anima",
              "type": "image",
              "adapter": "anima",
              "captionStyle": "tags",
              "capabilities": ["text_to_image"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/tag", "files": ["*.safetensors"], "default": true }
              ],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["anima"] },
              "ui": { "label": "Tag" }
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        manifests.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    // A tiny Style catalog shaped exactly like the shipped builtin.styles.jsonc: one group
    // (its `description` is the group's "overall" style text) with one sub-style (`prompt`).
    std::fs::write(
        manifests.join("builtin.styles.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "source": "documents/style.txt",
          "promptTemplate": "Subject: {subject}\nStyle: {style}",
          "groups": [
            {
              "id": "test-anime",
              "name": "Test Anime",
              "description": "broad test anime look",
              "styles": [
                { "id": "test-ghibli", "name": "Test Ghibli", "prompt": "gentle hand-painted" }
              ]
            }
          ]
        }
        "#,
    )
    .expect("builtin styles writes");
}

#[tokio::test]
async fn styled_image_job_folds_style_server_side_from_style_id() {
    // Headless/MCP parity (sc-13134): a client that sends a `styleId` + a RAW prompt gets the
    // exact `Subject:`/`Style:` composition the web composer produces — including the
    // directive-collision splice (a user `Setting:` line stays a top-level sibling, the free text
    // folds into Subject). This is the whole point of the story: the same styled prompt whether
    // the fold happens on the web or on the server.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Styled Job Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    // Sub-style id → its `prompt` ("gentle hand-painted"); "a fox" becomes the leading Subject and
    // the user's `Setting:` directive is kept as a trailing sibling — byte-identical to
    // composeStyledPrompt.
    let (status, styled) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox\nSetting: snowy field",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "test-ghibli"
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "styled job created: {styled:?}"
    );
    assert_eq!(
        styled["payload"]["prompt"],
        "Subject: a fox\nStyle: gentle hand-painted\nSetting: snowy field",
        "server-side fold must match the web composer output"
    );

    // A GROUP id resolves to that group's `description` (the "overall" style), sc-13171 parity.
    let (status, group_styled) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "test-anime"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{group_styled:?}");
    assert_eq!(
        group_styled["payload"]["prompt"],
        "Subject: a fox\nStyle: broad test anime look"
    );

    // An unknown styleId is a clean 400, not a silent no-op.
    let (status, err) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": "a fox",
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "styleId": "no-such-style"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");
}

#[tokio::test]
async fn styled_image_job_rejects_a_style_on_a_tag_convention_model() {
    // Booru-tag models (`captionStyle: "tags"`) take comma-separated tags, so the prose Style catalog
    // does not apply. The studio hides the axis; a headless/MCP caller that sends a styleId anyway
    // must get a named 400 rather than a silently prose-wrapped prompt.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Tag Model Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let body = json!({
        "projectId": project_id,
        "mode": "text_to_image",
        "prompt": "1girl, solo, pink hair",
        "model": "tag-model",
        "count": 1,
        "width": 1024,
        "height": 1024,
        "styleId": "test-ghibli"
    });
    let (status, err) = request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");
    let message = err["detail"].as_str().unwrap_or_default().to_owned();
    assert!(
        message.contains("tag-model") && message.contains("tag-style"),
        "the rejection must name the model and the reason: {message}"
    );

    // The gate is a MODEL capability, not a fold detail: it fires even when the caller claims the
    // prompt was already composed client-side (the flag that otherwise short-circuits the fold).
    let mut claimed = body;
    claimed["presetPromptResolvedClientSide"] = json!(true);
    let (status, err) = request(app.clone(), "POST", "/api/v1/image/jobs", claimed).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{err:?}");

    // Same model with NO styleId is untouched — the gate rejects the axis, not the model.
    let (status, plain) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project["id"].as_str().expect("project id"),
            "mode": "text_to_image",
            "prompt": "1girl, solo, pink hair",
            "model": "tag-model",
            "count": 1,
            "width": 1024,
            "height": 1024
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{plain:?}");
    assert_eq!(plain["payload"]["prompt"], "1girl, solo, pink hair");
}

#[tokio::test]
async fn styled_image_job_skips_fold_when_prompt_resolved_client_side() {
    // The web app composes the styled prompt CLIENT-side and sends it verbatim plus
    // presetPromptResolvedClientSide (mirroring the preset skip). The server must take the prompt
    // as-is and NOT re-fold — even when a `styleId` rides along (the studio records it for replay).
    // Otherwise a web-submitted "Subject: …\nStyle: …" would be double-wrapped.
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Client Styled Project" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();

    let already_composed = "Subject: a fox\nStyle: gentle hand-painted";
    let (status, verbatim) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "mode": "text_to_image",
            "prompt": already_composed,
            "model": "img-model",
            "count": 1,
            "width": 1024,
            "height": 1024,
            // styleId-with-flag: the web records the picked id but has already composed the prompt.
            "styleId": "test-ghibli",
            "presetPromptResolvedClientSide": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{verbatim:?}");
    assert_eq!(
        verbatim["payload"]["prompt"], already_composed,
        "a client-composed styled prompt must be taken verbatim, never double-folded"
    );
}

#[tokio::test]
async fn styles_endpoint_serves_the_builtin_catalog() {
    // GET /api/v1/styles gives headless/MCP clients the same catalog the web reads: the grouped
    // Style picker data they need to choose a styleId.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_style_test_manifests(&temp_dir.path().join("config"));
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, catalog) = request(app.clone(), "GET", "/api/v1/styles", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{catalog:?}");
    assert_eq!(catalog["schemaVersion"], 1);
    assert_eq!(catalog["groups"][0]["id"], "test-anime");
    assert_eq!(catalog["groups"][0]["styles"][0]["id"], "test-ghibli");
    assert_eq!(
        catalog["groups"][0]["styles"][0]["prompt"],
        "gentle hand-painted"
    );
}

/// **THE CLASS GUARD, API half (sc-17159, GH #2074).** Every video mode a shipped model advertises
/// in `capabilities` must be a mode `POST /api/v1/video/jobs` will actually ACCEPT.
///
/// `validate_video_job`'s allow-list is a SEPARATE reachability gate from the catalog: a mode
/// missing from it 400s with "Unsupported video mode" no matter how completely the rest of the
/// stack is wired. That is exactly how SCAIL-2's `animate_character` shipped — catalog,
/// `VIDEO_UI_MODES`, `video_mode_is_mlx_eligible`, the candle claim gate, the worker's
/// `generate_scail2` AND the Video Studio tab, all correct, and every submission rejected.
///
/// Read off BOTH real sources so it cannot go stale: the advertisement comes from the shipped
/// `builtin.models.jsonc` bytes and the admission from [`crate::VIDEO_JOB_MODES`], the constant
/// `validate_video_job` itself consults. A guard that re-typed the mode list would assert against
/// its own copy and stay green through exactly the drift it exists to catch.
///
/// The routing half of the same class is
/// `every_declared_video_capability_is_claimable_by_some_lane` (sceneworks-core routing/catalog.rs),
/// which proves some lane will CLAIM each of these once submitted.
#[test]
fn every_declared_video_capability_is_submittable() {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    let videos: Vec<&Value> = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .filter(|entry| entry["type"] == "video")
        .collect();
    assert!(
        videos.len() >= 12,
        "only {} shipped video models were read — the manifest parse is wrong and this guard is \
         vacuous",
        videos.len()
    );

    let mut checked = 0_usize;
    for entry in &videos {
        let id = entry["id"].as_str().expect("every model row has an id");
        let capabilities = entry["capabilities"]
            .as_array()
            .unwrap_or_else(|| panic!("{id}: every shipped video model declares capabilities"));
        assert!(!capabilities.is_empty(), "{id}: declares no capability");
        for capability in capabilities {
            let mode = capability
                .as_str()
                .unwrap_or_else(|| panic!("{id}: capabilities entries are strings"));
            checked += 1;
            assert!(
                crate::VIDEO_JOB_MODES.contains(&mode),
                "{id} advertises `{mode}` in builtin.models.jsonc — the Video Studio builds a tab \
                 for it — but `validate_video_job`'s allow-list does not admit it, so every \
                 submission 400s with \"Unsupported video mode\" and the mode is unreachable from \
                 the moment it ships (GH #2074). Add it to VIDEO_JOB_MODES with its per-mode \
                 required-asset arm, or stop advertising it."
            );
        }
    }
    assert!(
        checked >= 30,
        "only {checked} (model, capability) pairs were checked — this guard is vacuous"
    );

    // …and the converse, so the allow-list cannot quietly accumulate modes nothing serves: every
    // admitted mode is advertised by at least one shipped video model.
    let advertised: std::collections::BTreeSet<&str> = videos
        .iter()
        .filter_map(|entry| entry["capabilities"].as_array())
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    for mode in crate::VIDEO_JOB_MODES {
        assert!(
            advertised.contains(mode),
            "`{mode}` is admitted by validate_video_job but no shipped video model advertises it \
             — either a model lost the capability or the allow-list has a dead entry"
        );
    }
}

/// Seed the SHIPPED manifests into a temp config dir and return a live app + project id, so a
/// video-jobs test drives the real `minimax_h3` / `minimax_h3_ref` entries rather than a fixture
/// whose numbers the test itself chose (sc-17159 — a seeded probe would prove the ROUTE works, not
/// that the shipped family is reachable).
async fn shipped_manifest_app(temp_dir: &tempfile::TempDir) -> (axum::Router, String) {
    shipped_manifest_app_on_os(temp_dir, "macos").await
}

/// The same app, told it is running on `os` (sc-19570). The ONLY difference from
/// [`shipped_manifest_app`] is `Settings::host_os`, which production always fills with
/// `std::env::consts::OS`.
///
/// It exists because macOS structurally cannot detect the defect sc-19570 fixed by running on
/// itself: the per-mode reachability sweep terminates exactly what no Windows/Linux lane will
/// claim, and on a Mac that branch never executes. Tagging the fixture with the FOREIGN OS is what
/// makes the check run everywhere, on the sc-17227 precedent.
///
/// What varies with `os` is the JOB's outcome, never the response. `POST /api/v1/video/jobs`
/// answers `201` for the same body on every value passed here — that is asserted directly by
/// [`the_video_enqueue_contract_is_identical_on_every_platform`] — and only the created job's
/// `status` differs. A guard that expects a different STATUS CODE per `os` is asserting the shape
/// this story removed.
///
/// The OS is always passed explicitly — never read from `std::env::consts::OS`. The two lanes that
/// run this suite disagree (`parity-rust` is `ubuntu-latest`, the hosted workspace job is macOS),
/// so reading the runner would make every assertion here mean something different depending on
/// which lane executed it. [`shipped_manifest_app`] therefore pins macOS and these guards pin the
/// foreign OS, and both lanes reach the same verdict.
async fn shipped_manifest_app_on_os(
    temp_dir: &tempfile::TempDir,
    os: &str,
) -> (axum::Router, String) {
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let mut settings = test_settings(temp_dir);
    settings.host_os = os.to_owned();
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .expect("builtin manifests seed");
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "MiniMax-H3 Reachability" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    (app, project_id)
}

/// sc-17159 (epic 17137) — every mode MiniMax-H3 DECLARES is submittable end to end against the
/// SHIPPED manifest, and the enqueued payload carries what the worker needs.
///
/// "Declaration is not enforcement is not reachability." sc-17158 declared the two entries and
/// their geometry; this asserts a caller can actually get each of the four declared modes past
/// EVERY gate on the enqueue path — the allow-list, the per-mode required-asset arm, the three
/// reference-list blankets, the payload-sanity duration/fps/dimension bounds, and then the
/// per-model `duration_limit_error` / `fps_limit_error` / `reference_limit_error` that only fire
/// once the manifest entry is resolved. A `201` here is the whole point: asserting the mode string
/// appears in a list is what GH #2074 already passed.
#[tokio::test]
async fn minimax_h3_every_declared_mode_is_accepted_end_to_end() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id) = shipped_manifest_app(&temp_dir).await;

    // (label, model, the mode-specific half of the payload). One case per declared capability:
    // `minimax_h3` = [text_to_video, image_to_video, first_last_frame], `minimax_h3_ref` =
    // [reference_to_video]. fl2va is "0, 1 or 2 images", which is three payload shapes across two
    // modes, so all three are driven.
    let cases = [
        ("t2va", "minimax_h3", json!({ "mode": "text_to_video" })),
        (
            "fl2va first frame only",
            "minimax_h3",
            json!({ "mode": "image_to_video", "sourceAssetId": "asset-first" }),
        ),
        (
            "fl2va both frames",
            "minimax_h3",
            json!({
                "mode": "first_last_frame",
                "sourceAssetId": "asset-first",
                "lastFrameAssetId": "asset-last"
            }),
        ),
        (
            "Ref2VA images + clips + audio",
            "minimax_h3_ref",
            json!({
                "mode": "reference_to_video",
                "referenceAssetIds": ["img-0", "img-1", "img-2", "img-3", "img-4", "img-5", "img-6", "img-7", "img-8"],
                "sourceClipAssetIds": ["clip-0", "clip-1"],
                "referenceAudioAssetIds": ["aud-0"]
            }),
        ),
        // Audio references as a COMPANION to a visual one — the shape sc-17159 unblocked, minus
        // the audio-ONLY case it also unblocked by mistake. sc-19574 refused that one again (see
        // `minimax_h3_refusals_each_name_their_own_reason`): upstream's `before_encoder.py` raises
        // on `set(kinds) == {"audio"}`, so it is not a shape the checkpoint serves.
        (
            "Ref2VA one image with three audio references",
            "minimax_h3_ref",
            json!({
                "mode": "reference_to_video",
                "referenceAssetIds": ["img-0"],
                "referenceAudioAssetIds": ["aud-0", "aud-1", "aud-2"]
            }),
        ),
        (
            "Ref2VA video clips only",
            "minimax_h3_ref",
            json!({
                "mode": "reference_to_video",
                "sourceClipAssetIds": ["clip-0", "clip-1", "clip-2"]
            }),
        ),
    ];

    for (label, model, extra) in cases {
        let mut body = json!({
            "projectId": project_id,
            "model": model,
            "prompt": "a lighthouse keeper hums while the storm rolls in"
        });
        body.as_object_mut()
            .expect("body object")
            .extend(extra.as_object().expect("case object").clone());
        let (status, job) = request(app.clone(), "POST", "/api/v1/video/jobs", body).await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "{label}: a mode the manifest declares must be accepted: {job}"
        );
        // All four modes map to the base job type, which is what routes to `run_video_generate_job`
        // — none of them is an extend/bridge/replace shape.
        assert_eq!(job["type"], "video_generate", "{label}");
        // The resolved geometry the worker will read: the model's own declared defaults, not the
        // route's historical blankets (6.0 s / 25 fps / 768x512 — none of which is on this
        // family's lattice, menu or bucket list).
        assert_eq!(
            job["payload"]["duration"], 5.1667,
            "{label}: defaults.duration is the shortest lattice rung"
        );
        assert_eq!(job["payload"]["fps"], 24, "{label}: the one cadence");
        assert_eq!(job["payload"]["width"], 1344, "{label}");
        assert_eq!(job["payload"]["height"], 768, "{label}");
        // The entry actually resolved — the worker reads geometry off this, so a job enqueued
        // without it silently loses the lattice and the area budget.
        assert_eq!(
            job["payload"]["modelManifestEntry"]["id"], model,
            "{label}: the resolved manifest entry must be the one the caller asked for"
        );
    }

    // The conditioning media reaches the worker verbatim rather than merely validating — all
    // three reference lists survive into the enqueued payload. (These pass-through assertions
    // were withheld together with the route and restored by sc-18650, alongside the same four in
    // `ref2va_reference_caps_refuse_fifteen_files_and_admit_twelve`.)
    let (status, ref2va) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "model": "minimax_h3_ref",
            "mode": "reference_to_video",
            "prompt": "the subject speaks over the rain",
            "referenceAssetIds": ["img-0", "img-1"],
            "sourceClipAssetIds": ["clip-0"],
            "referenceAudioAssetIds": ["aud-0"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{ref2va}");
    assert_eq!(
        ref2va["payload"]["referenceAssetIds"],
        json!(["img-0", "img-1"]),
        "the reference images must reach the worker verbatim: {ref2va}"
    );
    assert_eq!(
        ref2va["payload"]["sourceClipAssetIds"],
        json!(["clip-0"]),
        "the source clips must reach the worker verbatim: {ref2va}"
    );
    assert_eq!(
        ref2va["payload"]["referenceAudioAssetIds"],
        json!(["aud-0"]),
        "the audio references must reach the worker verbatim: {ref2va}"
    );

    // A named duration ON the lattice is honoured verbatim — the accepting side of the menu, so
    // the refusals below are not just "everything is rejected".
    let (status, on_rung) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id, "model": "minimax_h3", "mode": "text_to_video",
            "prompt": "a fox", "duration": 14.375
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{on_rung}");
    assert_eq!(on_rung["payload"]["duration"], 14.375);
}

/// sc-17159 — the refusals sc-17158 established are INTACT after the family is made reachable, and
/// each one names its OWN reason.
///
/// Pinning the exact `detail` matters more than the status here: a bare "it 400'd" assertion goes
/// inert the moment the request would have been rejected for some other reason (sc-19488). Each
/// case below is constructed to be legal in every respect but the one under test, and the message
/// is compared in full.
#[tokio::test]
async fn minimax_h3_refusals_each_name_their_own_reason() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id) = shipped_manifest_app(&temp_dir).await;

    let submit = |model: &str, extra: Value| {
        let app = app.clone();
        let project_id = project_id.clone();
        let model = model.to_owned();
        async move {
            let mut body = json!({
                "projectId": project_id, "model": model, "mode": "text_to_video",
                "prompt": "a fox runs"
            });
            body.as_object_mut()
                .expect("body object")
                .extend(extra.as_object().expect("case object").clone());
            request(app, "POST", "/api/v1/video/jobs", body).await
        }
    };

    // 15.0 s: the reference ADVERTISES it and the lattice cannot reach it (the next rung, 362
    // frames, is 15.083 s). Refused rather than silently delivered as 14.375 — which is exactly
    // what sc-17147 was doing.
    let (status, over) = submit("minimax_h3", json!({ "duration": 15.0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        over["detail"],
        "minimax_h3 renders clips up to 14.375s, but this request asks for 15s. Shorten the clip \
         to 14.375s or less, or choose a model that renders longer clips."
    );

    // 3.0 s: under `hardMinDuration`. The floor is checked FIRST, so the message offers the
    // lengthen lever and never the shorten one.
    let (status, under) = submit("minimax_h3", json!({ "duration": 3.0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        under["detail"],
        "minimax_h3 renders clips of at least 5.1667s, but this request asks for 3s. Lengthen the \
         clip to 5.1667s or more, or choose a model that renders shorter clips."
    );

    // T = 1. There is NO image lane for this family — `min_duration` is a hardcoded 5.0 upstream,
    // so a single-frame request does not render at all. Both spellings are refused, and they are
    // refused by DIFFERENT gates, so both messages are pinned rather than assumed:
    //   * 1/24 s is below the route's payload-sanity blanket and never reaches the model's floor;
    //   * 1.0 s clears the blanket and is refused by the model's own declared floor.
    let (status, single_frame) = submit("minimax_h3", json!({ "duration": 1.0 / 24.0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(single_frame["detail"], "duration must be between 1 and 30");
    let (status, one_second) = submit("minimax_h3", json!({ "duration": 1.0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        one_second["detail"],
        "minimax_h3 renders clips of at least 5.1667s, but this request asks for 1s. Lengthen the \
         clip to 5.1667s or more, or choose a model that renders shorter clips."
    );

    // fps: the family declares ONE cadence. 30 is off-menu.
    let (status, fps) = submit("minimax_h3", json!({ "fps": 30 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        fps["detail"]
            .as_str()
            .unwrap_or_default()
            .starts_with("minimax_h3 renders at 24 fps"),
        "the fps refusal must name the model's own menu: {fps}"
    );

    // Canvas. The AREA budget is 1,032,192 px and it is NOT an enqueue refusal — an over-cap
    // canvas is REFIT by `normalized_dimensions` at the worker, so the route accepts it. What the
    // route refuses is the per-edge blanket. Both halves are pinned so "over-cap canvas" is never
    // read as "rejected at enqueue".
    let (status, square) = submit(
        "minimax_h3",
        json!({ "width": 1344, "height": 1344, "duration": 5.1667 }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "1344x1344 is accepted at enqueue and refit at the worker: {square}"
    );
    let entry = square["payload"]["modelManifestEntry"].clone();
    assert_eq!(entry["limits"]["maxPixels"], 1_032_192);
    let refit = sceneworks_core::video_request::VideoRequest::from_payload(
        square["payload"].as_object().expect("payload object"),
    );
    assert!(
        u64::from(refit.width) * u64::from(refit.height) <= 1_032_192,
        "the enqueued payload must refit under the area budget, got {}x{}",
        refit.width,
        refit.height
    );
    assert_ne!(
        (refit.width, refit.height),
        (1344, 1344),
        "the refit must actually fire, else the cap is decoration"
    );
    let (status, too_wide) = submit("minimax_h3", json!({ "width": 2016, "height": 512 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(too_wide["detail"], "width must be between 256 and 1920");

    // Reference caps. The base partition declares 0/0/0 precisely because its checkpoint has no
    // reference path — a Ref2VA-shaped request must not reach a checkpoint that would ignore it.
    let (status, base_images) = submit(
        "minimax_h3",
        json!({ "mode": "reference_to_video", "referenceAssetIds": ["img-0"] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        base_images["detail"],
        "minimax_h3 takes no reference images, but this request supplies 1. Remove \
         referenceAssetIds, or choose a model that conditions on reference images."
    );
    let (status, base_audio) =
        submit("minimax_h3", json!({ "referenceAudioAssetIds": ["aud-0"] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        base_audio["detail"],
        "minimax_h3 takes no audio references, but this request supplies 1. Remove \
         referenceAudioAssetIds, or choose a model that conditions on audio references."
    );

    // The reference partition's own caps: 9 / 3 / 3 per list and 12 combined. 15 files clears
    // every per-list cap and is refused only by the combined budget.
    let ids = |prefix: &str, count: usize| -> Value {
        (0..count)
            .map(|i| Value::String(format!("{prefix}-{i}")))
            .collect::<Vec<_>>()
            .into()
    };
    let (status, combined) = submit(
        "minimax_h3_ref",
        json!({
            "mode": "reference_to_video",
            "referenceAssetIds": ids("img", 9),
            "sourceClipAssetIds": ids("clip", 3),
            "referenceAudioAssetIds": ids("aud", 3)
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        combined["detail"],
        "minimax_h3_ref takes up to 12 reference files in total, but this request supplies 15 \
         (9 reference images + 3 source clips + 3 audio references). Remove 3 of them."
    );
    // A 10th image is over the payload-sanity blanket, which is one layer ABOVE the per-model cap,
    // so it names the field rather than the model — a different gate, a different message.
    let (status, tenth) = submit(
        "minimax_h3_ref",
        json!({ "mode": "reference_to_video", "referenceAssetIds": ids("img", 10) }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        tenth["detail"],
        "referenceAssetIds must contain at most 9 ids"
    );
    // A 4th audio reference is inside the blanket (3 is the blanket AND the model's cap here, so
    // drive the model's cap through the clips list instead: 4 clips against its declared 3, well
    // inside the blanket 8).
    let (status, clips) = submit(
        "minimax_h3_ref",
        json!({ "mode": "reference_to_video", "sourceClipAssetIds": ids("clip", 4) }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        clips["detail"],
        "minimax_h3_ref takes up to 3 source clips, but this request supplies 4. Reduce \
         sourceClipAssetIds to 3 or fewer, or choose a model that takes more."
    );

    // fl2va's required media is still required: `first_last_frame` means BOTH frames.
    let (status, one_frame) = submit(
        "minimax_h3",
        json!({ "mode": "first_last_frame", "sourceAssetId": "asset-first" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        one_frame["detail"],
        "First/Last Frame requires first and last image assets."
    );
    // …and a reference-driven request with NO reference of any kind is still refused. sc-17159
    // loosened this arm from "at least one reference IMAGE" to "at least one reference", not to
    // "no reference needed".
    let (status, no_refs) = submit("minimax_h3_ref", json!({ "mode": "reference_to_video" })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        no_refs["detail"],
        "Reference to Video requires at least one reference image or video clip. Audio references \
         condition the soundtrack and cannot be the only reference."
    );
    // THE sc-19574 SHAPE: audio references and nothing else. sc-17159's widening went one list too
    // far — upstream's `before_encoder.py` raises on `set(kinds) == {"audio"}` because an audio
    // reference never reaches the visual conditioner — so the API accepted a request the worker
    // then refused. It is refused HERE now, which is the first point the user could learn it.
    let (status, audio_only) = submit(
        "minimax_h3_ref",
        json!({ "mode": "reference_to_video", "referenceAudioAssetIds": ["aud-0", "aud-1"] }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an audio-only reference set must not be admitted: {audio_only}"
    );
    assert_eq!(
        audio_only["detail"],
        "Reference to Video requires at least one reference image or video clip. Audio references \
         condition the soundtrack and cannot be the only reference."
    );
    // …and the SAME audio references alongside one image are accepted, so the refusal above is
    // about the missing visual reference and not about the audio list existing at all.
    let (status, audio_with_image) = submit(
        "minimax_h3_ref",
        json!({
            "mode": "reference_to_video",
            "referenceAssetIds": ["img-0"],
            "referenceAudioAssetIds": ["aud-0", "aud-1"]
        }),
    )
    .await;
    // ACCEPTED — the contrast the pairing draws: the audio-only case above is refused by the
    // ref2va payload rule, while the SAME audio list alongside one image is the shape Ref2VA
    // serves and enqueues (the route is live since sc-18650). If the payload rule ever wrongly
    // rejected audio-alongside-an-image, this arm would surface that refusal and go red.
    assert_eq!(
        status,
        StatusCode::CREATED,
        "audio alongside a visual reference is the shape Ref2VA serves — it must enqueue, never \
         be caught by the reference rule above: {audio_with_image}"
    );
    // Bernini's engine takes image references alone, so the loosened arm must not become a way to
    // hand its r2v path a clips-only conditioning set. The API admits it (the arm is
    // model-independent by design) and the worker's own `resolve_bernini_conditioning` refuses it,
    // naming bernini — the model-specific half of the requirement lives with the model.
    let (status, bernini_clips) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id, "model": "bernini", "mode": "reference_to_video",
            "prompt": "a fox", "sourceClipAssetIds": ["clip-0"]
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "the enqueue arm is model-independent: {bernini_clips}"
    );
    let (status, bernini_bare) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id, "model": "bernini", "mode": "reference_to_video",
            "prompt": "a fox"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bernini_bare["detail"],
        "Reference to Video requires at least one reference image or video clip. Audio references \
         condition the soundtrack and cannot be the only reference."
    );
}

/// **THE ENFORCEMENT GUARD (sc-19504).** A video mode NO lane will claim is refused at submission,
/// with a real request, rather than enqueued to wait forever.
///
/// This is the half `every_declared_video_capability_is_claimable_by_some_lane` and
/// `every_declared_video_capability_is_submittable` structurally cannot cover. Both read the
/// manifest, so both only see what a model ADVERTISES — and `VIDEO_JOB_MODES` is global: any
/// caller may name any admitted mode against any model, and the studio is not the only caller
/// (`sceneworks-mcp`'s `submit_video_job` picks `first_last_frame` on its own the moment a
/// `lastFrameAssetId` is present, then POSTs here). Withdrawing `first_last_frame` from
/// `wan_2_2_i2v_14b` removes the TAB; only the enqueue gate removes the HANG.
///
/// Driven against the SHIPPED manifest through the REAL route, because "the mode is in a list" is
/// what GH #2074 already passed. A `queued` job here is the defect: no worker claims it, no sweep
/// fails it (both enforce sweeps default to warn), and the user sees "Waiting for an available
/// worker." forever next to an idle worker (sc-15328).
#[tokio::test]
async fn a_video_mode_no_lane_serves_is_refused_at_submission() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id) = shipped_manifest_app(&temp_dir).await;

    let submit = |body: Value| {
        let app = app.clone();
        let project_id = project_id.clone();
        async move {
            let mut full = json!({ "projectId": project_id, "prompt": "a fox runs" });
            full.as_object_mut()
                .expect("body object")
                .extend(body.as_object().expect("case object").clone());
            request(app, "POST", "/api/v1/video/jobs", full).await
        }
    };

    // THE REPORTED DEFECT. Legal in every other respect — the mode is in `VIDEO_JOB_MODES`, both
    // required frames are present, the model exists, the geometry is default — and claimable by
    // nobody: the MLX I2V-A14B descriptor declares `conditioning: [Reference]` with no `Keyframe`,
    // and the candle i2v gate requires `mode == "image_to_video"`.
    let (status, flf) = submit(json!({
        "model": "wan_2_2_i2v_14b",
        "mode": "first_last_frame",
        "sourceAssetId": "img-first",
        "lastFrameAssetId": "img-last"
    }))
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a first_last_frame request on the 14B I2V must be REFUSED, not enqueued to wait for a \
         worker that will never claim it: {flf}"
    );
    assert_eq!(
        flf["detail"],
        "wan_2_2_i2v_14b cannot render the \"first_last_frame\" mode — no backend implements it, \
         so this job would wait for a worker that will never claim it. Choose a mode this model \
         lists in its capabilities, or a model that supports this one."
    );

    // …and the same shape on the 5B, which DOES have the mask-blend keyframe path, is accepted.
    // Without this the assertion above would be satisfied by a gate that refused every FLF request.
    let (status, five_b) = submit(json!({
        "model": "wan_2_2",
        "mode": "first_last_frame",
        "sourceAssetId": "img-first",
        "lastFrameAssetId": "img-last"
    }))
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "wan_2_2 (TI2V-5B) serves first_last_frame on MLX and must still be accepted: {five_b}"
    );

    // The gate is NOT a platform gate. `extend_clip` on this same model is served only by the
    // candle Wan-VACE lane — no MLX path at all — and must still enqueue, on a Mac included: the
    // two shipped topologies share one queue and a job waits for ITS lane's worker.
    let (status, extend) = submit(json!({
        "model": "wan_2_2_i2v_14b",
        "mode": "extend_clip",
        "sourceClipAssetId": "clip-1"
    }))
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "extend_clip is candle-VACE-served on this model and must still enqueue: {extend}"
    );
    assert_eq!(extend["type"], "video_extend");

    // The gate is NOT a capability gate either, and this case is why it must not become one:
    // `wan_2_2_t2v_14b` does not ADVERTISE `extend_clip`, but the candle VACE lane genuinely
    // renders it. A capabilities-shaped gate would 400 a working shape — a real regression for
    // every non-studio caller — while fixing nothing this gate does not already fix.
    let (status, undeclared) = submit(json!({
        "model": "wan_2_2_t2v_14b",
        "mode": "extend_clip",
        "sourceClipAssetId": "clip-1"
    }))
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "an undeclared but genuinely-served mode must not be refused: {undeclared}"
    );

    // The mode's own required-asset arm still runs FIRST, so a malformed request keeps its own
    // precise message instead of being flattened into "no backend implements it".
    let (status, missing) = submit(json!({
        "model": "wan_2_2_i2v_14b",
        "mode": "first_last_frame",
        "sourceAssetId": "img-first"
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        missing["detail"],
        "First/Last Frame requires first and last image assets."
    );

    // Every mode `wan_2_2_i2v_14b` still advertises survives the gate, so the withdrawal narrowed
    // exactly one capability and not the model.
    let (status, i2v) = submit(json!({
        "model": "wan_2_2_i2v_14b",
        "mode": "image_to_video",
        "sourceAssetId": "img-first"
    }))
    .await;
    assert_eq!(status, StatusCode::CREATED, "{i2v}");
    let (status, bridge) = submit(json!({
        "model": "wan_2_2_i2v_14b",
        "mode": "video_bridge",
        "sourceClipAssetId": "clip-1",
        "bridgeRightClipAssetId": "clip-2"
    }))
    .await;
    assert_eq!(status, StatusCode::CREATED, "{bridge}");
}

/// The measured MLX-only, candle-unclaimable pairs (sc-19570), each with the media its mode
/// requires so the request is legal in every OTHER respect — a 400 from a missing asset would be
/// the wrong outcome entirely and would leave a guard below passing for a reason that has nothing
/// to do with platform.
///
/// The two remaining mac-only-download families (`krea_realtime_14b`, `minimax_h3_ref`) ride
/// the same route as everything else, because "install state is not a reachability gate" is only an
/// argument until the REST leg actually exercises it.
///
/// It listed ALL TWENTY when sc-19570 measured it. Syncing `main` into this epic branch gave
/// thirteen of them a candle lane, and sc-20755 gave the three MiniMax-H3 base modes a Candle lane,
/// so four remain — exactly the two mac-only families above.
fn mlx_only_stranded_pairs() -> Vec<(&'static str, Value)> {
    vec![
        // SIXTEEN PAIRS WERE REMOVED HERE: thirteen when `main` was synced into the epic branch
        // and MiniMax-H3 base's three modes when sc-20755 added its measured Candle lane. The
        // `ltx_2_3` / `ltx_2_3_eros` five apiece, `wan_2_2`'s two keyframe shapes, and
        // `wan_2_2_vace_fun_14b`'s `replace_person`, plus MiniMax-H3 t2va and both fl2va shapes.
        // They now belong to `candle_served_pairs` rather than here. Keeping them would assert that
        // a served pair must TERMINATE off-Mac — the exact inversion of this guard. The same rows
        // left `MLX_ONLY_ADVERTISED_PAIRS` in
        // `routing/catalog.rs`, which is this table's core-side twin; the two must agree.
        // The four pairs the story's measurement did NOT list, because these two families ship
        // `platforms: ["macos"]` downloads and it scoped itself to Windows/Linux-installable
        // models. They belong on the REST leg specifically: the whole argument for having an
        // enqueue gate at all is that it must refuse a raw REST call REGARDLESS of install state,
        // and a core-predicate assertion cannot exercise that. Install state is not a reachability
        // gate — a mac-only download list is one manifest edit from changing — and the route
        // resolves these ids from the seeded manifest on every OS, so there is nothing to exempt.
        ("krea_realtime_14b", json!({ "mode": "text_to_video" })),
        (
            "krea_realtime_14b",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "krea_realtime_14b",
            json!({ "mode": "video_to_video", "sourceClipAssetId": "clip-1" }),
        ),
    ]
}

/// The pairs the candle lane genuinely serves off-Mac — the other half of every guard below. A
/// mechanism that simply failed everything off-Mac would satisfy the stranded assertions and go red
/// here.
fn candle_served_pairs() -> Vec<(&'static str, Value)> {
    vec![
        ("wan_2_2", json!({ "mode": "text_to_video" })),
        (
            "wan_2_2",
            json!({ "mode": "extend_clip", "sourceClipAssetId": "clip-1" }),
        ),
        (
            "wan_2_2",
            json!({ "mode": "replace_person", "sourceClipAssetId": "clip-1", "personTrackId": "track-1", "characterId": "char-1" }),
        ),
        ("ltx_2_3", json!({ "mode": "text_to_video" })),
        // Two of the thirteen pairs that moved out of `mlx_only_stranded_pairs` when `main` was
        // synced in: `candle_video_engine_id` resolves the LTX pair to `ltx_2_3_distilled`, which
        // serves both of these adapter-free. Asserted on this side rather than merely deleted from
        // the other, so the sync's claim — "these gained a lane" — is proved rather than assumed.
        // The advanced three (extend / bridge / replacement) are deliberately NOT here: they are
        // candle-served only with an IC-LoRA, so an adapter-free body would be refused at enqueue
        // and would prove the opposite of what this table is for.
        (
            "ltx_2_3",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "ltx_2_3",
            json!({ "mode": "first_last_frame", "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "wan_2_2_i2v_14b",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        // MiniMax-H3 base moved here in sc-20755; SC-20756 adds the Ref2VA twin.
        ("minimax_h3", json!({ "mode": "text_to_video" })),
        (
            "minimax_h3",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "minimax_h3",
            json!({ "mode": "first_last_frame", "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "minimax_h3_ref",
            json!({ "mode": "reference_to_video", "referenceAssetIds": ["img-1"] }),
        ),
        (
            "svd",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        ("bernini", json!({ "mode": "text_to_video" })),
        (
            "bernini",
            json!({ "mode": "video_to_video", "sourceClipAssetId": "clip-1" }),
        ),
    ]
}

/// Build the full `POST /api/v1/video/jobs` body for one `(model, case)` row.
fn video_job_body(project_id: &str, model: &str, case: &Value) -> Value {
    let mut full = json!({ "projectId": project_id, "prompt": "a fox runs", "model": model });
    full.as_object_mut()
        .expect("body object")
        .extend(case.as_object().expect("case object").clone());
    full
}

/// **THE HTTP CONTRACT GUARD (sc-19570).** `POST /api/v1/video/jobs` answers `201 Created` for
/// byte-identical bodies on macOS, Windows and Linux alike — for the MLX-only pairs AND for
/// the candle-served ones.
///
/// This is the property Michael ruled on: *"http contracts are not platform dependant and never
/// should be."* sc-19570's first shipped fix refused the MLX-only pairs with a `400` off-Mac, so
/// the published surface disagreed with itself across hosts and
/// `test_person_tracking_and_replace_person_contracts` (the cross-runtime parity suite, which runs
/// on Linux) caught it as a 400 where it expected 201.
///
/// The assertion is deliberately status-code-shaped and platform-blind: every row, every OS, one
/// expected value. A future edit that reintroduces ANY platform-conditional refusal on this route
/// — for any subset, with any message — turns this red. The companion guard
/// [`an_mlx_only_video_job_reaches_a_terminal_failed_state_off_mac`] owns the other half, that
/// accepting these off-Mac does not resurrect the hang.
#[tokio::test]
async fn the_video_enqueue_contract_is_identical_on_every_platform() {
    let stranded = mlx_only_stranded_pairs();
    let served = candle_served_pairs();
    // Was 20 when sc-19570 measured it. Syncing `main` into this epic branch gave thirteen of those
    // pairs a real Candle lane; sc-20755 moved the three MiniMax-H3 base modes, and sc-20756 moved
    // Ref2VA. The guard is KEPT, not deleted, and kept EXACT: a drop below three means a genuinely
    // stranded pair stopped being covered without its replacement lane being recorded here.
    assert_eq!(
        stranded.len(),
        3,
        "the stranded set is the three remaining mac-only-download pairs that enqueue — a shrunken table \
         would narrow every guard that reads it"
    );

    for os in ["macos", "windows", "linux"] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, os).await;
        for (model, case) in stranded.iter().chain(served.iter()) {
            let mode = case["mode"].as_str().expect("case names a mode");
            let body_json = video_job_body(&project_id, model, case);
            let (status, body) =
                request(app.clone(), "POST", "/api/v1/video/jobs", body_json).await;
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{model} + {mode} on {os}: the enqueue contract must not vary by platform — the \
                 same body answers 201 everywhere, and what this host can RENDER is reported on \
                 the job, not in the status code: {body}"
            );
            // The response SHAPE is part of the contract too: a job snapshot with an id, on every
            // platform. A host that answered 201 with a different envelope would be just as
            // platform-dependent as one that answered 400.
            assert!(
                body["id"].as_str().is_some_and(|id| !id.is_empty()),
                "{model} + {mode} on {os}: 201 must carry a job snapshot: {body}"
            );
        }
    }
}

/// The cross-runtime PARITY fixture, pinned here so its platform-independence is proved by a test
/// that runs on every lane (sc-19570).
///
/// `tests/test_rust_api_contract_snapshots.py::test_person_tracking_and_replace_person_contracts`
/// submits this exact body and snapshots the whole response, including the job's `status`, `stage`
/// and `error`. That suite runs on `ubuntu-latest` only, so a fixture whose job outcome depends on
/// the host records a Linux-shaped snapshot no other lane can reproduce — and it did: the fixture
/// used to omit `model`, inheriting the catalog default `ltx_2_3`, whose `replace_person` is
/// MLX-only and now terminates at once off-Mac.
///
/// `wan_2_2` serves `replace_person` on both lanes, so this asserts `201` AND `queued` on all
/// three platforms. If someone repoints that fixture at an MLX-only model, this goes red on a Mac
/// developer's machine instead of only on the Linux CI lane hours later.
#[tokio::test]
async fn the_parity_replace_person_fixture_is_platform_independent() {
    for os in ["macos", "windows", "linux"] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, os).await;
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "projectName": "Parity Project",
                "model": "wan_2_2",
                "mode": "replace_person",
                "prompt": "hero walks through rain",
                "sourceClipAssetId": "asset-video",
                "personTrackId": "track_fixture",
                "characterId": "character_fixture",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "on {os}: {body}");
        assert_eq!(
            body["status"], "queued",
            "the parity fixture must enqueue identically on every platform, or its snapshot is \
             only reproducible on the lane that recorded it — on {os}: {body}"
        );
        assert!(
            body["error"].is_null(),
            "on {os} the parity fixture must carry no error: {body}"
        );
    }
}

/// **THE TERMINAL-STATE GUARD (sc-19570) — the property this story actually owns.** An MLX-only
/// pair submitted off-Mac must reach a terminal `failed` state with a legible reason, NOT sit
/// `queued` forever.
///
/// That hang is the real defect. `ltx_2_3` + `image_to_video` and the other nineteen were admitted
/// by sc-19504's (correct, platform-independent) gate, offered as Video Studio tabs off-Mac, and
/// then claimed by nothing — no `mlx` worker can register on Windows or Linux — leaving the job at
/// `queued` / "Waiting for an available worker." with no error and no terminal state. None of the
/// four pre-existing sweeps rescues it: `fail_stranded_candle_jobs` bails the instant any live
/// candle worker exists (the job is unclaimable, not unserved), its `mlx` twin is inert off-Mac,
/// and both `fail_unsupported_*` sweeps default to warn.
///
/// The current stranded-pair table drives it, on BOTH off-Mac platforms, so coverage does not shrink
/// when the refusal moved off the HTTP boundary. Three further arms keep it honest:
///   * the terminal state is asserted on the enqueue RESPONSE, proving it does not wait for a
///     worker poll — the deployments that need this most are the ones where no worker ever polls;
///   * the failure names WHICH reason (`platform_unreachable:`), because a `status == "failed"`
///     assertion alone would be satisfied by any unrelated failure path;
///   * the same pairs stay `queued` on macOS, and the candle-served pairs stay `queued` off-Mac,
///     so a sweep that failed everything cannot pass.
#[tokio::test]
async fn an_mlx_only_video_job_reaches_a_terminal_failed_state_off_mac() {
    let stranded = mlx_only_stranded_pairs();
    let served = candle_served_pairs();

    for os in ["windows", "linux"] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, os).await;
        for (model, case) in &stranded {
            let mode = case["mode"].as_str().expect("case names a mode");
            let body_json = video_job_body(&project_id, model, case);
            let (status, body) =
                request(app.clone(), "POST", "/api/v1/video/jobs", body_json).await;
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{model} + {mode} on {os}: {body}"
            );
            assert_eq!(
                body["status"], "failed",
                "{model} + {mode} on {os} has no lane here — the job must TERMINATE, not sit \
                 queued waiting for a worker that can never exist: {body}"
            );
            // WHICH failure, not merely "a failure". Every one of these requests is well-formed, so
            // a bare `status == failed` assertion could be satisfied by an unrelated terminal path
            // and would still be green if the platform verdict never ran.
            let error = body["error"].as_str().unwrap_or_default();
            assert!(
                error.starts_with("platform_unreachable: "),
                "{model} + {mode} on {os} must fail for the PLATFORM reason, not some other \
                 terminal path: {body}"
            );
            assert!(
                error.contains(model) && error.contains(mode) && error.contains(os),
                "{model} + {mode} on {os}: the reason must name the model, the mode and the host \
                 so the job card explains itself: {error}"
            );
            // Terminal means terminal: re-reading the job returns the same failed state, so this is
            // a persisted transition and not a response-only decoration.
            let job_id = body["id"].as_str().expect("job id");
            let (status, reread) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{reread}");
            assert_eq!(
                reread["status"], "failed",
                "{model} + {mode} on {os}: the terminal state must be PERSISTED: {reread}"
            );
            assert_eq!(reread["error"], body["error"]);
        }

        // The sweep is not "fail everything off-Mac": a candle-served pair stays queued on the same
        // host, in the same run, waiting for the worker that will claim it.
        for (model, case) in &served {
            let mode = case["mode"].as_str().expect("case names a mode");
            let body_json = video_job_body(&project_id, model, case);
            let (status, body) =
                request(app.clone(), "POST", "/api/v1/video/jobs", body_json).await;
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{model} + {mode} on {os}: {body}"
            );
            assert_eq!(
                body["status"], "queued",
                "{model} + {mode} is candle-served on {os} and must stay claimable: {body}"
            );
        }
    }

    // …and on a Mac every stranded pair stays QUEUED, because the MLX engine renders it there.
    // Without this arm the assertions above would be satisfied by a sweep that failed these pairs
    // on every platform — which would break the Mac to fix Windows.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, "macos").await;
    for (model, case) in &stranded {
        let mode = case["mode"].as_str().expect("case names a mode");
        let body_json = video_job_body(&project_id, model, case);
        let (status, body) = request(app.clone(), "POST", "/api/v1/video/jobs", body_json).await;
        assert_eq!(status, StatusCode::CREATED, "{model} + {mode}: {body}");
        assert_eq!(
            body["status"], "queued",
            "{model} + {mode} renders on macOS and must stay claimable there: {body}"
        );
        assert!(
            body["error"].as_str().unwrap_or_default().is_empty(),
            "{model} + {mode} on macOS must carry no error: {body}"
        );
    }
}

/// The claim-path arm of the same sweep (sc-19570). `POST /api/v1/video/jobs` terminates an
/// unreachable job inline, but that route is not the only way a job reaches `queued`: **retry**
/// and **duplicate** re-queue an existing job without passing through it, and a job already sitting
/// `queued` from a build that predates this sweep never saw it at all.
///
/// So the store method also runs on every `POST /api/v1/jobs/claim`. This drives the retry door: a
/// stranded pair is submitted (terminal off-Mac), retried back to `queued` with no reachability
/// check anywhere on that path, and must be terminal again after a single claim.
#[tokio::test]
async fn a_requeued_unreachable_job_is_failed_by_the_claim_sweep() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, "windows").await;

    // `krea_realtime_14b`, not `ltx_2_3`: syncing `main` gave the LTX pair a candle lane, so it is
    // no longer stranded off-Mac and this test would assert `failed` on a job that is correctly
    // `queued`. Krea Realtime has no candle generator at all and stays in the stranded set.
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "prompt": "a fox runs",
            "model": "krea_realtime_14b",
            "mode": "image_to_video",
            "sourceAssetId": "img-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["status"], "failed", "{created}");
    let job_id = created["id"].as_str().expect("job id").to_owned();

    // Retry re-queues verbatim — no video validation, no reachability check. Without the claim-path
    // arm this is the hang, reopened one button-press later.
    let (status, retried) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/retry"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{retried}");
    assert_eq!(
        retried["status"], "queued",
        "retry re-queues without a reachability check, which is why the claim path needs the \
         sweep: {retried}"
    );

    let retry_id = retried["id"].as_str().expect("retry id").to_owned();
    assert_ne!(retry_id, job_id, "retry creates a fresh queued row");

    // A LIVE, capable candle worker polls — the realistic off-Mac deployment, and the exact
    // condition under which `fail_stranded_candle_jobs` declines to act. The job is unclaimable,
    // not unserved, so only the reachability sweep can terminate it.
    let (status, registered) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": "worker-sweep-probe",
            "gpuId": "0",
            "gpuName": "Test GPU",
            "capabilities": ["gpu", "candle", "video_generate"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{registered}");
    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({ "workerId": "worker-sweep-probe" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{claimed}");

    let (status, after) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{retry_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{after}");
    assert_eq!(
        after["status"], "failed",
        "a queued job no lane on this host can claim must be swept terminal at claim time: {after}"
    );
    assert!(
        after["error"]
            .as_str()
            .unwrap_or_default()
            .starts_with("platform_unreachable: "),
        "the claim sweep must fail it for the PLATFORM reason: {after}"
    );
    // The claim itself must not have HANDED the unreachable job to the worker — the sweep runs
    // first, so there is nothing left for `claim_next_job_routed` to return.
    assert!(
        claimed["job"].is_null(),
        "the swept job must not also be claimed: {claimed}"
    );
}

/// **sc-19504 IS INTACT AND DISTINGUISHABLE (sc-19570).** The no-lane-ANYWHERE gate is a correct
/// `400` and stays one: a mode no backend implements is a malformed request on every host, so
/// refusing it is platform-independent. Only the platform-conditional refusal moved.
///
/// The two must never be collapsed again, so this asserts them side by side on the SAME host:
/// `wan_2_2_i2v_14b` + `first_last_frame` (no lane anywhere) is a 400 on macOS, Windows AND Linux
/// with the same wording, while `krea_realtime_14b` + `image_to_video` (no lane HERE) is a 201 on
/// all three.
/// A future edit that turns either into the other turns this red.
#[tokio::test]
async fn the_no_lane_anywhere_gate_still_400s_and_is_distinct_from_the_platform_case() {
    for os in ["macos", "windows", "linux"] {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let (app, project_id) = shipped_manifest_app_on_os(&temp_dir, os).await;

        // NO LANE ANYWHERE → 400, identically on every platform.
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "prompt": "a fox runs",
                "model": "wan_2_2_i2v_14b",
                "mode": "first_last_frame",
                "sourceAssetId": "img-1",
                "lastFrameAssetId": "img-2",
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "the sc-19504 gate is platform-INdependent and must still refuse on {os}: {body}"
        );
        assert_eq!(
            body["detail"],
            "wan_2_2_i2v_14b cannot render the \"first_last_frame\" mode — no backend implements \
             it, so this job would wait for a worker that will never claim it. Choose a mode this \
             model lists in its capabilities, or a model that supports this one.",
            "the no-lane-anywhere refusal must keep its own wording on {os}, never the platform one"
        );
        assert!(
            !body["detail"]
                .as_str()
                .unwrap_or_default()
                .contains("platform"),
            "the two refusals must stay distinguishable to a reader on {os}: {body}"
        );

        // NO LANE *HERE* → 201 on every platform, with the verdict on the job instead.
        // `krea_realtime_14b` since the `main` sync: LTX gained a candle lane and so is served
        // everywhere now, which would make this arm assert nothing about the platform case.
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "prompt": "a fox runs",
                "model": "krea_realtime_14b",
                "mode": "image_to_video",
                "sourceAssetId": "img-1",
            }),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::CREATED,
            "a pair some lane serves must never be refused by STATUS CODE on {os}: {body}"
        );
        let expected_status = if os == "macos" { "queued" } else { "failed" };
        assert_eq!(
            body["status"], expected_status,
            "on {os} the platform verdict belongs on the job, not the response code: {body}"
        );
    }
}

/// The `candleSupport` block itself (sc-19570), read off the real `GET /api/v1/models` response —
/// the off-Mac twin of `macSupport`, and what `candleVideoModeBlock` in the web client reads.
///
/// Emitted on EVERY platform (the client decides whether to act on it from `candleGatingActive`),
/// so this asserts it from a macOS test run too. A block that only appeared off-Mac could never be
/// asserted by the macOS rust lane — which is how the off-Mac half of this defect stayed invisible.
#[tokio::test]
async fn the_models_endpoint_carries_a_candle_support_block_for_every_video_model() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let (app, _project_id) = shipped_manifest_app(&temp_dir).await;
    let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{models}");
    let models = models["models"]
        .as_array()
        .or_else(|| models.as_array())
        .expect("models list");
    let mut video_models = std::collections::BTreeSet::<String>::new();
    for model in models {
        if model["type"].as_str() != Some("video") {
            continue;
        }
        let id = model["id"].as_str().expect("model id");
        video_models.insert(id.to_owned());
        let candle = &model["candleSupport"];
        assert!(
            candle.is_object(),
            "{id}: every video model must carry a candleSupport block"
        );
        // The block must AGREE with the routing predicate, per mode. Restating the verdict here
        // would assert nothing; deriving it from `model_candle_support` is what makes a routing
        // change move this guard with it.
        let expected = serde_json::to_value(sceneworks_core::jobs_store::model_candle_support(
            id, "video",
        ))
        .expect("candle support serializes");
        assert_eq!(
            *candle, expected,
            "{id}: the serialized candleSupport drifted from the predicate"
        );
    }
    // The population is DERIVED from the shipped manifest, never pinned to a number.
    //
    // `GET /api/v1/models` filters the catalog with `retain_models_for_os(std::env::consts::OS)` —
    // the RUNNER's real OS, not the fixture's `host_os` — so the two lanes that run this suite
    // legitimately see different video sets: `ltx_2_3_eros` is a product withdrawal off-Mac
    // (sc-18902), so the macOS workspace job sees it and `parity-rust` on Linux does not. A literal
    // floor was therefore green on one lane and red on the other for no defect at all, and it also
    // went stale every time the catalog grew (MiniMax-H3's two entries, here).
    //
    // Re-deriving through the same withdrawal predicate the endpoint uses keeps the guard exact on
    // both lanes and moves it with the catalog instead of breaking on it.
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    let expected_video_models = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .filter(|entry| entry["type"].as_str() == Some("video"))
        .filter_map(|entry| {
            let id = entry["id"].as_str().expect("model id");
            (!crate::models::video_model_withdrawn_on_platform(id, entry, std::env::consts::OS))
                .then(|| id.to_owned())
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        video_models, expected_video_models,
        "the catalog's video rows must be exactly the shipped video entries this platform still \
         serves — a mismatch means the catalog read is wrong and every assertion above it is \
         vacuous"
    );
    assert!(
        video_models.len() >= 8,
        "only {} video models were checked — the shipped catalog cannot have shrunk this far, so \
         the manifest read itself is wrong",
        video_models.len()
    );

    // ...and the split itself, BY NAME, because a derived set is only as discriminating as the
    // predicate it derives from: silently swapping which entries the withdrawal covers would keep
    // the equality above green. Exactly one shipped video product is platform-split — the sc-18902
    // `ltx_2_3_eros` withdrawal — and MiniMax-H3's two partitions are NOT part of it. Both are
    // listed on every platform (their off-Mac gap is a LANE gap, reported in `candleSupport`
    // above, which is a different thing from a catalog withdrawal); asserting that here is what
    // keeps "epic 17137's entries went missing off-Mac" from ever being mistaken for this split.
    assert_eq!(
        video_models.contains("ltx_2_3_eros"),
        std::env::consts::OS == "macos",
        "`ltx_2_3_eros` is the one off-Mac product withdrawal: listed on macOS, absent elsewhere"
    );
    for id in ["minimax_h3", "minimax_h3_ref"] {
        assert!(
            video_models.contains(id),
            "{id} must be listed on EVERY platform — it carries no `macOnly` and is not a product \
             withdrawal, so a missing row here is a catalog defect, not a platform split"
        );
    }

    // The two gating switches the client reads, and the reason `candleGatingActive` is
    // platform-intrinsic rather than the `candle_required` rollout flag: the pairs it hides are
    // unreachable off-Mac whether or not a deployment opted into terminal gap reporting.
    let caps = |os: &str| sceneworks_core::jobs_store::mac_capabilities(os, false);
    assert!(!caps("macos").candle_gating_active, "inert on a Mac");
    assert!(!caps("darwin").candle_gating_active, "inert on the alias");
    assert!(caps("windows").candle_gating_active, "engaged on Windows");
    assert!(caps("linux").candle_gating_active, "engaged on Linux");
}

/// The withdrawal itself (sc-19504), read off the SHIPPED manifest bytes rather than restated: the
/// `wan_2_2_i2v_14b` entry must not advertise `first_last_frame` in `capabilities` OR in
/// `ui.recommendedFor` — the Video Studio builds its mode tabs from the first and highlights from
/// the second, and off-Mac (where the Mac gate is inactive) `capabilities` is the ONLY thing
/// standing between a user and the tab that hangs.
///
/// Its siblings assert the CLASS; this asserts the specific advertisement stays withdrawn, so
/// re-adding the string is red here as well as in the class guard.
#[test]
fn the_14b_i2v_no_longer_advertises_first_last_frame() {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    let entry = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "wan_2_2_i2v_14b")
        .expect("wan_2_2_i2v_14b is a shipped model");

    assert_eq!(
        entry["capabilities"],
        json!(["image_to_video", "extend_clip", "video_bridge"]),
        "the three modes a lane genuinely serves — `extend_clip` / `video_bridge` via the candle \
         Wan-VACE engine, `image_to_video` on both lanes. `first_last_frame` is not one of them."
    );
    assert_eq!(
        entry["ui"]["recommendedFor"],
        json!(["image_to_video", "extend_clip", "video_bridge"]),
        "`recommendedFor` must track `capabilities`: it is the second array the studio reads, so a \
         mode left here re-surfaces the withdrawn advertisement"
    );
}

/// sc-17159 — the resolved-default write-back preserves the manifest's own decimal.
///
/// `contract_number`'s fractional branch had never carried a shipped value: every video model
/// before MiniMax-H3 declares a whole-number `defaults.duration`, so the branch existed only for
/// the integral case's sake. The first fractional default exposed it — `Value::from(5.1667_f32)`
/// widens to `f64` and enqueues `5.1666998863220215`, which is not equal to ANY of the fourteen
/// entries in that model's `limits.durations`. The duration dropdown preselects by comparing
/// against that menu and a recipe replay is compared against the enqueued row, so the value has to
/// be the declared one, not merely a number that rounds to it.
#[test]
fn a_fractional_resolved_duration_keeps_the_manifests_own_decimal() {
    // The value under test, and the assertion that makes it load-bearing: it must equal the
    // manifest's own number, which the f64-widened form does not.
    assert_eq!(crate::generation::contract_number(5.1667), json!(5.1667));
    assert_ne!(json!(5.1666998863220215), json!(5.1667));
    // Every other lattice rung the menu offers.
    for rung in [
        5.875_f32, 6.5833, 7.2917, 8.7083, 9.4167, 10.125, 10.8333, 11.5417, 12.25, 12.9583,
        13.6667, 14.375,
    ] {
        let enqueued = crate::generation::contract_number(rung);
        assert_eq!(
            enqueued.as_f64().map(|v| v as f32),
            Some(rung),
            "{rung}: the enqueued row must read back as the same f32"
        );
        assert!(
            enqueued.to_string().len() <= rung.to_string().len(),
            "{rung}: enqueued as {enqueued}, which is longer than the declared decimal"
        );
    }
    // The integral branch is untouched: a whole-number default stays an INT on the wire, because
    // `duration` is a `ContractNumber` that carries int-vs-float across it (the sc-12400 contract).
    assert_eq!(crate::generation::contract_number(5.0), json!(5));
    assert_eq!(crate::generation::contract_number(6.0), json!(6));
    assert!(crate::generation::contract_number(4.0).is_i64());
}

/// **sc-19563 — the declared-partition refusal is reachable from a REAL request.**
///
/// The unit tests in `loras.rs` call `validate_lora_specs_for_model` directly. That proves the
/// function refuses; it does not prove a user can ever reach it. This one goes through the actual
/// route — `POST /api/v1/video/jobs` — with the manifests on disk, the adapter file on disk and the
/// app assembled by `create_app`, and asserts the exact body a client receives.
///
/// The epic's own rule is why this arm exists at all: **declaration ≠ enforcement ≠ reachability.**
/// `loraCompatibility` declared the relationship, the validator enforces it, and only a submission
/// shows the enforcement is on the path a request takes.
///
/// Both directions, plus both positive controls. The controls are load-bearing twice over: they
/// prove the gate is not refusing every H3 LoRA, and they prove the fixture is well-formed enough to
/// reach the LoRA check at all.
#[tokio::test]
async fn cross_selecting_a_minimax_h3_partition_lora_is_refused_by_the_video_job_route() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let config_dir = temp_dir.path().join("config/manifests");
    std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
    // Both H3 partitions, declared exactly as `builtin.models.jsonc` declares them: ONE family,
    // ONE architecture. That identity is the point — the family check passes both ways, so the
    // refusal below can only come from the declared-partition gate.
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "minimax_h3",
              "name": "MiniMax-H3",
              "family": "minimax-h3",
              "type": "video",
              "adapter": "minimax_h3",
              "capabilities": ["text_to_video", "image_to_video", "first_last_frame"],
              "downloads": [],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["minimax-h3"] },
              "ui": {}
            },
            {
              "id": "minimax_h3_ref",
              "name": "MiniMax-H3 References",
              "family": "minimax-h3",
              "type": "video",
              "adapter": "minimax_h3",
              "capabilities": ["text_to_video", "reference_to_video"],
              "downloads": [],
              "paths": {},
              "defaults": {},
              "limits": {},
              "loraCompatibility": { "families": ["minimax-h3"] },
              "ui": {}
            }
          ]
        }
        "#,
    )
    .expect("builtin models writes");
    std::fs::write(
        config_dir.join("user.models.jsonc"),
        r#"{ "schemaVersion": 1, "models": [] }"#,
    )
    .expect("user models writes");
    // The two adapters, each declaring the ONE partition it is distilled for — the same shape
    // `builtin.loras.jsonc` now ships.
    std::fs::write(
        config_dir.join("builtin.loras.jsonc"),
        r#"
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "minimax_h3_turbo_8step",
              "name": "MiniMax-H3 Turbo (8-step)",
              "family": "minimax-h3",
              "modelIds": ["minimax_h3"],
              "triggerWords": [],
              "compatibility": { "families": ["minimax-h3"] },
              "source": { "provider": "local", "path": "loras/h3_fl2v.safetensors" }
            },
            {
              "id": "minimax_h3_ref2v_turbo_4step",
              "name": "MiniMax-H3 Ref2VA Turbo (4-step)",
              "family": "minimax-h3",
              "modelIds": ["minimax_h3_ref"],
              "triggerWords": [],
              "compatibility": { "families": ["minimax-h3"] },
              "source": { "provider": "local", "path": "loras/h3_ref2v.safetensors" }
            }
          ]
        }
        "#,
    )
    .expect("builtin loras writes");
    std::fs::write(
        config_dir.join("user.loras.jsonc"),
        r#"{ "schemaVersion": 1, "loras": [] }"#,
    )
    .expect("user loras writes");

    let lora_dir = temp_dir.path().join("data/loras");
    std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
    // Real MiniMax-H3 diffusers keys, so `detect_lora_family` reports `minimax-h3` and the
    // detected-family check is exercised rather than skipped on a `None`.
    let keys: Vec<String> = ["attn.to_q", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
        .iter()
        .flat_map(|target| {
            ["lora_A.default.weight", "lora_B.default.weight"]
                .iter()
                .flat_map(move |suffix| {
                    [
                        format!("transformer_blocks.0.{target}.{suffix}"),
                        format!("token_refiner.refiner_blocks.0.{target}.{suffix}"),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect();
    write_test_safetensors_with_keys(&lora_dir.join("h3_fl2v.safetensors"), &keys);
    write_test_safetensors_with_keys(&lora_dir.join("h3_ref2v.safetensors"), &keys);

    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "H3 partitions" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");

    let submit = |model: &'static str, lora: &'static str| {
        let app = app.clone();
        let project_id = project_id.to_owned();
        async move {
            request(
                app,
                "POST",
                "/api/v1/video/jobs",
                json!({
                    "projectId": project_id,
                    "mode": "text_to_video",
                    "prompt": "a cellist on a rooftop",
                    "model": model,
                    "loras": [{ "id": lora, "weight": 1.0 }]
                }),
            )
            .await
        }
    };

    // ── the ref2v adapter on the fl2v partition.
    let (status, body) = submit("minimax_h3", "minimax_h3_ref2v_turbo_4step").await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "cross-selecting must be refused at submit; got {body:?}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("minimax_h3_ref2v_turbo_4step"),
        "the message must name the LoRA; got {detail}"
    );
    // The exact phrase, not a substring search: `minimax_h3_ref2v_turbo_4step` CONTAINS
    // `minimax_h3_ref`, so `detail.contains("minimax_h3_ref")` is satisfied by the LoRA id already
    // in the message and asserts nothing about the partition.
    assert!(
        detail.contains("is declared for model minimax_h3_ref"),
        "the message must name the partition it IS for; got {detail}"
    );

    // ── and the reverse.
    let (status, body) = submit("minimax_h3_ref", "minimax_h3_turbo_8step").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "got {body:?}");
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(detail.contains("minimax_h3_turbo_8step"), "{detail}");
    assert!(
        detail.contains("is declared for model minimax_h3,"),
        "{detail}"
    );

    // ── the controls. Each adapter on its OWN partition must NOT be refused for a LoRA reason.
    //    A later gate in `create_video_job` (the sc-19504 no-lane check) may still stop a job in a
    //    test environment with no worker, so the assertion is on the REASON rather than on a 201:
    //    what this proves is that the LoRA gate let it past, which is exactly the control needed.
    for (model, lora) in [
        ("minimax_h3", "minimax_h3_turbo_8step"),
        ("minimax_h3_ref", "minimax_h3_ref2v_turbo_4step"),
    ] {
        let (status, body) = submit(model, lora).await;
        let detail = body["detail"].as_str().unwrap_or_default().to_owned();
        assert!(
            !detail.contains("is declared for model"),
            "{lora} on its OWN partition {model} must not trip the partition gate; got \
             {status} {detail}"
        );
    }
}
