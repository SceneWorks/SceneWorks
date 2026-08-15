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

/// The other half of the guard: the job types the generic route legitimately serves keep
/// working. `image_upscale` / `image_detail` are the real web callers (batch ops), and
/// neither has a typed door — so the sc-12305 rejection must not touch them.
#[tokio::test]
async fn generic_jobs_route_still_serves_non_generation_types() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for job_type in ["image_upscale", "image_detail"] {
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
    }
}

/// sc-13617 / F-055: raw jobs whose `model` reaches worker-side model path resolution must
/// reject traversal before a job row exists. This table is the API-side inventory for that key.
#[tokio::test]
async fn generic_model_backed_jobs_reject_path_unsafe_model_before_create() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    for job_type in ["image_upscale", "prompt_refine"] {
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
        3,
        "only the original and two clean operations may persist"
    );
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

    // 9 + 2 + 1 = 12. ACCEPTED, at the cap, and every list reaches the enqueued payload.
    let (status, job) = submit("minimax_h3_ref", 9, 2, 1).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(job["type"], "video_generate");
    assert_eq!(
        job["payload"]["referenceAssetIds"].as_array().map(Vec::len),
        Some(9)
    );
    assert_eq!(
        job["payload"]["sourceClipAssetIds"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        job["payload"]["referenceAudioAssetIds"],
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
///   * `ltx_2_3_eros`      — menu `[8]`, the LTX-2.3 shape
///
/// Both ids are REAL routed video models, exactly as the floor test above uses `mochi_1`. The menu
/// model cannot be a made-up id: the sc-19504 no-lane gate runs last on the enqueue path and refuses
/// any video request no backend's claim predicate accepts, so a synthetic id 400s for
/// "no backend implements it" and the ON-menu `201` arm below — the one that keeps the rejections
/// above from passing for a gate that simply refuses every step count — can never be reached.
/// `ltx_2_3_eros` is `video_mlx_routed` + `candle_video_routed` and serves `text_to_video`, and it is
/// one of the two models this story actually declares `limits.steps` for.
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
              "id": "ltx_2_3_eros",
              "name": "Distilled Vid",
              "family": "ltx-video",
              "type": "video",
              "adapter": "ltx_video",
              "capabilities": ["text_to_video"],
              "downloads": [
                { "provider": "huggingface", "repo": "owner/ltx_2_3_eros", "files": ["*.safetensors"], "default": true }
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
              "model": "ltx_2_3_eros",
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
        "30 steps is off ltx_2_3_eros's exact menu and must be refused at enqueue: {body}"
    );
    let detail = body["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("ltx_2_3_eros"),
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
    assert_eq!(body["payload"]["model"], "ltx_2_3_eros");
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
        .find(|model| model["id"] == "ltx_2_3_eros")
        .expect("ltx_2_3_eros is listed");
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

    // The conditioning media reaches the worker verbatim rather than merely validating.
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
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(
        ref2va["payload"]["referenceAssetIds"],
        json!(["img-0", "img-1"])
    );
    assert_eq!(ref2va["payload"]["sourceClipAssetIds"], json!(["clip-0"]));
    assert_eq!(
        ref2va["payload"]["referenceAudioAssetIds"],
        json!(["aud-0"])
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
    assert_eq!(
        status,
        StatusCode::CREATED,
        "audio alongside a visual reference is the shape Ref2VA serves: {audio_with_image}"
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
/// ALL TWENTY are listed, not a sample and not only the thirteen the story measured: the three
/// mac-only-download families (`krea_realtime_14b`, `minimax_h3`, `minimax_h3_ref`) ride the same
/// route, because "install state is not a reachability gate" is only an argument until the REST leg
/// actually exercises it.
fn mlx_only_stranded_pairs() -> Vec<(&'static str, Value)> {
    vec![
        (
            "ltx_2_3",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "ltx_2_3",
            json!({ "mode": "first_last_frame", "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "ltx_2_3",
            json!({ "mode": "extend_clip", "sourceClipAssetId": "clip-1" }),
        ),
        (
            "ltx_2_3",
            json!({ "mode": "video_bridge", "sourceClipAssetId": "clip-1", "bridgeRightClipAssetId": "clip-2" }),
        ),
        (
            "ltx_2_3",
            json!({ "mode": "replace_person", "sourceClipAssetId": "clip-1", "personTrackId": "track-1", "characterId": "char-1" }),
        ),
        (
            "ltx_2_3_eros",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "ltx_2_3_eros",
            json!({ "mode": "first_last_frame", "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "ltx_2_3_eros",
            json!({ "mode": "extend_clip", "sourceClipAssetId": "clip-1" }),
        ),
        (
            "ltx_2_3_eros",
            json!({ "mode": "video_bridge", "sourceClipAssetId": "clip-1", "bridgeRightClipAssetId": "clip-2" }),
        ),
        (
            "ltx_2_3_eros",
            json!({ "mode": "replace_person", "sourceClipAssetId": "clip-1", "personTrackId": "track-1", "characterId": "char-1" }),
        ),
        (
            "wan_2_2",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
        ),
        (
            "wan_2_2",
            json!({ "mode": "first_last_frame", "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "wan_2_2_vace_fun_14b",
            json!({ "mode": "replace_person", "sourceClipAssetId": "clip-1", "personTrackId": "track-1", "characterId": "char-1" }),
        ),
        // The seven pairs the story's measurement did NOT list, because those three families ship
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
        (
            "wan_2_2_i2v_14b",
            json!({ "mode": "image_to_video", "sourceAssetId": "img-1" }),
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
/// byte-identical bodies on macOS, Windows and Linux alike — for the twenty MLX-only pairs AND for
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
    assert_eq!(
        stranded.len(),
        20,
        "the measured stranded set is twenty pairs — a shrunken table would narrow every guard \
         that reads it"
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
/// The whole twenty-pair table drives it, on BOTH off-Mac platforms, so coverage did not shrink
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

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/video/jobs",
        json!({
            "projectId": project_id,
            "prompt": "a fox runs",
            "model": "ltx_2_3",
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
/// with the same wording, while `ltx_2_3` + `image_to_video` (no lane HERE) is a 201 on all three.
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
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": project_id,
                "prompt": "a fox runs",
                "model": "ltx_2_3",
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
    let mut video_models = 0_usize;
    for model in models {
        if model["type"].as_str() != Some("video") {
            continue;
        }
        video_models += 1;
        let id = model["id"].as_str().expect("model id");
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
    assert!(
        video_models >= 12,
        "only {video_models} video models were checked — the catalog read is wrong and this guard \
         is vacuous"
    );

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
