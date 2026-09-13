//! End-to-end tests for the local filmmaking harness (sc-22710): the REAL API routes in-process
//! (`create_app_with_state`), a scripted fake worker that claims the jobs through the worker API
//! exactly as the GPU worker would, and the harness driving both through `ApiTransport`.
//!
//! What the fake worker replaces is only the render: it claims `video_generate` / `timeline_export`
//! jobs, writes a placeholder file where the real worker would write the MP4, and reports the same
//! `assetWrites` / `assetIds` result shapes the real worker reports. Asset persistence, timeline
//! validation, export dispatch and every enqueue gate are the production code paths.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use parking_lot::Mutex;
use sceneworks_core::film_plan::{RunOutcome, ShotOutcome};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::film_harness::{
    self, ApiRequest, ApiResponse, ApiTransport, HarnessError, RequestBody, RunOptions,
    TransportFuture, FIXTURE_REFERENCES,
};
use crate::tests::support::{create_app_with_state, request, test_settings};

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop"
);

/// [`ApiTransport`] over the in-process router: the same `oneshot` driver every route test uses.
struct RouterTransport {
    app: axum::Router,
}

impl ApiTransport for RouterTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        let app = self.app.clone();
        Box::pin(async move {
            let mut builder = Request::builder()
                .method(request.method)
                .uri(request.path.clone());
            let body = match request.body {
                RequestBody::None => Body::empty(),
                RequestBody::Json(value) => {
                    builder = builder.header("content-type", "application/json");
                    Body::from(value.to_string())
                }
                RequestBody::Multipart { boundary, bytes } => {
                    builder = builder.header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    );
                    Body::from(bytes)
                }
            };
            let response = app
                .oneshot(builder.body(body).expect("request builds"))
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .map_err(|error| HarnessError::Transport(error.to_string()))?;
            let body = if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap_or(Value::Null)
            };
            Ok(ApiResponse { status, body })
        })
    }
}

/// How the fake worker treats one video job, keyed by the shot id the harness stamps into
/// `advanced.filmHarness.shotId`.
#[derive(Debug, Clone, Copy)]
enum VideoBehavior {
    /// Complete after `delay`, reporting `peak_pct` as the observed GPU memory peak.
    Complete { delay_secs: u64, peak_pct: f64 },
    /// Fail once (first attempt), then complete.
    FailFirst,
    /// Never complete; honour a cancel request by reporting `canceled`.
    Hang,
}

#[derive(Debug, Clone, Default)]
struct WorkerScript {
    behaviors: Vec<(String, VideoBehavior)>,
    /// Jobs the fake worker has claimed, in order: (type, job id, payload).
    claimed: Vec<(String, String, Value)>,
    failed_once: Vec<String>,
}

impl WorkerScript {
    fn behavior_for(&self, shot_id: &str) -> VideoBehavior {
        self.behaviors
            .iter()
            .find(|(id, _)| id == shot_id)
            .map(|(_, behavior)| *behavior)
            .unwrap_or(VideoBehavior::Complete {
                delay_secs: 1,
                peak_pct: 40.0,
            })
    }
}

const WORKER_ID: &str = "fake-mlx-worker";
const HOST_MEMORY_MB: u64 = 128 * 1024;

async fn register_fake_worker(app: &axum::Router) {
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": WORKER_ID,
            "gpuId": "mlx",
            "gpuName": "Apple M-series (fake)",
            "capabilities": ["video_generate", "timeline_export", "frame_extract"],
            "loadedModels": [],
            "utilization": { "memoryTotalMb": HOST_MEMORY_MB }
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
}

fn project_path(
    app: &axum::Router,
    project_id: &str,
) -> impl std::future::Future<Output = PathBuf> {
    let app = app.clone();
    let project_id = project_id.to_owned();
    async move {
        let (_, project) = request(
            app,
            "GET",
            &format!("/api/v1/projects/{project_id}"),
            Value::Null,
        )
        .await;
        PathBuf::from(project["path"].as_str().expect("project path"))
    }
}

/// Spawn the scripted worker loop. It registers, then claims and completes jobs until the task is
/// aborted. Budgets in these tests are real seconds: tokio's paused clock is NOT used, because the
/// API's blocking store calls let the auto-advancing clock race ahead of the harness's own
/// `Instant::now()` reads and spend a plan's budget during setup.
fn spawn_fake_worker(
    app: axum::Router,
    script: Arc<Mutex<WorkerScript>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        register_fake_worker(&app).await;
        loop {
            let (status, claim) = request(
                app.clone(),
                "POST",
                "/api/v1/jobs/claim",
                json!({ "workerId": WORKER_ID }),
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::OK, "{claim}");
            let Some(job) = claim.get("job").filter(|job| !job.is_null()).cloned() else {
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            };
            let job_id = job["id"].as_str().expect("job id").to_owned();
            let job_type = job["type"].as_str().expect("job type").to_owned();
            script
                .lock()
                .claimed
                .push((job_type.clone(), job_id.clone(), job["payload"].clone()));
            match job_type.as_str() {
                "video_generate" => run_fake_video_job(&app, &script, &job_id, &job).await,
                "timeline_export" => run_fake_export_job(&app, &job_id, &job).await,
                other => panic!("fake worker claimed an unexpected job type {other}"),
            }
        }
    })
}

async fn post_progress(app: &axum::Router, job_id: &str, body: Value) -> Value {
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        body,
    )
    .await;
    assert!(
        status == axum::http::StatusCode::OK || status == axum::http::StatusCode::CONFLICT,
        "progress {job_id}: {status} {response}"
    );
    response
}

async fn run_fake_video_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let shot_id = payload["advanced"]["filmHarness"]["shotId"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let behavior = script.lock().behavior_for(&shot_id);
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    post_progress(
        app,
        job_id,
        json!({
            "status": "running", "stage": "generating", "progress": 0.2,
            "message": "fake render", "workerId": WORKER_ID, "backend": "mlx"
        }),
    )
    .await;
    match behavior {
        VideoBehavior::Hang => loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let (_, snapshot) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            if snapshot["cancelRequested"].as_bool() == Some(true) {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "canceled", "stage": "canceled", "progress": 1,
                        "message": "Canceled by user.", "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        },
        VideoBehavior::FailFirst => {
            let first = {
                let mut script = script.lock();
                if script.failed_once.contains(&shot_id) {
                    false
                } else {
                    script.failed_once.push(shot_id.clone());
                    true
                }
            };
            if first {
                post_progress(
                    app,
                    job_id,
                    json!({
                        "status": "failed", "stage": "failed", "progress": 1,
                        "message": "fake engine fault", "error": "fake engine fault: transient",
                        "workerId": WORKER_ID
                    }),
                )
                .await;
                return;
            }
        }
        VideoBehavior::Complete { delay_secs, .. } => {
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
        }
    }
    let peak_pct = match behavior {
        VideoBehavior::Complete { peak_pct, .. } => peak_pct,
        _ => 40.0,
    };
    let asset_id = format!("asset_{}", &job_id.replace('-', "")[..16]);
    let media_rel = format!("assets/videos/{asset_id}.mp4");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/videos")).expect("videos dir");
    std::fs::write(project_dir.join(&media_rel), b"not really an mp4").expect("fake mp4");
    let duration = payload["duration"].as_f64().unwrap_or(5.0);
    let fps = payload["fps"].as_u64().unwrap_or(24);
    let frames = (duration * fps as f64).round() as u64;
    let fact = json!({
        "type": "video",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "video/mp4",
        "width": payload["width"], "height": payload["height"],
        "duration": payload["duration"], "fps": payload["fps"],
        "encodedFrameCount": frames, "encodedDuration": duration, "encodedFps": fps,
        "hasAudio": true,
        "quality": payload["quality"],
        "family": "minimax-h3",
        "seed": payload["seed"].as_i64().unwrap_or(1),
        "displayName": format!("fake take {shot_id}"),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": payload["mode"], "model": payload["model"], "adapter": "fake_minimax_h3",
        "prompt": payload["prompt"], "negativePrompt": "", "loras": [],
        "rawAdapterSettings": { "tier": "q4", "task": if payload["sourceAssetId"].is_string() { "fl2va" } else { "t2va" }, "advanced": payload["advanced"] },
        "sourceAssetId": payload["sourceAssetId"], "lastFrameAssetId": payload["lastFrameAssetId"],
        "fitMode": payload["fitMode"],
        "sourceClipAssetIds": [], "referenceAssetIds": payload["referenceAssetIds"].as_array().cloned().unwrap_or_default(),
        "referenceAudioAssetIds": [],
        "timelineContext": {}
    });
    let genset_id = format!("genset_{}", &job_id.replace('-', "")[..16]);
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake render done", "workerId": WORKER_ID, "backend": "mlx",
            "peakGpuMemoryPct": peak_pct,
            "result": {
                "generationSetId": genset_id,
                "expectedCount": 1,
                "adapter": "fake_minimax_h3",
                "model": payload["model"],
                "generationSet": {
                    "id": genset_id, "mode": payload["mode"], "model": payload["model"],
                    "prompt": payload["prompt"], "negativePrompt": "", "count": 1,
                    "createdAt": sceneworks_core::time::utc_now()
                },
                "assetWrites": [fact]
            }
        }),
    )
    .await;
}

async fn run_fake_export_job(app: &axum::Router, job_id: &str, job: &Value) {
    let payload = &job["payload"];
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let render_rel = format!(
        "assets/renders/fake-export-{}.mp4",
        &job_id.replace('-', "")[..8]
    );
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/renders")).expect("renders dir");
    std::fs::write(project_dir.join(&render_rel), b"not really an mp4").expect("fake render");
    let asset_id = format!("asset_render_{}", &job_id.replace('-', "")[..12]);
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Timeline MP4 export saved.", "workerId": WORKER_ID,
            "result": {
                "assetIds": [asset_id],
                "assets": [{ "id": asset_id, "type": "render", "file": { "path": render_rel } }],
                "timelineId": payload["timelineId"],
                "renderPath": render_rel,
                "adapter": "ffmpeg_timeline"
            }
        }),
    )
    .await;
}

struct Harness {
    app: axum::Router,
    transport: RouterTransport,
    temp_dir: tempfile::TempDir,
    script: Arc<Mutex<WorkerScript>>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl Harness {
    async fn start(with_worker: bool, behaviors: Vec<(&str, VideoBehavior)>) -> Self {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        // `/api/v1/models` serves the manifests under the config dir, so seed the REAL shipped
        // builtin manifest: the harness validates the fixture against `minimax_h3`'s actual
        // declared menus, caps and memory minimum, not a stand-in.
        let manifests_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&manifests_dir).expect("manifest dir creates");
        std::fs::write(
            manifests_dir.join("builtin.models.jsonc"),
            include_str!("../../../../config/manifests/builtin.models.jsonc"),
        )
        .expect("builtin models writes");
        let (app, state) =
            create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
        // The fixture's image-conditioned shot is only MLX-routed; judge the enqueue as a Mac on
        // every lane so the suite means the same thing on ubuntu and on the hosted macOS job.
        *state.video_platform_override.lock() = Some("macos");
        let script = Arc::new(Mutex::new(WorkerScript {
            behaviors: behaviors
                .into_iter()
                .map(|(id, behavior)| (id.to_owned(), behavior))
                .collect(),
            ..WorkerScript::default()
        }));
        let worker = with_worker.then(|| spawn_fake_worker(app.clone(), script.clone()));
        if with_worker {
            // Wait for the worker to register before the harness looks for it — bounded, not a
            // fixed sleep, so a slow CI runner cannot turn this into a spurious refusal.
            let mut registered = false;
            for _ in 0..200 {
                let (_, workers) =
                    request(app.clone(), "GET", "/api/v1/workers", Value::Null).await;
                if workers
                    .as_array()
                    .is_some_and(|workers| workers.iter().any(|w| w["id"] == WORKER_ID))
                {
                    registered = true;
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            assert!(registered, "fake worker did not register within 5s");
        }
        Self {
            transport: RouterTransport { app: app.clone() },
            app,
            temp_dir,
            script,
            worker,
        }
    }

    fn options(
        &self,
        plan_path: PathBuf,
        pack_path: PathBuf,
        shots: Option<&[&str]>,
    ) -> RunOptions {
        RunOptions {
            plan_path,
            reference_pack_path: pack_path,
            project_id: None,
            shot_ids: shots.map(|ids| ids.iter().map(|id| (*id).to_owned()).collect()),
            out_dir: self.temp_dir.path().join("run-out"),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
        }
    }

    async fn jobs(&self) -> Vec<Value> {
        let (_, jobs) = request(self.app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        jobs.as_array().cloned().unwrap_or_default()
    }

    fn fixture_plan(&self) -> PathBuf {
        Path::new(FIXTURE_DIR).join("plan.jsonc")
    }

    fn fixture_pack(&self) -> PathBuf {
        Path::new(FIXTURE_DIR).join("references.jsonc")
    }

    /// Copy the checked-in fixture into the temp dir with `edit` applied to the parsed plan, so a
    /// test can break one field without touching the shipped documents.
    fn edited_plan(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_plan()).expect("fixture plan");
        let mut plan: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture plan parses");
        edit(&mut plan);
        let path = self.temp_dir.path().join("plan.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        path
    }

    fn run_record(&self) -> Value {
        let text = std::fs::read_to_string(self.temp_dir.path().join("run-out/run.json"))
            .expect("run.json written");
        serde_json::from_str(&text).expect("run.json parses")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.abort();
        }
    }
}

/// One line per attempt — the shot, status and error of every attempt plus the export — for
/// assertion messages, so a failing run explains itself without the full Debug dump.
fn summary(record: &sceneworks_core::film_plan::RunRecord) -> String {
    let mut lines = vec![format!("outcome={:?}", record.outcome)];
    for shot in &record.shots {
        lines.push(format!("{} {:?}", shot.shot_id, shot.outcome));
        for attempt in &shot.attempts {
            lines.push(format!(
                "  attempt {} job={} status={} error={:?} take={:?}",
                attempt.attempt,
                attempt.job_id.as_deref().unwrap_or("-"),
                attempt.status,
                attempt.error,
                attempt.take.as_ref().map(|take| take.asset_id.as_str())
            ));
        }
    }
    if let Some(export) = &record.export {
        lines.push(format!(
            "export job={} status={} error={:?}",
            export.job_id, export.status, export.error
        ));
    }
    lines.join("\n")
}

#[tokio::test]
async fn two_shot_run_renders_imports_assembles_and_exports_through_the_real_routes() {
    let harness = Harness::start(true, vec![]).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");

    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.selected_shot_ids, vec!["SH010", "SH020"]);
    let project_id = record.project_id.clone().expect("project created");

    // Every approved reference is a project asset, tagged with its role, addressable on its own.
    assert_eq!(record.references.len(), 7);
    for reference in &record.references {
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!(
                "/api/v1/projects/{project_id}/assets/{}",
                reference.asset_id
            ),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "image");
        assert_eq!(asset["extra"]["filmHarness"]["role"], reference.role);
        assert_eq!(
            asset["extra"]["filmHarness"]["referencePackId"],
            "courier-workshop-refs"
        );
        let tags: Vec<&str> = asset["tags"]
            .as_array()
            .map(|tags| tags.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            tags.contains(&format!("role:{}", reference.role).as_str()),
            "{tags:?}"
        );
    }

    // Two real jobs went through POST /api/v1/video/jobs and produced two persisted video assets.
    let rendered: Vec<_> = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::Rendered)
        .collect();
    assert_eq!(rendered.len(), 2, "{:#?}", record.shots);
    let not_selected = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::NotSelected)
        .count();
    assert_eq!(
        not_selected, 4,
        "the other four shots stay in the record, unrendered"
    );
    let claimed = harness.script.lock().claimed.clone();
    let video_jobs: Vec<_> = claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .collect();
    assert_eq!(video_jobs.len(), 2);
    for shot in &rendered {
        let attempt = shot.attempts.last().expect("an attempt");
        let job_id = attempt.job_id.clone().expect("job id");
        let take = attempt.take.as_ref().expect("a take");
        let (status, job) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/jobs/{job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(job["status"], "completed");
        assert_eq!(job["payload"]["model"], "minimax_h3");
        assert_eq!(job["payload"]["advanced"]["mlxQuantize"], 4);
        assert_eq!(
            job["payload"]["advanced"]["filmHarness"]["shotId"],
            shot.shot_id
        );
        assert_eq!(
            job["payload"]["advanced"]["filmHarness"]["runId"],
            record.run_id
        );
        assert_eq!(job["payload"]["fps"], 24);
        assert_eq!(job["payload"]["width"], 576);
        assert_eq!(job["payload"]["height"], 320);
        assert_eq!(job["result"]["assetIds"][0], take.asset_id);
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets/{}", take.asset_id),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "video");
        assert_eq!(
            asset["lineage"]["jobId"], job_id,
            "asset links back to its job"
        );
        assert_eq!(take.backend.as_deref(), Some("mlx"));
        assert_eq!(take.raw_adapter_settings["tier"], "q4");
        match shot.shot_id.as_str() {
            "SH010" => {
                assert_eq!(job["payload"]["mode"], "text_to_video");
                assert!(job["payload"]["sourceAssetId"].is_null());
            }
            "SH020" => {
                assert_eq!(job["payload"]["mode"], "image_to_video");
                let plate = record
                    .references
                    .iter()
                    .find(|reference| reference.role == "workshop_plate")
                    .expect("plate imported");
                assert_eq!(job["payload"]["sourceAssetId"], plate.asset_id);
                assert_eq!(
                    shot.conditioning_assets.first_frame_asset_id.as_deref(),
                    Some(plate.asset_id.as_str())
                );
            }
            other => panic!("unexpected rendered shot {other}"),
        }
    }
    let model = record.model.as_ref().expect("model record");
    assert_eq!(model.id, "minimax_h3");
    assert_eq!(model.tier_requested.as_deref(), Some("q4"));
    assert_eq!(model.backend_observed.as_deref(), Some("mlx"));
    assert_eq!(model.hardware.worker_id.as_deref(), Some(WORKER_ID));
    assert_eq!(model.hardware.host_memory_gb, Some(128.0));
    assert_eq!(
        model.weights.as_ref().and_then(|w| w["repo"].as_str()),
        Some("SceneWorks/minimax-h3-mlx")
    );

    // The timeline holds the two takes back to back and the export ran through timeline_export.
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    assert_eq!(timeline.items.len(), 2);
    assert_eq!(timeline.fps, 24);
    assert_eq!(timeline.aspect_ratio, "16:9");
    assert_eq!(timeline.items[0].shot_id, "SH010");
    assert_eq!(timeline.items[1].shot_id, "SH020");
    assert!((timeline.items[0].timeline_end - 5.1667).abs() < 1e-6);
    assert!((timeline.items[1].timeline_start - 5.1667).abs() < 1e-6);
    let (status, saved) = request(
        harness.app.clone(),
        "GET",
        &format!(
            "/api/v1/projects/{project_id}/timelines/{}",
            timeline.timeline_id
        ),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{saved}");
    let items = saved["tracks"][0]["items"]
        .as_array()
        .expect("main track items");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["assetId"], timeline.items[0].asset_id);
    assert_eq!(
        items[1]["versionHistory"][0]["jobId"],
        rendered[1].attempts[0].job_id.clone().unwrap()
    );
    let export = record.export.as_ref().expect("export ran");
    assert_eq!(export.status, "completed");
    assert!(export.asset_id.is_some());
    assert!(export
        .render_path
        .as_deref()
        .unwrap_or("")
        .starts_with("assets/renders/"));
    let export_jobs: Vec<_> = claimed
        .iter()
        .filter(|(kind, _, _)| kind == "timeline_export")
        .collect();
    assert_eq!(export_jobs.len(), 1);
    assert_eq!(export_jobs[0].2["timelineId"], timeline.timeline_id);
    assert_eq!(export_jobs[0].2["resolution"], 640);
    assert_eq!(export_jobs[0].2["fps"], 24);

    // The run record is on disk beside copies of both documents, and inside the project.
    let on_disk = harness.run_record();
    assert_eq!(on_disk["runId"], record.run_id);
    assert_eq!(on_disk["outcome"], "completed");
    assert_eq!(on_disk["plan"]["id"], "courier-workshop");
    assert_eq!(on_disk["referencePack"]["id"], "courier-workshop-refs");
    assert!(harness.temp_dir.path().join("run-out/plan.json").exists());
    assert!(harness
        .temp_dir
        .path()
        .join("run-out/references.json")
        .exists());
    let project_record = PathBuf::from(record.project_path.as_deref().unwrap())
        .join("film-harness")
        .join(&record.run_id)
        .join("run.json");
    assert!(project_record.exists(), "{}", project_record.display());
}

#[tokio::test]
async fn malformed_plan_is_refused_before_any_job_exists() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        plan["shots"][1]["id"] = json!("SH010");
        plan["shots"][2]["prompt"] = json!("");
        plan["shots"][3]["conditioning"] =
            json!({ "mode": "image_to_video", "referenceRoles": ["courier"] });
        plan["limits"]["maxAttemptsPerShot"] = json!(0);
    });
    let options = harness.options(plan, harness.fixture_pack(), None);
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("malformed plan is refused");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("duplicate shot id")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.starts_with("[SH030] prompt:")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.starts_with("[SH040] conditioning.firstFrameRole:")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.starts_with("[SH040] conditioning.referenceRoles:")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("limits.maxAttemptsPerShot")),
        "{text:?}"
    );
    assert!(
        harness.jobs().await.is_empty(),
        "no job may exist after a refusal"
    );
    let (_, projects) = request(harness.app.clone(), "GET", "/api/v1/projects", Value::Null).await;
    assert!(
        projects.as_array().unwrap().is_empty(),
        "no project is created either"
    );
    let record = harness.run_record();
    assert_eq!(record["outcome"], "rejected");
    assert_eq!(
        record["diagnostics"].as_array().unwrap().len(),
        findings.len()
    );
}

#[tokio::test]
async fn missing_reference_files_and_dangling_roles_are_refused_before_dispatch() {
    let harness = Harness::start(true, vec![]).await;
    // A pack in the temp dir whose files are absent, plus a role the plan never defines.
    let pack_dir = harness.temp_dir.path().join("pack");
    std::fs::create_dir_all(pack_dir.join("references")).unwrap();
    let pack_text = std::fs::read_to_string(harness.fixture_pack()).unwrap();
    let mut pack: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&pack_text)).unwrap();
    pack["references"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["role"] != "workshop_plate");
    let pack_path = pack_dir.join("references.json");
    std::fs::write(&pack_path, pack.to_string()).unwrap();
    // Only some of the remaining files exist.
    for role in ["courier", "recipient"] {
        std::fs::write(pack_dir.join(format!("references/{role}.png")), b"png").unwrap();
    }
    let options = harness.options(harness.fixture_plan(), pack_path, Some(&["SH010", "SH020"]));
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("dangling roles and missing files are refused");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] conditioning.firstFrameRole")
                && m.contains("\"workshop_plate\" is not in reference pack")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("\"red_parcel\"") && m.contains("is missing")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());
}

#[tokio::test]
async fn unsupported_conditioning_and_off_menu_timing_are_refused_against_the_catalog() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        // References on the base MiniMax-H3 checkpoint (limits.maxReferenceAssets = 0).
        plan["shots"][0]["conditioning"] =
            json!({ "mode": "reference_to_video", "referenceRoles": ["courier", "red_parcel"] });
        // A negative prompt the model has no axis for.
        plan["shots"][0]["negativePrompt"] = json!("blurry, low quality");
        // Off the fourteen-length menu, and an undeclared canvas.
        plan["shots"][1]["targetDurationSeconds"] = json!(6.0);
        plan["shots"][1]["resolution"] = json!("640x360");
        // Below the model's declared mlx minimum.
        plan["limits"]["maxMemoryGb"] = json!(32);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("catalog-level refusal");
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("[SH010] conditioning.mode")
            && m.contains("does not declare reference_to_video")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("[SH010] conditioning.referenceRoles")
                && m.contains("maxReferenceAssets")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("[SH010] negativePrompt")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] targetDurationSeconds") && m.contains("menu")),
        "{text:?}"
    );
    assert!(
        text.iter()
            .any(|m| m.contains("[SH020] resolution") && m.contains("640x360")),
        "{text:?}"
    );
    assert!(
        // The lane (and so the minimum: mlx 64 GB, candle 43 GB) follows the host the test runs
        // on; the finding's shape is what is pinned here.
        text.iter()
            .any(|m| m.contains("limits.maxMemoryGb") && m.contains("minMemoryGb of")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());
    assert_eq!(harness.run_record()["outcome"], "rejected");
}

#[tokio::test]
async fn unknown_model_and_missing_workers_are_refused_before_dispatch() {
    let harness = Harness::start(false, vec![]).await;
    let plan = harness.edited_plan(|plan| plan["model"]["id"] = json!("no_such_model"));
    let options = harness.options(plan, harness.fixture_pack(), None);
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|f| f.field == "model.id" && f.message.contains("not in this API's model catalog")),
        "{findings:?}"
    );

    // A valid plan with no registered worker: the memory budget cannot be checked and nothing
    // could claim the job, so it is refused rather than queued forever.
    let options = harness.options(harness.fixture_plan(), harness.fixture_pack(), None);
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    let text: Vec<String> = findings.iter().map(ToString::to_string).collect();
    assert!(
        text.iter().any(|m| m.contains("video_generate")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("timeline_export")),
        "{text:?}"
    );
    assert!(
        text.iter().any(|m| m.contains("cannot be checked")),
        "{text:?}"
    );
    assert!(harness.jobs().await.is_empty());

    // The install gate: a tier the catalog reports missing is refused when the gate is on.
    let mut options = harness.options(harness.fixture_plan(), harness.fixture_pack(), None);
    options.require_installed = true;
    let error = film_harness::run(&harness.transport, &options)
        .await
        .unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert!(
        findings
            .iter()
            .any(|f| f.field == "model.tier" && f.message.contains("not installed")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn shot_budget_cancels_a_hung_job_and_the_attempt_cap_bounds_retries() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 2, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run finishes with a record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let sh010 = &record.shots[0];
    assert_eq!(sh010.outcome, ShotOutcome::TimedOut);
    assert_eq!(
        sh010.attempts.len(),
        2,
        "exactly the declared attempt cap: {:#?}",
        sh010.attempts
    );
    for attempt in &sh010.attempts {
        assert_eq!(attempt.status, "timed_out");
        assert!(attempt
            .error
            .as_deref()
            .unwrap()
            .contains("per-shot budget of 2s"));
        let job_id = attempt.job_id.as_deref().unwrap();
        let (_, job) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/jobs/{job_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(job["status"], "canceled", "{job}");
        assert!(attempt.elapsed_seconds >= 2.0);
    }
    let sh020 = &record.shots[1];
    assert_eq!(
        sh020.outcome,
        ShotOutcome::Rendered,
        "an unaffected shot still renders\n{}",
        summary(&record)
    );
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(
        video_jobs, 3,
        "2 capped attempts + 1 rendered shot, no unbounded retry"
    );
    assert!(
        record.timeline.is_some(),
        "the rendered take is still assembled"
    );
    assert_eq!(harness.run_record()["outcome"], "failed");
}

#[tokio::test]
async fn a_failed_attempt_is_retried_only_up_to_the_declared_cap() {
    let harness = Harness::start(true, vec![("SH020", VideoBehavior::FailFirst)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"]["maxAttemptsPerShot"] = json!(2);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let sh020 = &record.shots[1];
    assert_eq!(sh020.attempts.len(), 2);
    assert_eq!(sh020.attempts[0].status, "failed");
    assert_eq!(
        sh020.attempts[0].error.as_deref(),
        Some("fake engine fault: transient")
    );
    assert_eq!(sh020.attempts[1].status, "completed");
    assert_ne!(sh020.attempts[0].job_id, sh020.attempts[1].job_id);
    assert_eq!(sh020.outcome, ShotOutcome::Rendered);
}

#[tokio::test]
async fn run_budget_stops_dispatch_and_leaves_the_remaining_shots_undispatched() {
    let harness = Harness::start(
        true,
        vec![(
            "SH010",
            // Longer than the run budget, so the only thing that can end the attempt is the
            // harness's own budget (the fake worker ignores the cancel until its delay elapses).
            VideoBehavior::Complete {
                delay_secs: 6,
                peak_pct: 40.0,
            },
        )],
    )
    .await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 3, "maxShotSeconds": 3, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::StoppedRunBudget,
        "{:#?}",
        record.shots
    );
    assert_eq!(
        record.shots[0].outcome,
        ShotOutcome::TimedOut,
        "{:#?}",
        record.shots[0]
    );
    assert_eq!(
        record.shots[0].attempts.len(),
        1,
        "the run budget stops retries too"
    );
    assert!(record.shots[0].attempts[0]
        .error
        .as_deref()
        .unwrap()
        .contains("run exceeded its budget of 3s"));
    for shot in &record.shots[1..] {
        assert_eq!(shot.outcome, ShotOutcome::NotDispatched, "{shot:?}");
        assert!(shot.attempts.is_empty());
    }
    assert!(record.timeline.is_none());
    assert!(record.export.is_none());
    let claimed = harness.script.lock().claimed.len();
    assert_eq!(claimed, 1, "exactly one job was ever dispatched");
    assert_eq!(harness.run_record()["outcome"], "stopped_run_budget");
}

#[tokio::test]
async fn observed_memory_over_budget_stops_new_dispatch() {
    // 90% of the 128 GB the fake worker reports is 115 GB, over the plan's 96 GB budget.
    let harness = Harness::start(
        true,
        vec![(
            "SH010",
            VideoBehavior::Complete {
                delay_secs: 1,
                peak_pct: 90.0,
            },
        )],
    )
    .await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_eq!(
        record.outcome,
        RunOutcome::StoppedMemoryLimit,
        "{}",
        summary(&record)
    );
    assert_eq!(
        record.shots[0].outcome,
        ShotOutcome::Rendered,
        "the finished take is kept\n{}",
        summary(&record)
    );
    assert_eq!(record.shots[0].attempts[0].peak_gpu_memory_pct, Some(90.0));
    assert_eq!(record.shots[1].outcome, ShotOutcome::NotDispatched);
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(video_jobs, 1);
    assert_eq!(harness.run_record()["outcome"], "stopped_memory_limit");
}

#[test]
fn checked_in_fixture_plates_match_the_generator_byte_for_byte() {
    for (role, rgb) in FIXTURE_REFERENCES {
        let path = Path::new(FIXTURE_DIR).join(format!("references/{role}.png"));
        let on_disk = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "{}: {error} (regenerate with `film-harness fixture-images`)",
                path.display()
            )
        });
        let generated = film_harness::fixture_plate_png(role, *rgb).unwrap();
        assert_eq!(
            on_disk,
            generated,
            "{} drifted from the generator; regenerate with `film-harness fixture-images --out {}`",
            path.display(),
            Path::new(FIXTURE_DIR).join("references").display()
        );
    }
    // Every plate the pack names is one the generator writes, and vice versa.
    let pack_text =
        std::fs::read_to_string(Path::new(FIXTURE_DIR).join("references.jsonc")).unwrap();
    let pack = sceneworks_core::film_plan::parse_reference_pack(&pack_text).unwrap();
    let mut pack_roles: Vec<&str> = pack.references.iter().map(|r| r.role.as_str()).collect();
    let mut generated_roles: Vec<&str> = FIXTURE_REFERENCES.iter().map(|(role, _)| *role).collect();
    pack_roles.sort_unstable();
    generated_roles.sort_unstable();
    assert_eq!(pack_roles, generated_roles);
}

#[tokio::test]
async fn validate_subcommand_path_checks_documents_without_an_api() {
    let temp_dir = tempfile::tempdir().unwrap();
    let options = RunOptions {
        plan_path: Path::new(FIXTURE_DIR).join("plan.jsonc"),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        project_id: None,
        shot_ids: Some(vec!["SH010".into(), "SH999".into()]),
        out_dir: temp_dir.path().to_path_buf(),
        poll_interval: Duration::from_secs(1),
        export: true,
        require_installed: true,
    };
    let error = film_harness::validate(None, &options).await.unwrap_err();
    let HarnessError::Validation(findings) = error else {
        panic!("expected a validation refusal, got {error}");
    };
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].message.contains("\"SH999\" is not in plan"));
    let mut options = options;
    options.shot_ids = None;
    let (plan, pack) = film_harness::validate(None, &options).await.unwrap();
    assert_eq!(plan.shots.len(), 6);
    assert_eq!(pack.references.len(), 7);
}
