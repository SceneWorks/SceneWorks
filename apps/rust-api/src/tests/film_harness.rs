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

/// An [`ApiTransport`] wrapping [`RouterTransport`] that can fail or rewrite chosen responses, so
/// a test can inject the failures a live API can produce (a 500 mid-run, a two-phase asset handoff
/// that never finishes) without changing production code to make itself testable.
struct ScriptedTransport {
    inner: RouterTransport,
    /// `(method, path substring)` → status to answer with instead of calling the router.
    failures: Vec<(&'static str, String, u16)>,
    /// A path substring and the rewrite applied to the successful responses it matches.
    rewrite: Option<ResponseRewrite>,
}

/// A path substring plus the mutation applied to every successful response whose path contains it.
type ResponseRewrite = (String, fn(&mut Value));

impl ScriptedTransport {
    fn failing(app: axum::Router, method: &'static str, path: &str, status: u16) -> Self {
        Self {
            inner: RouterTransport { app },
            failures: vec![(method, path.to_owned(), status)],
            rewrite: None,
        }
    }

    fn rewriting(app: axum::Router, path: &str, rewrite: fn(&mut Value)) -> Self {
        Self {
            inner: RouterTransport { app },
            failures: Vec::new(),
            rewrite: Some((path.to_owned(), rewrite)),
        }
    }
}

impl ApiTransport for ScriptedTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        if let Some((_, _, status)) = self
            .failures
            .iter()
            .find(|(method, path, _)| *method == request.method && request.path.contains(path))
        {
            let status = *status;
            return Box::pin(async move {
                Ok(ApiResponse {
                    status,
                    body: json!({ "detail": "injected failure" }),
                })
            });
        }
        let rewrite = self
            .rewrite
            .as_ref()
            .filter(|(path, _)| request.path.contains(path.as_str()))
            .map(|(_, rewrite)| *rewrite);
        let inner = self.inner.call(request);
        Box::pin(async move {
            let mut response = inner.await?;
            if let Some(rewrite) = rewrite {
                if (200..300).contains(&response.status) {
                    rewrite(&mut response.body);
                }
            }
            Ok(response)
        })
    }
}

/// How the fake worker treats one video job, keyed by the shot id the harness stamps into
/// `advanced.filmHarness.shotId`.
#[derive(Debug, Clone, Copy)]
enum VideoBehavior {
    /// Complete after `delay`, reporting `peak_pct` as the observed peak in its metrics block.
    Complete { delay_secs: u64, peak_pct: f64 },
    /// Fail once (first attempt), then complete.
    FailFirst,
    /// Never complete; honour a cancel request by reporting `canceled`.
    Hang,
    /// Never complete and never honour a cancel — a worker whose cooperative checkpoint is minutes
    /// away, or one wedged in a command buffer. The job stays `running` forever.
    HangIgnoringCancel,
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
        VideoBehavior::HangIgnoringCancel => loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
        },
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
    let peak_memory_bytes = (HOST_MEMORY_MB as f64 * 1024.0 * 1024.0 * peak_pct / 100.0) as u64;
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
    // The hardware peak goes where the real worker puts it: a metrics block POSTed to
    // `/api/v1/jobs/:id/metrics` AFTER the terminal progress (sceneworks-worker `lib.rs`
    // `metrics_probe.finish()` → `post_generation_metrics`). No `ProgressRequest` anywhere in the
    // worker ever sets `peakGpuMemoryPct`, so a fake that reported the peak on the progress update
    // would be testing a signal that does not exist on a real render.
    post_generation_metrics(
        app,
        job_id,
        json!({
            "backend": "mlx",
            "totalMs": 1_000,
            "peakMemoryBytes": peak_memory_bytes,
            "peakMemoryPct": peak_pct,
        }),
    )
    .await;
}

/// The worker's `post_generation_metrics`: an upsert of the run's metrics block, posted after the
/// job is already terminal.
async fn post_generation_metrics(app: &axum::Router, job_id: &str, metrics: Value) {
    let (status, response) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/metrics"),
        metrics,
    )
    .await;
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "metrics {job_id}: {response}"
    );
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

    /// A plan and pack with ONE reference and no conditioning, for the tests that measure a
    /// wall-clock budget.
    ///
    /// Reference import happens INSIDE the run, so it spends the run's budget: a test with a
    /// three-second budget and seven plates to import is racing its own setup, and under a loaded
    /// test binary it loses — the budget expires before the first shot is dispatched and the run
    /// records `NotDispatched` where the test meant to observe `TimedOut`. Importing one plate
    /// instead of seven puts setup an order of magnitude inside the budget, so what the test
    /// measures is the thing it is named after (sc-22712).
    fn lean_budget_fixture(&self, edit: impl FnOnce(&mut Value)) -> (PathBuf, PathBuf) {
        let plan = self.edited_plan(|plan| {
            for shot in plan["shots"].as_array_mut().expect("shots") {
                shot["conditioning"] = json!({ "mode": "text_to_video" });
                shot["continuityRoles"] = json!([]);
            }
            edit(plan);
        });
        let dir = self.temp_dir.path().join("lean-pack");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        std::fs::copy(
            Path::new(FIXTURE_DIR).join("references/workshop_plate.png"),
            dir.join("references/workshop_plate.png"),
        )
        .expect("plate copies");
        let pack = json!({
            "schemaVersion": 1,
            "id": "courier-workshop-refs",
            "version": 1,
            "references": [
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png" }
            ]
        });
        let pack_path = dir.join("references.json");
        std::fs::write(&pack_path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        (plan, pack_path)
    }

    /// The shipped pack with its `sound` array emptied, copied into the temp dir so the relative
    /// `file` paths still resolve. For the lanes with no ffmpeg to transcode an audio upload with.
    fn fixture_pack_without_sound(&self) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
        pack["sound"] = json!([]);
        let dir = self.temp_dir.path().join("pack");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        for entry in std::fs::read_dir(Path::new(FIXTURE_DIR).join("references"))
            .expect("fixture references dir")
        {
            let entry = entry.expect("directory entry");
            std::fs::copy(entry.path(), dir.join("references").join(entry.file_name()))
                .expect("plate copies");
        }
        let path = dir.join("references.json");
        std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        path
    }

    /// Copy the checked-in fixture into the temp dir with `edit` applied to the parsed plan, so a
    /// test can break one field without touching the shipped documents.
    ///
    /// The plan's SOUND is stripped first (sc-22712). Every caller is testing validation or a
    /// budget, and importing sound costs an ffmpeg transcode per clip — enough real time to spend a
    /// three-second run budget during setup, which is a test measuring the wrong thing. A test that
    /// wants a sound field back sets it inside `edit`.
    fn edited_plan(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_plan()).expect("fixture plan");
        let mut plan: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture plan parses");
        plan.as_object_mut().expect("plan object").remove("sound");
        for shot in plan["shots"].as_array_mut().expect("shots") {
            shot.as_object_mut()
                .expect("shot object")
                .remove("dialogueClip");
        }
        edit(&mut plan);
        let path = self.temp_dir.path().join("plan.json");
        std::fs::write(&path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        path
    }

    /// The checked-in pack (and its plates) copied into the temp dir with `edit` applied, so a test
    /// can change an entry without touching the shipped documents.
    /// The checked-in pack with `edit` applied, copied into the temp dir.
    ///
    /// Its SOUND is emptied first, the mirror of `edited_plan` and for the same reason (sc-22712):
    /// every caller is testing something about references, and importing sound needs an ffmpeg
    /// that is not on every lane. Pair it with `edited_plan`, which drops the roles that would
    /// otherwise dangle.
    fn edited_pack(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
        pack["sound"] = json!([]);
        edit(&mut pack);
        let dir = self.temp_dir.path().join("pack");
        std::fs::create_dir_all(dir.join("references")).expect("pack dir");
        for (role, _) in FIXTURE_REFERENCES {
            let name = format!("references/{role}.png");
            std::fs::copy(Path::new(FIXTURE_DIR).join(&name), dir.join(&name))
                .expect("plate copies");
        }
        let path = dir.join("references.json");
        std::fs::write(&path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
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
    // Sound import transcodes through ffmpeg (`ProjectStore::import_asset`), which is not on every
    // lane — the same posture as `import_asset_admits_audio_and_normalizes_it_to_pcm16_wav`. Where
    // there is no ffmpeg this test runs against the pack's pictures alone, so everything sc-22710
    // established still runs everywhere; the sound assertions below are gated on the same check and
    // `the_assembled_sequence_carries_three_independently_controlled_sound_buses` owns them in full.
    let sound = ffmpeg_reachable();
    let (plan, pack) = if sound {
        (harness.fixture_plan(), harness.fixture_pack())
    } else {
        (
            harness.edited_plan(|_| {}),
            harness.fixture_pack_without_sound(),
        )
    };
    let options = harness.options(plan, pack, Some(&["SH010", "SH020"]));
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
    assert_eq!(timeline.items[0].shot_id.as_deref(), Some("SH010"));
    assert_eq!(timeline.items[1].shot_id.as_deref(), Some("SH020"));
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
    if sound {
        // The clips this two-shot selection places — both beds and SH020's line, but not the
        // recipient's, whose shots are not in the selection. The detail lives in the dedicated
        // test; this is the shipped fixture's smoke.
        assert_eq!(record.sound.len(), 3, "{:#?}", record.sound);
        let roles: Vec<&str> = timeline
            .tracks
            .iter()
            .filter(|track| track.kind == "audio")
            .map(|track| track.role.as_str())
            .collect();
        assert_eq!(roles, vec!["dialogue", "ambience", "music"], "{roles:?}");
    }
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
                delay_secs: 9,
                peak_pct: 40.0,
            },
        )],
    )
    .await;
    // The run budget is wall-clock from the START of the run, so it has to leave room for the
    // pre-dispatch work (catalog, host, project, reference imports) on a loaded CI runner —
    // otherwise the budget expires before the first attempt and the test measures its own setup.
    // Two independent margins, because one of them alone was not enough: a six-second budget
    // instead of three, AND a fixture that imports one plate instead of seven.
    let (plan, pack) = harness.lean_budget_fixture(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 6, "maxShotSeconds": 6, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, pack, None);
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
        .contains("run exceeded its budget of 6s"));
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

/// The memory limit reads the SAME signal a real render produces: the job's `generation_metrics`
/// block (`peakMemoryBytes`), posted through `POST /api/v1/jobs/:id/metrics` after the terminal
/// progress. Nothing in this file writes `peakGpuMemoryPct` — a job snapshot field every shipped
/// worker leaves null — so the limit cannot be satisfied by a fabricated signal.
#[tokio::test]
async fn observed_memory_over_budget_stops_new_dispatch() {
    // 90% of the 128 GiB the fake worker reports is 115.2 GiB, over the plan's 96 GB budget.
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
    let attempt = &record.shots[0].attempts[0];
    assert_eq!(
        attempt.peak_memory_source.as_deref(),
        Some("metrics.peakMemoryBytes"),
        "the limit must read the production signal, not the job snapshot"
    );
    assert!(
        attempt
            .peak_memory_gb
            .is_some_and(|gb| (gb - 115.2).abs() < 0.01),
        "{:?}",
        attempt.peak_memory_gb
    );
    assert_eq!(attempt.peak_gpu_memory_pct, Some(90.0));
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
    // The peak the harness compared is the one the metrics route holds for that job.
    let job_id = attempt.job_id.as_deref().expect("job id");
    let (status, metrics) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}/metrics"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(
        metrics["peakMemoryBytes"].as_u64(),
        Some((HOST_MEMORY_MB as f64 * 1024.0 * 1024.0 * 0.90) as u64)
    );
}

/// E2's production record survives the failure it exists to explain: a run that has already
/// created a project and imported assets must still leave a `run.json` when the API fails mid-run.
#[tokio::test]
async fn an_api_failure_mid_run_still_writes_the_run_record() {
    let harness = Harness::start(true, vec![]).await;
    // The tag PATCH is the first write AFTER the project exists and the first reference has been
    // imported, so the run fails with real side effects already on disk.
    let transport = ScriptedTransport::failing(harness.app.clone(), "PATCH", "/tags", 500);
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("the injected 500 fails the run");
    assert!(
        matches!(&error, HarnessError::Api { status: 500, path, .. } if path.contains("/tags")),
        "{error}"
    );

    let record = harness.run_record();
    assert_eq!(record["outcome"], "failed");
    assert!(
        record["projectId"].is_string(),
        "the record names the project the run created: {record}"
    );
    let diagnostics: Vec<String> = record["diagnostics"]
        .as_array()
        .expect("diagnostics")
        .iter()
        .map(|finding| finding["message"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        diagnostics.iter().any(|message| message.contains("500")
            && message.contains("/tags")
            && message.contains("the run stopped on an error")),
        "{diagnostics:?}"
    );
    assert!(harness.temp_dir.path().join("run-out/plan.json").exists());
}

/// A cancel the worker never honours must stop the run, not free the harness to dispatch a second
/// render against the one memory budget the plan declared.
#[tokio::test]
async fn a_cancel_the_worker_ignores_stops_dispatch_instead_of_retrying() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::HangIgnoringCancel)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 2, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
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
        1,
        "the uncancelable attempt consumes the shot: {:#?}",
        sh010.attempts
    );
    assert!(
        sh010.attempts[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("still running")
            && sh010.attempts[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("stopping dispatch rather than running a second render"),
        "{:?}",
        sh010.attempts[0].error
    );
    assert_eq!(
        record.shots[1].outcome,
        ShotOutcome::NotDispatched,
        "no further shot goes out while the first render is still in flight"
    );
    let video_jobs = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "video_generate")
        .count();
    assert_eq!(video_jobs, 1, "exactly one render was ever dispatched");
    assert_eq!(harness.run_record()["outcome"], "failed");
}

/// A terminal job whose assets the API never finishes persisting is bounded and reported for what
/// it is, instead of being polled until the shot budget expires and blamed on a timeout.
#[tokio::test]
async fn a_terminal_job_whose_assets_never_settle_is_reported_as_unsettled() {
    let harness = Harness::start(true, vec![]).await;
    // Every job snapshot the harness reads is rewritten to look like the two-phase handoff never
    // completed: raw `assetWrites` still in place, no `assets` / `assetIds`.
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/api/v1/jobs/", |body| {
        if body.get("type").and_then(Value::as_str) == Some("video_generate")
            && body["status"] == "completed"
        {
            body["result"]["assetWrites"] = json!([{ "type": "video" }]);
            if let Some(result) = body["result"].as_object_mut() {
                result.remove("assets");
                result.remove("assetIds");
            }
        }
    });
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 3, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010"]));
    let record = film_harness::run(&transport, &options)
        .await
        .expect("run finishes with a record");
    let attempt = &record.shots[0].attempts[0];
    assert_eq!(attempt.status, "completed", "{}", summary(&record));
    assert!(
        attempt
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("assets never settled"),
        "{:?}",
        attempt.error
    );
    assert_eq!(
        record.shots[0].attempts.len(),
        1,
        "a server-side persistence stall is not retried into a second render"
    );
    assert_eq!(record.shots[0].outcome, ShotOutcome::Failed);
    assert!(
        attempt.elapsed_seconds < 60.0,
        "the wait is bounded, not left to the shot budget: {}",
        attempt.elapsed_seconds
    );
}

/// A timeline with no track to hold the takes is an error, not a silently empty save: the record
/// must never list a sequence the project does not hold.
#[tokio::test]
async fn a_timeline_with_no_video_track_fails_the_run_instead_of_saving_nothing() {
    let harness = Harness::start(true, vec![]).await;
    let transport = ScriptedTransport::rewriting(harness.app.clone(), "/timelines", |body| {
        if body.get("tracks").is_some() {
            body["tracks"] = json!([{ "id": "track_audio", "kind": "audio", "items": [] }]);
        }
    });
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    let error = film_harness::run(&transport, &options)
        .await
        .expect_err("a timeline with nowhere to put the takes stops the run");
    assert!(
        error
            .to_string()
            .contains("has no track_main and no video track"),
        "{error}"
    );
    // E2 again: the record still lands, with the rendered take and the failure both in it.
    let record = harness.run_record();
    assert_eq!(record["outcome"], "failed");
    assert_eq!(record["shots"][0]["outcome"], "rendered");
    assert!(record["timeline"].is_null(), "{record}");
}

/// An unapproved reference is still imported and recorded, but is distinguishable in the record and
/// in the project — AC1 is about APPROVED references staying addressable.
#[tokio::test]
async fn unapproved_references_are_tagged_and_recorded_apart_from_approved_ones() {
    let harness = Harness::start(true, vec![]).await;
    let pack = harness.edited_pack(|pack| {
        for entry in pack["references"].as_array_mut().expect("references") {
            if entry["role"] == "house_style" {
                entry["approved"] = json!(false);
            }
        }
    });
    let options = harness.options(harness.edited_plan(|_| {}), pack, Some(&["SH010"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let project_id = record.project_id.clone().expect("project created");
    let unapproved = record
        .references
        .iter()
        .find(|reference| reference.role == "house_style")
        .expect("the unapproved reference is still imported and recorded");
    assert!(!unapproved.approved);
    assert!(
        record
            .references
            .iter()
            .filter(|reference| reference.role != "house_style")
            .all(|reference| reference.approved),
        "{:#?}",
        record.references
    );
    let tags_for = |asset_id: &str| {
        let app = harness.app.clone();
        let path = format!("/api/v1/projects/{project_id}/assets/{asset_id}");
        async move {
            let (_, asset) = request(app, "GET", &path, Value::Null).await;
            asset["tags"]
                .as_array()
                .map(|tags| {
                    tags.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        }
    };
    let tags = tags_for(&unapproved.asset_id).await;
    assert!(
        tags.iter()
            .any(|tag| tag == "film-harness-reference-unapproved"),
        "{tags:?}"
    );
    assert!(
        !tags.iter().any(|tag| tag == "film-harness-reference"),
        "an unapproved plate must not carry the conditioning-eligible tag: {tags:?}"
    );
    let approved = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("approved plate");
    let tags = tags_for(&approved.asset_id).await;
    assert!(
        tags.iter().any(|tag| tag == "film-harness-reference"),
        "{tags:?}"
    );
}

/// Ctrl-C on the binary: the in-flight job is canceled through the API, nothing else is dispatched,
/// and the record still lands with the reason in it.
#[tokio::test]
async fn an_operator_cancel_stops_the_run_and_still_writes_the_record() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 120, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    let control = film_harness::RunControl::new();
    // Flip the control once a render is actually in flight, not after a fixed sleep, so the test
    // exercises "canceled with a job running" on every runner rather than racing the import.
    let signal = tokio::spawn({
        let control = control.clone();
        let app = harness.app.clone();
        async move {
            for _ in 0..400 {
                let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
                let running = jobs.as_array().is_some_and(|jobs| {
                    jobs.iter()
                        .any(|job| job["type"] == "video_generate" && job["status"] == "running")
                });
                if running {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            control.cancel();
        }
    });
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("run finishes with a record");
    signal.await.expect("signal task");

    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let sh010 = &record.shots[0];
    assert_eq!(sh010.attempts.len(), 1, "{:#?}", sh010.attempts);
    assert_eq!(sh010.attempts[0].status, "canceled_by_operator");
    let job_id = sh010.attempts[0].job_id.as_deref().expect("job id");
    let (_, job) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(job["status"], "canceled", "{job}");
    assert_eq!(record.shots[1].outcome, ShotOutcome::NotDispatched);
    let on_disk = harness.run_record();
    assert_eq!(on_disk["outcome"], "failed");
    let diagnostics: Vec<String> = on_disk["diagnostics"]
        .as_array()
        .expect("diagnostics")
        .iter()
        .map(|finding| finding["message"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("canceled by operator")),
        "{diagnostics:?}"
    );
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

// -------------------------------------------------------------------------------------------
// Editable picture and continuous sound (sc-22712)
// -------------------------------------------------------------------------------------------
//
// What is proved HERE is the shape of the saved sequence and what the editing commands do to it:
// which track each clip lands on, what it is linked to, and how everything re-times. What the
// exported MP4 actually SOUNDS like is proved where the real ffmpeg runs, in
// `sceneworks_worker::media_jobs::timeline_audio_mix_tests` — an assertion about a timeline is not
// an assertion about audio, and neither one substitutes for the other.

/// Whether an ffmpeg the store can transcode an audio upload with is reachable.
///
/// Sound import goes through `ProjectStore::import_asset` -> `transcode_to_wav_pcm16`, and ffmpeg
/// is not on every lane. Soft-skipping is the posture the sibling store test
/// (`import_asset_admits_audio_and_normalizes_it_to_pcm16_wav`) already takes for exactly this
/// call; `SCENEWORKS_REQUIRE_FFMPEG` turns the skip into a failure on the lane that installs one.
fn ffmpeg_reachable() -> bool {
    let reachable = match std::env::var("SCENEWORKS_FFMPEG") {
        Ok(path) if !path.trim().is_empty() => Path::new(path.trim()).is_file(),
        _ => std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success()),
    };
    assert!(
        reachable || std::env::var("SCENEWORKS_REQUIRE_FFMPEG").is_err(),
        "SCENEWORKS_REQUIRE_FFMPEG is set but no ffmpeg is reachable, so the film-harness sound \
         tests would have silently reported ok without importing a single clip"
    );
    reachable
}

/// Read the saved timeline document straight from the API — the thing the exporter reads and the
/// editor opens, rather than the run record's description of it.
async fn saved_timeline(app: &axum::Router, project_id: &str, timeline_id: &str) -> Value {
    let (status, timeline) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{timeline}");
    timeline
}

fn track_of<'a>(timeline: &'a Value, id: &str) -> &'a Value {
    timeline["tracks"]
        .as_array()
        .expect("tracks")
        .iter()
        .find(|track| track["id"] == json!(id))
        .unwrap_or_else(|| panic!("timeline has no track {id}: {timeline}"))
}

fn items_of<'a>(timeline: &'a Value, track_id: &str) -> &'a Vec<Value> {
    track_of(timeline, track_id)["items"]
        .as_array()
        .unwrap_or_else(|| panic!("track {track_id} has no items"))
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-3
}

/// AC2, on the saved sequence: dialogue, ambience and music are three separately controlled buses,
/// and the beds are placed ONCE rather than per shot.
#[tokio::test]
async fn the_assembled_sequence_carries_three_independently_controlled_sound_buses() {
    if !ffmpeg_reachable() {
        return;
    }
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline = record.timeline.as_ref().expect("timeline assembled");

    // Every clip this selection places is a project asset of type `audio`, tagged with its role —
    // and only those: `recipient_line` belongs to shots SH050/SH060, which this run left out.
    let imported: Vec<&str> = record.sound.iter().map(|clip| clip.role.as_str()).collect();
    assert_eq!(
        imported,
        // Pack order, filtered — not plan order.
        vec!["courier_line", "workshop_room_tone", "main_theme"],
        "{imported:?}"
    );
    for clip in &record.sound {
        let (status, asset) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets/{}", clip.asset_id),
            Value::Null,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
        assert_eq!(asset["type"], "audio", "{asset}");
        assert_eq!(asset["extra"]["filmHarness"]["role"], clip.role);
        assert_eq!(asset["extra"]["filmHarness"]["kind"], "sound");
        let tags: Vec<&str> = asset["tags"]
            .as_array()
            .map(|tags| tags.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert!(
            tags.contains(&format!("role:{}", clip.role).as_str()),
            "{tags:?}"
        );
    }

    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let total = timeline.duration_seconds;
    assert!(
        close(total, 2.0 * 5.1667),
        "two shots back to back: {total}"
    );

    // Picture: two takes, contiguous, both muting their own audio by default.
    let picture = items_of(&saved, "track_main");
    assert_eq!(picture.len(), 2);
    assert!(close(picture[0]["timelineEnd"].as_f64().unwrap(), 5.1667));
    assert!(close(picture[1]["timelineStart"].as_f64().unwrap(), 5.1667));
    for item in picture {
        assert_eq!(
            item["generatedAudio"], "mute",
            "the fixture's run-level policy is mute: {item}"
        );
    }

    // Dialogue: the plan places a line against SH020 only, at its own offset INTO that shot.
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1, "{dialogue:#?}");
    assert_eq!(dialogue[0]["filmHarness"]["shotId"], "SH020");
    assert!(
        close(dialogue[0]["timelineStart"].as_f64().unwrap(), 5.1667 + 1.2),
        "the line sits 1.2s into SH020, not 1.2s into the sequence: {}",
        dialogue[0]
    );
    assert!(
        close(
            dialogue[0]["timelineEnd"].as_f64().unwrap(),
            5.1667 + 1.2 + 2.0
        ),
        "the clip is the 2s fixture take: {}",
        dialogue[0]
    );
    assert_eq!(track_of(&saved, "track_dialogue")["gain"], 1.0);
    assert_eq!(track_of(&saved, "track_dialogue")["muted"], false);

    // Beds: ONE item each, spanning the whole sequence through the cut, at their own gains.
    for (track_id, gain, fade_in, fade_out) in [
        ("track_ambience", 0.35, 1.0, 1.5),
        ("track_music", 0.2, 2.0, 3.0),
    ] {
        let track = track_of(&saved, track_id);
        assert_eq!(track["kind"], "audio");
        assert_eq!(track["muted"], false);
        assert!(
            close(track["gain"].as_f64().unwrap(), gain),
            "{track_id} gain: {track}"
        );
        let items = items_of(&saved, track_id);
        assert_eq!(
            items.len(),
            1,
            "a bed is placed ONCE for the whole sequence — one item per shot is exactly the \
             per-shot restart this design exists to avoid: {items:#?}"
        );
        assert!(close(items[0]["timelineStart"].as_f64().unwrap(), 0.0));
        assert!(
            close(items[0]["timelineEnd"].as_f64().unwrap(), total),
            "the bed must reach the last frame: {} vs {total}",
            items[0]
        );
        assert!(
            close(items[0]["fadeInSeconds"].as_f64().unwrap(), fade_in)
                && close(items[0]["fadeOutSeconds"].as_f64().unwrap(), fade_out),
            "the plan's fades travel with the bed: {}",
            items[0]
        );
        // The source range follows the span, so the whole stretch that plays is asked for.
        assert!(close(
            items[0]["sourceOut"].as_f64().unwrap() - items[0]["sourceIn"].as_f64().unwrap(),
            total
        ));
    }

    // Sound never extends the picture: the timeline's own recomputed duration is the picture's.
    assert!(
        close(saved["duration"].as_f64().unwrap(), total),
        "the store recomputes duration across every track; sound must not stretch it: {}",
        saved["duration"]
    );

    // The run record says the same thing, so run.json alone explains the mix.
    assert_eq!(
        timeline.generated_audio_default,
        sceneworks_core::film_plan::GeneratedAudio::Mute
    );
    let buses: Vec<(&str, f64, bool, usize)> = timeline
        .tracks
        .iter()
        .filter(|track| track.kind == "audio")
        .map(|track| {
            (
                track.role.as_str(),
                track.gain,
                track.muted,
                track.items.len(),
            )
        })
        .collect();
    assert_eq!(
        buses,
        vec![
            ("dialogue", 1.0, false, 1),
            ("ambience", 0.35, false, 1),
            ("music", 0.2, false, 1),
        ],
        "{buses:?}"
    );
    for item in &timeline.items {
        assert_eq!(
            item.generated_audio,
            Some(sceneworks_core::film_plan::GeneratedAudio::Mute),
            "every picture item records the policy the export obeyed: {item:?}"
        );
    }

    // And the sound survives an edit to the picture: put SH020 first, and its line must travel
    // with it while both beds re-span the sequence. This is the half of AC1 that a picture-only
    // assertion misses — a reorder that leaves a line under the wrong shot has kept the shot/asset
    // links and still broken the film.
    film_harness::edit_timeline(
        &harness.transport,
        &film_harness::EditOptions {
            run_record_path: harness.temp_dir.path().join("run-out/run.json"),
            export: false,
            poll_interval: Duration::from_millis(250),
        },
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");
    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let total = saved["duration"].as_f64().expect("duration");
    let dialogue = items_of(&saved, "track_dialogue");
    assert_eq!(dialogue.len(), 1);
    assert!(
        close(dialogue[0]["timelineStart"].as_f64().unwrap(), 1.2),
        "SH020 now starts at 0, so its line sits at its own 1.2s offset into it: {}",
        dialogue[0]
    );
    for track_id in ["track_ambience", "track_music"] {
        let items = items_of(&saved, track_id);
        assert_eq!(items.len(), 1, "{track_id} is still ONE continuous bed");
        assert!(close(items[0]["timelineStart"].as_f64().unwrap(), 0.0));
        assert!(
            close(items[0]["timelineEnd"].as_f64().unwrap(), total),
            "{track_id} must still reach the last frame after the edit: {}",
            items[0]
        );
    }
}

/// AC3's policy half: a shot's own setting wins over the run's, and both are recorded.
///
/// Needs no sound files, so it runs on every lane. That the policy is OBEYED — that `mute` really
/// keeps a generated line out of the mix and `include` really brings it in — is measured against a
/// real export in `generated_audio_doubles_with_dialogue_only_when_explicitly_included`.
#[tokio::test]
async fn a_shots_generated_audio_policy_overrides_the_runs_and_both_are_recorded() {
    let harness = Harness::start(true, vec![]).await;
    let plan = harness.edited_plan(|plan| {
        plan["sound"]["generatedAudio"] = json!("mute");
        plan["shots"][1]["generatedAudio"] = json!("include");
    });
    let options = harness.options(
        plan,
        harness.fixture_pack_without_sound(),
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline = record.timeline.as_ref().expect("timeline assembled");
    let saved = saved_timeline(&harness.app, &project_id, &timeline.timeline_id).await;
    let picture = items_of(&saved, "track_main");
    assert_eq!(
        picture[0]["generatedAudio"], "mute",
        "SH010 inherits the run"
    );
    assert_eq!(
        picture[1]["generatedAudio"], "include",
        "SH020 declared its own"
    );

    // The resolved policy is on the shot record too, so run.json says what the export obeyed
    // without anyone re-deriving it from the plan.
    let resolved: Vec<(&str, &str)> = record
        .shots
        .iter()
        .filter(|shot| shot.outcome == ShotOutcome::Rendered)
        .map(|shot| {
            (
                shot.shot_id.as_str(),
                shot.intended.generated_audio.as_timeline_str(),
            )
        })
        .collect();
    assert_eq!(
        resolved,
        vec![("SH010", "mute"), ("SH020", "include")],
        "{resolved:?}"
    );
}

/// AC1: trim, reorder and replace-a-take, each keeping the shot -> asset links and re-timing the
/// sequence, and each persisted in both the project timeline and the run record.
#[tokio::test]
async fn trimming_reordering_and_replacing_a_take_keep_links_and_retime_the_sequence() {
    let harness = Harness::start(true, vec![]).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
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
    let project_id = record.project_id.clone().expect("project created");
    let timeline_id = record
        .timeline
        .as_ref()
        .expect("timeline")
        .timeline_id
        .clone();
    let original: Vec<(String, String)> = record
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .map(|item| {
            (
                item.shot_id.clone().expect("picture items carry a shot"),
                item.asset_id.clone(),
            )
        })
        .collect();

    let edit_options = film_harness::EditOptions {
        run_record_path: harness.temp_dir.path().join("run-out/run.json"),
        export: false,
        poll_interval: Duration::from_millis(250),
    };

    // 1. TRIM. SH010 keeps only 1.0..3.0 of its take, and SH020 slides up to meet it.
    let trimmed = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(1.0),
            source_out: Some(3.0),
        },
    )
    .await
    .expect("trim applies");
    let items = &trimmed.timeline.as_ref().expect("timeline").items;
    assert!(close(items[0].source_in, 1.0) && close(items[0].source_out, 3.0));
    assert!(close(items[0].timeline_start, 0.0) && close(items[0].timeline_end, 2.0));
    assert!(
        close(items[1].timeline_start, 2.0) && close(items[1].timeline_end, 2.0 + 5.1667),
        "the trim must RIPPLE — a cut does not leave a hole: {items:#?}"
    );
    assert_eq!(
        items
            .iter()
            .map(|item| (item.shot_id.clone().unwrap(), item.asset_id.clone()))
            .collect::<Vec<_>>(),
        original,
        "a trim changes timing, never which take a shot points at"
    );

    // 2. REORDER. The takes swap places and keep their own lengths.
    let reordered = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH020".to_owned(), "SH010".to_owned()],
        },
    )
    .await
    .expect("reorder applies");
    let items = &reordered.timeline.as_ref().expect("timeline").items;
    assert_eq!(
        items
            .iter()
            .map(|item| item.shot_id.clone().unwrap())
            .collect::<Vec<_>>(),
        vec!["SH020", "SH010"]
    );
    assert_eq!(
        items[0].asset_id, original[1].1,
        "SH020 keeps its own take at the head"
    );
    assert_eq!(items[1].asset_id, original[0].1);
    assert!(close(items[0].timeline_start, 0.0) && close(items[0].timeline_end, 5.1667));
    assert!(
        close(items[1].timeline_start, 5.1667) && close(items[1].timeline_end, 5.1667 + 2.0),
        "SH010 is still the trimmed 2s: {items:#?}"
    );

    // 3. REPLACE THE TAKE. Point SH010 at a different asset in the project; the sequence re-times
    //    around the replacement's own length and the version history remembers what it was.
    let replacement = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("plate imported")
        .asset_id
        .clone();
    let replaced = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::ReplaceTake {
            shot_id: "SH010".to_owned(),
            asset_id: replacement.clone(),
        },
    )
    .await
    .expect("replacement applies");
    let items = &replaced.timeline.as_ref().expect("timeline").items;
    let swapped = items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is still in the sequence");
    assert_eq!(swapped.asset_id, replacement);
    assert_ne!(swapped.asset_id, original[0].1);

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let picture = items_of(&saved, "track_main");
    let swapped_item = picture
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!("SH010"))
        .expect("SH010 on the saved picture track");
    assert_eq!(swapped_item["assetId"], replacement);
    assert_eq!(swapped_item["currentVersionAssetId"], replacement);
    let history: Vec<&str> = swapped_item["versionHistory"]
        .as_array()
        .expect("version history")
        .iter()
        .filter_map(|entry| entry["source"].as_str())
        .collect();
    assert_eq!(
        history,
        vec!["original", "replacement"],
        "the take that was there is still addressable: {swapped_item}"
    );
    assert!(
        swapped_item["versionAssetIds"]
            .as_array()
            .expect("version asset ids")
            .iter()
            .any(|value| value == &json!(replacement)),
        "{swapped_item}"
    );
    // Contiguity survived all three edits.
    let mut cursor = 0.0;
    for item in picture {
        assert!(
            close(item["timelineStart"].as_f64().unwrap(), cursor),
            "picture items must abut: {item} expected start {cursor}"
        );
        cursor = item["timelineEnd"].as_f64().unwrap();
    }

    // Every edit is in the run record on disk, oldest first, with the duration it produced.
    let on_disk = harness.run_record();
    let edits: Vec<&str> = on_disk["timeline"]["edits"]
        .as_array()
        .expect("edits recorded")
        .iter()
        .filter_map(|edit| edit["kind"].as_str())
        .collect();
    assert_eq!(edits, vec!["trim", "reorder", "replace_take"], "{on_disk}");
    assert!(on_disk["timeline"]["edits"][2]["detail"]
        .as_str()
        .unwrap_or_default()
        .contains(&replacement));

    // A reorder that does not name the whole sequence is refused rather than silently dropping a
    // shot — the one way a "reorder" could quietly become a delete.
    let error = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Reorder {
            shot_ids: vec!["SH010".to_owned()],
        },
    )
    .await
    .expect_err("a partial order is refused");
    let message = error.to_string();
    assert!(
        message.contains("SH020") && message.contains("exactly once"),
        "{message}"
    );
}

/// The checked-in fixture clips are exactly what the generator writes, the same guarantee
/// `checked_in_fixture_plates_match_the_generator_byte_for_byte` gives the plates.
#[test]
fn checked_in_fixture_sound_matches_the_generator_byte_for_byte() {
    for (role, seconds, hz, amplitude) in film_harness::FIXTURE_SOUNDS {
        let path = Path::new(FIXTURE_DIR)
            .join("sound")
            .join(format!("{role}.wav"));
        let checked_in = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("{} is missing ({error})", path.display());
        });
        let generated = film_harness::fixture_sound_wav(*seconds, *hz, *amplitude);
        assert_eq!(
            checked_in.len(),
            generated.len(),
            "{role}: checked-in clip is a different length from the generator's"
        );
        assert!(
            checked_in == generated,
            "{role}: the checked-in clip no longer matches `fixture_sound_wav`. Regenerate it \
             with `film-harness fixture-sound --out config/film-harness/courier-workshop/sound`."
        );
    }
}
