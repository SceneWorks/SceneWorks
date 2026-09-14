//! End-to-end tests for the local filmmaking harness (sc-22710): the REAL API routes in-process
//! (`create_app_with_state`), a scripted fake worker that claims the jobs through the worker API
//! exactly as the GPU worker would, and the harness driving both through `ApiTransport`.
//!
//! What the fake worker replaces is only the render: it claims `video_generate` / `timeline_export`
//! jobs, writes a placeholder file where the real worker would write the MP4, and reports the same
//! `assetWrites` / `assetIds` result shapes the real worker reports. Asset persistence, timeline
//! validation, export dispatch and every enqueue gate are the production code paths.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use parking_lot::Mutex;
use sceneworks_core::film_plan::{RunOutcome, RunRecord, RunState, ShotOutcome};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::film_harness::{
    self, ApiRequest, ApiResponse, ApiTransport, HarnessError, RequestBody, ResumeOptions,
    RunControl, RunOptions, TransportFuture, FIXTURE_REFERENCES,
};
use crate::film_planner;
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
    /// Fail every time — a shot no retry will rescue.
    FailAlways,
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
    /// Make `timeline_export` jobs fail, for the failed-export resume path (sc-22711).
    export_fails: bool,
    /// Replies the fake worker returns for `prompt_refine` jobs whose task is `film_plan`, in
    /// order. The last one repeats once the list runs out, which is what lets a test prove the
    /// repair loop STOPS rather than looping on a reply that never validates.
    plan_replies: Vec<String>,
    plan_calls: usize,
    /// Reply for the per-shot prompt-refinement (the ordinary rewrite task). `{prompt}` is replaced
    /// by the shot's own prompt.
    refine_template: Option<String>,
}

impl WorkerScript {
    fn next_plan_reply(&mut self) -> String {
        let index = self
            .plan_calls
            .min(self.plan_replies.len().saturating_sub(1));
        self.plan_calls += 1;
        self.plan_replies
            .get(index)
            .cloned()
            .unwrap_or_else(|| "the planner has nothing to say".to_owned())
    }
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
            "capabilities": ["video_generate", "timeline_export", "frame_extract", "prompt_refine"],
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
                "timeline_export" => run_fake_export_job(&app, &script, &job_id, &job).await,
                "prompt_refine" => run_fake_refine_job(&app, &script, &job_id, &job).await,
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
        VideoBehavior::FailAlways => {
            post_progress(
                app,
                job_id,
                json!({
                    "status": "failed", "stage": "failed", "progress": 1,
                    "message": "fake engine fault", "error": "fake engine fault: persistent",
                    "workerId": WORKER_ID
                }),
            )
            .await;
            return;
        }
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

/// The scripted stand-in for the native `prompt_refine` worker. It replaces ONLY the decode: the
/// job was created through the real `POST /api/v1/prompts/refine` route, claimed through the real
/// worker API, and its result is read back through the real job snapshot — so the planner is
/// exercised against the seam it will use on the GPU, with the model's answer scripted.
async fn run_fake_refine_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let payload = &job["payload"];
    let task = payload["task"].as_str().unwrap_or_default().to_owned();
    let prompt = payload["prompt"].as_str().unwrap_or_default().to_owned();
    let refined = if task == "film_plan" {
        script.lock().next_plan_reply()
    } else {
        script
            .lock()
            .refine_template
            .clone()
            .unwrap_or_else(|| "{prompt}".to_owned())
            .replace("{prompt}", &prompt)
    };
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "fake refine done", "workerId": WORKER_ID, "backend": "mlx",
            "result": { "originalPrompt": prompt, "refinedPrompt": refined }
        }),
    )
    .await;
}

async fn run_fake_export_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    if script.lock().export_fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake ffmpeg fault", "error": "fake ffmpeg fault: no such codec",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
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
            compiled_path: None,
            project_id: None,
            shot_ids: shots.map(|ids| ids.iter().map(|id| (*id).to_owned()).collect()),
            out_dir: self.out_dir(),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
        }
    }

    fn out_dir(&self) -> PathBuf {
        self.temp_dir.path().join("run-out")
    }

    /// What `resume` / `replace-take` are driven with in these tests: the same run directory, a
    /// tight poll cadence, and a control only the test can trip (sc-22711).
    fn resume_options(&self) -> ResumeOptions {
        ResumeOptions {
            out_dir: self.out_dir(),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
            control: RunControl::new(),
        }
    }

    /// Resume until the run reaches a state it will not leave on its own, so a test asserts about
    /// the end of the story rather than about how many crashes it took to get there. Bounded: a
    /// resume that makes no progress is a failure, not a retry.
    async fn resume_to_completion(&self) -> RunRecord {
        let mut last = None;
        for round in 0..12 {
            match film_harness::resume(&self.transport, &self.resume_options()).await {
                Ok(record) => {
                    if record.outcome == RunOutcome::Completed || !record.is_resumable() {
                        return record;
                    }
                    last = Some(record);
                }
                Err(error) => panic!("resume {round} failed: {error}"),
            }
        }
        panic!("run never settled after 12 resumes: {last:#?}");
    }

    fn video_job_count(&self) -> usize {
        self.script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "video_generate")
            .count()
    }

    /// Video jobs the API holds, which — unlike the claim log — counts a job the worker has not
    /// picked up yet.
    async fn api_video_job_count(&self) -> usize {
        self.jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "video_generate")
            .count()
    }

    async fn project_count(&self) -> usize {
        let (_, projects) = request(self.app.clone(), "GET", "/api/v1/projects", Value::Null).await;
        projects.as_array().map(Vec::len).unwrap_or_default()
    }

    /// Every timeline the project holds. `POST /timelines` always creates a new row, so this is
    /// what says whether a replay adopted the run's timeline or created a second one.
    async fn timelines(&self, project_id: &str) -> Vec<Value> {
        let (_, timelines) = request(
            self.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines"),
            Value::Null,
        )
        .await;
        timelines.as_array().cloned().unwrap_or_default()
    }

    /// The timelines of whichever project the record names, or none when it named no project.
    async fn timelines_for(&self, record: &RunRecord) -> Vec<Value> {
        match record.project_id.as_deref() {
            Some(project_id) => self.timelines(project_id).await,
            None => Vec::new(),
        }
    }

    fn export_job_count(&self) -> usize {
        self.script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "timeline_export")
            .count()
    }

    /// Rewrite the record on disk, exactly as a test that needs a run to look older than it is has
    /// to: the harness reads `run.json` back on every `resume`.
    fn edit_run_record(&self, edit: impl FnOnce(&mut Value)) {
        let path = self.out_dir().join("run.json");
        let mut record: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("run.json")).expect("json");
        edit(&mut record);
        std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    /// A two-shot plan with a one-reference pack, written into the temp dir.
    ///
    /// The wall-clock-budget tests need the budget to be spent by the RENDER, not by setup:
    /// importing the shipped seven-reference pack through the real import route costs several
    /// seconds in a debug build under a loaded runner, which would otherwise eat a small budget
    /// before the first shot is ever dispatched. Same model, same menus — just less to import.
    fn minimal_documents(&self, limits: Value) -> (PathBuf, PathBuf) {
        let dir = self.temp_dir.path().join("minimal");
        std::fs::create_dir_all(dir.join("references")).expect("minimal dir");
        std::fs::copy(
            Path::new(FIXTURE_DIR).join("references/workshop_plate.png"),
            dir.join("references/workshop_plate.png"),
        )
        .expect("plate copies");
        let shot = |id: &str, depends_on: Value| {
            json!({
                "id": id,
                "beat": format!("{id} beat"),
                "framing": "wide static",
                "prompt": format!("a quiet workshop, shot {id}"),
                "targetDurationSeconds": 5.1667,
                "startState": "before",
                "endState": "after",
                "conditioning": { "mode": "text_to_video" },
                // Every shot binds at least one approved role (sc-22713): the pack below approves
                // exactly one, and these shots are about the workshop.
                "continuityRoles": ["workshop_plate"],
                "dependsOn": depends_on,
            })
        };
        let plan = json!({
            "schemaVersion": 2,
            "id": "budget-fixture",
            "version": 1,
            "title": "Budget fixture",
            "model": { "id": "minimax_h3", "tier": "q4", "fps": 24, "resolution": "576x320" },
            "limits": limits,
            "shots": [
                shot("SH010", json!([])),
                shot("SH020", json!([{ "shotId": "SH010", "kind": "continuity" }])),
            ],
        });
        let pack = json!({
            "schemaVersion": 1,
            "id": "budget-refs",
            "version": 1,
            "references": [
                { "role": "workshop_plate", "kind": "plate", "file": "references/workshop_plate.png" }
            ],
        });
        let plan_path = dir.join("plan.json");
        let pack_path = dir.join("references.json");
        std::fs::write(&plan_path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
        std::fs::write(&pack_path, serde_json::to_string_pretty(&pack).unwrap()).unwrap();
        (plan_path, pack_path)
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
    // Each refusal gets its own run directory: a refused run still writes its record there, and a
    // second `run` over a directory that already holds one is itself refused.
    let mut options = harness.options(plan, harness.fixture_pack(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-unknown-model");
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
    let mut options = harness.options(harness.fixture_plan(), harness.fixture_pack(), None);
    options.out_dir = harness.temp_dir.path().join("run-out-no-worker");
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
    options.out_dir = harness.temp_dir.path().join("run-out-install-gate");
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
    // A job that never finishes on its own, so the ONLY thing that can end the attempt is the run
    // budget — whether setup took a moment or a while on a loaded runner. `maxShotSeconds` equals
    // `maxRunSeconds` and the run deadline always comes first (it starts before the attempt does),
    // so this is the run budget tripping, not the shot budget. The two-shot / one-reference
    // documents keep the pre-dispatch work (catalog, host, project, imports) small enough that the
    // budget bounds the RENDER rather than the setup — sc-22711 after a 3s budget expired during
    // setup on a runner busy with other suites.
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 12, "maxShotSeconds": 12, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
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
        .contains("run exceeded its budget of 12s"));
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
    // Sound-free: this test is about the memory budget, and importing the pack's clips transcodes
    // through ffmpeg, which the hosted macOS lane does not have. Same reasoning as `edited_plan`.
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
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
    // Sound-free for the same reason as the budget test above: nothing here is about sound, and the
    // import transcodes through an ffmpeg the hosted macOS lane does not have.
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
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

    // sc-22711 gives a cancel its own outcome and a RESUMABLE stop, rather than folding it into
    // `failed` with a diagnostic: nothing went wrong, and the run can be picked back up.
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "canceled");
    assert!(stop.resumable, "a cancel leaves work a resume can finish");
    let sh010 = &record.shots[0];
    assert_eq!(sh010.attempts.len(), 1, "{:#?}", sh010.attempts);
    assert_eq!(sh010.attempts[0].status, "canceled_by_operator");
    assert_eq!(sh010.outcome, ShotOutcome::Canceled);
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
    assert_eq!(on_disk["outcome"], "canceled");
    assert_eq!(on_disk["state"], "finished");
    assert!(
        on_disk["stop"]["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("canceled"),
        "the record says what stopped it: {}",
        on_disk["stop"]
    );
    assert_eq!(on_disk["stop"]["resumable"], true);
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
        compiled_path: None,
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

// ---------------------------------------------------------------------------------------------
// sc-22711 — durable run state: crash windows, reconciliation, cancellation, take replacement
// ---------------------------------------------------------------------------------------------

/// Where a simulated controller death lands relative to the API call it died on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultMode {
    /// The call never reaches the API: the mutation did not happen.
    Before,
    /// The call reaches the API and succeeds, but the controller never sees the answer. This is the
    /// window that matters — the job (or project, or asset) exists and its id is nowhere in the
    /// record, so only an idempotency key can stop a replay from creating a second one.
    After,
}

/// A transport that stops answering the controller after `die_at` calls. Everything after the fault
/// fails too, because the process it models is gone.
struct FaultTransport {
    inner: RouterTransport,
    die_at: usize,
    mode: FaultMode,
    /// When set, only POSTs whose path contains this needle are counted, which aims the fault at
    /// one specific transition instead of sweeping every call.
    route: Option<&'static str>,
    matched: AtomicUsize,
    dead: AtomicBool,
}

impl FaultTransport {
    fn new(app: axum::Router, die_at: usize, mode: FaultMode) -> Self {
        Self {
            inner: RouterTransport { app },
            die_at,
            mode,
            route: None,
            matched: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
        }
    }

    fn on_post_route(mut self, needle: &'static str) -> Self {
        self.route = Some(needle);
        self
    }

    fn fired(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }
}

impl ApiTransport for FaultTransport {
    fn call(&self, request: ApiRequest) -> TransportFuture<'_> {
        if self.fired() {
            return Box::pin(async {
                Err(HarnessError::Transport("controller is gone".to_owned()))
            });
        }
        let targeted = self
            .route
            .is_none_or(|needle| request.method == "POST" && request.path.contains(needle));
        if !targeted || self.matched.fetch_add(1, Ordering::SeqCst) + 1 < self.die_at {
            return self.inner.call(request);
        }
        self.dead.store(true, Ordering::SeqCst);
        let deliver = self.mode == FaultMode::After;
        Box::pin(async move {
            if deliver {
                // The API really performs the write; the controller just never learns the answer.
                let _ = self.inner.call(request).await?;
            }
            Err(HarnessError::Transport(
                "simulated controller death".to_owned(),
            ))
        })
    }
}

fn fast(shots: &[&str]) -> Vec<(&'static str, VideoBehavior)> {
    const IDS: &[&str] = &["SH010", "SH020", "SH030", "SH040", "SH050", "SH060"];
    IDS.iter()
        .filter(|id| shots.contains(id))
        .map(|id| {
            (
                *id,
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            )
        })
        .collect()
}

/// What identifies one shot's selected take: which attempt, which job, which asset. Compared
/// instead of the whole `AttemptRecord` whenever one side has been through the record on disk — an
/// `elapsedSeconds` measurement does not survive a JSON round trip in its last bit, and none of
/// these assertions are about that.
fn selected_identity(
    record: &RunRecord,
    shot_id: &str,
) -> Option<(u32, Option<String>, Option<String>)> {
    let attempt = record.shot(shot_id)?.selected()?;
    Some((
        attempt.attempt,
        attempt.job_id.clone(),
        attempt.take.as_ref().map(|take| take.asset_id.clone()),
    ))
}

/// The selected take of every shot, as `shot -> (attempt, asset)`. The thing replay must never move.
fn selections(record: &RunRecord) -> Vec<(String, Option<(u32, String)>)> {
    record
        .shots
        .iter()
        .map(|shot| {
            (
                shot.shot_id.clone(),
                shot.selected().and_then(|attempt| {
                    Some((attempt.attempt, attempt.take.as_ref()?.asset_id.clone()))
                }),
            )
        })
        .collect()
}

/// Kill the controller at EVERY point it can die — before and after each API call it makes — and
/// resume from the record it left behind. Whatever the window, the run must finish with exactly the
/// work a clean run does: two video jobs, one project, seven reference assets, and the same take
/// selected for each shot.
#[tokio::test]
async fn a_crash_at_every_window_replays_without_duplicate_work_or_a_moved_take() {
    for mode in [FaultMode::Before, FaultMode::After] {
        let mut window = 1_usize;
        loop {
            assert!(window < 300, "sweep did not terminate ({mode:?})");
            let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
            let transport = FaultTransport::new(harness.app.clone(), window, mode);
            // The two-shot / one-reference documents: every distinct transition a run makes, none
            // of the near-identical repeats seven reference imports would add to the sweep. The
            // multi-reference replay has its own targeted test below.
            let (plan, pack) = harness.minimal_documents(json!({
                "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2,
                "maxMemoryGb": 96
            }));
            let options = harness.options(plan, pack, None);
            let crashed = film_harness::run(&transport, &options).await;
            if !transport.fired() {
                // Past the last call a clean run makes: the sweep has covered every window.
                let record = crashed.expect("a run that never faulted completes");
                assert_eq!(
                    record.outcome,
                    RunOutcome::Completed,
                    "{}",
                    summary(&record)
                );
                assert_eq!(harness.video_job_count(), 2);
                assert!(
                    window > 20,
                    "{mode:?}: only {} windows swept — the run stopped making calls far earlier \
                     than a project + reference + two dispatches + timeline + export should",
                    window - 1
                );
                break;
            }
            crashed.expect_err("the fault stops the controller");

            let context = format!("{mode:?} window {window}");
            match film_harness::read_run_record(&harness.out_dir()) {
                Err(_) => {
                    // Died before the record existed. Nothing was created, so there is nothing to
                    // resume — and nothing to clean up either.
                    assert_eq!(harness.video_job_count(), 0, "{context}");
                    assert_eq!(harness.project_count().await, 0, "{context}");
                }
                Ok(crashed_record) => {
                    assert_eq!(
                        crashed_record.state,
                        RunState::Running,
                        "{context}: a record left by a dead controller must say so"
                    );
                    let before = selections(&crashed_record);
                    let record = harness.resume_to_completion().await;
                    assert_eq!(
                        record.outcome,
                        RunOutcome::Completed,
                        "{context}\n{}",
                        summary(&record)
                    );
                    assert_eq!(record.run_id, crashed_record.run_id, "{context}");
                    assert_eq!(
                        harness.video_job_count(),
                        2,
                        "{context}: replay enqueued duplicate work\n{}",
                        summary(&record)
                    );
                    assert_eq!(harness.project_count().await, 1, "{context}");
                    assert_eq!(
                        harness.timelines_for(&record).await.len(),
                        1,
                        "{context}: replay left the project holding two timelines"
                    );
                    assert_eq!(record.references.len(), 1, "{context}");
                    let mut asset_ids: Vec<&str> = record
                        .references
                        .iter()
                        .map(|reference| reference.asset_id.as_str())
                        .collect();
                    asset_ids.sort_unstable();
                    let unique = asset_ids.len();
                    asset_ids.dedup();
                    assert_eq!(
                        asset_ids.len(),
                        unique,
                        "{context}: a reference was imported twice"
                    );
                    // A take the crashed record had already selected is still the selected take.
                    for (shot_id, selected) in before {
                        if let Some(expected) = selected {
                            let actual = record
                                .shot(&shot_id)
                                .and_then(|shot| shot.selected())
                                .map(|attempt| {
                                    (
                                        attempt.attempt,
                                        attempt.take.as_ref().unwrap().asset_id.clone(),
                                    )
                                });
                            assert_eq!(
                                actual,
                                Some(expected),
                                "{context}: {shot_id}'s selected take moved"
                            );
                        }
                    }
                    for shot_id in ["SH010", "SH020"] {
                        let shot = record.shot(shot_id).expect("shot recorded");
                        assert_eq!(shot.outcome, ShotOutcome::Rendered, "{context} {shot_id}");
                        assert_eq!(
                            shot.attempts.len(),
                            1,
                            "{context} {shot_id}: one attempt, not one per restart"
                        );
                    }
                    let export = record.export.as_ref().expect("export recorded");
                    assert_eq!(export.status, "completed", "{context}");
                    let exports = harness
                        .script
                        .lock()
                        .claimed
                        .iter()
                        .filter(|(kind, _, _)| kind == "timeline_export")
                        .count();
                    assert_eq!(exports, 1, "{context}: the export ran twice");
                }
            }
            window += 1;
        }
    }
}

/// Dying right after a reference upload the record never learned about must not import it twice.
/// Swept over every reference in the shipped seven-role pack, so "the first one" and "the last one"
/// are both covered.
#[tokio::test]
async fn a_reference_imported_but_not_recorded_is_adopted_not_imported_again() {
    for nth in [1_usize, 4, 7] {
        let harness = Harness::start(true, fast(&["SH010"])).await;
        let transport = FaultTransport::new(harness.app.clone(), nth, FaultMode::After)
            .on_post_route("/assets");
        let options = harness.options(
            harness.fixture_plan(),
            harness.fixture_pack(),
            Some(&["SH010"]),
        );
        film_harness::run(&transport, &options)
            .await
            .expect_err("the fault stops the controller");
        assert!(transport.fired(), "reference {nth}");

        let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
        assert_eq!(
            crashed.references.len(),
            nth - 1,
            "reference {nth}: the upload that was answered is the one the record missed"
        );
        let record = harness.resume_to_completion().await;
        assert_eq!(
            record.outcome,
            RunOutcome::Completed,
            "{}",
            summary(&record)
        );
        assert_eq!(record.references.len(), 7, "reference {nth}");

        // Exactly seven reference assets exist in the project — the interrupted upload was adopted
        // by its provenance rather than uploaded a second time.
        let project_id = record.project_id.clone().expect("project");
        let (_, assets) = request(
            harness.app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets"),
            Value::Null,
        )
        .await;
        let references: Vec<&Value> = assets
            .as_array()
            .into_iter()
            .flatten()
            .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
            .collect();
        assert_eq!(
            references.len(),
            7,
            "reference {nth}: a reference was imported twice"
        );
        let mut roles: Vec<&str> = references
            .iter()
            .filter_map(|asset| asset["extra"]["filmHarness"]["role"].as_str())
            .collect();
        roles.sort_unstable();
        roles.dedup();
        assert_eq!(roles.len(), 7, "reference {nth}: duplicate roles");

        // The upload and the tag PATCH are two writes: the ADOPTED asset — the one whose upload the
        // record missed — must end up tagged like every other, or a query for the
        // conditioning-eligible references silently misses it while the record claims it is tagged.
        for asset in &references {
            let role = asset["extra"]["filmHarness"]["role"]
                .as_str()
                .expect("role provenance");
            let tags: Vec<&str> = asset["tags"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            assert!(
                tags.contains(&"film-harness-reference"),
                "reference {nth}: {role} carries no kind tag: {tags:?}"
            );
            assert!(
                tags.contains(&format!("role:{role}").as_str()),
                "reference {nth}: {role} carries no role tag: {tags:?}"
            );
        }
    }
}

#[tokio::test]
async fn resuming_a_finished_run_reuses_every_take_and_enqueues_nothing() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(first.outcome, RunOutcome::Completed, "{}", summary(&first));
    assert!(
        !first.is_resumable(),
        "a completed run is not something to resume"
    );

    // A completed run refuses a resume outright rather than re-dispatching anything.
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("a completed run is not resumable");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("not resumable")),
        "{error}"
    );
    assert_eq!(harness.video_job_count(), 2);
    assert_eq!(selections(&harness_record(&harness)), selections(&first));
}

/// Reading the record back off disk, which is what a separate `film-harness` invocation does.
fn harness_record(harness: &Harness) -> RunRecord {
    film_harness::read_run_record(&harness.out_dir()).expect("run record on disk")
}

/// Block until `shot_id`'s job has finished, its assets are persisted AND its metrics block has
/// landed — everything a resume needs to ADOPT the attempt rather than poll it. Without this a
/// resume races the fake worker and exercises the dispatch path instead of the reconciliation one.
async fn wait_for_settled_shot(app: &axum::Router, shot_id: &str) -> String {
    for _ in 0..800 {
        let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        let job_id = jobs
            .as_array()
            .into_iter()
            .flatten()
            .find(|job| job["payload"]["advanced"]["filmHarness"]["shotId"] == shot_id)
            .and_then(|job| job["id"].as_str())
            .map(str::to_owned);
        if let Some(job_id) = job_id {
            let (_, job) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}"),
                Value::Null,
            )
            .await;
            let (_, metrics) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{job_id}/metrics"),
                Value::Null,
            )
            .await;
            if job["status"] == "completed"
                && job["result"]["assets"].is_array()
                && !metrics.is_null()
            {
                return job_id;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the job for {shot_id} never settled");
}

/// Block until `shot_id`'s job is actually running, so a cancel lands mid-flight rather than at
/// whatever point a fixed sleep happens to reach on a loaded runner.
async fn wait_for_running_shot(app: &axum::Router, shot_id: &str) {
    for _ in 0..800 {
        let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        let running = jobs.as_array().into_iter().flatten().any(|job| {
            job["payload"]["advanced"]["filmHarness"]["shotId"] == shot_id
                && job["status"] == "running"
        });
        if running {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{shot_id} never reached running");
}

#[tokio::test]
async fn a_cancel_stops_dispatch_keeps_finished_takes_and_the_run_resumes() {
    let harness = Harness::start(
        true,
        vec![
            (
                "SH010",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            ),
            ("SH020", VideoBehavior::Hang),
        ],
    )
    .await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan,
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    // Trip the cancel once SH010 is safely rendered and SH020 is hanging.
    let control = RunControl::new();
    let app = harness.app.clone();
    let waiter = tokio::spawn({
        let control = control.clone();
        async move {
            wait_for_running_shot(&app, "SH020").await;
            control.cancel();
        }
    });
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    waiter.await.expect("waiter joins");

    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "canceled");
    assert!(stop.resumable, "a cancel leaves work a resume can finish");
    assert!(record.is_resumable());

    // The finished take is kept, the hung job is canceled through the API, and the shot that was
    // never reached was never dispatched.
    let sh010 = record.shot("SH010").expect("SH010 recorded");
    assert_eq!(sh010.outcome, ShotOutcome::Rendered);
    assert_eq!(sh010.selected_attempt, Some(1));
    let sh020 = record.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 1, "a cancel is not a retry");
    assert_eq!(sh020.attempts[0].status, "canceled_by_operator");
    assert_eq!(
        sh020.outcome,
        ShotOutcome::Canceled,
        "a canceled shot is not a failed one — nothing went wrong with it"
    );
    let job_id = sh020.attempts[0].job_id.as_deref().expect("a job id");
    let (_, job) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(job["status"], "canceled", "{job}");
    assert_eq!(
        record.shot("SH030").expect("SH030 recorded").outcome,
        ShotOutcome::NotDispatched
    );
    assert!(record.export.is_none(), "a cancel dispatches no export");
    assert_eq!(harness.run_record()["outcome"], "canceled");

    // Resuming finishes the job: SH020 gets its remaining attempt, SH030 is dispatched, and SH010's
    // take is reused rather than re-rendered.
    harness
        .script
        .lock()
        .behaviors
        .retain(|(id, _)| id != "SH020");
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        resumed.shot("SH010").unwrap().attempts.len(),
        1,
        "SH010 was not re-rendered"
    );
    // Identity, not the whole struct: the resumed record came back through JSON, and an
    // `elapsedSeconds` measurement does not survive that round trip in its last bit.
    assert_eq!(
        selected_identity(&resumed, "SH010"),
        selected_identity(&record, "SH010"),
        "the selected take survived the cancel and the resume"
    );
    assert_eq!(resumed.shot("SH020").unwrap().selected_attempt, Some(2));
    assert!(resumed
        .decisions
        .iter()
        .any(|decision| decision.action == "resume"));
}

#[tokio::test]
async fn a_cancel_sentinel_written_by_another_process_stops_the_run() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
        });
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    // The control a `film-harness run` builds: it watches its own run directory, which is how
    // `film-harness cancel --out DIR` in another shell reaches it.
    let control = RunControl::watching(&harness.out_dir());
    let out_dir = harness.out_dir();
    let app = harness.app.clone();
    let waiter = tokio::spawn(async move {
        wait_for_running_shot(&app, "SH010").await;
        film_harness::request_cancel(&out_dir).expect("sentinel writes");
    });
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    waiter.await.expect("waiter joins");
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    assert!(record.stop.as_ref().expect("stop").resumable);
    assert_eq!(
        record.shot("SH010").unwrap().attempts.len(),
        1,
        "the attempt cap was never spent on a cancel"
    );
    assert!(
        harness.out_dir().join("cancel.requested").exists(),
        "the sentinel is left for the operator to see"
    );

    // The stale sentinel must not cancel the resume the operator asked for.
    harness.script.lock().behaviors.clear();
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert!(!harness.out_dir().join("cancel.requested").exists());
}

#[tokio::test]
async fn an_exhausted_budget_is_terminal_and_refuses_a_resume() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 12, "maxShotSeconds": 12, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(record.outcome, RunOutcome::StoppedRunBudget);
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "run_budget");
    assert!(
        !stop.resumable,
        "resuming an exhausted budget is exactly the unbounded retry the budget exists to prevent"
    );
    assert!(
        stop.detail.contains("maxRunSeconds"),
        "the terminal reason says what to change: {}",
        stop.detail
    );
    let before = harness.video_job_count();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("terminal runs refuse a resume");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("not resumable") && message.contains("run_budget")),
        "{error}"
    );
    assert_eq!(
        harness.video_job_count(),
        before,
        "the refusal dispatched nothing"
    );
}

/// The wall-clock budget bounds the RUN, not one attempt at it. A resume inherits what earlier
/// controllers already spent, which is what stops a restart loop from turning a bounded run into an
/// unbounded one.
#[tokio::test]
async fn a_resume_inherits_the_wall_clock_already_spent() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, None);
    let control = RunControl::watching(&harness.out_dir());
    let out_dir = harness.out_dir();
    let app = harness.app.clone();
    let waiter = tokio::spawn(async move {
        wait_for_running_shot(&app, "SH010").await;
        film_harness::request_cancel(&out_dir).expect("sentinel writes");
    });
    let record = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .unwrap();
    waiter.await.expect("waiter joins");
    assert_eq!(record.outcome, RunOutcome::Canceled, "{}", summary(&record));
    assert!(record.is_resumable());
    let dispatched = harness.api_video_job_count().await;

    // Book the run as having spent its whole budget, exactly as a long first controller would have.
    let path = harness.out_dir().join("run.json");
    let mut on_disk: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    on_disk["elapsedSeconds"] = json!(record.limits.max_run_seconds);
    std::fs::write(&path, serde_json::to_string_pretty(&on_disk).unwrap()).unwrap();

    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("a spent budget still returns a record");
    assert_eq!(resumed.outcome, RunOutcome::StoppedRunBudget);
    let stop = resumed.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "run_budget");
    assert!(
        !stop.resumable,
        "the budget is spent; another resume would be the unbounded retry it exists to prevent"
    );
    assert_eq!(
        harness.api_video_job_count().await,
        dispatched,
        "a resume with no budget left dispatches nothing"
    );
    assert!(resumed.timeline.is_none(), "and assembles nothing");
    // And it really is terminal now.
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("terminal");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("run_budget")),
        "{error}"
    );
}

#[tokio::test]
async fn resume_refuses_a_plan_that_changed_under_it() {
    let harness = Harness::start(true, vec![("SH010", VideoBehavior::Hang)]).await;
    let plan_path = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 1, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan_path.clone(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .unwrap();
    assert_ne!(record.outcome, RunOutcome::Completed);

    // Edit the plan the run was started from, then try to resume onto it.
    let text = std::fs::read_to_string(&plan_path).unwrap();
    let mut plan: Value = serde_json::from_str(&text).unwrap();
    plan["shots"][0]["prompt"] = json!("an entirely different shot");
    std::fs::write(&plan_path, serde_json::to_string_pretty(&plan).unwrap()).unwrap();
    // The copy the run kept beside its record would otherwise satisfy the read, so remove it: the
    // point is that the SOURCE no longer hashes to what the run recorded.
    std::fs::remove_file(harness.out_dir().join("plan.json")).unwrap();
    let error = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect_err("an edited plan is a new run");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("the plan changed since run")),
        "{error}"
    );
}

#[tokio::test]
async fn replacing_a_take_renders_one_more_and_leaves_every_other_shot_untouched() {
    let harness = Harness::start(true, fast(&["SH010", "SH020", "SH030"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(
        before.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&before)
    );
    assert_eq!(harness.video_job_count(), 3);
    let untouched_before = other_shots_digest(&before, "SH020");
    let references_before = serde_json::to_value(&before.references).unwrap();
    let rejected_asset = before
        .shot("SH020")
        .and_then(|shot| shot.selected())
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone())
        .expect("SH020 had a take");

    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "the parcel is the wrong red",
    )
    .await
    .expect("replacement runs");

    // Exactly one more job, for SH020 alone.
    assert_eq!(
        harness.video_job_count(),
        4,
        "one replacement attempt, no re-render of anything else\n{}",
        summary(&after)
    );
    assert_eq!(
        other_shots_digest(&after, "SH020"),
        untouched_before,
        "every other shot's record changed under a replacement"
    );
    assert_eq!(
        serde_json::to_value(&after.references).unwrap(),
        references_before,
        "the imported references are untouched"
    );

    // The rejected take is still there, with its reason, beside the one that replaced it.
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 2);
    let rejection = sh020.attempts[0]
        .rejection
        .as_ref()
        .expect("the old take is marked rejected");
    assert_eq!(rejection.reason, "the parcel is the wrong red");
    assert_eq!(
        sh020.attempts[0].take.as_ref().unwrap().asset_id,
        rejected_asset,
        "the rejected take keeps its asset and provenance"
    );
    assert_eq!(sh020.selected_attempt, Some(2));
    assert!(
        sh020.attempts[1].human_requested,
        "a replacement is a decision, not a retry"
    );
    assert_ne!(
        sh020.attempts[1].take.as_ref().unwrap().asset_id,
        rejected_asset
    );

    // The rejected asset is still the project's — a replacement discards nothing.
    let project_id = after.project_id.clone().unwrap();
    let (status, asset) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{rejected_asset}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{asset}");

    // The declared dependent is flagged, and only it; nothing was regenerated for it.
    let sh030 = after.shot("SH030").expect("SH030 recorded");
    assert_eq!(sh030.needs_review.len(), 1, "{:#?}", sh030.needs_review);
    let flag = &sh030.needs_review[0];
    assert_eq!(flag.source_shot_id, "SH020");
    assert_eq!(flag.dependency, "continuity");
    assert!(
        flag.reason.contains("the parcel is the wrong red"),
        "{}",
        flag.reason
    );
    assert_eq!(sh030.attempts.len(), 1, "a flagged shot is not re-rendered");
    assert!(
        after.shot("SH010").unwrap().needs_review.is_empty(),
        "SH020 depends on SH010, not the other way round"
    );

    // The timeline now carries the new take and nothing else moved; the export is flagged stale.
    let timeline = after.timeline.as_ref().expect("timeline");
    let item = timeline
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH020"))
        .expect("SH020 on the timeline");
    assert_eq!(
        item.asset_id,
        sh020.selected().unwrap().take.as_ref().unwrap().asset_id
    );
    for shot_id in ["SH010", "SH030"] {
        let before_item = before
            .timeline
            .as_ref()
            .unwrap()
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .unwrap();
        let after_item = timeline
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .unwrap();
        assert_eq!(before_item, after_item, "{shot_id}'s timeline item moved");
    }
    let export = after.export.as_ref().expect("the export record is kept");
    assert!(export.stale, "a changed take makes the rendered MP4 stale");
    assert_eq!(
        harness
            .script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "timeline_export")
            .count(),
        1,
        "without --export the replacement re-renders nothing"
    );
    assert!(after
        .decisions
        .iter()
        .any(|decision| decision.action == "replace_take"
            && decision.shot_id.as_deref() == Some("SH020")
            && decision.detail.contains("the parcel is the wrong red")));
    // Choosing not to re-export is not a failure: the replacement landed, the MP4 is just stale.
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));
    assert!(after.stop.is_none());
}

/// A failed export is the one stop a plain `resume` is meant to fix, so it must actually dispatch a
/// NEW export job rather than re-adopt the failed one.
#[tokio::test]
async fn a_failed_export_is_resumable_and_the_retry_is_a_new_job() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    harness.script.lock().export_fails = true;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("a failed export still returns its record");
    assert_eq!(record.outcome, RunOutcome::Failed, "{}", summary(&record));
    let stop = record.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "export_failed");
    assert!(
        stop.resumable,
        "the shots rendered; only the export needs redoing"
    );
    let failed_export = record
        .export
        .as_ref()
        .expect("export recorded")
        .job_id
        .clone();
    assert_eq!(
        record.shot("SH010").unwrap().selected_attempt,
        Some(1),
        "the take is kept"
    );

    harness.script.lock().export_fails = false;
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    let export = resumed.export.as_ref().expect("export recorded");
    assert_ne!(
        export.job_id, failed_export,
        "the resume re-adopted the failed export instead of running a new one"
    );
    assert_eq!(export.status, "completed");
    assert!(export.asset_id.is_some());
    assert_eq!(
        harness.video_job_count(),
        1,
        "resuming to fix the export re-rendered nothing"
    );
}

/// Every shot except `except`, serialized and hashed — the "nothing else moved" assertion.
///
/// Review flags are cleared first, because a flag is precisely the change a replacement IS allowed
/// to make to a dependent. Everything else — attempts, jobs, takes, selection, outcome — must hash
/// identically before and after.
fn other_shots_digest(record: &RunRecord, except: &str) -> String {
    let value = serde_json::to_value(
        record
            .shots
            .iter()
            .filter(|shot| shot.shot_id != except)
            .map(|shot| {
                let mut shot = shot.clone();
                shot.needs_review.clear();
                shot
            })
            .collect::<Vec<_>>(),
    )
    .expect("shots serialize");
    let digest = <sha2::Sha256 as sha2::Digest>::digest(value.to_string().as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[tokio::test]
async fn replacing_a_take_with_export_re_renders_the_timeline_once() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let first_export = before.export.as_ref().expect("export").job_id.clone();

    let after = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "flat lighting",
    )
    .await
    .expect("replacement runs");
    assert_eq!(after.outcome, RunOutcome::Completed, "{}", summary(&after));
    let export = after.export.as_ref().expect("export");
    assert!(!export.stale, "the re-export is current again");
    assert_ne!(export.job_id, first_export, "a new export job ran");
    assert_eq!(export.status, "completed");
    assert_eq!(harness.export_job_count(), 2, "exactly one re-export");
    assert!(after.export_pending.is_none());

    // A SECOND replacement is where excluding only the most recently superseded export breaks: the
    // candidate set still holds the ORIGINAL export job, and adopting it would record the MP4 of
    // the timeline from before BOTH replacements as the current one.
    let again = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "still flat",
    )
    .await
    .expect("second replacement runs");
    let third = again.export.as_ref().expect("export");
    assert_ne!(
        third.job_id,
        first_export,
        "the second re-export adopted the first export\n{}",
        summary(&again)
    );
    assert_ne!(third.job_id, export.job_id);
    assert_eq!(third.status, "completed");
    assert!(!third.stale);
    assert_eq!(
        harness.export_job_count(),
        3,
        "the second replacement ran no export at all\n{}",
        summary(&again)
    );
    assert_eq!(
        again.superseded_export_job_ids,
        vec![first_export, export.job_id.clone()],
        "every export the record has held is excluded, not only the last"
    );
}

#[tokio::test]
async fn a_failed_replacement_keeps_the_rejection_and_does_not_loop() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let before = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(before.outcome, RunOutcome::Completed);
    let rejected_asset = before
        .shot("SH020")
        .and_then(|shot| shot.selected())
        .and_then(|attempt| attempt.take.as_ref())
        .map(|take| take.asset_id.clone())
        .expect("SH020 had a take");
    // Replace SH020's scripted behavior; `behavior_for` takes the first match, so pushing would
    // be shadowed by the entry the run already used.
    {
        let mut script = harness.script.lock();
        script.behaviors.retain(|(id, _)| id != "SH020");
        script
            .behaviors
            .push(("SH020".to_owned(), VideoBehavior::FailAlways));
    }

    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH020",
        "reshoot the approach",
    )
    .await
    .expect("a failed replacement still returns its record");
    assert_eq!(
        harness.video_job_count(),
        3,
        "exactly one replacement attempt, no retry loop\n{}",
        summary(&after)
    );
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.attempts.len(), 2);
    assert!(
        sh020.attempts[0].rejection.is_some(),
        "the human's rejection stands even though the replacement failed"
    );
    assert!(
        sh020.attempts[0].take.is_some(),
        "the failure evidence — and the old take — are preserved"
    );
    assert_eq!(sh020.attempts[1].status, "failed");
    assert_eq!(
        sh020.attempts[1].error.as_deref(),
        Some("fake engine fault: persistent")
    );
    assert_eq!(sh020.selected_attempt, None);
    assert_eq!(sh020.outcome, ShotOutcome::Failed);
    let stop = after.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "replacement_failed");
    assert!(!stop.resumable, "nothing automatic will try again");
    assert!(
        after.shot("SH010").unwrap().needs_review.is_empty(),
        "a failed replacement flags nothing"
    );

    // The shot now has NO selected take, and the timeline was deliberately not rewritten — so the
    // sequence and the MP4 rendered from it still carry the take the human REJECTED. The record has
    // to say so rather than leave `stale: false` claiming the export is current.
    let timeline = after.timeline.as_ref().expect("the timeline is kept");
    let item = timeline
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH020"))
        .expect("SH020 is still in the sequence");
    assert_eq!(
        item.asset_id, rejected_asset,
        "the timeline still names the rejected take"
    );
    let export = after.export.as_ref().expect("the export record is kept");
    assert!(
        export.stale,
        "the MP4 renders a timeline that carries a take the human rejected, so it is NOT current"
    );
    assert!(
        after
            .decisions
            .iter()
            .any(|decision| decision.action == "replace_take"
                && decision.shot_id.as_deref() == Some("SH020")
                && decision.detail.contains("still carry the REJECTED take")),
        "{:#?}",
        after.decisions
    );
    assert!(
        stop.detail.contains("REJECTED take"),
        "the stop says what the delivered sequence actually holds: {}",
        stop.detail
    );
}

#[tokio::test]
async fn replace_take_refuses_a_shot_the_run_does_not_hold() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    for (shot, expected) in [
        ("SH999", "is not a shot in plan"),
        ("SH020", "is not in run"),
    ] {
        let error =
            film_harness::replace_take(&harness.transport, &harness.resume_options(), shot, "nope")
                .await
                .expect_err("refused");
        assert!(
            matches!(&error, HarnessError::Refused(message) if message.contains(expected)),
            "{shot}: {error}"
        );
    }
    assert_eq!(harness.video_job_count(), 1, "a refusal dispatches nothing");
}

/// Replacing a take while the shot still owes an attempt would orphan that attempt's job, so it is
/// refused and the operator is pointed at the command that settles it.
#[tokio::test]
async fn replace_take_refuses_while_the_shot_still_has_an_unsettled_attempt() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    // Die right after the video job is created: the record holds an attempt with no settled status.
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
        .on_post_route("/api/v1/video/jobs");
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    // The API holds the job even though the worker may not have claimed it yet.
    assert_eq!(harness.api_video_job_count().await, 1);

    let error = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "too soon",
    )
    .await
    .expect_err("an unsettled attempt blocks a replacement");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("still has attempt 1 in flight") && message.contains("resume")),
        "{error}"
    );
    assert_eq!(
        harness.api_video_job_count().await,
        1,
        "the refusal dispatched nothing"
    );
    // The record is untouched by the refusal: no rejection, no extra attempt.
    let record = harness_record(&harness);
    let shot = record.shot("SH010").expect("SH010 recorded");
    assert_eq!(shot.attempts.len(), 1);
    assert!(shot.attempts[0].rejection.is_none());

    // Settling it first is exactly what the refusal asked for, and then the replacement works.
    let resumed = harness.resume_to_completion().await;
    assert_eq!(resumed.shot("SH010").unwrap().selected_attempt, Some(1));
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let replaced = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "now it can be replaced",
    )
    .await
    .expect("replacement runs once the attempt settled");
    assert_eq!(replaced.shot("SH010").unwrap().selected_attempt, Some(2));
    assert_eq!(harness.video_job_count(), 2);
}

#[tokio::test]
async fn a_conditioning_dependency_is_flagged_with_its_own_kind() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    // SH020's conditioning is declared as coming out of SH010's take rather than a pack plate.
    let plan = harness.edited_plan(|plan| {
        plan["shots"][1]["dependsOn"] = json!([
            { "shotId": "SH010", "kind": "conditioning", "note": "first frame is SH010's last frame" }
        ]);
    });
    let options = harness.options(plan, harness.fixture_pack(), Some(&["SH010", "SH020"]));
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after =
        film_harness::replace_take(&harness.transport, &resume_options, "SH010", "wrong door")
            .await
            .expect("replacement runs");
    let sh020 = after.shot("SH020").expect("SH020 recorded");
    assert_eq!(sh020.needs_review.len(), 1);
    assert_eq!(sh020.needs_review[0].dependency, "conditioning");
    assert!(
        sh020.needs_review[0]
            .reason
            .contains("first frame is SH010's last frame"),
        "{}",
        sh020.needs_review[0].reason
    );
    assert_eq!(
        sh020.attempts.len(),
        1,
        "the flagged shot was not re-rendered"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-22711 review: the windows the first pass left open
// ---------------------------------------------------------------------------------------------

/// A SECOND re-export must dispatch a third job, not adopt the FIRST export.
///
/// Every export the run ever dispatches renders the same timeline, which is the only key the export
/// route gives the harness. Excluding just the most recently superseded job leaves the original
/// export in the candidate set, and it is a completed `timeline_export` job — so the harness would
/// adopt it, record `status: completed, stale: false` with its pre-replacement asset, and the record
/// would claim the delivered MP4 is current while it is the sequence from before both replacements.
#[tokio::test]
async fn a_second_re_export_dispatches_a_new_job_instead_of_adopting_the_first() {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let export_one = first.export.as_ref().expect("export").job_id.clone();

    let second = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "flat lighting",
    )
    .await
    .expect("first replacement runs");
    let export_two = second.export.as_ref().expect("export").job_id.clone();
    assert_ne!(export_two, export_one, "the first re-export ran a new job");

    let third = film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH020",
        "the parcel is the wrong red",
    )
    .await
    .expect("second replacement runs");
    let export_three = third.export.as_ref().expect("export");
    assert_ne!(
        export_three.job_id,
        export_one,
        "the second re-export adopted the ORIGINAL export — its asset is the timeline from before \
         both replacements\n{}",
        summary(&third)
    );
    assert_ne!(export_three.job_id, export_two);
    assert_eq!(export_three.status, "completed");
    assert!(!export_three.stale);
    assert_eq!(
        harness.export_job_count(),
        3,
        "three exports were asked for, so three must have run\n{}",
        summary(&third)
    );
    // Both superseded exports are remembered, which is what keeps the exclusion cumulative.
    assert_eq!(
        third.superseded_export_job_ids,
        vec![export_one, export_two]
    );
    // And the MP4 the record points at really is the one rendered from the current takes.
    let timeline = third.timeline.as_ref().expect("timeline");
    for shot_id in ["SH010", "SH020"] {
        let selected = third
            .shot(shot_id)
            .and_then(|shot| shot.selected())
            .and_then(|attempt| attempt.take.as_ref())
            .expect("a selected take");
        let item = timeline
            .items
            .iter()
            .find(|item| item.shot_id.as_deref() == Some(shot_id))
            .expect("on the timeline");
        assert_eq!(item.asset_id, selected.asset_id, "{shot_id}");
    }
}

/// `POST /timelines` always creates a NEW row, and the record only learns the timeline id after the
/// PUT — so a controller killed in that window leaves a timeline nothing names. The resume must
/// adopt it by the name it was created under, exactly as it adopts the project.
#[tokio::test]
async fn a_crash_after_the_timeline_was_created_adopts_it_instead_of_creating_a_second() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport =
        FaultTransport::new(harness.app.clone(), 1, FaultMode::After).on_post_route("/timelines");
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    assert!(transport.fired());

    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.timeline.is_none(),
        "the controller died before it could record the timeline"
    );
    let created = harness.timelines_for(&crashed).await;
    assert_eq!(created.len(), 1, "the API really created one: {created:#?}");

    let record = harness.resume_to_completion().await;
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    let timelines = harness.timelines_for(&record).await;
    assert_eq!(
        timelines.len(),
        1,
        "the resume created a SECOND timeline instead of adopting the one it already made: \
         {timelines:#?}"
    );
    let timeline = record.timeline.as_ref().expect("timeline recorded");
    assert_eq!(
        timelines[0]["id"].as_str(),
        Some(timeline.timeline_id.as_str())
    );
    assert_eq!(timelines[0]["name"].as_str(), Some(timeline.name.as_str()));
    assert_eq!(timeline.items.len(), 1);
}

/// The memory budget is judged on the EVIDENCE, not on who was watching when it landed: a resume
/// that adopts an over-budget terminal attempt must stop new dispatch exactly as the controller that
/// watched it would have (E5 / AC3).
#[tokio::test]
async fn an_over_budget_peak_adopted_on_a_resume_stops_new_dispatch() {
    // 90% of the 128 GiB the fake worker reports is 115.2 GiB, over the fixture's 96 GB budget.
    let harness = Harness::start(
        true,
        vec![(
            "SH010",
            VideoBehavior::Complete {
                delay_secs: 0,
                peak_pct: 90.0,
            },
        )],
    )
    .await;
    // Die right after the video job POST: the API holds the job, the record does not know its id.
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::After)
        .on_post_route("/api/v1/video/jobs");
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.shot("SH010").expect("SH010").attempts[0]
            .job_id
            .is_none(),
        "the controller never learned the job id, so the resume must reconcile it"
    );
    // Let the render finish before the resume, so the attempt is ADOPTED rather than polled.
    wait_for_settled_shot(&harness.app, "SH010").await;

    let resumed = film_harness::resume(&harness.transport, &harness.resume_options())
        .await
        .expect("a stopped run still returns its record");
    assert_eq!(
        resumed.outcome,
        RunOutcome::StoppedMemoryLimit,
        "an adopted over-budget peak must stop the run\n{}",
        summary(&resumed)
    );
    let stop = resumed.stop.as_ref().expect("a stop reason");
    assert_eq!(stop.reason, "memory_limit");
    assert!(
        !stop.resumable,
        "the budget is blown; another pass is not the fix"
    );
    let attempt = &resumed.shot("SH010").expect("SH010").attempts[0];
    assert_eq!(
        attempt.peak_memory_source.as_deref(),
        Some("metrics.peakMemoryBytes")
    );
    assert!(
        attempt
            .peak_memory_gb
            .is_some_and(|gb| (gb - 115.2).abs() < 0.01),
        "{:?}",
        attempt.peak_memory_gb
    );
    assert_eq!(
        resumed.shot("SH010").expect("SH010").outcome,
        ShotOutcome::Rendered,
        "the take that was produced is kept"
    );
    assert_eq!(
        resumed.shot("SH020").expect("SH020").outcome,
        ShotOutcome::NotDispatched,
        "nothing new goes out against a budget the evidence says was blown\n{}",
        summary(&resumed)
    );
    assert_eq!(
        harness.api_video_job_count().await,
        1,
        "the resume dispatched a second render anyway"
    );
}

/// `replace-take` decides ONE shot's outcome. Closing the run through the whole-run classifier
/// overwrote a resumable stop with `attempts_exhausted` / `resumable: false`, which permanently
/// blocks the `resume` that was going to render the remaining shots.
#[tokio::test]
async fn replacing_a_take_leaves_a_canceled_runs_resumable_stop_in_place() {
    let harness = Harness::start(
        true,
        vec![
            (
                "SH010",
                VideoBehavior::Complete {
                    delay_secs: 0,
                    peak_pct: 40.0,
                },
            ),
            ("SH020", VideoBehavior::Hang),
        ],
    )
    .await;
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 2, "maxMemoryGb": 96
        });
    });
    let options = harness.options(
        plan,
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    let control = RunControl::new();
    let app = harness.app.clone();
    let waiter = tokio::spawn({
        let control = control.clone();
        async move {
            wait_for_running_shot(&app, "SH020").await;
            control.cancel();
        }
    });
    let canceled = film_harness::run_with_control(&harness.transport, &options, &control)
        .await
        .expect("a canceled run still returns its record");
    waiter.await.expect("waiter joins");
    assert_eq!(
        canceled.outcome,
        RunOutcome::Canceled,
        "{}",
        summary(&canceled)
    );
    assert!(canceled.is_resumable());
    assert!(canceled
        .shot("SH020")
        .expect("SH020")
        .selected_attempt
        .is_none());

    // The human replaces the ONE take the run does have. SH020 and SH030 still owe work.
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "flat lighting",
    )
    .await
    .expect("replacement runs");
    assert_eq!(
        after.shot("SH010").expect("SH010").selected_attempt,
        Some(2)
    );
    let stop = after.stop.as_ref().expect("the run's own stop is kept");
    assert_eq!(
        stop.reason,
        "canceled",
        "replacing one take must not re-classify the run\n{}",
        summary(&after)
    );
    assert!(
        stop.resumable,
        "the replacement made a resumable run terminal, so SH020/SH030 can never be rendered\n{}",
        summary(&after)
    );
    assert_eq!(after.outcome, RunOutcome::Canceled);
    assert!(after.is_resumable());

    // And the resume that stop promises really does finish the run, keeping the replacement.
    harness
        .script
        .lock()
        .behaviors
        .retain(|(id, _)| id != "SH020");
    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&resumed)
    );
    assert_eq!(
        resumed.shot("SH010").expect("SH010").selected_attempt,
        Some(2),
        "the replacement take survived the resume"
    );
    for shot_id in ["SH020", "SH030"] {
        assert_eq!(
            resumed.shot(shot_id).expect("recorded").outcome,
            ShotOutcome::Rendered,
            "{shot_id}"
        );
    }
}

/// An attempt is recorded BEFORE its job is created — that is what makes the idempotency key work.
/// A controller that died in that window rendered nothing, so the next morning's resume must not
/// charge it the wall clock since: with `maxAttemptsPerShot: 1` that spends the shot's only attempt
/// on a job it dispatches and cancels on the first poll.
#[tokio::test]
async fn an_attempt_whose_job_post_never_landed_is_not_charged_wall_clock() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport = FaultTransport::new(harness.app.clone(), 1, FaultMode::Before)
        .on_post_route("/api/v1/video/jobs");
    let (plan, pack) = harness.minimal_documents(json!({
        "maxRunSeconds": 600, "maxShotSeconds": 600, "maxAttemptsPerShot": 1, "maxMemoryGb": 96
    }));
    let options = harness.options(plan, pack, Some(&["SH010"]));
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    assert_eq!(
        harness.api_video_job_count().await,
        0,
        "the POST never reached the API"
    );
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    let attempt = &crashed.shot("SH010").expect("SH010").attempts[0];
    assert_eq!(attempt.status, "dispatching");
    assert!(attempt.job_id.is_none());

    // The operator comes back the next morning.
    let yesterday = sceneworks_core::time::format_unix_seconds(
        sceneworks_core::time::now_unix_seconds() - 86_400,
    );
    harness.edit_run_record(|record| {
        record["shots"][0]["attempts"][0]["startedAt"] = json!(yesterday);
    });

    let resumed = harness.resume_to_completion().await;
    assert_eq!(
        resumed.outcome,
        RunOutcome::Completed,
        "an attempt that never dispatched was charged a night's wall clock\n{}",
        summary(&resumed)
    );
    let shot = resumed.shot("SH010").expect("SH010");
    assert_eq!(
        shot.attempts.len(),
        1,
        "the plan allows exactly one automatic attempt"
    );
    assert_eq!(
        shot.attempts[0].status,
        "completed",
        "the attempt timed out against a budget it had never spent\n{}",
        summary(&resumed)
    );
    assert_eq!(shot.selected_attempt, Some(1));
    assert_eq!(shot.outcome, ShotOutcome::Rendered);
    assert_eq!(harness.video_job_count(), 1);
}

/// `run` over a directory that already holds a record would mint a new run id over the old run's
/// takes, decisions and provenance — while `persist_record` keeps the plan/pack copies of the run it
/// just destroyed. `scripts/film-harness-smoke.sh` pins its `--out`, so a second invocation is
/// exactly this (E2).
#[tokio::test]
async fn run_refuses_a_directory_that_already_holds_a_run_record() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    let first = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    assert_eq!(first.outcome, RunOutcome::Completed, "{}", summary(&first));
    let dispatched = harness.video_job_count();

    let error = film_harness::run(&harness.transport, &options)
        .await
        .expect_err("a second run over the same --out is refused");
    assert!(
        matches!(&error, HarnessError::Refused(message)
            if message.contains("already holds run") && message.contains("resume")),
        "{error}"
    );
    // The previous run's record is exactly as it was, and nothing was dispatched against it.
    let on_disk = harness_record(&harness);
    assert_eq!(on_disk.run_id, first.run_id);
    assert_eq!(selections(&on_disk), selections(&first));
    assert_eq!(harness.video_job_count(), dispatched);
    assert_eq!(harness.project_count().await, 1);
}

/// `cancel --out /typo/path` used to create the directory, print "cancel requested" and exit 0 while
/// the real render kept going. A cancel that reaches nothing must say so.
#[test]
fn cancel_refuses_a_directory_that_holds_no_run_record() {
    let temp = tempfile::tempdir().expect("temp dir");
    let missing = temp.path().join("typo").join("run");
    let error = film_harness::request_cancel(&missing).expect_err("there is no run there");
    assert!(
        matches!(&error, HarnessError::Refused(message) if message.contains("no run record in")),
        "{error}"
    );
    assert!(
        !missing.exists(),
        "a mistyped --out must not be created on the way to a cancel nobody receives"
    );

    // With a record in it, the sentinel is written as before.
    let held = temp.path().join("held");
    std::fs::create_dir_all(&held).expect("run dir");
    std::fs::write(held.join("run.json"), "{}").expect("record");
    let sentinel = film_harness::request_cancel(&held).expect("a real run directory is cancelable");
    assert!(sentinel.exists());
}

/// Replacing the same upstream take twice is the same unread signal, not two.
#[tokio::test]
async fn replacing_the_same_take_twice_does_not_duplicate_a_review_flag() {
    let harness = Harness::start(true, fast(&["SH010", "SH020", "SH030"])).await;
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010", "SH020", "SH030"]),
    );
    film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    for reason in ["the parcel is the wrong red", "still the wrong red"] {
        film_harness::replace_take(&harness.transport, &resume_options, "SH020", reason)
            .await
            .expect("replacement runs");
    }
    let record = harness_record(&harness);
    let sh030 = record.shot("SH030").expect("SH030 recorded");
    assert_eq!(
        sh030.needs_review.len(),
        1,
        "one standing flag per (source shot, dependency), not one per replacement: {:#?}",
        sh030.needs_review
    );
    assert_eq!(sh030.needs_review[0].source_shot_id, "SH020");
    assert_eq!(sh030.needs_review[0].dependency, "continuity");
}

/// A human can reject a plate between two controllers, and the asset listing hides rejected and
/// trashed assets by default — so the adoption lookup has to ask for them. Otherwise the replay
/// cannot see the reference it already uploaded and imports a second copy, while the record claims
/// one import.
#[tokio::test]
async fn a_reference_rejected_between_controllers_is_still_adopted_not_imported_again() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let transport =
        FaultTransport::new(harness.app.clone(), 1, FaultMode::After).on_post_route("/assets");
    let options = harness.options(
        harness.fixture_plan(),
        harness.fixture_pack(),
        Some(&["SH010"]),
    );
    film_harness::run(&transport, &options)
        .await
        .expect_err("the fault stops the controller");
    let crashed = film_harness::read_run_record(&harness.out_dir()).expect("record on disk");
    assert!(
        crashed.references.is_empty(),
        "the upload that was answered is the one the record missed"
    );
    let project_id = crashed.project_id.clone().expect("project created");

    // The human rejects the uploaded plate before the resume.
    let (_, assets) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets"),
        Value::Null,
    )
    .await;
    let uploaded = assets
        .as_array()
        .into_iter()
        .flatten()
        .find(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
        .and_then(|asset| asset["id"].as_str())
        .expect("the interrupted upload landed")
        .to_owned();
    let (status, _) = request(
        harness.app.clone(),
        "PATCH",
        &format!("/api/v1/projects/{project_id}/assets/{uploaded}/status"),
        json!({ "rejected": true }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);

    let record = harness.resume_to_completion().await;
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.references.len(), 7);
    let (_, assets) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets?includeRejected=true&includeTrashed=true"),
        Value::Null,
    )
    .await;
    let references: Vec<&Value> = assets
        .as_array()
        .into_iter()
        .flatten()
        .filter(|asset| asset["extra"]["filmHarness"]["kind"] == "reference")
        .collect();
    assert_eq!(
        references.len(),
        7,
        "the rejected plate was invisible to the adoption lookup, so it was imported twice"
    );
    assert!(
        references
            .iter()
            .any(|asset| asset["id"].as_str() == Some(uploaded.as_str())),
        "the adopted asset is the one that was already uploaded"
    );
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
        film_harness::TimelineEdit::SwapTake {
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
    assert_eq!(edits, vec!["trim", "reorder", "swap_take"], "{on_disk}");
    assert!(on_disk["timeline"]["edits"][2]["detail"]
        .as_str()
        .unwrap_or_default()
        .contains(&replacement));

    // The edits reach the RUN RECORD's own shape, not only the timeline block (sc-22712 on
    // sc-22711's schema 2). Three things have to follow an edit, and none of them did before the
    // two stories were merged:

    // 1. The decision log carries every edit, in order, with the shot the edit named.
    let decisions: Vec<(&str, Option<&str>)> = on_disk["decisions"]
        .as_array()
        .expect("decisions recorded")
        .iter()
        .filter_map(|decision| {
            let action = decision["action"].as_str()?;
            ["trim", "reorder", "swap_take"]
                .contains(&action)
                .then(|| (action, decision["shotId"].as_str()))
        })
        .collect();
    assert_eq!(
        decisions,
        vec![
            ("trim", Some("SH010")),
            ("reorder", None),
            ("swap_take", Some("SH010")),
        ],
        "every edit is a human decision about the run: {on_disk}"
    );

    // 2. The export these edits ran against is FLAGGED stale — the sequence moved under the MP4,
    //    and none of these edits passed `--export`. It is never silently re-rendered.
    assert_eq!(
        on_disk["export"]["stale"],
        json!(true),
        "an edited sequence leaves the existing export stale: {on_disk}"
    );

    // 3. The asset swapped in here is a reference PLATE, not a take this run rendered, so there is
    //    no attempt to select and the shot's selection stays exactly where it was. (The other
    //    branch — a swap onto an asset that IS one of the shot's takes — moves the selection, and
    //    is proved in `swapping_onto_an_existing_take_moves_the_shots_selected_attempt`.)
    let sh010 = on_disk["shots"]
        .as_array()
        .expect("shots recorded")
        .iter()
        .find(|shot| shot["shotId"] == json!("SH010"))
        .expect("SH010 recorded");
    assert_eq!(
        sh010["selectedAttempt"],
        json!(1),
        "a swap onto a foreign asset selects no attempt: {sh010}"
    );

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

/// AC1, bounded: a trim is measured against the take it cuts, not taken on faith.
///
/// An out point past the end of the media used to be accepted in silence. `relayout_timeline` then
/// wrote a `timelineEnd` longer than the file, `render_item_segment` returned the DECLARED duration
/// while `-t` gave ffmpeg a short segment, and the exported picture came out shorter than the saved
/// sequence — sound drifting against picture, and a duration in the render sidecar that no file
/// has. `SwapTake` always measured its asset first; `Trim` now does the same.
#[tokio::test]
async fn a_trim_past_the_end_of_the_take_is_clamped_to_the_takes_real_length() {
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
    let take_asset = record
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is in the sequence")
        .asset_id
        .clone();
    let (status, asset) = request(
        harness.app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/assets/{take_asset}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
    let take_seconds = asset["file"]["duration"]
        .as_f64()
        .expect("the imported take carries a measured duration");

    let edit_options = film_harness::EditOptions {
        run_record_path: harness.temp_dir.path().join("run-out/run.json"),
        export: false,
        poll_interval: Duration::from_millis(250),
    };
    let trimmed = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(0.5),
            // Far past the end of a take this caller never measured.
            source_out: Some(take_seconds * 10.0),
        },
    )
    .await
    .expect("a trim past the end of the take is clamped, not refused");
    let trimmed_item = trimmed
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is still in the sequence")
        .clone();
    assert!(
        close(trimmed_item.source_out, take_seconds),
        "the out point must be clamped to the take's real {take_seconds}s, got {}",
        trimmed_item.source_out
    );
    assert!(
        close(
            trimmed_item.timeline_end - trimmed_item.timeline_start,
            take_seconds - 0.5
        ),
        "and the slot the sequence gives the shot must be the media that exists: {trimmed_item:#?}"
    );

    // The saved timeline says the same thing — this is what the exporter reads.
    let saved = saved_timeline(
        &harness.app,
        &project_id,
        &record.timeline.as_ref().expect("timeline").timeline_id,
    )
    .await;
    let saved_item = items_of(&saved, "track_main")
        .iter()
        .find(|item| item["filmHarness"]["shotId"] == json!("SH010"))
        .expect("SH010 on the saved picture track")
        .clone();
    assert!(
        close(saved_item["sourceOut"].as_f64().unwrap(), take_seconds),
        "{saved_item}"
    );

    // A range that lands entirely past the end is an error rather than a clamp, and it says how
    // long the take actually is so the caller can pick a real number.
    let error = film_harness::edit_timeline(
        &harness.transport,
        &edit_options,
        film_harness::TimelineEdit::Trim {
            shot_id: "SH010".to_owned(),
            source_in: Some(take_seconds + 5.0),
            source_out: Some(take_seconds + 9.0),
        },
    )
    .await
    .expect_err("an in point past the end of the take has no usable range");
    let message = error.to_string();
    assert!(
        message.contains(&format!("{take_seconds:.3}")),
        "the refusal must name the take's real length: {message}"
    );
}

/// A re-layout is not a licence to delete the editor's own clips (sc-22712 review).
///
/// The harness re-places what IT placed — a line follows its shot, a bed re-spans the sequence —
/// and a harness-owned clip with nowhere left to go is dropped because the plan can put it back.
/// A clip the harness did not place is none of its business: it is clamped into the sequence and
/// kept. The shared drop used to apply to both, so any editor clip sitting within 40 ms of the end
/// was destroyed by the next edit, with no diagnostic and no undo.
#[tokio::test]
async fn a_relayout_keeps_an_editor_placed_clip_the_harness_never_put_there() {
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
    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let total = saved["duration"].as_f64().expect("duration");
    let asset_id = record
        .references
        .iter()
        .find(|reference| reference.role == "workshop_plate")
        .expect("plate imported")
        .asset_id
        .clone();

    // A clip the editor placed: no `filmHarness` block, sitting right at the end of the sequence
    // and overhanging it.
    let mut edited = saved.clone();
    edited["tracks"]
        .as_array_mut()
        .expect("tracks")
        .iter_mut()
        .find(|track| track["id"] == json!("track_dialogue"))
        .expect("the editor's dialogue lane")["items"]
        .as_array_mut()
        .expect("items")
        .push(json!({
            "id": "editor_clip",
            "trackId": "track_dialogue",
            "assetId": asset_id,
            "type": "audio",
            "displayName": "an editor's own clip",
            "sourceIn": 0.0,
            "sourceOut": 1.0,
            "timelineStart": total - 0.02,
            "timelineEnd": total + 0.5,
            "speed": 1.0,
            "fit": "fit",
            "volume": 1.0,
            "fadeInSeconds": 0.0,
            "fadeOutSeconds": 0.0,
        }));
    let (status, body) = request(
        harness.app.clone(),
        "PUT",
        &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
        json!({ "timeline": edited }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");

    // A reorder does not change the sequence's length, so nothing about this clip is stale.
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

    let saved = saved_timeline(&harness.app, &project_id, &timeline_id).await;
    let kept = items_of(&saved, "track_dialogue")
        .iter()
        .find(|item| item["id"] == json!("editor_clip"))
        .unwrap_or_else(|| {
            panic!(
                "the harness deleted a clip it never placed. A re-layout may re-place the \
                 harness's own dialogue, ambience and music items; the editor's belong to the \
                 editor: {saved}"
            )
        })
        .clone();
    assert!(
        close(kept["timelineStart"].as_f64().unwrap(), total - 0.02),
        "an unowned clip keeps where the editor put it: {kept}"
    );
    assert!(
        close(kept["timelineEnd"].as_f64().unwrap(), total),
        "clamped to the sequence, never stretching it: {kept} against {total}"
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

/// The seam between the two ways a shot's take can change (sc-22711 `replace-take` re-renders,
/// sc-22712 `swap-take` re-cuts): swapping the sequence back onto a take the run ALREADY rendered
/// must move the shot's `selectedAttempt` with it.
///
/// The generation side owns `selectedAttempt` and the edit side owns the timeline item, and before
/// the two stories were merged neither told the other anything. A selection left behind is not
/// cosmetic: `status` reports the wrong take, and a subsequent `replace-take` rejects the attempt
/// the sequence no longer shows while leaving the one it does show in the cut.
#[tokio::test]
async fn swapping_onto_an_existing_take_moves_the_shots_selected_attempt() {
    let harness = Harness::start(true, vec![]).await;
    let options = harness.options(
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
        Some(&["SH010", "SH020"]),
    );
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("run completes");
    let first_take = record
        .shot("SH010")
        .expect("SH010 recorded")
        .attempts
        .last()
        .and_then(|attempt| attempt.take.as_ref())
        .expect("SH010 rendered")
        .asset_id
        .clone();

    // Re-render SH010, which makes attempt 2 the selected one and leaves attempt 1's take in the
    // project, still addressable.
    let mut resume_options = harness.resume_options();
    resume_options.export = false;
    let after = film_harness::replace_take(
        &harness.transport,
        &resume_options,
        "SH010",
        "prefer the first one after all",
    )
    .await
    .expect("replacement runs");
    assert_eq!(
        after
            .shot("SH010")
            .expect("SH010 recorded")
            .selected_attempt,
        Some(2),
        "the re-render is what the shot now carries"
    );

    // Now change our mind on the EDIT side: put attempt 1's take back into the cut.
    let swapped = film_harness::edit_timeline(
        &harness.transport,
        &film_harness::EditOptions {
            run_record_path: harness.temp_dir.path().join("run-out/run.json"),
            export: false,
            poll_interval: Duration::from_millis(250),
        },
        film_harness::TimelineEdit::SwapTake {
            shot_id: "SH010".to_owned(),
            asset_id: first_take.clone(),
        },
    )
    .await
    .expect("swap applies");

    assert_eq!(
        swapped
            .shot("SH010")
            .expect("SH010 recorded")
            .selected_attempt,
        Some(1),
        "the selection follows the sequence back onto attempt 1"
    );
    let item = swapped
        .timeline
        .as_ref()
        .expect("timeline")
        .items
        .iter()
        .find(|item| item.shot_id.as_deref() == Some("SH010"))
        .expect("SH010 is in the sequence");
    assert_eq!(
        item.asset_id, first_take,
        "the record and the sequence name the same take"
    );
}

// ----------------------------------------------------------------------------------------------
// Local planner (sc-22713): brief -> plan -> compiled requests -> dispatch.
//
// The planner drives the SHIPPED LLM seam — `POST /api/v1/prompts/refine`, the `prompt_refine`
// job, the worker's native TextLlm — so these tests create the jobs through the real route and
// script only the model's answer (`run_fake_refine_job`). Nothing here loads weights or touches a
// GPU: what is under test is the request the planner composes, the strictness of the parse, the
// bound on the repair loop, and the conformance of the compiled requests to the INSTALLED manifest.
// ----------------------------------------------------------------------------------------------

const BRIEF_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop/brief.jsonc"
);

/// The six beats the checked-in brief requires, in order.
const BRIEF_BEATS: &[&str] = &[
    "arrival",
    "approach",
    "handover",
    "departure",
    "discovery",
    "opening",
];

/// The roles the checked-in brief declares each beat must show (`requiredBeats[].requiredRoles`).
/// A scripted draft binds them because the validator checks them — the same demand the real
/// planner is held to (sc-22713).
fn beat_roles(beat_id: &str) -> Vec<&'static str> {
    let mut roles = match beat_id {
        "arrival" => vec!["courier", "red_parcel", "workshop_location"],
        "discovery" => vec!["recipient", "red_parcel", "workbench_table"],
        "opening" => vec!["recipient", "red_parcel"],
        // approach / handover / departure, and anything a test invents.
        _ => vec!["courier", "red_parcel", "workbench_table"],
    };
    roles.push("house_style");
    roles
}

/// One shot of a scripted planner draft, on the H3 envelope (24 fps, 576x320, 5.1667s).
fn draft_shot(id: &str, beat_id: &str) -> Value {
    json!({
        "id": id,
        "beatId": beat_id,
        "beat": format!("beat {beat_id}"),
        "framing": "wide static, eye level",
        "prompt": format!("A cluttered woodworking workshop in warm late-afternoon light; {beat_id}."),
        "targetDurationSeconds": 5.1667,
        "startState": "the workshop before this shot",
        "endState": "the workshop after this shot",
        "sound": "room tone, distant birds",
        "conditioning": { "mode": "text_to_video" },
        "seed": 22713,
        "continuityRoles": beat_roles(beat_id)
    })
}

/// A well-formed draft covering every beat of the checked-in brief.
fn full_draft() -> Value {
    json!({
        "shots": BRIEF_BEATS
            .iter()
            .enumerate()
            .map(|(index, beat)| draft_shot(&format!("SH{:03}0", index + 1), beat))
            .collect::<Vec<_>>()
    })
}

fn draft_text(draft: &Value) -> String {
    serde_json::to_string_pretty(draft).expect("draft serializes")
}

fn planner_options(harness: &Harness, out: &str) -> film_planner::PlannerOptions {
    film_planner::PlannerOptions {
        brief_path: PathBuf::from(BRIEF_FIXTURE),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        out_dir: harness.temp_dir.path().join(out),
        max_repair_rounds: 2,
        refine_prompts: false,
        prompt_guide_path: None,
        require_installed: false,
        // Empty: the in-process transport has no URL. The local-only rule is exercised as a unit
        // test in `film_planner` and end to end below.
        api_url: String::new(),
        force: false,
        poll_interval: Duration::from_millis(50),
        job_timeout: Duration::from_secs(30),
    }
}

fn planner_llm(harness: &Harness) -> film_planner::SceneWorksLlm<'_> {
    film_planner::SceneWorksLlm::new(
        &harness.transport,
        Duration::from_millis(50),
        Duration::from_secs(30),
    )
}

fn set_plan_replies(harness: &Harness, replies: Vec<String>) {
    let mut script = harness.script.lock();
    script.plan_replies = replies;
    script.plan_calls = 0;
}

fn refine_job_payloads(harness: &Harness, plan_task_only: bool) -> Vec<Value> {
    harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(job_type, _, payload)| {
            job_type == "prompt_refine" && (!plan_task_only || payload["task"] == "film_plan")
        })
        .map(|(_, _, payload)| payload.clone())
        .collect()
}

fn findings_of(error: HarnessError) -> Vec<String> {
    match error {
        HarnessError::Validation(findings) => findings.iter().map(ToString::to_string).collect(),
        other => panic!("expected a validation refusal, got {other}"),
    }
}

#[tokio::test]
async fn the_brief_produces_a_plan_the_existing_controller_accepts_unchanged() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    harness.script.lock().refine_template = Some(
        "integrated_multimodal_description: {prompt}\noverall_soundscape: room tone".to_owned(),
    );
    let mut options = planner_options(&harness, "planned");
    options.refine_prompts = true;

    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the planner produces a plan");

    // A complete plan: every beat, every narrative field, timing, camera, reference bindings and
    // the intended start/end state of each shot.
    assert_eq!(artifacts.repair_rounds, 0);
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    assert_eq!(artifacts.plan.id, "courier-workshop-planned");
    assert_eq!(artifacts.plan.model.id, "minimax_h3");
    assert_eq!(artifacts.plan.limits.max_memory_gb, 96.0);
    for shot in &artifacts.plan.shots {
        assert!(!shot.beat.trim().is_empty(), "{shot:?}");
        assert!(!shot.framing.trim().is_empty(), "{shot:?}");
        assert!(!shot.start_state.trim().is_empty(), "{shot:?}");
        assert!(!shot.end_state.trim().is_empty(), "{shot:?}");
        assert!(shot.target_duration_seconds > 0.0, "{shot:?}");
        assert!(
            !shot.continuity_roles.is_empty(),
            "every shot binds canonical reference roles: {shot:?}"
        );
    }

    // The plan is a file, and the SAME validator the hand-authored path uses accepts it unchanged.
    assert_eq!(artifacts.plan_path, options.out_dir.join("plan.json"));
    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("planned-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let (plan, _) = film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the generated plan validates against the live catalog");
    assert_eq!(plan.shots.len(), BRIEF_BEATS.len());

    // The compiled requests are a second versioned document, on the installed model's envelope.
    let compiled: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.compiled_path).expect("compiled.json written"),
    )
    .expect("compiled.json parses");
    assert_eq!(compiled["schemaVersion"], 1);
    assert_eq!(compiled["planId"], "courier-workshop-planned");
    assert_eq!(compiled["model"]["fps"], 24);
    let requests = compiled["requests"].as_array().expect("requests");
    assert_eq!(requests.len(), BRIEF_BEATS.len());
    for request in requests {
        assert_eq!(request["fps"], 24);
        assert_eq!(request["width"], 576);
        assert_eq!(request["height"], 320);
        assert_eq!(request["durationSeconds"], 5.1667);
        assert_eq!(request["mode"], "text_to_video");
        assert_eq!(request["promptSource"], "refined");
        assert!(
            request["prompt"]
                .as_str()
                .unwrap()
                .starts_with("integrated_multimodal_description:"),
            "the H3 refinement produced the dispatched prompt: {request}"
        );
        assert!(request.get("negativePrompt").is_none(), "{request}");
        assert!(
            request["referenceRoles"].as_array().unwrap().is_empty(),
            "H3 declares maxReferenceAssets 0: {request}"
        );
    }

    // The planner drove the shipped seam: one film_plan job plus one rewrite job per shot, all of
    // them `prompt_refine`, with the target model forwarded (which is what selects the H3 asset).
    let refine_jobs = refine_job_payloads(&harness, false);
    assert_eq!(refine_jobs.len(), 1 + BRIEF_BEATS.len(), "{refine_jobs:?}");
    assert_eq!(refine_jobs[0]["task"], "film_plan");
    assert_eq!(refine_jobs[0]["modelId"], "minimax_h3");
    assert_eq!(refine_jobs[0]["workflow"], "video");
    for payload in &refine_jobs[1..] {
        assert!(payload.get("task").is_none(), "{payload}");
        assert_eq!(payload["modelId"], "minimax_h3");
    }
    // The planning request carried the brief's beats and only APPROVED roles.
    let request = refine_jobs[0]["prompt"].as_str().unwrap();
    for beat in BRIEF_BEATS {
        assert!(request.contains(beat), "{beat} missing from the request");
    }
    assert!(request.contains("workshop_plate (plate)"), "{request}");
}

#[tokio::test]
async fn a_dropped_beat_is_repaired_and_the_repair_loop_is_bounded() {
    let harness = Harness::start(true, vec![]).await;
    // The first draft drops "handover"; the repair round returns the whole plan.
    let mut short = full_draft();
    short["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    set_plan_replies(
        &harness,
        vec![draft_text(&short), draft_text(&full_draft())],
    );
    let options = planner_options(&harness, "repaired");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("the repair round produces a plan");
    assert_eq!(artifacts.repair_rounds, 1);
    assert_eq!(artifacts.plan.shots.len(), BRIEF_BEATS.len());
    // The repair round was told exactly what was wrong, and the dropped beat was never accepted.
    let repair = refine_job_payloads(&harness, true)
        .get(1)
        .map(|payload| payload["prompt"].as_str().unwrap_or_default().to_owned())
        .expect("a second planning job was created");
    assert!(repair.contains("Repair round 1 of 2"), "{repair}");
    assert!(
        repair.contains("required beat \"handover\""),
        "the finding is handed back verbatim: {repair}"
    );

    // A planner that never covers the beat is refused after the declared rounds — not looped on.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&short)]);
    let options = planner_options(&harness, "exhausted");
    let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect_err("an uncorrected draft is refused");
    let findings = findings_of(error);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("required beat \"handover\"")),
        "{findings:?}"
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("did not produce a valid plan within 2 repair round")),
        "{findings:?}"
    );
    // Exactly 1 + 2 model calls: the loop is bounded by the declared rounds.
    assert_eq!(harness.script.lock().plan_calls, 3);
    // Nothing was written but the diagnosable refusal.
    assert!(!options.out_dir.join("plan.json").exists());
    let rejected = std::fs::read_to_string(options.out_dir.join("planner-rejected.txt"))
        .expect("the refused answer is written out");
    assert!(rejected.contains("\"beatId\": \"arrival\""), "{rejected}");
    // Planning dispatches nothing: no video job was ever created.
    assert!(
        harness
            .jobs()
            .await
            .iter()
            .all(|job| job["type"] == "prompt_refine"),
        "planning created a non-planning job"
    );
}

#[tokio::test]
async fn malformed_and_out_of_envelope_drafts_are_refused_rather_than_coerced() {
    let mut unknown_field = full_draft();
    unknown_field["shots"][0]["cameraLens"] = json!("35mm");
    let mut off_menu = full_draft();
    off_menu["shots"][1]["targetDurationSeconds"] = json!(6.0);
    let mut unsupported = full_draft();
    unsupported["shots"][2]["conditioning"] =
        json!({ "mode": "reference_to_video", "referenceRoles": ["courier"] });
    let mut unanchored = full_draft();
    unanchored["shots"][3]["continuityRoles"] = json!([]);
    unanchored["shots"][3]["conditioning"] =
        json!({ "mode": "text_to_video", "chainFromShotId": "SH0030" });
    let mut missing_asset = full_draft();
    missing_asset["shots"][4]["conditioning"] =
        json!({ "mode": "image_to_video", "firstFrameRole": "a_plate_nobody_approved" });

    let cases: Vec<(&str, String, &str)> = vec![
        (
            "prose",
            "I'd love to help! Here are some ideas for your film...".to_owned(),
            "not a plan document",
        ),
        ("unknown field", draft_text(&unknown_field), "cameraLens"),
        ("off-menu duration", draft_text(&off_menu), "duration menu"),
        (
            "unsupported conditioning",
            draft_text(&unsupported),
            "does not declare reference_to_video",
        ),
        (
            "chain as the only anchor",
            draft_text(&unanchored),
            "the only continuity this shot declares is the chain",
        ),
        (
            "missing reference asset",
            draft_text(&missing_asset),
            "is not in reference pack",
        ),
    ];
    for (label, reply, expected) in cases {
        let harness = Harness::start(true, vec![]).await;
        set_plan_replies(&harness, vec![reply]);
        let mut options = planner_options(&harness, "refused");
        options.max_repair_rounds = 0;
        let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .err()
            .unwrap_or_else(|| panic!("{label}: expected a refusal"));
        let findings = findings_of(error);
        assert!(
            findings.iter().any(|finding| finding.contains(expected)),
            "{label}: {findings:?}"
        );
        assert!(!options.out_dir.join("plan.json").exists(), "{label}");
        // One call: with zero repair rounds the planner asks once and stops.
        assert_eq!(harness.script.lock().plan_calls, 1, "{label}");
    }
}

#[tokio::test]
async fn a_hosted_endpoint_refuses_the_planner_before_it_creates_a_job() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let mut options = planner_options(&harness, "hosted");
    options.api_url = "https://api.openai.com/v1".to_owned();
    let error = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect_err("a hosted endpoint is refused");
    let findings = findings_of(error);
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("not this machine or a private-network host"),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0);
    assert!(harness.jobs().await.is_empty(), "no job was created");
}

#[tokio::test]
async fn planning_without_a_local_refiner_is_refused_rather_than_queued_forever() {
    // No worker at all: nothing can run an LLM job, so the planner says so instead of waiting out
    // its timeout on a job nobody will claim.
    let harness = Harness::start(false, vec![]).await;
    let options = planner_options(&harness, "no-refiner");
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("planning without a refiner is refused"),
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("no registered worker advertises prompt_refine"),
        "{findings:?}"
    );
    assert!(harness.jobs().await.is_empty(), "no job was created");
}

#[tokio::test]
async fn the_plan_is_editable_between_generation_and_dispatch_and_a_stale_compile_is_refused() {
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let options = planner_options(&harness, "editable");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    // The human edits the plan by hand: a different prompt and a longer legal clip.
    let mut plan: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.plan_path).expect("plan readable"),
    )
    .expect("plan parses");
    plan["shots"][0]["prompt"] = json!("A hand-written prompt the planner never wrote.");
    plan["shots"][0]["targetDurationSeconds"] = json!(8.0);
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();

    // The stale compiled document is refused rather than dispatched.
    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("editable-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &run_options)
            .await
            .expect_err("a plan edited after the compile is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("has changed since it was compiled")),
        "{findings:?}"
    );

    // Recompiling adopts the edit, and `--no-refine` keeps the hand-written text verbatim.
    let artifacts = film_planner::compile_existing(
        &harness.transport,
        &planner_llm(&harness),
        &options,
        &artifacts.plan_path,
    )
    .await
    .expect("the edited plan recompiles");
    let first = &artifacts.compiled.requests[0];
    assert_eq!(
        first.prompt,
        "A hand-written prompt the planner never wrote."
    );
    assert_eq!(first.duration_seconds, 8.0);
    film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the recompiled plan validates");

    // An edit that deletes a beat is reported against the brief rather than quietly compiled.
    let mut plan: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.plan_path).expect("plan readable"),
    )
    .unwrap();
    plan["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &options,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a deleted beat is reported"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("\"handover\"")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn a_generated_plan_dispatches_its_compiled_prompts_through_the_same_run_path() {
    let harness = Harness::start(true, vec![]).await;
    // Two beats is enough to prove the dispatch path; the run itself is the sc-22710 one.
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(2);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(2);
    brief["targetTotalSeconds"] = json!({ "min": 10.0, "max": 12.0 });
    let brief_path = harness.temp_dir.path().join("two-beat-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();

    set_plan_replies(&harness, vec![draft_text(&draft)]);
    harness.script.lock().refine_template =
        Some("integrated_multimodal_description: {prompt}".to_owned());
    let mut options = planner_options(&harness, "dispatch");
    options.brief_path = brief_path;
    options.refine_prompts = true;
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        // Found beside the plan, exactly as a run started from the plan directory would.
        compiled_path: None,
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("dispatch-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let record = film_harness::run(&harness.transport, &run_options)
        .await
        .expect("the generated plan runs");
    assert_eq!(
        record.outcome,
        RunOutcome::Completed,
        "{}",
        summary(&record)
    );
    assert_eq!(record.shots.len(), 2);
    assert!(record
        .shots
        .iter()
        .all(|shot| shot.outcome == ShotOutcome::Rendered));
    // The run record pins the compiled document it dispatched, not just the plan.
    let compiled_source = record
        .compiled
        .as_ref()
        .expect("compiled document recorded");
    assert_eq!(compiled_source.id, "courier-workshop-planned");
    assert_eq!(compiled_source.sha256.len(), 64);
    assert!(run_options.out_dir.join("compiled.json").is_file());

    // Every dispatched video job carried the COMPILED prompt, not the plan's authored one.
    let dispatched: Vec<Value> = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(job_type, _, _)| job_type == "video_generate")
        .map(|(_, _, payload)| payload.clone())
        .collect();
    assert_eq!(dispatched.len(), 2);
    for payload in &dispatched {
        let shot_id = payload["advanced"]["filmHarness"]["shotId"]
            .as_str()
            .unwrap();
        let request = artifacts
            .compiled
            .request(shot_id)
            .unwrap_or_else(|| panic!("{shot_id} is in the compiled document"));
        assert_eq!(payload["prompt"], json!(request.prompt));
        assert_eq!(payload["duration"], json!(request.duration_seconds));
        assert_eq!(payload["width"], json!(request.width));
        assert_eq!(payload["height"], json!(request.height));
        assert_eq!(
            payload["advanced"]["filmHarness"]["promptSource"],
            "refined"
        );
        assert_eq!(payload["advanced"]["mlxQuantize"], 4);
        assert!(
            payload["prompt"]
                .as_str()
                .unwrap()
                .starts_with("integrated_multimodal_description:"),
            "{payload}"
        );
    }
}

#[tokio::test]
async fn a_hand_edited_compiled_request_is_refused_instead_of_dispatched() {
    // The compiled document — not the plan — is what becomes the job body: `execute_run` takes the
    // mode, model, duration, fps, geometry, seed, negative prompt and every role slot straight out
    // of it. An edit here therefore reaches the engine unless something judges THIS document, and
    // 9.0s is the case that would not even fail loudly: it is inside H3's hard bounds, so the
    // engine snaps it up onto the 17n+5 lattice and renders a length the plan never claimed.
    let harness = Harness::start(true, vec![]).await;
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(2);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(2);
    brief["targetTotalSeconds"] = json!({ "min": 10.0, "max": 12.0 });
    let brief_path = harness.temp_dir.path().join("tamper-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();

    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "tamper");
    options.brief_path = brief_path;
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    // Edit ONE field of ONE compiled request, leaving the plan and its sha256 untouched, so the
    // staleness check has nothing to say.
    let mut compiled: Value = serde_json::from_str(
        &std::fs::read_to_string(&artifacts.compiled_path).expect("compiled.json"),
    )
    .expect("compiled.json parses");
    let tampered_shot = compiled["requests"][1]["shotId"]
        .as_str()
        .expect("a second request")
        .to_owned();
    compiled["requests"][1]["durationSeconds"] = json!(9.0);
    std::fs::write(
        &artifacts.compiled_path,
        serde_json::to_string_pretty(&compiled).unwrap(),
    )
    .unwrap();

    let run_options = RunOptions {
        plan_path: artifacts.plan_path.clone(),
        reference_pack_path: Path::new(FIXTURE_DIR).join("references.jsonc"),
        compiled_path: Some(artifacts.compiled_path.clone()),
        project_id: None,
        shot_ids: None,
        out_dir: harness.temp_dir.path().join("tamper-run"),
        poll_interval: Duration::from_millis(100),
        export: false,
        require_installed: false,
    };
    let findings = findings_of(
        film_harness::run(&harness.transport, &run_options)
            .await
            .expect_err("a hand-edited compiled request is refused"),
    );
    assert!(
        findings.iter().any(|finding| {
            finding.contains(&format!("[{tampered_shot}]"))
                && finding.contains("compiled.durationSeconds")
                && finding.contains("9s")
        }),
        "{findings:?}"
    );
    // Refused BEFORE dispatch: nothing rendered.
    assert!(
        harness
            .jobs()
            .await
            .iter()
            .all(|job| job["type"] == "prompt_refine"),
        "a video job was created for a request that was refused"
    );
    // `validate` refuses it on the same grounds, so the operator sees it without starting a run.
    let findings = findings_of(
        film_harness::validate(Some(&harness.transport), &run_options)
            .await
            .expect_err("validate refuses it too"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("compiled.durationSeconds")),
        "{findings:?}"
    );

    // Restored, the same documents run.
    compiled["requests"][1]["durationSeconds"] = json!(5.1667);
    std::fs::write(
        &artifacts.compiled_path,
        serde_json::to_string_pretty(&compiled).unwrap(),
    )
    .unwrap();
    film_harness::validate(Some(&harness.transport), &run_options)
        .await
        .expect("the untampered document validates");
}

#[tokio::test]
async fn a_brief_the_model_cannot_render_is_refused_before_a_single_decode() {
    // fps 30 is a property of the BRIEF, which every draft copies verbatim — the planner is not
    // allowed to change it. Discovered on the first draft it would cost 1 + rounds full local
    // decodes (minutes each on an 8B) and then blame the planner for its input.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["model"]["fps"] = json!(30);
    let brief_path = harness.temp_dir.path().join("off-menu-fps-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let mut options = planner_options(&harness, "off-menu-fps");
    options.brief_path = brief_path.clone();

    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("an unrenderable brief is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("model.fps") && finding.contains("30 fps")),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0, "a decode was spent");
    assert!(harness.jobs().await.is_empty(), "a job was created");

    // The memory budget is judged the same way, against the lane the API HOST renders on.
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["limits"]["maxMemoryGb"] = json!(8.0);
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
            .await
            .expect_err("a budget below the model's minimum is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("limits.maxMemoryGb")),
        "{findings:?}"
    );
    assert_eq!(harness.script.lock().plan_calls, 0);
}

#[tokio::test]
async fn the_default_correction_loop_rechecks_beat_coverage_without_a_brief_flag() {
    // The documented loop is `plan --out DIR`, edit `DIR/plan.json`, `compile --plan DIR/plan.json
    // --out DIR`. `compile` looks for a brief beside the plan, so `plan` has to leave one there —
    // otherwise the recompile AC3 asks a human to run performs NO coverage check and a hand edit
    // that deletes a required beat compiles and dispatches silently.
    let harness = Harness::start(true, vec![]).await;
    set_plan_replies(&harness, vec![draft_text(&full_draft())]);
    let options = planner_options(&harness, "loop");
    let artifacts = film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");
    let sibling = options.out_dir.join("brief.json");
    assert!(sibling.is_file(), "the brief travels with the plan");
    assert_eq!(
        std::fs::read(&sibling).unwrap(),
        std::fs::read(BRIEF_FIXTURE).unwrap(),
        "the brief is copied byte for byte, comments and all"
    );

    // Now the human deletes a beat's shot, and compiles WITHOUT naming a brief.
    let mut plan: Value =
        serde_json::from_str(&std::fs::read_to_string(&artifacts.plan_path).unwrap()).unwrap();
    plan["shots"]
        .as_array_mut()
        .unwrap()
        .retain(|shot| shot["beatId"] != "handover");
    std::fs::write(
        &artifacts.plan_path,
        serde_json::to_string_pretty(&plan).unwrap(),
    )
    .unwrap();
    let mut blind = options.clone();
    blind.brief_path = harness.temp_dir.path().join("no-such-brief.json");
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &blind,
            &artifacts.plan_path,
        )
        .await
        .expect_err("the deleted beat is caught by the discovered sibling brief"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("\"handover\"")),
        "{findings:?}"
    );

    // A brief the caller NAMED but that cannot be read is an error, not a silent skip: a typo in
    // `--brief` must not be indistinguishable from "coverage verified".
    let malformed = harness.temp_dir.path().join("malformed-brief.json");
    std::fs::write(&malformed, "{ \"schemaVersion\": 1, ").unwrap();
    let mut named = options.clone();
    named.brief_path = malformed.clone();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &named,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a malformed named brief is an error"),
    );
    assert!(
        findings.iter().any(|finding| finding.contains("brief")),
        "{findings:?}"
    );

    // So is one that parses but is not a brief this build accepts.
    let mut stale: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    stale["schemaVersion"] = json!(99);
    std::fs::write(&malformed, serde_json::to_string_pretty(&stale).unwrap()).unwrap();
    let findings = findings_of(
        film_planner::compile_existing(
            &harness.transport,
            &planner_llm(&harness),
            &named,
            &artifacts.plan_path,
        )
        .await
        .expect_err("a stale schema version is an error"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("brief schema version 99")),
        "{findings:?}"
    );
}

#[tokio::test]
async fn the_prompt_guide_reaches_the_rewrite_the_way_video_studio_sends_it() {
    // The per-shot rewrite is only the SAME rewrite the "Refine" button runs if the model's prompt
    // guide rides with it: the web forwards `guide`, and the worker appends it to the H3 system
    // turn under `# Model prompt guide`. The harness cannot fetch it from `--api` (the rust-api
    // serves `/prompt-guides/` only in an `embed-web` build), so it is read from disk.
    let harness = Harness::start(true, vec![]).await;
    let mut draft = full_draft();
    draft["shots"].as_array_mut().unwrap().truncate(1);
    let mut brief: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        &std::fs::read_to_string(BRIEF_FIXTURE).unwrap(),
    ))
    .unwrap();
    brief["requiredBeats"].as_array_mut().unwrap().truncate(1);
    brief["targetTotalSeconds"] = json!({ "min": 5.0, "max": 6.0 });
    let brief_path = harness.temp_dir.path().join("one-beat-brief.json");
    std::fs::write(&brief_path, serde_json::to_string_pretty(&brief).unwrap()).unwrap();
    let guide_path = harness.temp_dir.path().join("h3-guide.md");
    std::fs::write(&guide_path, "# H3\nWrite one paragraph.").unwrap();

    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let mut options = planner_options(&harness, "guided");
    options.brief_path = brief_path;
    options.refine_prompts = true;
    options.prompt_guide_path = Some(guide_path);
    film_planner::generate(&harness.transport, &planner_llm(&harness), &options)
        .await
        .expect("plan generates");

    let jobs = refine_job_payloads(&harness, false);
    assert_eq!(jobs.len(), 2, "{jobs:?}");
    // The planning turn carries no guide: its system turn is the plan contract, not prompt advice.
    assert!(jobs[0].get("guide").is_none(), "{:?}", jobs[0]);
    assert_eq!(jobs[1]["guide"], "# H3\nWrite one paragraph.");

    // A guide the caller NAMED and that is not there is an error, not a guide-less rewrite.
    let mut missing = options.clone();
    missing.out_dir = harness.temp_dir.path().join("guide-missing");
    missing.prompt_guide_path = Some(harness.temp_dir.path().join("no-such-guide.md"));
    set_plan_replies(&harness, vec![draft_text(&draft)]);
    let findings = findings_of(
        film_planner::generate(&harness.transport, &planner_llm(&harness), &missing)
            .await
            .expect_err("an unreadable named guide is refused"),
    );
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("cannot read the prompt guide")),
        "{findings:?}"
    );
}

#[test]
fn the_checked_in_brief_is_valid_and_matches_the_hand_authored_baseline() {
    let text = std::fs::read_to_string(BRIEF_FIXTURE).expect("brief fixture readable");
    let brief = sceneworks_core::film_planner::parse_brief(&text).expect("brief parses");
    let findings = sceneworks_core::film_planner::validate_brief(&brief);
    assert!(findings.is_empty(), "{findings:?}");
    let ids: Vec<&str> = brief
        .required_beats
        .iter()
        .map(|beat| beat.id.as_str())
        .collect();
    assert_eq!(ids, BRIEF_BEATS);
    // The brief plans the SAME model, tier and canvas the hand-authored baseline renders, so the
    // two plans are comparable, and it carries a DIFFERENT plan id so neither overwrites the other.
    let baseline = sceneworks_core::film_plan::parse_plan(
        &std::fs::read_to_string(Path::new(FIXTURE_DIR).join("plan.jsonc")).unwrap(),
    )
    .unwrap();
    assert_eq!(brief.model, baseline.model);
    assert_eq!(brief.limits, baseline.limits);
    assert_ne!(brief.id, baseline.id);
    // Every beat is coverable inside the model's shortest legal clip and the declared window.
    assert!(brief.required_beats.len() as f64 * 5.1667 >= brief.target_total_seconds.min);
    assert!(brief.max_shots >= brief.required_beats.len());
}
