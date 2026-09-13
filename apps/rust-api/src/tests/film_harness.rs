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
                "timeline_export" => run_fake_export_job(&app, &job_id, &job).await,
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
            compiled_path: None,
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

    /// The checked-in pack (and its plates) copied into the temp dir with `edit` applied, so a test
    /// can change an entry without touching the shipped documents.
    fn edited_pack(&self, edit: impl FnOnce(&mut Value)) -> PathBuf {
        let text = std::fs::read_to_string(self.fixture_pack()).expect("fixture pack");
        let mut pack: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&text))
                .expect("fixture pack parses");
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
                delay_secs: 9,
                peak_pct: 40.0,
            },
        )],
    )
    .await;
    // The run budget is wall-clock from the START of the run, so it has to leave room for the
    // pre-dispatch work (catalog, host, project, seven reference imports) on a loaded CI runner —
    // otherwise the budget expires before the first attempt and the test measures its own setup.
    let plan = harness.edited_plan(|plan| {
        plan["limits"] = json!({
            "maxRunSeconds": 6, "maxShotSeconds": 6, "maxAttemptsPerShot": 3, "maxMemoryGb": 96
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
    let options = harness.options(harness.fixture_plan(), pack, Some(&["SH010"]));
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
