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
use crate::tests::support::{create_app_with_state, request, test_settings};

pub(crate) const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop"
);

/// [`ApiTransport`] over the in-process router: the same `oneshot` driver every route test uses.
pub(crate) struct RouterTransport {
    pub(crate) app: axum::Router,
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
pub(crate) enum VideoBehavior {
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
pub(crate) struct WorkerScript {
    behaviors: Vec<(String, VideoBehavior)>,
    /// Jobs the fake worker has claimed, in order: (type, job id, payload).
    claimed: Vec<(String, String, Value)>,
    failed_once: Vec<String>,
    /// Make `timeline_export` jobs fail, for the failed-export resume path (sc-22711).
    export_fails: bool,
    /// sc-22714 VQA answers, keyed by the `[questionId@frameId]` tag the reviewer stamps onto
    /// every question. Most specific first: `questionId@frameId`, then `questionId`, then
    /// `vqa_fallback`. An unmatched question answers "I cannot tell", so a test that forgets one
    /// records it as UNOBSERVED rather than as silent agreement.
    pub(crate) vqa_answers: std::collections::BTreeMap<String, String>,
    /// Fail every `image_vqa` job, for the "the backend answered nothing at all" path.
    pub(crate) vqa_fails: bool,
    /// Questions the fake worker has been asked, in order: (tag, question).
    pub(crate) vqa_asked: Vec<(String, String)>,
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
            // `image_vqa` (sc-22714) is what the reviewer's questions ride; `frame_extract` is
            // what turns a take into timestamped frame evidence. Both are job types the real
            // worker already advertises.
            "capabilities": ["video_generate", "timeline_export", "frame_extract", "image_vqa"],
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
                // sc-22714: the two understanding seams the reviewer drives. Neither renders
                // anything — `frame_extract` writes a placeholder still where FFmpeg would, and
                // `image_vqa` answers from the script's table in the shape SenseNova-U1 posts.
                "frame_extract" => run_fake_frame_job(&app, &job_id, &job).await,
                "image_vqa" => run_fake_vqa_job(&app, &script, &job_id, &job).await,
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

/// The `frame_extract` job, faked: write a placeholder still where FFmpeg would and report it as
/// an `assetWrites` fact, exactly as `run_frame_extract` does. Asset persistence, the sidecar, the
/// index and the two-phase result rewrite are all production code paths (sc-22714).
async fn run_fake_frame_job(app: &axum::Router, job_id: &str, job: &Value) {
    let payload = &job["payload"];
    let project_id = job["projectId"].as_str().expect("project id").to_owned();
    let timestamp = payload["sourceTimestamp"].as_f64().unwrap_or(0.0);
    let asset_id = format!("asset_frame_{}", &job_id.replace('-', "")[..12]);
    let media_rel = format!("assets/frames/{asset_id}.png");
    let project_dir = project_path(app, &project_id).await;
    std::fs::create_dir_all(project_dir.join("assets/frames")).expect("frames dir");
    std::fs::write(
        project_dir.join(&media_rel),
        film_harness::fixture_plate_png("review-frame", [90, 82, 70]).expect("plate encodes"),
    )
    .expect("fake frame");
    let fact = json!({
        "type": "frame",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": 576, "height": 320,
        "displayName": format!("Frame {timestamp:.2}s"),
        "createdAt": sceneworks_core::time::utc_now(),
        "mode": "frame_extract", "model": "timeline-frame-extract",
        "adapter": "ffmpeg-frame-extract",
        "prompt": format!("Extract frame at {timestamp:.2}s"),
        "negativePrompt": "", "loras": [],
        "normalizedSettings": {
            "timelineId": payload["timelineId"],
            "timelineItemId": payload["timelineItemId"],
            "playheadSeconds": payload["playheadSeconds"],
            "sourceTimestamp": timestamp,
        },
    });
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Frame saved.", "workerId": WORKER_ID,
            "result": { "assetWrites": [fact], "adapter": "ffmpeg-frame-extract" }
        }),
    )
    .await;
}

/// The `image_vqa` job, faked: answer from the script's table in the shape
/// `sensenova_jobs::vqa_result_json` posts. No weights ran, and `realModelInference: false` says
/// so — which is what keeps a scripted review from ever reading as a real one.
async fn run_fake_vqa_job(
    app: &axum::Router,
    script: &Arc<Mutex<WorkerScript>>,
    job_id: &str,
    job: &Value,
) {
    let question = job["payload"]["question"].as_str().unwrap_or("").to_owned();
    let tag = question
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']'))
        .map(|(tag, _)| tag.to_owned())
        .unwrap_or_default();
    let (answer, fails) = {
        let mut script = script.lock();
        script.vqa_asked.push((tag.clone(), question.clone()));
        let question_id = tag.split('@').next().unwrap_or("").to_owned();
        let answer = script
            .vqa_answers
            .get(&tag)
            .or_else(|| script.vqa_answers.get(&question_id))
            .or_else(|| script.vqa_answers.get("vqa_fallback"))
            .cloned()
            .unwrap_or_else(|| "I cannot tell from this frame.".to_owned());
        (answer, script.vqa_fails)
    };
    if fails {
        post_progress(
            app,
            job_id,
            json!({
                "status": "failed", "stage": "failed", "progress": 1,
                "message": "fake vqa fault", "error": "fake vqa fault: no weights",
                "workerId": WORKER_ID
            }),
        )
        .await;
        return;
    }
    post_progress(
        app,
        job_id,
        json!({
            "status": "completed", "stage": "completed", "progress": 1,
            "message": "Answer ready.", "workerId": WORKER_ID, "backend": "mlx",
            "result": {
                "answer": answer,
                "question": question,
                "sourceAssetId": job["payload"]["sourceAssetId"],
                "model": job["payload"]["model"],
                "realModelInference": false,
            }
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

pub(crate) struct Harness {
    pub(crate) app: axum::Router,
    pub(crate) transport: RouterTransport,
    temp_dir: tempfile::TempDir,
    pub(crate) script: Arc<Mutex<WorkerScript>>,
    worker: Option<tokio::task::JoinHandle<()>>,
}

impl Harness {
    pub(crate) async fn start(with_worker: bool, behaviors: Vec<(&str, VideoBehavior)>) -> Self {
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

    pub(crate) fn options(
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
            out_dir: self.out_dir(),
            poll_interval: Duration::from_millis(250),
            export: true,
            require_installed: false,
        }
    }

    pub(crate) fn out_dir(&self) -> PathBuf {
        self.temp_dir.path().join("run-out")
    }

    /// What `resume` / `replace-take` are driven with in these tests: the same run directory, a
    /// tight poll cadence, and a control only the test can trip (sc-22711).
    pub(crate) fn resume_options(&self) -> ResumeOptions {
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
    pub(crate) async fn resume_to_completion(&self) -> RunRecord {
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

    pub(crate) async fn jobs(&self) -> Vec<Value> {
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

pub(crate) fn fast(shots: &[&str]) -> Vec<(&'static str, VideoBehavior)> {
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
pub(crate) fn harness_record(harness: &Harness) -> RunRecord {
    film_harness::read_run_record(&harness.out_dir()).expect("run record on disk")
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
        .find(|item| item.shot_id == "SH020")
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
            .find(|item| item.shot_id == shot_id)
            .unwrap();
        let after_item = timeline
            .items
            .iter()
            .find(|item| item.shot_id == shot_id)
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
    assert_eq!(
        harness
            .script
            .lock()
            .claimed
            .iter()
            .filter(|(kind, _, _)| kind == "timeline_export")
            .count(),
        2,
        "exactly one re-export"
    );
    assert!(after.export_pending.is_none());
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
