//! Inert filesystem receipts and API control only: no trainer/model/device calls.
use super::*;
use axum::{
    extract::{Path as ApiPath, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Success,
    Cancel,
    HeartbeatFailure,
    Abandon,
}

struct Gate {
    entered: AtomicBool,
    beats: AtomicUsize,
    posts: Mutex<Vec<(String, Value)>>,
    changed: tokio::sync::Notify,
    released: Mutex<bool>,
    release: Condvar,
    cancel: Mutex<Option<gen_core::CancelFlag>>,
    root: Mutex<Option<PathBuf>>,
    finished: AtomicBool,
    mode: Mode,
}
impl Gate {
    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            entered: AtomicBool::new(false),
            beats: AtomicUsize::new(0),
            posts: Mutex::new(Vec::new()),
            changed: tokio::sync::Notify::new(),
            released: Mutex::new(false),
            release: Condvar::new(),
            cancel: Mutex::new(None),
            root: Mutex::new(None),
            finished: AtomicBool::new(false),
            mode,
        })
    }
    fn unblock(&self) {
        *self.released.lock().unwrap() = true;
        self.release.notify_all();
    }
    fn canceled(&self) -> bool {
        self.cancel
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(gen_core::CancelFlag::is_cancelled)
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        self.unblock();
    }
}

async fn stub(gate: Arc<Gate>) -> (String, tokio::task::JoinHandle<()>) {
    async fn heartbeat(
        State(gate): State<Arc<Gate>>,
        ApiPath(worker): ApiPath<String>,
    ) -> axum::response::Response {
        loop {
            let changed = gate.changed.notified();
            if gate.entered.load(Ordering::SeqCst) || gate.finished.load(Ordering::SeqCst) {
                break;
            }
            changed.await;
        }
        gate.beats.fetch_add(1, Ordering::SeqCst);
        gate.changed.notify_waiters();
        if gate.mode == Mode::HeartbeatFailure {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        Json(crate::tests::worker_snapshot_json(&worker)).into_response()
    }
    async fn job(State(gate): State<Arc<Gate>>, ApiPath(job): ApiPath<String>) -> Json<Value> {
        Json(crate::tests::job_snapshot_json(
            &job,
            gate.mode == Mode::Cancel,
        ))
    }
    async fn progress(
        State(gate): State<Arc<Gate>>,
        ApiPath(job): ApiPath<String>,
        Json(payload): Json<Value>,
    ) -> Json<Value> {
        gate.posts.lock().unwrap().push((job.clone(), payload));
        gate.changed.notify_waiters();
        Json(crate::tests::job_snapshot_json(
            &job,
            gate.mode == Mode::Cancel,
        ))
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/api/v1/workers/:worker/heartbeat", post(heartbeat))
        .route("/api/v1/jobs/:job", get(job))
        .route("/api/v1/jobs/:job/progress", post(progress))
        .with_state(gate);
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, server)
}

fn fixture() -> (tempfile::TempDir, Settings, TrainingPlan, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().canonicalize().unwrap();
    let settings = super::tests::test_settings(&data);
    let dataset = data.join("datasets/ds-1");
    std::fs::create_dir_all(&dataset).unwrap();
    let bytes = b"literal prepared input bytes; no tensor or learned work".repeat(2500);
    let source = dataset.join("prepared.safetensors");
    std::fs::write(&source, &bytes).unwrap();
    let image = dataset.join("item.png");
    std::fs::write(&image, b"inert image placeholder").unwrap();
    let mut plan = super::tests::plan_json(
        &data,
        "z_image_lora",
        "z_image_turbo",
        "lora",
        &[image.to_str().unwrap(), image.to_str().unwrap()],
    );
    for item in plan["dataset"]["items"].as_array_mut().unwrap() {
        item["ltxPreparedBundlePath"] = json!("prepared.safetensors");
        item["ltxPreparedBundleSize"] = json!(bytes.len());
        item["ltxPreparedBundleSha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    }
    (dir, settings, serde_json::from_value(plan).unwrap(), bytes)
}

fn inert_preflight(
    settings: &Settings,
    plan: &TrainingPlan,
    inputs: PreparedTrainingInputs,
    expected: &[u8],
) -> WorkerResult<TrainingPreflight> {
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    {
        // This is the production descriptor-only validator, never load_trainer/validate/train.
        let prepared = preflight_training_run_with_inputs(settings, plan, false, inputs)?;
        let paths: Vec<_> = prepared
            .request
            .items
            .iter()
            .map(|item| {
                PathBuf::from(
                    item.model_options["ltxPreparedBundlePath"]
                        .as_str()
                        .unwrap(),
                )
            })
            .collect();
        assert_eq!(paths[0], paths[1]);
        assert_eq!(prepared.prepared_inputs.materialization_count, 1);
        assert_eq!(std::fs::read(&paths[0]).unwrap(), expected);
        Ok(prepared)
    }
    #[cfg(not(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )))]
    {
        // No trainer registry exists in this build. Exercise the same pure snapshot mapper;
        // the real no-backend preflight still refuses the absent registry before any copy.
        let mut inputs = inputs;
        let operation = (|| {
            let first = inputs.resolve_model_options(
                settings,
                &plan.dataset.root_path,
                &plan.dataset.items[0].extra,
            )?;
            let second = inputs.resolve_model_options(
                settings,
                &plan.dataset.root_path,
                &plan.dataset.items[1].extra,
            )?;
            assert_eq!(first, second);
            assert_eq!(inputs.materialization_count, 1);
            assert_eq!(
                std::fs::read(first["ltxPreparedBundlePath"].as_str().unwrap()).unwrap(),
                expected
            );
            Ok(())
        })();
        finish_prepared_input_operation(operation, inputs)
    }
}

fn preparation(
    settings: Settings,
    plan: TrainingPlan,
    bytes: Vec<u8>,
    gate: Arc<Gate>,
    stage: PreparedBundleCheckpoint,
    runtime_thread: std::thread::ThreadId,
) -> impl FnOnce(gen_core::CancelFlag) -> WorkerResult<TrainingPreflight> + Send {
    move |cancel| {
        *gate.cancel.lock().unwrap() = Some(cancel.clone());
        let hook_gate = gate.clone();
        let data_dir = settings.data_dir.clone();
        let mut visited = false;
        let inputs = PreparedTrainingInputs {
            cancel: Some(cancel),
            copy_checkpoint: Some(Box::new(move |phase| {
                if phase != stage || visited {
                    return Ok(());
                }
                visited = true;
                if std::thread::current().id() == runtime_thread {
                    return Err(WorkerError::Engine(
                        "prepared copy/hash ran on the async runtime thread".to_owned(),
                    ));
                }
                let context = gen_core::LoadActivityContext::capture();
                assert!(
                    context.is_observed(),
                    "load-progress context must reach actual copy/hash"
                );
                context.report(Some(gen_core::LoadActivity {
                    phase: gen_core::LoadActivityPhase::Verifying,
                    component: "prepared-input-context".to_owned(),
                    file: PathBuf::from("literal.safetensors"),
                    bytes_read: 1,
                    total_bytes: Some(2),
                    component_bytes_read: 1,
                    component_total_bytes: Some(2),
                    files_checked: 0,
                    total_files: None,
                }));
                let root = std::fs::read_dir(data_dir.join("cache/training-inputs"))?
                    .filter_map(Result::ok)
                    .find(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with(PREPARED_BUNDLE_DIRECTORY_PREFIX)
                    })
                    .unwrap()
                    .path();
                *hook_gate.root.lock().unwrap() = Some(root);
                hook_gate.entered.store(true, Ordering::SeqCst);
                hook_gate.changed.notify_waiters();
                let released = hook_gate.released.lock().unwrap();
                let (released, timeout) = hook_gate
                    .release
                    .wait_timeout_while(released, Duration::from_secs(5), |released| !*released)
                    .unwrap();
                assert!(
                    *released && !timeout.timed_out(),
                    "async control must advance while copy/hash waits"
                );
                Ok(())
            })),
            ..Default::default()
        };
        let result = inert_preflight(&settings, &plan, inputs, &bytes);
        gate.finished.store(true, Ordering::SeqCst);
        gate.changed.notify_waiters();
        result
    }
}

async fn wait_ready(gate: &Gate, require_cancel: bool) {
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let changed = gate.changed.notified();
            let context_post = gate.posts.lock().unwrap().iter().any(|(job, payload)| job == "prepared-job" && payload["message"].as_str().is_some_and(|message| message.contains("prepared input context")));
            if gate.entered.load(Ordering::SeqCst) && gate.beats.load(Ordering::SeqCst) > 0 && context_post && (!require_cancel || gate.canceled()) { break; }
            tokio::select! { _ = changed => {}, _ = tokio::time::sleep(Duration::from_millis(5)) => {} }
        }
    }).await.expect("heartbeat, cancel polling and exact job context must advance before release");
}

async fn baseline(api: &ApiClient, settings: &Settings) {
    update_job(
        api,
        "prepared-job",
        training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing LoRA training.",
            None,
            backend_label(&settings.gpu_id),
        ),
    )
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn copy_and_hash_leave_runtime_live_and_preserve_exact_snapshot_and_context() {
    for stage in [
        PreparedBundleCheckpoint::Copy,
        PreparedBundleCheckpoint::Verify,
    ] {
        let (_dir, mut settings, plan, bytes) = fixture();
        let gate = Gate::new(Mode::Success);
        let (url, server) = stub(gate.clone()).await;
        settings.api_url = url;
        let api = ApiClient::new(&settings);
        let prepare = preparation(
            settings.clone(),
            plan,
            bytes,
            gate.clone(),
            stage,
            std::thread::current().id(),
        );
        let run = crate::load_progress::with_job(&api, "prepared-job", async {
            baseline(&api, &settings).await;
            preflight_training_run_async_using(&api, &settings, "prepared-job", prepare).await
        });
        let control = async {
            wait_ready(&gate, false).await;
            assert!(gate.root.lock().unwrap().as_ref().unwrap().is_dir());
            gate.unblock();
            Ok::<(), WorkerError>(())
        };
        let (prepared, ()) =
            tokio::try_join!(run, control).expect("weights-free preflight snapshots succeed");
        let root = gate.root.lock().unwrap().clone().unwrap();
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        {
            assert!(
                root.exists(),
                "request retains ownership until explicit finish"
            );
            gate.cancel.lock().unwrap().as_ref().unwrap().cancel();
            assert!(
                prepared.request.cancel.is_cancelled(),
                "preflight and training retain the same flag"
            );
        }
        finish_training_preflight(prepared, Ok(())).unwrap();
        assert!(!root.exists());
        server.abort();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_and_heartbeat_failure_wait_for_copy_then_cleanup_without_losing_primary_error(
) {
    for mode in [Mode::Cancel, Mode::HeartbeatFailure] {
        for stage in [
            PreparedBundleCheckpoint::Copy,
            PreparedBundleCheckpoint::Verify,
        ] {
            let (_dir, mut settings, plan, bytes) = fixture();
            let gate = Gate::new(mode);
            let (url, server) = stub(gate.clone()).await;
            settings.api_url = url;
            let api = ApiClient::new(&settings);
            let prepare = preparation(
                settings.clone(),
                plan,
                bytes,
                gate.clone(),
                stage,
                std::thread::current().id(),
            );
            let run = crate::load_progress::with_job(&api, "prepared-job", async {
                baseline(&api, &settings).await;
                preflight_training_run_async_using(&api, &settings, "prepared-job", prepare).await
            });
            let control = async {
                wait_ready(&gate, true).await;
                assert!(
                    !gate.finished.load(Ordering::SeqCst),
                    "cancel must not abandon live snapshot ownership"
                );
                assert!(gate.root.lock().unwrap().as_ref().unwrap().exists());
                gate.unblock();
            };
            let (result, ()) = tokio::join!(run, control);
            let error = result.expect_err("control failure must fail preflight");
            if mode == Mode::Cancel {
                assert!(matches!(error, WorkerError::Canceled(_)));
            } else {
                assert!(!matches!(error, WorkerError::Canceled(_)));
                assert!(error.to_string().contains("503"), "{error}");
            }
            assert!(gate.finished.load(Ordering::SeqCst));
            assert!(!gate.root.lock().unwrap().as_ref().unwrap().exists());
            server.abort();
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn abandoned_async_waiter_trips_copy_flag_and_retains_cleanup_owner_until_winddown() {
    let (_dir, mut settings, plan, bytes) = fixture();
    let gate = Gate::new(Mode::Abandon);
    let (url, server) = stub(gate.clone()).await;
    settings.api_url = url;
    let api = ApiClient::new(&settings);
    let prepare = preparation(
        settings.clone(),
        plan,
        bytes,
        gate.clone(),
        PreparedBundleCheckpoint::Verify,
        std::thread::current().id(),
    );
    let run = crate::load_progress::spawn(async move {
        crate::load_progress::with_job(&api, "prepared-job", async {
            baseline(&api, &settings).await;
            preflight_training_run_async_using(&api, &settings, "prepared-job", prepare).await
        })
        .await
    });
    wait_ready(&gate, false).await;
    run.abort();
    assert!(run.await.unwrap_err().is_cancelled());
    assert!(gate.canceled());
    assert!(gate.root.lock().unwrap().as_ref().unwrap().exists());
    gate.unblock();
    tokio::time::timeout(Duration::from_secs(4), async {
        while !gate.finished.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!gate.root.lock().unwrap().as_ref().unwrap().exists());
    server.abort();
}

#[test]
fn unclaimed_completed_preflight_handoff_explicitly_releases_snapshots() {
    let (_dir, settings, plan, bytes) = fixture();
    let prepared = inert_preflight(&settings, &plan, PreparedTrainingInputs::default(), &bytes)
        .expect("descriptor-only preparation");
    let parent = settings.data_dir.join("cache/training-inputs");
    let roots = || {
        std::fs::read_dir(&parent)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(PREPARED_BUNDLE_DIRECTORY_PREFIX)
            })
            .count()
    };
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    assert_eq!(roots(), 1);
    let handoff = Arc::new(TrainingPreflightHandoff(Mutex::new(Some(prepared))));
    let worker = handoff.clone();
    drop(handoff);
    drop(worker);
    assert_eq!(
        roots(),
        0,
        "last handoff owner performs explicit ordered cleanup"
    );
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test(flavor = "current_thread")]
async fn production_dry_run_preserves_summary_and_closes_prepared_inputs_without_a_trainer() {
    let (_dir, mut settings, plan, _) = fixture();
    let gate = Gate::new(Mode::Success);
    gate.entered.store(true, Ordering::SeqCst);
    let (url, server) = stub(gate.clone()).await;
    settings.api_url = url;
    let api = ApiClient::new(&settings);
    let job: JobSnapshot =
        serde_json::from_value(crate::tests::job_snapshot_json("prepared-job", false)).unwrap();
    run_training_dry_run(&api, &settings, &job, &plan)
        .await
        .unwrap();
    let posts = gate.posts.lock().unwrap();
    let result = &posts.last().unwrap().1["result"];
    assert_eq!(result["mode"], "dry_run");
    assert_eq!(result["datasetItemCount"], 2);
    assert_eq!(result["fileName"], "lora.safetensors");
    assert_eq!(result["baseModelInstalled"], false);
    assert_eq!(
        std::fs::read_dir(settings.data_dir.join("cache/training-inputs"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with(PREPARED_BUNDLE_DIRECTORY_PREFIX))
            .count(),
        0
    );
    assert!(
        !settings.data_dir.join("loras").exists(),
        "dry run must not create training output"
    );
    server.abort();
}
