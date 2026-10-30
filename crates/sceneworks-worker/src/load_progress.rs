//! One request-scoped, latest-wins model I/O progress path for every worker lane.
//!
//! Floor callbacks only replace a small in-memory slot. Network writes run on an async task,
//! are rate limited, and share an ordering lock with ordinary job progress, so an old loading
//! report can never land after a newer compute/terminal phase.
use crate::{ApiClient, JobSnapshot, JobStatus, ProgressRequest, ProgressStage, WorkerResult};
use gen_core::{LoadActivity, LoadActivityContext, LoadActivityObserver, LoadActivityPhase};
use std::cell::RefCell;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{watch, Mutex as AsyncMutex};

pub(crate) const PREPARING_MODEL: &str = "Preparing model.";

const POST_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Default)]
struct Latest {
    revision: u64,
    activity: Option<LoadActivity>,
    pending: bool,
    baseline: Option<Arc<ProgressRequest>>,
    compute: Option<(String, ProgressStage)>,
    prompt_encoding: bool,
    closed: bool,
}

pub(crate) struct Session {
    job_id: String,
    state: Mutex<Latest>,
    changed: watch::Sender<u64>,
    post_order: AsyncMutex<()>,
}

impl Session {
    fn observe(&self, activity: Option<LoadActivity>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return;
        }
        state.activity = activity;
        state.pending = true;
        state.revision = state.revision.wrapping_add(1);
        self.changed.send_replace(state.revision);
    }

    pub(crate) async fn post(
        &self,
        api: &ApiClient,
        mut payload: ProgressRequest,
    ) -> WorkerResult<JobSnapshot> {
        let phase_update = matches!(
            payload.stage,
            ProgressStage::LoadingModel
                | ProgressStage::Generating
                | ProgressStage::Training
                | ProgressStage::CachingLatents
                | ProgressStage::Checkpointing
        ) && !matches!(
            payload.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Canceled
        );
        // Clone potentially large result/provenance fields outside the floor callback's lock.
        let baseline = Arc::new(payload.clone());
        // Invalidate queued I/O reports BEFORE awaiting an already-running slow POST.
        let previous = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.pending = false;
            let previous = state.baseline.replace(baseline);
            if matches!(
                payload.status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Canceled
            ) {
                state.closed = true;
            }
            if !phase_update {
                state.activity = None;
                state.compute = None;
                state.prompt_encoding = false;
            }
            state.revision = state.revision.wrapping_add(1);
            previous
        };
        drop(previous);
        let _order = self.post_order.lock().await;
        // Re-read AFTER network ordering: callbacks may have advanced while an earlier POST was
        // in flight. Never send a phase snapshot prepared before that wait.
        if phase_update {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(activity) = &state.activity {
                payload.message = message(Some(activity));
                payload.status = JobStatus::LoadingModel;
                payload.stage = ProgressStage::LoadingModel;
            } else if let Some((message, stage)) = &state.compute {
                payload.message = message.clone();
                payload.stage = stage.clone();
                payload.status = if *stage == ProgressStage::LoadingModel {
                    JobStatus::LoadingModel
                } else {
                    JobStatus::Running
                };
            }
        }
        crate::progress::post_job_progress(api, &self.job_id, payload).await
    }

    async fn publish(self: Arc<Self>, api: ApiClient, mut changed: watch::Receiver<u64>) {
        let mut next_post = tokio::time::Instant::now();
        loop {
            if changed.changed().await.is_err() {
                return;
            }
            tokio::time::sleep_until(next_post).await;
            let _order = self.post_order.lock().await;
            let (baseline, activity, compute) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.closed {
                    return;
                }
                if !state.pending {
                    continue;
                }
                state.pending = false;
                (
                    state.baseline.clone(),
                    state.activity.clone(),
                    state.compute.clone(),
                )
            };
            // Formatting, large payload clones, serialization and HTTP run outside the IO lock.
            let mut payload = baseline.as_deref().cloned().unwrap_or_else(|| {
                let mut payload = crate::progress::progress_payload(
                    JobStatus::LoadingModel,
                    ProgressStage::LoadingModel,
                    0.0,
                    PREPARING_MODEL,
                    None,
                    None,
                    None,
                );
                // The API's first non-null backend sticks; admission must never guess CPU.
                payload.backend = None;
                payload
            });
            if let Some(activity) = &activity {
                payload.message = message(Some(activity));
                payload.status = JobStatus::LoadingModel;
                payload.stage = ProgressStage::LoadingModel;
            } else if let Some((message, stage)) = compute {
                payload.message = message;
                payload.stage = stage;
                payload.status = JobStatus::Running;
            }
            if let Err(error) =
                crate::progress::post_job_progress(&api, &self.job_id, payload).await
            {
                tracing::debug!(event = "model_load_progress_post_failed", job_id = %self.job_id,
                    %error, "the next ordinary job progress reports its outcome");
            }
            next_post = tokio::time::Instant::now() + POST_INTERVAL;
        }
    }
}

tokio::task_local! { static JOB: Arc<Session>; }
thread_local! { static THREAD_JOB: RefCell<Option<Arc<Session>>> = const { RefCell::new(None) }; }

fn current_session() -> Option<Arc<Session>> {
    JOB.try_with(Arc::clone)
        .ok()
        .or_else(|| THREAD_JOB.with(|slot| slot.borrow().clone()))
}

pub(crate) fn current(job_id: &str) -> Option<Arc<Session>> {
    current_session().filter(|session| session.job_id == job_id)
}

fn note_compute(message: String, stage: ProgressStage) {
    note_compute_phase(message, stage, false);
}

fn note_compute_phase(message: String, stage: ProgressStage, prompt_encoding: bool) {
    let Some(session) = current_session() else {
        return;
    };
    let mut state = session.state.lock().unwrap_or_else(|e| e.into_inner());
    if state.closed {
        return;
    }
    state.activity = None;
    state.compute = Some((message, stage));
    state.prompt_encoding = prompt_encoding;
    state.pending = true;
    state.revision = state.revision.wrapping_add(1);
    session.changed.send_replace(state.revision);
}

/// Called at the producer's universal callback, before a possibly backpressured event channel.
/// A slow ordinary progress POST therefore cannot leave an earlier I/O label current.
pub(crate) fn generator_progress(progress: &gen_core::Progress) {
    // This logical end only retires the underlying prompt label. Actual file I/O may still
    // override it, and a later compute phase must survive a delayed completion callback.
    if matches!(progress, gen_core::Progress::PromptEncodingComplete) {
        let Some(session) = current_session() else {
            return;
        };
        let mut state = session.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed || !state.prompt_encoding {
            return;
        }
        state.prompt_encoding = false;
        state.compute = Some((
            "Preparing generation.".to_owned(),
            ProgressStage::Generating,
        ));
        state.pending = true;
        state.revision = state.revision.wrapping_add(1);
        session.changed.send_replace(state.revision);
        return;
    }
    let (message, stage) = match progress {
        gen_core::Progress::PromptEncodingComplete => unreachable!("handled above"),
        gen_core::Progress::Loading(phase) => (
            generator_phase(*phase).to_owned(),
            if matches!(phase, gen_core::LoadPhase::EncodingPrompt) {
                ProgressStage::Generating
            } else {
                ProgressStage::LoadingModel
            },
        ),
        gen_core::Progress::Step { current, total } => (
            format!("Generating step {current}/{total}."),
            ProgressStage::Generating,
        ),
        gen_core::Progress::Decoding => ("Decoding.".to_owned(), ProgressStage::Generating),
        gen_core::Progress::Upsampling { tokens, max_tokens } => (
            format!("Enhancing prompt ({tokens}/{max_tokens} tokens)."),
            ProgressStage::Generating,
        ),
        gen_core::Progress::ConditioningComplete => {
            ("Generating.".to_owned(), ProgressStage::Generating)
        }
        gen_core::Progress::PromptUpsamplingComplete => (
            "Preparing generation.".to_owned(),
            ProgressStage::Generating,
        ),
    };
    note_compute_phase(
        message,
        stage,
        matches!(
            progress,
            gen_core::Progress::Loading(gen_core::LoadPhase::EncodingPrompt)
        ),
    );
}

#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn training_progress(progress: &gen_core::TrainingProgress) {
    let (message, stage) = match progress {
        gen_core::TrainingProgress::Preparing => {
            ("Preparing dataset.".to_owned(), ProgressStage::Preparing)
        }
        gen_core::TrainingProgress::LoadingModel => {
            (PREPARING_MODEL.to_owned(), ProgressStage::LoadingModel)
        }
        gen_core::TrainingProgress::Caching { current, total } => (
            format!("Caching dataset latents ({current}/{total})."),
            ProgressStage::CachingLatents,
        ),
        gen_core::TrainingProgress::Training { step, total, loss } => (
            format!("Training step {step} of {total} (loss {loss:.4})."),
            ProgressStage::Training,
        ),
        gen_core::TrainingProgress::Checkpoint { step } => (
            format!("Saved checkpoint at step {step}."),
            ProgressStage::Checkpointing,
        ),
        gen_core::TrainingProgress::Sample { .. } => return,
        gen_core::TrainingProgress::Saving => ("Saving adapter.".to_owned(), ProgressStage::Saving),
    };
    note_compute(message, stage);
}

/// Attach one request-local callback when a universal text request is built. The captured
/// context also covers decoder threads; it never consults another job's current observer.
// Backend-neutral request builders (including StarVector) also compile without a native backend.
pub(crate) fn text_observer() -> gen_core::core_llm::GenerationProgressObserver {
    let context = Context::capture();
    gen_core::core_llm::GenerationProgressObserver::new(move |phase| {
        let _guard = context.install();
        let message = match phase {
            Some(gen_core::core_llm::GenerationPhase::EncodingPrompt) => "Encoding prompt.",
            Some(gen_core::core_llm::GenerationPhase::Prefilling) => "Prefilling prompt.",
            Some(gen_core::core_llm::GenerationPhase::Generating) => "Generating text.",
            None => "Preparing text generation.",
        };
        note_compute(message.to_owned(), ProgressStage::Generating);
    })
}

/// Universal text execution starts with preparation; the request's phase observer supplies actual
/// tokenization/prefill/decode boundaries. Token counts use the same current-state slot.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
pub(crate) fn generate_text(
    model: &dyn gen_core::core_llm::TextLlm,
    request: &gen_core::core_llm::TextLlmRequest,
    on_event: &mut dyn FnMut(gen_core::core_llm::StreamEvent),
) -> gen_core::core_llm::Result<gen_core::core_llm::TextLlmOutput> {
    note_compute(
        "Preparing text generation.".to_owned(),
        ProgressStage::Generating,
    );
    model.generate(request, &mut |event| {
        if let gen_core::core_llm::StreamEvent::Token { index, .. } = &event {
            note_compute(
                format!("Generating text ({} tokens).", index + 1),
                ProgressStage::Generating,
            );
        }
        on_event(event);
    })
}

/// Captures both the inference floor callback and the worker's progress-ordering session.
#[derive(Clone)]
pub(crate) struct Context {
    activity: LoadActivityContext,
    session: Option<Arc<Session>>,
}
impl Context {
    pub(crate) fn capture() -> Self {
        Self {
            activity: LoadActivityContext::capture(),
            session: JOB
                .try_with(Arc::clone)
                .ok()
                .or_else(|| THREAD_JOB.with(|slot| slot.borrow().clone())),
        }
    }
    pub(crate) fn install(&self) -> impl Drop + '_ {
        struct Guard {
            previous: Option<Arc<Session>>,
            _observer: LoadActivityObserver,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                THREAD_JOB.with(|slot| *slot.borrow_mut() = self.previous.take());
            }
        }
        Guard {
            previous: THREAD_JOB.with(|slot| slot.replace(self.session.clone())),
            _observer: self.activity.install(),
        }
    }
    pub(crate) async fn scope<F: Future>(&self, future: F) -> F::Output {
        tokio::pin!(future);
        std::future::poll_fn(|cx| {
            // Install on each poll: Tokio may move the task to another runtime thread.
            let _guard = self.install();
            future.as_mut().poll(cx)
        })
        .await
    }
}

pub(crate) fn spawn_blocking<F, R>(run: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let context = Context::capture();
    tokio::task::spawn_blocking(move || {
        let _guard = context.install();
        run()
    })
}

pub(crate) fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let context = Context::capture();
    tokio::spawn(async move { context.scope(future).await })
}

/// Covers route preparation, cache loads, render drives, analysis and training with one observer.
pub(crate) async fn with_job<F: Future>(api: &ApiClient, job_id: &str, future: F) -> F::Output {
    let (changed, rx) = watch::channel(0);
    let session = Arc::new(Session {
        job_id: job_id.to_owned(),
        state: Mutex::new(Latest::default()),
        changed,
        post_order: AsyncMutex::new(()),
    });
    let observed = session.clone();
    let context = {
        let _observer = LoadActivityObserver::new(move |activity| observed.observe(activity));
        Context {
            activity: LoadActivityContext::capture(),
            session: Some(session.clone()),
        }
    };
    let poster = tokio::spawn(session.clone().publish(api.clone(), rx));
    struct Close(Arc<Session>, tokio::task::JoinHandle<()>);
    impl Drop for Close {
        fn drop(&mut self) {
            self.0
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .closed = true;
            self.1.abort();
        }
    }
    let _close = Close(session.clone(), poster);
    JOB.scope(session, context.scope(future)).await
}

pub(crate) fn message(activity: Option<&LoadActivity>) -> String {
    let Some(activity) = activity else {
        return "Preparing model.".to_owned();
    };
    if activity.phase == LoadActivityPhase::Checking {
        return match activity.total_files {
            Some(total) => format!("Checking model files ({}/{total}).", activity.files_checked),
            None => "Checking model files.".to_owned(),
        };
    }
    let verb = match activity.phase {
        LoadActivityPhase::Checking => unreachable!(),
        LoadActivityPhase::Verifying => "Verifying",
        LoadActivityPhase::Loading => "Loading",
    };
    let mut component = activity.component.replace(['_', '-'], " ");
    if activity.component_total_bytes.is_none() {
        if let Some(file) = activity.file.file_name().and_then(|name| name.to_str()) {
            if file != activity.component {
                component.push_str(&format!(" ({file})"));
            }
        }
    }
    let read = if activity.component_total_bytes.is_some() {
        activity.component_bytes_read
    } else {
        activity.bytes_read
    };
    let total = activity.component_total_bytes.or(activity.total_bytes);
    let bytes = total.map_or_else(
        || byte_count(read),
        |total| {
            let unit = byte_unit(total);
            format!(
                "{:.1} / {:.1} {}",
                read as f64 / unit.0,
                total as f64 / unit.0,
                unit.1
            )
        },
    );
    format!("{verb} {component} — {bytes}")
}
fn byte_unit(bytes: u64) -> (f64, &'static str) {
    if bytes >= 1_000_000_000 {
        (1_000_000_000.0, "GB")
    } else if bytes >= 1_000_000 {
        (1_000_000.0, "MB")
    } else if bytes >= 1_000 {
        (1_000.0, "KB")
    } else {
        (1.0, "bytes")
    }
}
fn byte_count(bytes: u64) -> String {
    let (divisor, unit) = byte_unit(bytes);
    format!("{:.1} {unit}", bytes as f64 / divisor)
}

pub(crate) fn generator_phase_stage(phase: gen_core::LoadPhase) -> (JobStatus, ProgressStage) {
    if matches!(phase, gen_core::LoadPhase::EncodingPrompt) {
        (JobStatus::Running, ProgressStage::Generating)
    } else {
        (JobStatus::LoadingModel, ProgressStage::LoadingModel)
    }
}

/// Universal Generator compute phases. Weight-read labels come exclusively from floor reports.
pub(crate) fn generator_phase(phase: gen_core::LoadPhase) -> &'static str {
    match phase {
        gen_core::LoadPhase::TextEncoder => "Preparing text encoder.",
        gen_core::LoadPhase::Renderer => "Preparing generation.",
        gen_core::LoadPhase::EncodingPrompt => "Encoding prompt.",
    }
}

/// Coalesced process bridge. Used by rank workers whose I/O reports cross stdout, rather than
/// sharing the coordinator's observer. The sink runs on its own thread; it never blocks readers.
#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
pub(crate) struct ProcessBridge {
    state: Arc<Mutex<(bool, Option<Option<LoadActivity>>)>>,
    wake: std::sync::mpsc::SyncSender<()>,
    post_order: Arc<Mutex<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
    _observer: LoadActivityObserver,
}
#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
impl ProcessBridge {
    pub(crate) fn new(sink: impl Fn(Option<LoadActivity>) + Send + 'static) -> Self {
        let state = Arc::new(Mutex::new((false, None)));
        let (wake, rx) = std::sync::mpsc::sync_channel(1);
        let pending = state.clone();
        let post_order = Arc::new(Mutex::new(()));
        let order = post_order.clone();
        let thread = std::thread::spawn(move || {
            while rx.recv().is_ok() {
                let order = order.lock().unwrap_or_else(|e| e.into_inner());
                let (closed, activity) = {
                    let mut state = pending.lock().unwrap_or_else(|e| e.into_inner());
                    (state.0, state.1.take())
                };
                if let Some(activity) = activity {
                    sink(activity);
                }
                if closed {
                    return;
                }
                drop(order);
                std::thread::sleep(POST_INTERVAL);
            }
        });
        let observed = state.clone();
        let notify = wake.clone();
        let observer = LoadActivityObserver::new(move |activity| {
            let mut state = observed.lock().unwrap_or_else(|e| e.into_inner());
            if !state.0 {
                state.1 = Some(activity);
                let _ = notify.try_send(());
            }
        });
        Self {
            state,
            wake,
            post_order,
            thread: Some(thread),
            _observer: observer,
        }
    }
    pub(crate) fn regular<T>(&self, write: impl FnOnce() -> T) -> T {
        let _order = self.post_order.lock().unwrap_or_else(|e| e.into_inner());
        self.state.lock().unwrap_or_else(|e| e.into_inner()).1 = None;
        write()
    }
}
#[cfg(any(target_os = "macos", feature = "backend-candle", test))]
impl Drop for ProcessBridge {
    fn drop(&mut self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).0 = true;
        let _ = self.wake.try_send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::{Path, State},
        routing::post,
        Json, Router,
    };
    use serde_json::Value;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::sync::Notify;

    fn report(component: &str, bytes: u64) -> LoadActivity {
        LoadActivity {
            phase: LoadActivityPhase::Loading,
            component: component.to_owned(),
            file: std::path::PathBuf::from(format!("{component}/model.safetensors")),
            bytes_read: bytes,
            total_bytes: Some(2_000_000_000),
            component_bytes_read: bytes,
            component_total_bytes: Some(2_000_000_000),
            files_checked: 0,
            total_files: None,
        }
    }
    #[test]
    fn one_formatter_distinguishes_hashing_checks_unknown_and_shard_totals() {
        let mut activity = report("text_encoder", 1_000_000_000);
        assert_eq!(
            message(Some(&activity)),
            "Loading text encoder — 1.0 / 2.0 GB"
        );
        activity.phase = LoadActivityPhase::Verifying;
        assert!(message(Some(&activity)).starts_with("Verifying text encoder —"));
        activity.phase = LoadActivityPhase::Checking;
        activity.files_checked = 3;
        activity.total_files = Some(7);
        assert_eq!(message(Some(&activity)), "Checking model files (3/7).");
        activity.phase = LoadActivityPhase::Loading;
        activity.component_total_bytes = None;
        activity.total_bytes = None;
        assert_eq!(
            message(Some(&activity)),
            "Loading text encoder (model.safetensors) — 1.0 GB"
        );
        assert_eq!(
            generator_phase(gen_core::LoadPhase::EncodingPrompt),
            "Encoding prompt."
        );
    }

    #[derive(Clone, Default)]
    struct Recorded {
        posts: Arc<Mutex<Vec<(String, Value, std::time::Instant)>>>,
        changed: Arc<Notify>,
        slow: Arc<AtomicBool>,
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }
    async fn receive(
        Path(job): Path<String>,
        State(state): State<Recorded>,
        Json(payload): Json<Value>,
    ) -> Json<Value> {
        if state.slow.swap(false, Ordering::SeqCst) {
            state.entered.notify_one();
            state.release.notified().await;
        }
        state
            .posts
            .lock()
            .unwrap()
            .push((job.clone(), payload, std::time::Instant::now()));
        state.changed.notify_one();
        Json(crate::tests::job_snapshot_json(&job, false))
    }
    async fn api() -> (ApiClient, Recorded, tokio::task::JoinHandle<()>) {
        let state = Recorded::default();
        let app = Router::new()
            .route("/api/v1/jobs/:job/progress", post(receive))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut settings = crate::tests::test_settings("http://unused".to_owned(), None);
        settings.api_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (ApiClient::new(&settings), state, server)
    }
    async fn wait_for(
        state: &Recorded,
        predicate: impl Fn(&[(String, Value, std::time::Instant)]) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let notified = state.changed.notified();
                if predicate(&state.posts.lock().unwrap()) {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("progress should arrive without running a model");
    }
    fn normal(message: &str) -> ProgressRequest {
        crate::progress::progress_payload(
            JobStatus::Running,
            ProgressStage::Generating,
            0.4,
            message,
            None,
            None,
            None,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_async_and_cache_boundaries_share_one_observer_without_inference() {
        let (api, state, server) = api().await;
        with_job(&api, "job", async {
            spawn(async {
                spawn_blocking(|| {
                    gen_core::LoadActivityContext::capture()
                        .report(Some(report("audio", 1_000_000_000)));
                })
                .await
                .unwrap();
            })
            .await
            .unwrap();
            wait_for(&state, |posts| {
                posts
                    .iter()
                    .any(|p| p.1["message"] == "Loading audio — 1.0 / 2.0 GB")
            })
            .await;
            // Exercise the actual dedicated-cache submission mechanism with a plain integer model.
            let (tx, rx) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                let mut cache = crate::cache_thread::CacheThread::new(false);
                let job: crate::cache_thread::CacheJob<u8, u8> = rx.recv().unwrap();
                job(&mut cache);
            });
            crate::cache_thread::run_cached_with_access_after_cold_admission(
                &tx,
                1u8,
                |_| Ok(()),
                || {
                    gen_core::LoadActivityContext::capture()
                        .report(Some(report("training", 1_000_000_000)));
                    Ok(7u8)
                },
                |model, _| {
                    assert_eq!(*model, 7);
                    Ok(())
                },
                crate::cache_thread::SeamMessages {
                    entry_missing: "missing",
                    panic_reset: "panic",
                    worker_stopped: "stopped",
                    worker_dropped: "dropped",
                },
            )
            .await
            .unwrap();
            thread.join().unwrap();
            wait_for(&state, |posts| {
                posts
                    .iter()
                    .any(|p| p.1["message"] == "Loading training — 1.0 / 2.0 GB")
            })
            .await;
            generator_progress(&gen_core::Progress::Loading(
                gen_core::LoadPhase::EncodingPrompt,
            ));
            crate::progress::update_job(&api, "job", normal("Encoding prompt."))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            state.posts.lock().unwrap().last().unwrap().1["message"],
            "Encoding prompt."
        );
        server.abort();
    }

    #[tokio::test]
    async fn prompt_end_retires_only_prompt_label_and_preserves_actual_io() {
        let settings = crate::tests::test_settings("http://unused".to_owned(), None);
        let api = ApiClient::new(&settings);
        with_job(&api, "job", async {
            let context = gen_core::LoadActivityContext::capture();
            generator_progress(&gen_core::Progress::Loading(
                gen_core::LoadPhase::EncodingPrompt,
            ));
            context.report(Some(report("transformer", 1_000_000_000)));
            generator_progress(&gen_core::Progress::PromptEncodingComplete);
            let session = current("job").unwrap();
            {
                let state = session.state.lock().unwrap();
                assert_eq!(state.activity.as_ref().unwrap().component, "transformer");
                assert_eq!(state.compute.as_ref().unwrap().0, "Preparing generation.");
                assert!(!state.prompt_encoding);
            }
            context.report(None);
            let state = session.state.lock().unwrap();
            assert!(state.activity.is_none());
            assert_eq!(state.compute.as_ref().unwrap().0, "Preparing generation.");
        })
        .await;
    }

    #[tokio::test]
    async fn late_prompt_end_preserves_newer_compute_and_other_prompt_contracts() {
        let settings = crate::tests::test_settings("http://unused".to_owned(), None);
        let api = ApiClient::new(&settings);
        with_job(&api, "job", async {
            for (progress, expected) in [
                (
                    gen_core::Progress::Step {
                        current: 4,
                        total: 8,
                    },
                    "Generating step 4/8.",
                ),
                (gen_core::Progress::Decoding, "Decoding."),
                (
                    gen_core::Progress::Loading(gen_core::LoadPhase::Renderer),
                    "Preparing generation.",
                ),
                (gen_core::Progress::ConditioningComplete, "Generating."),
            ] {
                generator_progress(&gen_core::Progress::Loading(
                    gen_core::LoadPhase::EncodingPrompt,
                ));
                generator_progress(&progress);
                let session = current("job").unwrap();
                let revision = session.state.lock().unwrap().revision;
                generator_progress(&gen_core::Progress::PromptEncodingComplete);
                let state = session.state.lock().unwrap();
                assert_eq!(state.compute.as_ref().unwrap().0, expected);
                assert_eq!(state.revision, revision, "late completion must be a no-op");
            }
            // LLM prompt encoding uses the same display text, but a separate typed contract.
            generator_progress(&gen_core::Progress::Loading(
                gen_core::LoadPhase::EncodingPrompt,
            ));
            let request = gen_core::core_llm::TextLlmRequest {
                progress: Some(text_observer()),
                ..Default::default()
            };
            let _guard = request.install_progress();
            gen_core::core_llm::report_generation_phase(
                gen_core::core_llm::GenerationPhase::EncodingPrompt,
            );
            generator_progress(&gen_core::Progress::PromptEncodingComplete);
            let session = current("job").unwrap();
            assert_eq!(
                session.state.lock().unwrap().compute.as_ref().unwrap().0,
                "Encoding prompt."
            );
        })
        .await;
    }

    #[tokio::test]
    async fn slow_posts_and_queued_prompt_updates_cannot_restore_completed_encoding() {
        let (api, state, server) = api().await;
        state.slow.store(true, Ordering::SeqCst);
        with_job(&api, "job", async {
            generator_progress(&gen_core::Progress::Loading(
                gen_core::LoadPhase::EncodingPrompt,
            ));
            tokio::time::timeout(Duration::from_secs(5), state.entered.notified())
                .await
                .unwrap();
            let context = gen_core::LoadActivityContext::capture();
            context.report(Some(report("transformer", 1_000_000_000)));
            generator_progress(&gen_core::Progress::PromptEncodingComplete);
            // Model an older ordinary channel event already waiting behind the slow POST.
            let queued = spawn({
                let api = api.clone();
                async move {
                    crate::progress::update_job(&api, "job", normal("Encoding prompt."))
                        .await
                        .unwrap();
                }
            });
            tokio::task::yield_now().await;
            state.release.notify_one();
            queued.await.unwrap();
            wait_for(&state, |posts| {
                posts
                    .last()
                    .is_some_and(|p| p.1["message"] == "Loading transformer — 1.0 / 2.0 GB")
            })
            .await;
            context.report(None);
            wait_for(&state, |posts| {
                posts
                    .last()
                    .is_some_and(|p| p.1["message"] == "Preparing generation.")
            })
            .await;
            crate::progress::update_job(&api, "job", normal("Encoding prompt."))
                .await
                .unwrap();
            assert_eq!(
                state.posts.lock().unwrap().last().unwrap().1["message"],
                "Preparing generation."
            );
            generator_progress(&gen_core::Progress::Step {
                current: 5,
                total: 8,
            });
            generator_progress(&gen_core::Progress::PromptEncodingComplete);
            crate::progress::update_job(&api, "job", normal("Encoding prompt."))
                .await
                .unwrap();
            assert_eq!(
                state.posts.lock().unwrap().last().unwrap().1["message"],
                "Generating step 5/8."
            );
        })
        .await;
        let posts = state.posts.lock().unwrap();
        let transformer = posts
            .iter()
            .position(|p| p.1["message"] == "Loading transformer — 1.0 / 2.0 GB")
            .unwrap();
        assert!(posts[transformer..]
            .iter()
            .all(|p| p.1["message"] != "Encoding prompt."));
        server.abort();
    }

    #[tokio::test]
    async fn universal_text_request_observer_reports_real_phase_changes_without_model_compute() {
        let (api, state, server) = api().await;
        with_job(&api, "job", async {
            let observer = text_observer();
            let (tx, rx) = std::sync::mpsc::channel();
            let task = spawn_blocking(move || {
                let request = gen_core::core_llm::TextLlmRequest {
                    progress: Some(observer),
                    ..Default::default()
                };
                let _guard = request.install_progress();
                while let Ok(phase) = rx.recv() {
                    gen_core::core_llm::report_generation_phase(phase);
                }
            });
            for (phase, expected) in [
                (
                    gen_core::core_llm::GenerationPhase::EncodingPrompt,
                    "Encoding prompt.",
                ),
                (
                    gen_core::core_llm::GenerationPhase::Prefilling,
                    "Prefilling prompt.",
                ),
                (
                    gen_core::core_llm::GenerationPhase::Generating,
                    "Generating text.",
                ),
            ] {
                tx.send(phase).unwrap();
                wait_for(&state, |posts| {
                    posts.last().is_some_and(|p| p.1["message"] == expected)
                })
                .await;
            }
            drop(tx);
            task.await.unwrap();
            wait_for(&state, |posts| {
                posts
                    .last()
                    .is_some_and(|p| p.1["message"] == "Preparing text generation.")
            })
            .await;
        })
        .await;
        server.abort();
    }

    #[tokio::test]
    async fn slow_posts_never_block_io_and_obsolete_reports_cannot_follow_compute() {
        let (api, state, server) = api().await;
        state.slow.store(true, Ordering::SeqCst);
        with_job(&api, "job", async {
            let context = gen_core::LoadActivityContext::capture();
            context.report(Some(report("transformer", 1)));
            state.entered.notified().await;
            let start = std::time::Instant::now();
            for bytes in 2..1_000 {
                context.report(Some(report("transformer", bytes)));
            }
            assert!(
                start.elapsed() < Duration::from_secs(1),
                "HTTP backpressure reached I/O callback"
            );
            let compute = spawn({
                let api = api.clone();
                async move {
                    generator_progress(&gen_core::Progress::Loading(
                        gen_core::LoadPhase::EncodingPrompt,
                    ));
                    crate::progress::update_job(&api, "job", normal("Encoding prompt."))
                        .await
                        .unwrap();
                }
            });
            tokio::task::yield_now().await;
            state.release.notify_one();
            compute.await.unwrap();
            context.report(None);
            wait_for(&state, |posts| posts.len() >= 3).await;
        })
        .await;
        let posts = state.posts.lock().unwrap();
        let compute = posts
            .iter()
            .position(|p| p.1["message"] == "Encoding prompt.")
            .unwrap();
        assert!(posts[compute..]
            .iter()
            .all(|p| p.1["message"] == "Encoding prompt."));
        assert!(
            posts.len() <= 3,
            "obsolete byte updates must coalesce: {} posts",
            posts.len()
        );
        server.abort();
    }

    async fn isolation() {
        let (api, state, server) = api().await;
        let a = with_job(&api, "job-a", async {
            for _ in 0..10 {
                spawn(async {
                    tokio::task::yield_now().await;
                    spawn_blocking(|| {
                        gen_core::LoadActivityContext::capture().report(Some(report("alpha", 9)))
                    })
                    .await
                    .unwrap();
                })
                .await
                .unwrap();
            }
            wait_for(&state, |posts| posts.iter().any(|p| p.0 == "job-a")).await;
        });
        let b = with_job(&api, "job-b", async {
            for _ in 0..10 {
                spawn(async {
                    tokio::task::yield_now().await;
                    spawn_blocking(|| {
                        gen_core::LoadActivityContext::capture().report(Some(report("beta", 9)))
                    })
                    .await
                    .unwrap();
                })
                .await
                .unwrap();
            }
            wait_for(&state, |posts| posts.iter().any(|p| p.0 == "job-b")).await;
        });
        tokio::join!(a, b);
        for (job, payload, _) in state.posts.lock().unwrap().iter() {
            let component = if job == "job-a" { "alpha" } else { "beta" };
            assert!(
                payload["message"].as_str().unwrap().contains(component),
                "{job}: {payload}"
            );
        }
        assert!(!gen_core::LoadActivityContext::capture().is_observed());
        server.abort();
    }
    #[tokio::test]
    async fn interleaved_jobs_on_one_runtime_thread_keep_their_own_context() {
        isolation().await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn migrated_jobs_keep_their_own_context() {
        isolation().await;
    }

    #[test]
    fn rank_bridge_drops_obsolete_loads_and_finishes_before_terminal_output() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let messages = received.clone();
        let bridge = ProcessBridge::new(move |activity| {
            messages.lock().unwrap().push(message(activity.as_ref()));
        });
        let context = gen_core::LoadActivityContext::capture();
        for bytes in 0..1000 {
            context.report(Some(report("transformer", bytes)));
        }
        bridge.regular(|| received.lock().unwrap().push("Training step 1".to_owned()));
        drop(bridge);
        received.lock().unwrap().push("Done".to_owned());
        let received = received.lock().unwrap();
        let compute = received
            .iter()
            .position(|v| v == "Training step 1")
            .unwrap();
        assert_eq!(&received[compute..], &["Training step 1", "Done"]);
    }
}
