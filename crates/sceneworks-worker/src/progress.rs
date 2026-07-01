//! Job status/progress plumbing: building [`ProgressRequest`]s and posting terminal/cancel states.
use super::*;

/// RAII guard binding a running blocking/GPU task to the `select!`-loop that streams its progress
/// (sc-8804, F-003). Every streaming consumer spawns a blocking task that owns a long GPU denoise
/// or training run, then sits in a `select! { channel, interval }` loop posting progress via
/// `update_job(...).await?` / `heartbeat(...).await?`. On a transient POST failure or a 409 (the
/// stale-sweep reclaimed the job) that `?` returns the consumer early — but a bare
/// `JoinHandle` does NOT stop its task when dropped, so the GPU/training thread keeps burning
/// unified memory while the worker returns and claims the next job. That is two concurrent GPU
/// workloads on one Metal device (the sc-8390 SIGKILL/OOM class).
///
/// This guard closes that gap: it holds the engine [`CancelFlag`] and the task [`JoinHandle`], and
/// on `Drop` (i.e. any early return, including the `?` error paths) it trips the flag so a
/// cooperative engine bails, then `abort()`s the task so a non-cooperative one is torn down. The
/// happy path calls [`Self::into_handle`] to reclaim the raw handle and `.await` it normally — the
/// guard is then inert (its `Drop` is a no-op), so a clean completion never aborts anything.
///
/// A cooperative cancel flag the guard can trip on drop. Implemented for BOTH cancel-flag types the
/// worker threads through its blocking tasks — `gen_core::CancelFlag` (image/video/training/analysis)
/// and `gen_core::core_llm::CancelFlag` (prompt-refine LLM), which are distinct types with the same
/// `cancel()` surface — so one guard covers every streaming consumer regardless of which flag it holds.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) trait CancelHandle {
    fn cancel(&self);
}

impl CancelHandle for gen_core::CancelFlag {
    fn cancel(&self) {
        gen_core::CancelFlag::cancel(self)
    }
}

// `gen_core::core_llm::CancelFlag` is a DIFFERENT type from `gen_core::CancelFlag` (the LLM stack's
// own flag, re-exported through core_llm); it exposes the same `cancel()`. Implemented separately so
// the prompt-refine consumer can use the same guard. On the plain-Linux parity build core_llm may
// not link, but the guard is unused there anyway; guard the impl behind the same gate as the callers.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl CancelHandle for gen_core::core_llm::CancelFlag {
    fn cancel(&self) {
        gen_core::core_llm::CancelFlag::cancel(self)
    }
}

/// Gated to the job handlers that use it (`any(target_os = "macos", feature = "backend-candle")`);
/// on the plain-Linux parity build it is unused, so allow dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) struct CancelJoinGuard<C: CancelHandle, R> {
    cancel: Option<C>,
    handle: Option<tokio::task::JoinHandle<R>>,
}

#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
impl<C: CancelHandle, R> CancelJoinGuard<C, R> {
    /// Bind a spawned blocking task to its engine cancel flag. While this guard is alive, any early
    /// return cancels + aborts the task before the job function unwinds. Pass `None` for the flag
    /// only when the task exposes no cooperative cancel — the abort-on-drop still applies.
    pub(crate) fn new(cancel: impl Into<Option<C>>, handle: tokio::task::JoinHandle<R>) -> Self {
        Self {
            cancel: cancel.into(),
            handle: Some(handle),
        }
    }

    /// Reclaim the raw [`JoinHandle`] on the success path so the caller can `.await` it. This
    /// disarms the guard — its `Drop` becomes a no-op — so a clean completion never aborts the task.
    pub(crate) fn into_handle(mut self) -> tokio::task::JoinHandle<R> {
        // Disarm the guard: clear the cancel flag too so the ensuing `Drop` is a total no-op — a
        // clean completion must neither cancel nor abort the reclaimed task.
        self.cancel = None;
        self.handle.take().expect("guard handle taken exactly once")
    }

    /// Mutable access to the wrapped [`JoinHandle`] so a consumer that owns its own `select!` loop
    /// can poll the task in-place (`&mut *guard.handle_mut()`) while the guard keeps the
    /// cancel-and-abort-on-drop protection armed for every early `?` return in that loop.
    pub(crate) fn handle_mut(&mut self) -> &mut tokio::task::JoinHandle<R> {
        self.handle.as_mut().expect("guard handle present")
    }

    /// Disarm the guard after the wrapped task has already resolved through `handle_mut()`: drop
    /// the (finished) handle and clear the cancel flag so the ensuing `Drop` neither cancels nor
    /// aborts. Used by the two heartbeat helpers, which poll the task in-place and then need to
    /// stand the guard down once its arm has completed.
    pub(crate) fn disarm(&mut self) {
        self.cancel = None;
        // The task has resolved; drop its handle without touching the flag. (A resolved handle's
        // drop is a no-op teardown, and dropping the raw handle here avoids a future-drop lint.)
        drop(self.handle.take());
    }
}

impl<C: CancelHandle, R> Drop for CancelJoinGuard<C, R> {
    fn drop(&mut self) {
        // Reached only on an early/error return that never called `into_handle` — trip the engine
        // cancel flag (cooperative bail) and abort the task (non-cooperative teardown) so the GPU
        // work stops instead of leaking alongside the next claimed job (sc-8804).
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

pub(crate) async fn fail_job(
    api: &ApiClient,
    job_id: &str,
    message: &str,
    error: Option<String>,
) -> WorkerResult<()> {
    update_job(
        api,
        job_id,
        progress_payload(
            JobStatus::Failed,
            ProgressStage::Failed,
            1.0,
            message,
            error,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn check_cancel(api: &ApiClient, job_id: &str, message: &str) -> WorkerResult<()> {
    let job: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{job_id}")).await?;
    if job.cancel_requested {
        mark_job_canceled(api, job_id, message).await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    Ok(())
}

pub(crate) async fn mark_job_canceled(
    api: &ApiClient,
    job_id: &str,
    message: &str,
) -> WorkerResult<()> {
    update_job(
        api,
        job_id,
        progress_payload(
            JobStatus::Canceled,
            ProgressStage::Canceled,
            1.0,
            message,
            None,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

/// Run a long, self-contained blocking `task` while keeping the worker's heartbeat alive (and,
/// when given a `cancel` flag, honoring a user cancel). This is the SHARED keepalive every
/// long-inline compute path should use: the Rust worker sends no periodic heartbeat on its own
/// during a job, and posting job *progress* does NOT refresh the worker's `last_seen` (only a
/// terminal status does), so without a `Busy` ping every `progress_report_interval` (5–15s) a job
/// that runs silently past the API's worker-timeout (default 90s) gets swept to `interrupted`
/// mid-flight — then the worker's next post is 409'd and the job looks "stuck" (sc-8200, sc-8390).
///
/// Before this existed the same `select!` was open-coded per handler, so new paths (LoRA training,
/// VQA, pose/kps/person-detect) were missed — exactly the gap that hung a Krea2 LoRA run at a slow
/// step-500 checkpoint save. Streaming consumers that already own an event loop
/// (`training_jobs::consume_training_events`, caption/model/prompt/media/video) inline the same
/// interval arm instead; this helper is for the single-blocking-task shape.
///
/// `task` must own all its work — it cannot report progress between ticks. The helper pings
/// `WorkerStatus::Busy` every interval without posting any intermediate job status. When `cancel`
/// is `Some`, it also polls the API for a user cancel and trips the flag; un-interruptible compute
/// finishes its current op, then we post the terminal `Canceled` (`cancel_message`) and return
/// `WorkerError::Canceled`. Pass `None` for paths with no cancelable work (heartbeat only).
///
/// Every consumer is a job handler gated behind `any(target_os = "macos", feature =
/// "backend-candle")`, so on the plain-Linux parity build (neither) this is unused — allow
/// dead_code there only, keeping real dead-code detection on the configs that do call it.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) async fn run_blocking_with_heartbeat<R>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    cancel: Option<gen_core::CancelFlag>,
    cancel_message: &str,
    task_label: &'static str,
    task: tokio::task::JoinHandle<WorkerResult<R>>,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    // Bind the blocking task to its cancel flag: a `heartbeat(...).await?` failure below returns
    // early, and this guard trips-and-aborts the still-running task on drop instead of leaking it
    // alongside the next claimed job (sc-8804, F-003).
    let mut guard: CancelJoinGuard<gen_core::CancelFlag, WorkerResult<R>> =
        CancelJoinGuard::new(cancel.clone(), task);
    let mut canceled = false;
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut *guard.handle_mut() => {
                let value = result.map_err(|error| task_join_error(task_label, error))??;
                // Success: disarm the guard so a clean completion never aborts.
                guard.disarm();
                if canceled {
                    mark_job_canceled(api, job_id, cancel_message).await?;
                    return Err(WorkerError::Canceled(cancel_message.to_owned()));
                }
                return Ok(value);
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await?;
                if let Some(flag) = &cancel {
                    if !canceled && cancel_requested_peek(api, job_id).await {
                        flag.cancel();
                        canceled = true;
                    }
                }
            }
        }
    }
}

/// Check-only cancel poll (sc-5515): returns `true` when the user requested
/// cancellation, WITHOUT posting any status. Unlike [`check_cancel`] this never
/// writes the terminal `Canceled`. In-loop generation/training pollers that sit in
/// front of a long, un-interruptible compute use this so the job stays non-terminal
/// ("Cancelling…") until the in-flight work actually stops; they post the terminal
/// `Canceled` themselves only once it does (sc-5515 image, sc-5516 video/training/detail).
/// Posting terminal at acknowledgement time frees the worker row
/// (`jobs_store::update_job_progress`) while the worker process is still busy, so
/// the next queued job is told a worker is free that isn't — deferring the
/// terminal write to actual-stop keeps the two in sync. Transient GET failures are
/// tolerated (read as "not canceled", retried on the next poll) so an API hiccup
/// never aborts a multi-minute run by being misread as a user cancel (sc-4174).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) async fn cancel_requested_peek(api: &ApiClient, job_id: &str) -> bool {
    let outcome: WorkerResult<JobSnapshot> = api.get_json(&format!("/api/v1/jobs/{job_id}")).await;
    match outcome {
        Ok(job) => job.cancel_requested,
        Err(error) => {
            tracing::warn!(
                event = "cancel_poll_failed",
                jobId = %job_id,
                error = %error,
                "cancel poll failed; retrying on the next poll"
            );
            false
        }
    }
}

pub(crate) async fn update_job(
    api: &ApiClient,
    job_id: &str,
    mut payload: ProgressRequest,
) -> WorkerResult<JobSnapshot> {
    // Stamp the reporting worker so the server can reject the write if this
    // worker no longer owns the job (swept stale / canceled / reclaimed). The
    // resulting 409 propagates as WorkerError::Api and aborts the local job
    // handling — i.e. the worker abandons the job (sc-4172).
    payload.worker_id = Some(api.worker_id.clone());
    api.post_json(&format!("/api/v1/jobs/{job_id}/progress"), &payload)
        .await
}

pub(crate) fn progress_payload(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    error: Option<String>,
    result: Option<JsonObject>,
    eta_seconds: Option<ContractNumber>,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error,
        result,
        eta_seconds,
        // The Rust utility worker doesn't run GPU work, so it never reports
        // per-job peak GPU stats. The Python GPU worker (scene_worker) sets
        // these (sc-2086). Same for `backend` — utility jobs run on the CPU
        // worker which never advertises a GPU runtime.
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some("cpu".to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

pub(crate) fn number_from_f64(value: f64) -> ContractNumber {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}
