//! Job status/progress plumbing: building [`ProgressRequest`]s and posting terminal/cancel states.
use super::*;

/// Grace window the [`CancelJoinGuard::cancel_and_join`] teardown gives the tripped engine to wind
/// down cooperatively (sc-8804, F-003) before it force-abandons the task. The converted consumers
/// wrap `tokio::task::spawn_blocking`, and `JoinHandle::abort()` is INERT on an already-running
/// blocking task — so the teardown's real job is to AWAIT the task long enough for the engine's
/// between-steps cancel-flag poll (a denoise step, a VAE decode, a checkpoint save, or the cold
/// `gen_core::load()`/quantize) to actually stop the GPU work before the consumer returns and the
/// worker claims the next job. 30s covers a between-steps cooperative stop for the slowest of those
/// ops without being so long a genuinely wedged engine can hang the worker for minutes.
pub(crate) const CANCEL_JOIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Hard abandon deadline (sc-8804, F-003): once the grace window elapses the teardown `abort()`s the
/// handle (best-effort — inert on a running blocking task) and waits at most this long for the
/// runtime to reap a cooperative task before giving up and letting the worker proceed regardless. A
/// wedged blocking task can outlive this; the bound guarantees the worker is never blocked forever.
pub(crate) const CANCEL_JOIN_ABANDON: std::time::Duration = std::time::Duration::from_secs(5);

/// RAII guard binding a running blocking/GPU task to the `select!`-loop that streams its progress
/// (sc-8804, F-003). Every streaming consumer spawns a blocking task that owns a long GPU denoise
/// or training run, then sits in a `select! { channel, interval }` loop posting progress via
/// `update_job(...).await?` / `heartbeat(...).await?`. On a transient POST failure or a 409 (the
/// stale-sweep reclaimed the job) that `?` returns the consumer early — but a bare
/// `JoinHandle` does NOT stop its task when dropped, so the GPU/training thread keeps burning
/// unified memory while the worker returns and claims the next job. That is two concurrent GPU
/// workloads on one Metal device (the sc-8390 SIGKILL/OOM class).
///
/// This guard closes that gap. The PRIMARY teardown is [`Self::cancel_and_join`]: on any error path
/// (the `?`-return shapes) the consumer calls it explicitly to trip the engine [`CancelFlag`] and
/// then AWAIT a bounded join ([`CANCEL_JOIN_GRACE`] + [`CANCEL_JOIN_ABANDON`]) so the blocking GPU
/// task has actually wound down (or hit the hard abandon deadline) before the job function yields
/// and the worker claims the next job. `abort()` alone can't do this — it is inert on a running
/// blocking task, so an awaited bounded join is the only teardown that satisfies "worker does not
/// claim the next job until the GPU task wound down". The synchronous `Drop` remains as a
/// best-effort backstop (trip + `abort()`) for any path that forgets the explicit teardown, but it
/// cannot await and so must not be relied on as the primary mechanism. The happy path calls
/// [`Self::into_handle`] to reclaim the raw handle and `.await` it normally — the guard is then
/// inert (its `Drop` is a no-op), so a clean completion never aborts anything.
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
    /// Bind a spawned blocking task to its engine cancel flag. On an error path the consumer calls
    /// [`Self::cancel_and_join`] to trip the flag and awaited-bounded-join the task before the job
    /// function unwinds.
    ///
    /// Pass `None` for the flag ONLY when the task's engine exposes no cooperative cancel. In that
    /// case there is a HONEST RESIDUAL: `cancel_and_join` still awaits the task for the grace window
    /// (so the consumer does not drop-and-run), but with no flag to trip, a `spawn_blocking` task
    /// that ignores `abort()` runs to its natural end — it is NOT stopped, only waited-for up to the
    /// bounded deadline. Do not read the old "abort-on-drop still applies" as protection: `abort()`
    /// is inert on a running blocking task. The remaining `None`-cancel callers are single-shot
    /// bounded operations with nothing to interrupt (kps/SCRFD, person-detect/YOLO11, ArcFace
    /// compare, the interleave document write) — a deliberate per-engine decision, see
    /// [`run_blocking_with_heartbeat`]. Everything loop-shaped passes `Some` (sc-9123): the
    /// multi-minute engine-API-locked case (SenseNova VQA/interleave) got a real per-token/per-step
    /// cancel flag, and the worker-side pose loops (DWPose batch detect + skeleton render) check a
    /// real flag between images/persons.
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

    /// PRIMARY error-path teardown (sc-8804, F-003). Trip the engine cancel flag (cooperative bail),
    /// then AWAIT a bounded join of the blocking task so the GPU/training work has actually wound
    /// down before the consumer returns and the worker claims the next job. `abort()` is inert on an
    /// already-running blocking task, so the awaited join — NOT the abort — is what closes the
    /// double-GPU window: it holds the job function until the engine's between-steps cancel poll
    /// stops the work, up to [`CANCEL_JOIN_GRACE`]. If the grace window elapses (a wedged / not-yet-
    /// cancellable op) it `abort()`s (best effort) and waits at most [`CANCEL_JOIN_ABANDON`] more for
    /// a cooperative task to be reaped, then returns regardless so a stuck engine can never hang the
    /// worker forever. Idempotent: after it runs, the guard is disarmed (its `Drop` is a no-op).
    ///
    /// Must be called on EVERY error path — the `?` on a progress/heartbeat POST failure and the
    /// channel-close path alike — before the error propagates out of the job function.
    pub(crate) async fn cancel_and_join(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        let Some(mut handle) = self.handle.take() else {
            return;
        };
        // Bounded cooperative wind-down: await the task itself so the consumer does not return while
        // the GPU work is still live. A between-steps cancel poll resolves the handle here.
        if tokio::time::timeout(CANCEL_JOIN_GRACE, &mut handle)
            .await
            .is_ok()
        {
            return;
        }
        // Grace exceeded — the op did not reach a cancel checkpoint in time. `abort()` (inert on a
        // running blocking task, effective on a cooperative async one), then give the runtime a
        // bounded window to reap it before we abandon the wait and let the worker proceed.
        handle.abort();
        let _ = tokio::time::timeout(CANCEL_JOIN_ABANDON, handle).await;
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
/// `WorkerError::Canceled`. On a heartbeat POST failure it performs the F-003 teardown
/// (`guard.cancel_and_join()`) before returning the error, so a `Some(cancel)` task is bounded-
/// joined rather than dropped-and-run (sc-8804).
///
/// Pass `None` ONLY for paths whose work is a single bounded operation with no loop for a flag to
/// interrupt. HONEST RESIDUAL: with `None` there is no flag to trip, so the teardown can only
/// AWAIT the task (up to the grace window) — a `spawn_blocking` task that ignores `abort()` still
/// runs to its natural end. The COMPLETE list of remaining `None` callers, each an explicit
/// per-engine decision documented at its call site (sc-9123): the single-shot detectors
/// (kps/SCRFD, person-detect/YOLO11) and the ArcFace embedding compare — each is one bounded
/// forward pass on one image/frame (plus a cold weight load), so threading cancel through those
/// engine surfaces would buy nothing the bounded join doesn't already give — the smart-select SAM3
/// image ops (box/points, `segment_jobs`, sc-8908 / F-106), likewise one bounded SAM3 forward pass
/// on one image with no per-step loop for a flag to poll (the video `propagate` path DOES thread a
/// real flag) — and the interleave document write (bounded PNG encode + fs rename, where a mid-write
/// abort is worse than finishing). Everything loop-shaped passes `Some`: SenseNova VQA/interleave threads a REAL
/// `CancelFlag` the engines poll per decoded token and per denoise step (mlx-gen #634 + the candle
/// sc-9123 sibling), and the worker-side pose loops (DWPose multi-image batch detect + the
/// per-person skeleton render) check a real flag between iterations in worker code — no engine
/// change needed (sc-9123). If you add a `None` caller, add it to this list and document the
/// decision at the call site. Do NOT re-add a "abort-on-drop still applies" claim — abort is inert
/// on a running blocking task.
///
/// Every consumer is a job handler gated behind `any(target_os = "macos", feature =
/// "backend-candle")`, so on the plain-Linux parity build (neither) this is unused — allow
/// dead_code there only, keeping real dead-code detection on the configs that do call it.
/// The instant a user cancel is first observed (the flag has just been tripped), the optional
/// `on_cancel_acknowledged` closure runs once — before the terminal `Canceled` is posted. Callers
/// use it to post an intermediate "Canceling…" job update so the UI acknowledges the cancel while an
/// un-interruptible op finishes, instead of appearing frozen until it flips terminal (the image
/// upscale path, sc-8928). Its error is treated exactly like a heartbeat POST failure: bounded-join
/// teardown, then propagate. Pass [`no_cancel_ack`] when there's nothing extra to post.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_blocking_with_heartbeat<R, F, Fut>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    cancel: Option<gen_core::CancelFlag>,
    cancel_message: &str,
    task_label: &'static str,
    on_cancel_acknowledged: Option<F>,
    task: tokio::task::JoinHandle<WorkerResult<R>>,
) -> WorkerResult<R>
where
    R: Send + 'static,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = WorkerResult<()>>,
{
    let mut on_cancel_acknowledged = on_cancel_acknowledged;
    // Bind the blocking task to its cancel flag. On any heartbeat/`?` early return below we perform
    // the explicit awaited bounded-join teardown (`guard.cancel_and_join()`) BEFORE the error
    // propagates, so the still-running GPU task is actually wound down (or hard-abandoned) before
    // this function yields and the worker claims the next job — a bare `abort()` on drop is inert on
    // a running blocking task and would leak it (sc-8804, F-003).
    let mut guard: CancelJoinGuard<gen_core::CancelFlag, WorkerResult<R>> =
        CancelJoinGuard::new(cancel.clone(), task);
    let mut canceled = false;
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut *guard.handle_mut() => {
                // The task has RESOLVED. Disarm the guard first (before any `?`) so a task/join
                // error never drops an armed guard and re-aborts an already-finished handle
                // (reviewer note: `??` used to precede disarm). The reclaimed value is handled
                // below; the guard is now inert.
                guard.disarm();
                let value = result.map_err(|error| task_join_error(task_label, error))?;
                // An engine that honors the tripped `cancel` flag itself (the per-frame video
                // cancel contract, gen-core d8038beb) surfaces `WorkerError::Canceled` from
                // inside the task; post the terminal `Canceled` here exactly like the
                // poll-detected path below so the job never dangles non-terminal (sc-8807).
                if let Err(WorkerError::Canceled(message)) = &value {
                    let message = message.clone();
                    mark_job_canceled(api, job_id, &message).await?;
                    return Err(WorkerError::Canceled(message));
                }
                let value = value?;
                if canceled {
                    mark_job_canceled(api, job_id, cancel_message).await?;
                    return Err(WorkerError::Canceled(cancel_message.to_owned()));
                }
                return Ok(value);
            }
            _ = interval.tick() => {
                // A heartbeat POST failure / 409 here must NOT drop-and-run: bounded-join the task
                // first so the GPU work has wound down before we return the error (sc-8804).
                if let Err(error) =
                    heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await
                {
                    guard.cancel_and_join().await;
                    return Err(error);
                }
                if let Some(flag) = &cancel {
                    // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API
                    // poll (a local flag read) so a quit trips the blocking task's engine cancel at the
                    // next heartbeat tick instead of winding down only at the loop grace window.
                    if !canceled && (shutdown_requested() || cancel_requested_peek(api, job_id).await) {
                        flag.cancel();
                        canceled = true;
                        // Fire the one-shot cancel-acknowledged hook (e.g. post an intermediate
                        // "Canceling…" update). Its failure is teardown-then-propagate like a
                        // heartbeat POST failure (sc-8928).
                        if let Some(hook) = on_cancel_acknowledged.take() {
                            if let Err(error) = hook().await {
                                guard.cancel_and_join().await;
                                return Err(error);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A well-typed `None` for [`run_blocking_with_heartbeat`]'s `on_cancel_acknowledged` param, for the
/// callers with no intermediate "Canceling…" update to post. Naming a concrete future type lets the
/// compiler infer `F`/`Fut` at the `None` call sites without a turbofish (sc-8928).
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn no_cancel_ack() -> Option<fn() -> std::future::Ready<WorkerResult<()>>> {
    None
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

tokio::task_local! {
    /// The process-shutdown [`gen_core::CancelFlag`] for the in-flight job (sc-9618, F-043 follow-up).
    /// `run_job_with_shutdown` trips a shared flag on SIGTERM/Ctrl-C and awaits the un-dropped job
    /// future (bounded by `shutdown_timeout_seconds`) — that guarantees correctness (no dangling
    /// `running`) for EVERY handler regardless of whether it observes the flag. This task-local lets the
    /// per-engine GPU consumer loops ALSO honor it at their existing per-step cancel checkpoints so a
    /// prompt/gen stops mid-step on quit instead of waiting out the grace window, WITHOUT threading the
    /// flag through ~30 stream-handler signatures (and desyncing an MLX/candle twin in the process).
    /// `run_utility_job` scopes it via [`with_shutdown_flag`] around its whole dispatch, and every
    /// consumer loop (image `consume_gen_events`, video `generate_video`, training
    /// `consume_training_events`, `run_batched_analysis_job`, image-detail) runs inside that same task
    /// (the only `tokio::spawn`s below it are the model-producer tasks, which poll the ENGINE flag), so
    /// the scope reaches every checkpoint. Unset outside a job (e.g. unit tests) ⇒ [`shutdown_requested`]
    /// reads `false`.
    pub(crate) static SHUTDOWN_FLAG: gen_core::CancelFlag;
}

/// Run `future` with the process-shutdown [`SHUTDOWN_FLAG`] task-local bound to `shutdown`, so every
/// consumer loop awaited within it can consult [`shutdown_requested`] at its cancel checkpoints
/// (sc-9618). Scoped once around `run_utility_job`'s dispatch.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) async fn with_shutdown_flag<F>(shutdown: gen_core::CancelFlag, future: F) -> F::Output
where
    F: std::future::Future,
{
    SHUTDOWN_FLAG.scope(shutdown, future).await
}

/// Whether a process shutdown has been requested for the in-flight job (sc-9618). Reads the
/// [`SHUTDOWN_FLAG`] task-local scoped by [`with_shutdown_flag`]; `false` when unset (outside a job, or
/// a unit test that calls a consumer directly). Consumer-loop cancel checkpoints OR this in alongside
/// the API `cancel_requested_peek` user-cancel poll, so a shutdown trips the engine cancel at the next
/// step exactly like a user cancel would.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn shutdown_requested() -> bool {
    SHUTDOWN_FLAG
        .try_with(|flag| flag.is_cancelled())
        .unwrap_or(false)
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
        // per-job peak GPU stats. The native GPU worker (MLX/candle) sets
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

#[cfg(test)]
mod shutdown_flag_tests {
    use super::*;

    /// sc-9618: outside a scoped job the task-local is unset, so a consumer loop's checkpoint reads
    /// "no shutdown" and keeps running (never spuriously cancels a job in a unit test).
    #[tokio::test]
    async fn shutdown_requested_is_false_when_unscoped() {
        assert!(!shutdown_requested());
    }

    /// sc-9618: inside `with_shutdown_flag`, the checkpoint reads the scoped flag — `false` until it is
    /// tripped, `true` once the process-shutdown flag is cancelled — exactly what
    /// `run_job_with_shutdown` does on SIGTERM/Ctrl-C. This is the observable both `consume_gen_events`
    /// and every twin consult at each per-step cancel checkpoint.
    #[tokio::test]
    async fn shutdown_requested_tracks_the_scoped_flag() {
        let flag = gen_core::CancelFlag::new();
        let flag_for_scope = flag.clone();
        with_shutdown_flag(flag_for_scope, async {
            // Not yet tripped inside the scope.
            assert!(!shutdown_requested());
            // Tripping the (shared clone of the) flag is observed at the next checkpoint read.
            flag.cancel();
            assert!(shutdown_requested());
        })
        .await;
        // Back outside the scope, the task-local is gone again.
        assert!(!shutdown_requested());
    }

    /// sc-9618: the caption (`caption_jobs::run_training_caption_job`) and prompt-refine
    /// (`prompt_refine_jobs`) GPU consumer loops both short-circuit their `check_cancel` API poll with
    /// `if shutdown_requested() { cancel.cancel() }` at the heartbeat checkpoint. This asserts the exact
    /// semantics that arm relies on: outside a shutdown, the engine `cancel` flag is left untouched
    /// (normal operation is unaffected); once the process-shutdown flag trips, the checkpoint fires the
    /// engine cancel so the captioner/decode bails at its next per-item/per-token check.
    #[tokio::test]
    async fn shutdown_checkpoint_trips_the_engine_cancel_only_on_shutdown() {
        let shutdown = gen_core::CancelFlag::new();
        let shutdown_for_scope = shutdown.clone();
        with_shutdown_flag(shutdown_for_scope, async {
            // The engine-side cancel flag the producer (blocking captioner / decode) polls.
            let engine_cancel = gen_core::CancelFlag::new();
            // Before shutdown, the checkpoint's `if shutdown_requested()` is false — no spurious cancel.
            if shutdown_requested() {
                engine_cancel.cancel();
            }
            assert!(!engine_cancel.is_cancelled());
            // A process shutdown trips the task-local; the next checkpoint tick fires the engine cancel.
            shutdown.cancel();
            if shutdown_requested() {
                engine_cancel.cancel();
            }
            assert!(engine_cancel.is_cancelled());
        })
        .await;
    }
}
