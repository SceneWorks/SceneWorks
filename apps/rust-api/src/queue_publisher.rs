//! Coalesced `queue.updated` publication.
//!
//! Every queue mutation used to rebuild the whole queue summary (stale-worker sweep,
//! per-status counts, active jobs, every worker row) and publish it while holding
//! `queue_snapshot_lock`, before responding. Concurrent mutations therefore lined up
//! behind each other's rebuilds: a cancel whose own DB work took <0.1 s answered after
//! ~20 s on a large install. A mutation now only *requests* a publish; one background
//! driver per [`AppState`] rebuilds and publishes at most once per
//! [`QUEUE_PUBLISH_MIN_INTERVAL`], covering every request made before it started.
//!
//! Ordering and completeness:
//! - A caller requests only after its own transaction committed. The driver reads the
//!   request counter *before* it builds the snapshot, so every request it marks complete
//!   committed before the snapshot's reads began: no request's state is lost. A request
//!   that arrives during a build keeps the driver running for one more round.
//! - The build-and-publish still runs under `queue_snapshot_lock`, so an SSE reconnect
//!   snapshot (`job_events`) is still serialized against it exactly as before.
//! - The sweep is requested, not mandatory: a round sweeps stale workers when any request
//!   it covers asked for one (`claim_job` already swept in its own transaction).
//! - No response depends on this publish: `publish_queue` never returned the summary, and
//!   a failed rebuild is now logged and retried with backoff instead of failing the
//!   mutation that had already committed.
//!
//! [`QueuePublisher::flush`] waits until everything requested so far has been published
//! (skipping the remaining interval), for tests and any caller that must observe it.

use super::*;

/// Upper bound on how often the queue summary is rebuilt and published.
pub(crate) const QUEUE_PUBLISH_MIN_INTERVAL: Duration = Duration::from_millis(250);
const RETRY_BACKOFF_FIRST: Duration = Duration::from_secs(1);
const RETRY_BACKOFF_MAX: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Default)]
struct PublishProgress {
    /// Every request numbered at or below this has been covered by a finished round.
    completed: u64,
    /// The failure of the round that advanced `completed`, if it failed.
    error: Option<String>,
}

#[derive(Default)]
struct DriverState {
    /// Requests numbered so far (monotonic).
    requested: u64,
    /// Requests covered by a finished round (mirrors `PublishProgress::completed`).
    completed: u64,
    /// Some request since the last round started asked for a stale-worker sweep.
    sweep: bool,
    /// A flush is waiting: skip whatever remains of the interval.
    urgent: bool,
    /// The driver task is alive and will look at `requested` again before exiting.
    running: bool,
    driver: Option<tokio::task::JoinHandle<()>>,
    last_started: Option<Instant>,
    /// Set after a failed round; the retry waits this long unless a flush is urgent.
    backoff: Option<Duration>,
}

pub(crate) struct QueuePublisher {
    state: Mutex<DriverState>,
    progress: tokio::sync::watch::Sender<PublishProgress>,
    wake: tokio::sync::Notify,
    min_interval: Duration,
}

impl Default for QueuePublisher {
    fn default() -> Self {
        Self::new(QUEUE_PUBLISH_MIN_INTERVAL)
    }
}

impl QueuePublisher {
    pub(crate) fn new(min_interval: Duration) -> Self {
        Self {
            state: Mutex::new(DriverState::default()),
            progress: tokio::sync::watch::Sender::new(PublishProgress::default()),
            wake: tokio::sync::Notify::new(),
            min_interval,
        }
    }

    /// Ask for a `queue.updated` publish reflecting everything committed so far.
    /// Returns at once; must be called from within the Tokio runtime.
    pub(crate) fn request(&self, app: &AppState, sweep: bool) -> u64 {
        let mut state = self.state.lock();
        state.requested += 1;
        state.sweep |= sweep;
        let request = state.requested;
        self.ensure_driver(&mut state, app);
        request
    }

    /// Wait until every request made before this call has been published; returns the
    /// error of the round that covered it, if that round failed.
    pub(crate) async fn flush(&self, app: &AppState) -> Result<(), ApiError> {
        let target = {
            let mut state = self.state.lock();
            if state.completed >= state.requested {
                return Ok(());
            }
            state.urgent = true;
            self.ensure_driver(&mut state, app);
            state.requested
        };
        self.wake.notify_one();
        let mut progress = self.progress.subscribe();
        let progress = progress
            .wait_for(|progress| progress.completed >= target)
            .await
            .map_err(|_| ApiError::internal("queue publisher stopped"))?;
        match &progress.error {
            Some(error) => Err(ApiError::internal(error.clone())),
            None => Ok(()),
        }
    }

    fn ensure_driver(&self, state: &mut DriverState, app: &AppState) {
        // A driver spawned on a runtime that has since shut down (a test that reuses a
        // state across runtimes) is finished without having cleared `running`.
        let alive = state.running
            && state
                .driver
                .as_ref()
                .is_some_and(|driver| !driver.is_finished());
        if !alive {
            state.running = true;
            let app = app.clone();
            state.driver = Some(tokio::spawn(async move { drive(app).await }));
        }
    }

    /// How long the driver should still wait before starting the next round.
    fn remaining_wait(&self, state: &DriverState) -> Duration {
        if state.urgent {
            return Duration::ZERO;
        }
        let gap = state.backoff.unwrap_or(self.min_interval);
        state
            .last_started
            .map(|started| (started + gap).saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::ZERO)
    }
}

async fn drive(app: AppState) {
    let publisher = app.queue_publisher.clone();
    loop {
        // Honour the interval (or the retry backoff), re-evaluating whenever a flush
        // marks the queue urgent.
        loop {
            let wait = publisher.remaining_wait(&publisher.state.lock());
            if wait.is_zero() {
                break;
            }
            tokio::select! {
                () = tokio::time::sleep(wait) => {}
                () = publisher.wake.notified() => {}
            }
        }
        let (target, sweep) = {
            let mut state = publisher.state.lock();
            if state.completed >= state.requested {
                state.running = false;
                state.urgent = false;
                return;
            }
            state.urgent = false;
            state.last_started = Some(Instant::now());
            (state.requested, std::mem::take(&mut state.sweep))
        };
        let result = rebuild_and_publish(&app, sweep).await;
        let error = result.err().map(|error| error.detail);
        {
            let mut state = publisher.state.lock();
            state.completed = target;
            if error.is_some() {
                // Re-arm a retry so subscribers still converge on the final state.
                state.requested += 1;
                state.sweep |= sweep;
                state.backoff = Some(
                    state
                        .backoff
                        .map_or(RETRY_BACKOFF_FIRST, |backoff| backoff * 2)
                        .min(RETRY_BACKOFF_MAX),
                );
            } else {
                state.backoff = None;
            }
        }
        if let Some(error) = &error {
            tracing::warn!(
                event = "queue_publish_failed",
                error = %error,
                "could not rebuild the queue summary; retrying"
            );
        }
        publisher.progress.send_replace(PublishProgress {
            completed: target,
            error,
        });
    }
}

async fn rebuild_and_publish(app: &AppState, sweep: bool) -> Result<(), ApiError> {
    let _snapshot_guard = app.queue_snapshot_lock.lock().await;
    let queue = queue_summary_snapshot_inner(app.clone(), !sweep).await?;
    publish(app, "queue.updated", &queue);
    Ok(())
}

/// Test-only router layer: after each request, wait for the queue publishes it
/// requested, so route tests keep observing `queue.updated` (and the stale-worker
/// sweep it runs) as part of the request — the explicit flush the coalescing
/// publisher needs, applied in one place instead of before every assertion. A round
/// that fails turns the response into that error, as the inline publish used to.
/// `TestHooks::defer_queue_publishes` disables it for tests of the coalescing itself.
#[cfg(test)]
pub(crate) async fn flush_after_test_request(
    State(app): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    if app
        .test_hooks
        .defer_queue_publishes
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return response;
    }
    match app.queue_publisher.flush(&app).await {
        Ok(()) => response,
        Err(error) => error.into_response(),
    }
}

/// Request a `queue.updated` publish covering everything this request committed.
/// Returns immediately: the rebuild runs on the coalescing publisher (see
/// this module), at most every `QUEUE_PUBLISH_MIN_INTERVAL`, and runs the
/// stale-worker sweep before reading the summary.
pub(crate) fn request_queue_publish(state: &AppState) {
    state.queue_publisher.request(state, true);
}

/// Like [`request_queue_publish`], but does not ask for the stale-worker sweep
/// because the caller already ran one in its own transaction this request
/// (sc-8889 / F-087). The coalesced round still sweeps if any other request it
/// covers asked for one. Use only right after a `mark_stale_workers_interrupted`
/// call.
pub(crate) fn request_queue_publish_skip_sweep(state: &AppState) {
    state.queue_publisher.request(state, false);
}

/// Wait until every queue publish requested so far has gone out, skipping the
/// coalescing interval. Production handlers never need this (no response depends
/// on the published snapshot); tests use it to observe `queue.updated`
/// deterministically.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) async fn flush_queue_publishes(state: &AppState) -> Result<(), ApiError> {
    state.queue_publisher.flush(state).await
}
