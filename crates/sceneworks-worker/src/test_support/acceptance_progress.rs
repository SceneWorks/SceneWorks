//! Test-support-only observation/cancellation. No training algorithm is implemented here.
//! Native events are forwarded exactly once. Cancellation comes only from the real API.
use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Default)]
pub struct AcceptanceProgressObservation {
    pub steps: Vec<u32>,
    pub training_losses: Vec<f64>,
    pub observed_cancel: Option<gen_core::CancelFlag>,
    pub token_id: String,
    pub checkpoint_stall: Option<serde_json::Value>,
    pub api_cancel: Option<serde_json::Value>,
    pub errors: Vec<String>,
    pub native_returned: bool,
}

pub type AcceptanceProgressLedger = Arc<Mutex<AcceptanceProgressObservation>>;

fn unsupported(message: impl Into<String>) -> gen_core::Error {
    gen_core::Error::Unsupported(message.into())
}
fn lock_until(
    ledger: &AcceptanceProgressLedger,
    deadline: Instant,
) -> gen_core::Result<MutexGuard<'_, AcceptanceProgressObservation>> {
    loop {
        match ledger.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(error)) => return Err(unsupported(error.to_string())),
            Err(TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    return Err(unsupported("acceptance observation lock timed out"));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}
fn unix_ms() -> gen_core::Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| unsupported(format!("acceptance clock unavailable: {error}")))?
        .as_millis();
    u64::try_from(millis).map_err(|error| unsupported(error.to_string()))
}

/// Snapshot observations only. A missing/contended ledger is an error, not an empty success.
pub fn acceptance_progress_snapshot(
    ledger: &AcceptanceProgressLedger,
) -> gen_core::Result<serde_json::Value> {
    let observation = ledger
        .try_lock()
        .map_err(|error| unsupported(error.to_string()))?;
    Ok(serde_json::json!({
        "kind":"moss-native-progress-observations",
        "steps":observation.steps,
        "trainingLosses":observation.training_losses,
        "tokenId":observation.token_id,
        "tokenObservedCanceled":observation.observed_cancel.as_ref().map(|token| token.is_cancelled()),
        "checkpointStall":observation.checkpoint_stall,
        "apiCancel":observation.api_cancel,
        "errors":observation.errors,
        "nativeReturned":observation.native_returned
    }))
}

struct ObservedTrainer {
    inner: Box<dyn gen_core::Trainer>,
    ledger: AcceptanceProgressLedger,
    token_id: String,
    stall_checkpoint20: bool,
    deadline: Instant,
}

/// Construct inside the existing blocking loader. The immutable deadline cannot
/// be more than the frozen case bound; no environment knob can enlarge it.
pub fn observe_acceptance_trainer(
    inner: Box<dyn gen_core::Trainer>,
    ledger: AcceptanceProgressLedger,
    token_id: String,
    stall_checkpoint20: bool,
    deadline: Instant,
) -> gen_core::Result<Box<dyn gen_core::Trainer>> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if token_id.is_empty() || remaining.is_zero() || remaining > Duration::from_secs(3600) {
        return Err(unsupported(
            "acceptance trainer needs a nonempty token identity and remaining frozen deadline",
        ));
    }
    Ok(Box::new(ObservedTrainer {
        inner,
        ledger,
        token_id,
        stall_checkpoint20,
        deadline,
    }))
}

impl gen_core::Trainer for ObservedTrainer {
    fn descriptor(&self) -> &gen_core::TrainerDescriptor {
        self.inner.descriptor()
    }
    fn validate(&self, request: &gen_core::TrainingRequest) -> gen_core::Result<()> {
        self.inner.validate(request)
    }
    fn train(
        &mut self,
        request: &gen_core::TrainingRequest,
        progress: &mut dyn FnMut(gen_core::TrainingProgress),
    ) -> gen_core::Result<gen_core::TrainingOutput> {
        let ledger = Arc::clone(&self.ledger);
        let failed = Arc::new(AtomicBool::new(false));
        let observer_failed = Arc::clone(&failed);
        let token = request.cancel.clone();
        let token_id = self.token_id.clone();
        let stall = self.stall_checkpoint20;
        let deadline = self.deadline;
        {
            let mut observation = lock_until(&ledger, deadline)?;
            if observation.observed_cancel.is_some() {
                return Err(unsupported(
                    "one progress ledger per native subrun is required",
                ));
            }
            observation.observed_cancel = Some(token.clone());
            observation.token_id = token_id.clone();
        }
        let result = self.inner.train(request, &mut |event| {
            let checkpoint20 =
                matches!(&event, gen_core::TrainingProgress::Checkpoint { step: 20 });
            match lock_until(&ledger, deadline) {
                Ok(mut observation) => {
                    if let gen_core::TrainingProgress::Training { step, total, loss } = &event {
                        observation.steps.push(*step);
                        observation.training_losses.push(f64::from(*loss));
                        if !loss.is_finite() || *total != 60 {
                            observation
                                .errors
                                .push("nonfinite native loss or changed original horizon".into());
                            observer_failed.store(true, Ordering::Release);
                        }
                    }
                }
                Err(_) => observer_failed.store(true, Ordering::Release),
            }
            // Production consumes the original event before the checkpoint wait.
            // Never omit, duplicate, synthesize or reshape any native update/event.
            progress(event);
            if checkpoint20 && stall {
                let durable = crate::paths::latest_moss_training_resume_step(
                    &request.output_dir,
                    &request.file_name,
                );
                if !matches!(durable, Ok(Some(20))) {
                    observer_failed.store(true, Ordering::Release);
                    if let Ok(mut observation) = lock_until(&ledger, deadline) {
                        observation
                            .errors
                            .push("native checkpoint 20 is not durably verified".into());
                    }
                    return;
                }
                let entered = match unix_ms() {
                    Ok(value) => value,
                    Err(_) => {
                        observer_failed.store(true, Ordering::Release);
                        return;
                    }
                };
                let remaining = deadline.saturating_duration_since(Instant::now());
                if let Ok(mut observation) = lock_until(&ledger, deadline) {
                    observation.checkpoint_stall = Some(serde_json::json!({
                        "scope":"test-support-only", "referenceStallEnabled":false,
                        "checkpointStep":20, "enteredUnixMs":entered, "exitedUnixMs":null,
                        "waitedForTokenId":token_id, "timeoutSeconds":remaining.as_secs_f64(),
                        "state":"waiting-for-api-cancellation", "timedOut":false,
                        "tokenObservedCanceled":false, "updatesSkippedOrReplayed":0,
                        "durableCheckpointVerified":true
                    }));
                } else {
                    observer_failed.store(true, Ordering::Release);
                    return;
                }
                // Do not mutate the token. Only the normal worker's real API poll
                // can release this wait. It always has an internal absolute timeout.
                while !token.is_cancelled() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                let timed_out = Instant::now() >= deadline;
                if timed_out {
                    observer_failed.store(true, Ordering::Release);
                }
                match (lock_until(&ledger, deadline), unix_ms()) {
                    (Ok(mut observation), Ok(exited)) => {
                        if let Some(state) = observation.checkpoint_stall.as_mut() {
                            state["exitedUnixMs"] = serde_json::json!(exited);
                            state["state"] = serde_json::json!(if timed_out {
                                "timeout-failed-closed"
                            } else {
                                "released-by-api-cancellation"
                            });
                            state["timedOut"] = serde_json::json!(timed_out);
                            state["tokenObservedCanceled"] =
                                serde_json::json!(token.is_cancelled());
                        }
                    }
                    _ => observer_failed.store(true, Ordering::Release),
                }
            }
        });
        // Returning an error only after normal native teardown avoids unwinding
        // through a live Metal buffer. The external guard blocks any next case if
        // this scope/owned process cannot positively release.
        match lock_until(&self.ledger, self.deadline) {
            Ok(mut observation) => {
                observation.native_returned = true;
                if let Err(error) = &result {
                    observation.errors.push(format!("native result: {error}"));
                }
            }
            Err(_) => failed.store(true, Ordering::Release),
        }
        if failed.load(Ordering::Acquire) {
            return Err(unsupported("acceptance observation/checkpoint wait failed closed; native result retained in the ledger"));
        }
        result
    }
}

/// Poll a test-owned cancellation request or the durable checkpoint wait, then
/// invoke the REAL cancellation endpoint. This never calls CancelFlag::cancel.
/// The case owner must continue driving/joining execution after an error; do not
/// drop a live native task or force-kill its process to satisfy a wall clock.
pub async fn acceptance_api_cancel_controller(
    settings: &Settings,
    job_id: &str,
    ledger: AcceptanceProgressLedger,
    stop: Arc<AtomicBool>,
    cancel_file: Option<PathBuf>,
    deadline: Instant,
) -> gen_core::Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() || remaining > Duration::from_secs(3600) {
        return Err(unsupported(
            "acceptance controller requires the remaining frozen case deadline",
        ));
    }
    if settings.access_token.is_some() {
        return Err(unsupported(
            "acceptance controller refuses persisted credentials",
        ));
    }
    let url =
        reqwest::Url::parse(&settings.api_url).map_err(|error| unsupported(error.to_string()))?;
    if url.scheme() != "http"
        || !matches!(url.host_str(), Some("127.0.0.1" | "::1"))
        || !url.username().is_empty()
        || url.password().is_some()
        || job_id.is_empty()
    {
        return Err(unsupported(
            "acceptance cancellation requires the isolated loopback API",
        ));
    }
    let client = reqwest::Client::new();
    let mut sent = false;
    while !stop.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(unsupported(
                "acceptance API cancellation controller deadline exceeded",
            ));
        }
        let waiting = {
            let observation = lock_until(&ledger, deadline)?;
            observation
                .checkpoint_stall
                .as_ref()
                .is_some_and(|value| value["state"] == "waiting-for-api-cancellation")
        };
        let external = if let Some(file) = &cancel_file {
            match std::fs::read(file) {
                Ok(bytes) => {
                    let value: serde_json::Value =
                        serde_json::from_slice(&bytes).map_err(|error| {
                            unsupported(format!("malformed owned cancellation request: {error}"))
                        })?;
                    value.get("cancel").and_then(serde_json::Value::as_bool) == Some(true)
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => return Err(unsupported(error.to_string())),
            }
        } else {
            false
        };
        if !sent && (waiting || external) {
            let response = client
                .post(format!(
                    "{}/api/v1/jobs/{job_id}/cancel",
                    settings.api_url.trim_end_matches('/')
                ))
                .timeout(remaining)
                .json(&serde_json::json!({}))
                .send()
                .await
                .map_err(|error| unsupported(format!("real API cancel failed: {error}")))?;
            let status = response.status();
            let body = response
                .text()
                .await
                .map_err(|error| unsupported(error.to_string()))?;
            let at = unix_ms()?;
            lock_until(&ledger, deadline)?.api_cancel = Some(serde_json::json!({
                "requestedUnixMs":at, "httpStatus":status.as_u16(), "response":body,
                "stimulus":if external { "owned-external-cancel-request" } else { "durable-checkpoint-20" }
            }));
            if !status.is_success() {
                return Err(unsupported(format!(
                    "real API cancel refused with {status}"
                )));
            }
            sent = true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}
