//! Optional sc-26757 source-bound observation journal. Never GPU ownership authority.
//! Counts conservative allocation-capable boundaries, not individual allocator calls.
use crate::{Settings, WorkerError, WorkerResult};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::{Mutex, OnceLock};

struct Journal {
    file: File,
    identity: Value,
    previous: String,
    sequence: u64,
    allocation_capable_entries: u64,
    provider_loads: u64,
    provider_runs: u64,
    bounded_context_entries: u64,
    cache_dispatch_entries: u64,
    job_id: Option<String>,
    engine: Option<String>,
    failed: bool,
}
static JOURNAL: OnceLock<Mutex<Journal>> = OnceLock::new();
fn error(message: impl std::fmt::Display) -> WorkerError {
    WorkerError::Engine(format!("MOSS terminal observation unavailable: {message}"))
}
fn record(journal: &mut Journal, event: &str, engine: Option<&str>) -> WorkerResult<()> {
    if journal.failed {
        return Err(error("journal previously failed"));
    }
    let next = journal
        .sequence
        .checked_add(1)
        .ok_or_else(|| error("sequence overflow"))?;
    let body = serde_json::to_string(&json!({"version":1,"sequence":next,"identity":journal.identity,
        "event":event,"engine":engine,"jobId":journal.job_id,"at":sceneworks_core::time::utc_now(),
        "allocationCapableEntries":journal.allocation_capable_entries,"providerLoads":journal.provider_loads,"providerRuns":journal.provider_runs,
        "boundedContextEntries":journal.bounded_context_entries,"cacheDispatchEntries":journal.cache_dispatch_entries}))
        .map_err(error)?;
    let hash = format!(
        "{:x}",
        Sha256::digest(format!("{}\n{body}", journal.previous).as_bytes())
    );
    let line =
        serde_json::to_string(&json!({"body":body,"previous":journal.previous,"sha256":hash}))
            .map_err(error)?;
    if let Err(failure) = writeln!(journal.file, "{line}").and_then(|()| journal.file.sync_data()) {
        journal.failed = true;
        return Err(error(failure));
    }
    journal.sequence = next;
    journal.previous = hash;
    Ok(())
}
pub(crate) fn initialize(settings: &Settings) -> WorkerResult<()> {
    let Some(path) = std::env::var_os("SCENEWORKS_MOSS_OBSERVATION_JOURNAL") else {
        return Ok(());
    };
    // The API's in-process CPU utility workers must not open the GPU worker's journal.
    if !matches!(settings.gpu_id.as_str(), "mlx" | "1") {
        return Ok(());
    }
    let required = |key| {
        std::env::var(key)
            .ok()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| error(format!("missing {key}")))
    };
    let mut binary = File::open(std::env::current_exe().map_err(error)?).map_err(error)?;
    let mut hasher = Sha256::new();
    let mut bytes = [0_u8; 65536];
    loop {
        let size = binary.read(&mut bytes).map_err(error)?;
        if size == 0 {
            break;
        }
        hasher.update(&bytes[..size]);
    }
    let identity = json!({"candidateId":required("SCENEWORKS_MOSS_CANDIDATE_ID")?,
        "sourceSha":required("SCENEWORKS_MOSS_SOURCE_SHA")?,"inferenceSha":required("SCENEWORKS_MOSS_INFERENCE_SHA")?,
        "workerId":settings.worker_id,"gpuId":settings.gpu_id,"binarySha256":format!("{:x}",hasher.finalize()),
        "producerSha256":format!("{:x}",Sha256::digest(include_bytes!("moss_terminal_observation.rs"))),
        "coverage":"moss_image_base+sft; bounded cache-context before bind; cache-dispatch; runtime load before media catalog; actual MOSS generate/edit calls after admission"});
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(error)?;
    let mut journal = Journal {
        file,
        identity,
        previous: "0".repeat(64),
        sequence: 0,
        allocation_capable_entries: 0,
        provider_loads: 0,
        provider_runs: 0,
        bounded_context_entries: 0,
        cache_dispatch_entries: 0,
        job_id: None,
        engine: None,
        failed: false,
    };
    record(&mut journal, "startup_begin", None)?;
    JOURNAL
        .set(Mutex::new(journal))
        .map_err(|_| error("worker journal initialized twice"))
}
pub(crate) fn ready() -> WorkerResult<()> {
    if let Some(lock) = JOURNAL.get() {
        record(
            &mut *lock.lock().map_err(error)?,
            "ready_after_bounded_startup_probe",
            None,
        )?;
    }
    Ok(())
}
pub(crate) fn heartbeat() -> WorkerResult<()> {
    if let Some(lock) = JOURNAL.get() {
        record(&mut *lock.lock().map_err(error)?, "heartbeat", None)?;
    }
    Ok(())
}
pub(crate) fn enabled() -> bool {
    JOURNAL.get().is_some()
}
pub(crate) fn entry(engine: &str, event: &str) -> WorkerResult<()> {
    if !matches!(engine, "moss_image_base" | "moss_image_sft") {
        return Ok(());
    }
    if let Some(lock) = JOURNAL.get() {
        let mut journal = lock.lock().map_err(error)?;
        if journal.job_id.is_none() || journal.engine.as_deref() != Some(engine) {
            journal.failed = true;
            return Err(error("MOSS boundary bypassed its job binding"));
        }
        if matches!(event, "provider_load" | "provider_run") {
            journal.allocation_capable_entries = journal
                .allocation_capable_entries
                .checked_add(1)
                .ok_or_else(|| error("boundary counter overflow"))?;
        }
        match event {
            "provider_load" => {
                journal.provider_loads = journal
                    .provider_loads
                    .checked_add(1)
                    .ok_or_else(|| error("load counter overflow"))?
            }
            "provider_run" => {
                journal.provider_runs = journal
                    .provider_runs
                    .checked_add(1)
                    .ok_or_else(|| error("run counter overflow"))?
            }
            "cache_context" => {
                journal.bounded_context_entries = journal
                    .bounded_context_entries
                    .checked_add(1)
                    .ok_or_else(|| error("context counter overflow"))?
            }
            "cache_dispatch" => {
                journal.cache_dispatch_entries = journal
                    .cache_dispatch_entries
                    .checked_add(1)
                    .ok_or_else(|| error("dispatch counter overflow"))?
            }
            _ => {
                journal.failed = true;
                return Err(error("unknown MOSS boundary"));
            }
        }
        record(&mut journal, event, Some(engine))?;
    }
    Ok(())
}
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn provider_run() -> WorkerResult<()> {
    let Some(lock) = JOURNAL.get() else {
        return Ok(());
    };
    let engine = lock
        .lock()
        .map_err(error)?
        .engine
        .clone()
        .ok_or_else(|| error("native MOSS call bypassed job scope"))?;
    entry(&engine, "provider_run")
}
pub(crate) struct JobObservation(bool);
pub(crate) fn begin_job(job: &sceneworks_core::contracts::JobSnapshot) -> JobObservation {
    if !job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|id| matches!(id, "moss_image_base" | "moss_image_sft"))
    {
        return JobObservation(false);
    }
    if let Some(lock) = JOURNAL.get() {
        if let Ok(mut journal) = lock.lock() {
            if journal.job_id.is_some() {
                journal.failed = true;
                return JobObservation(false);
            }
            journal.job_id = Some(job.id.clone());
            journal.engine = job
                .payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if record(&mut journal, "job_begin", None).is_err() {
                journal.failed = true;
            }
            return JobObservation(true);
        }
    }
    JobObservation(false)
}
impl Drop for JobObservation {
    fn drop(&mut self) {
        if self.0 {
            if let Some(lock) = JOURNAL.get() {
                if let Ok(mut journal) = lock.lock() {
                    if record(&mut journal, "job_end", None).is_err() {
                        journal.failed = true;
                    }
                    journal.job_id = None;
                    journal.engine = None;
                }
            }
        }
    }
}
