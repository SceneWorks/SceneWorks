//! Job status/progress plumbing: building [`ProgressRequest`]s and posting terminal/cancel states.
use super::*;

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
