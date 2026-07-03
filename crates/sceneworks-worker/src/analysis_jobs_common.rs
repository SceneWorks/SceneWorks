//! Shared scaffold for the batched dataset-analysis jobs (sc-8836, F-034).
//!
//! Both `dataset_analysis` (CLIP embeddings, [`dataset_analysis_jobs`]) and `dataset_face_analysis`
//! (SCRFD + ArcFace, [`face_analysis_jobs`]) embed every image of a training dataset, stream per-item
//! progress off a `spawn_blocking` model loop, then POST the resulting records to a rust-api sidecar.
//! The orchestration around that — the `CancelJoinGuard` bound to the blocking task, the `select!`
//! stream loop that scales per-item progress into `0.12..0.90` while polling cancel on the interval
//! tick, the saving update, and the sidecar POST — was a byte-for-byte structural clone in both
//! modules. This module owns it once, parameterized by the two things that legitimately differ: the
//! per-run [`AnalysisJobConfig`] strings/messages and the caller-spawned blocking task that produces
//! the records.
//!
//! Each module keeps its own item parsing (the item shape differs), its own model loading (CLIP loads
//! through the `gen_core` registry inside the blocking task; the face stack stages a weights bundle
//! first), and its own record → sidecar-payload fold (the record fields differ) supplied as
//! `records_payload`. Everything else — the cancel/heartbeat handling reconciled in sc-8835 (F-035) —
//! lives here so a future analysis job re-stamps a config, not ~400 lines of scaffold, and the cancel
//! handling can only ever be fixed in one place.

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::CancelFlag;

/// Per-item progress scaling shared by every batched analysis job: item `index` (0-based) of `total`
/// maps to `0.12 + 0.78 * ((index + 1) / total)`, leaving `0.04/0.08` for prepare/load and `0.94/1.0`
/// for save/complete. Extracted so the one magic ramp is defined (and unit-tested) once (sc-8836).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn item_progress(index: usize, total: usize) -> f64 {
    0.12 + 0.78 * ((index + 1) as f64 / total.max(1) as f64)
}

/// Build a batched-analysis-job [`ProgressRequest`] with the shared field defaults. Every stage of both
/// analysis jobs stamped these identical constructors (`analysis_progress` / `face_progress`); this is
/// the one copy (sc-8836).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn analysis_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

/// The per-run knobs the shared [`run_batched_analysis_job`] scaffold needs: the sidecar endpoint +
/// embedding space, the cancel message, the saving-stage message, and the per-item progress-message
/// builder. Everything that differs between the CLIP and face analysis jobs (beyond the item shape,
/// the model loading, and the record fold) is expressed here.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct AnalysisJobConfig<'a> {
    /// The `.../training/datasets/{dataset_id}/<endpoint_suffix>` sidecar path segment
    /// (`analysis-embeddings` for CLIP, `face-embeddings` for the face stack).
    pub endpoint_suffix: &'a str,
    /// The stable embedding-space tag POSTed alongside the records (guards the sidecar ingest merge).
    pub space: &'a str,
    /// The user-facing cancel message both the in-loop poll and the blocking task raise.
    pub cancel_message: &'a str,
    /// Saving-stage status message (e.g. "Saving embeddings.").
    pub saving_message: &'a str,
    /// The `task_join_error` label if the blocking analysis task panics — kept per-run so each job
    /// preserves its original distinct diagnostic string (`"dataset analysis task join"` for CLIP,
    /// `"dataset face analysis task join"` for the face stack).
    pub join_error_label: &'a str,
    /// Progress-line message for item `index` (0-based) of `total` (e.g. "Analyzed image 3 of 10.").
    /// `Send + Sync` so the enclosing job future stays `Send` (rust-api `tokio::spawn`s the loop).
    pub item_message: &'a (dyn Fn(usize, usize) -> String + Send + Sync),
}

/// Drive one batched dataset-analysis job to completion once its records-producing blocking task is
/// spawned. Owns the shared scaffold (sc-8836): binds `blocking` to `cancel` via a [`CancelJoinGuard`],
/// runs the `select!` stream loop that scales per-item progress into `0.12..0.90` and polls cancel on
/// each heartbeat tick, joins the records on clean exit, POSTs them (folded through `records_payload`)
/// to the sidecar, and emits the completed update built by `completed`. The caller has already parsed
/// items, emitted the preparing/loading progress, created the `(tx, rx)` pair, and spawned `blocking`
/// (whose per-item `tx.blocking_send(index)` drives `rx`).
///
/// The per-item `index` a producer sends (0-based) is reported through [`item_progress`]. Returns the
/// joined records so the caller can derive its completed-result fields (e.g. a with-face count).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_batched_analysis_job<R, P, C>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    cfg: &AnalysisJobConfig<'_>,
    total: usize,
    backend: &str,
    cancel: CancelFlag,
    mut rx: tokio::sync::mpsc::Receiver<usize>,
    blocking: tokio::task::JoinHandle<WorkerResult<Vec<R>>>,
    records_payload: P,
    completed: C,
) -> WorkerResult<Vec<R>>
where
    P: FnOnce(&[R]) -> Vec<Value>,
    C: FnOnce(&[R], Value) -> ProgressRequest,
{
    // Bind the blocking analysis task to its cancel flag (sc-8804, F-003): every `update_job`/
    // `heartbeat` `?` below returns early on a transient POST failure or a 409 (stale-sweep reclaim);
    // on that early return this guard trips `cancel` and aborts the analysis thread instead of leaving
    // it running on a job nobody is consuming. `cancel` is kept alongside (it's `Clone`) for the
    // in-loop cancel poll; the guard drives only the drop-time teardown.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Run the stream loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(index) => {
                            update_job(
                                api,
                                &job.id,
                                analysis_progress(
                                    JobStatus::Running,
                                    ProgressStage::Running,
                                    item_progress(index, total),
                                    &(cfg.item_message)(index, total),
                                    None,
                                    backend,
                                ),
                            )
                            .await?;
                        }
                        None => break,
                    }
                }
                _ = interval.tick() => {
                    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                    match check_cancel(api, &job.id, cfg.cancel_message).await {
                        Ok(()) => {}
                        Err(WorkerError::Canceled(_)) => cancel.cancel(),
                        Err(error) => return Err(error),
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly (channel closed) — reclaim the handle (disarming the drop-guard) and join.
    let records = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error(cfg.join_error_label, error))??;

    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            cfg.saving_message,
            None,
            backend,
        ),
    )
    .await?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let dataset_id = required_payload_string(&job.payload, "datasetId")?;
    let items_payload = records_payload(&records);
    let stored: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/{}",
                cfg.endpoint_suffix
            ),
            &json!({ "space": cfg.space, "items": items_payload }),
        )
        .await?;
    update_job(api, &job.id, completed(&records, stored)).await?;
    Ok(records)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn item_progress_scales_into_the_running_band() {
        // The shared per-item ramp: first of 10 lands just above the 0.12 floor, last lands at 0.90.
        let total = 10;
        assert!((item_progress(0, total) - (0.12 + 0.78 * 0.1)).abs() < 1e-12);
        assert!((item_progress(total - 1, total) - 0.90).abs() < 1e-12);
        // Monotonic across the batch.
        for i in 1..total {
            assert!(item_progress(i, total) > item_progress(i - 1, total));
        }
    }

    #[test]
    fn item_progress_guards_a_zero_total() {
        // A degenerate `total == 0` must not divide by zero (callers reject empty item lists upstream,
        // but the ramp stays total-safe regardless).
        assert!(item_progress(0, 0).is_finite());
    }

    #[test]
    fn analysis_progress_stamps_backend_and_number() {
        let request = analysis_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            "half",
            None,
            "mlx",
        );
        assert_eq!(request.message, "half");
        assert_eq!(request.backend.as_deref(), Some("mlx"));
        assert!(request.result.is_none());
    }
}
