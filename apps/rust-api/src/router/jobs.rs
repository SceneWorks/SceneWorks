use super::*;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/api/v1/jobs/:job_id/yue2-eligibility",
            get(get_job_yue2_eligibility),
        )
        .route("/api/v1/jobs", get(list_jobs).post(create_job))
        .route("/api/v1/jobs/claim", post(claim_job))
        // Clear completed items from the queue (sc-12231, issue #1556). A static
        // segment like `/claim`, so it takes priority over `/jobs/:job_id`.
        .route("/api/v1/jobs/clear", post(clear_jobs))
        // Cancel every pending (queued / pending_caption) item at once (sc-13448).
        // Static segment like `/clear`, so it takes priority over `/jobs/:job_id`.
        .route("/api/v1/jobs/cancel-pending", post(cancel_pending_jobs))
        // Move selected not-yet-started jobs to the front. This is a static segment and must be
        // registered before the per-job routes below.
        .route("/api/v1/jobs/prioritize", post(prioritize_jobs))
        .route("/api/v1/jobs/events", get(job_events))
        .route("/api/v1/jobs/events/ticket", post(create_event_ticket))
        .route("/api/v1/jobs/:job_id", get(get_job))
        .route("/api/v1/jobs/:job_id/cancel", post(cancel_job))
        // Per-job "clear" (sc-12231, issue #1556) — the per-card × dismiss. Distinct
        // from the bulk `/api/v1/jobs/clear` above (2 segments vs 3, no conflict).
        .route("/api/v1/jobs/:job_id/clear", post(clear_job))
        .route("/api/v1/jobs/:job_id/retry", post(retry_job))
        .route("/api/v1/jobs/:job_id/duplicate", post(duplicate_job))
        .route("/api/v1/jobs/:job_id/progress", post(update_job_progress))
        // Per-run generation metrics (epic 10402): worker POSTs on completion;
        // GET returns a single job's block; the aggregate feed powers the
        // Generation Stats comparison charts.
        .route(
            "/api/v1/jobs/:job_id/metrics",
            get(get_job_metrics).post(upsert_job_metrics),
        )
        // Per-item job results (SW-7082): the owning worker appends each finished item; the UI
        // pages them back in order. Kept out of the progress result, which every POST replaces.
        .route(
            "/api/v1/jobs/:job_id/result-items",
            get(list_job_result_items).post(append_job_result_items),
        )
        .route("/api/v1/metrics", get(list_metrics))
        .route("/api/v1/queue", get(queue_summary))
        .route("/api/v1/logs", get(list_logs))
}
