use super::*;

pub(crate) async fn queue_summary(
    State(state): State<AppState>,
) -> Result<Json<QueueSummary>, ApiError> {
    Ok(Json(queue_summary_snapshot(state).await?))
}

pub(crate) async fn list_workers(
    State(state): State<AppState>,
) -> Result<Json<Vec<WorkerSnapshot>>, ApiError> {
    let (sweep, workers) = store_call(state.clone(), move |store, timeout| {
        let sweep = store.mark_stale_workers_interrupted(timeout)?;
        let workers = store.list_workers();
        Ok((sweep, workers))
    })
    .await?;
    handle_stale_sweep(&state, &sweep);
    let workers = workers?;
    Ok(Json(workers))
}

/// Person-workflow readiness derived from the live (non-offline) workers: a
/// capability is ready when some live worker advertises it. Surfaces, per
/// dependency, whether real detection/tracking/segmentation/replacement (and the
/// procedural previews) can actually run, so the UI can gate Replace Person and
/// explain why an action is unavailable (sc-1484).
pub(crate) fn person_readiness_from_workers(workers: &[WorkerSnapshot]) -> Value {
    let live: Vec<&WorkerSnapshot> = workers
        .iter()
        .filter(|worker| worker.status != WorkerStatus::Offline)
        .collect();
    let entry = |capability: WorkerCapability| {
        let cap = capability.as_str();
        let ready = live.iter().any(|worker| {
            worker
                .capabilities
                .iter()
                .any(|owned| owned.as_str() == cap)
        });
        json!({ "capability": cap, "ready": ready })
    };
    json!({
        "detect": entry(WorkerCapability::PersonDetect),
        "track": entry(WorkerCapability::PersonTrack),
        "segment": entry(WorkerCapability::PersonSegment),
        "replace": entry(WorkerCapability::PersonReplace),
        "detectPreview": entry(WorkerCapability::PersonDetectPreview),
        "trackPreview": entry(WorkerCapability::PersonTrackPreview),
    })
}

pub(crate) async fn person_capability_readiness(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let (sweep, workers) = store_call(state.clone(), move |store, timeout| {
        let sweep = store.mark_stale_workers_interrupted(timeout)?;
        let workers = store.list_workers();
        Ok((sweep, workers))
    })
    .await?;
    handle_stale_sweep(&state, &sweep);
    let workers = workers?;
    Ok(Json(
        json!({ "person": person_readiness_from_workers(&workers) }),
    ))
}

/// Mac UI gating capabilities (sc-3486): the master switch (`macGatingActive`, the
/// `SCENEWORKS_MLX_REQUIRED` rollout flag) plus every non-model capability surface the web client
/// may gate on Mac (LyCORIS, upscale, pose-from-photo, person detect/track, captioning, advanced
/// video, training kernels). Platform is the API host's OS, so a Docker/Windows/Linux client reads
/// `macGatingActive=false` and applies no gating at all. Per-model support rides on
/// `GET /api/v1/models` (`macSupport`); this endpoint carries the global/feature half.
pub(crate) async fn mac_capability_support(State(state): State<AppState>) -> Json<MacCapabilities> {
    Json(mac_capabilities(
        std::env::consts::OS,
        state.settings.mlx_required,
    ))
}

/// Host memory for remote-browser model gating (epic 4484 story 9). A remote browser
/// can't call the desktop-only Tauri `get_gpu_info`, so it reads the host's memory
/// here, derived from the registered GPU worker's reported utilization: the macOS MLX
/// worker reports unified memory (`sysctl hw.memsize`) as its total, and the Windows
/// candle worker reports discrete GPU VRAM. Auth-protected (not in `PUBLIC_PATHS`) and
/// leaks no paths/secrets — only aggregate memory totals + the platform string. The
/// desktop keeps using the richer Tauri probe; this serves the remote browser.
pub(crate) async fn host_capabilities(
    State(state): State<AppState>,
) -> Result<Json<HostCapabilitiesResponse>, ApiError> {
    let workers = store_call(state, move |store, _timeout| store.list_workers()).await?;
    let mb_to_gb = |mb: u64| (mb as f64) / 1024.0;
    // Unified memory: the macOS MLX worker reports sysctl hw.memsize as its total.
    let unified_memory_gb = workers
        .iter()
        .find(|worker| worker.gpu_id == "mlx")
        .and_then(|worker| worker.utilization.as_ref())
        .and_then(|util| util.memory_total_mb)
        .map(mb_to_gb);
    // GPU VRAM: largest total across the discrete GPU workers (Windows candle / CUDA).
    let gpu_memory_gb = workers
        .iter()
        .filter(|worker| worker.gpu_id != "mlx" && worker.gpu_id != "cpu")
        .filter_map(|worker| {
            worker
                .utilization
                .as_ref()
                .and_then(|util| util.memory_total_mb)
        })
        .max()
        .map(mb_to_gb);
    Ok(Json(HostCapabilitiesResponse {
        platform: std::env::consts::OS,
        unified_memory_gb,
        gpu_memory_gb,
    }))
}

/// Remote-admin GPU worker restart (epic 4484 story 12). The API process does NOT
/// supervise the GPU worker — the desktop shell does — so it can't kill it directly.
/// Instead it prints a unique sentinel to stdout that the desktop's existing
/// sidecar-output reader matches, then the desktop performs the same kill-and-respawn
/// as its local "Restart worker" button. In a server/Docker deployment (no desktop
/// supervisor reading stdout) the sentinel is just a log line and nothing restarts —
/// container workers are managed by their own supervisors. Auth-protected (not a public
/// path), so only an authenticated remote admin can trigger it.
pub(crate) async fn request_worker_restart() -> Json<Value> {
    // A plain stdout line (Rust stdout is line-buffered, so the trailing newline
    // flushes it) — not structured tracing — so the desktop reader matches it
    // unambiguously regardless of the active log format.
    println!("{}", sceneworks_core::WORKER_RESTART_SENTINEL);
    tracing::info!(
        event = "worker_restart_requested",
        "remote worker restart requested over REST"
    );
    Json(json!({ "ok": true }))
}

pub(crate) async fn register_worker(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<WorkerRegisterRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.register_worker(RegisterWorker {
            worker_id: payload.worker_id,
            gpu_id: payload.gpu_id,
            gpu_name: payload.gpu_name,
            capabilities: payload.capabilities,
            loaded_models: payload.loaded_models,
            utilization: payload.utilization,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    publish_queue(&state).await?;
    Ok(Json(worker))
}

pub(crate) async fn heartbeat_worker(
    State(state): State<AppState>,
    Path(worker_id): Path<String>,
    ApiJson(payload): ApiJson<WorkerHeartbeatRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let outcome = store_call(state.clone(), move |store, _timeout| {
        store.heartbeat_worker(WorkerHeartbeat {
            worker_id,
            status: payload.status,
            current_job_id: payload.current_job_id,
            loaded_models: payload.loaded_models,
            utilization: payload.utilization,
        })
    })
    .await?;
    // An idle heartbeat from a restarted worker terminates the job its previous
    // incarnation was running. Announce that job, exactly as `handle_stale_sweep`
    // does for the time-based sweep (sc-18182): the web client is SSE-driven and does
    // not poll jobs, so an unpublished terminal transition leaves the studio showing
    // "generating" forever even though the job is already terminal in the database.
    if let Some(job) = &outcome.interrupted_job {
        invalidate_model_catalog_for_terminal_jobs(&state, std::slice::from_ref(job));
        publish(&state, "job.updated", job);
        // Not `?`: the interrupt is already committed and `job.updated` already sent,
        // so a queue-snapshot failure must not turn this into a 500 that denies the
        // worker its snapshot and makes it retry a heartbeat it already delivered.
        // (`handle_stale_sweep` avoids the question entirely — it publishes only
        // `job.updated` and never refreshes the queue.)
        if let Err(error) = publish_queue(&state).await {
            tracing::warn!(
                event = "queue_publish_failed_after_heartbeat_interrupt",
                job_id = %job.id,
                error = ?error,
                "job.updated was published but the queue snapshot failed"
            );
        }
    }
    publish(&state, "worker.updated", &outcome.worker);
    Ok(Json(outcome.worker))
}

/// The supervisor reports a worker child that terminated abnormally — killed by an
/// uncatchable signal (SIGKILL/OOM, SIGABRT, SIGSEGV, …) or exited on its own with
/// a non-zero status (e.g. a Rust panic, exit code 101). We fail that worker's
/// still-active job with an attributed error instead of waiting for the heartbeat
/// sweep to mark it the generic `interrupted` — so the user sees a real, actionable
/// failure rather than a frozen progress bar (sc-4881 signals; sc-6320 non-signal
/// exits). Returns the failed job, if any.
pub(crate) async fn worker_terminated(
    State(state): State<AppState>,
    Path(worker_id): Path<String>,
    ApiJson(payload): ApiJson<WorkerTerminationRequest>,
) -> Result<Json<Option<JobSnapshot>>, ApiError> {
    let failed = store_call(state.clone(), move |store, _timeout| {
        store.fail_worker_job_terminated(&worker_id, payload.signal, payload.exit_code)
    })
    .await?;
    if let Some(job) = &failed {
        invalidate_model_catalog_for_terminal_jobs(&state, std::slice::from_ref(job));
        publish(&state, "job.updated", job);
        // Not `?` (sc-18182): the job is already committed as failed and `job.updated`
        // is already out, so a queue-snapshot error must not turn a SUCCESSFUL
        // attribution into a 500 — the desktop supervisor would then log "failed to
        // report worker termination" for a report that actually landed. This endpoint
        // only became reachable on macOS with sc-18182, which is what makes the
        // mislabelling reachable too.
        if let Err(error) = publish_queue(&state).await {
            tracing::warn!(
                event = "queue_publish_failed_after_worker_termination",
                job_id = %job.id,
                error = ?error,
                "job.updated was published but the queue snapshot failed"
            );
        }
    }
    Ok(Json(failed))
}
