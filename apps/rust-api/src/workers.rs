use super::*;

pub(crate) async fn queue_summary(
    State(state): State<AppState>,
) -> Result<Json<QueueSummary>, ApiError> {
    Ok(Json(queue_summary_snapshot(state).await?))
}

pub(crate) async fn list_workers(
    State(state): State<AppState>,
) -> Result<Json<Vec<WorkerSnapshot>>, ApiError> {
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_workers()
        })
        .await?,
    ))
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
    let workers = store_call(state, move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.list_workers()
    })
    .await?;
    Ok(Json(
        json!({ "person": person_readiness_from_workers(&workers) }),
    ))
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
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.heartbeat_worker(WorkerHeartbeat {
            worker_id,
            status: payload.status,
            current_job_id: payload.current_job_id,
            loaded_models: payload.loaded_models,
            utilization: payload.utilization,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    Ok(Json(worker))
}
