//! Admission-time route probes use the same predicates as worker claim.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::contracts::{ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage};

use super::gaps::{job_is_any_candle_eligible, job_is_any_mlx_eligible};

/// Whether a native worker would claim this exact request shape. The API uses this before
/// freezing portable sources for a route; catalog support blocks describe any possible shape
/// for a model and are insufficient for an individual job.
pub fn native_route_claims_request(
    backend: &str,
    job_type: &JobType,
    payload: &Map<String, Value>,
) -> Result<bool, String> {
    let probe = JobSnapshot {
        id: "portable-route-probe".to_owned(),
        job_type: job_type.clone(),
        status: JobStatus::Queued,
        project_id: None,
        project_name: None,
        payload: payload.clone(),
        result: Map::new(),
        requested_gpu: "auto".to_owned(),
        assigned_gpu: None,
        worker_id: None,
        progress: ContractNumber::from(0_u64),
        stage: ProgressStage::Queued,
        message: String::new(),
        error: None,
        eta_seconds: None,
        elapsed_seconds: None,
        attempts: 0,
        source_job_id: None,
        duplicate_of_job_id: None,
        cancel_requested: false,
        created_at: String::new(),
        updated_at: String::new(),
        started_at: None,
        completed_at: None,
        canceled_at: None,
        last_heartbeat_at: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: None,
        title: None,
        extra: BTreeMap::new(),
    };
    match backend {
        "mlx" => Ok(job_is_any_mlx_eligible(&probe)),
        "candle" => Ok(job_is_any_candle_eligible(&probe)),
        other => Err(format!("unknown native backend {other:?}")),
    }
}
