//! Admission-time route probes use the same predicates as worker claim.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::contracts::{
    ContractNumber, JobSnapshot, JobStatus, JobType, ProgressStage, WorkerCapability,
    WorkerSnapshot, WorkerStatus,
};

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
    let gpu_id = match backend {
        "mlx" => "mlx",
        "candle" => "0",
        other => return Err(format!("unknown native backend {other:?}")),
    };
    // Candle claims unsupported pose shapes only to reject them with a typed error.
    // That is not a render route and must not authorize a portable source closure.
    if backend == "candle"
        && matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit)
        && !super::gaps::job_is_any_candle_eligible(&probe)
    {
        return Ok(false);
    }
    let required = super::super::required_capability(&probe);
    let required_capability: WorkerCapability =
        serde_json::from_value(Value::String(required.to_owned()))
            .map_err(|error| error.to_string())?;
    let mut capabilities = vec![
        WorkerCapability::Gpu,
        required_capability,
        WorkerCapability::LoraTrainExecute,
    ];
    if backend == "candle" {
        capabilities.push(WorkerCapability::Unknown("candle".to_owned()));
    }
    let worker = WorkerSnapshot {
        id: "portable-route-worker-probe".to_owned(),
        gpu_id: gpu_id.to_owned(),
        platform: None,
        gpu_name: None,
        status: WorkerStatus::Idle,
        current_job_id: None,
        capabilities,
        loaded_models: Vec::new(),
        utilization: None,
        status_reason: None,
        registered_at: String::new(),
        last_seen_at: String::new(),
        extra: BTreeMap::new(),
    };
    Ok(super::super::worker_supports_job(&worker, &probe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn capability_only_catalog_analysis_uses_real_worker_claim_policy() {
        let payload = json!({"catalogId":"catalog"});
        for backend in ["mlx", "candle"] {
            assert!(native_route_claims_request(
                backend,
                &JobType::CatalogAnalysis,
                payload.as_object().unwrap()
            )
            .unwrap());
        }
    }

    #[test]
    fn unsupported_conditioned_image_remains_unclaimable() {
        let payload = json!({"model":"chroma","advanced":{"poses":[{}]}});
        assert!(!native_route_claims_request(
            "mlx",
            &JobType::ImageGenerate,
            payload.as_object().unwrap()
        )
        .unwrap());
        assert!(!native_route_claims_request(
            "candle",
            &JobType::ImageGenerate,
            payload.as_object().unwrap()
        )
        .unwrap());
    }
}
