use super::*;
use crate::execution_receipt::{OutputExecutionBinding, WorkerExecutionReceipt};
use serde_json::json;

fn fixture() -> (tempfile::TempDir, JobsStore, JobSnapshot) {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.sqlite"));
    store.initialize().unwrap();
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-one".into(),
            gpu_id: "cpu".into(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::Placeholder],
            loaded_models: vec![],
            utilization: None,
        })
        .unwrap();
    let job = store
        .create_job(CreateJob {
            job_type: JobType::Placeholder,
            project_id: None,
            project_name: None,
            payload: json!({}).as_object().unwrap().clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    (temp, store, job)
}

fn completed_update(job_id: &str) -> ProgressUpdate {
    ProgressUpdate {
        status: JobStatus::Completed,
        stage: ProgressStage::Completed,
        progress: 1.0,
        message: "done".into(),
        error: None,
        result: Some(
            json!({"answer":"ok","assetWrites":[{"assetId":job_id}]})
                .as_object()
                .unwrap()
                .clone(),
        ),
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some("cpu".into()),
        worker_id: Some("worker-one".into()),
    }
}

fn report(job_id: &str) -> WorkerExecutionReceipt {
    WorkerExecutionReceipt {
        version: 1,
        execution: json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable", "detail":"fixture", "missing_facts":["provider_resolved_execution"]
        }}),
        outputs: vec![OutputExecutionBinding {
            output_index: 0,
            artifact_ids: vec![job_id.into()],
        }],
    }
}

#[test]
fn private_attempt_capability_gates_atomic_terminal_receipt() {
    let (_temp, store, job) = fixture();
    let (claimed, _, _) = store
        .claim_next_job_routed_with_portable_binding_authenticated(
            "worker-one",
            false,
            0,
            None,
            None,
            true,
        )
        .unwrap();
    assert_eq!(claimed.unwrap().id, job.id);
    let secret = store
        .execution_claim_capability(&job.id, "worker-one")
        .unwrap()
        .unwrap();
    assert!(store
        .execution_claim_capability(&job.id, "worker-two")
        .unwrap()
        .is_none());
    assert!(!serde_json::to_string(&store.get_job(&job.id).unwrap())
        .unwrap()
        .contains(&secret));

    // Even omitting the report cannot let a caller who knows the worker ID
    // consume the terminal slot before the worker publishes its facts.
    let missing_secret = store.update_job_progress_with_execution_receipt(
        &job.id,
        completed_update(&job.id),
        None,
        None,
    );
    assert!(matches!(
        missing_secret,
        Err(JobsStoreError::NotJobOwner { .. })
    ));

    let invalid = store.update_job_progress_with_execution_receipt(
        &job.id,
        completed_update(&job.id),
        Some(&report(&job.id)),
        Some("client-spoof"),
    );
    assert!(matches!(invalid, Err(JobsStoreError::NotJobOwner { .. })));
    assert_eq!(store.get_job(&job.id).unwrap().status, JobStatus::Preparing);

    let accepted = store
        .update_job_progress_with_execution_receipt(
            &job.id,
            completed_update(&job.id),
            Some(&report(&job.id)),
            Some(&secret),
        )
        .unwrap();
    assert!(accepted.applied);
    let receipt = store.completed_execution_receipt(&job.id).unwrap();
    assert!(
        receipt.is_none(),
        "durable terminal side effects must finish first"
    );
    let stored: String = store
        .open_connection()
        .unwrap()
        .query_row(
            "select execution_receipt_json from jobs where id = ?1",
            params![job.id],
            |row| row.get(0),
        )
        .unwrap();
    assert!(stored.contains("acceptedSourceRoute"));
    assert!(!stored.contains(&secret));
}

#[test]
fn changed_accepted_payload_refuses_receipt_without_completing_job() {
    let (_temp, store, job) = fixture();
    store
        .claim_next_job_routed_with_portable_binding_authenticated(
            "worker-one",
            false,
            0,
            None,
            None,
            true,
        )
        .unwrap();
    let secret = store
        .execution_claim_capability(&job.id, "worker-one")
        .unwrap()
        .unwrap();
    store
        .open_connection()
        .unwrap()
        .execute(
            "update jobs set payload_json = ?1 where id = ?2",
            params![r#"{"changed":true}"#, job.id],
        )
        .unwrap();
    let result = store.update_job_progress_with_execution_receipt(
        &job.id,
        completed_update(&job.id),
        Some(&report(&job.id)),
        Some(&secret),
    );
    assert!(matches!(result, Err(JobsStoreError::InvalidResultItem(_))));
    assert_eq!(store.get_job(&job.id).unwrap().status, JobStatus::Preparing);
}
