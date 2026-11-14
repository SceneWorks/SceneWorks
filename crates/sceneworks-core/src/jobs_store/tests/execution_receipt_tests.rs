use super::*;
use crate::execution_receipt::{OutputExecutionBinding, WorkerExecutionReceipt};
use crate::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, PortableJobBinding, PortableModelBinding,
    PortableRouteBinding,
};
use serde_json::json;
use std::collections::BTreeMap;

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
            execution: None,
            artifact_ids: vec![job_id.into()],
            artifact_sha256: Vec::new(),
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

#[test]
fn legacy_completion_does_not_gain_exact_replay_from_retry_or_duplicate() {
    let (_temp, store, job) = fixture();
    store.claim_next_job("worker-one").unwrap().unwrap();
    store
        .update_job_progress(&job.id, completed_update(&job.id))
        .unwrap();
    let refusal = store.create_exact_replay_job(&job.id, 60).unwrap_err();
    assert!(matches!(
        refusal,
        JobsStoreError::ReplayUnavailable(reason) if reason.reason == "receipt_missing"
    ));
    assert!(store
        .completed_execution_receipt(&job.id)
        .unwrap()
        .is_none());
}

#[test]
fn deterministic_image_receipt_creates_separate_pinned_replay_job() {
    let (_temp, store, _) = fixture();
    store
        .register_worker_with_platform(
            RegisterWorker {
                worker_id: "mlx-worker".into(),
                gpu_id: "mlx".into(),
                gpu_name: None,
                capabilities: vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate],
                loaded_models: vec![],
                utilization: None,
            },
            Some(std::env::consts::OS),
        )
        .unwrap();
    let original_binding: PortableJobBinding =
        serde_json::from_value(super::portable_binding_tests::binding()).unwrap();
    let authored = json!({"model":"sdxl"});
    let model = PortableModelBinding::new(
        "modelManifestEntry",
        "sdxl",
        "mlx",
        "default",
        None,
        canonical_authored_options_sha256(&authored).unwrap(),
        original_binding.routes.values().next().unwrap().models[0]
            .members
            .clone(),
    )
    .unwrap();
    let route = PortableRouteBinding::new("mlx", std::env::consts::OS, vec![model]).unwrap();
    let binding = serde_json::to_value(
        PortableJobBinding::new(authored, BTreeMap::from([(route.key(), route.clone())])).unwrap(),
    )
    .unwrap();
    let source = store
        .create_job_with_portable_binding(
            CreateJob {
                job_type: JobType::ImageGenerate,
                project_id: None,
                project_name: None,
                payload: json!({"model":"sdxl","mode":"text_to_image","seed":41,"count":1})
                    .as_object()
                    .unwrap()
                    .clone(),
                requested_gpu: "mlx".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            },
            None,
            None,
            Some(&binding),
        )
        .unwrap();
    let expected = crate::execution_receipt::TrustedExecutionReceipt {
        version: 1,
        job_id: source.id.clone(),
        job_type: source.job_type.as_str().into(),
        worker_id: "mlx-worker".into(),
        attempt_id: "e".repeat(64),
        accepted_backend: "mlx".into(),
        accepted_device: "mlx".into(),
        accepted_source_route: Some(serde_json::to_value(&route).unwrap()),
        inference_revision: crate::inference_provenance::INFERENCE_REVISION.into(),
        accepted_binding_sha256: "a".repeat(64),
        accepted_result_sha256: "b".repeat(64),
        execution: json!({"state":"resolved","identity":{
            "version":1,"provider_id":"sdxl","backend":"mlx","runtime_id":"pinned-fixture",
            "output_units":1,
            "deterministic_replay":{"kind":"deterministic","scope":"same_device_pinned_runtime"},
            "randomness":{"kind":"draws","scheme":"cpu_std_rng","draws":[
                {"output_index":0,"purpose":"initial_noise","seed":41,"noise_sha256":null}
            ]}
        }}),
        outputs: vec![OutputExecutionBinding {
            output_index: 0,
            execution: None,
            artifact_ids: vec!["asset-one".into()],
            artifact_sha256: vec!["c".repeat(64)],
        }],
    };
    store
        .open_connection()
        .unwrap()
        .execute(
            "update jobs set status = 'completed', stage = 'completed', worker_id = 'mlx-worker',
                    assigned_gpu = 'mlx', execution_claim_binding_sha256 = ?1,
                    execution_receipt_json = ?2 where id = ?3",
            params![
                expected.accepted_binding_sha256,
                serde_json::to_string(&expected).unwrap(),
                source.id
            ],
        )
        .unwrap();
    let replay = store.create_exact_replay_job(&source.id, 60).unwrap();
    assert_ne!(replay.id, source.id);
    assert_eq!(replay.status, JobStatus::Queued);
    assert_eq!(replay.requested_gpu, "mlx");
    assert_eq!(replay.payload, source.payload);
    assert_eq!(
        store.portable_model_binding(&replay.id).unwrap(),
        Some(binding.clone())
    );
    let expected_json: String = store
        .open_connection()
        .unwrap()
        .query_row(
            "select exact_replay_expected_json from jobs where id = ?1",
            params![replay.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&expected_json).unwrap()["jobId"],
        source.id
    );
    let (claimed, _, _) = store
        .claim_next_job_routed_with_portable_binding_authenticated(
            "mlx-worker",
            false,
            crate::model_artifacts::portable_binding::PORTABLE_MODEL_BINDING_VERSION,
            Some("mlx"),
            Some(std::env::consts::OS),
            true,
        )
        .unwrap();
    assert_eq!(claimed.unwrap().id, replay.id);
    let secret = store
        .execution_claim_capability(&replay.id, "mlx-worker")
        .unwrap()
        .unwrap();
    let claimed_snapshot = store.get_job(&replay.id).unwrap();
    let selected_route = Some(serde_json::to_value(&route).unwrap());
    let reconstructed = execution_binding_sha256(
        &claimed_snapshot,
        claimed_snapshot.assigned_gpu.as_deref().unwrap(),
        &Some(binding.clone()),
        "mlx",
        &selected_route,
    )
    .unwrap();
    let frozen: String = store
        .open_connection()
        .unwrap()
        .query_row(
            "select execution_claim_binding_sha256 from jobs where id = ?1",
            params![replay.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(reconstructed, frozen);
    let completed = WorkerExecutionReceipt {
        version: 1,
        execution: expected.execution,
        outputs: vec![OutputExecutionBinding {
            output_index: 0,
            execution: None,
            artifact_ids: vec![replay.id.clone()],
            artifact_sha256: vec!["c".repeat(64)],
        }],
    };
    let mut changed_bytes = completed.clone();
    changed_bytes.outputs[0].artifact_sha256 = vec!["d".repeat(64)];
    let refused = store.update_job_progress_with_execution_receipt(
        &replay.id,
        ProgressUpdate {
            backend: Some("mlx".into()),
            worker_id: Some("mlx-worker".into()),
            ..completed_update(&replay.id)
        },
        Some(&changed_bytes),
        Some(&secret),
    );
    assert!(matches!(
        refused,
        Err(JobsStoreError::ReplayUnavailable(reason)) if reason.reason == "completed_execution_changed"
    ));
    assert_eq!(
        store.get_job(&replay.id).unwrap().status,
        JobStatus::Preparing
    );
    let outcome = store
        .update_job_progress_with_execution_receipt(
            &replay.id,
            ProgressUpdate {
                backend: Some("mlx".into()),
                worker_id: Some("mlx-worker".into()),
                ..completed_update(&replay.id)
            },
            Some(&completed),
            Some(&secret),
        )
        .unwrap();
    assert_eq!(outcome.job.status, JobStatus::Completed);
    let changed = store.retry_job_with_portable_binding(
        &replay.id,
        RetryJob {
            payload_changes: json!({"seed":42}).as_object().unwrap().clone(),
        },
        None,
        None,
        Some(&binding),
    );
    assert!(matches!(
        changed,
        Err(JobsStoreError::ReplayUnavailable(reason)) if reason.reason == "replay_payload_changed"
    ));
    let retry = store
        .retry_job_with_portable_binding(
            &replay.id,
            RetryJob {
                payload_changes: Map::new(),
            },
            None,
            None,
            Some(&binding),
        )
        .unwrap();
    let retry_expected: String = store
        .open_connection()
        .unwrap()
        .query_row(
            "select exact_replay_expected_json from jobs where id = ?1",
            params![retry.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(retry_expected, expected_json);
}
