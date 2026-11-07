use super::*;
use serde_json::json;

fn request() -> CreateJob {
    CreateJob {
        job_type: JobType::ModelDownload,
        project_id: None,
        project_name: None,
        payload: json!({"model":"example", "repo":"fixture/model", "files":["weights"]})
            .as_object()
            .unwrap()
            .clone(),
        requested_gpu: "auto".into(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

#[test]
fn portable_binding_is_private_atomic_and_gates_old_workers() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("jobs.db");
    let store = JobsStore::new(&path);
    store.initialize().unwrap();
    let binding = json!({"schemaVersion":1,"modelId":"example"});
    {
        let mut guard = store.lock.lock();
        let connection = store.write_connection(&mut guard).unwrap();
        connection.execute_batch("create trigger reject_portable_binding before update of portable_model_binding_json on jobs begin select raise(abort, 'injected binding failure'); end;").unwrap();
    }
    assert!(store
        .create_job_with_portable_binding(request(), None, None, Some(&binding))
        .is_err());
    assert!(store.list_jobs(None, None, 10).unwrap().is_empty());
    {
        let mut guard = store.lock.lock();
        store
            .write_connection(&mut guard)
            .unwrap()
            .execute_batch("drop trigger reject_portable_binding")
            .unwrap();
    }
    let created = store
        .create_job_with_portable_binding(request(), None, None, Some(&binding))
        .unwrap();
    assert!(!created.extra.contains_key("portableModelBinding"));
    assert!(!created.payload.contains_key("portableModelBinding"));
    assert_eq!(
        store.portable_model_binding(&created.id).unwrap(),
        Some(binding.clone())
    );
    let reopened = JobsStore::new(&path);
    reopened.initialize().unwrap();
    assert_eq!(
        reopened.portable_model_binding(&created.id).unwrap(),
        Some(binding.clone())
    );
    store
        .register_worker(RegisterWorker {
            worker_id: "worker".into(),
            gpu_id: "cpu".into(),
            gpu_name: None,
            capabilities: vec![crate::contracts::WorkerCapability::ModelDownload],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .unwrap();
    assert!(store
        .claim_next_job_routed_with_portable_binding("worker", false, 0)
        .unwrap()
        .0
        .is_none());
    let (claimed, _, private_binding) = store
        .claim_next_job_routed_with_portable_binding("worker", false, 1)
        .unwrap();
    assert_eq!(claimed.unwrap().id, created.id);
    assert_eq!(private_binding, Some(binding));
}

#[test]
fn retry_and_duplicate_cannot_replace_a_frozen_portable_binding() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let binding = json!({"schemaVersion":1,"modelId":"example"});
    let original = store
        .create_job_with_portable_binding(request(), None, None, Some(&binding))
        .unwrap();
    let changed = json!({"schemaVersion":1,"modelId":"other"});
    let count = store.list_jobs(None, None, 10).unwrap().len();
    assert!(store
        .retry_job_with_portable_binding(
            &original.id,
            RetryJob {
                payload_changes: Default::default()
            },
            None,
            None,
            Some(&changed)
        )
        .is_err());
    assert!(store
        .duplicate_job_with_portable_binding(
            None,
            &original.id,
            DuplicateJob {
                payload_changes: Default::default(),
                requested_gpu: None
            },
            None,
            None,
            Some(&changed)
        )
        .is_err());
    assert_eq!(store.list_jobs(None, None, 10).unwrap().len(), count);
    let retried = store
        .retry_job_with_portable_binding(
            &original.id,
            RetryJob {
                payload_changes: Default::default(),
            },
            None,
            None,
            Some(&binding),
        )
        .unwrap();
    assert_eq!(
        store.portable_model_binding(&retried.id).unwrap(),
        Some(binding)
    );
}
