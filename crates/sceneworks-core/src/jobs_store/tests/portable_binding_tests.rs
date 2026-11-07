use super::*;
use crate::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, PortableArtifactFile, PortableArtifactMember,
    PortableJobBinding, PortableModelBinding, PortableRouteBinding, PortableSourceIdentity,
};
use serde_json::json;
use std::collections::BTreeMap;

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

fn binding() -> Value {
    let platform = std::env::consts::OS;
    let authored = json!({"model":"example"});
    let digest = canonical_authored_options_sha256(&authored).unwrap();
    let member = PortableArtifactMember {
        role: crate::model_artifacts::ArtifactMemberRole::Primary,
        component_id: None,
        source: PortableSourceIdentity::Revision {
            identity: crate::model_artifacts::ArtifactIdentity::pinned(
                "fixture/model",
                "1".repeat(40),
                "default",
            )
            .unwrap(),
        },
        tier: None,
        files: vec![PortableArtifactFile {
            relative_path: "weights.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }],
    };
    let model = PortableModelBinding::new(
        "modelManifestEntry",
        "example",
        "cpu",
        "default",
        None,
        digest,
        vec![member],
    )
    .unwrap();
    let route = PortableRouteBinding::new("cpu", platform, vec![model]).unwrap();
    serde_json::to_value(
        PortableJobBinding::new(authored, BTreeMap::from([(route.key(), route)])).unwrap(),
    )
    .unwrap()
}

#[test]
fn portable_binding_is_private_atomic_and_gates_old_workers() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("jobs.db");
    let store = JobsStore::new(&path);
    store.initialize().unwrap();
    let binding = binding();
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
    let fixed = store
        .create_job_with_id_and_portable_binding("job_fixed".into(), request(), Some(&binding))
        .unwrap();
    assert_eq!(fixed.id, "job_fixed");
    assert_eq!(
        store.portable_model_binding(&fixed.id).unwrap(),
        Some(binding.clone())
    );
    store
        .register_worker_with_platform(
            RegisterWorker {
                worker_id: "worker".into(),
                gpu_id: "cpu".into(),
                gpu_name: None,
                capabilities: vec![crate::contracts::WorkerCapability::ModelDownload],
                loaded_models: Vec::new(),
                utilization: None,
            },
            Some(std::env::consts::OS),
        )
        .unwrap();
    assert_eq!(
        store.get_worker("worker").unwrap().platform.as_deref(),
        Some(std::env::consts::OS)
    );
    assert!(store
        .claim_next_job_routed_with_portable_binding("worker", false, 0, None, None)
        .unwrap()
        .0
        .is_none());
    assert!(store
        .claim_next_job_routed_with_portable_binding("worker", false, 1, Some("cpu"), Some("wrong"))
        .unwrap()
        .0
        .is_none());
    {
        let mut guard = store.lock.lock();
        let connection = store.write_connection(&mut guard).unwrap();
        let mut corrupted = binding.clone();
        corrupted["routesSha256"] = json!(format!("sha256:{}", "0".repeat(64)));
        connection
            .execute(
                "update jobs set portable_model_binding_json = ?1 where id = ?2",
                params![dumps(&corrupted).unwrap(), created.id],
            )
            .unwrap();
    }
    assert!(store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            1,
            Some("cpu"),
            Some(std::env::consts::OS)
        )
        .is_err());
    assert_eq!(
        store.get_job(&created.id).unwrap().status,
        JobStatus::Queued
    );
    {
        let mut guard = store.lock.lock();
        store
            .write_connection(&mut guard)
            .unwrap()
            .execute(
                "update jobs set portable_model_binding_json = ?1 where id = ?2",
                params![dumps(&binding).unwrap(), created.id],
            )
            .unwrap();
    }
    let (claimed, _, private_binding) = store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            1,
            Some("cpu"),
            Some(std::env::consts::OS),
        )
        .unwrap();
    assert_eq!(claimed.unwrap().id, created.id);
    assert_eq!(private_binding, Some(binding));
}

#[test]
fn retry_and_duplicate_cannot_replace_a_frozen_portable_binding() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let binding = binding();
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
