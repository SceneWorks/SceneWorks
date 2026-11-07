use super::*;
use crate::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, PortableArtifactFile, PortableArtifactMember,
    PortableJobBinding, PortableModelBinding, PortableRouteBinding, PortableSourceClaim,
    PortableSourceIdentity, PortableSourceProvenance,
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

#[test]
fn worker_source_claims_are_private_fresh_and_explicitly_cleared() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let claim = PortableSourceClaim::new(
        "ltx/encoder",
        PortableSourceProvenance::OperatorOverride,
        vec![PortableArtifactFile {
            relative_path: "encoder.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }],
    )
    .unwrap();
    let worker = store
        .register_worker_with_portable_sources(
            RegisterWorker {
                worker_id: "worker".into(),
                gpu_id: "cpu".into(),
                gpu_name: None,
                capabilities: vec![],
                loaded_models: vec![],
                utilization: None,
            },
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&claim)),
        )
        .unwrap();
    assert!(!serde_json::to_string(&worker)
        .unwrap()
        .contains("ltx/encoder"));
    assert_eq!(
        store
            .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
            .unwrap(),
        vec![claim.clone()]
    );
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker".into(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            loaded_models: vec![],
            utilization: None,
            status_reason: None,
        })
        .unwrap();
    assert_eq!(
        store
            .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
            .unwrap(),
        vec![claim.clone()]
    );
    store
        .heartbeat_worker_with_portable_sources(
            WorkerHeartbeat {
                worker_id: "worker".into(),
                status: WorkerStatus::Idle,
                current_job_id: None,
                loaded_models: vec![],
                utilization: None,
                status_reason: None,
            },
            Some(&[]),
        )
        .unwrap();
    assert!(store
        .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
        .unwrap()
        .is_empty());
}

#[test]
fn worker_claim_member_requires_exact_current_worker_proof_before_claim() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let original: PortableJobBinding = serde_json::from_value(binding()).unwrap();
    let mut routes = original.routes.clone();
    let member = &mut routes.values_mut().next().unwrap().models[0].members[0];
    let claim = PortableSourceClaim::new(
        "aux/encoder",
        PortableSourceProvenance::OperatorOverride,
        member.files.clone(),
    )
    .unwrap();
    member.source = PortableSourceIdentity::WorkerClaim {
        source_id: claim.source_id.clone(),
        provenance: claim.provenance.clone(),
        closure_sha256: claim.closure_sha256.clone(),
    };
    let binding =
        serde_json::to_value(PortableJobBinding::new(original.authored_options, routes).unwrap())
            .unwrap();
    let job = store
        .create_job_with_portable_binding(request(), None, None, Some(&binding))
        .unwrap();
    let register = || RegisterWorker {
        worker_id: "worker".into(),
        gpu_id: "cpu".into(),
        gpu_name: None,
        capabilities: vec![crate::contracts::WorkerCapability::ModelDownload],
        loaded_models: vec![],
        utilization: None,
    };
    store
        .register_worker_with_portable_sources(register(), Some(std::env::consts::OS), None)
        .unwrap();
    assert!(store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            1,
            Some("cpu"),
            Some(std::env::consts::OS)
        )
        .unwrap()
        .0
        .is_none());
    let mut wrong = claim.clone();
    wrong.files[0].sha256 = format!("sha256:{}", "b".repeat(64));
    wrong = PortableSourceClaim::new(wrong.source_id, wrong.provenance, wrong.files).unwrap();
    store
        .register_worker_with_portable_sources(
            register(),
            Some(std::env::consts::OS),
            Some(&[wrong]),
        )
        .unwrap();
    assert!(store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            1,
            Some("cpu"),
            Some(std::env::consts::OS)
        )
        .unwrap()
        .0
        .is_none());
    store
        .register_worker_with_portable_sources(
            register(),
            Some(std::env::consts::OS),
            Some(&[claim]),
        )
        .unwrap();
    let claimed = store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            1,
            Some("cpu"),
            Some(std::env::consts::OS),
        )
        .unwrap()
        .0
        .unwrap();
    assert_eq!(claimed.id, job.id);
}
