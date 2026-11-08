use super::*;
use crate::model_artifacts::portable_binding::{
    canonical_authored_options_sha256, PortableArtifactFile, PortableArtifactMember,
    PortableJobBinding, PortableModelBinding, PortableRouteBinding, PortableSourceClaim,
    PortableSourceIdentity, PortableSourceProvenance, PORTABLE_MODEL_BINDING_VERSION,
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
fn private_portable_binding_roundtrips_nested_plan_context_for_claim() {
    use crate::checkpoint_import::{
        CheckpointContainerV1, ImportLayerV1, ImportPlanV1, SourceLocatorV1,
    };
    use crate::model_artifacts::portable_binding::PortableCheckpointPlan;

    let mut plan = ImportPlanV1::new(
        "plan-fixture",
        "example",
        vec![ImportLayerV1 {
            layer_id: "artifact:weights.safetensors".into(),
            role: "model".into(),
            target_path: "weights.safetensors".into(),
            container: CheckpointContainerV1::Safetensors,
            source: SourceLocatorV1::linked("approved-root", "weights.safetensors", &"a".repeat(64))
                .unwrap(),
        }],
    )
    .unwrap();
    plan.context
        .insert("declaration".into(), json!({"z":1,"a":2}));
    let portable = PortableCheckpointPlan::from_import_plan(
        &plan,
        Some("artifact:weights.safetensors".into()),
    )
    .unwrap();
    let mut envelope: PortableJobBinding = serde_json::from_value(binding()).unwrap();
    let mut routes = envelope.routes.clone();
    let model = &mut routes.values_mut().next().unwrap().models[0];
    model.members[0].files[0].relative_path = "artifact".into();
    model.members[0].source = PortableSourceIdentity::Digest {
        sha256: format!("sha256:{}", "a".repeat(64)),
    };
    model.checkpoint_plan = Some(portable);
    envelope = PortableJobBinding::new(envelope.authored_options, routes).unwrap();

    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let mut model_job = request();
    model_job.job_type = JobType::ImageGenerate;
    let job = store
        .create_job_with_portable_binding(
            model_job,
            None,
            None,
            Some(&serde_json::to_value(envelope).unwrap()),
        )
        .unwrap();
    let stored: PortableJobBinding =
        serde_json::from_value(store.portable_model_binding(&job.id).unwrap().unwrap()).unwrap();
    stored.validate().unwrap();
    let model = &stored.routes.values().next().unwrap().models[0];
    let claim = PortableSourceClaim::new(
        crate::model_artifacts::portable_binding::registered_digest_member_claim_id(
            model,
            &model.members[0],
            &Value::Object(request().payload),
        )
        .unwrap(),
        PortableSourceProvenance::OperatorOverride,
        model.members[0].files.clone(),
    )
    .unwrap();
    let connection = store.open_connection().unwrap();
    assert!(candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        Some(("cpu", std::env::consts::OS)),
        &[claim],
        false,
    )
    .unwrap());
}

#[test]
fn historical_unbound_model_jobs_cannot_reach_versionless_claims() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let mut image = request();
    image.job_type = JobType::ImageGenerate;
    image.payload = json!({"model":"example", "modelManifestEntry":{"id":"example"}})
        .as_object()
        .unwrap()
        .clone();
    let image = store.create_job(image).unwrap();
    let download = store.create_job(request()).unwrap();
    let mut training = request();
    training.job_type = JobType::LoraTrain;
    training.payload = json!({"model":"example"}).as_object().unwrap().clone();
    let training = store.create_job(training).unwrap();
    let connection = store.open_connection().unwrap();
    assert!(
        !candidate_matches_portable_worker_sources(&connection, &image.id, None, &[], false,)
            .unwrap()
    );
    assert!(
        candidate_matches_portable_worker_sources(&connection, &download.id, None, &[], false,)
            .unwrap()
    );
    let queued = store.unbound_queued_model_jobs(8).unwrap();
    assert_eq!(
        queued.iter().map(|job| &job.id).collect::<Vec<_>>(),
        vec![&image.id, &training.id]
    );
}

#[test]
fn historical_binding_install_is_atomic_and_refuses_a_changed_job() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let mut request = request();
    request.job_type = JobType::ImageGenerate;
    request.payload = json!({"model":"example", "modelManifestEntry":{"id":"example"}})
        .as_object()
        .unwrap()
        .clone();
    let original = store.create_job(request).unwrap();
    let mut canonical = original.payload.clone();
    canonical.insert("prompt".into(), json!("canonical historical prompt"));
    let frozen = binding();
    assert!(store
        .bind_historical_queued_job(&original, &canonical, &frozen)
        .unwrap());
    assert_eq!(
        store.portable_model_binding(&original.id).unwrap(),
        Some(frozen)
    );
    assert_eq!(store.get_job(&original.id).unwrap().payload, canonical);
    let connection = store.open_connection().unwrap();
    assert!(candidate_matches_portable_worker_sources(
        &connection,
        &original.id,
        Some(("cpu", std::env::consts::OS)),
        &[],
        false,
    )
    .unwrap());
    assert!(!candidate_matches_portable_worker_sources(
        &connection,
        &original.id,
        None,
        &[],
        false,
    )
    .unwrap());
    assert!(!store
        .bind_historical_queued_job(&original, &original.payload, &binding())
        .unwrap());
}

#[test]
fn unauthenticated_historical_carrier_fails_before_claim() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let mut request = request();
    request.job_type = JobType::ImageGenerate;
    request.payload = json!({"model":"example", "modelManifestEntry":{"id":"example"}})
        .as_object()
        .unwrap()
        .clone();
    let original = store.create_job(request).unwrap();
    let failed = store
        .fail_historical_unbound_queued_job(&original)
        .unwrap()
        .unwrap();
    assert_eq!(failed.status, JobStatus::Failed);
    assert!(store.unbound_queued_model_jobs(8).unwrap().is_empty());
    assert!(store
        .fail_historical_unbound_queued_job(&original)
        .unwrap()
        .is_none());
}

#[test]
fn unavailable_historical_source_rotates_without_starving_a_later_job() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let mut request = request();
    request.job_type = JobType::ImageGenerate;
    request.payload = json!({"model":"example", "modelManifestEntry":{"id":"example"}})
        .as_object()
        .unwrap()
        .clone();
    let first = store.create_job(request.clone()).unwrap();
    let second = store.create_job(request).unwrap();
    assert_eq!(store.unbound_queued_model_jobs(1).unwrap()[0].id, first.id);
    store.defer_historical_queued_job(&first).unwrap();
    assert_eq!(store.unbound_queued_model_jobs(1).unwrap()[0].id, second.id);
}

#[test]
fn claim_requires_one_worker_with_primary_digest_and_auxiliary_claim() {
    use crate::model_artifacts::portable_binding::{
        portable_closure_sha256, registered_digest_member_claim_id,
    };

    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let original: PortableJobBinding = serde_json::from_value(binding()).unwrap();
    let mut routes = original.routes.clone();
    let route = routes.values_mut().next().unwrap();
    let model = &mut route.models[0];
    model.members[0].source = PortableSourceIdentity::Digest {
        sha256: portable_closure_sha256(&model.members[0].files).unwrap(),
    };
    let digest_claim = PortableSourceClaim::new(
        registered_digest_member_claim_id(
            model,
            &model.members[0],
            &Value::Object(request().payload),
        )
        .unwrap(),
        PortableSourceProvenance::OperatorOverride,
        model.members[0].files.clone(),
    )
    .unwrap();
    let auxiliary_claim = PortableSourceClaim::new(
        "aux/encoder",
        PortableSourceProvenance::OperatorOverride,
        vec![PortableArtifactFile {
            relative_path: "encoder.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "b".repeat(64)),
        }],
    )
    .unwrap();
    model.members.push(PortableArtifactMember {
        role: crate::model_artifacts::ArtifactMemberRole::RequiredComponent,
        component_id: Some("encoder".into()),
        source: PortableSourceIdentity::WorkerClaim {
            source_id: auxiliary_claim.source_id.clone(),
            provenance: auxiliary_claim.provenance.clone(),
            closure_sha256: auxiliary_claim.closure_sha256.clone(),
        },
        tier: None,
        files: auxiliary_claim.files.clone(),
    });
    let frozen =
        serde_json::to_value(PortableJobBinding::new(original.authored_options, routes).unwrap())
            .unwrap();
    let mut legacy = request();
    legacy.job_type = JobType::ImageGenerate;
    let job = store.create_job(legacy).unwrap();
    let original = store.get_job(&job.id).unwrap();
    assert!(store
        .bind_historical_queued_job(&original, &original.payload, &frozen)
        .unwrap());
    let connection = store.open_connection().unwrap();
    let route = Some(("cpu", std::env::consts::OS));
    assert!(
        !candidate_matches_portable_worker_sources(&connection, &job.id, route, &[], false)
            .unwrap()
    );
    assert!(!candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        route,
        std::slice::from_ref(&digest_claim),
        false,
    )
    .unwrap());
    assert!(!candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        route,
        std::slice::from_ref(&auxiliary_claim),
        false,
    )
    .unwrap());
    assert!(candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        route,
        &[digest_claim, auxiliary_claim],
        false,
    )
    .unwrap());
}

#[test]
fn local_utility_digest_requires_keyed_worker_even_when_raw_kind_is_missing_or_spoofed() {
    use crate::model_artifacts::portable_binding::{
        portable_closure_sha256, registered_digest_member_claim_id,
    };

    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let original: PortableJobBinding = serde_json::from_value(binding()).unwrap();
    let mut routes = original.routes.clone();
    let route = routes.values_mut().next().unwrap();
    let model = &mut route.models[0];
    model.slot = "modelManifestEntries/0".into();
    model.members[0].source = PortableSourceIdentity::Digest {
        sha256: portable_closure_sha256(&model.members[0].files).unwrap(),
    };
    let mut create = request();
    create.job_type = JobType::DatasetAnalysis;
    create
        .payload
        .insert("modelManifestEntries".into(), json!([{"id":"example"}]));
    create
        .payload
        .insert("modelNameOrPath".into(), json!("/managed/selected"));
    let claim = PortableSourceClaim::new(
        registered_digest_member_claim_id(model, &model.members[0], &json!(create.payload))
            .unwrap(),
        PortableSourceProvenance::OperatorOverride,
        model.members[0].files.clone(),
    )
    .unwrap();
    let mut mlx_model = model.clone();
    mlx_model.backend = "mlx".into();
    let mlx_route =
        PortableRouteBinding::new("mlx", std::env::consts::OS, vec![mlx_model]).unwrap();
    routes.insert(mlx_route.key(), mlx_route);
    let frozen =
        serde_json::to_value(PortableJobBinding::new(original.authored_options, routes).unwrap())
            .unwrap();
    let job = store
        .create_job_with_portable_binding(create, None, None, Some(&frozen))
        .unwrap();
    for worker_id in ["unkeyed", "keyed"] {
        store
            .register_worker_with_portable_sources(
                RegisterWorker {
                    worker_id: worker_id.into(),
                    gpu_id: "cpu".into(),
                    gpu_name: None,
                    capabilities: vec![crate::contracts::WorkerCapability::DatasetAnalysis],
                    loaded_models: Vec::new(),
                    utilization: None,
                },
                Some(std::env::consts::OS),
                Some(std::slice::from_ref(&claim)),
            )
            .unwrap();
    }
    store
        .register_worker_with_portable_sources(
            RegisterWorker {
                worker_id: "mlx-alternative".into(),
                gpu_id: "mlx".into(),
                gpu_name: None,
                capabilities: vec![crate::contracts::WorkerCapability::DatasetAnalysis],
                loaded_models: Vec::new(),
                utilization: None,
            },
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&claim)),
        )
        .unwrap();
    let connection = store.open_connection().unwrap();
    let route = Some(("cpu", std::env::consts::OS));
    assert!(!candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        route,
        std::slice::from_ref(&claim),
        false,
    )
    .unwrap());
    assert!(candidate_matches_portable_worker_sources(
        &connection,
        &job.id,
        route,
        std::slice::from_ref(&claim),
        true,
    )
    .unwrap());
    let unkeyed = store.get_worker("unkeyed").unwrap();
    let keyed = store.get_worker("keyed").unwrap();
    assert!(!portable_worker_can_claim_job(&connection, &job, &unkeyed).unwrap());
    store
        .set_worker_source_proof_authenticated("keyed", true)
        .unwrap();
    assert!(portable_worker_can_claim_job(&connection, &job, &keyed).unwrap());
    assert!(!portable_mlx_worker_can_claim(&connection, &job, &keyed, true).unwrap());
    store
        .set_worker_source_proof_authenticated("mlx-alternative", true)
        .unwrap();
    assert!(portable_mlx_worker_can_claim(&connection, &job, &keyed, true).unwrap());
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
    assert!(store
        .claim_next_job_routed_with_portable_binding(
            "worker",
            false,
            PORTABLE_MODEL_BINDING_VERSION - 1,
            Some("cpu"),
            Some(std::env::consts::OS)
        )
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
            PORTABLE_MODEL_BINDING_VERSION,
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
            PORTABLE_MODEL_BINDING_VERSION,
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
            Some(&changed),
        )
        .is_err());
    assert!(
        store
            .duplicate_job_with_portable_binding(
                None,
                &original.id,
                DuplicateJob {
                    payload_changes: Default::default(),
                    requested_gpu: None,
                },
                None,
                None,
                DuplicatePortableBinding {
                    binding: Some(&changed),
                    allow_rebind: true,
                },
            )
            .is_err(),
        "the rebind flag cannot authorize changed bytes for the same choices"
    );
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
            DuplicatePortableBinding {
                binding: Some(&changed),
                allow_rebind: false,
            },
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
fn grouped_worker_claims_do_not_invent_one_worker_with_split_sources() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    for (worker_id, source_id) in [
        ("worker-a", "registered_digest:model:a"),
        ("worker-b", "registered_digest:lora:b"),
    ] {
        let claim = PortableSourceClaim::new(
            source_id,
            PortableSourceProvenance::OperatorOverride,
            vec![PortableArtifactFile {
                relative_path: "weights.safetensors".into(),
                size_bytes: 1,
                sha256: format!("sha256:{}", "a".repeat(64)),
            }],
        )
        .unwrap();
        store
            .register_worker_with_portable_sources(
                RegisterWorker {
                    worker_id: worker_id.into(),
                    gpu_id: "cpu".into(),
                    gpu_name: None,
                    capabilities: vec![],
                    loaded_models: vec![],
                    utilization: None,
                },
                Some(std::env::consts::OS),
                Some(&[claim]),
            )
            .unwrap();
    }
    let grouped = store
        .portable_source_claim_sets_for_route("cpu", std::env::consts::OS, "auto", 90)
        .unwrap();
    assert_eq!(grouped.len(), 2);
    assert!(grouped.iter().all(|claims| claims.len() == 1));
    assert!(grouped.iter().all(|claims| {
        !["registered_digest:model:a", "registered_digest:lora:b"]
            .into_iter()
            .all(|id| claims.iter().any(|claim| claim.source_id == id))
    }));
    let catalog_payload = json!({
        "structuredAnalysisEnabled":false,
        "visionAnalysisEnabled":false,
        "semanticEmbeddingsEnabled":false
    })
    .as_object()
    .unwrap()
    .clone();
    assert!(
        store
            .portable_source_claim_sets_for_job_route(
                "cpu",
                std::env::consts::OS,
                "auto",
                90,
                &JobType::CatalogAnalysis,
                &catalog_payload,
            )
            .unwrap()
            .is_empty(),
        "source inventory from a worker without the requested job capability is not availability"
    );
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
            PORTABLE_MODEL_BINDING_VERSION,
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
            PORTABLE_MODEL_BINDING_VERSION,
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
            PORTABLE_MODEL_BINDING_VERSION,
            Some("cpu"),
            Some(std::env::consts::OS),
        )
        .unwrap()
        .0
        .unwrap();
    assert_eq!(claimed.id, job.id);
}

#[test]
fn candle_claim_does_not_defer_portable_image_to_ineligible_mlx_worker() {
    for (mlx_route_without_claim, mlx_required) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let temp = tempfile::tempdir().unwrap();
        let store = JobsStore::new(temp.path().join("jobs.db"));
        store.initialize().unwrap();
        let platform = std::env::consts::OS;
        let authored = json!({"model":"flux_dev"});
        let digest = canonical_authored_options_sha256(&authored).unwrap();
        let member = serde_json::from_value::<PortableJobBinding>(binding())
            .unwrap()
            .routes
            .values()
            .next()
            .unwrap()
            .models[0]
            .members[0]
            .clone();
        let candle_model = PortableModelBinding::new(
            "modelManifestEntry",
            "flux_dev",
            "candle",
            "default",
            None,
            digest.clone(),
            vec![member.clone()],
        )
        .unwrap();
        let candle_route =
            PortableRouteBinding::new("candle", platform, vec![candle_model]).unwrap();
        let mut routes = BTreeMap::from([(candle_route.key(), candle_route)]);
        if mlx_route_without_claim {
            let claim = PortableSourceClaim::new(
                "aux/encoder",
                PortableSourceProvenance::OperatorOverride,
                member.files.clone(),
            )
            .unwrap();
            let mut mlx_member = member;
            mlx_member.source = PortableSourceIdentity::WorkerClaim {
                source_id: claim.source_id,
                provenance: claim.provenance,
                closure_sha256: claim.closure_sha256,
            };
            let mlx_model = PortableModelBinding::new(
                "modelManifestEntry",
                "flux_dev",
                "mlx",
                "default",
                None,
                digest,
                vec![mlx_member],
            )
            .unwrap();
            let route = PortableRouteBinding::new("mlx", platform, vec![mlx_model]).unwrap();
            routes.insert(route.key(), route);
        }
        let portable =
            serde_json::to_value(PortableJobBinding::new(authored, routes).unwrap()).unwrap();
        let mut create = request();
        create.job_type = JobType::ImageGenerate;
        create.payload = json!({"model":"flux_dev", "mode":"text_to_image", "prompt":"test"})
            .as_object()
            .unwrap()
            .clone();
        let job = store
            .create_job_with_portable_binding(create, None, None, Some(&portable))
            .unwrap();
        store
            .register_worker_with_platform(
                RegisterWorker {
                    worker_id: "candle".into(),
                    gpu_id: "cuda:0".into(),
                    gpu_name: None,
                    capabilities: vec![
                        crate::contracts::WorkerCapability::Gpu,
                        crate::contracts::WorkerCapability::ImageGenerate,
                        crate::contracts::WorkerCapability::Unknown("candle".into()),
                    ],
                    loaded_models: vec![],
                    utilization: (!mlx_required).then_some(
                        crate::contracts::WorkerUtilizationSnapshot {
                            memory_total_mb: Some(1000),
                            memory_used_mb: Some(900),
                            memory_free_mb: Some(100),
                            gpu_load_percent: Some(90.0),
                        },
                    ),
                },
                Some(platform),
            )
            .unwrap();
        store
            .register_worker_with_platform(
                RegisterWorker {
                    worker_id: "mlx".into(),
                    gpu_id: "mlx".into(),
                    gpu_name: None,
                    capabilities: vec![
                        crate::contracts::WorkerCapability::Gpu,
                        crate::contracts::WorkerCapability::ImageGenerate,
                    ],
                    loaded_models: vec![],
                    utilization: (!mlx_required).then_some(
                        crate::contracts::WorkerUtilizationSnapshot {
                            memory_total_mb: Some(1000),
                            memory_used_mb: Some(100),
                            memory_free_mb: Some(900),
                            gpu_load_percent: Some(10.0),
                        },
                    ),
                },
                Some(platform),
            )
            .unwrap();
        let claimed = store
            .claim_next_job_routed_with_portable_binding(
                "candle",
                mlx_required,
                PORTABLE_MODEL_BINDING_VERSION,
                Some("candle"),
                Some(platform),
            )
            .unwrap()
            .0
            .unwrap();
        assert_eq!(
            claimed.id, job.id,
            "mlx route exists: {mlx_route_without_claim}, mlx required: {mlx_required}"
        );
    }
}
