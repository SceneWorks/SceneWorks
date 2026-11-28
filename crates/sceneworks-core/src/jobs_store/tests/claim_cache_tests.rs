//! Admission reads every eligible worker's private claim set several times per
//! submission. A set is parsed once and reused until its `worker_claim_versions`
//! token changes; these pin both the reuse and every write that must invalidate it.
use super::*;
use crate::model_artifacts::portable_binding::{PortableArtifactFile, PortableSourceProvenance};

fn claim(source_id: &str, digit: char) -> PortableSourceClaim {
    PortableSourceClaim::new(
        source_id,
        PortableSourceProvenance::OperatorOverride,
        vec![PortableArtifactFile {
            relative_path: "weights.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", digit.to_string().repeat(64)),
        }],
    )
    .unwrap()
}

fn registration() -> RegisterWorker {
    RegisterWorker {
        worker_id: "worker".into(),
        gpu_id: "cpu".into(),
        gpu_name: None,
        capabilities: vec![],
        loaded_models: vec![],
        utilization: None,
    }
}

fn heartbeat() -> WorkerHeartbeat {
    WorkerHeartbeat {
        worker_id: "worker".into(),
        status: WorkerStatus::Idle,
        current_job_id: None,
        loaded_models: vec![],
        utilization: None,
        status_reason: None,
    }
}

fn route_claims(store: &JobsStore) -> Vec<PortableSourceClaim> {
    store
        .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
        .unwrap()
}

#[test]
fn claim_sets_are_parsed_once_until_they_change() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let first = claim("model/a", 'a');
    store
        .register_worker_with_portable_sources(
            registration(),
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&first)),
        )
        .unwrap();

    // Three reads (admission's per-request pattern) parse the set once.
    for _ in 0..3 {
        assert_eq!(route_claims(&store), vec![first.clone()]);
    }
    assert_eq!(store.portable_claim_parses(), 1);

    // A heartbeat that omits its claims, or re-sends the same ones, changes nothing.
    store.heartbeat_worker(heartbeat()).unwrap();
    store
        .heartbeat_worker_with_portable_sources(heartbeat(), Some(std::slice::from_ref(&first)))
        .unwrap();
    assert_eq!(route_claims(&store), vec![first.clone()]);
    assert_eq!(store.portable_claim_parses(), 1);

    // New claims on a heartbeat are seen on the next read, parsed once.
    let second = claim("model/b", 'b');
    store
        .heartbeat_worker_with_portable_sources(heartbeat(), Some(std::slice::from_ref(&second)))
        .unwrap();
    assert_eq!(route_claims(&store), vec![second.clone()]);
    assert_eq!(route_claims(&store), vec![second.clone()]);
    assert_eq!(store.portable_claim_parses(), 2);

    // A re-registration (the upsert path) invalidates too.
    let third = claim("model/c", 'c');
    store
        .register_worker_with_portable_sources(
            registration(),
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&third)),
        )
        .unwrap();
    assert_eq!(route_claims(&store), vec![third.clone()]);
    assert_eq!(store.portable_claim_parses(), 3);
}

#[test]
fn a_deleted_and_reregistered_worker_never_reuses_its_old_claims() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let old = claim("model/old", 'a');
    store
        .register_worker_with_portable_sources(
            registration(),
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&old)),
        )
        .unwrap();
    assert_eq!(route_claims(&store), vec![old]);
    {
        let mut guard = store.lock.lock();
        let connection = store.write_connection(&mut guard).unwrap();
        connection
            .execute("delete from workers where id = 'worker'", [])
            .unwrap();
        let versions: i64 = connection
            .query_row("select count(*) from worker_claim_versions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(versions, 0, "the version row goes with its worker");
    }
    let new = claim("model/new", 'b');
    store
        .register_worker_with_portable_sources(
            registration(),
            Some(std::env::consts::OS),
            Some(std::slice::from_ref(&new)),
        )
        .unwrap();
    assert_eq!(route_claims(&store), vec![new]);
    assert_eq!(store.portable_claim_parses(), 2);
}
