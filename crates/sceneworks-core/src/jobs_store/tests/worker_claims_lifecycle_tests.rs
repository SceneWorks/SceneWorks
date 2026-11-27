//! Worker rows carry a private, multi-MB source-claim inventory (`portable_source_claims_json`)
//! that `WorkerSnapshot` never reads. These tests pin the three halves of keeping it off the hot
//! path: snapshot queries name their columns instead of `select *`, every path that takes a worker
//! offline drops its claims, and startup releases the claims (and the long-dead rows) an older
//! build left behind.

use super::*;
use crate::contracts::WorkerUtilizationSnapshot;
use crate::model_artifacts::portable_binding::{
    PortableArtifactFile, PortableSourceClaim, PortableSourceProvenance,
};

fn claim(source_id: &str) -> PortableSourceClaim {
    PortableSourceClaim::new(
        source_id,
        PortableSourceProvenance::OperatorOverride,
        vec![PortableArtifactFile {
            relative_path: "weights.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "a".repeat(64)),
        }],
    )
    .unwrap()
}

fn register(store: &JobsStore, worker_id: &str, gpu_id: &str, claims: &[PortableSourceClaim]) {
    store
        .register_worker_with_portable_sources(
            RegisterWorker {
                worker_id: worker_id.into(),
                gpu_id: gpu_id.into(),
                gpu_name: Some(format!("GPU {gpu_id}")),
                capabilities: vec![
                    WorkerCapability::Gpu,
                    WorkerCapability::ImageGenerate,
                    WorkerCapability::Unknown("candle".into()),
                ],
                loaded_models: vec!["model-a".into()],
                utilization: Some(WorkerUtilizationSnapshot {
                    memory_total_mb: Some(98_000),
                    memory_used_mb: Some(1_000),
                    memory_free_mb: Some(97_000),
                    gpu_load_percent: Some(12.5),
                }),
            },
            Some(std::env::consts::OS),
            Some(claims),
        )
        .unwrap();
}

fn heartbeat(worker_id: &str, status: WorkerStatus) -> WorkerHeartbeat {
    WorkerHeartbeat {
        worker_id: worker_id.into(),
        status,
        current_job_id: None,
        loaded_models: vec![],
        utilization: None,
        status_reason: None,
    }
}

fn raw_claims(store: &JobsStore, worker_id: &str) -> Option<String> {
    rusqlite::Connection::open(store.db_path())
        .unwrap()
        .query_row(
            "select portable_source_claims_json from workers where id = ?1",
            params![worker_id],
            |row| row.get(0),
        )
        .unwrap()
}

fn sql(store: &JobsStore, statement: &str, values: impl rusqlite::Params) {
    rusqlite::Connection::open(store.db_path())
        .unwrap()
        .execute(statement, values)
        .unwrap();
}

fn store() -> (tempfile::TempDir, JobsStore) {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    (temp, store)
}

/// Regression guard for the queue-publish stall: `select * from workers` makes SQLite read (and,
/// with `order by`, sort) every row's claim blob. Whitespace is collapsed so a multi-line query
/// cannot slip past.
#[test]
fn worker_queries_never_select_star() {
    let source = include_str!("../../jobs_store.rs");
    let normalized = source
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    assert!(
        !normalized.contains("select * from workers"),
        "a workers query uses `select *`; select WORKER_SNAPSHOT_COLUMNS (and the claim column \
         only for the worker that needs it) instead"
    );
}

/// The named column list must return exactly what `select *` did, through every converted query.
#[test]
fn snapshot_column_queries_return_the_same_workers_as_select_star() {
    let (_temp, store) = store();
    register(
        &store,
        "w-b",
        "cuda:1",
        &[claim("registered_digest:model:b")],
    );
    register(
        &store,
        "w-a",
        "cuda:0",
        &[claim("registered_digest:model:a")],
    );
    register(&store, "w-mlx", "mlx", &[]);
    store
        .heartbeat_worker(WorkerHeartbeat {
            status_reason: Some("GPU failed its startup probe".into()),
            ..heartbeat("w-b", WorkerStatus::Unhealthy)
        })
        .unwrap();
    register(
        &store,
        "w-gone",
        "cuda:0",
        &[claim("registered_digest:model:c")],
    );
    store
        .heartbeat_worker(heartbeat("w-gone", WorkerStatus::Offline))
        .unwrap();

    let connection = rusqlite::Connection::open(store.db_path()).unwrap();
    let mut statement = connection
        .prepare("select * from workers order by gpu_id, id")
        .unwrap();
    let reference = collect_workers(statement.query_map([], row_to_worker).unwrap()).unwrap();
    assert_eq!(reference.len(), 4);

    assert_eq!(store.list_workers().unwrap(), reference);
    for worker in &reference {
        assert_eq!(&store.get_worker(&worker.id).unwrap(), worker);
    }
    let ids = reference
        .iter()
        .map(|worker| worker.id.clone())
        .collect::<Vec<_>>();
    assert_eq!(store.workers_by_ids(&connection, &ids).unwrap(), reference);
    assert_eq!(store.queue_summary().unwrap().workers, reference);
}

#[test]
fn every_offline_transition_drops_the_worker_claims() {
    let (_temp, store) = store();
    let claims = [claim("registered_digest:model:a")];

    // Graceful shutdown: the worker's own `offline` heartbeat.
    register(&store, "graceful", "cuda:0", &claims);
    assert!(raw_claims(&store, "graceful").is_some());
    store
        .heartbeat_worker(heartbeat("graceful", WorkerStatus::Offline))
        .unwrap();
    assert_eq!(raw_claims(&store, "graceful"), None);

    // Stale sweep.
    register(&store, "stale", "cuda:0", &claims);
    sql(
        &store,
        "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = 'stale'",
        [],
    );
    let sweep = store.mark_stale_workers_interrupted(90).unwrap();
    assert_eq!(sweep.workers.len(), 1);
    assert_eq!(sweep.workers[0].status, WorkerStatus::Offline);
    assert_eq!(raw_claims(&store, "stale"), None);

    // Worker process terminated.
    register(&store, "terminated", "cuda:0", &claims);
    store
        .fail_worker_jobs_terminated("terminated", Some(9), None, None)
        .unwrap();
    assert_eq!(raw_claims(&store, "terminated"), None);

    // API restart recovery takes every previous incarnation offline.
    register(&store, "previous-session", "cuda:0", &claims);
    store.mark_interrupted_on_startup().unwrap();
    assert_eq!(raw_claims(&store, "previous-session"), None);
}

/// A worker whose row went offline (and lost its claims) must republish them before the row is
/// revived; otherwise every portable route would silently stop seeing its local sources.
#[test]
fn reviving_an_offline_worker_requires_its_claims_to_be_republished() {
    let (_temp, store) = store();
    let claims = [claim("registered_digest:model:a")];
    register(&store, "worker", "cpu", &claims);
    sql(
        &store,
        "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = 'worker'",
        [],
    );
    store.mark_stale_workers_interrupted(90).unwrap();

    let refused = store
        .heartbeat_worker(heartbeat("worker", WorkerStatus::Idle))
        .unwrap_err();
    assert!(
        matches!(&refused, JobsStoreError::PortableSourceClaimsRequired { worker_id } if worker_id == "worker"),
        "{refused:?}"
    );
    assert_eq!(
        store.get_worker("worker").unwrap().status,
        WorkerStatus::Offline,
        "a refused heartbeat writes nothing"
    );

    // An offline heartbeat needs no claims.
    store
        .heartbeat_worker(heartbeat("worker", WorkerStatus::Offline))
        .unwrap();

    let revived = store
        .heartbeat_worker_with_portable_sources(
            heartbeat("worker", WorkerStatus::Idle),
            Some(&claims),
        )
        .unwrap();
    assert_eq!(revived.worker.status, WorkerStatus::Idle);
    assert_eq!(
        store
            .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
            .unwrap(),
        claims.to_vec()
    );
    // Once republished, claim-less heartbeats keep the stored set as before.
    store
        .heartbeat_worker(heartbeat("worker", WorkerStatus::Idle))
        .unwrap();
    assert_eq!(
        store
            .portable_source_claims_for_route("cpu", std::env::consts::OS, "auto", 90)
            .unwrap(),
        claims.to_vec()
    );
}

#[test]
fn startup_releases_offline_claims_and_prunes_long_dead_rows() {
    let (_temp, store) = store();
    let claims = [claim("registered_digest:model:a")];
    for id in ["live", "offline-recent", "offline-old", "busy-old"] {
        register(&store, id, "cuda:0", &claims);
    }
    // What an older build left behind: offline rows that still carry a large claim blob.
    let blob = format!("[{}]", "\"x\",".repeat(200_000).trim_end_matches(','));
    let now = now_unix_seconds();
    let recent = format_unix_seconds(now - 29 * 86_400);
    let old = format_unix_seconds(now - 31 * 86_400);
    sql(
        &store,
        "update workers set status = 'offline', portable_source_claims_json = ?1, last_seen_at = ?2
          where id = 'offline-recent'",
        params![blob, recent],
    );
    sql(
        &store,
        "update workers set status = 'offline', portable_source_claims_json = ?1, last_seen_at = ?2
          where id = 'offline-old'",
        params![blob, old],
    );
    // Not offline: retention never deletes a row that might still be running something.
    sql(
        &store,
        "update workers set status = 'busy', last_seen_at = ?1 where id = 'busy-old'",
        params![old],
    );
    let live_claims = raw_claims(&store, "live");

    store.initialize().unwrap();

    assert_eq!(raw_claims(&store, "offline-recent"), None);
    assert_eq!(
        store.get_worker("offline-recent").unwrap().status,
        WorkerStatus::Offline,
        "a recently offline row is kept for display"
    );
    assert!(matches!(
        store.get_worker("offline-old"),
        Err(JobsStoreError::NotFound(_))
    ));
    assert_eq!(raw_claims(&store, "live"), live_claims);
    assert!(live_claims.is_some());
    assert_eq!(raw_claims(&store, "busy-old"), live_claims);
    assert_eq!(
        store
            .list_workers()
            .unwrap()
            .into_iter()
            .map(|worker| worker.id)
            .collect::<Vec<_>>(),
        ["busy-old", "live", "offline-recent"]
    );
}

#[test]
fn route_worker_ids_match_the_claim_set_owners() {
    let (_temp, store) = store();
    let register_cpu = |worker_id: &str, gpu_id: &str, claims: &[PortableSourceClaim]| {
        store
            .register_worker_with_portable_sources(
                RegisterWorker {
                    worker_id: worker_id.into(),
                    gpu_id: gpu_id.into(),
                    gpu_name: None,
                    capabilities: vec![WorkerCapability::ModelDownload],
                    loaded_models: vec![],
                    utilization: None,
                },
                Some(std::env::consts::OS),
                Some(claims),
            )
            .unwrap();
    };
    register_cpu("with-claims", "cpu", &[claim("registered_digest:model:a")]);
    register_cpu("empty-claims", "cpu", &[]);
    register_cpu("gone", "cpu", &[claim("registered_digest:model:b")]);
    store
        .heartbeat_worker(heartbeat("gone", WorkerStatus::Offline))
        .unwrap();
    register_cpu(
        "other-backend",
        "cuda:0",
        &[claim("registered_digest:model:c")],
    );
    let payload = serde_json::Map::new();

    let ids = store
        .portable_source_claim_worker_ids_for_job_route(
            "cpu",
            std::env::consts::OS,
            "auto",
            90,
            &JobType::ModelDownload,
            &payload,
        )
        .unwrap();
    let sets = store
        .portable_source_claim_sets_for_job_route(
            "cpu",
            std::env::consts::OS,
            "auto",
            90,
            &JobType::ModelDownload,
            &payload,
        )
        .unwrap();
    let mut sorted = ids.clone();
    sorted.sort();
    // The offline row dropped its claims, so it no longer advertises anything; an empty set is
    // still a (vacuous) claim set, exactly as before.
    assert_eq!(sorted, ["empty-claims", "with-claims"]);
    assert_eq!(ids.len(), sets.len());
}
