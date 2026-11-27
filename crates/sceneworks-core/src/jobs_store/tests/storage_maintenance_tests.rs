//! Jobs DB file upkeep: the startup compaction threshold and the WAL truncate
//! checkpoint. The filler table stands in for released worker claim blobs: what
//! matters is free pages in the file, not which table owned them.
use super::*;
use std::time::Duration;

fn initialized_store(dir: &tempfile::TempDir) -> JobsStore {
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    store
}

/// Write `megabytes` of blob pages into a scratch table, then optionally drop it so
/// every one of those pages lands on the freelist.
fn write_filler(store: &JobsStore, megabytes: usize, release: bool) {
    let mut guard = store.lock.lock();
    let connection = store
        .write_connection(&mut guard)
        .expect("write connection");
    connection
        .execute_batch("create table if not exists filler (body blob not null)")
        .expect("filler table");
    for _ in 0..megabytes {
        connection
            .execute("insert into filler values (zeroblob(1048576))", [])
            .expect("filler row");
    }
    if release {
        connection
            .execute_batch("drop table filler")
            .expect("filler drops");
    }
}

fn wal_bytes(store: &JobsStore) -> u64 {
    let mut wal = store.db_path().as_os_str().to_owned();
    wal.push("-wal");
    std::fs::metadata(std::path::PathBuf::from(wal))
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

const TEST_POLICY: CompactionPolicy = CompactionPolicy {
    min_free_fraction: 0.25,
    min_free_bytes: 1024 * 1024,
    max_live_bytes: u64::MAX,
};

#[test]
fn startup_policy_thresholds() {
    let policy = CompactionPolicy::STARTUP;
    let page = 4096;
    // The measured real-world file: 227,124 pages, 141,964 free after releasing
    // offline-worker claims (62%, ~555 MiB free, ~339 MiB live).
    assert!(policy.compaction_due(page, 227_124, 141_964));
    // The same file before the release: 1,027 free pages is ordinary churn.
    assert!(!policy.compaction_due(page, 227_124, 1_027));
    // A large fraction of a small file is not worth a startup rewrite.
    assert!(!policy.compaction_due(page, 1_000, 900));
    // Plenty of free bytes but under a quarter of the file.
    assert!(!policy.compaction_due(page, 1_000_000, 200_000));
    // Over the live-size cap the rewrite would hold startup too long.
    let live_cap_pages = policy.max_live_bytes / page;
    assert!(!policy.compaction_due(page, live_cap_pages * 2 + 1, live_cap_pages));
    assert!(!policy.compaction_due(page, 0, 0));
}

#[test]
fn compaction_reclaims_a_fragmented_file_and_skips_a_dense_one() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = initialized_store(&dir);
    write_filler(&store, 4, false);

    // Dense: nothing released, so nothing to reclaim.
    let dense = store
        .compact_if_fragmented(TEST_POLICY)
        .expect("compaction check runs");
    assert!(!dense.vacuumed, "{dense:?}");
    assert_eq!(dense.page_count_after, dense.page_count_before);

    write_filler(&store, 0, true);
    let job = store
        .create_job(CreateJob {
            job_type: JobType::ImageDetail,
            project_id: Some("project-1".into()),
            project_name: Some("Project 1".into()),
            payload: serde_json::json!({"prompt": "kept"})
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("job creates");

    let report = store
        .compact_if_fragmented(TEST_POLICY)
        .expect("compaction runs");
    assert!(report.vacuumed, "{report:?}");
    assert!(
        report.freelist_count_before * report.page_size >= 4 * 1024 * 1024,
        "{report:?}"
    );
    assert!(
        report.page_count_after + report.freelist_count_before <= report.page_count_before + 1,
        "VACUUM returned the free pages: {report:?}"
    );
    // Rows survive and the store keeps working on the same write connection.
    assert_eq!(store.get_job(&job.id).expect("job reads").id, job.id);
    let again = store
        .compact_if_fragmented(TEST_POLICY)
        .expect("second check runs");
    assert!(!again.vacuumed, "a compacted file is not compacted again");
}

#[test]
fn truncate_checkpoint_empties_the_wal() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = initialized_store(&dir);
    // A reader holding an open snapshot keeps auto-checkpoints from resetting the
    // WAL, so the filler accumulates there the way a busy queue's writes do.
    let reader = store.open_connection().expect("reader opens");
    reader.execute_batch("begin").expect("reader begins");
    let _: i64 = reader
        .query_row("select count(*) from jobs", [], |row| row.get(0))
        .expect("reader snapshot");
    write_filler(&store, 2, false);
    assert!(wal_bytes(&store) > 1024 * 1024, "the WAL holds the burst");

    let blocked = store
        .checkpoint_wal_truncate(Duration::from_millis(0))
        .expect("blocked checkpoint reports instead of failing");
    assert!(blocked.busy, "an open reader blocks TRUNCATE: {blocked:?}");

    reader.execute_batch("commit").expect("reader ends");
    let checkpoint = store
        .checkpoint_wal_truncate(Duration::from_secs(5))
        .expect("checkpoint runs");
    assert!(!checkpoint.busy, "{checkpoint:?}");
    assert_eq!(wal_bytes(&store), 0, "TRUNCATE resets the WAL file");
}
