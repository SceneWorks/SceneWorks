//! `queue_summary` runs on every queue publish. Its two scans must be served by
//! the partial indexes that hold only the uncleared / unfinished rows, not by a
//! walk of the whole history table — and the indexed queries must still return
//! exactly what the full scans did.
use super::*;

fn plan(connection: &Connection, sql: &str) -> String {
    let mut statement = connection
        .prepare(&format!("explain query plan {sql}"))
        .expect("plan prepares");
    statement
        .query_map([], |row| row.get::<_, String>(3))
        .expect("plan runs")
        .collect::<Result<Vec<_>, _>>()
        .expect("plan rows decode")
        .join(" | ")
}

fn image_job(store: &JobsStore, prompt: &str) -> JobSnapshot {
    store
        .create_job(CreateJob {
            job_type: JobType::ImageDetail,
            project_id: Some("project-1".into()),
            project_name: Some("Project 1".into()),
            payload: serde_json::json!({ "prompt": prompt })
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("job creates")
}

#[test]
fn queue_summary_scans_use_the_partial_indexes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let connection = store.open_connection().expect("read connection");

    let counts = plan(&connection, queue_summary_counts_sql());
    assert!(
        counts.contains("idx_jobs_uncleared_status"),
        "count query plan: {counts}"
    );
    let active = plan(&connection, queue_summary_active_jobs_sql());
    assert!(
        active.contains("idx_jobs_unfinished_created"),
        "active-jobs query plan: {active}"
    );
    assert!(
        !active.contains("TEMP B-TREE"),
        "the index already orders by created_at: {active}"
    );
}

#[test]
fn queue_summary_counts_and_active_jobs_are_unchanged_by_the_indexes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");

    let queued = image_job(&store, "queued");
    let running = image_job(&store, "running");
    let completed = image_job(&store, "completed");
    let cleared = image_job(&store, "cleared");
    let unknown = image_job(&store, "unknown");
    {
        let mut guard = store.lock.lock();
        let connection = store
            .write_connection(&mut guard)
            .expect("write connection");
        connection
            .execute(
                "update jobs set status = 'running' where id = ?1",
                params![running.id],
            )
            .unwrap();
        connection
            .execute(
                "update jobs set status = 'completed' where id in (?1, ?2)",
                params![completed.id, cleared.id],
            )
            .unwrap();
        connection
            .execute(
                "update jobs set cleared_at = '2026-01-01T00:00:00Z' where id = ?1",
                params![cleared.id],
            )
            .unwrap();
        // A status this build does not know stays visible as active, as with the
        // `not in (terminal)` scan the index replaced.
        connection
            .execute(
                "update jobs set status = 'future_status' where id = ?1",
                params![unknown.id],
            )
            .unwrap();
    }

    let summary = store.queue_summary().expect("summary reads");
    assert_eq!(summary.counts[&JobStatus::Queued], 1);
    assert_eq!(summary.counts[&JobStatus::Running], 1);
    assert_eq!(
        summary.counts[&JobStatus::Completed],
        1,
        "cleared run excluded"
    );
    assert_eq!(
        summary.counts[&JobStatus::Unknown("future_status".into())],
        1
    );
    let mut active = summary
        .active_jobs
        .iter()
        .map(|job| job.id.clone())
        .collect::<Vec<_>>();
    active.sort();
    let mut expected = vec![queued.id, running.id, unknown.id];
    expected.sort();
    assert_eq!(active, expected);
}
