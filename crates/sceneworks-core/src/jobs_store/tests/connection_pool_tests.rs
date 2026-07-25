//! sc-11202 / F-025: the store keeps ONE long-lived write connection behind
//! its write mutex instead of opening a fresh one per op. These pin that the
//! long-lived connection still establishes WAL + busy_timeout + foreign_keys
//! exactly once, so the concurrency contract the per-op opener guaranteed is
//! unchanged.
use super::*;

/// The long-lived write connection carries the same three pragmas every
/// opener set before the refactor: a 5s `busy_timeout` (so a concurrent
/// worker+API writer waits on the lock instead of failing instantly),
/// `foreign_keys=on`, and WAL journal mode.
#[test]
fn long_lived_write_connection_has_wal_busy_timeout_and_foreign_keys() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");

    let mut guard = store.lock.lock();
    let connection = store
        .write_connection(&mut guard)
        .expect("long-lived write connection");

    let busy_timeout: i64 = connection
        .pragma_query_value(None, "busy_timeout", |row| row.get(0))
        .expect("busy_timeout pragma reads");
    assert_eq!(busy_timeout, 5000, "5s busy_timeout established");

    let foreign_keys: i64 = connection
        .pragma_query_value(None, "foreign_keys", |row| row.get(0))
        .expect("foreign_keys pragma reads");
    assert_eq!(foreign_keys, 1, "foreign_keys enforcement on");

    let journal_mode: String = connection
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .expect("journal_mode pragma reads");
    assert_eq!(journal_mode, "wal", "WAL journal mode established");
}

/// The write connection is opened once and REUSED across ops: a connection-
/// scoped `TEMP` table created on it survives into a later `write_connection`
/// call. A fresh `Connection::open` per op (the pre-refactor churn) would not
/// carry the temp table, so this fails if the long-lived connection is ever
/// re-created behind the mutex.
#[test]
fn write_connection_is_reused_across_calls() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");

    {
        let mut guard = store.lock.lock();
        let connection = store.write_connection(&mut guard).expect("write conn");
        connection
            .execute_batch("create temp table reuse_probe(x integer)")
            .expect("temp table created on the long-lived connection");
    }

    let mut guard = store.lock.lock();
    let connection = store.write_connection(&mut guard).expect("write conn");
    let exists: i64 = connection
        .query_row(
            "select count(*) from sqlite_temp_master \
                 where type = 'table' and name = 'reuse_probe'",
            [],
            |row| row.get(0),
        )
        .expect("temp probe query");
    assert_eq!(
        exists, 1,
        "the connection-scoped temp table persists → same long-lived connection reused"
    );
}

#[test]
fn generation_stats_created_at_sort_keys_are_indexed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let connection = store.open_connection().expect("read connection");

    let index_exists = |table: &str, index: &str| {
        let mut statement = connection
            .prepare(&format!("pragma index_list({table})"))
            .expect("index list prepares");
        statement
            .query_map([], |row| row.get::<_, String>(1))
            .expect("index list reads")
            .collect::<Result<Vec<_>, _>>()
            .expect("index names decode")
            .iter()
            .any(|name| name == index)
    };

    assert!(index_exists("jobs", "idx_jobs_created"));
    assert!(index_exists(
        "generation_metrics_history",
        "idx_genmetrics_history_created"
    ));
}
