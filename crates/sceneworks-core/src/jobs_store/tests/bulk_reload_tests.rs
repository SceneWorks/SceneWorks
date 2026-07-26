use super::*;

#[test]
fn jobs_by_ids_uses_one_bind_beyond_sqlite_variable_limit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let connection = Connection::open(store.db_path()).expect("db opens");
    let job_ids = (0..40_000)
        .map(|index| format!("missing-{index}"))
        .collect::<Vec<_>>();

    let error = store
        .jobs_by_ids(&connection, &job_ids)
        .expect_err("missing ids return not found after the query executes");
    assert!(
        matches!(&error, JobsStoreError::NotFound(id) if id == "missing-0"),
        "large batches must execute without SQLite's bind-variable limit: {error}"
    );
}
