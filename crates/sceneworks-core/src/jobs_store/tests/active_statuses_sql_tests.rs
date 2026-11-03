use super::{active_statuses_sql, ACTIVE_STATUSES};

/// Anti-drift guard for sc-4207 / F-CORE-3: the five `status in (...)` SQL
/// statements now interpolate [`active_statuses_sql`] instead of a
/// copy-pasted literal, so the generated list must stay exactly in sync with
/// [`ACTIVE_STATUSES`] — every status quoted, comma-separated, none dropped.
#[test]
fn sql_list_matches_active_statuses_const() {
    let expected = ACTIVE_STATUSES
        .iter()
        .map(|status| format!("'{status}'"))
        .collect::<Vec<_>>()
        .join(", ");
    assert_eq!(active_statuses_sql(), expected);

    // Each status appears as a quoted token, guarding against a future const
    // edit that silently fails to reach the SQL filters.
    for status in ACTIVE_STATUSES {
        assert!(
            active_statuses_sql().contains(&format!("'{status}'")),
            "active status {status:?} missing from SQL list"
        );
    }
}

#[test]
fn status_groups_partition_known_wire_values_and_sql_lists() {
    use super::{JobStatus, JOB_STATUSES, PENDING_STATUSES, TERMINAL_STATUSES};
    let mut grouped = [ACTIVE_STATUSES, PENDING_STATUSES, TERMINAL_STATUSES].concat();
    grouped.sort_unstable();
    let mut known = JOB_STATUSES.to_vec();
    known.sort_unstable();
    assert_eq!(
        grouped, known,
        "every known state belongs to exactly one lifecycle group"
    );
    assert_eq!(JOB_STATUSES, JobStatus::known_values());
    assert_eq!(
        super::terminal_statuses_sql(),
        "'completed', 'failed', 'canceled', 'interrupted'"
    );
    assert_eq!(
        super::pending_statuses_sql(),
        "'queued', 'pending_caption', 'pending_workflow'"
    );
    assert_eq!(
        super::status_sql_list(&["future's status"]),
        "'future''s status'"
    );
}

#[test]
fn fail_queued_jobs_updates_only_queued_candidates() {
    use super::*;
    let mut connection = Connection::open_in_memory().unwrap();
    connection.execute_batch("create table jobs (id text primary key, status text, stage text, message text, error text, completed_at text, updated_at text, worker_id text);
        insert into jobs (id, status, worker_id) values ('queued', 'queued', 'stale_worker'), ('active', 'running', 'live_worker'), ('terminal', 'completed', null), ('untouched', 'queued', null);").unwrap();
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let failed = fail_queued_jobs(
        &transaction,
        "fixture-time",
        ["queued", "active", "terminal", "missing"]
            .map(|id| (id.to_owned(), "fixture message", "fixture error".to_owned())),
    )
    .unwrap();
    assert_eq!(failed, ["queued"]);
    transaction.commit().unwrap();
    let row: (String, String, String, String, String, Option<String>) = connection.query_row("select status, stage, message, error, completed_at, worker_id from jobs where id = 'queued'", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?))).unwrap();
    assert_eq!(
        row,
        (
            "failed".into(),
            "failed".into(),
            "fixture message".into(),
            "fixture error".into(),
            "fixture-time".into(),
            None
        )
    );
    assert_eq!(
        connection
            .query_row(
                "select updated_at from jobs where id = 'queued'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "fixture-time"
    );
    for (id, expected) in [
        ("active", "running"),
        ("terminal", "completed"),
        ("untouched", "queued"),
    ] {
        assert_eq!(
            connection
                .query_row("select status from jobs where id = ?1", [id], |row| row
                    .get::<_, String>(
                    0
                ))
                .unwrap(),
            expected
        );
    }
}
