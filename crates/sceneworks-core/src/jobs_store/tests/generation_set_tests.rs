use super::*;
use serde_json::json;

fn seed_job(store: &JobsStore, id: &str, project: &str, generation: &str, cleared: bool) {
    // Seed retained results directly so the test covers cleared and ambiguous historical rows
    // independently of worker claim/routing behavior.
    store
        .open_connection()
        .unwrap()
        .execute(
            "insert into jobs (id, type, status, project_id, payload_json, result_json,
                           created_at, updated_at, cleared_at)
         values (?1, 'image_generate', 'completed', ?2, '{}', ?3, 'now', 'now', ?4)",
            params![
                id,
                project,
                json!({"generationSetId": generation}).to_string(),
                cleared.then_some("now")
            ],
        )
        .unwrap();
}

#[test]
fn generation_set_batch_preserves_scope_cleared_rows_and_duplicate_detection() {
    let dir = tempfile::tempdir().unwrap();
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().unwrap();
    seed_job(&store, "cleared", "p", "ready", true);
    seed_job(&store, "other-project", "other", "ready", false);
    for id in ["duplicate-1", "duplicate-2", "duplicate-3"] {
        seed_job(&store, id, "p", "ambiguous", false);
    }
    let mut ids = (0..40_000)
        .map(|i| format!("missing-{i}"))
        .collect::<Vec<_>>();
    ids.extend(["ready".into(), "ready".into(), "ambiguous".into()]);
    let found = store.jobs_for_generation_sets("p", &ids).unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!(
        found["ready"].len(),
        1,
        "repeated requests must not duplicate jobs"
    );
    assert_eq!(found["ready"][0].id, "cleared");
    assert_eq!(
        found["ambiguous"].len(),
        2,
        "two matches suffice to reject ambiguity"
    );
    assert_eq!(
        store.jobs_for_generation_set("p", "ready").unwrap()[0].id,
        "cleared"
    );
    assert!(store.jobs_for_generation_sets("p", &[]).unwrap().is_empty());
}

#[test]
fn generation_set_readiness_uses_expression_index_after_result_updates() {
    let dir = tempfile::tempdir().unwrap();
    let store = JobsStore::new(dir.path().join("jobs.db"));
    store.initialize().unwrap();
    seed_job(&store, "job", "p", "before", false);
    let connection = store.open_connection().unwrap();
    // Existing installations gain the index on their next startup, including retained results.
    connection
        .execute_batch("drop index idx_jobs_project_generation_set")
        .unwrap();
    store.initialize().unwrap();
    connection
        .execute(
            "update jobs set result_json = ?1 where id = 'job'",
            [json!({"generationSetId": "after"}).to_string()],
        )
        .unwrap();
    let found = store
        .jobs_for_generation_sets("p", &["before".into(), "after".into()])
        .unwrap();
    assert!(!found.contains_key("before"));
    assert_eq!(found["after"][0].id, "job");
    let plan = connection
        .prepare(
            "explain query plan select * from jobs
         where project_id = ?1 and json_extract(result_json, '$.generationSetId') = ?2 limit 2",
        )
        .unwrap()
        .query_map(params!["p", "after"], |row| row.get::<_, String>(3))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        plan.iter()
            .any(|step| step.contains("idx_jobs_project_generation_set")),
        "{plan:?}"
    );
}
