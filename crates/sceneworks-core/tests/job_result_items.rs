//! Per-item job results stored beside the job row (SW-7082): owner-only appends, replace by item
//! id, ordered pages with a byte budget, the per-item size cap, and removal with the job.
use rusqlite::{params, Connection};
use sceneworks_core::contracts::{JobStatus, JobType, ProgressStage, WorkerCapability};
use sceneworks_core::jobs_store::{
    CreateJob, JobResultItemWrite, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    JOB_RESULT_ITEMS_PAGE_BYTES, MAX_JOB_RESULT_ITEMS_PER_APPEND, MAX_JOB_RESULT_ITEM_BYTES,
};
use serde_json::{json, Value};

struct Fixture {
    store: JobsStore,
    db_path: std::path::PathBuf,
    job_id: String,
    _dir: tempfile::TempDir,
}

const WORKER: &str = "worker-1";

/// A store with one image job claimed by `WORKER`.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temp dir");
    let db_path = dir.path().join("jobs.db");
    let store = JobsStore::new(&db_path);
    store.initialize().expect("store initializes");
    store
        .register_worker(RegisterWorker {
            worker_id: WORKER.to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(CreateJob {
            job_type: JobType::ImageGenerate,
            project_id: None,
            project_name: None,
            payload: json!({ "prompt": "mist over hills" })
                .as_object()
                .expect("object")
                .clone(),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("job creates");
    let job_id = store
        .claim_next_job(WORKER)
        .expect("claim succeeds")
        .expect("job claimed")
        .id;
    Fixture {
        store,
        db_path,
        job_id,
        _dir: dir,
    }
}

fn item(position: u64, id: &str, text: &str) -> JobResultItemWrite {
    let Value::Object(item) = json!({ "itemId": id, "description": text }) else {
        unreachable!()
    };
    JobResultItemWrite { position, item }
}

fn ids(items: &[Value]) -> Vec<&str> {
    items
        .iter()
        .map(|item| item["itemId"].as_str().unwrap())
        .collect()
}

fn finish(fixture: &Fixture, status: JobStatus) {
    fixture
        .store
        .update_job_progress(
            &fixture.job_id,
            ProgressUpdate {
                status,
                stage: ProgressStage::Completed,
                progress: 1.0,
                message: "done".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: Some(WORKER.to_owned()),
            },
        )
        .expect("terminal progress");
}

#[test]
fn items_page_back_in_position_order_and_a_reported_item_is_replaced() {
    let fixture = fixture();
    let store = &fixture.store;
    // Appended out of order; the read is by position.
    for (position, id) in [(2, "c"), (0, "a"), (1, "b")] {
        store
            .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(position, id, id)])
            .expect("append");
    }
    let total = store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(1, "b", "again")])
        .expect("a re-reported item replaces the stored one");
    assert_eq!(total, 3);

    let page = store
        .list_job_result_items(&fixture.job_id, 0, 2)
        .expect("page");
    assert_eq!(ids(&page.items), vec!["a", "b"]);
    assert_eq!(page.items[1]["description"], "again");
    assert_eq!((page.total, page.next_offset), (3, Some(2)));
    let rest = store
        .list_job_result_items(&fixture.job_id, 2, 2)
        .expect("page");
    assert_eq!(ids(&rest.items), vec!["c"]);
    assert_eq!(rest.next_offset, None);
}

#[test]
fn appends_come_only_from_the_owning_worker_of_an_active_job() {
    let fixture = fixture();
    let store = &fixture.store;
    let error = store
        .append_job_result_items(&fixture.job_id, Some("other"), &[item(0, "a", "x")])
        .expect_err("another worker cannot write");
    assert!(
        matches!(error, JobsStoreError::NotJobOwner { .. }),
        "{error}"
    );
    let error = store
        .append_job_result_items(&fixture.job_id, None, &[item(0, "a", "x")])
        .expect_err("an unattributed write is refused");
    assert!(
        matches!(error, JobsStoreError::NotJobOwner { .. }),
        "{error}"
    );

    store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(0, "a", "x")])
        .expect("the owner writes");
    finish(&fixture, JobStatus::Canceled);
    let error = store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(1, "b", "y")])
        .expect_err("a terminal job takes no more items");
    assert!(
        matches!(error, JobsStoreError::TerminalJobImmutable { .. }),
        "{error}"
    );
    // A canceled job still serves the items it finished.
    let page = store
        .list_job_result_items(&fixture.job_id, 0, 100)
        .expect("page");
    assert_eq!(ids(&page.items), vec!["a"]);

    let error = store
        .list_job_result_items("missing-job", 0, 10)
        .expect_err("an unknown job is not an empty list");
    assert!(matches!(error, JobsStoreError::NotFound(_)), "{error}");
}

#[test]
fn an_oversized_item_an_item_without_an_id_and_an_overfull_append_are_refused() {
    let fixture = fixture();
    let store = &fixture.store;
    let huge = "x".repeat(MAX_JOB_RESULT_ITEM_BYTES);
    let error = store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(0, "a", &huge)])
        .expect_err("an item over the cap is refused");
    assert!(
        matches!(error, JobsStoreError::InvalidResultItem(_)),
        "{error}"
    );
    let Value::Object(no_id) = json!({ "description": "x" }) else {
        unreachable!()
    };
    let error = store
        .append_job_result_items(
            &fixture.job_id,
            Some(WORKER),
            &[JobResultItemWrite {
                position: 0,
                item: no_id,
            }],
        )
        .expect_err("an item needs an id");
    assert!(
        matches!(error, JobsStoreError::InvalidResultItem(_)),
        "{error}"
    );
    let many: Vec<_> = (0..=MAX_JOB_RESULT_ITEMS_PER_APPEND as u64)
        .map(|index| item(index, &format!("i{index}"), "x"))
        .collect();
    let error = store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &many)
        .expect_err("too many items in one append");
    assert!(
        matches!(error, JobsStoreError::InvalidResultItem(_)),
        "{error}"
    );
    assert_eq!(
        store
            .list_job_result_items(&fixture.job_id, 0, 10)
            .unwrap()
            .total,
        0,
        "a refused append stores nothing"
    );
}

#[test]
fn a_page_stops_at_its_byte_budget_but_always_holds_one_item() {
    let fixture = fixture();
    let store = &fixture.store;
    // Each item is near the per-item cap, so the page budget holds only a few of them.
    let text = "y".repeat(900_000);
    for index in 0..6u64 {
        store
            .append_job_result_items(
                &fixture.job_id,
                Some(WORKER),
                &[item(index, &format!("i{index}"), &text)],
            )
            .expect("append");
    }
    let mut offset = 0;
    let mut seen = Vec::new();
    let mut pages = 0;
    loop {
        let page = store
            .list_job_result_items(&fixture.job_id, offset, 500)
            .expect("page");
        let bytes: usize = page.items.iter().map(|item| item.to_string().len()).sum();
        assert!(!page.items.is_empty());
        assert!(
            page.items.len() == 1 || bytes <= JOB_RESULT_ITEMS_PAGE_BYTES,
            "{bytes} bytes in {} items",
            page.items.len()
        );
        pages += 1;
        seen.extend(ids(&page.items).into_iter().map(str::to_owned));
        match page.next_offset {
            Some(next) => offset = next,
            None => break,
        }
    }
    assert_eq!(seen, vec!["i0", "i1", "i2", "i3", "i4", "i5"]);
    assert!(
        pages >= 2,
        "the byte budget split the items over {pages} page(s)"
    );
}

#[test]
fn retention_removes_a_purged_jobs_items() {
    let fixture = fixture();
    fixture
        .store
        .append_job_result_items(&fixture.job_id, Some(WORKER), &[item(0, "a", "x")])
        .expect("append");
    finish(&fixture, JobStatus::Completed);
    let purged = fixture
        .store
        .purge_terminal_jobs_completed_before("2999-01-01T00:00:00Z")
        .expect("purge");
    assert_eq!(purged, 1);
    let connection = Connection::open(&fixture.db_path).expect("db opens");
    let remaining: i64 = connection
        .query_row(
            "select count(*) from job_result_items where job_id = ?1",
            params![fixture.job_id],
            |row| row.get(0),
        )
        .expect("count");
    assert_eq!(remaining, 0);
}
