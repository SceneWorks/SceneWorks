use super::*;
use serde_json::json;

fn create_request() -> CreateJob {
    CreateJob {
        job_type: JobType::ModelDownload,
        project_id: None,
        project_name: None,
        payload: json!({"repo":"fixture/repo", "files":["file"]})
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

#[test]
fn download_binding_transaction_rolls_back_and_remains_private() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.db"));
    store.initialize().unwrap();
    let initial = store.create_job(create_request()).unwrap();
    let binding = json!({"modelId":"fixture", "payload":initial.payload});
    {
        let mut guard = store.lock.lock();
        let conn = store.write_connection(&mut guard).unwrap();
        conn.execute_batch("create trigger reject_binding before update of download_binding_json on jobs begin select raise(abort, 'injected binding write failure'); end;").unwrap();
    }
    assert!(store
        .create_job_with_download_binding(create_request(), None, Some(&binding))
        .is_err());
    assert!(store
        .retry_job_with_download_binding(
            &initial.id,
            RetryJob {
                payload_changes: Default::default()
            },
            None,
            Some(&binding)
        )
        .is_err());
    assert!(store
        .duplicate_job_with_download_binding(
            None,
            &initial.id,
            DuplicateJob {
                payload_changes: Default::default(),
                requested_gpu: None
            },
            None,
            Some(&binding)
        )
        .is_err());
    assert_eq!(store.list_jobs(None, None, 100).unwrap().len(), 1);
    {
        let mut guard = store.lock.lock();
        store
            .write_connection(&mut guard)
            .unwrap()
            .execute_batch("drop trigger reject_binding")
            .unwrap();
    }
    let created = store
        .create_job_with_download_binding(create_request(), None, Some(&binding))
        .unwrap();
    assert_eq!(
        store.download_binding(&created.id).unwrap(),
        Some(binding.clone())
    );
    let reopened = JobsStore::new(temp.path().join("jobs.db"));
    reopened.initialize().unwrap();
    assert_eq!(
        reopened.download_binding(&created.id).unwrap(),
        Some(binding.clone())
    );
    assert!(!created.extra.contains_key("downloadBinding"));
    assert!(!created.extra.contains_key("modelProvenance"));
    let replay = store
        .retry_job_with_download_binding(
            &created.id,
            RetryJob {
                payload_changes: Default::default(),
            },
            None,
            Some(&binding),
        )
        .unwrap();
    assert_eq!(
        store.download_binding(&replay.id).unwrap(),
        Some(binding.clone())
    );

    // Simulate a source field appearing after the API read its snapshot but before enqueue.
    let job_count = store.list_jobs(None, None, 100).unwrap().len();
    {
        let mut guard = store.lock.lock();
        let conn = store.write_connection(&mut guard).unwrap();
        let mut changed = created.payload.clone();
        changed.insert("sourceRepo".into(), json!("other/repo"));
        conn.execute(
            "update jobs set payload_json = ?1 where id = ?2",
            params![dumps(&changed).unwrap(), created.id],
        )
        .unwrap();
    }
    assert!(store
        .retry_job_with_download_binding(
            &created.id,
            RetryJob {
                payload_changes: created.payload.clone()
            },
            None,
            Some(&binding),
        )
        .is_err());
    assert!(store
        .duplicate_job_with_download_binding(
            None,
            &created.id,
            DuplicateJob {
                payload_changes: created.payload.clone(),
                requested_gpu: None
            },
            None,
            Some(&binding),
        )
        .is_err());
    assert_eq!(store.list_jobs(None, None, 100).unwrap().len(), job_count);
}
