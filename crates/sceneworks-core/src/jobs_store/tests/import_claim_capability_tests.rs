use super::*;
use serde_json::json;

fn proof(job: &JobSnapshot) -> &str {
    job.payload[IMPORT_CLAIM_CAPABILITY_KEY]
        .as_str()
        .expect("server-minted import proof")
}

#[test]
fn import_creation_retry_and_duplicate_mint_distinct_private_claims() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.sqlite"));
    store.initialize().unwrap();
    let original = store
        .create_job(CreateJob {
            job_type: JobType::ModelImport,
            project_id: None,
            project_name: None,
            payload: json!({"modelId":"checkpoint","sourcePath":"/private/source","importClaimCapability":"caller-value"})
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    assert_eq!(proof(&original).len(), 64);
    assert_ne!(proof(&original), "caller-value");
    let retry = store
        .retry_job(
            &original.id,
            RetryJob {
                payload_changes: Map::new(),
            },
        )
        .unwrap();
    let duplicate = store
        .duplicate_job(
            &original.id,
            DuplicateJob {
                payload_changes: Map::new(),
                requested_gpu: None,
            },
        )
        .unwrap();
    assert_ne!(proof(&original), proof(&retry));
    assert_ne!(proof(&original), proof(&duplicate));
    assert_ne!(proof(&retry), proof(&duplicate));
    assert_eq!(retry.source_job_id.as_deref(), Some(original.id.as_str()));
}

#[test]
fn legacy_queued_import_receives_private_proof_on_claim() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobsStore::new(temp.path().join("jobs.sqlite"));
    store.initialize().unwrap();
    let job = store
        .create_job(CreateJob {
            job_type: JobType::ModelImport,
            project_id: None,
            project_name: None,
            payload: json!({"modelId":"legacy","sourcePath":"/private/legacy"})
                .as_object()
                .unwrap()
                .clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    let mut old_payload = job.payload;
    old_payload.remove(IMPORT_CLAIM_CAPABILITY_KEY);
    let connection = Connection::open(&store.db_path).unwrap();
    connection.execute(
        "update jobs set payload_json = ?1, claim_payload_revision = payload_revision + 1 where id = ?2",
        params![dumps(&old_payload).unwrap(), job.id],
    ).unwrap();
    store
        .register_worker(RegisterWorker {
            worker_id: "import-worker".into(),
            gpu_id: "cpu".into(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ModelImport],
            loaded_models: vec![],
            utilization: None,
        })
        .unwrap();
    let claimed = store.claim_next_job("import-worker").unwrap().unwrap();
    assert_eq!(claimed.id, job.id);
    assert_eq!(proof(&claimed).len(), 64);
    assert_eq!(store.get_job(&job.id).unwrap().payload, claimed.payload);
}
