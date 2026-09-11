use super::*;
use serde_json::json;

fn workflow_parent(store: &JobsStore) -> JobSnapshot {
    store
        .create_job(CreateJob {
            job_type: JobType::VectorGenerate,
            project_id: Some("project_1".to_owned()),
            project_name: Some("Vectors".to_owned()),
            payload: json!({
                "model": "starvector_1b",
                "workflow": {
                    "kind": "create_from_prompt",
                    "id": "vwf_1",
                    "parentJobId": "pending"
                }
            })
            .as_object()
            .expect("payload object")
            .clone(),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: Some(JobStatus::PendingWorkflow),
        })
        .expect("workflow parent creates")
}

#[test]
fn pending_workflow_is_restart_safe_and_cancel_cannot_be_resurrected() {
    let directory = tempfile::tempdir().expect("temp directory");
    let store = JobsStore::new(directory.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let parent = workflow_parent(&store);
    assert_eq!(parent.status, JobStatus::PendingWorkflow);
    assert_eq!(parent.stage, ProgressStage::PendingWorkflow);

    // Restart recovery owns this state at the API coordinator layer. Core must neither interrupt
    // it as worker-owned nor degrade it to queued like the best-effort caption pre-step.
    assert!(store
        .mark_interrupted_on_startup()
        .expect("restart sweep")
        .is_empty());
    assert_eq!(
        store.get_job(&parent.id).expect("parent remains").status,
        JobStatus::PendingWorkflow
    );

    let canceled = store.cancel_job(&parent.id).expect("parent cancels");
    assert_eq!(canceled.status, JobStatus::Canceled);
    let replacement = json!({ "model": "other" })
        .as_object()
        .expect("replacement")
        .clone();
    let update = store
        .update_pending_workflow_payload(&parent.id, replacement.clone())
        .expect("guarded update");
    assert!(!update.changed);
    assert_eq!(update.job.status, JobStatus::Canceled);
    let promotion = store
        .promote_pending_workflow_job(&parent.id, replacement)
        .expect("guarded promotion");
    assert!(!promotion.changed);
    assert_eq!(promotion.job.status, JobStatus::Canceled);
}

#[test]
fn pending_workflow_promotes_once_with_the_resolved_source_payload() {
    let directory = tempfile::tempdir().expect("temp directory");
    let store = JobsStore::new(directory.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let parent = workflow_parent(&store);
    let resolved = json!({
        "model": "starvector_1b",
        "mode": "image_to_svg",
        "sourceAssetId": "asset_intermediate",
        "workflow": { "kind": "create_from_prompt", "id": "vwf_1" }
    })
    .as_object()
    .expect("resolved payload")
    .clone();
    let promotion = store
        .promote_pending_workflow_job(&parent.id, resolved)
        .expect("parent promotes");
    assert!(promotion.changed);
    assert_eq!(promotion.job.status, JobStatus::Queued);
    assert_eq!(promotion.job.stage, ProgressStage::Queued);
    assert_eq!(
        promotion.job.payload["sourceAssetId"],
        json!("asset_intermediate")
    );

    let second = store
        .promote_pending_workflow_job(&parent.id, promotion.job.payload.clone())
        .expect("second promotion is idempotent");
    assert!(!second.changed);
    assert_eq!(second.job.status, JobStatus::Queued);
}

#[test]
fn startup_recovery_cannot_lose_a_workflow_behind_the_public_list_cap() {
    let directory = tempfile::tempdir().expect("temp directory");
    let store = JobsStore::new(directory.path().join("jobs.db"));
    store.initialize().expect("store initializes");
    let parent = workflow_parent(&store);
    for index in 0..501 {
        store
            .create_job(CreateJob {
                job_type: JobType::Placeholder,
                project_id: None,
                project_name: None,
                payload: json!({ "index": index })
                    .as_object()
                    .expect("payload")
                    .clone(),
                requested_gpu: "auto".to_owned(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .expect("newer job creates");
    }
    let recovery = store
        .list_vector_prompt_workflow_jobs_for_recovery()
        .expect("recovery list");
    assert_eq!(recovery.len(), 1);
    assert_eq!(recovery[0].id, parent.id);
}
