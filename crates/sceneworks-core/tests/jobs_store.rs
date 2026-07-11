use std::fs;
use std::path::PathBuf;

use rusqlite::{params, Connection};
use sceneworks_core::contracts::{
    GenerationMetrics, JobSnapshot, JobStatus, JobType, ProgressStage, WorkerCapability,
    WorkerStatus, WorkerUtilizationSnapshot,
};
use sceneworks_core::jobs_store::{
    candle_supported, mac_capabilities, mac_rust_supported, model_mac_support, CreateJob,
    DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker, RetryJob,
    WorkerHeartbeat, MAC_NOT_AVAILABLE_LABEL, MAX_JOB_ATTEMPTS,
};
use serde_json::{json, Map, Value};

fn temp_db(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("sceneworks-core-{name}-{}.db", std::process::id()));
    let _ = fs::remove_file(&path);
    path
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("test value is an object").clone()
}

fn store(name: &str) -> JobsStore {
    let store = JobsStore::new(temp_db(name));
    store.initialize().expect("store initializes");
    store
}

fn image_job(payload: Map<String, Value>) -> CreateJob {
    CreateJob {
        job_type: JobType::ImageGenerate,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload,
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

fn register_image_worker(store: &JobsStore) {
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
}

#[test]
fn generation_metrics_upsert_get_list_and_merge() {
    let store = store("gen-metrics");
    let job = store
        .create_job(image_job(object(json!({ "prompt": "mist over hills" }))))
        .expect("job creates");

    let metrics = GenerationMetrics {
        model: Some("qwen_image".to_owned()),
        quant_label: Some("q8".to_owned()),
        quant_bits: Some(8),
        sampler: Some("euler".to_owned()),
        scheduler: Some("karras".to_owned()),
        steps: Some(20),
        image_count: Some(4),
        guidance_scale: serde_json::Number::from_f64(4.0),
        use_pid: Some(false),
        width: Some(1024),
        height: Some(1024),
        seed: Some(42),
        loras: Some(vec!["style-a".to_owned()]),
        load_ms: Some(2100),
        sample_ms: Some(6400),
        decode_ms: Some(900),
        total_ms: Some(9400),
        peak_memory_bytes: Some(12_884_901_888),
        peak_memory_pct: serde_json::Number::from_f64(71.5),
        peak_gpu_load_pct: serde_json::Number::from_f64(88.0),
        backend: Some("mlx".to_owned()),
        ..Default::default()
    };
    store
        .upsert_generation_metrics(&job.id, &metrics)
        .expect("metrics upsert");

    let read = store
        .get_generation_metrics(&job.id)
        .expect("metrics read")
        .expect("metrics present");
    assert_eq!(read.model.as_deref(), Some("qwen_image"));
    assert_eq!(read.quant_label.as_deref(), Some("q8"));
    assert_eq!(read.steps, Some(20));
    assert_eq!(read.image_count, Some(4));
    assert_eq!(read.use_pid, Some(false));
    assert_eq!(read.seed, Some(42));
    assert_eq!(read.loras.as_deref(), Some(&["style-a".to_owned()][..]));
    assert_eq!(read.total_ms, Some(9400));
    assert_eq!(read.peak_memory_bytes, Some(12_884_901_888));
    assert_eq!(
        read.guidance_scale
            .as_ref()
            .and_then(serde_json::Number::as_f64),
        Some(4.0)
    );

    let rows = store
        .list_generation_metrics(None, None, None, 100)
        .expect("metrics list");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].job_id, job.id);
    assert_eq!(rows[0].job_type, JobType::ImageGenerate);
    assert_eq!(rows[0].metrics.model.as_deref(), Some("qwen_image"));

    // A partial second report (only total_ms) must merge, not wipe prior fields.
    let partial = GenerationMetrics {
        total_ms: Some(9100),
        ..Default::default()
    };
    store
        .upsert_generation_metrics(&job.id, &partial)
        .expect("metrics merge upsert");
    let merged = store
        .get_generation_metrics(&job.id)
        .expect("metrics reread")
        .expect("metrics still present");
    assert_eq!(merged.total_ms, Some(9100), "total_ms overwritten");
    assert_eq!(
        merged.model.as_deref(),
        Some("qwen_image"),
        "prior fields preserved on partial upsert"
    );

    // Filters narrow the aggregate feed.
    assert_eq!(
        store
            .list_generation_metrics(None, Some("qwen_image"), Some("q8"), 100)
            .expect("filtered list")
            .len(),
        1
    );
    assert!(store
        .list_generation_metrics(None, Some("no-such-model"), None, 100)
        .expect("empty filtered list")
        .is_empty());
}

#[test]
fn job_lifecycle_create_claim_complete() {
    let store = store("lifecycle");
    register_image_worker(&store);

    let created = store
        .create_job(image_job(object(json!({ "prompt": "mist over hills" }))))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.status, JobStatus::Preparing);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-0"));

    let completed = store
        .update_job_progress(
            &claimed.id,
            ProgressUpdate {
                status: JobStatus::Completed,
                stage: ProgressStage::Completed,
                progress: 1.0,
                message: "Done".to_owned(),
                error: None,
                result: Some(object(json!({ "assetIds": ["asset-1"] }))),
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: Some("worker-1".to_owned()),
            },
        )
        .expect("progress updates");
    let worker = store.get_worker("worker-1").expect("worker loads");

    assert_eq!(completed.status, JobStatus::Completed);
    assert_eq!(completed.result, object(json!({ "assetIds": ["asset-1"] })));
    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
}

/// An Ideogram auto-caption image job (sc-9120). Created NON-claimable in `pending_caption`.
fn pending_caption_job(payload: Value) -> CreateJob {
    CreateJob {
        initial_status: Some(JobStatus::PendingCaption),
        ..image_job(object(payload))
    }
}

/// A `prompt_refine` magic-prompt job, mirroring what the API's caption watcher enqueues.
fn magic_prompt_job(prompt: &str, aspect_ratio: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::PromptRefine,
        project_id: None,
        project_name: None,
        payload: object(json!({
            "task": "magic_prompt",
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
        })),
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

/// sc-9120: a job created in `pending_caption` is persisted with that status/stage and is NOT
/// claimable by a worker (the claim SELECT is hard `where status='queued'`), so the caption can be
/// produced async without a worker ever seeing the un-rewritten prompt.
#[test]
fn pending_caption_job_is_not_claimable() {
    let store = store("pending-caption-not-claimable");
    register_image_worker(&store);
    let created = store
        .create_job(pending_caption_job(
            json!({ "prompt": "a fox", "model": "ideogram_4" }),
        ))
        .expect("job creates");
    assert_eq!(created.status, JobStatus::PendingCaption);
    assert_eq!(created.stage, ProgressStage::PendingCaption);

    // A worker must not be able to claim it while it is pending.
    assert!(
        store
            .claim_next_job("worker-1")
            .expect("claim runs")
            .is_none(),
        "a pending_caption job must never be claimed"
    );

    // It still counts as an in-flight (non-terminal) job in the queue summary.
    let summary = store.queue_summary().expect("summary");
    assert_eq!(
        summary.counts.get(&JobStatus::PendingCaption).copied(),
        Some(1)
    );
    assert!(summary.active_jobs.iter().any(|job| job.id == created.id));
}

/// sc-9120: promoting with a rewritten payload flips the job to `queued` (claimable) and replaces the
/// stored prompt with the rich caption; the worker then claims it normally.
#[test]
fn promote_pending_caption_rewrites_prompt_and_queues() {
    let store = store("promote-caption-rewrite");
    register_image_worker(&store);
    let created = store
        .create_job(pending_caption_job(
            json!({ "prompt": "a fox", "model": "ideogram_4" }),
        ))
        .expect("job creates");

    let mut new_payload = created.payload.clone();
    new_payload.insert(
        "prompt".to_owned(),
        Value::String(r#"{"compositional_deconstruction": {}}"#.to_owned()),
    );
    let promotion = store
        .promote_pending_caption_job(&created.id, Some(new_payload))
        .expect("promotes");
    assert!(promotion.promoted, "the pending job should be promoted");
    assert_eq!(promotion.job.status, JobStatus::Queued);
    assert_eq!(
        promotion.job.payload.get("prompt").and_then(Value::as_str),
        Some(r#"{"compositional_deconstruction": {}}"#)
    );

    // Now claimable.
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim runs")
        .expect("promoted job is claimable");
    assert_eq!(claimed.id, created.id);
}

/// sc-9120: promoting with `None` (expansion unavailable/timeout) degrades to `queued` with the
/// ORIGINAL prompt untouched — never leaving a stranded pending_caption row.
#[test]
fn promote_pending_caption_degrades_to_original_prompt() {
    let store = store("promote-caption-degrade");
    let created = store
        .create_job(pending_caption_job(
            json!({ "prompt": "a fox", "model": "ideogram_4" }),
        ))
        .expect("job creates");
    let promotion = store
        .promote_pending_caption_job(&created.id, None)
        .expect("promotes");
    assert!(promotion.promoted);
    assert_eq!(promotion.job.status, JobStatus::Queued);
    assert_eq!(
        promotion.job.payload.get("prompt").and_then(Value::as_str),
        Some("a fox"),
        "the original prompt must be preserved on degrade"
    );
}

/// sc-9120: the promotion is race-free — a job that was canceled while its caption was expanding is
/// NOT resurrected. The guarded UPDATE (`where status='pending_caption'`) matches nothing, so the
/// promotion reports `promoted = false` and the job stays canceled.
#[test]
fn promote_pending_caption_does_not_resurrect_a_canceled_job() {
    let store = store("promote-caption-canceled");
    let created = store
        .create_job(pending_caption_job(
            json!({ "prompt": "a fox", "model": "ideogram_4" }),
        ))
        .expect("job creates");
    // A pending_caption job cancels immediately (no worker to acknowledge), like `queued`.
    let canceled = store.cancel_job(&created.id).expect("cancels");
    assert_eq!(canceled.status, JobStatus::Canceled);

    let promotion = store
        .promote_pending_caption_job(&created.id, None)
        .expect("promotion runs");
    assert!(
        !promotion.promoted,
        "a canceled job must not be promoted back to queued"
    );
    assert_eq!(
        promotion.job.status,
        JobStatus::Canceled,
        "the job must stay canceled"
    );
}

/// sc-9120: a mid-flight API restart loses the caption watcher, so the startup recovery must flip any
/// stranded `pending_caption` row to `queued` (degraded to the original prompt) rather than stranding
/// it un-claimable. Active jobs still go to `interrupted`; a `queued` job is left alone.
#[test]
fn startup_recovers_stranded_pending_caption_to_queued() {
    let store = store("startup-recover-pending");
    register_image_worker(&store);
    let pending = store
        .create_job(pending_caption_job(
            json!({ "prompt": "a fox", "model": "ideogram_4" }),
        ))
        .expect("pending job creates");
    // A separate job taken into an ACTIVE status: it must be INTERRUPTED, not degraded — proving the
    // pending_caption recovery is distinct from (and runs alongside) the active-job interrupt sweep.
    let active = store
        .create_job(image_job(object(json!({ "prompt": "active" }))))
        .expect("active job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim runs")
        .expect("the queued job is claimed");
    assert_eq!(claimed.id, active.id);

    store
        .mark_interrupted_on_startup()
        .expect("startup recovery runs");

    let recovered = store.get_job(&pending.id).expect("pending job loads");
    assert_eq!(
        recovered.status,
        JobStatus::Queued,
        "a stranded pending_caption job must be recovered to queued"
    );
    assert_eq!(
        recovered.payload.get("prompt").and_then(Value::as_str),
        Some("a fox"),
        "recovery keeps the original prompt"
    );
    // The active job is interrupted by the same sweep.
    assert_eq!(
        store.get_job(&active.id).expect("active loads").status,
        JobStatus::Interrupted
    );
}

/// sc-9120: a repeated auto-caption (an impatient client re-POSTing the same image job) reuses an
/// in-flight refine job instead of stacking a fresh one; a terminal refine job is never reused.
#[test]
fn find_reusable_prompt_refine_matches_in_flight_only() {
    let store = store("reuse-refine-job");
    let created = store
        .create_job(magic_prompt_job("a fox on a beach", "16:9"))
        .expect("refine job creates");

    // An in-flight refine job with the same prompt+aspect is reused.
    let found = store
        .find_reusable_prompt_refine_job("a fox on a beach", "16:9")
        .expect("query runs")
        .expect("an in-flight refine job matches");
    assert_eq!(found.id, created.id);

    // A different prompt or aspect does not match.
    assert!(store
        .find_reusable_prompt_refine_job("a different scene", "16:9")
        .expect("query runs")
        .is_none());
    assert!(store
        .find_reusable_prompt_refine_job("a fox on a beach", "1:1")
        .expect("query runs")
        .is_none());

    // Once the refine job reaches a terminal state it is no longer reusable.
    store.cancel_job(&created.id).expect("cancels");
    assert!(store
        .find_reusable_prompt_refine_job("a fox on a beach", "16:9")
        .expect("query runs")
        .is_none());
}

/// sc-9120: a job can only be born `queued` or `pending_caption`; any other initial status is a
/// programmer error and is rejected rather than creating a job mid-lifecycle.
#[test]
fn create_job_rejects_illegal_initial_status() {
    let store = store("illegal-initial-status");
    let error = store
        .create_job(CreateJob {
            initial_status: Some(JobStatus::Running),
            ..image_job(object(json!({ "prompt": "a fox" })))
        })
        .expect_err("running is not a legal initial status");
    assert!(matches!(error, JobsStoreError::InvalidInitialStatus(_)));
}

/// sc-2086 — successive progress reports must ratchet the per-job peak GPU
/// stats up only, so a stale low sample later in the run can't clobber the
/// max. Also covers clamp-to-100 and the None-passthrough case for status-only
/// updates.
#[test]
fn progress_keeps_running_max_for_peak_gpu_meters() {
    let store = store("peak-gpu-meters");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(object(json!({ "prompt": "p" }))))
        .expect("job creates");
    store.claim_next_job("worker-1").expect("claim ok");

    fn progress(memory: Option<f64>, load: Option<f64>) -> ProgressUpdate {
        ProgressUpdate {
            status: JobStatus::Running,
            stage: ProgressStage::Running,
            progress: 0.5,
            message: "running".to_owned(),
            error: None,
            result: None,
            eta_seconds: None,
            peak_gpu_memory_pct: memory,
            peak_gpu_load_pct: load,
            backend: None,
            worker_id: Some("worker-1".to_owned()),
        }
    }

    let job = store
        .update_job_progress(&created.id, progress(Some(40.0), Some(60.0)))
        .expect("first sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(40.0)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(60.0)
    );

    // Higher samples ratchet up.
    let job = store
        .update_job_progress(&created.id, progress(Some(72.5), Some(85.0)))
        .expect("higher sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // Lower samples are ignored — peak stays at the previous max.
    let job = store
        .update_job_progress(&created.id, progress(Some(20.0), Some(10.0)))
        .expect("lower sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // None passes through (status-only update) and leaves peaks untouched.
    let job = store
        .update_job_progress(&created.id, progress(None, None))
        .expect("status-only update");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // Over-100 samples (rare but possible from buggy backends) clamp.
    let job = store
        .update_job_progress(&created.id, progress(Some(120.0), Some(150.0)))
        .expect("clamped sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(100.0)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(100.0)
    );
}

/// A job whose every progress update omits the peak fields (e.g. a CPU-only
/// utility worker, or a path where gpu_utilization() returned nothing) must
/// keep peak_gpu_memory_pct / peak_gpu_load_pct NULL across the whole
/// lifecycle — otherwise the snapshot diverges from a job that ran on a
/// peerless backend, breaking parity (sc-2086 fix-forward).
#[test]
fn progress_leaves_peaks_null_when_no_samples_arrive() {
    let store = store("peak-null-no-samples");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(object(json!({ "prompt": "p" }))))
        .expect("job creates");
    store.claim_next_job("worker-1").expect("claim ok");

    let progress_no_peaks = ProgressUpdate {
        status: JobStatus::Running,
        stage: ProgressStage::Running,
        progress: 0.5,
        message: "running".to_owned(),
        error: None,
        result: None,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: None,
        worker_id: Some("worker-1".to_owned()),
    };
    for _ in 0..3 {
        store
            .update_job_progress(&created.id, progress_no_peaks.clone())
            .expect("progress update");
    }
    let final_job = store.get_job(&created.id).expect("loads");
    assert!(final_job.peak_gpu_memory_pct.is_none());
    assert!(final_job.peak_gpu_load_pct.is_none());
}

/// sc-2087 — server-side job-title derivation populates the JobSnapshot.title
/// field per the design spec table. Front-end falls back to its own derivation
/// only when this is None, so the queue never displays a raw job id.
#[test]
fn job_snapshot_title_is_derived_from_payload() {
    let store = store("title-derivation");
    register_image_worker(&store);

    fn create(store: &JobsStore, job_type: JobType, payload: Value) -> String {
        store
            .create_job(CreateJob {
                job_type,
                project_id: Some("p".to_owned()),
                project_name: Some("P".to_owned()),
                payload: object(payload),
                requested_gpu: "auto".to_owned(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .expect("job creates")
            .id
    }

    let image_id = create(
        &store,
        JobType::ImageGenerate,
        json!({ "prompt": "a sunset over the mountains" }),
    );
    let lora_train_id = create(
        &store,
        JobType::LoraTrain,
        json!({ "loraName": "kelsie-v3" }),
    );
    let caption_id = create(
        &store,
        JobType::TrainingCaption,
        json!({ "datasetName": "kelsie-set" }),
    );
    let video_id = create(
        &store,
        JobType::VideoGenerate,
        json!({ "prompt": "slow push-in on a foggy lighthouse" }),
    );
    let character_id_job = create(
        &store,
        JobType::ImageGenerate,
        json!({ "prompt": "ignored", "characterId": "char-1", "characterName": "Aria" }),
    );
    let lora_import_id = create(
        &store,
        JobType::LoraImport,
        json!({ "loraName": "detail_lora" }),
    );
    let model_download_id = create(
        &store,
        JobType::ModelDownload,
        json!({ "modelName": "Z-Image Turbo" }),
    );
    let prompt_refine_id = create(
        &store,
        JobType::PromptRefine,
        json!({ "prompt": "make it better please" }),
    );
    let unnamed_lora_id = create(&store, JobType::LoraTrain, json!({}));
    let person_detect_id = create(&store, JobType::PersonDetect, json!({}));

    let title = |id: &str| store.get_job(id).expect("loads").title.clone();
    assert_eq!(
        title(&image_id).as_deref(),
        Some("Generate Image — a sunset over the mountains"),
    );
    assert_eq!(
        title(&lora_train_id).as_deref(),
        Some("Training Run — kelsie-v3"),
    );
    assert_eq!(
        title(&caption_id).as_deref(),
        Some("Dataset Captioning — kelsie-set"),
    );
    assert_eq!(
        title(&video_id).as_deref(),
        Some("Generate Video — slow push-in on a foggy lighthouse"),
    );
    assert_eq!(
        title(&character_id_job).as_deref(),
        Some("Character Turnaround — Aria"),
    );
    assert_eq!(
        title(&lora_import_id).as_deref(),
        Some("LoRA Import — detail_lora"),
    );
    assert_eq!(
        title(&model_download_id).as_deref(),
        Some("Model Import — Z-Image Turbo"),
    );
    assert_eq!(
        title(&prompt_refine_id).as_deref(),
        Some("Prompt Refine — make it better please"),
    );
    assert_eq!(
        title(&unnamed_lora_id).as_deref(),
        Some("Training Run — (unnamed LoRA)"),
    );
    // person_detect (and other types without a meaningful subject) intentionally
    // return None so the frontend can fall back to its own derivation.
    assert_eq!(title(&person_detect_id), None);
}

/// Long image-generation prompts are truncated on a word boundary with an
/// ellipsis so the title doesn't blow out the queue row.
#[test]
fn job_snapshot_title_truncates_long_prompts() {
    let store = store("title-truncation");
    register_image_worker(&store);
    // 100 chars of "a " repeating, well over the 80-char cap.
    let long_prompt = "a ".repeat(60);
    let id = store
        .create_job(CreateJob {
            job_type: JobType::ImageGenerate,
            project_id: Some("p".to_owned()),
            project_name: Some("P".to_owned()),
            payload: object(json!({ "prompt": long_prompt })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("job creates")
        .id;
    let title = store.get_job(&id).expect("loads").title.unwrap();
    assert!(title.starts_with("Generate Image — "));
    assert!(
        title.ends_with("…"),
        "title should end with ellipsis: {title}"
    );
    assert!(title.len() < 110, "title should be short: {title}");
}

#[test]
fn non_gpu_jobs_can_claim_while_gpu_is_busy() {
    let store = store("non-gpu-claim");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::ImageGenerate,
                WorkerCapability::ModelDownload,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    let gpu_job = store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let download_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelDownload,
            project_id: None,
            project_name: None,
            payload: object(json!({ "repo": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("download job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("first claim succeeds")
            .expect("first job")
            .id,
        gpu_job.id
    );
    let second = store
        .claim_next_job("worker-1")
        .expect("second claim succeeds")
        .expect("second job");
    assert_eq!(second.id, download_job.id);
    assert_eq!(second.assigned_gpu.as_deref(), Some("cpu"));
}

#[test]
fn model_convert_can_claim_while_gpu_is_busy() {
    // sc-1629: model_convert is declared non-GPU (NON_GPU_JOB_TYPES) and the
    // worker/UI treat it that way, but the dispatch SQL used to omit it from its
    // non-GPU lists — so a queued model_convert would be gated behind GPU work.
    // It must claim on the CPU lane even while a GPU job is active on the worker.
    let store = store("model-convert-claim");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::ImageGenerate,
                WorkerCapability::ModelConvert,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    let gpu_job = store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let convert_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelConvert,
            project_id: None,
            project_name: None,
            payload: object(json!({ "model": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("convert job creates");

    // First claim takes the GPU job; it is now active on gpu-0.
    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("first claim succeeds")
            .expect("first job")
            .id,
        gpu_job.id
    );
    // With a GPU job active only non-GPU work is claimable; model_convert must
    // still claim and land on the CPU lane.
    let second = store
        .claim_next_job("worker-1")
        .expect("second claim succeeds")
        .expect("second job");
    assert_eq!(second.id, convert_job.id);
    assert_eq!(second.assigned_gpu.as_deref(), Some("cpu"));
}

#[test]
fn claim_skips_jobs_not_supported_by_worker_capabilities() {
    let store = store("capabilities");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ModelDownload],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let download_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelDownload,
            project_id: None,
            project_name: None,
            payload: object(json!({ "repo": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("download job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        download_job.id
    );
}

#[test]
fn claim_finds_compatible_job_behind_large_incompatible_prefix() {
    // sc-1630: a worker must still claim a compatible job even when far more than the
    // old 50-row query cap of incompatible jobs precede it in the queue — otherwise a
    // specialized/utility worker sits idle behind a long incompatible prefix.
    let store = store("starvation");
    store
        .register_worker(RegisterWorker {
            worker_id: "downloader".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ModelDownload],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    // 60 image jobs the worker cannot run (no ImageGenerate capability), enqueued first
    // so they fill the front of the created_at ordering (well past the old limit 50).
    for index in 0..60 {
        let prompt = format!("incompatible {index}");
        store
            .create_job(image_job(object(json!({ "prompt": prompt }))))
            .expect("incompatible job creates");
    }
    let download_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelDownload,
            project_id: None,
            project_name: None,
            payload: object(json!({ "repo": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("download job creates");

    let claimed = store
        .claim_next_job("downloader")
        .expect("claim succeeds")
        .expect("compatible job claimed despite the incompatible prefix");
    assert_eq!(claimed.id, download_job.id);
}

#[test]
fn real_lora_train_requires_execute_capability() {
    let store = store("lora-train-execute-routing");
    // A GPU worker that can validate dry-run plans but lacks the inference backend
    // advertises lora_train but not lora_train_execute.
    store
        .register_worker(RegisterWorker {
            worker_id: "dry-only".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let real = store
        .create_job(lora_train_job("auto", false))
        .expect("real training job creates");

    // The dry-run-only worker must not claim the real job; it stays queued for a
    // backend-capable worker instead of being claimed and failed.
    assert!(store
        .claim_next_job("dry-only")
        .expect("claim succeeds")
        .is_none());

    store
        .register_worker(RegisterWorker {
            worker_id: "trainer".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::Gpu,
                WorkerCapability::LoraTrain,
                WorkerCapability::LoraTrainExecute,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    assert_eq!(
        store
            .claim_next_job("trainer")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        real.id
    );
}

#[test]
fn dry_run_lora_train_does_not_require_execute_capability() {
    let store = store("lora-train-dry-routing");
    store
        .register_worker(RegisterWorker {
            worker_id: "dry-only".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let dry = store
        .create_job(lora_train_job("auto", true))
        .expect("dry-run training job creates");

    assert_eq!(
        store
            .claim_next_job("dry-only")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        dry.id
    );
}

#[test]
fn training_progress_stages_persist_under_running_and_reject_unknown_status() {
    let store = store("training-progress-stages");
    let job = store
        .create_job(lora_train_job("auto", false))
        .expect("training job creates");

    // The trainer reports caching/training/checkpointing stages under the running
    // status; all must be accepted and persisted, not rejected as invalid.
    for (stage, label) in [
        (ProgressStage::CachingLatents, "caching_latents"),
        (ProgressStage::Training, "training"),
        (ProgressStage::Checkpointing, "checkpointing"),
    ] {
        let updated = store
            .update_job_progress(
                &job.id,
                ProgressUpdate {
                    status: JobStatus::Running,
                    stage,
                    progress: 0.5,
                    message: "training".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: None,
                },
            )
            .expect("running status with a training stage is accepted");
        assert_eq!(updated.status, JobStatus::Running);
        assert_eq!(updated.stage.as_str(), label);
    }

    // A non-contract status like "caching" (an earlier kernel bug) must be rejected
    // rather than silently persisted.
    let error = store
        .update_job_progress(
            &job.id,
            ProgressUpdate {
                status: JobStatus::Unknown("caching".to_owned()),
                stage: ProgressStage::CachingLatents,
                progress: 0.5,
                message: "caching".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: None,
            },
        )
        .expect_err("an unknown status is rejected");
    assert!(matches!(error, JobsStoreError::InvalidStatus(_)));
}

#[test]
fn training_progress_merges_latest_sample_batches_into_history() {
    let store = store("training-sample-history");
    let job = store
        .create_job(lora_train_job("auto", false))
        .expect("training job creates");

    for (step, path) in [
        (250, "training/job-1/samples/step-000250/front.png"),
        (500, "training/job-1/samples/step-000500/front.png"),
    ] {
        store
            .update_job_progress(
                &job.id,
                ProgressUpdate {
                    status: JobStatus::Running,
                    stage: ProgressStage::Rendering,
                    progress: 0.5,
                    message: format!("Rendered samples at step {step}."),
                    error: None,
                    result: Some(object(json!({
                        "latestTrainingSamples": [{
                            "step": step,
                            "prompt": "front",
                            "relativePath": path,
                        }]
                    }))),
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: None,
                },
            )
            .expect("sample progress updates");
    }

    let updated = store.get_job(&job.id).expect("job loads");
    let samples = updated
        .result
        .get("trainingSamples")
        .and_then(Value::as_array)
        .expect("training sample history is present");

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0]["step"], json!(250));
    assert_eq!(samples[1]["step"], json!(500));
    assert_eq!(
        updated.result["latestTrainingSamples"][0]["relativePath"],
        json!("training/job-1/samples/step-000500/front.png")
    );
}

#[test]
fn gpu_generation_jobs_reject_cpu_requested_gpu() {
    let store = store("gpu-jobs-reject-cpu");

    let error = store
        .create_job(CreateJob {
            requested_gpu: " CPU ".to_owned(),
            ..image_job(Map::new())
        })
        .expect_err("cpu requestedGpu should be rejected");

    assert!(matches!(error, JobsStoreError::InvalidRequestedGpu(_)));
    assert!(error.to_string().contains("cannot target CPU workers"));
}

#[test]
fn cpu_worker_cannot_claim_auto_gpu_generation_job_even_with_capability() {
    let store = store("cpu-cannot-claim-auto-gpu-job");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "CPU".to_owned(),
            gpu_name: Some("CPU inference worker".to_owned()),
            capabilities: vec![WorkerCapability::Cpu, WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");

    assert!(store
        .claim_next_job("worker-cpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn auto_claim_prefers_job_matching_loaded_model() {
    let store = store("loaded-model-preference");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec![
                "z_image_turbo".to_owned(),
                "Tongyi-MAI/Z-Image-Turbo".to_owned(),
            ],
            utilization: None,
        })
        .expect("worker registers");
    let other_model_job = store
        .create_job(image_job(object(json!({ "model": "qwen_image_edit" }))))
        .expect("other model job creates");
    let warm_model_job = store
        .create_job(image_job(object(json!({ "model": "z_image_turbo" }))))
        .expect("warm model job creates");

    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, warm_model_job.id);
    assert_eq!(
        store
            .get_job(&other_model_job.id)
            .expect("other model job loads")
            .status,
        JobStatus::Queued
    );
}

#[test]
fn loaded_model_preference_does_not_skip_explicit_gpu_job() {
    let store = store("loaded-model-explicit-gpu");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec!["z_image_turbo".to_owned()],
            utilization: None,
        })
        .expect("worker registers");
    let explicit_job = store
        .create_job(CreateJob {
            requested_gpu: "gpu-0".to_owned(),
            ..image_job(object(json!({ "model": "qwen_image_edit" })))
        })
        .expect("explicit job creates");
    store
        .create_job(image_job(object(json!({ "model": "z_image_turbo" }))))
        .expect("warm model job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        explicit_job.id
    );
}

#[test]
fn explicit_gpu_job_beats_younger_warm_auto_match() {
    let store = store("explicit-gpu-before-warm-auto");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec!["model-x".to_owned()],
            utilization: None,
        })
        .expect("worker registers");
    let auto_other = store
        .create_job(image_job(object(json!({ "model": "model-y" }))))
        .expect("auto other job creates");
    let explicit_job = store
        .create_job(CreateJob {
            requested_gpu: "gpu-0".to_owned(),
            ..image_job(object(json!({ "model": "model-y" })))
        })
        .expect("explicit job creates");
    let warm_auto = store
        .create_job(image_job(object(json!({ "model": "model-x" }))))
        .expect("warm auto job creates");

    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, explicit_job.id);
    assert_eq!(
        store
            .get_job(&auto_other.id)
            .expect("auto other job loads")
            .status,
        JobStatus::Queued
    );
    assert_eq!(
        store
            .get_job(&warm_auto.id)
            .expect("warm auto job loads")
            .status,
        JobStatus::Queued
    );
}

#[test]
fn auto_gpu_claim_defers_to_less_loaded_compatible_worker() {
    let store = store("auto-gpu-utilization-preference");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-loaded".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("Loaded GPU".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(24_000),
                memory_used_mb: Some(22_000),
                memory_free_mb: Some(2_000),
                gpu_load_percent: Some(92.0),
            }),
        })
        .expect("loaded worker registers");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-idle".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: Some("Idle GPU".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(24_000),
                memory_used_mb: Some(4_000),
                memory_free_mb: Some(20_000),
                gpu_load_percent: Some(8.0),
            }),
        })
        .expect("idle worker registers");
    let job = store
        .create_job(image_job(object(json!({ "prompt": "mist" }))))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-loaded")
        .expect("loaded claim succeeds")
        .is_none());
    let claimed = store
        .claim_next_job("worker-idle")
        .expect("idle claim succeeds")
        .expect("idle worker claims");

    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-1"));
}

#[test]
fn cpu_utility_worker_does_not_claim_gpu_generation_job() {
    let store = store("cpu-utility-no-gpu-jobs");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "cpu".to_owned(),
            gpu_name: Some("CPU utility worker".to_owned()),
            capabilities: vec![
                WorkerCapability::Cpu,
                WorkerCapability::ModelDownload,
                WorkerCapability::LoraImport,
                WorkerCapability::FrameExtract,
                WorkerCapability::TimelineExport,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");

    assert!(store
        .claim_next_job("worker-cpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn idle_heartbeat_interrupts_previous_heartbeated_job() {
    let store = store("idle-heartbeat");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);

    // The owning worker reports at least one heartbeat for the job (records
    // last_heartbeat_at), so a later idle heartbeat is a genuine restart and
    // must reclaim the now-orphaned active job.
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Busy,
            current_job_id: Some(created.id.clone()),
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("busy heartbeat succeeds");

    let worker = store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("heartbeat succeeds");
    let job = store.get_job(&created.id).expect("job loads");

    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
    assert_eq!(job.status, JobStatus::Interrupted);
    assert_eq!(job.worker_id, None);
}

#[test]
fn startup_interrupt_returns_post_update_job_snapshots() {
    let store = store("startup-interrupt-post-update");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");
    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.status, JobStatus::Preparing);

    let interrupted = store
        .mark_interrupted_on_startup()
        .expect("startup interrupt succeeds");
    let persisted = store.get_job(&created.id).expect("job loads");
    let worker = store.get_worker("worker-1").expect("worker loads");

    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0].id, created.id);
    assert_eq!(interrupted[0].status, JobStatus::Interrupted);
    assert_eq!(interrupted[0].stage, ProgressStage::Interrupted);
    assert_eq!(interrupted[0].worker_id, None);
    assert_eq!(persisted.status, JobStatus::Interrupted);
    assert_eq!(persisted.stage, ProgressStage::Interrupted);
    assert_eq!(worker.status, WorkerStatus::Offline);
    assert_eq!(worker.current_job_id, None);
}

#[test]
fn startup_interrupt_collects_every_active_status() {
    // sc-8896 / F-094: mark_interrupted_on_startup selects the in-flight jobs via
    // list_jobs_by_status_on_connection(ACTIVE_STATUSES), now a single
    // `status in (...)` scan instead of a per-status loop. This pins that the fold
    // still gathers jobs sitting in DIFFERENT active statuses (running vs saving),
    // not just one, and leaves a queued (non-active) job alone.
    let store = store("startup-interrupt-all-statuses");
    register_image_worker(&store);
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-2".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: Some("GPU 1".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("second worker registers");

    // Job A -> running (worker-1); Job B -> saving (worker-2). Both are active but
    // in distinct statuses, so a per-status miss would drop one.
    let running = store
        .create_job(image_job(Map::new()))
        .expect("job A creates");
    store.claim_next_job("worker-1").expect("worker-1 claims A");
    store
        .update_job_progress(
            &running.id,
            ProgressUpdate {
                status: JobStatus::Running,
                stage: ProgressStage::Running,
                progress: 0.4,
                message: "running".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: Some("worker-1".to_owned()),
            },
        )
        .expect("A -> running");

    let saving = store
        .create_job(image_job(Map::new()))
        .expect("job B creates");
    store.claim_next_job("worker-2").expect("worker-2 claims B");
    store
        .update_job_progress(
            &saving.id,
            ProgressUpdate {
                status: JobStatus::Saving,
                stage: ProgressStage::Saving,
                progress: 0.9,
                message: "saving".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: Some("worker-2".to_owned()),
            },
        )
        .expect("B -> saving");

    // A third job stays queued (not an active status) and must be left untouched.
    let queued = store
        .create_job(image_job(Map::new()))
        .expect("job C creates");

    let interrupted = store
        .mark_interrupted_on_startup()
        .expect("startup interrupt succeeds");
    let interrupted_ids = interrupted
        .iter()
        .map(|job| job.id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        interrupted.len(),
        2,
        "both active jobs across running+saving are collected, got {interrupted_ids:?}"
    );
    assert!(interrupted_ids.contains(&running.id.as_str()));
    assert!(interrupted_ids.contains(&saving.id.as_str()));
    for job in &interrupted {
        assert_eq!(job.status, JobStatus::Interrupted);
    }
    // The queued job is untouched — not active, so not swept.
    let queued = store.get_job(&queued.id).expect("queued job loads");
    assert_eq!(queued.status, JobStatus::Queued);
}

#[test]
fn signal_death_fails_active_job_with_attributed_error() {
    // sc-4881: a worker hard-killed by SIGKILL/OOM can't report its own death, so
    // the supervisor attributes it. The worker's active job must become a real
    // FAILURE (with an actionable, signal-attributed error), not a heartbeat-sweep
    // `interrupted` that reads as a frozen progress bar. sc-5567: the remediation
    // must fit the dead job's kind — an image job points at count/resolution, NOT the
    // training-only gradient-checkpointing hint.
    let store = store("signal-death-fails-job");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");
    assert_eq!(claimed.id, created.id);

    let failed = store
        .fail_worker_job_terminated("worker-1", Some(9), None)
        .expect("signal death recorded")
        .expect("the worker's active job is failed");

    assert_eq!(failed.id, created.id);
    assert_eq!(failed.status, JobStatus::Failed);
    assert_eq!(failed.worker_id, None);
    let error = failed.error.clone().unwrap_or_default();
    assert!(
        error.contains("signal 9 (SIGKILL)")
            && error.contains("out-of-memory")
            && error.contains("image count or resolution"),
        "error should attribute the OOM SIGKILL and give image-job remediation, got {error:?}"
    );
    // sc-5567: the training-only hint must not leak onto an image job.
    assert!(
        !error.contains("Gradient Checkpointing"),
        "image-job OOM must not surface the training gradient-checkpointing hint, got {error:?}"
    );

    // The worker is released so the UI never shows it pinned to the dead job.
    let worker = store.get_worker("worker-1").expect("worker loads");
    assert_eq!(worker.current_job_id, None);
    assert_eq!(worker.status, WorkerStatus::Offline);
}

#[test]
fn signal_death_with_no_active_job_is_a_noop() {
    // A worker that dies idle between jobs has nothing to fail (sc-4881).
    let store = store("signal-death-idle-worker");
    register_image_worker(&store);

    let failed = store
        .fail_worker_job_terminated("worker-1", Some(11), None)
        .expect("signal death recorded");

    assert!(failed.is_none(), "an idle worker death fails no job");
}

#[test]
fn non_signal_exit_fails_active_job_with_exit_code_error() {
    // sc-6320: a worker child that exited on its own with a non-zero status (e.g. a
    // Rust panic that unwound to exit 101) is just as dead as a signaled one, but
    // the supervisor reports an exit code, not a signal. It must still become a real
    // FAILURE that names the exit code, not a heartbeat-sweep `interrupted` that
    // reads as a frozen progress bar.
    let store = store("non-signal-exit-fails-job");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");
    assert_eq!(claimed.id, created.id);

    let failed = store
        .fail_worker_job_terminated("worker-1", None, Some(101))
        .expect("termination recorded")
        .expect("the worker's active job is failed");

    assert_eq!(failed.id, created.id);
    assert_eq!(failed.status, JobStatus::Failed);
    assert_eq!(failed.worker_id, None);
    let error = failed.error.clone().unwrap_or_default();
    assert!(
        error.contains("panicked") && error.contains("101"),
        "a code-101 exit must self-name the panic, got {error:?}"
    );
    // A non-signal exit is not a signal — it must not claim to be one.
    assert!(
        !error.contains("signal"),
        "a non-signal exit must not report a signal, got {error:?}"
    );

    // The worker is released, exactly as on a signal death.
    let worker = store.get_worker("worker-1").expect("worker loads");
    assert_eq!(worker.current_job_id, None);
    assert_eq!(worker.status, WorkerStatus::Offline);
}

#[test]
fn idle_heartbeat_does_not_interrupt_just_claimed_job() {
    // A job claimed by one worker incarnation must survive an idle heartbeat
    // (currentJobId=null) that races the claim — e.g. from another process
    // sharing the same worker_id, or a restart firing its first idle heartbeat
    // before any progress is reported. Without a recorded heartbeat there is no
    // evidence the job was abandoned, so it must not be interrupted here.
    let store = store("idle-heartbeat-race");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);

    let worker = store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("heartbeat succeeds");
    let job = store.get_job(&created.id).expect("job loads");

    assert_eq!(worker.status, WorkerStatus::Idle);
    assert!(
        matches!(job.status, JobStatus::Preparing),
        "just-claimed job should stay active, got {:?}",
        job.status
    );
    assert_eq!(job.worker_id.as_deref(), Some("worker-1"));
}

#[test]
fn heartbeat_only_refreshes_a_job_the_reporting_worker_owns() {
    // sc-8873 / F-071: a heartbeat may only refresh the liveness timestamps of a
    // job the reporting worker actually owns. A stale/second worker that reports
    // someone else's `current_job_id` must NOT bump last_heartbeat_at — otherwise
    // it keeps the job looking alive and the time-based stale sweep can never
    // reclaim it. The owning worker's heartbeat still refreshes the job.
    let store = store("heartbeat-ownership");
    register_image_worker(&store);
    // A second, distinct worker that does not own the claimed job.
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-2".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: Some("GPU 1".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("second worker registers");

    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");
    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.worker_id.as_deref(), Some("worker-1"));

    // The owning worker records a first heartbeat, establishing a baseline
    // last_heartbeat_at we can watch for (non-)refresh.
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Busy,
            current_job_id: Some(created.id.clone()),
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("owner heartbeat succeeds");
    let baseline = store.get_job(&created.id).expect("job loads");
    let baseline_heartbeat = baseline
        .last_heartbeat_at
        .clone()
        .expect("owner heartbeat recorded last_heartbeat_at");

    // A NON-owning worker heartbeats the same job id. It must be a no-op on the
    // job's liveness — the timestamps stay exactly where the owner left them, and
    // the job keeps its owner. (Timestamps are second-granular, so equality is a
    // faithful "did not touch" assertion regardless of wall-clock drift.)
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-2".to_owned(),
            status: WorkerStatus::Busy,
            current_job_id: Some(created.id.clone()),
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("non-owner heartbeat still returns the worker snapshot");
    let after_intruder = store.get_job(&created.id).expect("job loads");
    assert_eq!(
        after_intruder.last_heartbeat_at.as_deref(),
        Some(baseline_heartbeat.as_str()),
        "a non-owning worker must not refresh the job's last_heartbeat_at"
    );
    assert_eq!(
        after_intruder.worker_id.as_deref(),
        Some("worker-1"),
        "a non-owning heartbeat must not steal or clear ownership"
    );

    // The owner can still refresh the job it owns — and the refresh must actually
    // ADVANCE last_heartbeat_at, not just leave a non-null value (a no-op owner
    // heartbeat would still read as `is_some()`). Timestamps are second-granular
    // and the store stamps `utc_now()` with no injectable clock, so a real sleep
    // would need to cross a whole-second boundary to be observable — flaky and
    // slow. Instead, deterministically age the stored last_heartbeat_at back to a
    // known OLD baseline via a test-only UPDATE, then require the owner heartbeat
    // to stamp a strictly greater (current-wall-clock) value.
    let old_baseline = "2000-01-01T00:00:00Z";
    {
        let connection = Connection::open(store.db_path()).expect("db opens");
        let updated = connection
            .execute(
                "update jobs set last_heartbeat_at = ?1 where id = ?2",
                params![old_baseline, created.id],
            )
            .expect("ages the stored heartbeat to a known baseline");
        assert_eq!(updated, 1, "exactly the target job's heartbeat is aged");
    }
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Busy,
            current_job_id: Some(created.id.clone()),
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("owner heartbeat succeeds");
    let owner_refreshed = store.get_job(&created.id).expect("job loads");
    let refreshed_heartbeat = owner_refreshed
        .last_heartbeat_at
        .as_deref()
        .expect("the owning worker's heartbeat keeps refreshing the job it owns");
    assert!(
        refreshed_heartbeat > old_baseline,
        "the owning worker's heartbeat must ADVANCE last_heartbeat_at past the \
         aged baseline, not leave it unchanged: got {refreshed_heartbeat:?}"
    );
}

#[test]
fn retry_job_is_capped() {
    let store = store("retry-cap");
    let job = store
        .create_job(CreateJob {
            attempts: MAX_JOB_ATTEMPTS,
            ..image_job(Map::new())
        })
        .expect("job creates");

    assert!(matches!(
        store.retry_job(
            &job.id,
            RetryJob {
                payload_changes: Map::new(),
            },
        ),
        Err(JobsStoreError::RetryLimit {
            max_attempts: MAX_JOB_ATTEMPTS
        })
    ));
}

#[test]
fn cancel_retry_and_duplicate_preserve_python_metadata_shapes() {
    let store = store("cancel-retry-duplicate");
    let original = store
        .create_job(image_job(object(json!({ "prompt": "mist over hills" }))))
        .expect("job creates");

    let canceled = store.cancel_job(&original.id).expect("job cancels");
    assert_eq!(canceled.status, JobStatus::Canceled);
    assert_eq!(canceled.stage, ProgressStage::Canceled);
    assert!(canceled.cancel_requested);
    assert_eq!(canceled.progress.as_f64(), Some(1.0));
    assert!(canceled.completed_at.is_some());
    assert!(canceled.canceled_at.is_some());

    let retry = store
        .retry_job(
            &canceled.id,
            RetryJob {
                payload_changes: Map::new(),
            },
        )
        .expect("job retries");
    assert_eq!(retry.source_job_id.as_deref(), Some(canceled.id.as_str()));
    assert_eq!(retry.attempts, canceled.attempts + 1);
    assert_eq!(retry.duplicate_of_job_id, None);
    assert_eq!(retry.payload, canceled.payload);

    let resume_retry = store
        .retry_job(
            &canceled.id,
            RetryJob {
                payload_changes: object(json!({ "downloadAction": "resume" })),
            },
        )
        .expect("resume retry creates");
    assert_eq!(
        resume_retry.source_job_id.as_deref(),
        Some(canceled.id.as_str())
    );
    assert_eq!(resume_retry.payload["prompt"], json!("mist over hills"));
    assert_eq!(resume_retry.payload["downloadAction"], json!("resume"));

    let duplicate = store
        .duplicate_job(
            &canceled.id,
            DuplicateJob {
                payload_changes: object(json!({ "prompt": "clear morning", "seed": 42 })),
                requested_gpu: Some("gpu-1".to_owned()),
            },
        )
        .expect("job duplicates");
    assert_eq!(
        duplicate.duplicate_of_job_id.as_deref(),
        Some(canceled.id.as_str())
    );
    assert_eq!(duplicate.source_job_id, None);
    assert_eq!(duplicate.requested_gpu, "gpu-1");
    assert_eq!(duplicate.payload["prompt"], json!("clear morning"));
    assert_eq!(duplicate.payload["seed"], json!(42));
}

#[test]
fn stale_sweep_marks_worker_offline_and_job_interrupted() {
    let store = store("stale-sweep");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = ?1",
            params!["worker-1"],
        )
        .expect("worker timestamp updates");
    connection
        .execute(
            "update jobs set last_heartbeat_at = '2000-01-01T00:00:00Z' where id = ?1",
            params![created.id],
        )
        .expect("job timestamp updates");

    let sweep = store
        .mark_stale_workers_interrupted(1)
        .expect("sweep succeeds");

    assert_eq!(sweep.workers[0].status, WorkerStatus::Offline);
    assert_eq!(sweep.workers[0].current_job_id, None);
    assert_eq!(sweep.jobs[0].status, JobStatus::Interrupted);
    assert_eq!(sweep.jobs[0].worker_id, None);
    // sc-6320: the timeout error must be honest (the worker may have crashed, hung,
    // or lost its connection — not a confirmed death) and name the timeout window.
    let error = sweep.jobs[0].error.clone().unwrap_or_default();
    assert!(
        error.contains("No heartbeat from the worker for 1s")
            && error.contains("crashed, hung, or lost"),
        "stale-sweep error must state the ambiguity + timeout, got {error:?}"
    );
    assert_eq!(sweep.jobs[0].message, "Lost contact with the worker.");
}

#[test]
fn json_columns_use_python_compatible_sorted_key_order() {
    let store = store("json-order");
    let job = store
        .create_job(image_job(object(
            json!({ "z": 1, "a": { "b": 2, "a": 1 } }),
        )))
        .expect("job creates");

    let connection = Connection::open(store.db_path()).expect("db opens");
    let payload_json: String = connection
        .query_row(
            "select payload_json from jobs where id = ?1",
            params![job.id],
            |row| row.get(0),
        )
        .expect("payload json loads");

    assert_eq!(payload_json, r#"{"a":{"a":1,"b":2},"z":1}"#);
}

#[test]
fn invalid_progress_numbers_are_rejected() {
    let store = store("invalid-progress");
    let job = store
        .create_job(image_job(Map::new()))
        .expect("job creates");

    assert!(matches!(
        store.update_job_progress(
            &job.id,
            ProgressUpdate {
                status: JobStatus::Running,
                stage: ProgressStage::Running,
                progress: f64::NAN,
                message: "bad progress".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: None,
            },
        ),
        Err(JobsStoreError::InvalidNumber(field)) if field == "progress"
    ));
}

#[test]
fn elapsed_seconds_accepts_fractional_rfc3339_timestamps() {
    let store = store("fractional-time");
    let job = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            r#"
            update jobs
               set started_at = '2026-05-17T13:00:04.521Z',
                   completed_at = '2026-05-17T13:00:09.999Z'
             where id = ?1
            "#,
            params![job.id.clone()],
        )
        .expect("timestamps update");

    let loaded = store.get_job(&job.id).expect("job loads");

    assert_eq!(
        loaded.elapsed_seconds.and_then(|value| value.as_i64()),
        Some(5)
    );
}

fn lora_train_job(requested_gpu: &str, dry_run: bool) -> CreateJob {
    CreateJob {
        job_type: JobType::LoraTrain,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(json!({ "dryRun": dry_run, "plan": { "planVersion": 1 } })),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

#[test]
fn lora_train_rejects_cpu_requested_gpu() {
    let store = store("lora-train-rejects-cpu");

    let error = store
        .create_job(lora_train_job("cpu", true))
        .expect_err("cpu requestedGpu should be rejected for lora_train");

    assert!(matches!(error, JobsStoreError::InvalidRequestedGpu(_)));
    assert!(error.to_string().contains("cannot target CPU workers"));
}

#[test]
fn cpu_worker_cannot_claim_lora_train_even_with_capability() {
    let store = store("cpu-cannot-claim-lora-train");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "cpu".to_owned(),
            gpu_name: Some("CPU inference worker".to_owned()),
            capabilities: vec![WorkerCapability::Cpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    assert!(store
        .claim_next_job("worker-cpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn gpu_worker_with_capability_claims_lora_train() {
    let store = store("gpu-claims-lora-train");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-gpu".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let created = store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    let claimed = store
        .claim_next_job("worker-gpu")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.job_type, JobType::LoraTrain);
    assert_eq!(claimed.status, JobStatus::Preparing);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-0"));
}

#[test]
fn gpu_worker_without_training_capability_skips_lora_train() {
    let store = store("gpu-without-training-cap");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-gpu".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    assert!(store
        .claim_next_job("worker-gpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn create_job_with_id_uses_supplied_id() {
    let store = store("create-job-with-id");

    let job = store
        .create_job_with_id(
            "job_lora_train_fixture".to_owned(),
            lora_train_job("auto", true),
        )
        .expect("job creates with supplied id");

    assert_eq!(job.id, "job_lora_train_fixture");
    assert_eq!(
        store
            .get_job("job_lora_train_fixture")
            .expect("job loads")
            .job_type,
        JobType::LoraTrain
    );
}

// --- Epic 3018: MLX-vs-torch image-job routing (sc-3021) ---

fn register_gpu_worker(
    store: &JobsStore,
    worker_id: &str,
    gpu_id: &str,
    capabilities: Vec<WorkerCapability>,
) {
    store
        .register_worker(RegisterWorker {
            worker_id: worker_id.to_owned(),
            gpu_id: gpu_id.to_owned(),
            gpu_name: None,
            capabilities,
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
}

fn image_caps() -> Vec<WorkerCapability> {
    vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate]
}

fn image_job_with(payload: Value, requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::ImageGenerate,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

/// Capabilities a real image GPU worker advertises once it also serves the distinct
/// `image_edit` job type (sc-3513) — both the Python torch worker (`IMAGE_JOB_TYPES`)
/// and the macOS mlx worker (`gpu::mlx_gpu`) carry `image_edit` alongside `image_generate`.
fn image_edit_caps() -> Vec<WorkerCapability> {
    vec![
        WorkerCapability::Gpu,
        WorkerCapability::ImageGenerate,
        WorkerCapability::ImageEdit,
    ]
}

/// A `JobType::ImageEdit` job — "plain Image Edit" (Image Studio/Editor, epic 2427),
/// the sibling of [`image_job_with`] for the edit job type the bug missed (sc-3513).
fn image_edit_job_with(payload: Value, requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::ImageEdit,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

fn complete_job(store: &JobsStore, job_id: &str) {
    let worker_id = store.get_job(job_id).expect("job loads").worker_id;
    store
        .update_job_progress(
            job_id,
            ProgressUpdate {
                status: JobStatus::Completed,
                stage: ProgressStage::Completed,
                progress: 1.0,
                message: "Done".to_owned(),
                error: None,
                result: Some(Map::new()),
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id,
            },
        )
        .expect("job completes");
}

#[test]
fn mlx_eligible_image_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-routing-defer");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-eligible job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    // The mlx worker claims it and runs it in-process.
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn qwen_edit_image_job_defers_to_mlx_worker() {
    let store = store("mlx-routing-qwen-edit");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // Qwen-Image-Edit (sc-3397): an edit_image job with a source routes to the mlx worker.
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "qwen_image_edit_2511",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "make it a watercolor painting"
            }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers it to the idle mlx worker, which claims it.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn sensenova_understanding_jobs_defer_to_mlx_worker() {
    // sc-3905: SenseNova-U1 VQA (`image_vqa`) + Document-Studio interleave (`image_interleave`)
    // are served in-process via the concrete `T2iModel`. A worker that advertises the
    // understanding capabilities defers an eligible job to the idle mlx worker, which claims it.
    let understanding_caps = vec![
        WorkerCapability::Gpu,
        WorkerCapability::ImageVqa,
        WorkerCapability::ImageInterleave,
    ];
    let cases = [
        (
            "vqa",
            JobType::ImageVqa,
            json!({ "model": "sensenova_u1_8b", "sourceAssetId": "asset_1", "question": "what is this?" }),
        ),
        (
            "interleave",
            JobType::ImageInterleave,
            json!({ "model": "sensenova_u1_8b_fast", "prompt": "a short illustrated guide" }),
        ),
    ];
    for (label, job_type, payload) in cases {
        let store = store(&format!("mlx-routing-sensenova-{label}"));
        register_gpu_worker(&store, "worker-torch", "mps", understanding_caps.clone());
        register_gpu_worker(&store, "worker-mlx", "mlx", understanding_caps.clone());

        let job = store
            .create_job(CreateJob {
                job_type,
                project_id: Some("project-1".to_owned()),
                project_name: Some("Project 1".to_owned()),
                payload: object(payload),
                requested_gpu: "auto".to_owned(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .expect("job creates");

        // The torch worker defers it to the idle mlx worker, which claims it in-process.
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{label}: torch worker should defer to the idle mlx worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .expect("mlx claims the job");
        assert_eq!(claimed.id, job.id, "{label}");
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"), "{label}");
    }
}

#[test]
fn qwen_edit_lightning_image_job_defers_to_mlx_worker() {
    let store = store("mlx-routing-qwen-edit-lightning");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // Qwen-Image-Edit Lightning (sc-3398): the distilled id routes through the same
    // gate as the base edit ids — an edit_image job with a source goes to the mlx worker.
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "make it a watercolor painting"
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_torch_only_image_job() {
    let store = store("mlx-routing-exclude");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // edit_image on a Z-Image model is not a txt2img request → torch path only.
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "z_image_turbo",
                "mode": "edit_image",
                "referenceAssetId": "asset_1"
            }),
            "auto",
        ))
        .expect("job creates");

    // The mlx worker must not claim a torch-only image job.
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    // A torch worker is the home for it.
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_eligible_image_job_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-routing-fallback");
    // No mlx worker registered (Windows/Linux, or the mlx worker is down).
    register_gpu_worker(&store, "worker-torch", "cuda:0", image_caps());

    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // With no idle mlx worker, nothing defers — the torch worker is the fallback.
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

// epic 3482 / sc-3483 — macOS "MLX-required": the MPS worker never claims an MLX-eligible
// job, and a job no live `mlx` worker takes within the grace window fails terminal with
// `mlx_unavailable` rather than silently running on MPS. Ships behind a flag (default OFF);
// the sibling `*_falls_back_to_torch_*` tests above pin the OFF behaviour.

/// Backdate a job's `created_at` so the grace sweep treats it as having outlived the
/// window (mirrors how `stale_sweep_*` backdates `last_seen_at`).
fn backdate_job_created_at(store: &JobsStore, job_id: &str) {
    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            "update jobs set created_at = '2000-01-01T00:00:00Z' where id = ?1",
            params![job_id],
        )
        .expect("job created_at backdates");
}

/// sc-4208 / F-CORE-4: `queue_summary` must derive per-status counts and the
/// active-jobs list from the WHOLE table, not from a newest-500 cap. Before the
/// fix, once a project exceeded 500 jobs the counts silently undercounted and an
/// old still-queued job could fall out of the newest-500 window and vanish from
/// `active_jobs` entirely.
#[test]
fn queue_summary_counts_and_active_jobs_ignore_500_row_cap() {
    let store = store("queue-summary-cap");

    // One old queued job, backdated so it is the OLDEST row by created_at —
    // exactly the row a newest-500 window would evict first.
    let old_queued = store
        .create_job(image_job(object(json!({ "prompt": "oldest queued" }))))
        .expect("old job creates");
    backdate_job_created_at(&store, &old_queued.id);

    // 501 newer jobs so the table exceeds the 500-row cap; mark them completed
    // via raw SQL so the newest-500 window is entirely terminal.
    for i in 0..501 {
        store
            .create_job(image_job(object(
                json!({ "prompt": format!("completed {i}") }),
            )))
            .expect("job creates");
    }
    let connection = Connection::open(store.db_path()).expect("db opens");
    let updated = connection
        .execute(
            "update jobs set status = 'completed' where id != ?1",
            params![old_queued.id],
        )
        .expect("bulk completes newer jobs");
    assert_eq!(updated, 501, "exactly the newer jobs are completed");

    let summary = store.queue_summary().expect("summary computes");

    // Counts come from `group by` over all 502 rows, not a 500-row sample.
    assert_eq!(
        summary.counts.get(&JobStatus::Completed).copied(),
        Some(501)
    );
    assert_eq!(summary.counts.get(&JobStatus::Queued).copied(), Some(1));

    // The old queued job survives in active_jobs despite 501 newer terminal rows.
    assert_eq!(summary.active_jobs.len(), 1);
    assert_eq!(summary.active_jobs[0].id, old_queued.id);
}

#[test]
fn mlx_required_defers_eligible_job_even_with_no_idle_mlx_worker() {
    let store = store("mlx-required-defer");
    // Only an MPS worker is registered — no idle `mlx` worker to take the job. With the
    // flag OFF this is exactly the torch fallback; with it ON the MPS worker yields
    // unconditionally ("never MPS" on Mac).
    register_gpu_worker(&store, "worker-mps", "mps", image_caps());
    store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-mps", true)
        .expect("claim ok");
    assert!(
        claimed.is_none(),
        "MPS worker must not claim the MLX-eligible job when mlx is required"
    );
    let decision = decision.expect("a routing decision is reported");
    assert_eq!(decision.decision, "deferred_to_mlx");
}

#[test]
fn mlx_required_fails_stranded_eligible_job_when_no_live_mlx_worker() {
    let store = store("mlx-required-strand");
    register_gpu_worker(&store, "worker-mps", "mps", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");
    backdate_job_created_at(&store, &job.id);

    let failed = store.fail_stranded_mlx_jobs(true, 90).expect("sweep ok");
    assert_eq!(failed.len(), 1, "the stranded MLX-eligible job is failed");
    assert_eq!(failed[0].id, job.id);
    assert_eq!(failed[0].status, JobStatus::Failed);
    assert!(
        failed[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("mlx_unavailable"),
        "error names the mlx_unavailable cause: {:?}",
        failed[0].error
    );
    // The terminal transition is persisted, not just reported.
    assert_eq!(
        store.get_job(&job.id).expect("job loads").status,
        JobStatus::Failed
    );
}

#[test]
fn mlx_required_does_not_fail_when_a_live_mlx_worker_is_present() {
    let store = store("mlx-required-live");
    // A live `mlx` worker exists (just registered → recent heartbeat); the job waits for
    // it instead of being failed — covers the "mlx worker merely busy" case.
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");
    backdate_job_created_at(&store, &job.id);

    let failed = store.fail_stranded_mlx_jobs(true, 90).expect("sweep ok");
    assert!(failed.is_empty(), "a live mlx worker keeps the job queued");
    assert_eq!(
        store.get_job(&job.id).expect("job loads").status,
        JobStatus::Queued
    );
}

#[test]
fn fail_stranded_mlx_jobs_is_noop_when_not_required() {
    let store = store("mlx-required-off");
    register_gpu_worker(&store, "worker-mps", "mps", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");
    backdate_job_created_at(&store, &job.id);

    // Flag off (Windows/Linux/Docker, Mac pre-cutover): the sweep never fails anything.
    let failed = store.fail_stranded_mlx_jobs(false, 90).expect("sweep ok");
    assert!(failed.is_empty());
    assert_eq!(
        store.get_job(&job.id).expect("job loads").status,
        JobStatus::Queued
    );
}

#[test]
fn mlx_required_still_lets_mps_claim_a_non_eligible_model() {
    // 3483 only kills the MPS fallback for MLX-*eligible* jobs. A torch-only model is not
    // eligible, so it is NOT deferred and still runs on MPS — surfacing it as a loud
    // `mlx_unsupported` failure is sc-3484's job, not this slice's.
    let store = store("mlx-required-noneligible");
    register_gpu_worker(&store, "worker-mps", "mps", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({ "model": "pulid_flux_dev", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job_routed("worker-mps", true)
        .expect("claim ok")
        .0
        .expect("MPS claims the non-eligible job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

// epic 5483 / sc-5502 — the off-Mac candle twin of the MLX sweeps: the `candle_supported` oracle
// (inverse of the candle eligibility predicates) + `fail_unsupported_candle_jobs` (enforce) +
// `fail_stranded_candle_jobs` (grace). The candle worker is identified by the `candle` marker
// capability (`WorkerCapability::Unknown("candle")`), not a `gpu_id` — it runs on a real CUDA index.

fn register_candle_worker(store: &JobsStore, worker_id: &str) {
    register_gpu_worker(
        store,
        worker_id,
        "0",
        vec![
            WorkerCapability::Gpu,
            WorkerCapability::ImageGenerate,
            WorkerCapability::Unknown("candle".to_owned()),
        ],
    );
}

#[test]
fn candle_supported_accepts_eligible_and_in_process_jobs() {
    let store = store("candle-oracle-ok");
    // A candle-eligible txt2img → supported (consistent with candle routing by construction).
    let eligible = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    assert!(candle_supported(&eligible).is_ok());
    // In-process job types run off-Mac with zero torch.
    let download = job_of(&store, JobType::ModelDownload, json!({ "repo": "x/y" }));
    assert!(candle_supported(&download).is_ok());
    let refine = job_of(&store, JobType::PromptRefine, json!({ "prompt": "p" }));
    assert!(candle_supported(&refine).is_ok());
}

#[test]
fn candle_supported_rejects_unsupported_strict_pose() {
    // The sc-5968 case generalized: sdxl + poses has no candle strict-pose lane, so the candle/CUDA
    // flow can't serve it — it must fail loudly off-Mac, not silently render an unconditioned T2I.
    let store = store("candle-oracle-pose");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "sdxl", "prompt": "p", "advanced": { "poses": [{ "id": "p" }] } }),
    );
    let reason = candle_supported(&job).unwrap_err();
    assert_eq!(reason.model.as_deref(), Some("sdxl"));
    assert!(reason.feature.contains("strict-pose"));
    assert!(reason
        .candle_error_message()
        .starts_with("candle_unsupported:"));
}

#[test]
fn candle_supported_flags_a_torch_only_image_model() {
    let store = store("candle-oracle-torch-model");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "bernini_image", "prompt": "p" }),
    );
    let reason = candle_supported(&job).unwrap_err();
    assert_eq!(reason.model.as_deref(), Some("bernini_image"));
    assert!(reason
        .candle_error_message()
        .starts_with("candle_unsupported:"));
}

#[test]
fn candle_required_enforce_fails_unsupported_job() {
    let store = store("candle-enforce");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "sdxl", "prompt": "p", "advanced": { "poses": [{ "id": "p" }] } }),
    );
    // Warn mode (enforce = false): no-op, the job stays queued (still runs on torch).
    let warn = store
        .fail_unsupported_candle_jobs(true, false)
        .expect("sweep ok");
    assert!(warn.is_empty());
    assert_eq!(
        store.get_job(&job.id).expect("loads").status,
        JobStatus::Queued
    );
    // Off (candle not required): no-op even with enforce requested.
    assert!(store
        .fail_unsupported_candle_jobs(false, true)
        .expect("sweep ok")
        .is_empty());
    // Enforce: the unsupported job fails terminal with candle_unsupported.
    let failed = store
        .fail_unsupported_candle_jobs(true, true)
        .expect("sweep ok");
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0.id, job.id);
    assert_eq!(failed[0].0.status, JobStatus::Failed);
    assert!(failed[0]
        .0
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("candle_unsupported"));
    assert_eq!(
        store.get_job(&job.id).expect("loads").status,
        JobStatus::Failed
    );
}

#[test]
fn candle_required_enforce_leaves_eligible_job_queued() {
    // Partition: a candle-eligible job is Ok in the oracle, so the enforce sweep never touches it.
    let store = store("candle-enforce-eligible");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    let failed = store
        .fail_unsupported_candle_jobs(true, true)
        .expect("sweep ok");
    assert!(failed.is_empty());
    assert_eq!(
        store.get_job(&job.id).expect("loads").status,
        JobStatus::Queued
    );
}

#[test]
fn candle_required_fails_stranded_eligible_job_when_no_live_candle_worker() {
    let store = store("candle-strand");
    // A non-candle torch GPU worker exists, but no candle worker — the candle-eligible job strands.
    register_gpu_worker(&store, "worker-torch", "0", image_caps());
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    backdate_job_created_at(&store, &job.id);

    let failed = store.fail_stranded_candle_jobs(true, 90).expect("sweep ok");
    assert_eq!(
        failed.len(),
        1,
        "the stranded candle-eligible job is failed"
    );
    assert_eq!(failed[0].id, job.id);
    assert_eq!(failed[0].status, JobStatus::Failed);
    assert!(
        failed[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("candle_unavailable"),
        "error names the candle_unavailable cause: {:?}",
        failed[0].error
    );
    assert_eq!(
        store.get_job(&job.id).expect("job loads").status,
        JobStatus::Failed
    );
}

#[test]
fn candle_required_does_not_fail_when_a_live_candle_worker_is_present() {
    let store = store("candle-strand-live");
    // A live candle worker exists (just registered → recent heartbeat); the job waits for it
    // instead of being failed — covers the "candle worker merely busy" case.
    register_candle_worker(&store, "worker-candle");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    backdate_job_created_at(&store, &job.id);

    let failed = store.fail_stranded_candle_jobs(true, 90).expect("sweep ok");
    assert!(
        failed.is_empty(),
        "a live candle worker keeps the job queued"
    );
    assert_eq!(
        store.get_job(&job.id).expect("job loads").status,
        JobStatus::Queued
    );
}

#[test]
fn candle_stranded_sweep_partitions_and_is_noop_when_off() {
    let store = store("candle-strand-partition");
    // No candle worker. An UNSUPPORTED job (sdxl+poses) is not candle-eligible, so the stranded
    // sweep leaves it for the enforce sweep — the two partition the queue.
    let unsupported = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "sdxl", "prompt": "p", "advanced": { "poses": [{ "id": "p" }] } }),
    );
    backdate_job_created_at(&store, &unsupported.id);
    let failed = store.fail_stranded_candle_jobs(true, 90).expect("sweep ok");
    assert!(
        failed.is_empty(),
        "stranded sweep only fails candle-eligible jobs"
    );
    assert_eq!(
        store.get_job(&unsupported.id).expect("loads").status,
        JobStatus::Queued
    );
    // And the whole sweep is a no-op when candle is not required (today's off-Mac default).
    let eligible = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    backdate_job_created_at(&store, &eligible.id);
    assert!(store
        .fail_stranded_candle_jobs(false, 90)
        .expect("sweep ok")
        .is_empty());
    assert_eq!(
        store.get_job(&eligible.id).expect("loads").status,
        JobStatus::Queued
    );
}

// epic 3482 / sc-3484 — mac_rust_supported oracle (the inverse of the eligibility predicates)
// + the enforce sweep that fails unsupported jobs terminal with `mlx_unsupported`.

fn job_of(store: &JobsStore, job_type: JobType, payload: Value) -> JobSnapshot {
    store
        .create_job(CreateJob {
            job_type,
            project_id: Some("project-1".to_owned()),
            project_name: Some("Project 1".to_owned()),
            payload: object(payload),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("job creates")
}

#[test]
fn mac_rust_supported_accepts_eligible_and_mlx_agnostic_jobs() {
    let store = store("oracle-ok");
    // MLX-eligible generation → supported (consistent with routing by construction).
    let eligible = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );
    assert!(mac_rust_supported(&eligible).is_ok());
    // Chroma text-to-image (epic 3531 / sc-3843) is now MLX-eligible on every variant.
    for id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
        let chroma = job_of(
            &store,
            JobType::ImageGenerate,
            json!({ "model": id, "prompt": "p" }),
        );
        assert!(
            mac_rust_supported(&chroma).is_ok(),
            "{id} should be MLX-eligible"
        );
    }
    // MLX-agnostic job types run in-process with no Python torch dependency.
    let download = job_of(&store, JobType::ModelDownload, json!({ "repo": "x/y" }));
    assert!(mac_rust_supported(&download).is_ok());
    let refine = job_of(&store, JobType::PromptRefine, json!({ "prompt": "p" }));
    assert!(mac_rust_supported(&refine).is_ok());
}

#[test]
fn mac_capabilities_features_agree_with_the_rust_oracle() {
    // sc-4206 (F-CORE-2): a feature the Mac UI gates must agree with what
    // mac_rust_supported refuses for the corresponding job type — "what the UI hides
    // can never drift from what routing refuses". poseFromPhoto/PoseDetect was the
    // drift this guards (DWPose is ported to Rust, sc-3487).
    let store = store("capabilities-oracle-agreement");
    let features = mac_capabilities("darwin", true).features;
    for (feature, job_type) in [
        ("poseFromPhoto", JobType::PoseDetect),
        ("personDetect", JobType::PersonDetect),
    ] {
        let supported = features
            .get(feature)
            .unwrap_or_else(|| panic!("{feature} capability is present"))
            .supported;
        let job = job_of(&store, job_type, json!({}));
        assert_eq!(
            supported,
            mac_rust_supported(&job).is_ok(),
            "{feature} capability must agree with its mac_rust_supported job-type arm"
        );
    }
}

#[test]
fn mac_rust_supported_flags_an_unported_image_model_as_needing_a_port_epic() {
    let store = store("oracle-torch-model");
    // Every shipping image model now has a Rust/MLX engine and is in `MLX_ROUTED_MODELS`, so none
    // reaches the whole-model torch-only classifier anymore: z_image_edit (epic 3529 / sc-3923),
    // Kolors (epic 3090 / sc-3875), pulid_flux_dev (epic 3069 / sc-3344), instantid_realvisxl
    // (sc-3345), SenseNova-U1 (epic 3180 / sc-3900 + sc-3905), and finally Lens / Lens-Turbo
    // (epic 3164 / sc-5105) — the LAST whole-model torch-only image family. The torch-only gap path
    // now only fires for a hypothetical *unported* model id; lacking a dedicated port epic, the
    // oracle flags it `mlx_unsupported` and reports "needs an epic" (suggested_epic None — epic 3482
    // policy: file a porting epic + drop on Mac until it lands).
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "unported_image_model", "prompt": "p" }),
    );
    let reason = mac_rust_supported(&job).unwrap_err();
    assert_eq!(reason.model.as_deref(), Some("unported_image_model"));
    assert_eq!(reason.suggested_epic.as_deref(), None);
    assert!(reason.error_message().starts_with("mlx_unsupported:"));
}

#[test]
fn mac_rust_supported_instantid_full_surface_ok() {
    let store = store("oracle-instantid");
    // The full InstantID surface is native on Mac: identity (sc-3345), angle set (sc-3345),
    // pose-library mode + face-restore (sc-3381, #193 engine). All character_image + reference
    // shapes are supported.
    for advanced in [
        json!({}),
        json!({ "angleSet": true }),
        json!({ "poses": [{ "id": "a" }] }),
        json!({ "faceRestore": true }),
        json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
    ] {
        let job = job_of(
            &store,
            JobType::ImageGenerate,
            json!({
                "model": "instantid_realvisxl",
                "mode": "character_image",
                "referenceAssetId": "asset_1",
                "prompt": "p",
                "advanced": advanced,
            }),
        );
        assert!(
            mac_rust_supported(&job).is_ok(),
            "InstantID character_image should be MLX-supported"
        );
    }
}

#[test]
fn mac_rust_supported_bernini_image_t2i_and_i2i() {
    // Bernini still-image companion (epic 4699 / sc-5424): the `bernini_image` id routes its t2i +
    // i2i jobs to the in-process Rust worker (same `engine_id:"bernini"`, `frames:1`).
    let store = store("oracle-bernini-image");
    // Plain text-to-image → supported.
    let t2i = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "bernini_image", "prompt": "p" }),
    );
    assert!(
        mac_rust_supported(&t2i).is_ok(),
        "bernini_image t2i should be MLX-supported"
    );
    // i2i (`edit_image`) WITH a source → supported (the source becomes the engine's Reference).
    let i2i = job_of(
        &store,
        JobType::ImageGenerate,
        json!({
            "model": "bernini_image",
            "mode": "edit_image",
            "sourceAssetId": "asset_1",
            "prompt": "p",
        }),
    );
    assert!(
        mac_rust_supported(&i2i).is_ok(),
        "bernini_image i2i with a source should be MLX-supported"
    );
    // `edit_image` WITHOUT a source has nothing to edit → not routed (mirrors z_image_edit). The
    // oracle flags it rather than silently degrading to t2i against a dropped source.
    let i2i_no_source = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "bernini_image", "mode": "edit_image", "prompt": "p" }),
    );
    assert!(
        mac_rust_supported(&i2i_no_source).is_err(),
        "bernini_image edit_image without a source must not be MLX-eligible"
    );
}

#[test]
fn mac_rust_supported_names_qwen_strict_pose_and_lycoris() {
    let store = store("oracle-features");
    // Strict-pose ControlNet on Qwen is now Rust/MLX (epic 3401 / sc-3575).
    let pose = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "qwen_image", "prompt": "p", "advanced": { "poses": [{ "x": 1 }] } }),
    );
    assert!(mac_rust_supported(&pose).is_ok());
    // Third-party LyCORIS now applies on every MLX provider (epic 3641: core sc-3642/3643 +
    // SDXL/Wan/LTX sc-3671) → no longer a gap, runs on the Rust/MLX flow.
    let lycoris = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "sdxl", "prompt": "p", "loras": [{ "networkType": "lycoris" }] }),
    );
    assert!(mac_rust_supported(&lycoris).is_ok());
}

#[test]
fn mac_rust_supported_names_infra_job_types() {
    let store = store("oracle-infra");
    // Person detection + tracking are ported to the Rust worker (sc-3488 /
    // sc-3633/3634/3709): native-MLX YOLO11 + SORT/ByteTrack + SAM2 segmentation → both
    // supported. (replace_person end-to-end still needs epic 3040 — asserted below.)
    let person_detect = job_of(&store, JobType::PersonDetect, json!({}));
    assert!(mac_rust_supported(&person_detect).is_ok());
    let person_track = job_of(&store, JobType::PersonTrack, json!({}));
    assert!(mac_rust_supported(&person_track).is_ok());
    // replace_person → native Wan-VACE is supported on a replace-capable MLX video model
    // (sc-3521); a replace_person job on a model with no MLX video engine stays a torch gap.
    let replace_mlx = job_of(
        &store,
        JobType::PersonReplace,
        json!({ "model": "wan_2_2", "mode": "replace_person" }),
    );
    assert!(mac_rust_supported(&replace_mlx).is_ok());
    let replace = job_of(&store, JobType::PersonReplace, json!({}));
    let replace_reason = mac_rust_supported(&replace).unwrap_err();
    assert!(replace_reason
        .suggested_epic
        .as_deref()
        .is_some_and(|epic| epic.contains("epic 3040")));
    // DWPose pose detection is ported to the Rust worker (sc-3487) → supported.
    let pose = job_of(&store, JobType::PoseDetect, json!({}));
    assert!(mac_rust_supported(&pose).is_ok());
    // SCRFD kps extraction is native-MLX on the Rust worker (sc-4433) → supported.
    let kps = job_of(&store, JobType::KpsExtract, json!({}));
    assert!(mac_rust_supported(&kps).is_ok());
    // Real-ESRGAN upscaling is ported to the Rust worker (sc-3489): the default engine
    // (real-esrgan) is supported; the AuraSR engine is dropped on Mac (sc-3668) and off-Mac as an
    // offered engine (sc-5499). SeedVR2 is the one-step diffusion upscaler (epic 4811 / sc-4815) and
    // is also supported.
    let upscale = job_of(&store, JobType::ImageUpscale, json!({}));
    assert!(mac_rust_supported(&upscale).is_ok());
    let seedvr2 = job_of(
        &store,
        JobType::ImageUpscale,
        json!({ "engine": "seedvr2" }),
    );
    assert!(mac_rust_supported(&seedvr2).is_ok());
    let aura = job_of(
        &store,
        JobType::ImageUpscale,
        json!({ "engine": "aura-sr" }),
    );
    let aura_reason = mac_rust_supported(&aura).unwrap_err();
    assert!(aura_reason.feature.contains("AuraSR"));
    assert_eq!(aura_reason.suggested_epic.as_deref(), Some("sc-5499"));
    // JoyCaption dataset captioning is ported to the Rust/MLX worker (sc-3556).
    let caption = job_of(
        &store,
        JobType::TrainingCaption,
        json!({ "captioner": "joy_caption" }),
    );
    assert!(mac_rust_supported(&caption).is_ok());
    let unknown_caption = job_of(&store, JobType::TrainingCaption, json!({}));
    let reason = mac_rust_supported(&unknown_caption).unwrap_err();
    assert_eq!(reason.suggested_epic.as_deref(), Some("sc-3556"));
}

#[test]
fn mac_rust_supported_names_advanced_video_and_svd() {
    let store = store("oracle-video");
    // extend / bridge on a 14B Wan MoE engine: no `Keyframe` path → torch gap (sc-3522 / sc-3357).
    // The mode is derived from the job type, so a missing payload `mode` still classifies correctly.
    let extend = job_of(
        &store,
        JobType::VideoExtend,
        json!({ "model": "wan_2_2_t2v_14b" }),
    );
    assert_eq!(
        mac_rust_supported(&extend)
            .unwrap_err()
            .suggested_epic
            .as_deref(),
        Some("epic 3040")
    );
    // extend / bridge on the LTX IC-LoRA path + Wan TI2V-5B boundary-keyframe path are supported
    // (sc-3522 / sc-3357).
    let wan_extend = job_of(&store, JobType::VideoExtend, json!({ "model": "wan_2_2" }));
    assert!(mac_rust_supported(&wan_extend).is_ok());
    let ltx_extend = job_of(&store, JobType::VideoExtend, json!({ "model": "ltx_2_3" }));
    assert!(mac_rust_supported(&ltx_extend).is_ok());
    let ltx_bridge = job_of(
        &store,
        JobType::VideoBridge,
        json!({ "model": "ltx_2_3_eros" }),
    );
    assert!(mac_rust_supported(&ltx_bridge).is_ok());
    // SVD image→video is now MLX-supported (sc-3523: `svd`→`svd_xt`, image-conditioned only).
    let svd = job_of(
        &store,
        JobType::VideoGenerate,
        json!({ "model": "svd", "mode": "image_to_video" }),
    );
    assert!(
        mac_rust_supported(&svd).is_ok(),
        "svd image_to_video should be MLX-supported (sc-3523)"
    );
    // SVD is image-conditioned only — text→video on it is not in the Rust/MLX flow.
    let svd_text = job_of(
        &store,
        JobType::VideoGenerate,
        json!({ "model": "svd", "mode": "text_to_video" }),
    );
    assert!(
        mac_rust_supported(&svd_text).is_err(),
        "svd text_to_video is not MLX-eligible (image-conditioned only)"
    );
}

#[test]
fn mac_rust_supported_convert_flux2_ok_else_python_gap() {
    let store = store("oracle-convert");
    // The in-process Rust FLUX.2 converters are supported.
    let flux2 = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "flux2_klein_9b_true_v2", "converter": "flux2_klein_diffusers" }),
    );
    assert!(mac_rust_supported(&flux2).is_ok());
    // FLUX.2-dev pre-quantization (sc-5921) is likewise an in-process Rust/MLX convert.
    let flux2_dev = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "flux2_dev", "converter": "flux2_dev_quant" }),
    );
    assert!(mac_rust_supported(&flux2_dev).is_ok());
    // Anima (sc-10517) is an in-process Rust/MLX convert (the on-device q4/q8/bf16 matrix).
    let anima = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "anima_base", "converter": "anima_quant" }),
    );
    assert!(mac_rust_supported(&anima).is_ok());
    // LTX-2.3 (mlx-gen-ltx, sc-3240) is a live in-process convert — the sc-10573 regression: it
    // ships in the builtin manifest yet the gate's old hardcoded list omitted it, so a valid LTX
    // install-convert was mis-classified as an mlx gap.
    let ltx = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "ltx_2_3_eros", "converter": "ltx_video" }),
    );
    assert!(
        mac_rust_supported(&ltx).is_ok(),
        "ltx_video is a native in-process convert (sc-10573)"
    );
    // SD3.5 transformer pre-quantization (sc-7871) is likewise a native convert.
    let sd3 = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "sd3_5_medium", "converter": "sd3_5_medium_quant" }),
    );
    assert!(mac_rust_supported(&sd3).is_ok());
    // The default/absent converter is the Python mlx-video Wan/LTX path → gap.
    let wan = job_of(&store, JobType::ModelConvert, json!({ "model": "wan_2_2" }));
    assert_eq!(
        mac_rust_supported(&wan)
            .unwrap_err()
            .suggested_epic
            .as_deref(),
        Some("sc-3491 / sc-3224")
    );
    // A genuinely unknown converter (not in NATIVE_CONVERTERS) is still a gap.
    let bogus = job_of(
        &store,
        JobType::ModelConvert,
        json!({ "model": "whatever", "converter": "not_a_real_converter" }),
    );
    assert!(mac_rust_supported(&bogus).is_err());
}

/// sc-10573 drift guard (the core half): every converter a shipped builtin model declares in
/// `mlx.converter` MUST be covered by the convert-gap allow-list ([`NATIVE_CONVERTERS`]) — otherwise
/// its install-convert job is misclassified as an mlx gap by [`mac_rust_supported`]. That is exactly
/// the bug that shipped: `ltx_video` was a live builtin converter absent from the gate's old
/// hardcoded list. This test goes RED if a manifest converter is missing from the const (verified by
/// temporarily dropping `ltx_video` from `NATIVE_CONVERTERS`).
#[test]
fn every_builtin_manifest_converter_is_in_the_convert_gap_allowlist() {
    let manifest = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded in BUILTIN_MANIFESTS");
    let parsed: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(manifest))
            .expect("builtin.models.jsonc parses after comment stripping");
    let models = parsed
        .get("models")
        .and_then(Value::as_array)
        .expect("manifest has a models array");
    let declared: std::collections::BTreeSet<String> = models
        .iter()
        .filter_map(|model| {
            model
                .get("mlx")
                .and_then(Value::as_object)
                .and_then(|mlx| mlx.get("converter"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|converter| !converter.is_empty())
                .map(str::to_owned)
        })
        .collect();
    // Guard against the test silently rotting into a no-op if the manifest ever stops declaring any
    // `mlx.converter` (or the parse shape drifts): the guard is only meaningful with something to check.
    assert!(
        !declared.is_empty(),
        "expected builtin.models.jsonc to declare at least one mlx.converter"
    );
    for converter in &declared {
        assert!(
            sceneworks_core::jobs_store::NATIVE_CONVERTERS.contains(&converter.as_str()),
            "builtin converter '{converter}' is not covered by NATIVE_CONVERTERS (the convert-gap \
             allow-list) — its install-convert job would be misclassified as an mlx gap (sc-10573)"
        );
    }
}

/// sc-10573 arm→const drift guard for the LATENT (manifest-absent) converters. The sibling
/// `every_builtin_manifest_converter_is_in_the_convert_gap_allowlist` only covers converters a
/// shipped builtin declares in `mlx.converter`; `flux2_dev_quant` and the `sd3_5_*_quant` set ship
/// turnkey (no `mlx.converter` in the builtin manifest), so that guard can't see them. Silently
/// dropping one of those from the const would disable it — the gate and the worker would both reject
/// the convert consistently (a latent regression, not a runtime lie) — and NO other test would fail.
/// Pinning the exact set here forces every edit (add OR remove, including the latent ones) to be a
/// deliberate, reviewed change: additions MUST also gain a `resolve_convert_plan` arm (enforced by
/// the worker's `native_converters_match_resolve_convert_plan_arms` guard), and removals MUST be
/// intentional (the converter is truly retired, not merely absent from the manifest).
#[test]
fn native_converters_registry_contents_are_pinned() {
    use std::collections::BTreeSet;
    let expected: BTreeSet<&str> = [
        "flux2_klein_diffusers",
        "ltx_video",
        "flux2_dev_quant",
        "sd3_5_large_quant",
        "sd3_5_large_turbo_quant",
        "sd3_5_medium_quant",
        "anima_quant",
    ]
    .into_iter()
    .collect();
    let actual: BTreeSet<&str> = sceneworks_core::jobs_store::NATIVE_CONVERTERS
        .iter()
        .copied()
        .collect();
    assert_eq!(
        actual, expected,
        "NATIVE_CONVERTERS drifted from its pinned set — every add/remove (including the latent, \
         manifest-absent converters flux2_dev_quant + sd3_5_*_quant) must be a deliberate, reviewed \
         update: additions need a resolve_convert_plan arm, removals must be intentional (sc-10573)"
    );
}

#[test]
fn mac_rust_supported_feature_gaps_point_at_their_spikes() {
    let store = store("oracle-feature-spikes");
    // FLUX.1 reference (XLabs IP-Adapter) is now ported to MLX (spike sc-3535 → epic 3621,
    // sc-3625): the Rust engine drives the IP-Adapter natively on both schnell + dev, so it is
    // supported on the Rust/MLX worker rather than a torch-fallback gap.
    let flux_ref = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "flux_dev", "prompt": "p", "referenceAssetId": "asset_1" }),
    );
    assert!(
        mac_rust_supported(&flux_ref).is_ok(),
        "FLUX.1 reference (IP-Adapter) should be MLX-supported (epic 3621)"
    );
    // Z-Image reference without a pose set is now ported to MLX (sc-3536 spike → sc-3619):
    // the base engine's plain img2img-init path drives the reference identity natively, so
    // it is supported on the Rust/MLX worker rather than a torch-fallback gap.
    let z_ref = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p", "referenceAssetId": "asset_1" }),
    );
    assert!(
        mac_rust_supported(&z_ref).is_ok(),
        "z-image reference-without-pose should be MLX-supported (sc-3619)"
    );
    // Image understanding (VQA) + Document-Studio interleave are now ported to the Rust MLX worker
    // for the SenseNova-U1 ids (epic 3180 / sc-3905, via the concrete `T2iModel`), so they are
    // MLX-supported rather than a torch-fallback gap.
    for (job_type, payload) in [
        (
            JobType::ImageVqa,
            json!({ "model": "sensenova_u1_8b", "sourceAssetId": "asset_1", "question": "what is this?" }),
        ),
        (
            JobType::ImageVqa,
            json!({ "model": "sensenova_u1_8b_fast", "sourceAssetId": "asset_1", "question": "?" }),
        ),
        (
            JobType::ImageInterleave,
            json!({ "model": "sensenova_u1_8b", "prompt": "a tutorial" }),
        ),
    ] {
        let job = job_of(&store, job_type, payload);
        assert!(
            mac_rust_supported(&job).is_ok(),
            "SenseNova-U1 understanding/interleave should be MLX-supported (sc-3905)"
        );
    }
    // A non-SenseNova understanding job has no in-process path and stays gap-classified to the
    // SenseNova epic (the only model that serves these modes).
    let other_vqa = job_of(
        &store,
        JobType::ImageVqa,
        json!({ "model": "some_future_vlm" }),
    );
    assert_eq!(
        mac_rust_supported(&other_vqa)
            .unwrap_err()
            .suggested_epic
            .as_deref(),
        Some("epic 3180")
    );
}

#[test]
fn model_mac_support_hides_torch_only_models_keeps_mlx_models() {
    // sc-3486: the picker-gating view of the same routing predicates. An *unported* image model (no
    // Rust/MLX engine, not in `MLX_ROUTED_MODELS`) is unsupported — hidden/disabled on Mac — and,
    // lacking a dedicated port epic, reports "needs an epic" (suggested_epic None). No real image
    // model is torch-only anymore: Lens — formerly this example — is MLX-routed after epic 3164 /
    // sc-5105 (asserted below), as is pulid_flux_dev after sc-3344.
    let torch_only = model_mac_support("unported_image_model", "image");
    assert!(!torch_only.supported);
    assert!(torch_only.reason.is_some());
    assert_eq!(
        torch_only
            .reason
            .as_ref()
            .and_then(|r| r.suggested_epic.as_deref()),
        None
    );
    // Lens / Lens-Turbo (epic 3164 / sc-5105) are MLX-routed now — pure T2I, so both are supported
    // and in the picker with no gap reason. The edit-mode gating (Lens has no edit path) is asserted
    // in `model_mac_support_feature_flags_mirror_routing_without_over_gating`.
    for id in ["lens", "lens_turbo"] {
        let lens = model_mac_support(id, "image");
        assert!(lens.supported, "{id} should be MLX-supported");
        assert!(lens.reason.is_none(), "{id} should carry no gap reason");
    }
    // PuLID-FLUX (sc-3344) is MLX-routed now — it stays in the picker as a supported face-ID
    // backbone (character_image reference), no longer a torch-only port-epic gap.
    let pulid = model_mac_support("pulid_flux_dev", "image");
    assert!(pulid.supported);
    assert!(pulid.features.reference);
    // Kolors runs its full surface on MLX (epic 3090): T2I (sc-3875), img2img (sc-4765), the
    // IP-Adapter-Plus reference (sc-4767) and the strict-pose tier (sc-4766 / engine sc-5012), so all
    // three advanced features are enabled.
    let kolors = model_mac_support("kolors", "image");
    assert!(kolors.supported);
    assert!(kolors.features.edit);
    assert!(kolors.features.pose);
    assert!(kolors.features.reference);
    // An MLX-routed base family stays in the picker.
    let z_image = model_mac_support("z_image_turbo", "image");
    assert!(z_image.supported);
    assert!(z_image.reason.is_none());
    // Chroma (epic 3531 / sc-3843) is now MLX-routed — all three variants stay in the picker
    // and no longer report an `mlx_unsupported` port-epic gap.
    for id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
        let chroma = model_mac_support(id, "image");
        assert!(chroma.supported, "{id} should be MLX-supported");
        assert!(chroma.reason.is_none(), "{id} should carry no gap reason");
    }
    // SVD is now MLX-routed (sc-3523, image→video only) so it stays in the picker, like Wan/LTX;
    // a genuinely engine-less video model id is still hidden.
    assert!(model_mac_support("svd", "video").supported);
    assert!(model_mac_support("wan_2_2", "video").supported);
    assert!(!model_mac_support("some_torch_only_video", "video").supported);
    // Utility/infra models are never hidden by model-level gating (their actions are
    // gated by mac_capabilities at the job-type level instead).
    assert!(model_mac_support("real_esrgan", "utility").supported);
}

#[test]
fn model_mac_support_feature_flags_mirror_routing_without_over_gating() {
    // Base Qwen strict-pose ControlNet, Z-Image, and SDXL pose are MLX → enabled.
    assert!(model_mac_support("qwen_image", "image").features.pose);
    assert!(model_mac_support("z_image_turbo", "image").features.pose);
    assert!(model_mac_support("sdxl", "image").features.pose);
    // Z-Image reference-identity (no pose) is ported to MLX (sc-3619) → reference enabled;
    // img2img-edit is now ported too (epic 3529 / sc-3923) → edit enabled.
    let z_image = model_mac_support("z_image_turbo", "image");
    assert!(z_image.features.reference);
    assert!(z_image.features.edit);
    // The dedicated z_image_edit model is supported on Mac with edit enabled (no dead-end).
    let z_image_edit = model_mac_support("z_image_edit", "image");
    assert!(z_image_edit.supported);
    assert!(z_image_edit.features.edit);
    // FLUX.1 reference (XLabs IP-Adapter) is ported to MLX (epic 3621) → reference enabled on
    // both variants; edit_image stays off (no FLUX.1 edit on any platform — future Kontext).
    let flux = model_mac_support("flux_dev", "image");
    assert!(flux.features.reference);
    assert!(!flux.features.edit);
    let flux_schnell = model_mac_support("flux_schnell", "image");
    assert!(flux_schnell.features.reference);
    assert!(!flux_schnell.features.edit);
    // SDXL + FLUX.2 do reference/edit on MLX (epic 3041 / MLX-only family) → enabled.
    let sdxl = model_mac_support("sdxl", "image");
    assert!(sdxl.features.reference);
    assert!(sdxl.features.edit);
    let flux2 = model_mac_support("flux2_klein_9b", "image");
    assert!(flux2.features.reference);
    assert!(flux2.features.edit);
    // Qwen-Image-Edit conditions reference/edit on its modes → both enabled (no over-gate).
    let qwen_edit = model_mac_support("qwen_image_edit_2511", "image");
    assert!(qwen_edit.features.reference);
    assert!(qwen_edit.features.edit);
    // Lens / Lens-Turbo (epic 3164 / sc-5105) are pure T2I — no edit path on any platform, so
    // `edit_image` is gated off (mirrors Chroma / FLUX.1); a lens edit job never silently degrades to
    // T2I against a dropped source image.
    for id in ["lens", "lens_turbo"] {
        assert!(
            !model_mac_support(id, "image").features.edit,
            "{id} has no edit path → edit must be gated off"
        );
    }
    // Bernini still-image companion (epic 4699 / sc-5424): the image-typed `bernini_image` id
    // serves t2i + i2i (`edit_image`) on MLX → supported with edit enabled. (Like lens/chroma it
    // gates only `edit_image`, so the "available in some MLX config" `reference`/`pose` probes come
    // out permissively true — harmless: the manifest `capabilities` are the real UI gate and the
    // engine ignores a stray reference on a t2i job. The meaningful flag here is `edit`.)
    let bernini_image = model_mac_support("bernini_image", "image");
    assert!(
        bernini_image.supported,
        "bernini_image should be MLX-supported"
    );
    assert!(bernini_image.reason.is_none());
    assert!(bernini_image.features.edit, "bernini_image i2i is MLX");
    // Third-party LyCORIS now applies on every MLX provider (epic 3641) → supported.
    assert!(model_mac_support("sdxl", "image").features.lycoris);
    // Video models expose per-mode eligibility.
    let wan = model_mac_support("wan_2_2", "video").features.video_modes;
    assert_eq!(wan.get("text_to_video"), Some(&true));
    assert_eq!(wan.get("image_to_video"), Some(&true));
    assert_eq!(wan.get("first_last_frame"), Some(&true)); // Wan TI2V-5B FLF is MLX
    assert_eq!(wan.get("replace_person"), Some(&true)); // → native Wan-VACE (sc-3521)
                                                        // Wan TI2V-5B serves extend/bridge via single-frame boundary keyframe conditioning (sc-3357).
    assert_eq!(wan.get("extend_clip"), Some(&true));
    assert_eq!(wan.get("video_bridge"), Some(&true));
    // LTX serves the IC-LoRA clip-conditioning modes on MLX → extend/bridge enabled (sc-3522/3773).
    let ltx = model_mac_support("ltx_2_3", "video").features.video_modes;
    assert_eq!(ltx.get("extend_clip"), Some(&true));
    assert_eq!(ltx.get("video_bridge"), Some(&true));
    // The 14B Wan MoE engines have no FLF Keyframe path → torch.
    assert_eq!(
        model_mac_support("wan_2_2_t2v_14b", "video")
            .features
            .video_modes
            .get("first_last_frame"),
        Some(&false)
    );
    // Bernini (epic 4699) is MLX-routed text-to-video only. Its renderer is
    // Wan2.2-T2V, so still-image-to-video is off; the editing/reference video
    // modes are net-new vocabulary (sc-4703), off until then.
    let bernini = model_mac_support("bernini", "video");
    assert!(bernini.supported, "bernini should be MLX-supported");
    assert!(bernini.reason.is_none());
    assert_eq!(
        bernini.features.video_modes.get("text_to_video"),
        Some(&true)
    );
    assert_eq!(
        bernini.features.video_modes.get("image_to_video"),
        Some(&false)
    );
    assert_eq!(
        bernini.features.video_modes.get("first_last_frame"),
        Some(&false)
    );
    assert_eq!(
        bernini.features.video_modes.get("replace_person"),
        Some(&false)
    );
    // SCAIL-2 (epic 5439) is MLX-routed for the standalone character-animation mode (sc-5448) AND
    // cross-identity replace_person (sc-5452 — the same engine, replace_flag=true, as the integrated
    // backend behind the person-track pipeline); the worker paints its masks from native SAM3. No
    // text/image-to-video.
    let scail2 = model_mac_support("scail2_14b", "video");
    assert!(scail2.supported, "scail2 should be MLX-supported");
    assert!(scail2.reason.is_none());
    assert_eq!(
        scail2.features.video_modes.get("animate_character"),
        Some(&true)
    );
    assert_eq!(
        scail2.features.video_modes.get("replace_person"),
        Some(&true)
    );
    assert_eq!(
        scail2.features.video_modes.get("text_to_video"),
        Some(&false)
    );
    assert_eq!(
        scail2.features.video_modes.get("image_to_video"),
        Some(&false)
    );
}

#[test]
fn mac_capabilities_master_switch_and_infra_features() {
    // On a non-Mac host (or a Mac still in observe mode) the gating is inert.
    let inert = mac_capabilities("linux", false);
    assert!(!inert.mac_gating_active);
    assert_eq!(inert.platform, "linux");
    assert_eq!(inert.not_available_label, MAC_NOT_AVAILABLE_LABEL);
    // Unsupported infra surfaces carry their port spike so the UI affordance can name it.
    let mac = mac_capabilities("macos", true);
    assert!(mac.mac_gating_active);
    let epic = |key: &str| {
        mac.features
            .get(key)
            .and_then(|f| f.reason.as_ref())
            .and_then(|r| r.suggested_epic.as_deref())
            .map(str::to_owned)
    };
    // Real-ESRGAN upscaling is ported (sc-3489) → the tool is supported, no reason/epic.
    assert_eq!(epic("imageUpscale"), None);
    assert!(mac.features["imageUpscale"].supported);
    // The AuraSR engine is dropped on Mac (sc-3668) AND off-Mac as an offered engine (sc-5499) → its
    // per-engine feature is unsupported on every platform and names the drop; this must agree with the
    // AuraSR arm of `mac_rust_supported`.
    assert!(!mac.features["imageUpscaleAuraSr"].supported);
    assert_eq!(epic("imageUpscaleAuraSr"), Some("sc-5499".to_owned()));
    // SeedVR2 is the native-MLX upscaler (epic 4811 / sc-4815) → supported on Mac, no reason/epic.
    assert!(mac.features["imageUpscaleSeedvr2"].supported);
    assert_eq!(epic("imageUpscaleSeedvr2"), None);
    // DWPose pose detection is ported (sc-3487) → supported, no reason/epic (sc-4206).
    assert_eq!(epic("poseFromPhoto"), None);
    assert!(mac.features["poseFromPhoto"].supported);
    // Person detect/track is ported (sc-3488 / sc-3633/3634/3709) → supported, no epic.
    assert_eq!(epic("personDetect"), None);
    assert!(mac.features["personDetect"].supported);
    // Smart-select segmentation is native-MLX SAM3 on Mac (sc-6105) → supported, no epic.
    assert_eq!(epic("imageSegment"), None);
    assert!(mac.features["imageSegment"].supported);
    assert_eq!(epic("datasetCaptioning"), None);
    // LyCORIS is ported to MLX (epic 3641) → no longer a capability gap entry at all.
    assert!(!mac.features.contains_key("lycoris"));
    // The global advancedVideoModes flag is gone (sc-3773) — extend/bridge are gated per-model
    // via each video model's macSupport.features.videoModes instead.
    assert!(!mac.features.contains_key("advancedVideoModes"));
    assert!(mac.features["datasetCaptioning"].supported);
    // Video upscale is net-new on Mac (epic 4811 / sc-4816, native-MLX SeedVR2) → supported, no epic.
    assert_eq!(epic("videoUpscale"), None);
    assert!(mac.features["videoUpscale"].supported);
    // datasetCaptioning + imageSegment + imageUpscale + imageUpscaleSeedvr2 + personDetect +
    // poseFromPhoto + videoUpscale are the ported (supported) infra features; the rest stay gated
    // until their port lands. poseFromPhoto joined the supported set in sc-4206 (DWPose ported,
    // sc-3487); imageUpscaleSeedvr2 in sc-4815, videoUpscale in sc-4816 (both native-MLX SeedVR2,
    // epic 4811); imageSegment in sc-6105 (native-MLX SAM3 smart-select).
    assert!(mac
        .features
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "datasetCaptioning"
                    | "imageSegment"
                    | "imageUpscale"
                    | "imageUpscaleSeedvr2"
                    | "personDetect"
                    | "poseFromPhoto"
                    | "videoUpscale"
            )
        })
        .all(|(_, f)| !f.supported));
    // SeedVR2 now has a backend on every GPU platform: native MLX on Mac, and the candle CUDA/NVIDIA
    // port on Windows (sc-5928) + Linux (sc-5160 — candle is CPU+CUDA cross-platform, so Linux rides
    // the Windows port). On the (inert-gated) Linux host the capability is platform-true with no
    // reason, so the web picker offers `engine=seedvr2` there too.
    assert!(inert.features["imageUpscaleSeedvr2"].supported);
    assert!(inert.features["imageUpscaleSeedvr2"].reason.is_none());
    // Windows carries the same candle SeedVR2 backend (sc-5928): platform-true there too (not just Mac).
    let windows = mac_capabilities("windows", false);
    assert!(windows.features["imageUpscaleSeedvr2"].supported);
    assert!(windows.features["imageUpscaleSeedvr2"].reason.is_none());
    // AuraSR is dropped as an offered engine off-Mac too (sc-5499): unsupported on Windows (and Linux),
    // not just under active Mac gating — so the web picker hides it on every platform.
    assert!(!windows.features["imageUpscaleAuraSr"].supported);
    assert!(!inert.features["imageUpscaleAuraSr"].supported);
    // Training kernels with a native Rust trainer stay enabled; LoKr-on-Wan does not.
    assert!(mac
        .training
        .supported_kernels
        .iter()
        .any(|k| k == "z_image_lora"));
    assert!(mac
        .training
        .supported_kernels
        .iter()
        .any(|k| k == "sdxl_lora"));
    // Kolors gained a native Rust trainer (sc-4568) and is now advertised (sc-4732).
    assert!(mac
        .training
        .supported_kernels
        .iter()
        .any(|k| k == "kolors_lora"));
    assert!(!mac.training.lokr_on_wan_supported);
}

#[test]
fn fail_unsupported_mlx_jobs_enforce_fails_only_unsupported() {
    let store = store("oracle-enforce");
    let unsupported = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "pulid_flux_dev", "prompt": "p" }),
    );
    let eligible = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "z_image_turbo", "prompt": "p" }),
    );

    let failed = store
        .fail_unsupported_mlx_jobs(true, true)
        .expect("sweep ok");
    assert_eq!(failed.len(), 1, "only the unsupported job is failed");
    assert_eq!(failed[0].0.id, unsupported.id);
    assert_eq!(failed[0].0.status, JobStatus::Failed);
    assert!(failed[0]
        .0
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("mlx_unsupported"));
    // The eligible job is untouched — it's routing/`fail_stranded`'s concern, not this sweep's.
    assert_eq!(
        store.get_job(&eligible.id).expect("loads").status,
        JobStatus::Queued
    );
}

#[test]
fn fail_unsupported_mlx_jobs_noop_when_warn_or_off() {
    let store = store("oracle-warn-off");
    let job = job_of(
        &store,
        JobType::ImageGenerate,
        json!({ "model": "pulid_flux_dev", "prompt": "p" }),
    );
    // Warn-only (enforce=false): logged at claim time, never failed by the sweep.
    assert!(store
        .fail_unsupported_mlx_jobs(true, false)
        .expect("ok")
        .is_empty());
    // Flag off (not mlx-required): never touches anything (Windows/Linux/Docker).
    assert!(store
        .fail_unsupported_mlx_jobs(false, true)
        .expect("ok")
        .is_empty());
    assert_eq!(
        store.get_job(&job.id).expect("loads").status,
        JobStatus::Queued
    );
}

// sc-3449 — claim_next_job_routed reports *why* an MLX-eligible job landed where it did.

#[test]
fn routing_decision_reports_claimed_by_gpu_for_uncategorized_worker() {
    let store = store("route-decision-gpu");
    // Defensive catch-all: a GPU worker that is neither `mlx` nor candle (no `candle` marker)
    // claims an MLX-eligible auto job. With the Python torch worker retired from every surface
    // this should not happen in practice, so the decision is named generically
    // (`claimed_by_gpu`) rather than after a backend that no longer exists.
    register_gpu_worker(&store, "worker-gpu", "cuda:0", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "p"
            }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-gpu", false)
        .expect("gpu claim ok");
    assert_eq!(claimed.expect("gpu worker claims it").id, job.id);
    let decision = decision.expect("routing decision present");
    assert_eq!(decision.decision, "claimed_by_gpu");
    assert_eq!(decision.reason, "gpu_worker");
    assert_eq!(decision.gpu_id, "cuda:0");
    assert_eq!(
        decision.model.as_deref(),
        Some("qwen_image_edit_2511_lightning")
    );
}

#[test]
fn routing_decision_reports_claimed_by_candle() {
    let store = store("route-decision-candle");
    // The Windows/Linux happy path: the candle (CUDA) worker — identified by the `candle`
    // capability marker, on a real gpu index ("0") — claims an MLX-eligible auto job. The
    // decision names candle, never "fell back to torch": nothing is missing here.
    register_candle_worker(&store, "worker-candle");
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-candle", false)
        .expect("candle claim ok");
    assert_eq!(claimed.expect("candle claims it").id, job.id);
    let decision = decision.expect("routing decision present");
    assert_eq!(decision.decision, "claimed_by_candle");
    assert_eq!(decision.reason, "candle_worker");
    assert_eq!(decision.gpu_id, "0");
}

#[test]
fn routing_decision_reports_deferred_to_mlx_for_torch_worker() {
    let store = store("route-decision-defer");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());
    store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-torch", false)
        .expect("torch claim ok");
    assert!(claimed.is_none(), "torch defers to the idle mlx worker");
    let decision = decision.expect("routing decision present");
    assert_eq!(decision.decision, "deferred_to_mlx");
    assert_eq!(decision.reason, "idle_mlx_available");
}

#[test]
fn routing_decision_reports_claimed_by_mlx() {
    let store = store("route-decision-mlx");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-mlx", false)
        .expect("mlx claim ok");
    assert_eq!(claimed.expect("mlx claims it").id, job.id);
    let decision = decision.expect("routing decision present");
    assert_eq!(decision.decision, "claimed_by_mlx");
    assert_eq!(decision.reason, "mlx_worker");
    assert_eq!(decision.gpu_id, "mlx");
}

#[test]
fn routing_decision_is_none_for_non_mlx_model() {
    let store = store("route-decision-none");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    store
        .create_job(image_job_with(
            json!({ "model": "some_torch_only_model", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let (claimed, decision) = store
        .claim_next_job_routed("worker-torch", false)
        .expect("torch claim ok");
    assert!(claimed.is_some(), "torch claims the torch-only job");
    assert!(
        decision.is_none(),
        "a non-MLX-eligible claim is routing-neutral (no event)"
    );
}

// --- sc-3513: the `image_edit` job type (plain Image Edit) routes to MLX too ---
//
// "Plain Image Edit" is submitted as JobType::ImageEdit (mode=edit_image + sourceAssetId),
// a *distinct* job type from the character/reference flow (JobType::ImageGenerate). The
// engine dispatches edits on payload model+mode, not job type, so the edit-capable families
// (qwen/flux2/sdxl) must route to the in-process mlx worker exactly like the generate flow;
// torch-only edit models stay on torch. Before sc-3513 the routing gate excluded every
// non-ImageGenerate type, so these jobs ran on torch silently with no `gpu_route_decision`.

#[test]
fn qwen_image_edit_job_type_defers_to_mlx_worker() {
    let store = store("mlx-routing-image-edit-qwen");
    register_gpu_worker(&store, "worker-torch", "mps", image_edit_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());

    let job = store
        .create_job(image_edit_job_with(
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "make it a watercolor painting"
            }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-eligible edit to the idle mlx worker, which claims it.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn sdxl_masked_image_edit_job_type_defers_to_mlx_worker() {
    let store = store("mlx-routing-image-edit-sdxl-mask");
    register_gpu_worker(&store, "worker-torch", "mps", image_edit_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());

    // A masked inpaint is an image_edit job carrying a maskAssetId. Only SDXL/RealVisXL
    // are image_inpaint-capable, and sc-3060's advanced SDXL stream handles masked
    // inpaint/outpaint on the engine — so the mask is honoured on MLX, not dropped.
    let job = store
        .create_job(image_edit_job_with(
            json!({
                "model": "sdxl",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "maskAssetId": "asset_mask",
                "prompt": "replace the sky"
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn active_mlx_image_edit_blocks_mps_image_edit_claim() {
    let store = store("mlx-mps-shared-gpu-active-mlx");
    register_gpu_worker(&store, "worker-mps", "mps", image_edit_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());

    let mlx_job = store
        .create_job(image_edit_job_with(
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "make it a watercolor painting"
            }),
            "auto",
        ))
        .expect("mlx job creates");
    let claimed_mlx = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the first job");
    assert_eq!(claimed_mlx.id, mlx_job.id);
    assert_eq!(claimed_mlx.assigned_gpu.as_deref(), Some("mlx"));

    // An unported image model is the MPS job: not in MLX_ROUTED_MODELS, so it isn't MLX-eligible and
    // isn't soft-deferred to the mlx worker — it stays on the torch worker, exercising the shared-GPU
    // exclusion cleanly. (No real image model is torch-only anymore: Lens — formerly this example —
    // is MLX after epic 3164 / sc-5105, as Kolors/PuLID-FLUX were before it.)
    let mps_job = store
        .create_job(image_job_with(
            json!({
                "model": "unported_image_model",
                "prompt": "p"
            }),
            "auto",
        ))
        .expect("mps job creates");

    assert!(
        store
            .claim_next_job("worker-mps")
            .expect("mps claim ok")
            .is_none(),
        "MPS must wait while the MLX worker is using the shared Apple GPU"
    );

    complete_job(&store, &claimed_mlx.id);
    let claimed_mps = store
        .claim_next_job("worker-mps")
        .expect("mps claim ok")
        .expect("mps claims once mlx completes");
    assert_eq!(claimed_mps.id, mps_job.id);
    assert_eq!(claimed_mps.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn active_mps_image_edit_blocks_mlx_image_edit_claim() {
    let store = store("mlx-mps-shared-gpu-active-mps");
    register_gpu_worker(&store, "worker-mps", "mps", image_edit_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());

    // An unported image model is the MPS job (not in MLX_ROUTED_MODELS, so it isn't soft-deferred to
    // the mlx worker). No real image model is torch-only anymore — Lens, the last one, is MLX after
    // epic 3164 / sc-5105 (PuLID-FLUX was MLX after sc-3344).
    let mps_job = store
        .create_job(image_job_with(
            json!({
                "model": "unported_image_model",
                "prompt": "p"
            }),
            "auto",
        ))
        .expect("mps job creates");
    let claimed_mps = store
        .claim_next_job("worker-mps")
        .expect("mps claim ok")
        .expect("mps claims the first job");
    assert_eq!(claimed_mps.id, mps_job.id);
    assert_eq!(claimed_mps.assigned_gpu.as_deref(), Some("mps"));

    let mlx_job = store
        .create_job(image_edit_job_with(
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "edit_image",
                "sourceAssetId": "asset_2",
                "prompt": "make it a watercolor painting"
            }),
            "auto",
        ))
        .expect("mlx job creates");

    assert!(
        store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .is_none(),
        "MLX must wait while the MPS worker is using the shared Apple GPU"
    );

    complete_job(&store, &claimed_mps.id);
    let claimed_mlx = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims once mps completes");
    assert_eq!(claimed_mlx.id, mlx_job.id);
    assert_eq!(claimed_mlx.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn routing_decision_reports_claimed_by_mlx_for_image_edit() {
    let store = store("route-decision-image-edit-mlx");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());
    let job = store
        .create_job(image_edit_job_with(
            json!({
                "model": "flux2_klein_9b_true_v2",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "prompt": "p"
            }),
            "auto",
        ))
        .expect("job creates");

    // The fix makes the claim non-routing-neutral: a gpu_route_decision is now emitted.
    let (claimed, decision) = store
        .claim_next_job_routed("worker-mlx", false)
        .expect("mlx claim ok");
    assert_eq!(claimed.expect("mlx claims it").id, job.id);
    let decision = decision.expect("routing decision present");
    assert_eq!(decision.decision, "claimed_by_mlx");
    assert_eq!(decision.reason, "mlx_worker");
    assert_eq!(decision.gpu_id, "mlx");
}

#[test]
fn torch_only_image_model_stays_on_torch() {
    let store = store("mlx-routing-image-torch-only");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());

    // An unported image model (not in MLX_ROUTED_MODELS) stays on the Python torch path and the mlx
    // worker must refuse it. Every ported image family — incl. Kolors' full surface (epic 3090),
    // PuLID-FLUX (sc-3344), and Lens (epic 3164 / sc-5105, the last one) — is MLX now, so a
    // torch-only example must come from an unported model id.
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "unported_image_model",
                "prompt": "p"
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    // A torch worker is the home for it, and the claim is routing-neutral (no event).
    register_gpu_worker(&store, "worker-torch", "mps", image_edit_caps());
    let (claimed, decision) = store
        .claim_next_job_routed("worker-torch", false)
        .expect("torch claim ok");
    let claimed = claimed.expect("torch claims it");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
    assert!(
        decision.is_none(),
        "a torch-only image model is routing-neutral (no gpu_route_decision)"
    );
}

#[test]
fn z_image_edit_routes_to_mlx() {
    // epic 3529 / sc-3923: z_image_edit (and z_image_turbo edit_image) img2img-edit now runs
    // on the in-process Rust MLX worker — the engine's `Conditioning::Reference` img2img path.
    for model in ["z_image_edit", "z_image_turbo"] {
        let store = store(&format!("mlx-routing-z-image-edit-{model}"));
        register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());
        let job = store
            .create_job(image_edit_job_with(
                json!({
                    "model": model,
                    "mode": "edit_image",
                    "sourceAssetId": "asset_1",
                    "prompt": "make it a watercolor painting"
                }),
                "auto",
            ))
            .expect("job creates");

        let (claimed, decision) = store
            .claim_next_job_routed("worker-mlx", false)
            .expect("mlx claim ok");
        assert_eq!(claimed.expect("mlx claims it").id, job.id, "{model}");
        let decision = decision.expect("routing decision present");
        assert_eq!(decision.decision, "claimed_by_mlx", "{model}");
        assert_eq!(decision.gpu_id, "mlx", "{model}");
    }
}

#[test]
fn explicit_gpu_image_job_is_not_deferred_to_mlx_worker() {
    let store = store("mlx-routing-explicit-gpu");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // The user explicitly pinned this MLX-eligible job to the torch GPU; honour it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "mps",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the explicit-gpu job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

// --- Training routing (epic 3039, sc-3043/3049) ---

fn training_caps() -> Vec<WorkerCapability> {
    vec![
        WorkerCapability::Gpu,
        WorkerCapability::LoraTrain,
        WorkerCapability::LoraTrainExecute,
    ]
}

fn mlx_training_job(
    kernel: &str,
    base_model: &str,
    network_type: &str,
    dry_run: bool,
    requested_gpu: &str,
) -> CreateJob {
    CreateJob {
        job_type: JobType::LoraTrain,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(json!({
            "dryRun": dry_run,
            "plan": {
                "planVersion": 1,
                "target": { "kernel": kernel, "baseModel": base_model },
                "config": { "advanced": { "networkType": network_type } }
            }
        })),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

fn caption_caps() -> Vec<WorkerCapability> {
    vec![WorkerCapability::Gpu, WorkerCapability::TrainingCaption]
}

fn joy_caption_job(requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::TrainingCaption,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(json!({
            "provider": "training",
            "kind": "training_caption",
            "captioner": "joy_caption",
            "modelNameOrPath": "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "projectId": "project-1",
            "datasetId": "dataset-1",
            "items": [{
                "itemId": "item_0001",
                "imagePath": "/tmp/item_0001.png",
                "triggerWords": ["miraStyle"]
            }]
        })),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

#[test]
fn joy_caption_routes_to_idle_mlx_worker() {
    let store = store("mlx-caption-routing");
    register_gpu_worker(&store, "worker-torch", "mps", caption_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", caption_caps());

    let job = store
        .create_job(joy_caption_job("auto"))
        .expect("caption job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims joy caption");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn joy_caption_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-caption-fallback");
    register_gpu_worker(&store, "worker-torch", "cuda:0", caption_caps());

    let job = store
        .create_job(joy_caption_job("auto"))
        .expect("caption job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims caption job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

#[test]
fn explicit_gpu_joy_caption_is_not_deferred_to_mlx_worker() {
    let store = store("mlx-caption-explicit-gpu");
    register_gpu_worker(&store, "worker-torch", "mps", caption_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", caption_caps());

    let job = store
        .create_job(joy_caption_job("mps"))
        .expect("caption job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims explicit caption job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_eligible_training_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-training-defer");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    let job = store
        .create_job(mlx_training_job(
            "z_image_lora",
            "z_image_turbo",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-native training job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    // The mlx worker claims it and trains in-process via mlx-gen.
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_eligible_kolors_training_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-training-kolors");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // Kolors gained a native mlx-gen Rust trainer (sc-4568) and routes to the mlx
    // worker (sc-4732), exactly like the other MLX-native training families.
    let job = store
        .create_job(mlx_training_job(
            "kolors_lora",
            "kolors",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the now-MLX-native kolors training job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    // The mlx worker claims it and trains in-process via mlx-gen's KolorsTrainer.
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the kolors training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn kolors_lokr_training_job_also_routes_to_the_mlx_worker() {
    // Unlike LoKr-on-Wan (no Kronecker merge in the mlx Wan path), the Kolors trainer
    // supports LoKr (descriptor.supports_lokr = true, sc-4568), so a LoKr kolors job is
    // MLX-eligible too.
    let store = store("mlx-training-kolors-lokr");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    let job = store
        .create_job(mlx_training_job(
            "kolors_lora",
            "kolors",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the kolors lokr training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_lokr_wan_training_job() {
    let store = store("mlx-training-lokr-wan");
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // LoKr-on-Wan has no Kronecker merge in the mlx Wan path → torch only.
    let job = store
        .create_job(mlx_training_job(
            "wan_moe_lora",
            "wan_2_2_t2v_14b",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the LoKr-on-Wan job");
    assert_eq!(claimed.id, job.id);
}

#[test]
fn lokr_z_image_training_stays_mlx_eligible() {
    let store = store("mlx-training-lokr-zimage");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // LoKr on Z-Image/SDXL/LTX is fine — the Rust engine applies it natively.
    let job = store
        .create_job(mlx_training_job(
            "z_image_lora",
            "z_image_turbo",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LoKr Z-Image job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_eligible_training_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-training-fallback");
    // No mlx worker (Windows/Linux, or it's down) — torch is the only path.
    register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());

    let job = store
        .create_job(mlx_training_job("sdxl_lora", "sdxl", "lora", false, "auto"))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

// ── sc-7817: off-Mac candle training routing (epic 5164) ───────────────────────────────────────
// A candle (Windows/CUDA) worker — a real CUDA gpu_id + the `candle` marker + training caps.
fn candle_training_caps() -> Vec<WorkerCapability> {
    vec![
        WorkerCapability::Gpu,
        WorkerCapability::LoraTrain,
        WorkerCapability::LoraTrainExecute,
        WorkerCapability::Unknown("candle".to_owned()),
    ]
}

#[test]
fn candle_worker_claims_candle_native_training_kernels() {
    // The five families with a candle trainer (sdxl / z_image / lens / Krea 2 Raw / Wan A14B T2V)
    // route to the candle worker off-Mac. `krea_lora` is no-torch-fallback (MLX_ONLY) yet candle
    // claims it (sc-8614) — the gate exempts a candle-eligible job from the mlx-only refusal. Each in
    // its own store so the claim is unambiguous.
    let cases: &[(&str, &str, &str)] = &[
        ("sdxl_lora", "sdxl", "lora"),
        ("z_image_lora", "z_image_turbo", "lokr"),
        ("lens_lora", "lens", "lora"),
        ("krea_lora", "krea_2_raw", "lokr"),
        ("wan_moe_lora", "wan_2_2_t2v_14b", "lora"),
    ];
    for (kernel, base_model, network_type) in cases {
        let store = store(&format!("candle-training-{kernel}-{base_model}"));
        register_gpu_worker(&store, "worker-candle", "0", candle_training_caps());
        let job = store
            .create_job(mlx_training_job(
                kernel,
                base_model,
                network_type,
                false,
                "auto",
            ))
            .expect("job creates");
        let claimed = store
            .claim_next_job("worker-candle")
            .unwrap_or_else(|error| panic!("candle claim ok ({kernel}): {error:?}"))
            .unwrap_or_else(|| panic!("candle should claim {kernel}/{base_model}"));
        assert_eq!(claimed.id, job.id, "kernel={kernel} base={base_model}");
    }
}

#[test]
fn candle_worker_refuses_torch_served_training_kernels() {
    // Kernels with no candle trainer but a torch fallback: the candle worker must REFUSE them (the
    // `lora_train_execute` advertisement is coarse), leaving them for the co-resident torch worker —
    // otherwise the candle worker would claim and fail terminally. Covers Kolors, the dense Wan 5B,
    // and the I2V A14B (candle has only the T2V A14B trainer).
    let cases: &[(&str, &str)] = &[
        ("kolors_lora", "kolors"),
        ("wan_lora", "wan_2_2"),
        ("wan_moe_lora", "wan_2_2_i2v_14b"),
    ];
    for (kernel, base_model) in cases {
        let store = store(&format!("candle-refuse-{kernel}-{base_model}"));
        register_gpu_worker(&store, "worker-candle", "0", candle_training_caps());
        let job = store
            .create_job(mlx_training_job(kernel, base_model, "lora", false, "auto"))
            .expect("job creates");
        assert!(
            store
                .claim_next_job("worker-candle")
                .unwrap_or_else(|error| panic!("candle claim ok ({kernel}): {error:?}"))
                .is_none(),
            "candle must refuse {kernel}/{base_model} (no candle trainer)"
        );
        // The co-resident torch worker serves it.
        register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());
        let claimed = store
            .claim_next_job("worker-torch")
            .expect("torch claim ok")
            .unwrap_or_else(|| panic!("torch should claim {kernel}/{base_model}"));
        assert_eq!(claimed.id, job.id, "kernel={kernel} base={base_model}");
    }
}

#[test]
fn candle_worker_refuses_ltx_mlx_only_training() {
    // ltx_mlx_lora has no torch fallback (MLX_ONLY_TRAINING_KERNELS) AND no candle trainer, so off-Mac
    // neither the candle worker nor a torch worker may claim it — it stays queued for an mlx worker.
    let store = store("candle-refuse-ltx");
    register_gpu_worker(&store, "worker-candle", "0", candle_training_caps());
    register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());
    store
        .create_job(mlx_training_job(
            "ltx_mlx_lora",
            "ltx_2_3",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");
    assert!(
        store
            .claim_next_job("worker-candle")
            .expect("candle claim ok")
            .is_none(),
        "candle must refuse ltx_mlx_lora (no candle trainer)"
    );
    assert!(
        store
            .claim_next_job("worker-torch")
            .expect("torch claim ok")
            .is_none(),
        "torch must refuse ltx_mlx_lora (mlx-only, no torch trainer)"
    );
}

#[test]
fn ltx_training_is_mlx_worker_only_with_no_torch_fallback() {
    let store = store("mlx-training-ltx-only");
    // sc-3049 retired the Python MLX LTX trainer, so `ltx_mlx_lora` has no torch
    // fallback: a torch worker must NOT claim it — it stays queued for the mlx worker.
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    let job = store
        .create_job(mlx_training_job(
            "ltx_mlx_lora",
            "ltx_2_3",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());

    // The mlx worker is the only home for it.
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LTX training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn krea_training_has_no_torch_fallback_but_runs_on_either_rust_backend() {
    let store = store("training-krea-no-torch");
    // Krea 2 (epic 7565) trains on a native Rust trainer and has NO torch path, so — like LTX — a
    // torch worker must NOT claim a `krea_lora` job. UNLIKE LTX it is Rust-only on BOTH backends:
    // the mlx worker (Mac) and, since sc-8614, the candle worker (off-Mac) each run it.
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    let job = store
        .create_job(mlx_training_job(
            "krea_lora",
            "krea_2_raw",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    // Torch defers (no Krea torch trainer).
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());

    // The candle worker claims it off-Mac (sc-8614): no-torch-fallback no longer means mlx-only.
    register_gpu_worker(&store, "worker-candle", "0", candle_training_caps());
    let claimed = store
        .claim_next_job("worker-candle")
        .expect("candle claim ok")
        .expect("candle claims the Krea training job");
    assert_eq!(claimed.id, job.id);
}

#[test]
fn krea_training_runs_on_the_mlx_worker() {
    let store = store("mlx-training-krea");
    // On Mac the mlx worker is Krea's home (the off-Mac candle path is covered separately).
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    let job = store
        .create_job(mlx_training_job(
            "krea_lora",
            "krea_2_raw",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the Krea training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn lokr_krea_training_stays_mlx_eligible() {
    let store = store("mlx-training-krea-lokr");
    // Unlike Wan (no MLX Kronecker merge), Krea's native trainer + Turbo inference seam both
    // handle LoKr, so a LoKr `krea_lora` job stays MLX-eligible and is NOT shed to torch.
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());
    let job = store
        .create_job(mlx_training_job(
            "krea_lora",
            "krea_2_raw",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    // Torch defers (Krea is MLX-only); the mlx worker claims the LoKr job.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the Krea LoKr training job");
    assert_eq!(claimed.id, job.id);
}

// --- Video routing (epic 3018, sc-3036) ---

fn video_caps() -> Vec<WorkerCapability> {
    vec![
        WorkerCapability::Gpu,
        WorkerCapability::VideoGenerate,
        // The macOS MLX worker also advertises the clip-conditioning job types (sc-3522)
        // and replace_person → Wan-VACE (sc-3521).
        WorkerCapability::VideoExtend,
        WorkerCapability::VideoBridge,
        WorkerCapability::PersonReplace,
    ]
}

fn video_job_with(payload: Value, requested_gpu: &str) -> CreateJob {
    video_job_typed(JobType::VideoGenerate, payload, requested_gpu)
}

fn video_job_typed(job_type: JobType, payload: Value, requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    }
}

#[test]
fn mlx_eligible_video_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-video-routing-defer");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2", "mode": "text_to_video", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-eligible video job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_advanced_mode_video_job() {
    let store = store("mlx-video-routing-exclude");
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // extend_clip / video_bridge stay torch on engines with no keyframe path — the 14B Wan MoE
    // here (sc-3522 / sc-3357: LTX + Wan TI2V-5B serve them on MLX, the MoE engines do not); the
    // mlx worker must not claim this one.
    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2_t2v_14b", "mode": "extend_clip" }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn replace_person_job_defers_from_torch_worker_to_idle_mlx_worker() {
    // sc-3521 cutover: replace_person → native Wan-VACE is MLX-eligible on the replace-capable
    // models, so a torch worker defers the `PersonReplace` job to an idle mlx worker.
    let store = store("mlx-video-routing-replace");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_typed(
            JobType::PersonReplace,
            json!({
                "model": "wan_2_2", "mode": "replace_person",
                "sourceClipAssetId": "clip", "personTrackId": "track_1", "characterId": "char_1"
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the replace_person job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn flf_video_job_defers_from_torch_worker_to_idle_mlx_worker() {
    // sc-3055 cutover: first_last_frame is MLX-eligible on the FLF-capable engines (LTX +
    // Wan TI2V-5B `wan_2_2`), so a torch worker defers it to an idle mlx worker.
    for model in ["ltx_2_3", "wan_2_2"] {
        let store = store(&format!("mlx-video-routing-flf-{model}"));
        register_gpu_worker(&store, "worker-torch", "mps", video_caps());
        register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

        let job = store
            .create_job(video_job_with(
                json!({
                    "model": model, "mode": "first_last_frame",
                    "sourceAssetId": "a", "lastFrameAssetId": "b"
                }),
                "auto",
            ))
            .expect("job creates");

        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model}: torch defers FLF to the idle mlx worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .expect("mlx claims the FLF job");
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
    }
}

#[test]
fn flf_video_job_stays_on_torch_for_non_flf_capable_wan_moe() {
    // FLF on the 14B Wan MoE engines has no engine Keyframe path → stays torch.
    let store = store("mlx-video-routing-flf-moe");
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2_t2v_14b", "mode": "first_last_frame" }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn clip_conditioning_video_job_defers_from_torch_worker_to_idle_mlx_worker() {
    // sc-3522 / sc-3357 cutover: extend_clip / video_bridge are MLX-eligible on the LTX IC-LoRA
    // path and Wan TI2V-5B (`wan_2_2`, boundary-keyframe conditioning), so a torch worker defers
    // the dedicated job types to an idle mlx worker.
    for (job_type, mode) in [
        (JobType::VideoExtend, "extend_clip"),
        (JobType::VideoBridge, "video_bridge"),
    ] {
        for model in ["ltx_2_3", "ltx_2_3_eros", "wan_2_2"] {
            let store = store(&format!("mlx-video-routing-clip-{mode}-{model}"));
            register_gpu_worker(&store, "worker-torch", "mps", video_caps());
            register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

            let job = store
                .create_job(video_job_typed(
                    job_type.clone(),
                    json!({
                        "model": model, "mode": mode,
                        "sourceClipAssetId": "left", "bridgeRightClipAssetId": "right"
                    }),
                    "auto",
                ))
                .expect("job creates");

            assert!(
                store
                    .claim_next_job("worker-torch")
                    .expect("torch claim ok")
                    .is_none(),
                "{model}/{mode}: torch defers the clip job to the idle mlx worker"
            );
            let claimed = store
                .claim_next_job("worker-mlx")
                .expect("mlx claim ok")
                .expect("mlx claims the clip job");
            assert_eq!(claimed.id, job.id);
            assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        }
    }
}

#[test]
fn clip_conditioning_video_job_stays_on_torch_for_wan_moe_engines() {
    // extend_clip / video_bridge have no `Keyframe` path on the 14B Wan MoE engines → stays torch,
    // even though the mlx worker advertises the VideoExtend/VideoBridge capabilities. (Wan TI2V-5B
    // `wan_2_2` IS MLX-eligible — sc-3357 — and is covered by the defer test above.)
    for (job_type, mode) in [
        (JobType::VideoExtend, "extend_clip"),
        (JobType::VideoBridge, "video_bridge"),
    ] {
        let store = store(&format!("mlx-video-routing-clip-wan-moe-{mode}"));
        register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

        let job = store
            .create_job(video_job_typed(
                job_type.clone(),
                json!({ "model": "wan_2_2_t2v_14b", "mode": mode, "sourceClipAssetId": "left" }),
                "auto",
            ))
            .expect("job creates");

        assert!(
            store
                .claim_next_job("worker-mlx")
                .expect("mlx claim ok")
                .is_none(),
            "{mode}: mlx worker must not claim a 14B Wan MoE clip job"
        );

        register_gpu_worker(&store, "worker-torch", "mps", video_caps());
        let claimed = store
            .claim_next_job("worker-torch")
            .expect("torch claim ok")
            .expect("torch claims the Wan clip job");
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
    }
}

#[test]
fn mlx_eligible_video_job_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-video-routing-fallback");
    register_gpu_worker(&store, "worker-torch", "cuda:0", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "ltx_2_3", "mode": "text_to_video", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

#[test]
fn explicit_gpu_video_job_is_not_deferred_to_mlx_worker() {
    let store = store("mlx-video-routing-explicit-gpu");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2", "mode": "text_to_video", "prompt": "p" }),
            "mps",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the explicit-gpu job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn lokr_on_wan_video_routes_to_mlx() {
    let store = store("mlx-video-lokr-wan");
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // LoKr-on-Wan now routes to MLX (epic 3641 / sc-3644): the Wan engine merges the Kronecker
    // delta in-place (merge_one_lokr, sc-2393) — the old torch gate was a routing caution, not an
    // engine limit. (Wan LoKr *training* still stays torch, epic 3039 — a separate path.)
    let job = store
        .create_job(video_job_with(
            json!({
                "model": "wan_2_2_t2v_14b",
                "mode": "text_to_video",
                "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
            }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LoKr-on-Wan video job");
    assert_eq!(claimed.id, job.id);
}

#[test]
fn lokr_on_ltx_video_routes_to_mlx_worker() {
    let store = store("mlx-video-lokr-ltx");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // LoKr-on-LTX stays MLX: the torch LTX path has no LoKr loader; the Rust engine
    // applies it natively.
    let job = store
        .create_job(video_job_with(
            json!({
                "model": "ltx_2_3",
                "mode": "text_to_video",
                "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LoKr-on-LTX job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn flux_schnell_txt2img_routes_to_mlx_worker() {
    let store = store("mlx-routing-flux");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.1 txt2img (sc-3023) is MLX-eligible → defers to the idle mlx worker.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "flux_schnell", "prompt": "a red fox" }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims flux txt2img");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn kolors_txt2img_routes_to_mlx_worker() {
    let store = store("mlx-routing-kolors");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // Kolors plain T2I (sc-3875) is MLX-eligible → defers to the idle mlx worker.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "kolors", "prompt": "a red fox" }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims kolors txt2img");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn kolors_advanced_modes_route_to_mlx() {
    // epic 3090: kolors runs its full surface on MLX — img2img (sc-4765, `edit_image` +
    // `sourceAssetId`), the IP-Adapter-Plus reference (sc-4767, `referenceAssetId`) and the
    // strict-pose tier (sc-4766 / engine sc-5012, `advanced.poses` + a reference) — all defer to the
    // idle mlx worker.
    for (index, payload) in [
        json!({
            "model": "kolors",
            "mode": "edit_image",
            "sourceAssetId": "asset_1",
            "prompt": "p"
        }),
        json!({
            "model": "kolors",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
            "prompt": "p"
        }),
        json!({
            "model": "kolors",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
            "advanced": { "poses": [{ "keypoints": [] }] },
            "prompt": "p"
        }),
    ]
    .into_iter()
    .enumerate()
    {
        let store = store(&format!("mlx-routing-kolors-advanced-{index}"));
        register_gpu_worker(&store, "worker-torch", "mps", image_edit_caps());
        register_gpu_worker(&store, "worker-mlx", "mlx", image_edit_caps());
        let job = store
            .create_job(image_edit_job_with(payload, "auto"))
            .expect("job creates");
        assert!(store
            .claim_next_job("worker-torch")
            .expect("torch claim ok")
            .is_none());
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .expect("mlx claims kolors advanced job");
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
    }
}

#[test]
fn flux_reference_job_routes_to_mlx() {
    let store = store("mlx-routing-flux-reference");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.1 reference/IP-Adapter (epic 3621) now runs natively on the Rust/MLX worker →
    // torch refuses it, mlx claims it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "flux_dev", "prompt": "p", "referenceAssetId": "asset_1" }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims flux reference job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn qwen_txt2img_and_strict_pose_route_to_mlx() {
    let store = store("mlx-routing-qwen");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // Plain qwen txt2img → MLX worker.
    let txt2img = store
        .create_job(image_job_with(
            json!({ "model": "qwen_image", "prompt": "a red fox" }),
            "auto",
        ))
        .expect("txt2img job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims qwen txt2img");
    assert_eq!(claimed.id, txt2img.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
    store
        .update_job_progress(
            &claimed.id,
            ProgressUpdate {
                status: JobStatus::Completed,
                stage: ProgressStage::Completed,
                progress: 1.0,
                message: "Done".to_owned(),
                error: None,
                result: Some(object(json!({ "assetIds": ["asset-qwen"] }))),
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
                worker_id: Some("worker-mlx".to_owned()),
            },
        )
        .expect("txt2img job completes");

    // A strict-pose qwen job routes to the MLX ControlNet path (epic 3401 / sc-3575).
    let pose = store
        .create_job(image_job_with(
            json!({
                "model": "qwen_image",
                "prompt": "a red fox",
                "advanced": { "poses": [{ "id": "p1" }] }
            }),
            "auto",
        ))
        .expect("pose job creates");
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims qwen pose job");
    assert_eq!(claimed.id, pose.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn flux2_klein_variants_route_to_mlx_worker() {
    let store = store("mlx-routing-flux2");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // All three FLUX.2-klein txt2img variants + FLUX.2-dev (MLX-only family) route to the mlx
    // worker (dev is txt2img-only today — epic 5914 / sc-5921).
    for model in [
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
        "flux2_dev",
    ] {
        let job = store
            .create_job(image_job_with(
                json!({ "model": model, "prompt": "a red fox" }),
                "auto",
            ))
            .unwrap_or_else(|_| panic!("{model} job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model}"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        // Completing the job returns the mlx worker to idle (the deferral only fires
        // toward an *idle* mlx worker), so the next variant defers to it too.
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("worker-mlx".to_owned()),
                },
            )
            .expect("complete job");
    }
}

#[test]
fn flux2_edit_reference_job_routes_to_mlx_worker() {
    let store = store("mlx-routing-flux2-edit");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.2 is MLX-only, so an edit/reference job (sc-3029) routes to the mlx worker
    // (sc-3025 kept these on Python; the edit path now exists on Rust).
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "flux2_klein_9b_kv",
                "mode": "edit_image",
                "prompt": "make it golden hour",
                "sourceAssetId": "asset_1"
            }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims flux2 edit job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn sdxl_and_realvisxl_route_to_mlx_worker() {
    let store = store("mlx-routing-sdxl");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // `realvisxl_lightning` (sc-6075) is a txt2img-only few-step sibling on the same `sdxl`
    // engine, so a plain text-to-image job routes to MLX just like the base ids.
    for model in ["sdxl", "realvisxl", "realvisxl_lightning"] {
        let job = store
            .create_job(image_job_with(
                json!({ "model": model, "prompt": "a red fox" }),
                "auto",
            ))
            .unwrap_or_else(|_| panic!("{model} job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model}"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("worker-mlx".to_owned()),
                },
            )
            .expect("complete job");
    }

    // sc-3060: SDXL reference/IP-Adapter + edit_image (inpaint/outpaint) now run on the Rust
    // engine, so they route to the mlx worker (the torch worker defers).
    for payload in [
        json!({ "model": "sdxl", "prompt": "p", "referenceAssetId": "asset_1" }),
        json!({ "model": "sdxl", "prompt": "p", "mode": "edit_image", "sourceAssetId": "src_1" }),
        json!({ "model": "sdxl", "prompt": "p", "mode": "edit_image",
                "sourceAssetId": "src_1", "maskAssetId": "mask_1" }),
    ] {
        let job = store
            .create_job(image_job_with(payload, "auto"))
            .expect("advanced job creates");
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "sdxl advanced should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .expect("mlx claims sdxl advanced job");
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("worker-mlx".to_owned()),
                },
            )
            .expect("complete job");
    }

    // A third-party LyCORIS LoRA now applies on the SDXL merge path (epic 3641, sc-3671) → MLX.
    let lycoris = store
        .create_job(image_job_with(
            json!({ "model": "sdxl", "prompt": "p", "loras": [{ "networkType": "lycoris" }] }),
            "auto",
        ))
        .expect("lycoris job creates");
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims sdxl lycoris job");
    assert_eq!(claimed.id, lycoris.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn realvisxl_lightning_reference_falls_back_off_mlx() {
    // sc-6075: RealVisXL Lightning is txt2img-only — the engine's few-step `lightning` sampler
    // rejects reference/img2img conditioning. A reference or edit_image job is therefore NOT
    // MLX-eligible (`realvisxl_lightning_mlx_eligible`), so the torch worker claims it directly
    // instead of deferring to the mlx worker (the txt2img case is covered above).
    let store = store("mlx-routing-realvisxl-lightning");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    for payload in [
        json!({ "model": "realvisxl_lightning", "prompt": "p", "referenceAssetId": "asset_1" }),
        json!({ "model": "realvisxl_lightning", "prompt": "p", "mode": "edit_image",
                "sourceAssetId": "src_1" }),
    ] {
        let job = store
            .create_job(image_job_with(payload, "auto"))
            .expect("lightning conditioned job creates");
        let claimed = store
            .claim_next_job("worker-torch")
            .expect("torch claim ok")
            .expect("torch claims the txt2img-only-gated lightning job");
        assert_eq!(claimed.id, job.id);
        assert_ne!(
            claimed.assigned_gpu.as_deref(),
            Some("mlx"),
            "txt2img-only lightning conditioning must not route to the mlx worker"
        );
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("worker-torch".to_owned()),
                },
            )
            .expect("complete job");
    }
}

#[test]
fn image_detail_routes_to_mlx_worker() {
    // sc-3060: the tile-ControlNet detail refine (`image_detail`) now runs on the Rust
    // engine for SDXL-family backbones, so it routes to the `mlx` worker (the torch worker
    // defers); a third-party LyCORIS LoRA also runs on MLX now (epic 3641, sc-3671).
    let store = store("mlx-routing-detail");
    let caps = vec![
        WorkerCapability::Gpu,
        WorkerCapability::ImageGenerate,
        WorkerCapability::ImageDetail,
    ];
    register_gpu_worker(&store, "worker-torch", "mps", caps.clone());
    register_gpu_worker(&store, "worker-mlx", "mlx", caps);

    let detail_job = |payload: Value| CreateJob {
        job_type: JobType::ImageDetail,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
        initial_status: None,
    };

    for model in ["sdxl", "realvisxl"] {
        let job = store
            .create_job(detail_job(
                json!({ "model": model, "sourceAssetId": "asset_src" }),
            ))
            .unwrap_or_else(|_| panic!("{model} detail job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} detail should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model} detail"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                    worker_id: Some("worker-mlx".to_owned()),
                },
            )
            .expect("complete detail job");
    }

    // A LyCORIS detail job now applies on the SDXL merge path (epic 3641) → MLX.
    let lycoris = store
        .create_job(detail_job(json!({
            "model": "realvisxl",
            "sourceAssetId": "asset_src",
            "loras": [{ "networkType": "lycoris" }]
        })))
        .expect("lycoris detail job creates");
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims lycoris detail job");
    assert_eq!(claimed.id, lycoris.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn non_mlx_model_image_job_is_not_routed_to_mlx_worker() {
    let store = store("mlx-routing-non-mlx-model");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // A torch-only image model with no mlx-gen engine (pulid_flux_dev — PuLID has no MLX crate;
    // Kolors base T2I is ported via sc-3875, InstantID via sc-3345, SenseNova-U1 via sc-3900) stays
    // on the Python path: the torch worker claims it without deferral, and the mlx worker refuses it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "pulid_flux_dev", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the non-MLX-model job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_worker_claims_eligible_job_with_idle_mps_worker_present() {
    // Regression for the auto-GPU deferral deadlock (sc-3289): an Apple-Silicon
    // mlx worker reports no utilization (the real `gpu_utilization("mlx")` probes
    // nvidia-smi and finds nothing -> None), while an idle Python mps worker does
    // report utilization. A queued auto MLX-eligible job (here flux2_klein_9b_kv
    // text_to_image) must be claimed by the mlx worker; the mps worker must defer
    // it. Before the fix, `dispatch_score` scored the no-utilization mlx worker as
    // a GPU with 0 MB free, so `should_defer_auto_gpu_claim` made the mlx worker
    // defer to the "healthier" mps worker, which deferred the same job back to the
    // mlx worker -> the job sat on "Waiting for an available worker" forever.
    let store = store("mlx-claims-with-mps-present");
    store
        .register_worker(RegisterWorker {
            worker_id: "mlx-worker".to_owned(),
            gpu_id: "mlx".to_owned(),
            gpu_name: Some("Apple Silicon (MLX)".to_owned()),
            capabilities: vec![
                WorkerCapability::Gpu,
                WorkerCapability::ImageGenerate,
                WorkerCapability::ImageDetail,
                WorkerCapability::VideoGenerate,
                WorkerCapability::LoraTrain,
                WorkerCapability::LoraTrainExecute,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("mlx worker registers");
    store
        .register_worker(RegisterWorker {
            worker_id: "mps-worker".to_owned(),
            gpu_id: "mps".to_owned(),
            gpu_name: Some("Apple GPU (unified)".to_owned()),
            capabilities: vec![
                WorkerCapability::Gpu,
                WorkerCapability::ImageGenerate,
                WorkerCapability::ImageEdit,
                WorkerCapability::VideoGenerate,
                WorkerCapability::LoraTrain,
                WorkerCapability::LoraTrainExecute,
            ],
            loaded_models: Vec::new(),
            utilization: Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(131_072),
                memory_used_mb: Some(1_318),
                memory_free_mb: Some(129_754),
                gpu_load_percent: Some(28.0),
            }),
        })
        .expect("mps worker registers");

    let job = store
        .create_job(image_job(object(json!({
            "model": "flux2_klein_9b_kv",
            "mode": "text_to_image",
            "loras": [],
            "advanced": { "resolution": "1024x1024" },
        }))))
        .expect("job creates");

    // The mps worker must defer (an idle mlx worker can run it).
    assert!(
        store
            .claim_next_job("mps-worker")
            .expect("mps claim ok")
            .is_none(),
        "mps worker should defer the flux2 job to the mlx worker"
    );

    // The mlx worker must claim it.
    let claimed = store
        .claim_next_job("mlx-worker")
        .expect("mlx claim ok")
        .expect("mlx worker should claim the flux2 t2i job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

/// sc-3448 — multiple workers claiming the same database concurrently must never
/// surface `database is locked`, and every job must be claimed exactly once.
///
/// Each thread uses its *own* `JobsStore` on the *same* db file, so the per-instance
/// `Mutex` no longer serializes them and claims genuinely race at the SQLite layer —
/// the cross-process contention the `busy_timeout` + `BEGIN IMMEDIATE` fix targets.
/// IMMEDIATE makes each claimer read the queued set only after holding the write lock,
/// so two claimers can't both see one job as `queued` (the old DEFERRED path raced the
/// read→write upgrade, which SQLite fails immediately as `database is locked`).
#[test]
fn concurrent_claims_never_lock_and_stay_exactly_once() {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::thread;
    use std::time::Duration;

    const WORKERS: usize = 4;
    const JOBS: usize = 60;

    let path = temp_db("concurrent-claim");
    let primary = JobsStore::new(path.clone());
    primary.initialize().expect("store initializes");

    for w in 0..WORKERS {
        primary
            .register_worker(RegisterWorker {
                worker_id: format!("worker-{w}"),
                gpu_id: format!("gpu-{w}"),
                gpu_name: Some(format!("GPU {w}")),
                capabilities: vec![WorkerCapability::ImageGenerate],
                loaded_models: Vec::new(),
                utilization: None,
            })
            .expect("worker registers");
    }
    for j in 0..JOBS {
        primary
            .create_job(image_job(object(json!({ "prompt": format!("p{j}") }))))
            .expect("job creates");
    }

    let claimed: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let errors: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let remaining = Arc::new(AtomicUsize::new(JOBS));

    let mut handles = Vec::new();
    for w in 0..WORKERS {
        let path = path.clone();
        let claimed = Arc::clone(&claimed);
        let errors = Arc::clone(&errors);
        let remaining = Arc::clone(&remaining);
        handles.push(thread::spawn(move || {
            let store = JobsStore::new(path);
            let worker_id = format!("worker-{w}");
            while remaining.load(Ordering::SeqCst) > 0 {
                match store.claim_next_job(&worker_id) {
                    Ok(Some(job)) => {
                        claimed.lock().unwrap().push(job.id.clone());
                        remaining.fetch_sub(1, Ordering::SeqCst);
                        // Free the worker so it keeps claiming and the queue drains.
                        if let Err(error) = store.update_job_progress(
                            &job.id,
                            ProgressUpdate {
                                status: JobStatus::Completed,
                                stage: ProgressStage::Completed,
                                progress: 1.0,
                                message: "done".to_owned(),
                                error: None,
                                result: None,
                                eta_seconds: None,
                                peak_gpu_memory_pct: None,
                                peak_gpu_load_pct: None,
                                backend: None,
                                worker_id: Some(worker_id.clone()),
                            },
                        ) {
                            errors.lock().unwrap().push(error.to_string());
                        }
                    }
                    Ok(None) => thread::sleep(Duration::from_millis(1)),
                    Err(error) => errors.lock().unwrap().push(error.to_string()),
                }
            }
        }));
    }
    for handle in handles {
        handle.join().expect("claimer thread joins");
    }

    let errors = errors.lock().unwrap();
    assert!(
        errors.is_empty(),
        "claims/updates must never error under contention; saw: {errors:?}"
    );
    let claimed = claimed.lock().unwrap();
    let unique: HashSet<&String> = claimed.iter().collect();
    assert_eq!(claimed.len(), JOBS, "every job claimed (count)");
    assert_eq!(unique.len(), JOBS, "no job claimed twice");
}

/// sc-8950 / F-148 — read-only methods (list_jobs/get_job/list_workers/
/// get_worker/queue_summary) no longer take the process-wide write mutex; they
/// rely on WAL reader isolation instead. This pins the property the change
/// depends on: a reader returns correct, committed data promptly even while a
/// writer is mid-transaction holding the SQLite write lock — it takes the WAL
/// snapshot rather than blocking until the writer commits.
#[test]
fn reads_proceed_while_a_writer_holds_the_write_lock() {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    let path = temp_db("read-during-write");
    let store = JobsStore::new(path.clone());
    store.initialize().expect("store initializes");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(object(json!({ "prompt": "committed" }))))
        .expect("job creates");
    // The committed message right after create — the reader must observe this,
    // never the writer's uncommitted overwrite below.
    let committed_message = created.message.clone();

    // A separate connection opens a BEGIN IMMEDIATE transaction (acquiring the
    // SQLite write lock) and holds it open for a beat WITHOUT committing, so its
    // pending mutation is invisible to snapshots. This models a real in-flight
    // writer; crucially it does NOT touch our JobsStore mutex, so a mutex-taking
    // reader could only stall on the SQLite lock — which WAL lets a reader skip.
    let (holding_tx, holding_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let writer_path = path.clone();
    let writer_job_id = created.id.clone();
    let writer = thread::spawn(move || {
        let connection = Connection::open(&writer_path).expect("writer db opens");
        connection
            .busy_timeout(Duration::from_millis(5000))
            .expect("busy timeout set");
        connection
            .execute_batch("begin immediate")
            .expect("write lock acquired");
        connection
            .execute(
                "update jobs set message = 'uncommitted' where id = ?1",
                params![writer_job_id],
            )
            .expect("uncommitted write applied");
        holding_tx.send(()).expect("signal holding");
        // Hold the write lock until the reader has finished.
        release_rx.recv().expect("await release");
        connection.execute_batch("rollback").expect("rollback");
    });

    holding_rx.recv().expect("writer holds the lock");

    // The read must return the COMMITTED snapshot (original message), quickly,
    // without waiting on the writer to release. A generous bound catches a
    // regression where a reader serializes behind the writer.
    let started = Instant::now();
    let loaded = store
        .get_job(&created.id)
        .expect("read while write in flight");
    let jobs = store
        .list_jobs(None, None, 100)
        .expect("list while write in flight");
    let summary = store
        .queue_summary()
        .expect("summary while write in flight");
    let elapsed = started.elapsed();

    release_tx.send(()).expect("release writer");
    writer.join().expect("writer thread joins");

    assert_eq!(
        loaded.message, committed_message,
        "reader must see the committed snapshot, not the writer's uncommitted change"
    );
    assert_ne!(
        loaded.message, "uncommitted",
        "reader must never observe the writer's uncommitted overwrite"
    );
    assert_eq!(
        jobs.len(),
        1,
        "the one committed job is visible to the reader"
    );
    assert!(
        summary.active_jobs.iter().any(|job| job.id == created.id),
        "queue summary reflects the committed job"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "reads must not block for the writer's full hold; took {elapsed:?}"
    );
}

/// sc-4172 — a zombie worker's late progress report must not resurrect a job
/// the stale sweep marked `interrupted`; terminal statuses are immutable except
/// for an idempotent re-report of the same terminal status.
#[test]
fn progress_cannot_resurrect_terminal_jobs() {
    fn report(status: JobStatus, stage: ProgressStage, worker_id: Option<&str>) -> ProgressUpdate {
        ProgressUpdate {
            status,
            stage,
            progress: 0.9,
            message: "late report".to_owned(),
            error: None,
            result: None,
            eta_seconds: None,
            peak_gpu_memory_pct: None,
            peak_gpu_load_pct: None,
            backend: None,
            worker_id: worker_id.map(str::to_owned),
        }
    }

    let store = store("terminal-immutable");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    // Age the worker out and sweep its job to interrupted (worker_id cleared).
    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = ?1",
            params!["worker-1"],
        )
        .expect("worker timestamp updates");
    store
        .mark_stale_workers_interrupted(1)
        .expect("sweep succeeds");

    // The zombie's "running" report must be rejected, not resurrect the job.
    let error = store
        .update_job_progress(
            &created.id,
            report(JobStatus::Running, ProgressStage::Running, Some("worker-1")),
        )
        .expect_err("terminal job rejects an active-status report");
    assert!(matches!(
        error,
        JobsStoreError::TerminalJobImmutable { ref status, .. } if status == "interrupted"
    ));

    // Its late "completed" report must not flip the terminal status either.
    let error = store
        .update_job_progress(
            &created.id,
            report(
                JobStatus::Completed,
                ProgressStage::Completed,
                Some("worker-1"),
            ),
        )
        .expect_err("terminal job rejects a different terminal status");
    assert!(matches!(error, JobsStoreError::TerminalJobImmutable { .. }));

    // An idempotent re-report of the same terminal status is a no-op success
    // (e.g. a worker retrying its own "canceled" POST).
    let job = store
        .update_job_progress(
            &created.id,
            report(
                JobStatus::Interrupted,
                ProgressStage::Interrupted,
                Some("worker-1"),
            ),
        )
        .expect("same-terminal re-report succeeds");
    assert_eq!(job.status, JobStatus::Interrupted);
    assert_eq!(
        job.message, "Lost contact with the worker.",
        "no-op re-report must not rewrite the row"
    );

    let job = store.get_job(&created.id).expect("job loads");
    assert_eq!(job.status, JobStatus::Interrupted);
    assert_eq!(job.worker_id, None);
}

/// sc-4172 / sc-5751 — progress reports must name the claimed job's owner.
#[test]
fn progress_from_non_owner_worker_is_rejected() {
    fn running(worker_id: Option<&str>) -> ProgressUpdate {
        ProgressUpdate {
            status: JobStatus::Running,
            stage: ProgressStage::Running,
            progress: 0.5,
            message: "running".to_owned(),
            error: None,
            result: None,
            eta_seconds: None,
            peak_gpu_memory_pct: None,
            peak_gpu_load_pct: None,
            backend: None,
            worker_id: worker_id.map(str::to_owned),
        }
    }

    let store = store("non-owner-progress");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    let error = store
        .update_job_progress(&created.id, running(Some("worker-2")))
        .expect_err("non-owner report is rejected");
    assert!(matches!(error, JobsStoreError::NotJobOwner { .. }));
    let job = store.get_job(&created.id).expect("job loads");
    assert_eq!(
        job.status,
        JobStatus::Preparing,
        "rejected write must not land"
    );

    let job = store
        .update_job_progress(&created.id, running(Some("worker-1")))
        .expect("owner report lands");
    assert_eq!(job.status, JobStatus::Running);

    let error = store
        .update_job_progress(&created.id, running(None))
        .expect_err("ownerless report on an owned job is rejected");
    assert!(matches!(error, JobsStoreError::NotJobOwner { .. }));
}

/// Regression for the Anima routing gap (epic 10512 / sc-10523): `anima_base`/`anima_aesthetic`/
/// `anima_turbo` shipped as `mlx_routed = true` rows in `IMAGE_MODEL_CAPS`, but
/// `image_request_mlx_eligible` had no dispatch arm for them, so they fell through to `_ => false`.
/// The mlx worker then refused to claim them (`worker_supports_job`), and because Anima advertises
/// no candle/torch lane (`candle_routed = false`, macOnly) NOTHING could claim the job — every
/// Anima generation sat on "Waiting for an available worker." forever.
#[test]
fn mlx_worker_claims_anima_text_to_image_jobs() {
    for model in ["anima_base", "anima_aesthetic", "anima_turbo"] {
        // A fresh store + worker per model: a worker that claims a job goes `busy` and holds a
        // `current_job_id`, so it cannot claim a second job in the same store.
        let store = store(&format!("anima-claim-{model}"));
        store
            .register_worker(RegisterWorker {
                worker_id: "mlx-worker".to_owned(),
                gpu_id: "mlx".to_owned(),
                gpu_name: Some("Apple Silicon (MLX)".to_owned()),
                capabilities: vec![
                    WorkerCapability::Gpu,
                    WorkerCapability::ImageGenerate,
                    WorkerCapability::ImageDetail,
                ],
                loaded_models: Vec::new(),
                utilization: None,
            })
            .expect("mlx worker registers");

        let mut payload = Map::new();
        payload.insert("model".to_owned(), Value::String(model.to_owned()));
        payload.insert("mode".to_owned(), Value::String("text_to_image".to_owned()));
        let job = store.create_job(image_job(payload)).expect("job created");

        let claimed = store
            .claim_next_job("mlx-worker")
            .expect("claim ok")
            .unwrap_or_else(|| {
                panic!(
                    "{model}: mlx worker must claim the Anima job, but nothing claimed it \
                     (job {} sits on 'Waiting for an available worker.')",
                    job.id
                )
            });
        assert_eq!(claimed.id, job.id, "{model}: claimed the wrong job");
        assert_eq!(
            claimed.assigned_gpu.as_deref(),
            Some("mlx"),
            "{model}: Anima must land on the mlx worker"
        );
    }
}

/// Anima has no edit path (`capabilities: ["text_to_image"]`), so an `edit_image` request must NOT
/// be MLX-eligible — the defensive shape SANA / SD3.5 / Krea / Lens use.
#[test]
fn mlx_worker_refuses_anima_edit_image_jobs() {
    let store = store("anima-refuses-edit");
    store
        .register_worker(RegisterWorker {
            worker_id: "mlx-worker".to_owned(),
            gpu_id: "mlx".to_owned(),
            gpu_name: Some("Apple Silicon (MLX)".to_owned()),
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("mlx worker registers");

    let mut payload = Map::new();
    payload.insert("model".to_owned(), Value::String("anima_base".to_owned()));
    payload.insert("mode".to_owned(), Value::String("edit_image".to_owned()));
    payload.insert(
        "sourceAssetId".to_owned(),
        Value::String("asset-1".to_owned()),
    );
    store.create_job(image_job(payload)).expect("job created");

    assert!(
        store
            .claim_next_job("mlx-worker")
            .expect("claim ok")
            .is_none(),
        "an Anima edit_image job must not be claimed by the mlx worker"
    );
}
