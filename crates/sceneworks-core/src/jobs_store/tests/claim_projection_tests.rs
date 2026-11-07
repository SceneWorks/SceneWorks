use super::*;
use serde_json::json;

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("object fixture").clone()
}

fn worker(gpu_id: &str, capabilities: &[&str], loaded_models: &[&str]) -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": format!("worker-{gpu_id}"),
        "gpuId": gpu_id,
        "status": "idle",
        "capabilities": capabilities,
        "loadedModels": loaded_models,
        "registeredAt": "2026-08-26T00:00:00Z",
        "lastSeenAt": "2026-08-26T00:00:00Z"
    }))
    .expect("valid worker")
}

fn create_job(store: &JobsStore, job_type: JobType, payload: Value) -> JobSnapshot {
    store
        .create_job(CreateJob {
            job_type,
            project_id: None,
            project_name: None,
            payload: object(payload),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .expect("routing corpus job creates")
}

#[test]
fn persisted_claim_projection_matches_full_worker_policy_for_routing_corpus() {
    let directory = tempfile::tempdir().expect("temp directory");
    let db_path = directory.path().join("jobs.db");
    let store = JobsStore::new(&db_path);
    store.initialize().expect("store initializes");

    // Representative cells from the existing routing corpus: compatible and
    // incompatible image model/mode/tier/adapter shapes plus video, training,
    // preview capability, and native upscaling.
    let jobs = vec![
        create_job(
            &store,
            JobType::ImageGenerate,
            json!({ "model": "z_image_turbo", "prompt": "base" }),
        ),
        create_job(
            &store,
            JobType::ImageGenerate,
            json!({ "model": "pulid_flux_dev", "prompt": "unsupported model" }),
        ),
        create_job(
            &store,
            JobType::ImageGenerate,
            json!({ "model": "flux2_dev", "mode": "style_variations", "prompt": "mode" }),
        ),
        create_job(
            &store,
            JobType::ImageGenerate,
            json!({ "model": "flux_schnell", "advanced": { "mlxQuantize": 6 } }),
        ),
        create_job(
            &store,
            JobType::ImageGenerate,
            json!({ "model": "boogu_image", "loras": [{ "id": "style" }] }),
        ),
        create_job(
            &store,
            JobType::VideoGenerate,
            json!({ "model": "ltx_2_3", "mode": "text_to_video", "prompt": "clip" }),
        ),
        create_job(
            &store,
            JobType::LoraTrain,
            json!({ "dryRun": false, "plan": { "kernel": "krea_lora" } }),
        ),
        create_job(&store, JobType::PersonDetect, json!({ "preview": true })),
        create_job(
            &store,
            JobType::ImageUpscale,
            json!({ "engine": "seedvr2", "model": "seedvr2" }),
        ),
    ];
    let workers = [
        worker(
            "mlx",
            &[
                "gpu",
                "image_generate",
                "video_generate",
                "lora_train",
                "lora_train_execute",
                "person_detect_preview",
                "image_upscale",
            ],
            &["z_image_turbo"],
        ),
        worker(
            "0",
            &[
                "gpu",
                "image_generate",
                "video_generate",
                "lora_train",
                "lora_train_execute",
                "person_detect_preview",
                "image_upscale",
                "candle",
            ],
            &["ltx_2_3"],
        ),
        worker(
            "gpu-generic",
            &[
                "gpu",
                "image_generate",
                "video_generate",
                "lora_train",
                "lora_train_execute",
                "person_detect_preview",
                "image_upscale",
            ],
            &[],
        ),
    ];

    let connection = Connection::open(&db_path).expect("db opens");
    for job in jobs {
        let candidate = connection
            .query_row(
                "select id, type, queue_rank, requested_gpu,
                        claim_mlx_eligible, claim_candle_eligible,
                        claim_candle_pose_reject, claim_training_mlx_only,
                        claim_seedvr2_upscale, claim_required_capability,
                        claim_real_training, claim_model_key_1, claim_model_key_2,
                        claim_model_key_3, claim_model_key_4
                   from jobs where id = ?1",
                params![job.id],
                row_to_claim_candidate,
            )
            .expect("persisted candidate reads");
        assert_eq!(
            candidate.facts,
            ClaimRoutingFacts::from_job(&job),
            "persisted facts must round-trip the routing corpus for {}",
            job.id
        );
        for worker in &workers {
            assert_eq!(
                worker_supports_claim_routing_facts(worker, &candidate.facts),
                worker_supports_job(worker, &job),
                "facts/full policy parity for job {} on worker {}",
                job.id,
                worker.id
            );
        }
    }
}

/// The tier-aware policy changes persisted facts, not just new request routing. A pre-S8 FP8R
/// job must not retain its generic image_generate requirement and be handed to a pre-Ada worker
/// after upgrading; startup migrates its required marker and eligible lane facts.
#[test]
fn startup_reprojects_pre_flux3_tier_claim_facts() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("jobs.db");
    let store = JobsStore::new(&db_path);
    store.initialize().unwrap();
    let job = create_job(
        &store,
        JobType::ImageGenerate,
        json!({
            "model": "flux3_klein", "prompt": "a lighthouse", "mode": "text_to_image",
            "advanced": {"quantTier": "fp8r"}
        }),
    );
    let connection = Connection::open(&db_path).unwrap();
    connection
        .execute(
            "update jobs set claim_facts_version = 2, claim_required_capability = 'image_generate',
         claim_mlx_eligible = 1, claim_candle_eligible = 0 where id = ?1",
            params![job.id],
        )
        .unwrap();
    store.initialize().unwrap();
    let (version, capability, mlx, candle): (i64, String, bool, bool) = connection
        .query_row(
            "select claim_facts_version, claim_required_capability, claim_mlx_eligible,
         claim_candle_eligible from jobs where id = ?1",
            params![job.id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(version, CLAIM_ROUTING_FACTS_VERSION);
    assert!(
        version > 2,
        "the version bump is what migrates persisted pre-tier facts"
    );
    assert_eq!(capability, "fp8r");
    assert!(!mlx && candle);
}
