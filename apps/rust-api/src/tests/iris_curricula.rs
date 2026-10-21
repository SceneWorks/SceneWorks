//! Iris-3B curricula, REPA, holdout validation and distributed training through the API (story
//! sc-25689, epic sc-25678): the official stage presets, the pre-launch plan check (global batch,
//! topology, teacher, the engine's own refusal), the curriculum lineage of a parent run, the REPA
//! drop a replicated full-optimizer resume refuses, the receipts on the job and on the registered
//! output, holdout items and the run's metrics.
use super::iris_training::{
    complete, dataset, iris_env, iris_target, project, seed_iris_install, submit,
};
use super::support::*;

const IRIS_TARGET: &str = "iris_3b_generation";
const BACKBONE_REPO: &str = "speridlabs/iris-3b";
const BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
const RELEASE_CONFIG: &str =
    include_str!("../../../../crates/sceneworks-worker/tests/fixtures/iris/config.yaml");

fn snapshot_file(
    data_dir: &std::path::Path,
    repo: &str,
    revision: &str,
    file: &str,
) -> std::path::PathBuf {
    let root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo).unwrap();
    let path = root.join("snapshots").join(revision).join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::create_dir_all(root.join("refs")).unwrap();
    std::fs::write(root.join("refs").join("main"), revision).unwrap();
    path
}

/// The installed backbone's real `config.yaml`, so the API's engine resolution runs.
fn stage_backbone_config(settings: &Settings) {
    std::fs::write(
        snapshot_file(
            &settings.data_dir,
            BACKBONE_REPO,
            BACKBONE_REVISION,
            "config.yaml",
        ),
        RELEASE_CONFIG,
    )
    .unwrap();
}

/// The REPA teacher install at its pinned size (a sparse file: the API checks the install, the
/// worker hashes it before a run).
fn stage_teacher(settings: &Settings) {
    use sceneworks_core::training::iris::curricula::*;
    let path = snapshot_file(
        &settings.data_dir,
        IRIS_REPA_TEACHER_REPO,
        IRIS_REPA_TEACHER_REVISION,
        IRIS_REPA_TEACHER_FILE,
    );
    let file = std::fs::File::create(path).unwrap();
    file.set_len(IRIS_REPA_TEACHER_SIZE_BYTES).unwrap();
}

async fn plan_check(app: &axum::Router, config: Value) -> Value {
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/training/iris/plan-check",
        json!({ "targetId": IRIS_TARGET, "config": config }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

/// The four official stages come from the engine as editable configs with their stage, parent and
/// data-shaping identity; edited for one host, a first stage submits.
#[tokio::test]
async fn the_official_stage_presets_are_served_and_submit_once_edited() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let (status, body) = request(
        app.clone(),
        "GET",
        "/api/v1/training/iris/stage-presets",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let presets = body["presets"].as_array().unwrap();
    let ids: Vec<&str> = presets.iter().map(|p| p["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["stage1_256", "stage2_512", "stage3_1024", "sft_1024"]);
    assert_eq!(presets[1]["parentStage"], "stage1_256");
    let stage1 = presets[0]["config"].clone();
    assert_eq!(stage1["advanced"]["iris"]["stage"], "stage1_256");
    assert_eq!(stage1["advanced"]["iris"]["repa"]["weight"], 0.5);
    assert_eq!(stage1["advanced"]["networkType"], "full");

    // Verbatim, the 64-GPU recipe is refused before launch — typed, naming the topology.
    let check = plan_check(&app, stage1.clone()).await;
    assert_eq!(check["ok"], false, "{check}");
    assert!(
        check["refusal"]["message"]
            .as_str()
            .is_some_and(|text| text.contains("mesh") || text.contains("lobal batch")),
        "{check}"
    );
    // Edited for one rank (and with the teacher missing) the check names the teacher.
    let mut edited = stage1.clone();
    let iris = edited["advanced"]["iris"].as_object_mut().unwrap();
    iris.remove("expected_global_batch");
    iris.insert(
        "dist".to_owned(),
        json!({"strategy": "fsdp2", "sharding": "full"}),
    );
    let check = plan_check(&app, edited.clone()).await;
    assert_eq!(check["refusal"]["kind"], "missing_teacher", "{check}");
    assert_eq!(check["teacher"]["installed"], false);
    assert_eq!(check["teacher"]["license"], "Apache-2.0");
    stage_teacher(&settings);
    let check = plan_check(&app, edited.clone()).await;
    assert_eq!(check["ok"], true, "{check}");
    assert_eq!(check["globalBatch"], 16);
    assert_eq!(check["receipt"]["stage"], "stage1_256");
    assert_eq!(check["receipt"]["dist"]["strategy"], "fsdp2");

    // A continuing stage without its parent run is refused with the lineage it needs.
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, json!([{ "caption": { "text": "a" } }])).await;
    let mut stage2 = presets[1]["config"].clone();
    let iris = stage2["advanced"]["iris"].as_object_mut().unwrap();
    iris.remove("expected_global_batch");
    iris.insert(
        "dist".to_owned(),
        json!({"strategy": "fsdp2", "sharding": "full"}),
    );
    iris.remove("resume_data_policy");
    iris.remove("override_lr_on_resume");
    iris.remove("resume_optimizer");
    let (status, error) = submit(&app, &project_id, &dataset_id, stage2, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "lineage", "{error}");
}

/// The plan check reports the global batch and the engine's own typed refusal (the backbone is
/// installed, so the API resolves the plan exactly as the trainer does).
#[tokio::test]
async fn the_plan_check_shows_the_global_batch_and_the_engines_refusal() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    stage_backbone_config(&settings);
    stage_teacher(&settings);
    let defaults = iris_target(&app).await["defaults"].clone();

    let mut config = defaults.clone();
    config["batchSize"] = json!(2);
    config["gradientAccumulation"] = json!(2);
    config["saveEvery"] = json!(250);
    config["advanced"]["iris"]["expected_global_batch"] = json!(8);
    config["advanced"]["iris"]["dist"] =
        json!({"strategy": "ddp", "world": {"world_size": 2, "share_device": true}});
    let check = plan_check(&app, config.clone()).await;
    assert_eq!(check["globalBatch"], 8, "{check}");
    assert_eq!(check["worldSize"], 2);
    assert_eq!(check["ok"], true, "{check}");
    assert_eq!(check["engineChecked"], cfg!(target_os = "macos"));
    // sc-25690: one rank's floor in GiB with the provenance of the row it was read from (the
    // default 256² adapter row is a declaration) — no blanket `measured` flag.
    let memory = &check["memory"];
    assert_eq!(memory["unit"], "GiB", "{check}");
    assert!(memory["perRankFloorGib"].as_u64() > Some(0), "{check}");
    assert_eq!(memory["provenance"], "declared", "{check}");
    assert!(memory.get("measured").is_none(), "{check}");
    let mut full = config.clone();
    full["resolution"] = json!(512);
    full["advanced"]["networkType"] = json!("full");
    full["advanced"]["iris"] = json!({});
    full["batchSize"] = json!(1);
    full["gradientAccumulation"] = json!(1);
    assert_eq!(
        plan_check(&app, full).await["memory"]["provenance"],
        "measured",
        "a single-rank full 512² run reads a measured row"
    );
    if cfg!(target_os = "macos") {
        assert_eq!(check["resolved"]["globalBatch"], 8, "{check}");
        assert!(check["resolved"]["dataIdentity"].is_string(), "{check}");
    }

    // The expected batch no longer matches with three ranks: typed, before launch.
    config["advanced"]["iris"]["dist"]["world"]["world_size"] = json!(3);
    let check = plan_check(&app, config.clone()).await;
    assert_eq!(check["ok"], false);
    assert!(
        check["refusal"]["message"]
            .as_str()
            .unwrap()
            .contains("expected global batch 8"),
        "{check}"
    );

    // Only the engine knows the trunk depth: a REPA capture block past it.
    let mut repa = defaults.clone();
    repa["advanced"]["iris"]["repa"] = json!({"weight": 0.5, "layer": 99});
    let check = plan_check(&app, repa).await;
    if cfg!(target_os = "macos") {
        assert_eq!(check["ok"], false, "{check}");
        assert!(
            check["refusal"]["message"]
                .as_str()
                .unwrap()
                .contains("capture block"),
            "{check}"
        );
    }

    // On the Mac (one Metal device) more ranks than devices need the share-device opt-in.
    if cfg!(target_os = "macos") {
        let mut unshared = defaults.clone();
        unshared["advanced"]["iris"]["dist"] = json!({"world": {"world_size": 2}});
        let check = plan_check(&app, unshared).await;
        assert_eq!(check["refusal"]["kind"], "unsupported_topology", "{check}");
    }
}

/// A stage-2 run continues the stage-1 run it names as a new phase; the lineage, the REPA drop a
/// replicated full-optimizer resume refuses, and the receipts on the job and the registered
/// adapter.
#[tokio::test]
async fn a_continuing_stage_names_its_parent_run_and_carries_the_receipts() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    stage_teacher(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(
        &app,
        &project_id,
        json!([
            { "caption": { "text": "a red square" } },
            { "caption": { "text": "a blue circle" }, "split": "holdout" }
        ]),
    )
    .await;
    let defaults = iris_target(&app).await["defaults"].clone();

    // Stage 1 with REPA and holdout validation.
    let mut stage1 = defaults.clone();
    stage1["advanced"]["iris"]["stage"] = json!("stage1_256");
    stage1["advanced"]["iris"]["repa"] = json!({"weight": 0.5});
    stage1["advanced"]["iris"]["val_every_steps"] = json!(50);
    stage1["advanced"]["mixedPrecision"] = json!("fp16");
    let (status, job) = submit(&app, &project_id, &dataset_id, stage1.clone(), false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let receipts = &job["payload"]["irisReceipts"];
    assert_eq!(receipts["stage"], "stage1_256", "{receipts}");
    assert_eq!(receipts["dataset"]["holdoutItems"], 1, "{receipts}");
    assert_eq!(receipts["dist"]["worldSize"], 1);
    assert_eq!(job["payload"]["manifestEntry"]["irisReceipts"], *receipts);
    let plan = &job["payload"]["plan"];
    assert_eq!(plan["dataset"]["items"][1]["split"], "holdout");
    assert_eq!(plan["config"]["advanced"]["mixedPrecision"], "fp16");
    let stage1_job = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();
    let stem = file_name.trim_end_matches(".safetensors").to_owned();
    claim_training_job_as_mlx_worker(&app, &stage1_job).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    write_test_safetensors(&output_dir.join(&file_name));
    let checkpoints = output_dir.join(format!("{stem}.checkpoints"));
    std::fs::create_dir_all(checkpoints.join("step_00000250")).unwrap();
    std::fs::write(checkpoints.join("step_00000250/state.json"), b"{}").unwrap();
    std::fs::write(
        checkpoints.join("metrics.jsonl"),
        [
            json!({"kind": "run", "stage": "stage1_256", "trainer": "iris_3b"}),
            json!({"kind": "step", "step": 50, "loss": 0.4}),
            json!({"kind": "validation", "step": 50, "val/loss": 0.31, "val/loss_t0": 0.2}),
            json!({"kind": "validation", "step": 100, "val/loss": 0.29, "val/loss_t0": 0.19}),
        ]
        .iter()
        .map(|record| record.to_string() + "\n")
        .collect::<String>(),
    )
    .unwrap();
    complete(&app, &stage1_job, "completed", json!({})).await;

    // The run's metrics, grouped for the Studio.
    let (status, metrics) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/jobs/{stage1_job}/iris-metrics"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{metrics}");
    assert_eq!(metrics["available"], true);
    assert_eq!(metrics["run"]["stage"], "stage1_256");
    assert_eq!(metrics["validation"].as_array().unwrap().len(), 2);
    assert_eq!(metrics["validation"][1]["val/loss"], 0.29);
    assert_eq!(metrics["steps"].as_array().unwrap().len(), 1);

    // The registered adapter carries the receipts.
    let (_, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    let entry = loras
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Iris Style")
        .expect("the adapter registered")
        .clone();
    assert_eq!(entry["irisReceipts"]["stage"], "stage1_256", "{entry}");

    // Stage 2 continues it: new_phase required, and dropping REPA under a replicated full
    // optimizer resume is the engine's refusal — core resumes without the projector.
    let mut stage2 = defaults.clone();
    stage2["advanced"]["resumeFromJobId"] = json!(stage1_job);
    stage2["advanced"]["iris"]["stage"] = json!("stage2_512");
    stage2["advanced"]["iris"]["parent_stage"] = json!("stage1_256");
    stage2["advanced"]["iris"]["resume_data_policy"] = json!("exact");
    let (status, error) = submit(&app, &project_id, &dataset_id, stage2.clone(), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "lineage", "{error}");
    stage2["advanced"]["iris"]["resume_data_policy"] = json!("new_phase");
    let (status, error) = submit(&app, &project_id, &dataset_id, stage2.clone(), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "unsupported", "{error}");
    assert!(error.to_string().contains("resume_optimizer"), "{error}");
    stage2["advanced"]["iris"]["resume_optimizer"] = json!("core");
    let (status, job) = submit(&app, &project_id, &dataset_id, stage2.clone(), false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let receipts = &job["payload"]["irisReceipts"];
    assert_eq!(receipts["stage"], "stage2_512");
    assert_eq!(receipts["parentStage"], "stage1_256");
    assert_eq!(receipts["parentRunJobId"], json!(stage1_job));

    // A run of an unrelated stage is not a parent.
    let mut wrong = stage2.clone();
    wrong["advanced"]["iris"]["stage"] = json!("sft_1024");
    wrong["advanced"]["iris"]["parent_stage"] = json!("stage3_1024");
    let (status, error) = submit(&app, &project_id, &dataset_id, wrong, false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "lineage", "{error}");
}

/// Holdout validation needs holdout items; REPA needs the installed teacher.
#[tokio::test]
async fn holdout_and_teacher_refusals_are_typed_at_submit() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, json!([{ "caption": { "text": "a" } }])).await;
    let defaults = iris_target(&app).await["defaults"].clone();
    let mut config = defaults.clone();
    config["advanced"]["iris"]["val_every_steps"] = json!(10);
    let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error.to_string().contains("holdout"), "{error}");

    let mut config = defaults.clone();
    config["advanced"]["iris"]["repa"] = json!({"weight": 0.5});
    let (status, error) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "missing_teacher", "{error}");
    stage_teacher(&settings);
    let (status, job) = submit(&app, &project_id, &dataset_id, config, false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["irisReceipts"]["repa"]["weight"], 0.5);
}

async fn register_gpu_worker(app: &axum::Router, worker_id: &str, gpu_id: &str) {
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/workers/register",
        json!({
            "workerId": worker_id,
            "gpuId": gpu_id,
            "gpuName": "Test GPU",
            "capabilities": ["gpu", "lora_train", "lora_train_execute"],
            "loadedModels": []
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The topology is judged before launch on EVERY platform, against the devices the registered
/// workers own: a job runs on one worker, so more ranks than the biggest worker's devices need the
/// share-device opt-in — at the plan check and at submit, never only after a worker claims it.
#[tokio::test]
async fn the_topology_is_refused_before_launch_against_the_registered_workers() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let defaults = iris_target(&app).await["defaults"].clone();
    let ranks = |world: u64, share: bool| {
        let mut config = defaults.clone();
        config["advanced"]["iris"]["dist"] =
            json!({"world": {"world_size": world, "share_device": share}});
        config
    };

    // No worker registered yet, then two single-GPU workers: one device per job.
    let check = plan_check(&app, ranks(2, false)).await;
    assert_eq!(check["refusal"]["kind"], "unsupported_topology", "{check}");
    register_gpu_worker(&app, "gpu-0", "0").await;
    register_gpu_worker(&app, "gpu-1", "1").await;
    let check = plan_check(&app, ranks(2, false)).await;
    assert_eq!(check["refusal"]["kind"], "unsupported_topology", "{check}");
    assert!(
        check["refusal"]["message"]
            .as_str()
            .unwrap()
            .contains("owns 1"),
        "{check}"
    );
    let check = plan_check(&app, ranks(2, true)).await;
    assert_ne!(check["refusal"]["kind"], "unsupported_topology", "{check}");

    // A worker configured with two GPUs hosts two unshared ranks; three still need sharing.
    register_gpu_worker(&app, "gpu-pair", "0,1").await;
    let check = plan_check(&app, ranks(2, false)).await;
    assert_ne!(check["refusal"]["kind"], "unsupported_topology", "{check}");

    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, json!([{ "caption": { "text": "a" } }])).await;
    let (status, error) = submit(&app, &project_id, &dataset_id, ranks(3, false), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "unsupported_topology", "{error}");
    let (status, job) = submit(&app, &project_id, &dataset_id, ranks(2, false), false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
}
