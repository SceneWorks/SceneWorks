//! Iris-3B generation training through the API (epic sc-25678; S7 sc-25685 MLX, S8 sc-25686
//! Candle): the target's submit-time contract, the pre-launch data walk, resume / load-from
//! resolution, registration of the adapter and of the full-model export (and of an export a
//! cancel published), and generation of a registered full checkpoint through the `iris_3b` engine.
use super::support::*;

const IRIS_TARGET: &str = "iris_3b_generation";

/// Holds the HF-env lock (via [`isolate_hf_cache`]) and pins the MLX unified-memory emulation cap,
/// so the macOS submit-time memory floor judges a deterministic budget on any host.
pub(super) struct IrisEnv {
    _hf: HfCacheEnvGuard,
    previous_cap: Option<String>,
}

const MLX_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

pub(super) fn iris_env(cap_gb: &str) -> IrisEnv {
    let hf = isolate_hf_cache();
    let previous_cap = std::env::var(MLX_CAP_ENV).ok();
    std::env::set_var(MLX_CAP_ENV, cap_gb);
    IrisEnv {
        _hf: hf,
        previous_cap,
    }
}

impl Drop for IrisEnv {
    fn drop(&mut self) {
        match &self.previous_cap {
            Some(previous) => std::env::set_var(MLX_CAP_ENV, previous),
            None => std::env::remove_var(MLX_CAP_ENV),
        }
    }
}

/// The adapter registry accepts the `iris` family once the generation install advertises it (S3,
/// sc-25681, owns that declaration); stand that in so the training guardrail is satisfied.
pub(super) fn seed_iris_install(settings: &Settings) {
    seed_installed_training_base(&settings.data_dir, "iris_3b");
    seed_iris_generation_files(&settings.data_dir, true, true);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir");
    std::fs::write(
        manifests.join("user.models.jsonc"),
        serde_json::to_vec_pretty(&json!({
            "models": [{
                "id": "iris_adapter_host",
                "name": "Iris adapter host",
                "type": "image",
                "family": "iris",
                "loraCompatibility": { "families": ["iris"], "types": ["lora", "lokr"] }
            }]
        }))
        .unwrap(),
    )
    .expect("user models manifest");
}

const IRIS_BACKBONE_REPO: &str = "speridlabs/iris-3b";
const IRIS_BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
const IRIS_TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const IRIS_TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";
const IRIS_RELEASE_CONFIG: &str =
    include_str!("../../../../crates/sceneworks-worker/tests/fixtures/iris/config.yaml");

fn iris_snapshot_file(data_dir: &std::path::Path, repo: &str, revision: &str, file: &str) {
    let root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo).unwrap();
    let path = root.join("snapshots").join(revision).join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::create_dir_all(root.join("refs")).unwrap();
    std::fs::write(root.join("refs").join("main"), revision).unwrap();
    let bytes: &[u8] = if file == "config.yaml" {
        IRIS_RELEASE_CONFIG.as_bytes()
    } else {
        b"x"
    };
    std::fs::write(path, bytes).unwrap();
}

/// The generation install's pinned files, per task resource (S7): the ROOT backbone
/// (`config.yaml` + `model.safetensors` at the pinned revision) and/or the Qwen3-VL text encoder.
pub(super) fn seed_iris_generation_files(
    data_dir: &std::path::Path,
    backbone: bool,
    text_encoder: bool,
) {
    if backbone {
        for file in ["config.yaml", "model.safetensors"] {
            iris_snapshot_file(data_dir, IRIS_BACKBONE_REPO, IRIS_BACKBONE_REVISION, file);
        }
    }
    if text_encoder {
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            iris_snapshot_file(data_dir, IRIS_TE_REPO, IRIS_TE_REVISION, file);
        }
    }
}

/// The depth task export alone (`depth/` of the pinned repo) — not a generation install.
pub(super) fn seed_iris_depth_export_only(data_dir: &std::path::Path) {
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        iris_snapshot_file(
            data_dir,
            IRIS_BACKBONE_REPO,
            IRIS_BACKBONE_REVISION,
            &format!("depth/{file}"),
        );
    }
}

pub(super) async fn project(app: &axum::Router) -> String {
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Iris Project" }),
    )
    .await;
    project["id"].as_str().expect("project id").to_owned()
}

pub(super) async fn dataset(app: &axum::Router, project_id: &str, items: Value) -> String {
    let mut built = Vec::new();
    for (index, item) in items
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            &format!("Item{index}.png"),
            "image/png",
            b"png-bytes",
        )
        .await;
        let mut item = item;
        item["assetId"] = asset["id"].clone();
        built.push(item);
    }
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Iris set", "items": built }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    dataset["id"].as_str().expect("dataset id").to_owned()
}

pub(super) async fn iris_target(app: &axum::Router) -> Value {
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    registry["targets"]
        .as_array()
        .expect("targets")
        .iter()
        .find(|target| target["id"] == IRIS_TARGET)
        .expect("the Iris target is served")
        .clone()
}

pub(super) async fn submit(
    app: &axum::Router,
    project_id: &str,
    dataset_id: &str,
    config: Value,
    dry_run: bool,
) -> (StatusCode, Value) {
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": IRIS_TARGET,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Iris Style",
            "dryRun": dry_run
        }),
    )
    .await
}

fn one_item() -> Value {
    json!([{ "caption": { "text": "a red square" } }])
}

/// S7: generation training is judged on ITS OWN install — the pinned root backbone and the Qwen3-VL
/// text encoder — at plan-check and at submit, with typed refusals. A depth export alone (which the
/// repo-wide marker counted as installed) and an encoder without its backbone are both refused.
#[tokio::test]
async fn iris_generation_training_needs_its_own_backbone_and_text_encoder() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    // The adapter-family guardrail is satisfied, and the repo-wide marker says "installed".
    seed_installed_training_base(&settings.data_dir, "iris_3b");
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::write(
        manifests.join("user.models.jsonc"),
        serde_json::to_vec_pretty(&json!({ "models": [{
            "id": "iris_adapter_host", "name": "Iris adapter host", "type": "image",
            "family": "iris", "loraCompatibility": { "families": ["iris"], "types": ["lora"] }
        }] }))
        .unwrap(),
    )
    .unwrap();
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let defaults = iris_target(&app).await["defaults"].clone();
    let refusal_kind = |app: axum::Router, config: Value| async move {
        let (status, check) = request(
            app,
            "POST",
            "/api/v1/training/iris/plan-check",
            json!({ "targetId": IRIS_TARGET, "config": config }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{check}");
        check["refusal"]["kind"].clone()
    };

    // Only the depth export installed.
    seed_iris_depth_export_only(&settings.data_dir);
    assert_eq!(
        refusal_kind(app.clone(), defaults.clone()).await,
        "missing_backbone"
    );
    let (status, error) = submit(&app, &project_id, &dataset_id, defaults.clone(), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "iris_training_refused", "{error}");
    assert_eq!(error["context"]["kind"], "missing_backbone", "{error}");

    // The backbone deleted, the encoder kept.
    seed_iris_generation_files(&settings.data_dir, false, true);
    let (status, error) = submit(&app, &project_id, &dataset_id, defaults.clone(), false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "missing_backbone", "{error}");

    // The backbone without its encoder: a fresh data dir.
    let bare_dir = tempfile::tempdir().unwrap();
    let bare_settings = test_settings(&bare_dir);
    let bare_app = create_app(bare_settings.clone()).unwrap();
    seed_installed_training_base(&bare_settings.data_dir, "iris_3b");
    std::fs::create_dir_all(bare_settings.config_dir.join("manifests")).unwrap();
    std::fs::copy(
        manifests.join("user.models.jsonc"),
        bare_settings
            .config_dir
            .join("manifests")
            .join("user.models.jsonc"),
    )
    .unwrap();
    seed_iris_generation_files(&bare_settings.data_dir, true, false);
    assert_eq!(
        refusal_kind(bare_app.clone(), defaults.clone()).await,
        "missing_text_encoder"
    );
    let bare_project = project(&bare_app).await;
    let bare_dataset = dataset(&bare_app, &bare_project, one_item()).await;
    let (status, error) = submit(
        &bare_app,
        &bare_project,
        &bare_dataset,
        defaults.clone(),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["context"]["kind"], "missing_text_encoder", "{error}");

    // Both installed: the install check passes (whatever the plan check says next is not an
    // install refusal).
    seed_iris_generation_files(&settings.data_dir, true, true);
    let kind = refusal_kind(app.clone(), defaults.clone()).await;
    assert!(
        !matches!(
            kind.as_str(),
            Some("missing_backbone" | "missing_text_encoder")
        ),
        "{kind}"
    );
}

/// Every refusal the trainer would raise after load is a 400 at submit — typed for the advertised
/// limits (the micro-step rule names the field and the multiple).
#[tokio::test]
async fn iris_submit_refuses_what_the_trainer_refuses() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let defaults = target["defaults"].clone();

    // Defaults are runnable as served.
    let (status, job) = submit(&app, &project_id, &dataset_id, defaults.clone(), true).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["plan"]["target"]["kernel"], "iris_train");
    assert_eq!(
        job["payload"]["plan"]["config"]["advanced"]["iris"]["text_conditioning"],
        "cached"
    );

    // Micro-steps: steps 1000 at accumulation 3 is not a whole number of windows.
    let mut config = defaults.clone();
    config["gradientAccumulation"] = json!(3);
    config["steps"] = json!(999);
    config["saveEvery"] = json!(250);
    let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "training_target_limit", "{error}");
    assert_eq!(error["context"]["kind"], "not_multiple_of", "{error}");
    assert_eq!(error["context"]["field"], "saveEvery", "{error}");
    assert_eq!(error["context"]["multiple"], 3);

    for (mutate, needle) in [
        (
            json!({"advanced": {"lrWarmupSteps": 1}, "gradientAccumulation": 2, "steps": 1000, "saveEvery": 250}),
            "lrWarmupSteps",
        ),
        (json!({"optimizer": "adamw8bit"}), "optimizer"),
        (
            json!({"advanced": {"lrScheduler": "linear"}}),
            "lrScheduler",
        ),
        (
            json!({"advanced": {"timestepBias": "high_noise"}}),
            "timestepBias",
        ),
        (json!({"advanced": {"lossType": "mae"}}), "lossType"),
        (
            json!({"advanced": {"timestepType": "sigmoid_shifted"}}),
            "timestepType",
        ),
        (
            json!({"advanced": {"mixedPrecision": "fp8"}}),
            "mixedPrecision",
        ),
        (
            json!({"advanced": {"gradientCheckpointing": true}}),
            "gradient checkpointing",
        ),
        (json!({"advanced": {"sampleCount": 5}}), "sampleCount"),
        (
            json!({"advanced": {"iris": {"not_an_option": 1}}}),
            "not_an_option",
        ),
        (
            json!({"advanced": {"iris": {"resume_from": "/tmp/x"}}}),
            "resumeFromJobId",
        ),
        (
            json!({"advanced": {"iris": {"init": "random"}}}),
            "Random initialization",
        ),
        (
            json!({"advanced": {"iris": {"ema_enabled": false, "export_weights": "ema"}}}),
            "ema",
        ),
        (
            json!({"advanced": {"iris": {"text_dropout": 2.0}}}),
            "text_dropout",
        ),
        (
            json!({"advanced": {"iris": {"resume_data_policy": "new_phase"}}}),
            "resuming",
        ),
        (
            json!({"advanced": {"loadFromJobId": "job_x"}}),
            "full fine-tune",
        ),
        (
            json!({"advanced": {"iris": {"upstream_validation_prompts": true}, "samplePrompts": ["mine"]}}),
            "upstream",
        ),
    ] {
        let mut config = defaults.clone();
        for (key, value) in mutate.as_object().unwrap() {
            if key == "advanced" {
                for (inner, value) in value.as_object().unwrap() {
                    if inner == "iris" {
                        for (option, value) in value.as_object().unwrap() {
                            config["advanced"]["iris"][option] = value.clone();
                        }
                    } else {
                        config["advanced"][inner] = value.clone();
                    }
                }
            } else {
                config[key] = value.clone();
            }
        }
        let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mutate} → {error}");
        assert!(
            error.to_string().contains(needle),
            "{mutate}: the refusal names {needle:?}: {error}"
        );
    }

    // A caption field no item carries is refused before launch.
    let mut config = defaults.clone();
    config["advanced"]["iris"]["caption_field"] = json!("short");
    let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error.to_string().contains("short"), "{error}");
}

/// The trainer's unused tail is reported before launch, from the trainer's own walk.
#[tokio::test]
async fn data_walk_reports_the_unused_tail_before_launch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp_dir)).unwrap();
    let (status, walk) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/data-walk?targetId={IRIS_TARGET}&items=1000&batchSize=4"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(walk["applies"], true);
    assert_eq!(walk["covered"], 640);
    assert_eq!(walk["unusedItems"], 360);
    assert!(walk["warning"]
        .as_str()
        .is_some_and(|text| text.contains("360 of 1000")));

    let (_, whole) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/data-walk?targetId={IRIS_TARGET}&items=640"),
        Value::Null,
    )
    .await;
    assert_eq!(whole["unusedItems"], 0);
    assert!(whole["warning"].is_null());

    let (_, other) = request(
        app,
        "GET",
        "/api/v1/training/data-walk?targetId=z_image_turbo_lora&items=1000",
        Value::Null,
    )
    .await;
    assert_eq!(other["applies"], false);
}

pub(super) async fn complete(
    app: &axum::Router,
    job_id: &str,
    status: &str,
    result: Value,
) -> Value {
    let (code, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": status,
            "stage": status,
            "progress": 1,
            "message": "done",
            "workerId": TEST_TRAINING_WORKER_ID,
            "result": result
        }),
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    body
}

/// S9a: one output id is one run. Every fresh submit mints its own output folder; a DUPLICATE
/// (which copies the payload) is a new run under its own fresh output id and folder — also of a
/// queued job, and twice over; a RETRY keeps the run's output id, so it is refused while another job
/// still writes that output and once the run has a complete checkpoint (resume it) — but a run that
/// died before its first COMPLETE checkpoint retries as before.
#[tokio::test]
async fn an_iris_rerun_never_shares_an_output_with_another_run() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let config = iris_target(&app).await["defaults"].clone();
    let output_dir = |job: &Value| {
        std::path::PathBuf::from(
            job["payload"]["plan"]["output"]["outputDir"]
                .as_str()
                .unwrap(),
        )
    };
    let output_id = |job: &Value| {
        job["payload"]["manifestEntry"]["id"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let stem = |job: &Value| {
        job["payload"]["plan"]["output"]["fileName"]
            .as_str()
            .unwrap()
            .trim_end_matches(".safetensors")
            .to_owned()
    };
    let rerun = |job_id: String, action: &'static str| {
        let app = app.clone();
        async move {
            request(
                app,
                "POST",
                &format!("/api/v1/jobs/{job_id}/{action}"),
                json!({ "payloadChanges": {} }),
            )
            .await
        }
    };
    let refused_in_use = |status: StatusCode, error: &Value| {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert_eq!(error["code"], "iris_training_refused", "{error}");
        assert_eq!(error["context"]["kind"], "output_in_use", "{error}");
    };
    let fresh_run = |source: &Value, copy: &Value| {
        assert_ne!(output_id(copy), output_id(source), "{copy:#}");
        assert_ne!(output_dir(copy), output_dir(source));
        assert!(output_dir(copy).ends_with(output_id(copy)), "{copy:#}");
        assert_eq!(
            copy["payload"]["plan"]["output"]["loraId"],
            json!(output_id(copy))
        );
        assert!(
            !copy["payload"].to_string().contains(&output_id(source)),
            "no trace of the source's output id: {copy:#}"
        );
    };

    let (_, first) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;

    // ---- a run with a COMPLETE checkpoint: retry refused (resume it), duplicate twice is fine --------
    let job_id = first["id"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    let checkpoints = output_dir(&first).join(format!("{}.checkpoints", stem(&first)));
    std::fs::create_dir_all(checkpoints.join("step_00000002")).unwrap();
    complete(&app, &job_id, "failed", json!({})).await;
    // sc-25785's generic retry-resume never applies: an Iris run resumes from the Training Studio
    // (`resumeFromJobId`), so `resume: true` is refused, typed, before anything is queued.
    let (status, error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/retry"),
        json!({ "resume": true }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error.to_string().contains("Training Studio"), "{error}");
    // An INCOMPLETE checkpoint (no state.json) is not a resumable run: the retry is allowed...
    let (status, retried) = rerun(job_id.clone(), "retry").await;
    assert_eq!(status, StatusCode::CREATED, "{retried}");
    assert_eq!(
        output_id(&retried),
        output_id(&first),
        "a retry is the same run"
    );
    // ...and a second retry while the first is queued is refused, as is retrying again later.
    let (status, error) = rerun(job_id.clone(), "retry").await;
    refused_in_use(status, &error);
    let retried_id = retried["id"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &retried_id).await;
    std::fs::write(checkpoints.join("step_00000002/state.json"), b"{}").unwrap();
    complete(&app, &retried_id, "failed", json!({})).await;
    let (status, error) = rerun(retried_id.clone(), "retry").await;
    refused_in_use(status, &error);
    assert!(error.to_string().contains("Resume"), "{error}");
    let (status, one) = rerun(retried_id.clone(), "duplicate").await;
    assert_eq!(status, StatusCode::CREATED, "{one}");
    let (status, two) = rerun(retried_id.clone(), "duplicate").await;
    assert_eq!(status, StatusCode::CREATED, "{two}");
    fresh_run(&first, &one);
    fresh_run(&first, &two);
    assert_ne!(
        output_id(&one),
        output_id(&two),
        "each duplicate is its own run"
    );

    // ---- a queued run (nothing claims it): retry refused (it is the writer), a duplicate is a
    // separate run with its own output --------------------------------------------------------------
    let (_, second) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    assert_ne!(output_dir(&first), output_dir(&second));
    let queued_id = second["id"].as_str().unwrap().to_owned();
    let (status, error) = rerun(queued_id.clone(), "retry").await;
    refused_in_use(status, &error);
    let (status, copy) = rerun(queued_id.clone(), "duplicate").await;
    assert_eq!(status, StatusCode::CREATED, "{copy}");
    fresh_run(&second, &copy);
}

/// A LoRA run's adapter registers as an Iris generation adapter with its task identity; per-item
/// named captions reach the plan; a later run resumes the first run's newest checkpoint.
#[tokio::test]
async fn iris_adapter_registers_with_task_identity_and_resumes() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(
        &app,
        &project_id,
        json!([{ "caption": { "text": "a red square" }, "captions": { "short": "red" } }]),
    )
    .await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");
    config["advanced"]["iris"]["caption_fields"] = json!(["short"]);
    let (status, job) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(
        plan["dataset"]["items"][0]["captions"],
        json!({ "short": "red" }),
        "named captions ride on the plan item"
    );
    assert_eq!(job["payload"]["manifestEntry"]["irisTask"], "generation");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    write_test_safetensors(&output_dir.join(&file_name));
    let stem = file_name.trim_end_matches(".safetensors");
    let checkpoint = output_dir.join(format!("{stem}.checkpoints/step_00000250"));
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::write(checkpoint.join("state.json"), b"{}").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(completed["result"]["loraRegistered"], true, "{completed}");

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
        .expect("the adapter is registered")
        .clone();
    assert_eq!(entry["family"], "iris");
    assert_eq!(entry["baseModel"], "iris_3b");
    assert_eq!(entry["irisTask"], "generation");
    assert_eq!(entry["networkType"], "lora");
    assert_eq!(entry["installState"], "installed");

    // Resume: the second run names the first; SceneWorks hands the trainer its checkpoint root.
    let mut resume = config.clone();
    resume["advanced"]["resumeFromJobId"] = json!(job_id);
    resume["advanced"]["iris"]["resume_data_policy"] = json!("new_phase");
    let (status, resumed) = submit(&app, &project_id, &dataset_id, resume, false).await;
    assert_eq!(status, StatusCode::CREATED, "{resumed}");
    let advanced = &resumed["payload"]["plan"]["config"]["advanced"];
    assert_eq!(advanced["resume"], true);
    assert_eq!(
        advanced["iris"]["resume_from"],
        json!(output_dir
            .join(format!("{stem}.checkpoints"))
            .display()
            .to_string())
    );
    assert_eq!(advanced["iris"]["resume_data_policy"], "new_phase");

    // A run with no checkpoint cannot be resumed; a different artifact kind cannot either.
    std::fs::remove_dir_all(output_dir.join(format!("{stem}.checkpoints"))).unwrap();
    let mut resume = config.clone();
    resume["advanced"]["resumeFromJobId"] = json!(job_id);
    let (status, error) = submit(&app, &project_id, &dataset_id, resume, false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error.to_string().contains("no complete checkpoint"),
        "{error}"
    );
    // Deleting the adapter deletes the run's checkpoints AND the one `.prior` sibling the trainer
    // retires a foreign run's checkpoints into (sc-25690), named by the engine contract.
    let [root, prior] = sceneworks_worker::iris_run_checkpoint_dirs(&output_dir, &file_name);
    for dir in [&root, &prior] {
        std::fs::create_dir_all(dir.join("step_00000004")).unwrap();
        std::fs::write(dir.join("step_00000004/state.json"), b"{}").unwrap();
    }
    assert_eq!(prior, output_dir.join(format!("{stem}.checkpoints.prior")));
    let lora_id = entry["id"].as_str().unwrap();
    let scope = entry["scope"].as_str().unwrap();
    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/loras/{lora_id}?scope={scope}&projectId={project_id}&permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    for gone in [&output_dir.join(&file_name), &root, &prior] {
        assert!(!gone.exists(), "orphaned after delete: {}", gone.display());
    }
}

/// A full run's export registers as a selectable user image model with its Iris task identity, and
/// generating it runs the `iris_3b` engine on that export with the generation install's encoder.
#[tokio::test]
async fn iris_full_export_registers_and_generates_through_the_iris_engine() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");
    config["advanced"]["networkType"] = json!("full");
    config["advanced"]["iris"]["init"] = json!("random");
    let (status, job) = submit(&app, &project_id, &dataset_id, config, false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(plan["target"]["outputKind"], "base_checkpoint");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    assert!(output_dir.starts_with(settings.data_dir.join("models").join("finetunes")));
    let stem = plan["output"]["fileName"]
        .as_str()
        .unwrap()
        .trim_end_matches(".safetensors")
        .to_owned();
    let export = output_dir.join(&stem);

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&export).unwrap();
    std::fs::write(export.join("config.yaml"), b"model: {}\n").unwrap();
    write_test_safetensors(&export.join("model.safetensors"));
    // The trainer's resumable checkpoints sit beside the export.
    let checkpoints = output_dir.join(format!("{stem}.checkpoints"));
    std::fs::create_dir_all(checkpoints.join("step_00000001")).unwrap();
    std::fs::write(checkpoints.join("step_00000001/state.json"), b"{}").unwrap();
    // …and so does the one `.prior` sibling the trainer retires a foreign run's checkpoints into
    // (sc-25690), named by the engine contract.
    let [_, prior] = sceneworks_worker::iris_run_checkpoint_dirs(
        &output_dir,
        plan["output"]["fileName"].as_str().unwrap(),
    );
    assert_eq!(prior, output_dir.join(format!("{stem}.checkpoints.prior")));
    std::fs::create_dir_all(prior.join("step_00000009")).unwrap();
    std::fs::write(prior.join("step_00000009/state.json"), b"{}").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(
        completed["result"]["baseCheckpointRegistered"], true,
        "{completed}"
    );
    assert!(completed["result"].get("loraRegistered").is_none());
    let model_id = completed["result"]["baseCheckpointModelId"]
        .as_str()
        .unwrap()
        .to_owned();

    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    let entry = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is a catalog model")
        .clone();
    assert_eq!(entry["type"], "image");
    assert_eq!(entry["family"], "iris");
    assert_eq!(entry["irisTask"], "generation");
    assert_eq!(entry["trainedCheckpoint"]["engineModel"], "iris_3b");
    assert_eq!(entry["trainedCheckpoint"]["task"], "generation");
    assert_eq!(entry["installState"], "installed");
    assert_eq!(entry["adapter"], "mlx_iris");
    assert!(entry["capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("text_to_image")));
    assert_eq!(entry["paths"]["model"], json!(export.display().to_string()));
    #[cfg(target_os = "macos")]
    assert_eq!(
        entry["macSupport"]["supported"], true,
        "the export renders through iris_3b's MLX lane: {}",
        entry["macSupport"]
    );

    // The stored entry keeps the identity only; the generation surface is the engine's, read at
    // catalog time — so a later change to the `iris_3b` entry reaches an already-registered export.
    let manifests = settings.config_dir.join("manifests");
    let stored: Value = serde_json::from_str(&crate::strip_jsonc_comments(
        &std::fs::read_to_string(manifests.join("user.models.jsonc")).unwrap(),
    ))
    .unwrap();
    let stored = stored["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is stored")
        .clone();
    assert_eq!(stored["trainedCheckpoint"]["engineModel"], "iris_3b");
    for key in [
        "loraCompatibility",
        "limits",
        "defaults",
        "capabilities",
        "adapter",
    ] {
        assert!(
            stored.get(key).is_none(),
            "{key} is not snapshotted: {stored:#}"
        );
    }
    let mut builtin: Value = serde_json::from_str(&crate::strip_jsonc_comments(include_str!(
        "../../../../config/manifests/builtin.models.jsonc"
    )))
    .unwrap();
    let engine = builtin["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|model| model["id"] == "iris_3b")
        .expect("the builtin iris_3b entry");
    engine["loraCompatibility"] =
        json!({ "families": ["iris", "later_family"], "types": ["lora"] });
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        serde_json::to_vec_pretty(&builtin).unwrap(),
    )
    .unwrap();
    let later_app = create_app(settings.clone()).unwrap();
    let (_, models) = request(later_app, "GET", "/api/v1/models", Value::Null).await;
    let later = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is still a catalog model")
        .clone();
    assert_eq!(
        later["loraCompatibility"]["families"],
        json!(["iris", "later_family"]),
        "{later:#}"
    );

    // Generate it: the job runs `iris_3b` on the export, with the generation install's entry (its
    // text-encoder co-requisite) and the user's pick kept for provenance.
    let (status, generation) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": model_id,
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{generation}");
    let payload = &generation["payload"];
    assert_eq!(payload["model"], "iris_3b");
    assert_eq!(payload["checkpointModelId"], json!(model_id));
    let worker_entry = &payload["modelManifestEntry"];
    assert_eq!(worker_entry["id"], "iris_3b");
    assert_eq!(
        worker_entry["modelPath"],
        json!(export.canonicalize().unwrap().display().to_string())
    );
    assert_eq!(
        worker_entry["trainedCheckpoint"]["modelId"],
        json!(model_id)
    );
    assert!(
        worker_entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .any(|download| download["componentId"] == "text_encoder"
                && download["coRequisite"] == true),
        "the generation install's text_encoder co-requisite still applies: {worker_entry}"
    );

    // A duplicate that picks ANOTHER model is not re-aliased onto the recorded checkpoint.
    let generation_id = generation["id"].as_str().unwrap().to_owned();
    let (status, duplicate) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{generation_id}/duplicate"),
        json!({ "payloadChanges": { "model": "sdxl" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{duplicate}");
    let payload = &duplicate["payload"];
    assert_eq!(payload["model"], "sdxl", "{payload:#}");
    assert!(payload["checkpointModelId"].is_null(), "{payload:#}");
    assert_eq!(payload["modelManifestEntry"]["id"], "sdxl", "{payload:#}");
    // A plain duplicate of the aliased job still re-derives the checkpoint alias.
    let (status, replay) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{generation_id}/duplicate"),
        json!({ "payloadChanges": {} }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replay}");
    assert_eq!(replay["payload"]["checkpointModelId"], json!(model_id));
    assert_eq!(
        replay["payload"]["modelManifestEntry"]["trainedCheckpoint"]["modelId"],
        json!(model_id)
    );

    // A client-authored `checkpointModelId` naming something that is not a trained checkpoint of
    // `model` is dropped: `model`'s own manifest is what resolves.
    let (status, spoofed) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": "iris_3b",
            "checkpointModelId": "iris_adapter_host",
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{spoofed}");
    let payload = &spoofed["payload"];
    assert_eq!(payload["model"], "iris_3b");
    assert!(payload.get("checkpointModelId").is_none(), "{payload:#}");
    assert_eq!(
        payload["modelManifestEntry"]["id"], "iris_3b",
        "{payload:#}"
    );
    assert!(
        payload["modelManifestEntry"]
            .get("trainedCheckpoint")
            .is_none(),
        "{payload:#}"
    );

    // Model Manager delete removes the whole run directory — the export AND its checkpoints.
    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/{model_id}?permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    assert!(
        !checkpoints.exists(),
        "the run's checkpoints are not orphaned: {}",
        checkpoints.display()
    );
    assert!(
        !prior.exists(),
        "the run's retired checkpoints are not orphaned: {}",
        prior.display()
    );
    assert!(!output_dir.exists(), "{}", output_dir.display());
}

/// Epic AT6 (sc-25690) — a FAILED export of a generation run, for the full fine-tune, the LoRA and
/// the LoKr. The worst case is staged on purpose: the export IS on disk at the run's output path,
/// and the job is then reported `failed`. Nothing is registered — no result flag, no catalog model,
/// no LoRA — and the run's sources (dataset files, installed backbone + text encoder) are
/// byte-identical. The same export reported `completed` registers, so it is the failure that
/// withholds registration, not the files.
#[tokio::test]
async fn a_failed_generation_export_registers_nothing_and_leaves_its_sources_untouched() {
    use super::iris_task_training::{model_ids, source_tree};
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let lora_ids = |app: axum::Router, project_id: String| async move {
        let (status, loras) = request(
            app,
            "GET",
            &format!("/api/v1/loras?projectId={project_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{loras}");
        loras
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|lora| lora["id"].as_str().map(str::to_owned))
            .collect::<Vec<_>>()
    };

    for network in ["full", "lora", "lokr"] {
        let mut config = target["defaults"].clone();
        config["triggerWord"] = json!("irisstyle");
        config["advanced"]["networkType"] = json!(network);
        if network == "full" {
            config["advanced"]["iris"]["init"] = json!("random");
        }
        let mut outcomes = Vec::new();
        for status in ["failed", "completed"] {
            let models_before = model_ids(&app).await;
            let loras_before = lora_ids(app.clone(), project_id.clone()).await;
            let (code, job) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
            assert_eq!(code, StatusCode::CREATED, "{network}: {job}");
            let plan = &job["payload"]["plan"];
            let job_id = job["id"].as_str().unwrap().to_owned();
            let output_dir =
                std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
            let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();
            claim_training_job_as_mlx_worker(&app, &job_id).await;
            std::fs::create_dir_all(&output_dir).unwrap();
            if network == "full" {
                let export = output_dir.join(file_name.trim_end_matches(".safetensors"));
                std::fs::create_dir_all(&export).unwrap();
                std::fs::write(export.join("config.yaml"), b"model: {}\n").unwrap();
                write_test_safetensors(&export.join("model.safetensors"));
            } else {
                write_test_safetensors(&output_dir.join(&file_name));
            }
            let sources_before = source_tree(temp_dir.path());
            assert!(
                sources_before
                    .keys()
                    .any(|path| path.components().any(|part| part.as_os_str() == "datasets"))
                    && sources_before
                        .keys()
                        .any(|path| path.ends_with("config.yaml")),
                "{network}: the dataset and the install are among the sources"
            );
            let reported = complete(&app, &job_id, status, json!({})).await;
            assert_eq!(reported["status"], status, "{network}: {reported}");
            assert_eq!(
                source_tree(temp_dir.path()),
                sources_before,
                "{network} ({status}): the run's sources are untouched"
            );
            let new_models = model_ids(&app)
                .await
                .into_iter()
                .filter(|id| !models_before.contains(id))
                .count();
            let new_loras = lora_ids(app.clone(), project_id.clone())
                .await
                .into_iter()
                .filter(|id| !loras_before.contains(id))
                .count();
            outcomes.push((new_models, new_loras, reported));
        }
        let (new_models, new_loras, reported) = &outcomes[0];
        assert_eq!(
            (*new_models, *new_loras),
            (0, 0),
            "{network}: a failed export registers nothing: {reported}"
        );
        assert!(
            reported["result"].get("baseCheckpointRegistered").is_none()
                && reported["result"].get("loraRegistered").is_none(),
            "{network}: {reported}"
        );
        let (new_models, new_loras, reported) = &outcomes[1];
        assert_eq!(
            (*new_models, *new_loras),
            if network == "full" { (1, 0) } else { (0, 1) },
            "{network}: the same export, completed, registers: {reported}"
        );
    }
}

/// A cancel after the first optimizer step published an export: it registers; a cancel that
/// published nothing registers nothing.
#[tokio::test]
async fn a_canceled_run_registers_only_what_the_trainer_exported() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");

    // Nothing exported.
    let (_, job) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    let job_id = job["id"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    let canceled = complete(&app, &job_id, "canceled", Value::Null).await;
    assert!(
        canceled["result"].get("loraRegistered").is_none(),
        "{canceled}"
    );

    // Exported on cancel.
    let (_, job) = submit(&app, &project_id, &dataset_id, config, false).await;
    let job_id = job["id"].as_str().unwrap().to_owned();
    let plan = &job["payload"]["plan"];
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    write_test_safetensors(&output_dir.join(&file_name));
    let canceled = complete(
        &app,
        &job_id,
        "canceled",
        json!({ "trainedArtifactExported": true, "stepsCompleted": 2 }),
    )
    .await;
    assert_eq!(canceled["status"], "canceled");
    assert_eq!(canceled["result"]["loraRegistered"], true, "{canceled}");
}

/// HF-style Parquet imports can bring named caption columns; malformed names are refused.
#[tokio::test]
async fn parquet_import_forwards_caption_field_columns() {
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Empty", "items": [] }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().unwrap().to_owned();
    let source = temp_dir.path().join("laion.parquet");
    std::fs::write(&source, b"PAR1").unwrap();
    let path =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-jobs");
    let (status, job) = request(
        app.clone(),
        "POST",
        &path,
        json!({
            "sourcePath": source.display().to_string(),
            "captionFieldColumns": ["short_caption", "tags"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["captionFieldColumns"],
        json!(["short_caption", "tags"])
    );
    let (status, error) = request(
        app,
        "POST",
        &path,
        json!({
            "sourcePath": source.display().to_string(),
            "captionFieldColumns": [" padded"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
}

/// The S7 adapter (Training Studio) meets the S3 adapter gate (sc-25681): a registered Iris LoRA
/// whose file carries the engine's stamp (`family=iris`, `irisTask=generation`,
/// `baseModel=iris_3b`) is accepted on an `iris_3b` image job — the shipped entry's
/// `loraCompatibility` (families `iris`, `irisTask` generation) admits it end to end.
#[tokio::test]
async fn a_trained_iris_adapter_is_accepted_by_the_iris_image_job_gate() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    // The catalog is the SHIPPED `iris_3b` row alone — no stand-in adapter host: its
    // `loraCompatibility` must satisfy both the training family guardrail and the image-job gate.
    seed_installed_training_base(&settings.data_dir, "iris_3b");
    seed_iris_generation_files(&settings.data_dir, true, true);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir");
    let shipped: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1,
    ))
    .expect("builtin.models.jsonc parses");
    let iris = shipped["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|model| model["id"] == "iris_3b")
        .expect("iris_3b is shipped")
        .clone();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [iris] }).to_string(),
    )
    .expect("builtin models writes");
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let (status, job) = submit(
        &app,
        &project_id,
        &dataset_id,
        target["defaults"].clone(),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let plan = &job["payload"]["plan"];
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    // What the Iris trainer writes: an adapter whose safetensors metadata names its task identity.
    let header = json!({
        "__metadata__": { "family": "iris", "irisTask": "generation", "baseModel": "iris_3b" },
        "blocks.0.attn.qkv.lora_down.weight": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }
    })
    .to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0; 4]);
    std::fs::write(output_dir.join(&file_name), bytes).unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(completed["result"]["loraRegistered"], true, "{completed}");

    let (_, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    let lora_id = loras
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Iris Style")
        .expect("the adapter is registered")["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let (status, generation) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": "iris_3b",
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1,
            "loras": [{ "id": lora_id, "weight": 0.8 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{generation}");
    assert!(
        generation["payload"]["loras"]
            .as_array()
            .is_some_and(|loras| loras.iter().any(|lora| lora["id"] == json!(lora_id))),
        "{generation:#}"
    );
}

/// sc-25690: the delete seams of a trained Iris adapter and a full / task export NAME the run's
/// checkpoint root and the `.prior` retirement sibling, from the engine contract — whatever else
/// the delete removes. Mutation: drop either `iris_run_checkpoint_paths` call ⇒ red; key the
/// adapter on anything but the file the run named, or the export on anything but its directory's
/// stem ⇒ red. A builtin, non-training or non-Iris entry names none.
#[test]
fn iris_delete_paths_name_the_checkpoint_root_and_its_prior() {
    let data_dir = std::path::Path::new("/data");
    let lora_dir = data_dir.join("loras/lora_abc");
    let adapter = json!({
        "id": "lora_abc",
        "irisTask": "generation",
        "source": { "provider": "training", "path": "loras/lora_abc" },
        "files": ["iris.style.safetensors"],
    });
    let [root, prior] =
        sceneworks_worker::iris_run_checkpoint_dirs(&lora_dir, "iris.style.safetensors");
    assert_eq!(prior, lora_dir.join("iris.style.checkpoints.prior"));
    let paths = crate::loras::lora_artifact_paths(&adapter, data_dir);
    assert_eq!(paths, vec![lora_dir.clone(), root, prior]);

    let run_dir = data_dir.join("models/finetunes/model_abc");
    let export = json!({
        "id": "model_abc",
        "irisTask": "depth",
        "trainedCheckpoint": { "task": "depth" },
        "source": { "provider": "training", "path": "models/finetunes/model_abc" },
        "paths": { "model": run_dir.join("depth.v2").display().to_string() },
        "files": ["config.yaml", "model.safetensors", "empty_prompt.safetensors"],
    });
    let [root, prior] =
        sceneworks_worker::iris_run_checkpoint_dirs(&run_dir, "depth.v2.safetensors");
    assert_eq!(root, run_dir.join("depth.v2.checkpoints"));
    let paths = crate::models::model_artifact_paths(&export, data_dir);
    assert!(paths.contains(&root), "{paths:?}");
    assert!(paths.contains(&prior), "{paths:?}");

    for unrelated in [
        json!({ "id": "x", "source": { "provider": "training", "path": "loras/x" }, "files": ["x.safetensors"] }),
        json!({ "id": "y", "irisTask": "generation", "catalogScope": "builtin", "source": { "provider": "training", "path": "loras/y" }, "files": ["y.safetensors"] }),
        json!({ "id": "z", "irisTask": "generation", "source": { "provider": "local", "path": "loras/z" }, "files": ["z.safetensors"] }),
    ] {
        let paths = crate::loras::lora_artifact_paths(&unrelated, data_dir);
        assert!(
            paths
                .iter()
                .all(|path| !path.to_string_lossy().contains(".checkpoints")),
            "{unrelated}: {paths:?}"
        );
    }
}
