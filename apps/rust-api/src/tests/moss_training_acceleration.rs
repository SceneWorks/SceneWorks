//! One manual API -> worker -> four native updates -> final Library adapter -> PNG.
//! Separate from the original 60-update campaign and all its recovery/evaluation work.
use super::{moss_training_acceptance::*, support::*};
use futures_util::FutureExt;
use sceneworks_worker::test_support::{self as native, gen_core};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};
type Result<T> = std::result::Result<T, String>;
const SELECTOR: &str = "tests::moss_training_acceleration::moss_bounded_product_acceptance";
const UUID: &str = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe";
const CAP: u64 = 80 * 1024 * 1024 * 1024;
const PROMPT: &str = "sksmoss red circle centered on a pale background";

fn config() -> Value {
    json!({"steps":4,"batchSize":1,"gradientAccumulation":1,"saveEvery":0,"resolution":256,
        "rank":2,"alpha":2.0,"learningRate":0.001,"seed":7,"optimizer":"adamw","triggerWord":"sksmoss",
        "advanced":{"mixedPrecision":"bf16","gradientCheckpointing":false,"cacheLatents":true,
        "cacheTextEmbeddings":true,"decomposeFactor":-1,"loraTargetModules":[],"networkType":"lora",
        "timestepType":"moss_stage3_v1","timestepBias":"balanced","lossType":"mse",
        "lrScheduler":"constant","weightDecay":0.0,"lrWarmupSteps":0,"sampleEvery":0,
        "resume":false,"modelOptions":{"vae_posterior":"posterior_mean_v1"}}})
}

fn admitted() -> Result<Case> {
    require(
        env("MOSS_REAL_WEIGHTS_ACCEPTANCE")? == "1"
            && env("GITHUB_EVENT_NAME")? == "workflow_dispatch",
        "manual workflow admission required",
    )?;
    require(
        env("MOSS_ACCEPTANCE_PHASE")? == "bounded-product"
            && env("MODEL_INFERENCE_CASE_FQN")? == SELECTOR,
        "bounded source-owned selector required",
    )?;
    require(
        env("MOSS_ACCEPTANCE_CASE_ID")? == "cuda/base/lora",
        "bounded Base/LoRA identity required",
    )?;
    require(
        env("CUDA_DEVICE_ORDER")? == "PCI_BUS_ID"
            && env("CUDA_VISIBLE_DEVICES")? == "1"
            && env("NVIDIA_VISIBLE_DEVICES")? == "1"
            && env("MOSS_ACCEPTANCE_CUDA_UUID")? == UUID,
        "bounded physical GPU1 visibility/UUID changed",
    )?;
    require(
        env("HF_HUB_OFFLINE")? == "1" && env("TRANSFORMERS_OFFLINE")? == "1",
        "offline weights required",
    )?;
    require(
        env("MOSS_ACCEPTANCE_INFERENCE_PIN")? == source_pin()?,
        "binary permanent inference pin mismatch",
    )?;
    require(
        env("MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES")? == CAP.to_string(),
        "source-owned supervisor cap changed",
    )?;
    let claim = json_file(Path::new(&env("MOSS_ACCEPTANCE_DEVICE_CLAIM")?))?;
    require(
        claim["state"] == "admitted"
            && claim["selector"] == SELECTOR
            && claim["deviceUuid"] == UUID
            && claim["physicalIndex"] == 1
            && claim["activityProfile"] == "non-isolated-correctness-headroom"
            && claim["childGpuCapBytes"] == CAP
            && claim["gpuFreeReserveBytes"] == 2_u64 * 1024 * 1024 * 1024
            && claim["leaseReleased"] == false
            && claim["workflowSource"] == env("MOSS_ACCEPTANCE_WORKFLOW_SOURCE")?
            && claim["inferencePin"] == source_pin()?
            && claim["runId"] == env("GITHUB_RUN_ID")?
            && claim["runAttempt"] == env("GITHUB_RUN_ATTEMPT")?,
        "device claim does not bind the actual source/run/lease",
    )?;
    let root = PathBuf::from(env("MOSS_ACCEPTANCE_OUTPUT")?);
    require(
        root.is_absolute()
            && root.file_name().is_some_and(|s| s == "lora")
            && root
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|s| s == "base"),
        "owned bounded output layout required",
    )?;
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    require(
        !std::fs::symlink_metadata(&root)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink(),
        "output symlink refused",
    )?;
    let started_at = env("MOSS_ACCEPTANCE_CASE_STARTED_AT")?;
    let started =
        sceneworks_core::time::parse_utc_seconds(&started_at).ok_or("invalid case clock")?;
    let start_ms = u64::try_from(started)
        .map_err(|e| e.to_string())?
        .checked_mul(1000)
        .ok_or("clock overflow")?;
    let now = unix_ms()?;
    require(
        start_ms <= now && now - start_ms < 3600000,
        "bounded case deadline expired",
    )?;
    let mut manifest: Value = serde_json::from_str(include_str!(
        "../../../../scripts/ci/moss-training-acceptance/campaign.json"
    ))
    .map_err(|e| e.to_string())?;
    manifest["dataset"]["items"]
        .as_array_mut()
        .ok_or("dataset source absent")?
        .truncate(1);
    let case = Case {
        source_backend: "cuda".into(),
        backend: "cuda",
        parent: "base".into(),
        method: "lora".into(),
        id: "cuda/base/lora".into(),
        root,
        dataset: PathBuf::from(env("MOSS_ACCEPTANCE_DATASET_ROOT")?),
        manifest,
        deadline: Instant::now() + Duration::from_millis(3600000 - (now - start_ms)),
        started_at: started_at.clone(),
        preflight_only: false,
        cross: false,
        record: json!({"schemaVersion":1,"recordKind":"bounded-product-acceptance-evidence","isCampaignEvidence":false,
            "acceptanceDisposition":"missing","selector":SELECTOR,"caseId":"cuda/base/lora","phase":"bounded-product",
            "executionBackend":"cuda","failureReason":null,"source":{"workflowSource":env("MOSS_ACCEPTANCE_WORKFLOW_SOURCE")?,
            "inferencePin":source_pin()?,"runId":env("GITHUB_RUN_ID")?,"runAttempt":env("GITHUB_RUN_ATTEMPT")?},
            "raw":{"api":[],"images":{},"subruns":{},"startedAt":started_at,"wallLimitSeconds":3600,
                "deviceClaim":claim,"submittedConfig":config(),"gpuPolicy":"non-isolated-correctness-headroom"}}),
        resources: Arc::new(Mutex::new(RootProcessResources::with_ceiling(CAP))),
    };
    case.cancel_file()?;
    case.persist()?;
    Ok(case)
}

fn validate_request(request: &gen_core::TrainingRequest, cap: u64) -> gen_core::Result<()> {
    let c = &request.config;
    let valid = c.steps == 4
        && c.seed == 7
        && c.rank == 2
        && c.alpha == 2.0
        && c.learning_rate == 0.001
        && c.batch_size == 1
        && c.gradient_accumulation == 1
        && c.save_every == 0
        && c.resolution == 256
        && c.train_dtype == "bf16"
        && !c.gradient_checkpointing
        && c.cache_latents
        && c.cache_text_embeddings
        && c.optimizer == "adamw"
        && c.weight_decay == 0.0
        && c.lr_scheduler == gen_core::LrSchedule::Constant
        && c.lr_warmup_steps == 0
        && c.network_type == gen_core::NetworkType::Lora
        && c.decompose_factor == -1
        && c.lora_target_modules.is_empty()
        && !c.full_finetune
        && c.control_type.is_none()
        && c.subject_mask_loss.is_none()
        && c.timestep_type == "moss_stage3_v1"
        && c.timestep_bias == "balanced"
        && c.loss_type == "mse"
        && c.trigger_word.as_deref() == Some("sksmoss")
        && c.sample_every == 0
        && !c.resume
        && c.model_options.get("vae_posterior") == Some(&json!("posterior_mean_v1"))
        && c.memory_limit_bytes == Some(cap)
        && c.memory_limit_fraction.is_none()
        && request.items.len() == 1
        && request.items.iter().all(|item| {
            item.caption == PROMPT
                && item.subject_mask_path.is_none()
                && item.control_image_path.is_none()
                && item.reference_image_paths.is_empty()
        });
    if !valid {
        return Err(gen_core::Error::Unsupported(
            "prepared bounded product request differs from source-owned controls".into(),
        ));
    }
    Ok(())
}

fn validate_metrics(v: &Value) -> Result<()> {
    require(
        v["optimizer_updates"] == 4,
        "expected four real optimizer updates",
    )?;
    require(
        v["preparation_including_load_cache_seconds"]
            .as_f64()
            .is_some_and(|v| v.is_finite() && v > 0.0),
        "missing preparation timing",
    )?;
    let rows = v["steps"].as_array().ok_or("missing metric rows")?;
    require(rows.len() == 4, "four metric rows required")?;
    for (index, row) in rows.iter().enumerate() {
        require(
            row["update"] == index + 1
                && row["warmup"] == (index == 0)
                && row["adapter_changed"] == true
                && row["finite_gradients"].as_u64().is_some_and(|n| n > 0)
                && row["nonzero_gradients"].as_u64().is_some_and(|n| n > 0)
                && row["loss"].as_f64().is_some_and(f64::is_finite)
                && row["seconds"]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v > 0.0),
            "invalid bounded optimizer observation",
        )?;
    }
    for tag in [
        "moss:geometry-storage-hit",
        "moss:frozen-caption-storage-hit",
        "moss:frozen-projection-pack",
    ] {
        require(
            v["mechanism_consumption"][tag]
                .as_u64()
                .is_some_and(|n| n >= 2),
            format!("missing repeated mechanism {tag}"),
        )?;
    }
    Ok(())
}

async fn bounded_dataset(case: &mut Case, app: &axum::Router, project: &str) -> Result<Value> {
    let seal = json_file(&case.dataset.join("dataset-manifest.json"))?;
    let items = case.manifest["dataset"]["items"]
        .as_array()
        .ok_or("campaign dataset absent")?
        .clone();
    require(
        items.len() == 1,
        "sealed dataset must have one training item",
    )?;
    let sealed_items = seal["items"]
        .as_array()
        .ok_or("shared dataset manifest items missing")?;
    require(
        sealed_items.len() == 3,
        "shared dataset artifact has unexpected population",
    )?;
    let mut uploaded = Vec::new();
    let mut identities = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let image_name = text(item, "image")?;
        let caption = text(item, "caption")?;
        require(item["mask"].is_null(), "campaign forbids masks")?;
        let bytes = std::fs::read(case.dataset.join(owned_relative(image_name)?))
            .map_err(|e| e.to_string())?;
        let hash = sha(&bytes);
        let declared = sealed_items
            .iter()
            .find(|v| v["image"] == image_name)
            .ok_or("shared dataset image absent")?;
        require(
            declared["imageSha256"] == hash
                && declared["caption"] == caption
                && declared["captionSha256"] == sha(caption.as_bytes()),
            "sealed source image/caption bytes changed",
        )?;
        let caption_bytes = std::fs::read(
            case.dataset
                .join(owned_relative(text(declared, "captionFile")?)?),
        )
        .map_err(|e| e.to_string())?;
        require(
            caption_bytes == caption.as_bytes() && declared["captionSha256"] == sha(&caption_bytes),
            "sealed actual caption file bytes changed",
        )?;
        let source_image = case.retain(&format!("dataset/{image_name}"), &bytes)?;
        let source_caption = case.retain(&format!("dataset/shape-{index}.txt"), &caption_bytes)?;
        let decoded = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
        require(
            decoded.width() == 256 && decoded.height() == 256,
            "shared dataset dimensions changed",
        )?;
        let (status, upload) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project}/training/uploads"),
            image_name,
            "image/png",
            &bytes,
        )
        .await;
        let upload_receipt = case.retain_json(
            &format!("dataset/upload-{index}.json"),
            &json!({"method":"POST","route":format!("/api/v1/projects/{project}/training/uploads"),"httpStatus":status.as_u16(),"request":{"fileName":image_name,"mimeType":"image/png","sourceImage":source_image,"sourceCaption":source_caption,"caption":caption},"response":upload}),
        )?;
        identities.push(json!({"image":image_name,"imageSha256":hash,"caption":caption,"captionSha256":sha(caption.as_bytes()),"uploadId":upload["id"],"uploadReceipt":upload_receipt}));
        case.record["raw"]["datasetUploads"] = json!(identities);
        case.persist()?;
        require(
            status == StatusCode::CREATED && text(&upload, "id")?.starts_with("dataset_upload_"),
            format!("real dataset upload refused: {upload}"),
        )?;
        require(
            upload["file"]["contentHash"] == hash
                && upload["file"]["width"] == 256
                && upload["file"]["height"] == 256,
            "uploaded bytes differ from shared dataset",
        )?;
        uploaded.push(json!({"path":upload["file"]["path"],"displayName":image_name,"caption":{"text":caption,"triggerWords":["sksmoss"]},"width":256,"height":256}));
    }

    let (status, dataset) = case
        .api(
            app,
            "POST",
            &format!("/api/v1/projects/{project}/training/datasets"),
            json!({"name":"MOSS shared sealed dataset","modality":"image","items":uploaded}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("dataset creation: {dataset}"),
    )?;
    let stored = dataset["items"].as_array().ok_or("dataset has no items")?;
    require(stored.len() == 1, "stored dataset population changed")?;
    for (source, stored) in identities.iter().zip(stored) {
        require(
            stored["contentHash"] == source["imageSha256"]
                && stored["caption"]["text"] == source["caption"]
                && stored["caption"]["triggerWords"] == json!(["sksmoss"]),
            "stored dataset identity/caption mismatch",
        )?;
        text(stored, "id")?;
        text(stored, "path")?;
    }
    case.record["raw"]["datasetManifest"] = case.retain_json("dataset/manifest.json", &seal)?;
    case.persist()?;
    Ok(dataset)
}

async fn body(
    case: &mut Case,
    app: &axum::Router,
    settings: &crate::Settings,
    worker: &sceneworks_worker::Settings,
) -> Result<()> {
    // Native total-capacity product fraction remains an upper bound. The exact
    // source-owned 80 GiB lease adds a second ceiling, never increases it.
    set_cap(worker, json!({"fraction":0.85}))?;
    let product = native::moss_generation_acceptance_device(worker).map_err(|e| e.to_string())?;
    require(
        text(&product, "deviceIdentity")?.eq_ignore_ascii_case(UUID),
        "allocator CUDA UUID differs from leased physical GPU1",
    )?;
    let cap = positive(&product, "resolvedCapBytes")?.min(CAP);
    set_cap(worker, json!(cap))?;
    let metadata = native::moss_generation_acceptance_device(worker).map_err(|e| e.to_string())?;
    require(
        positive(&metadata, "resolvedCapBytes")? == cap,
        "product cap intersection changed",
    )?;
    case.record["raw"]["productFractionDevice"] = product;
    case.record["raw"]["devicePreflight"] = metadata;
    case.resources = Arc::new(Mutex::new(RootProcessResources::with_ceiling(cap)));
    case.check_resources(true)?;
    let (mut weights, revisions) = weight_inventory(settings, case)?;
    for item in weights.as_array_mut().ok_or("weight inventory missing")? {
        require(
            item["trustedHashMatches"] == true,
            "installed source bytes differ from native published identity",
        )?;
        item["canonicalPath"] =
            json!(std::fs::canonicalize(text(item, "path")?).map_err(|e| e.to_string())?);
    }
    case.record["raw"]["weightFiles"] = weights;
    case.record["raw"]["modelRevisions"] = revisions;
    case.persist()?;
    let (status, project) = case
        .api(
            app,
            "POST",
            "/api/v1/projects",
            json!({"name":"MOSS bounded four-update product acceptance"}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("project creation refused: {project}"),
    )?;
    let project = text(&project, "id")?.to_owned();
    let registered = native::register_real_acceptance_worker(worker)
        .await
        .map_err(|e| e.to_string())?;
    let mut selected_worker = worker.clone();
    selected_worker.gpu_id = registered.gpu_id.clone();
    let worker = &selected_worker;
    require(
        worker.gpu_id == "1" && claimed_device_identity(worker)?.eq_ignore_ascii_case(UUID),
        "real registered worker differs from allocator/lease UUID",
    )?;
    case.record["raw"]["workerRegistration"] =
        serde_json::to_value(registered).map_err(|e| e.to_string())?;
    let dataset = bounded_dataset(case, app, &project).await?;
    let (status, submitted) = case
        .api(
            app,
            "POST",
            &format!("/api/v1/projects/{project}/training/jobs"),
            json!({
        "targetId":"moss_image_base_lora","datasetId":dataset["id"],"config":config(),
        "outputName":"moss-bounded-four-updates","dryRun":false}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("bounded product submission refused: {submitted}"),
    )?;
    let job_id = text(&submitted, "id")?.to_owned();
    let snapshot = case.claim(app, worker, &job_id).await?;
    let plan: sceneworks_core::training::TrainingPlan =
        serde_json::from_value(submitted["payload"]["plan"].clone()).map_err(|e| e.to_string())?;
    let request_slot = Arc::new(Mutex::new(None::<gen_core::TrainingRequest>));
    let observer = Arc::clone(&request_slot);
    let loader_request = Arc::clone(&request_slot);
    let receipt: native::BoundedMossReceipt = Default::default();
    let loader_receipt = Arc::clone(&receipt);
    let metrics_path = case.root.join("metrics.json");
    let loader_metrics = metrics_path.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let completed = Arc::clone(&stop);
    let execution = native::run_training_execution_observing_request(
        worker,
        &snapshot,
        &plan,
        move |engine, spec| {
            let request = loader_request
                .lock()
                .map_err(|e| gen_core::Error::Msg(e.to_string()))?
                .clone()
                .ok_or_else(|| {
                    gen_core::Error::Msg("prepared request not observed before loading".into())
                })?;
            validate_request(&request, cap)?;
            native::real_bounded_moss_trainer(engine, spec, loader_metrics, loader_receipt)
        },
        move |request| {
            if let Ok(mut slot) = observer.lock() {
                *slot = Some(request.clone());
            }
        },
    );
    let (result, control) = tokio::join!(
        async {
            let result = execution.await;
            completed.store(true, Ordering::Release);
            result
        },
        cancel_controller(
            worker,
            &job_id,
            Default::default(),
            stop,
            case.cancel_file()?,
            case.deadline,
            Arc::clone(&case.resources)
        )
    );
    let receipt = receipt.lock().map_err(|e| e.to_string())?.clone();
    case.record["raw"]["nativeTraining"] = receipt.clone();
    case.record["raw"]["trainingError"] = json!(result.as_ref().err().map(ToString::to_string));
    case.record["raw"]["controllerError"] = json!(control.as_ref().err().map(ToString::to_string));
    let request = request_slot.lock().map_err(|e| e.to_string())?.clone();
    case.record["raw"]["preparedRequest"] = json!(request.as_ref().map(|r| format!("{r:#?}")));
    case.persist()?;
    result.map_err(|e| e.to_string())?;
    control.map_err(|e| e.to_string())?;
    validate_request(&request.ok_or("missing prepared request")?, cap)
        .map_err(|e| e.to_string())?;
    let terminal = case.get_job(app, &job_id).await?;
    case.record["raw"]["trainingTerminal"] = terminal.clone();
    case.persist()?;
    require(
        terminal["status"] == "completed"
            && receipt["nativeReturned"] == true
            && receipt["boundedObserverCompleted"] == true
            && receipt["output"]["finalLoss"]
                .as_f64()
                .is_some_and(f64::is_finite)
            && receipt["output"]["steps"] == 4,
        format!("actual native training did not complete: {terminal}"),
    )?;
    let steps = receipt["steps"]
        .as_array()
        .ok_or("missing native progress")?;
    require(
        steps.len() == 4
            && steps.iter().enumerate().all(|(i, r)| {
                r["step"] == i + 1
                    && r["total"] == 4
                    && r["loss"].as_f64().is_some_and(f64::is_finite)
            }),
        "native progress must be exactly [1,2,3,4]",
    )?;
    let metrics = json_file(&metrics_path)?;
    case.record["raw"]["metrics"] = metrics.clone();
    case.persist()?;
    validate_metrics(&metrics)?;
    let (status, library) = case
        .api(
            app,
            "GET",
            &format!("/api/v1/loras?projectId={project}"),
            Value::Null,
        )
        .await?;
    require(status == StatusCode::OK, "real Library read failed")?;
    let id = text(&terminal["result"], "loraId")?;
    let entry = library
        .as_array()
        .ok_or("Library is not an array")?
        .iter()
        .find(|e| e["id"] == id)
        .cloned()
        .ok_or("native completion did not register final Library adapter")?;
    let output = PathBuf::from(text(&terminal["result"], "outputPath")?);
    let native_output = PathBuf::from(text(&receipt["output"], "adapterPath")?);
    require(
        std::fs::canonicalize(&output).map_err(|e| e.to_string())?
            == std::fs::canonicalize(&native_output).map_err(|e| e.to_string())?
            && file_sha(&output)? == file_sha(&native_output)?,
        "Library completion is not native final output",
    )?;
    resolve_training_library_adapter(&entry, &output)?;
    require(
        entry["baseModel"] == "moss_image_base"
            && entry["networkType"] == "lora"
            && entry["source"]["provider"] == "training"
            && entry["provenance"]["configSnapshot"]
                == submitted["payload"]["plan"]["provenance"]["configSnapshot"],
        "Library identity or full submitted provenance differs",
    )?;
    let final_bytes = std::fs::read(&output).map_err(|e| e.to_string())?;
    case.record["raw"]["finalAdapter"] = case.retain("adapter/final.safetensors", &final_bytes)?;
    case.record["raw"]["trainingLibraryRegistration"] =
        json!({"entry":entry,"outputPath":output,"sha256":file_sha(&output)?});
    case.persist()?;
    // Ordinary generation, using the selected Library ID, is the only sample.
    let sample = generate_with_controls(
        case,
        app,
        worker,
        &project,
        "adapter-sample",
        Some((&entry, 1.0)),
        GenerationControls {
            mode: "off",
            seed: 7,
            advanced: json!({"steps":2,"guidanceScale":1.0,"mossImage":{"acceleration":"off"}}),
        },
    )
    .await?;
    case.record["raw"]["sample"] = json!({"relativePath":sample["artifact"]["path"],"sha256":sample["artifact"]["sha256"],"execution":sample});
    case.record["acceptanceDisposition"] = json!("accepted");
    case.persist()?;
    Ok(())
}

#[tokio::test]
#[ignore = "manual bounded product acceptance; exact source/UUID lease required"]
async fn moss_bounded_product_acceptance() {
    let mut case = admitted().expect("source-owned bounded admission before model work");
    let temp = tempfile::tempdir().expect("isolated app state outside evidence");
    let mut settings = test_settings(&temp);
    settings.host_os = std::env::consts::OS.into();
    seed_acceptance_manifests(&settings).expect("shipped manifests");
    let app = create_app(settings.clone()).expect("isolated API");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("owned loopback");
    let worker = worker_settings(
        format!("http://{}", listener.local_addr().unwrap()),
        &settings,
        "auto".into(),
    );
    let directory = Arc::new(case.root.join("worker-http"));
    std::fs::create_dir_all(directory.as_path()).unwrap();
    let sequence = Arc::new(AtomicU64::new(0));
    let served = app
        .clone()
        .layer(axum::middleware::from_fn(move |request, next| {
            observe_worker_http(Arc::clone(&directory), Arc::clone(&sequence), request, next)
        }));
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, served)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let sampler = native::PeakSampler::start_rendered();
    let started = Instant::now();
    let result = std::panic::AssertUnwindSafe(body(&mut case, &app, &settings, &worker))
        .catch_unwind()
        .await;
    let peak = sampler.stop() * 1024 * 1024;
    case.record["raw"]["peakDevice"] = json!({"bytes":peak,"physicalIndex":1,"uuid":UUID,"scope":"selected-device-total-including-other-residency","preparationTrainingAndSampleSeconds":started.elapsed().as_secs_f64()});
    let _ = stop.send(());
    let server_result = server.await;
    let outcome = match result {
        Ok(v) => v,
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "native body panicked".into())),
    };
    let outcome = outcome
        .and_then(|_| require(peak > 0, "selected-device peak unavailable"))
        .and_then(|_| case.check_resources(true))
        .and_then(|_| {
            require(
                matches!(server_result, Ok(Ok(()))),
                "API server did not join",
            )
        });
    case.record["raw"]["serverJoined"] = json!(matches!(server_result, Ok(Ok(()))));
    case.record["raw"]["nativeBodyReturnedAtUnixMs"] = json!(unix_ms().ok());
    if let Err(error) = &outcome {
        case.record["acceptanceDisposition"] = json!("failed");
        case.record["failureReason"] = json!(error);
    }
    case.persist()
        .expect("retain bounded evidence before assertions");
    outcome.expect("bounded real product acceptance");
    println!("\nMODEL_INFERENCE_BODY_COMPLETED {SELECTOR}");
}

// Raw metadata only: no model, tensor, fixture evaluation, or CUDA call.
#[test]
fn bounded_metric_contract_rejects_partial_or_unobserved_work() {
    let valid = json!({"optimizer_updates":4,"preparation_including_load_cache_seconds":1.0,
        "steps":(1..=4).map(|i|json!({"update":i,"warmup":i==1,"adapter_changed":true,
            "finite_gradients":2,"nonzero_gradients":1,"loss":0.5,"seconds":1.0})).collect::<Vec<_>>(),
        "mechanism_consumption":{"moss:geometry-storage-hit":4,"moss:frozen-caption-storage-hit":3,"moss:frozen-projection-pack":4}});
    validate_metrics(&valid).unwrap();
    let mut partial = valid.clone();
    partial["optimizer_updates"] = json!(3);
    assert!(validate_metrics(&partial).is_err());
    let mut missing = valid.clone();
    missing["steps"][2]["finite_gradients"] = json!(0);
    assert!(validate_metrics(&missing).is_err());
    let mut unchanged = valid.clone();
    unchanged["steps"][3]["adapter_changed"] = json!(false);
    assert!(validate_metrics(&unchanged).is_err());
    let mut cold = valid;
    cold["mechanism_consumption"]["moss:frozen-caption-storage-hit"] = json!(1);
    assert!(validate_metrics(&cold).is_err());
}
