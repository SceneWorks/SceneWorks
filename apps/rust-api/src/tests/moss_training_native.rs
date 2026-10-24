//! CPU-native integration witness, not real-weight or GPU acceptance.
//!
//! Tiny, generated model tensors exercise the actual Qwen text encoder, four-stage
//! FLUX VAE, MOSS trainer and generator. Catalog placeholders are used only for
//! the unchanged pre-loader namespace resolution; they are NEVER loaded as model
//! tensors or represented as authenticated published weights.

use super::support::*;
use futures_util::FutureExt;
use sceneworks_worker::test_support::{
    gen_core::{
        self, moss_image::MossImageVariant, AdapterKind, CancelFlag, GenerationOutput,
        WeightsSource,
    },
    generation_request, Fixture, TrainingProbe,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn builtin_model(id: &str) -> Value {
    let (_, contents) = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .unwrap();
    let manifest: Value = serde_json::from_str(&strip_jsonc_comments(contents)).unwrap();
    manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == id)
        .unwrap()
        .clone()
}

fn snapshot_path(temp: &tempfile::TempDir, download: &Value) -> PathBuf {
    temp.path()
        .join("data/cache/huggingface/hub")
        .join(format!(
            "models--{}",
            download["repo"].as_str().unwrap().replace('/', "--")
        ))
        .join("snapshots")
        .join(download["revision"].as_str().unwrap())
}

fn seed_catalog_namespace(temp: &tempfile::TempDir, model: &Value) -> (PathBuf, PathBuf) {
    let downloads = model["downloads"].as_array().unwrap();
    let primary = downloads
        .iter()
        .find(|download| download["default"] == true)
        .unwrap();
    super::yue2_catalog::seed_snapshot(temp, primary);
    let shared = downloads
        .iter()
        .find(|component| component["componentId"] == "moss-shared-snapshot")
        .unwrap();
    super::yue2_catalog::seed_snapshot(temp, shared);
    (snapshot_path(temp, primary), snapshot_path(temp, shared))
}

fn worker_settings(api_url: String, api_settings: &crate::Settings) -> sceneworks_worker::Settings {
    sceneworks_worker::Settings {
        api_url,
        access_token: None,
        data_dir: api_settings.data_dir.clone(),
        resolved_cache: sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy::from_env_or_safe_default(),
        config_dir: api_settings.config_dir.clone(),
        worker_id: TEST_TRAINING_WORKER_ID.to_owned(),
        gpu_id: if cfg!(target_os = "macos") { "mlx" } else { "test-gpu" }.to_owned(),
        is_child_worker: false,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 10,
        huggingface_base_url: "http://127.0.0.1:9".to_owned(),
        huggingface_token: None,
        credentials: Vec::new(),
        max_lora_url_bytes: 1,
        max_model_url_bytes: 1,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: cfg!(target_os = "macos"),
        backend_candle_enabled: cfg!(all(not(target_os = "macos"), feature = "backend-candle")),
        external_model_roots: Vec::new(),
        gpu_memory_limit_bytes: 0,
    }
}

async fn claim_as_existing_worker(app: &axum::Router, job_id: &str) {
    let (status, claimed) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs/claim",
        json!({"workerId":TEST_TRAINING_WORKER_ID}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{claimed}");
    assert_eq!(claimed["job"]["id"], job_id);
}

struct FixtureSources {
    primary: PathBuf,
    shared: PathBuf,
    observed_cancel: Arc<Mutex<Option<CancelFlag>>>,
    probe: Option<TrainingProbe>,
}

struct FixtureMask {
    display_name: String,
    path: PathBuf,
    width: u32,
    height: u32,
}

async fn upload_fixture_masks(
    app: &axum::Router,
    project_id: &str,
    dataset: &Value,
    masks: Vec<FixtureMask>,
) {
    let dataset_id = dataset["id"].as_str().unwrap();
    let count = masks.len();
    assert!(count > 0);
    for mask in masks {
        let item = dataset["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["displayName"] == mask.display_name)
            .unwrap();
        let item_id = item["id"].as_str().unwrap();
        let bytes = std::fs::read(&mask.path).unwrap();
        let (status, report) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/items/{item_id}/subject-mask"),
            "fixture-mask.png", "image/png", &bytes,
        ).await;
        assert_eq!(status, StatusCode::OK, "{report}");
        let stored = report["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|stored| stored["itemId"] == item_id)
            .unwrap();
        assert_eq!(stored["source"], "upload");
        assert_eq!(stored["hasMask"], true);
        assert_eq!(stored["empty"], false);
        let path = stored["maskPath"].as_str().unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/projects/{project_id}/files/{path}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let normalized = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let decoded = image::load_from_memory(&normalized).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height()),
            (mask.width, mask.height)
        );
    }
    let (status, coverage) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/subject-masks"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{coverage}");
    assert_eq!(coverage["total"], json!(count));
    assert_eq!(coverage["masked"], json!(count));
    assert_eq!(coverage["uploaded"], json!(count));
    assert_eq!(coverage["empty"], 0);
}

// This runs only inside an explicitly selected ignored fixture, before model loading.
// It observes this private API process, never a user's worker or another process tree.
fn fixture_bootstrap_rss_bytes() -> Result<u64, String> {
    let pid = std::process::id().to_string();
    let (output, multiplier) = if cfg!(target_os = "macos") {
        (
            std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &pid])
                .output(),
            1024_u64,
        )
    } else if cfg!(target_os = "windows") {
        (
            std::process::Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    &format!("(Get-Process -Id {pid} -ErrorAction Stop).WorkingSet64"),
                ])
                .output(),
            1_u64,
        )
    } else {
        return Err(
            "CPU API bootstrap RSS requires the explicit macOS or Windows validation route"
                .to_owned(),
        );
    };
    let output = output.map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "CPU API bootstrap RSS query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| error.to_string())?
        .trim()
        .parse::<u64>()
        .map_err(|error| error.to_string())?
        .checked_mul(multiplier)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| "CPU API bootstrap RSS is zero or overflowed".to_owned())
}

async fn execute_fixture_training(
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    job_id: &str,
    fixture: &Fixture,
    variant: MossImageVariant,
    base: &str,
    sources: FixtureSources,
) -> sceneworks_worker::WorkerResult<()> {
    let (_, claimed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    let snapshot = serde_json::from_value(claimed.clone()).unwrap();
    let plan: sceneworks_core::training::TrainingPlan =
        serde_json::from_value(claimed["payload"]["plan"].clone()).unwrap();
    assert_eq!(
        serde_json::to_value(&plan.config).unwrap(),
        claimed["payload"]["plan"]["config"]
    );
    let trainer = match sources.probe {
        Some(probe) => fixture.trainer_with_probe(variant, probe).unwrap(),
        None => fixture.trainer(variant).unwrap(),
    };
    let expected_primary = std::fs::canonicalize(sources.primary).unwrap();
    let expected_shared = std::fs::canonicalize(sources.shared).unwrap();
    let expected_base = base.to_owned();
    let observed = Arc::clone(&sources.observed_cancel);
    let admission_fixture = fixture.clone();
    let execution = sceneworks_worker::test_support::run_training_execution_observing_request(
        worker,
        &snapshot,
        &plan,
        move |engine, spec| {
            assert_eq!(engine, expected_base);
            assert!(
                matches!(&spec.weights, WeightsSource::Dir(path) if std::fs::canonicalize(path).unwrap() == expected_primary)
            );
            assert!(
                matches!(spec.components.get("moss-shared-snapshot"), Some(WeightsSource::Dir(path)) if std::fs::canonicalize(path).unwrap() == expected_shared)
            );
            Ok(trainer)
        },
        move |request| {
            assert!(
                !request.cancel.is_cancelled(),
                "every new worker execution owns a fresh token"
            );
            *observed.lock().unwrap() = Some(request.cancel.clone());
            let memory_plan = admission_fixture.memory_plan(variant, request);
            let bootstrap_rss = fixture_bootstrap_rss_bytes();
            let envelope = match (&memory_plan, &bootstrap_rss) {
                (Ok(plan), Ok(rss)) => {
                    let bytes = |key: &str| {
                        plan[key]
                            .as_u64()
                            .ok_or_else(|| format!("native memory plan omits {key}"))
                    };
                    bytes("tensor_peak_bytes").and_then(|tensor| {
                        let staged = tensor
                            .checked_add(bytes("host_staging_peak_bytes")?)
                            .ok_or_else(|| "native staging demand overflow".to_owned())?;
                        staged
                            .max(bytes("encoder_only_peak_bytes")?)
                            .checked_add(*rss)
                            .ok_or_else(|| "API bootstrap plus native demand overflow".to_owned())
                    })
                }
                (Err(error), _) => Err(error.to_string()),
                (_, Err(error)) => Err(error.clone()),
            };
            let demand = match &memory_plan {
                Ok(value) => value.clone(),
                Err(error) => json!({"error":error.to_string()}),
            };
            emit_case_diagnostic(
                &json!({
                    "phase":"prepared_native_request",
                    "sourceIdentity":admission_fixture.data.source_identity,
                    "nativeConfig":{
                        "rank":request.config.rank,"alpha":request.config.alpha,
                        "steps":request.config.steps,"seed":request.config.seed,
                        "batchSize":request.config.batch_size,
                        "gradientAccumulation":request.config.gradient_accumulation,
                        "gradientCheckpointing":request.config.gradient_checkpointing,
                        "trainDtype":request.config.train_dtype,
                        "networkType":format!("{:?}",request.config.network_type),
                        "cacheLatents":request.config.cache_latents,
                        "cacheTextEmbeddings":request.config.cache_text_embeddings,
                        "memoryLimitBytes":request.config.memory_limit_bytes,
                        "memoryLimitFraction":request.config.memory_limit_fraction,
                        "modelOptions":request.config.model_options,
                    },
                    "itemCount":request.items.len(),
                    "nativeHeaderDemand":demand,
                    "bootstrapRssBytes":bootstrap_rss.as_ref().ok(),
                    "bootstrapRssError":bootstrap_rss.as_ref().err(),
                    "conservativeBootstrapPlusDemandBytes":envelope.as_ref().ok(),
                    "envelopeError":envelope.as_ref().err(),
                    "demandIsEstimateNotMeasuredRss":true,
                    "externalAggregateOwnedRssCapBytes":2_u64 * 1024 * 1024 * 1024,
                }),
                None,
            )
            .expect("record actual prepared-request demand before native loading");
            memory_plan.expect("native header-derived request admission must succeed");
            assert!(
                envelope.expect("API bootstrap RSS and native demand must be observable") <= 2_u64 * 1024 * 1024 * 1024,
                "the API bootstrap plus conservative native header demand does not fit the admitted 2 GiB CPU validation contract",
            );
        },
    );
    tokio::pin!(execution);
    match tokio::time::timeout(Duration::from_secs(120), &mut execution).await {
        Ok(result) => result,
        Err(_) => {
            // Safety teardown only, never the successful cancellation stimulus: keep
            // driving the real consumer while the owned native task cooperatively stops.
            if let Some(flag) = sources.observed_cancel.lock().unwrap().as_ref() {
                flag.cancel();
            }
            let _ = tokio::time::timeout(Duration::from_secs(30), &mut execution)
                .await
                .expect("timed-out CPU fixture must still join its owned trainer");
            Err(sceneworks_worker::WorkerError::InvalidPayload(
                "CPU native integration fixture exceeded its 120-second deadline".to_owned(),
            ))
        }
    }
}

fn assert_installed_adapter(
    reports: &[sceneworks_worker::test_support::gen_core::AdapterApplyReport],
    saved_path: &Path,
) {
    let report = reports
        .iter()
        .find(|report| report.adapter_path == saved_path)
        .expect("the actual generator reports the selected saved adapter");
    assert!(report.applied > 0);
    assert!(report.skipped.is_empty(), "{:?}", report.skipped);
}

/// Write already-computed diagnostics directly to stderr before assertions so
/// the CI process log retains successful rows as well as failures. No new output
/// root, runtime setting, global panic hook, or tensor/weight cloning is involved.
fn emit_case_diagnostic(context: &Value, probe: Option<&TrainingProbe>) -> Result<(), String> {
    use std::io::Write;
    let (receipts, read_error) = match probe.map(TrainingProbe::receipts) {
        Some(Ok(receipts)) => (receipts, None),
        Some(Err(error)) => (Vec::new(), Some(error.to_string())),
        None => (Vec::new(), None),
    };
    let record = json!({"context":context,"rawProbeReceipts":receipts,"probeReadError":read_error});
    let encoded = serde_json::to_string(&record).map_err(|error| error.to_string())?;
    writeln!(
        std::io::stderr().lock(),
        "MOSS_CONSUMER_CASE_DIAGNOSTIC {encoded}"
    )
    .map_err(|error| error.to_string())?;
    match read_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

async fn capture_run_diagnostic(
    app: &axum::Router,
    job_id: &str,
    context: &mut Value,
    probe: Option<&TrainingProbe>,
    run: &sceneworks_worker::WorkerResult<()>,
) -> Result<(), String> {
    let (status, observed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{job_id}"),
        Value::Null,
    )
    .await;
    context["jobId"] = json!(job_id);
    context["lastObservedHttpStatus"] = json!(status.as_u16());
    context["lastObservedStatus"] = observed["status"].clone();
    context["config"] = observed["payload"]["plan"]["config"].clone();
    context["lastObservedError"] = observed["error"].clone();
    context["lastObservedMetrics"] = json!({
        "stepsCompleted":observed["result"]["stepsCompleted"],
        "finalLoss":observed["result"]["finalLoss"],
        "resumeCheckpointStep":observed["result"]["resumeCheckpointStep"],
        "outputPath":observed["result"]["outputPath"],
    });
    context["returnedOutcome"] = match run {
        Ok(()) => json!({"kind":"ok"}),
        Err(error) => json!({"kind":"error","message":error.to_string()}),
    };
    emit_case_diagnostic(context, probe)
}

#[tokio::test]
#[ignore = "full native model execution; explicit moss-native-api manual validation only"]
async fn moss_real_dataset_base_lora_round_trip_cpu() {
    run_moss_case("moss_image_base", MossImageVariant::Base, "lora").await;
}

#[tokio::test]
#[ignore = "full native model execution; explicit moss-native-api manual validation only"]
async fn moss_real_dataset_base_lokr_round_trip_cpu() {
    run_moss_case("moss_image_base", MossImageVariant::Base, "lokr").await;
}

#[tokio::test]
#[ignore = "full native model execution; explicit moss-native-api manual validation only"]
async fn moss_real_dataset_sft_lora_round_trip_cpu() {
    run_moss_case("moss_image_sft", MossImageVariant::Sft, "lora").await;
}

#[tokio::test]
#[ignore = "full native model execution; explicit moss-native-api manual validation only"]
async fn moss_real_dataset_sft_lokr_round_trip_cpu() {
    run_moss_case("moss_image_sft", MossImageVariant::Sft, "lokr").await;
}

async fn run_moss_case(base: &str, variant: MossImageVariant, method: &str) {
    // Hold the ONE shared guard for fixture creation, API resolution, training and teardown.
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    for base in ["moss_image_base", "moss_image_sft"] {
        seed_catalog_namespace(&temp, &builtin_model(base));
    }
    let observe_orientation = base == "moss_image_sft" && method == "lokr";
    let fixture = Fixture::create(&temp.path().join(format!("native-{base}-{method}"))).unwrap_or_else(|error| {
        eprintln!("MOSS_CONSUMER_CASE_DIAGNOSTIC {}", json!({"base":base,"method":method,"phase":"fixture_create","error":error.to_string()}));
        panic!("native fixture construction failed: {error}");
    });
    let probe = observe_orientation
        .then(|| TrainingProbe::new(fixture.data.evaluation_manifest().unwrap()));
    let mut diagnostic = json!({
        "base":base,"method":method,"execution":"cpu-private-fixture-not-gpu-proof",
        "sourceIdentity":fixture.data.source_identity,"sourceFiles":fixture.data.files,
        "sourceTensorBytes":fixture.data.tensor_file_bytes,
        "evaluation":probe.as_ref().map(|probe| json!({"identity":probe.evaluation().identity,"width":probe.evaluation().width,"height":probe.evaluation().height,"noiseSeed":probe.evaluation().noise_seed,"sigma":probe.evaluation().sigma})),
    });
    let mut diagnostic_errors = Vec::new();
    diagnostic["phase"] = json!("fixture_constructed");
    if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
        diagnostic_errors.push(error);
    }
    let app = create_app(settings.clone()).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let cancel_once = Arc::new(Mutex::new(None::<String>));
    let cancel_control = Arc::clone(&cancel_once);
    let cancel_app = app.clone();
    let served_app = app.clone().layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let control = Arc::clone(&cancel_control);
            let cancel_app = cancel_app.clone();
            async move {
                if !req.uri().path().ends_with("/progress") {
                    return next.run(req).await;
                }
                let (parts, body) = req.into_parts();
                let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
                let progress: Value = serde_json::from_slice(&bytes).unwrap();
                let id = parts.uri.path().strip_prefix("/api/v1/jobs/")
                    .and_then(|path| path.strip_suffix("/progress"));
                let cancel_id = {
                    let mut control = control.lock().unwrap();
                    if progress["stage"] == "checkpointing"
                        && progress["result"]["resumeCheckpointStep"].as_u64().is_some_and(|step| step >= 3)
                        && control.as_deref() == id
                    {
                        control.take()
                    } else {
                        None
                    }
                };
                if let Some(id) = cancel_id {
                    let (_, checkpoint_job) = request(cancel_app.clone(), "GET", &format!("/api/v1/jobs/{id}"), Value::Null).await;
                    let output = &checkpoint_job["payload"]["plan"]["output"];
                    let state = gen_core::moss_image_training::recovery::resume_state(
                        Path::new(output["outputDir"].as_str().unwrap()),
                        output["fileName"].as_str().unwrap(),
                    ).unwrap();
                    assert!(state.step >= 3 && state.step < 100 && state.update > 0, "cancellation must follow actual verified optimizer state before completion");
                    let (status, response) = request(cancel_app, "POST", &format!("/api/v1/jobs/{id}/cancel"), Value::Null).await;
                    assert_eq!(status, StatusCode::OK, "{response}");
                    // Real HTTP backpressure, not a clock override or trainer stub. The
                    // bounded native event channel stalls the producer before step 100,
                    // and the next worker poll observes the real API cancellation.
                    tokio::time::sleep(Duration::from_millis(2100)).await;
                }
                next.run(axum::extract::Request::from_parts(parts, Body::from(bytes))).await
            }
        },
    ));
    let server = tokio::spawn(async move {
        axum::serve(listener, served_app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    // Catch assertion failures so this owned router is shut down and joined even on failure.
    let outcome = std::panic::AssertUnwindSafe(async {
        let (_, project) = request(app.clone(), "POST", "/api/v1/projects", json!({"name":"Native MOSS CPU"})).await;
        let project_id = project["id"].as_str().unwrap();
        let worker = worker_settings(api_url, &settings);
        let (status, registered) = request(app.clone(), "POST", "/api/v1/workers/register", json!({
            "workerId":worker.worker_id,"gpuId":worker.gpu_id,
            "gpuName":"CPU-native fixture, not GPU execution evidence",
            "capabilities":["gpu","lora_train","lora_train_execute"],"loadedModels":[]
        })).await;
        assert_eq!(status, StatusCode::OK, "{registered}");
            let model = builtin_model(base);
            let (primary, shared) = seed_catalog_namespace(&temp, &model);
            // The app uses the real builtin catalog, not a fixture-specific model entry.
                let observed_cancel = Arc::new(Mutex::new(None));
                let mut dataset_items = Vec::new();
                let mut masks = Vec::new();
                for (index, item) in fixture.data.items().unwrap().into_iter().enumerate() {
                    let bytes = std::fs::read(&item.image_path).unwrap();
                    let file_name = format!("native-{base}-{method}-{index}.png");
                    if observe_orientation {
                        let decoded = image::load_from_memory(&bytes).unwrap();
                        masks.push(FixtureMask {
                            display_name: file_name.clone(),
                            path: item.subject_mask_path.as_ref().expect("native fixture supplies its real training mask").clone(),
                            width: decoded.width(), height: decoded.height(),
                        });
                    }
                    let (status, asset) = request_multipart_upload(
                        app.clone(),
                        &format!("/api/v1/projects/{project_id}/assets"),
                        &file_name,
                        "image/png",
                        &bytes,
                    ).await;
                    assert_eq!(status, StatusCode::CREATED, "{asset}");
                    dataset_items.push(json!({"assetId":asset["id"],"displayName":file_name,"caption":{"text":item.caption}}));
                }
                assert!(!dataset_items.is_empty());
                let (status, dataset) = request(
                    app.clone(), "POST", &format!("/api/v1/projects/{project_id}/training/datasets"),
                    json!({"name":format!("Native {base} {method}"),"items":dataset_items}),
                ).await;
                assert_eq!(status, StatusCode::CREATED, "{dataset}");
                if observe_orientation {
                    upload_fixture_masks(&app, project_id, &dataset, masks).await;
                }
                let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
                let target = registry["targets"].as_array().unwrap().iter()
                    .find(|target| target["id"] == format!("{base}_lora")).unwrap();
                let mut config = target["defaults"].clone();
                config["rank"] = json!(2);
                config["alpha"] = json!(0.5);
                config["learningRate"] = json!(0.01);
                let exercise_resume = base == "moss_image_base" && method == "lora";
                let total_steps = if exercise_resume { 100 } else { 6 };
                config["steps"] = json!(total_steps);
                config["batchSize"] = json!(1);
                config["gradientAccumulation"] = json!(2);
                config["saveEvery"] = json!(3);
                // Randomize produces a concrete product-representable seed BEFORE
                // submission; the other Base method explicitly exercises literal zero.
                let literal_seed = if exercise_resume {
                    crate::random_image_seeds(1)[0].as_u64().unwrap()
                } else if base == "moss_image_base" {
                    0
                } else {
                    173
                };
                config["seed"] = json!(literal_seed);
                config["resolution"] = json!(256);
                config["advanced"]["networkType"] = json!(method);
                config["advanced"]["decomposeFactor"] = json!(4);
                config["advanced"]["mixedPrecision"] = json!("f32");
                config["advanced"]["gradientCheckpointing"] = json!(false);
                // Cover all four cache combinations. The 100-step cancellation
                // case caches both frozen encoders; uncached modes stay bounded at six steps.
                config["advanced"]["cacheLatents"] = json!(method == "lora");
                config["advanced"]["cacheTextEmbeddings"] = json!(base == "moss_image_base");
                // A genuine non-square native encode/train geometry, not a square
                // merely carried in rectangular syntax. Generation below stays 32×32.
                config["advanced"]["resolutionBuckets"] = json!([{"width":32,"height":64,"repeats":1}]);
                if observe_orientation {
                    config["advanced"]["subjectMaskLoss"] = json!(true);
                }
                // Preview OFF: this witness never renders the scalar 256px preview canvas.
                config["advanced"]["sampleEvery"] = json!(0);
                let output_name = format!("Native {base} {method}");
                diagnostic["config"] = config.clone();
                diagnostic["phase"] = json!("before_training_submit");
                if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
                    diagnostic_errors.push(error);
                }
                let (status, job) = request(
                    app.clone(), "POST", &format!("/api/v1/projects/{project_id}/training/jobs"),
                    json!({"targetId":target["id"],"datasetId":dataset["id"],"config":config,"outputName":output_name,"dryRun":false}),
                ).await;
                diagnostic["jobId"] = job["id"].clone();
                diagnostic["lastObservedStatus"] = job["status"].clone();
                diagnostic["lastObservedHttpStatus"] = json!(status.as_u16());
                diagnostic["phase"] = json!("training_submit_returned");
                if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
                    diagnostic_errors.push(error);
                }
                assert_eq!(status, StatusCode::CREATED, "{job}");
                assert!(job["payload"].get("subjectMaskPrepass").is_none(), "uploaded masks must not schedule SAM3 or another model job");
                let original_job_id = job["id"].as_str().unwrap().to_owned();
                let mut job_id = original_job_id.clone();
                claim_as_existing_worker(&app, &job_id).await;
                if exercise_resume {
                    *cancel_once.lock().unwrap() = Some(job_id.clone());
                }
                let run = execute_fixture_training(
                    &app, &worker, &job_id, &fixture, variant, base, FixtureSources {
                            primary: primary.clone(), shared: shared.clone(),
                            observed_cancel: Arc::clone(&observed_cancel),
                            probe: probe.clone(),
                        },
                ).await;
                diagnostic["phase"] = json!("initial_training_returned");
                if let Err(error) = capture_run_diagnostic(&app, &job_id, &mut diagnostic, probe.as_ref(), &run).await {
                    diagnostic_errors.push(error);
                }
                if exercise_resume {
                    assert!(matches!(run, Err(sceneworks_worker::WorkerError::Canceled(_))));
                    assert!(cancel_once.lock().unwrap().is_none(), "the real progress route must trigger cancellation");
                    let canceled_flag = observed_cancel.lock().unwrap().as_ref().unwrap().clone();
                    assert!(canceled_flag.is_cancelled(), "the real worker API poll must set the actual request token");
                    let (_, canceled) = request(app.clone(), "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
                    assert_eq!(canceled["status"], "canceled", "{canceled}");
                    let checkpoint = canceled["result"]["resumeCheckpointStep"].as_u64().unwrap();
                    assert!(checkpoint > 0 && checkpoint < total_steps);
                    assert_ne!(canceled["result"]["trainedArtifactExported"], true);
                    let output = &job["payload"]["plan"]["output"];
                    let sealed = gen_core::moss_image_training::recovery::resume_state(
                        Path::new(output["outputDir"].as_str().unwrap()), output["fileName"].as_str().unwrap(),
                    ).unwrap();
                    assert_eq!(sealed.resolved_seed, literal_seed);
                    assert_eq!(job["payload"]["plan"]["config"]["seed"], literal_seed);
                    let bundle = gen_core::moss_image_training::recovery::resume_bundle_path(
                        Path::new(output["outputDir"].as_str().unwrap()),
                        output["fileName"].as_str().unwrap(),
                        u32::try_from(checkpoint).unwrap(),
                    ).unwrap();
                    // Corrupt ONLY this joined run's private fixture state. The API must
                    // refuse it, not enqueue a fresh run or fall back silently. Restore
                    // exact native-produced bytes before the valid resume below.
                    let state_path = bundle.join("run.json");
                    let sealed_bytes = std::fs::read(&state_path).unwrap();
                    std::fs::write(&state_path, b"corrupt native fixture state").unwrap();
                    let (status, corrupt) = request(app.clone(), "POST", &format!("/api/v1/jobs/{job_id}/retry"), json!({"resume":true})).await;
                    std::fs::write(&state_path, sealed_bytes).unwrap();
                    assert_eq!(status, StatusCode::BAD_REQUEST, "{corrupt}");
                    assert!(corrupt.to_string().contains("checkpoint verification failed"), "{corrupt}");
                    let (status, resumed) = request(app.clone(), "POST", &format!("/api/v1/jobs/{job_id}/retry"), json!({"resume":true})).await;
                    assert_eq!(status, StatusCode::CREATED, "{resumed}");
                    let mut expected_plan = job["payload"]["plan"].clone();
                    expected_plan["config"]["advanced"]["resume"] = json!(true);
                    assert_eq!(resumed["payload"]["plan"], expected_plan, "retry must preserve the original horizon, seed, dataset and numerical controls");
                    job_id = resumed["id"].as_str().unwrap().to_owned();
                    claim_as_existing_worker(&app, &job_id).await;
                    // A fresh real trainer on the SAME worker restores the existing bundle;
                    // no worker process, model loop or mock output is substituted.
                    let resumed_run = execute_fixture_training(
                        &app, &worker, &job_id, &fixture, variant, base, FixtureSources {
                            primary: primary.clone(), shared: shared.clone(),
                            observed_cancel: Arc::clone(&observed_cancel),
                            probe: probe.clone(),
                        },
                    ).await;
                    diagnostic["phase"] = json!("resumed_training_returned");
                    if let Err(error) = capture_run_diagnostic(&app, &job_id, &mut diagnostic, probe.as_ref(), &resumed_run).await {
                        diagnostic_errors.push(error);
                    }
                    resumed_run.unwrap();
                    assert!(canceled_flag.is_cancelled(), "re-entry must never clear the old request token");
                    assert!(!observed_cancel.lock().unwrap().as_ref().unwrap().is_cancelled(), "re-entry must own an independent non-cancelled token");
                } else {
                    run.unwrap();
                }
                let (_, completed) = request(app.clone(), "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
                assert_eq!(completed["status"], "completed", "{completed}");
                assert_eq!(completed["result"]["loraRegistered"], true, "{completed}");
                if let Some(probe) = &probe {
                    let receipts = probe.receipts().unwrap();
                    assert!(!receipts.is_empty());
                    assert!(receipts.iter().all(|receipt| receipt.complete && receipt.run_scope_returned && receipt.error.is_none()));
                    let actual_rows = receipts.iter().filter_map(|receipt| receipt.masks.get("32x64")).collect::<Vec<_>>();
                    assert!(!actual_rows.is_empty(), "actual encoded mask geometry must retain portrait orientation, not a planned-key echo or 64x32 transpose");
                    assert!(actual_rows.iter().all(|row| row.len() == 2), "actual packed mask width must be 32/16");
                }
                assert_eq!(completed["result"]["stepsCompleted"], total_steps);
                assert!(completed["result"]["finalLoss"].as_f64().unwrap().is_finite());
                assert_eq!(completed["result"]["resumeCheckpointStep"], total_steps);
                let output = &completed["payload"]["plan"]["output"];
                // Step 3 is a retained partial-accumulation checkpoint, independent
                // of how far the producer advanced while the 64-event channel filled.
                let third_bundle = gen_core::moss_image_training::recovery::resume_bundle_path(
                    Path::new(output["outputDir"].as_str().unwrap()), output["fileName"].as_str().unwrap(), 3,
                ).unwrap();
                let third = gen_core::moss_image_training::recovery::bundle::read(&third_bundle);
                diagnostic["phase"] = json!("sealed_step3_read");
                diagnostic["sealedStep3"] = match &third {
                    Ok(state) => json!({"step":state.step,"update":state.update,"resolvedSeed":state.resolved_seed,"geometry":state.geometry,"accumulatedTokens":state.accumulated_tokens}),
                    Err(error) => json!({"error":error.to_string()}),
                };
                if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
                    diagnostic_errors.push(error);
                }
                let third = third.unwrap();
                assert_eq!((third.step, third.update), (3, 1));
                assert!(!third.geometry.is_empty());
                assert!(third.geometry.iter().all(|entry| (entry.width, entry.height) == (32, 64)));
                // This denominator comes from actual encoded [N,128,H/16,W/16]
                // tensors, not planned area: a 32×32 fallback produces 4, not 8.
                assert_eq!(third.accumulated_tokens, 8);
                let sealed = gen_core::moss_image_training::recovery::resume_state(
                    Path::new(output["outputDir"].as_str().unwrap()), output["fileName"].as_str().unwrap(),
                ).unwrap();
                assert_eq!(sealed.resolved_seed, literal_seed);
                assert_eq!(u64::from(sealed.step), total_steps);
                assert_eq!(u64::from(sealed.update), total_steps / 2);
                let saved_path = PathBuf::from(completed["result"]["outputPath"].as_str().unwrap());
                assert!(saved_path.is_file());
                let (_, library) = request(app.clone(), "GET", &format!("/api/v1/loras?projectId={project_id}"), Value::Null).await;
                let entry = library.as_array().unwrap().iter().find(|entry| entry["name"] == output_name).unwrap();
                assert_eq!(entry["baseModel"], base);
                assert_eq!(entry["networkType"], method);
                assert_eq!(entry["source"]["provider"], "training");
                assert_eq!(entry["provenance"]["trainingJobId"], original_job_id);
                assert_eq!(entry["provenance"]["configSnapshot"]["alpha"], 0.5);
                assert_eq!(entry["provenance"]["configSnapshot"]["seed"], literal_seed);
                let (status, selected) = request(app.clone(), "POST", "/api/v1/image/jobs", json!({
                    "projectId":project_id,"model":base,"prompt":"a tiny native fixture subject",
                    "loras":[{"id":entry["id"],"weight":1.0}],
                    "advanced":{"mossImage":{"acceleration":"off"}}
                })).await;
                assert_eq!(status, StatusCode::CREATED, "{selected}");
                assert_eq!(selected["payload"]["loras"][0]["id"], entry["id"]);
                assert_eq!(selected["payload"]["loras"][0]["weight"], 1.0);
                let other_parent = if base == "moss_image_base" { "moss_image_sft" } else { "moss_image_base" };
                let (status, refused) = request(app.clone(), "POST", "/api/v1/image/jobs", json!({
                    "projectId":project_id,"model":other_parent,"prompt":"wrong checkpoint",
                    "loras":[{"id":entry["id"],"weight":1.0}],
                    "advanced":{"mossImage":{"acceleration":"off"}}
                })).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
                assert!(refused.to_string().contains("not interchangeable"), "{refused}");
                for acceleration in ["unmerged_bf16", "full_fp32"] {
                    // Namespace-only staging tests API selection/replay, NOT execution of
                    // published DMD weights. Actual DMD+custom acceptance belongs to S5.
                    if base == "moss_image_sft" {
                        let purpose = format!("moss_dmd_{acceleration}");
                        for download in model["conditionalComponents"].as_array().unwrap() {
                            if download["requiredFor"].as_array().unwrap().iter().any(|value| value == &purpose) {
                                super::yue2_catalog::seed_snapshot(&temp, download);
                            }
                        }
                    }
                    let (status, accelerated) = request(app.clone(), "POST", "/api/v1/image/jobs", json!({
                        "projectId":project_id,"model":base,"prompt":"custom adapter acceleration",
                        "loras":[{"id":entry["id"],"weight":1.0}],
                        "advanced":{"mossImage":{"acceleration":acceleration}}
                    })).await;
                    if base == "moss_image_base" {
                        assert_eq!(status, StatusCode::BAD_REQUEST, "{accelerated}");
                        assert!(accelerated.to_string().contains("DMD requires the exact SFT parent"), "{accelerated}");
                    } else {
                        assert_eq!(status, StatusCode::CREATED, "{accelerated}");
                        assert_eq!(accelerated["payload"]["advanced"]["mossImage"]["acceleration"], acceleration);
                        assert_eq!(accelerated["payload"]["loras"][0]["id"], entry["id"]);
                        assert_eq!(accelerated["payload"]["loras"][0]["weight"], 1.0);
                        let (status, replayed) = request(
                            app.clone(), "POST", &format!("/api/v1/jobs/{}/duplicate", accelerated["id"].as_str().unwrap()),
                            json!({"payloadChanges":{}}),
                        ).await;
                        assert_eq!(status, StatusCode::CREATED, "{replayed}");
                        assert_eq!(replayed["payload"]["advanced"]["mossImage"]["acceleration"], acceleration);
                        assert_eq!(replayed["payload"]["loras"][0]["id"], entry["id"]);
                        let selected_adapters = sceneworks_worker::test_support::prepare_selected_adapters(
                            &worker, replayed["payload"].as_object().unwrap(),
                        ).unwrap();
                        selected_adapters.ensure_unchanged().unwrap();
                        assert_eq!(selected_adapters.specs().len(), 1);
                        assert_eq!(std::fs::canonicalize(&selected_adapters.specs()[0].path).unwrap(), std::fs::canonicalize(&saved_path).unwrap());
                        assert_eq!(selected_adapters.specs()[0].scale, 1.0);
                    }
                }
                let prepared = sceneworks_worker::test_support::prepare_selected_adapters(
                    &worker, selected["payload"].as_object().unwrap(),
                ).unwrap();
                prepared.ensure_unchanged().unwrap();
                assert_eq!(prepared.specs().len(), 1);
                let adapter = prepared.specs()[0].clone();
                assert_eq!(std::fs::canonicalize(&adapter.path).unwrap(), std::fs::canonicalize(&saved_path).unwrap(), "the actual prepared resolver must select the declared final adapter, not an intermediate checkpoint");
                assert_eq!(adapter.kind, if method == "lora" { AdapterKind::Lora } else { AdapterKind::Lokr });
                assert_eq!(adapter.scale, 1.0);
                let generation = generation_request();
                diagnostic["phase"] = json!("before_generator_reload");
                diagnostic["selectedAdapter"] = json!({"path":adapter.path,"scale":adapter.scale,"kind":format!("{:?}",adapter.kind)});
                if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
                    diagnostic_errors.push(error);
                }
                let (before, before_shape, _) = fixture.generate_latents(variant, Vec::new(), &generation).unwrap();
                let (after, after_shape, reports) = fixture.generate_latents(variant, vec![adapter.clone()], &generation).unwrap();
                assert_eq!(before_shape, after_shape);
                assert_eq!(before.len(), after.len());
                assert!(!before.is_empty());
                assert!(before.iter().chain(&after).all(|value| value.is_finite()));
                assert!(before.iter().zip(&after).any(|(a, b)| a != b), "{base}/{method}: actual scale-1 generation must change after reload");
                assert_installed_adapter(&reports, &saved_path);
                let mut zero_adapter = adapter.clone();
                zero_adapter.scale = 0.0;
                let (zero, zero_shape, _) = fixture.generate_latents(variant, vec![zero_adapter], &generation).unwrap();
                assert_eq!(zero_shape, before_shape);
                assert_eq!(zero, before, "zero-strength negative control must reproduce the baseline");
                let generator = fixture.generator(variant, vec![adapter]).unwrap();
                let output = generator.generate(&generation, &mut |_| {}).unwrap();
                match output {
                    GenerationOutput::Images(images) => {
                        assert_eq!(images.len(), 1);
                        assert_eq!((images[0].width, images[0].height), (32, 32));
                        assert_eq!(images[0].pixels.len(), 32 * 32 * 3);
                    }
                    _ => panic!("actual MOSS Generator must return an image"),
                }
                assert_installed_adapter(&generator.adapter_apply_reports(), &saved_path);
                assert!(generator.moss_image_execution_report().is_none(), "synthetic fixtures must not mint published-author provenance");
                prepared.ensure_unchanged().unwrap();
    }).catch_unwind().await;
    diagnostic["phase"] = json!("case_body_returned");
    diagnostic["panicObserved"] = json!(outcome.is_err());
    if let Err(error) = emit_case_diagnostic(&diagnostic, probe.as_ref()) {
        diagnostic_errors.push(error);
    }
    let _ = stop.send(());
    let teardown_error = match tokio::time::timeout(Duration::from_secs(15), server).await {
        Ok(Ok(Ok(()))) => None,
        other => Some(format!("owned API teardown failed: {other:?}")),
    };
    if let Some(error) = &teardown_error {
        diagnostic["teardownError"] = json!(error);
        let _ = emit_case_diagnostic(&diagnostic, probe.as_ref());
    }
    // Preserve an original assertion, native error or timeout panic as the
    // primary failure even when diagnostic capture or teardown also failed.
    if let Err(error) = outcome {
        std::panic::resume_unwind(error);
    }
    assert!(teardown_error.is_none(), "{teardown_error:?}");
    assert!(
        diagnostic_errors.is_empty(),
        "diagnostic capture failed: {diagnostic_errors:?}"
    );
}
