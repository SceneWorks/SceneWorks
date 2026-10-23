#[cfg(test)]
mod dispatch_transaction_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Attempt {
        Success,
        Fail,
        CancelAfterOutput,
        MissingReport,
        InvalidReport,
        NoOutput,
        ApiCancel,
    }
    struct Control {
        budget: Mutex<gen_core::MemoryBudget>,
        attempt: Mutex<Attempt>,
        loads: AtomicUsize,
        cold: AtomicUsize,
        warm: AtomicUsize,
        renders: AtomicUsize,
        reads: AtomicUsize,
        ordinary: AtomicUsize,
        api_cancel: AtomicBool,
        posts: Mutex<Vec<Value>>,
        jobs: Mutex<std::collections::HashMap<String, Value>>,
    }
    impl Control {
        fn new() -> Self {
            Self {
                budget: Mutex::new(gen_core::MemoryBudget {
                    total_bytes: 100,
                    committed_bytes: 30,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: 10,
                }),
                attempt: Mutex::new(Attempt::Success),
                loads: 0.into(),
                cold: 0.into(),
                warm: 0.into(),
                renders: 0.into(),
                reads: 0.into(),
                ordinary: 0.into(),
                api_cancel: false.into(),
                posts: Mutex::new(Vec::new()),
                jobs: Mutex::new(Default::default()),
            }
        }
        fn committed(&self, bytes: u64) {
            self.budget.lock().unwrap().committed_bytes = bytes;
        }
    }
    struct FixtureRuntime {
        control: Arc<Control>,
        cache: crate::generator_cache::GeneratorCacheHandle,
    }
    impl MossStreamRuntime for FixtureRuntime {
        fn prepare(
            &self,
            input: MossPreparation<'_>,
            _: &mut dyn FnMut(usize, usize),
        ) -> WorkerResult<LoadSpec> {
            // This is explicitly a synthetic provider source, never a forged official frozen SHA.
            // NativeMossStreamRuntime still owns the unchanged official package/seal implementation.
            let selected = input.entry["downloads"].as_array().unwrap();
            assert_eq!(selected.len(), usize::from(input.edit));
            if input.edit {
                assert_eq!(selected[0]["repo"], "synthetic/moss-edit-dispatch-fixture");
            }
            let path = input.settings.data_dir.join("synthetic-provider.bin");
            let mut spec =
                LoadSpec::new(WeightsSource::File(path.clone())).with_resolved_route(format!(
                    "{}:{:?}:edit={}",
                    input.variant.provider_id(),
                    input.mode,
                    input.edit
                ));
            let pin = gen_core::PinnedWeightsFile::pin(&path)
                .map_err(|e| crate::classify_engine_error("synthetic source", e))?;
            crate::paths::prepare_load_spec_with_file_pins(
                &mut spec,
                vec![pin],
                "synthetic source",
            )?;
            Ok(spec)
        }
        fn cold_peak(&self, input: MossColdRequest<'_>) -> WorkerResult<u64> {
            input.spec.validate_prepared_file_pins().unwrap();
            assert_eq!(input.resolved.acceleration, MossImageAcceleration::Off);
            assert_eq!(input.total_count, 1);
            if input.edit {
                assert_eq!(
                    input.advanced.negative_prompts.unwrap(),
                    &["fixture negative"]
                );
                assert_eq!(input.advanced.output, MossImageOutputKind::Images);
                input
                    .advanced
                    .validate(input.native, input.resolved)
                    .unwrap();
                let request = gen_core::moss_image_edit::edit_request(input.native).unwrap();
                assert_eq!(
                    request.moss_image.unwrap().acceleration,
                    MossImageAcceleration::Off
                );
                assert_eq!(input.references.len(), 1);
                assert_eq!(
                    (input.references[0].width, input.references[0].height),
                    (48, 16)
                );
                assert_eq!(input.references[0].pixels, vec![73; 48 * 16 * 3]);
                assert_eq!(input.edit_options.reference_max_edge, 32);
                assert!(!input.edit_options.use_kv_cache);
            } else {
                assert!(input.references.is_empty());
            }
            self.control.cold.fetch_add(1, Ordering::SeqCst);
            Ok(73)
        }
        fn budget(&self, _: &Settings) -> WorkerResult<gen_core::MemoryBudget> {
            Ok(*self.control.budget.lock().unwrap())
        }
        fn cache(&self) -> crate::generator_cache::GeneratorCacheHandle {
            self.cache.clone()
        }
    }
    struct TransactionProbe {
        inner: Probe,
        control: Arc<Control>,
    }
    impl Generator for TransactionProbe {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            self.inner.descriptor()
        }
        fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
            self.inner.validate(request)
        }
        fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
            Some(20)
        }
        fn moss_image_edit_request_peak(
            &self,
            request: &GenerationRequest,
            inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
        ) -> gen_core::Result<u64> {
            assert_eq!(
                inputs.advanced.negative_prompts.unwrap(),
                &["fixture negative"]
            );
            self.control.warm.fetch_add(1, Ordering::SeqCst);
            self.inner.moss_image_edit_request_peak(request, inputs)
        }
        fn generate(
            &self,
            _: &GenerationRequest,
            _: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            self.control.ordinary.fetch_add(1, Ordering::SeqCst);
            Err(gen_core::Error::Unsupported(
                "ordinary fallback must never serve this edit".into(),
            ))
        }
        fn generate_moss_image_edit(
            &self,
            request: &GenerationRequest,
            inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
            progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<MossImageAdvancedOutput> {
            assert_eq!(
                inputs.advanced.negative_prompts.unwrap(),
                &["fixture negative"]
            );
            self.control.renders.fetch_add(1, Ordering::SeqCst);
            let attempt = *self.control.attempt.lock().unwrap();
            self.inner.failure.set(attempt == Attempt::Fail);
            self.inner.cancel.set(attempt == Attempt::CancelAfterOutput);
            let mut output = self
                .inner
                .generate_moss_image_edit(request, inputs, progress)?;
            match attempt {
                Attempt::MissingReport => *self.inner.edit_report.borrow_mut() = None,
                Attempt::InvalidReport => {
                    self.inner
                        .edit_report
                        .borrow_mut()
                        .as_mut()
                        .unwrap()
                        .applied_projections = 0
                }
                Attempt::NoOutput => output = MossImageAdvancedOutput::Images(vec![]),
                Attempt::ApiCancel => {
                    // Exercise consume_gen_events' real cancel poll and acknowledgement, rather than
                    // manually canceling its token. Its existing poll throttle is two seconds.
                    std::thread::sleep(Duration::from_millis(2100));
                    progress(Progress::Step {
                        current: 1,
                        total: 1,
                    });
                    let until = Instant::now() + Duration::from_secs(3);
                    while !request.cancel.is_cancelled() && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    assert!(
                        request.cancel.is_cancelled(),
                        "actual event consumer must cancel the request"
                    );
                }
                _ => {}
            }
            Ok(output)
        }
        fn moss_image_edit_execution_report(
            &self,
        ) -> Option<gen_core::moss_image_edit::MossImageEditExecutionReport> {
            self.control.reads.fetch_add(1, Ordering::SeqCst);
            self.inner.moss_image_edit_execution_report()
        }
    }

    async fn transaction_api(control: Arc<Control>) -> (String, tokio::task::JoinHandle<()>) {
        use axum::{
            extract::{Path as RoutePath, State},
            routing::{get, post},
            Json, Router,
        };
        async fn job(
            State(c): State<Arc<Control>>,
            RoutePath(id): RoutePath<String>,
        ) -> Json<Value> {
            Json(crate::tests::job_snapshot_json(
                &id,
                c.api_cancel.load(Ordering::SeqCst),
            ))
        }
        async fn progress(
            State(c): State<Arc<Control>>,
            RoutePath(id): RoutePath<String>,
            Json(body): Json<Value>,
        ) -> Json<Value> {
            c.posts.lock().unwrap().push(json!({"job":id,"body":body}));
            let mut saved =
                crate::tests::job_snapshot_json(&id, c.api_cancel.load(Ordering::SeqCst));
            let payload = c.jobs.lock().unwrap().get(&id).cloned().unwrap();
            saved["payload"] = payload.clone();
            saved["projectId"] = payload["projectId"].clone();
            if body["status"] == "completed" {
                saved["status"] = json!("completed");
                saved["result"] = body["result"].clone();
                let generation = body["result"]["generationSetId"].clone();
                let facts = body["result"]["assetWrites"].as_array().unwrap();
                saved["result"]["assetIds"] = json!(facts
                    .iter()
                    .map(|f| f["assetId"].clone())
                    .collect::<Vec<_>>());
                saved["result"]["assets"] = json!(facts.iter().map(|f| json!({
                    "id":f["assetId"], "projectId":payload["projectId"], "generationSetId":generation,
                    "file":{"path":f["mediaPath"]}, "lineage":{"jobId":id}
                })).collect::<Vec<_>>());
            }
            Json(saved)
        }
        async fn heartbeat(RoutePath(id): RoutePath<String>) -> Json<Value> {
            Json(
                json!({"id": id, "gpuId":"cpu", "gpuName":null, "status":"busy",
                "currentJobId":"moss-dispatch", "capabilities":[], "loadedModels":[],
                "registeredAt":"2026-07-01T00:00:00Z", "lastSeenAt":"2026-07-01T00:00:00Z"}),
            )
        }
        let app = Router::new()
            .route("/api/v1/jobs/:job", get(job))
            .route("/api/v1/jobs/:job/progress", post(progress))
            .route("/api/v1/workers/:worker/heartbeat", post(heartbeat))
            .with_state(control);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, task)
    }
    fn image_files(root: &Path) -> std::collections::BTreeSet<PathBuf> {
        let mut result = std::collections::BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    result.extend(image_files(&path));
                } else {
                    result.insert(path);
                }
            }
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_attempt(
        settings: &Settings,
        project: &str,
        project_path: &Path,
        reference: &str,
        variant: MossImageVariant,
        edit: bool,
        runtime: Arc<FixtureRuntime>,
        control: Arc<Control>,
    ) -> (WorkerResult<()>, Vec<Value>, String) {
        let model = variant.provider_id();
        let manifest = json!({"limits":{"minDimension":16},"conditionalComponents":[{
        "provider":"huggingface","repo":"synthetic/moss-edit-dispatch-fixture","revision":"synthetic",
        "componentId":"moss-image-edit","requiredFor":["moss_image_edit"],"files":["fixture.bin"]}]});
        let mut payload = json!({"projectId":project,"model":model,"mode":if edit {"edit_image"} else {"text_to_image"},
        "prompt":"blue cup","width":16,"height":16,"count":1,"modelManifestEntry":manifest,
        "advanced":{"mossImage":{"acceleration":"off"}}});
        if edit {
            payload["referenceAssetIds"] = json!([reference]);
            payload["advanced"]["mossImageAdvanced"] =
                json!({"negativePrompts":["fixture negative"]});
            payload["advanced"]["mossImageEdit"] =
                json!({"reference_max_edge":32,"use_kv_cache":false});
        }
        let id = format!("moss-dispatch-{}", Uuid::new_v4());
        control
            .jobs
            .lock()
            .unwrap()
            .insert(id.clone(), payload.clone());
        let mut job = crate::tests::job_snapshot_json(&id, false);
        job["projectId"] = json!(project);
        job["payload"] = payload.clone();
        let job: JobSnapshot = serde_json::from_value(job).unwrap();
        let request = ImageRequest::from_payload(payload.as_object().unwrap());
        let plan = ImagePlan::with_count_and_adapter(
            &request,
            1,
            moss_adapter_label(model, "candle"),
            Some(Arc::new(job.payload.clone())),
        );
        let mut writes = vec![];
        let load = move |engine: &str, spec: &LoadSpec| -> gen_core::Result<Box<dyn Generator>> {
            assert_eq!(engine, model);
            spec.validate_prepared_file_pins()?;
            control.loads.fetch_add(1, Ordering::SeqCst);
            let mut inner = Probe::new();
            inner.descriptor.id = model;
            inner.descriptor.backend = "candle";
            Ok(Box::new(TransactionProbe { inner, control }))
        };
        let result = tokio::time::timeout(
            Duration::from_secs(15),
            generate_candle_stream_using(
                &ApiClient::new(settings),
                settings,
                &job,
                &plan,
                project_path,
                "cuda:fixture",
                &mut writes,
                load,
                runtime,
            ),
        )
        .await
        .expect("bounded actual dispatch/cache/event transaction");
        // Mirror the actual outer worker boundary: staged edits publish only after the native
        // task/event consumer succeeded; the shared publication helper owns the acknowledgment.
        let batch = { plan.moss_staged.lock().unwrap().take() };
        let result = match (result, batch) {
            (Ok(()), Some(batch)) => {
                publish_moss_batch(
                    &ApiClient::new(settings),
                    &job,
                    &plan,
                    "candle",
                    &writes,
                    batch,
                )
                .await
            }
            (result, batch) => {
                drop(batch);
                result
            }
        };
        (result, writes, id)
    }

    /// Removing the MOSS diversion from generate_candle_stream_using makes this fail on both hosts:
    /// the generic fallback cannot produce the typed edit receipt, cache hit and real streamed PNG.
    async fn exercise_candle_moss_dispatch(variant: MossImageVariant) {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("private-hf");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", hub.to_str().unwrap()),
            ("HUGGINGFACE_HUB_CACHE", hub.to_str().unwrap()),
        ]);
        {
            let control = Arc::new(Control::new());
            let (url, server) = transaction_api(control.clone()).await;
            let mut settings = Settings::for_test(temp.path().join(variant.provider_id()));
            settings.api_url = url;
            settings.access_token = None;
            settings.backend_candle_enabled = true;
            let store = ProjectStore::new(settings.data_dir.clone(), "test");
            let project = store.create_project("Candle MOSS dispatch").unwrap();
            let source = settings.data_dir.join("reference.png");
            image::RgbImage::from_pixel(48, 16, image::Rgb([73; 3]))
                .save(&source)
                .unwrap();
            let reference = store
                .import_asset(
                    &project.id,
                    sceneworks_core::project_store::UploadAsset {
                        filename: "reference.png".into(),
                        content_type: Some("image/png".into()),
                        source_path: source,
                        source_asset_id: None,
                        provenance: None,
                    },
                )
                .unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned();
            let synthetic = settings.data_dir.join("synthetic-provider.bin");
            std::fs::write(&synthetic, b"synthetic-source-a").unwrap();
            let (cache, join) = crate::generator_cache::GeneratorCacheHandle::isolated();
            let runtime = Arc::new(FixtureRuntime {
                control: control.clone(),
                cache: cache.clone(),
            });
            let attempt = |edit| {
                dispatch_attempt(
                    &settings,
                    &project.id,
                    Path::new(&project.path),
                    &reference,
                    variant,
                    edit,
                    runtime.clone(),
                    control.clone(),
                )
            };

            let (result, writes, _) = attempt(true).await;
            assert!(result.unwrap_err().to_string().contains("additional bytes"));
            assert!(writes.is_empty());
            assert_eq!(control.cold.load(Ordering::SeqCst), 1);
            assert_eq!(
                control.loads.load(Ordering::SeqCst),
                0,
                "cold typed estimate must refuse BEFORE loader"
            );
            control.committed(0);
            let (result, writes, job) = attempt(true).await;
            result.unwrap();
            assert_eq!(writes.len(), 1);
            assert_eq!(control.loads.load(Ordering::SeqCst), 1);
            assert_eq!(
                writes[0]["adapter"],
                moss_adapter_label(variant.provider_id(), "candle")
            );
            let report =
                &writes[0]["rawAdapterSettings"][sceneworks_core::moss_image::EDIT_REPORT_KEY];
            assert_eq!(
                report["references"][0]["original_dimensions"],
                json!([48, 16])
            );
            assert_eq!(report["reference_max_edge"], 32);
            assert_eq!(report["use_kv_cache"], false);
            assert_eq!(report["base"]["samples"][0]["seed"], 173);
            assert_eq!(
                writes[0]["rawAdapterSettings"]["mossImageAdvanced"]["negativePrompts"],
                json!(["fixture negative"])
            );
            assert_eq!(report["base"]["conditioning"][0]["negative"]["sha256"], {
                use sha2::{Digest, Sha256};
                format!("{:x}", Sha256::digest(b"fixture negative"))
            });
            assert!(Path::new(&project.path)
                .join(writes[0]["mediaPath"].as_str().unwrap())
                .is_file());
            assert!(control.posts.lock().unwrap().iter().any(|p| p["job"] == job
                && p["body"]["backend"] == "candle"
                && p["body"]["result"]["assetWrites"]
                    .as_array()
                    .is_some_and(|v| v.len() == 1)));

            // This peak cannot fit cold (73 > 60), but the same live cache entry owns 20 bytes.
            control.committed(30);
            let (result, writes, _) = attempt(true).await;
            result.unwrap();
            assert_eq!(writes.len(), 1);
            assert_eq!(
                control.loads.load(Ordering::SeqCst),
                1,
                "second dispatch must hit the actual matching cache"
            );
            assert_eq!(control.warm.load(Ordering::SeqCst), 2);
            for outcome in [
                Attempt::Fail,
                Attempt::CancelAfterOutput,
                Attempt::MissingReport,
                Attempt::InvalidReport,
                Attempt::NoOutput,
                Attempt::ApiCancel,
            ] {
                *control.attempt.lock().unwrap() = outcome;
                control
                    .api_cancel
                    .store(outcome == Attempt::ApiCancel, Ordering::SeqCst);
                let before = image_files(&Path::new(&project.path).join("assets/images"));
                let reads = control.reads.load(Ordering::SeqCst);
                let (result, writes, id) = attempt(true).await;
                assert!(result.is_err(), "{variant:?} {outcome:?} must fail");
                assert!(
                    writes.is_empty(),
                    "{outcome:?} must not publish an asset fact"
                );
                assert_eq!(
                    image_files(&Path::new(&project.path).join("assets/images")),
                    before
                );
                assert!(!control.posts.lock().unwrap().iter().any(|p| p["job"] == id
                    && p["body"]["result"]["assetWrites"]
                        .as_array()
                        .is_some_and(|v| !v.is_empty())));
                if matches!(
                    outcome,
                    Attempt::Fail | Attempt::CancelAfterOutput | Attempt::ApiCancel
                ) {
                    assert_eq!(
                        control.reads.load(Ordering::SeqCst),
                        reads,
                        "failed/canceled attempts cannot read stale receipts"
                    );
                }
                if outcome == Attempt::ApiCancel {
                    assert!(matches!(result, Err(WorkerError::Canceled(_))));
                    assert!(control
                        .posts
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|p| p["job"] == id && p["body"]["status"] == "canceled"));
                }
                control.api_cancel.store(false, Ordering::SeqCst);
                *control.attempt.lock().unwrap() = Attempt::Success;
                let (result, writes, _) = attempt(true).await;
                result.unwrap();
                assert_eq!(writes.len(), 1);
                assert_eq!(
                    control.loads.load(Ordering::SeqCst),
                    1,
                    "clean reentry uses this same cached provider"
                );
            }
            std::fs::write(&synthetic, b"synthetic-source-b").unwrap();
            let (result, writes, _) = attempt(true).await;
            assert!(result.is_err());
            assert!(writes.is_empty());
            assert_eq!(
                control.loads.load(Ordering::SeqCst),
                1,
                "a different pinned source cannot borrow old ownership"
            );
            control.committed(0);
            attempt(true).await.0.unwrap();
            assert_eq!(control.loads.load(Ordering::SeqCst), 2);
            control.committed(30);
            let (result, writes, _) = attempt(false).await;
            assert!(result.is_err());
            assert!(writes.is_empty());
            assert_eq!(
                control.loads.load(Ordering::SeqCst),
                2,
                "ordinary mode cannot borrow edit ownership"
            );
            assert_eq!(control.ordinary.load(Ordering::SeqCst), 0);
            control.committed(0);
            attempt(true).await.0.unwrap();
            assert_eq!(control.loads.load(Ordering::SeqCst), 3);
            drop(runtime);
            drop(cache);
            join.join().unwrap();
            server.abort();
            let _ = server.await;
        }
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candle_moss_base_dispatch_runs_real_cache_admission_events_and_reentry() {
        exercise_candle_moss_dispatch(MossImageVariant::Base).await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candle_moss_sft_dispatch_runs_real_cache_admission_events_and_reentry() {
        exercise_candle_moss_dispatch(MossImageVariant::Sft).await;
    }
}
