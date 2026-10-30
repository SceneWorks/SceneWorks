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
        WrongReferenceReceipt,
        NoOutput,
        ApiCancel,
    }
    struct Control {
        backend: &'static str,
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
        expected_references: Mutex<Vec<(u32, u32, Vec<u8>)>>,
        use_kv_cache: AtomicBool,
        recording_request: Mutex<Option<Value>>,
    }
    impl Control {
        fn new() -> Self {
            Self {
                backend: "candle",
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
                expected_references: Mutex::new(vec![(48, 16, vec![73; 48 * 16 * 3])]),
                use_kv_cache: false.into(),
                recording_request: Mutex::new(None),
            }
        }
        fn committed(&self, bytes: u64) {
            self.budget.lock().unwrap().committed_bytes = bytes;
        }
        fn check_recording_request(
            &self,
            request: &GenerationRequest,
            inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
            render: bool,
        ) -> gen_core::Result<bool> {
            let expected = self.recording_request.lock().unwrap();
            let Some(expected) = expected.as_ref() else {
                return Ok(false);
            };
            let actual = inputs
                .references
                .iter()
                .map(|image| (image.width, image.height, image.pixels.clone()))
                .collect::<Vec<_>>();
            if actual != *self.expected_references.lock().unwrap() {
                return Err(gen_core::Error::Unsupported(
                    "selected water-bottle reference bytes changed".into(),
                ));
            }
            assert_eq!(request.prompt, expected["prompt"].as_str().unwrap());
            assert_eq!(
                json!([request.width, request.height, request.count]),
                json!([expected["width"], expected["height"], expected["count"]])
            );
            assert_eq!(
                request.steps.map(u64::from),
                expected["advanced"]["steps"].as_u64()
            );
            assert_eq!(
                request.guidance.map(f64::from),
                expected["advanced"]["guidanceScale"].as_f64()
            );
            assert_eq!(
                request.scheduler_shift.map(f64::from),
                expected["advanced"]["schedulerShift"].as_f64()
            );
            let options: gen_core::moss_image::MossImageOptions =
                serde_json::from_value(expected["advanced"]["mossImage"].clone()).unwrap();
            assert_eq!(
                request.moss_image.as_ref().unwrap(),
                &options,
                "all authored native options"
            );
            assert_eq!(
                json!({"reference_max_edge":inputs.reference_max_edge,"use_kv_cache":inputs.use_kv_cache}),
                expected["advanced"]["mossImageEdit"]
            );
            assert!(
                inputs.advanced.positive.is_none()
                    && inputs.advanced.negative.is_none()
                    && inputs.advanced.noise.is_none()
                    && inputs.advanced.negative_prompts.is_none()
            );
            assert_eq!(inputs.advanced.output, MossImageOutputKind::Images);
            if render {
                assert_eq!(request.seed, expected["seed"].as_u64());
            }
            Ok(true)
        }
        fn assert_references(&self, references: &[Image], use_kv_cache: bool) {
            assert_eq!(
                references
                    .iter()
                    .map(|image| (image.width, image.height, image.pixels.clone()))
                    .collect::<Vec<_>>(),
                *self.expected_references.lock().unwrap(),
                "all decoded semantic RGB slots retain order and identity"
            );
            assert_eq!(use_kv_cache, self.use_kv_cache.load(Ordering::SeqCst));
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
            if self
                .control
                .check_recording_request(
                    input.native,
                    gen_core::moss_image_edit::MossImageEditInputs {
                        references: input.references,
                        reference_max_edge: input.edit_options.reference_max_edge,
                        use_kv_cache: input.edit_options.use_kv_cache,
                        advanced: input.advanced,
                    },
                    false,
                )
                .map_err(|e| crate::classify_engine_error("recording reference admission", e))?
            {
                self.control.cold.fetch_add(1, Ordering::SeqCst);
                return Ok(73);
            }
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
                let request = gen_core_moss_image::moss_image_edit::edit_request(input.native).unwrap();
                assert_eq!(
                    request.moss_image.unwrap().acceleration,
                    MossImageAcceleration::Off
                );
                self.control
                    .assert_references(input.references, input.edit_options.use_kv_cache);
                assert_eq!(input.edit_options.reference_max_edge, 32);
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
            if self
                .control
                .check_recording_request(request, inputs, false)?
            {
                self.control.warm.fetch_add(1, Ordering::SeqCst);
                return Ok(73);
            }
            assert_eq!(
                inputs.advanced.negative_prompts.unwrap(),
                &["fixture negative"]
            );
            self.control.warm.fetch_add(1, Ordering::SeqCst);
            self.control
                .assert_references(inputs.references, inputs.use_kv_cache);
            *self.inner.expected_edit_inputs.borrow_mut() =
                self.control.expected_references.lock().unwrap().clone();
            self.inner.expected_edit_cache.set(inputs.use_kv_cache);
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
            if self
                .control
                .check_recording_request(request, inputs, true)?
            {
                self.control.renders.fetch_add(1, Ordering::SeqCst);
                return Err(gen_core::Error::Unsupported(
                    "recorded native edit boundary; no inference executed".into(),
                ));
            }
            assert_eq!(
                inputs.advanced.negative_prompts.unwrap(),
                &["fixture negative"]
            );
            self.control.renders.fetch_add(1, Ordering::SeqCst);
            self.control
                .assert_references(inputs.references, inputs.use_kv_cache);
            let attempt = *self.control.attempt.lock().unwrap();
            self.inner.failure.set(attempt == Attempt::Fail);
            self.inner.cancel.set(attempt == Attempt::CancelAfterOutput);
            let mut output = self
                .inner
                .generate_moss_image_edit(request, inputs, progress)?;
            match attempt {
                Attempt::MissingReport => *self.inner.edit_report.borrow_mut() = None,
                Attempt::WrongReferenceReceipt => {
                    self.inner
                        .edit_report
                        .borrow_mut()
                        .as_mut()
                        .unwrap()
                        .references[0]
                        .original_dimensions[0] += 1
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
        references: &[String],
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
            payload["referenceAssetIds"] = json!(references);
            payload["advanced"]["mossImageAdvanced"] =
                json!({"negativePrompts":["fixture negative"]});
            payload["advanced"]["mossImageEdit"] = json!({"reference_max_edge":32,"use_kv_cache":control.use_kv_cache.load(Ordering::SeqCst)});
        }
        dispatch_payload(settings, project_path, payload, variant, runtime, control).await
    }

    async fn dispatch_payload(
        settings: &Settings,
        project_path: &Path,
        payload: Value,
        variant: MossImageVariant,
        runtime: Arc<FixtureRuntime>,
        control: Arc<Control>,
    ) -> (WorkerResult<()>, Vec<Value>, String) {
        let model = variant.provider_id();
        let backend = control.backend;
        let id = format!("moss-dispatch-{}", Uuid::new_v4());
        control
            .jobs
            .lock()
            .unwrap()
            .insert(id.clone(), payload.clone());
        let mut job = crate::tests::job_snapshot_json(&id, false);
        job["projectId"] = payload["projectId"].clone();
        job["payload"] = payload.clone();
        let job: JobSnapshot = serde_json::from_value(job).unwrap();
        let mut request = ImageRequest::from_payload(payload.as_object().unwrap());
        if request.model_manifest_entry.is_empty() {
            // Test-owned source metadata stays separate from the shared submitted request.
            request.model_manifest_entry = json!({"limits":{"minDimension":16},"conditionalComponents":[{
                "provider":"huggingface","repo":"synthetic/moss-edit-dispatch-fixture","revision":"synthetic",
                "componentId":"moss-image-edit","requiredFor":["moss_image_edit"],"files":["fixture.bin"]
            }]}).as_object().unwrap().clone();
        }
        let plan = ImagePlan::with_count_and_adapter(
            &request,
            1,
            moss_adapter_label(model, backend),
            Some(Arc::new(job.payload.clone())),
        );
        let mut writes = vec![];
        let load = move |engine: &str, spec: &LoadSpec| -> gen_core::Result<Box<dyn Generator>> {
            assert_eq!(engine, model);
            spec.validate_prepared_file_pins()?;
            control.loads.fetch_add(1, Ordering::SeqCst);
            let mut inner = Probe::new();
            inner.descriptor.id = model;
            inner.descriptor.backend = backend;
            Ok(Box::new(TransactionProbe { inner, control }))
        };
        let api = ApiClient::new(settings);
        let result = tokio::time::timeout(Duration::from_secs(15), async {
            if backend == "candle" {
                generate_candle_stream_using(
                    &api,
                    settings,
                    &job,
                    &plan,
                    project_path,
                    "cuda:fixture",
                    &mut writes,
                    load,
                    runtime,
                )
                .await
            } else {
                // This is the shared MLX transaction with an injected CPU provider, not Metal math.
                generate_moss_stream_with_runtime(
                    &api,
                    settings,
                    &job,
                    &plan,
                    project_path,
                    &mut writes,
                    load,
                    runtime,
                    backend,
                )
                .await
            }
        })
        .await
        .expect("bounded actual dispatch/cache/event transaction");
        (result, writes, id)
    }

    /// Linked to the UI/API request fixture; this checks the actual worker decode and native edit
    /// dispatch boundary. The provider records inputs and returns a sentinel before any inference.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn water_bottle_request_reaches_native_edit_with_exact_selected_photo() {
        use sha2::{Digest, Sha256};
        const PHOTO: &[u8] = include_bytes!("../../../../tests/fixtures/images/water-bottle.png");
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/water-bottle-edit-request.json"
        ))
        .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(PHOTO)),
            "c6ec194e6d7e04a2893449e530cf51d064c6adfd9847f0b028921b21e1c1bcb5"
        );
        let expected_photo = image::load_from_memory(PHOTO).unwrap().to_rgb8();
        let temp = tempfile::tempdir().unwrap();
        let control = Arc::new(Control {
            backend: "mlx",
            ..Control::new()
        });
        control.committed(0);
        *control.recording_request.lock().unwrap() = Some(fixture.clone());
        *control.expected_references.lock().unwrap() = vec![(
            expected_photo.width(),
            expected_photo.height(),
            expected_photo.into_raw(),
        )];
        control.use_kv_cache.store(true, Ordering::SeqCst);
        let (url, server) = transaction_api(control.clone()).await;
        let mut settings = Settings::for_test(temp.path().join("worker"));
        settings.api_url = url;
        settings.access_token = None;
        let store = ProjectStore::new(settings.data_dir.clone(), "test");
        let project = store
            .create_project("Water bottle linked contract")
            .unwrap();
        let source = settings.data_dir.join("water-bottle.png");
        std::fs::write(&source, PHOTO).unwrap();
        let asset = store
            .import_asset(
                &project.id,
                sceneworks_core::project_store::UploadAsset {
                    filename: "water-bottle.png".into(),
                    content_type: Some("image/png".into()),
                    source_path: source,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .unwrap();
        let mut payload = fixture.clone();
        payload["projectId"] = json!(project.id);
        payload["referenceAssetIds"] = json!([asset["id"]]);
        let mut normalized = payload.clone();
        normalized.as_object_mut().unwrap().remove("projectId");
        normalized["referenceAssetIds"] = fixture["referenceAssetIds"].clone();
        assert_eq!(
            normalized, fixture,
            "only owned project/asset identifiers are substituted"
        );
        std::fs::write(
            settings.data_dir.join("synthetic-provider.bin"),
            b"recording-provider-source",
        )
        .unwrap();
        let (cache, join) = crate::generator_cache::GeneratorCacheHandle::isolated();
        let runtime = Arc::new(FixtureRuntime {
            control: control.clone(),
            cache: cache.clone(),
        });
        let run = |payload| {
            dispatch_payload(
                &settings,
                Path::new(&project.path),
                payload,
                MossImageVariant::Sft,
                runtime.clone(),
                control.clone(),
            )
        };
        let (result, writes, _) = run(payload.clone()).await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("recorded native edit boundary; no inference executed"));
        assert!(writes.is_empty());
        assert_eq!(control.renders.load(Ordering::SeqCst), 1);
        assert_eq!(control.ordinary.load(Ordering::SeqCst), 0);
        assert_eq!(
            control.reads.load(Ordering::SeqCst),
            0,
            "recording does not fabricate a successful receipt"
        );
        let owned = store
            .project_file(&project.id, asset["file"]["path"].as_str().unwrap())
            .unwrap()
            .path;
        let mut changed = image::open(&owned).unwrap().to_rgb8();
        changed.get_pixel_mut(0, 0).0[0] ^= 1;
        changed.save(&owned).unwrap();
        let (result, writes, _) = run(payload.clone()).await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("selected water-bottle reference bytes changed"));
        assert!(writes.is_empty());
        payload["referenceAssetIds"] = json!([]);
        let (result, writes, _) = run(payload).await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("one to four ordered reference slots"));
        assert!(writes.is_empty());
        assert_eq!(
            control.renders.load(Ordering::SeqCst),
            1,
            "changed/empty inputs refuse before native edit"
        );
        assert_eq!(control.ordinary.load(Ordering::SeqCst), 0);
        drop(runtime);
        drop(cache);
        join.join().unwrap();
        server.abort();
        let _ = server.await;
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
                    std::slice::from_ref(&reference),
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
                Attempt::WrongReferenceReceipt,
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

    async fn exercise_multi_reference_dispatch(variant: MossImageVariant, backend: &'static str) {
        use sha2::Digest;
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("private-hf");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", hub.to_str().unwrap()),
            ("HUGGINGFACE_HUB_CACHE", hub.to_str().unwrap()),
        ]);
        let control = Arc::new(Control {
            backend,
            ..Control::new()
        });
        let (url, server) = transaction_api(control.clone()).await;
        let mut settings = Settings::for_test(temp.path().join(variant.provider_id()));
        settings.api_url = url;
        settings.access_token = None;
        settings.backend_candle_enabled = true;
        let store = ProjectStore::new(settings.data_dir.clone(), "test");
        let project = store.create_project("MOSS ordered slots").unwrap();
        let project_path = Path::new(&project.path);
        let mut ids = Vec::new();
        let mut images = Vec::new();
        for (index, (width, height, pixel)) in [(48, 16, 73), (16, 48, 127), (32, 32, 211)]
            .into_iter()
            .enumerate()
        {
            let source = settings.data_dir.join(format!("reference-{index}.png"));
            image::RgbImage::from_pixel(width, height, image::Rgb([pixel; 3]))
                .save(&source)
                .unwrap();
            ids.push(
                store
                    .import_asset(
                        &project.id,
                        sceneworks_core::project_store::UploadAsset {
                            filename: format!("reference-{index}.png"),
                            content_type: Some("image/png".into()),
                            source_path: source,
                            source_asset_id: None,
                            provenance: None,
                        },
                    )
                    .unwrap()["id"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            );
            images.push((width, height, vec![pixel; (width * height * 3) as usize]));
        }
        std::fs::write(
            settings.data_dir.join("synthetic-provider.bin"),
            b"synthetic-source-a",
        )
        .unwrap();
        let (cache, join) = crate::generator_cache::GeneratorCacheHandle::isolated();
        let runtime = Arc::new(FixtureRuntime {
            control: control.clone(),
            cache: cache.clone(),
        });
        // Real asset decode, ordered full-slice dispatch and provider-cache transport only.
        // The injected 73-byte estimate is deliberately synthetic; native memory math is not tested here.
        for use_kv_cache in [false, true] {
            control.use_kv_cache.store(use_kv_cache, Ordering::SeqCst);
            for ordinals in [
                vec![0, 1],
                vec![1, 0],
                vec![0, 1, 0],
                vec![1, 0, 0],
                vec![0, 1, 0, 2],
                vec![2, 0, 1, 0],
                vec![1, 2, 1],
                vec![2],
            ] {
                let references = ordinals.iter().map(|&i| ids[i].clone()).collect::<Vec<_>>();
                let expected = ordinals
                    .iter()
                    .map(|&i| images[i].clone())
                    .collect::<Vec<_>>();
                *control.expected_references.lock().unwrap() = expected.clone();
                if control.loads.load(Ordering::SeqCst) == 0 {
                    let (result, writes, _) = dispatch_attempt(
                        &settings,
                        &project.id,
                        project_path,
                        &references,
                        variant,
                        true,
                        runtime.clone(),
                        control.clone(),
                    )
                    .await;
                    assert!(result.unwrap_err().to_string().contains("additional bytes"));
                    assert!(writes.is_empty());
                    assert_eq!(
                        control.loads.load(Ordering::SeqCst),
                        0,
                        "cold full-slice refusal precedes factory"
                    );
                    control.committed(0);
                }
                let (result, writes, _) = dispatch_attempt(
                    &settings,
                    &project.id,
                    project_path,
                    &references,
                    variant,
                    true,
                    runtime.clone(),
                    control.clone(),
                )
                .await;
                result.unwrap();
                assert_eq!(writes.len(), 1);
                assert_eq!(
                    control.loads.load(Ordering::SeqCst),
                    1,
                    "changed slots/cache option reuse owned weights"
                );
                assert_eq!(
                    writes[0]["adapter"],
                    moss_adapter_label(variant.provider_id(), backend)
                );
                let report =
                    &writes[0]["rawAdapterSettings"][sceneworks_core::moss_image::EDIT_REPORT_KEY];
                assert_eq!(report["use_kv_cache"], use_kv_cache);
                let receipts = report["references"].as_array().unwrap();
                assert_eq!(receipts.len(), expected.len());
                for (receipt, (width, height, pixels)) in receipts.iter().zip(&expected) {
                    assert_eq!(receipt["original_dimensions"], json!([width, height]));
                    assert_eq!(
                        receipt["pixels_sha256"],
                        format!("{:x}", sha2::Sha256::digest(pixels))
                    );
                }
                assert!(project_path
                    .join(writes[0]["mediaPath"].as_str().unwrap())
                    .is_file());
                control.committed(30); // Warm credit belongs only to the unchanged owned weights.
                for outcome in [Attempt::Fail, Attempt::CancelAfterOutput] {
                    *control.attempt.lock().unwrap() = outcome;
                    let files = image_files(&project_path.join("assets/images"));
                    let reads = control.reads.load(Ordering::SeqCst);
                    let (result, writes, id) = dispatch_attempt(
                        &settings,
                        &project.id,
                        project_path,
                        &references,
                        variant,
                        true,
                        runtime.clone(),
                        control.clone(),
                    )
                    .await;
                    assert!(result.is_err());
                    assert!(writes.is_empty());
                    assert_eq!(
                        control.reads.load(Ordering::SeqCst),
                        reads,
                        "failure/cancel cannot read prior slot receipts"
                    );
                    assert_eq!(image_files(&project_path.join("assets/images")), files);
                    assert!(!control.posts.lock().unwrap().iter().any(|p| p["job"] == id
                        && p["body"]["result"]["assetWrites"]
                            .as_array()
                            .is_some_and(|v| !v.is_empty())));
                }
                *control.attempt.lock().unwrap() = Attempt::Success;
                // The next success changes the slot list on the same live provider after these errors.
            }
        }
        // Last error/cancel must also permit a clean, changed two-slot reentry.
        let references = vec![ids[2].clone(), ids[1].clone()];
        *control.expected_references.lock().unwrap() = vec![images[2].clone(), images[1].clone()];
        let (result, writes, _) = dispatch_attempt(
            &settings,
            &project.id,
            project_path,
            &references,
            variant,
            true,
            runtime.clone(),
            control.clone(),
        )
        .await;
        result.unwrap();
        assert_eq!(
            writes[0]["rawAdapterSettings"][sceneworks_core::moss_image::EDIT_REPORT_KEY]
                ["references"][0]["original_dimensions"],
            json!([32, 32])
        );
        assert_eq!(control.loads.load(Ordering::SeqCst), 1);
        let before = (
            control.cold.load(Ordering::SeqCst),
            control.warm.load(Ordering::SeqCst),
            control.renders.load(Ordering::SeqCst),
        );
        for invalid in [
            vec![],
            vec![ids[0].clone(); 5],
            vec![ids[0].clone(), "missing-later-slot".into()],
        ] {
            let (result, writes, _) = dispatch_attempt(
                &settings,
                &project.id,
                project_path,
                &invalid,
                variant,
                true,
                runtime.clone(),
                control.clone(),
            )
            .await;
            assert!(result.is_err());
            assert!(writes.is_empty());
        }
        assert_eq!(
            (
                control.cold.load(Ordering::SeqCst),
                control.warm.load(Ordering::SeqCst),
                control.renders.load(Ordering::SeqCst)
            ),
            before,
            "invalid slots refuse before any peak or render"
        );
        assert_eq!(control.loads.load(Ordering::SeqCst), 1);
        assert_eq!(control.ordinary.load(Ordering::SeqCst), 0);
        drop(runtime);
        drop(cache);
        join.join().unwrap();
        server.abort();
        let _ = server.await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candle_moss_base_multi_reference_slots_reach_cold_warm_render_and_reentry() {
        exercise_multi_reference_dispatch(MossImageVariant::Base, "candle").await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candle_moss_sft_multi_reference_slots_reach_cold_warm_render_and_reentry() {
        exercise_multi_reference_dispatch(MossImageVariant::Sft, "candle").await;
    }
    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mlx_moss_base_multi_reference_slots_reach_cold_warm_render_and_reentry() {
        exercise_multi_reference_dispatch(MossImageVariant::Base, "mlx").await;
    }
    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mlx_moss_sft_multi_reference_slots_reach_cold_warm_render_and_reentry() {
        exercise_multi_reference_dispatch(MossImageVariant::Sft, "mlx").await;
    }
}
