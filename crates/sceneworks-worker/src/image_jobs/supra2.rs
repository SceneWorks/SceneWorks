//! Supra2-IMG wiring through the generic image lane (sc-24213 MLX / sc-24214 Candle, epic 24212).
//!
//! The `supra2_img` engine loads from three local resources and nothing else:
//! `LoadSpec::weights` = the converted native bundle (the injected `modelPath`, produced at install
//! by the `supra2_img` converter in `model_jobs.rs`), and the two catalog co-requisites the
//! descriptor declares — `text_encoder` (google/flan-t5-base) and `vae` (stabilityai/sd-vae-ft-mse) —
//! staged cache-only at their pinned revisions by the generic `attach_required_components`. There is
//! no Supra2-specific production code on the generation path: routing, defaults (seed 0, CFG 3.0,
//! 50 steps, 256x256) and refusals all come from the catalog row and the linked descriptor. These
//! tests pin that wiring against the shipped catalog entry on whichever backend the build links:
//! `mlx-gen-supra2` on macOS, `candle-gen-supra2` off-Mac (sc-24214) — ONE engine id, one bundle,
//! the same two pinned components. The `#[ignore]`d real-weight tests drive one 256x256 render
//! through the production MLX job arm (macOS) and one production `image_generate` job through the
//! Candle route (CUDA, `windows-candle.yml` `run_supra2_worker_smoke`).

#[cfg(test)]
mod tests {
    use crate::settings::Settings;
    use gen_core::{LoadSpec, WeightsSource};
    use sceneworks_core::contracts::JsonObject;
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};

    const T5_REPO: &str = "google/flan-t5-base";
    const T5_REVISION: &str = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2";
    const VAE_REPO: &str = "stabilityai/sd-vae-ft-mse";
    const VAE_REVISION: &str = "31f26fdeee1355a5c34592e401dd41e45d25a493";

    fn settings_at(data_dir: &Path) -> Settings {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn shipped_supra2_entry() -> JsonObject {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                .expect("parse builtin.models.jsonc");
        manifest["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == "supra2_img"))
            .and_then(Value::as_object)
            .cloned()
            .expect("supra2_img is in the builtin catalog")
    }

    fn stage(root: &Path, repo: &str, revision: &str, files: &[&str]) -> PathBuf {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        for file in files {
            std::fs::write(snapshot.join(file), b"x").unwrap();
        }
        snapshot
    }

    /// The linked engine declares exactly the two components the catalog row provisions, and the
    /// catalog pins the lane's defaults to the release's own.
    #[test]
    fn the_descriptor_and_catalog_row_agree_on_components_and_defaults() {
        let descriptor = crate::inference_runtime::media_descriptor("supra2_img")
            .expect("supra2_img is linked on this backend");
        // The SAME engine id resolves to this build's own provider: `mlx-gen-supra2` on macOS,
        // `candle-gen-supra2` off-Mac (sc-24214).
        let expected_backend = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };
        assert_eq!(descriptor.backend, expected_backend);
        assert_eq!(descriptor.required_components, &["text_encoder", "vae"]);
        assert!(!descriptor.capabilities.supports_negative_prompt);
        // The 128-token Flan-T5 context the prompt guide documents is the engine's own published
        // budget; an over-long prompt is truncated and reported as `prompt_truncated`, which the
        // generic lane forwards onto the job and the asset like any engine warning.
        assert_eq!(
            descriptor.capabilities.max_prompt_tokens,
            Some(gen_core::supra2::MAX_PROMPT_TOKENS)
        );
        assert_eq!(gen_core::supra2::MAX_PROMPT_TOKENS, 128);
        assert_eq!(descriptor.capabilities.min_size, 256);
        assert_eq!(descriptor.capabilities.max_size, 256);
        let entry = shipped_supra2_entry();
        assert_eq!(entry["defaults"]["resolution"], "256x256");
        assert_eq!(entry["defaults"]["steps"], gen_core::supra2::DEFAULT_STEPS);
        assert_eq!(
            entry["defaults"]["guidanceScale"],
            json!(gen_core::supra2::DEFAULT_GUIDANCE)
        );
        let row = crate::engines::mlx_model("supra2_img").expect("MODEL_TABLE row resolves");
        assert_eq!(row.engine_id(), gen_core::supra2::PROVIDER_ID);
        assert_eq!(row.backend(), expected_backend);
    }

    /// The catalog row is not platform-scoped (sc-24214): every download — the checkpoint and both
    /// co-requisites — applies on Windows/Linux exactly as on macOS, and the declared Candle floor
    /// exists so the fit gate never admits a load blind (`FitDecision::Unknown`).
    #[test]
    fn the_catalog_row_installs_the_same_components_on_every_platform() {
        let entry = shipped_supra2_entry();
        assert_eq!(entry["macOnly"], false);
        let downloads = entry["downloads"].as_array().expect("downloads");
        assert_eq!(downloads.len(), 3);
        for download in downloads {
            assert!(
                download.get("platforms").is_none(),
                "no download is platform-scoped: {download}"
            );
        }
        assert!(entry["candle"]["minMemoryGb"]
            .as_f64()
            .is_some_and(|gb| gb > 0.0));
        assert_eq!(entry["candle"]["measured"], false);
    }

    /// Generation stages both co-requisites at their EXACT pinned revisions next to the bundle.
    #[test]
    fn generation_stages_the_pinned_text_encoder_and_vae() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let t5 = stage(
            data.path(),
            T5_REPO,
            T5_REVISION,
            &["config.json", "model.safetensors", "tokenizer.json"],
        );
        let vae = stage(
            data.path(),
            VAE_REPO,
            VAE_REVISION,
            &["config.json", "diffusion_pytorch_model.safetensors"],
        );
        let bundle = data.path().join("models/mlx/supra2_img");
        let spec = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(bundle.clone())),
            "supra2_img",
            &shipped_supra2_entry(),
            &settings_at(data.path()),
        )
        .expect("installed components stage");
        assert_eq!(spec.weights, WeightsSource::Dir(bundle));
        assert_eq!(spec.components.len(), 2);
        assert_eq!(spec.components["text_encoder"], WeightsSource::Dir(t5));
        assert_eq!(spec.components["vae"], WeightsSource::Dir(vae));
    }

    /// A missing or torn component is a typed failure before any load — never a download.
    #[test]
    fn a_missing_or_torn_component_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        stage(
            data.path(),
            T5_REPO,
            T5_REVISION,
            &["config.json", "model.safetensors", "tokenizer.json"],
        );
        let attach = |data: &Path| {
            super::super::attach_required_components(
                LoadSpec::new(WeightsSource::Dir(data.join("bundle"))),
                "supra2_img",
                &shipped_supra2_entry(),
                &settings_at(data),
            )
        };
        let error = attach(data.path()).expect_err("no VAE installed");
        assert!(
            error.to_string().contains("'vae'") && error.to_string().contains("not installed"),
            "{error}"
        );
        // The VAE snapshot without its weights.
        stage(data.path(), VAE_REPO, VAE_REVISION, &["config.json"]);
        let error = attach(data.path()).expect_err("torn VAE");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    /// sc-24216: an injected converted bundle that fails the engine's pin contract (here: the
    /// install marker survived but the bundle manifest did not) is refused at weight resolution
    /// with an actionable message naming the Model Manager remedy — before any load and never as a
    /// stubbed or downloaded fallback — and the dispatch gap surfaces that same message.
    #[test]
    fn a_damaged_converted_bundle_is_refused_with_an_actionable_message() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let bundle = data.path().join("models/mlx/supra2_img");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join("model_index.json"), b"{}").unwrap();
        let mut entry = shipped_supra2_entry();
        entry.insert("modelPath".to_owned(), json!(bundle));
        let payload = json!({
            "model": "supra2_img",
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 256,
            "height": 256,
            "modelManifestEntry": Value::Object(entry),
        });
        let request = sceneworks_core::image_request::ImageRequest::from_payload(
            payload.as_object().unwrap(),
        );
        let settings = settings_at(data.path());
        let error = super::super::resolve_weights_dir(&request, &settings)
            .expect_err("a damaged bundle never resolves");
        let message = error.to_string();
        assert!(
            message.contains("Supra2-IMG's converted model files are damaged or incomplete")
                && message.contains("run Convert again")
                && message.contains("is not a converted Supra2 bundle"),
            "{message}"
        );
        // The MLX dispatch gap (the stub-fallback guard) surfaces the same refusal; the Candle
        // lane's route resolves weights through the same `resolve_weights_dir` and propagates it.
        #[cfg(target_os = "macos")]
        assert_eq!(
            super::super::mlx_weights_gap(&request, &settings).as_deref(),
            Some(message.as_str())
        );
    }

    // ---------------------------------------------------------------------------------------
    // Real weights: one 256x256 render through the production MLX job arm.
    // ---------------------------------------------------------------------------------------
    #[cfg(target_os = "macos")]
    mod mlx_real_weights {
        use super::*;

        use crate::api_client::ApiClient;
        use axum::extract::{Path as AxumPath, State};
        use axum::response::{IntoResponse, Response};
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use sceneworks_core::contracts::JobSnapshot;
        use std::sync::{Arc, Mutex};

        type Posts = Arc<Mutex<Vec<Value>>>;

        /// The stub jobs API's state: every progress POST, plus a user-cancel that (when armed) is
        /// requested the moment the first denoise step is reported — a cancel mid-denoise.
        #[derive(Clone, Default)]
        struct JobApi {
            posts: Posts,
            cancel_on_first_step: bool,
            cancel_requested: Arc<std::sync::atomic::AtomicBool>,
        }

        impl JobApi {
            fn cancel_requested(&self) -> bool {
                self.cancel_requested
                    .load(std::sync::atomic::Ordering::SeqCst)
            }
        }

        async fn spawn_job_api() -> (String, Posts) {
            spawn_job_api_with(false).await
        }

        async fn spawn_job_api_with(cancel_on_first_step: bool) -> (String, Posts) {
            async fn job_route(
                State(api): State<JobApi>,
                AxumPath(job_id): AxumPath<String>,
            ) -> Response {
                Json(crate::tests::job_snapshot_json(
                    &job_id,
                    api.cancel_requested(),
                ))
                .into_response()
            }
            async fn progress_route(
                State(api): State<JobApi>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Response {
                let is_step = body["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(" step "));
                api.posts.lock().unwrap().push(body);
                if api.cancel_on_first_step
                    && is_step
                    && !api
                        .cancel_requested
                        .swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    // The cancel is requested AT the first progress event. Holding this response past
                    // the lane's 2 s cancel-poll throttle makes the next event the worker handles
                    // observe it — independent of GPU speed: while the event loop waits here, the
                    // engine can run at most the 64-event channel ahead (`blocking_send`
                    // back-pressure), far less than a 4 x 80-step batch.
                    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
                }
                Json(crate::tests::job_snapshot_json(
                    &job_id,
                    api.cancel_requested(),
                ))
                .into_response()
            }
            async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
                Json(json!({
                    "id": worker_id, "gpuId": "mlx", "gpuName": null, "status": "busy",
                    "currentJobId": "job-supra2", "capabilities": [], "loadedModels": [],
                    "registeredAt": "2026-10-20T00:00:00Z", "lastSeenAt": "2026-10-20T00:00:00Z"
                }))
                .into_response()
            }
            let api = JobApi {
                cancel_on_first_step,
                ..JobApi::default()
            };
            let posts = api.posts.clone();
            let app = Router::new()
                .route("/api/v1/jobs/:job_id", get(job_route))
                .route("/api/v1/jobs/:job_id/progress", post(progress_route))
                .route(
                    "/api/v1/workers/:worker_id/heartbeat",
                    post(heartbeat_route),
                )
                .with_state(api);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            (format!("http://{address}"), posts)
        }

        fn files_under(root: &Path) -> Vec<PathBuf> {
            let mut found = Vec::new();
            let mut stack = vec![root.to_path_buf()];
            while let Some(dir) = stack.pop() {
                for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else {
                        found.push(path);
                    }
                }
            }
            found
        }

        /// Install-time conversion (the `supra2_img` converter) followed by a default text-to-image job
        /// through `generate_stream` — the production MLX arm with the real registry loader — on the
        /// real pinned weights. Env: `SUPRA2_HF_HUB_CACHE` (the hub dir holding the three pinned
        /// snapshots) and optional `SUPRA2_RENDER_OUT` (a dir the rendered PNG is copied to).
        #[test]
        #[ignore = "real weights + Metal: needs SUPRA2_HF_HUB_CACHE with the three pinned snapshots"]
        fn supra2_real_weight_render_through_the_worker_arm() {
            let hub = std::env::var("SUPRA2_HF_HUB_CACHE").expect("SUPRA2_HF_HUB_CACHE");
            let _env = crate::test_env::EnvVars::set(&[
                ("HF_HUB_CACHE", hub.as_str()),
                ("HUGGINGFACE_HUB_CACHE", ""),
                ("HF_HOME", ""),
            ]);
            let data = tempfile::tempdir().unwrap();
            let project = tempfile::tempdir().unwrap();
            let snapshot = |pin: gen_core::supra2::ComponentPin| {
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    data.path(),
                    pin.repo,
                    pin.revision,
                )
                .unwrap_or_else(|| panic!("{}@{} is not cached", pin.repo, pin.revision))
            };
            // The install-time convert, exactly as the convert job runs it.
            let bundle = data.path().join("models/mlx/supra2_img");
            let started = std::time::Instant::now();
            crate::model_jobs::convert_supra2_bundle(
                &snapshot(gen_core::supra2::SUPRA2_SOURCE).join("model_final_ema.pt"),
                &snapshot(gen_core::supra2::FLAN_T5_BASE),
                &snapshot(gen_core::supra2::SD_VAE),
                &bundle,
            )
            .expect("install-time conversion");
            eprintln!("convert: {:.1}s", started.elapsed().as_secs_f32());

            // What the API hands the worker once the convert ran: the catalog row + its `modelPath`.
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let started = std::time::Instant::now();
            let (outcome, posts, asset_writes) = runtime.block_on(async {
                let (base_url, posts) = spawn_job_api().await;
                let mut settings = settings_at(data.path());
                settings.api_url = base_url;
                let api = ApiClient::new(&settings);
                let job: JobSnapshot =
                    serde_json::from_value(crate::tests::job_snapshot_json("job-supra2", false))
                        .unwrap();
                let payload = json!({
                    "projectId": "project-supra2",
                    "model": "supra2_img",
                    "mode": "text_to_image",
                    "prompt": "a red fox sitting in fresh snow, soft morning light",
                    "width": 256,
                    "height": 256,
                    "count": 1,
                    "modelManifestEntry": Value::Object(entry),
                });
                let request = sceneworks_core::image_request::ImageRequest::from_payload(
                    payload.as_object().unwrap(),
                );
                let plan = super::super::super::ImagePlan::with_count(
                    &request,
                    request.image_count(),
                    None,
                );
                let mut asset_writes = Vec::new();
                let outcome = super::super::super::generate_stream(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "mlx",
                    &mut asset_writes,
                )
                .await;
                let posts = posts.lock().unwrap().clone();
                (outcome, posts, asset_writes)
            });
            eprintln!("render: {:.1}s", started.elapsed().as_secs_f32());
            outcome.unwrap_or_else(|error| panic!("render failed: {error}\n{posts:#?}"));
            assert_eq!(asset_writes.len(), 1, "{asset_writes:#?}");
            let pngs: Vec<PathBuf> = files_under(project.path())
                .into_iter()
                .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
                .collect();
            assert_eq!(pngs.len(), 1, "{pngs:?}");
            let image = image::open(&pngs[0]).expect("a viewable PNG");
            assert_eq!((image.width(), image.height()), (256, 256));
            if let Ok(out) = std::env::var("SUPRA2_RENDER_OUT") {
                std::fs::create_dir_all(&out).unwrap();
                let target = Path::new(&out).join("supra2_img_worker_seed0_cfg3_steps50.png");
                std::fs::copy(&pngs[0], &target).unwrap();
                eprintln!("render saved: {}", target.display());
            }
        }

        /// One job through the production MLX arm against a fresh stub jobs API.
        async fn run_lifecycle_job(
            settings: &Settings,
            entry: &JsonObject,
            project: &Path,
            job_id: &str,
            (count, steps): (u32, u32),
            cancel_on_first_step: bool,
        ) -> (crate::WorkerResult<()>, Vec<Value>, Vec<Value>) {
            let (base_url, posts) = spawn_job_api_with(cancel_on_first_step).await;
            let mut settings = settings.clone();
            settings.api_url = base_url;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json(job_id, false)).unwrap();
            let payload = json!({
                "projectId": "project-supra2",
                "model": "supra2_img",
                "mode": "text_to_image",
                "prompt": "a red fox sitting in fresh snow, soft morning light",
                "width": 256,
                "height": 256,
                "count": count,
                "seed": 0,
                "advanced": { "steps": steps },
                "modelManifestEntry": Value::Object(entry.clone()),
            });
            let request = sceneworks_core::image_request::ImageRequest::from_payload(
                payload.as_object().unwrap(),
            );
            let plan = super::super::ImagePlan::with_count(&request, request.image_count(), None);
            let mut asset_writes = Vec::new();
            let outcome = super::super::generate_stream(
                &api,
                &settings,
                &job,
                &plan,
                project,
                "mlx",
                &mut asset_writes,
            )
            .await;
            let posts = posts.lock().unwrap().clone();
            (outcome, posts, asset_writes)
        }

        fn messages(posts: &[Value]) -> Vec<String> {
            posts
                .iter()
                .filter_map(|post| post["message"].as_str().map(str::to_owned))
                .collect()
        }

        fn pngs_under(root: &Path) -> Vec<PathBuf> {
            files_under(root)
                .into_iter()
                .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
                .collect()
        }

        /// The installed-model job lifecycle (sc-24216) on real weights, fully OFFLINE: after the
        /// install-time conversion, with `HF_HUB_OFFLINE=1` and the Hub base URL pointed at a closed
        /// port (any network attempt fails the job), the production MLX arm
        ///   1. reports per-step progress and is CANCELED by a user cancel issued at its first progress
        ///      event — the job ends `Canceled`, posts the terminal canceled state, and never renders
        ///      the whole 4-image batch (deterministic: see `progress_route`; a Supra2 image takes only
        ///      ~1.5 s, under the lane's 2 s cancel-poll throttle, so timing alone would race);
        ///   2. RECOVERS: the next job in the same process completes with step 1..N + decoding progress
        ///      and one 256x256 image, and the generator stays resident;
        ///   3. UNLOADS: evicting the cached generator frees it, and a job after the unload reloads
        ///      from the local files and renders the identical image (same seed, same pinned weights).
        ///
        /// Env as [`supra2_real_weight_render_through_the_worker_arm`].
        #[test]
        #[ignore = "real weights + Metal: needs SUPRA2_HF_HUB_CACHE with the three pinned snapshots"]
        fn supra2_real_weight_offline_lifecycle_cancel_recover_unload() {
            let hub = std::env::var("SUPRA2_HF_HUB_CACHE").expect("SUPRA2_HF_HUB_CACHE");
            let _env = crate::test_env::EnvVars::set(&[
                ("HF_HUB_CACHE", hub.as_str()),
                ("HUGGINGFACE_HUB_CACHE", ""),
                ("HF_HOME", ""),
                ("HF_HUB_OFFLINE", "1"),
            ]);
            let data = tempfile::tempdir().unwrap();
            let snapshot = |pin: gen_core::supra2::ComponentPin| {
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    data.path(),
                    pin.repo,
                    pin.revision,
                )
                .unwrap_or_else(|| panic!("{}@{} is not cached", pin.repo, pin.revision))
            };
            let bundle = data.path().join("models/mlx/supra2_img");
            crate::model_jobs::convert_supra2_bundle(
                &snapshot(gen_core::supra2::SUPRA2_SOURCE).join("model_final_ema.pt"),
                &snapshot(gen_core::supra2::FLAN_T5_BASE),
                &snapshot(gen_core::supra2::SD_VAE),
                &bundle,
            )
            .expect("install-time conversion");
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));
            let mut settings = settings_at(data.path());
            // Offline: a Hub fetch anywhere on the load/generate path would hit a closed port.
            settings.huggingface_base_url = "http://127.0.0.1:9".to_owned();

            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                // 1. Cancel issued at the first progress event (held past the poll throttle by the stub),
                //    observed at the next event, long before 4 x 80 steps finish. The job only returns
                //    once the blocking generation has actually stopped (`CancelJoinGuard`), so its
                //    return is the release of the render's resources; the recovery job below proves the
                //    worker is usable again.
                let canceled_project = tempfile::tempdir().unwrap();
                let (outcome, posts, canceled_assets) = run_lifecycle_job(
                    &settings,
                    &entry,
                    canceled_project.path(),
                    "job-supra2-cancel",
                    (4, 80),
                    true,
                )
                .await;
                assert!(
                    matches!(outcome, Err(crate::WorkerError::Canceled(_))),
                    "the user cancel ends the job Canceled: {outcome:?}\n{posts:#?}"
                );
                let text = messages(&posts);
                assert!(
                    text.iter().any(|m| m.contains("Image 1/4 — step 1/80")),
                    "{text:#?}"
                );
                assert!(
                    !text.iter().any(|m| m.contains("Image 4/4 — decoding")),
                    "canceled before the batch finished: {text:#?}"
                );
                assert_eq!(posts.last().unwrap()["status"], "canceled", "{posts:#?}");
                assert!(canceled_assets.len() < 4, "{canceled_assets:#?}");
                assert!(pngs_under(canceled_project.path()).len() < 4);

                // 2. Recovery: the next job completes in the same process.
                let first = tempfile::tempdir().unwrap();
                let (outcome, posts, assets) = run_lifecycle_job(
                    &settings,
                    &entry,
                    first.path(),
                    "job-supra2-a",
                    (1, 50),
                    false,
                )
                .await;
                outcome
                    .unwrap_or_else(|error| panic!("recovery render failed: {error}\n{posts:#?}"));
                assert_eq!(assets.len(), 1, "{assets:#?}");
                let text = messages(&posts);
                for expected in ["step 1/50", "step 50/50", "decoding"] {
                    assert!(
                        text.iter().any(|m| m.contains(expected)),
                        "{expected}: {text:#?}"
                    );
                }
                let first_png = pngs_under(first.path());
                assert_eq!(first_png.len(), 1);
                let first_image = image::open(&first_png[0]).unwrap().to_rgb8();
                assert_eq!(first_image.dimensions(), (256, 256));
                assert!(
                    crate::generator_cache::cached_generator_resident_bytes()
                        .await
                        .unwrap()
                        .is_some(),
                    "the completed job leaves Supra2 resident for the next one"
                );

                // 3. Unload, then a job reloads from the local files.
                assert!(crate::generator_cache::evict_cached_generator()
                    .await
                    .unwrap());
                assert!(crate::generator_cache::cached_generator_resident_bytes()
                    .await
                    .unwrap()
                    .is_none());
                let second = tempfile::tempdir().unwrap();
                let (outcome, posts, _) = run_lifecycle_job(
                    &settings,
                    &entry,
                    second.path(),
                    "job-supra2-b",
                    (1, 50),
                    false,
                )
                .await;
                outcome.unwrap_or_else(|error| {
                    panic!("post-unload render failed: {error}\n{posts:#?}")
                });
                let second_png = pngs_under(second.path());
                assert_eq!(second_png.len(), 1);
                assert_eq!(
                    image::open(&second_png[0]).unwrap().to_rgb8(),
                    first_image,
                    "the reload renders the identical seed-0 image"
                );
                assert!(crate::generator_cache::evict_cached_generator()
                    .await
                    .unwrap());
            });
        }
    }

    // ---------------------------------------------------------------------------------------
    // Real weights (sc-24214): one production `image_generate` job through the Candle route.
    // ---------------------------------------------------------------------------------------
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    mod candle_real_weights {
        use super::*;
        use crate::api_client::ApiClient;
        use axum::extract::{Path as AxumPath, State};
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use sceneworks_core::contracts::JobSnapshot;
        use sceneworks_core::project_store::ProjectStore;
        use std::sync::{Arc, Mutex};

        const JOB_ID: &str = "supra2-cuda-image-generate";
        type Posts = Arc<Mutex<Vec<(String, Value)>>>;

        fn job_snapshot(job_id: &str) -> Value {
            crate::tests::job_snapshot_json(job_id, false)
        }

        async fn spawn_worker_api() -> (String, Posts) {
            async fn job_route(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
                Json(job_snapshot(&job_id))
            }
            async fn progress_route(
                State(posts): State<Posts>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Json<Value> {
                posts.lock().unwrap().push((job_id.clone(), body));
                Json(job_snapshot(&job_id))
            }
            async fn metrics_route(Json(body): Json<Value>) -> Json<Value> {
                Json(body)
            }
            async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
                Json(json!({
                    "id": worker_id, "gpuId": "0", "gpuName": null, "status": "busy",
                    "currentJobId": JOB_ID, "capabilities": [], "loadedModels": [],
                    "registeredAt": "2026-10-20T00:00:00Z", "lastSeenAt": "2026-10-20T00:00:00Z"
                }))
            }
            let posts: Posts = Arc::default();
            let app = Router::new()
                .route("/api/v1/jobs/:job_id", get(job_route))
                .route("/api/v1/jobs/:job_id/progress", post(progress_route))
                .route("/api/v1/jobs/:job_id/metrics", post(metrics_route))
                .route(
                    "/api/v1/workers/:worker_id/heartbeat",
                    post(heartbeat_route),
                )
                .with_state(posts.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            (format!("http://{address}"), posts)
        }

        /// Install-time conversion (the `supra2_img` converter, exactly as the convert job runs it)
        /// followed by ONE default text-to-image job through `run_image_generate_job` — the
        /// production off-Mac dispatch: the scheduler's candle routing table, the generic
        /// `CandleTxt2Img` route, the injected bundle `modelPath`, cache-only staging of the pinned
        /// `text_encoder` + `vae`, the cached `candle-gen-supra2` load and the asset write. Nothing
        /// is downloaded: `HF_HUB_CACHE` must already hold the three pinned snapshots (inference's
        /// `candle-supra2` real-weight job materializes them in `E:\huggingface\hub` on the
        /// real-weights host). A missing snapshot is a failure, never a skip.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[ignore = "real weights + CUDA: needs HF_HUB_CACHE with the three pinned Supra2 snapshots"]
        async fn supra2_worker_gpu_smoke() {
            let hub = std::path::PathBuf::from(
                std::env::var("HF_HUB_CACHE").expect("set HF_HUB_CACHE to the real-weights hub"),
            );
            assert!(
                hub.is_dir(),
                "HF_HUB_CACHE is not a directory: {}",
                hub.display()
            );
            let temp = tempfile::tempdir().unwrap();
            let (api_url, posts) = spawn_worker_api().await;
            let mut settings = Settings::for_test(temp.path().join("data"));
            settings.api_url = api_url;
            settings.config_dir = temp.path().join("config");
            settings.worker_id = "supra2-smoke-worker".to_owned();
            settings.gpu_id = "cuda".to_owned();
            settings.backend_candle_enabled = true;
            settings.heartbeat_seconds = 30;

            let snapshot = |pin: gen_core::supra2::ComponentPin| {
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    &settings.data_dir,
                    pin.repo,
                    pin.revision,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "HF_HUB_CACHE must already hold {}@{} (no download at generation time)",
                        pin.repo, pin.revision
                    )
                })
            };
            let bundle = settings.data_dir.join("models/mlx/supra2_img");
            let started = std::time::Instant::now();
            crate::model_jobs::convert_supra2_bundle(
                &snapshot(gen_core::supra2::SUPRA2_SOURCE).join("model_final_ema.pt"),
                &snapshot(gen_core::supra2::FLAN_T5_BASE),
                &snapshot(gen_core::supra2::SD_VAE),
                &bundle,
            )
            .expect("install-time conversion");
            let convert_s = started.elapsed().as_secs_f64();

            // What the API hands the worker once the convert ran: the catalog row + its `modelPath`.
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));
            let project = ProjectStore::new(settings.data_dir.clone(), "supra2-smoke")
                .create_project("Supra2-IMG CUDA worker smoke")
                .expect("smoke project creates");
            let mut job = job_snapshot(JOB_ID);
            job["type"] = json!("image_generate");
            job["payload"] = json!({
                "projectId": project.id,
                "model": "supra2_img",
                "mode": "text_to_image",
                "prompt": "a red fox sitting in fresh snow, soft morning light",
                "width": 256,
                "height": 256,
                "count": 1,
                "seed": 0,
                "modelManifestEntry": Value::Object(entry),
            });
            let job: JobSnapshot = serde_json::from_value(job).expect("smoke job deserializes");
            let api = ApiClient::new(&settings);
            let started = std::time::Instant::now();
            crate::image_jobs::run_image_generate_job(&api, &settings, &job)
                .await
                .expect("the Supra2-IMG CUDA worker job completes");
            let render_s = started.elapsed().as_secs_f64();

            let posts = posts.lock().unwrap().clone();
            assert!(
                posts.iter().any(|(_, body)| body["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("/50"))),
                "the engine's step progress must reach the job API: {posts:#?}"
            );
            let result = posts
                .iter()
                .rev()
                .find(|(id, body)| id == JOB_ID && body["status"] == "completed")
                .map(|(_, body)| body["result"].clone())
                .unwrap_or_else(|| panic!("the job posted a completed result: {posts:#?}"));
            let assets = result["assetWrites"].as_array().expect("asset writes");
            assert_eq!(assets.len(), 1, "{assets:#?}");
            let asset = &assets[0];
            assert_eq!(
                result["adapter"], "candle_supra2",
                "the generation set is stamped by the Candle Supra2 lane: {result}"
            );
            let media_path = asset["mediaPath"].as_str().expect("asset media path");
            let rendered_path = std::path::PathBuf::from(&project.path).join(media_path);
            let decoded = image::open(&rendered_path)
                .unwrap_or_else(|error| panic!("open {}: {error}", rendered_path.display()))
                .to_rgb8();
            let rendered = gen_core::Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            };
            assert_eq!((rendered.width, rendered.height), (256, 256));
            let std = crate::smoke_support::image_std(&rendered);
            assert!(
                std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
                "the worker render looks degenerate (RGB std {std:.2})"
            );
            eprintln!(
                "RESULT status=pass model=supra2_img backend=candle geometry=256x256 steps=50 \
                 cfg=3.0 seed=0 convert_s={convert_s:.1} render_s={render_s:.1} \
                 rgb_std={std:.3} output={}",
                rendered_path.display()
            );
        }
    }
}
