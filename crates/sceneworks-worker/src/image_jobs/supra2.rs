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
//! Candle route (CUDA, `windows-candle.yml` `run_supra2_worker_smoke`); the same two lanes also run
//! the epic acceptance matrix (`acceptance_matrix`) through `run_image_generate_job` on a bundle a
//! separate convert step produced.

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

    /// `(current, total)` of a "Batch of 4 — step k/N" progress message (the lane's one-pass label
    /// for a native count batch; a streaming suffix may follow).
    fn batch_of_four_step(message: &str) -> Option<(u32, u32)> {
        let (_, rest) = message.split_once("Batch of 4 — step ")?;
        let (current, rest) = rest.split_once('/')?;
        let total: String = rest.chars().take_while(char::is_ascii_digit).collect();
        Some((current.parse().ok()?, total.parse().ok()?))
    }

    /// The lifecycle cancel (sc-24216) against a count-4 job. Supra2 is a native count batch
    /// (`image.nativeCountBatch`, sc-24215), so the job is ONE engine call whose progress names the
    /// batch ("Batch of 4 — step k/N."), never a per-image slice. The cancel lands mid-denoise: the
    /// batch's first step was reported, and neither its final step nor its decode ever was.
    fn assert_canceled_mid_batch(messages: &[String]) {
        let steps: Vec<(u32, u32)> = messages
            .iter()
            .filter_map(|message| batch_of_four_step(message))
            .collect();
        assert!(
            steps.iter().any(|&(current, _)| current == 1),
            "the one-pass batch reported its first step: {messages:#?}"
        );
        assert!(
            !steps.iter().any(|&(current, total)| current >= total),
            "canceled before the batch's final step: {messages:#?}"
        );
        assert!(
            !messages.iter().any(|message| message.contains("decoding")),
            "canceled before the batch decoded: {messages:#?}"
        );
    }

    /// The cancel assertion itself: the pre-sc-24215 per-image labels, a batch that ran to its
    /// last step, and a batch that decoded all fail it; a mid-denoise cancel passes.
    #[test]
    fn the_mid_batch_cancel_assertion_reads_the_one_pass_batch_label() {
        let owned =
            |messages: &[&str]| messages.iter().map(|m| (*m).to_owned()).collect::<Vec<_>>();
        let fails = |messages: Vec<String>| {
            std::panic::catch_unwind(|| assert_canceled_mid_batch(&messages)).is_err()
        };
        assert!(!fails(owned(&[
            "Loading Supra2-IMG.",
            "Batch of 4 — step 1/80.",
            "Batch of 4 — step 2/80. Streaming",
            "Cancelling…",
        ])));
        assert!(fails(owned(&[
            "Image 1/4 — step 1/80.",
            "Image 1/4 — step 2/80."
        ])));
        assert!(fails(owned(&[
            "Batch of 4 — step 1/80.",
            "Batch of 4 — step 80/80."
        ])));
        assert!(fails(owned(&[
            "Batch of 4 — step 1/80.",
            "Batch of 4 — decoding."
        ])));
        assert_eq!(
            batch_of_four_step("Batch of 4 — step 12/80. (x)"),
            Some((12, 80))
        );
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

        /// The epic acceptance matrix on Metal through `run_image_generate_job` (see
        /// `acceptance_matrix::run`). Run by `scripts/ci/supra2-mlx-real-weights/run.sh` AFTER the
        /// convert test wrote `SUPRA2_BUNDLE_OUT` in its own process.
        #[test]
        #[ignore = "real weights + Metal: needs SUPRA2_BUNDLE_DIR (pre-converted) and SUPRA2_HF_HUB_CACHE"]
        fn supra2_real_weight_acceptance_matrix() {
            super::acceptance_matrix::run();
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
            let plan =
                super::super::super::ImagePlan::with_count(&request, request.image_count(), None);
            let mut asset_writes = Vec::new();
            let outcome = super::super::super::generate_stream(
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
            // The real conversion satisfies the integrity check the catalog (sizes) and the worker
            // preflight (digests) apply, so a genuine install reads `converted` and loads.
            for check in [
                sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
                sceneworks_core::converted_artifacts::ArtifactCheck::Digests,
            ] {
                assert_eq!(
                    sceneworks_core::converted_artifacts::converted_artifact_problem(
                        "supra2_img",
                        &bundle,
                        check
                    ),
                    None,
                    "{check:?}"
                );
            }
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
                assert_canceled_mid_batch(&messages(&posts));
                assert_eq!(posts.last().unwrap()["status"], "canceled", "{posts:#?}");
                assert!(canceled_assets.is_empty(), "{canceled_assets:#?}");
                assert!(pngs_under(canceled_project.path()).is_empty());

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

        /// The epic acceptance matrix on CUDA through `run_image_generate_job` (see
        /// `acceptance_matrix::run`). Run by `windows-candle.yml` `supra2-worker-smoke` AFTER the
        /// convert test wrote `SUPRA2_BUNDLE_OUT` in its own process.
        #[test]
        #[ignore = "real weights + CUDA: needs SUPRA2_BUNDLE_DIR (pre-converted) and HF_HUB_CACHE"]
        fn supra2_worker_gpu_acceptance_matrix() {
            super::acceptance_matrix::run();
        }

        // -----------------------------------------------------------------------------------
        // sc-24216: the installed-model job lifecycle on the Candle route.
        // -----------------------------------------------------------------------------------

        /// Stub jobs API for the lifecycle: every progress POST, plus a user cancel that (when
        /// armed) is requested AT the first reported denoise step. That response is held past the
        /// lane's 2 s cancel-poll throttle, so the next event the worker handles observes it —
        /// independent of GPU speed (the engine can run at most the 64-event channel ahead under
        /// `blocking_send` back-pressure, far less than a 4 x 80-step batch).
        #[derive(Clone, Default)]
        struct LifecycleApi {
            posts: Posts,
            cancel_on_first_step: bool,
            cancel_requested: Arc<std::sync::atomic::AtomicBool>,
        }

        async fn spawn_lifecycle_api(cancel_on_first_step: bool) -> (String, Posts) {
            fn snapshot(api: &LifecycleApi, job_id: &str) -> Json<Value> {
                Json(crate::tests::job_snapshot_json(
                    job_id,
                    api.cancel_requested
                        .load(std::sync::atomic::Ordering::SeqCst),
                ))
            }
            async fn job_route(
                State(api): State<LifecycleApi>,
                AxumPath(job_id): AxumPath<String>,
            ) -> Json<Value> {
                snapshot(&api, &job_id)
            }
            async fn progress_route(
                State(api): State<LifecycleApi>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Json<Value> {
                let is_step = body["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(" step "));
                api.posts.lock().unwrap().push((job_id.clone(), body));
                if api.cancel_on_first_step
                    && is_step
                    && !api
                        .cancel_requested
                        .swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
                }
                snapshot(&api, &job_id)
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
            let api = LifecycleApi {
                cancel_on_first_step,
                ..LifecycleApi::default()
            };
            let posts = api.posts.clone();
            let app = Router::new()
                .route("/api/v1/jobs/:job_id", get(job_route))
                .route("/api/v1/jobs/:job_id/progress", post(progress_route))
                .route("/api/v1/jobs/:job_id/metrics", post(metrics_route))
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

        fn pngs_under(root: &Path) -> Vec<PathBuf> {
            let mut found = Vec::new();
            let mut stack = vec![root.to_path_buf()];
            while let Some(dir) = stack.pop() {
                for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if path.extension().is_some_and(|ext| ext == "png") {
                        found.push(path);
                    }
                }
            }
            found
        }

        /// One production `image_generate` job (`run_image_generate_job`) against a fresh stub
        /// API, in its own project; returns the outcome, the posts and the project dir.
        async fn run_lifecycle_job(
            settings: &Settings,
            entry: &JsonObject,
            job_id: &str,
            (count, steps): (u32, u32),
            cancel_on_first_step: bool,
        ) -> (crate::WorkerResult<()>, Vec<(String, Value)>, PathBuf) {
            let (api_url, posts) = spawn_lifecycle_api(cancel_on_first_step).await;
            let mut settings = settings.clone();
            settings.api_url = api_url;
            let project = ProjectStore::new(settings.data_dir.clone(), job_id)
                .create_project(job_id)
                .expect("lifecycle project creates");
            let mut job = job_snapshot(job_id);
            job["type"] = json!("image_generate");
            job["payload"] = json!({
                "projectId": project.id,
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
            let job: JobSnapshot = serde_json::from_value(job).expect("lifecycle job deserializes");
            let api = ApiClient::new(&settings);
            let outcome = crate::image_jobs::run_image_generate_job(&api, &settings, &job).await;
            let posts = posts.lock().unwrap().clone();
            (outcome, posts, PathBuf::from(&project.path))
        }

        fn messages(posts: &[(String, Value)]) -> Vec<String> {
            posts
                .iter()
                .filter_map(|(_, body)| body["message"].as_str().map(str::to_owned))
                .collect()
        }

        /// The installed-model job lifecycle (sc-24216) on the production Candle route, OFFLINE
        /// (`HF_HUB_OFFLINE=1`, the Hub base URL at a closed port): a user cancel issued at the
        /// first progress event ends a 4-image job `Canceled` before the batch completes; the next
        /// job in the same process RECOVERS and renders with step + decoding progress; evicting the
        /// cached generator UNLOADS it (a second eviction finds nothing resident) and a job after
        /// the unload reloads from the local files and renders. Env as [`supra2_worker_gpu_smoke`].
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        #[ignore = "real weights + CUDA: needs HF_HUB_CACHE with the three pinned Supra2 snapshots"]
        async fn supra2_worker_gpu_lifecycle() {
            let hub = std::path::PathBuf::from(
                std::env::var("HF_HUB_CACHE").expect("set HF_HUB_CACHE to the real-weights hub"),
            );
            assert!(
                hub.is_dir(),
                "HF_HUB_CACHE is not a directory: {}",
                hub.display()
            );
            let _offline = crate::test_env::EnvVars::set(&[("HF_HUB_OFFLINE", "1")]);
            let temp = tempfile::tempdir().unwrap();
            let mut settings = Settings::for_test(temp.path().join("data"));
            settings.config_dir = temp.path().join("config");
            settings.worker_id = "supra2-lifecycle-worker".to_owned();
            settings.gpu_id = "cuda".to_owned();
            settings.backend_candle_enabled = true;
            settings.heartbeat_seconds = 30;
            settings.huggingface_base_url = "http://127.0.0.1:9".to_owned();

            let snapshot = |pin: gen_core::supra2::ComponentPin| {
                crate::model_jobs::huggingface_pinned_snapshot_dir(
                    &settings.data_dir,
                    pin.repo,
                    pin.revision,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "HF_HUB_CACHE must already hold {}@{}",
                        pin.repo, pin.revision
                    )
                })
            };
            let bundle = settings.data_dir.join("models/mlx/supra2_img");
            crate::model_jobs::convert_supra2_bundle(
                &snapshot(gen_core::supra2::SUPRA2_SOURCE).join("model_final_ema.pt"),
                &snapshot(gen_core::supra2::FLAN_T5_BASE),
                &snapshot(gen_core::supra2::SD_VAE),
                &bundle,
            )
            .expect("install-time conversion");
            for check in [
                sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
                sceneworks_core::converted_artifacts::ArtifactCheck::Digests,
            ] {
                assert_eq!(
                    sceneworks_core::converted_artifacts::converted_artifact_problem(
                        "supra2_img",
                        &bundle,
                        check
                    ),
                    None,
                    "{check:?}"
                );
            }
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));

            // 1. Cancel at the first progress event.
            let (outcome, posts, project) =
                run_lifecycle_job(&settings, &entry, "supra2-lifecycle-cancel", (4, 80), true)
                    .await;
            assert!(
                matches!(outcome, Err(crate::WorkerError::Canceled(_))),
                "the user cancel ends the job Canceled: {outcome:?}\n{posts:#?}"
            );
            assert_canceled_mid_batch(&messages(&posts));
            assert_eq!(posts.last().unwrap().1["status"], "canceled", "{posts:#?}");
            assert!(
                !posts.iter().any(|(_, body)| body["result"]["assetWrites"]
                    .as_array()
                    .is_some_and(|assets| !assets.is_empty())),
                "a canceled batch publishes no asset: {posts:#?}"
            );
            assert!(pngs_under(&project).is_empty());

            // 2. Recovery in the same process.
            let (outcome, posts, project) =
                run_lifecycle_job(&settings, &entry, "supra2-lifecycle-a", (1, 50), false).await;
            outcome.unwrap_or_else(|error| panic!("recovery job failed: {error}\n{posts:#?}"));
            let text = messages(&posts);
            for expected in ["step 1/50", "step 50/50", "decoding"] {
                assert!(
                    text.iter().any(|m| m.contains(expected)),
                    "{expected}: {text:#?}"
                );
            }
            assert_eq!(pngs_under(&project).len(), 1);

            // 3. Unload, then a job reloads from the local files.
            assert!(
                crate::generator_cache::evict_cached_generator()
                    .await
                    .unwrap(),
                "the completed job left Supra2 resident"
            );
            assert!(
                !crate::generator_cache::evict_cached_generator()
                    .await
                    .unwrap(),
                "nothing stays resident after the unload"
            );
            let (outcome, posts, project) =
                run_lifecycle_job(&settings, &entry, "supra2-lifecycle-b", (1, 50), false).await;
            outcome.unwrap_or_else(|error| panic!("post-unload job failed: {error}\n{posts:#?}"));
            let pngs = pngs_under(&project);
            assert_eq!(pngs.len(), 1);
            let decoded = image::open(&pngs[0]).unwrap().to_rgb8();
            assert_eq!(decoded.dimensions(), (256, 256));
            assert!(crate::generator_cache::evict_cached_generator()
                .await
                .unwrap());
            eprintln!("RESULT status=pass model=supra2_img backend=candle lifecycle=cancel,recover,unload,reload");
        }
    }

    // ---------------------------------------------------------------------------------------
    // Real weights (epic-review AT matrix): the published interface's acceptance cells through
    // the PRODUCTION job path (`run_image_generate_job`) on whichever backend this build links.
    // Driven by one `#[ignore]`d test per backend (`mlx_real_weights` / `candle_real_weights`).
    // ---------------------------------------------------------------------------------------
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    mod acceptance_matrix {
        use super::*;
        use crate::api_client::ApiClient;
        use axum::extract::{Path as AxumPath, State};
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use sceneworks_core::contracts::JobSnapshot;
        use sceneworks_core::project_store::ProjectStore;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;

        type Posts = Arc<Mutex<Vec<Value>>>;

        const SEED: u64 = 2417;
        const PROMPT: &str = "a red fox sitting in fresh snow, soft morning light";
        const BACKEND: &str = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };

        async fn spawn_api() -> (String, Posts) {
            async fn job_route(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
                Json(crate::tests::job_snapshot_json(&job_id, false))
            }
            async fn progress_route(
                State(posts): State<Posts>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Json<Value> {
                posts.lock().unwrap().push(body);
                Json(crate::tests::job_snapshot_json(&job_id, false))
            }
            async fn metrics_route(Json(body): Json<Value>) -> Json<Value> {
                Json(body)
            }
            async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
                Json(json!({
                    "id": worker_id, "gpuId": BACKEND, "gpuName": null, "status": "busy",
                    "currentJobId": null, "capabilities": [], "loadedModels": [],
                    "registeredAt": "2026-10-21T00:00:00Z", "lastSeenAt": "2026-10-21T00:00:00Z"
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

        /// One completed job: its asset writes (in index order), each asset's decoded image and
        /// PNG path, and the job's wall time.
        struct Render {
            assets: Vec<Value>,
            images: Vec<image::RgbImage>,
            pngs: Vec<PathBuf>,
            latency_s: f64,
        }

        /// One production `image_generate` job (`run_image_generate_job`) against a fresh stub API,
        /// in its own project. `extra` overrides the default request (prompt, 256x256, count 1).
        async fn render(
            settings: &Settings,
            entry: &JsonObject,
            job_id: &str,
            extra: Value,
        ) -> Render {
            let (api_url, posts) = spawn_api().await;
            let mut settings = settings.clone();
            settings.api_url = api_url;
            let project = ProjectStore::new(settings.data_dir.clone(), "worker")
                .create_project(job_id)
                .expect("matrix project creates");
            let mut payload = json!({
                "projectId": project.id,
                "model": "supra2_img",
                "mode": "text_to_image",
                "prompt": PROMPT,
                "width": 256,
                "height": 256,
                "count": 1,
                "modelManifestEntry": Value::Object(entry.clone()),
            });
            for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                payload[key] = value;
            }
            let mut job = crate::tests::job_snapshot_json(job_id, false);
            job["type"] = json!("image_generate");
            job["payload"] = payload;
            let job: JobSnapshot = serde_json::from_value(job).expect("matrix job deserializes");
            let api = ApiClient::new(&settings);
            let started = Instant::now();
            let outcome = crate::image_jobs::run_image_generate_job(&api, &settings, &job).await;
            let latency_s = started.elapsed().as_secs_f64();
            let posts = posts.lock().unwrap().clone();
            outcome.unwrap_or_else(|error| panic!("{job_id} failed: {error}\n{posts:#?}"));
            let mut assets: Vec<Value> = posts
                .iter()
                .rev()
                .find(|body| body["status"] == "completed")
                .and_then(|body| body["result"]["assetWrites"].as_array().cloned())
                .unwrap_or_else(|| panic!("{job_id} posted no completed result: {posts:#?}"));
            assets.sort_by_key(|asset| asset["index"].as_u64());
            let pngs: Vec<PathBuf> = assets
                .iter()
                .map(|asset| {
                    PathBuf::from(&project.path)
                        .join(asset["mediaPath"].as_str().expect("asset media path"))
                })
                .collect();
            let images = pngs
                .iter()
                .map(|png| {
                    image::open(png)
                        .unwrap_or_else(|error| panic!("open {}: {error}", png.display()))
                        .to_rgb8()
                })
                .collect();
            Render {
                assets,
                images,
                pngs,
                latency_s,
            }
        }

        /// `(mean |Δ|, max |Δ|)` over two images' RGB bytes (0..=255 scale).
        fn pixel_diff(a: &image::RgbImage, b: &image::RgbImage) -> (f64, u32) {
            assert_eq!(a.dimensions(), b.dimensions());
            let (mut sum, mut max) = (0_u64, 0_u32);
            for (x, y) in a.as_raw().iter().zip(b.as_raw()) {
                let d = u32::from(x.abs_diff(*y));
                sum += u64::from(d);
                max = max.max(d);
            }
            (sum as f64 / a.as_raw().len() as f64, max)
        }

        fn rgb_std(image: &image::RgbImage) -> f64 {
            crate::smoke_support::image_std(&gen_core::Image {
                width: image.width(),
                height: image.height(),
                pixels: image.as_raw().clone(),
            })
        }

        /// `(max mean |Δ| / 255, max max |Δ|)` for image k of a COUNT-4 batch vs the count-1 render
        /// on seed S + k on Metal. A batch of four runs differently shaped GEMMs than a batch of
        /// one, so the reduction order (and the last bits) move: measured on nax-macos-2, run
        /// 39695321783, worst case mean 0.050 / max 4 over k = 0..3. No margin, and the run is
        /// deterministic on that runner, so a Metal/MLX change that moves it trips this as a
        /// signal. This does not contradict inference's documented MLX claim, which is bit-exact
        /// for batch 2 vs batch 1 on image 0 only. The same REQUEST repeated stays bit-exact
        /// (cell 4).
        const MLX_BATCH4_VS_SINGLE_BOUND: (f64, u32) = (0.1, 4);
        /// The same for CUDA, declared by inference (`candle-gen-supra2/tests/real_weights.rs`): a
        /// batch and a single image run differently tiled GEMMs, mean |Δ| <= 0.1 / 255, max <= 1.
        const CUDA_BATCH_VS_SINGLE_BOUND: (f64, u32) = (0.1, 1);
        const BATCH_VS_SINGLE_BOUND: (f64, u32) = if cfg!(target_os = "macos") {
            MLX_BATCH4_VS_SINGLE_BOUND
        } else {
            CUDA_BATCH_VS_SINGLE_BOUND
        };

        fn within_batch_bound(mean: f64, max: u32) -> bool {
            mean <= BATCH_VS_SINGLE_BOUND.0 && max <= BATCH_VS_SINGLE_BOUND.1
        }

        /// Every cell's verdict, rewritten to `<out>/supra2_matrix_<backend>.json` after each cell
        /// (so a failing cell still leaves the earlier evidence) and printed as a `CELL` line.
        struct Summary {
            out: Option<PathBuf>,
            cells: Vec<Value>,
        }

        impl Summary {
            fn check(&mut self, cell: &str, pass: bool, latency_s: Option<f64>, detail: Value) {
                eprintln!(
                    "CELL backend={BACKEND} cell={cell} status={} latency_s={} detail={detail}",
                    if pass { "pass" } else { "FAIL" },
                    latency_s.map_or("-".to_owned(), |s| format!("{s:.2}"))
                );
                self.cells.push(json!({
                    "cell": cell, "pass": pass, "latencyS": latency_s, "detail": detail
                }));
                self.write();
                assert!(pass, "cell {cell} failed: {detail}");
            }

            /// A cell this process has no verdict for, recorded as such — never as a pass.
            fn not_applicable(&mut self, cell: &str, detail: Value) {
                eprintln!(
                    "CELL backend={BACKEND} cell={cell} status=not_applicable latency_s=- \
                     detail={detail}"
                );
                self.cells.push(json!({
                    "cell": cell, "pass": null, "status": "not_applicable", "latencyS": null,
                    "detail": detail
                }));
                self.write();
            }

            fn write(&self) {
                if let Some(out) = &self.out {
                    std::fs::write(
                        out.join(format!("supra2_matrix_{BACKEND}.json")),
                        serde_json::to_vec_pretty(&json!({
                            "backend": BACKEND, "seed": SEED, "cells": self.cells
                        }))
                        .unwrap(),
                    )
                    .unwrap();
                }
            }

            fn keep(&self, render: &Render, cell: &str) {
                let Some(out) = &self.out else { return };
                for (k, png) in render.pngs.iter().enumerate() {
                    let seed = render.assets[k]["seed"].as_u64().unwrap_or_default();
                    std::fs::copy(
                        png,
                        out.join(format!("matrix_{BACKEND}_{cell}_{k}_seed{seed}.png")),
                    )
                    .unwrap();
                }
            }
        }

        /// The lifecycle test cell 7 folds in: the lane runs it in its own process before the
        /// matrix, on the same build and card.
        const LIFECYCLE_TEST: &str = if cfg!(target_os = "macos") {
            "mlx_real_weights::supra2_real_weight_offline_lifecycle_cancel_recover_unload"
        } else {
            "candle_real_weights::supra2_worker_gpu_lifecycle"
        };

        /// Cell 7's verdict from the lifecycle test's own libtest log (`SUPRA2_LIFECYCLE_LOG`,
        /// written by the lane script, which runs it with `--exact`): passed iff the log shows the
        /// lifecycle test itself started (`test …::<LIFECYCLE_TEST> ...`) and the run's summary is
        /// exactly one test passing, none failing. The test's own `--nocapture` output lands
        /// between its name and its `ok`, so the summary — not the `... ok` suffix — carries the
        /// verdict; with one test run and that test named, it is that test's. `None` when the lane
        /// named no log (a standalone matrix run), which the matrix records as not applicable
        /// rather than as a pass.
        fn lifecycle_verdict(log: Option<&str>) -> Option<bool> {
            let log = log?;
            let started = format!("::{LIFECYCLE_TEST} ...");
            Some(
                log.lines()
                    .any(|line| line.starts_with("test ") && line.contains(&started))
                    && log.contains("test result: ok. 1 passed; 0 failed"),
            )
        }

        /// sc-24217 pre-merge review: cell 7 is the lifecycle test's REAL result, never a constant.
        ///
        /// *Mutations that red this:* return `Some(true)` for any log; drop the which-test check (a
        /// log whose one passing test is some other test reads as a pass).
        #[test]
        fn cell_7_folds_the_lifecycle_tests_own_result() {
            let passed = format!(
                "running 1 test\ntest image_jobs::supra2::tests::{LIFECYCLE_TEST} ... ok\n\n\
                 test result: ok. 1 passed; 0 failed; 0 ignored"
            );
            assert_eq!(lifecycle_verdict(Some(&passed)), Some(true));
            // `--nocapture`: the test's own output sits between its name and its `ok` (the
            // windows-candle lane's real shape, CRLF line endings included).
            let interleaved = format!(
                "running 1 test\r\ntest image_jobs::supra2::tests::{LIFECYCLE_TEST} ... RESULT \
                 status=pass lifecycle=cancel,recover,unload,reload\r\nok\r\n\r\n\
                 test result: ok. 1 passed; 0 failed; 0 ignored\r\n"
            );
            assert_eq!(lifecycle_verdict(Some(&interleaved)), Some(true));
            let failed = format!(
                "running 1 test\ntest image_jobs::supra2::tests::{LIFECYCLE_TEST} ... FAILED\n\n\
                 test result: FAILED. 0 passed; 1 failed; 0 ignored"
            );
            assert_eq!(lifecycle_verdict(Some(&failed)), Some(false));
            let other = "running 1 test\ntest image_jobs::supra2::tests::other ... ok\n\n\
                         test result: ok. 1 passed; 0 failed; 0 ignored";
            assert_eq!(lifecycle_verdict(Some(other)), Some(false));
            assert_eq!(
                lifecycle_verdict(Some("")),
                Some(false),
                "a missing log fails"
            );
            assert_eq!(lifecycle_verdict(None), None);
        }

        fn warnings(asset: &Value) -> Vec<String> {
            asset["rawAdapterSettings"]["generationWarnings"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|warning| warning["code"].as_str().map(str::to_owned))
                .collect()
        }

        /// The acceptance matrix. Env: `SUPRA2_BUNDLE_DIR` — the converted bundle a SEPARATE step
        /// produced (`supra2_real_weight_convert_writes_a_loadable_bundle` with
        /// `SUPRA2_BUNDLE_OUT`), at the installed layout `<data dir>/models/mlx/supra2_img`;
        /// `SUPRA2_HF_HUB_CACHE` (else `HF_HUB_CACHE`) — the hub holding the
        /// pinned Flan-T5-Base and SD VAE snapshots; optional `SUPRA2_RENDER_OUT` — the artifact dir
        /// the PNGs and `supra2_matrix_<backend>.json` are written to.
        ///
        /// Cells: (1) a short prompt renders a non-degenerate 256x256 (the COLD first job);
        /// (2) a >128-token prompt renders and every asset carries `prompt_truncated`; (3) CFG 3.0
        /// and CFG 1.0 (unguided) both render and differ, each recording its unconditional source;
        /// (4) seed S count 1 twice is pixel-identical; (5) count 4 on seed S publishes seeds
        /// S..S+3 and image k matches the count-1 render on seed S+k within the backend's declared
        /// bound; (6) the whole matrix runs OFFLINE (`HF_HUB_OFFLINE=1`, Hub base URL on a closed
        /// port); (7) cancel → recover → unload/reload is the lifecycle test's (not duplicated):
        /// its actual result is folded in from the libtest log the lane names in
        /// `SUPRA2_LIFECYCLE_LOG` (recorded `not_applicable`, never passed, when none is named).
        pub(super) fn run() {
            let bundle = PathBuf::from(std::env::var("SUPRA2_BUNDLE_DIR").expect(
                "SUPRA2_BUNDLE_DIR: the bundle the separate convert step wrote (SUPRA2_BUNDLE_OUT)",
            ));
            assert_eq!(
                sceneworks_core::converted_artifacts::converted_artifact_problem(
                    "supra2_img",
                    &bundle,
                    sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
                ),
                None,
                "SUPRA2_BUNDLE_DIR is not an installed Supra2 bundle: {}",
                bundle.display()
            );
            let hub = std::env::var("SUPRA2_HF_HUB_CACHE")
                .or_else(|_| std::env::var("HF_HUB_CACHE"))
                .expect("SUPRA2_HF_HUB_CACHE or HF_HUB_CACHE: the hub with the pinned components");
            let _env = crate::test_env::EnvVars::set(&[
                ("HF_HUB_CACHE", hub.as_str()),
                ("HUGGINGFACE_HUB_CACHE", ""),
                ("HF_HOME", ""),
                ("HF_HUB_OFFLINE", "1"),
            ]);
            let out = std::env::var("SUPRA2_RENDER_OUT").ok().map(PathBuf::from);
            if let Some(out) = &out {
                std::fs::create_dir_all(out).unwrap();
            }
            let mut summary = Summary {
                out,
                cells: Vec::new(),
            };
            // The production path admits a `modelPath` only inside an app-managed root, so the
            // shared bundle sits where an install puts it — `<data dir>/models/mlx/supra2_img` —
            // and that data dir is this worker's.
            let data_dir = bundle
                .ends_with("models/mlx/supra2_img")
                .then(|| bundle.ancestors().nth(3).map(Path::to_path_buf))
                .flatten()
                .unwrap_or_else(|| {
                    panic!(
                        "SUPRA2_BUNDLE_DIR must be <data dir>/models/mlx/supra2_img (the installed \
                         layout): {}",
                        bundle.display()
                    )
                });
            let temp = tempfile::tempdir().unwrap();
            let mut settings = Settings::for_test(data_dir);
            settings.config_dir = temp.path().join("config");
            settings.worker_id = format!("supra2-matrix-{BACKEND}");
            settings.heartbeat_seconds = 30;
            if cfg!(target_os = "macos") {
                settings.gpu_id = "mlx".to_owned();
                settings.backend_mlx_enabled = true;
            } else {
                settings.gpu_id = "cuda".to_owned();
                settings.backend_candle_enabled = true;
            }
            // (6) Offline: any Hub fetch anywhere on the path hits a closed port and fails the job.
            settings.huggingface_base_url = "http://127.0.0.1:9".to_owned();
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));

            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                // A cold first job: nothing is resident in this process yet.
                let _ = crate::generator_cache::evict_cached_generator().await;

                // (1) Short prompt, release defaults (CFG 3.0, 50 steps) on seed S — the cold job.
                let base = render(&settings, &entry, "matrix-short", json!({ "seed": SEED })).await;
                summary.keep(&base, "short");
                let asset = &base.assets[0];
                let raw = &asset["rawAdapterSettings"];
                let std = rgb_std(&base.images[0]);
                summary.check(
                    "1_short_prompt_renders",
                    base.assets.len() == 1
                        && base.images[0].dimensions() == (256, 256)
                        && std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT
                        && asset["seed"] == json!(SEED)
                        && warnings(asset).is_empty()
                        && raw["numInferenceSteps"] == json!(gen_core::supra2::DEFAULT_STEPS)
                        && raw["guidanceScale"] == json!(gen_core::supra2::DEFAULT_GUIDANCE)
                        && raw[sceneworks_core::supra2::UNCONDITIONAL_KEY] == "stored"
                        && raw[sceneworks_core::supra2::BACKEND_KEY] == BACKEND,
                    Some(base.latency_s),
                    json!({
                        "phase": "cold (includes load)", "rgbStd": std,
                        "assets": base.assets.len(), "dimensions": base.images[0].dimensions(),
                        "seed": asset["seed"], "warnings": warnings(asset),
                        "unconditional": raw[sceneworks_core::supra2::UNCONDITIONAL_KEY],
                        "backend": raw[sceneworks_core::supra2::BACKEND_KEY],
                    }),
                );

                // (4) The same request again (warm): pixel-identical.
                let again = render(&settings, &entry, "matrix-repeat", json!({ "seed": SEED })).await;
                summary.keep(&again, "repeat");
                let (mean, max) = pixel_diff(&base.images[0], &again.images[0]);
                summary.check(
                    "4_seed_repeat_is_identical",
                    again.assets.len() == 1 && max == 0,
                    Some(again.latency_s),
                    json!({ "phase": "warm", "meanAbsDiff": mean, "maxAbsDiff": max }),
                );

                // (3) CFG 1.0 is upstream's unguided branch: it renders, differs from CFG 3.0, and
                // records that no unconditional context was used.
                let unguided = render(
                    &settings,
                    &entry,
                    "matrix-cfg1",
                    json!({ "seed": SEED, "advanced": { "guidanceScale": 1.0 } }),
                )
                .await;
                summary.keep(&unguided, "cfg1");
                let raw = &unguided.assets[0]["rawAdapterSettings"];
                let std = rgb_std(&unguided.images[0]);
                let (mean, max) = pixel_diff(&base.images[0], &unguided.images[0]);
                summary.check(
                    "3_cfg3_vs_cfg1_both_render_and_differ",
                    std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT
                        && max > 0
                        && raw["guidanceScale"] == json!(1.0)
                        && raw[sceneworks_core::supra2::UNCONDITIONAL_KEY] == "none",
                    Some(unguided.latency_s),
                    json!({
                        "rgbStdCfg1": std, "meanAbsDiffVsCfg3": mean, "maxAbsDiffVsCfg3": max,
                        "unconditionalCfg1": raw[sceneworks_core::supra2::UNCONDITIONAL_KEY],
                    }),
                );

                // (2) A prompt far past the 128-token Flan-T5 context: renders, and every asset
                // records the truncation.
                let long_prompt = (0..24)
                    .map(|i| {
                        format!("detail {i}: a weathered stone lighthouse on a rocky cliff above a stormy sea")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let long = render(
                    &settings,
                    &entry,
                    "matrix-long",
                    json!({ "seed": SEED, "count": 2, "prompt": long_prompt }),
                )
                .await;
                summary.keep(&long, "long");
                let all_truncated = long.assets.iter().all(|asset| {
                    warnings(asset)
                        .iter()
                        .any(|code| code == gen_core::supra2::PROMPT_TRUNCATED_WARNING)
                });
                let stds: Vec<f64> = long.images.iter().map(rgb_std).collect();
                summary.check(
                    "2_long_prompt_truncates_and_renders",
                    long.assets.len() == 2
                        && all_truncated
                        && stds
                            .iter()
                            .all(|std| *std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT),
                    Some(long.latency_s),
                    json!({
                        "promptChars": long_prompt.len(), "assets": long.assets.len(),
                        "warnings": long.assets.iter().map(warnings).collect::<Vec<_>>(),
                        "rgbStd": stds,
                    }),
                );

                // (5) count 4 on seed S: seeds S..S+3, image k == the single render on S + k.
                let batch = render(
                    &settings,
                    &entry,
                    "matrix-batch",
                    json!({ "seed": SEED, "count": 4 }),
                )
                .await;
                summary.keep(&batch, "batch4");
                let seeds: Vec<Value> = batch.assets.iter().map(|a| a["seed"].clone()).collect();
                let mut diffs = vec![pixel_diff(&batch.images[0], &base.images[0])];
                let mut single_latency = Vec::new();
                for k in 1..4_u64 {
                    let single = render(
                        &settings,
                        &entry,
                        &format!("matrix-single-{k}"),
                        json!({ "seed": SEED + k }),
                    )
                    .await;
                    summary.keep(&single, &format!("single_seed_plus_{k}"));
                    single_latency.push(single.latency_s);
                    diffs.push(pixel_diff(&batch.images[k as usize], &single.images[0]));
                }
                summary.check(
                    "5_count4_matches_single_renders_on_seed_plus_k",
                    seeds == (0..4).map(|k| json!(SEED + k)).collect::<Vec<_>>()
                        && batch.images.len() == 4
                        && diffs.iter().all(|&(mean, max)| within_batch_bound(mean, max)),
                    Some(batch.latency_s),
                    json!({
                        "seeds": seeds,
                        "perImage": diffs.iter().map(|(mean, max)| json!({"meanAbsDiff": mean, "maxAbsDiff": max})).collect::<Vec<_>>(),
                        "bound": { "meanAbsDiff": BATCH_VS_SINGLE_BOUND.0, "maxAbsDiff": BATCH_VS_SINGLE_BOUND.1 },
                        "singleLatencyS": single_latency,
                    }),
                );

                // (6) Every job above ran with the Hub unreachable and offline mode on.
                summary.check(
                    "6_offline",
                    std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1")
                        && settings.huggingface_base_url == "http://127.0.0.1:9",
                    None,
                    json!({ "HF_HUB_OFFLINE": "1", "hubBaseUrl": settings.huggingface_base_url }),
                );
                // (7) Not duplicated here: the lifecycle test owns it, in its own process earlier in
                // the same lane run; its actual result is folded in from its log.
                let log_path = std::env::var("SUPRA2_LIFECYCLE_LOG").ok();
                let log = log_path
                    .as_ref()
                    .map(|path| std::fs::read_to_string(path).unwrap_or_default());
                let detail = json!({ "coveredBy": LIFECYCLE_TEST, "lifecycleLog": log_path });
                match lifecycle_verdict(log.as_deref()) {
                    Some(pass) => {
                        summary.check("7_cancel_recover_unload_reload", pass, None, detail)
                    }
                    None => summary.not_applicable("7_cancel_recover_unload_reload", detail),
                }
                let _ = crate::generator_cache::evict_cached_generator().await;
            });
        }
    }

    /// sc-24215: the published controls and the multi-image batch on the linked lane, driven
    /// through the production pieces every generic-lane render shares (native-batch resolution,
    /// `render_on_surface`, `drive_native_prompt_batch`, `mlx_raw_settings`, `consume_gen_events`)
    /// against a probe engine — no weights. The pre-load admission/load path is the real-weight
    /// tests' job; these pin what reaches the engine and what every asset records.
    mod controls_and_batch {
        use super::super::super as lane;
        use super::*;
        use crate::api_client::ApiClient;
        use gen_core::{CancelFlag, GenerationOutput, GenerationRequest, Generator, Progress};
        use sceneworks_core::contracts::JobSnapshot;
        use sceneworks_core::image_request::ImageRequest;
        use std::sync::{Arc, Mutex};

        /// Every request field the published interface maps onto.
        #[derive(Clone, Debug, PartialEq)]
        struct Seen {
            prompt: String,
            prompt_batch: Vec<String>,
            negative_prompt: Option<String>,
            width: u32,
            height: u32,
            count: u32,
            seed: Option<u64>,
            steps: Option<u32>,
            guidance: Option<f32>,
        }

        struct Supra2Probe {
            descriptor: gen_core::ModelDescriptor,
            /// Stop the way the engine does when its request cancel flag is set.
            canceled: bool,
            seen: Arc<Mutex<Vec<Seen>>>,
        }

        impl Generator for Supra2Probe {
            fn descriptor(&self) -> &gen_core::ModelDescriptor {
                &self.descriptor
            }

            fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
                Ok(())
            }

            /// Like the engine: progress per step and `count` images back from the one call
            /// (image `k`'s pixels are all `k`).
            fn generate(
                &self,
                req: &GenerationRequest,
                on_progress: &mut dyn FnMut(Progress),
            ) -> gen_core::Result<GenerationOutput> {
                self.seen.lock().unwrap().push(Seen {
                    prompt: req.prompt.clone(),
                    prompt_batch: req.prompt_batch.clone(),
                    negative_prompt: req.negative_prompt.clone(),
                    width: req.width,
                    height: req.height,
                    count: req.count,
                    seed: req.seed,
                    steps: req.steps,
                    guidance: req.guidance,
                });
                on_progress(Progress::Step {
                    current: 1,
                    total: req.steps.unwrap_or(1),
                });
                if self.canceled {
                    return Err(gen_core::Error::Canceled);
                }
                Ok(GenerationOutput::Images(
                    (0..req.count)
                        .map(|k| gen_core::Image {
                            width: req.width,
                            height: req.height,
                            pixels: vec![k as u8; (req.width * req.height * 3) as usize],
                        })
                        .collect(),
                ))
            }

            /// The engine's primary path: the output plus its over-long-prompt warning.
            fn generate_with_report(
                &self,
                req: &GenerationRequest,
                on_progress: &mut dyn FnMut(Progress),
            ) -> gen_core::Result<gen_core::GenerationReport> {
                let mut report =
                    gen_core::GenerationReport::from_output(self.generate(req, on_progress)?);
                report
                    .warnings
                    .extend(gen_core::supra2::prompt_truncated_warning(
                        140,
                        gen_core::supra2::MAX_PROMPT_TOKENS as usize,
                    ));
                Ok(report)
            }
        }

        fn request(extra: Value) -> ImageRequest {
            let mut payload = json!({
                "projectId": "project-supra2",
                "model": "supra2_img",
                "mode": "text_to_image",
                "prompt": "a red fox sitting in fresh snow",
                "width": 256,
                "height": 256,
                "count": 1,
                "modelManifestEntry": Value::Object(shipped_supra2_entry()),
            });
            for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                payload[key] = value;
            }
            ImageRequest::from_payload(payload.as_object().unwrap())
        }

        fn model() -> crate::engines::ResolvedModel {
            crate::engines::mlx_model("supra2_img").expect("supra2_img is linked")
        }

        /// The generic lane's producer for one native count batch: resolve the batch against the
        /// linked descriptor, then the one engine call through `render_on_surface`, published by
        /// `drive_native_prompt_batch` (exactly the lane's batch branch, minus admission).
        fn produce(
            request: &ImageRequest,
            canceled: bool,
            seen: Arc<Mutex<Vec<Seen>>>,
            tx: tokio::sync::mpsc::Sender<lane::GenEvent>,
        ) -> crate::error::WorkerResult<()> {
            let model = model();
            let batch = lane::lane_controls::resolve_native_prompt_batch(
                request,
                &model.descriptor.capabilities,
                &[],
            )?
            .expect("image.nativeCountBatch ⇒ one native call");
            let probe = Supra2Probe {
                descriptor: model.descriptor.clone(),
                canceled,
                seen,
            };
            let seed = lane::resolve_seed(request, 0);
            let steps = lane::resolve_steps(request, &model);
            let guidance = lane::resolve_guidance(request, &model);
            let enhance = lane::PromptEnhance::from_advanced(&request.advanced)?;
            let cancel = CancelFlag::new();
            lane::drive_native_prompt_batch(
                tx,
                seed,
                batch.images() as usize,
                |preview, on_progress| {
                    let warnings = std::cell::RefCell::new(Vec::new());
                    let images = lane::render_on_surface(
                        &probe,
                        "",
                        request.width,
                        request.height,
                        seed,
                        steps,
                        guidance,
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                        None,
                        &enhance,
                        gen_core::PromptEnhancementSink::default(),
                        preview,
                        &lane::LaneRgbaSurface {
                            warnings: Some(&warnings),
                            prompt_batch: Some(&batch),
                            ..Default::default()
                        },
                        &cancel,
                        on_progress,
                    )?;
                    let facts = lane::lane_controls::image_facts(&warnings.take(), &[], &[], None);
                    Ok((images, facts))
                },
            )
        }

        /// A count-3 job is ONE engine call carrying the prompt (no prompt batch), the base seed,
        /// the steps, the guidance and `count: 3`, published by the production consumer as three
        /// assets on seeds 40, 41, 42 — the seeds the engine drew image k's noise from
        /// (`image_seed(seed, k)`) — each recording its prompt, seed, steps, guidance, the pinned
        /// component revisions and the truncation warning, with progress naming the batch.
        ///
        /// *Mutations that red this:* drop `image.nativeCountBatch` handling (no batch resolves);
        /// send the one-prompt batch as `prompt_batch`; publish image k on the base seed; drop
        /// `modelRevisions` from `mlx_raw_settings`.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_count_n_job_is_one_engine_call_published_as_one_asset_per_image() {
            let dir = tempfile::tempdir().unwrap();
            let project_path = dir.path().join("project");
            std::fs::create_dir_all(project_path.join("assets").join("images")).unwrap();
            let request = request(json!({
                "seed": 40, "count": 3, "advanced": { "steps": 12, "guidanceScale": 4.5 }
            }));
            let plan = lane::ImagePlan::with_count(&request, request.image_count(), None);
            let steps = lane::resolve_steps(&request, &model());
            let raw_settings = lane::mlx_raw_settings(
                &request,
                "SupraLabs/Supra2-IMG",
                steps,
                None,
                lane::resolve_guidance(&request, &model()),
            );
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (tx, rx) = tokio::sync::mpsc::channel::<lane::GenEvent>(64);
            let producer_request = request.clone();
            let producer_seen = Arc::clone(&seen);
            let blocking = tokio::task::spawn_blocking(move || {
                produce(&producer_request, false, producer_seen, tx)
            });
            let (base_url, posts) = crate::tests::spawn_no_user_cancel_capture_stub().await;
            let mut settings = crate::tests::test_settings(base_url.clone(), None);
            settings.api_url = base_url;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-supra2", false))
                    .unwrap();
            let mut asset_writes = Vec::new();
            lane::consume_gen_events(
                &api,
                &settings,
                &job,
                &plan,
                &project_path,
                "mlx",
                "mlx_supra2",
                &raw_settings,
                3,
                rx,
                CancelFlag::new(),
                blocking,
                &mut asset_writes,
            )
            .await
            .expect("the batch completes");

            assert_eq!(
                *seen.lock().unwrap(),
                vec![Seen {
                    prompt: "a red fox sitting in fresh snow".to_owned(),
                    prompt_batch: Vec::new(),
                    negative_prompt: None,
                    width: 256,
                    height: 256,
                    count: 3,
                    seed: Some(40),
                    steps: Some(12),
                    guidance: Some(4.5),
                }],
                "one native call with every published control"
            );
            let published: Vec<(Value, Value)> = asset_writes
                .iter()
                .map(|asset| (asset["seed"].clone(), asset["index"].clone()))
                .collect();
            assert_eq!(
                published,
                vec![
                    (json!(40), json!(0)),
                    (json!(41), json!(1)),
                    (json!(42), json!(2))
                ]
            );
            let entry = shipped_supra2_entry();
            for asset in &asset_writes {
                assert_eq!(asset["prompt"], "a red fox sitting in fresh snow");
                let settings = &asset["rawAdapterSettings"];
                assert_eq!(settings["numInferenceSteps"], 12, "{settings}");
                assert_eq!(settings["guidanceScale"], 4.5, "{settings}");
                assert_eq!(
                    settings[sceneworks_core::supra2::MODEL_REVISIONS_KEY],
                    sceneworks_core::supra2::pinned_revisions(&entry),
                    "{settings}"
                );
                assert_eq!(
                    settings[sceneworks_core::supra2::MODEL_REVISIONS_KEY]
                        .as_array()
                        .map(Vec::len),
                    Some(3),
                    "denoiser + text encoder + VAE: {settings}"
                );
                assert_eq!(
                    settings["generationWarnings"][0]["code"],
                    gen_core::supra2::PROMPT_TRUNCATED_WARNING,
                    "{settings}"
                );
            }
            let posts = posts.lock().unwrap().clone();
            assert!(
                posts.iter().any(|post| post["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("Batch of 3 — step 1/12")),
                "progress names the batch: {posts:#?}"
            );
        }

        /// A seeded single image is the same engine request as image k of a batch on seed − k:
        /// the seed + k contract means a count-1 job on seed 41 reproduces image 1 of the seed-40
        /// batch.
        #[test]
        fn a_single_image_on_seed_k_is_the_batchs_image_k_request() {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            produce(
                &request(json!({ "seed": 41, "count": 1 })),
                false,
                Arc::clone(&seen),
                tx,
            )
            .expect("renders");
            let seen = seen.lock().unwrap().clone();
            assert_eq!(seen.len(), 1);
            assert_eq!((seen[0].seed, seen[0].count), (Some(41), 1));
            assert_eq!(gen_core::supra2::image_seed(40, 1), 41);
            let mut seeds = Vec::new();
            while let Ok(event) = rx.try_recv() {
                if let lane::GenEvent::Image { seed, .. } = event {
                    seeds.push(seed);
                }
            }
            assert_eq!(seeds, vec![41]);
        }

        /// The default API path: no `seed`, a `seeds` batch. The one engine call draws image k's
        /// noise from `seeds[0] + k`, so the published (recorded) seeds are exactly those, never the
        /// unused `seeds[1..]`.
        #[test]
        fn an_unseeded_batch_publishes_the_seeds_the_engine_used() {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            produce(
                &request(json!({ "count": 3, "seeds": [9000, 17, 4242] })),
                false,
                Arc::clone(&seen),
                tx,
            )
            .expect("renders");
            let seen = seen.lock().unwrap().clone();
            assert_eq!(
                (seen.len(), seen[0].seed, seen[0].count),
                (1, Some(9000), 3)
            );
            let mut published = Vec::new();
            while let Ok(event) = rx.try_recv() {
                if let lane::GenEvent::Image { index, seed, .. } = event {
                    published.push((index, seed));
                }
            }
            assert_eq!(published, vec![(0, 9000), (1, 9001), (2, 9002)]);
        }

        /// A cancel inside the one call publishes NOTHING.
        #[test]
        fn a_cancelled_batch_publishes_no_image() {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            let outcome = produce(
                &request(json!({ "seed": 40, "count": 4 })),
                true,
                Arc::clone(&seen),
                tx,
            );
            assert!(outcome.is_err());
            assert_eq!(seen.lock().unwrap().len(), 1);
            while let Ok(event) = rx.try_recv() {
                assert!(
                    !matches!(event, lane::GenEvent::Image { .. }),
                    "a canceled batch publishes no image"
                );
            }
        }

        /// The declared step ceiling: a request AT `limits.hardMaxSteps` reaches the engine with
        /// exactly that count (the generic 80-step clamp does not apply to Supra2).
        #[test]
        fn a_request_at_the_declared_step_ceiling_runs_exactly_that_count() {
            let ceiling = sceneworks_core::video_request::hard_max_steps(&shipped_supra2_entry())
                .expect("supra2_img declares limits.hardMaxSteps");
            let request = request(json!({ "seed": 1, "advanced": { "steps": ceiling } }));
            assert_eq!(lane::resolve_steps(&request, &model()), ceiling);
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (tx, _rx) = tokio::sync::mpsc::channel(64);
            produce(&request, false, Arc::clone(&seen), tx).expect("renders");
            assert_eq!(seen.lock().unwrap()[0].steps, Some(ceiling));
        }

        /// ...and one above it is refused by the boundary `run_image_generate_job` applies before
        /// load, while every OTHER shipped image model declares no ceiling and so keeps the lane's
        /// historical clamp unchanged.
        #[test]
        fn above_the_ceiling_is_refused_and_other_models_keep_their_clamp() {
            let entry = shipped_supra2_entry();
            let ceiling = sceneworks_core::video_request::hard_max_steps(&entry).unwrap();
            let payload = |steps: u32| {
                json!({ "model": "supra2_img", "prompt": "a fox", "advanced": { "steps": steps } })
                    .as_object()
                    .cloned()
                    .unwrap()
            };
            assert!(sceneworks_core::supra2::validate_request(&payload(ceiling), &entry).is_ok());
            let error = sceneworks_core::supra2::validate_request(&payload(ceiling + 1), &entry)
                .expect_err("above the declared ceiling");
            assert!(error.contains(&ceiling.to_string()), "{error}");

            let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .unwrap()
                .1;
            let manifest: Value =
                serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
            for model in manifest["models"].as_array().unwrap() {
                if model["id"] == "supra2_img" || model["type"] != "image" {
                    continue;
                }
                assert_eq!(
                    sceneworks_core::video_request::hard_max_steps(model.as_object().unwrap()),
                    None,
                    "{} must keep the generic clamp",
                    model["id"]
                );
            }
            assert_eq!(lane::max_requested_steps(50), 80);
        }

        // -----------------------------------------------------------------------------------
        // Through the PRODUCTION arm with admission satisfied: a weightless but priceable
        // install (every pinned file at its exact pinned size, safetensors headers the memory
        // contract can read, configs the provider checks), so `generate_*_stream_using` runs its
        // real pre-load provider contract + admission and only the engine is a probe.
        // -----------------------------------------------------------------------------------

        /// Write exactly `bytes`: `json` padded with spaces, else a one-tensor (`tensor`, U8)
        /// safetensors whose single tensor fills the file, else zeros. `set_len` keeps the pinned
        /// multi-hundred-MB sizes sparse.
        fn write_sized(path: &Path, bytes: u64, json: Option<&Value>, tensor: &str) {
            use std::io::Write;
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut file = std::fs::File::create(path).unwrap();
            if let Some(json) = json {
                let mut text = json.to_string();
                assert!(
                    text.len() as u64 <= bytes,
                    "{} does not fit",
                    path.display()
                );
                text.push_str(&" ".repeat((bytes - text.len() as u64) as usize));
                file.write_all(text.as_bytes()).unwrap();
                return;
            }
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                const HEADER: u64 = 256;
                let data = bytes - 8 - HEADER;
                let mut entry = serde_json::Map::new();
                entry.insert(
                    tensor.to_owned(),
                    json!({ "dtype": "U8", "shape": [data], "data_offsets": [0, data] }),
                );
                let mut header = Value::Object(entry).to_string();
                header.push_str(&" ".repeat(HEADER as usize - header.len()));
                file.write_all(&HEADER.to_le_bytes()).unwrap();
                file.write_all(header.as_bytes()).unwrap();
            }
            file.set_len(bytes).unwrap();
        }

        /// Stage one pinned component's snapshot (cache-only, at its exact revision) with every
        /// pinned file at its pinned size.
        fn stage_component(
            data: &Path,
            pin: gen_core::supra2::ComponentPin,
            config: &Value,
            tensor: &str,
        ) {
            let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data, pin.repo)
                .unwrap()
                .join("snapshots")
                .join(pin.revision);
            for file in pin.files {
                let json = (file.path == "config.json").then_some(config);
                write_sized(&snapshot.join(file.path), file.bytes, json, tensor);
            }
        }

        /// The converted bundle the pin contract accepts at `BundleCheck::SizesOnly` (the
        /// admission depth): the manifest `convert_pinned_checkpoint` writes for the pinned
        /// checkpoint, and both converted files at their pinned sizes.
        fn stage_priceable_install(data: &Path) -> PathBuf {
            use gen_core::supra2::{
                BundleComponent, BundleFile, BundleManifest, BundleSource, Supra2Config,
                FLAN_T5_BASE, PINNED_CONDITIONING, PINNED_DENOISER, SD_VAE, SUPRA2_SOURCE,
            };
            let cfg = Supra2Config::published();
            stage_component(
                data,
                FLAN_T5_BASE,
                &json!({
                    "d_model": cfg.ctx_dim,
                    "num_layers": gen_core::supra2::T5_LAYERS,
                    "num_heads": gen_core::supra2::T5_HEADS,
                    "d_kv": gen_core::supra2::T5_D_KV,
                    "relative_attention_num_buckets": 32,
                    "relative_attention_max_distance": 128,
                    "feed_forward_proj": "gated-gelu"
                }),
                "encoder.embed",
            );
            stage_component(
                data,
                SD_VAE,
                &json!({
                    "latent_channels": cfg.latent_channels,
                    "layers_per_block": gen_core::supra2::VAE_LAYERS_PER_BLOCK,
                    "norm_num_groups": gen_core::supra2::VAE_NORM_GROUPS,
                    "block_out_channels": gen_core::supra2::VAE_BLOCK_OUT_CHANNELS
                }),
                "decoder.weights",
            );
            let bundle = data.join("models/mlx/supra2_img");
            let file = |pin: gen_core::supra2::PinnedFile| BundleFile {
                file: pin.path.to_owned(),
                bytes: pin.bytes,
                sha256: pin.sha256.to_owned(),
            };
            let source = SUPRA2_SOURCE.files[0];
            let component = |id: &str, pin: gen_core::supra2::ComponentPin| BundleComponent {
                component: id.to_owned(),
                repo: pin.repo.to_owned(),
                revision: pin.revision.to_owned(),
                license: pin.license.to_owned(),
                download_bytes: pin.download_bytes(),
            };
            let manifest = BundleManifest {
                schema: gen_core::supra2::BUNDLE_SCHEMA.to_owned(),
                provider_id: gen_core::supra2::PROVIDER_ID.to_owned(),
                source: BundleSource {
                    repo: Some(SUPRA2_SOURCE.repo.to_owned()),
                    revision: Some(SUPRA2_SOURCE.revision.to_owned()),
                    file: source.path.to_owned(),
                    bytes: source.bytes,
                    sha256: source.sha256.to_owned(),
                    weights_key: "ema".to_owned(),
                },
                config: cfg,
                vae_scale: gen_core::supra2::VAE_SCALE,
                denoiser: file(PINNED_DENOISER),
                denoiser_tensors: cfg.expected_tensors().len(),
                conditioning: Some(file(PINNED_CONDITIONING)),
                components: vec![
                    component(gen_core::supra2::TEXT_ENCODER_COMPONENT, FLAN_T5_BASE),
                    component(gen_core::supra2::VAE_COMPONENT, SD_VAE),
                ],
            };
            std::fs::create_dir_all(&bundle).unwrap();
            std::fs::write(
                bundle.join(gen_core::supra2::BUNDLE_MANIFEST),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();
            write_sized(
                &bundle.join(PINNED_DENOISER.path),
                PINNED_DENOISER.bytes,
                None,
                "dit",
            );
            write_sized(
                &bundle.join(PINNED_CONDITIONING.path),
                PINNED_CONDITIONING.bytes,
                None,
                "uncond_text",
            );
            bundle
        }

        /// The full production arm (`generate_stream_using` on macOS,
        /// `generate_candle_stream_using` off-Mac) on the priceable install: the real provider
        /// memory contract prices it, admission passes, the probe engine is called ONCE with
        /// `count: 3`, and three assets are published on seeds 40, 41, 42.
        ///
        /// *Mutation that reds this:* drop the native-count branch in `resolve_native_prompt_batch`
        /// (three engine calls of count 1).
        #[test]
        fn the_production_arm_renders_a_count_n_job_in_one_native_call() {
            let _env = crate::test_env::EnvVars::set(&[
                ("HF_HUB_CACHE", ""),
                ("HUGGINGFACE_HUB_CACHE", ""),
                ("HF_HOME", ""),
                #[cfg(target_os = "macos")]
                (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64"),
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                (crate::vram_gate::CUDA_VRAM_CAP_ENV, "64"),
            ]);
            let data = tempfile::tempdir().unwrap();
            let project = tempfile::tempdir().unwrap();
            let bundle = stage_priceable_install(data.path());
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));
            let seen = Arc::new(Mutex::new(Vec::new()));
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let (outcome, posts, asset_writes) = runtime.block_on(async {
                let (base_url, posts) = crate::tests::spawn_no_user_cancel_capture_stub().await;
                let mut settings = settings_at(data.path());
                settings.api_url = base_url;
                settings.heartbeat_seconds = 5;
                let api = ApiClient::new(&settings);
                let job: JobSnapshot =
                    serde_json::from_value(crate::tests::job_snapshot_json("job-supra2", false))
                        .unwrap();
                let payload = json!({
                    "projectId": "project-supra2",
                    "model": "supra2_img",
                    "mode": "text_to_image",
                    "prompt": "a red fox sitting in fresh snow",
                    "width": 256,
                    "height": 256,
                    "count": 3,
                    "seed": 40,
                    "advanced": { "steps": 12, "guidanceScale": 4.5 },
                    "modelManifestEntry": Value::Object(entry),
                });
                let request = ImageRequest::from_payload(payload.as_object().unwrap());
                let plan = lane::ImagePlan::with_count(&request, request.image_count(), None);
                let loader_seen = Arc::clone(&seen);
                let loader = move |_engine: &str, _spec: &gen_core::LoadSpec| {
                    Ok(Box::new(Supra2Probe {
                        descriptor: model().descriptor,
                        canceled: false,
                        seen: loader_seen,
                    }) as Box<dyn Generator>)
                };
                let mut asset_writes = Vec::new();
                #[cfg(target_os = "macos")]
                let arm = lane::generate_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "mlx",
                    &mut asset_writes,
                    loader,
                );
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                let arm = lane::generate_candle_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "cuda",
                    &mut asset_writes,
                    loader,
                );
                let outcome = tokio::time::timeout(std::time::Duration::from_secs(120), arm)
                    .await
                    .expect("the supra2 arm finished");
                let posts = posts.lock().unwrap().clone();
                (outcome, posts, asset_writes)
            });
            assert!(outcome.is_ok(), "{outcome:?}\n{posts:#?}");
            let seen = seen.lock().unwrap().clone();
            assert_eq!(seen.len(), 1, "one engine call: {seen:#?}");
            assert_eq!(
                (seen[0].count, seen[0].seed, seen[0].steps, seen[0].guidance),
                (3, Some(40), Some(12), Some(4.5))
            );
            assert!(seen[0].prompt_batch.is_empty());
            let seeds: Vec<Value> = asset_writes
                .iter()
                .map(|asset| asset["seed"].clone())
                .collect();
            assert_eq!(seeds, vec![json!(40), json!(41), json!(42)]);
            // Reproducibility record: CFG 4.5 steered against the bundle's STORED unconditional
            // embedding (the manifest records its conditioning file), on this build's backend.
            for asset in &asset_writes {
                let raw = &asset["rawAdapterSettings"];
                assert_eq!(
                    raw[sceneworks_core::supra2::UNCONDITIONAL_KEY],
                    "stored",
                    "{raw}"
                );
                assert_eq!(
                    raw[sceneworks_core::supra2::BACKEND_KEY],
                    if cfg!(target_os = "macos") {
                        "mlx"
                    } else {
                        "candle"
                    },
                    "{raw}"
                );
            }
        }

        /// CFG <= 1 is upstream's unguided branch, recorded as no unconditional context; a replayed
        /// recipe cannot forge the worker-owned record.
        #[test]
        fn an_unguided_render_records_no_unconditional_context() {
            let request = request(json!({
                "advanced": { "guidanceScale": 1.0, "unconditional": "stored", "backend": "x" }
            }));
            let raw = lane::mlx_raw_settings(&request, "SupraLabs/Supra2-IMG", 50, None, Some(1.0));
            assert_eq!(raw[sceneworks_core::supra2::UNCONDITIONAL_KEY], "none");
            assert_ne!(raw[sceneworks_core::supra2::BACKEND_KEY], "x");
        }
    }
}
