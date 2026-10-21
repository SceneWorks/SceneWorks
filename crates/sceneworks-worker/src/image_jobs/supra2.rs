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

        async fn spawn_job_api() -> (String, Posts) {
            async fn job_route(AxumPath(job_id): AxumPath<String>) -> Response {
                Json(crate::tests::job_snapshot_json(&job_id, false)).into_response()
            }
            async fn progress_route(
                State(posts): State<Posts>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Response {
                posts.lock().unwrap().push(body);
                Json(crate::tests::job_snapshot_json(&job_id, false)).into_response()
            }
            async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
                Json(json!({
                    "id": worker_id, "gpuId": "mlx", "gpuName": null, "status": "busy",
                    "currentJobId": "job-supra2", "capabilities": [], "loadedModels": [],
                    "registeredAt": "2026-10-20T00:00:00Z", "lastSeenAt": "2026-10-20T00:00:00Z"
                }))
                .into_response()
            }
            let posts: Posts = Arc::default();
            let app = Router::new()
                .route("/api/v1/jobs/:job_id", get(job_route))
                .route("/api/v1/jobs/:job_id/progress", post(progress_route))
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

    /// sc-24215: the published controls and multi-image batch through the production arm of the
    /// linked lane, against a probe engine (no weights).
    mod controls_and_batch {
        use super::*;

        use crate::api_client::ApiClient;
        use axum::extract::{Path as AxumPath, State};
        use axum::response::{IntoResponse, Response};
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use sceneworks_core::contracts::JobSnapshot;
        use std::sync::{Arc, Mutex};

        type Posts = Arc<Mutex<Vec<Value>>>;

        /// A job API stub: the job GET reports `cancel_requested`, every progress POST is recorded.
        async fn spawn_job_api(cancel_requested: bool) -> (String, Posts) {
            async fn job_route(
                State((_, cancel)): State<(Posts, bool)>,
                AxumPath(job_id): AxumPath<String>,
            ) -> Response {
                Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
            }
            async fn progress_route(
                State((posts, cancel)): State<(Posts, bool)>,
                AxumPath(job_id): AxumPath<String>,
                Json(body): Json<Value>,
            ) -> Response {
                posts.lock().unwrap().push(body);
                Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
            }
            async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
                Json(json!({
                    "id": worker_id, "gpuId": "mlx", "gpuName": null, "status": "busy",
                    "currentJobId": "job-supra2", "capabilities": [], "loadedModels": [],
                    "registeredAt": "2026-10-20T00:00:00Z", "lastSeenAt": "2026-10-20T00:00:00Z"
                }))
                .into_response()
            }
            let posts: Posts = Arc::default();
            let app = Router::new()
                .route("/api/v1/jobs/:job_id", get(job_route))
                .route("/api/v1/jobs/:job_id/progress", post(progress_route))
                .route(
                    "/api/v1/workers/:worker_id/heartbeat",
                    post(heartbeat_route),
                )
                .with_state((posts.clone(), cancel_requested));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            (format!("http://{address}"), posts)
        }

        // ---------------------------------------------------------------------------------------
        // sc-24215: the published controls and multi-image batch through the production MLX arm,
        // against a probe engine (no weights): ONE engine call carries prompt / seed / steps /
        // guidance / count, and every image becomes its own asset with reproducibility metadata.
        // ---------------------------------------------------------------------------------------

        use gen_core::{GenerationOutput, GenerationRequest, Generator, Progress};
        use std::time::Duration;

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
            cancel_aware: bool,
            seen: Arc<Mutex<Vec<Seen>>>,
            stopped_on_cancel: Arc<Mutex<bool>>,
        }

        impl Generator for Supra2Probe {
            fn descriptor(&self) -> &gen_core::ModelDescriptor {
                &self.descriptor
            }

            fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
                Ok(())
            }

            /// Like the engine: progress per step, the request's own cancel flag checked, and
            /// `count` images back from the one call (image `k`'s pixels are all `k`).
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
                if self.cancel_aware {
                    // Outlive the consumer's 2-second cancel-poll throttle, then stop on the flag.
                    std::thread::sleep(Duration::from_millis(2_300));
                    let deadline = std::time::Instant::now() + Duration::from_secs(20);
                    while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    if req.cancel.is_cancelled() {
                        *self.stopped_on_cancel.lock().unwrap() = true;
                        return Err(gen_core::Error::Canceled);
                    }
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

        struct ArmRun {
            outcome: crate::error::WorkerResult<()>,
            seen: Vec<Seen>,
            stopped_on_cancel: bool,
            posts: Vec<Value>,
            asset_writes: Vec<Value>,
        }

        /// Drive `generate_stream_using` for one Supra2 job on the installed (staged) components;
        /// `extra` is merged into the job payload.
        fn run_supra2_arm(cancel_aware: bool, cancel_requested: bool, extra: Value) -> ArmRun {
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
            stage(
                data.path(),
                T5_REPO,
                T5_REVISION,
                &["config.json", "model.safetensors", "tokenizer.json"],
            );
            stage(
                data.path(),
                VAE_REPO,
                VAE_REVISION,
                &["config.json", "diffusion_pytorch_model.safetensors"],
            );
            let bundle = data.path().join("models/mlx/supra2_img");
            std::fs::create_dir_all(&bundle).unwrap();
            std::fs::write(bundle.join(gen_core::supra2::BUNDLE_MANIFEST), b"{}").unwrap();
            let mut entry = shipped_supra2_entry();
            entry.insert("modelPath".to_owned(), json!(bundle));
            let seen = Arc::new(Mutex::new(Vec::new()));
            let stopped_on_cancel = Arc::new(Mutex::new(false));
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let (outcome, posts, asset_writes) = runtime.block_on(async {
                let (base_url, posts) = spawn_job_api(cancel_requested).await;
                let mut settings = settings_at(data.path());
                settings.api_url = base_url;
                settings.heartbeat_seconds = 5;
                let api = ApiClient::new(&settings);
                let job: JobSnapshot =
                    serde_json::from_value(crate::tests::job_snapshot_json("job-supra2", false))
                        .unwrap();
                let mut payload = json!({
                    "projectId": "project-supra2",
                    "model": "supra2_img",
                    "mode": "text_to_image",
                    "prompt": "a red fox sitting in fresh snow",
                    "width": 256,
                    "height": 256,
                    "count": 1,
                    "modelManifestEntry": Value::Object(entry),
                });
                for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                    payload[key] = value;
                }
                let request = sceneworks_core::image_request::ImageRequest::from_payload(
                    payload.as_object().unwrap(),
                );
                let plan = super::super::super::ImagePlan::with_count(
                    &request,
                    request.image_count(),
                    None,
                );
                // The linked engine's own descriptor (`mlx_model` resolves the linked lane's row, as the
                // Iris arm test relies on).
                let descriptor = crate::engines::mlx_model("supra2_img")
                    .expect("supra2_img is linked")
                    .descriptor;
                let loader_seen = Arc::clone(&seen);
                let loader_stopped = Arc::clone(&stopped_on_cancel);
                let loader = move |_engine: &str, _spec: &LoadSpec| {
                    Ok(Box::new(Supra2Probe {
                        descriptor,
                        cancel_aware,
                        seen: loader_seen,
                        stopped_on_cancel: loader_stopped,
                    }) as Box<dyn Generator>)
                };
                let mut asset_writes = Vec::new();
                // The SAME assertions on both lanes: the MLX `generate_stream_using` on macOS, the
                // Candle `generate_candle_stream_using` off-Mac (sc-24214 registered the Candle engine).
                #[cfg(target_os = "macos")]
                let arm = super::super::super::generate_stream_using(
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
                let arm = super::super::super::generate_candle_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "cuda",
                    &mut asset_writes,
                    loader,
                );
                let outcome = tokio::time::timeout(Duration::from_secs(60), arm)
                    .await
                    .expect("the supra2 arm finished");
                let posts = posts.lock().unwrap().clone();
                (outcome, posts, asset_writes)
            });
            let seen = seen.lock().unwrap().clone();
            let stopped_on_cancel = *stopped_on_cancel.lock().unwrap();
            ArmRun {
                outcome,
                seen,
                stopped_on_cancel,
                posts,
                asset_writes,
            }
        }

        /// A count-3 job is ONE engine call carrying the prompt (no prompt batch), the base seed, the
        /// steps, the guidance and `count: 3`, published as three assets on seeds 40, 41, 42 — the
        /// seeds the engine drew image k's noise from (`image_seed(seed, k)`) — each recording its
        /// prompt, seed, steps, guidance, the pinned component revisions and the truncation warning.
        ///
        /// *Mutations that red this:* drop `image.nativeCountBatch` (three calls, count 1 each); send
        /// the one-prompt batch as `prompt_batch`; publish image k on the base seed.
        #[test]
        fn a_count_n_job_is_one_engine_call_published_as_one_asset_per_image() {
            let run = run_supra2_arm(
                false,
                false,
                json!({ "seed": 40, "count": 3, "advanced": { "steps": 12, "guidanceScale": 4.5 } }),
            );
            assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
            assert_eq!(
                run.seen,
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
            let entry = shipped_supra2_entry();
            let published: Vec<(Value, Value)> = run
                .asset_writes
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
            for asset in &run.asset_writes {
                assert_eq!(asset["prompt"], "a red fox sitting in fresh snow");
                assert_eq!(asset["model"], "supra2_img");
                assert_eq!(
                    (&asset["width"], &asset["height"]),
                    (&json!(256), &json!(256))
                );
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
            assert!(
                run.posts.iter().any(|post| post["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("Batch of 3 — step 1/12")),
                "progress names the batch: {:#?}",
                run.posts
            );
        }

        /// A seeded single image is the same engine request as image 0 of a batch on that seed: the
        /// seed + k contract means a count-1 job on seed 41 reproduces image 1 of the seed-40 batch.
        #[test]
        fn a_single_image_on_seed_k_is_the_batchs_image_k_request() {
            let run = run_supra2_arm(false, false, json!({ "seed": 41, "count": 1 }));
            assert!(run.outcome.is_ok(), "{:?}", run.outcome);
            assert_eq!(run.seen.len(), 1);
            assert_eq!((run.seen[0].seed, run.seen[0].count), (Some(41), 1));
            assert_eq!(
                gen_core::supra2::image_seed(40, 1),
                41,
                "image k of a batch draws its noise from seed + k"
            );
            assert_eq!(run.asset_writes.len(), 1);
            assert_eq!(run.asset_writes[0]["seed"], 41);
        }

        /// A cancel mid-batch publishes NOTHING: the one engine call stops on its own request flag.
        #[test]
        fn a_cancelled_batch_publishes_no_asset() {
            let run = run_supra2_arm(true, true, json!({ "seed": 40, "count": 4 }));
            assert!(
                matches!(run.outcome, Err(crate::error::WorkerError::Canceled(_))),
                "{:?}",
                run.outcome
            );
            assert_eq!(run.seen.len(), 1);
            assert!(run.stopped_on_cancel);
            assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        }

        /// sc-24215: the declared step ceiling. A request AT `limits.hardMaxSteps` reaches the engine
        /// with exactly that count (the generic 80-step clamp does not apply to Supra2).
        #[test]
        fn a_request_at_the_declared_step_ceiling_runs_exactly_that_count() {
            let ceiling = sceneworks_core::video_request::hard_max_steps(&shipped_supra2_entry())
                .expect("supra2_img declares limits.hardMaxSteps");
            let run = run_supra2_arm(
                false,
                false,
                json!({ "seed": 1, "advanced": { "steps": ceiling } }),
            );
            assert!(run.outcome.is_ok(), "{:?}", run.outcome);
            assert_eq!(run.seen.len(), 1);
            assert_eq!(run.seen[0].steps, Some(ceiling));
            assert_eq!(
                run.asset_writes[0]["rawAdapterSettings"]["numInferenceSteps"],
                ceiling
            );
        }

        /// ...and one above it is refused by the boundary `run_image_generate_job` applies before
        /// load, while every OTHER shipped model declares no ceiling and so keeps the lane's
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
            assert_eq!(super::super::super::max_requested_steps(50), 80);
        }
    }
}
