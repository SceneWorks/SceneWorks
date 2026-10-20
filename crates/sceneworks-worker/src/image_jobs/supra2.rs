//! Supra2-IMG wiring through the generic MLX image lane (sc-24213, epic 24212).
//!
//! The `supra2_img` engine loads from three local resources and nothing else:
//! `LoadSpec::weights` = the converted native bundle (the injected `modelPath`, produced at install
//! by the `supra2_img` converter in `model_jobs.rs`), and the two catalog co-requisites the
//! descriptor declares — `text_encoder` (google/flan-t5-base) and `vae` (stabilityai/sd-vae-ft-mse) —
//! staged cache-only at their pinned revisions by the generic `attach_required_components`. There is
//! no Supra2-specific production code on the generation path: routing, defaults (seed 0, CFG 3.0,
//! 50 steps, 256x256) and refusals all come from the catalog row and the linked descriptor. These
//! tests pin that wiring against the shipped catalog entry; the `#[ignore]`d real-weight test drives
//! one 256x256 render through the production MLX job arm with the real loader.

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
            .expect("supra2_img is linked on macOS");
        assert_eq!(descriptor.required_components, &["text_encoder", "vae"]);
        assert!(!descriptor.capabilities.supports_negative_prompt);
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
            crate::model_jobs::huggingface_pinned_snapshot_dir(data.path(), pin.repo, pin.revision)
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
            let plan = super::super::ImagePlan::with_count(&request, request.image_count(), None);
            let mut asset_writes = Vec::new();
            let outcome = super::super::generate_stream(
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
