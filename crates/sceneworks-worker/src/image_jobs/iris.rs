//! Iris-3B generation-task wiring (sc-25679, epic 25678).
//!
//! The `iris_3b` engine (`mlx-gen-iris`) loads the GENERATION task from two resources:
//! `LoadSpec::weights` = the `speridlabs/iris-3b` backbone root (`config.yaml` + `model.safetensors`)
//! and `LoadSpec::components["text_encoder"]` = the pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot. The
//! engine's descriptor advertises no `required_components` (the generic
//! `attach_required_components` seam therefore stages nothing for it), so the generation task's
//! resource set is read from the backend-neutral contract instead:
//! [`gen_core::iris::IrisTask::Generation`] names the text encoder as a resource and
//! [`gen_core::iris::TEXT_ENCODER_COMPONENT`] its component key. The component is resolved
//! cache-only through the same all-or-nothing `coRequisite` seam every other split model uses, so a
//! job never downloads at generation time and a missing encoder is a typed failure before load.
//!
//! The depth and restoration tasks (later stories) use [`IrisTask::uses_text_encoder`] = false and
//! never reach this staging.

use gen_core::iris::{IrisTask, FAMILY, GENERATION_MODEL_ID, TEXT_ENCODER_COMPONENT};
use gen_core::{LoadSpec, WeightsSource};
use sceneworks_core::contracts::JsonObject;
use serde_json::Value;

use crate::error::WorkerResult;
use crate::settings::Settings;

/// Is this registered engine id the Iris-3B generation route?
pub(crate) fn is_iris_generation_engine(engine_id: &str) -> bool {
    engine_id == GENERATION_MODEL_ID
}

/// The component keys the Iris generation task stages besides its backbone, from the shared
/// task contract (E4: only generation conditions on text).
fn generation_components() -> &'static [&'static str] {
    if IrisTask::Generation.uses_text_encoder() {
        &[TEXT_ENCODER_COMPONENT]
    } else {
        &[]
    }
}

/// Resolve the generation task's installed components (cache-only) for `model_id`'s catalog entry.
/// All-or-nothing: a missing or torn co-requisite is a typed `InvalidPayload` naming the component
/// and the Model Manager repair path, raised before any weights load.
pub(crate) fn resolve_generation_components(
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<Vec<(String, WeightsSource)>> {
    let descriptor = gen_core::ModelDescriptor {
        id: GENERATION_MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: gen_core::Modality::Image,
        capabilities: gen_core::Capabilities::default(),
        encoder_contract: None,
        denoiser_output_latent_space: None,
        required_components: generation_components(),
        control_kinds: None,
    };
    let manifest = Value::Object(manifest_entry.clone());
    let components = crate::model_jobs::resolve_co_requisites(&descriptor, &manifest, settings)
        .map_err(|error| match error {
            crate::error::WorkerError::InvalidPayload(message) => {
                crate::error::WorkerError::InvalidPayload(message.replacen(
                    GENERATION_MODEL_ID,
                    model_id,
                    1,
                ))
            }
            other => other,
        })?;
    Ok(components.into_iter().collect())
}

/// Stage the Iris generation task's components onto its `LoadSpec`. A no-op for every other
/// engine, and for a component the generic descriptor seam already staged.
pub(crate) fn attach_iris_generation_components(
    spec: LoadSpec,
    engine_id: &str,
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    if !is_iris_generation_engine(engine_id) {
        return Ok(spec);
    }
    let components = resolve_generation_components(model_id, manifest_entry, settings)?;
    Ok(components.into_iter().fold(spec, |spec, (id, source)| {
        if spec.components.contains_key(&id) {
            spec
        } else {
            spec.with_component(id, source)
        }
    }))
}

/// The manifest's `image.negativePromptRequiresGuidance` declaration: the negative prompt is the CFG
/// unconditional, so it only exists while guidance is on. ABSENT MEANS FALSE.
pub(crate) fn negative_prompt_requires_guidance(manifest_entry: &JsonObject) -> bool {
    manifest_entry
        .get("image")
        .and_then(|image| image.get("negativePromptRequiresGuidance"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Who a dropped knob belongs to, for the `sampling_knob_unsupported` event a drop emits (the same
/// fields [`super::normalize_sampling_knob`] reports).
pub(crate) struct KnobDropContext<'a> {
    pub(crate) model_id: &'a str,
    pub(crate) job_id: &'a str,
    pub(crate) engine: &'a str,
}

impl KnobDropContext<'_> {
    fn report(&self, knob: &str, requested: Value) {
        tracing::warn!(
            "{}: requested {knob} {requested} is not honored by {}; dropping it",
            self.engine,
            self.model_id
        );
        super::emit_sampling_knob_unsupported(
            knob,
            requested,
            &[],
            self.model_id,
            self.job_id,
            self.engine,
        );
    }
}

/// Drop the negative prompt when the model declares it CFG-bound and the resolved guidance turns CFG
/// off (exactly 1.0, the engine's `uses_cfg` test). The engine refuses a non-empty negative prompt
/// there rather than ignoring it, and with CFG off it has no effect on the image either way. A drop
/// of a prompt that was actually set emits `sampling_knob_unsupported` (knob `negativePrompt`).
pub(crate) fn gate_negative_prompt_on_guidance(
    manifest_entry: &JsonObject,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    drop: &KnobDropContext<'_>,
) -> Option<String> {
    if negative_prompt_requires_guidance(manifest_entry) && guidance == Some(1.0) {
        if let Some(dropped) = negative_prompt.filter(|value| !value.trim().is_empty()) {
            drop.report("negativePrompt", Value::String(dropped));
        }
        None
    } else {
        negative_prompt
    }
}

/// Iris has no scheduler axis yet: the engine refuses `scheduler_shift` by name. A stale or
/// API-supplied `schedulerShift` therefore never reaches it (the Image Studio hides the scheduler
/// controls for Iris, so a normal submit never carries one); a drop emits
/// `sampling_knob_unsupported` (knob `schedulerShift`). This is the ONE seam that withholds the
/// shift — delete it (and its call in `generate_stream`) when the engine honours `scheduler_shift`.
pub(crate) fn honored_scheduler_shift(
    engine_id: &str,
    scheduler_shift: Option<f32>,
    drop: &KnobDropContext<'_>,
) -> Option<f32> {
    if is_iris_generation_engine(engine_id) {
        if let Some(shift) = scheduler_shift {
            drop.report("schedulerShift", serde_json::json!(shift));
        }
        None
    } else {
        scheduler_shift
    }
}

/// The Iris text-encoder directory a resolved spec carries, for diagnostics/tests.
#[cfg(test)]
pub(crate) fn staged_text_encoder_dir(spec: &LoadSpec) -> Option<&std::path::Path> {
    match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(dir)) => Some(dir.as_path()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    const TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
    const TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

    fn settings_at(data_dir: &Path) -> Settings {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    /// Neutralize the ambient HF cache env so the cache resolves under the temp data dir.
    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    /// The shipped catalog entry, so the test exercises the real `coRequisite` declaration.
    fn shipped_iris_entry() -> JsonObject {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                .expect("parse builtin.models.jsonc");
        manifest
            .get("models")
            .and_then(Value::as_array)
            .and_then(|models| {
                models
                    .iter()
                    .find(|model| model.get("id").and_then(Value::as_str) == Some("iris_3b"))
            })
            .and_then(Value::as_object)
            .cloned()
            .expect("iris_3b is in the builtin catalog")
    }

    fn stage(root: &Path, repo: &str, revision: &str, file: &str) {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    fn stage_text_encoder(root: &Path) {
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            stage(root, TE_REPO, TE_REVISION, file);
        }
    }

    #[test]
    fn the_contract_names_only_the_text_encoder_for_generation() {
        assert_eq!(generation_components(), &["text_encoder"]);
        assert!(!IrisTask::Depth.uses_text_encoder());
        assert!(!IrisTask::Restoration.uses_text_encoder());
    }

    #[test]
    fn generation_stages_the_installed_text_encoder_snapshot() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        stage_text_encoder(data.path());
        let settings = settings_at(data.path());
        let backbone = data.path().join("backbone");
        let spec = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(backbone.clone())),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect("an installed encoder stages");
        let staged = staged_text_encoder_dir(&spec).expect("text_encoder staged as a Dir");
        assert!(
            staged.ends_with(format!("snapshots/{TE_REVISION}")),
            "{staged:?}"
        );
        assert_eq!(
            spec.components.len(),
            1,
            "only the generation task's encoder is staged"
        );
        assert_eq!(spec.weights, WeightsSource::Dir(backbone));
    }

    #[test]
    fn a_missing_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path());
        let error = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect_err("no encoder installed");
        let message = error.to_string();
        assert!(
            message.contains("text_encoder") && message.contains("not installed"),
            "{message}"
        );
    }

    #[test]
    fn a_torn_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        // Everything but the shards.
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            stage(data.path(), TE_REPO, TE_REVISION, file);
        }
        let error = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings_at(data.path()),
        )
        .expect_err("torn encoder");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn other_engines_are_untouched() {
        let data = tempfile::tempdir().unwrap();
        let spec = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().to_path_buf())),
            "mage_flow",
            "mage_flow",
            &JsonObject::new(),
            &settings_at(data.path()),
        )
        .expect("no-op");
        assert!(spec.components.is_empty());
    }

    const DROP: KnobDropContext<'static> = KnobDropContext {
        model_id: "iris_3b",
        job_id: "job-knob",
        engine: "mlx",
    };

    /// The `sampling_knob_unsupported` events `body` emitted, as `(knob, requested)` pairs.
    fn dropped_knobs<T>(body: impl FnOnce() -> T) -> (T, Vec<(String, Value)>) {
        let (value, logged) = crate::ladder_e2e_sc18101::with_captured_tracing(body);
        let events = logged
            .lines()
            .filter(|line| line.contains("sampling_knob_unsupported"))
            .filter_map(|line| {
                let payload = &line[line.find("sw_payload=")? + "sw_payload=".len()..];
                serde_json::from_str::<Value>(payload.trim()).ok()
            })
            .filter(|event| event["jobId"] == "job-knob")
            .map(|event| {
                assert_eq!(
                    (event["engine"].as_str(), event["model"].as_str()),
                    (Some("mlx"), Some("iris_3b")),
                    "{event}"
                );
                (
                    event["knob"].as_str().unwrap_or_default().to_owned(),
                    event["requested"].clone(),
                )
            })
            .collect();
        (value, events)
    }

    #[test]
    fn negative_prompt_is_dropped_only_when_cfg_is_off_for_a_declaring_model() {
        let iris = shipped_iris_entry();
        assert!(negative_prompt_requires_guidance(&iris));
        let negative = || Some("blurry".to_owned());
        assert_eq!(
            gate_negative_prompt_on_guidance(&iris, Some(1.0), negative(), &DROP),
            None
        );
        assert_eq!(
            gate_negative_prompt_on_guidance(&iris, Some(3.0), negative(), &DROP),
            negative()
        );
        let undeclared = json!({ "id": "other" }).as_object().unwrap().clone();
        assert_eq!(
            gate_negative_prompt_on_guidance(&undeclared, Some(1.0), negative(), &DROP),
            negative()
        );
    }

    #[test]
    fn dropping_a_set_negative_prompt_at_guidance_one_is_reported() {
        let iris = shipped_iris_entry();
        let (kept, events) = dropped_knobs(|| {
            gate_negative_prompt_on_guidance(&iris, Some(1.0), Some("blurry".to_owned()), &DROP)
        });
        assert_eq!(kept, None);
        assert_eq!(
            events,
            vec![("negativePrompt".to_owned(), json!("blurry"))],
            "a dropped negative prompt must emit sampling_knob_unsupported"
        );
    }

    #[test]
    fn an_empty_or_absent_or_honored_negative_prompt_is_not_reported() {
        let iris = shipped_iris_entry();
        let (_, events) = dropped_knobs(|| {
            gate_negative_prompt_on_guidance(&iris, Some(1.0), None, &DROP);
            gate_negative_prompt_on_guidance(&iris, Some(1.0), Some("  ".to_owned()), &DROP);
            gate_negative_prompt_on_guidance(&iris, Some(3.0), Some("blurry".to_owned()), &DROP);
        });
        assert_eq!(events, Vec::<(String, Value)>::new());
    }

    #[test]
    fn schedule_shift_never_reaches_iris() {
        assert_eq!(honored_scheduler_shift("iris_3b", Some(3.0), &DROP), None);
        assert_eq!(
            honored_scheduler_shift("sensenova_u1_8b", Some(3.0), &DROP),
            Some(3.0)
        );
    }

    #[test]
    fn a_dropped_schedule_shift_is_reported_and_an_absent_one_is_not() {
        let (kept, events) = dropped_knobs(|| honored_scheduler_shift("iris_3b", Some(3.0), &DROP));
        assert_eq!(kept, None);
        assert_eq!(
            events,
            vec![("schedulerShift".to_owned(), json!(3.0))],
            "a dropped schedule shift must emit sampling_knob_unsupported"
        );
        let (_, events) = dropped_knobs(|| {
            honored_scheduler_shift("iris_3b", None, &DROP);
            honored_scheduler_shift("sensenova_u1_8b", Some(3.0), &DROP);
        });
        assert_eq!(events, Vec::<(String, Value)>::new());
    }

    /// The generic lane derives what it forwards from the linked descriptor: Iris advertises no
    /// quant tier (so `mlx_candidate_quant` never sets `LoadSpec::quantize`), no sampler/scheduler
    /// names (so `normalize_sampling_knob` drops any requested one), and the catalog row offers none
    /// of those controls, so the request carries exactly the engine-honored surface.
    #[test]
    fn the_linked_descriptor_and_catalog_expose_only_honored_controls() {
        let model = crate::engines::mlx_model("iris_3b").expect("iris_3b is linked on macOS");
        assert_eq!(model.engine_id(), "iris_3b");
        assert!(!model.supports_quant());
        assert!(model.supports_guidance() && model.supports_negative_prompt());
        let caps = &model.descriptor.capabilities;
        assert!(caps.samplers.is_empty() && caps.schedulers.is_empty());
        assert!(!caps.supports_lora && !caps.supports_lokr && !caps.supports_true_cfg);
        assert_eq!(model.default_steps(), gen_core::iris::DEFAULT_STEPS);
        assert_eq!(model.default_guidance(), gen_core::iris::DEFAULT_CFG_SCALE);

        let entry = shipped_iris_entry();
        let limits = entry.get("limits").and_then(Value::as_object).unwrap();
        for absent in ["samplers", "schedulers", "guidanceMethods"] {
            assert!(!limits.contains_key(absent), "iris must not offer {absent}");
        }
        assert!(entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .is_none());
        assert_eq!(
            entry["loraCompatibility"]["families"]
                .as_array()
                .map(Vec::len),
            Some(0)
        );
        assert_eq!(
            limits["requiresDimensionsMultipleOf"].as_u64(),
            Some(u64::from(gen_core::iris::SIZE_MULTIPLE))
        );
        assert_eq!(
            limits["maxDimension"].as_u64(),
            Some(u64::from(caps.max_size))
        );
        assert_eq!(
            limits["minDimension"].as_u64(),
            Some(u64::from(caps.min_size))
        );
        assert_eq!(
            entry["defaults"]["steps"].as_u64(),
            Some(u64::from(gen_core::iris::DEFAULT_STEPS))
        );
    }

    /// Iris's release default (100 steps) is above the generic lane's historical 80-step ceiling;
    /// the ceiling rises to the model's own default so it is never clamped away.
    #[test]
    fn the_steps_ceiling_admits_the_models_own_default() {
        assert_eq!(
            super::super::max_requested_steps(gen_core::iris::DEFAULT_STEPS),
            gen_core::iris::DEFAULT_STEPS
        );
        assert_eq!(super::super::max_requested_steps(40), 80);
    }

    // ---------------------------------------------------------------------------------------
    // sc-25679 — the Iris job arm end to end with a stub engine.
    //
    // `generate_stream_using` is the production arm (`generate_stream` passes the real loader):
    // weights/encoder resolution, the tier/admission preflight, the cached load, the per-item drive
    // and `consume_gen_events` all run for real against a staged (weightless) cache. Only the engine
    // is a stub, so these pin what the worker does with the engine's Step progress, a user cancel
    // and an engine failure — not a re-implementation of the arm.
    // ---------------------------------------------------------------------------------------

    use crate::api_client::ApiClient;
    use axum::extract::{Path as AxumPath, State};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use gen_core::{GenerationOutput, GenerationRequest, Generator, Progress};
    use sceneworks_core::contracts::JobSnapshot;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const BACKBONE_REPO: &str = "speridlabs/iris-3b";
    const BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum ArmBehavior {
        /// Report a step, outlive the consumer's 2-second cancel-poll throttle, report another, then
        /// stop only when the request's own cancel flag trips.
        CancelAware,
        /// Report a step, then fail the way an engine refusal does.
        Fails,
    }

    struct IrisArmProbe {
        descriptor: gen_core::ModelDescriptor,
        behavior: ArmBehavior,
        generate_calls: Arc<Mutex<u32>>,
        /// The (guidance, negative prompt, schedule shift) the engine was handed.
        seen: Arc<Mutex<Option<SeenKnobs>>>,
        /// Set when the engine stopped because its OWN request cancel flag tripped.
        stopped_on_cancel: Arc<Mutex<bool>>,
    }

    type SeenKnobs = (Option<f32>, Option<String>, Option<f32>);

    impl Generator for IrisArmProbe {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        fn memory_strategy_safety_check(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::MemorySafetyDecision {
            gen_core::MemorySafetyDecision::Accept
        }

        fn begin_memory_strategy_request(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
            Ok(None)
        }

        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            *self.generate_calls.lock().unwrap() += 1;
            *self.seen.lock().unwrap() = Some((
                req.guidance,
                req.negative_prompt.clone(),
                req.scheduler_shift,
            ));
            on_progress(Progress::Step {
                current: 1,
                total: 100,
            });
            match self.behavior {
                ArmBehavior::Fails => Err(gen_core::Error::Unsupported(
                    "iris fixture engine failure".to_owned(),
                )),
                ArmBehavior::CancelAware => {
                    std::thread::sleep(Duration::from_millis(2_300));
                    on_progress(Progress::Step {
                        current: 2,
                        total: 100,
                    });
                    let deadline = std::time::Instant::now() + Duration::from_secs(20);
                    while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    if req.cancel.is_cancelled() {
                        *self.stopped_on_cancel.lock().unwrap() = true;
                        return Err(gen_core::Error::Canceled);
                    }
                    Ok(GenerationOutput::Images(vec![gen_core::Image {
                        width: req.width,
                        height: req.height,
                        pixels: vec![0; (req.width * req.height * 3) as usize],
                    }]))
                }
            }
        }
    }

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
                "id": worker_id, "gpuId": "cpu", "gpuName": null, "status": "busy",
                "currentJobId": "job-iris", "capabilities": [], "loadedModels": [],
                "registeredAt": "2026-07-01T00:00:00Z", "lastSeenAt": "2026-07-01T00:00:00Z"
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

    /// What the arm produced: its outcome, the progress posts, the asset writes, whether the
    /// stub engine ran, and any file written under the project.
    struct ArmRun {
        outcome: WorkerResult<()>,
        posts: Vec<Value>,
        asset_writes: Vec<Value>,
        generate_calls: u32,
        seen: Option<SeenKnobs>,
        stopped_on_cancel: bool,
        project_files: Vec<std::path::PathBuf>,
        /// Everything the arm logged (its worker events included).
        logged: String,
    }

    fn files_under(root: &Path) -> Vec<std::path::PathBuf> {
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

    /// Drive `generate_stream_using` for one iris text-to-image job; `extra` is merged into the
    /// job payload.
    fn run_iris_arm(behavior: ArmBehavior, cancel_requested: bool, extra: Value) -> ArmRun {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        // A weightless but complete install: the pinned backbone and its text encoder.
        for file in ["config.yaml", "model.safetensors"] {
            stage(data.path(), BACKBONE_REPO, BACKBONE_REVISION, file);
        }
        stage_text_encoder(data.path());
        for (repo, revision) in [(BACKBONE_REPO, BACKBONE_REVISION), (TE_REPO, TE_REVISION)] {
            let refs = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
                .unwrap()
                .join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            std::fs::write(refs.join("main"), revision).unwrap();
        }
        let calls = Arc::new(Mutex::new(0_u32));
        let seen = Arc::new(Mutex::new(None));
        let stopped_on_cancel = Arc::new(Mutex::new(false));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        // `block_on` polls the arm on THIS thread, so the thread-local capture sees its events.
        let ((outcome, posts, asset_writes), logged) =
            crate::ladder_e2e_sc18101::with_captured_tracing(|| {
                runtime.block_on(async {
                    let (base_url, posts) = spawn_job_api(cancel_requested).await;
                    let mut settings = settings_at(data.path());
                    settings.api_url = base_url;
                    settings.heartbeat_seconds = 5;
                    let api = ApiClient::new(&settings);
                    let job: JobSnapshot =
                        serde_json::from_value(crate::tests::job_snapshot_json("job-iris", false))
                            .unwrap();
                    let mut payload = json!({
                        "projectId": "project-iris",
                        "model": "iris_3b",
                        "mode": "text_to_image",
                        "prompt": "a lighthouse at dusk",
                        "width": 64,
                        "height": 64,
                        "count": 1,
                        "modelManifestEntry": Value::Object(shipped_iris_entry()),
                    });
                    for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                        payload[key] = value;
                    }
                    let request = sceneworks_core::image_request::ImageRequest::from_payload(
                        payload.as_object().unwrap(),
                    );
                    let plan = super::super::ImagePlan::with_count(&request, 1, None);
                    let descriptor = crate::engines::mlx_model("iris_3b")
                        .expect("iris_3b is linked")
                        .descriptor;
                    let loader_calls = Arc::clone(&calls);
                    let loader_seen = Arc::clone(&seen);
                    let loader_stopped = Arc::clone(&stopped_on_cancel);
                    let loader = move |_engine: &str, _spec: &LoadSpec| {
                        Ok(Box::new(IrisArmProbe {
                            descriptor,
                            behavior,
                            generate_calls: loader_calls,
                            seen: loader_seen,
                            stopped_on_cancel: loader_stopped,
                        }) as Box<dyn Generator>)
                    };
                    let mut asset_writes = Vec::new();
                    let outcome = tokio::time::timeout(
                        Duration::from_secs(60),
                        super::super::generate_stream_using(
                            &api,
                            &settings,
                            &job,
                            &plan,
                            project.path(),
                            "mlx",
                            &mut asset_writes,
                            loader,
                        ),
                    )
                    .await
                    .expect("the iris arm finished");
                    let posts = posts.lock().unwrap().clone();
                    (outcome, posts, asset_writes)
                })
            });
        let generate_calls = *calls.lock().unwrap();
        let seen = seen.lock().unwrap().clone();
        let stopped_on_cancel = *stopped_on_cancel.lock().unwrap();
        ArmRun {
            outcome,
            posts,
            asset_writes,
            generate_calls,
            seen,
            stopped_on_cancel,
            project_files: files_under(project.path()),
            logged,
        }
    }

    fn message(post: &Value) -> &str {
        post["message"].as_str().unwrap_or_default()
    }

    /// The first post reporting the engine's step 1/100, asserted NON-terminal `generating`.
    fn forwarded_step(posts: &[Value]) -> usize {
        let index = posts
            .iter()
            .position(|post| message(post).contains("step 1/100"))
            .unwrap_or_else(|| panic!("the engine's Step must reach the job API: {posts:#?}"));
        assert_eq!(
            (
                posts[index]["status"].as_str(),
                posts[index]["stage"].as_str()
            ),
            (Some("running"), Some("generating")),
            "{posts:#?}"
        );
        index
    }

    #[test]
    fn the_iris_arm_forwards_steps_and_a_user_cancel_ends_canceled_with_no_asset() {
        let run = run_iris_arm(ArmBehavior::CancelAware, true, json!({}));
        assert!(
            matches!(run.outcome, Err(crate::error::WorkerError::Canceled(_))),
            "a user-cancelled iris job ends Canceled, got {:?}",
            run.outcome
        );
        assert_eq!(run.generate_calls, 1, "the stub engine ran exactly once");
        assert!(
            run.stopped_on_cancel,
            "the user cancel must reach the engine's request flag, so the denoise stops"
        );
        let step = forwarded_step(&run.posts);
        let acknowledged = run
            .posts
            .iter()
            .position(|post| message(post).contains("Cancelling"))
            .unwrap_or_else(|| panic!("the cancel must be acknowledged: {:#?}", run.posts));
        assert!(acknowledged > step, "{:#?}", run.posts);
        assert_eq!(
            run.posts.last().unwrap()["status"],
            "canceled",
            "{:#?}",
            run.posts
        );
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    #[test]
    fn an_iris_engine_failure_is_a_typed_failure_with_no_asset() {
        let run = run_iris_arm(ArmBehavior::Fails, false, json!({}));
        match &run.outcome {
            Err(crate::error::WorkerError::Engine(message)) => assert!(
                message.contains("iris fixture engine failure"),
                "the engine's reason survives: {message}"
            ),
            other => panic!("an engine failure must be a typed Engine error, got {other:?}"),
        }
        assert_eq!(run.generate_calls, 1);
        forwarded_step(&run.posts);
        assert!(
            run.posts.iter().all(|post| post["status"] != "completed"),
            "{:#?}",
            run.posts
        );
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    /// The `sampling_knob_unsupported` (knob, requested) pairs a job logged for `job_id`.
    fn knob_events(logged: &str, job_id: &str) -> Vec<(String, Value)> {
        logged
            .lines()
            .filter(|line| line.contains("sampling_knob_unsupported"))
            .filter_map(|line| {
                let payload = &line[line.find("sw_payload=")? + "sw_payload=".len()..];
                serde_json::from_str::<Value>(payload.trim()).ok()
            })
            .filter(|event| event["jobId"] == job_id)
            .map(|event| {
                (
                    event["knob"].as_str().unwrap_or_default().to_owned(),
                    event["requested"].clone(),
                )
            })
            .collect()
    }

    /// The real arm drops a negative prompt at guidance 1.0 and any schedule shift BEFORE the
    /// engine, and reports each drop — not just the gate functions in isolation.
    #[test]
    fn the_iris_arm_withholds_and_reports_refused_knobs() {
        let run = run_iris_arm(
            ArmBehavior::Fails,
            false,
            json!({
                "negativePrompt": "blurry",
                "advanced": { "guidanceScale": 1.0, "schedulerShift": 3.0 },
            }),
        );
        assert_eq!(
            run.seen,
            Some((Some(1.0), None, None)),
            "the engine must see CFG off with neither refused knob"
        );
        let mut events = knob_events(&run.logged, "job-iris");
        events.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            events,
            vec![
                ("negativePrompt".to_owned(), json!("blurry")),
                ("schedulerShift".to_owned(), json!(3.0)),
            ],
            "{}",
            run.logged
        );

        // The same knobs the engine honours pass through, unreported.
        let honored = run_iris_arm(
            ArmBehavior::Fails,
            false,
            json!({ "negativePrompt": "blurry", "advanced": { "guidanceScale": 3.0 } }),
        );
        assert_eq!(
            honored.seen,
            Some((Some(3.0), Some("blurry".to_owned()), None))
        );
        assert_eq!(knob_events(&honored.logged, "job-iris"), Vec::new());
    }
}
