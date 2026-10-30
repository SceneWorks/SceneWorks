//! Iris-3B generation-task wiring (sc-25679, epic 25678).
//!
//! The `iris_3b` engine loads the GENERATION task from two resources: `LoadSpec::weights` = the
//! `speridlabs/iris-3b` backbone root (`config.yaml` + `model.safetensors`) and
//! `LoadSpec::components["text_encoder"]` = the pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot. The
//! engine's descriptor declares `required_components: [TEXT_ENCODER_COMPONENT]`, so the generic
//! `attach_required_components` seam stages the encoder cache-only through the catalog's
//! all-or-nothing `coRequisite` (`componentId: "text_encoder"`): a job never downloads at
//! generation time, and a missing or torn encoder is a typed failure before load (pinned by the
//! tests below against the shipped catalog entry).
//!
//! The Iris request controls (sampler = solver order, schedule shift, CFG interval, caption-overflow
//! policy, CFG-bound negative prompt, adapters, previews; sc-25681) ride the GENERIC image lane:
//! [`super::lane_controls`] resolves and refuses them for every engine against its linked
//! descriptor, so nothing here is Iris-only production code. This module pins the Iris job arm end
//! to end on both lanes with a recording stub engine.

/// The Iris text-encoder directory a resolved spec carries, for diagnostics/tests.
#[cfg(test)]
pub(crate) fn staged_text_encoder_dir(spec: &gen_core::LoadSpec) -> Option<&std::path::Path> {
    match spec.components.get(gen_core_iris::TEXT_ENCODER_COMPONENT) {
        Some(gen_core::WeightsSource::Dir(dir)) => Some(dir.as_path()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WorkerResult;
    use crate::settings::Settings;
    use gen_core::{LoadSpec, WeightsSource};
    use gen_core_iris::{IrisTask, TEXT_ENCODER_COMPONENT};
    use sceneworks_core::contracts::JsonObject;
    use serde_json::json;
    use serde_json::Value;
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
        stage_bytes(root, repo, revision, file, b"x");
    }

    fn stage_bytes(root: &Path, repo: &str, revision: &str, file: &str, bytes: &[u8]) {
        let path = snapshot_file(root, repo, revision, file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// The text encoder as a snapshot both engines can PRICE (sc-25690): the memory-strategy
    /// contract reads the tower config and every indexed shard's header before any load, so the
    /// arm's pre-load admission needs real headers, not placeholder bytes.
    fn stage_text_encoder(root: &Path) {
        for (file, bytes) in crate::iris_test_weights::priceable_text_encoder_files() {
            stage_bytes(root, TE_REPO, TE_REVISION, file, &bytes);
        }
        #[cfg(target_os = "macos")]
        if let Some(gib) = PRICED_COMPONENT_GIB.with(std::cell::Cell::get) {
            // bf16 as stored: 64 bytes a row.
            crate::iris_test_weights::write_sparse_zero_tensors(
                &snapshot_file(
                    root,
                    TE_REPO,
                    TE_REVISION,
                    crate::iris_test_weights::TEXT_ENCODER_LANGUAGE_SHARD,
                ),
                &[(
                    crate::iris_test_weights::TEXT_ENCODER_LANGUAGE_TENSOR,
                    "BF16",
                    &[gib * (1 << 30) / 64, 32],
                )],
            );
        }
    }

    fn snapshot_file(root: &Path, repo: &str, revision: &str, file: &str) -> std::path::PathBuf {
        sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision)
            .join(file)
    }

    #[cfg(target_os = "macos")]
    thread_local! {
        /// GiB each of the staged text encoder's language tower and the backbone PRICE at (bf16
        /// resident) for the arm runs on THIS thread. `None` keeps the miniature install, whose
        /// weights are too small for staging to save anything: a staged run holds
        /// `max(tower, backbone)` where a resident one holds the sum. The files are sparse — the
        /// pricing reads headers, and the stub engine reads nothing.
        static PRICED_COMPONENT_GIB: std::cell::Cell<Option<usize>> =
            const { std::cell::Cell::new(None) };
    }

    /// The generation backbone as a root both engines can PRICE: the release `config.yaml` and a
    /// header-valid `model.safetensors`.
    fn stage_backbone(root: &Path) {
        stage_bytes(
            root,
            BACKBONE_REPO,
            BACKBONE_REVISION,
            "config.yaml",
            crate::iris_test_weights::RELEASE_CONFIG_YAML.as_bytes(),
        );
        stage_bytes(
            root,
            BACKBONE_REPO,
            BACKBONE_REVISION,
            "model.safetensors",
            &crate::iris_test_weights::priceable_backbone_weights(),
        );
        #[cfg(target_os = "macos")]
        if let Some(gib) = PRICED_COMPONENT_GIB.with(std::cell::Cell::get) {
            // f32 as stored, cast to bf16 on load: 16 stored bytes price as 8.
            crate::iris_test_weights::write_sparse_zero_tensors(
                &snapshot_file(root, BACKBONE_REPO, BACKBONE_REVISION, "model.safetensors"),
                &[("blocks.0.attn.qkv.weight", "F32", &[gib * (1 << 30) / 8, 4])],
            );
        }
    }

    #[test]
    fn the_contract_names_only_the_text_encoder_for_generation() {
        let descriptor = crate::inference_runtime::media_descriptor("iris_3b")
            .expect("iris_3b is linked on macOS");
        assert_eq!(descriptor.required_components, &[TEXT_ENCODER_COMPONENT]);
        assert_eq!(TEXT_ENCODER_COMPONENT, "text_encoder");
        assert!(IrisTask::Generation.uses_text_encoder());
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
        let spec = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(backbone.clone())),
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
        let error = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
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
        let error = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            &shipped_iris_entry(),
            &settings_at(data.path()),
        )
        .expect_err("torn encoder");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    /// Run `body` under an INFO subscriber whose output is captured, so a test can read the worker
    /// events it emitted. Local (not `ladder_e2e_sc18101`'s macOS-only twin) because these tests run
    /// on both lanes.
    fn with_captured_tracing<T>(body: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        crate::test_env::install_tracing_interest_floor();
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_target(false)
            .without_time()
            .finish();
        let value = tracing::subscriber::with_default(subscriber, body);
        let text = String::from_utf8_lossy(&capture.0.lock().unwrap()).into_owned();
        (value, text)
    }

    /// The catalog row's honored-controls surface is EXACTLY the linked engine's (sc-25681): the
    /// sampler menu is the descriptor's solver menu, there is no scheduler menu but a declared shift
    /// axis, the CFG interval / caption-overflow declarations mirror the descriptor's capability
    /// flags, adapters are offered iff the engine applies them (generation-task stamps only), and
    /// the bf16 / q8 / q4 tier rows are exactly the engine's load-time tiers (sc-26266): one
    /// shared download, q8 / q4 macOS-only, bf16 the default and the no-pick tier.
    ///
    /// *Mutation that reds this:* drop `euler` from `limits.samplers`, or `supportsCfgInterval`.
    #[test]
    fn the_catalog_row_offers_exactly_the_linked_engines_controls() {
        let model = crate::engines::mlx_model("iris_3b").expect("iris_3b is linked");
        assert_eq!(model.engine_id(), "iris_3b");
        assert!(model.supports_guidance() && model.supports_negative_prompt());
        let caps = &model.descriptor.capabilities;
        assert_eq!(model.default_steps(), gen_core_iris::DEFAULT_STEPS);
        assert_eq!(model.default_guidance(), gen_core_iris::DEFAULT_CFG_SCALE);

        let entry = shipped_iris_entry();
        let limits = entry.get("limits").and_then(Value::as_object).unwrap();
        let samplers: Vec<&str> = limits["samplers"]
            .as_array()
            .expect("iris declares its solver menu")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(samplers, caps.samplers, "the menu IS the engine's");
        assert_eq!(samplers, gen_core_iris::SAMPLERS);
        assert_eq!(
            entry["defaults"]["sampler"].as_str(),
            Some(gen_core_iris::SOLVER_ORDER_2_SAMPLER)
        );
        assert!(caps.schedulers.is_empty());
        for absent in ["schedulers", "guidanceMethods"] {
            assert!(!limits.contains_key(absent), "iris must not offer {absent}");
        }
        assert!(
            sceneworks_core::image_sampling_controls::supports_scheduler_shift(&entry),
            "the shift axis is declared without a scheduler menu"
        );
        assert!(entry["defaults"]["schedulerShift"].as_f64().unwrap() > 0.0);
        assert_eq!(
            sceneworks_core::image_sampling_controls::supports_cfg_interval(&entry),
            caps.supports_cfg_interval
        );
        assert_eq!(
            !sceneworks_core::image_sampling_controls::caption_overflow_policies(&entry).is_empty(),
            caps.supports_caption_overflow_policy
        );
        assert!(
            sceneworks_core::image_sampling_controls::negative_prompt_requires_guidance(&entry)
        );
        assert!(caps.supports_preview, "previews are wired generically");
        assert!(caps.supports_lora && caps.supports_lokr);
        assert_eq!(entry["loraCompatibility"]["families"], json!(["iris"]));
        assert_eq!(
            entry["loraCompatibility"]["irisTask"].as_str(),
            Some(IrisTask::Generation.name())
        );
        // sc-26266: both engines advertise q8/q4, loaded from the same dense upstream files.
        assert_eq!(
            caps.supported_quants,
            [gen_core::Quant::Q8, gen_core::Quant::Q4]
        );
        assert_eq!(
            entry["mlx"]["quantize"].as_i64(),
            Some(0),
            "no pick loads bf16"
        );
        let tiers: Vec<(&str, Option<&Value>, bool, &Value)> = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|download| download.get("componentId").is_none())
            .map(|download| {
                (
                    download["variant"]
                        .as_str()
                        .expect("every backbone row names a tier"),
                    download.get("platforms"),
                    download.get("default").and_then(Value::as_bool) == Some(true),
                    &download["files"],
                )
            })
            .collect();
        assert_eq!(
            tiers.iter().map(|tier| tier.0).collect::<Vec<_>>(),
            ["bf16", "q8", "q4"]
        );
        for (tier, platforms, default, files) in &tiers {
            assert_eq!(
                *files,
                &json!(["config.yaml", "model.safetensors"]),
                "{tier}"
            );
            assert_eq!(*default, *tier == "bf16", "{tier}");
            assert!(platforms.is_none(), "all native platforms serve {tier}");
        }
        assert_eq!(
            limits["requiresDimensionsMultipleOf"].as_u64(),
            Some(u64::from(gen_core_iris::SIZE_MULTIPLE))
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
            Some(u64::from(gen_core_iris::DEFAULT_STEPS))
        );
        assert_eq!(
            entry["defaults"]["resolution"].as_str(),
            Some(format!("{0}x{0}", gen_core_iris::DEFAULT_SIZE).as_str()),
            "the default stays 1024-class"
        );
        assert_eq!(
            entry["image"]["patchGridVisibleBelow"].as_u64(),
            Some(u64::from(gen_core_iris::DEFAULT_SIZE))
        );
    }

    /// Iris's release default (100 steps) is above the generic lane's historical 80-step ceiling;
    /// the ceiling rises to the model's own default so it is never clamped away.
    #[test]
    fn the_steps_ceiling_admits_the_models_own_default() {
        assert_eq!(
            super::super::max_requested_steps(gen_core_iris::DEFAULT_STEPS),
            gen_core_iris::DEFAULT_STEPS
        );
        assert_eq!(super::super::max_requested_steps(40), 80);
    }

    // ---------------------------------------------------------------------------------------
    // sc-25679 — the Iris job arm end to end with a stub engine.
    //
    // `generate_stream_using` (MLX) / `generate_candle_stream_using` (Candle, sc-25680) is the
    // production arm (the plain names pass the real loader):
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
        /// Report a step, emit one (W/16)×(H/16) x0 preview frame the way the Iris engine does every
        /// step, report a second step (the post the preview rides), then complete with an image and
        /// a `caption_truncated` warning through `generate_with_report`.
        Completes,
        /// [`Self::Completes`], but the engine's request scope admits ONE IMAGE FEWER than the
        /// geometry the worker admitted — the engine refusing a request wider than its admission.
        UnderAdmits,
    }

    /// The request scope the stub engine opens: the Iris engines' own rule and nothing else. Both
    /// `candle-gen-iris` and `mlx-gen-iris` bound a request's images (`prompts × count`) by the
    /// admitted `MemoryGeometry::batch` in `configure_request`, because a prompt batch denoises its
    /// rows together.
    struct ProbeScope {
        admitted_images: u32,
    }

    impl gen_core::MemoryRequestScope for ProbeScope {
        fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
            let images = request.prompt_batch.len().max(1) as u64 * u64::from(request.count);
            if images > u64::from(self.admitted_images) {
                return Err(gen_core::Error::Unsupported(format!(
                    "iris_3b: the request renders {images} images but was admitted for {}",
                    self.admitted_images
                )));
            }
            Ok(())
        }

        fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }

        fn configure_decode(
            &mut self,
            _edge: u32,
            _overlap: u32,
            _geometry: gen_core::MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }

        fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
            Ok(())
        }

        fn materialize_transformer_window(
            &mut self,
            _first_block: u32,
            _block_count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }

        fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
            Ok(())
        }
    }

    /// What the engine was handed: every request field the Iris controls map onto.
    #[derive(Clone, Debug, PartialEq)]
    struct Seen {
        prompt: String,
        negative_prompt: Option<String>,
        width: u32,
        height: u32,
        seed: Option<u64>,
        steps: Option<u32>,
        guidance: Option<f32>,
        sampler: Option<String>,
        scheduler: Option<String>,
        scheduler_shift: Option<f32>,
        cfg_interval: Option<(f32, f32)>,
        caption_overflow: Option<gen_core::CaptionOverflowPolicy>,
        prompt_batch: Vec<String>,
        count: u32,
    }

    impl Seen {
        fn of(req: &GenerationRequest) -> Self {
            Self {
                prompt: req.prompt.clone(),
                negative_prompt: req.negative_prompt.clone(),
                width: req.width,
                height: req.height,
                seed: req.seed,
                steps: req.steps,
                guidance: req.guidance,
                sampler: req.sampler.clone(),
                scheduler: req.scheduler.clone(),
                scheduler_shift: req.scheduler_shift,
                cfg_interval: req.cfg_interval,
                caption_overflow: req.caption_overflow,
                prompt_batch: req.prompt_batch.clone(),
                count: req.count,
            }
        }
    }

    const FIXTURE_WARNING: &str = "iris: caption overflow — 7 caption tokens dropped (fixture)";

    struct IrisArmProbe {
        descriptor: gen_core::ModelDescriptor,
        behavior: ArmBehavior,
        generate_calls: Arc<Mutex<u32>>,
        /// Every request the engine was handed, in order.
        seen: Arc<Mutex<Vec<Seen>>>,
        /// Set when the engine stopped because its OWN request cancel flag tripped.
        stopped_on_cancel: Arc<Mutex<bool>>,
        /// The adapters the loader handed this engine (`LoadSpec::adapters`).
        adapters: Vec<gen_core::AdapterSpec>,
        /// Every memory-strategy context the arm opened a request scope under, in order.
        contexts: Arc<Mutex<Vec<gen_core::MemoryRunContext>>>,
        /// The contract the REAL engine publishes for the spec this probe was loaded with. The
        /// MLX arm grades the loaded request against the generator's own contract, so a probe
        /// without one would be graded on the resident baseline alone and never offered the
        /// optimized rungs the pre-load gate priced.
        #[cfg(target_os = "macos")]
        contract: Option<gen_core::MemoryProviderContract>,
    }

    impl Generator for IrisArmProbe {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        /// One report per loaded adapter, the way the Iris engine publishes its install outcome:
        /// everything applied, one stem skipped.
        fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
            self.adapters
                .iter()
                .map(|adapter| gen_core::AdapterApplyReport {
                    adapter_path: adapter.path.clone(),
                    applied: 24,
                    skipped: vec!["pixel.blocks.0.attn".to_owned()],
                })
                .collect()
        }

        #[cfg(target_os = "macos")]
        fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
            self.contract.as_ref()
        }

        fn memory_strategy_safety_check(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::MemorySafetyDecision {
            gen_core::MemorySafetyDecision::Accept
        }

        fn begin_memory_strategy_request(
            &self,
            context: &gen_core::MemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
            self.contexts.lock().unwrap().push(context.clone());
            let admitted_images = match self.behavior {
                ArmBehavior::UnderAdmits => context.geometry.batch.saturating_sub(1),
                _ => context.geometry.batch,
            };
            Ok(Some(Box::new(ProbeScope { admitted_images })))
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
            self.seen.lock().unwrap().push(Seen::of(req));
            on_progress(Progress::Step {
                current: 1,
                total: 100,
            });
            // One image per prompt per `count` repetition, like the engine: a native prompt batch
            // returns `prompt_batch.len() × count`, an ordinary request `count` (always 1 here).
            // Image `k`'s pixels are all `k`, so the test can see which asset got which image.
            let image = || {
                let images = req.prompt_batch.len().max(1) * req.count as usize;
                GenerationOutput::Images(
                    (0..images)
                        .map(|k| gen_core::Image {
                            width: req.width,
                            height: req.height,
                            pixels: vec![k as u8; (req.width * req.height * 3) as usize],
                        })
                        .collect(),
                )
            };
            match self.behavior {
                ArmBehavior::Fails => Err(gen_core::Error::Unsupported(
                    "iris fixture engine failure".to_owned(),
                )),
                ArmBehavior::Completes | ArmBehavior::UnderAdmits => {
                    req.preview.emit(gen_core::PreviewFrame {
                        current: 1,
                        total: 100,
                        image: gen_core::Image {
                            width: req.width / 16,
                            height: req.height / 16,
                            pixels: vec![128; ((req.width / 16) * (req.height / 16) * 3) as usize],
                        },
                    });
                    on_progress(Progress::Step {
                        current: 2,
                        total: 100,
                    });
                    Ok(image())
                }
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
                    Ok(image())
                }
            }
        }

        /// The Iris engine's primary path: the output plus its `caption_truncated` warnings.
        fn generate_with_report(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<gen_core::GenerationReport> {
            let mut report =
                gen_core::GenerationReport::from_output(self.generate(req, on_progress)?);
            if self.behavior == ArmBehavior::Completes {
                report.warnings.push(gen_core::GenerationWarning {
                    code: "caption_truncated".to_owned(),
                    message: FIXTURE_WARNING.to_owned(),
                });
            }
            Ok(report)
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
        /// How many times the arm called the loader (0 ⇒ refused before load).
        load_calls: u32,
        posts: Vec<Value>,
        asset_writes: Vec<Value>,
        generate_calls: u32,
        seen: Vec<Seen>,
        /// The adapter stack the loader was handed (`LoadSpec::adapters`).
        loaded_adapters: Vec<gen_core::AdapterSpec>,
        /// The residency policy the loader was handed (`LoadSpec::offload_policy`); read by the
        /// Candle-only staged-admission test.
        #[cfg_attr(target_os = "macos", allow(dead_code))]
        loaded_offload: Option<gen_core::OffloadPolicy>,
        /// The materialization shape the loader was handed (`LoadSpec::load_shape`).
        loaded_shape: Option<gen_core::LoadShape>,
        /// The tier the loader was handed (`LoadSpec::quantize`; `None` = bf16).
        #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
        loaded_quant: Option<Option<gen_core::Quant>>,
        stopped_on_cancel: bool,
        /// Every memory-strategy context the arm opened an engine request scope under.
        #[cfg_attr(target_os = "macos", allow(dead_code))]
        memory_contexts: Vec<gen_core::MemoryRunContext>,
        project_files: Vec<std::path::PathBuf>,
        /// The task switch around the job (sc-25684), recorded only by [`run_iris_task_switch`].
        switch: Option<TaskSwitch>,
    }

    /// What the process-global caches held around one generation job that started with a
    /// restoration transform resident and was followed by a restoration cold load.
    #[derive(Debug)]
    struct TaskSwitch {
        /// The transform was still resident after the generation job.
        transform_resident_after_generation: bool,
        /// The resident generator's component ids after the generation job (`None` = none).
        generator_after_generation: Option<Vec<String>>,
        /// The resident generator's component ids after the restoration cold load.
        generator_after_restoration: Option<Vec<String>>,
        /// The transform was resident after the restoration cold load.
        transform_resident_after_restoration: bool,
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
        run_iris_arm_with_memory_cap(behavior, cancel_requested, extra, "64")
    }

    /// [`run_iris_arm`] against an emulated `memory_cap_gb` budget: the MLX unified-memory cap on
    /// macOS, the `SCENEWORKS_CUDA_VRAM_CAP_GB` card cap on the Candle lane.
    fn run_iris_arm_with_memory_cap(
        behavior: ArmBehavior,
        cancel_requested: bool,
        extra: Value,
        memory_cap_gb: &str,
    ) -> ArmRun {
        run_iris_arm_with(behavior, cancel_requested, extra, memory_cap_gb, &[])
    }

    /// One adapter file the arm installs under the data dir's `loras/` before the job runs: its
    /// catalog id, its safetensors `__metadata__`, and the job weight.
    struct TestAdapter {
        id: &'static str,
        metadata: Value,
        weight: f64,
    }

    /// Write a minimal adapter with `metadata` that both engines can PRICE against the staged
    /// backbone's one `[4, 4]` projection (sc-25690: the memory contract reads every adapter's
    /// header before load and refuses factors that match no projection): a rank-1 LoRA pair, or —
    /// when the metadata stamps `networkType: lokr` — two `[2, 2]` Kronecker factors.
    fn write_adapter(path: &Path, metadata: &Value) {
        let target = "blocks.0.attn.qkv";
        let lokr = metadata.get("networkType").and_then(Value::as_str) == Some("lokr");
        let factors: [(String, [usize; 2]); 2] = if lokr {
            [
                (format!("{target}.lokr_w1"), [2, 2]),
                (format!("{target}.lokr_w2"), [2, 2]),
            ]
        } else {
            [
                (format!("{target}.lora_down.weight"), [1, 4]),
                (format!("{target}.lora_up.weight"), [4, 1]),
            ]
        };
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), metadata.clone());
        let mut offset = 0_usize;
        for (name, shape) in factors {
            let bytes = shape.iter().product::<usize>() * 4;
            header.insert(
                name,
                json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes] }),
            );
            offset += bytes;
        }
        let header = Value::Object(header).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// [`run_iris_arm_with_memory_cap`] with `adapters` installed and selected on the job.
    fn run_iris_arm_with(
        behavior: ArmBehavior,
        cancel_requested: bool,
        extra: Value,
        memory_cap_gb: &str,
        adapters: &[TestAdapter],
    ) -> ArmRun {
        run_iris_arm_inner(
            behavior,
            cancel_requested,
            extra,
            memory_cap_gb,
            adapters,
            false,
        )
    }

    /// One completed generation job between two restoration loads (sc-25684): a restoration
    /// transform is made resident in the process-global transform cache first, the iris arm runs,
    /// then a restoration transform cold-loads again — all under one env lock so no other test
    /// touches the global caches in between.
    fn run_iris_task_switch() -> ArmRun {
        run_iris_arm_inner(ArmBehavior::Completes, false, json!({}), "64", &[], true)
    }

    fn run_iris_arm_inner(
        behavior: ArmBehavior,
        cancel_requested: bool,
        extra: Value,
        memory_cap_gb: &str,
        adapters: &[TestAdapter],
        task_switch: bool,
    ) -> ArmRun {
        // The arm admits against the host's memory before loading; pin an emulated budget so the
        // outcome does not depend on the host (a 7 GiB hosted runner otherwise refuses the request
        // before the arm under test is reached, and a GPU-less runner yields no budget at all).
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            #[cfg(target_os = "macos")]
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, memory_cap_gb),
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            (crate::vram_gate::CUDA_VRAM_CAP_ENV, memory_cap_gb),
        ]);
        let data = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        // A weightless but complete AND priceable install: the pinned backbone and its text
        // encoder, with the config and tensor headers the pre-load memory contract reads.
        stage_backbone(data.path());
        stage_text_encoder(data.path());
        for (repo, revision) in [(BACKBONE_REPO, BACKBONE_REVISION), (TE_REPO, TE_REVISION)] {
            let refs = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
                .unwrap()
                .join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            std::fs::write(refs.join("main"), revision).unwrap();
        }
        let loras: Vec<Value> = adapters
            .iter()
            .map(|adapter| {
                let path = data
                    .path()
                    .join("loras")
                    .join(adapter.id)
                    .join("adapter.safetensors");
                write_adapter(&path, &adapter.metadata);
                json!({ "id": adapter.id, "path": path, "weight": adapter.weight })
            })
            .collect();
        let calls = Arc::new(Mutex::new(0_u32));
        let loads = Arc::new(Mutex::new(0_u32));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let loaded_adapters = Arc::new(Mutex::new(Vec::new()));
        let loaded_offload = Arc::new(Mutex::new(None));
        let loaded_shape = Arc::new(Mutex::new(None));
        let loaded_quant = Arc::new(Mutex::new(None));
        let stopped_on_cancel = Arc::new(Mutex::new(false));
        let memory_contexts = Arc::new(Mutex::new(Vec::new()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        // `block_on` polls the arm on THIS thread, so the thread-local capture sees its events.
        let restoration_export = tempfile::tempdir().unwrap();
        let ((outcome, posts, asset_writes, switch), _logged) = with_captured_tracing(|| {
            runtime.block_on(async {
                if task_switch {
                    crate::transform_cache::load_stub_into_global_cache(restoration_export.path())
                        .await
                        .unwrap();
                    assert!(crate::transform_cache::global_transform_resident().await);
                }
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
                    "loras": loras,
                    "modelManifestEntry": Value::Object(shipped_iris_entry()),
                });
                for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                    payload[key] = value;
                }
                let request = sceneworks_core::image_request::ImageRequest::from_payload(
                    payload.as_object().unwrap(),
                );
                // The generic routes' image count (`ImageRoute::Mlx` / `CandleTxt2Img`): `count`,
                // or `prompts × count` for a native prompt batch.
                let plan =
                    super::super::ImagePlan::with_count(&request, request.image_count(), None);
                let descriptor = crate::engines::mlx_model("iris_3b")
                    .expect("iris_3b is linked")
                    .descriptor;
                let loader_calls = Arc::clone(&calls);
                let loader_loads = Arc::clone(&loads);
                let loader_seen = Arc::clone(&seen);
                let loader_stopped = Arc::clone(&stopped_on_cancel);
                let loader_contexts = Arc::clone(&memory_contexts);
                let loader_adapters = Arc::clone(&loaded_adapters);
                let loader_offload = Arc::clone(&loaded_offload);
                let loader_shape = Arc::clone(&loaded_shape);
                let loader_quant = Arc::clone(&loaded_quant);
                let loader = move |_engine: &str, spec: &LoadSpec| {
                    #[cfg(target_os = "macos")]
                    let contract = crate::inference_runtime::media()
                        .memory_strategy_contract(_engine, spec)?;
                    *loader_loads.lock().unwrap() += 1;
                    *loader_offload.lock().unwrap() = Some(spec.offload_policy);
                    *loader_shape.lock().unwrap() = Some(spec.load_shape);
                    *loader_quant.lock().unwrap() = Some(spec.quantize);
                    *loader_adapters.lock().unwrap() = spec.adapters.clone();
                    Ok(Box::new(IrisArmProbe {
                        descriptor,
                        behavior,
                        generate_calls: loader_calls,
                        seen: loader_seen,
                        stopped_on_cancel: loader_stopped,
                        contexts: Arc::clone(&loader_contexts),
                        adapters: spec.adapters.clone(),
                        #[cfg(target_os = "macos")]
                        contract,
                    }) as Box<dyn Generator>)
                };
                let mut asset_writes = Vec::new();
                // The SAME arm body on both lanes: the MLX `generate_stream_using` on macOS, the
                // Candle `generate_candle_stream_using` off-Mac (sc-25680) — so progress, cancel,
                // failure and knob shaping are pinned per backend by one set of assertions.
                #[cfg(target_os = "macos")]
                let arm = super::super::generate_stream_using(
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
                let arm = super::super::generate_candle_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "cuda",
                    &mut asset_writes,
                    loader,
                    Arc::new(super::super::NativeMossStreamRuntime),
                );
                let outcome = tokio::time::timeout(Duration::from_secs(60), arm)
                    .await
                    .expect("the iris arm finished");
                let posts = posts.lock().unwrap().clone();
                let switch = if task_switch {
                    let transform_resident_after_generation =
                        crate::transform_cache::global_transform_resident().await;
                    let generator_after_generation =
                        crate::generator_cache::global_generator_resident().await;
                    // A restoration cold load (the production pre-load hook evicts the generator).
                    let second_export = tempfile::tempdir().unwrap();
                    crate::transform_cache::load_stub_into_global_cache(second_export.path())
                        .await
                        .unwrap();
                    Some(TaskSwitch {
                        transform_resident_after_generation,
                        generator_after_generation,
                        generator_after_restoration:
                            crate::generator_cache::global_generator_resident().await,
                        transform_resident_after_restoration:
                            crate::transform_cache::global_transform_resident().await,
                    })
                } else {
                    None
                };
                (outcome, posts, asset_writes, switch)
            })
        });
        let generate_calls = *calls.lock().unwrap();
        let seen = seen.lock().unwrap().clone();
        let loaded_adapters = loaded_adapters.lock().unwrap().clone();
        let loaded_offload = *loaded_offload.lock().unwrap();
        let loaded_shape = *loaded_shape.lock().unwrap();
        let loaded_quant = *loaded_quant.lock().unwrap();
        let memory_contexts = memory_contexts.lock().unwrap().clone();
        let stopped_on_cancel = *stopped_on_cancel.lock().unwrap();
        let load_calls = *loads.lock().unwrap();
        ArmRun {
            outcome,
            load_calls,
            posts,
            asset_writes,
            generate_calls,
            seen,
            loaded_adapters,
            loaded_offload,
            loaded_shape,
            loaded_quant,
            stopped_on_cancel,
            memory_contexts,
            project_files: files_under(project.path()),
            switch,
        }
    }

    /// sc-26266: both native arms route a tier pick to the quantized load — `advanced.mlxQuantize` 4 /
    /// 8 reach the engine as `LoadSpec::quantize` Q4 / Q8, and no pick (or an explicit 0) loads
    /// bf16 through the catalog's `mlx.quantize: 0` — and the job completes at that tier.
    ///
    /// *Mutations that red this:* dropping `mlx.quantize: 0` (a no-pick request loads the
    /// resolver's q8 default); an engine that advertises no quant (every pick loads bf16).
    #[test]
    fn a_tier_pick_reaches_the_quantized_load() {
        for (advanced, want) in [
            (json!({}), None),
            (json!({ "mlxQuantize": 0 }), None),
            (json!({ "mlxQuantize": 8 }), Some(gen_core::Quant::Q8)),
            (json!({ "mlxQuantize": 4 }), Some(gen_core::Quant::Q4)),
            (json!({ "mlxQuantize": "0" }), None),
            (json!({ "mlxQuantize": "8" }), Some(gen_core::Quant::Q8)),
            (json!({ "mlxQuantize": "4" }), Some(gen_core::Quant::Q4)),
        ] {
            let run = run_iris_arm(
                ArmBehavior::Completes,
                false,
                json!({ "advanced": advanced.clone() }),
            );
            assert!(run.outcome.is_ok(), "{advanced}: {:?}", run.outcome);
            assert_eq!(run.load_calls, 1, "{advanced}");
            assert_eq!(run.loaded_quant, Some(want), "{advanced}");
            assert_eq!(run.generate_calls, 1, "{advanced}");
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
        assert_eq!(run.load_calls, 1, "an admitted job loads exactly once");
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

    // ---------------------------------------------------------------------------------------
    // sc-25681 — every Iris control through the REAL arm (MLX on macOS, Candle off-Mac).
    // ---------------------------------------------------------------------------------------

    /// What the engine sees with nothing but a prompt: the catalog defaults the arm resolves (steps
    /// 100, guidance 3.0) and the engine's own for everything else.
    fn default_seen(seed: Option<u64>) -> Seen {
        Seen {
            prompt: "a lighthouse at dusk".to_owned(),
            negative_prompt: None,
            width: 64,
            height: 64,
            seed,
            steps: Some(gen_core_iris::DEFAULT_STEPS),
            guidance: Some(gen_core_iris::DEFAULT_CFG_SCALE),
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            cfg_interval: None,
            caption_overflow: None,
            prompt_batch: Vec::new(),
            count: 1,
        }
    }

    /// Table: a job payload → the exact request the engine is handed. Every control the catalog
    /// row offers reaches the engine verbatim (nothing accepted-then-ignored), and its absence
    /// means the engine's own default.
    ///
    /// *Mutation that reds this:* drop `cfg_interval` / `caption_overflow` from the request
    /// `generate_one_on_surface` builds, or re-introduce the shift withholding.
    #[test]
    fn every_iris_control_reaches_the_engine_through_the_real_arm() {
        use gen_core::CaptionOverflowPolicy as Policy;
        let cases: Vec<(Value, Seen)> = vec![
            (json!({ "seed": 11 }), default_seen(Some(11))),
            (
                json!({
                    "seed": 7,
                    "width": 512,
                    "height": 768,
                    "negativePrompt": "blurry, watermark",
                    "advanced": {
                        "steps": 30,
                        "guidanceScale": 4.5,
                        "sampler": "euler",
                        "schedulerShift": 2.5,
                        "cfgInterval": [0.1, 0.9],
                        "captionOverflow": "error",
                    },
                }),
                Seen {
                    negative_prompt: Some("blurry, watermark".to_owned()),
                    width: 512,
                    height: 768,
                    seed: Some(7),
                    steps: Some(30),
                    guidance: Some(4.5),
                    sampler: Some("euler".to_owned()),
                    scheduler_shift: Some(2.5),
                    cfg_interval: Some((0.1, 0.9)),
                    caption_overflow: Some(Policy::Error),
                    ..default_seen(None)
                },
            ),
            (
                json!({
                    "seed": 3,
                    "advanced": {
                        "sampler": "dpmpp_2m",
                        "schedulerShift": "6",
                        "captionOverflow": "silent",
                        "cfgInterval": [0, 0.5],
                    },
                }),
                Seen {
                    seed: Some(3),
                    sampler: Some("dpmpp_2m".to_owned()),
                    scheduler_shift: Some(6.0),
                    caption_overflow: Some(Policy::Silent),
                    cfg_interval: Some((0.0, 0.5)),
                    ..default_seen(None)
                },
            ),
            (
                // CFG off with nothing CFG-bound: honoured, guidance 1.0 reaches the engine.
                json!({ "seed": 5, "advanced": { "guidanceScale": 1.0, "captionOverflow": "warn" } }),
                Seen {
                    seed: Some(5),
                    guidance: Some(1.0),
                    caption_overflow: Some(Policy::Warn),
                    ..default_seen(None)
                },
            ),
        ];
        for (extra, want) in cases {
            let run = run_iris_arm(ArmBehavior::Fails, false, extra.clone());
            assert_eq!(run.seen, vec![want], "{extra}");
        }
    }

    /// A count-N job renders N engine requests on per-image seeds (seed, seed+1, …), each
    /// carrying the same controls.
    #[test]
    fn a_batch_renders_one_request_per_image_on_consecutive_seeds() {
        let run = run_iris_arm(
            ArmBehavior::Completes,
            false,
            json!({ "seed": 40, "count": 2, "advanced": { "sampler": "euler" } }),
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let seeds: Vec<_> = run.seen.iter().map(|seen| seen.seed).collect();
        assert_eq!(seeds, vec![Some(40), Some(41)]);
        assert!(run
            .seen
            .iter()
            .all(|seen| seen.sampler.as_deref() == Some("euler")));
        assert_eq!(run.asset_writes.len(), 2);
    }

    /// CFG off makes the bound negative prompt and the interval inert, and the engine refuses
    /// both: the arm refuses them too, typed and BEFORE load (no load, no render, no asset) —
    /// sc-25679 silently dropped the negative prompt; sc-25681 refuses it like the API does.
    /// Malformed values are typed refusals before load as well.
    #[test]
    fn inert_or_malformed_controls_are_refused_before_load() {
        let cases = [
            (
                json!({ "negativePrompt": "blurry", "advanced": { "guidanceScale": 1.0 } }),
                "negative prompt",
            ),
            (
                json!({ "advanced": { "guidanceScale": 1.0, "cfgInterval": [0.2, 0.8] } }),
                "guidance interval",
            ),
            (
                json!({ "advanced": { "cfgInterval": [0.9, 0.1] } }),
                "0 <= start",
            ),
            (
                json!({ "advanced": { "cfgInterval": "early" } }),
                "pair of numbers",
            ),
            (
                json!({ "advanced": { "captionOverflow": "loud" } }),
                "caption-overflow",
            ),
            (json!({ "advanced": { "schedulerShift": 0 } }), "> 0"),
        ];
        for (extra, fragment) in cases {
            let run = run_iris_arm(ArmBehavior::Completes, false, extra.clone());
            match &run.outcome {
                Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("iris_3b") && message.contains(fragment),
                    "{extra}: {message}"
                ),
                other => panic!("{extra}: expected a typed refusal, got {other:?}"),
            }
            assert_eq!(run.load_calls, 0, "{extra}: refused before load");
            assert!(
                run.asset_writes.is_empty() && run.project_files.is_empty(),
                "{extra}"
            );
        }
    }

    fn generation_adapter(
        id: &'static str,
        network_type: Option<&str>,
        weight: f64,
    ) -> TestAdapter {
        let mut metadata = json!({
            "family": "iris",
            "irisTask": "generation",
            "baseModel": "iris_3b",
        });
        if let Some(network_type) = network_type {
            metadata["networkType"] = json!(network_type);
        }
        TestAdapter {
            id,
            metadata,
            weight,
        }
    }

    /// A completed render through the real arm: the engine's x0 step preview reaches the job API
    /// on the preview path, its `caption_truncated` warning and the applied LoRA + LoKr stack
    /// (identity joined with the engine's own apply report) land in the asset's provenance, and the
    /// controls ride the recorded settings so the recipe reopens with them.
    ///
    /// *Mutation that reds this:* skip `send_image_facts`, or drop the `ImageFacts` merge in
    /// `consume_gen_events`.
    #[test]
    fn a_completed_render_reports_previews_warnings_and_adapter_provenance() {
        let run = run_iris_arm_with(
            ArmBehavior::Completes,
            false,
            json!({
                "seed": 9,
                "width": 256,
                "height": 256,
                "advanced": {
                    "sampler": "euler",
                    "schedulerShift": 3.0,
                    "cfgInterval": [0.1, 0.9],
                    "captionOverflow": "warn",
                },
            }),
            "64",
            &[
                generation_adapter("iris-style", None, 0.8),
                generation_adapter("iris-lokr", Some("lokr"), 1.2),
            ],
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        // Both adapters reached the load spec, in order, with their kind and job weight.
        let loaded: Vec<_> = run
            .loaded_adapters
            .iter()
            .map(|adapter| (adapter.kind, adapter.scale))
            .collect();
        assert_eq!(
            loaded,
            vec![
                (gen_core::AdapterKind::Lora, 0.8),
                (gen_core::AdapterKind::Lokr, 1.2)
            ]
        );
        // The preview rode a non-terminal progress post at the engine's (W/16)x(H/16) frame.
        let preview = run
            .posts
            .iter()
            .find_map(|post| post["result"]["previewFrame"].as_object())
            .unwrap_or_else(|| panic!("a preview frame must reach the job API: {:#?}", run.posts));
        assert_eq!(preview["current"], 1);
        assert!(preview["dataUrl"]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/")));
        // Provenance on the written asset.
        assert_eq!(run.asset_writes.len(), 1);
        let settings = &run.asset_writes[0]["rawAdapterSettings"];
        assert_eq!(
            settings["generationWarnings"],
            json!([{ "code": "caption_truncated", "message": FIXTURE_WARNING }])
        );
        let applied = settings["adapterApply"]
            .as_array()
            .expect("adapter provenance");
        assert_eq!(applied.len(), 2);
        for (entry, (id, kind, adapter)) in applied.iter().zip([
            ("iris-style", "lora", &run.loaded_adapters[0]),
            ("iris-lokr", "lokr", &run.loaded_adapters[1]),
        ]) {
            assert_eq!(entry["id"], id);
            assert_eq!(entry["kind"], kind);
            assert_eq!(entry["path"], json!(adapter.path.display().to_string()));
            assert_eq!(
                entry["scale"].as_f64().map(|v| v as f32),
                Some(adapter.scale)
            );
            assert_eq!(
                entry["sha256"].as_str().map(str::len),
                Some(64),
                "the loaded file's digest: {entry}"
            );
            assert_eq!(entry["applied"], 24);
            assert_eq!(entry["skipped"], json!(["pixel.blocks.0.attn"]));
        }
        // The controls are recorded with the image, so reopening the recipe restores them.
        for (key, value) in [
            ("sampler", json!("euler")),
            ("schedulerShift", json!(3.0)),
            ("cfgInterval", json!([0.1, 0.9])),
            ("captionOverflow", json!("warn")),
        ] {
            assert_eq!(settings[key], value, "{key}: {settings}");
        }
        assert_eq!(
            run.asset_writes[0]["loras"].as_array().map(Vec::len),
            Some(2)
        );
    }

    /// A replayed `advanced` block cannot forge the worker-owned provenance keys: a render with
    /// nothing to report records neither.
    #[test]
    fn replayed_provenance_keys_are_scrubbed() {
        let run = run_iris_arm(
            ArmBehavior::Completes,
            false,
            json!({
                "width": 256,
                "height": 256,
                "advanced": {
                    "adapterApply": [{ "id": "forged" }],
                    "generationWarnings": [{ "code": "forged", "message": "forged" }],
                },
            }),
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let settings = &run.asset_writes[0]["rawAdapterSettings"];
        // The stub always reports its fixture warning, so the key is present with ONLY that.
        assert_eq!(
            settings["generationWarnings"],
            json!([{ "code": "caption_truncated", "message": FIXTURE_WARNING }])
        );
        assert!(settings.get("adapterApply").is_none(), "{settings}");
    }

    // ---------------------------------------------------------------------------------------
    // sc-25681 — the NATIVE prompt batch through the real arm (MLX on macOS, Candle off-Mac).
    // ---------------------------------------------------------------------------------------

    /// An emulated budget that admits ONE weightless 64² Iris image but not eight: the MLX generic
    /// estimate charges its activation term per batched image; the Candle gate adds
    /// `(images × max(1, MP) − 1) × candle.imageActivationGbPerMegapixel` to the 24 GB floor (8
    /// images: 66 GB).
    #[cfg(target_os = "macos")]
    const BATCH_ADMISSION_CAP_GB: &str = "40";
    #[cfg(not(target_os = "macos"))]
    const BATCH_ADMISSION_CAP_GB: &str = "28";

    fn batch_payload(prompts: &[&str], count: u32) -> Value {
        json!({
            "prompt": prompts[0],
            "prompts": prompts,
            "count": count,
            "seed": 40,
            "advanced": { "sampler": "euler", "cfgInterval": [0.1, 0.9] },
        })
    }

    /// A 2-prompt × count-2 job is ONE engine call carrying `prompt_batch` + `count` (and no
    /// `prompt`), published as four assets in the engine's prompt-minor order — image `k` renders
    /// `prompts[k % 2]` on seed `40 + k` — each recording its own prompt, its seed and the batch.
    ///
    /// *Mutation that reds this:* route the batch through the per-seed loop (drop the
    /// `drive_native_prompt_batch` branch), or publish image `k` on the base seed.
    #[test]
    fn a_native_prompt_batch_is_one_engine_call_published_as_one_asset_per_image() {
        // A budget roomy enough for four images: admission charges every image of the batch.
        let run = run_iris_arm_with_memory_cap(
            ArmBehavior::Completes,
            false,
            batch_payload(&["a red fox", "a blue whale"], 2),
            "128",
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(run.load_calls, 1);
        assert_eq!(
            run.generate_calls, 1,
            "one engine call renders the whole batch"
        );
        assert_eq!(
            run.seen,
            vec![Seen {
                prompt: String::new(),
                prompt_batch: vec!["a red fox".to_owned(), "a blue whale".to_owned()],
                count: 2,
                seed: Some(40),
                sampler: Some("euler".to_owned()),
                cfg_interval: Some((0.1, 0.9)),
                ..default_seen(None)
            }],
            "the engine gets the list alone, the base seed, and the count"
        );
        let published: Vec<(Value, Value, Value)> = run
            .asset_writes
            .iter()
            .map(|asset| {
                (
                    asset["prompt"].clone(),
                    asset["seed"].clone(),
                    asset["index"].clone(),
                )
            })
            .collect();
        assert_eq!(
            published,
            vec![
                (json!("a red fox"), json!(40), json!(0)),
                (json!("a blue whale"), json!(41), json!(1)),
                (json!("a red fox"), json!(42), json!(2)),
                (json!("a blue whale"), json!(43), json!(3)),
            ]
        );
        for asset in &run.asset_writes {
            let settings = &asset["rawAdapterSettings"];
            assert_eq!(
                settings["promptBatch"],
                json!(["a red fox", "a blue whale"])
            );
            assert_eq!(
                settings["generationWarnings"][0]["code"], "caption_truncated",
                "every image of the call carries the call's warnings: {settings}"
            );
            assert_eq!(asset["count"], 4, "the generation set expects every image");
        }
        assert!(
            run.posts
                .iter()
                .any(|post| message(post).contains("Batch of 4 — step 1/100")),
            "progress names the batch, not image 1 of 4: {:#?}",
            run.posts
        );
    }

    /// A cancel mid-batch publishes NOTHING: the one engine call stops on its own request flag and
    /// returns no image, so no asset (and no file) is written.
    #[test]
    fn a_cancelled_native_prompt_batch_publishes_no_asset() {
        let run = run_iris_arm(
            ArmBehavior::CancelAware,
            true,
            batch_payload(&["a red fox", "a blue whale"], 1),
        );
        assert!(
            matches!(run.outcome, Err(crate::error::WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        assert_eq!(run.generate_calls, 1);
        assert!(run.stopped_on_cancel);
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    /// A batch the engine cannot render, or one combined with a per-image lane feature, is a typed
    /// refusal BEFORE load: over the engine's `max_count` (`prompts × count`), a single-entry list,
    /// and hires fix.
    ///
    /// *Mutation that reds this:* drop the `resolve_native_prompt_batch` call in the arm.
    #[test]
    fn an_unrenderable_native_prompt_batch_is_refused_before_load() {
        let mut hires = batch_payload(&["a red fox", "a blue whale"], 1);
        hires["hiresFix"] = json!({ "enabled": true });
        let cases = [
            (batch_payload(&["a", "b", "c"], 4), "12 images"),
            (batch_payload(&["only"], 1), "at least two"),
            (hires, "hires fix"),
        ];
        for (extra, fragment) in cases {
            let run = run_iris_arm(ArmBehavior::Completes, false, extra.clone());
            match &run.outcome {
                Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("iris_3b") && message.contains(fragment),
                    "{extra}: {message}"
                ),
                other => panic!("{extra}: expected a typed refusal, got {other:?}"),
            }
            assert_eq!(run.load_calls, 0, "{extra}: refused before load");
            assert!(run.asset_writes.is_empty(), "{extra}");
        }
    }

    /// Admission is scaled for the batch: a budget that admits a single image refuses an
    /// eight-image native batch BEFORE load, instead of admitting it on a one-image estimate.
    ///
    /// *Mutation that reds this:* `request_batch` back to a constant 1 (MLX) /
    /// `declared_activation_overlay_bytes` back to 0 (Candle).
    #[test]
    fn a_native_prompt_batch_is_admitted_against_its_whole_image_count() {
        let cap = BATCH_ADMISSION_CAP_GB;
        let single = run_iris_arm_with_memory_cap(ArmBehavior::Fails, false, json!({}), cap);
        assert_eq!(single.load_calls, 1, "one image fits: {:?}", single.outcome);
        let batch = run_iris_arm_with_memory_cap(
            ArmBehavior::Fails,
            false,
            batch_payload(&["a", "b"], 4),
            cap,
        );
        match &batch.outcome {
            Err(crate::error::WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("iris_3b"), "{message}")
            }
            other => panic!("an 8-image batch must be refused before load, got {other:?}"),
        }
        assert_eq!(batch.load_calls, 0, "refused before the loader ran");
    }

    // ---------------------------------------------------------------------------------------
    // sc-25684 (S6) — admission above the 1024² the floors were declared at.
    // ---------------------------------------------------------------------------------------

    /// A budget that admits a weightless 1024² image but not a 2048² one: on MLX the generic estimate
    /// scales its activation term with output megapixels; on Candle the gate adds
    /// `candle.imageActivationGbPerMegapixel × (MP − 1)` (2048²: +18 GB on the 24 GB floor).
    #[cfg(target_os = "macos")]
    const AREA_ADMISSION_CAP_GB: &str = "40";
    #[cfg(not(target_os = "macos"))]
    const AREA_ADMISSION_CAP_GB: &str = "28";

    /// The limits allow 2048² but every floor was declared at 1024²: a 4 MP request is admitted
    /// against its area — refused BEFORE load on a budget that fits 1024² — never admitted on the
    /// 1024² floor and never silently resized.
    ///
    /// *Mutation that reds this:* `declared_activation_overlay_bytes` back to the batch-only
    /// `(images − 1) × …` (Candle) / `generic_headroom_bytes` without the megapixel scale (MLX).
    #[test]
    fn an_output_above_1024_squared_is_admitted_against_its_area() {
        let cap = AREA_ADMISSION_CAP_GB;
        let square = run_iris_arm_with_memory_cap(
            ArmBehavior::Fails,
            false,
            json!({ "width": 1024, "height": 1024 }),
            cap,
        );
        assert_eq!(square.load_calls, 1, "1024² fits: {:?}", square.outcome);
        let large = run_iris_arm_with_memory_cap(
            ArmBehavior::Fails,
            false,
            json!({ "width": 2048, "height": 2048 }),
            cap,
        );
        match &large.outcome {
            Err(crate::error::WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("iris_3b"), "{message}")
            }
            other => panic!("a 2048² render must be refused before load, got {other:?}"),
        }
        assert_eq!(large.load_calls, 0, "refused before the loader ran");
        assert!(large.asset_writes.is_empty());
        assert!(large.project_files.is_empty(), "{:?}", large.project_files);
    }

    /// Generation ↔ restoration task switching keeps ONE Iris task model resident: a generation job
    /// evicts a resident restoration transform before its own load, and loads the backbone WITH its
    /// `text_encoder` component; a restoration cold load then evicts that generator — and the text
    /// encoder with it (restoration runs without one). Depth's side is pinned in
    /// `depth_jobs::tests::a_depth_job_evicts_a_resident_generator_or_transform_before_loading`.
    ///
    /// *Mutations that red this:* drop `evict_resident_transform()` from the generator cache entry
    /// points (the transform survives the generation job) / the transform cache's
    /// `evict_resident_generator_before_transform_load` hook (the generator and its text encoder
    /// survive the restoration load).
    #[test]
    fn generation_and_restoration_evict_each_other_and_the_text_encoder_leaves_with_its_generator()
    {
        let run = run_iris_task_switch();
        run.outcome.as_ref().unwrap();
        assert_eq!(run.load_calls, 1);
        let switch = run.switch.expect("task switch recorded");
        assert!(
            !switch.transform_resident_after_generation,
            "the generation job must evict the resident restoration transform: {switch:?}"
        );
        assert_eq!(
            switch.generator_after_generation,
            Some(vec![gen_core_iris::TEXT_ENCODER_COMPONENT.to_owned()]),
            "the generator is resident with its text encoder: {switch:?}"
        );
        assert_eq!(
            switch.generator_after_restoration, None,
            "a restoration cold load evicts the generator and its text encoder: {switch:?}"
        );
        assert!(switch.transform_resident_after_restoration, "{switch:?}");
    }

    /// On CUDA a resident restoration transform holds real VRAM that the free-VRAM probe would count
    /// as occupied — while the generator cache evicts it anyway on load — so the Candle arm evicts it
    /// BEFORE probing (a stub transform holds no VRAM, so this is pinned on the source order; the
    /// eviction itself is pinned by the task-switch test above).
    ///
    /// *Mutation that reds this:* move/drop the `evict_resident_transform()` before the probe.
    #[test]
    fn the_candle_arm_evicts_a_resident_transform_before_probing_vram() {
        let base = include_str!("base.rs");
        let body = &base[base
            .find("async fn generate_candle_stream_using(")
            .expect("candle arm")..];
        let evict = body
            .find("crate::transform_cache::evict_resident_transform().await?")
            .expect("the candle arm evicts a resident transform");
        let probe = body
            .find("crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await")
            .expect("the candle arm probes VRAM");
        assert!(evict < probe, "evict before the probe");
    }

    // ---------------------------------------------------------------------------------------
    // sc-25680 review — the Candle lane's declared VRAM floor (E8). Candle-only: `vram_gate` and
    // the Candle arm compile only off-Mac with `backend-candle`, so these run on the
    // windows-candle lane, never on a Mac.
    // ---------------------------------------------------------------------------------------

    /// Every tier key the Candle gate can name for the single bf16 artifact resolves a resident
    /// AND a staged prediction, so `fit_decision` is never `Unknown` (admit-anything) for iris.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn the_candle_gate_has_a_resident_and_a_staged_prediction_for_every_iris_tier_key() {
        let iris = shipped_iris_entry();
        for advanced in [
            json!({}),
            json!({ "mlxQuantize": 0 }),
            json!({ "mlxQuantize": 4 }),
            json!({ "mlxQuantize": 8 }),
        ] {
            let tier =
                crate::vram_gate::requested_tier_key(advanced.as_object().unwrap(), &iris, false);
            assert_eq!(
                crate::vram_gate::predicted_peak_gb(&iris, tier),
                Some(24.0),
                "{tier}: the padded resident floor"
            );
            assert_eq!(
                crate::vram_gate::predicted_sequential_peak_gb(&iris, tier),
                Some(12.0 + crate::vram_gate::HEADROOM_GB),
                "{tier}: the measured staged peak + reserve"
            );
        }
    }

    /// sc-25684: a 16 GB card runs a 1024² Iris image STAGED (the measured 12.0 GB staged peak +
    /// 2 GB reserve fits; the 24 GB resident floor does not) — the loader is handed
    /// `OffloadPolicy::Sequential` — while a 2048² image on the same card (+18 GB of declared
    /// activations) and any image on a 12 GB card are refused BEFORE the loader runs.
    ///
    /// *Mutations that red this:* the staged row back to the resident 21.4 (16 GB refused), or
    /// `declared_activation_overlay_bytes` ignoring area (2048² admitted).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn a_16_gb_card_runs_1024_staged_and_refuses_what_does_not_fit() {
        let staged = run_iris_arm_with_memory_cap(
            ArmBehavior::Fails,
            false,
            json!({ "width": 1024, "height": 1024 }),
            "16",
        );
        assert_eq!(staged.load_calls, 1, "{:?}", staged.outcome);
        assert_eq!(
            staged.loaded_offload,
            Some(gen_core::OffloadPolicy::Sequential),
            "a 16 GB card loads staged"
        );
        for (extra, cap) in [
            (json!({ "width": 2048, "height": 2048 }), "16"),
            (json!({ "width": 1024, "height": 1024 }), "12"),
        ] {
            let run = run_iris_arm_with_memory_cap(ArmBehavior::Fails, false, extra.clone(), cap);
            match &run.outcome {
                Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("iris_3b") && message.contains("VRAM"),
                    "{extra} on {cap} GB: the refusal names the model and its VRAM need: {message}"
                ),
                other => {
                    panic!("{extra} on a {cap} GB card must be refused before load, got {other:?}")
                }
            }
            assert_eq!(
                run.load_calls, 0,
                "{extra} on {cap} GB: refused before the loader ran"
            );
            assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
            assert!(run.project_files.is_empty(), "{:?}", run.project_files);
        }
    }

    // ---------------------------------------------------------------------------------------
    // sc-25690 — the engines' memory-strategy contract through the real arm.
    // ---------------------------------------------------------------------------------------

    /// An engine that refuses a request wider than its admission (`prompts × count` over the
    /// admitted `MemoryGeometry::batch`, the rule both Iris engines enforce in their request scope)
    /// ends the job with that refusal: a typed failure naming it, no asset and no file — never a
    /// partial batch, and never swallowed.
    ///
    /// *Mutation that reds this:* ignore the `configure_request` error in
    /// `memory_strategy::run_with_scope` (the batch then renders and four assets are written).
    #[test]
    fn an_engine_refusal_of_an_over_wide_batch_fails_the_job_with_no_asset() {
        let run = run_iris_arm_with_memory_cap(
            ArmBehavior::UnderAdmits,
            false,
            batch_payload(&["a red fox", "a blue whale"], 2),
            OPTIMIZED_BATCH_CAP_GB,
        );
        match &run.outcome {
            Err(crate::error::WorkerError::Engine(message)) => assert!(
                message.contains("renders 4 images but was admitted for 3"),
                "the engine's refusal survives: {message}"
            ),
            other => panic!("an over-wide batch must fail the job, got {other:?}"),
        }
        assert_eq!(run.generate_calls, 0, "refused before the engine rendered");
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    /// Every Iris loader now FAILS on `LoadShape::DeferredMaterialization` (no Iris route has a
    /// block-streaming loader), so no worker path may hand one a deferred spec. Deferral has
    /// exactly two sources in the worker — a `memory_route_registry` rule and the legacy shaper
    /// that reads those rules — and every Iris load spec starts from gen-core's eager default:
    /// the generation arm (fresh and cached loads share one spec), the depth job, the restoration
    /// transform, and the trainers' base load (which the training-export render reuses).
    ///
    /// *Mutation that reds this:* add a `legacy_shaping` rule for `iris_3b` to
    /// `memory_route_registry::RULES` (the witness list and the shaper both name it, and the arm
    /// hands its loader a deferred spec).
    #[test]
    fn no_worker_path_hands_an_iris_route_a_deferred_load_shape() {
        use crate::memory_route_registry::{
            apply_registered_load_shape, deferred_route_witnesses, MemoryRouteBackend,
            MemoryRouteMode,
        };
        let eager = gen_core::LoadShape::EagerMaterialization;
        let ids = [
            "iris_3b",
            gen_core_iris::DEPTH_MODEL_ID,
            crate::iris_restore::IRIS_RESTORE_MODEL_ID,
        ];
        for witness in deferred_route_witnesses() {
            assert!(
                !ids.contains(&witness.provider),
                "no route rule may name an Iris provider: {witness:?}"
            );
        }
        for backend in [MemoryRouteBackend::Mlx, MemoryRouteBackend::Candle] {
            for id in ids {
                for mode in MemoryRouteMode::ALL {
                    for sequential in [false, true] {
                        let shaped = apply_registered_load_shape(
                            backend,
                            id,
                            mode,
                            LoadSpec::new(WeightsSource::Dir("iris".into())),
                            sequential,
                        );
                        assert_eq!(
                            shaped.load_shape, eager,
                            "{backend:?} {id} {mode:?} sequential={sequential}"
                        );
                    }
                }
            }
        }
        // What each route actually builds.
        assert_eq!(
            crate::depth_jobs::depth_load_spec(Path::new("depth")).load_shape,
            eager
        );
        for id in ids {
            let trainer = crate::training_jobs::TrainerLoadInputs::from_task(
                id,
                std::path::PathBuf::from("iris"),
                std::collections::BTreeMap::new(),
            )
            .load_spec(id, "bf16");
            assert_eq!(trainer.load_shape, eager, "{id} trainer base load");
        }
        // The generation arm, resident and — on the lane where a small card stages — staged.
        let resident = run_iris_arm(ArmBehavior::Fails, false, json!({}));
        assert_eq!(resident.load_calls, 1, "{:?}", resident.outcome);
        assert_eq!(resident.loaded_shape, Some(eager));
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        {
            let staged = run_iris_arm_with_memory_cap(
                ArmBehavior::Fails,
                false,
                json!({ "width": 1024, "height": 1024 }),
                "16",
            );
            assert_eq!(
                staged.loaded_offload,
                Some(gen_core::OffloadPolicy::Sequential)
            );
            assert_eq!(staged.loaded_shape, Some(eager));
        }
    }

    /// Both Iris generation engines implement resident, staged and bounded-attention rungs. The
    /// shared contract must emit the selected, declared chunk budget only for bounded attention;
    /// neither a missing budget nor an undeclared one is a valid selection. Encoder-free tasks
    /// retain resident attention. Decode tiling and transformer streaming remain unavailable.
    ///
    /// *Mutations that red this:* dropping or changing the emitted chunk budget, or withdrawing
    /// the generation descriptor's bounded-attention rung. No GPU or weights are needed.
    #[test]
    fn no_selectable_iris_rung_emits_a_request_memory_block_the_engine_refuses() {
        use gen_core::MemoryStrategy;
        let registry = crate::inference_runtime::media();
        let spec = LoadSpec::new(WeightsSource::Dir("iris-weights-free".into()));
        let mut generation_rungs = Vec::new();
        for id in [
            "iris_3b",
            gen_core_iris::DEPTH_MODEL_ID,
            crate::iris_restore::IRIS_RESTORE_MODEL_ID,
        ] {
            let fixture = registry
                .memory_contract_fixture_registrations()
                .find(|fixture| fixture.provider_id == id)
                .unwrap_or_else(|| panic!("{id} publishes a weights-free memory contract"));
            let contract = (fixture.contract)(&spec).expect("weights-free contract");
            assert_eq!(contract.provider_id, id);
            assert_eq!(
                contract.load_shape,
                gen_core::LoadShape::EagerMaterialization
            );
            for strategy in MemoryStrategy::ALL {
                let implemented = matches!(
                    contract.capability(strategy).map(|cap| &cap.support),
                    Some(gen_core::MemoryStrategySupport::Implemented)
                );
                if !implemented {
                    continue;
                }
                let tier = gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                };
                let selection = contract
                    .representative_selection(strategy, tier, false)
                    .expect("an implemented rung has a representative selection");
                contract
                    .validate_selection(&selection)
                    .unwrap_or_else(|error| panic!("{id} {strategy:?}: {error}"));
                if id == "iris_3b" {
                    generation_rungs.push(strategy);
                }
                let Some(memory) = contract.generation_memory(&selection) else {
                    assert_eq!(strategy, MemoryStrategy::Resident, "{id}");
                    continue;
                };
                assert!(!memory.tile_vae_decode, "{id} {strategy:?}: {memory:?}");
                assert!(
                    !memory.stream_transformer_blocks,
                    "{id} {strategy:?}: {memory:?}"
                );
                assert_eq!(memory.decode_tile_edge, None, "{id} {strategy:?}");
                assert_eq!(memory.transformer_window_size, None, "{id} {strategy:?}");
                assert_eq!(
                    memory.chunk_attention,
                    strategy == MemoryStrategy::BoundedAttention,
                    "{id} {strategy:?}: only the bounded-attention rung chunks attention"
                );
                if strategy == MemoryStrategy::BoundedAttention {
                    assert_eq!(id, "iris_3b", "encoder-free tasks keep resident attention");
                    let budget = memory
                        .attention_chunk_size
                        .expect("a bounded-attention request must name its budget");
                    assert!(budget > 0, "{id}: the attention budget is nonzero");
                    assert_eq!(Some(budget), selection.parameters.attention_chunk_size);
                    assert_eq!(
                        contract
                            .capability(MemoryStrategy::BoundedAttention)
                            .unwrap()
                            .parameters
                            .attention_chunk_sizes,
                        [budget],
                        "{id}: the request carries the engine's one declared budget"
                    );
                    for invalid_budget in [None, Some(budget / 2)] {
                        let mut invalid = selection;
                        invalid.parameters.attention_chunk_size = invalid_budget;
                        assert!(
                            contract.validate_selection(&invalid).is_err(),
                            "{id}: a missing or undeclared budget {invalid_budget:?} must be refused"
                        );
                    }
                } else {
                    assert_eq!(memory.attention_chunk_size, None, "{id} {strategy:?}");
                }
            }
        }
        assert_eq!(
            generation_rungs,
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedAttention,
            ],
            "both native generation ladders retain all three implemented rungs"
        );
    }

    /// A budget on which a 2 × 2 native batch runs under an OPTIMIZED rung on the Candle lane: the
    /// batch adds 18 GB of declared activations to both floors, so the 24 GB resident floor (42 GB)
    /// does not fit and the staged one does. Roomy on MLX, where the test only needs a request
    /// scope to be opened.
    #[cfg(target_os = "macos")]
    const OPTIMIZED_BATCH_CAP_GB: &str = "128";
    #[cfg(not(target_os = "macos"))]
    const OPTIMIZED_BATCH_CAP_GB: &str = "36";

    /// The Candle arm admits a native prompt batch at its IMAGE COUNT when the shared ladder
    /// selects an optimized rung, so the engine's request scope — which refuses
    /// `prompts × count > MemoryGeometry::batch` — renders it; and the batch is charged exactly
    /// once: the admitted peak is the single-image staged peak plus the declared activation
    /// overlay, not that overlay AND a batch-scaled derivation.
    ///
    /// *Mutations that red this:* the shared geometry's `batch` back to a constant 1 (the scope
    /// refuses the 4-image request and the job fails) / `candle_memory_strategy::law_geometry`
    /// returning the request geometry unchanged (the batch is priced twice and the peak exceeds
    /// single + overlay) / `candle_shared_memory_tier` returning the gate's key (the selection
    /// names a q8 tier the bf16 generator did not load).
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn a_native_prompt_batch_under_an_optimized_rung_is_admitted_at_its_image_count() {
        // One image on a 16 GB card: staged (the same rung the batch takes).
        let single = run_iris_arm_with_memory_cap(
            ArmBehavior::Completes,
            false,
            json!({ "seed": 40 }),
            "16",
        );
        assert!(single.outcome.is_ok(), "{:?}", single.outcome);
        let [single_context] = single.memory_contexts.as_slice() else {
            panic!(
                "one optimized request scope per image: {:?}",
                single.memory_contexts
            )
        };

        let batch = run_iris_arm_with_memory_cap(
            ArmBehavior::Completes,
            false,
            batch_payload(&["a red fox", "a blue whale"], 2),
            OPTIMIZED_BATCH_CAP_GB,
        );
        assert!(batch.outcome.is_ok(), "{:?}", batch.outcome);
        assert_eq!(batch.generate_calls, 1, "one engine call renders the batch");
        assert_eq!(batch.asset_writes.len(), 4, "{:?}", batch.asset_writes);
        assert_eq!(
            batch.loaded_offload,
            Some(gen_core::OffloadPolicy::Sequential),
            "the staged selection loads staged"
        );
        let [context] = batch.memory_contexts.as_slice() else {
            panic!(
                "one request scope for the batch: {:?}",
                batch.memory_contexts
            )
        };
        assert!(
            context.selection.strategy.is_optimized(),
            "{:?}",
            context.selection
        );
        assert_eq!(
            context.selection.strategy,
            single_context.selection.strategy
        );
        assert_eq!(
            context.geometry.batch, 4,
            "admitted at prompts × count, the bound the engine's scope enforces"
        );
        assert_eq!(single_context.geometry.batch, 1);
        // The loaded tier: the one dense artifact, whatever `advanced.mlxQuantize` defaults to.
        for admitted in [single_context, context] {
            assert_eq!(
                admitted.selection.tier.quant, None,
                "{:?}",
                admitted.selection
            );
            assert_eq!(admitted.selection.tier.precision, gen_core::Precision::Bf16);
        }
        // Charged ONCE: (4 images × 1 MP − 1) × the declared 6 GB per megapixel, on top of the
        // single-image peak of the same rung.
        let overlay_bytes = (18.0 * crate::fit_gate::BYTES_PER_GIB) as u64;
        assert_eq!(
            context.predicted_peak_bytes,
            single_context.predicted_peak_bytes + overlay_bytes,
            "the batch is priced by the declared overlay alone"
        );
    }

    /// Restores the miniature install for whatever test this thread runs next.
    #[cfg(target_os = "macos")]
    struct PricedComponents;

    #[cfg(target_os = "macos")]
    impl PricedComponents {
        fn gib(gib: usize) -> Self {
            PRICED_COMPONENT_GIB.with(|cell| cell.set(Some(gib)));
            Self
        }
    }

    #[cfg(target_os = "macos")]
    impl Drop for PricedComponents {
        fn drop(&mut self) {
            PRICED_COMPONENT_GIB.with(|cell| cell.set(None));
        }
    }

    /// On MLX the request is graded twice against the engine's contract: before the load (which
    /// also picks the load policy) and again on the loaded generator. With a text encoder and a
    /// backbone that each price at 8 GiB, sweeping the emulated Mac from large to small crosses
    /// three bands, and in every one the second grading agrees with the first — a Mac the first
    /// one let load always RUNS, under the rung and at the price the first one admitted:
    ///
    /// * roomy — both components resident;
    /// * between — `StagedResidency` (the tower is dropped before the backbone's phases), whether
    ///   the weights were loaded warm or `Sequential`, at one price below the resident one;
    /// * too small even staged — refused BEFORE the loader runs, naming the model.
    ///
    /// *Mutations that red this:* `synthesize_estimate_ladder_with_conditioning` skipping
    /// `MemoryStrategy::StagedResidency` (the between band is refused instead of run) /
    /// the probe engine publishing no contract (the between band loads and is then refused at the
    /// resident price) / the install left miniature (staging saves nothing: no between band) /
    /// `engine_supports_sequential` answering `false` (every between Mac loads warm).
    #[cfg(target_os = "macos")]
    #[test]
    fn every_mlx_memory_band_either_refuses_before_load_or_runs_the_rung_it_priced() {
        use gen_core::MemoryStrategy;
        let _priced = PricedComponents::gib(8);
        let run = |cap: &str| {
            let run = run_iris_arm_with_memory_cap(ArmBehavior::Completes, false, json!({}), cap);
            if run.load_calls > 0 {
                assert!(
                    run.outcome.is_ok() && run.generate_calls == 1,
                    "{cap} GiB: a request that was let load must run: {:?}",
                    run.outcome
                );
            }
            run
        };
        let selected = |run: &ArmRun, cap: &str| {
            let [context] = run.memory_contexts.as_slice() else {
                panic!("{cap} GiB: one request scope: {:?}", run.memory_contexts)
            };
            (context.selection.strategy, context.predicted_peak_bytes)
        };

        let roomy = run("64");
        let (strategy, resident_peak) = selected(&roomy, "64");
        assert_eq!(strategy, MemoryStrategy::Resident);
        assert_eq!(
            roomy.loaded_offload,
            Some(gen_core::OffloadPolicy::Resident)
        );

        let mut staged_peaks = Vec::new();
        let mut staged_loads = Vec::new();
        for cap in ["44", "36", "27"] {
            let between = run(cap);
            let (strategy, peak) = selected(&between, cap);
            assert_eq!(strategy, MemoryStrategy::StagedResidency, "{cap} GiB");
            staged_peaks.push(peak);
            staged_loads.push(between.loaded_offload);
        }
        assert!(
            staged_peaks.iter().all(|peak| *peak == staged_peaks[0]),
            "one staged price whatever the Mac: {staged_peaks:?}"
        );
        assert!(
            staged_peaks[0] < resident_peak,
            "staging holds max(tower, backbone), not the sum: {staged_peaks:?} vs {resident_peak}"
        );
        assert!(
            staged_loads.contains(&Some(gen_core::OffloadPolicy::Sequential)),
            "the smaller Macs load the weights staged too: {staged_loads:?}"
        );

        for cap in ["26", "16"] {
            let small = run(cap);
            assert_eq!(small.load_calls, 0, "{cap} GiB: refused before the loader");
            match &small.outcome {
                Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("iris_3b") && message.contains("GiB"),
                    "{cap} GiB: the refusal names the model and its need: {message}"
                ),
                other => panic!("{cap} GiB must be refused before load, got {other:?}"),
            }
            assert!(small.asset_writes.is_empty(), "{:?}", small.asset_writes);
        }
    }
}
