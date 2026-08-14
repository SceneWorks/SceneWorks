#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::candle::*;
use super::seedvr2::*;
use super::*;
#[cfg(target_os = "macos")]
use super::{bernini::*, krea_realtime::*, ltx::*, mochi::*, scail2::*, svd::*, vace::*, wan::*};
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::{bernini::*, ltx::*, mochi::*, svd::*, wan::*};

/// Admission must see the exact temporal chunk already resolved onto the engine input. Re-reading
/// the sparse payload or dropping this field would turn the real 25-frame/chunk-8 SVD execution
/// into a fictional 25-frame VAE invocation.
#[test]
fn video_admission_captures_the_resolved_decode_chunk_from_the_engine_input() {
    const WAN: &str = include_str!("wan.rs");
    let capture = WAN
        .split_once("let admission_geometry = (")
        .expect("shared video funnel captures its admission geometry")
        .1
        .split_once(");")
        .expect("admission geometry tuple closes")
        .0;
    assert!(
        capture.contains("input.decode_chunk_size"),
        "the admission shape must capture the provider-resolved decode chunk: {capture}"
    );

    let inputs = WAN
        .split_once("crate::video_admission::VideoAdmissionInputs {")
        .expect("shared video funnel invokes admission")
        .1
        .split_once("},")
        .expect("admission inputs close")
        .0;
    assert!(
        inputs.contains("decode_chunk_size: admission_geometry.3"),
        "the captured chunk must reach VideoAdmissionInputs: {inputs}"
    );
}

/// Closure currency must use the resolved provider id, not the catalog alias. On macOS the Wan
/// 5B route therefore resolves the lane key `mlx:wan2_2_ti2v_5b`; the generated closure catalog is
/// stamped only after the final inference head is frozen.
#[cfg(target_os = "macos")]
#[test]
fn video_admission_closure_currency_uses_lane_and_resolved_provider() {
    const WAN: &str = include_str!("wan.rs");
    let lookup = WAN
        .split_once("let admission_closure_digest =")
        .expect("shared video funnel resolves closure currency")
        .1
        .split_once(";")
        .expect("closure lookup statement closes")
        .0;
    assert!(
        lookup.contains("packaged_closure_digest(")
            && lookup.contains("crate::video_admission::LANE.as_key()")
            && lookup.contains("input.engine_id"),
        "closure lookup must key the active lane and resolved provider: {lookup}"
    );
    assert_eq!(
        format!(
            "{}:{}",
            crate::video_admission::LANE.as_key(),
            "wan2_2_ti2v_5b"
        ),
        "mlx:wan2_2_ti2v_5b"
    );
}

#[test]
fn video_jobs_remains_split_into_real_engine_modules() {
    const PARENT: &str = include_str!("mod.rs");
    const MODULES: &[(&str, &str, &str)] = &[
        (
            "seedvr2",
            include_str!("seedvr2.rs"),
            "pub(crate) async fn run_video_upscale_job(",
        ),
        (
            "wan",
            include_str!("wan.rs"),
            "pub(super) async fn generate_wan(",
        ),
        (
            "candle",
            include_str!("candle.rs"),
            "pub(super) async fn generate_candle_video(",
        ),
        (
            "bernini",
            include_str!("bernini.rs"),
            "pub(super) async fn generate_bernini(",
        ),
        (
            "scail2",
            include_str!("scail2.rs"),
            "pub(super) async fn generate_scail2(",
        ),
        (
            "krea_realtime",
            include_str!("krea_realtime.rs"),
            "pub(super) async fn generate_krea_realtime(",
        ),
        (
            "ltx",
            include_str!("ltx.rs"),
            "pub(super) async fn generate_ltx(",
        ),
        (
            "mochi",
            include_str!("mochi.rs"),
            "pub(super) async fn generate_mochi(",
        ),
        (
            "svd",
            include_str!("svd.rs"),
            "pub(super) async fn generate_svd(",
        ),
        (
            "vace",
            include_str!("vace.rs"),
            "pub(super) async fn generate_wan_vace(",
        ),
    ];

    assert!(
        !PARENT.contains("include!("),
        "engine families must be real Rust modules, not textual include! composition"
    );
    for family in [
        "seedvr2",
        "wan",
        "candle",
        "bernini",
        "scail2",
        "krea_realtime",
        "ltx",
        "mochi",
        "svd",
        "vace",
    ] {
        assert!(
            !PARENT.contains(&format!("{family}::*")),
            "video_jobs parent must not flatten the {family} family namespace"
        );
    }
    assert!(
        PARENT.lines().count() < 2_500,
        "shared video_jobs parent re-expanded to {} lines",
        PARENT.lines().count()
    );
    for (name, source, entrypoint) in MODULES {
        assert!(
            PARENT.contains(&format!("mod {name};")),
            "video_jobs parent no longer declares the {name} module"
        );
        assert!(
            source.lines().count() >= 100,
            "{name}.rs is unexpectedly empty or collapsed"
        );
        assert!(
            !source.contains("use super::*"),
            "{name}.rs must import its parent dependencies through the explicit prelude"
        );
        assert!(
            source.contains(entrypoint),
            "{name}.rs no longer owns its principal semantic entrypoint {entrypoint}"
        );
        for (other_name, other_source, _) in MODULES {
            if other_name != name {
                assert!(
                    !other_source.contains(entrypoint),
                    "{entrypoint} belongs to {name}.rs, not {other_name}.rs"
                );
            }
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VideoLifecycleBehavior {
    Success,
    SafetyReject,
    ConfigureError,
    Canceled,
    GenerateError,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Default, Debug)]
struct VideoLifecycleRecord {
    safety_checks: usize,
    begin_calls: usize,
    configure_calls: usize,
    generate_calls: usize,
    generated_memory: Vec<Option<gen_core::GenerationMemory>>,
    finishes: Vec<gen_core::MemoryRunOutcome>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct VideoLifecycleScope {
    record: std::sync::Arc<std::sync::Mutex<VideoLifecycleRecord>>,
    configure_error: bool,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl gen_core::MemoryRequestScope for VideoLifecycleScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        let mut record = self.record.lock().unwrap();
        record.configure_calls += 1;
        record.generated_memory.push(request.memory);
        if self.configure_error {
            Err(gen_core::Error::Unsupported(
                "fixture configure failure".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }

    fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }

    fn configure_decode(
        &mut self,
        _tile_edge: u32,
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

    fn finish(&mut self, outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.record.lock().unwrap().finishes.push(outcome);
        Ok(())
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct VideoLifecycleGenerator {
    descriptor: gen_core::ModelDescriptor,
    behavior: VideoLifecycleBehavior,
    record: std::sync::Arc<std::sync::Mutex<VideoLifecycleRecord>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl VideoLifecycleGenerator {
    fn new(behavior: VideoLifecycleBehavior) -> Self {
        Self {
            descriptor: gen_core::ModelDescriptor {
                id: "video_lifecycle_fixture",
                family: "ltx",
                backend: "mlx",
                modality: gen_core::Modality::Video,
                capabilities: Default::default(),
                required_components: &[],
                control_kinds: None,
            },
            behavior,
            record: Default::default(),
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Generator for VideoLifecycleGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _request: &GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn memory_strategy_safety_check(
        &self,
        _context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        self.record.lock().unwrap().safety_checks += 1;
        if self.behavior == VideoLifecycleBehavior::SafetyReject {
            gen_core::MemorySafetyDecision::Reject {
                reason: "fixture safety rejection".to_owned(),
            }
        } else {
            gen_core::MemorySafetyDecision::Accept
        }
    }

    fn begin_memory_strategy_request(
        &self,
        _context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        self.record.lock().unwrap().begin_calls += 1;
        Ok(Some(Box::new(VideoLifecycleScope {
            record: self.record.clone(),
            configure_error: self.behavior == VideoLifecycleBehavior::ConfigureError,
        })))
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        {
            let mut record = self.record.lock().unwrap();
            record.generate_calls += 1;
            record.generated_memory.push(request.memory);
        }
        match self.behavior {
            VideoLifecycleBehavior::Canceled => Err(gen_core::Error::Canceled),
            VideoLifecycleBehavior::GenerateError => Err(gen_core::Error::Unsupported(
                "fixture generation failure".to_owned(),
            )),
            VideoLifecycleBehavior::Success
            | VideoLifecycleBehavior::ConfigureError
            | VideoLifecycleBehavior::SafetyReject => Ok(GenerationOutput::Video {
                frames: vec![gen_core::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                }],
                fps: request.fps.unwrap_or(24),
                audio: None,
            }),
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_lifecycle_context(strategy: gen_core::MemoryStrategy) -> gen_core::MemoryRunContext {
    gen_core::MemoryRunContext {
        selection: gen_core::MemorySelection {
            strategy,
            parameters: if strategy == gen_core::MemoryStrategy::BoundedDecode {
                gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(256),
                    decode_overlap: Some(32),
                    ..Default::default()
                }
            } else {
                Default::default()
            },
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q8),
                component_precision_floors: &[],
            },
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: "video-lifecycle-fixture-v1".to_owned(),
        load_shape: gen_core::LoadShape::EagerMaterialization,
        mode: gen_core::MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: gen_core::MemoryGeometry {
            width: 768,
            height: 512,
            batch: 1,
            frames: 121,
            reference_count: 0,
        },
        overlay: None,
        budget: gen_core::MemoryBudget {
            total_bytes: 64 * 1024 * 1024 * 1024,
            committed_bytes: 20 * 1024 * 1024 * 1024,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2 * 1024 * 1024 * 1024,
        },
        predicted_peak_bytes: 18 * 1024 * 1024 * 1024,
        cache_state: gen_core::MemoryCacheState::Warm,
        evidence_revision: "fixture-evidence".to_owned(),
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn video_lifecycle_input(strategy: gen_core::MemoryStrategy) -> VideoGenInput {
    let memory =
        (strategy != gen_core::MemoryStrategy::Resident).then(|| gen_core::GenerationMemory {
            tile_vae_decode: strategy == gen_core::MemoryStrategy::BoundedDecode,
            decode_tile_edge: (strategy == gen_core::MemoryStrategy::BoundedDecode).then_some(256),
            decode_overlap: (strategy == gen_core::MemoryStrategy::BoundedDecode).then_some(32),
            ..Default::default()
        });
    VideoGenInput {
        engine_id: "video_lifecycle_fixture",
        prompt: "fixture".to_owned(),
        width: 768,
        height: 512,
        frames: 121,
        fps: 24,
        memory,
        memory_context: Some(video_lifecycle_context(strategy)),
        ..VideoGenInput::default()
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_lifecycle_safety_rejection_prevents_begin_and_generate() {
    let generator = VideoLifecycleGenerator::new(VideoLifecycleBehavior::SafetyReject);
    let result = run_loaded_video_generation(
        &generator,
        video_lifecycle_input(gen_core::MemoryStrategy::StagedResidency),
        &CancelFlag::new(),
        &mut |_| {},
    );
    assert!(result.is_err());
    let record = generator.record.lock().unwrap();
    assert_eq!(record.safety_checks, 1);
    assert_eq!(record.begin_calls, 0);
    assert_eq!(record.configure_calls, 0);
    assert_eq!(record.generate_calls, 0);
    assert!(record.finishes.is_empty());
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn legacy_video_without_context_bypasses_memory_safety_and_generates() {
    let generator = VideoLifecycleGenerator::new(VideoLifecycleBehavior::SafetyReject);
    let mut input = video_lifecycle_input(gen_core::MemoryStrategy::Resident);
    input.memory_context = None;
    run_loaded_video_generation(&generator, input, &CancelFlag::new(), &mut |_| {})
        .expect("an unsupported/no-contract route keeps the legacy direct-generate path");
    let record = generator.record.lock().unwrap();
    assert_eq!(record.safety_checks, 0);
    assert_eq!(record.begin_calls, 0);
    assert_eq!(record.configure_calls, 0);
    assert_eq!(record.generate_calls, 1);
    assert!(record.finishes.is_empty());
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_lifecycle_finishes_once_for_success_cancel_error_and_configuration_failure() {
    for (behavior, expected_finish, expected_generate) in [
        (
            VideoLifecycleBehavior::Success,
            gen_core::MemoryRunOutcome::Complete,
            1,
        ),
        (
            VideoLifecycleBehavior::Canceled,
            gen_core::MemoryRunOutcome::Canceled,
            1,
        ),
        (
            VideoLifecycleBehavior::GenerateError,
            gen_core::MemoryRunOutcome::Error {
                message: "unsupported: fixture generation failure".to_owned(),
            },
            1,
        ),
        (
            VideoLifecycleBehavior::ConfigureError,
            gen_core::MemoryRunOutcome::Error {
                message: "unsupported: fixture configure failure".to_owned(),
            },
            0,
        ),
    ] {
        let generator = VideoLifecycleGenerator::new(behavior);
        let result = run_loaded_video_generation(
            &generator,
            video_lifecycle_input(gen_core::MemoryStrategy::BoundedDecode),
            &CancelFlag::new(),
            &mut |_| {},
        );
        assert_eq!(result.is_ok(), behavior == VideoLifecycleBehavior::Success);
        let record = generator.record.lock().unwrap();
        assert_eq!(record.safety_checks, 1, "{behavior:?}");
        assert_eq!(record.begin_calls, 1, "{behavior:?}");
        assert_eq!(record.configure_calls, 1, "{behavior:?}");
        assert_eq!(record.generate_calls, expected_generate, "{behavior:?}");
        assert_eq!(record.finishes, vec![expected_finish], "{behavior:?}");
        assert_eq!(
            record.generated_memory[0].unwrap().decode_tile_edge,
            Some(256)
        );
        assert_eq!(record.generated_memory[0].unwrap().decode_overlap, Some(32));
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn resident_video_context_runs_the_handshake_without_overriding_provider_defaults() {
    let generator = VideoLifecycleGenerator::new(VideoLifecycleBehavior::Success);
    run_loaded_video_generation(
        &generator,
        video_lifecycle_input(gen_core::MemoryStrategy::Resident),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .expect("resident request");
    let record = generator.record.lock().unwrap();
    assert_eq!(record.safety_checks, 1);
    assert_eq!(record.begin_calls, 1);
    assert_eq!(record.configure_calls, 1);
    assert_eq!(record.generate_calls, 1);
    assert_eq!(record.generated_memory, vec![None, None]);
    assert_eq!(record.finishes, vec![gen_core::MemoryRunOutcome::Complete]);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn production_admission_handoff_drives_provider_safety_and_lifecycle() {
    let generator = VideoLifecycleGenerator::new(VideoLifecycleBehavior::Success);
    let mut input = video_lifecycle_input(gen_core::MemoryStrategy::Resident);
    input.memory = None;
    input.memory_context = None;
    let selected_memory = gen_core::GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(256),
        decode_overlap: Some(32),
        ..Default::default()
    };
    apply_video_admission_outcome(
        &mut input,
        crate::video_admission::VideoAdmissionOutcome {
            memory: Some(selected_memory),
            context: Some(video_lifecycle_context(
                gen_core::MemoryStrategy::BoundedDecode,
            )),
            refusal: None,
        },
    )
    .expect("selected admission reaches the loaded input");
    run_loaded_video_generation(&generator, input, &CancelFlag::new(), &mut |_| {})
        .expect("production handoff generates");
    let record = generator.record.lock().unwrap();
    assert_eq!(record.safety_checks, 1);
    assert_eq!(record.begin_calls, 1);
    assert_eq!(record.configure_calls, 1);
    assert_eq!(record.generate_calls, 1);
    assert_eq!(
        record.generated_memory,
        vec![Some(selected_memory), Some(selected_memory)]
    );
    assert_eq!(record.finishes, vec![gen_core::MemoryRunOutcome::Complete]);
}

#[test]
fn production_callback_retains_request_state_and_the_tested_admission_handoff() {
    const SOURCE: &str = include_str!("wan.rs");
    let funnel = SOURCE
        .split_once("pub(super) async fn generate_video_using(")
        .and_then(|(_, rest)| rest.split_once("// Bind the blocking generation task"))
        .map(|(body, _)| body)
        .expect("generate_video_using production callback region remains identifiable");
    assert!(
        funnel.contains("with_cached_generator_for_request_using("),
        "the production funnel must use the request-aware cache callback that supplies cold/warm, \
         load policy, external baseline, and fixed provider-resident delta"
    );
    assert!(
        funnel.contains("with_uncached_generator("),
        "the in-place Candle load must use the uncached request-aware callback that captures its \
         own cold-load provider-resident delta"
    );
    assert!(
        !funnel.contains("MemoryCacheState::Cold,\n                                OffloadPolicy::Sequential,\n                                0,\n                                0,"),
        "the uncached Candle load may not forge a zero-byte provider attribution"
    );
    assert!(
        funnel.contains("apply_video_admission_outcome(&mut input, outcome)?;"),
        "the production callback must pass both selected memory and lifecycle context through the \
         same handoff exercised by production_admission_handoff_drives_provider_safety_and_lifecycle"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn calibrated_video_memory_surface_is_exact_t2v_only() {
    let mut input = VideoGenInput {
        fps: 24,
        ..VideoGenInput::default()
    };
    assert!(calibrated_video_memory_surface(&input, "text_to_video"));
    assert!(calibrated_video_memory_surface(
        &VideoGenInput {
            fps: 30,
            ..VideoGenInput::default()
        },
        "text_to_video"
    ));

    assert!(!calibrated_video_memory_surface(&input, "image_to_video"));
    input.conditioning.push(Conditioning::Reference {
        image: gen_core::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        },
        strength: None,
    });
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.conditioning.clear();
    input.enhance_prompt = true;
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.enhance_prompt = false;
    input.use_uncensored_enhancer = true;
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.use_uncensored_enhancer = false;
    input.uncensored_enhancer_dir = Some(PathBuf::from("/fixture/enhancer"));
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.uncensored_enhancer_dir = None;
    input.adapters.push(AdapterSpec {
        path: PathBuf::from("/fixture/adapter.safetensors"),
        scale: 1.0,
        kind: gen_core::AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    });
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.adapters.clear();
    input.video_mode = Some("no_audio".to_owned());
    assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    input.video_mode = None;
    for fps in [0, 1, 23, 31, 60] {
        input.fps = fps;
        assert!(!calibrated_video_memory_surface(&input, "text_to_video"));
    }
}

#[test]
fn sibling_video_modules_construct_private_types_through_owner_seams() {
    const CANDLE: &str = include_str!("candle.rs");
    const TESTS: &str = include_str!("tests.rs");

    assert!(
        !CANDLE.contains("ComfyuiWanExperts {"),
        "candle must construct wan-owned ComfyuiWanExperts through its owner-module constructor"
    );
    assert!(
        !TESTS.contains(&["ComfyuiWanPaths", " {"].concat()),
        "centralized tests must construct candle-owned ComfyuiWanPaths through its test fixture"
    );
}

#[test]
fn backend_neutral_video_policy_has_no_candle_twins() {
    const CANDLE: &str = include_str!("candle.rs");
    const BERNINI: &str = include_str!("bernini.rs");

    for stale_twin in [
        "fn candle_wan_lightning_on(",
        "fn candle_wan_sampling(",
        "fn candle_resolve_lightning_loras(",
        "fn candle_resolve_wan_adapters(",
        "fn candle_scail2_sampling(",
        "fn candle_scail2_engine_video_mode(",
    ] {
        assert!(
            !CANDLE.contains(stale_twin),
            "backend-neutral policy regressed to candle twin {stale_twin}"
        );
    }
    for stale_twin in [
        "fn candle_bernini_engine_id(",
        "fn candle_bernini_engine_video_mode(",
        "fn candle_bernini_raw_settings(",
    ] {
        assert!(
            !BERNINI.contains(stale_twin),
            "backend-neutral Bernini policy regressed to candle twin {stale_twin}"
        );
    }
}

#[test]
fn candle_video_families_keep_explicit_cross_module_boundaries() {
    const PARENT: &str = include_str!("mod.rs");
    const BERNINI: &str = include_str!("bernini.rs");
    const CANDLE: &str = include_str!("candle.rs");
    const TESTS: &str = include_str!("tests.rs");
    const VACE: &str = include_str!("vace.rs");
    let bernini_imports = BERNINI
        .split_once(
            "// ---------------------------------------------------------------------------",
        )
        .map(|(imports, _)| imports)
        .expect("bernini.rs keeps its imports before the family implementation banner");
    let candle_imports = CANDLE
        .split_once(
            "// ---------------------------------------------------------------------------",
        )
        .map(|(imports, _)| imports)
        .expect("candle.rs keeps its imports before the family implementation banner");
    let test_imports = TESTS
        .split_once("#[test]")
        .map(|(imports, _)| imports)
        .expect("video_jobs tests keep cfg imports before the first test");

    for dependency in ["mochi::{", "svd::{", "vace::{", "wan::{"] {
        assert!(
            candle_imports.contains(dependency),
            "candle.rs import region lost its explicit {dependency} family dependency"
        );
    }
    assert!(
        bernini_imports.contains("use super::ltx::extract_clip_frames;"),
        "Bernini's Candle import region must retain the shared LTX clip extractor"
    );
    for symbol in [
        "load_clip_anchor_frames",
        "ClipFramePosition",
        "VideoGenInput",
        "generate_video_using",
    ] {
        assert!(
            candle_imports.contains(symbol),
            "candle.rs import region lost its explicit shared dependency on {symbol}"
        );
    }
    assert!(
        VACE.contains("feature = \"backend-candle\"")
            && VACE.contains("use super::wan::ClipFramePosition;"),
        "VACE shared helpers must import only their shared Wan contract on the Candle cfg"
    );
    assert!(
        VACE.contains("generate_video, resolve_wan_model_dir, resolve_wan_quant,")
            && VACE.contains("VideoGenInput,"),
        "VACE macOS implementation must import generation and MLX-only Wan resolvers separately"
    );
    assert!(
        !VACE.contains("use super::wan::{generate_video, ClipFramePosition")
            && !VACE.contains("ClipFramePosition, VideoGenInput"),
        "Candle-shared VACE imports must not pull in macOS-only generation symbols"
    );
    assert!(
        test_imports.contains("use super::{bernini::*, ltx::*, mochi::*, svd::*, wan::*};")
            && !test_imports
                .contains("use super::{bernini::*, ltx::*, mochi::*, svd::*, vace::*, wan::*};"),
        "Candle tests must import the LTX helpers without the unused VACE family glob"
    );
    for dispatch in [
        "generate_candle_video",
        "generate_candle_scail2",
        "generate_candle_wan_vace",
        "generate_candle_bernini",
    ] {
        assert!(
            PARENT.contains(dispatch),
            "video_jobs parent lost the Candle dispatch import for {dispatch}"
        );
    }
}

use serde_json::json;

fn request(value: Value) -> VideoRequest {
    VideoRequest::from_payload(&value.as_object().cloned().unwrap())
}

#[test]
fn seedvr2_probe_and_scratch_plan_cover_both_sequences_before_decode() {
    let stderr = r#"
          Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 30 fps
        frame=  347 fps=0.0 q=-0.0 Lsize=N/A time=00:00:11.56
        "#;
    assert_eq!(
        parse_seedvr2_source_probe(stderr),
        Some(Seedvr2SourceProbe {
            frame_count: 347,
            width: 1920,
            height: 1080,
        })
    );
    let source = seedvr2_estimated_output_bytes(347, 1920, 1080);
    let output = seedvr2_estimated_output_bytes(347, 3840, 2160);
    assert_eq!(
        seedvr2_estimated_scratch_bytes(347, 1920, 1080, 3840, 2160),
        source + output,
        "the pre-decode plan must reserve the coexisting native and upscaled PNG sequences"
    );
}

#[test]
fn seedvr2_probe_uses_effective_output_geometry_after_rotation() {
    let stderr = r#"
          Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 30 fps
        Output #0, null, to 'pipe:':
          Stream #0:0: Video: wrapped_avframe, yuv420p, 1080x1920, 30 fps
        frame=  347 fps=0.0 q=-0.0 Lsize=N/A time=00:00:11.56
        "#;
    assert_eq!(
        parse_seedvr2_source_probe(stderr),
        Some(Seedvr2SourceProbe {
            frame_count: 347,
            width: 1080,
            height: 1920,
        }),
        "rotation metadata must use the effective mapped geometry produced by the PNG decode"
    );
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn comfyui_wan_a14b_gate_rejects_32gb_and_admits_large_card() {
    let root_guard = tempfile::Builder::new()
        .prefix("sc13621_comfy_wan_gate_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let high = root.join("high.safetensors");
    let low = root.join("low.safetensors");
    std::fs::File::create(&high)
        .unwrap()
        .set_len(20 * 1024 * 1024 * 1024)
        .unwrap();
    std::fs::File::create(&low)
        .unwrap()
        .set_len(20 * 1024 * 1024 * 1024)
        .unwrap();
    let make_paths =
        || ComfyuiWanPaths::test_fixture(high.clone(), low.clone(), None, None, root.to_path_buf());
    assert!(
        comfyui_wan_vram_preflight(
            make_paths(),
            "0",
            Some(crate::vram_gate::VramBudget {
                free_gb: 32.0,
                total_gb: 32.0,
            }),
        )
        .is_err(),
        "a 32 GB card must not reach the external A14B load"
    );
    assert!(
        comfyui_wan_vram_preflight(
            make_paths(),
            "0",
            Some(crate::vram_gate::VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }),
        )
        .is_ok(),
        "larger-card behavior remains admitted"
    );
    assert_eq!(
        candle_video_offload_policy(WAN_COMFYUI_T2V_ENGINE),
        OffloadPolicy::Sequential
    );
}

/// sc-13585: on Windows the free-space probe was inert — the old code matched an `fsutil` output
/// line containing both "avail" and "free bytes", which no modern Windows 11 layout prints, so
/// `available_disk_bytes` returned `None` and the sc-9646 disk-exhaustion guard silently
/// fail-opened on the primary candle platform. The fix uses `fs2::available_space` (a safe wrapper
/// over Win32 `GetDiskFreeSpaceExW`, no localized `fsutil` text to mis-parse), so the probe returns
/// a real, positive figure on every Windows locale — exactly where the pre-fix code returned
/// `None`. Reverting to the old scrape makes this `expect` panic (the dev box prints the modern
/// layout the matcher misses). `seedvr2_disk_guard_rejects_an_impossibly_large_clip` then exercises
/// the full guard on Windows — its rejection assertion only runs once this probe returns `Some` —
/// so together they prove the guard is armed, not just present.
///
/// Windows-only, gated to the candle lane that compiles it — runs in the `windows-candle.yml` CI
/// job (`cargo test -p sceneworks-worker --features backend-candle`).
#[cfg(all(not(unix), feature = "backend-candle"))]
#[test]
fn available_disk_bytes_reports_real_free_space_on_windows() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A real, mounted scratch volume must report positive free space — the whole point of the fix
    // (the pre-fix probe returned None here, disarming the guard).
    let free = available_disk_bytes(dir.path())
        .expect("the Windows free-space probe must return a value for a real directory");
    assert!(
        free > 0,
        "a mounted scratch volume has free space, got {free}"
    );
    // Sanity bound: available space can never exceed the volume's total — pins that we read a
    // coherent volume figure rather than an arbitrary non-zero number.
    let total = fs2::total_space(dir.path()).expect("total volume space");
    assert!(
        free <= total,
        "available {free} must not exceed total {total}"
    );
}

/// sc-12297: the worker's backstop refuses a clip past the model's declared
/// `limits.hardMaxDuration` BEFORE any weights load — pinned at the seam
/// `run_video_generate_job` actually calls, not just as a sceneworks-core free function.
///
/// Kills the mutation that matters: deleting the `duration_limit_error` call from
/// `video_preflight`. Every core test stays green through that — the core function is still
/// correct, it is simply no longer CALLED — which is the precise shape of the dropped-gate bug
/// the sc-11992 review caught surviving 835 green tests.
///
/// Ungated on purpose. `mochi_preflight`'s fit gate is macOS-only, so the candle lane's ONLY
/// duration ceiling is this one; a `#[cfg(target_os = "macos")]` here would leave the platform
/// that has no other protection untested.
#[test]
fn video_preflight_refuses_a_clip_past_the_models_declared_hard_max_duration() {
    // The story's mochi_1 shape: cap 5s, asked for 30s. 30 x 30fps = 900 raw frames, which snap
    // to 901 on Mochi's 6k+1 lattice — and `901 % 6 == 1`, so the engine's own validate_request
    // ACCEPTS it. On the candle lane nothing else says no, which is what makes this gate the
    // whole defense rather than a nicety.
    let over = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
        "duration": 30, "fps": 30,
        "modelManifestEntry": { "limits": { "hardMaxDuration": 5 } }
    }));
    assert_eq!(over.raw_frame_count(), 900);
    assert_eq!(
        over.frame_count(),
        901,
        "on-lattice, and the engine would accept it"
    );
    let Err(WorkerError::InvalidPayload(message)) = video_preflight(&over) else {
        panic!("a 30s clip on a 5s-capped model must be refused before the load");
    };
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(message.contains("5s"), "states the cap: {message}");
    assert!(message.contains("30s"), "states what was asked: {message}");

    // At-cap admits — the shipped 5s default must still run, or the gate has bricked the model.
    let at_cap = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
        "duration": 5, "fps": 30,
        "modelManifestEntry": { "limits": { "hardMaxDuration": 5 } }
    }));
    assert!(
        video_preflight(&at_cap).is_ok(),
        "5s is exactly the cap and is the model's own shipped default"
    );

    // A job carrying no manifest entry (the stub lane, an uncatalogued id whose entry resolves
    // to {}) is UNCONSTRAINED — the gate never blocks without a declared cap.
    let no_entry = request(json!({
        "projectId": "p", "model": "stub-model", "mode": "text_to_video", "prompt": "p",
        "duration": 30
    }));
    assert!(
        video_preflight(&no_entry).is_ok(),
        "no cap declared => no cap"
    );

    // The pre-existing projectId invariant still holds at this seam.
    let no_project = request(json!({ "model": "mochi_1", "mode": "text_to_video" }));
    assert!(matches!(
        video_preflight(&no_project),
        Err(WorkerError::InvalidPayload(m)) if m.contains("projectId")
    ));
}

/// sc-12347: the same backstop on the fps axis — refuses a rate the model does not advertise,
/// and admits a payload that names no rate at all.
///
/// Kills the same class of mutation: deleting the `fps_limit_error` call from `video_preflight`
/// leaves every core test green, because the core function is still correct — just not CALLED.
///
/// Ungated for the same reason as the duration test: `mochi_preflight`'s fit gate is macOS-only
/// AND is a *memory* test, so it cannot refuse an off-spec-but-affordable rate on any lane.
#[test]
fn video_preflight_refuses_an_fps_the_model_does_not_advertise() {
    let mochi_entry = json!({
        "limits": { "hardMaxDuration": 5, "fps": [30] }, "defaults": { "fps": 30 }
    });

    // The story's shape: a request that is LEGALLY 5 seconds — sc-12297's cap admits it — but
    // asks for 60fps. 5 x 60 = 300 raw frames snapping to 301, double the shipped default's
    // 151, and `301 % 6 == 1` so the engine's own validate_request ACCEPTS it. The duration gate
    // cannot see this; only the fps menu refuses it.
    let off_menu = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
        "duration": 5, "fps": 60, "modelManifestEntry": mochi_entry
    }));
    assert_eq!(off_menu.raw_frame_count(), 300);
    assert_eq!(
        off_menu.frame_count(),
        301,
        "on-lattice, and the engine would accept it"
    );
    assert_eq!(
        duration_limit_error(
            &off_menu.model,
            off_menu.duration,
            &off_menu.model_manifest_entry
        ),
        None,
        "the duration cap ADMITS this request — it is legally 5 seconds"
    );
    let Err(WorkerError::InvalidPayload(message)) = video_preflight(&off_menu) else {
        panic!("60fps on a model that advertises only 30 must be refused before the load");
    };
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(
        message.contains("30 fps"),
        "states what is allowed: {message}"
    );
    assert!(
        message.contains("60 fps"),
        "states what was asked: {message}"
    );

    // The model's own advertised rate admits — the shipped default must still run.
    let on_menu = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
        "duration": 5, "fps": 30, "modelManifestEntry": mochi_entry
    }));
    assert!(
        video_preflight(&on_menu).is_ok(),
        "30 is what mochi advertises"
    );
    assert_eq!(on_menu.frame_count(), 151);

    // THE REGRESSION THIS STORY NEARLY SHIPPED: a payload naming NO fps must be ADMITTED. It
    // resolves to the model's declared 30, not the blanket 25 — which is off mochi's menu and
    // would make this gate refuse a perfectly ordinary job (7 of the 10 shipped video models
    // are in that position, so this is the common case, not an edge).
    let no_fps = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p",
        "duration": 5, "modelManifestEntry": mochi_entry
    }));
    assert_eq!(
        no_fps.fps, 30,
        "resolved from the model's declared defaults.fps"
    );
    assert!(
        video_preflight(&no_fps).is_ok(),
        "an fps-less payload must not be refused by the menu"
    );
    assert_eq!(
        no_fps.frame_count(),
        151,
        "the frame count the manifest documents"
    );

    // A job carrying no manifest entry is UNCONSTRAINED — never block without a declared menu.
    let no_entry = request(json!({
        "projectId": "p", "model": "stub-model", "mode": "text_to_video", "prompt": "p",
        "duration": 5, "fps": 60
    }));
    assert!(
        video_preflight(&no_entry).is_ok(),
        "no menu declared => no menu"
    );
}

// -----------------------------------------------------------------------------------------
// Mochi 1 (epic 1788 / sc-11992). These sit on the SUPERSET cfg because the tier resolver, the
// stride and the on-demand fetches are SHARED by both lanes (one repo, both backends) — so they
// run on the macOS lane AND on the windows-candle lane, which is what makes the candle half
// genuinely covered rather than asserted.
// -----------------------------------------------------------------------------------------

/// Build a Mochi model root in the A6 installed layout: tier dirs as SIBLINGS of the shared
/// T5/tokenizer/VAE co-requisite. `tiers` lists the tier dirs to materialize.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_root(tag: &str, tiers: &[&str], shared: bool) -> tempfile::TempDir {
    let root_guard = tempfile::Builder::new()
        .prefix(&format!("mochi_root_{tag}_"))
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    for tier in tiers {
        let transformer = root.join(tier).join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(transformer.join("model.safetensors"), b"w").unwrap();
        let (quantized, bits) = match *tier {
            "q4" => (true, 4),
            "q8" => (true, 8),
            _ => (false, 0),
        };
        let manifest = if quantized {
            json!({ "quantized": true, "quantization_bits": bits, "quantization_group_size": 64 })
        } else {
            json!({ "quantized": false })
        };
        std::fs::write(
            root.join(tier).join("split_model.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
    }
    if shared {
        for component in ["text_encoder", "tokenizer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
    }
    root_guard
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_request(advanced: Value) -> VideoRequest {
    request(json!({
        "projectId": "p",
        "model": "mochi_1",
        "mode": "text_to_video",
        "prompt": "a calico kitten",
        "advanced": advanced,
    }))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_receipts_are_atomic_across_tier_and_shared_corequisite() {
    let data = tempfile::tempdir().unwrap();
    let hub = data.path().join("hub");
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", hub.to_str().unwrap())]);
    let repo_cache = hub.join("models--SceneWorks--mochi-1-mlx");
    let installed = repo_cache.join("snapshots/installed");
    let newer = repo_cache.join("snapshots/newer");
    for root in [&installed, &newer] {
        std::fs::create_dir_all(root.join("q4/transformer")).unwrap();
        std::fs::write(root.join("q4/transformer/model.safetensors"), b"x").unwrap();
        std::fs::write(root.join("q4/split_model.json"), b"{}").unwrap();
        for component in ["text_encoder", "tokenizer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
            std::fs::write(root.join(component).join("receipt-file"), b"x").unwrap();
        }
    }
    std::fs::create_dir_all(repo_cache.join("refs")).unwrap();
    std::fs::write(repo_cache.join("refs/main"), b"newer").unwrap();
    let marker = data
        .path()
        .join("models")
        .join(safe_download_dir(MOCHI_REPO));
    std::fs::create_dir_all(&marker).unwrap();
    let receipt = |variant: &str, revision: &str, files: Value| {
        json!({
            "repo": MOCHI_REPO, "modelId": "mochi_1", "variant": variant,
            "snapshotRevision": revision, "resolvedFiles": files
        })
    };
    let write_receipts = |tier_revision: &str, shared_revision: &str| {
        std::fs::write(marker.join(INSTALL_MARKER), serde_json::to_vec(&json!({
                "repo": MOCHI_REPO,
                "receipts": [
                    receipt("q4", tier_revision, json!(["q4/split_model.json", "q4/transformer/model.safetensors"])),
                    receipt("default", shared_revision, json!(["text_encoder/receipt-file", "tokenizer/receipt-file", "vae/receipt-file"]))
                ]
            })).unwrap()).unwrap();
    };
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        ..Settings::from_env()
    };
    let request = mochi_request(json!({ "mlxQuantize": 4 }));

    write_receipts("installed", "installed");
    assert_eq!(
        resolve_mochi_model_dir(&settings, &request).unwrap(),
        installed.join("q4")
    );
    write_receipts("installed", "newer");
    assert_ne!(
        resolve_mochi_model_dir(&settings, &request).unwrap(),
        installed.join("q4")
    );
}

/// `advanced.mlxQuantize` selects WHICH TIER DIR to load — the story's "selecting q4/q8/bf16
/// loads the right tier" AC. A tier is a DIRECTORY, not a requant toggle (`supported_quants: &[]`
/// on both descriptors), so this mapping is the entire quant mechanism.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_tier_order_maps_mlx_quantize_to_a_tier_dir() {
    // No explicit pick ⇒ q4-first (the video lane's sc-10859 carve-out; bf16 never a default).
    assert_eq!(mochi_tier_order(&mochi_request(json!({}))), &["q4", "q8"]);
    assert!(!mochi_tier_order(&mochi_request(json!({}))).contains(&"bf16"));
    // Explicit small ⇒ still q4-first.
    assert_eq!(
        mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 4 }))),
        &["q4", "q8"]
    );
    // q8 ⇒ q8-first. Accepted as int OR string (the manifest/UI can send either).
    assert_eq!(
        mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 8 }))),
        &["q8", "q4"]
    );
    assert_eq!(
        mochi_tier_order(&mochi_request(json!({ "mlxQuantize": "8" }))),
        &["q8", "q4"]
    );
    // Dense bf16 ⇒ the `<= 0` opt-in rule, and a literal 16 (bf16 IS 16-bit).
    for dense in [json!({ "mlxQuantize": 0 }), json!({ "mlxQuantize": -1 })] {
        assert_eq!(
            mochi_tier_order(&mochi_request(dense)),
            &["bf16", "q8", "q4"]
        );
    }
    assert_eq!(
        mochi_tier_order(&mochi_request(json!({ "mlxQuantize": 16 }))),
        &["bf16", "q8", "q4"]
    );
}

/// The resolver returns the TIER DIR (not the model root), honors the requested tier, and folds
/// in the shared co-requisite gate. Pins the `WeightsSource::Dir` = tier-dir contract the
/// provider's parent-resolution (`resolve_component_root`) depends on.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn resolve_mochi_model_dir_picks_the_requested_tier_dir() {
    let root_guard = mochi_root("resolve", &["q4", "q8", "bf16"], true);
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..Settings::from_env()
    };
    // Route the resolver at the fixture via the operator override.
    temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
        for (advanced, want) in [
            (json!({}), "q4"),
            (json!({ "mlxQuantize": 4 }), "q4"),
            (json!({ "mlxQuantize": 8 }), "q8"),
            (json!({ "mlxQuantize": 0 }), "bf16"),
        ] {
            let dir = resolve_mochi_model_dir(&settings, &mochi_request(advanced.clone()))
                .unwrap_or_else(|e| panic!("resolve {advanced} failed: {e:?}"));
            assert_eq!(
                dir,
                root.join(want),
                "advanced {advanced} must resolve the {want} TIER dir"
            );
            // The tier dir's parent is the model root — where the provider finds the shared
            // T5/VAE. If the resolver ever returned the root itself this breaks.
            assert_eq!(dir.parent().unwrap(), root);
        }
    });
}

/// A complete tier with NO shared co-requisite must NOT resolve. The T5/tokenizer/VAE are a
/// SEPARATE download (`coRequisite`, sc-9696), so "q4 present, text_encoder absent" is a real
/// user state — and resolving it would dead-end inside the provider on a missing-file error
/// instead of telling the user what to re-download.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn resolve_mochi_model_dir_requires_the_shared_corequisite() {
    let root_guard = mochi_root("noshared", &["q4"], false);
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..Settings::from_env()
    };
    temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
        assert!(
            mochi_tier_subdir(root, &["q4"]).is_none(),
            "a tier without the shared T5/VAE siblings is not loadable"
        );
        assert!(
            resolve_mochi_model_dir(&settings, &mochi_request(json!({}))).is_err(),
            "must error rather than resolve an unloadable tier"
        );
    });
}

/// A half-downloaded tier (manifest without weights, or weights without the manifest) must fall
/// THROUGH to the next tier in the order rather than half-loading.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_tier_subdir_skips_an_incomplete_tier() {
    let root_guard = mochi_root("partial", &["q4"], true);
    let root = root_guard.path();
    // A `q8/` that carries only the manifest — the shape a mid-flight download leaves on disk.
    std::fs::create_dir_all(root.join("q8")).unwrap();
    std::fs::write(root.join("q8").join("split_model.json"), b"{}").unwrap();

    assert!(!mochi_tier_dir_is_complete(&root.join("q8")));
    assert_eq!(
        mochi_tier_subdir(root, &["q8", "q4"]),
        Some(root.join("q4")),
        "a torn q8 must fall through to the complete q4, never half-load"
    );
}

/// The quant marker rides the RESOLVED tier's own `split_model.json` — the same manifest the
/// provider asserts against — so the spec can never disagree with the dir we picked (which would
/// be a hard load error), and the asset record names the tier actually loaded.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_tier_quant_reads_the_tier_manifest() {
    let root_guard = mochi_root("quant", &["q4", "q8", "bf16"], true);
    let root = root_guard.path();
    assert_eq!(mochi_tier_quant(&root.join("q4")), Some(Quant::Q4));
    assert_eq!(mochi_tier_quant(&root.join("q8")), Some(Quant::Q8));
    // Dense tier ⇒ no quant assertion (the provider loads it dense).
    assert_eq!(mochi_tier_quant(&root.join("bf16")), None);
    // A manifest-less dir is dense-by-absence, never a guess.
    assert_eq!(mochi_tier_quant(&root.join("nope")), None);
}

/// `mochi_1` is the engine id on BOTH backends — no `_distilled`-style split, unlike LTX. (The
/// candle lane's equivalent is pinned by `candle_mochi_resolves_the_engine_and_never_falls_to_the_stub`.)
#[cfg(target_os = "macos")]
#[test]
fn mochi_engine_id_is_the_model_id_and_matches_nothing_else() {
    assert_eq!(mochi_engine_id("mochi_1"), Some("mochi_1"));
    for other in [
        "wan_2_2",
        "ltx_2_3",
        "svd",
        "scail2_14b",
        "bernini",
        "mochi",
    ] {
        assert!(mochi_engine_id(other).is_none(), "{other} is not mochi_1");
    }
}

/// **The frame-stride seam** (sc-11992): Mochi must use its own `6k+1` snap, not the
/// `wan_frame_count` else-arm every non-LTX model used to fall into.
///
/// The failure mode is worth stating precisely, because the story brief mis-stated it. It claimed
/// `wan_frame_count(150) = 145`, that `145 % 6 == 1` makes the runtime ACCEPT it, and that a 5 s
/// request would therefore silently render 4.83 s. The arithmetic does not hold:
/// `wan_frame_count(150)` is **149** (`150 − ((150−1) % 4)` = 150 − 1), and `149 % 6 == 5`, so the
/// engine's `validate_request` **hard-rejects** it. The real bug is a LOUD failure — every
/// default-duration Mochi job dies on "num_frames must be 1 + 6·k (got 149)" — not a silent
/// wrong-length clip.
///
/// Stronger still, a silent wrong length is IMPOSSIBLE on this pairing, and the test pins that:
/// whenever the Wan stride's answer happens to land on Mochi's lattice it is provably EQUAL to
/// Mochi's snap, so the substitution either agrees or errors — it never lies. (Why: if
/// `wan(raw) = W` is `≡ 1 mod 6` then `raw − W ∈ 0..=3`, so `W` is Mochi's `lower` and
/// `raw − W ≤ 3 ≤ 6 − (raw − W)` makes Mochi's nearest-with-ties-low pick `W` too.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn frames_are_the_mochi_lattice_not_wans() {
    // THE WIRING (not just the functions): the shared ladder BOTH generation lanes call must map
    // the mochi_1 MODEL ID onto the 6k+1 stride. This is what fails if a lane is (re-)pointed at
    // `wan_frame_count` — asserting `mochi_frame_count` vs `wan_frame_count` in isolation would
    // NOT, since both functions stay correct while the caller picks the wrong one.
    assert_eq!(
        video_frame_count("mochi_1", 150),
        151,
        "mochi_1 must resolve the 6k+1 stride through the shared ladder"
    );
    assert_eq!(
        video_frame_count("mochi_1", 150),
        mochi_frame_count(150),
        "the mochi arm must BE mochi_frame_count"
    );
    assert_ne!(
        video_frame_count("mochi_1", 150),
        wan_frame_count(150),
        "routing mochi through the wan stride is the bug this pins"
    );
    // The other families keep their own strides — adding Mochi must not perturb them.
    assert_eq!(video_frame_count("ltx_2_3", 150), ltx_frame_count(150));
    assert_eq!(video_frame_count("ltx_2_3_eros", 150), ltx_frame_count(150));
    assert_eq!(video_frame_count("wan_2_2", 150), wan_frame_count(150));
    assert_eq!(video_frame_count("scail2_14b", 150), wan_frame_count(150));
    assert_eq!(video_frame_count("bernini", 150), wan_frame_count(150));

    // The shipped default: 5 s x 30 fps = 150 raw frames.
    assert_eq!(
        mochi_frame_count(150),
        151,
        "the manifest's documented 5 s count"
    );
    assert_eq!(
        wan_frame_count(150),
        149,
        "NOT 145 — the brief's arithmetic was wrong"
    );
    assert_ne!(mochi_frame_count(150), wan_frame_count(150));
    // The default job's real failure on the wan arm: OFF-lattice ⇒ the engine rejects.
    assert_eq!(
        wan_frame_count(150) % 6,
        5,
        "149 is off Mochi's 6k+1 lattice, so validate_request hard-rejects the 5 s default"
    );

    // Mochi's own snap is always ON-lattice — the engine accepts every value we can produce.
    for raw in [1_u32, 6, 7, 30, 90, 150, 151, 163, 900] {
        assert_eq!(
            mochi_frame_count(raw) % 6,
            1,
            "mochi_frame_count({raw}) must sit on the 6k+1 lattice"
        );
    }

    // The invariant: the wan stride can never SILENTLY render a wrong length — over the whole
    // supported duration range it either equals mochi's snap or is rejected by the engine.
    let mut accepted = 0;
    for raw in 1_u32..=1_800 {
        let wan = wan_frame_count(raw);
        if wan % 6 == 1 {
            accepted += 1;
            assert_eq!(
                wan,
                mochi_frame_count(raw),
                "raw {raw}: a wan value the mochi runtime ACCEPTS must equal mochi's own snap, \
                     or the substitution would silently change clip length"
            );
        }
    }
    assert!(accepted > 0, "the invariant must be exercised, not vacuous");
}

/// sc-12371: **no `*_raw_settings` builder may have an opinion about clip length.**
///
/// This is the structural half of the fix, and the reason the class is dead rather than merely
/// patched. Every builder used to compute its own `frameCount` from the REQUEST, independently
/// of the arm that had just told the engine how many frames to render — two answers to one
/// question. They diverged: the arms driving a Wan engine under a non-Wan model id (`bernini`,
/// `scail2_14b`, `external_base_*`) rendered `wan_frame_count(raw)` = 149 while their builder
/// recorded the unsnapped 150, and nothing failed loudly.
///
/// Making the two computations agree would only fix today's models. Deleting the second
/// computation means a builder has nothing to drift FROM: `record_frame_count` stamps the count
/// off the encoded clip at the one seam every video job funnels through. This test is what stops
/// a future builder from quietly reintroducing a second opinion.
#[test]
#[cfg(target_os = "macos")]
fn no_raw_settings_builder_records_its_own_frame_count() {
    let req = |model: &str| {
        request(json!({
            "projectId": "p", "model": model, "mode": "text_to_video",
            "duration": 6.0, "fps": 25, "advanced": { "userKnob": "keep-me" }
        }))
    };
    let builders: Vec<(&str, Value)> = vec![
        ("stub", stub_raw_settings(&req("stub"))),
        ("wan", wan_raw_settings(&req("wan_2_2"), "wan2_2_ti2v_5b")),
        ("ltx", ltx_raw_settings(&req("ltx_2_3"))),
        ("mochi", mochi_raw_settings(&req("mochi_1"), None)),
        ("bernini", bernini_raw_settings(&req("bernini"))),
        ("scail2", scail2_raw_settings(&req("scail2_14b"), false)),
        ("svd", svd_raw_settings(&req("svd"))),
        (
            "wan_vace",
            wan_vace_raw_settings(&req("wan_2_2"), "wan_vace"),
        ),
        (
            "wan_vace_extend",
            wan_vace_extend_raw_settings(&req("wan_2_2")),
        ),
    ];
    for (name, raw) in &builders {
        let map = raw.as_object().expect("raw settings is an object");
        assert!(
            !map.contains_key("frameCount"),
            "{name}_raw_settings must not record a frameCount — `record_frame_count` counts it \
                 off the encoded clip; a builder that recomputes it is the sc-12371 bug returning"
        );
        // The builders still carry their own knobs — this is not vacuously passing on an empty
        // object, and SVD keeps `numFrames` (its engine burst KNOB, a different thing).
        assert!(
            !map.is_empty(),
            "{name}_raw_settings records nothing at all"
        );
    }
    // SVD's `numFrames` is a dispatched knob, not a clip length — it must survive.
    let svd = svd_raw_settings(&req("svd"));
    assert_eq!(svd["numFrames"], json!(25));
}

/// sc-12371: the stamped `frameCount` is the ENCODED CLIP's length, not any prediction of it.
///
/// DISCRIMINATING BY CONSTRUCTION: every probe stamps a count that differs from what the request
/// would have predicted, so a `record_frame_count` that reached back to `request.frame_count()`
/// (or to any lattice at all) is RED. `bernini` at 6 s x 25 fps predicts 149 from the ladder and
/// 150 from the old raw arm — the stamp records neither, because it records the clip.
///
/// This is the property the "make both sides agree" fix could NOT buy: the engine is free to
/// return a count that is nobody's prediction (SVD's fixed burst clamps to its own `numFrames`;
/// a `validate()` re-snap or a truncated decode would too). The asset follows the file.
#[test]
#[cfg(target_os = "macos")]
fn record_frame_count_records_the_encoded_clip_not_the_request() {
    let bernini = request(json!({
        "projectId": "p", "model": "bernini", "mode": "text_to_video",
        "duration": 6.0, "fps": 25
    }));
    assert_eq!(bernini.raw_frame_count(), 150);
    assert_eq!(bernini.frame_count(), 149, "the ladder predicts 149 here");

    // A clip whose length matches NEITHER prediction — the shape sc-12318's stub probe produces
    // (asked for 151, returned 1). The record must follow the frames that exist.
    let odd = EncodedClip { frames: 7, fps: 25 };
    let stamped = odd.record_frame_count(bernini_raw_settings(&bernini));
    assert_eq!(
        stamped["frameCount"],
        json!(7),
        "the asset records the clip that was encoded, not what the request predicted"
    );
    assert_ne!(stamped["frameCount"], json!(bernini.frame_count()));
    assert_ne!(stamped["frameCount"], json!(bernini.raw_frame_count()));
    // The builder's own knobs survive the stamp.
    assert_eq!(stamped["model"], json!("bernini"));
    assert_eq!(stamped["realModelInference"], json!(true));

    // The real pairing: `generate_bernini` hands the engine `wan_frame_count(raw)` and the
    // engine returns that many frames, so the asset records 149 — never the raw 150 the old
    // builder wrote.
    let rendered = wan_frame_count(bernini.raw_frame_count()) as usize;
    assert_eq!(rendered, 149);
    let clip = EncodedClip {
        frames: rendered,
        fps: 25,
    };
    let stamped = clip.record_frame_count(bernini_raw_settings(&bernini));
    assert_eq!(stamped["frameCount"], json!(149));

    // An already-stamped record is corrected, not duplicated (the stamp is the ONE writer).
    let restamped = EncodedClip { frames: 5, fps: 25 }.record_frame_count(stamped);
    assert_eq!(restamped["frameCount"], json!(5));

    // A non-object carries no keys to correct and passes through rather than being wrapped.
    assert_eq!(clip.record_frame_count(Value::Null), Value::Null);
}

/// sc-12371: `EncodedClip::measure` reads the clip, and `duration_seconds` is the FILE's running
/// time — the second half of the story's harm ("claims a duration/frame count the file does not
/// have"). The asset used to echo `request.duration`, so a 6.0 s ask on a Wan engine produced a
/// 149-frame / 5.96 s file described as 6.0 s.
///
/// DISCRIMINATING: 149 frames at 25 fps is 5.96 s, which is NOT the requested 6.0 — so a
/// `duration` that reached back to `request.duration` is RED here.
#[test]
fn encoded_clip_measures_the_file_not_the_request() {
    let frame = |()| RgbFrame {
        width: 2,
        height: 2,
        pixels: vec![0; 12],
    };
    // What a bernini 6 s x 25 fps job actually produces: the Wan stride's 149 frames.
    let decoded = DecodedVideo {
        frames: (0..149).map(|_| frame(())).collect(),
        fps: 25,
        audio: None,
        adapter_apply_reports: Vec::new(),
    };
    let clip = EncodedClip::measure(&decoded);
    assert_eq!(clip.frames, 149, "counted off the frames, not predicted");
    assert_eq!(clip.fps, 25);
    assert!(
        (clip.duration_seconds() - 5.96).abs() < 1e-9,
        "149 frames at 25 fps is 5.96 s — the file's real length, not the 6.0 s requested"
    );
    assert_ne!(
        clip.duration_seconds(),
        6.0,
        "the probe must discriminate: a `duration` echoing request.duration passes otherwise"
    );

    // `encode_inner` clamps the framerate with `.max(1)`; the record mirrors that clamp so the
    // asset always describes the container ffmpeg was actually handed.
    let zero_fps = DecodedVideo {
        frames: vec![frame(()), frame(())],
        fps: 0,
        audio: None,
        adapter_apply_reports: Vec::new(),
    };
    let clip = EncodedClip::measure(&zero_fps);
    assert_eq!(clip.fps, 1, "mirrors encode_inner's `decoded.fps.max(1)`");
    assert_eq!(clip.duration_seconds(), 2.0);
}

/// The candle-lane half of [`no_raw_settings_builder_records_its_own_frame_count`] (sc-12371).
/// Same invariant, different `#[cfg]`: these builders were the other half of the live bug —
/// `candle_bernini` / `candle_scail2`(`_replace`) and `candle_wan_comfyui` (under an
/// `external_base_*` id) all drive Wan engines whose ids `is_wan_model` does not match.
#[test]
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn no_candle_raw_settings_builder_records_its_own_frame_count() {
    let req = |model: &str| {
        request(json!({
            "projectId": "p", "model": model, "mode": "text_to_video",
            "duration": 6.0, "fps": 25, "advanced": { "userKnob": "keep-me" }
        }))
    };
    let builders: Vec<(&str, Value)> = vec![
        (
            "candle_video",
            candle_video_raw_settings(&req("wan_2_2"), "repo"),
        ),
        (
            "candle_video/comfyui",
            candle_video_raw_settings(&req("external_base_wan22_comfyui"), "repo"),
        ),
        (
            "candle_scail2",
            scail2_raw_settings(&req("scail2_14b"), false),
        ),
        ("candle_bernini", bernini_raw_settings(&req("bernini"))),
        (
            "wan_vace",
            wan_vace_raw_settings(&req("wan_2_2"), "wan_vace"),
        ),
        (
            "wan_vace_extend",
            wan_vace_extend_raw_settings(&req("wan_2_2")),
        ),
    ];
    for (name, raw) in &builders {
        let map = raw.as_object().expect("raw settings is an object");
        assert!(
            !map.contains_key("frameCount"),
            "{name}_raw_settings must not record a frameCount — `record_frame_count` counts it \
                 off the encoded clip; a builder that recomputes it is the sc-12371 bug returning"
        );
        assert!(
            !map.is_empty(),
            "{name}_raw_settings records nothing at all"
        );
    }
    // And the stamp still writes the clip's real length on this lane.
    let stamped = EncodedClip { frames: 7, fps: 25 }
        .record_frame_count(bernini_raw_settings(&req("bernini")));
    assert_eq!(stamped["frameCount"], json!(7));
}

/// Mochi's REAL hosted q4 resident footprint, split by component: AsymmDiT 9.007 GiB + T5-XXL
/// bf16 8.871 + AsymmVAE 0.856 = 18.73 GiB. Mirrors `mlx_fit_gate`'s `MOCHI_Q4_RESIDENT_BYTES`
/// (which asserts the same total against the pure gate) — the preflight tests need them SPLIT
/// across the A6 sibling layout so the on-disk scan has to fold the shared components to get the
/// total right.
///
/// On the SUPERSET cfg (sc-12306): both lanes ingest these same hosted tiers, so both gates budget
/// against these exact bytes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_Q4_DIT_BYTES: u64 = 9_670_883_602;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_Q4_TE_BYTES: u64 = 9_524_669_250;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MOCHI_Q4_VAE_BYTES: u64 = 919_551_200;

/// A Mochi A6-layout root (q4 tier + shared `text_encoder`/`vae`/`tokenizer` siblings) whose
/// `.safetensors` report the REAL hosted byte sizes, so the on-disk scan behind either lane's fit
/// gate resolves the true ~18.73 GiB resident footprint instead of the 1-byte stubs `mochi_root`
/// writes.
///
/// The files are SPARSE: `set_len` sets the apparent size with zero allocated blocks on APFS/NTFS,
/// and `sum_safetensors_bytes` reads `metadata().len()`. So this is instant and costs no disk —
/// materializing 18.7 GiB of real zeros per test would not be viable.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_root_real_sized(tag: &str) -> tempfile::TempDir {
    let root_guard = tempfile::Builder::new()
        .prefix(&format!("mochi_preflight_{tag}_"))
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    for (relative, len) in [
        ("q4/transformer/model.safetensors", MOCHI_Q4_DIT_BYTES),
        ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
        ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
    ] {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path).unwrap().set_len(len).unwrap();
    }
    std::fs::create_dir_all(root.join("tokenizer")).unwrap();
    std::fs::write(
        root.join("q4").join("split_model.json"),
        serde_json::to_string(
            &json!({ "quantized": true, "quantization_bits": 4, "quantization_group_size": 64 }),
        )
        .unwrap(),
    )
    .unwrap();
    root_guard
}

/// sc-11992 review: pin the frame lattice AT THE SEAM THE GENERATION ARM CALLS, not just as a free
/// function — `mochi_preflight` is the seam `generate_mochi` delegates the decision to. The arm
/// ITSELF is pinned by `generate_mochi_using_*` (sc-12318); this stays the cheap, direct check of
/// the seam's own contract across budgets and clip lengths.
///
/// Kills the review mutation `video_frame_count(...)` → `wan_frame_count(...)`, which survived
/// 835/0 green when this logic sat inline in `generate_mochi`: the wan stride answers 149 for the
/// shipped 5 s default, and `149 % 6 == 5` is off the lattice the engine accepts.
#[cfg(target_os = "macos")]
#[test]
fn mochi_preflight_coerces_the_frame_count_on_mochis_lattice_by_model_id() {
    let root_guard = mochi_root_real_sized("lattice");
    let root = root_guard.path();
    // A budget no clip can overflow, so ONLY the lattice is under test here.
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "512", || {
        mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 150, 848, 480)
    });

    let preflight = out.expect("a 151-frame clip fits a 512 GB budget");
    assert_eq!(
        preflight.frames, 151,
        "the seam must snap the shipped 5 s default onto the 6k+1 lattice"
    );
    assert_eq!(
        preflight.frames,
        mochi_frame_count(150),
        "the seam's mochi arm must BE mochi_frame_count"
    );
    assert_ne!(
        preflight.frames,
        wan_frame_count(150),
        "routing the seam through the wan stride is the bug this story exists to fix"
    );
    assert_eq!(
        preflight.quant,
        Some(Quant::Q4),
        "the tier dir's split_model.json carries the quant the load asserts"
    );
}

/// sc-11992 review / epic AC "a missing-weights or unsupported environment shows an ACTIONABLE
/// ERROR, not a crash": the fit gate must run on the generation seam, with the REAL coerced frame
/// count. MLX's default error handler is `exit(-1)` — a hard process kill that cannot be mapped to
/// a job error after the fact — so an over-budget job must be refused here or it takes the worker
/// down with it.
///
/// Kills BOTH remaining review mutations, each of which survived 835/0 green:
///   * hardcoding the gate's `frames` argument to `7` ⇒ the gate sees 4.66 GiB of decode instead
///     of 60.56, totals ~25 GiB, and ADMITS this job ⇒ `expect_err` fails.
///   * deleting the `mochi_fit_check(...)?` call ⇒ nothing rejects ⇒ `expect_err` fails.
#[cfg(target_os = "macos")]
#[test]
fn mochi_preflight_rejects_the_5s_default_on_a_64gb_mac() {
    let root_guard = mochi_root_real_sized("reject");
    let root = root_guard.path();
    // A 64 GB Mac — the machine the epic's crash report names.
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 150, 848, 480)
    });

    let message = out
            .expect_err(
                "the shipped 5 s default (151 frames) needs 18.73 GiB weights + 60.56 GiB untiled \
                 decode + 2 GiB OS reserve = 81.3 GiB, which does NOT fit a 64 GB Mac — admitting it \
                 is exactly the SIGKILL this gate exists to prevent",
            )
            .to_string();
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(
        message.contains("Shorten the clip"),
        "leads with the only lever that moves the dominant term: {message}"
    );
}

/// The other half of the gate's contract: on the SAME 64 GB Mac a short clip is ADMITTED. Without
/// this, `mochi_preflight_rejects_the_5s_default_on_a_64gb_mac` would still pass against a gate
/// that blanket-refuses everything — so this is what makes the pair prove the gate is
/// FRAME-SENSITIVE, which is the whole point of a decode peak that scales with clip length.
///
/// Also independently kills the `wan_frame_count` substitution: `wan_frame_count(19) = 17`.
#[cfg(target_os = "macos")]
#[test]
fn mochi_preflight_admits_a_short_clip_on_the_same_64gb_mac() {
    let root_guard = mochi_root_real_sized("admit");
    let root = root_guard.path();
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_preflight("mochi_1", "mochi_1", &root.join("q4"), 19, 848, 480)
    });

    let preflight = out.expect(
        "19 frames (the engine's own DEFAULT_FRAMES) needs 18.73 + 9.32 + 2 = 30.1 GiB, which \
             fits a 64 GB Mac and must NOT be rejected",
    );
    assert_eq!(
        preflight.frames, 19,
        "19 already sits on the 6k+1 lattice, so the snap is a no-op"
    );
}

// -----------------------------------------------------------------------------------------
// Mochi 1 candle/CUDA VRAM fit gate (sc-12306). These run on the windows-candle.yml lane, which
// is the ONLY lane that compiles `generate_candle_video` — the macOS lane and the Linux `parity`
// job never see this code. `mochi_vram_preflight` takes its budget as a parameter precisely so the
// whole decision is exercisable here with no CUDA driver and no GPU.
// -----------------------------------------------------------------------------------------

/// An RTX 5090 — the biggest consumer NVIDIA card, and the machine the story names. 32 GB total,
/// all free (a cold card with nothing loaded).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn rtx_5090() -> Option<crate::vram_gate::VramBudget> {
    crate::vram_gate::apply_vram_cap(None, Some(32.0))
}

/// THE story: the shipped 5 s / 151-frame default is refused BEFORE the load + 64-step denoise, with
/// a message naming the clip-length lever. Needs 18.73 GiB weights + 60.56 GiB untiled decode + 2 GiB
/// headroom ≈ 81.3 GB against 32 GB — so on consumer hardware this is the DEFAULT path, not an edge
/// case.
///
/// Kills the mutations that a compile alone would not:
///   * hardcoding the gate's `frames` to a small number (e.g. B2's 7-frame smoke geometry) ⇒ the gate
///     sees 4.66 GiB of decode instead of 60.56, totals ~25 GB, and ADMITS ⇒ `expect_err` fails.
///   * passing `request.raw_frame_count()` instead of the coerced count ⇒ 150 not 151 — caught by
///     `mochi_vram_preflight_coerces_the_frame_count_on_the_candle_lane` below.
///   * dropping the frames term entirely (reusing the resolution-blind `predicted_peak_gb` shape) ⇒
///     admits ⇒ `expect_err` fails.
///   * reusing the MLX message ⇒ the "unified memory" assert fails.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn mochi_vram_preflight_rejects_the_5s_default_on_an_rtx_5090() {
    let root_guard = mochi_root_real_sized("candle_reject");
    let root = root_guard.path();
    let out = mochi_vram_preflight(
        "mochi_1",
        &root.join("q4"),
        video_frame_count("mochi_1", 150),
        848,
        480,
        "0",
        rtx_5090(),
    );

    let message = out
        .expect_err(
            "the shipped 5 s default (151 frames) needs ~81 GB and NO consumer NVIDIA GPU has \
                 that — admitting it burns a full 64-step denoise before a raw CUDA OOM",
        )
        .to_string();
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(
        message.contains("151-frame"),
        "names the clip length that was refused, so the user can act on it: {message}"
    );
    assert!(
        message.contains("Shorten the clip"),
        "leads with the only lever that moves the dominant term — Mochi has one trained bucket \
             and the tier delta is ~11 GiB against a ~60 GiB decode: {message}"
    );
    assert!(
        message.contains("VRAM") && !message.contains("unified memory"),
        "must be CUDA-worded, not the MLX lane's Mac prose: {message}"
    );
    assert!(
        !message.contains("run on a Mac"),
        "telling a Windows/CUDA user to buy a Mac is the MLX message leaking: {message}"
    );
}

/// The other half of the contract: on the SAME 32 GB card a short clip is ADMITTED. Without this,
/// the reject test above would pass against a gate that blanket-refuses Mochi on every consumer
/// card — so this pair is what proves the gate is FRAME-SENSITIVE, which is the whole point of a
/// decode peak that scales with clip length.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn mochi_vram_preflight_admits_a_short_clip_on_the_same_rtx_5090() {
    let root_guard = mochi_root_real_sized("candle_admit");
    let root = root_guard.path();
    // 7 frames: 18.73 weights + 4.66 decode + 2 headroom ≈ 25.4 GB, which fits 32 GB.
    let out = mochi_vram_preflight("mochi_1", &root.join("q4"), 7, 848, 480, "0", rtx_5090());

    let preflight = out.expect(
        "a 7-frame clip needs ~25 GB and FITS a 32 GB card — a gate that refuses this has \
             wall-rejected hardware that works",
    );
    assert_eq!(
        preflight.quant,
        Some(Quant::Q4),
        "the tier dir's split_model.json carries the quant the candle loader asserts — and it is \
             reachable ONLY through the gate"
    );
}

/// The gate must budget on the COERCED frame count, not the raw request: the decode peak is linear
/// in frames, so gating on a length that never renders sizes the check against fiction. 150 raw
/// snaps to 151 on Mochi's 6k+1 lattice.
///
/// Kills the `wan_frame_count` substitution independently of the MLX lane's copy of this check:
/// `wan_frame_count(150) = 149`, and `149 % 6 == 5` is off the lattice the engine accepts.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn mochi_vram_preflight_coerces_the_frame_count_on_the_candle_lane() {
    let root_guard = mochi_root_real_sized("candle_lattice");
    let root = root_guard.path();
    let tier = root.join("q4");
    // A budget nothing overflows, so ONLY the frame arithmetic is under test.
    let huge = crate::vram_gate::apply_vram_cap(None, Some(512.0));

    // The seam the generation arm passes: `video_frame_count(&request.model, raw)`.
    let frames = video_frame_count("mochi_1", 150);
    assert_eq!(
        frames, 151,
        "the shipped 5 s default snaps onto the 6k+1 lattice"
    );
    assert_ne!(
        frames,
        wan_frame_count(150),
        "routing the candle lane through the wan stride would gate on 149 — off Mochi's lattice"
    );
    assert!(
        mochi_vram_preflight("mochi_1", &tier, frames, 848, 480, "0", huge).is_ok(),
        "151 frames fits a 512 GB budget — the gate rejects by BUDGET, never by duration alone"
    );

    // Frame-sensitivity at the seam: the SAME tier + card, two lengths, two verdicts. A 48 GB card
    // sits between the 7-frame (~25 GB) and 151-frame (~81 GB) totals.
    let card_48 = crate::vram_gate::apply_vram_cap(None, Some(48.0));
    assert!(
        mochi_vram_preflight("mochi_1", &tier, 7, 848, 480, "0", card_48).is_ok(),
        "a short clip fits 48 GB"
    );
    assert!(
        mochi_vram_preflight("mochi_1", &tier, frames, 848, 480, "0", card_48).is_err(),
        "the 5 s default does not fit the SAME 48 GB card — the gate must see the frame count"
    );
}

/// sc-12344: the Wan weights-floor gate is keyed on the ENGINE id, and [`wan_vram_preflight`] is
/// handed `engine_id` — pin that the two agree for every candle video model the router serves.
///
/// This is the mutation nothing else catches. The sceneworks model id and the engine id differ only
/// by underscores (`wan_2_2_t2v_14b` vs `wan2_2_t2v_14b`, `wan_2_2` vs `wan2_2_ti2v_5b`), so passing
/// `&request.model` at the call site instead of `engine_id` still compiles, still renders, and reads
/// **0 bytes** ⇒ no signal ⇒ the gate silently admits every job — dead code that reads as coverage,
/// the exact failure this story was filed to avoid. It also fails loudly if a future engine is added
/// to `candle_video_engine_id` without a decision about its fit gate.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn every_candle_video_model_maps_to_a_gated_or_recorded_exempt_engine() {
    let root_guard = tempfile::Builder::new()
        .prefix("sc12344_engine_ids_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // A populated Wan component tree, so a 0 read can only mean "not keyed to this engine".
    for component in ["transformer", "transformer_2", "text_encoder", "vae"] {
        let path = root.join(component).join("model.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path)
            .unwrap()
            .set_len(1_000)
            .unwrap();
    }

    // The GATED engines: every Wan model must read real bytes through its resolved engine id.
    for model in ["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"] {
        let engine_id = candle_video_engine_id(model).expect("a candle video engine");
        assert!(
            crate::vram_gate::wan_weight_bytes(engine_id, root) > 0,
            "{model} → {engine_id} must be gated; a 0 read means the component map and the engine \
                 id disagree, and the gate is INERT for this model"
        );
        // …and the sceneworks id is NOT the key. This is the slip above, made concrete.
        assert_eq!(
            crate::vram_gate::wan_weight_bytes(model, root),
            0,
            "{model} is the sceneworks id, not the engine id — passing it reads no bytes"
        );
    }

    // The EXEMPT engines (sc-12344's recorded reason: their on-disk bytes are not their loaded set,
    // so a byte-derived floor would wall-reject working cards — see `vram_gate::wan_weight_components`).
    for model in ["ltx_2_3", "ltx_2_3_eros", "svd"] {
        let engine_id = candle_video_engine_id(model).expect("a candle video engine");
        assert_eq!(
            crate::vram_gate::wan_weight_bytes(engine_id, root),
            0,
            "{model} → {engine_id} is exempt by decision; gating it on a dir sum would over-count"
        );
    }

    // Mochi rides its own frame-dependent gate (sc-12306), never this one.
    assert_eq!(
        crate::vram_gate::wan_weight_bytes(
            candle_video_engine_id("mochi_1").expect("a candle video engine"),
            root
        ),
        0
    );
}

/// The gate NO-OPS without a budget signal (the story's explicit AC) and without a weight signal.
/// A worker on a card `nvidia-smi` cannot read, or pointed at a weights dir it cannot scan, must
/// keep rendering exactly as it did before this gate existed — a fit gate that blocks on missing
/// evidence is a regression, not a safety net (sc-12179: never wall-reject a machine that worked).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn mochi_vram_preflight_no_ops_without_a_budget_or_weight_signal() {
    let root_guard = mochi_root_real_sized("candle_nosignal");
    let root = root_guard.path();

    // No budget: `nvidia_vram_budget_gb` → None (non-NVIDIA / unreadable) and no cap set.
    assert!(
        mochi_vram_preflight("mochi_1", &root.join("q4"), 151, 848, 480, "0", None).is_ok(),
        "no budget signal ⇒ admit — the 5 s default is refused ONLY against a real reading"
    );
    assert_eq!(
        crate::vram_gate::apply_vram_cap(None, None),
        None,
        "no real reading + no cap ⇒ no budget, so the wiring above really can pass None"
    );

    // No weights: a dir with no safetensors scans to 0 bytes ⇒ unmeasurable ⇒ admit, even on a
    // tiny card. The fixture needs its OWN root, for two reasons: `mochi_resident_bytes` folds the
    // shared `text_encoder/` + `vae/` siblings from the tier dir's PARENT, so (a) an "empty" tier
    // under `root` above would still scan ~9.7 GiB and be REJECTED — testing the exact opposite of
    // the intent — and (b) hanging it directly off `temp_dir()` would make the parent scan read
    // /tmp, so an unrelated `/tmp/text_encoder` would flake it. A private root has neither problem.
    let bare_root_guard = tempfile::Builder::new()
        .prefix("mochi_candle_nosignal_bare_")
        .tempdir()
        .expect("temp dir");
    let bare_root = bare_root_guard.path();
    let bare = bare_root.join("q4");
    std::fs::create_dir_all(&bare).unwrap();
    assert_eq!(
        crate::mlx_fit_gate::mochi_resident_bytes(&bare),
        0,
        "the fixture must really have no weight signal, or the assert below proves nothing"
    );
    let out = mochi_vram_preflight(
        "mochi_1",
        &bare,
        151,
        848,
        480,
        "0",
        crate::vram_gate::apply_vram_cap(None, Some(4.0)),
    );
    assert!(out.is_ok(), "unmeasurable weights ⇒ no signal ⇒ admit");
}

/// The on-disk scan must fold the SHARED `text_encoder/` + `vae/` siblings from the tier dir's
/// PARENT — both providers set `supports_sequential_offload: false`, so all three components are
/// held for the whole run. Summing only the tier dir would miss ~9.7 GiB (T5-XXL + VAE), over half
/// the resident footprint, and silently under-gate every candle Mochi job.
///
/// This pins that the candle lane reuses the SHARED scan rather than growing its own tier-only one.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn mochi_vram_preflight_folds_the_shared_siblings_on_the_candle_lane() {
    let root_guard = mochi_root_real_sized("candle_siblings");
    let root = root_guard.path();
    assert_eq!(
        crate::mlx_fit_gate::mochi_resident_bytes(&root.join("q4")),
        MOCHI_Q4_DIT_BYTES + MOCHI_Q4_TE_BYTES + MOCHI_Q4_VAE_BYTES,
        "the candle gate must budget on DiT + T5 + VAE (~18.73 GiB), not the tier dir alone"
    );

    // Behavioral consequence: a card sized to fit the DiT alone must still refuse. DiT-only is
    // 9.01 + 4.66 decode + 2 = ~15.7 GB; the true total is 18.73 + 4.66 + 2 = ~25.4 GB. A 20 GB
    // card admits the former and must reject the latter.
    let out = mochi_vram_preflight(
        "mochi_1",
        &root.join("q4"),
        7,
        848,
        480,
        "0",
        crate::vram_gate::apply_vram_cap(None, Some(20.0)),
    );
    assert!(
        out.is_err(),
        "a tier-only scan would admit this job on a 20 GB card and then OOM on the T5 + VAE"
    );
}

// -----------------------------------------------------------------------------------------
// Mochi 1 candle/CUDA PRE-DOWNLOAD fit gate (sc-12373) — the CUDA half of sc-12322.
//
// The gate above refuses before the DENOISE; these refuse before the ~13-20 GiB tier DOWNLOAD.
// The fixture is the real pre-download state: an HF cache holding ONLY the shared co-requisites
// (T5 + VAE + tokenizer), with no tier dir at all.
// -----------------------------------------------------------------------------------------

/// The pre-check refuses the shipped 5 s default on an RTX 5090 with **no tier on disk** — the
/// answer is knowable from the shared co-requisites alone, so the user never pays for the download.
///
/// Floor = 9.73 GiB (T5 + VAE, already present) + 60.56 decode + 2 headroom ≈ 72.3 GB vs 32.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_mochi_precheck_refuses_the_5s_default_before_any_tier_is_downloaded() {
    let root_guard = mochi_root_shared_only("candle_reject");
    let root = root_guard.path();
    // The requested tier does NOT exist yet — the pre-check must still decide.
    let tier_dir = root.join("bf16");
    assert!(
        !tier_dir.exists(),
        "the fixture must model the PRE-download state — no tier dir on disk"
    );
    let out = mochi_vram_precheck(
        "mochi_1",
        "mochi_1",
        Some(&tier_dir),
        150,
        848,
        480,
        "0",
        rtx_5090(),
    );

    let message = out
        .expect_err(
            "the shared components alone already put this job ~40 GB over a 32 GB card — \
                 downloading the tier cannot change that",
        )
        .to_string();
    assert!(
        message.contains("Shorten the clip"),
        "the pre-download refusal carries the same actionable lever: {message}"
    );
    assert!(
        message.contains("151-frame"),
        "the pre-check must gate on the COERCED count (150 raw ⇒ 151), like the full gate: \
             {message}"
    );
}

/// **Why the pre-check measures the WEIGHTS FLOOR and not just the decode term** — the candle twin
/// of `mochi_precheck_needs_the_shared_weights_floor_to_catch_the_64gb_default`, and it needs its
/// OWN budget: at 32 GB the decode term alone already busts the card, so the RTX-5090 test above
/// cannot tell a floor-aware pre-check from a weights-free one.
///
/// At a 64 GB budget it can: decode + headroom = 62.56 (FITS, by 1.44), floor + decode + headroom
/// = 72.28 (does not). So a weights-free pre-check ADMITS, downloads ~20 GiB, and only then
/// refuses — the exact bug. This test fails the moment someone "simplifies" the floor away.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_mochi_precheck_needs_the_shared_weights_floor_to_catch_the_64gb_card() {
    let root_guard = mochi_root_shared_only("candle_floor");
    let root = root_guard.path();
    let tier_dir = root.join("bf16");
    let card_64 = crate::vram_gate::apply_vram_cap(None, Some(64.0));

    // Decode + headroom alone does NOT bust 64 …
    let decode_only = crate::fit_gate::mochi_decode_peak_gb(151, 848, 480) + 2.0;
    assert!(
        decode_only < 64.0,
        "a weights-free pre-check would ADMIT here ({decode_only:.2} < 64) — which is precisely \
             why the shared-components floor is load-bearing"
    );

    // … but the floor the already-present shared components put on disk does.
    let out = mochi_vram_precheck(
        "mochi_1",
        "mochi_1",
        Some(&tier_dir),
        150,
        848,
        480,
        "0",
        card_64,
    );
    assert!(
        out.is_err(),
        "the shared T5 + VAE (~9.73 GiB) push the total to ~72 GB, over the 64 GB card — a \
             decode-only pre-check misses this and charges the user ~20 GiB to learn it"
    );
}

/// The other half of the contract: the pre-check must NOT refuse a job that fits, or it becomes a
/// wall-reject on hardware that works (sc-12179). A short clip on the same 32 GB card is admitted
/// and goes on to download. Without this, the reject tests pass against a pre-check that refuses
/// Mochi outright.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_mochi_precheck_admits_a_short_clip_on_the_same_rtx_5090() {
    let root_guard = mochi_root_shared_only("candle_admit");
    let root = root_guard.path();
    let tier_dir = root.join("bf16");
    // 7 frames: 9.73 floor + 4.66 decode + 2 ≈ 16.4 GB, well inside 32.
    let out = mochi_vram_precheck(
        "mochi_1",
        "mochi_1",
        Some(&tier_dir),
        7,
        848,
        480,
        "0",
        rtx_5090(),
    );
    assert!(
        out.is_ok(),
        "a 7-frame clip fits a 32 GB card — refusing it here would block the download for a job \
             that renders fine"
    );
}

/// No signal ⇒ admit, on every axis: no tier dir resolvable (a first-ever install with no model
/// root), no budget reading, and no weights on disk. The pre-check is the EARLIEST gate, so a
/// false refusal here blocks a download the user may well need — it must never fire on a guess.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_mochi_precheck_admits_without_a_signal() {
    // No dir at all (`mochi_precheck_dir` → None) ⇒ nothing to measure ⇒ admit.
    assert!(
        mochi_vram_precheck("mochi_1", "mochi_1", None, 150, 848, 480, "0", rtx_5090()).is_ok(),
        "no locatable model root ⇒ no signal ⇒ the download must proceed"
    );

    let root_guard = mochi_root_shared_only("candle_nosignal");
    let root = root_guard.path();
    let tier_dir = root.join("bf16");
    // A real floor, but no budget ⇒ admit.
    assert!(
        mochi_vram_precheck(
            "mochi_1",
            "mochi_1",
            Some(&tier_dir),
            150,
            848,
            480,
            "0",
            None
        )
        .is_ok(),
        "no budget signal ⇒ admit — the pre-check refuses only against a real reading"
    );
}

// -----------------------------------------------------------------------------------------
// sc-12318 — driving the ASYNC generation arms.
//
// The `mochi_preflight_*` tests above pin the pure seam, but nothing asserted that
// `generate_mochi` CALLS it: the sc-11992 review found a bypass (destructure the preflight, then
// re-bind `frames = wan_frame_count(...)`) that stayed clippy-clean and green, because a unit test
// could reach the free function but never the caller.
//
// The arm was believed undrivable — `async`, with a live `ApiClient`/`JobSnapshot`. That premise
// was wrong on every count. `ApiClient::new` is pure (a reqwest client + a base URL, no I/O);
// `JobSnapshot` builds from `serde_json::from_value` as it already does elsewhere; and both
// `ensure_mochi_*_present` calls return `Ok(())` on their first line for a request that names no
// tier. The ONLY real obstacle was `generate_video`'s hardcoded `inference_runtime::load`, which
// `generate_video_using` now takes as a parameter — the same split `with_cached_generator` /
// `with_cached_generator_using` already had one level down.
//
// No stub HTTP server is needed: `update_job` is the one API call the progress loop `?`-propagates,
// and it fires only per progress event, so a silent probe generator never reaches it (`heartbeat`
// swallows `WorkerError::Http`; `cancel_requested_peek` swallows everything).
// -----------------------------------------------------------------------------------------

/// What `generate_mochi_using` actually handed the engine, captured from inside the load+run.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Default)]
struct ArmProbe {
    /// The `GenerationRequest` the arm built, or `None` if generation never ran.
    request: std::sync::Arc<std::sync::Mutex<Option<GenerationRequest>>>,
    /// The `LoadSpec` the arm resolved — carries the tier dir + quant marker.
    spec: std::sync::Arc<std::sync::Mutex<Option<LoadSpec>>>,
    /// Provider-owned adapter outcomes returned after generation.
    adapter_reports: std::sync::Arc<std::sync::Mutex<Vec<gen_core::AdapterApplyReport>>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ArmProbe {
    /// The loader to hand `generate_mochi_using`. Records the spec, then yields a generator that
    /// records the request — so a test can assert on both the load and the generate side of the arm
    /// without a tensor backend, weights, or a GPU.
    fn loader(
        &self,
    ) -> impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>> + Send + 'static {
        let seen_spec = std::sync::Arc::clone(&self.spec);
        let seen_request = std::sync::Arc::clone(&self.request);
        let adapter_reports = std::sync::Arc::clone(&self.adapter_reports);
        move |engine_id, spec| {
            *seen_spec.lock().unwrap() = Some(spec.clone());
            Ok(Box::new(ProbeGenerator {
                descriptor: gen_core::ModelDescriptor {
                    // Leaked so the descriptor's `&'static str` id reflects the engine actually
                    // asked for; the probe outlives nothing, so the leak is bounded by the test.
                    id: Box::leak(engine_id.to_owned().into_boxed_str()),
                    family: "test",
                    backend: "probe",
                    modality: gen_core::Modality::Video,
                    capabilities: gen_core::Capabilities::default(),
                    control_kinds: None,
                    required_components: &[],
                },
                request: seen_request,
                adapter_reports,
            }))
        }
    }

    /// The frame count that reached the engine. Panics if generation never ran.
    fn engine_frames(&self) -> u32 {
        self.request
            .lock()
            .unwrap()
            .as_ref()
            .expect("the arm reached the engine")
            .frames
            .expect("a video request carries a frame count")
    }

    /// Whether the arm got as far as loading an engine at all — the "nothing loaded on a refused
    /// job" assertion behind both lanes' pre-download gates.
    ///
    /// Was macOS-only on the stated grounds that "the fit gate is an MLX-lane concern — the candle
    /// lane has none". That stopped being true in sc-12306 (the candle lane grew a pre-load gate)
    /// and again in sc-12373 (a pre-download one), so it is now on the superset cfg and is live
    /// code on both.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn loaded(&self) -> bool {
        self.spec.lock().unwrap().is_some()
    }
}

/// A backend-neutral `Generator` that records the request and returns a minimal clip.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
struct ProbeGenerator {
    descriptor: gen_core::ModelDescriptor,
    request: std::sync::Arc<std::sync::Mutex<Option<GenerationRequest>>>,
    adapter_reports: std::sync::Arc<std::sync::Mutex<Vec<gen_core::AdapterApplyReport>>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl Generator for ProbeGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
        Ok(())
    }

    fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
        self.adapter_reports.lock().unwrap().clone()
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        *self.request.lock().unwrap() = Some(req.clone());
        // Deliberately silent: a `Progress` event would drive `update_job`, the only API call the
        // progress loop hard-fails on, and this test has no server behind `api`.
        Ok(GenerationOutput::Video {
            frames: vec![gen_core::Image {
                width: 2,
                height: 2,
                pixels: vec![0u8; 12],
            }],
            fps: req.fps.unwrap_or(30),
            audio: None,
        })
    }
}

/// A `JobSnapshot` for a Mochi video job. `payload.model` is what the completion metrics read.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_job_snapshot() -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job-mochi-1",
        "type": "video_generate",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": { "model": "mochi_1" },
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.0,
        "stage": "queued",
        "message": "queued",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-07-16T00:00:00Z",
        "updatedAt": "2026-07-16T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("the mochi job snapshot deserializes")
}

/// Drive `generate_mochi_using` against `probe` with the Mochi tier at `root` and an MLX budget of
/// `cap_gb`, from a plain `#[test]` — the env overrides are process-global, so the whole async run
/// has to sit inside the crate-wide env lock the sync `temp_env_vars` holds.
///
/// Both URLs point at a closed port on purpose: reaching the network would be a bug in these tests,
/// and an unroutable URL makes that fail loudly rather than depending on a stub's fidelity.
/// `api_url` alone is NOT enough — a tier fetch dials `huggingface_base_url`
/// (`HuggingFaceSnapshot::resolve` → `{base_url}/api/models/…/tree/…`), which `Settings::from_env()`
/// defaults to the real `https://huggingface.co`; `api_url` only carries progress/heartbeat. That is
/// exactly the trap #1577 fixed in `generate_mochi_using_refuses_before_paying_for_the_tier_download`
/// (see its doc comment), and this helper had it one field over: a caller passing `mlxQuantize: 8`
/// reaches `ensure_mochi_q8_present` → `ensure_mochi_tier_present`, and `SceneWorks/mochi-1-mlx` is a
/// public repo, so the tree resolve SUCCEEDS against the live hub and a ~13 GiB pull can start
/// landing before the heartbeat to the closed `api_url` trips it (sc-12380).
///
/// `HF_HUB_CACHE` is pinned for the same reason: `huggingface_hub_cache_dir` reads it (and
/// `HUGGINGFACE_HUB_CACHE` / `HF_HOME`) BEFORE `data_dir`, so pinning `data_dir` does NOT make the
/// cache lookup hermetic — left ambient, a dev box's real cache decides it, and the desktop injects
/// `HF_HOME` as a matter of course. Pointed at a dir with no Mochi repo, so `huggingface_snapshot_dir`
/// is `None` and `ensure_mochi_tier_present` early-returns before any network touch.
#[cfg(target_os = "macos")]
fn drive_mochi_arm(
    root: &Path,
    cap_gb: &str,
    probe: &ArmProbe,
    request: &VideoRequest,
) -> WorkerResult<(DecodedVideo, Value)> {
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..offline_settings()
    };
    let job = mochi_job_snapshot();
    let loader = probe.loader();
    let hf_cache = root.join("unused-hf-cache");
    temp_env_vars(
        &[
            (
                "HF_HUB_CACHE",
                hf_cache.to_str().expect("utf-8 fixture hub"),
            ),
            (MOCHI_DIR_ENV, root.to_str().expect("utf-8 fixture root")),
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, cap_gb),
        ],
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_mochi_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    request,
                    "mochi_1",
                    "mlx",
                    loader,
                ))
        },
    )
}

/// **The caller-side pin the story asked for.** Drives `generate_mochi_using` end to end and asserts
/// on the frame count that REACHED the engine — so it fails for any way the arm can get the lattice
/// wrong, not just the one the pure seam covers.
///
/// Kills the sc-11992 review bypass that survived `mochi_preflight`: destructuring the preflight and
/// then re-binding `frames = wan_frame_count(request.raw_frame_count())` is clippy-clean and leaves
/// the gate running on the correct 151, so every existing test stays green — but the engine is handed
/// 149, and `149 % 6 == 5` is off Mochi's lattice, which `validate_request` hard-rejects.
#[cfg(target_os = "macos")]
#[test]
fn generate_mochi_using_hands_the_engine_the_mochi_lattice_frame_count() {
    let root_guard = mochi_root_real_sized("arm_lattice");
    let root = root_guard.path();
    let probe = ArmProbe::default();
    // A budget no clip can overflow, so the fit gate cannot be what fails this test.
    let out = drive_mochi_arm(root, "512", &probe, &mochi_request(json!({})));

    let (decoded, raw_settings) = out.expect("a 151-frame clip fits a 512 GB budget");
    assert_eq!(
        probe.engine_frames(),
        151,
        "the ARM must hand the engine the shipped 5 s default snapped onto Mochi's 6k+1 lattice"
    );
    assert_ne!(
        probe.engine_frames(),
        wan_frame_count(150),
        "routing the arm through the wan stride is the bug epic 1788 exists to fix — the engine \
             hard-rejects 149 (149 % 6 == 5)"
    );
    assert_eq!(
        probe
            .spec
            .lock()
            .unwrap()
            .as_ref()
            .expect("a load ran")
            .quantize,
        Some(Quant::Q4),
        "the arm must carry the resolved tier's quant onto the LoadSpec"
    );
    // sc-12371 REPLACED this pin's premise. It used to assert that the arm's `raw_settings`
    // carried a `frameCount` equal to `probe.engine_frames()` — i.e. that core's INDEPENDENT
    // mirror of the lattice happened to agree with the arm. That mirror is gone: the builder no
    // longer has an opinion, and the asset's `frameCount` is stamped from the ENCODED clip at
    // the funnel (`record_frame_count`). So the arm must carry no count...
    assert!(
        raw_settings.get("frameCount").is_none(),
        "the arm must not record a clip length — the funnel stamps it off the encoded frames"
    );
    // ...and this probe is the exact case that proves why recording beats predicting: the arm
    // asked the engine for 151 frames and the stub returned ONE. The old builder would have
    // written 151 onto a 1-frame clip — a sidecar lying about a file that is right there. The
    // stamp records what actually came back.
    assert_eq!(
        decoded.frames.len(),
        1,
        "the probe returns a one-frame clip"
    );
    assert_eq!(
        EncodedClip::measure(&decoded).record_frame_count(raw_settings)["frameCount"],
        json!(1),
        "the asset records the clip that exists (1), not the 151 the arm requested"
    );
    assert_ne!(
        u64::from(probe.engine_frames()),
        1,
        "the probe must discriminate: if the stub returned exactly what was requested, \
             recording and predicting would agree and this would pin nothing"
    );
}

/// The gate half of the same pin: on a 64 GB Mac the shipped 5 s default must be REFUSED, and
/// refused **before the engine is loaded** — MLX's default error handler is `exit(-1)`, so an
/// over-budget job that reaches the load takes the worker down with it rather than failing the job.
///
/// `probe.loaded()` is what makes this a pre-load assertion rather than just an error check: it is
/// only ever set from inside the loader, so it proves the arm short-circuited on the gate. Kills the
/// "inline the seam" bypass (`video_frame_count` + `mochi_tier_quant`, no `mochi_fit_check`), which
/// is green today and caught only by clippy's dead-code cascade.
#[cfg(target_os = "macos")]
#[test]
fn generate_mochi_using_refuses_an_over_budget_clip_before_loading_the_engine() {
    let root_guard = mochi_root_real_sized("arm_reject");
    let root = root_guard.path();
    let probe = ArmProbe::default();
    let out = drive_mochi_arm(root, "64", &probe, &mochi_request(json!({})));

    let message = out
        .err()
        .expect(
            "the shipped 5 s default (151 frames) needs 18.73 GiB weights + 60.56 GiB untiled \
                 decode + 2 GiB OS reserve = 81.3 GiB, which does NOT fit a 64 GB Mac",
        )
        .to_string();
    assert!(
        message.contains("Shorten the clip"),
        "the arm must surface the gate's actionable error: {message}"
    );
    assert!(
            !probe.loaded(),
            "the fit gate must refuse BEFORE the engine load — a load that starts over-budget is the \
             exit(-1) SIGKILL the gate exists to prevent, and cannot be mapped to a job error after \
             the fact"
        );
}

/// **The floor-vs-full seam** — the one case where sc-12322's two gates DISAGREE, and the only
/// thing that pins the post-download `mochi_preflight` call now that `mochi_precheck` runs the same
/// `mochi_fit_check` earlier.
///
/// Why the sibling tests can't cover this: they stage a COMPLETE tier, so the pre-check measures the
/// full weights and refuses first — the preflight becomes unobservable, and inlining it away stays
/// green (caught only by clippy's dead-code cascade, not by `cargo test`; sc-12318 AC #1).
///
/// This makes the two verdicts diverge, with **no download and no network**:
///   * `mlxQuantize: 8` ⇒ the pre-check names the **absent** `q8/` dir, so `mochi_resident_bytes`
///     folds only the shared `text_encoder/` + `vae/` siblings — a 9.73 GiB FLOOR (`q4/` is not a
///     folded component). At 115 frames: 9.73 + 46.58 + 2 = **58.31 ⇒ ADMITS** a 64 GB budget.
///   * `ensure_mochi_q8_present` then no-ops rather than fetching: `drive_mochi_arm`'s `data_dir`
///     has no HF cache, so `huggingface_snapshot_dir` is `None` and `ensure_mochi_tier_present`
///     early-returns `Ok(())`. That is what keeps this hermetic — the divergence normally needs a
///     real ~13 GiB download to materialize the tier.
///   * `resolve_mochi_model_dir` falls back to the installed `q4/`, so `mochi_preflight` sees the
///     FULL 18.73 GiB: 18.73 + 46.58 + 2 = **67.32 ⇒ REFUSES**.
///
/// So the job is admitted by the floor and refused by the full check — exactly the case the
/// post-download gate exists for. It is REACHABLE in production (a 64 GB Mac, bf16/q8 not yet
/// installed, ~2.8-4.2 s clip — well under `hardMaxDuration`'s 151 frames), and without the
/// preflight those jobs run into MLX's `exit(-1)`.
///
/// Kills the sc-12318 AC #1 mutation — inline `video_frame_count` + `mochi_tier_quant` in place of
/// `mochi_preflight` — which passes 856/0 on main: the pre-check admits here, so nothing else
/// refuses, the engine loads, and both assertions below fail.
#[cfg(target_os = "macos")]
#[test]
fn generate_mochi_using_refuses_after_the_fetch_when_the_precheck_floor_admitted() {
    // q4 installed at its real hosted size; q8 is NOT — the fixture stages no other tier.
    let root_guard = mochi_root_real_sized("arm_floor_vs_full");
    let root = root_guard.path();
    let probe = ArmProbe::default();
    // 3.8 s x 30 fps = 114 raw ⇒ 115 frames at 848x480, inside the 109..127 divergence window on a
    // 64 GB Mac. Every geometry term is EXPLICIT because `VideoRequest`'s defaults are none of
    // Mochi's, and each default silently moves this OFF the seam — into the region where the floor
    // and the full check agree — leaving a green test that proves nothing. The window is only ~9 GiB
    // wide (the q4 tier), so it is unforgiving of drift:
    //   * `fps` defaults to 25, not Mochi's 30 ⇒ 95 raw ⇒ 97 frames.
    //   * the frame defaults to 768x512, not Mochi's one trained 848x480 bucket.
    //   * `modelManifestEntry` carries `requiresDimensionsMultipleOf: 16` (Mochi's declared value)
    //     because the DEFAULT stride is 64 — under which 848 floors to 832 and the decode term,
    //     which is linear in pixels, lands ~2% low. Production passes the manifest entry; a fixture
    //     that omits it is not running the geometry it claims to.
    let request = request(json!({
        "projectId": "p",
        "model": "mochi_1",
        "mode": "text_to_video",
        "prompt": "a calico kitten",
        "duration": 3.8,
        "fps": 30,
        "width": 848,
        "height": 480,
        "modelManifestEntry": { "limits": { "requiresDimensionsMultipleOf": 16 } },
        "advanced": { "mlxQuantize": 8 },
    }));
    // Pin the fixture's own premises — the window is narrow, so a drifted default would silently
    // move this off the seam and the test would prove nothing.
    assert_eq!(request.raw_frame_count(), 114, "3.8 s x 30 fps");
    assert_eq!(
        mochi_frame_count(114),
        115,
        "115 sits in the divergence window"
    );
    assert_eq!((request.width, request.height), (848, 480));
    assert_eq!(
        mochi_tier_order(&request).first().copied(),
        Some("q8"),
        "the pre-check must name the ABSENT q8 tier, or it would measure q4 and refuse early"
    );
    assert!(!root.join("q8").exists(), "q8 must NOT be installed");
    assert_eq!(
        crate::mlx_fit_gate::mochi_resident_bytes(&root.join("q8")),
        MOCHI_Q4_TE_BYTES + MOCHI_Q4_VAE_BYTES,
        "the pre-check's floor must be the shared siblings ONLY — if it folded q4 it would refuse \
             before the fetch and this test would collapse into its sibling"
    );

    let out = drive_mochi_arm(root, "64", &probe, &request);

    let message = out
        .err()
        .expect(
            "the 9.73 GiB shared FLOOR admits this clip (58.31 < 64), but the resolved q4 tier's \
                 full 18.73 GiB does not (67.32 > 64) — the post-download gate must refuse it",
        )
        .to_string();
    assert!(
        message.contains("Shorten the clip"),
        "the refusal must be the fit gate's, i.e. `mochi_preflight` really ran: {message}"
    );
    assert!(
        !probe.loaded(),
        "an over-budget job that reaches the load is the exit(-1) SIGKILL the gate exists to \
             prevent — the pre-check's floor cannot catch this one, only the preflight can"
    );
}

/// Stage a REAL HuggingFace hub cache holding Mochi's shared `coRequisite` components and **no
/// tier dir** — the disk state of a machine that has the ~9.7 GiB shared download but has not
/// fetched a tier. Returns the hub dir to point `HF_HUB_CACHE` at.
///
/// Goes through `safe_repo_dir_name` rather than hand-writing `models--SceneWorks--mochi-1-mlx`, so
/// the fixture cannot drift from the slug the real resolver computes. Sparse, like
/// [`mochi_root_real_sized`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_hf_cache_shared_only(tag: &str) -> tempfile::TempDir {
    let hub_guard = tempfile::Builder::new()
        .prefix(&format!("mochi_hf_hub_{tag}_"))
        .tempdir()
        .expect("temp dir");
    let hub = hub_guard.path();
    let repo_dir = hub.join(format!(
        "models--{}",
        sceneworks_core::hf_home::safe_repo_dir_name(MOCHI_REPO).expect("the mochi repo slug")
    ));
    let snapshot = repo_dir.join("snapshots").join(MOCHI_REVISION);
    for (relative, len) in [
        ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
        ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
    ] {
        let path = snapshot.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path).unwrap().set_len(len).unwrap();
    }
    std::fs::create_dir_all(snapshot.join("tokenizer")).unwrap();
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs").join("main"), MOCHI_REVISION).unwrap();
    hub_guard
}

/// **Closes the residual sc-12322 recorded against this story.** Its own mutation table ends with
/// *"pre-check moved after the fetches ⇒ survives everything. Structural to the untestable async
/// arm; tracked by sc-12318"* — the ORDER of `mochi_precheck` against `ensure_mochi_*_present` is
/// the one decision `generate_mochi` itself owns, and the whole point of sc-12322: refusing after
/// the download is the bug, not the fix.
///
/// How it discriminates. The job asks for **q8**, and the staged cache has the shared components
/// but no q8 tier — so `ensure_mochi_q8_present` is past its `!mochi_wants_q8` early-out and MUST
/// attempt a real `ensure_hf_files_cached` fetch. The two orders then give different errors:
///   * pre-check first (correct) ⇒ the gate's actionable "Shorten the clip", no network touched.
///   * pre-check after the fetches ⇒ a transport error from the download instead ⇒ RED.
///
/// **`huggingface_base_url` is what makes that hermetic, and it is NOT `api_url`.** The fetch dials
/// `settings.huggingface_base_url` (`HuggingFaceSnapshot::resolve` → `{base_url}/api/models/…/tree/…`);
/// `api_url` only carries progress/heartbeat. `Settings::from_env()` defaults the former to the real
/// `https://huggingface.co`, so overriding only `api_url` would send this test's FAILURE path to the
/// live internet: it still goes red (the download's `report_download_progress` hard-`?`s on a
/// heartbeat to the closed `api_url`), but only after really resolving the tree — and the tier is a
/// public ungated repo, so that resolve succeeds and bytes can start landing before the heartbeat
/// trips. Pinning BOTH at a closed port keeps the whole thing offline and instant. Verified by
/// pointing the base at a sentinel host and reading it back out of the mutation's error.
///
/// `HF_HUB_CACHE` is pinned at the fixture because `huggingface_hub_cache_dir` reads it (and
/// `HF_HOME`) BEFORE `data_dir` — left ambient, a dev box's real cache would decide this test.
/// `SCENEWORKS_MLX_MOCHI_DIR` is cleared for the same reason: it would bypass the HF resolution
/// this test is built on.
#[cfg(target_os = "macos")]
#[test]
fn generate_mochi_using_refuses_before_paying_for_the_tier_download() {
    let hub_guard = mochi_hf_cache_shared_only("arm_precheck");
    let hub = hub_guard.path();
    let probe = ArmProbe::default();
    let request = mochi_request(json!({ "mlxQuantize": 8 }));
    // `offline_settings` pins BOTH `api_url` and the `huggingface_base_url` the tier fetch
    // actually dials — see above, and its own doc comment.
    let settings = Settings {
        data_dir: hub.join("unused-data-dir"),
        ..offline_settings()
    };
    let job = mochi_job_snapshot();
    let loader = probe.loader();
    let out = temp_env_vars(
        &[
            ("HF_HUB_CACHE", hub.to_str().expect("utf-8 fixture hub")),
            (MOCHI_DIR_ENV, ""),
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64"),
        ],
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_mochi_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &request,
                    "mochi_1",
                    "mlx",
                    loader,
                ))
        },
    );

    let message = out
        .err()
        .expect("a 151-frame clip cannot fit a 64 GB Mac even on the shared-components floor")
        .to_string();
    assert!(
        message.contains("Shorten the clip"),
        "the refusal must come from the PRE-DOWNLOAD gate, before `ensure_mochi_q8_present` \
             charges ~13-20 GiB for an answer it never needed. A transport error here means the \
             pre-check now runs after the fetches: {message}"
    );
    assert!(!probe.loaded(), "nothing may load on a refused job");
}

/// The candle half of the sc-12318 pin, and the reason the probe harness sits on the superset cfg:
/// `generate_candle_video`'s `video_frame_count(&request.model, request.raw_frame_count())` is the
/// same unpinned caller `generate_mochi` had — swapping the call for `wan_frame_count` hands the
/// engine 149 for the shipped 5 s Mochi default, which `validate_request` hard-rejects
/// (`149 % 6 == 5`).
///
/// Runs on the windows-candle lane (`cargo test -p sceneworks-worker --features backend-candle`),
/// which is what makes this arm genuinely covered rather than asserted — the macOS lane never
/// compiles it.
///
/// 1-byte stub weights are enough: unlike the MLX arm there is no on-disk byte scan to satisfy, and
/// the residency policy reads them as negligible and stays `Resident`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn generate_candle_video_using_hands_the_engine_the_mochi_lattice_frame_count() {
    let root_guard = mochi_root("arm_lattice_candle", &["q4"], true);
    let root = root_guard.path();
    let probe = ArmProbe::default();
    let request = mochi_request(json!({}));
    // This pinned `api_url` alone until sc-12380. It never dialed the hub only because an empty
    // `advanced` makes `mochi_wants_q8`/`_bf16` false, so `ensure_mochi_*_present` early-outs
    // before the fetch — i.e. it was one request-shape change away from the #1577 bug.
    // `offline_settings` pins `huggingface_base_url` too, so it cannot come back.
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..offline_settings()
    };
    let job = mochi_job_snapshot();
    let loader = probe.loader();
    let out = temp_env_var(
        MOCHI_DIR_ENV,
        root.to_str().expect("utf-8 fixture root"),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_candle_video_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &request,
                    std::path::Path::new(""),
                    "candle",
                    loader,
                ))
        },
    );

    let (_decoded, adapter, _raw_settings) = out.expect("the candle mochi arm runs to completion");
    assert_eq!(
        probe.engine_frames(),
        151,
        "the candle ARM must hand the engine the 5 s default snapped onto Mochi's 6k+1 lattice"
    );
    assert_ne!(
        probe.engine_frames(),
        wan_frame_count(150),
        "routing this call through the wan stride hands the engine 149, which it hard-rejects"
    );
    assert_eq!(
        adapter, CANDLE_MOCHI_ADAPTER,
        "a candle-rendered clip records the candle adapter, not the MLX one"
    );
}

/// **THE sc-12373 behavior, and the only test that can prove it.** The pre-download gate's entire
/// value is its ORDER against `ensure_mochi_*_present`: a `mochi_vram_precheck` that runs AFTER the
/// fetches is a no-op the user paid ~13–20 GiB for. Nothing structural pins that — clippy's
/// dead-code pass proves the call EXISTS, never that it runs FIRST — so this drives the real async
/// arm against a stub engine (sc-12318's `_using` seam), which is what makes the ordering assertable
/// at all.
///
/// Kills the mutation the seam tests above cannot: move `mochi_vram_precheck(...)?` below the two
/// `ensure_mochi_*_present` calls ⇒ the fetch dials the unroutable hub and the failure becomes a
/// TRANSPORT error instead of "Shorten the clip" ⇒ RED. It is also the sc-12306 mutation's twin:
/// deleting the pre-check entirely leaves only the post-download gate, and the same assert fires.
///
/// Hermeticity, per #1577's lesson on the MLX twin: the tier fetch dials
/// `settings.huggingface_base_url` (NOT `api_url` — that only carries progress/heartbeat), and
/// `Settings::from_env()` defaults it to the real `https://huggingface.co`. Both are pinned at a
/// closed port so the FAILURE path can never reach the live hub.
///
/// The budget is synthetic and deterministic: `gpu_id` defaults to `"cpu"` ⇒
/// `nvidia_vram_budget_gb` → `None` ⇒ `apply_vram_cap(None, Some(64))` ⇒ `free = total = 64`. So
/// this asserts nothing about the runner's live VRAM. 64 is also the budget at which the
/// shared-components FLOOR is load-bearing at the arm level: decode + headroom alone is 62.56 and
/// would ADMIT.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn generate_candle_video_using_refuses_before_paying_for_the_tier_download() {
    let hub_guard = mochi_hf_cache_shared_only("candle_arm_precheck");
    let hub = hub_guard.path();
    let probe = ArmProbe::default();
    let request = mochi_request(json!({ "mlxQuantize": 8 }));
    // `offline_settings` pins BOTH `api_url` and the `huggingface_base_url` the tier fetch
    // actually dials, so the failure path can never reach the real hub.
    let settings = Settings {
        data_dir: hub.join("unused-data-dir"),
        ..offline_settings()
    };
    let job = mochi_job_snapshot();
    let loader = probe.loader();
    let out = temp_env_vars(
        &[
            ("HF_HUB_CACHE", hub.to_str().expect("utf-8 fixture hub")),
            (MOCHI_DIR_ENV, ""),
            (crate::vram_gate::CUDA_VRAM_CAP_ENV, "64"),
        ],
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_candle_video_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &request,
                    std::path::Path::new(""),
                    "candle",
                    loader,
                ))
        },
    );

    let message = out
        .err()
        .expect("a 151-frame clip cannot fit a 64 GB card even on the shared-components floor")
        .to_string();
    assert!(
        message.contains("Shorten the clip"),
        "the refusal must come from the PRE-DOWNLOAD gate, before `ensure_mochi_q8_present` \
             charges ~13-20 GiB for an answer it never needed. A transport error here means the \
             pre-check now runs after the fetches: {message}"
    );
    assert!(
        message.contains("VRAM"),
        "and it must be the CANDLE gate's message, not the MLX one leaking: {message}"
    );
    assert!(!probe.loaded(), "nothing may load on a refused job");
}

/// The PRE-DOWNLOAD disk state (sc-12322): the shared `coRequisite` components at their REAL
/// hosted sizes, and NO tier dir — exactly what a machine looks like when a job has toggled q8/bf16
/// but `ensure_mochi_*_present` has not run yet. Sparse, like [`mochi_root_real_sized`].
///
/// Superset cfg (sc-12373): both lanes' pre-download gates measure this same disk state.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn mochi_root_shared_only(tag: &str) -> tempfile::TempDir {
    let root_guard = tempfile::Builder::new()
        .prefix(&format!("mochi_precheck_{tag}_"))
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    for (relative, len) in [
        ("text_encoder/model.safetensors", MOCHI_Q4_TE_BYTES),
        ("vae/model.safetensors", MOCHI_Q4_VAE_BYTES),
    ] {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path).unwrap().set_len(len).unwrap();
    }
    std::fs::create_dir_all(root.join("tokenizer")).unwrap();
    root_guard
}

/// THE sc-12322 behavior: the 5 s default is refused on a 64 GB Mac with NO tier on disk — i.e.
/// before the ~13-20 GiB download the old ordering charged for the same answer.
///
/// The fixture's premise — the decision is reached with NO tier dir on disk — is what makes this a
/// PRE-download test rather than a second copy of the `mochi_preflight` pair.
///
/// The three mutations, MEASURED not assumed:
///   * pre-check unconditionally admits ⇒ **RED here** (this test + the floor test). This is the
///     story's required mutation-check.
///   * `mochi_precheck(...)?` deleted from `generate_mochi` ⇒ **RED** at
///     `generate_mochi_using_refuses_before_paying_for_the_tier_download` (sc-12318), which drives
///     the arm itself. Previously this was green-but-for-clippy's dead-code cascade.
///   * pre-check MOVED to after `ensure_mochi_*_present` ⇒ **RED**, same test: the refusal comes
///     back as a transport error from the download instead of the gate's message. This was recorded
///     here as "survives everything — structural to the untestable async arm"; that premise was
///     wrong (the arm IS drivable) and the residual is now closed.
#[cfg(target_os = "macos")]
#[test]
fn mochi_precheck_refuses_the_5s_default_before_any_tier_is_downloaded() {
    let root_guard = mochi_root_shared_only("reject");
    let root = root_guard.path();
    // The requested tier does NOT exist yet — the pre-check must still decide.
    let tier_dir = root.join("bf16");
    assert!(
        !tier_dir.exists(),
        "the fixture must model the pre-download state"
    );
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_precheck("mochi_1", "mochi_1", Some(&tier_dir), 150, 848, 480)
    });

    let message = out
            .expect_err(
                "the shipped 5 s default (151 frames) needs 9.73 GiB of already-present shared \
                 components + 60.56 GiB untiled decode + 2 GiB OS reserve = 72.29 GiB, which does NOT \
                 fit a 64 GB Mac — so the answer is knowable BEFORE the ~20 GiB bf16 download",
            )
            .to_string();
    assert!(message.contains("mochi_1"), "names the model: {message}");
    assert!(
        message.contains("Shorten the clip"),
        "the early refusal must carry the same actionable lever as the full gate: {message}"
    );
}

/// Why the pre-check measures the WEIGHTS FLOOR and not just the decode term (sc-12322's own
/// description proposed decode-only). On a 64 GB Mac the decode + OS reserve for the shipped
/// default is 62.56 GiB — it FITS, by 1.44 GiB. So a weights-free pre-check admits exactly the job
/// in this story's title, downloads ~20 GiB, and only then refuses: the bug, unfixed.
///
/// The already-present shared co-requisites are what push the floor over the line. This test fails
/// the moment someone "simplifies" the pre-check to decode-only.
#[cfg(target_os = "macos")]
#[test]
fn mochi_precheck_needs_the_shared_weights_floor_to_catch_the_64gb_default() {
    // Decode + OS alone does NOT bust a 64 GB budget at the shipped default …
    // (`mochi_decode_peak_gb` moved to the backend-neutral `fit_gate` in sc-12306, when the candle
    // video lane grew the same frame-dependent gate; the arithmetic is unchanged.)
    let decode_only = crate::fit_gate::mochi_decode_peak_gb(151, 848, 480) + 2.0;
    assert!(
        decode_only < 64.0,
        "a weights-free pre-check would ADMIT the titled case ({decode_only:.2} < 64) — which is \
             precisely why the floor is load-bearing"
    );

    // … but the floor the shared components already put on disk does.
    let root_guard = mochi_root_shared_only("floor");
    let root = root_guard.path();
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_precheck(
            "mochi_1",
            "mochi_1",
            Some(&root.join("bf16")),
            150,
            848,
            480,
        )
    });
    assert!(
        out.is_err(),
        "the shared co-requisites (~9.73 GiB) are already on disk before any tier fetch, and \
             {decode_only:.2} + 9.73 > 64 — the pre-check must use them"
    );
}

/// The other direction, and the property that makes an early gate SAFE: a clip that fits is NOT
/// refused early. Without this the pre-check could blanket-reject and still pass the test above —
/// and a pre-check that refuses a fitting config is strictly worse than the late refusal it
/// replaces, because no download can ever redeem it.
#[cfg(target_os = "macos")]
#[test]
fn mochi_precheck_admits_a_short_clip_on_the_same_64gb_mac() {
    let root_guard = mochi_root_shared_only("admit");
    let root = root_guard.path();
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_precheck("mochi_1", "mochi_1", Some(&root.join("bf16")), 19, 848, 480)
    });

    out.expect(
        "19 frames needs 9.73 floor + 9.32 decode + 2 OS = 21.1 GiB — it fits, so the pre-check \
             must let the download proceed",
    );
}

/// Fail-open, twice over: no locatable root (`None`) and a root with nothing measurable on disk
/// both ADMIT. The pre-check never blocks without evidence — same rule as the gate it calls
/// (`weight_bytes == 0` ⇒ no signal). A first-ever Mochi job must reach the download, not be
/// refused for not having downloaded yet.
#[cfg(target_os = "macos")]
#[test]
fn mochi_precheck_admits_without_a_signal() {
    // No root at all ⇒ nothing to measure.
    temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_precheck("mochi_1", "mochi_1", None, 150, 848, 480)
    })
    .expect("no locatable model root ⇒ no signal ⇒ admit, never a phantom refusal");

    // An empty root ⇒ the scan sums 0 bytes ⇒ still no signal, even though the clip is huge.
    let root_guard = tempfile::Builder::new()
        .prefix("mochi_precheck_empty_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    std::fs::create_dir_all(root.join("bf16")).unwrap();
    let out = temp_env_var(crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, "64", || {
        mochi_precheck(
            "mochi_1",
            "mochi_1",
            Some(&root.join("bf16")),
            150,
            848,
            480,
        )
    });
    out.expect("unmeasurable weights ⇒ no signal ⇒ admit and let the full gate decide later");
}

/// The pre-check must measure the tier the job ASKED for, not the smaller one already installed.
/// `resolve_mochi_model_dir` deliberately falls back to a smaller complete tier — reusing it here
/// would measure q4's bytes for a bf16 job and, worse, hard-error when nothing is installed at all.
#[cfg(target_os = "macos")]
#[test]
fn mochi_precheck_dir_names_the_requested_tier_not_the_installed_fallback() {
    let root_guard = mochi_root("precheck_dir", &["q4"], true);
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..Settings::from_env()
    };
    let dir = temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
        // bf16 requested; only q4 is on disk.
        mochi_precheck_dir(&settings, &mochi_request(json!({ "mlxQuantize": 16 })))
    });

    assert_eq!(
        dir.as_deref(),
        Some(root.join("bf16").as_path()),
        "the requested tier's path, existence NOT required — measuring the q4 fallback would \
             under-count a bf16 job's weights against a budget it must clear"
    );
}

/// Mochi is text-to-video ONLY (`conditioning: []` on both descriptors). `VideoRequest`'s mode
/// DEFAULTS to `image_to_video` when the payload omits one, so the mode gate is what keeps a
/// conditioning-shaped job out of the Mochi route — without it an unmoded payload would route to
/// Mochi and fail deep in the engine.
#[cfg(target_os = "macos")]
#[test]
fn mochi_available_is_text_to_video_only() {
    let root_guard = mochi_root("mode", &["q4"], true);
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..Settings::from_env()
    };
    temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
        let t2v = request(json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        }));
        assert!(
            mochi_available(&t2v, &settings),
            "a t2v mochi job with resolvable weights must route to the Mochi engine"
        );
        // Every non-t2v mode — including the DEFAULT the DTO applies when `mode` is absent.
        for mode in [
            "image_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
            "animate_character",
        ] {
            let req = request(json!({
                "projectId": "p", "model": "mochi_1", "mode": mode, "prompt": "p"
            }));
            assert!(
                !mochi_available(&req, &settings),
                "mochi has no {mode} surface (conditioning: [])"
            );
        }
        let unmoded = request(json!({ "projectId": "p", "model": "mochi_1", "prompt": "p" }));
        assert_eq!(unmoded.mode, "image_to_video", "the DTO's default mode");
        assert!(
            !mochi_available(&unmoded, &settings),
            "a payload with no mode defaults to i2v and must NOT reach the t2v-only engine"
        );
    });
}

#[test]
fn mochi_mode_gate_rejects_every_conditioning_shape() {
    let mut request = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
    }));
    assert!(mochi::validate_mochi_mode(&request).is_ok());

    for mode in [
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
    ] {
        request.mode = mode.to_owned();
        let error = mochi::validate_mochi_mode(&request).expect_err("conditioning mode must fail");
        assert!(
            matches!(error, WorkerError::InvalidPayload(ref message)
                if message.contains("text_to_video only") && message.contains(mode)),
            "unexpected error for {mode}: {error:?}"
        );
    }
}

/// A Mochi t2v job with resolvable weights routes to `VideoRoute::Mochi` — and, critically, a
/// Mochi job whose weights DON'T resolve must NOT silently become `VideoRoute::Stub` output:
/// `ensure_video_engine_weights` (the sc-4176 fail-loud gate the Stub arm calls) must error.
#[cfg(target_os = "macos")]
#[test]
fn mochi_routes_to_the_mochi_engine_and_never_degrades_to_a_fake_video() {
    let root_guard = mochi_root("route", &["q4"], true);
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..Settings::from_env()
    };
    let req = request(json!({
        "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
    }));
    temp_env_var(MOCHI_DIR_ENV, root.to_str().unwrap(), || {
        assert_eq!(
            resolve_video_route(&req, &settings),
            VideoRoute::Mochi("mochi_1")
        );
        // Adding Mochi must not perturb any pre-existing route.
        let wan = request(json!({
            "projectId": "p", "model": "wan_2_2", "mode": "text_to_video", "prompt": "p"
        }));
        assert!(!matches!(
            resolve_video_route(&wan, &settings),
            VideoRoute::Mochi(_)
        ));
    });

    // Weights absent (no override, empty data dir) ⇒ the route falls to Stub, BUT the Stub arm's
    // fail-loud gate must refuse rather than hand back a procedural fake video.
    let empty_guard = tempfile::Builder::new()
        .prefix("mochi_empty_")
        .tempdir()
        .expect("temp dir");
    let empty = empty_guard.path();
    let bare = Settings {
        data_dir: empty.to_path_buf(),
        ..Settings::from_env()
    };
    temp_env_var(MOCHI_DIR_ENV, "", || {
        assert_eq!(resolve_video_route(&req, &bare), VideoRoute::Stub);
        let err = ensure_video_engine_weights(&req, &bare)
            .expect_err("an unprovisioned mochi MUST fail loudly, never render a fake video");
        let WorkerError::InvalidPayload(message) = err else {
            panic!("expected an actionable InvalidPayload");
        };
        assert!(
            message.contains("mochi_1") && message.contains("Model Manager"),
            "the error must name the model and tell the user what to do: {message}"
        );
    });
}

// ---------------------------------------------------------------------------
// Krea Realtime 14B (epic 8431 / sc-8443 S10) — routing + VideoRequest→GenerationRequest mapping.
// ---------------------------------------------------------------------------

/// A temp dir holding just a `config.json`, so `local_mlx_dir` accepts it as a resolvable Krea
/// Realtime snapshot (the loader is stubbed in the mapping test, so the contents are never read).
///
/// Returns the guard rather than a bare path: the `temp_dir()/krea_{tag}_{pid}` this replaces was
/// never removed on ANY path — not even a passing one — and keyed uniqueness on the process, so a
/// run landing on a recycled PID resolved against an unrelated earlier run's snapshot (sc-17707).
/// Callers must hold the guard for as long as the resolver reads the directory.
#[cfg(target_os = "macos")]
fn krea_fake_model_dir(tag: &str) -> tempfile::TempDir {
    let dir_guard = tempfile::Builder::new()
        .prefix(&format!("krea_{tag}_"))
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    // A FLAT locally-converted snapshot: `config.json` is what `local_mlx_dir` counts, and
    // `dit.safetensors` (+ the stock Wan companions) is what makes it actually loadable — the
    // `krea_realtime_dir_has_transformer` probe the resolver runs, mirroring the engine's own
    // `t2v::open_transformer_weights`. A `config.json`-only dir would resolve as a model dir and then
    // die inside the loader, so the fixture must carry weights (sc-15258).
    for file in [
        "config.json",
        "dit.safetensors",
        "t5_encoder.safetensors",
        "tokenizer.json",
        "vae.safetensors",
    ] {
        std::fs::write(dir.join(file), "{}").unwrap();
    }
    dir_guard
}

/// A `JobSnapshot` for a Krea Realtime video job.
#[cfg(target_os = "macos")]
fn krea_job_snapshot() -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job-krea-1",
        "type": "video_generate",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": { "model": "krea_realtime_14b" },
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.0,
        "stage": "queued",
        "message": "queued",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-07-26T00:00:00Z",
        "updatedAt": "2026-07-26T00:00:00Z",
        "startedAt": null,
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": null
    }))
    .expect("the krea job snapshot deserializes")
}

/// The SceneWorks id maps to the engine id 1:1 and matches nothing else — so appending the krea arm to
/// the route ladder can't shadow (or be shadowed by) any pre-existing engine.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_engine_id_maps_only_krea() {
    assert_eq!(
        krea_realtime_engine_id("krea_realtime_14b"),
        Some("krea_realtime_14b")
    );
    for other in [
        "wan_2_2",
        "wan_2_2_t2v_14b",
        "scail2_14b",
        "bernini",
        "mochi_1",
        "svd",
        "ltx_2_3",
        "krea_2_turbo", // the unrelated IMAGE crate — must NOT collide with the video engine
    ] {
        assert_eq!(
            krea_realtime_engine_id(other),
            None,
            "{other} must not resolve to the Krea Realtime engine"
        );
    }
}

/// The invariant this story exists to wire, tested on the pure shape helper without a GPU or assets:
/// a reference still → i2v `Reference` with `strength` DROPPED (`None`); a source clip → v2v
/// `VideoClip` with `strength` HONORED; a clip WINS over a still (v2v precedence, matching the engine's
/// `run`); no media → t2v (empty). Discriminating: the `VideoClip` strength must equal the value passed
/// (not a hardcoded default), and the `Reference` strength must be `None` — a future edit that started
/// honoring i2v strength would turn this red.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_conditioning_i2v_drops_strength_v2v_honors_it() {
    let still = || gen_core::Image {
        width: 2,
        height: 2,
        pixels: vec![0u8; 12],
    };
    let clip = || {
        vec![gen_core::Image {
            width: 2,
            height: 2,
            pixels: vec![0u8; 12],
        }]
    };

    // i2v: a still, strength argument is a no-op → Reference { strength: None }.
    let i2v = krea_realtime_conditioning(Some(still()), None, 0.9);
    assert_eq!(i2v.len(), 1);
    match &i2v[0] {
        Conditioning::Reference { strength, .. } => assert_eq!(
            *strength, None,
            "i2v strength is a NO-OP — the reference must carry no strength"
        ),
        other => panic!("i2v must be a Reference, got {other:?}"),
    }

    // v2v: a clip → VideoClip whose strength equals the honored value (0.35, not a default).
    let v2v = krea_realtime_conditioning(None, Some(clip()), 0.35);
    assert_eq!(v2v.len(), 1);
    match &v2v[0] {
        Conditioning::VideoClip {
            frames, strength, ..
        } => {
            assert_eq!(frames.len(), 1);
            assert_eq!(*strength, 0.35, "v2v must HONOR the resolved strength");
        }
        other => panic!("v2v must be a VideoClip, got {other:?}"),
    }

    // A clip and a still together → v2v wins (the engine prefers VideoClip).
    let both = krea_realtime_conditioning(Some(still()), Some(clip()), 0.5);
    assert_eq!(both.len(), 1);
    assert!(
        matches!(both[0], Conditioning::VideoClip { .. }),
        "a clip must take precedence over a still (v2v)"
    );

    // No media → t2v (no conditioning).
    assert!(krea_realtime_conditioning(None, None, 1.0).is_empty());
}

/// A Krea Realtime job with resolvable weights routes to `VideoRoute::KreaRealtime` — and, critically,
/// one whose weights DON'T resolve must NOT silently become `VideoRoute::Stub` output: the Stub arm's
/// fail-loud gate (`ensure_video_engine_weights`) must error. Also pins that adding the krea arm does
/// not pull any other model onto it.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_routes_to_the_krea_engine_and_never_degrades_to_a_fake_video() {
    let model_dir_guard = krea_fake_model_dir("route");
    let model_dir = model_dir_guard.path();
    // Never created and never cleaned before: a bare `temp_dir()/krea_route_data_{pid}` path
    // handed straight to `Settings` (sc-17707).
    let data_dir_guard = tempfile::Builder::new()
        .prefix("krea_route_data_")
        .tempdir()
        .expect("temp dir");
    let settings = Settings {
        data_dir: data_dir_guard.path().to_path_buf(),
        ..Settings::from_env()
    };
    let req = request(json!({
        "projectId": "p", "model": "krea_realtime_14b", "mode": "text_to_video", "prompt": "p"
    }));
    temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        model_dir.to_str().unwrap(),
        || {
            assert_eq!(
                resolve_video_route(&req, &settings),
                VideoRoute::KreaRealtime("krea_realtime_14b")
            );
            // A non-krea model never lands on the krea route (even though krea weights ARE available):
            // routing is keyed on the model id, not merely availability.
            for other in ["wan_2_2", "scail2_14b", "bernini", "mochi_1"] {
                let r = request(json!({
                    "projectId": "p", "model": other, "mode": "text_to_video", "prompt": "p"
                }));
                assert!(
                    !matches!(
                        resolve_video_route(&r, &settings),
                        VideoRoute::KreaRealtime(_)
                    ),
                    "{other} must not route to KreaRealtime"
                );
            }
        },
    );

    // Weights absent (empty env override, empty data dir) ⇒ the route falls to Stub, BUT the Stub
    // arm's fail-loud gate must refuse rather than hand back a procedural fake video.
    let empty_guard = tempfile::Builder::new()
        .prefix("krea_empty_")
        .tempdir()
        .expect("temp dir");
    let empty = empty_guard.path();
    let bare = Settings {
        data_dir: empty.to_path_buf(),
        ..Settings::from_env()
    };
    temp_env_var("SCENEWORKS_MLX_KREA_REALTIME_DIR", "", || {
        assert_eq!(resolve_video_route(&req, &bare), VideoRoute::Stub);
        let err = ensure_video_engine_weights(&req, &bare)
            .expect_err("an unprovisioned krea MUST fail loudly, never render a fake video");
        let WorkerError::InvalidPayload(message) = err else {
            panic!("expected an actionable InvalidPayload");
        };
        assert!(
            message.contains("krea_realtime_14b") && message.contains("Model Manager"),
            "the error must name the model and tell the user what to do: {message}"
        );
    });
}

/// The caller-side mapping pin: drives `generate_krea_realtime_using` end to end (through the
/// `generate_video` heartbeat funnel) against a stub generator and asserts on the `GenerationRequest`
/// that REACHED the engine — prompt, `steps` (from advanced), seed, fps, and the Wan-lattice frame
/// count — for a t2v job (no conditioning). CFG off: no negative prompt reaches the engine. This is
/// what proves the VideoRequest→GenerationRequest scalar mapping AND that the generate path
/// structurally runs through the funnel (a clip comes back).
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_using_hands_the_engine_the_mapped_t2v_request() {
    let model_dir_guard = krea_fake_model_dir("map");
    let model_dir = model_dir_guard.path();
    let settings = Settings {
        data_dir: model_dir.join("unused-data-dir"),
        ..offline_settings()
    };
    let req = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "aurora over a fjord",
        "seed": 4242,
        "fps": 16,
        "duration": 2.0,
        "advanced": { "steps": 6 }
    }));
    // raw = round(2.0 * 16) = 32; the Wan 1-mod-4 stride floors 32 → 29 (the DiT + z16 VAE ARE stock Wan).
    assert_eq!(wan_frame_count(req.raw_frame_count()), 29);

    let job = krea_job_snapshot();
    let probe = ArmProbe::default();
    let loader = probe.loader();
    let decoded = temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        model_dir.to_str().unwrap(),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &req,
                    // project_path — never touched by a t2v request (no media loaded).
                    model_dir,
                    "krea_realtime_14b",
                    "mlx",
                    loader,
                ))
        },
    )
    .expect("krea t2v generation runs through the funnel against the probe");

    // The clip came back through the funnel — the generate path structurally ran the heartbeat loop.
    let (decoded, raw_settings) = decoded;
    assert!(!decoded.frames.is_empty());
    // The fixture is a flat snapshot with no explicit `mlxQuantize`, so the dense-video Q4 default
    // is what loads — and the arm now stamps it on the `rawSettings` it returns rather than handing
    // back a bare tier string. Reading the field through `as_str` targets the stamp itself, which is
    // what the caller actually records on the asset.
    assert_eq!(
        raw_settings.get("kreaRealtimeTier").and_then(Value::as_str),
        Some("q4"),
    );

    let seen = probe.request.lock().unwrap();
    let gen = seen.as_ref().expect("the arm reached the engine");
    assert_eq!(gen.prompt, "aurora over a fjord");
    assert_eq!(gen.steps, Some(6), "advanced steps map to the engine steps");
    assert_eq!(gen.seed, Some(4242), "the explicit seed maps through");
    assert_eq!(gen.fps, Some(16));
    assert_eq!(gen.frames, Some(29), "frame count uses the Wan lattice");
    assert_eq!(
        gen.negative_prompt, None,
        "CFG off — no negative prompt reaches the engine"
    );
    assert!(
        gen.conditioning.is_empty(),
        "a t2v request carries no conditioning"
    );

    drop(seen);
}

// ---------------------------------------------------------------------------
// Krea Realtime Wan-family LoRA passthrough (sc-15017, S15).
// ---------------------------------------------------------------------------

/// 🔴 The story: a user-selected Wan-family LoRA must actually REACH the engine on
/// `LoadSpec::adapters`. Before this the krea arm hard-coded an empty `adapters` (the S15 seam), so
/// a selected LoRA was silently ignored and the base DiT rendered.
///
/// Discriminating on purpose: the asserted scale is the request's **0.35**, not `lora_scale`'s 0.8
/// default, and the second file is tagged `Lokr` from its own metadata — so an arm that dropped the
/// specs, or one that synthesized a default spec, fails.
///
/// Mutation check (verified RED): restore `..VideoGenInput::default()`'s empty `adapters` in
/// `generate_krea_realtime_using`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_passes_selected_loras_to_the_engine_load_spec() {
    let root_guard = krea_fake_model_dir("lora");
    let root = root_guard.path();
    let style = root.join("origami_style.safetensors");
    let character = root.join("hero.safetensors");
    write_lora_fixture(&style, None);
    write_lora_fixture(&character, Some("lokr"));
    // sc-5723: LoRA paths are confined to the app data dir, so point `data_dir` at the fixture root.
    let settings = Settings {
        data_dir: root.to_path_buf(),
        ..offline_settings()
    };
    let req = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "a red fox trotting through snowy pines",
        "duration": 2.0,
        "fps": 24,
        "loras": [
            { "path": style.to_string_lossy(), "weight": 0.35 },
            { "path": character.to_string_lossy(), "weight": 0.9 },
        ],
    }));

    let probe = ArmProbe::default();
    let loader = probe.loader();
    let job = krea_job_snapshot();
    temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &req,
                    root,
                    "krea_realtime_14b",
                    "mlx",
                    loader,
                ))
                .expect("krea generation with LoRAs runs through the funnel");
        },
    );

    let seen = probe.spec.lock().unwrap();
    let spec = seen.as_ref().expect("the arm loaded an engine");
    assert_eq!(
        spec.adapters.len(),
        2,
        "both selected LoRAs must reach the engine's LoadSpec"
    );
    assert_eq!(spec.adapters[0].path, style.canonicalize().unwrap());
    assert!(
        (spec.adapters[0].scale - 0.35).abs() < 1e-6,
        "the request's own weight must ride through, not the 0.8 default: {}",
        spec.adapters[0].scale
    );
    assert_eq!(spec.adapters[0].kind, AdapterKind::Lora);
    assert_eq!(spec.adapters[1].kind, AdapterKind::Lokr);
    assert!(
        spec.adapters
            .iter()
            .all(|adapter| adapter.moe_expert.is_none()),
        "Krea Realtime is a single dense transformer — no MoE expert tagging"
    );

    drop(seen);
}

/// The per-job LoRA cap is enforced on the krea arm too, and it is enforced BEFORE the tier fetch —
/// so an over-cap job fails immediately instead of after pulling a multi-GB tier. The probe proves
/// the second half: nothing was loaded.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_rejects_more_loras_than_the_cap_before_loading_anything() {
    let root_guard = krea_fake_model_dir("loracap");
    let root = root_guard.path();
    let style = root.join("style.safetensors");
    write_lora_fixture(&style, None);
    let settings = Settings {
        data_dir: root.to_path_buf(),
        ..offline_settings()
    };
    let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
        .map(|_| json!({ "path": style.to_string_lossy() }))
        .collect();
    let over = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
        "duration": 2.0,
        "fps": 24,
        "loras": many,
    }));
    // The same request one LoRA lighter resolves — so the rejection is the CAP, not the fixture.
    let under = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
        "loras": (0..MAX_JOB_LORAS)
            .map(|_| json!({ "path": style.to_string_lossy() }))
            .collect::<Vec<Value>>(),
    }));
    assert_eq!(
        resolve_krea_realtime_adapters(&settings, &under)
            .expect("at the cap resolves")
            .len(),
        MAX_JOB_LORAS
    );

    let probe = ArmProbe::default();
    let loader = probe.loader();
    let job = krea_job_snapshot();
    let result = temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &over,
                    root,
                    "krea_realtime_14b",
                    "mlx",
                    loader,
                ))
        },
    );
    assert!(matches!(result, Err(WorkerError::InvalidPayload(_))));
    assert!(
        probe.spec.lock().unwrap().is_none(),
        "an over-cap job must be refused before any engine load"
    );
}

/// Coverage comes only from the engine's actual report. A fully-applied lightx2v-shaped file must not
/// inherit the old header prediction, while a genuinely out-of-surface target produces an accurate
/// 1-of-2 stamp.
///
/// Mutation checks (verified RED): substitute stale header-derived counts; sever the
/// `Generator::adapter_apply_reports` read in the decoded-video funnel.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_coverage_uses_engine_reports_not_header_predictions() {
    #[derive(Clone)]
    struct LogWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogWriter {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    let reports = vec![
        gen_core::AdapterApplyReport {
            adapter_path: PathBuf::from("/models/lightx2v_step_distill.safetensors"),
            applied: 647,
            skipped: Vec::new(),
        },
        gen_core::AdapterApplyReport {
            adapter_path: PathBuf::from("/models/i2v_foreign_target.safetensors"),
            applied: 1,
            skipped: vec!["blocks.0.cross_attn.norm_k_img".to_owned()],
        },
    ];
    let clean_coverage = krea_realtime_lora_coverage(&reports[..1]);
    assert!(
        clean_coverage.partial.is_empty(),
        "all 647 real lightx2v diff-patch targets landed, so no partial stamp is truthful"
    );
    let coverage = krea_realtime_lora_coverage(&reports);
    assert_eq!(
        coverage.partial,
        vec![("i2v_foreign_target.safetensors".to_owned(), 1, 2)],
        "only the engine-reported skip is partial, with applied+skipped as the exact target total"
    );
    let req = request(json!({ "projectId": "p", "model": "krea_realtime_14b" }));
    let raw = krea_realtime_raw_settings(&req, "q4", &coverage);
    let stamped = raw
        .get("kreaRealtimeLoraPartial")
        .and_then(Value::as_array)
        .expect("the partial application is stamped on the asset");
    assert_eq!(stamped.len(), 1);
    assert_eq!(
        stamped[0].get("file").and_then(Value::as_str),
        Some("i2v_foreign_target.safetensors")
    );
    assert_eq!(
        stamped[0].get("unappliedKeys").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(stamped[0].get("totalKeys").and_then(Value::as_u64), Some(2));

    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(LogWriter(std::sync::Arc::clone(&captured)))
        .finish();
    tracing::subscriber::with_default(subscriber, || coverage.warn());
    let log = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
    assert!(
        log.contains("\"event\":\"krea_realtime_lora_partially_applied\"")
            && log.contains("\"unapplied_keys\":1")
            && log.contains("\"total_keys\":2")
            && log.contains("i2v_foreign_target.safetensors"),
        "the warn event carries the same accurate engine counts as the stamp: {log}"
    );
}

/// The Krea arm must carry the provider report through the shared generation funnel before building
/// raw settings. This is the non-circular plumbing witness: the injected generator owns the report;
/// neither the request nor any adapter header contains these counts.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_arm_stamps_the_report_returned_by_the_loaded_generator() {
    let root_guard = krea_fake_model_dir("reported_coverage");
    let root = root_guard.path();
    let settings = Settings {
        data_dir: root.to_path_buf(),
        ..offline_settings()
    };
    let req = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
    }));
    let probe = ArmProbe::default();
    *probe.adapter_reports.lock().unwrap() = vec![gen_core::AdapterApplyReport {
        adapter_path: PathBuf::from("/engine/resolved/out_of_surface.safetensors"),
        applied: 1,
        skipped: vec!["blocks.0.cross_attn.norm_k_img".to_owned()],
    }];
    let result = temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &krea_job_snapshot(),
                    &req,
                    root,
                    "krea_realtime_14b",
                    "mlx",
                    probe.loader(),
                ))
        },
    )
    .expect("probe generation succeeds");
    let stamp = result
        .1
        .get("kreaRealtimeLoraPartial")
        .and_then(Value::as_array)
        .expect("the provider report reaches rawSettings");
    assert_eq!(
        stamp,
        &[json!({
            "file": "out_of_surface.safetensors",
            "unappliedKeys": 1,
            "totalKeys": 2,
        })]
    );
}

/// 🔴 The hard-error class a real user hits: an I2V-family Wan LoRA targets `cross_attn.k_img` /
/// `v_img`, which a T2V backbone does not have at any surface width. The engine is right to refuse —
/// but the raw text is an engine-internal module list behind a generic "video load failed". The
/// annotation must name the LoRA, say WHY, and say what to do, while keeping the engine detail.
///
/// Discriminating: the generic (non-I2V) arm produces a DIFFERENT hint, a non-adapter `Msg` is
/// returned byte-identical, and the typed `Canceled` — which the funnel matches on to report a clean
/// user cancellation — must never be laundered into a `Msg`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_lora_load_errors_are_rewritten_into_actionable_guidance() {
    let loras = vec!["Squish_I2V.safetensors".to_owned()];
    let i2v = gen_core::Error::Msg(
        "krea_realtime_14b adapters: 80 adapter target(s) matched no module (surfaced, not \
         silently dropped): [\"blocks.0.cross_attn.k_img\", \"blocks.0.cross_attn.v_img\"]"
            .to_owned(),
    );
    let annotated = annotate_krea_realtime_lora_error(i2v, &loras).to_string();
    assert!(
        annotated.contains("Squish_I2V.safetensors"),
        "the message must name the LoRA the user picked: {annotated}"
    );
    assert!(
        annotated.contains("image-to-video") && annotated.contains("cross_attn.k_img"),
        "the I2V class must be diagnosed, not left as a module dump: {annotated}"
    );
    assert!(
        annotated.contains("blocks.0.cross_attn.v_img"),
        "the engine detail is kept, not swallowed: {annotated}"
    );

    // A non-I2V adapter failure gets the generic Wan-family hint — so the I2V wording above is a
    // real branch, not text that always renders.
    let other = gen_core::Error::Msg(
        "krea_realtime_14b adapters: no target modules matched across 1 adapter file(s)".to_owned(),
    );
    let generic = annotate_krea_realtime_lora_error(other, &loras).to_string();
    assert!(
        !generic.contains("image-to-video"),
        "a non-I2V failure must not claim an I2V cause: {generic}"
    );
    assert!(
        generic.contains("Wan-family") && generic.contains("Squish_I2V.safetensors"),
        "the generic branch still names the LoRA and the family that works: {generic}"
    );

    // Not an adapter failure → untouched.
    let unrelated = gen_core::Error::Msg("missing dit.safetensors".to_owned());
    assert_eq!(
        annotate_krea_realtime_lora_error(unrelated, &loras).to_string(),
        "missing dit.safetensors"
    );
    // No LoRA selected → untouched even if the text mentions adapters.
    let no_loras = gen_core::Error::Msg("krea_realtime_14b adapters: something".to_owned());
    assert_eq!(
        annotate_krea_realtime_lora_error(no_loras, &[]).to_string(),
        "krea_realtime_14b adapters: something"
    );
    // 🔴 The typed cancellation must survive: the funnel matches on it to report a clean cancel.
    assert!(matches!(
        annotate_krea_realtime_lora_error(gen_core::Error::Canceled, &loras),
        gen_core::Error::Canceled
    ));
}

// ---------------------------------------------------------------------------
// Krea Realtime quant matrix (sc-15258, S20): per-tier resolution + the one-decision quant.
// ---------------------------------------------------------------------------

/// A Krea Realtime `VideoRequest` carrying `advanced` — the tier selector's only input.
#[cfg(target_os = "macos")]
fn krea_tier_request(advanced: Value) -> VideoRequest {
    request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
        "advanced": advanced,
    }))
}

/// The directory a `LoadSpec`'s weights point at. `WeightsSource` has no `PartialEq`, so the model
/// dir that reached the engine is compared through this rather than `assert_eq!` on the enum.
#[cfg(target_os = "macos")]
fn spec_weights_dir(spec: &LoadSpec) -> PathBuf {
    match &spec.weights {
        WeightsSource::Dir(dir) => dir.clone(),
        WeightsSource::File(file) => panic!("a video load is a Dir source, got the file {file:?}"),
    }
}

/// Write a COMPLETE Krea Realtime tier subdir under `root` — every file in that tier's own
/// `KREA_REALTIME_TIER_FILES` row, including the `bf16/` tier's NESTED seven-shard `transformer/`.
#[cfg(target_os = "macos")]
fn write_complete_krea_tier(root: &Path, tier: &str) {
    let files = krea_realtime_tier_files(tier).expect("a published krea tier");
    for file in files {
        let path = root.join(tier).join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }
}

/// The worker's hard-coded tier table IS the shipped manifest — same repo, same pinned revision, same
/// exact file paths per tier. The worker resolves a tier by checking those files on disk while the
/// Model Manager downloads whatever `builtin.models.jsonc` declares, so any drift between the two
/// silently produces a tier that installs but never resolves (or resolves while torn).
///
/// This is what closes the loop the sc-8444 layout test opened: that one pins the manifest against the
/// PUBLISHED repo; this one pins the WORKER against the manifest.
///
/// Mutation checks (each verified RED): rename any `bf16/transformer/dit-…` shard in
/// `KREA_REALTIME_BF16_TIER_FILES`; drop `config.json` from the packed row; change one hex digit of
/// `KREA_REALTIME_REVISION`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_worker_tier_table_matches_the_shipped_manifest() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"].as_str() == Some("krea_realtime_14b"))
        .expect("krea_realtime_14b present in the builtin catalog");

    let downloads = model["downloads"]
        .as_array()
        .expect("krea_realtime_14b declares downloads");
    let mut seen: Vec<&str> = Vec::new();
    for entry in downloads {
        let variant = entry["variant"]
            .as_str()
            .expect("every krea download is a tier variant");
        seen.push(variant);
        assert_eq!(
            entry["repo"].as_str(),
            Some(KREA_REALTIME_REPO),
            "{variant}: the worker fetches from KREA_REALTIME_REPO, so the manifest must install from it"
        );
        assert_eq!(
            entry["revision"].as_str(),
            Some(KREA_REALTIME_REVISION),
            "{variant}: a worker pin that differs from the manifest's would materialize a SECOND \
             snapshot dir beside the installed one"
        );
        let declared: Vec<String> = entry["files"]
            .as_array()
            .expect("explicit file paths")
            .iter()
            .map(|file| file.as_str().expect("a file path").to_owned())
            .collect();
        let expected: Vec<String> = krea_realtime_tier_files(variant)
            .unwrap_or_else(|| panic!("{variant} is declared in the manifest but not in the worker's KREA_REALTIME_TIER_FILES"))
            .iter()
            .map(|file| format!("{variant}/{file}"))
            .collect();
        let (mut declared, mut expected) = (declared, expected);
        declared.sort();
        expected.sort();
        assert_eq!(
            declared, expected,
            "{variant}: the worker's completeness predicate must check exactly the files the \
             manifest installs"
        );
    }
    seen.sort();
    let mut tiers: Vec<&str> = KREA_REALTIME_TIER_FILES
        .iter()
        .map(|(tier, _)| *tier)
        .collect();
    tiers.sort();
    assert_eq!(
        seen, tiers,
        "the worker must know every published tier and no others"
    );

    // The bf16 row is the structurally different one — assert its shape directly so a future edit
    // that "simplifies" it to the flat packed row cannot pass by accident.
    let bf16 = krea_realtime_tier_files("bf16").expect("bf16 is published");
    assert_eq!(
        bf16.iter()
            .filter(|file| file.starts_with("transformer/"))
            .count(),
        7,
        "the bf16 DiT is a 7-shard transformer/ directory, not a flat dit.safetensors"
    );
    assert!(
        !bf16.contains(&"dit.safetensors"),
        "the bf16 tier has no flat DiT"
    );
    assert!(
        krea_realtime_tier_files("q4")
            .expect("q4 is published")
            .contains(&"dit.safetensors"),
        "the packed tiers DO ship a flat single-file DiT"
    );
    assert_eq!(
        krea_realtime_tier_files("q6"),
        None,
        "an unpublished tier must have NO file list — an empty one would read COMPLETE from an \
         empty directory"
    );
}

/// The tier search order comes from the SAME `resolve_mlx_dense_quant` decision that would pick a
/// load-time quant, and — the sc-15258 trap — `bf16` is in EVERY order rather than only the explicit
/// one. The Krea rehost has no flat root to fall back on (unlike SCAIL-2), so dropping `bf16` from the
/// default order is exactly how a deliberately-installed bf16 tier becomes unreachable and gets
/// silently re-quantized to the Q4 default.
///
/// Mutation checks (each verified RED): drop `"bf16"` from the default arm; flip the default arm to
/// `bf16`-first; make the `Some(Q8)` arm fall back to `q8` only.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_tier_order_tracks_the_quant_and_never_drops_bf16() {
    let default = krea_tier_request(json!({}));
    let q4 = krea_tier_request(json!({ "mlxQuantize": 4 }));
    let q8 = krea_tier_request(json!({ "mlxQuantize": 8 }));
    let bf16 = krea_tier_request(json!({ "mlxQuantize": 0 }));

    // The dense-video Q4 default (sc-10750) reaching the VIDEO lane: no explicit pick ⇒ q4-first.
    assert_eq!(krea_realtime_tier_order(&default), &["q4", "q8", "bf16"]);
    assert_eq!(krea_realtime_tier_order(&q4), &["q4", "q8", "bf16"]);
    assert_eq!(krea_realtime_tier_order(&q8), &["q8", "q4", "bf16"]);
    assert_eq!(krea_realtime_tier_order(&bf16), &["bf16", "q8", "q4"]);

    // One decision: the order's head is the tier the shared resolver would have quantized to.
    for (req, head) in [(&default, "q4"), (&q4, "q4"), (&q8, "q8"), (&bf16, "bf16")] {
        assert_eq!(krea_realtime_tier_order(req)[0], head);
    }
    assert_eq!(resolve_mlx_dense_quant(&default), Some(Quant::Q4));
    assert_eq!(resolve_mlx_dense_quant(&q8), Some(Quant::Q8));
    assert_eq!(resolve_mlx_dense_quant(&bf16), None);

    // EVERY order can still reach EVERY published tier — the no-silent-switch precondition.
    for req in [&default, &q4, &q8, &bf16] {
        let order = krea_realtime_tier_order(req);
        for (tier, _) in KREA_REALTIME_TIER_FILES {
            assert!(
                order.contains(tier),
                "{tier} must be reachable from every tier order — the Krea rehost has no flat root \
                 fallback, so an omitted tier is an unloadable install"
            );
        }
    }
}

/// Each tier resolves to its OWN subdir — including the bf16 tier, whose DiT lives one level deeper in
/// a seven-shard `transformer/` dir — and a torn tier is skipped rather than half-loaded.
///
/// Mutation checks (each verified RED): resolve `root` instead of `root.join(tier)`; make
/// `krea_realtime_tier_is_complete` return `true` for an unknown tier; drop the per-tier file lookup so
/// bf16 is checked against the packed row (a bf16 dir would then read INCOMPLETE).
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_tier_subdir_resolves_each_tier_including_the_sharded_bf16() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_krea_tier_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let default = krea_tier_request(json!({}));
    let q8_req = krea_tier_request(json!({ "mlxQuantize": 8 }));
    let bf16_req = krea_tier_request(json!({ "mlxQuantize": 0 }));

    // A bare root (the published repo before any tier is installed) resolves nothing.
    assert_eq!(krea_realtime_tier_subdir(root, &default), None);

    // q4 only: every request falls back to it (there is nothing else on disk).
    write_complete_krea_tier(root, "q4");
    assert_eq!(
        krea_realtime_tier_subdir(root, &default),
        Some(("q4", root.join("q4")))
    );
    assert_eq!(
        krea_realtime_tier_subdir(root, &q8_req),
        Some(("q4", root.join("q4")))
    );

    // A torn q8 (config.json only) is skipped — not resolved and then failed inside the loader.
    std::fs::create_dir_all(root.join("q8")).unwrap();
    std::fs::write(root.join("q8").join("config.json"), b"x").unwrap();
    assert!(!krea_realtime_tier_is_complete(root, "q8"));
    assert_eq!(
        krea_realtime_missing_tier_files(root, "q8"),
        vec![
            "q8/dit.safetensors",
            "q8/t5_encoder.safetensors",
            "q8/tokenizer.json",
            "q8/vae.safetensors",
        ],
        "the loud-failure path must NAME what a torn tier is missing"
    );
    assert_eq!(
        krea_realtime_tier_subdir(root, &q8_req),
        Some(("q4", root.join("q4")))
    );

    // Completed q8 wins for an explicit q8 pick; a default job still resolves q4-first.
    write_complete_krea_tier(root, "q8");
    assert_eq!(
        krea_realtime_tier_subdir(root, &q8_req),
        Some(("q8", root.join("q8")))
    );
    assert_eq!(
        krea_realtime_tier_subdir(root, &default),
        Some(("q4", root.join("q4")))
    );

    // bf16: the nested transformer/ layout resolves to the TIER dir (what the engine's
    // `open_transformer_weights` probes), and the shards really are one level below it.
    write_complete_krea_tier(root, "bf16");
    let (name, resolved) = krea_realtime_tier_subdir(root, &bf16_req).expect("bf16 resolves");
    assert_eq!(
        name, "bf16",
        "the resolved tier names ITSELF — the asset records this"
    );
    assert_eq!(resolved, root.join("bf16"));
    assert!(
        resolved
            .join("transformer/dit-00004-of-00007.safetensors")
            .is_file(),
        "the resolved dir must be the tier dir the sharded DiT hangs off"
    );
    assert!(
        !resolved.join("dit.safetensors").exists(),
        "the bf16 tier deliberately has no flat DiT — resolving the tier dir is what makes it loadable"
    );
    // Still q4-first for a default job even with all three installed: the tier order prefers the
    // lean tier, it does not merely take whatever exists.
    assert_eq!(
        krea_realtime_tier_subdir(root, &default),
        Some(("q4", root.join("q4")))
    );

    // Losing ONE of the seven bf16 shards makes the tier incomplete — the sc-13513 "installed but
    // unloadable" class — so an explicit bf16 pick falls through to q8 instead.
    std::fs::remove_file(root.join("bf16/transformer/dit-00004-of-00007.safetensors")).unwrap();
    assert!(!krea_realtime_tier_is_complete(root, "bf16"));
    assert_eq!(
        krea_realtime_missing_tier_files(root, "bf16"),
        vec!["bf16/transformer/dit-00004-of-00007.safetensors"],
        "a torn bf16 must name the exact missing shard, not just report incomplete"
    );
    assert_eq!(
        krea_realtime_tier_subdir(root, &bf16_req),
        Some(("q8", root.join("q8")))
    );

    // An unpublished tier is never complete, even as a real (empty) directory.
    std::fs::create_dir_all(root.join("q6")).unwrap();
    assert!(!krea_realtime_tier_is_complete(root, "q6"));
    assert!(
        !krea_realtime_missing_tier_files(root, "q6").is_empty(),
        "an unpublished tier must report ITSELF missing — an empty vec would read COMPLETE"
    );
}

/// 🔴 The story's trap, end to end: a user who deliberately installed ONLY the **bf16** tier must get
/// bf16 **dense**, never the Q4 default applied to it in memory. The tier dir and the load-time quant
/// come out of ONE decision, so a resolved tier always loads exactly as it sits on disk
/// (`quantize: None`) — `mlxQuantize` picks WHICH tier, never a requant.
///
/// Discriminating rather than "asserts None everywhere": the same default request against a q4 install
/// resolves the q4 dir, and a FLAT snapshot in the same test does get `Some(Q4)` — so a resolver that
/// simply always returned `None`, or always returned the root, fails here.
///
/// Mutation checks (each verified RED): return `resolve_mlx_dense_quant(request)` alongside a resolved
/// tier; drop `"bf16"` from the default tier order (⇒ the bf16-only install falls to the root branch).
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_bf16_install_is_served_dense_not_downgraded_to_q4() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_krea_bf16_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // `local_mlx_dir` counts an override dir only when it holds a `config.json`; the tier matrix's
    // own config.json files live inside the tier dirs, so the fixture carries a root marker.
    std::fs::write(root.join("config.json"), "{}").unwrap();
    write_complete_krea_tier(root, "bf16");
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..offline_settings()
    };
    let default = krea_tier_request(json!({}));

    temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            let load = resolve_krea_realtime_tier_dir_and_quant(&settings, &default)
                .expect("a bf16-only install is loadable");
            assert_eq!(
                load.dir,
                root.join("bf16"),
                "the ONLY installed tier must be the one resolved, even under the Q4 default"
            );
            assert_eq!(
                load.quant, None,
                "a deliberately-installed bf16 tier must load DENSE — quantizing it to Q4 in memory \
                 is the silent creative-tier switch this story exists to prevent"
            );
            assert_eq!(
                load.tier, "bf16",
                "the asset must record the tier that LOADED — a default job's `advanced` says \
                 nothing about bf16, so only this stamp tells the user what actually ran"
            );
            // An explicit bf16 pick agrees, and the route is live (not Stub).
            let explicit = krea_tier_request(json!({ "mlxQuantize": 0 }));
            let explicit_load =
                resolve_krea_realtime_tier_dir_and_quant(&settings, &explicit).unwrap();
            assert_eq!(explicit_load.dir, root.join("bf16"));
            assert_eq!(explicit_load.quant, None);
            assert_eq!(explicit_load.tier, "bf16");
            assert!(krea_realtime_available(&default, &settings));

            // Now install q4 too: the SAME default request moves to q4 — proving the bf16 answer
            // above was the tier order resolving, not a constant.
            write_complete_krea_tier(root, "q4");
            let load = resolve_krea_realtime_tier_dir_and_quant(&settings, &default).unwrap();
            assert_eq!(load.dir, root.join("q4"));
            assert_eq!(
                load.quant, None,
                "a packed tier is never asked to re-quantize"
            );
            assert_eq!(load.tier, "q4");

            // 🔴 The provenance case this stamp exists for: an explicit `mlxQuantize: 8` with only
            // q4/bf16 on disk FALLS BACK — so the asset's own `advanced` would claim 8 while the
            // clip rendered at q4. The stamp must contradict the request, not echo it.
            let asked_q8 = krea_tier_request(json!({ "mlxQuantize": 8 }));
            let fell_back = resolve_krea_realtime_tier_dir_and_quant(&settings, &asked_q8).unwrap();
            assert_eq!(fell_back.tier, "q4", "q8 was asked for; q4 is what loaded");
            let raw = krea_realtime_raw_settings(
                &asked_q8,
                fell_back.tier,
                &KreaRealtimeLoraCoverage::default(),
            );
            assert!(
                raw.get("kreaRealtimeLoraPartial").is_none(),
                "no LoRA was dropped, so the partial-application key must be ABSENT — the sidecar \
                 of an ordinary render is unchanged by sc-15017"
            );
            assert_eq!(
                raw.get("kreaRealtimeTier").and_then(Value::as_str),
                Some("q4"),
                "the sidecar must name the tier that ran, not the one requested"
            );
            assert_eq!(
                raw.get("mlxQuantize").and_then(Value::as_i64),
                Some(8),
                "the request's own knob is still recorded verbatim — the tier stamp is what \
                 disambiguates it, so both must be present and DIFFERENT here"
            );
        },
    );
}

/// A FLAT snapshot (a local convert / env override with the DiT at its root — the only layout that has
/// no tier to speak for it) keeps the load-time quant path, and its default is **Q4**: sc-10750's
/// dense-video convention, which reaches the video lane through `resolve_mlx_dense_quant` here.
///
/// This is the assertion that discriminates against the shipped behaviour before this story: the arm
/// hard-coded `quant: None`, so a default job loaded a ~28 GB dense DiT. The end-to-end leg drives the
/// real generate path and reads the `LoadSpec` that reached the engine, not just the resolver.
///
/// Mutation checks (each verified RED): restore `quant: None` in `generate_krea_realtime_using`; make
/// the flat branch return `None` instead of `resolve_mlx_dense_quant(request)`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_flat_snapshot_takes_the_q4_dense_video_default() {
    let flat_guard = krea_fake_model_dir("flatquant");
    let flat = flat_guard.path();
    let settings = Settings {
        data_dir: flat.join("unused-data-dir"),
        ..offline_settings()
    };

    let probe = ArmProbe::default();
    let loader = probe.loader();
    let job = krea_job_snapshot();
    let req = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
        "duration": 2.0,
        "fps": 24
    }));

    let loaded_tier = temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        flat.to_str().unwrap(),
        || {
            // The resolver's own answers, across the whole `mlxQuantize` axis.
            for (advanced, expected) in [
                (json!({}), Some(Quant::Q4)),
                (json!({ "mlxQuantize": 4 }), Some(Quant::Q4)),
                (json!({ "mlxQuantize": 8 }), Some(Quant::Q8)),
                (json!({ "mlxQuantize": 0 }), None),
            ] {
                let load = resolve_krea_realtime_tier_dir_and_quant(
                    &settings,
                    &krea_tier_request(advanced.clone()),
                )
                .expect("a flat snapshot is loadable");
                assert_eq!(load.dir, flat, "a flat snapshot loads from its own root");
                assert_eq!(
                    load.quant, expected,
                    "flat snapshot with advanced {advanced}: the load-time quant must follow \
                     mlxQuantize, defaulting to Q4"
                );
                // A flat snapshot has no tier dir to name itself, so the stamp comes from the
                // quant that will pack it — the `mochiTier` mapping.
                assert_eq!(
                    load.tier,
                    match expected {
                        Some(Quant::Q4) => "q4",
                        Some(Quant::Q8) => "q8",
                        _ => "bf16",
                    },
                    "flat snapshot with advanced {advanced}: the recorded tier must be the \
                     precision the DiT actually runs at"
                );
            }

            // …and what actually reaches the engine for a DEFAULT job.
            let (_, raw_settings) = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &req,
                    flat,
                    "krea_realtime_14b",
                    "mlx",
                    loader,
                ))
                .expect("krea t2v generation runs through the funnel against the probe");
            raw_settings
                .get("kreaRealtimeTier")
                .and_then(Value::as_str)
                .map(str::to_owned)
        },
    );
    assert_eq!(
        loaded_tier.as_deref(),
        Some("q4"),
        "the arm must stamp the tier it loaded on the asset it returns"
    );

    let seen = probe.spec.lock().unwrap();
    let spec = seen.as_ref().expect("the arm loaded an engine");
    assert_eq!(
        spec.quantize,
        Some(Quant::Q4),
        "a default krea job must reach the engine at Q4 — the shipped arm hard-coded `quant: None`, \
         so a default job loaded the dense ~28 GB DiT"
    );
    assert_eq!(spec_weights_dir(spec), flat);

    drop(seen);
}

/// A resolved TIER reaches the engine as the tier DIR with `quantize: None` — the loadability half of
/// the release gate. Before this story the arm handed the engine the snapshot ROOT, where
/// `t2v::open_transformer_weights` finds neither `dit.safetensors` nor `transformer/`, so nothing Krea
/// shipped could load at all.
///
/// Mutation check (verified RED): hand `resolve_krea_realtime_model_dir(settings)?` to `VideoGenInput`
/// instead of the tier-resolved dir.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_hands_the_engine_the_installed_tier_dir_not_the_snapshot_root() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_krea_arm_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    write_complete_krea_tier(root, "q4");
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..offline_settings()
    };
    let probe = ArmProbe::default();
    let loader = probe.loader();
    let job = krea_job_snapshot();
    let req = request(json!({
        "projectId": "p",
        "model": "krea_realtime_14b",
        "mode": "text_to_video",
        "prompt": "p",
        "duration": 2.0,
        "fps": 24
    }));

    temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_krea_realtime_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &req,
                    root,
                    "krea_realtime_14b",
                    "mlx",
                    loader,
                ))
                .expect("krea t2v generation runs through the funnel against the probe");
        },
    );

    let seen = probe.spec.lock().unwrap();
    let spec = seen.as_ref().expect("the arm loaded an engine");
    assert_eq!(
        spec_weights_dir(spec),
        root.join("q4"),
        "the engine must be handed the installed TIER dir — every published Krea file lives under a \
         q4/ / q8/ / bf16/ prefix, so the snapshot root holds no weights at all"
    );
    assert_eq!(
        spec.quantize, None,
        "a pre-packed tier loads exactly as it sits on disk"
    );

    drop(seen);
}

/// A tier-matrix root with NO complete tier fails loudly and actionably — it must never fall back to
/// handing the engine the root, which holds no weights and would surface as an opaque loader error.
/// The route degrades to `Stub`, whose fail-loud gate then reports it.
///
/// Mutation check (verified RED): add a `_ => Ok((root, resolve_mlx_dense_quant(request)))` fallback
/// to `resolve_krea_realtime_tier_dir_and_quant`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_torn_install_fails_loudly_instead_of_resolving_the_root() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_krea_torn_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // The published repo root: a card + tier dirs, no weights of its own. `config.json` is the
    // `local_mlx_dir` marker so the ROOT resolves — the point is that resolving the root is not
    // enough, which is exactly the bug this story fixes.
    std::fs::write(root.join("config.json"), "{}").unwrap();
    std::fs::write(root.join("README.md"), "# card").unwrap();
    // A torn q4: everything but the DiT.
    std::fs::create_dir_all(root.join("q4")).unwrap();
    for file in ["config.json", "t5_encoder.safetensors", "tokenizer.json"] {
        std::fs::write(root.join("q4").join(file), b"x").unwrap();
    }
    let settings = Settings {
        data_dir: root.join("unused-data-dir"),
        ..offline_settings()
    };
    let req = krea_tier_request(json!({}));

    temp_env_var(
        "SCENEWORKS_MLX_KREA_REALTIME_DIR",
        root.to_str().unwrap(),
        || {
            // The root itself DOES resolve — so a test that only checked `resolve_..._model_dir`
            // would be green here. That is the false green this story was filed against.
            assert!(resolve_krea_realtime_model_dir(&settings).is_ok());

            let err = resolve_krea_realtime_tier_dir_and_quant(&settings, &req)
                .expect_err("a torn install must not resolve");
            let WorkerError::InvalidPayload(message) = err else {
                panic!("expected an actionable InvalidPayload");
            };
            assert!(
                message.contains("krea_realtime_14b")
                    && message.contains("q4")
                    && message.contains("bf16")
                    && message.contains("Model Manager"),
                "the error must name the model, the tiers, and what to do: {message}"
            );
            assert!(!krea_realtime_available(&req, &settings));
            assert_eq!(resolve_video_route(&req, &settings), VideoRoute::Stub);
            assert!(ensure_video_engine_weights(&req, &settings).is_err());

            // Completing the tier flips every one of those.
            write_complete_krea_tier(root, "q4");
            let load = resolve_krea_realtime_tier_dir_and_quant(&settings, &req).unwrap();
            assert_eq!(load.dir, root.join("q4"));
            assert_eq!(load.quant, None);
            assert_eq!(load.tier, "q4");
            assert!(krea_realtime_available(&req, &settings));
        },
    );
}

/// The tier matrix installed the way users actually get it — the pinned HF snapshot under the data
/// dir, with no env override in play. Pins that `resolve_krea_realtime_model_dir`'s HF branch feeds the
/// tier descent (the production path), not just the override branch the other tests exercise.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_resolves_the_tier_from_a_downloaded_hf_snapshot() {
    let hub_guard = tempfile::Builder::new()
        .prefix("sw_krea_hub_")
        .tempdir()
        .expect("temp dir");
    let hub = hub_guard.path();
    let repo_dir = hub.join(format!(
        "models--{}",
        sceneworks_core::hf_home::safe_repo_dir_name(KREA_REALTIME_REPO)
            .expect("the krea repo slug")
    ));
    let snapshot = repo_dir.join("snapshots").join(KREA_REALTIME_REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("README.md"), "# card").unwrap();
    write_complete_krea_tier(&snapshot, "q8");
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs").join("main"), KREA_REALTIME_REVISION).unwrap();

    let settings = Settings {
        data_dir: hub.join("data"),
        ..offline_settings()
    };
    // The HF cache dir is its own axis: a pinned `data_dir` alone does NOT make it hermetic.
    temp_env_vars(
        &[
            ("SCENEWORKS_MLX_KREA_REALTIME_DIR", ""),
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ],
        || {
            // Only q8 is installed, so even a DEFAULT (q4-first) job resolves it — and dense-loads
            // nothing: it is served as the packed tier it is.
            let load =
                resolve_krea_realtime_tier_dir_and_quant(&settings, &krea_tier_request(json!({})))
                    .expect("the downloaded q8 tier resolves");
            assert_eq!(load.dir, snapshot.join("q8"));
            assert_eq!(load.quant, None);
            assert_eq!(load.tier, "q8");
            // The ROOT resolves too — which is precisely why resolving it was never enough.
            assert_eq!(
                resolve_krea_realtime_model_dir(&settings).unwrap(),
                snapshot
            );
        },
    );
}

/// The on-demand tier fetch is gated on the request's tier toggle — the krea twin of
/// `mochi_on_demand_tier_fetch_is_gated_on_the_toggle`, and it matters more here: krea's opt-in tiers
/// are 27 GB (q8) and 40 GB (bf16), so a default job that wanted one would drag an enormous
/// unrequested download into every generation.
///
/// This tests the decision DIRECTLY. Both end-to-end tests only ever reach
/// `ensure_krea_realtime_tier_present`'s default arm — and their fixtures point `data_dir` at a
/// non-existent dir, so `huggingface_snapshot_dir` returns `None` and the function exits before the
/// tier mapping is even consulted. A mutation making a DEFAULT job want q8/bf16, or dropping the
/// `{tier}/` prefix from the fetched paths, would stay GREEN through them.
///
/// Mutation checks (each verified RED): make the `_` arm return `Some("q8")`; swap the `None`/`Q8`
/// arms; drop the `{tier}/` prefix in `krea_realtime_tier_fetch_files`.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_on_demand_tier_fetch_is_gated_on_the_toggle() {
    // Default ⇒ nothing to fetch: q4 ships with the base install.
    assert_eq!(
        krea_realtime_on_demand_tier(&krea_tier_request(json!({}))),
        None
    );
    // An explicit q4 pick is still the base install.
    assert_eq!(
        krea_realtime_on_demand_tier(&krea_tier_request(json!({ "mlxQuantize": 4 }))),
        None
    );
    // q8 toggle ⇒ q8 ONLY (never bf16 too — that would drag 40 GB on a 27 GB request).
    assert_eq!(
        krea_realtime_on_demand_tier(&krea_tier_request(json!({ "mlxQuantize": 8 }))),
        Some("q8")
    );
    // bf16 toggle ⇒ bf16 only.
    assert_eq!(
        krea_realtime_on_demand_tier(&krea_tier_request(json!({ "mlxQuantize": 0 }))),
        Some("bf16")
    );
    // A numeric STRING is the same toggle (the shared resolver parses both forms).
    assert_eq!(
        krea_realtime_on_demand_tier(&krea_tier_request(json!({ "mlxQuantize": "8" }))),
        Some("q8")
    );

    // The fetched paths are the tier's EXPLICIT file list, each prefixed with `<tier>/` — the same
    // paths the manifest declares, not a `<tier>/*` glob.
    assert_eq!(
        krea_realtime_tier_fetch_files("q8").expect("q8 is published"),
        vec![
            "q8/config.json",
            "q8/dit.safetensors",
            "q8/t5_encoder.safetensors",
            "q8/tokenizer.json",
            "q8/vae.safetensors",
        ]
    );
    // bf16 fetches its NESTED shards, individually — the packed row would miss the transformer/ dir
    // entirely, leaving an "installed" tier with no DiT.
    let bf16 = krea_realtime_tier_fetch_files("bf16").expect("bf16 is published");
    assert_eq!(bf16.len(), 11);
    assert!(bf16.contains(&"bf16/transformer/dit-00007-of-00007.safetensors".to_owned()));
    assert!(
        bf16.iter().all(|file| file.starts_with("bf16/")),
        "every fetched path must be tier-prefixed, or the fetch pulls the wrong files: {bf16:?}"
    );
    assert_eq!(krea_realtime_tier_fetch_files("q6"), None);
}

/// 🔴 The guarantee the on-demand fetch's own doc makes, exercised against a stub hub whose `q8/` tier
/// is MISSING its DiT — the "published but torn remotely" case.
///
/// This is the hole the fetch would otherwise have. `ensure_hf_files_cached` does NOT run the sc-12283
/// `unmatched_patterns` per-pattern hard-fail (that lives in `model_jobs::run_model_download_job`), so
/// a declared path matching nothing remotely is silently a no-op: the other four q8 files land, the
/// tier reads INCOMPLETE, and `krea_realtime_tier_subdir` quietly serves q4 instead — the exact silent
/// tier downgrade this lane exists to prevent, arriving through the back door. So the fetch verifies on
/// disk afterwards and fails loud, NAMING the missing file.
///
/// The stub serves the HF tree + file bytes AND the worker's own progress endpoint, so the whole fetch
/// path runs for real (resolve → download → verify) rather than being mocked out at the seam.
///
/// Mutation checks (each verified RED): delete the post-fetch verification block; make it `warn!`
/// instead of returning `Err`; have it check the request's tier instead of the fetched one.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_fetch_of_a_remotely_torn_tier_fails_loudly_instead_of_downgrading() {
    use axum::body::Body;
    use axum::http::{Response, StatusCode};
    use axum::Router;

    let hub_guard = tempfile::Builder::new()
        .prefix("sw_krea_fetch_")
        .tempdir()
        .expect("temp dir");
    let hub = hub_guard.path();
    let repo_dir = hub.join(format!(
        "models--{}",
        sceneworks_core::hf_home::safe_repo_dir_name(KREA_REALTIME_REPO)
            .expect("the krea repo slug")
    ));
    let snapshot = repo_dir.join("snapshots").join(KREA_REALTIME_REVISION);
    // The repo IS downloaded (so `huggingface_snapshot_dir` resolves and the fetch is attempted) but
    // holds only the default q4 tier — a q8 job must fetch q8.
    std::fs::create_dir_all(&snapshot).unwrap();
    write_complete_krea_tier(&snapshot, "q4");
    std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
    std::fs::write(repo_dir.join("refs").join("main"), KREA_REALTIME_REVISION).unwrap();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("test runtime builds");

    // A stub hub whose q8 tier is missing `dit.safetensors` — every other declared file resolves.
    let served: Vec<String> = krea_realtime_tier_fetch_files("q8")
        .expect("q8 is published")
        .into_iter()
        .filter(|file| file != "q8/dit.safetensors")
        .collect();
    let job_json = serde_json::to_string(&krea_job_snapshot()).expect("job snapshot serializes");
    let address = runtime.block_on(async move {
        let app = Router::new().fallback(move |uri: axum::http::Uri| {
            let served = served.clone();
            let job_json = job_json.clone();
            async move {
                let path = uri.path().to_owned();
                if path.contains("/tree/") {
                    let entries: Vec<Value> = served
                        .iter()
                        .map(|file| json!({ "type": "file", "path": file, "size": 4 }))
                        .collect();
                    return Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::to_string(&entries).expect("tree serializes"),
                        ))
                        .expect("response");
                }
                if path.ends_with("/progress") {
                    return Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(job_json))
                        .expect("response");
                }
                if path.contains("/resolve/") {
                    return Response::builder()
                        .header("content-length", "4")
                        .body(Body::from(b"xxxx".to_vec()))
                        .expect("response");
                }
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::empty())
                    .expect("response")
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        address
    });
    let base = format!("http://{address}");

    let settings = Settings {
        data_dir: hub.join("data"),
        api_url: base.clone(),
        huggingface_base_url: base,
        ..offline_settings()
    };
    let job = krea_job_snapshot();
    let q8_req = krea_tier_request(json!({ "mlxQuantize": 8 }));

    let error = temp_env_vars(
        &[
            ("SCENEWORKS_MLX_KREA_REALTIME_DIR", ""),
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ],
        || {
            runtime.block_on(ensure_krea_realtime_tier_present(
                &ApiClient::new(&settings),
                &settings,
                &job,
                &q8_req,
            ))
        },
    )
    .expect_err("a remotely-torn tier must not be accepted");

    let WorkerError::InvalidPayload(message) = error else {
        panic!("expected an actionable InvalidPayload, got {error:?}");
    };
    assert!(
        message.contains("q8/dit.safetensors"),
        "the failure must NAME the file that never arrived: {message}"
    );
    assert!(
        message.contains("q8"),
        "the failure must name the tier the user picked: {message}"
    );
    // The other four q8 files DID land — proving the fetch really ran and it is the verification,
    // not a transport error, that refused the job.
    for file in ["config.json", "tokenizer.json", "vae.safetensors"] {
        assert!(
            snapshot.join("q8").join(file).is_file(),
            "the fetch should have downloaded q8/{file}"
        );
    }
}

/// The Krea tier repo must pin an exact commit (not the mutable `main`) so an upstream re-push can't
/// swap a checkpoint the on-demand fetch loads (mirrors the Wan / SCAIL-2 pins), and
/// `krea_realtime_tier_repo` routes only the krea id to it.
#[cfg(target_os = "macos")]
#[test]
fn krea_realtime_tier_revision_is_a_pinned_commit_not_main() {
    assert_ne!(
        KREA_REALTIME_REVISION, "main",
        "the Krea tier repo must pin a fixed revision before release"
    );
    assert_eq!(
        KREA_REALTIME_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        KREA_REALTIME_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned Krea revision must be lowercase hex"
    );
    assert_eq!(
        krea_realtime_tier_repo("krea_realtime_14b"),
        Some((KREA_REALTIME_REPO, KREA_REALTIME_REVISION))
    );
    for other in ["scail2_14b", "wan_2_2", "bernini", "krea_2_turbo"] {
        assert_eq!(
            krea_realtime_tier_repo(other),
            None,
            "{other} must not route to the Krea Realtime tier repo"
        );
    }
}

/// The on-demand tier fetches are gated on the request's tier toggle — the "fetched on demand if
/// toggled but not downloaded" AC. Pins that a default (q4) job asks for NEITHER extra tier, so
/// no job accidentally pulls the ~13 GB q8 / ~20 GB bf16.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_on_demand_tier_fetch_is_gated_on_the_toggle() {
    // Default ⇒ neither on-demand tier is wanted.
    assert!(!mochi_wants_q8(&mochi_request(json!({}))));
    assert!(!mochi_wants_bf16(&mochi_request(json!({}))));
    // q8 toggle ⇒ q8 only.
    assert!(mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 8 }))));
    assert!(!mochi_wants_bf16(&mochi_request(
        json!({ "mlxQuantize": 8 })
    )));
    // bf16 toggle ⇒ bf16 only (the two must be mutually exclusive, or a bf16 job would also drag
    // the q8 tier down).
    assert!(mochi_wants_bf16(&mochi_request(
        json!({ "mlxQuantize": 0 })
    )));
    assert!(!mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 0 }))));
    // An explicit q4 pick pulls nothing extra.
    assert!(!mochi_wants_q8(&mochi_request(json!({ "mlxQuantize": 4 }))));
    assert!(!mochi_wants_bf16(&mochi_request(
        json!({ "mlxQuantize": 4 })
    )));
}

/// The pinned revision must be a full 40-char commit, not a mutable branch: the repo is a
/// hard-coded const, so `main` would let an upstream re-push swap a checkpoint we load.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn mochi_repo_revision_is_pinned_to_a_commit() {
    assert_eq!(MOCHI_REPO, "SceneWorks/mochi-1-mlx");
    assert_eq!(MOCHI_REVISION.len(), 40, "a full commit sha, not a branch");
    assert!(MOCHI_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    // The revision B1's manifest sizes were read from.
    assert!(MOCHI_REVISION.starts_with("90a87786"));
}

/// The MLX asset record names the tier that was actually LOADED, so provenance can't drift from
/// the tier the resolver picked when a requested tier isn't installed.
#[cfg(target_os = "macos")]
#[test]
fn mochi_raw_settings_record_the_loaded_tier() {
    let req = mochi_request(json!({ "mlxQuantize": 8 }));
    // Requested q8 but q4 was what resolved ⇒ the record says q4, not q8.
    let raw = mochi_raw_settings(&req, Some(Quant::Q4));
    assert_eq!(raw["mochiTier"], json!("q4"));
    assert_eq!(raw["realModelInference"], json!(true));
    assert_eq!(raw["model"], json!("mochi_1"));
    // No frameCount: the builder records the dispatched KNOBS; the clip's length is stamped from
    // the encoded frames at the funnel (sc-12371, `record_frame_count`).
    assert!(raw.get("frameCount").is_none());
    assert_eq!(
        mochi_raw_settings(&req, Some(Quant::Q8))["mochiTier"],
        json!("q8")
    );
    assert_eq!(mochi_raw_settings(&req, None)["mochiTier"], json!("bf16"));
}

// The env seam is crate-wide (`crate::test_env`), not per-module: this binary runs every
// module's tests in ONE process, so a lock private to `video_jobs` serializes nothing against
// the `image_jobs` / `training_jobs` tests that write the same vars (sc-12380).
//
// Gated to match the consumers exactly. The parity lane builds this crate with NEITHER backend
// and runs `clippy -D warnings` over the test target, so an import held under a wider cfg than
// its users is an `unused_imports` error there rather than a warning.
// `offline_settings` is the one way to build a `Settings` for a test whose path could reach a
// download: it pins every network field unroutable, so no call site has to remember that
// `huggingface_base_url` (not `api_url`) is what a tier fetch dials. See its doc comment.
// Its consumers include the candle Mochi twins, so it rides the same `any(macos, candle)` gate
// as the helpers above rather than the macOS-only one below.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::test_env::{offline_settings, temp_env_var, temp_env_vars};

// The macOS-gated adapter fixtures (wan Lightning + ltx distill) pin a var for a whole test body —
// see the cfg note above.
#[cfg(target_os = "macos")]
use crate::test_env::EnvVars;

/// sc-10418: the resolved video quant maps to the same normalized tier labels the
/// image lane uses, so the Stats charts group video + image runs on one axis.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_quant_label_maps_tiers() {
    assert_eq!(
        video_quant_label(Some(Quant::Q8)),
        (Some("q8".to_owned()), Some(8))
    );
    assert_eq!(
        video_quant_label(Some(Quant::Q4)),
        (Some("q4".to_owned()), Some(4))
    );
    assert_eq!(video_quant_label(None), (Some("bf16".to_owned()), None));
}

/// sc-11042 / epic 11037 SC#5 — **the NVFP4 aliasing guard**.
///
/// `Quant::Nvfp4::bits()` returns **4** (E2M1 elements are 4-bit), so the previous
/// `format!("q{bits}")` implementation stamped an NVFP4 video render as `"q4"` + bits 4 — reporting
/// the int4-affine tier the user did NOT pick. The compiler cannot catch that class of bug (reading
/// `.bits()` raises no E0004 on a new variant), so it is pinned here instead.
///
/// The `bits()`-is-4 assertion is deliberate: it documents WHY the label must be matched from the
/// variant, and fails loudly if a future contract change makes the old bits-derived form look safe
/// again.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_quant_label_never_aliases_nvfp4_to_q4() {
    let (label, bits) = video_quant_label(Some(Quant::Nvfp4));
    // The label names the tier the user actually picked — NOT "q4".
    assert_eq!(label, Some("nvfp4".to_owned()));
    assert_ne!(
        label,
        Some("q4".to_owned()),
        "NVFP4 must never be stamped as the q4 tier (epic 11037 SC#5)"
    );
    // No integer bit-width is honest for NVFP4 (~4.5 EFFECTIVE bits/weight), so `quant_bits` stays
    // empty rather than repeating the q4 aliasing in the numeric column.
    assert_eq!(bits, None);
    // The trap this guards: the element width really is 4, which is exactly why a bits-derived
    // label silently aliased onto q4.
    assert_eq!(Quant::Nvfp4.bits(), 4);
    assert_eq!(Quant::Q4.bits(), 4);
    // And the q4 tier still reports itself truthfully — NVFP4 took nothing away from it.
    assert_eq!(
        video_quant_label(Some(Quant::Q4)),
        (Some("q4".to_owned()), Some(4))
    );
}

/// sc-10418 (mirrors the S4 image `image_settings_metrics` tests): the video
/// settings builder folds the resolved effective settings onto the phase-timing
/// block WITHOUT clobbering the timings, records one output (sc-10426), and carries
/// quant / sampler / scheduler / guidance / dims / seed through.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn build_video_metrics_folds_settings_onto_timing() {
    let timing = GenerationMetrics {
        load_ms: Some(3000),
        sample_ms: Some(12000),
        decode_ms: Some(2000),
        ..Default::default()
    };
    let settings = VideoSettingsSnapshot {
        quant: Some(Quant::Q4),
        sampler: Some("euler".to_owned()),
        scheduler: Some("karras".to_owned()),
        scheduler_shift: Some(3.0),
        guidance: Some(5.0),
        width: 832,
        height: 480,
        seed: 1234,
    };
    let metrics = build_video_metrics(
        timing,
        &settings,
        Some("wan2_2_t2v_14b".to_owned()),
        Some(4),
    );
    // Phase timings survive the fold.
    assert_eq!(metrics.load_ms, Some(3000));
    assert_eq!(metrics.sample_ms, Some(12000));
    assert_eq!(metrics.decode_ms, Some(2000));
    // Effective settings folded in.
    assert_eq!(metrics.model.as_deref(), Some("wan2_2_t2v_14b"));
    assert_eq!(metrics.quant_label.as_deref(), Some("q4"));
    assert_eq!(metrics.quant_bits, Some(4));
    assert_eq!(metrics.sampler.as_deref(), Some("euler"));
    assert_eq!(metrics.scheduler.as_deref(), Some("karras"));
    assert_eq!(
        metrics.scheduler_shift.as_ref().and_then(|n| n.as_f64()),
        Some(3.0)
    );
    assert_eq!(metrics.steps, Some(4));
    assert_eq!(metrics.image_count, Some(1));
    assert_eq!(
        metrics.guidance_scale.as_ref().and_then(|n| n.as_f64()),
        Some(5.0)
    );
    assert_eq!(metrics.guidance_method.as_deref(), Some("cfg"));
    assert_eq!(metrics.width, Some(832));
    assert_eq!(metrics.height, Some(480));
    assert_eq!(metrics.seed, Some(1234));
}

/// sc-10418: a default-settings video run (no user sampler/scheduler override,
/// engine-default guidance) still reports non-blank sampler/scheduler so the charts
/// have a group, but leaves guidance / scheduler-shift `None` — an honest "engine
/// default, not captured" rather than a fabricated value.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn build_video_metrics_defaults_when_unset() {
    let settings = VideoSettingsSnapshot {
        quant: None,
        sampler: None,
        scheduler: None,
        scheduler_shift: None,
        guidance: None,
        width: 1280,
        height: 720,
        seed: 7,
    };
    let metrics = build_video_metrics(GenerationMetrics::default(), &settings, None, None);
    assert_eq!(metrics.quant_label.as_deref(), Some("bf16"));
    assert_eq!(metrics.quant_bits, None);
    assert_eq!(metrics.sampler.as_deref(), Some("default"));
    assert_eq!(metrics.scheduler.as_deref(), Some("default"));
    assert!(metrics.scheduler_shift.is_none());
    assert!(metrics.guidance_scale.is_none());
    assert_eq!(metrics.image_count, Some(1));
    assert_eq!(metrics.width, Some(1280));
    assert_eq!(metrics.height, Some(720));
}

/// sc-17632 (epic 17625) — the video lane no longer owns a SeedVR2 pin, a checkpoint destination or
/// a download.
///
/// This REPLACES `seedvr2_video_revision_is_pinned_commit_not_main` +
/// `seedvr2_video_revision_matches_image_lane`, which held two verbatim copies of the same
/// repo/revision together. Deleting one copy makes that drift impossible by construction, which is
/// strictly stronger than asserting the copies agree — the format check still runs once, on the
/// surviving const, in `upscale_jobs::tests::seedvr2_revision_is_pinned_commit_not_main`.
///
/// What CAN still regress is someone re-introducing the duplicate (that is exactly how the two
/// destinations appeared in the first place), so gate on that: this module must name neither the
/// mirror nor the pinned sha, and must reach the checkpoint through the shared image-lane resolver.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_video_lane_has_no_private_checkpoint_pin() {
    const SOURCE: &str = include_str!("seedvr2.rs");
    assert!(
        !SOURCE.contains(crate::upscale_jobs::SEEDVR2_REPO),
        "the video lane must not name the SeedVR2 mirror itself — `upscale_jobs::SEEDVR2_REPO` is \
         the single source of truth (sc-17632)"
    );
    assert!(
        !SOURCE.contains(crate::upscale_jobs::SEEDVR2_REVISION),
        "the video lane must not carry its own copy of the pinned SeedVR2 revision (sc-17632)"
    );
    assert!(
        SOURCE.contains("upscale_jobs::require_seedvr2_checkpoint_dir"),
        "the video lane must resolve its checkpoint through the shared cache-only resolver, not a \
         private one (sc-17632)"
    );
}

/// sc-11168 / F-007 (completes the sc-9879 rollout on the video lanes): both the MLX
/// MLX and Candle A14B Lightning self-heal fetches share `ensure_wan_lightning_present` and pull the
/// FIXED `lightx2v/Wan2.2-Lightning` distill pair from the SHARED
/// `WAN_LIGHTNING_REVISION` const, so it must pin an exact commit rather than the mutable `main`
/// branch — an upstream re-push would otherwise silently swap the high/low distill weights we load.
/// Lock the pin to a real 40-hex lowercase commit id (mirrors the SeedVR2 format test above).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn wan_lightning_revision_is_pinned_commit_not_main() {
    assert_ne!(
        WAN_LIGHTNING_REVISION, "main",
        "Wan Lightning distill pair must pin a fixed revision"
    );
    assert_eq!(
        WAN_LIGHTNING_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        WAN_LIGHTNING_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned revision must be lowercase hex"
    );
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn wan_lightning_subdir_is_canonical_for_fetch_and_resolution() {
    assert_eq!(
        wan_lightning_subdir("wan2_2_t2v_14b"),
        Some("Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1")
    );
    assert_eq!(
        wan_lightning_subdir("wan2_2_i2v_14b"),
        Some("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1")
    );
    assert_eq!(wan_lightning_subdir("wan2_2_ti2v_5b"), None);
    assert_eq!(wan_lightning_subdir("bernini"), None);
}

/// sc-9879 (F-077 follow-up): `ensure_ltx_q8_present` pulls `q8/*` from the FIXED SceneWorks LTX-2.3
/// bundle const (non-overridable here), so it must pin an exact commit rather than the mutable `main`
/// branch — an upstream re-push would otherwise silently swap the Q8 checkpoint we load. Lock the pin
/// to a real 40-hex lowercase commit id (mirrors the SeedVR2 format test above).
#[cfg(target_os = "macos")]
#[test]
fn ltx_bundle_revision_is_pinned_commit_not_main() {
    assert_eq!(
        LTX_BUNDLE_REVISION, "01df27d308466533aa09d251e3aebdcc627d07eb",
        "the LTX pin must name the first revision that contains bf16/ (sc-18853)"
    );
    assert_ne!(
        LTX_BUNDLE_REVISION, "main",
        "LTX q8 bundle must pin a fixed revision"
    );
    assert_eq!(
        LTX_BUNDLE_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        LTX_BUNDLE_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned revision must be lowercase hex"
    );
}

/// The Models-screen install and generation-time tier fetch are two entry points into the same
/// bundle. Every LTX row must therefore use the worker pin, and the bf16 row must request the
/// directory that exists at that pin (sc-18853).
#[cfg(target_os = "macos")]
#[test]
fn ltx_bundle_manifest_rows_match_the_worker_pin() {
    use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
    use sceneworks_core::jsonc::strip_jsonc_comments;

    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present");
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(raw)).expect("builtin models parses as JSON");
    let model = manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"].as_str() == Some("ltx_2_3"))
        .expect("ltx_2_3 present in the builtin catalog");
    let downloads = model["downloads"]
        .as_array()
        .expect("ltx_2_3 declares downloads");

    assert_eq!(downloads.len(), 4, "gemma plus q4/q8/bf16 rows");
    for download in downloads {
        assert_eq!(download["repo"].as_str(), Some(LTX_BUNDLE_REPO));
        assert_eq!(
            download["revision"].as_str(),
            Some(LTX_BUNDLE_REVISION),
            "manifest and on-demand fetches must populate the same snapshot"
        );
    }
    let bf16 = downloads
        .iter()
        .find(|download| download["variant"].as_str() == Some("bf16"))
        .expect("bf16 tier row");
    assert_eq!(bf16["files"], json!(["bf16/*"]));
}

// sc-8828 (F-026): `resolve_video_route` is the extracted native (MLX) dispatch decision — the
// predicate ladder pulled out of `run_video_generate_job`'s 4-tuple match. These lock the branches
// whose routing depends only on the mode + model id (not on staged weights), so a future family edit
// that reorders the ladder or mis-routes `replace_person` fails loudly here.
#[cfg(target_os = "macos")]
#[test]
fn video_route_replace_person_dispatches_by_model() {
    let settings = Settings::from_env();

    // `replace_person` on the SCAIL-2 model → the SCAIL-2 replacement backend (sc-5452), carrying
    // the resolved engine id — NOT Wan-VACE.
    let scail2 = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "replace_person",
    }));
    assert_eq!(
        resolve_video_route(&scail2, &settings),
        VideoRoute::ReplacePersonScail2("scail2_14b"),
    );

    // `replace_person` on the dual-expert Wan2.2 VACE-Fun model → its own variant (sc-3459), never
    // the single-expert Wan-VACE checkpoint.
    let vace_fun = request(json!({
        "projectId": "p", "model": "wan_2_2_vace_fun_14b", "mode": "replace_person",
    }));
    assert_eq!(
        resolve_video_route(&vace_fun, &settings),
        VideoRoute::ReplacePersonWanVaceFun,
    );

    // `replace_person` on any other replace-capable model → single-expert Wan-VACE (sc-3521).
    let other = request(json!({
        "projectId": "p", "model": "wan_2_2_ti2v_5b", "mode": "replace_person",
    }));
    assert_eq!(
        resolve_video_route(&other, &settings),
        VideoRoute::ReplacePersonWanVace,
    );

    // An unknown model with no engine + no resolvable weights → the stub (the loud-fail path lives in
    // the execution arm via `ensure_video_engine_weights`).
    let unknown = request(json!({
        "projectId": "p", "model": "definitely_not_a_video_engine", "mode": "text_to_video",
    }));
    assert_eq!(resolve_video_route(&unknown, &settings), VideoRoute::Stub);
}

// sc-8828 (F-026): the candle sibling — `resolve_candle_video_route`. Locks the `backend_candle_enabled`
// gate (off → Stub, so routing is unchanged until parity) + the mode/model dispatch.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_video_route_gates_on_backend_flag_then_mode() {
    let mut settings = Settings::from_env();

    // Candle disabled (default) → always the stub, regardless of model/mode.
    settings.backend_candle_enabled = false;
    let scail2_replace = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "replace_person",
    }));
    assert_eq!(
        resolve_candle_video_route(&scail2_replace, &settings),
        CandleVideoRoute::Stub,
    );

    // Enabled: `replace_person` on the candle SCAIL-2 model → the SCAIL-2 replacement variant;
    // any other replace-capable model → candle Wan-VACE.
    settings.backend_candle_enabled = true;
    assert_eq!(
        resolve_candle_video_route(&scail2_replace, &settings),
        CandleVideoRoute::ReplacePersonScail2(scail2_engine_id("scail2_14b").unwrap()),
    );
    let extend = request(json!({
        "projectId": "p", "model": "wan_2_2_ti2v_5b", "mode": "extend_clip",
    }));
    assert_eq!(
        resolve_candle_video_route(&extend, &settings),
        CandleVideoRoute::WanVaceExtendBridge,
    );
}

/// sc-10997 (epic 6562): the candle Bernini VIDEO lane routes t2v + every editing/reference/
/// multi-source mode to `CandleVideoRoute::Bernini` (a DISTINCT engine, NOT the generic wan/ltx
/// txt2video arm), and only when the backend-candle flag is on. This per-mode dispatch is the
/// story's validation (GPU-val is gated on the `SceneWorks/bernini` weights, sc-11003).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_video_route_bernini_every_mode() {
    let mut settings = Settings::from_env();
    settings.backend_candle_enabled = true;
    let engine = bernini_engine_id("bernini").expect("bernini engine id");
    for mode in [
        "text_to_video",
        "video_to_video",
        "reference_to_video",
        "reference_video_to_video",
        "multi_video_to_video",
        "ads2v",
    ] {
        let req = request(json!({ "projectId": "p", "model": "bernini", "mode": mode }));
        assert_eq!(
            resolve_candle_video_route(&req, &settings),
            CandleVideoRoute::Bernini(engine),
            "bernini {mode} must route to CandleVideoRoute::Bernini",
        );
    }
    // Backend-candle off → Stub (routing unchanged until the flag is set), even for bernini.
    settings.backend_candle_enabled = false;
    let off = request(json!({
        "projectId": "p", "model": "bernini", "mode": "text_to_video",
    }));
    assert_eq!(
        resolve_candle_video_route(&off, &settings),
        CandleVideoRoute::Stub,
    );
}

/// sc-10997: the candle Bernini engine id + the SceneWorks-mode → engine `video_mode` task mapping,
/// the byte-identical twin of the MLX `bernini_engine_id` / `bernini_engine_video_mode`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn bernini_backends_share_engine_id_and_video_mode_mapping() {
    assert_eq!(bernini_engine_id("bernini"), Some("bernini"));
    // The still `bernini_image` id + every other video family keep their own routing.
    assert_eq!(bernini_engine_id("bernini_image"), None);
    assert_eq!(bernini_engine_id("wan_2_2"), None);
    assert_eq!(bernini_engine_id("scail2_14b"), None);
    assert_eq!(bernini_engine_id(""), None);

    assert_eq!(bernini_engine_video_mode("text_to_video"), "t2v");
    assert_eq!(bernini_engine_video_mode("video_to_video"), "v2v");
    assert_eq!(bernini_engine_video_mode("reference_to_video"), "r2v");
    assert_eq!(
        bernini_engine_video_mode("reference_video_to_video"),
        "rv2v"
    );
    assert_eq!(bernini_engine_video_mode("multi_video_to_video"), "mv2v");
    assert_eq!(bernini_engine_video_mode("ads2v"), "ads2v");
    // Unknown / image_to_video ⇒ plain t2v (the renderer is text-conditioned Wan2.2-T2V).
    assert_eq!(bernini_engine_video_mode("image_to_video"), "t2v");
    assert_eq!(bernini_engine_video_mode(""), "t2v");
}

#[cfg(target_os = "macos")]
#[test]
fn stall_timeout_override_parses_or_falls_back() {
    // A valid positive override wins.
    assert_eq!(
        parse_stall_timeout(Some("120".to_owned())),
        Duration::from_secs(120)
    );
    assert_eq!(
        parse_stall_timeout(Some("  90 ".to_owned())),
        Duration::from_secs(90)
    );
    // Unset, blank, non-numeric, or zero all fall back to the default.
    assert_eq!(parse_stall_timeout(None), VIDEO_STALL_TIMEOUT);
    assert_eq!(
        parse_stall_timeout(Some(String::new())),
        VIDEO_STALL_TIMEOUT
    );
    assert_eq!(
        parse_stall_timeout(Some("nope".to_owned())),
        VIDEO_STALL_TIMEOUT
    );
    assert_eq!(
        parse_stall_timeout(Some("0".to_owned())),
        VIDEO_STALL_TIMEOUT
    );
}

#[test]
fn plan_builds_nested_media_path() {
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox runs"
    }));
    let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
    assert!(plan
        .media_rel
        .starts_with(&format!("assets/videos/{}/", plan.genset_id)));
    assert!(plan.media_rel.ends_with(".mp4"));
    // The model id is slugified into the filename (F-003 / sc-11159): `ltx_2_3` -> `ltx-2-3`.
    assert!(plan.media_rel.contains("_ltx-2-3_"));
    assert!(plan.asset_id.starts_with("asset_"));
    assert_eq!(plan.family, "ltx-video");
    assert_eq!(
        plan.media_path,
        Path::new("/tmp/project").join(&plan.media_rel)
    );
}

/// F-003 / sc-11159: a path-traversal / absolute model id in the payload must NOT let the
/// rendered `.mp4` escape the project dir. The model id becomes a filename component in
/// `media_rel`, so a `../`, `..\`, or `/abs` id would otherwise place the media path (and
/// its later `create_dir_all` + write) outside `project_path`. Assert confinement per vector.
#[test]
fn plan_confines_malicious_model_id() {
    let project = Path::new("/tmp/project");
    for evil in [
        "../../../../etc/passwd",
        "..\\..\\..\\windows\\evil",
        "/etc/cron.d/pwn",
        "sub/dir/model",
    ] {
        let request = request(json!({
            "projectId": "p", "model": evil, "prompt": "A red fox runs"
        }));
        let plan = VideoPlan::new(&request, project);
        assert!(
            !plan.media_rel.contains(".."),
            "media_rel must not contain `..` for model id {evil:?}: {}",
            plan.media_rel
        );
        assert!(
            plan.media_rel
                .starts_with(&format!("assets/videos/{}/", plan.genset_id)),
            "media_rel escaped the videos dir for model id {evil:?}: {}",
            plan.media_rel
        );
        assert!(
            plan.media_path.starts_with(project),
            "media_path {:?} escaped project dir for model id {evil:?}",
            plan.media_path
        );
    }
}

#[test]
fn family_prefers_manifest_then_infers_from_model() {
    let manifest = request(json!({
        "projectId": "p", "model": "ltx_2_3",
        "modelManifestEntry": { "family": "ltx-custom" }
    }));
    assert_eq!(resolve_family(&manifest), "ltx-custom");
    assert_eq!(
        resolve_catalog_video_family(&manifest.model, &manifest.model_manifest_entry),
        resolve_family(&manifest),
        "admission and asset planning must use the exact same custom-family policy"
    );
    let ltx = request(json!({ "projectId": "p", "model": "ltx_2_3" }));
    assert_eq!(resolve_family(&ltx), "ltx-video");
    assert_eq!(
        resolve_catalog_video_family(&ltx.model, &ltx.model_manifest_entry),
        resolve_family(&ltx),
        "admission and asset planning must use the exact same LTX fallback"
    );
    let wan = request(json!({ "projectId": "p", "model": "wan_2_2_t2v_14b" }));
    assert_eq!(resolve_family(&wan), "wan-video");
    let other = request(json!({ "projectId": "p", "model": "mystery" }));
    assert_eq!(resolve_family(&other), "video");
}

#[test]
fn resolve_seed_prefers_explicit_then_hashes_prompt() {
    let explicit = request(json!({ "projectId": "p", "seed": 123 }));
    assert_eq!(resolve_video_seed(&explicit), 123);
    // No seed → deterministic from the prompt (re-run reproduces).
    let a = request(json!({ "projectId": "p", "prompt": "sunset" }));
    let b = request(json!({ "projectId": "p", "prompt": "sunset" }));
    assert_eq!(resolve_video_seed(&a), resolve_video_seed(&b));
    let c = request(json!({ "projectId": "p", "prompt": "sunrise" }));
    assert_ne!(resolve_video_seed(&a), resolve_video_seed(&c));
}

#[test]
fn stub_video_frames_have_correct_size_and_audio_split() {
    // LTX → audio present; the frame buffers are exactly width*height*3.
    let ltx = request(json!({
        "projectId": "p", "model": "ltx_2_3", "width": 256, "height": 256,
        "duration": 1.0, "fps": 9
    }));
    let decoded = generate_stub_video(&ltx, 7);
    assert_eq!(decoded.frames.len(), ltx.frame_count() as usize);
    assert_eq!(decoded.fps, 9);
    for frame in &decoded.frames {
        assert_eq!(frame.pixels.len(), 256 * 256 * 3);
    }
    let audio = decoded.audio.expect("LTX stub emits audio");
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());

    // Wan → no audio track (mirrors the engine).
    let wan = request(json!({
        "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 1.0, "fps": 16
    }));
    assert!(generate_stub_video(&wan, 7).audio.is_none());
}

#[test]
fn stub_frames_differ_across_time() {
    // The sweeping band makes frame 0 and a later frame differ (real motion).
    let request = request(json!({
        "projectId": "p", "model": "wan_2_2", "width": 256, "height": 64,
        "duration": 1.0, "fps": 16
    }));
    let decoded = generate_stub_video(&request, 3);
    assert!(decoded.frames.len() >= 2);
    assert_ne!(
        decoded.frames[0].pixels,
        decoded.frames[decoded.frames.len() - 1].pixels
    );
}

#[test]
fn wav_header_is_canonical_and_preserves_in_range_amplitude() {
    let audio = AudioTrack {
        samples: vec![0.0, 0.5, -0.25, 0.5],
        sample_rate: 48_000,
        channels: 1,
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("a.wav");
    write_wav_pcm16(&audio, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[36..40], b"data");
    // 4 mono 16-bit samples → 8 bytes of PCM, 44-byte header.
    assert_eq!(bytes.len(), 44 + 8);
    // In-range PCM retains its amplitude rather than being normalized to full scale.
    let first = i16::from_le_bytes([bytes[44], bytes[45]]);
    let peak = i16::from_le_bytes([bytes[46], bytes[47]]);
    let trough = i16::from_le_bytes([bytes[48], bytes[49]]);
    assert_eq!(first, 0);
    assert_eq!(peak, 16_384);
    assert_eq!(trough, -8_192);
}

#[test]
fn wav_normalizes_only_over_range_audio() {
    let audio = AudioTrack {
        samples: vec![2.0, -1.0],
        sample_rate: 48_000,
        channels: 1,
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("over-range.wav");
    write_wav_pcm16(&audio, &path).unwrap();
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(i16::from_le_bytes([bytes[44], bytes[45]]), i16::MAX);
    assert_eq!(i16::from_le_bytes([bytes[46], bytes[47]]), -16_384);
}

#[tokio::test]
async fn seedvr2_failed_mux_cleanup_removes_silent_and_partial_outputs() {
    let dir_guard = tempfile::tempdir().expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("upscaled.mp4");
    let mux_tmp = dir.join("upscaled.audiomux.mp4");
    let poster_path = media_path.with_extension("poster.jpg");
    std::fs::write(&media_path, b"silent output").unwrap();
    std::fs::write(&mux_tmp, b"partial mux").unwrap();
    std::fs::write(&poster_path, b"orphaned poster").unwrap();

    cleanup_failed_seedvr2_mux(&media_path, &mux_tmp).await;

    assert!(!media_path.exists());
    assert!(!mux_tmp.exists());
    assert!(!poster_path.exists());
}

#[test]
fn seedvr2_output_guard_covers_the_full_post_encode_interval() {
    let dir_guard = tempfile::tempdir().expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("upscaled.mp4");
    let mux_tmp = dir.join("upscaled.audiomux.mp4");
    let poster_path = media_path.with_extension("poster.jpg");
    std::fs::write(&media_path, b"silent output").unwrap();
    std::fs::write(&mux_tmp, b"partial mux").unwrap();
    std::fs::write(&poster_path, b"poster").unwrap();

    {
        let _guard = Seedvr2OutputGuard::new(&media_path, &mux_tmp);
        // Simulates any `?` after encode and before the completion update succeeds.
    }

    assert!(!media_path.exists());
    assert!(!mux_tmp.exists());
    assert!(!poster_path.exists());
}

#[test]
fn seedvr2_output_guard_preserves_published_outputs_after_disarm() {
    let dir_guard = tempfile::tempdir().expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("upscaled.mp4");
    let mux_tmp = dir.join("upscaled.audiomux.mp4");
    let poster_path = media_path.with_extension("poster.jpg");
    std::fs::write(&media_path, b"published output").unwrap();
    std::fs::write(&mux_tmp, b"completed mux").unwrap();
    std::fs::write(&poster_path, b"published poster").unwrap();

    {
        let mut guard = Seedvr2OutputGuard::new(&media_path, &mux_tmp);
        guard.disarm();
    }

    assert!(media_path.exists());
    assert!(mux_tmp.exists());
    assert!(poster_path.exists());
}

#[test]
fn seedvr2_video_upscale_fact_records_lineage_flags() {
    let source = include_str!("seedvr2.rs");
    for required in [
        "\"sourceAssetId\": req.source_asset_id",
        "\"parents\": [req.source_asset_id]",
        "\"extra\": {",
        "\"isUpscaled\": true",
        "\"upscaledFromAssetId\": req.source_asset_id",
        "\"factor\": factor",
        "\"engine\": \"seedvr2\"",
    ] {
        assert!(
            source.contains(required),
            "video upscale fact must contain {required}"
        );
    }
}

#[test]
fn silent_audio_does_not_divide_by_zero() {
    let audio = AudioTrack {
        samples: vec![0.0; 16],
        sample_rate: 48_000,
        channels: 1,
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("silent.wav");
    write_wav_pcm16(&audio, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes[44..].iter().all(|&b| b == 0));
}

#[test]
fn asset_fact_and_streaming_result_shape() {
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "A red fox",
        "duration": 4.0, "fps": 24, "width": 768, "height": 512,
        "sourceAssetId": "asset_src", "personTrackId": "track_1"
    }));
    let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
    // Drive the real stub arm and measure what it produced — the funnel's exact sequence.
    let decoded = generate_stub_video(&request, 42);
    let clip = EncodedClip::measure(&decoded);
    let fact = video_asset_fact(
        &plan,
        42,
        "procedural_video",
        stub_raw_settings(&request),
        None,
        clip,
    );
    // Exhaustive, mirroring the image lane's key sweep (`image_jobs/tests.rs`). The recipe is
    // the ONLY record of what a user asked for, so a field silently dropped here is a field
    // the replay path (sc-12324) cannot reproduce — which is exactly how `fitMode` and the
    // multi-source ids went missing until sc-12345. Add a key here when you add one to the
    // fact; a spot-check would let the next one through.
    for key in [
        "type",
        "assetId",
        "mediaPath",
        "mimeType",
        "width",
        "height",
        "duration",
        "fps",
        "quality",
        "family",
        "seed",
        "displayName",
        "createdAt",
        "mode",
        "model",
        "adapter",
        "prompt",
        "negativePrompt",
        "loras",
        "rawAdapterSettings",
        "sourceAssetId",
        "lastFrameAssetId",
        "sourceClipAssetId",
        "bridgeRightClipAssetId",
        "fitMode",
        "sourceClipAssetIds",
        "referenceAssetIds",
        "referenceClipAssetId",
        "characterId",
        "characterLookId",
        "personTrackId",
        "replacementMode",
        "timelineContext",
    ] {
        assert!(fact.get(key).is_some(), "fact missing key {key}");
    }
    assert_eq!(fact["type"], json!("video"));
    assert_eq!(fact["mimeType"], json!("video/mp4"));
    assert_eq!(fact["mediaPath"], json!(plan.media_rel));
    assert_eq!(fact["adapter"], json!("procedural_video"));
    assert_eq!(fact["seed"], json!(42));
    // sc-12371 — THE SPLIT. LTX snaps 4.0 s x 24 fps (96 raw) onto its 8k+1 lattice, so the clip
    // is NOT 4.0 s. The fact must carry BOTH truths, because they feed different consumers:
    //
    //   `duration`/`fps`              -> `recipe.normalizedSettings` -> "re-run this generation"
    //   `encodedDuration`/`encodedFps` -> `asset.file`               -> what the mp4 really is
    //
    // Collapsing them either way is a bug: measured-into-normalizedSettings replays a 4 s ask as
    // 4.04 s (off the model's `limits.durations` menu, which sc-12347 now enforces), and
    // requested-into-file is the sc-12371 lie this story was filed for.
    let rendered = ltx_frame_count(96) as usize;
    assert_eq!(decoded.frames.len(), rendered);
    assert_ne!(
        rendered, 96,
        "the probe must discriminate: LTX has to actually snap 96 here, or requested and \
             measured agree and this pins nothing"
    );
    // The REPLAY knobs: exactly what the user picked, untouched.
    assert_eq!(fact["duration"], json!(4.0));
    assert_eq!(fact["fps"], json!(24));
    // The FILE facts: measured off the clip.
    assert_eq!(fact["encodedFrameCount"], json!(rendered));
    assert_eq!(fact["encodedFps"], json!(24));
    assert_eq!(fact["rawAdapterSettings"]["frameCount"], json!(rendered));
    assert!(
        (fact["encodedDuration"].as_f64().expect("encodedDuration") - rendered as f64 / 24.0).abs()
            < 1e-9,
        "encodedDuration must be the encoded frames over the encoded fps"
    );
    assert_ne!(
        fact["encodedDuration"],
        json!(4.0),
        "the REQUESTED 4.0 s is not the file's duration — if this ever matches, a \
             `request.duration` regression in the file block would pass unnoticed"
    );
    assert_eq!(fact["sourceAssetId"], json!("asset_src"));
    assert_eq!(fact["personTrackId"], json!("track_1"));
    assert_eq!(fact["displayName"], json!("A red fox"));
    assert_eq!(fact["rawAdapterSettings"]["stub"], json!(true));

    let result = streaming_result(&plan, &fact, "procedural_video");
    assert_eq!(result["expectedCount"], json!(1));
    assert_eq!(result["adapter"], json!("procedural_video"));
    assert_eq!(result["assetWrites"].as_array().unwrap().len(), 1);
    assert_eq!(result["generationSet"]["count"], json!(1));
}

/// The fit and the list-valued source ids reach the fact (sc-12345). These arrive as
/// TOP-LEVEL payload fields, so the `advanced.clone()` every real `*_raw_settings` builder
/// starts with does not carry them — `video_asset_fact` is their only path onto the recipe.
/// ads2v is the densest mode: source clip + reference clip + subject references at once.
#[test]
fn asset_fact_records_fit_and_multi_source_ids() {
    let ads2v = request(json!({
        "projectId": "p", "model": "bernini_2", "mode": "ads2v",
        "prompt": "the hero drives past",
        "sourceClipAssetId": "clip_main",
        "referenceClipAssetId": "clip_ref",
        "referenceAssetIds": ["ref_1", "ref_2"],
        "fitMode": "pad",
    }));
    let plan = VideoPlan::new(&ads2v, Path::new("/tmp/project"));
    // The clip these ids ride on is irrelevant to them, but it is not optional: `video_asset_fact`
    // requires the MEASURED clip so no caller can record a predicted length (sc-12371).
    let clip = EncodedClip {
        frames: 81,
        fps: 16,
    };
    let fact = video_asset_fact(&plan, 5, "mlx_bernini", json!({}), None, clip);
    assert_eq!(fact["referenceClipAssetId"], json!("clip_ref"));
    assert_eq!(fact["referenceAssetIds"], json!(["ref_1", "ref_2"]));
    assert_eq!(fact["fitMode"], json!("pad"));

    // mv2v carries the clip array instead; the other list stays empty rather than absent.
    let mv2v = request(json!({
        "projectId": "p", "model": "bernini_2", "mode": "multi_video_to_video",
        "prompt": "stitch them", "sourceClipAssetIds": ["clip_a", "clip_b"],
    }));
    let mv2v_plan = VideoPlan::new(&mv2v, Path::new("/tmp/project"));
    let mv2v_fact = video_asset_fact(&mv2v_plan, 5, "mlx_bernini", json!({}), None, clip);
    assert_eq!(mv2v_fact["sourceClipAssetIds"], json!(["clip_a", "clip_b"]));
    assert_eq!(mv2v_fact["referenceAssetIds"], json!([]));
}

/// A replace_person asset fact carries the `replacementStatus` object the API folds into
/// the video sidecar (sc-3521); a non-replace fact omits it.
#[test]
fn asset_fact_embeds_replacement_status_when_present() {
    let request = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "replace_person",
        "prompt": "swap the hero", "personTrackId": "track_9"
    }));
    let plan = VideoPlan::new(&request, Path::new("/tmp/project"));
    let status = json!({ "replacementActive": true, "maskMode": "segmentation" });
    let clip = EncodedClip {
        frames: 81,
        fps: 16,
    };
    let fact = video_asset_fact(&plan, 7, "mlx_wan_vace", json!({}), Some(status), clip);
    assert_eq!(fact["replacementStatus"]["replacementActive"], json!(true));
    assert_eq!(fact["replacementStatus"]["maskMode"], json!("segmentation"));
    // Without a status the key is absent (the non-replace paths).
    let bare = video_asset_fact(&plan, 7, "mlx_wan", json!({}), None, clip);
    assert!(bare.get("replacementStatus").is_none());
}

/// SceneWorks `replacementMode` strings → engine `ReplacementMode` (default FaceOnly).
#[cfg(target_os = "macos")]
#[test]
fn replacement_mode_maps_the_three_granularities() {
    assert_eq!(
        replacement_mode_from("face_only"),
        ReplacementMode::FaceOnly
    );
    assert_eq!(
        replacement_mode_from("full_person_keep_outfit"),
        ReplacementMode::FullPersonKeepOutfit
    );
    assert_eq!(
        replacement_mode_from("full_person_replace_outfit"),
        ReplacementMode::FullPersonReplaceOutfit
    );
    assert_eq!(replacement_mode_from("nonsense"), ReplacementMode::FaceOnly);
}

/// SCAIL-2 maps the SceneWorks video mode to the engine `video_mode` task: cross-identity
/// `replace_person` → "replacement" (engine flips `replace_flag`), everything else (standalone
/// `animate_character`) → "animation" (sc-5448 / sc-5452).
#[cfg(target_os = "macos")]
#[test]
fn scail2_mode_maps_replacement_vs_animation() {
    assert_eq!(scail2_engine_video_mode("replace_person"), "replacement");
    assert_eq!(scail2_engine_video_mode("animate_character"), "animation");
    assert_eq!(scail2_engine_video_mode("text_to_video"), "animation");
}

/// The replacement status records the actual engine adapter, so a SCAIL-2-backed person-replace
/// (sc-5452) reports `mlx_scail2` (not the Wan-VACE default).
#[cfg(target_os = "macos")]
#[test]
fn replacement_status_records_scail2_adapter() {
    let track = json!({ "id": "trk_1", "status": { "maskState": "active" } });
    let status =
        replacement_status_value(&track, "trk_1", "segmentation", 1.0, 1, 81, SCAIL2_ADAPTER);
    assert_eq!(status["replacementAdapter"], json!("mlx_scail2"));
    assert_eq!(status["replacementActive"], json!(true));
    assert_eq!(status["controlFrameCount"], json!(81));
}

/// The Wan-VACE conditioning is one ControlClip (frames + per-frame mask) followed by one
/// Reference per character image; mismatched frame/mask counts fail clearly.
#[cfg(target_os = "macos")]
#[test]
fn vace_conditioning_builds_control_clip_plus_references() {
    let frame = |v: u8| Image {
        width: 2,
        height: 2,
        pixels: vec![v; 12],
    };
    let mask = || image::RgbImage::from_pixel(2, 2, image::Rgb([255, 255, 255]));
    let conditioning = build_vace_conditioning(
        vec![frame(10), frame(20)],
        vec![mask(), mask()],
        vec![frame(30)],
        0.75,
        ReplacementMode::FullPersonKeepOutfit,
    )
    .expect("conditioning builds");
    assert_eq!(conditioning.len(), 2); // 1 ControlClip + 1 Reference
    match &conditioning[0] {
        Conditioning::ControlClip {
            frames,
            mask,
            masking_strength,
            start_frame,
            mode,
        } => {
            assert_eq!(frames.len(), 2);
            assert_eq!(mask.len(), 2);
            assert_eq!(*masking_strength, 0.75);
            assert_eq!(*start_frame, 0);
            assert_eq!(*mode, ReplacementMode::FullPersonKeepOutfit);
        }
        other => panic!("expected ControlClip, got {other:?}"),
    }
    assert!(matches!(conditioning[1], Conditioning::Reference { .. }));
    // A frame/mask count mismatch is rejected.
    assert!(build_vace_conditioning(
        vec![frame(1)],
        vec![mask(), mask()],
        Vec::new(),
        1.0,
        ReplacementMode::FaceOnly,
    )
    .is_err());
}

/// sc-3812 extend: the ControlClip pins the real tail frames at the front (mask black = keep)
/// and fills the rest of the budget with a neutral-gray generated span (mask white), no refs.
#[cfg(target_os = "macos")]
#[test]
fn extend_vace_conditioning_pins_tail_and_generates_rest() {
    let anchor = |v: u8| Image {
        width: 2,
        height: 2,
        pixels: vec![v; 12],
    };
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
        "prompt": "keep walking", "sourceClipAssetId": "clip_a"
    }));
    let conditioning =
        build_extend_bridge_vace_conditioning(&req, 2, 2, 5, vec![anchor(11), anchor(22)], None)
            .expect("extend conditioning builds");
    assert_eq!(conditioning.len(), 1); // ControlClip only, no Reference
    match &conditioning[0] {
        Conditioning::ControlClip { frames, mask, .. } => {
            assert_eq!(frames.len(), 5);
            assert_eq!(mask.len(), 5);
            // First two are the real tail frames, kept (black mask).
            assert_eq!(frames[0].pixels[0], 11);
            assert_eq!(frames[1].pixels[0], 22);
            assert_eq!(mask[0].pixels[0], 0);
            assert_eq!(mask[1].pixels[0], 0);
            // The rest is the neutral-gray generated span (white mask).
            assert_eq!(frames[2].pixels[0], 128);
            assert_eq!(frames[4].pixels[0], 128);
            assert_eq!(mask[2].pixels[0], 255);
            assert_eq!(mask[4].pixels[0], 255);
        }
        other => panic!("expected ControlClip, got {other:?}"),
    }
}

/// sc-3812 bridge: both clips' boundary anchors are kept at the two ends; the gap between them
/// is the generated span. A missing right clip / over-budget anchors fail clearly.
#[cfg(target_os = "macos")]
#[test]
fn bridge_vace_conditioning_keeps_both_ends_generates_gap() {
    let anchor = |v: u8| Image {
        width: 1,
        height: 1,
        pixels: vec![v; 3],
    };
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "video_bridge",
        "prompt": "connect", "sourceClipAssetId": "left", "bridgeRightClipAssetId": "right"
    }));
    let conditioning = build_extend_bridge_vace_conditioning(
        &req,
        1,
        1,
        5,
        vec![anchor(10)],
        Some(vec![anchor(90)]),
    )
    .expect("bridge conditioning builds");
    match &conditioning[0] {
        Conditioning::ControlClip { frames, mask, .. } => {
            assert_eq!(frames.len(), 5);
            // Left end kept, gap generated, right end kept.
            assert_eq!((frames[0].pixels[0], mask[0].pixels[0]), (10, 0));
            assert_eq!((frames[1].pixels[0], mask[1].pixels[0]), (128, 255));
            assert_eq!((frames[3].pixels[0], mask[3].pixels[0]), (128, 255));
            assert_eq!((frames[4].pixels[0], mask[4].pixels[0]), (90, 0));
        }
        other => panic!("expected ControlClip, got {other:?}"),
    }
    // video_bridge without a right clip is rejected.
    assert!(build_extend_bridge_vace_conditioning(&req, 1, 1, 5, vec![anchor(10)], None).is_err());
    // Anchors that leave no gap are rejected.
    assert!(build_extend_bridge_vace_conditioning(
        &req,
        1,
        1,
        5,
        vec![anchor(1), anchor(2), anchor(3)],
        Some(vec![anchor(4), anchor(5)]),
    )
    .is_err());
}

/// sc-3812 motion anchor: defaults to ~⅓ of the budget (halved per side for bridge), honors an
/// explicit `motionAnchorFrames`, and always clamps so ≥5 frames stay generatable.
#[cfg(target_os = "macos")]
#[test]
fn extend_anchor_frames_defaults_and_clamps() {
    let extend = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x"
    }));
    assert_eq!(extend_anchor_frames(&extend, 81), 27); // 81/3
    let bridge = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "video_bridge", "prompt": "x"
    }));
    assert_eq!(extend_anchor_frames(&bridge, 81), 13); // (81/3)/2

    let explicit = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
        "advanced": { "motionAnchorFrames": 4 }
    }));
    assert_eq!(extend_anchor_frames(&explicit, 81), 4);

    // Over-budget request clamps so 5 frames remain to generate (81 - 5 = 76).
    let greedy = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "extend_clip", "prompt": "x",
        "advanced": { "motionAnchorFrames": 999 }
    }));
    assert_eq!(extend_anchor_frames(&greedy, 81), 76);
    // Minimum-length clip still yields a usable anchor.
    assert_eq!(extend_anchor_frames(&extend, 5), 1);
}

/// `replacement_status_value` reports the honest mask/track provenance the sidecar folds in.
#[cfg(target_os = "macos")]
#[test]
fn replacement_status_reads_track_and_counts() {
    let track = json!({
        "id": "track_42",
        "status": { "maskState": "active", "personTrackingActive": true },
        "corrections": [ { "frameIndex": 0 }, { "frameIndex": 3 } ]
    });
    // 0.5 is exactly representable as f32 so the JSON widen to f64 is exact.
    let status = replacement_status_value(
        &track,
        "ignored",
        "segmentation",
        0.5,
        2,
        81,
        WAN_VACE_ADAPTER,
    );
    assert_eq!(status["personDetectionActive"], json!(true));
    assert_eq!(status["personTrackingActive"], json!(true));
    assert_eq!(status["replacementActive"], json!(true));
    assert_eq!(status["replacementAdapter"], json!("mlx_wan_vace"));
    assert_eq!(status["maskMode"], json!("segmentation"));
    assert_eq!(status["maskState"], json!("active"));
    assert_eq!(status["maskingStrength"], json!(0.5));
    assert_eq!(status["personTrackId"], json!("track_42"));
    assert_eq!(status["characterReferenceCount"], json!(2));
    assert_eq!(status["controlFrameCount"], json!(81));
    assert_eq!(status["usedCorrections"], json!(true));
    assert_eq!(status["correctionCount"], json!(2));
}

/// An assembled Wan-VACE snapshot dir if one is present (env override or the app-managed
/// default), else `None` so the real-weight smoke skips.
#[cfg(target_os = "macos")]
fn wan_vace_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
        let path = PathBuf::from(dir.trim());
        if wan_vace_dir_is_complete(&path) {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let path =
        PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx/wan_vace");
    wan_vace_dir_is_complete(&path).then_some(path)
}

/// Real in-process Wan-VACE replace_person through the engine: load the assembled snapshot
/// and denoise a tiny 5-frame clip from a synthetic control clip (gray frames + a centered
/// box mask) + one reference, asserting frames come back RGB8-sized with streamed progress.
/// `#[ignore]` — the weights live outside CI; run manually on a Mac where the snapshot is
/// assembled (the real-Mac GPU parity gate, sc-3521).
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
#[test]
fn wan_vace_real_weights() {
    let Some(model_dir) = wan_vace_dir() else {
        eprintln!("skipping wan_vace_real_weights: no assembled wan_vace snapshot found");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let gray = || Image {
        width: w,
        height: h,
        pixels: vec![118u8; (w * h * 3) as usize],
    };
    let frames: Vec<Image> = (0..5).map(|_| gray()).collect();
    let masks: Vec<image::RgbImage> = (0..5)
        .map(|_| {
            crate::person_replace::box_mask(
                Some(&json!({ "x": 0.3, "y": 0.2, "width": 0.4, "height": 0.6 })),
                w,
                h,
            )
        })
        .collect();
    let conditioning =
        build_vace_conditioning(frames, masks, vec![gray()], 1.0, ReplacementMode::FaceOnly)
            .expect("conditioning builds");
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: "a person walking, cinematic".to_owned(),
        width: w,
        height: h,
        frames: 5,
        fps: 16,
        steps: Some(8),
        seed: 7,
        control_scale: Some(1.0),
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded = run_video_generation(input, &cancel, &mut on_progress).expect("VACE generation");
    assert!(decoded.fps >= 1);
    assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
    assert!(steps > 0, "denoise progress streamed");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// A Bernini MLX snapshot dir if present: env override (`SCENEWORKS_MLX_BERNINI_DIR`), then the
/// bring-up staging / cache dirs, then the app-managed default. `None` ⇒ the smoke skips.
#[cfg(target_os = "macos")]
fn bernini_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_BERNINI_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    for rel in [
        ".cache/mlx-gen-models/bernini-mlx-upload",
        ".cache/mlx-gen-models/bernini_full_mlx_bf16",
        "Library/Application Support/SceneWorks/data/models/mlx/bernini",
    ] {
        let path = PathBuf::from(&home).join(rel);
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    None
}

/// Test-only override for the quant tier a Bernini smoke loads (sc-4709): `SCENEWORKS_BERNINI_
/// SMOKE_QUANT` = `8` → Q8, `0` → bf16 (no quant), anything else / unset → Q4 (the committed
/// default). Lets the manual smokes profile peak memory at either tier without changing the
/// product default (e.g. capturing the Q8 video peak this story tracks).
#[cfg(target_os = "macos")]
fn bernini_smoke_quant() -> Option<Quant> {
    match std::env::var("SCENEWORKS_BERNINI_SMOKE_QUANT")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
    {
        Some(bits) if bits <= 0 => None,
        Some(bits) if bits > 4 => Some(Quant::Q8),
        _ => Some(Quant::Q4),
    }
}

/// A test-only `usize`/`u32` env override, falling back to `default` when unset/unparseable.
#[cfg(target_os = "macos")]
fn bernini_smoke_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Real in-process Bernini text-to-video through the WORKER registry path (epic 4699 / sc-4707):
/// `crate::inference_runtime::load("bernini")` (proves runtime-macos includes Bernini — the
/// "no generator registered" trap) → Q4 → a tiny t2v clip, asserting RGB8 frames
/// stream back with denoise progress. Also confirms the lean published `SceneWorks/bernini-mlx`
/// snapshot loads. `#[ignore]` — weights live outside CI; run on a Mac with the snapshot present.
/// The quant tier (`SCENEWORKS_BERNINI_SMOKE_QUANT`) and dims (`..._W`/`_H`/`_FRAMES`/`_STEPS`)
/// are env-overridable for memory profiling (sc-4709 Q8 peak); defaults reproduce the Q4 run.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
#[test]
fn bernini_t2v_real_weights() {
    let Some(model_dir) = bernini_dir() else {
        eprintln!("skipping bernini_t2v_real_weights: no Bernini MLX snapshot found");
        return;
    };
    let w = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_W", 832);
    let h = bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_H", 480);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "bernini",
        model_dir,
        quant: bernini_smoke_quant(),
        prompt: "a golden retriever puppy running across a sunlit meadow, cinematic".to_owned(),
        width: w,
        height: h,
        frames: bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_FRAMES", 17),
        fps: 16,
        steps: Some(bernini_smoke_u32("SCENEWORKS_BERNINI_SMOKE_STEPS", 20)),
        seed: 7,
        video_mode: Some("t2v".to_owned()),
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded =
        run_video_generation(input, &cancel, &mut on_progress).expect("bernini t2v generation");
    assert!(decoded.fps >= 1);
    assert!(steps > 0, "denoise progress streamed");
    assert!(!decoded.frames.is_empty(), "frames returned");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// The SceneWorks mode → engine guidance-task mapping (sc-4703). Pure; runs in CI on Mac.
#[cfg(target_os = "macos")]
#[test]
fn bernini_engine_video_mode_maps_each_sceneworks_mode() {
    assert_eq!(bernini_engine_video_mode("text_to_video"), "t2v");
    assert_eq!(bernini_engine_video_mode("video_to_video"), "v2v");
    assert_eq!(bernini_engine_video_mode("reference_to_video"), "r2v");
    assert_eq!(
        bernini_engine_video_mode("reference_video_to_video"),
        "rv2v"
    );
    // Multi-source modes (sc-5425).
    assert_eq!(bernini_engine_video_mode("multi_video_to_video"), "mv2v");
    assert_eq!(bernini_engine_video_mode("ads2v"), "ads2v");
    // Unknown / unset falls back to plain text-to-video.
    assert_eq!(bernini_engine_video_mode("image_to_video"), "t2v");
    assert_eq!(bernini_engine_video_mode(""), "t2v");
}

/// Only the `bernini` catalog id routes to the Bernini engine (sc-4707). Other video
/// families (Wan/LTX/SVD/image-typed ids) are `None` so they keep their own routing.
#[cfg(target_os = "macos")]
#[test]
fn bernini_engine_id_maps_only_the_bernini_family() {
    assert_eq!(bernini_engine_id("bernini"), Some("bernini"));
    assert_eq!(bernini_engine_id("wan_2_2"), None);
    assert_eq!(bernini_engine_id("wan_2_2_t2v_14b"), None);
    assert_eq!(bernini_engine_id("ltx_2_3"), None);
    assert_eq!(bernini_engine_id("z_image_turbo"), None);
    assert_eq!(bernini_engine_id(""), None);
}

/// Bernini load quantization (sc-4709): Q4 is the default (the validated 64 GB-fitting tier),
/// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), and the control parses from a JSON
/// number or a string. The snapshot is ~93 GB at bf16 so a missing control NEVER means bf16.
#[cfg(target_os = "macos")]
#[test]
fn bernini_resolve_quant_defaults_q4_and_honors_override() {
    let quant = |advanced: Value| {
        resolve_bernini_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
    };
    // No control → Q4 default (never bf16).
    assert!(matches!(quant(json!({})), Some(Quant::Q4)));
    // Explicit tiers (number).
    assert!(matches!(
        quant(json!({ "mlxQuantize": 4 })),
        Some(Quant::Q4)
    ));
    assert!(matches!(
        quant(json!({ "mlxQuantize": 8 })),
        Some(Quant::Q8)
    ));
    // `<= 0` opts into bf16 (no quantization) for power users with ample RAM.
    assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
    assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
    // String forms parse the same (the advanced map can carry stringly-typed values).
    assert!(matches!(
        quant(json!({ "mlxQuantize": "8" })),
        Some(Quant::Q8)
    ));
    assert!(matches!(
        quant(json!({ "mlxQuantize": " 4 " })),
        Some(Quant::Q4)
    ));
    assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
    // A bits value between 4 and 8 rounds to the nearest supported tier (> 4 ⇒ Q8).
    assert!(matches!(
        quant(json!({ "mlxQuantize": 6 })),
        Some(Quant::Q8)
    ));
}

/// Raw-settings lineage captured on a real Bernini asset (sc-4709): the real-inference marker,
/// the catalog model id, the produced frame count / fps, the resolved engine guidance task
/// (`berniniTask`), and a pass-through of the user's advanced controls.
#[cfg(target_os = "macos")]
#[test]
fn bernini_raw_settings_capture_lineage_and_task() {
    let req = request(json!({
        "projectId": "p",
        "model": "bernini",
        "mode": "reference_video_to_video",
        "duration": 5,
        "fps": 16,
        "advanced": { "mlxQuantize": 8, "userKnob": "keep-me" }
    }));
    let raw = bernini_raw_settings(&req);
    let raw = raw.as_object().expect("raw settings is an object");
    assert_eq!(raw["realModelInference"], json!(true));
    assert_eq!(raw["model"], json!("bernini"));
    assert_eq!(raw["fps"], json!(req.fps));
    // No frameCount: this builder is exactly where the sc-12371 bug lived (it recorded
    // `request.frame_count()` = the raw 150 while `generate_bernini` rendered 149). The clip's
    // length is now stamped from the encoded frames at the funnel — see
    // `no_raw_settings_builder_records_its_own_frame_count`.
    assert!(raw.get("frameCount").is_none());
    // The SceneWorks mode resolved to its engine guidance task for observability/lineage.
    assert_eq!(raw["berniniTask"], json!("rv2v"));
    // The user's advanced controls survive verbatim (provenance).
    assert_eq!(raw["userKnob"], json!("keep-me"));
    assert_eq!(raw["mlxQuantize"], json!(8));
}

/// Bernini conditioning resolution enforces each editing/reference mode's required media
/// (sc-4703 / sc-4709), failing loudly BEFORE any IO when it is missing — defense in depth
/// behind the API-side validation. `text_to_video` needs none and resolves to empty
/// conditioning. The guards fire before touching the api / job / disk, so a minimal snapshot
/// suffices (mirrors `video_clip_conditioning_requires_ic_lora`).
#[cfg(target_os = "macos")]
#[tokio::test]
async fn bernini_conditioning_enforces_required_media() {
    // Offline: this builds a real `ApiClient`, and the guards under test are all that stand
    // between it and the network. Pinned so a regression that moves a guard AFTER the I/O fails
    // locally and loudly instead of dialing the real API/hub (sc-12380).
    let settings = offline_settings();
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-bernini-1",
        "type": "video_generate",
        "status": "preparing",
        "projectId": "p",
        "projectName": "P",
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": null,
        "progress": 0,
        "stage": "preparing",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-14T00:00:00Z",
        "updatedAt": "2026-06-14T00:00:00Z"
    }))
    .expect("job snapshot");
    let resolve = |mode: &str, extra: Value| {
        let mut payload = json!({ "projectId": "p", "model": "bernini", "prompt": "go" });
        payload
            .as_object_mut()
            .unwrap()
            .insert("mode".to_owned(), json!(mode));
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            payload.as_object_mut().unwrap().insert(k, v);
        }
        let req = request(payload);
        let api = &api;
        let settings = &settings;
        let job = &job;
        async move { resolve_bernini_conditioning(api, settings, job, &req, Path::new("/tmp/p")).await }
    };

    // text_to_video needs no source media → empty conditioning, no IO.
    let t2v = resolve("text_to_video", json!({}))
        .await
        .expect("t2v resolves");
    assert!(t2v.is_empty(), "t2v needs no conditioning");

    // video_to_video / reference_video_to_video require a source clip.
    let v2v_err = resolve("video_to_video", json!({}))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        v2v_err.contains("source clip"),
        "v2v missing-clip error: {v2v_err}"
    );
    let rv2v_err = resolve("reference_video_to_video", json!({}))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        rv2v_err.contains("source clip"),
        "rv2v missing-clip error: {rv2v_err}"
    );

    // reference_to_video requires at least one reference image.
    let r2v_err = resolve("reference_to_video", json!({}))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        r2v_err.contains("reference image"),
        "r2v missing-refs error: {r2v_err}"
    );
}

/// Real in-process Bernini editing/reference/multi-source video modes through the engine
/// (sc-4703 / sc-4709 / sc-5425): drive v2v (synthetic source clip → `VideoClip`), r2v
/// (synthetic reference → `MultiReference`), rv2v (both), mv2v (two source clips), and ads2v
/// (source clip + reference clip + reference), asserting each mode loads, consumes its
/// conditioning, and streams RGB8 frames. `#[ignore]` — weights live outside CI; run on a Mac
/// with the `SceneWorks/bernini-mlx` snapshot.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
#[test]
fn bernini_editing_reference_modes_real_weights() {
    let Some(model_dir) = bernini_dir() else {
        eprintln!("skipping bernini_editing_reference_modes_real_weights: no snapshot found");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let frame = |shade: u8| Image {
        width: w,
        height: h,
        pixels: vec![shade; (w * h * 3) as usize],
    };
    // Two 5-frame synthetic source clips and one synthetic subject reference.
    let clip: Vec<Image> = (0..5).map(|i| frame(60 + i * 12)).collect();
    let clip_b: Vec<Image> = (0..5).map(|i| frame(40 + i * 16)).collect();
    let reference = frame(150);
    let video_clip = |frames: Vec<Image>| Conditioning::VideoClip {
        frames,
        frame_idx: 0,
        strength: 1.0,
    };

    let cases: &[(&str, Vec<Conditioning>)] = &[
        ("v2v", vec![video_clip(clip.clone())]),
        (
            "r2v",
            vec![Conditioning::MultiReference {
                images: vec![reference.clone()],
            }],
        ),
        (
            "rv2v",
            vec![
                video_clip(clip.clone()),
                Conditioning::MultiReference {
                    images: vec![reference.clone()],
                },
            ],
        ),
        // mv2v: two source clips, no references.
        (
            "mv2v",
            vec![video_clip(clip.clone()), video_clip(clip_b.clone())],
        ),
        // ads2v: source clip + reference clip + reference image (videos first, then images).
        (
            "ads2v",
            vec![
                video_clip(clip.clone()),
                video_clip(clip_b.clone()),
                Conditioning::MultiReference {
                    images: vec![reference.clone()],
                },
            ],
        ),
    ];

    for (task, conditioning) in cases {
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "bernini",
            model_dir: model_dir.clone(),
            quant: Some(Quant::Q4),
            conditioning: conditioning.clone(),
            prompt: "the subject walks through a neon-lit street, cinematic".to_owned(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 11,
            video_mode: Some((*task).to_owned()),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|error| panic!("bernini {task} generation: {error}"));
        assert!(steps > 0, "{task}: denoise progress streamed");
        assert!(!decoded.frames.is_empty(), "{task}: frames returned");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{task}: frames are RGB8-sized"
        );
    }
}

// ===================== SCAIL-2 validation (epic 5439 / sc-5450) =====================

/// Only the `scail2_14b` catalog id routes to the SCAIL-2 engine (sc-5448). Every other video
/// family (Wan/LTX/SVD/Bernini/image ids) is `None`, so they keep their own routing.
#[cfg(target_os = "macos")]
#[test]
fn scail2_engine_id_maps_only_the_scail2_family() {
    assert_eq!(scail2_engine_id("scail2_14b"), Some("scail2_14b"));
    assert_eq!(scail2_engine_id("bernini"), None);
    assert_eq!(scail2_engine_id("wan_2_2"), None);
    assert_eq!(scail2_engine_id("ltx_2_3"), None);
    assert_eq!(scail2_engine_id("scail2"), None);
    assert_eq!(scail2_engine_id(""), None);
}

/// SCAIL-2 load quantization (sc-5450): Q4 is the default (the validated ~16 GB tier),
/// `mlxQuantize` opts up to Q8 or down to bf16 (`<= 0`), parsing a JSON number or string. The
/// bf16 snapshot is ~47 GB so a missing control NEVER means bf16. Mirrors the Bernini quant test.
#[cfg(target_os = "macos")]
#[test]
fn scail2_resolve_quant_defaults_q4_and_honors_override() {
    let quant = |advanced: Value| {
        resolve_scail2_quant(&request(json!({ "projectId": "p", "advanced": advanced })))
    };
    assert!(matches!(quant(json!({})), Some(Quant::Q4)));
    assert!(matches!(
        quant(json!({ "mlxQuantize": 4 })),
        Some(Quant::Q4)
    ));
    assert!(matches!(
        quant(json!({ "mlxQuantize": 8 })),
        Some(Quant::Q8)
    ));
    assert!(quant(json!({ "mlxQuantize": 0 })).is_none());
    assert!(quant(json!({ "mlxQuantize": -1 })).is_none());
    // String forms parse the same; a tier between 4 and 8 rounds up to Q8.
    assert!(matches!(
        quant(json!({ "mlxQuantize": "8" })),
        Some(Quant::Q8)
    ));
    assert!(matches!(
        quant(json!({ "mlxQuantize": " 4 " })),
        Some(Quant::Q4)
    ));
    assert!(quant(json!({ "mlxQuantize": "0" })).is_none());
    assert!(matches!(
        quant(json!({ "mlxQuantize": 6 })),
        Some(Quant::Q8)
    ));
}

/// Raw-settings lineage on a real SCAIL-2 asset (sc-5450): the real-inference marker, the catalog
/// model id, the produced frame count / fps, the resolved engine task (`scail2Task`), and a
/// pass-through of the user's advanced controls. `replace_person` → "replacement",
/// `animate_character` → "animation".
#[cfg(target_os = "macos")]
#[test]
fn scail2_raw_settings_capture_lineage_and_task() {
    let raw_for = |mode: &str| {
        let req = request(json!({
            "projectId": "p",
            "model": "scail2_14b",
            "mode": mode,
            "duration": 5,
            "fps": 16,
            "advanced": { "mlxQuantize": 4, "userKnob": "keep-me" }
        }));
        let raw = scail2_raw_settings(&req, false);
        (raw, req)
    };
    let (raw, req) = raw_for("animate_character");
    let raw = raw.as_object().expect("raw settings is an object");
    assert_eq!(raw["realModelInference"], json!(true));
    assert_eq!(raw["model"], json!("scail2_14b"));
    assert_eq!(raw["fps"], json!(req.fps));
    // No frameCount — the other half of the sc-12371 bug (recorded the raw count while
    // `generate_scail2` rendered the Wan stride). Stamped at the funnel now.
    assert!(raw.get("frameCount").is_none());
    assert_eq!(raw["scail2Task"], json!("animation"));
    assert_eq!(raw["userKnob"], json!("keep-me"));
    // No lightning LoRA ⇒ no effective-recipe override recorded (engine quality defaults stand).
    assert!(raw.get("scail2Lightning").is_none());
    assert!(raw.get("effectiveSteps").is_none());
    // replace_person resolves to the replacement task (flips the engine replace_flag).
    let (raw_replace, _) = raw_for("replace_person");
    assert_eq!(raw_replace["scail2Task"], json!("replacement"));
}

/// The lightx2v lightning toggle (sc-5700): selecting a diff-patch LoRA applies the step-distill
/// recipe — 8 steps, CFG off (guidance 1.0), scheduler shift 1.0 — with an explicit user step
/// count still honored, and the effective recipe is recorded on the asset. Without lightning the
/// sampling is all-`None` (the engine's quality defaults stand, unchanged).
#[cfg(target_os = "macos")]
#[test]
fn scail2_lightning_recipe_overrides_sampling_and_records() {
    let req = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
        "duration": 5, "fps": 16, "advanced": {}
    }));
    // Non-lightning: untouched (engine defaults).
    assert_eq!(scail2_sampling(&req, false), (None, None, None));
    // Lightning: CFG off + shift 1.0 forced, steps default 8.
    assert_eq!(
        scail2_sampling(&req, true),
        (Some(8), Some(1.0), Some(1.0)),
        "lightning applies the 8-step CFG-off recipe"
    );
    // An explicit user step count is honored; CFG/shift stay forced.
    let req_steps = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "animate_character",
        "duration": 5, "fps": 16, "advanced": { "steps": 4 }
    }));
    assert_eq!(
        scail2_sampling(&req_steps, true),
        (Some(4), Some(1.0), Some(1.0))
    );
    // The effective recipe is recorded for observability when lightning is active.
    let raw = scail2_raw_settings(&req, true);
    let raw = raw.as_object().unwrap();
    assert_eq!(raw["scail2Lightning"], json!(true));
    assert_eq!(raw["effectiveSteps"], json!(8));
    assert_eq!(raw["effectiveGuidanceScale"], json!(1.0));
    assert_eq!(raw["effectiveSchedulerShift"], json!(1.0));
}

/// SCAIL-2 conditioning resolution fails loudly BEFORE any IO when its required media is missing
/// (sc-5450 — defense in depth behind the API-side validation): `animate_character` needs a
/// reference character image, and the integrated `replace_person` backend needs a person track.
/// Both guards fire before touching the api / job / disk, so a minimal job suffices.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn scail2_conditioning_guards_fire_before_io() {
    // "before_io" is the whole claim of this test — `offline_settings` is what makes a broken
    // claim fail locally rather than reach the real API/hub (sc-12380).
    let settings = offline_settings();
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-scail2-1",
        "type": "video_generate",
        "status": "preparing",
        "projectId": "p",
        "projectName": "P",
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": null,
        "progress": 0,
        "stage": "preparing",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-14T00:00:00Z",
        "updatedAt": "2026-06-14T00:00:00Z"
    }))
    .expect("job snapshot");

    // animate_character with no reference image / source asset → reference guard.
    let animate_req = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "animate_character", "prompt": "go"
    }));
    let animate_err =
        resolve_scail2_conditioning(&api, &settings, &job, &animate_req, Path::new("/tmp/p"))
            .await
            .unwrap_err()
            .to_string();
    assert!(
        animate_err.contains("reference character image"),
        "animate missing-reference error: {animate_err}"
    );

    // replace_person with no person track → person-track guard.
    let replace_req = request(json!({
        "projectId": "p", "model": "scail2_14b", "mode": "replace_person", "prompt": "go"
    }));
    let replace_err = resolve_scail2_replace_conditioning(
        &api,
        &settings,
        &job,
        &replace_req,
        Path::new("/tmp/p"),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        replace_err.contains("person track"),
        "replace missing-track error: {replace_err}"
    );
}

/// A SCAIL-2 MLX snapshot dir if present: env override (`SCENEWORKS_MLX_SCAIL2_DIR`), then the
/// bring-up convert/staging dirs, then the app-managed default. `None` ⇒ the smoke skips.
#[cfg(target_os = "macos")]
fn scail2_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SCAIL2_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    for rel in [
        ".cache/scail2-mlx-convert",
        ".cache/mlx-gen-models/scail2-mlx-upload",
        "Library/Application Support/SceneWorks/data/models/mlx/scail2",
    ] {
        let path = PathBuf::from(&home).join(rel);
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    None
}

/// A synthetic SCAIL-2 color-coded mask: a centered blue rectangle (person 0) on a solid `bg`,
/// matching the engine's 7-class color scheme (only the chromatic channels matter, not the exact
/// silhouette — the smoke proves the engine consumes the shape and streams frames, not parity).
#[cfg(target_os = "macos")]
fn scail2_smoke_color_mask(w: u32, h: u32, bg: [u8; 3]) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut px = vec![0u8; wu * hu * 3];
    for chunk in px.chunks_exact_mut(3) {
        chunk.copy_from_slice(&bg);
    }
    for y in (hu / 4)..(3 * hu / 4) {
        for x in (wu / 4)..(3 * wu / 4) {
            let o = (y * wu + x) * 3;
            px[o..o + 3].copy_from_slice(&crate::scail2_masks::PALETTE[0]);
        }
    }
    Image {
        width: w,
        height: h,
        pixels: px,
    }
}

/// Real in-process SCAIL-2 through the WORKER registry path (epic 5439 / sc-5450): drive both
/// engine tasks — `animation` (standalone `animate_character`) and `replacement` (cross-identity
/// `replace_person`) — with a synthetic reference + color mask + driving clip + per-frame color
/// masks, asserting each `video_mode` loads via `crate::inference_runtime::load("scail2_14b")`
/// (proving runtime-macos includes SCAIL-2 — the "no generator registered"
/// trap), consumes the conditioning, and streams RGB8 frames with denoise progress. The driving
/// background follows the replacement convention (animation → black, replacement → white).
/// `#[ignore]` — the ~47 GB snapshot lives outside CI; run manually on a Mac where it is present
/// (Q4 ≈ 16 GB peak). The bg conventions mirror `scail2_masks` (sc-5448 / sc-5452).
#[cfg(target_os = "macos")]
#[ignore = "loads the real SCAIL-2 snapshot (~47 GB); run manually on a Mac where the scail2 weights are present"]
#[test]
fn scail2_animation_and_replacement_real_weights() {
    let Some(model_dir) = scail2_dir() else {
        eprintln!("skipping scail2_animation_and_replacement_real_weights: no snapshot found");
        return;
    };
    let (w, h) = (256u32, 256u32);
    let frame = |shade: u8| Image {
        width: w,
        height: h,
        pixels: vec![shade; (w * h * 3) as usize],
    };
    // 5 driving frames → one clean VAE-aligned segment (the engine keeps ((T-1)/4)*4 + 1).
    let driving: Vec<Image> = (0..5).map(|i| frame(50 + i * 20)).collect();
    let reference = frame(160);

    // animation keeps the reference's world (driving bg black, ref bg white); cross-identity
    // replacement keeps the driving's world (driving bg white, ref bg black) — sc-5448 / sc-5452.
    for (task, driving_bg, reference_bg) in [
        (
            "animation",
            crate::scail2_masks::BG_BLACK,
            crate::scail2_masks::BG_WHITE,
        ),
        (
            "replacement",
            crate::scail2_masks::BG_WHITE,
            crate::scail2_masks::BG_BLACK,
        ),
    ] {
        let reference_mask = scail2_smoke_color_mask(w, h, reference_bg);
        let driving_masks: Vec<Image> = driving
            .iter()
            .map(|_| scail2_smoke_color_mask(w, h, driving_bg))
            .collect();
        let conditioning = vec![
            Conditioning::Reference {
                image: reference.clone(),
                strength: None,
            },
            Conditioning::Mask {
                image: reference_mask,
            },
            Conditioning::ControlClip {
                frames: driving.clone(),
                mask: driving_masks,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ];
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "scail2_14b",
            model_dir: model_dir.clone(),
            quant: Some(Quant::Q4),
            conditioning,
            prompt: "the character walks forward, cinematic".to_owned(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(8),
            seed: 11,
            video_mode: Some(task.to_owned()),
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|error| panic!("scail2 {task} generation: {error}"));
        assert!(steps > 0, "{task}: denoise progress streamed");
        assert!(!decoded.frames.is_empty(), "{task}: frames returned");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{task}: frames are RGB8-sized"
        );
    }
}

/// Real in-process Wan-VACE extend/bridge through the engine (sc-3812): build the tier-C
/// control clip (real anchor frames pinned + a neutral generated span, no references) and run
/// the assembled snapshot, asserting RGB8 frames stream back. `#[ignore]` — the weights live
/// outside CI; run manually on a Mac with the snapshot assembled (the real-Mac gate; the A/B
/// vs the TI2V-5B single-frame path is the practical fidelity judge, sc-3800).
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan-VACE snapshot; run manually on a Mac with it assembled"]
#[test]
fn wan_vace_extend_bridge_real_weights() {
    let Some(model_dir) = wan_vace_dir() else {
        eprintln!("skipping wan_vace_extend_bridge_real_weights: no assembled snapshot found");
        return;
    };
    let (w, h) = (256u32, 256u32);
    // Two distinct real "source" frames as the extend motion anchor; the engine generates the
    // remaining 3 frames of the 5-frame budget over the neutral span.
    let anchor = |v: u8| Image {
        width: w,
        height: h,
        pixels: vec![v; (w * h * 3) as usize],
    };
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "mode": "extend_clip",
        "prompt": "the camera keeps gliding forward, cinematic",
        "sourceClipAssetId": "clip_a"
    }));
    let conditioning =
        build_extend_bridge_vace_conditioning(&req, w, h, 5, vec![anchor(90), anchor(110)], None)
            .expect("extend conditioning builds");
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        conditioning,
        prompt: req.prompt.clone(),
        width: w,
        height: h,
        frames: 5,
        fps: 16,
        steps: Some(8),
        seed: 11,
        control_scale: Some(1.0),
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded =
        run_video_generation(input, &cancel, &mut on_progress).expect("VACE extend generation");
    assert_eq!(decoded.frames.len(), 5);
    assert!(decoded.audio.is_none(), "Wan-VACE emits no audio");
    assert!(steps > 0, "denoise progress streamed");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// Wan model-id → engine-id mapping + the family predicates that drive routing.
#[cfg(target_os = "macos")]
#[test]
fn wan_engine_id_maps_the_three_models() {
    assert_eq!(wan_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
    assert_eq!(wan_engine_id("wan_2_2_t2v_14b"), Some("wan2_2_t2v_14b"));
    assert_eq!(wan_engine_id("wan_2_2_i2v_14b"), Some("wan2_2_i2v_14b"));
    assert_eq!(wan_engine_id("ltx_2_3"), None);
    assert_eq!(wan_engine_id("z_image_turbo"), None);
    // sc-3459: VACE-Fun is a replace_person/control engine, NOT a base Wan T2V/I2V engine —
    // it must stay out of `wan_engine_id` so a replace_person job routes to `generate_wan_vace_fun`
    // (the dual-expert `wan2_2_vace_fun_14b` engine), never the base txt/img→video path.
    assert_eq!(wan_engine_id("wan_2_2_vace_fun_14b"), None);
}

/// Per-model sampling (sc-4997 / sc-10047): with the Lightning toggle on (the default) both A14B
/// MoE models (T2V + I2V) force the 4-step Lightning preset (CFG off); the dense 5B honors an
/// explicit user `steps`/`guidanceScale` and otherwise applies the interim default with CFG retained.
#[cfg(target_os = "macos")]
#[test]
fn wan_sampling_overrides_both_14b_and_5b_interim() {
    let req = request(json!({ "projectId": "p" }));
    // Both A14B MoE models default to Lightning on → forced 4-step / guide 1.0.
    assert_eq!(wan_sampling("wan2_2_t2v_14b", &req), (Some(4), Some(1.0)));
    assert_eq!(wan_sampling("wan2_2_i2v_14b", &req), (Some(4), Some(1.0)));
    // 5B, no override → interim default steps, CFG left to the engine (None ⇒ guide 5.0).
    assert_eq!(
        wan_sampling("wan2_2_ti2v_5b", &req),
        (Some(WAN5B_INTERIM_STEPS), None)
    );
    // 5B honors an explicit user steps + guidanceScale (e.g. the ComfyUI fast settings).
    let over = request(json!({
        "projectId": "p", "advanced": { "steps": 6, "guidanceScale": 1.0 }
    }));
    assert_eq!(wan_sampling("wan2_2_ti2v_5b", &over), (Some(6), Some(1.0)));
    // The 14B Lightning preset (toggle on) ignores user steps/guidance — the distill forces 4/1.0.
    assert_eq!(wan_sampling("wan2_2_t2v_14b", &over), (Some(4), Some(1.0)));
    assert_eq!(wan_sampling("wan2_2_i2v_14b", &over), (Some(4), Some(1.0)));
}

/// sc-10047: `wan_lightning_on` — default-on toggle. Absent / explicit `true` on an A14B MoE model
/// ⇒ on; explicit `false` ⇒ off; the dense 5B and any non-Wan engine are never Lightning.
#[cfg(target_os = "macos")]
#[test]
fn wan_lightning_on_defaults_on_for_a14b_and_honors_toggle() {
    let absent = request(json!({ "projectId": "p" }));
    // A14B MoE: absent ⇒ default-on (backward compatible with the prior always-on behavior).
    assert!(wan_lightning_on("wan2_2_t2v_14b", &absent));
    assert!(wan_lightning_on("wan2_2_i2v_14b", &absent));
    // Explicit true ⇒ on.
    let on = request(json!({ "projectId": "p", "advanced": { "lightning": true } }));
    assert!(wan_lightning_on("wan2_2_t2v_14b", &on));
    assert!(wan_lightning_on("wan2_2_i2v_14b", &on));
    // Explicit false ⇒ off.
    let off = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
    assert!(!wan_lightning_on("wan2_2_t2v_14b", &off));
    assert!(!wan_lightning_on("wan2_2_i2v_14b", &off));
    // The dense 5B and non-Wan engines are never Lightning regardless of the flag.
    assert!(!wan_lightning_on("wan2_2_ti2v_5b", &absent));
    assert!(!wan_lightning_on("wan2_2_ti2v_5b", &on));
    assert!(!wan_lightning_on("some_other_engine", &on));
}

/// sc-10047: with the Lightning toggle OFF, the A14B MoE models run the native multi-step CFG
/// recipe — honor an explicit user `steps`/`guidanceScale`, else `None` so the engine's config.json
/// A14B non-distill defaults (40 steps, dual CFG) stand. 5B is unaffected by the toggle.
#[cfg(target_os = "macos")]
#[test]
fn wan_sampling_toggle_off_runs_native_multistep_cfg() {
    // Toggle off, no user override → (None, None): the engine's A14B non-distill defaults stand.
    let off = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
    assert_eq!(wan_sampling("wan2_2_t2v_14b", &off), (None, None));
    assert_eq!(wan_sampling("wan2_2_i2v_14b", &off), (None, None));
    // Toggle off, user override → honored (multi-step + CFG on).
    let off_over = request(json!({
        "projectId": "p",
        "advanced": { "lightning": false, "steps": 30, "guidanceScale": 4.0 }
    }));
    assert_eq!(
        wan_sampling("wan2_2_t2v_14b", &off_over),
        (Some(30), Some(4.0))
    );
    assert_eq!(
        wan_sampling("wan2_2_i2v_14b", &off_over),
        (Some(30), Some(4.0))
    );
    // Toggle on with the same overrides → still the forced 4-step Lightning preset.
    let on_over = request(json!({
        "projectId": "p",
        "advanced": { "lightning": true, "steps": 30, "guidanceScale": 4.0 }
    }));
    assert_eq!(
        wan_sampling("wan2_2_t2v_14b", &on_over),
        (Some(4), Some(1.0))
    );
    // 5B ignores the lightning flag entirely (no toggle) — interim default steps still apply.
    let five_b = request(json!({ "projectId": "p", "advanced": { "lightning": false } }));
    assert_eq!(
        wan_sampling("wan2_2_ti2v_5b", &five_b),
        (Some(WAN5B_INTERIM_STEPS), None)
    );
}

/// `advanced.mlxQuantize` maps to a quant level; absent → dense / engine-resolved.
#[cfg(target_os = "macos")]
#[test]
fn wan_quant_maps_mlx_quantize() {
    let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(resolve_wan_quant(&q4), Some(Quant::Q4));
    let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(resolve_wan_quant(&q8), Some(Quant::Q8));
    let dense = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(resolve_wan_quant(&dense), None);
    let absent = request(json!({ "projectId": "p" }));
    assert_eq!(resolve_wan_quant(&absent), None);
}

/// Write the six files that make a Wan2.2 A14B tier subdir COMPLETE
/// ([`wan_tier_is_complete`]), so [`wan_tier_subdir`] treats it as present.
#[cfg(target_os = "macos")]
fn write_complete_wan_tier(root: &Path, tier: &str) {
    let dir = root.join(tier);
    std::fs::create_dir_all(&dir).unwrap();
    for file in [
        "high_noise_model.safetensors",
        "low_noise_model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
        "config.json",
    ] {
        std::fs::write(dir.join(file), b"x").unwrap();
    }
}

/// `mlxQuantize` selects the preferred A14B tier, then falls back to the always-smaller
/// present tiers (bf16 is only ever tried when explicitly requested).
#[cfg(target_os = "macos")]
#[test]
fn wan_tier_order_prefers_then_falls_back() {
    let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(wan_tier_order(&bf16), &["bf16", "q8", "q4"]);
    let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(wan_tier_order(&q8), &["q8", "q4"]);
    // An explicit Q4 pick stays q4-first (never overridden by the q8 default).
    let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(wan_tier_order(&q4), &["q4", "q8"]);
    // sc-10859: an absent knob defaults q4-first on the video lane (the sc-10726 app-wide q8 default
    // is NOT applied to video — no MLX video Q8 lever, so a silent q8 only risks an OOM). bf16 is
    // never in a default job's search path (no OOM by accident).
    let absent = request(json!({ "projectId": "p" }));
    assert_eq!(wan_tier_order(&absent), &["q4", "q8"]);
}

/// [`wan_tier_subdir`] resolves the requested tier, falls back to a smaller COMPLETE
/// tier, ignores a partially-downloaded one, and returns `None` for a legacy flat root.
#[cfg(target_os = "macos")]
#[test]
fn wan_tier_subdir_resolves_and_falls_back() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_wan_tier_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // Legacy flat root (no tier subdirs) → None (caller keeps root + load-time quant).
    let q8_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(wan_tier_subdir(root, &q8_req), None);

    // Only q4 present: a q8 request falls back to the smaller complete q4 tier.
    write_complete_wan_tier(root, "q4");
    assert_eq!(wan_tier_subdir(root, &q8_req), Some(root.join("q4")));

    // A partial q8 (missing a file) is skipped, still falling back to q4.
    std::fs::create_dir_all(root.join("q8")).unwrap();
    std::fs::write(root.join("q8").join("config.json"), b"x").unwrap();
    assert_eq!(wan_tier_subdir(root, &q8_req), Some(root.join("q4")));

    // Completed q8 now wins for an explicit q8 request; but a DEFAULT job (no mlxQuantize) resolves
    // q4-first on the video lane even with q8 installed (sc-10859: the sc-10726 app-wide q8 default
    // is not applied to video — no MLX video Q8 lever). An explicit q4 pick still resolves q4.
    write_complete_wan_tier(root, "q8");
    assert_eq!(wan_tier_subdir(root, &q8_req), Some(root.join("q8")));
    let default_req = request(json!({ "projectId": "p" }));
    assert_eq!(wan_tier_subdir(root, &default_req), Some(root.join("q4")));
    let q4_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(wan_tier_subdir(root, &q4_req), Some(root.join("q4")));

    // bf16 request with no bf16 tier falls back to q8 (never silently to a default).
    let bf16_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(wan_tier_subdir(root, &bf16_req), Some(root.join("q8")));
}

/// Every Wan quant-matrix tier repo (TI2V-5B sc-9941 / T2V sc-9942 / I2V sc-9943) must pin an
/// exact commit (not the mutable `main`) so an upstream re-push can't swap a checkpoint the
/// on-demand fetch loads (mirrors the LTX/SeedVR2 pins). `wan_tier_repo` also routes each model id
/// to its own repo+revision.
#[cfg(target_os = "macos")]
#[test]
fn wan_tier_revisions_are_pinned_commits_not_main() {
    for (label, revision) in [
        ("TI2V-5B", WAN_TI2V_5B_REVISION),
        ("T2V", WAN_T2V_14B_REVISION),
        ("I2V", WAN_I2V_14B_REVISION),
    ] {
        assert_ne!(
            revision, "main",
            "the {label} tier repo must pin a fixed revision before release"
        );
        assert_eq!(
            revision.len(),
            40,
            "a pinned HF revision is a 40-char commit sha ({label})"
        );
        assert!(
            revision
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned {label} revision must be lowercase hex"
        );
    }
    // Each quant-matrix model id routes to its own repo + revision (the 5B engine's request.model
    // is `wan_2_2`); a model with no hosted matrix has no tier repo.
    assert_eq!(
        wan_tier_repo("wan_2_2"),
        Some((WAN_TI2V_5B_REPO, WAN_TI2V_5B_REVISION))
    );
    assert_eq!(
        wan_tier_repo("wan_2_2_t2v_14b"),
        Some((WAN_T2V_14B_REPO, WAN_T2V_14B_REVISION))
    );
    assert_eq!(
        wan_tier_repo("wan_2_2_i2v_14b"),
        Some((WAN_I2V_14B_REPO, WAN_I2V_14B_REVISION))
    );
    assert_eq!(wan_tier_repo("bernini"), None);
}

/// sc-9945: the Bernini quant-matrix repo must pin an exact commit (not the mutable `main`) so an
/// upstream re-push can't swap a checkpoint the on-demand tier fetch loads. INTENTIONALLY red until
/// the tiers are hosted and `BERNINI_REVISION` is pinned to the real commit (mirrors sc-9941's flow).
#[cfg(target_os = "macos")]
#[test]
fn bernini_tier_revision_is_pinned_commit_not_main() {
    assert_ne!(
        BERNINI_REVISION, "main",
        "the Bernini tier repo must pin a fixed revision before release"
    );
    assert_eq!(
        BERNINI_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        BERNINI_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned Bernini revision must be lowercase hex"
    );
}

/// sc-9945: a COMPLETE Bernini tier (all [`BERNINI_TIER_FILES`], incl. the planner's nested
/// `mllm/tokenizer.json`) resolves for the requested quant, and a missing preferred tier falls
/// through to the next smaller complete tier — never silently to the wrong one. Composite: the three
/// packable weights live in the SAME tier subdir as the dense remainder, so one resolution covers
/// both the planner and the renderer.
#[cfg(target_os = "macos")]
#[test]
fn bernini_tier_subdir_resolves_complete_tier() {
    let root_guard = tempfile::Builder::new()
        .prefix("bernini-tier-test-")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let make_tier = |tier: &str| {
        for file in BERNINI_TIER_FILES {
            let path = root.join(tier).join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
    };
    make_tier("q4");
    make_tier("q8");

    // No explicit pick: the default is LANE-DEPENDENT (sc-10859), clamped to what's on disk. The
    // VIDEO lane defaults q4-first (OOM carve-out); the IMAGE lane keeps epic-10721's q8 default.
    assert_eq!(
        bernini_tier_subdir(root, None, BERNINI_VIDEO_DEFAULT_TIER_ORDER),
        Some(root.join("q4"))
    );
    assert_eq!(
        bernini_tier_subdir(root, None, BERNINI_IMAGE_DEFAULT_TIER_ORDER),
        Some(root.join("q8"))
    );
    // Explicit picks are lane-independent (`default_order` is consulted ONLY for `None`).
    // An explicit Q4 pick stays q4 (never overridden by a default).
    assert_eq!(
        bernini_tier_subdir(root, Some(4), BERNINI_IMAGE_DEFAULT_TIER_ORDER),
        Some(root.join("q4"))
    );
    // Q8 (bits >= 8) → q8.
    assert_eq!(
        bernini_tier_subdir(root, Some(8), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
        Some(root.join("q8"))
    );
    // bf16 (bits <= 0) with no bf16 tier falls back to q8 (never silently to a default).
    assert_eq!(
        bernini_tier_subdir(root, Some(0), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
        Some(root.join("q8"))
    );
    // An incomplete tier (missing the nested planner tokenizer) is not resolved.
    std::fs::remove_file(root.join("q8").join("mllm/tokenizer.json")).unwrap();
    assert_eq!(
        bernini_tier_subdir(root, Some(8), BERNINI_VIDEO_DEFAULT_TIER_ORDER),
        Some(root.join("q4"))
    );
}

/// The single-expert TI2V-5B tier (sc-9941) is COMPLETE with one `model.safetensors` (not the
/// A14B two-expert set), and `wan_tier_subdir` resolves it for the `wan_2_2` model id.
#[cfg(target_os = "macos")]
#[test]
fn wan_tier_ti2v_5b_single_expert_completeness() {
    assert_eq!(wan_tier_files("wan_2_2"), WAN_TI2V_5B_TIER_FILES);
    assert_eq!(wan_tier_files("wan_2_2_t2v_14b"), WAN_A14B_TIER_FILES);

    let root_guard = tempfile::Builder::new()
        .prefix("sw_wan5b_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let dir = root.join("q4");
    std::fs::create_dir_all(&dir).unwrap();
    for file in WAN_TI2V_5B_TIER_FILES {
        std::fs::write(dir.join(file), b"x").unwrap();
    }
    // Complete for the 5B (single transformer) but NOT for the A14B set (missing the experts).
    assert!(wan_tier_is_complete(&dir, WAN_TI2V_5B_TIER_FILES));
    assert!(!wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES));

    let req = request(json!({ "projectId": "p", "model": "wan_2_2" }));
    assert_eq!(wan_tier_subdir(root, &req), Some(dir));
}

/// Write the six files that make a SCAIL-2 tier subdir COMPLETE ([`scail2_tier_is_complete`]), so
/// [`scail2_tier_subdir`] treats it as present.
#[cfg(target_os = "macos")]
fn write_complete_scail2_tier(root: &Path, tier: &str) {
    let dir = root.join(tier);
    std::fs::create_dir_all(&dir).unwrap();
    for file in SCAIL2_TIER_FILES {
        std::fs::write(dir.join(file), b"x").unwrap();
    }
}

/// `mlxQuantize` selects the preferred SCAIL-2 tier, then falls back to the always-smaller present
/// tiers (bf16 is only ever tried when explicitly requested) — mirrors [`wan_tier_order`].
#[cfg(target_os = "macos")]
#[test]
fn scail2_tier_order_prefers_then_falls_back() {
    let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(scail2_tier_order(&bf16), &["bf16", "q8", "q4"]);
    let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(scail2_tier_order(&q8), &["q8", "q4"]);
    // An explicit Q4 pick stays q4-first — the SAME arm as the no-pick default now (sc-10859),
    // which is exactly why the resolver no longer needs its old raw-mlxQuantize-presence check.
    let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(scail2_tier_order(&q4), &["q4", "q8"]);
    // sc-10859: an absent knob defaults q4-first on the video lane (NOT the sc-10726 app-wide q8 —
    // no MLX video Q8 lever); bf16 is never in a default job's search path (no OOM by accident).
    let absent = request(json!({ "projectId": "p" }));
    assert_eq!(scail2_tier_order(&absent), &["q4", "q8"]);
}

/// [`scail2_tier_subdir`] resolves the requested tier, falls back to a smaller COMPLETE tier,
/// ignores a partially-downloaded one, and returns `None` for a legacy flat root.
#[cfg(target_os = "macos")]
#[test]
fn scail2_tier_subdir_resolves_and_falls_back() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_scail2_tier_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    // Legacy flat root (no tier subdirs) → None (caller keeps root + load-time quant).
    let q8_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(scail2_tier_subdir(root, &q8_req), None);

    // Only q4 present: a q8 request falls back to the smaller complete q4 tier.
    write_complete_scail2_tier(root, "q4");
    assert_eq!(scail2_tier_subdir(root, &q8_req), Some(root.join("q4")));

    // A partial q8 (missing files) is skipped, still falling back to q4.
    std::fs::create_dir_all(root.join("q8")).unwrap();
    std::fs::write(root.join("q8").join("config.json"), b"x").unwrap();
    assert_eq!(scail2_tier_subdir(root, &q8_req), Some(root.join("q4")));

    // Completed q8 now wins for an explicit q8 request; but a DEFAULT job (no mlxQuantize) resolves
    // q4-first on the video lane even with q8 installed (sc-10859: the sc-10726 app-wide q8 default
    // is not applied to video — no MLX video Q8 lever). An explicit q4 pick still resolves q4.
    write_complete_scail2_tier(root, "q8");
    assert_eq!(scail2_tier_subdir(root, &q8_req), Some(root.join("q8")));
    let default_req = request(json!({ "projectId": "p" }));
    assert_eq!(
        scail2_tier_subdir(root, &default_req),
        Some(root.join("q4"))
    );
    let q4_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(scail2_tier_subdir(root, &q4_req), Some(root.join("q4")));

    // bf16 request with no bf16 tier falls back to q8 (never silently to a default).
    let bf16_req = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(scail2_tier_subdir(root, &bf16_req), Some(root.join("q8")));
}

/// The SCAIL-2 quant-matrix tier repo (sc-9944) must pin an exact commit (not the mutable `main`)
/// so an upstream re-push can't swap a checkpoint the on-demand fetch loads (mirrors the Wan pins),
/// and `scail2_tier_repo` routes only the `scail2_14b` id to it.
#[cfg(target_os = "macos")]
#[test]
fn scail2_tier_revision_is_pinned_commit_not_main() {
    assert_ne!(
        SCAIL2_REVISION, "main",
        "the SCAIL-2 tier repo must pin a fixed revision before release"
    );
    assert_eq!(
        SCAIL2_REVISION.len(),
        40,
        "a pinned HF revision is a 40-char commit sha"
    );
    assert!(
        SCAIL2_REVISION
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the pinned SCAIL-2 revision must be lowercase hex"
    );
    assert_eq!(
        scail2_tier_repo("scail2_14b"),
        Some((SCAIL2_REPO, SCAIL2_REVISION))
    );
    assert_eq!(scail2_tier_repo("wan_2_2"), None);
    assert_eq!(scail2_tier_repo("bernini"), None);
}

/// The `.high_noise.safetensors` → `.low_noise.safetensors` sibling convention
/// (case-insensitive; only fires when the sibling file exists).
#[cfg(target_os = "macos")]
#[test]
fn wan_moe_sibling_pairs_high_and_low() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_moe_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let high = dir.join("char.high_noise.safetensors");
    let low = dir.join("char.low_noise.safetensors");
    std::fs::write(&high, b"x").unwrap();
    // No sibling yet → None.
    assert_eq!(wan_moe_low_noise_sibling(&high), None);
    std::fs::write(&low, b"x").unwrap();
    assert_eq!(wan_moe_low_noise_sibling(&high), Some(low));
    // A single-file (non-high-noise) LoRA never pairs.
    let single = dir.join("plain.safetensors");
    std::fs::write(&single, b"x").unwrap();
    assert_eq!(wan_moe_low_noise_sibling(&single), None);
}

/// Minimal valid safetensors (8-byte LE header length + JSON header), optionally stamping
/// `__metadata__.networkType` so `classify_adapter` can distinguish peft LoKr from plain LoRA.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_lora_fixture(path: &Path, network_type: Option<&str>) {
    let mut meta = serde_json::Map::new();
    meta.insert("format".to_owned(), json!("pt"));
    if let Some(nt) = network_type {
        meta.insert("networkType".to_owned(), json!(nt));
    }
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), Value::Object(meta));
    let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    std::fs::write(path, buffer).unwrap();
}

/// Wan-VACE is single-dense: each user LoRA/LoKr resolves to one shared spec with
/// `moe_expert: None` (no Lightning, no high/low split), the kind set by the file's metadata,
/// and the scale taken from the request `weight` (sc-3893).
#[cfg(target_os = "macos")]
#[test]
fn wan_vace_adapters_are_single_dense() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_vace_lora_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let plain = dir.join("style.safetensors");
    let lokr = dir.join("char.safetensors");
    write_lora_fixture(&plain, None);
    write_lora_fixture(&lokr, Some("lokr"));

    let req = request(json!({
        "projectId": "p",
        "loras": [
            { "path": plain.to_string_lossy(), "weight": 0.5 },
            { "path": lokr.to_string_lossy(), "weight": 0.9 },
        ],
    }));
    // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
    // the fixture dir the temp LoRAs live in.
    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };
    let specs = resolve_wan_vace_adapters(&settings, &req).expect("resolve vace adapters");
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].path, plain.canonicalize().unwrap());
    assert_eq!(specs[0].kind, AdapterKind::Lora);
    assert!((specs[0].scale - 0.5).abs() < 1e-6);
    assert!(specs[0].moe_expert.is_none(), "VACE is single-dense");
    assert!(specs[0].pass_scales.is_none());
    assert_eq!(specs[1].kind, AdapterKind::Lokr);
    assert!((specs[1].scale - 0.9).abs() < 1e-6);
    assert!(specs[1].moe_expert.is_none());

    // Over the per-job cap → a clear payload error (mirrors the base Wan path).
    let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
        .map(|_| json!({ "path": plain.to_string_lossy() }))
        .collect();
    let over = request(json!({ "projectId": "p", "loras": many }));
    assert!(matches!(
        resolve_wan_vace_adapters(&settings, &over),
        Err(WorkerError::InvalidPayload(_))
    ));
}

/// Lay down a fake `lightx2v/Wan2.2-Lightning` HF snapshot under `data_dir` with the
/// per-architecture high/low pair, so [`resolve_lightning_loras`] resolves the Lightning
/// distill in a hermetic test (mirrors `write_complete_wan_tier`). Returns the two file paths.
#[cfg(target_os = "macos")]
fn write_fake_wan_lightning(data_dir: &Path, engine_id: &str) -> (PathBuf, PathBuf) {
    let subdir = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        other => panic!("no Lightning subdir for {other}"),
    };
    let base = data_dir
        .join("cache")
        .join("huggingface")
        .join("hub")
        .join("models--lightx2v--Wan2.2-Lightning")
        .join("snapshots")
        .join("deadbeef")
        .join(subdir);
    std::fs::create_dir_all(&base).unwrap();
    let high = base.join("high_noise_model.safetensors");
    let low = base.join("low_noise_model.safetensors");
    std::fs::write(&high, b"x").unwrap();
    std::fs::write(&low, b"x").unwrap();
    (high, low)
}

/// sc-10047: `resolve_wan_adapters` gates the A14B Lightning distill pair on the default-on
/// toggle. Toggle on (default) → the high/low Lightning pair is prepended (per-expert tagged),
/// then user LoRAs. Toggle off → NO Lightning adapters, but user LoRAs are still honored. Only an
/// explicit `lightning:false` opts out; absent defaults to on (backward compatible). This test is
/// hermetic: it pins `HF_HUB_CACHE` at its own fixture hub for the whole body (sc-13898), so the
/// fake snapshot resolves regardless of the ambient HF cache env.
#[cfg(target_os = "macos")]
#[test]
fn wan_adapters_gate_lightning_on_toggle() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_wan_light_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    // PIN the hub cache at this fixture's own dir for the whole test. `huggingface_hub_cache_dir`
    // reads `HF_HUB_CACHE` (then `HUGGINGFACE_HUB_CACHE` / `HF_HOME`) BEFORE `data_dir`, so the old
    // "skip if any HF var is set" guard raced a concurrent test's `EnvVars::set` writer: a real HF
    // path landing between the guard and the resolve pointed the cache elsewhere and the fake
    // Lightning snapshot went missing (sc-13898, the "env locks protect writers, not readers"
    // class). `EnvVars` holds the crate-wide env lock until it drops at end of test, closing the
    // window; pinning the value also means this never silently no-ops on a box with an HF var set.
    let _env = EnvVars::set(&[(
        "HF_HUB_CACHE",
        fake_hf_hub_dir(dir).to_str().expect("utf-8 fixture hub"),
    )]);
    let (high, low) = write_fake_wan_lightning(dir, "wan2_2_t2v_14b");
    // A user LoRA lives under data_dir so the confinement check passes.
    let user_lora = dir.join("style.safetensors");
    write_lora_fixture(&user_lora, None);

    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };

    // Toggle ON (default / absent) → Lightning high/low pair, then the user LoRA (3 specs).
    for adv in [json!({}), json!({ "lightning": true })] {
        let req = request(json!({
            "projectId": "p",
            "advanced": adv,
            "loras": [ { "path": user_lora.to_string_lossy(), "weight": 0.5 } ],
        }));
        let specs = resolve_wan_adapters(&settings, &req, "wan2_2_t2v_14b")
            .expect("resolve adapters (lightning on)");
        assert_eq!(specs.len(), 3, "lightning pair + user lora");
        assert_eq!(specs[0].path, high);
        assert_eq!(specs[0].moe_expert, Some(MoeExpert::High));
        assert_eq!(specs[1].path, low);
        assert_eq!(specs[1].moe_expert, Some(MoeExpert::Low));
        // The user LoRA is the single-file shared spec (no high_noise sibling).
        assert_eq!(specs[2].path, user_lora.canonicalize().unwrap());
        assert!(specs[2].moe_expert.is_none());
        assert!((specs[2].scale - 0.5).abs() < 1e-6);
    }

    // Toggle OFF → NO Lightning adapters; the user LoRA is still applied (1 spec).
    let off = request(json!({
        "projectId": "p",
        "advanced": { "lightning": false },
        "loras": [ { "path": user_lora.to_string_lossy(), "weight": 0.5 } ],
    }));
    let specs = resolve_wan_adapters(&settings, &off, "wan2_2_t2v_14b")
        .expect("resolve adapters (lightning off)");
    assert_eq!(specs.len(), 1, "no Lightning, user LoRA only");
    assert_eq!(specs[0].path, user_lora.canonicalize().unwrap());
    assert!(specs[0].moe_expert.is_none());

    // Toggle OFF works even when the Lightning snapshot is absent (nothing to resolve): remove
    // the fake snapshot and confirm the off path still succeeds with just the user LoRA.
    std::fs::remove_file(&high).unwrap();
    std::fs::remove_file(&low).unwrap();
    let specs_no_light = resolve_wan_adapters(&settings, &off, "wan2_2_t2v_14b")
        .expect("lightning-off needs no Lightning snapshot");
    assert_eq!(specs_no_light.len(), 1);
}

/// sc-5723 (WKA-002): a LoRA path from the (attacker-controllable) payload that
/// resolves outside every app-managed root is rejected before the file is opened —
/// the worker must not be pointed at an arbitrary host `.safetensors`. The fixture
/// lives in a sibling temp dir that is NOT under the configured `data_dir`.
#[cfg(target_os = "macos")]
#[test]
fn lora_path_outside_app_managed_root_is_rejected() {
    let data_dir_guard = tempfile::Builder::new()
        .prefix("sw_lora_data_")
        .tempdir()
        .expect("temp dir");
    let data_dir = data_dir_guard.path();
    let outside_guard = tempfile::Builder::new()
        .prefix("sw_lora_evil_")
        .tempdir()
        .expect("temp dir");
    let outside = outside_guard.path();
    let evil = outside.join("evil.safetensors");
    write_lora_fixture(&evil, None);

    let settings = Settings {
        data_dir: data_dir.to_path_buf(),
        ..Settings::from_env()
    };
    let req = request(json!({
        "projectId": "p",
        "loras": [ { "path": evil.to_string_lossy(), "weight": 0.5 } ],
    }));
    // The HF cache roots are env-derived; this temp path is under neither, so it
    // must be refused. (Guard against the host's real HF cache happening to be a
    // parent — vanishingly unlikely for a fresh uuid temp dir.)
    let result = resolve_wan_vace_adapters(&settings, &req);
    assert!(
        matches!(result, Err(WorkerError::InvalidPayload(_))),
        "expected out-of-root LoRA to be rejected, got {result:?}"
    );
}

/// SCAIL-2 is single-dense like Wan-VACE (sc-5451/sc-5686): each user LoRA/LoKr (and the bundled
/// Bias-Aware DPO LoRA, which arrives the same way) resolves to one shared spec with
/// `moe_expert: None` and no `pass_scales`, the kind set by the file's metadata, scale from
/// `weight`; over the per-job cap is a clear payload error.
#[cfg(target_os = "macos")]
#[test]
fn scail2_adapters_are_single_dense() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_scail2_lora_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let plain = dir.join("dpo.safetensors");
    let lokr = dir.join("char.safetensors");
    write_lora_fixture(&plain, None);
    write_lora_fixture(&lokr, Some("lokr"));

    let req = request(json!({
        "projectId": "p",
        "loras": [
            { "path": plain.to_string_lossy(), "weight": 1.0 },
            { "path": lokr.to_string_lossy(), "weight": 0.7 },
        ],
    }));
    // sc-5723: LoRA paths are confined to the app data dir, so point data_dir at
    // the fixture dir the temp LoRAs live in.
    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };
    let specs = resolve_scail2_adapters(&settings, &req).expect("resolve scail2 adapters");
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].path, plain.canonicalize().unwrap());
    assert_eq!(specs[0].kind, AdapterKind::Lora);
    assert!((specs[0].scale - 1.0).abs() < 1e-6);
    assert!(specs[0].moe_expert.is_none(), "SCAIL-2 is single-dense");
    assert!(specs[0].pass_scales.is_none());
    assert_eq!(specs[1].kind, AdapterKind::Lokr);
    assert!((specs[1].scale - 0.7).abs() < 1e-6);
    assert!(specs[1].moe_expert.is_none());

    let many: Vec<Value> = (0..MAX_JOB_LORAS + 1)
        .map(|_| json!({ "path": plain.to_string_lossy() }))
        .collect();
    let over = request(json!({ "projectId": "p", "loras": many }));
    assert!(matches!(
        resolve_scail2_adapters(&settings, &over),
        Err(WorkerError::InvalidPayload(_))
    ));
}

/// sc-8830 drift-fix regression: the shared [`resolve_lora_file`] resolves a `.safetensors`
/// nested in a **subdirectory** of the LoRA dir (via core's recursive `first_safetensors_path`).
/// The old candle twin `candle_resolve_lora_file` used a shallow non-recursive `read_dir`, so it
/// returned "LoRA has no .safetensors" for exactly this shape — a latent candle-lane bug that
/// converging on the core helper fixes. Both backends now resolve it identically.
#[cfg(target_os = "macos")]
#[test]
fn resolve_lora_file_finds_nested_safetensors() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_lora_nested_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let nested = dir.join("adapter");
    std::fs::create_dir_all(&nested).unwrap();
    let weight = nested.join("model.safetensors");
    write_lora_fixture(&weight, None);

    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };
    // Point the resolver at the LoRA dir (not the file); it must recurse into `adapter/`.
    // No declared file → falls back to the recursive scan.
    let resolved = resolve_lora_file(&settings, dir.to_path_buf(), None)
        .expect("nested .safetensors must resolve");
    assert_eq!(resolved, weight.canonicalize().unwrap());
}

/// sc-10221: a trained LoRA's folder holds step checkpoints alongside the final
/// adapter; the video resolver must load the manifest-declared final file, not an
/// arbitrary sibling the directory scan might pick.
#[cfg(target_os = "macos")]
#[test]
fn resolve_lora_file_prefers_declared_over_checkpoint() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_lora_ckpt_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let final_adapter = dir.join("my_style.safetensors");
    write_lora_fixture(&dir.join("my_style-step250.safetensors"), None);
    write_lora_fixture(&final_adapter, None);

    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };
    let resolved = resolve_lora_file(&settings, dir.to_path_buf(), Some("my_style.safetensors"))
        .expect("declared adapter must resolve");
    assert_eq!(resolved, final_adapter.canonicalize().unwrap());
}

/// sc-8830: the shared [`resolve_dense_adapters`] enforces exactly the `max_loras` cap it is
/// handed — one shared count guard for every dense family (Wan-VACE / SCAIL-2, both backends).
#[cfg(target_os = "macos")]
#[test]
fn resolve_dense_adapters_honors_the_max_lora_cap() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_dense_cap_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let plain = dir.join("style.safetensors");
    write_lora_fixture(&plain, None);
    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };

    // Two LoRAs under a cap of 1 → rejected; the same two under a cap of 2 → accepted.
    let two: Vec<Value> = (0..2)
        .map(|_| json!({ "path": plain.to_string_lossy(), "weight": 0.5 }))
        .collect();
    let req = request(json!({ "projectId": "p", "loras": two }));
    assert!(matches!(
        resolve_dense_adapters(&settings, &req, 1),
        Err(WorkerError::InvalidPayload(_))
    ));
    let specs = resolve_dense_adapters(&settings, &req, 2).expect("under cap resolves");
    assert_eq!(specs.len(), 2);
    assert!(specs.iter().all(|s| s.moe_expert.is_none()));
}

/// sc-8830: [`non_empty_negative_prompt`] trims and maps empty/whitespace to `None` (so engines
/// see "no negative", not the empty string) and passes a real negative through trimmed.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn non_empty_negative_prompt_trims_and_drops_empty() {
    let empty = request(json!({ "projectId": "p", "negativePrompt": "   " }));
    assert_eq!(non_empty_negative_prompt(&empty), None);
    let missing = request(json!({ "projectId": "p" }));
    assert_eq!(non_empty_negative_prompt(&missing), None);
    let real = request(json!({ "projectId": "p", "negativePrompt": "  blurry  " }));
    assert_eq!(non_empty_negative_prompt(&real), Some("blurry".to_owned()));
}

/// sc-8830: the shared [`resolve_mlx_dense_quant`] (Bernini / SCAIL-2) defaults to Q4, honors
/// `mlxQuantize: 8` → Q8, and treats `<= 0` as bf16 (`None`) — never defaulting to bf16.
///
/// The Q4 default is the OWNED exception to epic 10721's app-wide Q8 default (sc-10750): this
/// resolver is the legacy flat-snapshot load-time quant + the on-demand tier-fetch gate, where Q4 is
/// the OOM-safe / no-surprise-download floor. The modern turnkey default (Q8-clamped-to-installed)
/// lives in [`bernini_tier_order`] / [`scail2_tier_order`], not here. This assertion guards that
/// decision — do not flip it to Q8 without the sc-10733 (S8) capability clamp.
#[cfg(target_os = "macos")]
#[test]
fn resolve_mlx_dense_quant_defaults_q4_and_honors_override() {
    let default = request(json!({ "projectId": "p" }));
    assert_eq!(resolve_mlx_dense_quant(&default), Some(Quant::Q4));
    let q8 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 8 } }));
    assert_eq!(resolve_mlx_dense_quant(&q8), Some(Quant::Q8));
    let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert_eq!(resolve_mlx_dense_quant(&q4), Some(Quant::Q4));
    let bf16 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert_eq!(resolve_mlx_dense_quant(&bf16), None);
    // String forms parse the same way (the payload knob can arrive as a JSON string).
    let q8_str = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": "8" } }));
    assert_eq!(resolve_mlx_dense_quant(&q8_str), Some(Quant::Q8));
}

/// A locally-converted Wan2.2 TI2V-5B dir if one is present (env override or the
/// app-managed default), else `None` so the real-weight smoke skips.
#[cfg(target_os = "macos")]
fn wan_5b_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_WAN5B_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let path =
        PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2");
    path.join("config.json").is_file().then_some(path)
}

/// Real in-process Wan2.2 TI2V-5B T2V through the engine (the lightest Wan model —
/// ~10 GB, safe on a 128 GB Mac). Loads the converted 5B snapshot and denoises a
/// tiny 5-frame clip, asserting frames come back RGB8-sized with streamed progress.
/// `#[ignore]` — the weights live outside CI; run manually where they are present.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
#[test]
fn wan_5b_real_weights() {
    let Some(model_dir) = wan_5b_dir() else {
        eprintln!("skipping wan_5b_real_weights: no converted TI2V-5B dir found");
        return;
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan2_2_ti2v_5b",
        model_dir,
        prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
        width: 256,
        height: 256,
        frames: 5,
        fps: 16,
        steps: Some(8),
        seed: 7,
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded =
        run_video_generation(input, &cancel, &mut on_progress).expect("5B T2V generation");
    assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
    assert!(decoded.fps >= 1);
    assert!(decoded.audio.is_none(), "Wan emits no audio");
    assert!(steps > 0, "denoise progress streamed");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// Real in-process load+generate verification for the Wan2.2 T2V-A14B quant-matrix tiers
/// (sc-9942, epic 8506). For each tier subdir present under `$SCENEWORKS_WAN_T2V_14B_TIER_OUT`
/// (`bf16/`/`q8/`/`q4/`, e.g. the wan_t2v_14b_tier_build output), loads the tier through the SAME
/// engine path a job uses (`run_video_generation`, `quant: None` — the pre-packed config.json is
/// authoritative) and denoises a tiny 5-frame clip, asserting the frames come back RGB8-sized and
/// NON-degenerate (real pixel variance, so a broken packed load surfaces instead of silent noise).
/// This is the on-device evidence that every hosted tier loads packed with no install-time convert
/// peak. Runs a few CFG steps WITHOUT the Lightning distill so it is self-contained (no LoRA
/// download); output is a coherence check, not a quality bar. `#[ignore]` — needs the built tiers
/// (~28–69 GB each) + a big-memory Mac; the cache limit is capped so loading bf16 then q8 then q4
/// in one process does not accumulate residue (sc-5567).
///
/// ```sh
/// export SCENEWORKS_WAN_T2V_14B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
/// cargo test -p sceneworks-worker --release --lib wan_t2v_14b_tier_real_weights -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[ignore = "loads the built Wan2.2 T2V-A14B tiers; run manually on a Mac where they are present"]
#[test]
fn wan_t2v_14b_tier_real_weights() {
    let Some(root) = std::env::var_os("SCENEWORKS_WAN_T2V_14B_TIER_OUT").map(PathBuf::from) else {
        eprintln!("skipping wan_t2v_14b_tier_real_weights: set SCENEWORKS_WAN_T2V_14B_TIER_OUT");
        return;
    };
    // Cap the buffer cache so three sequential heavy loads release between tiers (sc-5567).
    mlx_rs::memory::set_cache_limit(0);
    let mut verified = 0;
    for tier in ["bf16", "q8", "q4"] {
        let dir = root.join(tier);
        if !wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
            eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
            continue;
        }
        eprintln!("verifying {tier} tier → {}", dir.display());
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_t2v_14b",
            model_dir: dir.clone(),
            // Pre-packed tier: config.json carries the quant; the engine reconstructs the experts
            // packed and rejects a conflicting override, so load with None (the worker path does
            // the same in resolve_wan_tier_dir_and_quant).
            quant: None,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            // A few CFG steps WITHOUT the Lightning distill — enough to exercise the packed
            // experts + VAE decode and produce non-degenerate frames for a load check.
            width: 256,
            height: 256,
            frames: 5,
            fps: 16,
            steps: Some(6),
            guidance: Some(5.0),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|e| panic!("{tier} tier T2V generation failed: {e:?}"));
        // The engine's temporal VAE fixes the decoded frame count (a 5-latent-frame request
        // decodes to 8 RGB frames for the A14B); assert a real multi-frame clip, not an exact
        // count.
        assert!(
            decoded.frames.len() > 1,
            "{tier}: got {} frames, expected a multi-frame clip",
            decoded.frames.len()
        );
        assert!(steps > 0, "{tier}: denoise progress streamed");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{tier}: frames are RGB8-sized"
        );
        // Non-degenerate: a broken packed load tends to decode to a flat/NaN frame. Require real
        // spatial variance in the first frame.
        let f0 = &decoded.frames[0];
        let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
        let var = f0
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / f0.pixels.len() as f64;
        assert!(
            var.sqrt() > 3.0,
            "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
            var.sqrt()
        );
        mlx_rs::memory::clear_cache();
        verified += 1;
    }
    assert!(
        verified > 0,
        "no tier subdirs found under SCENEWORKS_WAN_T2V_14B_TIER_OUT"
    );
    eprintln!("verified {verified} tier(s)");
}

/// Locate a tier subdir (`q4`/`q8`/`bf16`) for a hosted Wan A14B mirror, honoring an explicit
/// env override first, then the HF hub cache (`~/.cache/huggingface/hub/models--<repo>/snapshots/
/// <rev>/<tier>`). Returns the tier dir only when it is a COMPLETE A14B tier (the six-file set),
/// else `None` so the on-device test skips cleanly. Used by
/// [`wan_a14b_lightning_additive_ondevice`] to drive the real q4 tier + Lightning additive path
/// (sc-10049, epic 10043) exactly as a job does.
#[cfg(target_os = "macos")]
fn wan_a14b_cached_tier(env_var: &str, repo: &str, tier: &str) -> Option<PathBuf> {
    if let Ok(root) = std::env::var(env_var) {
        let dir = PathBuf::from(root.trim()).join(tier);
        if wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
            return Some(dir);
        }
    }
    // HF hub cache: models--<org>--<name>/snapshots/<rev>/<tier>.
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let snapshots = std::fs::read_dir(&cache).ok()?;
    for entry in snapshots.flatten() {
        let dir = entry.path().join(tier);
        if wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
            return Some(dir);
        }
    }
    None
}

/// Resolve the cached Lightning distill high/low pair for an A14B engine straight from the HF
/// hub cache (the same layout [`resolve_lightning_loras`] reads under `data_dir`, but pointed at
/// `~/.cache/huggingface`). Returns `None` when the pair is absent so the on-device test skips.
#[cfg(target_os = "macos")]
fn wan_a14b_cached_lightning(engine_id: &str) -> Option<(PathBuf, PathBuf)> {
    let subdir = match engine_id {
        "wan2_2_t2v_14b" => "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1",
        "wan2_2_i2v_14b" => "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
        _ => return None,
    };
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--lightx2v--Wan2.2-Lightning/snapshots");
    for entry in std::fs::read_dir(&cache).ok()?.flatten() {
        let base = entry.path().join(subdir);
        let high = base.join("high_noise_model.safetensors");
        let low = base.join("low_noise_model.safetensors");
        if high.is_file() && low.is_file() {
            return Some((high, low));
        }
    }
    None
}

/// **sc-10049 (epic 10043) — the on-device close of the loop on sc-10030.** Drives the REAL
/// Wan2.2 T2V-A14B **q4** tier through the exact engine path a job uses (`run_video_generation`
/// → `crate::inference_runtime::load` → `Generator::generate`) with the Lightning distill pair installed as
/// **forward-time additive adapters** on the pre-quantized (packed) experts. This is the exact
/// case that FAILED before this epic with the mlx-gen `model.rs:614` rejection —
/// `"LoRA adapters on a pre-quantized snapshot need dequantize-then-merge … not yet wired"` —
/// and must now complete a 4-step distilled clip with the base STAYING packed (the low-memory-Mac
/// guarantee this epic exists for: NO ~28 GB/expert dense dequant spike).
///
/// Three sub-cases on the same cached q4 tier:
///   1. **Lightning ON, 4 steps** — the Lightning high/low pair (per-expert tagged) as additive
///      adapters; asserts a coherent multi-frame clip (the additive path completed, no 614 error).
///   2. **Lightning OFF, multi-step CFG** — no adapters, several CFG steps; the native recipe
///      still runs on the packed tier (regression guard the toggle-off path is intact).
///   3. **Lightning ON + a plain user LoRA** — the distill pair PLUS a single-file plain LoRA
///      (the additive user-LoRA path on a packed base). A real Civitai-style LoRA is used when
///      `SCENEWORKS_WAN_USER_LORA` points at one; otherwise the low-noise Lightning half doubles
///      as a valid single-file plain-LoRA fixture (same PEFT `diffusion_model.`-prefixed keys),
///      still exercising the plain-LoRA additive install onto the packed experts.
///
/// Peak RSS is sampled around each generation and printed so the run's log is the memory evidence
/// (a dequant-then-merge of a 14B bf16 expert would spike ~28 GB/expert; the packed q4 path holds
/// far below that). `#[ignore]` — needs the cached q4 tier + the Lightning pair + a Metal device.
///
/// ```sh
/// # tiers + Lightning are auto-discovered in ~/.cache/huggingface; override the tier root with
/// # SCENEWORKS_WAN_T2V_14B_TIER_OUT and a user LoRA with SCENEWORKS_WAN_USER_LORA if desired.
/// cargo test -p sceneworks-worker --release --lib \
///   wan_a14b_lightning_additive_ondevice -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan2.2 T2V-A14B q4 tier + Lightning pair; run manually on a Mac (sc-10049)"]
#[test]
fn wan_a14b_lightning_additive_ondevice() {
    const ENGINE: &str = "wan2_2_t2v_14b";
    let Some(tier) = wan_a14b_cached_tier(
        "SCENEWORKS_WAN_T2V_14B_TIER_OUT",
        "SceneWorks/wan2.2-t2v-a14b-mlx",
        "q4",
    ) else {
        eprintln!(
            "skipping wan_a14b_lightning_additive_ondevice: no complete q4 T2V-A14B tier in \
                 ~/.cache/huggingface (or SCENEWORKS_WAN_T2V_14B_TIER_OUT/q4)"
        );
        return;
    };
    let Some((light_high, light_low)) = wan_a14b_cached_lightning(ENGINE) else {
        eprintln!(
            "skipping wan_a14b_lightning_additive_ondevice: Lightning distill pair not cached"
        );
        return;
    };
    eprintln!("q4 tier   → {}", tier.display());
    eprintln!("lightning → {}", light_high.display());

    // Cap the buffer cache so the sequential heavy loads release between sub-cases (sc-5567).
    mlx_rs::memory::set_cache_limit(0);

    // Sub-case common frame/coherence assertions.
    fn assert_coherent(label: &str, decoded: &DecodedVideo, steps: u32) {
        assert!(
            decoded.frames.len() > 1,
            "{label}: got {} frames, expected a multi-frame clip",
            decoded.frames.len()
        );
        assert!(steps > 0, "{label}: denoise progress streamed");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{label}: frames are RGB8-sized"
        );
        let f0 = &decoded.frames[0];
        let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
        let var = f0
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / f0.pixels.len() as f64;
        assert!(
            var.sqrt() > 3.0,
            "{label}: frame 0 looks degenerate (std {:.2}) — additive load likely broken",
            var.sqrt()
        );
        eprintln!(
            "{label}: OK — {} frames, {steps} steps, frame0 std {:.2}",
            decoded.frames.len(),
            var.sqrt()
        );
    }

    let run = |label: &str, adapters: Vec<AdapterSpec>, steps: u32, guidance: Option<f32>| {
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: ENGINE,
            model_dir: tier.clone(),
            // Pre-packed q4 tier: config.json is authoritative, so load with None (the worker
            // path does the same). Adapters install ADDITIVELY on the packed experts.
            quant: None,
            adapters,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width: 256,
            height: 256,
            frames: 5,
            fps: 16,
            steps: Some(steps),
            guidance,
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut n_steps = 0u32;
        let mut on_progress = |p: Progress| {
            if let Progress::Step { .. } = p {
                n_steps += 1;
            }
        };
        // Reset the MLX GPU allocator's peak counter so `get_peak_memory` below reports THIS
        // sub-case's peak — the number a dense dequant-then-merge would blow up (a 14B bf16
        // expert dequant is ~28 GB/expert; the packed q4 additive path stays far below that).
        mlx_rs::memory::reset_peak_memory();
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|e| panic!("{label} generation failed: {e:?}"));
        assert_coherent(label, &decoded, n_steps);
        let mlx_peak_gib = mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "{label}: MLX peak GPU mem {mlx_peak_gib:.2} GiB, process RSS {} MiB",
            peak_rss_mib()
        );
        // Guard: the packed q4 additive path must hold well under a dense dequant spike. Two
        // dequantized 14B bf16 experts alone would be ~56 GB resident; assert the MLX peak stays
        // below a generous 40 GB ceiling so a regression that dequantizes-then-merges is caught.
        assert!(
            mlx_peak_gib < 40.0,
            "{label}: MLX peak {mlx_peak_gib:.2} GiB — a dense per-expert dequant spike \
                 (~28 GB/expert) appears to have happened; the packed q4 additive path regressed"
        );
        mlx_rs::memory::clear_cache();
    };

    // 1) Lightning ON, 4 steps — the exact pre-epic model.rs:614 failure case, now additive.
    run(
        "q4+lightning(4-step)",
        vec![
            moe_adapter(light_high.clone(), 1.0, AdapterKind::Lora, MoeExpert::High),
            moe_adapter(light_low.clone(), 1.0, AdapterKind::Lora, MoeExpert::Low),
        ],
        4,
        None,
    );

    // 2) Lightning OFF — native multi-step CFG on the packed tier (regression guard).
    run("q4+lightning-off(6-step CFG)", Vec::new(), 6, Some(5.0));

    // 3) Lightning ON + a plain single-file user LoRA (additive user-LoRA on a packed base).
    let user_lora = std::env::var("SCENEWORKS_WAN_USER_LORA")
        .ok()
        .map(|s| PathBuf::from(s.trim().to_owned()))
        .filter(|p| p.is_file())
        // Fall back to the low-noise Lightning half as a valid single-file plain LoRA fixture.
        .unwrap_or_else(|| light_low.clone());
    eprintln!("user LoRA → {}", user_lora.display());
    run(
        "q4+lightning+userLoRA",
        vec![
            moe_adapter(light_high.clone(), 1.0, AdapterKind::Lora, MoeExpert::High),
            moe_adapter(light_low.clone(), 1.0, AdapterKind::Lora, MoeExpert::Low),
            // Single-file → shared across both experts (moe_expert: None), the plain-LoRA
            // additive path (the exact user-Civitai-LoRA shape).
            AdapterSpec::new(user_lora, 0.6, AdapterKind::Lora),
        ],
        4,
        None,
    );
}

/// Best-effort current-process resident set size in MiB, read dependency-free via `ps -o rss=`
/// (macOS reports RSS in KiB). Used only to print memory evidence in the on-device tests — a
/// coarse guard that the packed q4 additive path holds well below a dense per-expert dequant
/// spike (~28 GB/expert). Returns 0 when `ps` is unavailable / unparsable (evidence-only).
#[cfg(target_os = "macos")]
fn peak_rss_mib() -> u64 {
    let pid = std::process::id().to_string();
    let out = match std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => return 0,
    };
    String::from_utf8_lossy(&out)
        .trim()
        .parse::<u64>()
        .map(|kib| kib / 1024)
        .unwrap_or(0)
}

/// **sc-10049 (epic 10043) — the I2V sibling of [`wan_a14b_lightning_additive_ondevice`].**
/// Drives the REAL Wan2.2 **I2V**-A14B **q4** tier with a synthesized start frame + the Lightning
/// distill pair installed additively on the packed experts. Same close-the-loop assertion as the
/// T2V case (the pre-epic `model.rs:614` rejection is gone; a 4-step distilled clip completes with
/// the base staying packed), for the image-to-video engine. A start image is REQUIRED for I2V, so
/// the test builds a small gradient RGB8 frame (no asset store needed) and passes it as the
/// `Conditioning::Reference`. Prints the MLX GPU peak per sub-case as the memory evidence and
/// asserts it holds below the dense-dequant ceiling. `#[ignore]` — needs the cached I2V q4 tier +
/// the I2V Lightning pair + a Metal device.
///
/// ```sh
/// cargo test -p sceneworks-worker --release --lib \
///   wan_i2v_a14b_lightning_additive_ondevice -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan2.2 I2V-A14B q4 tier + Lightning pair; run manually on a Mac (sc-10049)"]
#[test]
fn wan_i2v_a14b_lightning_additive_ondevice() {
    const ENGINE: &str = "wan2_2_i2v_14b";
    let Some(tier) = wan_a14b_cached_tier(
        "SCENEWORKS_WAN_I2V_14B_TIER_OUT",
        "SceneWorks/wan2.2-i2v-a14b-mlx",
        "q4",
    ) else {
        eprintln!(
            "skipping wan_i2v_a14b_lightning_additive_ondevice: no complete q4 I2V-A14B tier"
        );
        return;
    };
    let Some((light_high, light_low)) = wan_a14b_cached_lightning(ENGINE) else {
        eprintln!("skipping wan_i2v_a14b_lightning_additive_ondevice: Lightning pair not cached");
        return;
    };
    eprintln!("q4 I2V tier → {}", tier.display());

    mlx_rs::memory::set_cache_limit(0);

    // A small non-flat start frame (diagonal gradient) so the VAE-encoded `y` is meaningful.
    let (w, h) = (256u32, 256u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    let start = gen_core::Image {
        width: w,
        height: h,
        pixels,
    };
    let conditioning = vec![Conditioning::Reference {
        image: start,
        strength: None,
    }];

    let run = |label: &str, adapters: Vec<AdapterSpec>, steps: u32, guidance: Option<f32>| {
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: ENGINE,
            model_dir: tier.clone(),
            quant: None,
            adapters,
            conditioning: conditioning.clone(),
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width: w,
            height: h,
            frames: 5,
            fps: 16,
            steps: Some(steps),
            guidance,
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut n_steps = 0u32;
        let mut on_progress = |p: Progress| {
            if let Progress::Step { .. } = p {
                n_steps += 1;
            }
        };
        mlx_rs::memory::reset_peak_memory();
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|e| panic!("{label} I2V generation failed: {e:?}"));
        assert!(
            decoded.frames.len() > 1,
            "{label}: expected a multi-frame clip, got {}",
            decoded.frames.len()
        );
        assert!(n_steps > 0, "{label}: denoise progress streamed");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{label}: frames RGB8-sized"
        );
        let f0 = &decoded.frames[0];
        let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
        let var = f0
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / f0.pixels.len() as f64;
        assert!(
            var.sqrt() > 3.0,
            "{label}: frame 0 degenerate (std {:.2})",
            var.sqrt()
        );
        let mlx_peak_gib = mlx_rs::memory::get_peak_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "{label}: OK — {} frames, {n_steps} steps, std {:.2}, MLX peak {mlx_peak_gib:.2} GiB",
            decoded.frames.len(),
            var.sqrt()
        );
        assert!(
            mlx_peak_gib < 40.0,
            "{label}: MLX peak {mlx_peak_gib:.2} GiB — dense dequant spike; q4 additive regressed"
        );
        mlx_rs::memory::clear_cache();
    };

    // I2V q4 Lightning ON (4-step) — the close-the-loop case for the image-to-video engine.
    run(
        "i2v-q4+lightning(4-step)",
        vec![
            moe_adapter(light_high, 1.0, AdapterKind::Lora, MoeExpert::High),
            moe_adapter(light_low, 1.0, AdapterKind::Lora, MoeExpert::Low),
        ],
        4,
        None,
    );
}

/// Real in-process load+generate verification for the Wan2.2 **TI2V-5B** quant-matrix tiers
/// (sc-9941, epic 8506) — the single-expert sibling of [`wan_t2v_14b_tier_real_weights`]. For each
/// tier subdir present under `$SCENEWORKS_WAN_TI2V_5B_TIER_OUT` (`bf16/`/`q8/`/`q4/`, the
/// wan_ti2v_5b_tier_build output), loads the tier through the SAME engine path a job uses
/// (`run_video_generation`, `quant: None` — the pre-packed config.json is authoritative) and
/// denoises a tiny text-to-video clip, asserting the frames come back RGB8-sized and NON-degenerate
/// (real pixel variance, so a broken packed load surfaces instead of silent noise). This is the
/// on-device evidence that every hosted 5B tier loads packed (single `model.safetensors`
/// transformer) with no install-time convert peak. Runs a few CFG steps so it is self-contained;
/// output is a coherence check, not a quality bar. `#[ignore]` — needs the built tiers + a Metal
/// device; the cache limit is capped so loading bf16 then q8 then q4 in one process does not
/// accumulate residue (sc-5567).
///
/// ```sh
/// export SCENEWORKS_WAN_TI2V_5B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
/// cargo test -p sceneworks-worker --release --lib wan_ti2v_5b_tier_real_weights -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[ignore = "loads the built Wan2.2 TI2V-5B tiers; run manually on a Mac where they are present"]
#[test]
fn wan_ti2v_5b_tier_real_weights() {
    let Some(root) = std::env::var_os("SCENEWORKS_WAN_TI2V_5B_TIER_OUT").map(PathBuf::from) else {
        eprintln!("skipping wan_ti2v_5b_tier_real_weights: set SCENEWORKS_WAN_TI2V_5B_TIER_OUT");
        return;
    };
    // Cap the buffer cache so three sequential loads release between tiers (sc-5567).
    mlx_rs::memory::set_cache_limit(0);
    let mut verified = 0;
    for tier in ["bf16", "q8", "q4"] {
        let dir = root.join(tier);
        if !wan_tier_is_complete(&dir, WAN_TI2V_5B_TIER_FILES) {
            eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
            continue;
        }
        eprintln!("verifying {tier} tier → {}", dir.display());
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_ti2v_5b",
            model_dir: dir.clone(),
            // Pre-packed tier: config.json carries the quant; the engine reconstructs the
            // transformer packed and rejects a conflicting override, so load with None (the worker
            // path does the same in resolve_wan_tier_dir_and_quant).
            quant: None,
            prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
            width: 256,
            height: 256,
            frames: 5,
            fps: 24,
            steps: Some(8),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|e| panic!("{tier} tier TI2V-5B generation failed: {e:?}"));
        assert!(
            decoded.frames.len() > 1,
            "{tier}: got {} frames, expected a multi-frame clip",
            decoded.frames.len()
        );
        assert!(steps > 0, "{tier}: denoise progress streamed");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{tier}: frames are RGB8-sized"
        );
        // Non-degenerate: a broken packed load tends to decode to a flat/NaN frame. Require real
        // spatial variance in the first frame.
        let f0 = &decoded.frames[0];
        let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
        let var = f0
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / f0.pixels.len() as f64;
        assert!(
            var.sqrt() > 3.0,
            "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
            var.sqrt()
        );
        mlx_rs::memory::clear_cache();
        verified += 1;
    }
    assert!(
        verified > 0,
        "no tier subdirs found under SCENEWORKS_WAN_TI2V_5B_TIER_OUT"
    );
    eprintln!("verified {verified} tier(s)");
}

/// Real in-process load+generate verification for the Wan2.2 **I2V-A14B** quant-matrix tiers
/// (sc-9943, epic 8506) — the image→video sibling of [`wan_t2v_14b_tier_real_weights`]. For each
/// tier subdir present under `$SCENEWORKS_WAN_I2V_14B_TIER_OUT` (`bf16/`/`q8/`/`q4/`, the
/// wan_i2v_14b_tier_build output), loads the tier through the SAME engine path a job uses
/// (`run_video_generation`, `quant: None` — the pre-packed config.json is authoritative) and,
/// conditioning on a synthetic START IMAGE (the I2V in_dim-36 image-concat path a T2V run does not
/// exercise), denoises a tiny clip — asserting the frames come back RGB8-sized and NON-degenerate
/// (real pixel variance, so a broken packed load or a mis-wired image-cond path surfaces instead of
/// silent noise). This is the on-device evidence that every hosted I2V tier loads packed with no
/// install-time convert peak AND drives the image conditioning. Runs a few CFG steps WITHOUT the
/// Lightning distill so it is self-contained (no LoRA download); output is a coherence check, not a
/// quality bar. `#[ignore]` — needs the built tiers (~28–69 GB each) + a big-memory Mac; the cache
/// limit is capped so loading bf16 then q8 then q4 in one process does not accumulate residue
/// (sc-5567).
///
/// ```sh
/// export SCENEWORKS_WAN_I2V_14B_TIER_OUT=<tier-out-root>   # holds bf16/ q8/ q4/
/// cargo test -p sceneworks-worker --release --lib wan_i2v_14b_tier_real_weights -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[ignore = "loads the built Wan2.2 I2V-A14B tiers; run manually on a Mac where they are present"]
#[test]
fn wan_i2v_14b_tier_real_weights() {
    let Some(root) = std::env::var_os("SCENEWORKS_WAN_I2V_14B_TIER_OUT").map(PathBuf::from) else {
        eprintln!("skipping wan_i2v_14b_tier_real_weights: set SCENEWORKS_WAN_I2V_14B_TIER_OUT");
        return;
    };
    const W: u32 = 256;
    const H: u32 = 256;
    // A non-degenerate synthetic start frame (a smooth two-axis gradient) so the image-concat
    // conditioning carries real content the engine can propagate — fit through the SAME
    // `fit_engine_image` the I2V resolve path calls, so the reference reaches the engine exactly
    // as a job's would.
    let source = {
        let mut pixels = Vec::with_capacity((W * H * 3) as usize);
        for y in 0..H {
            for x in 0..W {
                pixels.push((x * 255 / W) as u8);
                pixels.push((y * 255 / H) as u8);
                pixels.push(128);
            }
        }
        crate::image_jobs::fit_engine_image(
            Image {
                width: W,
                height: H,
                pixels,
            },
            W,
            H,
            "pad",
        )
        .expect("fit synthetic I2V start frame")
    };
    // Cap the buffer cache so three sequential heavy loads release between tiers (sc-5567).
    mlx_rs::memory::set_cache_limit(0);
    let mut verified = 0;
    for tier in ["bf16", "q8", "q4"] {
        let dir = root.join(tier);
        if !wan_tier_is_complete(&dir, WAN_A14B_TIER_FILES) {
            eprintln!("skipping {tier}: {} is not a complete tier", dir.display());
            continue;
        }
        eprintln!("verifying {tier} tier → {}", dir.display());
        let input = VideoGenInput {
            sampler: None,
            scheduler: None,
            engine_id: "wan2_2_i2v_14b",
            model_dir: dir.clone(),
            // Pre-packed tier: config.json carries the quant; the engine reconstructs the experts
            // packed and rejects a conflicting override, so load with None (the worker path does
            // the same in resolve_wan_tier_dir_and_quant).
            quant: None,
            // The I2V-defining input: a source frame the engine VAE-encodes into its channel-concat
            // conditioning (the in_dim-36 path). Without it the I2V engine errors, so this also
            // proves the image-cond wiring on each packed tier.
            conditioning: vec![Conditioning::Reference {
                image: source.clone(),
                strength: None,
            }],
            prompt: "the camera slowly pushes in, cinematic".to_owned(),
            // A few CFG steps WITHOUT the Lightning distill — enough to exercise the packed
            // experts + VAE decode and produce non-degenerate frames for a load check.
            width: W,
            height: H,
            frames: 5,
            fps: 16,
            steps: Some(6),
            guidance: Some(5.0),
            seed: 7,
            ..VideoGenInput::default()
        };
        let cancel = CancelFlag::new();
        let mut steps = 0u32;
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { .. } = progress {
                steps += 1;
            }
        };
        let decoded = run_video_generation(input, &cancel, &mut on_progress)
            .unwrap_or_else(|e| panic!("{tier} tier I2V generation failed: {e:?}"));
        // The engine's temporal VAE fixes the decoded frame count (a 5-latent-frame request
        // decodes to a multi-frame clip for the A14B); assert a real multi-frame clip.
        assert!(
            decoded.frames.len() > 1,
            "{tier}: got {} frames, expected a multi-frame clip",
            decoded.frames.len()
        );
        assert!(steps > 0, "{tier}: denoise progress streamed");
        assert!(
            decoded
                .frames
                .iter()
                .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
            "{tier}: frames are RGB8-sized"
        );
        // Non-degenerate: a broken packed load or dropped image-cond tends to decode to a
        // flat/NaN frame. Require real spatial variance in the first frame.
        let f0 = &decoded.frames[0];
        let mean = f0.pixels.iter().map(|&p| p as f64).sum::<f64>() / f0.pixels.len() as f64;
        let var = f0
            .pixels
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / f0.pixels.len() as f64;
        assert!(
            var.sqrt() > 3.0,
            "{tier}: frame 0 looks degenerate (std {:.2}) — packed load likely broken",
            var.sqrt()
        );
        mlx_rs::memory::clear_cache();
        verified += 1;
    }
    assert!(
        verified > 0,
        "no tier subdirs found under SCENEWORKS_WAN_I2V_14B_TIER_OUT"
    );
    eprintln!("verified {verified} tier(s)");
}

/// Real-weight image→video fit smoke (sc-6139): proves the Crop/Pad pre-fit reaches the
/// engine and the engine renders the chosen off-aspect output without re-stretching. A
/// solid-color SQUARE source is fit to a 448×256 landscape via the SAME `fit_engine_image`
/// the I2V resolve paths call — `pad` letterboxes (black side bars), `crop` fills+trims —
/// asserted deterministically; then the pad-fit reference drives a tiny TI2V-5B I2V clip and
/// the decoded frames are asserted to be exactly 448×256 (a re-stretch would change the
/// aspect). Frame 0 is dumped to `$TMPDIR/sc6139_i2v_pad_frame0.png` for a visual bar check.
/// `#[ignore]` — needs the converted TI2V-5B weights; run manually on a Mac where present.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Wan2.2 TI2V-5B weights; run manually on a Mac with them present"]
#[test]
fn image_to_video_fit_real_weights() {
    // Off-aspect output: a square source must letterbox (pad) or fill+trim (crop).
    const OUT_W: u32 = 448; // 14·32
    const OUT_H: u32 = 256; // 8·32
                            // A solid non-black square so pad's black bars are unambiguous against the content.
    let mk_square = || Image {
        width: 256,
        height: 256,
        pixels: [200u8, 80, 40].repeat((256 * 256) as usize),
    };
    let is_black_col = |img: &Image, x: u32| -> bool {
        (0..img.height).all(|y| {
            let i = ((y * img.width + x) * 3) as usize;
            img.pixels[i] == 0 && img.pixels[i + 1] == 0 && img.pixels[i + 2] == 0
        })
    };

    // PAD: contained (fit height), centered → black bars on the left/right edges.
    let padded =
        crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "pad").expect("pad fit");
    assert_eq!((padded.width, padded.height), (OUT_W, OUT_H));
    assert!(is_black_col(&padded, 0), "pad: left edge is a black bar");
    assert!(
        is_black_col(&padded, OUT_W - 1),
        "pad: right edge is a black bar"
    );
    let ci = ((OUT_H / 2 * OUT_W + OUT_W / 2) * 3) as usize;
    assert_eq!(
        &padded.pixels[ci..ci + 3],
        &[200, 80, 40],
        "pad: center keeps the source color"
    );

    // CROP: covered (fit width), center-cropped → fully filled, no black border.
    let cropped =
        crate::image_jobs::fit_engine_image(mk_square(), OUT_W, OUT_H, "crop").expect("crop fit");
    assert_eq!((cropped.width, cropped.height), (OUT_W, OUT_H));
    assert!(
        !is_black_col(&cropped, 0),
        "crop: left edge is filled, not a bar"
    );
    assert!(
        !is_black_col(&cropped, OUT_W - 1),
        "crop: right edge is filled, not a bar"
    );

    // Real-weight run: the pad-fit reference drives a tiny TI2V-5B I2V clip. If the engine
    // re-stretched the reference it could not land at the output aspect; asserting the decoded
    // frames are exactly OUT_W×OUT_H confirms the pre-fit reference is consumed as-is.
    let Some(model_dir) = wan_5b_dir() else {
        eprintln!("skipping image_to_video_fit_real_weights: no converted TI2V-5B dir found");
        return;
    };
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan2_2_ti2v_5b",
        model_dir,
        conditioning: vec![Conditioning::Reference {
            image: padded,
            strength: None,
        }],
        prompt: "a slow gentle camera move, cinematic".to_owned(),
        width: OUT_W,
        height: OUT_H,
        frames: 5,
        fps: 16,
        steps: Some(8),
        seed: 7,
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded = run_video_generation(input, &cancel, &mut on_progress)
        .expect("TI2V-5B I2V generation from a pad-fit square reference");
    assert_eq!(decoded.frames.len(), 5, "5 frames (1 + 4·1)");
    assert!(steps > 0, "denoise progress streamed");
    for f in &decoded.frames {
        assert_eq!(
            (f.width, f.height),
            (OUT_W, OUT_H),
            "engine renders the requested off-aspect output (no re-stretch)"
        );
        assert_eq!(f.pixels.len(), (f.width * f.height * 3) as usize);
    }
    // Dump frame 0 so the pad letterbox is visually verifiable.
    let frame0 = &decoded.frames[0];
    if let Some(buf) = image::RgbImage::from_raw(frame0.width, frame0.height, frame0.pixels.clone())
    {
        let out = std::env::temp_dir().join("sc6139_i2v_pad_frame0.png");
        let _ = buf.save(&out);
        eprintln!("wrote pad-fit I2V frame 0 to {}", out.display());
    }
}

/// Real-weight perf probe for the TI2V-5B (sc-4997): measures the load / DiT-denoise /
/// VAE-decode wall-clock split at a configurable resolution, frame count, step count, and
/// CFG — to ground the "under 10 min" target on real numbers instead of estimates. Env-driven
/// so configs run without recompiling. MUST be `--release` (debug MLX timing is meaningless):
///   SCENEWORKS_MLX_WAN5B_DIR=~/.cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16 \
///   WAN_TIMING_W=1280 WAN_TIMING_H=720 WAN_TIMING_FRAMES=121 WAN_TIMING_STEPS=20 WAN_TIMING_CFG=on \
///   cargo test -p sceneworks-worker --release --lib wan_5b_timing -- --ignored --nocapture
/// CFG: `off`/`1` ⇒ guide 1.0 (no CFG); a number ⇒ that scale; `on`/unset ⇒ engine default (5.0).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight perf probe; needs the converted TI2V-5B snapshot + a Metal device"]
fn wan_5b_timing() {
    use std::time::Instant;
    let Some(model_dir) = wan_5b_dir() else {
        eprintln!("skipping wan_5b_timing: no converted TI2V-5B dir found");
        return;
    };
    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(d)
    };
    let width = env_u32("WAN_TIMING_W", 1280);
    let height = env_u32("WAN_TIMING_H", 720);
    let frames = env_u32("WAN_TIMING_FRAMES", 121);
    let steps = env_u32("WAN_TIMING_STEPS", 20);
    let guidance = match std::env::var("WAN_TIMING_CFG")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("off") | Some("1") | Some("1.0") => Some(1.0_f32),
        Some("on") | Some("") | None => None,
        Some(other) => other.parse().ok(),
    };
    // Optional MLX memory-limit override (GB): lowers `get_memory_limit()` so the z48 vae22
    // decode tile planner (`auto_tiling_budgeted`) picks SMALLER tiles — to test whether a
    // smaller decode working-set is faster than the default "largest tile under budget" (sc-5089).
    let mem_limit_gb = env_u32("WAN_TIMING_MEM_LIMIT_GB", 0);
    if mem_limit_gb > 0 {
        let prev = mlx_rs::memory::set_memory_limit((mem_limit_gb as usize) * 1024 * 1024 * 1024);
        eprintln!(
            "WAN5B_TIMING mem_limit set to {mem_limit_gb} GB (was {:.0} GB)",
            prev as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    mlx_rs::memory::reset_peak_memory();
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan2_2_ti2v_5b",
        model_dir,
        prompt: "a calm ocean wave at sunset, cinematic".to_owned(),
        width,
        height,
        frames,
        fps: 24,
        steps: Some(steps),
        guidance,
        seed: 7,
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let start = Instant::now();
    let mut first_step: Option<Instant> = None;
    let mut last_step: Option<Instant> = None;
    let mut decode_at: Option<Instant> = None;
    let mut dit_peak_bytes: Option<usize> = None;
    let mut on_progress = |progress: Progress| match progress {
        Progress::Step { .. } => {
            let now = Instant::now();
            first_step.get_or_insert(now);
            last_step = Some(now);
        }
        Progress::Decoding => {
            decode_at.get_or_insert(Instant::now());
            // Peak so far = the DiT-denoise peak (the VAE decode is the *next* stage).
            dit_peak_bytes.get_or_insert(mlx_rs::memory::get_peak_memory());
        }
        Progress::Loading(_) => {}
    };
    let decoded = run_video_generation(input, &cancel, &mut on_progress).expect("5B generation");
    let end = Instant::now();
    let total_peak_bytes = mlx_rs::memory::get_peak_memory();
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let secs = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64();
    let fs = first_step.unwrap_or(start);
    let dec = decode_at.or(last_step).unwrap_or(end);
    eprintln!(
        "WAN5B_TIMING {width}x{height} frames={frames} steps={steps} cfg={} => \
             total={:.1}s load={:.1}s dit={:.1}s vae={:.1}s | peak_dit={:.0}GB peak_total={:.0}GB \
             out_frames={}",
        guidance.map_or_else(|| "5.0(default)".to_owned(), |g| format!("{g}")),
        secs(start, end),
        secs(start, fs),
        secs(fs, dec),
        secs(dec, end),
        gib(dit_peak_bytes.unwrap_or(0)),
        gib(total_peak_bytes),
        decoded.frames.len(),
    );
}

/// LTX model-id → engine-id mapping (both base + eros load the one engine model).
#[cfg(target_os = "macos")]
#[test]
fn ltx_engine_id_maps_base_and_eros() {
    assert_eq!(ltx_engine_id("ltx_2_3"), Some("ltx_2_3"));
    assert_eq!(ltx_engine_id("ltx_2_3_eros"), Some("ltx_2_3"));
    assert_eq!(ltx_engine_id("wan_2_2"), None);
    assert_eq!(ltx_engine_id("z_image_turbo"), None);
}

/// SceneWorks bundle resolution (sc-5608): `ltx_bundle_subdir` picks the requested quant subdir,
/// prefers a complete one over an incomplete sibling, and `bundled_ltx_gemma_dir` finds `gemma/`.
#[cfg(target_os = "macos")]
#[test]
fn ltx_bundle_subdir_picks_quant_and_finds_gemma() {
    fn write_complete_ltx_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for file in [
            "connector.safetensors",
            "transformer.safetensors",
            "upsampler.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
        ] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
    }
    let root_guard = tempfile::Builder::new()
        .prefix("sw_ltx_bundle_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let (q4, q8, bf16) = (root.join("q4"), root.join("q8"), root.join("bf16"));
    write_complete_ltx_dir(&q4);
    write_complete_ltx_dir(&q8);
    write_complete_ltx_dir(&bf16);
    write_complete_gemma_dir(&root.join("gemma"));

    // Each tier order prefers its own subdir (default q4, mlxQuantize:8 q8, mlxQuantize<=0 bf16).
    assert_eq!(
        ltx_bundle_subdir(root, &["q4", "q8"]).as_deref(),
        Some(q4.as_path())
    );
    assert_eq!(
        ltx_bundle_subdir(root, &["q8", "q4"]).as_deref(),
        Some(q8.as_path())
    );
    assert_eq!(
        ltx_bundle_subdir(root, &["bf16", "q8", "q4"]).as_deref(),
        Some(bf16.as_path())
    );
    // The default order never loads the huge bf16 tier even when present.
    assert_eq!(
        ltx_bundle_subdir(root, &["q4", "q8"]).as_deref(),
        Some(q4.as_path())
    );

    // The gemma encoder is found as a sibling of the loaded quant dir.
    assert_eq!(
        bundled_ltx_gemma_dir(&q4).as_deref(),
        Some(root.join("gemma").as_path())
    );

    // An incomplete preferred subdir falls back to the complete sibling (q8 → q4, bf16 → q8).
    std::fs::remove_file(q8.join("vocoder.safetensors")).unwrap();
    assert_eq!(
        ltx_bundle_subdir(root, &["q8", "q4"]).as_deref(),
        Some(q4.as_path())
    );
    std::fs::remove_file(bf16.join("vocoder.safetensors")).unwrap();
    assert_eq!(
        ltx_bundle_subdir(root, &["bf16", "q8", "q4"]).as_deref(),
        Some(q4.as_path())
    );

    // No complete subdir → None; no gemma sibling → None.
    let bare_guard = tempfile::Builder::new()
        .prefix("sw_ltx_bare_")
        .tempdir()
        .expect("temp dir");
    let bare = bare_guard.path();
    std::fs::create_dir_all(bare.join("q4")).unwrap();
    assert!(ltx_bundle_subdir(bare, &["q4", "q8"]).is_none());
    assert!(bundled_ltx_gemma_dir(&bare.join("q4")).is_none());
}

/// A pre-hotfix install keeps q4/q8 in the old snapshot while bf16 lands in the bumped snapshot.
/// The requested tier must win across revisions; otherwise a bf16 request silently runs q8.
#[cfg(target_os = "macos")]
#[test]
fn ltx_bundle_subdir_across_revisions_prefers_tier_over_revision() {
    fn write_complete_ltx_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for file in [
            "connector.safetensors",
            "transformer.safetensors",
            "upsampler.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
        ] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
    }

    let cache = tempfile::tempdir().expect("temp cache");
    let snapshots = cache.path().join("snapshots");
    let old = snapshots.join(LTX_BUNDLE_PRE_BF16_REVISION);
    let bumped = snapshots.join(LTX_BUNDLE_REVISION);
    write_complete_ltx_dir(&old.join("q4"));
    write_complete_ltx_dir(&old.join("q8"));
    write_complete_ltx_dir(&bumped.join("bf16"));

    assert_eq!(
        ltx_bundle_subdir(&old, &["bf16", "q8", "q4"]).as_deref(),
        Some(old.join("q8").as_path()),
        "precondition: a single-snapshot lookup downgrades bf16 to q8"
    );
    assert_eq!(
        ltx_bundle_subdir_across_revisions(&old, &["bf16", "q8", "q4"]).as_deref(),
        Some(bumped.join("bf16").as_path()),
        "bf16 in a sibling revision must beat q8 in the selected revision"
    );
    assert_eq!(
        ltx_bundle_subdir_across_revisions(&bumped, &["q4", "q8"]).as_deref(),
        Some(old.join("q4").as_path()),
        "the default tier remains reachable in the pre-hotfix snapshot"
    );

    write_complete_ltx_dir(&bumped.join("q4"));
    assert_eq!(
        ltx_bundle_subdir_across_revisions(&old, &["q4"]).as_deref(),
        Some(bumped.join("q4").as_path()),
        "the immutable current pin wins when both compatible revisions contain the tier"
    );

    let unrelated = snapshots.join("ffffffffffffffffffffffffffffffffffffffff");
    write_complete_ltx_dir(&unrelated.join("bf16"));
    std::fs::remove_file(bumped.join("bf16/vocoder.safetensors")).unwrap();
    assert_eq!(
        ltx_bundle_subdir_across_revisions(&unrelated, &["bf16", "q8", "q4"]).as_deref(),
        Some(old.join("q8").as_path()),
        "an arbitrary cached revision must never bypass the immutable pin or its proven parent"
    );

    let flat = tempfile::tempdir().expect("flat cache");
    write_complete_ltx_dir(&flat.path().join("sibling").join("bf16"));
    write_complete_ltx_dir(&flat.path().join("selected").join("q4"));
    assert_eq!(
        ltx_bundle_subdir_across_revisions(&flat.path().join("selected"), &["bf16", "q8", "q4"])
            .as_deref(),
        Some(flat.path().join("selected").join("q4").as_path()),
        "legacy flat layouts must not scan unrelated sibling directories"
    );
}

/// Build the exact split cache created when an existing install downloads bf16 after the pin bump.
/// No refs/main is written, so the real resolver's most-files fallback selects the old snapshot.
#[cfg(target_os = "macos")]
fn ltx_split_revision_hub(tag: &str) -> tempfile::TempDir {
    fn write_complete_ltx_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for file in [
            "connector.safetensors",
            "transformer.safetensors",
            "upsampler.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
        ] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
    }

    let hub = tempfile::Builder::new()
        .prefix(&format!("sw_ltx_hub_{tag}_"))
        .tempdir()
        .expect("temp hub");
    let snapshots = hub
        .path()
        .join(format!(
            "models--{}",
            sceneworks_core::hf_home::safe_repo_dir_name(LTX_BUNDLE_REPO).expect("LTX repo slug")
        ))
        .join("snapshots");
    let old = snapshots.join(LTX_BUNDLE_PRE_BF16_REVISION);
    write_complete_ltx_dir(&old.join("q4"));
    write_complete_ltx_dir(&old.join("q8"));
    write_complete_gemma_dir(&old.join("gemma"));
    write_complete_ltx_dir(&snapshots.join(LTX_BUNDLE_REVISION).join("bf16"));
    hub
}

#[cfg(target_os = "macos")]
fn ltx_with_hermetic_cache<T>(hub: &Path, body: impl FnOnce() -> T) -> T {
    temp_env_vars(
        &[
            ("HF_HUB_CACHE", hub.to_str().expect("utf-8 hub")),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            ("SCENEWORKS_MLX_LTX_DIR", ""),
            ("SCENEWORKS_MLX_LTX_EROS_DIR", ""),
            ("LTX_GEMMA_DIR", ""),
        ],
        body,
    )
}

/// Drive the production resolver over the split cache, rather than only testing its helper.
#[cfg(target_os = "macos")]
#[test]
fn resolve_ltx_model_dir_reaches_bf16_in_a_sibling_revision() {
    let hub = ltx_split_revision_hub("resolve");
    let data = tempfile::tempdir().expect("temp data");
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        ..offline_settings()
    };
    let request_for = |bits: Option<i64>| {
        let advanced = bits.map_or_else(|| json!({}), |bits| json!({ "mlxQuantize": bits }));
        request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": advanced
        }))
    };
    let snapshots = hub
        .path()
        .join(format!(
            "models--{}",
            sceneworks_core::hf_home::safe_repo_dir_name(LTX_BUNDLE_REPO).expect("slug")
        ))
        .join("snapshots");

    let (bf16, q8, default) = ltx_with_hermetic_cache(hub.path(), || {
        (
            resolve_ltx_model_dir(&settings, &request_for(Some(0))),
            resolve_ltx_model_dir(&settings, &request_for(Some(8))),
            resolve_ltx_model_dir(&settings, &request_for(None)),
        )
    });

    assert_eq!(
        bf16.expect("bf16 resolves"),
        snapshots.join(LTX_BUNDLE_REVISION).join("bf16")
    );
    let old = snapshots.join(LTX_BUNDLE_PRE_BF16_REVISION);
    assert_eq!(q8.expect("q8 resolves"), old.join("q8"));
    assert_eq!(default.expect("default resolves"), old.join("q4"));
}

/// A complete tier in any revision is already provisioned. Presence checks must not contact the Hub
/// for it, while a tier missing from every revision must still attempt the fetch.
#[cfg(target_os = "macos")]
#[test]
fn ensure_ltx_tier_present_skips_fetch_for_a_sibling_revision() {
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(future)
    }

    let hub = ltx_split_revision_hub("ensure");
    let data = tempfile::tempdir().expect("temp data");
    let settings = Settings {
        data_dir: data.path().to_path_buf(),
        ..offline_settings()
    };
    let api = ApiClient::new(&settings);
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-ltx-tier-1",
        "type": "video_generate",
        "status": "running",
        "projectId": "p",
        "projectName": "P",
        "payload": { "model": "ltx_2_3" },
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-08-13T00:00:00Z",
        "updatedAt": "2026-08-13T00:00:00Z"
    }))
    .expect("job snapshot");
    let ltx = |bits: i64| {
        request(json!({
            "projectId": "p", "model": "ltx_2_3", "prompt": "x",
            "advanced": { "mlxQuantize": bits }
        }))
    };

    let (bf16, q8) = ltx_with_hermetic_cache(hub.path(), || {
        (
            block_on(ensure_ltx_bf16_present(&api, &settings, &job, &ltx(0))),
            block_on(ensure_ltx_q8_present(&api, &settings, &job, &ltx(8))),
        )
    });
    assert!(
        bf16.is_ok(),
        "bf16 in a sibling must not dial the Hub: {bf16:?}"
    );
    assert!(q8.is_ok(), "q8 in the selected snapshot is present: {q8:?}");

    let bare = tempfile::tempdir().expect("bare hub");
    let bare_snapshot = bare
        .path()
        .join(format!(
            "models--{}",
            sceneworks_core::hf_home::safe_repo_dir_name(LTX_BUNDLE_REPO).expect("slug")
        ))
        .join("snapshots")
        .join(LTX_BUNDLE_PRE_BF16_REVISION);
    write_complete_gemma_dir(&bare_snapshot.join("gemma"));
    let missing = ltx_with_hermetic_cache(bare.path(), || {
        block_on(ensure_ltx_bf16_present(&api, &settings, &job, &ltx(0)))
    });
    assert!(
        missing.is_err(),
        "bf16 absent from every revision must still attempt a fetch"
    );
}

/// Lay down a complete Gemma-3 text-encoder snapshot at `dir`: config, tokenizer, a two-shard
/// index, and the shards it maps. Mirrors the real bundle `gemma/` so the completeness +
/// eros-resolution tests are hermetic.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_tiny_safetensors(path: &Path) {
    let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.push(0);
    std::fs::write(path, bytes).unwrap();
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_complete_gemma_dir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), br#"{"model_type":"gemma3_text"}"#).unwrap();
    std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
    std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
    write_tiny_safetensors(&dir.join("model-00001-of-00002.safetensors"));
    write_tiny_safetensors(&dir.join("model-00002-of-00002.safetensors"));
}

#[cfg(target_os = "macos")]
fn advanced_object(value: Value) -> JsonObject {
    value.as_object().expect("advanced object").clone()
}

/// sc-13800: the public option list is generic/stable, but the operator-staged alternate is only
/// enumerated after the shared completeness resolver says it is available.
#[cfg(target_os = "macos")]
#[test]
fn ltx_text_encoder_options_hide_unstaged_alternate() {
    let default_only = ltx_text_encoder_options(false);
    assert_eq!(
        default_only
            .iter()
            .map(|option| option.id)
            .collect::<Vec<_>>(),
        vec![DEFAULT_TEXT_ENCODER_ID]
    );
    assert!(default_only[0].is_default);

    let staged = ltx_text_encoder_options(true);
    assert_eq!(
        staged.iter().map(|option| option.id).collect::<Vec<_>>(),
        vec![DEFAULT_TEXT_ENCODER_ID, AMORAL_TEXT_ENCODER_ID]
    );
    assert!(!staged[1].is_default);
}

/// sc-13800: the stable selector generalizes the raw boolean without weakening compatibility.
/// Unknown/non-string/conflicting choices are validation errors, never silent defaults.
#[cfg(target_os = "macos")]
#[test]
fn ltx_text_encoder_selector_is_strict_and_legacy_compatible() {
    assert_eq!(
        selected_ltx_text_encoder(&advanced_object(json!({}))).unwrap(),
        LtxTextEncoderSelection::Default
    );
    assert_eq!(
        selected_ltx_text_encoder(&advanced_object(json!({
            "textEncoderModel": AMORAL_TEXT_ENCODER_ID
        })))
        .unwrap(),
        LtxTextEncoderSelection::Amoral
    );
    assert_eq!(
        selected_ltx_text_encoder(&advanced_object(json!({
            "useUncensoredEnhancer": true
        })))
        .unwrap(),
        LtxTextEncoderSelection::Amoral
    );

    for bad in [
        json!({ "textEncoderModel": "unknown_encoder" }),
        json!({ "textEncoderModel": 7 }),
        json!({
            "textEncoderModel": DEFAULT_TEXT_ENCODER_ID,
            "useUncensoredEnhancer": true
        }),
        json!({
            "textEncoderModel": AMORAL_TEXT_ENCODER_ID,
            "useUncensoredEnhancer": false
        }),
        json!({ "useUncensoredEnhancer": "true" }),
    ] {
        assert!(
            matches!(
                selected_ltx_text_encoder(&advanced_object(bad)),
                Err(WorkerError::InvalidPayload(_))
            ),
            "invalid selector input must fail as a user-facing payload error"
        );
    }
}

/// A selected alternate must both be active (prompt enhancement on) and fully staged before model
/// load. The returned directory is the one later placed in LoadSpec.components, so this assertion is
/// mutation-sensitive to the non-default choice.
#[cfg(target_os = "macos")]
#[test]
fn selected_ltx_text_encoder_fails_early_or_stages_exact_alternate() {
    let advanced = advanced_object(json!({
        "enhancePrompt": true,
        "textEncoderModel": AMORAL_TEXT_ENCODER_ID
    }));
    let staged = PathBuf::from("/operator/staged/amoral-gemma");
    let (use_alternate, resolved) =
        resolve_selected_ltx_text_encoder(&advanced, Some(staged.clone())).unwrap();
    assert!(use_alternate);
    assert_eq!(resolved.as_deref(), Some(staged.as_path()));

    let missing = resolve_selected_ltx_text_encoder(&advanced, None)
        .expect_err("a selected but unstaged alternate must fail");
    assert!(matches!(missing, WorkerError::InvalidPayload(_)));
    assert!(missing.to_string().contains("not staged"));

    let disabled = advanced_object(json!({
        "enhancePrompt": false,
        "textEncoderModel": AMORAL_TEXT_ENCODER_ID
    }));
    let error = resolve_selected_ltx_text_encoder(&disabled, Some(staged))
        .expect_err("an alternate that would be ignored must not silently pass");
    assert!(error.to_string().contains("prompt enhancement is off"));

    let (use_alternate, resolved) =
        resolve_selected_ltx_text_encoder(&advanced_object(json!({})), None).unwrap();
    assert!(!use_alternate);
    assert!(resolved.is_none());
}

/// Keep selector rejection ahead of every expensive or externally visible phase, and keep the
/// resolved directory wired into the engine input. This source-order guard complements the pure
/// resolver tests because the async generation seam otherwise requires real LTX weights.
#[cfg(target_os = "macos")]
#[test]
fn generate_ltx_validates_and_stages_the_selector_before_side_effects() {
    let source = include_str!("ltx.rs");
    let generate = source
        .split_once("pub(super) async fn generate_ltx(")
        .expect("generate_ltx exists")
        .1;
    let selector = generate
        .find("resolve_selected_ltx_text_encoder(")
        .expect("selector resolution");
    for side_effect in [
        "resolve_video_clip_conditioning(",
        "ensure_ltx_q8_present(",
        "ensure_ltx_bf16_present(",
        "ensure_ltx_gemma_present(",
    ] {
        assert!(
            selector < generate.find(side_effect).expect("side-effect call"),
            "selector validation must precede {side_effect}"
        );
    }
    let input = generate
        .split_once("let input = VideoGenInput {")
        .expect("engine input")
        .1;
    assert!(
        input.contains("uncensored_enhancer_dir,"),
        "the resolved alternate directory must reach VideoGenInput"
    );
}

/// `ltx_gemma_dir_is_complete`: needs config + tokenizer + every shard a non-empty index maps
/// (or a lone single-file checkpoint); missing load inputs and malformed/empty maps all fail.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn ltx_gemma_completeness_requires_config_and_all_shards() {
    // The traversal case below plants a decoy at `root.parent()`, so the guard has to own the
    // PARENT as well — handing back the guard's own directory would put that decoy straight in the
    // shared temp root, where only a trailing `remove_file` (skipped on panic) took it away.
    let root_guard = tempfile::Builder::new()
        .prefix("sw_gemma_ok_")
        .tempdir()
        .expect("temp dir");
    let root_owned = root_guard.path().join("gemma");
    let root = root_owned.as_path();
    write_complete_gemma_dir(root);
    assert!(ltx_gemma_dir_is_complete(root));

    // A missing shard the index references → incomplete (a partial download must not pass).
    std::fs::remove_file(root.join("model-00002-of-00002.safetensors")).unwrap();
    assert!(!ltx_gemma_dir_is_complete(root));

    // Missing config.json → incomplete even with weights present.
    write_tiny_safetensors(&root.join("model-00002-of-00002.safetensors"));
    std::fs::remove_file(root.join("config.json")).unwrap();
    assert!(!ltx_gemma_dir_is_complete(root));

    // Missing tokenizer.json → incomplete even with config + all weights present.
    std::fs::write(root.join("config.json"), br#"{"model_type":"gemma3_text"}"#).unwrap();
    std::fs::remove_file(root.join("tokenizer.json")).unwrap();
    assert!(!ltx_gemma_dir_is_complete(root));

    // Empty or non-string weight maps must not pass vacuously.
    std::fs::write(root.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        br#"{"weight_map":{}}"#,
    )
    .unwrap();
    assert!(!ltx_gemma_dir_is_complete(root));
    std::fs::write(
        root.join("model.safetensors.index.json"),
        br#"{"weight_map":{"bad":42}}"#,
    )
    .unwrap();
    assert!(!ltx_gemma_dir_is_complete(root));

    // An index cannot escape its snapshot through either portable traversal syntax or an absolute
    // path, even when the target outside the Gemma dir is a structurally valid safetensors file.
    let outside_name = format!("sw_gemma_outside_{}.safetensors", Uuid::new_v4().simple());
    let outside = root.parent().unwrap().join(&outside_name);
    write_tiny_safetensors(&outside);
    for unsafe_shard in [
        format!("../{outside_name}"),
        format!(r"..\{outside_name}"),
        outside.display().to_string(),
    ] {
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map":{"bad":&unsafe_shard}}).to_string(),
        )
        .unwrap();
        assert!(
            !ltx_gemma_dir_is_complete(root),
            "unsafe indexed shard must be rejected: {unsafe_shard}"
        );
    }

    // A present but structurally invalid shard is just as incomplete as a missing shard.
    std::fs::write(
        root.join("model.safetensors.index.json"),
        br#"{"weight_map":{"bad":"model-00001-of-00002.safetensors"}}"#,
    )
    .unwrap();
    std::fs::write(
        root.join("model-00001-of-00002.safetensors"),
        b"placeholder",
    )
    .unwrap();
    assert!(
        !ltx_gemma_dir_is_complete(root),
        "filename-only shard placeholders must not be resolved by the worker"
    );

    // Single-file checkpoint (no index) is complete once config + tokenizer + weights exist.
    let single_guard = tempfile::Builder::new()
        .prefix("sw_gemma_1f_")
        .tempdir()
        .expect("temp dir");
    let single = single_guard.path();
    std::fs::write(
        single.join("config.json"),
        br#"{"model_type":"gemma3_text"}"#,
    )
    .unwrap();
    std::fs::write(
        single.join("tokenizer.json"),
        br#"{"model":{"type":"BPE"}}"#,
    )
    .unwrap();
    assert!(!ltx_gemma_dir_is_complete(single));
    std::fs::write(single.join("model.safetensors"), b"x").unwrap();
    assert!(
        !ltx_gemma_dir_is_complete(single),
        "filename-only placeholder weights must not be advertised as loadable"
    );
    write_tiny_safetensors(&single.join("model.safetensors"));
    assert!(ltx_gemma_dir_is_complete(single));
}

/// sc-14377: filtered HF downloads can split one managed bundle across revisions. The selected
/// Q8 tier may live in one snapshot while the complete Gemma co-requisite lives in another; the
/// base resolver must still pass that Gemma through `LoadSpec::text_encoder`. A partial Gemma
/// beside the selected tier must not shadow the complete sibling revision.
#[cfg(target_os = "macos")]
#[test]
fn bundled_ltx_gemma_resolves_across_split_snapshot_revisions() {
    let root_guard = tempfile::Builder::new()
        .prefix("sw_ltx_split_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let snapshots = root.join("models--SceneWorks--ltx-2.3-mlx/snapshots");
    let tier_snapshot = snapshots.join("tier-revision");
    let tier = tier_snapshot.join("q8");
    std::fs::create_dir_all(&tier).unwrap();

    // A torn co-requisite beside the selected tier cannot win.
    let partial = tier_snapshot.join("gemma");
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("config.json"), b"{}").unwrap();

    // The managed co-requisite is complete, but was downloaded at a different revision.
    let complete = snapshots.join("gemma-revision").join("gemma");
    write_complete_gemma_dir(&complete);

    assert_eq!(
        bundled_ltx_gemma_dir(&tier).as_deref(),
        Some(complete.as_path()),
        "a complete Gemma in another snapshot revision must satisfy the selected LTX tier"
    );

    // This fallback is deliberately limited to HF's snapshots/<revision>/<tier> shape: a local
    // conversion with no sibling Gemma must not scan unrelated directories.
    let local = root.join("models/mlx/ltx_2_3_base_q8");
    std::fs::create_dir_all(&local).unwrap();
    assert!(bundled_ltx_gemma_dir(&local).is_none());
}

/// sc-13664: the operator `$LTX_GEMMA_DIR` override now RIDES `LoadSpec::text_encoder` (the MLX LTX
/// provider deleted its own env read), so the worker resolves it to a path instead of returning
/// `None` to defer to the deleted fallback. Pure over the raw env value (no process-global mutation):
/// a complete dir resolves to `Some`, an incomplete dir → `None` (a bad override never shadows a good
/// bundled/cache gemma), and an unset override → `None`.
#[cfg(target_os = "macos")]
#[test]
fn ltx_gemma_override_path_requires_complete_dir() {
    let complete_guard = tempfile::Builder::new()
        .prefix("sw_gemma_ovr_ok_")
        .tempdir()
        .expect("temp dir");
    let complete = complete_guard.path();
    write_complete_gemma_dir(complete);
    assert_eq!(
        ltx_gemma_override_path(Some(complete.as_os_str().to_owned())).as_deref(),
        Some(complete),
        "a complete $LTX_GEMMA_DIR resolves to the TE path (rides the spec)"
    );

    // A set-but-incomplete override yields None so a good bundled/cache gemma still wins.
    std::fs::remove_file(complete.join("model-00002-of-00002.safetensors")).unwrap();
    assert!(
        ltx_gemma_override_path(Some(complete.as_os_str().to_owned())).is_none(),
        "an incomplete $LTX_GEMMA_DIR must not win"
    );

    // Unset override → None (the bundled/cache resolution proceeds).
    assert!(ltx_gemma_override_path(None).is_none());
}

/// `resolve_ltx_eros_gemma_dir`: a complete `models/mlx/gemma` sibling of the eros checkpoint wins;
/// an incomplete/absent sibling with no bundle snapshot → `None` (provider surfaces the clear
/// "set LTX_GEMMA_DIR" error). Skipped when an operator `$LTX_GEMMA_DIR` is set (the override path
/// returns `None` by design, exercised by [`resolve_bundled_ltx_gemma_dir`]'s own coverage).
#[cfg(target_os = "macos")]
#[test]
fn resolve_ltx_eros_gemma_prefers_local_sibling() {
    if std::env::var_os("LTX_GEMMA_DIR").is_some() {
        return;
    }
    let data_guard = tempfile::Builder::new()
        .prefix("sw_eros_gemma_")
        .tempdir()
        .expect("temp dir");
    let data = data_guard.path();
    let mlx = data.join("models").join("mlx");
    let eros = mlx.join("ltx_2_3_eros");
    std::fs::create_dir_all(&eros).unwrap();
    let settings = Settings {
        data_dir: data.to_path_buf(),
        ..Settings::from_env()
    };

    // No sibling gemma and no bundle snapshot in this fresh data dir → None.
    assert!(resolve_ltx_eros_gemma_dir(&settings, &eros).is_none());

    // A complete `models/mlx/gemma` sibling is resolved as the eros TE.
    let gemma = mlx.join("gemma");
    write_complete_gemma_dir(&gemma);
    assert_eq!(
        resolve_ltx_eros_gemma_dir(&settings, &eros).as_deref(),
        Some(gemma.as_path())
    );

    // An incomplete sibling does not win (falls through to the absent bundle → None).
    std::fs::remove_file(gemma.join("model-00001-of-00002.safetensors")).unwrap();
    assert!(resolve_ltx_eros_gemma_dir(&settings, &eros).is_none());
}

/// sc-18809: the eros gemma RESOLVER's bundle fallback must share ONE predicate with its probe.
/// Widening `ensure_ltx_bundle_gemma_present` to the cross-revision `bundled_ltx_gemma_dir` scan while
/// this resolver kept a single-snapshot `huggingface_snapshot_dir(..).join("gemma")` check left the
/// probe STRICTLY MORE PERMISSIVE than the resolver — an inversion that is fatal exactly here: the
/// probe reports gemma present and skips the fetch, then the resolver returns `None`, and the eros job
/// dead-ends on the provider's required-`LoadSpec::text_encoder` error instead of self-healing by
/// fetching as it did before this story.
///
/// The fixture moves the split-revision hub's `gemma/` onto the BUMPED snapshot, so the snapshot
/// `resolve_huggingface_snapshot_dir` selects (pre-bump, q4 7 + q8 7 = 14 files) has no `gemma/` at
/// all while the sibling (bf16 7 + gemma 5 = 12) holds the only complete one. Under the single-snapshot
/// form this resolves `None`; under the shared predicate it reaches the sibling revision.
#[cfg(target_os = "macos")]
#[test]
fn resolve_ltx_eros_gemma_reaches_a_sibling_revision() {
    let hub_guard = ltx_split_revision_hub("eros_gemma");
    let hub = hub_guard.path();
    let snapshots = hub
        .join(format!(
            "models--{}",
            sceneworks_core::hf_home::safe_repo_dir_name(LTX_BUNDLE_REPO).expect("slug")
        ))
        .join("snapshots");
    let pre_bump = snapshots.join("254989c3ca7ee691187647f350b112c0c448789d");
    let bumped = snapshots.join(LTX_BUNDLE_REVISION);
    // Move gemma OFF the most-files snapshot onto the sibling: the selected snapshot keeps 14 files
    // (q4 + q8) against the sibling's 12 (bf16 + gemma), so the SELECTION is unchanged and the only
    // complete `gemma/` is one revision over.
    std::fs::rename(pre_bump.join("gemma"), bumped.join("gemma")).unwrap();

    let data_guard = tempfile::Builder::new()
        .prefix("sw_eros_gemma_sibling_")
        .tempdir()
        .expect("temp dir");
    let eros = data_guard
        .path()
        .join("models")
        .join("mlx")
        .join("ltx_2_3_eros");
    std::fs::create_dir_all(&eros).unwrap();
    let settings = Settings {
        data_dir: data_guard.path().to_path_buf(),
        ..offline_settings()
    };

    let (selected, resolved) = ltx_with_hermetic_cache(hub, || {
        (
            crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, LTX_BUNDLE_REPO),
            resolve_ltx_eros_gemma_dir(&settings, &eros),
        )
    });
    assert_eq!(
        selected.as_deref(),
        Some(pre_bump.as_path()),
        "fixture premise: the most-files heuristic must still select the gemma-LESS snapshot, or \
         this test stops discriminating"
    );
    assert_eq!(
        resolved.as_deref(),
        Some(bumped.join("gemma").as_path()),
        "the eros resolver must see the gemma its own probe already sees — a single-snapshot bundle \
         fallback returns `None` here while `ensure_ltx_bundle_gemma_present` reports it present, so \
         the job dead-ends on the required-`LoadSpec::text_encoder` error"
    );
}

/// sc-8827 (F-025): the LTX Gemma-encoder dir rides `LoadSpec::text_encoder` (via
/// `VideoGenInput::text_encoder_dir`) instead of the process-global `$LTX_GEMMA_DIR`. This asserts
/// the path flows through `video_load_spec` onto the spec — `None` maps to `None` (env/`<root>`
/// fallback in the provider), a dir maps to `Some(WeightsSource::Dir(..))`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_load_spec_threads_text_encoder_dir() {
    let base = VideoGenInput {
        engine_id: "ltx_2_3",
        model_dir: PathBuf::from("/models/ltx/q4"),
        ..VideoGenInput::default()
    };
    // No override → text_encoder None (the provider reads its env/`<root>` fallback).
    assert!(
        video_load_spec(&base).text_encoder.is_none(),
        "no text_encoder_dir ⇒ no spec override"
    );
    // No uncensored enhancer ⇒ empty components map (video providers stage nothing by default).
    assert!(
        video_load_spec(&base).components.is_empty(),
        "no uncensored_enhancer_dir ⇒ empty components"
    );
    // An explicit gemma dir rides the spec as a Dir source.
    let gemma = PathBuf::from("/models/ltx/gemma");
    let with_te = VideoGenInput {
        text_encoder_dir: Some(gemma.clone()),
        ..base
    };
    assert!(
        matches!(video_load_spec(&with_te).text_encoder, Some(WeightsSource::Dir(ref p)) if *p == gemma),
        "text_encoder_dir rides LoadSpec::text_encoder as a Dir source"
    );

    // sc-13664: a resolved amoral-enhancer dir is staged as the `uncensored_enhancer` component so
    // the MLX LTX provider loads it on demand (the provider's `$LTX_UNCENSORED_GEMMA_DIR` / HF-cache
    // scan is deleted); no other components are added.
    let amoral = PathBuf::from("/models/ltx/amoral-gemma");
    let with_enh = VideoGenInput {
        engine_id: "ltx_2_3",
        model_dir: PathBuf::from("/models/ltx/q4"),
        uncensored_enhancer_dir: Some(amoral.clone()),
        ..VideoGenInput::default()
    };
    let spec = video_load_spec(&with_enh);
    assert!(
        matches!(spec.components.get("uncensored_enhancer"), Some(WeightsSource::Dir(p)) if *p == amoral),
        "uncensored_enhancer_dir rides LoadSpec::components as a Dir source"
    );
    assert_eq!(
        spec.components.len(),
        1,
        "only the enhancer component is staged"
    );
}

/// sc-12631 / sc-13175 / sc-14625: the candle video engines that require the provider's sequential
/// `candle.vramGbByTier` peak load with `OffloadPolicy::Sequential` — the A14B (T2V + I2V) so its two
/// 14B experts swap one-resident-at-a-time, and the dense TI2V-5B (sc-13175) so its UMT5 TE + z48 VAE
/// are flushed off-GPU around the denoise. That residency is the coupling that makes the gate's
/// admission number truthful. SVD-XT stages conditioner → UNet → VAE decode, while LTX has no
/// offload lifecycle and stays `Resident`. Both halves are pinned: the policy predicate AND that
/// `video_load_spec` actually threads it onto the `LoadSpec`
/// (a decoupled predicate that never reached the spec would silently OOM on the resident path the gate
/// did not size for).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_sequential_video_engines_offload_others_resident() {
    for id in [
        "svd_xt",
        "wan2_2_ti2v_5b",
        "wan2_2_t2v_14b",
        "wan2_2_i2v_14b",
    ] {
        assert_eq!(
            candle_video_offload_policy(id),
            OffloadPolicy::Sequential,
            "{id} must offload (sized by its measured sequential peak, not the resident load)"
        );
        // Mirror `generate_candle_video_using`, which sets the input's policy from the predicate;
        // this pins that `video_load_spec` then threads it onto the `LoadSpec` (the coupling that
        // makes the sequential-peak gate truthful).
        let input = VideoGenInput {
            engine_id: id,
            offload_policy: candle_video_offload_policy(id),
            ..VideoGenInput::default()
        };
        assert_eq!(
            video_load_spec(&input).offload_policy,
            OffloadPolicy::Sequential,
            "{id}: video_load_spec must thread Sequential onto the LoadSpec"
        );
    }
    // LTX carries no offload lifecycle — forcing it sequential would be a no-op that misleads.
    let id = "ltx_2_3_distilled";
    assert_eq!(
        candle_video_offload_policy(id),
        OffloadPolicy::Resident,
        "{id} must stay resident"
    );
    let input = VideoGenInput {
        engine_id: id,
        offload_policy: candle_video_offload_policy(id),
        ..VideoGenInput::default()
    };
    assert_eq!(
        video_load_spec(&input).offload_policy,
        OffloadPolicy::Resident,
        "{id}: video_load_spec must keep Resident"
    );
}

/// The SVD production arm returns before the generic Wan/LTX input builder. Exercise the exact
/// builder that arm calls so moving Sequential only onto the later generic input cannot silently
/// restore the resident/OOM path.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn candle_svd_production_input_threads_sequential_into_load_spec() {
    let request = request(json!({
        "projectId": "p",
        "model": "svd",
        "mode": "image_to_video",
        "width": 1024,
        "height": 576,
        "fps": 7,
        "seed": 42,
        "advanced": { "numFrames": 25, "decodeChunkSize": 8, "steps": 25 }
    }));
    let input = candle_svd_input(
        "svd_xt",
        PathBuf::from("/models/svd"),
        Vec::new(),
        &request,
        SvdVramPreflight {
            frames: 25,
            decode_chunk_size: 8,
            steps: 25,
        },
    );
    assert_eq!(input.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(
        video_load_spec(&input).offload_policy,
        OffloadPolicy::Sequential,
        "the exact production SVD input must load the measured sequential lifecycle"
    );
}

/// Q8 opt-in detection (sc-5679): `advanced.mlxQuantize: 8` (int or string) → true; absent / Q4
/// → false. Drives both the resolve quant preference and the on-demand q8 fetch.
#[cfg(target_os = "macos")]
#[test]
fn ltx_wants_q8_reads_mlx_quantize() {
    let with = |adv: Value| {
        request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": adv }))
    };
    assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": 8 }))));
    assert!(ltx_wants_q8(&with(json!({ "mlxQuantize": "8" }))));
    assert!(!ltx_wants_q8(&with(json!({ "mlxQuantize": 4 }))));
    assert!(!ltx_wants_q8(&request(
        json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x" })
    )));
}

/// bf16 opt-in detection (sc-8513): `advanced.mlxQuantize <= 0` (int or string) → true; Q4/Q8 /
/// absent → false. Drives the dense-tier preference + the on-demand bf16 fetch. Also asserts the
/// tier order the three cases resolve to (bf16 only ever tried on an explicit opt-in).
#[cfg(target_os = "macos")]
#[test]
fn ltx_wants_bf16_and_tier_order() {
    let with = |adv: Value| {
        request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x", "advanced": adv }))
    };
    assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": 0 }))));
    assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": -1 }))));
    assert!(ltx_wants_bf16(&with(json!({ "mlxQuantize": "0" }))));
    assert!(!ltx_wants_bf16(&with(json!({ "mlxQuantize": 4 }))));
    assert!(!ltx_wants_bf16(&with(json!({ "mlxQuantize": 8 }))));
    let plain = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "x" }));
    assert!(!ltx_wants_bf16(&plain));

    assert_eq!(
        ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 0 }))),
        &["bf16", "q8", "q4"]
    );
    assert_eq!(
        ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 8 }))),
        &["q8", "q4"]
    );
    // sc-10859: a plain request (no mlxQuantize) defaults q4-first on the video lane (NOT the
    // sc-10726 app-wide q8 — no MLX video Q8 lever); bf16 stays out of the default search path.
    assert_eq!(ltx_bundle_tier_order(&plain), &["q4", "q8"]);
    // An explicit Q4 pick stays q4-first.
    assert_eq!(
        ltx_bundle_tier_order(&with(json!({ "mlxQuantize": 4 }))),
        &["q4", "q8"]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn svd_engine_id_maps_only_svd() {
    assert_eq!(svd_engine_id("svd"), Some("svd_xt"));
    assert_eq!(svd_engine_id("ltx_2_3"), None);
    assert_eq!(svd_engine_id("wan_2_2"), None);
    assert_eq!(svd_engine_id("svd_xt"), None);
}

/// SVD knobs resolve advanced → manifest entry → builtin default, then clamp; `conditioningFps`
/// reads the `condFps` manifest key. Mirrors the torch `svd_video` `_pipeline_kwargs` mapping.
#[cfg(target_os = "macos")]
#[test]
fn svd_knobs_resolve_advanced_over_manifest_over_default() {
    // Bare request → builtin defaults.
    let bare = request(json!({ "projectId": "p", "model": "svd", "sourceAssetId": "a" }));
    assert_eq!(svd_i32(&bare, "numFrames", "numFrames", 25, 1, 25), 25);
    assert_eq!(
        svd_i32(&bare, "motionBucketId", "motionBucketId", 127, 1, 255),
        127
    );
    assert_eq!(svd_i32(&bare, "conditioningFps", "condFps", 7, 1, 30), 7);
    assert_eq!(
        svd_i32(&bare, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
        8
    );
    assert_eq!(
        svd_f32(&bare, "noiseAugStrength", "noiseAugStrength", 0.02),
        0.02
    );
    assert_eq!(svd_steps(&bare), 25); // balanced

    // Manifest entry overrides the builtin default; advanced overrides the manifest.
    let layered = request(json!({
        "projectId": "p", "model": "svd", "sourceAssetId": "a", "quality": "fast",
        "modelManifestEntry": {
            "motionBucketId": 180, "condFps": 6, "noiseAugStrength": 0.1,
            "steps": { "fast": 12, "balanced": 25, "best": 30 }
        },
        "advanced": { "motionBucketId": 200, "decodeChunkSize": "16" }
    }));
    // advanced wins for motionBucketId; manifest wins for condFps + noiseAug.
    assert_eq!(
        svd_i32(&layered, "motionBucketId", "motionBucketId", 127, 1, 255),
        200
    );
    assert_eq!(svd_i32(&layered, "conditioningFps", "condFps", 7, 1, 30), 6);
    assert_eq!(
        svd_f32(&layered, "noiseAugStrength", "noiseAugStrength", 0.02),
        0.1
    );
    // numeric string parses.
    assert_eq!(
        svd_i32(&layered, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
        16
    );
    // steps from manifest's quality ladder (fast).
    assert_eq!(svd_steps(&layered), 12);

    // Out-of-range values clamp to the engine-safe bounds.
    let extreme = request(json!({
        "projectId": "p", "model": "svd", "sourceAssetId": "a",
        "advanced": { "motionBucketId": 999, "numFrames": 99, "decodeChunkSize": 0 }
    }));
    assert_eq!(
        svd_i32(&extreme, "motionBucketId", "motionBucketId", 127, 1, 255),
        255
    );
    assert_eq!(svd_i32(&extreme, "numFrames", "numFrames", 25, 1, 25), 25);
    assert_eq!(
        svd_i32(&extreme, "decodeChunkSize", "decodeChunkSize", 8, 1, 64),
        1
    );
}

/// Locate the cached SVD-XT diffusers snapshot (the stock HF repo the engine loads directly), or
/// `None` if absent — `$SCENEWORKS_MLX_SVD_DIR` else the default HF hub cache.
#[cfg(target_os = "macos")]
fn svd_real_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
        let path = PathBuf::from(dir.trim());
        if svd_dir_is_complete(&path) {
            return Some(path);
        }
    }
    let snaps = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub")
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| svd_dir_is_complete(path))
}

/// Real in-process SVD-XT image→video through the engine: load the stock diffusers snapshot,
/// animate a synthetic reference image into a tiny clip, and assert the decode seam returns the
/// requested RGB8 frames at the OUTPUT/playback fps (decoupled from the conditioning fps; sc-3764)
/// with NO audio and streamed denoise progress. Exercises the worker's `run_video_generation`
/// path (with the sc-3523 motion knobs) end to end. `#[ignore]` — the weights live outside CI.
#[cfg(target_os = "macos")]
#[ignore = "loads the real SVD-XT weights; run manually on a Mac with the checkpoint cached"]
#[test]
fn svd_real_weights_image_to_video() {
    let Some(model_dir) = svd_real_dir() else {
        eprintln!("skipping svd_real_weights_image_to_video: no SVD-XT checkpoint found");
        return;
    };
    let (w, h) = (64u32, 64u32);
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = (x * 255 / w) as u8;
            pixels[i + 1] = (y * 255 / h) as u8;
            pixels[i + 2] = 128;
        }
    }
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "svd_xt",
        model_dir,
        width: 256,
        height: 256,
        frames: 4,
        // Playback fps (24) distinct from the conditioning fps (7) to prove the decouple (sc-3764).
        fps: 24,
        steps: Some(2),
        seed: 7,
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: w,
                height: h,
                pixels,
            },
            strength: None,
        }],
        motion_bucket_id: Some(127.0),
        noise_aug_strength: Some(0.02),
        decode_chunk_size: Some(2),
        conditioning_fps: Some(7),
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded =
        run_video_generation(input, &cancel, &mut on_progress).expect("svd i2v generation");
    assert_eq!(decoded.frames.len(), 4, "4 frames");
    assert_eq!(
        decoded.fps, 24,
        "output fps follows the playback fps, not the conditioning fps (sc-3764)"
    );
    assert!(decoded.audio.is_none(), "SVD emits no audio");
    assert!(steps > 0, "denoise progress streamed");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// sc-14625's SceneWorks-side real route: load the CUDA provider through the pinned runtime bundle
/// with sequential offload, render a one-step 25-frame 1024x576 clip, run the same libx264 encoder
/// used by production, and verify both the physical MP4 and routing/provenance facts. The full
/// 25-step memory gate lives in inference's ignored smoke; one step here keeps this independent
/// worker/encoder/provenance gate practical while retaining the same peak shapes and decode recipe.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[ignore = "loads real SVD-XT weights and requires CUDA plus ffmpeg/ffprobe"]
#[test]
fn svd_cuda_real_weights_encode_h264_and_preserve_provenance() {
    let model_dir = PathBuf::from(
        std::env::var("SCENEWORKS_CUDA_SVD_DIR")
            .expect("set SCENEWORKS_CUDA_SVD_DIR to the complete SVD-XT snapshot"),
    );
    assert!(
        svd_dir_is_complete(&model_dir),
        "SCENEWORKS_CUDA_SVD_DIR is not a complete SVD-XT snapshot"
    );

    let (source_w, source_h) = (64u32, 64u32);
    let mut source_pixels = vec![0u8; (source_w * source_h * 3) as usize];
    for y in 0..source_h {
        for x in 0..source_w {
            let i = ((y * source_w + x) * 3) as usize;
            source_pixels[i] = (x * 255 / source_w) as u8;
            source_pixels[i + 1] = (y * 255 / source_h) as u8;
            source_pixels[i + 2] = ((x + y) % 251) as u8;
        }
    }

    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "svd_xt",
        model_dir,
        width: 1024,
        height: 576,
        frames: 25,
        fps: 7,
        steps: Some(1),
        seed: 42,
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: source_w,
                height: source_h,
                pixels: source_pixels,
            },
            strength: None,
        }],
        motion_bucket_id: Some(127.0),
        noise_aug_strength: Some(0.02),
        decode_chunk_size: Some(8),
        conditioning_fps: Some(7),
        ..VideoGenInput::default()
    };
    let spec = LoadSpec::new(WeightsSource::Dir(input.model_dir.clone()))
        .with_offload_policy(OffloadPolicy::Sequential);
    let generator = crate::inference_runtime::load("svd_xt", &spec)
        .expect("load pinned candle SVD provider sequentially");
    assert!(
        generator
            .descriptor()
            .capabilities
            .supports_sequential_offload
    );

    let cancel = CancelFlag::new();
    let mut progress_events = 0usize;
    let decoded =
        run_loaded_video_generation(generator.as_ref(), input, &cancel, &mut |_progress| {
            progress_events += 1
        })
        .expect("real CUDA SVD generation");
    assert_eq!(decoded.frames.len(), 25);
    assert_eq!(decoded.fps, 7);
    assert!(decoded.audio.is_none());
    assert!(progress_events > 0);
    assert!(decoded
        .frames
        .iter()
        .all(|frame| (frame.width, frame.height) == (1024, 576)));

    let clip = EncodedClip::measure(&decoded);
    let temp = tempfile::tempdir().expect("temp output dir");
    let media_path = temp.path().join("svd-real.mp4");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(encode_media(&media_path, decoded, None, None))
        .expect("SceneWorks libx264 encode");

    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
        ])
        .arg(&media_path)
        .output()
        .expect("run ffprobe");
    assert!(
        probe.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let probe: Value = serde_json::from_slice(&probe.stdout).expect("ffprobe JSON");
    let stream = &probe["streams"][0];
    assert_eq!(stream["codec_name"], json!("h264"));
    assert_eq!(stream["width"], json!(1024));
    assert_eq!(stream["height"], json!(576));
    assert_eq!(stream["avg_frame_rate"], json!("7/1"));
    assert_eq!(stream["nb_read_frames"], json!("25"));

    let request = request(json!({
        "projectId": "p",
        "model": "svd",
        "mode": "image_to_video",
        "width": 1024,
        "height": 576,
        "fps": 7,
        "seed": 42,
        "advanced": { "numFrames": 25, "decodeChunkSize": 8, "steps": 1 }
    }));
    let plan = VideoPlan::new(&request, temp.path());
    let fact = video_asset_fact(
        &plan,
        resolve_video_seed(&request),
        candle_video_adapter_label("svd_xt"),
        svd_raw_settings(&request),
        None,
        clip,
    );
    let raw = &fact["rawAdapterSettings"];
    assert_eq!(candle_video_engine_id(&request.model), Some("svd_xt"));
    assert_eq!(candle_video_adapter_label("svd_xt"), "candle_svd");
    assert_eq!(fact["seed"], json!(42));
    assert_eq!(fact["model"], json!("svd"));
    assert_eq!(fact["adapter"], json!("candle_svd"));
    assert_eq!(raw["model"], json!("svd"));
    assert_eq!(raw["frameCount"], json!(25));
    assert_eq!(raw["fps"], json!(7));
    assert_eq!(raw["realModelInference"], json!(true));
}

/// SeedVR2 video upscale (epic 4811 / sc-4816) end-to-end against the real 3B weights:
/// drives the same `run_loaded_video_generation` path the `video_upscale` handler uses
/// (a `VideoClip` of LR frames + a target size + `softness`), asserting an upscaled,
/// frame-count-preserving clip comes back. `#[ignore]` — the ~7 GB checkpoint lives outside
/// CI; run manually with `SCENEWORKS_SEEDVR2_DIR` pointed at a `numz/SeedVR2_comfyUI` snapshot
/// (e.g. the HF cache dir), on a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the real SeedVR2 3B weights (~7GB); run manually with SCENEWORKS_SEEDVR2_DIR set"]
fn seedvr2_video_upscale_real_weights() {
    let dir = std::env::var("SCENEWORKS_SEEDVR2_DIR")
        .expect("set SCENEWORKS_SEEDVR2_DIR to a numz/SeedVR2_comfyUI checkpoint snapshot dir");
    // 8 low-res frames (a multiple of 4 ≥ 8 so the 4:1 causal-VAE compression preserves the
    // count) fed as the LR `VideoClip`; target = 2× (both ÷16).
    let frames: Vec<Image> = (0..8)
        .map(|i| Image {
            width: 64,
            height: 48,
            pixels: stub_video_rgb8(64, 48, 7, i, 8),
        })
        .collect();
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "seedvr2_3b",
        model_dir: PathBuf::from(dir),
        conditioning: vec![Conditioning::VideoClip {
            frames,
            frame_idx: 0,
            strength: 1.0,
        }],
        width: 128,
        height: 96,
        frames: 8,
        fps: 16,
        seed: 0,
        softness: Some(0.3),
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let decoded = run_video_generation(input, &cancel, &mut |_| {})
        .expect("seedvr2 video upscale generation");
    assert_eq!(
        decoded.frames.len(),
        8,
        "frame count preserved (chunk multiple-of-4, ≥8)"
    );
    assert!(
        decoded
            .frames
            .iter()
            .all(|f| f.width == 128 && f.height == 96),
        "every frame upscaled to the target 128x96"
    );
    assert!(
        decoded
            .frames
            .iter()
            .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize),
        "RGB8 buffers are well-formed"
    );
    assert!(
        decoded.audio.is_none(),
        "the engine emits no audio (worker muxes the source)"
    );
}

// sc-9595: streaming worker-window chunking for the SeedVR2 upscale (replaces the sc-8829 host-RAM
// cap). The seam-correctness guarantee is that WORKER-WINDOWED streaming assembly is FRAME-IDENTICAL
// to a single whole-clip `video::assemble_overlap` — same frame count, order, and cross-fade blend —
// reusing the engine's own public `plan_chunks`/`assemble_overlap` one level up. These tests model
// the engine as a deterministic per-chunk upscale (a pure function of its source pixel window +
// seed, which is what `pipeline::preprocess_chunk` guarantees), exactly as the real driver does, and
// exercise the `Seedvr2StreamAssembler`. Real-weight PIXEL identity of an internally sub-chunked
// worker window is a GPU-golden concern the ignore-gated smokes don't run (see the PR notes).

/// A single-channel synthetic frame (1×1 RGB, all bytes = `v`) — enough to test frame identity,
/// ordering, and the cross-fade blend byte-exactly.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn syn_frame(v: u8) -> Image {
    Image {
        width: 1,
        height: 1,
        pixels: vec![v, v, v],
    }
}

/// Model the engine's whole-clip upscale over synthetic frames: `plan_chunks` at engine chunk `c`,
/// each chunk = the (identity-)upscaled source window (clamp-padded past the end, like
/// `preprocess_chunk`), then `assemble_overlap`. Mirrors `pipeline::generate_video` exactly, minus
/// the real tensor pass — the deterministic per-chunk structure is what worker windowing must
/// reproduce.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn engine_whole_clip_synthetic(src: &[Image], c: i32, ov: i32) -> Vec<Image> {
    let n = src.len() as i32;
    let plan = seedvr2_video::plan_chunks(n, c, ov);
    let mut chunk_frames: Vec<Vec<Image>> = Vec::with_capacity(plan.len());
    for chunk in &plan {
        let mut frames = Vec::with_capacity(chunk.len as usize);
        for j in 0..chunk.len {
            let idx = (chunk.start + j).clamp(0, n - 1) as usize;
            frames.push(src[idx].clone());
        }
        chunk_frames.push(frames);
    }
    seedvr2_video::assemble_overlap(&plan, &chunk_frames, n, ov)
}

/// Drive the worker-window streaming path over synthetic frames: `plan_chunks` at the worker chunk
/// size, upscale each window via `engine_whole_clip_synthetic` (the engine re-plans internally at
/// engine chunk `engine_c` over the LOCAL window), and cross-fade across worker windows with the
/// `Seedvr2StreamAssembler`. Returns the streamed frame sequence + the max retained-tail length
/// observed (the memory bound).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn worker_streamed_synthetic(
    src: &[Image],
    worker_chunk: i32,
    engine_c: i32,
    ov: i32,
) -> (Vec<Image>, usize) {
    let n = src.len() as i32;
    let plan = seedvr2_video::plan_chunks(n, worker_chunk, ov);
    let mut assembler = Seedvr2StreamAssembler::new(n, ov);
    let mut out: Vec<Image> = Vec::new();
    let mut max_tail = 0usize;
    for chunk in &plan {
        let start = chunk.start.max(0) as usize;
        if start >= src.len() {
            break;
        }
        let end = (start + chunk.len.max(0) as usize).min(src.len());
        let window_src: Vec<Image> = src[start..end].to_vec();
        let upscaled = engine_whole_clip_synthetic(&window_src, engine_c, ov);
        out.extend(assembler.push_window(chunk.start, upscaled));
        max_tail = max_tail.max(assembler.retained.len());
    }
    out.extend(assembler.finish());
    (out, max_tail)
}

/// THE core seam-correctness guarantee (sc-9595): windowed streaming assembly is frame-identical to
/// a single whole-clip `assemble_overlap`, across a sweep of frame counts, worker-chunk sizes, and
/// engine-chunk sizes — including where the worker chunk and engine chunk differ. Byte-exact on the
/// synthetic frames, so both the frame ORDER and the cross-fade BLEND match the whole-clip path.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_stream_matches_whole_clip_assembly() {
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    let mut cases = 0;
    for n in [1usize, 5, 8, 12, 16, 20, 28, 40, 41, 64, 100, 137, 256, 257] {
        let src: Vec<Image> = (0..n)
            .map(|i| syn_frame(((i * 13 + 5) % 251) as u8))
            .collect();
        for &worker_chunk in &[16i32, 20, 32, 64] {
            for &engine_c in &[8i32, 12, 16, 24, 32] {
                let whole = engine_whole_clip_synthetic(&src, engine_c, ov);
                let (streamed, _) = worker_streamed_synthetic(&src, worker_chunk, engine_c, ov);
                assert_eq!(
                    streamed.len(),
                    n,
                    "frame count preserved (n={n}, worker={worker_chunk}, engine={engine_c})"
                );
                assert_eq!(
                    streamed, whole,
                    "windowed == whole-clip (n={n}, worker={worker_chunk}, engine={engine_c})"
                );
                cases += 1;
            }
        }
    }
    assert!(cases > 200, "swept a broad matrix ({cases} cases)");
}

/// The streaming assembler holds BOUNDED memory: the retained tail never exceeds one worker window
/// plus the overlap, regardless of clip length — the property that removes the whole-clip host-RAM
/// footprint the sc-8829 cap existed to bound.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_stream_retains_bounded_tail() {
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    let worker_chunk = SEEDVR2_WORKER_CHUNK_FRAMES;
    // A long clip that would be tens of GB of RGB8 whole — the exact case the cap rejected.
    let n = 4000usize;
    let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
    let (streamed, max_tail) = worker_streamed_synthetic(&src, worker_chunk, 16, ov);
    assert_eq!(streamed.len(), n, "every frame emitted, in order");
    // The retained tail is at most one merged worker window (worker_chunk + a partial next window's
    // overlap), never the whole clip — orders of magnitude below `n`.
    assert!(
        max_tail as i32 <= worker_chunk + ov + 1,
        "retained tail {max_tail} bounded by one window (~{}), not the {n}-frame clip",
        worker_chunk + ov
    );
}

/// Frame ORDER is preserved end-to-end: each streamed frame carries its source ordinal through the
/// (identity) upscale + cross-fade, so the sequence is monotonic and complete (no drops/dupes/swaps)
/// even across worker-window boundaries with a partial trailing window.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_stream_preserves_frame_order() {
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    // 150 frames, worker chunk 64 → 3 worker windows with a partial last one; a value per ordinal.
    let n = 150usize;
    let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
    let (streamed, _) = worker_streamed_synthetic(&src, SEEDVR2_WORKER_CHUNK_FRAMES, 16, ov);
    assert_eq!(streamed.len(), n);
    // Non-overlap frames pass through unblended → equal to the source ordinal value; overlap frames
    // are cross-faded but between two IDENTICAL-value windows here only at seams. Compare to the
    // whole-clip oracle for the exact expected values.
    let whole = engine_whole_clip_synthetic(&src, 16, ov);
    assert_eq!(
        streamed, whole,
        "streamed frames match the whole-clip oracle"
    );
}

/// Cap removal (sc-9595): a clip that the sc-8829 cap would have REJECTED (huge whole-clip RGB8
/// footprint) now streams to completion with the correct frame count and no error — the capability
/// narrowing is gone. (The removed `check_seedvr2_host_ram` / `seedvr2_estimated_host_bytes`
/// helpers no longer exist; this test would not compile if a hard cap were reintroduced in the
/// streaming path.)
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn seedvr2_stream_upscales_formerly_capped_clip() {
    let ov = seedvr2_video::DEFAULT_OVERLAP;
    // ~4300 frames — the finding's pathological few-minute 1080p clip the old cap rejected outright.
    let n = 4300usize;
    let src: Vec<Image> = (0..n).map(|i| syn_frame((i % 251) as u8)).collect();
    let (streamed, max_tail) = worker_streamed_synthetic(&src, SEEDVR2_WORKER_CHUNK_FRAMES, 16, ov);
    assert_eq!(
        streamed.len(),
        n,
        "the formerly-capped clip streams in full"
    );
    assert!(
        (max_tail as i32) < n as i32 / 10,
        "peak retained frames stay far below the whole clip (bounded memory)"
    );
}

/// The `ScratchDir` RAII guard removes the (multi-GB, in prod) output PNG scratch on the ERROR path
/// (sc-9595 review): dropping an ARMED guard deletes a populated dir — this is the guarantee that
/// covers a create_dir_all / progress-POST / encode `?`-return or a cancel between the stream and the
/// encode, where the old code leaked the whole 4× sequence. Disarming (the success path, after the
/// caller already removed the dir) makes Drop a no-op so it can't fight a concurrent same-name reuse.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn scratch_dir_guard_cleans_up_on_drop_and_respects_disarm() {
    let base_guard = tempfile::Builder::new()
        .prefix("sceneworks_seedvr2_scratchtest_")
        .tempdir()
        .expect("temp dir");
    let base = base_guard.path();

    // ARMED guard dropped (error path): a populated dir + a nested file are removed on Drop.
    let armed_dir = base.join("armed");
    std::fs::create_dir_all(&armed_dir).unwrap();
    std::fs::write(armed_dir.join("frame_00001.png"), b"not-a-real-png").unwrap();
    assert!(armed_dir.exists(), "precondition: scratch populated");
    {
        let _guard = ScratchDir::new(armed_dir.clone());
        // Simulate the post-stream encode span returning Err before any manual cleanup: the guard
        // goes out of scope here without disarming.
    }
    assert!(
        !armed_dir.exists(),
        "armed guard's Drop must remove the leaked output scratch on the error path"
    );

    // DISARMED guard dropped (success path): the caller already removed the dir, and Drop must NOT
    // re-remove — modelled by a dir the guard deliberately leaves intact.
    let disarmed_dir = base.join("disarmed");
    std::fs::create_dir_all(&disarmed_dir).unwrap();
    {
        let mut guard = ScratchDir::new(disarmed_dir.clone());
        guard.disarm();
    }
    assert!(
        disarmed_dir.exists(),
        "disarmed guard's Drop must be a no-op (caller owns cleanup)"
    );

    // A guard over an ALREADY-removed dir drops cleanly (benign NotFound, no panic).
    let missing_dir = base.join("missing");
    {
        let _guard = ScratchDir::new(missing_dir.clone());
    }
    assert!(!missing_dir.exists());
}

/// `advanced.noAudio` maps to the engine's `video_mode = "no_audio"`; enhance flags
/// flow through. Asserts the LTX request build (the VideoGenInput, pre-load).
#[cfg(target_os = "macos")]
#[test]
fn ltx_advanced_flags_map_to_video_gen_input() {
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
        "advanced": { "noAudio": true, "enhancePrompt": true }
    }));
    assert!(advanced::bool(&req.advanced, "noAudio"));
    assert!(advanced::bool(&req.advanced, "enhancePrompt"));
    assert!(!advanced::bool(&req.advanced, "useUncensoredEnhancer"));
    // LTX adapters: a plain user LoRA is uniform (no per-pass schedule, no moe tag).
    let settings = Settings::from_env();
    let none = resolve_ltx_adapters(&settings, &req).unwrap();
    assert!(none.is_empty());
}

/// The fixture's own HF hub cache: the path `huggingface_hub_cache_dir` falls back to for this
/// `data_dir`. Pin `HF_HUB_CACHE` here (see [`ltx_eros_auto_injects_distill_lora_per_pass`]) and
/// the resolver returns this dir outright rather than by fallback, so the fixture no longer
/// depends on the ambient environment being unset.
#[cfg(target_os = "macos")]
fn fake_hf_hub_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("cache").join("huggingface").join("hub")
}

/// Lay down a fake HF snapshot file under `data_dir`
/// (`cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/<file>`) and return its path, so
/// [`resolve_ltx_distill_adapter`] resolves hermetically (mirrors `write_fake_wan_lightning`).
#[cfg(target_os = "macos")]
fn write_fake_hf_lora(data_dir: &Path, repo: &str, file: &str) -> PathBuf {
    let snapshot = fake_hf_hub_dir(data_dir)
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join("deadbeef");
    std::fs::create_dir_all(&snapshot).unwrap();
    let path = snapshot.join(file);
    write_lora_fixture(&path, None);
    path
}

/// The `ltx_2_3_eros` manifest shape [`resolve_ltx_distill_adapter`] reads: `mlx.autoDistillLora`
/// per-pass strengths plus the `resources.distilledLora` repo/file.
#[cfg(target_os = "macos")]
fn eros_manifest_entry(repo: &str, file: &str) -> Value {
    json!({
        "mlx": { "autoDistillLora": { "stage1Strength": 1.0, "stage2Strength": 0.4 } },
        "resources": { "distilledLora": { "repo": repo, "file": file } },
    })
}

/// Regression (10Eros noise): `ltx_2_3_eros` is not pre-distilled, so `resolve_ltx_adapters` must
/// auto-inject its cond_safe distill LoRA at per-pass strengths (1.0 first pass / 0.4 upscale) ahead
/// of any user LoRAs — the injection was lost in the sc-3037 Python→Rust video cutover, leaving the
/// undistilled base to collapse to noise. Strengths honor `advanced.distillStage*Strength`;
/// `advanced.useDistillLora = false` opts out.
#[cfg(target_os = "macos")]
#[test]
fn ltx_eros_auto_injects_distill_lora_per_pass() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_eros_distill_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    // PIN the hub cache at this fixture's own dir for the whole test, rather than depending on
    // the ambient environment. `huggingface_hub_cache_dir` reads `HF_HUB_CACHE` FIRST and returns
    // it outright, so this alone decides the resolver — whatever the host or a concurrent test
    // holds in the lower-priority `HUGGINGFACE_HUB_CACHE` / `HF_HOME`.
    //
    // This replaces a read-then-use ("skip if any HF var is set", then resolve) that was wrong
    // twice over: process-global `set_var` from another test could land between the check and the
    // use (sc-12380), AND on any box with an ambient HF cache the guard turned "env is set" into
    // a silent PASS that asserted nothing. Pinning removes both — there is no window to race and
    // no configuration in which this test quietly stops testing. `EnvVars` holds the crate-wide
    // env lock until it drops at end of test, so no concurrent writer can retarget the resolver.
    let _env = EnvVars::set(&[(
        "HF_HUB_CACHE",
        fake_hf_hub_dir(dir).to_str().expect("utf-8 fixture hub"),
    )]);
    let repo = "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments";
    let file = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors";
    let distill = write_fake_hf_lora(dir, repo, file);
    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };

    // Default: the distill LoRA is injected alone, at per-pass 1.0/0.4.
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
        "modelManifestEntry": eros_manifest_entry(repo, file),
    }));
    let specs = resolve_ltx_adapters(&settings, &req).expect("resolve eros adapters");
    assert_eq!(specs.len(), 1, "the distill LoRA must be injected");
    assert_eq!(specs[0].path, distill);
    assert_eq!(specs[0].kind, AdapterKind::Lora);
    assert_eq!(specs[0].pass_scales, Some(vec![1.0, 0.4]));

    // A user LoRA stacks AFTER the distill (the distill is the model's base recipe).
    let user = dir.join("style.safetensors");
    write_lora_fixture(&user, None);
    let req_user = request(json!({
        "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
        "modelManifestEntry": eros_manifest_entry(repo, file),
        "loras": [ { "path": user.to_string_lossy(), "weight": 0.5 } ],
    }));
    let specs = resolve_ltx_adapters(&settings, &req_user).expect("resolve eros + user");
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].pass_scales, Some(vec![1.0, 0.4]), "distill first");
    assert!(
        specs[1].pass_scales.is_none(),
        "user LoRA is uniform per-pass"
    );
    assert!((specs[1].scale - 0.5).abs() < 1e-6);

    // `advanced` overrides the per-pass strengths.
    let req_override = request(json!({
        "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
        "modelManifestEntry": eros_manifest_entry(repo, file),
        "advanced": { "distillStage1Strength": 0.8, "distillStage2Strength": 0.2 },
    }));
    let specs = resolve_ltx_adapters(&settings, &req_override).expect("override strengths");
    assert_eq!(specs[0].pass_scales, Some(vec![0.8, 0.2]));

    // Opt-out disables the injection entirely (no user LoRAs → empty).
    let req_off = request(json!({
        "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
        "modelManifestEntry": eros_manifest_entry(repo, file),
        "advanced": { "useDistillLora": false },
    }));
    assert!(resolve_ltx_adapters(&settings, &req_off)
        .unwrap()
        .is_empty());
}

/// A model that declares `mlx.autoDistillLora` but whose co-requisite distill LoRA is not installed
/// fails with an actionable payload error rather than silently degrading to noise; a model that
/// declares no `autoDistillLora` (base `ltx_2_3`) injects nothing.
#[cfg(target_os = "macos")]
#[test]
fn ltx_distill_lora_missing_errors_and_base_model_is_noop() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_eros_missing_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    // PIN the hub cache at this fixture's own (empty) dir for the whole test, the hermetic pattern
    // of `ltx_eros_auto_injects_distill_lora_per_pass`. Nothing is staged under it, so the distill
    // LoRA is genuinely missing → the resolver must error. The old "skip if any HF var is set"
    // guard both raced a concurrent `EnvVars::set` writer (sc-13898) and was a silent PASS on any
    // box with an ambient HF cache set; pinning removes both. `EnvVars` holds the crate-wide env
    // lock until it drops, so no concurrent writer can retarget the resolver mid-test.
    let _env = EnvVars::set(&[(
        "HF_HUB_CACHE",
        fake_hf_hub_dir(dir).to_str().expect("utf-8 fixture hub"),
    )]);
    let settings = Settings {
        data_dir: dir.to_path_buf(),
        ..Settings::from_env()
    };

    // Declared distill LoRA, but nothing on disk → actionable error (not silent noise).
    let repo = "TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments";
    let file = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors";
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3_eros", "prompt": "a fox",
        "modelManifestEntry": eros_manifest_entry(repo, file),
    }));
    assert!(matches!(
        resolve_ltx_adapters(&settings, &req),
        Err(WorkerError::InvalidPayload(_))
    ));

    // Base `ltx_2_3` declares no `autoDistillLora` → no injection.
    let base = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
        "modelManifestEntry": { "family": "ltx-video" },
    }));
    assert!(resolve_ltx_adapters(&settings, &base).unwrap().is_empty());
}

/// The FLF keyframe knobs (sc-3055) parse from JSON numbers + numeric strings and fall back
/// to their defaults — these drive the two `Keyframe` strengths + the first keyframe's index.
#[cfg(target_os = "macos")]
#[test]
fn advanced_numeric_helpers_parse_flf_keyframe_knobs() {
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
        "mode": "first_last_frame",
        "advanced": {
            "imageConditioningStrength": 0.8,        // JSON number
            "lastFrameConditioningStrength": "0.65",  // numeric string
            "imageFrameIndex": 2
        }
    }));
    assert_eq!(
        advanced::f32(&req.advanced, "imageConditioningStrength", 1.0),
        0.8
    );
    assert_eq!(
        advanced::f32(&req.advanced, "lastFrameConditioningStrength", 1.0),
        0.65
    );
    assert_eq!(advanced::i32(&req.advanced, "imageFrameIndex", 0), 2);
    // Absent keys → the fully-pinned defaults (strength 1.0, first index 0).
    let bare = request(json!({ "projectId": "p", "model": "ltx_2_3", "prompt": "a fox" }));
    assert_eq!(
        advanced::f32(&bare.advanced, "imageConditioningStrength", 1.0),
        1.0
    );
    assert_eq!(
        advanced::f32(&bare.advanced, "lastFrameConditioningStrength", 1.0),
        1.0
    );
    assert_eq!(advanced::i32(&bare.advanced, "imageFrameIndex", 0), 0);
}

/// `resolve_keyframe_conditioning` fails clearly when an FLF source/last-frame asset id is
/// missing (the guards run before any project/image IO, so no fixture is needed).
#[cfg(target_os = "macos")]
#[test]
fn keyframe_conditioning_requires_both_frame_assets() {
    let settings = Settings::from_env();
    // No sourceAssetId.
    let no_first = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
        "mode": "first_last_frame", "lastFrameAssetId": "asset_last"
    }));
    let err = resolve_keyframe_conditioning(&settings, &no_first, Path::new("/tmp/p"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("source image"), "got: {err}");
    // sourceAssetId but no lastFrameAssetId.
    let no_last = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "a fox",
        "mode": "first_last_frame", "sourceAssetId": "asset_first"
    }));
    let err = resolve_keyframe_conditioning(&settings, &no_last, Path::new("/tmp/p"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("last-frame image"), "got: {err}");
}

/// FLF on a 14B Wan MoE engine is rejected at the conditioning resolver (defence-in-depth
/// behind the routing gate, which already restricts FLF to `wan_2_2`/TI2V-5B).
#[cfg(target_os = "macos")]
#[test]
fn wan_flf_rejected_on_non_ti2v_engine() {
    let settings = Settings::from_env();
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2_t2v_14b", "prompt": "a fox",
        "mode": "first_last_frame",
        "sourceAssetId": "a", "lastFrameAssetId": "b"
    }));
    let err = resolve_wan_conditioning(&settings, &req, Path::new("/tmp/p"), "wan2_2_t2v_14b")
        .unwrap_err()
        .to_string();
    assert!(err.contains("TI2V-5B"), "got: {err}");
}

/// A 1×1 RGB [`Image`] for clip-conditioning construction tests (the engine resizes; the
/// content is irrelevant — only the variant / frame_idx / strength mapping is under test).
#[cfg(target_os = "macos")]
fn pixel(n: u8) -> Image {
    Image {
        width: 1,
        height: 1,
        pixels: vec![n, n, n],
    }
}

/// extend_clip → one `VideoClip` pinned at latent frame 0, strength `videoConditioningStrength`
/// (default 1.0); the bridge-only right knob is ignored.
#[cfg(target_os = "macos")]
#[test]
fn video_clip_conditioning_extend_maps_single_clip_at_zero() {
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
        "mode": "extend_clip",
        "advanced": { "videoConditioningStrength": 0.7 }
    }));
    let cond = build_video_clip_conditioning(&req, vec![pixel(1), pixel(2)], None).unwrap();
    assert_eq!(cond.len(), 1);
    match &cond[0] {
        Conditioning::VideoClip {
            frames,
            frame_idx,
            strength,
        } => {
            assert_eq!(frames.len(), 2);
            assert_eq!(*frame_idx, 0);
            assert_eq!(*strength, 0.7);
        }
        other => panic!("expected VideoClip, got {other:?}"),
    }
}

/// video_bridge → left clip at 0 (`videoConditioningStrength`) + right clip at -1
/// (`bridgeRightVideoConditioningStrength`); both default to 1.0 when absent.
#[cfg(target_os = "macos")]
#[test]
fn video_clip_conditioning_bridge_maps_left_zero_right_tail() {
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
        "mode": "video_bridge",
        "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
    }));
    let cond = build_video_clip_conditioning(&req, vec![pixel(1)], Some(vec![pixel(2), pixel(3)]))
        .unwrap();
    assert_eq!(cond.len(), 2);
    match (&cond[0], &cond[1]) {
        (
            Conditioning::VideoClip {
                frames: left,
                frame_idx: left_idx,
                strength: left_strength,
            },
            Conditioning::VideoClip {
                frames: right,
                frame_idx: right_idx,
                strength: right_strength,
            },
        ) => {
            assert_eq!(left.len(), 1);
            assert_eq!(*left_idx, 0);
            assert_eq!(*left_strength, 1.0); // default
            assert_eq!(right.len(), 2);
            assert_eq!(*right_idx, -1); // engine negative-from-end (lf + idx)
            assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
        }
        other => panic!("expected two VideoClips, got {other:?}"),
    }
}

/// video_bridge without the right clip frames is a construction error (defence behind the
/// resolver's `bridgeRightClipAssetId` guard).
#[cfg(target_os = "macos")]
#[test]
fn video_clip_conditioning_bridge_requires_right_clip() {
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "bridge them",
        "mode": "video_bridge"
    }));
    let err = build_video_clip_conditioning(&req, vec![pixel(1)], None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("right-side source clip"), "got: {err}");
}

/// Wan extend_clip → one boundary `Keyframe` at latent frame 0 (the source clip's last frame),
/// strength `videoConditioningStrength` (default 1.0); the right frame is ignored (sc-3357).
#[cfg(target_os = "macos")]
#[test]
fn wan_boundary_conditioning_extend_pins_last_frame_at_zero() {
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "prompt": "extend it",
        "mode": "extend_clip",
        "advanced": { "videoConditioningStrength": 0.7 }
    }));
    let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
    assert_eq!(cond.len(), 1);
    match &cond[0] {
        Conditioning::Keyframe {
            frame_idx,
            strength,
            ..
        } => {
            assert_eq!(*frame_idx, 0);
            assert_eq!(*strength, 0.7);
        }
        other => panic!("expected Keyframe, got {other:?}"),
    }
}

/// Wan video_bridge → left clip's last frame at 0 (`videoConditioningStrength`) + right clip's
/// first frame at -1 (`bridgeRightVideoConditioningStrength`); mechanically FLF (sc-3357).
#[cfg(target_os = "macos")]
#[test]
fn wan_boundary_conditioning_bridge_pins_both_boundaries() {
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
        "mode": "video_bridge",
        "advanced": { "bridgeRightVideoConditioningStrength": "0.5" }
    }));
    let cond = build_wan_boundary_conditioning(&req, pixel(1), Some(pixel(2))).unwrap();
    assert_eq!(cond.len(), 2);
    match (&cond[0], &cond[1]) {
        (
            Conditioning::Keyframe {
                frame_idx: left_idx,
                strength: left_strength,
                ..
            },
            Conditioning::Keyframe {
                frame_idx: right_idx,
                strength: right_strength,
                ..
            },
        ) => {
            assert_eq!(*left_idx, 0);
            assert_eq!(*left_strength, 1.0); // default
            assert_eq!(*right_idx, -1); // engine negative-from-end
            assert_eq!(*right_strength, 0.5); // numeric-string advanced knob
        }
        other => panic!("expected two Keyframes, got {other:?}"),
    }
}

/// Wan video_bridge without the right boundary frame is a construction error (defence behind
/// the resolver's `bridgeRightClipAssetId` guard).
#[cfg(target_os = "macos")]
#[test]
fn wan_boundary_conditioning_bridge_requires_right_frame() {
    let req = request(json!({
        "projectId": "p", "model": "wan_2_2", "prompt": "bridge them",
        "mode": "video_bridge"
    }));
    let err = build_wan_boundary_conditioning(&req, pixel(1), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("right-side source clip"), "got: {err}");
}

/// The IC-LoRA detector matches the torch `lora_looks_like_ic_lora` markers (flags, role,
/// and "ic-lora" / "ltx-2-3-ic-" in id/name/path/files) and rejects an ordinary LoRA.
#[cfg(target_os = "macos")]
#[test]
fn ic_lora_detection_matches_torch_markers() {
    // Explicit flags.
    assert!(loras_contain_ic_lora(&[json!({ "icLora": true })]));
    assert!(loras_contain_ic_lora(&[json!({ "isIcLora": true })]));
    // conditioningRole (with the `-`/`_` normalisation).
    assert!(loras_contain_ic_lora(&[
        json!({ "conditioningRole": "IC-Lora" })
    ]));
    // Name / id / path markers.
    assert!(loras_contain_ic_lora(&[
        json!({ "name": "LTX-2.3-22b-IC-LoRA-Union-Control" })
    ]));
    assert!(loras_contain_ic_lora(&[
        json!({ "id": "ltx_2_3_ic_union" })
    ]));
    assert!(loras_contain_ic_lora(&[
        json!({ "source": { "files": ["my-ic-lora.safetensors"] } })
    ]));
    // A bare string id.
    assert!(loras_contain_ic_lora(&[json!("some-ic-lora-v2")]));
    // An ordinary LoRA is not an IC-LoRA.
    assert!(!loras_contain_ic_lora(&[
        json!({ "name": "cinematic-style", "path": "/loras/cinematic.safetensors" })
    ]));
    assert!(!loras_contain_ic_lora(&[]));
}

/// extend/bridge conditioning fails clearly (before any IO) when no IC-LoRA is installed —
/// mirrors the torch gate; without the adapter the appended clip tokens are inert.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn video_clip_conditioning_requires_ic_lora() {
    // The IC-LoRA gate returning before any I/O is the claim; pin the network fields so a
    // regression that breaks it fails locally instead of dialing out (sc-12380).
    let settings = offline_settings();
    let api = ApiClient::new(&settings);
    // The IC-LoRA gate is the resolver's first check, so it returns before touching `job`
    // / the api / disk — a minimal snapshot suffices.
    let job: JobSnapshot = serde_json::from_value(json!({
        "id": "job-extend-1",
        "type": "video_extend",
        "status": "preparing",
        "projectId": "p",
        "projectName": "P",
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": null,
        "progress": 0,
        "stage": "preparing",
        "message": "",
        "error": null,
        "etaSeconds": null,
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-09T00:00:00Z",
        "updatedAt": "2026-06-09T00:00:00Z"
    }))
    .expect("job snapshot");
    let req = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "extend it",
        "mode": "extend_clip", "sourceClipAssetId": "clip_a",
        "loras": [{ "name": "cinematic-style" }]
    }));
    let err = resolve_video_clip_conditioning(&api, &settings, &job, &req, Path::new("/tmp/p"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("IC-LoRA"), "got: {err}");
}

/// An LTX-2.3 snapshot **complete for the current engine** ([`ltx_dir_is_complete`]),
/// else `None` so the smoke skips. Checks `$SCENEWORKS_MLX_LTX_DIR`, the turnkey SceneWorks
/// bundle's `q4/`/`q8/` subdirs in the HF cache ([`LTX_BUNDLE_REPO`], `SceneWorks/ltx-2.3-mlx`),
/// and the app-managed `<data>/models/mlx/*` dirs (which predate the audio+i2v layout, so skip).
#[cfg(target_os = "macos")]
fn ltx_complete_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_LTX_DIR") {
        candidates.push(PathBuf::from(dir.trim()));
    }
    if let Ok(home) = std::env::var("HOME") {
        let hub = PathBuf::from(&home).join(".cache/huggingface/hub");
        // The turnkey SceneWorks bundle: each snapshot carries `q4/` + `q8/` checkpoint subdirs.
        let snapshots = hub
            .join("models--SceneWorks--ltx-2.3-mlx")
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for snapshot in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
                candidates.push(snapshot.join("q4"));
                candidates.push(snapshot.join("q8"));
            }
        }
        let base =
            PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx");
        for id in [
            "ltx_2_3_base_q4",
            "ltx_2_3_base_q8",
            "ltx_2_3_eros",
            "ltx_2_3",
        ] {
            candidates.push(base.join(id));
        }
    }
    candidates.into_iter().find(|dir| ltx_dir_is_complete(dir))
}

/// Real in-process LTX-2.3 T2V **with synchronized audio** through the engine. Loads a
/// complete converted snapshot + the cached Gemma TE and denoises a tiny 9-frame clip,
/// asserting frames come back RGB8-sized **and an audio track is produced** with streamed
/// progress. `#[ignore]` + skips unless a complete snapshot is present (the cached
/// `ltx_2_3_base_*` dirs predate the engine's vocoder/vae_encoder layout).
#[cfg(target_os = "macos")]
#[ignore = "loads the real LTX-2.3 weights + Gemma TE; needs a snapshot complete for the current engine"]
#[test]
fn ltx_real_weights_with_audio() {
    let Some(model_dir) = ltx_complete_dir() else {
        eprintln!("skipping ltx_real_weights_with_audio: no complete LTX snapshot found");
        return;
    };
    // The bundle ships gemma beside the q4/q8 dir; thread it onto the LoadSpec (matches the worker
    // path, sc-8827) so the smoke needs no separate mlx-community/gemma snapshot in the HF cache.
    let text_encoder_dir = resolve_bundled_ltx_gemma_dir(&model_dir);
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "ltx_2_3",
        model_dir,
        prompt: "a calm ocean wave at sunset, gentle surf".to_owned(),
        width: 256,
        height: 256,
        frames: 9,
        fps: 24,
        seed: 7,
        text_encoder_dir,
        ..VideoGenInput::default()
    };
    let cancel = CancelFlag::new();
    let mut steps = 0u32;
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { .. } = progress {
            steps += 1;
        }
    };
    let decoded =
        run_video_generation(input, &cancel, &mut on_progress).expect("LTX A/V generation");
    assert_eq!(decoded.frames.len(), 9, "9 frames (1 + 8·1)");
    let audio = decoded
        .audio
        .expect("LTX produces a synchronized audio track");
    assert!(audio.sample_rate >= 1 && !audio.samples.is_empty());
    assert!(steps > 0, "denoise progress streamed");
    assert!(decoded
        .frames
        .iter()
        .all(|f| f.pixels.len() == (f.width * f.height * 3) as usize));
}

/// Full encode → mp4 + poster, exercised against a real ffmpeg. Skips when no
/// ffmpeg is reachable (SCENEWORKS_FFMPEG or `ffmpeg` on PATH) so it never fails a
/// host without the binary; CI with ffmpeg runs it for real.
#[tokio::test]
async fn encode_stub_to_mp4_with_audio_and_poster() {
    if !ffmpeg_reachable() {
        eprintln!("skipping encode_stub_to_mp4_with_audio_and_poster: ffmpeg not found");
        return;
    }
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
        "duration": 1.0, "fps": 9, "width": 128, "height": 128
    }));
    let decoded = generate_stub_video(&request, 11);
    assert!(decoded.audio.is_some());
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_vid_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("clip.mp4");
    encode_media(&media_path, decoded, None, None)
        .await
        .unwrap();
    assert!(media_path.exists(), "mp4 must be written");
    assert!(media_path.metadata().unwrap().len() > 0);
    assert!(
        media_path.with_extension("poster.jpg").exists(),
        "poster must be extracted"
    );
    // Intermediates are cleaned up.
    assert!(!media_path.with_extension("frames").exists());
    assert!(!media_path.with_extension("enc.mp4").exists());
    assert!(!media_path.with_extension("audio.wav").exists());
}

#[tokio::test]
async fn encode_media_rejects_malformed_raw_frame_before_starting_ffmpeg() {
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_vid_bad_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("clip.mp4");
    let decoded = DecodedVideo {
        frames: vec![RgbFrame {
            width: 4,
            height: 4,
            pixels: vec![0; 4 * 4 * 3 - 1],
        }],
        fps: 24,
        audio: None,
        adapter_apply_reports: Vec::new(),
    };

    let error = encode_media(&media_path, decoded, None, None)
        .await
        .expect_err("malformed RGB buffer must be rejected before FFmpeg");

    assert!(error.to_string().contains("frame buffer size mismatch"));
    assert!(!media_path.exists());
}

fn ffmpeg_reachable() -> bool {
    if let Ok(path) = std::env::var("SCENEWORKS_FFMPEG") {
        if !path.trim().is_empty() && Path::new(&path).exists() {
            return true;
        }
    }
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// F-118: video upscale accepts only 2x and 4x. Any other factor is rejected with a clear error
/// rather than silently coerced to 2x (which produced a quietly-different output). Compiled on
/// both the macOS and candle lanes, matching the gate on [`resolve_video_upscale_factor`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn video_upscale_factor_accepts_2_and_4_rejects_others() {
    assert_eq!(resolve_video_upscale_factor(2).expect("2x"), 2);
    assert_eq!(resolve_video_upscale_factor(4).expect("4x"), 4);
    for bad in [0u8, 1, 3, 5, 8] {
        let err = resolve_video_upscale_factor(bad)
            .expect_err("unsupported factor must be rejected, not coerced");
        assert!(
            matches!(err, WorkerError::InvalidPayload(ref m) if m.contains("factor 2 or 4")),
            "factor {bad} should yield a clear InvalidPayload error, got {err:?}"
        );
    }
}

// Candle video lane labeling + engine-mapping unit tests (sc-5099). Windows/candle-gated; pure maps.
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_video_label_tests {
    use super::*;

    #[test]
    fn candle_video_engine_ids_map_5b_ltx_and_14b() {
        assert_eq!(candle_video_engine_id("wan_2_2"), Some("wan2_2_ti2v_5b"));
        // The Wan2.2 14B dual-expert MoE pair (sc-5175): T2V (text→video) + I2V (image→video).
        assert_eq!(
            candle_video_engine_id("wan_2_2_t2v_14b"),
            Some("wan2_2_t2v_14b")
        );
        assert_eq!(
            candle_video_engine_id("wan_2_2_i2v_14b"),
            Some("wan2_2_i2v_14b")
        );
        // ltx maps to the candle distilled id, not the MLX `ltx_2_3`.
        assert_eq!(candle_video_engine_id("ltx_2_3"), Some("ltx_2_3_distilled"));
        // SVD maps to the candle `svd_xt` engine (sc-5493).
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        // eros now shares the one candle LTX-2.3 engine with the base (sc-5495 wired the eros
        // dense checkpoint through `ltx_2_3_distilled`; they differ only in `candle_video_repo`).
        assert_eq!(
            candle_video_engine_id("ltx_2_3_eros"),
            Some("ltx_2_3_distilled")
        );
        assert!(is_candle_video_engine("ltx_2_3_eros"));
        for model in [
            "wan_2_2",
            "wan_2_2_t2v_14b",
            "wan_2_2_i2v_14b",
            "ltx_2_3",
            "ltx_2_3_eros",
            "svd",
        ] {
            assert!(is_candle_video_engine(model), "{model}");
        }
    }

    #[test]
    fn candle_ltx_user_lora_resolves_and_reaches_the_load_spec() {
        let data_dir_guard = tempfile::Builder::new()
            .prefix("sw_candle_ltx_lora_")
            .tempdir()
            .expect("temp data dir");
        let data_dir = data_dir_guard.path();
        let lora_dir = data_dir.join("loras");
        std::fs::create_dir_all(&lora_dir).unwrap();
        let lora = lora_dir.join("style.safetensors");
        write_lora_fixture(&lora, None);
        let settings = Settings {
            data_dir: data_dir.to_path_buf(),
            ..Settings::from_env()
        };
        let request = request(json!({
            "projectId": "p",
            "model": "ltx_2_3",
            "mode": "text_to_video",
            "prompt": "a fox",
            "loras": [{ "path": lora.to_string_lossy(), "weight": "0.55" }],
        }));

        let adapters = resolve_ltx_user_adapters(&settings, &request)
            .expect("Candle LTX user LoRA should resolve");
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].path, lora.canonicalize().unwrap());
        assert_eq!(adapters[0].kind, gen_core::AdapterKind::Lora);
        assert!((adapters[0].scale - 0.55).abs() < 1e-6);
        assert!(adapters[0].pass_scales.is_none());

        let input = VideoGenInput {
            engine_id: "ltx_2_3_distilled",
            adapters,
            ..VideoGenInput::default()
        };
        let spec = video_load_spec(&input);
        assert_eq!(spec.adapters.len(), 1);
        assert_eq!(spec.adapters[0].path, lora.canonicalize().unwrap());
        assert!((spec.adapters[0].scale - 0.55).abs() < 1e-6);
    }

    #[test]
    fn candle_video_adapter_labels_are_per_family() {
        // Every wan engine (5B + 14B T2V/I2V) reports the shared `candle_wan` adapter.
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            assert_eq!(
                candle_video_adapter_label(engine_id),
                "candle_wan",
                "{engine_id}"
            );
        }
        assert_eq!(
            candle_video_adapter_label("ltx_2_3_distilled"),
            "candle_ltx"
        );
        assert_eq!(candle_video_adapter_label("svd_xt"), "candle_svd");
        // Mochi (sc-11992) must have its OWN arm. The `_` fall-through is the Wan default, so a
        // missing arm silently stamps `candle_wan` onto a Mochi asset's sidecar + telemetry — wrong
        // provenance with no error anywhere.
        assert_eq!(candle_video_adapter_label("mochi_1"), "candle_mochi");
        assert_ne!(
            candle_video_adapter_label("mochi_1"),
            CANDLE_WAN_ADAPTER,
            "mochi_1 must not fall through to the Wan adapter label"
        );
    }

    /// **The candle-lane routing trap** (sc-11992). B1 declared Windows served
    /// (`VideoModelCaps::new("mochi_1", true, true, false, false)` ⇒ `candle_video_routed`), and A6
    /// validated the lane on Blackwell against the very same hosted tiers. If `candle_video_engine_id`
    /// does not resolve `mochi_1`, `is_candle_video_engine` is false, `resolve_candle_video_route`
    /// never reaches its generic arm, and the job lands on `CandleVideoRoute::Stub` — handing the user
    /// a PROCEDURAL FAKE VIDEO rather than an error. That is a silent wrong-output bug, so it is
    /// pinned here rather than left to the descriptor.
    #[test]
    fn candle_mochi_resolves_the_engine_and_never_falls_to_the_stub() {
        // ONE engine id on BOTH backends — no `_distilled`-style split (contrast ltx above).
        assert_eq!(candle_video_engine_id("mochi_1"), Some("mochi_1"));
        assert!(
            is_candle_video_engine("mochi_1"),
            "mochi_1 MUST be a candle video engine or Windows silently renders a procedural stub"
        );

        // The generic candle arm is reached for a t2v mochi job (the shape B1's router allows
        // through: text_to_video, no source asset, no LoRA).
        let settings = Settings {
            backend_candle_enabled: true,
            ..Settings::from_env()
        };
        let payload = json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        });
        let req = VideoRequest::from_payload(payload.as_object().expect("object"));
        assert_eq!(
            resolve_candle_video_route(&req, &settings),
            CandleVideoRoute::CandleVideo,
            "a t2v mochi job must route to the candle video engine, NOT the stub"
        );
    }

    /// The candle default repo must be Mochi's own. No Mochi manifest entry carries a top-level
    /// `repo`, so `candle_video_repo` falls through to this default — and the `_` arm is Wan's, which
    /// would send the loader at a Wan diffusers snapshot.
    #[test]
    fn candle_mochi_default_repo_is_the_shared_mochi_turnkey() {
        assert_eq!(
            candle_video_default_repo("mochi_1"),
            "SceneWorks/mochi-1-mlx"
        );
        assert_ne!(
            candle_video_default_repo("mochi_1"),
            CANDLE_WAN_5B_REPO,
            "mochi_1 must not inherit Wan's default repo"
        );
        // ONE repo serves BOTH backends — the candle default IS the MLX repo (A6's `.scales`-detect
        // seam ingests the mlx-affine tiers 1:1; `SceneWorks/mochi-1-candle` was never published).
        assert_eq!(candle_video_default_repo("mochi_1"), MOCHI_REPO);

        // With no manifest `repo` (the shipped Mochi entry), the resolved repo is that default.
        let payload = json!({
            "projectId": "p", "model": "mochi_1", "mode": "text_to_video", "prompt": "p"
        });
        let req = VideoRequest::from_payload(payload.as_object().expect("object"));
        assert_eq!(candle_video_repo(&req, "mochi_1"), "SceneWorks/mochi-1-mlx");
    }

    #[test]
    fn candle_video_default_repos_are_per_engine() {
        assert_eq!(
            candle_video_default_repo("wan2_2_ti2v_5b"),
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_t2v_14b"),
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("wan2_2_i2v_14b"),
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        );
        assert_eq!(
            candle_video_default_repo("ltx_2_3_distilled"),
            "SceneWorks/ltx-2.3-mlx"
        );
        // SVD-XT loads the stock diffusers img2vid-xt snapshot directly (sc-5493).
        assert_eq!(
            candle_video_default_repo("svd_xt"),
            "stabilityai/stable-video-diffusion-img2vid-xt"
        );
    }

    #[test]
    fn candle_ltx_resolves_the_shared_q4_turnkey_tier_only_for_the_base_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        let q4 = temp.path().join("q4");
        std::fs::create_dir_all(&q4).expect("q4");
        std::fs::write(q4.join("transformer.safetensors"), "x").expect("transformer");
        std::fs::write(q4.join("quantize_config.json"), "{}").expect("quant config");

        let (resolved, quant) = candle_ltx_tier_subdir(temp.path(), "ltx_2_3_distilled", "ltx_2_3")
            .expect("base LTX q4 tier");
        assert_eq!(resolved, q4);
        assert_eq!(
            quant, None,
            "the q4 files are already packed; the provider must not quantize them again"
        );

        // Drive the exact production hand-off into the provider. The resolver's marker is copied to
        // `VideoGenInput::quant`, then `video_load_spec` copies that field onto `LoadSpec::quantize`.
        // Pinning the final spec prevents a future tier-selection refactor from reintroducing Q4
        // on-the-fly quantization while a resolver-only assertion still passes.
        let input = VideoGenInput {
            engine_id: "ltx_2_3_distilled",
            model_dir: resolved,
            quant,
            ..VideoGenInput::default()
        };
        let spec = video_load_spec(&input);
        assert!(
            spec.quantize.is_none(),
            "production LTX provider spec must load the pre-packed q4 tier with quantize=None"
        );
        assert!(
            candle_ltx_tier_subdir(temp.path(), "ltx_2_3_distilled", "ltx_2_3_eros").is_none(),
            "Eros stays on its dense standalone checkpoint"
        );

        std::fs::remove_file(q4.join("quantize_config.json")).expect("tear tier");
        assert!(
            candle_ltx_tier_subdir(temp.path(), "ltx_2_3_distilled", "ltx_2_3").is_none(),
            "a partial q4 tier is never selected"
        );
    }

    #[test]
    fn candle_video_engine_ids_map_svd() {
        assert_eq!(candle_video_engine_id("svd"), Some("svd_xt"));
        assert!(is_candle_video_engine("svd"));
    }

    /// sc-14625: the complete default recipe proven on 32 GB must pass the real candle preflight and
    /// retain the routing/recipe provenance required by the story.
    #[test]
    fn candle_svd_default_profile_preserves_provenance_after_vram_preflight() {
        let request = request(json!({
            "projectId": "p",
            "model": "svd",
            "mode": "image_to_video",
            "width": 1024,
            "height": 576,
            "advanced": { "numFrames": 25, "decodeChunkSize": 8, "steps": 25 }
        }));
        let preflight = svd_vram_preflight(
            &request,
            "0",
            crate::vram_gate::apply_vram_cap(None, Some(32.0)),
        )
        .expect("the real-hardware-validated default profile is admitted");
        assert_eq!(preflight.frames, 25);
        assert_eq!(preflight.decode_chunk_size, 8);
        assert_eq!(preflight.steps, 25);
        assert_eq!(candle_video_engine_id(&request.model), Some("svd_xt"));
        assert_eq!(candle_video_adapter_label("svd_xt"), "candle_svd");

        let raw = svd_raw_settings(&request);
        assert_eq!(raw["model"], json!("svd"));
        assert_eq!(raw["numFrames"], json!(25));
        assert_eq!(raw["decodeChunkSize"], json!(8));
        assert_eq!(raw["steps"], json!(25));
        assert_eq!(raw["realModelInference"], json!(true));
    }

    /// The production-order pin remains discriminating after admitting the measured default: a recipe
    /// just beyond the measured boundary must still refuse before snapshot resolution, source-image
    /// IO, or the supplied loader.
    #[test]
    fn generate_candle_svd_refuses_an_unvalidated_32gb_recipe_before_loading() {
        let probe = ArmProbe::default();
        let request = request(json!({
            "projectId": "p",
            "model": "svd",
            "mode": "image_to_video",
            "quality": "fast",
            "width": 1024,
            "height": 576,
            "advanced": { "numFrames": 25, "decodeChunkSize": 9, "steps": 25 }
        }));
        let settings = offline_settings();
        let job = mochi_job_snapshot();
        let loader = probe.loader();
        let out = temp_env_var(crate::vram_gate::CUDA_VRAM_CAP_ENV, "32", || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("test runtime builds")
                .block_on(generate_candle_video_using(
                    &ApiClient::new(&settings),
                    &settings,
                    &job,
                    &request,
                    std::path::Path::new(""),
                    "candle",
                    loader,
                ))
        });

        let message = out
            .err()
            .expect("the over-limit SVD recipe must be rejected on a 32 GB card")
            .to_string();
        assert!(
            message.contains("rejected before model load")
                && message.contains("decode-chunk-9")
                && message.contains("decodeChunkSize=8")
                && message.contains("25 steps"),
            "the production arm must surface the measured SVD boundary: {message}"
        );
        assert!(
            !probe.loaded(),
            "the SVD preflight must refuse before snapshot resolution and engine load"
        );
    }

    #[test]
    fn candle_video_conditioning_only_for_i2v() {
        let settings = crate::Settings::from_env();
        let project_path = std::path::Path::new("");
        // txt2video engines never build conditioning (even if a stray source asset is present).
        for engine_id in ["wan2_2_ti2v_5b", "wan2_2_t2v_14b", "ltx_2_3_distilled"] {
            let payload = json!({ "sourceAssetId": "asset_1" });
            let request = VideoRequest::from_payload(payload.as_object().expect("object"));
            let conditioning =
                resolve_candle_video_conditioning(&settings, &request, project_path, engine_id)
                    .expect("txt2video conditioning resolves");
            assert!(
                conditioning.is_empty(),
                "{engine_id} must be txt2video-only"
            );
        }
        // The 14B I2V + SVD-XT require a source image — a request without one errors before touching
        // disk (sc-5175 / sc-5493).
        for engine_id in ["wan2_2_i2v_14b", "svd_xt"] {
            let payload = json!({ "mode": "image_to_video" });
            let no_source = VideoRequest::from_payload(payload.as_object().expect("object"));
            assert!(
                resolve_candle_video_conditioning(&settings, &no_source, project_path, engine_id)
                    .is_err(),
                "{engine_id} without a source image must error"
            );
        }
    }

    /// The candle Wan Lightning toggle (sc-10138) drives the sampling recipe: A14B MoE default-on
    /// (absent flag) ⇒ 4-step / CFG-off; explicit opt-out ⇒ native multi-step CFG (engine defaults, or a
    /// user override); the dense 5B has no toggle and applies the interim step default.
    #[test]
    fn candle_wan_lightning_toggle_drives_sampling() {
        let req = |advanced: Value| {
            VideoRequest::from_payload(json!({ "advanced": advanced }).as_object().unwrap())
        };

        for moe in ["wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            // Default-on (absent flag): the 4-step / CFG-off Lightning recipe.
            let on = req(json!({}));
            assert!(wan_lightning_on(moe, &on), "{moe} default-on");
            assert_eq!(
                wan_sampling(moe, &on),
                (Some(4), Some(1.0)),
                "{moe} lightning recipe"
            );

            // Explicit opt-out → native multi-step CFG: no user override ⇒ (None, None) so the engine
            // config defaults stand; a user override is honored verbatim.
            let off = req(json!({ "lightning": false }));
            assert!(!wan_lightning_on(moe, &off), "{moe} opt-out");
            assert_eq!(
                wan_sampling(moe, &off),
                (None, None),
                "{moe} native defaults"
            );
            let off_override =
                req(json!({ "lightning": false, "steps": 30, "guidanceScale": 3.5 }));
            assert_eq!(wan_sampling(moe, &off_override), (Some(30), Some(3.5)));
        }

        // Dense TI2V-5B: no Lightning toggle (always off), interim step default, user override wins.
        let five_b = req(json!({}));
        assert!(!wan_lightning_on("wan2_2_ti2v_5b", &five_b));
        assert_eq!(
            wan_sampling("wan2_2_ti2v_5b", &five_b),
            (Some(WAN5B_INTERIM_STEPS), None)
        );
        assert_eq!(
            wan_sampling("wan2_2_ti2v_5b", &req(json!({ "steps": 12 }))).0,
            Some(12)
        );
    }

    /// (sc-10027 / sc-10726) `candle_wan_tier_subdir` picks the q4/q8 subdir per `advanced.mlxQuantize`
    /// (default q8 when installed, clamped to q4), falls back through the tier order, requires a complete
    /// tier, and returns `None` for a non-wan engine or a flat/dense repo with no tier subdirs.
    #[test]
    fn candle_wan_tier_subdir_resolves_by_mlx_quantize() {
        let root_guard = tempfile::Builder::new()
            .prefix("sc10027_wan_tier_")
            .tempdir()
            .expect("temp dir");
        let root = root_guard.path();
        // Complete q4 + q8 A14B tiers (transformer + transformer_2 + text_encoder + vae + tokenizer).
        for tier in ["q4", "q8"] {
            for sub in [
                "transformer",
                "transformer_2",
                "text_encoder",
                "vae",
                "tokenizer",
            ] {
                std::fs::create_dir_all(root.join(tier).join(sub)).unwrap();
            }
        }
        let req = |adv: Value| {
            VideoRequest::from_payload(json!({ "advanced": adv }).as_object().unwrap())
        };

        // Default (no mlxQuantize) → q8 when installed (sc-10726), clamped to what's on disk.
        let (d, q) = candle_wan_tier_subdir(root, "wan2_2_t2v_14b", &req(json!({}))).unwrap();
        assert!(d.ends_with("q8"));
        assert_eq!(q, Some(Quant::Q8));
        // An explicit Q4 pick stays q4 (never overridden by the q8 default).
        let (d, q) =
            candle_wan_tier_subdir(root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 4 })))
                .unwrap();
        assert!(d.ends_with("q4"));
        assert_eq!(q, Some(Quant::Q4));
        // mlxQuantize = 8 → q8.
        let (d, q) =
            candle_wan_tier_subdir(root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 8 })))
                .unwrap();
        assert!(d.ends_with("q8"));
        assert_eq!(q, Some(Quant::Q8));
        // bf16 requested but no bf16 tier present → falls back through q8/q4 (never a missing dir).
        let (d, _) =
            candle_wan_tier_subdir(root, "wan2_2_t2v_14b", &req(json!({ "mlxQuantize": 0 })))
                .unwrap();
        assert!(d.ends_with("q8") || d.ends_with("q4"));
        // Non-wan engine → None (ltx loads its flat snapshot).
        assert!(candle_wan_tier_subdir(root, "ltx_2_3_distilled", &req(json!({}))).is_none());
        // A flat repo (the dense Wan-AI/*-Diffusers fallback — no tier subdirs) → None.
        let flat_guard = tempfile::Builder::new()
            .prefix("sc10027_flat_")
            .tempdir()
            .expect("temp dir");
        let flat = flat_guard.path();
        assert!(candle_wan_tier_subdir(flat, "wan2_2_t2v_14b", &req(json!({}))).is_none());
    }
}

// ---------------------------------------------------------------------------
// The embedded workflow envelope (sc-15956)
// ---------------------------------------------------------------------------

/// Build the envelope a clip of `payload` would carry, and write it where the encoder reads it.
fn workflow_document(directory: &Path, payload: serde_json::Map<String, Value>) -> PathBuf {
    use sceneworks_core::workflow_share::{embeddable_video_workflow_share, WorkflowAssetFacts};
    let facts = WorkflowAssetFacts {
        mode: "text_to_video".to_owned(),
        model: "ltx_2_3".to_owned(),
        prompt: "a fox crossing a frozen river".to_owned(),
        negative_prompt: String::new(),
        seed: 4242,
        width: Some(128),
        height: Some(128),
    };
    let share = embeddable_video_workflow_share(&facts, &payload).expect("under the ceiling");
    let path = directory.join("workflow.ffmeta");
    sceneworks_core::workflow_mp4::write_workflow_metadata_file(&share, &path).expect("written");
    path
}

fn video_workflow_payload() -> serde_json::Map<String, Value> {
    json!({
        "mode": "text_to_video",
        "model": "ltx_2_3",
        "prompt": "a fox crossing a frozen river",
        "duration": 1.0,
        "fps": 9,
        "quality": "balanced",
        "width": 128,
        "height": 128,
        "personTrackId": "track_9f3c1b2a",
        "replacementMode": "full_person_replace_outfit",
        "advanced": {
            "motion": "slow push-in",
            "precision": "fp8",
            "replacementModeLabel": "Full Person, Replace Outfit",
            "selectedPersonTrack": { "name": "Sarah Whitfield" }
        }
    })
    .as_object()
    .expect("an object")
    .clone()
}

/// **The end-to-end proof.** The envelope written before the encode is readable out of the
/// finished mp4, after every step the encoder runs.
///
/// That chain is three ffmpeg invocations, not one — the libx264 encode, the AAC audio mux, and the
/// `+faststart` remux — and the tag has to survive all three. The audio arm matters most: it is the
/// only step with two inputs, and ffmpeg's multi-input metadata default is the exact behaviour that
/// bites the timeline export (see `media_jobs::export_metadata_tests`).
///
/// Skips when no ffmpeg is reachable, like every other real-binary test in this file.
#[tokio::test]
async fn the_envelope_survives_the_whole_encode_chain() {
    if !ffmpeg_reachable() {
        eprintln!("skipping the_envelope_survives_the_whole_encode_chain: ffmpeg not found");
        return;
    }
    // `ltx_2_3` makes the stub emit an audio track, so the two-input mux step runs.
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
        "duration": 1.0, "fps": 9, "width": 128, "height": 128
    }));
    let decoded = generate_stub_video(&request, 11);
    assert!(decoded.audio.is_some(), "the audio arm must actually run");

    let dir_guard = tempfile::Builder::new()
        .prefix("sw_wf_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("clip.mp4");
    let document = workflow_document(dir, video_workflow_payload());

    encode_media(&media_path, decoded, Some(document.as_path()), None)
        .await
        .unwrap();

    let read = sceneworks_core::workflow_mp4::read_workflow_metadata_file(&media_path)
        .expect("the clip is readable")
        .expect("the clip carries its recipe through encode + audio mux + faststart");
    assert_eq!(
        read.kind,
        sceneworks_core::workflow_share::WORKFLOW_KIND_VIDEO
    );
    assert_eq!(read.prompt, "a fox crossing a frozen river");
    assert_eq!(read.duration_seconds, Some(1.0));
    assert_eq!(read.fps, Some(9));
    assert_eq!(read.advanced.get("motion"), Some(&json!("slow push-in")));

    // The clip still plays: the tag did not displace the moov or break the framing.
    assert!(media_path.metadata().unwrap().len() > 0);
    assert!(media_path.with_extension("poster.jpg").exists());

    // And the withholding holds all the way to the bytes on disk — the acceptance criterion, read
    // out of a real file rather than off a struct.
    let bytes = std::fs::read(&media_path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    for secret in [
        "track_9f3c1b2a",
        "Sarah Whitfield",
        "full_person_replace_outfit",
        "Full Person, Replace Outfit",
        "selectedPersonTrack",
    ] {
        assert!(
            !text.contains(secret),
            "`{secret}` reached the mp4 on disk — a shared video must not disclose that it was \
             made by replacing a specific person"
        );
    }
    // A hardware-budget knob is dropped too, so this is the allow-list running and not a blanket.
    assert!(
        !text.contains("fp8"),
        "`precision` is a hardware budget and must not travel"
    );
}

/// `None` still writes exactly the file it wrote before — the setting-off / no-payload path.
#[tokio::test]
async fn no_document_means_no_tag_and_an_otherwise_identical_clip() {
    if !ffmpeg_reachable() {
        eprintln!("skipping no_document_means_no_tag_and_an_otherwise_identical_clip: no ffmpeg");
        return;
    }
    let request = request(json!({
        "projectId": "p", "model": "wan2_2_t2v_14b", "prompt": "fox",
        "duration": 1.0, "fps": 9, "width": 128, "height": 128
    }));
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_wf_off_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let media_path = dir.join("clip.mp4");
    encode_media(&media_path, generate_stub_video(&request, 11), None, None)
        .await
        .unwrap();
    assert!(media_path.exists());
    assert_eq!(
        sceneworks_core::workflow_mp4::read_workflow_metadata_file(&media_path).expect("readable"),
        None,
        "with the setting off there must be no recipe in the file at all"
    );
}

/// **The survival measurement, as a test.** The tag outlives a representative re-encode, and dies
/// on a deliberate strip.
///
/// The negative case is asserted alongside the positive one on purpose: "it survives everything"
/// would be a false claim about this container, and the story asks for the honest matrix. What is
/// proved here is the boundary — a transcode that carries metadata forward keeps the recipe, and a
/// pipeline that strips metadata removes it, which is what a platform re-encode does.
#[tokio::test]
async fn the_tag_survives_a_re_encode_and_dies_on_a_deliberate_strip() {
    if !ffmpeg_reachable() {
        eprintln!(
            "skipping the_tag_survives_a_re_encode_and_dies_on_a_deliberate_strip: no ffmpeg"
        );
        return;
    }
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "fox",
        "duration": 1.0, "fps": 9, "width": 128, "height": 128
    }));
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_wf_re_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let source = dir.join("clip.mp4");
    let document = workflow_document(dir, video_workflow_payload());
    encode_media(
        &source,
        generate_stub_video(&request, 11),
        Some(document.as_path()),
        None,
    )
    .await
    .unwrap();

    let read_back = |path: &Path| {
        sceneworks_core::workflow_mp4::read_workflow_metadata_file(path).expect("readable")
    };
    assert!(read_back(&source).is_some(), "the source must carry it");

    // 1. A `-c copy` remux — a container rewrite, what a "fix this file" tool does.
    let remux = dir.join("remux.mp4");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            remux.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .unwrap();
    assert!(
        read_back(&remux).is_some(),
        "a `-c copy` remux must keep it"
    );

    // 2. A scale + bitrate change — the shape of a delivery transcode.
    let scaled = dir.join("scaled.mp4");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-vf".to_owned(),
            "scale=64:64".to_owned(),
            "-b:v".to_owned(),
            "200k".to_owned(),
            scaled.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .unwrap();
    let survived = read_back(&scaled).expect("a scale + bitrate re-encode must keep it");
    assert_eq!(
        survived.prompt, "a fox crossing a frozen river",
        "and keep it INTACT, not truncated"
    );

    // 3. THE NEGATIVE CASE: a pipeline that strips metadata. This is what a platform re-encode
    //    does, and it is why the sharing premise for video is weaker than for a PNG chunk.
    let stripped = dir.join("stripped.mp4");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-map_metadata".to_owned(),
            "-1".to_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            stripped.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        read_back(&stripped),
        None,
        "a deliberate strip removes it — stated as a fact of this container rather than hidden"
    );

    // 4. THE SECOND NEGATIVE (sc-15956 review): HLS segmentation, at `-c copy` throughout. Nothing
    //    is re-encoded and the streams are untouched, so this is the case a reader would predict
    //    survives — and it does not, because MPEG-TS has no `ilst` to carry the tag through and the
    //    mp4 is rebuilt from segments that never held it. Named rather than inferred, because
    //    "I only re-packaged it" is exactly the reasoning that gets this wrong.
    let segments = dir.join("hls");
    std::fs::create_dir_all(&segments).unwrap();
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            source.to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            "-f".to_owned(),
            "hls".to_owned(),
            "-hls_time".to_owned(),
            "1".to_owned(),
            "-hls_list_size".to_owned(),
            "0".to_owned(),
            "-hls_segment_filename".to_owned(),
            segments.join("seg%03d.ts").to_string_lossy().into_owned(),
            segments.join("out.m3u8").to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .unwrap();
    let reassembled = dir.join("reassembled.mp4");
    run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            segments.join("out.m3u8").to_string_lossy().into_owned(),
            "-c".to_owned(),
            "copy".to_owned(),
            reassembled.to_string_lossy().into_owned(),
        ],
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        read_back(&reassembled),
        None,
        "an HLS round trip loses the tag even at `-c copy` — MPEG-TS has no `ilst`, so the mp4 is          rebuilt from segments that never carried it. Recorded as a measured limit of this          container, not hidden behind \"it survives a remux\"."
    );
}

// ---------------------------------------------------------------------------
// The SECOND libx264 site: the video upscale (sc-15956 review)
// ---------------------------------------------------------------------------

/// Worker settings pointed at a scratch config dir, so the embed toggle is this test's rather than
/// the install's. The preference defaults to ON when the file is absent, which is what an empty
/// scratch dir gives.
fn upscale_settings(config_dir: &Path) -> Settings {
    let mut settings = Settings::from_env();
    settings.config_dir = config_dir.to_path_buf();
    settings
}

/// A `video_upscale` job carrying the payload `VideoUpscalePanel.onUpscale` really posts.
fn upscale_job_snapshot(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job-upscale-1",
        "type": "video_upscale",
        "status": "running",
        "projectId": "p",
        "projectName": null,
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": null,
        "workerId": "test-worker",
        "progress": 0.0,
        "stage": "queued",
        "message": "queued",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-07-16T00:00:00Z",
        "updatedAt": "2026-07-16T00:00:00Z",
        "startedAt": null,
        "finishedAt": null
    }))
    .expect("the upscale job snapshot parses")
}

/// The payload the panel posts, with the source clip's display name in it — which is routinely
/// somebody's original filename, and must not travel.
fn upscale_payload() -> Value {
    json!({
        "projectId": "p",
        "sourceAssetId": "asset_9f3c1b2a",
        "factor": 4,
        "engine": "seedvr2",
        "model": "seedvr2_3b",
        "softness": 0.25,
        "displayName": "sarah-audition-take3 (4x upscaled)"
    })
}

/// **The upscale lane's end-to-end proof.** An upscaled clip carries a recipe of its OWN.
///
/// sc-15956 shipped with `encode_media` described as "the ONE funnel every generated clip is
/// encoded through". There are two libx264 sites in this crate, and this was the other one: a
/// `video_upscale` wrote a clip with no envelope at all while the image lane's exact analogue
/// (`upscale_jobs::run_image_upscale_job`) embedded one. The seam lint could not see it, because it
/// discovers by `WorkflowShare` mentions and this file had none.
///
/// What is asserted is the whole chain the job runs: build the envelope from the real payload,
/// write the `ffmetadata` document, encode a PNG sequence through `encode_seedvr2_stream`, and read
/// the recipe back out of the finished mp4.
#[tokio::test]
async fn the_upscaled_clip_carries_its_own_recipe() {
    if !ffmpeg_reachable() {
        eprintln!("skipping the_upscaled_clip_carries_its_own_recipe: ffmpeg not found");
        return;
    }
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_up_wf_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let frames = dir.join("frames");
    std::fs::create_dir_all(&frames).unwrap();
    for index in 0..3_u32 {
        image::RgbImage::from_pixel(64, 64, image::Rgb([16, 32, 48]))
            .save(frames.join(format!("frame_{index:05}.png")))
            .unwrap();
    }

    let job = upscale_job_snapshot(upscale_payload());
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(upscale_payload()).unwrap();
    let settings = upscale_settings(&dir.join("config"));
    let media_path = dir.join("upscaled.mp4");
    let document = media_path.with_extension("workflow.ffmeta");

    let embedded = seedvr2_workflow_metadata(
        &settings, &job, &req, "seedvr2", 4, 8080, 640, 360, &document,
    );
    assert!(
        embedded,
        "with the preference at its default the upscale lane must embed"
    );
    assert!(document.exists(), "the ffmetadata document must be written");

    encode_seedvr2_stream(&media_path, &frames, 3, 8, Some(document.as_path()), None)
        .await
        .unwrap();

    let read = sceneworks_core::workflow_mp4::read_workflow_metadata_file(&media_path)
        .expect("the upscaled clip is readable")
        .expect("...and carries its recipe");
    assert_eq!(
        read.kind,
        sceneworks_core::workflow_share::WORKFLOW_KIND_VIDEO
    );
    // THIS pass, not the source clip's: an upscale has no prompt, and its mode and model name the
    // upscaler rather than whatever generated the input.
    assert_eq!(read.mode, "video_upscale");
    assert_eq!(read.model, "seedvr2_3b");
    assert_eq!(read.prompt, "");
    let upscale = read.upscale.expect("the applied pass is recorded");
    assert!(upscale.enabled);
    assert_eq!(upscale.factor, Some(4));
    assert_eq!(upscale.engine.as_deref(), Some("seedvr2"));
    assert_eq!(upscale.softness, Some(0.25));
    // The geometry is the SOURCE geometry — the recipe is "take this clip and upscale it 4x".
    assert_eq!((read.width, read.height), (Some(640), Some(360)));
    // And the input travels by SHAPE: a clip to start from, never the asset id.
    assert_eq!(read.inputs.len(), 1);
    assert_eq!(
        read.inputs[0].kind,
        sceneworks_core::workflow_share::INPUT_KIND_SOURCE_CLIP
    );
    assert_eq!(read.inputs[0].count, 1);

    // The two things in that payload that must never reach a shared file.
    let bytes = std::fs::read(&media_path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    for secret in ["asset_9f3c1b2a", "sarah-audition-take3"] {
        assert!(
            !text.contains(secret),
            "`{secret}` reached the upscaled mp4 on disk"
        );
    }

    assert!(media_path.with_extension("poster.jpg").exists());
}

/// The same lane with the preference OFF writes exactly the clip it wrote before.
#[tokio::test]
async fn an_upscale_with_the_setting_off_carries_nothing() {
    if !ffmpeg_reachable() {
        eprintln!("skipping an_upscale_with_the_setting_off_carries_nothing: ffmpeg not found");
        return;
    }
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_up_off_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();
    let frames = dir.join("frames");
    let config = dir.join("config");
    std::fs::create_dir_all(&frames).unwrap();
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        sceneworks_core::app_paths::ui_preferences_file(&config),
        r#"{"embedWorkflowInImages":false}"#,
    )
    .unwrap();
    for index in 0..2_u32 {
        image::RgbImage::from_pixel(64, 64, image::Rgb([8, 8, 8]))
            .save(frames.join(format!("frame_{index:05}.png")))
            .unwrap();
    }

    let job = upscale_job_snapshot(upscale_payload());
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(upscale_payload()).unwrap();
    let settings = upscale_settings(&config);
    let media_path = dir.join("upscaled.mp4");
    let document = media_path.with_extension("workflow.ffmeta");

    assert!(
        !seedvr2_workflow_metadata(&settings, &job, &req, "seedvr2", 4, 1, 640, 360, &document),
        "the video upscale honours the SAME switch as every other lane"
    );
    assert!(
        !document.exists(),
        "and writes no metadata document when it is off"
    );

    encode_seedvr2_stream(&media_path, &frames, 2, 8, None, None)
        .await
        .unwrap();
    assert_eq!(
        sceneworks_core::workflow_mp4::read_workflow_metadata_file(&media_path).expect("readable"),
        None
    );
}

/// The `.workflow.ffmeta` scratch is removed BEFORE the encode's error propagates, on both lanes.
///
/// A structural pin rather than a behavioural one, for the reason `media_jobs`'s
/// "`mux_segments` must pass `-map_metadata` explicitly" is: the failure is an ORDERING inside an
/// async job function that needs a live `ApiClient` to reach, and the ordering is exactly what a
/// future edit would undo. `encode_media(...).await?` used to sit in front of the `remove_file`, so
/// a failed or cancelled encode left the full envelope — prompt in plaintext — in the user's
/// project permanently, beside no video.
#[test]
fn the_workflow_metadata_scratch_does_not_outlive_a_failed_encode() {
    for (name, source, call) in [
        (
            "video_jobs/mod.rs",
            include_str!("mod.rs"),
            "let encoded = encode_media(",
        ),
        (
            "video_jobs/seedvr2.rs",
            include_str!("seedvr2.rs"),
            "let encode_result = encode_seedvr2_stream(",
        ),
    ] {
        let start = source
            .find(call)
            .unwrap_or_else(|| panic!("{name} no longer binds its encode result: {call}"));
        let rest = &source[start..];
        let removal = rest
            .find("remove_file(&workflow_metadata)")
            .unwrap_or_else(|| panic!("{name} never removes the workflow metadata scratch"));
        let propagation = rest
            .find("?;")
            .unwrap_or_else(|| panic!("{name} never propagates the encode result"));
        assert!(
            removal < propagation,
            "{name} propagates the encode error before removing `*.workflow.ffmeta`. \
             That leaves the whole envelope — the prompt in plaintext — in the user's project \
             after a failed or cancelled encode. Bind the result, remove the scratch, THEN `?`."
        );
    }
}

/// Every person-shaped string a hostile payload can carry, checked against the PUBLISHED bytes of
/// both encode lanes and of the poster frame beside them (sc-15956 review).
///
/// The story's acceptance criterion is about a file, so this reads files. The existing
/// `the_envelope_survives_the_whole_encode_chain` greps five strings out of one clip; this is the
/// widened form the review asked to see re-run after the second write seam landed: every id, name,
/// filename, path, label and KEY NAME the payload carries, against the generated clip, the upscaled
/// clip, and both posters — a poster is a separate file written from the same source and is just as
/// shareable.
///
/// Key names are in the list on purpose. A key that survived would mean the allow-list emitted an
/// empty-valued entry rather than dropping the field, which is the shape a sanitizer regression
/// takes before it starts leaking values.
const PERSON_TRACK_STRINGS: &[&str] = &[
    "track_9f3c1b2a",
    "Sarah Whitfield",
    "Whitfield",
    "sarah-audition-take3.mov",
    "sarah-audition-take3",
    "asset_7c1d3e",
    "D:/Clients/Acme",
    "Clients/Acme",
    "/Users/michael",
    "0001.png",
    "0002.png",
    "selectedPersonTrack",
    "sourceDisplayName",
    "replacementModeLabel",
    "Full Person, Keep Outfit",
    "full_person_keep_outfit",
    "personTrackId",
    "replacementMode",
    "timelineContext",
    "Acme Q4 Cut",
    "tl_44ff21",
    "quantTier",
];

/// The widest person-replacement payload the Video Studio can post.
fn hostile_person_payload() -> serde_json::Map<String, Value> {
    json!({
        "mode": "replace_person",
        "model": "wan_2_2_vace_fun_14b",
        "prompt": "the same shot, continuous motion",
        "duration": 1.0,
        "fps": 9,
        "quality": "balanced",
        "width": 128,
        "height": 128,
        "personTrackId": "track_9f3c1b2a",
        "replacementMode": "full_person_keep_outfit",
        "advanced": {
            "motion": "static",
            "quantTier": "q8",
            "replacementModeLabel": "Full Person, Keep Outfit",
            "selectedPersonTrack": {
                "id": "track_9f3c1b2a",
                "name": "Sarah Whitfield",
                "sourceDisplayName": "sarah-audition-take3.mov",
                "sourceAssetId": "asset_7c1d3e",
                "frames": [
                    { "mask": "D:/Clients/Acme/masks/0001.png" },
                    { "mask": "/Users/michael/masks/0002.png" }
                ]
            },
            "timelineContext": {
                "timelineId": "tl_44ff21",
                "timelineName": "Acme Q4 Cut",
                "sourceAssetId": "asset_7c1d3e"
            }
        }
    })
    .as_object()
    .expect("an object")
    .clone()
}

fn assert_no_person_track_strings(path: &Path, what: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{what} reads: {error}"));
    let text = String::from_utf8_lossy(&bytes);
    for secret in PERSON_TRACK_STRINGS {
        assert!(
            !text.contains(secret),
            "`{secret}` reached {what} on disk. A shared video must not disclose WHO was replaced, \
             which track, or which variant of a replacement ran."
        );
    }
}

#[tokio::test]
async fn no_person_track_string_reaches_a_published_clip_or_its_poster() {
    if !ffmpeg_reachable() {
        eprintln!(
            "skipping no_person_track_string_reaches_a_published_clip_or_its_poster: no ffmpeg"
        );
        return;
    }
    let dir_guard = tempfile::Builder::new()
        .prefix("sw_regrep_")
        .tempdir()
        .expect("temp dir");
    let dir = dir_guard.path();

    // Lane one: a GENERATED clip, through `encode_media` and its three ffmpeg steps.
    let request = request(json!({
        "projectId": "p", "model": "ltx_2_3", "prompt": "the same shot, continuous motion",
        "duration": 1.0, "fps": 9, "width": 128, "height": 128
    }));
    let generated = dir.join("generated.mp4");
    let document = workflow_document(dir, hostile_person_payload());
    encode_media(
        &generated,
        generate_stub_video(&request, 11),
        Some(document.as_path()),
        None,
    )
    .await
    .unwrap();
    assert!(
        sceneworks_core::workflow_mp4::read_workflow_metadata_file(&generated)
            .expect("readable")
            .is_some(),
        "the clip must be carrying an envelope, or this test proves nothing"
    );
    assert_no_person_track_strings(&generated, "the generated clip");
    assert_no_person_track_strings(&generated.with_extension("poster.jpg"), "its poster");

    // Lane two: an UPSCALED clip, through `encode_seedvr2_stream`. Its payload has no person track
    // in it at all — the point is that the SOURCE clip's recipe is never inherited, so nothing the
    // source disclosed can arrive by that route either.
    let frames = dir.join("frames");
    std::fs::create_dir_all(&frames).unwrap();
    for index in 0..2_u32 {
        image::RgbImage::from_pixel(64, 64, image::Rgb([9, 9, 9]))
            .save(frames.join(format!("frame_{index:05}.png")))
            .unwrap();
    }
    let job = upscale_job_snapshot(upscale_payload());
    let req: sceneworks_core::contracts::VideoUpscaleRequest =
        serde_json::from_value(upscale_payload()).unwrap();
    let settings = upscale_settings(&dir.join("config"));
    let upscaled = dir.join("upscaled.mp4");
    let upscale_document = upscaled.with_extension("workflow.ffmeta");
    assert!(seedvr2_workflow_metadata(
        &settings,
        &job,
        &req,
        "seedvr2",
        4,
        3,
        640,
        360,
        &upscale_document
    ));
    encode_seedvr2_stream(
        &upscaled,
        &frames,
        2,
        8,
        Some(upscale_document.as_path()),
        None,
    )
    .await
    .unwrap();
    assert!(
        sceneworks_core::workflow_mp4::read_workflow_metadata_file(&upscaled)
            .expect("readable")
            .is_some(),
        "the upscaled clip must be carrying an envelope too"
    );
    assert_no_person_track_strings(&upscaled, "the upscaled clip");
    assert_no_person_track_strings(&upscaled.with_extension("poster.jpg"), "its poster");
}
