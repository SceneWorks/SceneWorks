//! Physical MLX capture arm for the three Iris-3B task routes (sc-25690, epic sc-25678).
//!
//! THREE engine providers — generation (`iris_3b`), monocular depth (`iris_3b_depth`) and
//! restoration (`iris_3b_restore`) — and ONE arm, because the measurement is the same shape for all
//! three: bind the pinned upstream artifact, load it through the seam production loads it through,
//! read the LOADED model's own memory contract, drive the admission probes through that model's own
//! safety check, and measure three synchronized phase peaks off the boundaries the request already
//! emits. What differs is only how a route is loaded and driven:
//!
//! * **generation** is a registry [`Generator`]: the backbone at the snapshot root plus the staged
//!   Qwen3-VL text encoder, loaded through `runtime_macos::catalog().media()`. Conditioning ends at
//!   the first solver step; denoise ends at `Progress::Decoding`.
//! * **depth** is NOT a generator. It is `mlx-gen-iris`'s `IrisDepth` (one backbone forward on the
//!   export's shipped empty-prompt states), so it gets its own drive: ONE estimate at the plan
//!   geometry — the model-input grid `depth_memory_geometry` admits the request at.
//! * **restoration** is a `Transform`, also not a generator: `IrisRestorer`, driven for ONE tiled
//!   pass at the plan geometry — the processing size `restoration_memory_geometry` admits it at.
//!
//! The two downstream routes load no text encoder and compute nothing before their forward, so
//! their "conditioning" phase is the resident model with its empty-prompt states — the floor every
//! later phase stands on — "denoise" is the forward (every tile of it, for restoration) and
//! "decode" is the host-side finish (resize back / fusion, colour fix, quantize).
//!
//! Every route refuses, before any weight is read, a plan row that is not its own mode, the one
//! dense bf16 tier, a still single-image geometry, overlay-free and eagerly materialized — the
//! adapter's batch/frames validation, unchanged (`protocol::validate_iris_plan`). No Iris loader has
//! a deferred materialization shape; a deferred spec fails the load.

use super::*;
use gen_core_iris::depth::{plan_depth_input, DepthRequest, DepthResolution, IrisDepthEstimator};
use gen_core_iris::TEXT_ENCODER_COMPONENT;
use mlx_gen::gen_core::{InputSizing, TargetSize, Transform, TransformRequest};
use runtime_macos::providers::iris as engine;

/// The label the depth and restoration fixtures carry in their name. Neither request takes a
/// seed: both routes are deterministic, and the restorer refuses one by name. Generation renders
/// the turnkey still family's two-step calibration request.
const SEED: u64 = 25_690;
const GENERATION_SEED: u64 = TURNKEY_SEED;
const GENERATION_STEPS: u32 = TURNKEY_STEPS;
/// Restoration upscales by the release's default factor, so the source is the plan geometry / 4.
const RESTORE_SCALE: u32 = 4;

/// Generation: the backbone at the snapshot root, plus the staged text encoder.
const GENERATION: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_PROVIDER,
    mode: "text_to_image",
    repository_env: "SCENEWORKS_IRIS_3B_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_ROOT",
    subdir: None,
    slug: "iris-3b",
};
/// Monocular depth: the `depth/` export.
const DEPTH: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_DEPTH_PROVIDER,
    mode: "depth",
    repository_env: "SCENEWORKS_IRIS_3B_DEPTH_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_DEPTH_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_DEPTH_ROOT",
    subdir: Some(protocol::IRIS_3B_DEPTH_SUBDIR),
    slug: "iris-3b-depth",
};
/// Restoration: the `upscaler/` export.
const RESTORE: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_RESTORE_PROVIDER,
    mode: "restoration",
    repository_env: "SCENEWORKS_IRIS_3B_RESTORE_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_RESTORE_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_RESTORE_ROOT",
    subdir: Some(protocol::IRIS_3B_RESTORE_SUBDIR),
    slug: "iris-3b-restore",
};

/// The route a plan row's provider names. Only the three ids `run_with` dispatches here reach it.
pub(super) fn route(provider: &str) -> Result<protocol::IrisRoute, String> {
    [GENERATION, DEPTH, RESTORE]
        .into_iter()
        .find(|route| route.provider == provider)
        .ok_or_else(|| format!("the MLX Iris arm does not implement provider {provider:?}"))
}

/// The production calibration identity the loaded route publishes — the engine's own constant.
pub(super) fn calibration_fingerprint(route: protocol::IrisRoute) -> &'static str {
    match route.mode {
        "text_to_image" => engine::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
        "depth" => engine::memory_strategy::DEPTH_MEMORY_CALIBRATION_FINGERPRINT,
        _ => engine::memory_strategy::RESTORE_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// The fixture a plan row must name for `route` at one geometry.
pub(super) fn fixture(route: protocol::IrisRoute, width: u32, height: u32) -> String {
    let (seed, steps) = match route.mode {
        "text_to_image" => (GENERATION_SEED, GENERATION_STEPS),
        _ => (SEED, 0),
    };
    protocol::iris_fixture(route, "mlx", width, height, seed, steps)
}

/// The load spec production builds for `route` over a validated weights root: resident and eager,
/// with the staged text encoder for generation alone (the downstream engines refuse one).
pub(super) fn load_spec(
    route: protocol::IrisRoute,
    root: PathBuf,
    text_encoder: Option<PathBuf>,
) -> Result<LoadSpec, String> {
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    match (route.mode, text_encoder) {
        ("text_to_image", Some(dir)) => {
            Ok(spec.with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(dir)))
        }
        ("text_to_image", None) => Err(format!(
            "{} loads its Qwen3-VL text encoder as a staged component; {} is required",
            route.provider,
            protocol::IRIS_3B_TEXT_ENCODER_COMPONENT_ENV
        )),
        (_, Some(_)) => Err(format!(
            "{} runs on its export's shipped empty-prompt states and refuses a staged text encoder",
            route.provider
        )),
        (_, None) => Ok(spec),
    }
}

/// The two request boundaries a capture is split at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    /// Conditioning is finished; the denoise (the forward) begins.
    ConditioningDone,
    /// The last forward is finished; the host-side finish begins.
    ForwardDone,
}

/// A loaded Iris route, driven the way its own task is driven.
enum Loaded {
    Generation(Box<dyn Generator>),
    Depth(Box<engine::depth::IrisDepth>),
    Restore(Box<engine::IrisRestorer>),
}

impl Loaded {
    fn load(route: protocol::IrisRoute, spec: &LoadSpec) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| {
            format!("load real {} bf16 provider: {error}", route.provider)
        };
        match route.mode {
            "text_to_image" => runtime_macos::catalog()
                .map_err(|error| format!("build MLX catalog: {error}"))?
                .media()
                .load(route.provider, spec)
                .map(Self::Generation)
                .map_err(|error| failed(&error)),
            "depth" => engine::depth::load_depth(spec)
                .map(|model| Self::Depth(Box::new(model)))
                .map_err(|error| failed(&error)),
            _ => engine::IrisRestorer::load(spec)
                .map(|model| Self::Restore(Box::new(model)))
                .map_err(|error| failed(&error)),
        }
    }

    fn contract(&self) -> Result<&MemoryProviderContract, String> {
        match self {
            Self::Generation(generator) => generator
                .memory_strategy_contract()
                .ok_or_else(|| "loaded iris_3b exposed no memory-strategy contract".to_owned()),
            Self::Depth(model) => Ok(model.memory_strategy_contract()),
            Self::Restore(model) => Ok(model.memory_strategy_contract()),
        }
    }

    fn safety(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        match self {
            Self::Generation(generator) => generator.memory_strategy_safety_check(context),
            Self::Depth(model) => model.memory_strategy_safety_check(context),
            Self::Restore(model) => model.memory_strategy_safety_check(context),
        }
    }
}

/// One planned request of a loaded route: the geometry it is admitted at, and how to run it.
enum Planned {
    Generation(Box<GenerationRequest>),
    Depth(DepthRequest),
    Restore(TransformRequest),
}

/// The synthetic source image a downstream capture feeds its model.
fn source_image(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: protocol::synthetic_reference_rgb(width, height),
    }
}

/// Build the one request a route's capture runs at the plan geometry, and the geometry the ENGINE
/// admits that request at — which must be the plan's, or the record would price another shape.
fn plan_request(
    loaded: &Loaded,
    width: u32,
    height: u32,
) -> Result<(Planned, MemoryGeometry), String> {
    match loaded {
        Loaded::Generation(_) => Ok((
            Planned::Generation(Box::new(GenerationRequest {
                prompt: "a portrait of a person in a sunlit studio, editorial photograph"
                    .to_owned(),
                width,
                height,
                count: 1,
                seed: Some(GENERATION_SEED),
                steps: Some(GENERATION_STEPS),
                ..Default::default()
            })),
            MemoryGeometry {
                width,
                height,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
        )),
        Loaded::Depth(model) => {
            // The source IS the plan geometry and the cap is its long side, so the model-input grid
            // the engine plans is the plan geometry itself whenever it sits on the patch lattice.
            let request = DepthRequest {
                image: source_image(width, height),
                resolution: DepthResolution::Capped(width.max(height)),
                cancel: Default::default(),
            };
            let plan = plan_depth_input(
                width,
                height,
                request.resolution,
                model.export().config.model.patch_size,
            )
            .map_err(|error| format!("plan the depth input: {error}"))?;
            Ok((
                Planned::Depth(request),
                engine::memory_strategy::depth_memory_geometry(&plan),
            ))
        }
        Loaded::Restore(model) => {
            if !width.is_multiple_of(RESTORE_SCALE) || !height.is_multiple_of(RESTORE_SCALE) {
                return Err(format!(
                    "the restoration plan geometry {width}x{height} must be a multiple of the \
                     {RESTORE_SCALE}x capture scale"
                ));
            }
            let request = restoration_request(width, height);
            let plan = model
                .plan(&request)
                .map_err(|error| format!("plan the restoration request: {error}"))?;
            if plan.single_pass() {
                return Err(format!(
                    "the restoration plan geometry {width}x{height} fits one {}px tile; the anchor \
                     measures a TILED pass",
                    plan.tile
                ));
            }
            Ok((
                Planned::Restore(request),
                engine::memory_strategy::restoration_memory_geometry(&plan),
            ))
        }
    }
}

/// The restoration capture's request: the plan geometry is the OUTPUT, so the source is it over
/// the capture scale, taken at its own size. No seed, strength or step count — the restorer is one
/// deterministic forward and refuses each of them by name.
fn restoration_request(width: u32, height: u32) -> TransformRequest {
    TransformRequest {
        image: source_image(width / RESTORE_SCALE, height / RESTORE_SCALE),
        target: TargetSize::Scale(RESTORE_SCALE as f32),
        input_sizing: InputSizing::Original,
        ..Default::default()
    }
}

/// A depth map as an RGB image, so the shared determinism envelope judges it like any render: the
/// map min-max normalized to 8 bits, replicated over the three channels.
fn depth_image(width: u32, height: u32, values: &[f32]) -> Result<Image, String> {
    if values.len() != (width as usize) * (height as usize) {
        return Err(format!(
            "the depth map holds {} values for a {width}x{height} source",
            values.len()
        ));
    }
    let (low, high) = values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    // `f32::min`/`max` skip a NaN operand, so finiteness is asked of every value, not of the fold.
    if values.iter().any(|value| !value.is_finite()) {
        return Err("the depth map holds a non-finite value".to_owned());
    }
    let span = (high - low).max(f32::EPSILON);
    let pixels = values
        .iter()
        .flat_map(|&value| {
            let level = (((value - low) / span) * 255.0).round().clamp(0.0, 255.0) as u8;
            [level, level, level]
        })
        .collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}

/// The capture boundary one generation progress event crosses, if any, advancing `phase`.
///
/// The engine's two request shapes report conditioning's end differently: a staged request emits
/// `Progress::Loading(LoadPhase::Renderer)` once the text encoder is dropped and BEFORE the
/// backbone loads; a resident one has no such event and its first solver step is the boundary.
/// `protocol::next_reference_phase` is the one lifecycle every still arm converges on, so the first
/// of the two closes conditioning and the later one is ignored. Splitting a staged request at the
/// first solver step instead puts the backbone load and that step inside "conditioning", which is
/// what made all three phases of the first staged capture read the same peak (sc-25690).
fn generation_boundary(
    phase: &mut protocol::ReferencePhase,
    progress: &Progress,
) -> Option<Boundary> {
    let reference = match progress {
        Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer) => {
            protocol::ReferenceBoundary::RendererLoad
        }
        Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
        Progress::Decoding => protocol::ReferenceBoundary::Decoding,
        _ => return None,
    };
    let next = protocol::next_reference_phase(*phase, reference)?;
    *phase = next;
    match next {
        protocol::ReferencePhase::Denoise => Some(Boundary::ConditioningDone),
        protocol::ReferencePhase::Decode => Some(Boundary::ForwardDone),
        protocol::ReferencePhase::Conditioning => None,
    }
}

/// Run `planned` once on `loaded`, reporting each boundary as the request crosses it.
fn render(
    loaded: &Loaded,
    planned: &Planned,
    boundary: &mut dyn FnMut(Boundary),
) -> Result<Image, String> {
    match (loaded, planned) {
        (Loaded::Generation(generator), Planned::Generation(request)) => one_image(
            generator
                .generate(request, &mut {
                    let mut phase = protocol::ReferencePhase::Conditioning;
                    move |progress| {
                        if let Some(crossed) = generation_boundary(&mut phase, &progress) {
                            boundary(crossed);
                        }
                    }
                })
                .map_err(|error| format!("generate measured iris_3b render: {error}"))?,
        ),
        (Loaded::Depth(model), Planned::Depth(request)) => {
            // Nothing is computed before the forward: the shipped empty-prompt states are the whole
            // conditioning and they are resident from the load.
            boundary(Boundary::ConditioningDone);
            let output = model
                .estimate(request, &mut |progress| {
                    if matches!(progress, Progress::Step { .. }) {
                        boundary(Boundary::ForwardDone);
                    }
                })
                .map_err(|error| format!("estimate measured iris_3b_depth map: {error}"))?;
            depth_image(output.map.width, output.map.height, &output.map.values)
        }
        (Loaded::Restore(model), Planned::Restore(request)) => {
            boundary(Boundary::ConditioningDone);
            model
                .apply(request, &mut |progress| {
                    // `Decoding` marks the post-processing after the LAST tile's forward.
                    if progress == Progress::Decoding {
                        boundary(Boundary::ForwardDone);
                    }
                })
                .map_err(|error| format!("apply measured iris_3b_restore pass: {error}"))
        }
        _ => Err("an Iris request was planned for another route's model".to_owned()),
    }
}

/// A phase reading that can never be zero while a model is resident: the peak over the window, or
/// the live active figure when nothing was allocated inside it (a downstream route's conditioning
/// window is empty by construction).
fn phase_now() -> PhaseMemory {
    PhaseMemory {
        active: (get_peak_memory() as u64).max(get_active_memory() as u64),
        cache: get_cache_memory() as u64,
    }
}

fn context(
    route: protocol::IrisRoute,
    selection: &MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: MemoryGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: *selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: match route.mode {
            "text_to_image" => MemoryMode::TextToImage,
            other => MemoryMode::Other(other.to_owned()),
        },
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry,
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-25690@{}", protocol::INFERENCE_PIN),
    }
}

/// The `mlx:iris_3b` / `mlx:iris_3b_depth` / `mlx:iris_3b_restore` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    let provider = protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let route = route(provider)?;
    // Mode, tier, still geometry, overlay and load shape — all before an env var or a weight.
    let (width, height) = protocol::validate_iris_plan(request, route)?;
    let selection = planned_selection(request)?;
    let planned_fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let expected_fixture = fixture(route, width, height);
    if planned_fixture != expected_fixture {
        return Err(format!(
            "planned.fixture {planned_fixture:?} is not the {} capture at {width}x{height}: \
             expected {expected_fixture:?}",
            route.provider
        ));
    }
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    let expected_fingerprint = calibration_fingerprint(route);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} bf16 \
             production identity is {expected_fingerprint}",
            route.provider
        ));
    }
    let (repository, revision, root) = protocol::iris_weights_root(route)?;
    let text_encoder = if route.mode == "text_to_image" {
        let staged = protocol::required_env(protocol::IRIS_3B_TEXT_ENCODER_COMPONENT_ENV)?;
        Some(std::fs::canonicalize(&staged).map_err(|error| {
            format!(
                "canonicalize {}: {error}",
                protocol::IRIS_3B_TEXT_ENCODER_COMPONENT_ENV
            )
        })?)
    } else {
        None
    };
    let spec = load_spec(route, root, text_encoder)?;

    let loaded = Loaded::load(route, &spec)?;
    let contract = loaded.contract()?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            route.provider
        )
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity",
            route.provider,
            protocol::INFERENCE_PIN
        )
    })?;
    if calibration.fingerprint != planned_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if calibration.load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            protocol::LOAD_SHAPE_EAGER,
            load_shape_key(calibration.load_shape)
        ));
    }
    let (planned, geometry) = plan_request(&loaded, width, height)?;
    if (geometry.width, geometry.height) != (width, height) {
        return Err(format!(
            "the {} engine admits this request at {}x{}, not the planned {width}x{height}; the \
             record would price another geometry",
            route.provider, geometry.width, geometry.height
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        loaded.safety(&context(
            route,
            &selection,
            calibration,
            fingerprint,
            geometry,
            total_bytes,
            predicted,
        ))
    };
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the two rejections
    // below cannot pass through a blanket refusal.
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected a fitting probe budget; the scenario rejections below would be \
             a blanket refusal, not evidence",
            route.provider
        ));
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted an unknown/zero memory budget",
            route.provider
        ));
    }
    if !matches!(
        safety("stale-iris-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            route.provider
        ));
    }

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let selected = render(&loaded, &planned, &mut |boundary| {
        match boundary {
            Boundary::ConditioningDone => conditioning.set(phase_now()),
            Boundary::ForwardDone => denoise.set(phase_now()),
        }
        reset_peak_memory();
    })?;
    let decode = phase_now();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak (a request boundary was \
             never crossed)",
            route.provider
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    if !matches!(
        safety(&calibration.fingerprint, predicted, predicted),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected an exact-fit calibrated budget",
            route.provider
        ));
    }

    // Warm-repeat determinism and allocator cleanup bounds on this exact loaded model.
    clear_cache();
    reset_peak_memory();
    let baseline = render(&loaded, &planned, &mut |_| {})?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let cleanup_bounds =
        LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if !flux2_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "{} warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
             mean={mean_error:.6}, rms={rms_error:.6}",
            route.provider
        ));
    }
    reset_peak_memory();
    let warm = render(&loaded, &planned, &mut |_| {})?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "{} warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%",
            route.provider
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "{} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the clean warm \
             cleanup {clean_post_cleanup:?} plus {} bytes",
            route.provider, cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            route.provider
        ));
    }
    // Falsifiability: a runtime_complete record keeps `negativeMutation` null, so the breach is
    // verified here and the numbers land in diagnostics.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            route.provider
        ));
    }

    let lifecycle_blocker = concat!(
        "the pinned Iris-3B MLX crate exposes no calibration fault-injection site (generation ",
        "opens a memory-strategy request scope; the depth and restoration requests carry no ",
        "memory block and open none), so the scoped lifecycle scenarios cannot execute; unscoped ",
        "repeat determinism and allocator cleanup bounds are attested in quality and diagnostics ",
        "instead"
    );
    let execution_path = format!("the {} resident path", route.provider);
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "bf16",
        },
        "sweep": sweep,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "identical artifact, request, geometry, tier, and loaded model; cold measured run versus warm unscoped repeats",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": FLUX2_MAX_THRESHOLD,
            "meanErrorThreshold": FLUX2_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": FLUX2_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": match route.subdir {
                Some(subdir) => format!("{repository}@{revision}:{subdir}"),
                None => format!("{repository}@{revision}:bf16"),
            },
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-resident", route.slug),
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("admittedGeometryWidth", "count", u64::from(geometry.width)),
                ("admittedGeometryHeight", "count", u64::from(geometry.height)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, &execution_path)?;
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A staged generation request closes conditioning when the backbone is about to load, not at
    /// the first solver step (which then changes nothing); a resident one, which reports no load
    /// phase, closes it at that step. Both end the forward at `Decoding`.
    ///
    /// *Mutation that reds this:* drop the `Progress::Loading(LoadPhase::Renderer)` arm from
    /// `generation_boundary` (the staged request then splits at the first solver step).
    #[test]
    fn a_staged_request_ends_conditioning_at_the_renderer_load() {
        let boundaries = |events: &[Progress]| {
            let mut phase = protocol::ReferencePhase::Conditioning;
            events
                .iter()
                .enumerate()
                .filter_map(|(index, event)| {
                    generation_boundary(&mut phase, event).map(|boundary| (index, boundary))
                })
                .collect::<Vec<_>>()
        };
        let step = |current| Progress::Step { current, total: 2 };
        let staged = [
            Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder),
            Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer),
            step(1),
            step(2),
            Progress::Decoding,
        ];
        assert_eq!(
            boundaries(&staged),
            [(1, Boundary::ConditioningDone), (4, Boundary::ForwardDone)]
        );
        let resident = [step(1), step(2), Progress::Decoding];
        assert_eq!(
            boundaries(&resident),
            [(0, Boundary::ConditioningDone), (2, Boundary::ForwardDone)]
        );
    }

    /// The restoration capture's request is one the restorer's own request resolution accepts,
    /// at the scale the capture names — so the anchor cannot die in planning on a control the
    /// route does not have (sc-25690: a `seed` did exactly that on the first real capture).
    ///
    /// *Mutation that reds this:* `seed: Some(SEED)` back on `restoration_request`.
    #[test]
    fn the_restoration_capture_request_is_one_the_restorer_accepts() {
        let request = restoration_request(2048, 2048);
        let options = gen_core_iris::restoration::RestorationOptions::from_request(
            &request,
            "iris_3b_restore",
        )
        .expect("the restorer resolves the capture request");
        assert_eq!(options.scale, f64::from(RESTORE_SCALE));
        assert_eq!((request.image.width, request.image.height), (512, 512));
    }

    fn planned(provider: &str, mode: &str, tier: &str, width: u32, height: u32) -> Value {
        json!({
            "planned": {
                "backend": "mlx",
                "loadShape": protocol::LOAD_SHAPE_EAGER,
                "target": {
                    "provider": provider,
                    "mode": mode,
                    "tier": tier,
                    "overlay": "none",
                    "geometry": { "width": width, "height": height, "batch": 1, "frames": 1 },
                },
            },
        })
    }

    /// Every Iris plan row names exactly the fixture and the engine identity its arm demands, so a
    /// row cannot plan a capture the arm then refuses by name.
    ///
    /// *Mutation that reds this:* change a fingerprint or a fixture in
    /// `config/memory-calibration-plan.json`, or `fixture()` here.
    #[test]
    fn every_planned_iris_row_names_its_arm_fixture_and_engine_identity() {
        let plan: Value = serde_json::from_str(include_str!(
            "../../../../config/memory-calibration-plan.json"
        ))
        .expect("the anchor plan parses");
        let mut seen = Vec::new();
        for (key, entry) in plan["anchors"].as_object().expect("anchors object") {
            let provider = entry["provider"].as_str().unwrap();
            let Ok(route) = route(provider) else {
                continue;
            };
            if !key.ends_with(":mlx") {
                continue;
            }
            assert_eq!(
                key,
                &format!("{provider}:bf16:mlx"),
                "bf16 is the only tier"
            );
            let geometry = &entry["geometry"];
            let (width, height) = (
                geometry["width"].as_u64().unwrap() as u32,
                geometry["height"].as_u64().unwrap() as u32,
            );
            assert_eq!(entry["mode"], route.mode, "{key}");
            assert_eq!(entry["fixture"], fixture(route, width, height), "{key}");
            assert_eq!(
                entry["calibrationFingerprint"],
                calibration_fingerprint(route),
                "{key}"
            );
            assert_eq!(entry["loadShape"], protocol::LOAD_SHAPE_EAGER, "{key}");
            assert_eq!(
                protocol::validate_iris_plan(
                    &planned(provider, route.mode, "bf16", width, height),
                    route
                ),
                Ok((width, height)),
                "{key}"
            );
            seen.push(route.provider);
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            [IRIS_PROVIDER, IRIS_DEPTH_PROVIDER, IRIS_RESTORE_PROVIDER],
            "each route is planned once on this lane"
        );
    }

    /// A row that is not the route's own mode, tier, single still image or eager shape is refused
    /// before any environment variable is read — the batch/frames validation every image arm keeps.
    ///
    /// *Mutation that reds this:* drop `validate_still_geometry` (or the tier / mode / load-shape
    /// check) from `protocol::validate_iris_plan`.
    #[test]
    fn a_plan_row_outside_a_routes_envelope_is_refused_before_any_weight_work() {
        for route in [GENERATION, DEPTH, RESTORE] {
            let good = planned(route.provider, route.mode, "bf16", 1024, 1024);
            assert_eq!(protocol::validate_iris_plan(&good, route), Ok((1024, 1024)));
            for (pointer, value, needle) in [
                (
                    "/planned/target/geometry/batch",
                    json!(2),
                    "geometry.batch == 1",
                ),
                (
                    "/planned/target/geometry/frames",
                    json!(2),
                    "geometry.frames == 1",
                ),
                (
                    "/planned/target/tier",
                    json!("q4"),
                    "one dense bf16 artifact",
                ),
                (
                    "/planned/target/mode",
                    json!("edit_image"),
                    "is measured in mode",
                ),
                (
                    "/planned/loadShape",
                    json!(protocol::LOAD_SHAPE_DEFERRED),
                    "loads eagerly only",
                ),
                ("/planned/target/overlay", json!("lora"), "overlay"),
            ] {
                let mut bad = good.clone();
                *bad.pointer_mut(pointer).unwrap() = value;
                let error = protocol::validate_iris_plan(&bad, route)
                    .expect_err("an out-of-envelope row is refused");
                assert!(
                    error.contains(needle),
                    "{} {pointer}: {error}",
                    route.provider
                );
            }
        }
    }

    /// Each route binds its OWN folder of the one upstream snapshot: generation the snapshot root,
    /// depth `depth/`, restoration `upscaler/`. Another route's folder, a tier-shaped folder and a
    /// foreign repository are all refused, so one task's weights can never be measured as another's.
    ///
    /// *Mutation that reds this:* swap two routes' `subdir`, or validate every route against the
    /// snapshot root.
    #[test]
    fn each_route_binds_only_its_own_folder_of_the_upstream_snapshot() {
        let revision = "c".repeat(40);
        let scratch = std::env::temp_dir().join(format!(
            "sceneworks-mlx-iris-arm-{}-{revision}",
            std::process::id()
        ));
        let snapshot = scratch
            .join("models--speridlabs--iris-3b")
            .join("snapshots")
            .join(&revision);
        for folder in ["depth", "upscaler", "bf16"] {
            std::fs::create_dir_all(snapshot.join(folder)).unwrap();
        }
        let root_of = |route: protocol::IrisRoute| match route.subdir {
            Some(subdir) => snapshot.join(subdir),
            None => snapshot.clone(),
        };
        for route in [GENERATION, DEPTH, RESTORE] {
            let bound = protocol::iris_weights_root_at(
                route,
                protocol::IRIS_3B_REPOSITORY.to_owned(),
                revision.clone(),
                root_of(route),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", route.provider));
            assert_eq!(bound.2, std::fs::canonicalize(root_of(route)).unwrap());
            for other in [GENERATION, DEPTH, RESTORE] {
                if other == route {
                    continue;
                }
                protocol::iris_weights_root_at(
                    route,
                    protocol::IRIS_3B_REPOSITORY.to_owned(),
                    revision.clone(),
                    root_of(other),
                )
                .expect_err("another route's folder is not this route's weights");
            }
            protocol::iris_weights_root_at(
                route,
                protocol::IRIS_3B_REPOSITORY.to_owned(),
                revision.clone(),
                snapshot.join("bf16"),
            )
            .expect_err("no Iris root carries a tier sub-directory");
            let error = protocol::iris_weights_root_at(
                route,
                "SceneWorks/iris-3b-mlx".to_owned(),
                revision.clone(),
                root_of(route),
            )
            .expect_err("a foreign repository is refused");
            assert!(error.contains(protocol::IRIS_3B_REPOSITORY), "{error}");
        }
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// Generation stages the text encoder as its one component; the downstream routes refuse one,
    /// exactly as their engines do. Every spec is resident and eagerly materialized.
    ///
    /// *Mutation that reds this:* attach the component on every route, or build a deferred spec.
    #[test]
    fn only_generation_stages_a_text_encoder_and_every_spec_is_resident_and_eager() {
        let generation = load_spec(GENERATION, "iris".into(), Some("te".into())).unwrap();
        assert_eq!(
            generation.components.get(TEXT_ENCODER_COMPONENT),
            Some(&WeightsSource::Dir("te".into()))
        );
        assert_eq!(generation.components.len(), 1);
        load_spec(GENERATION, "iris".into(), None).expect_err("generation needs its encoder");
        for route in [DEPTH, RESTORE] {
            let spec = load_spec(route, "export".into(), None).unwrap();
            assert!(spec.components.is_empty(), "{}", route.provider);
            load_spec(route, "export".into(), Some("te".into()))
                .expect_err("a downstream route refuses a staged text encoder");
        }
        for (route, encoder) in [
            (GENERATION, Some(PathBuf::from("te"))),
            (DEPTH, None),
            (RESTORE, None),
        ] {
            let spec = load_spec(route, "root".into(), encoder).unwrap();
            assert_eq!(spec.offload_policy, OffloadPolicy::Resident);
            assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(spec.quantize, None);
        }
    }

    /// A depth map is judged by the same determinism envelope as a render, so its image form must
    /// span the full 8-bit range and refuse a malformed or non-finite map.
    #[test]
    fn a_depth_map_becomes_a_full_range_image() {
        let image = depth_image(2, 2, &[0.0, 1.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            image.pixels,
            [0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255]
        );
        depth_image(2, 2, &[0.0; 3]).expect_err("a short map is refused");
        depth_image(1, 2, &[0.0, f32::NAN]).expect_err("a non-finite map is refused");
    }
}
