//! Physical Candle/CUDA capture arm for the three Iris-3B task routes (sc-25690, epic sc-25678).
//!
//! The Candle twin of `mlx_iris.rs`: generation (`iris_3b`), monocular depth (`iris_3b_depth`) and
//! restoration (`iris_3b_restore`), one arm, three drives.
//!
//! * **generation** is a registry [`Generator`] and takes the lane's default anchor composition —
//!   staged residency — exactly the way the worker runs it: the staged (`Sequential`) load, the
//!   provider's own safety check, its request scope, then the render. Conditioning ends when the
//!   staged request reports the backbone load (`Progress::Loading(LoadPhase::Renderer)`: the text
//!   encoder is gone and the backbone is not yet on the card), denoise at `Progress::Decoding`.
//! * **depth** is NOT a generator: `candle-gen-iris`'s `IrisDepth`, one backbone forward on the
//!   export's shipped empty-prompt states, driven for ONE estimate at the plan geometry.
//! * **restoration** is a `Transform`: `IrisRestorer`, driven for ONE tiled pass at the plan
//!   geometry.
//!
//! The downstream contracts declare `Resident` alone (staged residency is structurally not
//! applicable: no text encoder is loaded), which is why their plan rows override the lane default
//! and why this arm refuses any other rung for them by name. Neither request type carries a memory
//! block, so they open no request scope; admission is the loaded model's own safety check.
//!
//! Every phase figure is the `nvidia-smi` device-memory peak over that phase's window above the
//! probe's idle baseline — the one residency counter this lane has. For a downstream route the
//! conditioning window is empty by construction (nothing is computed before the forward), so it
//! reads the resident model and its empty-prompt states.

use super::*;
use runtime_cuda::gen_core::iris::depth::{
    plan_depth_input, DepthRequest, DepthResolution, IrisDepthEstimator,
};
use runtime_cuda::gen_core::iris::TEXT_ENCODER_COMPONENT;
use runtime_cuda::gen_core::{
    InputSizing, MemoryCalibrationIdentity, MemoryProviderContract, TargetSize, Transform,
    TransformRequest,
};
use runtime_cuda::providers::iris as engine;

/// The label the depth and restoration fixtures carry in their name. Neither request takes a
/// seed: both routes are deterministic, and the restorer refuses one by name. Generation renders
/// the five-rung reference seed and step count every Candle still anchor uses.
const SEED: u64 = 25_690;
const GENERATION_SEED: u64 = 16_402;
const GENERATION_STEPS: u32 = 2;
/// Restoration upscales by the release's default factor, so the source is the plan geometry / 4.
const RESTORE_SCALE: u32 = 4;

/// `candle-gen-iris`'s three production calibration identities at the pin, as literals for the
/// reason every identity table in this binary is one (it is `compile_error!` on macOS, where the
/// plan rows are authored). `the_iris_candle_identities_are_the_engine_constants` binds them on a
/// CUDA host, and the capture still refuses a loaded contract whose identity differs.
const GENERATION_FINGERPRINT: &str = "iris-3b-candle-derived-2026-10-12-v1";
const DEPTH_FINGERPRINT: &str = "iris-3b-depth-candle-derived-2026-10-12-v1";
const RESTORE_FINGERPRINT: &str = "iris-3b-restore-candle-derived-2026-10-12-v1";

/// Generation: the backbone at the snapshot root, plus the staged text encoder.
const GENERATION: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_ID,
    mode: "text_to_image",
    repository_env: "SCENEWORKS_IRIS_3B_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_ROOT",
    subdir: None,
    slug: "iris-3b",
};
/// Monocular depth: the `depth/` export.
const DEPTH: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_DEPTH_ID,
    mode: "depth",
    repository_env: "SCENEWORKS_IRIS_3B_DEPTH_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_DEPTH_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_DEPTH_ROOT",
    subdir: Some(protocol::IRIS_3B_DEPTH_SUBDIR),
    slug: "iris-3b-depth",
};
/// Restoration: the `upscaler/` export.
const RESTORE: protocol::IrisRoute = protocol::IrisRoute {
    provider: IRIS_RESTORE_ID,
    mode: "restoration",
    repository_env: "SCENEWORKS_IRIS_3B_RESTORE_REPOSITORY",
    revision_env: "SCENEWORKS_IRIS_3B_RESTORE_REVISION",
    root_env: "SCENEWORKS_IRIS_3B_RESTORE_ROOT",
    subdir: Some(protocol::IRIS_3B_RESTORE_SUBDIR),
    slug: "iris-3b-restore",
};

/// Whether this module serves `provider` — held to the pre-gate in `run` by
/// `the_iris_pre_gate_names_exactly_the_providers_the_module_implements`.
pub(super) fn implements(provider: &str) -> bool {
    route(provider).is_ok()
}

/// The route a plan row's provider names.
pub(super) fn route(provider: &str) -> Result<protocol::IrisRoute, String> {
    [GENERATION, DEPTH, RESTORE]
        .into_iter()
        .find(|route| route.provider == provider)
        .ok_or_else(|| format!("the Candle Iris arm does not implement provider {provider:?}"))
}

/// The production calibration identity the loaded route publishes.
pub(super) fn calibration_fingerprint(route: protocol::IrisRoute) -> &'static str {
    match route.mode {
        "text_to_image" => GENERATION_FINGERPRINT,
        "depth" => DEPTH_FINGERPRINT,
        _ => RESTORE_FINGERPRINT,
    }
}

/// The fixture a plan row must name for `route` at one geometry.
pub(super) fn fixture(route: protocol::IrisRoute, width: u32, height: u32) -> String {
    let (seed, steps) = match route.mode {
        "text_to_image" => (GENERATION_SEED, GENERATION_STEPS),
        _ => (SEED, 0),
    };
    protocol::iris_fixture(route, "candle", width, height, seed, steps)
}

/// The rung a route's anchor measures: the lane default (staged residency) for generation, and
/// `Resident` for the downstream routes, whose contracts declare staged residency structurally not
/// applicable.
pub(super) fn anchor_strategy(route: protocol::IrisRoute) -> MemoryStrategy {
    match route.mode {
        "text_to_image" => MemoryStrategy::StagedResidency,
        _ => MemoryStrategy::Resident,
    }
}

/// The load spec production builds for `route` over a validated weights root, eagerly
/// materialized: the staged (`Sequential`) load with the text-encoder component for generation —
/// the load the worker issues when the ladder selects staged residency — and a plain resident load
/// for the downstream routes, which refuse a staged text encoder.
pub(super) fn load_spec(
    route: protocol::IrisRoute,
    root: PathBuf,
    text_encoder: Option<PathBuf>,
) -> Result<LoadSpec, String> {
    let spec =
        LoadSpec::new(WeightsSource::Dir(root)).with_load_shape(LoadShape::EagerMaterialization);
    match (route.mode, text_encoder) {
        ("text_to_image", Some(dir)) => Ok(spec
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(dir))),
        ("text_to_image", None) => Err(format!(
            "{} loads its Qwen3-VL text encoder as a staged component; {} is required",
            route.provider,
            protocol::IRIS_3B_TEXT_ENCODER_COMPONENT_ENV
        )),
        (_, Some(_)) => Err(format!(
            "{} runs on its export's shipped empty-prompt states and refuses a staged text encoder",
            route.provider
        )),
        (_, None) => Ok(spec.with_offload_policy(OffloadPolicy::Resident)),
    }
}

/// The two request boundaries a capture is split at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    ConditioningDone,
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
            "text_to_image" => runtime_cuda::catalog()
                .map_err(|error| format!("build CUDA catalog: {error}"))?
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

/// One planned request of a loaded route.
enum Planned {
    Generation(Box<GenerationRequest>),
    Depth(DepthRequest),
    Restore(TransformRequest),
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

fn context(
    route: protocol::IrisRoute,
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: MemoryGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
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
        has_phases: route.mode == "text_to_image",
        geometry,
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
        Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
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

/// Run `planned` once on `loaded`, reporting each boundary as the request crosses it. Generation
/// renders under the request scope the provider opens for `scoped`, the way the worker runs it.
fn render(
    loaded: &Loaded,
    planned: &Planned,
    scoped: &MemoryRunContext,
    boundary: &mut dyn FnMut(Boundary),
) -> Result<(), String> {
    match (loaded, planned) {
        (Loaded::Generation(generator), Planned::Generation(request)) => {
            let mut scope = open_memory_request_scope(generator.as_ref(), scoped, "iris_3b")?;
            let mut request = request.as_ref().clone();
            scope
                .configure_request(&mut request)
                .map_err(|error| format!("configure the iris_3b request scope: {error}"))?;
            let mut phase = protocol::ReferencePhase::Conditioning;
            let outcome = generator.generate(&request, &mut |progress| {
                if let Some(crossed) = generation_boundary(&mut phase, &progress) {
                    boundary(crossed);
                }
            });
            match outcome {
                Ok(_) => scope
                    .finish(MemoryRunOutcome::Complete)
                    .map_err(|error| format!("finish the iris_3b request scope: {error}")),
                Err(error) => {
                    let message = error.to_string();
                    let _ = scope.finish(MemoryRunOutcome::Error {
                        message: message.clone(),
                    });
                    Err(format!("iris_3b measured generation failed: {message}"))
                }
            }
        }
        (Loaded::Depth(model), Planned::Depth(request)) => {
            boundary(Boundary::ConditioningDone);
            model
                .estimate(request, &mut |progress| {
                    if matches!(progress, Progress::Step { .. }) {
                        boundary(Boundary::ForwardDone);
                    }
                })
                .map(|_| ())
                .map_err(|error| format!("iris_3b_depth measured estimate failed: {error}"))
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
                .map(|_| ())
                .map_err(|error| format!("iris_3b_restore measured pass failed: {error}"))
        }
        _ => Err("an Iris request was planned for another route's model".to_owned()),
    }
}

/// The `candle:iris_3b` / `candle:iris_3b_depth` / `candle:iris_3b_restore` arm.
pub(super) fn run(request: &Value) -> Result<Value, String> {
    let route = route(planned_provider(request)?)?;
    // Mode, tier, still geometry, overlay and load shape — all before an env var or a weight.
    let (width, height) = protocol::validate_iris_plan(request, route)?;
    let selection = planned_selection(request)?;
    if selection.strategy != anchor_strategy(route) {
        return Err(format!(
            "{} measures the {:?} anchor on this lane, not {:?}: generation takes the lane's \
             staged default, and the downstream contracts declare Resident alone (staged \
             residency is structurally not applicable without a text encoder)",
            route.provider,
            anchor_strategy(route),
            selection.strategy
        ));
    }
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

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let loaded = Loaded::load(route, &spec)?;
    vram.end_load(load_sample);

    let contract = loaded.contract()?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            route.provider
        )
    })?;
    let strategy = measured_strategy(
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
    if calibration.fingerprint != expected_fingerprint {
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
    let admitted = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        context(
            route,
            selection,
            calibration,
            fingerprint,
            geometry,
            total_bytes,
            predicted,
        )
    };
    // Admission mutation hygiene: the gate must ACCEPT a fitting request, so the two rejections
    // below cannot pass through a blanket refusal.
    if !matches!(
        loaded.safety(&admitted(&calibration.fingerprint, hardware_bytes, 1)),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected a fitting probe budget; the scenario rejections below would be \
             a blanket refusal, not evidence",
            route.provider
        ));
    }
    if !matches!(
        loaded.safety(&admitted(&calibration.fingerprint, 0, 1)),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted an unknown/zero memory budget",
            route.provider
        ));
    }
    if !matches!(
        loaded.safety(&admitted("stale-iris-fingerprint", hardware_bytes, 1)),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            route.provider
        ));
    }

    let scoped = admitted(&calibration.fingerprint, hardware_bytes, 1);
    let run_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let result = render(&loaded, &planned, &scoped, &mut |boundary| {
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match boundary {
            Boundary::ConditioningDone => conditioning_peak_gb = peak,
            Boundary::ForwardDone => denoise_peak_gb = peak,
        }
        phase_sample = Some(vram.phase());
    });
    let decode_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
    vram.end_gen(run_sample);
    result?;

    let phase_bytes = |peak: Option<f64>, phase: &str| {
        peak.map(decimal_gb_to_bytes)
            .ok_or_else(|| format!("{} did not expose a {phase} boundary", route.provider))
    };
    let conditioning_bytes = phase_bytes(conditioning_peak_gb, "conditioning")?;
    let denoise_bytes = phase_bytes(denoise_peak_gb, "forward")?;
    let decode_bytes = phase_bytes(decode_peak_gb, "finish")?;
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-25690 anchor capture measures exact per-phase device memory and the strategy identity ",
        "for the Candle Iris-3B lane; it stays gated because this run does not repeat the full ",
        "promotion-quality, negative-mutation and lifecycle scenario suite, and the pinned crate ",
        "exposes no calibration fault-injection site"
    );
    let execution_path = format!("the {} resident path", route.provider);
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": { "repository": repository, "resolvedRevision": revision, "variant": "bf16" },
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded Iris admission check rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded Iris admission check rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
        ],
        "predictedPeakBytes": null,
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "quality": { "result": "not_run" },
        "negativeMutation": Value::Null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": match route.subdir {
                Some(subdir) => format!("{repository}@{revision}:{subdir}"),
                None => format!("{repository}@{revision}:bf16"),
            },
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-candle-adapter:{}-anchor", route.slug),
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
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
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::TextEncoder),
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer),
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
        let options = runtime_cuda::gen_core::iris::restoration::RestorationOptions::from_request(
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
                "backend": "candle",
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

    /// The literal identities this lane binds its plan rows to ARE the pinned engine's constants —
    /// the binding the macOS host that authors the plan cannot compile.
    #[test]
    fn the_iris_candle_identities_are_the_engine_constants() {
        assert_eq!(
            GENERATION_FINGERPRINT,
            engine::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            DEPTH_FINGERPRINT,
            engine::memory_strategy::DEPTH_MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            RESTORE_FINGERPRINT,
            engine::memory_strategy::RESTORE_MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(DEPTH.mode, engine::memory_strategy::DEPTH_MEMORY_MODE);
        assert_eq!(RESTORE.mode, engine::memory_strategy::RESTORE_MEMORY_MODE);
    }

    /// The bespoke pre-gate in `run` names exactly the providers this module serves.
    #[test]
    fn the_iris_pre_gate_names_exactly_the_providers_the_module_implements() {
        for provider in [IRIS_ID, IRIS_DEPTH_ID, IRIS_RESTORE_ID] {
            assert!(implements(provider), "{provider}");
        }
        for provider in [KREA_ID, QWEN_ID, "iris_3b_unknown"] {
            assert!(!implements(provider), "{provider}");
        }
    }

    /// Every Iris plan row names exactly the fixture, identity and rung its arm demands.
    #[test]
    fn every_planned_iris_row_names_its_arm_fixture_identity_and_rung() {
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
            if !key.ends_with(":candle") {
                continue;
            }
            assert_eq!(
                key,
                &format!("{provider}:bf16:candle"),
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
            // The row overrides the lane's staged default exactly where the contract refuses it.
            match anchor_strategy(route) {
                MemoryStrategy::Resident => {
                    assert_eq!(entry["strategy"]["rung"], "resident", "{key}");
                    assert_eq!(
                        entry["strategy"]["engagedRungs"],
                        json!(["resident"]),
                        "{key}"
                    );
                }
                _ => assert!(entry.get("strategy").is_none(), "{key}"),
            }
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
            [IRIS_ID, IRIS_DEPTH_ID, IRIS_RESTORE_ID],
            "each route is planned once on this lane"
        );
    }

    /// Generation takes the staged (`Sequential`) load with its one text-encoder component; the
    /// downstream routes load resident and refuse a staged encoder. Every spec is eager.
    #[test]
    fn only_generation_stages_a_text_encoder_and_every_spec_is_eager() {
        let generation = load_spec(GENERATION, "iris".into(), Some("te".into())).unwrap();
        assert_eq!(generation.offload_policy, OffloadPolicy::Sequential);
        assert_eq!(
            generation.components.get(TEXT_ENCODER_COMPONENT),
            Some(&WeightsSource::Dir("te".into()))
        );
        load_spec(GENERATION, "iris".into(), None).expect_err("generation needs its encoder");
        for route in [DEPTH, RESTORE] {
            let spec = load_spec(route, "export".into(), None).unwrap();
            assert_eq!(
                spec.offload_policy,
                OffloadPolicy::Resident,
                "{}",
                route.provider
            );
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
            assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(spec.quantize, None);
        }
    }

    /// Each route measures the rung its engine's contract actually runs as this lane's anchor,
    /// checked against the weights-free contract the registry publishes.
    #[test]
    fn each_route_anchors_on_a_rung_its_contract_implements() {
        let catalog = runtime_cuda::catalog().expect("CUDA catalog");
        let registry = catalog.media();
        for route in [GENERATION, DEPTH, RESTORE] {
            let fixture = registry
                .memory_contract_fixture_registrations()
                .find(|fixture| fixture.provider_id == route.provider)
                .unwrap_or_else(|| panic!("{} publishes a weights-free contract", route.provider));
            let contract = (fixture.contract)(&LoadSpec::new(WeightsSource::Dir("iris".into())))
                .expect("weights-free contract");
            let staged_implemented = matches!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .map(|capability| &capability.support),
                Some(runtime_cuda::gen_core::MemoryStrategySupport::Implemented)
            );
            assert_eq!(
                anchor_strategy(route) == MemoryStrategy::StagedResidency,
                staged_implemented,
                "{}",
                route.provider
            );
        }
    }
}
