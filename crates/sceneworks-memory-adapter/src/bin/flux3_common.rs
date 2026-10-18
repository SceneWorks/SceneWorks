//! Tensor-neutral FLUX3 producer binding and scoped lifecycle, shared by both lanes.
//! Nothing in this module opens a device. Only the lane's `run` loads a registry generator.
use super::*;
#[cfg(not(target_os = "macos"))]
use runtime_cuda::gen_core as core;
#[cfg(target_os = "macos")]
use runtime_macos::gen_core as core;

pub(super) const PROVIDER: &str = "flux3_klein";
pub(super) const FLUX3_REPOSITORY: &str = "black-forest-labs/flux-3-image-klein";
pub(super) const REVISION: &str = "8454017f9ebaada30c998fccd9f67a80c7580f68";
pub(super) const SEED: u64 = 26_243;
pub(super) const STEPS: u32 = 16;

pub(super) fn quant(lane: &str, tier: &str) -> Result<Option<Quant>, String> {
    match (lane, tier) {
        ("candle" | "mlx", "bf16") => Ok(None),
        ("candle" | "mlx", "q4") => Ok(Some(Quant::Q4)),
        ("candle" | "mlx", "q8") => Ok(Some(Quant::Q8)),
        ("candle", "fp8r") => Ok(Some(Quant::Fp8Rowwise)),
        ("candle", "nvfp4") => Ok(Some(Quant::Nvfp4)),
        _ => Err(format!(
            "unsupported FLUX3 native tier {lane}:{tier}; no dense fallback"
        )),
    }
}

pub(super) fn conversion(lane: &str, tier: &str) -> Option<&'static str> {
    match (lane, tier) {
        ("candle", "q4") => Some("flux3-candle-ggml-q4_1-v1"),
        ("candle", "q8") => Some("flux3-candle-ggml-q8_0-v1"),
        ("mlx", "q4") => Some("flux3-mlx-affine-q4-g64-v1"),
        ("mlx", "q8") => Some("flux3-mlx-affine-q8-g64-v1"),
        _ => None,
    }
}

pub(super) fn fingerprint(lane: &str, tier: &str) -> Result<String, String> {
    quant(lane, tier)?;
    let suffix = match (lane, tier) {
        (_, "bf16") => "",
        ("candle", "q4") => "-ggml-q4-1",
        ("candle", "q8") => "-ggml-q8-0",
        ("mlx", "q4") => "-mlx-affine-q4-g64",
        ("mlx", "q8") => "-mlx-affine-q8-g64",
        (_, "fp8r") => "-fp8r",
        (_, "nvfp4") => "-nvfp4",
        _ => unreachable!("quant validated the tier"),
    };
    Ok(format!("flux3-klein-{lane}-derived-2026-10-15-v1{suffix}"))
}

pub(super) struct Plan {
    pub tier: String,
    pub fingerprint: String,
    pub selection: MemorySelection,
}

pub(super) fn plan(request: &Value, lane: &str) -> Result<Plan, String> {
    let planned = protocol::planned(request)?;
    protocol::validate_still_geometry(request, "FLUX3 calibration")?;
    protocol::validate_plain_overlay_target(request, "the native FLUX3 text-to-image path")?;
    for (pointer, expected) in [
        ("/backend", lane),
        ("/target/provider", PROVIDER),
        ("/target/modelId", PROVIDER),
        ("/target/mode", "text_to_image"),
        ("/loadShape", protocol::LOAD_SHAPE_EAGER),
    ] {
        if planned.pointer(pointer).and_then(Value::as_str) != Some(expected) {
            return Err(format!("FLUX3 planned{pointer} must be {expected:?}"));
        }
    }
    if protocol::target_geometry(request)? != (1024, 1024) {
        return Err(
            "FLUX3 producer measures the 1024x1024 seed26243 step16 fixture only".to_owned(),
        );
    }
    let tier = planned
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "FLUX3 planned.target.tier must be a string".to_owned())?;
    let fingerprint = fingerprint(lane, tier)?;
    let fixture = format!("flux3-klein-{lane}-{tier}-1024-seed26243-step16");
    if planned.get("fixture").and_then(Value::as_str) != Some(fixture.as_str()) {
        return Err(format!("FLUX3 planned.fixture must be {fixture:?}"));
    }
    if planned
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        != Some(fingerprint.as_str())
    {
        return Err(format!(
            "FLUX3 plan/provider calibration mismatch: expected {fingerprint}"
        ));
    }
    if !protocol::strategy_parameters(request)?.is_empty() {
        return Err("FLUX3 anchor producer accepts no unmeasured strategy parameters".to_owned());
    }
    let selection = MemorySelection {
        strategy: planned_memory_strategy(request)?,
        parameters: MemoryStrategyParameters::default(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: quant(lane, tier)?,
            component_precision_floors: &[],
        },
    };
    let anchor = if lane == "candle" {
        MemoryStrategy::StagedResidency
    } else {
        MemoryStrategy::Resident
    };
    if selection.strategy != anchor {
        return Err(format!(
            "FLUX3 {lane} producer measures {anchor:?}, not {:?}",
            selection.strategy
        ));
    }
    Ok(Plan {
        tier: tier.to_owned(),
        fingerprint,
        selection,
    })
}

/// The harness passes the two actual roots, not a guessed snapshot/<tier>. Packed
/// outputs own the DiT only; both external components ALWAYS come from the original.
pub(super) fn spec(lane: &str, plan: &Plan) -> Result<LoadSpec, String> {
    let repo = protocol::required_env("SCENEWORKS_FLUX3_KLEIN_REPOSITORY")?;
    let rev = protocol::required_env("SCENEWORKS_FLUX3_KLEIN_REVISION")?;
    if repo != FLUX3_REPOSITORY || rev != REVISION {
        return Err(format!(
            "FLUX3 requires pinned original {FLUX3_REPOSITORY}@{REVISION}, got {repo}@{rev}"
        ));
    }
    let root = |name| -> Result<PathBuf, String> {
        let raw = protocol::required_env(name).map_err(|error| {
            format!("weights_missing: {error}; run the local-install harness probe")
        })?;
        std::fs::canonicalize(&raw).map_err(|error| {
            format!("weights_missing: {name}={raw}: {error}; provision this exact native package")
        })
    };
    let original = root("SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT")?;
    protocol::validate_huggingface_revision_root(&original, &repo, &rev, FLUX3_REPOSITORY)?;
    let package = root("SCENEWORKS_FLUX3_KLEIN_ROOT")?;
    validate_package_root(lane, &plan.tier, &original, &package)?;
    for file in [
        package.join("config.json"),
        package.join("model.safetensors"),
        original.join("ae.safetensors"),
        original.join("text_encoder/config.json"),
        original.join("text_encoder/tokenizer.json"),
        original.join("text_encoder/tokenizer_config.json"),
    ] {
        if !file.is_file() {
            return Err(format!("weights_missing: FLUX3 requires {}; install this exact package and its original external TE/AE", file.display()));
        }
    }
    if conversion(lane, &plan.tier).is_some()
        && !package.join("flux3-conversion-receipt.json").is_file()
    {
        return Err(format!(
            "weights_missing: {} has no native conversion receipt; derive this exact backend tier",
            package.display()
        ));
    }
    if !original.join("text_encoder/model.safetensors").is_file()
        && !original
            .join("text_encoder/model.safetensors.index.json")
            .is_file()
    {
        return Err(format!(
            "weights_missing: {} requires its text encoder single-file weights or shard index",
            original.display()
        ));
    }
    load_spec(lane, &plan.tier, package, original)
}

pub(super) fn validate_package_root(
    lane: &str,
    tier: &str,
    original: &std::path::Path,
    package: &std::path::Path,
) -> Result<(), String> {
    quant(lane, tier)?;
    let valid = if let Some(id) = conversion(lane, tier) {
        package.ends_with(std::path::Path::new(tier).join(id)) && package != original
    } else if tier == "bf16" {
        package == original
    } else {
        package == original.join("variants").join(tier)
    };
    if !valid {
        return Err(format!("FLUX3 {lane}:{tier} package {} is not the exact native install (conversion {:?}); no snapshot tier-dir or dense substitution", package.display(), conversion(lane, tier)));
    }
    Ok(())
}

pub(super) fn load_spec(
    lane: &str,
    tier: &str,
    package: PathBuf,
    original: PathBuf,
) -> Result<LoadSpec, String> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(package))
        .with_offload_policy(if lane == "candle" {
            OffloadPolicy::Sequential
        } else {
            OffloadPolicy::Resident
        })
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_component(
            "text_encoder",
            WeightsSource::Dir(original.join("text_encoder")),
        )
        .with_component(
            "autoencoder",
            WeightsSource::File(original.join("ae.safetensors")),
        );
    spec.precision = Precision::Bf16;
    spec.quantize = quant(lane, tier)?;
    Ok(spec)
}

pub(super) fn context(plan: &Plan, abi: u32, hardware_bytes: u64) -> MemoryRunContext {
    MemoryRunContext {
        selection: plan.selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: abi,
        calibration_fingerprint: plan.fingerprint.clone(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        reference_resources: None,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        // This is a capture request, NOT an estimated peak or synthetic measurement.
        // Admission's zero-budget and fitting-control probes are recorded separately.
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-26241@{}", protocol::INFERENCE_PIN),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BoundaryError {
    MissingPreDenoise,
    InvalidZeroStep,
}
impl std::fmt::Display for BoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingPreDenoise => "missing_phase_boundary: native FLUX3 must emit typed ConditioningComplete after materialized, stream-fenced positive/negative/reference conditioning and before the first DiT forward; neither Renderer loading nor completed Step1 can label conditioning",
            Self::InvalidZeroStep => "invalid_phase_boundary: Progress::Step is 1-based; Step0 is not a conditioning-completion event",
        })
    }
}
impl std::error::Error for BoundaryError {}

/// Reducer for the native typed ConditioningComplete event, shared by both capture
/// adapters. Duplicate completion never resets a phase's peak or reopens conditioning.
pub(super) fn conditioning_complete(
    phase: &mut protocol::ReferencePhase,
) -> Option<protocol::ReferencePhase> {
    if *phase != protocol::ReferencePhase::Conditioning {
        return None;
    }
    *phase = protocol::ReferencePhase::Denoise;
    Some(*phase)
}

/// Cross only real pre-denoise boundaries, never completed solver Step1. Older
/// pins without the typed completion event fail rather than mislabeling phases.
pub(super) fn boundary(
    phase: &mut protocol::ReferencePhase,
    progress: &Progress,
) -> Result<Option<protocol::ReferencePhase>, BoundaryError> {
    if matches!(progress, Progress::Step { current: 0, .. }) {
        return Err(BoundaryError::InvalidZeroStep);
    }
    // Renderer loading can precede reference encoding. Only the provider's typed,
    // synchronized completion covers ALL conditioning on resident and staged routes.
    if matches!(progress, Progress::ConditioningComplete) {
        return Ok(conditioning_complete(phase));
    }
    if *phase == protocol::ReferencePhase::Conditioning
        && matches!(
            progress,
            Progress::Step { current: 1.., .. } | Progress::Decoding
        )
    {
        return Err(BoundaryError::MissingPreDenoise);
    }
    let reference = match progress {
        Progress::Decoding => protocol::ReferenceBoundary::Decoding,
        _ => return Ok(None),
    };
    let Some(next) = protocol::next_reference_phase(*phase, reference) else {
        return Ok(None);
    };
    *phase = next;
    Ok(Some(next))
}

pub(super) fn render(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    observed: &mut dyn FnMut(protocol::ReferencePhase),
) -> Result<(), String> {
    let mut request = GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(SEED),
        steps: Some(STEPS),
        guidance: Some(1.0),
        ..Default::default()
    };
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin FLUX3 memory request: {error}"))?
        .ok_or_else(|| "FLUX3 provider did not open its native memory request scope".to_owned())?;
    // Every post-open exit finishes the scope, including configure/enter errors.
    let mut canceled = false;
    let result = (|| {
        scope
            .configure_request(&mut request)
            .map_err(|error| error.to_string())?;
        scope
            .enter_phase(MemoryPhase::Conditioning)
            .map_err(|error| error.to_string())?;
        let mut phase = protocol::ReferencePhase::Conditioning;
        let mut phase_error = None;
        let cancel = request.cancel.clone();
        let output = generator.generate(&request, &mut |progress| {
            if phase_error.is_some() {
                return;
            }
            let next = match boundary(&mut phase, &progress) {
                Ok(next) => next,
                Err(error) => {
                    phase_error = Some(error.to_string());
                    cancel.cancel();
                    return;
                }
            };
            if let Some(next) = next {
                observed(next);
                let (before, after) = match next {
                    protocol::ReferencePhase::Denoise => {
                        (MemoryPhase::Conditioning, MemoryPhase::Denoise)
                    }
                    protocol::ReferencePhase::Decode => (MemoryPhase::Denoise, MemoryPhase::Decode),
                    _ => return,
                };
                if phase_error.is_none() {
                    if let Err(error) = scope
                        .leave_phase(before)
                        .and_then(|_| scope.enter_phase(after))
                    {
                        phase_error = Some(error.to_string());
                    }
                }
            }
        });
        if let Some(error) = phase_error {
            return Err(error);
        }
        canceled = matches!(&output, Err(core::Error::Canceled));
        let output = output.map_err(|error| format!("native FLUX3 generation failed: {error}"))?;
        if phase != protocol::ReferencePhase::Decode {
            return Err("FLUX3 generation omitted a measured phase boundary".to_owned());
        }
        scope
            .leave_phase(MemoryPhase::Decode)
            .map_err(|error| error.to_string())?;
        if !matches!(output, GenerationOutput::Images(ref images) if images.len() == 1 && images[0].width == 1024 && images[0].height == 1024)
        {
            return Err("FLUX3 fixture did not return exactly one real image".to_owned());
        }
        Ok(())
    })();
    let outcome = match &result {
        Ok(()) => MemoryRunOutcome::Complete,
        Err(_) if canceled => MemoryRunOutcome::Canceled,
        Err(message) => MemoryRunOutcome::Error {
            message: message.clone(),
        },
    };
    let finished = scope
        .finish(outcome)
        .map_err(|error| format!("finish FLUX3 memory scope: {error}"));
    match (result, finished) {
        (Ok(()), finished) => finished,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(finish)) => Err(format!("{error}; {finish}")),
    }
}

/// Admission controls against the loaded native generator, never a copied fit formula.
pub(super) fn admit(
    generator: &dyn Generator,
    plan: &Plan,
    request: &Value,
) -> Result<MemoryRunContext, String> {
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded FLUX3 provider exposes no memory contract".to_owned())?;
    contract
        .validate_selection(&plan.selection)
        .map_err(|error| error.to_string())?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "loaded FLUX3 provider exposes no calibration identity".to_owned())?;
    if calibration.fingerprint != plan.fingerprint
        || calibration.load_shape != LoadShape::EagerMaterialization
    {
        return Err(format!(
            "FLUX3 plan/provider calibration mismatch: planned {}, loaded {:?}",
            plan.fingerprint, calibration
        ));
    }
    let hardware = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "FLUX3 request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = context(plan, calibration.abi, hardware);
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "FLUX3 admission refused the fitting control; no capture is attested".to_owned(),
        );
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX3 admission accepted an unknown budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint.push_str("-mutation");
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX3 admission accepted a mutated calibration identity".to_owned());
    }
    Ok(context)
}

pub(super) fn fragment(
    request: &Value,
    lane: &str,
    plan: &Plan,
    strategy: Value,
    observed: Value,
    diagnostics: Value,
) -> Result<Value, String> {
    let blocker = "native FLUX3 anchor producer: real scoped generation and synchronized phase observations only; promotion quality, repeat/cancel/error lifecycle scenarios and fault injection have NOT been attested by this capture";
    let mut fragment = json!({
        "status": "gated", "strategy": strategy,
        "loadShape": protocol::LOAD_SHAPE_EAGER,
        "artifact": { "repository": FLUX3_REPOSITORY, "resolvedRevision": REVISION, "variant": plan.tier },
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below" }
        ],
        "predictedPeakBytes": null, "observedMemory": observed,
        "quality": { "result": "not_run" }, "negativeMutation": null,
        "loadability": { "result": "passed", "resolvedPathFingerprint": format!("{FLUX3_REPOSITORY}@{REVISION}:{}:{}:external-original-te-ae", plan.tier, conversion(lane, &plan.tier).unwrap_or("publisher")) },
        "diagnostics": diagnostics, "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(
        request,
        &mut fragment,
        "the native FLUX3 text-to-image path",
    )?;
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(lane: &str, tier: &str) -> Value {
        json!({ "planned": {
            "backend": lane,
            "target": { "provider": PROVIDER, "modelId": PROVIDER, "tier": tier, "mode": "text_to_image", "overlay": "none", "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 } },
            "loadShape": protocol::LOAD_SHAPE_EAGER,
            "calibrationFingerprint": fingerprint(lane, tier).unwrap(),
            "fixture": format!("flux3-klein-{lane}-{tier}-1024-seed26243-step16"),
            "strategy": { "rung": if lane == "candle" { "staged_residency" } else { "resident" }, "parameters": {} }
        } })
    }

    #[test]
    fn flux3_plan_validation_is_exact_and_precedes_any_weights_or_device() {
        for lane in ["candle", "mlx"] {
            for tier in ["bf16", "q4", "q8"] {
                let valid = request(lane, tier);
                assert_eq!(
                    plan(&valid, lane).unwrap().selection.tier.quant,
                    quant(lane, tier).unwrap()
                );
                for (pointer, mutation) in [
                    ("/planned/target/mode", json!("edit_image")),
                    ("/planned/target/overlay", json!("lora")),
                    ("/planned/target/geometry/batch", json!(2)),
                    ("/planned/loadShape", json!("deferred_materialization")),
                    ("/planned/fixture", json!("wrong-seed")),
                    ("/planned/calibrationFingerprint", json!("wrong-tier")),
                ] {
                    let mut invalid = valid.clone();
                    *invalid.pointer_mut(pointer).unwrap() = mutation;
                    assert!(plan(&invalid, lane).is_err(), "accepted mutation {pointer}");
                }
            }
        }
        for tier in ["fp8r", "nvfp4"] {
            assert_eq!(
                plan(&request("candle", tier), "candle")
                    .unwrap()
                    .selection
                    .tier
                    .quant,
                quant("candle", tier).unwrap()
            );
        }
    }

    #[test]
    fn flux3_phase_boundaries_follow_native_staged_and_resident_progress() {
        for staged in [false, true] {
            let mut phase = protocol::ReferencePhase::Conditioning;
            assert_eq!(
                boundary(&mut phase, &Progress::Loading(LoadPhase::TextEncoder)),
                Ok(None)
            );
            if staged {
                assert_eq!(
                    boundary(&mut phase, &Progress::Loading(LoadPhase::Renderer)),
                    Ok(None),
                    "reference encoding can still follow Renderer loading"
                );
            }
            assert_eq!(phase, protocol::ReferencePhase::Conditioning);
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(Some(protocol::ReferencePhase::Denoise))
            );
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(None),
                "duplicate completion must not reset peaks"
            );
            for current in 1..=STEPS {
                assert_eq!(
                    boundary(
                        &mut phase,
                        &Progress::Step {
                            current,
                            total: STEPS
                        }
                    ),
                    Ok(None)
                );
                assert_eq!(phase, protocol::ReferencePhase::Denoise);
            }
            assert_eq!(
                boundary(&mut phase, &Progress::Decoding),
                Ok(Some(protocol::ReferencePhase::Decode))
            );
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(None)
            );
            assert_eq!(phase, protocol::ReferencePhase::Decode);
        }
    }

    #[test]
    fn flux3_missing_or_late_boundary_and_zero_steps_fail_closed() {
        for staged in [false, true] {
            for current in [1, STEPS] {
                let mut phase = protocol::ReferencePhase::Conditioning;
                if staged {
                    assert_eq!(
                        boundary(&mut phase, &Progress::Loading(LoadPhase::Renderer)),
                        Ok(None)
                    );
                }
                assert_eq!(
                    boundary(
                        &mut phase,
                        &Progress::Step {
                            current,
                            total: STEPS
                        }
                    ),
                    Err(BoundaryError::MissingPreDenoise)
                );
                assert_eq!(
                    boundary(&mut phase, &Progress::Decoding),
                    Err(BoundaryError::MissingPreDenoise)
                );
                assert_eq!(phase, protocol::ReferencePhase::Conditioning);
            }
        }
        for mut phase in [
            protocol::ReferencePhase::Conditioning,
            protocol::ReferencePhase::Denoise,
            protocol::ReferencePhase::Decode,
        ] {
            assert_eq!(
                boundary(
                    &mut phase,
                    &Progress::Step {
                        current: 0,
                        total: STEPS
                    }
                ),
                Err(BoundaryError::InvalidZeroStep)
            );
        }
    }

    /// Scripted tensor-neutral provider: exercise the real render callback and scope
    /// cleanup without a registry, weights, allocator, or device.
    struct ScriptedGenerator {
        descriptor: core::ModelDescriptor,
        events: Vec<Progress>,
        cancel_at_end: bool,
        lifecycle: std::cell::RefCell<Vec<String>>,
        outcomes: std::cell::RefCell<Vec<MemoryRunOutcome>>,
        cancel_seen: std::cell::Cell<bool>,
    }

    impl ScriptedGenerator {
        fn new(events: Vec<Progress>, cancel_at_end: bool) -> Self {
            Self {
                descriptor: core::ModelDescriptor {
                    encoder_contract: None,
                    denoiser_output_latent_space: None,
                    id: "flux3-capture-script",
                    family: "flux3",
                    backend: "script",
                    modality: core::Modality::Image,
                    capabilities: core::Capabilities::default(),
                    required_components: &[],
                    control_kinds: None,
                },
                events,
                cancel_at_end,
                lifecycle: Default::default(),
                outcomes: Default::default(),
                cancel_seen: Default::default(),
            }
        }
    }

    struct ScriptedScope<'a>(&'a ScriptedGenerator);

    impl core::MemoryRequestScope for ScriptedScope<'_> {
        fn configure_request(&mut self, _: &mut GenerationRequest) -> core::Result<()> {
            Ok(())
        }
        fn enter_phase(&mut self, phase: MemoryPhase) -> core::Result<()> {
            self.0
                .lifecycle
                .borrow_mut()
                .push(format!("enter:{phase:?}"));
            Ok(())
        }
        fn leave_phase(&mut self, phase: MemoryPhase) -> core::Result<()> {
            self.0
                .lifecycle
                .borrow_mut()
                .push(format!("leave:{phase:?}"));
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _: u32) -> core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, outcome: MemoryRunOutcome) -> core::Result<()> {
            self.0.outcomes.borrow_mut().push(outcome);
            Ok(())
        }
    }

    impl Generator for ScriptedGenerator {
        fn descriptor(&self) -> &core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _: &GenerationRequest) -> core::Result<()> {
            Ok(())
        }
        fn begin_memory_strategy_request(
            &self,
            _: &MemoryRunContext,
        ) -> core::Result<Option<Box<dyn core::MemoryRequestScope + '_>>> {
            Ok(Some(Box::new(ScriptedScope(self))))
        }
        fn generate(
            &self,
            request: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> core::Result<GenerationOutput> {
            for event in &self.events {
                // Deliberately deliver late callbacks even after cancellation: the
                // capture must preserve the FIRST refusal rather than relabel peaks.
                progress(*event);
            }
            self.cancel_seen.set(request.cancel.is_cancelled());
            if self.cancel_at_end || request.cancel.is_cancelled() {
                return Err(core::Error::Canceled);
            }
            Ok(GenerationOutput::Images(vec![core::Image {
                width: 1024,
                height: 1024,
                pixels: vec![0; 1024 * 1024 * 3],
            }]))
        }
    }

    #[test]
    fn flux3_render_callback_orders_scopes_and_ignores_duplicate_boundaries() {
        for lane in ["candle", "mlx"] {
            let plan = plan(&request(lane, "bf16"), lane).unwrap();
            let mut events = vec![Progress::Loading(LoadPhase::TextEncoder)];
            if lane == "candle" {
                events.push(Progress::Loading(LoadPhase::Renderer));
            }
            events.extend([
                Progress::ConditioningComplete,
                Progress::ConditioningComplete,
                Progress::Step {
                    current: 1,
                    total: STEPS,
                },
                Progress::Step {
                    current: STEPS,
                    total: STEPS,
                },
                Progress::Decoding,
                Progress::ConditioningComplete,
            ]);
            let generator = ScriptedGenerator::new(events, false);
            let mut observed = Vec::new();
            render(&generator, &context(&plan, 1, 1 << 34), &mut |phase| {
                observed.push(phase)
            })
            .unwrap();
            assert_eq!(
                observed,
                [
                    protocol::ReferencePhase::Denoise,
                    protocol::ReferencePhase::Decode
                ]
            );
            assert_eq!(
                *generator.lifecycle.borrow(),
                [
                    "enter:Conditioning",
                    "leave:Conditioning",
                    "enter:Denoise",
                    "leave:Denoise",
                    "enter:Decode",
                    "leave:Decode",
                ]
            );
            assert!(matches!(
                generator.outcomes.borrow().as_slice(),
                [MemoryRunOutcome::Complete]
            ));
            assert!(!generator.cancel_seen.get());
        }
    }

    #[test]
    fn flux3_render_missing_or_late_boundary_cancels_and_preserves_typed_refusal() {
        let plan = plan(&request("candle", "bf16"), "candle").unwrap();
        for current in [0, 1] {
            let generator = ScriptedGenerator::new(
                vec![
                    Progress::Loading(LoadPhase::Renderer),
                    Progress::Step {
                        current,
                        total: STEPS,
                    },
                    Progress::ConditioningComplete,
                    Progress::Decoding,
                ],
                false,
            );
            let mut observed = Vec::new();
            let error = render(&generator, &context(&plan, 1, 1 << 34), &mut |phase| {
                observed.push(phase)
            })
            .unwrap_err();
            let expected = if current == 0 {
                BoundaryError::InvalidZeroStep
            } else {
                BoundaryError::MissingPreDenoise
            };
            assert_eq!(error, expected.to_string());
            assert!(generator.cancel_seen.get());
            assert!(
                observed.is_empty(),
                "late completion must not recover a failed capture"
            );
            assert_eq!(*generator.lifecycle.borrow(), ["enter:Conditioning"]);
            assert!(
                matches!(generator.outcomes.borrow().as_slice(), [MemoryRunOutcome::Error { message }] if message == &error)
            );
        }
    }

    #[test]
    fn flux3_render_provider_cancellation_finishes_once_without_fabricating_phases() {
        let plan = plan(&request("mlx", "bf16"), "mlx").unwrap();
        for after_conditioning in [false, true] {
            let events = if after_conditioning {
                vec![Progress::ConditioningComplete]
            } else {
                vec![]
            };
            let generator = ScriptedGenerator::new(events, true);
            let mut observed = Vec::new();
            let error = render(&generator, &context(&plan, 1, 1 << 34), &mut |phase| {
                observed.push(phase)
            })
            .unwrap_err();
            assert_eq!(error, "native FLUX3 generation failed: cancelled");
            assert_eq!(
                observed,
                if after_conditioning {
                    vec![protocol::ReferencePhase::Denoise]
                } else {
                    vec![]
                }
            );
            assert!(matches!(
                generator.outcomes.borrow().as_slice(),
                [MemoryRunOutcome::Canceled]
            ));
        }
    }

    #[test]
    fn flux3_exact_native_specs_and_external_components() {
        for lane in ["candle", "mlx"] {
            for (tier, expected) in [
                ("bf16", None),
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
            ] {
                let spec = load_spec(lane, tier, "derived".into(), "original".into()).unwrap();
                assert_eq!(spec.quantize, expected);
                assert_eq!(spec.precision, Precision::Bf16);
                assert_eq!(
                    spec.components.get("text_encoder"),
                    Some(&WeightsSource::Dir(PathBuf::from("original/text_encoder")))
                );
                assert_eq!(
                    spec.components.get("autoencoder"),
                    Some(&WeightsSource::File(PathBuf::from(
                        "original/ae.safetensors"
                    )))
                );
                if tier != "bf16" {
                    assert!(conversion(lane, tier)
                        .unwrap()
                        .starts_with(&format!("flux3-{lane}-")));
                }
            }
        }
        assert!(quant("mlx", "fp8r").is_err());
        assert!(quant("mlx", "nvfp4").is_err());
        assert_eq!(quant("candle", "fp8r").unwrap(), Some(Quant::Fp8Rowwise));
        assert_eq!(quant("candle", "nvfp4").unwrap(), Some(Quant::Nvfp4));
    }

    #[test]
    fn flux3_rejects_wrong_backend_package_and_snapshot_tier_guess() {
        let original = std::path::Path::new("original");
        for lane in ["candle", "mlx"] {
            for tier in ["q4", "q8"] {
                let id = conversion(lane, tier).unwrap();
                assert!(validate_package_root(
                    lane,
                    tier,
                    original,
                    &PathBuf::from("derived").join(tier).join(id)
                )
                .is_ok());
                assert!(validate_package_root(lane, tier, original, &original.join(tier)).is_err());
                let other = if lane == "candle" { "mlx" } else { "candle" };
                assert!(validate_package_root(
                    lane,
                    tier,
                    original,
                    &PathBuf::from("derived")
                        .join(tier)
                        .join(conversion(other, tier).unwrap())
                )
                .is_err());
            }
        }
    }
}
