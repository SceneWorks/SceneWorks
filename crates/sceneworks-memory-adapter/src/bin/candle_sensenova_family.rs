//! CANDLE sensenova capture owner.
use super::*;

/// Which family member the plan asks for. Refuses by name — an unserved model id, or a
/// `(provider, modelId)` pair that crosses the quality/fast split, must not be measured as its
/// nearest neighbour.
pub(super) fn sensenova_candle_arm(request: &Value) -> Result<SenseNovaCandleArm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let model_id = planned
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    SENSENOVA_CANDLE_FAMILY
        .into_iter()
        .find(|arm| arm.model_id == model_id && arm.provider == provider)
        .ok_or_else(|| {
            format!(
                "the Candle SenseNova arm does not implement modelId {model_id:?} on provider \
                 {provider:?} (family: {})",
                SENSENOVA_CANDLE_FAMILY
                    .iter()
                    .map(|arm| format!("{}:{}", arm.provider, arm.model_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// The production calibration identity `candle-gen-sensenova` publishes for one `(route, tier)`
/// cell. The engine binds it to the ARTIFACT-PROVEN tier — `validate_numeric_tier` compares
/// `spec.quantize` against the `.scales` companions `detect_checkpoint_quantization` reads off the
/// pinned shards and errors on a disagreement, so the contract only reaches this table once the
/// requested tier IS the tier on disk. The offload policy is deliberately not in the key.
pub(super) fn sensenova_candle_calibration_fingerprint(
    arm: SenseNovaCandleArm,
    tier: &str,
) -> String {
    format!(
        "sensenova-u1-{}-{tier}-candle-request-memory-ladder-v1",
        runtime_cuda::providers::sensenova::memory_strategy::route_label(arm.model_id)
            .expect("registered SenseNova route")
    )
}

/// The fixture binds the member, the tier, the geometry edge, the seed and the step count — the
/// MLX arm's `validate_sensenova_fixture` on this lane's token.
pub(super) fn validate_sensenova_candle_fixture(
    request: &Value,
    arm: SenseNovaCandleArm,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{}-candle-{tier}-{width}-seed", arm.slug);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse SenseNova fixture seed {seed:?}: {error}"))?;
    if seed != SENSENOVA_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SenseNova calibration seed \
             {SENSENOVA_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse SenseNova fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The env-free half of the load binding, so the route, tier and composition are unit-testable
/// without weights.
pub(super) fn sensenova_candle_spec_at(
    arm: SenseNovaCandleArm,
    repository: &str,
    revision: &str,
    root: PathBuf,
    tier: &str,
) -> Result<LoadSpec, String> {
    protocol::validate_artifact_identity(repository, revision, arm.expected_repository)?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        repository,
        revision,
        tier,
        arm.expected_repository,
    )?;
    // Exactly the worker's Candle shape for this family, and NOT the shared five-rung
    // Sequential+Deferred composition: `engine_supports_sequential("sensenova_u1_8b")` is false, so
    // `image_jobs/base.rs` never sets `OffloadPolicy::Sequential` here and no SenseNova manifest
    // declares a BTR route entry — the load stays `Resident` + eager. The quant is the RESOLVED
    // tier's (`candle_quant_for_resolved_tier`: q4 -> Q4, q8 -> Q8, bf16 -> none); unlike the
    // Qwen/Z-Image snapshots, `candle-gen-sensenova` REQUIRES it, because `validate_numeric_tier`
    // hard-errors when `spec.quantize` disagrees with the checkpoint's own `.scales` packing.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_resolved_route(arm.model_id);
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(spec)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sensenova_candle_context(
    selection: MemorySelection,
    calibration: &runtime_cuda::gen_core::MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
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
        evidence_revision: format!("sc-22734@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn sensenova_candle_request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "an annotated infographic of a coastal lighthouse, crisp dense labels".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(SENSENOVA_SEED),
        steps: Some(2),
        ..Default::default()
    }
}

/// The `candle:sensenova_u1_8b` / `candle:sensenova_u1_8b_fast` family arm (sc-22734).
///
/// Loads through the SAME seam the worker loads all six routes through on this lane —
/// `runtime_cuda::catalog().media().load(engine_id, &spec)` — and measures the RESIDENT anchor.
/// This family does not ride the shared five-rung reference path: `candle-gen-sensenova` classifies
/// `StagedResidency` and `BoundedDecode` `StructurallyNotApplicable` (one fused dual-path
/// transformer, no separable conditioning component, no VAE decode phase — the manifest quotes both
/// reasons verbatim as `memoryStrategyStructuralExemptions`), so the five-rung path's
/// `Sequential` + deferred composition is not a shape this engine can be measured in, and the
/// staged rung is not a capturable anchor.
pub(super) fn run_sensenova_capture(request: &Value) -> Result<Value, String> {
    let arm = sensenova_candle_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let selection = planned_selection(request)?;
    if selection.strategy != MemoryStrategy::Resident {
        return Err(format!(
            "{} measures the resident anchor: the pinned SenseNova contract classifies \
             StagedResidency and BoundedDecode StructurallyNotApplicable, so no other rung is a \
             capturable anchor on this family",
            arm.still_calibration
        ));
    }
    let planned_shape = protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
        .to_owned();
    if planned_shape != load_shape_key(LoadShape::EagerMaterialization) {
        return Err(format!(
            "{} must use the eager_materialization load shape the worker loads this family under; \
             deferred materialization is the rung-4 (bounded transformer residency) shape, not the \
             resident anchor's",
            arm.still_calibration
        ));
    }
    let tier = planned_tier(request)?;
    validate_sensenova_candle_fixture(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    // The plan row must name the production identity this cell's loaded generator publishes —
    // checked BEFORE the load, so a stale row fails in milliseconds instead of after a
    // multi-gigabyte load. `run` re-checks it against the real contract below, which is what keeps
    // the two copies from drifting.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    let expected_fingerprint = sensenova_candle_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.model_id
        ));
    }

    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    let spec = sensenova_candle_spec_at(arm, &repository, &revision, root, tier)?;

    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog.media().load(arm.provider, &spec).map_err(|error| {
        format!(
            "load real {} {tier} provider on {}: {error}",
            arm.provider, arm.model_id
        )
    })?;
    vram.end_load(load_sample);

    let contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} exposed no memory-strategy contract",
            arm.provider
        )
    })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            arm.provider
        )
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    // The LOADED generator's own identity is what the record attests. An absent one is refused by
    // name: an anchor recorded against no identity would claim measured authority the engine never
    // granted.
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity for the {} \
             {tier} artifact; the production identity for this cell is {expected_fingerprint} \
             (the engine publishes one per (route, artifact-proven tier)), so this cell captures \
             only at a pin that carries it",
            arm.provider,
            protocol::INFERENCE_PIN,
            arm.model_id,
        )
    })?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if planned_shape != load_shape_key(calibration.load_shape) {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_shape}, pinned provider={}",
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&sensenova_candle_context(
            selection,
            calibration,
            fingerprint,
            width,
            height,
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
            arm.provider
        ));
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted an unknown/zero memory budget",
            arm.provider
        ));
    }
    if !matches!(
        safety("stale-sensenova-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            arm.provider
        ));
    }

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result = generator.generate(&sensenova_candle_request(width, height), &mut |progress| {
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = peak,
            MemoryPhase::Denoise => denoise_peak_gb = peak,
            MemoryPhase::Decode => decode_peak_gb = peak,
        }
        phase = memory_phase(next);
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    result.map_err(|error| format!("{} measured generation failed: {error}", arm.model_id))?;

    let conditioning_bytes = decimal_gb_to_bytes(
        conditioning_peak_gb
            .ok_or_else(|| format!("{} did not expose a conditioning boundary", arm.model_id))?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        denoise_peak_gb
            .ok_or_else(|| format!("{} did not expose a denoise boundary", arm.model_id))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb.ok_or_else(|| format!("{} did not complete decode", arm.model_id))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-22734 anchor capture measures exact per-phase device memory and the strategy identity ",
        "for the Candle SenseNova lane; it stays gated because this run does not repeat the full ",
        "promotion-quality, negative-mutation and lifecycle scenario suite, and the pinned crate ",
        "opens no memory-strategy request scope to inject a calibration fault into"
    );
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": { "repository": repository, "resolvedRevision": revision, "variant": tier },
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded SenseNova admission check rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded SenseNova admission check rejected a mutated calibration fingerprint" },
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
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-candle-adapter:sensenova-u1-{}-resident-anchor", arm.slug),
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
