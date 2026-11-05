//! MLX sd3 capture owner.
use super::*;

/// The production calibration identity `mlx-gen-sd3` publishes for a clean base load of one
/// `(route, tier)` cell of the shipped turnkey — the string
/// `mlx-gen-sd3::memory_strategy::production_calibration_fingerprint` mints (inference PR 950,
/// `crates/media/mlx-gen/mlx-gen-sd3/src/memory_strategy.rs:269-288` at `d606395b5`), where the
/// merged engine keys the identity on the PROVEN ARTIFACT TIER and not on the route alone: the
/// three tiers of one route are three different resident sets, so one string cannot price all
/// three.
///
/// Written here as well so the plan/arm binding is weights-free and holds at inference
/// `c6d6a4db`, whose `mlx-gen-sd3` publishes no production identity at all. The measured
/// `sd3-5-large-bf16-mlx-shared-ladder-2026-08-03-v1` is deliberately NOT reachable from this
/// table: it belongs only to the content-pinned SC-15522 snapshot loaded with `quantize == None`,
/// which is not a shipped turnkey and which no anchor plans. A capture that loads a different
/// identity — that measured string, or the weights-free registry string
/// `sd3-5-mlx-registry-behavior-v2-*` — is refused by name in `run_sd3` rather than emitted as
/// this cell's evidence, so the two copies cannot drift unnoticed once the pin bump lands.
pub(super) fn sd3_calibration_fingerprint(arm: Sd3Arm, tier: &str) -> String {
    format!("{}-{tier}-mlx-shared-ladder-v1", arm.slug)
}

/// Which family member the plan asks for, from `(target.provider, target.mode)` — never assumed.
/// SD3.5 declares an `image_to_image` route on both lanes, which no anchor plans and this arm does
/// not implement; measuring it as its nearest text-to-image neighbour would be a mislabelled record.
pub(super) fn sd3_arm(request: &Value) -> Result<Sd3Arm, String> {
    let planned = protocol::planned(request)?;
    let provider = protocol::plan_str(planned, "/target/provider")?;
    let mode = protocol::plan_str(planned, "/target/mode")?;
    SD3_ARMS
        .into_iter()
        .find(|arm| arm.provider == provider && mode == "text_to_image")
        .ok_or_else(|| {
            format!("the MLX SD3.5 arm does not implement provider {provider:?} in mode {mode:?}")
        })
}

/// The env-free half of [`sd3_load_spec`], so the tier and shape bindings are unit-testable without
/// a snapshot on disk.
///
/// The `LoadSpec` is the WORKER's still-image shape, not the engine's calibration-gate preference:
/// `image_jobs/base.rs` builds `LoadSpec::new(Dir(standard_tier_subdir(...)))` with the load quant
/// settled onto the resolved tier and then never touches `offload_policy` or `load_shape` — sd3 has
/// no `mlx.memoryStrategyContract`, so `evaluate_declared_mlx_load_shape_for_request` returns early,
/// `apply_declared_mlx_load_policy_for_request` early-returns on `NotEvaluated`, and
/// `memory_route_registry::RULES` carries no MLX sd3 row. The defaults are therefore what production
/// loads: `OffloadPolicy::Resident` + `LoadShape::EagerMaterialization`, which is also the MLX anchor
/// composition (`ANCHOR_STRATEGY.mlx`, `isDerivable`). Both are set EXPLICITLY here so a gen-core
/// default change cannot silently move the measured shape.
pub(super) fn sd3_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<Sd3Artifact, String> {
    let arm = sd3_arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let tier = match planned_qwen_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_qwen_tier returned an unsupported tier"),
    };
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export can never satisfy
    // a q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    let (precision, quant) = tier_precision_quant(tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    spec.precision = precision;
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    Ok(Sd3Artifact {
        arm,
        repository,
        revision,
        tier,
        spec,
    })
}

pub(super) fn sd3_load_spec(request: &Value, load_shape: LoadShape) -> Result<Sd3Artifact, String> {
    let arm = sd3_arm(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = PathBuf::from(protocol::required_env(arm.root_env)?);
    sd3_load_spec_at(request, load_shape, repository, revision, root)
}

/// The two-step calibration request, in the worker's own shape: SD3.5 is a true-CFG family, so the
/// guidance-carrying routes forward guidance and a negative prompt and Turbo forwards neither.
pub(super) fn sd3_request(arm: Sd3Arm, width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        negative_prompt: arm
            .guidance
            .map(|_| "low quality, blurry, distorted".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(SD3_SEED),
        // The first `Step` callback closes a conservative conditioning envelope; the second gives
        // denoise its own measured interval before `Decoding`.
        steps: Some(2),
        guidance: arm.guidance,
        ..Default::default()
    }
}

/// The fixture must name the member, tier and geometry it measured, so a Medium q8 record can never
/// be attributed to a Large bf16 capture that merely reused the string.
pub(super) fn planned_sd3_seed(
    arm: Sd3Arm,
    request: &Value,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned_fixture(request)?;
    let prefix = format!("{}-mlx-{tier}-{width}-seed", arm.slug);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse SD3.5 fixture seed {seed:?}: {error}"))?;
    if steps != "2" {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

pub(super) fn sd3_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
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
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22730@{}", protocol::INFERENCE_PIN),
    }
}

/// Real capture arm for the three SD3.5 base text-to-image routes, on every shipped tier.
///
/// Loads through the production catalog (`runtime_macos::catalog().media()`) — the same seam the
/// worker's `inference_runtime::load` uses — never a replica of the engine's loader.
/// The SD3.5 arm's runtime-complete quality block. Split out of [`run_sd3`] so the shape a record
/// is filed with can be asserted without real weights and a GPU — the defect this closes (sc-22738)
/// was a MISSING field in exactly this literal, which only a physical render could reach.
pub(super) fn sd3_quality(maximum_error: f64, mean_error: f64, rms_error: f64) -> Value {
    json!({
        "contract": "identical artifact, prompt, negative prompt, guidance, seed, geometry, steps and tier; selected SD3.5 rung versus unselected request",
        "identicalInputs": true,
        "result": "passed",
        "maximumError": maximum_error,
        "meanError": mean_error,
        "rootMeanSquareError": rms_error,
        "maximumErrorThreshold": MAX_THRESHOLD,
        "meanErrorThreshold": MEAN_THRESHOLD,
        "rootMeanSquareErrorThreshold": SD3_RMS_THRESHOLD,
    })
}

pub(super) fn run_sd3(request: &Value) -> Result<Value, String> {
    // Refuse an unimplemented `(provider, mode)` BEFORE any environment or weight work — the arm
    // itself is taken off the resolved artifact below, once the tier and repository are bound.
    sd3_arm(request)?;
    let planned_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    if selection.strategy != MemoryStrategy::Resident {
        return Err(format!(
            "MLX SD3.5 anchor capture measures the resident composition the worker loads, got {:?}",
            selection.strategy
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let artifact = sd3_load_spec(request, planned_shape)?;
    // Attribute the record to the arm the ARTIFACT was resolved for, not to the one the request
    // was matched against before any of the env or tier binding ran. They cannot differ — both go
    // through `sd3_arm` — but reading it off the artifact is what makes that a fact of the code
    // rather than of this function's ordering.
    let arm = artifact.arm;
    let tier = artifact.tier;
    let seed = planned_sd3_seed(arm, request, tier, width)?;
    if seed != SD3_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SD3.5 calibration seed {SD3_SEED}"
        ));
    }
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(arm.provider, &artifact.spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {} has no memory-strategy contract", arm.provider))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} provider rejected planned selection: {error}",
            arm.provider
        )
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "pinned {} provider has no calibration identity for the shipped turnkey; \
             mlx-gen-sd3 publishes one only from `production_calibration_fingerprint` (sc-22730)",
            arm.provider
        )
    })?;
    let expected_fingerprint = sd3_calibration_fingerprint(arm, artifact.tier);
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned {} published calibration {:?}, not this route's {} production identity {:?}",
            arm.provider, calibration.fingerprint, artifact.tier, expected_fingerprint
        ));
    }
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if calibration.load_shape != planned_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(planned_shape),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = sd3_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
    );

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
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        sd3_request(arm, width, height),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                conditioning.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            Progress::Decoding => {
                denoise.set(PhaseMemory::capture());
                reset_peak_memory();
            }
            _ => {}
        },
    )?)?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak",
            arm.provider
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    // Admission hygiene at the loaded provider's OWN gate: an exact fit is accepted (so the two
    // rejections below are evidence, not a blanket refusal), an unknown budget and a mutated
    // fingerprint are refused.
    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} provider rejected an exact-fit calibrated budget",
            arm.provider
        ));
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} provider accepted an unknown/zero memory budget",
            arm.provider
        ));
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-sd3-5-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} provider accepted stale calibration evidence",
            arm.provider
        ));
    }

    let baseline = one_image(
        generator
            .generate(&sd3_request(arm, width, height), &mut |_| {})
            .map_err(|error| format!("generate unselected {} reference: {error}", arm.provider))?,
    )?;
    // The RMS metric is measured, compared and published alongside max/mean because the
    // runtime-complete quality shape requires all three
    // (`memory-calibration-harness.mjs#validateRuntimeComplete`). Before sc-22738 this arm computed
    // only max/mean through `image_max_mean_abs` and published no `rootMeanSquareError` at all, so
    // three successfully rendered SD3.5 anchors were refused at record validation.
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if maximum_error > MAX_THRESHOLD || mean_error > MEAN_THRESHOLD || rms_error > SD3_RMS_THRESHOLD
    {
        return Err(format!(
            "{} selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}",
            arm.provider
        ));
    }
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if mutated_maximum <= MAX_THRESHOLD
        && mutated_mean <= MEAN_THRESHOLD
        && mutated_rms <= SD3_RMS_THRESHOLD
    {
        return Err(format!(
            "{} output mutation did not breach the parity envelope",
            arm.provider
        ));
    }

    let lifecycle_blocker = concat!(
        "sc-22730 anchor capture measures exact per-phase memory, strategy identity and admission ",
        "for the SD3.5 base routes; the formal warm/cancel/error lifecycle scenarios remain not_run ",
        "because this run does not repeat the full promotion-quality and lifecycle certification suite"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": sdxl_runtime_complete_sweep(request)?,
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
        "quality": sd3_quality(maximum_error, mean_error, rms_error),
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{}@{}:{tier}", artifact.repository, artifact.revision),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-shared-ladder", arm.slug),
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
