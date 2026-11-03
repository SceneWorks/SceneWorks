//! MLX krea base capture owner.
use super::*;

/// Which family member the plan asks for. Refuses an unimplemented provider BY NAME, before any
/// environment variable is read or any path canonicalized — the same defense-in-depth `run`'s
/// dispatch already provides, restated here so a direct caller cannot measure one member's weights
/// against another member's contract.
pub(super) fn krea_base_arm(request: &Value) -> Result<KreaBaseArm, String> {
    let provider = protocol::planned(request)?
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    match provider {
        KREA_BASE_PROVIDER => Ok(KREA_TURBO_ARM),
        KREA_RAW_PROVIDER => Ok(KREA_RAW_ARM),
        provider => Err(format!(
            "the MLX Krea base arm does not implement provider {provider:?}"
        )),
    }
}

/// The pinned calibration identity this arm expects the loaded provider to publish for
/// `(member, tier)`.
///
/// sc-22735: every member is tier-keyed. The per-member frozen-string escape hatch this used to
/// carry existed only for `krea_2_turbo`, whose "measured" MLX key turned out to back no anchor at
/// all; it is gone rather than left standing for the next route to re-freeze itself with.
pub(super) fn krea_base_pinned_fingerprint(arm: KreaBaseArm, tier: &str) -> Option<String> {
    mlx_production_fingerprint(arm.provider, tier)
}

pub(super) fn validate_krea_base_target(request: &Value) -> Result<KreaBaseArm, String> {
    let arm = krea_base_arm(request)?;
    let label = arm.still_calibration;
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != arm.provider {
        return Err(format!(
            "{label} requires modelId {:?}, got {model_id:?}",
            arm.provider
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_image" {
        return Err(format!(
            "{label} requires reference-free text_to_image mode, got {mode:?}"
        ));
    }
    protocol::validate_still_geometry(request, label)?;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!("{label} requires {field} == 0 when declared"));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!("{label} requires {field} == false when declared"));
            }
        }
    }
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    Ok(arm)
}

pub(super) fn planned_krea_base_seed(
    request: &Value,
    arm: KreaBaseArm,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("{}-{tier}-{width}-seed", arm.fixture_stem);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Krea fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Krea fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

pub(super) fn krea_base_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // Each Krea plan row executes exactly one production parameter tuple. The singleton axes are
    // derived from that tuple, so a parameterized `complete` receipt satisfies the harness without
    // claiming any sibling value was exercised.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

pub(super) fn krea_base_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "an editorial photograph of a glass sculpture in a sunlit studio".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        // Two steps produce distinct conditioning/denoise/decode boundaries while keeping the
        // future physical capture bounded. This story adds no records or manifest bindings.
        steps: Some(2),
        ..Default::default()
    }
}

pub(super) fn krea_base_load_spec(
    request: &Value,
    arm: KreaBaseArm,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    validate_krea_base_target(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    krea_base_spec_at(arm, tier, &repository, &revision, root, selection)
        .map(|spec| (repository, revision, spec))
}

/// The env-free half of [`krea_base_load_spec`], so the tier binding and the member's quant policy
/// are unit-testable without an environment (the shape `z_image_load_spec_at` established).
///
/// The root must end in the PLANNED tier's directory (`.../snapshots/<revision>/<tier>`), so a stale
/// `…/q4` export can never satisfy a q8 or bf16 plan and quietly re-label another tier's peaks.
pub(super) fn krea_base_spec_at(
    arm: KreaBaseArm,
    tier: &str,
    repository: &str,
    revision: &str,
    root: PathBuf,
    selection: &MemorySelection,
) -> Result<LoadSpec, String> {
    protocol::validate_huggingface_snapshot_root(
        &root,
        repository,
        revision,
        tier,
        arm.expected_repository,
    )?;
    let offload = if selection.strategy == MemoryStrategy::Resident {
        OffloadPolicy::Resident
    } else {
        OffloadPolicy::Sequential
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(arm.load_shape);
    if arm.load_spec_carries_quant {
        if let Some(quant) = selection.tier.quant {
            spec = spec.with_quant(quant);
        }
    }
    Ok(spec)
}

pub(super) fn krea_base_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
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
        evidence_revision: format!("sc-18377@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn verify_krea_base_lifecycle(
    generator: &dyn Generator,
    label: &str,
    context: &MemoryRunContext,
    selected: &Image,
    width: u32,
    height: u32,
    seed: u64,
) -> Result<KreaBaseLifecycleMetrics, String> {
    clear_cache();
    reset_peak_memory();
    let clean_warm = one_image(scoped_generate(
        generator,
        krea_base_request(width, height, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let bounds = LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (warm_maximum, warm_mean) = image_max_mean_abs(selected, &clean_warm)?;
    if warm_maximum > KREA_MAX_THRESHOLD || warm_mean > KREA_MEAN_THRESHOLD {
        return Err("{label} clean warm control changed the deterministic output".to_owned());
    }

    let mut metrics = KreaBaseLifecycleMetrics {
        clean_warm_peak,
        clean_post_cleanup,
        ..Default::default()
    };
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let cancel = mlx_gen::CancelFlag::new();
        if phase == MemoryPhase::Conditioning {
            cancel.cancel();
        }
        let mut cancelled = krea_base_request(width, height, seed);
        cancelled.cancel = cancel.clone();
        let result = scoped_generate(generator, cancelled, context, None, &mut |progress| {
            if (phase == MemoryPhase::Denoise
                && matches!(progress, Progress::Step { current: 1, .. }))
                || (phase == MemoryPhase::Decode && matches!(progress, Progress::Decoding))
            {
                cancel.cancel();
            }
        });
        match result {
            Err(error) if error.to_ascii_lowercase().contains("cancel") => {}
            Err(error) => {
                return Err(format!(
                    "{label} {phase:?} cancellation returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{label} {phase:?} cancellation returned images instead of the typed cancellation path"
                ));
            }
        }
        clear_cache();
        let fault_cleanup = AllocatorState::capture_current();
        metrics.max_fault_post_cleanup.active = metrics
            .max_fault_post_cleanup
            .active
            .max(fault_cleanup.active);
        metrics.max_fault_post_cleanup.cache = metrics
            .max_fault_post_cleanup
            .cache
            .max(fault_cleanup.cache);
        if !bounds.allows_retained(fault_cleanup) {
            return Err(format!(
                "{label} {phase:?} cancellation retained active/cache bytes {fault_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        reset_peak_memory();
        let recovery = one_image(scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        metrics.max_recovery_peak = metrics.max_recovery_peak.max(recovery_peak);
        if !bounds.allows_warm_peak(recovery_peak) {
            return Err(format!(
                "{label} {phase:?} cancellation left the warm follow-up peak at {recovery_peak} bytes, above the clean warm control {clean_warm_peak} bytes plus 2%"
            ));
        }
        clear_cache();
        let recovery_cleanup = AllocatorState::capture_current();
        metrics.max_recovery_post_cleanup.active = metrics
            .max_recovery_post_cleanup
            .active
            .max(recovery_cleanup.active);
        metrics.max_recovery_post_cleanup.cache = metrics
            .max_recovery_post_cleanup
            .cache
            .max(recovery_cleanup.cache);
        if !bounds.allows_retained(recovery_cleanup) {
            return Err(format!(
                "{label} {phase:?} cancellation warm follow-up retained active/cache bytes {recovery_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        let (maximum, mean) = image_max_mean_abs(selected, &recovery)?;
        if maximum > KREA_MAX_THRESHOLD || mean > KREA_MEAN_THRESHOLD {
            return Err(format!(
                "{label} {phase:?} cancellation cleanup changed the warm follow-up"
            ));
        }
    }

    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let result = scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            Some(phase),
            &mut |_| {},
        );
        match result {
            Err(error) if error.contains("injected memory-strategy calibration error") => {}
            Err(error) => {
                return Err(format!(
                    "{label} {phase:?} error injection returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{label} {phase:?} error injection returned images instead of failing at its physical boundary"
                ));
            }
        }
        clear_cache();
        let fault_cleanup = AllocatorState::capture_current();
        metrics.max_fault_post_cleanup.active = metrics
            .max_fault_post_cleanup
            .active
            .max(fault_cleanup.active);
        metrics.max_fault_post_cleanup.cache = metrics
            .max_fault_post_cleanup
            .cache
            .max(fault_cleanup.cache);
        if !bounds.allows_retained(fault_cleanup) {
            return Err(format!(
                "{label} {phase:?} injected error retained active/cache bytes {fault_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        reset_peak_memory();
        let recovery = one_image(scoped_generate(
            generator,
            krea_base_request(width, height, seed),
            context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        metrics.max_recovery_peak = metrics.max_recovery_peak.max(recovery_peak);
        if !bounds.allows_warm_peak(recovery_peak) {
            return Err(format!(
                "{label} {phase:?} injected error left the warm follow-up peak at {recovery_peak} bytes, above the clean warm control {clean_warm_peak} bytes plus 2%"
            ));
        }
        clear_cache();
        let recovery_cleanup = AllocatorState::capture_current();
        metrics.max_recovery_post_cleanup.active = metrics
            .max_recovery_post_cleanup
            .active
            .max(recovery_cleanup.active);
        metrics.max_recovery_post_cleanup.cache = metrics
            .max_recovery_post_cleanup
            .cache
            .max(recovery_cleanup.cache);
        if !bounds.allows_retained(recovery_cleanup) {
            return Err(format!(
                "{label} {phase:?} injected-error warm follow-up retained active/cache bytes {recovery_cleanup:?} above the clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                bounds.tolerance_bytes,
            ));
        }
        let (maximum, mean) = image_max_mean_abs(selected, &recovery)?;
        if maximum > KREA_MAX_THRESHOLD || mean > KREA_MEAN_THRESHOLD {
            return Err(format!(
                "{label} {phase:?} error cleanup changed the warm follow-up"
            ));
        }
    }
    Ok(metrics)
}

/// Capture arm for the shipped, reference-free `mlx:krea_2_turbo` lane. The pose-control arm is
/// intentionally separate: its overlay, geometry and provider fingerprint are not interchangeable.
pub(super) fn run_krea_base(request: &Value) -> Result<Value, String> {
    let arm = validate_krea_base_target(request)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != arm.load_shape {
        return Err(format!(
            "{} must use the production {} load shape, but the plan declares {}",
            arm.still_calibration,
            load_shape_key(arm.load_shape),
            load_shape_key(planned_shape)
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    // PRE-LOAD fingerprint conformance, before any environment or weight work.
    validate_planned_fingerprint_is_producible(arm.provider, tier, request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_krea_base_seed(request, arm, tier, width)?;
    if seed != arm.seed {
        return Err(format!(
            "planned.fixture seed {seed} does not match the {} calibration seed {}",
            arm.still_calibration, arm.seed
        ));
    }
    let (repository, revision, spec) = krea_base_load_spec(request, arm, tier, &selection)?;
    let registry = mlx_gen_krea::provider_registry()
        .map_err(|error| format!("build Krea registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(arm.provider, &spec)
        .map_err(|error| format!("read {} memory-strategy contract: {error}", arm.provider))?
        .ok_or_else(|| {
            format!(
                "{} has no memory-strategy contract at the pin",
                arm.provider
            )
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} contract rejected planned selection: {error}",
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
            "pinned {} contract has no calibration identity",
            arm.provider
        )
    })?;
    if let Some(expected) = krea_base_pinned_fingerprint(arm, tier) {
        if calibration.fingerprint != expected {
            return Err(format!(
                "pinned {} fingerprint changed at tier {tier:?}: expected {expected}, got {}",
                arm.provider, calibration.fingerprint
            ));
        }
    }
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
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
    let generator = registry
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
    let loaded_contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} generator exposed no memory contract",
            arm.provider
        )
    })?;
    if loaded_contract != &contract {
        return Err(format!(
            "loaded {} generator contract differs from the registry contract",
            arm.provider
        ));
    }
    let context = krea_base_context(
        selection,
        calibration,
        &calibration.fingerprint,
        width,
        height,
        hardware_bytes,
        1,
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
        krea_base_request(width, height, seed),
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
    stale.calibration_fingerprint = "stale-krea-base-fingerprint".to_owned();
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
            .generate(&krea_base_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected {} reference: {error}", arm.provider))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if maximum_error > KREA_MAX_THRESHOLD || mean_error > KREA_MEAN_THRESHOLD {
        return Err(format!(
            "{} selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}",
            arm.provider
        ));
    }
    let lifecycle = verify_krea_base_lifecycle(
        generator.as_ref(),
        arm.provider,
        &context,
        &selected,
        width,
        height,
        seed,
    )?;
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if mutated_maximum <= KREA_MAX_THRESHOLD && mutated_mean <= KREA_MEAN_THRESHOLD {
        return Err(format!(
            "{} output mutation did not breach the parity envelope",
            arm.provider
        ));
    }

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": krea_base_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation; retained memory and warm recovery stayed within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries; retained memory and warm recovery stayed within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
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
            "contract": "identical artifact, prompt, seed, geometry, steps and tier; selected Krea rung versus unselected request",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": KREA_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": protocol::strategy_parameters(request)?,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}", arm.slug),
            "executed",
            [],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", lifecycle.clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", lifecycle.clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", lifecycle.clean_post_cleanup.cache),
                ("lifecycleMaximumFaultPostCleanupActive", "bytes", lifecycle.max_fault_post_cleanup.active),
                ("lifecycleMaximumFaultPostCleanupCache", "bytes", lifecycle.max_fault_post_cleanup.cache),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle.max_recovery_peak),
                ("lifecycleMaximumRecoveryPostCleanupActive", "bytes", lifecycle.max_recovery_post_cleanup.active),
                ("lifecycleMaximumRecoveryPostCleanupCache", "bytes", lifecycle.max_recovery_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("loadShapeDeferred", "count", u64::from(planned_shape == LoadShape::DeferredMaterialization)),
                ("loadSpecCarriesQuant", "count", u64::from(arm.load_spec_carries_quant)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
