//! MLX sdxl capture owner.
use super::*;

/// Which family member the plan asks for. Refuses by name — a model id no member serves must not
/// be measured as its nearest neighbour, and the engine id alone is NOT an artifact identity.
pub(super) fn sdxl_arm(request: &Value) -> Result<SdxlArm, String> {
    let target = protocol::planned_object(request, "/target")?;
    let provider = protocol::target_str(target, "provider")?;
    if provider != SDXL_PROVIDER {
        return Err(format!(
            "MLX SDXL base calibration does not implement provider {provider:?}"
        ));
    }
    let model_id = protocol::target_str(target, "modelId")?;
    SDXL_FAMILY
        .into_iter()
        .find(|arm| arm.model_id == model_id)
        .ok_or_else(|| {
            format!(
                "the MLX SDXL arm does not implement modelId {model_id:?} on provider \
                 {SDXL_PROVIDER:?} (family: {})",
                SDXL_FAMILY
                    .iter()
                    .map(|arm| arm.model_id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

pub(super) fn validate_sdxl_target(request: &Value) -> Result<SdxlArm, String> {
    let arm = sdxl_arm(request)?;
    let target = protocol::planned_object(request, "/target")?;
    let mode = protocol::target_str(target, "mode")?;
    if mode != "text_to_image" {
        return Err(format!(
            "{} requires reference-free text_to_image mode, got {mode:?}",
            arm.still_calibration
        ));
    }
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "{} requires {field} == 0 when declared",
                    arm.still_calibration
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "{} requires {field} == false when declared",
                    arm.still_calibration
                ));
            }
        }
    }
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    Ok(arm)
}

pub(super) fn validate_sdxl_selection_parameters(
    request: &Value,
    selection: &MemorySelection,
) -> Result<(), String> {
    let parameters = protocol::strategy_parameters(request)?;
    let mut keys = parameters.keys().map(String::as_str).collect::<Vec<_>>();
    keys.sort_unstable();
    let expected: &[&str] = match selection.strategy {
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency => &[],
        MemoryStrategy::BoundedDecode => &["decodeOverlap", "decodeTileEdge"],
        MemoryStrategy::BoundedTransformerResidency => &[
            "decodeOverlap",
            "decodeTileEdge",
            "transformerWindowComponent",
            "transformerWindowSize",
        ],
        MemoryStrategy::BoundedAttention => {
            return Err("MLX SDXL bounded attention is measured Missing at the pinned provider and is not capturable".to_owned());
        }
    };
    if keys != expected {
        return Err(format!(
            "MLX SDXL {:?} calibration requires exactly {expected:?}, got {keys:?}",
            selection.strategy
        ));
    }
    if matches!(
        selection.strategy,
        MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedTransformerResidency
    ) && (!selection
        .parameters
        .decode_tile_edge
        .is_some_and(|edge| mlx_gen_sdxl::memory_strategy::DECODE_TILE_EDGES.contains(&edge))
        || !selection.parameters.decode_overlap.is_some_and(|overlap| {
            mlx_gen_sdxl::memory_strategy::DECODE_OVERLAPS.contains(&overlap)
        }))
    {
        return Err(
            "MLX SDXL bounded decode requires a published native VAE tile and overlap".to_owned(),
        );
    }
    if selection.strategy == MemoryStrategy::BoundedTransformerResidency
        && (selection.parameters.transformer_window_size.is_none()
            || selection.parameters.transformer_window_component != Some(TransformerComponent::Dit))
    {
        return Err(
            "MLX SDXL bounded transformer calibration requires an explicit Dit window".to_owned(),
        );
    }
    Ok(())
}

pub(super) fn planned_sdxl_seed(
    request: &Value,
    arm: SdxlArm,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned_fixture(request)?;
    // sc-22729: the fixture stem names the MODEL, not the engine, so a `realvisxl` capture can
    // never be attributed to `sdxl` (both load through the same `sdxl` provider id).
    let prefix = format!("{}-{tier}-{width}-seed", arm.fixture_prefix);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse SDXL fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse SDXL fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

pub(super) fn sdxl_runtime_complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let axes = parameters
        .iter()
        .filter_map(|(name, value)| value.as_u64().map(|value| (name, value)))
        .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
        .collect::<Vec<_>>();
    Ok(json!({
        "axes": axes,
        "cases": [{ "parameters": parameters, "result": "passed" }],
        "rangeVerified": true,
    }))
}

pub(super) fn sdxl_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "an editorial photograph of a glass sculpture in a sunlit studio".to_owned(),
        negative_prompt: Some("low quality, blurry, distorted".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(2),
        guidance: Some(7.0),
        ..Default::default()
    }
}

pub(super) fn sdxl_load_spec(
    arm: SdxlArm,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    Ok((
        repository,
        revision,
        sdxl_spec_at(arm, root, tier, selection)?,
    ))
}

/// The env-free half of [`sdxl_load_spec`], so the tier and route binding are unit-testable.
///
/// Mirrors the worker's still-image MLX shape: `load_spec` + the resolved tier's advisory quant
/// (`mlx_load_quant_for_resolved_artifact` returns the quant for `sdxl` — it suppresses it only for
/// the krea/klein turnkeys), `DeferredMaterialization` from the registry's legacy shaper
/// (`memory_route_registry.rs` MLX `sdxl` rules are `legacy_shaping: true`), and
/// `with_resolved_route(request.model)` (`image_jobs/base.rs:8785`, "a shared engine such as `sdxl`
/// serves several independently pinned catalog routes"). The offload policy is not set on the
/// worker's MLX branch — it is decided later by `mlx_fit_gate::apply_residency_policy` — so the arm
/// binds the policy the SELECTED rung implies, which is what that gate would settle on.
pub(super) fn sdxl_spec_at(
    arm: SdxlArm,
    root: PathBuf,
    tier: &str,
    selection: &MemorySelection,
) -> Result<LoadSpec, String> {
    let (_, expected_quant) = tier_precision_quant(tier);
    if selection.tier.quant != expected_quant {
        return Err(format!(
            "{}: planned tier {tier} implies quant {expected_quant:?}, selection carries {:?}",
            arm.still_calibration, selection.tier.quant
        ));
    }
    // SDXL declares staging as a prerequisite of rung 4. Bounding decode alone only bounds
    // scratch and must not move the component load into the measured request.
    let offload = if selection.strategy == MemoryStrategy::StagedResidency
        || selection.strategy == MemoryStrategy::BoundedTransformerResidency
    {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(offload)
        .with_load_shape(LoadShape::DeferredMaterialization)
        .with_resolved_route(arm.model_id);
    if let Some(quant) = expected_quant {
        spec = spec.with_quant(quant);
    }
    Ok(spec)
}

pub(super) fn sdxl_context(
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
        evidence_revision: format!("sc-18379@{}", protocol::INFERENCE_PIN),
    }
}

/// Real capture arm for the exact recommended `mlx:sdxl` base T2I identity. It deliberately does
/// not expose SDXL edit/reference/adapter surfaces or the provider's measured-Missing attention rung.
pub(super) fn run_sdxl(request: &Value) -> Result<Value, String> {
    let arm = validate_sdxl_target(request)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != LoadShape::DeferredMaterialization {
        return Err(
            "plain SDXL calibration must use the production deferred_materialization load shape"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    validate_sdxl_selection_parameters(request, &selection)?;
    let tier = planned_qwen_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_sdxl_seed(request, arm, tier, width)?;
    if seed != SDXL_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SDXL calibration seed {SDXL_SEED}"
        ));
    }
    let (repository, revision, spec) = sdxl_load_spec(arm, tier, &selection)?;
    let registry = mlx_gen_sdxl::provider_registry()
        .map_err(|error| format!("build SDXL registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(SDXL_PROVIDER, &spec)
        .map_err(|error| format!("read {SDXL_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| format!("{SDXL_PROVIDER} has no memory-strategy contract at the pin"))?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned SDXL contract rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned SDXL contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != SDXL_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned SDXL fingerprint changed: expected {SDXL_CALIBRATION_FINGERPRINT}, got {}",
            calibration.fingerprint
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
    let generator = registry
        .load(SDXL_PROVIDER, &spec)
        .map_err(|error| format!("load real SDXL {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded SDXL generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err("loaded SDXL generator contract differs from the registry contract".to_owned());
    }
    let context = sdxl_context(
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
        sdxl_request(width, height, seed),
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
        return Err("a synchronized SDXL lifecycle phase reported a zero active peak".to_owned());
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
        return Err("SDXL provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("SDXL provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-sdxl-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("SDXL provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&sdxl_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected SDXL reference: {error}"))?,
    )?;
    let (maximum_error, mean_error, rms_error) = image_max_mean_rms_abs(&selected, &baseline)?;
    if maximum_error > MAX_THRESHOLD
        || mean_error > MEAN_THRESHOLD
        || rms_error > SDXL_RMS_THRESHOLD
    {
        return Err(format!(
            "SDXL selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if mutated_maximum <= MAX_THRESHOLD
        && mutated_mean <= MEAN_THRESHOLD
        && mutated_rms <= SDXL_RMS_THRESHOLD
    {
        return Err("SDXL output mutation did not breach the parity envelope".to_owned());
    }

    // The pinned SDXL provider exposes synchronized request scopes and phase telemetry, but not the
    // exhaustive lifecycle hooks required for `complete`: it never reads the calibration error
    // authorization. That missing error path alone prevents exhaustive lifecycle certification.
    // Preserve the real phase/quality/admission measurements as `runtime_complete`, keep the formal
    // lifecycle scenarios explicitly not_run, and record the internally verified mutation only as
    // diagnostics (the runtime-complete schema requires `negativeMutation: null`).
    let lifecycle_blocker = concat!(
        "the pinned mlx-gen-sdxl provider does not implement calibration error injection, so ",
        "exhaustive warm/cancel/",
        "error lifecycle scenarios remain not_run here; synchronized phase telemetry, selected-versus-",
        "unselected parity, and the internal negative discriminator are attested separately"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
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
        "quality": {
            "contract": "identical artifact, prompt, negative prompt, guidance, seed, geometry, steps and tier; selected SDXL rung versus unselected request",
            "identicalInputs": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "rootMeanSquareError": rms_error,
            "maximumErrorThreshold": MAX_THRESHOLD,
            "meanErrorThreshold": MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": SDXL_RMS_THRESHOLD,
        },
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
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
                ("loadShapeDeferred", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
