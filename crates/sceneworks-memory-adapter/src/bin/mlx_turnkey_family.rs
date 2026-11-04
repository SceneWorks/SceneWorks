//! MLX turnkey capture owner.
use super::*;

/// Which turnkey member the plan asks for. Refuses by name: a `(provider, mode)` pair no member
/// serves must not be measured as its nearest neighbour. Kolors' `edit_image` and `character_image`
/// capabilities and Ideogram's `edit_image` / `image_inpaint` capabilities are real routes with
/// their own admission contexts, and none of them is this arm's plain text-to-image measurement.
pub(super) fn turnkey_arm(request: &Value) -> Result<TurnkeyArm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let mode = planned
        .pointer("/target/mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    match (provider, mode) {
        (KOLORS_PROVIDER, "text_to_image") => Ok(KOLORS_ARM),
        (IDEOGRAM_PROVIDER, "text_to_image") => Ok(IDEOGRAM_ARM),
        (IDEOGRAM_TURBO_PROVIDER, "text_to_image") => Ok(IDEOGRAM_TURBO_ARM),
        (LENS_PROVIDER, "text_to_image") => Ok(LENS_ARM),
        (LENS_TURBO_PROVIDER, "text_to_image") => Ok(LENS_TURBO_ARM),
        (QWEN_IMAGE_2_1_PROVIDER, "text_to_image") => Ok(QWEN_IMAGE_2_1_ARM),
        (QWEN_IMAGE_2_1_TURBO_PROVIDER, "text_to_image") => Ok(QWEN_IMAGE_2_1_TURBO_ARM),
        (provider, mode) => Err(format!(
            "the MLX turnkey still arm does not implement provider {provider:?} in mode {mode:?}"
        )),
    }
}

/// The env-free half of [`turnkey_load_spec`], so the tier and family bindings are unit-testable
/// without weights. The root must end in the PLANNED tier's directory
/// (`.../snapshots/<revision>/<tier>`), so a stale `…/q4` export can never satisfy a q8 or bf16 plan
/// and quietly re-label another tier's peaks (the sc-17097 defect class).
pub(super) fn turnkey_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<TurnkeyArtifact, String> {
    let arm = turnkey_arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let tier = match planned_qwen_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_qwen_tier returned an unsupported tier"),
    };
    let family = arm.family_for(tier);
    protocol::validate_artifact_identity(&repository, &revision, family.expected_repository)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize {}: {error}", family.root_env))?;
    if family.tiered {
        protocol::validate_huggingface_snapshot_root(
            &root,
            &repository,
            &revision,
            tier,
            family.expected_repository,
        )?;
    } else {
        // Qwen-Image 2.1 bf16 (sc-24112): the upstream snapshot ROOT, with no tier component to
        // check — inventing a `bf16/` suffix would bind a path no production load opens.
        protocol::validate_huggingface_revision_root(
            &root,
            &repository,
            &revision,
            family.expected_repository,
        )?;
    }
    // Resident + the plan's materialization shape, which is what the worker loads all five under.
    // `apply_declared_mlx_load_policy_for_request` returns the spec untouched here: Kolors and Lens
    // carry `legacy_shaping: true` MLX route rules (`memory_route_registry.rs`), so the declaration
    // is not owned and no `requiredOffloadPolicy` is admissible, and Ideogram has no MLX route rule
    // and no `mlx.memoryStrategyContract` at all. None of the five manifests declares a
    // `requiredOffloadPolicy` on any MLX row, so gen-core's `Resident` default stands and the
    // legacy `apply_registered_load_shape` path is what runs.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    // The worker FORWARDS the request-derived quant for all five —
    // `mlx_load_quant_for_resolved_artifact` (`image_jobs/base.rs`) nulls it only for the krea and
    // flux2-klein ids, and every one of these five manifests declares `mlx.quantize: 4`, so a
    // default job hands the provider a real `Quant`. It is an advisory no-op on the already-packed
    // tier, but binding `None` here would be a DIFFERENT load from the one the worker issues.
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    Ok(TurnkeyArtifact {
        arm,
        repository,
        revision,
        tier,
        spec,
    })
}

pub(super) fn turnkey_load_spec(
    request: &Value,
    load_shape: LoadShape,
) -> Result<TurnkeyArtifact, String> {
    let arm = turnkey_arm(request)?;
    let family = arm.family_for(planned_qwen_tier(request)?);
    let repository = protocol::required_env(family.repository_env)?;
    let revision = protocol::required_env(family.revision_env)?;
    let root = PathBuf::from(protocol::required_env(family.root_env)?);
    turnkey_load_spec_at(request, load_shape, repository, revision, root)
}

/// The one fresh planned request every turnkey capture renders.
pub(super) fn turnkey_request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(TURNKEY_SEED),
        steps: Some(TURNKEY_STEPS),
        ..Default::default()
    }
}

/// Bind the fixture to the planned member, tier, geometry edge, seed and step count, so a bf16
/// record can never be emitted against a q4 capture that merely reused the fixture string.
pub(super) fn validate_turnkey_fixture(
    request: &Value,
    arm: TurnkeyArm,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{}-mlx-{tier}-{width}-seed", arm.slug);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse turnkey fixture seed {seed:?}: {error}"))?;
    if seed != TURNKEY_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the turnkey calibration seed {TURNKEY_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse turnkey fixture step count {steps:?}: {error}"))?;
    if steps != TURNKEY_STEPS {
        return Err(format!(
            "planned.fixture {fixture:?} must use the {TURNKEY_STEPS}-step calibration request"
        ));
    }
    Ok(())
}

/// The production calibration identity the loaded turnkey generator publishes for one
/// `(member, tier)` cell — the tables `mlx-gen-kolors`, `mlx-gen-ideogram` and `mlx-gen-lens`
/// `memory_strategy::production_calibration_fingerprint` mint (inference PR
/// `story/sc-22732-epic-22723-memory-anchor-measurability`): the two measured keys are preserved
/// byte-for-byte and read off the engine constants, and every other cell is
/// `kolors-<tier>-mlx-shared-ladder-v1` / `ideogram-4[-turbo]-<tier>-mlx-shared-ladder-v1` /
/// `lens[-turbo]-<tier>-mlx-shared-ladder-v1`. Written here as well so the plan/arm binding is
/// weights-free and holds at inference `c6d6a4db`, whose engines publish the two preserved keys
/// only (and `mlx-gen-ideogram` none at all); the capture refuses a loaded contract whose identity
/// differs from this table, so the two copies cannot drift unnoticed once the epic's pin bump
/// lands.
///
/// `lens_turbo` bf16 is deliberately NOT `mlx-gen-lens`'s `LEGACY_TEXT_ENCODER_FINGERPRINT`: that
/// string names the SC-15800 narrowed text-encoder envelope, reachable only under
/// `Sequential + DeferredMaterialization` (`is_streamable_spec`), and this arm loads every member
/// `Resident` + the plan's eager shape, where the engine publishes the full-ladder key.
///
/// FAIL-CLOSED on the tier axis too (sc-22732 review). The per-member arms below bind `tier` as a
/// free variable, so before this guard any string at all — `"q2"`, `"nvfp4"`, a typo in a plan row —
/// minted a plausible-looking identity no engine publishes, and the caller's equality check would
/// then accept it. Only the three tiers the turnkey family ships are nameable; every other tier is
/// `None`, which the caller turns into a refusal before any weight work.
pub(super) fn turnkey_calibration_fingerprint(arm: TurnkeyArm, tier: &str) -> Option<String> {
    if !matches!(tier, "bf16" | "q4" | "q8") {
        return None;
    }
    Some(match (arm.provider, tier) {
        (KOLORS_PROVIDER, "q4") => {
            runtime_macos::providers::kolors::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (LENS_PROVIDER, "q4") => {
            runtime_macos::providers::lens::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        // sc-24112: `mlx-gen-qwen-image-2-1` publishes ONE derived identity for every tier (the
        // load shape is its separate typed axis), read off the engine constant.
        (QWEN_IMAGE_2_1_PROVIDER, _) => {
            runtime_macos::providers::qwen_image_2_1::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        // sc-25907: the Turbo id publishes its OWN single identity (it prices a different
        // checkpoint), so a base record can never satisfy a Turbo cell's handshake.
        (QWEN_IMAGE_2_1_TURBO_PROVIDER, _) => {
            runtime_macos::providers::qwen_image_2_1::memory_strategy::TURBO_MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (KOLORS_PROVIDER, tier) => format!("kolors-{tier}-mlx-shared-ladder-v1"),
        (IDEOGRAM_PROVIDER, tier) => format!("ideogram-4-{tier}-mlx-shared-ladder-v1"),
        (IDEOGRAM_TURBO_PROVIDER, tier) => format!("ideogram-4-turbo-{tier}-mlx-shared-ladder-v1"),
        (LENS_PROVIDER, tier) => format!("lens-{tier}-mlx-shared-ladder-v1"),
        (LENS_TURBO_PROVIDER, tier) => format!("lens-turbo-{tier}-mlx-shared-ladder-v1"),
        // A provider that is not a turnkey member is unnameable rather than collapsed onto one.
        _ => return None,
    })
}

/// The weights-free conformance identities the three engines publish for a registry-behaviour
/// contract that loaded no weights — `mlx-gen-lens` `STATIC_BEHAVIOR_FINGERPRINT` (an engine
/// constant at the pin) and the `kolors-mlx-registry-behavior-v1` / `ideogram4-mlx-registry-behavior-v1`
/// namespaces the sc-22732 inference head introduces, each with its `-<route>-…` suffixes — plus
/// `mlx-gen-lens`'s legacy dense text-encoder key, which no `Resident` load returns. A plan row
/// naming any of these could never be satisfied by a production load. Test-only: the sole caller
/// is the plan-identity conformance test in `turnkey_still_tests`.
#[cfg(test)]
pub(super) fn is_turnkey_weights_free_fingerprint(fingerprint: &str) -> bool {
    fingerprint
        .starts_with(runtime_macos::providers::lens::memory_strategy::STATIC_BEHAVIOR_FINGERPRINT)
        || fingerprint.starts_with("kolors-mlx-registry-behavior-v1")
        || fingerprint.starts_with("ideogram4-mlx-registry-behavior-v1")
        || fingerprint
            == runtime_macos::providers::lens::memory_strategy::LEGACY_TEXT_ENCODER_FINGERPRINT
}

/// The admission context for the turnkey safety scenarios, in the shape the worker admits each
/// member under: plain reference-free text-to-image, no overlay, no PiD.
pub(super) fn turnkey_context(
    selection: &MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: *selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        // A parameter only so the stale-evidence probe can pass a deliberate mismatch; the real
        // call sites pass `calibration.fingerprint` (the Krea-arm lesson at `krea_context`).
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
        evidence_revision: format!("sc-22732@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn turnkey_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:kolors` / `mlx:ideogram_4` / `mlx:ideogram_4_turbo` / `mlx:lens` / `mlx:lens_turbo` arm
/// (sc-22732).
///
/// Loads through the SAME seam the worker loads these five through —
/// `runtime_macos::catalog().media().load(engine_id, &spec)`, which is exactly what
/// `crates/sceneworks-worker/src/inference_runtime.rs` wraps — reads the LOADED generator's own
/// contract, and measures the resident anchor composition against it.
pub(super) fn run_turnkey_still(request: &Value) -> Result<Value, String> {
    // Before the load, not inside it: a non-still target must be refused without paying for
    // weights. The arm is resolved first so the refusal carries the member's own label.
    let arm = turnkey_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let load_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    validate_turnkey_fixture(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    // The plan row must name the production identity this cell's loaded generator publishes —
    // checked against the weights-free table BEFORE the load, so a row still carrying a
    // conformance string (or the legacy dense Lens-Turbo key) fails in milliseconds rather than
    // after a multi-gigabyte load.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    let expected_fingerprint = turnkey_calibration_fingerprint(arm, tier).ok_or_else(|| {
        format!(
            "no turnkey member {} at tier {tier}: the turnkey family ships bf16, q4 and q8 only",
            arm.provider
        )
    })?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }
    let artifact = turnkey_load_spec(request, load_shape)?;
    // From the ARTIFACT, so the member that was actually bound to a root is the member that is
    // measured — not a second, independent resolution of the same plan.
    let arm = artifact.arm;

    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(arm.provider, &artifact.spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
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
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    // The LOADED generator's own identity is what the record attests. An absent one is refused by
    // name — an anchor recorded against no identity would claim measured authority the engine never
    // granted.
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity for the \
             {tier} artifact",
            arm.provider,
            protocol::INFERENCE_PIN
        )
    })?;
    if calibration.fingerprint != planned_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if load_shape_key(calibration.load_shape) != load_shape_key(load_shape) {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(load_shape),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&turnkey_context(
            &selection,
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
        safety("stale-turnkey-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            arm.provider
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
    let selected = one_image(
        generator
            .generate(
                &turnkey_request(width, height),
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
            )
            .map_err(|error| format!("generate measured {} render: {error}", arm.provider))?,
    )?;
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
    if !matches!(
        safety(&calibration.fingerprint, predicted, predicted),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected an exact-fit calibrated budget",
            arm.provider
        ));
    }

    // Warm-repeat determinism and allocator cleanup bounds on this exact loaded provider.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generator
            .generate(&turnkey_request(width, height), &mut |_| {})
            .map_err(|error| format!("generate warm {} control: {error}", arm.provider))?,
    )?;
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
            arm.provider
        ));
    }
    reset_peak_memory();
    let warm = one_image(
        generator
            .generate(&turnkey_request(width, height), &mut |_| {})
            .map_err(|error| format!("generate warm {} repeat: {error}", arm.provider))?,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "{} warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%",
            arm.provider
        ));
    }
    clear_cache();
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "{} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the clean warm \
             cleanup {clean_post_cleanup:?} plus {} bytes",
            arm.provider, cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            arm.provider
        ));
    }

    // Arm-internal negative-mutation falsifiability check: a runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here and the numbers land in diagnostics.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.provider
        ));
    }

    let lifecycle_blocker = if matches!(
        arm.provider,
        QWEN_IMAGE_2_1_PROVIDER | QWEN_IMAGE_2_1_TURBO_PROVIDER
    ) {
        concat!(
            "the pinned Qwen-Image 2.1 MLX crate opens a memory-strategy request scope ",
            "(registered_begin_request) but exposes no calibration fault-injection site, so the ",
            "scoped lifecycle scenarios cannot execute; unscoped repeat determinism and allocator ",
            "cleanup bounds are attested in quality and diagnostics instead"
        )
    } else {
        concat!(
            "the pinned Kolors, Ideogram and Lens crates open no memory-strategy request scope for ",
            "the resident anchor composition and expose no calibration fault-injection site, so the ",
            "scoped lifecycle scenarios cannot execute; unscoped repeat determinism and allocator ",
            "cleanup bounds are attested in quality and diagnostics instead"
        )
    };
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": artifact.artifact_json(),
        "sweep": turnkey_complete_sweep(request)?,
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
            "contract": "identical artifact, prompt, seed, geometry, steps, tier, and loaded provider; cold measured render versus warm unscoped repeats",
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
            "resolvedPathFingerprint": artifact.loadability_fingerprint(),
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
                ("lifecycleCleanWarmPeak", "bytes", clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", cleanup_bounds.tolerance_bytes),
                ("lifecycleWarmRepeatPeak", "bytes", warm_peak),
                ("lifecycleWarmRepeatPostCleanupActive", "bytes", warm_post_cleanup.active),
                ("lifecycleWarmRepeatPostCleanupCache", "bytes", warm_post_cleanup.cache),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("loadShapeDeferred", "count", u64::from(calibration.load_shape == LoadShape::DeferredMaterialization)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
