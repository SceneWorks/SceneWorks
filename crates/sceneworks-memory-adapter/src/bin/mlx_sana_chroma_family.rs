//! MLX sana chroma capture owner.
use super::*;

/// Which family member the plan asks for, refused by name. Unlike the FLUX.1 arm the key is the
/// provider alone: neither family has a second mode on this lane that loads a different artifact
/// (Chroma is `text_to_image` only, and SANA's `image_to_image` is the same base load driven from
/// a reference, which no anchor plans).
pub(super) fn sana_chroma_arm(request: &Value) -> Result<SanaChromaArm, String> {
    let provider = protocol::planned_provider(request)?;
    SANA_CHROMA_ARMS
        .iter()
        .copied()
        .find(|arm| arm.provider == provider)
        .ok_or_else(|| format!("the MLX SANA/Chroma1 arm does not implement provider {provider:?}"))
}

/// The production calibration identity the loaded generator publishes for one `(member, tier)`
/// cell — the tables `mlx-gen-sana::memory_strategy::production_calibration_fingerprint` and
/// `mlx-gen-chroma::memory_strategy::production_calibration_fingerprint` mint (inference PR 951).
///
/// The two MEASURED keys are read off the engine constants byte-for-byte rather than spelled here,
/// so this table cannot drift off the crates it mirrors; every other cell is the format string the
/// engine interpolates. SANA's key carries `-resident` because that is the policy this arm — and
/// the worker's MLX still-image path — loads under; Chroma's key carries no policy at all.
///
/// Written here as well as in the engines so the plan/arm binding is weights-free and can be proven
/// on a CPU-only host. The capture refuses a loaded contract whose identity differs from this
/// table, so the two copies cannot drift unnoticed once the epic's pin bump lands.
pub(super) fn sana_chroma_calibration_fingerprint(arm: SanaChromaArm, tier: &str) -> String {
    match (arm.family, arm.provider, tier) {
        (SanaChromaFamily::Sana, SANA_PROVIDER, "q4") => {
            runtime_macos::providers::sana::memory_strategy::RESIDENT_MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (SanaChromaFamily::Chroma, CHROMA1_BASE_PROVIDER, "q4") => {
            runtime_macos::providers::chroma::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (SanaChromaFamily::Sana, _, tier) => {
            format!("sana-{}-{tier}-mlx-full-ladder-v1-resident", arm.route)
        }
        (SanaChromaFamily::Chroma, _, tier) => {
            format!("chroma1-{}-{tier}-mlx-shared-ladder-v1", arm.route)
        }
    }
}

/// The weights-free conformance identities the two MLX engines publish for a registry-behaviour
/// contract that loaded no weights (`sana-<route>-<tier>-mlx-weights-free-conformance-v1-<policy>`
/// and `chroma1-mlx-registry-behavior-v1-<route>`). A plan row naming one of these could never be
/// satisfied by a production load. Test-only: the sole caller is the plan-identity conformance
/// test in `sana_chroma_tests`.
///
/// `_mlx_` in the name, and not extended to cover the Candle lane, because the Candle lane has NO
/// such namespace to guard at inference `c6d6a4db`: `candle-gen-sana::memory_strategy` mints the
/// single identity `sana-candle-dense-<route>-full-ladder-v1` (its `memory_strategy.rs:581`)
/// whether or not asset facts were injected — the weights-free fixture differs only in publishing
/// no `resident_components` — and `candle-gen-chroma` publishes no conformance string at all. A
/// shared predicate would therefore have been a guard over an empty set, which reads as coverage.
#[cfg(test)]
pub(super) fn is_sana_chroma_mlx_weights_free_fingerprint(fingerprint: &str) -> bool {
    fingerprint.contains("-mlx-weights-free-conformance-")
        || fingerprint.starts_with("chroma1-mlx-registry-behavior-")
}

/// The env-free half of [`sana_chroma_load_spec`], so the tier and root bindings are unit-testable
/// without weights. The root must end in the PLANNED tier's directory
/// (`.../snapshots/<revision>/<tier>`) — both engines then re-prove that answer against the
/// transformer's own packed marker and withhold the identity if the two disagree, so a stale
/// `…/q4` export can never satisfy a q8 or bf16 plan.
pub(super) fn sana_chroma_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<SanaChromaArtifact, String> {
    let arm = sana_chroma_arm(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let tier = match planned_qwen_tier(request)? {
        "bf16" => "bf16",
        "q4" => "q4",
        "q8" => "q8",
        _ => unreachable!("planned_qwen_tier returned an unsupported tier"),
    };
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    let root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    // Resident, like every other MLX still-image arm: the anchor rung is `resident`
    // (`memory-calibration-harness.mjs` `ANCHOR_STRATEGY.mlx`), and that is the policy the worker
    // loads both families under at that rung — `apply_declared_mlx_load_policy_for_request` finds
    // no `requiredOffloadPolicy` on either manifest's resident implementation, so gen-core's
    // `Resident` default stands. It matters more here than on FLUX.1: SANA's identity key CARRIES
    // the policy, so a Sequential load would publish a different string for the same cell.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    // The tier the worker requests, exactly as `image_jobs/base.rs` sends it: neither family is in
    // `mlx_load_quant_for_resolved_artifact`'s strip list, so the resolved tier rides
    // `LoadSpec::quantize` (`None` at bf16). Both engines then compare it to the artifact's own
    // packed marker before publishing an identity.
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    // P9: the worker binds the exact resolved route on every MLX image load
    // (`image_jobs/base.rs`: `.with_resolved_route(request.model.clone())`). These five catalog ids
    // are their own engine ids, so the route is the provider.
    let spec = spec.with_resolved_route(arm.provider.to_owned());
    Ok(SanaChromaArtifact {
        arm,
        repository,
        revision,
        tier,
        spec,
    })
}

pub(super) fn sana_chroma_load_spec(
    request: &Value,
    load_shape: LoadShape,
) -> Result<SanaChromaArtifact, String> {
    let arm = sana_chroma_arm(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = PathBuf::from(protocol::required_env(arm.root_env)?);
    sana_chroma_load_spec_at(request, load_shape, repository, revision, root)
}

/// The one fresh planned request every SANA/Chroma1 capture renders. Two steps, for the same reason
/// the FLUX.1 and Z-Image arms use two: a resident route has no provider loading boundary between
/// text encode and denoise, so the first Step callback closes a conservative conditioning envelope
/// and the second gives denoise its own measured interval before Decoding.
pub(super) fn sana_chroma_request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(SANA_CHROMA_SEED),
        steps: Some(2),
        ..Default::default()
    }
}

/// Bind the fixture to the planned member, tier AND geometry edge, so a bf16 record can never be
/// emitted against a q4 capture that merely reused the fixture string.
pub(super) fn validate_sana_chroma_fixture(
    request: &Value,
    arm: SanaChromaArm,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned_fixture(request)?;
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
        .map_err(|error| format!("parse SANA/Chroma1 fixture seed {seed:?}: {error}"))?;
    if seed != SANA_CHROMA_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SANA/Chroma1 calibration seed \
             {SANA_CHROMA_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse SANA/Chroma1 fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The admission context for the SANA/Chroma1 safety scenarios, in the shape the worker admits
/// these routes under: plain, reference-free, still text-to-image.
#[allow(clippy::too_many_arguments)]
pub(super) fn sana_chroma_context(
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
        evidence_revision: format!("sc-22731@{}", protocol::INFERENCE_PIN),
    }
}

/// The `mlx:sana_1600m` / `mlx:sana_sprint_1600m` / `mlx:chroma1_{hd,base,flash}` arm (sc-22731).
///
/// Loads through the SAME seam the worker loads these five routes through —
/// `runtime_macos::catalog().media().load(engine_id, &spec)`, which is exactly what
/// `crates/sceneworks-worker/src/inference_runtime.rs` wraps — reads the LOADED generator's own
/// contract, and measures the resident anchor composition against it.
pub(super) fn run_sana_chroma(request: &Value) -> Result<Value, String> {
    // Before the load, not inside it: a non-still target must be refused without paying for
    // weights. The arm is resolved first so the refusal carries the member's own label.
    let arm = sana_chroma_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let load_shape = planned_load_shape(request)?;
    // The member's materialization shape, refused BEFORE the load (sc-22731 review): the two
    // families genuinely differ, and a row carrying the other family's shape would otherwise pay
    // for a multi-gigabyte load before the identity comparison below could notice.
    if load_shape != arm.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, the {} route is captured {}",
            load_shape_key(load_shape),
            arm.provider,
            load_shape_key(arm.load_shape)
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    validate_sana_chroma_fixture(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    // The plan row must name the production identity this cell's loaded generator publishes —
    // checked against the weights-free table BEFORE the load, so a row still carrying a
    // conformance string fails in milliseconds rather than after a multi-gigabyte load.
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?.to_owned();
    let expected_fingerprint = sana_chroma_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }
    let artifact = sana_chroma_load_spec(request, load_shape)?;
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
    // name — an anchor recorded against no identity would claim measured authority the engine
    // never granted — and so is one that differs from the table the plan was bound to above.
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "the loaded {} provider at inference {} published no calibration identity for the \
             {tier} artifact; the production identity for this cell is {expected_fingerprint} \
             (inference PR 951 publishes one per (provider, tier) for every shipped cell), so this \
             cell captures only at a pin that carries it",
            arm.provider,
            protocol::INFERENCE_PIN
        )
    })?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    // ...and the identity's SHAPE, beside its name (sc-22731 review), as `mlx.rs`'s other arms do.
    // `MemoryCalibrationIdentity` carries the load shape it was minted under, so a plan row that
    // asked for SANA's eager resident ladder but reached a deferred contract — or a Chroma row the
    // other way round — would otherwise record a peak under an identity minted for a different
    // materialization.
    if calibration.load_shape != load_shape {
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
        generator.memory_strategy_safety_check(&sana_chroma_context(
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
        safety("stale-sana-chroma-fingerprint", hardware_bytes, 1),
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
                &sana_chroma_request(width, height),
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
            .generate(&sana_chroma_request(width, height), &mut |_| {})
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
            .generate(&sana_chroma_request(width, height), &mut |_| {})
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

    let lifecycle_blocker = concat!(
        "the pinned SANA/Chroma1 crates open no memory-strategy request scope for the resident ",
        "anchor composition and expose no calibration fault-injection site, so the scoped ",
        "lifecycle scenarios cannot execute; unscoped repeat determinism and allocator cleanup ",
        "bounds are attested in quality and diagnostics instead"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": artifact.artifact_json(),
        "sweep": sana_chroma_complete_sweep(request)?,
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

pub(super) fn sana_chroma_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact tuple per plan row; marking it range-verified promotes no sibling tuple (the
    // generated matrix still requires a matching manifest binding per cell).
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}
