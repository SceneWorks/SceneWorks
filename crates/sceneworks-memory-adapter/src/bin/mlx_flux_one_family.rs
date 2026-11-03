//! MLX flux one capture owner.
use super::*;

/// Which family member the plan asks for. Refuses by name: a `(provider, mode)` pair no member
/// serves — `flux1_schnell` in `character_image` (schnell has no identity route at all),
/// `pulid_flux` in `text_to_image` (PuLID is text-to-image-WITH-A-FACE only; the worker's
/// `pulid_flux_available` requires `character_image` plus a reference) — must not be measured as
/// its nearest neighbour.
pub(super) fn flux_one_arm(request: &Value) -> Result<FluxOneArm, String> {
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
        (FLUX1_DEV_PROVIDER, "text_to_image") => Ok(FLUX1_DEV_ARM),
        (FLUX1_SCHNELL_PROVIDER, "text_to_image") => Ok(FLUX1_SCHNELL_ARM),
        (PULID_FLUX_PROVIDER, "character_image") => Ok(PULID_FLUX_ARM),
        (provider, mode) => Err(format!(
            "the MLX FLUX.1 arm does not implement provider {provider:?} in mode {mode:?}"
        )),
    }
}

/// The production calibration identity the loaded FLUX.1 generator publishes for one
/// `(member, tier)` cell — the table `mlx-gen-flux::memory_strategy::production_calibration_fingerprint`
/// and `mlx-gen-pulid::memory_strategy::production_calibration_fingerprint` mint (inference PR
/// 943): the two measured keys are preserved byte-for-byte and read off the engine constants, and
/// every other cell is `flux1-<route>-<tier>-mlx-shared-ladder-v1` /
/// `pulid-flux-dev-<tier>-mlx-shared-ladder-v1`. Written here as well so the plan/arm binding is
/// weights-free and holds at inference `c6d6a4db`, whose engines publish the two preserved keys
/// only; the capture refuses a loaded contract whose identity differs from this table, so the two
/// copies cannot drift unnoticed once the epic's pin bump lands.
pub(super) fn flux_one_calibration_fingerprint(arm: FluxOneArm, tier: &str) -> String {
    match (arm.provider, tier) {
        (FLUX1_DEV_PROVIDER, "q4") => {
            runtime_macos::providers::flux::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (PULID_FLUX_PROVIDER, "q4") => {
            runtime_macos::providers::pulid::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (FLUX1_DEV_PROVIDER, tier) => format!("flux1-dev-{tier}-mlx-shared-ladder-v1"),
        (FLUX1_SCHNELL_PROVIDER, tier) => format!("flux1-schnell-{tier}-mlx-shared-ladder-v1"),
        (PULID_FLUX_PROVIDER, tier) => format!("pulid-flux-dev-{tier}-mlx-shared-ladder-v1"),
        (provider, tier) => unreachable!("no FLUX.1 member {provider} at tier {tier}"),
    }
}

/// The weights-free conformance identities (`mlx-gen-flux` `STATIC_BEHAVIOR_FINGERPRINT` and
/// its `-<route>` suffixes, `mlx-gen-pulid` `STATIC_CALIBRATION`) the engines publish for a
/// registry-behaviour contract that loaded no weights. A plan row naming one of these could never
/// be satisfied by a production load. Test-only: the sole caller is the plan-identity
/// conformance test in `flux_one_tests`.
#[cfg(test)]
pub(super) fn is_flux_one_weights_free_fingerprint(fingerprint: &str) -> bool {
    fingerprint
        .starts_with(runtime_macos::providers::flux::memory_strategy::STATIC_BEHAVIOR_FINGERPRINT)
        || fingerprint == "pulid-flux-static-registry-behavior-v2"
}

/// The env-free half of [`flux_one_load_spec`], so the tier and identity bindings are unit-testable
/// without weights. The root must end in the PLANNED tier's directory
/// (`.../snapshots/<revision>/<tier>`), so a stale `…/q4` export can never satisfy a q8 or bf16
/// plan and quietly re-label another tier's peaks (the sc-17097 defect class).
pub(super) fn flux_one_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
    identity: Option<protocol::PulidIdentityBundle>,
) -> Result<FluxOneArtifact, String> {
    let arm = flux_one_arm(request)?;
    if arm.identity {
        protocol::validate_exact_overlay_target(request, "identity", arm.execution_path)?;
    } else {
        protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    }
    if arm.identity != identity.is_some() {
        return Err(format!(
            "{} requires an identity stack iff it is the PuLID route (identity route: {}, bundle \
             supplied: {})",
            arm.execution_path,
            arm.identity,
            identity.is_some()
        ));
    }
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
    // Resident, like every other MLX still-image arm (flux2, qwen, krea-control): the anchor rung
    // is `resident` (`memory-calibration-harness.mjs` `ANCHOR_STRATEGY.mlx`), and that is the
    // policy the worker loads all three routes under at that rung. The base providers go through
    // `image_jobs/base.rs`, whose declared-policy pass
    // (`apply_declared_mlx_load_policy_for_request`) finds no `requiredOffloadPolicy` on the
    // FLUX.1 manifests' `resident` implementation and leaves gen-core's `Resident` default; PuLID's
    // own pass (`image_jobs/pulid.rs` `pulid_declared_load_policy`) reads the same shape off the
    // `pulid_flux_dev` manifest. Only the `staged_residency` rows declare `sequential`, and no
    // anchor is captured there. The calibration identity is a property of the loaded artifact and
    // tier (inference PR 943), so the policy does not decide whether one is published.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    if let Some(bundle) = &identity {
        // The typed weight seam the `pulid_flux` loader reads (sc-8827): the adapter checkpoint,
        // the EVA tower, and the directory it then reads the three face models out of by name. The
        // worker builds exactly this (`image_jobs/pulid.rs` `pulid_identity_weights`) — no env is
        // consulted inside the engine.
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File(bundle.adapter.clone())),
            eva: Some(WeightsSource::File(bundle.eva.clone())),
            face_dir: Some(WeightsSource::Dir(bundle.face_dir.clone())),
        });
    }
    Ok(FluxOneArtifact {
        arm,
        repository,
        revision,
        tier,
        identity,
        spec,
    })
}

pub(super) fn flux_one_load_spec(
    request: &Value,
    load_shape: LoadShape,
) -> Result<FluxOneArtifact, String> {
    let arm = flux_one_arm(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = PathBuf::from(protocol::required_env(arm.root_env)?);
    let identity = if arm.identity {
        Some(protocol::pulid_identity_bundle()?)
    } else {
        None
    };
    flux_one_load_spec_at(request, load_shape, repository, revision, root, identity)
}

/// The one fresh planned request every FLUX.1 capture renders. Two steps are intentional: a
/// resident FLUX.1 route has no provider loading boundary between text encode and denoise, so the
/// first Step callback closes a conservative conditioning envelope and the second gives denoise its
/// own measured interval before Decoding — the shared z_image/qwen/flux2 phase-boundary pattern.
/// The identity route additionally carries the one reference face the worker requires
/// (`pulid_flux_available`), at the manifest's default reference strength.
pub(super) fn flux_one_request(arm: FluxOneArm, width: u32, height: u32) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(FLUX1_SEED),
        steps: Some(2),
        ..Default::default()
    };
    if arm.identity {
        request.conditioning = vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: Some(PULID_REFERENCE_STRENGTH),
        }];
    }
    request
}

/// Bind the fixture to the planned tier AND geometry edge, the same fixture-to-plan binding
/// `planned_flux2_seed` enforces, so a bf16 record can never be emitted against a q4 capture that
/// merely reused the fixture string.
pub(super) fn validate_flux_one_fixture(
    request: &Value,
    arm: FluxOneArm,
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
        .map_err(|error| format!("parse FLUX.1 fixture seed {seed:?}: {error}"))?;
    if seed != FLUX1_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the FLUX.1 calibration seed {FLUX1_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.1 fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The admission context for the FLUX.1 safety scenarios, in the shape the worker admits each
/// route under. The identity route is `MemoryMode::ImageToImage` with one reference and
/// `overlay: identity` — the PROVIDER mode, which is what `mlx-gen-pulid`'s route gate checks
/// (`memory_strategy::safety_check`), and which the manifest's `requestContexts` declare beside the
/// catalog mode `character_image`.
#[allow(clippy::too_many_arguments)]
pub(super) fn flux_one_context(
    arm: FluxOneArm,
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
        mode: if arm.identity {
            MemoryMode::ImageToImage
        } else {
            MemoryMode::TextToImage
        },
        has_reference: arm.identity,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(arm.identity),
        },
        reference_resources: None,
        overlay: arm.identity.then(|| "identity".to_owned()),
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22726@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn flux_one_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact tuple per plan row; marking it range-verified promotes no sibling tuple (the
    // generated matrix still requires a matching manifest binding per cell).
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:flux1_dev` / `mlx:flux1_schnell` / `mlx:pulid_flux` arm (sc-22726).
///
/// Loads through the SAME seam the worker loads these three routes through —
/// `runtime_macos::catalog().media().load(engine_id, &spec)`, which is exactly what
/// `crates/sceneworks-worker/src/inference_runtime.rs:356-358` wraps — reads the LOADED
/// generator's own contract, and measures the resident anchor composition against it.
pub(super) fn run_flux_one(request: &Value) -> Result<Value, String> {
    // Before the load, not inside it: a non-still target must be refused without paying for
    // weights. The arm is resolved first so the refusal carries the member's own label.
    let arm = flux_one_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let load_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    validate_flux_one_fixture(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    // The plan row must name the production identity this cell's loaded generator publishes —
    // checked against the weights-free table BEFORE the load, so a row still carrying a
    // conformance string fails in milliseconds rather than after a multi-gigabyte load.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();
    let expected_fingerprint = flux_one_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }
    let artifact = flux_one_load_spec(request, load_shape)?;
    // From the ARTIFACT, so the member that was actually bound to a root and an identity stack is
    // the member that is measured — not a second, independent resolution of the same plan.
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
             (inference PR 943 publishes one per (route, tier) for every load shape), so this \
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
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&flux_one_context(
            arm,
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
        safety("stale-flux1-fingerprint", hardware_bytes, 1),
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
                &flux_one_request(arm, width, height),
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
            .generate(&flux_one_request(arm, width, height), &mut |_| {})
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
            .generate(&flux_one_request(arm, width, height), &mut |_| {})
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
        "the pinned FLUX.1 crates open no memory-strategy request scope for the resident anchor ",
        "composition and expose no calibration fault-injection site, so the scoped lifecycle ",
        "scenarios cannot execute; unscoped repeat determinism and allocator cleanup bounds are ",
        "attested in quality and diagnostics instead"
    );
    let overlay_scenario = if arm.identity {
        json!({
            "name": "overlay",
            "result": "passed",
            "reason": "the PuLID identity stack (adapter, EVA tower, and the three face models) was resident for every measured render and is declared as its own resident component by the pinned contract",
        })
    } else {
        json!({ "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" })
    };
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": artifact.artifact_json(),
        "sweep": flux_one_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            overlay_scenario
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
                // The identity route conditions every request on one reference face.
                ("referenceImages", "count", u64::from(arm.identity)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    if !arm.identity {
        protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    }
    Ok(fragment)
}
