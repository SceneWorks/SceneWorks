//! MLX qwen edit capture owner.
use super::*;

/// Which family member the plan asks for. Refuses by name: the two ids share one engine provider, so
/// nothing else in the request distinguishes them, and measuring one under the other's key would
/// publish the distilled stack's peaks as the production path's (or the reverse).
pub(super) fn qwen_edit_arm(request: &Value) -> Result<QwenEditArm, String> {
    let planned = protocol::planned(request)?;
    let provider = protocol::plan_str(planned, "/target/provider")?;
    let model_id = protocol::plan_str(planned, "/target/modelId")?;
    match (provider, model_id) {
        (QWEN_EDIT_PROVIDER, QWEN_EDIT_MODEL) => Ok(QWEN_EDIT_ARM),
        (QWEN_EDIT_PROVIDER, QWEN_EDIT_LIGHTNING_MODEL) => Ok(QWEN_EDIT_LIGHTNING_ARM),
        (provider, model_id) => Err(format!(
            "the MLX Qwen edit arm does not implement provider {provider:?} for model {model_id:?}"
        )),
    }
}

/// The seed and step count this member's fixture binds, checked against the arm's own prefix, the
/// planned tier and the recipe's step count — the `planned_qwen_seed` rule, extended to a family
/// whose two members differ in exactly those two axes.
pub(super) fn planned_qwen_edit_seed(
    request: &Value,
    arm: QwenEditArm,
    tier: &str,
) -> Result<u64, String> {
    protocol::qwen_edit_fixture_seed(
        request,
        &format!("{}-{tier}-seed", arm.fixture_prefix),
        arm.steps,
    )
}

pub(super) use protocol::qwen_edit_lightning_adapter;

pub(super) fn qwen_edit_artifact_source(
    arm: QwenEditArm,
) -> Result<protocol::QwenEditArtifactSource, String> {
    protocol::qwen_edit_artifact_source(arm.lightning)
}

/// The env-free half of the Qwen edit load, so both of its load-time bindings are unit-testable:
///
/// * the root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a q8
///   or bf16 plan and quietly re-label another tier's peaks (the sc-17097 defect class); and
/// * the built-in distill lands in `spec.adapters` exactly when the arm is the Lightning member —
///   the ONE thing that makes that member Lightning at load time. Every downstream overlay claim in
///   [`run_qwen_edit_provider`] is read back off `spec.adapters` rather than off `arm.lightning`, so
///   a record can never assert an overlay the load did not carry.
pub(super) fn qwen_edit_load_spec(
    arm: QwenEditArm,
    tier: &str,
    source: &QwenEditArtifactSource,
    selection: &MemorySelection,
    offload: OffloadPolicy,
    load_shape: LoadShape,
) -> Result<QwenEditArtifact, String> {
    protocol::validate_artifact_identity(
        &source.repository,
        &source.revision,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root)
        .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_EDIT_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &source.repository,
        &source.revision,
        tier,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let mut spec = qwen_load_spec(root.clone(), selection, offload, load_shape);
    if arm.lightning {
        let lightning = source.lightning.as_ref().ok_or_else(|| {
            format!(
                "{} is the Lightning member but no distill snapshot was supplied",
                arm.model_id
            )
        })?;
        spec = spec.with_adapters(vec![qwen_edit_lightning_adapter(lightning)?]);
    }
    Ok(QwenEditArtifact { root, spec })
}

/// The generation request one Qwen edit capture renders: the worker's edit request shape — one
/// `Conditioning::Reference` fitted to the target geometry (`fit_engine_image` does that fit in
/// `image_jobs/base.rs`, so the engine always sees the request geometry) plus, on the Lightning
/// member, the distill's own sampler. The engine REFUSES a reference-free edit request
/// (`model_edit.rs` `validate`), so this is what makes the capture measure the edit path — the VL
/// vision tower and the dual-latent VAE encode — rather than text-to-image under an edit label.
pub(super) fn qwen_edit_request(
    arm: QwenEditArm,
    width: u32,
    height: u32,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: QWEN_EDIT_PROMPT.to_owned(),
        negative_prompt: (!arm.lightning).then(|| "blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(arm.steps),
        sampler: arm
            .lightning
            .then(|| QWEN_EDIT_LIGHTNING_SAMPLER.to_owned()),
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: None,
        }],
        ..Default::default()
    }
}

pub(super) fn qwen_edit_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// One Qwen-Image-Edit-2511 anchor capture, on either catalog id, at any shipped tier.
///
/// E4: the generator comes from `catalog.media().load(QWEN_EDIT_PROVIDER, &spec)` — the same
/// registry load the worker performs for this lane (`image_jobs/qwen.rs` builds a `LoadSpec` with
/// the tier quant plus the Lightning adapter stack and hands it to the cached-gen stream) — never a
/// re-implementation of the loader.
pub(super) fn run_qwen_edit_provider(request: &Value) -> Result<Value, String> {
    let arm = qwen_edit_arm(request)?;
    protocol::validate_exact_overlay_target(request, arm.overlay, arm.execution_path)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let seed = planned_qwen_edit_seed(request, arm, tier)?;
    let load_shape = planned_load_shape(request)?;
    let offload = if matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency | MemoryStrategy::BoundedTransformerResidency
    ) {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    let (width, height) = protocol::target_geometry(request)?;
    let source = qwen_edit_artifact_source(arm)?;
    let repository = source.repository.clone();
    let revision = source.revision.clone();
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks; and the distill is attached here,
    // on exactly the Lightning member.
    let QwenEditArtifact { root: _root, spec } =
        qwen_edit_load_spec(arm, tier, &source, &selection, offload, load_shape)?;
    // Every overlay claim below is read off the stack the LOAD carries, never off `arm.lightning`:
    // if the attachment above were ever lost, the record must say so rather than assert a distill
    // that never participated.
    let loaded_adapters = spec.adapters.len();
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(QWEN_EDIT_PROVIDER, &spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.model_id))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {QWEN_EDIT_PROVIDER} has no memory-strategy contract"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned Qwen edit provider rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Qwen edit provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if load_shape != calibration.load_shape {
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
    let mut context = qwen_provider_context(
        selection,
        calibration,
        width,
        height,
        hardware_bytes,
        1,
        22728,
    );
    // The mode the worker admits an edit job under, with the one reference the request carries, and
    // the overlay the target declares — the Lightning member really does load a second network, so
    // recording it under a reference-free, overlay-free context would be a false claim about what
    // the measured peaks contain.
    context.mode = MemoryMode::Edit;
    context.has_reference = true;
    context.geometry.reference_count = 1;
    context.overlay = (loaded_adapters > 0).then(|| arm.overlay.to_owned());

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
    let selected = one_image(scoped_generate(
        generator.as_ref(),
        qwen_edit_request(arm, width, height, seed),
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
        return Err(
            "a synchronized Qwen edit lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget.total_bytes = predicted;
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Qwen edit provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen edit provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-qwen-edit-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Qwen edit provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&qwen_edit_request(arm, width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate unselected Qwen edit reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !qwen_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Qwen edit selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let warm = one_image(scoped_generate(
        generator.as_ref(),
        qwen_edit_request(arm, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !qwen_quality_passes(warm_maximum, warm_mean) {
        return Err("Qwen edit warm repeat changed the deterministic output".to_owned());
    }

    let cancelled = qwen_edit_request(arm, width, height, seed);
    let cancel_signal = cancelled.cancel.clone();
    let cancel_during_decode = selection.strategy == MemoryStrategy::BoundedDecode;
    let mut cancel_triggered = false;
    let cancel_error = scoped_generate(
        generator.as_ref(),
        cancelled,
        &context,
        None,
        &mut |progress| {
            if cancel_triggered {
                return;
            }
            match progress {
                Progress::Step { current: 1, .. } if !cancel_during_decode => {
                    cancel_triggered = true;
                    cancel_signal.cancel();
                }
                Progress::Decoding if cancel_during_decode => {
                    cancel_triggered = true;
                    let signal = cancel_signal.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        signal.cancel();
                    });
                }
                _ => {}
            }
        },
    )
    .expect_err("in-flight Qwen edit cancellation must fail");
    if !cancel_triggered {
        return Err(
            "Qwen edit cancellation probe never reached the active rung boundary".to_owned(),
        );
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Qwen edit cancellation returned the wrong error: {cancel_error}"
        ));
    }
    let cancel_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_edit_request(arm, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !qwen_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Qwen edit cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected_phase = if selection.strategy == MemoryStrategy::BoundedDecode {
        MemoryPhase::Decode
    } else {
        MemoryPhase::Denoise
    };
    let injected = scoped_generate(
        generator.as_ref(),
        qwen_edit_request(arm, width, height, seed),
        &context,
        Some(injected_phase),
        &mut |_| {},
    )
    .expect_err("injected Qwen edit error must fail");
    if !injected.contains("injected memory-strategy calibration error") {
        return Err(format!(
            "Qwen edit error injection returned the wrong error: {injected}"
        ));
    }
    let error_recovery = one_image(scoped_generate(
        generator.as_ref(),
        qwen_edit_request(arm, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !qwen_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Qwen edit error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if qwen_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Qwen edit output mutation did not breach the production parity envelope".to_owned(),
        );
    }
    let overlay_scenario = if loaded_adapters > 0 {
        json!({
            "name": "overlay",
            "result": "passed",
            "reason": "the built-in lightx2v Lightning distill LoRA was folded into the MMDiT at load and participated in every measured render",
        })
    } else {
        json!({ "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" })
    };
    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": qwen_edit_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
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
            "contract": "same seed, reference, prompt, sampling, precision, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": QWEN_MAX_THRESHOLD,
            "meanErrorThreshold": QWEN_MEAN_THRESHOLD,
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
            &format!("memory-mlx-adapter:{}-shared-ladder", arm.slug),
            "executed",
            [],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                // Every Qwen edit capture conditions on exactly one reference image, and the
                // Lightning member additionally carries exactly one built-in distill adapter.
                ("referenceImages", "count", 1),
                ("builtInAdapters", "count", loaded_adapters as u64),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    if loaded_adapters == 0 {
        protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    }
    Ok(fragment)
}
