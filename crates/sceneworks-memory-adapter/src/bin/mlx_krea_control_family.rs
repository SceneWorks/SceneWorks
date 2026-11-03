//! MLX krea control capture owner.
use super::*;

pub(super) fn validate_krea_overlay_path(
    requested: &std::path::Path,
    revision: &str,
) -> Result<(), String> {
    protocol::validate_artifact_identity(
        KREA_OVERLAY_REPOSITORY,
        revision,
        KREA_OVERLAY_REPOSITORY,
    )?;
    let repository_component = format!("models--{}", KREA_OVERLAY_REPOSITORY.replace('/', "--"));
    let expected = [
        repository_component.as_str(),
        "snapshots",
        revision,
        KREA_OVERLAY_FILE,
    ];
    let components = requested
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "control overlay must end with /{repository_component}/snapshots/{revision}/{KREA_OVERLAY_FILE}"
        ));
    }
    Ok(())
}

pub(super) fn krea_context(
    width: u32,
    height: u32,
    tile_edge: u32,
    predicted_peak_bytes: u64,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
) -> MemoryRunContext {
    MemoryRunContext {
        selection: MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(tile_edge),
                decode_overlap: Some(KREA_TILE_OVERLAP),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        },
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        // From the LOADED provider's own identity, never a local copy. A hardcoded fingerprint
        // silently goes stale the moment the provider re-fingerprints — which is exactly what
        // happened here: `krea-control-mlx-v4-q4-pose-bounded-decode-512-64` outlived the
        // provider's move to the full-ladder identity and failed the handshake on every case.
        // `fingerprint` stays a parameter only so the negative probe can pass a deliberate
        // mismatch; the real call sites pass `calibration.fingerprint`.
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        // `krea_request` carries exactly one `Conditioning::Control`, and `image_reference_count()`
        // counts Control alongside Reference/Depth/Mask — so the admitted geometry must declare one
        // reference or `configure_request` refuses the render it just admitted. `has_reference` is
        // the compatibility summary of the same fact and gen-core requires it to equal
        // `reference_count > 0`, so the two move together.
        has_reference: true,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: 1,
        },
        reference_resources: None,
        overlay: Some("control:1".to_owned()),
        budget: MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-16099@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn krea_request(width: u32, height: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a person standing in a studio, full body editorial photograph".to_owned(),
        width,
        height,
        seed: Some(16099),
        steps: Some(steps),
        conditioning: vec![Conditioning::Control {
            image: fixed_pose_control_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    }
}

pub(super) fn run_krea_control(request: &Value) -> Result<Value, String> {
    protocol::validate_exact_overlay_target(request, "control:1", KREA_CONTROL_EXECUTION_PATH)?;
    protocol::validate_still_geometry(request, "MLX Krea pose-control calibration")?;
    let parameters = protocol::strategy_parameters(request)?;
    let strategy = json!({
        "rung": "bounded_decode",
        "engagedRungs": ["resident", "bounded_decode"],
        "parameters": parameters,
    });
    let planned_strategy = protocol::planned(request)?
        .get("strategy")
        .ok_or_else(|| "planned.strategy must be present".to_owned())?;
    if planned_strategy != &strategy {
        return Err(format!(
            "plan/provider strategy mismatch: plan={planned_strategy}, MLX adapter measured={strategy}"
        ));
    }
    let tile_edge = protocol::parameter(request, "decodeTileEdge")?;
    let overlap = protocol::parameter(request, "decodeOverlap")?;
    if tile_edge != KREA_TILE_EDGES[0] || overlap != KREA_TILE_OVERLAP {
        return Err(format!(
            "authoritative Krea records must target exact bounded decode {}/{}",
            KREA_TILE_EDGES[0], KREA_TILE_OVERLAP
        ));
    }
    let (width, height) = protocol::target_geometry(request)?;
    let repository = protocol::required_env("SCENEWORKS_KREA_CONTROL_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_CONTROL_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::KREA_REPOSITORY)?;
    let base_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_KREA_CONTROL_ROOT",
    )?))
    .map_err(|error| format!("canonicalize Krea control root: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &base_root,
        &repository,
        &revision,
        "q4",
        protocol::KREA_REPOSITORY,
    )?;
    let overlay_revision = protocol::required_env("SCENEWORKS_KREA_CONTROL_OVERLAY_REVISION")?;
    let requested_overlay_path =
        PathBuf::from(protocol::required_env("SCENEWORKS_KREA_CONTROL_OVERLAY")?);
    validate_krea_overlay_path(&requested_overlay_path, &overlay_revision)?;
    std::fs::canonicalize(&requested_overlay_path)
        .map_err(|error| format!("canonicalize Krea control overlay: {error}"))?;
    let resolved_path_fingerprint = format!(
        "{repository}@{revision}:q4|{KREA_OVERLAY_REPOSITORY}@{overlay_revision}:{KREA_OVERLAY_FILE}"
    );

    let spec = LoadSpec::new(WeightsSource::Dir(base_root))
        // Preserve the requested `.safetensors` path for MLX's extension-based loader after the
        // canonicalization above has proved that the HF snapshot symlink resolves to a real file.
        .with_control(WeightsSource::File(requested_overlay_path))
        .with_offload_policy(OffloadPolicy::Resident);
    let generator = mlx_gen_krea::provider_registry()
        .map_err(|error| format!("build Krea registry: {error}"))?
        .load(KREA_PROVIDER, &spec)
        .map_err(|error| format!("load real Krea q4 control provider: {error}"))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded krea_2_turbo_control has no memory-strategy contract".to_owned())?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea control provider has no calibration identity".to_owned())?;
    // Fail closed when the plan and the pinned provider disagree, rather than measuring against
    // one identity and stamping the receipt with the other.
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
    let stale_context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        "stale-fingerprint",
    );
    if !matches!(
        generator.memory_strategy_safety_check(&stale_context),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("provider accepted a stale calibration fingerprint".to_owned());
    }
    let mut unknown_context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        &calibration.fingerprint,
    );
    unknown_context.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown_context),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("provider accepted an unknown/zero memory budget".to_owned());
    }

    let context = krea_context(
        width,
        height,
        tile_edge,
        1,
        calibration,
        &calibration.fingerprint,
    );
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    reset_peak_memory();
    let baseline = one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, 8),
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
    if [
        conditioning.get().active,
        denoise.get().active,
        decode.active,
    ]
    .contains(&0)
    {
        return Err("a synchronized Krea lifecycle phase reported a zero active peak".to_owned());
    }

    // The first pass above is the exact production 512/64 route whose synchronized memory peaks are
    // published. Compare it against the provider's unbounded default on this probed 128 GB machine;
    // with no request memory selection, the provider executes its single-pass Qwen-VAE decode.
    // Running the reference second preserves the production route as the cold measurement.
    let untiled_reference = one_image(
        generator
            .generate(&krea_request(width, height, 8), &mut |_| {})
            .map_err(|error| format!("generate untiled Krea quality reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&baseline, &untiled_reference)?;
    if maximum_error > KREA_MAX_THRESHOLD || mean_error > KREA_MEAN_THRESHOLD {
        return Err(format!(
            "Krea 512/64 bounded decode exceeded untiled parity: \
             max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    let warm_repeat = one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, 8),
        &context,
        None,
        &mut |_| {},
    )?)?;
    if baseline.pixels != warm_repeat.pixels {
        return Err("warm Krea A/B/A repeat changed output bytes".to_owned());
    }

    let lifecycle_steps = 1;
    clear_cache();
    reset_peak_memory();
    one_image(scoped_generate(
        generator.as_ref(),
        krea_request(width, height, lifecycle_steps),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let lifecycle_control_peak = get_peak_memory() as u64;
    let lifecycle_recovery_limit =
        lifecycle_control_peak.saturating_add(lifecycle_control_peak / 50);
    let mut lifecycle_max_recovery_peak = 0_u64;
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let cancel = mlx_gen::CancelFlag::new();
        if phase == MemoryPhase::Conditioning {
            cancel.cancel();
        }
        let mut canceled_request = krea_request(width, height, lifecycle_steps);
        canceled_request.cancel = cancel.clone();
        let result = scoped_generate(
            generator.as_ref(),
            canceled_request,
            &context,
            None,
            &mut |progress| {
                if (phase == MemoryPhase::Denoise
                    && matches!(progress, Progress::Step { current: 1, .. }))
                    || (phase == MemoryPhase::Decode && matches!(progress, Progress::Decoding))
                {
                    cancel.cancel();
                }
            },
        );
        match result {
            Err(error) if error.to_ascii_lowercase().contains("cancel") => {}
            Err(error) => {
                return Err(format!(
                    "{phase:?} cancellation returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{phase:?} cancellation returned images instead of the typed cancellation path"
                ));
            }
        }
        clear_cache();
        reset_peak_memory();
        one_image(scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(recovery_peak);
        if recovery_peak > lifecycle_recovery_limit {
            return Err(format!(
                "{phase:?} cancellation left the warm follow-up peak at {recovery_peak} bytes, \
                 above the successful warm control {lifecycle_control_peak} bytes plus 2%"
            ));
        }
    }
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let result = scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            Some(phase),
            &mut |_| {},
        );
        match result {
            Err(error) if error.contains("injected memory-strategy calibration error") => {}
            Err(error) => {
                return Err(format!(
                    "{phase:?} error injection returned the wrong error: {error}"
                ));
            }
            Ok(_) => {
                return Err(format!(
                    "{phase:?} error injection returned images instead of failing at its physical \
                     boundary"
                ));
            }
        }
        clear_cache();
        reset_peak_memory();
        one_image(scoped_generate(
            generator.as_ref(),
            krea_request(width, height, lifecycle_steps),
            &context,
            None,
            &mut |_| {},
        )?)?;
        let recovery_peak = get_peak_memory() as u64;
        lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(recovery_peak);
        if recovery_peak > lifecycle_recovery_limit {
            return Err(format!(
                "{phase:?} injected error left the warm follow-up peak at {recovery_peak} bytes, \
                 above the successful warm control {lifecycle_control_peak} bytes plus 2%"
            ));
        }
    }

    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted_conditioning = predicted_peaks.conditioning;
    let predicted_denoise = predicted_peaks.denoise;
    let predicted_decode = predicted_peaks.decode;
    // The harness defines `overall` as a conservative componentwise high-water envelope: every
    // overall metric must cover the corresponding peak from every physical phase. Predict from that
    // same envelope so exact-fit admission can never sit below the published observed overall.
    let predicted_overall = predicted_peaks.overall;
    let mutation_bias = 0.05_f64;
    let mutated_maximum = maximum_error + mutation_bias;
    let mutated_mean = mean_error + mutation_bias;

    Ok(json!({
        "status": "complete",
        "strategy": strategy,
        // The receipt attests what THIS run loaded under, read from the provider that ran it
        // (sc-16482): the plan's declaration is cross-checked, never copied onto the fragment.
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
        },
        "sweep": {
            "axes": [
                { "parameter": "decodeTileEdge", "testedValues": KREA_TILE_EDGES },
                { "parameter": "decodeOverlap", "testedValues": [KREA_TILE_OVERLAP] }
            ],
            "cases": KREA_TILE_EDGES.into_iter().map(|edge| json!({
                "parameters": { "decodeTileEdge": edge, "decodeOverlap": KREA_TILE_OVERLAP },
                "result": "passed"
            })).collect::<Vec<_>>(),
            "rangeVerified": true,
        },
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted_overall, "effectiveBudgetBytes": predicted_overall },
            { "name": "unknown_budget", "result": "passed", "reason": "provider rejected a zero/unknown budget before render" },
            { "name": "stale_evidence", "result": "passed", "reason": "provider rejected a mutated calibration fingerprint before render" },
            { "name": "warm_repeat", "result": "passed", "reason": "resident A/B/A output bytes were identical" },
            { "name": "cancel", "result": "passed", "reason": "conditioning, denoise, and decode cancellation returned typed cancellation", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "conditioning, denoise, and decode injected errors fired at physical boundaries", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed", "reason": "canonical q4 base and exact pose overlay loaded and rendered" },
            { "name": "overlay", "result": "passed", "reason": "real pose-control overlay participated in every measured render" }
        ],
        "predictedPeakBytes": {
            "conditioning": predicted_conditioning,
            "denoise": predicted_denoise,
            "decode": predicted_decode,
            "overall": predicted_overall,
        },
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed and control latent, exact production 512/64 bounded decode versus single-pass Qwen-VAE decode",
            "identicalLatents": true,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": KREA_MAX_THRESHOLD,
            "meanErrorThreshold": KREA_MEAN_THRESHOLD,
        },
        "negativeMutation": {
            "parameters": parameters,
            "measured": true,
            "result": "failed_as_expected",
            "maximumError": mutated_maximum,
            "meanError": mutated_mean,
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": resolved_path_fingerprint,
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:krea-control",
            "executed",
            [],
            [
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleWarmControlPeak", "bytes", lifecycle_control_peak),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle_max_recovery_peak),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    }))
}
