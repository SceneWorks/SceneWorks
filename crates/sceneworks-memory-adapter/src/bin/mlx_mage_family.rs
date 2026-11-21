//! MLX mage capture owner.
use super::*;

pub(super) fn image_max_mean_abs(left: &Image, right: &Image) -> Result<(f64, f64), String> {
    if (left.width, left.height) != (right.width, right.height)
        || left.pixels.len() != right.pixels.len()
        || left.pixels.is_empty()
    {
        return Err(format!(
            "image shape mismatch: {}x{} ({} bytes) versus {}x{} ({} bytes)",
            left.width,
            left.height,
            left.pixels.len(),
            right.width,
            right.height,
            right.pixels.len()
        ));
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
        let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
        maximum = maximum.max(difference);
        sum += difference;
    }
    Ok((maximum, sum / left.pixels.len() as f64))
}

pub(super) fn persist_physical_mlx_image(
    output_dir: &Path,
    source_prefix: &str,
    logical_case_id: &str,
    role: &str,
    image: &Image,
) -> Result<Value, String> {
    let content_sha256 = format!("{:x}", Sha256::digest(&image.pixels));
    let file_name = format!(
        "{logical_case_id}-{role}-{}x{}-{content_sha256}.rgb",
        image.width, image.height,
    );
    let local_path = output_dir.join(&file_name);
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&local_path)
    {
        Ok(mut file) => file
            .write_all(&image.pixels)
            .map_err(|error| format!("write physical MLX {role} output: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(&local_path).map_err(|read_error| {
                format!("read existing physical MLX {role} output: {read_error}")
            })?;
            if existing != image.pixels {
                return Err(format!(
                    "content-addressed physical MLX {role} output already exists with different bytes"
                ));
            }
        }
        Err(error) => {
            return Err(format!(
                "create immutable physical MLX {role} output: {error}"
            ));
        }
    }
    Ok(json!({
        "role": role,
        "path": format!("{source_prefix}/{file_name}"),
        "localPath": local_path,
        "sha256": content_sha256,
        "bytes": image.pixels.len(),
    }))
}

pub(super) fn z_image_quality_passes(maximum: f64, mean: f64) -> bool {
    maximum <= Z_IMAGE_MAX_THRESHOLD && mean <= Z_IMAGE_MEAN_THRESHOLD
}

pub(super) fn fixed_pose_control_image(width: u32, height: u32) -> Image {
    let mut pixels = vec![0_u8; width as usize * height as usize * 3];
    let mut line = |start: (i32, i32), end: (i32, i32), color: [u8; 3]| {
        let (mut x, mut y) = start;
        let dx = (end.0 - start.0).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let dy = -(end.1 - start.1).abs();
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            for offset_y in -2..=2 {
                for offset_x in -2..=2 {
                    let px = x + offset_x;
                    let py = y + offset_y;
                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let index = (py as usize * width as usize + px as usize) * 3;
                        pixels[index..index + 3].copy_from_slice(&color);
                    }
                }
            }
            if (x, y) == end {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += sx;
            }
            if doubled <= dx {
                error += dx;
                y += sy;
            }
        }
    };
    // A deterministic whole-body stick pose: head/neck, shoulders, elbows, wrists, hips, knees,
    // and ankles. Using a pose-shaped control fixture exercises the real pose branch rather than
    // merely setting the `ControlKind::Pose` enum on arbitrary pixels.
    let scale = |x: u32, y: u32| ((x * width / 512) as i32, (y * height / 512) as i32);
    for (start, end, color) in [
        (scale(256, 82), scale(256, 138), [255, 255, 255]),
        (scale(190, 158), scale(322, 158), [255, 128, 0]),
        (scale(256, 138), scale(256, 296), [255, 255, 0]),
        (scale(190, 158), scale(145, 238), [0, 255, 0]),
        (scale(145, 238), scale(116, 320), [0, 255, 255]),
        (scale(322, 158), scale(370, 225), [0, 128, 255]),
        (scale(370, 225), scale(404, 292), [0, 0, 255]),
        (scale(214, 296), scale(298, 296), [255, 0, 255]),
        (scale(214, 296), scale(194, 396), [255, 0, 0]),
        (scale(194, 396), scale(176, 482), [128, 0, 255]),
        (scale(298, 296), scale(330, 390), [255, 64, 128]),
        (scale(330, 390), scale(366, 472), [128, 255, 0]),
    ] {
        line(start, end, color);
    }
    Image {
        width,
        height,
        pixels,
    }
}

pub(super) fn one_image(output: GenerationOutput) -> Result<Image, String> {
    match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => Ok(images.remove(0)),
        other => Err(format!("expected one Krea image, got {other:?}")),
    }
}

pub(super) fn image_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> PredictedPhasePeaks {
    predicted_phase_peaks(conditioning, denoise, decode, IMAGE_PREDICTED_PEAK_BASIS)
}

/// Which family member the plan asks for. Refuses by name: a `(provider, mode)` pair no member
/// serves (the base has no edit route, `image_to_image` is not planned anywhere) must not be
/// measured as its nearest neighbour.
pub(super) fn z_image_arm(request: &Value) -> Result<ZImageArm, String> {
    let planned = protocol::planned(request)?;
    let provider = protocol::plan_str(planned, "/target/provider")?;
    let mode = protocol::plan_str(planned, "/target/mode")?;
    match (provider, mode) {
        (Z_IMAGE_PROVIDER, "text_to_image") => Ok(Z_IMAGE_TURBO_ARM),
        (Z_IMAGE_PROVIDER, "edit_image") => Ok(Z_IMAGE_EDIT_ARM),
        (Z_IMAGE_BASE_PROVIDER, "text_to_image") => Ok(Z_IMAGE_BASE_ARM),
        (provider, mode) => Err(format!(
            "the MLX Z-Image arm does not implement provider {provider:?} in mode {mode:?}"
        )),
    }
}

/// The env-free half of [`z_image_load_spec`], so the tier binding is unit-testable: the root must
/// end in the PLANNED tier's directory (`.../snapshots/<revision>/<tier>`), so a stale `…/q4`
/// export can never satisfy a q8 or bf16 plan and quietly re-label another tier's peaks — the
/// sc-17097 defect class the Candle arm closed, which this arm carried as a literal `"q4"` until
/// sc-22724.
pub(super) fn z_image_load_spec_at(
    request: &Value,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<ZImageArtifact, String> {
    let arm = z_image_arm(request)?;
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
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    Ok(ZImageArtifact {
        arm,
        repository,
        revision,
        tier,
        spec,
    })
}

pub(super) fn z_image_load_spec(
    request: &Value,
    load_shape: LoadShape,
) -> Result<ZImageArtifact, String> {
    let arm = z_image_arm(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = PathBuf::from(protocol::required_env(arm.root_env)?);
    z_image_load_spec_at(request, load_shape, repository, revision, root)
}

pub(super) fn load_z_image_generator(
    request: &Value,
    load_shape: LoadShape,
) -> Result<(ZImageArtifact, Box<dyn Generator>), String> {
    let artifact = z_image_load_spec(request, load_shape)?;
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(artifact.arm.provider, &artifact.spec)
        .map_err(|error| {
            format!(
                "load real {} {} provider: {error}",
                artifact.arm.provider, artifact.tier
            )
        })?;
    Ok((artifact, generator))
}

pub(super) fn z_image_request(arm: ZImageArm, width: u32, height: u32) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(16402),
        // The first Step callback closes the conservative conditioning envelope; the second
        // step then supplies a real denoise-only interval before Decoding.
        steps: Some(2),
        ..Default::default()
    };
    if arm.edit {
        // The worker's edit route: one `Conditioning::Reference` fitted to the request geometry
        // plus the strength lever (`resolve_zimage_edit_init`). The start step is derived from
        // `steps * strength` by the engine, so the step count is raised to keep two executed
        // denoise steps behind the conditioning boundary.
        request.steps = Some(Z_IMAGE_EDIT_STEPS);
        // `request.strength` stays None: the worker sets ONLY the per-reference strength
        // (`build_lane_conditioning`, image_jobs/base.rs:7136) and leaves the request-level lever —
        // gen-core's documented fallback for a single `Reference` with no strength of its own —
        // unset. This arm reproduces the worker's request shape, so it does the same (sc-22724).
        request.conditioning = vec![Conditioning::Reference {
            image: Image {
                width,
                height,
                pixels: protocol::synthetic_reference_rgb(width, height),
            },
            strength: Some(Z_IMAGE_EDIT_STRENGTH),
        }];
    }
    request
}

pub(super) fn z_image_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // Each plan row is one exact production parameter tuple. Marking this exact tuple's range
    // verified does not promote sibling tuples: the generated matrix still requires a matching
    // manifest calibration binding for each cell.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

pub(super) fn run_z_image_reference_loaded(
    request: &Value,
    generator: &dyn Generator,
    artifact: &ZImageArtifact,
    load_shape: LoadShape,
) -> Result<Value, String> {
    let arm = artifact.arm;
    let (repository, revision) = (artifact.repository.as_str(), artifact.revision.as_str());
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let (width, height) = protocol::target_geometry(request)?;
    let selection = planned_selection(request)?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {} has no memory-strategy contract", arm.provider))?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned Z-Image provider rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Z-Image provider has no calibration identity".to_owned())?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        // The mode the plan declared, as the worker would admit it: an edit request carries one
        // reference and is admitted under `MemoryMode::Edit` (the contract the `z_image_edit`
        // manifest entry declares for the Turbo provider).
        mode: if arm.edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        },
        has_reference: arm.edit,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(arm.edit),
        },
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-15510@{}", protocol::INFERENCE_PIN),
    };
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    // sc-22667 (epic 22657 D3): Z-Image's residency is REQUEST-SCOPED even under eager
    // materialization — each component is materialized the first time a request reaches its
    // phase and retained afterwards — so a window opened on the freshly loaded generator measured
    // a cold first request whose conditioning phase saw only the text encoder it was materializing
    // (2.27 GB against a 5.83 GB resident set), which the core anchor law refuses to decompose.
    // One unmeasured request on the same scope materializes the whole resident set FIRST; the
    // window then opens above it, the way the candle adapter measures every phase above weights
    // already on device. `protocol::open_resident_phase_window` fixes that order and refuses a
    // resident set the counters did not see.
    let opening = protocol::open_resident_phase_window(&mut MlxResidencyCounters, || {
        one_image(scoped_generate(
            generator,
            z_image_request(arm, width, height),
            &context,
            None,
            &mut |_| {},
        )?)
        .map(|_| ())
    })?;
    let pre_rung_active = opening.resident_active;
    let pre_rung_cache = opening.resident_cache;
    let peak_after_reset = opening.peak_after_reset;
    let selected = one_image(scoped_generate(
        generator,
        z_image_request(arm, width, height),
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
            "a synchronized Z-Image lifecycle phase reported a zero active peak".to_owned(),
        );
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
        return Err("Z-Image provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Z-Image provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-z-image-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Z-Image provider accepted stale calibration evidence".to_owned());
    }

    let baseline = one_image(
        generator
            .generate(&z_image_request(arm, width, height), &mut |_| {})
            .map_err(|error| format!("generate unselected Z-Image reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !z_image_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Z-Image selected rung exceeded unselected parity: max={maximum_error:.6}, mean={mean_error:.6}"
        ));
    }
    // Establish the cleanup oracle on this exact loaded provider before injecting either fault.
    // `clear_cache` is the same explicit retained-buffer release used by the production stream and
    // the pinned Krea lifecycle harness; live model weights remain active, so the post-cleanup
    // snapshot still catches fault-owned arrays that escaped their request scope.
    clear_cache();
    reset_peak_memory();
    let warm = one_image(scoped_generate(
        generator,
        z_image_request(arm, width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let lifecycle_clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let lifecycle_clean_post_cleanup = AllocatorState::capture_current();
    let lifecycle_bounds = LifecycleMemoryBounds::from_clean_warm(
        lifecycle_clean_warm_peak,
        lifecycle_clean_post_cleanup,
    );
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !z_image_quality_passes(warm_maximum, warm_mean) {
        return Err("Z-Image warm repeat changed the deterministic output".to_owned());
    }

    let mut lifecycle_max_fault_active = 0_u64;
    let mut lifecycle_max_fault_cache = 0_u64;
    let mut lifecycle_max_recovery_active = 0_u64;
    let mut lifecycle_max_recovery_cache = 0_u64;
    let mut lifecycle_max_recovery_peak = 0_u64;

    let cancelled = z_image_request(arm, width, height);
    let cancel_signal = cancelled.cancel.clone();
    let cancel_during_decode = selection.strategy == MemoryStrategy::BoundedDecode;
    let mut cancel_triggered = false;
    let cancel_error = scoped_generate(generator, cancelled, &context, None, &mut |progress| {
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
    })
    .expect_err("in-flight Z-Image cancellation must fail");
    if !cancel_triggered {
        return Err("Z-Image cancellation probe never reached the active rung boundary".to_owned());
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Z-Image cancellation returned the wrong error: {cancel_error}"
        ));
    }
    clear_cache();
    let cancel_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_fault_active = lifecycle_max_fault_active.max(cancel_post_cleanup.active);
    lifecycle_max_fault_cache = lifecycle_max_fault_cache.max(cancel_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(cancel_post_cleanup) {
        return Err(format!(
            "Z-Image cancellation retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            cancel_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    let cancel_recovery = one_image(scoped_generate(
        generator,
        z_image_request(arm, width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let cancel_recovery_peak = get_peak_memory() as u64;
    lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(cancel_recovery_peak);
    if !lifecycle_bounds.allows_warm_peak(cancel_recovery_peak) {
        return Err(format!(
            "Z-Image cancellation left the warm follow-up peak at {cancel_recovery_peak} bytes, above the clean warm control {lifecycle_clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let cancel_recovery_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_recovery_active =
        lifecycle_max_recovery_active.max(cancel_recovery_post_cleanup.active);
    lifecycle_max_recovery_cache =
        lifecycle_max_recovery_cache.max(cancel_recovery_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(cancel_recovery_post_cleanup) {
        return Err(format!(
            "Z-Image cancellation warm follow-up retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            cancel_recovery_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !z_image_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Z-Image cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected_phase = if selection.strategy == MemoryStrategy::BoundedDecode {
        MemoryPhase::Decode
    } else {
        MemoryPhase::Denoise
    };
    let injected = scoped_generate(
        generator,
        z_image_request(arm, width, height),
        &context,
        Some(injected_phase),
        &mut |_| {},
    )
    .expect_err("injected Z-Image error must fail");
    if !injected.contains("injected memory-strategy calibration error") {
        return Err(format!(
            "Z-Image error injection returned the wrong error: {injected}"
        ));
    }
    clear_cache();
    let error_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_fault_active = lifecycle_max_fault_active.max(error_post_cleanup.active);
    lifecycle_max_fault_cache = lifecycle_max_fault_cache.max(error_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(error_post_cleanup) {
        return Err(format!(
            "Z-Image injected error retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            error_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    let error_recovery = one_image(scoped_generate(
        generator,
        z_image_request(arm, width, height),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let error_recovery_peak = get_peak_memory() as u64;
    lifecycle_max_recovery_peak = lifecycle_max_recovery_peak.max(error_recovery_peak);
    if !lifecycle_bounds.allows_warm_peak(error_recovery_peak) {
        return Err(format!(
            "Z-Image injected error left the warm follow-up peak at {error_recovery_peak} bytes, above the clean warm control {lifecycle_clean_warm_peak} bytes plus 2%"
        ));
    }
    clear_cache();
    let error_recovery_post_cleanup = AllocatorState::capture_current();
    lifecycle_max_recovery_active =
        lifecycle_max_recovery_active.max(error_recovery_post_cleanup.active);
    lifecycle_max_recovery_cache =
        lifecycle_max_recovery_cache.max(error_recovery_post_cleanup.cache);
    if !lifecycle_bounds.allows_retained(error_recovery_post_cleanup) {
        return Err(format!(
            "Z-Image injected-error warm follow-up retained active/cache bytes {:?} above the clean warm cleanup {:?} plus {} bytes",
            error_recovery_post_cleanup,
            lifecycle_clean_post_cleanup,
            lifecycle_bounds.tolerance_bytes,
        ));
    }
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !z_image_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Z-Image error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if z_image_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Z-Image output mutation did not breach the production parity envelope".to_owned(),
        );
    }

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": artifact.tier,
        },
        "sweep": z_image_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "passed" },
            { "name": "cancel", "result": "passed", "reason": "post-cancel active/cache retention and the warm follow-up peak remained within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "post-error active/cache retention and the warm follow-up peak remained within the clean-warm control plus 2%", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "the authoritative Z-Image target has no overlay" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed, conditioning, sampling, precision, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": Z_IMAGE_MAX_THRESHOLD,
            "meanErrorThreshold": Z_IMAGE_MEAN_THRESHOLD,
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
            "resolvedPathFingerprint": artifact.loadability_fingerprint(),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-shared-ladder", arm.slug),
            "executed",
            [],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("peakAfterReset", "bytes", peak_after_reset),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("lifecycleCleanWarmPeak", "bytes", lifecycle_clean_warm_peak),
                ("lifecycleCleanPostCleanupActive", "bytes", lifecycle_clean_post_cleanup.active),
                ("lifecycleCleanPostCleanupCache", "bytes", lifecycle_clean_post_cleanup.cache),
                ("lifecycleCleanupTolerance", "bytes", lifecycle_bounds.tolerance_bytes),
                ("lifecycleMaximumFaultPostCleanupActive", "bytes", lifecycle_max_fault_active),
                ("lifecycleMaximumFaultPostCleanupCache", "bytes", lifecycle_max_fault_cache),
                ("lifecycleMaximumRecoveryPeak", "bytes", lifecycle_max_recovery_peak),
                ("lifecycleMaximumRecoveryPostCleanupActive", "bytes", lifecycle_max_recovery_active),
                ("lifecycleMaximumRecoveryPostCleanupCache", "bytes", lifecycle_max_recovery_cache),
                ("loadShapeDeferred", "count", u64::from(load_shape == LoadShape::DeferredMaterialization)),
                // The window opened ABOVE the materialized resident set (sc-22667 D3); a record
                // without this measurement was captured on a cold first request.
                ("residentSetMaterializedBeforeWindow", "count", 1),
                // sc-22724: the edit arm conditions every request on one reference image.
                ("referenceImages", "count", u64::from(arm.edit)),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}

pub(super) fn run_z_image_reference(request: &Value) -> Result<Value, String> {
    // Before the load, not inside `..._loaded`: a non-still target must be refused without paying
    // for weights, the same ordering every other image arm now has. The arm is resolved first so
    // the refusal carries the member's own label.
    let arm = z_image_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let load_shape = planned_load_shape(request)?;
    let (artifact, generator) = load_z_image_generator(request, load_shape)?;
    run_z_image_reference_loaded(request, generator.as_ref(), &artifact, load_shape)
}

/// Maximum, mean, and root-mean-square absolute error between two images, in [0,1] units. The
/// runtime_complete quality shape requires the RMS metric alongside max/mean
/// (`memory-calibration-harness.mjs#validateRuntimeComplete`), which is why this exists next to
/// `image_max_mean_abs` instead of replacing it.
pub(super) fn image_max_mean_rms_abs(
    left: &Image,
    right: &Image,
) -> Result<(f64, f64, f64), String> {
    let (maximum, mean) = image_max_mean_abs(left, right)?;
    let mut sum_squares = 0.0_f64;
    for (&left, &right) in left.pixels.iter().zip(&right.pixels) {
        let difference = (f64::from(left) - f64::from(right)).abs() / 255.0;
        sum_squares += difference * difference;
    }
    Ok((
        maximum,
        mean,
        (sum_squares / left.pixels.len() as f64).sqrt(),
    ))
}

pub(super) fn validate_z_image_batch(request: &Value) -> Result<&[Value], String> {
    let planned = request
        .get("planned")
        .and_then(Value::as_array)
        .ok_or_else(|| "assess_batch request.planned must be an array".to_owned())?;
    let expected = [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
    ];
    // sc-18104: refuse a foreign provider by name FIRST, for the same reason `run` does. This batch
    // path is Z-Image-only — `assess_z_image_batch` hardcodes `Z_IMAGE_PROVIDER` when it reads the
    // memory-strategy contract — but nothing below inspects `target.provider`, so without this check
    // a batch for another provider is MISROUTED into the Z-Image contract and dies on a
    // Z-Image-shaped fingerprint complaint, after `runtime_macos::catalog()` has already done real
    // environment work. That is the same silent-misroute class the `run` refusal closes, and it is
    // reachable: `assessProviderReuse` (scripts/memory-calibration-harness.mjs) selects candidates by
    // backend and optional fixture only, never by provider.
    //
    // This runs BEFORE the length check deliberately. A foreign batch of the wrong length is still
    // stopped safely there, but it would be told `Z-Image rung batch must contain exactly 5 cases` —
    // a Z-Image-named complaint about a provider that is not Z-Image, the same misleading-diagnostic
    // problem in miniature. That is reachable for the very lane which motivated this fix:
    // `mlx-gen-flux2` marks every non-Resident strategy `Missing`, so an `assess-reuse` on a flux2
    // fixture submits a ONE-element batch. Refusing by name is therefore unconditional; only an empty
    // batch, which has no `planned[0]` to read a provider from, falls through to the length check.
    if let Some(provider) = planned
        .first()
        .and_then(|case| case.pointer("/target/provider"))
        .and_then(Value::as_str)
    {
        if provider != Z_IMAGE_PROVIDER && provider != Z_IMAGE_BASE_PROVIDER {
            return Err(format!(
                "MLX five-rung batch assessment does not implement provider {provider:?}"
            ));
        }
    }
    if planned.len() != expected.len() {
        return Err(format!(
            "Z-Image rung batch must contain exactly {} cases, got {}",
            expected.len(),
            planned.len()
        ));
    }
    let target = planned[0]
        .get("target")
        .ok_or_else(|| "assess_batch planned target must be present".to_owned())?;
    for (index, (item, expected_rung)) in planned.iter().zip(expected).enumerate() {
        let rung = item
            .pointer("/strategy/rung")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("assess_batch planned[{index}].strategy.rung must be a string")
            })?;
        if rung != expected_rung {
            return Err(format!(
                "Z-Image rung batch must use canonical order; index {index} is {rung:?}, expected {expected_rung:?}"
            ));
        }
        if item.get("target") != Some(target) {
            return Err("Z-Image rung batch must keep one exact target tuple".to_owned());
        }
    }
    Ok(planned)
}

pub(super) fn z_image_reuse_identity(fingerprint: &str, load_shape: LoadShape) -> String {
    format!("{fingerprint}@{}", load_shape_key(load_shape))
}

pub(super) fn assess_z_image_batch(request: &Value) -> Result<Value, String> {
    let planned = validate_z_image_batch(request)?;
    let mut representative = request.clone();
    representative["action"] = json!("run");
    representative["planned"] = planned[0].clone();
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let mut actual_fingerprints = Vec::new();
    let mut actual_identities = Vec::new();
    for load_shape in [
        LoadShape::EagerMaterialization,
        LoadShape::DeferredMaterialization,
    ] {
        let artifact = z_image_load_spec(&representative, load_shape)?;
        let provider = artifact.arm.provider;
        let contract = catalog
            .media()
            .memory_strategy_contract(provider, &artifact.spec)
            .map_err(|error| format!("read {provider} memory-strategy contract: {error}"))?
            .ok_or_else(|| format!("{provider} has no memory-strategy contract"))?;
        let calibration = contract
            .calibration
            .as_ref()
            .ok_or_else(|| "pinned Z-Image provider has no calibration identity".to_owned())?;
        actual_fingerprints.push(calibration.fingerprint.clone());
        actual_identities.push(z_image_reuse_identity(
            &calibration.fingerprint,
            calibration.load_shape,
        ));
    }
    for item in planned {
        let planned_fingerprint = item
            .get("calibrationFingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| "batch calibrationFingerprint must be a string".to_owned())?;
        let expected = if item.pointer("/strategy/rung").and_then(Value::as_str)
            == Some("bounded_transformer_residency")
        {
            &actual_fingerprints[1]
        } else {
            &actual_fingerprints[0]
        };
        if planned_fingerprint != expected {
            return Err(format!(
                "plan/provider calibration mismatch in reuse assessment: plan={planned_fingerprint}, pinned provider={expected}"
            ));
        }
    }
    actual_fingerprints.sort_unstable();
    actual_fingerprints.dedup();
    actual_identities.sort_unstable();
    actual_identities.dedup();
    if actual_identities.len() > 1 {
        return Ok(json!({
            "verdict": "unable_to_amortize",
            "reason": format!(
                "one MLX model load cannot preserve the distinct calibrated load-shape identities required by the five rungs: {}",
                actual_identities.join(", ")
            ),
            "calibrationFingerprints": actual_fingerprints,
            "calibrationIdentities": actual_identities,
            "evidence": "pinned provider contracts for eager and deferred load specs",
        }));
    }
    Ok(json!({
        "verdict": "eligible_for_measurement",
        "reason": "all MLX rungs share one calibrated load-shape identity",
        "calibrationFingerprints": actual_fingerprints,
        "calibrationIdentities": actual_identities,
        "evidence": "pinned provider contracts for eager and deferred load specs",
    }))
}

pub(super) fn mage_quality_passes(maximum: f64, mean: f64) -> bool {
    maximum <= MAGE_MAX_THRESHOLD && mean <= MAGE_MEAN_THRESHOLD
}

/// The production calibration identity the loaded MLX Mage generator publishes for one
/// `(member, tier)` cell — the table `mlx-gen-mage::model::production_calibration_fingerprint`
/// mints (inference PR 953): `mage-flow-<route>-<tier>-mlx-shared-ladder-v1`, eighteen distinct
/// strings. The engine binds one only when the tier is PROVEN off the `quantization.bits` marker of
/// the component directories the loader itself opened (DiT and text encoder agreeing) and equals
/// the request knob; a dense snapshot requantized at load withholds it, which the capture reports
/// as a plan/provider mismatch rather than measuring under a borrowed name. Written here as well so
/// the plan/arm binding is weights-free and holds at inference `c6d6a4db`, whose engine still
/// publishes the retired single string; the capture refuses a loaded contract whose identity
/// differs from the plan, so the two copies cannot drift unnoticed once the epic's pin bump lands.
pub(super) fn mage_calibration_fingerprint(arm: MageArm, tier: &str) -> String {
    format!("mage-flow-{}-{tier}-mlx-shared-ladder-v1", arm.slug)
}

/// The weights-free conformance identities `mlx-gen-mage` publishes for a registry surface that
/// loaded no weights (`model.rs` `STATIC_BEHAVIOR_FINGERPRINT` = `mage-flow-mlx-registry-behavior-v1`,
/// suffixed `-<route>-<tier>`), and the retired pre-sc-22733 single string. A plan row naming one
/// of these could never be satisfied by a production load. Test-only: the sole caller is the
/// plan-identity conformance test in `mage_tests`.
#[cfg(test)]
pub(super) fn is_mage_weights_free_fingerprint(fingerprint: &str) -> bool {
    fingerprint.starts_with("mage-flow-mlx-registry-behavior-v1")
        || fingerprint == "mage-flow-mlx-shared-ladder-2026-08-03-v1"
}

/// The admission budget the pinned Mage provider requires a caller to present, as
/// `(required_total_peak_bytes, resident_credit_bytes)`.
///
/// Mage is the only MLX provider that cross-checks the caller's declared peak against its OWN fit
/// model (`model.rs` `request_context_error`): whatever the caller does NOT declare in
/// `predicted_peak_bytes` is credited as already-resident, and that credit must fit both the
/// contract's resident envelope and the budget's committed snapshot. The placeholder peak of `1`
/// every other arm passes therefore claims the entire model as resident credit against a committed
/// snapshot of zero, and is refused before a single step runs. So the capture declares the pair the
/// worker declares (`mlx_fit_gate.rs`: committed bytes are the live resident snapshot, the
/// predicted peak is the request's incremental demand above them) and the engine's own weights-free
/// test builds verbatim (`model.rs`
/// `resident_safety_recomputes_peak_and_binds_calibration_identity`: committed = the contract's
/// resident bytes, predicted = the modelled total minus them).
///
/// The total comes from the provider's own `memory::generation_peak_gb` rather than a copy of its
/// measured anchors, so a re-measured tier in a later pin moves this with it.
pub(super) fn mage_admission_budget(
    contract: &MemoryProviderContract,
    tier: Option<Quant>,
    width: u32,
    height: u32,
) -> (u64, u64) {
    let required_total_peak_bytes =
        ((mlx_gen_mage::memory::generation_peak_gb(tier, width, height, 1) * 1_000_000_000.0)
            .round() as u64)
            .saturating_add(contract.auxiliary_resident_bytes());
    (required_total_peak_bytes, contract.total_resident_bytes())
}

/// The FINDING a measured Mage-Flow peak above the engine's own modelled total files (sc-22738).
///
/// WHY THIS IS NOT A REFUSAL. It used to be one, and every `mage_flow*:*:mlx` anchor died on it
/// after a complete 11–38 s render: `measured Mage-Flow peak … exceeds the pinned provider's own
/// modelled total …`. That inverted the epic's premise. `mlx_gen_mage::memory::generation_peak_gb`
/// is a MODEL, not an observation — `generation_resident_gb(tier) + vae_peak_gb(w, h)`, where
/// `generation_resident_gb` is a single 512² anchor per tier with that anchor's VAE term subtracted
/// out and the ONLY geometry-dependent term in the whole estimator is the linear VAE **decode**
/// fit. So the model asserts that everything except the VAE decode is resolution-independent, and
/// it carries no term whatsoever for a reference image: the six family members share one scalar per
/// tier, edit routes included. Both assertions are false against measurement, which is precisely
/// what this capture exists to establish. Refusing here discards the only evidence that says so.
///
/// WHAT THE EXCEEDANCE MEANS FOR PRODUCTION. The worker admits a Mage request on that same model
/// (`mlx_fit_gate.rs` `request_total_peak_bytes`, whose `mage_flow` branch IS
/// `providers::mage::memory::generation_peak_gb`, against a budget from
/// `production_safe_budget_gb`). An under-prediction is therefore a live production hazard, not a
/// capture artifact, and the measured peak recorded here is the evidence admission should anchor
/// on instead (E5: measurement never gates; E4: the production path is what is measured).
///
/// The blocker states the pair AND the delta, plus the geometry, tier and route the pair belongs
/// to, because a bare "exceeded" sentence would not let a reader size the hazard or tell the
/// resolution-independence gap (`mage_flow` 768² bf16: ~0.6 GB) from the missing reference-image
/// term (`mage_flow_edit` 768² bf16: ~7.3 GB on the same tier and geometry).
pub(super) fn mage_model_shortfall_blocker(
    arm: MageArm,
    tier: &str,
    width: u32,
    height: u32,
    measured_peak_bytes: u64,
    modelled_total_bytes: u64,
) -> Option<String> {
    let delta = measured_peak_bytes.checked_sub(modelled_total_bytes)?;
    if delta == 0 {
        return None;
    }
    Some(format!(
        "the pinned {} provider's own fit model UNDER-PREDICTS this cell: measured peak \
         {measured_peak_bytes} bytes exceeds its modelled total {modelled_total_bytes} bytes by \
         {delta} bytes at {width}x{height} on the {} route at {tier}. Recorded as measured \
         evidence rather than discarding the capture: `mlx_gen_mage::memory::generation_peak_gb` \
         scales only its linear VAE decode term with geometry and carries no reference-image term \
         at all, and the worker admits this route on that same model \
         (`mlx_fit_gate.rs::request_total_peak_bytes`), so the measured peak is what production \
         admission should anchor on.",
        arm.provider,
        mage_mode(arm),
    ))
}

/// The peak-below-weights REFUSAL, which is a different claim from the shortfall finding above.
///
/// A render cannot peak below the weights it is holding: MLX active memory is absolute, and the
/// provider declares those weights itself as the resident credit half of
/// [`mage_admission_budget`]. A measured peak underneath them is not evidence about the engine's
/// fit model — it is proof this capture did not measure a loaded render (a drained or
/// mis-scoped peak counter, a generator that never materialized), and a record built from it would
/// publish an anchor far below the real floor. That stays a refusal.
pub(super) fn mage_peak_below_weights_error(
    arm: MageArm,
    tier: &str,
    measured_peak_bytes: u64,
    resident_weight_bytes: u64,
) -> Option<String> {
    (measured_peak_bytes < resident_weight_bytes).then(|| {
        format!(
            "measured {} {tier} peak {measured_peak_bytes} is BELOW the {resident_weight_bytes} \
             resident weight bytes the loaded provider declares; the capture did not measure a \
             loaded render",
            arm.provider
        )
    })
}

pub(super) fn validate_mage_plan_identity(
    request: &Value,
    arm: MageArm,
    tier: &str,
) -> Result<(), String> {
    protocol::validate_plan_identity(
        request,
        arm.provider,
        tier,
        &mage_calibration_fingerprint(arm, tier),
    )
}

/// Which family member the plan asks for. Refuses by name: the six checkpoints are
/// architecturally identical (byte-identical configs and tensor schemas — `model.rs`
/// `verify_checkpoint_identity` exists precisely because of that), so nothing downstream of a
/// mis-keyed plan row could notice that Turbo's peaks had been recorded as Base's.
pub(super) fn mage_arm(request: &Value) -> Result<MageArm, String> {
    let planned = protocol::planned(request)?;
    let provider = protocol::plan_str(planned, "/target/provider")?;
    let model_id = protocol::plan_str(planned, "/target/modelId")?;
    MAGE_ARMS
        .into_iter()
        .find(|arm| arm.provider == provider && arm.provider == model_id)
        .ok_or_else(|| {
            format!(
                "the MLX Mage-Flow arm does not implement provider {provider:?} for model \
                 {model_id:?}"
            )
        })
}

/// The seed and step count this member's fixture binds, checked against the arm's own prefix, the
/// planned tier, the rendered edge and the recipe's step count.
pub(super) fn planned_mage_seed(request: &Value, arm: MageArm, tier: &str) -> Result<u64, String> {
    let fixture = protocol::planned_fixture(request)?;
    let (width, _) = protocol::target_geometry(request)?;
    let prefix = format!("{}-{tier}-{width}-seed", arm.fixture_prefix);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Mage-Flow fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Mage-Flow fixture step count {steps:?}: {error}"))?;
    if steps != arm.steps {
        return Err(format!(
            "planned.fixture {fixture:?} must use this arm's {}-step calibration request",
            arm.steps
        ));
    }
    Ok(seed)
}

pub(super) fn mage_mode(arm: MageArm) -> &'static str {
    protocol::mage_mode(arm.edit)
}

/// The plan's declared mode must be the one this member renders.
///
/// The record's `mode` comes from the PLAN, while the reference this arm attaches and the
/// `MemoryMode` it admits under come from the ARM. Left unchecked, a row declaring `text_to_image`
/// for an edit member would emit a reference-conditioned render's peaks under a reference-free
/// label — and the pinned provider would not object, because the context it grades is built from the
/// arm and is internally consistent.
pub(super) fn validate_mage_mode(request: &Value, arm: MageArm) -> Result<(), String> {
    let declared = protocol::planned_mode(request)?;
    let expected = mage_mode(arm);
    if declared != expected {
        return Err(format!(
            "{} renders {expected:?}, but the plan declares mode {declared:?}",
            arm.provider
        ));
    }
    Ok(())
}

pub(super) fn mage_artifact_source(arm: MageArm) -> Result<MageArtifactSource, String> {
    Ok(MageArtifactSource {
        repository: protocol::required_env(arm.repository_env)?,
        revision: protocol::required_env(arm.revision_env)?,
        root: PathBuf::from(protocol::required_env(arm.root_env)?),
        components_repository: protocol::required_env(
            "SCENEWORKS_MAGE_FLOW_COMPONENTS_REPOSITORY",
        )?,
        components_revision: protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REVISION")?,
        components_root: PathBuf::from(protocol::required_env(
            "SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT",
        )?),
    })
}

/// The env-free half of the Mage load, so every load-time binding is unit-testable:
///
/// * the variant root must end in `<repository>/snapshots/<revision>/<PLANNED tier>`, so a stale
///   `…/q4` export cannot satisfy a q8 or bf16 plan and quietly re-label another tier's peaks;
/// * the components root must be the components snapshot ITSELF (it has no tier component of its
///   own at the root — the tier is the FIRST path element inside it), and both component dirs are
///   staged EXPLICITLY rather than left to `resolve_component_dirs`'s flat-layout fallback: the
///   variant rehost has no `text_encoder/` or `vae/` sibling at all, so the fallback would resolve
///   two directories that do not exist; and
/// * `spec.quantize` carries the planned tier. It packs nothing at load — every shipped Mage tier is
///   PREPACKED under `<snapshot>/<tier>/` and `pipeline::load_time_quant_bits` returns `None` — but
///   the provider's declaration surface requires the selector and the spec to agree
///   (`model.rs` `surface_selector_matches_spec`), so it is an assertion about the directory on
///   disk rather than a conversion request.
pub(super) fn mage_load_spec(
    arm: MageArm,
    tier: &str,
    source: &MageArtifactSource,
    selection: &MemorySelection,
    offload: OffloadPolicy,
    load_shape: LoadShape,
) -> Result<MageArtifact, String> {
    protocol::validate_artifact_identity(&source.repository, &source.revision, arm.repository)?;
    protocol::validate_artifact_identity(
        &source.components_repository,
        &source.components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root)
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &source.repository,
        &source.revision,
        tier,
        arm.repository,
    )?;
    let components_root = std::fs::canonicalize(&source.components_root)
        .map_err(|error| format!("canonicalize SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &components_root,
        &source.components_repository,
        &source.components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let tier_components = components_root.join(tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(offload)
        .with_load_shape(load_shape)
        .with_component(
            protocol::MAGE_COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_TEXT_ENCODER)),
        )
        .with_component(
            protocol::MAGE_COMPONENT_VAE,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_VAE)),
        );
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok(MageArtifact { root, spec })
}

/// The generation request one Mage capture renders: the worker's request shape for this member.
///
/// The edit members REFUSE a reference-free request (`model.rs` `edit_references`), so the single
/// `Conditioning::Reference` is what makes an edit capture measure the multimodal conditioning path
/// and the encode half of the CoD autoencoder (`assemble` selects `VaePart::Both` on an edit
/// variant) rather than text-to-image under an edit label. The reference is fitted to the target
/// geometry, exactly as `image_jobs/base.rs` `fit_engine_image` fits it in production.
///
/// GUIDANCE AND THE NEGATIVE PROMPT ARE READ OFF THE LOADED DESCRIPTOR, NOT OFF THE TABLE (sc-22738).
/// The six `mage_flow*_turbo:*:mlx` cells were refused at request time with
/// `unsupported: mage_flow_turbo: guidance is not supported`, before any render: the shared floor
/// `gen-core/src/generator.rs:3216-3221` (`Capabilities::validate_request`) rejects
/// `req.guidance.is_some() && !self.supports_guidance`, and `mlx-gen-mage`
/// `src/model.rs:866-868` publishes `supports_negative_prompt`/`supports_guidance` as
/// `!variant.is_distilled()` — false on Turbo and EditTurbo (`model.rs:747-749`).
/// Production never sends either knob on such a route: `image_jobs/base.rs` `resolve_guidance`
/// (:4632) returns `None` when `!model.supports_guidance()` and `resolve_negative_prompt` (:4870)
/// returns `None` when `!model.supports_negative_prompt()`, and the generic MLX image path reads
/// both through them (`generate_stream`, :8609 and :8659). This arm asks the SAME question of the
/// SAME source — the descriptor the load resolved — so a capability the engine flips is followed
/// rather than restated, and `arm.guidance` supplies the scale only where one is accepted (the role
/// `model.default_guidance()` plays in production).
pub(super) fn mage_request(
    arm: MageArm,
    capabilities: &Capabilities,
    width: u32,
    height: u32,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: if arm.edit {
            MAGE_EDIT_PROMPT.to_owned()
        } else {
            MAGE_PROMPT.to_owned()
        },
        // The distilled members run with CFG genuinely off (`default_cfg` 1.0), at which the engine
        // builds no unconditional branch — a negative prompt there would be a claim about a branch
        // the render does not execute, and the same floor refuses it outright.
        negative_prompt: capabilities
            .supports_negative_prompt
            .then(|| "blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(arm.steps),
        guidance: capabilities.supports_guidance.then_some(arm.guidance),
        conditioning: if arm.edit {
            vec![Conditioning::Reference {
                image: Image {
                    width,
                    height,
                    pixels: protocol::synthetic_reference_rgb(width, height),
                },
                strength: None,
            }]
        } else {
            Vec::new()
        },
        ..Default::default()
    }
}

/// The `MemoryRunContext` the pinned Mage provider admits this member under.
///
/// `mode`, `has_reference` and `reference_count` are derived from the ARM, and the provider gates on
/// exactly those (`model.rs` `request_context_error`: the request mode must equal the variant's own
/// `MemoryMode`). `has_phases` is false on every Mage route — Mage has no multi-phase request shape
/// — and `overlay` is `None`, which is what a plain load carries and what the provider's
/// declaration surface publishes for it.
#[allow(clippy::too_many_arguments)]
pub(super) fn mage_run_context(
    arm: MageArm,
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    width: u32,
    height: u32,
    total_bytes: u64,
    predicted_peak_bytes: u64,
    committed_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        // From the LOADED provider's own identity, never the plan: the safety check compares abi,
        // fingerprint AND load shape.
        load_shape: calibration.load_shape,
        mode: if arm.edit {
            MemoryMode::Edit
        } else {
            MemoryMode::TextToImage
        },
        has_reference: arm.edit,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count: u32::from(arm.edit),
        },
        reference_resources: None,
        overlay: None,
        // The generator is already loaded when this context is built, so its resident bytes are
        // charged to the snapshot and `predicted_peak_bytes` is the request's INCREMENTAL demand
        // above them — the currency `MemoryBudget::fits` documents and the one Mage's own
        // peak/credit cross-check reads (see [`mage_admission_budget`]).
        budget: MemoryBudget {
            total_bytes,
            committed_bytes,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22733@{}", protocol::INFERENCE_PIN),
    }
}

pub(super) fn mage_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The scenario verdicts a completed Mage capture files.
///
/// WHY THIS IS A FUNCTION AND NOT AN INLINE LITERAL (sc-22738). The list is the input to
/// [`protocol::settle_plain_overlay_scenario`], which REPLACES the `overlay` entry with a verdict
/// derived from `planned.target.overlay` — and REFUSES, with "provider fragment is missing the
/// required overlay scenario", a fragment that has no `overlay` entry to replace. This arm's list
/// omitted one, so all eighteen `mage_flow*:*:mlx` cells rendered for 11-38 s and were then refused
/// while building the receipt. Every other arm in this binary declares the placeholder (some as an
/// `overlay_scenario` binding); Mage was the only omission. Hoisting the list out of the `json!`
/// literal lets the settler actually be run over it in a test, weights-free, which is what makes the
/// omission catchable before a campaign pays for the render rather than after.
///
/// The verdict itself is NOT authored here: the placeholder is inert and the settler derives the
/// real one from the plan row, so a Mage anchor re-planned with a material overlay fails closed
/// instead of acquiring a `not_applicable` it did not earn.
pub(super) fn mage_scenarios(predicted: u64) -> Value {
    json!([
        { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
        { "name": "unknown_budget", "result": "passed" },
        { "name": "stale_evidence", "result": "passed" },
        { "name": "warm_repeat", "result": "passed" },
        { "name": "cancel", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
        { "name": "error", "result": "passed", "cleanupVerified": true, "warmFollowUpPassed": true },
        { "name": "loadability", "result": "passed" },
        { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared target" }
    ])
}

/// One Mage-Flow anchor capture, on any of the six catalog ids, at any shipped tier.
///
/// E4: the generator comes from `catalog.media().load(arm.provider, &spec)` — the same registry load
/// the worker performs for this lane (`engines.rs` `MODEL_TABLE` maps each catalog id onto the
/// identically-named engine id, and the generic MLX stream hands the registry a `LoadSpec` carrying
/// the tier quant plus the two staged component dirs) — never a re-implementation of the loader.
pub(super) fn run_mage_provider(request: &Value) -> Result<Value, String> {
    let arm = mage_arm(request)?;
    validate_mage_mode(request, arm)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let seed = planned_mage_seed(request, arm, tier)?;
    validate_mage_plan_identity(request, arm, tier)?;
    // The plan's shape is the worker's: the six MLX Mage `memory_route_registry::RULES` are typed
    // (`requires_sequential_selection: false`), every Mage manifest entry's MLX BTR row matches
    // the anchor's exact request context, and `mlx-gen-mage` publishes BTR `Implemented` for a
    // streamable Deferred candidate, so `image_jobs/base.rs` evaluates `Applied + Deferred` before
    // any rung is selected — the resident anchor rows bind `deferred_materialization` (the
    // worker's `mage_mlx_production_load_shape_is_deferred_and_sequential_for_the_resident_selection`
    // drives that evaluator over the real manifest entries and pins the 18 `mage_flow*:*:mlx` plan
    // rows to it). The capture executes that shape and re-asserts it against the loaded identity.
    let load_shape = planned_load_shape(request)?;
    // Mage never reads `LoadSpec::offload_policy` — `assemble` builds a `Residency::request_scoped`
    // pipeline and staging is selected per request by `GenerationRequest::memory` — but the spec
    // still records which policy the worker asked for, and the resident anchor asks for Resident.
    let offload = if matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency | MemoryStrategy::BoundedTransformerResidency
    ) {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    let (width, height) = protocol::target_geometry(request)?;
    let source = mage_artifact_source(arm)?;
    let repository = source.repository.clone();
    let revision = source.revision.clone();
    let components_repository = source.components_repository.clone();
    let components_revision = source.components_revision.clone();
    let MageArtifact { root: _root, spec } =
        mage_load_spec(arm, tier, &source, &selection, offload, load_shape)?;
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.provider))?;
    // The knobs the LOADED provider accepts. Read once, here, and passed to every request this
    // capture builds — see [`mage_request`] for why the table cannot answer this question.
    let capabilities = generator.descriptor().capabilities.clone();
    // The plan names a tier; the LOADED provider decides which of its components it holds ABOVE
    // that tier, and the request must carry those promotions or the shared tier check refuses it
    // (see [`active_component_floors`]).
    let mut selection = selection;
    selection.tier.component_precision_floors = active_component_floors(
        generator
            .descriptor()
            .capabilities
            .component_precision_floors,
        selection.tier.quant,
    );
    let selection = selection;
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
            "pinned {} provider has no calibration identity",
            arm.provider
        )
    })?;
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
    let (required_total_peak_bytes, resident_credit_bytes) =
        mage_admission_budget(contract, selection.tier.quant, width, height);
    let context = mage_run_context(
        arm,
        selection,
        calibration,
        width,
        height,
        hardware_bytes,
        required_total_peak_bytes.saturating_sub(resident_credit_bytes),
        resident_credit_bytes,
    );

    // Three phase peaks cut on the lifecycle boundaries the pinned engine emits on the PRODUCTION
    // `Generator::generate` path (sc-22738, inference #963 at e16c6a55e): every checked-in Mage plan
    // row selects `MemoryStrategy::Resident`, i.e. the warm arm of
    // `gen_core::residency::run_staged_request_scoped`, which now opens the renderer phase with
    // `Loading(Renderer)` AFTER the prompt encode (`residency.rs` warm arms) — so that event IS the
    // conditioning/denoise edge — and `mlx-gen-mage`'s decode closures forward the progress sink,
    // so `Progress::Decoding` (`pipeline.rs` `decode_generation_phase` / `decode_edit_phase`) closes
    // denoise. The first `Progress::Step` is deliberately NOT cut on: it fires only after a full
    // denoise step, with the whole renderer resident, and is not the conditioning peak.
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
        mage_request(arm, &capabilities, width, height, seed),
        &context,
        None,
        &mut |progress| match progress {
            Progress::Loading(LoadPhase::Renderer) => {
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
            "a synchronized Mage-Flow lifecycle phase reported a zero active peak; the engine \
             stopped emitting a boundary and the attribution collapsed"
                .to_owned(),
        );
    }
    let phases = [conditioning, denoise, decode];
    let overall = PhaseMemory::overall(&phases);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;

    // A peak below the declared weights proves the capture is invalid, so it still refuses; a peak
    // ABOVE the engine's modelled total is a finding ABOUT that model and is recorded, never
    // refused (see [`mage_peak_below_weights_error`] and [`mage_model_shortfall_blocker`]).
    if let Some(error) = mage_peak_below_weights_error(arm, tier, predicted, resident_credit_bytes)
    {
        return Err(error);
    }
    let model_shortfall_blocker = mage_model_shortfall_blocker(
        arm,
        tier,
        width,
        height,
        predicted,
        required_total_peak_bytes,
    );
    // Exact fit in the provider's own currency: capacity for exactly the committed snapshot plus
    // the admitted incremental demand, and not one byte less. Restating the measured peak as the
    // demand here would be a different claim from the one the request was admitted under — Mage
    // credits every byte the caller does not declare against its resident envelope, so the admitted
    // pair is what an exact budget has to be built from (see [`mage_admission_budget`]).
    let mut exact = context.clone();
    exact.budget.total_bytes = exact
        .budget
        .required_total_bytes(exact.predicted_peak_bytes);
    if !matches!(
        generator.memory_strategy_safety_check(&exact),
        MemorySafetyDecision::Accept
    ) {
        return Err("Mage-Flow provider rejected an exact-fit calibrated budget".to_owned());
    }
    let mut one_byte_short = exact.clone();
    one_byte_short.budget.total_bytes = one_byte_short.budget.total_bytes.saturating_sub(1);
    if !matches!(
        generator.memory_strategy_safety_check(&one_byte_short),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(
            "Mage-Flow provider accepted a budget one byte short of the admitted request"
                .to_owned(),
        );
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Mage-Flow provider accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-mage-flow-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Mage-Flow provider accepted stale calibration evidence".to_owned());
    }
    // The six checkpoints are architecturally identical, so the provider's own route gate is the
    // only thing standing between an edit anchor and a text-to-image render recorded under it.
    // Prove it is live on THIS loaded generator rather than trusting the contract's declaration.
    let mut crossed = context.clone();
    crossed.mode = if arm.edit {
        MemoryMode::TextToImage
    } else {
        MemoryMode::Edit
    };
    crossed.has_reference = !arm.edit;
    crossed.geometry.reference_count = u32::from(!arm.edit);
    if !matches!(
        generator.memory_strategy_safety_check(&crossed),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "pinned {} provider accepted the opposite request mode",
            arm.provider
        ));
    }

    let baseline = one_image(
        generator
            .generate(
                &mage_request(arm, &capabilities, width, height, seed),
                &mut |_| {},
            )
            .map_err(|error| format!("generate unselected Mage-Flow reference: {error}"))?,
    )?;
    let (maximum_error, mean_error) = image_max_mean_abs(&selected, &baseline)?;
    if !mage_quality_passes(maximum_error, mean_error) {
        return Err(format!(
            "Mage-Flow selected rung exceeded unselected parity: max={maximum_error:.6}, \
             mean={mean_error:.6}"
        ));
    }
    let warm = one_image(scoped_generate(
        generator.as_ref(),
        mage_request(arm, &capabilities, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (warm_maximum, warm_mean) = image_max_mean_abs(&selected, &warm)?;
    if !mage_quality_passes(warm_maximum, warm_mean) {
        return Err("Mage-Flow warm repeat changed the deterministic output".to_owned());
    }

    let cancelled = mage_request(arm, &capabilities, width, height, seed);
    let cancel_signal = cancelled.cancel.clone();
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
            if let Progress::Step { current: 1, .. } = progress {
                cancel_triggered = true;
                cancel_signal.cancel();
            }
        },
    )
    .expect_err("in-flight Mage-Flow cancellation must fail");
    if !cancel_triggered {
        return Err(
            "Mage-Flow cancellation probe never reached the active rung boundary".to_owned(),
        );
    }
    if !cancel_error.to_ascii_lowercase().contains("cancel") {
        return Err(format!(
            "Mage-Flow cancellation returned the wrong error: {cancel_error}"
        ));
    }
    let cancel_recovery = one_image(scoped_generate(
        generator.as_ref(),
        mage_request(arm, &capabilities, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (cancel_maximum, cancel_mean) = image_max_mean_abs(&selected, &cancel_recovery)?;
    if !mage_quality_passes(cancel_maximum, cancel_mean) {
        return Err("Mage-Flow cancellation cleanup changed the warm follow-up".to_owned());
    }

    let injected = scoped_generate(
        generator.as_ref(),
        mage_request(arm, &capabilities, width, height, seed),
        &context,
        Some(MemoryPhase::Denoise),
        &mut |_| {},
    )
    .expect_err("injected Mage-Flow error must fail");
    if !injected.contains("authorized calibration fault") {
        return Err(format!(
            "Mage-Flow error injection returned the wrong error: {injected}"
        ));
    }
    let error_recovery = one_image(scoped_generate(
        generator.as_ref(),
        mage_request(arm, &capabilities, width, height, seed),
        &context,
        None,
        &mut |_| {},
    )?)?;
    let (recovery_maximum, recovery_mean) = image_max_mean_abs(&selected, &error_recovery)?;
    if !mage_quality_passes(recovery_maximum, recovery_mean) {
        return Err("Mage-Flow error cleanup changed the warm follow-up".to_owned());
    }

    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean) = image_max_mean_abs(&mutated, &baseline)?;
    if mage_quality_passes(mutated_maximum, mutated_mean) {
        return Err(
            "Mage-Flow output mutation did not breach the production parity envelope".to_owned(),
        );
    }

    let lifecycle_measurements = [
        ("conditioningActivePeak", "bytes", conditioning.active),
        ("denoiseActivePeak", "bytes", denoise.active),
        ("decodeActivePeak", "bytes", decode.active),
    ];
    // The finding rides on the SAME record the measurement does, so a reader of this cell cannot
    // see the peak without seeing that the engine's model did not cover it.
    let mut lifecycle_blockers: Vec<String> = Vec::new();
    lifecycle_blockers.extend(model_shortfall_blocker);
    // ...and the same pair machine-readably, so a consumer can size the gap without parsing prose.
    let model_measurements = [
        (
            "engineModelledTotalBytes",
            "bytes",
            required_total_peak_bytes,
        ),
        (
            "engineModelShortfallBytes",
            "bytes",
            predicted.saturating_sub(required_total_peak_bytes),
        ),
    ];

    let mut fragment = json!({
        "status": "complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": mage_complete_sweep(request)?,
        "scenarios": mage_scenarios(predicted),
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": {
            "contract": "same seed, prompt, sampling, precision, staged components, and loaded provider; selected rung versus unselected request",
            "identicalInputs": true,
            "identicalLatents": false,
            "result": "passed",
            "maximumError": maximum_error,
            "meanError": mean_error,
            "maximumErrorThreshold": MAGE_MAX_THRESHOLD,
            "meanErrorThreshold": MAGE_MEAN_THRESHOLD,
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
            // BOTH artifact triples: a Mage load that resolved the DiT from the planned variant
            // but the text encoder from another tier's components would otherwise leave no trace.
            "resolvedPathFingerprint": format!(
                "{repository}@{revision}:{tier}+{components_repository}@{components_revision}:{tier}"
            ),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-shared-ladder", arm.slug),
            "executed",
            lifecycle_blockers,
            [
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                // An edit capture conditions on exactly one reference image; a text-to-image
                // capture on none. Read off the arm the LOAD resolved, so a record can never claim
                // a reference the request did not carry.
                ("referenceImages", "count", u64::from(arm.edit)),
                // Both shared components are staged explicitly — never left to the loader's
                // flat-layout fallback, which the split rehost cannot satisfy.
                ("stagedComponents", "count", 2),
            ]
            .into_iter()
            .chain(model_measurements)
            .chain(lifecycle_measurements),
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
