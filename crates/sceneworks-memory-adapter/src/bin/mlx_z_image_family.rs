//! MLX Z-Image calibration and shared still-image capture helpers.
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
        low_res_start: None,
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
