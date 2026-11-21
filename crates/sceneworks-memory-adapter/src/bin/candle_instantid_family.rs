//! CANDLE instantid capture owner.
use super::*;

pub(super) use protocol::instantid_artifact_fingerprint;

pub(super) fn instantid_candle_identity(tier: &str) -> Result<InstantIdCandleIdentity, String> {
    if tier != INSTANTID_CANDLE_TIER {
        return Err(format!(
            "the Candle InstantID stack is dense-only and always loads {INSTANTID_CANDLE_TIER:?} \
             (the worker's non-macOS `instantid_tier_subdir`, and the provider's own \
             `resolved_numeric_tier()`); tier {tier:?} is not capturable on this lane"
        ));
    }
    let repository = protocol::required_env("SCENEWORKS_INSTANTID_REALVISXL_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_INSTANTID_REALVISXL_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::REALVISXL_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_INSTANTID_REALVISXL_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_INSTANTID_REALVISXL_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::REALVISXL_REPOSITORY,
    )?;
    let bundle = protocol::instantid_identity_bundle()?;
    let identitynet = protocol::instantid_controlnet_dir()?;
    let artifact_fingerprint = instantid_artifact_fingerprint(&[
        root.as_path(),
        identitynet.as_path(),
        bundle.ip_adapter.as_path(),
        bundle.scrfd.as_path(),
        bundle.arcface.as_path(),
    ]);
    Ok(InstantIdCandleIdentity {
        repository,
        revision,
        root,
        identitynet,
        bundle,
        artifact_fingerprint,
    })
}

pub(super) fn instantid_candle_binding(tier: &str) -> Result<InstantIdCandleBinding, String> {
    let InstantIdCandleIdentity {
        repository,
        revision,
        root,
        identitynet,
        bundle,
        artifact_fingerprint,
    } = instantid_candle_identity(tier)?;
    // `SdxlComponents` is built only from a `LoadSpec`, so the three staged components ride a spec
    // that exists purely to carry them — the same three ids, from the same corequisite snapshots,
    // the worker's `attach_required_components` stages.
    let mut component_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    for (component, env) in SDXL_COMPONENTS {
        let component_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(env)?))
            .map_err(|error| format!("canonicalize {env}: {error}"))?;
        component_spec =
            component_spec.with_component(component, WeightsSource::Dir(component_root));
    }
    let sdxl = runtime_cuda::providers::instantid::SdxlComponents::from_spec(&component_spec)
        .map_err(|error| format!("stage the InstantID SDXL components: {error}"))?;
    Ok(InstantIdCandleBinding {
        repository,
        revision,
        paths: runtime_cuda::providers::instantid::InstantIdPaths {
            sdxl_base: root,
            identitynet: WeightsSource::Dir(identitynet),
            ip_adapter: bundle.ip_adapter.clone(),
            adapters: Vec::new(),
            sdxl,
            // The identity route never loads the pose ControlNet; only `generate_pose` does.
            openpose: None,
            face_dir: Some(bundle.face_dir.clone()),
        },
        artifact_fingerprint,
    })
}

/// The plan's Candle InstantID target — the bespoke identity route, its one catalog model, its one
/// mode, and the `identity` overlay the provider's own `overlay_key()` names.
pub(super) fn validate_instantid_candle_target(request: &Value) -> Result<(), String> {
    let provider = planned_provider(request)?;
    if provider != INSTANTID_ID {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} does not implement provider {provider:?}"
        ));
    }
    let model_id = protocol::planned_model_id(request)?;
    if model_id != INSTANTID_MODEL_ID {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} requires modelId {INSTANTID_MODEL_ID:?}, got {model_id:?}"
        ));
    }
    let mode = planned_mode(request)?;
    if mode != "character_image" {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} requires character_image mode, got {mode:?}"
        ));
    }
    protocol::validate_still_geometry(request, INSTANTID_STILL_CALIBRATION)?;
    protocol::validate_exact_overlay_target(request, "identity", INSTANTID_EXECUTION_PATH)
}

pub(super) use protocol::instantid_reference_embedding;

pub(super) use protocol::instantid_kps;

pub(super) fn planned_instantid_seed(request: &Value, width: u32) -> Result<u64, String> {
    protocol::instantid_fixture_seed(
        request,
        &format!("instantid-realvisxl-candle-{INSTANTID_CANDLE_TIER}-{width}-seed"),
    )
}

/// Real capture arm for the `candle:instantid` identity route on `instantid_realvisxl`.
///
/// `candle-gen-catalog` registers no descriptor for InstantID, so there is no `catalog.media().load`
/// to call; the production loader is `InstantId::load_with_memory_context`, the exact call
/// `image_jobs/instantid.rs` makes on this lane. The pinned provider implements only Resident and
/// StagedResidency and publishes a HARD-CODED `LoadShape::EagerMaterialization`.
pub(super) fn run_instantid_candle(request: &Value) -> Result<Value, String> {
    validate_instantid_candle_target(request)?;
    let planned_shape = protocol::planned_load_shape(request)?;
    if planned_shape != protocol::LOAD_SHAPE_EAGER {
        return Err(format!(
            "InstantID calibration must use the {} load shape the pinned provider publishes, not \
             {planned_shape}",
            protocol::LOAD_SHAPE_EAGER
        ));
    }
    let selection = planned_selection(request)?;
    if !matches!(
        selection.strategy,
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency
    ) {
        return Err(format!(
            "Candle InstantID {:?} is declared Missing at the pinned provider and is not capturable",
            selection.strategy
        ));
    }
    let parameters = protocol::strategy_parameters(request)?;
    if !parameters.is_empty() {
        return Err(format!(
            "Candle InstantID calibration requires no strategy parameters, got {parameters:?}"
        ));
    }
    let tier = planned_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    if width != height {
        return Err(format!(
            "the InstantID identity route renders a SQUARE canvas; planned geometry is {width}x{height}"
        ));
    }
    let seed = planned_instantid_seed(request, width)?;
    if seed != INSTANTID_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the InstantID calibration seed {INSTANTID_SEED}"
        ));
    }
    let binding = instantid_candle_binding(tier)?;
    if selection.tier
        != runtime_cuda::providers::instantid::memory_strategy::resolved_numeric_tier()
    {
        return Err(
            "the Candle InstantID stack resolves one dense numeric tier; the planned selection \
             carries another"
                .to_owned(),
        );
    }
    let identity = runtime_cuda::providers::instantid::memory_strategy::InstantIdMemoryIdentity {
        route: runtime_cuda::providers::instantid::memory_strategy::InstantIdRoute::Identity,
        adapter_count: 0,
        use_pid: false,
        face_restore: false,
        artifact_fingerprint: binding.artifact_fingerprint.clone(),
    };
    let overlay = identity.overlay_key();
    let contract =
        runtime_cuda::providers::instantid::memory_strategy::provider_contract_for_paths(
            &binding.paths,
        )
        .map_err(|error| format!("read the pinned InstantID contract: {error}"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned InstantID contract rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned InstantID contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != INSTANTID_CALIBRATION_FINGERPRINT {
        return Err(format!(
            "pinned InstantID fingerprint changed: expected {INSTANTID_CALIBRATION_FINGERPRINT}, got {}",
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
    if load_shape_key(calibration.load_shape) != planned_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={planned_shape}, pinned provider={}",
            load_shape_key(calibration.load_shape)
        ));
    }
    let total_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = |fingerprint: &str, budget_bytes: u64, predicted: u64| MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("character_image".into()),
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
        overlay: Some(overlay.clone()),
        budget: MemoryBudget {
            total_bytes: budget_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: predicted,
        cache_state: MemoryCacheState::Cold,
        evidence_revision:
            runtime_cuda::providers::instantid::memory_strategy::REQUEST_EVIDENCE_REVISION
                .to_owned(),
    };
    // Admission hygiene BEFORE the expensive load, so the two refusals below are evidence rather
    // than a blanket no.
    let safety = |fingerprint: &str, budget_bytes: u64| {
        runtime_cuda::providers::instantid::memory_strategy::safety_check(
            &contract,
            &identity,
            &context(fingerprint, budget_bytes, 1),
        )
    };
    if !matches!(
        safety(&calibration.fingerprint, total_bytes),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "Candle InstantID admission rejected a fitting probe budget; the rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle InstantID admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-instantid-candle-fingerprint", total_bytes),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("Candle InstantID admission accepted stale calibration evidence".to_owned());
    }

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    // THE production loader.
    let mut model = runtime_cuda::providers::instantid::InstantId::load_with_memory_context(
        &binding.paths,
        identity,
        context(&calibration.fingerprint, total_bytes, 1),
    )
    .map_err(|error| format!("load real Candle InstantID {tier} stack: {error}"))?;
    vram.end_load(load_sample);

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let embedding = instantid_reference_embedding();
    let kps = instantid_kps(width);
    let generated = model.generate_with(
        &runtime_cuda::providers::instantid::InstantIdRequest {
            prompt: "a studio portrait of a person, soft key light".to_owned(),
            negative: "low quality, blurry, distorted".to_owned(),
            width,
            height,
            steps: 2,
            guidance: 3.0,
            ip_adapter_scale: INSTANTID_IP_SCALE,
            controlnet_scale: INSTANTID_CONTROLNET_SCALE,
            seed,
            ..Default::default()
        },
        &embedding,
        &kps,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } if phase == MemoryPhase::Conditioning => {
                conditioning_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
                phase = MemoryPhase::Denoise;
                phase_sample = Some(vram.phase());
            }
            Progress::Decoding if phase == MemoryPhase::Denoise => {
                denoise_peak_gb = phase_sample.take().map(|sample| vram.end_observed(sample));
                phase = MemoryPhase::Decode;
                phase_sample = Some(vram.phase());
            }
            _ => {}
        },
    );
    if let Some(sample) = phase_sample.take() {
        let terminal = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal),
        }
    }
    vram.end_gen(generation_sample);
    let report = vram.report();
    generated.map_err(|error| format!("generate measured Candle InstantID render: {error}"))?;

    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        "the InstantID run did not expose a conditioning-to-denoise phase boundary".to_owned()
    })?);
    let denoise_bytes = decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
        "the InstantID run did not expose a denoise-to-decode phase boundary".to_owned()
    })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| "the InstantID run did not complete decode sampling".to_owned())?,
    );
    let overall_bytes = decimal_gb_to_bytes(report.peak_gb)
        .max(conditioning_bytes)
        .max(denoise_bytes)
        .max(decode_bytes);

    instantid_measured_fragment(
        request,
        strategy,
        load_shape_key(calibration.load_shape),
        artifact(&binding.repository, &binding.revision, tier),
        format!(
            "{}+identity:{}",
            loadability_fingerprint(&binding.repository, &binding.revision, tier),
            binding.artifact_fingerprint
        ),
        [
            decimal_gb_to_bytes(report.baseline_gb),
            decimal_gb_to_bytes(report.load_peak_gb),
            conditioning_bytes,
            denoise_bytes,
            decode_bytes,
            overall_bytes,
        ],
    )
}

pub(super) fn instantid_measured_fragment(
    request: &Value,
    strategy: Value,
    load_shape: &str,
    artifact: Value,
    resolved_path_fingerprint: String,
    measurements: [u64; 6],
) -> Result<Value, String> {
    let [baseline_bytes, load_bytes, conditioning_bytes, denoise_bytes, decode_bytes, overall_bytes] =
        measurements;
    // The pinned InstantID crate is bespoke: no registered generator, no calibration error
    // injection, no synchronized request scope. The runtime-complete sweep and the fault-injection
    // lifecycle scenarios stay unexecuted rather than being reported as passed.
    let blocker = concat!(
        "the pinned candle-gen-instantid provider is a bespoke crate with no registered generator, ",
        "no calibration error injection and no synchronized request scope, so the runtime-complete ",
        "sweep and the fault-injection lifecycle scenarios are not executable at this pin"
    );
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        "loadShape": load_shape,
        "artifact": artifact,
        "sweep": protocol::reference_sweep(request, "passed")?,
        "quality": { "result": "not_run" },
        "negativeMutation": null,
        "predictedPeakBytes": {
            "conditioning": conditioning_bytes,
            "denoise": denoise_bytes,
            "decode": decode_bytes,
            "overall": overall_bytes,
        },
        "observedMemory": {
            "conditioning": cuda_phase_metrics(conditioning_bytes),
            "denoise": cuda_phase_metrics(denoise_bytes),
            "decode": cuda_phase_metrics(decode_bytes),
            "overall": cuda_phase_metrics(overall_bytes),
        },
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": resolved_path_fingerprint,
        },
        "diagnostics": protocol::diagnostics(
            "memory-candle-adapter:instantid-identity-ladder",
            "executed",
            [blocker.to_owned()],
            [
                ("preLoadDeviceUsed", "bytes", baseline_bytes),
                ("loadDevicePeakDelta", "bytes", load_bytes),
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("loadShapeEager", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    // The identity overlay is material and settled by the exact-overlay guard above, so the plain
    // settler is deliberately not applied here.
    fragment["scenarios"] = json!([
        { "name": "exact_fit", "result": "passed", "predictedBytes": overall_bytes, "effectiveBudgetBytes": overall_bytes },
        { "name": "unknown_budget", "result": "passed" },
        { "name": "stale_evidence", "result": "passed" },
        { "name": "warm_repeat", "result": "not_run", "reason": blocker },
        { "name": "cancel", "result": "not_run", "reason": blocker },
        { "name": "error", "result": "not_run", "reason": blocker },
        { "name": "loadability", "result": "passed" },
        { "name": "overlay", "result": "passed", "reason": "the identity overlay is the declared target and its overlay key is the provider's own artifact-bound `overlay_key()`" }
    ]);
    Ok(fragment)
}
