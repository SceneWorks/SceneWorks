//! MLX instantid capture owner.
use super::*;

/// The ORDERED artifact set one InstantID capture consumes, hashed exactly as the worker hashes it
/// (`crates/sceneworks-worker/src/image_jobs/instantid.rs` `instantid_artifact_entries` /
/// `instantid_artifact_fingerprint`): the paths in role order, record-separated, hex sha256. The
/// role tag is positional, so it is not part of the hashed stream. The digest lands verbatim inside
/// the provider's `overlay_key()`, which is the axis `validate_context` checks — two different
/// artifact sets can therefore never price each other.
pub(super) fn instantid_artifact_fingerprint(paths: &[&Path]) -> String {
    let mut hasher = Sha256::new();
    for path in paths {
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(b"\x1e");
    }
    format!("{:x}", hasher.finalize())
}

/// The plan's InstantID target: the bespoke `instantid` provider on the `instantid_realvisxl`
/// catalog model, in the only mode the worker's route serves (`character_image`), under the
/// `identity` overlay the provider's own `overlay_key()` names.
pub(super) fn validate_instantid_target(request: &Value) -> Result<(), String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != INSTANTID_PROVIDER {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != INSTANTID_MODEL_ID {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} requires modelId {INSTANTID_MODEL_ID:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "character_image" {
        return Err(format!(
            "{INSTANTID_STILL_CALIBRATION} requires character_image mode, got {mode:?}"
        ));
    }
    protocol::validate_still_geometry(request, INSTANTID_STILL_CALIBRATION)?;
    protocol::validate_exact_overlay_target(request, "identity", INSTANTID_EXECUTION_PATH)
}

pub(super) fn instantid_binding(tier: &str) -> Result<InstantIdBinding, String> {
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
    Ok(InstantIdBinding {
        repository,
        revision,
        paths: mlx_gen_instantid::InstantIdPaths {
            sdxl_base: root,
            identitynet: WeightsSource::Dir(identitynet),
            ip_adapter: bundle.ip_adapter.clone(),
            adapters: Vec::new(),
        },
        bundle,
        artifact_fingerprint,
    })
}

/// The MLX InstantID numeric tier, mirroring the worker's `instantid_memory_tier`: the crate's own
/// dense sentinel with the packed quant the resolved tier selected (MLX honours q4/q8/bf16 —
/// `image_jobs/instantid.rs` `instantid_memory_backend_keys`).
pub(super) fn instantid_numeric_tier(tier: &str) -> MemoryNumericTier {
    MemoryNumericTier {
        quant: tier_precision_quant(tier).1,
        ..mlx_gen_instantid::memory_strategy::dense_numeric_tier()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn instantid_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    overlay: String,
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
        overlay: Some(overlay),
        budget: MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: mlx_gen_instantid::memory_strategy::REQUEST_EVIDENCE_REVISION.to_owned(),
    }
}

/// A deterministic 512-d ArcFace-shaped embedding. `InstantId::generate_with` is the engine's own
/// "face-stack-independent path": the identity VALUE never changes what the denoise and decode
/// phases materialize, so a fixed embedding makes the capture reproducible without needing a real
/// face photograph on the capture host. The face stack itself is still attached and resident (the
/// worker attaches it too), so the composition the peaks describe is production's.
pub(super) fn instantid_reference_embedding() -> Vec<f32> {
    (0..512)
        .map(|index| ((index as f32) * 0.017_f32).sin() * 0.05)
        .collect()
}

/// Five landmarks in the canonical `draw_kps` order (both eyes, nose, both mouth corners), scaled
/// from the normalized square the worker's angle presets use to output-canvas pixels.
pub(super) fn instantid_kps(side: u32) -> Vec<(f32, f32)> {
    let side = side as f32;
    [
        (0.37_f32, 0.42_f32),
        (0.63, 0.42),
        (0.50, 0.55),
        (0.40, 0.68),
        (0.60, 0.68),
    ]
    .into_iter()
    .map(|(x, y)| (x * side, y * side))
    .collect()
}

pub(super) fn instantid_request(
    width: u32,
    height: u32,
    seed: u64,
) -> mlx_gen_instantid::InstantIdRequest {
    mlx_gen_instantid::InstantIdRequest {
        prompt: "a studio portrait of a person, soft key light".to_owned(),
        negative: "low quality, blurry, distorted".to_owned(),
        width,
        height,
        // Two executed denoise steps — the same conditioning/denoise/decode phase split every other
        // still arm captures at.
        steps: 2,
        guidance: 3.0,
        ip_adapter_scale: INSTANTID_IP_SCALE,
        controlnet_scale: INSTANTID_CONTROLNET_SCALE,
        seed,
        ..Default::default()
    }
}

pub(super) fn planned_instantid_seed(
    request: &Value,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("instantid-realvisxl-mlx-{tier}-{width}-seed");
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    if steps != "2" {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    seed.parse::<u64>()
        .map_err(|error| format!("parse InstantID fixture seed {seed:?}: {error}"))
}

/// Real capture arm for the `mlx:instantid` identity route on `instantid_realvisxl`.
///
/// InstantID is a BESPOKE crate: `mlx-gen-catalog` lists it in `BESPOKE_UTILITY_CRATES` and
/// registers no `ModelDescriptor`, so there is no `registry.load` to call. The production loader is
/// `InstantId::load_with_memory_context` — literally the call `image_jobs/instantid.rs` makes — and
/// that is what this arm drives. The pinned provider implements only Resident and StagedResidency
/// (`memory_strategy.rs` declares the three bounded rungs Missing), and publishes its calibration
/// identity with a HARD-CODED `LoadShape::EagerMaterialization` that no spec can move.
pub(super) fn run_instantid(request: &Value) -> Result<Value, String> {
    validate_instantid_target(request)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != LoadShape::EagerMaterialization {
        return Err(
            "InstantID calibration must use the eager_materialization load shape the pinned \
             provider publishes"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    if !matches!(
        selection.strategy,
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency
    ) {
        return Err(format!(
            "MLX InstantID {:?} is declared Missing at the pinned provider and is not capturable",
            selection.strategy
        ));
    }
    let parameters = protocol::strategy_parameters(request)?;
    if !parameters.is_empty() {
        return Err(format!(
            "MLX InstantID calibration requires no strategy parameters, got {parameters:?}"
        ));
    }
    let tier = planned_qwen_tier(request)?;
    let (width, height) = protocol::target_geometry(request)?;
    if width != height {
        return Err(format!(
            "the InstantID identity route renders a SQUARE canvas (kps are normalized to a square); \
             planned geometry is {width}x{height}"
        ));
    }
    let seed = planned_instantid_seed(request, tier, width)?;
    if seed != INSTANTID_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the InstantID calibration seed {INSTANTID_SEED}"
        ));
    }
    let binding = instantid_binding(tier)?;
    let numeric_tier = instantid_numeric_tier(tier);
    if selection.tier.quant != numeric_tier.quant {
        return Err(format!(
            "planned tier {tier} implies quant {:?}, selection carries {:?}",
            numeric_tier.quant, selection.tier.quant
        ));
    }
    let identity = mlx_gen_instantid::memory_strategy::InstantIdMemoryIdentity {
        route: mlx_gen_instantid::memory_strategy::InstantIdRoute::Identity,
        adapter_count: 0,
        use_pid: false,
        face_restore: false,
        artifact_fingerprint: binding.artifact_fingerprint.clone(),
    };
    let overlay = identity.overlay_key();
    let contract = mlx_gen_instantid::memory_strategy::provider_contract_for_paths(
        &binding.paths,
        numeric_tier,
    );
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned InstantID contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
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

    // Admission hygiene BEFORE the expensive load: the gate must accept a fitting request (so the
    // refusals below are evidence rather than a blanket no), reject a zero budget, and reject a
    // mutated calibration fingerprint.
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        mlx_gen_instantid::memory_strategy::safety_check(
            &contract,
            numeric_tier,
            &identity,
            &instantid_context(
                selection,
                calibration,
                fingerprint,
                overlay.clone(),
                width,
                height,
                total_bytes,
                predicted,
            ),
        )
    };
    if !matches!(
        safety(&calibration.fingerprint, hardware_bytes, 1),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "InstantID admission rejected a fitting probe budget; the rejections below would be a \
             blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("InstantID admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-instantid-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("InstantID admission accepted stale calibration evidence".to_owned());
    }

    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let load_context = instantid_context(
        selection,
        calibration,
        &calibration.fingerprint,
        overlay.clone(),
        width,
        height,
        hardware_bytes,
        1,
    );
    // THE production loader: the exact call `image_jobs/instantid.rs` makes.
    let model = mlx_gen_instantid::InstantId::load_with_memory_context(
        &binding.paths,
        numeric_tier,
        identity.clone(),
        load_context,
    )
    .map_err(|error| format!("load real InstantID {tier} stack: {error}"))?;
    // The worker attaches the SCRFD/ArcFace stack too; `with_face_paths` is the reloadable seam
    // staged residency requires, so the same call serves both rungs.
    let mut model = model
        .with_face_paths(&binding.bundle.scrfd, &binding.bundle.arcface)
        .map_err(|error| format!("attach the InstantID face stack: {error}"))?;

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    reset_peak_memory();
    let embedding = instantid_reference_embedding();
    let kps = instantid_kps(width);
    model
        .generate_with(
            &instantid_request(width, height, seed),
            &embedding,
            &kps,
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
        .map_err(|error| format!("generate measured InstantID render: {error}"))?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized InstantID lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);

    // The pinned InstantID crates expose no calibration error injection and no request-scope
    // handle, so the runtime-complete and fault-injection scenarios stay unexecuted rather than
    // being reported as passed.
    let blocker = concat!(
        "the pinned mlx-gen-instantid provider is a bespoke crate with no registered generator, no ",
        "calibration error injection and no synchronized request scope, so the runtime-complete ",
        "sweep and the fault-injection lifecycle scenarios are not executable at this pin"
    );
    let mut fragment = json!({
        "strategy": strategy,
        "loadShape": protocol::LOAD_SHAPE_EAGER,
        "memory": {
            "conditioningPeakBytes": conditioning.active,
            "denoisePeakBytes": denoise.active,
            "decodePeakBytes": decode.active,
            "overallPeakBytes": overall.active,
            "predictedPeakBytes": predicted_peaks.overall,
        },
        "sweep": null,
        "quality": null,
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": format!(
                "{}@{}:{tier}+identity:{}",
                binding.repository, binding.revision, binding.artifact_fingerprint
            ),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:instantid-identity-ladder",
            "executed",
            [blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("decodeActivePeak", "bytes", decode.active),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("loadShapeEager", "count", 1),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    // The identity overlay is MATERIAL, so the plain-overlay settler (which only ever settles an
    // `overlay: none` target) must not run here. `validate_instantid_target` already proved the
    // declared overlay is exactly `identity`, and the overlay key the provider admitted is its own
    // artifact-bound `overlay_key()`, so the scenario is settled from that instead.
    fragment["scenarios"] = json!([
        { "name": "exact_fit", "result": "passed", "predictedBytes": predicted_peaks.overall },
        { "name": "unknown_budget", "result": "passed" },
        { "name": "stale_evidence", "result": "passed" },
        { "name": "loadability", "result": "passed" },
        {
            "name": "overlay",
            "result": "passed",
            "reason": "the identity overlay is the declared target, and the pinned provider admitted \
                       the composition under its own artifact-bound overlay key",
            "overlayKey": overlay,
        }
    ]);
    Ok(fragment)
}
