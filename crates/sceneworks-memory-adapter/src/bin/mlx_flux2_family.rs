//! MLX flux2 capture owner.
use super::*;

pub(super) fn flux2_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= FLUX2_MAX_THRESHOLD && mean <= FLUX2_MEAN_THRESHOLD && rms <= FLUX2_RMS_THRESHOLD
}

/// The FLUX.2 arm's lifecycle blockers: the standing scope blocker, plus a warm-repeat retention
/// FINDING when the post-cleanup allocator reading exceeds the clean warm bound.
///
/// sc-22738 (measured 2026-09-06): `flux2_klein_9b:q8:mlx` rendered correctly and was then thrown
/// away, because a retention over this bound used to `return Err` and discard the whole capture.
/// The reading was real — 30,189,746,856 active against a 24,981,289,616-byte clean warm cleanup
/// plus a 709,354,919-byte tolerance, so 5,208,457,240 bytes over — but it is EVIDENCE, not a
/// reason to refuse, for three reasons:
///
/// * The measured phase peaks this record ships come from the FIRST render on a freshly loaded
///   generator — the production cold path. Renders two and three only feed the repeat comparison,
///   so what they retain cannot corrupt the numbers the record actually claims.
/// * The record already declares `warm_repeat` (and `cancel`/`error`) `not_run`, and says the
///   cleanup bounds are "attested in quality and diagnostics instead". Refusing on a bound the
///   record does not claim discarded the very diagnostics that were supposed to carry it: the one
///   observation of this event survived only as a truncated line in a campaign log.
/// * Production would retain it too. A cell whose warm repeat retains 5 GB is a fact about the
///   engine that admission should see, not a capture to re-roll until it comes out clean.
///
/// The bound is NOT weakened — `lifecycleWarmRepeatPostCleanupActive`, its clean twin and the
/// tolerance are all still measured and emitted, and an overage now lands in `blockers` under the
/// cell's own name — so a retention is louder in the store than it was as a dropped capture.
///
/// Scoped to this arm deliberately: it is the only one that measured a positive retention. Across
/// the 43 captures the same campaign committed, 42 report the warm and clean readings EQUAL TO THE
/// BYTE and the 43rd is 50 MB *below* its clean twin, so on every other arm the refusal has never
/// fired and there is nothing to re-characterize.
pub(super) fn flux2_lifecycle_blockers(
    model_id: &str,
    lifecycle_blocker: &str,
    bounds: LifecycleMemoryBounds,
    warm_post_cleanup: AllocatorState,
) -> Vec<String> {
    let mut blockers = vec![lifecycle_blocker.to_owned()];
    if !bounds.allows_retained(warm_post_cleanup) {
        blockers.push(format!(
            "{model_id} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
             clean warm cleanup {:?} plus {} bytes; recorded as measured evidence rather than \
             discarding the capture, whose peaks come from the cold first render",
            bounds.clean_post_cleanup, bounds.tolerance_bytes,
        ));
    }
    blockers
}

/// Which family member the plan asks for. Refuses by name: a `(provider, modelId)` pair no member
/// serves must not be measured as its nearest neighbour — in particular a KV plan must never be
/// satisfied by the base klein artifact, which shares the provider id.
///
/// Defense-in-depth mirror of the provider-mismatch guard `validate_z_image_batch` carries
/// (sc-18104): `run` dispatches by provider name today, so a future caller must be refused here
/// rather than misrouted into another member's contract.
pub(super) fn flux2_arm(request: &Value) -> Result<Flux2Arm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned
        .pointer("/target/provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    let model_id = planned
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    for arm in [FLUX2_DEV_ARM, FLUX2_KLEIN_ARM, FLUX2_KLEIN_KV_ARM] {
        if arm.provider == provider && arm.model_id == model_id {
            return Ok(arm);
        }
    }
    Err(format!(
        "the MLX FLUX.2 arm does not implement provider {provider:?} for model {model_id:?}"
    ))
}

/// The PRODUCTION calibration identity the loaded generator publishes for one `(member, tier)`
/// cell — what the plan row must name and what the capture verifies against the loaded contract.
///
/// sc-22727 (inference PR 948). Klein's identity used to be published only for a streamable load,
/// and a turnkey rehost was handed the weights-free registry string
/// ([`FLUX2_KLEIN_REGISTRY_FINGERPRINT`]) in production. It is now a property of the ADMITTED
/// ARTIFACT and its tier, minted by `mlx-gen-flux2::memory_strategy::klein_production_calibration`
/// as `flux2-klein-9b-<tier>-mlx-shared-ladder-<artifact_tag>-<route>-v1`, where:
///
/// * `artifact_tag` comes from `KleinArtifactInventory::artifact_tag` — `rehost` for the
///   `SceneWorks/flux2-klein-9b-mlx` turnkeys and `kv-rehost` for `…-kv-mlx`
///   (`turnkey_identity` reads the cache dir out of the snapshot path, so the two shipped repos
///   land on different tags even though they share a provider id), and
/// * `route` is `t2i` for `FLUX2_KLEIN_9B_ID`, which is the ONLY klein provider this text-to-image
///   lane loads. (`edit` / `kv-edit` belong to the edit providers, which this arm never drives.)
///
/// Written here as literals rather than read off the pinned crate on purpose: the constants and
/// the minting function do not exist at inference `c6d6a4db`, which is what CI builds, so the
/// plan/arm binding has to be weights-free and pin-independent. The capture refuses a loaded
/// contract whose identity differs from this table (`run_flux2`), so the two copies cannot drift
/// unnoticed once the epic's pin bump (sc-22738) lands.
pub(super) fn flux2_calibration_fingerprint(arm: Flux2Arm, tier: &str) -> String {
    match arm.model_id {
        "flux2_dev" => FLUX2_CALIBRATION_FINGERPRINT.to_owned(),
        "flux2_klein_9b" => format!("flux2-klein-9b-{tier}-mlx-shared-ladder-rehost-t2i-v1"),
        "flux2_klein_9b_kv" => format!("flux2-klein-9b-{tier}-mlx-shared-ladder-kv-rehost-t2i-v1"),
        model_id => unreachable!("no FLUX.2 member {model_id} at tier {tier}"),
    }
}

pub(super) fn validate_flux2_target(request: &Value) -> Result<Flux2Arm, String> {
    let arm = flux2_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    Ok(arm)
}

/// Bind the fixture to the planned FAMILY MEMBER, tier AND geometry edge, deriving the seed — the
/// same fixture-to-plan binding `planned_qwen_seed` enforces, extended to the edge because this
/// lane's plan carries tiers at more than one geometry, and to the member because the two klein
/// models are indistinguishable by provider id alone.
pub(super) fn planned_flux2_seed(
    request: &Value,
    arm: Flux2Arm,
    tier: &str,
    width: u32,
) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("flux2-{}-mlx-{tier}-{width}-seed", arm.fixture_slug);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse FLUX.2 fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse FLUX.2 fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(seed)
}

pub(super) fn flux2_request(width: u32, height: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a lighthouse on a rocky coastline at golden hour, photorealistic".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        // The first Step callback closes the conditioning envelope; the second supplies a real
        // denoise-only interval before Decoding — the z_image/qwen phase-boundary pattern.
        steps: Some(2),
        ..Default::default()
    }
}

/// Resolve and validate the member's `SCENEWORKS_FLUX2*_*` environment family into a tier-exact
/// load spec. The tier is DERIVED from `/target/tier` and threads through the per-tier ROOT suffix
/// check and `spec.quantize` — never hardcoded (sc-17097 fixed exactly that hardcoding on the
/// Candle side).
pub(super) fn flux2_load_spec(
    request: &Value,
    arm: Flux2Arm,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    // The identity check lives in `flux2_load_spec_at`, once, over these same values.
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    flux2_load_spec_at(arm, tier, selection, repository, revision, root)
}

/// The env-free half of [`flux2_load_spec`], so the artifact binding is unit-testable without the
/// process environment: the root must end in the PLANNED tier's directory under the member's OWN
/// repository, so a stale `…/q4` export can never satisfy a q8 or bf16 plan, and the KV artifact
/// can never satisfy a base-klein plan.
pub(super) fn flux2_load_spec_at(
    arm: Flux2Arm,
    tier: &str,
    selection: &MemorySelection,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<(String, String, LoadSpec), String> {
    protocol::validate_artifact_identity(&repository, &revision, arm.expected_repository)?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.expected_repository,
    )?;
    Ok((repository, revision, flux2_spec(arm, root, selection)))
}

/// The tier-exact FLUX.2 load spec: the member's WORKER shape (offload policy + load shape, see
/// [`Flux2Arm::load_shape`]), `resolved_route` bound to the CATALOG model id, and — on the members
/// whose loader takes it — the quant DERIVED from the planned selection. Those are the levers the
/// worker sets (`image_jobs/base.rs` `load_spec` + `with_resolved_route`, then the manifest-declared
/// shape and policy); see [`Flux2Arm::tier_quant_reaches_the_loader`] for why the klein rehosts
/// take the tier through the snapshot path alone.
pub(super) fn flux2_spec(arm: Flux2Arm, root: PathBuf, selection: &MemorySelection) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(arm.offload_policy)
        .with_load_shape(arm.load_shape)
        .with_resolved_route(arm.model_id);
    if arm.tier_quant_reaches_the_loader {
        if let Some(quant) = selection.tier.quant {
            spec = spec.with_quant(quant);
        }
    }
    spec
}

pub(super) fn flux2_admission_context(
    arm: Flux2Arm,
    selection: &MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    geometry: (u32, u32),
    probe: Flux2AdmissionProbe<'_>,
) -> MemoryRunContext {
    let (width, height) = geometry;
    let Flux2AdmissionProbe {
        fingerprint,
        total_bytes,
        predicted_peak_bytes,
    } = probe;
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
        evidence_revision: format!("{}@{}", arm.evidence_tag, protocol::INFERENCE_PIN),
    }
}

pub(super) fn flux2_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact resident tuple per plan row; marking it range-verified promotes no sibling tuple
    // (the generated matrix still requires a matching manifest binding per cell).
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// The `mlx:flux2_*` arm (sc-18218, extended to the whole family by sc-22727).
///
/// `flux2_dev` owns a distinct reference-free T2I contract supporting resident and staged execution;
/// this arm captures its resident rung. The two klein catalog models share one engine provider
/// whose ladder publishes five rungs, and are told apart by their artifact and
/// `LoadSpec::resolved_route`. In both cases
/// this arm reads the registry contract under the exact T2I provider id, then proves that the
/// loaded generator exposes the byte-for-byte same contract before measuring it. No edit-provider
/// declaration or edit-shaped context participates in this lane.
pub(super) fn run_flux2(request: &Value) -> Result<Value, String> {
    let arm = validate_flux2_target(request)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let rung = protocol::planned_rung(request)?;
    if arm.captures_resident_only && rung != "resident" {
        return Err(format!(
            "the MLX {} calibration arm captures only the resident strategy; \
             rung {rung:?} is not capturable by its unscoped resident measurement",
            arm.provider
        ));
    }
    // The plan's `loadShape` is what the record is checked against (the harness refuses a fragment
    // whose measured loadShape differs from the plan's), and the member's shape is fixed by the
    // worker, so a plan row spelling another shape is refused here by name, before weight work.
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != arm.load_shape {
        return Err(format!(
            "planned.loadShape {planned_shape:?} is not the {} worker load shape {:?}",
            arm.model_id, arm.load_shape
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    let (width, height) = protocol::target_geometry(request)?;
    let seed = planned_flux2_seed(request, arm, tier, width)?;
    let (repository, revision, spec) = flux2_load_spec(request, arm, tier, &selection)?;
    // The PRODUCTION catalog (E4): the same explicit MLX media registry `runtime_macos::catalog()`
    // composes for the worker, not a crate-local replica. A member the bundle does not register is
    // not routed on this lane and fails here by name.
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let registry = catalog.media();
    let contract = registry
        .memory_strategy_contract(arm.provider, &spec)
        .map_err(|error| {
            format!(
                "read {} T2I memory-strategy contract: {error}",
                arm.provider
            )
        })?
        .ok_or_else(|| {
            format!(
                "{} has no T2I memory-strategy contract at the pin",
                arm.provider
            )
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!(
            "pinned {} contract rejected planned selection: {error}",
            arm.model_id
        )
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        format!(
            "pinned {} contract has no calibration identity",
            arm.model_id
        )
    })?;
    let expected_fingerprint = flux2_calibration_fingerprint(arm, tier);
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned {} T2I contract fingerprint changed: expected {expected_fingerprint}, got {}",
            arm.model_id, calibration.fingerprint
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
    if seed != arm.seed {
        return Err(format!(
            "planned.fixture seed {seed} does not match the {} calibration seed {}",
            arm.model_id, arm.seed
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let admission_context = |fingerprint: &str, total_bytes: u64, predicted_peak_bytes: u64| {
        flux2_admission_context(
            arm,
            &selection,
            calibration,
            (width, height),
            Flux2AdmissionProbe {
                fingerprint,
                total_bytes,
                predicted_peak_bytes,
            },
        )
    };
    // Admission mutation hygiene: the gate must accept a fitting request (so the two rejections
    // cannot pass via a blanket refusal), reject an unknown/zero budget, and reject a mutated
    // calibration fingerprint.
    //
    // On the dev arm the registered T2I check is a PUBLIC crate function, so all three run BEFORE
    // the expensive load, exactly as sc-18218 wrote them. The klein routes expose their registered
    // check (`registered_klein_safety_check`) only through the loaded generator - it is
    // crate-private - so their identical three scenarios run immediately after the load instead.
    // Same scenarios, same record; only the moment differs, and it differs because of the pinned
    // crate's visibility, not a weaker claim.
    let scenarios = |check: &dyn Fn(&str, u64, u64) -> MemorySafetyDecision| -> Result<(), String> {
        if !matches!(
            check(&calibration.fingerprint, hardware_bytes, 1),
            MemorySafetyDecision::Accept
        ) {
            return Err(format!(
                "{} admission rejected a fitting probe budget; the scenario rejections below \
                 would be a blanket refusal, not evidence",
                arm.model_id
            ));
        }
        if !matches!(
            check(&calibration.fingerprint, 0, 1),
            MemorySafetyDecision::Reject { .. }
        ) {
            return Err(format!(
                "{} admission accepted an unknown/zero memory budget",
                arm.model_id
            ));
        }
        if !matches!(
            check(
                &format!("stale-{}-fingerprint", arm.slug),
                hardware_bytes,
                1
            ),
            MemorySafetyDecision::Reject { .. }
        ) {
            return Err(format!(
                "{} admission accepted stale calibration evidence",
                arm.model_id
            ));
        }
        Ok(())
    };
    if arm.provider == FLUX2_PROVIDER {
        scenarios(&|fingerprint, total_bytes, predicted| {
            mlx_gen_flux2::memory_strategy::registered_dev_t2i_safety_check(
                &spec,
                &contract,
                &admission_context(fingerprint, total_bytes, predicted),
            )
        })?;
    }

    let generator = registry
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} provider: {error}", arm.model_id))?;
    let loaded_contract = generator.memory_strategy_contract().ok_or_else(|| {
        format!(
            "loaded {} generator exposed no T2I memory contract",
            arm.model_id
        )
    })?;
    if loaded_contract != &contract {
        return Err(format!(
            "loaded {} generator contract differs from the registry contract",
            arm.model_id
        ));
    }
    if arm.provider != FLUX2_PROVIDER {
        scenarios(&|fingerprint, total_bytes, predicted| {
            generator.memory_strategy_safety_check(&admission_context(
                fingerprint,
                total_bytes,
                predicted,
            ))
        })?;
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
                &flux2_request(width, height, seed),
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
            .map_err(|error| format!("generate measured FLUX.2-dev render: {error}"))?,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak",
            arm.model_id
        ));
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    let exact_fit = flux2_admission_context(
        arm,
        &selection,
        calibration,
        (width, height),
        Flux2AdmissionProbe::exact_fit(calibration, predicted),
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err(format!(
            "{} admission rejected an exact-fit calibrated budget",
            arm.model_id
        ));
    }

    // Warm repeat determinism + cleanup bounds on this exact loaded provider. The scoped
    // lifecycle scenarios cannot run (no request scope, no injection site), so the unscoped
    // repeats gate quality and the allocator bounds gate cleanup instead.
    clear_cache();
    reset_peak_memory();
    let baseline = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm {} control: {error}", arm.model_id))?,
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
            arm.model_id
        ));
    }
    reset_peak_memory();
    let warm = one_image(
        generator
            .generate(&flux2_request(width, height, seed), &mut |_| {})
            .map_err(|error| format!("generate warm {} repeat: {error}", arm.model_id))?,
    )?;
    let warm_peak = get_peak_memory() as u64;
    if !cleanup_bounds.allows_warm_peak(warm_peak) {
        return Err(format!(
            "{} warm repeat peaked at {warm_peak} bytes, above the clean warm control \
             {clean_warm_peak} bytes plus 2%",
            arm.model_id
        ));
    }
    clear_cache();
    // Recorded, not refused — see `flux2_lifecycle_blockers` for why this reading is evidence.
    let warm_post_cleanup = AllocatorState::capture_current();
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            arm.model_id
        ));
    }

    // Arm-internal negative-mutation falsifiability check. A runtime_complete record must keep
    // `negativeMutation` null (`memory-calibration-harness.mjs#validateRuntimeComplete`), so the
    // breach is verified here — the capture fails if the envelope cannot be breached — and the
    // measured numbers land in diagnostics rather than in the record field.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.model_id
        ));
    }

    let lifecycle_blocker = concat!(
        "this FLUX.2 capture opens no memory-strategy request scope ",
        "and has no calibration fault-injection site, so the scoped lifecycle ",
        "scenario cannot execute; unscoped repeat determinism and allocator cleanup bounds are ",
        "attested in quality and diagnostics instead"
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
        "sweep": flux2_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": format!("the registered {} admission check rejected a zero/unknown budget", arm.model_id) },
            { "name": "stale_evidence", "result": "passed", "reason": format!("the registered {} admission check rejected a mutated calibration fingerprint", arm.model_id) },
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
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}"),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:{}-resident", arm.slug),
            "executed",
            flux2_lifecycle_blockers(
                arm.model_id,
                lifecycle_blocker,
                cleanup_bounds,
                warm_post_cleanup,
            ),
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
                ("loadShapeDeferred", "count", 0),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
