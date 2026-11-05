//! MLX sensenova capture owner.
use super::*;

/// Which family member the plan asks for. Refuses by name — a model id no member serves must not be
/// measured as its nearest neighbour, and the engine id alone is NOT an artifact identity, so a
/// `(provider, modelId)` pair that crosses the quality/fast split is refused too.
pub(super) fn sensenova_arm(request: &Value) -> Result<SenseNovaArm, String> {
    let planned = protocol::planned(request)?;
    let provider = protocol::plan_str(planned, "/target/provider")?;
    let model_id = protocol::plan_str(planned, "/target/modelId")?;
    SENSENOVA_FAMILY
        .into_iter()
        .find(|arm| arm.model_id == model_id && arm.provider == provider)
        .ok_or_else(|| {
            format!(
                "the MLX SenseNova arm does not implement modelId {model_id:?} on provider \
                 {provider:?} (family: {})",
                SENSENOVA_FAMILY
                    .iter()
                    .map(|arm| format!("{}:{}", arm.provider, arm.model_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// The production calibration identity `mlx-gen-sensenova` publishes for one `(route, tier)` cell.
///
/// The engine keys the identity on the resolved route and the ARTIFACT-PROVEN tier (never
/// `spec.quantize` alone), and deliberately not on the offload policy: SenseNova's rung 4 keys off
/// `LoadSpec::load_shape`, which `MemoryCalibrationIdentity::load_shape` already carries. The two
/// q8 cells of the two base routes keep the strings their `2026-08-03` campaign measured, bound
/// here to the engine's own constants rather than re-spelled, so a rename cannot drift the plan.
/// The capture refuses a loaded contract whose identity differs from this table, so the two copies
/// cannot drift unnoticed.
pub(super) fn sensenova_calibration_fingerprint(arm: SenseNovaArm, tier: &str) -> String {
    match (arm.model_id, tier) {
        ("sensenova_u1_8b", "q8") => {
            runtime_macos::providers::sensenova::memory_strategy::QUALITY_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        ("sensenova_u1_8b_fast", "q8") => {
            runtime_macos::providers::sensenova::memory_strategy::FAST_CALIBRATION_FINGERPRINT
                .to_owned()
        }
        (_, tier) => format!(
            "sensenova-u1-{}-{tier}-mlx-shared-ladder-v1",
            runtime_macos::providers::sensenova::memory_strategy::route_label(arm.model_id)
                .expect("registered SenseNova route")
        ),
    }
}

/// Bind the fixture to the planned tier AND geometry edge, so a bf16 record can never be emitted
/// against a q4 capture that merely reused the fixture string.
pub(super) fn validate_sensenova_fixture(
    request: &Value,
    arm: SenseNovaArm,
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
        .map_err(|error| format!("parse SenseNova fixture seed {seed:?}: {error}"))?;
    if seed != SENSENOVA_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the SenseNova calibration seed \
             {SENSENOVA_SEED}"
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse SenseNova fixture step count {steps:?}: {error}"))?;
    if steps != 2 {
        return Err(format!(
            "planned.fixture {fixture:?} must use the two-step calibration request"
        ));
    }
    Ok(())
}

/// The env-free half of [`sensenova_load_spec`], so the tier and route bindings are unit-testable
/// without weights. The root must end in the PLANNED tier's directory
/// (`.../snapshots/<revision>/<tier>`), so a stale `…/q4` export can never satisfy a q8 or bf16 plan
/// and quietly re-label another tier's peaks.
pub(super) fn sensenova_load_spec_at(
    arm: SenseNovaArm,
    repository: &str,
    revision: &str,
    root: PathBuf,
    tier: &str,
) -> Result<LoadSpec, String> {
    protocol::validate_artifact_identity(repository, revision, arm.expected_repository)?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        repository,
        revision,
        tier,
        arm.expected_repository,
    )?;
    // Exactly the worker's MLX still-image shape for this family. `Resident` + eager is what
    // `image_jobs/base.rs` settles on: SenseNova is off the shared `Residency` seam
    // (`engine_supports_sequential("sensenova_u1_8b")` is false), so nothing ever moves it to
    // `Sequential`, and no SenseNova manifest declares a `requiredOffloadPolicy`. The quant is the
    // RESOLVED tier's — `reconcile_resolved_tier_quant` rewrites the requested quant to the tier
    // the directory actually is, and the engine's `validate_artifact_tier` hard-errors on any other
    // pairing — so bf16 carries no quant at all and the packed tiers carry theirs.
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_resolved_route(arm.model_id);
    if let (_, Some(quant)) = tier_precision_quant(tier) {
        spec = spec.with_quant(quant);
    }
    Ok(spec)
}

pub(super) fn sensenova_load_spec(
    arm: SenseNovaArm,
    tier: &str,
) -> Result<(String, String, LoadSpec), String> {
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    let spec = sensenova_load_spec_at(arm, &repository, &revision, root, tier)?;
    Ok((repository, revision, spec))
}

/// The one fresh planned request every SenseNova capture renders. Two steps: the first Step
/// callback closes a conservative conditioning envelope and the second gives denoise its own
/// measured interval before Decoding — the shared z_image/qwen/flux phase-boundary pattern. The
/// family takes a guidance scale and no negative prompt (`image.supportsNegativePrompt: false`).
pub(super) fn sensenova_request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "an annotated infographic of a coastal lighthouse, crisp dense labels".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(SENSENOVA_SEED),
        steps: Some(2),
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sensenova_context(
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
        // call sites pass `calibration.fingerprint`.
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        // The pinned SenseNova route gate fails a phase-bearing context closed on the registry
        // fixture surface; the anchor is the plain single-phase text-to-image route.
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
        evidence_revision: format!("sc-22734@{}", protocol::INFERENCE_PIN),
    }
}

/// The safetensors inventory of a tier root, in the terms the pinned engine's `checkpoint_files`
/// decides on (`mlx-gen-sensenova/src/memory_strategy.rs:630-645`): every non-hidden
/// `*.safetensors` entry, sorted, plus whether an HF shard index sits beside them.
pub(super) fn sensenova_safetensors_inventory(root: &Path) -> (Vec<String>, bool) {
    let mut entries: Vec<String> = std::fs::read_dir(root)
        .map(|dir| {
            dir.filter_map(Result::ok)
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.ends_with(".safetensors") && !name.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    entries.sort();
    let index = root.join("model.safetensors.index.json").is_file();
    (entries, index)
}

/// Before the load, with the engine's OWN predicates (sc-22738): will the pinned engine publish a
/// calibration identity for this tier root at all?
///
/// `production_calibration_identity` (`mlx-gen-sensenova/src/memory_strategy.rs:900-931`)
/// publishes only for an artifact `verified_artifact` (:656-669) pinned AND whose on-disk tier
/// equals the requested one. Since inference #966 (e34d7b46a) `verified_artifact` pins two
/// inventories — exactly one `model.safetensors`, or exactly the shards that
/// `model.safetensors.index.json`'s `weight_map` names (`checkpoint_files`, :630-645) — so the
/// three dense quality rehosts' 8-shard `bf16/` roots pin like the single-file `_fast` turnkeys
/// and the packed tiers do. All three predicates are `pub` on the engine
/// (`verified_artifact_identity`, :671; `resolved_artifact_tier`, :733; `requested_tier`, :842),
/// so this asks them directly rather than re-spelling their rules — a pin that widens them widens
/// this preflight with it — and refuses in seconds, naming the layout and the engine site, instead
/// of after a two-minute load. The digests `verified_artifact_identity` computes are the ones the
/// loader reuses (the process-wide digest cache), so a passing preflight costs the load nothing.
pub(super) fn sensenova_identity_preflight(
    arm: SenseNovaArm,
    tier: &str,
    spec: &LoadSpec,
    expected_fingerprint: &str,
) -> Result<(), String> {
    use runtime_macos::providers::sensenova::memory_strategy as engine;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(format!(
            "{} loads a snapshot directory, not a single weights file",
            arm.model_id
        ));
    };
    if engine::verified_artifact_identity(spec).is_none() {
        let (entries, index) = sensenova_safetensors_inventory(root);
        let layout = match (entries.as_slice(), index) {
            ([], _) => "no safetensors entry at all".to_owned(),
            ([only], false) if only == "model.safetensors" => {
                "a single `model.safetensors` the engine could not pin (unreadable, or changed \
                 while it was being hashed)"
                    .to_owned()
            }
            ([only], false) => {
                format!("one safetensors entry [{only}], which is not `model.safetensors`")
            }
            (entries, true) => format!(
                "{} safetensors entries [{}] beside `model.safetensors.index.json` — a Hugging \
                 Face sharded layout the index does not account for exactly (its `weight_map` is \
                 unreadable, names a shard outside this directory, or names a set other than \
                 these entries), or a shard the engine could not pin",
                entries.len(),
                entries.join(", ")
            ),
            (entries, false) => format!(
                "{} safetensors entries [{}] with no `model.safetensors.index.json` — an \
                 unindexed multi-file layout",
                entries.len(),
                entries.join(", ")
            ),
        };
        return Err(format!(
            "the pinned {} provider at inference {} publishes no calibration identity for the {} \
             {tier} tier root, so this cell cannot capture at this pin and is refused before the \
             load: `verified_artifact` (mlx-gen-sensenova/src/memory_strategy.rs:656-669, through \
             `checkpoint_files` :630-645) pins exactly one `model.safetensors`, or exactly the \
             shards `model.safetensors.index.json`'s `weight_map` names, and returns None for any \
             other inventory, and {} carries {layout}; `production_calibration_identity` (:917, \
             `artifact?`) therefore publishes nothing for the production identity \
             {expected_fingerprint}, even though the loader (src/model.rs:446-451) loads such a \
             root through `load_raw`. Repair the snapshot (a complete, index-matching shard set or \
             a single `model.safetensors`); the plan row stays runnable and captures once the root \
             pins.",
            arm.provider,
            protocol::INFERENCE_PIN,
            arm.model_id,
            root.display(),
        ));
    }
    let requested = engine::requested_tier(spec);
    match engine::resolved_artifact_tier(spec) {
        Err(error) => Err(format!(
            "the pinned {} provider cannot prove the {} {tier} artifact's tier, so it publishes no \
             calibration identity (`resolved_artifact_tier`, \
             mlx-gen-sensenova/src/memory_strategy.rs:733-808, fails closed): {error}",
            arm.provider, arm.model_id
        )),
        Ok(resolved) if resolved != requested => Err(format!(
            "the {} {tier} tier root carries a {resolved:?} artifact while the plan asks the \
             loader for {requested:?}; the pinned engine keys the identity on the ARTIFACT-proven \
             tier (`production_calibration_identity`, \
             mlx-gen-sensenova/src/memory_strategy.rs:926) and publishes none for the pair, so \
             this root cannot serve this cell",
            arm.model_id
        )),
        Ok(_) => Ok(()),
    }
}

/// The attribution this arm files, or the refusal by name. `decode` is the read taken after
/// `generate` returns — the peak since the reset at `Decoding`.
pub(super) fn sensenova_phase_peaks(
    arm: SenseNovaArm,
    phases: &SenseNovaPhases,
    decode: PhaseMemory,
) -> Result<SenseNovaPhasePeaks, String> {
    if phases.renderer_boundaries != 1
        || phases.decoding_boundaries != 1
        || phases.other_loading_boundaries != 0
    {
        return Err(format!(
            "the pinned {} engine emitted {} `Progress::Loading(LoadPhase::Renderer)`, {} \
             `Progress::Decoding` and {} other `Progress::Loading(_)` boundaries where this arm's \
             three-phase cut expects exactly one, one and none per request (`StepReporter` opens \
             each phase once, t2i.rs `request_phase_bounds` / `finish_interleave_progress`); the \
             stream is not the one the cut was written against, so nothing is filed",
            arm.model_id,
            phases.renderer_boundaries,
            phases.decoding_boundaries,
            phases.other_loading_boundaries
        ));
    }
    let (Some(conditioning), Some(denoise)) = (phases.conditioning, phases.denoise) else {
        return Err(format!(
            "the pinned {} engine reported `Progress::Decoding` before \
             `Progress::Loading(LoadPhase::Renderer)`, so the conditioning/denoise edge was never \
             cut and no denoise phase closed; nothing is filed",
            arm.model_id
        ));
    };
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(format!(
            "a synchronized {} lifecycle phase reported a zero active peak (conditioning: {} \
             bytes, denoise: {} bytes, decode: {} bytes); with the resident backbone live no phase \
             can be empty, so the capture is invalid rather than cheap",
            arm.model_id, conditioning.active, denoise.active, decode.active
        ));
    }
    Ok(SenseNovaPhasePeaks {
        conditioning,
        denoise,
        decode,
    })
}

/// The `mlx:sensenova_u1_8b` / `mlx:sensenova_u1_8b_fast` family arm (sc-22734).
///
/// Loads through the SAME seam the worker loads all six routes through —
/// `runtime_macos::catalog().media().load(engine_id, &spec)`, which is what
/// `crates/sceneworks-worker/src/inference_runtime.rs` wraps — reads the LOADED generator's own
/// contract, and measures the resident anchor composition against it. `StagedResidency` and
/// `BoundedDecode` are structurally not applicable on this family (one flat fused dual-path
/// checkpoint, no separable conditioning component and no VAE decode phase), which is why the
/// anchor is the resident rung on this lane AND on Candle rather than the usual staged one.
pub(super) fn run_sensenova(request: &Value) -> Result<Value, String> {
    // Before the load, not inside it: a non-still target must be refused without paying for
    // weights. The arm is resolved first so the refusal carries the member's own label.
    let arm = sensenova_arm(request)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let planned_shape = planned_load_shape(request)?;
    if planned_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "{} must use the eager_materialization load shape the worker loads this family under; \
             deferred materialization is the rung-4 (bounded transformer residency) shape, not the \
             resident anchor's",
            arm.still_calibration
        ));
    }
    let selection = planned_selection(request)?;
    if selection.strategy != MemoryStrategy::Resident {
        return Err(format!(
            "{} measures the resident anchor: the pinned SenseNova contract classifies \
             StagedResidency and BoundedDecode StructurallyNotApplicable, so no other rung is a \
             capturable anchor on this family",
            arm.still_calibration
        ));
    }
    let tier = planned_qwen_tier(request)?;
    validate_sensenova_fixture(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    // The plan row must name the production identity this cell's loaded generator publishes —
    // checked against the table BEFORE the load, so a row still carrying a conformance string
    // fails in milliseconds rather than after a multi-gigabyte load.
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?.to_owned();
    let expected_fingerprint = sensenova_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.model_id
        ));
    }
    let (repository, revision, spec) = sensenova_load_spec(arm, tier)?;
    // With the engine's OWN identity predicates, before paying for the load (sc-22738): a root the
    // pinned engine cannot publish an identity for is refused in seconds, naming the inference
    // change, rather than after a two-minute load.
    sensenova_identity_preflight(arm, tier, &spec, &expected_fingerprint)?;

    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build MLX catalog: {error}"))?;
    let generator = catalog.media().load(arm.provider, &spec).map_err(|error| {
        format!(
            "load real {} {tier} provider on {}: {error}",
            arm.provider, arm.model_id
        )
    })?;
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
            "the loaded {} provider at inference {} published no calibration identity for the {} \
             {tier} artifact although the pre-load preflight (the engine's own \
             `verified_artifact_identity` and `resolved_artifact_tier`) passed; the production \
             identity for this cell is {expected_fingerprint} (the engines publish one per \
             (route, artifact-proven tier), `production_calibration_identity` at \
             mlx-gen-sensenova/src/memory_strategy.rs:900-931), so one of that gate's composition \
             clauses (:905-916) or the `_fast` pre-merged marker clause (:918-925) refused this \
             load and this cell captures only at a pin that carries it",
            arm.provider,
            protocol::INFERENCE_PIN,
            arm.model_id,
        )
    })?;
    if calibration.fingerprint != expected_fingerprint {
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
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        generator.memory_strategy_safety_check(&sensenova_context(
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
        safety("stale-sensenova-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(format!(
            "{} admission accepted stale calibration evidence",
            arm.provider
        ));
    }

    // Three phase peaks cut on the lifecycle boundaries the pinned engine emits on EVERY route
    // (sc-22738, inference #966 at e34d7b46a; see `SenseNovaPhases`): `Loading(Renderer)` after
    // the prefills is the conditioning/denoise edge, `Decoding` after the last step closes
    // denoise, and the returned image closes decode. The first `Step` is deliberately NOT cut on:
    // it fires only after a full denoise step, with the whole backbone resident, and is not the
    // conditioning peak. The stream reaches the cut only through the fold, which refuses by name
    // any stream that is not the one the engine emits.
    let mut phases = SenseNovaPhases::default();
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let selected = one_image(
        generator
            .generate(&sensenova_request(width, height), &mut |progress| {
                phases.observe(&progress, || {
                    let phase = PhaseMemory::capture();
                    reset_peak_memory();
                    phase
                });
            })
            .map_err(|error| format!("generate measured {} render: {error}", arm.model_id))?,
    )?;
    let decode = PhaseMemory::capture();
    let SenseNovaPhasePeaks {
        conditioning,
        denoise,
        decode,
    } = sensenova_phase_peaks(arm, &phases, decode)?;
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
            .generate(&sensenova_request(width, height), &mut |_| {})
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
            .generate(&sensenova_request(width, height), &mut |_| {})
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
    let warm_post_cleanup = AllocatorState::capture_current();
    if !cleanup_bounds.allows_retained(warm_post_cleanup) {
        return Err(format!(
            "{} warm repeat retained active/cache bytes {warm_post_cleanup:?} above the clean warm \
             cleanup {clean_post_cleanup:?} plus {} bytes",
            arm.model_id, cleanup_bounds.tolerance_bytes,
        ));
    }
    let (warm_maximum, warm_mean, warm_rms) = image_max_mean_rms_abs(&selected, &warm)?;
    if !flux2_quality_passes(warm_maximum, warm_mean, warm_rms) {
        return Err(format!(
            "{} second warm repeat changed the deterministic output",
            arm.model_id
        ));
    }

    // Arm-internal negative-mutation falsifiability check: a runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here and the numbers land in diagnostics.
    let mutated = qwen_negative_mutation(&selected);
    let (mutated_maximum, mutated_mean, mutated_rms) = image_max_mean_rms_abs(&mutated, &baseline)?;
    if flux2_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(format!(
            "{} output mutation did not breach the determinism envelope",
            arm.model_id
        ));
    }

    let lifecycle_blocker = concat!(
        "the pinned SenseNova crate opens no memory-strategy request scope for the resident anchor ",
        "composition and exposes no calibration fault-injection site, so the scoped lifecycle ",
        "scenarios cannot execute; unscoped repeat determinism and allocator cleanup bounds are ",
        "attested in quality and diagnostics instead"
    );
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": { "repository": repository, "resolvedRevision": revision, "variant": tier },
        "sweep": flux_one_complete_sweep(request)?,
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
            "contract": "identical artifact, prompt, guidance, seed, geometry, steps, tier, and loaded provider; cold measured render versus warm unscoped repeats",
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
            &format!("memory-mlx-adapter:sensenova-u1-{}-shared-ladder", arm.slug),
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
