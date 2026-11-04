//! MLX bernini capture owner.
use super::*;

/// The production calibration identity `mlx-gen-bernini` mints for one `(route, tier)` cell.
///
/// The engine keys on `(provider, ARTIFACT-proven tier)` and interpolates
/// `bernini-{route}-{tier}-mlx-dual-expert-ladder-v1`. The route token is the ENGINE's
/// (`calibration_route`), resolved for [`BERNINI_PROVIDER`] — every SceneWorks load resolves the
/// full planner+renderer pipeline (see the block comment above), which the engine names `image`,
/// as opposed to the renderer-only sibling `RENDERER_ID` it names `renderer`. Deriving it rather
/// than spelling `image` keeps the token from drifting from the crate that mints it.
///
/// The calibrated tier reproduces
/// [`MEMORY_CALIBRATION_FINGERPRINT`](runtime_macos::providers::bernini::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT)
/// byte-for-byte, which is what keeps the retained key of the family's existing record valid.
///
/// Written here as well as in the engine so the plan/arm binding is weights-free and provable on a
/// CPU-only host. The capture then refuses a loaded contract whose identity differs from this
/// table, so a drift is caught at the pin bump rather than in a record.
///
/// A tier outside [`BERNINI_TIERS`] is refused BY NAME rather than interpolated into a plausible
/// identity no production load can return.
pub(super) fn bernini_calibration_fingerprint(tier: &str) -> Result<String, String> {
    use runtime_macos::providers::bernini::memory_strategy as engine;
    if !BERNINI_TIERS.contains(&tier) {
        return Err(format!(
            "the MLX Bernini calibration family ships {BERNINI_TIERS:?}; tier {tier:?} has no \
             production identity"
        ));
    }
    let route = engine::calibration_route(BERNINI_PROVIDER).ok_or_else(|| {
        format!(
            "the pinned mlx-gen-bernini crate mints no calibration route for {BERNINI_PROVIDER:?}"
        )
    })?;
    if tier == BERNINI_CALIBRATED_TIER {
        return Ok(engine::MEMORY_CALIBRATION_FINGERPRINT.to_owned());
    }
    Ok(format!("bernini-{route}-{tier}-mlx-dual-expert-ladder-v1"))
}

/// Bernini's own geometry envelope, replacing the image arms' blanket `frames == 1` refusal.
///
/// The canvas half is the engine's `validate_geometry` — an exact membership test against
/// [`ADVERTISED_GEOMETRIES`](runtime_macos::providers::bernini::memory_strategy::ADVERTISED_GEOMETRIES),
/// read off the crate rather than transcribed — and it is PROVIDER-WIDE: `memory_strategy` applies
/// it to the still route and the video route alike, so a `bernini_image` anchor may not use one of
/// that entry's wider manifest resolutions (512², 768², 1024²) either. The temporal half is the
/// member's: the still entry renders exactly one frame, and the video entry renders the Wan
/// `1 mod 4` lattice count this arm plans.
pub(super) fn validate_bernini_geometry(
    arm: BerniniArm,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<BerniniGeometry, String> {
    let advertised = runtime_macos::providers::bernini::memory_strategy::ADVERTISED_GEOMETRIES;
    if !advertised.contains(&(width, height)) {
        return Err(format!(
            "{BERNINI_VIDEO_LABEL} memory evidence requires one of the advertised geometries \
             {advertised:?}, got {width}x{height}"
        ));
    }
    if frames != arm.frames {
        return Err(format!(
            "{BERNINI_VIDEO_LABEL} {} renders {} frame(s); got geometry.frames {frames}",
            arm.model_id, arm.frames
        ));
    }
    Ok(BerniniGeometry {
        width,
        height,
        frames,
    })
}

/// The concrete VAE geometry the pinned Bernini provider decodes through, resolved BY PROVIDER ID
/// from the engine's OWN registry-facing resolver
/// ([`mlx_gen_bernini::vae_tiling`](runtime_macos::providers::bernini::vae_tiling)) rather than
/// tabled here — the same way this arm reads [`ADVERTISED_GEOMETRIES`] and the retained calibration
/// key off the crate. It carries the two facts the decoded-frame rule needs: the temporal scale and
/// whether the temporal decode is causal.
pub(super) fn bernini_engine_vae() -> Result<VaeTiling, String> {
    runtime_macos::providers::bernini::vae_tiling(BERNINI_PROVIDER).ok_or_else(|| {
        format!(
            "{BERNINI_PROVIDER} publishes no VAE geometry; its decoded frame count cannot be \
             derived"
        )
    })
}

/// How many frames this member's engine RENDERS for its planned request (sc-22738).
///
/// Not `geometry.frames` on the video route: Bernini decodes through the NON-causal Wan z16 VAE
/// (`mlx-gen-bernini/src/lib.rs:45-47`, `ProviderVae = mlx_gen_wan::WanVae`; the crate's own
/// `provider_ids_are_bound_to_the_wan_z16_geometry` asserts `!causal_temporal`), so the shared rule
/// [`protocol::vae_decoded_frame_count`] answers `t_lat · 4` — 52 for the 49-frame video member —
/// and neither `bernini.rs`'s `generate_impl` nor the worker trims the longer clip back down.
///
/// The still member is the engine's own exception, applied on the engine's own predicate: for
/// `frames == 1` `bernini.rs` takes the FIRST decoded image out of the temporal chunk and returns a
/// one-image `GenerationOutput::Images`, so this arm expects exactly one frame there.
pub(super) fn bernini_rendered_frame_count(geometry: BerniniGeometry) -> Result<u32, String> {
    let vae = bernini_engine_vae()?;
    if geometry.frames == 1 {
        return Ok(1);
    }
    protocol::vae_decoded_frame_count(geometry.frames, vae.temporal_scale, vae.causal_temporal)
        .map_err(|error| format!("{BERNINI_PROVIDER}: {error}"))
}

/// Resolve the member and its geometry, refusing a foreign target BY NAME before any environment
/// variable is read, any path canonicalized, or any weight file opened.
///
/// `run` already dispatches by provider id, so the provider check here is defense in depth — but
/// the MODEL ID check is not: one provider serves two catalog entries whose records are filed
/// separately, and nothing else in this arm distinguishes them.
pub(super) fn bernini_target(request: &Value) -> Result<(BerniniArm, BerniniGeometry), String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let string = |field: &str| {
        target
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("planned.target.{field} must be a string"))
    };
    let provider = string("provider")?;
    if provider != BERNINI_PROVIDER {
        return Err(format!(
            "the MLX Bernini arm does not implement provider {provider:?}"
        ));
    }
    let model_id = string("modelId")?;
    let arm = BERNINI_ARMS
        .iter()
        .copied()
        .find(|arm| arm.model_id == model_id)
        .ok_or_else(|| {
            format!(
                "the MLX Bernini arm serves catalog entries {:?}, got modelId {model_id:?}",
                BERNINI_ARMS.map(|arm| arm.model_id)
            )
        })?;
    let mode = string("mode")?;
    if mode != arm.mode {
        return Err(format!(
            "{model_id} is measured in {:?} mode, got {mode:?}",
            arm.mode
        ));
    }
    // The published ladder is the PLAIN one. A reference-carrying Bernini target loads the same
    // block stack but seals its sources into a different evidence identity
    // (`R2V_REFERENCE_RECEIPT_DOMAIN` and its MV2V/ADS2V siblings), so a plain anchor may never be
    // filed against one.
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "{model_id}: this arm measures the reference-free ladder; a non-zero {field} \
                     seals a different Bernini evidence identity and cannot be recorded from it"
                ));
            }
        }
    }
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let axis = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "{BERNINI_VIDEO_LABEL} requires geometry.batch == 1, got {batch}"
        ));
    }
    let geometry =
        validate_bernini_geometry(arm, axis("width")?, axis("height")?, axis("frames")?)?;
    Ok((arm, geometry))
}

/// The target's declared reference count, defaulting to zero when the plan omits the field.
///
/// [`bernini_target`] has already refused any non-zero value by name, so on every cell this arm
/// can plan the answer is `0` — but the arm's admission decision READS it rather than assuming it,
/// so a reference-carrying member added later cannot silently inherit a decision taken for the
/// reference-free one.
pub(super) fn bernini_reference_count(request: &Value) -> Result<u32, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            return value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| format!("planned.target.{field} must fit u32"));
        }
    }
    Ok(0)
}

/// Bind the fixture to the member, the planned tier and the full rendered geometry, recovering the
/// cadence and the seed. Like MiniMax-H3's, this arm's `fps` has exactly one legal value, so the
/// fixture cannot declare a cadence the engine would refuse.
pub(super) fn planned_bernini_capture(
    request: &Value,
    arm: BerniniArm,
    tier: &str,
    geometry: BerniniGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "bernini-{}-mlx-{tier}-{}x{}-f{}-fps",
        arm.slug, geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse Bernini fixture fps {fps:?}: {error}"))?;
    if fps != BERNINI_FPS {
        return Err(format!(
            "planned.fixture declares fps {fps}, but the shipped Bernini manifest publishes \
             {BERNINI_FPS} fps only"
        ));
    }
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Bernini fixture seed {seed:?}: {error}"))?;
    if seed != BERNINI_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the Bernini calibration seed {BERNINI_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// Resolve and validate the `SCENEWORKS_BERNINI_*` environment family into a tier-exact load spec.
///
/// ONE triple, unlike MiniMax-H3's two: the `SceneWorks/bernini-mlx` rehost is self-contained
/// (`video_jobs/bernini.rs`: "the turnkey `SceneWorks/bernini-mlx` snapshot is self-contained"), so
/// `_ROOT` is the tier directory and nothing is redirected onto a second tree.
///
/// **`Resident` + `EagerMaterialization`, and no `quantize`.** All three are the worker's own
/// shape, not a choice made here:
///
/// * the tiers ship PRE-PACKED, so `resolve_bernini_tier_dir_and_quant` returns `quant = None` for
///   a resolved tier subdir and only a legacy flat snapshot keeps a load-time quant. Passing the
///   planned tier's `Quant` would ask the engine to re-quantize already-packed weights;
/// * the still lane builds its spec with no adapters and the MLX route rule
///   (`memory_route_registry.rs`, `backend: Mlx`, `provider: "bernini"`) declares `PLAIN`, so no
///   overlay is in scope;
/// * `structurally_streamable` — the engine's own gate on the deferred block stream — requires
///   `DeferredMaterialization`, which the registry rule reaches only through the worker's
///   declaration evaluator. The ANCHOR is the plain resident cell, so this arm executes the eager
///   resident shape the cold load actually takes and attests the loaded contract's own
///   `load_shape` back against it.
pub(super) fn bernini_load_spec(
    request: &Value,
    tier: &str,
    load_shape: LoadShape,
) -> Result<BerniniArtifact, String> {
    bernini_load_spec_at(
        request,
        tier,
        load_shape,
        protocol::required_env("SCENEWORKS_BERNINI_REPOSITORY")?,
        protocol::required_env("SCENEWORKS_BERNINI_REVISION")?,
        PathBuf::from(protocol::required_env("SCENEWORKS_BERNINI_ROOT")?),
    )
}

/// The environment-free half of [`bernini_load_spec`]: the same validations over caller-supplied
/// identity and root, so a test can hand it a root of the WRONG tier and read the refusal — the
/// sc-17097 defect class — without touching the process environment.
pub(super) fn bernini_load_spec_at(
    request: &Value,
    tier: &str,
    load_shape: LoadShape,
    repository: String,
    revision: String,
    root: PathBuf,
) -> Result<BerniniArtifact, String> {
    protocol::validate_artifact_identity(&repository, &revision, protocol::BERNINI_REPOSITORY)?;
    let tier_root = std::fs::canonicalize(&root)
        .map_err(|error| format!("canonicalize SCENEWORKS_BERNINI_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &tier_root,
        &repository,
        &revision,
        tier,
        protocol::BERNINI_REPOSITORY,
    )?;
    protocol::validate_plain_overlay_target(request, bernini_target(request)?.0.execution_path)?;
    let spec = LoadSpec::new(WeightsSource::Dir(tier_root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape);
    Ok(BerniniArtifact {
        repository,
        revision,
        tier_root,
        spec,
    })
}

/// Whether this capture asks Bernini's memory admission ANYTHING — the worker's own decision,
/// mirrored (sc-22738).
///
/// `crates/sceneworks-worker/src/video_admission.rs#bernini_memory_attempt` is the predicate that
/// decides whether a Bernini request is a memory-optimization attempt at all: the request must
/// name the video catalog entry AND either ask one of the four reference-carrying video routes or
/// carry at least one reference. For a reference-free `text_to_video` it answers `false`, so
/// production never treats such a request as one Bernini admission has an opinion about — no
/// `MemoryRunContext` is built for it and the provider's safety check is never called on it. The
/// still member is even further from that seam: `image_jobs/bernini.rs` passes `None`/`None` for
/// `memory`/`memory_strategy_context` on the MLX lane unconditionally ("The MLX still lane has no
/// shared Candle memory evaluation, so the honest values are `None`/`None`"), so `frames == 1`
/// Bernini renders resident with no memory context on every path.
///
/// The engine agrees from the other side: `mlx-gen-bernini`'s `memory_strategy::safety_check`
/// route gate admits exactly the reference-carrying video routes and refuses everything else by
/// name, and gen-core runs that gate unconditionally. sc-22737 probed the anchor's own
/// reference-free surface anyway, and all six `bernini*:*:mlx` anchors died on the first probe —
/// after a full dual-expert weight load — with "admission rejected a fitting probe budget".
///
/// So the arm skips the probe exactly where the worker skips the context. It does NOT present a
/// synthetic reference-carrying surface to make the gate answer: that would measure an admission
/// decision the product never makes for this request, and the anchor would then carry an admission
/// characterization no shipped path exercises.
///
/// This is a decision, not a constant: it is computed from the resolved member and the target's
/// reference count so a future reference-carrying Bernini anchor cannot inherit the skip silently.
pub(super) fn bernini_probes_admission(model_id: &str, mode: &str, reference_count: u32) -> bool {
    model_id == BERNINI_VIDEO_MODEL_ID
        && (matches!(
            mode,
            "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
        ) || reference_count > 0)
}

/// The measured request.
///
/// `steps` and `guidance` are left UNSET on purpose. `mlx-gen-bernini`'s `FullDefaults` is private,
/// so a literal here would be a second, unbindable copy of the engine's own default that could
/// drift silently at a pin bump; `None` executes exactly what the worker's cold render executes
/// (`video_jobs/bernini.rs` leaves both at the engine defaults too) and is identical across the
/// measured render and its warm repeats, which is all the determinism comparison needs.
pub(super) fn bernini_request(
    arm: BerniniArm,
    geometry: BerniniGeometry,
    fps: u32,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt:
            "a slow crane over a terracotta rooftop at golden hour, swallows turning, cinematic"
                .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: (arm.model_id == BERNINI_VIDEO_MODEL_ID).then_some(fps),
        ..Default::default()
    }
}

pub(super) fn bernini_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= BERNINI_MAX_THRESHOLD
        && mean <= BERNINI_MEAN_THRESHOLD
        && rms <= BERNINI_RMS_THRESHOLD
}

/// Unwrap the member's own output shape into the frame list every downstream comparison uses. The
/// still member returns `Images`; the video member returns `Video` and its cadence is asserted
/// against the fixture's, so a record can never claim a rate the engine did not produce.
pub(super) fn bernini_output_frames(
    arm: BerniniArm,
    output: GenerationOutput,
    fps: u32,
) -> Result<Vec<Image>, String> {
    match (arm.model_id, output) {
        (BERNINI_IMAGE_MODEL_ID, GenerationOutput::Images(images)) => {
            if images.len() != 1 {
                return Err(format!(
                    "{BERNINI_IMAGE_MODEL_ID} returned {} images for a count=1 request",
                    images.len()
                ));
            }
            Ok(images)
        }
        (
            BERNINI_VIDEO_MODEL_ID,
            GenerationOutput::Video {
                frames,
                fps: rendered,
                ..
            },
        ) => {
            if rendered != fps {
                return Err(format!(
                    "{BERNINI_VIDEO_MODEL_ID} returned {rendered} fps for a {fps} fps request"
                ));
            }
            if frames.is_empty() {
                return Err(format!("{BERNINI_VIDEO_MODEL_ID} returned no frames"));
            }
            Ok(frames)
        }
        (model_id, other) => Err(format!(
            "{model_id} returned an output shape this arm cannot record: {other:?}"
        )),
    }
}

/// The `mlx:bernini` arm — one measured render plus two warm repeats on the loaded provider, with
/// the four admission probes taken against the provider's own registered `safety_check`.
///
/// Structurally the MiniMax-H3 arm without the audio half: same three fingerprint checks (plan vs
/// arm, provider vs arm, plan vs provider), same pre-load admission hygiene, same three phase peaks
/// cut on the boundaries the shipped `generate` already emits.
pub(super) fn run_bernini(request: &Value) -> Result<Value, String> {
    let (arm, geometry) = bernini_target(request)?;
    // The lane's render plan (sc-22738). This ONE arm serves both catalog members: the still
    // `bernini_image` entry (frames == 1) is an IMAGE capture and keeps its clean warm control and
    // warm repeat exactly as before; the video entry is ONE measured render. The branch below is
    // the lane's — derived from the plan's frame count — never the member's.
    let capture = protocol::capture_policy(request)?;
    if (capture.lane == protocol::CaptureLane::Video) != (arm.frames > 1) {
        return Err(format!(
            "the {} member renders {} frame(s) but the plan's geometry puts it on the {:?} lane",
            arm.model_id, arm.frames, capture.lane
        ));
    }
    protocol::validate_plain_overlay_target(request, arm.execution_path)?;
    let load_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    let (fps, seed) = planned_bernini_capture(request, arm, tier, geometry)?;
    let expected_fingerprint = bernini_calibration_fingerprint(tier)?;
    // Fingerprint check 1 of 3: the PLAN against this arm's own expectation, before any environment
    // or weight work. A provider re-fingerprint must not silently reuse the epic's plan.
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/adapter calibration mismatch: plan={planned_fingerprint}, MLX Bernini arm \
             implements {expected_fingerprint}"
        ));
    }
    let artifact = bernini_load_spec(request, tier, load_shape)?;
    let spec = &artifact.spec;
    // The on-disk footprint of the staged tier, read BEFORE the load for the reason the LTX and
    // MiniMax arms do it: a mis-staged directory fails here, before any GPU work, rather than after
    // three full dual-expert renders.
    let staged_tier_bytes = safetensors_bytes(&artifact.tier_root)?;

    let registry = runtime_macos::providers::bernini::provider_registry()
        .map_err(|error| format!("build Bernini registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(BERNINI_PROVIDER, spec)
        .map_err(|error| format!("read {BERNINI_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX Bernini provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned Bernini contract rejected planned selection: {error}"))?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Bernini contract has no calibration identity".to_owned())?;
    // Fingerprint check 2 of 3: the PROVIDER contract against this arm's expectation.
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned Bernini contract fingerprint changed: expected {expected_fingerprint}, got {}",
            calibration.fingerprint
        ));
    }
    // Fingerprint check 3 of 3: the plan against the provider, so the two cannot agree with the arm
    // separately while disagreeing with each other.
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if calibration.load_shape != load_shape {
        return Err(format!(
            "pinned Bernini contract resolved load shape {:?} for a plan that declares {:?}",
            calibration.load_shape, load_shape
        ));
    }

    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let wired_limit_bytes = request
        .pointer("/hardware/wiredLimitBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.wiredLimitBytes must be an integer".to_owned())?;

    // The arm's admission decision, taken the way the WORKER takes it (sc-22738): see
    // [`bernini_probes_admission`]. On every cell this arm can plan the answer is `false` — the
    // reference-free ladder has no admission surface on either side of the seam — so no probe
    // context is built, nothing is asked of the provider's safety check, and the render below runs
    // resident exactly as production runs it. The record says so rather than claiming an admission
    // result it never obtained.
    //
    // A `true` answer would mean the plan reached this arm with a reference-carrying Bernini
    // target, which `bernini_target` has already refused BY NAME: such a request seals a different
    // evidence identity and needs its own capture, not a probe synthesized here. It is refused
    // rather than assumed away.
    let reference_count = bernini_reference_count(request)?;
    if bernini_probes_admission(arm.model_id, arm.mode, reference_count) {
        return Err(format!(
            "{}: production WOULD build a Bernini memory context for this surface, but this arm \
             measures only the reference-free ladder; a reference-carrying anchor seals its own \
             evidence identity and needs its own capture",
            arm.model_id
        ));
    }

    let generator = registry
        .load(BERNINI_PROVIDER, spec)
        .map_err(|error| format!("load real Bernini {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded Bernini generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded Bernini generator contract differs from the registry contract".to_owned(),
        );
    }

    // Three phase peaks cut on the boundaries the shipped `generate` already emits, in the ordering
    // the engine's own staged tests use: read the peak AND the footprint first, reset only after.
    let pre_generate = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let conditioning_close = Cell::new(AllocatorState::default());
    let denoise_close = Cell::new(AllocatorState::default());
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let measured = bernini_output_frames(
        arm,
        generator
            .generate(
                &bernini_request(arm, geometry, fps, seed),
                &mut |progress| match progress {
                    Progress::Loading(LoadPhase::TextEncoder) => {
                        pre_generate.set(PhaseMemory::capture());
                        reset_peak_memory();
                    }
                    Progress::Loading(LoadPhase::Renderer) => {
                        conditioning.set(PhaseMemory::capture());
                        conditioning_close.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    Progress::Decoding => {
                        denoise.set(PhaseMemory::capture());
                        denoise_close.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    _ => {}
                },
            )
            .map_err(|error| format!("generate measured Bernini render: {error}"))?,
        fps,
    )?;
    let decode = PhaseMemory::capture();
    let decode_close = AllocatorState::capture_current();
    let pre_generate = pre_generate.get();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let conditioning_close = conditioning_close.get();
    let denoise_close = denoise_close.get();
    // All three declared phases, on both routes: the pinned `mlx-gen-bernini` `generate_impl`
    // (inference #963 at e16c6a55e) emits `Loading(TextEncoder)` before the planner load and
    // `Loading(Renderer)` at the expert loads, so the conditioning/denoise edge is bounded again.
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized Bernini lifecycle phase reported a zero active peak; the engine \
             stopped emitting a boundary and the attribution collapsed"
                .to_owned(),
        );
    }
    // The engine's own decoded depth for this request, NOT the requested count: Bernini's z16
    // decode is NON-causal, so it materializes four output frames per latent frame and a 49-frame
    // video request renders 52 (sc-22738 — the same defect that refused the SCAIL-2 anchor after a
    // completed render). Production asks for the same count and keeps the clip it gets, so the
    // adapter measures the same path instead of refusing it.
    let expected_frames = bernini_rendered_frame_count(geometry)?;
    if measured.len() as u64 != u64::from(expected_frames) {
        return Err(format!(
            "Bernini rendered {} frames for a {}-frame request; its {} VAE decodes that request \
             to {expected_frames} frames",
            measured.len(),
            geometry.frames,
            if bernini_engine_vae()?.causal_temporal {
                "causal"
            } else {
                "non-causal"
            }
        ));
    }
    let first = measured
        .first()
        .ok_or_else(|| "Bernini render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("Bernini render returned a degenerate first frame".to_owned());
    }

    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    // The video member charges the RESIDENT ACTIVE peak for the reason every video lane does; the
    // still member keeps the historical componentwise image bound, so a `bernini_image` record is
    // admissible under exactly the arithmetic shipped image admission already applies. Each member
    // binds its receipt through its OWN lane wrapper — the source-text guard
    // `every_receipt_builder_is_bound_to_its_lane_prediction_wrapper` reads every `predicted_peaks`
    // binding in this file, and a binding that merely selects between the two wrappers is not one
    // it can tell from a bypass.
    let (predicted_peaks_json, predicted) = if arm.model_id == BERNINI_IMAGE_MODEL_ID {
        let predicted_peaks = image_predicted_peak_bytes(conditioning, denoise, decode);
        (predicted_peaks.json(), predicted_peaks.overall)
    } else {
        let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
        (predicted_peaks.json(), predicted_peaks.overall)
    };
    // The same two ceilings `memory-calibration-harness.mjs#assertResidencyFitsHardware` applies —
    // checked HERE so a capture that cannot be admitted fails loudly during the campaign rather
    // than producing a well-formed record the harness rejects afterwards.
    if overall.active > hardware_bytes {
        return Err(format!(
            "Bernini observed overall active {} bytes above the probed hardware memory \
             {hardware_bytes} bytes",
            overall.active
        ));
    }
    if overall.active > wired_limit_bytes {
        return Err(format!(
            "Bernini observed overall active {} bytes above the probed wired ceiling \
             {wired_limit_bytes} bytes",
            overall.active
        ));
    }
    // The post-render exact-fit probe is skipped for the SAME reason the pre-load ones are: it
    // derives the decision the worker derives, and no context means no probe. Asking the loaded
    // generator to admit a synthetic reference-carrying budget here would price an admission
    // decision this request never triggers, and the receipt would carry it as though it did.

    /// The image lane's two warm passes and what they measured (sc-22738): `Some` only when the
    /// lane runs them. On the video lane there is no clean warm control to judge determinism
    /// against and no warm repeat to bound, and the receipt says so (`warm_repeat` not_run,
    /// `quality.warmPasses: 0`) instead of writing zeros where these figures would be.
    struct BerniniWarmPasses {
        baseline: Vec<Image>,
        clean_warm_peak: u64,
        clean_post_cleanup: AllocatorState,
        cleanup_bounds: LifecycleMemoryBounds,
        warm_peak: u64,
        warm_post_cleanup: AllocatorState,
        maximum_error: f64,
        mean_error: f64,
        rms_error: f64,
    }
    let warm = if capture.warm_passes == 0 {
        None
    } else {
        // Warm repeat determinism + cleanup bounds on this exact loaded provider.
        clear_cache();
        reset_peak_memory();
        let baseline = bernini_output_frames(
            arm,
            generator
                .generate(&bernini_request(arm, geometry, fps, seed), &mut |_| {})
                .map_err(|error| format!("generate warm Bernini control: {error}"))?,
            fps,
        )?;
        let clean_warm_peak = get_peak_memory() as u64;
        clear_cache();
        let clean_post_cleanup = AllocatorState::capture_current();
        let cleanup_bounds =
            LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
        let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(&measured, &baseline)?;
        if !bernini_quality_passes(maximum_error, mean_error, rms_error) {
            return Err(format!(
                "Bernini warm repeat exceeded the determinism envelope: max={maximum_error:.6}, \
                 mean={mean_error:.6}, rms={rms_error:.6}"
            ));
        }
        reset_peak_memory();
        let repeat = bernini_output_frames(
            arm,
            generator
                .generate(&bernini_request(arm, geometry, fps, seed), &mut |_| {})
                .map_err(|error| format!("generate warm Bernini repeat: {error}"))?,
            fps,
        )?;
        let warm_peak = get_peak_memory() as u64;
        if !cleanup_bounds.allows_warm_peak(warm_peak) {
            return Err(format!(
                "Bernini warm repeat peaked at {warm_peak} bytes, above the clean warm control \
                 {clean_warm_peak} bytes plus 2%"
            ));
        }
        clear_cache();
        let warm_post_cleanup = AllocatorState::capture_current();
        if !cleanup_bounds.allows_retained(warm_post_cleanup) {
            return Err(format!(
                "Bernini warm repeat retained active/cache bytes {warm_post_cleanup:?} above the \
                 clean warm cleanup {clean_post_cleanup:?} plus {} bytes",
                cleanup_bounds.tolerance_bytes,
            ));
        }
        let (warm_maximum, warm_mean, warm_rms) = video_max_mean_rms_abs(&measured, &repeat)?;
        if !bernini_quality_passes(warm_maximum, warm_mean, warm_rms) {
            return Err("Bernini second warm repeat changed the deterministic output".to_owned());
        }
        Some(BerniniWarmPasses {
            baseline,
            clean_warm_peak,
            clean_post_cleanup,
            cleanup_bounds,
            warm_peak,
            warm_post_cleanup,
            maximum_error,
            mean_error,
            rms_error,
        })
    };

    // Arm-internal negative-mutation falsifiability check: a runtime_complete record keeps
    // `negativeMutation` null, so the breach is verified here and the numbers land in diagnostics.
    // Against the clean warm control where one was rendered, else against the measured output
    // itself: either way the mutation must breach the envelope the arm declares.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let mutation_reference = warm.as_ref().map_or(&measured, |warm| &warm.baseline);
    let (mutated_maximum, mutated_mean, mutated_rms) =
        video_max_mean_rms_abs(&mutated, mutation_reference)?;
    if bernini_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err("Bernini output mutation did not breach the determinism envelope".to_owned());
    }

    let renders_executed = match &warm {
        Some(_) => {
            "the measured render plus two unscoped warm repeats on the loaded provider".to_owned()
        }
        None => format!(
            "ONE measured render on the loaded provider and no warm pass ({})",
            protocol::VIDEO_WARM_PASSES_NOT_RUN
        ),
    };
    let lifecycle_blocker = format!(
        concat!(
            "this arm executes {}; it opens no memory-strategy request scope and injects no ",
            "calibration fault, so the scoped cancellation and authorized-error scenarios and ",
            "their recovery renders are unexecuted. The pinned provider does register ",
            "begin_request, so they are implementable — they are not run here, and this record ",
            "claims nothing about them. ",
            // sc-22738: stated in the same breath as the lifecycle exclusion because it is the
            // same kind of claim — what this record deliberately does NOT say.
            "Additionally: {}"
        ),
        renders_executed, BERNINI_ADMISSION_BLOCKER
    );
    let lifecycle_blocker = lifecycle_blocker.as_str();
    const BERNINI_QUALITY_CONTRACT: &str = "identical artifact, prompt, seed, geometry, frames, fps, tier and loaded provider contract; cold measured output versus two warm unscoped repeats, compared over every frame";
    let warm_repeat_scenario = match &warm {
        Some(_) => {
            json!({ "name": "warm_repeat", "result": "passed", "reason": "two warm repeats on the loaded provider reproduced the measured output frame-for-frame inside the declared envelope, within the clean warm peak and cleanup bounds" })
        }
        None => capture.not_run_warm_repeat_scenario()?,
    };
    let quality = match &warm {
        Some(warm) => json!({
            "contract": BERNINI_QUALITY_CONTRACT,
            "identicalInputs": true,
            "result": "passed",
            "maximumError": warm.maximum_error,
            "meanError": warm.mean_error,
            "rootMeanSquareError": warm.rms_error,
            "maximumErrorThreshold": BERNINI_MAX_THRESHOLD,
            "meanErrorThreshold": BERNINI_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": BERNINI_RMS_THRESHOLD,
        }),
        None => capture.not_run_quality(
            BERNINI_QUALITY_CONTRACT,
            (
                BERNINI_MAX_THRESHOLD,
                BERNINI_MEAN_THRESHOLD,
                BERNINI_RMS_THRESHOLD,
            ),
        )?,
    };
    let mut measurements = vec![
        ("preRungActiveAfterClear", "bytes", pre_rung_active),
        ("preRungCacheAfterClear", "bytes", pre_rung_cache),
        ("preGenerateActivePeak", "bytes", pre_generate.active),
        ("conditioningActivePeak", "bytes", conditioning.active),
        (
            "conditioningCloseActive",
            "bytes",
            conditioning_close.active,
        ),
        ("conditioningCloseCache", "bytes", conditioning_close.cache),
        ("denoiseActivePeak", "bytes", denoise.active),
        ("denoiseCloseActive", "bytes", denoise_close.active),
        ("denoiseCloseCache", "bytes", denoise_close.cache),
        ("decodeActivePeak", "bytes", decode.active),
        ("decodeCloseActive", "bytes", decode_close.active),
        ("decodeCloseCache", "bytes", decode_close.cache),
        (
            "overallAllocatorEnvelope",
            "bytes",
            overall.allocator_bytes(),
        ),
        ("predictedOverallCeiling", "bytes", predicted),
        ("warmPasses", "count", u64::from(capture.warm_passes)),
    ];
    // sc-22738: on the video lane no `lifecycleClean*` / `lifecycleWarmRepeat*` figure is emitted —
    // the warm passes were not run (`quality.warmPasses`), and an unmeasured figure is omitted,
    // never written as 0.
    if let Some(warm) = &warm {
        measurements.extend([
            ("lifecycleCleanWarmPeak", "bytes", warm.clean_warm_peak),
            (
                "lifecycleCleanPostCleanupActive",
                "bytes",
                warm.clean_post_cleanup.active,
            ),
            (
                "lifecycleCleanPostCleanupCache",
                "bytes",
                warm.clean_post_cleanup.cache,
            ),
            (
                "lifecycleCleanupTolerance",
                "bytes",
                warm.cleanup_bounds.tolerance_bytes,
            ),
            ("lifecycleWarmRepeatPeak", "bytes", warm.warm_peak),
            (
                "lifecycleWarmRepeatPostCleanupActive",
                "bytes",
                warm.warm_post_cleanup.active,
            ),
            (
                "lifecycleWarmRepeatPostCleanupCache",
                "bytes",
                warm.warm_post_cleanup.cache,
            ),
        ]);
    }
    measurements.extend([
        (
            "negativeMutationMaximumErrorPer255",
            "count",
            (mutated_maximum * 255.0).round() as u64,
        ),
        (
            "negativeMutationMeanErrorPer255",
            "count",
            (mutated_mean * 255.0).round() as u64,
        ),
        (
            "negativeMutationRootMeanSquareErrorPer255",
            "count",
            (mutated_rms * 255.0).round() as u64,
        ),
        (
            "loadShapeDeferred",
            "count",
            u64::from(load_shape == LoadShape::DeferredMaterialization),
        ),
        ("stagedTierBytes", "bytes", staged_tier_bytes),
        // sc-22738: the render receipts, published as measurements rather than as a top-level
        // `output` object the record schema rejects. The member this record is filed under is
        // `target.modelId`, so the dropped `output.modelId` is not lost.
        ("renderedFrames", "count", u64::from(expected_frames)),
        ("requestedFrames", "count", u64::from(geometry.frames)),
        ("outputFps", "count", u64::from(fps)),
    ]);
    // GATED, not `runtime_complete` (sc-22738). Runtime activation is exactly the claim that the
    // provider's admission gate accepted an exact-fit budget and rejected the two mutations, and
    // this capture asked it nothing — because production asks it nothing for this request. The
    // MEASUREMENT is unaffected and is carried in full: `observedMemory` and `predictedPeakBytes`
    // are the resident-load prices this anchor exists to publish, and `extract-memory-anchors.mjs`
    // reads them off this record exactly as it reads them off a runtime-complete one.
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": minimax_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": BERNINI_ADMISSION_BLOCKER },
            { "name": "unknown_budget", "result": "not_run", "reason": BERNINI_ADMISSION_BLOCKER },
            { "name": "stale_evidence", "result": "not_run", "reason": BERNINI_ADMISSION_BLOCKER },
            warm_repeat_scenario,
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ],
        "predictedPeakBytes": predicted_peaks_json,
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": quality,
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.resolved_path_fingerprint(arm, tier),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:bernini-dual-expert",
            "executed",
            [lifecycle_blocker.to_owned()],
            measurements,
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, arm.execution_path)?;
    Ok(fragment)
}
