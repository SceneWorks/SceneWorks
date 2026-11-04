//! MLX krea realtime capture owner.
use super::*;

/// Krea Realtime 14B's own geometry envelope, which replaces the image arms' `frames == 1` refusal
/// for this arm. Every spatial clause is READ off the pinned engine crate's own advertised
/// descriptor ([`mlx_gen_krea_realtime::descriptor`]) rather than transcribed, so it cannot drift
/// from the provider a capture actually loads:
///
/// * the spatial grid — `Capabilities::size_floor`, which this engine declares as
///   `SizeFloor::RangeCheckedOnGrid { multiple: 16 }` (the z16 VAE's spatial stride of 8 times the
///   Wan DiT's 2x2 latent patch). An off-grid edge would survive the VAE and then render a silently
///   smaller clip through integer division;
/// * the per-edge range — `Capabilities::min_size ..= Capabilities::max_size`. This is a PER-EDGE
///   bound, not an area budget, and it is applied as one. **The engine has NO area/pixel cap**:
///   `crates/sceneworks-worker/src/pinned_engine_geometry.rs` lists `krea_realtime_14b` as
///   `PinnedAreaCap::EngineHasNone`. Refusing a large area here would be an invented gate that made
///   this harness reject geometries the shipped engine renders, so nothing below multiplies the
///   edges together;
/// * the temporal lattice — `4k + 1` frames with a floor of 5, the rule
///   `sceneworks_core::video_request::wan_frame_count` floors a raw count to
///   (`crates/sceneworks-core/src/video_request.rs:522-525`). `sceneworks-core` is a DEV-dependency
///   of this crate, so the arm cannot call that function; the test
///   `the_realtime_frame_lattice_is_sceneworks_cores_own_wan_rule` drives the real symbol over the
///   whole accepted and refused range so the transcription cannot drift;
/// * the latent ceiling — [`KREA_REALTIME_MAX_LATENT_FRAMES`], the `pub(crate)` `MAX_LATENT_FRAMES`
///   the crate enforces in `bounded_latent_frame_count`;
/// * `batch == 1` — read from `Capabilities::max_count`, which this engine declares as 1.
pub(super) fn validate_krea_realtime_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<KreaRealtimeGeometry, String> {
    let capabilities = mlx_gen_krea_realtime::descriptor().capabilities;
    let multiple = match capabilities.size_floor {
        mlx_gen::gen_core::SizeFloor::RangeCheckedOnGrid { multiple } => multiple,
        other => {
            return Err(format!(
                "{KREA_REALTIME_LABEL} expects the pinned engine to advertise a grid size floor, \
                 got {other:?}"
            ))
        }
    };
    if multiple == 0 || !width.is_multiple_of(multiple) || !height.is_multiple_of(multiple) {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires geometry on the engine's {multiple}px grid, got \
             {width}x{height}"
        ));
    }
    for (axis, value) in [("width", width), ("height", height)] {
        if value < capabilities.min_size || value > capabilities.max_size {
            return Err(format!(
                "{KREA_REALTIME_LABEL} requires geometry.{axis} within the engine's per-edge range \
                 {}..={}, got {value}",
                capabilities.min_size, capabilities.max_size
            ));
        }
    }
    let stride = KREA_REALTIME_TEMPORAL_STRIDE;
    if frames < stride + 1 || !(frames - 1).is_multiple_of(stride) {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires geometry.frames on the {stride}k+1 Wan lattice at or \
             above {}, got {frames}",
            stride + 1
        ));
    }
    let latent_frames = (frames - 1) / stride + 1;
    if latent_frames > KREA_REALTIME_MAX_LATENT_FRAMES {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires geometry.frames within the engine's \
             {KREA_REALTIME_MAX_LATENT_FRAMES} latent-frame maximum, but {frames} frames resolve to \
             {latent_frames} latent frames"
        ));
    }
    Ok(KreaRealtimeGeometry {
        width,
        height,
        frames,
        latent_frames,
        autoregressive_blocks: latent_frames.div_ceil(KREA_REALTIME_LATENT_FRAMES_PER_BLOCK),
    })
}

/// Read the four declared geometry axes. Like the LTX and MiniMax arms this reads `frames` as a real
/// value rather than asserting it away, and pins `batch` to the engine's own advertised `max_count`
/// before anything derived is computed.
pub(super) fn krea_realtime_target_geometry(
    request: &Value,
) -> Result<KreaRealtimeGeometry, String> {
    let geometry = protocol::planned(request)?
        .pointer("/target/geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let axis = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    let max_count = mlx_gen_krea_realtime::descriptor().capabilities.max_count;
    let batch = axis("batch")?;
    if batch != max_count {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires geometry.batch == {max_count} (the engine's advertised \
             max_count; it renders one clip per request), got {batch}"
        ));
    }
    validate_krea_realtime_geometry(axis("width")?, axis("height")?, axis("frames")?)
}

/// Defense-in-depth mirror of `validate_minimax_target`, plus this engine's t2v-specific target
/// shape.
///
/// `run` dispatches by provider id today, but this arm hardcodes the Krea Realtime contract, so a
/// foreign caller must be refused BY NAME here — before any environment variable is read, any path
/// canonicalized, or any weight file opened.
///
/// The reference surfaces are refused for a sharper reason than tidiness: this engine routes purely
/// on CONDITIONING (`pipeline::run`) — nothing routes to t2v, a `Conditioning::Reference` still
/// routes to i2v, a `Conditioning::VideoClip` routes to v2v. i2v additionally warms the causal KV
/// cache from the encoded reference and prepends it to the output, and v2v runs a
/// strength-controlled init, so both have a materially different memory profile from the t2v route
/// this plan declares. A record measured on t2v may never be filed against a reference-carrying
/// target.
pub(super) fn validate_krea_realtime_target(
    request: &Value,
) -> Result<KreaRealtimeGeometry, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let provider = target
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.provider must be a string".to_owned())?;
    if provider != KREA_REALTIME_PROVIDER {
        return Err(format!(
            "{KREA_REALTIME_LABEL} does not implement provider {provider:?}"
        ));
    }
    let model_id = target
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != KREA_REALTIME_PROVIDER {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires modelId {KREA_REALTIME_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?;
    if mode != "text_to_video" {
        return Err(format!(
            "{KREA_REALTIME_LABEL} requires reference-free text_to_video mode, got {mode:?}"
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "{KREA_REALTIME_LABEL} requires {field} == 0 when declared; a reference routes \
                     the engine to i2v, which warms the causal KV cache from the encoded still and \
                     cannot be recorded from the t2v route"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "{KREA_REALTIME_LABEL} requires {field} == false when declared; a reference \
                     routes the engine to i2v, which warms the causal KV cache from the encoded \
                     still and cannot be recorded from the t2v route"
                ));
            }
        }
    }
    krea_realtime_target_geometry(request)
}

/// Bind the fixture to the planned tier AND the full rendered geometry, recovering the cadence and
/// the seed. Like MiniMax's, this arm's `fps` has exactly one legal value, so the fixture cannot
/// declare a cadence the engine would never produce.
pub(super) fn planned_krea_realtime_capture(
    request: &Value,
    tier: &str,
    geometry: KreaRealtimeGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!(
        "krea-realtime-14b-mlx-{tier}-{}x{}-f{}-fps",
        geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse Krea Realtime fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Krea Realtime fixture seed {seed:?}: {error}"))?;
    if fps != KREA_REALTIME_FPS {
        return Err(format!(
            "planned.fixture declares fps {fps}, but the released Krea Realtime 14B checkpoint \
             generates at {KREA_REALTIME_FPS} fps only"
        ));
    }
    if seed != KREA_REALTIME_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the Krea Realtime calibration seed \
             {KREA_REALTIME_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// Resolve `SCENEWORKS_KREA_REALTIME_{REPOSITORY,REVISION,ROOT}` into a tier-exact load spec, in the
/// shape the SHIPPED WORKER builds.
///
/// The spec is `crates/sceneworks-worker/src/video_jobs/wan.rs::video_load_spec` applied to the
/// directory `crates/sceneworks-worker/src/video_jobs/krea_realtime.rs`'s
/// `resolve_krea_realtime_tier_dir_and_quant` resolves: a single tier subdirectory of the turnkey
/// snapshot, `Precision::Bf16`, and — the load-bearing part —
/// **`quantize = None` AT EVERY TIER**. That resolver states the reason in one line: *the tier on
/// disk IS the quantization*. Every published Krea Realtime file lives under a `q4/`/`q8/`/`bf16/`
/// prefix carrying its own packed `dit.safetensors` and quantization manifest, so passing a
/// `LoadSpec::quantize` would be a load-time recipe asserted over already-packed weights rather
/// than a description of what is being measured. There is no external text encoder and no adapter:
/// the tier tree is self-contained (`t2v::load_text_encoder` / `load_vae` open the stock dense Wan
/// companions from the same directory).
///
/// The root must end in the PLANNED tier's directory, so a stale `…/q4` export can never satisfy a
/// q8 or bf16 plan.
pub(super) fn krea_realtime_load_spec(tier: &str) -> Result<KreaRealtimeArtifact, String> {
    let repository = protocol::required_env("SCENEWORKS_KREA_REALTIME_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_KREA_REALTIME_REVISION")?;
    protocol::validate_artifact_identity(
        &repository,
        &revision,
        protocol::KREA_REALTIME_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_KREA_REALTIME_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_KREA_REALTIME_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        protocol::KREA_REALTIME_REPOSITORY,
    )?;
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    spec.precision = Precision::Bf16;
    Ok(KreaRealtimeArtifact {
        repository,
        revision,
        root,
        spec,
    })
}

pub(super) fn krea_realtime_target_surface(request: &Value) -> Result<KreaRealtimeSurface, String> {
    let target = protocol::planned(request)?
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target must be an object".to_owned())?;
    let mode = target
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.mode must be a string".to_owned())?
        .to_owned();
    let mut reference_count = 0_u32;
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            let declared = value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| {
                    format!("planned.target.{field} must be a non-negative integer when declared")
                })?;
            reference_count = reference_count.max(declared);
        }
    }
    Ok(KreaRealtimeSurface {
        mode,
        reference_count,
    })
}

/// Whether production would ask the Krea Realtime provider's admission gate anything at all about
/// this request. The twin of [`bernini_probes_admission`], and false for the same reason.
///
/// THE ENGINE SIDE. `mlx-gen-krea-realtime`'s registered `safety_check`
/// (`crates/media/mlx-gen/mlx-gen-krea-realtime/src/memory_strategy.rs:686-694` at inference
/// `3b922bac`) admits exactly two request modes — `image_to_video` and `video_to_video` — and
/// refuses every other key BY NAME (`crossed resident request mode`) before it reaches the shared
/// `default_memory_strategy_safety_check`. It then requires `reference_count == 1` and an
/// `evidence_revision` sealed under
/// `provider-resident-video-request-v3:krea_realtime_14b:mode=…:shape=…:`.
///
/// THE WORKER SIDE agrees, and since sc-22738 it agrees BY CONSTRUCTION rather than by accident.
/// `video_jobs/krea_realtime.rs` builds its `VideoGenInput` with `..VideoGenInput::default()`, but
/// the shared funnel overwrites that field from admission (`video_jobs/wan.rs`
/// `apply_video_admission_outcome`), so the `..default()` is not what keeps the context off this
/// request. The context `video_admission.rs` would build carries
/// `mode = MemoryMode::Other("text_to_video")`, `reference_count = 0` and an `evidence_revision`
/// that is a curve/anchor id or `video-estimate-floor-v1`
/// (`video_admission.rs#admit_video_generation_with_curves_and_profiles`), never that sealed
/// receipt — so `video_admission.rs#engine_declines_advisory_context` asks the loaded provider's
/// own `memory_strategy_safety_check` before attaching it, sees the refusal, and drops both the
/// context and the rung knobs. A reference-free t2v render therefore reaches the engine through
/// `memory_strategy::generate_with_scope`'s no-context early return; the gate is asked nothing.
///
/// sc-22738 probed that surface anyway, and all three `krea_realtime_14b:*:mlx` anchors died on the
/// FIRST probe — after a full 14B weight load — with "admission rejected a fitting probe budget".
/// The arm now asks the gate exactly what the worker asks it. It does NOT synthesize a
/// reference-carrying surface to make the gate answer: that would measure an admission decision the
/// product never makes for this request, and the anchor would then carry an admission
/// characterization no shipped path exercises.
pub(super) fn krea_realtime_probes_admission(mode: &str, reference_count: u32) -> bool {
    matches!(mode, "image_to_video" | "video_to_video") || reference_count > 0
}

pub(super) fn krea_realtime_request(
    geometry: KreaRealtimeGeometry,
    fps: u32,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: "a handheld shot drifting through a neon arcade at night, reflections on wet tile"
            .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: Some(fps),
        steps: Some(KREA_REALTIME_STEPS),
        ..Default::default()
    }
}

pub(super) fn krea_realtime_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= KREA_REALTIME_MAX_THRESHOLD
        && mean <= KREA_REALTIME_MEAN_THRESHOLD
        && rms <= KREA_REALTIME_RMS_THRESHOLD
}

/// One exact tuple per plan row, the shape `minimax_complete_sweep` established.
pub(super) fn krea_realtime_complete_sweep(request: &Value) -> Result<Value, String> {
    let parameters = protocol::strategy_parameters(request)?;
    let axes = parameters
        .iter()
        .filter_map(|(name, value)| value.as_u64().map(|value| (name, value)))
        .map(|(name, value)| json!({ "parameter": name, "testedValues": [value] }))
        .collect::<Vec<_>>();
    Ok(json!({
        "axes": axes,
        "cases": [{ "parameters": parameters, "result": "passed" }],
        "rangeVerified": true,
    }))
}

/// The `mlx:krea_realtime_14b` arm (sc-22735) — the third MLX video lane.
///
/// It reads the registry contract under the exact provider id, proves the loaded generator exposes
/// the byte-for-byte same contract, runs the admission probes against the provider's own registered
/// `safety_check`, and measures three phase peaks off the boundaries the shipped `generate` already
/// emits (`t2v.rs` emits `Loading(TextEncoder)`, `Loading(Renderer)`, per-step `Step`, and
/// `Decoding`). No rung allowlist is hardcoded: the pinned contract's own `validate_selection`
/// decides which rungs are capturable, and this provider registers a resident-only witness.
pub(super) fn run_krea_realtime(request: &Value) -> Result<Value, String> {
    let geometry = validate_krea_realtime_target(request)?;
    // The lane's render plan (sc-22738): a video capture is ONE measured render, no warm pass.
    let capture = protocol::capture_policy(request)?.require_video(KREA_REALTIME_LABEL)?;
    let surface = krea_realtime_target_surface(request)?;
    protocol::validate_plain_overlay_target(request, KREA_REALTIME_PLAIN_EXECUTION_PATH)?;
    let load_shape = planned_load_shape(request)?;
    if load_shape != LoadShape::EagerMaterialization {
        return Err(format!(
            "{KREA_REALTIME_LABEL} must use the production {} load shape, but the plan declares {}",
            protocol::LOAD_SHAPE_EAGER,
            load_shape_key(load_shape)
        ));
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
                                            // PRE-LOAD fingerprint conformance, before any environment or weight work.
    validate_planned_fingerprint_is_producible(KREA_REALTIME_PROVIDER, tier, request)?;
    let (fps, seed) = planned_krea_realtime_capture(request, tier, geometry)?;
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?
        .to_owned();

    let artifact = krea_realtime_load_spec(tier)?;
    let spec = &artifact.spec;
    // Read the on-disk footprint BEFORE the load, for the same reason the LTX and MiniMax arms do:
    // grounded in the artifact rather than in an allocator reading a broken staging would itself
    // corrupt, and a mis-staged tier fails here rather than after three full renders.
    let staged_tier_bytes = safetensors_bytes(&artifact.root)?;

    let registry = mlx_gen_krea_realtime::provider_registry()
        .map_err(|error| format!("build Krea Realtime registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(KREA_REALTIME_PROVIDER, spec)
        .map_err(|error| {
            format!("read {KREA_REALTIME_PROVIDER} memory-strategy contract: {error}")
        })?
        .ok_or_else(|| {
            "pinned MLX Krea Realtime provider has no memory-strategy contract".to_owned()
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned Krea Realtime contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned Krea Realtime contract has no calibration identity".to_owned())?;
    // The POST-LOAD binding the pre-load table only makes locally diagnosable: what the pinned
    // provider actually publishes for this `(provider, tier)`, compared against both the arm's own
    // expectation and the plan, so the two cannot agree with the arm separately while disagreeing
    // with each other.
    if let Some(expected) = mlx_production_fingerprint(KREA_REALTIME_PROVIDER, tier) {
        if calibration.fingerprint != expected {
            return Err(format!(
                "pinned Krea Realtime contract fingerprint changed at tier {tier:?}: expected \
                 {expected}, got {}",
                calibration.fingerprint
            ));
        }
    }
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if calibration.load_shape != load_shape {
        return Err(format!(
            "pinned Krea Realtime contract resolved load shape {:?} for a plan that declares {:?}",
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

    let generator = registry
        .load(KREA_REALTIME_PROVIDER, spec)
        .map_err(|error| format!("load real Krea Realtime {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded Krea Realtime generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded Krea Realtime generator contract differs from the registry contract".to_owned(),
        );
    }

    // The arm's admission decision, taken the way the WORKER takes it (sc-22738): see
    // [`krea_realtime_probes_admission`]. On every cell this arm can plan the answer is `false` —
    // the reference-free t2v route has no admission surface on either side of the seam — so nothing
    // is asked of the provider's safety check and the render below runs resident exactly as
    // production runs it. The record says so rather than claiming an admission result it never got.
    //
    // A `true` answer would mean the plan reached this arm with a reference-carrying Krea Realtime
    // target, which `validate_krea_realtime_target` has already refused BY NAME: i2v warms the
    // causal KV cache from the encoded still and v2v runs a strength-controlled init, so such a
    // request seals a different evidence identity and needs its own capture, not a probe
    // synthesized here. It is refused rather than assumed away.
    if krea_realtime_probes_admission(&surface.mode, surface.reference_count) {
        return Err(format!(
            "{KREA_REALTIME_PROVIDER}: production WOULD build a Krea Realtime memory context for \
             this surface, but this arm measures only the reference-free t2v ladder; a \
             reference-carrying anchor seals its own evidence identity and needs its own capture"
        ));
    }

    // Three phase peaks off the boundaries the shipped `generate` already emits, cut exactly the way
    // the MiniMax arm cuts them, and for the same reason: each recorded phase is exactly one of the
    // contract's declared `[Conditioning, Denoise, Decode]`.
    //
    // * conditioning = the UMT5 map, prompt encode, and release, closed at `Loading(Renderer)`;
    // * denoise = the DiT map plus every autoregressive chunk's few-step denoise AND the rolling
    //   causal KV cache the AR loop threads across chunks, closed at `Decoding`;
    // * decode = the z16 VAE, closed when `generate` returns.
    //
    // ORDERING IS LOAD-BEARING at every boundary: read `get_peak_memory()` AND the `active + cache`
    // footprint FIRST, and only then `reset_peak_memory()`. Reset early and the closing stage's
    // high-water leaks into the next one. `get_peak_memory` reports ACTIVE only, so the footprint is
    // read alongside it rather than instead — the KV cache is the largest elastic term this lane
    // has, and a drain that turns into a no-op migrates it from active into the allocator cache
    // where active alone cannot see it.
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
    let (measured, output_fps, audio) = diagnostic_video_frames(
        generator
            .generate(
                &krea_realtime_request(geometry, fps, seed),
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
            .map_err(|error| format!("generate measured Krea Realtime render: {error}"))?,
        KREA_REALTIME_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let decode_close = AllocatorState::capture_current();
    let pre_generate = pre_generate.get();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let conditioning_close = conditioning_close.get();
    let denoise_close = denoise_close.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized Krea Realtime lifecycle phase reported a zero active peak; the engine \
             stopped emitting a boundary and the attribution collapsed"
                .to_owned(),
        );
    }
    // sc-22738 sweep: Krea Realtime decodes through the SAME non-causal Wan z16 VAE that makes
    // Bernini and SCAIL-2 over-deliver — but it is the one engine that TRIMS. `t2v.rs:348-353`
    // drains the leading excess (`frames.drain(0..excess)`) against the request's own
    // `out_frames`, wired from the job at `t2v.rs:926`, before returning `GenerationOutput::Video`.
    // So the equality here is what production produces and must stay an equality; applying the
    // decoded-depth rule would over-predict by three frames and refuse a correct render.
    if measured.len() as u64 != u64::from(geometry.frames) {
        return Err(format!(
            "Krea Realtime rendered {} frames for a {}-frame request",
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != fps {
        return Err(format!(
            "Krea Realtime returned fps {output_fps} for a {fps} fps request"
        ));
    }
    // A pure video model: an audio track here would mean the output shape is not what this record
    // describes.
    if audio.is_some() {
        return Err(
            "Krea Realtime render returned an audio track; this engine is video-only".to_owned(),
        );
    }
    let first = measured
        .first()
        .ok_or_else(|| "Krea Realtime render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("Krea Realtime render returned a degenerate first frame".to_owned());
    }

    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    // Video evidence charges the RESIDENT ACTIVE peak, for the reason stated at the MiniMax arm.
    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    // The same two ceilings `memory-calibration-harness.mjs#assertResidencyFitsHardware` applies,
    // checked HERE so a capture that cannot be admitted fails loudly during the campaign rather
    // than producing a well-formed record the harness rejects afterwards.
    if overall.active > hardware_bytes {
        return Err(format!(
            "Krea Realtime observed overall active {} bytes above the probed hardware memory \
             {hardware_bytes} bytes",
            overall.active
        ));
    }
    if overall.active > wired_limit_bytes {
        return Err(format!(
            "Krea Realtime observed overall active {} bytes above the probed wired ceiling \
             {wired_limit_bytes} bytes",
            overall.active
        ));
    }
    // No exact-fit probe against the MEASURED evidence either, and for the same reason the three
    // pre-render scenarios are unexecuted: the gate this arm would ask refuses the reference-free
    // t2v surface by name, so an answer here would characterize a decision production never takes.
    // See [`KREA_REALTIME_ADMISSION_BLOCKER`].

    // No warm pass: the video lane captures ONE measured render (sc-22738, `capture` above), so
    // there is no clean warm control to judge determinism against and no warm repeat to bound;
    // the receipt says so (`warm_repeat` not_run, `quality.warmPasses: 0`) instead of writing
    // zeros where `lifecycleClean*` / `lifecycleWarmRepeat*` used to be.

    // Arm-internal negative-mutation falsifiability check: the capture FAILS if the envelope cannot
    // be breached, and the measured numbers land in diagnostics rather than in the field. Against
    // the measured clip itself: the mutation must breach the envelope the arm declares.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &measured)?;
    if krea_realtime_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "Krea Realtime output mutation did not breach the determinism envelope".to_owned(),
        );
    }

    let lifecycle_blocker = format!(
        concat!(
            "this arm executes ONE measured render on the loaded provider and no warm pass ({}); ",
            "it opens no memory-strategy request scope and injects no calibration fault, so the ",
            "scoped cancellation and authorized-error scenarios and their recovery renders are ",
            "unexecuted. This record claims nothing about them. ",
            // sc-22738: stated in the same breath as the lifecycle exclusion because it is the
            // same kind of claim — what this record deliberately does NOT say.
            "Additionally: {}"
        ),
        protocol::VIDEO_WARM_PASSES_NOT_RUN,
        KREA_REALTIME_ADMISSION_BLOCKER,
    );
    let lifecycle_blocker = lifecycle_blocker.as_str();
    // GATED, not `runtime_complete` (sc-22738). Runtime activation is exactly the claim that the
    // provider's admission gate accepted an exact-fit budget and rejected the two mutations, and
    // this capture asked it nothing — because production asks it nothing for this request. The
    // MEASUREMENT is unaffected and is carried in full: `observedMemory` and `predictedPeakBytes`
    // are the resident-load prices this anchor exists to publish, and `extract-memory-anchors.mjs`
    // reads them off this record exactly as it reads them off a runtime-complete one.
    let mut fragment = json!({
        "status": "gated",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan: a receipt may
        // only testify to the materialization shape its own run used (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": krea_realtime_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": KREA_REALTIME_ADMISSION_BLOCKER },
            { "name": "unknown_budget", "result": "not_run", "reason": KREA_REALTIME_ADMISSION_BLOCKER },
            { "name": "stale_evidence", "result": "not_run", "reason": KREA_REALTIME_ADMISSION_BLOCKER },
            capture.not_run_warm_repeat_scenario()?,
            { "name": "cancel", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "error", "result": "not_run", "reason": lifecycle_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ],
        "predictedPeakBytes": predicted_peaks.json(),
        "observedMemory": {
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "overall": overall.json(),
        },
        "quality": capture.not_run_quality(
            "identical artifact, prompt, seed, geometry, frames, fps, steps, tier and loaded provider contract; the cold measured clip versus a warm repeat, compared over every frame",
            (KREA_REALTIME_MAX_THRESHOLD, KREA_REALTIME_MEAN_THRESHOLD, KREA_REALTIME_RMS_THRESHOLD),
        )?,
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.resolved_path_fingerprint(tier),
        },
        "diagnostics": protocol::diagnostics(
            &format!("memory-mlx-adapter:krea-realtime-{}", tier),
            "executed",
            [lifecycle_blocker.to_owned()],
            [
                ("preRungActiveAfterClear", "bytes", pre_rung_active),
                ("preRungCacheAfterClear", "bytes", pre_rung_cache),
                ("preGenerateActivePeak", "bytes", pre_generate.active),
                ("conditioningActivePeak", "bytes", conditioning.active),
                ("conditioningCloseActive", "bytes", conditioning_close.active),
                ("conditioningCloseCache", "bytes", conditioning_close.cache),
                ("denoiseActivePeak", "bytes", denoise.active),
                ("denoiseCloseActive", "bytes", denoise_close.active),
                ("denoiseCloseCache", "bytes", denoise_close.cache),
                ("decodeActivePeak", "bytes", decode.active),
                ("decodeCloseActive", "bytes", decode_close.active),
                ("decodeCloseCache", "bytes", decode_close.cache),
                ("overallAllocatorEnvelope", "bytes", overall.allocator_bytes()),
                ("predictedOverallCeiling", "bytes", predicted),
                // sc-22738: no `lifecycleClean*` / `lifecycleWarmRepeat*` figure — the warm
                // passes were not run (`quality.warmPasses`), and an unmeasured figure is omitted,
                // never written as 0.
                ("warmPasses", "count", u64::from(capture.warm_passes)),
                ("negativeMutationMaximumErrorPer255", "count", (mutated_maximum * 255.0).round() as u64),
                ("negativeMutationMeanErrorPer255", "count", (mutated_mean * 255.0).round() as u64),
                ("negativeMutationRootMeanSquareErrorPer255", "count", (mutated_rms * 255.0).round() as u64),
                ("latentFrames", "count", u64::from(geometry.latent_frames)),
                ("autoregressiveBlocks", "count", u64::from(geometry.autoregressive_blocks)),
                // sc-22738: the render receipts, published as measurements rather than as a
                // top-level `output` object the record schema rejects. This route decodes no audio
                // track, so `audioTrackDecoded` is a measured 0 rather than an absent claim.
                ("renderedFrames", "count", u64::from(geometry.frames)),
                ("outputFps", "count", u64::from(fps)),
                ("audioTrackDecoded", "count", 0),
                ("loadSpecCarriesQuant", "count", u64::from(spec.quantize.is_some())),
                ("stagedTierBytes", "bytes", staged_tier_bytes),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(
        request,
        &mut fragment,
        KREA_REALTIME_PLAIN_EXECUTION_PATH,
    )?;
    Ok(fragment)
}
