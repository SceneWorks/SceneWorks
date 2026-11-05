//! MLX minimax capture owner.
use super::*;

/// The production calibration identity `mlx-gen-minimax-h3` mints for one `mlx:minimax_h3` numeric
/// tier — on BOTH catalog entries, because they are one provider loaded from one `LoadSpec`
/// (`video_jobs/minimax_h3.rs` stages the base `transformer` even for a ref2va job and the engine
/// resolves `transformer_ref` per render), so the loaded contract can only ever publish one
/// identity per tier and the anchors are told apart by their own `(modelId, mode)` key.
///
/// The engine keys on `(provider, ARTIFACT-proven tier)`: the calibrated tier retains the bare
/// `MEMORY_CALIBRATION_FINGERPRINT` and every other shipped tier receives that same key with its
/// tier token inserted after the family stem. The string names `eager` because the provider chose
/// that spelling before it grew a deferred loader — the RESOLVED shape travels beside it in
/// `MemoryCalibrationIdentity::load_shape`, not in the fingerprint, and this arm attests the
/// resolved one.
///
/// Written here as well as in the engine so the plan/arm binding is weights-free and provable on a
/// CPU-only host; the capture then refuses a loaded contract whose identity differs from this
/// table. It used to be ONE literal for all six MLX cells (sc-22737 review), which the engine
/// never mints for q4 or q8: four of the six cells could not return a record.
///
/// A tier outside [`MINIMAX_TIERS`] is refused BY NAME rather than interpolated into a plausible
/// identity no production load can return.
pub(super) fn minimax_calibration_fingerprint(tier: &str) -> Result<String, String> {
    if !MINIMAX_TIERS.contains(&tier) {
        return Err(format!(
            "the MLX MiniMax-H3 calibration family ships {MINIMAX_TIERS:?}; tier {tier:?} has no \
             production identity"
        ));
    }
    if tier == MINIMAX_CALIBRATED_TIER {
        return Ok(mlx_gen_minimax_h3::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT.to_owned());
    }
    Ok(format!(
        "minimax-h3-{tier}-mlx-staged-joint-av-eager-abi3-v1"
    ))
}

/// MiniMax-H3's own geometry envelope, which replaces the image arms' `frames == 1` refusal for this
/// arm. It is the same four-clause gate the pinned provider's `route_gate` applies, read off the
/// engine crate rather than transcribed:
///
/// * the temporal lattice — `frames ∈ LEGAL_FRAME_COUNTS`, i.e. `17n + 5` clamped to the released
///   checkpoint's hardcoded 5–15 s duration range. `T = 1` is off the lattice and does not render,
///   so a still geometry is refused here too;
/// * the spatial stride — `SPATIAL_STRIDE` (32: the VAE's 16x compression times the DiT's width
///   patch of 2). A 16-aligned canvas survives the VAE and then has an odd latent column count with
///   no patched representation at all;
/// * the canvas budget — `CANVAS_MAX_PIXELS` as a **PRODUCT**, not per edge. The published
///   resolution list contains 1536x672 and 1344x768, whose long edges differ by 192 px and whose
///   areas are identical; a per-edge cap would refuse the first and admit the second while both sit
///   exactly at the budget;
/// * `batch == 1` — the engine renders one clip per request.
pub(super) fn validate_minimax_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<MinimaxGeometry, String> {
    let lattice_frames = i32::try_from(frames)
        .ok()
        .filter(|frames| mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS.contains(frames))
        .ok_or_else(|| {
            format!(
                "{MINIMAX_LABEL} requires geometry.frames on the 17n+5 lattice {:?}, got {frames}",
                mlx_gen_minimax_h3::LEGAL_FRAME_COUNTS
            )
        })?;
    let stride = mlx_gen_minimax_h3::SPATIAL_STRIDE;
    if !width.is_multiple_of(stride) || !height.is_multiple_of(stride) {
        return Err(format!(
            "{MINIMAX_LABEL} requires geometry divisible by the {stride}px stride, got {width}x{height}"
        ));
    }
    let pixels = width.saturating_mul(height);
    let budget = mlx_gen_minimax_h3::CANVAS_MAX_PIXELS;
    if pixels > budget {
        return Err(format!(
            "{MINIMAX_LABEL} requires width*height within the {budget}px canvas budget, got \
             {width}x{height} ({pixels}px)"
        ));
    }
    let video_latent_frames = mlx_gen_minimax_h3::video_latent_num_frames(lattice_frames)
        .map_err(|error| format!("derive MiniMax-H3 video latent frames: {error}"))
        .and_then(|latents| {
            u32::try_from(latents)
                .map_err(|_| "MiniMax-H3 video latent frame count must fit u32".to_owned())
        })?;
    let audio_latent_frames =
        u32::try_from(mlx_gen_minimax_h3::audio_latent_num_frames(lattice_frames))
            .map_err(|_| "MiniMax-H3 audio latent count must fit u32".to_owned())?;
    Ok(MinimaxGeometry {
        width,
        height,
        frames,
        video_latent_frames,
        audio_latent_frames,
    })
}

/// Read the four declared geometry axes. Like the LTX arm this reads `frames` as a real value rather
/// than asserting it away, and pins `batch` to 1 before anything derived is computed.
pub(super) fn minimax_target_geometry(request: &Value) -> Result<MinimaxGeometry, String> {
    let geometry = protocol::planned_object(request, "/target/geometry")?;
    let axis = |name: &str| protocol::geometry_axis_u32(geometry, name);
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "{MINIMAX_LABEL} requires geometry.batch == 1 (the engine renders one clip per \
             request), got {batch}"
        ));
    }
    validate_minimax_geometry(axis("width")?, axis("height")?, axis("frames")?)
}

/// Defense-in-depth mirror of `validate_ltx_target`, plus the per-member target shape.
///
/// `run` dispatches by provider id today, but this arm hardcodes the MiniMax-H3 contract, so a
/// foreign caller must be refused BY NAME here — before any environment variable is read, any path
/// canonicalized, or any weight file opened. The reference surfaces are checked against the
/// MEMBER's declared count rather than pinned to zero: `ref2va` is a different checkpoint, so a
/// base record may never be filed against a reference-carrying target, and a `minimax_h3_ref`
/// record may never be filed against a reference-free one — which would silently be a second base
/// measurement wearing the reference entry's key.
pub(super) fn validate_minimax_target(
    request: &Value,
) -> Result<(MinimaxMember, MinimaxGeometry), String> {
    let target = protocol::planned_object(request, "/target")?;
    let provider = protocol::target_str(target, "provider")?;
    if provider != MINIMAX_PROVIDER {
        return Err(format!(
            "{MINIMAX_LABEL} does not implement provider {provider:?}"
        ));
    }
    let model_id = protocol::target_str(target, "modelId")?;
    let member = MINIMAX_MEMBERS
        .iter()
        .copied()
        .find(|member| member.model_id == model_id)
        .ok_or_else(|| {
            format!(
                "{MINIMAX_LABEL} serves catalog entries {:?}, got modelId {model_id:?}",
                MINIMAX_MEMBERS.map(|member| member.model_id)
            )
        })?;
    let mode = protocol::target_str(target, "mode")?;
    if mode != member.mode {
        return Err(format!(
            "{MINIMAX_LABEL} measures {model_id} in {:?} mode, got {mode:?}",
            member.mode
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(u64::from(member.reference_count)) {
                return Err(format!(
                    "{MINIMAX_LABEL} requires {field} == {} for {model_id}; the two catalog \
                     entries denoise on different DiT partitions and a record measured on one \
                     cannot be filed against the other",
                    member.reference_count
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(member.reference_count > 0) {
                return Err(format!(
                    "{MINIMAX_LABEL} requires {field} == {} for {model_id}; the two catalog \
                     entries denoise on different DiT partitions and a record measured on one \
                     cannot be filed against the other",
                    member.reference_count > 0
                ));
            }
        }
    }
    Ok((member, minimax_target_geometry(request)?))
}

/// Bind the fixture to the planned tier AND the full rendered geometry, recovering the cadence and
/// the seed. Unlike LTX's, this arm's `fps` has exactly one legal value, so the fixture cannot
/// declare a cadence the engine would refuse.
pub(super) fn planned_minimax_capture(
    request: &Value,
    member: MinimaxMember,
    tier: &str,
    geometry: MinimaxGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned_fixture(request)?;
    let prefix = format!(
        "{}-{tier}-{}x{}-f{}-fps",
        member.fixture_prefix, geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse MiniMax-H3 fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse MiniMax-H3 fixture seed {seed:?}: {error}"))?;
    if f64::from(fps) != mlx_gen_minimax_h3::MINIMAX_H3_FPS {
        return Err(format!(
            "planned.fixture declares fps {fps}, but the released MiniMax-H3 checkpoint generates \
             at {} fps only",
            mlx_gen_minimax_h3::MINIMAX_H3_FPS
        ));
    }
    if seed != MINIMAX_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the MiniMax-H3 calibration seed \
             {MINIMAX_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// Which tree a tier's text encoder comes from. Extracted from [`minimax_load_spec`] so the rule is
/// checkable without an environment: `q4`/`q8` are packed and rehosted, `bf16` is the dense upstream
/// checkpoint the rehost deliberately does not carry a second copy of.
pub(super) fn minimax_text_encoder_source(tier: &str) -> &'static str {
    if tier == "bf16" {
        MINIMAX_UPSTREAM_TEXT_ENCODER
    } else {
        MINIMAX_TIERED_TEXT_ENCODER
    }
}

/// Resolve and validate the `SCENEWORKS_MINIMAX_H3_*` environment family into a tier-exact load spec.
///
/// # The env contract, and why it is six variables rather than three
///
/// `mlx_gen_minimax_h3::model::load` reads TWO artifacts, so the arm resolves two triples:
///
/// * `SCENEWORKS_MINIMAX_H3_{REPOSITORY,REVISION,ROOT}` — the tiered rehost
///   [`protocol::MINIMAX_REPOSITORY`], whose `ROOT` is the **tier directory**
///   (`…/snapshots/<revision>/<tier>`), exactly like every other arm's `_ROOT`. It supplies
///   `transformer/`, its `transformer_ref/` sibling, and — at q4/q8 — `text_encoder/`.
/// * `SCENEWORKS_MINIMAX_H3_UPSTREAM_{REPOSITORY,REVISION,ROOT}` — the upstream
///   [`protocol::MINIMAX_UPSTREAM_REPOSITORY`] snapshot ROOT itself, with no tier component. The
///   loader probes `vae/config.json`, `audio_vae/config.json`, `tokenizer/tokenizer.json` and the
///   three `FL2VA/audio_vae/` documents under the spec's own weights root, and the rehost publishes
///   none of them — so the upstream root is what `spec.weights` must be, and the tiered components
///   are REDIRECTED onto it rather than the other way round.
///
/// # There is no text-encoder variable, deliberately
///
/// The text encoder's tier is DERIVED from the DiT's, because a tier is a whole-pipeline contract
/// rather than a per-component knob: `q4`/`q8` take `<tier>/text_encoder` from the rehost, `bf16`
/// takes the dense `text_encoder/` from upstream — which is exactly what the shipped manifest's
/// three `componentId: "text_encoder"` co-requisite rows declare, the bf16 one against the upstream
/// repository. A seventh variable would make the conditioning stage's tier a free axis that the
/// record's `artifact.variant` could not describe, and the conditioning stage is the tallest phase
/// this family has at every tier.
///
/// Both components are staged EXPLICITLY through `LoadSpec::with_component` even where the loader's
/// own fallback would find the same directory, because `ComponentBytes::resolve` reads the same map
/// to build `asset_facts` — an implicitly-resolved component is charged from a path this arm never
/// stated, and an under-declared conditioning floor admits a render that then OOMs.
pub(super) fn minimax_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
    load_shape: LoadShape,
) -> Result<MinimaxArtifact, String> {
    protocol::validate_plain_overlay_target(request, MINIMAX_PLAIN_EXECUTION_PATH)?;
    let repository = protocol::required_env("SCENEWORKS_MINIMAX_H3_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_MINIMAX_H3_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::MINIMAX_REPOSITORY)?;
    let tier_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MINIMAX_H3_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MINIMAX_H3_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &tier_root,
        &repository,
        &revision,
        tier,
        protocol::MINIMAX_REPOSITORY,
    )?;

    let upstream_repository = protocol::required_env("SCENEWORKS_MINIMAX_H3_UPSTREAM_REPOSITORY")?;
    let upstream_revision = protocol::required_env("SCENEWORKS_MINIMAX_H3_UPSTREAM_REVISION")?;
    protocol::validate_artifact_identity(
        &upstream_repository,
        &upstream_revision,
        protocol::MINIMAX_UPSTREAM_REPOSITORY,
    )?;
    let upstream_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MINIMAX_H3_UPSTREAM_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &upstream_root,
        &upstream_repository,
        &upstream_revision,
        protocol::MINIMAX_UPSTREAM_REPOSITORY,
    )?;

    let dit_root = tier_root.join(mlx_gen_minimax_h3::model::DIT_COMPONENT);
    let text_encoder_source = minimax_text_encoder_source(tier);
    let text_encoder_tree = if text_encoder_source == MINIMAX_UPSTREAM_TEXT_ENCODER {
        &upstream_root
    } else {
        &tier_root
    };
    let text_encoder_root =
        text_encoder_tree.join(mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT);

    let mut spec = LoadSpec::new(WeightsSource::Dir(upstream_root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(load_shape)
        .with_component(
            mlx_gen_minimax_h3::model::DIT_COMPONENT,
            WeightsSource::Dir(dit_root.clone()),
        )
        .with_component(
            mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(text_encoder_root.clone()),
        );
    // `spec.quantize` never packs anything at load: `model::load` RECONCILES it against the staged
    // tier's own `config.json` quantization marker and refuses a disagreement. Passing the planned
    // tier's quant is therefore an assertion about the directory on disk, not an instruction.
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok(MinimaxArtifact {
        repository,
        revision,
        upstream_repository,
        upstream_revision,
        upstream_root,
        dit_root,
        text_encoder_root,
        text_encoder_source,
        spec,
    })
}

/// The admission context for the MiniMax-H3 safety scenarios, describing the base t2va route.
///
/// `mode` IS AN EVIDENCE KEY, not a label. gen-core's `standard_memory_strategy_safety_check`
/// builds `MemoryDecodePolicyQuery { mode_key: context.mode.as_key(), .. }` and matches it against
/// each adopted decode-geometry record's own mode, so a probe run under one spelling cannot answer
/// a request asked under another. The runtime asks under `text_to_video` for every MiniMax render:
/// `video_jobs::wan` resolves the admission mode with
/// `sceneworks_core::video_request::payload_video_mode`, `video_admission` admits only
/// `"text_to_video"`, and it types that string with `memory_mode_from_mode_key`, which maps every
/// non-canonical key to [`MemoryMode::Other`]. This capture therefore carries the same `Other`
/// spelling `ltx_context` does.
///
/// This was [`MemoryMode::TextToImage`] until sc-18663, taken from the pinned provider's
/// `memory_strategy::routes()`, which really does spell t2va with the shared text-to-image key.
/// That list enumerates the provider's weights-free BEHAVIOR fixtures; it is not the key the
/// shipped video funnel queries under, and following it made this harness probe `text_to_image`
/// while the plan it validates ([`validate_minimax_target`] hard-requires `text_to_video`), the
/// record it emits, and the runtime all said `text_to_video`. The split was inert only because
/// MiniMax declares `decode_geometry_policy_authoritative: false` with an empty policy table, so
/// the lookup returns `Ok(None)` under either spelling — the day it declares a policy, the harness
/// would measure under one key and the runtime ask under the other, silently. Pinned by
/// `the_minimax_capture_context_binds_the_mode_key_the_runtime_video_route_sends`.
pub(super) fn minimax_context(
    member: MinimaxMember,
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: MinimaxGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        // A parameter only so the stale-evidence probe can pass a deliberate mismatch; every real
        // call site passes `calibration.fingerprint`.
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        // Both members' keys go through `MemoryMode::Other`: `video_admission` types every video
        // mode key with `memory_mode_from_mode_key`, which maps every non-canonical key — both
        // `text_to_video` and `reference_to_video` — to this variant.
        mode: MemoryMode::Other(member.mode.to_owned()),
        has_reference: member.reference_count > 0,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
            reference_count: member.reference_count,
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
        evidence_revision: format!("sc-18663@{}", protocol::INFERENCE_PIN),
    }
}

/// The measured request.
///
/// The CONDITIONING is what selects the DiT partition (upstream `MiniMaxH3Task::resolve`), so the
/// reference member carries exactly one image reference and the base member carries none. The
/// reference is synthetic and content-independent by construction — this measures residency, not
/// fidelity — and it is presented at the target geometry, which is the shape a real
/// reference-to-video request carries after the worker's own decode.
pub(super) fn minimax_request(
    member: MinimaxMember,
    geometry: MinimaxGeometry,
    fps: u32,
    seed: u64,
) -> GenerationRequest {
    let mut request = GenerationRequest {
        prompt: "a slow dolly along a rain-slick harbour wall at dusk, gulls calling, cinematic"
            .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: Some(fps),
        steps: Some(MINIMAX_STEPS),
        ..Default::default()
    };
    if member.reference_count > 0 {
        request.conditioning = vec![Conditioning::Reference {
            image: Image {
                width: geometry.width,
                height: geometry.height,
                pixels: protocol::synthetic_reference_rgb(geometry.width, geometry.height),
            },
            // The engine owns the ref2va conditioning strength; the request-level lever stays
            // unset, exactly as `resolve_minimax_h3_conditioning` leaves it.
            strength: None,
        }];
    }
    request
}

pub(super) fn minimax_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= MINIMAX_MAX_THRESHOLD
        && mean <= MINIMAX_MEAN_THRESHOLD
        && rms <= MINIMAX_RMS_THRESHOLD
}

/// One exact tuple per plan row. Non-numeric parameters (rung 4's `transformerWindowComponent`) stay
/// in the case but are not promoted to an axis, the shape `sdxl_runtime_complete_sweep` established:
/// an axis is a swept numeric range, and a component name is not one.
pub(super) fn minimax_complete_sweep(request: &Value) -> Result<Value, String> {
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

/// The `mlx:minimax_h3` arm (sc-18663) — the first JOINT AUDIO+VIDEO calibration lane.
///
/// It reads the registry contract under the exact t2va provider id, proves the loaded generator
/// exposes the byte-for-byte same contract, runs the four admission probes against the provider's
/// own registered `safety_check`, and then measures three phase peaks off the boundaries the shipped
/// `generate` already emits. No rung allowlist is hardcoded: the pinned contract's own
/// `validate_selection` decides which rungs are capturable, and it answers differently for the two
/// load shapes (rung 4 is `Implemented` only under `deferred_materialization`).
pub(super) fn run_minimax_h3(request: &Value) -> Result<Value, String> {
    let (member, geometry) = validate_minimax_target(request)?;
    // The lane's render plan (sc-22738): a video capture is ONE measured render, no warm pass.
    let capture = protocol::capture_policy(request)?.require_video(MINIMAX_LABEL)?;
    protocol::validate_plain_overlay_target(request, MINIMAX_PLAIN_EXECUTION_PATH)?;
    let load_shape = planned_load_shape(request)?;
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
                                            // The identity is the PLAN's tier's, and an unshipped tier is refused by name here — before the
                                            // fixture, the environment or any weight work.
    let expected_fingerprint = minimax_calibration_fingerprint(tier)?;
    let (fps, seed) = planned_minimax_capture(request, member, tier, geometry)?;
    // Fingerprint check 1 of 3: the PLAN against this arm's own expectation, before any environment
    // or weight work. A provider re-fingerprint must not silently reuse the epic's plan.
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/adapter calibration mismatch: plan={planned_fingerprint}, MLX MiniMax-H3 arm \
             implements {expected_fingerprint} for tier {tier}"
        ));
    }
    let artifact = minimax_load_spec(request, tier, &selection, load_shape)?;
    let spec = &artifact.spec;
    // Read the on-disk component footprints BEFORE the load, for the same reason the LTX arm does:
    // grounded in the artifact rather than in an allocator reading a broken staging would itself
    // corrupt — and a mis-staged directory then fails here, before any GPU work, rather than after
    // three full renders.
    let staged_dit_bytes = safetensors_bytes(&artifact.dit_root)?;
    let staged_text_encoder_bytes = safetensors_bytes(&artifact.text_encoder_root)?;
    let shared_video_vae_bytes = safetensors_bytes(&artifact.upstream_root.join("vae"))?;
    let shared_audio_vae_bytes = safetensors_bytes(&artifact.upstream_root.join("audio_vae"))?;

    let registry = mlx_gen_minimax_h3::provider_registry()
        .map_err(|error| format!("build MiniMax-H3 registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(MINIMAX_PROVIDER, spec)
        .map_err(|error| format!("read {MINIMAX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| {
            "pinned MLX MiniMax-H3 provider has no memory-strategy contract".to_owned()
        })?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned MiniMax-H3 contract rejected planned selection: {error}")
    })?;
    let strategy = attested_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned MiniMax-H3 contract has no calibration identity".to_owned())?;
    // Fingerprint check 2 of 3: the PROVIDER contract against this arm's expectation for the
    // plan's tier. The engine proves the tier from the DiT's own packed marker, so a mismatch here
    // is a staged root of another tier or a re-fingerprinted provider — never a plan typo, which
    // check 1 already caught.
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned MiniMax-H3 contract fingerprint changed: expected {expected_fingerprint} for \
             tier {tier}, got {}",
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
    // The contract resolves the SPEC's shape (`resolved_load_shape`), so the identity the record
    // will carry must be the shape the plan asked for and this arm executed.
    if calibration.load_shape != load_shape {
        return Err(format!(
            "pinned MiniMax-H3 contract resolved load shape {:?} for a plan that declares {:?}",
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

    // Admission mutation hygiene BEFORE the expensive load, through the provider's OWN registered
    // check: the gate must accept a fitting request (so the two rejections are not a blanket
    // refusal), reject an unknown/zero budget, and reject a mutated calibration fingerprint.
    let safety = |fingerprint: &str, total_bytes: u64, predicted: u64| {
        mlx_gen_minimax_h3::memory_strategy::safety_check(
            spec,
            &contract,
            &minimax_context(
                member,
                selection,
                calibration,
                fingerprint,
                geometry,
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
            "MiniMax-H3 admission rejected a fitting probe budget; the scenario rejections below \
             would be a blanket refusal, not evidence"
                .to_owned(),
        );
    }
    if !matches!(
        safety(&calibration.fingerprint, 0, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("MiniMax-H3 admission accepted an unknown/zero memory budget".to_owned());
    }
    if !matches!(
        safety("stale-minimax-h3-fingerprint", hardware_bytes, 1),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("MiniMax-H3 admission accepted stale calibration evidence".to_owned());
    }

    let generator = registry
        .load(MINIMAX_PROVIDER, spec)
        .map_err(|error| format!("load real MiniMax-H3 {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded MiniMax-H3 generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded MiniMax-H3 generator contract differs from the registry contract".to_owned(),
        );
    }

    // Three phase peaks off the boundaries the shipped `generate` already emits. The engine emits
    // FOUR — `Loading(TextEncoder)`, `Loading(Renderer)`, the first `Step` and `Decoding` — and this
    // arm cuts on the first, second and fourth so each recorded phase is exactly one of the
    // contract's declared `[Conditioning, Denoise, Decode]`:
    //
    // * conditioning = TE map → prompt encode → release, closed at `Loading(Renderer)`;
    // * denoise = the DiT map, the AdaLN precompute-and-evict, and every step, closed at `Decoding`;
    // * decode = both VAEs, closed when `generate` returns.
    //
    // ORDERING IS LOAD-BEARING at every boundary, and it is the ordering of the engine's own
    // `tests/te_tier_generate_stages.rs::rotate`: read `get_peak_memory()` AND the `active + cache`
    // footprint FIRST, and only then `reset_peak_memory()`. Reset early and the closing stage's
    // high-water leaks into the next one. `get_peak_memory` reports ACTIVE only, so the footprint is
    // read alongside it rather than instead — a drain that turns into a no-op migrates a shed
    // component from active into the allocator cache, where active alone cannot see it.
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
    // Live allocator readings at the two staging handoffs, distinct from the per-phase PEAKS.
    let conditioning_close = Cell::new(AllocatorState::default());
    let denoise_close = Cell::new(AllocatorState::default());
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    let (measured, output_fps, audio) = diagnostic_video_frames(
        generator
            .generate(
                &minimax_request(member, geometry, fps, seed),
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
            .map_err(|error| format!("generate measured MiniMax-H3 render: {error}"))?,
        MINIMAX_VIDEO_LABEL,
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
            "a synchronized MiniMax-H3 lifecycle phase reported a zero active peak; the engine \
             stopped emitting a boundary and the attribution collapsed"
                .to_owned(),
        );
    }
    // sc-22738 sweep: MiniMax-H3 is the one video family that publishes NO `VaeTiling` at all — it
    // is absent from `mlx_gen_catalog::vae_tiling`'s chain and from `vae_tiling_unmodelled_reason`
    // — so the shared decoded-frame rule has nothing to resolve here and is deliberately not
    // applied. The equality is still right, and for a stronger reason than arithmetic: the engine
    // asserts the same equality ITSELF immediately before returning
    // (`mlx-gen-minimax-h3/src/model.rs:1368-1373` and `:1589-1594`), its VAE strips the repeat-pad
    // tail in `decode_temporal` (`.../vae.rs:654-658`), and an off-`17n+5` request is refused
    // rather than snapped (`.../pipeline.rs:188-235`). This family cannot over-deliver.
    if measured.len() as u64 != u64::from(geometry.frames) {
        return Err(format!(
            "MiniMax-H3 rendered {} frames for a {}-frame request",
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != fps {
        return Err(format!(
            "MiniMax-H3 returned fps {output_fps} for a {fps} fps request"
        ));
    }
    // The soundtrack is half of what this family denoises; a record that did not observe one is not
    // a record of a joint A/V render.
    let audio = audio
        .filter(|audio| audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0)
        .ok_or_else(|| "MiniMax-H3 render returned no non-empty audio track".to_owned())?;
    let first = measured
        .first()
        .ok_or_else(|| "MiniMax-H3 render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("MiniMax-H3 render returned a degenerate first frame".to_owned());
    }

    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);
    // Video evidence charges the RESIDENT ACTIVE peak: a staged A/V provider leaves an enormous
    // elastic cache at its phase boundaries, and charging that would make a successful capture
    // inadmissible on its own capture host.
    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    // The same two ceilings `memory-calibration-harness.mjs#assertResidencyFitsHardware` applies —
    // checked HERE so a capture that cannot be admitted fails loudly during the campaign rather
    // than producing a well-formed record the harness rejects afterwards.
    if overall.active > hardware_bytes {
        return Err(format!(
            "MiniMax-H3 observed overall active {} bytes above the probed hardware memory \
             {hardware_bytes} bytes",
            overall.active
        ));
    }
    if overall.active > wired_limit_bytes {
        return Err(format!(
            "MiniMax-H3 observed overall active {} bytes above the probed wired ceiling \
             {wired_limit_bytes} bytes",
            overall.active
        ));
    }
    // Probe 4 of 4, on the LOADED generator and against the measured evidence: the calibrated
    // budget admits a request predicted to consume exactly it.
    let exact_fit = minimax_context(
        member,
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        predicted,
        predicted,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&exact_fit),
        MemorySafetyDecision::Accept
    ) {
        return Err("MiniMax-H3 admission rejected an exact-fit calibrated budget".to_owned());
    }
    // THE EXACT-FIT ACCEPT ABOVE IS NOT SELF-VALIDATING. The permanent pin publishes the
    // MiniMax-H3 `memory_strategy_contract` and overrides `memory_strategy_safety_check`, so this
    // loaded generator's accept is checked against the provider contract rather than a trait
    // default. The contract identity check after load is the first guard; this independent probe
    // stays meaningful if a future provider publishes a contract but regresses the check: the same
    // loaded generator must REJECT a zero budget, or its accept is not evidence.
    let mut unknown_budget = exact_fit.clone();
    unknown_budget.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown_budget),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err(
            "the loaded MiniMax-H3 generator accepted a zero/unknown budget, so its exact-fit \
             accept is a blanket accept rather than admission evidence"
                .to_owned(),
        );
    }

    // No warm pass: the video lane captures ONE measured render (sc-22738, `capture` above), so
    // there is no clean warm control to judge determinism against and no warm repeat to bound;
    // the receipt says so (`warm_repeat` not_run, `quality.warmPasses: 0`) instead of writing
    // zeros where `lifecycleClean*` / `lifecycleWarmRepeat*` used to be.

    // Arm-internal negative-mutation falsifiability check. A runtime_complete record must keep
    // `negativeMutation` null, so the breach is verified here — the capture FAILS if the envelope
    // cannot be breached — and the measured numbers land in diagnostics rather than in the field.
    // Against the measured clip itself: the mutation must breach the envelope the arm declares.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &measured)?;
    if minimax_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err(
            "MiniMax-H3 output mutation did not breach the determinism envelope".to_owned(),
        );
    }

    let lifecycle_blocker = format!(
        "this arm executes ONE measured render on the loaded provider and no warm pass ({}); it \
         opens no memory-strategy request scope and injects no calibration fault, so the scoped \
         cancellation and authorized-error scenarios and their recovery renders are unexecuted. \
         The pinned provider does register begin_request, so they are implementable — they are \
         not run here, and this record claims nothing about them",
        protocol::VIDEO_WARM_PASSES_NOT_RUN
    );
    let lifecycle_blocker = lifecycle_blocker.as_str();
    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        // From the CONTRACT's own calibration identity, never copied from the plan: a receipt may
        // only testify to the materialization shape its own run used (sc-16482).
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": artifact.repository,
            "resolvedRevision": artifact.revision,
            "variant": tier,
        },
        "sweep": minimax_complete_sweep(request)?,
        "scenarios": [
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": predicted },
            { "name": "unknown_budget", "result": "passed", "reason": "the registered MiniMax-H3 admission check rejected a zero/unknown budget before load" },
            { "name": "stale_evidence", "result": "passed", "reason": "the registered MiniMax-H3 admission check rejected a mutated calibration fingerprint before load" },
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
            "identical artifact, prompt, seed, geometry, frames, fps, steps, tier, staged components and loaded provider contract; the cold measured clip versus a warm repeat, compared over every frame",
            (MINIMAX_MAX_THRESHOLD, MINIMAX_MEAN_THRESHOLD, MINIMAX_RMS_THRESHOLD),
        )?,
        "negativeMutation": null,
        "loadability": {
            "result": "passed",
            "resolvedPathFingerprint": artifact.resolved_path_fingerprint(member, tier),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:minimax-h3-joint-av",
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
                ("videoLatentFrames", "count", u64::from(geometry.video_latent_frames)),
                ("audioLatentFrames", "count", u64::from(geometry.audio_latent_frames)),
                // sc-22738. The render receipts, in the ONE place the record schema has for them.
                // They were a top-level `output` object until the catalog campaign proved the
                // record is `additionalProperties: false` and rejected the whole bundle after the
                // render; `docs/generated/ltx-mlx-video-sc-18808.json` is the committed precedent
                // for publishing them here, under these names.
                ("renderedFrames", "count", u64::from(geometry.frames)),
                ("outputFps", "count", u64::from(fps)),
                ("audioTrackDecoded", "count", 1),
                ("audioSamples", "count", audio.samples),
                ("audioSampleRate", "count", u64::from(audio.sample_rate)),
                ("audioChannels", "count", u64::from(audio.channels)),
                ("loadShapeDeferred", "count", u64::from(load_shape == LoadShape::DeferredMaterialization)),
                ("textEncoderFromTierTree", "count", u64::from(artifact.text_encoder_source == MINIMAX_TIERED_TEXT_ENCODER)),
                ("stagedDitBytes", "bytes", staged_dit_bytes),
                ("stagedTextEncoderBytes", "bytes", staged_text_encoder_bytes),
                ("sharedVideoVaeBytes", "bytes", shared_video_vae_bytes),
                ("sharedAudioVaeBytes", "bytes", shared_audio_vae_bytes),
            ],
        ),
        "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(request, &mut fragment, MINIMAX_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}
