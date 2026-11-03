//! CANDLE mage capture owner.
use super::*;

/// Whether this case is the `z_image_edit` route: the Turbo provider in `edit_image` mode. Every
/// other provider this adapter implements is measured text-to-image regardless of the mode
/// spelling, exactly as before; only the Turbo arm has a second mode it can actually execute.
pub(super) fn is_z_image_edit(request: &Value) -> Result<bool, String> {
    Ok(planned_provider(request)? == Z_IMAGE_TURBO_ID && planned_mode(request)? == "edit_image")
}

/// The `(env family, expected repository)` binding for Qwen-Image 2.1 at one tier (sc-24112): the
/// split-repo shape [`ideogram_five_rung_family`] has, with the halves swapped — `q4`/`q8` are the
/// SceneWorks re-host's tier subdirs and `bf16` is the upstream `Qwen/Qwen-Image-2.1` snapshot root.
/// A function for the same parser reason Ideogram's is.
pub(super) fn qwen_image_2_1_five_rung_family(
    tier: &str,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if tier == "bf16" {
        (
            QWEN_IMAGE_2_1_ID,
            QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH,
            "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REPOSITORY",
            "SCENEWORKS_QWEN_IMAGE_2_1_BF16_REVISION",
            "SCENEWORKS_QWEN_IMAGE_2_1_BF16_ROOT",
            protocol::QWEN_IMAGE_2_1_BF16_REPOSITORY,
        )
    } else {
        (
            QWEN_IMAGE_2_1_ID,
            QWEN_IMAGE_2_1_PLAIN_EXECUTION_PATH,
            "SCENEWORKS_QWEN_IMAGE_2_1_REPOSITORY",
            "SCENEWORKS_QWEN_IMAGE_2_1_REVISION",
            "SCENEWORKS_QWEN_IMAGE_2_1_ROOT",
            protocol::QWEN_IMAGE_2_1_REPOSITORY,
        )
    }
}

/// The Qwen-Image 2.1 Turbo twin of [`qwen_image_2_1_five_rung_family`] (sc-25907): the same split
/// over the distilled checkpoint's OWN repositories and `SCENEWORKS_QWEN_IMAGE_2_1_TURBO_*` env
/// families, so a Turbo plan can never be satisfied by base weights.
pub(super) fn qwen_image_2_1_turbo_five_rung_family(
    tier: &str,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if tier == "bf16" {
        (
            QWEN_IMAGE_2_1_TURBO_ID,
            QWEN_IMAGE_2_1_TURBO_PLAIN_EXECUTION_PATH,
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY",
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_REVISION",
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_BF16_ROOT",
            protocol::QWEN_IMAGE_2_1_TURBO_BF16_REPOSITORY,
        )
    } else {
        (
            QWEN_IMAGE_2_1_TURBO_ID,
            QWEN_IMAGE_2_1_TURBO_PLAIN_EXECUTION_PATH,
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REPOSITORY",
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_REVISION",
            "SCENEWORKS_QWEN_IMAGE_2_1_TURBO_ROOT",
            protocol::QWEN_IMAGE_2_1_TURBO_REPOSITORY,
        )
    }
}

pub(super) fn mage_registry_bypass(provider: &str) -> String {
    format!(
        "{provider} binds two artifact triples and stages two components; it is served by \
         load_mage_generator and must not reach the single-root five-rung loader"
    )
}

/// The Mage half of [`load_five_rung_generator`]. Split out because a Mage load binds TWO artifact
/// triples and stages two components, which the single-root env tuple below cannot express.
pub(super) fn load_mage_generator(
    request: &Value,
    arm: MageArm,
) -> Result<LoadedFiveRungGenerator, String> {
    validate_mage_mode(request, arm)?;
    let tier = planned_tier(request)?;
    validate_fixture_binds_tier_and_geometry(request)?;
    validate_mage_plan_identity(request, arm, tier)?;
    let load_shape = mage_planned_load_shape(request)?;
    let repository = protocol::required_env(arm.repository_env)?;
    let revision = protocol::required_env(arm.revision_env)?;
    protocol::validate_artifact_identity(&repository, &revision, arm.repository)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(arm.root_env)?))
        .map_err(|error| format!("canonicalize {}: {error}", arm.root_env))?;
    // The root must end in the PLANNED tier's directory, so a stale `…/q4` export cannot satisfy a
    // q8 or bf16 plan and quietly re-label another tier's peaks.
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        tier,
        arm.repository,
    )?;
    let components_repository =
        protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REPOSITORY")?;
    let components_revision = protocol::required_env("SCENEWORKS_MAGE_FLOW_COMPONENTS_REVISION")?;
    protocol::validate_artifact_identity(
        &components_repository,
        &components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let components_root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_MAGE_FLOW_COMPONENTS_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &components_root,
        &components_repository,
        &components_revision,
        protocol::MAGE_COMPONENTS_REPOSITORY,
    )?;
    let spec = mage_load_spec(tier, load_shape, root, &components_root)?;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let generator = catalog
        .media()
        .load(arm.provider, &spec)
        .map_err(|error| format!("load real {} {tier} generator: {error}", arm.provider))?;
    vram.end_load(load_sample);
    Ok((
        arm.provider,
        arm.execution_path,
        repository,
        revision,
        generator,
        vram,
    ))
}

/// The table lookup behind the six literal dispatch arms in [`plain_execution_path`] and
/// [`still_calibration_label`]. Infallible by construction — every caller passes a
/// [`MAGE_ARMS`] provider — but returns a `Result` so the dispatch arms stay expression-bodied
/// literals, which is what the derived capturability report can parse.
pub(super) fn mage_execution_path(provider: &str) -> Result<&'static str, String> {
    MAGE_ARMS
        .iter()
        .find(|arm| arm.provider == provider)
        .map(|arm| arm.execution_path)
        .ok_or_else(|| format!("no Candle Mage-Flow arm for provider {provider:?}"))
}

pub(super) fn mage_still_calibration(provider: &str) -> Result<&'static str, String> {
    MAGE_ARMS
        .iter()
        .find(|arm| arm.provider == provider)
        .map(|arm| arm.still_calibration)
        .ok_or_else(|| format!("no Candle Mage-Flow arm for provider {provider:?}"))
}

/// The Mage member this case names, or `None` when the plan is not a Mage case at all. Keyed on
/// `(provider, modelId)` like the MLX arm: the catalog id and the engine provider id are equal on
/// all six Mage rows (`engines.rs` `MODEL_TABLE`), and this asserts that rather than assuming it.
pub(super) fn mage_arm(request: &Value) -> Result<Option<MageArm>, String> {
    let provider = planned_provider(request)?;
    let Some(arm) = MAGE_ARMS.into_iter().find(|arm| arm.provider == provider) else {
        return Ok(None);
    };
    let model_id = protocol::planned(request)?
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    if model_id != arm.provider {
        return Err(format!(
            "the Candle Mage-Flow arm does not implement provider {provider:?} for model \
             {model_id:?}"
        ));
    }
    Ok(Some(arm))
}

/// The mode the ARM will actually render — `edit_image` on the three instruction editors.
pub(super) fn mage_mode(arm: MageArm) -> &'static str {
    if arm.edit {
        "edit_image"
    } else {
        "text_to_image"
    }
}

/// The plan's declared mode must be the one this member renders. The record's `mode` comes from the
/// PLAN while the reference and the admitted `MemoryMode` come from the ARM, so an unchecked
/// mismatch would emit a reference-conditioned render's peaks under a reference-free label.
pub(super) fn validate_mage_mode(request: &Value, arm: MageArm) -> Result<(), String> {
    let declared = planned_mode(request)?;
    let expected = mage_mode(arm);
    if declared != expected {
        return Err(format!(
            "{} renders {expected:?}, but the plan declares mode {declared:?}",
            arm.provider
        ));
    }
    Ok(())
}

/// The Mage fixture binds the member, the tier, the edge, the seed and the recipe's step count.
pub(super) fn validate_mage_fixture(
    request: &Value,
    arm: MageArm,
    tier: &str,
) -> Result<(), String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
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
    if seed != MAGE_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the seed {} renders at ({MAGE_SEED})",
            arm.provider
        ));
    }
    let steps = steps
        .parse::<u32>()
        .map_err(|error| format!("parse Mage-Flow fixture step count {steps:?}: {error}"))?;
    if steps != arm.steps {
        return Err(format!(
            "planned.fixture {fixture:?} must use this arm's {}-step calibration request",
            arm.steps
        ));
    }
    Ok(())
}

/// The `LoadSpec` one Mage capture opens, built exactly as the worker's generic Candle stream does.
///
/// * A Mage variant rehost ships the DiT ALONE. The text encoder and the VAE come from the shared
///   components snapshot, staged EXPLICITLY through `LoadSpec::with_component` under the two ids
///   `candle-gen-mage` advertises — this is the "Mage's split text-encoder/VAE component paths"
///   the worker's own selector comment names (`image_jobs/base.rs`).
/// * The quant is the PLANNED tier's. `candle-gen-mage` `memory_strategy::resolved_quant` reads
///   `spec.quantize` directly, so unlike the packed Diffusers snapshots on this lane the tier is
///   carried on the spec rather than read out of `transformer/config.json`.
/// * The offload policy is `Sequential`: the candle anchor rung is `staged_residency`
///   (`memory-calibration-harness.mjs` `ANCHOR_STRATEGY.candle`), which is exactly the rung the
///   worker selects `use_sequential` for (`image_jobs/base.rs` binds `OffloadPolicy::Sequential`
///   after the declaration pass).
/// * The load shape is the PLAN's, and the plan's is the worker's, per tier. Every Mage manifest
///   entry's `candle.memoryStrategyContract.implementations` declares
///   `bounded_transformer_residency`, and the six Candle Mage rows of
///   `memory_route_registry::RULES` (`ALL_TIERS` × `ALL_MODES`, `PLAIN`,
///   `requires_sequential_selection: false`) exist for every planned cell, so the worker's
///   `apply_declared_candle_image_load_shape` → `evaluate_declared_candle_load_shape` owns the
///   shape rather than leaving it at its default. That evaluator matches the declaration on the
///   BTR row's OWN `tiers`, and the generated BTR row lists `["bf16"]` only — `candle-gen-mage`
///   publishes BTR `Implemented` where `memory_strategy::streamable` AND
///   `transformer_has_device_format` hold, and the stage-1 dump found device-format blocks on the
///   bf16 snapshot alone. So bf16 is `Applied + DeferredMaterialization` (the engine is asked about
///   a Deferred candidate and `streamable` holds for this plain directory load), and q4/q8 are
///   `Refused + EagerMaterialization`; both regardless of the selected rung. The worker's own
///   `mage_candle_production_load_shape_is_deferred_on_bf16_and_eager_on_the_packed_tiers` drives
///   that evaluator over the real manifest entries and pins the 18 `mage_flow*:*:candle` plan rows
///   to it; `the_load_spec_is_the_shape_the_worker_loads` below pins THIS spec to those same rows,
///   and a capture re-asserts the loaded contract's `MemoryCalibrationIdentity::load_shape` against
///   the plan. The candle Mage identity is keyed on the resolved tier alone (`-v3`), so the shape is
///   not an identity axis and neither planned shape can name the other tier's cell.
pub(super) fn mage_load_spec(
    tier: &str,
    load_shape: LoadShape,
    root: PathBuf,
    components_root: &std::path::Path,
) -> Result<LoadSpec, String> {
    let tier_components = components_root.join(tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(load_shape)
        .with_component(
            protocol::MAGE_COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_TEXT_ENCODER)),
        )
        .with_component(
            protocol::MAGE_COMPONENT_VAE,
            WeightsSource::Dir(tier_components.join(protocol::MAGE_COMPONENT_VAE)),
        );
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(spec)
}

/// The production calibration identity the loaded Candle Mage generator publishes for one
/// `(member, tier)` cell — the table `candle-gen-mage::memory_strategy::production_calibration_fingerprint`
/// mints (inference PR 953): `mage-flow-cuda-<provider>-<tier>-shared-ladder-v3`, eighteen distinct
/// strings, the tier read off `resolved_quant(spec)` (the quant this arm carries on the spec).
/// Written here as well so the plan/arm binding is weights-free and holds at inference
/// `c6d6a4db`, whose engine still publishes the tier-free `-v2` string; the capture refuses a
/// loaded contract whose identity differs from the plan, so the two copies cannot drift unnoticed
/// once the epic's pin bump lands.
pub(super) fn mage_calibration_fingerprint(arm: MageArm, tier: &str) -> String {
    format!(
        "mage-flow-cuda-{}-{tier}-shared-ladder-v3",
        arm.provider.replace('_', "-")
    )
}

/// The load shape the plan declares for this cell, which the capture must execute under and then
/// re-assert against the loaded contract's identity. Deriving it from the selected rung would
/// silently rewrite the per-tier shape the worker binds (see [`mage_load_spec`]).
pub(super) fn mage_planned_load_shape(request: &Value) -> Result<LoadShape, String> {
    match protocol::planned(request)?
        .get("loadShape")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.loadShape must be a string".to_owned())?
    {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported planned.loadShape {other:?}")),
    }
}

/// The plan row must name the production identity this cell's loaded generator publishes —
/// checked against the weights-free table BEFORE the load, so a row still carrying the retired
/// tier-free `-v2` string (or an MLX identity) fails in milliseconds rather than after a
/// multi-gigabyte load.
pub(super) fn validate_mage_plan_identity(
    request: &Value,
    arm: MageArm,
    tier: &str,
) -> Result<(), String> {
    let planned_fingerprint = protocol::planned(request)?
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.calibrationFingerprint must be a string".to_owned())?;
    let expected_fingerprint = mage_calibration_fingerprint(arm, tier);
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, the {} {tier} \
             production identity is {expected_fingerprint}",
            arm.provider
        ));
    }
    Ok(())
}

/// The generation request one Mage capture renders — the worker's request shape for this member.
/// The edit members refuse a reference-free request (`candle-gen-mage` `edit_provider.rs`), and the
/// engine's route gate additionally requires `MemoryMode::Edit` with 1..=8 references
/// (`memory_strategy.rs` `route_is_supported`), so the single fitted reference is what makes an edit
/// capture measure the edit path rather than text-to-image under an edit label.
/// GUIDANCE AND THE NEGATIVE PROMPT ARE READ OFF THE LOADED DESCRIPTOR (sc-22738). The MLX twin of
/// this arm had every `mage_flow*_turbo` cell refused at request time — before any render — with
/// `unsupported: mage_flow_turbo: guidance is not supported`: the shared floor
/// `gen-core/src/generator.rs:3216-3221` rejects `req.guidance.is_some() && !supports_guidance`, and
/// `candle-gen-mage` publishes both flags per variant (`src/lib.rs:132-148` `generation_descriptor`,
/// `:939-940` for the edit variants) with the distilled members declaring `false`. Reading the
/// TABLE's `guidance` scale instead of the descriptor is what sent the knob anyway. Production asks
/// the descriptor (`image_jobs/base.rs` `resolve_guidance` :4632, `resolve_negative_prompt` :4870),
/// so this arm asks the descriptor; `arm.guidance` is the scale used only where one is accepted.
pub(super) fn mage_generation_request(
    arm: MageArm,
    capabilities: &Capabilities,
    width: u32,
    height: u32,
) -> GenerationRequest {
    GenerationRequest {
        prompt: if arm.edit {
            "replace the background with a plain grey studio backdrop".to_owned()
        } else {
            "a weathered brass astrolabe on a linen cloth, soft window light".to_owned()
        },
        // The distilled members run CFG genuinely off at guidance 1.0, at which the engine builds no
        // unconditional branch at all — and refuses a negative prompt outright.
        negative_prompt: capabilities
            .supports_negative_prompt
            .then(|| "blurry, distorted, text".to_owned()),
        width,
        height,
        count: 1,
        seed: Some(MAGE_SEED),
        steps: Some(arm.steps),
        guidance: capabilities.supports_guidance.then_some(arm.guidance),
        conditioning: if arm.edit {
            vec![Conditioning::Reference {
                image: Image {
                    width,
                    height,
                    // `strength` stays None: Mage's `edit_references` consumes the reference image
                    // itself and never reads a strength lever, so declaring one would state a
                    // parameter the render does not use.
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

pub(super) fn pulid_reference_image() -> Result<Image, String> {
    // A generated face already exercised by inference's sc-16956 PuLID render.
    // A synthetic color gradient cannot pass the production face detector.
    let decoded = image::load_from_memory(include_bytes!("../../fixtures/pulid-reference.png"))
        .map_err(|error| format!("decode bundled PuLID reference face: {error}"))?
        .into_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}
