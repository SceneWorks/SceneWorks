//! CANDLE qwen edit capture owner.
use super::*;

pub(super) fn qwen_edit_generation_recipe(arm: QwenEditArm) -> QwenEditRequest {
    QwenEditRequest {
        prompt: QWEN_EDIT_PROMPT.to_owned(),
        negative: String::new(),
        steps: arm.steps,
        guidance: if arm.lightning {
            1.0
        } else {
            QWEN_EDIT_GUIDANCE
        },
        lightning: arm.lightning,
        ..Default::default()
    }
}

pub(super) fn qwen_edit_arm(request: &Value) -> Result<QwenEditArm, String> {
    let planned = protocol::planned(request)?;
    let provider = planned_provider(request)?;
    let model_id = planned
        .pointer("/target/modelId")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.target.modelId must be a string".to_owned())?;
    match (provider, model_id) {
        (QWEN_EDIT_ID, id) if id == QWEN_EDIT_ARM.model_id => Ok(QWEN_EDIT_ARM),
        (QWEN_EDIT_ID, id) if id == QWEN_EDIT_LIGHTNING_ARM.model_id => {
            Ok(QWEN_EDIT_LIGHTNING_ARM)
        }
        (provider, model_id) => Err(format!(
            "the Candle Qwen edit arm does not implement provider {provider:?} for model {model_id:?}"
        )),
    }
}

/// The seed and step count this member's fixture binds, checked against the arm's own prefix, the
/// planned tier and the recipe's step count — the MLX arm's `planned_qwen_edit_seed` rule, on this
/// lane's fixture spelling.
pub(super) fn planned_qwen_edit_seed(
    request: &Value,
    arm: QwenEditArm,
    tier: &str,
) -> Result<u64, String> {
    let fixture = protocol::planned(request)?
        .get("fixture")
        .and_then(Value::as_str)
        .ok_or_else(|| "planned.fixture must be a string".to_owned())?;
    let prefix = format!("{}-{tier}-seed", arm.fixture_prefix);
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (seed, steps) = remainder
        .split_once("-step")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -step<count>"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse Qwen edit fixture seed {seed:?}: {error}"))?;
    let steps = steps
        .parse::<usize>()
        .map_err(|error| format!("parse Qwen edit fixture step count {steps:?}: {error}"))?;
    if steps != arm.steps {
        return Err(format!(
            "planned.fixture {fixture:?} must use this arm's {}-step calibration request",
            arm.steps
        ));
    }
    Ok(seed)
}

/// The built-in Lightning distill adapter, exactly as the worker stacks it and exactly as the engine
/// pins it: the one file in the pinned snapshot, one LoRA at scale 1.0, no pass scales and no MoE
/// expert. `validate_memory_artifact_recipe` refuses anything else by name, so this is asserted here
/// rather than left to the environment.
pub(super) fn qwen_edit_lightning_adapter(
    source: &QwenEditLightningSource,
) -> Result<AdapterSpec, String> {
    protocol::validate_artifact_identity(
        &source.repository,
        &source.revision,
        protocol::QWEN_EDIT_LIGHTNING_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root).map_err(|error| {
        format!("canonicalize SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT: {error}")
    })?;
    let path = root.join(protocol::QWEN_EDIT_LIGHTNING_FILE);
    if !path.is_file() {
        return Err(format!(
            "the Lightning distill LoRA is not at {}",
            path.display()
        ));
    }
    Ok(AdapterSpec::new(path, 1.0, AdapterKind::Lora))
}

pub(super) fn qwen_edit_lightning_source() -> Result<QwenEditLightningSource, String> {
    Ok(QwenEditLightningSource {
        repository: protocol::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REPOSITORY")?,
        revision: protocol::required_env("SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_REVISION")?,
        root: PathBuf::from(protocol::required_env(
            "SCENEWORKS_QWEN_EDIT_LIGHTNING_LORA_ROOT",
        )?),
    })
}

pub(super) fn qwen_edit_artifact_source(
    arm: QwenEditArm,
) -> Result<QwenEditArtifactSource, String> {
    Ok(QwenEditArtifactSource {
        repository: protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REPOSITORY")?,
        revision: protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_REVISION")?,
        root: PathBuf::from(protocol::required_env("SCENEWORKS_QWEN_IMAGE_EDIT_ROOT")?),
        lightning: arm.lightning.then(qwen_edit_lightning_source).transpose()?,
    })
}

/// The env-free half of the Candle Qwen edit load, so both of its load-time bindings are
/// unit-testable:
///
/// * the root must end in the PLANNED tier's directory. The engine independently pins the same
///   suffix (`exact_base_tier`), so this refusal only makes the diagnosis local instead of a load
///   failure three hundred lines later; and
/// * the built-in distill lands in `spec.adapters` exactly when the arm is the Lightning member —
///   the ONE thing that makes that member Lightning at load time. Every downstream overlay claim in
///   [`run_qwen_edit`] is read back off `spec.adapters` rather than off `arm.lightning`, so a record
///   can never assert an overlay the load did not carry.
///
/// The spec deliberately leaves `offload_policy` at the gen-core default `Resident`, which is what
/// the worker's own `provider_load_spec` for this lane produces (`qwen_edit_candle.rs` never sets
/// it; request-scoped residency travels in `GenerationMemory.stage_residency` instead).
pub(super) fn qwen_edit_load_spec(
    arm: QwenEditArm,
    tier: &str,
    source: &QwenEditArtifactSource,
    load_shape: LoadShape,
) -> Result<QwenEditArtifact, String> {
    protocol::validate_artifact_identity(
        &source.repository,
        &source.revision,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(&source.root)
        .map_err(|error| format!("canonicalize SCENEWORKS_QWEN_IMAGE_EDIT_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &source.repository,
        &source.revision,
        tier,
        protocol::QWEN_EDIT_REPOSITORY,
    )?;
    let adapters = if arm.lightning {
        let lightning = source.lightning.as_ref().ok_or_else(|| {
            format!(
                "{} is the Lightning member but no distill snapshot was supplied",
                arm.model_id
            )
        })?;
        vec![qwen_edit_lightning_adapter(lightning)?]
    } else {
        Vec::new()
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_load_shape(load_shape)
        .with_adapters(adapters)
        .with_resolved_route(arm.model_id.to_owned());
    // The edit loader REQUIRES the tier's quant to be stated and to equal the packed snapshot's
    // (`exact_base_tier` + the `spec.quantize != loaded_quant` refusal) — unlike the txt2img Qwen
    // loader, which infers it from `transformer/config.json` and rejects a stated one.
    if let Some(quant) = numeric_tier(tier)?.quant {
        spec = spec.with_quant(quant);
    }
    Ok(QwenEditArtifact { root, spec })
}

/// The materialization shape the plan declares, as a typed `LoadShape`. The edit contract ECHOES
/// `spec.load_shape` back (`candle-gen-qwen-image` `memory_strategy.rs`), so the plan's declaration
/// is what the capture must execute under, and the echo is then re-asserted against it — deriving it
/// from the selected rung would silently rewrite a declared production shape (sc-16482).
pub(super) fn qwen_edit_planned_load_shape(request: &Value) -> Result<LoadShape, String> {
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

/// The one reference every Qwen edit capture conditions on, at the request geometry — the shape the
/// worker hands the engine after `fit_edit_references`. The engine hard-requires at least one
/// (`edit.rs` `encode_conditioning`), which is what makes this capture measure the edit path (the VL
/// vision tower plus the dual-latent VAE encode) rather than text-to-image under an edit label.
pub(super) fn qwen_edit_reference(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: protocol::synthetic_reference_rgb(width, height),
    }
}

/// One Qwen-Image-Edit-2511 anchor capture on the Candle lane, on either catalog id, at any shipped
/// tier.
///
/// E4: the model comes from `QwenEdit::load_with_memory_context` — the exact call the worker makes
/// (`image_jobs/qwen_edit_candle.rs`), with the same `QwenEditPaths`, the same admitted `LoadSpec`
/// (tier quant, resolved route and the Lightning adapter stack) and the same `MemoryRunContext` —
/// and the render is `generate_with_memory_context`, the worker's own generate. The bespoke edit
/// provider is deliberately NOT a registered generator (`edit.rs`: "driven **directly** by the
/// worker … the registered `qwen_image` descriptor stays txt2img-only"), so it cannot ride
/// `load_five_rung_generator`; going through the catalog here would measure the txt2img provider.
pub(super) fn run_qwen_edit(request: &Value) -> Result<Value, String> {
    let arm = qwen_edit_arm(request)?;
    protocol::validate_exact_overlay_target(request, arm.overlay, arm.execution_path)?;
    protocol::validate_still_geometry(request, arm.still_calibration)?;
    let tier = planned_tier(request)?;
    let seed = planned_qwen_edit_seed(request, arm, tier)?;
    let (width, height) = protocol::target_geometry(request)?;
    let source = qwen_edit_artifact_source(arm)?;
    let repository = source.repository.clone();
    let revision = source.revision.clone();
    let selection = planned_selection(request)?;
    let planned_load_shape_value = qwen_edit_planned_load_shape(request)?;
    // The root must end in the PLANNED tier's directory, and the distill is attached here, on
    // exactly the Lightning member.
    let QwenEditArtifact { root, mut spec } =
        qwen_edit_load_spec(arm, tier, &source, planned_load_shape_value)?;
    // Every overlay claim below is read off the stack the LOAD carries, never off `arm.lightning`:
    // if the attachment in `qwen_edit_load_spec` were ever lost, the record must say so rather than
    // assert a distill that never participated.
    let adapters = spec.adapters.clone();
    let loaded_adapters = adapters.len();
    spec.prepare_file_sources()
        .map_err(|error| format!("prepare Qwen edit file sources: {error}"))?;

    // The contract is read weights-free from the registered memory surface, so the planned
    // fingerprint and load shape are checked BEFORE a 28-57 GB load.
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let contract = catalog
        .media()
        .memory_strategy_contract(QWEN_EDIT_ID, &spec)
        .map_err(|error| format!("read {QWEN_EDIT_ID} memory-strategy contract: {error}"))?
        .ok_or_else(|| format!("{QWEN_EDIT_ID} has no memory-strategy contract"))?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {QWEN_EDIT_ID} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("pinned {QWEN_EDIT_ID} provider has no calibration identity"))?;
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
    if planned_load_shape_value != calibration.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(planned_load_shape_value),
            load_shape_key(calibration.load_shape)
        ));
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let stage_residency = matches!(
        selection.strategy,
        MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency
    );
    let context = MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Edit,
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
        // `validate_edit_route` requires this to be exactly `Some("lora")` when the load carries
        // adapters and `None` when it does not — so it is derived from the spec's own stack.
        reference_resources: None,
        overlay: (loaded_adapters > 0).then(|| "lora".to_owned()),
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-22728@{}", protocol::INFERENCE_PIN),
    };

    let mut vram = certifying_vram_probe();
    let load_sample = vram.phase();
    let model = QwenEdit::load_with_memory_context(
        &QwenEditPaths {
            root,
            text_encoder: None,
            adapters,
            // Compatibility-only load field; residency is request-scoped, exactly as the worker
            // passes it.
            offload_policy: OffloadPolicy::Resident,
        },
        &spec,
        &context,
    )
    .map_err(|error| format!("load real {} {tier} provider: {error}", arm.model_id))?;
    vram.end_load(load_sample);

    let generation = QwenEditRequest {
        width,
        height,
        seed,
        stage_residency,
        memory: Some(GenerationMemory {
            stage_residency,
            tile_vae_decode: selection.parameters.decode_tile_edge.is_some(),
            chunk_attention: selection.parameters.attention_chunk_size.is_some(),
            stream_transformer_blocks: selection.parameters.transformer_window_size.is_some(),
            decode_tile_edge: selection.parameters.decode_tile_edge,
            decode_overlap: selection.parameters.decode_overlap,
            attention_chunk_size: selection.parameters.attention_chunk_size,
            transformer_window_size: selection.parameters.transformer_window_size,
            ..Default::default()
        }),
        ..qwen_edit_generation_recipe(arm)
    };
    let references = [qwen_edit_reference(width, height)];

    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut conditioning_peak_gb = None;
    let mut denoise_peak_gb = None;
    let mut decode_peak_gb = None;
    let result =
        model.generate_with_memory_context(&context, &generation, &references, &mut |progress| {
            let boundary = match progress {
                Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                    protocol::ReferenceBoundary::RendererLoad
                }
                Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
                Progress::Decoding => protocol::ReferenceBoundary::Decoding,
                _ => return,
            };
            let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary)
            else {
                return;
            };
            let peak = phase_sample.take().map(|sample| vram.end_observed(sample));
            match phase {
                MemoryPhase::Conditioning => conditioning_peak_gb = peak,
                MemoryPhase::Denoise => denoise_peak_gb = peak,
                MemoryPhase::Decode => decode_peak_gb = peak,
            }
            phase = memory_phase(next);
            phase_sample = Some(vram.phase());
        });
    if let Some(sample) = phase_sample.take() {
        let terminal_peak_gb = vram.end_observed(sample);
        match phase {
            MemoryPhase::Conditioning => conditioning_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Denoise => denoise_peak_gb = Some(terminal_peak_gb),
            MemoryPhase::Decode => decode_peak_gb = Some(terminal_peak_gb),
        }
    }
    vram.end_gen(generation_sample);
    let image =
        result.map_err(|error| format!("{} edit generation failed: {error}", arm.model_id))?;
    if image.width != width || image.height != height {
        return Err(format!(
            "{} returned {}x{}, not the requested {width}x{height}",
            arm.model_id, image.width, image.height
        ));
    }
    let conditioning_bytes = decimal_gb_to_bytes(conditioning_peak_gb.ok_or_else(|| {
        format!(
            "{} capture did not expose a conditioning boundary",
            arm.model_id
        )
    })?);
    let denoise_bytes =
        decimal_gb_to_bytes(denoise_peak_gb.ok_or_else(|| {
            format!("{} capture did not expose a denoise boundary", arm.model_id)
        })?);
    let decode_bytes = decimal_gb_to_bytes(
        decode_peak_gb
            .ok_or_else(|| format!("{} capture did not complete decode", arm.model_id))?,
    );
    let overall_bytes = conditioning_bytes.max(denoise_bytes).max(decode_bytes);

    let blocker = concat!(
        "sc-22728 anchor capture measures exact per-phase memory and strategy identity for the ",
        "Candle Qwen-Image-Edit-2511 lane; it intentionally remains gated because this run does ",
        "not repeat the full promotion-quality, negative-mutation, and lifecycle scenario suite"
    );
    let sweep = protocol::reference_sweep(request, "passed")?;
    let parts = || protocol::PlainGatedFragment {
        artifact: artifact(&repository, &revision, tier),
        sweep: sweep.clone(),
        blocker,
        quality: json!({ "result": "not_run" }),
        negative_mutation: Value::Null,
        loadability: json!({
            "result": "passed",
            "resolvedPathFingerprint": loadability_fingerprint(&repository, &revision, tier),
        }),
        diagnostics: protocol::diagnostics(
            &format!("memory-candle-adapter:{}-anchor", arm.slug),
            "executed",
            [blocker.to_owned()],
            [
                ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                ("decodeDevicePeakDelta", "bytes", decode_bytes),
                ("overallDevicePeakDelta", "bytes", overall_bytes),
                ("referenceImages", "count", 1),
                ("builtInAdapters", "count", loaded_adapters as u64),
            ],
        ),
    };
    let mut fragment = if loaded_adapters > 0 {
        protocol::overlay_gated_fragment(
            request,
            arm.overlay,
            arm.execution_path,
            "the built-in lightx2v Lightning distill LoRA was folded into the MMDiT at load and participated in the measured render",
            parts(),
        )?
    } else {
        protocol::plain_gated_fragment(request, arm.execution_path, parts())?
    };
    fragment["strategy"] = strategy;
    fragment["loadShape"] = json!(load_shape_key(calibration.load_shape));
    fragment["observedMemory"] = json!({
        "conditioning": cuda_phase_metrics(conditioning_bytes),
        "denoise": cuda_phase_metrics(denoise_bytes),
        "decode": cuda_phase_metrics(decode_bytes),
        "overall": cuda_phase_metrics(overall_bytes),
    });
    Ok(fragment)
}
