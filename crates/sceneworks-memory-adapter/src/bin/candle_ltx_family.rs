//! CANDLE ltx capture owner.
use super::*;

pub(super) fn ltx25_planned_load_shape(
    request: &Value,
    transformer_variant: protocol::Ltx25TransformerVariant,
) -> Result<LoadShape, String> {
    let declared = protocol::planned_load_shape(request)?;
    transformer_variant.validate_load_shape(declared)?;
    match declared {
        protocol::LOAD_SHAPE_EAGER => Ok(LoadShape::EagerMaterialization),
        protocol::LOAD_SHAPE_DEFERRED => Ok(LoadShape::DeferredMaterialization),
        other => Err(format!("unsupported LTX-2.5 Candle loadShape {other:?}")),
    }
}

pub(super) fn ltx25_official_dev_adapter(revision: &str) -> Result<AdapterSpec, String> {
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_DISTILL_LORA_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_DISTILL_LORA_ROOT: {error}"))?;
    protocol::validate_huggingface_revision_root(
        &root,
        protocol::LTX_2_5_REPOSITORY,
        revision,
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let path = root.join(LTX25_DISTILL_LORA_RELATIVE_PATH);
    if !path.is_file() {
        return Err(format!(
            "LTX-2.5 dev capture requires the pinned official stage-two refinement LoRA at {}",
            path.display()
        ));
    }
    // The base scale MUST be the stage-one scale (0.0), not 1.0: production's
    // `resolve_ltx_distill_adapter` builds `AdapterSpec::new(path, stage1, ..)` with the manifest's
    // required `[0, 1]` contract, and the MLX capture arm does the same. gen_core's
    // `adapter_stack_identity` digests the scale bits into the admission overlay, so capturing at
    // 1.0 would mint evidence under an overlay identity no production request can ever present.
    Ok(AdapterSpec::new(path, 0.0, AdapterKind::Lora).with_pass_scales(vec![0.0, 1.0]))
}

pub(super) fn ltx25_load_spec(
    request: &Value,
    target: &protocol::Ltx25CandleTarget,
) -> Result<Ltx25LoadPlan, String> {
    let load_shape = ltx25_planned_load_shape(request, target.transformer_variant)?;
    let repository = protocol::required_env("SCENEWORKS_LTX25_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX25_REVISION")?;
    protocol::validate_ltx25_artifact_identity(&repository, &revision)?;
    let inventory_sha256 = protocol::required_env("SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256")?;
    protocol::validate_lowercase_sha256(
        &inventory_sha256,
        "SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256",
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX25_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX25_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_subpath(
        &root,
        &repository,
        &revision,
        &[target.transformer_variant.as_str(), target.tier.as_str()],
        protocol::LTX_2_5_REPOSITORY,
    )?;
    let adapters = if target
        .transformer_variant
        .requires_official_refinement_lora()
    {
        vec![ltx25_official_dev_adapter(&revision)?]
    } else {
        Vec::new()
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(load_shape);
    if let Some(quant) = numeric_tier(&target.tier)?.quant {
        spec = spec.with_quant(quant);
    }
    if target.decoder == protocol::Ltx25Decoder::DiffVae {
        spec = spec.with_component(
            LTX25_DIFFUSION_VAE_COMPONENT,
            WeightsSource::File(root.join("vae_diffusion_decoder.safetensors")),
        );
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters.clone());
    }
    Ok(Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    })
}

/// Execute a real selected LTX-2.5 provider path while leaving promotion decisions outside this
/// apparatus. The full published ladder — `q4`, `q8`, `bf16` — is executable here, symmetric with
/// the MLX arm. Candle's distinct NVFP4 evaluation selectors still need an inference-owned
/// producer and are never aliased to an ordinary bundle tier here.
pub(super) fn run_ltx25_capture(request: &Value) -> Result<Value, String> {
    protocol::validate_plain_overlay_target(request, LTX25_EXECUTION_PATH)?;
    let target = protocol::ltx25_candle_target(request)?;
    let Ltx25LoadPlan {
        repository,
        revision,
        inventory_sha256,
        load_shape,
        adapters,
        spec,
    } = ltx25_load_spec(request, &target)?;
    let catalog =
        runtime_cuda::catalog().map_err(|error| format!("build CUDA catalog: {error}"))?;
    let mut vram = VramProbe::start_rendered().assert_idle(1.0);
    let load_sample = vram.phase();
    let generator = catalog.media().load(LTX25_ID, &spec).map_err(|error| {
        format!(
            "load real {LTX25_ID} {}/{}/{} generator: {error}",
            target.transformer_variant.as_str(),
            target.decoder.as_str(),
            target.tier
        )
    })?;
    vram.end_load(load_sample);
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no memory-strategy contract"))?;
    let selection = planned_selection(request)?;
    contract.validate_selection(&selection).map_err(|error| {
        format!("pinned {LTX25_ID} provider rejected planned selection: {error}")
    })?;
    let strategy = measured_strategy(
        request,
        &selection,
        &contract.engaged_composition(selection.strategy),
    )?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| format!("loaded {LTX25_ID} has no calibration identity"))?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    if load_shape != calibration.load_shape {
        return Err(format!(
            "plan/provider load-shape mismatch: plan={}, pinned provider={}",
            load_shape_key(load_shape),
            load_shape_key(calibration.load_shape)
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
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: target.width,
            height: target.height,
            batch: 1,
            frames: target.frames,
            reference_count: 0,
        },
        reference_resources: None,
        overlay: adapter_stack_identity(&adapters),
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-18783-adapter@{}", protocol::INFERENCE_PIN),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .map_err(|error| format!("begin {LTX25_ID} capture scope: {error}"))?
        .ok_or_else(|| format!("{LTX25_ID} selection did not create a provider scope"))?;
    let parameters = context.selection.parameters;
    match (parameters.decode_tile_edge, parameters.decode_overlap) {
        (Some(edge), Some(overlap)) => scope
            .configure_decode(edge, overlap, context.geometry)
            .map_err(|error| format!("configure {LTX25_ID} decode: {error}"))?,
        (None, None) => {}
        _ => {
            return Err(format!(
                "{LTX25_ID} decode edge and overlap must be selected together"
            ))
        }
    }
    if let Some(attention) = parameters.attention_chunk_size {
        scope
            .configure_attention(attention)
            .map_err(|error| format!("configure {LTX25_ID} attention: {error}"))?;
    }
    if let Some(window) = parameters.transformer_window_size {
        scope
            .materialize_transformer_window(0, window)
            .map_err(|error| format!("configure {LTX25_ID} transformer window: {error}"))?;
    }
    let mut generation = GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        width: target.width,
        height: target.height,
        count: 1,
        seed: Some(target.seed),
        steps: Some(target.transformer_variant.steps()),
        // The dev provider owns its fixed multimodal guider parameters; neither packed variant
        // advertises the generic request-level guidance axis.
        guidance: None,
        frames: Some(target.frames),
        fps: Some(target.fps),
        // The production default A/V T2V path leaves this unset. `Some("no_audio")` is the only
        // LTX T2V override; the evidence mode belongs in `MemoryRunContext`, not this provider knob.
        video_mode: None,
        ..Default::default()
    };
    scope
        .configure_request(&mut generation)
        .map_err(|error| format!("apply {LTX25_ID} capture strategy: {error}"))?;
    scope
        .enter_phase(MemoryPhase::Conditioning)
        .map_err(|error| format!("enter {LTX25_ID} conditioning: {error}"))?;
    let generation_sample = vram.phase();
    let mut phase_sample = Some(vram.phase());
    let mut phase = MemoryPhase::Conditioning;
    let mut peaks = [None, None, None];
    let mut phase_error = None;
    let result = generator.generate(&generation, &mut |progress| {
        if phase_error.is_some() {
            return;
        }
        let boundary = match progress {
            Progress::Loading(runtime_cuda::gen_core::LoadPhase::Renderer) => {
                protocol::ReferenceBoundary::RendererLoad
            }
            Progress::Step { current: 1, .. } => protocol::ReferenceBoundary::FirstDenoiseStep,
            Progress::Decoding => protocol::ReferenceBoundary::Decoding,
            _ => return,
        };
        let Some(next) = protocol::next_reference_phase(reference_phase(phase), boundary) else {
            return;
        };
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = phase_sample.take().map(|sample| vram.end_observed(sample));
        if let Err(error) = scope.leave_phase(phase) {
            phase_error = Some(format!("leave {LTX25_ID} {phase:?}: {error}"));
            return;
        }
        let next = memory_phase(next);
        if let Err(error) = scope.enter_phase(next) {
            phase_error = Some(format!("enter {LTX25_ID} {next:?}: {error}"));
            return;
        }
        phase = next;
        phase_sample = Some(vram.phase());
    });
    if let Some(sample) = phase_sample.take() {
        let index = match phase {
            MemoryPhase::Conditioning => 0,
            MemoryPhase::Denoise => 1,
            MemoryPhase::Decode => 2,
        };
        peaks[index] = Some(vram.end_observed(sample));
    }
    vram.end_gen(generation_sample);
    let cumulative_run_peak_bytes = decimal_gb_to_bytes(vram.report().peak_gb);
    if let Some(message) = phase_error {
        let _ = scope.finish(MemoryRunOutcome::Error {
            message: message.clone(),
        });
        return Err(message);
    }
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            let message = error.to_string();
            let _ = scope.finish(MemoryRunOutcome::Error {
                message: message.clone(),
            });
            return Err(format!("{LTX25_ID} generation failed: {message}"));
        }
    };
    scope
        .leave_phase(phase)
        .map_err(|error| format!("leave {LTX25_ID} terminal phase: {error}"))?;
    scope
        .finish(MemoryRunOutcome::Complete)
        .map_err(|error| format!("finish {LTX25_ID} capture scope: {error}"))?;
    let (frames, fps, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        GenerationOutput::Images(_) | GenerationOutput::ImagesRgba(_) => {
            return Err(format!("{LTX25_ID} returned images, not a video clip"))
        }
        GenerationOutput::Audio(_) => {
            return Err(format!("{LTX25_ID} returned audio without video frames"))
        }
    };
    if fps != target.fps || audio.is_none() {
        return Err(format!(
            "{LTX25_ID} returned {fps} fps with audio={}, expected {} fps with audio",
            audio.is_some(),
            target.fps
        ));
    }
    let frame_shapes = frames
        .iter()
        .map(|frame| (frame.width, frame.height, frame.pixels.len()))
        .collect::<Vec<_>>();
    protocol::validate_ltx25_rgb_frames(
        usize::try_from(target.frames)
            .map_err(|_| "LTX-2.5 frame count does not fit usize".to_owned())?,
        target.width,
        target.height,
        &frame_shapes,
    )?;
    let conditioning_bytes = decimal_gb_to_bytes(
        peaks[0].ok_or_else(|| format!("{LTX25_ID} did not expose the conditioning boundary"))?,
    );
    let denoise_bytes = decimal_gb_to_bytes(
        peaks[1].ok_or_else(|| format!("{LTX25_ID} did not expose the denoise boundary"))?,
    );
    let decode_bytes = decimal_gb_to_bytes(
        peaks[2].ok_or_else(|| format!("{LTX25_ID} did not complete decode sampling"))?,
    );
    let overall_bytes = protocol::validated_cumulative_peak(
        cumulative_run_peak_bytes,
        [conditioning_bytes, denoise_bytes, decode_bytes],
    )?;
    let blocker = concat!(
        "SC-18783 LTX-2.5 Candle capture measured the selected real provider path; promotion remains ",
        "gated on terminal CUDA repetition/quality evidence; advanced INT8-ConvRot/NVFP4 receipt production remains inference-owned"
    );
    let mut fragment = protocol::plain_gated_fragment(
        request,
        LTX25_EXECUTION_PATH,
        protocol::PlainGatedFragment {
            // Artifact variant retains the manifest/download identity. Transformer and decoder are
            // independent target axes and are stamped into the final record from `planned.target`.
            artifact: {
                let mut artifact = artifact(&repository, &revision, &target.tier);
                artifact["inventorySha256"] = json!(inventory_sha256);
                artifact
            },
            sweep: protocol::reference_sweep(request, "passed")?,
            blocker,
            quality: json!({ "result": "not_run" }),
            negative_mutation: Value::Null,
            loadability: json!({
                "result": "passed",
                "resolvedPathFingerprint": format!(
                    "{}:transformer={}:decoder={}:f{}:{}x{}:fps{}:seed{}",
                    loadability_fingerprint(&repository, &revision, &target.tier),
                    target.transformer_variant.as_str(),
                    target.decoder.as_str(),
                    target.frames,
                    target.width,
                    target.height,
                    target.fps,
                    target.seed,
                ),
            }),
            diagnostics: protocol::diagnostics(
                "memory-candle-adapter:ltx-2.5",
                "executed",
                [blocker.to_owned()],
                [
                    ("conditioningDevicePeakDelta", "bytes", conditioning_bytes),
                    ("denoiseDevicePeakDelta", "bytes", denoise_bytes),
                    ("decodeDevicePeakDelta", "bytes", decode_bytes),
                    ("overallDevicePeakDelta", "bytes", overall_bytes),
                    ("renderedFrames", "count", u64::from(target.frames)),
                    ("renderedFps", "fps", u64::from(target.fps)),
                ],
            ),
        },
    )?;
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

/// LTX-2.3's own geometry envelope, the same three constraints the MLX arm applies: the declared
/// `limits.resolutions`, `limits.requiresDimensionsMultipleOf`, and the `1 + 8k` temporal lattice
/// the engine's `validate_request` hard-rejects outside of. A still geometry is on the lattice but
/// below the floor, so it is refused here too — this arm may not capture a single-frame record for
/// a video model.
pub(super) fn validate_ltx23_candle_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), String> {
    if !LTX23_CANDLE_RESOLUTIONS.contains(&(width, height)) {
        return Err(format!(
            "Candle LTX-2.3 calibration requires one of the declared limits.resolutions \
             {LTX23_CANDLE_RESOLUTIONS:?}, got {width}x{height}"
        ));
    }
    if !width.is_multiple_of(LTX23_CANDLE_DIMENSION_MULTIPLE)
        || !height.is_multiple_of(LTX23_CANDLE_DIMENSION_MULTIPLE)
    {
        return Err(format!(
            "Candle LTX-2.3 calibration requires geometry divisible by \
             {LTX23_CANDLE_DIMENSION_MULTIPLE}, got {width}x{height}"
        ));
    }
    if frames % LTX23_CANDLE_TEMPORAL_SCALE != 1 || frames < 9 {
        return Err(format!(
            "Candle LTX-2.3 calibration requires geometry.frames == 1 + \
             {LTX23_CANDLE_TEMPORAL_SCALE}k (>= 9; the LTX video VAE is \
             {LTX23_CANDLE_TEMPORAL_SCALE}x causal in time), got {frames}"
        ));
    }
    Ok(())
}

/// LTX-2.3's Candle staging: the packed tier sub-directory plus the separate Gemma-3-12B encoder.
///
/// TWO roots under ONE repository, exactly as the MLX arm resolves them. The encoder is a hard
/// load-time requirement — `video_jobs/candle.rs` resolves `ltx_gemma_dir` for every non-2.5 LTX
/// load and the provider surfaces a required-`text_encoder` error without it — so it is threaded
/// through `LoadSpec::text_encoder` and snapshot-validated with the same identity check as the tier
/// root: a mismatched TE would silently change the measured conditioning peak.
///
/// `quantize` stays unset because the worker's own resolver leaves it unset:
/// `candle_ltx_tier_subdir` returns `(dir, None)` for LTX-2.3 ("LTX-2.3 keeps its legacy `None`
/// marker"), and the tier is carried by the directory the load opens.
pub(super) fn ltx23_candle_load_plan(
    _request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let repository = protocol::required_env("SCENEWORKS_LTX_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_LTX_REVISION")?;
    protocol::validate_artifact_identity(&repository, &revision, protocol::LTX_REPOSITORY)?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        &target.tier,
        protocol::LTX_REPOSITORY,
    )?;
    let text_encoder = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_LTX_TEXT_ENCODER_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_LTX_TEXT_ENCODER_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &text_encoder,
        &repository,
        &revision,
        "gemma",
        protocol::LTX_REPOSITORY,
    )?;
    // sc-22738: EAGER, not deferred. `candle-gen-ltx` refuses any other LTX-2.3 load shape on its
    // production contract path, and its registry fixture pins the same shape — so the `Deferred`
    // this arm shipped with in sc-22737 could never produce a contract, and the plan rows that
    // declared `deferred_materialization` alongside it were uncapturable. The plan rows now say
    // eager too; `run_sc22737_video_capture` still re-asserts the plan's shape against the LOADED
    // contract.
    let spec = ltx23_candle_load_spec(root, text_encoder, &target.tier)?;
    Ok(Sc22737LoadPlan {
        artifact: artifact(&repository, &revision, &target.tier),
        resolved_path_fingerprint: format!(
            "{}:gemma",
            loadability_fingerprint(&repository, &revision, &target.tier)
        ),
        spec,
    })
}

pub(super) fn ltx23_candle_load_spec(
    root: PathBuf,
    text_encoder: PathBuf,
    tier: &str,
) -> Result<LoadSpec, String> {
    let quant = match tier {
        "q4" => Quant::Q4,
        "q8" => Quant::Q8,
        other => return Err(format!("Candle LTX-2.3 requires q4 or q8, got {other:?}")),
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_quant(quant)
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LTX23_CANDLE_LOAD_SHAPE);
    spec.text_encoder = Some(WeightsSource::Dir(text_encoder));
    Ok(spec)
}
