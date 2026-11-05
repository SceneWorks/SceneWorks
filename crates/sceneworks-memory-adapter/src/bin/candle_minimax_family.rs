//! CANDLE minimax capture owner.
use super::*;

/// MiniMax-H3's own geometry envelope, read off the pinned Candle crate: the `17n + 5` temporal
/// lattice, the spatial stride, and the canvas budget as a PRODUCT rather than per edge — the
/// published resolution list contains 1536x672 and 1344x768, whose long edges differ by 192 px and
/// whose areas are identical, so a per-edge cap would refuse the first and admit the second while
/// both sit exactly at the budget.
pub(super) fn validate_minimax_candle_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), String> {
    let lattice = runtime_cuda::providers::minimax_h3::denoise::geometry::LEGAL_FRAME_COUNTS;
    let frames_usize =
        usize::try_from(frames).map_err(|_| "MiniMax-H3 frame count must fit usize".to_owned())?;
    if !lattice.contains(&frames_usize) {
        return Err(format!(
            "Candle MiniMax-H3 requires geometry.frames on the 17n+5 lattice {lattice:?}, got \
             {frames}"
        ));
    }
    let stride = runtime_cuda::providers::minimax_h3::pipeline::SPATIAL_STRIDE;
    if !width.is_multiple_of(stride) || !height.is_multiple_of(stride) {
        return Err(format!(
            "Candle MiniMax-H3 requires geometry divisible by the {stride}px stride, got \
             {width}x{height}"
        ));
    }
    let budget = runtime_cuda::providers::minimax_h3::pipeline::CANVAS_MAX_PIXELS;
    let pixels = width.saturating_mul(height);
    if pixels > budget {
        return Err(format!(
            "Candle MiniMax-H3 requires width*height within the {budget}px canvas budget, got \
             {width}x{height} ({pixels}px)"
        ));
    }
    Ok(())
}

/// MiniMax-H3's Candle staging: the UPSTREAM snapshot root, with the packed components redirected
/// onto the tiered rehost.
///
/// The direction is the opposite of what the repository names suggest, and it is the worker's:
/// `resolve_candle_minimax_h3_load` makes `MiniMaxAI/MiniMax-H3` the load ROOT on every tier — it is
/// the only tree carrying `vae/`, `audio_vae/`, `tokenizer/` and the `FL2VA/` documents — and stages
/// `transformer/` and `text_encoder/` out of `SceneWorks/minimax-h3-mlx/<tier>` when the tier is
/// packed. `bf16` on the BASE entry stages neither, because the dense partitions live in the
/// upstream root already; `minimax_h3_ref` stages the tier tree at every tier, because the reference
/// partition is only published in the rehost.
///
/// The `transformer` component is ALWAYS the base partition, even for the reference entry: the
/// provider derives `transformer_ref/` as its sibling, so the worker validates both halves and
/// stages only the base. This arm does the same, and its `resolved_path_fingerprint` names the
/// partition the CONDITIONING will select so two records over one triple stay distinguishable.
pub(super) fn minimax_candle_load_plan(
    request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let arm = sc22737_video_arm(request)?
        .ok_or_else(|| "the MiniMax-H3 Candle plan resolved no member".to_owned())?;
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

    // sc-22738: EAGER — the loader pins it, so the loaded contract's calibration identity is
    // eager and the `Deferred` this arm shipped with could only ever fail the plan/provider
    // load-shape re-assert below; the six plan rows now say eager too.
    let mut spec = LoadSpec::new(WeightsSource::Dir(upstream_root))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(MINIMAX_CANDLE_LOAD_SHAPE);
    let quant = numeric_tier(&target.tier)?.quant;
    let is_reference = arm.model_id == MINIMAX_CANDLE_REFERENCE_MODEL_ID;
    let staged = if quant.is_some() || is_reference {
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
            &target.tier,
            protocol::MINIMAX_REPOSITORY,
        )?;
        spec = spec.with_component(
            "transformer",
            WeightsSource::Dir(tier_root.join("transformer")),
        );
        // The packed text encoder ships beside the packed DiT; the dense one lives upstream and the
        // loader resolves it from the spec's own weights root, so it is not redirected.
        if quant.is_some() {
            spec = spec.with_component(
                "text_encoder",
                WeightsSource::Dir(tier_root.join("text_encoder")),
            );
        }
        Some((repository, revision))
    } else {
        None
    };
    if let Some(quant) = quant {
        // Never a load-time pack: the loader RECONCILES `spec.quantize` against the staged tier's
        // own marker and refuses a disagreement, so passing it is an assertion about the directory
        // on disk rather than an instruction.
        spec = spec.with_quant(quant);
    }
    let partition = if is_reference {
        "transformer_ref"
    } else {
        "transformer"
    };
    Ok(Sc22737LoadPlan {
        // The schema's artifact is the load root. The staged repository, revision,
        // tier and selected partition remain bound by resolved_path_fingerprint below.
        artifact: artifact(&upstream_repository, &upstream_revision, &target.tier),
        resolved_path_fingerprint: format!(
            "{upstream_repository}@{upstream_revision}:{}+partition:{partition}+staged:{}",
            target.tier,
            staged
                .as_ref()
                .map(|(repository, revision)| format!("{repository}@{revision}/{}", target.tier))
                .unwrap_or_else(|| "upstream-dense".to_owned()),
        ),
        spec,
    })
}
