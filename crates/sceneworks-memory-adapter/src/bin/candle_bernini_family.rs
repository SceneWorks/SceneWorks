//! CANDLE bernini capture owner.
use super::*;

/// Refuse any Bernini catalog entry but the STILL one at the still gate (sc-22737).
///
/// `bernini` (video) and `bernini_image` (still) share the engine provider id `bernini`, so the
/// provider alone cannot say which member a plan row means. The video member is dispatched above
/// the still gate and never arrives here; if it does, the plan is asking for a video record on the
/// image path, and the two are not interchangeable — a still record carries `frames: 1` and the
/// `text_to_image` mode key, which is a different admission key from the one the video route asks
/// under. Refused BY NAME rather than defaulted.
pub(super) fn bernini_candle_still_member(request: &Value) -> Result<&'static str, String> {
    let model_id = protocol::planned_model_id(request)?;
    if model_id != BERNINI_CANDLE_IMAGE_MODEL_ID {
        return Err(format!(
            "the Candle Bernini still path serves catalog entry \
             {BERNINI_CANDLE_IMAGE_MODEL_ID:?}, got modelId {model_id:?}; \
             {BERNINI_CANDLE_VIDEO_MODEL_ID:?} is the video entry and is measured by the sc-22737 \
             video arm instead"
        ));
    }
    Ok(BERNINI_CANDLE_IMAGE_MODEL_ID)
}

/// Bernini's own geometry envelope: the engine's `validate_geometry`, which is an exact membership
/// test against `ADVERTISED_GEOMETRIES` and is PROVIDER-WIDE — `candle-gen-bernini` applies it to
/// the still route and the video route alike. Read off the crate rather than transcribed.
pub(super) fn validate_bernini_candle_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<(), String> {
    let advertised = runtime_cuda::providers::bernini::memory_strategy::ADVERTISED_GEOMETRIES;
    if !advertised.contains(&(width, height)) {
        return Err(format!(
            "Candle Bernini memory evidence requires one of the advertised geometries \
             {advertised:?}, got {width}x{height}"
        ));
    }
    // The renderer is Wan2.2-A14B, whose frame count is `1 mod 4` (`video_jobs/wan.rs`'s
    // `wan_frame_count`, which the Bernini video path calls for exactly that reason).
    if !matches!(frames, 45 | 61 | 77) {
        return Err(format!(
            "Candle Bernini reference memory requires 45, 61, or 77 frames, got {frames}"
        ));
    }
    Ok(())
}

/// Bernini's Candle staging: ONE tier directory inside the off-Mac bundle.
///
/// `SceneWorks/bernini` is the only download either Bernini entry ships for
/// `platforms: ["windows", "linux"]`, and it carries the three tier sub-directories INSIDE it —
/// which is why `memory_route_registry.rs` declares `BF16_Q4_Q8` for `candle:bernini` while the
/// download row names no `variant` at all. `quantize` stays unset: the tiers ship pre-packed, so the
/// worker's own resolver returns no load-time quant for a resolved tier, and `production_assets`
/// reconciles the packing against the transformer's own `quantize_config.json` before the identity
/// is minted.
pub(super) fn bernini_candle_load_plan(
    _request: &Value,
    target: &Sc22737VideoTarget,
) -> Result<Sc22737LoadPlan, String> {
    let repository = protocol::required_env("SCENEWORKS_BERNINI_CANDLE_REPOSITORY")?;
    let revision = protocol::required_env("SCENEWORKS_BERNINI_CANDLE_REVISION")?;
    protocol::validate_artifact_identity(
        &repository,
        &revision,
        protocol::BERNINI_CANDLE_REPOSITORY,
    )?;
    let root = std::fs::canonicalize(PathBuf::from(protocol::required_env(
        "SCENEWORKS_BERNINI_CANDLE_ROOT",
    )?))
    .map_err(|error| format!("canonicalize SCENEWORKS_BERNINI_CANDLE_ROOT: {error}"))?;
    protocol::validate_huggingface_snapshot_root(
        &root,
        &repository,
        &revision,
        &target.tier,
        protocol::BERNINI_CANDLE_REPOSITORY,
    )?;
    Ok(Sc22737LoadPlan {
        artifact: artifact(&repository, &revision, &target.tier),
        resolved_path_fingerprint: loadability_fingerprint(&repository, &revision, &target.tier),
        spec: bernini_candle_load_spec(
            LoadSpec::new(WeightsSource::Dir(root)).with_offload_policy(OffloadPolicy::Sequential),
            &target.tier,
            BERNINI_CANDLE_VIDEO_MODEL_ID,
        )?,
    })
}

/// The `LoadSpec` the worker hands `candle-gen-bernini` for one published tier directory, on
/// either lane.
///
/// * EAGER — `candle-gen-bernini`'s production contract refuses anything else (sc-22738). Its
///   registry fixture merely mirrors the spec's shape, which is why the weights-free walk in
///   `inference_runtime.rs` could not see that the `Deferred` the video arm shipped with (and the
///   six plan rows beside it) was uncapturable; the plan rows say eager too. The still lane's
///   `LoadSpec::new` default is eager and its registry rule shapes nothing
///   (`image_jobs/bernini.rs`), so eager is the ONLY shape either lane loads.
/// * The packed tier's quant, EXPLICITLY. Both worker lanes resolve
///   `candle_bernini_tier_quant(tier)` — `Some(Q4)`/`Some(Q8)` for the packed subtrees, `None`
///   for `bf16/` — and bind it (`image_jobs/bernini.rs` `load_spec(weights_dir, quant, …)`,
///   `video_jobs/wan.rs` `spec.quantize = input.quant`). It is an assertion, not a load-time
///   pack: `candle-gen-bernini`'s `production_assets` reads the EXPECTED packing off
///   `spec.quantize` (`expected_packing`) and then proves every component against it
///   (`component_receipt`), so a packed root loaded with `quantize: None` is checked as DENSE,
///   the receipt refuses, and the load carries no contract at all. sc-22738 (CUDA run
///   34356681566): `bernini q4/q8` and `bernini_image q4/q8` were exactly that — "loaded bernini
///   has no memory-strategy contract" — because neither arm bound the quant.
/// * The catalog entry as `resolved_route`, the worker's `.with_resolved_route(request.model)`:
///   `bernini` on the video lane, `bernini_image` on the still lane.
pub(super) fn bernini_candle_load_spec(
    spec: LoadSpec,
    tier: &str,
    resolved_route: &str,
) -> Result<LoadSpec, String> {
    let spec = spec
        .with_load_shape(BERNINI_CANDLE_LOAD_SHAPE)
        .with_resolved_route(resolved_route.to_owned());
    Ok(match numeric_tier(tier)?.quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    })
}
