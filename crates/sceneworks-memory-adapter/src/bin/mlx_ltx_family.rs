//! MLX ltx capture owner.
use super::*;

/// The production calibration identity `mlx-gen-ltx` mints for one `mlx:ltx_2_3` numeric tier.
///
/// The engine keys on `(provider, ARTIFACT-proven tier)`: the calibrated tier retains the bare
/// `CALIBRATION_FINGERPRINT` and every other shipped tier receives that same key with its tier
/// token inserted after the family stem. The calibrated cell reproduces
/// [`mlx_gen_ltx::memory_strategy::CALIBRATION_FINGERPRINT`] byte-for-byte — read off the engine
/// rather than spelled, so the two copies cannot drift — which is what keeps the retained key of
/// the family's existing record valid.
///
/// Written here as well as in the engine so the plan/arm binding is weights-free and provable on a
/// CPU-only host. The capture then refuses a loaded contract whose identity differs from this
/// table, so a drift is caught at the pin bump rather than in a record.
///
/// A tier outside [`LTX_TIERS`] is refused BY NAME rather than interpolated into a plausible
/// identity no production load can return.
pub(super) fn ltx_calibration_fingerprint(tier: &str) -> Result<String, String> {
    if !LTX_TIERS.contains(&tier) {
        return Err(format!(
            "the MLX LTX-2.3 calibration family ships {LTX_TIERS:?}; tier {tier:?} has no \
             production identity"
        ));
    }
    if tier == LTX_CALIBRATED_TIER {
        return Ok(mlx_gen_ltx::memory_strategy::CALIBRATION_FINGERPRINT.to_owned());
    }
    Ok(format!("sc-20772-ltx-2-3-{tier}-mlx-memory-ladder-v2"))
}

/// Port of `sceneworks_core::video_request::ltx_frame_count` — frames snap to the NEAREST `8k + 1`,
/// minimum 9, ties to the lower. Ported before `sceneworks-core` became a dependency of this binary
/// (sc-22738 made it one, for the worker's shared admission policy in `memory_anchor`); the port
/// stays because its two-sided pin below is what binds this arm to the shipped ladder, and it is
/// `const`.
///
/// Because it is a port and NOT a call, the binding to the shipped ladder is by transcription, and
/// it takes TWO tests to close: `ltx_frame_ladder_port_matches_the_transcribed_shipped_ladder` here
/// pins the 18 shipped (duration, fps) pairs against this port, and
/// `ltx_frame_count_matches_the_sc_18808_calibration_ladder` in
/// `crates/sceneworks-core/src/video_request.rs` pins the SAME 18 pairs against
/// `ltx_frame_count` itself. `sceneworks-core` is a workspace default member, so a change to the
/// shipped ladder reds under a plain `cargo test`; without that half, a drift would silently move
/// what this arm accepts and neither crate would notice (sc-18808 review).
pub(super) const fn ltx_snapped_frame_count(raw_frames: u32) -> u32 {
    let frame_count = if raw_frames < 9 { 9 } else { raw_frames };
    let lower = frame_count - ((frame_count - 1) % 8);
    let upper = lower + 8;
    if lower < 9 {
        return upper;
    }
    if frame_count - lower <= upper - frame_count {
        lower
    } else {
        upper
    }
}

/// The closed frame envelope the declared `limits` can actually produce, derived over the FULL
/// 18-cell `durations x fps` cross product through the ladder the product itself uses. Derived
/// rather than written down so the bounds cannot drift away from the arrays above.
pub(super) const fn ltx_frame_envelope() -> (u32, u32) {
    let (mut minimum, mut maximum) = (u32::MAX, 0);
    let mut duration = 0;
    while duration < LTX_DURATIONS_SECONDS.len() {
        let mut fps = 0;
        while fps < LTX_FPS.len() {
            let frames = ltx_snapped_frame_count(LTX_DURATIONS_SECONDS[duration] * LTX_FPS[fps]);
            if frames < minimum {
                minimum = frames;
            }
            if frames > maximum {
                maximum = frames;
            }
            fps += 1;
        }
        duration += 1;
    }
    (minimum, maximum)
}

/// The LTX-2.3 twin of [`flux2_quality_passes`], over the LTX-named thresholds so an `mlx:ltx_2_3`
/// receipt's numbers trace to an LTX constant.
pub(super) fn ltx_quality_passes(maximum: f64, mean: f64, rms: f64) -> bool {
    maximum <= LTX_MAX_THRESHOLD && mean <= LTX_MEAN_THRESHOLD && rms <= LTX_RMS_THRESHOLD
}

/// The LTX-2.3 arm's lifecycle blockers: a recovery-clip FINDING when a post-fault recovery render
/// leaves the determinism envelope, recorded instead of discarding the capture.
///
/// sc-22738 (measured 2026-09-07): `ltx_2_3:q4:mlx` and `ltx_2_3:q8:mlx` each rendered the full
/// ladder — roughly six minutes of real MLX work apiece — and were then thrown away, because the
/// post-cancel recovery clip left the envelope and the arm `return Err`ed with a bare sentence that
/// carried no numbers at all. `ltx_2_3:bf16:mlx` passes the identical sequence bit-identically, so
/// the difference is the quantized tier, and the same run's CLEAN warm control (render two, taken
/// before any fault, compared against the measured clip with this very helper) had already passed.
///
/// That makes the reading evidence about the ENGINE, not a reason to refuse:
///
/// * The peaks, phase boundaries and staging proofs this record ships come from the FIRST render on
///   a freshly loaded generator. Renders three onward only feed the recovery comparison, so what
///   they produce cannot corrupt the numbers the record actually claims.
/// * The refusal destroyed its own evidence. The one observation of this event survived as a single
///   truncated line in a campaign log with no max/mean/rms, no phase and no clean-warm reference —
///   the exact diagnostics that were supposed to characterize it.
/// * Production would see it too. If a cancelled LTX-2.3 render changes the NEXT render on a
///   quantized tier, that is a user-visible hazard admission and the engine should be told about,
///   not a capture to re-roll until it comes out clean.
///
/// The envelope is NOT weakened: the comparison still runs, still uses [`ltx_quality_passes`] — the
/// same helper and the same three thresholds the warm repeat is judged by — and an excursion now
/// lands in `blockers` under the cell's own name with both references and all three metrics, so it
/// is louder in the store than it was as a dropped capture.
///
/// A genuinely invalid capture still refuses: [`video_max_mean_rms_abs`] rejects a frame-count,
/// dimension or pixel-length change before any of this is reached, and every allocator bound, typed
/// cancellation outcome and injected-error outcome above remains a hard `Err`.
/// The recovery-clip diagnostics the record carries beside any blocker: both faults, both
/// references, all three metrics, plus what each recovery clip IS against the measured clip's own
/// level. They are emitted whether or not the envelope was left, so a clean capture states the
/// deltas were zero rather than leaving a reader to infer it from silence.
pub(super) fn ltx_recovery_measurements(
    cancel: LtxRecoveryEvidence,
    error: LtxRecoveryEvidence,
    measured: LtxClipSummary,
) -> Vec<(&'static str, &'static str, u64)> {
    let mut measurements = Vec::with_capacity(17);
    for (names, evidence) in [
        (
            [
                [
                    "cancelRecoveryVersusMeasuredMaximumErrorMilliPer255",
                    "cancelRecoveryVersusMeasuredMeanErrorMilliPer255",
                    "cancelRecoveryVersusMeasuredRootMeanSquareErrorMilliPer255",
                ],
                [
                    "cancelRecoveryVersusCleanWarmMaximumErrorMilliPer255",
                    "cancelRecoveryVersusCleanWarmMeanErrorMilliPer255",
                    "cancelRecoveryVersusCleanWarmRootMeanSquareErrorMilliPer255",
                ],
            ],
            cancel,
        ),
        (
            [
                [
                    "errorRecoveryVersusMeasuredMaximumErrorMilliPer255",
                    "errorRecoveryVersusMeasuredMeanErrorMilliPer255",
                    "errorRecoveryVersusMeasuredRootMeanSquareErrorMilliPer255",
                ],
                [
                    "errorRecoveryVersusCleanWarmMaximumErrorMilliPer255",
                    "errorRecoveryVersusCleanWarmMeanErrorMilliPer255",
                    "errorRecoveryVersusCleanWarmRootMeanSquareErrorMilliPer255",
                ],
            ],
            error,
        ),
    ] {
        for (names, delta) in names
            .into_iter()
            .zip([evidence.vs_measured, evidence.vs_clean_warm])
        {
            for (name, value) in names
                .into_iter()
                .zip([delta.maximum, delta.mean, delta.rms])
            {
                measurements.push((name, "count", LtxClipDelta::milli_per_255(value)));
            }
        }
    }
    measurements.push((
        "measuredClipMeanLevelMilliPer255",
        "count",
        LtxClipDelta::milli_per_255(measured.mean_level),
    ));
    for (degenerate, level, evidence) in [
        (
            "cancelRecoveryFirstFrameDegenerate",
            "cancelRecoveryMeanLevelMilliPer255",
            cancel,
        ),
        (
            "errorRecoveryFirstFrameDegenerate",
            "errorRecoveryMeanLevelMilliPer255",
            error,
        ),
    ] {
        measurements.push((
            degenerate,
            "count",
            u64::from(evidence.clip.degenerate_first_frame),
        ));
        measurements.push((
            level,
            "count",
            LtxClipDelta::milli_per_255(evidence.clip.mean_level),
        ));
    }
    measurements
}

pub(super) fn ltx_lifecycle_blockers(
    tier: &str,
    fault_phase: MemoryPhase,
    cancel: LtxRecoveryEvidence,
    error: LtxRecoveryEvidence,
) -> Vec<String> {
    let mut blockers = Vec::new();
    for (fault, evidence) in [("cancellation", cancel), ("injected error", error)] {
        if evidence.vs_measured.passes() {
            continue;
        }
        blockers.push(format!(
            "LTX-2.3 {tier} {fault} cleanup at {fault_phase:?} changed the warm recovery clip: \
             against the measured clip max={:.6}, mean={:.6}, rms={:.6} over thresholds \
             max={LTX_MAX_THRESHOLD:.6}, mean={LTX_MEAN_THRESHOLD:.6}, rms={LTX_RMS_THRESHOLD:.6}; \
             against the clean warm control max={:.6}, mean={:.6}, rms={:.6}; the recovery clip's \
             own mean level is {:.6} with a {}degenerate first frame. Recorded as measured \
             evidence rather than discarding the capture, whose peaks come from the cold first \
             render",
            evidence.vs_measured.maximum,
            evidence.vs_measured.mean,
            evidence.vs_measured.rms,
            evidence.vs_clean_warm.maximum,
            evidence.vs_clean_warm.mean,
            evidence.vs_clean_warm.rms,
            evidence.clip.mean_level,
            if evidence.clip.degenerate_first_frame {
                ""
            } else {
                "non-"
            },
        ));
    }
    blockers
}

pub(super) fn ltx_declared_resolutions() -> String {
    LTX_RESOLUTIONS
        .iter()
        .map(|(width, height)| format!("{width}x{height}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// LTX's own geometry envelope, which REPLACES the image arms' `frames == 1` refusal for this arm
/// alone. Three independent constraints, each from a stated source:
///
/// * `limits.resolutions` — the five declared pairs (the catalog contract a real request can name).
/// * `limits.requiresDimensionsMultipleOf` — 64, mirroring the engine's `SIZE_MULTIPLE`. Redundant
///   with the pair list today; kept because it is the engine's hard rule and the pair list is not.
/// * the temporal lattice and envelope — `frames = 1 + 8k` (the engine's `validate_request` hard
///   reject) inside `[97, 449]`, the closed span the declared `durations x fps` cross product
///   produces through the shipped LTX frame ladder.
///
/// A still geometry (`frames == 1`) is on the lattice but below the envelope floor, so it is refused
/// here too: this arm may not silently capture a single-frame record for a video model.
pub(super) fn validate_ltx_geometry(
    width: u32,
    height: u32,
    frames: u32,
) -> Result<LtxGeometry, String> {
    if !LTX_RESOLUTIONS.contains(&(width, height)) {
        return Err(format!(
            "MLX LTX-2.3 calibration requires one of the declared limits.resolutions ({}), got {width}x{height}",
            ltx_declared_resolutions()
        ));
    }
    if !width.is_multiple_of(LTX_DIMENSION_MULTIPLE)
        || !height.is_multiple_of(LTX_DIMENSION_MULTIPLE)
    {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry divisible by {LTX_DIMENSION_MULTIPLE}, got {width}x{height}"
        ));
    }
    if frames % LTX_TEMPORAL_SCALE != 1 {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry.frames == 1 + {LTX_TEMPORAL_SCALE}k (the LTX \
             video VAE is {LTX_TEMPORAL_SCALE}x causal in time), got {frames}"
        ));
    }
    let (minimum, maximum) = LTX_FRAME_ENVELOPE;
    if frames < minimum || frames > maximum {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry.frames within the declared duration/fps \
             envelope [{minimum}, {maximum}], got {frames}"
        ));
    }
    Ok(LtxGeometry {
        width,
        height,
        frames,
        latent_frames: 1 + (frames - 1) / LTX_TEMPORAL_SCALE,
    })
}

/// Read the four declared geometry axes. Unlike the image arms this reads `frames` as a real value
/// rather than asserting it away; `batch` is still pinned to 1 because LTX's descriptor advertises
/// `max_count: 1` and the arm renders exactly one clip.
pub(super) fn ltx_target_geometry(request: &Value) -> Result<LtxGeometry, String> {
    let geometry = protocol::planned_object(request, "/target/geometry")?;
    let axis = |name: &str| protocol::geometry_axis_u32(geometry, name);
    let batch = axis("batch")?;
    if batch != 1 {
        return Err(format!(
            "MLX LTX-2.3 calibration requires geometry.batch == 1 (the provider advertises \
             max_count 1), got {batch}"
        ));
    }
    validate_ltx_geometry(axis("width")?, axis("height")?, axis("frames")?)
}

/// Defense-in-depth mirror of `validate_flux2_target`, plus the T2V-specific target shape. The
/// `run` dispatcher routes by provider id today, but this arm hardcodes the LTX contract, so a
/// foreign caller must be refused BY NAME here rather than misrouted into it.
pub(super) fn validate_ltx_target(request: &Value) -> Result<LtxGeometry, String> {
    let target = protocol::planned_object(request, "/target")?;
    let provider = protocol::target_str(target, "provider")?;
    if provider != LTX_PROVIDER {
        return Err(format!(
            "MLX LTX-2.3 calibration does not implement provider {provider:?}"
        ));
    }
    let model_id = protocol::target_str(target, "modelId")?;
    if model_id != LTX_PROVIDER {
        return Err(format!(
            "MLX LTX-2.3 calibration requires modelId {LTX_PROVIDER:?}, got {model_id:?}"
        ));
    }
    let mode = protocol::target_str(target, "mode")?;
    if mode != "text_to_video" {
        return Err(format!(
            "MLX LTX-2.3 calibration requires reference-free text_to_video mode, got {mode:?}"
        ));
    }
    for field in ["referenceCount", "reference_count"] {
        if let Some(value) = target.get(field) {
            if value.as_u64() != Some(0) {
                return Err(format!(
                    "MLX LTX-2.3 calibration requires {field} == 0 when declared"
                ));
            }
        }
    }
    for field in ["hasReference", "has_reference"] {
        if let Some(value) = target.get(field) {
            if value.as_bool() != Some(false) {
                return Err(format!(
                    "MLX LTX-2.3 calibration requires {field} == false when declared"
                ));
            }
        }
    }
    ltx_target_geometry(request)
}

/// Bind the fixture to the planned tier AND the full rendered geometry, and recover the two request
/// parameters the geometry envelope cannot carry: the output cadence `fps` and the seed. fps rides
/// here because `GeometryEnvelope` has no temporal-cadence axis and the audio latent length — which
/// is denoised jointly with the video on every step — is `compute_audio_frames(frames, fps)`.
pub(super) fn planned_ltx_capture(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
) -> Result<(u32, u64), String> {
    let fixture = protocol::planned_fixture(request)?;
    let prefix = format!(
        "ltx-2-3-mlx-{tier}-{}x{}-f{}-fps",
        geometry.width, geometry.height, geometry.frames
    );
    let remainder = fixture
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("planned.fixture {fixture:?} must start with {prefix:?}"))?;
    let (fps_and_carrier, seed) = remainder
        .split_once("-seed")
        .ok_or_else(|| format!("planned.fixture {fixture:?} must end with -seed<seed>"))?;
    let fps = fps_and_carrier
        .strip_suffix("-bounded-decode-192-64")
        .unwrap_or(fps_and_carrier);
    let fps = fps
        .parse::<u32>()
        .map_err(|error| format!("parse LTX fixture fps {fps:?}: {error}"))?;
    let seed = seed
        .parse::<u64>()
        .map_err(|error| format!("parse LTX fixture seed {seed:?}: {error}"))?;
    if !LTX_FPS.contains(&fps) {
        return Err(format!(
            "planned.fixture declares fps {fps}, which is not one of the declared limits.fps {LTX_FPS:?}"
        ));
    }
    if seed != LTX_SEED {
        return Err(format!(
            "planned.fixture seed {seed} does not match the LTX-2.3 calibration seed {LTX_SEED}"
        ));
    }
    Ok((fps, seed))
}

/// The SC-18946 incident EVIDENCE a frozen canary row must carry, checked before any model path,
/// provider registry, or MLX allocation is touched.
///
/// The q4 f305 incident (a 96,970,084,480-byte physical footprint that panicked the host watchdog)
/// forbids that exact case, and the frozen SC-20191/20318/20430 identities carry its arithmetic —
/// tier inventory, crash footprint, incident-calibrated projection — as the record of what was
/// known when they were minted. This validator is what keeps those private canary profiles
/// byte-bound to that evidence.
///
/// sc-22738: it is NOT an admission gate for the ordinary `run` action. SC-19642 froze every
/// ordinary LTX-2.3 MLX capture behind this block plus an unconditional refusal with no success
/// path, which is a hard-coded "never measure", not a safety check — and the epic's E4 requires
/// the record to measure what ships. The ordinary path now runs the production admission the
/// worker runs ([`ltx_ordinary_admission`]) and lets the budget accept or refuse; the SC-18946
/// records under `docs/calibration/sc-18946` keep their `safety_refused_open` dispositions as
/// history. Physical containment is the harness's footprint watchdog, which the catalog runner
/// wraps around every Darwin capture.
pub(super) fn validate_ltx_safety_evidence(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(&'static str, u64, u64), String> {
    let inventory_bytes = match tier {
        "q4" => LTX_Q4_INVENTORY_BYTES,
        "q8" => LTX_Q8_INVENTORY_BYTES,
        "bf16" => LTX_BF16_INVENTORY_BYTES,
        other => return Err(format!("unsupported SC-18946 safety tier {other:?}")),
    };
    let safety = protocol::planned(request)?
        .get("_measurementSafety")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "SC-18946 row is missing required _measurementSafety; refusing before model load"
                .to_owned()
        })?;
    let exact_u64 = |field: &str, expected: u64| -> Result<(), String> {
        let actual = safety.get(field).and_then(Value::as_u64);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-18946 safety field {field} must be {expected}, got {actual:?}; refusing before model load"
            ))
        }
    };
    let bounded = selection.strategy == MemoryStrategy::BoundedDecode;
    let incident_carrier = selection.parameters.decode_tile_edge == Some(384)
        && selection.parameters.decode_overlap == Some(64);
    let incident = tier == "q4"
        && bounded
        && incident_carrier
        && geometry.width == 1280
        && geometry.height == 704
        && geometry.frames == 305;
    let monotonic_rung2 = tier == "q4"
        && bounded
        && incident_carrier
        && geometry.width == 1280
        && geometry.height == 704
        && geometry.frames > 305;
    let expected_disposition = if incident {
        LTX_INCIDENT_FORBIDDEN
    } else if monotonic_rung2 {
        LTX_ARITHMETIC_UNMEASURABLE
    } else {
        LTX_SAFETY_REFUSED_OPEN
    };
    if safety.get("disposition").and_then(Value::as_str) != Some(expected_disposition) {
        return Err(format!(
            "SC-18946 safety disposition must be {expected_disposition:?}; refusing before model load"
        ));
    }
    exact_u64("tierInventoryBytes", inventory_bytes)?;
    exact_u64(
        "incidentCrashFootprintBytes",
        LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
    )?;
    exact_u64(
        "incidentPredictedDecodeBytes",
        LTX_INCIDENT_PREDICTED_DECODE_BYTES,
    )?;
    let voxels = u64::from(geometry.width)
        .saturating_mul(u64::from(geometry.height))
        .saturating_mul(u64::from(geometry.frames));
    let predicted_decode_bytes = if bounded {
        let tile_edge = u64::from(selection.parameters.decode_tile_edge.ok_or_else(|| {
            "SC-18946 bounded-decode safety row omitted decodeTileEdge".to_owned()
        })?);
        3_300_000_000_u64
            .saturating_add(40_u64.saturating_mul(voxels))
            .saturating_add(
                300_u64
                    .saturating_mul(tile_edge)
                    .saturating_mul(tile_edge)
                    .saturating_mul(96),
            )
    } else {
        3_300_000_000_u64.saturating_add(340_u64.saturating_mul(voxels))
    };
    exact_u64("predictedDecodeBytes", predicted_decode_bytes)?;
    let projection = i128::from(LTX_Q4_F305_CRASH_FOOTPRINT_BYTES)
        + (i128::from(inventory_bytes) - i128::from(LTX_Q4_INVENTORY_BYTES))
        + (i128::from(predicted_decode_bytes) - i128::from(LTX_INCIDENT_PREDICTED_DECODE_BYTES));
    let projection = u64::try_from(projection)
        .map_err(|_| "SC-18946 incident-calibrated projection must fit u64".to_owned())?;
    exact_u64("incidentCalibratedProjectionBytes", projection)?;
    Ok((expected_disposition, inventory_bytes, projection))
}

/// The first supervised entry into SC-18946. Unlike the ordinary `run` action — which admits
/// through the production budget ([`ltx_ordinary_admission`]) — this private action requires the
/// live watchdog channel and may reach only the first frozen q4 staged row. The runner injects the
/// two private objects after the canonical harness has constructed the row, so neither object
/// participates in the campaign logical-case identity.
pub(super) fn validate_ltx_campaign_entry(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    if protocol::action(request)? != LTX_CAMPAIGN_ENTRY_ACTION {
        return Err("SC-20191 campaign entry action changed".to_owned());
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-20191 campaign entry {pointer} must be {expected}, got {actual:?}"
            ))
        }
    };
    exact("/logicalCaseId", &json!(LTX_CAMPAIGN_ENTRY_LOGICAL_CASE_ID))?;
    exact("/evidenceScope", &json!("authoritative"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!("q4"))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("staged_residency"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency"]),
    )?;
    exact("/strategy/parameters", &json!({}))?;
    // `tier` is the plan's own `target.tier`, which the pinned check above already fixed to the
    // frozen q4 row, so the identity this row must name is derived rather than transcribed.
    exact(
        "/calibrationFingerprint",
        &json!(ltx_calibration_fingerprint(tier)?),
    )?;
    exact("/fixture", &json!(LTX_CAMPAIGN_ENTRY_FIXTURE))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_campaignEntry",
        &json!({
            "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    exact(
        "/_measurementSafety",
        &json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": LTX_Q4_INVENTORY_BYTES,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 19_476_906_240_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": 97_906_593_920_u64,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": "incident-calibrated projection is diagnostic only; no proved bound or hard containment admits this row",
        }),
    )?;
    if tier != "q4"
        || geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::StagedResidency
        || selection.parameters.decode_tile_edge.is_some()
        || selection.parameters.decode_overlap.is_some()
    {
        return Err("SC-20191 campaign entry resolved a non-canonical tuple or carrier".to_owned());
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    if fps != LTX_CAMPAIGN_ENTRY_FPS || seed != LTX_SEED {
        return Err("SC-20191 campaign entry fps or seed changed".to_owned());
    }
    let (disposition, _, _) = validate_ltx_safety_evidence(request, tier, geometry, selection)?;
    if disposition != LTX_SAFETY_REFUSED_OPEN {
        return Err("SC-20191 campaign entry safety disposition changed".to_owned());
    }
    Ok(())
}

/// A private diagnostic carrier, deliberately outside the frozen SC-18946 plan. It reuses the
/// campaign containment machinery but cannot inherit the failed staged row's action, fixture,
/// logical case, or strategy identity.
pub(super) fn validate_ltx_bounded_carrier_proof(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    if protocol::action(request)? != LTX_BOUNDED_CARRIER_ACTION {
        return Err("SC-20254 bounded-carrier action changed".to_owned());
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "SC-20254 bounded carrier {pointer} must be {expected}, got {actual:?}"
            ))
        }
    };
    exact("/_diagnosticOnly", &json!(true))?;
    exact(
        "/logicalCaseId",
        &json!(LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID),
    )?;
    exact("/evidenceScope", &json!("fixture"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!("q4"))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("bounded_decode"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency", "bounded_decode"]),
    )?;
    exact(
        "/strategy/parameters",
        &json!({
            "decodeTileEdge": LTX_CANARY_TILE_EDGE,
            "decodeOverlap": LTX_CANARY_OVERLAP,
        }),
    )?;
    // Plan-derived, and pinned to the frozen q4 carrier by the `/target/tier` check above.
    exact(
        "/calibrationFingerprint",
        &json!(ltx_calibration_fingerprint(tier)?),
    )?;
    exact("/fixture", &json!(LTX_BOUNDED_CARRIER_FIXTURE))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_boundedCarrier",
        &json!({
            "identity": LTX_BOUNDED_CARRIER_IDENTITY,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": LTX_CANARY_Q4_INVENTORY_FILES,
                    "bytes": LTX_Q4_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_Q4_INVENTORY_SHA256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    if tier != "q4"
        || geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::BoundedDecode
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err("SC-20254 bounded carrier resolved a foreign tuple or strategy".to_owned());
    }
    Ok(())
}

pub(super) fn ltx_bounded_campaign_spec(tier: &str) -> Result<LtxBoundedCampaignSpec, String> {
    match tier {
        "q4" => Ok(LtxBoundedCampaignSpec {
            tier: "q4",
            identity: LTX_BOUNDED_CAMPAIGN_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_FIXTURE,
            inventory_files: LTX_CANARY_Q4_INVENTORY_FILES,
            inventory_bytes: LTX_Q4_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_Q4_INVENTORY_SHA256,
            projection_bytes: 84_694_536_320,
            story: "SC-20318",
        }),
        "q8" => Ok(LtxBoundedCampaignSpec {
            tier: "q8",
            identity: LTX_BOUNDED_CAMPAIGN_Q8_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_Q8_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_Q8_FIXTURE,
            inventory_files: LTX_CANARY_Q8_INVENTORY_FILES,
            inventory_bytes: LTX_Q8_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_Q8_INVENTORY_SHA256,
            projection_bytes: 93_955_566_576,
            story: "SC-20430",
        }),
        "bf16" => Ok(LtxBoundedCampaignSpec {
            tier: "bf16",
            identity: LTX_BOUNDED_CAMPAIGN_BF16_IDENTITY,
            logical_case_id: LTX_BOUNDED_CAMPAIGN_BF16_LOGICAL_CASE_ID,
            fixture: LTX_BOUNDED_CAMPAIGN_BF16_FIXTURE,
            inventory_files: LTX_CANARY_BF16_INVENTORY_FILES,
            inventory_bytes: LTX_BF16_INVENTORY_BYTES,
            inventory_sha256: LTX_CANARY_BF16_INVENTORY_SHA256,
            projection_bytes: 111_319_657_852,
            story: "SC-20430",
        }),
        other => Err(format!(
            "SC-20430 bounded campaign tier {other:?} is not exactly allowlisted"
        )),
    }
}

/// The three exact promotable bounded campaign entries. The ordinary `run` action continues
/// through the all-row pre-load refusal; the controller may inject this private action only after
/// the canonical harness has selected one allowlisted row and acquired the contained-execution claim.
pub(super) fn validate_ltx_bounded_campaign_entry(
    request: &Value,
    tier: &str,
    geometry: LtxGeometry,
    selection: &MemorySelection,
) -> Result<(), String> {
    let spec = ltx_bounded_campaign_spec(tier)?;
    if protocol::action(request)? != LTX_BOUNDED_CAMPAIGN_ACTION {
        return Err(format!("{} bounded campaign action changed", spec.story));
    }
    let planned = protocol::planned(request)?;
    let exact = |pointer: &str, expected: &Value| -> Result<(), String> {
        let actual = planned.pointer(pointer);
        if actual == Some(expected) {
            Ok(())
        } else {
            Err(format!(
                "{} bounded campaign {pointer} must be {expected}, got {actual:?}",
                spec.story
            ))
        }
    };
    exact("/logicalCaseId", &json!(spec.logical_case_id))?;
    exact("/evidenceScope", &json!("authoritative"))?;
    exact("/backend", &json!("mlx"))?;
    exact("/loadShape", &json!("eager_materialization"))?;
    exact("/target/provider", &json!(LTX_PROVIDER))?;
    exact("/target/modelId", &json!(LTX_PROVIDER))?;
    exact("/target/tier", &json!(spec.tier))?;
    exact("/target/mode", &json!("text_to_video"))?;
    exact("/target/overlay", &json!("none"))?;
    exact("/target/geometry/width", &json!(LTX_CAMPAIGN_ENTRY_WIDTH))?;
    exact("/target/geometry/height", &json!(LTX_CAMPAIGN_ENTRY_HEIGHT))?;
    exact("/target/geometry/batch", &json!(1))?;
    exact("/target/geometry/frames", &json!(LTX_CAMPAIGN_ENTRY_FRAMES))?;
    exact("/strategy/rung", &json!("bounded_decode"))?;
    exact(
        "/strategy/engagedRungs",
        &json!(["resident", "staged_residency", "bounded_decode"]),
    )?;
    exact(
        "/strategy/parameters",
        &json!({
            "decodeTileEdge": LTX_CANARY_TILE_EDGE,
            "decodeOverlap": LTX_CANARY_OVERLAP,
        }),
    )?;
    // The three bounded campaign rows span q4, q8 and bf16, so the identity is resolved from the
    // row's own tier: a bf16 row may not name the q4 cell's fingerprint.
    exact(
        "/calibrationFingerprint",
        &json!(ltx_calibration_fingerprint(spec.tier)?),
    )?;
    exact("/fixture", &json!(spec.fixture))?;
    exact("/negative", &json!(false))?;
    exact("/expectedResult", &json!("passed"))?;
    exact("/modelLoadPolicy", &json!("fresh_per_case"))?;
    exact("/modelLoadGroup", &Value::Null)?;
    exact(
        "/_watchdog",
        &json!({ "maxFootprintBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES }),
    )?;
    exact(
        "/_measurementSafety",
        &json!({
            "disposition": LTX_SAFETY_REFUSED_OPEN,
            "tierInventoryBytes": spec.inventory_bytes,
            "incidentCrashFootprintBytes": LTX_Q4_F305_CRASH_FOOTPRINT_BYTES,
            "incidentCase": "mlx-ltx-2-3-q4-1280x704-f305-fps30-bounded_decode",
            "commonLoad": "complete numeric tier plus shared Gemma stack before geometry-specific work",
            "predictedDecodeBytes": 6_264_848_640_u64,
            "incidentPredictedDecodeBytes": LTX_INCIDENT_PREDICTED_DECODE_BYTES,
            "incidentCalibratedProjectionBytes": spec.projection_bytes,
            "projectionAssumptions": [
                "pinned provider decode cost is the only geometry-varying term used",
                "immutable tier inventory delta is added byte-for-byte",
                "incident binding phase is unknown, so the projection is not a physical-footprint bound and cannot admit execution",
            ],
            "reason": if spec.tier == "q4" {
                "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20318 action is admitted"
            } else {
                "incident-calibrated projection is diagnostic only; ordinary run remains refused and only the exact privately contained SC-20430 action is admitted"
            },
        }),
    )?;
    exact(
        "/_boundedCampaignEntry",
        &json!({
            "identity": spec.identity,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "spatialDecodeTiles": 24,
            "artifact": {
                "repository": protocol::LTX_REPOSITORY,
                "revision": LTX_CANARY_ARTIFACT_REVISION,
                "numericTierInventory": {
                    "files": spec.inventory_files,
                    "bytes": spec.inventory_bytes,
                    "sha256": spec.inventory_sha256,
                },
                "textEncoderInventory": {
                    "files": LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES,
                    "bytes": LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES,
                    "sha256": LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256,
                },
            },
        }),
    )?;
    if geometry.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || geometry.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || geometry.frames != LTX_CAMPAIGN_ENTRY_FRAMES
        || selection.strategy != MemoryStrategy::BoundedDecode
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err(format!(
            "{} bounded campaign resolved a foreign tuple or strategy",
            spec.story
        ));
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    if fps != LTX_CAMPAIGN_ENTRY_FPS || seed != LTX_SEED {
        return Err("SC-20318 bounded campaign fps or seed changed".to_owned());
    }
    let (disposition, _, _) = validate_ltx_safety_evidence(request, tier, geometry, selection)?;
    if disposition != LTX_SAFETY_REFUSED_OPEN {
        return Err("SC-20318 ordinary safety disposition changed".to_owned());
    }
    Ok(())
}

/// The production A/V request. `video_mode` is deliberately left unset so the arm measures the same
/// path a real job takes (`crates/sceneworks-worker/src/video_jobs/ltx.rs` only sets `"no_audio"`
/// when the caller asks for it): the audio latents are denoised jointly with the video regardless,
/// and skipping the audio DECODE would understate the decode phase. `steps` is likewise unset —
/// LTX's distilled schedule is baked (8 + 3 folded to an 11-step bar), so a step count is not a knob.
pub(super) fn ltx_request(geometry: LtxGeometry, fps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "a slow dolly through a sunlit pine forest, drifting motes of pollen, cinematic"
            .to_owned(),
        width: geometry.width,
        height: geometry.height,
        count: 1,
        seed: Some(seed),
        frames: Some(geometry.frames),
        fps: Some(fps),
        ..Default::default()
    }
}

pub(super) fn validate_ltx_bounded_carrier_generation_request(
    request: &GenerationRequest,
) -> Result<(), String> {
    if request.width != LTX_CAMPAIGN_ENTRY_WIDTH
        || request.height != LTX_CAMPAIGN_ENTRY_HEIGHT
        || request.count != 1
        || request.frames != Some(LTX_CAMPAIGN_ENTRY_FRAMES)
        || request.fps != Some(LTX_CAMPAIGN_ENTRY_FPS)
        || request.seed != Some(LTX_SEED)
        || request.video_mode.is_some()
        || request.memory.is_some()
    {
        return Err("SC-20254 constructed a foreign provider generation request".to_owned());
    }
    Ok(())
}

/// One non-promotable diagnostic request. The safety profile is intentionally outside the product
/// envelope and skips only downstream audio decode. The product-envelope profile is the smallest
/// shipped four-second geometry and leaves `video_mode` unset so it follows the ordinary full-A/V
/// provider path. Both force the same bounded-decode carrier without admitting a campaign row.
pub(super) fn ltx_canary_generation_request_for(profile: LtxCanaryProfile) -> GenerationRequest {
    GenerationRequest {
        prompt: profile.prompt().to_owned(),
        width: profile.width(),
        height: profile.height(),
        count: 1,
        seed: Some(LTX_CANARY_SEED),
        frames: Some(profile.frames()),
        fps: Some(profile.fps()),
        video_mode: profile.video_mode().map(str::to_owned),
        memory: Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(LTX_CANARY_TILE_EDGE),
            decode_overlap: Some(LTX_CANARY_OVERLAP),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub(super) fn ltx_canary_generation_request() -> GenerationRequest {
    ltx_canary_generation_request_for(LtxCanaryProfile::Safety)
}

pub(super) fn ltx_product_envelope_canary_generation_request() -> GenerationRequest {
    ltx_canary_generation_request_for(LtxCanaryProfile::ProductEnvelope)
}

pub(super) fn ltx_canary_request_for_provider_admission(
    mut request: GenerationRequest,
) -> Result<GenerationRequest, String> {
    let memory = request
        .memory
        .as_ref()
        .ok_or_else(|| "LTX safety canary omitted bounded-decode memory parameters".to_owned())?;
    if request.width != LTX_CANARY_WIDTH
        || request.height != LTX_CANARY_HEIGHT
        || request.count != 1
        || request.frames != Some(LTX_CANARY_FRAMES)
        || request.fps != Some(LTX_CANARY_FPS)
        || request.seed != Some(LTX_CANARY_SEED)
        || request.video_mode.as_deref() != Some("no_audio")
        || !memory.tile_vae_decode
        || memory.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || memory.decode_overlap != Some(LTX_CANARY_OVERLAP)
    {
        return Err(
            "LTX no-audio admission bridge is private to the exact safety canary tuple".to_owned(),
        );
    }
    // The pinned provider's production request scope rejects every explicit video-mode override.
    // Admit/configure the ordinary unconditional A/V denoise with its default, then restore the
    // strictly smaller no-audio decode request immediately before generation.
    request.video_mode = None;
    Ok(request)
}

pub(super) fn restore_ltx_canary_no_audio_after_configuration(
    request: &mut GenerationRequest,
) -> Result<(), String> {
    if request.video_mode.is_some() {
        return Err(
            "LTX safety canary provider configuration introduced an unexpected video-mode override"
                .to_owned(),
        );
    }
    request.video_mode = Some("no_audio".to_owned());
    Ok(())
}

pub(super) fn scoped_generate_ltx_no_audio_canary(
    generator: &dyn Generator,
    request: GenerationRequest,
    context: &MemoryRunContext,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput, String> {
    let request = ltx_canary_request_for_provider_admission(request)?;
    let mut observed_failure = None;
    scoped_generate_observed_after_configuration(
        generator,
        request,
        context,
        None,
        &mut observed_failure,
        on_progress,
        Some(restore_ltx_canary_no_audio_after_configuration),
    )
}

/// Resolve and validate the `SCENEWORKS_LTX_*` environment family into a tier-exact load spec.
///
/// Two roots under ONE repository: the numeric tier and the `gemma/` co-requisite. The Gemma-3-12B
/// text encoder is a hard load-time requirement of the pinned provider (`resolve_gemma_dir`,
/// sc-13664 removed the env/HF-cache fallbacks), so it is threaded through `LoadSpec::text_encoder`
/// and is snapshot-validated with the same `validate_huggingface_snapshot_root` identity check as
/// the tier root — a mismatched TE would silently change the measured conditioning peak.
pub(super) fn ltx_load_spec(
    request: &Value,
    tier: &str,
    selection: &MemorySelection,
) -> Result<(String, String, PathBuf, PathBuf, LoadSpec), String> {
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
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
        tier,
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
    // `LoadShape` is inert for this provider — `mlx-gen-ltx` never reads `spec.load_shape`, because
    // the two giants are rebuilt inside every `generate` rather than materialized through a block
    // schedule. Refuse a plan that claims otherwise instead of emitting a receipt whose declared
    // materialization shape the run did not use (sc-16482: a receipt may only testify to its own run).
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_offload_policy(OffloadPolicy::Resident)
        .with_load_shape(LoadShape::EagerMaterialization);
    spec.text_encoder = Some(WeightsSource::Dir(text_encoder.clone()));
    if let Some(quant) = selection.tier.quant {
        spec = spec.with_quant(quant);
    }
    Ok((repository, revision, root, text_encoder, spec))
}

pub(super) fn ltx_canary_wired_limit(requested: usize, untouched_memory_limit: usize) -> usize {
    requested.min(untouched_memory_limit / 3 * 2)
}

/// Bind the plan to the provider contract's actual engaged composition. SC-19109 made bounded
/// decode an explicit request-scoped control; the adapter must no longer infer the rung from the
/// host-dependent automatic selector used by historical SC-18810 captures.
pub(super) fn ltx_attested_strategy(
    request: &Value,
    selection: &MemorySelection,
    contract: &mlx_gen::gen_core::MemoryProviderContract,
) -> Result<Value, String> {
    attested_strategy(
        request,
        selection,
        &contract.engaged_composition(selection.strategy),
    )
}

pub(super) fn ltx_complete_sweep(request: &Value) -> Result<Value, String> {
    let mut sweep = protocol::reference_sweep(request, "passed")?;
    // One exact staged tuple per plan row; the row has no parameter axes, so the singleton case is
    // the whole exercised domain.
    sweep["rangeVerified"] = json!(true);
    Ok(sweep)
}

/// Does this run prove sc-10976's load→use→drop actually happened?
///
/// sc-18808 asked a different question — `overall_peak < text_encoder + transformer` — and that
/// question **does not survive the declared envelope**. It infers "the two giants never co-resided"
/// from an aggregate that ACTIVATIONS dominate at scale, so it is only sound while activations are
/// small next to the weights. It was calibrated at one geometry, 768x512 x 97 frames, where the peak
/// (35.56 GB) sat 17.8 GB under the 53.35 GB co-staged bound.
///
/// sc-18810 broke it with a real capture: q8 at **704x1280 x 177 frames** peaked at 54,153,098,156
/// active bytes, 0.8 GB ABOVE that bound, and the arm refused a perfectly good run. The peak was the
/// **decode** phase, not co-resident weights — this sweep's own measured decode relation
/// (~2.75 GB + 322 B per output voxel, against the engine's own committed `3.3e9 + 340 B/voxel`
/// model at `mlx-gen-ltx/src/pipeline.rs:218-228`) predicts 54,108,433,280 for that geometry's
/// 159,498,240 output voxels, i.e. within 0.08% of what was measured.
///
/// The right witness is a PHASE BOUNDARY reading, not a peak, because a boundary reading contains
/// the resident weights and almost nothing else. `denoise_entry` is sampled at
/// `Progress::Step { current: 1 }` — after the text encoder is dropped and `clear_cache()`d and
/// after the AvDiT is built. Across every sc-18810 capture it sits at 20.72-20.80 GB against a
/// 20.61 GB transformer, flat in geometry over a 3.3x latent-token range. Were the Gemma encoder
/// still resident it could not be below 53.3 GB.
///
/// Two-sided on purpose: the upper bound catches the regression the old check was aiming at, and the
/// lower bound catches a progress hook that fires BEFORE the transformer materializes, which would
/// otherwise make the upper bound pass vacuously.
pub(super) fn ltx_staging_is_proven(
    denoise_entry: AllocatorState,
    costaged_bytes: u64,
    text_encoder_bytes: u64,
    transformer_bytes: u64,
) -> Result<(), String> {
    if denoise_entry.active >= costaged_bytes {
        return Err(format!(
            "LTX-2.3 held {} active bytes entering the first denoise step, at or above the \
             {costaged_bytes} bytes the staged text encoder ({text_encoder_bytes}) and transformer \
             ({transformer_bytes}) occupy together — the text encoder was not dropped before the \
             AvDiT materialized, so the staged-residency claim this record makes is not supported \
             by the run",
            denoise_entry.active
        ));
    }
    if denoise_entry.active < transformer_bytes {
        return Err(format!(
            "LTX-2.3 held only {} active bytes entering the first denoise step, below the \
             {transformer_bytes}-byte transformer — the denoise boundary was sampled before the \
             AvDiT materialized, so the staged-residency bound above it proves nothing",
            denoise_entry.active
        ));
    }
    Ok(())
}

/// The eight required scenarios for a gated LTX receipt.
///
/// NOT `protocol::not_run_scenarios(blocker)`. That helper marks all eight `not_run`, which would
/// report loadability as unexecuted — but loadability is the one scenario that has nothing to do
/// with the missing memory-strategy seam, and it demonstrably RAN: the real tier provider plus its
/// Gemma co-requisite loaded from the snapshot-validated roots, and every measurement in the
/// fragment came out of that load. A gated receipt understating what it executed is the same class
/// of dishonesty as a complete one overstating it. `overlay` is emitted `not_run` here and then
/// replaced by `settle_plain_overlay_scenario`, which derives the verdict from the declared target
/// rather than letting this arm assert it.
/// The LTX arm's `predictedPeakBytes`, derived from its own measured phases (sc-18864).
///
/// This was hardcoded `null`, which fails BOTH `EvidenceRecord::mlx_admission_envelope` and the
/// `RequiredNullable::Value` check in the worker's estimate seeding — so a record could not seed an
/// estimate even once its status question was resolved. Nothing about the field is
/// contract-dependent: every other MLX arm in this adapter derives it the same way, from
/// `predicted_ceiling` over its own phases. The missing `MemoryStrategyContract` blocks the
/// SCENARIOS (there is no admission check to interrogate), not this arithmetic.
///
/// It differs from the image arms in ONE respect, deliberately: the ceiling is taken over each
/// phase's `active` peak, NOT over `PhaseMemory::allocator_bytes()`. `allocator_bytes` is the
/// two-instant co-existence bound (see [`PhaseMemory::json`]), and for a staged video capture the
/// allocator cache reaches 72-106 GB, so a ceiling over it predicts ~150 GB of demand on a 128 GiB
/// host for a render that completed comfortably. A gate must budget the RESIDENT demand; the cache
/// is elastic and MLX releases it under pressure.
///
/// The image arms are left on their historical `allocator_bytes` input, and the reason is
/// DIRECTIONAL, not a matter of magnitude. Measured over the 69 MLX image records in
/// `docs/generated/memory-calibration-evidence.json`, the two definitions are NOT close: 39 of 69
/// records carry a phase whose `reclaimableBytes` exceeds 6.4 GB (up to 45.62 GB), 60 of 69 diverge
/// by more than 5%, and the worst phase diverges by 117.7% (`imc-2cd840a85ce33b4f22a9` denoise,
/// krea-2-turbo, 19.98 GB of cache). Per arm the worst divergences are krea-2-turbo 117.7%,
/// flux2-dev 105.6%, z-image-turbo 65.2% and qwen-image 14.4% — only the qwen ladder's 6.45 GB
/// worst-case cache is anywhere near "6.4 GB", and generalising that one arm to the lane was wrong.
/// The shipped image records consequently over-predict their own measured resident peak by
/// 1.05x-2.16x (`imc-b6537074420d51413b38` predicts 93.28 GB against a 43.18 GB resident peak).
///
/// That over-prediction is why sc-19115 keeps the image input unchanged: a ceiling over the
/// co-existence bound is CONSERVATIVE — it never under-predicts — while switching to `active`
/// would LOOSEN shipped admission. [`PredictedPeakBasis`] and its image/video constants are the
/// single policy declaration both lanes consume. The corpus test
/// `resident_peak_counterfactual_only_loosens_shipped_image_admission` re-derives the decision on
/// every run, against whatever MLX image records the committed corpus currently holds and
/// production's scaled foreign-reserve currency: some record predictions change and some host-grid
/// decisions flip, and every flip is refusal -> admission. The totals are deliberately not quoted
/// here — the corpus grows at each calibration campaign and each main sync (69 records at
/// sc-19115, 74 at the sc-18304 sync, 93 at the 2026-08-19 epic-17137 sync).
#[cfg(test)]
pub(super) fn ltx_predicted_peak_bytes(
    conditioning: PhaseMemory,
    denoise: PhaseMemory,
    decode: PhaseMemory,
) -> Value {
    video_predicted_peak_bytes(conditioning, denoise, decode).json()
}

pub(super) fn ltx_context(
    selection: MemorySelection,
    calibration: &MemoryCalibrationIdentity,
    fingerprint: &str,
    geometry: LtxGeometry,
    total_bytes: u64,
    predicted_peak_bytes: u64,
) -> MemoryRunContext {
    MemoryRunContext {
        selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: fingerprint.to_owned(),
        load_shape: calibration.load_shape,
        mode: MemoryMode::Other("text_to_video".to_owned()),
        has_reference: false,
        use_pid: false,
        // As the worker admits it (`video_admission.rs`, "Phase-resolved evidence is not a
        // multi-phase request modifier. LTX's canonical reference-free T2V scope carries this
        // false."). The pinned provider's route gate REJECTS a context carrying `has_phases`
        // (`mlx-gen-ltx/src/memory_strategy.rs` `safety_check`: `|| context.has_phases`), so the
        // `true` this arm used to carry would have failed every capture at the provider's own
        // admission — masked until sc-22738 by the unconditional pre-load refusal.
        has_phases: false,
        geometry: MemoryGeometry {
            width: geometry.width,
            height: geometry.height,
            batch: 1,
            frames: geometry.frames,
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
        evidence_revision: format!("sc-18946@{}", protocol::INFERENCE_PIN),
    }
}

/// The worker's generic activation allowance in bytes — `HeadroomAllowance::GENERIC.total_gb`
/// converted the way `mlx_fit_gate` converts it — read from the shared declaration, never
/// restated here (sc-22738 review). 19,327,352,832 bytes at the current declaration.
pub(super) fn ltx_generic_headroom_bytes() -> u64 {
    sceneworks_core::memory_anchor::gib_to_bytes(
        sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB,
    )
}

/// The worker's fixed unified-memory reserve in bytes — what `mlx_fit_gate::live_request_budget`
/// presents as `MemoryBudget::reserved_headroom_bytes` for every non-Mage engine — read from the
/// shared declaration, never restated here (sc-22738 review). 2,147,483,648 bytes at the current
/// declaration.
pub(super) fn ltx_unified_reserve_bytes() -> u64 {
    sceneworks_core::memory_anchor::gib_to_bytes(
        sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB,
    )
}

/// The worker's floor weights term for one engaged LTX-2.3 composition: the contract's own asset
/// facts lifted onto the SHARED arithmetic (`sceneworks_core::memory_anchor::floor_weights_bytes`,
/// which `mlx_fit_gate::floor_weights_bytes` calls with the same lift), on the video lane's terms
/// (`estimate_floor_weights_bytes`: a zero resident window).
///
/// The pinned LTX-2.3 provider implements `resident`, `staged_residency` and the transient-only
/// `bounded_decode`; it declares `bounded_transformer_residency` UNIMPLEMENTED
/// (`MemoryStrategySupport::Missing`). A composition that engages rung 4 here is therefore a
/// contract or plan drift this arm has no measured basis to price, and it is refused explicitly
/// rather than priced by an arm that was never asked to (sc-22738 review).
pub(super) fn ltx_floor_weights_bytes(
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Result<u64, String> {
    if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
        return Err(format!(
            "LTX-2.3 admission cannot price a composition engaging bounded_transformer_residency \
             (engaged {engaged:?}): the pinned provider declares rung 4 unimplemented and this \
             arm plans only resident/staged_residency"
        ));
    }
    let facts = contract.asset_facts;
    let auxiliary_resident_bytes = contract
        .resident_components()
        .iter()
        .filter(|component| component.kind.is_auxiliary())
        .filter(|component| match component.bounded_by {
            Some(bounding) => !engaged.contains(&bounding),
            None => true,
        })
        .fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        });
    Ok(sceneworks_core::memory_anchor::floor_weights_bytes(
        sceneworks_core::memory_anchor::FloorWeightsFacts {
            conditioning_bytes: facts.conditioning_bytes,
            base_bytes: facts.base_bytes,
            transformer_bytes: facts.transformer_bytes,
            intra_transformer_evicted_bytes: facts
                .transformer_bytes
                .saturating_sub(contract.steady_state_transformer_bytes()),
            auxiliary_resident_bytes,
        },
        sceneworks_core::memory_anchor::FloorWeightsComposition {
            staged: engaged.contains(&MemoryStrategy::StagedResidency),
            bounded_transformer: false,
            resident_window_bytes: 0,
        },
    ))
}

/// The worker's projected peak for one LTX-2.3 cell (`video_admission::profiled_floor_phase_peaks`):
/// `max(weights + generic headroom, decode profile composed onto the weights)`. Pure, so the table
/// tests can drive it over every shipped frame count without an allocator.
pub(super) fn ltx_projected_peak_bytes(
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    weights_bytes: u64,
    profile: mlx_gen::VideoDecodeMemoryProfile,
) -> Result<u64, String> {
    let composed = profile
        .checked_composed_peak(weights_bytes, contract.asset_facts.decoder_bytes)
        .ok_or_else(|| {
            format!(
                "the pinned MLX LTX-2.3 decode profile cannot compose contract weights {weights_bytes} with decoder bytes {}",
                contract.asset_facts.decoder_bytes
            )
        })?;
    Ok(composed.max(weights_bytes.saturating_add(ltx_generic_headroom_bytes())))
}

/// The ordinary LTX-2.3 capture's PRODUCTION admission, run before the load (sc-22738).
///
/// This is the path the worker takes for an unmeasured `(ltx_2_3, tier, mlx)` cell, in the same
/// order and against the same figures: the contract's asset facts give the weights floor, the
/// generic allowance and the engine's own decode profile give the activation term
/// (`profiled_floor_phase_peaks`: the larger of the two), and gen-core's shared safety predicate
/// (`default_memory_strategy_safety_check`, the budget half of the provider's registered
/// `safety_check`) decides whether `predicted_peak_bytes` fits the budget. The provider's full
/// check — the route gate on top of the same predicate — runs again on the loaded generator, so a
/// refusal here is the budget's refusal and a refusal there is the provider's; neither is this
/// arm's opinion. There is deliberately no other exit: a request the budget admits proceeds to the
/// load, and a request it refuses is refused in the budget's own words.
///
/// The budget is presented as `mlx_fit_gate::live_request_budget` presents it for a non-Mage
/// engine: the probed host memory as `total_bytes`, the allocator's live active/cache readings
/// after a `clear_cache` as the committed/reclaimable pair, and the fixed unified-memory reserve
/// as `reserved_headroom_bytes` (which `MemoryBudget::effective_bytes` subtracts). Physical
/// containment of the run itself is the harness's footprint watchdog, not this predicate.
pub(super) fn ltx_ordinary_admission(
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    calibration: &MemoryCalibrationIdentity,
    selection: MemorySelection,
    geometry: LtxGeometry,
    hardware_bytes: u64,
) -> Result<(MemoryRunContext, LtxOrdinaryProjection), String> {
    let engaged = contract.engaged_composition(selection.strategy);
    let weights_bytes = ltx_floor_weights_bytes(contract, &engaged)?;
    let profile = mlx_gen_ltx::conservative_video_decode_memory_profile(
        LTX_PROVIDER,
        geometry.width,
        geometry.height,
        geometry.frames,
    )
    .ok_or_else(|| {
        format!(
            "the pinned MLX LTX-2.3 provider has no conservative decode profile for {}x{} x {} frames",
            geometry.width, geometry.height, geometry.frames
        )
    })?;
    let decode_working_set_bytes = profile.working_set_bytes();
    let predicted_peak_bytes = ltx_projected_peak_bytes(contract, weights_bytes, profile)?;
    clear_cache();
    let mut context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        hardware_bytes,
        predicted_peak_bytes,
    );
    context.budget.committed_bytes = get_active_memory() as u64;
    context.budget.reclaimable_bytes = get_cache_memory() as u64;
    context.budget.reserved_headroom_bytes = ltx_unified_reserve_bytes();
    match mlx_gen::gen_core::default_memory_strategy_safety_check(contract, &context) {
        MemorySafetyDecision::Accept => Ok((
            context,
            LtxOrdinaryProjection {
                weights_bytes,
                decode_working_set_bytes,
                predicted_peak_bytes,
            },
        )),
        MemorySafetyDecision::Reject { reason } => Err(format!(
            "LTX-2.3 admission refused by the production budget before load: {reason} \
             (weights floor {weights_bytes} bytes, generic headroom {} bytes, engine decode \
             working set {decode_working_set_bytes} bytes for {}x{} x {} frames; projected peak \
             {predicted_peak_bytes} bytes against hardware.memoryBytes {hardware_bytes} less the \
             {} byte unified reserve)",
            ltx_generic_headroom_bytes(),
            geometry.width,
            geometry.height,
            geometry.frames,
            ltx_unified_reserve_bytes(),
        )),
    }
}

/// The exact-fit admission budget, expressed in the currency production's gate actually spends
/// (sc-22738).
///
/// THE DEFECT THIS REPLACES. This probe used to set `budget.total_bytes = predicted` and then file
/// `effectiveBudgetBytes: predicted`. Those were the same number only while the arm handed the
/// gate a bare budget. `a1c855ad5` routed this arm through the worker's real presentation
/// (`ltx_ordinary_admission`, which sets `reserved_headroom_bytes` from
/// `mlx_fit_gate::live_request_budget`'s non-Mage branch and the live committed/reclaimable pair),
/// and from that point the budget the gate weighed was
/// `MemoryBudget::effective_bytes` = `total − committed + reclaimable − reserved`, i.e. the
/// unified reserve BELOW the predicted need. `standard_memory_strategy_safety_check`'s only
/// budget clause is `budget.fits(predicted_peak_bytes)`, so the gate refused every LTX-2.3 tier
/// after a full load — correctly. The record's `effectiveBudgetBytes` was the field that was
/// wrong, not the gate.
///
/// THE FIX IS A UNITS FIX, not a retired scenario. `memory_calibration.rs`'s
/// `validate_runtime_complete` requires `exact_fit` to pass with
/// `predictedBytes == effectiveBudgetBytes`, so the scenario the schema asks for is precisely
/// "a budget whose EFFECTIVE bytes equal the predicted need" — the boundary host, not a host whose
/// nameplate total equals the need. Solving `effective_bytes() == predicted` for `total_bytes`
/// with production's own reserve and live snapshot held fixed gives
/// `total = predicted + reserved + committed − reclaimable`; `fits` accepts exact boundaries, and
/// the ceiling term `total − reserved = predicted + committed − reclaimable` cannot bind because
/// the same gate already rejects `reclaimable > committed`. Nothing here is synthesized that
/// production would not present: the reserve, the committed snapshot and the reclaim credit are
/// carried through untouched from the live context, and only the host nameplate moves to the
/// boundary the scenario is defined at.
pub(super) fn ltx_exact_fit_budget(live: MemoryBudget, predicted: u64) -> MemoryBudget {
    MemoryBudget {
        total_bytes: predicted
            .saturating_add(live.reserved_headroom_bytes)
            .saturating_add(live.committed_bytes)
            .saturating_sub(live.reclaimable_bytes),
        ..live
    }
}

/// The SC-20318 quality minimum: one identical-input warm request scope, with its own synchronized
/// phase memory. Cancellation/error/recovery deliberately do not run on this contained row.
pub(super) fn verify_ltx_bounded_warm_repeat(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    selected: &[Image],
    input: LtxLifecycleInput,
) -> Result<LtxBoundedWarmMetrics, String> {
    clear_cache();
    reset_peak_memory();
    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let (warm, fps, audio) = diagnostic_video_frames(
        scoped_generate(
            generator,
            ltx_request(input.geometry, input.fps, input.seed),
            context,
            None,
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
        )?,
        LTX_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    if warm.len() != usize::try_from(input.geometry.frames).unwrap_or(usize::MAX)
        || fps != input.fps
        || [conditioning.active, denoise.active, decode.active].contains(&0)
    {
        return Err("SC-20318 warm repeat changed output or phase shape".to_owned());
    }
    let audio = audio
        .filter(|value| value.samples > 0 && value.sample_rate > 0 && value.channels == 2)
        .ok_or_else(|| "SC-20318 warm repeat did not return non-empty stereo audio".to_owned())?;
    let first = warm
        .first()
        .ok_or_else(|| "SC-20318 warm repeat returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("SC-20318 warm repeat returned a degenerate first frame".to_owned());
    }
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(selected, &warm)?;
    if !ltx_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "SC-20318 warm repeat exceeded the LTX determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    Ok(LtxBoundedWarmMetrics {
        conditioning,
        denoise,
        decode,
        maximum_error,
        mean_error,
        rms_error,
        audio,
    })
}

/// Execute the provider-owned warm/cancel/error request scope on the exact selected rung. One
/// cancellation and one injected error target the rung's binding phase, and each is followed by a
/// deterministic recovery render plus allocator bounds against a clean warm control.
pub(super) fn verify_ltx_lifecycle(
    generator: &dyn Generator,
    context: &MemoryRunContext,
    selected: &[Image],
    input: LtxLifecycleInput,
    phase_sink: &mut dyn LtxCampaignPhaseSink,
) -> Result<LtxLifecycleMetrics, String> {
    let LtxLifecycleInput {
        geometry,
        fps,
        seed,
        fault_phase,
    } = input;
    clear_cache();
    reset_peak_memory();
    phase_sink.mark("lifecycle_warm_repeat")?;
    let (clean_warm, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let clean_warm_peak = get_peak_memory() as u64;
    clear_cache();
    let clean_post_cleanup = AllocatorState::capture_current();
    let bounds = LifecycleMemoryBounds::from_clean_warm(clean_warm_peak, clean_post_cleanup);
    let (maximum_error, mean_error, rms_error) = video_max_mean_rms_abs(selected, &clean_warm)?;
    if !ltx_quality_passes(maximum_error, mean_error, rms_error) {
        return Err(format!(
            "LTX-2.3 clean warm control exceeded the determinism envelope: max={maximum_error:.6}, mean={mean_error:.6}, rms={rms_error:.6}"
        ));
    }
    let mut metrics = LtxLifecycleMetrics {
        clean_warm_peak,
        clean_post_cleanup,
        maximum_error,
        mean_error,
        rms_error,
        measured_clip: LtxClipSummary::of(selected),
        ..Default::default()
    };
    phase_sink.mark("lifecycle_cancel")?;
    let cancelled = ltx_request(geometry, fps, seed);
    let cancel_signal = cancelled.cancel.clone();
    let mut cancel_triggered = false;
    let mut cancel_failure = None;
    let cancel_result = scoped_generate_observed(
        generator,
        cancelled,
        context,
        None,
        &mut cancel_failure,
        &mut |progress| {
            let at_boundary = match fault_phase {
                MemoryPhase::Denoise => matches!(progress, Progress::Step { current: 1, .. }),
                MemoryPhase::Decode => matches!(progress, Progress::Decoding),
                _ => false,
            };
            if at_boundary && !cancel_triggered {
                cancel_triggered = true;
                cancel_signal.cancel();
            }
        },
    );
    let cancel_error = match cancel_result {
        Ok(_) => {
            return Err(format!(
                "LTX-2.3 cancellation completed successfully at {fault_phase:?} instead of returning the typed canceled outcome"
            ));
        }
        Err(error) => error,
    };
    if !cancel_triggered || cancel_failure != Some(ScopedGenerationFailureKind::Canceled) {
        return Err(format!(
            "LTX-2.3 cancellation did not return the typed path at {fault_phase:?}: triggered={cancel_triggered}, failure={cancel_failure:?}, error={cancel_error}"
        ));
    }
    clear_cache();
    let cancel_cleanup = AllocatorState::capture_current();
    metrics.max_fault_post_cleanup = cancel_cleanup;
    if !bounds.allows_retained(cancel_cleanup) {
        return Err(format!(
            "LTX-2.3 cancellation retained {cancel_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    phase_sink.mark("lifecycle_cancel_recovery")?;
    let (cancel_recovery, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let cancel_recovery_peak = get_peak_memory() as u64;
    metrics.max_recovery_peak = cancel_recovery_peak;
    if !bounds.allows_warm_peak(cancel_recovery_peak) {
        return Err(format!(
            "LTX-2.3 post-cancel recovery peaked at {cancel_recovery_peak}, above clean warm {clean_warm_peak} plus 2%"
        ));
    }
    clear_cache();
    let cancel_recovery_cleanup = AllocatorState::capture_current();
    metrics.max_recovery_post_cleanup = cancel_recovery_cleanup;
    if !bounds.allows_retained(cancel_recovery_cleanup) {
        return Err(format!(
            "LTX-2.3 post-cancel recovery retained {cancel_recovery_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    // Recorded, not refused — see `ltx_lifecycle_blockers` for why this reading is evidence. Both
    // references are taken here because only the pair discriminates a fault-caused change from a
    // cold-versus-warm one.
    metrics.cancel_recovery = LtxRecoveryEvidence {
        vs_measured: LtxClipDelta::new(video_max_mean_rms_abs(selected, &cancel_recovery)?),
        vs_clean_warm: LtxClipDelta::new(video_max_mean_rms_abs(&clean_warm, &cancel_recovery)?),
        clip: LtxClipSummary::of(&cancel_recovery),
    };

    phase_sink.mark("lifecycle_error")?;
    let mut injected_failure = None;
    let injected_result = scoped_generate_observed(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        Some(fault_phase),
        &mut injected_failure,
        &mut |_| {},
    );
    let injected = match injected_result {
        Ok(_) => {
            return Err(format!(
                "LTX-2.3 fault injection completed successfully at {fault_phase:?} instead of returning an error"
            ));
        }
        Err(error) => error,
    };
    if injected_failure != Some(ScopedGenerationFailureKind::Error)
        || !injected.contains("injected memory-strategy calibration error")
    {
        return Err(format!(
            "LTX-2.3 error injection returned the wrong outcome at {fault_phase:?}: failure={injected_failure:?}, error={injected}"
        ));
    }
    clear_cache();
    let error_cleanup = AllocatorState::capture_current();
    metrics.max_fault_post_cleanup.active = metrics
        .max_fault_post_cleanup
        .active
        .max(error_cleanup.active);
    metrics.max_fault_post_cleanup.cache = metrics
        .max_fault_post_cleanup
        .cache
        .max(error_cleanup.cache);
    if !bounds.allows_retained(error_cleanup) {
        return Err(format!(
            "LTX-2.3 injected error retained {error_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    reset_peak_memory();
    phase_sink.mark("lifecycle_error_recovery")?;
    let (error_recovery, _, _) = video_frames(scoped_generate(
        generator,
        ltx_request(geometry, fps, seed),
        context,
        None,
        &mut |_| {},
    )?)?;
    let error_recovery_peak = get_peak_memory() as u64;
    metrics.max_recovery_peak = metrics.max_recovery_peak.max(error_recovery_peak);
    if !bounds.allows_warm_peak(error_recovery_peak) {
        return Err(format!(
            "LTX-2.3 post-error recovery peaked at {error_recovery_peak}, above clean warm {clean_warm_peak} plus 2%"
        ));
    }
    clear_cache();
    let error_recovery_cleanup = AllocatorState::capture_current();
    metrics.max_recovery_post_cleanup.active = metrics
        .max_recovery_post_cleanup
        .active
        .max(error_recovery_cleanup.active);
    metrics.max_recovery_post_cleanup.cache = metrics
        .max_recovery_post_cleanup
        .cache
        .max(error_recovery_cleanup.cache);
    if !bounds.allows_retained(error_recovery_cleanup) {
        return Err(format!(
            "LTX-2.3 post-error recovery retained {error_recovery_cleanup:?} above clean warm {clean_post_cleanup:?} plus {} bytes",
            bounds.tolerance_bytes,
        ));
    }
    metrics.error_recovery = LtxRecoveryEvidence {
        vs_measured: LtxClipDelta::new(video_max_mean_rms_abs(selected, &error_recovery)?),
        vs_clean_warm: LtxClipDelta::new(video_max_mean_rms_abs(&clean_warm, &error_recovery)?),
        clip: LtxClipSummary::of(&error_recovery),
    };
    Ok(metrics)
}

pub(super) fn validate_ltx_canary_plan_for(
    request: &Value,
    profile: LtxCanaryProfile,
) -> Result<MemorySelection, String> {
    if protocol::action(request)? != profile.action() {
        return Err(format!(
            "LTX diagnostic canary action must be {:?}",
            profile.action()
        ));
    }
    let planned = protocol::planned(request)?;
    if protocol::plan_optional_bool(planned, "/_diagnosticOnly") != Some(true) {
        return Err("LTX safety canary requires planned._diagnosticOnly == true".to_owned());
    }
    if protocol::plan_optional_str(planned, "/evidenceScope") != Some("fixture") {
        return Err("LTX safety canary requires non-promotable fixture evidenceScope".to_owned());
    }
    if protocol::plan_optional_str(planned, "/backend") != Some("mlx") {
        return Err("LTX safety canary requires planned.backend == mlx".to_owned());
    }
    let engaged_rungs = planned
        .pointer("/strategy/engagedRungs")
        .and_then(Value::as_array)
        .ok_or_else(|| "LTX safety canary requires strategy.engagedRungs".to_owned())?
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "LTX safety canary strategy.engagedRungs must contain strings".to_owned())?;
    if engaged_rungs != ["resident", "staged_residency", "bounded_decode"] {
        return Err(
            "LTX safety canary requires exact resident/staged_residency/bounded_decode rungs"
                .to_owned(),
        );
    }
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    let target = protocol::plan_object(planned, "/target")?;
    for field in ["provider", "modelId"] {
        if protocol::target_optional_str(target, field) != Some(LTX_PROVIDER) {
            return Err(format!(
                "LTX safety canary target.{field} must be {LTX_PROVIDER:?}"
            ));
        }
    }
    if protocol::target_optional_str(target, "tier") != Some("q4")
        || protocol::target_optional_str(target, "mode") != Some("text_to_video")
        || protocol::target_optional_str(target, "overlay") != Some("none")
    {
        return Err("LTX safety canary requires q4 text_to_video with overlay none".to_owned());
    }
    let geometry = target
        .get("geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let exact_geometry = [
        ("width", u64::from(profile.width())),
        ("height", u64::from(profile.height())),
        ("batch", 1),
        ("frames", u64::from(profile.frames())),
    ];
    for (field, expected) in exact_geometry {
        if geometry.get(field).and_then(Value::as_u64) != Some(expected) {
            return Err(format!(
                "LTX safety canary geometry.{field} must be {expected}"
            ));
        }
    }
    if protocol::plan_optional_str(planned, "/fixture") != Some(profile.fixture()) {
        return Err(format!(
            "LTX safety canary fixture must be {:?}",
            profile.fixture()
        ));
    }
    // The canary is the frozen q4 row — the `target.tier` check above refuses anything else — so
    // the identity it must name is the q4 member of the family, never the calibrated tier's key.
    let expected_fingerprint = ltx_calibration_fingerprint("q4")?;
    let planned_fingerprint = protocol::plan_str(planned, "/calibrationFingerprint")?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "LTX safety canary fingerprint does not match the pinned provider: plan={planned_fingerprint}, q4 identity={expected_fingerprint}"
        ));
    }
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("LTX safety canary requires eager_materialization".to_owned());
    }
    if protocol::plan_optional_u64(planned, "/_watchdog/maxFootprintBytes")
        != Some(LTX_CANARY_MAX_FOOTPRINT_BYTES)
    {
        return Err(format!(
            "LTX safety canary watchdog ceiling must be the SC-18808 co-staged bound {LTX_CANARY_MAX_FOOTPRINT_BYTES}"
        ));
    }
    let canary = planned
        .get("_canary")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires planned._canary".to_owned())?;
    if canary.get("identity").and_then(Value::as_str) != Some(profile.identity())
        || canary.get("videoMode").and_then(Value::as_str) != Some(profile.video_mode_identity())
        || canary.get("fps").and_then(Value::as_u64) != Some(u64::from(profile.fps()))
        || canary.get("seed").and_then(Value::as_u64) != Some(LTX_CANARY_SEED)
    {
        return Err(format!(
            "LTX safety canary requires identity {:?}, video mode {:?}, fps {}, seed {LTX_CANARY_SEED}",
            profile.identity(),
            profile.video_mode_identity(),
            profile.fps(),
        ));
    }
    let artifact = planned
        .get("_artifact")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires planned._artifact".to_owned())?;
    let numeric_inventory = artifact
        .get("numericTierInventory")
        .and_then(Value::as_object)
        .ok_or_else(|| "LTX safety canary requires the numeric-tier inventory".to_owned())?;
    if artifact.get("repository").and_then(Value::as_str) != Some(protocol::LTX_REPOSITORY)
        || artifact.get("revision").and_then(Value::as_str) != Some(LTX_CANARY_ARTIFACT_REVISION)
        || numeric_inventory.get("files").and_then(Value::as_u64)
            != Some(LTX_CANARY_Q4_INVENTORY_FILES)
        || numeric_inventory.get("bytes").and_then(Value::as_u64) != Some(LTX_Q4_INVENTORY_BYTES)
        || numeric_inventory.get("sha256").and_then(Value::as_str)
            != Some(LTX_CANARY_Q4_INVENTORY_SHA256)
    {
        return Err(
            "LTX safety canary requires the exact immutable q4 artifact inventory".to_owned(),
        );
    }
    let text_encoder = artifact
        .get("textEncoderInventory")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            "LTX safety canary requires the immutable bundled Gemma inventory".to_owned()
        })?;
    if text_encoder.get("files").and_then(Value::as_u64)
        != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_FILES)
        || text_encoder.get("bytes").and_then(Value::as_u64)
            != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_BYTES)
        || text_encoder.get("sha256").and_then(Value::as_str)
            != Some(LTX_CANARY_TEXT_ENCODER_INVENTORY_SHA256)
    {
        return Err(
            "LTX safety canary requires the exact immutable bundled Gemma inventory".to_owned(),
        );
    }
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary hardware.memoryBytes must be an integer".to_owned())?;
    if hardware_bytes < LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 {
        return Err(format!(
            "LTX safety canary host memory {hardware_bytes} must preserve two {LTX_CANARY_MAX_FOOTPRINT_BYTES}-byte stop boundaries"
        ));
    }
    let selection = planned_selection(request)?;
    if selection.strategy != MemoryStrategy::BoundedDecode
        || selection.tier.quant != Some(Quant::Q4)
        || selection.parameters.decode_tile_edge != Some(LTX_CANARY_TILE_EDGE)
        || selection.parameters.decode_overlap != Some(LTX_CANARY_OVERLAP)
        || selection.parameters.attention_chunk_size.is_some()
        || selection.parameters.transformer_window_size.is_some()
        || selection.parameters.transformer_window_component.is_some()
    {
        return Err(format!(
            "LTX safety canary requires exact bounded_decode q4 {LTX_CANARY_TILE_EDGE}/{LTX_CANARY_OVERLAP} selection"
        ));
    }
    Ok(selection)
}

#[cfg(test)]
pub(super) fn validate_ltx_canary_plan(request: &Value) -> Result<MemorySelection, String> {
    validate_ltx_canary_plan_for(request, LtxCanaryProfile::Safety)
}

#[cfg(test)]
pub(super) fn validate_ltx_product_envelope_canary_plan(
    request: &Value,
) -> Result<MemorySelection, String> {
    validate_ltx_canary_plan_for(request, LtxCanaryProfile::ProductEnvelope)
}

pub(super) fn wait_for_ltx_phase_acknowledgement(
    acknowledgements: &std::sync::mpsc::Receiver<Result<(usize, String), String>>,
    sequence: usize,
    name: &str,
    timeout: Duration,
) -> Result<(), String> {
    let acknowledged = acknowledgements.recv_timeout(timeout).map_err(|error| {
        format!("wait for SC-20216 provider phase {name} acknowledgement: {error}")
    })??;
    if acknowledged != (sequence, name.to_owned()) {
        return Err(format!(
            "SC-20216 watchdog acknowledged a foreign provider phase: expected {sequence} {name}, got {} {}",
            acknowledged.0, acknowledged.1
        ));
    }
    Ok(())
}

pub(super) fn consume_ltx_canary_watchdog_attestation(
    request: &Value,
) -> Result<LtxCanaryWatchdogAttestation, String> {
    let socket_path = std::env::var("SCENEWORKS_MEMORY_WATCHDOG_SOCKET")
        .map_err(|_| "LTX safety canary requires the live external watchdog channel".to_owned())?;
    let stream = UnixStream::connect(&socket_path)
        .map_err(|error| format!("connect to live LTX canary watchdog: {error}"))?;
    consume_ltx_canary_watchdog_attestation_stream(request, stream)
}

pub(super) fn consume_ltx_canary_watchdog_attestation_stream(
    request: &Value,
    mut stream: UnixStream,
) -> Result<LtxCanaryWatchdogAttestation, String> {
    // The watchdog creates this one-run socket in a private random directory only after its
    // launch-owned process group is anchored. Direct stdin invocation has no socket and therefore
    // refuses before installing MLX limits or reading any model path. The monitor does not release
    // allocation work until the ACK/GO pair completes around a second full telemetry sample.
    let timeout = Some(Duration::from_secs(1));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| format!("configure LTX watchdog read timeout: {error}"))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| format!("configure LTX watchdog write timeout: {error}"))?;
    let payload: Value = serde_json::from_str(&read_watchdog_line(&mut stream)?)
        .map_err(|error| format!("parse LTX canary watchdog attestation: {error}"))?;
    if payload.get("protocol").and_then(Value::as_str) != Some(LTX_CANARY_WATCHDOG_PROTOCOL) {
        return Err("LTX safety canary watchdog protocol changed".to_owned());
    }
    let expected_phase_contract: Option<(&str, &[&str])> =
        match request.get("action").and_then(Value::as_str) {
            Some(LTX_CAMPAIGN_ENTRY_ACTION) => {
                Some((LTX_CAMPAIGN_ENTRY_PHASE_PROFILE, &LTX_PROVIDER_PHASE_NAMES))
            }
            Some(LTX_BOUNDED_CARRIER_ACTION) => Some((
                LTX_BOUNDED_CARRIER_PHASE_PROFILE,
                &LTX_BOUNDED_CARRIER_PHASE_NAMES,
            )),
            Some(LTX_BOUNDED_CAMPAIGN_ACTION) => Some((
                LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
                &LTX_BOUNDED_CARRIER_PHASE_NAMES,
            )),
            _ => None,
        };
    let provider_phase_protocol = payload.get("providerPhaseProtocol").and_then(Value::as_str);
    let provider_phase_profile = payload.get("providerPhaseProfile").and_then(Value::as_str);
    let provider_phase_names = payload.get("providerPhases");
    if let Some((expected_profile, expected_names)) = expected_phase_contract {
        if provider_phase_protocol != Some(LTX_PROVIDER_PHASE_PROTOCOL)
            || provider_phase_profile != Some(expected_profile)
            || provider_phase_names != Some(&json!(expected_names))
        {
            return Err(
                "watchdog omitted the exact action-bound authenticated provider phase contract"
                    .to_owned(),
            );
        }
    } else if provider_phase_protocol.is_some()
        || provider_phase_profile.is_some()
        || provider_phase_names.is_some()
    {
        return Err(
            "provider phase telemetry is restricted to exact contained LTX actions".to_owned(),
        );
    }
    let nonce = payload
        .get("nonce")
        .and_then(Value::as_str)
        .ok_or_else(|| "LTX safety canary watchdog attestation omitted nonce".to_owned())?;
    if nonce.len() != 64 || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("LTX safety canary watchdog nonce is not 32-byte hexadecimal".to_owned());
    }
    let max_footprint_bytes = payload
        .get("maxFootprintBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted maxFootprintBytes".to_owned())?;
    let max_runtime_seconds = payload
        .get("maxRuntimeSeconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| "LTX safety canary watchdog omitted maxRuntimeSeconds".to_owned())?;
    let host_memory_bytes = payload
        .get("hostMemoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted hostMemoryBytes".to_owned())?;
    let min_memory_free_bytes = payload
        .get("minMemoryFreeBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted minMemoryFreeBytes".to_owned())?;
    let min_initial_memory_free_bytes = payload
        .get("minInitialMemoryFreeBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary watchdog omitted minInitialMemoryFreeBytes".to_owned())?;
    let requested_host_memory = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "LTX safety canary hardware.memoryBytes must be an integer".to_owned())?;
    let telemetry_resolution = host_memory_bytes.div_ceil(100);
    if max_footprint_bytes != LTX_CANARY_MAX_FOOTPRINT_BYTES
        || max_runtime_seconds != LTX_CANARY_MAX_RUNTIME_SECONDS
        || host_memory_bytes != requested_host_memory
        || min_initial_memory_free_bytes
            != 2 * LTX_CANARY_MAX_FOOTPRINT_BYTES + telemetry_resolution
        || payload.get("minInitialMemoryFreePercent").is_some()
        || min_memory_free_bytes != LTX_CANARY_MAX_FOOTPRINT_BYTES + telemetry_resolution
        || payload.get("minSwapFreeBytes").is_some()
    {
        return Err(
            "LTX safety canary watchdog did not attest the exact reviewed bounds".to_owned(),
        );
    }
    stream
        .write_all(format!("ACK {nonce}\n").as_bytes())
        .map_err(|error| format!("acknowledge LTX canary watchdog: {error}"))?;
    if read_watchdog_line(&mut stream)? != format!("GO {nonce}") {
        return Err("LTX safety canary watchdog did not release allocation work".to_owned());
    }
    Ok(LtxCanaryWatchdogAttestation {
        max_footprint_bytes,
        max_runtime_seconds,
        host_memory_bytes,
        min_initial_memory_free_bytes,
        min_memory_free_bytes,
        nonce: nonce.to_owned(),
        stream: Some(stream),
    })
}

/// One deliberately tiny, non-ingestible real-weight probe for the permanent-pin tiled-release
/// repair. This is not a campaign arm: it executes one render, never the six-render lifecycle
/// sweep, and emits a status the calibration harness schema rejects.
pub(super) fn ltx_canary_ones_cache_bytes() -> Result<u64, String> {
    LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION
        .checked_add(LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION)
        .and_then(|elements| elements.checked_mul(BFLOAT16_BYTES_PER_ELEMENT))
        .ok_or_else(|| "LTX safety canary ONES_CACHE byte arithmetic overflowed".to_owned())
}

pub(super) fn validate_ltx_canary_pre_provider(pre_provider: AllocatorState) -> Result<(), String> {
    if pre_provider.cache != 0 {
        return Err(format!(
            "LTX safety canary preProviderCacheBytes {} did not attest the cleared cache 0",
            pre_provider.cache
        ));
    }
    Ok(())
}

pub(super) fn validate_ltx_canary_cleanup(
    pre_provider: AllocatorState,
    post_cleanup: AllocatorState,
    expected_persistent_active: u64,
) -> Result<(), String> {
    let expected_post_active = pre_provider
        .active
        .checked_add(expected_persistent_active)
        .ok_or_else(|| "LTX safety canary cleanup active-byte arithmetic overflowed".to_owned())?;
    if post_cleanup.active != expected_post_active {
        return Err(format!(
            "LTX safety canary postCleanupActiveBytes {} did not equal pre-provider active {} plus intentional persistent active {} = {}",
            post_cleanup.active,
            pre_provider.active,
            expected_persistent_active,
            expected_post_active
        ));
    }
    if post_cleanup.cache != pre_provider.cache {
        return Err(format!(
            "LTX safety canary postCleanupCacheBytes {} did not return to preProviderCacheBytes {}",
            post_cleanup.cache, pre_provider.cache
        ));
    }
    Ok(())
}

pub(super) fn run_ltx_canary_for(
    request: &Value,
    profile: LtxCanaryProfile,
) -> Result<Value, String> {
    let selection = validate_ltx_canary_plan_for(request, profile)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let watchdog_lease = watchdog.start_lease()?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let geometry = LtxGeometry {
        width: profile.width(),
        height: profile.height(),
        frames: profile.frames(),
        latent_frames: 1 + (profile.frames() - 1) / LTX_TEMPORAL_SCALE,
    };
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, "q4", &selection)?;
    if revision != LTX_CANARY_ARTIFACT_REVISION {
        return Err(format!(
            "LTX safety canary artifact revision must be {LTX_CANARY_ARTIFACT_REVISION}, got {revision}"
        ));
    }
    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected canary selection: {error}"))?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    // The canary loads the frozen q4 artifact (`ltx_load_spec(request, "q4", ..)` above), so the
    // loaded contract must publish the q4 member of the identity family.
    let expected_fingerprint = ltx_calibration_fingerprint("q4")?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned LTX-2.3 contract fingerprint changed: expected {expected_fingerprint}, got {}",
            calibration.fingerprint
        ));
    }
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    if !decode_plan.tiling_engaged()
        || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
        || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
    {
        return Err(
            "LTX safety canary did not resolve the exact multi-tile decode carrier".to_owned(),
        );
    }
    let spatial_decode_tile_count = decode_plan.spatial_tile_count(geometry)?;
    if spatial_decode_tile_count <= 1 {
        return Err("LTX safety canary resolved no physical multi-tile decode".to_owned());
    }
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 q4 canary provider: {error}"))?;
    if generator.memory_strategy_contract() != Some(&contract) {
        return Err("loaded LTX-2.3 canary contract differs from the registry contract".to_owned());
    }
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 provider rejected the exact canary budget".to_owned());
    }
    clear_cache();
    reset_peak_memory();
    let mut conditioning = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut denoise = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut observe_progress = |progress| match progress {
        Progress::Step { current: 1, .. } => {
            conditioning = PhaseMemory::capture();
            reset_peak_memory();
        }
        Progress::Decoding => {
            denoise = PhaseMemory::capture();
            reset_peak_memory();
        }
        _ => {}
    };
    let generated = match profile {
        LtxCanaryProfile::Safety => scoped_generate_ltx_no_audio_canary(
            generator.as_ref(),
            ltx_canary_generation_request(),
            &context,
            &mut observe_progress,
        ),
        LtxCanaryProfile::ProductEnvelope => scoped_generate(
            generator.as_ref(),
            ltx_product_envelope_canary_generation_request(),
            &context,
            None,
            &mut observe_progress,
        ),
    }?;
    let (frames, fps, audio) = diagnostic_video_frames(generated, LTX_VIDEO_LABEL)?;
    let decode = PhaseMemory::capture();
    let expected_audio = profile == LtxCanaryProfile::ProductEnvelope;
    if frames.len() != profile.frames() as usize || fps != profile.fps() {
        return Err(format!(
            "LTX safety canary returned frames={}, fps={fps}, audio={audio:?}",
            frames.len(),
        ));
    }
    validate_diagnostic_audio(profile, audio)?;
    let first = frames
        .first()
        .ok_or_else(|| "LTX safety canary returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("LTX safety canary returned a degenerate first frame".to_owned());
    }
    let peak_active = [conditioning.active, denoise.active, decode.active]
        .into_iter()
        .max()
        .unwrap_or(0);
    drop(frames);
    drop(generator);
    clear_cache();
    let cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, cleanup, expected_persistent_active)?;
    let planned_artifact = protocol::planned(request)?
        .get("_artifact")
        .cloned()
        .ok_or_else(|| "LTX safety canary lost planned._artifact".to_owned())?;
    let mut strategy = ltx_attested_strategy(request, &context.selection, &contract)?;
    strategy["spatialDecodeTiles"] = json!(spatial_decode_tile_count);
    watchdog_lease.complete()?;
    Ok(json!({
        "status": profile.completion_status(),
        "canaryIdentity": profile.identity(),
        "diagnosticOnly": true,
        "promotable": false,
        "ingestible": false,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "calibrationFingerprint": calibration.fingerprint,
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
            "tierRoot": root,
            "textEncoderRoot": text_encoder_root,
            "numericTierInventory": planned_artifact["numericTierInventory"],
            "textEncoderInventory": planned_artifact["textEncoderInventory"],
        },
        "target": {
            "provider": LTX_PROVIDER,
            "tier": "q4",
            "geometry": {
                "width": profile.width(),
                "height": profile.height(),
                "frames": profile.frames(),
                "fps": profile.fps(),
            },
            "videoMode": profile.video_mode_identity(),
            "audio": expected_audio,
        },
        "strategy": strategy,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
            "source": "conservative-stop-from-sc-18808-q8-costaged-safetensor-arithmetic-not-physical-bound",
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "observedMemory": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "peakActiveBytes": peak_active,
            "postCleanupActiveBytes": cleanup.active,
            "postCleanupCacheBytes": cleanup.cache,
        },
        "output": {
            "frames": profile.frames(),
            "fps": profile.fps(),
            "audio": {
                "present": audio.is_some(),
                "samples": audio.map_or(0, |value| value.samples),
                "sampleRate": audio.map_or(0, |value| value.sample_rate),
                "channels": audio.map_or(0, |value| value.channels),
            },
            "frameTimelineSeconds": f64::from(profile.frames() - 1) / f64::from(profile.fps()),
            "firstFrameNondegenerate": true,
        },
        "capturedAt": protocol::captured_at(),
    }))
}

pub(super) fn run_ltx_canary(request: &Value) -> Result<Value, String> {
    run_ltx_canary_for(request, LtxCanaryProfile::Safety)
}

pub(super) fn run_ltx_product_envelope_canary(request: &Value) -> Result<Value, String> {
    run_ltx_canary_for(request, LtxCanaryProfile::ProductEnvelope)
}

pub(super) fn run_ltx_with_admission(
    request: &Value,
    admission: LtxRunAdmission,
    phase_sink: &mut dyn LtxCampaignPhaseSink,
) -> Result<Value, String> {
    let geometry = validate_ltx_target(request)?;
    // The lane's render plan (sc-22738): a video capture is ONE measured render, no warm pass.
    // The SC-20318 campaign entries below are lifecycle PROOFS, not catalog captures, and keep
    // their own render counts; the ordinary capture honours this.
    let capture = protocol::capture_policy(request)?.require_video(LTX_VIDEO_LABEL)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    let load_shape = planned_load_shape(request)?;
    if load_shape != LoadShape::EagerMaterialization {
        return Err(
            "the pinned mlx-gen-ltx contract calibrates only eager_materialization; the provider \
             stages text, transformer and decode phases inside each request rather than exposing a \
             deferred block loader"
                .to_owned(),
        );
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?; // shared numeric-tier parser
    if !matches!(tier, "q4" | "q8" | "bf16") {
        return Err(format!(
            "the MLX LTX-2.3 plan supports only the manifest's q4, q8 and bf16 tiers; tier {tier:?} \
             is not capturable"
        ));
    }
    let (fps, seed) = planned_ltx_capture(request, tier, geometry)?;
    // The identity is derived from the PLANNED tier: the engine keys calibration on
    // `(provider, artifact-proven tier)`, so a q4 plan may not name the q8 cell's key.
    let expected_fingerprint = ltx_calibration_fingerprint(tier)?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "plan/adapter calibration mismatch: plan={planned_fingerprint}, MLX LTX-2.3 arm \
             implements {expected_fingerprint} for tier {tier}"
        ));
    }
    match admission {
        // sc-22738: no arm-local gate. The ordinary capture is admitted or refused by the
        // production budget below (`ltx_ordinary_admission`), never by this arm's opinion.
        LtxRunAdmission::Ordinary => {}
        LtxRunAdmission::CampaignEntry => {
            validate_ltx_campaign_entry(request, tier, geometry, &selection)?
        }
        LtxRunAdmission::BoundedCampaignEntry => {
            validate_ltx_bounded_campaign_entry(request, tier, geometry, &selection)?
        }
    }
    // Read after every plan check and before any environment, model-path or provider work: the
    // budget is a fact about the host, not the plan, and a malformed plan must name its own error.
    let hardware_bytes = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "run request.hardware.memoryBytes must be an integer".to_owned())?;
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, tier, &selection)?;
    // Read BEFORE the load so the staging bound below is grounded in the artifact on disk rather
    // than in an allocator reading that a broken staging would itself corrupt. The staged text
    // phase is `build_text_encoder`, which materializes the Gemma snapshot AND the tier's
    // `connector.safetensors` into one `LtxTextEncoder` — so the connector belongs on the TE side of
    // the bound, not with the resident small components.
    let component_bytes = |name: &str| {
        std::fs::metadata(root.join(name))
            .map_err(|error| format!("stat the {tier} {name}: {error}"))
            .map(|metadata| metadata.len())
    };
    let connector_bytes = component_bytes("connector.safetensors")?;
    let gemma_bytes = safetensors_bytes(&text_encoder_root)?;
    let text_encoder_bytes = gemma_bytes.saturating_add(connector_bytes);
    let transformer_bytes = component_bytes("transformer.safetensors")?;
    let tier_bytes = safetensors_bytes(&root)?;

    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected planned selection: {error}"))?;
    let strategy = ltx_attested_strategy(request, &selection, &contract)?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "pinned LTX-2.3 contract fingerprint changed for tier {tier}: expected {expected_fingerprint}, got {}",
            calibration.fingerprint
        ));
    }
    if planned_fingerprint != calibration.fingerprint {
        return Err(format!(
            "plan/provider calibration mismatch: plan={planned_fingerprint}, pinned provider={}",
            calibration.fingerprint
        ));
    }
    // Host-dependent and process-global: resolve only after every cheap target, fixture,
    // fingerprint, and provider-contract check has passed, so an injected test budget cannot mask
    // a deterministic malformed-plan error. This remains before generator load or capture.
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    let spatial_decode_tiles = decode_plan.spatial_tile_count(geometry)?;
    if admission == LtxRunAdmission::BoundedCampaignEntry
        && (!decode_plan.tiling_engaged()
            || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
            || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
            || spatial_decode_tiles != 24)
    {
        return Err("SC-20318 did not resolve the exact 24-tile 192/64 carrier".to_owned());
    }
    // THE PRODUCTION ADMISSION, before the load: the worker's projection for this cell against
    // this host's budget, decided by gen-core's shared predicate. Accept proceeds to the load;
    // Reject is returned in the budget's own words. Every admission mode runs it — the canary
    // profiles add their own tighter ceilings on top, they do not replace the production one.
    let (context, projection) =
        ltx_ordinary_admission(&contract, calibration, selection, geometry, hardware_bytes)?;
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 {tier} provider: {error}"))?;
    let loaded_contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded LTX-2.3 generator exposed no memory contract".to_owned())?;
    if loaded_contract != &contract {
        return Err(
            "loaded LTX-2.3 generator contract differs from the registry contract".to_owned(),
        );
    }
    // The provider's OWN registered check on the loaded generator — the same predicate plus its
    // route gate — with the same context. A rejection is the provider's, quoted verbatim.
    if let MemorySafetyDecision::Reject { reason } =
        generator.memory_strategy_safety_check(&context)
    {
        return Err(format!(
            "LTX-2.3 admission refused by the pinned provider after load: {reason}"
        ));
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.3 admission accepted an unknown/zero memory budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint = "stale-ltx-2-3-fingerprint".to_owned();
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("LTX-2.3 admission accepted stale calibration evidence".to_owned());
    }

    let conditioning = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    let denoise = Cell::new(PhaseMemory {
        active: 0,
        cache: 0,
    });
    // Live allocator readings at the two staging handoffs, distinct from the per-phase PEAKS above.
    let denoise_entry = Cell::new(AllocatorState::default());
    let decode_entry = Cell::new(AllocatorState::default());
    clear_cache();
    reset_peak_memory();
    let pre_rung_active = get_active_memory() as u64;
    let pre_rung_cache = get_cache_memory() as u64;
    phase_sink.mark("primary_conditioning")?;
    let (measured, output_fps, audio) = diagnostic_video_frames(
        scoped_generate(
            generator.as_ref(),
            ltx_request(geometry, fps, seed),
            &context,
            None,
            &mut |progress| {
                match progress {
                    // LTX emits no boundary between the staged text phase and the DiT build, so
                    // this phase legitimately spans BOTH giants' materializations; the staging
                    // bound below is what proves they did not co-reside.
                    Progress::Step { current: 1, .. } => {
                        if let Err(error) = phase_sink.mark("primary_denoise") {
                            eprintln!("{error}");
                            std::process::abort();
                        }
                        conditioning.set(PhaseMemory::capture());
                        denoise_entry.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    Progress::Decoding => {
                        if let Err(error) = phase_sink.mark("primary_decode") {
                            eprintln!("{error}");
                            std::process::abort();
                        }
                        denoise.set(PhaseMemory::capture());
                        decode_entry.set(AllocatorState::capture_current());
                        reset_peak_memory();
                    }
                    _ => {}
                }
            },
        )?,
        LTX_VIDEO_LABEL,
    )?;
    let decode = PhaseMemory::capture();
    let conditioning = conditioning.get();
    let denoise = denoise.get();
    let denoise_entry = denoise_entry.get();
    let decode_entry = decode_entry.get();
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err(
            "a synchronized LTX-2.3 lifecycle phase reported a zero active peak".to_owned(),
        );
    }
    // sc-22738 sweep: an equality here is CORRECT for LTX, and it is not a coincidence.
    // `VaeTiling::LTX` is CAUSAL at ×8, and the engine refuses any request off the `1 + 8k` lattice
    // (`mlx-gen-ltx/src/model.rs:2755-2760`) rather than snapping it, so `out_f = 1 + (f_lat−1)·8`
    // is exactly the requested count and the engine has no clip to trim. The non-causal families
    // (Bernini, SCAIL-2, the Wan A14B experts) cannot use a bare equality and do not;
    // `the_arms_that_keep_a_bare_frame_equality_render_exactly_what_they_request` holds this apart
    // at every pin.
    if measured.len() as u32 != geometry.frames {
        return Err(format!(
            "LTX-2.3 rendered {} frames for a {}-frame request",
            measured.len(),
            geometry.frames
        ));
    }
    if output_fps != fps {
        return Err(format!(
            "LTX-2.3 returned fps {output_fps} for a {fps} fps request"
        ));
    }
    let audio = audio
        .filter(|audio| audio.samples > 0 && audio.sample_rate > 0 && audio.channels > 0)
        .ok_or_else(|| {
            "LTX-2.3 full-A/V campaign render returned no non-empty audio track".to_owned()
        })?;
    if admission == LtxRunAdmission::BoundedCampaignEntry && audio.channels != 2 {
        return Err("SC-20318 selected render did not return stereo audio".to_owned());
    }
    let first = measured
        .first()
        .ok_or_else(|| "LTX-2.3 campaign render returned no first frame".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("LTX-2.3 campaign render returned a degenerate first frame".to_owned());
    }
    let overall = PhaseMemory::overall(&[conditioning, denoise, decode]);

    // THE STAGED-RESIDENCY PROOF, and the reason the record may claim rung 1 at all.
    let costaged_bytes = text_encoder_bytes.saturating_add(transformer_bytes);
    ltx_staging_is_proven(
        denoise_entry,
        costaged_bytes,
        text_encoder_bytes,
        transformer_bytes,
    )?;
    // The second handoff: the DiT is dropped and the cache cleared before the VAE decode, so the
    // allocator at the decode boundary must hold less than it did entering the denoise.
    if decode_entry.active >= denoise_entry.active {
        return Err(format!(
            "LTX-2.3 held {} active bytes entering decode against {} entering denoise; the \
             transformer was not released before the VAE decode",
            decode_entry.active, denoise_entry.active
        ));
    }

    let predicted_peaks = video_predicted_peak_bytes(conditioning, denoise, decode);
    let predicted = predicted_peaks.overall;
    // The exact-fit scenario, posed the way the gate weighs it: see [`ltx_exact_fit_budget`]. The
    // effective figure is READ BACK off the budget rather than assumed, so the record's
    // `effectiveBudgetBytes` is what the predicate actually spent and the schema's
    // `predictedBytes == effectiveBudgetBytes` clause is a statement about this run.
    let mut exact = context.clone();
    exact.predicted_peak_bytes = predicted;
    exact.budget = ltx_exact_fit_budget(context.budget, predicted);
    let exact_effective_bytes = exact.budget.effective_bytes();
    if exact_effective_bytes != predicted {
        return Err(format!(
            "LTX-2.3 exact-fit budget resolved {exact_effective_bytes} effective bytes for a \
             {predicted} byte predicted peak; the probe must present the boundary host, not a \
             nameplate total"
        ));
    }
    if let MemorySafetyDecision::Reject { reason } = generator.memory_strategy_safety_check(&exact)
    {
        return Err(format!(
            "LTX-2.3 admission rejected an exact-fit calibrated budget: {reason}"
        ));
    }

    let lifecycle_input = LtxLifecycleInput {
        geometry,
        fps,
        seed,
        fault_phase: decode_plan.lifecycle_fault_phase(),
    };
    // `lifecycle` is `Some` only when a warm control was rendered: the SC-20318 campaign entries,
    // which exist to prove the request-scope lifecycle. The ORDINARY capture — the catalog anchor —
    // is one measured render on the video lane (sc-22738, `capture` above): no clean warm control,
    // no warm repeat, no fault injection, no recovery render. The 2026-09-07 corpus
    // (`docs/calibration/sc-22738/ltx-2-3-*-mlx-evidence.json`) was captured under the previous
    // six-render lifecycle and stays valid as it is.
    let (lifecycle, bounded_warm): (Option<LtxLifecycleMetrics>, Option<LtxBoundedWarmMetrics>) =
        match admission {
            LtxRunAdmission::CampaignEntry => (
                Some(verify_ltx_lifecycle(
                    generator.as_ref(),
                    &context,
                    &measured,
                    lifecycle_input,
                    phase_sink,
                )?),
                None,
            ),
            LtxRunAdmission::BoundedCampaignEntry => {
                let warm = verify_ltx_bounded_warm_repeat(
                    generator.as_ref(),
                    &context,
                    &measured,
                    lifecycle_input,
                )?;
                (
                    Some(LtxLifecycleMetrics {
                        maximum_error: warm.maximum_error,
                        mean_error: warm.mean_error,
                        rms_error: warm.rms_error,
                        ..Default::default()
                    }),
                    Some(warm),
                )
            }
            LtxRunAdmission::Ordinary => (None, None),
        };

    // Runtime-complete keeps the schema's unexecuted mutation slot null. The adapter still proves
    // falsifiability and records the breach in diagnostics.
    let mutated = measured
        .iter()
        .map(qwen_negative_mutation)
        .collect::<Vec<_>>();
    let (mutated_maximum, mutated_mean, mutated_rms) = video_max_mean_rms_abs(&mutated, &measured)?;
    if ltx_quality_passes(mutated_maximum, mutated_mean, mutated_rms) {
        return Err("LTX-2.3 output mutation did not breach the determinism envelope".to_owned());
    }

    let single_render_blocker = format!(
        "this capture executes ONE measured render inside the selected request scope and no \
         warm pass ({}); it injects no calibration fault, so the cancellation and \
         authorized-error scenarios and their recovery renders are unexecuted and this record \
         claims nothing about them",
        protocol::VIDEO_WARM_PASSES_NOT_RUN
    );
    let scenarios = if lifecycle.is_none() {
        json!([
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": exact_effective_bytes },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider contract rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider contract rejected a mutated calibration fingerprint" },
            capture.not_run_warm_repeat_scenario()?,
            { "name": "cancel", "result": "not_run", "reason": single_render_blocker },
            { "name": "error", "result": "not_run", "reason": single_render_blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ])
    } else if admission == LtxRunAdmission::BoundedCampaignEntry {
        let blocker = "SC-20318 executes only selected plus warm-repeat parity; cancellation, authorized-error, and recovery renders remain unexecuted";
        json!([
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": exact_effective_bytes },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider contract rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider contract rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "passed", "reason": "the selected request scope repeated deterministically within the declared clip-wide envelope" },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ])
    } else {
        json!([
            { "name": "exact_fit", "result": "passed", "predictedBytes": predicted, "effectiveBudgetBytes": exact_effective_bytes },
            { "name": "unknown_budget", "result": "passed", "reason": "the loaded provider contract rejected a zero/unknown budget" },
            { "name": "stale_evidence", "result": "passed", "reason": "the loaded provider contract rejected a mutated calibration fingerprint" },
            { "name": "warm_repeat", "result": "passed", "reason": "the selected request scope repeated deterministically within the declared clip-wide envelope" },
            { "name": "cancel", "result": "passed", "reason": "typed cancellation at the selected rung boundary returned the typed outcome and cleaned up within the clean-warm ALLOCATOR bounds, and a recovery render followed inside the same peak and retention bounds; what that recovery clip CONTAINS is reported in diagnostics, and any excursion from the determinism envelope is named in blockers (sc-22738)", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "error", "result": "passed", "reason": "provider fault injection at the selected rung boundary returned the typed outcome and cleaned up within the clean-warm ALLOCATOR bounds, and a recovery render followed inside the same peak and retention bounds; what that recovery clip CONTAINS is reported in diagnostics, and any excursion from the determinism envelope is named in blockers (sc-22738)", "cleanupVerified": true, "warmFollowUpPassed": true },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below from the declared reference-free target" }
        ])
    };
    let mut diagnostic_measurements = vec![
        ("preRungActiveAfterClear", "bytes", pre_rung_active),
        ("preRungCacheAfterClear", "bytes", pre_rung_cache),
        ("conditioningActivePeak", "bytes", conditioning.active),
        ("denoiseActivePeak", "bytes", denoise.active),
        ("decodeActivePeak", "bytes", decode.active),
        (
            "overallAllocatorEnvelope",
            "bytes",
            overall.allocator_bytes(),
        ),
        ("predictedOverallCeiling", "bytes", predicted_peaks.overall),
        ("denoiseEntryActive", "bytes", denoise_entry.active),
        ("decodeEntryActive", "bytes", decode_entry.active),
        ("stagedGemmaBytes", "bytes", gemma_bytes),
        ("stagedConnectorBytes", "bytes", connector_bytes),
        ("stagedTextEncoderBytes", "bytes", text_encoder_bytes),
        ("stagedTransformerBytes", "bytes", transformer_bytes),
        ("costagedGiantsBytes", "bytes", costaged_bytes),
        ("tierArtifactBytes", "bytes", tier_bytes),
        (
            "admissionWeightsFloorBytes",
            "bytes",
            projection.weights_bytes,
        ),
        (
            "admissionDecodeWorkingSetBytes",
            "bytes",
            projection.decode_working_set_bytes,
        ),
        (
            "admissionPredictedPeakBytes",
            "bytes",
            projection.predicted_peak_bytes,
        ),
        ("admissionHardwareBytes", "bytes", hardware_bytes),
        ("renderedFrames", "count", u64::from(geometry.frames)),
        (
            "latentTemporalDepth",
            "count",
            u64::from(geometry.latent_frames),
        ),
        (
            "latentTokens",
            "count",
            u64::from(geometry.latent_frames)
                * u64::from(geometry.width / 32)
                * u64::from(geometry.height / 32),
        ),
        ("outputFps", "count", u64::from(fps)),
        ("audioTrackDecoded", "count", 1),
        (
            "decodeTilingEngaged",
            "count",
            u64::from(decode_plan.tiling_engaged()),
        ),
        (
            "decodeWritableFrameCap",
            "count",
            decode_plan.writable_frame_cap.max(0) as u64,
        ),
        (
            "decodeTileSpatialPx",
            "count",
            decode_plan.spatial_tile_px(),
        ),
        (
            "decodeTileOverlapPx",
            "count",
            decode_plan.spatial_overlap_px(),
        ),
        ("spatialDecodeTiles", "count", spatial_decode_tiles),
        ("mlxMemoryLimitBytes", "bytes", get_memory_limit() as u64),
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
    ];
    if let Some(warm) = bounded_warm {
        diagnostic_measurements.extend([
            (
                "warmConditioningActivePeak",
                "bytes",
                warm.conditioning.active,
            ),
            ("warmDenoiseActivePeak", "bytes", warm.denoise.active),
            ("warmDecodeActivePeak", "bytes", warm.decode.active),
            ("warmOutputAudioSamples", "count", warm.audio.samples),
            (
                "warmOutputAudioSampleRate",
                "hertz",
                u64::from(warm.audio.sample_rate),
            ),
            (
                "warmOutputAudioChannels",
                "count",
                u64::from(warm.audio.channels),
            ),
            ("providerRequestScopeRenders", "count", 2),
        ]);
    } else if let Some(lifecycle) = &lifecycle {
        diagnostic_measurements.extend([
            ("warmRepeatPeak", "bytes", lifecycle.clean_warm_peak),
            (
                "warmRepeatPostCleanupActive",
                "bytes",
                lifecycle.clean_post_cleanup.active,
            ),
            (
                "warmRepeatPostCleanupCache",
                "bytes",
                lifecycle.clean_post_cleanup.cache,
            ),
            (
                "lifecycleMaxFaultPostCleanupActive",
                "bytes",
                lifecycle.max_fault_post_cleanup.active,
            ),
            (
                "lifecycleMaxFaultPostCleanupCache",
                "bytes",
                lifecycle.max_fault_post_cleanup.cache,
            ),
            (
                "lifecycleMaxRecoveryPeak",
                "bytes",
                lifecycle.max_recovery_peak,
            ),
            (
                "lifecycleMaxRecoveryPostCleanupActive",
                "bytes",
                lifecycle.max_recovery_post_cleanup.active,
            ),
            (
                "lifecycleMaxRecoveryPostCleanupCache",
                "bytes",
                lifecycle.max_recovery_post_cleanup.cache,
            ),
        ]);
        // The recovery-clip evidence, in thousandths of one 0-255 level so a sub-level delta
        // survives the integer channel. Both references travel: see `ltx_lifecycle_blockers`.
        diagnostic_measurements.extend(ltx_recovery_measurements(
            lifecycle.cancel_recovery,
            lifecycle.error_recovery,
            lifecycle.measured_clip,
        ));
    } else {
        // sc-22738: no `warmRepeat*` / `lifecycleMax*` figure — the warm passes were not run
        // (`quality.warmPasses`), and an unmeasured figure is omitted, never written as 0.
        diagnostic_measurements.push(("warmPasses", "count", u64::from(capture.warm_passes)));
    }
    const LTX_QUALITY_CONTRACT: &str = "identical artifact, prompt, seed, geometry, frames, fps, tier, provider contract and selected request scope; measured clip versus a clean warm repeat, compared over every frame";
    let quality = match &lifecycle {
        Some(lifecycle) => json!({
            "contract": LTX_QUALITY_CONTRACT,
            "identicalInputs": true,
            "result": "passed",
            "maximumError": lifecycle.maximum_error,
            "meanError": lifecycle.mean_error,
            "rootMeanSquareError": lifecycle.rms_error,
            "maximumErrorThreshold": LTX_MAX_THRESHOLD,
            "meanErrorThreshold": LTX_MEAN_THRESHOLD,
            "rootMeanSquareErrorThreshold": LTX_RMS_THRESHOLD,
        }),
        None => capture.not_run_quality(
            LTX_QUALITY_CONTRACT,
            (LTX_MAX_THRESHOLD, LTX_MEAN_THRESHOLD, LTX_RMS_THRESHOLD),
        )?,
    };
    let blockers = match &lifecycle {
        Some(lifecycle) => ltx_lifecycle_blockers(
            tier,
            lifecycle_input.fault_phase,
            lifecycle.cancel_recovery,
            lifecycle.error_recovery,
        ),
        None => vec![single_render_blocker.clone()],
    };

    let mut fragment = json!({
        "status": "runtime_complete",
        "strategy": strategy,
        "loadShape": load_shape_key(calibration.load_shape),
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": tier,
        },
        "sweep": ltx_complete_sweep(request)?,
        "scenarios": scenarios,
        "predictedPeakBytes": predicted_peaks.json(),
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
            "resolvedPathFingerprint": format!("{repository}@{revision}:{tier}+gemma"),
        },
        "diagnostics": protocol::diagnostics(
            "memory-mlx-adapter:ltx-2-3-provider-contract-video",
            "executed",
            blockers,
            diagnostic_measurements,
        ),
        "capturedAt": protocol::captured_at(),
    });
    // sc-22738. The `output` descriptor belongs to the CAMPAIGN CARRIERS and to nothing else.
    //
    // A calibration fragment becomes a `records[]` member verbatim (the harness spreads it —
    // `capturePlannedCase`'s `{ ...fragment }`), and `memory-calibration.schema.json`'s record is
    // `additionalProperties: false` with no `output` property: every committed video record —
    // `docs/generated/ltx-mlx-video-sc-18808.json` included — publishes the same four facts as
    // `renderedFrames` / `outputFps` / `audioTrackDecoded` / `latentTemporalDepth` diagnostic
    // measurements, which is where the harness and the derivations read them from. Emitting the
    // block on an ordinary capture therefore does not enrich the record, it makes the bundle
    // unschedulable — `schema validation failed: $.records[0].output: unexpected property`, AFTER
    // the render, which is the most expensive place in the system to fail.
    //
    // The campaign entries are not calibration records: `run_ltx_campaign_entry` and
    // `run_ltx_bounded_campaign_entry` stamp their own `_campaignEntry` / `_boundedCampaignEntry`
    // keys (which the record schema would reject just as flatly) and their fragments are consumed
    // by `validate_ltx_campaign_entry_fragment` / `validate_ltx_bounded_campaign_fragment`, which
    // cross-check `/output/*` AGAINST the diagnostics of the same run. That redundancy is the whole
    // point of the carrier check, so it is kept exactly where it is read.
    if !matches!(admission, LtxRunAdmission::Ordinary) {
        fragment["output"] = json!({
            "frames": geometry.frames,
            "fps": fps,
            "audio": {
                "present": true,
                "samples": audio.samples,
                "sampleRate": audio.sample_rate,
                "channels": audio.channels,
            },
            "firstFrameNondegenerate": true,
        });
    }
    protocol::settle_plain_overlay_scenario(request, &mut fragment, LTX_PLAIN_EXECUTION_PATH)?;
    Ok(fragment)
}

pub(super) fn validate_ltx_bounded_campaign_fragment(fragment: &Value) -> Result<(), String> {
    for (name, expected) in [
        ("renderedFrames", u64::from(LTX_CAMPAIGN_ENTRY_FRAMES)),
        ("outputFps", u64::from(LTX_CAMPAIGN_ENTRY_FPS)),
        ("audioTrackDecoded", 1),
        ("decodeTilingEngaged", 1),
        ("decodeTileSpatialPx", u64::from(LTX_CANARY_TILE_EDGE)),
        ("decodeTileOverlapPx", u64::from(LTX_CANARY_OVERLAP)),
        ("spatialDecodeTiles", 24),
        ("latentTemporalDepth", 16),
        ("latentTokens", 6_144),
        ("warmOutputAudioChannels", 2),
        ("providerRequestScopeRenders", 2),
    ] {
        let actual = campaign_entry_diagnostic(fragment, name)?;
        if actual != expected {
            return Err(format!(
                "SC-20318 bounded campaign diagnostic {name} must be {expected}, got {actual}"
            ));
        }
    }
    for name in [
        "warmConditioningActivePeak",
        "warmDenoiseActivePeak",
        "warmDecodeActivePeak",
        "warmOutputAudioSamples",
        "warmOutputAudioSampleRate",
    ] {
        if campaign_entry_diagnostic(fragment, name)? == 0 {
            return Err(format!(
                "SC-20318 bounded campaign diagnostic {name} must be positive"
            ));
        }
    }
    let scenarios = fragment
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| "SC-20318 bounded campaign omitted scenarios".to_owned())?;
    let result = |name: &str| {
        scenarios
            .iter()
            .find(|scenario| scenario.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|scenario| scenario.get("result"))
            .and_then(Value::as_str)
    };
    if fragment.pointer("/strategy/rung").and_then(Value::as_str) != Some("bounded_decode")
        || fragment.pointer("/strategy/engagedRungs")
            != Some(&json!(["resident", "staged_residency", "bounded_decode"]))
        || fragment.pointer("/strategy/parameters")
            != Some(&json!({
                "decodeTileEdge": LTX_CANARY_TILE_EDGE,
                "decodeOverlap": LTX_CANARY_OVERLAP,
            }))
        || result("warm_repeat") != Some("passed")
        || result("cancel") != Some("not_run")
        || result("error") != Some("not_run")
        || fragment.pointer("/output/frames").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FRAMES))
        || fragment.pointer("/output/fps").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FPS))
        || fragment
            .pointer("/output/audio/present")
            .and_then(Value::as_bool)
            != Some(true)
        || fragment
            .pointer("/output/audio/samples")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/sampleRate")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/channels")
            .and_then(Value::as_u64)
            != Some(2)
        || fragment
            .pointer("/output/firstFrameNondegenerate")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "SC-20318 bounded campaign response changed its exact canonical carrier".to_owned(),
        );
    }
    Ok(())
}

pub(super) fn validate_ltx_campaign_entry_fragment(fragment: &Value) -> Result<(), String> {
    for (name, expected) in [
        ("renderedFrames", u64::from(LTX_CAMPAIGN_ENTRY_FRAMES)),
        ("outputFps", u64::from(LTX_CAMPAIGN_ENTRY_FPS)),
        ("audioTrackDecoded", 1),
        ("decodeTilingEngaged", 0),
        ("decodeTileSpatialPx", 0),
        ("decodeTileOverlapPx", 0),
        ("latentTemporalDepth", 16),
        ("latentTokens", 6_144),
    ] {
        let actual = campaign_entry_diagnostic(fragment, name)?;
        if actual != expected {
            return Err(format!(
                "SC-20191 campaign entry diagnostic {name} must be {expected}, got {actual}"
            ));
        }
    }
    if fragment.pointer("/strategy/rung").and_then(Value::as_str) != Some("staged_residency")
        || fragment.pointer("/strategy/engagedRungs")
            != Some(&json!(["resident", "staged_residency"]))
        || fragment.pointer("/strategy/parameters") != Some(&json!({}))
        || fragment.pointer("/output/frames").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FRAMES))
        || fragment.pointer("/output/fps").and_then(Value::as_u64)
            != Some(u64::from(LTX_CAMPAIGN_ENTRY_FPS))
        || fragment
            .pointer("/output/audio/present")
            .and_then(Value::as_bool)
            != Some(true)
        || fragment
            .pointer("/output/audio/samples")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/sampleRate")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/audio/channels")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        || fragment
            .pointer("/output/firstFrameNondegenerate")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "SC-20191 campaign entry response changed the exact untiled full-A/V carrier"
                .to_owned(),
        );
    }
    Ok(())
}

pub(super) fn prevalidate_ltx_campaign_entry(request: &Value) -> Result<(), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20191 campaign entry requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    let expected_fingerprint = ltx_calibration_fingerprint(tier)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "SC-20191 campaign entry calibration fingerprint changed: plan={planned_fingerprint}, \
             tier {tier} identity={expected_fingerprint}"
        ));
    }
    validate_ltx_campaign_entry(request, tier, geometry, &selection)
}

pub(super) fn prevalidate_ltx_bounded_campaign_entry(request: &Value) -> Result<(), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20318 bounded campaign requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    let expected_fingerprint = ltx_calibration_fingerprint(tier)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "SC-20318 bounded campaign calibration fingerprint changed: plan={planned_fingerprint}, \
             tier {tier} identity={expected_fingerprint}"
        ));
    }
    validate_ltx_bounded_campaign_entry(request, tier, geometry, &selection)
}

pub(super) fn prevalidate_ltx_bounded_carrier_proof(
    request: &Value,
) -> Result<(LtxGeometry, MemorySelection), String> {
    let geometry = validate_ltx_target(request)?;
    protocol::validate_plain_overlay_target(request, LTX_PLAIN_EXECUTION_PATH)?;
    if planned_load_shape(request)? != LoadShape::EagerMaterialization {
        return Err("SC-20254 bounded carrier requires eager materialization".to_owned());
    }
    let selection = planned_selection(request)?;
    let tier = planned_qwen_tier(request)?;
    let planned_fingerprint = protocol::planned_calibration_fingerprint(request)?;
    let expected_fingerprint = ltx_calibration_fingerprint(tier)?;
    if planned_fingerprint != expected_fingerprint {
        return Err(format!(
            "SC-20254 bounded-carrier calibration fingerprint changed: plan={planned_fingerprint}, \
             tier {tier} identity={expected_fingerprint}"
        ));
    }
    validate_ltx_bounded_carrier_proof(request, tier, geometry, &selection)?;
    let host_memory = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "SC-20254 hardware.memoryBytes must be an integer".to_owned())?;
    if host_memory < LTX_CANARY_MAX_FOOTPRINT_BYTES * 2 {
        return Err("SC-20254 host cannot preserve two exact stop boundaries".to_owned());
    }
    Ok((geometry, selection))
}

pub(super) fn run_ltx_bounded_carrier_proof(request: &Value) -> Result<Value, String> {
    // The exact private identity is fully checked before the live watchdog handshake, limits,
    // artifact paths, registry construction, or model allocation.
    let (geometry, selection) = prevalidate_ltx_bounded_carrier_proof(request)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let (repository, revision, root, text_encoder_root, spec) =
        ltx_load_spec(request, "q4", &selection)?;
    if revision != LTX_CANARY_ARTIFACT_REVISION {
        return Err(format!(
            "SC-20254 artifact revision must be {LTX_CANARY_ARTIFACT_REVISION}, got {revision}"
        ));
    }
    let registry =
        mlx_gen_ltx::provider_registry().map_err(|error| format!("build LTX registry: {error}"))?;
    let contract = registry
        .memory_strategy_contract(LTX_PROVIDER, &spec)
        .map_err(|error| format!("read {LTX_PROVIDER} memory-strategy contract: {error}"))?
        .ok_or_else(|| "pinned MLX LTX-2.3 provider has no memory-strategy contract".to_owned())?;
    contract
        .validate_selection(&selection)
        .map_err(|error| format!("pinned LTX-2.3 contract rejected bounded carrier: {error}"))?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "pinned LTX-2.3 contract has no calibration identity".to_owned())?;
    // The carrier proof loads the frozen q4 artifact above, so the loaded contract must publish the
    // q4 member of the identity family.
    let expected_fingerprint = ltx_calibration_fingerprint("q4")?;
    if calibration.fingerprint != expected_fingerprint {
        return Err(format!(
            "SC-20254 pinned provider calibration identity changed: expected {expected_fingerprint}, got {}",
            calibration.fingerprint
        ));
    }
    let decode_plan = LtxDecodePlan::resolve_for_selection(&selection, geometry)?;
    decode_plan.validate_selected_strategy(&selection)?;
    let spatial_decode_tile_count = decode_plan.spatial_tile_count(geometry)?;
    if !decode_plan.tiling_engaged()
        || decode_plan.spatial_tile_px() != u64::from(LTX_CANARY_TILE_EDGE)
        || decode_plan.spatial_overlap_px() != u64::from(LTX_CANARY_OVERLAP)
        || spatial_decode_tile_count != 24
    {
        return Err("SC-20254 did not resolve the exact 24-tile 192/64 carrier".to_owned());
    }
    let generator = registry
        .load(LTX_PROVIDER, &spec)
        .map_err(|error| format!("load real LTX-2.3 q4 bounded carrier: {error}"))?;
    if generator.memory_strategy_contract() != Some(&contract) {
        return Err("SC-20254 loaded generator contract differs from registry".to_owned());
    }
    let context = ltx_context(
        selection,
        calibration,
        &calibration.fingerprint,
        geometry,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
        LTX_CANARY_MAX_FOOTPRINT_BYTES,
    );
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err("LTX-2.3 provider rejected the exact SC-20254 budget".to_owned());
    }
    clear_cache();
    reset_peak_memory();
    let mut conditioning = PhaseMemory {
        active: 0,
        cache: 0,
    };
    let mut denoise = PhaseMemory {
        active: 0,
        cache: 0,
    };
    watchdog_lease.mark("primary_conditioning")?;
    let generation_request = ltx_request(geometry, LTX_CAMPAIGN_ENTRY_FPS, LTX_SEED);
    validate_ltx_bounded_carrier_generation_request(&generation_request)?;
    let generated = scoped_generate(
        generator.as_ref(),
        generation_request,
        &context,
        None,
        &mut |progress| match progress {
            Progress::Step { current: 1, .. } => {
                if let Err(error) = watchdog_lease.mark("primary_denoise") {
                    eprintln!("{error}");
                    std::process::abort();
                }
                conditioning = PhaseMemory::capture();
                reset_peak_memory();
            }
            Progress::Decoding => {
                if let Err(error) = watchdog_lease.mark("primary_decode") {
                    eprintln!("{error}");
                    std::process::abort();
                }
                denoise = PhaseMemory::capture();
                reset_peak_memory();
            }
            _ => {}
        },
    )?;
    let (frames, fps, audio) = diagnostic_video_frames(generated, LTX_VIDEO_LABEL)?;
    let decode = PhaseMemory::capture();
    if frames.len() != LTX_CAMPAIGN_ENTRY_FRAMES as usize || fps != LTX_CAMPAIGN_ENTRY_FPS {
        return Err(format!(
            "SC-20254 returned frames={}, fps={fps}",
            frames.len()
        ));
    }
    let audio = audio
        .filter(|value| value.samples > 0 && value.sample_rate > 0 && value.channels > 0)
        .ok_or_else(|| "SC-20254 full-A/V render returned no non-empty audio track".to_owned())?;
    let first = frames
        .first()
        .ok_or_else(|| "SC-20254 bounded carrier returned no frames".to_owned())?;
    if first.pixels.is_empty() || first.pixels.iter().all(|pixel| *pixel == first.pixels[0]) {
        return Err("SC-20254 bounded carrier returned a degenerate first frame".to_owned());
    }
    if [conditioning.active, denoise.active, decode.active].contains(&0) {
        return Err("SC-20254 synchronized phase reported a zero active peak".to_owned());
    }
    let peak_active = [conditioning.active, denoise.active, decode.active]
        .into_iter()
        .max()
        .unwrap_or(0);
    drop(frames);
    drop(generator);
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    let planned_artifact = protocol::planned(request)?
        .pointer("/_boundedCarrier/artifact")
        .cloned()
        .ok_or_else(|| "SC-20254 lost its exact artifact identity".to_owned())?;
    let mut strategy = ltx_attested_strategy(request, &context.selection, &contract)?;
    strategy["spatialDecodeTiles"] = json!(spatial_decode_tile_count);
    watchdog_lease.complete()?;
    Ok(json!({
        "schemaVersion": 1,
        "recordType": "sceneworks_bounded_carrier_proof_response_v1",
        "status": "diagnostic_bounded_carrier_complete",
        "story": "sc-20254",
        "logicalCaseId": LTX_BOUNDED_CARRIER_LOGICAL_CASE_ID,
        "fixture": LTX_BOUNDED_CARRIER_FIXTURE,
        "identity": LTX_BOUNDED_CARRIER_IDENTITY,
        "diagnosticOnly": true,
        "promotable": false,
        "ingestible": false,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "calibrationFingerprint": calibration.fingerprint,
        "artifact": {
            "repository": repository,
            "resolvedRevision": revision,
            "variant": "q4",
            "tierRoot": root,
            "textEncoderRoot": text_encoder_root,
            "numericTierInventory": planned_artifact["numericTierInventory"],
            "textEncoderInventory": planned_artifact["textEncoderInventory"],
        },
        "target": {
            "provider": LTX_PROVIDER,
            "tier": "q4",
            "geometry": {
                "width": LTX_CAMPAIGN_ENTRY_WIDTH,
                "height": LTX_CAMPAIGN_ENTRY_HEIGHT,
                "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
                "fps": LTX_CAMPAIGN_ENTRY_FPS,
            },
            "seed": LTX_SEED,
            "videoMode": "default_av",
            "audio": true,
        },
        "strategy": strategy,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CARRIER_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "observedMemory": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "conditioning": conditioning.json(),
            "denoise": denoise.json(),
            "decode": decode.json(),
            "peakActiveBytes": peak_active,
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
        "output": {
            "frames": LTX_CAMPAIGN_ENTRY_FRAMES,
            "fps": LTX_CAMPAIGN_ENTRY_FPS,
            "audio": {
                "present": true,
                "samples": audio.samples,
                "sampleRate": audio.sample_rate,
                "channels": audio.channels,
            },
            "frameTimelineSeconds": f64::from(LTX_CAMPAIGN_ENTRY_FRAMES - 1)
                / f64::from(LTX_CAMPAIGN_ENTRY_FPS),
            "firstFrameNondegenerate": true,
        },
        "capturedAt": protocol::captured_at(),
    }))
}

pub(super) fn run_ltx_campaign_entry(request: &Value) -> Result<Value, String> {
    // Every request identity check precedes the live watchdog handshake, process-global limits,
    // model-path resolution and provider registry construction.
    prevalidate_ltx_campaign_entry(request)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease()?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let mut fragment =
        run_ltx_with_admission(request, LtxRunAdmission::CampaignEntry, &mut watchdog_lease)?;
    validate_ltx_campaign_entry_fragment(&fragment)?;
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    fragment["_campaignEntry"] = json!({
        "identity": LTX_CAMPAIGN_ENTRY_IDENTITY,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "cleanup": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
    });
    watchdog_lease.complete()?;
    Ok(fragment)
}

pub(super) fn run_ltx_bounded_campaign_entry(request: &Value) -> Result<Value, String> {
    prevalidate_ltx_bounded_campaign_entry(request)?;
    let tier = protocol::planned_optional_str(request, "/target/tier")?
        .ok_or_else(|| "bounded campaign target tier must be a string".to_owned())?;
    let bounded_spec = ltx_bounded_campaign_spec(tier)?;
    let mut watchdog = consume_ltx_canary_watchdog_attestation(request)?;
    let mut watchdog_lease = watchdog.start_lease_for(&LTX_BOUNDED_CARRIER_PHASE_NAMES)?;
    watchdog_lease.mark("common_load")?;
    let limits = LtxCanaryLimits::install()?;
    clear_cache();
    let pre_provider = AllocatorState::capture_current();
    validate_ltx_canary_pre_provider(pre_provider)?;
    let expected_persistent_active = ltx_canary_ones_cache_bytes()?;
    let mut fragment = run_ltx_with_admission(
        request,
        LtxRunAdmission::BoundedCampaignEntry,
        &mut watchdog_lease,
    )?;
    validate_ltx_bounded_campaign_fragment(&fragment)?;
    watchdog_lease.mark("cleanup")?;
    clear_cache();
    let post_cleanup = AllocatorState::capture_current();
    validate_ltx_canary_cleanup(pre_provider, post_cleanup, expected_persistent_active)?;
    fragment["_boundedCampaignEntry"] = json!({
        "identity": bounded_spec.identity,
        "inferenceRevision": protocol::INFERENCE_PIN,
        "watchdog": {
            "required": true,
            "protocol": LTX_CANARY_WATCHDOG_PROTOCOL,
            "providerPhaseProtocol": LTX_PROVIDER_PHASE_PROTOCOL,
            "providerPhaseProfile": LTX_BOUNDED_CAMPAIGN_PHASE_PROFILE,
            "providerPhases": LTX_BOUNDED_CARRIER_PHASE_NAMES,
            "maxFootprintBytes": watchdog.max_footprint_bytes,
            "maxRuntimeSeconds": watchdog.max_runtime_seconds,
            "hostMemoryBytes": watchdog.host_memory_bytes,
            "minInitialMemoryFreeBytes": watchdog.min_initial_memory_free_bytes,
            "minMemoryFreeBytes": watchdog.min_memory_free_bytes,
        },
        "mlxLimits": {
            "memoryLimitBytes": LTX_CANARY_MAX_FOOTPRINT_BYTES,
            "wiredLimitBytes": limits.wired,
        },
        "cleanup": {
            "preProviderActiveBytes": pre_provider.active,
            "preProviderCacheBytes": pre_provider.cache,
            "expectedPersistentActive": {
                "identity": LTX_CANARY_ONES_CACHE_IDENTITY,
                "videoDimension": LTX_CANARY_ONES_CACHE_VIDEO_DIMENSION,
                "audioDimension": LTX_CANARY_ONES_CACHE_AUDIO_DIMENSION,
                "dtype": "bfloat16",
                "bytesPerElement": BFLOAT16_BYTES_PER_ELEMENT,
                "bytes": expected_persistent_active,
            },
            "postCleanupActiveBytes": post_cleanup.active,
            "postCleanupCacheBytes": post_cleanup.cache,
        },
    });
    watchdog_lease.complete()?;
    Ok(fragment)
}

pub(super) fn run_ltx(request: &Value) -> Result<Value, String> {
    let mut phases = NoLtxCampaignPhases;
    run_ltx_with_admission(request, LtxRunAdmission::Ordinary, &mut phases)
}
