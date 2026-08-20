//! **The two-basis estimate synthesis mechanism, backend-generic** (sc-19050, epic 19048 R1).
//!
//! Epic 19048 R1: *"One prediction mechanism, two bases. Measured-evidence extrapolation (exact
//! calibration identity, geometry-scaled) + declared floor. Geometry axes (area, frames, batch,
//! reference_count) defined once at the mechanism level; a new axis lands in one place and both
//! lanes get it. No prediction law lives in a backend-local module; backends supply budget probe,
//! margin, and failure posture as parameters."*
//!
//! This module is that home. Everything here was extracted verbatim from
//! `crates/sceneworks-worker/src/mlx_fit_gate.rs` (sc-18096's `collect_estimate_bases` filter,
//! `synthesize_estimate_ladder`, `estimate_floor_weights_bytes`, `estimate_floor_parameters`,
//! `estimate_evidence`, `binding_phase`) — pure code motion, zero admission changes, proven by
//! [`crate::mlx_admission_decisions`] whose committed artifact was emitted by the PRE-extraction
//! gate and is re-verified against this module's output.
//!
//! sc-19058 folded the second and last home of prediction law in the tree onto this module:
//! `krea_2_turbo`'s declared phase curves, which until then were evaluated inside `vram_gate.rs`'s
//! candle cfg straight off manifest JSON. Same discipline, same bar — code motion only.
//!
//! **Its evidence is NOT the sc-19050 shape above, and the difference matters.** There is no
//! committed decision artifact standing behind the krea fold, because
//! `docs/generated/candle-admission-decisions.json` is structurally BLIND to this route: all four
//! `krea_2_turbo` rows in it are `resolution: "not_evaluated"`, carrying no geometry and no budget
//! axis, for the reason `candle_admission_decisions.rs` states in its own header — the ladder takes
//! a `KreaRuntimeEvidenceContext` whose only non-test constructor walks a resolved artifact tree, so
//! no CPU lane can drive it and stamping it with another gate's answer would fabricate a column.
//! What actually holds the fold is the `vram_gate` unit suite, which grades the SHIPPED manifest
//! through the shipped reader — `committed_image_curves_evaluate_bit_identically_to_the_two_coefficient_form`
//! (36 curves, bitwise, over the pre-sc-18812 expression *with its original association*), the three
//! sc-19056 lane guards, and the historical q4/q8/bf16 rung-selection tests — plus the verbatim code
//! motion itself. Claiming the decisions artifact here would be citing a file that cannot disagree.
//!
//! # What is a parameter, and why
//!
//! | backend-supplied | how it arrives | why it cannot be a constant here |
//! |---|---|---|
//! | backend identity | [`EstimateLane::backend`] | the evidence key is lane-scoped |
//! | headroom law | [`EstimateRequest::headroom_bytes`] | MLX charges a fixed OS/app reserve plus an area-scaled activation transient (`MlxRequestPlan::generic_headroom_bytes`); candle charges `vram_gate::HEADROOM_GB` |
//! | margin | [`EstimateLane::estimate_margin`] | derived per lane in [`crate::ladder_margin_policy`] from that lane's own repeat-capture corpus |
//! | failure posture | [`EstimateLane::failure_posture`] | an MLX allocator overshoot aborts the process; a CUDA OOM is a recoverable `Err`. This is *why* the margins differ, and recording it beside them stops a future edit from equalizing them by eye |
//!
//! The **budget probe** is the fourth backend-supplied input R1 names, and it is deliberately absent
//! from this module's signatures: synthesis produces candidates, it never grades them against a
//! budget. Grading is [`crate::memory_strategy::select_strategy`], which already takes the probed
//! `MemoryBudget` as a parameter — MLX from `mlx_fit_gate::live_request_budget`, candle from
//! `vram_gate`'s `VramBudget`. Adding an unread probe field here would be a speculative parameter,
//! not a satisfied requirement.
//!
//! # The two bases, in preference order
//!
//! 1. **Fitted basis** — a verified measured cell of the same provider/tier/mode/overlay at a
//!    different geometry ([`MeasuredRungBasis`]), extrapolated over the scaling regressor
//!    ([`voxels`]). Gated by the binding-phase constraint
//!    ([`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]).
//! 2. **Weights + headroom floor** — [`floor_weights_bytes`] plus the lane's own headroom law.
//!
//! The estimate margin is NOT applied here — the selector owns margin widening
//! (`memory_strategy::select_strategy`), exactly as it owns the sc-18095 stale widening.

use gen_core::{
    MemoryConformanceState, MemoryDecodePolicyQuery, MemoryDecodeQualityDisposition,
    MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey, MemoryEvidenceVerdict,
    MemoryGeometry, MemoryNumericTier, MemoryParityContract, MemoryParityResult,
    MemoryProviderContract, MemorySelection, MemoryStrategy,
};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::memory_calibration::{
    Geometry as CalibrationGeometry, MeasurementLane, StrategyRung,
};
use serde_json::Value;

use crate::memory_strategy::{memory_mode_from_mode_key, CandidateBasis};
use crate::payload::json_f64;

/// The request-scope evidence revision stamped on every synthesized candidate. Lives here rather
/// than in `mlx_fit_gate` because the video lane already stamps the same value through the same
/// constructor; it identifies the REQUEST-SCOPE evidence shape, not a backend.
pub(crate) const REQUEST_EVIDENCE_REVISION: &str = "sc-15507-request-scope-v1";

/// The inference contract revision stamped alongside it. Same reasoning.
pub(crate) const INFERENCE_CONTRACT_REVISION: &str = "1c4354b4b22d7f2cf5c4ea5fe17a83ab6c655e82";

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Geometry axes — defined ONCE (epic 19048 R1)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The geometry axes the mechanism knows about, in the order R1 names them.
///
/// A PIN, not a declaration: the axes are really defined by [`voxels`] (the regressor),
/// [`basis_geometry_is_scalable`] (which axes an extrapolation may span) and the evidence key. This
/// list exists so the set is asserted rather than narrated, and so a fifth axis is a visible edit
/// here as well as in those two places.
#[cfg(test)]
const GEOMETRY_AXES: [&str; 4] = ["area", "frames", "batch", "reference_count"];

/// **The scaling regressor: voxels = area x frames.**
///
/// sc-18812 established the discipline (extrapolate within a measured hull, over the regressor the
/// workload actually scales in); sc-18829 landed `frames` as a first-class axis on the MLX basis
/// filter. Defining the regressor as voxels rather than area is what makes the frames axis land in
/// ONE place for both lanes.
///
/// This is behaviour-preserving for every basis this mechanism will accept today:
/// [`basis_geometry_is_scalable`] requires the basis and the request to agree on `frames`, so
/// `request_voxels / basis_voxels` reduces exactly to the area ratio the sc-18096 law used. The
/// axis becomes live the moment a lane relaxes that conjunct with a fitted temporal term behind it
/// — which is a decision change, and therefore a later slice's, not this extraction's.
pub(crate) const fn voxels(width: u32, height: u32, frames: u32) -> u128 {
    (width as u128) * (height as u128) * (frames as u128)
}

/// [`voxels`] of a request geometry.
pub(crate) const fn request_voxels(geometry: MemoryGeometry) -> u128 {
    voxels(geometry.width, geometry.height, geometry.frames)
}

/// [`voxels`] of a measured calibration cell.
pub(crate) const fn basis_voxels(geometry: CalibrationGeometry) -> u128 {
    voxels(geometry.width, geometry.height, geometry.frames)
}

/// Whether a measured cell may seed an extrapolation for this request.
///
/// The cell must differ from the request (an identical cell is exact evidence, not a basis) and
/// must differ ONLY in axes the regressor spans. `batch` and `frames` are held equal: a different
/// batch or frame count is a different workload SHAPE, not a scalable geometry, and no per-axis
/// term has been fitted for either. Extracted as-is from `mlx_fit_gate::collect_estimate_bases`
/// (the `frames` conjunct is sc-18829's, taken over unchanged).
///
/// `reference_count` is deliberately NOT compared: it does not exist on
/// [`CalibrationGeometry`] — the calibration cell identity is `(width, height, batch, frames)` —
/// so a basis cannot disagree about it. The axis is still named in [`GEOMETRY_AXES`] because it
/// exists on the REQUEST side ([`MemoryGeometry::reference_count`]) and rides the evidence key;
/// the day it gains a calibration axis, this predicate is where it lands.
pub(crate) fn basis_geometry_is_scalable(
    basis: CalibrationGeometry,
    request: CalibrationGeometry,
) -> bool {
    basis != request && basis.batch == request.batch && basis.frames == request.frames
}

/// The extrapolation scale from a measured cell to the request, floored at 1.0.
///
/// The floor is the conservative reading of the corpus, not a leftover: below 1024² the measured
/// transient stops falling off proportionally (illustrious 0.305x and qwen 0.512x of their anchors
/// at 0.25x area, both ABOVE the 0.25x a proportional term predicts), so a smaller-than-measured
/// request never predicts below the measurement.
pub(crate) fn extrapolation_scale(basis: CalibrationGeometry, request: MemoryGeometry) -> f64 {
    let measured = basis_voxels(basis) as f64;
    if measured <= 0.0 {
        return 1.0;
    }
    (request_voxels(request) as f64 / measured).max(1.0)
}

/// Whether a measured cell is CLOSE enough to seed an extrapolation for this request (sc-19054).
///
/// The linear voxel law is trusted only as far as the calibration corpus has witnessed it —
/// [`crate::ladder_margin_policy::MAX_EXTRAPOLATION_VOXEL_SCALE`] carries the bound and its
/// derivation. A degenerate zero-voxel basis is refused here outright: its clamp-at-1.0 scale
/// would silently present a meaningless measurement as the request's peak.
///
/// Applied in [`synthesize_estimate_ladder`]'s basis FILTER rather than in the fitted arm, so a
/// nearer basis can still serve when several exist; only a request beyond EVERY basis's bound
/// loses its fitted candidates and falls through to the floor arm, exactly like the
/// binding-phase-flip refusal.
pub(crate) fn basis_within_extrapolation_bound(
    basis: CalibrationGeometry,
    request: MemoryGeometry,
) -> bool {
    let measured = basis_voxels(basis) as f64;
    measured > 0.0
        && request_voxels(request) as f64
            <= measured * crate::ladder_margin_policy::MAX_EXTRAPOLATION_VOXEL_SCALE
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Fitted per-phase curves — the TOP of the evidence hierarchy (epic 19048 R1, sc-19058)
// ─────────────────────────────────────────────────────────────────────────────────────────────
//
// Everything in this section came out of `vram_gate.rs` VERBATIM — arithmetic, association,
// fail-closed set and documentation — as sc-19058's fold. `krea_2_turbo`'s `turboFit` block was the
// last declared-curve reader still evaluating a fitted model inside a backend-local module, which is
// what R1 forbids: *"No prediction law lives in a backend-local module; backends supply budget
// probe, margin, and failure posture as parameters."* The lane is the fourth such parameter
// ([`container_measurement_lane`]'s `reader_lane`), and it is a parameter for the same reason the
// others are — `vram_gate::READER_MEASUREMENT_LANE` is a fact about that module's cfg, not about
// this mechanism.
//
// The bar for this fold was BYTE-IDENTITY, not the decision-diff every other epic-19048 slice
// works to (R6 names sc-19058 as its single exception). So nothing here is tidied on the way
// through: the association in [`fitted_phase_curve_gb`] is the one
// `committed_image_curves_evaluate_bit_identically_to_the_two_coefficient_form` pins bit-for-bit,
// and the `?`-order of the reads is the one the guards' refusals depend on.
//
// Note what is deliberately NOT here: `KreaTurboPhasePeaks::peak_gb`'s `f64::max` chain. `f64::max`
// returns the non-NaN operand, while [`binding_phase`] refuses to let a NaN claim the peak at all —
// so routing the magnitude through the shared argmax would change what a NaN triple predicts. The
// two rules answer different questions and only one of them is shared.

/// The geometry a phase curve is evaluated at (sc-18812).
///
/// `frames` is the temporal axis the image lane never had. It is a separate type rather than a
/// second `u32` argument because the two axes are not interchangeable and a transposed call site
/// would otherwise compile: `pixels` is an AREA (already multiplied out) while `frames` is a count.
///
/// This is [`voxels`]'s regressor in the COEFFICIENTS' vocabulary rather than a second geometry
/// idea: `pixels * frames` is exactly what [`voxels`] computes, and [`voxels_within_measured_hull`]
/// bounds the extrapolation of the term that multiplies it.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CurveGeometry {
    pub pixels: u64,
    pub frames: u32,
}

/// **The lane guard for a coefficient container** (epic 19048 R4, sc-19056).
///
/// `Some(lane)` only when the container declares a lane tag AND that lane is the reader's. Missing,
/// malformed, and foreign all return `None`, which fails the whole fit closed: the caller abandons
/// the optimized ladder and keeps its pre-existing floor, rather than pricing a render off numbers
/// measured on other hardware.
///
/// Missing is refused rather than defaulted for the reason the epic's decision names — measurements
/// never transfer across lanes, and the backend KEY a fit block hangs under is authored, not
/// measured. An operator `user.models.jsonc` entry replaces the whole `candle` object wholesale
/// (`apps/rust-api/src/manifest.rs::merge_entries_by_id`), so "it is under `candle`, therefore it
/// was measured on candle" is precisely the inference this guard exists to stop. A legacy or
/// third-party block that predates the tag degrades to the legacy scalar gate, which is the
/// fail-closed direction; it never degrades to using the untagged numbers anyway.
///
/// This deliberately does NOT subsume the calibration handshake beside it. `calibrationAbi` /
/// `calibrationFingerprint` / `inferenceClosureDigest` answer "is this measurement still current for
/// this build"; a fingerprint is a free-form provider string two lanes can collide on, and the ABI
/// is one repo-wide number. Neither can express "wrong GPU".
///
/// `reader_lane` is a parameter rather than the candle constant it was before sc-19058: the guard is
/// mechanism law (R4 says *every* container names its lane and *every* reader fails closed on a
/// foreign one), while WHICH lane is reading is a property of the calling module.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn container_measurement_lane(
    container: &Value,
    reader_lane: MeasurementLane,
) -> Option<MeasurementLane> {
    let declared = MeasurementLane::from_key(container.get("measurementLane")?.as_str()?)?;
    (declared == reader_lane).then_some(declared)
}

/// Whether a request AREA is inside the area hull a fit block was measured across.
///
/// `None` when the block declares no readable `maxMeasuredPixels` at all — an unbounded curve is not
/// a curve this mechanism will extrapolate, so absence fails closed rather than admitting
/// everything.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn area_within_measured_hull(container: &Value, pixels: u64) -> Option<bool> {
    Some(pixels <= container.get("maxMeasuredPixels")?.as_u64()?)
}

/// The largest output VOXEL count (pixels x frames) a fit block's curves were measured across
/// (sc-18812). Absent is read as `pixels`, i.e. one output frame — which is what every image-lane
/// fit is, so omitting the key refuses exactly the multi-frame requests that lane never measured.
///
/// ## Why voxels and not a frame count
///
/// 1. Voxels are the regressor `perMpxFrameGb` multiplies, so this bounds the extrapolation of
///    the term it governs rather than a loosely correlated proxy.
/// 2. **The tiling discontinuity is itself a constant-voxel surface.** At the pinned revision
///    `VaeTiling::writable_frame_cap(out_h, out_w)` is `MAX_WRITABLE_ELEMS / (full_res_channels *
///    out_h * out_w)` with `MAX_WRITABLE_ELEMS = i32::MAX`, so a single pass is legal exactly
///    while `out_voxels <= i32::MAX / full_res_channels` — 268,435,455 for LTX's 8 full-res
///    channels. That one surface is the 297-output-frame cap quoted at 0.90 MP and 682 at
///    0.39 MP. A scalar frame bound would admit a small-area request and refuse an
///    identically-priced large-area one, which is the wrong shape of guard.
///
/// The bound is not politeness about unvalidated territory — past it the affine form is KNOWN
/// wrong. Single-pass decode climbs to ~94.3 GB at the cap and tiled decode drops it to ~63.8 GB
/// on this 128 GiB host; no affine curve represents that step, so the fit is refused across it
/// rather than extrapolated through it. Note the cap is only ONE-SIDED machine-independent: no
/// host exceeds it single-pass, but a smaller host tiles EARLIER via the memory bound, and tiled
/// cost RISES with host memory because the selector keeps the largest tile that fits.
fn max_measured_voxels(fit: &Value, pixels: u64) -> Option<u64> {
    match fit.get("maxMeasuredVoxels") {
        None => Some(pixels),
        Some(value) => value.as_u64().filter(|max| *max >= 1),
    }
}

/// Whether a request geometry is inside the VOXEL hull a fit block was measured across (sc-18812).
///
/// `None` when the geometry overflows or the declared bound is unreadable; `Some(false)` when the
/// request is past the bound. Both fail the fit closed at every caller — they are kept apart because
/// "this block is not readable evidence" and "this block is readable and says no" are different
/// facts, and one caller ([`crate::vram_gate::krea_turbo_fit_with_runtime`]) reports them with
/// different verdicts.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn voxels_within_measured_hull(fit: &Value, geometry: CurveGeometry) -> Option<bool> {
    let voxels = geometry.pixels.checked_mul(u64::from(geometry.frames))?;
    Some(voxels <= max_measured_voxels(fit, geometry.pixels)?)
}

/// **The fitted per-phase curve: the top of the evidence hierarchy** (sc-18810, sc-18812).
///
/// Read a measured phase curve `fixedGb + perMpxGb * megapixels + perMpxFrameGb * megapixels *
/// frames`. The manifest stores fixed weight / allocator residency separately from the
/// geometry-dependent activation slopes. Invalid or incomplete evidence fails closed to `None`;
/// callers retain their established floor instead of inventing a fit.
///
/// ## The temporal term (sc-18812, form chosen by sc-18810)
///
/// `perMpxFrameGb` is OPTIONAL and absent on every committed image curve. Absent is read as
/// `0.0`, and the sum is deliberately written so that the absent case reduces to the pre-sc-18812
/// expression **bit for bit** — the area term keeps its original association
/// (`per_mpx * pixels as f64 / 1_000_000.0`, not `per_mpx * (pixels as f64 / 1_000_000.0)`, which
/// is a DIFFERENT f64 in general), and `x + 0.0` is bitwise `x` for every finite `x` EXCEPT
/// `x == -0.0`, where IEEE-754 round-to-nearest gives `+0.0` — a different bit pattern for the
/// same numeric value. Reaching that here takes a curve declaring BOTH `fixedGb` and `perMpxGb`
/// as `-0.0` (either one alone leaves the pre-term sum at `+0.0`); such a curve validates, since
/// JSON Schema `minimum: 0` accepts `-0.0`, and clears the `< 0.0` guards below. No committed
/// curve declares one. It is documented rather than guarded: a signed-zero branch would exist
/// only to preserve the sign bit of a zero-valued phase prediction, which no consumer can
/// observe. The identity itself is pinned bitwise over the real committed curves, not asserted
/// against a default — and it is the same association sc-19058's fold out of `vram_gate.rs` had to
/// carry across unedited.
///
/// ## The lane tag (sc-18812's sibling hazard, closed by sc-19056 / epic 19048 R4)
///
/// The three coefficients are a backend-neutral FORM carrying lane-specific NUMBERS. That is
/// not a hypothetical: `docs/generated/ltx-temporal-form-fit-sc-18810.json` emits objects of exactly
/// this shape from MLX LTX captures, `vram_gate::tests::ltx_cross_curve` lifts them without
/// adaptation, and `vram_gate::tests::the_ltx_fitted_curve_round_trips_through_the_shipped_reader`
/// evaluates them through this very function. Nothing about the object says which GPU produced it.
/// `expected_lane` is therefore a required argument rather than an ambient assumption: a curve that
/// NAMES a lane is evaluated only by that lane's reader, and a mismatch (or an unrecognized lane
/// spelling) fails the curve closed exactly as a malformed coefficient does.
///
/// An ABSENT `measurementLane` inherits the enclosing fit block's REQUIRED tag, which the caller has
/// already checked through [`container_measurement_lane`] — so there is no untagged state, and the
/// 36 shipped image curves need no per-curve migration.
///
/// A PRESENT `perMpxFrameGb` that [`json_f64`] cannot read still fails closed, and so does a
/// negative or non-finite one. Only true absence is zero; a malformed value must not silently
/// degrade a video curve into an image curve. "Unreadable" is [`json_f64`]'s notion, not JSON's:
/// it accepts a NUMERIC STRING, so `"0.3"` evaluates exactly as `0.3` would — deliberately, and
/// identically to `fixedGb`/`perMpxGb`, which have always been read the same way. Rejecting a
/// string-typed coefficient is the SCHEMA's job (`model-manifest.schema.json`
/// `#/$defs/phaseVramCurve` types all three as `number`), and
/// `test_schema_admits_the_temporal_coefficient_additively` is where that rejection is pinned.
/// The string case is carried as an explicitly-ACCEPTED control so the fail-closed set is not read
/// as broader than it is.
///
/// SC-16514 recovered the q8/bf16 768² captures from SC-15205 activity 15272 and SC-15206 activity
/// 15314 into `turboFit.evidenceRecords`. Every tier now carries 768² and 1024² phase cells, and every
/// `perMpxGb` is fitted from that tier's own phase delta. The recovered cells are explicitly
/// `phase_fit_only`: the cited activities do not establish geometry-specific 768² output parity, so
/// they characterize the bounded curve without authorizing exact runtime admission. Q8's 7.98
/// denoise slope equals q4's because both measured rises are 3.658 GiB, not because the coefficient
/// was borrowed. Zero geometry-sensitive slopes remain only where the two samples are flat or
/// decrease; the manifest names each such pair. `maxMeasuredPixels` remains 1024² because larger
/// attention shapes have not been validated, so the curve is fitted within that bound rather than
/// extrapolated beyond it.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn fitted_phase_curve_gb(
    phase: &JsonObject,
    geometry: CurveGeometry,
    expected_lane: MeasurementLane,
) -> Option<f64> {
    if let Some(declared) = phase.get("measurementLane") {
        // Present-but-foreign and present-but-unreadable both refuse. Only true ABSENCE inherits
        // the container's tag — the same discipline `perMpxFrameGb` follows below, for the same
        // reason: a value we cannot read must never degrade into a value we invent.
        if MeasurementLane::from_key(declared.as_str()?)? != expected_lane {
            return None;
        }
    }
    let fixed = phase.get("fixedGb").and_then(json_f64)?;
    let per_mpx = phase.get("perMpxGb").and_then(json_f64)?;
    let per_mpx_frame = match phase.get("perMpxFrameGb") {
        None => 0.0,
        Some(value) => json_f64(value)?,
    };
    if !fixed.is_finite()
        || !per_mpx.is_finite()
        || !per_mpx_frame.is_finite()
        || fixed < 0.0
        || per_mpx < 0.0
        || per_mpx_frame < 0.0
    {
        return None;
    }
    // A zero-frame request is not a still image, it is a nonsense geometry. Fail closed rather
    // than silently pricing it as the intercept.
    if geometry.frames == 0 {
        return None;
    }
    let area_term = per_mpx * geometry.pixels as f64 / 1_000_000.0;
    let temporal_term =
        per_mpx_frame * geometry.pixels as f64 / 1_000_000.0 * f64::from(geometry.frames);
    Some(fixed + area_term + temporal_term)
}

/// A rung's complete fitted-curve prediction: one [`fitted_phase_curve_gb`] per canonical phase,
/// in [`BindingPhase`] order.
///
/// `phase_keys` is the lane's spelling of `(conditioning, denoise, decode)` — the Krea ladder's
/// historical manifest vocabulary calls the first phase `text`, the video lane calls it
/// `conditioning`, and translating at the evidence-read boundary is the same discipline
/// `vram_gate::krea_turbo_manifest_key` applies to rung names. The ORDER is the mechanism's, so a
/// triple can be handed to [`binding_phase`] without a per-lane permutation to get wrong.
///
/// Every phase is required: a rung missing one curve is an incomplete fit, and an incomplete fit
/// predicts nothing. The reads short-circuit in phase order, so the first unreadable curve is the
/// one that refuses.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn fitted_phase_triple(
    rung: &JsonObject,
    phase_keys: [&str; 3],
    geometry: CurveGeometry,
    lane: MeasurementLane,
) -> Option<[f64; 3]> {
    let phase = |name: &str| {
        rung.get(name)
            .and_then(Value::as_object)
            .and_then(|curve| fitted_phase_curve_gb(curve, geometry, lane))
    };
    Some([
        phase(phase_keys[0])?,
        phase(phase_keys[1])?,
        phase(phase_keys[2])?,
    ])
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Declared manifest scalars — floors, not peaks (epic 19048 R3, sc-19054)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// What a declared per-tier manifest scalar (`vramGbByTier` / `sequentialPeakGb` /
/// `minMemoryGb`) may truthfully claim for one request geometry (epic 19048 R3).
///
/// Production consumers are `candle_scalar_gate` (compiled on `any(macos, backend-candle)`) and
/// the candle selector bridge — hence the dead-code allowance on the "neither" build, the same
/// shape [`BindingPhase::index`] carries for its candle-only consumer.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeclaredScalarClass {
    /// The scalar was measured (epic 18472's `measured` flag) at a declared geometry that COVERS
    /// this request: the peak is monotone non-decreasing in output geometry, so a measurement at
    /// `vramMeasuredPixels` bounds every smaller same-shape request from above. Presentable as the
    /// request's peak, exactly as before this story.
    MeasuredPeak,
    /// Everything else — unmeasured (`measured` false or absent), no declared geometry, a request
    /// beyond the declared geometry, or a workload shape (batch/frames) the scalar's single-image
    /// capture never saw. The scalar is only the DECLARED FLOOR it always truthfully was:
    /// admission may still lean on it (a floor that under-counts only admits — the pre-epic
    /// posture) but it must be graded as floor-class evidence, never presented as the peak.
    DeclaredFloor,
}

/// Classify one declared scalar for one request (epic 19048 R3). This is mechanism law, not lane
/// plumbing: which geometries a single-cell measurement may claim is the same question
/// [`basis_geometry_is_scalable`] answers for record bases, and the answer must not fork per lane.
///
/// * `measured` — epic 18472's bare-boolean evidence-class flag, the sibling of `vramGbByTier`.
///   Adopted as-is; a parallel evidence-class marker must not be invented.
/// * `declared_pixels` — the scalar's `vramMeasuredPixels` capture geometry. It declares PIXELS of
///   a batch-1 single-image capture, so a request with `batch != 1` or `frames != 1` is a
///   different workload SHAPE (the exact discipline [`basis_geometry_is_scalable`] applies to
///   record bases) and can only take the floor reading.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn declared_scalar_class(
    measured: bool,
    declared_pixels: Option<u64>,
    request: MemoryGeometry,
) -> DeclaredScalarClass {
    let covered = measured
        && request.batch == 1
        && request.frames == 1
        && declared_pixels.is_some_and(|pixels| {
            pixels > 0 && u64::from(request.width) * u64::from(request.height) <= pixels
        });
    if covered {
        DeclaredScalarClass::MeasuredPeak
    } else {
        DeclaredScalarClass::DeclaredFloor
    }
}

/// **Grade a declared scalar for admission** (epic 19048 R1/R3; hoisted here by sc-19055).
///
/// A [`DeclaredScalarClass::MeasuredPeak`] compares as-is — it IS the peak for this request. A
/// [`DeclaredScalarClass::DeclaredFloor`] is widened by the lane's own estimate margin, which is
/// exactly the grade [`crate::memory_strategy::select_strategy`] gives a
/// [`CandidateBasis::EstimateFloor`] candidate. Gates that have no selector downstream of them (the
/// candle image scalar gate, and since sc-19055 the flat video fit errors) apply it here instead, so
/// the two positions cannot drift into two different gradings of the same class of evidence.
///
/// Widening is [`crate::memory_strategy::widened_peak_bytes`] over integer bytes — the selector's
/// own arithmetic, never a second law — and the margin is
/// [`EstimateLane::estimate_margin`], never a copied constant.
///
/// Lives in the mechanism rather than in either lane because R1 forbids a prediction law in a
/// backend-local module: before sc-19055 this function was `candle_scalar_gate::graded_scalar_gb`,
/// reachable only from the image lane, so the video lane could not have consumed it without either
/// importing the image gate or growing a second copy.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn graded_scalar_bytes(
    peak_bytes: u64,
    class: DeclaredScalarClass,
    lane: EstimateLane,
) -> u64 {
    match class {
        DeclaredScalarClass::MeasuredPeak => peak_bytes,
        DeclaredScalarClass::DeclaredFloor => {
            crate::memory_strategy::widened_peak_bytes(peak_bytes, lane.estimate_margin())
        }
    }
}

/// [`graded_scalar_bytes`] for a caller that holds GB rather than bytes.
///
/// The GB -> integer-byte -> GB round trip is deliberate and is the arithmetic the pre-sc-19055
/// candle image gate already used: widening MUST happen in the selector's integer-byte unit so a
/// gate-side grade and a selector-side grade of the same number agree exactly rather than to within
/// a float rounding.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn graded_scalar_gb(
    peak_gb: f64,
    class: DeclaredScalarClass,
    lane: EstimateLane,
) -> f64 {
    match class {
        DeclaredScalarClass::MeasuredPeak => peak_gb,
        DeclaredScalarClass::DeclaredFloor => {
            let bytes = (peak_gb * (1024.0 * 1024.0 * 1024.0))
                .ceil()
                .clamp(0.0, u64::MAX as f64) as u64;
            crate::memory_strategy::peak_bytes_to_gb(graded_scalar_bytes(bytes, class, lane))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The binding phase — ONE argmax, three lanes
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The canonical measurement phases, in ladder order.
///
/// A property of the GEOMETRY, not of the model: the binding phase is measured to flip inside a
/// single model's envelope (sc-18812). Every lane labels these phases in its own vocabulary
/// (`vram_gate::BindingPhase::Text`, `video_admission::VideoBindingPhase::Conditioning`) but there
/// is exactly one rule for picking one, and it is [`binding_phase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindingPhase {
    Conditioning,
    Denoise,
    Decode,
}

impl BindingPhase {
    /// The phase index (0 conditioning, 1 denoise, 2 decode) the sc-18097 comparison seam is
    /// written in. Lives on the shared type so the Krea ladder does not have to keep a lane-local
    /// copy of the ordering.
    ///
    /// Its only production consumer is `vram_gate::krea_binding_phase`, which compiles on the
    /// candle lane alone — hence the dead-code allowance on every other build, the same shape
    /// `lib.rs` uses for the candle-only modules themselves.
    #[cfg_attr(
        not(all(not(target_os = "macos"), feature = "backend-candle")),
        allow(dead_code)
    )]
    pub(crate) const fn index(self) -> u8 {
        match self {
            Self::Conditioning => 0,
            Self::Denoise => 1,
            Self::Decode => 2,
        }
    }
}

/// The phase carrying the peak of a `(conditioning, denoise, decode)` triple.
///
/// Ties resolve to the LATER phase deterministically. Generic in the magnitude type because the
/// lanes carry different ones — MLX and the video lane predict `u64` bytes, the Krea turbo ladder
/// predicts `f64` GB — while the rule is identical.
///
/// ## What NaN actually does here (corrected in sc-19058)
///
/// An earlier revision of this comment claimed "a NaN phase can never claim the binding label".
/// That is FALSE in first position. `>=` on a NaN is false, so a NaN `conditioning` is never
/// DISPLACED: `binding_phase(NaN, 5.0, 3.0)` returns `Conditioning`. NaN in the second or third
/// position genuinely cannot claim the label, but the first is not screened at all — the seed is
/// taken, not compared.
///
/// This is reachable rather than theoretical. `vram_gate::krea_record_phase_peaks` builds a triple
/// straight out of `predictedPhasesGb` through `crate::payload::json_f64` with no finite guard, and
/// `json_f64` falls back to `str::parse`, which accepts `"NaN"`. A manifest record spelling a phase
/// peak that way therefore produces a triple on which this function and
/// `vram_gate::KreaTurboPhasePeaks::peak_gb` disagree: `f64::max` discards the NaN and reports a
/// finite peak from another phase, while this argmax names the NaN phase.
///
/// Left as a recorded inconsistency rather than repaired here, deliberately. A finite guard is a
/// DECISION change on the one route epic 19048 R6 holds to byte-identity, and it belongs where the
/// triple is built (a malformed record should fail its estimate closed, exactly as a malformed
/// curve coefficient already does) rather than inside a shared argmax that several lanes call with
/// integer magnitudes that cannot be NaN at all.
///
/// Before sc-19050 this rule existed three times, each copy documented as "mirroring" the others.
/// That is precisely the drift hazard epic 19048 R1 forbids: three mirrors have three chances to
/// stop mirroring.
pub(crate) fn binding_phase<T: PartialOrd>(conditioning: T, denoise: T, decode: T) -> BindingPhase {
    let mut phase = BindingPhase::Conditioning;
    let mut peak = conditioning;
    if denoise >= peak {
        phase = BindingPhase::Denoise;
        peak = denoise;
    }
    if decode >= peak {
        phase = BindingPhase::Decode;
    }
    phase
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Rung <-> calibration-rung mapping
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The calibration rung a gen-core strategy records evidence under.
pub(crate) const fn strategy_rung(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

/// The inverse of [`strategy_rung`].
pub(crate) const fn evidence_strategy(rung: StrategyRung) -> MemoryStrategy {
    match rung {
        StrategyRung::Resident => MemoryStrategy::Resident,
        StrategyRung::StagedResidency => MemoryStrategy::StagedResidency,
        StrategyRung::BoundedDecode => MemoryStrategy::BoundedDecode,
        StrategyRung::BoundedAttention => MemoryStrategy::BoundedAttention,
        StrategyRung::BoundedTransformerResidency => MemoryStrategy::BoundedTransformerResidency,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Lane parameters
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// What an admission overshoot costs on a lane. This is the *reason* the two lanes' margins differ,
/// carried beside them so a future edit cannot equalize the numbers without confronting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailurePosture {
    /// MLX: the default error handler calls `exit(-1)`; there is no recoverable `Err` on that lane,
    /// so an under-prediction takes the whole worker process down mid-render.
    FatalProcessAbort,
    /// candle/CUDA: an allocation failure is a recoverable `Err` that fails the job while the
    /// worker survives.
    RecoverableError,
}

/// Everything the mechanism needs to know about the backend it is synthesizing for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EstimateLane {
    pub(crate) backend: gen_core::MemoryBackend,
    /// Stable tracing label for this lane's synthesis events.
    pub(crate) label: &'static str,
    pub(crate) failure_posture: FailurePosture,
}

impl EstimateLane {
    /// The margin the SELECTOR will widen this lane's candidates by.
    ///
    /// **Read from the one per-backend lookup, never restated.** Copying the constant into this
    /// struct would create a second declaration of a derived number — the drift hazard this module
    /// exists to remove — so the lane carries the POSTURE (which is nowhere else expressed as data)
    /// and defers the margin to `memory_strategy::estimate_margin`.
    pub(crate) const fn estimate_margin(self) -> f64 {
        crate::memory_strategy::estimate_margin(self.backend)
    }
}

/// The MLX lane: an allocator overshoot aborts the process, which is why its margin is the wider
/// one.
pub(crate) const MLX_LANE: EstimateLane = EstimateLane {
    backend: gen_core::MemoryBackend::Mlx,
    label: "mlx",
    failure_posture: FailurePosture::FatalProcessAbort,
};

/// The candle lane: an allocation failure is a recoverable `Err`, which is why its margin is the
/// narrower one. Consumed by the video gate, which runs on both lanes.
pub(crate) const CANDLE_LANE: EstimateLane = EstimateLane {
    backend: gen_core::MemoryBackend::Candle,
    label: "candle",
    failure_posture: FailurePosture::RecoverableError,
};

/// **The one place the two lane vocabularies are reconciled** (epic 19048 R4, sc-19056).
///
/// [`MeasurementLane`] is where a number was captured (it rides calibration evidence, the packaged
/// video curves and the manifest fit blocks); [`gen_core::MemoryBackend`] is which engine is
/// executing. They are separate types because `sceneworks-core` deliberately carries no gen-core
/// dependency, and they answer different questions — but the R4 comparison is meaningless unless
/// one maps onto the other, so the mapping lives here in the mechanism rather than being re-guessed
/// at each producer.
///
/// Both arms are spelled out so a third lane cannot be added on one side without deciding what it
/// is on the other.
pub(crate) const fn lane_of(measured_on: MeasurementLane) -> gen_core::MemoryBackend {
    match measured_on {
        MeasurementLane::Mlx => gen_core::MemoryBackend::Mlx,
        MeasurementLane::Candle => gen_core::MemoryBackend::Candle,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The measured basis
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// One verified measured cell usable as the extrapolation basis for a fitted estimate (sc-18096):
/// same provider, tier, mode, and overlay as the request, artifact-current AND closure-current
/// binding, but a geometry that differs only in the scalable axes
/// ([`basis_geometry_is_scalable`]) — the cell the request itself could not be admitted on.
///
/// Closure-current is a deliberate restriction, not an oversight: the estimate margin was derived
/// to cover extrapolation error on top of same-closure re-capture variance
/// ([`crate::ladder_margin_policy`]). A stale-closure record already carries its own corpus-derived
/// drift allowance on the MEASURED path; stacking that drift under an extrapolation would spend the
/// estimate margin twice, and no derivation covers the sum — so a stale record may keep serving its
/// own cell behind the stale margin (sc-18095) but may not seed an extrapolated estimate.
///
/// Everything the extrapolation, the binding-phase constraint, and the loaded-provider identity
/// gate need is captured here, so synthesis never re-reads the evidence bundle.
#[derive(Clone, Debug)]
pub(crate) struct MeasuredRungBasis {
    /// **The lane this basis was MEASURED on** (epic 19048 R4, sc-19056).
    ///
    /// Every other field here is a number or an identity that reads identically on both backends —
    /// a byte count, a geometry, a rung, a fingerprint string. Nothing in them says which GPU the
    /// measurement came off, and the epic's settled position is that "measurements never transfer
    /// across lanes". So the lane is carried explicitly and
    /// [`synthesize_estimate_ladder`] refuses a basis whose lane is not the requesting lane's.
    ///
    /// The calibration-identity conjunct already in that filter is NOT this guard. A fingerprint is
    /// a free-form provider string: two lanes publishing a same-named contract, or a candle capture
    /// arm reusing an MLX arm's fingerprint (sc-19057 is adding exactly such an arm), collide on it
    /// silently. And an ABI is deliberately lane-neutral — `MEMORY_CALIBRATION_ABI` is one number
    /// for the whole repo. Neither can express "wrong GPU".
    pub(crate) lane: gen_core::MemoryBackend,
    pub(crate) rung: StrategyRung,
    pub(crate) parameters: gen_core::MemoryStrategyParameters,
    pub(crate) engaged_composition: Vec<MemoryStrategy>,
    pub(crate) load_shape: gen_core::LoadShape,
    /// The calibration identity the basis binding was measured under. [`synthesize_estimate_ladder`]
    /// requires it to equal the LOADED contract's identity: a provider whose estimator drifted from
    /// the packaged records must not receive fitted candidates built from them (sc-18096 review).
    /// This gate cannot be left to the `carries_verified_claim` demotion, which only fires when the
    /// route carries a verified claim — bases ride on legacy routes where that may not hold.
    pub(crate) calibration_abi: u32,
    pub(crate) calibration_fingerprint: String,
    pub(crate) geometry: CalibrationGeometry,
    /// Per-phase predicted peaks from the measured record, in canonical phase order
    /// (conditioning, denoise, decode). The binding phase is their argmax.
    pub(crate) conditioning_peak_bytes: u64,
    pub(crate) denoise_peak_bytes: u64,
    pub(crate) decode_peak_bytes: u64,
    /// The measured admission envelope peak the extrapolated estimate scales.
    pub(crate) envelope_peak_bytes: u64,
    pub(crate) record_id: String,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Synthesized candidates
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// One estimate-backed candidate synthesized for an implemented-but-unmeasured rung (sc-18096).
#[derive(Clone, Debug)]
pub(crate) struct SynthesizedEstimate {
    pub(crate) selection: MemorySelection,
    pub(crate) evidence: MemoryEvidence,
    pub(crate) basis: CandidateBasis,
    pub(crate) decode_quality: DecodeQualityRequestDecision,
}

/// What the provider's declared decode-quality scope said about this request's tile domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DecodeQualityRequestDecision {
    NotDeclared,
    NotEngaged {
        strategy: MemoryStrategy,
    },
    Admitted {
        strategy: MemoryStrategy,
        tile_edge: u32,
        overlap: u32,
        evidence_sha256: String,
    },
    Refused {
        strategy: MemoryStrategy,
        tile_edge: Option<u32>,
        overlap: Option<u32>,
        evidence_sha256: Option<String>,
        reason: String,
    },
    Unmeasured {
        strategy: MemoryStrategy,
        reason: String,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct EstimateParameterCandidate {
    pub(crate) parameters: gen_core::MemoryStrategyParameters,
    pub(crate) decode_quality: DecodeQualityRequestDecision,
}

#[derive(Default)]
pub(crate) struct SynthesizedEstimateLadder {
    pub(crate) estimates: Vec<SynthesizedEstimate>,
    pub(crate) decode_quality_decisions: Vec<DecodeQualityRequestDecision>,
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Fabricated evidence for a synthesized candidate
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Fabricated evidence for a synthesized estimate candidate, following the
/// `generic_mlx_shared_observation` / `resident_evidence` pattern: `ImplementedUnverified`
/// conformance, no observed peak, parity not run — the record claims exactly what an estimate can
/// claim and nothing more. The selector's estimate-scoped eligibility wrap (sc-18096,
/// `memory_strategy::candidate_exclusion`) is what admits it.
///
/// `backend` is a parameter rather than a constant because sc-18814 routes the VIDEO lane through
/// this same shape on both backends.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_evidence(
    contract: &MemoryProviderContract,
    backend: gen_core::MemoryBackend,
    tier: MemoryNumericTier,
    mode: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    selection: MemorySelection,
    predicted_peak_bytes: u64,
    calibration_fingerprint: Option<&str>,
) -> MemoryEvidence {
    MemoryEvidence {
        key: MemoryEvidenceKey {
            resolved_route: contract.provider_id.clone(),
            backend,
            tier,
            load_shape: contract.load_shape,
            mode: memory_mode_from_mode_key(mode),
            overlay: overlay.map(str::to_owned),
            geometry,
            strategy: selection.strategy,
            engaged_composition: contract.engaged_composition_for_selection(&selection),
            parameters: selection.parameters,
        },
        conformance: MemoryConformanceState::ImplementedUnverified,
        dimensions: MemoryEvidenceDimensions {
            static_implementation: MemoryEvidenceVerdict::Satisfied,
            declared_calibration: MemoryEvidenceVerdict::Missing,
            historical_verification: MemoryEvidenceVerdict::Missing,
            current_environment_verification: MemoryEvidenceVerdict::Missing,
            canonical_route_loadability: MemoryEvidenceVerdict::Unverified,
            exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: calibration_fingerprint.unwrap_or_default().to_owned(),
        sceneworks_revision: REQUEST_EVIDENCE_REVISION.to_owned(),
        inference_revision: INFERENCE_CONTRACT_REVISION.to_owned(),
        harness_version: String::new(),
        predicted_peak_bytes,
        observed_peak_bytes: None,
        parity: MemoryParityContract::Exact,
        parity_result: MemoryParityResult::NotRun,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Basis 2 — the weights + headroom floor
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The weights one load holds resident, split the way [`compose_resident_floor_bytes`] needs them.
///
/// A provider contract is one source of these numbers ([`floor_weights_bytes`]); an on-disk
/// `.safetensors` sum is another ([`crate::conditioning_fit`], sc-19055). The composition law must
/// not fork between them, which is why the law takes this rather than a contract.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) struct ResidentWeights {
    /// The conditioning stack (text/vision encoders) — the half `StagedResidency` drops.
    pub(crate) conditioning_bytes: u64,
    /// Everything else the base model holds resident.
    pub(crate) heavy_bytes: u64,
    /// The transformer's share of [`Self::heavy_bytes`], which `BoundedTransformerResidency`
    /// windows out. Zero when the source cannot separate it — a source that cannot see the split
    /// must not promise the saving.
    pub(crate) transformer_bytes: u64,
}

/// **The resident-composition law: which declared components a composition actually holds**
/// (sc-18096; hoisted off the contract by sc-19055).
///
/// Nothing here is a tuned coefficient:
///
/// * `StagedResidency` engaged ⇒ the co-residency drop the rung exists for: the resident working
///   set is the larger of the conditioning stack and everything else, exactly the
///   `staged_weights_gb` split the load-time gate has always used.
/// * `BoundedTransformerResidency` engaged ⇒ the transformer's declared bytes leave the resident
///   floor: the rung windows them, and the window slice plus scratch is carried by the headroom
///   term and the estimate margin, not by a guessed window fraction.
/// * Rungs 2 and 3 bound TRANSIENTS, not weights, so they take no weights reduction here — and
///   deliberately no transient reduction either, because no measured basis for one exists on an
///   unmeasured cell. Their floor equals rung 1's, which keeps them selectable without ever
///   promising an unmeasured saving.
/// * Auxiliary components (control branches, adapter stacks, identity encoders, …) stay resident
///   unless their source declares them `bounded_by` a rung the composition engages.
///
/// `auxiliary` is an iterator of `(resident_bytes, bounded_by)` so neither caller has to allocate a
/// vector to describe components it already holds in another shape.
///
/// **Why this is the mechanism's and not each gate's** (epic 19048 R1). The additive overlay
/// accounting the candle conditioning gate does — a base model plus a co-resident second network —
/// is this exact law with the auxiliary term populated from disk instead of from a contract. Before
/// sc-19055 that gate summed its two terms itself, so an overlay could never drop out of the floor
/// when a rung bounded it, and the two positions could drift.
#[cfg_attr(
    not(any(target_os = "macos", feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn compose_resident_floor_bytes<I>(
    weights: &ResidentWeights,
    engaged: &[MemoryStrategy],
    auxiliary: I,
) -> u64
where
    I: IntoIterator<Item = (u64, Option<MemoryStrategy>)>,
{
    let conditioning = weights.conditioning_bytes;
    let mut heavy = weights.heavy_bytes;
    if engaged.contains(&MemoryStrategy::BoundedTransformerResidency) {
        heavy = heavy.saturating_sub(weights.transformer_bytes);
    }
    let base = if engaged.contains(&MemoryStrategy::StagedResidency) {
        conditioning.max(heavy)
    } else {
        conditioning.saturating_add(heavy)
    };
    let auxiliary = auxiliary
        .into_iter()
        .filter(|(_, bounded_by)| match bounded_by {
            Some(bounding) => !engaged.contains(bounding),
            None => true,
        })
        .fold(0_u64, |total, (resident_bytes, _)| {
            total.saturating_add(resident_bytes)
        });
    base.saturating_add(auxiliary)
}

/// The floor's per-rung WEIGHTS term for a load described by a provider contract (sc-18096) —
/// [`compose_resident_floor_bytes`] over the contract's own declarations. The contract supplies the
/// component bytes; the law lives one function up.
pub(crate) fn floor_weights_bytes(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> u64 {
    let facts = contract.asset_facts;
    compose_resident_floor_bytes(
        &ResidentWeights {
            conditioning_bytes: facts.conditioning_bytes,
            heavy_bytes: facts.base_bytes.saturating_sub(facts.conditioning_bytes),
            transformer_bytes: facts.transformer_bytes,
        },
        engaged,
        contract
            .resident_components()
            .iter()
            .filter(|component| component.kind.is_auxiliary())
            .map(|component| (component.resident_bytes, component.bounded_by)),
    )
}

/// The smallest declared value for every numeric knob the engaged composition requires — the most
/// deeply bounding parameters the provider publishes, which keeps the true runtime transient as far
/// below the floor's unreduced headroom charge as the provider allows. `None` when a required knob
/// has no declared range: such a selection cannot be validated, so no candidate is synthesized for
/// the rung.
///
/// This is the whole parameter law for the video lane and for the candle floor arm. The image
/// lane's [`floor_parameter_candidates`] wraps it with the decode-quality decisions those two have
/// no use for.
pub(crate) fn floor_smallest_parameters(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
) -> Option<gen_core::MemoryStrategyParameters> {
    Some(gen_core::MemoryStrategyParameters {
        decode_tile_edge: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedDecode,
            |r| &r.decode_tile_edges,
        )?,
        decode_overlap: smallest_declared(contract, engaged, MemoryStrategy::BoundedDecode, |r| {
            &r.decode_overlaps
        })?,
        attention_chunk_size: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedAttention,
            |r| &r.attention_chunk_sizes,
        )?,
        transformer_window_size: smallest_declared(
            contract,
            engaged,
            MemoryStrategy::BoundedTransformerResidency,
            |r| &r.transformer_window_sizes,
        )?,
        transformer_window_component: None,
    })
}

/// `Some(None)` when the rung is not engaged (the knob is not required), `Some(Some(v))` for the
/// smallest declared value, `None` when the rung IS engaged but publishes no range — the
/// unvalidatable case every caller turns into "synthesize nothing for this rung".
fn smallest_declared(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    strategy: MemoryStrategy,
    pick: fn(&gen_core::MemoryParameterRanges) -> &Vec<u32>,
) -> Option<Option<u32>> {
    if !engaged.contains(&strategy) {
        return Some(None);
    }
    pick(&contract.capability(strategy)?.parameters)
        .iter()
        .copied()
        .min()
        .map(Some)
}

/// The image lane's per-candidate parameter builder: [`floor_smallest_parameters`]'s knobs plus the
/// declared decode-quality scope's admitted tile domain, one candidate per admitted `(edge,
/// overlap)` row, with every refusal/unmeasured decision reported alongside.
#[allow(clippy::too_many_arguments)]
pub(crate) fn floor_parameter_candidates(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    strategy: MemoryStrategy,
    tier: MemoryNumericTier,
    mode_key: &str,
    overlay: Option<&str>,
    geometry: MemoryGeometry,
    use_pid: bool,
) -> (
    Vec<EstimateParameterCandidate>,
    Vec<DecodeQualityRequestDecision>,
) {
    let Some(attention_chunk_size) =
        smallest_declared(contract, engaged, MemoryStrategy::BoundedAttention, |r| {
            &r.attention_chunk_sizes
        })
    else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "required attention chunk range is empty".to_owned(),
            }],
        );
    };
    let Some(transformer_window_size) = smallest_declared(
        contract,
        engaged,
        MemoryStrategy::BoundedTransformerResidency,
        |r| &r.transformer_window_sizes,
    ) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "required transformer window range is empty".to_owned(),
            }],
        );
    };
    let base_parameters = gen_core::MemoryStrategyParameters {
        decode_tile_edge: None,
        decode_overlap: None,
        attention_chunk_size,
        transformer_window_size,
        transformer_window_component: None,
    };
    if !engaged.contains(&MemoryStrategy::BoundedDecode) {
        return (
            vec![EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            }],
            Vec::new(),
        );
    }
    let Some(decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        return (
            Vec::new(),
            vec![DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode capability is absent".to_owned(),
            }],
        );
    };
    if decode.parameters.decode_geometry_policies.is_empty() {
        if contract.decode_geometry_policy_authoritative {
            let refusal = DecodeQualityRequestDecision::Refused {
                strategy,
                tile_edge: None,
                overlap: None,
                evidence_sha256: None,
                reason: "bounded decode has a declared quality scope but no applicable row for the loaded tier/load shape"
                    .to_owned(),
            };
            // The decode rung itself has no exact authority. A higher independent rung retains its
            // own estimate with decode omitted, so request-selection-aware engagement cannot
            // accidentally resurrect the route-blind tile domain.
            let candidates = (strategy != MemoryStrategy::BoundedDecode)
                .then_some(EstimateParameterCandidate {
                    parameters: base_parameters,
                    decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
                })
                .into_iter()
                .collect();
            return (candidates, vec![refusal]);
        }
        let pair = if let Some(routes) = &contract.pid_decode_routes {
            let route = if use_pid { &routes.pid } else { &routes.native };
            route
                .tile_edges
                .iter()
                .copied()
                .min()
                .zip(Some(route.tile_overlap))
        } else {
            decode
                .parameters
                .decode_tile_edges
                .iter()
                .copied()
                .min()
                .zip(decode.parameters.decode_overlaps.iter().copied().min())
        };
        let Some((tile_edge, overlap)) = pair else {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: "legacy bounded decode has no complete edge/overlap pair".to_owned(),
                }],
            );
        };
        let mut parameters = base_parameters;
        parameters.decode_tile_edge = Some(tile_edge);
        parameters.decode_overlap = Some(overlap);
        return (
            vec![EstimateParameterCandidate {
                parameters,
                decode_quality: DecodeQualityRequestDecision::NotDeclared,
            }],
            Vec::new(),
        );
    }

    let query = MemoryDecodePolicyQuery {
        tier,
        load_shape: contract.load_shape,
        mode_key,
        overlay,
        geometry,
        use_pid,
    };
    let rows = match contract.decode_geometry_policies_for_request(query) {
        Ok(rows) => rows,
        Err(error) => {
            return (
                Vec::new(),
                vec![DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: None,
                    overlap: None,
                    evidence_sha256: None,
                    reason: error.to_string(),
                }],
            )
        }
    };
    let mut rows = rows;
    rows.sort_by_key(|policy| (policy.tile_edge, policy.overlap));
    let mut candidates = Vec::new();
    let mut decisions = Vec::new();
    for policy in rows {
        match &policy.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                let mut parameters = base_parameters;
                parameters.decode_tile_edge = Some(policy.tile_edge);
                parameters.decode_overlap = Some(policy.overlap);
                candidates.push(EstimateParameterCandidate {
                    parameters,
                    decode_quality: DecodeQualityRequestDecision::Admitted {
                        strategy,
                        tile_edge: policy.tile_edge,
                        overlap: policy.overlap,
                        evidence_sha256: policy.production_evidence_sha256.clone(),
                    },
                });
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                decisions.push(DecodeQualityRequestDecision::Refused {
                    strategy,
                    tile_edge: Some(policy.tile_edge),
                    overlap: Some(policy.overlap),
                    evidence_sha256: Some(policy.production_evidence_sha256.clone()),
                    reason: reason.clone(),
                });
            }
        }
    }
    if candidates.is_empty() {
        if decisions.is_empty() {
            decisions.push(DecodeQualityRequestDecision::Unmeasured {
                strategy,
                reason: format!(
                    "no exact production-latent row for {}x{} load_shape={:?} mode={} overlay={overlay:?} use_pid={use_pid}",
                    geometry.width, geometry.height, contract.load_shape, mode_key,
                ),
            });
        }
        // Bounded decode itself cannot exist without an admitted pair. A higher independent rung
        // keeps its own parameters and omits decode so selection-aware engagement excludes rung 2.
        if strategy != MemoryStrategy::BoundedDecode {
            candidates.push(EstimateParameterCandidate {
                parameters: base_parameters,
                decode_quality: DecodeQualityRequestDecision::NotEngaged { strategy },
            });
        }
    }
    (candidates, decisions)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The synthesis entry point
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Everything one lane's request contributes to synthesis. Grouped into a struct rather than passed
/// as eleven positional arguments so a caller cannot transpose two `&str`s silently.
pub(crate) struct EstimateRequest<'a> {
    pub(crate) lane: EstimateLane,
    pub(crate) contract: &'a MemoryProviderContract,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) mode_key: &'a str,
    pub(crate) overlay: Option<&'a str>,
    pub(crate) geometry: MemoryGeometry,
    pub(crate) use_pid: bool,
    pub(crate) calibration_fingerprint: Option<&'a str>,
    /// **The backend-supplied headroom law**, already evaluated at this request's geometry: the
    /// fixed reserve plus the geometry-scaled activation transient this lane charges. MLX passes
    /// `MlxRequestPlan::generic_headroom_bytes(geometry)` — the exact same headroom convention its
    /// resident baseline charges, so only the weights term differs per rung.
    pub(crate) headroom_bytes: u64,
    /// Whether the ladder may synthesize a FITTED candidate for the RESIDENT rung (sc-19054,
    /// epic 19048 R3) — a verified resident-rung measured cell at another geometry, extrapolated
    /// exactly like the optimized rungs' bases.
    ///
    /// A lane parameter because the two resident baselines differ in kind: MLX's resident
    /// baseline already carries its geometry law (weights plus the area-scaled
    /// `generic_headroom_bytes` transient) so it passes `false` and keeps the historical skip;
    /// candle's resident baseline is a geometry-blind declared manifest scalar
    /// ([`DeclaredScalarClass`]), so it passes `true` and lets measured evidence supersede the
    /// scalar where records exist. Only the fitted arm ever fires for resident: the floor arm is
    /// skipped because every lane already submits its own resident floor candidate on every
    /// request, and a second floor here would duplicate that cell.
    pub(crate) synthesize_resident: bool,
}

/// Synthesize estimate-backed candidates for every optimized rung the provider contract marks
/// `Implemented` (sc-18096, epic 18093 R1a) — and, when the lane opts in
/// ([`EstimateRequest::synthesize_resident`], sc-19054), a fitted-only candidate for the RESIDENT
/// rung. Called only on legacy admission routes — a covered cell is authorized by its exact
/// measured ladder and gets no synthetic sibling.
///
/// Peak source per rung, in preference order:
///
/// 1. **Fitted basis** — a verified measured cell at a different geometry ([`MeasuredRungBasis`]),
///    extrapolated over [`voxels`]: the conditioning peak is regressor-flat (text encoding does not
///    grow with the render target) while denoise, decode, and the admission envelope scale by
///    [`extrapolation_scale`]. Gated by
///    [`crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`]: if the
///    extrapolated triple's binding phase differs from the measured cell's, the fitted candidate is
///    NOT emitted (no per-phase variance re-derivation exists) and the rung falls back to the
///    floor, whose no-measured-basis path the constraint's scope sentence explicitly exempts.
/// 2. **Weights + headroom floor** — [`floor_weights_bytes`] plus
///    [`EstimateRequest::headroom_bytes`].
pub(crate) fn synthesize_estimate_ladder(
    request: &EstimateRequest<'_>,
    bases: &[MeasuredRungBasis],
) -> SynthesizedEstimateLadder {
    use crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE;

    let contract = request.contract;
    let geometry = request.geometry;
    let lane = request.lane.label;
    let mut ladder = SynthesizedEstimateLadder::default();
    for strategy in MemoryStrategy::ALL {
        if strategy == MemoryStrategy::Resident && !request.synthesize_resident {
            // The resident baseline candidate already exists on every legacy route; only a lane
            // whose baseline is a geometry-blind declared scalar opts into a fitted sibling
            // (sc-19054; see `EstimateRequest::synthesize_resident`).
            continue;
        }
        if !matches!(
            contract.capability(strategy).map(|cap| &cap.support),
            Some(gen_core::MemoryStrategySupport::Implemented)
        ) {
            continue;
        }
        let declared_engaged = contract.engaged_composition(strategy);
        let (parameter_candidates, decisions) = floor_parameter_candidates(
            contract,
            &declared_engaged,
            strategy,
            request.tier,
            request.mode_key,
            request.overlay,
            geometry,
            request.use_pid,
        );
        ladder.decode_quality_decisions.extend(decisions);

        for parameter_candidate in parameter_candidates {
            let floor_selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: request.tier,
            };
            let engaged = contract.engaged_composition_for_selection(&floor_selection);

            // 1. Fitted basis: the closest measured geometry below the request, else the smallest
            //    above it (whose clamp-at-1.0 scaling degenerates to the measurement itself). The
            //    basis must have been measured under the LOADED provider's exact calibration
            //    identity: a drifted estimator invalidates the measured numbers as an extrapolation
            //    seed, and this is the only gate on legacy routes (the `carries_verified_claim`
            //    demotion never fires without a verified claim on the route). A contract with no
            //    calibration identity gets no fitted candidates at all — fail closed.
            //
            //    The load-shape conjunct compares CONTRACT shape only — deliberately, and unlike
            //    the Evidence-path filter's measured-candidate leg, which also compares
            //    `identity.load_shape` (sc-18251). An estimate-basis candidate is graded downstream
            //    by the estimate wrap of `optimized_eligibility`, which short-circuits at the
            //    conformance gate's `Unverified` BEFORE the identity load-shape comparison ever
            //    runs, so the identity's shape is never consulted for an estimate. Adding the
            //    conjunct here would be stricter than the gate this filter anticipates.
            let fitted = bases
                .iter()
                .filter(|basis| {
                    // **R4 (sc-19056): a foreign lane's measurement is not a basis.** First
                    // conjunct deliberately — it is the cheapest and the one whose failure is least
                    // recoverable. An MLX allocator peak extrapolated into a CUDA admission (or the
                    // reverse) is not "approximately right"; the two lanes measure different
                    // hardware, different allocators and different residency semantics, which is
                    // the whole reason this mechanism keeps per-lane coefficients under one law.
                    //
                    // Fail-closed means the basis is DROPPED, not silently rescaled: the rung falls
                    // through to the weights+headroom floor below, which is this lane's own number.
                    basis.lane == request.lane.backend
                        // sc-19054: the corpus-witnessed extrapolation bound. A basis too far
                        // below the request is not a seed; a NEARER basis in the same set can
                        // still serve, and with none in range the rung falls to the floor arm
                        // exactly like the binding-phase-flip refusal below.
                        && basis_within_extrapolation_bound(basis.geometry, geometry)
                        && basis.rung == strategy_rung(strategy)
                        && basis.load_shape == contract.load_shape
                        && basis.engaged_composition == engaged
                        && contract.calibration.as_ref().is_some_and(|identity| {
                            identity.abi == basis.calibration_abi
                                && identity.fingerprint == basis.calibration_fingerprint
                        })
                })
                .max_by_key(|basis| {
                    let measured = basis_voxels(basis.geometry);
                    let below = measured <= request_voxels(geometry);
                    // Rank every below-request basis above every above-request one; among "below"
                    // take the largest, among "above" the smallest.
                    (
                        below,
                        if below {
                            measured as i128
                        } else {
                            -(measured as i128)
                        },
                    )
                })
                .and_then(|basis| {
                    let mut parameters = basis.parameters;
                    parameters.decode_tile_edge = parameter_candidate.parameters.decode_tile_edge;
                    parameters.decode_overlap = parameter_candidate.parameters.decode_overlap;
                    let selection = MemorySelection {
                        strategy,
                        parameters,
                        tier: request.tier,
                    };
                    if contract.validate_selection(&selection).is_err() {
                        return None;
                    }
                    let scale = extrapolation_scale(basis.geometry, geometry);
                    let scaled = |bytes: u64| {
                        (bytes as f64 * scale).ceil().clamp(0.0, u64::MAX as f64) as u64
                    };
                    let measured_binding = binding_phase(
                        basis.conditioning_peak_bytes,
                        basis.denoise_peak_bytes,
                        basis.decode_peak_bytes,
                    );
                    let extrapolated_binding = binding_phase(
                        basis.conditioning_peak_bytes,
                        scaled(basis.denoise_peak_bytes),
                        scaled(basis.decode_peak_bytes),
                    );
                    if ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
                        && extrapolated_binding != measured_binding
                    {
                        // The pinned sc-18094 constraint: the corpus shows a 17.14% per-phase
                        // re-capture spread that no margin in the policy absorbs, so an
                        // extrapolation that moves the request peak onto a different phase than the
                        // one measured is refused rather than margined. The rung falls back to the
                        // floor path below.
                        tracing::info!(
                            route = contract.provider_id,
                            backend = lane,
                            ?strategy,
                            failure_posture = ?request.lane.failure_posture,
                            basis_record = basis.record_id,
                            measured_binding_phase = ?measured_binding,
                            extrapolated_binding_phase = ?extrapolated_binding,
                            "fitted-curve estimate rejected: extrapolation flips the binding phase \
                             (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
                        );
                        return None;
                    }
                    let predicted_peak_bytes = scaled(basis.envelope_peak_bytes);
                    tracing::info!(
                        route = contract.provider_id,
                        backend = lane,
                        ?strategy,
                        basis_record = basis.record_id,
                        basis_geometry =
                            format!("{}x{}", basis.geometry.width, basis.geometry.height),
                        raw_peak_bytes = predicted_peak_bytes,
                        estimate_margin = request.lane.estimate_margin(),
                        voxel_scale = scale,
                        "synthesized fitted-curve estimate candidate from a measured cell"
                    );
                    Some(SynthesizedEstimate {
                        selection,
                        evidence: estimate_evidence(
                            contract,
                            request.lane.backend,
                            request.tier,
                            request.mode_key,
                            request.overlay,
                            geometry,
                            selection,
                            predicted_peak_bytes,
                            request.calibration_fingerprint,
                        ),
                        basis: CandidateBasis::EstimateFittedCurve,
                        decode_quality: parameter_candidate.decode_quality.clone(),
                    })
                });
            if let Some(candidate) = fitted {
                ladder
                    .decode_quality_decisions
                    .push(candidate.decode_quality.clone());
                ladder.estimates.push(candidate);
                continue;
            }
            if strategy == MemoryStrategy::Resident {
                // Fitted-only for resident (sc-19054): every lane already submits its own resident
                // floor candidate on every request, so a refused/absent resident basis simply
                // leaves that baseline in place rather than duplicating it here.
                continue;
            }

            // 2. Weights + headroom floor — no measured basis, so the binding-phase constraint does
            //    not gate it (scope sentence on the constraint's doc).
            let selection = MemorySelection {
                strategy,
                parameters: parameter_candidate.parameters,
                tier: request.tier,
            };
            if contract.validate_selection(&selection).is_err() {
                continue;
            }
            let predicted_peak_bytes =
                floor_weights_bytes(contract, &engaged).saturating_add(request.headroom_bytes);
            tracing::info!(
                route = contract.provider_id,
                backend = lane,
                ?strategy,
                failure_posture = ?request.lane.failure_posture,
                estimate_margin = request.lane.estimate_margin(),
                raw_peak_bytes = predicted_peak_bytes,
                "synthesized weights+headroom floor estimate candidate"
            );
            let candidate = SynthesizedEstimate {
                selection,
                evidence: estimate_evidence(
                    contract,
                    request.lane.backend,
                    request.tier,
                    request.mode_key,
                    request.overlay,
                    geometry,
                    selection,
                    predicted_peak_bytes,
                    request.calibration_fingerprint,
                ),
                basis: CandidateBasis::EstimateFloor,
                decode_quality: parameter_candidate.decode_quality,
            };
            ladder
                .decode_quality_decisions
                .push(candidate.decode_quality.clone());
            ladder.estimates.push(candidate);
        }
    }
    ladder
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: u32, height: u32, frames: u32) -> CalibrationGeometry {
        CalibrationGeometry {
            width,
            height,
            batch: 1,
            frames,
        }
    }

    fn request(width: u32, height: u32, frames: u32) -> MemoryGeometry {
        MemoryGeometry {
            width,
            height,
            batch: 1,
            frames,
            reference_count: 0,
        }
    }

    /// R1's axis list is asserted, not narrated. A fifth axis must be added here and to
    /// [`basis_geometry_is_scalable`] in the same edit.
    #[test]
    fn the_mechanism_names_exactly_the_four_geometry_axes() {
        assert_eq!(
            GEOMETRY_AXES,
            ["area", "frames", "batch", "reference_count"]
        );
    }

    /// The regressor is voxels, and it REDUCES to the pre-extraction area law on every basis the
    /// scalable-axis predicate admits. This is the assertion that makes "voxels" a safe extraction
    /// rather than a decision change: `basis_geometry_is_scalable` holds frames equal, so the
    /// voxel ratio and the area ratio are the same number.
    #[test]
    fn the_voxel_regressor_equals_the_area_law_wherever_a_basis_is_admissible() {
        for (bw, bh, rw, rh, frames) in [
            (1024_u32, 1024_u32, 2048_u32, 2048_u32, 1_u32),
            (1024, 1024, 1024, 1536, 1),
            (768, 512, 1280, 704, 121),
            (2048, 2048, 1024, 1024, 9),
        ] {
            let basis = geometry(bw, bh, frames);
            let req = request(rw, rh, frames);
            assert!(
                basis_geometry_is_scalable(
                    basis,
                    CalibrationGeometry {
                        width: rw,
                        height: rh,
                        batch: 1,
                        frames
                    }
                ) || (bw == rw && bh == rh),
                "fixture must be an admissible basis"
            );
            let area_law =
                (f64::from(rw) * f64::from(rh) / (f64::from(bw) * f64::from(bh))).max(1.0);
            assert_eq!(extrapolation_scale(basis, req), area_law);
        }
    }

    /// The frames axis is LIVE in the regressor even though today's basis predicate holds it
    /// equal: this is what "a new axis lands in one place and both lanes get it" has to mean.
    #[test]
    fn the_regressor_spans_frames_not_only_area() {
        let basis = geometry(1024, 1024, 1);
        assert_eq!(extrapolation_scale(basis, request(1024, 1024, 4)), 4.0);
        assert_eq!(voxels(1024, 1024, 4), voxels(2048, 1024, 2));
    }

    /// A cell that differs in batch or frames is NOT a scalable basis, and an identical cell is
    /// exact evidence rather than a basis. Extracted as-is from `collect_estimate_bases`.
    #[test]
    fn only_the_scalable_axes_may_differ_between_a_basis_and_its_request() {
        let req = geometry(1024, 1024, 1);
        assert!(!basis_geometry_is_scalable(req, req), "identical cell");
        assert!(basis_geometry_is_scalable(geometry(768, 768, 1), req));
        assert!(
            !basis_geometry_is_scalable(geometry(768, 768, 2), req),
            "a different frame count is a different workload shape"
        );
        assert!(
            !basis_geometry_is_scalable(
                CalibrationGeometry {
                    width: 768,
                    height: 768,
                    batch: 2,
                    frames: 1
                },
                req
            ),
            "a different batch is a different workload shape"
        );
    }

    /// The scale never predicts below the measurement.
    /// **The resident-composition law, exercised on the shape a gate with no contract supplies**
    /// (sc-19055).
    ///
    /// [`compose_resident_floor_bytes`] was hoisted out of [`floor_weights_bytes`] so the candle
    /// conditioning gate — which sources component bytes from an on-disk `.safetensors` scan rather
    /// than from a `MemoryProviderContract` — composes them under the SAME law. Each clause is
    /// separated by a distinct byte count, so no assertion can pass on a coincidence:
    ///
    /// * an unbounded auxiliary (the overlay) is always held;
    /// * an auxiliary bounded by an ENGAGED rung drops out — the clause that makes routing through
    ///   the mechanism load-bearing rather than a rename, because a hand-written `base + overlay`
    ///   cannot express it;
    /// * an auxiliary bounded by a rung the composition does NOT engage is still held;
    /// * `BoundedTransformerResidency` removes the transformer's declared share, and only it.
    #[test]
    fn the_resident_composition_holds_an_overlay_until_a_rung_bounds_it() {
        let weights = ResidentWeights {
            conditioning_bytes: 100,
            heavy_bytes: 1_000,
            transformer_bytes: 700,
        };
        let overlay = |bounded_by| [(7_u64, bounded_by)];

        // Resident: every term is held. 100 + 1000 + 7.
        assert_eq!(
            compose_resident_floor_bytes(&weights, &[MemoryStrategy::Resident], overlay(None)),
            1_107
        );
        // Staged: the co-residency drop, max(100, 1000) = 1000, overlay still held.
        assert_eq!(
            compose_resident_floor_bytes(
                &weights,
                &[MemoryStrategy::Resident, MemoryStrategy::StagedResidency],
                overlay(None),
            ),
            1_007
        );
        // The overlay is bounded by a rung the composition ENGAGES ⇒ it leaves the floor.
        assert_eq!(
            compose_resident_floor_bytes(
                &weights,
                &[MemoryStrategy::Resident, MemoryStrategy::BoundedAttention],
                overlay(Some(MemoryStrategy::BoundedAttention)),
            ),
            1_100,
            "an auxiliary bounded by an engaged rung must not stay in the resident floor"
        );
        // The same declaration, with that rung NOT engaged ⇒ still held. This is the pair that
        // proves the `bounded_by` filter reads the composition rather than ignoring it.
        assert_eq!(
            compose_resident_floor_bytes(
                &weights,
                &[MemoryStrategy::Resident],
                overlay(Some(MemoryStrategy::BoundedAttention)),
            ),
            1_107
        );
        // Windowing the transformer removes its declared share of `heavy`, and nothing else.
        assert_eq!(
            compose_resident_floor_bytes(
                &weights,
                &[
                    MemoryStrategy::Resident,
                    MemoryStrategy::BoundedTransformerResidency,
                ],
                overlay(None),
            ),
            407,
            "100 conditioning + (1000 - 700) heavy + 7 overlay"
        );
    }

    #[test]
    fn the_extrapolation_scale_is_floored_at_one() {
        assert_eq!(
            extrapolation_scale(geometry(2048, 2048, 1), request(512, 512, 1)),
            1.0
        );
        // A degenerate zero-voxel basis cannot divide; it degenerates to the measurement.
        assert_eq!(
            extrapolation_scale(geometry(0, 1024, 1), request(1024, 1024, 1)),
            1.0
        );
    }

    /// One argmax, ties to the LATER phase, on both magnitude types the lanes carry.
    #[test]
    fn the_binding_phase_is_the_argmax_with_ties_to_the_later_phase() {
        assert_eq!(binding_phase(9_u64, 1, 1), BindingPhase::Conditioning);
        assert_eq!(binding_phase(1_u64, 9, 1), BindingPhase::Denoise);
        assert_eq!(binding_phase(1_u64, 1, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9_u64, 9, 1), BindingPhase::Denoise);
        assert_eq!(binding_phase(9_u64, 1, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9_u64, 9, 9), BindingPhase::Decode);
        assert_eq!(binding_phase(9.0_f64, 9.0, 1.0), BindingPhase::Denoise);
        assert_eq!(binding_phase(1.0_f64, 1.0, 1.0), BindingPhase::Decode);
        // NaN can never claim the binding label — the fail-closed direction.
        assert_eq!(
            binding_phase(5.0_f64, f64::NAN, f64::NAN),
            BindingPhase::Conditioning
        );
    }

    /// The rung mapping is a bijection. A new rung on either side must be added to both arms, and
    /// a transposed arm reds here rather than silently mislabelling every record.
    #[test]
    fn the_rung_mapping_round_trips_in_both_directions() {
        for strategy in MemoryStrategy::ALL {
            assert_eq!(evidence_strategy(strategy_rung(strategy)), strategy);
        }
        for rung in [
            StrategyRung::Resident,
            StrategyRung::StagedResidency,
            StrategyRung::BoundedDecode,
            StrategyRung::BoundedAttention,
            StrategyRung::BoundedTransformerResidency,
        ] {
            assert_eq!(strategy_rung(evidence_strategy(rung)), rung);
        }
    }

    /// A contract that DECLARES a decode-quality scope but publishes no admitted `(edge, overlap)`
    /// row for this request: `BoundedDecode` itself cannot exist, while a higher independent rung
    /// keeps its own estimate with decode omitted.
    ///
    /// Written for sc-19050 because this guard survived every mutation the shipped corpus could
    /// reach — the weights-free contract surfaces the decision baseline drives carry no adopted
    /// decode-geometry policies at all (production adopts them from the manifest at load time), so
    /// the `candidates.is_empty()` arm was unreachable there. Dropping the
    /// `strategy != BoundedDecode` conjunct would let rung 2 be selected with a tile domain nobody
    /// measured, which is exactly the route-blind admission the declared scope exists to stop.
    #[test]
    fn bounded_decode_cannot_survive_without_an_admitted_tile_domain() {
        let mut contract = MemoryProviderContract::compatibility_default(
            "sc19050-decode-scope",
            gen_core::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        for capability in &mut contract.strategies {
            match capability.strategy {
                MemoryStrategy::BoundedDecode => {
                    capability.support = gen_core::MemoryStrategySupport::Implemented;
                    capability.parameters.decode_tile_edges = vec![512];
                    capability.parameters.decode_overlaps = vec![64];
                    // Declared, but keyed to a DIFFERENT mode, so no row matches this request and
                    // `decode_geometry_policies_for_request` returns an empty set.
                    capability.parameters.decode_geometry_policies =
                        vec![unmatched_decode_policy()];
                }
                MemoryStrategy::BoundedAttention => {
                    capability.support = gen_core::MemoryStrategySupport::Implemented;
                    capability.parameters.attention_chunk_sizes = vec![256];
                }
                _ => {}
            }
        }

        let tier = gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        let geometry = request(1024, 1024, 1);

        let (decode_candidates, decode_decisions) = floor_parameter_candidates(
            &contract,
            &[MemoryStrategy::Resident, MemoryStrategy::BoundedDecode],
            MemoryStrategy::BoundedDecode,
            tier,
            "text_to_image",
            None,
            geometry,
            false,
        );
        assert!(
            decode_candidates.is_empty(),
            "bounded decode was offered without an admitted tile domain: {decode_candidates:?}"
        );
        assert!(
            decode_decisions.iter().any(|decision| matches!(
                decision,
                DecodeQualityRequestDecision::Unmeasured { .. }
            )),
            "the absence of an exact production-latent row must be REPORTED, not silent: \
             {decode_decisions:?}"
        );

        // A higher independent rung keeps its estimate, with decode omitted so selection-aware
        // engagement cannot resurrect the route-blind tile domain.
        let (attention_candidates, _) = floor_parameter_candidates(
            &contract,
            &[MemoryStrategy::Resident, MemoryStrategy::BoundedAttention],
            MemoryStrategy::BoundedAttention,
            tier,
            "text_to_image",
            None,
            geometry,
            false,
        );
        assert_eq!(attention_candidates.len(), 1);
        assert_eq!(attention_candidates[0].parameters.decode_tile_edge, None);
        assert_eq!(attention_candidates[0].parameters.decode_overlap, None);
        assert_eq!(
            attention_candidates[0].parameters.attention_chunk_size,
            Some(256)
        );
    }

    /// A syntactically complete decode-quality row keyed to a mode this test never requests, so the
    /// scope is DECLARED (non-empty) while no row matches — the state that reaches the
    /// `candidates.is_empty()` arm.
    fn unmatched_decode_policy() -> gen_core::MemoryDecodeGeometryPolicy {
        gen_core::MemoryDecodeGeometryPolicy {
            quality_abi: gen_core::MEMORY_DECODE_QUALITY_ABI,
            family: "sc19050".to_owned(),
            resolved_route: "sc19050-decode-scope".to_owned(),
            backend: gen_core::MemoryBackend::Mlx,
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            load_shape: gen_core::LoadShape::EagerMaterialization,
            artifact: gen_core::MemoryDecodeArtifactIdentity {
                repository: "SceneWorks/sc19050".to_owned(),
                revision: "0".repeat(40),
                variant: "bf16".to_owned(),
                fingerprint: "sc19050-fingerprint".to_owned(),
            },
            implementation_fingerprint: "sc19050-implementation".to_owned(),
            // The axis that makes this row unmatched: the request under test is text-to-image.
            mode: gen_core::MemoryMode::Edit,
            overlay: None,
            geometry: request(1024, 1024, 1),
            use_pid: false,
            tile_edge: 512,
            overlap: 64,
            metric: "sc19050".to_owned(),
            maximum_error: 0,
            fixtures: Vec::new(),
            production_evidence_sha256: "sc19050-evidence".to_owned(),
            disposition: gen_core::MemoryDecodeQualityDisposition::Admitted,
        }
    }

    /// A contract with one Implemented optimized rung and a calibration identity, so a fitted basis
    /// can actually bind. `BoundedAttention` rather than `BoundedDecode`: the attention rung takes
    /// no decode-geometry policy, so the basis is not additionally gated by the tile-domain scope
    /// the test above covers.
    fn lane_fixture_contract() -> MemoryProviderContract {
        let mut contract = MemoryProviderContract::compatibility_default(
            "sc19056-lane",
            gen_core::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        for capability in &mut contract.strategies {
            if capability.strategy == MemoryStrategy::BoundedAttention {
                capability.support = gen_core::MemoryStrategySupport::Implemented;
                capability.parameters.attention_chunk_sizes = vec![256];
            }
        }
        contract.calibration = Some(gen_core::MemoryCalibrationIdentity::new(
            "sc19056-lane-fingerprint",
            contract.load_shape,
        ));
        contract
    }

    /// A measured `BoundedAttention` cell at 1024², tagged with `lane`. Every phase peak is chosen
    /// so the binding phase is DECODE both at the measured cell and after the 4x extrapolation
    /// (conditioning is regressor-flat), which keeps
    /// `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` satisfied — this test is about the lane
    /// conjunct, so no other guard may be the thing that fires.
    fn lane_basis(
        contract: &MemoryProviderContract,
        lane: gen_core::MemoryBackend,
    ) -> Vec<MeasuredRungBasis> {
        let engaged = contract.engaged_composition(MemoryStrategy::BoundedAttention);
        let parameters =
            floor_smallest_parameters(contract, &engaged).expect("the fixture rung has parameters");
        let identity = contract.calibration.as_ref().expect("fixture calibration");
        vec![MeasuredRungBasis {
            lane,
            rung: strategy_rung(MemoryStrategy::BoundedAttention),
            parameters,
            engaged_composition: engaged,
            load_shape: contract.load_shape,
            calibration_abi: identity.abi,
            calibration_fingerprint: identity.fingerprint.clone(),
            geometry: geometry(1024, 1024, 1),
            conditioning_peak_bytes: 1 << 30,
            denoise_peak_bytes: 2 << 30,
            decode_peak_bytes: 3 << 30,
            envelope_peak_bytes: 7 << 30,
            record_id: "sc19056-lane-basis".to_owned(),
        }]
    }

    fn lane_ladder(
        contract: &MemoryProviderContract,
        lane: EstimateLane,
        bases: &[MeasuredRungBasis],
    ) -> SynthesizedEstimateLadder {
        synthesize_estimate_ladder(
            &EstimateRequest {
                lane,
                contract,
                tier: gen_core::MemoryNumericTier {
                    precision: gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                mode_key: "text_to_image",
                overlay: None,
                // 2048²: voxel scale 4.0 against the 1024² basis, so a fitted candidate is
                // numerically distinguishable from the floor rather than coincidentally equal.
                geometry: request(2048, 2048, 1),
                use_pid: false,
                calibration_fingerprint: Some("sc19056-lane-fingerprint"),
                headroom_bytes: 1 << 30,
                synthesize_resident: false,
            },
            bases,
        )
    }

    /// **Epic 19048 R4 / sc-19056: the generic mechanism refuses a foreign lane's measured basis.**
    ///
    /// One basis, two requesting lanes, opposite outcomes — and the assertion is on the BASIS KIND
    /// and the PEAK, not on presence. The MLX request extrapolates the MLX cell (7 GiB envelope x
    /// voxel scale 4 = 28 GiB, `EstimateFittedCurve`); the candle request, handed the identical
    /// bytes, refuses them and falls to its own weights+headroom floor (`EstimateFloor`) at a
    /// different number. A reader that "fell back to the number anyway" would produce the fitted
    /// peak on both lanes and fail here.
    ///
    /// The `_` in the middle is the control that makes this attributable: swapping ONLY the basis
    /// tag flips the verdict, so the refusal is the lane conjunct and not the lane parameter set
    /// (margins, posture, headroom) that also differs between `MLX_LANE` and `CANDLE_LANE`.
    #[test]
    fn a_foreign_lane_basis_is_refused_rather_than_extrapolated() {
        let contract = lane_fixture_contract();
        let mlx_bases = lane_basis(&contract, gen_core::MemoryBackend::Mlx);
        let candle_bases = lane_basis(&contract, gen_core::MemoryBackend::Candle);

        let native = lane_ladder(&contract, MLX_LANE, &mlx_bases);
        assert_eq!(native.estimates.len(), 1);
        assert_eq!(
            native.estimates[0].basis,
            CandidateBasis::EstimateFittedCurve
        );
        let fitted_peak = native.estimates[0].evidence.predicted_peak_bytes;
        assert_eq!(fitted_peak, 28 << 30, "7 GiB envelope at voxel scale 4.0");

        // Same bytes, same identity, same geometry — only the measurement lane differs.
        let foreign = lane_ladder(&contract, MLX_LANE, &candle_bases);
        assert_eq!(foreign.estimates.len(), 1);
        assert_eq!(
            foreign.estimates[0].basis,
            CandidateBasis::EstimateFloor,
            "a candle-measured cell must not seed an MLX extrapolation"
        );
        let floor_peak = foreign.estimates[0].evidence.predicted_peak_bytes;
        assert_ne!(
            floor_peak, fitted_peak,
            "falling closed must reach a DIFFERENT number, not re-derive the foreign one"
        );

        // ...and symmetrically, so the guard is not an MLX-only accident.
        let reverse = lane_ladder(&contract, CANDLE_LANE, &mlx_bases);
        assert_eq!(
            reverse.estimates[0].basis,
            CandidateBasis::EstimateFloor,
            "an MLX-measured cell must not seed a candle extrapolation"
        );
        let reverse_native = lane_ladder(&contract, CANDLE_LANE, &candle_bases);
        assert_eq!(
            reverse_native.estimates[0].basis,
            CandidateBasis::EstimateFittedCurve,
            "the candle lane still consumes its OWN basis — the guard is the lane, not a blanket \
             refusal of fitted candidates on this lane"
        );
    }

    /// [`lane_of`] is a bijection onto the lane constants, not merely total.
    ///
    /// A transposed arm would be invisible to the guard it feeds: every MLX basis would arrive
    /// tagged candle, the R4 conjunct would compare candle to candle, and the refusal would never
    /// fire — a silent hole rather than a failure. Graded against `MLX_LANE`/`CANDLE_LANE`'s own
    /// backends, which [`the_fatal_lane_carries_the_wider_margin`] independently pins to the
    /// gen-core variants, so this cannot be satisfied by transposing both sides together.
    #[test]
    fn the_measurement_lane_and_the_runtime_lane_map_onto_each_other() {
        assert_eq!(lane_of(MeasurementLane::Mlx), MLX_LANE.backend);
        assert_eq!(lane_of(MeasurementLane::Candle), CANDLE_LANE.backend);
        assert_ne!(
            lane_of(MeasurementLane::Mlx),
            lane_of(MeasurementLane::Candle),
            "a collapsing map would make every lane comparison trivially true"
        );
    }

    /// **sc-19054: the extrapolation bound.** A basis farther below the request than
    /// `MAX_EXTRAPOLATION_VOXEL_SCALE` is refused in the basis filter and the rung falls to the
    /// floor arm — the same fall-back shape as the binding-phase-flip refusal, already
    /// mutation-tested on both lanes.
    ///
    /// Mutation coverage: making the cap infinite (or deleting the
    /// `basis_within_extrapolation_bound` conjunct) turns the far-basis arm's `EstimateFloor`
    /// assertion red — the 2560² request would gain a fitted candidate at 6.25× the 1024² basis.
    /// The near-basis control pins that the refusal is DISTANCE, not a blanket fitted shutdown,
    /// and the exact-boundary arm pins that the witnessed 4× ratio itself still serves.
    #[test]
    fn a_basis_beyond_the_extrapolation_bound_falls_to_the_floor_arm() {
        let contract = lane_fixture_contract();
        let bases = lane_basis(&contract, gen_core::MemoryBackend::Mlx);
        let ladder_at = |width: u32, height: u32| {
            synthesize_estimate_ladder(
                &EstimateRequest {
                    lane: MLX_LANE,
                    contract: &contract,
                    tier: gen_core::MemoryNumericTier {
                        precision: gen_core::Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    },
                    mode_key: "text_to_image",
                    overlay: None,
                    geometry: request(width, height, 1),
                    use_pid: false,
                    calibration_fingerprint: Some("sc19056-lane-fingerprint"),
                    headroom_bytes: 1 << 30,
                    synthesize_resident: false,
                },
                &bases,
            )
        };

        // Exactly the witnessed bound: 2048² is 4.0× the 1024² basis — still a seed.
        let at_bound = ladder_at(2048, 2048);
        assert_eq!(at_bound.estimates.len(), 1);
        assert_eq!(
            at_bound.estimates[0].basis,
            CandidateBasis::EstimateFittedCurve,
            "the corpus witnessed 4.0× exactly; the bound must be inclusive"
        );
        assert_eq!(
            at_bound.estimates[0].evidence.predicted_peak_bytes,
            28 << 30,
            "7 GiB envelope at voxel scale 4.0"
        );

        // Beyond it: 2560² is 6.25× — the basis is refused and the rung falls to the floor arm.
        let beyond = ladder_at(2560, 2560);
        assert_eq!(beyond.estimates.len(), 1);
        assert_eq!(
            beyond.estimates[0].basis,
            CandidateBasis::EstimateFloor,
            "a basis beyond MAX_EXTRAPOLATION_VOXEL_SCALE must not seed a fitted estimate; the \
             rung falls through to the weights+headroom floor"
        );
        assert_ne!(
            beyond.estimates[0].evidence.predicted_peak_bytes,
            ((7u64 << 30) as f64 * 6.25).ceil() as u64,
            "falling closed must reach the floor's own number, never re-derive the unbounded \
             extrapolation"
        );

        // The predicate itself, both directions plus the degenerate zero-voxel basis.
        assert!(basis_within_extrapolation_bound(
            geometry(1024, 1024, 1),
            request(2048, 2048, 1)
        ));
        assert!(!basis_within_extrapolation_bound(
            geometry(1024, 1024, 1),
            request(2560, 2560, 1)
        ));
        assert!(
            !basis_within_extrapolation_bound(geometry(0, 1024, 1), request(64, 64, 1)),
            "a zero-voxel basis is never a seed — its clamp-at-1.0 scale would present a \
             meaningless measurement as the request peak"
        );
    }

    /// **sc-19054 (epic 19048 R3): what a declared manifest scalar may claim.** Measured at a
    /// covering geometry ⇒ the peak (monotone bound); everything else — unmeasured, undeclared or
    /// exceeded geometry, or a workload shape (batch/frames) the single-image capture never saw —
    /// ⇒ the declared floor.
    ///
    /// Mutation coverage: dropping the `measured` conjunct flips the unmeasured arm; dropping the
    /// pixel comparison flips the 2048² arm; dropping the batch/frames shape conjuncts flips
    /// their arms; treating a missing/zero declaration as covering flips the last two.
    #[test]
    fn a_declared_scalar_is_a_peak_only_where_its_measurement_covers_the_request() {
        let declared = Some(1024_u64 * 1024);
        let class = |measured, pixels, req| declared_scalar_class(measured, pixels, req);

        assert_eq!(
            class(true, declared, request(1024, 1024, 1)),
            DeclaredScalarClass::MeasuredPeak
        );
        assert_eq!(
            class(true, declared, request(512, 512, 1)),
            DeclaredScalarClass::MeasuredPeak,
            "the measured peak bounds every smaller same-shape request from above"
        );
        assert_eq!(
            class(true, declared, request(2048, 2048, 1)),
            DeclaredScalarClass::DeclaredFloor,
            "a floor captured at 1024² must not be presented as the peak for 4 MP"
        );
        assert_eq!(
            class(false, declared, request(1024, 1024, 1)),
            DeclaredScalarClass::DeclaredFloor,
            "epic 18472's measured=false means the numbers were never captured"
        );
        assert_eq!(
            class(true, declared, request(1024, 1024, 4)),
            DeclaredScalarClass::DeclaredFloor,
            "frames is a workload shape, not a covered geometry"
        );
        assert_eq!(
            class(
                true,
                declared,
                MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 2,
                    frames: 1,
                    reference_count: 0
                }
            ),
            DeclaredScalarClass::DeclaredFloor,
            "batch is a workload shape, not a covered geometry"
        );
        assert_eq!(
            class(true, None, request(64, 64, 1)),
            DeclaredScalarClass::DeclaredFloor,
            "no declared geometry means no covered request"
        );
        assert_eq!(
            class(true, Some(0), request(64, 64, 1)),
            DeclaredScalarClass::DeclaredFloor,
            "a zero declaration covers nothing"
        );
    }

    /// **sc-19054: the resident rung synthesizes fitted-only, and only on request.** The lane
    /// whose resident baseline is a declared scalar (candle) opts in and gets a fitted resident
    /// candidate from a resident-rung basis; a lane that keeps `false` (MLX) gets none, and even
    /// the opted-in lane never receives a resident FLOOR from the ladder — the lane's own resident
    /// baseline is that floor.
    ///
    /// Mutation coverage: unconditionally skipping resident (reverting the sc-19054 arm) turns the
    /// opted-in arm red; unconditionally synthesizing it turns the opted-out arm red; letting
    /// resident reach the floor arm turns the refused-basis arm red (it would emit an
    /// `EstimateFloor` for resident).
    #[test]
    fn the_resident_rung_synthesizes_fitted_only_and_only_on_request() {
        let mut contract = lane_fixture_contract();
        // Make resident the ONLY implemented rung so every emitted candidate is attributable.
        for capability in &mut contract.strategies {
            capability.support = if capability.strategy == MemoryStrategy::Resident {
                gen_core::MemoryStrategySupport::Implemented
            } else {
                gen_core::MemoryStrategySupport::Missing
            };
        }
        let identity = contract.calibration.as_ref().expect("fixture calibration");
        let resident_basis = MeasuredRungBasis {
            lane: gen_core::MemoryBackend::Candle,
            rung: StrategyRung::Resident,
            parameters: Default::default(),
            engaged_composition: contract.engaged_composition(MemoryStrategy::Resident),
            load_shape: contract.load_shape,
            calibration_abi: identity.abi,
            calibration_fingerprint: identity.fingerprint.clone(),
            geometry: geometry(1024, 1024, 1),
            conditioning_peak_bytes: 1 << 30,
            denoise_peak_bytes: 2 << 30,
            decode_peak_bytes: 3 << 30,
            envelope_peak_bytes: 5 << 30,
            record_id: "sc19054-resident-basis".to_owned(),
        };
        let ladder = |synthesize_resident: bool, bases: &[MeasuredRungBasis]| {
            synthesize_estimate_ladder(
                &EstimateRequest {
                    lane: CANDLE_LANE,
                    contract: &contract,
                    tier: gen_core::MemoryNumericTier {
                        precision: gen_core::Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    },
                    mode_key: "text_to_image",
                    overlay: None,
                    geometry: request(2048, 2048, 1),
                    use_pid: false,
                    calibration_fingerprint: Some("sc19056-lane-fingerprint"),
                    headroom_bytes: 1 << 30,
                    synthesize_resident,
                },
                bases,
            )
        };

        // Opted in, basis present: one fitted resident candidate at the scaled envelope.
        let fitted = ladder(true, std::slice::from_ref(&resident_basis));
        assert_eq!(fitted.estimates.len(), 1);
        assert_eq!(
            fitted.estimates[0].selection.strategy,
            MemoryStrategy::Resident
        );
        assert_eq!(
            fitted.estimates[0].basis,
            CandidateBasis::EstimateFittedCurve
        );
        assert_eq!(
            fitted.estimates[0].evidence.predicted_peak_bytes,
            20 << 30,
            "5 GiB resident envelope at voxel scale 4.0"
        );

        // Opted out: the identical basis synthesizes nothing (the MLX posture).
        assert!(
            ladder(false, std::slice::from_ref(&resident_basis))
                .estimates
                .is_empty(),
            "a lane whose resident baseline carries its own geometry law gets no fitted sibling"
        );

        // Opted in, no usable basis: nothing — never a ladder-made resident floor.
        assert!(
            ladder(true, &[]).estimates.is_empty(),
            "the resident rung must not reach the floor arm; the lane's own baseline is the floor"
        );
    }

    /// The two lanes' postures and margins travel together: the fatal-abort lane is never the
    /// narrower one. `ladder_margin_policy`'s compile-time block pins the margin ordering; this
    /// pins that the POSTURE explaining it is attached to the right lane.
    #[test]
    fn the_fatal_lane_carries_the_wider_margin() {
        assert_eq!(MLX_LANE.failure_posture, FailurePosture::FatalProcessAbort);
        assert_eq!(
            CANDLE_LANE.failure_posture,
            FailurePosture::RecoverableError
        );
        assert!(MLX_LANE.estimate_margin() > CANDLE_LANE.estimate_margin());
        assert_eq!(MLX_LANE.backend, gen_core::MemoryBackend::Mlx);
        assert_eq!(CANDLE_LANE.backend, gen_core::MemoryBackend::Candle);
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────
    // Fitted per-phase curves (sc-19058's fold)
    // ─────────────────────────────────────────────────────────────────────────────────────────
    //
    // `vram_gate`'s own suite still grades this law end to end over the shipped Krea manifest, and
    // that suite is the byte-identity evidence. What it CANNOT show is that the law stopped being
    // candle-shaped, because every call it makes passes the candle lane. These tests run on the
    // ordinary `cargo test -p sceneworks-worker --lib` lane — where `vram_gate` is not even
    // compiled — and each one is written so that re-hardcoding the lane, or dropping a conjunct,
    // turns it red.

    fn curve(value: serde_json::Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    /// **The lane is a PARAMETER, not the candle constant it was before the fold.**
    ///
    /// The discriminating shape is one container graded by both readers: an assertion that only
    /// ever passes `Candle` would keep passing if the comparison were re-hardcoded to
    /// `MeasurementLane::Candle`, which is precisely the failure this epic has already seen once.
    /// Here the same MLX-tagged block ADMITS for the MLX reader and REFUSES for the candle one, and
    /// the candle-tagged block does the mirror image, so no constant can satisfy all four rows.
    #[test]
    fn the_container_lane_guard_answers_per_reader_not_per_constant() {
        for measured_on in [MeasurementLane::Mlx, MeasurementLane::Candle] {
            let block = serde_json::json!({ "measurementLane": measured_on.as_key() });
            for reader in [MeasurementLane::Mlx, MeasurementLane::Candle] {
                assert_eq!(
                    container_measurement_lane(&block, reader),
                    (reader == measured_on).then_some(measured_on),
                    "a {} block read by the {} reader",
                    measured_on.as_key(),
                    reader.as_key()
                );
            }
        }
        // Missing, unrecognized and non-string all refuse for EVERY reader: there is no reader a
        // untagged container is evidence for.
        for absent_or_bad in [
            serde_json::json!({}),
            serde_json::json!({ "measurementLane": "cuda" }),
            serde_json::json!({ "measurementLane": "Candle" }),
            serde_json::json!({ "measurementLane": true }),
            serde_json::json!({ "measurementLane": serde_json::Value::Null }),
        ] {
            for reader in [MeasurementLane::Mlx, MeasurementLane::Candle] {
                assert_eq!(
                    container_measurement_lane(&absent_or_bad, reader),
                    None,
                    "{absent_or_bad} must not read as {} evidence",
                    reader.as_key()
                );
            }
        }
    }

    /// The per-curve lane LEAF is the same parametric question, and absence still inherits.
    ///
    /// Graded with coefficients that make the admitting answer a specific number rather than merely
    /// `Some`, so a reader that admitted everything at zero could not pass.
    #[test]
    fn the_curve_lane_leaf_answers_per_reader_and_absence_inherits() {
        let geometry = CurveGeometry {
            pixels: 1_000_000,
            frames: 1,
        };
        let untagged = curve(serde_json::json!({ "fixedGb": 1.0, "perMpxGb": 2.0 }));
        for reader in [MeasurementLane::Mlx, MeasurementLane::Candle] {
            assert_eq!(
                fitted_phase_curve_gb(&untagged, geometry, reader),
                Some(3.0),
                "an untagged curve inherits the container tag its caller already checked"
            );
        }
        for measured_on in [MeasurementLane::Mlx, MeasurementLane::Candle] {
            let tagged = curve(serde_json::json!({
                "fixedGb": 1.0,
                "perMpxGb": 2.0,
                "measurementLane": measured_on.as_key(),
            }));
            for reader in [MeasurementLane::Mlx, MeasurementLane::Candle] {
                assert_eq!(
                    fitted_phase_curve_gb(&tagged, geometry, reader),
                    (reader == measured_on).then_some(3.0),
                    "a {} curve read by {}",
                    measured_on.as_key(),
                    reader.as_key()
                );
            }
        }
    }

    /// The affine form and its ASSOCIATION, graded on the mechanism's own lane-agnostic surface.
    ///
    /// `vram_gate`'s `committed_image_curves_evaluate_bit_identically_to_the_two_coefficient_form`
    /// pins this over the 36 shipped candle curves. This pins the same association where the law now
    /// lives, and on a lane that never measured a Krea curve — because after the fold the MLX lane
    /// can reach this function too, and it must get the identical f64.
    ///
    /// The coefficients and geometries are not decorative and are not arbitrary. `8.58` is the
    /// shipped q4 `threeStage.denoise` slope and `0.2998482076533136` is the sc-18810 LTX `cross`
    /// temporal coefficient, chosen because those two, at these two areas, are cells where
    /// `c * px as f64 / 1e6` and `c * (px as f64 / 1e6)` are DIFFERENT f64s. Most cells are not:
    /// the first draft of this test used the sc-18810 AREA coefficient at 1280x704 and stayed green
    /// under a deliberate re-association, which is a test that proves nothing. So the separation is
    /// asserted first, and only then is the reader graded against it — a future edit that picks
    /// "nicer" numbers goes red on the premise rather than quietly losing the pin.
    #[test]
    fn the_fitted_form_keeps_its_association_on_either_lane() {
        const FIXED: f64 = 2.5151564578656265;
        const PER_MPX: f64 = 8.58;
        const PER_MPX_FRAME: f64 = 0.2998482076533136;
        let coefficients = curve(serde_json::json!({
            "fixedGb": FIXED,
            "perMpxGb": PER_MPX,
            "perMpxFrameGb": PER_MPX_FRAME,
        }));
        // The production association, spelled out: multiply first, divide second.
        let expected = |pixels: u64, frames: u32| {
            FIXED
                + PER_MPX * pixels as f64 / 1_000_000.0
                + PER_MPX_FRAME * pixels as f64 / 1_000_000.0 * f64::from(frames)
        };
        // The two rewrites a "harmless tidy-up" reaches for: factor the megapixel conversion out of
        // the area term, and out of the temporal term.
        let re_associated_area = |pixels: u64, frames: u32| {
            FIXED
                + PER_MPX * (pixels as f64 / 1_000_000.0)
                + PER_MPX_FRAME * pixels as f64 / 1_000_000.0 * f64::from(frames)
        };
        let re_associated_temporal = |pixels: u64, frames: u32| {
            FIXED
                + PER_MPX * pixels as f64 / 1_000_000.0
                + PER_MPX_FRAME * (pixels as f64 / 1_000_000.0) * f64::from(frames)
        };

        let cells = [(901_120_u64, 1_u32), (901_120, 121), (1024 * 1024, 241)];
        assert!(
            cells
                .iter()
                .any(|&(px, f)| expected(px, f).to_bits() != re_associated_area(px, f).to_bits()),
            "PREMISE: no graded cell separates the area association, so this test cannot see one move"
        );
        assert!(
            cells.iter().any(
                |&(px, f)| expected(px, f).to_bits() != re_associated_temporal(px, f).to_bits()
            ),
            "PREMISE: no graded cell separates the temporal association"
        );

        for (pixels, frames) in cells {
            for reader in [MeasurementLane::Mlx, MeasurementLane::Candle] {
                let actual =
                    fitted_phase_curve_gb(&coefficients, CurveGeometry { pixels, frames }, reader)
                        .expect("an untagged curve evaluates for its caller's lane");
                assert_eq!(
                    actual.to_bits(),
                    expected(pixels, frames).to_bits(),
                    "{pixels}px x {frames}f on {}: the association moved",
                    reader.as_key()
                );
            }
        }
    }

    /// The area hull fails CLOSED on a container that declares no bound, rather than admitting
    /// everything. An unbounded curve is not a curve this mechanism extrapolates.
    #[test]
    fn an_undeclared_area_hull_refuses_rather_than_admitting_everything() {
        assert_eq!(
            area_within_measured_hull(&serde_json::json!({}), 1),
            None,
            "no declared bound is not an infinite bound"
        );
        assert_eq!(
            area_within_measured_hull(&serde_json::json!({ "maxMeasuredPixels": "1048576" }), 1),
            None,
            "a bound this reader cannot read is not a bound it may guess at"
        );
        let bounded = serde_json::json!({ "maxMeasuredPixels": 1024 * 1024 });
        assert_eq!(area_within_measured_hull(&bounded, 1024 * 1024), Some(true));
        assert_eq!(
            area_within_measured_hull(&bounded, 1024 * 1024 + 1),
            Some(false)
        );
    }

    /// The temporal hull is a VOXEL surface: one frame count, two areas, opposite verdicts. A
    /// scalar frame bound could not tell the two apart, and an absent bound reads as ONE frame.
    #[test]
    fn the_temporal_hull_is_a_voxel_surface_and_absence_means_one_frame() {
        let unbounded = serde_json::json!({});
        let at = |fit: &serde_json::Value, pixels: u64, frames: u32| {
            voxels_within_measured_hull(fit, CurveGeometry { pixels, frames })
        };
        assert_eq!(at(&unbounded, 1024 * 1024, 1), Some(true));
        assert_eq!(
            at(&unbounded, 1024 * 1024, 2),
            Some(false),
            "an image fit never measured a second output frame"
        );

        let bounded = serde_json::json!({ "maxMeasuredVoxels": 1024 * 1024 * 100_u64 });
        assert_eq!(at(&bounded, 1024 * 1024, 100), Some(true), "at the bound");
        assert_eq!(at(&bounded, 1024 * 1024, 101), Some(false), "past it");
        // The discriminating pair.
        assert_eq!(at(&bounded, 512 * 512, 400), Some(true));
        assert_eq!(at(&bounded, 1024 * 1024, 400), Some(false));

        assert_eq!(
            at(&serde_json::json!({ "maxMeasuredVoxels": 0 }), 1, 1),
            None,
            "a zero bound is unreadable, not a bound that admits nothing quietly"
        );
        assert_eq!(
            at(&unbounded, u64::MAX, 2),
            None,
            "an overflowing geometry refuses instead of wrapping into the hull"
        );
    }

    /// A rung is three curves or it is not a fit. The triple also carries the lane down to every
    /// leaf, so one foreign phase disqualifies the whole rung rather than being skipped.
    #[test]
    fn a_rung_missing_or_mistagging_one_phase_predicts_nothing() {
        let geometry = CurveGeometry {
            pixels: 1_000_000,
            frames: 1,
        };
        let keys = ["text", "denoise", "decode"];
        let full = curve(serde_json::json!({
            "text": { "fixedGb": 1.0, "perMpxGb": 0.0 },
            "denoise": { "fixedGb": 2.0, "perMpxGb": 0.0 },
            "decode": { "fixedGb": 3.0, "perMpxGb": 0.0 },
        }));
        assert_eq!(
            fitted_phase_triple(&full, keys, geometry, MeasurementLane::Candle),
            Some([1.0, 2.0, 3.0]),
            "the triple is returned in BindingPhase order, not manifest key order"
        );
        for dropped in keys {
            let mut partial = full.clone();
            partial.remove(dropped);
            assert_eq!(
                fitted_phase_triple(&partial, keys, geometry, MeasurementLane::Candle),
                None,
                "a rung without its {dropped} curve is an incomplete fit"
            );
        }
        for foreign in keys {
            let mut mixed = full.clone();
            mixed[foreign]["measurementLane"] = serde_json::json!("mlx");
            assert_eq!(
                fitted_phase_triple(&mixed, keys, geometry, MeasurementLane::Candle),
                None,
                "one MLX-measured {foreign} curve disqualifies the whole candle rung"
            );
        }
    }
}
