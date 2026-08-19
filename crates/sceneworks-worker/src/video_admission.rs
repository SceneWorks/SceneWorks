//! The worker half of the video memory gate (sc-18814, epic 18803).
//!
//! `sceneworks_core::video_request` owns the video admission POLICY — which family is routed on
//! which lane, which geometries have to be graded, what a refusal says. It cannot own the
//! SELECTION, because `sceneworks-core` deliberately carries no gen-core dependency and the
//! ladder selector (`crate::memory_strategy::select_strategy`) is gen-core-typed. This module is
//! the bridge: it implements core's [`VideoStrategySelector`] seam by building candidates from
//! the loaded provider's own [`MemoryProviderContract`] and calling that one shared selector.
//!
//! **Epic decision 3 stands** (sc-18814, reaffirmed at activity-19060): the video gate is
//! `video_request.rs`, not a unified `mlx_fit_gate`. Ordering, first-fit and margin grading are
//! not re-implemented here — `select_strategy` does all three, exactly as it does for
//! `mlx_fit_gate` (the MLX image lane) and `candle_memory_strategy` (the candle image lane).
//!
//! **Prediction is exact-or-floor.** A candidate whose full catalog/provider/lane/tier/mode/rung/
//! load-shape/ABI/fingerprint/closure/decode-regime identity matches a packaged fitted curve uses
//! its three residual-bounded affine cross laws
//! (`fixed + perMpx*mpx + perMpxFrame*mpx*frames + maxResidual`). Every curve mismatch —
//! including geometry outside the measured area-by-voxel hull — falls back to the established
//! [`crate::estimate_synthesis::floor_weights_bytes`] plus activation-headroom lower bound,
//! strengthened when the selected provider exports a larger decode working-set profile. Requested
//! rows key `gen_core::MemoryGeometry` to the real clip length; synthetic cap rows key it to the cap
//! they evaluate. The eventual provider run context still receives the actual request geometry,
//! and the image lane remains untouched at one frame.

use gen_core::tiling::VideoDecodeMemoryProfile;
use gen_core::{
    MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryNumericTier, MemoryProviderContract,
    MemoryRunContext, MemorySelection, MemoryStrategy, OffloadPolicy,
};
use sceneworks_core::memory_calibration::StrategyRung;
use sceneworks_core::video_memory_curves::{
    VideoCurveBackend, VideoCurveDecodePass, VideoCurveGeometry, VideoCurveLoadShape,
    VideoCurveQuery, VideoMemoryCurveBundle,
};
use sceneworks_core::video_request::{
    VideoAdmissionGeometry, VideoDecodePass, VideoLane, VideoRungSelection, VideoStrategySelector,
};

use crate::memory_strategy::{Budget, Candidate, CandidateBasis, RequestScope, Selection};

type VideoDecodeProfileResolver = fn(
    VideoLane,
    &str,
    VideoAdmissionGeometry,
    MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String>;

#[derive(Clone, Copy, Debug)]
struct ResolvedVideoDecodeProfile {
    profile: VideoDecodeMemoryProfile,
    evidence_revision: &'static str,
}

fn no_video_decode_profile(
    _lane: VideoLane,
    _provider_id: &str,
    _geometry: VideoAdmissionGeometry,
    _selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    Ok(None)
}

/// Resolve the exact provider-owned decode working set for the candidate being graded.
///
/// The selected MLX Wan rung-2 carrier has a narrower profile derived from the same provider planner
/// that executes the request. Every other supported candidate uses the provider's conservative
/// single-pass profile. A runtime bundle that exposes no profile returns `None`, preserving the
/// historical weights-plus-headroom floor; provider validation errors fail closed instead of being
/// rewritten as an unprofiled estimate.
fn packaged_video_decode_profile(
    lane: VideoLane,
    provider_id: &str,
    geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
) -> Result<Option<ResolvedVideoDecodeProfile>, String> {
    let frames = geometry.estimate_frames().max(1);
    #[cfg(target_os = "macos")]
    if lane == VideoLane::Mlx {
        if selection.strategy == MemoryStrategy::BoundedDecode {
            if let (Some(tile_edge), Some(overlap)) = (
                selection.parameters.decode_tile_edge,
                selection.parameters.decode_overlap,
            ) {
                let selected = runtime_macos::selected_video_decode_memory_profile(
                    provider_id,
                    geometry.width,
                    geometry.height,
                    frames,
                    tile_edge,
                    overlap,
                )
                .map_err(|error| {
                    format!(
                        "{provider_id}: selected video decode profile rejected the admitted carrier: {error}"
                    )
                })?;
                if let Some(profile) = selected {
                    return Ok(Some(ResolvedVideoDecodeProfile {
                        profile,
                        evidence_revision: "video-provider-selected-decode-profile-v1",
                    }));
                }
            }
            // No provider-selected profile means the runtime bundle has not exported a
            // load-bearing working set for this exact carrier. Do not apply the conservative
            // single-pass profile to a bounded-decode candidate: that would erase the very saving
            // the rung selects. The unchanged generic floor remains the honest fallback.
            return Ok(None);
        }
        return Ok(runtime_macos::conservative_video_decode_memory_profile(
            provider_id,
            geometry.width,
            geometry.height,
            frames,
        )
        .map(|profile| ResolvedVideoDecodeProfile {
            profile,
            evidence_revision: "video-provider-conservative-decode-profile-v1",
        }));
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    if lane == VideoLane::Candle {
        if selection.strategy == MemoryStrategy::BoundedDecode {
            return Ok(None);
        }
        return Ok(runtime_cuda::conservative_video_decode_memory_profile(
            provider_id,
            geometry.width,
            geometry.height,
            frames,
        )
        .map(|profile| ResolvedVideoDecodeProfile {
            profile,
            evidence_revision: "video-provider-conservative-decode-profile-v1",
        }));
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
    let _ = (lane, provider_id, selection, frames);
    Ok(None)
}

/// A video request's identity for the selector, minus the geometry (which arrives per-call).
pub(crate) struct VideoRequestIdentity<'a> {
    /// Catalog model id. Kept distinct from `route`: two catalog entries may share one engine but
    /// do not thereby share measured artifact/overlay memory (LTX base vs Eros is the live case).
    pub(crate) model_id: &'a str,
    /// Catalog family, kept separate from the provider descriptor's implementation family. A
    /// custom/imported LTX-family entry cannot inherit the built-in base model's measurements.
    pub(crate) model_family: &'a str,
    /// The resolved engine id — the same `resolved_route` spelling every other admission cell uses.
    pub(crate) route: &'a str,
    /// Calibration mode. A text-to-video curve must not silently price an image/keyframe-conditioned
    /// request whose encoder/residency surface was not measured.
    pub(crate) mode: &'a str,
    /// Exact reference count and overlay identity. The only promoted curve currently covers zero
    /// references and no overlay, but these still travel through evidence identity so a future
    /// calibrated surface cannot accidentally inherit the base T2V cell.
    pub(crate) reference_count: u32,
    pub(crate) overlay: Option<&'a str>,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
    /// The contract's live calibration ABI. Carried separately from the optional calibration
    /// identity so an ABI mismatch fails the fitted curve even if a malformed/legacy identity was
    /// minted with a misleading fingerprint.
    pub(crate) calibration_abi: u32,
    /// The live compile-closure digest of the provider being admitted (sc-17774). Both sides carry
    /// the same value on a route with no measured cell, which states plainly that there is no
    /// measured closure to be current against.
    pub(crate) expected_closure_digest: &'a str,
}

/// Core's [`VideoStrategySelector`] seam, answered by the shared ladder selector.
pub(crate) struct LadderVideoSelector<'a> {
    identity: VideoRequestIdentity<'a>,
    contract: &'a MemoryProviderContract,
    budget: Option<Budget>,
    /// The activation-headroom allowance the caller already charges this load. Supplied, never
    /// derived here — see the module doc.
    headroom_bytes: u64,
    /// The shared backend-neutral fitted-curve container. `None` is a normal fail-open state: every
    /// rung retains its pre-existing weights-plus-headroom floor candidate.
    curves: Option<&'a VideoMemoryCurveBundle>,
    /// Backend bundle resolver for the provider's load-bearing decode working set. Tests default to
    /// `None` so focused curve/floor fixtures do not accidentally inherit a real provider profile.
    decode_profile: VideoDecodeProfileResolver,
    /// Provider-resident bytes captured as the conservative committed delta around the exact cold
    /// load. Fitted/floor laws model the complete run peak, while the post-load budget is
    /// incremental, so every estimate candidate is reduced by this fixed attribution exactly once.
    attributable_resident_bytes: u64,
    /// A provider claiming more attributable resident bytes than a candidate's complete peak is
    /// an accounting contradiction, not a zero-byte request. Remember it so the admission funnel
    /// can fail closed after core returns through its non-error selector seam.
    accounting_error: Option<String>,
    profile_error: std::cell::RefCell<Option<String>>,
    /// Every `(geometry, selection)` the selector chose, so the caller can recover the selected
    /// PARAMETERS for the geometry core reports as binding. Core's seam returns a rung; the
    /// per-request knobs need the whole `MemorySelection`, and re-deriving it would be a second
    /// selection.
    selections: Vec<VideoSelectedCandidate>,
    /// Unwidened resident candidate for every geometry core graded. The refusal guard consumes the
    /// exact binding geometry's value, including any provider profile, instead of recomputing a
    /// profile-blind weights floor after selection.
    resident_floors: Vec<(VideoAdmissionGeometry, u64)>,
}

#[derive(Clone, Debug)]
struct VideoSelectedCandidate {
    binding_geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
    /// Raw fitted/floor peak before the shared estimate margin. This is the exact demand basis the
    /// provider lifecycle context must receive; `Selection::needed_gb` is already widened.
    predicted_peak_bytes: u64,
    evidence_revision: String,
}

impl<'a> LadderVideoSelector<'a> {
    #[cfg(test)]
    pub(crate) fn new(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
    ) -> Self {
        Self::with_curve_bundle(
            identity,
            contract,
            budget,
            headroom_bytes,
            attributable_resident_bytes,
            sceneworks_core::video_memory_curves::packaged_video_memory_curves(),
        )
    }

    fn with_curve_bundle(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
        curves: Option<&'a VideoMemoryCurveBundle>,
    ) -> Self {
        Self {
            identity,
            contract,
            budget,
            headroom_bytes,
            curves,
            decode_profile: no_video_decode_profile,
            attributable_resident_bytes,
            accounting_error: None,
            profile_error: std::cell::RefCell::new(None),
            selections: Vec::new(),
            resident_floors: Vec::new(),
        }
    }

    fn with_profiles(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
        attributable_resident_bytes: u64,
        curves: Option<&'a VideoMemoryCurveBundle>,
        decode_profile: VideoDecodeProfileResolver,
    ) -> Self {
        let mut selector = Self::with_curve_bundle(
            identity,
            contract,
            budget,
            headroom_bytes,
            attributable_resident_bytes,
            curves,
        );
        selector.decode_profile = decode_profile;
        selector
    }

    /// The estimate lane this video request synthesizes under (sc-19050).
    ///
    /// Exhaustive on [`VideoLane`] so a new lane cannot compile without choosing one — the same
    /// posture `memory_strategy::stale_measured_margin` takes on [`MemoryBackend`]. Returning the
    /// mechanism's own [`crate::estimate_synthesis::EstimateLane`] rather than a bare
    /// [`MemoryBackend`] is what makes the video gate a genuine two-lane consumer of the shared
    /// parameters: the backend identity, the derived estimate margin, and the failure posture that
    /// explains it all travel together instead of being re-chosen per call site.
    pub(crate) const fn lane(&self) -> crate::estimate_synthesis::EstimateLane {
        match self.identity.lane {
            VideoLane::Mlx => crate::estimate_synthesis::MLX_LANE,
            VideoLane::Candle => crate::estimate_synthesis::CANDLE_LANE,
        }
    }
}

/// Which phase carries a rung's peak at one geometry.
///
/// A property of the GEOMETRY, not of the model: it is measured to flip inside a single model's
/// envelope (text binds at 11,904 latent tokens, decode at 14,080 — sc-18812). It is therefore
/// derived on every call and never cached, and no code here may assume a model has "a" binding
/// phase. This is the MLX-side answer to the question `KreaTurboPhasePeaks::binding_phase()`
/// answers on the candle side.
///
/// sc-19050: this IS [`crate::estimate_synthesis::BindingPhase`], re-exported under the name this
/// module's callers already use. It was a hand-copied enum with a hand-copied argmax, documented as
/// "mirroring" the MLX gate's — a promise three copies cannot keep. There is now one type, one
/// rule, and no relabelling `match` for a later edit to get wrong.
pub(crate) use crate::estimate_synthesis::BindingPhase as VideoBindingPhase;

/// The per-phase peaks one rung's prediction is made of.
///
/// **Deliberately not reduced to a scalar until the last possible moment.** sc-18810 measured every
/// candidate temporal form missing the AGGREGATE peak by at least 10.26 GiB — about 94x the
/// replicate noise floor — while the same forms land at 0.019–0.44 GiB *per phase*. A prediction
/// that is accurate at all is therefore phase-resolved, and the admission number is the **max over
/// phases**, exactly the discipline `KreaTurboPhasePeaks::peak_gb` already applies on the candle
/// side.
///
/// The fitted laws are **affine cross curves**
/// (`fixedGb + perMpxGb*mpx + perMpxFrameGb*mpx*frames + maxResidualGb`) with large,
/// phase-specific intercepts and conservative per-phase residual floors. A through-origin scalar
/// ratio like `mlx_fit_gate`'s `scaled(bytes) = bytes * scale` cannot express them, which is why
/// this seam keeps the three values apart rather than handing one number across. On an inapplicable
/// curve all three values carry the same historical weights+headroom floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhasePeaks {
    pub(crate) conditioning_bytes: u64,
    pub(crate) denoise_bytes: u64,
    pub(crate) decode_bytes: u64,
}

impl PhasePeaks {
    /// The admission peak — the max over phases, taken at the single point a scalar is
    /// unavoidable (gen-core's `MemoryEvidence::predicted_peak_bytes` is one number).
    pub(crate) const fn peak_bytes(self) -> u64 {
        let mut peak = self.conditioning_bytes;
        if self.denoise_bytes > peak {
            peak = self.denoise_bytes;
        }
        if self.decode_bytes > peak {
            peak = self.decode_bytes;
        }
        peak
    }

    /// Which phase binds AT THIS GEOMETRY, with ties to the LATER phase.
    ///
    /// sc-19050: this used to be a hand-copied mirror of `mlx_fit_gate::binding_phase`, documented
    /// as "so the two never disagree" — a promise three copies cannot keep. The rule now has one
    /// implementation ([`crate::estimate_synthesis::binding_phase`]) and this method only relabels
    /// its answer in the video lane's vocabulary.
    pub(crate) fn binding_phase(self) -> VideoBindingPhase {
        crate::estimate_synthesis::binding_phase(
            self.conditioning_bytes,
            self.denoise_bytes,
            self.decode_bytes,
        )
    }
}

/// The weights+headroom floor for one engaged composition, expressed per phase.
///
/// The fallback scalar is exactly
/// `mlx_fit_gate::estimate_floor_weights_bytes(contract, engaged) + headroom_bytes`, the same
/// number the cold-load residency gate already charges this load. It is phase-blind, so all three
/// phases carry it and [`PhasePeaks::peak_bytes`] returns that scalar unchanged.
fn floor_phase_peaks(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    headroom_bytes: u64,
) -> PhasePeaks {
    let floor = crate::estimate_synthesis::floor_weights_bytes(contract, engaged)
        .saturating_add(headroom_bytes);
    PhasePeaks {
        conditioning_bytes: floor,
        denoise_bytes: floor,
        decode_bytes: floor,
    }
}

/// The historical activation floor, strengthened by the provider's own decode working set when the
/// selected runtime bundle exposes one. The generic allowance remains a lower bound: a decode-only
/// profile cannot prove that conditioning or denoise need less activation memory. Conversely, a
/// profile above that allowance is load-bearing and may create a real geometry-dependent refusal.
fn profiled_floor_phase_peaks(
    selector: &LadderVideoSelector<'_>,
    geometry: VideoAdmissionGeometry,
    selection: MemorySelection,
    engaged: &[MemoryStrategy],
) -> (PhasePeaks, Option<&'static str>) {
    let generic = floor_phase_peaks(selector.contract, engaged, selector.headroom_bytes);
    let resolved = match (selector.decode_profile)(
        selector.identity.lane,
        &selector.contract.provider_id,
        geometry,
        selection,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            *selector.profile_error.borrow_mut() = Some(error);
            return (generic, None);
        }
    };
    let Some(resolved) = resolved else {
        return (generic, None);
    };
    let weights = crate::estimate_synthesis::floor_weights_bytes(selector.contract, engaged);
    let Some(profiled) = resolved
        .profile
        .checked_composed_peak(weights, selector.contract.asset_facts.decoder_bytes)
    else {
        *selector.profile_error.borrow_mut() = Some(format!(
            "{} decode profile cannot compose contract weights {} with decoder bytes {}; refusing inconsistent provider accounting",
            selector.contract.provider_id,
            weights,
            selector.contract.asset_facts.decoder_bytes,
        ));
        return (generic, None);
    };
    let floor = generic.peak_bytes().max(profiled);
    (
        PhasePeaks {
            conditioning_bytes: floor,
            denoise_bytes: floor,
            decode_bytes: floor,
        },
        Some(resolved.evidence_revision),
    )
}

fn curve_backend(lane: VideoLane) -> VideoCurveBackend {
    match lane {
        VideoLane::Mlx => VideoCurveBackend::Mlx,
        VideoLane::Candle => VideoCurveBackend::Candle,
    }
}

fn curve_load_shape(load_shape: gen_core::LoadShape) -> VideoCurveLoadShape {
    match load_shape {
        gen_core::LoadShape::EagerMaterialization => VideoCurveLoadShape::EagerMaterialization,
        gen_core::LoadShape::DeferredMaterialization => {
            VideoCurveLoadShape::DeferredMaterialization
        }
    }
}

fn curve_decode_pass(decode_pass: VideoDecodePass) -> VideoCurveDecodePass {
    match decode_pass {
        VideoDecodePass::SinglePass => VideoCurveDecodePass::SinglePass,
        VideoDecodePass::Tiled => VideoCurveDecodePass::Tiled,
        VideoDecodePass::Unmodelled => VideoCurveDecodePass::Unmodelled,
    }
}

/// Prefer the exact fitted per-phase curve for this cell; otherwise preserve the established
/// weights-plus-headroom floor byte-for-byte. The lookup itself owns every fail-closed identity and
/// geometry check, including lane/closure/load-shape/mode and the measured area-by-voxel hull.
///
/// There is deliberately no binding-phase flip band here. Each phase has its own fitted law and the
/// scalar is the max over those three evaluations at this exact geometry, so a phase crossing is a
/// result of the measured curves rather than an extrapolation away from one scalar anchor.
fn fitted_or_floor_phase_peaks<'a>(
    selector: &LadderVideoSelector<'a>,
    geometry: VideoAdmissionGeometry,
    strategy: MemoryStrategy,
    engaged: &[MemoryStrategy],
) -> (
    PhasePeaks,
    CandidateBasis,
    &'a str,
    Option<&'a str>,
    Option<&'static str>,
) {
    let fitted = selector
        .curves
        .zip(selector.contract.calibration.as_ref())
        .and_then(|(bundle, calibration)| {
            if calibration.abi != selector.identity.calibration_abi {
                return None;
            }
            bundle.evaluate(VideoCurveQuery {
                model_id: selector.identity.model_id,
                model_family: selector.identity.model_family,
                provider: &selector.contract.provider_id,
                backend: curve_backend(selector.identity.lane),
                tier: crate::mlx_fit_gate::plan_tier_key(selector.identity.tier),
                mode: selector.identity.mode,
                rung: rung_of(strategy),
                load_shape: curve_load_shape(selector.contract.load_shape),
                closure_digest: selector.identity.expected_closure_digest,
                calibration_abi: selector.identity.calibration_abi,
                calibration_fingerprint: &calibration.fingerprint,
                decode_pass: curve_decode_pass(geometry.decode_pass),
                geometry: VideoCurveGeometry {
                    width: geometry.width,
                    height: geometry.height,
                    frames: geometry.estimate_frames(),
                    batch: geometry.batch,
                },
            })
        });
    if let Some(evaluation) = fitted {
        // This candidate is the narrowly ratified binding-phase exemption documented and pinned in
        // `ladder_margin_policy`: every phase has its own residual-bounded law and the scalar below
        // is their request-geometry maximum.
        return (
            PhasePeaks {
                conditioning_bytes: evaluation.phases.conditioning,
                denoise_bytes: evaluation.phases.denoise,
                decode_bytes: evaluation.phases.decode,
            },
            CandidateBasis::EstimateFittedCurve,
            evaluation.closure_digest,
            Some(evaluation.curve_id),
            None,
        );
    }
    let selection = MemorySelection {
        strategy,
        parameters: crate::estimate_synthesis::floor_smallest_parameters(
            selector.contract,
            engaged,
        )
        .unwrap_or_default(),
        tier: selector.identity.tier,
    };
    let (floor, profile_revision) =
        profiled_floor_phase_peaks(selector, geometry, selection, engaged);
    (
        floor,
        CandidateBasis::EstimateFloor,
        selector.identity.expected_closure_digest,
        None,
        profile_revision,
    )
}

/// The gen-core geometry for one video admission cell.
///
/// Requested rows retain the actual clip length because `MemoryGeometry` is calibration/evidence
/// identity: f9/chunk8 and f25/chunk8 must not collide. A synthetic single-pass-cap row uses the
/// interior cap it evaluates, so the non-monotonic cap peak can still bind the request.
fn video_memory_geometry(geometry: VideoAdmissionGeometry, reference_count: u32) -> MemoryGeometry {
    MemoryGeometry {
        width: geometry.width,
        height: geometry.height,
        batch: geometry.batch.max(1),
        frames: geometry.estimate_frames().max(1),
        reference_count,
    }
}

impl VideoStrategySelector for LadderVideoSelector<'_> {
    fn select(&mut self, geometry: VideoAdmissionGeometry) -> VideoRungSelection {
        let memory_geometry = video_memory_geometry(geometry, self.identity.reference_count);
        let backend = self.lane().backend;
        let calibration_fingerprint = self
            .contract
            .calibration
            .as_ref()
            .map(|identity| identity.fingerprint.as_str());

        // One exact-fitted-or-floor candidate per rung the provider's OWN contract can execute. A rung
        // the provider has not wired is never offered — predicting a saving the provider will
        // silently ignore is how a staged prediction turns into a SIGKILL
        // (`mlx_fit_gate::engine_supports_sequential`'s rationale, applied through the contract).
        //
        // The SHARED selector enforces it, and nothing here restates the check. Two candidate-side
        // guards were written, found unkillable by individual mutation, and removed rather than
        // kept with tests shaped to match them:
        //
        // * `support == MemoryStrategySupport::Implemented` — gen-core's `validate_selection`
        //   rejects every non-`Implemented` support at `memory_strategy.rs:1458`.
        // * `contract.validate_selection(&selection)` — `memory_strategy::candidate_exclusion`
        //   runs exactly that call on every candidate (`memory_strategy.rs:466`) and excludes it
        //   as `Invalid`.
        //
        // `mlx_fit_gate::synthesize_estimate_ladder` carries both as a pre-filter; here they would
        // be a second copy of selection policy, which is what epic decision 3 says not to build.
        // `a_rung_whose_prerequisite_is_unmet_is_not_offered` pins that the shared exclusion is in
        // force on the video lane, rather than assuming it.
        let mut synthesized = Vec::new();
        for strategy in MemoryStrategy::ALL {
            let engaged = self.contract.engaged_composition(strategy);
            let Some(parameters) =
                crate::estimate_synthesis::floor_smallest_parameters(self.contract, &engaged)
            else {
                continue;
            };
            let selection = MemorySelection {
                strategy,
                parameters,
                tier: self.identity.tier,
            };
            // Phase-resolved for as long as possible: the scalar is taken only here, where
            // gen-core's evidence type forces one.
            let (phase_peaks, basis, closure_digest, curve_id, profile_revision) =
                fitted_or_floor_phase_peaks(self, geometry, strategy, &engaged);
            if self.profile_error.borrow().is_some() {
                return VideoRungSelection::Undecidable;
            }
            let absolute_predicted_peak_bytes = phase_peaks.peak_bytes();
            let Some(predicted_peak_bytes) =
                absolute_predicted_peak_bytes.checked_sub(self.attributable_resident_bytes)
            else {
                self.accounting_error = Some(format!(
                    "{} live resident attribution {} exceeds modeled total peak {} for {:?}; \
                     refusing an inconsistent video budget",
                    self.identity.route,
                    self.attributable_resident_bytes,
                    absolute_predicted_peak_bytes,
                    strategy,
                ));
                return VideoRungSelection::Undecidable;
            };
            if strategy == MemoryStrategy::Resident {
                self.resident_floors.push((geometry, predicted_peak_bytes));
            }
            let evidence = crate::estimate_synthesis::estimate_evidence(
                self.contract,
                backend,
                self.identity.tier,
                self.identity.mode,
                self.identity.overlay,
                memory_geometry,
                selection,
                predicted_peak_bytes,
                calibration_fingerprint,
            );
            synthesized.push((
                selection,
                evidence,
                phase_peaks,
                basis,
                closure_digest,
                curve_id,
                profile_revision,
            ));
        }
        if synthesized.is_empty() {
            return VideoRungSelection::Undecidable;
        }

        let candidates = synthesized
            .iter()
            .map(
                |(selection, evidence, _, basis, closure_digest, _, _)| Candidate {
                    selection: *selection,
                    evidence,
                    closure_digest,
                    basis: *basis,
                },
            )
            .collect::<Vec<_>>();

        match crate::memory_strategy::select_strategy(
            RequestScope {
                resolved_route: self.identity.route,
                backend: self.identity.lane.as_key(),
                tier: self.identity.tier,
                mode: self.identity.mode,
                overlay: self.identity.overlay,
                geometry: memory_geometry,
                expected_closure_digest: self.identity.expected_closure_digest,
            },
            self.contract,
            self.budget,
            &candidates,
        ) {
            Selection::Selected {
                selection,
                needed_gb,
                available_gb,
            } => {
                tracing::info!(
                    event = "video_memory_strategy_selected",
                    route = self.identity.route,
                    backend = self.identity.lane.as_key(),
                    strategy = ?selection.strategy,
                    request_frames = geometry.frames,
                    estimate_frames = memory_geometry.frames,
                    decode_pass_frames = geometry.decode_pass_frames,
                    decode_pass = ?geometry.decode_pass,
                    geometry_role = ?geometry.role,
                    // Recomputed for THIS geometry's selected rung, never cached: which phase
                    // binds is a geometry property and flips inside one model's envelope.
                    binding_phase = ?synthesized
                        .iter()
                        .find(|(candidate, ..)| candidate.strategy == selection.strategy)
                        .map(|(_, _, peaks, ..)| peaks.binding_phase()),
                    curve_id = synthesized
                        .iter()
                        .find(|(candidate, ..)| candidate.strategy == selection.strategy)
                        .and_then(|(_, _, _, _, _, curve_id, _)| *curve_id)
                        .unwrap_or("none"),
                    needed_gb,
                    available_gb,
                );
                let selected = synthesized
                    .iter()
                    .find(|(candidate, ..)| *candidate == selection)
                    .expect("the shared selector can only return a submitted video candidate");
                self.selections.push(VideoSelectedCandidate {
                    binding_geometry: geometry,
                    selection,
                    predicted_peak_bytes: selected.1.predicted_peak_bytes,
                    evidence_revision: selected
                        .5
                        .or(selected.6)
                        .unwrap_or("video-estimate-floor-v1")
                        .to_owned(),
                });
                VideoRungSelection::Selected {
                    rung: rung_of(selection.strategy),
                    needed_gb,
                    available_gb,
                }
            }
            Selection::Reject {
                needed_gb,
                available_gb,
            } => VideoRungSelection::Reject {
                needed_gb,
                available_gb,
            },
            // No gradable candidate survived: never block without evidence.
            Selection::Unverified { .. } => VideoRungSelection::Undecidable,
        }
    }
}

/// What the video funnel does with the gate's verdict.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct VideoAdmissionOutcome {
    /// The per-request rung knobs to put on `GenerationRequest::memory`. `None` ⇒ leave the field
    /// exactly as it was before this gate existed (the provider's own defaults) — which is what
    /// [`VideoAdmission::NotRouted`], [`VideoAdmission::Undecidable`], and a selected
    /// [`StrategyRung::Resident`] all produce, so those paths are byte-identical to today.
    pub(crate) memory: Option<gen_core::GenerationMemory>,
    /// Exact selected contract/evidence handed to provider safety and request-scope lifecycle.
    /// Present for a contract-backed Resident selection too, even when `memory` is `None` to
    /// preserve the provider's historical request defaults.
    pub(crate) context: Option<MemoryRunContext>,
    /// The house-convention refusal, or `None` to run.
    pub(crate) refusal: Option<String>,
}

/// Route one video generation through the video gate and turn its verdict into per-request rung
/// knobs (sc-18814). The video lane's counterpart to the image lane's
/// `mlx_fit_gate::evaluate_request`, and deliberately at the same position: after the load, before
/// `generate`.
///
/// # The non-regression guard on floor refusal
///
/// A floor-only refusal here can only fire on a job that already cleared the PRE-load gate
/// (`mlx_fit_gate::apply_residency_policy` → `too_big_error`), because that gate runs first on the
/// cold-load path. The ladder is a superset of that gate's two rungs; suppressing the estimate-only
/// margin band preserves the established behavior. [`refusal_is_a_margin_artifact`] deliberately
/// excludes fitted peaks above that floor, so an applicable measured curve can still make a real,
/// geometry-dependent refusal.
///
/// The pinned provider bundle owns the contract and decode profile. Absence remains a fail-open
/// state; callers must never synthesize a compatibility contract because that could manufacture a
/// refusal for a carrier the provider has not promised to execute.
pub(crate) fn admit_video_generation(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
) -> VideoAdmissionOutcome {
    admit_video_generation_with_curves_and_profiles(
        generator,
        request,
        sceneworks_core::video_memory_curves::packaged_video_memory_curves(),
        packaged_video_decode_profile,
    )
}

/// Test seam for exact fixture contracts/curves. Production always calls
/// [`admit_video_generation`] and therefore consumes only the validated packaged bundle.
#[cfg(test)]
fn admit_video_generation_with_curves(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
    curves: Option<&VideoMemoryCurveBundle>,
) -> VideoAdmissionOutcome {
    admit_video_generation_with_curves_and_profiles(
        generator,
        request,
        curves,
        no_video_decode_profile,
    )
}

fn admit_video_generation_with_curves_and_profiles(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
    curves: Option<&VideoMemoryCurveBundle>,
    decode_profile: VideoDecodeProfileResolver,
) -> VideoAdmissionOutcome {
    // The promoted SC-18810 evidence is the reference-free, overlay-free T2V surface. Do not let a
    // provider contract make the floor look like calibration coverage for I2V/keyframe/clip loads,
    // adapters, or enhancers; those routes retain their pre-gate request byte-for-byte.
    if request.mode != "text_to_video"
        || request.reference_count != 0
        || request.overlay.is_some()
        || !(24..=30).contains(&request.fps)
    {
        return VideoAdmissionOutcome::default();
    }
    // Provider safety requires a same-moment post-load budget snapshot. A pre-load total-only probe
    // cannot describe unrelated committed bytes or credit already-resident provider bytes exactly
    // once, so a lane without this snapshot fails open instead of forging a context.
    let Some(runtime) = request.runtime else {
        return VideoAdmissionOutcome::default();
    };
    // No provider contract ⇒ no declared rungs ⇒ nothing for the ladder to select between. Fail
    // open, exactly as `mlx_fit_gate` does when a generator publishes no contract.
    let Some(contract) = generator.memory_strategy_contract() else {
        return VideoAdmissionOutcome::default();
    };
    let attributable_resident_bytes = runtime
        .provider_resident_bytes
        .min(runtime.budget.committed_bytes)
        .min(contract.total_resident_bytes());
    let mut selector = LadderVideoSelector::with_profiles(
        VideoRequestIdentity {
            model_id: request.model_id,
            model_family: request.model_family,
            route: request.route,
            mode: request.mode,
            reference_count: request.reference_count,
            overlay: request.overlay,
            lane: request.lane,
            tier: request.tier,
            calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
            expected_closure_digest: request.expected_closure_digest,
        },
        contract,
        runtime.selector_budget(),
        request.headroom_bytes,
        attributable_resident_bytes,
        curves,
        decode_profile,
    );
    let verdict = sceneworks_core::video_request::video_admission(
        request.model_id,
        request.lane,
        request.width,
        request.height,
        request.frames,
        request.decode_chunk_size,
        &mut selector,
    );
    if let Some(error) = selector.accounting_error {
        return VideoAdmissionOutcome {
            memory: None,
            context: None,
            refusal: Some(error),
        };
    }
    if let Some(error) = selector.profile_error.into_inner() {
        return VideoAdmissionOutcome {
            memory: None,
            context: None,
            refusal: Some(error),
        };
    }
    let resident_floors = selector.resident_floors;
    let selections = selector.selections;
    match verdict {
        sceneworks_core::video_request::VideoAdmission::NotRouted
        | sceneworks_core::video_request::VideoAdmission::Undecidable => {
            VideoAdmissionOutcome::default()
        }
        sceneworks_core::video_request::VideoAdmission::Admitted { rung, geometry, .. } => {
            let selected = selections.iter().find(|candidate| {
                candidate.binding_geometry == geometry
                    && rung_of(candidate.selection.strategy) == rung
            });
            let Some(selected) = selected else {
                tracing::error!(
                    event = "video_memory_selection_lost",
                    route = request.route,
                    ?rung,
                    ?geometry,
                    "the binding video selection was absent from the selector transcript"
                );
                return VideoAdmissionOutcome::default();
            };
            let calibration = contract.calibration.as_ref();
            // Video candidates are fitted-curve or floor syntheses graded behind the shared
            // estimate margin, and the selector transcript keeps no per-candidate measured-cell
            // basis — so no optimized video selection may claim Calibrated authority here.
            let optimization_authority = if selected.selection.strategy.is_optimized() {
                gen_core::MemoryOptimizationAuthority::Estimated
            } else {
                gen_core::MemoryOptimizationAuthority::Resident
            };
            let context = MemoryRunContext {
                selection: selected.selection,
                optimization_authority,
                calibration_abi: calibration.map_or(gen_core::MEMORY_CALIBRATION_ABI, |id| id.abi),
                calibration_fingerprint: calibration
                    .map(|id| id.fingerprint.clone())
                    .unwrap_or_default(),
                load_shape: contract.load_shape,
                mode: crate::memory_strategy::memory_mode_from_mode_key(request.mode),
                has_reference: request.reference_count > 0,
                use_pid: false,
                // Phase-resolved evidence is not a multi-phase request modifier. LTX's canonical
                // reference-free T2V scope carries this false.
                has_phases: false,
                // Provider safety needs the ACTUAL request geometry even when the interior
                // single-pass cap supplied the binding peak and selection.
                geometry: MemoryGeometry {
                    width: request.width,
                    height: request.height,
                    batch: 1,
                    frames: request.frames,
                    reference_count: request.reference_count,
                },
                overlay: request.overlay.map(str::to_owned),
                budget: runtime.budget,
                predicted_peak_bytes: selected.predicted_peak_bytes,
                cache_state: runtime.cache_state,
                evidence_revision: selected.evidence_revision.clone(),
            };
            tracing::info!(
                event = "video_memory_context_built",
                route = request.route,
                cache_state = ?runtime.cache_state,
                load_policy = ?runtime.load_policy,
                incremental_predicted_peak_bytes = selected.predicted_peak_bytes,
                attributable_resident_bytes,
                binding_frames = selected.binding_geometry.estimate_frames(),
                request_frames = request.frames,
            );
            VideoAdmissionOutcome {
                memory: contract.generation_memory(&selected.selection),
                context: Some(context),
                refusal: None,
            }
        }
        sceneworks_core::video_request::VideoAdmission::Refused {
            message,
            needed_gb,
            geometry,
            ..
        } => {
            let Some(resident_floor_bytes) = resident_floors
                .iter()
                .find_map(|(graded, bytes)| (*graded == geometry).then_some(*bytes))
            else {
                tracing::error!(
                    event = "video_memory_resident_floor_lost",
                    route = request.route,
                    ?geometry,
                    "the binding refusal was absent from the resident floor transcript"
                );
                return VideoAdmissionOutcome {
                    memory: None,
                    context: None,
                    refusal: Some(message),
                };
            };
            // This scope check is load-bearing on fitted curves: a measured phase peak may exceed
            // the resident weights floor, in which case the refusal is real and must survive.
            if refusal_is_a_margin_artifact(
                needed_gb,
                resident_floor_bytes,
                crate::memory_strategy::estimate_margin(contract.backend.backend_kind()),
                runtime.selector_budget().and_then(Budget::effective_gb),
            ) {
                tracing::info!(
                    event = "video_memory_strategy_refusal_suppressed",
                    route = request.route,
                    backend = request.lane.as_key(),
                    frames = request.frames,
                    needed_gb,
                    resident_floor_bytes,
                    "ladder rejected inside the estimate-margin band on a peak that IS the \
                     weights+headroom floor; the unwidened floor still fits, so the pre-existing \
                     load gate keeps owning refusal (sc-18814)"
                );
                return VideoAdmissionOutcome::default();
            }
            VideoAdmissionOutcome {
                memory: None,
                context: None,
                refusal: Some(message),
            }
        }
    }
}

/// Whether a ladder rejection may be suppressed as a pure **estimate-margin artifact** on a
/// **floor-shaped peak** — the only refusal `admit_video_generation`'s non-regression guard is
/// entitled to swallow.
///
/// Both conjuncts are load-bearing and neither implies the other:
///
/// 1. **`needed_gb` must not exceed the widened resident floor.** This is the scope check, and it
///    is the whole reason this function exists rather than the budget comparison alone. On a
///    fallback [`floor_phase_peaks`] candidate this holds; on a fitted affine per-phase curve it
///    may not. A guard that compared only the floor to the budget would suppress a genuine
///    all-rungs-reject on a host that fits the weights but not the fitted phase peak, return
///    [`VideoAdmissionOutcome::default()`], and run the job resident into an OOM. Comparing the
///    REJECTED peak keeps the suppression attached to the claim its name makes.
/// 2. **The unwidened resident floor must fit.** This is the non-regression condition proper: it
///    is the same weights+headroom-vs-budget comparison `mlx_fit_gate::fit_decision` already
///    applies to this load, so a job the pre-load gate admits today is never newly refused here.
///
/// No budget signal ⇒ `false`: with nothing to compare against, the ladder cannot have rejected on
/// a margin (`select_strategy` returns `Unverified`, not `Reject`), and suppressing on no evidence
/// would be the inverse of the house never-block-without-evidence posture.
fn refusal_is_a_margin_artifact(
    needed_gb: f64,
    resident_floor_bytes: u64,
    estimate_margin: f64,
    available_gb: Option<f64>,
) -> bool {
    let Some(available_gb) = available_gb else {
        return false;
    };
    // Widen with the SAME integer-byte helper `select_strategy` widened the candidate with, so the
    // two sides of the comparison are produced by one conversion rather than two roundings.
    let widened_floor_gb = crate::memory_strategy::peak_bytes_to_gb(
        crate::memory_strategy::widened_peak_bytes(resident_floor_bytes, estimate_margin),
    );
    let floor_gb = crate::memory_strategy::peak_bytes_to_gb(resident_floor_bytes);
    needed_gb <= widened_floor_gb && floor_gb <= available_gb
}

/// Everything `admit_video_generation` needs that is not on the generator.
pub(crate) struct VideoAdmissionInputs<'a> {
    pub(crate) model_id: &'a str,
    /// The resolved catalog family, not the engine descriptor family.
    pub(crate) model_family: &'a str,
    /// The resolved engine id.
    pub(crate) route: &'a str,
    /// Evidence-mode key (`text_to_video`, `image_to_video`, or another explicitly measured mode).
    pub(crate) mode: &'a str,
    pub(crate) reference_count: u32,
    pub(crate) overlay: Option<&'a str>,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    /// Provider-resolved VAE temporal chunk. `None` means one invocation sees the whole clip; zero
    /// is normalized by core to the providers' minimum of one.
    pub(crate) decode_chunk_size: Option<u32>,
    /// Output FPS is part of the measured surface even though the affine curve is frame-count
    /// keyed. SC-18810 exercised 24 and 30 fps; values outside that envelope fail open.
    pub(crate) fps: u32,
    pub(crate) runtime: Option<VideoRuntimeMemoryState>,
    pub(crate) headroom_bytes: u64,
    pub(crate) expected_closure_digest: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoRuntimeMemoryState {
    pub(crate) budget: MemoryBudget,
    pub(crate) cache_state: MemoryCacheState,
    pub(crate) load_policy: OffloadPolicy,
    /// Fixed backend-committed delta captured around this generator's cold load. This is not
    /// recomputed from a historical external baseline on warm requests.
    pub(crate) provider_resident_bytes: u64,
}

impl VideoRuntimeMemoryState {
    fn selector_budget(self) -> Option<Budget> {
        const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        Some(Budget {
            available_gb: self
                .budget
                .total_bytes
                .saturating_sub(self.budget.committed_bytes) as f64
                / BYTES_PER_GIB,
            reclaimable_gb: self.budget.reclaimable_bytes as f64 / BYTES_PER_GIB,
            total_gb: self.budget.total_bytes as f64 / BYTES_PER_GIB,
            reserved_headroom_gb: self.budget.reserved_headroom_bytes as f64 / BYTES_PER_GIB,
        })
    }
}

/// Capture the same post-load MLX budget snapshot the image request gate uses. The generator-cache
/// callback supplies the fixed cold-load provider delta and cache state separately, so admission
/// can credit only this provider's already-resident bytes while leaving unrelated live allocations
/// charged.
#[cfg(target_os = "macos")]
pub(crate) fn live_video_runtime_state(
    engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    Ok(Some(VideoRuntimeMemoryState {
        // Keep the image lane's canonical foreign/system reserve. The caller removes this exact
        // amount from the fallback activation allowance, yielding 16 + 2 rather than 18 + 2;
        // fitted active-memory curves likewise retain the 2 GiB reserve exactly once.
        budget: crate::mlx_fit_gate::live_request_budget(engine_id)?,
        cache_state,
        load_policy,
        provider_resident_bytes,
    }))
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
fn candle_budget_from_total_free(
    raw_total_bytes: u64,
    raw_free_bytes: u64,
    cap_bytes: Option<u64>,
) -> Option<MemoryBudget> {
    if raw_total_bytes == 0 || raw_free_bytes > raw_total_bytes {
        return None;
    }
    let raw_committed = raw_total_bytes.saturating_sub(raw_free_bytes);
    let total_bytes = cap_bytes.unwrap_or(raw_total_bytes).min(raw_total_bytes);
    Some(MemoryBudget {
        total_bytes,
        committed_bytes: raw_committed.min(total_bytes),
        reclaimable_bytes: 0,
        reserved_headroom_bytes: (crate::fit_gate::DEDICATED_VRAM_ALLOCATOR_SLACK_GB
            * 1024.0
            * 1024.0
            * 1024.0)
            .ceil() as u64,
    })
}

/// Candle reads CUDA's driver allocation counters synchronously after load, on the serialized
/// generator thread. This is the same snapshot used for both live committed pressure and the fixed
/// cold-load provider attribution; an emulation cap changes total capacity but never erases raw
/// committed bytes.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", not(test)))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    let (free, total) =
        runtime_cuda::media::candle_core::cuda::cudarc::driver::result::mem_get_info().map_err(
            |error| crate::WorkerError::Engine(format!("CUDA VRAM snapshot failed: {error}")),
        )?;
    let cap_bytes = crate::vram_gate::cuda_vram_cap_gb()
        .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0).floor() as u64);
    Ok(
        candle_budget_from_total_free(total as u64, free as u64, cap_bytes).map(|budget| {
            VideoRuntimeMemoryState {
                budget,
                cache_state,
                load_policy,
                provider_resident_bytes,
            }
        }),
    )
}

// Candle unit tests inject backend-neutral generators and create no CUDA context. Give those seams
// a deterministic synthetic device budget; non-test binaries retain the physical driver snapshot
// above, which the exact-head CUDA acceptance run exercises with real weights.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", test))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    cache_state: MemoryCacheState,
    load_policy: OffloadPolicy,
    provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    const TEST_TOTAL_BYTES: u64 = 24 * 1024 * 1024 * 1024;
    Ok(
        candle_budget_from_total_free(TEST_TOTAL_BYTES, TEST_TOTAL_BYTES, None).map(|budget| {
            VideoRuntimeMemoryState {
                budget,
                cache_state,
                load_policy,
                provider_resident_bytes,
            }
        }),
    )
}

#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) fn live_video_runtime_state(
    _engine_id: &str,
    _cache_state: MemoryCacheState,
    _load_policy: OffloadPolicy,
    _provider_resident_bytes: u64,
) -> crate::WorkerResult<Option<VideoRuntimeMemoryState>> {
    Ok(None)
}

/// Which lane this build's video path executes on.
#[cfg(target_os = "macos")]
pub(crate) const LANE: VideoLane = VideoLane::Mlx;
#[cfg(not(target_os = "macos"))]
pub(crate) const LANE: VideoLane = VideoLane::Candle;

/// Bridge the gen-core strategy back to the rung spelling `sceneworks-core` returns. The inverse of
/// `memory_strategy::strategy_from_rung`, and exhaustive for the same reason.
const fn rung_of(strategy: MemoryStrategy) -> StrategyRung {
    match strategy {
        MemoryStrategy::Resident => StrategyRung::Resident,
        MemoryStrategy::StagedResidency => StrategyRung::StagedResidency,
        MemoryStrategy::BoundedDecode => StrategyRung::BoundedDecode,
        MemoryStrategy::BoundedAttention => StrategyRung::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency => StrategyRung::BoundedTransformerResidency,
    }
}

#[cfg(test)]
mod tests;
