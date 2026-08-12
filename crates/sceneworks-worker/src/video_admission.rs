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
//! **No new prediction math** (activity-19060). Each rung's peak is
//! [`crate::mlx_fit_gate::estimate_floor_weights_bytes`] — the contract's OWN declared component
//! bytes reduced by the rung's declared composition — plus the caller's existing activation
//! headroom allowance. That is the same weights+headroom prediction the video lane already lives
//! under at cold load (`mlx_fit_gate::decide_residency_for_spec`), which never reached the
//! selector. Routing it there is the whole point of this story; inventing a second law would not
//! be.
//!
//! **The seam sc-18829 attaches to.** `gen_core::MemoryGeometry` already carries `frames`, and
//! [`LadderVideoSelector::select`] populates it from the video request's real frame count rather
//! than the image lane's hardcoded `1` (`mlx_fit_gate::request_geometry`). When the MLX
//! frames-aware term lands it changes how a peak is computed for a geometry this seam already
//! hands over complete — including its decode regime — so nothing here has to be re-plumbed.

use gen_core::{
    MemoryBackend, MemoryGeometry, MemoryNumericTier, MemoryProviderContract, MemorySelection,
    MemoryStrategy,
};
use sceneworks_core::memory_calibration::StrategyRung;
use sceneworks_core::video_request::{
    VideoAdmissionGeometry, VideoLane, VideoRungSelection, VideoStrategySelector,
};

use crate::memory_strategy::{Budget, Candidate, CandidateBasis, RequestScope, Selection};

/// The canonical mode-key a video admission cell is recorded under. `MemoryMode::Other`, like
/// every non-image mode, and stable so an evidence cell captured for video keys identically
/// wherever it is read.
pub(crate) const VIDEO_MODE_KEY: &str = "video_generation";

/// A video request's identity for the selector, minus the geometry (which arrives per-call).
pub(crate) struct VideoRequestIdentity<'a> {
    /// The resolved engine id — the same `resolved_route` spelling every other admission cell uses.
    pub(crate) route: &'a str,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
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
    /// Every `(geometry, selection)` the selector chose, so the caller can recover the selected
    /// PARAMETERS for the geometry core reports as binding. Core's seam returns a rung; the
    /// per-request knobs need the whole `MemorySelection`, and re-deriving it would be a second
    /// selection.
    selections: Vec<(VideoAdmissionGeometry, MemorySelection)>,
}

impl<'a> LadderVideoSelector<'a> {
    pub(crate) fn new(
        identity: VideoRequestIdentity<'a>,
        contract: &'a MemoryProviderContract,
        budget: Option<Budget>,
        headroom_bytes: u64,
    ) -> Self {
        Self {
            identity,
            contract,
            budget,
            headroom_bytes,
            selections: Vec::new(),
        }
    }

    /// The gen-core backend this lane grades under. Exhaustive on [`VideoLane`] so a new lane
    /// cannot compile without choosing one — the same posture
    /// `memory_strategy::stale_measured_margin` takes on [`MemoryBackend`].
    pub(crate) const fn backend(&self) -> MemoryBackend {
        match self.identity.lane {
            VideoLane::Mlx => MemoryBackend::Mlx,
            VideoLane::Candle => MemoryBackend::Candle,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VideoBindingPhase {
    Conditioning,
    Denoise,
    Decode,
}

/// The per-phase peaks one rung's prediction is made of.
///
/// **Deliberately not reduced to a scalar until the last possible moment.** sc-18810 measured every
/// candidate temporal form missing the AGGREGATE peak by at least 10.26 GiB — about 94x the
/// replicate noise floor — while the same forms land at 0.019–0.44 GiB *per phase*. A prediction
/// that is accurate at all is therefore phase-resolved, and the admission number is the **max over
/// phases**, exactly the discipline `KreaTurboPhasePeaks::peak_gb` already applies on the candle
/// side.
///
/// The fitted laws are **affine** (`fixedGb + k * voxels`) with large, phase-specific intercepts —
/// decode ~2.5 GB, denoise ~20.6 GB (the staged transformer floor), text ~32.8 GB with a tiny
/// slope. A through-origin scalar ratio like `mlx_fit_gate`'s `scaled(bytes) = bytes * scale`
/// (`mlx_fit_gate.rs:1699`) cannot express them, which is why this seam keeps the three values
/// apart rather than handing one number across.
///
/// **Today every field carries the same weights+headroom floor** — that floor has no phase
/// resolution, and inventing one would be new prediction math this story must not add. sc-18829
/// replaces [`floor_phase_peaks`] with a fitted affine evaluation per phase at the request
/// geometry; [`Self::peak_bytes`] and everything downstream of it are unchanged by that.
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

    /// Which phase binds AT THIS GEOMETRY. Ties resolve to the LATER phase, matching
    /// `mlx_fit_gate::binding_phase` so the two never disagree on the same triple.
    pub(crate) const fn binding_phase(self) -> VideoBindingPhase {
        let mut phase = VideoBindingPhase::Conditioning;
        let mut peak = self.conditioning_bytes;
        if self.denoise_bytes >= peak {
            phase = VideoBindingPhase::Denoise;
            peak = self.denoise_bytes;
        }
        if self.decode_bytes >= peak {
            phase = VideoBindingPhase::Decode;
        }
        phase
    }
}

/// The weights+headroom floor for one engaged composition, expressed per phase.
///
/// No new prediction math: the scalar is exactly
/// `mlx_fit_gate::estimate_floor_weights_bytes(contract, engaged) + headroom_bytes`, the same
/// number the cold-load residency gate already charges this load. It is phase-blind, so all three
/// phases carry it and [`PhasePeaks::peak_bytes`] returns that scalar unchanged — the SHAPE is
/// what this story establishes, not a new law. sc-18829 substitutes per-phase fitted values here.
fn floor_phase_peaks(
    contract: &MemoryProviderContract,
    engaged: &[MemoryStrategy],
    headroom_bytes: u64,
) -> PhasePeaks {
    let floor = crate::mlx_fit_gate::estimate_floor_weights_bytes(contract, engaged)
        .saturating_add(headroom_bytes);
    PhasePeaks {
        conditioning_bytes: floor,
        denoise_bytes: floor,
        decode_bytes: floor,
    }
}

/// The gen-core geometry for one video admission cell.
///
/// `frames` is the video request's real frame count. The image lane pins it at 1
/// (`mlx_fit_gate::request_geometry`), which is correct there and is exactly why the MLX area law
/// has never needed a temporal term; a video cell that reported 1 would make every frame count at
/// one resolution collide into a single evidence key.
fn video_memory_geometry(geometry: VideoAdmissionGeometry) -> MemoryGeometry {
    MemoryGeometry {
        width: geometry.width,
        height: geometry.height,
        batch: geometry.batch.max(1),
        frames: geometry.frames.max(1),
        reference_count: 0,
    }
}

impl VideoStrategySelector for LadderVideoSelector<'_> {
    fn select(&mut self, geometry: VideoAdmissionGeometry) -> VideoRungSelection {
        let memory_geometry = video_memory_geometry(geometry);
        let backend = self.backend();
        let calibration_fingerprint = self
            .contract
            .calibration
            .as_ref()
            .map(|identity| identity.fingerprint.as_str());

        // One estimate-floor candidate per rung the provider's OWN contract can execute. A rung
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
                crate::mlx_fit_gate::estimate_floor_parameters(self.contract, &engaged)
            else {
                continue;
            };
            let selection = MemorySelection {
                strategy,
                parameters,
                tier: self.identity.tier,
            };
            // Phase-resolved for as long as possible: the scalar is taken only here, where
            // gen-core's evidence type forces one. sc-18829 changes what the three phase values
            // ARE without moving this boundary.
            let phase_peaks = floor_phase_peaks(self.contract, &engaged, self.headroom_bytes);
            let predicted_peak_bytes = phase_peaks.peak_bytes();
            let evidence = crate::mlx_fit_gate::estimate_evidence(
                self.contract,
                backend,
                self.identity.tier,
                VIDEO_MODE_KEY,
                None,
                memory_geometry,
                selection,
                predicted_peak_bytes,
                calibration_fingerprint,
            );
            synthesized.push((selection, evidence, phase_peaks));
        }
        if synthesized.is_empty() {
            return VideoRungSelection::Undecidable;
        }

        let candidates = synthesized
            .iter()
            .map(|(selection, evidence, _)| Candidate {
                selection: *selection,
                evidence,
                // A floor is a declaration under the LIVE closure — there is nothing there for
                // currency to invalidate, so both sides carry the request's own digest.
                closure_digest: self.identity.expected_closure_digest,
                basis: CandidateBasis::EstimateFloor,
            })
            .collect::<Vec<_>>();

        match crate::memory_strategy::select_strategy(
            RequestScope {
                resolved_route: self.identity.route,
                backend: self.identity.lane.as_key(),
                tier: self.identity.tier,
                mode: VIDEO_MODE_KEY,
                overlay: None,
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
                    frames = memory_geometry.frames,
                    decode_pass = ?geometry.decode_pass,
                    geometry_role = ?geometry.role,
                    // Recomputed for THIS geometry's selected rung, never cached: which phase
                    // binds is a geometry property and flips inside one model's envelope.
                    binding_phase = ?synthesized
                        .iter()
                        .find(|(candidate, ..)| candidate.strategy == selection.strategy)
                        .map(|(.., peaks)| peaks.binding_phase()),
                    needed_gb,
                    available_gb,
                );
                self.selections.push((geometry, selection));
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
    /// The house-convention refusal, or `None` to run.
    pub(crate) refusal: Option<String>,
}

/// Route one video generation through the video gate and turn its verdict into per-request rung
/// knobs (sc-18814). The video lane's counterpart to the image lane's
/// `mlx_fit_gate::evaluate_request`, and deliberately at the same position: after the load, before
/// `generate`.
///
/// # The non-regression guard on refusal
///
/// A refusal here can only ever fire on a job that already cleared the PRE-load gate
/// (`mlx_fit_gate::apply_residency_policy` → `too_big_error`), because that gate runs first on the
/// cold-load path. The ladder is a superset of that gate's two rungs, so the only way this one
/// could newly refuse is the estimate margin `select_strategy` widens floor candidates by
/// (`ladder_margin_policy::{MLX,CANDLE}_ESTIMATE_MARGIN`) — i.e. a job inside the margin band that
/// runs today.
///
/// The story's success criterion is explicit that no video job which succeeds today may regress,
/// so a refusal is emitted only when the **unwidened** resident floor also exceeds the budget —
/// the same weights+headroom-vs-budget comparison `mlx_fit_gate::fit_decision` already applies to
/// this load. Inside the margin band the ladder still does its real job: it selects a lower rung.
/// Whether the image-derived margins should govern video at all is epic 18093's settled question,
/// reopened for this lane by activity-18996 and owned by sc-18829 — not re-decided here.
pub(crate) fn admit_video_generation(
    generator: &dyn gen_core::Generator,
    request: VideoAdmissionInputs<'_>,
) -> VideoAdmissionOutcome {
    // No provider contract ⇒ no declared rungs ⇒ nothing for the ladder to select between. Fail
    // open, exactly as `mlx_fit_gate` does when a generator publishes no contract.
    let Some(contract) = generator.memory_strategy_contract() else {
        return VideoAdmissionOutcome::default();
    };
    let mut selector = LadderVideoSelector::new(
        VideoRequestIdentity {
            route: request.route,
            lane: request.lane,
            tier: request.tier,
            expected_closure_digest: request.expected_closure_digest,
        },
        contract,
        request.budget,
        request.headroom_bytes,
    );
    let verdict = sceneworks_core::video_request::video_admission(
        request.model_id,
        request.lane,
        request.width,
        request.height,
        request.frames,
        &mut selector,
    );
    let selections = selector.selections;
    match verdict {
        sceneworks_core::video_request::VideoAdmission::NotRouted
        | sceneworks_core::video_request::VideoAdmission::Undecidable => {
            VideoAdmissionOutcome::default()
        }
        sceneworks_core::video_request::VideoAdmission::Admitted { rung, geometry, .. } => {
            // The resident rung engages nothing, so emitting knobs for it would replace the
            // provider's own defaults with an explicit all-false. Leave the field untouched.
            if rung == StrategyRung::Resident {
                return VideoAdmissionOutcome::default();
            }
            let memory = selections
                .iter()
                .find(|(candidate, selection)| {
                    *candidate == geometry && rung_of(selection.strategy) == rung
                })
                .map(|(_, selection)| {
                    crate::mlx_fit_gate::memory_for_selection(contract, *selection)
                });
            VideoAdmissionOutcome {
                memory,
                refusal: None,
            }
        }
        sceneworks_core::video_request::VideoAdmission::Refused { message, .. } => {
            let resident_floor_bytes = crate::mlx_fit_gate::estimate_floor_weights_bytes(
                contract,
                &contract.engaged_composition(MemoryStrategy::Resident),
            )
            .saturating_add(request.headroom_bytes);
            let unwidened_floor_fits =
                request
                    .budget
                    .and_then(Budget::effective_gb)
                    .is_some_and(|available_gb| {
                        (resident_floor_bytes as f64 / crate::fit_gate::BYTES_PER_GIB)
                            <= available_gb
                    });
            if unwidened_floor_fits {
                tracing::info!(
                    event = "video_memory_strategy_refusal_suppressed",
                    route = request.route,
                    backend = request.lane.as_key(),
                    frames = request.frames,
                    "ladder rejected inside the estimate-margin band; the unwidened resident floor \
                     still fits, so the pre-existing load gate keeps owning refusal (sc-18814)"
                );
                return VideoAdmissionOutcome::default();
            }
            VideoAdmissionOutcome {
                memory: None,
                refusal: Some(message),
            }
        }
    }
}

/// Everything `admit_video_generation` needs that is not on the generator.
pub(crate) struct VideoAdmissionInputs<'a> {
    pub(crate) model_id: &'a str,
    /// The resolved engine id.
    pub(crate) route: &'a str,
    pub(crate) lane: VideoLane,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) budget: Option<Budget>,
    pub(crate) headroom_bytes: u64,
    pub(crate) expected_closure_digest: &'a str,
}

/// The live budget this lane admits against, as the shared selector's [`Budget`].
///
/// Each lane reads the budget its own hardware actually has, from the source that lane's existing
/// gate already uses — MLX the unified-memory total (`mlx_fit_gate::live_unified_budget_gb`, the
/// same figure `decide_residency` budgets against, honoring the small-Mac emulation cap), candle
/// the card's free+reclaimable VRAM (`video_jobs::candle::candle_video_vram_budget`, the same
/// budget `wan_video_fit_error` / `svd_fit_error` are handed). Nothing new is measured here.
///
/// `reserved_headroom_gb` is 0 because the headroom is already inside every candidate's peak
/// (`estimate_floor_weights_bytes + headroom_bytes`); charging it on both sides would double-count
/// it and make this gate strictly stricter than the one it sits behind.
#[cfg(target_os = "macos")]
pub(crate) async fn live_video_budget(_settings: &crate::settings::Settings) -> Option<Budget> {
    crate::mlx_fit_gate::live_unified_budget_gb().map(|total_gb| Budget {
        available_gb: total_gb,
        reclaimable_gb: 0.0,
        total_gb,
        reserved_headroom_gb: 0.0,
    })
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn live_video_budget(settings: &crate::settings::Settings) -> Option<Budget> {
    crate::video_jobs::candle::candle_video_vram_budget(settings)
        .await
        .map(|budget| Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            // A synthetic zero-total emulated card still stands for a real device; keep the shared
            // budget type's physical-total invariant satisfied the way `krea_control_fit` does.
            total_gb: budget.total_gb.max(f64::EPSILON),
            reserved_headroom_gb: 0.0,
        })
}

/// The stub lane (no macOS, no candle) renders nothing, so it admits nothing.
#[cfg(all(not(target_os = "macos"), not(feature = "backend-candle")))]
pub(crate) async fn live_video_budget(_settings: &crate::settings::Settings) -> Option<Budget> {
    None
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
