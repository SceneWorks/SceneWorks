//! Backend-neutral image-memory strategy contract and least-cost selector (sc-15449).
//!
//! Providers own what they can realize and the formula/ABI used to estimate it. Generated evidence
//! owns calibrated coefficients and envelopes. The worker owns the live budget and this selection
//! algorithm. In particular, this module contains no model allowlist and no backend formula.
//!
//! The contract deliberately separates an ordinary resident attempt from an *optimized* strategy.
//! Unknown, stale, fingerprint-mismatched, structurally inapplicable, and out-of-envelope evidence
//! can never authorize an optimization. A caller may preserve its established provider-safe fallback
//! when [`Selection::Unverified`] is returned, but must not claim that fallback was memory-optimized.

use gen_core::{
    ImageMemoryBackendRealization, ImageMemoryCleanupSemantics, ImageMemoryProviderContract,
    ImageMemoryStrategySupport, IMAGE_MEMORY_CALIBRATION_ABI,
};
pub use gen_core::{
    ImageMemoryConformanceState as ConformanceState,
    ImageMemoryEvidenceDimensions as EvidenceDimensions, ImageMemoryStrategy as Strategy,
};

/// Backend semantics are explicit: CUDA budgets discrete VRAM; Metal budgets unified memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    CandleCuda,
    MlxMetal,
}

/// Provider-owned strategy capabilities and lifecycle semantics.
///
/// `warm_cache_reusable` means the realized strategy may remain resident across jobs.
/// `cancel_safe` and `error_safe` require every load/drop transition to leave the cache either
/// reusable or atomically invalidated. The hook bits describe backend realization, not calibration:
/// providers still gate unsupported requests in depth even after worker selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub backend: Backend,
    pub resident: bool,
    pub staged_residency: bool,
    pub bounded_decode_hook: bool,
    pub bounded_attention_hook: bool,
    pub bounded_transformer_hook: bool,
    pub warm_cache_reusable: bool,
    pub cancel_safe: bool,
    pub error_safe: bool,
}

impl ProviderCapabilities {
    /// Project the provider-owned inference contract into worker selection capabilities.
    pub fn from_provider(contract: &ImageMemoryProviderContract) -> Option<Self> {
        contract.conformance_errors().is_empty().then(|| Self {
            backend: match contract.backend {
                ImageMemoryBackendRealization::CandleCuda { .. } => Backend::CandleCuda,
                ImageMemoryBackendRealization::MlxMetal { .. } => Backend::MlxMetal,
            },
            resident: contract_supports(contract, Strategy::Resident),
            staged_residency: contract_supports(contract, Strategy::StagedResidency),
            bounded_decode_hook: contract_supports(contract, Strategy::BoundedDecode),
            bounded_attention_hook: contract_supports(contract, Strategy::BoundedAttention),
            bounded_transformer_hook: contract_supports(
                contract,
                Strategy::BoundedTransformerResidency,
            ),
            warm_cache_reusable: true,
            cancel_safe: contract.runtime.cancellation
                == ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
            error_safe: contract.runtime.error
                == ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
        })
    }

    pub fn supports(self, strategy: Strategy) -> bool {
        match strategy {
            Strategy::Resident => self.resident,
            Strategy::StagedResidency => self.staged_residency,
            Strategy::BoundedDecode => self.bounded_decode_hook,
            Strategy::BoundedAttention => self.bounded_attention_hook,
            Strategy::BoundedTransformerResidency => self.bounded_transformer_hook,
        }
    }

    fn lifecycle_safe(self) -> bool {
        self.cancel_safe && self.error_safe
    }
}

fn contract_supports(contract: &ImageMemoryProviderContract, strategy: Strategy) -> bool {
    matches!(
        contract
            .capability(strategy)
            .map(|capability| &capability.support),
        Some(ImageMemoryStrategySupport::Implemented)
    )
}

/// Request scope that calibration evidence must cover exactly.
///
/// The precision/tier key is immutable across the selector: a strategy can trade execution cost for
/// memory, but may never silently lower precision. Providers that offer a lower tier submit it as a
/// separate worker candidate before this selector is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestScope<'a> {
    pub backend: Backend,
    pub tier: &'a str,
    pub mode: &'a str,
    pub overlay: &'a str,
    pub width: u32,
    pub height: u32,
}

/// Calibration envelope owned by manifest/generated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Envelope<'a> {
    pub backend: Backend,
    pub tier: &'a str,
    pub mode: &'a str,
    pub overlay: &'a str,
    pub max_pixels: u64,
}

impl Envelope<'_> {
    fn covers(self, request: RequestScope<'_>) -> bool {
        self.backend == request.backend
            && self.tier == request.tier
            && self.mode == request.mode
            && self.overlay == request.overlay
            && u64::from(request.width)
                .checked_mul(u64::from(request.height))
                .is_some_and(|pixels| pixels <= self.max_pixels)
    }
}

/// Evidence attached to one estimate.
///
/// `provider_abi_fingerprint` is supplied by the provider capability ABI; `calibration_fingerprint`
/// is generated with the coefficients. A mismatch makes the evidence stale/unverified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evidence<'a> {
    pub state: ConformanceState,
    pub dimensions: EvidenceDimensions,
    pub provider_abi_fingerprint: &'a str,
    pub calibration_fingerprint: &'a str,
    pub provider_calibration_abi: u32,
    pub calibration_abi: u32,
    pub scene_works_revision: &'a str,
    pub expected_scene_works_revision: &'a str,
    pub inference_revision: &'a str,
    pub expected_inference_revision: &'a str,
    pub envelope: Envelope<'a>,
}

impl Evidence<'_> {
    fn verified_for(self, request: RequestScope<'_>) -> bool {
        self.state == ConformanceState::Verified
            && self.dimensions.all_satisfied()
            && !self.provider_abi_fingerprint.is_empty()
            && self.provider_calibration_abi == IMAGE_MEMORY_CALIBRATION_ABI
            && self.calibration_abi == self.provider_calibration_abi
            && self.provider_abi_fingerprint == self.calibration_fingerprint
            && self.scene_works_revision == self.expected_scene_works_revision
            && self.inference_revision == self.expected_inference_revision
            && self.envelope.covers(request)
    }
}

/// A provider estimate submitted to the worker-owned selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate<'a> {
    pub strategy: Strategy,
    /// Strictly increasing execution cost. Equal-cost candidates retain provider order.
    pub cost: u16,
    /// Provider formula evaluated with generated coefficients.
    pub needed_gb: f64,
    /// Must equal `request.tier`; this is the precision-invariance guard.
    pub precision_tier: &'a str,
    pub evidence: Evidence<'a>,
}

/// Worker-owned live budget. Reclaimable memory is credited once, reserved headroom is then
/// subtracted from that live pool, and the result is clamped to the equally reserved physical cap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub available_gb: f64,
    pub reclaimable_gb: f64,
    pub total_gb: f64,
    pub reserved_headroom_gb: f64,
}

impl Budget {
    pub fn effective_gb(self) -> Option<f64> {
        if !self.available_gb.is_finite()
            || !self.reclaimable_gb.is_finite()
            || !self.total_gb.is_finite()
            || !self.reserved_headroom_gb.is_finite()
            || self.available_gb < 0.0
            || self.reclaimable_gb < 0.0
            || self.total_gb <= 0.0
            || self.reserved_headroom_gb < 0.0
        {
            return None;
        }
        let reserved_live =
            (self.available_gb + self.reclaimable_gb - self.reserved_headroom_gb).max(0.0);
        let reserved_total = (self.total_gb - self.reserved_headroom_gb).max(0.0);
        Some(reserved_live.min(reserved_total))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Selection {
    Selected {
        strategy: Strategy,
        needed_gb: f64,
        available_gb: f64,
    },
    /// Every eligible verified strategy was evaluated and none fit.
    Reject { needed_gb: f64, available_gb: f64 },
    /// The worker lacked trustworthy evidence or a live budget. No optimized strategy was selected.
    Unverified,
}

/// Select the least-cost sufficient strategy without model-specific policy.
///
/// Numerical policy is explicit and shared: estimates and budgets are in GiB and equality fits
/// exactly (`needed <= available`). Calibrations that require a tolerance must include it in their
/// provider-owned estimate; the selector never invents a tolerance or changes precision.
pub fn select_strategy(
    request: RequestScope<'_>,
    capabilities: ProviderCapabilities,
    budget: Option<Budget>,
    candidates: &[Candidate<'_>],
) -> Selection {
    if capabilities.backend != request.backend || !capabilities.lifecycle_safe() {
        return Selection::Unverified;
    }
    let Some(available_gb) = budget.and_then(Budget::effective_gb) else {
        return Selection::Unverified;
    };

    let mut ordered = candidates.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|candidate| candidate.cost);
    let mut deepest = None;
    for candidate in ordered {
        if candidate.precision_tier != request.tier
            || !capabilities.supports(candidate.strategy)
            || !candidate.needed_gb.is_finite()
            || candidate.needed_gb < 0.0
            || !candidate.evidence.verified_for(request)
        {
            // One broken link means the ladder is incomplete. Do not skip a cheaper unknown rung and
            // claim a deeper optimized rung is authoritative.
            return Selection::Unverified;
        }
        deepest = Some(candidate.needed_gb);
        if candidate.needed_gb <= available_gb {
            return Selection::Selected {
                strategy: candidate.strategy,
                needed_gb: candidate.needed_gb,
                available_gb,
            };
        }
    }

    deepest.map_or(Selection::Unverified, |needed_gb| Selection::Reject {
        needed_gb,
        available_gb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: &str = "provider-formula-v1";
    const SW: &str = "source-tree:test";
    const INF: &str = "1deefff";

    fn request() -> RequestScope<'static> {
        RequestScope {
            backend: Backend::CandleCuda,
            tier: "q4",
            mode: "text_to_image",
            overlay: "none",
            width: 1024,
            height: 1024,
        }
    }

    fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            backend: Backend::CandleCuda,
            resident: true,
            staged_residency: true,
            bounded_decode_hook: true,
            bounded_attention_hook: true,
            bounded_transformer_hook: true,
            warm_cache_reusable: true,
            cancel_safe: true,
            error_safe: true,
        }
    }

    fn candidate(strategy: Strategy, cost: u16, needed_gb: f64) -> Candidate<'static> {
        Candidate {
            strategy,
            cost,
            needed_gb,
            precision_tier: "q4",
            evidence: Evidence {
                state: ConformanceState::Verified,
                dimensions: EvidenceDimensions::VERIFIED,
                provider_abi_fingerprint: FP,
                calibration_fingerprint: FP,
                provider_calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
                calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
                scene_works_revision: SW,
                expected_scene_works_revision: SW,
                inference_revision: INF,
                expected_inference_revision: INF,
                envelope: Envelope {
                    backend: Backend::CandleCuda,
                    tier: "q4",
                    mode: "text_to_image",
                    overlay: "none",
                    max_pixels: 1_048_576,
                },
            },
        }
    }

    #[test]
    fn selects_ordered_least_cost_and_exact_boundary_fits() {
        let candidates = [
            candidate(Strategy::Resident, 0, 24.0),
            candidate(Strategy::StagedResidency, 10, 16.0),
            candidate(Strategy::BoundedDecode, 20, 12.0),
        ];
        assert_eq!(
            select_strategy(
                request(),
                capabilities(),
                Some(Budget {
                    available_gb: 16.0,
                    reclaimable_gb: 0.0,
                    total_gb: 24.0,
                    reserved_headroom_gb: 0.0,
                }),
                &candidates,
            ),
            Selection::Selected {
                strategy: Strategy::StagedResidency,
                needed_gb: 16.0,
                available_gb: 16.0,
            }
        );
    }

    #[test]
    fn reclaimable_is_reserved_then_clamped() {
        let candidates = [candidate(Strategy::Resident, 0, 18.0)];
        assert_eq!(
            select_strategy(
                request(),
                capabilities(),
                Some(Budget {
                    available_gb: 8.0,
                    reclaimable_gb: 99.0,
                    total_gb: 20.0,
                    reserved_headroom_gb: 2.0,
                }),
                &candidates,
            ),
            Selection::Selected {
                strategy: Strategy::Resident,
                needed_gb: 18.0,
                available_gb: 18.0,
            }
        );
    }

    #[test]
    fn headroom_is_subtracted_from_live_availability_before_exact_boundary_selection() {
        let exact = [candidate(Strategy::Resident, 0, 10.0)];
        let budget = Some(Budget {
            available_gb: 8.0,
            reclaimable_gb: 4.0,
            total_gb: 24.0,
            reserved_headroom_gb: 2.0,
        });
        assert!(matches!(
            select_strategy(request(), capabilities(), budget, &exact),
            Selection::Selected {
                available_gb: 10.0,
                ..
            }
        ));

        let above = [candidate(Strategy::Resident, 0, 10.000_001)];
        assert_eq!(
            select_strategy(request(), capabilities(), budget, &above),
            Selection::Reject {
                needed_gb: 10.000_001,
                available_gb: 10.0,
            }
        );
    }

    #[test]
    fn headroom_underflow_saturates_to_zero() {
        let budget = Budget {
            available_gb: 1.0,
            reclaimable_gb: 0.5,
            total_gb: 24.0,
            reserved_headroom_gb: 2.0,
        };
        assert_eq!(budget.effective_gb(), Some(0.0));

        let over_reserved_total = Budget {
            available_gb: 24.0,
            reclaimable_gb: 99.0,
            total_gb: 24.0,
            reserved_headroom_gb: 25.0,
        };
        assert_eq!(over_reserved_total.effective_gb(), Some(0.0));
    }

    #[test]
    fn unknown_stale_mismatch_out_of_envelope_and_precision_drift_never_optimize() {
        let baseline = candidate(Strategy::StagedResidency, 10, 8.0);
        let mut cases = Vec::new();

        let mut unverified = baseline;
        unverified.evidence.state = ConformanceState::ImplementedUnverified;
        cases.push(unverified);
        let mut mismatch = baseline;
        mismatch.evidence.calibration_fingerprint = "different";
        cases.push(mismatch);
        let mut stale = baseline;
        stale.evidence.expected_inference_revision = "newer";
        cases.push(stale);
        let mut outside = baseline;
        outside.evidence.envelope.max_pixels = 100;
        cases.push(outside);
        let mut precision_drift = baseline;
        precision_drift.precision_tier = "q8";
        cases.push(precision_drift);

        for candidate in cases {
            assert_eq!(
                select_strategy(
                    request(),
                    capabilities(),
                    Some(Budget {
                        available_gb: 10.0,
                        reclaimable_gb: 0.0,
                        total_gb: 10.0,
                        reserved_headroom_gb: 0.0,
                    }),
                    &[candidate],
                ),
                Selection::Unverified
            );
        }
    }

    #[test]
    fn structural_na_is_evidence_not_a_fit() {
        let mut structurally_na = candidate(Strategy::BoundedDecode, 20, 1.0);
        structurally_na.evidence.state = ConformanceState::StructurallyNotApplicable;
        assert_eq!(
            select_strategy(
                request(),
                capabilities(),
                Some(Budget {
                    available_gb: 10.0,
                    reclaimable_gb: 0.0,
                    total_gb: 10.0,
                    reserved_headroom_gb: 0.0,
                }),
                &[structurally_na],
            ),
            Selection::Unverified
        );
    }
}
