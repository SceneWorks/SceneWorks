//! Backend-neutral fit-gate outcome and sequential-residency transition (sc-12127).
//!
//! MLX and candle derive their predicted resident and sequential peaks differently, but once a
//! resident fit decision exists they share this state transition exactly. Keeping it here prevents
//! the two admission gates from drifting while leaving their backend-specific budgeting untouched.

/// The shared outcome of comparing a predicted resident peak with an available memory budget.
/// `Unknown` means the backend has no reliable signal and therefore must not reject the job.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FitDecision {
    Unknown,
    Fits,
    /// The resident peak does not fit, but the provider supports sequential component residency.
    /// The caller must still run its backend-specific sequential-peak check before loading.
    Offload {
        needed_gb: f64,
        available_gb: f64,
    },
    TooBig {
        needed_gb: f64,
        available_gb: f64,
    },
}

/// Select sequential residency when a resident load is too large and the provider advertises that
/// capability. Every other decision is preserved unchanged.
pub(crate) fn resolve_offload(decision: FitDecision, sequential_capable: bool) -> FitDecision {
    match decision {
        FitDecision::TooBig {
            needed_gb,
            available_gb,
        } if sequential_capable => FitDecision::Offload {
            needed_gb,
            available_gb,
        },
        other => other,
    }
}
