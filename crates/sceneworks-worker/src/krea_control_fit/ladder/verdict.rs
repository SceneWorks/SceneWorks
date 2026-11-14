use super::*;

impl ControlLadderContext<'_> {
    pub(super) fn verdict(
        &self,
        selected: Selection,
        budget: crate::vram_gate::VramBudget,
        measured: &[(MemoryStrategy, f64)],
    ) -> KreaControlFit {
        let Self {
            contract,
            request_geometry,
            measured_geometry,
            ..
        } = *self;
        match selected {
            Selection::Selected { selection, .. } => KreaControlFit::Fits {
                offload_policy: if selection.strategy == MemoryStrategy::Resident {
                    OffloadPolicy::Resident
                } else {
                    OffloadPolicy::Sequential
                },
                tile_vae_decode: contract
                    .engages(selection.strategy, MemoryStrategy::BoundedDecode),
                chunk_attention: contract
                    .engages(selection.strategy, MemoryStrategy::BoundedAttention),
                // sc-18097: a selection is estimate-scoped when it cannot be the measured 1024² cell —
                // the request geometry is off the measured cell (every measured candidate was
                // structurally excluded there), or the selected rung has no measured row at all.
                estimate_scoped: request_geometry != measured_geometry
                    || self.adapter_gb > 0.0
                    || !self.runtime_verified
                    || !measured
                        .iter()
                        .any(|(strategy, _)| *strategy == selection.strategy),
            },
            Selection::Reject {
                needed_gb,
                available_gb,
            } if u64::from(request_geometry.width) * u64::from(request_geometry.height)
                <= u64::from(measured_geometry.width) * u64::from(measured_geometry.height)
                && self.adapter_gb == 0.0 =>
            {
                KreaControlFit::TooBig {
                    // Preserve the legacy outward-facing admission numbers while shared Budget arithmetic
                    // models the same headroom explicitly.
                    needed_gb: needed_gb + HEADROOM_GB,
                    available_gb: available_gb + HEADROOM_GB,
                }
            }
            // sc-18097 retired the `sequential_peak_gb.is_none()` ⇒ `Fits` arm that used to sit here —
            // the fail-open that silently ADMITTED sequential staging for a cell with no staged
            // measurement. With current evidence inside the measured envelope, every implemented rung
            // now carries an estimate-floor candidate, so that cell reaches `Selected`/`Reject` above
            // with an explicit estimate-graded verdict that CAN refuse when the floor plus the candle
            // estimate margin exceeds the budget. The `Unverified` arm below therefore remains only
            // for the cases the floors deliberately do not cover (superseded evidence, unverified
            // runtime, adapters, geometry above the measured envelope), where the pre-18097
            // best-effort never-reject contract still stands — now uniformly as `BestEffort`, never a
            // silent `Fits`. Adapter and larger-resolution estimates can now select a faster path,
            // while their no-fit result still preserves the recoverable fallback.
            // An extrapolated scalar floor can admit a roomy device, but it cannot prove that a
            // bounded high-resolution render is impossible. Keep the recoverable fallback if none
            // of those conservative estimates fit instead of refusing a previously runnable job.
            Selection::Unverified { .. } | Selection::Reject { .. } => {
                let (strategy, measured_peak_gb) = measured
                    .iter()
                    .min_by(|left, right| left.1.total_cmp(&right.1))
                    .copied()
                    .expect("resident evidence is always present");
                KreaControlFit::BestEffort {
                    offload_policy: OffloadPolicy::Sequential,
                    tile_vae_decode: contract.engages(strategy, MemoryStrategy::BoundedDecode),
                    chunk_attention: contract.engages(strategy, MemoryStrategy::BoundedAttention),
                    needed_gb: measured_peak_gb + HEADROOM_GB,
                    available_gb: budget.free_gb,
                }
            }
        }
    }
}
