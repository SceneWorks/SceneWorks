//! Candle/CUDA VRAM fit-gate + small-card emulation (epic 10765 Phase 0, sc-10766).
//!
//! The dynamic complement to the static per-tier `candle.minMemoryGb` manifest gate (sc-9094): before a
//! candle generation loads, predict the selected tier's peak VRAM and compare it against a LIVE budget
//! so a card that can't fit the model is rejected with an actionable message instead of being
//! SIGKILL'd / Metal-OOM'd mid-render. Also honors `SCENEWORKS_CUDA_VRAM_CAP_GB`, which emulates a
//! smaller card on big hardware so the Phase 1 offload paths (sc-10769) can be validated on the dev
//! box's 96 GB RTX PRO 6000s.
//!
//! ## Caching-allocator caveat (why this budgets on PREDICTED peak, not resident deltas)
//! candle's CUDA backend uses cudarc, with no `empty_cache` and no reclaim on `Device::synchronize()`.
//! A freed WITHIN-DEVICE component — the text encoders dropped before the DiT under Phase 1 sequential
//! residency, with the device still alive — returns to candle's in-process pool (reused by the next
//! allocation, which is what keeps a small card from OOMing) WITHOUT the DRIVER-resident `nvidia-smi`
//! free number dropping back. A full generator/device DROP is different: it destroys the device's pool
//! and returns its VRAM to the driver, so `free` RISES (GPU-measured, sc-13960). Either way this gate is
//! a pre-load ADMISSION check keyed off the manifest's measured per-tier peak — never a post-free
//! accounting number, since it runs while the model it is about to replace is still resident.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and
//! the wiring is in `generate_candle_stream` (image_jobs/base.rs).

#[path = "vram_gate/budget.rs"]
mod budget;
pub(crate) use budget::*;

#[path = "vram_gate/krea.rs"]
mod krea;
pub(crate) use krea::*;

#[path = "vram_gate/video.rs"]
mod video;
pub(crate) use video::*;

use super::*;
#[cfg(test)]
use gen_core::MemoryStrategy;
#[cfg(test)]
use serde_json::Value;

use crate::fit_gate::BYTES_PER_GIB;
pub(crate) use crate::fit_gate::{resolve_offload, FitDecision};

/// Emulate a smaller card: cap usable VRAM (GB). Set e.g. `SCENEWORKS_CUDA_VRAM_CAP_GB=10` to make the
/// fit-gate treat this GPU as a 10 GB card, so a too-big model is rejected (and, once Phase 1 lands,
/// offloaded) exactly as it would be on real small hardware. Unset / non-positive ⇒ use the real
/// live free VRAM.
pub(crate) const CUDA_VRAM_CAP_ENV: &str = "SCENEWORKS_CUDA_VRAM_CAP_GB";

/// Fixed transient/runtime headroom (GB) added on top of a per-tier MEASURED peak (`candle.vramGbByTier`)
/// to cover allocator slack + activation spikes not captured by the steady peak. Not added on top of
/// `candle.minMemoryGb`, which the manifest already pads over the measured peak (sc-9094).
/// Candle/CUDA's dedicated-VRAM allocator/context slack. This aliases the backend-neutral typed
/// policy so the generic gate, control gate, Mochi, and the web mirror cannot assign independent
/// meanings or values to the same dedicated pool.
pub(crate) const HEADROOM_GB: f64 = crate::fit_gate::dedicated_vram_reserve().gb;

/// The named margin the candle ladder's operational reserve carries ABOVE the measured idle
/// baseline (sc-22664, epic 22657 E4). The baseline is what the box holds before this load — the
/// CUDA context(s) and WDDM residency `nvidia-smi` reports as used — and this margin is the growth
/// of that residency the pre-load probe cannot yet see: the cuBLAS/cuDNN workspaces and module
/// images a first kernel launch materializes inside the context, which a cold worker pays once on
/// top of the modelled phase peaks. It is NOT allocator slack (candle's counted allocations carry
/// none — `ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`) and NOT activation headroom (the law
/// prices activations); it is only the context's own post-probe growth.
pub(crate) const LADDER_RESERVE_MARGIN_GB: f64 = 0.25;

// The margin is a fraction of the slack it replaces, by construction: a margin at or above the
// legacy reserve would make the ceiling in `ladder_reserve_gb` the whole answer.
const _: () = assert!(LADDER_RESERVE_MARGIN_GB > 0.0 && LADDER_RESERVE_MARGIN_GB < HEADROOM_GB);

/// The pre-load device residency a candle worker process MEASURABLY carries on this box, in bytes
/// (sc-22664 review, D3): the `preLoadDeviceUsed` diagnostic of the retained candle calibration
/// record `imc-06fbd2ff6dcba95f8555` (`docs/generated/memory-calibration-evidence.json`; Krea 2
/// Turbo q4, `memory-candle-adapter:krea-turbo-certifying`, RTX PRO 6000 Blackwell, captured
/// 2026-08-28) — the device's used bytes the harness read immediately before its load, i.e. the
/// CUDA context, module images and WDDM residency the process pays before the first weight lands.
/// The retained anchors measure every phase peak as a device DELTA above exactly this residency,
/// so it is the one term no derived peak contains.
///
/// It is the FLOOR of [`ladder_reserve_gb`]'s idle baseline rather than its whole answer because
/// the live probe can only see this process's context AFTER the context exists: on a cold worker
/// `total − free` is the card's residency before this process created its context, which on a
/// headless card is ~0, and a reserve derived from that alone would collapse to the bare margin
/// and under-charge the residency the record shows the process will carry.
///
/// `ladder_reserve_reads_the_measured_pre_load_residency_off_the_retained_record` reads this
/// constant back against the record, so it cannot drift from the evidence it cites.
pub(crate) const MEASURED_PRELOAD_RESIDENCY_BYTES: u64 = 1_045_430_272;

/// [`MEASURED_PRELOAD_RESIDENCY_BYTES`] in the budget's unit (GiB, what `nvidia-smi`'s MiB
/// reading divides down to): ~0.97.
pub(crate) const MEASURED_PRELOAD_RESIDENCY_GB: f64 =
    MEASURED_PRELOAD_RESIDENCY_BYTES as f64 / BYTES_PER_GIB;

// The measured residency plus the margin sits under the ceiling, so the ceiling is a genuine cap
// on foreign residency and not the answer on every idle card.
const _: () = assert!(MEASURED_PRELOAD_RESIDENCY_GB + LADDER_RESERVE_MARGIN_GB < HEADROOM_GB);

/// Fold this process's RECLAIMABLE VRAM into the live budget (sc-11023): add `reclaimable_gb` to
/// `free_gb`, clamped to the physical `total_gb`. The candle generator cache is a single exclusive slot
/// that evicts its current occupant BEFORE loading the incoming model. This budget is read while that
/// occupant is still resident (so raw `free` still counts the model it is about to replace); crediting
/// the reclaimable pool predicts the `free` the imminent evict will PRODUCE, so a warm same-model
/// re-gate or a swap-in isn't falsely rejected. That prediction holds whether evicting returns the VRAM
/// to the driver (GPU-measured sc-13960: a full generator drop frees most of it back — `nvidia-smi`
/// `free` rises) or a within-device component free keeps it pooled in-process — and the clamp to
/// `total_gb` is the safety net that keeps it honest against a stale high-water. A no-op when
/// `reclaimable_gb == 0` (nothing loaded yet), so a genuine cold first-load is gated exactly as before.
pub(crate) use crate::fit_gate::{with_reclaimable, VramBudget};

/// The resolved residency PLAN for a sequential-capable candle image lane: the pure composition of
/// [`fit_decision`] → [`resolve_offload`] → [`sequential_overflow_gb`] that base.rs's txt2img gate and
/// the bespoke Qwen-Edit gate share. Carries only the ACTION — never a budget-derived number — so two
/// plans computed against different budgets compare equal **iff they take the same action**. That is
/// the invariant the sc-13960 evict-then-reclaim two-pass leans on: gate once against raw free and once
/// against `free + reclaimable_pool`, and evict only when the plan actually improves (never for two
/// rejects that differ only in their reported free-VRAM number).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadPlan {
    /// Load at full residency — the whole-model resident peak fits (or the tier is unmeasured, so the
    /// gate never blocks).
    Resident,
    /// Load with sequential component residency — the resident peak overflows but the staged working
    /// set fits (or its peak is unmeasured, so best-effort staging).
    Sequential,
    /// Reject before OOM — the resident peak overflows AND either the measured sequential peak also
    /// overflows or the engine cannot stage.
    Reject,
}

/// The geometry a phase curve is evaluated at (sc-18812).
///
/// `frames` is the temporal axis the image lane never had. It is a separate type rather than a
/// second `u32` argument because the two axes are not interchangeable and a transposed call site
/// would otherwise compile: `pixels` is an AREA (already multiplied out) while `frames` is a count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CurveGeometry {
    pub pixels: u64,
    pub frames: u32,
}

/// Which phase carries the peak of a phase triple at one geometry (sc-18812).
///
/// Typed because the binding phase FLIPS inside a single model's envelope — on measured q8 LTX,
/// decode overtakes text between 11,904 and 14,080 latent tokens — so "which phase binds" is a
/// per-geometry question, not a per-model constant. sc-18829 needs to ask it of a geometry it has
/// no record for, which is exactly what a curve can answer and a record cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindingPhase {
    Text,
    Denoise,
    Decode,
}

/// Per-phase prediction for one Krea Turbo rung at the requested geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KreaTurboPhasePeaks {
    pub text_gb: f64,
    pub denoise_gb: f64,
    pub decode_gb: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum KreaTurboFit {
    Resident {
        peak_gb: f64,
        /// The admitted peak plus the operational reserve the selector charged once against the
        /// budget (`ladder_reserve_gb`, sc-22667) — what the lane quotes as its need.
        needed_gb: f64,
        selection: gen_core::MemorySelection,
        /// The selector's own admission figures (sc-22667, E7 telemetry).
        admitted: crate::candle_memory_strategy::AdmittedBudget,
    },
    Fits {
        phases: KreaTurboPhasePeaks,
        /// The admitted peak plus the operational reserve (see `Resident::needed_gb`).
        needed_gb: f64,
        selection: gen_core::MemorySelection,
        memory: gen_core::GenerationMemory,
        /// What priced the selected rung (sc-22667, E7): a measured record of this cell, the
        /// tier's fitted curve, or the anchor derivation through the image law.
        basis: crate::memory_strategy::CandidateBasis,
        /// The selector's own admission figures, so the telemetry agrees with the selector by
        /// construction.
        admitted: crate::candle_memory_strategy::AdmittedBudget,
        /// The rung was carried by a synthesized fitted-curve ESTIMATE, not by an exact measured
        /// record of this cell (sc-18097). True exactly when the request geometry has no
        /// `exact_request` record: a rung's measured candidates come only from that record, and a
        /// rung whose record exists but is structurally excluded emits no estimate either (the
        /// whole-tier fail-closed rule in `krea_turbo_fit_with_runtime`), so those are the only
        /// two states. Consumed by [`krea_turbo_smaller_fit_with_runtime`], which — mirroring
        /// `mlx_fit_gate::verified_lower_alternative` — must not offer an estimate-backed
        /// geometry as refusal advice.
        estimate_scoped: bool,
    },
    Reject {
        phases: KreaTurboPhasePeaks,
        needed_gb: f64,
    },
    Unverified {
        reason: gen_core::MemoryEvidenceVerdict,
    },
}

// Krea curves retain their captured producer revision. New derived estimates instead use the
// linked inference provenance; neither stamp is a runtime currency gate (sc-22738).

#[derive(Clone, Debug)]
pub(crate) struct KreaRuntimeEvidenceContext {
    resolved_route: String,
    backend: String,
    gpu_id: String,
    compute_capability: f32,
    artifact_provider: String,
    artifact_repository: String,
    resolved_revision: String,
    tier_root: String,
    provider_contract: Option<gen_core::MemoryProviderContract>,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_fixtures;
