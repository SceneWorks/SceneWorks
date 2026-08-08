//! Krea pose-ControlNet VRAM fit ladder (sc-11754, epic 8459 → epic 10765).
//!
//! The Krea control lane is diverted around the base.rs `generate_candle_stream` fit-gate ([`crate::vram_gate`]):
//! it loads through the bespoke `Krea2Control` provider, not the shared txt2img path, so the epic-10765
//! admission check never sees it. This module is its dedicated fit-gate — the SAME live/capped budget
//! ([`crate::gpu::nvidia_vram_budget_gb`] + `SCENEWORKS_CUDA_VRAM_CAP_GB`) vs a control-lane-specific
//! predicted peak, submitted to the shared memory-strategy selector so the memory optimizations engage
//! only when VRAM is actually constrained: on a 96 GB card no rung engages (zero penalty); on a 24/16 GB
//! card the minimum set engages to fit. The branch's TIER is not among the rungs (sc-15799) — a q8
//! selection carries a q8 branch on the 96 GB card too.
//!
//! ## The shared strategy candidates
//! The control lane's peak is the base tier + the control branch (at the tier
//! [`gen_core::tier_integrity::control_branch_tier`] assigns it) + activations + the end-of-render
//! VAE-decode spike. Each directly measured peak is submitted as a candidate. Ordering and the
//! first-fitting choice are owned exclusively by [`crate::memory_strategy::select_strategy`]:
//!  1. **packed base** — default, ~free (candle-gen #480, shipped). Always on; the peak below already
//!     reflects it.
//!  2. **sequential residency** (sc-12176) — encode/drop Qwen3-VL before loading the heavy phase. The
//!     cheapest adaptation and no quality cost; its measured per-tier peak enables a clean second-stage
//!     reject when even the staged working set will not fit.
//!  3. **VAE-decode tiling** (sc-11744, candle-gen #492) — force the seam-free tiled VAE tail (a *speed*
//!     cost, no quality cost) to cap the end-of-render decode spike. Engaged after residency, when the
//!     measured `decodeTileSaveGb` is enough to fit; also stays on underneath the chunking rung (it is
//!     cheaper and every bit helps).
//!  4. **activation chunking** (sc-11745, candle-gen #496) — engage sc-6217-style query-row attention
//!     chunking on the composable base stack + control branch when the denoise steady state is the
//!     overflow. A *speed* cost (~+6%) with byte-identical output; the deepest rung this lane has.
//!     Gated by the presence of the measured scalar `chunkAttnSaveGb`.
//!
//! ## An unpriced TIER is not "no signal" (sc-16069)
//!
//! A tier with no `peakGbByTier` row used to collapse into [`KreaControlFit::Unknown`], which
//! the lane maps to the big-card fast path with no log — so the gate neither staged, nor tiled, nor
//! chunked, nor rejected, nor left a trace. [`fit_ladder_for_entry`] now separates the two cases: no
//! `candle.control` block at all is still `Unknown`, while a block that carries no row for the resolved
//! tier is [`KreaControlFit::Unverified`] — an explicit, named, logged decision that stages residency and
//! still never rejects. SC-16013 now prices every hosted tier, including `int8-convrot`; this remains a
//! fail-safe for malformed catalog entries and future tiers, so it keys on the missing row rather than a
//! particular tier name.
//!
//! `decodeTileSaveGb` / `chunkAttnSaveGb` absent (unmeasured) ⇒ no MEASURED candidate exists for that
//! rung. Since sc-18097 (epic 18093 R1b) a current-evidence ladder inside the measured 1024² envelope
//! fills those gaps with estimate-FLOOR candidates (the staged row unreduced — never a promised
//! unmeasured saving), graded by the shared selector behind the candle estimate margin — so an
//! incomplete provider ladder yields an explicit estimate-graded verdict that can honestly REJECT
//! when even the widened floors overflow the budget. [`KreaControlFit::BestEffort`] (admit and let the
//! recoverable CUDA-OOM backstop decide) remains the outcome only where the floors deliberately do
//! not apply: superseded evidence (`candle.control.measured` `false` — a stale upper bound can refuse
//! a job that would run, see [`fit_ladder`]), an unverified runtime artifact, adapter overlays, or a
//! request above the measured envelope.
//!
//! ## The branch tier is NOT a rung (sc-15799)
//!
//! There used to be a fifth, last-resort rung: quantize the control branch bf16 → q8 → q4. It is gone.
//! **No component is resident above the user's selected tier unless a declared, measured exception says
//! otherwise** (`docs/memory-strategy-contract.md`, "Tier integrity"), so the branch's tier follows the
//! base tier — with the one declared exception that a q4 base floors its branch at q8, because a q4
//! control residual measures "pose-locked; non-pose details drift". As a rung it sat LAST, which meant a
//! constrained card staged the text phase, tiled its decode and chunked its attention to claw back
//! single-digit GB while ~3.3 GB of precision a q8 render never asked for sat in the branch the whole
//! time (the branch is 3.30 B params ≈ 6.6 GB bf16, ~3.3 GB at q8, ~1.7 GB at q4).
//! [`control_branch_tier_for_key`] is now the whole decision, and it reads no budget.
//!
//! ### Current calibration (sc-16013)
//!
//! The shipped `peakGbByTier` and `sequentialPeakGbByTier` rows were measured directly against each
//! tier's shipping branch on RTX PRO 6000 Blackwell at 1024² / 8 steps / control scale 0.6. They include
//! q4, q8, bf16, and INT8-ConvRot, so every reachable tier has a priced fast path and staged path. The
//! worker reads those rows verbatim and adds only the shared admission headroom; there is no branch
//! packing correction. The generic `BestEffort` path remains for any future superseded entry, while the
//! current Krea evidence is eligible for a hard reject after the deepest measured rung.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and the
//! wiring is in `generate_candle_krea_control_stream` (image_jobs/krea_control_candle.rs).

use super::*;
use gen_core::{
    MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
    MemoryEvidenceVerdict, MemoryGeometry, MemoryNumericTier, MemoryParityContract,
    MemoryParityResult, MemoryProviderContract, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, OffloadPolicy, Precision, Quant,
};
use serde_json::Value;

use crate::memory_strategy::{Budget, Candidate, RequestScope, Selection};

/// Fixed transient/runtime headroom (GB) added on top of the control-lane peak
/// ([`predicted_control_peak_gb`]), mirroring [`crate::vram_gate`]'s `HEADROOM_GB` — covers allocator
/// slack + activation spikes not captured by the steady peak.
const HEADROOM_GB: f64 = crate::vram_gate::HEADROOM_GB;

const KREA_CONTROL_ROUTE: &str = "krea_2_turbo_control";
const KREA_CONTROL_CALIBRATION: &str = "sc-16013-krea-control-direct-1024-v1";
/// Evidence-revision stamp for the synthesized estimate-floor candidates (sc-18097): request-scoped
/// telemetry must be attributable to the estimate synthesis, never to the sc-16013 measured rows it
/// floors on.
const KREA_CONTROL_ESTIMATE_REVISION: &str = "sc-18097-krea-control-estimate-floor-v1";
const KREA_CONTROL_ATTN_CHUNK_SIZE: u32 = 128 * 1024 * 1024;
const KREA_CONTROL_DECODE_TILE_EDGE: u32 = 512;
const KREA_CONTROL_DECODE_OVERLAP: u32 = 128;

/// The outcome of walking the Krea control fit ladder. `Unknown` = no signal (no `candle.control` block,
/// or a non-NVIDIA host) ⇒ never block — exactly like [`crate::vram_gate::FitDecision::Unknown`].
///
/// No variant carries the branch tier: it is not a rung outcome (sc-15799). The caller derives it from
/// the resolved base tier with [`control_branch_tier_for_key`] on *every* path, `Unknown` included, so a
/// missing manifest signal can never leave an above-tier branch resident.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KreaControlFit {
    /// No measured control peak or no live budget ⇒ don't gate.
    Unknown,
    /// The predicted peak fits after engaging the minimal sufficient set of rungs. The big-card fast
    /// path uses resident components with `{ tile_vae_decode: false, chunk_attention: false }`
    /// (monolithic full-speed decode, unchunked attention, zero penalty). Sequential residency is the
    /// first adaptation; `tile_vae_decode = true` is the next rung (sc-11744 — a *speed* cost only,
    /// seam-free); `chunk_attention = true` is the deepest (sc-11745 — a *speed* cost only,
    /// byte-identical output).
    Fits {
        /// Component residency selected before any deeper rung. Sequential is the cheapest adaptation:
        /// it drops Qwen3-VL before loading the DiT + control branch + VAE.
        offload_policy: OffloadPolicy,
        /// Force the seam-free tiled VAE decode (sc-11744) to cap the end-of-render decode spike. The
        /// worker threads this into `Krea2ControlRequest::tile_vae_decode`.
        tile_vae_decode: bool,
        /// Engage sc-6217-style query-row attention chunking (sc-11745) to bound the denoise activation
        /// peak. The worker threads this into `Krea2ControlPaths::chunk_attention` (a load-time toggle).
        chunk_attention: bool,
        /// The admission was carried by a synthesized estimate-floor candidate (sc-18097) rather
        /// than the measured 1024² cell — the request geometry is off the measured cell, or the
        /// selected rung has no measured row. [`incurred_peak_gb`] records NOTHING for such an
        /// admit: its true peak was not measured at this geometry, and crediting the 1024² row to
        /// the reclaimable pool would over-count it (the same never-over-count rule `BestEffort`
        /// already follows).
        estimate_scoped: bool,
    },
    /// Won't fit even at the deepest rung, and the prediction rests on CURRENT evidence
    /// ([`control_evidence_is_current`]). Reject-before-OOM with an actionable message rather than a
    /// reactive CUDA OOM mid-render. `needed_gb` is the best-case sequential + tiled + chunked predicted
    /// peak.
    TooBig { needed_gb: f64, available_gb: f64 },
    /// Won't fit even at the deepest rung, but the prediction rests on **superseded** evidence
    /// (`candle.control.measured == false` — see [`control_evidence_is_current`]), so rejecting would be
    /// a fit verdict derived from a stale upper bound. Every speed-only rung is engaged and the reactive
    /// CUDA-OOM backstop takes it from there; `needed_gb`/`available_gb` are for the log line only and
    /// must never be recorded as an incurred peak (see [`incurred_peak_gb`]).
    ///
    /// This is the same best-effort contract the lane already applies when the Sequential row is absent:
    /// stage as far as the measured rungs allow and let the runtime decide, rather than asserting a
    /// non-fit nobody measured.
    BestEffort {
        offload_policy: OffloadPolicy,
        tile_vae_decode: bool,
        chunk_attention: bool,
        needed_gb: f64,
        available_gb: f64,
    },
    /// The lane HAS a `candle.control` block, but it carries **no peak row for this base tier** — so the
    /// render cannot be priced at all (sc-16069).
    ///
    /// This used to collapse into [`KreaControlFit::Unknown`], which the caller maps to the big-card fast
    /// path with no log: the gate neither staged residency, nor tiled decode, nor chunked attention, nor
    /// rejected, nor said anything. An unpriced tier on a lane that *is* otherwise measured is not "no
    /// signal about the control lane" — it is a specific, nameable coverage hole, and silently taking the
    /// zero-adaptation path on it is the most permissive choice available.
    ///
    /// The shipped direct measurements now price every hosted tier, including `int8-convrot`. A future
    /// or malformed tier such as `nvfp4` can still lack a measured control-peak row. Keying this on
    /// "block present, row
    /// missing" rather than on a tier NAME covers both, and any future tier added to `vramGbByTier`
    /// without a matching control row.
    ///
    /// The verdict engages **sequential residency only** — the cheapest adaptation, zero quality cost —
    /// and never rejects, because there is no evidence to reject on. The hard check for this tier is the
    /// shared [`crate::conditioning_fit`] weights floor, which
    /// `generate_candle_krea_control_stream` runs in its preamble BEFORE this ladder (it must precede the
    /// lane's own `note_loaded_peak`; see `ConditioningAdmission::GatedInPreamble`). So an unpriced tier is
    /// not unguarded — it is guarded by the floor alone, with no measured peak on top.
    ///
    /// Staging is the SAME contract this ladder already applies when the Sequential row alone is missing:
    /// stage anyway and let the runtime decide. The knobs coincide with `Fits { Sequential, false, false }`
    /// on purpose; only the verdict and its log line differ, exactly as `BestEffort` does — and, like
    /// `BestEffort`, it records no reclaimable peak ([`incurred_peak_gb`]). Note this DOES change behavior
    /// for an unpriced tier that previously took the resident fast path: it now stages, a speed-only cost
    /// (one text re-encode) accepted because an unpriced tier is exactly where a resident load is least
    /// justified.
    Unverified {
        offload_policy: OffloadPolicy,
        tile_vae_decode: bool,
        chunk_attention: bool,
        /// The base tier key that has no control-lane peak row — named in the log so the coverage hole
        /// is identifiable rather than anonymous.
        tier_key: String,
    },
}

/// The [`Quant`] a control-lane tier KEY names — the `candle.control.*ByTier` / `vramGbByTier` key space
/// (`bf16` / `q8` / `q4` / `nvfp4` / `int8-convrot`), not a bit width.
///
/// `int8-convrot` (sc-9300) maps to [`Quant::Q8`]: it is an int8 weight regime with online rotation, so
/// for the purpose of "what precision may a component be resident at" it sits exactly where q8 does. An
/// unrecognized key is treated as dense, the conservative direction — a tier we cannot classify must not
/// be reported as *below* something, or the branch would be packed further than the base.
fn quant_for_tier_key(tier_key: &str) -> Option<Quant> {
    match tier_key {
        "q8" | "int8-convrot" => Some(Quant::Q8),
        "q4" => Some(Quant::Q4),
        "nvfp4" => Some(Quant::Nvfp4),
        _ => None,
    }
}

/// The tier the pose-control branch is packed to for a resolved base tier KEY (sc-15799) — the shared
/// [`gen_core::tier_integrity::control_branch_tier`] rule, keyed off the manifest's tier names.
///
/// This is the whole branch-tier decision on the candle lane. It reads no budget and returns the same
/// answer on a 16 GB card and a 96 GB card, because the user's tier choice — not free VRAM — is what
/// decides how much precision the control residual is allowed to hold. `None` = dense branch, which
/// happens only for a dense base.
pub(crate) fn control_branch_tier_for_key(tier_key: &str) -> Option<Quant> {
    gen_core::tier_integrity::control_branch_tier(quant_for_tier_key(tier_key))
}

/// Read one `candle.control.<key>[tier_key]` number.
fn control_tier_gb(manifest_entry: &JsonObject, key: &str, tier_key: &str) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get(key)
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
}

/// Does this entry declare a `candle.control` block at all?
///
/// The discriminator between the ladder's two no-price outcomes (sc-16069): **no block** means the model
/// has no measured control lane, which is genuinely no signal ⇒ [`KreaControlFit::Unknown`]; **a block
/// with no row for the resolved tier** means a measured lane with an unpriced tier ⇒
/// [`KreaControlFit::Unverified`], an explicit decision rather than a silent no-op. Deliberately does NOT
/// consult `measured` / `supersededBy` — a superseded block is still a block, and
/// [`control_evidence_is_current`] is the separate question of whether its numbers may harden into a
/// reject.
pub(crate) fn control_block_present(manifest_entry: &JsonObject) -> bool {
    manifest_entry
        .get("candle")
        .and_then(|candle| candle.get("control"))
        .is_some()
}

/// Is `candle.control`'s evidence CURRENT — i.e. captured for the configuration that actually ships?
/// `candle.control.measured == true` **and** no `supersededBy`. Both fields are consulted, because a
/// block that claims `measured: true` while still naming what superseded it is self-contradictory and
/// must be read the conservative way.
///
/// The shipped `krea_2_turbo` block is current after sc-16013. This remains load-bearing for future or
/// synthetic stale entries: [`fit_ladder`] will not turn superseded evidence into a hard **reject**.
/// The closure digest `candle.control`'s directly measured rows were captured under (sc-17774).
///
/// Empty when absent, which fails closed at the selector: no declared digest means nothing states
/// what code these numbers describe, so they cannot be current against anything.
pub(crate) fn control_closure_digest(manifest_entry: &JsonObject) -> &str {
    manifest_entry
        .get("candle")
        .and_then(|candle| candle.get("control"))
        .and_then(|control| control.get("inferenceClosureDigest"))
        .and_then(Value::as_str)
        .unwrap_or_default()
}

pub(crate) fn control_evidence_is_current(manifest_entry: &JsonObject) -> bool {
    let Some(control) = manifest_entry.get("candle").and_then(|c| c.get("control")) else {
        return false;
    };
    control.get("measured").and_then(Value::as_bool) == Some(true)
        && control.get("supersededBy").is_none()
}

/// Predicted control-lane peak VRAM (GB) for `tier_key` (the BASE tier — `bf16`/`q8`/`q4`/…) with no
/// rungs engaged: `candle.control.peakGbByTier[tier_key]` **verbatim**, plus [`HEADROOM_GB`];
/// `None` (absent ⇒ the gate no-ops). The control peak exceeds the txt2img `candle.vramGbByTier`
/// because the control branch is co-resident.
///
/// **The row is read verbatim, on purpose.** sc-16013 measured every reachable tier against the branch
/// tier that actually ships, so the catalog value is already the complete device peak. The only
/// arithmetic here is the standard admission headroom.
///
/// **What is deliberately not done.** An earlier revision of this module subtracted
/// `candle.control.branchPackSaveGb[branch tier]` (q8 8.4 / q4 10.2 GB) to "convert" the row into the
/// shipping configuration. That is arithmetically unsound and it under-predicted by ~5 GB toward an
/// OOM: the branch's projections are 3.30 B params ≈ **6.6 GB** bf16 and ~3.3 GB at q8 / ~1.7 GB at q4
/// (`candle_gen_krea::control::ControlBranch::from_checkpoint_quantized`), so the true weight-side
/// deltas are 3.3 GB (bf16→q8) and 4.9 GB (bf16→q4). 8.4 exceeds the entire branch, so it was never a
/// weight-side quantity and cannot be subtracted from a peak. `branchPackSaveGb` is retracted in the
/// catalog and this module does not read it. The current direct measurements need no replacement
/// correction.
pub(crate) fn predicted_control_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let measured_row = control_tier_gb(manifest_entry, "peakGbByTier", tier_key)?;
    Some(measured_row.max(0.0) + HEADROOM_GB)
}

/// Predicted Sequential control-lane peak for `tier_key`:
/// `candle.control.sequentialPeakGbByTier[tier_key]`, read and headroomed exactly like
/// [`predicted_control_peak_gb`] — verbatim. This is the largest single working
/// set after Qwen3-VL has been encoded and dropped. `None` preserves best-effort offload behavior: the
/// lane still stages, but cannot perform an honest second-stage reject.
pub(crate) fn predicted_control_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let measured_row = control_tier_gb(manifest_entry, "sequentialPeakGbByTier", tier_key)?;
    Some(measured_row.max(0.0) + HEADROOM_GB)
}

/// Measured peak reduction (GB) from forcing the seam-free tiled VAE decode for BASE tier `tier_key`
/// (sc-11744, candle-gen #492): `candle.control.decodeTileSaveGb[tier_key]` — the monolithic whole-render
/// peak minus the tiled one (the decode spike capped toward the denoise steady state). `None` when
/// unmeasured for that tier ⇒ the VAE-tiling rung is unavailable there and the ladder walks straight to
/// attention chunking. Tier-keyed like [`predicted_control_peak_gb`] because whether the decode spike
/// *is* the global peak (and thus how much tiling recovers) depends on the tier's denoise-steady floor.
///
/// sc-16013 re-measured this delta on the shipping q4-base/q8-branch staged load: 29.6 → 22.4 GB,
/// a 7.2 GB whole-render saving.
pub(crate) fn decode_tile_save_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    control_tier_gb(manifest_entry, "decodeTileSaveGb", tier_key)
}

/// Measured peak reduction (GB) from engaging sc-6217-style query-row attention chunking on the composable
/// base stack + control branch (sc-11745, candle-gen #496): `candle.control.chunkAttnSaveGb`. Bounds the
/// DENOISE-phase activation peak — a *speed* cost (~+6%) with byte-identical output (the chunked forward is
/// numerically identical to the unchunked one, the sc-6217 query-row-independence invariant). Unlike the
/// tier-keyed [`decode_tile_save_gb`]/[`predicted_control_peak_gb`], this is a SCALAR: the denoise activation
/// peak is bf16 regardless of the base weight tier, so the saving is tier-independent. `None` when unmeasured
/// ⇒ the chunking rung is unavailable and the ladder can only reject. sc-16013 re-measured the scalar
/// on the shipping q4-base/q8-branch staged+tiled load: 22.4 → 19.6 GB, a 2.8 GB saving, with
/// byte-identical raw RGB.
pub(crate) fn chunk_attn_save_gb(manifest_entry: &JsonObject) -> Option<f64> {
    manifest_entry
        .get("candle")?
        .get("control")?
        .get("chunkAttnSaveGb")
        .and_then(json_f64)
}

fn control_numeric_tier(tier_key: &str) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: quant_for_tier_key(tier_key),
        component_precision_floors: &[],
    }
}

fn control_parameters(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> MemoryStrategyParameters {
    MemoryStrategyParameters {
        decode_tile_edge: contract
            .engages(strategy, MemoryStrategy::BoundedDecode)
            .then_some(KREA_CONTROL_DECODE_TILE_EDGE),
        decode_overlap: contract
            .engages(strategy, MemoryStrategy::BoundedDecode)
            .then_some(KREA_CONTROL_DECODE_OVERLAP),
        attention_chunk_size: contract
            .engages(strategy, MemoryStrategy::BoundedAttention)
            .then_some(KREA_CONTROL_ATTN_CHUNK_SIZE),
        ..Default::default()
    }
}

/// Submit the Krea control lane's directly measured resident, staged, tiled-decode, and
/// chunked-attention candidates to the one shared selector. This function reads evidence and maps the
/// selected shared strategy to the provider's legacy booleans; it does not own strategy order.
#[allow(clippy::too_many_arguments)]
fn fit_ladder_for_tier(
    contract: Option<&MemoryProviderContract>,
    tier_key: &str,
    request_geometry: MemoryGeometry,
    peak_gb: Option<f64>,
    sequential_peak_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    chunk_attn_save_gb: Option<f64>,
    evidence_is_current: bool,
    runtime_verified: bool,
    adapter_gb: f64,
    // The `candle.control.inferenceClosureDigest` these manifest rows were measured under
    // (sc-17774). Read from the manifest, never hardcoded.
    measured_closure_digest: &str,
) -> KreaControlFit {
    let (Some(peak), Some(budget)) = (peak_gb, budget) else {
        return KreaControlFit::Unknown;
    };
    let Some(contract) = contract else {
        return KreaControlFit::Unknown;
    };
    let numeric_tier = control_numeric_tier(tier_key);
    let measured_geometry = MemoryGeometry {
        width: 1024,
        height: 1024,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    let overlay = if adapter_gb > 0.0 {
        format!(
            "control_branch+adapters_bytes={}",
            (adapter_gb * crate::fit_gate::BYTES_PER_GIB).round() as u64
        )
    } else {
        "control_branch".to_owned()
    };
    let live_closure_digest = sceneworks_core::memory_calibration::packaged_closure_digest(
        "candle",
        "krea_2_turbo_control",
    )
    .unwrap_or_default();
    let request = RequestScope {
        resolved_route: KREA_CONTROL_ROUTE,
        backend: "candle",
        tier: numeric_tier,
        mode: "pose_control",
        overlay: Some(&overlay),
        geometry: request_geometry,
        // sc-17774: one mechanism. `unwrap_or_default` fails closed on an undeclared lane.
        expected_closure_digest: &live_closure_digest,
    };
    let bytes = |gb: f64| {
        (gb.max(0.0) * crate::fit_gate::BYTES_PER_GIB)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64
    };
    let make_evidence = |selection: MemorySelection, predicted_peak_gb: f64| {
        let calibration_matches = contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == KREA_CONTROL_CALIBRATION);
        let historical_verified = evidence_is_current && adapter_gb == 0.0;
        let dimensions = MemoryEvidenceDimensions {
            static_implementation: if contract.conformance_errors().is_empty()
                && contract.validate_selection(&selection).is_ok()
            {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::Invalid
            },
            declared_calibration: if calibration_matches {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::FingerprintMismatch
            },
            historical_verification: if historical_verified {
                MemoryEvidenceVerdict::Satisfied
            } else if evidence_is_current {
                MemoryEvidenceVerdict::OutOfEnvelope
            } else {
                MemoryEvidenceVerdict::Stale
            },
            current_environment_verification: if historical_verified
                && runtime_verified
                && request_geometry == measured_geometry
            {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::OutOfEnvelope
            },
            canonical_route_loadability: if runtime_verified {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::Unverified
            },
            exact_strategy_parameters: if contract.validate_selection(&selection).is_ok() {
                MemoryEvidenceVerdict::Satisfied
            } else {
                MemoryEvidenceVerdict::OutOfEnvelope
            },
        };
        let verified = dimensions.all_satisfied();
        MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: contract.load_shape,
                mode: crate::memory_strategy::memory_mode_from_mode_key("pose_control"),
                overlay: Some(overlay.clone()),
                geometry: measured_geometry,
                strategy: selection.strategy,
                engaged_composition: contract.engaged_composition(selection.strategy),
                parameters: selection.parameters,
            },
            conformance: if verified {
                MemoryConformanceState::Verified
            } else {
                MemoryConformanceState::ImplementedUnverified
            },
            dimensions,
            calibration_abi: contract
                .calibration
                .as_ref()
                .expect("control calibration")
                .abi,
            calibration_fingerprint: KREA_CONTROL_CALIBRATION.to_owned(),
            sceneworks_revision: "sc-16013".to_owned(),
            inference_revision: measured_closure_digest.to_owned(),
            harness_version: "krea-control-cuda-direct-v1".to_owned(),
            predicted_peak_bytes: bytes(predicted_peak_gb),
            // The manifest records the no-adapter rendered-device observation. Adapter bytes may extend
            // the prediction through the declared OverlayBytes variable, but never rewrite observation.
            observed_peak_bytes: Some(bytes((predicted_peak_gb - adapter_gb).max(0.0))),
            parity: MemoryParityContract::Exact,
            parity_result: if verified {
                MemoryParityResult::Passed
            } else {
                MemoryParityResult::NotRun
            },
        }
    };

    // Catalog helpers retain their legacy admission numbers (raw observation + HEADROOM_GB). Evidence
    // stores the raw rendered-device peak; the shared Budget removes headroom exactly once.
    let mut measured = vec![(MemoryStrategy::Resident, (peak - HEADROOM_GB).max(0.0))];
    if let Some(staged) = sequential_peak_gb {
        let staged = (staged - HEADROOM_GB).max(0.0);
        measured.push((MemoryStrategy::StagedResidency, staged));
        if let Some(tile_save) = decode_tile_save_gb {
            measured.push((MemoryStrategy::BoundedDecode, staged - tile_save));
        }
        if let Some(chunk_save) = chunk_attn_save_gb {
            let decode_save = if contract.engages(
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedDecode,
            ) {
                // Q4's provider contract composes attention chunking over tiled decode. Without the
                // decode row, the full engaged composition is not measured and must not become a
                // falsely verified attention candidate. Dense/Q8 contracts exclude that default edge,
                // so their independently measured chunking candidate remains valid without tiling.
                decode_tile_save_gb
            } else {
                Some(0.0)
            };
            if let Some(decode_save) = decode_save {
                measured.push((
                    MemoryStrategy::BoundedAttention,
                    staged - decode_save - chunk_save,
                ));
            }
        }
    }
    let selections = measured
        .iter()
        .map(|(strategy, _)| MemorySelection {
            strategy: *strategy,
            parameters: control_parameters(contract, *strategy),
            tier: numeric_tier,
        })
        .collect::<Vec<_>>();
    let evidence = selections
        .iter()
        .zip(&measured)
        .map(|(selection, (_, measured_peak_gb))| make_evidence(*selection, *measured_peak_gb))
        .collect::<Vec<_>>();
    // ── sc-18097 (epic 18093 R1b): estimate-floor candidates for every implemented rung. ──
    //
    // The control lane's mirror of the estimate ladder: manifest-row floors, never a promised
    // unmeasured saving. The resident floor is the measured `peakGbByTier` row (with adapters
    // already folded into `peak`); the staged floor is the measured `sequentialPeakGbByTier` row
    // where present, else the resident floor; every deeper rung takes the staged floor UNREDUCED —
    // selectable without promising an unmeasured saving. Where a rung's savings ARE measured, its
    // measured candidate exists in `measured` above and supersedes the floor in the selector, so
    // the priced 1024² ladder is byte-for-byte unchanged.
    //
    // Floors are synthesized only when every conjunct holds:
    //  * `evidence_is_current` — dimension-level staleness (`measured: false` / `supersededBy`)
    //    means the rows were ALREADY known non-current when recorded; they still exclude and may
    //    not seed floors either (the `BestEffort` never-reject contract stays for them).
    //  * `runtime_verified` and no adapters — the rows price the shipped no-adapter artifact.
    //  * the loaded contract's calibration identity matches the sc-16013 rows (the sc-18096
    //    drifted-provider gate).
    //  * the request geometry is inside the measured 1024² envelope. The rows are whole-render
    //    peaks read VERBATIM: at or below the measured area every phase is at most its measured
    //    value, so the constant extrapolation is an upper bound and no binding-phase flip can
    //    exceed it — which is why these are `EstimateFloor` candidates outside
    //    `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE`'s fitted-curve scope (no per-phase
    //    decomposition exists to re-check). ABOVE the measured area a verbatim row under-predicts,
    //    so no floor is emitted and the pre-18097 best-effort contract stands there.
    //
    // The candle estimate margin is applied by the shared selector, not here.
    let request_pixels = u64::from(request_geometry.width) * u64::from(request_geometry.height);
    let measured_pixels = u64::from(measured_geometry.width) * u64::from(measured_geometry.height);
    let contract_identity_matches = contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == KREA_CONTROL_CALIBRATION);
    let resident_floor_gb = (peak - HEADROOM_GB).max(0.0);
    let staged_floor_gb = sequential_peak_gb
        .map(|gb| (gb - HEADROOM_GB).max(0.0))
        .unwrap_or(resident_floor_gb);
    let mut estimates: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    if evidence_is_current
        && runtime_verified
        && adapter_gb == 0.0
        && contract_identity_matches
        && request_pixels <= measured_pixels
    {
        for strategy in MemoryStrategy::ALL {
            if !matches!(
                contract.capability(strategy).map(|cap| &cap.support),
                Some(gen_core::MemoryStrategySupport::Implemented)
            ) {
                continue;
            }
            let selection = MemorySelection {
                strategy,
                parameters: control_parameters(contract, strategy),
                tier: numeric_tier,
            };
            if contract.validate_selection(&selection).is_err() {
                continue;
            }
            let floor_gb = if strategy == MemoryStrategy::Resident {
                resident_floor_gb
            } else {
                staged_floor_gb
            };
            let predicted_peak_bytes = bytes(floor_gb);
            tracing::info!(
                route = KREA_CONTROL_ROUTE,
                backend = "candle",
                ?strategy,
                raw_peak_bytes = predicted_peak_bytes,
                "synthesized manifest-row floor estimate candidate"
            );
            estimates.push((
                selection,
                MemoryEvidence {
                    key: MemoryEvidenceKey {
                        resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                        backend: gen_core::MemoryBackend::Candle,
                        tier: numeric_tier,
                        load_shape: contract.load_shape,
                        mode: crate::memory_strategy::memory_mode_from_mode_key("pose_control"),
                        overlay: Some(overlay.clone()),
                        geometry: request_geometry,
                        strategy,
                        engaged_composition: contract.engaged_composition(strategy),
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
                    calibration_abi: contract
                        .calibration
                        .as_ref()
                        .expect("control calibration")
                        .abi,
                    calibration_fingerprint: KREA_CONTROL_CALIBRATION.to_owned(),
                    sceneworks_revision: KREA_CONTROL_ESTIMATE_REVISION.to_owned(),
                    inference_revision: measured_closure_digest.to_owned(),
                    harness_version: String::new(),
                    predicted_peak_bytes,
                    observed_peak_bytes: None,
                    parity: MemoryParityContract::Exact,
                    parity_result: MemoryParityResult::NotRun,
                },
            ));
        }
    }
    let mut candidates = selections
        .iter()
        .zip(&evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
            closure_digest: measured_closure_digest,
            basis: crate::memory_strategy::CandidateBasis::Measured,
        })
        .collect::<Vec<_>>();
    // A floor is a declaration under the LIVE closure — nothing there for currency to invalidate.
    candidates.extend(estimates.iter().map(|(selection, evidence)| Candidate {
        selection: *selection,
        evidence,
        closure_digest: &live_closure_digest,
        basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
    }));
    match crate::memory_strategy::select_strategy(
        request,
        contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            // A zero-free synthetic test budget still represents a real device whose total is not
            // zero. Preserve the old starved-card behavior while satisfying the shared budget type's
            // physical-total invariant.
            total_gb: budget.total_gb.max(f64::EPSILON),
            reserved_headroom_gb: HEADROOM_GB,
        }),
        &candidates,
    ) {
        Selection::Selected { selection, .. } => KreaControlFit::Fits {
            offload_policy: if selection.strategy == MemoryStrategy::Resident {
                OffloadPolicy::Resident
            } else {
                OffloadPolicy::Sequential
            },
            tile_vae_decode: contract.engages(selection.strategy, MemoryStrategy::BoundedDecode),
            chunk_attention: contract.engages(selection.strategy, MemoryStrategy::BoundedAttention),
            // sc-18097: a selection is estimate-scoped when it cannot be the measured 1024² cell —
            // the request geometry is off the measured cell (every measured candidate was
            // structurally excluded there), or the selected rung has no measured row at all.
            estimate_scoped: request_geometry != measured_geometry
                || !measured
                    .iter()
                    .any(|(strategy, _)| *strategy == selection.strategy),
        },
        Selection::Reject {
            needed_gb,
            available_gb,
        } => KreaControlFit::TooBig {
            // Preserve the legacy outward-facing admission numbers while shared Budget arithmetic
            // models the same headroom explicitly.
            needed_gb: needed_gb + HEADROOM_GB,
            available_gb: available_gb + HEADROOM_GB,
        },
        // sc-18097 retired the `sequential_peak_gb.is_none()` ⇒ `Fits` arm that used to sit here —
        // the fail-open that silently ADMITTED sequential staging for a cell with no staged
        // measurement. With current evidence inside the measured envelope, every implemented rung
        // now carries an estimate-floor candidate, so that cell reaches `Selected`/`Reject` above
        // with an explicit estimate-graded verdict that CAN refuse when the floor plus the candle
        // estimate margin exceeds the budget. The `Unverified` arm below therefore remains only
        // for the cases the floors deliberately do not cover (superseded evidence, unverified
        // runtime, adapters, geometry above the measured envelope), where the pre-18097
        // best-effort never-reject contract still stands — now uniformly as `BestEffort`, never a
        // silent `Fits`.
        Selection::Unverified { .. } => {
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

/// Registered control contract for a tier, resolved through a minimal ON-DISK snapshot.
///
/// Since the pin at cc5b30a9 the provider binds its rung composition to the snapshot's ACTUAL
/// packed tier (it reads `transformer/config.json`; "bind memory gates to loaded tiers"): a
/// nonexistent root reads as dense, which mis-shapes the q4 composition (the non-q4
/// attention/decode engagement exclusion would apply). Materialize the tier the test asks for.
#[cfg(any(test, doc))]
fn registered_contract_for_tier(tier_key: &str) -> Option<MemoryProviderContract> {
    // One directory per CALL: parallel tests sharing a per-process path race between this
    // writer and the provider's reader, which surfaces as a spurious `Unknown` ladder verdict.
    // The guard also takes the snapshot with it when the call returns — the previous
    // per-process path was never removed and had piled up 31,264 directories under `%TEMP%`
    // on one box, the single largest leaker there (sc-17641).
    let root = tempfile::Builder::new()
        .prefix(&format!("sw-krea-control-fit-{tier_key}-"))
        .tempdir()
        .ok()?;
    let transformer = root.path().join("transformer");
    std::fs::create_dir_all(&transformer).ok()?;
    let config = match tier_key {
        "q4" => Some(r#"{"quantization":{"bits":4,"group_size":64}}"#),
        "q8" => Some(r#"{"quantization":{"bits":8,"group_size":64}}"#),
        _ => None,
    };
    if let Some(text) = config {
        std::fs::write(transformer.join("config.json"), text).ok()?;
    }
    let mut spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(root.path().to_path_buf()));
    if let Some(quant) = quant_for_tier_key(tier_key) {
        spec = spec.with_quant(quant);
    }
    crate::inference_runtime::media()
        .memory_strategy_contract(KREA_CONTROL_ROUTE, &spec)
        .ok()
        .flatten()
}

#[cfg(any(test, doc))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_ladder(
    tier_key: &str,
    peak_gb: Option<f64>,
    sequential_peak_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
    decode_tile_save_gb: Option<f64>,
    chunk_attn_save_gb: Option<f64>,
    evidence_is_current: bool,
    measured_closure_digest: &str,
) -> KreaControlFit {
    let contract = registered_contract_for_tier(tier_key);
    fit_ladder_for_tier(
        contract.as_ref(),
        tier_key,
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        peak_gb,
        sequential_peak_gb,
        budget,
        decode_tile_save_gb,
        chunk_attn_save_gb,
        evidence_is_current,
        true,
        0.0,
        measured_closure_digest,
    )
}

/// Walk the ladder for one catalog entry at one resolved BASE tier — **the seam the lane calls**
/// (sc-16069).
///
/// This exists so that the check no lane may forget cannot be written anywhere else: only the code that
/// reads the manifest can tell "this model has no measured control lane" (⇒ [`KreaControlFit::Unknown`],
/// genuinely no signal) apart from "this measured control lane has no row for the tier we resolved" (⇒
/// [`KreaControlFit::Unverified`], a specific coverage hole). Before this, the lane hand-threaded six
/// separate reads into [`fit_ladder`] and a missing row arrived as an indistinguishable `None`, so an
/// unpriced tier silently took the zero-adaptation big-card path — no staging, no tiling, no chunking, no
/// reject, no log. Pulling the reads in here also means a lane cannot accidentally pair one tier's peak
/// with another tier's savings.
///
/// The `Unverified` test is deliberately budget-INDEPENDENT: the row is absent whatever the card reports,
/// so the two-pass evict-reclaim gate can never turn it into anything else (and is told so, in
/// `krea_control_candle`'s `reclaim_improves`).
///
/// Pure: the caller resolves the budget, so the whole decision is unit-testable with no CUDA and no GPU.
#[cfg(any(test, doc))]
pub(crate) fn fit_ladder_for_entry(
    manifest_entry: &JsonObject,
    tier_key: &str,
    budget: Option<crate::vram_gate::VramBudget>,
) -> KreaControlFit {
    fit_ladder_for_entry_with_adapter_bytes(manifest_entry, tier_key, budget, 0)
}

/// Adapter-aware control ladder. Krea retains adapter overlays independently at every residency rung.
#[cfg(any(test, doc))]
pub(crate) fn fit_ladder_for_entry_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    budget: Option<crate::vram_gate::VramBudget>,
    adapter_bytes: u64,
) -> KreaControlFit {
    let contract = registered_contract_for_tier(tier_key);
    fit_ladder_for_entry_with_runtime(
        manifest_entry,
        tier_key,
        budget,
        adapter_bytes,
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        contract.as_ref(),
        true,
    )
}

/// Production Krea-control selector seam. Provider capabilities come from inference's registered
/// contract; the caller supplies actual request geometry and an artifact/GPU loadability verdict.
pub(crate) fn fit_ladder_for_entry_with_runtime(
    manifest_entry: &JsonObject,
    tier_key: &str,
    budget: Option<crate::vram_gate::VramBudget>,
    adapter_bytes: u64,
    geometry: MemoryGeometry,
    contract: Option<&MemoryProviderContract>,
    runtime_verified: bool,
) -> KreaControlFit {
    let adapter_gb = adapter_bytes as f64 / crate::fit_gate::BYTES_PER_GIB;
    let peak = predicted_control_peak_gb(manifest_entry, tier_key).map(|gb| gb + adapter_gb);
    if peak.is_none() && control_block_present(manifest_entry) {
        return KreaControlFit::Unverified {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            tier_key: tier_key.to_owned(),
        };
    }
    fit_ladder_for_tier(
        contract,
        tier_key,
        geometry,
        peak,
        predicted_control_sequential_peak_gb(manifest_entry, tier_key).map(|gb| gb + adapter_gb),
        budget,
        decode_tile_save_gb(manifest_entry, tier_key),
        chunk_attn_save_gb(manifest_entry),
        control_evidence_is_current(manifest_entry),
        runtime_verified,
        adapter_gb,
        control_closure_digest(manifest_entry),
    )
}

/// The VRAM peak (GB) a Krea control render actually incurs under `fit`, for the reclaimable high-water
/// ([`crate::vram_gate::note_loaded_peak`], sc-13960 — the control lane, unlike the txt2img/edit lanes,
/// recorded NO peak before this, so a repeated control render could not reclaim the first render's
/// dropped-but-pooled pages). Mirrors [`fit_ladder`]'s own arithmetic: the resident or staged base
/// peak, less each engaged speed rung's measured saving.
///
/// Returns `None` for a non-admit (`Unknown` / `TooBig`), for a [`KreaControlFit::BestEffort`] admit, for
/// an admitted-but-unmeasured peak, and — since sc-15799 — whenever
/// [`control_evidence_is_current`] is `false`. There is nothing honest to record in any of those cases,
/// exactly as base.rs records nothing for an unmeasured tier.
///
/// **Why superseded evidence records nothing.** This function must never OVER-count the pool the load
/// leaves behind: [`crate::vram_gate::with_reclaimable`] credits a later gate with it, so an over-count
/// over-admits an OOM. Superseded peak rows are read UNCORRECTED (an upper bound — see
/// [`predicted_control_peak_gb`]), which is the SAFE direction for admission and the UNSAFE direction
/// here: the load actually holds a packed branch, so the row exceeds the true incurred peak by the
/// branch-packing delta the catalog cannot yet price. The same number therefore cannot serve both roles,
/// and the reclaim credit is the one that must yield. The cost is real and bounded: until evidence
/// re-measures, a repeated control render re-stages instead of crediting the previous render's pooled
/// pages (needless staging — a *speed* cost), rather than crediting pages that are not there.
///
/// With current evidence, only the ENGAGED rungs' MEASURED savings are subtracted (a rung the ladder
/// could not have engaged has an unmeasured, `None` saving and contributes no phantom reduction). The
/// base peak carries [`HEADROOM_GB`] like the txt2img `predicted_peak_gb` this mirrors, so the small
/// headroom over-count is the same one base.rs's reclaim already tolerates (bounded by the next load's
/// own headroom).
#[cfg(any(test, doc))]
pub(crate) fn incurred_peak_gb(
    fit: &KreaControlFit,
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    incurred_peak_gb_with_adapter_bytes(fit, manifest_entry, tier_key, 0)
}

/// Reclaimable peak for an admitted adapter-aware control load.
pub(crate) fn incurred_peak_gb_with_adapter_bytes(
    fit: &KreaControlFit,
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    if !control_evidence_is_current(manifest_entry) {
        return None;
    }
    let KreaControlFit::Fits {
        offload_policy,
        tile_vae_decode,
        chunk_attention,
        estimate_scoped,
    } = fit
    else {
        // Unknown / TooBig — no admitted load. BestEffort — admitted, but above the budget the ladder
        // saw, so its peak is not a pool a later gate may credit. Unverified — admitted, but the tier has
        // no priced row at all, so there is no number to credit (sc-16069).
        return None;
    };
    if *estimate_scoped {
        // sc-18097: an estimate-floor admit's true peak was not measured at this geometry. The
        // 1024² rows read below OVER-state a smaller render's pool, and an over-stated pool lets a
        // later gate over-admit — the same never-over-count rule that makes `BestEffort` record
        // nothing.
        return None;
    }
    // The base peak the ladder admitted at: the resident whole-model peak, or the staged working set.
    let mut peak = if *offload_policy == OffloadPolicy::Resident {
        predicted_control_peak_gb(manifest_entry, tier_key)?
    } else {
        predicted_control_sequential_peak_gb(manifest_entry, tier_key)?
    };
    if *tile_vae_decode {
        peak -= decode_tile_save_gb(manifest_entry, tier_key).unwrap_or(0.0);
    }
    if *chunk_attention {
        peak -= chunk_attn_save_gb(manifest_entry).unwrap_or(0.0);
    }
    // No branch-tier term: past the guard above, `candle.control.measured` is true, so the rows describe
    // the configuration that actually loads and there is no packing delta left to apply.
    Some((peak + adapter_bytes as f64 / crate::fit_gate::BYTES_PER_GIB).max(0.0))
}

#[cfg(test)]
mod tests {
    /// The live digest for this lane, so a fixture is current unless a test deliberately says
    /// otherwise. Read, never frozen — a literal here would go stale on the next pin bump and the
    /// tests would silently stop exercising the admitted path.
    fn live_test_closure_digest() -> String {
        sceneworks_core::memory_calibration::packaged_closure_digest(
            "candle",
            "krea_2_turbo_control",
        )
        .unwrap_or_default()
    }

    use super::*;
    use crate::vram_gate::VramBudget;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    /// A Krea-shaped control block in the post-sc-15799 schema. Illustrative values, NOT the shipped
    /// ones — the shipped numbers are pinned by `tests/gpu_and_manifest.rs`.
    fn krea_manifest() -> JsonObject {
        obj(json!({
            "candle": {
                "minMemoryGb": 32,
                "vramGbByTier": { "q4": 26.4, "q8": 31.0, "bf16": 44.0 },
                "control": {
                    // Synthetic stale capture retained to exercise conservative evidence handling.
                    // a lighter branch can only lower a peak, so the row is a sound UPPER BOUND.
                    "peakGbByTier": { "q4": 30.9, "q8": 35.5, "bf16": 46.2 },
                    "sequentialPeakGbByTier": { "q4": 25.0, "q8": 30.0, "bf16": 40.0 },
                    // VAE-decode tiling saving (sc-11744); measured only for the constrained q4 tier here
                    // (the decode spike is the peak there).
                    "decodeTileSaveGb": { "q4": 6.9 },
                    // A STRAY retracted key, deliberately present in the base fixture so that EVERY
                    // test below is a mutation detector: nothing may read it, so reintroducing the
                    // branch-packing correction moves the numbers the whole ladder is asserted on. The
                    // shipped catalog no longer carries it at all (pinned by
                    // `tests/gpu_and_manifest.rs`), and the values are the retracted ones.
                    "branchPackSaveGb": { "q8": 8.4, "q4": 10.2 },
                    "measured": false,
                    "supersededBy": "sc-15799"
                }
            }
        }))
    }

    /// [`krea_manifest`] with the control block declaring CURRENT evidence — what sc-16013's re-measure
    /// will ship. A hard reject becomes available only when the fixture also carries evidence through
    /// the deepest implemented rung; current evidence also makes [`incurred_peak_gb`] recordable.
    fn krea_manifest_measured() -> JsonObject {
        current_evidence(krea_manifest())
    }

    /// [`krea_manifest`] plus a measured scalar `chunkAttnSaveGb` (sc-11745, candle-gen #496) — the
    /// deepest rung. `2.43` mirrors the shipped measured Δ (RTX PRO 6000, dense bf16 base, 1024²/8-step).
    fn krea_manifest_with_chunking() -> JsonObject {
        let mut m = krea_manifest();
        m.get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("control"))
            .and_then(Value::as_object_mut)
            .expect("control block")
            .insert("chunkAttnSaveGb".to_owned(), json!(2.43));
        m
    }

    /// Flip a fixture's control block to CURRENT evidence (what sc-16013 ships).
    ///
    /// sc-17774: "current" now includes the closure term, so this stamps the LIVE digest for the
    /// lane. Read rather than frozen — a literal would go stale on the next pin bump and every
    /// ladder test below would quietly become a staleness test instead.
    fn current_evidence(mut entry: JsonObject) -> JsonObject {
        let control = entry
            .get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("control"))
            .and_then(Value::as_object_mut)
            .expect("control block");
        control.insert("measured".to_owned(), json!(true));
        control.insert(
            "inferenceClosureDigest".to_owned(),
            json!(live_test_closure_digest()),
        );
        control.remove("supersededBy");
        entry
    }

    fn budget(free: f64) -> VramBudget {
        VramBudget {
            free_gb: free,
            total_gb: free,
        }
    }

    /// Keep the rung tests focused on tiling/chunking by modeling a staged peak equal to the resident
    /// peak, against CURRENT evidence. Fully measured fixtures can hard-reject; incomplete provider
    /// ladders remain best-effort. Dedicated tests below cover both floors.
    fn fit_ladder(
        peak: Option<f64>,
        budget: Option<VramBudget>,
        tile: Option<f64>,
        chunk: Option<f64>,
    ) -> KreaControlFit {
        super::fit_ladder(
            "q4",
            peak,
            peak,
            budget,
            tile,
            chunk,
            true,
            &live_test_closure_digest(),
        )
    }

    #[test]
    fn a_stale_control_closure_stays_eligible_at_the_widened_margin() {
        // sc-17774 gave this lane a real currency comparison (before it, the check compared
        // `KREA_CONTROL_INFERENCE_REVISION` against evidence this module had stamped with THE SAME
        // CONSTANT and could never fire); the outcome it pinned was a BestEffort fallback.
        // sc-18095 (epic 18093) turns that currency into a signal: a control ladder whose provider
        // closure moved keeps serving its measured rows, graded at the candle stale-measured
        // margin (`crate::ladder_margin_policy::CANDLE_STALE_MEASURED_MARGIN`, 2%), instead of
        // being demoted. Both digest sides are still read rather than frozen: `expected` is the
        // live digest for `candle:krea_2_turbo_control` from the packaged closure table, and the
        // candidate carries `candle.control.inferenceClosureDigest` from the manifest. This test
        // drives them apart.
        const STALE_DIGEST: &str =
            "0000000000000000000000000000000000000000000000000000000000000000";
        let fit = |free_gb: f64, digest: &str| {
            super::fit_ladder(
                "q4",
                Some(10.0),
                Some(10.0),
                Some(budget(free_gb)),
                Some(5.0),
                Some(2.0),
                true,
                digest,
            )
        };

        // Roomy budget: the stale ladder's widened resident peak (8.0 GiB evidence x 1.02) still
        // fits 18 GiB effective, so the stale outcome is byte-identical to the fresh one.
        let fresh = fit(20.0, &live_test_closure_digest());
        assert!(
            matches!(fresh, KreaControlFit::Fits { .. }),
            "control point: an unmoved closure must report a fit, or the assertions below prove \
             nothing about staleness: {fresh:?}"
        );
        let stale = fit(20.0, STALE_DIGEST);
        assert_eq!(
            stale, fresh,
            "a stale closure is a signal, not a gate (sc-18095): the roomy-budget fit must survive"
        );

        // The widening still discriminates: 10.1 GiB free (8.1 GiB effective) admits the RAW
        // resident/staged evidence peak (8.0 GiB) but not the widened one (~8.16 GiB), so the
        // fresh ladder stays resident while the stale ladder walks down to the measured
        // bounded-decode row (3.0 GiB evidence, widened ~3.06 GiB). A zeroed stale margin would
        // collapse the two outcomes — the mutation check for this lane.
        let fresh_tight = fit(10.1, &live_test_closure_digest());
        assert_eq!(
            fresh_tight,
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Resident,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: false,
            },
            "current evidence at the raw peak must keep the resident fit"
        );
        let stale_tight = fit(10.1, STALE_DIGEST);
        assert_eq!(
            stale_tight,
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                estimate_scoped: false,
            },
            "the stale ladder must be graded at the WIDENED peaks: resident/staged no longer fit, \
             the measured bounded-decode row does"
        );
    }

    /// The big-card fast path: monolithic decode, unchunked attention, nothing engaged.
    fn fits_nothing_engaged() -> KreaControlFit {
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: false,
        }
    }

    #[test]
    fn additive_adapter_bytes_shift_every_control_rung_and_incurred_peak() {
        let manifest = krea_manifest_measured();
        let tier = "q4";
        let one_gib = crate::fit_gate::BYTES_PER_GIB as u64;
        let resident = predicted_control_peak_gb(&manifest, tier).unwrap();

        assert_eq!(
            fit_ladder_for_entry_with_adapter_bytes(
                &manifest,
                tier,
                Some(budget(resident + 0.5)),
                0,
            ),
            fits_nothing_engaged()
        );
        assert_ne!(
            fit_ladder_for_entry_with_adapter_bytes(
                &manifest,
                tier,
                Some(budget(resident + 0.5)),
                one_gib,
            ),
            fits_nothing_engaged(),
            "a one-GiB adapter must cross the resident fit boundary"
        );

        let fit =
            fit_ladder_for_entry_with_adapter_bytes(&manifest, tier, Some(budget(96.0)), one_gib);
        let plain = incurred_peak_gb(&fits_nothing_engaged(), &manifest, tier).unwrap();
        let adapted = incurred_peak_gb_with_adapter_bytes(&fit, &manifest, tier, one_gib).unwrap();
        assert!((adapted - plain - 1.0).abs() < 1e-6);
    }

    /// Assert a [`KreaControlFit::TooBig`] whose `needed_gb` is ~`expected` (float-tolerant).
    fn assert_too_big(fit: KreaControlFit, expected_needed: f64, expected_free: f64) {
        match fit {
            KreaControlFit::TooBig {
                needed_gb,
                available_gb,
            } => {
                assert!(
                    (needed_gb - expected_needed).abs() < 1e-6,
                    "needed_gb {needed_gb} ≉ {expected_needed}"
                );
                assert!((available_gb - expected_free).abs() < 1e-6);
            }
            other => panic!("expected TooBig, got {other:?}"),
        }
    }

    // ── sc-15799: the branch tier is derived from the base tier, and reads no budget. ───────────────

    /// The invariant: the branch follows the base tier, with the one declared q4 → q8 floor. `int8-convrot`
    /// is an int8 regime so its branch is q8; a dense base keeps a dense branch. A mutation restoring the
    /// old "bf16 unless constrained" behavior cannot pass this — there is no budget argument to vary.
    #[test]
    fn branch_tier_follows_the_base_tier_with_the_declared_q4_floor() {
        assert_eq!(control_branch_tier_for_key("bf16"), None);
        assert_eq!(control_branch_tier_for_key("q8"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("q4"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("int8-convrot"), Some(Quant::Q8));
        assert_eq!(control_branch_tier_for_key("nvfp4"), Some(Quant::Q8));
        // An unclassifiable tier is treated as dense — never packed further than the base.
        assert_eq!(control_branch_tier_for_key("who-knows"), None);
    }

    /// A packed base never carries a dense branch. This is the defect sc-15799 removes, stated as a test.
    #[test]
    fn a_packed_base_never_carries_a_dense_branch() {
        for tier in ["q4", "q8", "int8-convrot", "nvfp4"] {
            assert!(
                control_branch_tier_for_key(tier).is_some(),
                "{tier} must not carry a dense control branch"
            );
        }
    }

    // ── The predicted peaks: the declared row is read verbatim. ─────────────────────────────────────

    /// The declared peak row is the value, on EVERY tier — no branch-packing correction. That is the
    /// sc-15799 review fix: subtracting `branchPackSaveGb` (q8 8.4 / q4 10.2) under-predicted the q4 and
    /// q8 hosts by ~5 GB straight into an OOM, because neither number is a weight-side quantity (the
    /// whole branch is 6.6 GB bf16 → ~3.3 GB q8 → ~1.7 GB q4, so the true deltas are 3.3 and 4.9).
    ///
    /// MUTATION PROOF. Reintroducing any correction moves the packed tiers away from their row and this
    /// test goes red. The asserted values are also checked against the physically-true host below.
    #[test]
    fn predicted_control_peak_reads_the_declared_row_without_correction() {
        let m = krea_manifest();
        for (tier, row) in [("q4", 30.9), ("q8", 35.5), ("bf16", 46.2)] {
            assert_eq!(
                predicted_control_peak_gb(&m, tier),
                Some(row + HEADROOM_GB),
                "{tier}: the superseded row is an UPPER BOUND and must be read verbatim"
            );
        }
        // The bound must never sit BELOW the physically-true host, which is the row less the real
        // weight-side branch delta (bf16 6.6 → q8 3.3 for both packed tiers, since q4 floors at q8).
        for (tier, row) in [("q4", 30.9), ("q8", 35.5)] {
            let physically_true = row - 3.3 + HEADROOM_GB;
            let predicted = predicted_control_peak_gb(&m, tier).expect("row present");
            assert!(
                predicted >= physically_true,
                "{tier}: predicted {predicted} must not under-predict the true host {physically_true} \
                 — an under-prediction admits an OOM"
            );
        }
        // Tier absent from the row ⇒ None (gate no-ops for that tier).
        assert_eq!(predicted_control_peak_gb(&m, "int8-convrot"), None);
        // No control block ⇒ None (unmeasured control lane).
        let no_control = obj(json!({ "candle": { "vramGbByTier": { "q4": 26.4 } } }));
        assert_eq!(predicted_control_peak_gb(&no_control, "q4"), None);
        // No candle block ⇒ None.
        assert_eq!(predicted_control_peak_gb(&obj(json!({})), "q4"), None);
    }

    /// A stray `branchPackSaveGb` left in a catalog entry must not move the prediction: nothing reads it
    /// any more. Pins the retraction against a partial revert. (The base fixture already carries the
    /// stray key for exactly this reason, so this test states the property the others merely rely on.)
    #[test]
    fn a_leftover_branch_pack_save_is_inert() {
        let m = krea_manifest();
        assert_eq!(
            m.get("candle")
                .and_then(|c| c.get("control"))
                .and_then(|c| c.get("branchPackSaveGb"))
                .and_then(|s| s.get("q8"))
                .and_then(json_f64),
            Some(8.4),
            "the fixture must actually carry the stray key, or this proves nothing"
        );
        assert_eq!(
            predicted_control_peak_gb(&m, "q4"),
            Some(30.9 + HEADROOM_GB),
            "branchPackSaveGb is retracted — reading it under-predicts by ~5 GB toward an OOM"
        );
        assert_eq!(
            predicted_control_sequential_peak_gb(&m, "q4"),
            Some(25.0 + HEADROOM_GB)
        );
    }

    #[test]
    fn predicted_control_sequential_peak_is_read_the_same_way() {
        let m = krea_manifest();
        assert_eq!(predicted_control_sequential_peak_gb(&m, "q4"), Some(27.0));
        assert_eq!(predicted_control_sequential_peak_gb(&m, "bf16"), Some(42.0));
        assert_eq!(predicted_control_sequential_peak_gb(&m, "nvfp4"), None);
        assert_eq!(
            predicted_control_sequential_peak_gb(&obj(json!({})), "q4"),
            None
        );
    }

    // ── sc-15799 review: `measured: false` / `supersededBy` are HONOURED, not ignored. ───────────────

    #[test]
    fn control_evidence_currency_reads_both_fields() {
        // Synthetic superseded shape.
        assert!(!control_evidence_is_current(&krea_manifest()));
        // Current shape: measured, nothing superseding it.
        assert!(control_evidence_is_current(&krea_manifest_measured()));
        // `measured: true` while still naming what superseded it is self-contradictory ⇒ not current.
        let mut contradictory = krea_manifest_measured();
        contradictory
            .get_mut("candle")
            .and_then(Value::as_object_mut)
            .and_then(|candle| candle.get_mut("control"))
            .and_then(Value::as_object_mut)
            .expect("control block")
            .insert("supersededBy".to_owned(), json!("sc-15799"));
        assert!(!control_evidence_is_current(&contradictory));
        // Absent block / absent flag ⇒ not current.
        assert!(!control_evidence_is_current(&obj(json!({}))));
        assert!(!control_evidence_is_current(&obj(
            json!({ "candle": { "control": {} } })
        )));
    }

    /// MUTATION PROOF for the review's `measured: false` finding: superseded evidence may not produce a
    /// hard reject, because the peaks it rests on are upper bounds and a reject from an upper bound can
    /// refuse a job that would run. Same numbers, same budget — only the evidence flag differs.
    #[test]
    fn superseded_evidence_never_rejects_and_current_evidence_does() {
        let peak = Some(40.0);
        let tile = Some(5.0);
        let chunk = Some(2.0);
        let starved = Some(budget(10.0)); // below every rung: 40 − 5 − 2 = 33
        assert_eq!(
            super::fit_ladder(
                "q4",
                peak,
                peak,
                starved,
                tile,
                chunk,
                false,
                &live_test_closure_digest()
            ),
            KreaControlFit::BestEffort {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
                needed_gb: 33.0,
                available_gb: 10.0,
            },
            "superseded evidence must adapt maximally, never assert a non-fit"
        );
        assert_too_big(
            super::fit_ladder(
                "q4",
                peak,
                peak,
                starved,
                tile,
                chunk,
                true,
                &live_test_closure_digest(),
            ),
            33.0,
            10.0,
        );
    }

    /// The best-effort floor engages only the rungs that are actually MEASURED — an unmeasured rung
    /// contributes no phantom flag, exactly as on the fitting paths.
    #[test]
    fn best_effort_engages_only_measured_rungs() {
        assert_eq!(
            super::fit_ladder(
                "q4",
                Some(40.0),
                Some(40.0),
                Some(budget(1.0)),
                None,
                None,
                false,
                &live_test_closure_digest(),
            ),
            KreaControlFit::BestEffort {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                needed_gb: 40.0,
                available_gb: 1.0,
            }
        );
    }

    /// Superseded evidence records NO reclaimable peak: the uncorrected row over-states what the packed
    /// load actually holds, and an over-stated pool lets a later gate over-admit.
    #[test]
    fn superseded_evidence_records_no_reclaimable_peak() {
        let resident = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: false,
        };
        assert_eq!(incurred_peak_gb(&resident, &krea_manifest(), "q4"), None);
        // With current evidence the row IS the load, so the credit is recordable again.
        assert_eq!(
            incurred_peak_gb(&resident, &krea_manifest_measured(), "q4"),
            Some(30.9 + HEADROOM_GB)
        );
    }

    #[test]
    fn decode_tile_save_reads_the_measured_delta() {
        let m = krea_manifest();
        assert_eq!(decode_tile_save_gb(&m, "q4"), Some(6.9));
        // Unmeasured for a tier ⇒ None (the tiling rung is unavailable there).
        assert_eq!(decode_tile_save_gb(&m, "bf16"), None);
        // No control block / no candle block ⇒ None.
        assert_eq!(decode_tile_save_gb(&obj(json!({})), "q4"), None);
    }

    #[test]
    fn chunk_attn_save_reads_the_measured_scalar() {
        // Present (chunk-enabled manifest) ⇒ the measured scalar; tier-independent (no tier arg).
        assert_eq!(
            chunk_attn_save_gb(&krea_manifest_with_chunking()),
            Some(2.43)
        );
        // Absent in the base manifest ⇒ None (the chunking rung is unavailable).
        assert_eq!(chunk_attn_save_gb(&krea_manifest()), None);
        // No control block / no candle block ⇒ None.
        assert_eq!(chunk_attn_save_gb(&obj(json!({}))), None);
    }

    // ── The ladder: staged residency → decode tiling → attention chunking, then reject. ─────────────

    #[test]
    fn sequential_is_the_first_rung_and_missing_deeper_evidence_is_estimate_graded() {
        let resident = Some(40.0);
        let sequential = Some(30.0);

        assert_eq!(
            super::fit_ladder(
                "q4",
                resident,
                sequential,
                Some(budget(35.0)),
                None,
                None,
                true,
                &live_test_closure_digest(),
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: false,
            }
        );
        // sc-18097 repin. Pre-18097 this was `BestEffort { needed_gb: 30.0, available_gb: 29.0 }`:
        // the unmeasured deeper rungs left the ladder "incomplete", which the selector refused to
        // harden into a reject. The estimate floors complete it — bounded decode/attention carry
        // the staged row UNREDUCED (28.0 evidence GiB, widened by the 4% candle estimate margin to
        // 29.12) — so nothing fits the 27 GiB effective budget and the honest outcome is the
        // reject, quoting the same measured staged requirement the old best-effort admit reported.
        assert_eq!(
            super::fit_ladder(
                "q4",
                resident,
                sequential,
                Some(budget(29.0)),
                None,
                None,
                true,
                &live_test_closure_digest(),
            ),
            KreaControlFit::TooBig {
                needed_gb: 30.0,
                available_gb: 29.0,
            }
        );
    }

    /// sc-18097: the `Unverified && sequential_peak_gb.is_none() ⇒ Fits` fail-open is GONE. A cell
    /// with no staged measurement used to silently ADMIT sequential staging; it now gets an
    /// explicit estimate-graded verdict. With no `sequentialPeakGbByTier` row the staged floor is
    /// the RESIDENT row unreduced (no unmeasured saving is ever promised), so on a card the
    /// resident peak overflows, the whole widened floor ladder overflows too and the fit REFUSES —
    /// the exact capability the old arm could not express.
    #[test]
    fn missing_sequential_measurement_no_longer_falls_open_to_a_silent_fit() {
        let fit = |free_gb: f64| {
            super::fit_ladder(
                "q4",
                Some(40.0),
                None,
                Some(budget(free_gb)),
                Some(5.0),
                Some(2.0),
                true,
                &live_test_closure_digest(),
            )
        };
        // 20 GiB free (18 effective): the resident evidence peak is 38 GiB and every deeper floor
        // equals it (no staged row), so the widened ladder (38 × 1.04 = 39.52) refuses. Before
        // sc-18097 this returned `Fits { Sequential, .. }` with no evidence at all.
        assert_eq!(
            fit(20.0),
            KreaControlFit::TooBig {
                needed_gb: 40.0,
                available_gb: 20.0,
            },
            "an unmeasured staged cell must be estimate-graded, not silently admitted"
        );
        // The same cell on a card the resident row fits is still admitted — the refusal above is
        // the margin's work, not a blanket reject of unmeasured staging.
        assert_eq!(
            fit(41.0),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Resident,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: false,
            }
        );
    }

    #[test]
    fn big_card_engages_nothing() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // 96 GB card: the monolithic peak fits outright — nothing engages.
        assert_eq!(
            fit_ladder(peak, Some(budget(90.0)), tile, chunk),
            fits_nothing_engaged()
        );
    }

    /// What the repack does NOT buy on this lane, pinned so nobody re-asserts it. The catalog's peaks
    /// were captured with a bf16 branch and are read verbatim, so a packed branch is invisible to the
    /// gate: a card sitting between the true packed host and the stale bound (here 26 GB against the
    /// 32.9 bound) still ladders. Claiming a benefit
    /// here would mean claiming a number the tree does not have.
    #[test]
    fn a_stale_row_does_not_receive_a_derived_repack_credit() {
        let m = krea_manifest();
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        assert_eq!(
            fit_ladder(
                predicted_control_peak_gb(&m, "q4"),
                Some(budget(26.0)),
                tile,
                chunk
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                estimate_scoped: false,
            },
            "the stale bf16-branch bound (32.9) still forces staging + tiling on a 26 GB card"
        );
    }

    #[test]
    fn vae_tiling_is_the_cheapest_deeper_rung() {
        let m = krea_manifest();
        // q4 base: monolithic bound 30.9 + 2.0 = 32.9; tiling saves 6.9 ⇒ tiled peak 26.0.
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // 27 GB card: monolithic (32.9) won't fit, the tiled decode (26.0) does ⇒ tiling on, no quality cost.
        assert_eq!(
            fit_ladder(peak, Some(budget(27.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: false,
                estimate_scoped: false,
            }
        );
    }

    #[test]
    fn chunking_is_the_deepest_rung_and_then_the_ladder_rejects() {
        let m = krea_manifest_with_chunking();
        // q4 base: bound 32.9; tiled 26.0; tiled + chunked 26.0 − 2.43 = 23.57.
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);

        // 24 GB card: tiling alone (26.0) > 24, tiling + chunking (23.57 ≤ 24) fits. Both rungs are
        // speed-only and byte-identical, so this card pays NO quality cost — where the old ladder
        // reached for a q4 branch at this budget.
        assert_eq!(
            fit_ladder(peak, Some(budget(24.0)), tile, chunk),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: true,
                chunk_attention: true,
                estimate_scoped: false,
            }
        );
        // 23 GB card: nothing deeper exists ⇒ reject-before-OOM at the honest best-case peak. (The
        // local helper passes `evidence_is_current = true`; the shipped block is superseded, so on the
        // real catalog this floor is `BestEffort` — see
        // `superseded_evidence_never_rejects_and_current_evidence_does`.)
        assert_too_big(
            fit_ladder(peak, Some(budget(23.0)), tile, chunk),
            23.57,
            23.0,
        );
    }

    #[test]
    fn chunking_engages_without_tiling_when_the_tier_has_no_tiling_measurement() {
        let m = krea_manifest_with_chunking();
        // The bf16 base tier carries no `decodeTileSaveGb` (the decode spike isn't its peak), but the
        // SCALAR chunk saving applies to every tier. Peak 48.2; chunked 48.2 − 2.43 = 45.77.
        let peak = predicted_control_peak_gb(&m, "bf16");
        let tile = decode_tile_save_gb(&m, "bf16"); // None
        let chunk = chunk_attn_save_gb(&m); // Some(2.43)
        assert_eq!(
            super::fit_ladder(
                "bf16",
                peak,
                peak,
                Some(budget(46.0)),
                tile,
                chunk,
                true,
                &live_test_closure_digest()
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: true,
                estimate_scoped: false,
            }
        );
    }

    #[test]
    fn q4_chunk_evidence_cannot_skip_its_missing_decode_composition_row() {
        let resident = Some(40.0);
        let staged = Some(40.0);
        let budget = Some(budget(36.0));

        // sc-18097 repin: previously `BestEffort { needed_gb: 40.0 }`. The missing decode row
        // still removes the full MEASURED attention candidate — its estimate floor carries the
        // staged row UNREDUCED (38 evidence GiB, widened 39.52), so a mutation that let the
        // chunk saving through without the decode row would admit 33 ≤ 34 effective and flip this
        // to `Fits { chunk_attention: true }`. With the floors complete, the honest outcome for a
        // ladder where nothing fits with margins is the reject.
        assert_eq!(
            super::fit_ladder("q4", resident, staged, budget, None, Some(5.0), true, &live_test_closure_digest()),
            KreaControlFit::TooBig {
                needed_gb: 40.0,
                available_gb: 36.0,
            },
            "Q4 attention engages decode, so missing decode evidence must remove the full candidate"
        );
        assert_eq!(
            super::fit_ladder(
                "bf16",
                resident,
                staged,
                budget,
                None,
                Some(5.0),
                true,
                &live_test_closure_digest()
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: true,
                estimate_scoped: false,
            },
            "dense attention is independently measured and excludes the default decode edge"
        );
    }

    #[test]
    fn missing_signal_never_blocks() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        let tile = decode_tile_save_gb(&m, "q4");
        let chunk = chunk_attn_save_gb(&m);
        // No live budget ⇒ Unknown (never block).
        assert_eq!(fit_ladder(peak, None, tile, chunk), KreaControlFit::Unknown);
        // No measured peak ⇒ Unknown.
        assert_eq!(
            fit_ladder(None, Some(budget(8.0)), tile, chunk),
            KreaControlFit::Unknown
        );
    }

    /// sc-18097 repin. The provider implements tiling and chunking; this synthetic cell measures
    /// neither. Pre-18097 the selector refused to "hard-reject from an incomplete ladder" and
    /// admitted best-effort. The estimate floors complete the ladder — every unmeasured rung
    /// carries the staged row UNREDUCED (30.9 evidence GiB, widened by the 4% candle estimate
    /// margin to 32.14) — so on an 18 GiB effective budget nothing fits with margins and the
    /// honest outcome is now the reject, quoting the same measured peak. This is a refusal from
    /// graded floors, not from absence: the widened floors are what the reject compares, and a
    /// budget that clears them still admits (second arm).
    #[test]
    fn unmeasured_rungs_carry_estimate_floors_and_refuse_only_below_the_widened_margin() {
        let m = krea_manifest();
        let peak = predicted_control_peak_gb(&m, "q4");
        assert_too_big(fit_ladder(peak, Some(budget(20.0)), None, None), 32.9, 20.0);
        // Control arm: the same unmeasured ladder on a card the resident row fits still admits at
        // the fast path — the floors grade, they do not blanket-refuse.
        assert_eq!(
            fit_ladder(peak, Some(budget(35.0)), None, None),
            fits_nothing_engaged()
        );
    }

    /// sc-13960: [`incurred_peak_gb`] records the peak a control render actually leaves in the cudarc
    /// pool — the resident/staged base peak less every ENGAGED rung's measured saving — so a later gate's
    /// reclaim credit is honest. It must mirror the ladder and never OVER-count (which would let a later
    /// gate over-admit an OOM).
    #[test]
    fn incurred_peak_mirrors_the_ladder_and_never_over_counts() {
        // CURRENT evidence: a superseded block records nothing at all (covered by
        // `superseded_evidence_records_no_reclaimable_peak`), so the mirroring is asserted on the shape
        // sc-16013 will ship.
        let m = current_evidence(krea_manifest_with_chunking());
        let tier = "q4";

        // Fast-path resident admit → exactly the resident control peak (with headroom).
        let resident = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: false,
        };
        assert_eq!(
            incurred_peak_gb(&resident, &m, tier),
            predicted_control_peak_gb(&m, tier) // 32.9
        );

        // Staged + tiled + chunked: the sequential peak (27.0) less every engaged measured saving.
        let laddered = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: true,
            chunk_attention: true,
            estimate_scoped: false,
        };
        let expected = predicted_control_sequential_peak_gb(&m, tier).unwrap()
            - decode_tile_save_gb(&m, tier).unwrap()
            - chunk_attn_save_gb(&m).unwrap(); // 27.0 − 6.9 − 2.43 = 17.67
        assert!((incurred_peak_gb(&laddered, &m, tier).unwrap() - expected).abs() < 1e-6);

        // Non-admits record nothing (no pooled pages to reclaim).
        assert_eq!(incurred_peak_gb(&KreaControlFit::Unknown, &m, tier), None);
        assert_eq!(
            incurred_peak_gb(
                &KreaControlFit::TooBig {
                    needed_gb: 50.0,
                    available_gb: 10.0
                },
                &m,
                tier
            ),
            None
        );

        // An admitted-but-unmeasured staged peak → None, not a guess (mirrors base.rs's None ⇒ no-op).
        let no_seq = obj(json!({
            "candle": { "control": {
                "peakGbByTier": { "q4": 30.9 },
                "measured": true
            } }
        }));
        let staged = KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: false,
        };
        assert_eq!(incurred_peak_gb(&staged, &no_seq, "q4"), None);

        // Safety: whatever the ladder ADMITS, the peak recorded for it never exceeds the budget it was
        // admitted against — so the pool the load leaves behind is never over-reported.
        let b = budget(24.0);
        let fit = super::fit_ladder(
            tier,
            predicted_control_peak_gb(&m, tier),
            predicted_control_sequential_peak_gb(&m, tier),
            Some(b),
            decode_tile_save_gb(&m, tier),
            chunk_attn_save_gb(&m),
            true,
            &live_test_closure_digest(),
        );
        if let Some(p) = incurred_peak_gb(&fit, &m, tier) {
            assert!(
                p <= b.free_gb + 1e-6,
                "incurred peak {p} must not exceed the admitting budget {}",
                b.free_gb
            );
        }
    }

    // ── sc-16069: an UNPRICED tier is an explicit decision, never a silent no-op. ───────────────────

    /// The defect, stated as a test. A `candle.control` block with NO peak row for the resolved tier used
    /// to arrive at [`fit_ladder`] as an indistinguishable `None` ⇒ [`KreaControlFit::Unknown`] ⇒ the
    /// caller's big-card fast path, with no staging, no rungs, no reject and no log. It must now be
    /// [`KreaControlFit::Unverified`], naming the tier.
    ///
    /// MUTATION PROOF: restoring the old collapse (dropping the `control_block_present` test from
    /// [`fit_ladder_for_entry`], or calling [`fit_ladder`] directly from the lane) makes this `Unknown`
    /// and the test goes red.
    #[test]
    fn an_unpriced_tier_on_a_measured_control_lane_is_unverified_not_unknown() {
        let m = krea_manifest();
        // Synthetic coverage for any control tier that has no measured peak row. The shipped manifest
        // now prices every hosted tier, but this generic behavior remains important for malformed and
        // future entries.
        for tier in ["nvfp4", "int8-convrot", "some-future-tier"] {
            assert_eq!(
                predicted_control_peak_gb(&m, tier),
                None,
                "{tier} is unpriced"
            );
            let fit = fit_ladder_for_entry(&m, tier, Some(budget(96.0)));
            assert_eq!(
                fit,
                KreaControlFit::Unverified {
                    offload_policy: OffloadPolicy::Sequential,
                    tile_vae_decode: false,
                    chunk_attention: false,
                    tier_key: tier.to_owned(),
                },
                "{tier}: an unpriced tier on a measured control lane must be an explicit verdict"
            );
            assert_ne!(fit, KreaControlFit::Unknown, "{tier} must not be a no-op");
        }
    }

    /// The verdict is budget-INDEPENDENT — the row is absent whatever the card reports. This is what lets
    /// `krea_control_candle`'s `reclaim_improves` skip a pointless generator evict for it: a second pass
    /// against a larger budget cannot produce anything different.
    #[test]
    fn unverified_does_not_move_with_the_budget() {
        let m = krea_manifest();
        let at = |free: Option<f64>| fit_ladder_for_entry(&m, "nvfp4", free.map(budget));
        let baseline = at(Some(96.0));
        for free in [Some(0.0), Some(8.0), Some(1024.0), None] {
            assert_eq!(
                at(free),
                baseline,
                "Unverified must not depend on the budget"
            );
        }
    }

    /// It must never REJECT. There is no evidence to reject on, and refusing a job that would run is the
    /// failure this lane is least allowed to have — even on a 0 GB budget, where a priced tier would be
    /// `TooBig`. (The shared `conditioning_fit` weights floor is what refuses a host that cannot hold the
    /// weights at all; this ladder declining to guess is the honest half.)
    #[test]
    fn unverified_never_rejects_even_on_a_starved_card() {
        let m = krea_manifest();
        let starved = fit_ladder_for_entry(&m, "nvfp4", Some(budget(0.0)));
        assert!(matches!(starved, KreaControlFit::Unverified { .. }));
        // Contrast: the SAME starved card on a PRICED tier with current evidence does reject, so this is
        // a property of the missing row, not of the budget being ignored everywhere.
        assert!(matches!(
            fit_ladder_for_entry(
                &current_evidence(krea_manifest_with_chunking()),
                "q4",
                Some(budget(0.0))
            ),
            KreaControlFit::TooBig { .. }
        ));
    }

    /// NO `candle.control` block at all is still genuinely no signal ⇒ [`KreaControlFit::Unknown`]. The
    /// two cases must stay distinct: a model with no measured control lane is not the same as a measured
    /// lane with an unpriced tier, and conflating them is what hid the hole.
    #[test]
    fn a_model_with_no_control_block_stays_unknown() {
        let no_control = obj(json!({ "candle": { "vramGbByTier": { "q4": 26.4 } } }));
        assert!(!control_block_present(&no_control));
        assert_eq!(
            fit_ladder_for_entry(&no_control, "q4", Some(budget(8.0))),
            KreaControlFit::Unknown
        );
        assert!(!control_block_present(&obj(json!({}))));
        assert_eq!(
            fit_ladder_for_entry(&obj(json!({})), "q4", Some(budget(8.0))),
            KreaControlFit::Unknown
        );
        // A SUPERSEDED block is still a block — `control_block_present` must not consult `measured`, or
        // today's `krea_2_turbo` (measured: false) would fall back into the silent `Unknown` path.
        assert!(control_block_present(&krea_manifest()));
        assert!(!control_evidence_is_current(&krea_manifest()));
    }

    /// An `Unverified` admit records NO reclaimable peak: there is no priced number to credit, and
    /// over-crediting the pool lets a later gate over-admit an OOM.
    #[test]
    fn unverified_records_no_reclaimable_peak() {
        let m = current_evidence(krea_manifest());
        let fit = fit_ladder_for_entry(&m, "nvfp4", Some(budget(96.0)));
        assert_eq!(incurred_peak_gb(&fit, &m, "nvfp4"), None);
    }

    /// The priced path is UNCHANGED by the new seam: `fit_ladder_for_entry` must agree with the hand-read
    /// `fit_ladder` on every tier that has a row. Pins that pulling the six reads into one place did not
    /// silently move any existing decision.
    #[test]
    fn the_entry_seam_matches_the_hand_read_ladder_on_priced_tiers() {
        for m in [
            krea_manifest(),
            krea_manifest_measured(),
            krea_manifest_with_chunking(),
            current_evidence(krea_manifest_with_chunking()),
        ] {
            for tier in ["q4", "q8", "bf16"] {
                for free in [0.0, 20.0, 24.0, 30.0, 35.0, 47.0, 96.0] {
                    let b = Some(budget(free));
                    assert_eq!(
                        fit_ladder_for_entry(&m, tier, b),
                        // `super::` — the module's real 6-arg ladder, not this test module's 4-arg helper.
                        // The seam derives its digest from the manifest, so the direct call must
                        // read the SAME one — otherwise this compares two different currency states
                        // rather than the ladder logic it exists to pin (sc-17774).
                        super::fit_ladder(
                            tier,
                            predicted_control_peak_gb(&m, tier),
                            predicted_control_sequential_peak_gb(&m, tier),
                            b,
                            decode_tile_save_gb(&m, tier),
                            chunk_attn_save_gb(&m),
                            control_evidence_is_current(&m),
                            control_closure_digest(&m),
                        ),
                        "{tier} @ {free} GB free: the seam must not change a priced decision"
                    );
                }
            }
        }
    }

    /// sc-18097: a request geometry off the measured 1024² cell — which excludes every measured
    /// candidate structurally and used to collapse into the never-reject `BestEffort` — is now
    /// estimate-graded inside the measured envelope: the manifest-row floors admit (or refuse)
    /// behind the candle estimate margin, the admit is flagged `estimate_scoped`, and
    /// [`incurred_peak_gb_with_adapter_bytes`] records NOTHING for it (the 1024² rows over-state a
    /// smaller render's pool). Above the envelope the rows are lower bounds, so the pre-18097
    /// best-effort contract still stands.
    ///
    /// Fixture rows (measured + chunking manifest, q4): resident evidence 30.9 GiB, staged 25.0 —
    /// floors take the staged row UNREDUCED for every deeper rung; widened ×1.04: resident
    /// 32.136, staged 26.0.
    #[test]
    fn an_unmeasured_geometry_is_estimate_graded_with_recoverable_oom_margins() {
        let m = current_evidence(krea_manifest_with_chunking());
        let tier = "q4";
        let contract = registered_contract_for_tier(tier);
        let geometry_768 = MemoryGeometry {
            width: 768,
            height: 768,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        let fit = |free_gb: f64| {
            fit_ladder_for_entry_with_runtime(
                &m,
                tier,
                Some(budget(free_gb)),
                0,
                geometry_768,
                contract.as_ref(),
                true,
            )
        };

        // Roomy card: the resident floor fits — the fast path, estimate-scoped, with NO
        // reclaimable-peak credit (recording the 1024² row for a 768² render would over-count).
        let roomy = fit(96.0);
        assert_eq!(
            roomy,
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Resident,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: true,
            }
        );
        assert_eq!(
            incurred_peak_gb_with_adapter_bytes(&roomy, &m, tier, 0),
            None,
            "an estimate-scoped admit must never credit the reclaimable pool"
        );

        // Constrained card: the staged floor's widened peak (26.0) fits a 28 GiB effective budget
        // where the resident floor (32.136) does not.
        assert_eq!(
            fit(30.0),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: true,
            }
        );

        // Margin mutation arm: at 27.5 GiB free (25.5 effective) the RAW staged floor (25.0) fits
        // but the widened one (26.0) does not — a zeroed estimate margin admits here and flips
        // this red. The reported requirement carries the widening (26.0 + 2 headroom;
        // float-tolerant, the widening rounds up in integer bytes).
        assert_too_big(fit(27.5), 28.0, 27.5);

        // The pre-18097 outcome for this cell was BestEffort at ANY budget — the starved card now
        // gets the honest refusal instead of an admit that could only OOM.
        assert!(matches!(fit(10.0), KreaControlFit::TooBig { .. }));

        // ABOVE the measured envelope the rows under-predict, so no floors are emitted and the
        // never-reject best-effort contract stands unchanged.
        let beyond = fit_ladder_for_entry_with_runtime(
            &m,
            tier,
            Some(budget(10.0)),
            0,
            MemoryGeometry {
                width: 1536,
                height: 1536,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            contract.as_ref(),
            true,
        );
        assert!(
            matches!(beyond, KreaControlFit::BestEffort { .. }),
            "beyond the measured envelope the best-effort contract must survive: {beyond:?}"
        );
    }

    /// sc-13960: on a warm worker, crediting the cudarc pool the previous control render left behind
    /// FLIPS the ladder off its needless rungs — the repeated-control-render scenario the story names.
    /// Pins the arithmetic the two-pass evict-reclaim gate performs (`fit_ladder(raw)` vs
    /// `fit_ladder(with_reclaimable(raw, pool))`).
    #[test]
    fn reclaim_flips_a_warm_control_gate_back_to_resident() {
        let m = krea_manifest();
        let tier = "bf16"; // control bound 48.2, sequential 42.0, no tiling rung for this tier
        let peak = predicted_control_peak_gb(&m, tier);
        let seq = predicted_control_sequential_peak_gb(&m, tier);
        let tile = decode_tile_save_gb(&m, tier); // None
        let chunk = chunk_attn_save_gb(&m); // None

        // Second bf16 control render on a 96 GB card: the first (48.2 peak) dropped but its pages are
        // pooled, so RAW free is ~47.8 GB — the resident peak no longer fits, so the ladder needlessly
        // stages sequentially.
        let raw = VramBudget {
            free_gb: 47.8,
            total_gb: 96.0,
        };
        assert_eq!(
            super::fit_ladder(
                "q4",
                peak,
                seq,
                Some(raw),
                tile,
                chunk,
                true,
                &live_test_closure_digest()
            ),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Sequential,
                tile_vae_decode: false,
                chunk_attention: false,
                estimate_scoped: false,
            }
        );
        // Crediting the 48.2 GB the first render left in-pool readmits it at the big-card fast path.
        let reclaimed = crate::vram_gate::with_reclaimable(raw, 48.2);
        assert_eq!(
            super::fit_ladder(
                "q4",
                peak,
                seq,
                Some(reclaimed),
                tile,
                chunk,
                true,
                &live_test_closure_digest()
            ),
            fits_nothing_engaged()
        );

        // A cold pool (reclaimable 0) is a no-op: the raw plan stands (a genuine first-load on a
        // constrained card is gated exactly as before).
        let reclaimed_cold = crate::vram_gate::with_reclaimable(raw, 0.0);
        assert_eq!(
            super::fit_ladder(
                "q4",
                peak,
                seq,
                Some(reclaimed_cold),
                tile,
                chunk,
                true,
                &live_test_closure_digest()
            ),
            super::fit_ladder(
                "q4",
                peak,
                seq,
                Some(raw),
                tile,
                chunk,
                true,
                &live_test_closure_digest()
            )
        );
    }
}
