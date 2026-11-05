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
//! packing correction. Since sc-18097 a hard reject on CURRENT evidence no longer waits for the
//! deepest MEASURED rung: every implemented rung inside the measured envelope carries at least an
//! estimate-floor candidate, so the reject fires once no rung fits at its graded (measured or
//! widened-floor) peak. `BestEffort` remains the outcome only where floors deliberately do not
//! apply — superseded entries, an unverified runtime artifact, adapter overlays, or a request
//! above the measured envelope.
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
/// Candle's bespoke Krea control provider models the separately supplied pose image as an img2img
/// reference. These values mirror its authoritative evidence probe and must describe both measured
/// cells and live requests identically.
pub(crate) const KREA_CONTROL_MODE: &str = "image_to_image";
pub(crate) const KREA_CONTROL_REFERENCE_COUNT: u32 = 1;
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
    /// Until sc-18097 this was also the outcome whenever the Sequential row was absent. It no longer
    /// is: a current-evidence cell inside the measured envelope now grades its unmeasured rungs at
    /// estimate floors and can reject honestly. `BestEffort` is reserved for the states where those
    /// floors are withheld — superseded evidence (here), an unverified runtime artifact, adapter
    /// overlays, or a request above the measured envelope — where asserting a non-fit nobody
    /// measured, from numbers nobody can stand behind, remains the wrong call.
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
    /// Staging without rejecting is the same posture [`KreaControlFit::BestEffort`] takes, and for the
    /// same reason: with no peak row for this tier there is nothing for sc-18097's estimate floors to
    /// floor ON, so no graded candidate exists to reject from. The knobs coincide with
    /// `Fits { Sequential, false, false }`
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
/// CAPTURE PROVENANCE ONLY (sc-22738): it is carried into the evidence receipt's
/// `inference_revision` so a trace names what the rows were measured under, and compared against
/// nothing — the ladder prices the rows identically whether or not the lane's closure has moved.
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
    // (sc-17774). Read from the manifest, never hardcoded — and since sc-22738 carried into the
    // evidence receipt as provenance only, never compared.
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
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
    };
    let overlay = if adapter_gb > 0.0 {
        format!(
            "control_branch+adapters_bytes={}",
            (adapter_gb * crate::fit_gate::BYTES_PER_GIB).round() as u64
        )
    } else {
        "control_branch".to_owned()
    };
    let request = RequestScope {
        resolved_route: KREA_CONTROL_ROUTE,
        backend: "candle",
        tier: numeric_tier,
        mode: KREA_CONTROL_MODE,
        overlay: Some(&overlay),
        geometry: request_geometry,
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
                model_family: KREA_CONTROL_ROUTE.to_owned(),
                resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: contract.load_shape,
                mode: crate::memory_strategy::memory_mode_from_mode_key(KREA_CONTROL_MODE),
                reference_shape: gen_core::MemoryReferenceShape::Image,
                overlay: Some(overlay.clone()),
                geometry: measured_geometry,
                frames_per_second: None,
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
    // where present, else the resident floor; every deeper rung whose engaged composition
    // includes `StagedResidency` takes the staged floor UNREDUCED — selectable without promising
    // an unmeasured saving. The staged row prices the staged WORKING SET, so a deep rung whose
    // engaged composition EXCLUDES staging (sc-18253 — the `engagedRungs` mechanism permits it)
    // runs whole-model resident and clamps to the resident floor instead, mirroring the MLX
    // floor's `engaged.contains(&StagedResidency)` max-vs-sum split
    // (`mlx_fit_gate::estimate_floor_weights_bytes`). Where a rung's savings ARE measured, its
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
            let engaged = contract.engaged_composition(strategy);
            // sc-18253: the staged row is only a sound floor for a composition that actually
            // engages staging; a rung excluding `StagedResidency` runs whole-model resident and
            // clamps to the resident floor (see the header comment above).
            let floor_gb = if strategy != MemoryStrategy::Resident
                && engaged.contains(&MemoryStrategy::StagedResidency)
            {
                staged_floor_gb
            } else {
                resident_floor_gb
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
                sceneworks_core::memory_contracts::unverified_estimate(
                    MemoryEvidenceKey {
                        model_family: KREA_CONTROL_ROUTE.to_owned(),
                        resolved_route: KREA_CONTROL_ROUTE.to_owned(),
                        backend: gen_core::MemoryBackend::Candle,
                        tier: numeric_tier,
                        load_shape: contract.load_shape,
                        mode: crate::memory_strategy::memory_mode_from_mode_key(KREA_CONTROL_MODE),
                        reference_shape: gen_core::MemoryReferenceShape::Image,
                        overlay: Some(overlay.clone()),
                        geometry: request_geometry,
                        frames_per_second: None,
                        strategy,
                        engaged_composition: engaged,
                        parameters: selection.parameters,
                    },
                    contract
                        .calibration
                        .as_ref()
                        .expect("control calibration")
                        .abi,
                    KREA_CONTROL_CALIBRATION.to_owned(),
                    KREA_CONTROL_ESTIMATE_REVISION.to_owned(),
                    predicted_peak_bytes,
                ),
            ));
        }
    }
    let mut candidates = selections
        .iter()
        .zip(&evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
            basis: crate::memory_strategy::CandidateBasis::Measured,
            unmodeled_activation_bytes: None,
        })
        .collect::<Vec<_>>();
    candidates.extend(estimates.iter().map(|(selection, evidence)| Candidate {
        selection: *selection,
        evidence,
        basis: crate::memory_strategy::CandidateBasis::EstimateFloor,
        // sc-22508: this control-lane floor is a manifest-row declaration, not a weights+headroom
        // split, so it declares no activation term and takes the candle whole-peak residual.
        unmodeled_activation_bytes: None,
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
            reference_count: KREA_CONTROL_REFERENCE_COUNT,
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
            reference_count: KREA_CONTROL_REFERENCE_COUNT,
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
mod tests;
