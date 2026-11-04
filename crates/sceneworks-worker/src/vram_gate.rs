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

/// The operational reserve the candle memory-strategy ladder charges EXACTLY ONCE per candidate
/// (sc-22664, epic 22657 E4) — against the selector budget for a candidate priced from a measured
/// device delta, and never ALSO against a candidate whose peak already carries the structural pad;
/// `crate::memory_strategy::ReserveCharge` states that rule once.
///
/// `reserve = max(total − free, MEASURED_PRELOAD_RESIDENCY_GB) + LADDER_RESERVE_MARGIN_GB`, capped
/// at `HEADROOM_GB`:
///
/// * `total − free` is the idle baseline of the selected card as probed before the load — the
///   CUDA context(s) and WDDM residency the box actually carries — and it MUST be the RAW probe,
///   never the reclaimable-credited budget (`with_reclaimable`), which predicts the free the
///   imminent evict will produce and so reads a warm card as nearly idle (review D2).
/// * [`MEASURED_PRELOAD_RESIDENCY_GB`] floors it at the residency the retained record shows this
///   process carrying before its load, because a cold probe cannot see its own context (D3).
/// * [`LADDER_RESERVE_MARGIN_GB`] is the context's post-probe growth.
/// * The legacy [`HEADROOM_GB`] stays the CEILING: on a card whose reported used bytes exceed it,
///   the "idle" reading is foreign residency (another process's model, a second worker on the
///   same GPU, or on a warm run this process's own resident model about to be evicted), which is
///   already excluded from `free` and says nothing about this process's context, so the reserve
///   falls back to the legacy allocator slack rather than charging those bytes a second time. A
///   budget with more free than total (a capped emulation) reads as a zero baseline and lands on
///   the measured floor.
///
/// The retained candle anchors measure every phase peak as a device DELTA above the pre-load
/// residency, so this baseline is the one term an anchor-derived peak does not contain; charging
/// it against the budget once for such a candidate, rather than folding a fixed 2 GB into the
/// derivation AND subtracting 2 GB from the budget, is what lets an 8 GB card admit a 4.5 GB
/// fully-engaged composition once a packaged anchor prices the cell.
pub(crate) fn ladder_reserve_gb(budget: VramBudget) -> f64 {
    let idle_baseline_gb = (budget.total_gb - budget.free_gb)
        .max(0.0)
        .max(MEASURED_PRELOAD_RESIDENCY_GB);
    (idle_baseline_gb + LADDER_RESERVE_MARGIN_GB).min(HEADROOM_GB)
}

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

/// Read the selected card and apply the emulation cap exactly once. Reclaim credit is a
/// separate, caller-owned decision; fresh post-eviction probes must bypass the cached reading.
pub(crate) async fn live_capped_budget(gpu_id: &str) -> Option<VramBudget> {
    apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(gpu_id).await,
        cuda_vram_cap_gb(),
    )
}

/// Fresh post-eviction device reading: deliberately bypasses the heartbeat cache.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn live_capped_budget_fresh_blocking(gpu_id: &str) -> Option<VramBudget> {
    apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb_fresh_blocking(gpu_id),
        cuda_vram_cap_gb(),
    )
}

/// Read the positive small-card emulation cap from the environment.
pub(crate) fn cuda_vram_cap_gb() -> Option<f64> {
    parse_vram_cap(std::env::var(CUDA_VRAM_CAP_ENV).ok().as_deref())
}

/// Parse the cap value: a positive float (GB), else `None`.
pub(crate) fn parse_vram_cap(raw: Option<&str>) -> Option<f64> {
    let value = raw?.trim().parse::<f64>().ok()?;
    (value.is_finite() && value > 0.0).then_some(value)
}

/// Apply the small-card cap to a real budget. The cap emulates a card whose TOTAL VRAM is `cap` GB:
/// `total := cap`, `free := min(real_free, cap)`. With no real reading a cap still yields a full budget
/// (`free = total = cap`) so the gate is exercisable in a no-GPU unit test. No cap ⇒ the real budget
/// unchanged.
pub(crate) fn apply_vram_cap(real: Option<VramBudget>, cap: Option<f64>) -> Option<VramBudget> {
    match cap {
        Some(cap) => {
            let free_gb = real.map_or(cap, |budget| budget.free_gb.min(cap));
            Some(VramBudget {
                free_gb,
                total_gb: cap,
            })
        }
        None => real,
    }
}

/// Per-GPU high-water of the peak VRAM (GB) this process has admitted a load at (sc-11023) — a bound on
/// what evicting the current resident can free back for a swap-in. Keyed by `gpu_id` (a worker process
/// is pinned to one card, but keying is defensive + explicit). It only ever grows (monotonic MAX), and
/// [`with_reclaimable`] clamps the credit to `total_gb`, so a later gate can only under-credit, never
/// over-admit — the safety that holds whether an evict frees to the driver or pools in-process.
fn reclaimable_pool_store() -> &'static std::sync::Mutex<std::collections::HashMap<String, f64>> {
    static STORE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, f64>>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record the peak VRAM (GB) an admitted load will occupy on `gpu_id`, as a monotonic high-water
/// (sc-11023). Called only after the fit-gate ADMITS a load, so it reflects a real allocation attempt.
/// A non-positive peak is ignored.
pub(crate) fn note_loaded_peak(gpu_id: &str, peak_gb: f64) {
    // Ignore a non-positive, NaN, or infinite peak (defensive — a real measured peak is finite > 0).
    if !peak_gb.is_finite() || peak_gb <= 0.0 {
        return;
    }
    let mut store = reclaimable_pool_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let entry = store.entry(gpu_id.to_owned()).or_insert(0.0);
    if peak_gb > *entry {
        *entry = peak_gb;
    }
}

/// The reclaimable in-process VRAM pool (GB) for `gpu_id` — the [`note_loaded_peak`] high-water, or
/// `0.0` when nothing has loaded on this card yet (so [`with_reclaimable`] is a no-op on a cold start).
pub(crate) fn reclaimable_pool_gb(gpu_id: &str) -> f64 {
    reclaimable_pool_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(gpu_id)
        .copied()
        .unwrap_or(0.0)
}

/// The tier key (`nvfp4`/`bf16`/`q8`/`q4`) the request selected — derived the SAME way the tier-subdir
/// resolvers pick their folder (`advanced.mlxQuantize` → manifest `mlx.quantize` → `q8` default;
/// `<= 0` ⇒ `bf16`; `<= 4` ⇒ `q4`; else `q8`). Deliberately NOT derived from the resolved `Quant`,
/// which is `None` on packed-tier candle families whose tier is the resolved subdir, not a load-time
/// quantize (see `resolve_quant`'s candle-lane note).
///
/// **`nvfp4` (sc-11042) is passed IN, not parsed here.** NVFP4 carries no `mlxQuantize` by design (no
/// integer is honest for a ~4.5-effective-bit tier), so a bits-derived key can only ever return `q8`
/// for it — which sized an NVFP4 render against `candle.vramGbByTier["q8"]`, roughly 2× its real
/// footprint. That failed CONSERVATIVE (a spurious `TooBig`/`Offload`, never an OOM), but it was the
/// fourth bits-derived site of the same aliasing and is now keyed off the tier IDENTITY like the other
/// three. The caller passes `image_jobs::base::nvfp4_selected` — the same predicate behind the load
/// quant and the recorded label, so all four agree on what ran by construction. This module stays pure
/// (no GPU probe, no filesystem) precisely because the flag arrives resolved.
///
/// Mirrors [`preferred_tier`](crate::image_jobs::base)'s `(bits, floor, nvfp4)` shape: `nvfp4 == false`
/// leaves every existing mapping byte-identical.
pub(crate) fn requested_tier_key(
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
    nvfp4: bool,
) -> &'static str {
    // The distinct NVFP4 tier short-circuits the bits map: it is a tier identity, not a point on the
    // bits ladder, and it has no honest `mlxQuantize` integer to be derived from.
    if nvfp4 {
        return NVFP4_TIER;
    }
    let raw = advanced.get("mlxQuantize").and_then(quant_int).or_else(|| {
        manifest_entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .and_then(quant_int)
    });
    crate::payload::tier_for_bits(raw, "q8")
}

/// Predicted peak VRAM (GB) for `tier_key` from the manifest `candle` block:
/// `candle.vramGbByTier[tier_key]` (measured peak) + [`HEADROOM_GB`], else `candle.minMemoryGb`
/// (already padded), else `None` — an unmeasured model (no `candle` block, e.g. the dense sc-3675
/// families) skips the gate entirely.
///
/// **An `nvfp4` tier with no measured row degrades to the `q8` row, NOT to `minMemoryGb` (sc-11042).**
/// sc-11043 owns the convert-at-install loop and **must add an `nvfp4` row to `vramGbByTier` when it
/// converts a tier** — this is the documented behavior until it does. `minMemoryGb` is the WRONG floor
/// to land on here: per the manifest schema it is "the measured overall-peak of the DEFAULT (lightest,
/// typically q4) hosted tier", which heavier tiers are explicitly allowed to exceed — so falling
/// through to it would size an FP4 render against the lightest tier's number and fail PERMISSIVELY
/// (an under-prediction admits a load that can OOM). The `q8` row instead OVER-predicts (q8's weights
/// are ~2× NVFP4's ~4.5 effective bits), which is both safe and exactly the number this gate already
/// used for an NVFP4 request before the tier had its own key — the bits-derived
/// [`requested_tier_key`] returned `"q8"` for it. So the sizing is no less conservative than the
/// status quo, and a missing row can only ever cost a spurious `TooBig`/`Offload`, never an OOM.
pub(crate) fn predicted_peak_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    if let Some(gb) = measured_resident_peak_gb(manifest_entry, tier_key) {
        return Some(gb + HEADROOM_GB);
    }
    // Spelled exactly so: the manifest constraint-contract registry anchors the `candle.minMemoryGb`
    // reader on this expression (`tests/test_builtin_manifest_audit.py`).
    let candle = manifest_entry.get("candle")?;
    // sc-24112: a per-tier ALREADY-PADDED floor, between the measured ladder and the scalar. It is
    // deliberately NOT a `vramGbByTier` row: that ladder is a raw peak this gate pads with
    // `HEADROOM_GB`, and its sibling `measured` flag files it as an evidence class — so a DERIVED
    // per-tier floor put there would be padded twice AND laundered into evidence. Returned as-is,
    // exactly like the scalar it generalizes. A tier with no row falls through to the scalar, which
    // must therefore stay the CONSERVATIVE number for every unlisted key (today `nvfp4`): landing
    // an unknown tier on a light tier's floor would be a permissive under-prediction, and an
    // under-prediction admits a load that OOMs.
    if let Some(gb) = tier_floor_gb(candle, tier_key) {
        return Some(gb);
    }
    candle.get("minMemoryGb").and_then(json_f64)
}

/// The derived per-tier floor `candle.minMemoryGbByTier[tier_key]`, or `None` when the block
/// declares no row for this tier. No headroom is added and none may be: the manifest pads this
/// number itself, exactly as it pads `candle.minMemoryGb`.
///
/// Unlike [`measured_resident_peak_gb`], an unmeasured `nvfp4` does NOT degrade to the `q8` row
/// here. That degradation exists because q8's measured PEAK over-predicts NVFP4's, which is safe;
/// a q8 FLOOR is a different quantity, and guessing with it would be the under-prediction this
/// whole path exists to avoid. NVFP4 falls through to the scalar instead.
fn tier_floor_gb(candle: &Value, tier_key: &str) -> Option<f64> {
    candle
        .get("minMemoryGbByTier")
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
}

/// The RAW measured resident row, `candle.vramGbByTier[tier_key]` (or the `q8` row for an
/// unmeasured `nvfp4` tier, exactly as [`predicted_peak_gb`] degrades), with NO headroom folded
/// in. `None` where the tier has no measured row — [`predicted_peak_gb`] then lands on
/// `minMemoryGb`, which the manifest pads itself.
///
/// This is the row the memory-strategy ladder prices its resident candidate from (sc-22664): the
/// ladder charges its operational reserve once against the selector budget
/// ([`ladder_reserve_gb`]), so a candidate that also carried [`HEADROOM_GB`] would pay twice. The
/// legacy fit gate keeps consuming the padded [`predicted_peak_gb`] because it charges no reserve
/// against its budget.
pub(crate) fn measured_resident_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let candle = manifest_entry.get("candle")?;
    let measured = |key: &str| {
        candle
            .get("vramGbByTier")
            .and_then(|tiers| tiers.get(key))
            .and_then(json_f64)
    };
    measured(tier_key)
        // NVFP4 with no measured row → the q8 row (a deliberate over-prediction; see the note above).
        .or_else(|| (tier_key == NVFP4_TIER).then(|| measured("q8")).flatten())
}

/// Resident prediction with a load-exact independently resident adapter stack. Callers pass zero
/// for a dense/folded load; packed providers pass the measured source bytes retained as residuals.
pub(crate) fn predicted_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Predicted SEQUENTIAL peak VRAM (GB) for `tier_key`: `candle.sequentialPeakGb[tier_key]` (the measured
/// largest single working set of the sequential-residency path, sc-10856) + [`HEADROOM_GB`], mirroring
/// [`predicted_peak_gb`]'s headroom. `None` when unmeasured (no `sequentialPeakGb`, or no entry for this
/// tier) — then the [`FitDecision::Offload`] path keeps today's best-effort behavior: run sequentially
/// and lean on the reactive Metal/CUDA-OOM containment backstop rather than reject.
///
/// An `nvfp4` tier with no measured row falls back to the `q8` row, mirroring [`predicted_peak_gb`]'s
/// NVFP4 note (q8's sequential working set is an over-prediction of NVFP4's, so the second-stage gate
/// degrades conservatively rather than best-effort). sc-11043 must add the real `nvfp4` rows.
pub(crate) fn predicted_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    measured_sequential_peak_gb(manifest_entry, tier_key).map(|gb| gb + HEADROOM_GB)
}

/// The RAW measured sequential row, `candle.sequentialPeakGb[tier_key]` (or the `q8` row for an
/// unmeasured `nvfp4` tier), with NO headroom folded in — the staged working set the ladder's
/// manifest-row floor prices (sc-22664; see [`measured_resident_peak_gb`] for why the ladder's
/// candidates carry no headroom).
pub(crate) fn measured_sequential_peak_gb(
    manifest_entry: &JsonObject,
    tier_key: &str,
) -> Option<f64> {
    let sequential = manifest_entry.get("candle")?.get("sequentialPeakGb")?;
    let measured = |key: &str| sequential.get(key).and_then(json_f64);
    measured(tier_key).or_else(|| (tier_key == NVFP4_TIER).then(|| measured("q8")).flatten())
}

/// Sequential prediction with the same adapter residency charged in every lifecycle policy.
pub(crate) fn predicted_sequential_peak_gb_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
) -> Option<f64> {
    predicted_sequential_peak_gb(manifest_entry, tier_key)
        .map(|peak| peak + adapter_bytes as f64 / BYTES_PER_GIB)
}

/// Decide whether the predicted peak fits the (possibly capped) live budget. Missing either input ⇒
/// `Unknown` (never block). Compares against `free_gb` — what is actually allocatable now — mirroring
/// the flux2 edit guard's use of `available_gb`.
/// FLUX3 base's catalog floor is weights/fixed reserve, not a measured one-row activation
/// anchor. Add its full two-row source allowance; keep adapters, optional VLM and decode separate.
pub(crate) fn flux3_source_activation_allowance_bytes(
    engine_id: &str,
    width: u32,
    height: u32,
) -> u64 {
    if engine_id != "flux3_klein_base" {
        return 0;
    }
    let workspace = crate::mlx_fit_gate::flux3_source_workspaces(engine_id, width, height);
    workspace.0.max(workspace.1)
}

pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<VramBudget>) -> FitDecision {
    crate::fit_gate::fit_decision(needed_gb, budget.map(|budget| budget.free_gb))
}

/// Second-stage gate for an [`FitDecision::Offload`] (epic 10765, sc-10856): sequential residency was
/// selected because the RESIDENT peak won't fit, on the promise that the sequential working set will.
/// If the tier's MEASURED sequential peak is known (`sequential_needed_gb` = [`predicted_sequential_peak_gb`])
/// and STILL exceeds the budget, this returns `Some(needed_gb)` so the caller can reject before load with
/// an actionable message instead of running into a reactive Metal/CUDA OOM. `None` — the sequential peak
/// fits, is unmeasured for this tier, or there is no live budget — keeps today's best-effort run.
pub(crate) fn sequential_overflow_gb(
    sequential_needed_gb: Option<f64>,
    budget: Option<VramBudget>,
) -> Option<f64> {
    crate::fit_gate::sequential_overflow_gb(
        sequential_needed_gb,
        budget.map(|budget| budget.free_gb),
    )
}

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

/// Translate the shared strategy vocabulary to Krea's historical manifest keys at the evidence-read
/// boundary. This is deliberately a function, not a second rung enum: selection order and execution
/// composition remain owned by [`gen_core::MemoryStrategy`] and the provider contract.
fn krea_turbo_manifest_key(strategy: gen_core::MemoryStrategy) -> &'static str {
    match strategy {
        gen_core::MemoryStrategy::Resident => "resident",
        gen_core::MemoryStrategy::StagedResidency => "threeStage",
        gen_core::MemoryStrategy::BoundedDecode => "tiledVae",
        gen_core::MemoryStrategy::BoundedAttention => "chunkedAttention",
        gen_core::MemoryStrategy::BoundedTransformerResidency => "streamedBlocks",
    }
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

impl BindingPhase {
    /// The phase index (0 text, 1 denoise, 2 decode) the sc-18097 comparison seam is written in.
    pub(crate) fn index(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Denoise => 1,
            Self::Decode => 2,
        }
    }
}

/// Per-phase prediction for one Krea Turbo rung at the requested geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KreaTurboPhasePeaks {
    pub text_gb: f64,
    pub denoise_gb: f64,
    pub decode_gb: f64,
}

impl KreaTurboPhasePeaks {
    pub(crate) fn peak_gb(self) -> f64 {
        self.text_gb.max(self.denoise_gb).max(self.decode_gb)
    }

    /// The phase carrying the peak. Ties resolve to the LATER phase deterministically, mirroring
    /// the MLX gate's `binding_phase` (sc-18096).
    ///
    /// This is why sc-18810 put the temporal term on each phase curve rather than on the
    /// aggregate: `peak_gb` is a `max` over three curves, and a max of linear pieces is not
    /// linear. Fitting one temporal coefficient to the aggregate missed it by >= 10.26 GiB on
    /// held-out geometries — about 94x the measured noise floor — while the ADOPTED per-phase
    /// form stays inside 0.44 GiB: 0.0185 decode, 0.1741 denoise, 0.4438 text. Keeping the curves
    /// per phase is what makes both `peak_gb` and this question answerable at an unmeasured
    /// geometry.
    ///
    /// That per-phase band is 0.019-0.44 GiB, not the tighter 0.30 an earlier revision of this
    /// comment quoted: 0.30 is `area_only`'s TEXT residual, and no cross-form phase equals it.
    /// The distinction matters because the win is not uniform. On text the rejected additive form
    /// scores 0.3445 against cross's 0.4438, so **cross is 1.29x worse there**. Cross is still the
    /// correct adoption: it beats additive 355x on decode and 15.4x on denoise — the phases that
    /// actually carry the temporal response and decide admission at scale — it strictly contains
    /// the constrained candidates sharing its regressors (`area_only`, `output_voxels`), and the
    /// 0.0993 GiB it concedes on text is under that phase's own 0.1095 GiB replicate floor, the
    /// largest of the three by orders of magnitude. Every figure here is graded from the committed
    /// artifact rather than from this comment, by the test
    /// `the_adopted_form_is_the_only_accurate_one_that_extends_the_shipped_curve`.
    pub(crate) fn binding_phase(self) -> BindingPhase {
        let mut phase = BindingPhase::Text;
        let mut peak = self.text_gb;
        if self.denoise_gb >= peak {
            phase = BindingPhase::Denoise;
            peak = self.denoise_gb;
        }
        if self.decode_gb >= peak {
            phase = BindingPhase::Decode;
        }
        phase
    }
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

impl KreaTurboFit {
    /// The `image_memory_strategy_selected` telemetry payload for a selection this lane made
    /// (sc-22667, epic 22657 E7) — the same shape the shared candle ladder emits
    /// (`candle_memory_strategy::CandleMemoryEvaluation::selection_telemetry`): the selected rung
    /// and its parameters, the basis that priced it, the three phase peaks where a phase
    /// decomposition priced it (every optimized rung: the measured curves, the fitted curves and
    /// the anchor derivation are all per phase), and the selector's admission figures. `None`
    /// for a refusal or an unverified result, which select nothing.
    pub(crate) fn selection_telemetry(&self, tier: &str, width: u32, height: u32) -> Option<Value> {
        let (selection, basis, phases, admitted, predicted_peak_gb) = match self {
            Self::Resident {
                peak_gb,
                selection,
                admitted,
                ..
            } => (
                selection,
                crate::memory_strategy::CandidateBasis::Measured,
                None,
                admitted,
                *peak_gb,
            ),
            Self::Fits {
                phases,
                selection,
                basis,
                admitted,
                ..
            } => (selection, *basis, Some(*phases), admitted, phases.peak_gb()),
            Self::Reject { .. } | Self::Unverified { .. } => return None,
        };
        let bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
        let parameters = selection.parameters;
        Some(serde_json::json!({
            "backend": "candle",
            "route": "krea_2_turbo",
            "actualTier": tier,
            "mode": "text_to_image",
            "geometry": {
                "width": width,
                "height": height,
                "batch": 1,
                "frames": KREA_LANE_FRAMES,
            },
            "referenceCount": 0,
            "overlay": Value::Null,
            "strategy": crate::candle_memory_strategy::strategy_label(selection.strategy),
            "parameters": {
                "decodeTileEdge": parameters.decode_tile_edge,
                "decodeOverlap": parameters.decode_overlap,
                "attentionChunkSize": parameters.attention_chunk_size,
                "transformerWindowSize": parameters.transformer_window_size,
            },
            "basis": basis.as_key(),
            "authority": if basis.is_estimate() { "estimated" } else { "calibrated" },
            "predictedPeakBytes": bytes(predicted_peak_gb),
            "phasePeakBytes": phases.map(|phases| serde_json::json!({
                "conditioning": bytes(phases.text_gb),
                "denoise": bytes(phases.denoise_gb),
                "decode": bytes(phases.decode_gb),
            })),
            "admittedPeakGb": admitted.needed_gb,
            "availableGb": admitted.available_gb,
            "reserveGb": admitted.reserve_gb,
            "evidenceRevision": KREA_TURBO_SCENEWORKS_REVISION,
        }))
    }
}

pub(crate) const KREA_TURBO_SCENEWORKS_REVISION: &str = "sc-15449-contract-v1";
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

impl KreaRuntimeEvidenceContext {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inspect(
        resolved_route: &str,
        backend: &str,
        gpu_id: &str,
        compute_capability: Option<f32>,
        artifact_provider: &str,
        artifact_repository: &str,
        resolved_revision: &str,
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<Self> {
        let compute_capability = compute_capability?;
        Self::inspect_artifact_structure(tier_root, resolved_artifact_root, pinned_snapshot_root)?;
        Some(Self {
            resolved_route: resolved_route.to_owned(),
            backend: backend.to_owned(),
            gpu_id: gpu_id.to_owned(),
            compute_capability,
            artifact_provider: artifact_provider.to_owned(),
            artifact_repository: artifact_repository.to_owned(),
            resolved_revision: resolved_revision.to_owned(),
            tier_root: tier_root.to_owned(),
            provider_contract: None,
        })
    }

    /// Inspect the stock Turbo artifact and acquire its full registered provider contract once.
    ///
    /// Krea Control also uses [`Self::inspect`] for structural base-snapshot validation, but its
    /// execution contract is different. Keep that path metadata-only; only the Turbo fit ladder
    /// needs this sealed, reusable provider contract.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inspect_turbo_fit(
        resolved_route: &str,
        backend: &str,
        gpu_id: &str,
        compute_capability: Option<f32>,
        artifact_provider: &str,
        artifact_repository: &str,
        resolved_revision: &str,
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<Self> {
        let mut context = Self::inspect(
            resolved_route,
            backend,
            gpu_id,
            compute_capability,
            artifact_provider,
            artifact_repository,
            resolved_revision,
            tier_root,
            resolved_artifact_root,
            pinned_snapshot_root,
        )?;
        // The registered lookup performs complete production load-source validation, including
        // encoder ArtifactSeal acquisition. The path-free result is safe to reuse for every fit
        // against this inspected request-local artifact; the eventual provider load independently
        // validates its retained seals before opening model bytes.
        let provider_contract = crate::inference_runtime::media()
            .memory_strategy_contract(
                "krea_2_turbo",
                &gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(
                    resolved_artifact_root.to_owned(),
                )),
            )
            .ok()
            .flatten()?;
        context.provider_contract = Some(provider_contract);
        Some(context)
    }

    /// Cheap snapshot-layout validation kept separate from the provider's full sealed-source
    /// validation so its missing-shard behavior can be exercised with a tiny structural fixture.
    fn inspect_artifact_structure(
        tier_root: &str,
        resolved_artifact_root: &std::path::Path,
        pinned_snapshot_root: &std::path::Path,
    ) -> Option<()> {
        let expected = pinned_snapshot_root.join(tier_root);
        if resolved_artifact_root.canonicalize().ok()? != expected.canonicalize().ok()? {
            return None;
        }
        let components = sceneworks_core::mlx_tier_completeness::tier_declared_components(
            resolved_artifact_root,
        )?;
        if components.is_empty()
            || !components.iter().all(|component| {
                std::fs::read_dir(resolved_artifact_root.join(component)).is_ok_and(|entries| {
                    entries
                        .flatten()
                        .any(|entry| !sceneworks_core::lora_family::is_hidden_file(&entry.path()))
                })
            })
        {
            return None;
        }
        let parse_json = |path: &std::path::Path| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        };
        let weights_loadable = |component: &str| {
            let dir = resolved_artifact_root.join(component);
            if parse_json(&dir.join("config.json")).is_none() {
                return false;
            }
            let index = std::fs::read_dir(&dir).ok().and_then(|entries| {
                entries.flatten().map(|entry| entry.path()).find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                })
            });
            if let Some(index) = index {
                return parse_json(&index)
                    .and_then(|value| value.get("weight_map")?.as_object().cloned())
                    .is_some_and(|weight_map| {
                        !weight_map.is_empty()
                            && weight_map.values().all(|shard| {
                                shard
                                    .as_str()
                                    .is_some_and(|shard| dir.join(shard).is_file())
                            })
                    });
            }
            sceneworks_core::mlx_tier_completeness::dir_has_visible_file_ending(
                &dir,
                ".safetensors",
            )
        };
        if !["transformer", "text_encoder", "vae"]
            .into_iter()
            .all(weights_loadable)
            || parse_json(&resolved_artifact_root.join("tokenizer/tokenizer.json")).is_none()
            || parse_json(&resolved_artifact_root.join("scheduler/scheduler_config.json")).is_none()
        {
            return None;
        }
        Some(())
    }

    #[cfg(test)]
    pub(crate) fn verified_for_test(tier_root: &str) -> Self {
        Self {
            resolved_route: "krea_2_turbo".into(),
            backend: "candle".into(),
            gpu_id: "test-gpu".into(),
            compute_capability: 12.0,
            artifact_provider: "huggingface".into(),
            artifact_repository: "SceneWorks/krea-2-turbo-mlx".into(),
            resolved_revision: "d009674080cc1bccf2b629d834c34bf5eccdb723".into(),
            tier_root: tier_root.into(),
            provider_contract: Some(krea_test_provider_contract(tier_root).clone()),
        }
    }
}

/// Sparse, weights-free Krea source used by the Candle-only memory-contract tests.
///
/// The provider's memory lookup now shares its complete production load-spec validation, including
/// the encoder and tokenizer contracts. An empty path therefore no longer represents a valid source.
/// Keep the tests on the real lookup seam with a structurally truthful fixture; no model tensor is
/// materialized and no memory evidence is measured or rewritten here.
///
/// ## Windows disk discipline (StorageFull incident, 2026-08-20)
///
/// "Sparse" above was only true on APFS/ext4. `gen_core_testkit::write_encoder_contract_fixture`
/// ends with a multi-gigabyte `set_len` — the encoder's dense logical size, which must stay because
/// the weight-bytes gates read it — and on NTFS that allocates in FULL. This fixture was measured
/// costing 21.84 GB of `C:` per test process, and because it lives in a `static` (never dropped)
/// every process left that behind permanently. Two mitigations, both load-bearing:
///
/// 1. Each tier's `.safetensors` tail is released right after the tier is written
///    ([`crate::test_fixture_disk::sparsify_written_safetensors`]). Logical size and every written
///    header byte are unchanged, so the gates read exactly the same numbers.
/// 2. Initialization sweeps prior runs' leftovers, which a `static`'s guard can never do for
///    itself. The directory name carries this process's PID so the sweep can tell its own from
///    another concurrent run's.
#[cfg(test)]
fn krea_test_artifact_root(tier: &str) -> std::path::PathBuf {
    struct Fixture {
        _temp: tempfile::TempDir,
        root: std::path::PathBuf,
    }

    /// Distinctive enough that the sweep can match nothing else under the shared temp root.
    const FIXTURE_FAMILY: &str = "sceneworks-krea-memfix-";

    static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
    let fixture = FIXTURE.get_or_init(|| {
        crate::test_fixture_disk::sweep_stale_fixtures(FIXTURE_FAMILY);
        let temp = tempfile::Builder::new()
            .prefix(&crate::test_fixture_disk::process_keyed_prefix(
                FIXTURE_FAMILY,
            ))
            .tempdir()
            .expect("Krea memory-contract fixture root");
        let root = temp.path().to_path_buf();
        let contract = crate::inference_runtime::media_encoder_contract("krea_2_turbo")
            .expect("Krea encoder contract");
        for (tier, bits) in [("q4", Some(4)), ("q8", Some(8)), ("bf16", None)] {
            let tier_root = root.join(tier);
            // AT THE TIER'S QUANTIZATION, not dense (sc-22667). The `bits` this loop already uses
            // for the transformer's `config.json` describe the whole tier's snapshot: a q4 Krea
            // store holds a q4 text encoder too. Writing a dense encoder for every tier was
            // invisible while the pinned contract published `MemoryAssetFacts::default()`, but the
            // sc-22657 contract prices `conditioning_bytes` from this fixture's safetensors
            // HEADERS — dense made q4's encoder 15.69 GB, four times the packaged anchor's whole
            // 3.83 GB measured conditioning peak, so the derivation law correctly refused a
            // conditioning residency the measurement contradicts and every Krea derivation test
            // below went unpriceable. At the tier's own width it is 3.76 GB, i.e. the anchor's
            // measured peak less ~64 MB of activation — the composition that measurement saw.
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &tier_root.join("text_encoder"),
                contract,
                bits,
            )
            .expect("write sparse Krea encoder fixture");
            // Per tier, not once at the end: the moment between the testkit's `set_len` and the
            // release is the only fully-allocated one, and this bounds it to ONE tier's file.
            crate::test_fixture_disk::sparsify_written_safetensors(&tier_root.join("text_encoder"));
            if let Some(bits) = bits {
                let transformer = tier_root.join("transformer");
                std::fs::create_dir_all(&transformer).expect("Krea transformer fixture dir");
                std::fs::write(
                    transformer.join("config.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "quantization": { "bits": bits, "group_size": 64 }
                    }))
                    .expect("Krea transformer fixture config"),
                )
                .expect("write Krea transformer fixture config");
            }
        }
        Fixture { _temp: temp, root }
    });
    assert!(
        matches!(tier, "q4" | "q8" | "bf16"),
        "unsupported Krea fixture tier {tier}"
    );
    fixture.root.join(tier)
}

#[cfg(test)]
fn krea_test_load_spec(tier: &str) -> gen_core::LoadSpec {
    gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(krea_test_artifact_root(tier)))
}

/// One complete provider validation per immutable test tier.
///
/// ArtifactSeal acquisition deliberately hashes the full logical encoder payload. The shared
/// Krea fixture is process-lifetime immutable, so reacquiring that same seal on every fit only
/// repeats multi-gigabyte I/O without testing another state. Production contexts acquire their
/// contract in [`KreaRuntimeEvidenceContext::inspect_turbo_fit`]; this cache is the test equivalent.
#[cfg(test)]
fn krea_test_provider_contract(tier: &str) -> &'static gen_core::MemoryProviderContract {
    static CONTRACTS: std::sync::OnceLock<[gen_core::MemoryProviderContract; 3]> =
        std::sync::OnceLock::new();
    let contracts = CONTRACTS.get_or_init(|| {
        ["q4", "q8", "bf16"].map(|fixture_tier| {
            crate::inference_runtime::media()
                .memory_strategy_contract("krea_2_turbo", &krea_test_load_spec(fixture_tier))
                .expect("Krea contract lookup succeeds")
                .expect("Krea contract exists")
        })
    });
    match tier {
        "q4" => &contracts[0],
        "q8" => &contracts[1],
        "bf16" => &contracts[2],
        _ => panic!("unsupported Krea fixture tier {tier}"),
    }
}

/// Read a measured phase curve `fixedGb + perMpxGb * megapixels + perMpxFrameGb * megapixels *
/// frames`. The manifest stores fixed weight / allocator residency separately from the
/// geometry-dependent activation slopes. Invalid or incomplete evidence fails closed to `None`;
/// callers retain the established sequential gate instead of inventing a fit.
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
/// against a default.
///
/// A PRESENT `perMpxFrameGb` that `json_f64` cannot read still fails closed, and so does a
/// negative or non-finite one. Only true absence is zero; a malformed value must not silently
/// degrade a video curve into an image curve. "Unreadable" is `json_f64`'s notion, not JSON's:
/// it accepts a NUMERIC STRING, so `"0.3"` evaluates exactly as `0.3` would — deliberately, and
/// identically to `fixedGb`/`perMpxGb`, which have always been read the same way. Rejecting a
/// string-typed coefficient is the SCHEMA's job (`model-manifest.schema.json`
/// `#/$defs/phaseVramCurve` types all three as `number`), and
/// `test_schema_admits_the_temporal_coefficient_additively` is where that rejection is pinned.
/// The string case is carried here as an explicitly-ACCEPTED control so the fail-closed set below
/// is not read as broader than it is.
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
fn krea_phase_curve(phase: &JsonObject, geometry: CurveGeometry) -> Option<f64> {
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

/// The typed materialization shape the shipped Krea Turbo curves were measured under (sc-17097).
///
/// `None` for a missing or unrecognized value, which fails the fit closed rather than defaulting: a
/// silently assumed shape is what let calibration ABI 2's load-shape axis do no work on this route.
pub(crate) fn krea_turbo_load_shape(turbo_fit: &Value) -> Option<gen_core::LoadShape> {
    match turbo_fit.get("loadShape")?.as_str()? {
        "eager_materialization" => Some(gen_core::LoadShape::EagerMaterialization),
        "deferred_materialization" => Some(gen_core::LoadShape::DeferredMaterialization),
        _ => None,
    }
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

fn krea_rung_phase_peaks(
    manifest_entry: &JsonObject,
    tier: &str,
    strategy: gen_core::MemoryStrategy,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<KreaTurboPhasePeaks> {
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    let max_measured_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    if pixels > max_measured_pixels {
        return None;
    }
    // `frames == 0` needs no check here: `pixels * 0` clears any bound, and `krea_phase_curve`
    // refuses the geometry outright a few lines later. A second check would be a guard no test
    // could kill.
    let voxels = pixels.checked_mul(u64::from(frames))?;
    if voxels > max_measured_voxels(turbo_fit, pixels)? {
        return None;
    }
    let rung = turbo_fit
        .get("phaseCurvesByTier")?
        .get(tier)?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let geometry = CurveGeometry { pixels, frames };
    let phase = |name: &str| {
        rung.get(name)
            .and_then(Value::as_object)
            .and_then(|curve| krea_phase_curve(curve, geometry))
    };
    Some(KreaTurboPhasePeaks {
        text_gb: phase("text")?,
        denoise_gb: phase("denoise")?,
        decode_gb: phase("decode")?,
    })
}

/// The per-phase predicted peaks a measured evidence record declares for one rung — the measured
/// basis a fitted-curve estimate extrapolates from (sc-18097). `None` when the record does not
/// carry a complete phase triple for the rung, which fails the estimate closed.
fn krea_record_phase_peaks(record: &Value, manifest_rung: &str) -> Option<KreaTurboPhasePeaks> {
    let phases = record.get("predictedPhasesGb")?.get(manifest_rung)?;
    Some(KreaTurboPhasePeaks {
        text_gb: phases.get("text").and_then(json_f64)?,
        denoise_gb: phases.get("denoise").and_then(json_f64)?,
        decode_gb: phases.get("decode").and_then(json_f64)?,
    })
}

/// The phase index (0 text, 1 denoise, 2 decode) carrying the peak of a phase triple (sc-18097).
/// Ties resolve to the LATER phase deterministically, mirroring the MLX gate's `binding_phase`
/// (sc-18096); the comparison below only ever contrasts two triples produced by the same per-phase
/// curves, so tie handling cannot manufacture a flip on its own.
///
/// sc-18812 moved the rule onto [`KreaTurboPhasePeaks::binding_phase`] so that the same question
/// can be asked of a CURVE-predicted triple at an arbitrary geometry, not only of a recorded one.
/// This wrapper keeps the index vocabulary the sc-18097 comparison seam below is written in.
fn krea_binding_phase(peaks: KreaTurboPhasePeaks) -> u8 {
    peaks.binding_phase().index()
}

fn krea_rung_parameters(
    turbo_fit: &Value,
    strategy: gen_core::MemoryStrategy,
) -> Option<gen_core::MemoryStrategyParameters> {
    let parameters = turbo_fit
        .get("strategyParameters")?
        .get(krea_turbo_manifest_key(strategy))?
        .as_object()?;
    let value = |name| {
        parameters
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(gen_core::MemoryStrategyParameters {
        stage_residency: None,
        decode_tile_edge: value("decodeTileEdge"),
        decode_overlap: value("decodeOverlap"),
        attention_chunk_size: value("attentionChunkSize"),
        transformer_window_size: value("transformerWindowSize"),
        // Rung 4's window scope (SC-15794). `Dit` is the only scope that is correct here:
        // `candle-gen-krea` leaves `transformer_window_components` empty, which the contract reads
        // as the DiT-only pre-SC-15794 behaviour, and every measured `strategyParameters` row in
        // the manifest was collected against DiT block streaming. It is also the published default
        // SC-15794 upheld -- a text-encoder scope cut z-image conditioning 46.5% but moved the
        // request peak 0.0%, so widening the scope buys an admission gate nothing.
        //
        // The scope is declared only on the rung that owns it. `validate_selected_parameters`
        // rejects any `Some(..)` below `BoundedTransformerResidency` as "irrelevant: the selection
        // does not engage its owning strategy rung", and the three cheaper rungs here pair with
        // cheaper strategies
        // (see `rung_pairs`); a blanket `Some` would fail `validate_selection` on three of the
        // four candidates and silently downgrade their evidence verdicts. `None` on those rungs
        // carries the identical DiT meaning without tripping that check.
        transformer_window_component: (strategy
            == gen_core::MemoryStrategy::BoundedTransformerResidency)
            .then_some(gen_core::TransformerComponent::Dit),
    })
}

/// Krea Turbo is a text-to-IMAGE route: every request and every measured record on it produces one
/// output frame. Named rather than repeated as a literal `1` so the curve evaluation and the
/// `MemoryGeometry` it is graded against cannot drift apart — sc-18810 found four of six MLX
/// capture arms hardcoding `frames: 1` into their admission context while reading only width and
/// height, which is exactly how a geometry gets attested that was never asked for.
const KREA_LANE_FRAMES: u32 = 1;

/// The packaged measured memory anchor for one Krea Turbo candle tier (sc-22509, epic 22505), with
/// every identity conjunct that can be answered without the loaded provider contract.
///
/// Fail-closed and deliberately narrow: the anchor was measured on ONE catalog model, route,
/// provider and mode, overlay-free and reference-free. The remaining conjunct (materialization
/// shape) is applied by the caller once the contract is available.
///
/// Currency is NOT a conjunct (sc-22738). The anchor's loader-closure digest (sc-22511) is what
/// the probe tooling re-captures on; the calibration fingerprint and ABI are PROVENANCE bound to
/// the source record by `validate_anchor`. None of them can demote the anchor here: it prices the
/// request whether or not the loader has moved since it was measured.
/// The anchor store this lane reads. Production has exactly one — the packaged, validated store.
///
/// The `cfg(test)` override lets a test grade the derivation ITSELF on a store whose contents it
/// states — the same shape `candle_memory_strategy::synthesize_estimate_floors` already takes as
/// a parameter, and the same test-seam precedent as
/// `video_admission::admit_video_generation_with_curves`.
fn krea_anchor_store() -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchorStore> {
    #[cfg(test)]
    if let Some(store) = tests::injected_anchor_store() {
        return Some(store);
    }
    sceneworks_core::memory_anchor::packaged_memory_anchors()
}

fn krea_store_anchor(
    tier: &str,
    allow_streamed_blocks: bool,
) -> Option<&'static sceneworks_core::memory_anchor::MemoryAnchor> {
    // A job carrying load-time adapters is exactly the case this lane keeps on its resident/staged
    // paths, and the anchor render carried no overlay at all.
    if !allow_streamed_blocks {
        return None;
    }
    krea_anchor_store()
        .and_then(|store| {
            store.image_anchor_for(
                "krea_2_turbo",
                sceneworks_core::memory_anchor::AnchorBackend::Candle,
                tier,
            )
        })
        .filter(|anchor| {
            anchor.route == "krea_2_turbo"
                && anchor.provider == "krea_2_turbo"
                && anchor.mode == "text_to_image"
                && anchor.overlay.is_none()
                && anchor.reference_count == 0
        })
}

/// Select the least-cost measured Krea Turbo fit rung for this tier, geometry, and live budget.
/// `allow_streamed_blocks` is false when the job carries load-time adapters: the provider preserves
/// those jobs on its existing resident/staged paths rather than silently omitting their residuals.
///
/// Returns `None` when live budget or complete measured manifest evidence is absent. This is distinct
/// from `Reject`: unknown evidence must not masquerade as proof that a configuration cannot fit.
///
/// Since sc-18097 (epic 18093 R1b) an in-envelope request geometry with no exact measured record
/// no longer freezes to `Unverified`: each optimized rung carries a fitted-curve ESTIMATE
/// candidate anchored to a verified measured record, graded by the shared selector behind the
/// candle estimate margin — see the synthesis block below for the anchoring and binding-phase
/// rules.
///
/// Since sc-22667 (epic 22657 feature-end round, E3/E4/E7) the lane is on the same law and the
/// same reserve as the shared candle ladder:
///
/// * an OUT-OF-ENVELOPE rung — one the fitted `turboFit` curves cannot price — is derived through
///   `MemoryAnchor::derive_phase_peaks` with the rung's OWN regime
///   (`candle_memory_strategy::request_regime` over the manifest's `strategyParameters`: the
///   decode tile, the attention score budget, the transformer window) and the contract's
///   architecture facts (`video_admission::architecture_facts_from_contract`), so rung 3 and
///   rung 4 price their bounded working sets rather than every rung repeating the shallow staged
///   derivation. In-envelope rungs keep the fitted curves: they are measured evidence.
/// * `reserve_gb` is the operational reserve — `ladder_reserve_gb` of the caller's RAW probe,
///   handed in explicitly so a reclaimable-credited `budget` can never derive it — charged
///   EXACTLY ONCE, on the budget side, against every candidate
///   (`memory_strategy::ReserveCharge::EveryCandidate`): every candidate this lane grades is a
///   device delta above the pre-load residency (the measured records, the curves fitted over
///   them, and the anchor derivation alike), so none carries the reserve inside its peak. The
///   quoted `needed_gb` is the admitted peak plus that reserve.
/// * the selection reports its basis and admission figures, so the caller can emit the
///   `image_memory_strategy_selected` event with the three phase peaks
///   ([`KreaTurboFit::selection_telemetry`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn krea_turbo_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<KreaTurboFit> {
    krea_turbo_fit_priced(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        reserve_gb,
        allow_streamed_blocks,
        runtime,
        None,
    )
}

/// [`krea_turbo_fit_with_runtime`] with the architecture facts the derivation scales by as an
/// explicit input: `None` reads them off the loaded contract (production — the contract's own
/// `architecture_facts` block through `architecture_facts_from_contract`, sc-22667), `Some` lets
/// a fixture grade the rung ratios with facts it states directly.
#[allow(clippy::too_many_arguments)]
fn krea_turbo_fit_priced(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
    facts_override: Option<sceneworks_core::memory_anchor::ArchitectureFacts>,
) -> Option<KreaTurboFit> {
    use crate::memory_strategy::{self, Budget, Candidate, RequestScope, Selection};
    use gen_core::{
        MemoryConformanceState, MemoryEvidence, MemoryEvidenceDimensions, MemoryEvidenceKey,
        MemoryEvidenceVerdict, MemoryGeometry, MemoryParityContract, MemoryParityResult,
        MemorySelection, MemoryStrategy, MemoryStrategyParameters,
    };

    let budget = budget?;
    let turbo_fit = manifest_entry.get("candle")?.get("turboFit")?;
    // sc-22735: the identity the LOADED contract publishes is keyed on the artifact-proven tier, so
    // the fingerprint this fit is graded against has to be read per tier. The scalar remains the
    // measured q4 string and is the fallback for a manifest that declares no per-tier map.
    let calibration_fingerprint = turbo_fit
        .get("calibrationFingerprintByTier")
        .and_then(|by_tier| by_tier.get(tier))
        .or_else(|| turbo_fit.get("calibrationFingerprint"))?
        .as_str()?;
    let calibration_abi = turbo_fit.get("calibrationAbi")?.as_u64()? as u32;
    // sc-17097: calibration ABI 2 added the typed load shape, but this route never read it - the
    // worker took the shape from the provider alone, so the axis could not detect drift here. The
    // manifest now states the shape its curves were MEASURED under and the handshake below compares
    // the two; eager and deferred measurements are not interchangeable.
    let declared_load_shape = krea_turbo_load_shape(turbo_fit)?;
    let scene_works_revision = turbo_fit.get("sceneWorksRevision")?.as_str()?;
    let inference_revision = turbo_fit.get("inferenceRevision")?.as_str()?;
    let max_pixels = turbo_fit.get("maxMeasuredPixels")?.as_u64()?;
    let geometry = MemoryGeometry {
        width,
        height,
        batch: 1,
        frames: KREA_LANE_FRAMES,
        reference_count: 0,
    };
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    // sc-22509 (epic 22505): a geometry the campaign never measured is no longer a refusal on its
    // own. The measured memory ANCHOR for this `(model, tier, candle lane)` prices it analytically,
    // so the hull only refuses when there is no anchor to derive from. The identity guards this
    // lookup cannot answer yet (contract load shape, live calibration identity) are applied once
    // `provider_contract` is resolved, and the hull refusal is re-asserted there if they fail.
    let store_anchor = krea_store_anchor(tier, allow_streamed_blocks);
    let out_of_envelope = pixels > max_pixels;
    if out_of_envelope && store_anchor.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::OutOfEnvelope,
        });
    }
    // Runtime artifact identity is a required evidence dimension. Preserve the explicit unverified
    // result when it is unavailable instead of attempting provider source validation against an
    // empty path and collapsing to `None`; callers keep the same resident-or-reject fallback, while
    // diagnostics retain the truthful reason.
    if runtime.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::Unverified,
        });
    }
    let provider_contract = runtime.and_then(|runtime| runtime.provider_contract.as_ref())?;
    let numeric_tier = gen_core::MemoryNumericTier {
        precision: gen_core::Precision::Bf16,
        quant: match tier {
            "q4" => Some(gen_core::Quant::Q4),
            "q8" => Some(gen_core::Quant::Q8),
            "bf16" => None,
            _ => return None,
        },
        component_precision_floors: &[],
    };
    // The remaining anchor identity conjunct, now that the loaded contract is available: the anchor
    // was measured under one materialization shape, and eager and deferred are not interchangeable.
    // Currency was already graded in `krea_store_anchor` on the loader closure and is deliberately
    // not re-asked here against the calibration campaign (sc-22511: provenance, not currency).
    let anchor = store_anchor.filter(|anchor| {
        let anchor_load_shape = match anchor.load_shape {
            sceneworks_core::memory_anchor::AnchorLoadShape::EagerMaterialization => {
                gen_core::LoadShape::EagerMaterialization
            }
            sceneworks_core::memory_anchor::AnchorLoadShape::DeferredMaterialization => {
                gen_core::LoadShape::DeferredMaterialization
            }
        };
        anchor_load_shape == provider_contract.load_shape
    });
    if out_of_envelope && anchor.is_none() {
        return Some(KreaTurboFit::Unverified {
            reason: MemoryEvidenceVerdict::OutOfEnvelope,
        });
    }
    if !reserve_gb.is_finite() || reserve_gb < 0.0 {
        return None;
    }
    // The law's inputs for the out-of-envelope rungs (sc-22667): the component bytes of the
    // loaded tier and the model's architecture facts, both off the contract.
    let components = crate::video_admission::anchor_component_bytes(provider_contract.asset_facts);
    let facts = facts_override.unwrap_or_else(|| {
        crate::video_admission::architecture_facts_from_contract(provider_contract)
    });
    let request = RequestScope {
        resolved_route: "krea_2_turbo",
        backend: "candle",
        tier: numeric_tier,
        mode: "text_to_image",
        overlay: (!allow_streamed_blocks).then_some("adapter"),
        geometry,
    };
    let resident_peak_gb = manifest_entry
        .get("candle")?
        .get("vramGbByTier")?
        .get(tier)
        .and_then(json_f64)?;
    if !resident_peak_gb.is_finite() || resident_peak_gb < 0.0 {
        return None;
    }
    let resident_parameters = turbo_fit
        .get("strategyParameters")?
        .get("resident")?
        .as_object()?;
    if !resident_parameters.is_empty() {
        return None;
    }
    let bytes = |gb: f64| (gb * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64;
    let tier_records = turbo_fit
        .get("evidenceRecords")?
        .as_array()?
        .iter()
        .filter(|record| {
            record.get("evidenceScope").and_then(Value::as_str) == Some("exact_request")
                && record.get("tier").and_then(Value::as_str) == Some(tier)
        })
        .filter_map(|record| {
            let record_width = u32::try_from(record.get("width")?.as_u64()?).ok()?;
            let record_height = u32::try_from(record.get("height")?.as_u64()?).ok()?;
            Some((record, record_width, record_height))
        })
        .collect::<Vec<_>>();
    let evidence_record = tier_records
        .iter()
        .find(|(_, record_width, record_height)| *record_width == width && *record_height == height)
        .map(|(record, _, _)| *record);
    // `record`/`at_geometry` are parameters rather than captures (sc-18097): the request cell's
    // candidates are graded against the request's own record and geometry, while the estimate
    // synthesis below re-grades a DIFFERENT record at its own measured geometry to decide whether
    // it is a verified extrapolation anchor.
    let make_evidence = |selection: MemorySelection,
                         manifest_rung: Option<&str>,
                         fallback_predicted_peak_gb: f64,
                         fallback_phases: Option<KreaTurboPhasePeaks>,
                         evidence_record: Option<&Value>,
                         at_geometry: MemoryGeometry| {
        let record_peak = |field: &str| {
            let rung = manifest_rung?;
            evidence_record?.get(field)?.get(rung).and_then(json_f64)
        };
        let record_phase = |phase: &str| {
            let rung = manifest_rung?;
            evidence_record?
                .get("predictedPhasesGb")?
                .get(rung)?
                .get(phase)
                .and_then(json_f64)
        };
        let engaged_composition = manifest_rung.map_or_else(
            || provider_contract.engaged_composition(selection.strategy),
            |rung| match evidence_record {
                Some(record) => record
                    .get("measuredCompositions")
                    .and_then(|compositions| compositions.get(rung))
                    .and_then(Value::as_array)
                    .and_then(|composition| {
                        composition
                            .iter()
                            .map(|rung| match rung.as_str()? {
                                "resident" => Some(MemoryStrategy::Resident),
                                "staged_residency" => Some(MemoryStrategy::StagedResidency),
                                "bounded_decode" => Some(MemoryStrategy::BoundedDecode),
                                "bounded_attention" => Some(MemoryStrategy::BoundedAttention),
                                "bounded_transformer_residency" => {
                                    Some(MemoryStrategy::BoundedTransformerResidency)
                                }
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .unwrap_or_default(),
                None => provider_contract.engaged_composition(selection.strategy),
            },
        );
        let predicted_peak_gb =
            record_peak("predictedPeaksGb").unwrap_or(fallback_predicted_peak_gb);
        let observed_peak_gb = record_peak("observedPeaksGb");
        let harness_version = evidence_record
            .and_then(|record| record.get("harnessVersion"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parity = evidence_record
            .and_then(|record| record.get("parity"))
            .and_then(Value::as_object);
        let verdict = |condition, failure| {
            if condition {
                MemoryEvidenceVerdict::Satisfied
            } else {
                failure
            }
        };
        let static_valid = provider_contract.conformance_errors().is_empty()
            && provider_contract.validate_selection(&selection).is_ok();
        let declared_calibration = provider_contract
            .calibration
            .as_ref()
            .is_some_and(|identity| {
                identity.abi == calibration_abi
                    && identity.fingerprint == calibration_fingerprint
                    && identity.load_shape == declared_load_shape
            });
        let valid_commit = |field: &str| {
            evidence_record
                .and_then(|record| record.get(field))
                .and_then(Value::as_str)
                .is_some_and(|revision| {
                    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
        };
        // sc-17774: `compatibleInferenceRevision` is gone. It was a per-record hand-declared "this
        // capture is also valid at revision X" claim — the same one-shot hatch `flux2_dev` had, in a
        // second spelling. Compatibility is now decided by the lane's closure digest below.
        let historical_compatible = valid_commit("sceneWorksCommit")
            && valid_commit("inferenceCommit")
            && evidence_record
                .and_then(|record| record.get("compatibleSceneWorksRevision"))
                .and_then(Value::as_str)
                == Some(KREA_TURBO_SCENEWORKS_REVISION);
        let loadability = evidence_record
            .and_then(|record| record.get("loadability"))
            .and_then(Value::as_object);
        let expected_compute_capability = loadability
            .and_then(|loadability| loadability.get("computeCapability"))
            .and_then(json_f64);
        // `inference_revision` stays parsed above as capture provenance for the receipt, and since
        // sc-22738 so does `turboFit.inferenceClosureDigest`: the lane's compile closure is a
        // re-capture signal for the probe tooling, never a conjunct that can un-verify this
        // dimension at runtime.
        let current_environment = scene_works_revision == KREA_TURBO_SCENEWORKS_REVISION
            && turbo_fit.get("measured").and_then(Value::as_bool) == Some(true)
            && runtime.is_some_and(|runtime| {
                !runtime.gpu_id.trim().is_empty()
                    && expected_compute_capability.is_some_and(|expected| {
                        (f64::from(runtime.compute_capability) - expected).abs() < f64::EPSILON
                    })
            });
        let loadability_matches_record =
            loadability
                .zip(runtime)
                .is_some_and(|(loadability, runtime)| {
                    loadability.get("provider").and_then(Value::as_str)
                        == Some(runtime.artifact_provider.as_str())
                        && loadability.get("repository").and_then(Value::as_str)
                            == Some(runtime.artifact_repository.as_str())
                        && loadability.get("resolvedRevision").and_then(Value::as_str)
                            == Some(runtime.resolved_revision.as_str())
                        && loadability.get("tierRoot").and_then(Value::as_str)
                            == Some(runtime.tier_root.as_str())
                        && loadability.get("route").and_then(Value::as_str)
                            == Some(runtime.resolved_route.as_str())
                        && loadability.get("backend").and_then(Value::as_str)
                            == Some(runtime.backend.as_str())
                        && runtime.resolved_route == provider_contract.provider_id
                        && runtime.backend == provider_contract.backend.backend_id()
                        && runtime.tier_root == tier
                        && manifest_entry
                            .get("downloads")
                            .and_then(Value::as_array)
                            .is_some_and(|downloads| {
                                downloads.iter().any(|download| {
                                    download.get("provider").and_then(Value::as_str)
                                        == loadability.get("provider").and_then(Value::as_str)
                                        && download.get("repo").and_then(Value::as_str)
                                            == loadability.get("repository").and_then(Value::as_str)
                                        && download.get("revision").and_then(Value::as_str)
                                            == loadability
                                                .get("resolvedRevision")
                                                .and_then(Value::as_str)
                                        && download.get("variant").and_then(Value::as_str)
                                            == Some(tier)
                                })
                            })
                });
        let phases_match = fallback_phases.is_some_and(|phases| {
            [
                ("text", phases.text_gb),
                ("denoise", phases.denoise_gb),
                ("decode", phases.decode_gb),
            ]
            .into_iter()
            .all(|(name, predicted)| {
                record_phase(name).is_some_and(|recorded| (recorded - predicted).abs() <= 0.01)
            })
        });
        let exact_parameters = manifest_rung.is_some()
            && record_peak("predictedPeaksGb").is_some_and(|record_peak_gb| {
                (record_peak_gb - fallback_predicted_peak_gb).abs() <= 0.01
            })
            && phases_match
            && provider_contract.validate_selection(&selection).is_ok();
        let evidence_dimensions = MemoryEvidenceDimensions {
            static_implementation: verdict(static_valid, MemoryEvidenceVerdict::Invalid),
            declared_calibration: verdict(
                declared_calibration,
                MemoryEvidenceVerdict::FingerprintMismatch,
            ),
            historical_verification: verdict(
                historical_compatible,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            current_environment_verification: verdict(
                current_environment,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Stale
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            canonical_route_loadability: verdict(
                loadability_matches_record,
                if evidence_record.is_some() {
                    MemoryEvidenceVerdict::Unverified
                } else {
                    MemoryEvidenceVerdict::Missing
                },
            ),
            exact_strategy_parameters: verdict(
                exact_parameters,
                MemoryEvidenceVerdict::OutOfEnvelope,
            ),
        };
        let verified = observed_peak_gb.is_some()
            && !harness_version.is_empty()
            && parity.is_some_and(|parity| {
                parity.get("contract").and_then(Value::as_str) == Some("golden")
                    && parity.get("result").and_then(Value::as_str) == Some("passed")
            })
            && evidence_dimensions.all_satisfied();
        let parity_contract = parity.map_or(MemoryParityContract::Exact, |parity| {
            MemoryParityContract::Golden {
                fixture: parity
                    .get("fixture")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                metric: parity
                    .get("metric")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                maximum_error: parity
                    .get("maximumError")
                    .and_then(json_f64)
                    .unwrap_or_default(),
            }
        });
        MemoryEvidence {
            key: MemoryEvidenceKey {
                model_family: "krea_2_turbo".to_owned(),
                resolved_route: "krea_2_turbo".to_owned(),
                backend: gen_core::MemoryBackend::Candle,
                tier: numeric_tier,
                load_shape: provider_contract.load_shape,
                mode: gen_core::MemoryMode::TextToImage,
                reference_shape: gen_core::MemoryReferenceShape::None,
                // The existing measurements cover ordinary T2I only.
                overlay: None,
                geometry: at_geometry,
                frames_per_second: None,
                strategy: selection.strategy,
                engaged_composition,
                parameters: selection.parameters,
            },
            conformance: if verified {
                MemoryConformanceState::Verified
            } else {
                MemoryConformanceState::ImplementedUnverified
            },
            dimensions: evidence_dimensions,
            calibration_abi,
            calibration_fingerprint: calibration_fingerprint.to_owned(),
            sceneworks_revision: scene_works_revision.to_owned(),
            inference_revision: inference_revision.to_owned(),
            harness_version: harness_version.to_owned(),
            predicted_peak_bytes: bytes(predicted_peak_gb),
            observed_peak_bytes: observed_peak_gb.map(bytes),
            parity: parity_contract,
            parity_result: if verified {
                MemoryParityResult::Passed
            } else {
                MemoryParityResult::NotRun
            },
        }
    };
    let resident_selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: numeric_tier,
    };

    let optimized_strategies = [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ];
    // sc-22509 review (blocker): the resident rung is priced by `candle.vramGbByTier[tier]`, a
    // SCALAR measured at `vramMeasuredPixels` that does not scale with geometry at all. Inside the
    // measured hull that is the row the lane has always quoted; OUTSIDE it the row is simply not a
    // price for the request, and the anchor cannot supply one either — the derivation refuses
    // `staged_residency: false`, because the shallow staged anchor does not price a whole-model
    // resident working set. So an out-of-envelope request emits NO resident candidate: a rung with
    // no priced basis is honestly Missing, not `Measured` at a 1024x1024 number. Without this the
    // hull-refusal bypass above would admit e.g. a 2048x2048 q4 request as `Resident` at 25.7 GiB
    // on a 32 GB card, which previously returned `OutOfEnvelope`.
    let mut evidence = Vec::new();
    let mut selections = Vec::new();
    if !out_of_envelope {
        evidence.push(make_evidence(
            resident_selection,
            None,
            resident_peak_gb,
            None,
            evidence_record,
            geometry,
        ));
        selections.push(resident_selection);
    }
    let mut measured = Vec::new();
    // sc-22509: rungs the manifest curves cannot price at this geometry, derived from the anchor.
    // They are candidates in their own right — never `Measured` — so they are collected apart from
    // the curve-priced evidence above.
    let mut anchor_derived: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    for strategy in optimized_strategies {
        let parameters = krea_rung_parameters(turbo_fit, strategy)?;
        let selection = MemorySelection {
            strategy,
            parameters,
            tier: numeric_tier,
        };
        let curve_phases = krea_rung_phase_peaks(
            manifest_entry,
            tier,
            strategy,
            width,
            height,
            geometry.frames,
        );
        let Some(phases) = curve_phases else {
            // No measured curve reaches this cell. Derive it from the anchor for the exact rung
            // composition the provider would execute. A rung the anchor cannot price — no anchor at
            // all, or a composition running whole-model resident, which the shallow staged anchor
            // does not cover — is SKIPPED, never propagated out of the function (sc-22509 review):
            // `?` here would collapse every other rung's candidate and destroy the selector's own
            // Missing/Reject verdict, turning a truthful `Unverified { OutOfEnvelope }` into a bare
            // `None`.
            let Some(anchor) = anchor else {
                continue;
            };
            if anchor.underived_reason.is_some()
                || anchor.backend != sceneworks_core::memory_anchor::AnchorBackend::Candle
            {
                continue;
            }
            let engaged = provider_contract.engaged_composition(strategy);
            // sc-22667: the rung's OWN regime — its engaged composition and the parameters the
            // manifest selects for it — through the one law, with the contract's facts. Rungs 3
            // and 4 therefore price their chunk and window where the shallow shim priced every
            // rung at the staged working set. A rung whose parameters are incomplete cannot be
            // executed as selected and is skipped, like a rung the law refuses.
            let Some(regime) = crate::candle_memory_strategy::request_regime(&engaged, &parameters)
            else {
                continue;
            };
            let Some(derived) = anchor.derive_phase_peaks(
                &sceneworks_core::memory_anchor::ImageDeriveRequest {
                    width,
                    height,
                    batch: 1,
                    conditioning_tokens: None,
                    regime,
                },
                components,
                facts,
            ) else {
                continue;
            };
            let phases = KreaTurboPhasePeaks {
                text_gb: derived.conditioning as f64 / BYTES_PER_GIB,
                denoise_gb: derived.denoise as f64 / BYTES_PER_GIB,
                decode_gb: derived.decode as f64 / BYTES_PER_GIB,
            };
            let predicted_peak_bytes = bytes(phases.peak_gb());
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                anchor = anchor.id.as_str(),
                anchor_geometry = format!("{}x{}", anchor.geometry.width, anchor.geometry.height),
                conditioning_peak_bytes = derived.conditioning,
                denoise_peak_bytes = derived.denoise,
                decode_peak_bytes = derived.decode,
                raw_peak_bytes = predicted_peak_bytes,
                "synthesized anchor-derived estimate candidate for a geometry outside the \
                 measured curve envelope"
            );
            measured.push((
                strategy,
                phases,
                phases.peak_gb() + reserve_gb,
                selection,
                true,
            ));
            anchor_derived.push((
                selection,
                sceneworks_core::memory_contracts::unverified_estimate(
                    MemoryEvidenceKey {
                        model_family: "krea_2_turbo".to_owned(),
                        resolved_route: "krea_2_turbo".to_owned(),
                        backend: gen_core::MemoryBackend::Candle,
                        tier: numeric_tier,
                        load_shape: provider_contract.load_shape,
                        mode: gen_core::MemoryMode::TextToImage,
                        reference_shape: gen_core::MemoryReferenceShape::None,
                        overlay: None,
                        geometry,
                        frames_per_second: None,
                        strategy,
                        engaged_composition: engaged,
                        parameters,
                    },
                    calibration_abi,
                    calibration_fingerprint.to_owned(),
                    scene_works_revision.to_owned(),
                    predicted_peak_bytes,
                ),
            ));
            continue;
        };
        let phase_peak_gb = phases.peak_gb();
        let needed_gb = phase_peak_gb + reserve_gb;
        measured.push((strategy, phases, needed_gb, selection, false));
        evidence.push(make_evidence(
            selection,
            Some(krea_turbo_manifest_key(strategy)),
            phase_peak_gb,
            Some(phases),
            evidence_record,
            geometry,
        ));
        selections.push(selection);
    }
    // ── sc-18097 (epic 18093 R1b): fitted-curve estimate candidates for unmeasured cells. ──
    //
    // The candle mirror of `mlx_fit_gate::synthesize_estimate_ladder`'s fitted arm. The manifest's
    // per-phase curves ARE the fitted model over this tier's measured cells, so an in-envelope
    // request geometry nobody measured gets an estimate candidate per optimized rung at the
    // curve-predicted peak, graded by the shared selector behind the candle ESTIMATE margin
    // (`crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`). Where the exact request cell has a
    // verified record, the selector's measured-supersedes-estimate rule keeps admission
    // byte-for-byte unchanged.
    //
    // A rung's fitted estimate is emitted only when EVERY `exact_request` record of this tier —
    // graded at ITS OWN geometry through the same `make_evidence` conjuncts as the request path —
    // passes the full measured eligibility predicate (`optimized_eligibility`). Whole-tier rather
    // than best-anchor, because the curve is fitted across every cell (see the fail-closed note at
    // the loop head). That check carries every restriction sc-18096 established for extrapolation
    // bases —
    // closure-CURRENT capture (a stale record may serve its own cell behind the stale margin but
    // may not seed an extrapolation; the estimate margin was derived over same-closure re-capture
    // variance and cannot also absorb closure drift), the loaded contract's calibration identity
    // (a drifted provider must not receive fitted candidates built from another identity's
    // records), artifact + hardware loadability, measured-composition agreement, parity, and
    // curve↔record phase agreement at the anchor cell (so a mutated curve that no longer describes
    // the measurement cannot smuggle its numbers back in as an estimate).
    //
    // `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` (sc-18094) is honored at this synthesis
    // seam: if the curve moves the request peak onto a different phase than the anchor record's
    // binding phase, the fitted candidate is NOT emitted. No weights+headroom floor path exists on
    // this lane (the constraint's scope exemption is therefore never exercised here): every rung
    // always has a curve, and a floor from the resident row could never admit deeper than the
    // resident baseline it equals, so a suppressed rung honestly falls out of estimate admission.
    let mut estimates: Vec<(MemorySelection, MemoryEvidence)> = Vec::new();
    for (strategy, phases, _, selection, from_anchor) in &measured {
        if *from_anchor {
            // Priced by the anchor derivation, not by the tier's curves: the fitted-estimate
            // synthesis below has nothing to say about it and must not relabel it `fitted`.
            continue;
        }
        let manifest_rung = krea_turbo_manifest_key(*strategy);
        let record_is_eligible = |record: &Value, record_width: u32, record_height: u32| {
            let anchor_geometry = MemoryGeometry {
                width: record_width,
                height: record_height,
                batch: 1,
                frames: KREA_LANE_FRAMES,
                reference_count: 0,
            };
            let Some(anchor_phases) = krea_rung_phase_peaks(
                manifest_entry,
                tier,
                *strategy,
                record_width,
                record_height,
                anchor_geometry.frames,
            ) else {
                return false;
            };
            // The FULL measured-eligibility predicate, not just `Verified` conformance: it
            // additionally requires the record's measured composition to agree with the loaded
            // contract's and the calibration identity to match (the mirror of
            // `mlx_fit_gate::collect_estimate_bases`' engaged-composition filter).
            make_evidence(
                *selection,
                Some(manifest_rung),
                anchor_phases.peak_gb(),
                Some(anchor_phases),
                Some(record),
                anchor_geometry,
            )
            .optimized_eligibility(provider_contract)
            .is_ok()
        };
        // FAIL CLOSED ON THE WHOLE TIER (sc-18097 review, major finding). The tier's phase curves
        // are fitted across EVERY measured cell, and the measured path's own
        // `exact_strategy_parameters` conjunct requires the record peak and the curve to agree
        // within 0.01 GiB — so the curve evaluated at an excluded cell's geometry reproduces that
        // cell's number. Anchoring on a surviving SIBLING record would therefore let a
        // structurally excluded cell (composition mismatch, drifted identity, stale capture) be
        // re-admitted at its own numbers behind nothing but the 4% estimate margin, laundering
        // the per-cell structural exclusion the epic requires to keep excluding. So a rung emits
        // fitted candidates only when EVERY `exact_request` record of the tier is eligible for it:
        // one bad row disqualifies the tier's curve for that rung, not merely that row.
        if let Some((_, bad_width, bad_height)) = tier_records
            .iter()
            .find(|(record, width, height)| !record_is_eligible(record, *width, *height))
        {
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                ineligible_record_geometry = format!("{bad_width}x{bad_height}"),
                "fitted-curve estimates suppressed for this rung: a measured record of this tier \
                 is structurally excluded, and the tier's curve is fitted across it"
            );
            continue;
        }
        let anchor = tier_records
            .iter()
            .copied()
            // Prefer the largest measured cell: the curve is fitted within `maxMeasuredPixels`,
            // and the top sample is the anchor closest to that envelope.
            .max_by_key(|(_, record_width, record_height)| {
                u64::from(*record_width) * u64::from(*record_height)
            });
        let Some((anchor_record, anchor_width, anchor_height)) = anchor else {
            continue;
        };
        let measured_binding =
            krea_record_phase_peaks(anchor_record, manifest_rung).map(krea_binding_phase);
        let request_binding = krea_binding_phase(*phases);
        if crate::ladder_margin_policy::ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE
            && measured_binding != Some(request_binding)
        {
            // The pinned sc-18094 constraint: the corpus shows a per-phase re-capture spread no
            // margin in the policy absorbs, so a curve that moves the request peak onto a
            // different phase than the one the anchor measured is refused rather than margined.
            tracing::info!(
                route = "krea_2_turbo",
                backend = "candle",
                ?strategy,
                anchor_geometry = format!("{anchor_width}x{anchor_height}"),
                ?measured_binding,
                request_binding,
                "fitted-curve estimate rejected: the curve moves the request peak onto a \
                 different phase than the measured anchor's \
                 (ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE)"
            );
            continue;
        }
        let predicted_peak_bytes = bytes(phases.peak_gb());
        tracing::info!(
            route = "krea_2_turbo",
            backend = "candle",
            ?strategy,
            anchor_geometry = format!("{anchor_width}x{anchor_height}"),
            raw_peak_bytes = predicted_peak_bytes,
            "synthesized fitted-curve estimate candidate from the anchored phase curves"
        );
        estimates.push((
            *selection,
            sceneworks_core::memory_contracts::unverified_estimate(
                MemoryEvidenceKey {
                    model_family: "krea_2_turbo".to_owned(),
                    resolved_route: "krea_2_turbo".to_owned(),
                    backend: gen_core::MemoryBackend::Candle,
                    tier: numeric_tier,
                    load_shape: provider_contract.load_shape,
                    mode: gen_core::MemoryMode::TextToImage,
                    reference_shape: gen_core::MemoryReferenceShape::None,
                    overlay: None,
                    geometry,
                    frames_per_second: None,
                    strategy: selection.strategy,
                    engaged_composition: provider_contract.engaged_composition(selection.strategy),
                    parameters: selection.parameters,
                },
                calibration_abi,
                calibration_fingerprint.to_owned(),
                scene_works_revision.to_owned(),
                predicted_peak_bytes,
            ),
        ));
    }
    let mut candidates = selections
        .iter()
        .zip(&evidence)
        .map(|(selection, evidence)| Candidate {
            selection: *selection,
            evidence,
            basis: memory_strategy::CandidateBasis::Measured,
            unmodeled_activation_bytes: None,
        })
        .collect::<Vec<_>>();
    candidates.extend(estimates.iter().map(|(selection, evidence)| Candidate {
        selection: *selection,
        evidence,
        basis: memory_strategy::CandidateBasis::EstimateFittedCurve,
        // A fitted per-phase curve carries no weights/activation split (sc-22508); its remaining
        // uncertainty is the same-cell recapture spread the policy charges on the whole peak.
        unmodeled_activation_bytes: None,
    }));
    // sc-22509: anchor-derived rungs.
    candidates.extend(
        anchor_derived
            .iter()
            .map(|(selection, evidence)| Candidate {
                selection: *selection,
                evidence,
                basis: memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: memory_strategy::AnchorDerivationLane::Image,
                },
                // No weights/activation split (sc-22508), for the same reason the fitted-curve
                // candidate above has none: the anchor derivation decomposes its peak by PHASE
                // (conditioning/denoise/decode), not into counted weights plus an activation
                // remainder. Only the manifest-row FLOOR declares that split, so the allowance is
                // charged on the whole peak here.
                unmodeled_activation_bytes: None,
            }),
    );
    // sc-22667: the operational reserve is charged ONCE, on the budget side, against every
    // candidate (`ReserveCharge::EveryCandidate`): none of this lane's candidates carries a pad
    // inside its peak — see the function doc. The legacy fixed `HEADROOM_GB` is gone from both
    // sides of the comparison.
    let selection = memory_strategy::select_strategy_charging(
        request,
        provider_contract,
        Some(Budget {
            available_gb: budget.free_gb,
            reclaimable_gb: 0.0,
            total_gb: budget.total_gb,
            reserved_headroom_gb: reserve_gb,
        }),
        &candidates,
        memory_strategy::ReserveCharge::EveryCandidate,
    );
    let admitted =
        |needed_gb: f64, available_gb: f64| crate::candle_memory_strategy::AdmittedBudget {
            needed_gb,
            available_gb,
            reserve_gb,
        };
    match selection {
        Selection::Selected {
            selection:
                selected @ MemorySelection {
                    strategy: MemoryStrategy::Resident,
                    ..
                },
            needed_gb,
            available_gb,
        } => Some(KreaTurboFit::Resident {
            peak_gb: resident_peak_gb,
            needed_gb: needed_gb + reserve_gb,
            selection: selected,
            admitted: admitted(needed_gb, available_gb),
        }),
        Selection::Selected {
            selection: selected,
            needed_gb,
            available_gb,
        } => {
            let (_, phases, _, _, from_anchor) = measured
                .into_iter()
                .find(|(_, _, _, selection, _)| selection.strategy == selected.strategy)?;
            // What priced the rung: the anchor derivation, else the request cell's own measured
            // record, else the tier's fitted curve (the only estimate an in-envelope cell without
            // a record can be carried by — see `Fits::estimate_scoped`).
            let basis = if from_anchor {
                memory_strategy::CandidateBasis::EstimateAnchorDerived {
                    lane: memory_strategy::AnchorDerivationLane::Image,
                }
            } else if evidence_record.is_some() {
                memory_strategy::CandidateBasis::Measured
            } else {
                memory_strategy::CandidateBasis::EstimateFittedCurve
            };
            let memory = gen_core::GenerationMemory {
                tile_vae_decode: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedDecode),
                chunk_attention: provider_contract
                    .engages(selected.strategy, MemoryStrategy::BoundedAttention),
                stream_transformer_blocks: provider_contract.engages(
                    selected.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                ),
                ..Default::default()
            };
            Some(KreaTurboFit::Fits {
                phases,
                needed_gb: needed_gb + reserve_gb,
                selection: selected,
                memory,
                basis,
                admitted: admitted(needed_gb, available_gb),
                // See the field doc: with no `exact_request` record at this geometry the rung's
                // measured candidates are structurally excluded, so only a synthesized estimate
                // can have carried it.
                estimate_scoped: evidence_record.is_none(),
            })
        }
        Selection::Reject { needed_gb, .. } => {
            let (_, phases, _, _, _) = measured.last().copied()?;
            Some(KreaTurboFit::Reject {
                phases,
                needed_gb: needed_gb + reserve_gb,
            })
        }
        // sc-18097 narrowed this arm's meaning: an in-envelope unmeasured geometry now carries a
        // fitted-curve estimate per rung, so the selector lands here only when a rung has NO
        // eligible candidate at all — a stale-closure or otherwise unverifiable manifest (whose
        // records may not seed extrapolation), a mutated record/curve pair, a binding-phase flip,
        // or an overlay the measurements do not cover. Those remain the explicit fallback to the
        // established generic gate, exactly as before.
        Selection::Unverified { reason } => Some(KreaTurboFit::Unverified { reason }),
    }
}

/// The reserve the threshold fixtures below were calibrated against: the legacy 2 GB, which is
/// exactly what [`ladder_reserve_gb`] produces for a card whose idle baseline is at or above
/// `HEADROOM_GB - LADDER_RESERVE_MARGIN_GB` (the ceiling arm). The reserve is an INPUT of the lane
/// since sc-22667 — production derives it from the raw probe — so the fixtures state it rather
/// than re-deriving a different number from their `free == total` budgets, which would move
/// every calibrated admission threshold for no reason connected to what they grade. The
/// derivation itself is graded by the `ladder_reserve_*` tests and by
/// `an_out_of_envelope_krea_request_charges_the_ladder_reserve_once`.
#[cfg(test)]
const KREA_FIXTURE_RESERVE_GB: f64 = HEADROOM_GB;

#[cfg(test)]
fn krea_turbo_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<KreaTurboFit> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        KREA_FIXTURE_RESERVE_GB,
        allow_streamed_blocks,
        Some(&runtime),
    )
}

/// Highest lower-pixel manifest bucket that the deepest available Krea Turbo rung can actually fit.
/// Used only to make rejection copy truthful: if this returns `None`, lowering resolution is not
/// presented as an escape hatch.
///
/// The advice is restricted to MEASURED admissions (sc-18097): an estimate-scoped `Fits` — the
/// fitted-curve admission of a geometry nobody measured — is not offered here, mirroring
/// `mlx_fit_gate::verified_lower_alternative` ("no formula, interpolation, tier heuristic, or
/// aspect-ratio rewrite is admitted"). Naming a fallback that itself rests on an estimate invites
/// a second refusal at the geometry the user was just told to switch to. `Resident` keeps its
/// pre-sc-18097 acceptance: that verdict is the manifest's geometry-independent `vramGbByTier`
/// row, which this helper has always been allowed to quote.
#[allow(clippy::too_many_arguments)]
pub(crate) fn krea_turbo_smaller_fit_with_runtime(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    reserve_gb: f64,
    allow_streamed_blocks: bool,
    runtime: Option<&KreaRuntimeEvidenceContext>,
) -> Option<(u32, u32)> {
    let current_pixels = u64::from(width) * u64::from(height);
    let resolutions = manifest_entry
        .get("limits")?
        .get("resolutions")?
        .as_array()?;
    let mut candidates = resolutions
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|resolution| resolution.split_once('x'))
        .filter_map(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|(w, h)| u64::from(*w) * u64::from(*h) < current_pixels)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(w, h)| std::cmp::Reverse(u64::from(*w) * u64::from(*h)));
    candidates.into_iter().find(|(w, h)| {
        matches!(
            krea_turbo_fit_with_runtime(
                manifest_entry,
                tier,
                *w,
                *h,
                budget,
                reserve_gb,
                allow_streamed_blocks,
                runtime,
            ),
            Some(
                KreaTurboFit::Resident { .. }
                    | KreaTurboFit::Fits {
                        estimate_scoped: false,
                        ..
                    }
            )
        )
    })
}

#[cfg(test)]
fn krea_turbo_smaller_fit(
    manifest_entry: &JsonObject,
    tier: &str,
    width: u32,
    height: u32,
    budget: Option<VramBudget>,
    allow_streamed_blocks: bool,
) -> Option<(u32, u32)> {
    let runtime = KreaRuntimeEvidenceContext::verified_for_test(tier);
    krea_turbo_smaller_fit_with_runtime(
        manifest_entry,
        tier,
        width,
        height,
        budget,
        KREA_FIXTURE_RESERVE_GB,
        allow_streamed_blocks,
        Some(&runtime),
    )
}

/// Resolve the [`LoadPlan`] for `needed` (resident peak) and `sequential_needed` (measured staged peak,
/// [`predicted_sequential_peak_gb`]) against `budget`. `None` peaks are unmeasured ⇒ never block (admit
/// resident, or stage best-effort). `sequential_capable` is the provider's staging capability: a lane
/// that cannot stage turns a resident overflow straight into [`LoadPlan::Reject`].
///
/// Monotonic in the budget: a larger `free_gb` can only move the plan `Reject → Sequential → Resident`,
/// never the other way — which is what makes "plan changed after reclaim ⇒ plan improved" sound.
pub(crate) fn load_plan(
    needed: Option<f64>,
    sequential_needed: Option<f64>,
    budget: Option<VramBudget>,
    sequential_capable: bool,
) -> LoadPlan {
    match resolve_offload(fit_decision(needed, budget), sequential_capable) {
        FitDecision::Offload { .. } => {
            if sequential_overflow_gb(sequential_needed, budget).is_some() {
                LoadPlan::Reject
            } else {
                LoadPlan::Sequential
            }
        }
        // Reached only when the engine cannot stage (resolve_offload leaves TooBig intact); a
        // sequential-capable lane rewrites TooBig → Offload above.
        FitDecision::TooBig { .. } => LoadPlan::Reject,
        // Fits (resident) or Unknown (unmeasured / no budget) — admit resident, never block.
        _ => LoadPlan::Resident,
    }
}

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode fit gate, candle/CUDA half (epic 1788 / sc-12306)
// ---------------------------------------------------------------------------------------------
//
// The candle twin of `mlx_fit_gate`'s Mochi gate. Both call the SAME arithmetic
// (`crate::fit_gate::mochi_needed_gb`); this half supplies the CUDA budget, the CUDA reserve, and the
// CUDA-worded message.
//
// Why Mochi needs its own gate here rather than riding `fit_decision` above:
//
//  1. `predicted_peak_gb` is MANIFEST-driven (`candle.vramGbByTier`), and **`mochi_1` has no `candle`
//     block at all** — it ships `mlx` only (builtin.models.jsonc:7233). So the generic gate returns
//     `None` ⇒ `Unknown` ⇒ admit. It could not protect Mochi even if the video lane called it. (No
//     candle VIDEO model has such a block; closing that is sc-12344.)
//
//  2. Even given a block, the generic gate is deliberately RESOLUTION-BLIND: a per-tier constant
//     calibrated at load time, where the seam cannot see the request geometry (the generator is cached
//     across resolutions). Mochi's AsymmVAE decode is UNTILED (candle-gen-mochi pipeline.rs:151 —
//     `vae.decode(&latents)` materializes the whole clip; sc-12291), so its peak grows LINEARLY IN CLIP
//     LENGTH: a 7-frame and a 151-frame request differ by ~55 GiB on the same model and the same tier.
//     A constant cannot express that.
//
//  3. Unlike the MLX lane, a CUDA OOM here is CATCHABLE (`classify_engine_error`) rather than MLX's
//     unmappable `exit(-1)` (sc-12178/12179). So this gate is not preventing a process kill — it is
//     preventing a raw allocator error that arrives only AFTER the full 64-step denoise has burned
//     minutes of GPU time, and that says nothing about the one lever that actually fixes it. At the
//     shipped 5 s / 151-frame default Mochi needs ~81 GB; an RTX 5090 has 32 GB, so on consumer
//     hardware this is not an edge case but the DEFAULT path.

/// The pure candle Mochi admission decision: `Some(error)` when the predicted peak overflows the VRAM
/// budget, `None` to admit. Missing either signal (unmeasurable weights / no budget) admits — the gate
/// never blocks without evidence, exactly like [`fit_decision`].
///
/// Budgets against `free_gb` (what is allocatable now), like [`fit_decision`] and unlike the MLX gate,
/// which has only a unified `total_gb`. Reserves [`HEADROOM_GB`] rather than the MLX gate's
/// the MLX unified reserve: the OS does not draw from discrete VRAM, so the term covers allocator slack and
/// CUDA context overhead instead. The two constants agree at 2.0 today but mean different things.
///
/// Pure (no GPU probe, no env) so the whole decision is unit-testable without CUDA; the caller resolves
/// the budget.
pub(crate) fn mochi_fit_error(
    model_label: &str,
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (
        crate::fit_gate::mochi_needed_gb(weight_bytes, frames, width, height, HEADROOM_GB)?,
        budget?,
    );
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        mochi_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            frames,
            width,
            height,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// Build Mochi's actionable over-budget rejection for the candle lane. Follows this module's existing
/// reject convention — name the model, state what it needs and what the GPU has — and adds the lever
/// that is UNIQUE to Mochi: the clip length.
///
/// The generic candle message's advice ("choose a smaller quant tier, lower the resolution") is nearly
/// useless here: Mochi has one trained bucket (848×480), and the decode dwarfs the tier delta (q4→bf16
/// is ~11 GiB against a ~60 GiB decode), so neither lever closes a 49 GB gap. The message therefore
/// leads with shortening the clip — the only lever that moves the dominant term.
///
/// Deliberately does NOT reuse `mlx_fit_gate::mochi_too_big_error`: that one says "unified memory" and
/// "run on a Mac with more memory", both false on CUDA. The shared thing between the lanes is the
/// arithmetic, not the prose.
#[allow(clippy::too_many_arguments)]
fn mochi_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    frames: u32,
    width: u32,
    height: u32,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render a {frames}-frame {width}x{height} clip \
         (~{weights} GB of weights, held resident for the whole run, plus an untiled VAE decode whose \
         peak grows with clip length) but GPU {gpu_id} has ~{available} GB available. Shorten the \
         clip — the decode peak scales roughly linearly with duration — or run on a card with more \
         VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

// ---------------------------------------------------------------------------------------------
// SVD candle-video long-burst gate (sc-14492)
// ---------------------------------------------------------------------------------------------

/// Pure SVD CUDA admission decision for the candle lane.
///
/// A 32 GB RTX PRO 4500 completed the supported/default 25-frame, decode-chunk-8, 25-step 1024x576
/// render after sc-14625's sequential CFG/residency and live-free VAE tiling changes. That complete
/// measured profile (or lower settings) is admitted on 32 GB; requests beyond any measured dimension
/// remain rejected on that card class. Missing telemetry admits, matching every other fit gate: no
/// invented signal means no rejection.
///
/// The check keys off physical `total_gb`, not momentary `free_gb`. This story establishes a hardware
/// boundary, not a trustworthy transient-working-set threshold for a busy card.
pub(crate) fn svd_fit_error(
    frames: u32,
    decode_chunk_size: u32,
    steps: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    if !crate::fit_gate::svd_profile_needs_larger_card(
        frames,
        decode_chunk_size,
        steps,
        width,
        height,
        budget.total_gb,
    ) {
        return None;
    }
    Some(WorkerError::InvalidPayload(format!(
        "Stable Video Diffusion's {frames}-frame, decode-chunk-{decode_chunk_size}, {steps}-step \
         {width}x{height} CUDA profile is outside the measured admission envelope on \
         GPU {gpu_id}'s {total} GB VRAM class. A real 32 GB RTX PRO 4500 completed up to \
         {validated_width}x{validated_height}, {validated_frames} frames, \
         decodeChunkSize={validated_chunk}, and {validated_steps} steps at a 17.521 GiB peak, so \
         even that bounded profile requires at least an {minimum} GB physical-card class. Requests \
         beyond any measured dimension remain unvalidated through 32 GB. Use a card and recipe \
         inside that measured envelope, or choose a larger GPU. The job was rejected before model \
         load.",
        total = budget.total_gb.round() as i64,
        validated_width = crate::fit_gate::SVD_32GB_VALIDATED_MAX_WIDTH,
        validated_height = crate::fit_gate::SVD_32GB_VALIDATED_MAX_HEIGHT,
        validated_frames = crate::fit_gate::SVD_32GB_VALIDATED_MAX_FRAMES,
        validated_chunk = crate::fit_gate::SVD_32GB_VALIDATED_MAX_DECODE_CHUNK,
        validated_steps = crate::fit_gate::SVD_32GB_VALIDATED_MAX_STEPS,
        minimum = crate::fit_gate::SVD_VALIDATED_PROFILE_MIN_VRAM_GB.round() as i64,
    )))
}

// ---------------------------------------------------------------------------------------------
// The Wan candle video weights-floor gate (epic 1788 / sc-12344)
// ---------------------------------------------------------------------------------------------
//
// Mochi above gets a bespoke gate because its decode peak is frame-dependent. Every OTHER candle video
// engine was admitted with NO pre-flight check at all until this: `predicted_peak_gb` is manifest-driven
// off `candle.vramGbByTier`, and **no candle video model carries a `candle` block**, so wiring the
// generic gate would return `None` ⇒ `Unknown` ⇒ admit for every one of them — dead code that reads as
// coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test warns about).
//
// So this budgets on the one signal that exists WITHOUT a measurement campaign: the on-disk weight
// bytes of the components the loader actually reads. Three facts make that a sound admission check for
// **Wan** rather than a guess:
//
//  1. **The engines this FLOOR still gates hold all components resident for the whole run.** ltx / svd /
//     mochi declare `supports_sequential_offload: false` (candle-gen-ltx / -svd / -mochi), so
//     `resolve_offload` is a no-op for them and Σweights is a genuine LOWER BOUND. **The Wan A14B T2V/I2V
//     and the dense TI2V-5B are the exception as of sc-12631 / epic sc-12732 (5B: sc-13175):** the candle
//     engine stages components **one-resident-at-a-time** (`render_sequential`,
//     `supports_sequential_offload: true`) — the A14B swaps TE → high-expert → low-expert → VAE (only ONE
//     14B expert live at a time, not both), the 5B flushes its TE + z48 VAE off-GPU around the dense
//     denoise (sc-12757) — so each is sized by its MEASURED sequential `candle.vramGbByTier` peak and this
//     floor never gates it (`wan_video_fit_error` takes the measured branch), while its load is forced
//     `OffloadPolicy::Sequential` (`video_jobs::candle_wan_offload_policy`) so the sized peak is the one
//     actually loaded. The floor stays the fallback for a Wan tier that was never measured.
//  2. **There is no host paging on CUDA.** Weights that do not fit VRAM cannot be demand-paged the way
//     MLX's unified pool swaps, so Σweights is a genuine LOWER BOUND on the job's need. This is the
//     asymmetry that makes the weights floor safe HERE but not on MLX's legacy transition override
//     exists to stop a pageable transient from wall-rejecting a small Mac (sc-12179), and that whole
//     class of false reject cannot arise on a discrete card.
//  3. Therefore rejecting when the weights alone overflow can never wall-reject a machine that would
//     have rendered — the sc-12179 regression this lane must not repeat.
//
// ⚠️ **(2) holds only where the on-disk bytes ARE the loaded set** — which is why [`wan_weight_bytes`]
// sums NAMED component subdirs and why `ltx`/`svd` are exempt. See that function's note; getting this
// wrong turns the floor from a lower bound into an over-count, i.e. straight into the sc-12179
// wall-reject.
//
// ⚠️ **The manifest's `footprint.peakMemoryBytes` for these models is an MLX measurement and MUST NOT be
// reused here.** `wan_2_2_t2v_14b`'s recorded 24.5 GiB peak sits BELOW its own `diskSizeBytes` because,
// as that manifest comment says, "A14B is MoE — only ONE 14B expert is resident at a time, not both".
// The candle A14B now ALSO holds one expert at a time (sc-12733, per (1)), but that number is still an
// MLX measurement at a different geometry/dtype and must not be reused: the candle lane is sized by its
// own MEASURED sequential `candle.vramGbByTier` (sc-12631). (Before sc-12732 the candle engine co-located
// both experts, which is why the pre-rework floor under-predicted it by a whole ~28 GiB bf16 expert.)
//
// ## This is a FLOOR — it under-predicts, deliberately — and it is now the FALLBACK, not the gate
// It counts weights only: no activation transient, no VAE decode, no attention working set. A MARGINAL
// job is still admitted and can still CUDA-OOM (catchably, via `classify_engine_error` — not MLX's
// unmappable `exit(-1)`). That was the honest bound from the data that existed before a measurement.
// [`wan_video_fit_error`] now prefers the MEASURED per-tier peak (`candle.vramGbByTier`, sc-12402) and
// only falls back here when a model/tier was never measured.
//
// ⚠️ **Point (1) above is WRONG for the DENSE tier, and sc-12402 measured how wrong.** "On-disk bytes ARE
// the loaded set" holds only for the PACKED q4/q8 tiers. `candle-gen-wan` casts dtypes on load
// (`wan14b.rs:49-52`: `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`, applied per-tensor by `mmap_var_builder`),
// so the dense `Wan-AI/*-Diffusers` experts — which ship **fp32**, not bf16: `transformer/…index.json`
// declares `total_size` 57,153,966,336 = 53.2 GiB for ONE 14B expert — HALVE on load, while the UMT5 TE
// DOUBLES (bf16 10.58 → f32 21.16 GiB) on EVERY tier. Measured error vs the real device set:
//
//   * packed q4/q8 — floor UNDER-counts the co-resident set by the transient it omits. Safe direction:
//     it just admits, the pre-gate status quo. For the sequential-offload Wan engines the relationship
//     even INVERTS: the 5B's on-disk q4 floor is 18.15 GiB, but its MEASURED SEQUENTIAL peak is only
//     ~12.1 GB pool high-water / 10.61 GiB USED_MEM_HIGH (sc-13175 — the denoise transient, with the bf16
//     TE + z48 VAE flushed off-GPU), so there the floor OVER-counts and the measured `candle.vramGbByTier`
//     block must win (it does; see below).
//   * dense bf16 — floor OVER-counts by ~44 GiB (117.5 GiB summed vs a ~75 GiB device set), which
//     wall-rejects a 96 GB card that would render. That is the sc-12179 class this gate's note claims
//     cannot arise here; it can, on this one tier.
//
// So the floor is kept for its ORIGINAL job — refusing the clearly-impossible where nothing was measured
// — and a measurement supersedes it wherever one exists. The dense over-count is why
// [`wan_video_fit_error`] does NOT `max` the two.
//
// Deliberately NOT reusing Mochi's `mochi_decode_peak_gb`: that term is specific to Mochi's untiled
// AsymmVAE (peak linear in clip length). Wan tiles/chunks differently, so applying it here would
// over-predict badly and wall-reject working hardware — hence this gate is frame-blind by construction.

/// The exact component subdirs a candle **Wan** engine's loader reads, or `None` for an engine whose
/// on-disk bytes are NOT its loaded set (see below) ⇒ no signal ⇒ the gate admits.
///
/// Each list is the set of dirs the provider enumerates with `sorted_safetensors`, which loads **every**
/// `.safetensors` it finds — no variant selection — so the sum over these dirs IS the resident set:
///  * 5B (`candle-gen-wan` lib.rs:122/141/124) — `transformer` + `text_encoder` + `vae`;
///  * A14B T2V/I2V (wan14b.rs:288/301/302/328) — the same plus `transformer_2`, the second MoE expert,
///    which is co-resident (see the module note) and must be counted.
///
/// Packed q4/q8 tiers and the dense `Wan-AI/*-Diffusers` snapshot use the SAME subdir names (the tier is
/// detected from tensor content — the `proj_out.scales` marker — not from the directory layout), so one
/// list serves both.
///
/// **Why NAMED subdirs and not a blind recursive sum of `model_dir`.** A blind sum cannot go inert,
/// which is attractive — but it can OVER-count, and over-counting is the direction that hurts: it
/// wall-rejects hardware that renders fine (sc-12179), whereas an under-count merely admits (the
/// pre-gate status quo). Naming the dirs makes a wrong name read `0` for that component — permissive,
/// the safe direction — and it also makes the tier-ROOT fallback safe: if no tier subdir resolved and
/// `model_dir` is a root holding `q4/`+`q8/`, there is no top-level `transformer/`, so this reads 0 and
/// admits rather than summing two tiers at once.
///
/// **Why `ltx_2_3_distilled` and `svd_xt` are `None` (sc-12344's recorded exemptions).** Their on-disk
/// bytes are categorically not their loaded set, so ANY floor built from a directory sum would
/// wall-reject working cards:
///  * **ltx dense** — `ltx_checkpoint()` (candle-gen-ltx lib.rs:129-159) picks exactly ONE root-level
///    file by substring rank (`distilled` > `bf16` > largest), skipping the `fp8`/`mixed`/lora/upscaler
///    siblings shipped beside it. The hosted `Lightricks/LTX-2.3` is ~146 GiB on disk
///    (`estimatedSizeBytes: 157004895813`) against a single-file load — summing it would refuse LTX on
///    every GPU in existence.
///  * **ltx packed tier** — reads 3 exact files (`transformer` / `connector` / `vae_decoder`,
///    tier.rs:152-173) while the tier dir also ships `vae_encoder` + `audio_vae` + `vocoder` +
///    `upsampler`, which the T2V render never loads (tier.rs:30).
///  * **svd_xt** — resolves ONE exact file per component (candle-gen-svd lib.rs:128-141), but the
///    upstream `stabilityai/stable-video-diffusion-img2vid-xt` snapshot ships `X.safetensors` AND
///    `X.fp16.safetensors` side by side in each of `unet`/`vae`/`image_encoder`, so a dir sum roughly
///    DOUBLES a ~8.9 GiB model — enough to false-reject a small card that runs it today.
///
/// Closing those two honestly needs the provider to own its split (`register_generators! { …;
/// footprint = … }` — gen-core's `PerComponentBytes`, whose own doc explains that a consumer guessing
/// component sizes is exactly this failure). None of the video crates registers one today, so that is a
/// cross-repo change on the inference monorepo: **sc-12397**. Not a fudge factor — the LTX dense case is
/// off by ~7x, not by a constant.
fn wan_weight_components(engine_id: &str) -> Option<&'static [&'static str]> {
    match engine_id {
        "wan2_2_ti2v_5b" => Some(&["transformer", "text_encoder", "vae"]),
        "wan2_2_t2v_14b" | "wan2_2_i2v_14b" => {
            Some(&["transformer", "transformer_2", "text_encoder", "vae"])
        }
        // `mochi_1` has its own frame-dependent gate above; `ltx_2_3_distilled` / `svd_xt` are exempt.
        _ => None,
    }
}

/// The on-disk bytes a candle Wan load holds resident: [`wan_weight_components`] summed under
/// `model_dir`. `0` for a non-Wan engine, or when nothing could be scanned ⇒ no signal ⇒ admit.
///
/// Reuses [`crate::mlx_fit_gate::sum_safetensors_bytes`] per component so the HF-cache symlink handling
/// (shards are symlinks into `blobs/`) and the AppleDouble `._*` skip are shared with the MLX lane
/// rather than re-implemented — the same reason sc-12306's Mochi gate reuses `mochi_resident_bytes`.
pub(crate) fn wan_weight_bytes(engine_id: &str, model_dir: &Path) -> u64 {
    wan_weight_components(engine_id).map_or(0, |components| {
        components
            .iter()
            .map(|component| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(component)))
            .sum()
    })
}

/// Exact resident-weight floor for the dedicated sequential Wan2.2 VACE-Fun provider. The shared
/// UMT5+VAE preparation phase is co-resident, then each complete expert is loaded one at a time. The
/// peak is therefore `max(shared, high + adapters, low + adapters)`, never the sum of both experts.
/// Missing any required component fails closed because this route has no calibrated CUDA peak yet.
pub(crate) fn wan_vace_fun_sequential_weight_bytes(
    model_dir: &Path,
    adapter_stage_bytes: u64,
) -> Result<u64, WorkerError> {
    let component = |name: &str| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(name));
    let high = component("transformer");
    let low = component("transformer_2");
    let text_encoder = component("text_encoder");
    let vae = component("vae");
    if [high, low, text_encoder, vae].contains(&0) {
        return Err(WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun admission cannot verify transformer, transformer_2, text_encoder, and vae weights; repair the model install before retrying."
                .to_owned(),
        ));
    }
    let shared = text_encoder.checked_add(vae).ok_or_else(|| {
        WorkerError::InvalidPayload("Wan2.2 VACE-Fun shared footprint overflowed u64.".to_owned())
    })?;
    let high = high.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun high-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    let low = low.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun low-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    Ok(shared.max(high).max(low))
}

/// The pure Wan candle video admission decision: `Some(error)` when the model's RESIDENT WEIGHTS alone
/// cannot fit the VRAM budget, `None` to admit. The non-Mochi twin of [`mochi_fit_error`], and pure for
/// the same reason — the caller resolves the budget, so the whole decision is unit-testable with no CUDA
/// driver and no GPU.
///
/// Missing either signal admits: no budget (`nvidia-smi` unreadable) or unmeasurable weights
/// (`weight_bytes == 0` — an exempt engine, or a dir that could not be scanned) ⇒ `None`, exactly like
/// [`fit_decision`]'s [`FitDecision::Unknown`]. A gate that blocks without evidence is a regression, not
/// a safety net.
pub(crate) fn video_weights_fit_error(
    model_label: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (video_weights_needed_gb(weight_bytes)?, budget?);
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        video_weights_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// The candle **video** admission decision (sc-12402): budget on the MEASURED per-tier peak
/// (`candle.vramGbByTier[tier_key]` + [`HEADROOM_GB`], via [`predicted_peak_gb`]) when the manifest
/// carries one, else fall back to the sc-12344 on-disk weights FLOOR ([`video_weights_fit_error`]).
///
/// This is the seam that makes the measured blocks LIVE. Until sc-12402 the video lane called the
/// floor directly and nothing read `candle.vramGbByTier` — so adding the blocks alone would have been
/// dead data that reads as coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test exists to
/// catch). Pure, like both halves it composes: the caller resolves the budget and the tier.
///
/// ## Why a measurement REPLACES the floor rather than composing with it (`max`)
///
/// A `max(measured, floor)` looks safer and is not: **the floor OVER-counts the dense tier**, so
/// maxing would re-introduce the exact wall-reject the measurement exists to remove.
///
/// The floor's premise — `wan_weight_components`' "on-disk bytes ARE the loaded set" — holds only for
/// the PACKED q4/q8 tiers. The dense `Wan-AI/*-Diffusers` snapshot the manifest routes `bf16` to ships
/// its experts in **fp32** (`transformer/…index.json` declares `total_size` 57,153,966,336 = 53.2 GiB
/// for ONE 14B expert), and `candle-gen-wan` casts on load — `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`
/// (wan14b.rs:49-52), applied per-tensor by `mmap_var_builder`. So the experts HALVE (57.2 → 28.6 GB
/// each) while the UMT5 TE DOUBLES (11.4 → 22.7 GB). Net, the dense floor sums ~117 GiB against a real
/// ~81 GB resident set and refuses a 96 GB card that renders it. (The packed tiers fail the other way
/// — the f32 TE doubling makes their floor UNDER-count — which is safe: a floor that under-counts just
/// admits, the pre-gate status quo.)
///
/// A real measurement is not a better proxy for the peak, it IS the peak; deferring to a byte-sum that
/// is provably wrong about dtype would be superstition. Where no measurement exists the floor is still
/// the best available signal, so it stays as the fallback.
///
/// Missing either signal admits, exactly like [`fit_decision`]'s [`FitDecision::Unknown`]: no budget
/// (`nvidia-smi` unreadable) ⇒ `None`, and an engine with neither a `candle` block nor countable
/// weights (`ltx`/`svd`) ⇒ `None` through the floor.
#[cfg(any(test, doc))]
pub(crate) fn wan_video_fit_error(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    wan_video_fit_error_with_adapter_bytes(
        model_label,
        manifest_entry,
        tier_key,
        weight_bytes,
        0,
        gpu_id,
        budget,
    )
}

/// Adapter-aware Wan admission. Packed callers pass the independently resident user stack; dense
/// callers pass zero because their factors are folded. The calibrated Lightning stack is already in
/// the manifest peak and must not be included here.
pub(crate) fn wan_video_fit_error_with_adapter_bytes(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    // Unmeasured (no `candle` block, or no row for this tier and no `minMemoryGb`) ⇒ the sc-12344
    // floor, byte-for-byte the shipped behavior.
    let Some(needed_gb) = predicted_peak_gb(manifest_entry, tier_key) else {
        return video_weights_fit_error(
            model_label,
            weight_bytes.saturating_add(adapter_bytes),
            gpu_id,
            budget,
        );
    };
    let needed_gb = needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let budget = budget?;
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb)
        .then(|| video_peak_too_big_error(model_label, tier_key, needed_gb, budget.free_gb, gpu_id))
}

/// Fail-closed admission for the MiniMax-H3 Candle lane. Every user-selectable tier has a
/// real-weight whole-render peak measured at the shipped 1344x768 geometry; admitting without the
/// exact tier row (or without a live free-VRAM reading) would make the catalog flip an unproved OOM
/// path. User LoRA bytes are independently resident and are charged above that baseline.
#[cfg(test)]
pub(crate) fn minimax_h3_fit_error(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    minimax_h3_fit_error_with_sparse(
        manifest_entry,
        tier_key,
        adapter_bytes,
        gpu_id,
        budget,
        None,
    )
}

pub(crate) fn minimax_h3_fit_error_with_sparse(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
    sparse: Option<&gen_core_h3::veda::workspace::WorkspaceBound>,
) -> Option<WorkerError> {
    const MEASURED_PIXELS: u64 = 1_032_192;
    let candle = manifest_entry.get("candle");
    let measured = candle
        .and_then(|value| value.get("measured"))
        .and_then(Value::as_bool);
    let measured_pixels = candle
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64);
    let peak_gb = candle
        .and_then(|value| value.get("vramGbByTier"))
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
        .filter(|value| value.is_finite() && *value > 0.0);
    let (true, Some(MEASURED_PIXELS), Some(peak_gb)) =
        (measured == Some(true), measured_pixels, peak_gb)
    else {
        return Some(WorkerError::InvalidPayload(format!(
            "MiniMax-H3 Candle admission is unavailable because the installed catalog has no \
             complete measured {tier_key} CUDA row at 1344x768. Refusing to load unproved \
             weights; update SceneWorks before retrying."
        )));
    };
    let Some(budget) = budget else {
        return Some(WorkerError::InvalidPayload(
            "MiniMax-H3 Candle admission could not read free GPU VRAM from nvidia-smi. Refusing \
             to load the measured tier without proving that it fits; verify the NVIDIA \
             driver/runtime and retry."
                .to_owned(),
        ));
    };
    // This catalog row remains the dense baseline; sparse has its own explicit additional
    // requirement including predictor loading, retained state and forced dense fallback.
    let needed_gb = crate::video_admission::veda_cold_peak_gb(
        peak_gb + HEADROOM_GB + adapter_bytes as f64 / BYTES_PER_GIB,
        sparse,
    );
    let sparse_reason = if sparse.is_some() {
        " plus the Veda predictor/workspace and safe dense fallback"
    } else {
        ""
    };
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        WorkerError::InvalidPayload(format!(
            "MiniMax-H3 needs ~{needed} GB of free GPU VRAM for its measured {tier_key} render \
             peak plus CUDA reserve{sparse_reason}, but GPU {gpu_id} has ~{available} GB available. Select a \
             smaller quant tier or use a GPU with more free VRAM.",
            needed = needed_gb.ceil() as u64,
            available = budget.free_gb.round() as u64,
        ))
    })
}

/// Fail-closed admission for the shared SCAIL-2 Candle bf16 package. Unlike the generic Wan gate,
/// SCAIL has no lower-memory Candle tier and no sequential/offload lifecycle, so falling back to
/// on-disk bytes or admitting an absent row would turn an incomplete catalog into a real OOM. The
/// exact production render owns `candle.vramGbByTier.bf16`; [`HEADROOM_GB`] remains the common CUDA
/// reserve added by every measured-peak gate.
#[cfg(test)]
pub(crate) fn scail2_video_fit_error(
    manifest_entry: &JsonObject,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    scail2_video_fit_error_with_adapter_bytes(manifest_entry, "bf16", 0, gpu_id, budget)
}

pub(crate) fn scail2_video_fit_error_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    const MEASURED_PIXELS: u64 = 832 * 480;
    let candle = manifest_entry.get("candle");
    let measured = candle
        .and_then(|value| value.get("measured"))
        .and_then(Value::as_bool);
    let peak_gb = candle
        .and_then(|value| value.get("vramGbByTier"))
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
        .filter(|value| value.is_finite() && *value > 0.0);
    let min_memory_gb = candle
        .and_then(|value| value.get("minMemoryGb"))
        .and_then(Value::as_u64);
    let measured_pixels = candle
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64);
    let (true, Some(peak_gb), Some(min_memory_gb), Some(MEASURED_PIXELS)) = (
        measured == Some(true),
        peak_gb,
        min_memory_gb,
        measured_pixels,
    ) else {
        return Some(WorkerError::InvalidPayload(
            format!(
                "SCAIL-2 Candle admission is unavailable because the installed catalog has no complete \
                 measured {tier_key} CUDA row (positive peak, 832x480 geometry, and minMemoryGb floor). \
                 Refusing to load the selected {tier_key} package; update SceneWorks before retrying."
            ),
        ));
    };
    let base_needed_gb = peak_gb + HEADROOM_GB;
    if (min_memory_gb as f64) + f64::EPSILON < base_needed_gb.ceil() {
        return Some(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Candle admission is unavailable because catalog minMemoryGb={min_memory_gb} \
             is below the measured {tier_key} CUDA peak plus reserve (~{} GB). Refusing to load the \
             selected {tier_key} package; update SceneWorks before retrying.",
            base_needed_gb.ceil() as u64,
        )));
    }
    let needed_gb = base_needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let Some(budget) = budget else {
        return Some(WorkerError::InvalidPayload(
            format!(
                "SCAIL-2 Candle admission could not read free GPU VRAM from nvidia-smi. Refusing to \
                 load the selected {tier_key} package without proving that its measured render peak \
                 fits; verify the NVIDIA driver/runtime and retry."
            ),
        ));
    };
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        WorkerError::InvalidPayload(format!(
            "SCAIL-2 shared {tier_key} needs ~{needed} GB of free GPU VRAM (the measured 832x480, \
             81-frame render peak plus CUDA reserve), but GPU {gpu_id} has ~{available} GB \
             available. Select a measured SCAIL-2 tier that fits, or use a GPU with more free VRAM.",
            needed = needed_gb.ceil() as i64,
            available = budget.free_gb.round() as i64,
        ))
    })
}

/// Build the MEASURED-peak candle video rejection (sc-12402). The measured sibling of
/// [`video_weights_too_big_error`], and worded apart from it on purpose: that one can only honestly
/// speak about weights, this one is the whole render's ceiling (weights + denoise + the budget-tiled
/// VAE decode), so it names the render rather than the weights.
///
/// Names the TIER, because the tier is the lever that moves this number and the gate now knows which
/// one it sized. Like the floor's message it does NOT offer "lower the resolution": the measured peak
/// is a per-tier CONSTANT taken at the model's default geometry (the manifest schema's "video =
/// default frames"), and Wan's decode is budget-tiled (`auto_tiling_budgeted_wan22`) so the weights
/// dominate and a smaller clip cannot move the number the user was just shown. That is the same
/// resolution-blind contract the image lane's `vramGbByTier` gate has always had, and the reason Mochi
/// needs its own frame-scaled gate instead of this one (its untiled AsymmVAE decode IS the peak).
fn video_peak_too_big_error(
    model_label: &str,
    tier_key: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render at its {tier_key} tier (the measured peak of \
         the whole render — the resident weights plus the denoise and VAE-decode transient, at whatever \
         component residency the engine uses) but GPU {gpu_id} has ~{available} GB available. \
         Select a smaller quant tier, or run on a GPU with more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
}

/// The predicted resident FLOOR (GiB) for a candle video job: the on-disk weights every component of
/// this engine holds for the whole run, plus [`HEADROOM_GB`].
///
/// [`HEADROOM_GB`] is the CUDA reserve — allocator slack + CUDA context overhead — not MLX's
/// unified reserve (the OS does not draw from discrete VRAM). The two may agree numerically but mean
/// different things; this is the same split `fit_gate::mochi_needed_gb` takes its `reserve_gb`
/// parameter for (sc-12306).
///
/// `None` when nothing was measured (`weight_bytes == 0`) ⇒ no signal ⇒ never block.
fn video_weights_needed_gb(weight_bytes: u64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// Build the generic candle video weights-floor rejection. Names the model, what its weights alone
/// need, and what the card has.
///
/// The levers are WEIGHTS levers, and only those. It deliberately does NOT say "lower the output
/// resolution": resolution has exactly ZERO effect on weight bytes, so offering it here would send
/// the user to a knob that cannot move the number they were just shown. Nor does it say Mochi's
/// "shorten the clip" — same reason (there is no decode term in this floor). The tier IS the lever
/// that moves weights, plus the card itself. The image lane's `vram_reject_tail` used to offer the
/// knob unconditionally against its equally resolution-blind `vramGbByTier` rows; issue #3197 is
/// what that cost, and it now takes an explicit `ResolutionLever` for the same reason stated here.
fn video_weights_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs at least ~{needed} GB of VRAM just to hold its weights (~{weights} GB, \
         every component held resident for the whole run — this engine does not stage components) but \
         GPU {gpu_id} has ~{available} GB available. Select a smaller quant tier, or run on a GPU with \
         more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

#[cfg(test)]
mod tests;
