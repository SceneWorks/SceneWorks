//! vram_gate budget owner.
use super::*;

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
pub(super) fn reclaimable_pool_store(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, f64>> {
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
    let raw = match sceneworks_core::payload_util::quant_selection(advanced) {
        sceneworks_core::payload_util::QuantSelection::Explicit(bits)
        | sceneworks_core::payload_util::QuantSelection::Bound(bits) => Some(bits),
        _ => None,
    }
    .or_else(|| {
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
pub(super) fn tier_floor_gb(candle: &Value, tier_key: &str) -> Option<f64> {
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
