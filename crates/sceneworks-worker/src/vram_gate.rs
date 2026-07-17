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
//! candle's CUDA backend uses cudarc's stream-ordered caching allocator: there is no `empty_cache`, and
//! `Device::synchronize()` does not reclaim. So a freed component's pages return to candle's in-process
//! pool (reused by the next allocation — which is what keeps a small card from OOMing under Phase 1
//! sequential residency), but the DRIVER-resident `nvidia-smi` free number does NOT drop back. This
//! gate is therefore a pre-load ADMISSION check keyed off the manifest's measured per-tier peak, never
//! a post-free accounting number.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and
//! the wiring is in `generate_candle_stream` (image_jobs/base.rs).

use super::*;
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
const HEADROOM_GB: f64 = 2.0;

/// A live (or capped) VRAM budget for the selected GPU, in GB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VramBudget {
    pub free_gb: f64,
    pub total_gb: f64,
}

/// Read the small-card cap from the environment. `Some(gb)` only for a positive number.
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

/// Fold this process's RECLAIMABLE in-process VRAM pool into the live budget (sc-11023): add
/// `reclaimable_gb` to `free_gb`, clamped to the physical `total_gb`. The candle generator cache is a
/// single exclusive slot that evicts its current occupant BEFORE loading the incoming model, and
/// cudarc's caching allocator reuses the freed pages in-process (nvidia-smi `free` never rises after an
/// evict — see the module caveat). So the VRAM this process already holds will be available to the next
/// load; without this, a warm same-model re-gate or a swap-in is measured against `free` that still
/// counts the model it is about to replace, falsely rejecting a load that fits. A no-op when
/// `reclaimable_gb == 0` (nothing loaded yet), so a genuine cold first-load is gated exactly as before.
pub(crate) fn with_reclaimable(budget: VramBudget, reclaimable_gb: f64) -> VramBudget {
    VramBudget {
        free_gb: (budget.free_gb + reclaimable_gb.max(0.0)).min(budget.total_gb),
        total_gb: budget.total_gb,
    }
}

/// Per-GPU high-water of the peak VRAM (GB) this process has admitted a load at — the size of the
/// reclaimable cudarc pool (sc-11023). Keyed by `gpu_id` (a worker process is pinned to one card, but
/// keying is defensive + explicit). The pool only ever grows (no `empty_cache`/trim), so the running
/// MAX is exactly what a later swap-in can reuse.
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
    match raw {
        None => "q8",
        Some(bits) if bits <= 0 => "bf16",
        Some(bits) if bits <= 4 => "q4",
        Some(_) => "q8",
    }
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
    let candle = manifest_entry.get("candle")?;
    let measured = |key: &str| {
        candle
            .get("vramGbByTier")
            .and_then(|tiers| tiers.get(key))
            .and_then(json_f64)
    };
    if let Some(gb) = measured(tier_key) {
        return Some(gb + HEADROOM_GB);
    }
    // NVFP4 with no measured row → the q8 row (a deliberate over-prediction; see the note above).
    if tier_key == NVFP4_TIER {
        if let Some(gb) = measured("q8") {
            return Some(gb + HEADROOM_GB);
        }
    }
    candle.get("minMemoryGb").and_then(json_f64)
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
    let sequential = manifest_entry.get("candle")?.get("sequentialPeakGb")?;
    let measured = |key: &str| sequential.get(key).and_then(json_f64);
    measured(tier_key)
        .or_else(|| (tier_key == NVFP4_TIER).then(|| measured("q8")).flatten())
        .map(|gb| gb + HEADROOM_GB)
}

/// Decide whether the predicted peak fits the (possibly capped) live budget. Missing either input ⇒
/// `Unknown` (never block). Compares against `free_gb` — what is actually allocatable now — mirroring
/// the flux2 edit guard's use of `available_gb`.
pub(crate) fn fit_decision(needed_gb: Option<f64>, budget: Option<VramBudget>) -> FitDecision {
    let (Some(needed_gb), Some(budget)) = (needed_gb, budget) else {
        return FitDecision::Unknown;
    };
    if budget.free_gb + f64::EPSILON < needed_gb {
        FitDecision::TooBig {
            needed_gb,
            available_gb: budget.free_gb,
        }
    } else {
        FitDecision::Fits
    }
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
    let (needed_gb, budget) = (sequential_needed_gb?, budget?);
    (budget.free_gb + f64::EPSILON < needed_gb).then_some(needed_gb)
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
/// `OS_RESERVE_GB`: the OS does not draw from discrete VRAM, so the term covers allocator slack and
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
    (budget.free_gb + f64::EPSILON < needed_gb).then(|| {
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
//  1. **Every candle video engine holds all components resident for the whole run.** All six declare
//     `supports_sequential_offload: false` (candle-gen-wan lib.rs:417 / wan14b.rs:727 / model_vace.rs:387,
//     candle-gen-ltx lib.rs:527, candle-gen-svd lib.rs:183, candle-gen-mochi lib.rs:115, at the pinned
//     `runtime-2026.07.6`), which `video_load_spec`'s "video providers have not wired sequential
//     residency (sc-10821)" states in-tree. So `resolve_offload(TooBig, false)` is a no-op here and NO
//     sequential-peak second stage applies — the `candle.sequentialPeakGb` this gate would otherwise
//     need does not enter the decision at all. In particular the A14B MoE holds **BOTH** experts
//     co-resident (`Components { high: Arc<_>, low: Arc<_> }`, wan14b.rs:337-343): the denoise picks an
//     expert per step, it does not swap them out.
//  2. **There is no host paging on CUDA.** Weights that do not fit VRAM cannot be demand-paged the way
//     MLX's unified pool swaps, so Σweights is a genuine LOWER BOUND on the job's need. This is the
//     asymmetry that makes the weights floor safe HERE but not on MLX: `mlx_fit_gate::weights_fit_floor`
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
// That is true of the MLX engine (measured on a 128 GB Mac, sc-10049/epic 10043) and FALSE of the candle
// one, which co-locates both experts per (1). Sizing the candle lane off that number would under-predict
// by a whole expert (~28 GiB at bf16).
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
//   * packed q4/q8 — floor UNDER-counts by ~9-11 GiB (the f32 TE). Safe direction: it just admits, the
//     pre-gate status quo. (It is also nowhere near the real peak: the 5B q4's floor is 18.13 GiB and its
//     MEASURED peak is 86.3 GB — the denoise's unchunked attention, not the weights, is the ceiling.)
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
    (budget.free_gb + f64::EPSILON < needed_gb).then(|| {
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
pub(crate) fn wan_video_fit_error(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    // Unmeasured (no `candle` block, or no row for this tier and no `minMemoryGb`) ⇒ the sc-12344
    // floor, byte-for-byte the shipped behavior.
    let Some(needed_gb) = predicted_peak_gb(manifest_entry, tier_key) else {
        return video_weights_fit_error(model_label, weight_bytes, gpu_id, budget);
    };
    let budget = budget?;
    (budget.free_gb + f64::EPSILON < needed_gb)
        .then(|| video_peak_too_big_error(model_label, tier_key, needed_gb, budget.free_gb, gpu_id))
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
        "{model_label} needs ~{needed} GB of VRAM to render at its {tier_key} tier (measured peak: \
         every component held resident for the whole run — this engine does not stage components — \
         plus the denoise and VAE-decode transient) but GPU {gpu_id} has ~{available} GB available. \
         Select a smaller quant tier, or run on a GPU with more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
}

/// The predicted resident FLOOR (GiB) for a candle video job: the on-disk weights every component of
/// this engine holds for the whole run, plus [`HEADROOM_GB`].
///
/// [`HEADROOM_GB`] is the CUDA reserve — allocator slack + CUDA context overhead — not MLX's
/// `OS_RESERVE_GB` (the OS does not draw from discrete VRAM). The two agree at 2.0 today but mean
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
/// resolution" like the image lane's `vram_reject_tail`: resolution has exactly ZERO effect on weight
/// bytes, so offering it here would send the user to a knob that cannot move the number they were just
/// shown. Nor does it say Mochi's "shorten the clip" — same reason (there is no decode term in this
/// floor). The tier IS the lever that moves weights, plus the card itself.
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

/// Parse a JSON uint/int from either a number or a numeric string (mirrors base.rs `quant_int`).
fn quant_int(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
}

/// Parse a JSON float from either a number or a numeric string.
fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.trim().parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: Value) -> JsonObject {
        value.as_object().expect("object literal").clone()
    }

    #[test]
    fn parse_vram_cap_accepts_positive_numbers_only() {
        assert_eq!(parse_vram_cap(Some("10")), Some(10.0));
        assert_eq!(parse_vram_cap(Some("  8.5 ")), Some(8.5));
        assert_eq!(parse_vram_cap(Some("0")), None);
        assert_eq!(parse_vram_cap(Some("-4")), None);
        assert_eq!(parse_vram_cap(Some("nan")), None);
        assert_eq!(parse_vram_cap(Some("abc")), None);
        assert_eq!(parse_vram_cap(Some("")), None);
        assert_eq!(parse_vram_cap(None), None);
    }

    #[test]
    fn apply_vram_cap_emulates_a_smaller_card() {
        let real = VramBudget {
            free_gb: 90.0,
            total_gb: 96.0,
        };
        // No cap ⇒ unchanged.
        assert_eq!(apply_vram_cap(Some(real), None), Some(real));
        // Cap below real free ⇒ total = cap, free clamped to cap.
        assert_eq!(
            apply_vram_cap(Some(real), Some(10.0)),
            Some(VramBudget {
                free_gb: 10.0,
                total_gb: 10.0,
            })
        );
        // Real free already below the cap ⇒ free preserved, total = cap.
        assert_eq!(
            apply_vram_cap(
                Some(VramBudget {
                    free_gb: 6.0,
                    total_gb: 96.0,
                }),
                Some(10.0)
            ),
            Some(VramBudget {
                free_gb: 6.0,
                total_gb: 10.0,
            })
        );
        // No live reading + a cap ⇒ synthetic full budget (exercisable in a no-GPU test).
        assert_eq!(
            apply_vram_cap(None, Some(10.0)),
            Some(VramBudget {
                free_gb: 10.0,
                total_gb: 10.0,
            })
        );
        // No reading, no cap ⇒ None.
        assert_eq!(apply_vram_cap(None, None), None);
    }

    #[test]
    fn requested_tier_key_mirrors_the_subdir_resolvers() {
        let empty = obj(json!({}));
        // No pick anywhere ⇒ q8 default (sc-10726).
        assert_eq!(requested_tier_key(&empty, &empty, false), "q8");
        // advanced.mlxQuantize wins, number or numeric string.
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 0})), &empty, false),
            "bf16"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 4})), &empty, false),
            "q4"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 8})), &empty, false),
            "q8"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": "4"})), &empty, false),
            "q4"
        );
        // Falls back to the manifest mlx.quantize when advanced is silent.
        assert_eq!(
            requested_tier_key(&empty, &obj(json!({"mlx": {"quantize": 4}})), false),
            "q4"
        );
    }

    /// sc-11042 (epic 11037 SC#5): the fit gate keys off the tier IDENTITY, never bits.
    ///
    /// The FOURTH bits-derived site of the same aliasing. NVFP4 carries no `mlxQuantize` by design (no
    /// integer is honest for a ~4.5-effective-bit tier), so a bits-derived key returned `"q8"` for it
    /// and sized an NVFP4 render against `vramGbByTier["q8"]` — ~2× its real footprint. Conservative
    /// (spurious TooBig/Offload, never an OOM), but wrong, and it made this the one selection site that
    /// disagreed with the load quant + the recorded label about which tier was running.
    #[test]
    fn requested_tier_key_honors_the_nvfp4_tier_identity() {
        let empty = obj(json!({}));
        // The selected NVFP4 tier short-circuits the bits map — including the `mlxQuantize: null` an
        // NVFP4 request actually carries (sc-12006), which `quant_int` reads as "no pick" ⇒ `q8`.
        assert_eq!(requested_tier_key(&empty, &empty, true), NVFP4_TIER);
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": null})), &empty, true),
            NVFP4_TIER
        );
        // …and it is NOT the q4 whose numerics it must never be confused with, nor the q8 the
        // bits-derived key produced.
        assert_ne!(requested_tier_key(&empty, &empty, true), "q4");
        assert_ne!(requested_tier_key(&empty, &empty, true), "q8");

        // `nvfp4 = false` ⇒ every existing mapping is byte-identical (the caller's `nvfp4_selected` is
        // false for every request that didn't explicitly pick the tier on eligible hardware with the
        // tier installed — i.e. all of them today).
        for (advanced, manifest, expected) in [
            (json!({}), json!({}), "q8"),
            (json!({"mlxQuantize": 0}), json!({}), "bf16"),
            (json!({"mlxQuantize": 4}), json!({}), "q4"),
            (json!({"mlxQuantize": 8}), json!({}), "q8"),
            (json!({}), json!({"mlx": {"quantize": 4}}), "q4"),
            // A `quantTier: "nvfp4"` label with the gate closed (not Blackwell, or the tier isn't
            // installed) sizes the tier that will actually load — never nvfp4.
            (json!({"quantTier": "nvfp4"}), json!({}), "q8"),
        ] {
            assert_eq!(
                requested_tier_key(&obj(advanced.clone()), &obj(manifest.clone()), false),
                expected,
                "advanced={advanced} manifest={manifest}"
            );
        }
    }

    /// sc-11042: an `nvfp4` tier with no measured `vramGbByTier` row degrades CONSERVATIVELY — to the
    /// `q8` row, not to `minMemoryGb`.
    ///
    /// `minMemoryGb` is the manifest's DEFAULT (lightest, typically q4) tier peak, which heavier tiers
    /// are explicitly allowed to exceed — landing there would UNDER-predict and admit a load that can
    /// OOM. The q8 row over-predicts (q8's weights are ~2× NVFP4's), which is exactly the number this
    /// gate already used for an NVFP4 request when the bits-derived key returned `"q8"`. sc-11043 must
    /// add the real `nvfp4` rows when it converts a tier; until then this is the documented behavior.
    #[test]
    fn nvfp4_without_a_measured_row_degrades_to_q8_not_min_memory() {
        let manifest = obj(json!({
            "candle": {
                "minMemoryGb": 40,
                "vramGbByTier": { "q4": 47.2, "q8": 58, "bf16": 72 },
                "sequentialPeakGb": { "q4": 20.0, "q8": 26.0 }
            }
        }));
        // No `nvfp4` row ⇒ the q8 row + headroom, NOT the lighter (permissive) minMemoryGb…
        assert_eq!(
            predicted_peak_gb(&manifest, NVFP4_TIER),
            Some(58.0 + HEADROOM_GB)
        );
        assert_ne!(predicted_peak_gb(&manifest, NVFP4_TIER), Some(40.0));
        // …and the same for the second-stage sequential gate.
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, NVFP4_TIER),
            Some(26.0 + HEADROOM_GB)
        );

        // Once sc-11043 measures the tier, its own row wins outright.
        let measured = obj(json!({
            "candle": {
                "minMemoryGb": 40,
                "vramGbByTier": { "q8": 58, "nvfp4": 31.5 },
                "sequentialPeakGb": { "q8": 26.0, "nvfp4": 14.0 }
            }
        }));
        assert_eq!(
            predicted_peak_gb(&measured, NVFP4_TIER),
            Some(31.5 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_sequential_peak_gb(&measured, NVFP4_TIER),
            Some(14.0 + HEADROOM_GB)
        );

        // The q8 back-stop is NVFP4-only: every other tier keeps the unchanged
        // measured-row → minMemoryGb chain.
        let sparse = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": { "q8": 58 } } }));
        assert_eq!(predicted_peak_gb(&sparse, "q4"), Some(40.0));
        assert_eq!(predicted_peak_gb(&sparse, "bf16"), Some(40.0));
        assert_eq!(predicted_sequential_peak_gb(&sparse, "q4"), None);
        // No q8 row either ⇒ nvfp4 falls all the way through to minMemoryGb rather than panicking.
        let no_q8 = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": {} } }));
        assert_eq!(predicted_peak_gb(&no_q8, NVFP4_TIER), Some(40.0));
        assert_eq!(predicted_sequential_peak_gb(&no_q8, NVFP4_TIER), None);
    }

    #[test]
    fn predicted_peak_prefers_measured_tier_then_min_memory() {
        let manifest = obj(json!({
            "candle": {
                "minMemoryGb": 56,
                "vramGbByTier": { "q4": 47.2, "q8": 58, "bf16": 72 }
            }
        }));
        // Measured tier peak + headroom.
        assert_eq!(predicted_peak_gb(&manifest, "q4"), Some(47.2 + HEADROOM_GB));
        // Missing tier in vramGbByTier ⇒ minMemoryGb (no extra headroom, already padded).
        let sparse = obj(json!({ "candle": { "minMemoryGb": 40, "vramGbByTier": {} } }));
        assert_eq!(predicted_peak_gb(&sparse, "q4"), Some(40.0));
        // No candle block ⇒ unmeasured ⇒ None (gate no-ops).
        assert_eq!(predicted_peak_gb(&obj(json!({})), "q4"), None);
    }

    /// sc-12090 numeric regression: `krea_2_turbo` Q4-only on a ~30 GB card. Budgeting the tier the
    /// disk-probing resolver returns (`q4`) ADMITS (26.4 + 2 = 28.4 ≤ 30), where the old manifest
    /// re-derivation budgeted `q8` (35.9 + 2 = 37.9) and false-rejected. This pins the fit math the
    /// gate now feeds the ON-DISK tier (`tier_key_from_resolved_dir`), not the manifest's q8 default.
    #[test]
    fn resolved_q4_admits_where_manifest_q8_would_reject() {
        // Krea 2 Turbo candle tiers (builtin.models.jsonc, measured — sc-12126).
        let manifest = obj(json!({
            "candle": { "vramGbByTier": { "q4": 26.4, "q8": 35.9 } }
        }));
        let budget = VramBudget {
            free_gb: 30.0,
            total_gb: 32.0,
        };
        // The resolved on-disk tier is q4 (only q4 installed) → admits.
        assert_eq!(
            fit_decision(predicted_peak_gb(&manifest, "q4"), Some(budget)),
            FitDecision::Fits
        );
        // The old manifest-derived q8 (never installed) would have false-rejected the SAME job.
        assert!(matches!(
            fit_decision(predicted_peak_gb(&manifest, "q8"), Some(budget)),
            FitDecision::TooBig { .. }
        ));
    }

    #[test]
    fn fit_decision_rejects_only_a_genuine_overflow() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        assert_eq!(fit_decision(Some(8.0), Some(budget)), FitDecision::Fits);
        // Exactly-fits is not a rejection.
        assert_eq!(fit_decision(Some(10.0), Some(budget)), FitDecision::Fits);
        assert_eq!(
            fit_decision(Some(49.2), Some(budget)),
            FitDecision::TooBig {
                needed_gb: 49.2,
                available_gb: 10.0,
            }
        );
        // Missing either input ⇒ never block.
        assert_eq!(fit_decision(None, Some(budget)), FitDecision::Unknown);
        assert_eq!(fit_decision(Some(8.0), None), FitDecision::Unknown);
    }

    #[test]
    fn resolve_offload_rewrites_toobig_only_when_sequential_capable() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        let too_big = fit_decision(Some(40.0), Some(budget));
        assert!(matches!(too_big, FitDecision::TooBig { .. }));
        // Sequential-capable (the candle FLUX lane) ⇒ Offload instead of reject, carrying the numbers.
        assert_eq!(
            resolve_offload(too_big.clone(), true),
            FitDecision::Offload {
                needed_gb: 40.0,
                available_gb: 10.0,
            }
        );
        // Not capable ⇒ unchanged TooBig (still rejects).
        assert!(matches!(
            resolve_offload(too_big, false),
            FitDecision::TooBig { .. }
        ));
        // Fits / Unknown are never rewritten, regardless of capability.
        assert_eq!(resolve_offload(FitDecision::Fits, true), FitDecision::Fits);
        assert_eq!(
            resolve_offload(FitDecision::Unknown, true),
            FitDecision::Unknown
        );
    }

    #[test]
    fn predicted_sequential_peak_reads_the_measured_tier_plus_headroom() {
        let manifest = obj(json!({
            "candle": {
                "vramGbByTier": { "q4": 47.2, "q8": 58.0, "bf16": 72.0 },
                "sequentialPeakGb": { "q4": 30.0, "q8": 40.0, "bf16": 55.0 }
            }
        }));
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, "q4"),
            Some(30.0 + HEADROOM_GB)
        );
        assert_eq!(
            predicted_sequential_peak_gb(&manifest, "bf16"),
            Some(55.0 + HEADROOM_GB)
        );
        // Tier absent from sequentialPeakGb ⇒ None (best-effort run, no reject).
        let sparse = obj(json!({ "candle": { "sequentialPeakGb": { "q4": 30.0 } } }));
        assert_eq!(predicted_sequential_peak_gb(&sparse, "q8"), None);
        // No sequentialPeakGb block ⇒ None (today's behavior: resident-only gate).
        let resident_only = obj(json!({ "candle": { "vramGbByTier": { "q4": 47.2 } } }));
        assert_eq!(predicted_sequential_peak_gb(&resident_only, "q4"), None);
        // No candle block ⇒ None.
        assert_eq!(predicted_sequential_peak_gb(&obj(json!({})), "q4"), None);
    }

    #[test]
    fn sequential_overflow_rejects_only_a_measured_genuine_overflow() {
        let budget = VramBudget {
            free_gb: 10.0,
            total_gb: 10.0,
        };
        // Measured sequential peak still exceeds the budget ⇒ reject, carrying the number.
        assert_eq!(sequential_overflow_gb(Some(32.0), Some(budget)), Some(32.0));
        // Sequential peak fits ⇒ proceed (None). Exactly-fits is not an overflow.
        assert_eq!(sequential_overflow_gb(Some(8.0), Some(budget)), None);
        assert_eq!(sequential_overflow_gb(Some(10.0), Some(budget)), None);
        // Unmeasured tier ⇒ best-effort run (None), even when the card is tiny.
        assert_eq!(sequential_overflow_gb(None, Some(budget)), None);
        // No live budget ⇒ never block (None).
        assert_eq!(sequential_overflow_gb(Some(32.0), None), None);
    }

    #[test]
    fn with_reclaimable_adds_the_pool_and_clamps_to_total() {
        let budget = VramBudget {
            free_gb: 14.0,
            total_gb: 96.0,
        };
        // A resident ~82 GB model is reclaimable → the incoming load sees free + 82, capped to total.
        // This is the sc-11023 fix: raw free (14) would reject a bf16 re-load that actually fits.
        assert_eq!(
            with_reclaimable(budget, 82.0),
            VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }
        );
        // Partial reclaim stays below total.
        assert_eq!(
            with_reclaimable(budget, 20.0),
            VramBudget {
                free_gb: 34.0,
                total_gb: 96.0,
            }
        );
        // Nothing reclaimable (cold start) ⇒ budget unchanged.
        assert_eq!(with_reclaimable(budget, 0.0), budget);
        // Negative is treated as zero (defensive).
        assert_eq!(with_reclaimable(budget, -5.0), budget);
        // Never exceeds the physical total, even with a huge/stale high-water.
        assert_eq!(
            with_reclaimable(budget, 1000.0),
            VramBudget {
                free_gb: 96.0,
                total_gb: 96.0,
            }
        );
    }

    #[test]
    fn reclaimable_pool_is_a_per_gpu_monotonic_high_water() {
        // Distinct gpu_ids so this test can't race the process-global against other tests.
        let gpu = "test-reclaimable-gpu-a";
        let other = "test-reclaimable-gpu-b";
        // Nothing loaded yet ⇒ 0 (so `with_reclaimable` no-ops on a cold card).
        assert_eq!(reclaimable_pool_gb(gpu), 0.0);
        note_loaded_peak(gpu, 30.0);
        assert_eq!(reclaimable_pool_gb(gpu), 30.0);
        // A bigger load raises the high-water…
        note_loaded_peak(gpu, 82.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // …a smaller later load does NOT lower it — the cudarc pool never returns pages to the driver.
        note_loaded_peak(gpu, 54.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // Non-positive peaks are ignored.
        note_loaded_peak(gpu, 0.0);
        note_loaded_peak(gpu, -1.0);
        assert_eq!(reclaimable_pool_gb(gpu), 82.0);
        // Keyed per GPU — a different card is independent.
        assert_eq!(reclaimable_pool_gb(other), 0.0);
    }

    // -----------------------------------------------------------------------------------------
    // The Wan candle video weights-floor gate (sc-12344).
    // -----------------------------------------------------------------------------------------

    /// An RTX 5090 — the biggest consumer NVIDIA card, and the hardware the story names. 32 GB total,
    /// all free (a cold card with nothing loaded). `apply_vram_cap(None, ..)` synthesizes the budget, so
    /// the whole decision is exercisable here with no CUDA driver and no GPU.
    fn rtx_5090() -> Option<VramBudget> {
        apply_vram_cap(None, Some(32.0))
    }

    /// Wan2.2 T2V-A14B candle tier bytes — the SHIPPED hosted sizes, straight from this platform's
    /// `downloads[]` `estimatedSizeBytes` in `builtin.models.jsonc`: q4/q8 from the packed
    /// `SceneWorks/wan2.2-t2v-a14b-candle`, bf16 from the dense `Wan-AI/Wan2.2-T2V-A14B-Diffusers` the
    /// manifest routes the bf16 variant to. Real numbers, so these tests prove the REAL jobs are
    /// admitted/refused rather than that arithmetic is arithmetic.
    const WAN_A14B_CANDLE_Q4_BYTES: u64 = 29_788_704_888; // 27.74 GiB
    const WAN_A14B_CANDLE_Q8_BYTES: u64 = 44_071_949_832; // 41.05 GiB
    const WAN_A14B_CANDLE_BF16_BYTES: u64 = 72_000_000_000; // 67.06 GiB

    /// THE story (sc-12344): Wan A14B at bf16 is a ~67 GiB dual-expert MoE and was admitted on EVERY
    /// consumer card with no pre-flight check. It is now refused before the load + denoise.
    ///
    /// Kills the mutations a compile alone would not:
    ///   * dropping `transformer_2` from the A14B component list ⇒ ~half the bytes ⇒ ADMITS.
    ///   * sizing off the manifest's `footprint.peakMemoryBytes` (24.5 GiB — an MLX measurement that
    ///     assumes only ONE expert is resident) ⇒ ADMITS. The candle engine co-locates both.
    ///   * reusing the Mochi message ⇒ the "Shorten the clip" assert fails.
    #[test]
    fn wan_a14b_bf16_is_refused_on_an_rtx_5090() {
        let message = video_weights_fit_error(
            "wan2_2_t2v_14b",
            WAN_A14B_CANDLE_BF16_BYTES,
            "0",
            rtx_5090(),
        )
        .expect(
            "wan A14B bf16 needs ~67 GiB of weights + 2 GiB headroom = ~69 GB and NO consumer NVIDIA \
             card has that — admitting it burns the load + denoise before a raw CUDA OOM",
        )
        .to_string();
        assert!(
            message.contains("wan2_2_t2v_14b"),
            "names the model: {message}"
        );
        assert!(
            message.contains("69") && message.contains("32"),
            "states what it needs and what the card has: {message}"
        );
        assert!(
            message.contains("smaller quant tier"),
            "names the lever that actually moves weight bytes: {message}"
        );
        // The levers must be WEIGHTS levers. Resolution cannot change weight bytes, and the clip length
        // is Mochi's decode lever — offering either here sends the user to a knob that cannot move the
        // number they were just shown.
        assert!(
            !message.contains("resolution"),
            "resolution has zero effect on a weights floor: {message}"
        );
        assert!(
            !message.contains("Shorten the clip"),
            "Mochi's decode-lever prose leaking into the weights floor: {message}"
        );
        assert!(
            message.contains("VRAM") && !message.contains("unified memory"),
            "must be CUDA-worded, not the MLX lane's Mac prose: {message}"
        );
    }

    /// The other half of the contract: on the SAME 32 GB card the q4 tier is ADMITTED, and bf16 fits the
    /// 96 GB dev box. Without this the reject test above would pass against a gate that blanket-refuses
    /// Wan on every card — so this pair is what proves the gate discriminates by BUDGET and by TIER
    /// rather than wall-rejecting the lane.
    #[test]
    fn wan_a14b_admits_the_tier_that_fits_the_card() {
        // q4 = 27.74 + 2 = 29.74 GB ≤ 32 — the tier a 5090 actually runs.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q4_BYTES, "0", rtx_5090())
                .is_none(),
            "q4 fits a 32 GB card — refusing it wall-rejects hardware that works today"
        );
        // q8 = 43.05 does NOT fit the same card…
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q8_BYTES, "0", rtx_5090())
                .is_some(),
            "q8 does not fit 32 GB"
        );
        // …but does fit a 48 GB card, where bf16 (69.06) still does not — two tiers, one card, two
        // verdicts: the gate reads the TIER's bytes, not the model id.
        let card_48 = apply_vram_cap(None, Some(48.0));
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q8_BYTES, "0", card_48)
                .is_none(),
            "q8 fits 48 GB"
        );
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_BF16_BYTES, "0", card_48)
                .is_some(),
            "bf16 does not fit the SAME 48 GB card"
        );
        // The 96 GB RTX PRO 6000 runs bf16 — the gate must not refuse the box this tier exists for.
        assert!(
            video_weights_fit_error(
                "wan2_2_t2v_14b",
                WAN_A14B_CANDLE_BF16_BYTES,
                "0",
                apply_vram_cap(None, Some(96.0))
            )
            .is_none(),
            "bf16 fits a 96 GB card"
        );
    }

    /// The CUDA reserve is real: a tier whose raw weights fit the card but whose weights + [`HEADROOM_GB`]
    /// do not must be refused. Pinned on a NON-default budget chosen so the two answers differ — q4's
    /// 27.74 GiB fits a 29 GB card outright, and only the reserve rejects it. Deleting the `+ HEADROOM_GB`
    /// from `video_weights_needed_gb` flips this to admit; nothing else in the suite would notice.
    #[test]
    fn video_weights_needed_gb_reserves_the_cuda_headroom() {
        let card_29 = apply_vram_cap(None, Some(29.0));
        assert!(
            WAN_A14B_CANDLE_Q4_BYTES as f64 / BYTES_PER_GIB < 29.0,
            "fixture guard: the raw weights must FIT the card, else this proves nothing"
        );
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_Q4_BYTES, "0", card_29)
                .is_some(),
            "weights alone fit 29 GB but weights + 2 GB of allocator/context reserve do not"
        );
        // And the reserve is the CUDA one, not MLX's OS_RESERVE_GB: they agree at 2.0 today, so pin the
        // arithmetic rather than the constant's name.
        assert_eq!(
            video_weights_needed_gb(WAN_A14B_CANDLE_Q4_BYTES),
            Some(WAN_A14B_CANDLE_Q4_BYTES as f64 / BYTES_PER_GIB + 2.0)
        );
    }

    /// The gate NO-OPS without a budget signal (the story's explicit AC) and without a weight signal — a
    /// worker on a card `nvidia-smi` cannot read, or pointed at a dir it cannot scan, must keep rendering
    /// exactly as it did before this gate existed. A fit gate that blocks on missing evidence is a
    /// regression, not a safety net (sc-12179: never wall-reject a machine that worked).
    #[test]
    fn video_weights_fit_error_no_ops_without_a_budget_or_weight_signal() {
        // No budget ⇒ admit, even for a model no card could hold.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_CANDLE_BF16_BYTES, "0", None)
                .is_none()
        );
        // No weight signal ⇒ admit, even on a tiny card.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", 0, "0", apply_vram_cap(None, Some(1.0)))
                .is_none()
        );
        assert_eq!(video_weights_needed_gb(0), None);
    }

    /// The component lists ARE the gate's correctness: they must name exactly the dirs the loader reads.
    ///
    /// Pins the A14B's second expert in particular — `transformer_2` is co-resident (wan14b.rs:337-343),
    /// and dropping it from the list halves the prediction and silently un-gates the bf16 tier this story
    /// exists to catch.
    #[test]
    fn wan_weight_bytes_sums_exactly_the_components_the_loader_reads() {
        let root = std::env::temp_dir().join(format!(
            "sc12344_wan_components_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        for (relative, len) in [
            ("transformer/model.safetensors", 1_000_u64),
            ("transformer_2/model.safetensors", 2_000),
            ("text_encoder/model.safetensors", 400),
            ("vae/model.safetensors", 30),
            // NOT read by the loader — a decoy that must not be counted. A blind recursive sum of the
            // dir would swallow it; naming the components is what keeps the floor from over-counting.
            ("upsampler/model.safetensors", 9_000_000),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }

        // A14B counts BOTH experts + TE + VAE, and nothing else.
        assert_eq!(
            wan_weight_bytes("wan2_2_t2v_14b", &root),
            1_000 + 2_000 + 400 + 30
        );
        assert_eq!(
            wan_weight_bytes("wan2_2_i2v_14b", &root),
            1_000 + 2_000 + 400 + 30
        );
        // The 5B is single-expert: `transformer_2` must NOT be counted (it would not exist on a real 5B
        // snapshot; counting it here would mean the list, not the disk, decided).
        assert_eq!(wan_weight_bytes("wan2_2_ti2v_5b", &root), 1_000 + 400 + 30);

        std::fs::remove_dir_all(&root).ok();
    }

    /// The EXEMPTIONS and the tier-ROOT fallback both read `0` ⇒ no signal ⇒ admit.
    ///
    /// This is the sc-12179 guard. `ltx`/`svd` are exempt because their on-disk bytes are not their
    /// loaded set (LTX dense is a ~146 GiB repo that loads ONE root file; SVD ships both dtype variants
    /// per component) — a floor built from a dir sum would refuse cards that render fine. And when NO wan
    /// tier subdir resolved, `model_dir` is a ROOT holding `q4/`+`q8/`: there is no top-level
    /// `transformer/`, so this reads 0 and admits rather than summing two tiers at once.
    #[test]
    fn wan_weight_bytes_is_zero_for_exempt_engines_and_a_tier_root() {
        let root = std::env::temp_dir().join(format!(
            "sc12344_wan_exempt_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&root);
        // A tier ROOT: the components live one level DOWN, under each tier.
        for tier in ["q4", "q8"] {
            for component in ["transformer", "transformer_2", "text_encoder", "vae"] {
                let path = root.join(tier).join(component).join("model.safetensors");
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::File::create(&path)
                    .unwrap()
                    .set_len(5_000_000_000)
                    .unwrap();
            }
        }

        // No top-level component dirs ⇒ no signal ⇒ admit. A blind recursive sum would read ~37 GiB
        // here — both tiers at once — and wall-reject a card that runs either one.
        assert_eq!(wan_weight_bytes("wan2_2_t2v_14b", &root), 0);
        assert!(
            video_weights_fit_error(
                "wan2_2_t2v_14b",
                wan_weight_bytes("wan2_2_t2v_14b", &root),
                "0",
                rtx_5090()
            )
            .is_none(),
            "an unrecognized layout must ADMIT, never reject on a number nothing verified"
        );

        // The exempt engines read 0 even pointed at a fully-populated component tree.
        let populated = root.join("q4");
        assert_eq!(wan_weight_bytes("ltx_2_3_distilled", &populated), 0);
        assert_eq!(wan_weight_bytes("svd_xt", &populated), 0);
        // Mochi has its own frame-dependent gate; it must not also ride this one.
        assert_eq!(wan_weight_bytes("mochi_1", &populated), 0);
        // …and the wan engines DO read that same tree, so the zeros above are the exemption, not a
        // broken fixture.
        assert!(wan_weight_bytes("wan2_2_t2v_14b", &populated) > 0);

        std::fs::remove_dir_all(&root).ok();
    }

    // -----------------------------------------------------------------------------------------
    // The MEASURED per-tier peak supersedes the floor (sc-12402).
    // -----------------------------------------------------------------------------------------

    /// The shipped `wan_2_2` candle block — MEASURED on an idle RTX PRO 6000 Blackwell (sc-12402) at
    /// the model's shipped 832x480 / 121-frame / 20-step default. Real numbers, so these tests prove the
    /// REAL jobs are admitted/refused rather than that arithmetic is arithmetic.
    ///
    /// The q8 − q4 delta (88.8 − 86.3 = 2.5 GB) is exactly the q8-vs-q4 DiT delta (5.25 − 2.92 GiB),
    /// which is the coherence check on the campaign: everything else in the peak is tier-independent
    /// (the f32 UMT5 TE + the f32 VAE + the denoise attention).
    fn wan_5b_entry() -> JsonObject {
        obj(json!({
            "candle": {
                "minMemoryGb": 88,
                "vramGbByTier": { "q4": 86.3, "q8": 88.8, "bf16": 94.1 },
                "measured": true
            }
        }))
    }

    /// THE story (sc-12402): a tier the weights FLOOR admits, and that then CUDA-OOMs, is now refused.
    ///
    /// The 5B q4's floor is 16.13 GiB of on-disk weights + 2 = ~18 GB, so an RTX 5090 (32 GB) is
    /// ADMITTED today — and the job dies in the denoise, because the MEASURED peak is 86.3 GB (the
    /// unchunked materialized attention, not the weights: `transformer.rs:46`). The floor under-counts
    /// by 4.4x, which is not a tuning error but a category error — it counts weights and the peak is
    /// not weights.
    ///
    /// Kills the mutations a compile alone would not:
    ///   * dropping the `predicted_peak_gb` branch (⇒ back to the floor) ⇒ the 5090 ADMITS;
    ///   * keying the manifest lookup off the request instead of the RESOLVED tier ⇒ wrong row;
    ///   * `max`-ing the measured peak with the floor ⇒ still passes here, but fails the dense test below.
    #[test]
    fn measured_peak_refuses_a_5b_job_the_weights_floor_would_admit() {
        let entry = wan_5b_entry();
        // The on-disk floor for the shipped q4 tier: 16.13 GiB.
        const WAN_5B_Q4_DISK_BYTES: u64 = 17_315_750_512;

        // Floor alone: ~18 GB needed vs 32 GB free ⇒ admits (the shipped behavior, and the bug).
        assert!(
            video_weights_fit_error("wan_2_2", WAN_5B_Q4_DISK_BYTES, "0", rtx_5090()).is_none(),
            "precondition: the weights floor admits this job on a 5090 — that IS the sc-12402 bug"
        );

        // Measured peak: 86.3 + 2 headroom = 88.3 GB vs 32 GB ⇒ REFUSED before the load + denoise.
        let message = wan_video_fit_error(
            "wan_2_2",
            &entry,
            "q4",
            WAN_5B_Q4_DISK_BYTES,
            "0",
            rtx_5090(),
        )
        .expect("the measured 86.3 GB peak cannot fit a 32 GB card — refuse before the OOM")
        .to_string();
        assert!(message.contains("wan_2_2"), "names the model: {message}");
        assert!(message.contains("q4"), "names the sized tier: {message}");
        assert!(
            message.contains("88") && message.contains("32"),
            "states what it needs and what the card has: {message}"
        );
        // The measured message is about the RENDER, not the weights — the floor's wording would be a
        // lie here (weights are 16 GiB of an 86 GB peak).
        assert!(
            !message.contains("just to hold its weights"),
            "must not reuse the weights-floor wording for a measured peak: {message}"
        );
        // Resolution is NOT offered as a lever: the measured peak is a per-tier constant at the model's
        // default geometry, so it cannot move the number the user was just shown.
        assert!(
            !message.contains("resolution"),
            "the resolution-blind gate must not send the user to a knob it cannot honor: {message}"
        );
    }

    /// The other half of the story's acceptance: a tier that FITS is still ADMITTED. A gate that
    /// refuses everything would pass the test above and be worthless.
    #[test]
    fn measured_peak_admits_a_tier_that_fits_the_card() {
        let entry = wan_5b_entry();
        const WAN_5B_Q4_DISK_BYTES: u64 = 17_315_750_512;
        // The dev box: 96 GB. q4's measured 86.3 + 2 = 88.3 ≤ 95.6 ⇒ ADMIT (and it really does render —
        // this exact configuration is the sc-12402 measurement).
        let card96 = apply_vram_cap(None, Some(95.6));
        assert!(
            wan_video_fit_error("wan_2_2", &entry, "q4", WAN_5B_Q4_DISK_BYTES, "0", card96)
                .is_none(),
            "the measured q4 job renders on this card — the gate must not wall-reject it"
        );
        // …and the heavier bf16 row on the SAME card does NOT fit: its measured 94.1 + 2 headroom =
        // 96.1 just overflows 95.6. Same card, same model, different tier ⇒ opposite verdict: the gate
        // reads the TIER, not the model. (This is a genuinely narrow 0.5 GB margin, and it is the real
        // measured one — the 5B's tier ladder spans only ~8 GB because everything but the DiT is
        // tier-independent, so the tiers really do land this close together.)
        assert!(
            wan_video_fit_error("wan_2_2", &entry, "bf16", WAN_5B_Q4_DISK_BYTES, "0", card96)
                .is_some(),
            "bf16's measured 94.1 GB peak + 2 headroom overflows a 95.6 GB card — refuse"
        );
    }

    /// sc-12402 regression pin: the measured peak REPLACES the floor, it does not compose with it.
    ///
    /// The dense `bf16` tier is the case that makes `max(measured, floor)` wrong. `Wan-AI/*-Diffusers`
    /// ships its experts in **fp32** (~117.5 GiB summed for A14B T2V) and `candle-gen-wan` loads them as
    /// bf16 (`wan14b.rs:49-52`), so the on-disk floor OVER-counts by ~44 GiB and would wall-reject a
    /// 96 GB card that renders the job at ~75 GB — the sc-12179 class `wan_weight_components` claims
    /// cannot arise on this lane. A measurement is not a better proxy for the peak; it IS the peak.
    #[test]
    fn a_measurement_supersedes_an_overcounting_dense_floor() {
        // A14B T2V dense: 117.51 GiB of fp32 on disk, ~75 GB resident once cast to bf16.
        const WAN_A14B_DENSE_DISK_BYTES: u64 = 126_170_000_000;
        let entry = obj(json!({
            "candle": { "vramGbByTier": { "bf16": 75.3 }, "measured": false }
        }));
        let card96 = apply_vram_cap(None, Some(95.6));

        // The floor alone WALL-REJECTS this card (117.5 + 2 = ~120 > 95.6) — the bug.
        assert!(
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_DENSE_DISK_BYTES, "0", card96)
                .is_some(),
            "precondition: the on-disk floor over-counts the fp32 dense tier and wall-rejects a 96 GB card"
        );
        // The measured/derived peak admits it: 75.3 + 2 = 77.3 ≤ 95.6.
        assert!(
            wan_video_fit_error(
                "wan2_2_t2v_14b",
                &entry,
                "bf16",
                WAN_A14B_DENSE_DISK_BYTES,
                "0",
                card96
            )
            .is_none(),
            "a `max(measured, floor)` would keep the floor's 120 GB over-count and wall-reject here"
        );
    }

    /// An unmeasured model keeps the sc-12344 floor EXACTLY — the fallback is not a silent un-gating.
    #[test]
    fn an_unmeasured_model_falls_back_to_the_weights_floor() {
        let no_block = obj(json!({}));
        const WAN_A14B_BF16_BYTES: u64 = 72_000_000_000;

        // No `candle` block ⇒ the floor decides, byte-identical to `video_weights_fit_error`.
        let via_wan = wan_video_fit_error(
            "wan2_2_t2v_14b",
            &no_block,
            "bf16",
            WAN_A14B_BF16_BYTES,
            "0",
            rtx_5090(),
        );
        let via_floor =
            video_weights_fit_error("wan2_2_t2v_14b", WAN_A14B_BF16_BYTES, "0", rtx_5090());
        assert_eq!(
            via_wan.map(|e| e.to_string()),
            via_floor.map(|e| e.to_string()),
            "an unmeasured model must behave exactly as it did before sc-12402"
        );

        // An exempt engine (0 weights) + no block ⇒ admit, never block without evidence.
        assert!(
            wan_video_fit_error("ltx_2_3_distilled", &no_block, "bf16", 0, "0", rtx_5090())
                .is_none()
        );
        // No live budget ⇒ admit, even with a measured row that would overflow.
        assert!(
            wan_video_fit_error("wan_2_2", &wan_5b_entry(), "bf16", 0, "0", None).is_none(),
            "no budget signal ⇒ never block"
        );
    }

    /// Live real-hardware validation (sc-10766): exercises the REAL `nvidia-smi` VRAM reading on GPU 0
    /// plus the full cap → predict → decide chain — the one piece the pure tests above can't cover.
    /// Ignored by default (needs a CUDA GPU); run on the box with
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored live_cuda_budget`.
    #[tokio::test]
    #[ignore]
    async fn live_cuda_budget_drives_a_real_fit_decision() {
        let real = crate::gpu::nvidia_vram_budget_gb("0")
            .await
            .expect("GPU 0 should report a live VRAM budget on a CUDA box");
        assert!(
            real.total_gb > 0.0
                && real.free_gb >= 0.0
                && real.free_gb <= real.total_gb + f64::EPSILON,
            "sane live budget: {real:?}"
        );
        eprintln!("live CUDA budget GPU0: {real:?}");
        // Emulate a 10 GB card: a ~47 GB q4 model must be rejected before load…
        let capped = apply_vram_cap(Some(real), Some(10.0)).expect("capped budget");
        let big = obj(json!({ "candle": { "vramGbByTier": { "q4": 47.2 } } }));
        assert!(matches!(
            fit_decision(predicted_peak_gb(&big, "q4"), Some(capped)),
            FitDecision::TooBig { .. }
        ));
        // …while a ~4 GB model fits that same emulated card.
        let small = obj(json!({ "candle": { "vramGbByTier": { "q4": 4.0 } } }));
        assert_eq!(
            fit_decision(predicted_peak_gb(&small, "q4"), Some(capped)),
            FitDecision::Fits
        );
    }

    /// GPU repro of the sc-11023 warm-swap false reject: occupy REAL VRAM on GPU 0 to mimic a
    /// resident model (nvidia-smi `free` drops the way a cached model makes it), then show that the
    /// SAME live budget REJECTS the incoming bf16 tier on raw `free` (the bug) but FITS once the
    /// reclaimable high-water is folded in (the fix). Run on an idle GPU 0:
    /// `cargo test -p sceneworks-worker --lib --features backend-candle -- --ignored --nocapture \
    ///  reclaimable_high_water_flips_a_real_warm_reject`
    #[tokio::test]
    #[ignore]
    async fn reclaimable_high_water_flips_a_real_warm_reject() {
        use runtime_cuda::media::candle_core::{DType, Device, Tensor};

        let gpu = "0";
        // Qwen-Image bf16 published tier peaks (builtin.models.jsonc `candle` block, sc-10969).
        let manifest = obj(json!({
            "candle": { "vramGbByTier": { "bf16": 82.5 }, "sequentialPeakGb": { "bf16": 71.7 } }
        }));
        let needed = predicted_peak_gb(&manifest, "bf16").expect("bf16 resident peak");
        let seq_needed = predicted_sequential_peak_gb(&manifest, "bf16").expect("bf16 seq peak");

        let cold = crate::gpu::nvidia_vram_budget_gb(gpu)
            .await
            .expect("GPU 0 live budget — run on a CUDA box");
        eprintln!("cold budget: {cold:?} (needed resident={needed}, sequential={seq_needed})");
        assert!(
            cold.free_gb > seq_needed,
            "GPU 0 must start with enough free VRAM to host bf16 for the repro: {cold:?}"
        );

        // Mimic run 1 admitting a bf16 load: the live gate records the model's predicted peak as the
        // reclaimable high-water (note_loaded_peak) AND the model actually occupies VRAM. Occupy enough
        // (8 GB f32 chunks) to push `free` below the sequential threshold, so raw-free rejects even
        // sequentially — capped defensively so we never exhaust the card.
        note_loaded_peak(gpu, needed);
        let device = Device::new_cuda(0).expect("cuda:0");
        let chunk_elems = 8usize * 1024 * 1024 * 1024 / 4; // 8 GB of f32
        let mut hogs: Vec<Tensor> = Vec::new();
        loop {
            let now = crate::gpu::nvidia_vram_budget_gb(gpu)
                .await
                .expect("live budget");
            if now.free_gb < seq_needed - 4.0 || hogs.len() >= 8 {
                break;
            }
            hogs.push(Tensor::zeros(chunk_elems, DType::F32, &device).expect("VRAM hog chunk"));
        }
        let warm = crate::gpu::nvidia_vram_budget_gb(gpu)
            .await
            .expect("GPU 0 live budget (warm)");
        eprintln!("warm budget after {} GB hog: {warm:?}", hogs.len() * 8);
        assert!(
            warm.free_gb < seq_needed,
            "the hog must push free below the sequential need to arm the reject: {warm:?}"
        );

        // BEFORE the fix (raw free): sequential residency is selected, then the second-stage overflow
        // check rejects — the exact "even with sequential residency" 2nd-run failure.
        let raw_decision = resolve_offload(fit_decision(Some(needed), Some(warm)), true);
        assert!(
            matches!(raw_decision, FitDecision::Offload { .. }),
            "raw free must not fit resident: {raw_decision:?}"
        );
        assert!(
            sequential_overflow_gb(Some(seq_needed), Some(warm)).is_some(),
            "raw-free gate must reject even sequentially — this is the sc-11023 bug"
        );

        // AFTER the fix: fold the reclaimable high-water (the resident model we are about to evict)
        // back into the budget → the incoming bf16 fits resident, no false reject.
        let reclaimable = reclaimable_pool_gb(gpu);
        let fixed = with_reclaimable(warm, reclaimable);
        eprintln!("reclaimable={reclaimable} → augmented budget: {fixed:?}");
        assert_eq!(
            fit_decision(Some(needed), Some(fixed)),
            FitDecision::Fits,
            "with the resident model counted reclaimable, the warm re-gate must FIT"
        );
        // Keep the hog alive to the end so the warm reading isn't reclaimed early.
        drop(hogs);
    }
}
