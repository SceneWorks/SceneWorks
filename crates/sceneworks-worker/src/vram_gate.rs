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

/// The gate outcome. `Unknown` = no signal (unmeasured model or non-NVIDIA host) ⇒ never block, exactly
/// like the flux2 edit guard's `available_gb == None` short-circuit.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FitDecision {
    Unknown,
    Fits,
    /// The predicted resident peak won't fit, but the model's provider supports sequential component
    /// residency (the candle FLUX lane, sc-10769) — run with `OffloadPolicy::Sequential` instead of
    /// rejecting. Sequential peak is always ≤ resident, so this is the best option before a reject.
    /// When the tier's sequential peak is MEASURED (`candle.sequentialPeakGb`, sc-10856) the caller runs
    /// a second [`sequential_overflow_gb`] check and rejects if even sequential won't fit; absent that
    /// number it's best-effort (the reactive Metal/CUDA-OOM containment is the backstop).
    Offload {
        needed_gb: f64,
        available_gb: f64,
    },
    TooBig {
        needed_gb: f64,
        available_gb: f64,
    },
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

/// Fold sequential-residency capability into a fit decision (epic 10765 Phase 1b, sc-10821): a
/// [`FitDecision::TooBig`] on a provider that supports sequential offload (the candle FLUX lane) becomes
/// [`FitDecision::Offload`] — load→use→drop each component so peak VRAM is the largest single working
/// set (≤ resident) rather than reject. Every other outcome (Fits / Unknown / TooBig on a non-capable
/// provider) is unchanged.
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
