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
    TooBig { needed_gb: f64, available_gb: f64 },
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

/// The tier key (`bf16`/`q8`/`q4`) the request selected — derived the SAME way the tier-subdir
/// resolvers pick their folder (`advanced.mlxQuantize` → manifest `mlx.quantize` → `q8` default;
/// `<= 0` ⇒ `bf16`; `<= 4` ⇒ `q4`; else `q8`). Deliberately NOT derived from the resolved `Quant`,
/// which is `None` on packed-tier candle families whose tier is the resolved subdir, not a load-time
/// quantize (see `resolve_quant`'s candle-lane note).
pub(crate) fn requested_tier_key(
    advanced: &JsonObject,
    manifest_entry: &JsonObject,
) -> &'static str {
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
pub(crate) fn predicted_peak_gb(manifest_entry: &JsonObject, tier_key: &str) -> Option<f64> {
    let candle = manifest_entry.get("candle")?;
    if let Some(gb) = candle
        .get("vramGbByTier")
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
    {
        return Some(gb + HEADROOM_GB);
    }
    candle.get("minMemoryGb").and_then(json_f64)
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
        assert_eq!(requested_tier_key(&empty, &empty), "q8");
        // advanced.mlxQuantize wins, number or numeric string.
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 0})), &empty),
            "bf16"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 4})), &empty),
            "q4"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": 8})), &empty),
            "q8"
        );
        assert_eq!(
            requested_tier_key(&obj(json!({"mlxQuantize": "4"})), &empty),
            "q4"
        );
        // Falls back to the manifest mlx.quantize when advanced is silent.
        assert_eq!(
            requested_tier_key(&empty, &obj(json!({"mlx": {"quantize": 4}}))),
            "q4"
        );
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
}
