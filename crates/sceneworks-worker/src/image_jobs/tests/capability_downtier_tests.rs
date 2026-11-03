#[cfg(not(target_os = "macos"))]
use super::super::quantization::gate_tier_key;
use super::super::quantization::tier_key_from_resolved_dir;
use super::super::tier_quality_rank;
#[cfg(not(target_os = "macos"))]
use super::super::INT8_CONVROT_TIER;
use super::super::NVFP4_TIER;
use super::choose_downtier;
use super::DowntierPick;
use super::TierFit;
/// sc-12425 — **a resolved ConvRot load must be gated by its OWN tier, never aliased to q8.**
///
/// The defect this kills: `generate_candle_stream` handed ConvRot to the BITS-derived
/// `vram_gate::requested_tier_key`. A ConvRot request carries no `mlxQuantize` (the picker sends
/// `advanced.convRot: true` — see [`wants_krea_convrot`]), so it hit that function's `None => "q8"`
/// arm, and a **measured 42.9 GB** render (sc-12381) was sized against q8's 35.9 + 2.0 = 37.9 GB row
/// — admitting loads that OOM.
///
/// The second assertion pins the ALIASING itself, so this test says why the first one matters
/// instead of just asserting a constant: q8's row is what was actually being read, which is why
/// "just correct the manifest row" would have fixed nothing.
///
/// Lives in `capability_downtier_tests` (not the `#[cfg(target_os = "macos")]`
/// `quant_tier_reconcile_tests`, where it would compile out on the candle lane), but carries its OWN
/// candle-only gate: it calls [`gate_tier_key`] + `crate::vram_gate`, both `not(macos)` — while this
/// module is `any(macos, candle)`, so without the attribute below it fails to compile on the MLX
/// build (E0433, no `vram_gate`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn gate_tier_key_names_convrot_by_identity_never_q8() {
    // The real shape: the ConvRot base surface IS the bf16 dir, and the request carries no bits.
    let convrot_base = std::path::Path::new("/models/krea-2-turbo-mlx/bf16");
    let advanced = serde_json::json!({ "convRot": true })
        .as_object()
        .expect("object")
        .clone();
    let entry = serde_json::Map::new();

    assert_eq!(
        gate_tier_key(true, convrot_base, &advanced, &entry, false),
        INT8_CONVROT_TIER,
        "a resolved ConvRot load must be named by tier identity, not sized against another tier"
    );

    // THE ALIASING, pinned. If this stops being q8, the aliasing changed shape and sc-12425 needs
    // re-reading — do not just bump it to whatever it now returns.
    assert_eq!(
        crate::vram_gate::requested_tier_key(&advanced, &entry, false),
        "q8",
        "a ConvRot request carries no mlxQuantize, so the bits-derived key aliases it to q8 — the \
             under-prediction sc-12425 fixes"
    );

    // The non-ConvRot path is untouched: the on-disk tier still wins (sc-12090).
    assert_eq!(
        gate_tier_key(false, convrot_base, &advanced, &entry, false),
        "bf16"
    );
}

/// sc-12090 / sc-12829: the basename → tier-NAME reader the candle VRAM gate keys off —
/// `generate_candle_stream` → `gate_tier_key` fallback and the sc-12090 disk-probe path
/// ([`resolve_tier_dir`] / `installed_tier_keys`) both resolve the on-disk tier through
/// [`tier_key_from_resolved_dir`]. That fn is `any(macos, candle)`, so its coverage lives HERE (this
/// cross-lane module), NOT in the `#[cfg(target_os = "macos")]` `quant_tier_reconcile_tests` where it
/// would compile out on the candle lane — leaving the exact basename→tier mapping the gate depends on
/// unexercised on `candle-worker`. A regression (e.g. a new Boogu `<variant>-<tier>` shape) now goes
/// RED on both lanes, not just the Mac runner. (`gate_tier_key`/`installed_tier_keys` are candle-only,
/// so they're named as plain code — only the cross-lane fns exist on both of this module's lanes.)
#[test]
fn tier_key_from_resolved_dir_reads_the_on_disk_tier() {
    let root = std::path::Path::new("/models/krea-2-turbo-mlx");
    assert_eq!(tier_key_from_resolved_dir(&root.join("q4")), Some("q4"));
    assert_eq!(tier_key_from_resolved_dir(&root.join("q8")), Some("q8"));
    assert_eq!(tier_key_from_resolved_dir(&root.join("bf16")), Some("bf16"));
    // Boogu `<variant>-<tier>` suffix + the bare `<variant>` packed Q8 default.
    assert_eq!(
        tier_key_from_resolved_dir(&root.join("base-q4")),
        Some("q4")
    );
    assert_eq!(
        tier_key_from_resolved_dir(&root.join("turbo-bf16")),
        Some("bf16")
    );
    assert_eq!(tier_key_from_resolved_dir(&root.join("edit")), Some("q8"));
    // Unrecognized basename (repo root / modelPath) → None; the gate keeps its manifest key.
    assert_eq!(tier_key_from_resolved_dir(root), None);
}

fn too_big(needed: f64, avail: f64) -> TierFit {
    TierFit::TooBig {
        needed_gb: needed,
        available_gb: avail,
    }
}

#[test]
fn keeps_the_default_when_it_fits() {
    // Q8 default fits → Keep, even though a smaller q4 is also installed and would fit.
    let candidates = [("q8", TierFit::Fits), ("q4", TierFit::Fits)];
    assert_eq!(choose_downtier("q8", &candidates), DowntierPick::Keep);
}

/// The ladder scores a tier only until one fits (sc-10733 cost, not semantics).
///
/// On the MLX lane `mlx_tier_fit` reaches `ProviderRegistry::footprint`, which seals that tier's
/// text encoder with a full SHA-256. The gate used to collect the WHOLE candidate list before
/// choosing, so a machine with room to spare hashed bf16, q8 and q4 in turn to answer a question
/// the first one had already settled — 15s + 10s + 6s on one measured krea_2_turbo job, against
/// a budget with 61 GB unused.
///
/// Stated as a call count, like `the_load_boundary_hashes_the_closure_exactly_once`: on the pure
/// chooser a duration would measure nothing, and the count is the actual claim.
#[test]
fn the_ladder_scores_no_tier_below_the_first_that_fits() {
    use std::cell::RefCell;
    let tiers = ["bf16", "q8", "q4"];
    let scored = RefCell::new(Vec::new());
    let score = |fits: fn(&str) -> TierFit| {
        scored.borrow_mut().clear();
        super::choose_downtier_lazy::<_, std::convert::Infallible>("bf16", &tiers, |tier| {
            scored.borrow_mut().push(tier);
            Ok(Some(fits(tier)))
        })
        .unwrap()
    };

    // The ordinary outcome: the default fits, so nothing below it is even looked at.
    assert_eq!(score(|_| TierFit::Fits), DowntierPick::Keep);
    assert_eq!(*scored.borrow(), vec!["bf16"]);

    // A genuine downtier stops at the tier it lands on, not at the end of the ladder.
    assert_eq!(
        score(|tier| if tier == "bf16" {
            too_big(72.0, 40.0)
        } else {
            TierFit::Fits
        }),
        DowntierPick::Downtier("q8")
    );
    assert_eq!(*scored.borrow(), vec!["bf16", "q8"]);

    // Only a rejection walks the whole ladder — and it must, to name the smallest tier tried.
    assert_eq!(
        score(|_| too_big(72.0, 8.0)),
        DowntierPick::Reject {
            tier: "q4",
            needed_gb: 72.0,
            available_gb: 8.0,
        }
    );
    assert_eq!(*scored.borrow(), vec!["bf16", "q8", "q4"]);
}

/// A candidate whose directory does not resolve is SKIPPED, exactly as the eager form's
/// `filter_map` dropped it before scoring — it must not become the named rejection.
#[test]
fn an_unresolvable_tier_is_skipped_rather_than_scored_as_too_big() {
    let tiers = ["bf16", "q8", "q4"];
    let pick = super::choose_downtier_lazy::<_, std::convert::Infallible>("bf16", &tiers, |tier| {
        match tier {
            "bf16" => Ok(Some(too_big(72.0, 40.0))),
            // Not installed: no directory, so nothing to score.
            "q8" => Ok(None),
            _ => Ok(Some(too_big(20.0, 8.0))),
        }
    })
    .unwrap();
    assert_eq!(
        pick,
        DowntierPick::Reject {
            tier: "q4",
            needed_gb: 20.0,
            available_gb: 8.0,
        }
    );
}

#[test]
fn downtiers_to_the_highest_installed_tier_that_fits() {
    // Q8 default won't fit, q4 does → downtier to q4 (acceptance #2).
    let candidates = [("q8", too_big(33.0, 30.0)), ("q4", TierFit::Fits)];
    assert_eq!(
        choose_downtier("q8", &candidates),
        DowntierPick::Downtier("q4")
    );
    // bf16 default won't fit, q8 is the highest that does (q4 also fits but is lower fidelity).
    let three = [
        ("bf16", too_big(72.0, 40.0)),
        ("q8", TierFit::Fits),
        ("q4", TierFit::Fits),
    ];
    assert_eq!(
        choose_downtier("bf16", &three),
        DowntierPick::Downtier("q8")
    );
}

#[test]
fn rejects_naming_the_smallest_evaluated_tier_when_nothing_fits() {
    // Neither q8 nor q4 fits → reject, naming q4 (the smallest / least-demanding tried) + its need.
    let candidates = [("q8", too_big(33.0, 10.0)), ("q4", too_big(28.0, 10.0))];
    assert_eq!(
        choose_downtier("q8", &candidates),
        DowntierPick::Reject {
            tier: "q4",
            needed_gb: 28.0,
            available_gb: 10.0,
        }
    );
}

#[test]
fn floor_wins_over_downtier_via_the_candidate_list() {
    // A floor-q8 model: the caller filters q4 OUT of the candidate list (rank < floor), so even when
    // only q8 is offered and it won't fit, we REJECT rather than silently rendering q4 (acceptance #5).
    let floored = [("q8", too_big(33.0, 20.0))];
    assert_eq!(
        choose_downtier("q8", &floored),
        DowntierPick::Reject {
            tier: "q8",
            needed_gb: 33.0,
            available_gb: 20.0,
        }
    );
}

#[test]
fn empty_candidates_keep_the_default() {
    // No installed candidate in range (defensive — the default itself is normally present) → Keep,
    // deferring to the plain gate.
    assert_eq!(choose_downtier("q8", &[]), DowntierPick::Keep);
}

/// sc-11042 (epic 11037 SC#5) × sc-10733: the capability clamp NEVER downtiers a selected NVFP4
/// tier. Downtiering it to q4/q8 would silently swap the numerics of an explicitly-picked tier —
/// the exact creative-choice violation SC#5 forbids.
///
/// This is load-bearing BECAUSE nvfp4 does not escape the clamp via `mlxQuantizeExplicit`: the web
/// emits that flag only inside the `tierQuantize(quantTier) !== null` bits branch, and nvfp4 has no
/// honest `mlxQuantize` integer, so an nvfp4 job reaches the clamp with `explicit_pick == false`.
/// What saves it is that nvfp4 is UNRANKABLE on purpose (`tier_quality_rank` ⇒ 0): it is a distinct
/// numeric regime, not a rung on the bf16/q8/q4 ladder, so no tier is ever in `[floor, nvfp4]` and
/// the chooser keeps it. These two facts are a silent pair — pin them together, since making nvfp4
/// rankable would quietly arm the downtier.
#[test]
fn nvfp4_tier_is_never_downtiered_by_the_capability_clamp() {
    // Unrankable by construction — the fact the whole guard rests on.
    assert_eq!(tier_quality_rank(NVFP4_TIER), 0);
    assert!(tier_quality_rank(NVFP4_TIER) < tier_quality_rank("q4"));

    // The `downtier_candidate_tiers` range math (installed-filtering aside) admits NOTHING when the
    // default is nvfp4: `rank <= 0 && rank >= floor_rank(>= 1)` is unsatisfiable for every tier.
    let in_range = |tier: &str, default: &str, floor: Option<&str>| {
        let default_rank = tier_quality_rank(default);
        let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
        let rank = tier_quality_rank(tier);
        rank <= default_rank && rank >= floor_rank
    };
    for floor in [None, Some("q4"), Some("q8"), Some("bf16")] {
        for tier in ["bf16", "q8", "q4", NVFP4_TIER] {
            assert!(
                !in_range(tier, NVFP4_TIER, floor),
                "nvfp4 must admit NO downtier candidate (tier={tier} floor={floor:?})"
            );
        }
    }

    // …so the chooser is handed an empty candidate list and KEEPS nvfp4, deferring to the plain gate.
    assert_eq!(choose_downtier(NVFP4_TIER, &[]), DowntierPick::Keep);
}

#[test]
fn downtier_candidate_range_is_floor_to_default_descending() {
    // The pure range math behind `downtier_candidate_tiers` (installed-filtering aside): candidates
    // run from the default DOWN to the floor, highest-fidelity first, never above the default.
    let in_range = |tier: &str, default: &str, floor: Option<&str>| {
        let default_rank = tier_quality_rank(default);
        let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
        let rank = tier_quality_rank(tier);
        rank <= default_rank && rank >= floor_rank
    };
    // Default q8, no floor (→ q4): q8 and q4 in range; bf16 (above default) excluded.
    assert!(in_range("q8", "q8", None));
    assert!(in_range("q4", "q8", None));
    assert!(!in_range("bf16", "q8", None));
    // Floor q8: q4 excluded (below floor), q8 in range.
    assert!(!in_range("q4", "q8", Some("q8")));
    assert!(in_range("q8", "q8", Some("q8")));
}
