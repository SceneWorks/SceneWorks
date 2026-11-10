/// The live digest for this lane, so a fixture is current unless a test deliberately says
/// otherwise. Read, never frozen — a literal here would go stale on the next pin bump and the
/// tests would silently stop exercising the admitted path.
fn live_test_closure_digest() -> String {
    sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo_control")
        .unwrap_or_default()
}

use super::*;
use crate::vram_gate::VramBudget;
use serde_json::json;

#[test]
fn candle_krea_pose_identity_matches_the_provider_evidence_probe_and_live_call_site() {
    assert_eq!(KREA_CONTROL_REFERENCE_COUNT, 1);
    assert_eq!(
        crate::memory_strategy::memory_mode_from_mode_key(KREA_CONTROL_MODE),
        gen_core::MemoryMode::ImageToImage
    );

    let candle_lane = include_str!("../image_jobs/krea_control_candle.rs");
    assert!(
        candle_lane
            .contains("reference_count: crate::krea_control_fit::KREA_CONTROL_REFERENCE_COUNT"),
        "the live Candle route must use the provider-matched one-reference geometry"
    );
    assert!(
        !candle_lane.contains("reference_count: 0"),
        "the live Candle pose route must not reintroduce the zero-reference declaration"
    );
}

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
/// will ship. Since sc-18097 current evidence is itself what makes a hard reject available: the
/// unmeasured deeper rungs are covered by estimate floors, so the reject fires once nothing fits
/// at its graded peak rather than waiting for measurements through the deepest implemented rung.
/// Current evidence also makes [`incurred_peak_gb`] recordable.
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
/// peak, against CURRENT evidence. Since sc-18097 a current-evidence fixture can hard-reject
/// whether or not its deeper rungs are measured — the unmeasured ones are graded at estimate
/// floors. Best-effort now belongs to superseded evidence, an unverified runtime, adapters, or a
/// request above the measured envelope; dedicated tests below cover each floor.
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

/// sc-22738 — "the App Runtime should ALWAYS continue behaving as if the measurement were
/// valid". `candle.control.inferenceClosureDigest` is now PROVENANCE: `fit_ladder_for_tier`
/// still takes it, copies it verbatim into every emitted evidence receipt's
/// `inference_revision`, and compares it against nothing. A control ladder whose provider
/// closure moved is therefore graded at exactly the same peaks as one whose closure matches
/// the live ledger — no demotion, no widening, no `BestEffort` fallback.
///
/// Two budgets, because a single roomy budget would pass even if a widening were reintroduced.
/// The tight budget (10.1 GiB free ⇒ 8.1 GiB effective) sits between the RAW resident evidence
/// peak (8.0 GiB) and any widened one (~8.16 GiB at the 2% candle recapture spread), so it is
/// the exact cell where a digest-keyed widening would still discriminate: it would push the
/// moved-digest arm off the resident row down to the measured bounded-decode row.
///
/// MUTATION that reds this: reintroduce any comparison of `measured_closure_digest` against
/// `packaged_closure_digest("candle", "krea_2_turbo_control")` in `fit_ladder_for_tier` — a
/// widening on mismatch flips the tight arm, a demotion flips both arms.
#[test]
fn a_control_closure_digest_is_provenance_and_never_moves_the_fit() {
    const MOVED_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";
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

    // Roomy budget. Control point first: an unmoved closure must actually admit, or the
    // equality below would be satisfied by two identical non-fits.
    let live_roomy = fit(20.0, &live_test_closure_digest());
    assert!(
        matches!(live_roomy, KreaControlFit::Fits { .. }),
        "control point: an unmoved closure must report a fit, or the equalities below prove \
             nothing: {live_roomy:?}"
    );
    assert_eq!(
        fit(20.0, MOVED_DIGEST),
        live_roomy,
        "sc-22738: a moved control closure is a re-capture signal for the probe tooling only \
             — the roomy-budget fit must be identical"
    );

    // Tight budget — the discriminating cell. The live-digest arm is graded at the RAW
    // resident evidence peak and stays resident; since nothing compares the digest, the moved
    // arm must land on that same verdict rather than walking down the ladder.
    let live_tight = fit(10.1, &live_test_closure_digest());
    assert_eq!(
        live_tight,
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: false,
        },
        "measured evidence at the raw peak keeps the resident fit"
    );
    assert_eq!(
        fit(10.1, MOVED_DIGEST),
        live_tight,
        "sc-22738: at the budget where a digest-keyed widening WOULD discriminate, the moved \
             closure must still be priced at the raw measured peak"
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
fn additive_adapter_bytes_shift_every_control_rung_without_claiming_a_measured_peak() {
    let manifest = krea_manifest_measured();
    let tier = "q4";
    let one_gib = crate::fit_gate::BYTES_PER_GIB as u64;
    let resident = predicted_control_peak_gb(&manifest, tier).unwrap();

    assert_eq!(
        fit_ladder_for_entry_with_adapter_bytes(&manifest, tier, Some(budget(resident + 0.5)), 0,),
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

    let fit = fit_ladder_for_entry_with_adapter_bytes(&manifest, tier, Some(budget(96.0)), one_gib);
    assert!(matches!(
        fit,
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            estimate_scoped: true,
            ..
        }
    ));
    assert_eq!(
        incurred_peak_gb_with_adapter_bytes(&fit, &manifest, tier, one_gib),
        None,
        "an adapter estimate must not claim a measured reclaimable peak"
    );
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
///
/// sc-18097 sharpens the contrast rather than blunting it: the current-evidence side now rejects
/// from graded floors even where the deeper rungs are unmeasured, while the superseded side is
/// exactly where the estimate floors are withheld, so its never-reject contract is untouched.
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

/// sc-18253: the staged `sequentialPeakGbByTier` row prices the staged WORKING SET, so it is
/// only a sound floor for a composition that actually engages `StagedResidency`. The gen-core
/// engagement mechanism permits a contract to implement a deep rung while excluding staging
/// (the default composition when no staged prerequisite edge is declared); such a request runs
/// whole-model resident and its floor must clamp to the resident row — the mirror of the MLX
/// floor's `engaged.contains(&StagedResidency)` max-vs-sum split.
///
/// The request runs at 512² — inside the measured envelope, so the floors are synthesized,
/// while every measured 1024² candidate is structurally out-of-envelope. That isolates the
/// FLOOR values in the outcome: admission below the resident row is possible only through a
/// floor. Both mutation directions are pinned: deleting the composition check re-admits the
/// staging-free arm on the staged row (red), and inverting it (always the resident floor)
/// strands the registered control arm's staged rung at the resident row (red).
#[test]
fn a_staging_free_deep_rung_floor_clamps_to_the_resident_row() {
    let request_geometry = MemoryGeometry {
        width: 512,
        height: 512,
        batch: 1,
        frames: 1,
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
    };
    let peak = 40.0;
    let sequential = 30.0;
    let staged_floor = sequential - HEADROOM_GB;
    let resident_floor = peak - HEADROOM_GB;
    let margin = crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD;
    // An effective budget between the WIDENED staged floor and the resident floor: a rung
    // priced on the staged row admits here; one clamped to the resident row cannot. Recomputed
    // from the policy margin, never a frozen literal.
    let free = staged_floor * (1.0 + margin) + HEADROOM_GB + 0.5;
    assert!(
        free - HEADROOM_GB < resident_floor,
        "the budget must stay below the resident floor to discriminate"
    );
    let run = |contract: &MemoryProviderContract| {
        fit_ladder_for_tier(
            Some(contract),
            "q4",
            request_geometry,
            Some(peak),
            Some(sequential),
            Some(budget(free)),
            None,
            None,
            true,
            true,
            0.0,
            &live_test_closure_digest(),
        )
    };

    // Control: the registered q4 contract binds its deep rungs to staging, so the staged
    // working-set floor stands and the ladder admits sequential staging at 512². Clamping
    // every floor to the resident row regardless of composition would flip this red.
    let registered = registered_contract_for_tier("q4").expect("registered control contract");
    assert_eq!(
        run(&registered),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: true,
        },
        "a staging-engaged composition keeps the staged working-set floor"
    );

    // The finding's contract shape: decode/attention implemented, staging not implemented and
    // no staged prerequisite edges — every deep rung's engaged composition excludes
    // `StagedResidency`, so the request runs whole-model resident. Before sc-18253 those
    // rungs took the staged row and ADMITTED here behind only the estimate margin; the clamp
    // makes the ladder refuse at the widened RESIDENT floor instead. Deleting the composition
    // check flips this arm back to `Fits` and red.
    let mut staging_free = registered.clone();
    staging_free.additional_prerequisites.clear();
    for capability in &mut staging_free.strategies {
        if capability.strategy == MemoryStrategy::StagedResidency {
            capability.support = gen_core::MemoryStrategySupport::Missing;
        }
    }
    match run(&staging_free) {
        KreaControlFit::TooBig {
            needed_gb,
            available_gb,
        } => {
            // The refusal quotes the widened resident clamp (byte-ceil wiggle tolerated),
            // proving the deep-rung floors were re-priced rather than merely excluded.
            let expected_needed = resident_floor * (1.0 + margin) + HEADROOM_GB;
            assert!(
                (needed_gb - expected_needed).abs() < 1e-6,
                "the refusal must quote the widened RESIDENT clamp: got {needed_gb}, want \
                     {expected_needed}"
            );
            assert!((available_gb - free).abs() < 1e-6);
        }
        other => panic!(
            "a staging-free deep rung must not admit on the staged working-set row: {other:?}"
        ),
    }
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
        super::fit_ladder(
            "q4",
            resident,
            staged,
            budget,
            None,
            Some(5.0),
            true,
            &live_test_closure_digest()
        ),
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
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
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

    // Constrained card: the staged floor's widened peak (25.0 × 1.02 = 25.5) fits a 28 GiB
    // effective budget where the resident floor's (30.9 × 1.02 = 31.518) does not.
    assert_eq!(
        fit(30.0),
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Sequential,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: true,
        }
    );

    // Margin mutation arm: a budget strictly BETWEEN the raw staged floor's requirement and its
    // allowance-widened one, so the raw floor fits and the widened one does not — a zeroed
    // allowance admits here and flips this red. The window is recomputed from the policy
    // (sc-22508: a manifest-row floor declares no weights/activation split, so it takes the
    // candle whole-peak `SameCellRecaptureSpread` residual) rather than written as a literal
    // picked for the retired blanket margin. The reported requirement carries the widening
    // (float-tolerant: the widening rounds up in integer bytes).
    let staged_floor_gb =
        predicted_control_sequential_peak_gb(&m, tier).expect("staged row") - HEADROOM_GB;
    let raw_needed_gb = staged_floor_gb + HEADROOM_GB;
    let widened_needed_gb = staged_floor_gb
        * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD)
        + HEADROOM_GB;
    let between_gb = (raw_needed_gb + widened_needed_gb) / 2.0;
    assert_too_big(fit(between_gb), widened_needed_gb, between_gb);

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
            reference_count: KREA_CONTROL_REFERENCE_COUNT,
        },
        contract.as_ref(),
        true,
    );
    assert!(
        matches!(beyond, KreaControlFit::BestEffort { .. }),
        "beyond the measured envelope the best-effort contract must survive: {beyond:?}"
    );
}

/// sc-18097 review: the floor-synthesis conjuncts this lane owns, each mutation-checked on the
/// SAME cell and budget where the unmutated ladder rejects from graded floors.
///
/// * **Loaded calibration identity** — a provider whose fingerprint drifted from the sc-16013
///   rows must not receive floors built from them (the candle mirror of the mlx gate's basis
///   identity gate). The mutated fingerprint is deliberately WELL-FORMED — it keeps the shipped
///   `…-v1` version-token grammar — and the mutated contract is asserted conformance-CLEAN, so
///   the outcome below is the identity conjunct's work and not a format rejection that would
///   have passed by grammar accident (the sc-18096 finding).
/// * **Evidence currency** — a superseded block (`measured: false`) keeps its never-reject
///   `BestEffort` contract; floors may not resurrect a reject from rows already known
///   non-current when recorded.
///
/// Both mutations must land on `BestEffort`, never on the `TooBig` the unmutated control arm
/// produces: withholding the floors must return the cell to the pre-sc-18097 posture, not
/// harden it.
#[test]
fn control_estimate_floors_require_current_evidence_and_the_loaded_identity() {
    use gen_core::MemoryCalibrationIdentity;

    let tier = "q4";
    let geometry_768 = MemoryGeometry {
        width: 768,
        height: 768,
        batch: 1,
        frames: 1,
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
    };
    let contract = registered_contract_for_tier(tier).expect("control contract");
    // Between the raw staged floor's requirement and its allowance-widened one, recomputed from
    // the policy (see the same window in
    // `an_unmeasured_geometry_is_estimate_graded_with_recoverable_oom_margins`): the raw floor
    // fits and the widened one does not, so the unmutated ladder rejects from its graded floors.
    let staged_floor_gb = predicted_control_sequential_peak_gb(
        &current_evidence(krea_manifest_with_chunking()),
        tier,
    )
    .expect("staged row")
        - HEADROOM_GB;
    let between_gb = staged_floor_gb
        * (1.0 + crate::ladder_margin_policy::CANDLE_RECAPTURE_SPREAD / 2.0)
        + HEADROOM_GB;
    let fit = |manifest: &JsonObject, contract: &MemoryProviderContract| {
        fit_ladder_for_entry_with_runtime(
            manifest,
            tier,
            Some(budget(between_gb)),
            0,
            geometry_768,
            Some(contract),
            true,
        )
    };

    let current = current_evidence(krea_manifest_with_chunking());
    assert!(
        matches!(fit(&current, &contract), KreaControlFit::TooBig { .. }),
        "control point: the unmutated cell must reject from its floors, or the arms below \
             prove nothing"
    );

    let mut drifted = contract.clone();
    drifted.calibration = Some(MemoryCalibrationIdentity::new(
        "sc-16013-krea-control-direct-1024-v2",
        contract.load_shape,
    ));
    assert!(
        drifted.conformance_errors().is_empty(),
        "the mutated fingerprint must be conformance-CLEAN so this arm exercises the identity \
             conjunct, not format validation"
    );
    assert!(
        matches!(fit(&current, &drifted), KreaControlFit::BestEffort { .. }),
        "a fingerprint drifted from the loaded provider must lose its estimate floors"
    );

    assert!(
        matches!(
            fit(&krea_manifest_with_chunking(), &contract),
            KreaControlFit::BestEffort { .. }
        ),
        "superseded evidence must not gain a reject through the estimate floors"
    );
}

#[test]
fn high_resolution_pose_with_adapters_selects_resident_only_when_its_estimate_fits() {
    let manifest = krea_manifest_measured();
    let contract = registered_contract_for_tier("bf16").unwrap();
    let geometry = MemoryGeometry {
        width: 1408,
        height: 2048,
        batch: 1,
        frames: 1,
        reference_count: KREA_CONTROL_REFERENCE_COUNT,
    };
    let adapter_bytes = crate::fit_gate::BYTES_PER_GIB as u64;
    let fit = |free_gb, runtime_verified| {
        fit_ladder_for_entry_with_runtime(
            &manifest,
            "bf16",
            Some(budget(free_gb)),
            adapter_bytes,
            geometry,
            Some(&contract),
            runtime_verified,
        )
    };
    let roomy = fit(1024.0, true);
    assert_eq!(
        roomy,
        KreaControlFit::Fits {
            offload_policy: OffloadPolicy::Resident,
            tile_vae_decode: false,
            chunk_attention: false,
            estimate_scoped: true,
        }
    );
    assert!(
        !matches!(
            fit(60.0, true),
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Resident,
                ..
            }
        ),
        "the 1024² row must not be reused unscaled for a 1408×2048 render"
    );
    assert_eq!(
        incurred_peak_gb_with_adapter_bytes(&roomy, &manifest, "bf16", adapter_bytes),
        None
    );
    assert!(
        matches!(fit(20.0, true), KreaControlFit::BestEffort { .. }),
        "a conservative high-resolution estimate must not hard-reject a bounded render"
    );
    assert!(
        matches!(fit(1024.0, false), KreaControlFit::BestEffort { .. }),
        "a custom or unverified artifact must not inherit the stock-model estimate"
    );
    let small = fit_ladder_for_entry_with_runtime(
        &manifest,
        "bf16",
        Some(budget(96.0)),
        adapter_bytes,
        MemoryGeometry {
            width: 768,
            height: 768,
            ..geometry
        },
        Some(&contract),
        true,
    );
    assert!(
        matches!(
            small,
            KreaControlFit::Fits {
                offload_policy: OffloadPolicy::Resident,
                estimate_scoped: true,
                ..
            }
        ),
        "in-envelope adapter requests must reuse the resident pipeline: {small:?}"
    );
}

/// Credit the previous control render's released pool before considering a slower residency rung.
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
