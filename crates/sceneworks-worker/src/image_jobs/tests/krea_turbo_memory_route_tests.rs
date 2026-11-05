use super::super::tier_quality_rank;
use super::super::{
    candle_adapter_resident_bytes, capability_downtier_floor,
    krea_streamed_blocks_adapter_evidence, krea_turbo_capability_reject_message,
    krea_turbo_memory_route, krea_unverified_resident_decision,
};
use super::choose_downtier;
use super::DowntierPick;
use super::TierFit;
use serde_json::{Map, Value};

fn historical_builtin_krea_turbo_manifest() -> Map<String, Value> {
    let jsonc = include_str!("../../../../../config/manifests/builtin.models.jsonc");
    let mut parsed: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(jsonc))
            .expect("builtin model manifest parses");
    let model = parsed["models"]
        .as_array_mut()
        .expect("models array")
        .iter_mut()
        .find(|model| model["id"].as_str() == Some("krea_2_turbo"))
        .expect("Krea 2 Turbo manifest entry");
    model["candle"]["turboFit"]["calibrationFingerprint"] =
        Value::String("krea-turbo-cuda-phase-curves-v1".into());
    // sc-17774: these tests make the ladder historical through the FINGERPRINT, which is their
    // subject. Stamp the live closure so currency is not a second, unintended historical axis —
    // the shipped digest is behind the pin (`gen-core` moved), and letting that decide too would
    // stop these tests exercising the fingerprint path they are named for.
    model["candle"]["turboFit"]["inferenceClosureDigest"] = Value::String(
        sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo")
            .unwrap_or_default(),
    );
    // sc-17097 removed the ABI re-stamp that used to sit here. Overwriting the shipped
    // `calibrationAbi` with the current constant is what kept every selector test green while
    // production rejected the same manifest: no test in the tree ever read the shipped value.
    model
        .as_object()
        .expect("Krea 2 Turbo manifest object")
        .clone()
}

#[test]
fn only_plain_turbo_text_to_image_uses_the_memory_ladder() {
    assert!(krea_turbo_memory_route(
        "krea_2_turbo",
        false,
        false,
        false,
        false,
        false,
        false,
        false
    ));
    for excluded_surface in 0..7 {
        let mut flags = [false; 7];
        flags[excluded_surface] = true;
        assert!(
            !krea_turbo_memory_route(
                "krea_2_turbo",
                flags[0],
                flags[1],
                flags[2],
                flags[3],
                flags[4],
                flags[5],
                flags[6],
            ),
            "surface flag {excluded_surface} must retain its established route"
        );
    }
    for engine in ["krea_2_raw", "krea_2_turbo_edit", "krea_2_turbo_control"] {
        assert!(!krea_turbo_memory_route(
            engine, false, false, false, false, false, false, false
        ));
    }
}

#[test]
fn adapter_bytes_disable_krea_streaming_and_missing_bytes_fail_closed() {
    let root_guard = tempfile::Builder::new()
        .prefix("krea-adapter-evidence-")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let adapter = root.join("adapter.safetensors");
    std::fs::write(&adapter, vec![0_u8; 321]).unwrap();
    let measured = vec![gen_core::AdapterSpec::new(
        adapter,
        1.0,
        gen_core::AdapterKind::Lora,
    )];
    assert_eq!(krea_streamed_blocks_adapter_evidence(&[]), (true, Some(0)));
    assert_eq!(
        krea_streamed_blocks_adapter_evidence(&measured),
        (false, Some(321))
    );
    assert_eq!(
        candle_adapter_resident_bytes("krea_2_turbo", "bf16", 321),
        321
    );
    assert_eq!(candle_adapter_resident_bytes("z_image", "bf16", 321), 321);
    assert_eq!(
        candle_adapter_resident_bytes("z_image_control", "bf16", 321),
        321
    );
    assert_eq!(candle_adapter_resident_bytes("sdxl", "bf16", 321), 0);
    assert_eq!(candle_adapter_resident_bytes("sdxl", "q4", 321), 321);
    for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
        assert_eq!(candle_adapter_resident_bytes(model, "bf16", 321), 321);
    }

    let missing = vec![gen_core::AdapterSpec::new(
        root.join("missing.safetensors"),
        1.0,
        gen_core::AdapterKind::Lora,
    )];
    assert_eq!(
        krea_streamed_blocks_adapter_evidence(&missing),
        (false, None),
        "missing byte evidence must retain the count-based fail-closed fallback"
    );
}

#[test]
fn adapter_bytes_flip_the_candle_resident_fit_boundary() {
    use crate::vram_gate::{VramBudget, HEADROOM_GB};

    let manifest = serde_json::json!({
        "candle": { "vramGbByTier": { "q4": 7.0 } }
    })
    .as_object()
    .cloned()
    .unwrap();
    let budget = Some(VramBudget {
        free_gb: 7.5 + HEADROOM_GB,
        total_gb: 16.0,
    });
    assert_eq!(
        super::candle_tier_fit(&manifest, "q4", budget, false, 0),
        super::TierFit::Fits
    );
    assert!(matches!(
        super::candle_tier_fit(&manifest, "q4", budget, false, 1024 * 1024 * 1024),
        super::TierFit::TooBig { .. }
    ));
}

#[test]
fn unverified_krea_never_uses_legacy_sequential_even_when_it_would_fit() {
    use crate::vram_gate::{FitDecision, VramBudget};

    let budget = Some(VramBudget {
        free_gb: 24.0,
        total_gb: 24.0,
    });
    assert!(matches!(
        krea_unverified_resident_decision(Some(27.7), budget),
        FitDecision::TooBig {
            needed_gb: 27.7,
            available_gb: 24.0,
        }
    ));
    assert!(
        crate::vram_gate::sequential_overflow_gb(Some(23.9), budget).is_none(),
        "the regression boundary intentionally leaves the legacy sequential estimate fitting"
    );
}

#[test]
fn historical_constrained_q8_reject_never_capability_downtiers_to_q4() {
    use super::{choose_downtier, tier_quality_rank, DowntierPick, TierFit};
    use crate::vram_gate::{
        krea_turbo_fit_with_runtime as krea_turbo_fit, KreaTurboFit, VramBudget,
    };

    let manifest = historical_builtin_krea_turbo_manifest();
    // sc-17097 re-measurement moved both streamed floors (q4 6.32 GiB, q8 7.01 GiB including the
    // 2 GiB reserve). This budget still sits in the only window that makes the test meaningful:
    // Q8 cannot fit at any rung while Q4 can, so a downtier is available and must NOT be taken.
    let available_gb = 7.00;
    let budget = Some(VramBudget {
        free_gb: available_gb,
        total_gb: available_gb,
    });
    let fit = |tier| {
        let runtime = crate::vram_gate::KreaRuntimeEvidenceContext::verified_for_test(tier);
        // The legacy 2 GB reserve these historical thresholds were calibrated against, which
        // `ladder_reserve_gb` reproduces for a card with >= 1.75 GB idle residency (sc-22667).
        match krea_turbo_fit(
            &manifest,
            tier,
            1024,
            1024,
            budget,
            crate::vram_gate::HEADROOM_GB,
            true,
            Some(&runtime),
        )
        .expect("Q8 and Q4 have measured ladder curves")
        {
            KreaTurboFit::Resident { .. } | KreaTurboFit::Fits { .. } => TierFit::Fits,
            KreaTurboFit::Reject { needed_gb, .. } => TierFit::TooBig {
                needed_gb,
                available_gb,
            },
            KreaTurboFit::Unverified { .. } => TierFit::TooBig {
                needed_gb: f64::INFINITY,
                available_gb,
            },
        }
    };
    assert!(matches!(fit("q8"), TierFit::TooBig { .. }));
    assert_eq!(fit("q4"), TierFit::Fits);

    let floor = capability_downtier_floor(true, "q8", None).expect("Q8 precision floor");
    let candidates: Vec<_> = ["q8", "q4"]
        .into_iter()
        .filter(|tier| tier_quality_rank(tier) >= tier_quality_rank(floor))
        .map(|tier| (tier, fit(tier)))
        .collect();
    assert!(matches!(
        choose_downtier("q8", &candidates),
        DowntierPick::Reject { tier: "q8", .. }
    ));
}

#[test]
fn historical_constrained_bf16_reject_never_capability_downtiers_to_q8_or_q4() {
    use super::{choose_downtier, tier_quality_rank, DowntierPick, TierFit};
    use crate::vram_gate::{
        krea_turbo_fit_with_runtime as krea_turbo_fit, KreaTurboFit, VramBudget,
    };

    let manifest = historical_builtin_krea_turbo_manifest();
    // sc-17097: BF16's streamed floor is 10.42 GiB after the re-measurement, so this budget keeps
    // BF16 unfittable while BOTH lower tiers fit - the case where downtiering is most tempting.
    let available_gb = 10.41;
    let budget = Some(VramBudget {
        free_gb: available_gb,
        total_gb: available_gb,
    });
    let fit = |tier| {
        let runtime = crate::vram_gate::KreaRuntimeEvidenceContext::verified_for_test(tier);
        // The legacy 2 GB reserve these historical thresholds were calibrated against, which
        // `ladder_reserve_gb` reproduces for a card with >= 1.75 GB idle residency (sc-22667).
        match krea_turbo_fit(
            &manifest,
            tier,
            1024,
            1024,
            budget,
            crate::vram_gate::HEADROOM_GB,
            true,
            Some(&runtime),
        )
        .expect("BF16, Q8, and Q4 have measured ladder curves")
        {
            KreaTurboFit::Resident { .. } | KreaTurboFit::Fits { .. } => TierFit::Fits,
            KreaTurboFit::Reject { needed_gb, .. } => TierFit::TooBig {
                needed_gb,
                available_gb,
            },
            KreaTurboFit::Unverified { .. } => TierFit::TooBig {
                needed_gb: f64::INFINITY,
                available_gb,
            },
        }
    };
    assert!(matches!(fit("bf16"), TierFit::TooBig { .. }));
    assert_eq!(fit("q8"), TierFit::Fits);
    assert_eq!(fit("q4"), TierFit::Fits);

    let floor = capability_downtier_floor(true, "bf16", None).expect("BF16 precision floor");
    let candidates: Vec<_> = ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| tier_quality_rank(tier) >= tier_quality_rank(floor))
        .map(|tier| (tier, fit(tier)))
        .collect();
    assert!(matches!(
        choose_downtier("bf16", &candidates),
        DowntierPick::Reject { tier: "bf16", .. }
    ));
}

#[test]
fn constrained_bf16_capability_reject_offers_only_truthful_manual_escapes() {
    let immediate_below = krea_turbo_capability_reject_message(
        "Krea 2 Turbo",
        "bf16",
        10.64,
        10.63,
        "0",
        None,
        &["q8", "q4"],
    );
    assert!(
        immediate_below.contains("bf16 precision floor"),
        "{immediate_below}"
    );
    assert!(
        immediate_below.contains("smaller installed tier (Q8 / Q4)"),
        "{immediate_below}"
    );
    assert!(
        immediate_below.contains("needs 10.64 GiB")
            && immediate_below.contains("has 10.63 GiB available"),
        "immediate-below copy must preserve the measured distinction: {immediate_below}"
    );
    assert!(
        !immediate_below.contains("lower the output resolution"),
        "the fixed BF16 text floor makes resolution advice false: {immediate_below}"
    );
    assert!(
        !immediate_below.contains("smallest installed tier"),
        "{immediate_below}"
    );

    let geometry_limited = krea_turbo_capability_reject_message(
        "Krea 2 Turbo",
        "bf16",
        12.0,
        11.0,
        "0",
        Some((768, 768)),
        &[],
    );
    assert!(
        geometry_limited.contains("lower the output resolution to 768x768 or below"),
        "{geometry_limited}"
    );
    assert!(
        !geometry_limited.contains("select a smaller installed tier"),
        "{geometry_limited}"
    );
}
