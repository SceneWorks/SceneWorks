use super::apply_request_scoped_candle_residency;
#[cfg(not(target_os = "macos"))]
use super::auto_tier_fit_decision;
use super::auto_tier_streaming_disclosure;
use super::candle_base_memory_request_mode;
use super::flux2_klein_reference_bearing_mode;
use super::krea_streaming_measurement;
use super::progress_with_auto_tier_streaming;
use super::rejects_unverified_shared_memory_fallback;
use super::scrub_untrusted_memory_strategy_disclosure;
use super::verified_only_memory_family_label;
use super::AutoTierFitDecision;
use super::AUTO_TIER_STREAMING_DISCLOSURE_KEY;
use crate::JsonObject;
use serde_json::json;
#[test]
fn sequential_admission_sets_the_request_bit_without_erasing_other_rungs() {
    let mut resident = None;
    apply_request_scoped_candle_residency(false, &mut resident);
    assert!(
        resident.is_none(),
        "resident admission leaves request memory absent"
    );

    let mut sequential = None;
    apply_request_scoped_candle_residency(true, &mut sequential);
    assert!(
        sequential.is_some_and(|memory| memory.stage_residency),
        "mutation guard: sequential admission must reach the request lifecycle bit"
    );

    let mut composed = Some(gen_core::GenerationMemory {
        tile_vae_decode: true,
        ..Default::default()
    });
    apply_request_scoped_candle_residency(true, &mut composed);
    let composed = composed.expect("existing rung memory is preserved");
    assert!(composed.stage_residency);
    assert!(composed.tile_vae_decode);
}

#[test]
fn verified_only_families_never_fall_back_to_an_unverified_optimized_strategy() {
    assert!(rejects_unverified_shared_memory_fallback("z_image", false));
    assert!(rejects_unverified_shared_memory_fallback(
        "flux2_dev",
        false
    ));
    assert!(rejects_unverified_shared_memory_fallback(
        "flux2_klein_9b",
        false
    ));
    for engine in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
        assert!(rejects_unverified_shared_memory_fallback(engine, false));
        assert!(!rejects_unverified_shared_memory_fallback(engine, true));
        assert_eq!(verified_only_memory_family_label(engine), Some("Chroma1"));
    }
    for engine in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
        assert_eq!(
            candle_base_memory_request_mode(engine, "image_generation"),
            "text_to_image"
        );
        assert_eq!(
            candle_base_memory_request_mode(engine, "text_to_image"),
            "text_to_image"
        );
        for refused in ["edit_image", "image_inpaint", "style_variations"] {
            assert_eq!(candle_base_memory_request_mode(engine, refused), refused);
        }
    }
    for engine in ["ideogram_4", "ideogram_4_turbo"] {
        assert_eq!(
            candle_base_memory_request_mode(engine, "image_generation"),
            "text_to_image"
        );
        for mode in [
            "text_to_image",
            "image_to_image",
            "edit_image",
            "image_inpaint",
        ] {
            assert_eq!(candle_base_memory_request_mode(engine, mode), mode);
        }
    }
    assert!(!rejects_unverified_shared_memory_fallback("z_image", true));
    assert!(!rejects_unverified_shared_memory_fallback(
        "flux2_dev",
        true
    ));
    assert!(!rejects_unverified_shared_memory_fallback(
        "flux2_klein_9b",
        true
    ));
    assert!(!rejects_unverified_shared_memory_fallback(
        "qwen_image",
        false
    ));
    assert_eq!(
        verified_only_memory_family_label("flux2_dev"),
        Some("FLUX.2-dev")
    );
    assert_eq!(
        verified_only_memory_family_label("flux2_klein_9b"),
        Some("FLUX.2 Klein")
    );
}

#[test]
fn flux2_base_mode_is_canonicalized_without_changing_other_families() {
    assert_eq!(
        candle_base_memory_request_mode("flux2_dev", "style_variations"),
        "text_to_image"
    );
    assert_eq!(
        candle_base_memory_request_mode("flux2_klein_9b", "image_generation"),
        "text_to_image"
    );
    for mode in [
        "edit_image",
        "reference",
        "image_to_image",
        "character_image",
        "style_variations",
    ] {
        assert_eq!(
            candle_base_memory_request_mode("flux2_klein_9b", mode),
            mode
        );
        assert!(flux2_klein_reference_bearing_mode(mode));
    }
    for engine in ["z_image", "qwen_image", "flux1_dev"] {
        assert_eq!(
            candle_base_memory_request_mode(engine, "style_variations"),
            "style_variations"
        );
    }
    for engine in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
        assert_eq!(
            candle_base_memory_request_mode(engine, "image_generation"),
            "text_to_image"
        );
        assert_eq!(
            candle_base_memory_request_mode(engine, "text_to_image"),
            "text_to_image"
        );
        assert_eq!(
            candle_base_memory_request_mode(engine, "style_variations"),
            "style_variations",
            "retired Chroma style alias must reach the exhaustive declaration and be refused"
        );
    }
}

#[test]
fn flux_without_exact_evidence_preserves_sequential_fallback_and_never_names_zimage() {
    for engine in ["flux1_schnell", "flux1_dev"] {
        assert!(!rejects_unverified_shared_memory_fallback(engine, false));
        assert_eq!(verified_only_memory_family_label(engine), None);
    }
    assert_eq!(
        verified_only_memory_family_label("z_image"),
        Some("Z-Image")
    );
}

#[test]
fn auto_rung_four_disclosure_names_cause_and_real_krea_cost() {
    let disclosure =
        auto_tier_streaming_disclosure(false, "q8", AutoTierFitDecision::StreamedBlocks)
            .expect("Auto plus rung 4 must disclose streaming");
    assert_eq!(
        disclosure.measured_materialization_ms_per_block,
        Some(101.7)
    );
    assert_eq!(disclosure.measured_materialization_ms_per_step, Some(3_051));
    let progress = disclosure.progress_message("Image 1/1 — step 1/8.");
    assert!(progress.contains("Streaming transformer blocks"));
    assert!(progress.contains("hold the auto-selected q8 tier"));
    let trace = disclosure.telemetry("job-1", "krea_2_turbo");
    assert_eq!(trace["strategy"], "bounded_transformer_residency");
    assert_eq!(
        trace["cause"],
        "streaming_transformer_blocks_to_hold_auto_selected_tier"
    );
    assert_eq!(trace["measuredMaterializationMsPerStep"], 3_051);
    let mut raw_settings = JsonObject::new();
    raw_settings.insert(
        AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
        disclosure.record(),
    );
    assert_eq!(
        raw_settings["memoryStrategyDisclosure"],
        disclosure.record()
    );

    // Mutation guards: removing either predicate would mislabel an explicit choice or a cheaper
    // rung as Auto's block-streaming decision.
    assert!(
        auto_tier_streaming_disclosure(true, "q8", AutoTierFitDecision::StreamedBlocks).is_none()
    );
    assert!(auto_tier_streaming_disclosure(false, "q8", AutoTierFitDecision::Other).is_none());
}

#[test]
fn client_raw_settings_cannot_forge_auto_rung_four_disclosure() {
    let mut advanced = JsonObject::new();
    advanced.insert(
        AUTO_TIER_STREAMING_DISCLOSURE_KEY.to_owned(),
        json!({
            "tier": "q8",
            "cause": "streaming_transformer_blocks_to_hold_auto_selected_tier",
        }),
    );

    scrub_untrusted_memory_strategy_disclosure(&mut advanced);
    assert!(!advanced.contains_key(AUTO_TIER_STREAMING_DISCLOSURE_KEY));
    assert_eq!(
        progress_with_auto_tier_streaming("Image 1/1 — step 1/8.", None),
        "Image 1/1 — step 1/8."
    );
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn final_krea_fit_enum_is_the_trusted_disclosure_source() {
    let fit = |strategy| crate::vram_gate::KreaTurboFit::Fits {
        phases: crate::vram_gate::KreaTurboPhasePeaks {
            text_gb: 1.0,
            denoise_gb: 1.0,
            decode_gb: 1.0,
        },
        needed_gb: 1.0,
        selection: gen_core::MemorySelection {
            strategy,
            parameters: Default::default(),
            tier: gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: Some(gen_core::Quant::Q8),
                component_precision_floors: &[],
            },
        },
        memory: gen_core::GenerationMemory::default(),
        basis: crate::memory_strategy::CandidateBasis::Measured,
        admitted: crate::candle_memory_strategy::AdmittedBudget {
            needed_gb: 1.0,
            available_gb: 1.0,
            reserve_gb: 0.0,
        },
        estimate_scoped: false,
    };
    let streamed = fit(gen_core::MemoryStrategy::BoundedTransformerResidency);
    let cheaper = fit(gen_core::MemoryStrategy::BoundedAttention);

    assert_eq!(
        auto_tier_fit_decision(Some(&streamed)),
        AutoTierFitDecision::StreamedBlocks
    );
    assert_eq!(
        auto_tier_fit_decision(Some(&cheaper)),
        AutoTierFitDecision::Other
    );
    assert_eq!(auto_tier_fit_decision(None), AutoTierFitDecision::Other);
}

#[test]
fn streaming_measurements_are_krea_specific_and_do_not_invent_bf16_cost() {
    assert_eq!(krea_streaming_measurement("q4"), (Some(56.2), Some(1_686)));
    assert_eq!(krea_streaming_measurement("q8"), (Some(101.7), Some(3_051)));
    assert_eq!(krea_streaming_measurement("bf16"), (None, None));
}
