//! Pure metadata contracts; separate from dispatch-only test discovery.
use super::moss_training_acceptance::{generation_advanced, selected_weight_files};

#[test]
fn dmd_regression_payload_omits_cfg_while_ordinary_preview_retains_sealed_guidance() {
    let ordinary = generation_advanced("off").unwrap();
    assert_eq!(ordinary["guidanceScale"], serde_json::json!(1.0));
    for mode in ["unmerged_bf16", "full_fp32"] {
        let dmd = generation_advanced(mode).unwrap();
        assert!(dmd.get("guidanceScale").is_none());
        assert!(dmd.get("negativePrompt").is_none());
        assert_eq!(dmd["steps"], serde_json::json!(8));
        assert_eq!(dmd["mossImage"]["acceleration"], mode);
    }
    assert!(generation_advanced("unknown").is_err());
}
use sceneworks_worker::test_support::gen_core;
use std::collections::BTreeSet;

#[test]
fn weight_preflight_uses_complete_production_component_map_without_model_execution() {
    for parent in ["base", "sft"] {
        let files = selected_weight_files(parent).unwrap();
        let repositories: BTreeSet<_> = files.iter().map(|(repo, _, _)| repo.as_str()).collect();
        println!("{parent}/lora and {parent}/lokr production repository union: {repositories:?}");
        assert!(files
            .iter()
            .all(
                |(repo, rev, file)| gen_core::moss_image::moss_image_asset(repo, rev, file)
                    .is_some()
            ));
        assert!(files
            .iter()
            .any(|(_, _, file)| file.starts_with("text_encoder/")));
        assert!(files.iter().any(|(_, _, file)| file.starts_with("vae/")));
        assert!(files
            .iter()
            .any(|(_, _, file)| file.starts_with("transformer/")));
        if parent == "base" {
            assert!(files
                .iter()
                .all(|(repo, _, _)| repo != gen_core::moss_image::DMD_REPO));
        } else {
            assert!(files
                .iter()
                .any(|(repo, _, _)| repo == gen_core::moss_image::DMD_REPO));
        }
    }
}
