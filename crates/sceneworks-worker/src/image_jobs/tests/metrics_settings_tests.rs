use super::super::artifact_resolution::completed_image_metrics_use_candle_flux1_exact_resolution;
use super::build_image_metrics;
#[cfg(not(test))]
use super::build_image_metrics;
use super::effective_quant_label;
use super::image_settings_metrics;
use super::super::ImageRequest;
use serde_json::json;

fn request(value: serde_json::Value) -> ImageRequest {
    ImageRequest::from_payload(value.as_object().unwrap())
}

#[cfg(native_lane)]
#[test]
fn completed_metrics_replay_exact_flux1_tiers_only_on_the_candle_build() {
    let candle_build = cfg!(all(not(target_os = "macos"), feature = "backend-candle"));
    for model in ["flux_schnell", "flux_dev"] {
        assert_eq!(
            completed_image_metrics_use_candle_flux1_exact_resolution(model),
            candle_build,
            "{model} must use Candle's q4-default exact resolver only on the Candle build"
        );
    }
    assert!(!completed_image_metrics_use_candle_flux1_exact_resolution(
        "qwen_image"
    ));
}

#[cfg(native_lane)]
#[test]
fn moss_edit_completion_metrics_do_not_invent_q8_for_the_r15_request_shape() {
    // Actual R15 MLX edit shape: no quant field, BF16 native components, 30 steps/CFG4,
    // shift5, 768x512 and seed173. Exercise the same helper used after event consumption,
    // rather than supplying an already-correct label to image_settings_metrics.
    for model in ["moss_image_base", "moss_image_sft"] {
        let req = request(json!({
            "model": model, "mode": "edit_image", "prompt": "Change only the blue bottle body and blue cap to deep red.",
            "negativePrompt": "", "width": 768, "height": 512, "count": 1, "seed": 173,
            "referenceAssetIds": ["asset_reference"],
            "advanced": {"steps": 30, "guidanceScale": 4, "schedulerShift": 5,
                "mossImage": {"acceleration": "off", "uncond_skip_layers": []},
                "mossImageEdit": {"reference_max_edge": 512, "use_kv_cache": true}}
        }));
        let metrics = build_image_metrics(&req, Some(30), 1, None);
        assert_eq!(metrics.quant_label.as_deref(), Some("bf16"));
        assert_eq!(metrics.quant_bits, None);
        assert_eq!(metrics.model.as_deref(), Some(model));
        assert_eq!(metrics.steps, Some(30));
        assert_eq!(metrics.image_count, Some(1));
        assert_eq!(
            (metrics.width, metrics.height, metrics.seed),
            (Some(768), Some(512), Some(173))
        );
    }
}

#[cfg(native_lane)]
#[test]
fn ordinary_moss_dmd_completion_keeps_generic_quant_metrics_semantics() {
    use gen_core::moss_image::{MossDmdPrecision, MossImageAcceleration, MossImageOptions};
    for precision in [MossDmdPrecision::UnmergedBf16, MossDmdPrecision::FullFp32] {
        let options = MossImageOptions {
            acceleration: MossImageAcceleration::Dmd(precision),
            ..Default::default()
        };
        let req = request(
            json!({"model": "moss_image_sft", "mode": "text_to_image", "prompt": "mist",
                "advanced": {"mossImage": options}}),
        );
        let expected = effective_quant_label(&req, None);
        let metrics = build_image_metrics(&req, Some(8), 1, None);
        assert_eq!(metrics.quant_label, expected.0);
        assert_eq!(metrics.quant_bits, expected.1.map(|bits| bits as u32));
    }
}

#[test]
fn default_run_reports_effective_settings_not_blank() {
    let req = request(json!({
        "projectId": "p", "model": "qwen_image", "prompt": "mist",
        "width": 1024, "height": 1024, "seed": 42
    }));
    let metrics =
        image_settings_metrics(&req, Some(20), Some(4.0), Some("q8".to_owned()), Some(8), 4);
    assert_eq!(metrics.model.as_deref(), Some("qwen_image"));
    assert_eq!(metrics.image_count, Some(4));
    assert_eq!(metrics.quant_label.as_deref(), Some("q8"));
    assert_eq!(metrics.quant_bits, Some(8));
    // A default run is not blank — sampler/scheduler/method carry the effective default.
    assert_eq!(metrics.sampler.as_deref(), Some("default"));
    assert_eq!(metrics.scheduler.as_deref(), Some("default"));
    assert_eq!(metrics.guidance_method.as_deref(), Some("cfg"));
    assert_eq!(metrics.use_pid, Some(false));
    assert_eq!(metrics.steps, Some(20));
    assert_eq!(metrics.seed, Some(42));
    assert_eq!(metrics.width, Some(1024));
    assert_eq!(
        metrics
            .guidance_scale
            .as_ref()
            .and_then(serde_json::Number::as_f64),
        Some(4.0)
    );
}

#[test]
fn advanced_overrides_are_reported() {
    let req = request(json!({
        "projectId": "p", "model": "sdxl", "prompt": "mist", "width": 832, "height": 1216,
        "advanced": {
            "sampler": "dpmpp_2m", "scheduler": "karras", "schedulerShift": 3.0,
            "usePid": true, "pidTarget": "2k", "guidanceMethod": "cfgpp"
        }
    }));
    let metrics = image_settings_metrics(&req, Some(30), None, Some("bf16".to_owned()), None, 1);
    assert_eq!(metrics.sampler.as_deref(), Some("dpmpp_2m"));
    assert_eq!(metrics.scheduler.as_deref(), Some("karras"));
    assert_eq!(
        metrics
            .scheduler_shift
            .as_ref()
            .and_then(serde_json::Number::as_f64),
        Some(3.0)
    );
    assert_eq!(metrics.use_pid, Some(true));
    assert_eq!(metrics.pid_target.as_deref(), Some("2k"));
    assert_eq!(metrics.guidance_method.as_deref(), Some("cfgpp"));
    assert_eq!(metrics.quant_label.as_deref(), Some("bf16"));
    assert_eq!(metrics.quant_bits, None);
}
