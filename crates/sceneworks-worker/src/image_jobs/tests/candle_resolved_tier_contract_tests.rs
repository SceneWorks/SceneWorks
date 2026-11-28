use super::super::artifact_resolution::candle_flux1_packed_requested_tier;
use super::super::artifact_resolution::candle_flux1_packed_resolution_is_exact;
use super::super::load_spec::apply_declared_candle_image_load_shape;
use super::super::load_spec::candle_shared_memory_tier;
use super::super::load_spec::load_spec;
use super::super::metrics::effective_quant_label_gated;
use super::super::mlx_model;
use super::super::quantization::resolve_quant;
use super::super::sampling::mlx_raw_settings;
use super::super::standard_tier_subdir_gated;
use super::super::ImageRequest;
use super::super::LoadSpec;
use super::super::Quant;
use super::super::WeightsSource;
use super::candle_quant_for_resolved_tier;
use super::candle_resolved_tier_key;
use crate::JsonObject;
use crate::Path;
use crate::PathBuf;
use crate::Value;
use serde_json::json;

/// sc-26241: the shared selector binds FLUX 3 to the package that LOADS — the plan's tier
/// identity, FP8R as FP8R — never relabelled bf16.
#[test]
fn flux3_shared_memory_tier_is_the_loaded_package() {
    for model in ["flux3_klein", "flux3_klein_base"] {
        for tier in ["bf16", "q4", "q8", "fp8r", "nvfp4"] {
            assert_eq!(
                candle_shared_memory_tier(model, tier),
                tier,
                "{model}/{tier}"
            );
        }
    }
    assert_eq!(candle_shared_memory_tier("iris_3b", "fp8r"), "bf16");
    assert_eq!(candle_shared_memory_tier("z_image", "q4"), "q4");
}

fn bound_request(value: Value) -> ImageRequest {
    let payload = value.as_object().expect("request object");
    ImageRequest::from_resolved(payload, payload).expect("fixture options bind")
}

/// sc-22732: `ideogram_4` / `ideogram_4_turbo` join the list. Their descriptor advertises
/// `supported_quants: [Q4, Q8]`, so without the carve-out this function answered
/// `(Some(Quant::Q4), Some(4))` at q4 — and `candle-gen-ideogram`'s production
/// `IdeogramLoadReceipt::capture` -> `validate_load_shape` refuses `quantize: Some(_)` by name,
/// so every candle Ideogram q4/q8 load failed at the loader. The artifact proves the tier; the
/// request knob never does.
#[test]
fn packed_turnkeys_keep_load_quantization_none_for_every_public_route() {
    for model in [
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "ideogram_4",
        "ideogram_4_turbo",
    ] {
        for (tier, expected_bits) in [("bf16", None), ("q4", Some(4)), ("q8", Some(8))] {
            let request = bound_request(json!({
                "model": model,
                "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
            }));
            let weights = PathBuf::from(tier);
            assert_eq!(
                candle_quant_for_resolved_tier(&request, tier, &weights, true, false),
                (None, expected_bits),
                "{model}:{tier}"
            );
        }
    }
}

fn flux1_request(model: &str, quantize: Option<Value>) -> ImageRequest {
    let mut payload = json!({
        "model": model,
        "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
    });
    if let Some(quantize) = quantize {
        payload["advanced"] = json!({ "mlxQuantize": quantize });
    }
    bound_request(payload)
}

#[test]
fn flux1_candle_packed_tiers_are_exact_and_keep_the_load_receipt_truthful() {
    for model_id in ["flux_dev", "flux_schnell"] {
        let descriptor = mlx_model(model_id).expect("linked Candle FLUX.1 descriptor");
        assert!(
            !descriptor.supports_quant(),
            "{model_id}'s pinned descriptor must keep supported_quants empty"
        );

        for (requested, expected) in [(None, "q4"), (Some(json!(4)), "q4"), (Some(json!(8)), "q8")]
        {
            let request = flux1_request(model_id, requested);
            assert_eq!(
                candle_flux1_packed_requested_tier(&request).unwrap(),
                expected
            );

            let exact = Path::new("/models").join(model_id).join(expected);
            let other = Path::new("/models")
                .join(model_id)
                .join(if expected == "q4" { "q8" } else { "q4" });
            assert!(candle_flux1_packed_resolution_is_exact(expected, &exact));
            assert!(
                !candle_flux1_packed_resolution_is_exact(expected, &other),
                "a requested {expected} must not fall back to {}",
                other.display()
            );

            let expected_bits = if expected == "q4" { Some(4) } else { Some(8) };
            let (load_quant, receipt_bits) = candle_quant_for_resolved_tier(
                &request,
                expected,
                &exact,
                descriptor.supports_quant(),
                false,
            );
            assert_eq!((load_quant, receipt_bits), (None, expected_bits));
            assert_eq!(
                load_spec(exact.clone(), load_quant, Vec::new(), None).quantize,
                None,
                "the prepacked FLUX.1 descriptor rejects a load-time quant instruction"
            );
            assert_eq!(
                mlx_raw_settings(&request, "SceneWorks/flux1-mlx", 28, receipt_bits, None)
                    .get("mlxQuantize")
                    .and_then(Value::as_i64),
                expected_bits,
                "the recipe must retain the exact packed artifact tier"
            );
            assert_eq!(
                effective_quant_label_gated(&request, false, Some(&exact)),
                (Some(expected.to_owned()), expected_bits),
                "asset telemetry must agree with the recipe's packed tier"
            );
        }
    }

    for unsupported in [json!(0), json!(6), json!(-1), json!(true), json!("4.0")] {
        let value = json!({ "model": "flux_dev", "advanced": { "mlxQuantize": unsupported } });
        let payload = value.as_object().unwrap();
        match ImageRequest::from_resolved(payload, payload) {
            Ok(request) => assert!(candle_flux1_packed_requested_tier(&request).is_err()),
            Err(_) => assert!(unsupported.is_boolean() || unsupported.is_string()),
        }
    }
}
