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

const MAGE_IDS: &[&str] = &[
    "mage_flow_base",
    "mage_flow",
    "mage_flow_turbo",
    "mage_flow_edit_base",
    "mage_flow_edit",
    "mage_flow_edit_turbo",
];

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

fn request(bits: i64) -> ImageRequest {
    ImageRequest::from_payload(
        json!({
            "model": "mage_flow_base",
            "advanced": { "mlxQuantize": bits },
            "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
        })
        .as_object()
        .expect("request object"),
    )
}

fn seed_only_tier(root: &Path, tier: &str) {
    let transformer = root.join(tier).join("transformer");
    std::fs::create_dir_all(&transformer).expect("tier transformer dir");
    // A syntactically VALID, tensor-free safetensors file rather than the pre-sc-22667 one-byte
    // `b"x"` marker. The tier probe this fixture serves only needs the file to EXIST, but since
    // the sc-22657 pin the shared image contract prices every staged component by reading its
    // safetensors HEADER (`gen_core::materialized_header_bytes`), and a one-byte file cannot
    // even yield the 8-byte header length — the read failed with "failed to fill whole buffer"
    // and took the whole contract down. An empty header keeps the marker's meaning (this tier is
    // present) while pricing it at the zero bytes it actually materializes.
    let header = br#"{}"#;
    let mut empty_safetensors = (header.len() as u64).to_le_bytes().to_vec();
    empty_safetensors.extend_from_slice(header);
    std::fs::write(transformer.join("model.safetensors"), &empty_safetensors)
        .expect("tier presence marker");
    let config = match tier {
        "q4" => r#"{"quantization":{"bits":4,"group_size":64}}"#,
        "q8" => r#"{"quantization":{"bits":8,"group_size":64}}"#,
        _ => "{}",
    };
    std::fs::write(transformer.join("config.json"), config).expect("tier config");
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
            let request = ImageRequest::from_payload(
                json!({
                    "model": model,
                    "modelManifestEntry": { "mlx": { "standardTierLayout": true } }
                })
                .as_object()
                .expect("request object"),
            );
            let weights = PathBuf::from(tier);
            assert_eq!(
                candle_quant_for_resolved_tier(&request, tier, &weights, true, false),
                (None, expected_bits),
                "{model}:{tier}"
            );
        }
    }
}

fn mage_spec(engine_id: &str, weights: &Path, quant: Option<Quant>) -> LoadSpec {
    let mut spec = load_spec(weights.to_path_buf(), quant, Vec::new(), None);
    spec = apply_declared_candle_image_load_shape(
        engine_id,
        weights.file_name().and_then(|value| value.to_str()),
        if engine_id.contains("_edit") {
            "edit_image"
        } else {
            "image_generation"
        },
        &JsonObject::new(),
        spec,
        false,
    );
    let descriptor = crate::inference_runtime::media_descriptor(engine_id)
        .unwrap_or_else(|| panic!("missing Mage descriptor {engine_id}"));
    descriptor
        .required_components
        .iter()
        .fold(spec, |spec, id| {
            spec.with_component(
                *id,
                WeightsSource::Dir(weights.join(format!("component-{id}"))),
            )
        })
}

#[test]
fn fallback_tier_drives_mage_spec_receipt_floors_and_provider_scope() {
    // requested q4 -> resolved q8/bf16, and requested q8/bf16 -> resolved q4.
    for (requested_bits, resolved_tier, expected_quant, expected_bits) in [
        (4, "q8", Some(Quant::Q8), Some(8)),
        (4, "bf16", None, None),
        (8, "q4", Some(Quant::Q4), Some(4)),
        (0, "q4", Some(Quant::Q4), Some(4)),
    ] {
        let temp = tempfile::tempdir().expect("temp tier root");
        seed_only_tier(temp.path(), resolved_tier);
        let request = request(requested_bits);
        let resolved = standard_tier_subdir_gated(temp.path(), &request, false);
        let tier = candle_resolved_tier_key(&request, &resolved, false);
        assert_eq!(tier, resolved_tier);
        let (resolved_quant, resolved_bits) =
            candle_quant_for_resolved_tier(&request, tier, &resolved, true, false);
        assert_eq!(
            (resolved_quant, resolved_bits),
            (expected_quant, expected_bits)
        );

        // This is the pre-fix request-derived value: intentionally retain it only to prove the
        // provider rejects a stale LoadSpec against the resolved receipt.
        let stale_requested_quant = resolve_quant(&request, Some(&resolved)).0;
        assert_ne!(stale_requested_quant, resolved_quant);

        for &engine_id in MAGE_IDS {
            let spec = mage_spec(engine_id, &resolved, resolved_quant);
            assert_eq!(
                spec.quantize, expected_quant,
                "engine={engine_id} tier={tier}"
            );
            let edit = engine_id.contains("_edit");
            let evaluation = crate::candle_memory_strategy::evaluate_shared_image(
                engine_id,
                engine_id,
                &spec,
                false,
                &JsonObject::new(),
                tier,
                if edit {
                    "edit_image"
                } else {
                    "image_generation"
                },
                None,
                gen_core::MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: u32::from(edit),
                },
                edit,
                false,
                false,
                false,
                Some(crate::vram_gate::VramBudget {
                    free_gb: 64.0,
                    total_gb: 64.0,
                }),
                crate::vram_gate::ladder_reserve_gb(crate::vram_gate::VramBudget {
                    free_gb: 64.0,
                    total_gb: 64.0,
                }),
                Some(1.0),
                0,
                gen_core::MemoryCacheState::Cold,
            )
            .expect("shared Mage evaluation")
            .expect("resident Mage selection");
            assert_eq!(
                evaluation.context.selection.tier.quant, expected_quant,
                "engine={engine_id} tier={tier}"
            );
            let declared = crate::inference_runtime::media_descriptor(engine_id)
                .expect("Mage descriptor")
                .capabilities
                .component_precision_floors;
            if tier == "q4" {
                assert_eq!(
                    evaluation.context.selection.tier.component_precision_floors, declared,
                    "engine={engine_id}"
                );
            } else {
                assert!(
                    evaluation
                        .context
                        .selection
                        .tier
                        .component_precision_floors
                        .is_empty(),
                    "engine={engine_id} tier={tier}"
                );
            }

            let registry = crate::inference_runtime::media();
            let registration = registry
                .memory_strategy_registrations()
                .find(|registration| registration.provider_id == engine_id)
                .unwrap_or_else(|| panic!("missing Mage memory registration {engine_id}"));
            let behavior = registry
                .memory_behavior_registrations()
                .find(|registration| registration.provider_id == engine_id)
                .unwrap_or_else(|| panic!("missing Mage memory behavior {engine_id}"));
            let contract = (registration.contract)(&spec).expect("Mage memory contract");
            assert_eq!(
                (registration.safety_check)(&spec, &contract, &evaluation.context),
                gen_core::MemorySafetyDecision::Accept,
                "engine={engine_id} tier={tier}"
            );
            assert!(
                (behavior.begin_request)(&spec, &contract, &evaluation.context)
                    .expect("matched Mage begin_request")
                    .is_some(),
                "engine={engine_id} tier={tier}"
            );

            let stale_spec = mage_spec(engine_id, &resolved, stale_requested_quant);
            assert!(
                matches!(
                    (registration.safety_check)(&stale_spec, &contract, &evaluation.context),
                    gen_core::MemorySafetyDecision::Reject { .. }
                ),
                "engine={engine_id} requested={requested_bits} resolved={tier}"
            );
            assert!(
                (behavior.begin_request)(&stale_spec, &contract, &evaluation.context).is_err(),
                "engine={engine_id} requested={requested_bits} resolved={tier}"
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
    ImageRequest::from_payload(payload.as_object().expect("request object"))
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
        assert!(
            candle_flux1_packed_requested_tier(&flux1_request("flux_dev", Some(unsupported)))
                .is_err(),
            "bf16, q6, and malformed FLUX.1 selections must fail closed"
        );
    }
}
