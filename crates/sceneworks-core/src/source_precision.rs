//! Generation precision owned by a checkpoint's verified source format.

use crate::checkpoint_weight_facts as codecs;
use crate::contracts::JsonObject;
use serde_json::{json, Value};

#[derive(Clone, Copy)]
pub struct SourcePrecision {
    format: &'static str,
    codec_id: &'static str,
    label: &'static str,
}

impl SourcePrecision {
    fn imported_codec(entry: &JsonObject) -> Option<&'static str> {
        if entry.get("catalogScope").and_then(Value::as_str) == Some("builtin")
            || !entry.get("importSourceShape").is_some_and(Value::is_string)
        {
            return None;
        }
        crate::jobs_store::imported_entry_source_codec(entry)
    }

    pub fn dense_import(entry: &JsonObject) -> Option<(&'static str, &'static str)> {
        match Self::imported_codec(entry)? {
            codecs::DENSE_BF16_CODEC_ID => Some(("bf16", "BF16")),
            codecs::DENSE_F16_CODEC_ID => Some(("f16", "FP16")),
            codecs::DENSE_F32_CODEC_ID => Some(("f32", "FP32")),
            _ => None,
        }
    }

    pub fn for_entry(entry: &JsonObject) -> Option<Self> {
        let codec_id = Self::imported_codec(entry)?;
        let (format, label) = match codec_id {
            codecs::FP8_E4M3_SCALAR_CODEC_ID => ("fp8_e4m3_scalar", "Scaled FP8 (E4M3)"),
            codecs::FP8_E4M3_ROWWISE_CODEC_ID => ("fp8_e4m3_rowwise", "Per-row FP8 (E4M3)"),
            codecs::FP8_E5M2_SCALAR_CODEC_ID => ("fp8_e5m2_scalar", "Scaled FP8 (E5M2)"),
            codecs::MXFP8_CODEC_ID => ("mxfp8", "MXFP8"),
            codecs::NVFP4_CODEC_ID => ("nvfp4", "NVFP4"),
            codecs::CONVROT_INT8_CODEC_ID => ("int8-convrot", "INT8-ConvRot"),
            codecs::INT8_PER_ROW_CODEC_ID => ("int8_tensorwise_per_row", "Per-row INT8"),
            codecs::W4A8_INT8_CODEC_ID => ("asym_w4a8_int8", "W4A8 INT8"),
            // Dense imports can still choose the provider's supported load-time conversion.
            _ => return None,
        };
        Some(Self {
            format,
            codec_id,
            label,
        })
    }

    pub fn project(entry: &mut JsonObject) {
        entry.remove("sourcePrecision");
        if let Some(precision) = Self::for_entry(entry) {
            entry.insert("sourcePrecision".to_owned(), precision.catalog_fact());
        }
    }

    pub fn format(self) -> &'static str {
        self.format
    }

    fn catalog_fact(self) -> Value {
        json!({ "format": self.format, "codecId": self.codec_id, "label": self.label })
    }
}

pub use sceneworks_options::source_precision::clear_advanced_precision_selectors;
pub use sceneworks_options::source_precision::clear_precision_selectors;

pub fn applies_to_job(job_type: &crate::contracts::JobType) -> bool {
    crate::request_validation::kind_for_job(job_type)
        .is_some_and(sceneworks_options::source_precision::applies_to)
}

/// New submissions validate all their selectors. A retry or duplicate validates its new patch
/// and then removes historical conversion controls from the merged request.
pub fn canonicalize_for_entry(
    payload: &mut JsonObject,
    entry: &JsonObject,
    replay_changes: Option<&JsonObject>,
) -> Result<(), crate::request_validation::ValidationRefusal> {
    if let Some(precision) = SourcePrecision::for_entry(entry) {
        let facts =
            JsonObject::from_iter([("sourcePrecision".to_owned(), precision.catalog_fact())]);
        sceneworks_options::source_precision::validate(replay_changes.unwrap_or(payload), &facts)?;
        clear_precision_selectors(payload);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(format: &str) -> JsonObject {
        json!({
            "id": "community-checkpoint", "type": "image", "catalogScope": "user",
            "importSourceShape": "transformer_file", "importQuantFormat": format,
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn shared_validator_checks_every_new_selector_and_clears_matching_ones() {
        for (classification, format) in [
            ("fp8_e4m3_scalar", "fp8_e4m3_scalar"),
            ("fp8_e4m3_rowwise", "fp8_e4m3_rowwise"),
            ("fp8_e5m2_scalar", "fp8_e5m2_scalar"),
            ("convrot_int8", "int8-convrot"),
            ("nvfp4", "nvfp4"),
        ] {
            let entry = fixture(classification);
            let authored = |options| {
                json!({
                    "projectId": "p", "model": "community-checkpoint", "prompt": "snow",
                    "modelManifestEntry": entry, "advanced": options,
                })
                .as_object()
                .unwrap()
                .clone()
            };
            for (options, field) in [
                (json!({"quantTier": "bf16"}), "advanced.quantTier"),
                (
                    json!({"quantTier": format, "quant": "q4"}),
                    "advanced.quant",
                ),
                (
                    json!({"quantTier": format, "mlxQuantize": 8}),
                    "advanced.mlxQuantize",
                ),
                (json!({"precision": "bf16"}), "advanced.precision"),
                (json!({"quantization": "q4"}), "advanced.quantization"),
            ] {
                let refusal =
                    crate::request_validation::resolve_request("image", &authored(options), &entry)
                        .unwrap_err();
                assert_eq!(refusal.field, field, "{format}");
            }
            let mut request = authored(json!({ "quantTier": format, "steps": 12 }));
            request.insert("variant".to_owned(), json!("q4"));
            assert_eq!(
                crate::request_validation::validate("image", &request, &entry)
                    .unwrap_err()
                    .field,
                "variant"
            );
            request.insert("variant".to_owned(), json!(format));
            let effective =
                crate::request_validation::resolve_request("image", &request, &entry).unwrap();
            let image =
                crate::image_request::ImageRequest::from_resolved(&request, effective.as_payload())
                    .unwrap();
            assert!(image.variant.is_none(), "{format}");
            assert!(image.resolved_quant_bits().is_none(), "{format}");
            assert!(image.advanced_text("quantTier").is_none(), "{format}");
        }
    }

    #[test]
    fn saved_precision_is_migrated_and_new_replay_changes_are_validated() {
        for format in ["fp8_e4m3_scalar", "convrot_int8", "nvfp4"] {
            let entry = fixture(format);
            let original = json!({
                "variant": "q8", "advanced": {"quantTier": "bf16", "mlxQuantize": 8, "steps": 12},
            });
            let mut request = original.as_object().unwrap().clone();
            canonicalize_for_entry(&mut request, &entry, Some(&JsonObject::new())).unwrap();
            assert_eq!(Value::Object(request), json!({"advanced": {"steps": 12}}));
            let mut request = original.as_object().unwrap().clone();
            let explicit = json!({ "advanced": { "mlxQuantize": 4 } });
            let err =
                canonicalize_for_entry(&mut request, &entry, explicit.as_object()).unwrap_err();
            assert_eq!(err.field, "advanced.mlxQuantize");
            assert_eq!(Value::Object(request), original);
        }
    }

    #[test]
    fn unknown_dense_and_builtin_entries_do_not_get_source_owned_precision() {
        for format in [
            "bf16",
            "native",
            "comfy_quant_packed",
            "scaled_fp8_companion",
        ] {
            assert!(
                SourcePrecision::for_entry(&fixture(format)).is_none(),
                "{format}"
            );
        }
        let mut builtin = fixture("nvfp4");
        builtin.insert("catalogScope".to_owned(), json!("builtin"));
        builtin.insert("sourcePrecision".to_owned(), json!({ "label": "fake" }));
        SourcePrecision::project(&mut builtin);
        assert!(!builtin.contains_key("sourcePrecision"));
    }
}
