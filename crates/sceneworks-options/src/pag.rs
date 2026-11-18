//! Portable Perturbed-Attention Guidance (`advanced.pag`) parsing, capability-driven validation,
//! and execution receipt (`advanced.pagExecutionReport`) wire helpers (SW-7180).

use std::borrow::Cow;

use gen_core_pag_policy::{
    PagCapability, PagConfig, PagExecutionReport, PagPolicyError, PagResolved,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::contracts::JsonObject;
use crate::request_validation::ValidationRefusal;

pub const PAG_KEY: &str = "pag";
pub const PAG_REPORT_KEY: &str = "pagExecutionReport";
pub const PAG_REPORT_SNAKE_KEY: &str = "pag_execution_report";

fn clean_f32_as_f64(value: f32) -> f64 {
    value
        .to_string()
        .parse::<f64>()
        .unwrap_or_else(|_| f64::from(value))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PagWireRaw {
    scale: f64,
    #[serde(default)]
    layers: Option<Vec<u32>>,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    stop: Option<f64>,
    #[serde(default, rename = "adaptiveScale")]
    adaptive_scale_camel: Option<f64>,
    #[serde(default, rename = "adaptive_scale")]
    adaptive_scale_snake: Option<f64>,
}

/// Authored `advanced.pag` wire object (`camelCase`, accepting `adaptive_scale` on read and
/// failing closed on unknown keys or conflicting alias values).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PagWireRaw")]
pub struct PagWireConfig {
    pub scale: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<f64>,
    #[serde(rename = "adaptiveScale", skip_serializing_if = "Option::is_none")]
    pub adaptive_scale: Option<f64>,
}

impl TryFrom<PagWireRaw> for PagWireConfig {
    type Error = &'static str;

    fn try_from(raw: PagWireRaw) -> Result<Self, Self::Error> {
        let adaptive_scale = match (raw.adaptive_scale_camel, raw.adaptive_scale_snake) {
            (Some(a), Some(b)) if a != b => {
                return Err("conflicting adaptiveScale and adaptive_scale values")
            }
            (Some(a), _) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Ok(Self {
            scale: raw.scale,
            layers: raw.layers,
            start: raw.start,
            stop: raw.stop,
            adaptive_scale,
        })
    }
}

impl PagWireConfig {
    pub fn to_policy_config(&self) -> Result<PagConfig, ValidationRefusal> {
        let finite_f32 = |field: &str, value: f64| -> Result<f32, ValidationRefusal> {
            let narrowed = value as f32;
            if value.is_finite() && narrowed.is_finite() {
                Ok(narrowed)
            } else {
                Err(ValidationRefusal::new(
                    field,
                    format!("{field} must be a finite number (got {value})"),
                ))
            }
        };
        let scale = finite_f32("advanced.pag.scale", self.scale)?;
        let start = self
            .start
            .map(|v| finite_f32("advanced.pag.start", v))
            .transpose()?;
        let stop = self
            .stop
            .map(|v| finite_f32("advanced.pag.stop", v))
            .transpose()?;
        let adaptive_scale = self
            .adaptive_scale
            .map(|v| finite_f32("advanced.pag.adaptiveScale", v))
            .transpose()?;
        Ok(PagConfig {
            scale,
            layers: self.layers.clone(),
            start,
            stop,
            adaptive_scale,
        })
    }

    pub fn from_policy_config(cfg: &PagConfig) -> Self {
        Self {
            scale: clean_f32_as_f64(cfg.scale),
            layers: cfg.layers.clone(),
            start: cfg.start.map(clean_f32_as_f64),
            stop: cfg.stop.map(clean_f32_as_f64),
            adaptive_scale: cfg.adaptive_scale.map(clean_f32_as_f64),
        }
    }
}

/// Projected `model["pag"]` capability metadata from `ModelSpec`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PagCapabilityWire {
    #[serde(rename = "numBlocks", alias = "num_blocks")]
    num_blocks: u32,
    #[serde(rename = "blockDomain", alias = "block_domain")]
    block_domain: String,
    #[serde(rename = "defaultLayers", alias = "default_layers")]
    default_layers: Vec<u32>,
    #[serde(rename = "defaultStart", alias = "default_start")]
    default_start: f32,
    #[serde(rename = "defaultStop", alias = "default_stop")]
    default_stop: f32,
    #[serde(rename = "recommendedScale", alias = "recommended_scale")]
    recommended_scale: f32,
    #[serde(rename = "maxScale", alias = "max_scale")]
    max_scale: f32,
}

/// Completed provider PAG execution receipt (`rawAdapterSettings["pagExecutionReport"]`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagExecutionReportWire {
    pub scale: f64,
    #[serde(rename = "adaptiveScale", alias = "adaptive_scale")]
    pub adaptive_scale: f64,
    pub layers: Vec<u32>,
    pub start: f64,
    pub stop: f64,
    #[serde(rename = "activeSteps", alias = "active_steps")]
    pub active_steps: u32,
    #[serde(rename = "totalSteps", alias = "total_steps")]
    pub total_steps: u32,
}

impl From<&PagExecutionReport> for PagExecutionReportWire {
    fn from(report: &PagExecutionReport) -> Self {
        Self {
            scale: clean_f32_as_f64(report.scale),
            adaptive_scale: clean_f32_as_f64(report.adaptive_scale),
            layers: report.layers.clone(),
            start: clean_f32_as_f64(report.start),
            stop: clean_f32_as_f64(report.stop),
            active_steps: report.active_steps,
            total_steps: report.total_steps,
        }
    }
}

/// Parse `entry["pag"]` into a validated [`PagCapability`].
pub fn capability_from_entry(
    entry: &JsonObject,
) -> Result<Option<PagCapability>, ValidationRefusal> {
    let Some(raw) = entry.get(PAG_KEY).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let wire: PagCapabilityWire = serde_json::from_value(raw.clone()).map_err(|error| {
        ValidationRefusal::new(
            "advanced.pag",
            format!("model PAG capability metadata is malformed: {error}"),
        )
    })?;
    let capability = PagCapability {
        num_blocks: wire.num_blocks,
        block_domain: Cow::Owned(wire.block_domain),
        default_layers: wire.default_layers,
        default_start: wire.default_start,
        default_stop: wire.default_stop,
        recommended_scale: wire.recommended_scale,
        max_scale: wire.max_scale,
    };
    capability.validate_definition().map_err(map_policy_error)?;
    Ok(Some(capability))
}

fn normalize_serde_path(raw_path: &str, detail: &str) -> String {
    if detail.contains("conflicting adaptiveScale and adaptive_scale") {
        return "advanced.pag.adaptiveScale".to_owned();
    }
    let trimmed = raw_path.trim_matches('.');
    if trimmed.is_empty() {
        return "advanced.pag".to_owned();
    }
    let head = trimmed.split(['.', '[']).next().unwrap_or_default();
    match head {
        "scale" => "advanced.pag.scale".to_owned(),
        "layers" => "advanced.pag.layers".to_owned(),
        "start" => "advanced.pag.start".to_owned(),
        "stop" => "advanced.pag.stop".to_owned(),
        "adaptiveScale" | "adaptive_scale" => "advanced.pag.adaptiveScale".to_owned(),
        _ => "advanced.pag".to_owned(),
    }
}

pub fn map_policy_error(error: PagPolicyError) -> ValidationRefusal {
    let detail = error.message().to_owned();
    let field = if detail.contains("pag.layers") {
        "advanced.pag.layers"
    } else if detail.contains("pag.scale") {
        "advanced.pag.scale"
    } else if detail.contains("pag.adaptive_scale") {
        "advanced.pag.adaptiveScale"
    } else if detail.contains("pag.start") {
        "advanced.pag.start"
    } else if detail.contains("pag.stop") {
        "advanced.pag.stop"
    } else {
        "advanced.pag"
    };
    ValidationRefusal::new(field, detail)
}

/// Strictly decode a raw JSON value as [`PagWireConfig`], rejecting unknown fields, non-integer or
/// negative layer indices, and non-numeric values with exact subfield paths.
pub fn parse_pag_wire_value(value: &Value) -> Result<PagWireConfig, ValidationRefusal> {
    if !value.is_object() {
        return Err(ValidationRefusal::new(
            "advanced.pag",
            format!("advanced.pag must be an object, got {value}"),
        ));
    }
    serde_path_to_error::deserialize::<_, PagWireConfig>(value.clone()).map_err(|error| {
        let inner = error.inner().to_string();
        let field = normalize_serde_path(&error.path().to_string(), &inner);
        ValidationRefusal::new(field, format!("advanced.pag: {inner}"))
    })
}

/// Capability-independent validation of a standalone PAG block (e.g. a general preset).
pub fn validate_standalone_pag_value(value: &Value) -> Result<PagConfig, ValidationRefusal> {
    let wire = parse_pag_wire_value(value)?;
    let cfg = wire.to_policy_config()?;
    cfg.validate().map_err(map_policy_error)?;
    Ok(cfg)
}

/// Validate `value` against an explicit [`PagCapability`].
pub fn validate_pag_value_with_capability(
    value: &Value,
    capability: &PagCapability,
) -> Result<(PagConfig, PagResolved), ValidationRefusal> {
    let wire = parse_pag_wire_value(value)?;
    let cfg = wire.to_policy_config()?;
    let resolved = capability.resolve(Some(&cfg)).map_err(map_policy_error)?;
    Ok((cfg, resolved))
}

/// Validate `value` against `entry["pag"]` when `entry` belongs to `model`.
pub fn validate_pag_value_for_model(
    model: &str,
    value: &Value,
    entry: &JsonObject,
) -> Result<(PagConfig, PagResolved), ValidationRefusal> {
    let Some(capability) = capability_from_entry(entry)? else {
        return Err(ValidationRefusal::new(
            "advanced.pag",
            format!("{model} does not support Perturbed-Attention Guidance (advanced.pag)"),
        ));
    };
    validate_pag_value_with_capability(value, &capability)
}

/// Validate `advanced.pag` on a request payload against `entry["pag"]`.
///
/// Returns `Ok(None)` when `advanced.pag` is absent or `null`.
pub fn requested_pag(
    model: &str,
    advanced: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<PagConfig>, ValidationRefusal> {
    let Some(raw) = advanced.get(PAG_KEY).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let (cfg, _) = validate_pag_value_for_model(model, raw, entry)?;
    Ok(Some(cfg))
}

/// Sanitize `advanced.pag` for workflow-share envelopes: drops malformed or unknown-field blocks,
/// and canonicalizes `adaptive_scale` -> `adaptiveScale`.
pub fn sanitize_share_pag(value: &Value) -> Option<Value> {
    let wire = parse_pag_wire_value(value).ok()?;
    let cfg = wire.to_policy_config().ok()?;
    cfg.validate().ok()?;
    serde_json::to_value(&wire).ok()
}

/// Sanitize `advanced.pagExecutionReport` for workflow-share envelopes.
pub fn sanitize_share_pag_report(value: &Value) -> Option<Value> {
    let mut report: PagExecutionReportWire = serde_json::from_value(value.clone()).ok()?;
    if !report.scale.is_finite()
        || !(report.scale as f32).is_finite()
        || report.scale <= 0.0
        || !report.adaptive_scale.is_finite()
        || !(report.adaptive_scale as f32).is_finite()
        || report.adaptive_scale < 0.0
        || !report.start.is_finite()
        || !report.stop.is_finite()
        || report.start < 0.0
        || report.start > report.stop
        || report.stop > 1.0
        || report.layers.is_empty()
        || report.layers.len() > 512
        || report.total_steps == 0
        || report.active_steps > report.total_steps
    {
        return None;
    }
    report.layers.sort_unstable();
    report.layers.dedup();
    serde_json::to_value(&report).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry_with_pag(num_blocks: u32, default_layers: &[u32], max_scale: f64) -> JsonObject {
        json!({
            "type": "image",
            "capabilities": ["text_to_image", "edit_image"],
            "pag": {
                "numBlocks": num_blocks,
                "blockDomain": "joint_blocks",
                "defaultLayers": default_layers,
                "defaultStart": 0.05,
                "defaultStop": 0.85,
                "recommendedScale": 1.5,
                "maxScale": max_scale
            }
        })
        .as_object()
        .cloned()
        .unwrap()
    }

    #[test]
    fn validates_bounds_windows_scales_aliases_and_unknown_fields() {
        let entry = entry_with_pag(8, &[2, 5], 10.0);

        // Omitted or null -> None
        assert_eq!(
            requested_pag("flux2_dev", &JsonObject::new(), &entry).unwrap(),
            None
        );
        assert_eq!(
            requested_pag(
                "flux2_dev",
                json!({ "pag": null }).as_object().unwrap(),
                &entry
            )
            .unwrap(),
            None
        );

        // Valid defaults + snake_case alias read -> canonicalized wire
        let (cfg, resolved) = validate_pag_value_for_model(
            "flux2_dev",
            &json!({ "scale": 1.5, "adaptive_scale": 0.25 }),
            &entry,
        )
        .unwrap();
        assert_eq!(cfg.scale, 1.5);
        assert_eq!(cfg.adaptive_scale, Some(0.25));
        assert_eq!(resolved.layers, vec![2, 5]);
        assert!((resolved.start - 0.05).abs() < 1e-6);
        assert!((resolved.stop - 0.85).abs() < 1e-6);
        assert_eq!(
            sanitize_share_pag(&json!({ "scale": 1.5, "adaptive_scale": 0.25 })),
            Some(json!({ "scale": 1.5, "adaptiveScale": 0.25 }))
        );

        // Exact boundary 0..numBlocks-1
        let (_, boundary_ok) = validate_pag_value_for_model(
            "flux2_dev",
            &json!({ "scale": 2.0, "layers": [7, 0, 7], "start": 0.0, "stop": 1.0 }),
            &entry,
        )
        .unwrap();
        assert_eq!(boundary_ok.layers, vec![0, 7]);

        // Out-of-range block index
        let out_of_range = validate_pag_value_for_model(
            "flux2_dev",
            &json!({ "scale": 2.0, "layers": [8] }),
            &entry,
        )
        .unwrap_err();
        assert_eq!(out_of_range.field, "advanced.pag.layers");

        // Empty layers with scale > 0 rejected; scale == 0.0 is inert no-op
        let empty_active = validate_pag_value_for_model(
            "flux2_dev",
            &json!({ "scale": 1.0, "layers": [] }),
            &entry,
        )
        .unwrap_err();
        assert_eq!(empty_active.field, "advanced.pag.layers");

        let (_, inert) =
            validate_pag_value_for_model("flux2_dev", &json!({ "scale": 0.0 }), &entry).unwrap();
        assert!(!inert.is_active());
        assert_eq!(inert.active_steps(28), 0);

        // Window and scale violations
        for (bad, expected_field) in [
            (json!({ "scale": -0.1 }), "advanced.pag.scale"),
            (json!({ "scale": 10.5 }), "advanced.pag.scale"),
            (json!({ "scale": 1.5, "start": -0.1 }), "advanced.pag.start"),
            (json!({ "scale": 1.5, "stop": 1.1 }), "advanced.pag.stop"),
            (
                json!({ "scale": 1.5, "start": 0.8, "stop": 0.2 }),
                "advanced.pag",
            ),
            (
                json!({ "scale": 1.5, "adaptiveScale": 0.2, "adaptive_scale": 0.4 }),
                "advanced.pag.adaptiveScale",
            ),
            (json!({ "scale": 1.5, "typoField": 1 }), "advanced.pag"),
            (json!("not-an-object"), "advanced.pag"),
        ] {
            let err = validate_pag_value_for_model("flux2_dev", &bad, &entry).unwrap_err();
            assert_eq!(
                err.field, expected_field,
                "unexpected field for {bad}: {err:?}"
            );
        }
    }

    #[test]
    fn rejects_unsupported_models_and_accepts_minimax_h3_and_moss_combinations() {
        let catalog: Value = serde_json::from_str(include_str!(
            "../../../apps/web/src/testUtils/catalogBaseline/candidate-catalog.json"
        ))
        .unwrap();
        let entry_for = |id: &str| -> &JsonObject {
            catalog["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("missing catalog model {id}"))
                .as_object()
                .unwrap()
        };

        // Unsupported models without `pag` capability (e.g. realvisxl, iris_3b)
        for unsupported in ["realvisxl", "iris_3b"] {
            let entry = entry_for(unsupported);
            let err = requested_pag(
                unsupported,
                json!({ "pag": { "scale": 1.5 } }).as_object().unwrap(),
                entry,
            )
            .unwrap_err();
            assert_eq!(err.field, "advanced.pag");
            assert!(err
                .detail
                .contains(&format!("{unsupported} does not support")));
        }

        // Supra2-IMG explicit refusal
        let supra_entry = entry_for("supra2_img");
        let supra_err = crate::supra2::validate_request(
            json!({
                "projectId": "p",
                "model": "supra2_img",
                "prompt": "a fox",
                "advanced": { "pag": { "scale": 1.5 } }
            })
            .as_object()
            .unwrap(),
            supra_entry,
        )
        .unwrap_err();
        assert!(supra_err.contains("advanced.pag "), "{supra_err}");

        // MiniMax-H3-Image (`supportsGuidance: false`, `supportsNegativePrompt: false`) accepts PAG
        let h3_entry = entry_for("minimax_h3_image");
        let h3_payload = json!({
            "projectId": "p",
            "model": "minimax_h3_image",
            "mode": "text_to_image",
            "prompt": "a neon street in rain",
            "advanced": {
                "pag": { "scale": 1.5, "layers": [10, 49] }
            }
        });
        crate::request_validation::resolve("image", h3_payload.as_object().unwrap(), h3_entry)
            .expect("MiniMax-H3-Image accepts PAG despite supportsGuidance=false");

        // MOSS-Image (`moss_image_sft`) accepts PAG alongside SLG, DMD, and edit_image kv_cache
        let moss_entry = entry_for("moss_image_sft");
        for moss_payload in [
            json!({
                "projectId": "p",
                "model": "moss_image_sft",
                "mode": "text_to_image",
                "prompt": "editorial portrait",
                "advanced": {
                    "steps": 28,
                    "guidanceScale": 4.0,
                    "mossImage": {
                        "acceleration": "off",
                        "slg": { "layers": [7, 8], "scale": 1.5, "start": 0.1, "stop": 0.9 }
                    },
                    "pag": { "scale": 1.5, "layers": [20, 41] }
                }
            }),
            json!({
                "projectId": "p",
                "model": "moss_image_sft",
                "mode": "text_to_image",
                "prompt": "editorial portrait",
                "advanced": {
                    "steps": 4,
                    "mossImage": { "acceleration": "unmerged_bf16" },
                    "pag": { "scale": 1.25, "layers": [15] }
                }
            }),
            json!({
                "projectId": "p",
                "model": "moss_image_sft",
                "mode": "edit_image",
                "prompt": "turn into winter",
                "sourceAssetId": "ref-1",
                "advanced": {
                    "mossImageEdit": { "use_kv_cache": true },
                    "pag": { "scale": 1.5, "layers": [20] }
                }
            }),
            json!({
                "projectId": "p",
                "model": "moss_image_sft",
                "mode": "edit_image",
                "prompt": "turn into winter",
                "sourceAssetId": "ref-1",
                "advanced": {
                    "mossImageEdit": { "use_kv_cache": false },
                    "pag": { "scale": 1.5, "layers": [20] }
                }
            }),
        ] {
            crate::request_validation::resolve(
                "image",
                moss_payload.as_object().unwrap(),
                moss_entry,
            )
            .unwrap_or_else(|e| panic!("MOSS PAG combination failed for {moss_payload}: {e:?}"));
        }
    }

    #[test]
    fn sanitizes_pag_execution_report_strictly() {
        let valid = json!({
            "scale": 1.5,
            "adaptiveScale": 0.25,
            "layers": [5, 2, 5],
            "start": 0.05,
            "stop": 0.85,
            "activeSteps": 18,
            "totalSteps": 28
        });
        assert_eq!(
            sanitize_share_pag_report(&valid),
            Some(json!({
                "scale": 1.5,
                "adaptiveScale": 0.25,
                "layers": [2, 5],
                "start": 0.05,
                "stop": 0.85,
                "activeSteps": 18,
                "totalSteps": 28
            }))
        );
        for invalid in [
            json!({
                "scale": 0.0,
                "adaptiveScale": 0.0,
                "layers": [2],
                "start": 0.0,
                "stop": 1.0,
                "activeSteps": 0,
                "totalSteps": 28
            }),
            json!({
                "scale": 1.5,
                "adaptiveScale": 0.0,
                "layers": [],
                "start": 0.0,
                "stop": 1.0,
                "activeSteps": 10,
                "totalSteps": 28
            }),
            json!({
                "scale": 1.5,
                "adaptiveScale": 0.0,
                "layers": [2],
                "start": 0.0,
                "stop": 1.0,
                "activeSteps": 29,
                "totalSteps": 28
            }),
            json!({
                "scale": 1.5,
                "adaptiveScale": 0.0,
                "layers": [2],
                "start": 0.0,
                "stop": 1.0,
                "activeSteps": 10,
                "totalSteps": 28,
                "localPath": "/Users/secret"
            }),
        ] {
            assert_eq!(sanitize_share_pag_report(&invalid), None);
        }
    }
}
