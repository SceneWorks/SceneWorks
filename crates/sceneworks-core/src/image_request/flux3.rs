//! The product-side ordered reference contract for FLUX.3 Klein.
//!
//! Asset ids remain opaque here. Only the project store may turn them into local paths; decoded
//! geometry and memory facts are computed by the worker, never accepted from this payload.

use crate::contracts::JsonObject;
use serde_json::Value;

pub const MAX_REFERENCES: usize = 10;

pub fn is_model(model: &str) -> bool {
    matches!(model, "flux3_klein" | "flux3_klein_base")
}

/// Authored size intent, retained before API/worker numeric defaults are applied. Missing sides
/// inherit the first prepared reference; neither side means the source's automatic output crop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceRequest {
    /// Source first, then the plural list in submitted order, then the singular reference. Repeated
    /// asset ids are separate conditioning slots, not aliases to eliminate.
    pub asset_ids: Vec<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Lossless authored box document; never replace with rounded conditioning rows.
    pub layout: Option<Value>,
    /// Unique attachment-instance identities, aligned with asset_ids (duplicates remain slots).
    pub reference_ids: Option<Vec<String>>,
}

pub fn reference_ids(payload: &JsonObject) -> Result<Vec<String>, String> {
    fn scalar(payload: &JsonObject, key: &str) -> Result<Option<String>, String> {
        match payload.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(id)) if !id.trim().is_empty() => Ok(Some(id.trim().to_owned())),
            _ => Err(format!(
                "FLUX.3 `{key}` must be a non-blank owned image asset id."
            )),
        }
    }
    if payload
        .get("maskAssetId")
        .is_some_and(|value| !value.is_null())
    {
        return Err("FLUX.3 reference editing has no mask tensor. Remove maskAssetId; include an annotated image as an ordinary reference instead.".to_owned());
    }
    let mut ids = Vec::new();
    if let Some(id) = scalar(payload, "sourceAssetId")? {
        ids.push(id);
    }
    match payload.get("referenceAssetIds") {
        None | Some(Value::Null) => {}
        Some(Value::Array(values)) => {
            for value in values {
                let id = value.as_str().filter(|id| !id.trim().is_empty()).ok_or_else(|| {
                    "FLUX.3 referenceAssetIds must contain only non-blank owned image asset ids.".to_owned()
                })?;
                ids.push(id.trim().to_owned());
            }
        }
        _ => {
            return Err(
                "FLUX.3 referenceAssetIds must be an ordered array of owned image asset ids."
                    .to_owned(),
            )
        }
    }
    if let Some(id) = scalar(payload, "referenceAssetId")? {
        ids.push(id);
    }
    if ids.len() > MAX_REFERENCES {
        return Err(format!("FLUX.3 accepts 1–{MAX_REFERENCES} ordered references, including duplicate slots; this request supplies {}.", ids.len()));
    }
    Ok(ids)
}

pub fn parse(payload: &JsonObject) -> Result<ReferenceRequest, String> {
    let asset_ids = reference_ids(payload)?;
    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("text_to_image");
    if !matches!(
        mode,
        "text_to_image" | "image_generation" | "edit_image" | "character_image" | "image_to_image"
    ) {
        return Err(format!("FLUX.3 reference generation does not support mode `{mode}`; use text_to_image or edit_image."));
    }
    if asset_ids.is_empty() && matches!(mode, "edit_image" | "character_image" | "image_to_image") {
        return Err("FLUX.3 editing requires 1–10 owned reference image assets. Add a reference or use text_to_image.".to_owned());
    }
    // Conditioning is not img2img initialization. Even an explicit 1.0 is unsupported, matching
    // the native request validator rather than silently accepting an ineffective control.
    let advanced = payload.get("advanced").and_then(Value::as_object);
    for key in [
        "strength",
        "referenceStrength",
        "denoisingStrength",
        "imageStrength",
        "ip_adapter_scale",
        "ipAdapterScale",
    ] {
        if payload.get(key).is_some_and(|value| !value.is_null())
            || advanced
                .and_then(|advanced| advanced.get(key))
                .is_some_and(|value| !value.is_null())
        {
            return Err(format!("FLUX.3 references are conditioning, not img2img noise initialization; `{key}` is unsupported, including 1.0. Remove this control."));
        }
    }
    // The 64px single-reference padding floor is not an output kernel floor. Retain legacy
    // no-reference bounds, but allow every positive token-grid side for reference conditioning.
    let min_output_side = if asset_ids.is_empty() { 64 } else { 16 };
    let side = |key: &str| -> Result<Option<u32>, String> {
        match payload.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value.as_u64().and_then(|value| u32::try_from(value).ok())
                .filter(|value| (min_output_side..=4096).contains(value) && value.is_multiple_of(16))
                .map(Some)
                .ok_or_else(|| format!("FLUX.3 explicit {key} must be a multiple of 16 in {min_output_side}..=4096; omit it for reference-derived sizing.")),
        }
    };
    let reference_ids = match advanced.and_then(|a| a.get("flux3_reference_ids")) {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) => {
            let ids = values
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .ok_or_else(|| {
                            "FLUX.3 reference identities must be nonempty text.".to_owned()
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ids.len() != asset_ids.len()
                || ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len()
            {
                return Err("FLUX.3 reference identities must be unique and match every ordered attachment slot.".into());
            }
            Some(ids)
        }
        _ => return Err("FLUX.3 reference identities must be an ordered list.".into()),
    };
    let layout = advanced
        .and_then(|a| a.get("flux3_layout"))
        .filter(|v| !v.is_null())
        .cloned();
    if let Some(layout) = &layout {
        validate_layout(layout, asset_ids.len(), reference_ids.as_deref())?;
    }
    Ok(ReferenceRequest {
        asset_ids,
        width: side("width")?,
        height: side("height")?,
        layout,
        reference_ids,
    })
}

/// Syntax/attachment validation at HTTP ingress, before queueing or model load. Native code
/// revalidates and performs geometry conversion with worker-owned decoded dimensions.
pub(crate) fn validate_layout(
    layout: &Value,
    count: usize,
    reference_ids: Option<&[String]>,
) -> Result<(), String> {
    let invalid = || {
        "FLUX.3 layout needs version 1 and labeled positive YXYX boxes; edits need an attached source and a target box or explicit null removal.".to_owned()
    };
    let doc = layout.as_object().ok_or_else(invalid)?;
    if doc.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(invalid());
    }
    let boxes = doc
        .get("boxes")
        .and_then(Value::as_array)
        .ok_or_else(invalid)?;
    let mut ids = std::collections::HashSet::new();
    for value in boxes {
        let row = value.as_object().ok_or_else(invalid)?;
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        if !ids.insert(id) || row.get("desc").is_some_and(|v| !v.is_string()) {
            return Err(invalid());
        }
        for key in ["units", "bbox_units", "src_units", "tgt_units"] {
            if row.contains_key(key) {
                return Err(format!(
                    "FLUX.3 layout `{key}` is unsupported; use an explicit coordinate space."
                ));
            }
        }
        let mut present = [false; 3];
        for (i, key) in ["bbox", "src_bbox", "tgt_bbox"].iter().enumerate() {
            if let Some(v) = row.get(*key).filter(|v| !v.is_null()) {
                let coords = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
                let coords = coords
                    .iter()
                    .map(|v| v.as_f64().filter(|v| v.is_finite()).ok_or_else(invalid))
                    .collect::<Result<Vec<_>, _>>()?;
                if coords[2] <= coords[0] || coords[3] <= coords[1] {
                    return Err(invalid());
                }
                present[i] = true;
            }
        }
        for (i, key) in ["bbox_space", "src_space", "tgt_space"].iter().enumerate() {
            if let Some(v) = row.get(*key) {
                if !present[i] {
                    return Err(invalid());
                }
                let allowed: &[&str] = if i == 1 {
                    &[
                        "normalized",
                        "original",
                        "oriented",
                        "fitted",
                        "reference_canvas",
                    ]
                } else {
                    &["normalized", "output", "output_canvas"]
                };
                if !v.as_str().is_some_and(|v| allowed.contains(&v)) {
                    return Err(invalid());
                }
            }
        }
        if present[0] {
            if ["src_bbox", "tgt_bbox", "from", "reference_id"]
                .iter()
                .any(|k| row.contains_key(*k))
            {
                return Err(invalid());
            }
            continue;
        }
        let origin = row
            .get("from")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        if origin == "target" {
            if present[1] || !present[2] || row.contains_key("reference_id") {
                return Err(invalid());
            }
        } else {
            let index = origin
                .strip_prefix("ref_image_")
                .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|n| n.parse::<usize>().ok())
                .ok_or_else(invalid)?;
            if !present[1] {
                return Err(invalid());
            }
            if let Some(identity) = row.get("reference_id") {
                let identity = identity
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(invalid)?;
                if !reference_ids.is_some_and(|ids| ids.iter().any(|id| id == identity)) {
                    return Err("FLUX.3 layout selected reference identity is not attached; restore its attachment or remove the edit.".into());
                }
            } else if index >= count {
                return Err("FLUX.3 layout references a missing attachment slot.".into());
            }
            if row.get("src_space").and_then(Value::as_str) == Some("original") {
                return Err("FLUX.3 product references use EXIF-oriented pixels; use oriented or normalized coordinates, not original-file pixels.".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn layout_keeps_authoring_precision_nulls_and_duplicate_attachment_instances() {
        let layout = json!({"version":1,"boxes":[
            {"id":"move","from":"ref_image_0","reference_id":"slot-b","src_bbox":[0.123456789012345,0.2,0.7,0.8],"tgt_bbox":[0.2,0.3,0.8,0.9]},
            {"id":"remove","from":"ref_image_1","reference_id":"slot-a","src_bbox":[0,0,1,1],"tgt_bbox":null}
        ]});
        let payload = object(
            json!({"referenceAssetIds":["same","same"],"advanced":{"flux3_layout":layout,"flux3_reference_ids":["slot-a","slot-b"]}}),
        );
        let parsed = parse(&payload).unwrap();
        assert_eq!(parsed.layout, Some(layout));
        assert_eq!(parsed.reference_ids.unwrap(), ["slot-a", "slot-b"]);
        for advanced in [
            json!({"flux3_reference_ids":["slot-a","slot-a"]}),
            json!({"flux3_reference_ids":["slot-a"]}),
            json!({"flux3_layout":{"version":1,"boxes":[{"id":"bad","bbox":[0,false,1,1]}]}}),
            json!({"flux3_layout":{"version":1,"boxes":[{"id":"bad","from":"ref_image_2","src_bbox":[0,0,1,1],"tgt_bbox":null}]}}),
            json!({"flux3_layout":{"version":1,"boxes":[{"id":"bad","from":"ref_image_0","reference_id":"missing","src_bbox":[0,0,1,1],"tgt_bbox":null}]},"flux3_reference_ids":["slot-a","slot-b"]}),
        ] {
            assert!(parse(&object(
                json!({"referenceAssetIds":["same","same"],"advanced":advanced})
            ))
            .is_err());
        }
    }

    #[test]
    fn order_and_duplicate_slots_are_not_qwen_deduplication() {
        let payload = object(
            json!({"sourceAssetId":"a", "referenceAssetIds":["b","a"], "referenceAssetId":"b"}),
        );
        assert_eq!(parse(&payload).unwrap().asset_ids, ["a", "b", "a", "b"]);
        let ten = object(json!({"referenceAssetIds":vec!["same";10]}));
        assert_eq!(parse(&ten).unwrap().asset_ids.len(), 10);
        assert!(parse(&object(json!({"referenceAssetIds":vec!["same";11]}))).is_err());
    }

    #[test]
    fn absent_one_and_both_explicit_sides_keep_their_intent() {
        for (fields, expected) in [
            (json!({"referenceAssetId":"a"}), (None, None)),
            (
                json!({"referenceAssetId":"a","width":48,"height":64}),
                (Some(48), Some(64)),
            ),
            (
                json!({"referenceAssetId":"a","width":16,"height":64}),
                (Some(16), Some(64)),
            ),
            (
                json!({"referenceAssetId":"a","width":1024}),
                (Some(1024), None),
            ),
            (
                json!({"referenceAssetId":"a","height":64}),
                (None, Some(64)),
            ),
            (
                json!({"referenceAssetId":"a","width":1024,"height":64}),
                (Some(1024), Some(64)),
            ),
        ] {
            let request = parse(&object(fields)).unwrap();
            assert_eq!((request.width, request.height), expected);
        }
    }

    #[test]
    fn invalid_references_strength_and_zero_sentinels_fail_closed() {
        for value in [
            json!({"mode":"edit_image"}),
            json!({"referenceAssetId":""}),
            json!({"referenceAssetIds":["a",null]}),
            json!({"referenceAssetIds":"a"}),
            json!({"referenceAssetId":"a","maskAssetId":"mask"}),
            json!({"referenceAssetId":"a","strength":1.0}),
            json!({"referenceAssetId":"a","advanced":{"referenceStrength":1.0}}),
            json!({"referenceAssetId":"a","advanced":{"denoisingStrength":0.5}}),
            json!({"referenceAssetId":"a","advanced":{"ipAdapterScale":1.0}}),
            json!({"referenceAssetId":"a","width":0}),
            json!({"referenceAssetId":"a","height":65}),
            json!({"mode":"text_to_image","width":48,"height":64}),
        ] {
            assert!(parse(&object(value.clone())).is_err(), "{value}");
        }
        assert!(parse(&object(json!({"mode":"text_to_image"})))
            .unwrap()
            .asset_ids
            .is_empty());
    }
}
