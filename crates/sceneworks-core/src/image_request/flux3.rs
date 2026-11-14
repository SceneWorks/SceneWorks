//! Native typed adapter for the portable FLUX.3 reference policy.
use crate::contracts::JsonObject;
use serde_json::Value;

pub use sceneworks_options::image_request::flux3::*;

/// Execution boundary lowering from registered, resolved options.
pub fn parse_resolved(
    payload: &JsonObject,
    options: &super::ImageAdvanced,
) -> Result<ReferenceRequest, String> {
    parse_with_carriers(
        payload,
        |key| match key {
            "strength" => authored(&options.strength),
            "referenceStrength" => authored(&options.reference_strength),
            "denoisingStrength" => authored(&options.denoising_strength),
            "imageStrength" => authored(&options.image_strength),
            "ip_adapter_scale" => authored(&options.ip_adapter_scale_legacy),
            "ipAdapterScale" => authored(&options.ip_adapter_scale),
            _ => false,
        },
        || {
            Ok(options
                .flux3_reference_ids
                .as_ref()
                .map(|bound| bound.value.clone()))
        },
        || {
            options
                .flux3_layout
                .as_ref()
                .map(|bound| Value::Object(bound.value.clone()))
        },
    )
}

fn authored<T>(value: &Option<crate::resolved_options::Bound<T>>) -> bool {
    value.as_ref().is_some_and(|bound| {
        matches!(
            bound.origin,
            crate::resolved_options::Origin::Authored { .. }
        )
    })
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
