//! Native MOSS adapter; portable options owns product and numerical request policy.
use crate::contracts::JsonObject;
#[cfg(test)]
use gen_core::moss_image::MossImageAcceleration;
use gen_core::{
    moss_image::{MossImageOptions, MossImageResolved, MossImageVariant},
    GenerationRequest,
};
pub use sceneworks_options::moss_validation::{
    edit_backend_eligible, edit_options, edit_options_schema, edit_reference_ids,
    edit_reference_slots, is_edit, is_moss, options_schema, EditOptions, EDIT_OPTIONS_KEY,
    EDIT_PURPOSE, EDIT_REPORT_KEY, OPTIONS_KEY, REPORT_KEY,
};
use serde_json::{json, Value};
pub fn options(advanced: &JsonObject) -> Result<MossImageOptions, String> {
    let value = sceneworks_options::moss_validation::options(advanced)?;
    serde_json::from_value(serde_json::to_value(value).expect("MOSS options"))
        .map_err(|error| error.to_string())
}
pub fn purpose(options: &MossImageOptions) -> Option<&'static str> {
    let portable = serde_json::from_value(serde_json::to_value(options).expect("MOSS options"))
        .expect("MOSS options ABI");
    sceneworks_options::moss_validation::purpose(&portable)
}
pub fn request_purpose(payload: &JsonObject, options: &MossImageOptions) -> Option<&'static str> {
    if is_edit(payload) {
        Some(EDIT_PURPOSE)
    } else {
        purpose(options)
    }
}
pub fn native_request(
    payload: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<GenerationRequest>, String> {
    sceneworks_options::moss_validation::native_request(payload, entry)?
        .map(|request| {
            let policy = request.policy;
            Ok(GenerationRequest {
                prompt: request.prompt,
                prompt_batch: request.prompt_batch,
                width: policy.width,
                height: policy.height,
                count: policy.count,
                steps: policy.steps,
                guidance: policy.guidance,
                sampler: policy.sampler,
                scheduler: policy.scheduler,
                scheduler_shift: policy.scheduler_shift,
                guidance_method: policy.guidance_method,
                negative_prompt: policy.negative_prompt,
                moss_image: policy
                    .moss_image
                    .map(|options| {
                        serde_json::from_value(serde_json::to_value(options).expect("MOSS options"))
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?,
                ..Default::default()
            })
        })
        .transpose()
}
/// Effective sampling controls persisted by every edit enqueue path, so history can replay
/// an omitted native default after the catalog or application defaults change.
pub fn edit_effective_controls(
    model: &str,
    request: &GenerationRequest,
) -> Result<JsonObject, String> {
    let variant = MossImageVariant::from_provider_id(model).ok_or("unknown MOSS model")?;
    let resolved =
        MossImageResolved::resolve(variant, request).map_err(|error| error.to_string())?;
    Ok(json!({"steps":resolved.steps,"guidanceScale":resolved.guidance,"schedulerShift":resolved.shift})
        .as_object().expect("controls object").clone())
}
/// Stage ONLY the selected optional package. Ordinary SFT remains usable without any DMD files.
pub fn selected_components(
    entry: &JsonObject,
    options: &MossImageOptions,
) -> Result<JsonObject, String> {
    selected_components_for_purpose(entry, purpose(options))
}
pub const EDIT_COMPONENT: &str = gen_core::moss_image_edit::EDIT_COMPONENT;

pub fn is_runtime_optional_component_id(id: &str) -> bool {
    id.starts_with("moss-dmd-") || id == EDIT_COMPONENT
}

pub fn selected_components_for_purpose(
    entry: &JsonObject,
    purpose: Option<&str>,
) -> Result<JsonObject, String> {
    let mut result = entry.clone();
    let downloads = result
        .entry("downloads".to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or("downloads must be an array")?;
    downloads.retain(|v| {
        !v.get("componentId")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("moss-dmd-") || id == "moss-image-edit")
    });
    if let Some(purpose) = purpose {
        let mut components = crate::model_usage_policy::conditional_component_downloads(
            &Value::Object(entry.clone()),
            purpose,
        )
        .map_err(|e| e.to_string())?;
        for row in &mut components {
            if let Some(source) = entry
                .get("conditionalComponents")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|source| source.get("componentId") == row.get("componentId"))
            {
                for key in ["license", "licenseBasis"] {
                    if let Some(value) = source.get(key) {
                        row.as_object_mut()
                            .expect("component row")
                            .insert(key.to_owned(), value.clone());
                    }
                }
            }
        }
        downloads.extend(components);
    }
    Ok(result)
}
/// Reports are exclusively worker-produced. Replay keeps requested options but cannot reuse outcomes.
pub fn scrub_reports(payload: &mut JsonObject) {
    payload.remove("mossImageSampleIndex");
    payload.remove(EDIT_REPORT_KEY);
    payload.remove("moss_image_edit_execution_report");
    payload.remove(REPORT_KEY);
    payload.remove("moss_image_execution_report");
    if let Some(advanced) = payload.get_mut("advanced").and_then(Value::as_object_mut) {
        advanced.remove("mossImageSampleIndex");
        advanced.remove(EDIT_REPORT_KEY);
        advanced.remove("moss_image_edit_execution_report");
        advanced.remove(REPORT_KEY);
        advanced.remove("moss_image_execution_report");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry() -> JsonObject {
        json!({"defaults":{"resolution":"1024x1024","count":1},"limits":{"minDimension":16,"maxDimension":4096},"conditionalComponents":[{"provider":"huggingface","repo":"dmd","revision":"pin","componentId":"moss-dmd-lora","requiredFor":["moss_dmd_unmerged_bf16"],"files":["lora.safetensors"]},{"provider":"huggingface","repo":"dmd","revision":"pin","componentId":"moss-dmd-transformer","requiredFor":["moss_dmd_full_fp32"],"files":["transformer/*"]}]}).as_object().unwrap().clone()
    }
    fn body(advanced: Value) -> JsonObject {
        json!({"model":"moss_image_sft","mode":"text_to_image","prompt":"日本語 <tag> {\"items\":[]}","advanced":advanced}).as_object().unwrap().clone()
    }
    #[test]
    fn explicit_edit_preserves_single_reference_and_author_defaults() {
        for model in ["moss_image_base", "moss_image_sft"] {
            let p = json!({"model":model,"mode":"edit_image","prompt":"change the cup to blue","sourceAssetId":"ref-a","referenceAssetIds":["ref-a"],"advanced":{"mossImageEdit":{"reference_max_edge":768,"use_kv_cache":false}}}).as_object().unwrap().clone();
            assert_eq!(edit_reference_ids(&p).unwrap(), ["ref-a"]);
            let request = native_request(&p, &entry()).unwrap().unwrap();
            assert!(request.conditioning.is_empty());
            assert_eq!((request.width, request.height), (512, 512));
            let resolved = MossImageResolved::resolve(
                MossImageVariant::from_provider_id(model).unwrap(),
                &request,
            )
            .unwrap();
            assert_eq!(resolved.steps, 30);
            assert_eq!(resolved.guidance, 4.0);
            assert_eq!(resolved.shift, 5.0);
            assert!(resolved.uncond_skip_layers.is_empty());
            assert_eq!(
                request_purpose(&p, request.moss_image.as_ref().unwrap()),
                Some(EDIT_PURPOSE)
            );
            let options = edit_options(p["advanced"].as_object().unwrap()).unwrap();
            assert_eq!(options.reference_max_edge, 768);
            assert!(!options.use_kv_cache);
        }
    }
    #[test]
    fn edit_reference_slots_match_native_bounds_without_deduplicating_or_reordering() {
        use gen_core::{moss_image_edit::MossImageEditInputs, Image};
        for model in ["moss_image_base", "moss_image_sft"] {
            for ids in [
                vec!["a"],
                vec!["a", "b"],
                vec!["a", "b", "a"],
                vec!["c", "a", "b", "a"],
            ] {
                for ordered in [ids.clone(), ids.iter().rev().copied().collect()] {
                    let payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup",
                        "referenceAssetIds":ordered,"sourceAssetId":ordered[0],"referenceAssetId":ordered[0]})
                        .as_object().unwrap().clone();
                    assert_eq!(edit_reference_ids(&payload).unwrap(), ordered);
                    assert!(native_request(&payload, &entry()).unwrap().is_some());
                    let references = ordered
                        .iter()
                        .map(|id| {
                            let (width, height, pixel) = match *id {
                                "a" => (48, 16, 73),
                                "b" => (16, 48, 127),
                                _ => (32, 32, 211),
                            };
                            Image {
                                width,
                                height,
                                pixels: vec![pixel; width as usize * height as usize * 3],
                            }
                        })
                        .collect::<Vec<_>>();
                    let expected = references
                        .iter()
                        .map(|image| [image.width, image.height])
                        .collect::<Vec<_>>();
                    assert_eq!(
                        MossImageEditInputs {
                            references: &references,
                            ..Default::default()
                        }
                        .validate()
                        .unwrap(),
                        expected
                    );
                    for mode in ["text_to_image", "edit_image"] {
                        let mut unsupported = payload.clone();
                        unsupported.insert("mode".into(), json!(mode));
                        if mode == "edit_image" {
                            unsupported.insert(
                                "advanced".into(),
                                json!({"mossImage":{"acceleration":"unmerged_bf16"}}),
                            );
                        }
                        assert!(
                            native_request(&unsupported, &entry()).is_err(),
                            "ordinary/DMD cannot absorb edit references"
                        );
                    }
                }
            }
            for slots in [Vec::new(), vec!["a"; 5]] {
                let payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup","referenceAssetIds":slots})
                    .as_object().unwrap().clone();
                assert!(edit_reference_ids(&payload).is_err());
                assert!(native_request(&payload, &entry()).is_err());
                let references = slots
                    .iter()
                    .map(|_| Image {
                        width: 16,
                        height: 16,
                        pixels: vec![0; 16 * 16 * 3],
                    })
                    .collect::<Vec<_>>();
                assert!(MossImageEditInputs {
                    references: &references,
                    ..Default::default()
                }
                .validate()
                .is_err());
            }
            for changes in [
                json!({"referenceAssetIds":["a",null]}),
                json!({"referenceAssetIds":["a",2]}),
                json!({"referenceAssetIds":["a"," b"]}),
                json!({"referenceAssetIds":["a",""]}),
                json!({"referenceAssetIds":"a"}),
                json!({"sourceAssetId":"b"}),
                json!({"referenceAssetId":"b"}),
            ] {
                let mut payload = json!({"model":model,"mode":"edit_image","prompt":"blue cup","referenceAssetIds":["a","b"]})
                    .as_object().unwrap().clone();
                payload.extend(changes.as_object().unwrap().clone());
                assert!(edit_reference_ids(&payload).is_err(), "{changes}");
                assert!(native_request(&payload, &entry()).is_err(), "{changes}");
            }
        }
    }
    #[test]
    fn edit_rejects_ambiguous_references_and_unimplemented_combinations() {
        let p = json!({"model":"moss_image_sft","mode":"edit_image","prompt":"blue cup","referenceAssetIds":["a"],"advanced":{}}).as_object().unwrap().clone();
        for (key, value) in [
            ("referenceAssetIds", json!([])),
            ("referenceAssetIds", json!(["a", "b", "a", "b", "a"])),
            ("referenceAssetIds", json!([1])),
            ("referenceAssetIds", json!([" a"])),
            ("sourceAssetId", json!("b")),
            ("maskAssetId", json!("mask")),
        ] {
            let mut malformed = p.clone();
            malformed.insert(key.into(), value);
            assert!(native_request(&malformed, &entry()).is_err(), "{key}");
        }
        for advanced in [
            json!({"strength":0.5}),
            json!({"mossImage":{"acceleration":"unmerged_bf16"}}),
            json!({"mossImage":{"uncond_skip_layers":[22]}}),
            json!({"mossImageEdit":{"reference_max_edge":0}}),
            json!({"mossImageEdit":{"use_kv_cache":"yes"}}),
            json!({"mossImageEdit":{"crop":true}}),
        ] {
            let mut malformed = p.clone();
            malformed.insert("advanced".into(), advanced);
            assert!(native_request(&malformed, &entry()).is_err());
        }
    }
    #[test]
    fn omitted_off_and_explicit_off_keep_author_defaults() {
        for advanced in [
            json!({}),
            json!({"mossImage":{"acceleration":"off"}}),
            json!({"steps":8}),
        ] {
            let req = native_request(&body(advanced), &entry()).unwrap().unwrap();
            let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &req).unwrap();
            assert_eq!(resolved.acceleration, MossImageAcceleration::Off);
            assert_eq!(resolved.guidance, 3.0);
            assert_eq!(resolved.uncond_skip_layers, vec![22]);
            assert_eq!(req.prompt, "日本語 <tag> {\"items\":[]}");
        }
    }
    #[test]
    fn both_dmd_precisions_and_explicit_empty_layers_are_lossless() {
        for mode in ["unmerged_bf16", "full_fp32"] {
            let req = native_request(&body(json!({"mossImage":{"acceleration":mode}})), &entry())
                .unwrap()
                .unwrap();
            assert_eq!(
                MossImageResolved::resolve(MossImageVariant::Sft, &req)
                    .unwrap()
                    .steps,
                8
            );
            let selected = selected_components(&entry(), req.moss_image.as_ref().unwrap()).unwrap();
            assert_eq!(selected["downloads"].as_array().unwrap().len(), 1);
        }
        let req = native_request(
            &body(json!({"mossImage":{"uncond_skip_layers":[]}})),
            &entry(),
        )
        .unwrap()
        .unwrap();
        assert!(MossImageResolved::resolve(MossImageVariant::Sft, &req)
            .unwrap()
            .uncond_skip_layers
            .is_empty());
        assert!(
            selected_components(&entry(), &MossImageOptions::default()).unwrap()["downloads"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn malformed_options_and_dmd_conflicts_fail_at_the_shared_boundary() {
        for advanced in [
            json!({"mossImage":null}),
            json!({"mossImage":{"unknown":1}}),
            json!({"mossImage":{"acceleration":"on"}}),
            json!({"mossImage":{"max_sequence_length":4097}}),
            json!({"mossImage":{"uncond_skip_layers":[42]}}),
            json!({"mossImage":{"slg":{"layers":[22],"scale":1,"start":0.8,"stop":0.2}}}),
        ] {
            assert!(native_request(&body(advanced), &entry()).is_err());
        }
        for (key, value) in [
            ("guidanceScale", json!(3)),
            ("steps", json!(9)),
            ("schedulerShift", json!(5)),
            ("sampler", json!("euler")),
            ("scheduler", json!("moss_training_grid")),
            ("guidanceMethod", json!("cfg")),
            ("cfgInterval", json!([0, 1])),
        ] {
            let mut advanced = json!({"mossImage":{"acceleration":"unmerged_bf16"}})
                .as_object()
                .unwrap()
                .clone();
            advanced.insert(key.into(), value);
            assert!(
                native_request(&body(Value::Object(advanced)), &entry()).is_err(),
                "{key}"
            );
        }
        for options in [
            json!({"acceleration":"unmerged_bf16","uncond_skip_layers":[]}),
            json!({"acceleration":"full_fp32","training_shift":1}),
        ] {
            assert!(native_request(&body(json!({"mossImage":options})), &entry()).is_err());
        }
    }
    #[test]
    fn variant_conditioning_stride_and_expanded_batch_fail_closed() {
        let mut p = body(json!({"mossImage":{"acceleration":"full_fp32"}}));
        p.insert("model".into(), json!("moss_image_base"));
        assert!(native_request(&p, &entry()).is_err());
        p.insert("model".into(), json!("iris_3b"));
        assert!(native_request(&p, &entry()).is_err());
        for (key, value) in [
            ("width", json!(17)),
            ("width", json!(0)),
            ("count", json!(9)),
            ("referenceAssetId", json!("ref")),
            ("mode", json!("edit_image")),
        ] {
            let mut p = body(json!({}));
            p.insert(key.into(), value);
            assert!(native_request(&p, &entry()).is_err(), "{key}");
        }
        let mut p = body(json!({}));
        p.extend(json!({"width":16,"height":16,"prompts":["日本語 <tag> {\"items\":[]}","other"],"count":4}).as_object().unwrap().clone());
        assert!(native_request(&p, &entry()).is_ok());
        p.insert("count".into(), json!(5));
        assert!(native_request(&p, &entry()).is_err());
    }
    #[test]
    fn reports_are_scrubbed_without_changing_requested_options() {
        let mut p = body(
            json!({"mossImage":{"acceleration":"off"},"mossImageExecutionReport":{"forged":true}}),
        );
        p.insert(REPORT_KEY.into(), json!({}));
        scrub_reports(&mut p);
        assert!(!p.contains_key(REPORT_KEY));
        assert!(p["advanced"].get(REPORT_KEY).is_none());
        assert_eq!(p["advanced"][OPTIONS_KEY]["acceleration"], "off");
    }
    #[test]
    fn shipped_catalog_closures_are_frozen_and_optional_packages_are_mode_specific() {
        use gen_core::moss_image::*;
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1;
        let manifest: Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw)).unwrap();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let entry = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == variant.provider_id())
                .unwrap();
            assert_eq!(entry["license"], "Apache-2.0");
            assert_eq!(entry["defaults"]["steps"], 50);
            assert_eq!(entry["capabilities"], json!(["text_to_image"]));
            for mode in ["off", "unmerged_bf16", "full_fp32"] {
                if variant == MossImageVariant::Base && mode != "off" {
                    continue;
                }
                let options: MossImageOptions =
                    serde_json::from_value(json!({"acceleration":mode})).unwrap();
                let selected = selected_components(entry.as_object().unwrap(), &options).unwrap();
                let mut components = Vec::new();
                for row in selected["downloads"].as_array().unwrap() {
                    let repo = row["repo"].as_str().unwrap();
                    let rev = row["revision"].as_str().unwrap();
                    let mut size = 0u64;
                    for path in row["files"].as_array().unwrap() {
                        let asset = moss_image_asset(repo, rev, path.as_str().unwrap())
                            .expect("every published download has frozen size+SHA identity");
                        size += asset.bytes;
                    }
                    assert_eq!(row["estimatedSizeBytes"].as_u64(), Some(size));
                    if let Some(id) = row["componentId"]
                        .as_str()
                        .filter(|id| id.starts_with("moss-dmd-"))
                    {
                        components.push(id.to_owned());
                        assert!(row.get("licenseBasis").is_some());
                    }
                }
                components.sort();
                let expected = match mode {
                    "off" => vec![],
                    "unmerged_bf16" => vec!["moss-dmd-dataspace", "moss-dmd-lora", "moss-dmd-time"],
                    _ => vec!["moss-dmd-dataspace", "moss-dmd-transformer"],
                };
                assert_eq!(components, expected);
            }
        }
    }
}
