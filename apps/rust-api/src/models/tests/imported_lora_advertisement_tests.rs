use super::*;

fn route(
    operation: &str,
    conditioning: &[&str],
) -> sceneworks_core::jobs_store::ImportedProviderSurface {
    sceneworks_core::jobs_store::ImportedProviderSurface {
        family: "fixture".to_owned(),
        source: "fixture".to_owned(),
        operation: operation.to_owned(),
        provider_id: "fixture_provider".to_owned(),
        conditioning: conditioning
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        supports_lora: true,
        supports_lokr: false,
        supported_quants: vec!["q4".to_owned(), "q8".to_owned()],
        supports_kv_cache: false,
        supports_sequential_offload: false,
        registry_cached: true,
    }
}

fn entry(id: &str, family: &str) -> JsonObject {
    let source = match family {
        "krea_2" => "transformer_file",
        "sdxl" => "fused_checkpoint",
        "mage-flow" => "transformer_directory",
        _ => "comfy_ui_tree",
    };
    json!({
        "id": id,
        "type": "image",
        "family": family,
        "catalogScope": "user",
        "importSourceShape": source,
    })
    .as_object()
    .expect("entry object")
    .clone()
}

fn withdrawn(object: &JsonObject) -> bool {
    object
        .get("loraCompatibility")
        .and_then(|value| value.get("families"))
        .and_then(Value::as_array)
        .is_some_and(|families| families.is_empty())
        && object["loraCompatibility"]["supported"] == json!(false)
}

/// A plan-backed imported MiniMax-H3 entry of ANY supported partition advertises the family's
/// adapter surface on its lanes. A Ref2VA import serves references, not text-only shapes: a
/// text_to_video-only probe cannot be claimed by it, so deriving the advertisement from that
/// single shape withdrew the adapter surface the CONTRACTED reference Singularity models
/// genuinely serve. The advertisement now asks whether ANY valid supported shape is claimable,
/// through the same mode-conditioning probes the support projection uses.
#[test]
fn imported_ref2va_h3_advertises_adapters_through_its_reference_shape() {
    let h3 = |partition: &str, base: &str| {
        json!({
            "id": format!("singularity_{partition}"),
            "type": "video",
            "family": sceneworks_core::minimax_h3_import::MINIMAX_H3_IMPORT_FAMILY,
            "catalogScope": "user",
            "importSourceShape":
                sceneworks_core::minimax_h3_import::MINIMAX_H3_IMPORT_SOURCE_SHAPE,
            "baseModel": base,
            "minimaxH3Partition": partition,
            "importPlan": {
                "checkpointId": format!("managed/singularity_{partition}"),
                "assembly": {
                    "schemaVersion":
                        sceneworks_core::minimax_h3_import::MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION,
                    "partition": partition,
                },
            },
        })
        .as_object()
        .expect("entry object")
        .clone()
    };
    assert!(
        sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(&h3(
            "ref2va",
            "minimax_h3_ref"
        )),
        "the fixture must be the imported plan-backed shape the projection keys on"
    );
    for (partition, base) in [
        ("ref2va", "minimax_h3_ref"),
        ("fl2va", "minimax_h3"),
        ("hybrid", "minimax_h3"),
    ] {
        for (mlx, candle) in [(true, false), (false, true)] {
            let mut entry = h3(partition, base);
            apply_imported_lora_advertisement_for_lanes(&mut entry, mlx, candle);
            assert_eq!(
                entry["loraCompatibility"]["families"],
                json!(["minimax-h3"]),
                "{partition}: the servable H3 adapter family must be advertised \
                 (mlx={mlx}, candle={candle}): {entry:?}"
            );
            assert_eq!(
                entry["loraCompatibility"]["supported"],
                json!(true),
                "{partition}: Studio must not hide adapters the lane can claim"
            );
            assert!(
                !withdrawn(&entry),
                "{partition}: the empty/unsupported withdrawal is the exact regression \
                 (mlx={mlx}, candle={candle})"
            );
        }
    }
}

/// Both native imported Krea 2 single-file loaders take adapters (sc-18480), so neither platform
/// projection may withdraw the synthesized family promise — each scheduler gate can claim the
/// adapter shape, so the projection AFFIRMS it instead.
#[test]
fn imported_krea_2_advertises_adapters_on_both_registered_backends() {
    let mut on_candle = entry("user_kreamania_variant5", "krea_2");
    apply_imported_lora_advertisement_for_lanes(&mut on_candle, false, true);
    assert_eq!(
        on_candle["loraCompatibility"]["families"],
        json!(["krea_2"])
    );
    assert_eq!(on_candle["loraCompatibility"]["supported"], json!(true));

    let mut on_mlx = entry("user_kreamania_variant5", "krea_2");
    apply_imported_lora_advertisement_for_lanes(&mut on_mlx, true, false);
    assert_eq!(on_mlx["loraCompatibility"]["families"], json!(["krea_2"]));
    assert_eq!(on_mlx["loraCompatibility"]["supported"], json!(true));
}

/// Generated Mage-Flow full fine-tunes render plain text-to-image on MLX, whose provider seam
/// rejects adapters — so that projection withdraws the LoRA promise while preserving the
/// routable model itself. Candle declares no `mage-flow` imported provider at all, so it has no
/// opinion to project and must leave the entry untouched: withdrawing there would advertise a
/// lane that exists and merely lacks adapters, which is a different and false claim.
#[test]
fn mage_flow_withdraws_adapters_only_on_the_lane_that_serves_it() {
    let mut on_mlx = entry("finetune_9f3c", "mage-flow");
    apply_imported_lora_advertisement_for_lanes(&mut on_mlx, true, false);
    assert!(
        withdrawn(&on_mlx),
        "MLX renders generated Mage t2i but cannot load adapters: {on_mlx:?}"
    );

    let untouched = entry("finetune_9f3c", "mage-flow");
    let mut on_candle = entry("finetune_9f3c", "mage-flow");
    apply_imported_lora_advertisement_for_lanes(&mut on_candle, false, true);
    assert_eq!(
        on_candle, untouched,
        "candle declares no mage-flow imported provider, so there is no promise to withdraw"
    );
}

/// The LANE-SCOPED verdict above is per-lane truth; the PRODUCTION binding must still publish a
/// withdrawal on a candle-only deployment. sc-15328's hang was exactly this shape — no explicit
/// empty family list, so `families_from_value_chain` fell back to `family`, the API accepted a
/// LoRA submission, and no worker could claim it. That regressed on every non-macOS build the
/// moment the advertisement became facts-derived, because candle registers no `mage-flow`
/// imported provider and therefore answers "no opinion" rather than "refuses adapters".
#[test]
fn a_candle_only_build_still_withdraws_mage_adapters_from_the_other_lane() {
    let mut object = entry("finetune_9f3c", "mage-flow");
    // The production binding's own lane (candle) has no opinion here.
    assert_eq!(
        imported_lora_advertisement_verdict(&object, false, true),
        None
    );
    // MLX is the lane that serves the family and refuses adapters, and that refusal is what
    // must reach the entry regardless of which engine this build links.
    assert_eq!(
        imported_lora_advertisement_verdict(&object, true, false),
        Some(false)
    );
    write_imported_lora_advertisement(&mut object, false);
    assert!(withdrawn(&object), "{object:?}");
    assert_eq!(object["loraCompatibility"]["families"], json!([]));

    // A positive advertisement must NOT cross lanes: offering adapters this build cannot claim
    // is the same hang pointed the other way.
    let krea = entry("user_kreamania_variant5", "krea_2");
    assert_eq!(
        imported_lora_advertisement_verdict(&krea, true, false),
        Some(true)
    );
}

/// SDXL genuinely serves adapters on both native loaders, and a builtin routes by id rather
/// than family. Neither may be touched — this projection only ever removes a promise nothing
/// can keep.
#[test]
fn honest_advertisements_and_builtins_are_never_rewritten() {
    for (mlx, candle) in [(true, false), (false, true)] {
        let mut sdxl = entry("community_xl", "sdxl");
        apply_imported_lora_advertisement_for_lanes(&mut sdxl, mlx, candle);
        assert_eq!(sdxl["loraCompatibility"]["families"], json!(["sdxl"]));
        assert_eq!(sdxl["loraCompatibility"]["supported"], json!(true));

        let mut builtin = entry("krea_2_turbo", "krea_2");
        builtin.insert("catalogScope".to_owned(), json!("builtin"));
        apply_imported_lora_advertisement_for_lanes(&mut builtin, mlx, candle);
        assert!(
            builtin.get("loraCompatibility").is_none(),
            "a builtin routes by id; its shipped manifest row must not be rewritten"
        );
    }
}

/// A supported advertisement is byte-preserved, while a real withdrawal changes only the
/// adapter verdict and keeps sibling compatibility metadata intact.
#[test]
fn support_and_withdrawal_preserve_sibling_compatibility_keys() {
    // The canonical family token, so the affirming branch finds it already present and the
    // entry is byte-preserved rather than gaining a second, duplicate family entry.
    let compatibility =
        json!({ "families": ["krea_2"], "supported": true, "types": ["character", "style"] });
    let mut supported = entry("user_kreamania_variant5", "krea_2");
    supported.insert("loraCompatibility".to_owned(), compatibility.clone());
    apply_imported_lora_advertisement_for_lanes(&mut supported, false, true);
    assert_eq!(
        supported["loraCompatibility"], compatibility,
        "Candle's supported Krea adapter family and sibling keys must remain untouched"
    );

    let mut withdrawn_entry = entry("finetune_9f3c", "mage-flow");
    withdrawn_entry.insert(
        "loraCompatibility".to_owned(),
        json!({ "families": ["mage-flow"], "supported": true, "types": ["character", "style"] }),
    );
    apply_imported_lora_advertisement_for_lanes(&mut withdrawn_entry, true, false);
    assert!(withdrawn(&withdrawn_entry));
    assert_eq!(
        withdrawn_entry["loraCompatibility"]["types"],
        json!(["character", "style"])
    );
}

/// A manifest-declared `families: []` (any model that ships without LoRA support; historically
/// Qwen Image 2.1, before sc-24156 gave it its own family) is refused by the LoRA validator, so
/// the catalog must say `supported: false` — otherwise the web's "cannot gate" branch offers
/// every LoRA and auto-applies the Krea `image_edit` LoRA in edit mode. Non-empty and absent
/// advertisements, and an explicit `supported`, are left alone.
#[test]
fn a_declared_empty_lora_advertisement_is_marked_unsupported() {
    let mut no_lora =
        json!({ "id": "no_lora_model", "loraCompatibility": { "families": [], "types": [] } })
            .as_object()
            .expect("object")
            .clone();
    mark_empty_lora_advertisement_unsupported(&mut no_lora);
    assert_eq!(
        no_lora["loraCompatibility"],
        json!({ "families": [], "types": [], "supported": false })
    );

    let serving = json!({ "families": ["qwen-image"], "types": ["style"] });
    let mut qwen_2512 = json!({ "id": "qwen_image", "loraCompatibility": serving.clone() })
        .as_object()
        .expect("object")
        .clone();
    mark_empty_lora_advertisement_unsupported(&mut qwen_2512);
    assert_eq!(qwen_2512["loraCompatibility"], serving);

    let mut absent = json!({ "id": "external_qwen", "loraCompatibility": {} })
        .as_object()
        .expect("object")
        .clone();
    mark_empty_lora_advertisement_unsupported(&mut absent);
    assert_eq!(absent["loraCompatibility"], json!({}));

    let explicit = json!({ "families": [], "supported": true });
    let mut kept = json!({ "id": "x", "loraCompatibility": explicit.clone() })
        .as_object()
        .expect("object")
        .clone();
    mark_empty_lora_advertisement_unsupported(&mut kept);
    assert_eq!(kept["loraCompatibility"], explicit);
}

#[test]
fn provider_surface_matches_exact_source_shape() {
    let mut krea = entry("user_krea", "krea_2");
    apply_imported_provider_surface_for_lanes(&mut krea, true, false);
    assert_eq!(krea["ui"]["img2img"], json!(true));
    assert_eq!(krea["ui"]["poseLibrary"], json!(true));
    assert!(krea["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "edit_image"));
    assert_eq!(krea["runtimeQuantTiers"], json!(["q4", "q8", "bf16"]));

    let mut wrong_shape = entry("wrong_shape", "krea_2");
    wrong_shape.insert("importSourceShape".to_owned(), json!("fused_checkpoint"));
    apply_model_manifest_defaults(&mut wrong_shape, "image", Some("krea_2"));
    apply_imported_provider_surface_for_lanes(&mut wrong_shape, true, true);
    assert!(wrong_shape["capabilities"]
        .as_array()
        .is_some_and(|values| values
            .iter()
            .all(|value| { !matches!(value.as_str(), Some("text_to_image" | "edit_image")) })));
    assert!(wrong_shape["ui"].get("img2img").is_none());
    assert!(wrong_shape["ui"].get("editReferences").is_none());
    assert!(wrong_shape["ui"].get("controlModes").is_none());
    assert!(wrong_shape["ui"].get("controlOverlay").is_none());
    assert!(wrong_shape.get("runtimeQuantTiers").is_none());
    assert_eq!(wrong_shape["macSupport"]["supported"], json!(false));
}

#[test]
fn krea_fine_tunes_keep_pose_controls_and_the_base_overlay_on_both_lanes() {
    let builtin = sceneworks_core::builtin_manifests::builtin_model_entry("krea_2_turbo").unwrap();
    for id in ["kreamania", "sick_ollie"] {
        for (mlx, candle) in [(true, false), (false, true)] {
            let mut imported = entry(id, "krea_2");
            apply_imported_provider_surface_for_lanes(&mut imported, mlx, candle);
            assert_eq!(imported["ui"]["poseLibrary"], true, "{id} mlx={mlx}");
            assert_eq!(imported["ui"]["controlModes"], json!(["pose"]));
            assert_eq!(
                imported["ui"]["controlScale"],
                builtin["ui"]["controlScale"]
            );
            assert_eq!(imported["ui"]["controlOverlay"], true);
            assert_eq!(imported["ui"]["controlOverlayBaseModel"], "krea_2_turbo");
        }
    }
}

/// sc-11043 (epic 11037): a checkpoint STORED as NVFP4 offers exactly its own tier.
///
/// The q4/q8/bf16 sweep describes what a route can quantize a dense import TO; these bytes are
/// already packed E2M1 and nothing re-quantizes them. Offering the sweep advertised `q4` for an
/// NVFP4 checkpoint — the alias E2 forbids — and every job the offer produced was then refused
/// by the admission gate, so the picker showed tiers the router would not accept.
#[test]
fn an_nvfp4_import_offers_its_own_tier_and_never_q4() {
    let mut native = entry("user_krea_nvfp4", "krea_2");
    native.insert("importQuantFormat".to_owned(), json!("nvfp4"));
    apply_imported_provider_surface_for_lanes(&mut native, false, true);
    assert_eq!(native["runtimeQuantTiers"], json!(["nvfp4"]));
    assert_eq!(native["ui"]["poseLibrary"], true);
    assert_eq!(native["ui"]["controlModes"], json!(["pose"]));
    assert_eq!(native["ui"]["controlOverlayBaseModel"], "krea_2_turbo");
    let mut on_mac = native.clone();
    apply_imported_provider_surface_for_lanes(&mut on_mac, true, false);
    assert!(on_mac["ui"].get("poseLibrary").is_none());
    assert_eq!(on_mac["macSupport"]["supported"], false);

    // A dense sibling through the same route is untouched: the projection keys on the entry's
    // own source codec, not on the family or the route's advertised quants.
    let mut dense = entry("user_krea_bf16", "krea_2");
    dense.insert("importQuantFormat".to_owned(), json!("bf16"));
    apply_imported_provider_surface_for_lanes(&mut dense, false, true);
    assert_eq!(dense["runtimeQuantTiers"], json!(["q4", "q8", "bf16"]));

    // A classification with no proved engine codec is not an NVFP4 claim.
    let mut packed = entry("user_krea_packed", "krea_2");
    packed.insert("importQuantFormat".to_owned(), json!("comfy_quant_packed"));
    apply_imported_provider_surface_for_lanes(&mut packed, false, true);
    assert_eq!(packed["runtimeQuantTiers"], json!(["q4", "q8", "bf16"]));
}

#[test]
fn exact_route_projection_withdraws_family_siblings_before_adding_supported_operations() {
    let mut candle_sdxl = entry("community_xl", "sdxl");
    apply_model_manifest_defaults(&mut candle_sdxl, "image", Some("sdxl"));
    let generate = route("generate", &[]);
    project_imported_operation_surface(&mut candle_sdxl, &[&generate]);
    assert!(candle_sdxl["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "text_to_image"));
    assert!(candle_sdxl["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| value != "edit_image"));
    assert!(candle_sdxl["ui"].get("img2img").is_none());

    let mut krea = entry("user_krea", "krea_2");
    apply_model_manifest_defaults(&mut krea, "image", Some("krea_2"));
    let edit = route("edit", &["reference", "multi_reference"]);
    let pose = route("pose", &["control"]);
    let multi_phase = route("multi_phase", &[]);
    let generate = route("generate", &["reference"]);
    project_imported_operation_surface(&mut krea, &[&generate, &edit, &pose, &multi_phase]);
    assert_eq!(krea["ui"]["img2img"], json!(true));
    assert_eq!(krea["ui"]["poseLibrary"], json!(true));
    assert!(krea["ui"].get("editReferences").is_some());
    assert!(krea["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "edit_image"));
    assert!(krea["loraCompatibility"]["types"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "acceleration"));
}

#[test]
fn external_comfyui_surface_uses_exact_candle_registration() {
    let mut zimage = entry("external_z", "z-image");
    zimage.insert("catalogScope".to_owned(), json!("external"));
    apply_imported_provider_surface_for_lanes(&mut zimage, false, true);
    assert_eq!(zimage["ui"]["img2img"], json!(true));

    let mut qwen = entry("external_qwen", "qwen-image");
    qwen.insert("catalogScope".to_owned(), json!("external"));
    apply_imported_provider_surface_for_lanes(&mut qwen, false, true);
    assert_eq!(qwen["capabilities"], json!(["text_to_image"]));
    assert_eq!(qwen["ui"], json!({}));
    assert!(qwen.get("runtimeQuantTiers").is_none());
    assert_eq!(qwen["loraCompatibility"], json!({}));

    let mut mage_on_mlx = entry("finetune_9f3c", "mage-flow");
    apply_model_manifest_defaults(&mut mage_on_mlx, "image", Some("mage-flow"));
    apply_imported_provider_surface_for_lanes(&mut mage_on_mlx, true, false);
    assert!(mage_on_mlx["capabilities"]
        .as_array()
        .is_some_and(|values| values.contains(&json!("text_to_image"))));
    assert_eq!(
        mage_on_mlx["runtimeQuantTiers"],
        json!(["q4", "q8", "bf16"])
    );

    let mut mage_on_candle = entry("finetune_9f3c", "mage-flow");
    apply_model_manifest_defaults(&mut mage_on_candle, "image", Some("mage-flow"));
    apply_imported_provider_surface_for_lanes(&mut mage_on_candle, false, true);
    assert!(mage_on_candle["capabilities"]
        .as_array()
        .is_some_and(|values| !values.contains(&json!("text_to_image"))));
    assert!(
        mage_on_candle.get("loraCompatibility").is_none(),
        "an empty route set must abstain instead of manufacturing compatibility metadata"
    );
    assert!(mage_on_candle.get("runtimeQuantTiers").is_none());
}

#[test]
fn imported_sampling_menus_follow_the_exact_provider_and_preserve_authored_values() {
    for (family, provider) in [
        ("krea_2", "krea_2_turbo"),
        ("mage-flow", "mage_flow_base"),
        ("sdxl", "sdxl"),
    ] {
        let mut imported = entry("my_fine_tune", family);
        apply_imported_provider_surface_for_lanes(&mut imported, true, false);
        let builtin = sceneworks_core::builtin_manifests::builtin_model_entry(provider).unwrap();
        let limits = builtin
            .get("mlx")
            .and_then(|lane| lane.get("limits"))
            .unwrap_or(&builtin["limits"]);
        for key in ["samplers", "schedulers"] {
            if let Some(expected) = limits.get(key) {
                assert_eq!(&imported["limits"][key], expected, "{family} {key}");
                assert_eq!(
                    &imported["mlx"]["limits"][key], expected,
                    "{family} mlx {key}"
                );
            }
        }
        imported.insert(
            "defaults".into(),
            json!({"steps": 19, "guidanceScale": 1.5, "schedulerShift": 17.25}),
        );
        imported.insert(
            "limits".into(),
            json!({"samplers": ["euler"], "schedulers": ["normal"]}),
        );
        apply_imported_provider_surface_for_lanes(&mut imported, true, false);
        assert_eq!(imported["defaults"]["schedulerShift"], 17.25);
        assert_eq!(imported["defaults"]["guidanceScale"], 1.5);
        assert_eq!(imported["limits"]["samplers"], json!(["euler"]));
        assert_eq!(imported["limits"]["schedulers"], json!(["normal"]));
    }
}
