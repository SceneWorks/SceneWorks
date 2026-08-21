//! Candle (Windows/CUDA) SDXL lane routing (epic 3672, sc-3678): the candle worker serves a
//! gated, narrow SDXL/RealVisXL **txt2img-only** lane and must refuse every other shape, leaving it
//! queued for a compatible native worker. These tests pin the lane boundary (`image_request_candle_eligible`) and
//! the full claim gate (`worker_supports_job` via the `candle` marker capability).
use super::*;
use serde_json::{json, Value};

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("test value is an object").clone()
}

/// A queued `image_generate` job carrying `payload`, built via serde so the test never has to
/// spell out the full `JobSnapshot` field set.
fn image_generate_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_1",
        "type": "image_generate",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-12T00:00:00Z",
        "updatedAt": "2026-06-12T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// A queued `image_edit` job carrying `payload` — the distinct job type the API stamps for the
/// Image Studio/Editor "plain Image Edit" (`mode == "edit_image"`, `apps/rust-api` generation.rs).
/// The candle edit lanes (sc-5487) are reached via this type, so the routing/claim tests must probe
/// it directly rather than only via `image_generate_job`.
fn image_edit_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_1",
        "type": "image_edit",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-12T00:00:00Z",
        "updatedAt": "2026-06-12T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// A worker on a real CUDA gpu index advertising `capabilities` (string ids). The candle worker
/// carries the `candle` marker; a synthetic generic descriptor does not.
fn gpu_worker(capabilities: &[&str]) -> WorkerSnapshot {
    gpu_worker_with_status(capabilities, "idle")
}

/// [`gpu_worker`] with an explicit status, so the sc-16260 unhealthy backstop can be exercised
/// against a worker that is otherwise fully eligible.
fn gpu_worker_with_status(capabilities: &[&str], status: &str) -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": "worker_1",
        "gpuId": "0",
        "status": status,
        "capabilities": capabilities,
        "loadedModels": [],
        "registeredAt": "2026-06-12T00:00:00Z",
        "lastSeenAt": "2026-06-12T00:00:00Z",
    }))
    .expect("valid WorkerSnapshot")
}

// Mirrors the real candle advertised set (`with_candle_capabilities`): `image_generate` (derived)
// plus the `image_edit` carve-out (sc-5487 edit lanes) and the `candle` lane marker.
const CANDLE_CAPS: &[&str] = &["gpu", "image_generate", "image_edit", "candle"];
// Synthetic generic descriptor used to exercise the non-candle branch; production has no fallback.
const TORCH_CAPS: &[&str] = &["gpu", "image_generate", "image_edit", "image_detail"];

#[test]
fn candle_marker_detection_parses_exact_capability_membership() {
    assert!(encoded_worker_has_capability(
        r#"["gpu","image_generate","candle"]"#,
        "candle"
    ));
    assert!(!encoded_worker_has_capability(
        r#"["gpu","image_generate","not-candle"]"#,
        "candle"
    ));
    assert!(!encoded_worker_has_capability(
        r#"{"capabilities":["candle"]}"#,
        "candle"
    ));
}

#[test]
fn candle_pose_model_catalog_matches_control_routes() {
    let mut routed = candle_pose_route_models();
    routed.sort_unstable();
    routed.dedup();
    let mut catalog = CANDLE_POSE_MODELS.to_vec();
    catalog.sort_unstable();
    catalog.dedup();
    assert_eq!(
        catalog, routed,
        "the unsupported-pose guard must derive the same model set as the control route table"
    );
}

#[test]
fn candle_image_dispatch_reports_named_lane_and_preserves_precedence() {
    assert_eq!(
        candle_image_route_lanes(),
        vec![
            CandleImageLane::InstantId,
            CandleImageLane::SdxlEdit,
            CandleImageLane::Flux2Edit,
            CandleImageLane::Flux2Edit,
            CandleImageLane::QwenEdit,
            CandleImageLane::SenseNovaEdit,
            CandleImageLane::ZImageEdit,
            CandleImageLane::ZImageIdentity,
            CandleImageLane::IdeogramEdit,
            CandleImageLane::IdeogramImg2Img,
            CandleImageLane::BooguEdit,
            CandleImageLane::MageEdit,
            CandleImageLane::BooguImg2Img,
            CandleImageLane::KreaEdit,
            CandleImageLane::BerniniEdit,
            CandleImageLane::SdxlIpAdapter,
            CandleImageLane::KolorsIpAdapter,
            CandleImageLane::FluxIpAdapter,
            CandleImageLane::QwenControl,
            CandleImageLane::KolorsControl,
            CandleImageLane::KolorsEdit,
            CandleImageLane::ZImageControl,
            CandleImageLane::ZImageImg2Img,
            CandleImageLane::Sd3Img2Img,
            CandleImageLane::SanaImg2Img,
            CandleImageLane::Flux1Control,
            CandleImageLane::Flux2Control,
            CandleImageLane::KreaControl,
            CandleImageLane::KreaImg2Img,
            CandleImageLane::Pulid,
        ],
        "the explicit route table's row identity and precedence are contractual"
    );

    let edit = |model| json!({ "model": model, "mode": "edit_image", "sourceAssetId": "source_1" });
    let reference = |model| json!({ "model": model, "referenceAssetId": "reference_1" });
    let pose = |model| json!({ "model": model, "advanced": { "poses": [{ "keypoints": [] }] } });
    let character = |model| {
        json!({
            "model": model,
            "mode": "character_image",
            "referenceAssetId": "reference_1"
        })
    };

    // Every specialized table row, every model selector in that row, the imported-family
    // pre-route, and the generic fallback get an exact named-lane oracle. A mislabeled row,
    // missing model, or row moved behind another matching shape changes one of these outcomes.
    let mut cases = vec![
        (
            json!({
                "model": "kreamania_variant4",
                "modelManifestEntry": {
                    "family": "krea_2",
                    "importSourceShape": "transformer_file",
                    "paths": { "model": "C:\\SceneWorks\\models\\imports\\kreamania_variant4" }
                }
            }),
            CandleImageLane::ImportedFamily,
        ),
        (character("instantid_realvisxl"), CandleImageLane::InstantId),
        (edit("flux2_klein_9b"), CandleImageLane::Flux2Edit),
        (edit("flux2_klein_9b_true_v2"), CandleImageLane::Flux2Edit),
        (edit("flux2_dev"), CandleImageLane::Flux2Edit),
        (edit("qwen_image_edit"), CandleImageLane::QwenEdit),
        (edit("qwen_image_edit_2509"), CandleImageLane::QwenEdit),
        (edit("qwen_image_edit_2511"), CandleImageLane::QwenEdit),
        (
            edit("qwen_image_edit_2511_lightning"),
            CandleImageLane::QwenEdit,
        ),
        (edit("z_image_turbo"), CandleImageLane::ZImageEdit),
        (edit("z_image_edit"), CandleImageLane::ZImageEdit),
        (
            json!({
                "model": "z_image_turbo",
                "mode": "character_image",
                "referenceAssetId": "reference_1",
                "advanced": { "referenceStrength": 0.8 }
            }),
            CandleImageLane::ZImageIdentity,
        ),
        (edit("ideogram_4"), CandleImageLane::IdeogramEdit),
        (edit("ideogram_4_turbo"), CandleImageLane::IdeogramEdit),
        (reference("ideogram_4"), CandleImageLane::IdeogramImg2Img),
        (
            reference("ideogram_4_turbo"),
            CandleImageLane::IdeogramImg2Img,
        ),
        (edit("boogu_image_edit"), CandleImageLane::BooguEdit),
        (edit("mage_flow_edit_base"), CandleImageLane::MageEdit),
        (edit("mage_flow_edit"), CandleImageLane::MageEdit),
        (edit("mage_flow_edit_turbo"), CandleImageLane::MageEdit),
        (reference("boogu_image"), CandleImageLane::BooguImg2Img),
        (
            reference("boogu_image_turbo"),
            CandleImageLane::BooguImg2Img,
        ),
        (edit("krea_2_raw"), CandleImageLane::KreaEdit),
        (edit("krea_2_turbo"), CandleImageLane::KreaEdit),
        (edit("bernini_image"), CandleImageLane::BerniniEdit),
        (reference("kolors"), CandleImageLane::KolorsIpAdapter),
        (reference("flux_dev"), CandleImageLane::FluxIpAdapter),
        (reference("flux_schnell"), CandleImageLane::FluxIpAdapter),
        (pose("qwen_image"), CandleImageLane::QwenControl),
        (pose("kolors"), CandleImageLane::KolorsControl),
        (pose("z_image_turbo"), CandleImageLane::ZImageControl),
        (pose("z_image"), CandleImageLane::ZImageControl),
        (reference("z_image"), CandleImageLane::ZImageImg2Img),
        (reference("z_image_turbo"), CandleImageLane::ZImageImg2Img),
        (reference("sd3_5_large"), CandleImageLane::Sd3Img2Img),
        (reference("sd3_5_large_turbo"), CandleImageLane::Sd3Img2Img),
        (reference("sd3_5_medium"), CandleImageLane::Sd3Img2Img),
        (reference("sana_1600m"), CandleImageLane::SanaImg2Img),
        (reference("sana_sprint_1600m"), CandleImageLane::SanaImg2Img),
        (pose("flux_dev"), CandleImageLane::Flux1Control),
        (pose("flux2_dev"), CandleImageLane::Flux2Control),
        (
            json!({
                "model": "flux2_dev",
                "mode": "character_image",
                "referenceAssetId": "identity_1",
                "advanced": { "poses": [{ "keypoints": [] }] }
            }),
            CandleImageLane::Flux2Control,
        ),
        (
            json!({
                "model": "qwen_image_edit_2511_lightning",
                "mode": "character_image",
                "referenceAssetId": "identity_1",
                "advanced": { "poses": [{ "keypoints": [] }] }
            }),
            CandleImageLane::QwenEdit,
        ),
        (pose("krea_2_turbo"), CandleImageLane::KreaControl),
        (reference("krea_2_turbo"), CandleImageLane::KreaImg2Img),
        (reference("krea_2_raw"), CandleImageLane::KreaImg2Img),
        (character("pulid_flux_dev"), CandleImageLane::Pulid),
        (
            json!({ "model": "sdxl", "prompt": "plain" }),
            CandleImageLane::TextToImage,
        ),
    ];
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        cases.push((edit(model), CandleImageLane::SdxlEdit));
        cases.push((reference(model), CandleImageLane::SdxlIpAdapter));
    }

    for (payload, expected) in cases {
        assert_eq!(
            image_job_candle_lane(&image_generate_job(payload)),
            Some(expected)
        );
    }

    // Real same-model overlaps pin the exact first-match precedence. Each tuple names the
    // predicates simultaneously satisfied and the route that must win.
    let overlaps = [
        (
            json!({
                "model": "z_image_turbo",
                "mode": "character_image",
                "referenceAssetId": "reference_1",
                "advanced": { "referenceStrength": 0.8 }
            }),
            CandleImageLane::ZImageIdentity,
            &[
                zimage_identity_candle_eligible as fn(&Map<String, Value>) -> bool,
                zimage_img2img_candle_eligible,
            ][..],
        ),
        (
            json!({
                "model": "z_image_turbo",
                "referenceAssetId": "reference_1",
                "advanced": { "poses": [{}] }
            }),
            CandleImageLane::ZImageControl,
            &[
                zimage_control_candle_eligible,
                zimage_img2img_candle_eligible,
            ][..],
        ),
        (
            json!({
                "model": "krea_2_turbo",
                "referenceAssetId": "reference_1",
                "advanced": { "poses": [{}] }
            }),
            CandleImageLane::KreaControl,
            &[krea_control_candle_eligible, krea_img2img_candle_eligible][..],
        ),
        (
            json!({
                "model": "kolors",
                "referenceAssetId": "reference_1",
                "advanced": { "poses": [{}] }
            }),
            CandleImageLane::KolorsIpAdapter,
            &[
                kolors_ipadapter_candle_eligible,
                kolors_control_candle_eligible,
            ][..],
        ),
        (
            json!({
                "model": "flux_dev",
                "referenceAssetId": "reference_1",
                "advanced": { "poses": [{}] }
            }),
            CandleImageLane::FluxIpAdapter,
            &[
                flux_ipadapter_candle_eligible,
                flux1_control_candle_eligible,
            ][..],
        ),
    ];
    for (payload, expected, predicates) in overlaps {
        let payload_object = object(payload.clone());
        assert!(
            predicates
                .iter()
                .all(|predicate| predicate(&payload_object)),
            "overlap fixture must actually satisfy every asserted predicate: {payload}"
        );
        assert_eq!(
            image_job_candle_lane(&image_generate_job(payload)),
            Some(expected)
        );
    }
}

#[test]
fn conditioned_routes_fail_closed_on_conflicts_malformed_references_and_cfg() {
    for model in ["qwen_image_edit_2511", "sensenova_u1_8b"] {
        assert!(image_job_is_candle_eligible(&image_generate_job(json!({
            "model": model, "mode": "character_image",
            "referenceAssetIds": ["a", "b"], "advanced": { "trueCfgScale": "2.5" }
        }))));
        for absent_plural in [Value::Null, json!([])] {
            assert!(image_job_is_candle_eligible(&image_generate_job(json!({
                "model": model, "mode": "character_image",
                "referenceAssetIds": absent_plural, "referenceAssetId": "a"
            }))));
        }
        for malformed in [
            json!({ "model": model, "mode": "character_image", "referenceAssetIds": ["a", ""] }),
            json!({ "model": model, "mode": "character_image", "referenceAssetIds": ["a", "b", "c", "d", "e", "f"] }),
            json!({ "model": model, "mode": "character_image", "referenceAssetIds": "a" }),
            json!({ "model": model, "mode": "character_image", "referenceAssetId": "" }),
            json!({ "model": model, "mode": "character_image", "referenceAssetId": "a", "referenceAssetIds": ["b"] }),
            json!({ "model": model, "mode": "character_image", "referenceAssetId": "a", "advanced": { "trueCfgScale": {} } }),
            json!({ "model": model, "mode": "character_image", "referenceAssetId": "a", "advanced": { "trueCfgScale": "not-a-number" } }),
        ] {
            assert!(!image_job_is_candle_eligible(&image_generate_job(
                malformed
            )));
        }
    }
    for model in ["krea_2_raw", "krea_2_turbo"] {
        assert!(image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "edit_image", "referenceAssetId": "a"
        }))));
        assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "edit_image", "referenceAssetIds": ["a", "b", "c"]
        }))));
        assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "b"
        }))));
        for absent_plural in [Value::Null, json!([])] {
            assert!(image_job_is_candle_eligible(&image_edit_job(json!({
                "model": model, "mode": "edit_image",
                "referenceAssetIds": absent_plural, "referenceAssetId": "a"
            }))));
        }
        for unsupported in [
            json!({ "model": model, "mode": "edit_image", "referenceAssetId": "a", "maskAssetId": "mask" }),
            json!({ "model": model, "mode": "edit_image", "referenceAssetId": "a", "controls": [{}] }),
            json!({ "model": model, "mode": "edit_image", "referenceAssetId": "a", "advanced": { "poses": [{}] } }),
            json!({ "model": model, "mode": "edit_image", "referenceAssetId": "a", "advanced": { "phases": [{}] } }),
        ] {
            assert!(!image_job_is_candle_eligible(&image_edit_job(unsupported)));
        }
    }

    for model in [
        "flux2_dev",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        assert!(image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "style_variations",
            "referenceAssetIds": null, "referenceAssetId": "a"
        }))));
        assert!(image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "reference",
            "referenceAssetIds": [], "referenceAssetId": "a"
        }))));
        for malformed in [
            json!({ "model": model, "mode": "reference", "referenceAssetIds": ["a", "b", "c", "d", "e"] }),
            json!({ "model": model, "mode": "reference", "referenceAssetIds": ["a", null] }),
            json!({ "model": model, "mode": "reference", "referenceAssetId": "a", "sourceAssetId": "b" }),
            json!({ "model": model, "mode": "reference", "referenceAssetId": "a", "advanced": { "trueCfgScale": [] } }),
        ] {
            assert!(!image_job_is_candle_eligible(&image_edit_job(malformed)));
        }
    }

    for strength in [json!(0.0), json!("0.6"), Value::Null] {
        assert!(image_job_is_candle_eligible(&image_edit_job(json!({
            "model": "kolors", "mode": "edit_image", "sourceAssetId": "source",
            "advanced": { "strength": strength }
        }))));
    }
    for strength in [json!(false), json!({}), json!("not-a-number")] {
        assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
            "model": "kolors", "mode": "edit_image", "sourceAssetId": "source",
            "advanced": { "strength": strength }
        }))));
    }
    assert_eq!(
        image_job_candle_lane(&image_generate_job(json!({
            "model": "kolors", "referenceAssetId": "ip-ref"
        }))),
        Some(CandleImageLane::KolorsIpAdapter)
    );
    assert_eq!(
        image_job_candle_lane(&image_generate_job(json!({
            "model": "kolors", "advanced": { "poses": [{}] }
        }))),
        Some(CandleImageLane::KolorsControl)
    );
    assert_eq!(
        image_job_candle_lane(&image_edit_job(json!({
            "model": "kolors", "mode": "edit_image", "sourceAssetId": "source"
        }))),
        Some(CandleImageLane::KolorsEdit)
    );
}

#[test]
fn qwen_edit_pose_carrier_is_strict_and_flux2_pose_stays_on_control() {
    let qwen = |poses: Value| {
        image_generate_job(json!({
            "model": "qwen_image_edit_2511_lightning",
            "mode": "character_image",
            "referenceAssetId": "identity_1",
            "advanced": { "poses": poses }
        }))
    };
    for poses in [Value::Null, json!([])] {
        assert_eq!(
            image_job_candle_lane(&qwen(poses)),
            Some(CandleImageLane::QwenEdit)
        );
    }
    assert_eq!(
        image_job_candle_lane(&qwen(json!([
            { "keypoints": [] },
            { "keypoints": [[0.1, 0.2, 0.9]] }
        ]))),
        Some(CandleImageLane::QwenEdit)
    );
    for malformed in [json!({}), json!(false), json!([{}, null])] {
        assert_eq!(image_job_candle_lane(&qwen(malformed)), None);
    }
    assert_eq!(
        image_job_candle_lane(&qwen(Value::Array(
            (0..=crate::image_request::MAX_JOB_POSES)
                .map(|_| json!({}))
                .collect()
        ))),
        None,
        "oversize pose sets must fail closed"
    );
    assert_eq!(
        image_job_candle_lane(&image_generate_job(json!({
            "model": "qwen_image_edit_2511_lightning",
            "mode": "character_image",
            "referenceAssetIds": ["identity_1", "identity_2"],
            "advanced": { "poses": [{}] }
        }))),
        None,
        "the pose recipe has exactly [identity, skeleton] and cannot drop extra identities"
    );
    assert_eq!(
        image_job_candle_lane(&image_edit_job(json!({
            "model": "qwen_image_edit_2511_lightning",
            "mode": "edit_image",
            "sourceAssetId": "source_1",
            "advanced": { "poses": [{}] }
        }))),
        None,
        "pose sets are a Character Studio shape, never instruction edit"
    );

    let flux_character_pose = image_generate_job(json!({
        "model": "flux2_dev",
        "mode": "character_image",
        "referenceAssetId": "identity_1",
        "advanced": { "poses": [{ "keypoints": [] }] }
    }));
    assert_eq!(
        image_job_candle_lane(&flux_character_pose),
        Some(CandleImageLane::Flux2Control),
        "the edit lane must not swallow FLUX.2 pose controls"
    );
    assert_eq!(
        image_job_candle_lane(&image_edit_job(json!({
            "model": "flux2_dev", "mode": "edit_image", "sourceAssetId": "source",
            "advanced": { "poses": [{}] }
        }))),
        None,
        "FLUX.2 edit plus pose has no scheduler lane and must fail closed"
    );
    for mode in [
        "text_to_image",
        "reference",
        "image_to_image",
        "style_variations",
    ] {
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": "flux2_dev", "mode": mode, "referenceAssetId": "identity",
                "advanced": { "poses": [{}] }
            }))),
            Some(CandleImageLane::Flux2Control),
            "FLUX.2 non-edit pose mode {mode} remains on control"
        );
    }
    for poses in [Value::Null, json!([])] {
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": "flux2_dev",
                "mode": "character_image",
                "referenceAssetId": "identity_1",
                "advanced": { "poses": poses }
            }))),
            Some(CandleImageLane::Flux2Edit)
        );
    }
    for malformed in [json!({}), json!(false), json!([{}, null])] {
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": "flux2_dev",
                "mode": "character_image",
                "referenceAssetId": "identity_1",
                "advanced": { "poses": malformed }
            }))),
            None
        );
    }
    for mode in [
        "text_to_image",
        "reference",
        "image_to_image",
        "style_variations",
    ] {
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": "qwen_image_edit_2511_lightning", "mode": mode,
                "referenceAssetId": "identity", "advanced": { "poses": [{}] }
            }))),
            None,
            "Qwen pose mode {mode} is not an advertised scheduler lane"
        );
    }
}

#[test]
fn candle_routed_models_plain_txt2img_are_eligible() {
    // SDXL/RealVisXL (sc-3678) + the image families wired in sc-5096 — every base txt2img id, now
    // INCLUDING base `z_image` (sc-8679): the registered candle `z_image` base generator makes a plain
    // txt2img `z_image` job candle-eligible, the base sibling of `z_image_turbo`. (Its strict-pose
    // control lane is branched out earlier in `image_job_is_candle_eligible`; its edit shapes reject
    // below — see the conditioning-shape refusal test.)
    for model in CANDLE_ROUTED_MODELS {
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
            "{model} plain txt2img should be candle-eligible"
        );
    }
}

#[test]
fn imported_krea_family_plain_single_file_job_is_candle_eligible() {
    let plain = json!({
        "projectId": "project_1",
        "model": "kreamania_variant4",
        "prompt": "a red fox",
        "modelManifestEntry": {
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": "C:\\SceneWorks\\models\\imports\\kreamania_variant4" }
        }
    });

    assert!(
        !image_request_candle_eligible(
            "kreamania_variant4",
            plain.as_object().expect("payload object")
        ),
        "the builtin id table must not accidentally contain the imported id"
    );
    assert!(
        image_job_is_candle_eligible(&image_generate_job(plain.clone())),
        "the full scheduler must claim a novel imported id through family routing"
    );
    assert!(
        worker_supports_job(&gpu_worker(CANDLE_CAPS), &image_generate_job(plain.clone())),
        "a Candle worker must claim the imported family-routed job"
    );

    let direct_file = json!({
        "model": "kreamania_variant4_direct",
        "modelManifestEntry": {
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "modelPath": "C:\\SceneWorks\\models\\kreamania_variant4.safetensors"
        }
    });
    assert!(image_job_is_candle_eligible(&image_generate_job(
        direct_file
    )));

    // img2img (a single non-edit `referenceAssetId`) is candle-eligible — the candle imported
    // lane serves it (sc-14071, `resolve_img2img_init_generic` → one `Conditioning::Reference`);
    // no adapter needed. This closes the sc-14071 routing gap (the gate previously rejected
    // `referenceAssetId`, stranding candle img2img).
    let mut img2img = plain.clone();
    img2img
        .as_object_mut()
        .expect("payload object")
        .insert("referenceAssetId".to_owned(), json!("reference_1"));
    assert!(
        image_job_is_candle_eligible(&image_generate_job(img2img)),
        "a non-edit imported-Krea img2img referenceAssetId must be candle-eligible (sc-14071)"
    );

    // The Candle imported registrations now inherit LoRA/LoKr, edit, and multi-phase from their
    // resolved Krea providers. Exercise those through the real scheduler gate, including a
    // multi-reference edit and phase-specific adapter plan.
    for extra in [
        json!({ "mode": "edit_image", "sourceAssetId": "source_1" }),
        json!({ "mode": "edit_image", "referenceAssetIds": ["scene_1", "person_1"] }),
        json!({ "loras": [{ "id": "adapter_1" }] }),
        json!({
            "mode": "edit_image",
            "sourceAssetId": "source_1",
            "loras": [{ "id": "edit", "conditioningRole": "image_edit" }]
        }),
        json!({ "advanced": { "phases": [{ "steps": 4 }] } }),
        json!({
            "loras": [{ "id": "adapter_1" }],
            "advanced": { "phases": [{ "steps": 4, "loras": [{ "index": 0, "weight": 0.8 }] }] }
        }),
    ] {
        let mut payload = plain.clone();
        payload
            .as_object_mut()
            .expect("payload object")
            .extend(extra.as_object().expect("extra object").clone());
        assert!(
            image_job_is_candle_eligible(&image_generate_job(payload.clone())),
            "provider-advertised imported-Krea shape must route on candle: {payload}"
        );
    }

    // Real structural gaps remain fail-closed: a plural reference outside edit, mask/character,
    // strict pose (the checked-in candle facts register no `pose` import route), an explicit
    // control map, and a bare non-edit source are not consumed by this lane and must never be
    // silently flattened to txt2img.
    let mut unsupported_shapes = Vec::new();
    for extra in [
        json!({ "mode": "edit_image" }),
        json!({ "referenceAssetIds": ["reference_1"] }),
        json!({ "mode": " edit_image ", "sourceAssetId": "source_1" }),
        json!({ "sourceAssetId": "source_1" }),
        json!({ "maskAssetId": "mask_1" }),
        json!({ "characterId": "character_1" }),
        json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        json!({
            "sourceAssetId": "source_1",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }),
        json!({
            "mode": "edit_image",
            "sourceAssetId": "source_1",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }),
        json!({ "advanced": { "controlImage": "control_1" } }),
        json!({ "advanced": { "controlMode": "pose" } }),
        json!({
            "advanced": {
                "phases": [{ "steps": 4 }],
                "controlImage": "control_1"
            }
        }),
    ] {
        let mut payload = plain.clone();
        payload
            .as_object_mut()
            .expect("payload object")
            .extend(extra.as_object().expect("extra object").clone());
        unsupported_shapes.push(payload);
    }
    for payload in unsupported_shapes {
        assert!(
            !image_job_is_candle_eligible(&image_generate_job(payload.clone())),
            "unsupported imported-Krea conditioning must fail closed on candle: {payload}"
        );
    }

    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "kreamania_variant4",
        "modelManifestEntry": { "family": "krea_2" }
    }))));
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "foreign_import",
        "modelManifestEntry": {
            "family": "z-image",
            "importSourceShape": "transformer_file",
            "paths": { "model": "C:\\SceneWorks\\models\\imports\\foreign" }
        }
    }))));
}

/// A strict-pose set on an imported Krea 2 checkpoint is served on MLX (the pose control branch
/// folds onto the file-loaded DiT) but NOT on candle: the checked-in candle provider facts register
/// no `pose` import route for `krea_2`, and none of its routes advertise `control` conditioning, so
/// the exact provider lookup refuses. This pins the shape of that asymmetry so the candle refusal
/// stays a LOUD, terminal, named gap rather than a job that silently never routes:
///   * `image_job_is_candle_eligible` is false, so no candle worker ever claims it — the claim gate
///     and the resolved imported-provider registration agree;
///   * `candle_supported` returns `Err(reason)`, which is what makes the enforce sweep
///     (`fail_unsupported_candle_jobs`) fail it terminally with a named feature instead of routing
///     it to the grace-window "no candle worker yet" sweep, where it would sit queued forever.
///
/// The MLX half of the same payload is asserted alongside it, so a future change that quietly drops
/// the imported pose surface on BOTH backends cannot pass this test by making the candle half green.
#[test]
fn imported_krea_pose_is_mlx_only_and_candle_refuses_it_terminally() {
    let pose = json!({
        "projectId": "project_1",
        "model": "kreamania_variant4",
        "prompt": "a woman in a red jacket",
        "advanced": { "poses": [{ "id": "pose_1" }] },
        "modelManifestEntry": {
            "family": "krea_2",
            "importSourceShape": "transformer_file",
            "paths": { "model": "C:\\SceneWorks\\models\\imports\\kreamania_variant4" }
        }
    });
    let job = image_generate_job(pose);

    assert!(
        image_job_is_mlx_eligible(&job),
        "MLX serves the imported pose set through the native control entrypoint"
    );
    assert!(
        !image_job_is_candle_eligible(&job),
        "no candle krea_2 pose import route is registered, so it must not claim the job"
    );

    let reason = candle_supported(&job)
        .expect_err("an imported pose set must be a NAMED candle gap, not a silent never-route");
    assert!(
        !reason.feature.trim().is_empty() && !reason.detail.trim().is_empty(),
        "the terminal failure must name the feature and explain it: {reason:?}"
    );
}

#[test]
fn base_z_image_txt2img_is_candle_eligible_but_edit_shapes_are_not() {
    // sc-8679: base `z_image` plain txt2img rides the candle lane (the base sibling of z_image_turbo);
    // its edit / reference / mask conditioning shapes are refused and remain queued (no candle
    // base-z-image edit provider).
    assert!(
        image_request_candle_eligible("z_image", &object(json!({ "prompt": "a red fox" }))),
        "base z_image plain txt2img must be candle-eligible (sc-8679)"
    );
    for payload in [
        json!({ "prompt": "p", "mode": "edit_image", "sourceAssetId": "a" }),
        json!({ "prompt": "p", "referenceAssetId": "a" }),
        json!({ "prompt": "p", "maskAssetId": "a" }),
    ] {
        assert!(
            !image_request_candle_eligible("z_image", &object(payload.clone())),
            "base z_image conditioning shape must fall back to torch: {payload}"
        );
    }
}

#[test]
fn non_candle_families_and_variants_are_never_candle_eligible() {
    // The still-unwired weight/shape variants of wired families (edit ids) plus a genuinely
    // unsupported plain-txt2img image id (`pulid_flux_dev` — its only candle lane is the bespoke
    // character-reference path, so a PLAIN txt2img prompt has no native route) all remain queued.
    // (chroma / kolors / sensenova ARE candle-routed now — sc-5484 / sc-5576 — for
    // txt2img; the FLUX.2-klein `_kv` / `_true_v2` weight variants are too — sc-7459 — see the
    // dedicated test below. `bernini_image` is now candle-routed off-Mac too — sc-10996. Base
    // `sana_1600m` AND the Sprint distill `sana_sprint_1600m` are candle-routed off-Mac too —
    // sc-11780 / sc-11781 — see `sana_candle_txt2img_routes_to_candle` below.)
    for model in ["pulid_flux_dev", "z_image_edit", "qwen_image_edit"] {
        assert!(
            !image_request_candle_eligible(model, &object(json!({ "prompt": "p" }))),
            "{model} must fall back to the Python worker"
        );
    }
}

#[test]
fn sana_candle_txt2img_and_single_reference_img2img_route_to_candle() {
    // sc-11780 (epic 8485): base `sana_1600m` plain txt2img rides the candle lane (the
    // `candle-gen-sana` provider, candle-gen #495 — the whole `Efficient-Large-Model/
    // Sana_1600M_1024px_diffusers` snapshot). sc-11781: the CFG-free SANA-Sprint distill
    // `sana_sprint_1600m` rides it too (the `candle-gen-sana` Sprint pipeline, candle-gen #498 — the
    // whole `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` snapshot). Both also accept one
    // singular reference for non-edit img2img; edit/control/multiple-reference, LoRA, and quant shapes
    // are refused rather than silently ignored.
    for model in ["sana_1600m", "sana_sprint_1600m"] {
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
            "{model} plain txt2img must be candle-eligible (sc-11780 / sc-11781)"
        );
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": model,
                "prompt": "a red fox",
                "referenceAssetId": "reference-1",
                "advanced": { "strength": 0.5 }
            }))),
            Some(CandleImageLane::SanaImg2Img)
        );
        for empty_carriers in [
            json!({
                "model": model,
                "prompt": "p",
                "controls": [],
                "controlnets": [],
                "referenceAssetIds": []
            }),
            json!({
                "model": model,
                "referenceAssetId": "reference-1",
                "controls": null,
                "controlnets": [],
                "referenceAssetIds": [],
                "loras": [],
                "sourceAssetId": " ",
                "maskAssetId": null,
                "advanced": {
                    "strength": 0.5,
                    "poses": [],
                    "phases": null,
                    "controlMode": null,
                    "controlImage": null,
                    "controlScale": null,
                    "controlWeights": null,
                    "convRot": null,
                    "quantTier": null,
                    "mlxQuantize": 0
                }
            }),
        ] {
            assert!(
                image_job_candle_lane(&image_generate_job(empty_carriers.clone())).is_some(),
                "{model} empty/null optional carriers preserve txt2img/img2img: {empty_carriers}"
            );
        }
        for payload in [
            json!({ "prompt": "p", "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "prompt": "p", "maskAssetId": "a" }),
            json!({ "prompt": "p", "loras": [{ "path": "x", "weight": 0.8 }] }),
            json!({ "prompt": "p", "advanced": { "mlxQuantize": 4 } }),
            json!({ "prompt": "p", "advanced": { "controlMode": "canny" } }),
            json!({ "prompt": "p", "advanced": { "controlImage": "control-1" } }),
            json!({ "prompt": "p", "advanced": { "controlScale": 0.9 } }),
            json!({ "prompt": "p", "advanced": { "controlWeights": { "overlayId": "overlay-1" } } }),
            json!({ "prompt": "p", "advanced": { "convRot": true } }),
            json!({ "prompt": "p", "advanced": { "quantTier": "nvfp4" } }),
        ] {
            assert!(
                !image_request_candle_eligible(model, &object(payload.clone())),
                "{model} conditioning/adapter/quant shape must fall back to torch: {payload}"
            );
        }

        for malformed in [
            json!({ "model": model, "referenceAssetIds": ["a"] }),
            json!({ "model": model, "referenceAssetId": 7 }),
            json!({ "model": model, "referenceAssetId": "a", "sourceAssetId": "b" }),
            json!({ "model": model, "referenceAssetId": "a", "maskAssetId": "m" }),
            json!({ "model": model, "referenceAssetId": "a", "loras": [{ "path": "x" }] }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "poses": [{}] } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "phases": [{}] } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "mlxQuantize": 4 } }),
            json!({ "model": model, "referenceAssetId": "a", "referenceAssetIds": [7] }),
            json!({ "model": model, "referenceAssetId": "a", "referenceAssetIds": 7 }),
            json!({ "model": model, "referenceAssetId": "a", "controls": [{}] }),
            json!({ "model": model, "referenceAssetId": "a", "controls": 7 }),
            json!({ "model": model, "referenceAssetId": "a", "controlnets": [{}] }),
            json!({ "model": model, "referenceAssetId": "a", "controlnets": "invalid" }),
            json!({ "model": model, "referenceAssetId": "a", "loras": 7 }),
            json!({ "model": model, "referenceAssetId": "a", "sourceAssetId": 7 }),
            json!({ "model": model, "referenceAssetId": "a", "maskAssetId": {} }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "poses": 7 } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "phases": {} } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "mlxQuantize": {} } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "controlMode": "canny" } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "controlImage": "control-1" } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "controlScale": 0.9 } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "controlWeights": { "overlayId": "overlay-1" } } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "convRot": true } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": { "quantTier": "nvfp4" } }),
            json!({ "model": model, "referenceAssetId": "a", "advanced": 7 }),
        ] {
            assert_eq!(
                image_job_candle_lane(&image_generate_job(malformed.clone())),
                None,
                "{model} malformed img2img shape must be rejected: {malformed}"
            );
        }
    }
}

#[test]
fn sana_absent_reference_carrier_routes_plain_txt2img_to_candle() {
    // sc-20525: EVERY real SANA text-to-image submission was enforce-failed `candle_unsupported`
    // off-Mac. The sc-18475 guard in `image_request_candle_eligible` read the singular carrier with
    // an inline `match value.as_str()` whose `None` arm meant "malformed" — but `Value::Null` also
    // lands in that arm, and `null` is exactly how the API normalizes every unset optional asset
    // carrier before the job is stored. So the normal absent-carrier encoding was classified as a
    // malformed conditioning shape and the job never reached the txt2img lane. The gate now shares
    // `has_nonempty_or_malformed_string` with `sana_has_unsupported_carrier`, which is the same fix
    // the MLX twin already carries (sc-19712 F-1) and the same convention the worker uses (a blank
    // asset id is treated as absent).
    for model in ["sana_1600m", "sana_sprint_1600m"] {
        // 1. Absent, `null`, and blank are all "not supplied" -> the plain candle txt2img lane.
        for absent in [
            json!({ "model": model, "prompt": "a red fox", "mode": "text_to_image" }),
            json!({
                "model": model,
                "prompt": "a red fox",
                "mode": "text_to_image",
                "referenceAssetId": null
            }),
            json!({
                "model": model,
                "prompt": "a red fox",
                "mode": "text_to_image",
                "referenceAssetId": "   "
            }),
            // The exact shape the web UI submits for a plain SANA render: mode `text_to_image`,
            // every optional carrier explicitly `null`/empty, and no quant override.
            json!({
                "model": model,
                "prompt": "a red fox",
                "mode": "text_to_image",
                "referenceAssetId": null,
                "referenceAssetIds": [],
                "sourceAssetId": null,
                "maskAssetId": null,
                "controls": null,
                "controlnets": [],
                "loras": [],
                "advanced": {
                    "poses": [],
                    "phases": null,
                    "controlMode": null,
                    "controlImage": null,
                    "controlScale": null,
                    "controlWeights": null,
                    "convRot": null,
                    "quantTier": null,
                    "mlxQuantize": null
                }
            }),
        ] {
            assert_eq!(
                image_job_candle_lane(&image_generate_job(absent.clone())),
                Some(CandleImageLane::TextToImage),
                "{model} plain txt2img must reach the candle txt2img lane: {absent}"
            );
            assert!(
                image_request_candle_eligible(model, &object(absent.clone())),
                "{model} plain txt2img must be candle-eligible: {absent}"
            );
        }
        // 2. A populated carrier still selects the specialized SANA reference lane.
        assert_eq!(
            image_job_candle_lane(&image_generate_job(json!({
                "model": model,
                "prompt": "a red fox",
                "referenceAssetId": "reference-1"
            }))),
            Some(CandleImageLane::SanaImg2Img),
            "{model} non-empty referenceAssetId must select the SANA reference lane"
        );
        // 3. A malformed carrier still fails closed rather than being reinterpreted as txt2img.
        for malformed in [
            json!(7),
            json!({ "id": "reference-1" }),
            json!(["reference-1"]),
        ] {
            let payload = json!({
                "model": model,
                "prompt": "a red fox",
                "mode": "text_to_image",
                "referenceAssetId": malformed
            });
            assert_eq!(
                image_job_candle_lane(&image_generate_job(payload.clone())),
                None,
                "{model} malformed referenceAssetId must stay unclaimed: {payload}"
            );
            assert!(
                !image_request_candle_eligible(model, &object(payload.clone())),
                "{model} malformed referenceAssetId must not be candle-eligible: {payload}"
            );
        }
    }
}

#[test]
fn flux2_klein_weight_variants_route_txt2img_to_candle() {
    // sc-7459 (epic 6564 story 3): both klein weight variants serve plain txt2img on the candle lane
    // via the shared `flux2_klein_9b` loader — a weights swap, not a new arch.
    for model in ["flux2_klein_9b_kv", "flux2_klein_9b_true_v2"] {
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "a red fox" }))),
            "{model} plain txt2img should be candle-eligible"
        );
    }
    // ...but their reference/edit shapes are NOT in scope (txt2img weight parity only). The `_kv`
    // checkpoint's whole point is the reference-edit KV-cache accel; candle has no klein edit path,
    // so the request is refused and remains queued.
    for payload in [
        json!({ "referenceAssetId": "a" }),
        json!({ "mode": "edit_image", "sourceAssetId": "a" }),
    ] {
        assert!(
            !image_request_candle_eligible("flux2_klein_9b_kv", &object(payload.clone())),
            "flux2_klein_9b_kv conditioning shape must fall back to torch: {payload}"
        );
    }
}

#[test]
fn new_candle_families_conditioning_shapes_fall_back_to_torch() {
    // These cases exercise unsupported conditioning on the generic base predicate. Specialized edit,
    // reference, identity, and control routes are asserted separately; anything not claimed by one
    // of them must be refused rather than silently rendered as unconditioned text-to-image.
    let cases = [
        (
            "z_image_turbo",
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        ),
        ("flux_dev", json!({ "referenceAssetId": "a" })),
        (
            "qwen_image",
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ),
        // NB: `flux2_klein_9b` + `edit_image` is NOT here — sc-5487 wired it to the candle `Flux2Edit`
        // lane (asserted via `image_job_is_candle_eligible` in `candle_worker_claims_*`), like SDXL
        // edit. The txt2img gate still rejects it (it rejects all `edit_image`), but the bespoke
        // candle edit lane claims it at the router level.
        // These assertions exercise only the generic T2I gate. Kolors and SenseNova conditioning is
        // intentionally rejected here and claimed by their dedicated routes at the full scheduler.
        (
            "chroma1_hd",
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        ),
        (
            "kolors",
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        ),
        ("kolors", json!({ "referenceAssetId": "a" })),
        (
            "kolors",
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ),
        (
            "sensenova_u1_8b",
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        ),
        ("sensenova_u1_8b_fast", json!({ "referenceAssetId": "a" })),
    ];
    for (model, payload) in cases {
        assert!(
            !image_request_candle_eligible(model, &object(payload.clone())),
            "{model} conditioning shape must fall back to torch: {payload}"
        );
    }
}

#[test]
fn flux1_lora_stays_on_candle() {
    for model in ["flux_schnell", "flux_dev"] {
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "prompt": "x", "loras": [{ "name": "x" }] }))
            ),
            "{model} LoRA must stay on Candle"
        );
    }
}

#[test]
fn ideogram_candle_txt2img_and_edit_route_to_candle() {
    // sc-6597 (epic 6561): `ideogram_4` + `ideogram_4_turbo` route to the candle lane for plain
    // text-to-image via the generic `image_request_candle_eligible` gate. sc-6598: img2img / Remix +
    // mask inpaint / outpaint now route to candle too — via the bespoke `ideogram_edit_candle_eligible`
    // branch in `image_job_is_candle_eligible` (the generic gate stays txt2img-only, like every other
    // candle edit family). A pure `referenceAssetId` (IP-Adapter — no candle Ideogram path) is
    // refused and remains queued.
    for model in ["ideogram_4", "ideogram_4_turbo"] {
        // Plain txt2img → the generic gate.
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "an aurora" }))),
            "{model} plain txt2img must be candle-eligible"
        );
        // sc-9607/sc-9983: a Q8/Q4 tier-select stays on candle (ideogram is in CANDLE_QUANT_MODELS —
        // the packed q4/q8 turnkeys load off-Mac; the `mlxQuantize` value picks the subdir).
        for bits in [8, 4] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "an aurora", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} tier-select should stay on candle"
            );
        }
        // Edit shapes → the bespoke dispatcher branch (img2img, inpaint, outpaint all need a source).
        for payload in [
            json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a", "maskAssetId": "m" }),
            json!({ "model": model, "mode": "edit_image", "sourceAssetId": "a", "fit_mode": "outpaint" }),
        ] {
            assert!(
                ideogram_edit_candle_eligible(&object(payload.clone())),
                "{model} edit shape must be candle-eligible: {payload}"
            );
            assert!(
                image_job_is_candle_eligible(&image_edit_job(payload.clone())),
                "{model} edit job must route to candle: {payload}"
            );
            // The generic txt2img gate still rejects the edit_image family (the bespoke lane handles it).
            assert!(!image_request_candle_eligible(model, &object(payload)));
        }
        // `edit_image` WITHOUT a source → nothing to edit → not this lane.
        assert!(!ideogram_edit_candle_eligible(&object(json!({
            "model": model, "mode": "edit_image"
        }))));
        // The raw txt2img gate still rejects any `referenceAssetId` (it stays txt2img-only). A
        // text_to_image reference is the `ui.img2img` "Image reference" tile, handled by the bespoke
        // `ideogram_img2img_candle_eligible` branch in the dispatcher (covered by
        // `ideogram_img2img_routes_to_candle`), NOT the generic gate.
        assert!(!image_request_candle_eligible(
            model,
            &object(json!({ "referenceAssetId": "a" }))
        ));
    }
}

#[test]
fn ideogram_img2img_routes_to_candle() {
    // sc-10261 (epic 8588): the candle parity of the MLX Ideogram generic img2img arm (sc-10192). An
    // `ideogram_4` / `ideogram_4_turbo` job in a non-edit mode with a `referenceAssetId` is the
    // `ui.img2img` "Image reference" tile — a single `Conditioning::Reference` with NO `Mask`, which
    // the candle `candle-gen-ideogram` pipeline denoises as plain img2img (`resolve_edit` →
    // `prepare_edit` with `mask = None`). Branched before the txt2img gate (which rejects any
    // `referenceAssetId`), disjoint from the `edit_image` Remix/inpaint lane. No worker/candle-gen
    // change — the worker `generate_candle_stream` already resolves the init generically (sc-10134).
    for model in ["ideogram_4", "ideogram_4_turbo"] {
        let img2img = json!({
            "model": model,
            "referenceAssetId": "asset_1",
            "advanced": { "strength": 0.6 }
        });
        assert!(
            image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
            "{model} img2img (referenceAssetId, non-edit) must be candle-eligible (sc-10261)"
        );
        // The eligibility predicate: a non-edit reference is img2img; an `edit_image` reference is the
        // Remix/inpaint edit lane, NOT this img2img arm.
        assert!(ideogram_img2img_candle_eligible(&object(json!({
            "model": model, "referenceAssetId": "asset_1"
        }))));
        assert!(!ideogram_img2img_candle_eligible(&object(json!({
            "model": model, "mode": "edit_image", "referenceAssetId": "asset_1"
        }))));
        // A blank/absent reference is plain txt2img, not img2img.
        assert!(!ideogram_img2img_candle_eligible(&object(json!({
            "model": model, "referenceAssetId": "  "
        }))));
        assert!(!ideogram_img2img_candle_eligible(&object(
            json!({ "model": model })
        )));
    }
}

#[test]
fn bernini_candle_txt2img_and_i2i_route_to_candle() {
    // sc-10996 (epic 6562): the candle parity of the MLX `bernini_image` still lane. `bernini_image`
    // is in `CANDLE_ROUTED_MODELS`, so plain t2i routes to candle via the generic
    // `image_request_candle_eligible` gate; i2i (`edit_image` + a `sourceAssetId`) routes via the
    // bespoke `bernini_image_edit_candle_eligible` branch in `image_job_is_candle_eligible` (the
    // dedicated `generate_candle_bernini_image_stream` worker lane, `frames:1`). Mirrors the MLX
    // `bernini_image_mlx_eligible` predicate exactly.
    //
    // Plain t2i → the generic gate.
    assert!(
        image_request_candle_eligible(
            "bernini_image",
            &object(json!({ "prompt": "a marble bust" }))
        ),
        "bernini_image plain t2i must be candle-eligible"
    );
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "bernini_image", "prompt": "a marble bust"
    }))));
    // i2i → the bespoke edit branch (needs a source).
    let i2i = json!({ "model": "bernini_image", "mode": "edit_image", "sourceAssetId": "asset_1" });
    assert!(bernini_image_edit_candle_eligible(&object(i2i.clone())));
    assert!(
        image_job_is_candle_eligible(&image_edit_job(i2i.clone())),
        "bernini_image i2i job must route to candle"
    );
    // The generic txt2img gate still rejects the edit_image family (the bespoke lane handles it).
    assert!(!image_request_candle_eligible(
        "bernini_image",
        &object(i2i)
    ));
    // `edit_image` WITHOUT a source → nothing to edit → not this lane (mirrors `bernini_image_mlx_eligible`).
    assert!(!bernini_image_edit_candle_eligible(&object(json!({
        "model": "bernini_image", "mode": "edit_image"
    }))));
    assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
        "model": "bernini_image", "mode": "edit_image"
    }))));
    // sc-11003 publishes bf16/q8/q4 subdirectories in the off-Mac snapshot, and the worker's shared
    // Bernini resolver selects them for both the image and video lanes. Quant requests therefore stay
    // on Candle instead of being bounced by a stale catalog flag.
    for bits in [4, 8] {
        assert!(image_request_candle_eligible(
            "bernini_image",
            &object(json!({
                "prompt": "a marble bust",
                "advanced": { "mlxQuantize": bits }
            }))
        ));
    }
}

#[test]
fn boogu_text_to_image_and_edit_route_to_candle() {
    // sc-7524 (epic 6831): the candle parity of `boogu_text_to_image_and_edit_route_to_mlx`. The
    // three Boogu ids are in `CANDLE_ROUTED_MODELS`; Base + Turbo are pure txt2img (the generic gate),
    // and `boogu_image_edit`'s `edit_image` shape routes via the bespoke `boogu_edit_candle_eligible`
    // branch (the source `Reference` is resolved in-lane by `generate_candle_stream`, like Ideogram).
    for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
        // Plain txt2img → the generic gate (the edit checkpoint can also T2I, mirroring MLX).
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "a red panda" }))),
            "{model} plain txt2img must be candle-eligible"
        );
    }
    // Edit (source instruction) is the Edit checkpoint's capability ONLY.
    let edit_payload =
        |model: &str| json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" });
    // `boogu_image_edit` + edit_image + source → the bespoke branch claims it for candle.
    assert!(boogu_edit_candle_eligible(&object(edit_payload(
        "boogu_image_edit"
    ))));
    assert!(image_job_is_candle_eligible(&image_edit_job(edit_payload(
        "boogu_image_edit"
    ))));
    // The generic txt2img gate still rejects the edit_image family (the bespoke lane handles it).
    assert!(!image_request_candle_eligible(
        "boogu_image_edit",
        &object(edit_payload("boogu_image_edit"))
    ));
    // Base / Turbo do NOT edit — an edit_image job on them is not candle-eligible (no edit lane; the
    // generic gate rejects edit_image and the boogu edit branch is gated to `boogu_image_edit`).
    assert!(!image_job_is_candle_eligible(&image_edit_job(
        edit_payload("boogu_image")
    )));
    assert!(!image_job_is_candle_eligible(&image_edit_job(
        edit_payload("boogu_image_turbo")
    )));
    // sc-7645: the multi-image picker sends plural `referenceAssetIds` (no `sourceAssetId`) — the
    // bespoke branch still claims it for candle (the Boogu DiT packs up to 5 references).
    assert!(boogu_edit_candle_eligible(&object(json!({
        "model": "boogu_image_edit", "mode": "edit_image",
        "referenceAssetIds": ["a", "b"]
    }))));
    // `edit_image` WITHOUT a source → nothing to edit → not this lane.
    assert!(!boogu_edit_candle_eligible(&object(json!({
        "model": "boogu_image_edit", "mode": "edit_image"
    }))));
    // An empty plural list with no `sourceAssetId` is also nothing to edit.
    assert!(!boogu_edit_candle_eligible(&object(json!({
        "model": "boogu_image_edit", "mode": "edit_image", "referenceAssetIds": []
    }))));
    // sc-9607/sc-9983: a Q8/Q4 tier-select now STAYS on candle (boogu is in CANDLE_QUANT_MODELS — the
    // packed q4/q8 turnkeys load off-Mac, the `mlxQuantize` value picks the subdir). A LoRA still
    // defers (boogu advertises no inference LoRA on candle).
    for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 8 } }))
            ),
            "{model} Q8 tier-select should stay on candle"
        );
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 4 } }))
            ),
            "{model} Q4 tier-select should stay on candle"
        );
        assert!(
            !image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} with a LoRA must defer to torch (no candle inference LoRA)"
        );
    }
}

#[test]
fn boogu_base_and_turbo_img2img_route_to_candle() {
    // sc-11786 (epic 8588): a `boogu_image` / `boogu_image_turbo` job in a non-edit mode with a
    // `referenceAssetId` is now the REGISTRY img2img path — the candle Base/Turbo generators advertise
    // `Reference` and VAE-encode it into the reduced-schedule denoise. Branched before the txt2img gate
    // (which rejects any `referenceAssetId`), disjoint from the `boogu_image_edit` instruction-edit lane.
    for model in ["boogu_image", "boogu_image_turbo"] {
        let img2img = json!({
            "model": model,
            "referenceAssetId": "asset_1",
            "advanced": { "strength": 0.6 }
        });
        assert!(
            image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
            "{model} img2img (referenceAssetId, non-edit) must be candle-eligible (sc-11786)"
        );
        // The eligibility predicate: a non-edit reference is img2img; an `edit_image` reference is NOT
        // (that is the `boogu_image_edit` instruction-edit lane, a different engine id).
        assert!(boogu_img2img_candle_eligible(&object(json!({
            "model": model, "referenceAssetId": "asset_1"
        }))));
        assert!(!boogu_img2img_candle_eligible(&object(json!({
            "model": model, "mode": "edit_image", "referenceAssetId": "asset_1"
        }))));
        // A blank/absent reference is plain txt2img, not img2img.
        assert!(!boogu_img2img_candle_eligible(&object(json!({
            "model": model, "referenceAssetId": "  "
        }))));
        assert!(!boogu_img2img_candle_eligible(&object(
            json!({ "model": model })
        )));
    }
    // Base/Turbo carry NO separate edit lane, so an `edit_image` job on them stays ineligible (the
    // img2img branch is gated to a non-edit mode; the edit branch is gated to `boogu_image_edit`).
    for model in ["boogu_image", "boogu_image_turbo"] {
        assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
            "model": model, "mode": "edit_image", "sourceAssetId": "asset_1"
        }))));
    }
}

#[test]
fn explicit_quantization_routes_only_to_advertised_candle_tiers() {
    // sc-5099: a candle provider that advertises NO quant (supported_quants: &[]) must route an
    // explicit `advanced.mlxQuantize > 0` is refused rather than silently running dense.
    // `flux_schnell` remains an image-side dense-only counter-example; Chroma's shared q4/q8
    // turnkey is covered by its own complete three-id matrix immediately after this test.
    // NOTE: qwen_image USED to be a dense-only counter-example here; sc-11020 moved it to
    // CANDLE_QUANT_MODELS (its turnkey q4/q8 packed tiers load off-Mac), so its quant tier-select now
    // STAYS on candle — covered by `qwen_image_quant_and_lora_stay_on_candle`.
    // sc-18478: Wan TI2V-5B now owns native Candle q4/q8 tiers as well as dense.
    assert!(video_request_candle_eligible(
        "wan_2_2",
        &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } }))
    ));
    // Mochi remains a dense-only Candle video family and is still a negative counter-example.
    assert!(!video_request_candle_eligible(
        "mochi_1",
        &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } }))
    ));
    // Dense (<= 0) or absent quant leaves a dense candle family on its native path → still eligible.
    assert!(image_request_candle_eligible(
        "chroma1_hd",
        &object(json!({ "advanced": { "mlxQuantize": 0 } }))
    ));
    assert!(image_request_candle_eligible(
        "chroma1_hd",
        &object(json!({ "advanced": { "steps": 30 } }))
    ));
}

#[test]
fn chroma_turnkey_q4_q8_tiers_stay_on_candle_without_adapters_or_conditioning() {
    // sc-20741: each Chroma provider id resolves its own hosted standard-tier snapshot, but all
    // three share the packed candle-gen-chroma loader. Test BOTH published packed directories for
    // EVERY id rather than proving one sibling and assuming the catalog aliases the others.
    for model in ["chroma1_base", "chroma1_flash", "chroma1_hd"] {
        for bits in [4, 8] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "a red fox", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} q{bits} packed tier-select must stay on Candle"
            );
        }

        assert!(
            !image_request_candle_eligible(
                model,
                &object(json!({ "prompt": "a red fox", "advanced": { "mlxQuantize": 6 } }))
            ),
            "{model} must refuse an unpublished q6 tier instead of silently remapping it"
        );

        // Chroma's packed-tier admission is plain txt2img only. Its catalog still lists LoRA
        // compatibility for other routes, but no Candle evidence proves composing an adapter with
        // the q4/q8 packed codes; refuse the adapter rather than silently dropping or dense-folding it.
        assert!(
            !image_request_candle_eligible(
                model,
                &object(json!({
                    "advanced": { "mlxQuantize": 4 },
                    "loras": [{ "name": "unverified", "path": "/tmp/unverified.safetensors" }]
                }))
            ),
            "{model} must refuse an unproven packed-tier LoRA composition"
        );

        // The native Candle Chroma route is text-to-image only; an image carrier must not be
        // reinterpreted as an unconditioned q4/q8 request.
        assert!(
            !image_request_candle_eligible(
                model,
                &object(json!({
                    "advanced": { "mlxQuantize": 8 },
                    "referenceAssetId": "asset_1"
                }))
            ),
            "{model} must refuse a conditioned packed-tier shape"
        );
    }
}

#[test]
fn qwen_image_quant_and_lora_stay_on_candle() {
    // sc-11020 (epic 9083): base `qwen_image` is a turnkey packed-quant candle family — its q4/q8/bf16
    // subdirs load off-Mac via `standard_tier_subdir` (sc-8669) and the tiers are GPU-measured
    // (sc-10969), so a `mlxQuantize` tier-select now STAYS on candle instead of enforce-failing
    // `candle_unsupported` at routing (the routing half sc-9983 flipped for krea/ideogram/boogu but
    // missed qwen). sc-18477 also admits its native Candle LoRA/LoKr path.
    for bits in [4, 8] {
        assert!(
            image_request_candle_eligible(
                "qwen_image",
                &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": bits } }))
            ),
            "qwen_image Q{bits} tier-select should stay on candle"
        );
    }
    // A plain (no tier-select) job is of course still eligible.
    assert!(image_request_candle_eligible(
        "qwen_image",
        &object(json!({ "prompt": "x" }))
    ));
    // A LoRA request stays on Candle and reaches the strict adapter loader.
    assert!(image_request_candle_eligible(
        "qwen_image",
        &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
    ));
}

#[test]
fn z_image_quant_tier_select_stays_on_candle() {
    // Z-Image's q4/q8/bf16 turnkeys are already packed. `mlxQuantize` selects the directory; it does
    // not ask candle-gen-z-image to quantize dense weights at load time. The engine intentionally
    // keeps `supported_quants: []` for that unsupported on-the-fly operation while packed-detecting
    // the chosen tier from its component configs. Routing must therefore admit the tier-select for
    // both Turbo and the base model instead of enforce-failing it as a conditioned shape.
    for model in ["z_image_turbo", "z_image"] {
        for bits in [4, 8] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "a red fox", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} packed-tier select should stay on candle"
            );
        }
        assert!(image_request_candle_eligible(
            model,
            &object(json!({ "prompt": "a red fox", "advanced": { "mlxQuantize": 0 } }))
        ));
    }
}

#[test]
fn flux2_turnkey_quant_and_lora_stay_on_candle() {
    // sc-10222 (epic 9083): the LAST families carrying the "engine wired, router half missed" skew
    // (sc-9983 krea/ideogram/boogu, sc-11020 qwen). `flux2_klein_9b`/`_kv`/`flux2_dev` are worker
    // `STANDARD_TIER_MODELS` members whose `SceneWorks/flux2-*-mlx` turnkeys ship q4/q8/bf16 packed
    // subdirs, resolved by `standard_tier_subdir` and MEASURED in the manifest `candle.vramGbByTier`
    // — but `candle_quant` was `false`, so an explicit tier-select enforce-failed `candle_unsupported`
    // at routing before ever reaching the worker. FLUX.2-klein has no torch path at all, and dev's own
    // fit-gate asks a 48 GB user for q4 (44.0 GB) over the q8 default (70.7) — the only tier that fits
    // was the one tier that could not be routed.
    for model in ["flux2_klein_9b", "flux2_klein_9b_kv", "flux2_dev"] {
        for bits in [4, 8] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} tier-select should stay on candle (sc-10222)"
            );
        }
        // An explicit bf16 pick (<= 0) and a plain job were always eligible — unchanged.
        for advanced in [json!({ "mlxQuantize": 0 }), json!({ "steps": 28 })] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": advanced }))
                ),
                "{model} dense/plain shape must stay candle-eligible"
            );
        }
        // sc-18477: every published FLUX.2 tier applies LoRA/LoKr natively.
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} LoRA must stay on Candle"
        );
    }
    // `_true_v2` is deliberately NOT flipped: the wikeeyang fine-tune installs by convert-at-install
    // into a FLAT `modelPath` dir with no q4/q8/bf16 tier matrix, so there is no tier for a pick to
    // select. Its plain txt2img stays candle-eligible; only the tier-select defers.
    assert!(image_request_candle_eligible(
        "flux2_klein_9b_true_v2",
        &object(json!({ "prompt": "x" }))
    ));
    assert!(!image_request_candle_eligible(
        "flux2_klein_9b_true_v2",
        &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 4 } }))
    ));
    assert!(image_request_candle_eligible(
        "flux2_klein_9b_true_v2",
        &object(json!({ "prompt": "x", "loras": [{ "name": "x" }] }))
    ));
}

#[test]
fn sensenova_family_quant_tier_select_stays_on_candle() {
    // sc-14249 (epic 9083): `candle-gen-sensenova` was dense-f32-only — it mmapped its backbone at
    // a hardcoded F32 and hard-rejected `spec.quantize`, so the candle lane could read only the
    // `bf16/` tier, at DOUBLE its on-disk size (a measured 70.5 GB peak on sm_120 for a 32.7 GiB
    // checkpoint — a 96 GB-card feature). It now packed-detects each backbone projection, so the
    // turnkey's q4/q8 tiers load natively and a tier-select must reach the worker instead of
    // enforce-failing `candle_unsupported` at routing.
    for model in [
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_infographic_v3",
        "sensenova_u1_8b_infographic_v3_fast",
    ] {
        for bits in [4, 8] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} tier-select should stay on candle (sc-14249)"
            );
        }
        // The dense/plain shapes were always eligible — unchanged.
        assert!(image_request_candle_eligible(
            model,
            &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": 0 } }))
        ));
        // Quant and LoRA stay decoupled: the family advertises no candle inference LoRA (the
        // fast ids' distill LoRA is merged internally by the loader, never user-supplied).
        assert!(
            !image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} must still defer a user LoRA"
        );
    }
}

#[test]
fn sdxl_family_quant_and_lora_stay_on_candle() {
    // sc-10767 (epic 9083): the SDXL family advertises Q4/Q8 packed tiers (candle-gen sc-9416/9527)
    // AND inference LoRA/LoKr on a packed tier (sc-9528), so a quant tier-select AND a LoRA both
    // stay on the candle lane rather than deferring to the retired torch fallback. Mirrors the
    // boogu/lens quant-stays coverage; the inverse of the old dense-only behavior.
    //
    // sc-10812: realvisxl_lightning (the few-step distilled sibling on the SAME `sdxl` engine /
    // descriptor) joins the family — same quant + LoRA stay-on-candle for its plain txt2img shape.
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "realvisxl_lightning",
    ] {
        for bits in [8, 4] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "prompt": "x", "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} tier-select should stay on candle (sc-10767)"
            );
        }
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} with a LoRA should stay on candle (sc-10767)"
        );
    }
}

#[test]
fn lens_quant_and_lora_stay_on_the_candle_lane() {
    // sc-5126: Lens / Lens-Turbo advertise Q4/Q8 + LoRA/LoKr, so — UNLIKE the sc-3675/sc-5096
    // families — a quant request or a LoRA stays on candle; the lane maps both into the LoadSpec.
    for model in ["lens", "lens_turbo"] {
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "advanced": { "mlxQuantize": 8 } }))
            ),
            "{model} Q8 request should stay on candle"
        );
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "advanced": { "mlxQuantize": 4 } }))
            ),
            "{model} Q4 request should stay on candle"
        );
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} with a LoRA should stay on candle"
        );
    }
}

#[test]
fn lens_conditioning_shapes_fall_back_to_torch() {
    // Lens is pure T2I (the port has no img2img/edit/reference/ControlNet), so every conditioning
    // shape is refused and remains queued — quant/LoRA being allowed does not widen this.
    let cases = [
        json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        json!({ "referenceAssetId": "a" }),
        json!({ "maskAssetId": "m" }),
        json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
    ];
    for model in ["lens", "lens_turbo"] {
        for case in &cases {
            assert!(
                !image_request_candle_eligible(model, &object(case.clone())),
                "{model} conditioning shape must fall back to torch: {case}"
            );
        }
    }
}

#[test]
fn sd3_5_quant_and_lora_stay_on_candle_but_conditioning_defers() {
    // sc-7880 (epic 7982) + sc-18477: the Candle SD3.5 descriptor advertises Q4/Q8 and native
    // LoRA/LoKr. Quant and adapter requests stay on Candle; unsupported conditioning still defers.
    for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
        // Plain txt2img is eligible.
        assert!(
            image_request_candle_eligible(model, &object(json!({ "prompt": "a misty fjord" }))),
            "{model} plain txt2img should be candle-eligible"
        );
        // Q8 / Q4 requests stay on candle (descriptor-gated quant, resolved worker-side).
        for bits in [8, 4] {
            assert!(
                image_request_candle_eligible(
                    model,
                    &object(json!({ "advanced": { "mlxQuantize": bits } }))
                ),
                "{model} Q{bits} request should stay on candle"
            );
        }
        // A LoRA stays on the strict native adapter path.
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
            ),
            "{model} with a LoRA must stay on Candle"
        );
        // Every conditioning shape defers (txt2img only).
        for case in [
            json!({ "mode": "edit_image", "sourceAssetId": "a" }),
            json!({ "referenceAssetId": "a" }),
            json!({ "maskAssetId": "m" }),
            json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
        ] {
            assert!(
                !image_request_candle_eligible(model, &object(case.clone())),
                "{model} conditioning shape must fall back to torch: {case}"
            );
        }
    }
}

#[test]
fn krea_lora_and_quant_stay_on_candle_but_conditioning_defers() {
    // sc-7836 (epic 7565 P4) + sc-9607/sc-9983 (epic 9083): the candle `candle-gen-krea` descriptor
    // advertises supports_lora/supports_lokr: true (it merges a `krea_2_raw`-trained adapter at Turbo
    // inference) AND, since sc-9607, `supported_quants: [Q4, Q8]` (a no-op on the already-packed q4/q8
    // turnkey subdir), so BOTH a LoRA and a Q8/Q4 tier-select stay on the candle lane (Krea is in
    // CANDLE_QUANT_LORA_MODELS). Only the conditioning shapes (edit/reference/mask/pose) defer to the
    // queue. Regression guard for the two missed router un-gates: before sc-7836 a Krea LoRA, and
    // before sc-9983 a Krea Q8/Q4, each hit the native-support gap off-Mac.
    let model = "krea_2_turbo";
    // Plain txt2img is eligible.
    assert!(
        image_request_candle_eligible(model, &object(json!({ "prompt": "an emerald forest" }))),
        "{model} plain txt2img should be candle-eligible"
    );
    // A LoRA stays on candle (descriptor-gated adapter merge, resolved worker-side).
    assert!(
        image_request_candle_eligible(
            model,
            &object(json!({ "loras": [{ "name": "x", "path": "/x.safetensors" }] }))
        ),
        "{model} with a LoRA should stay on candle"
    );
    // sc-9607/sc-9983: a Q8 / Q4 tier-select now STAYS on candle (the packed turnkey loads off-Mac).
    for bits in [8, 4] {
        assert!(
            image_request_candle_eligible(
                model,
                &object(json!({ "advanced": { "mlxQuantize": bits } }))
            ),
            "{model} Q{bits} tier-select should stay on candle"
        );
    }
    // Every conditioning shape defers AT THE TXT2IMG GATE (txt2img + LoRA only). NB `referenceAssetId`
    // here tests the low-level `image_request_candle_eligible` gate, which still rejects a raw
    // reference; the sc-10134 img2img lane is a SEPARATE branch in `image_job_is_candle_eligible` that
    // claims a `krea_2_turbo` reference BEFORE this gate (see `krea_2_turbo_img2img_routes_to_candle`).
    for case in [
        json!({ "mode": "edit_image", "sourceAssetId": "a" }),
        json!({ "referenceAssetId": "a" }),
        json!({ "maskAssetId": "m" }),
        json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),
    ] {
        assert!(
            !image_request_candle_eligible(model, &object(case.clone())),
            "{model} conditioning shape must fall back to torch: {case}"
        );
    }
}

#[test]
fn krea_2_turbo_img2img_routes_to_candle() {
    // sc-10134 (epic 8588 slice A): a `krea_2_turbo` job in a NON-edit mode carrying a
    // `referenceAssetId` is the bespoke candle `render_img2img` lane, branched out in
    // `image_job_is_candle_eligible` BEFORE the txt2img gate (which still rejects any raw reference —
    // see `krea_lora_and_quant_stay_on_candle_but_conditioning_defers`). The job-level predicate is
    // what claims it; the gate keeps deferring the reference shape.
    let img2img = json!({
        "model": "krea_2_turbo",
        "referenceAssetId": "asset_1",
        "advanced": { "strength": 0.55 }
    });
    assert!(
        image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
        "krea_2_turbo img2img (referenceAssetId, non-edit) must be candle-eligible"
    );
    assert!(krea_img2img_candle_eligible(&object(img2img)));
    // An explicit `text_to_image` mode is still img2img (the tile may send the mode or omit it).
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "krea_2_turbo",
        "mode": "text_to_image",
        "referenceAssetId": "asset_1"
    }))));
    // NOT the img2img lane: an `edit_image` reference is the Kontext `KreaEdit` surface.
    assert!(!krea_img2img_candle_eligible(&object(json!({
        "mode": "edit_image",
        "referenceAssetId": "asset_1"
    }))));
    // A plain txt2img job (no reference) is not img2img — it falls to the generic candle txt2img lane.
    assert!(!krea_img2img_candle_eligible(&object(
        json!({ "prompt": "an emerald forest" })
    )));
    // Raw img2img (sc-10226): the undistilled `krea_2_raw` sibling gets its own branch (engine
    // `render_base_img2img`), so a non-edit `krea_2_raw` reference job is candle-eligible too.
    let raw_img2img = json!({
        "model": "krea_2_raw",
        "referenceAssetId": "asset_1",
        "advanced": { "strength": 0.55 }
    });
    assert!(
        image_job_is_candle_eligible(&image_generate_job(raw_img2img.clone())),
        "krea_2_raw img2img (referenceAssetId, non-edit) must be candle-eligible (sc-10226)"
    );
    assert!(krea_img2img_candle_eligible(&object(raw_img2img)));
    // An `edit_image` `krea_2_raw` reference is still the Kontext edit surface, not img2img.
    assert!(!krea_img2img_candle_eligible(&object(json!({
        "mode": "edit_image",
        "referenceAssetId": "asset_1"
    }))));
}

#[test]
fn zimage_base_img2img_routes_to_candle() {
    // sc-10265 (epic 8588): a `z_image` (base, NOT Turbo) job in a non-edit mode with a
    // `referenceAssetId` is the REGISTRY img2img path — the base candle engine already serves it
    // (sc-8646) and the worker resolves the init generically (sc-10134), so only the router branch was
    // missing. Branched before the txt2img gate that rejects references.
    let img2img = json!({
        "model": "z_image",
        "referenceAssetId": "asset_1",
        "advanced": { "strength": 0.6 }
    });
    assert!(
        image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
        "z_image base img2img (referenceAssetId, non-edit) must be candle-eligible"
    );
    assert!(zimage_img2img_candle_eligible(&object(img2img)));
    // NOT the edit lane: `edit_image` is the bespoke `ZimageEdit` stream, not registry img2img.
    assert!(!zimage_img2img_candle_eligible(&object(json!({
        "mode": "edit_image",
        "referenceAssetId": "asset_1"
    }))));
    // A pose job stays on the control lane (poses branch first), not img2img.
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "z_image",
        "advanced": { "poses": [{ "keypoints": [] }] }
    }))));
}

#[test]
fn zimage_turbo_img2img_routes_to_candle() {
    // sc-11783 (epic 8588): `z_image_turbo` in a non-edit mode with a `referenceAssetId` is now the
    // REGISTRY img2img path too — the candle Turbo generator advertises `Reference` and blends the
    // reference into the CFG-free denoise. Branched AFTER identity/edit/control, before the txt2img gate.
    let img2img = json!({
        "model": "z_image_turbo",
        "referenceAssetId": "asset_1",
        "advanced": { "strength": 0.6 }
    });
    assert!(
        image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
        "z_image_turbo img2img (referenceAssetId, non-edit) must be candle-eligible (sc-11783)"
    );
    // Precedence: the identity-init shape (`character_image` mode + `referenceStrength`) stays on the
    // `zimage_identity` lane, NOT this img2img branch (both reach the candle worker, different lanes).
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "z_image_turbo",
        "mode": "character_image",
        "referenceAssetId": "asset_1",
        "advanced": { "referenceStrength": 0.7 }
    }))));
    // The `edit_image` masked-edit shape is the bespoke `ZimageEdit` stream, not this img2img branch.
    assert!(!zimage_img2img_candle_eligible(&object(json!({
        "model": "z_image_turbo",
        "mode": "edit_image",
        "referenceAssetId": "asset_1"
    }))));
    // A pose job stays on the Turbo control lane (poses branch first), not img2img.
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "z_image_turbo",
        "advanced": { "poses": [{ "keypoints": [] }] }
    }))));
}

#[test]
fn sd3_img2img_routes_to_candle() {
    // sc-11784 (epic 8588): each SD3.5 variant in a non-edit mode with a `referenceAssetId` is the
    // REGISTRY img2img path — the candle `candle-gen-sd3` generators advertise `Reference` and
    // VAE-encode it into the reduced denoise tail (real CFG for Large/Medium, distilled for Turbo).
    // Branched before the txt2img gate that rejects references. Candle/CUDA parity of MLX sc-10189.
    for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
        let img2img = json!({
            "model": model,
            "referenceAssetId": "asset_1",
            "advanced": { "strength": 0.6 }
        });
        assert!(
            image_job_is_candle_eligible(&image_generate_job(img2img.clone())),
            "{model} img2img (referenceAssetId, non-edit) must be candle-eligible (sc-11784)"
        );
        assert!(is_sd3_family_candle_model(model));
        assert!(sd3_img2img_candle_eligible(&object(img2img)));
        // A plain txt2img (no reference) still rides the generic candle gate — not the img2img branch.
        assert!(!sd3_img2img_candle_eligible(&object(
            json!({ "model": model })
        )));
        // The `edit_image` shape is NOT registry img2img (SD3.5 has no candle edit lane).
        assert!(!sd3_img2img_candle_eligible(&object(json!({
            "model": model,
            "mode": "edit_image",
            "referenceAssetId": "asset_1"
        }))));
    }
}

#[test]
fn sdxl_advanced_shapes_fall_back_to_torch() {
    // Every conditioning shape the txt2img candle lane can't honor must be ineligible. A LoRA is NOT
    // in this set anymore (sc-10767): the SDXL family advertises inference LoRA on candle, so a plain
    // LoRA txt2img stays on the candle lane (see `sdxl_family_quant_and_lora_stay_on_candle`). Only
    // the genuine conditioning shapes (img2img / reference / mask / strict-pose) fall back.
    let cases = [
        json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }), // img2img / inpaint / outpaint
        json!({ "referenceAssetId": "asset_1" }),                    // IP-Adapter reference
        json!({ "mode": "edit_image", "sourceAssetId": "a", "maskAssetId": "m" }), // inpaint
        json!({ "advanced": { "poses": [{ "id": "pose_1" }] } }),    // strict-pose ControlNet
    ];
    for case in cases {
        assert!(
            !image_request_candle_eligible("sdxl", &object(case.clone())),
            "sdxl shape must fall back to torch: {case}"
        );
    }
}

#[test]
fn blank_conditioning_ids_are_treated_as_absent() {
    // Whitespace/empty ids are not real conditioning → still plain txt2img → eligible.
    assert!(image_request_candle_eligible(
        "sdxl",
        &object(
            json!({ "referenceAssetId": "  ", "sourceAssetId": "", "advanced": { "poses": [] } })
        )
    ));
}

#[test]
fn candle_worker_claims_txt2img_but_refuses_unsupported_shapes() {
    let candle = gpu_worker(CANDLE_CAPS);
    // Claims the lane — SDXL plus every wired candle image family, all plain txt2img.
    for model in [
        "sdxl",
        "realvisxl",
        // sc-7176: RealVisXL Lightning routes to candle for plain txt2img (forced lightning sampler).
        "realvisxl_lightning",
        "z_image_turbo",
        "flux_dev",
        // sc-7458: FLUX.2-dev (the 32B flagship) routes to candle for plain txt2img off-Mac (loads
        // the dense snapshot + Q4-quantizes at load). Edit (sc-7736) + strict pose (sc-7736) are
        // candle lanes too now — covered by the dedicated assertions below.
        "flux2_dev",
        "qwen_image",
        "chroma1_hd",
        "kolors",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        // sc-10996 (epic 6562): the candle Bernini still-image companion routes to candle for plain
        // t2i (the dedicated `generate_candle_bernini_image_stream` lane, `frames:1`).
        "bernini_image",
        // sc-11780 (epic 8485): base `sana_1600m` plain txt2img rides the candle lane now (the
        // `candle-gen-sana` provider, candle-gen #495). It also accepts one singular img2img reference;
        // its unsupported adapter/quant/control refusal is asserted elsewhere.
        "sana_1600m",
        // sc-11781 (epic 8485): the CFG-free SANA-Sprint distill `sana_sprint_1600m` rides the candle
        // lane too (the `candle-gen-sana` Sprint pipeline, candle-gen #498). Its 1–4 step SCM/TrigFlow
        // path also accepts one singular img2img reference; quant / LoRA / control remain unsupported.
        "sana_sprint_1600m",
    ] {
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({ "model": model, "prompt": "a red fox" }))
            ),
            "candle worker should claim {model} plain txt2img"
        );
    }
    // Refuses a genuinely unsupported plain-txt2img image id (`pulid_flux_dev` — its only candle lane is the
    // bespoke character-reference path, so a PLAIN txt2img prompt has no candle route, candle_routed=
    // false), an adapter shape on dense SANA base (which advertises neither quant nor LoRA), and an
    // unsupported conditioning shape on a wired family — all are refused.
    assert!(!worker_supports_job(
        &candle,
        &image_generate_job(json!({ "model": "pulid_flux_dev", "prompt": "p" }))
    ));
    assert!(
        !worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "sana_1600m",
                "prompt": "p",
                "loras": [{ "path": "x", "weight": 0.8 }]
            }))
        ),
        "candle SANA base supports no adapters — a LoRA request defers to torch"
    );
    assert!(worker_supports_job(
        &candle,
        &image_generate_job(json!({
            "model": "kolors",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))
    ));
    // sc-5489: `qwen_image` + `advanced.poses` IS now a candle lane (the bespoke strict-pose
    // ControlNet route), so the candle worker claims it (was deferred to torch before this slice).
    assert!(
        worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "qwen_image",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ),
        "candle worker should claim qwen_image strict-pose (sc-5489)"
    );
    // sc-5489: `kolors` + `advanced.poses` is also a candle lane now (the Kolors strict-pose
    // ControlNet route), so the candle worker claims it too.
    assert!(
        worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "kolors",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ),
        "candle worker should claim kolors strict-pose (sc-5489)"
    );
    // sc-5489: `z_image_turbo` + `advanced.poses` is the LAST strict-pose family wired (the VACE
    // Fun-ControlNet route) — all three (qwen / kolors / z_image) are candle lanes now.
    assert!(
        worker_supports_job(
            &candle,
            &image_generate_job(json!({
                "model": "z_image_turbo",
                "advanced": { "poses": [{ "id": "pose_1" }] }
            }))
        ),
        "candle worker should claim z_image_turbo strict-pose (sc-5489)"
    );
    // sc-5968: plain `sdxl` + poses has NO candle pose lane (SDXL pose ships via InstantID), and
    // no native fallback can serve it off-Mac — so the candle worker CLAIMS it (to reject
    // with a typed error in the handler) rather than allowing a generic claimant to silently render
    // an unconditioned T2I image. `worker_supports_job` is therefore TRUE here (candle owns it to fail
    // it loudly); the handler's `candle_unsupported_pose_reject` guard does the rejecting.
    assert!(worker_supports_job(
        &candle,
        &image_generate_job(json!({
            "model": "sdxl",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }))
    ));
    // sc-5487: a plain SDXL edit (img2img: `edit_image` + a source) is now a candle lane (the
    // bespoke `SdxlEdit` route), so the candle worker CLAIMS it.
    assert!(worker_supports_job(
        &candle,
        &image_generate_job(json!({
            "model": "sdxl",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))
    ));
    // sc-5487: a FLUX.2-klein edit (`edit_image` + a source) is now the candle `Flux2Edit` lane.
    // candle is the only off-Mac lane for klein, so the worker CLAIMS it.
    assert!(worker_supports_job(
        &candle,
        &image_generate_job(json!({
            "model": "flux2_klein_9b",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))
    ));
    // SC-15831: all three Klein catalog entries share the Candle reference provider, while exact
    // artifact/calibration admission remains entry-specific inside the worker.
    for model in [
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        for (mode, reference_field) in [
            ("edit_image", "sourceAssetId"),
            ("reference", "referenceAssetId"),
            ("image_to_image", "referenceAssetId"),
            ("character_image", "referenceAssetId"),
            ("style_variations", "referenceAssetId"),
        ] {
            let mut payload = json!({ "model": model, "mode": mode });
            payload
                .as_object_mut()
                .expect("image payload")
                .insert(reference_field.to_owned(), json!("asset_1"));
            assert!(
                image_job_is_candle_eligible(&image_generate_job(payload)),
                "model={model} mode={mode} must reach Candle Flux2Edit"
            );
            assert!(!flux2_edit_candle_eligible(&object(
                json!({ "mode": mode })
            )));
        }
    }
    // sc-7736 (epic 6564): FLUX.2-dev edit (`edit_image` + a source) is NOW the candle `Flux2Edit`
    // dev lane (`load_dev`, Q4) — the worker CLAIMS it (was deferred to torch under sc-7458's
    // txt2img-only slice). Multi-reference (the plural `referenceAssetIds`) rides the same lane.
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "flux2_dev",
        "mode": "edit_image",
        "sourceAssetId": "asset_1"
    }))));
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "flux2_dev",
        "mode": "character_image",
        "referenceAssetIds": ["asset_1", "asset_2"]
    }))));
    // sc-18476: a pure-reference flux2_dev job is now claimed by Flux2Edit; it no longer falls through
    // the generic T2I gate or remains queued.
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "flux2_dev",
        "mode": "reference",
        "referenceAssetId": "asset_1"
    }))));
    for (mode, reference_field) in [
        ("image_to_image", "sourceAssetId"),
        ("style_variations", "referenceAssetId"),
    ] {
        let mut payload = json!({ "model": "flux2_dev", "mode": mode });
        payload
            .as_object_mut()
            .expect("image payload")
            .insert(reference_field.to_owned(), json!("asset_1"));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    }
    for malformed in [
        json!({ "model": "flux2_dev", "mode": "reference", "referenceAssetId": " " }),
        json!({ "model": "flux2_dev", "mode": "character_image", "referenceAssetIds": ["asset_1", ""] }),
        json!({ "model": "flux2_dev", "mode": "style_variations", "referenceAssetId": "asset_1", "sourceAssetId": "asset_2" }),
        json!({ "model": "flux2_dev", "mode": "edit_image", "sourceAssetId": "asset_1", "referenceAssetIds": ["asset_2"] }),
    ] {
        assert!(!image_job_is_candle_eligible(&image_generate_job(
            malformed
        )));
    }
    // sc-7736: FLUX.2-dev strict pose (`advanced.poses`, not edit) is the candle `Flux2Control`
    // Fun-Controlnet-Union lane — the worker CLAIMS it (the 4th wired strict-pose family). A pose job
    // with no poses array is plain txt2img (claimed by the generic candle lane, not the control one).
    assert!(image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "flux2_dev",
        "advanced": { "poses": [{ "keypoints": [] }] }
    }))));
    assert!(flux2_dev_control_candle_eligible(&object(json!({
        "advanced": { "poses": [{ "keypoints": [] }] }
    }))));
    // An `edit_image` flux2_dev job is the edit lane, not the control lane (disjoint gates).
    assert!(!flux2_dev_control_candle_eligible(&object(json!({
        "mode": "edit_image",
        "advanced": { "poses": [{ "keypoints": [] }] }
    }))));
    // sc-5487: a Qwen-Image-Edit edit (`edit_image` + a source) is now the candle `QwenEdit` lane
    // (dual-latent reference editing). Off-Mac this was a torch fallback; the candle worker CLAIMS
    // it. The `-2511_lightning` distill (sc-6220) is the same `-2511` base with the lightx2v 4-step
    // LoRA folded into the MMDiT at load, so it is candle-claimed too.
    for model in [
        "qwen_image_edit",
        "qwen_image_edit_2509",
        "qwen_image_edit_2511",
        "qwen_image_edit_2511_lightning",
    ] {
        assert!(
            worker_supports_job(
                &candle,
                &image_generate_job(json!({
                    "model": model,
                    "mode": "edit_image",
                    "sourceAssetId": "asset_1"
                }))
            ),
            "candle worker should claim {model} edit (sc-5487 / sc-6220)"
        );
    }
    // A Qwen-Image-Edit job with no source image is not the edit lane → not claimed (would defer).
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "qwen_image_edit",
        "mode": "edit_image"
    }))));
}

#[test]
fn torch_worker_claims_everything_the_candle_worker_defers() {
    // This synthetic generic descriptor (no `candle` marker) exercises the legacy non-candle branch.
    // Production has no fallback: shapes refused by candle remain queued, including unsupported-pose
    // shapes that candle owns-to-reject (sc-5968) to prevent unconditioned T2I rendering.
    let torch = gpu_worker(TORCH_CAPS);
    // The legacy torch worker may also claim a model that now has a native provider; scheduler
    // precedence still prefers the native SANA lane. The second assertion covers an unsupported
    // conditioning shape on another wired family.
    assert!(worker_supports_job(
        &torch,
        &image_generate_job(json!({ "model": "sana_1600m", "prompt": "p" }))
    ));
    assert!(worker_supports_job(
        &torch,
        &image_generate_job(json!({
            "model": "kolors",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))
    ));
    assert!(worker_supports_job(
        &torch,
        &image_generate_job(json!({
            "model": "qwen_image",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }))
    ));
    assert!(worker_supports_job(
        &torch,
        &image_generate_job(json!({
            "model": "sdxl",
            "mode": "edit_image",
            "sourceAssetId": "asset_1"
        }))
    ));
    // sc-5968: the synthetic generic descriptor DECLINES the unsupported-pose shape the candle
    // worker owns-to-reject (sdxl + poses), preventing silent unconditioned T2I; candle takes it and
    // rejects. On Mac the same shape is MLX-served, so the `mlx` worker still claims it (asserted
    // in the cross-descriptor unsupported-pose test).
    assert!(!worker_supports_job(
        &torch,
        &image_generate_job(json!({
            "model": "sdxl",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }))
    ));
}

/// sc-5968: unsupported-pose routing across descriptors — candle OWNS it (to reject), the synthetic
/// generic descriptor DECLINES it (no silent T2I), and the Mac `mlx` worker SERVES it (no regression,
/// `sdxl_mlx_eligible` is unconditional). Plus: the wired candle pose families are unaffected, and
/// `image_job_is_candle_eligible` still reports sdxl+poses as NOT candle-*served* (it's owned only
/// to reject — the distinction the worker's dispatch guard keys on).
#[test]
fn unsupported_pose_is_owned_by_candle_declined_by_torch_served_by_mlx() {
    let candle = gpu_worker(CANDLE_CAPS);
    let torch = gpu_worker(TORCH_CAPS);
    let mlx: WorkerSnapshot = serde_json::from_value(json!({
        "id": "worker_mlx",
        "gpuId": "mlx",
        "status": "idle",
        "capabilities": ["gpu", "image_generate"],
        "loadedModels": [],
        "registeredAt": "2026-06-16T00:00:00Z",
        "lastSeenAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid WorkerSnapshot");
    let sdxl_pose =
        image_generate_job(json!({ "model": "sdxl", "advanced": { "poses": [{ "id": "p" }] } }));

    assert!(image_request_candle_pose_reject(
        "sdxl",
        &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
    ));
    assert!(worker_supports_job(&candle, &sdxl_pose), "candle owns it");
    assert!(
        !worker_supports_job(&torch, &sdxl_pose),
        "torch declines it"
    );
    assert!(worker_supports_job(&mlx, &sdxl_pose), "mlx still serves it");
    // It is NOT candle-*served* (only owned-to-reject); the worker's dispatch guard rejects it.
    assert!(!image_job_is_candle_eligible(&sdxl_pose));

    // A wired candle pose family is NOT a reject shape, and edit_image is never a reject shape.
    assert!(!image_request_candle_pose_reject(
        "qwen_image",
        &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
    ));
    // sc-7736: flux2_dev now HAS a candle pose lane (Flux2Control), so its pose job is served, not
    // rejected.
    assert!(!image_request_candle_pose_reject(
        "flux2_dev",
        &object(json!({ "advanced": { "poses": [{ "id": "p" }] } }))
    ));
    assert!(!image_request_candle_pose_reject(
        "sdxl",
        &object(json!({ "mode": "edit_image", "advanced": { "poses": [{ "id": "p" }] } }))
    ));
    // No poses → not a reject shape (plain txt2img stays candle-eligible).
    assert!(!image_request_candle_pose_reject(
        "sdxl",
        &object(json!({ "prompt": "a fox" }))
    ));
}

// ---- Candle video lane (sc-5097) ----

/// A queued `video_generate` job carrying `payload`.
fn video_generate_job(payload: Value) -> JobSnapshot {
    video_job("video_generate", payload)
}

/// A queued video job carrying `payload`, with its API-stamped type preserved so the full claim
/// gate derives advanced modes from the job type rather than trusting a stale payload mode.
fn video_job(job_type: &str, payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_v",
        "type": job_type,
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-13T00:00:00Z",
        "updatedAt": "2026-06-13T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

// The candle worker on the video lane advertises `video_generate` + the `candle` marker.
const CANDLE_VIDEO_CAPS: &[&str] = &["gpu", "video_generate", "candle"];
const TORCH_VIDEO_CAPS: &[&str] = &["gpu", "video_generate"];

#[test]
fn candle_routed_video_models_are_eligible_in_their_native_shape() {
    // txt2video lane: the 5B, ltx, the 14B T2V (text-only), and Mochi are eligible for
    // text_to_video. Mochi (sc-11991) is on the candle lane because its CANDLE descriptor is
    // `mac_only: false` — the off-Mac engine ingests the same hosted mlx-affine tiers, so Mochi
    // must NOT be hard mac-gated even though its MLX descriptor is `mac_only: true`.
    for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b", "mochi_1"] {
        assert!(
            video_request_candle_eligible(
                model,
                &object(json!({ "mode": "text_to_video", "prompt": "a river at dawn" }))
            ),
            "{model} text_to_video should be candle-eligible"
        );
    }
    // image→video lane: the 14B I2V + SVD are eligible only with the i2v mode + a source image
    // (sc-5175 / sc-5493).
    for model in ["wan_2_2_i2v_14b", "svd"] {
        assert!(
            video_request_candle_eligible(
                model,
                &object(
                    json!({ "mode": "image_to_video", "sourceAssetId": "asset_1", "prompt": "p" })
                )
            ),
            "{model} image_to_video with a source should be candle-eligible"
        );
    }
}

#[test]
fn non_candle_video_models_and_conditioned_shapes_fall_back() {
    // sc-18902: exact-head Candle/CUDA acceptance proved that the dense, undistilled 10Eros
    // checkpoint produces unresolved noise through Candle's single-pass distilled engine. No
    // 10Eros shape may route to Candle; the validated MLX two-pass distill recipe remains available.
    for payload in [
        json!({ "mode": "text_to_video" }),
        json!({ "mode": "text_to_video", "loras": [{ "id": "ltx_style" }] }),
        json!({ "mode": "first_last_frame" }),
    ] {
        assert!(
            !video_request_candle_eligible("ltx_2_3_eros", &object(payload.clone())),
            "ltx_2_3_eros must remain off Candle for every shape: {payload}"
        );
    }
    // A genuinely non-candle video model is refused and remains queued.
    assert!(
        !video_request_candle_eligible(
            "some_unported_model",
            &object(json!({ "mode": "text_to_video" }))
        ),
        "an unported model must fall back to the Python worker"
    );
    // The 14B T2V is text-only: any image_to_video / sourced shape is refused (sc-5175).
    for case in [
        json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
        json!({ "mode": "text_to_video", "sourceAssetId": "a" }),
    ] {
        assert!(
            !video_request_candle_eligible("wan_2_2_t2v_14b", &object(case.clone())),
            "wan_2_2_t2v_14b conditioned shape must fall back to torch: {case}"
        );
    }
    // The 14B I2V + SVD are image→video only: a txt2video shape or an i2v with no source is refused
    // (sc-5175 / sc-5493).
    for model in ["wan_2_2_i2v_14b", "svd"] {
        for case in [
            json!({ "mode": "text_to_video", "prompt": "p" }),
            json!({ "mode": "image_to_video" }), // i2v but no source image
        ] {
            assert!(
                !video_request_candle_eligible(model, &object(case.clone())),
                "{model} non-i2v shape must fall back to torch: {case}"
            );
        }
    }
    // SVD has no candle LoRA slot, so a LoRA even on its valid i2v shape still falls back; the
    // Wan-14B I2V now ACCEPTS a user LoRA on candle (sc-10539) — see `candle_wan_14b_video_accepts_user_loras`.
    assert!(
        !video_request_candle_eligible(
            "svd",
            &object(
                json!({ "mode": "image_to_video", "sourceAssetId": "a", "loras": [{ "name": "x" }] })
            )
        ),
        "svd (no candle LoRA slot) must fall back to torch on an i2v+LoRA shape"
    );
    // Mochi 1 (sc-11991) is txt2video-only with NO candle LoRA slot (both descriptors set
    // `supports_lora`/`supports_lokr` = false), so every conditioned or LoRA-carrying shape is
    // refused — there is no torch fallback for it (epic 8283), it simply is not candle-claimed.
    for case in [
        json!({ "prompt": "p" }), // no mode → defaults to i2v
        json!({ "mode": "image_to_video", "sourceAssetId": "a" }),
        json!({ "mode": "first_last_frame" }),
        json!({ "mode": "text_to_video", "sourceAssetId": "a" }),
        json!({ "mode": "text_to_video", "referenceAssetId": "a" }),
        json!({ "mode": "text_to_video", "loras": [{ "name": "x" }] }),
    ] {
        assert!(
            !video_request_candle_eligible("mochi_1", &object(case.clone())),
            "mochi_1 non-t2v/conditioned shape must not be candle-claimed: {case}"
        );
    }
}

#[test]
fn candle_video_models_with_provider_slots_accept_user_loras() {
    // sc-10539: the Wan-14B MoE engines advertise `supports_lora` and their candle worker path
    // (`candle_resolve_wan_adapters`) applies each user LoRA — including an external ComfyUI file
    // read in place — so a LoRA-carrying job stays on candle instead of the old blanket exclusion
    // (there is no torch fallback now; epic 8283). GPU-validated: an external `Wan/detailz-wan`
    // adapter rendered a candle Wan-14B clip that differs from the no-LoRA baseline at the same seed.
    assert!(
        video_request_candle_eligible(
            "wan_2_2_t2v_14b",
            &object(json!({ "mode": "text_to_video", "loras": [{ "id": "external_x" }] }))
        ),
        "wan_2_2_t2v_14b text_to_video + user LoRA must stay on candle"
    );
    assert!(
        video_request_candle_eligible(
            "wan_2_2_i2v_14b",
            &object(json!({
                "mode": "image_to_video",
                "sourceAssetId": "a",
                "loras": [{ "id": "external_x" }],
            }))
        ),
        "wan_2_2_i2v_14b i2v + source + user LoRA must stay on candle"
    );
    // The pinned base-LTX Candle provider installs additive LoRA residuals on video-attention
    // projections. Eros is deliberately absent: its required distill adapter does not fit this
    // provider surface (sc-18902).
    assert!(video_request_candle_eligible(
        "ltx_2_3",
        &object(json!({
            "mode": "text_to_video",
            "loras": [{ "id": "ltx_style" }],
        }))
    ));
    assert!(video_request_candle_eligible(
        "wan_2_2",
        &object(json!({ "mode": "text_to_video", "loras": [{ "id": "wan_style" }] }))
    ));
    // Families whose candle provider advertises no LoRA slot still refuse a LoRA.
    let model = "svd";
    let payload =
        json!({ "mode": "image_to_video", "sourceAssetId": "a", "loras": [{ "id": "x" }] });
    assert!(
        !video_request_candle_eligible(model, &object(payload.clone())),
        "{model} has no candle LoRA slot — a LoRA job must not route to candle: {payload}"
    );
}

#[test]
fn candle_ltx_and_wan5_serve_new_conditioning_shapes() {
    for model in ["ltx_2_3", "wan_2_2"] {
        assert!(video_request_candle_eligible(
            model,
            &object(json!({ "mode": "image_to_video", "sourceAssetId": "first" }))
        ));
        assert!(video_request_candle_eligible(
            model,
            &object(json!({
                "mode": "first_last_frame",
                "sourceAssetId": "first",
                "lastFrameAssetId": "last"
            }))
        ));
    }
    assert!(video_request_candle_eligible(
        "ltx_2_3",
        &object(json!({
            "mode": "extend_clip",
            "sourceClipAssetId": "left",
            "loras": [{ "id": "ltx_2_3_ic_union_control" }]
        }))
    ));
    assert!(video_request_candle_eligible(
        "ltx_2_3",
        &object(json!({
            "mode": "video_bridge",
            "sourceClipAssetId": "left",
            "bridgeRightClipAssetId": "right",
            "loras": [{ "id": "ltx_2_3_ic_union_control" }]
        }))
    ));
}

#[test]
fn candle_video_tier_selects_match_published_platform_tiers() {
    for model in ["wan_2_2", "wan_2_2_t2v_14b", "wan_2_2_i2v_14b"] {
        let payload = if model == "wan_2_2_i2v_14b" {
            json!({ "mode": "image_to_video", "sourceAssetId": "first", "advanced": { "mlxQuantize": 8 } })
        } else {
            json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } })
        };
        assert!(video_request_candle_eligible(model, &object(payload)));
    }
    assert!(video_request_candle_eligible(
        "ltx_2_3",
        &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 4 } }))
    ));
    assert!(!video_request_candle_eligible(
        "ltx_2_3",
        &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 8 } }))
    ));
    assert!(!video_request_candle_eligible(
        "ltx_2_3_eros",
        &object(json!({ "mode": "text_to_video", "advanced": { "mlxQuantize": 4 } }))
    ));

    let explicit_torch = video_generate_job(json!({
        "model": "wan_2_2",
        "mode": "text_to_video",
        "advanced": { "quantization": "gguf-q4_k_m" }
    }));
    assert!(!video_request_candle_eligible(
        "wan_2_2",
        &explicit_torch.payload
    ));
    assert!(
        !video_job_is_candle_eligible(&explicit_torch),
        "Candle must reject rather than silently discard an explicit Torch GGUF selection"
    );

    for neutral in ["auto", " Auto ", ""] {
        let payload = object(json!({
            "model": "wan_2_2",
            "mode": "text_to_video",
            "advanced": { "quantization": neutral }
        }));
        assert!(
            video_request_candle_eligible("wan_2_2", &payload),
            "neutral Torch quantization marker {neutral:?} must not eject a native Candle request"
        );
    }
}

#[test]
fn candle_ltx_replace_is_model_native_and_requires_its_ic_adapter() {
    let shape = json!({
        "sourceClipAssetId": "clip",
        "personTrackId": "track",
        "characterId": "character",
        "loras": [{ "id": "ltx_2_3_ic_union_control" }]
    });
    assert!(ltx_replace_candle_eligible(
        "ltx_2_3",
        &object(shape.clone())
    ));
    let mut payload = object(shape.clone());
    payload.insert("model".into(), json!("ltx_2_3"));
    assert!(video_job_is_candle_eligible(&person_replace_job(
        Value::Object(payload)
    )));

    assert!(
        !ltx_replace_candle_eligible("ltx_2_3_eros", &object(shape.clone())),
        "SC-18902 withdrew Eros from every Candle route, including native replacement"
    );
    let mut eros_payload = object(shape.clone());
    eros_payload.insert("model".into(), json!("ltx_2_3_eros"));
    assert!(!video_job_is_candle_eligible(&person_replace_job(
        Value::Object(eros_payload)
    )));
    let mut missing_adapter = object(shape);
    missing_adapter.remove("loras");
    assert!(!ltx_replace_candle_eligible("ltx_2_3", &missing_adapter));
    let ordinary_adapter = object(json!({
        "sourceClipAssetId": "clip",
        "personTrackId": "track",
        "characterId": "character",
        "loras": [{ "id": "ordinary_ltx_style" }]
    }));
    assert!(
        !ltx_replace_candle_eligible("ltx_2_3", &ordinary_adapter),
        "any LoRA is not enough: native replacement requires the recognizable IC-LoRA"
    );
}

#[test]
fn candle_ltx_clip_modes_require_a_recognizable_ic_lora() {
    for mode in ["extend_clip", "video_bridge"] {
        let mut payload = object(json!({
            "mode": mode,
            "sourceClipAssetId": "left",
            "loras": [{ "id": "ordinary_ltx_style" }]
        }));
        if mode == "video_bridge" {
            payload.insert("bridgeRightClipAssetId".into(), json!("right"));
        }
        assert!(!video_request_candle_eligible("ltx_2_3", &payload));
        payload.insert("loras".into(), json!([{ "conditioningRole": "ic_lora" }]));
        assert!(video_request_candle_eligible("ltx_2_3", &payload));
    }
}

#[test]
fn candle_vace_modes_eligible_with_required_assets() {
    // replace_person (PersonReplace): needs the source clip + person track + character.
    assert!(video_request_candle_vace_eligible(
        "wan_2_2",
        &object(json!({
            "sourceClipAssetId": "clip_1",
            "personTrackId": "track_1",
            "characterId": "char_1"
        })),
        &JobType::PersonReplace
    ));
    // extend_clip (VideoExtend): needs a source clip.
    assert!(video_request_candle_vace_eligible(
        "wan_2_2_t2v_14b",
        &object(json!({ "sourceClipAssetId": "clip_1" })),
        &JobType::VideoExtend
    ));
    // video_bridge (VideoBridge): needs both clips.
    assert!(video_request_candle_vace_eligible(
        "wan_2_2_i2v_14b",
        &object(json!({ "sourceClipAssetId": "l", "bridgeRightClipAssetId": "r" })),
        &JobType::VideoBridge
    ));
    // sc-18478: VACE-Fun is an exact dual-expert provider and accepts its own user adapter stack.
    assert!(video_request_candle_vace_eligible(
        "wan_2_2_vace_fun_14b",
        &object(json!({
            "sourceClipAssetId": "clip_1",
            "personTrackId": "track_1",
            "characterId": "char_1",
            "loras": [{ "name": "vace-fun-style" }]
        })),
        &JobType::PersonReplace
    ));
}

#[test]
fn candle_vace_fun_is_dedicated_to_person_replace() {
    let model = "wan_2_2_vace_fun_14b";
    assert!(video_job_is_candle_eligible(&person_replace_job(json!({
        "model": model,
        "sourceClipAssetId": "clip_1",
        "personTrackId": "track_1",
        "characterId": "char_1"
    }))));

    let unsupported = [
        (
            "video_generate",
            json!({ "model": model, "mode": "text_to_video" }),
        ),
        (
            "video_extend",
            json!({ "model": model, "mode": "extend_clip", "sourceClipAssetId": "clip_1" }),
        ),
        (
            "video_bridge",
            json!({
                "model": model,
                "mode": "video_bridge",
                "sourceClipAssetId": "left",
                "bridgeRightClipAssetId": "right"
            }),
        ),
    ];
    for (job_type, payload) in unsupported {
        assert!(
            !video_job_is_candle_eligible(&video_job(job_type, payload)),
            "VACE-Fun must not cross-route {job_type} onto a base or single-expert VACE engine"
        );
    }
}

#[test]
fn candle_vace_modes_fall_back_without_assets_or_for_unsupported_models() {
    // Missing required assets make the request ineligible, so it remains queued.
    assert!(!video_request_candle_vace_eligible(
        "wan_2_2",
        &object(json!({ "sourceClipAssetId": "clip_1" })), // no personTrackId / characterId
        &JobType::PersonReplace
    ));
    assert!(!video_request_candle_vace_eligible(
        "wan_2_2",
        &object(json!({ "sourceClipAssetId": "l" })), // bridge needs the right clip too
        &JobType::VideoBridge
    ));
    // SCAIL-2 is a DISTINCT candle engine, not a VACE model → the VACE gate rejects it (the SCAIL-2
    // candle replace path is `scail2_replace_candle_eligible`, sc-6837).
    assert!(!video_request_candle_vace_eligible(
        "scail2_14b",
        &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
        &JobType::PersonReplace
    ));
    // A LoRA shape is refused (the candle VACE provider advertises no adapters).
    assert!(!video_request_candle_vace_eligible(
        "wan_2_2",
        &object(json!({
            "sourceClipAssetId": "c",
            "personTrackId": "t",
            "characterId": "ch",
            "loras": [{ "name": "x" }]
        })),
        &JobType::PersonReplace
    ));
    // A non-VACE job type is never VACE-eligible (the base txt2video gate handles VideoGenerate).
    assert!(!video_request_candle_vace_eligible(
        "wan_2_2",
        &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" })),
        &JobType::VideoGenerate
    ));
}

// ---- Candle SCAIL-2 character animation + replace_person (sc-6837, epic 6563) ----

/// A queued `person_replace` job carrying `payload` (the PersonReplace job type the API stamps for
/// the integrated replace_person pipeline).
fn person_replace_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_pr",
        "type": "person_replace",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-20T00:00:00Z",
        "updatedAt": "2026-06-20T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

#[test]
fn scail2_candle_serves_animation_and_replace_in_native_shape() {
    // Standalone character animation: scail2_14b + animate_character + a reference + a driving clip.
    // The reference can be referenceAssetIds, a bare referenceAssetId, or the i2v sourceAssetId.
    for reference in [
        json!({ "referenceAssetIds": ["ref_1"] }),
        json!({ "referenceAssetId": "ref_1" }),
        json!({ "sourceAssetId": "img_1" }),
    ] {
        let mut payload = object(reference);
        payload.insert("mode".into(), json!("animate_character"));
        payload.insert("sourceClipAssetId".into(), json!("clip_1"));
        assert!(
            scail2_animate_candle_eligible("scail2_14b", &payload),
            "scail2 animate_character must be candle-eligible: {payload:?}"
        );
    }
    // An animate job carrying an inference LoRA (DPO / lightning / user adapter) stays on candle —
    // the provider merges it into the dense DiT (sc-6838); only on-the-fly quant is refused.
    assert!(
        scail2_animate_candle_eligible(
            "scail2_14b",
            &object(json!({
                "mode": "animate_character",
                "referenceAssetIds": ["ref_1"],
                "sourceClipAssetId": "clip_1",
                "loras": [{ "name": "scail2-dpo" }]
            }))
        ),
        "scail2 animate with a LoRA must stay candle-eligible (sc-6838)"
    );
    // Cross-identity replacement: scail2_14b PersonReplace with the clip + track + character.
    assert!(scail2_replace_candle_eligible(
        "scail2_14b",
        &object(json!({
            "sourceClipAssetId": "clip_1",
            "personTrackId": "track_1",
            "characterId": "char_1"
        }))
    ));
    assert!(scail2_replace_candle_eligible(
        "scail2_14b",
        &object(json!({
            "sourceClipAssetId": "clip_1",
            "personTrackId": "track_1",
            "characterId": "char_1",
            "loras": [{ "name": "scail2-dpo" }]
        }))
    ));
    // Through the full video claim gate: animate_character (VideoGenerate) + replace (PersonReplace).
    assert!(video_job_is_candle_eligible(&video_generate_job(json!({
        "model": "scail2_14b",
        "mode": "animate_character",
        "referenceAssetIds": ["ref_1"],
        "sourceClipAssetId": "clip_1"
    }))));
    assert!(video_job_is_candle_eligible(&person_replace_job(json!({
        "model": "scail2_14b",
        "sourceClipAssetId": "clip_1",
        "personTrackId": "track_1",
        "characterId": "char_1"
    }))));
}

#[test]
fn bernini_video_candle_serves_every_mode_in_native_shape() {
    // sc-10997 (epic 6562): the candle Bernini VIDEO lane serves t2v + the editing/reference/
    // multi-source modes on the distinct `bernini` engine — the off-Mac parity of the MLX
    // `video_mode_is_mlx_eligible(bernini, mode)` gate. Each mode claims the candle lane (routed on
    // id + mode; the worker validates the per-mode media + `SceneWorks/bernini` weights loudly,
    // sc-11003). The story's "i2v"/"edit" map onto Bernini's `video_to_video` — its Wan2.2-T2V
    // renderer has no classic still-image-to-video (mirrors the MLX lane's mode set exactly).
    for (mode, payload) in [
        (
            "text_to_video",
            json!({ "model": "bernini", "mode": "text_to_video", "prompt": "a kite" }),
        ),
        (
            "video_to_video",
            json!({ "model": "bernini", "mode": "video_to_video", "sourceClipAssetId": "clip_1" }),
        ),
        (
            "reference_to_video",
            json!({ "model": "bernini", "mode": "reference_to_video", "referenceAssetIds": ["ref_1"] }),
        ),
        (
            "reference_video_to_video",
            json!({
                "model": "bernini",
                "mode": "reference_video_to_video",
                "sourceClipAssetId": "clip_1",
                "referenceAssetIds": ["ref_1"]
            }),
        ),
        (
            "multi_video_to_video",
            json!({
                "model": "bernini",
                "mode": "multi_video_to_video",
                "sourceClipAssetIds": ["clip_1", "clip_2"]
            }),
        ),
        (
            "ads2v",
            json!({
                "model": "bernini",
                "mode": "ads2v",
                "sourceClipAssetId": "clip_1",
                "referenceClipAssetId": "clip_2",
                "referenceAssetIds": ["ref_1"]
            }),
        ),
    ] {
        assert!(
            bernini_video_candle_eligible("bernini", &object(payload.clone())),
            "bernini {mode} must be candle-eligible"
        );
        // Through the full VideoGenerate claim gate (the OR-in wiring reaches the candle worker).
        assert!(
            video_job_is_candle_eligible(&video_generate_job(payload.clone())),
            "bernini {mode} must route to the candle worker via video_job_is_candle_eligible"
        );
    }
    // An explicit tier request (`mlxQuantize`) is lineage-only — it does NOT push Bernini off candle
    // (the loader reads the converted tree dense; there is no torch Bernini to fall back to, sc-11003).
    let mut quant = object(json!({ "model": "bernini", "mode": "text_to_video" }));
    quant.insert("advanced".into(), json!({ "mlxQuantize": 8 }));
    assert!(
        bernini_video_candle_eligible("bernini", &quant),
        "a tier-requesting bernini job stays on candle (dense load + lineage bits)"
    );
}

#[test]
fn bernini_video_candle_rejects_wrong_model_or_mode() {
    // Only the `bernini` id claims the Bernini candle VIDEO lane; other video models (and the still
    // `bernini_image` id) keep their own routing.
    for model in ["wan_2_2", "scail2_14b", "bernini_image", "ltx_2_3"] {
        assert!(
            !bernini_video_candle_eligible(model, &object(json!({ "mode": "text_to_video" }))),
            "{model} must not claim the bernini video candle lane"
        );
    }
    // A mode Bernini does not serve (no classic i2v; replace/animate/clip modes belong to other
    // engines) is not this lane — mirrors the MLX `video_mode_is_mlx_eligible(bernini, ..)` set.
    for mode in [
        "image_to_video",
        "replace_person",
        "animate_character",
        "first_last_frame",
        "extend_clip",
        "",
    ] {
        assert!(
            !bernini_video_candle_eligible("bernini", &object(json!({ "mode": mode }))),
            "bernini {mode:?} is not a served candle video mode"
        );
    }
    // No mode at all → not eligible (a real bernini job always carries an explicit mode).
    assert!(!bernini_video_candle_eligible(
        "bernini",
        &object(json!({ "prompt": "p" }))
    ));
}

#[test]
fn scail2_candle_rejects_incomplete_or_wrong_shape() {
    // animate_character needs BOTH a reference image and a driving clip.
    assert!(!scail2_animate_candle_eligible(
        "scail2_14b",
        &object(json!({ "mode": "animate_character", "referenceAssetIds": ["ref_1"] }))
    ));
    assert!(!scail2_animate_candle_eligible(
        "scail2_14b",
        &object(json!({ "mode": "animate_character", "sourceClipAssetId": "clip_1" }))
    ));
    // Wrong mode / wrong model never claim the SCAIL-2 candle lane.
    assert!(!scail2_animate_candle_eligible(
        "scail2_14b",
        &object(json!({
            "mode": "text_to_video",
            "sourceAssetId": "i",
            "sourceClipAssetId": "c"
        }))
    ));
    assert!(!scail2_animate_candle_eligible(
        "wan_2_2",
        &object(json!({
            "mode": "animate_character",
            "sourceAssetId": "i",
            "sourceClipAssetId": "c"
        }))
    ));
    // On-the-fly quant is still refused (the candle SCAIL-2 provider is dense).
    {
        let mut payload = object(json!({
            "mode": "animate_character",
            "sourceAssetId": "i",
            "sourceClipAssetId": "c"
        }));
        payload.insert("advanced".into(), json!({ "mlxQuantize": 8 }));
        assert!(
            !scail2_animate_candle_eligible("scail2_14b", &payload),
            "scail2 animate with on-the-fly quant must defer to torch: {payload:?}"
        );
    }
    // replace_person needs the clip + track + character; missing any makes it ineligible.
    for case in [
        json!({ "sourceClipAssetId": "c", "personTrackId": "t" }),
        json!({ "sourceClipAssetId": "c", "characterId": "ch" }),
        json!({ "personTrackId": "t", "characterId": "ch" }),
    ] {
        assert!(
            !scail2_replace_candle_eligible("scail2_14b", &object(case.clone())),
            "incomplete scail2 replace must defer to torch: {case}"
        );
    }
    // A non-SCAIL-2 model never claims the SCAIL-2 replace lane (it routes via Wan-VACE instead).
    assert!(!scail2_replace_candle_eligible(
        "wan_2_2",
        &object(json!({ "sourceClipAssetId": "c", "personTrackId": "t", "characterId": "ch" }))
    ));
}

#[test]
fn candle_worker_claims_native_video_shapes_and_refuses_invalid_ones() {
    let candle = gpu_worker(CANDLE_VIDEO_CAPS);
    // Claims wan + ltx + the 14B T2V plain txt2video.
    for model in ["wan_2_2", "ltx_2_3", "wan_2_2_t2v_14b"] {
        assert!(
            worker_supports_job(
                &candle,
                &video_generate_job(json!({ "model": model, "mode": "text_to_video" }))
            ),
            "candle worker should claim {model} txt2video"
        );
    }
    // Claims the 14B I2V + SVD in their image→video shape (with a source image) (sc-5175 / sc-5493).
    for model in ["wan_2_2_i2v_14b", "svd"] {
        assert!(
            worker_supports_job(
                &candle,
                &video_generate_job(json!({
                    "model": model,
                    "mode": "image_to_video",
                    "sourceAssetId": "a"
                }))
            ),
            "candle worker should claim {model} image_to_video"
        );
    }
    // sc-18902: Candle refuses Eros after exact-head runtime evidence showed its undistilled dense
    // checkpoint collapses to noise on this single-pass engine. It also refuses an unported model,
    // a conditioned (i2v) shape on a txt2video model, an image→video model (svd) in a txt2video shape,
    // and the 14B I2V in a txt2video shape (both image→video only).
    assert!(!worker_supports_job(
        &candle,
        &video_generate_job(json!({ "model": "ltx_2_3_eros", "mode": "text_to_video" }))
    ));
    assert!(!worker_supports_job(
        &candle,
        &video_generate_job(json!({ "model": "some_unported_model", "mode": "text_to_video" }))
    ));
    assert!(!worker_supports_job(
        &candle,
        &video_generate_job(json!({ "model": "svd", "mode": "text_to_video" }))
    ));
    assert!(worker_supports_job(
        &candle,
        &video_generate_job(
            json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
        )
    ));
    assert!(!worker_supports_job(
        &candle,
        &video_generate_job(json!({ "model": "wan_2_2", "mode": "image_to_video" }))
    ));
    assert!(!worker_supports_job(
        &candle,
        &video_generate_job(json!({ "model": "wan_2_2_i2v_14b", "mode": "text_to_video" }))
    ));
    // This synthetic generic-GPU descriptor can claim the legacy compatibility shape in isolation;
    // no deployed fallback worker registers it, so production leaves unsupported work queued.
    let torch = gpu_worker(TORCH_VIDEO_CAPS);
    assert!(worker_supports_job(
        &torch,
        &video_generate_job(
            json!({ "model": "wan_2_2", "mode": "image_to_video", "sourceAssetId": "a" })
        )
    ));
}

// ---- SeedVR2 video upscale (epic 4811 / sc-4816) ----

/// A queued `video_upscale` job carrying `payload`.
fn video_upscale_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_vu",
        "type": "video_upscale",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-13T00:00:00Z",
        "updatedAt": "2026-06-13T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// An idle MLX (`gpu_id = "mlx"`) worker advertising `capabilities`.
fn mlx_worker(capabilities: &[&str]) -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": "worker_mlx",
        "gpuId": "mlx",
        "status": "idle",
        "capabilities": capabilities,
        "loadedModels": [],
        "registeredAt": "2026-06-12T00:00:00Z",
        "lastSeenAt": "2026-06-12T00:00:00Z",
    }))
    .expect("valid WorkerSnapshot")
}

#[test]
fn video_upscale_seedvr2_is_mlx_eligible_other_engines_are_not() {
    // seedvr2 (alias + 3b id) and the absent-engine default are eligible.
    for engine in [json!("seedvr2"), json!("seedvr2_3b"), Value::Null] {
        let payload = if engine.is_null() {
            json!({ "sourceAssetId": "a" })
        } else {
            json!({ "sourceAssetId": "a", "engine": engine })
        };
        assert!(
            video_upscale_job_is_mlx_eligible(&video_upscale_job(payload.clone())),
            "video_upscale should be MLX-eligible for {payload}"
        );
    }
    // An unknown engine is not eligible (no torch video upscaler exists).
    assert!(!video_upscale_job_is_mlx_eligible(&video_upscale_job(
        json!({ "sourceAssetId": "a", "engine": "aura-sr" })
    )));
    // The predicate is gated to the job type.
    assert!(!video_upscale_job_is_mlx_eligible(&video_generate_job(
        json!({ "model": "wan_2_2" })
    )));
}

#[test]
fn mlx_worker_claims_seedvr2_video_upscale_and_refuses_other_engines() {
    let mlx = mlx_worker(&["gpu", "video_upscale"]);
    assert!(worker_supports_job(
        &mlx,
        &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
    ));
    // A non-SeedVR2 engine is refused by the MLX worker; the off-Mac Candle lane enforces the same
    // SeedVR2-only contract.
    assert!(!worker_supports_job(
        &mlx,
        &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
    ));
}

#[test]
fn video_upscale_requires_gpu() {
    assert!(job_requires_gpu(&JobType::VideoUpscale));
}

#[test]
fn mac_capabilities_advertises_video_upscale() {
    let caps = mac_capabilities("darwin", true);
    let feature = caps
        .features
        .get("videoUpscale")
        .expect("videoUpscale feature present");
    assert!(feature.supported);
    assert!(feature.reason.is_none());
}

#[test]
fn mac_rust_supports_seedvr2_video_upscale_only() {
    assert!(mac_rust_supported(&video_upscale_job(
        json!({ "sourceAssetId": "a", "engine": "seedvr2" })
    ))
    .is_ok());
    assert!(mac_rust_supported(&video_upscale_job(
        json!({ "sourceAssetId": "a", "engine": "aura-sr" })
    ))
    .is_err());
}

// ---- Candle SeedVR2 upscale lane (sc-5928) ----

/// A queued `image_upscale` job carrying `payload`.
fn image_upscale_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_iu",
        "type": "image_upscale",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-16T00:00:00Z",
        "updatedAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// sc-5499 + sc-5928: the candle worker claims both off-Mac image upscalers — Real-ESRGAN
/// (`ort`/CUDA, sc-5499, incl. the default engine) and SeedVR2 (`candle-gen-seedvr2`, sc-5928).
/// Only `aura-sr` (an offered engine dropped on every platform, sc-3668 / sc-5499) has no candle
/// path and is refused, so it remains queued.
#[test]
fn candle_worker_claims_real_esrgan_and_seedvr2_image_upscale_refuses_aura_sr() {
    let candle = gpu_worker(&["gpu", "image_upscale", "candle"]);
    assert!(worker_supports_job(
        &candle,
        &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
    ));
    // Real-ESRGAN (incl. the default engine) now has a candle path (the off-Mac ort/CUDA upscaler).
    assert!(worker_supports_job(
        &candle,
        &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "real-esrgan" }))
    ));
    assert!(worker_supports_job(
        &candle,
        &image_upscale_job(json!({ "sourceAssetId": "a" })) // default = real-esrgan
    ));
    // AuraSR is dropped as an offered engine → no candle path → refused.
    assert!(!worker_supports_job(
        &candle,
        &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
    ));
}

/// sc-5928: the candle worker claims the net-new SeedVR2 `video_upscale` (default/seedvr2 ids) and
/// refuses other engines, exactly like the mlx worker (the engine set is shared).
#[test]
fn candle_worker_claims_seedvr2_video_upscale_and_refuses_other_engines() {
    let candle = gpu_worker(&["gpu", "video_upscale", "candle"]);
    for engine in [json!("seedvr2"), json!("seedvr2_3b"), Value::Null] {
        let payload = if engine.is_null() {
            json!({ "sourceAssetId": "a" })
        } else {
            json!({ "sourceAssetId": "a", "engine": engine })
        };
        assert!(
            worker_supports_job(&candle, &video_upscale_job(payload.clone())),
            "candle should claim video_upscale for {payload}"
        );
    }
    assert!(!worker_supports_job(
        &candle,
        &video_upscale_job(json!({ "sourceAssetId": "a", "engine": "aura-sr" }))
    ));
}

/// sc-5928: a synthetic generic GPU descriptor (neither `mlx` nor candle) REFUSES a `seedvr2`
/// image upscale, so it stays queued for an mlx/candle worker. Real-ESRGAN exercises the generic
/// compatibility branch. The inverse of AuraSR.
#[test]
fn torch_worker_refuses_seedvr2_image_upscale_but_claims_real_esrgan() {
    let torch = gpu_worker(&["gpu", "image_upscale"]); // no candle marker, gpu_id != "mlx"
    assert!(!worker_supports_job(
        &torch,
        &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "seedvr2" }))
    ));
    assert!(worker_supports_job(
        &torch,
        &image_upscale_job(json!({ "sourceAssetId": "a", "engine": "real-esrgan" }))
    ));
}

// ---- Candle kps_extract lane (sc-5497, epic 5482) ----

/// A queued `kps_extract` job carrying `payload`.
fn kps_extract_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_kps",
        "type": "kps_extract",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-16T00:00:00Z",
        "updatedAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// sc-5497: the candle worker advertises `kps_extract` (the candle SCRFD/ArcFace face stack) and
/// claims a kps_extract job — the off-Mac sibling of the native-MLX path. The generic descriptor
/// below is a synthetic capability-routing check, not a deployed fallback worker. A worker that
/// never advertises the capability (e.g. a candle-disabled box) refuses it.
#[test]
fn candle_worker_claims_kps_extract_no_torch_refusal() {
    let payload = json!({ "sourceAssetId": "a", "projectId": "p" });
    let candle = gpu_worker(&["gpu", "kps_extract", "candle"]);
    assert!(
        worker_supports_job(&candle, &kps_extract_job(payload.clone())),
        "candle worker should claim kps_extract"
    );
    let torch = gpu_worker(&["gpu", "kps_extract"]);
    assert!(
        worker_supports_job(&torch, &kps_extract_job(payload.clone())),
        "torch worker still claims kps_extract (no refusal — it has the InsightFace path)"
    );
    let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
    assert!(
        !worker_supports_job(&no_cap, &kps_extract_job(payload)),
        "a worker not advertising kps_extract refuses it"
    );
}

// ---- Candle pose_detect (DWPose) lane (sc-5496, epic 5482) ----

/// A queued `pose_detect` job carrying `payload`.
fn pose_detect_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_pose",
        "type": "pose_detect",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-16T00:00:00Z",
        "updatedAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// sc-5496: the candle worker advertises `pose_detect` (the DWPose RTMW detector via the `ort` CUDA
/// EP) and claims a pose_detect job — the off-Mac sibling of the macOS `ort`/CoreML path. The generic
/// descriptor below is a synthetic capability-routing check, not a deployed fallback worker. A
/// worker that never advertises the capability (e.g. a candle-disabled box) refuses it.
#[test]
fn candle_worker_claims_pose_detect_no_torch_refusal() {
    let payload = json!({ "sources": [{ "assetId": "a" }], "projectId": "p" });
    let candle = gpu_worker(&["gpu", "pose_detect", "candle"]);
    assert!(
        worker_supports_job(&candle, &pose_detect_job(payload.clone())),
        "candle worker should claim pose_detect"
    );
    let torch = gpu_worker(&["gpu", "pose_detect"]);
    assert!(
        worker_supports_job(&torch, &pose_detect_job(payload.clone())),
        "torch worker still claims pose_detect (no refusal — it has the rtmlib path)"
    );
    let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
    assert!(
        !worker_supports_job(&no_cap, &pose_detect_job(payload)),
        "a worker not advertising pose_detect refuses it"
    );
}

// ---- Candle person detect/track lane (sc-5498) ----

/// A queued real (non-preview) `person_detect` job carrying `payload`.
fn person_detect_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_person_detect",
        "type": "person_detect",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-16T00:00:00Z",
        "updatedAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// A queued real (non-preview) `person_track` job carrying `payload`.
fn person_track_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_person_track",
        "type": "person_track",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-16T00:00:00Z",
        "updatedAt": "2026-06-16T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

/// sc-5498: the candle worker advertises `person_detect` + `person_track` (YOLO11 via the `ort`
/// CUDA EP + the pure-Rust ByteTrack) and claims both — the off-Mac sibling of the macOS
/// native-MLX path (sc-3633/sc-3634). Like kps_extract / pose_detect (and unlike SeedVR2), the
/// capability branch is also covered with a synthetic generic descriptor. Production has no
/// Python fallback; a worker that never advertises the capability refuses the job. (These are the real,
/// non-preview jobs; the procedural `preview: true` path keys off the separate
/// `person_detect_preview` / `person_track_preview` capabilities.)
#[test]
fn candle_worker_claims_person_detect_and_track_no_torch_refusal() {
    let payload = json!({ "projectId": "p", "sourceAssetId": "a" });
    let candle = gpu_worker(&["gpu", "person_detect", "person_track", "candle"]);
    assert!(
        worker_supports_job(&candle, &person_detect_job(payload.clone())),
        "candle worker should claim person_detect"
    );
    assert!(
        worker_supports_job(&candle, &person_track_job(payload.clone())),
        "candle worker should claim person_track"
    );
    let torch = gpu_worker(&["gpu", "person_detect", "person_track"]);
    assert!(
        worker_supports_job(&torch, &person_detect_job(payload.clone())),
        "torch worker still claims person_detect (no refusal — it has the Ultralytics path)"
    );
    assert!(
        worker_supports_job(&torch, &person_track_job(payload.clone())),
        "torch worker still claims person_track (no refusal — it has the Ultralytics path)"
    );
    let no_cap = gpu_worker(&["gpu", "image_generate", "candle"]);
    assert!(
        !worker_supports_job(&no_cap, &person_detect_job(payload.clone())),
        "a worker not advertising person_detect refuses it"
    );
    assert!(
        !worker_supports_job(&no_cap, &person_track_job(payload)),
        "a worker not advertising person_track refuses it"
    );
}

// ---- Candle caption lane (sc-5098) ----

/// A queued `training_caption` job carrying `payload`.
fn caption_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_c",
        "type": "training_caption",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-13T00:00:00Z",
        "updatedAt": "2026-06-13T00:00:00Z",
    }))
    .expect("valid JobSnapshot")
}

#[test]
fn candle_worker_claims_joycaption_but_refuses_other_captioners() {
    let candle = gpu_worker(&["gpu", "training_caption", "candle"]);
    // Claims a JoyCaption job.
    assert!(worker_supports_job(
        &candle,
        &caption_job(json!({ "captioner": "joy_caption", "datasetId": "ds_1" }))
    ));
    // Refuses a non-JoyCaption captioner; without another native captioner it remains queued.
    assert!(!worker_supports_job(
        &candle,
        &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
    ));
    let torch = gpu_worker(&["gpu", "training_caption"]);
    assert!(worker_supports_job(
        &torch,
        &caption_job(json!({ "captioner": "blip2", "datasetId": "ds_1" }))
    ));
}

/// sc-5501: the candle worker claims SenseNova-U1 `image_vqa` / `image_interleave` (served off-Mac
/// by the concrete candle `T2iModel::{vqa, interleave_gen}`) but refuses other models, which remain
/// queued without another compatible native worker.
#[test]
fn candle_worker_claims_sensenova_understanding_but_refuses_other_models() {
    let candle = gpu_worker(&["gpu", "image_vqa", "image_interleave", "candle"]);
    let understanding_job = |job_type: &str, payload: Value| -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job_u",
            "type": job_type,
            "status": "queued",
            "payload": payload,
            "result": {},
            "requestedGpu": "auto",
            "progress": 0,
            "stage": "queued",
            "message": "",
            "attempts": 1,
            "cancelRequested": false,
            "createdAt": "2026-06-14T00:00:00Z",
            "updatedAt": "2026-06-14T00:00:00Z",
        }))
        .expect("valid JobSnapshot")
    };
    // Claims SenseNova-U1 VQA + interleave (base + `_fast` ids).
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_vqa",
            json!({ "model": "sensenova_u1_8b", "question": "what is this?", "sourceAssetId": "a1" })
        )
    ));
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_interleave",
            json!({ "model": "sensenova_u1_8b_fast", "prompt": "a short illustrated story" })
        )
    ));
    // Infographic-V2 base advertises the SAME understanding surface (epic 9959): the eligibility
    // list must include its id, else V2 VQA / Document-Studio jobs never route to the in-process
    // worker (regression guard for the sc-9963 fix).
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_vqa",
            json!({ "model": "sensenova_u1_8b_infographic_v2", "question": "what is this?", "sourceAssetId": "a1" })
        )
    ));
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_interleave",
            json!({ "model": "sensenova_u1_8b_infographic_v2", "prompt": "an illustrated explainer" })
        )
    ));
    // Infographic-V3 base advertises the SAME understanding surface (epic 13095) — same regression
    // guard: its id must be in the eligibility list or V3 VQA / Document-Studio jobs never route.
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_vqa",
            json!({ "model": "sensenova_u1_8b_infographic_v3", "question": "what is this?", "sourceAssetId": "a1" })
        )
    ));
    assert!(worker_supports_job(
        &candle,
        &understanding_job(
            "image_interleave",
            json!({ "model": "sensenova_u1_8b_infographic_v3", "prompt": "an illustrated explainer" })
        )
    ));
    // Refuses a non-SenseNova understanding job; without another compatible native worker it remains queued.
    assert!(!worker_supports_job(
        &candle,
        &understanding_job(
            "image_vqa",
            json!({ "model": "some_other_vlm", "question": "?", "sourceAssetId": "a1" })
        )
    ));
}

#[test]
fn instantid_character_jobs_route_to_candle_off_mac() {
    // The candle InstantID provider (sc-5491) serves the SAME surface as the MLX path off-Mac, so
    // every character_image + referenceAssetId shape is candle-eligible — via the bespoke
    // `image_job_is_candle_eligible` branch, NOT the txt2img-only `image_request_candle_eligible`
    // gate (which rejects `referenceAssetId`, which InstantID requires).
    for advanced in [
        json!({}),
        json!({ "angleSet": true }),
        json!({ "poses": [{ "id": "a" }] }),
        json!({ "faceRestore": true }),
        json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
    ] {
        let payload = json!({
            "model": "instantid_realvisxl",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
            "advanced": advanced,
        });
        assert!(instantid_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    }

    // No reference face → not candle-eligible (mirrors the MLX gate).
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "instantid_realvisxl",
        "mode": "character_image"
    }))));
    // Non-character mode → not candle-eligible (InstantID is a character flow).
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "instantid_realvisxl",
        "mode": "text_to_image",
        "referenceAssetId": "asset_1"
    }))));
}

#[test]
fn sdxl_ipadapter_reference_jobs_route_to_candle() {
    // A pure SDXL/RealVisXL reference (IP-Adapter) job routes to the candle lane (sc-5488) via the
    // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
    // `referenceAssetId`).
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
        assert!(sdxl_ipadapter_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    }
    // No reference → not an IP-Adapter job (plain txt2img routes via the txt2img gate instead).
    assert!(!sdxl_ipadapter_candle_eligible(&object(
        json!({ "model": "sdxl" })
    )));
    // img2img / inpaint / edit shapes are NOT this lane; unsupported shapes remain queued.
    assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
        "model": "sdxl", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
        "model": "sdxl", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!sdxl_ipadapter_candle_eligible(&object(json!({
        "model": "sdxl", "referenceAssetId": "a", "maskAssetId": "m"
    }))));
}

#[test]
fn sdxl_edit_jobs_route_to_candle() {
    // SDXL/RealVisXL img2img / inpaint / outpaint edit jobs (sc-5487) route to the bespoke candle
    // `SdxlEdit` lane via the new branch, NOT the txt2img `image_request_candle_eligible` gate
    // (which rejects the whole `edit_image` family).
    for model in [
        "sdxl",
        "realvisxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
    ] {
        // img2img (source, no mask).
        let img2img = json!({ "model": model, "mode": "edit_image", "sourceAssetId": "src_1" });
        assert!(sdxl_edit_candle_eligible(&object(img2img.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(img2img)));
        // inpaint (source + mask).
        let inpaint = json!({
            "model": model, "mode": "edit_image", "sourceAssetId": "src_1", "maskAssetId": "m_1"
        });
        assert!(sdxl_edit_candle_eligible(&object(inpaint.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(inpaint)));
        // outpaint (source + fitMode outpaint).
        let outpaint = json!({
            "model": model, "mode": "edit_image", "sourceAssetId": "src_1", "fitMode": "outpaint"
        });
        assert!(sdxl_edit_candle_eligible(&object(outpaint.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(outpaint)));
    }
    // `edit_image` WITHOUT a source → not this lane (nothing to edit).
    assert!(!sdxl_edit_candle_eligible(&object(json!({
        "model": "sdxl", "mode": "edit_image"
    }))));
    // A reference (IP-Adapter) job is NOT the edit lane (no source, not `edit_image`) — it's sc-5488.
    assert!(!sdxl_edit_candle_eligible(&object(json!({
        "model": "sdxl", "referenceAssetId": "a"
    }))));
    // A plain txt2img sdxl job → not the edit lane.
    assert!(!sdxl_edit_candle_eligible(&object(
        json!({ "model": "sdxl" })
    )));
}

#[test]
fn zimage_edit_jobs_route_to_candle() {
    // Z-Image img2img / edit jobs (sc-6595) route to the bespoke candle `ZImageEdit` lane via the new
    // branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects `edit_image`). Both
    // the txt2img id in edit mode (`z_image_turbo`) and the dedicated `z_image_edit` id are served.
    for model in ["z_image_turbo", "z_image_edit"] {
        let edit = json!({ "model": model, "mode": "edit_image", "sourceAssetId": "src_1" });
        assert!(zimage_edit_candle_eligible(&object(edit.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(
            edit.clone()
        )));
        // Reached through the real `image_edit` job type too (the type the Image Editor submits).
        assert!(image_job_is_candle_eligible(&image_edit_job(edit)));
    }
    // `edit_image` WITHOUT a source → not this lane (nothing to edit).
    assert!(!zimage_edit_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "edit_image"
    }))));
    // A plain txt2img z_image_turbo job → not the edit lane (it routes via the txt2img gate instead).
    assert!(!zimage_edit_candle_eligible(&object(
        json!({ "model": "z_image_turbo" })
    )));
    // A z_image_turbo strict-pose job (advanced.poses, not edit_image) is the control lane, not edit.
    assert!(!zimage_edit_candle_eligible(&object(json!({
        "model": "z_image_turbo", "advanced": { "poses": [{}] }
    }))));
}

#[test]
fn zimage_identity_with_character_jobs_route_to_candle() {
    // Z-Image identity-init "With Character" jobs (sc-8409): a `z_image_turbo` `character_image` job
    // with a `referenceAssetId` + `advanced.referenceStrength > 0` routes to the bespoke candle
    // `ZImageEdit` identity lane via the new branch, NOT the txt2img `image_request_candle_eligible`
    // gate (which rejects any `referenceAssetId`). Without this the off-Mac job fell through to plain
    // txt2img, dropping the reference (no identity, no score).
    let with_character = json!({
        "model": "z_image_turbo",
        "mode": "character_image",
        "referenceAssetId": "asset_1",
        "advanced": { "referenceStrength": 0.6 }
    });
    assert!(zimage_identity_candle_eligible(&object(
        with_character.clone()
    )));
    assert!(image_job_is_candle_eligible(&image_generate_job(
        with_character
    )));
    // A numeric-string referenceStrength engages too (the web sends strings).
    assert!(zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image",
        "referenceAssetId": "asset_1", "advanced": { "referenceStrength": "0.45" }
    }))));

    // No referenceStrength (or <= 0) → stays plain txt2img on both backends (parity), NOT this lane.
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1"
    }))));
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image",
        "referenceAssetId": "asset_1", "advanced": { "referenceStrength": 0.0 }
    }))));
    // No reference face → no identity source → not this lane.
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image",
        "advanced": { "referenceStrength": 0.6 }
    }))));
    // Non-character mode → not this lane (an `edit_image` job is the edit lane, sc-6595).
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "edit_image",
        "referenceAssetId": "asset_1", "advanced": { "referenceStrength": 0.6 }
    }))));
    // Angle set + pose set are `character_image` too but route to their own lanes — excluded here so
    // this plain With-Character gate never steals them.
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1",
        "advanced": { "referenceStrength": 0.6, "angleSet": true }
    }))));
    assert!(!zimage_identity_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "character_image", "referenceAssetId": "asset_1",
        "advanced": { "referenceStrength": 0.6, "poses": [{ "id": "a" }] }
    }))));
}

#[test]
fn image_edit_job_type_routes_through_candle_edit_lane() {
    // Regression for the sc-5487 edit lanes being unreachable through the actual `image_edit` job
    // type the Image Editor submits (the prior tests only exercised `image_generate` jobs with
    // `mode == "edit_image"`, so the `JobType::ImageEdit`-only gap was invisible). A plain SDXL edit
    // submitted as `image_edit` must: be candle-eligible, survive the `candle_required` enforce sweep
    // (`candle_supported` → Ok), and be claimed by the candle worker — NOT enforce-failed
    // `candle_unsupported`.
    let sdxl_edit = json!({
        "model": "sdxl",
        "mode": "edit_image",
        "sourceAssetId": "asset_1"
    });
    assert!(
        image_job_is_candle_eligible(&image_edit_job(sdxl_edit.clone())),
        "an `image_edit`-typed SDXL edit must reach the candle SdxlEdit lane"
    );
    assert!(
        candle_supported(&image_edit_job(sdxl_edit.clone())).is_ok(),
        "an eligible `image_edit` SDXL job must not be enforce-failed candle_unsupported"
    );
    assert!(
        worker_supports_job(&gpu_worker(CANDLE_CAPS), &image_edit_job(sdxl_edit.clone())),
        "the candle worker (advertising `image_edit`) must claim the SDXL edit job"
    );
    // The FLUX.2-klein, Qwen-Image-Edit, and Z-Image edit lanes are reached through the same job type.
    for model in [
        "flux2_klein_9b",
        "qwen_image_edit",
        "qwen_image_edit_2511_lightning",
        "z_image_turbo",
        "z_image_edit",
    ] {
        let job = image_edit_job(json!({
            "model": model, "mode": "edit_image", "sourceAssetId": "asset_1"
        }));
        assert!(
            image_job_is_candle_eligible(&job) && candle_supported(&job).is_ok(),
            "{model} edit via the `image_edit` job type must reach its candle lane"
        );
    }
    // Kolors source img2img is now a registered Candle Reference route.
    let kolors_edit = json!({
        "model": "kolors",
        "mode": "edit_image",
        "sourceAssetId": "asset_1"
    });
    assert!(image_job_is_candle_eligible(&image_edit_job(
        kolors_edit.clone()
    )));
    assert!(worker_supports_job(
        &gpu_worker(CANDLE_CAPS),
        &image_edit_job(kolors_edit.clone())
    ));
    // An `image_edit` job with no source image is not the edit lane → not candle-eligible.
    assert!(!image_job_is_candle_eligible(&image_edit_job(json!({
        "model": "sdxl", "mode": "edit_image"
    }))));
}

#[test]
fn kolors_ipadapter_reference_jobs_route_to_candle() {
    // A pure Kolors reference (IP-Adapter) job routes to the candle lane (sc-5488) via the bespoke
    // branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects `referenceAssetId`).
    let payload = json!({ "model": "kolors", "referenceAssetId": "asset_1" });
    assert!(kolors_ipadapter_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // No reference → plain txt2img routes via the txt2img gate instead.
    assert!(!kolors_ipadapter_candle_eligible(&object(
        json!({ "model": "kolors" })
    )));
    // img2img / inpaint / edit shapes are NOT this lane; unsupported shapes remain queued.
    assert!(!kolors_ipadapter_candle_eligible(&object(json!({
        "model": "kolors", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!kolors_ipadapter_candle_eligible(&object(json!({
        "model": "kolors", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!kolors_ipadapter_candle_eligible(&object(json!({
        "model": "kolors", "referenceAssetId": "a", "maskAssetId": "m"
    }))));
}

#[test]
fn flux_ipadapter_reference_jobs_route_to_candle() {
    // A pure FLUX reference (XLabs IP-Adapter) job routes to the candle lane (sc-5872) via the
    // bespoke branch, NOT the txt2img `image_request_candle_eligible` gate (which rejects
    // `referenceAssetId`). Both variants.
    for model in ["flux_dev", "flux_schnell"] {
        let payload = json!({ "model": model, "referenceAssetId": "asset_1" });
        assert!(flux_ipadapter_candle_eligible(&object(payload.clone())));
        assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    }
    // No reference → plain txt2img routes via the txt2img gate instead.
    assert!(!flux_ipadapter_candle_eligible(&object(
        json!({ "model": "flux_dev" })
    )));
    // img2img / inpaint / edit shapes are NOT this lane; unsupported shapes remain queued.
    assert!(!flux_ipadapter_candle_eligible(&object(json!({
        "model": "flux_dev", "mode": "edit_image", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!flux_ipadapter_candle_eligible(&object(json!({
        "model": "flux_dev", "referenceAssetId": "a", "sourceAssetId": "s"
    }))));
    assert!(!flux_ipadapter_candle_eligible(&object(json!({
        "model": "flux_schnell", "referenceAssetId": "a", "maskAssetId": "m"
    }))));
}

#[test]
fn pulid_flux_character_jobs_route_to_candle_off_mac() {
    // The candle PuLID-FLUX provider (sc-5492) serves the SAME surface as the MLX path off-Mac, so
    // a `pulid_flux_dev` character_image + referenceAssetId job is candle-eligible via the bespoke
    // `image_job_is_candle_eligible` branch, NOT the txt2img-only `image_request_candle_eligible`
    // gate (which rejects `referenceAssetId`, which PuLID requires). The distinct `pulid_flux_dev`
    // model id cleanly disambiguates it from the FLUX XLabs IP-Adapter lane (`flux_dev`).
    let payload = json!({
        "model": "pulid_flux_dev",
        "mode": "character_image",
        "referenceAssetId": "asset_1",
    });
    assert!(pulid_flux_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));

    // No reference face → not candle-eligible (mirrors the MLX gate).
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "pulid_flux_dev",
        "mode": "character_image"
    }))));
    // Non-character mode → not candle-eligible (PuLID is a character flow).
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "pulid_flux_dev",
        "mode": "text_to_image",
        "referenceAssetId": "asset_1"
    }))));
}

#[test]
fn qwen_control_pose_jobs_route_to_candle() {
    // qwen_image + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
    // branch, NOT the txt2img gate (which refuses any `advanced.poses` job).
    let payload = json!({ "model": "qwen_image", "advanced": { "poses": [{ "keypoints": [] }] } });
    assert!(qwen_control_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
    assert!(!qwen_control_candle_eligible(&object(
        json!({ "model": "qwen_image" })
    )));
    assert!(!qwen_control_candle_eligible(&object(json!({
        "model": "qwen_image", "advanced": { "poses": [] }
    }))));
    // edit_image with poses is NOT this lane.
    assert!(!qwen_control_candle_eligible(&object(json!({
        "model": "qwen_image", "mode": "edit_image", "advanced": { "poses": [{}] }
    }))));
    // Plain `sdxl` + poses is NOT candle-*served* (no plain-SDXL pose lane — SDXL pose ships via
    // InstantID): the qwen branch is specific and the txt2img gate's has_poses check rejects it, so
    // `image_job_is_candle_eligible` is false. (It is, however, candle-*owned-to-reject* at the
    // worker layer per sc-5968 — see `unsupported_pose_is_owned_by_candle_*`; that claim lives in
    // `worker_supports_job`, not here. z_image_turbo + poses IS a candle lane — `zimage_control_*`.)
    assert!(!image_job_is_candle_eligible(&image_generate_job(json!({
        "model": "sdxl", "advanced": { "poses": [{}] }
    }))));
}

#[test]
fn kolors_control_pose_jobs_route_to_candle() {
    // kolors + advanced.poses routes to the candle strict-pose lane (sc-5489) via the bespoke
    // branch, NOT the txt2img gate (which refuses any `advanced.poses` job).
    let payload = json!({ "model": "kolors", "advanced": { "poses": [{ "keypoints": [] }] } });
    assert!(kolors_control_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
    assert!(!kolors_control_candle_eligible(&object(
        json!({ "model": "kolors" })
    )));
    assert!(!kolors_control_candle_eligible(&object(json!({
        "model": "kolors", "advanced": { "poses": [] }
    }))));
    // edit_image with poses is NOT this lane.
    assert!(!kolors_control_candle_eligible(&object(json!({
        "model": "kolors", "mode": "edit_image", "advanced": { "poses": [{}] }
    }))));
    // A kolors reference job (no poses) still routes via the IP-Adapter branch, not this one.
    assert!(!kolors_control_candle_eligible(&object(json!({
        "model": "kolors", "referenceAssetId": "asset_1"
    }))));
}

#[test]
fn zimage_control_pose_jobs_route_to_candle() {
    // z_image_turbo + advanced.poses routes to the candle VACE strict-pose lane (sc-5489, the last
    // family) via the bespoke branch, NOT the txt2img gate (which refuses any `advanced.poses` job).
    let payload =
        json!({ "model": "z_image_turbo", "advanced": { "poses": [{ "keypoints": [] }] } });
    assert!(zimage_control_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // No poses (or empty) → plain txt2img routes via the txt2img gate instead.
    assert!(!zimage_control_candle_eligible(&object(
        json!({ "model": "z_image_turbo" })
    )));
    assert!(!zimage_control_candle_eligible(&object(json!({
        "model": "z_image_turbo", "advanced": { "poses": [] }
    }))));
    // edit_image with poses is NOT this lane.
    assert!(!zimage_control_candle_eligible(&object(json!({
        "model": "z_image_turbo", "mode": "edit_image", "advanced": { "poses": [{}] }
    }))));
}

#[test]
fn zimage_base_control_pose_jobs_route_to_candle() {
    // sc-8379: the BASE z_image model + advanced.poses routes to the same candle strict-control lane
    // as Turbo (the base Fun-Controlnet-Union branch) via the bespoke branch, NOT the txt2img gate.
    let payload = json!({ "model": "z_image", "advanced": { "poses": [{ "keypoints": [] }] } });
    assert!(zimage_control_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // A base z_image with no poses is plain txt2img — now a candle lane too (sc-8679: the registered
    // candle `z_image` base generator), so it routes to the generic candle txt2img gate. It is NOT
    // this strict-control lane, though.
    let plain = json!({ "model": "z_image", "prompt": "a misty fjord" });
    assert!(!zimage_control_candle_eligible(&object(plain.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(plain)));
    // Both Turbo and base have a candle pose lane (so neither is pose-rejected).
    assert!(model_has_candle_pose_lane("z_image"));
    assert!(model_has_candle_pose_lane("z_image_turbo"));
}

#[test]
fn flux1_dev_control_pose_jobs_route_to_candle() {
    // sc-8412: flux_dev + advanced.poses routes to the candle Shakker Union-Pro-2.0 strict-control
    // lane via the bespoke branch, NOT the txt2img gate (which refuses any `advanced.poses` job).
    let payload = json!({ "model": "flux_dev", "advanced": { "poses": [{ "keypoints": [] }] } });
    assert!(flux1_control_candle_eligible(&object(payload.clone())));
    assert!(image_job_is_candle_eligible(&image_generate_job(payload)));
    // No poses → plain txt2img routes via the txt2img gate instead.
    assert!(!flux1_control_candle_eligible(&object(
        json!({ "model": "flux_dev" })
    )));
    assert!(!flux1_control_candle_eligible(&object(json!({
        "model": "flux_dev", "advanced": { "poses": [] }
    }))));
    // edit_image with poses is NOT this lane.
    assert!(!flux1_control_candle_eligible(&object(json!({
        "model": "flux_dev", "mode": "edit_image", "advanced": { "poses": [{}] }
    }))));
    // A flux_dev reference job (no poses) routes via the FLUX XLabs IP-Adapter branch, not this one.
    assert!(!flux1_control_candle_eligible(&object(json!({
        "model": "flux_dev", "referenceAssetId": "asset_1"
    }))));
    // flux_dev now HAS a candle pose lane (so it is not pose-rejected); schnell does not.
    assert!(model_has_candle_pose_lane("flux_dev"));
    assert!(!model_has_candle_pose_lane("flux_schnell"));
}

// ---------------------------------------------------------------------------
// sc-16260: an unusable GPU must not be routed generation it is certain to fail.
// ---------------------------------------------------------------------------

/// CHARACTERIZATION of the routing consequence (sc-16260 AC 2), from the store's side.
///
/// It asserts pre-existing `worker_supports_job` behaviour against the two capability sets rather
/// than any code this story added — the probe→withholding link itself is pinned by
/// `an_unusable_gpu_withholds_every_candle_capability` in the worker crate. Its value here is that
/// it fixes the OTHER end of the contract: it turns red if the withheld set ever becomes
/// claimable, which is the assumption the whole design rests on and which lives in a different
/// crate from the code that produces it.
///
/// The `healthy` side of each pair is the control: it proves the job is genuinely claimable by a
/// full candle worker, so a `false` from the degraded worker means "withheld", not "this job was
/// never routable here anyway".
#[test]
fn a_gpu_worker_that_withheld_its_candle_capabilities_is_routed_no_generation() {
    // Exactly what `with_candle_capabilities` leaves behind when the probe fails, and exactly what
    // a non-candle build advertises: no job capability at all.
    const WITHHELD_CAPS: &[&str] = &["placeholder", "gpu", "nvidia"];
    let degraded = gpu_worker(WITHHELD_CAPS);
    let healthy = gpu_worker(CANDLE_CAPS);

    for job in [
        image_generate_job(json!({"model": "sdxl", "prompt": "a cat"})),
        image_edit_job(json!({
            "model": "z_image_edit",
            "mode": "edit_image",
            "sourceAssetId": "asset_1",
        })),
    ] {
        assert!(
            worker_supports_job(&healthy, &job),
            "control: a healthy candle worker must claim {} — otherwise the assertion below is \
             vacuous",
            job.job_type.as_str()
        );
        assert!(
            !worker_supports_job(&degraded, &job),
            "a worker whose CUDA probe failed withheld its capabilities, so {} must stay QUEUED \
             rather than be claimed and failed",
            job.job_type.as_str()
        );
    }
}

/// THE BACKSTOP (sc-16260). Independent of the capability half: a worker that has declared itself
/// `unhealthy` is handed nothing, even while still advertising the full candle set.
///
/// That combination is reachable in practice — registration and heartbeat are separate round trips,
/// so a worker that goes unhealthy AFTER registering (or whose unhealthy reason is one whose
/// capability impact we don't yet know how to trim) still has its advertisement on file. Pinned
/// against the SAME capability set that the healthy control claims with, so deleting the status
/// check in `worker_supports_job` turns this red rather than shifting it onto the capability gate.
#[test]
fn an_unhealthy_worker_is_routed_nothing_even_with_full_capabilities() {
    let job = image_generate_job(json!({"model": "sdxl", "prompt": "a cat"}));

    let idle = gpu_worker_with_status(CANDLE_CAPS, "idle");
    assert!(
        worker_supports_job(&idle, &job),
        "control: the identical worker must claim this job when idle"
    );

    let unhealthy = gpu_worker_with_status(CANDLE_CAPS, "unhealthy");
    assert!(
        !worker_supports_job(&unhealthy, &job),
        "an unhealthy worker must be routed nothing, even while its registration still advertises \
         the capability"
    );

    // "I cannot run work" is the whole claim the status makes, so it holds for the non-GPU job
    // types too — not just the ones that need an accelerator.
    let download: JobSnapshot = serde_json::from_value(json!({
        "id": "job_2",
        "type": "model_download",
        "status": "queued",
        "payload": {"modelId": "sdxl_base"},
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-06-12T00:00:00Z",
        "updatedAt": "2026-06-12T00:00:00Z",
    }))
    .expect("valid JobSnapshot");
    const UTILITY_CAPS: &[&str] = &["gpu", "model_download", "candle"];
    assert!(
        worker_supports_job(&gpu_worker_with_status(UTILITY_CAPS, "idle"), &download),
        "control: the identical worker must claim the download when idle"
    );
    assert!(
        !worker_supports_job(
            &gpu_worker_with_status(UTILITY_CAPS, "unhealthy"),
            &download
        ),
        "an unhealthy worker must not claim utility work either"
    );

    // And the ordinary statuses are untouched — this must not have become a general status filter.
    for status in ["idle", "busy"] {
        assert!(
            worker_supports_job(&gpu_worker_with_status(CANDLE_CAPS, status), &job),
            "{status} routing must be unchanged by the unhealthy backstop"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// sc-20530: `candle_unsupported` message classification. Routing is NOT under test here (the gate
// tests above own that); these pin WHICH cause the terminal error blames. The 2026-08-18 sweep
// showed five families rejected with "conditioned shape on a txt2img candle family" for payloads
// that carried no conditioning at all — the real cause was `advanced.mlxQuantize` on a dense-only
// family. A message naming a conditioning bug that does not exist sends triage hunting one.
// ---------------------------------------------------------------------------------------------

/// The two dense families that remain in this wording regression (`flux_dev`, `flux_schnell`) advertise
/// `supported_quants: &[]` — dense bf16/fp16 only — so `advanced.mlxQuantize: 4` is correctly
/// refused rather than silently run dense (sc-5099). The refusal must SAY that: name the requested
/// tier and what the family does serve, and never claim a conditioning shape the payload lacks.
#[test]
fn quant_request_on_a_dense_only_candle_family_names_the_tier_not_a_conditioned_shape() {
    for model in ["flux_dev", "flux_schnell"] {
        for bits in [4, 8] {
            let job = image_generate_job(json!({
                "projectId": "project_1",
                "model": model,
                "prompt": "a red fox",
                "mode": "text_to_image",
                "referenceAssetId": null,
                "sourceAssetId": null,
                "maskAssetId": null,
                "advanced": { "mlxQuantize": bits }
            }));
            // Routing is unchanged: the request is still refused (the sc-5099 no-silent-dense posture).
            assert!(
                !image_job_is_candle_eligible(&job),
                "{model} q{bits} must stay refused — this story changes wording only"
            );
            let reason = candle_supported(&job).expect_err("a quant request must stay a named gap");
            let message = reason.candle_error_message();
            assert!(
                reason.feature.contains("quant"),
                "{model} q{bits} must be blamed on the quant tier: {reason:?}"
            );
            assert!(
                message.contains(&format!("q{bits}")),
                "the message must name the REQUESTED tier q{bits}: {message}"
            );
            assert!(
                message.contains("mlxQuantize"),
                "the message must name the payload field that caused it: {message}"
            );
            assert!(
                message.contains("dense"),
                "the message must say what the family DOES serve (dense only): {message}"
            );
            assert!(
                !message.contains("conditioned shape"),
                "an unconditioned payload must never be blamed on a conditioning shape: {message}"
            );
            assert!(
                !message.contains("edit / reference / inpaint"),
                "the message must not list causes the payload does not carry: {message}"
            );
        }
    }
}

/// AC-4 mutation guard: the classification split must not move a single routing decision. The same
/// five families with the quant override REMOVED still route to candle, and `mlxQuantize: 0` (the
/// dense encoding) still routes too — only the refusal STRING changed.
#[test]
fn quant_message_split_did_not_move_the_routing_decision() {
    for model in [
        "chroma1_base",
        "chroma1_flash",
        "chroma1_hd",
        "flux_dev",
        "flux_schnell",
    ] {
        let dense = image_generate_job(json!({
            "projectId": "project_1",
            "model": model,
            "prompt": "a red fox",
            "mode": "text_to_image",
            "referenceAssetId": null,
            "sourceAssetId": null,
            "maskAssetId": null
        }));
        assert!(
            image_job_is_candle_eligible(&dense),
            "{model} plain dense txt2img must still route to candle"
        );
        assert!(
            candle_supported(&dense).is_ok(),
            "{model} plain dense txt2img must not be a gap"
        );
        let zero = image_generate_job(json!({
            "projectId": "project_1",
            "model": model,
            "prompt": "a red fox",
            "mode": "text_to_image",
            "advanced": { "mlxQuantize": 0 }
        }));
        assert!(
            image_job_is_candle_eligible(&zero) && candle_supported(&zero).is_ok(),
            "{model} mlxQuantize:0 is dense and must still route"
        );
    }
}

/// A user LoRA on a candle family with no adapter lane is its own cause — not a "conditioned
/// shape". `boogu_image` is quant-capable but adapter-less (`CANDLE_QUANT_MODELS`, not the
/// quant+LoRA set), so the adapter half of the gate alone refuses it.
#[test]
fn lora_on_a_candle_family_with_no_adapter_lane_is_named_separately() {
    let job = image_generate_job(json!({
        "projectId": "project_1",
        "model": "boogu_image",
        "prompt": "a red fox",
        "mode": "text_to_image",
        "loras": [{ "id": "lora_1", "scale": 0.8 }]
    }));
    assert!(
        !image_job_is_candle_eligible(&job),
        "routing is unchanged: an adapter-less family still refuses a LoRA"
    );
    let reason = candle_supported(&job).expect_err("a LoRA gap must stay named");
    let message = reason.candle_error_message();
    assert!(
        reason.feature.contains("LoRA"),
        "the LoRA cause must be named: {reason:?}"
    );
    assert!(
        !message.contains("conditioned shape"),
        "a LoRA request is not a conditioning shape: {message}"
    );
    assert!(
        !message.contains("quant tier"),
        "the message must not blame a quant tier the payload never asked for: {message}"
    );
}

/// An edit/inpaint carrier on a family with no candle edit lane keeps a conditioning-flavored
/// message — but it names the carriers the payload ACTUALLY has instead of listing five candidates.
#[test]
fn edit_carrier_on_a_family_without_an_edit_lane_names_the_carrier() {
    let job = image_edit_job(json!({
        "projectId": "project_1",
        "model": "chroma1_base",
        "prompt": "a red fox",
        "mode": "edit_image",
        "sourceAssetId": "asset_1"
    }));
    assert!(
        !image_job_is_candle_eligible(&job),
        "routing is unchanged: chroma has no candle edit lane"
    );
    let reason = candle_supported(&job).expect_err("an edit gap must stay named");
    let message = reason.candle_error_message();
    assert!(
        message.contains("sourceAssetId"),
        "the message must name the carrier the payload carries: {message}"
    );
    assert!(
        !message.contains("mlxQuantize"),
        "the message must not blame a quant tier the payload never asked for: {message}"
    );
}

/// The catch-all is now a true last resort: reached only when no distinguishable cause applies, and
/// it must not assert a conditioning shape for a payload that carries none. Probed directly because
/// production has no reachable unconditioned-and-refused image shape today — which is exactly why
/// the catch-all's claim has to stay honest if one ever appears.
#[test]
fn catch_all_does_not_claim_a_conditioned_shape_for_an_unconditioned_payload() {
    use crate::jobs_store::routing::gaps::classify_candle_image_gap;

    let reason = classify_candle_image_gap(&object(json!({
        "model": "chroma1_base",
        "prompt": "a red fox",
        "mode": "text_to_image",
        "referenceAssetId": null,
        "sourceAssetId": null,
        "maskAssetId": null,
        "loras": []
    })));
    let message = reason.candle_error_message();
    assert!(
        !message.contains("conditioned shape"),
        "an unconditioned payload must not be blamed on a conditioning shape: {message}"
    );
    assert!(
        message.contains("no reference / source / mask / LoRA / pose"),
        "the catch-all must state what the payload does NOT carry: {message}"
    );

    // A reference-only mode with no carrier is its own cause, not the catch-all.
    let modeless = classify_candle_image_gap(&object(json!({
        "model": "flux2_dev",
        "prompt": "a red fox",
        "mode": "style_variations",
        "referenceAssetId": null
    })));
    assert!(
        modeless.detail.contains("style_variations"),
        "a reference-only mode with no carrier must name the mode: {modeless:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-20530 adversarial-review follow-up. The wording tests above pin WHICH cause is blamed for one
// payload each; these pin the ORDER the causes are blamed in, which is the property the doc comment
// on `classify_candle_image_gap` actually claims: "the reported cause is the FIRST check that
// actually refused, in `image_request_candle_eligible`'s own evaluation order".
//
// The order is no longer hand-kept in two places — the gate and the classifier both walk
// `CANDLE_IMAGE_CHECKS` — but a shared table can still be re-ordered by accident, and the review
// found two payload shapes where the old hand-kept order was measurably wrong:
//   * `advanced.phases` refuses BEFORE quant, so a payload carrying both was told to drop
//     `mlxQuantize` — advice that does not route the request;
//   * the reference-only modes refuse only on the families that RESERVE them, so a
//     `style_variations` chroma request was told it was missing a conditioning image when the mode
//     was not what refused at all.
// ---------------------------------------------------------------------------------------------

/// `classify_candle_image_gap` on a raw payload, returning `(feature, detail)`. Asserts the gate
/// really refuses the payload first, so no step can drift into pinning the wording of a shape that
/// actually routes.
fn candle_gap(model: &str, payload: Value) -> (String, String) {
    use crate::jobs_store::routing::gaps::classify_candle_image_gap;

    let mut payload = object(payload);
    payload.insert("model".to_owned(), Value::String(model.to_owned()));
    // Every real submission arrives with the optional scalar carriers stamped null (sc-20525).
    for key in ["sourceAssetId", "referenceAssetId", "maskAssetId"] {
        payload.entry(key.to_owned()).or_insert(Value::Null);
    }
    assert!(
        !image_request_candle_eligible(model, &payload),
        "the peel step must still be refused by the gate: {payload:?}"
    );
    let reason = classify_candle_image_gap(&payload);
    (reason.feature.clone(), reason.detail.clone())
}

/// Reviewer repro 1. `flux_schnell` t2i with BOTH `advanced.mlxQuantize` and `advanced.phases`: the
/// gate refuses on phases (candle.rs `Phases`) before it ever reaches the quant check, so blaming
/// the quant tier attached a remediation ("re-submit without advanced.mlxQuantize and the same
/// request routes to candle") that is FALSE — phases alone is still refused.
#[test]
fn phases_are_blamed_before_quant_because_the_gate_refuses_them_first() {
    let both = json!({
        "prompt": "a red fox",
        "mode": "text_to_image",
        "advanced": { "mlxQuantize": 4, "phases": [{ "steps": 4 }] }
    });
    let (feature, detail) = candle_gap("flux_schnell", both.clone());
    assert!(
        feature.contains("phases"),
        "the FIRST refusing check is advanced.phases, not the quant tier: {feature}"
    );
    assert!(
        !detail.contains("re-submit without advanced.phases"),
        "removing phases does NOT route this request (the quant tier still refuses it), so the \
         message must not promise it does: {detail}"
    );
    assert!(
        detail.contains("advanced.mlxQuantize"),
        "the message must name the OTHER check that also refuses, so the reader is not sent \
         through a fix that leaves the request refused: {detail}"
    );

    // Peel the phases off and the quant tier becomes the first refusal — with its remediation back,
    // because now it really is the only one.
    let (feature, detail) = candle_gap(
        "flux_schnell",
        json!({
            "prompt": "a red fox",
            "mode": "text_to_image",
            "advanced": { "mlxQuantize": 4 }
        }),
    );
    assert!(feature.contains("q4 quant tier"), "{feature}");
    assert!(
        detail.contains("re-submit without advanced.mlxQuantize and the same request routes"),
        "the quant remediation is honest once quant is the ONLY refusal: {detail}"
    );

    // …and peeling the quant off routes it, which is what makes that remediation true.
    assert!(
        image_request_candle_eligible(
            "flux_schnell",
            &object(json!({ "prompt": "a red fox", "mode": "text_to_image" }))
        ),
        "flux_schnell plain t2i with neither phases nor quant must route to candle"
    );
}

/// Reviewer repro 2. The gate's reference-only-mode refusal applies ONLY to the families that
/// reserve those modes for a specialized lane (flux2_* / qwen_image_edit* / sensenova_u1_8b*). On
/// `flux_schnell` a `style_variations` request with every carrier null is NOT refused for the mode —
/// the quant tier is what refuses it — so a "needs a source/reference image" message names a cause
/// the gate never applied.
#[test]
fn a_reference_mode_is_blamed_only_on_the_families_that_reserve_it() {
    let (feature, detail) = candle_gap(
        "flux_schnell",
        json!({
            "prompt": "a red fox",
            "mode": "style_variations",
            "advanced": { "mlxQuantize": 4 }
        }),
    );
    assert!(
        feature.contains("q4 quant tier"),
        "flux_schnell does not reserve style_variations, so the quant tier is the first refusal: \
         {feature}"
    );
    assert!(
        !detail.contains("needs a source/reference image"),
        "the mode is not what refused on this family — claiming a missing conditioning image sends \
         triage after a bug that does not exist: {detail}"
    );

    // The same mode on a family that DOES reserve it keeps the conditioning-image message.
    let (feature, detail) = candle_gap(
        "flux2_dev",
        json!({ "prompt": "a red fox", "mode": "style_variations" }),
    );
    assert!(
        feature.contains("conditioning mode"),
        "flux2_dev reserves style_variations for its specialized lane: {feature}"
    );
    assert!(detail.contains("style_variations"), "{detail}");

    // Every reserved family + mode pair, so the predicate cannot be narrowed without a red test.
    // The `qwen_image_edit*` ids are reserved by the gate but are NOT in `CANDLE_ROUTED_MODELS` at
    // all, and that check runs FIRST (candle.rs returns before the mode is ever read) — so the
    // classifier must blame the missing family, not the mode. Both halves are pinned here: this is
    // exactly the class of "a later check's message for an earlier check's refusal" the review
    // found, and it would read as a fixable payload problem on a family that has no lane at all.
    for model in [
        "flux2_dev",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
        "qwen_image_edit",
        "qwen_image_edit_2509",
        "qwen_image_edit_2511",
        "qwen_image_edit_2511_lightning",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
    ] {
        let routed = CANDLE_ROUTED_MODELS.contains(&model);
        for mode in [
            "reference",
            "image_to_image",
            "character_image",
            "style_variations",
        ] {
            let (feature, _) = candle_gap(model, json!({ "prompt": "p", "mode": mode }));
            if routed {
                assert!(
                    feature.contains("conditioning mode"),
                    "{model} reserves {mode} for a specialized lane: {feature}"
                );
            } else {
                assert!(
                    feature.contains("unsupported image model"),
                    "{model} has no candle lane at all, which the gate checks BEFORE the mode: \
                     {feature}"
                );
            }
        }
    }
}

/// The whole order, peeled one cause at a time. Each step asserts the gate still refuses AND that
/// the classifier names the check `image_request_candle_eligible` reaches first; removing that one
/// cause moves the blame to the next. `flux_schnell` advertises inference LoRA but no quant tier,
/// so its peel covers edit-mode → carrier → poses → phases → quant. The `loras` entry rides along
/// the whole way to prove a check that does NOT refuse never steals the blame.
#[test]
fn the_reported_cause_walks_the_gate_order_as_each_cause_is_peeled_off() {
    let steps: [(Value, &str); 5] = [
        (
            json!({
                "prompt": "p",
                "mode": "edit_image",
                "sourceAssetId": "asset_1",
                "loras": [{ "id": "lora_1" }],
                "advanced": {
                    "poses": [{ "id": "pose_1" }],
                    "phases": [{ "steps": 4 }],
                    "mlxQuantize": 4
                }
            }),
            "edit_image",
        ),
        (
            json!({
                "prompt": "p",
                "mode": "text_to_image",
                "sourceAssetId": "asset_1",
                "loras": [{ "id": "lora_1" }],
                "advanced": {
                    "poses": [{ "id": "pose_1" }],
                    "phases": [{ "steps": 4 }],
                    "mlxQuantize": 4
                }
            }),
            "conditioning carrier",
        ),
        (
            json!({
                "prompt": "p",
                "mode": "text_to_image",
                "loras": [{ "id": "lora_1" }],
                "advanced": {
                    "poses": [{ "id": "pose_1" }],
                    "phases": [{ "steps": 4 }],
                    "mlxQuantize": 4
                }
            }),
            "strict-pose",
        ),
        (
            json!({
                "prompt": "p",
                "mode": "text_to_image",
                "loras": [{ "id": "lora_1" }],
                "advanced": { "phases": [{ "steps": 4 }], "mlxQuantize": 4 }
            }),
            "phases",
        ),
        (
            json!({
                "prompt": "p",
                "mode": "text_to_image",
                "loras": [{ "id": "lora_1" }],
                "advanced": { "mlxQuantize": 4 }
            }),
            "quant tier",
        ),
    ];
    for (payload, expected) in steps {
        let (feature, _) = candle_gap("flux_schnell", payload.clone());
        assert!(
            feature.contains(expected),
            "expected the {expected:?} check to be blamed for {payload}: got {feature}"
        );
    }
    // The last peel routes: nothing else in the gate refuses it.
    assert!(image_request_candle_eligible(
        "flux_schnell",
        &object(json!({
            "prompt": "p",
            "mode": "text_to_image",
            "loras": [{ "id": "lora_1" }]
        }))
    ));

    // `boogu_image` is the mirror family — quant-capable, adapter-less — so its peel is what pins
    // the user-LoRA check's position between the carrier check and the pose check.
    let (feature, _) = candle_gap(
        "boogu_image",
        json!({
            "prompt": "p",
            "mode": "text_to_image",
            "referenceAssetId": "asset_1",
            "loras": [{ "id": "lora_1" }],
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }),
    );
    assert!(feature.contains("conditioning carrier"), "{feature}");
    let (feature, _) = candle_gap(
        "boogu_image",
        json!({
            "prompt": "p",
            "mode": "text_to_image",
            "loras": [{ "id": "lora_1" }],
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }),
    );
    assert!(feature.contains("LoRA"), "{feature}");
    let (feature, _) = candle_gap(
        "boogu_image",
        json!({
            "prompt": "p",
            "mode": "text_to_image",
            "advanced": { "poses": [{ "id": "pose_1" }] }
        }),
    );
    assert!(feature.contains("strict-pose"), "{feature}");

    // A reference-only mode on a family that reserves it precedes the carrier check, mirroring the
    // gate (candle.rs refuses the mode before it looks at carriers at all).
    let (feature, _) = candle_gap(
        "flux2_dev",
        json!({
            "prompt": "p",
            "mode": "style_variations",
            "sourceAssetId": "asset_1"
        }),
    );
    assert!(
        feature.contains("reference mode"),
        "the reserved mode is refused before the carrier check: {feature}"
    );

    // And the SANA carrier check precedes everything: a source carrier on SANA is blamed on the
    // SANA lane, not on the generic carrier check that would refuse it two entries later.
    let (feature, _) = candle_gap(
        "sana_1600m",
        json!({
            "prompt": "p",
            "mode": "text_to_image",
            "sourceAssetId": "asset_1",
            "advanced": { "phases": [{ "steps": 4 }] }
        }),
    );
    assert!(feature.contains("SANA"), "{feature}");
}

/// The sc-20525 malformed-carrier wording is reachable exactly where the gate refuses ON a
/// malformed carrier: the SANA lane. Everywhere else a malformed scalar reads as "absent" to the
/// gate, so if such a payload is refused at all something ELSE refused it — and the malformed
/// message's "the same request with a valid or absent carrier may well route" would be false.
#[test]
fn a_malformed_carrier_is_named_only_where_it_is_what_refused() {
    let (feature, detail) = candle_gap(
        "sana_1600m",
        json!({ "prompt": "p", "mode": "text_to_image", "referenceAssetId": 42 }),
    );
    assert!(feature.contains("malformed"), "{feature}");
    assert!(detail.contains("referenceAssetId"), "{detail}");

    // The same malformed carrier on flux1 does not refuse anything — the gate reads a non-string
    // scalar as absent — so a flux request carrying it AND a quant tier is blamed on the quant
    // tier, the check that actually refused.
    let (feature, detail) = candle_gap(
        "flux_schnell",
        json!({
            "prompt": "p",
            "mode": "text_to_image",
            "sourceAssetId": 42,
            "advanced": { "mlxQuantize": 4 }
        }),
    );
    assert!(feature.contains("q4 quant tier"), "{feature}");
    assert!(
        !detail.contains("malformed"),
        "the malformed carrier is not what refused this request: {detail}"
    );
    // Proof that the carrier really is inert on this family: drop the quant tier and it routes.
    assert!(image_request_candle_eligible(
        "flux_schnell",
        &object(json!({ "prompt": "p", "mode": "text_to_image", "sourceAssetId": 42 }))
    ));
}
