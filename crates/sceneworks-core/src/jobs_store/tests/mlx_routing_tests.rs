use super::{
    flux2_mlx_eligible, flux_mlx_eligible, image_job_is_mlx_eligible, image_request_mlx_eligible,
    instantid_mlx_eligible, model_mac_support, qwen_edit_mlx_eligible, qwen_mlx_eligible,
    sdxl_mlx_eligible, video_mode_is_mlx_eligible, worker_supports_job, z_image_mlx_eligible,
    JobSnapshot, WorkerSnapshot, CANDLE_VIDEO_ROUTED_MODELS, MLX_ROUTED_MODELS,
    VIDEO_MLX_ROUTED_MODELS,
};
use serde_json::{json, Map, Value};

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("test value is an object").clone()
}

fn image_generate_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_imported_krea",
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
        "createdAt": "2026-07-23T00:00:00Z",
        "updatedAt": "2026-07-23T00:00:00Z"
    }))
    .expect("valid image job")
}

fn mlx_worker() -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": "worker_mlx",
        "gpuId": "mlx",
        "status": "idle",
        "capabilities": ["gpu", "image_generate"],
        "loadedModels": [],
        "registeredAt": "2026-07-23T00:00:00Z",
        "lastSeenAt": "2026-07-23T00:00:00Z"
    }))
    .expect("valid MLX worker")
}

#[test]
fn imported_krea_family_plain_single_file_job_is_mlx_eligible() {
    let plain = json!({
        "projectId": "project_1",
        "model": "kreamania_variant4",
        "prompt": "a red fox",
        "modelManifestEntry": {
            "family": "krea_2",
            "paths": { "model": "/app/models/imports/kreamania_variant4" }
        }
    });

    assert!(
        !MLX_ROUTED_MODELS.contains(&"kreamania_variant4"),
        "the builtin id table must not accidentally contain the imported id"
    );
    assert!(
        image_request_mlx_eligible(
            "kreamania_variant4",
            plain.as_object().expect("payload object")
        ),
        "the public MLX predicate must apply the imported-family fallback"
    );
    let job = image_generate_job(plain.clone());
    assert!(
        image_job_is_mlx_eligible(&job),
        "the full scheduler must claim a novel imported id through family routing"
    );
    assert!(
        worker_supports_job(&mlx_worker(), &job),
        "an MLX worker must claim the imported family-routed job"
    );

    // On MLX the imported native loader takes adapters (inference #211), so a LoRA t2i job
    // (sc-14111), an img2img job (sc-14071), and the Kontext edit surface (sc-14119) are all
    // claim-eligible via the family fallback.
    let entry = json!({
        "family": "krea_2",
        "paths": { "model": "/app/models/imports/kreamania_variant4" }
    });
    for (label, extra) in [
        ("lora t2i", json!({ "loras": [{ "id": "adapter_1" }] })),
        ("img2img", json!({ "referenceAssetId": "reference_1" })),
        (
            "edit",
            json!({ "mode": "edit_image", "sourceAssetId": "source_1" }),
        ),
    ] {
        let mut payload = json!({ "model": "kreamania_variant4", "prompt": "a red fox" });
        payload
            .as_object_mut()
            .unwrap()
            .insert("modelManifestEntry".to_owned(), entry.clone());
        payload
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(
            image_job_is_mlx_eligible(&image_generate_job(payload)),
            "{label} imported job must be MLX-eligible via the adapter-capable family fallback"
        );
    }

    // A strict-pose set is now served too: the trained pose control branch folds onto the
    // file-loaded imported DiT (`load_control_from_native_dit_file`), so a same-shape Krea
    // fine-tune renders pose-locked sets exactly as the builtin Turbo base does.
    assert!(image_job_is_mlx_eligible(&image_generate_job(json!({
        "model": "kreamania_variant4",
        "advanced": { "poses": [{ "id": "pose_1" }] },
        "modelManifestEntry": {
            "family": "krea_2",
            "paths": { "model": "/app/models/imports/kreamania_variant4" }
        }
    }))));
    // A shape the pose render loop would silently drop (the plural edit reference set) and a
    // manifest entry with no installed path stay ineligible.
    assert!(!image_job_is_mlx_eligible(&image_generate_job(json!({
        "model": "kreamania_variant4",
        "referenceAssetIds": ["reference_1"],
        "advanced": { "poses": [{ "id": "pose_1" }] },
        "modelManifestEntry": {
            "family": "krea_2",
            "paths": { "model": "/app/models/imports/kreamania_variant4" }
        }
    }))));
    assert!(!image_job_is_mlx_eligible(&image_generate_job(json!({
        "model": "kreamania_variant4",
        "modelManifestEntry": { "family": "krea_2" }
    }))));
}

#[test]
fn z_image_plain_txt2img_is_eligible() {
    assert!(z_image_mlx_eligible(&object(
        json!({ "prompt": "a misty fjord" })
    )));
    assert!(z_image_mlx_eligible(&Map::new()));
}

#[test]
fn z_image_edit_mode_with_source_is_eligible() {
    // epic 3529: img2img-edit (sourceAssetId) now routes to MLX via the engine's
    // `Conditioning::Reference` img2img path.
    assert!(z_image_mlx_eligible(&object(json!({
        "mode": "edit_image",
        "sourceAssetId": "asset_1"
    }))));
}

#[test]
fn z_image_edit_mode_without_source_is_not_eligible() {
    // An edit with nothing to edit (no/blank sourceAssetId) stays off MLX.
    assert!(!z_image_mlx_eligible(&object(
        json!({ "mode": "edit_image" })
    )));
    assert!(!z_image_mlx_eligible(&object(json!({
        "mode": "edit_image",
        "sourceAssetId": "   "
    }))));
}

#[test]
fn z_image_reference_without_poses_is_eligible() {
    // sc-3619: reference-identity img2img-init (no pose) now routes to MLX — the
    // base engine already supports the plain img2img path, and torch dropped the
    // reference entirely (it was a no-op on the fallback).
    assert!(z_image_mlx_eligible(&object(
        json!({ "referenceAssetId": "asset_1" })
    )));
    // Empty/whitespace reference id is treated as absent → plain txt2img, eligible.
    assert!(z_image_mlx_eligible(&object(
        json!({ "referenceAssetId": "   " })
    )));
    // A reference with empty poses is still reference-only → eligible (not the
    // pose tier, which needs a non-empty pose set).
    assert!(z_image_mlx_eligible(&object(json!({
        "referenceAssetId": "asset_1",
        "advanced": { "poses": [] }
    }))));
}

#[test]
fn z_image_reference_with_poses_stays_on_mlx() {
    // The strict pose ControlNet tier lives only on MLX, so a reference+pose
    // job must route to the mlx worker; a generic claimant could otherwise drop the poses.
    assert!(z_image_mlx_eligible(&object(json!({
        "referenceAssetId": "asset_1",
        "advanced": { "poses": [{ "id": "pose_1" }] }
    }))));
}

#[test]
fn z_image_peft_lokr_and_thirdparty_lycoris_both_route_mlx() {
    // SceneWorks peft LoKr applies natively on the MLX Z-Image path → eligible.
    assert!(z_image_mlx_eligible(&object(json!({
        "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
    }))));
    // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX too.
    assert!(z_image_mlx_eligible(&object(json!({
        "loras": [{ "path": "b.safetensors", "networkType": "lycoris" }]
    }))));
}

#[test]
fn flux_plain_txt2img_is_eligible() {
    assert!(flux_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
    assert!(flux_mlx_eligible(&Map::new()));
    // A LoRA is fine on the MLX flux path (engine applies LoRA + peft LoKr).
    assert!(flux_mlx_eligible(&object(json!({
        "loras": [{ "path": "a.safetensors", "networkType": "lora" }]
    }))));
}

#[test]
fn flux_reference_is_eligible() {
    // Reference (XLabs IP-Adapter, epic 3621) now routes to MLX on both variants —
    // the Rust engine has no diffusers schnell limitation.
    assert!(flux_mlx_eligible(&object(
        json!({ "referenceAssetId": "asset_1" })
    )));
    // A reference + LoRA is still fine.
    assert!(flux_mlx_eligible(&object(json!({
        "referenceAssetId": "asset_1",
        "loras": [{ "networkType": "lora" }]
    }))));
}

#[test]
fn flux_only_edit_falls_back_lycoris_routes_mlx() {
    // edit_image (no FLUX.1 edit on any platform — future Kontext) is the only fall-back.
    assert!(!flux_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
    // Third-party LyCORIS now applies via the core MLX loader (epic 3641) → MLX.
    assert!(flux_mlx_eligible(&object(json!({
        "loras": [{ "networkType": "lycoris" }]
    }))));
    // Reference + a LyCORIS LoRA also routes MLX now.
    assert!(flux_mlx_eligible(&object(json!({
        "referenceAssetId": "asset_1",
        "loras": [{ "networkType": "lycoris" }]
    }))));
}

#[test]
fn qwen_plain_txt2img_is_eligible() {
    assert!(qwen_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
    // A negative prompt + LoRA are fine on the MLX qwen path (true CFG + LoRA wired).
    assert!(qwen_mlx_eligible(&object(json!({
        "negativePrompt": "blurry",
        "loras": [{ "networkType": "lokr" }]
    }))));
}

#[test]
fn qwen_edit_reference_falls_back_but_pose_and_lycoris_route_mlx() {
    assert!(!qwen_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
    assert!(!qwen_mlx_eligible(&object(
        json!({ "referenceAssetId": "asset_1" })
    )));
    // Strict pose ControlNet (sc-2291 / sc-3575) routes to MLX, even if a reference is
    // present; the strict-pose tier is pose-from-prompt and ignores the reference.
    assert!(qwen_mlx_eligible(&object(json!({
        "advanced": { "poses": [{ "id": "p1" }] }
    }))));
    assert!(qwen_mlx_eligible(&object(json!({
        "referenceAssetId": "asset_1",
        "advanced": { "poses": [{ "id": "p1" }] }
    }))));
    // Third-party LyCORIS on a plain txt2img qwen job now routes MLX (epic 3641).
    assert!(qwen_mlx_eligible(&object(json!({
        "loras": [{ "networkType": "lycoris" }]
    }))));
}

#[test]
fn qwen_edit_routes_edit_and_reference_flows_to_mlx() {
    // sc-3397: the qwen_image_edit ids run the engine's `qwen_image_edit` model.
    // edit_image with a source → eligible.
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "edit_image", "sourceAssetId": "src_1"
    }))));
    // character_image with a reference (subject variation) → eligible.
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "character_image", "referenceAssetId": "ref_1"
    }))));
    // character_image + reference + best-effort poses → still eligible. Unlike the base
    // Qwen strict-pose ControlNet (torch until epic 3401), the edit best-effort pose tier
    // is native multi-image ([reference, skeleton]) → MLX.
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "character_image", "referenceAssetId": "ref_1",
        "advanced": { "poses": [{ "id": "p1" }] }
    }))));
    // character_image + reference + angle set → eligible.
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "character_image", "referenceAssetId": "ref_1",
        "advanced": { "angleSet": true }
    }))));
    // A peft LoKr is fine on the MLX edit path.
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "edit_image", "sourceAssetId": "src_1",
        "loras": [{ "networkType": "lokr" }]
    }))));
}

#[test]
fn qwen_edit_without_reference_falls_back_to_torch() {
    // edit_image with nothing to edit (no source, no reference) is refused and remains queued.
    assert!(!qwen_edit_mlx_eligible(&object(
        json!({ "mode": "edit_image" })
    )));
    // character_image without a reference is refused and remains queued (the edit model needs a reference).
    assert!(!qwen_edit_mlx_eligible(&object(
        json!({ "mode": "character_image" })
    )));
    // A plain txt2img mode is not an edit job (that's the base qwen_image MLX path).
    assert!(!qwen_edit_mlx_eligible(&object(json!({
        "mode": "text_to_image", "sourceAssetId": "src_1"
    }))));
    // Whitespace-only ids are treated as absent.
    assert!(!qwen_edit_mlx_eligible(&object(json!({
        "mode": "edit_image", "sourceAssetId": "   "
    }))));
    // A third-party LyCORIS LoRA on an otherwise-eligible edit job now routes MLX (epic 3641).
    assert!(qwen_edit_mlx_eligible(&object(json!({
        "mode": "edit_image", "sourceAssetId": "src_1",
        "loras": [{ "networkType": "lycoris" }]
    }))));
}

#[test]
fn flux2_txt2img_edit_and_lycoris_all_route_mlx() {
    // FLUX.2 is MLX-only: txt2img (sc-3025), edit/reference (sc-3029), and — since epic 3641 —
    // third-party LyCORIS all route MLX.
    assert!(flux2_mlx_eligible(&object(
        json!({ "prompt": "a red fox" })
    )));
    assert!(flux2_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
    assert!(flux2_mlx_eligible(&object(
        json!({ "referenceAssetId": "asset_1" })
    )));
    assert!(flux2_mlx_eligible(&object(json!({
        "loras": [{ "networkType": "lycoris" }]
    }))));
}

#[test]
fn sdxl_eligible_for_txt2img_edit_reference_lokr_and_lycoris() {
    assert!(sdxl_mlx_eligible(&object(json!({ "prompt": "a red fox" }))));
    // peft LoKr stays on MLX (the Rust SDXL path supports LoKr, unlike the old vendored path).
    assert!(sdxl_mlx_eligible(&object(json!({
        "loras": [{ "networkType": "lokr" }]
    }))));
    // sc-3060: the Rust engine now handles the advanced shapes, so edit_image
    // (img2img / inpaint / outpaint) and reference/IP-Adapter route to MLX too.
    assert!(sdxl_mlx_eligible(&object(json!({ "mode": "edit_image" }))));
    assert!(sdxl_mlx_eligible(&object(
        json!({ "referenceAssetId": "asset_1" })
    )));
    assert!(sdxl_mlx_eligible(&object(json!({
        "mode": "edit_image",
        "maskAssetId": "mask_1"
    }))));
    // Third-party LyCORIS now applies on the SDXL merge path (epic 3641, sc-3671) → MLX,
    // including on an edit job.
    assert!(sdxl_mlx_eligible(&object(json!({
        "loras": [{ "networkType": "lycoris" }]
    }))));
    assert!(sdxl_mlx_eligible(&object(json!({
        "mode": "edit_image",
        "loras": [{ "networkType": "lycoris" }]
    }))));
}

#[test]
fn instantid_routes_all_character_modes_to_mlx() {
    // The full InstantID surface is native (sc-3345 identity + angle; sc-3381 pose + restore):
    // every character_image + referenceAssetId shape routes to MLX.
    for advanced in [
        json!({}),
        json!({ "angleSet": true }),
        json!({ "poses": [{ "id": "a" }] }),
        json!({ "faceRestore": true }),
        json!({ "poses": [{ "id": "a" }], "faceRestore": true }),
    ] {
        let payload = object(json!({
            "model": "instantid_realvisxl",
            "mode": "character_image",
            "referenceAssetId": "asset_1",
            "advanced": advanced,
        }));
        assert!(instantid_mlx_eligible(&payload));
        assert!(image_request_mlx_eligible("instantid_realvisxl", &payload));
    }

    // No reference face → not eligible.
    assert!(!instantid_mlx_eligible(&object(json!({
        "model": "instantid_realvisxl",
        "mode": "character_image"
    }))));

    // Non-character mode → not eligible (InstantID is a character flow).
    assert!(!instantid_mlx_eligible(&object(json!({
        "model": "instantid_realvisxl",
        "mode": "text_to_image"
    }))));
}

#[test]
fn ideogram_4_text_to_image_and_edit_route_to_mlx() {
    // sc-6302 + sc-6303: `ideogram_4` is in MLX_ROUTED_MODELS, and the native engine now serves
    // both text-to-image and img2img/mask-inpaint edit — both route to the in-process MLX worker.
    assert!(image_request_mlx_eligible(
        "ideogram_4",
        &object(json!({ "prompt": "a neon city skyline" }))
    ));
    assert!(image_request_mlx_eligible("ideogram_4", &Map::new()));
    // Edit (img2img / inpaint) now routes to MLX (sc-6303 — `resolve_ideogram_edit`).
    assert!(image_request_mlx_eligible(
        "ideogram_4",
        &object(json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }))
    ));

    // The UI gating oracle: Ideogram 4 is macSupport.supported (reaches the Text → Image picker)
    // and `features.edit` is now true (drives the Image Studio Edit tab alongside the catalog
    // `edit_image` capability). `reference`/`pose` remain true — inert, since the catalog
    // capabilities (not these flags) drive the UI affordances.
    let support = model_mac_support("ideogram_4", "image", None);
    assert!(support.supported, "ideogram_4 must be Mac-supported");
    assert!(
        support.features.edit,
        "ideogram_4 now supports edit (sc-6303)"
    );

    // Turbo is the same base model + the bundled TurboTime LoRA, so it routes + edits identically
    // (sc-6303). It was never registered in core before this (sc-6302 added only the base id), so
    // this also restores its Text → Image picker visibility.
    assert!(image_request_mlx_eligible("ideogram_4_turbo", &Map::new()));
    assert!(image_request_mlx_eligible(
        "ideogram_4_turbo",
        &object(json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }))
    ));
    let turbo = model_mac_support("ideogram_4_turbo", "image", None);
    assert!(turbo.supported, "ideogram_4_turbo must be Mac-supported");
    assert!(turbo.features.edit, "ideogram_4_turbo supports edit");
}

#[test]
fn boogu_text_to_image_and_edit_route_to_mlx() {
    // sc-6399 (epic 6387): the three Boogu ids are in MLX_ROUTED_MODELS and route to the native
    // `mlx-gen-boogu` engine. Base + Turbo are text-to-image; Edit is the instruction image-edit.
    for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
        assert!(
            image_request_mlx_eligible(model, &object(json!({ "model": model, "prompt": "p" }))),
            "{model} text-to-image must route to MLX"
        );
        assert!(
            image_request_mlx_eligible(model, &Map::new()),
            "{model} bare payload"
        );
    }

    // Edit routes to MLX for the Edit checkpoint only — Base/Turbo are text-to-image (their
    // semantic-edit path is incoherent without the Edit fine-tune, E7b-3).
    let edit_payload = |model: &str| {
        object(json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" }))
    };
    assert!(image_request_mlx_eligible(
        "boogu_image_edit",
        &edit_payload("boogu_image_edit")
    ));
    assert!(!image_request_mlx_eligible(
        "boogu_image",
        &edit_payload("boogu_image")
    ));
    assert!(!image_request_mlx_eligible(
        "boogu_image_turbo",
        &edit_payload("boogu_image_turbo")
    ));

    // UI gating oracle: all three are Mac-supported (reach the Text → Image picker); only Edit
    // advertises `features.edit` (Base/Turbo are T2I — the catalog `edit_image` capability +
    // this flag both gate the Edit tab to `boogu_image_edit`).
    for model in ["boogu_image", "boogu_image_turbo", "boogu_image_edit"] {
        assert!(
            model_mac_support(model, "image", None).supported,
            "{model} must be Mac-supported"
        );
    }
    assert!(
        model_mac_support("boogu_image_edit", "image", None)
            .features
            .edit,
        "boogu_image_edit supports edit"
    );
    assert!(
        !model_mac_support("boogu_image", "image", None)
            .features
            .edit,
        "boogu_image (Base) is text-to-image only"
    );
    assert!(
        !model_mac_support("boogu_image_turbo", "image", None)
            .features
            .edit,
        "boogu_image_turbo is text-to-image only"
    );
}

#[test]
fn krea_2_turbo_text_to_image_and_edit_route_to_mlx() {
    // sc-7572: Krea 2 Turbo has a native `mlx-gen-krea` text-to-image engine. sc-11640: it ALSO serves
    // the Kontext-style edit surface on the CFG-free distilled few-step recipe (`krea_2_turbo_edit`) —
    // the fast-path counterpart to the ~52-step Raw edit — so an `edit_image` job WITH a source routes
    // to the edit lane and `features.edit` flips true, exactly like Raw. Without a source it is rejected.
    assert!(image_request_mlx_eligible(
        "krea_2_turbo",
        &object(json!({ "model": "krea_2_turbo", "prompt": "cinematic editorial portrait" }))
    ));
    assert!(image_request_mlx_eligible("krea_2_turbo", &Map::new()));
    assert!(
        image_request_mlx_eligible(
            "krea_2_turbo",
            &object(
                json!({ "model": "krea_2_turbo", "mode": "edit_image", "sourceAssetId": "asset_1" })
            )
        ),
        "krea_2_turbo edit_image with a source must route to the edit lane (sc-11640)"
    );
    assert!(
        !image_request_mlx_eligible(
            "krea_2_turbo",
            &object(json!({ "model": "krea_2_turbo", "mode": "edit_image" }))
        ),
        "krea_2_turbo edit_image without a source is rejected"
    );

    let support = model_mac_support("krea_2_turbo", "image", None);
    assert!(support.supported, "krea_2_turbo must be Mac-supported");
    assert!(
        support.features.edit,
        "krea_2_turbo advertises the edit tab (sc-11640)"
    );
}

#[test]
fn krea_2_raw_edit_image_routes_to_mlx() {
    // epic 10871 (sc-10882): Krea 2 Raw serves BOTH text-to-image and the Kontext-style edit
    // surface. A t2i job routes to MLX; an `edit_image` job WITH a source routes to the edit lane;
    // an `edit_image` job WITHOUT a source is rejected (the defensive shape). Both Krea variants edit
    // (Raw full-CFG, Turbo CFG-free sc-11640), so the `features.edit` oracle flips true for each.
    assert!(image_request_mlx_eligible(
        "krea_2_raw",
        &object(json!({ "model": "krea_2_raw", "prompt": "full-CFG editorial portrait" }))
    ));
    assert!(image_request_mlx_eligible("krea_2_raw", &Map::new()));
    assert!(
        image_request_mlx_eligible(
            "krea_2_raw",
            &object(
                json!({ "model": "krea_2_raw", "mode": "edit_image", "sourceAssetId": "asset_1" })
            )
        ),
        "krea_2_raw edit_image with a source must route to the edit lane"
    );
    assert!(
        !image_request_mlx_eligible(
            "krea_2_raw",
            &object(json!({ "model": "krea_2_raw", "mode": "edit_image" }))
        ),
        "krea_2_raw edit_image without a source is rejected"
    );

    let support = model_mac_support("krea_2_raw", "image", None);
    assert!(support.supported, "krea_2_raw must be Mac-supported");
    assert!(
        support.features.edit,
        "krea_2_raw advertises the edit tab (epic 10871)"
    );
}

#[test]
fn sd3_5_text_to_image_routes_to_mlx() {
    // sc-7873 (epic 7841): the three SD3.5 variants have native `mlx-gen-sd3` text-to-image engines
    // (S2 MODEL_TABLE), so they must reach the Text → Image picker (macSupport.supported) rather than
    // being hidden as torch-only. All three are text-to-image only — `edit_image` is rejected.
    for model in ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"] {
        assert!(
            image_request_mlx_eligible(
                model,
                &object(json!({ "model": model, "prompt": "a misty alpine lake at dawn" }))
            ),
            "{model} text-to-image must route to MLX"
        );
        assert!(
            image_request_mlx_eligible(model, &Map::new()),
            "{model} bare payload"
        );
        assert!(
            !image_request_mlx_eligible(
                model,
                &object(
                    json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset_1" })
                )
            ),
            "{model} edit is not supported (text-to-image only)"
        );

        // UI gating oracle: Mac-supported (reaches the picker), text-to-image only (no edit tab).
        let support = model_mac_support(model, "image", None);
        assert!(support.supported, "{model} must be Mac-supported");
        assert!(!support.features.edit, "{model} is text-to-image only");
    }
}

#[test]
fn video_mode_eligibility_admits_flf_only_on_flf_capable_engines() {
    // image_to_video is MLX on every routed model EXCEPT Bernini (text_to_video only — its
    // renderer is Wan2.2-T2V, no still-image-to-video), SCAIL-2 (animate_character only) and
    // Mochi (text_to_video only — `conditioning: []` on both descriptors, sc-11991);
    // text_to_video on every routed model EXCEPT SVD (image-conditioned only, sc-3523) and
    // SCAIL-2 (animate_character only — sc-5448).
    // The models with their OWN arm, which the two generic-arm assertions below must exclude:
    // bernini / scail2_14b / mochi_1 (as above), `wan_2_2_vace_fun_14b` (replace_person ONLY — the
    // dual-expert control checkpoint, sc-3458) and `minimax_h3_ref` (reference_to_video ONLY — the
    // `transformer_ref` partition, sc-17159). `minimax_h3` itself DOES serve both generic modes, so
    // it is deliberately absent from these lists and is asserted by the generic arm.
    for model in VIDEO_MLX_ROUTED_MODELS {
        assert_eq!(
            video_mode_is_mlx_eligible(model, "image_to_video"),
            !matches!(
                *model,
                "bernini" | "scail2_14b" | "mochi_1" | "wan_2_2_vace_fun_14b" | "minimax_h3_ref"
            ),
            "image_to_video eligibility for {model}"
        );
        assert_eq!(
            video_mode_is_mlx_eligible(model, "text_to_video"),
            !matches!(
                *model,
                "svd" | "scail2_14b" | "wan_2_2_vace_fun_14b" | "minimax_h3_ref"
            ),
            "text_to_video eligibility for {model}"
        );
    }
    // SVD serves image_to_video ONLY — no text_to_video, FLF, or anything else.
    assert!(video_mode_is_mlx_eligible("svd", "image_to_video"));
    for mode in [
        "text_to_video",
        "first_last_frame",
        "replace_person",
        "nonsense",
    ] {
        assert!(!video_mode_is_mlx_eligible("svd", mode));
    }
    // Bernini serves text_to_video + the planner editing/reference video modes (sc-4703:
    // video_to_video / reference_to_video / reference_video_to_video) + the multi-source
    // modes (sc-5425: multi_video_to_video / ads2v). It has no classic still-image-to-video
    // / FLF / replace_person (its renderer is Wan2.2-T2V).
    for mode in [
        "text_to_video",
        "video_to_video",
        "reference_to_video",
        "reference_video_to_video",
        "multi_video_to_video",
        "ads2v",
    ] {
        assert!(
            video_mode_is_mlx_eligible("bernini", mode),
            "bernini should serve {mode}"
        );
    }
    for mode in [
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
        "nonsense",
    ] {
        assert!(
            !video_mode_is_mlx_eligible("bernini", mode),
            "bernini should not serve {mode}"
        );
    }
    // The reference + multi-source modes are Bernini-only — every other routed model rejects
    // them. `video_to_video` is deliberately NOT in this list: it is Bernini-plus-one since
    // sc-8444, because Krea Realtime's descriptor takes a `VideoClip` source to drive its
    // strength-controlled AR init, so it genuinely serves v2v too (see
    // `krea_realtime_is_mlx_routed_and_serves_exactly_its_advertised_modes`, which pins the
    // whole per-model v2v column so this relaxation cannot widen into the generic arm).
    for model in VIDEO_MLX_ROUTED_MODELS {
        if *model == "bernini" {
            continue;
        }
        if *model != "krea_realtime_14b" {
            assert!(
                !video_mode_is_mlx_eligible(model, "video_to_video"),
                "video_to_video should be Bernini/Krea-Realtime-only, not eligible on {model}"
            );
        }
        for mode in [
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ] {
            // `reference_to_video` is Bernini-plus-one since sc-17159, for the same reason
            // `video_to_video` is: MiniMax-H3's `transformer_ref` partition is a real Ref2VA
            // checkpoint, so `minimax_h3_ref` genuinely serves it. The OTHER three stay
            // Bernini-only, and `minimax_h3_ref` is asserted to refuse them by
            // `minimax_h3_partitions_are_mlx_routed_and_serve_exactly_their_declared_capabilities`
            // — so this relaxation cannot widen into "the reference family serves everything".
            if *model == "minimax_h3_ref" && mode == "reference_to_video" {
                continue;
            }
            assert!(
                !video_mode_is_mlx_eligible(model, mode),
                "{mode} should be Bernini-only, not eligible on {model}"
            );
        }
    }
    // SCAIL-2 serves the standalone character-animation mode (sc-5448, the worker paints its
    // masks from native SAM3) AND cross-identity replace_person (sc-5452, the integrated backend
    // behind the person-track pipeline). No text/image-to-video.
    for mode in ["animate_character", "replace_person"] {
        assert!(
            video_mode_is_mlx_eligible("scail2_14b", mode),
            "scail2 should serve {mode}"
        );
    }
    for mode in [
        "text_to_video",
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "video_to_video",
        "nonsense",
    ] {
        assert!(
            !video_mode_is_mlx_eligible("scail2_14b", mode),
            "scail2 should not serve {mode}"
        );
    }
    // animate_character is SCAIL-2-only — every other routed model rejects it.
    for model in VIDEO_MLX_ROUTED_MODELS {
        if *model == "scail2_14b" {
            continue;
        }
        assert!(
            !video_mode_is_mlx_eligible(model, "animate_character"),
            "animate_character should be SCAIL-2-only, not eligible on {model}"
        );
    }
    // Mochi 1 (sc-11991) serves text_to_video and NOTHING else: both descriptors declare
    // `conditioning: []`, so the engine has no image/keyframe/clip path — it must not fall
    // through to the generic `text_to_video | image_to_video => true` arm.
    assert!(
        video_mode_is_mlx_eligible("mochi_1", "text_to_video"),
        "mochi should serve text_to_video"
    );
    for mode in [
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
        "animate_character",
        "video_to_video",
        "nonsense",
    ] {
        assert!(
            !video_mode_is_mlx_eligible("mochi_1", mode),
            "mochi is text_to_video only and should not serve {mode}"
        );
    }
    // first_last_frame: MLX on LTX (base + eros) + Wan TI2V-5B (sc-3055 cutover).
    assert!(video_mode_is_mlx_eligible("ltx_2_3", "first_last_frame"));
    assert!(video_mode_is_mlx_eligible(
        "ltx_2_3_eros",
        "first_last_frame"
    ));
    assert!(video_mode_is_mlx_eligible("wan_2_2", "first_last_frame"));
    // …and on MiniMax-H3's t2va/fl2va partition (sc-17159): fl2va is `first_last_frame` with 0, 1
    // or 2 keyframes, and the generic arm's LTX/Wan-only list would have refused a mode this
    // family's `capabilities` and `ui.recommendedFor` both advertise.
    assert!(video_mode_is_mlx_eligible("minimax_h3", "first_last_frame"));
    // The REFERENCE partition is a different checkpoint and must NOT inherit it.
    assert!(!video_mode_is_mlx_eligible(
        "minimax_h3_ref",
        "first_last_frame"
    ));
    // FLF remains queued on the 14B Wan MoE engines (no native engine Keyframe path).
    assert!(!video_mode_is_mlx_eligible(
        "wan_2_2_t2v_14b",
        "first_last_frame"
    ));
    assert!(!video_mode_is_mlx_eligible(
        "wan_2_2_i2v_14b",
        "first_last_frame"
    ));
    // extend_clip / video_bridge: MLX on the LTX IC-LoRA path (sc-3522) and Wan TI2V-5B
    // (`wan_2_2`, single-frame boundary keyframe conditioning — sc-3357).
    for mode in ["extend_clip", "video_bridge"] {
        assert!(video_mode_is_mlx_eligible("ltx_2_3", mode));
        assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", mode));
        assert!(video_mode_is_mlx_eligible("wan_2_2", mode));
        // The 14B Wan MoE engines have no `Keyframe` path, so the request is refused.
        assert!(!video_mode_is_mlx_eligible("wan_2_2_t2v_14b", mode));
        assert!(!video_mode_is_mlx_eligible("wan_2_2_i2v_14b", mode));
    }
    // replace_person → native Wan-VACE is MLX on the replace-capable models (sc-3521).
    assert!(video_mode_is_mlx_eligible("ltx_2_3", "replace_person"));
    assert!(video_mode_is_mlx_eligible("ltx_2_3_eros", "replace_person"));
    assert!(video_mode_is_mlx_eligible("wan_2_2", "replace_person"));
    // …and on `wan_2_2_vace_fun_14b`, which routes to its OWN dual-expert engine
    // (`VideoRoute::ReplacePersonWanVaceFun`, sc-3459) rather than to single-expert `wan_vace`.
    // It advertises `replace_person` and nothing else, so this arm being false made its ONLY
    // capability unreachable on its ONLY lane (sc-17159).
    assert!(video_mode_is_mlx_eligible(
        "wan_2_2_vace_fun_14b",
        "replace_person"
    ));
    // Neither MiniMax-H3 partition serves replace_person — it declares no such capability, and
    // routing it would hand a Wan-VACE request to a MiniMax checkpoint.
    for id in ["minimax_h3", "minimax_h3_ref"] {
        assert!(!video_mode_is_mlx_eligible(id, "replace_person"));
    }
    // Unknown modes are never eligible.
    assert!(!video_mode_is_mlx_eligible("ltx_2_3", "nonsense"));
}

/// sc-8444 (epic 8431) — Krea Realtime 14B is MLX-ROUTED and serves exactly the three modes its
/// catalog entry advertises.
///
/// Both halves are regressions this test exists to prevent, and both are silent:
///
/// 1. **Absent from `VIDEO_MODEL_CAPS`** ⇒ `video_job_is_mlx_eligible` refuses the job (it never
///    reaches a worker) AND `video_model_mac_support` answers `supported: false` carrying
///    `classify_video_gap`'s "this video model has no MLX engine" — which is FALSE: sc-8443 wired
///    `mlx-gen-krea-realtime` in `video_jobs/krea_realtime.rs`. The Video Studio picker then hides
///    the model (`macAvailableModels`) and the Model Manager shows the untrue reason, so a shipped
///    20–40 GB catalog entry becomes unselectable and misdescribed at once.
/// 2. **No arm in `video_mode_is_mlx_eligible`** ⇒ the generic arm grants only
///    `text_to_video | image_to_video` and `video_to_video` falls to `_ => false` — an advertised
///    capability (manifest `capabilities` AND `ui.recommendedFor`) that the engine implements
///    (`conditioning: [Reference, VideoClip]`) and the worker maps (`krea_realtime_video_task`
///    → `"v2v"`) would be refused by the router alone.
///
/// Discriminating in both directions: it pins the modes krea does NOT serve too, so a future edit
/// that "fixes" v2v by widening the generic arm — which would hand `video_to_video` to every routed
/// model — fails on the models that must keep rejecting it.
#[test]
fn krea_realtime_is_mlx_routed_and_serves_exactly_its_advertised_modes() {
    assert!(
        VIDEO_MLX_ROUTED_MODELS.contains(&"krea_realtime_14b"),
        "krea_realtime_14b must be MLX-routed — sc-8443 wired the real engine, so a missing row \
         makes the app claim it has none"
    );

    // The three the catalog + descriptor + worker all agree on.
    for mode in ["text_to_video", "image_to_video", "video_to_video"] {
        assert!(
            video_mode_is_mlx_eligible("krea_realtime_14b", mode),
            "krea_realtime_14b advertises {mode} and must serve it"
        );
    }
    // ...and nothing else. The descriptor exposes no keyframe / clip-extend / bridge /
    // person-replace / character-animation surface.
    for mode in [
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
        "animate_character",
        "reference_to_video",
        "reference_video_to_video",
        "multi_video_to_video",
        "ads2v",
        "nonsense",
    ] {
        assert!(
            !video_mode_is_mlx_eligible("krea_realtime_14b", mode),
            "krea_realtime_14b does not implement {mode} and must not claim it"
        );
    }

    // `video_to_video` stays a per-model capability, NOT something the generic arm hands out: only
    // bernini (the planner editing modes) and krea serve it. Without this, "make krea's v2v work"
    // by relaxing the shared arm would pass every assertion above while silently enabling v2v on
    // LTX / Wan / SVD / Mochi, none of which implement it.
    for model in VIDEO_MLX_ROUTED_MODELS {
        let expected = matches!(*model, "bernini" | "krea_realtime_14b");
        assert_eq!(
            video_mode_is_mlx_eligible(model, "video_to_video"),
            expected,
            "video_to_video eligibility for {model}"
        );
    }

    // The UI gating oracle — the surface the user actually meets. `supported: false` here is what
    // hides the entry from the picker and prints the false "no MLX engine" reason.
    let support = model_mac_support("krea_realtime_14b", "video", None);
    assert!(
        support.supported,
        "krea_realtime_14b must be Mac-supported: {:?}",
        support.reason
    );
    assert!(support.reason.is_none());
    let modes = &support.features.video_modes;
    for mode in ["text_to_video", "image_to_video", "video_to_video"] {
        assert_eq!(
            modes.get(mode),
            Some(&true),
            "Video Studio must enable {mode} for krea_realtime_14b"
        );
    }
    for mode in ["replace_person", "animate_character", "extend_clip"] {
        assert_eq!(
            modes.get(mode),
            Some(&false),
            "Video Studio must disable {mode} for krea_realtime_14b"
        );
    }
}

/// A queued `video_generate` job for the MLX claim gate ([`video_job_is_mlx_eligible`], via
/// [`worker_supports_job`]) — the predicate that decides whether the in-process mlx worker picks
/// the job up at all. Asserting `video_mode_is_mlx_eligible` alone would skip the
/// `VIDEO_MLX_ROUTED_MODELS` membership half, which is exactly the half MiniMax-H3 was missing.
/// The in-process mlx worker on the VIDEO lane. `gpu_id: "mlx"` is what selects the mlx arm of
/// `worker_supports_job`, and `video_generate` is the advertised capability the final check
/// requires — the shared [`mlx_worker`] fixture advertises `image_generate` only.
fn mlx_video_worker() -> WorkerSnapshot {
    serde_json::from_value(json!({
        "id": "worker_mlx_video",
        "gpuId": "mlx",
        "status": "idle",
        "capabilities": ["gpu", "video_generate"],
        "loadedModels": [],
        "registeredAt": "2026-08-14T00:00:00Z",
        "lastSeenAt": "2026-08-14T00:00:00Z"
    }))
    .expect("valid MLX video worker")
}

fn video_generate_job(payload: Value) -> JobSnapshot {
    serde_json::from_value(json!({
        "id": "job_video",
        "type": "video_generate",
        "status": "queued",
        "payload": payload,
        "result": {},
        "requestedGpu": "auto",
        "progress": 0,
        "stage": "queued",
        "message": "",
        "attempts": 1,
        "cancelRequested": false,
        "createdAt": "2026-08-14T00:00:00Z",
        "updatedAt": "2026-08-14T00:00:00Z"
    }))
    .expect("valid video job")
}

/// sc-17159 (epic 17137) — MiniMax-H3 is MLX-ROUTED on both partitions, and each serves EXACTLY
/// the modes its own manifest entry advertises.
///
/// Three separate regressions, all of them silent, and the family had all three before this story:
///
/// 1. **Neither id was in [`VIDEO_MODEL_CAPS`]** ⇒ `video_job_is_mlx_eligible` refused every job
///    (queued forever, never claimed) AND `video_model_mac_support` answered `supported: false`
///    carrying `classify_video_gap`'s "this video model has no MLX engine". Every MiniMax-H3
///    download row is `platforms: ["macos"]`, so the Video Studio hid the family on the ONLY
///    platform it installs on.
/// 2. **No arm in `video_mode_is_mlx_eligible`** ⇒ `minimax_h3`'s `first_last_frame` (fl2va) fell
///    to the generic arm's LTX/Wan-only list and was refused — a mode both `capabilities` and
///    `ui.recommendedFor` advertise.
/// 3. **The generic arm was WRONG for the reference partition in the other direction** ⇒ it would
///    have granted `minimax_h3_ref` the `text_to_video | image_to_video` its `transformer_ref`
///    checkpoint cannot do while refusing `reference_to_video`, the only thing it can. The two
///    partitions are separate 18.78 GB DiTs, so that is a wrong-checkpoint load, not a spare mode.
///
/// Discriminating in both directions on purpose: it pins what each partition must NOT serve, so a
/// future "fix" that widens the generic arm fails on the partition that has to keep refusing.
#[test]
fn minimax_h3_partitions_are_mlx_routed_and_serve_exactly_their_declared_capabilities() {
    for id in ["minimax_h3", "minimax_h3_ref"] {
        assert!(
            VIDEO_MLX_ROUTED_MODELS.contains(&id),
            "{id} must be MLX-routed — macOS is its only platform, so an absent row makes the app \
             claim it has no engine and hides it from the picker"
        );
    }

    // t2va + fl2va on the `transformer` partition. `image_to_video` is fl2va with a FIRST frame
    // only; `first_last_frame` is fl2va with both.
    for mode in ["text_to_video", "image_to_video", "first_last_frame"] {
        assert!(
            video_mode_is_mlx_eligible("minimax_h3", mode),
            "minimax_h3 advertises {mode} and must serve it"
        );
    }
    // Ref2VA on the `transformer_ref` partition — and NOT t2v/i2v/flf, which would load the wrong
    // checkpoint for the request.
    assert!(video_mode_is_mlx_eligible(
        "minimax_h3_ref",
        "reference_to_video"
    ));
    for mode in ["text_to_video", "image_to_video", "first_last_frame"] {
        assert!(
            !video_mode_is_mlx_eligible("minimax_h3_ref", mode),
            "minimax_h3_ref is the reference checkpoint and must not claim {mode}"
        );
    }
    // …and `reference_to_video` must NOT leak onto the base partition, whose `limits` declare
    // `maxReferenceAssets: 0` precisely because its checkpoint has no reference path.
    assert!(!video_mode_is_mlx_eligible(
        "minimax_h3",
        "reference_to_video"
    ));
    // Neither partition serves anything else. `nonsense` is in the list so the arms are proven to
    // be allow-lists rather than "true for everything I did not think of".
    for id in ["minimax_h3", "minimax_h3_ref"] {
        for mode in [
            "extend_clip",
            "video_bridge",
            "replace_person",
            "animate_character",
            "video_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
            "nonsense",
        ] {
            assert!(
                !video_mode_is_mlx_eligible(id, mode),
                "{id} does not implement {mode} and must not claim it"
            );
        }
    }

    // The CLAIM gate, not just the mode predicate: a queued job in each declared mode must be
    // eligible for the in-process mlx worker. This is the half `VIDEO_MODEL_CAPS` membership
    // decides, and the half that was missing.
    for (model, mode, extra) in [
        ("minimax_h3", "text_to_video", json!({})),
        (
            "minimax_h3",
            "image_to_video",
            json!({ "sourceAssetId": "img-1" }),
        ),
        (
            "minimax_h3",
            "first_last_frame",
            json!({ "sourceAssetId": "img-1", "lastFrameAssetId": "img-2" }),
        ),
        (
            "minimax_h3_ref",
            "reference_to_video",
            json!({ "referenceAssetIds": ["img-1"], "referenceAudioAssetIds": ["aud-1"] }),
        ),
    ] {
        let mut payload = object(json!({ "model": model, "mode": mode }));
        payload.extend(object(extra));
        assert!(
            worker_supports_job(
                &mlx_video_worker(),
                &video_generate_job(Value::Object(payload))
            ),
            "the mlx worker must claim a {model} / {mode} job"
        );
    }
    // A Ref2VA job on the BASE partition is refused by the claim gate — the wrong-checkpoint case,
    // caught before any weights load.
    assert!(!worker_supports_job(
        &mlx_video_worker(),
        &video_generate_job(json!({
            "model": "minimax_h3", "mode": "reference_to_video", "referenceAssetIds": ["img-1"]
        }))
    ));

    // The UI gating oracle — the surface the user actually meets. `supported: false` is what hid
    // the family from the picker and printed the false "no MLX engine" reason.
    for (id, served) in [
        (
            "minimax_h3",
            ["text_to_video", "image_to_video", "first_last_frame"].as_slice(),
        ),
        ("minimax_h3_ref", ["reference_to_video"].as_slice()),
    ] {
        let support = model_mac_support(id, "video", None);
        assert!(
            support.supported,
            "{id} must be Mac-supported: {:?}",
            support.reason
        );
        assert!(support.reason.is_none());
        let modes = &support.features.video_modes;
        assert!(
            !modes.is_empty(),
            "{id}: an empty videoModes map would gate nothing"
        );
        for (mode, enabled) in modes {
            assert_eq!(
                *enabled,
                served.contains(&mode.as_str()),
                "{id}: Video Studio gating for {mode}"
            );
        }
    }
}

/// sc-17159 — `wan_2_2_vace_fun_14b` is MLX-routed and serves the ONE mode it advertises.
///
/// A live instance of the GH #2074 class, found by
/// `every_declared_video_capability_is_claimable_by_some_lane`: the dual-expert VACE-Fun control
/// checkpoint shipped with a manifest row (`capabilities: ["replace_person"]`, a macOS MLX
/// download) and a dedicated worker arm (`VideoRoute::ReplacePersonWanVaceFun` →
/// `generate_wan_vace_fun`, sc-3459) — but NO [`VIDEO_MODEL_CAPS`] row and no mention in
/// `video_mode_is_mlx_eligible`'s `replace_person` arm. It is in none of the `CANDLE_VIDEO_*` sets
/// either, so the MLX lane is its only lane, and its only capability was unreachable on it.
#[test]
fn wan_vace_fun_is_mlx_routed_and_serves_its_only_advertised_mode() {
    assert!(
        VIDEO_MLX_ROUTED_MODELS.contains(&"wan_2_2_vace_fun_14b"),
        "wan_2_2_vace_fun_14b must be MLX-routed — its dispatch arm has existed since sc-3459"
    );
    assert!(video_mode_is_mlx_eligible(
        "wan_2_2_vace_fun_14b",
        "replace_person"
    ));
    // Only that one: the control checkpoint has no generic T2V/I2V (the manifest says so too, and
    // `builtin_manifest_registers_the_wan_vace_fun_model` pins it).
    for mode in [
        "text_to_video",
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "animate_character",
        "nonsense",
    ] {
        assert!(
            !video_mode_is_mlx_eligible("wan_2_2_vace_fun_14b", mode),
            "wan_2_2_vace_fun_14b is a replace_person control checkpoint and must not claim {mode}"
        );
    }
    let support = model_mac_support("wan_2_2_vace_fun_14b", "video", None);
    assert!(
        support.supported,
        "wan_2_2_vace_fun_14b must be Mac-supported: {:?}",
        support.reason
    );
    assert_eq!(
        support.features.video_modes.get("replace_person"),
        Some(&true),
        "Video Studio must enable replace_person for wan_2_2_vace_fun_14b"
    );
}

/// sc-19558 (epic 17137) — a CANDLE-ROUTED video model must have weights a Windows or Linux user
/// can actually install.
///
/// This is the artifact half of `declaration ≠ enforcement ≠ reachability` (GH #2074), and it is the
/// half nothing checked. The routing table decides which lane serves a request; the manifest decides
/// what a user can obtain. Flip `candle_video_routed` for a model whose every download row is
/// `platforms: ["macos"]` and the app routes the job to a lane that cannot fetch a single byte — the
/// user sees a queued job and a download button that installs nothing.
///
/// REACH, stated so it is not taken on trust: this test CONSTRUCTS both sides it claims to compare.
/// `CANDLE_VIDEO_ROUTED_MODELS` is the real derived constant from `VIDEO_MODEL_CAPS`, not a retyped
/// list, and the manifest is the shipped `config/manifests/builtin.models.jsonc` parsed here — so a
/// column flip on one side or a `platforms` edit on the other both reach this assertion.
///
/// A CO-REQUISITE ROW IS NOT ENOUGH. Co-requisites install ALONGSIDE a primary and never AS one
/// (`is_co_requisite_download`), so a model whose only off-Mac rows are co-requisites still has no
/// base checkpoint. The count below is of PRIMARY rows for exactly that reason.
#[test]
fn candle_video_routed_models_have_an_installable_off_mac_download() {
    // Candle-routed ids with NO catalog entry at all. Mochi-1 is frozen and deliberately ships no
    // weights lane, so there is nothing to install on any platform and no row to check. Enumerated
    // rather than skipped silently, and the enumeration is asserted to be EXACT below, so a second
    // entry-less candle model cannot join it by accident.
    const NO_CATALOG_ENTRY: &[&str] = &["mochi_1"];

    // Primary (non-co-requisite) download rows that survive `retain_downloads_for_os` for `os`: a
    // row with no `platforms` is platform-agnostic and always applies.
    fn primary_rows_on(model: &Value, os: &str) -> usize {
        model["downloads"]
            .as_array()
            .map(|downloads| {
                downloads
                    .iter()
                    .filter(|download| download["coRequisite"].as_bool() != Some(true))
                    .filter(|download| match download["platforms"].as_array() {
                        Some(platforms) => platforms.iter().any(|value| value.as_str() == Some(os)),
                        None => true,
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    let manifest: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(include_str!(
        "../../../../../config/manifests/builtin.models.jsonc"
    )))
    .expect("builtin.models.jsonc parses");
    let models = manifest["models"]
        .as_array()
        .expect("builtin.models.jsonc has a models array");
    let entry = |id: &str| models.iter().find(|model| model["id"].as_str() == Some(id));

    let mut without_entry: Vec<&str> = Vec::new();
    for id in CANDLE_VIDEO_ROUTED_MODELS {
        let Some(model) = entry(id) else {
            without_entry.push(id);
            continue;
        };
        for os in ["windows", "linux"] {
            assert!(
                primary_rows_on(model, os) > 0,
                "{id} is candle-routed for video but has no primary download row installable on \
                 {os} — flipping a candle column without an off-Mac artifact routes the job to a \
                 lane that cannot obtain weights (sc-19558)"
            );
        }
    }
    assert_eq!(
        without_entry, NO_CATALOG_ENTRY,
        "the set of candle-routed video models with no catalog entry changed; a new one is not an \
         exemption from this guard, it is a model nothing can install"
    );

    // ── The MiniMax-H3 pair: the precondition sc-19558 satisfied, DEMONSTRATED rather than asserted
    // in prose. `minimax_h3`'s candle columns are still false (no measured ceiling, and
    // `crates/sceneworks-worker/src/video_jobs/minimax_h3.rs` is `#[cfg(target_os = "macos")]` end
    // to end, so there is no dispatch arm), but the WEIGHTS half no longer blocks the flip.
    let h3 = entry("minimax_h3").expect("minimax_h3 is in the builtin catalog");
    for os in ["windows", "linux"] {
        assert!(
            primary_rows_on(h3, os) > 0,
            "minimax_h3 must carry a primary download row installable on {os} — the raw upstream \
             MiniMaxAI/MiniMax-H3 snapshot set (sc-19558). Without it the candle columns cannot be \
             flipped, because `candle-gen-minimax-h3` reads that snapshot layout and nothing else \
             serves it off-Mac"
        );
    }

    // The reference partition is the deliberate opposite, and it must stay that way while candle
    // default-denies `ref2va` at its conditioning allowlist: advertising off-Mac weights for a mode
    // the only off-Mac engine refuses is the same defect pointing the other way.
    let h3_ref = entry("minimax_h3_ref").expect("minimax_h3_ref is in the builtin catalog");
    for os in ["windows", "linux"] {
        assert_eq!(
            primary_rows_on(h3_ref, os),
            0,
            "minimax_h3_ref must have NO {os} download row while `candle-gen-minimax-h3` refuses \
             ref2va (sc-17157 ports `transformer_ref`). If that port has landed, add the rows AND \
             the routing column in the same change, and update this assertion deliberately"
        );
    }
}
