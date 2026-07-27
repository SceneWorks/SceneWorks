use super::{
    flux2_mlx_eligible, flux_mlx_eligible, image_job_is_mlx_eligible, image_request_mlx_eligible,
    instantid_mlx_eligible, model_mac_support, qwen_edit_mlx_eligible, qwen_mlx_eligible,
    sdxl_mlx_eligible, video_mode_is_mlx_eligible, worker_supports_job, z_image_mlx_eligible,
    JobSnapshot, WorkerSnapshot, MLX_ROUTED_MODELS, VIDEO_MLX_ROUTED_MODELS,
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

    // A base-tier-only shape (pose) and a manifest entry with no installed path stay ineligible.
    assert!(!image_job_is_mlx_eligible(&image_generate_job(json!({
        "model": "kreamania_variant4",
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
    for model in VIDEO_MLX_ROUTED_MODELS {
        assert_eq!(
            video_mode_is_mlx_eligible(model, "image_to_video"),
            *model != "bernini" && *model != "scail2_14b" && *model != "mochi_1",
            "image_to_video eligibility for {model}"
        );
        assert_eq!(
            video_mode_is_mlx_eligible(model, "text_to_video"),
            *model != "svd" && *model != "scail2_14b",
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
