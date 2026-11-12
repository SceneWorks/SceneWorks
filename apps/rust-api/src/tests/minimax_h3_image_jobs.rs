//! SW-7100 (feature SW-F363): MiniMax-H3-Image at the image-job door — the run-time licence gate
//! (E23) and this model's own size envelope (E25).
use super::support::*;

fn shipped_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["id"] == "minimax_h3_image")
        })
        .cloned()
        .expect("minimax_h3_image is in the builtin catalog")
}

#[tokio::test]
async fn minimax_h3_image_runs_only_after_its_license_is_accepted_and_admits_its_own_edge() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_entry();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry.clone()] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Image").await;
    let job = |width: u32, height: u32| {
        json!({
            "projectId": project["id"], "model": "minimax_h3_image", "prompt": "a lighthouse",
            "width": width, "height": height,
        })
    };

    let (status, refused) =
        request(app.clone(), "POST", "/api/v1/image/jobs", job(2048, 2048)).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{refused}");
    assert_eq!(
        refused["code"], "license_acknowledgment_required",
        "{refused}"
    );

    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    // The 21:9 preset's 6272 px edge is this model's own; one past it is refused.
    let (status, created) =
        request(app.clone(), "POST", "/api/v1/image/jobs", job(6272, 2688)).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let (status, over) = request(app.clone(), "POST", "/api/v1/image/jobs", job(6273, 2688)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{over}");
}

fn shipped_pe_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["id"] == "minimax_h3_image_pe")
        })
        .cloned()
        .expect("minimax_h3_image_pe is in the builtin catalog")
}

/// SW-7102 (E16/E17/E23): prompt expansion rides the prompt-refine seam, runs only after the
/// expander's licence is accepted, forwards its catalog entry and the ordered references, and
/// serves MiniMax-H3-Image only.
#[tokio::test]
async fn minimax_h3_image_expansion_is_licence_gated_and_carries_its_entry_and_references() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_pe_entry();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [shipped_entry(), entry.clone()] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Expand").await;
    let project_id = project["id"].as_str().unwrap().to_owned();
    let expand = |extra: Value| {
        let mut body = json!({
            "task": "minimax_h3_image_expand",
            "prompt": "a lighthouse in fog",
            "modelId": "minimax_h3_image",
            "projectId": project_id,
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };

    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{refused}");
    assert_eq!(
        refused["code"], "license_acknowledgment_required",
        "{refused}"
    );

    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "prompt_refine");
    assert_eq!(job["payload"]["task"], "minimax_h3_image_expand");
    assert_eq!(job["payload"]["modelId"], "minimax_h3_image");
    assert_eq!(
        job["payload"]["modelManifestEntry"]["id"],
        "minimax_h3_image_pe"
    );
    assert!(job["payload"].get("imagePaths").is_none(), "{job}");

    // An edit: references ride the plural key in the user's order, even one of them.
    let mut asset_ids = Vec::new();
    for index in 1..=17 {
        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            &format!("Reference{index}.png"),
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        asset_ids.push(asset["id"].as_str().unwrap().to_owned());
    }
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({
            "sourceAssetIds": [asset_ids[1], asset_ids[0]],
            "aspectRatio": "adaptive",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let paths = job["payload"]["imagePaths"].as_array().unwrap();
    assert_eq!(paths.len(), 2);
    assert_eq!(job["payload"]["aspectRatio"], "adaptive");
    let (status, single) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "sourceAssetIds": [asset_ids[0]] })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{single}");
    assert_eq!(single["payload"]["imagePaths"].as_array().unwrap().len(), 1);
    assert!(single["payload"].get("imagePath").is_none(), "{single}");

    // Seventeen references is one past the image model's edit cap: refused, not truncated.
    let (status, over) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "sourceAssetIds": asset_ids })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{over}");
    let (status, bad_ratio) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "aspectRatio": "wide" })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{bad_ratio}");
    let (status, other) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "modelId": "qwen_image_2_1" })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{other}");
}

/// SW-7106: the job door keeps MiniMax-H3-Image's sampler block on the payload the worker reads,
/// and refuses an invalid block by its wire field before a job exists.
#[tokio::test]
async fn minimax_h3_image_sampler_block_rides_the_job_and_invalid_values_are_refused() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_entry();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry.clone()] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Image").await;
    let job = |sampler: Value| {
        json!({
            "projectId": project["id"], "model": "minimax_h3_image", "prompt": "a lighthouse",
            "width": 2048, "height": 2048, "advanced": { "h3ImageSampler": sampler },
        })
    };

    let lcm = json!({ "sampler": "lcm", "sNoise": 1.1, "sNoiseEnd": 0.9, "noiseClipStd": 0 });
    let (status, created) =
        request(app.clone(), "POST", "/api/v1/image/jobs", job(lcm.clone())).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["payload"]["advanced"]["h3ImageSampler"], lcm);

    for (sampler, fragment) in [
        (
            json!({ "sampler": "lcm", "sNoise": 0 }),
            "sNoise 0 must be greater than 0",
        ),
        (
            json!({ "sampler": "lcm", "sNoiseEnd": -1 }),
            "sNoiseEnd -1 must be 0 or greater",
        ),
        (
            json!({ "sampler": "lcm", "noiseClipStd": -0.5 }),
            "noiseClipStd -0.5 must be 0 (off)",
        ),
        (
            json!({ "sampler": "heun" }),
            "not a MiniMax-H3-Image sampler",
        ),
    ] {
        let (status, refused) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            job(sampler.clone()),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{sampler}: {refused}");
        assert!(
            refused.to_string().contains(fragment),
            "{sampler}: {refused}"
        );
    }
}

/// SW-7106: a recipe preset stores the sampler block intact and refuses an invalid one; a headless
/// job naming the preset inherits it (an explicit block wins); duplicate and retry keep it.
#[tokio::test]
async fn minimax_h3_image_sampler_survives_presets_duplicate_and_retry() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_entry();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry.clone()] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Image").await;
    let lcm = json!({ "sampler": "lcm", "sNoise": 1.1, "sNoiseEnd": 0.9, "noiseClipStd": 0 });
    let preset_body = |name: &str, sampler: Value| {
        json!({
            "name": name, "model": "minimax_h3_image", "workflow": "text_to_image",
            "defaults": { "h3ImageSampler": sampler },
        })
    };

    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        preset_body("H3 bad", json!({ "sampler": "lcm", "sNoise": 0 })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert!(refused.to_string().contains("sNoise 0"), "{refused}");

    let (status, preset) = request(
        app.clone(),
        "POST",
        "/api/v1/recipe-presets",
        preset_body("H3 LCM", lcm.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{preset}");
    assert_eq!(preset["defaults"]["h3ImageSampler"], lcm);

    let job_body = |advanced: Value| {
        json!({
            "projectId": project["id"], "model": "minimax_h3_image", "prompt": "a lighthouse",
            "width": 2048, "height": 2048, "recipePresetId": preset["id"], "advanced": advanced,
        })
    };
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job_body(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["advanced"]["h3ImageSampler"], lcm);
    let (status, explicit) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job_body(json!({ "h3ImageSampler": { "sampler": "euler" } })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{explicit}");
    assert_eq!(
        explicit["payload"]["advanced"]["h3ImageSampler"],
        json!({ "sampler": "euler" })
    );

    for (action, body) in [
        ("duplicate", json!({ "payloadChanges": {} })),
        ("retry", json!({})),
    ] {
        let route = format!("/api/v1/jobs/{}/{action}", job["id"].as_str().unwrap());
        let (status, again) = request(app.clone(), "POST", &route, body).await;
        assert_eq!(status, StatusCode::CREATED, "{action}: {again}");
        assert_eq!(
            again["payload"]["advanced"]["h3ImageSampler"], lcm,
            "{action}"
        );
    }
}

/// E23 at install: downloading MiniMax-H3-Image and downloading its prompt expander each require
/// that model's OWN accepted licence. Accepting one does not open the other, and nothing is queued
/// by a refused request.
#[tokio::test]
async fn minimax_h3_image_and_its_expander_each_install_only_after_their_own_license_is_accepted() {
    let _hf = isolate_hf_cache();
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let (image, expander) = (shipped_entry(), shipped_pe_entry());
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [image.clone(), expander.clone()] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    let app = create_app(settings).unwrap();
    let download = |id: &str| format!("/api/v1/models/{id}/download");

    for id in ["minimax_h3_image", "minimax_h3_image_pe"] {
        for body in [
            json!({ "requestedGpu": "auto" }),
            json!({ "requestedGpu": "auto", "licenseAcknowledged": false }),
        ] {
            let (status, refused) = request(app.clone(), "POST", &download(id), body).await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{id}: {refused}");
            assert_eq!(
                refused["code"], "license_acknowledgment_required",
                "{id}: {refused}"
            );
        }
    }
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert!(jobs.as_array().unwrap().is_empty(), "{jobs}");

    // The image model's acceptance is its own: the expander stays refused until its licence is
    // accepted too.
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &image,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let (status, job) = request(
        app.clone(),
        "POST",
        &download("minimax_h3_image"),
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "model_download");
    let (status, refused) = request(
        app.clone(),
        "POST",
        &download("minimax_h3_image_pe"),
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{refused}");
    assert_eq!(refused["code"], "license_acknowledgment_required");

    // Accepting at the download door records the expander's own acceptance and passes the licence
    // gate. Off-Mac the published NVFP4 tier installs; on macOS every expander tier is a converted
    // SceneWorks tier that is not published yet (E28), so the next gate — not the licence — answers.
    let (status, job) = request(
        app.clone(),
        "POST",
        &download("minimax_h3_image_pe"),
        json!({ "requestedGpu": "auto", "licenseAcknowledged": true }),
    )
    .await;
    if cfg!(target_os = "macos") {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{job}");
        assert!(
            job["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("not published yet")),
            "{job}"
        );
    } else {
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["type"], "model_download");
    }
    assert!(
        sceneworks_core::license_acknowledgments::current_acknowledgment(&ack_path, &expander)
            .unwrap()
            .is_some(),
        "the download door records the expander's acceptance"
    );
}
