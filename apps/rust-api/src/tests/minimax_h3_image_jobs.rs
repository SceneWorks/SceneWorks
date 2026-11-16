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

/// SW-7100 review: retry and duplicate reach the same run-licence gate as the typed image route.
/// A duplicate that switches another model's job to `minimax_h3_image` through `payloadChanges`,
/// and a retry or duplicate of an H3-Image job whose acceptance is no longer on record, are refused
/// with 403 before a job exists.
#[tokio::test]
async fn minimax_h3_image_retry_and_duplicate_need_the_run_licence() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_entry();
    let other = Value::Object(
        sceneworks_core::builtin_manifests::builtin_model_entry("z_image_turbo")
            .expect("z_image_turbo is registered")
            .clone(),
    );
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry.clone(), other] }).to_string(),
    )
    .unwrap();
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Replay").await;
    let job = |model: &str| {
        json!({
            "projectId": project["id"], "model": model, "prompt": "a lighthouse",
            "width": 1024, "height": 1024,
        })
    };
    let refused = |status: StatusCode, body: &Value| {
        assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
        assert_eq!(body["code"], "license_acknowledgment_required", "{body}");
    };

    // A duplicate that switches another model's job to H3-Image is refused like a fresh one.
    let (status, plain) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job("z_image_turbo"),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{plain}");
    let plain_id = plain["id"].as_str().unwrap();
    let (status, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{plain_id}/duplicate"),
        json!({ "payloadChanges": { "model": "minimax_h3_image" } }),
    )
    .await;
    refused(status, &body);

    // An admitted H3-Image job replays only while its acceptance is on record.
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let (status, h3) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job("minimax_h3_image"),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{h3}");
    let h3_id = h3["id"].as_str().unwrap();
    std::fs::remove_file(&ack_path).unwrap();
    for action in ["retry", "duplicate"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{h3_id}/{action}"),
            json!({ "payloadChanges": {} }),
        )
        .await;
        refused(status, &body);
    }
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

/// SW-7102 review (P1): an expansion that names no tier freezes the tier the worker will load.
/// With only the converted q4 expander installed, the job carries `peTier: "q4"` and its source
/// binding names that tier's files, not the catalog's default download (the unpublished bf16 row
/// on macOS, the NVFP4 original on Windows/Linux).
#[tokio::test]
async fn an_unselected_expansion_freezes_the_installed_expander_tier() {
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
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let (app, state) = crate::create_app_with_state(settings.clone()).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Expand").await;
    let body = json!({
        "task": "minimax_h3_image_expand",
        "prompt": "a lighthouse in fog",
        "modelId": "minimax_h3_image",
        "projectId": project["id"],
    });

    // Nothing installed: no tier is frozen, and the worker explains the missing install.
    let (status, job) = request(app.clone(), "POST", "/api/v1/prompts/refine", body.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(job["payload"].get("peTier").is_none(), "{job}");

    // Install this host's q4 layout and the upstream prompt files.
    let mut host = entry.clone();
    crate::models::retain_downloads_for_os(&mut host, std::env::consts::OS);
    let q4 = sceneworks_core::model_artifacts::artifact_selection::model_download_for_variant(
        &host, "q4",
    )
    .expect("q4 row");
    let q4_dir = q4["files"][0]
        .as_str()
        .unwrap()
        .trim_end_matches("/*")
        .to_owned();
    let snapshot = |repo: &str, revision: &str| {
        let root = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
            .unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs").join("main"), revision).unwrap();
        root.join("snapshots").join(revision)
    };
    let tier_root = snapshot(
        q4["repo"].as_str().unwrap(),
        q4["revision"].as_str().unwrap(),
    );
    for file in ["config.json", "model.safetensors"] {
        let path = tier_root.join(&q4_dir).join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    }
    let upstream = snapshot(
        "MiniMaxAI/MiniMax-H3-Image-PE",
        "35603d4b63949afa12cb098227fffefedf081afa",
    );
    for file in [
        "prompt/manifest.json",
        "prompt/t2i/system_prompt.txt",
        "prompt/t2i/task_prompt.txt",
        "prompt/t2i/request_params.json",
        "prompt/edit/task_prompt.txt",
    ] {
        let path = upstream.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    }

    let (status, job) = request(app.clone(), "POST", "/api/v1/prompts/refine", body).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["peTier"], "q4", "{job}");
    let binding = state
        .jobs_store
        .portable_model_binding(job["id"].as_str().unwrap())
        .expect("binding reads")
        .expect("the expansion is bound");
    let routes = binding["routes"].as_object().expect("bound routes");
    assert!(!routes.is_empty(), "{binding}");
    for (key, route) in routes {
        let expander = route["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["slot"] == "modelManifestEntry")
            .unwrap_or_else(|| panic!("{key} binds the expander: {binding}"));
        assert_eq!(expander["modelId"], "minimax_h3_image_pe", "{key}");
        assert_eq!(expander["variant"], "q4", "{key}: {expander}");
        assert_eq!(expander["tier"], "q4", "{key}: {expander}");
        let primary = expander["members"]
            .as_array()
            .unwrap()
            .iter()
            .find(|member| member["role"] == "primary")
            .unwrap_or_else(|| panic!("{key} binds a primary: {expander}"));
        assert_eq!(
            primary["source"]["identity"]["repository"], "SceneWorks/minimax-h3-image-pe",
            "{key}: {primary}"
        );
    }
    // This host's route names the q4 layout the worker on this OS loads.
    let local = &routes[&format!(
        "{}:{}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        std::env::consts::OS
    )];
    let files = local["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["slot"] == "modelManifestEntry")
        .and_then(|model| model["members"].as_array())
        .and_then(|members| members.iter().find(|member| member["role"] == "primary"))
        .and_then(|primary| primary["files"].as_array())
        .expect("primary files");
    assert!(
        !files.is_empty()
            && files.iter().all(|file| {
                file["relativePath"]
                    .as_str()
                    .is_some_and(|path| path.starts_with(&format!("{q4_dir}/")))
            }),
        "{local}"
    );
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

/// AT5 (E22, G7): installing MiniMax-H3-Image after MiniMax-H3 fetches no text-encoder bytes.
///
/// The text-encoder download is not skipped at plan time: every co-requisite row is queued, and the
/// worker answers each file already in the Hugging Face cache without a transfer (the cache-hit
/// tests in `crates/sceneworks-worker/src/downloads.rs`). So the property that makes H3's text
/// encoder satisfy H3-Image's is that both installs plan the SAME fetch: same repo, pinned
/// revision, file filter, tier and target directory, hence the same cache entries. This test
/// plans both shipped entries through the real download route at each tier and requires H3-Image's
/// text-encoder jobs to equal H3's, while its own DiT and VAE jobs name different repos.
#[tokio::test]
async fn minimax_h3_image_plans_the_same_text_encoder_fetch_as_h3_at_every_tier() {
    let _hf = isolate_hf_cache();
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    let shipped = |id: &str| {
        manifest["models"]
            .as_array()
            .and_then(|models| models.iter().find(|model| model["id"] == id))
            .cloned()
            .unwrap_or_else(|| panic!("{id} is in the builtin catalog"))
    };
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [shipped("minimax_h3"), shipped("minimax_h3_image")] })
            .to_string(),
    )
    .unwrap();
    write_empty_sibling_manifests(&config);
    let app = create_app(test_settings(&temp)).unwrap();

    for variant in ["q4", "q8", "bf16"] {
        let mut text_encoder_fetches = Vec::new();
        for model in ["minimax_h3", "minimax_h3_image"] {
            let (status, primary) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/models/{model}/download"),
                json!({ "requestedGpu": "auto", "variant": variant, "licenseAcknowledged": true }),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{model} {variant}: {primary}");
        }
        let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        let jobs = jobs.as_array().unwrap();
        for model in ["minimax_h3", "minimax_h3_image"] {
            let mut fetches = jobs
                .iter()
                .map(|job| &job["payload"])
                .filter(|payload| payload["modelId"] == model && payload["variant"] == variant)
                .filter(|payload| payload["files"].to_string().contains("text_encoder"))
                .map(|payload| {
                    json!([
                        payload["repo"],
                        payload["revision"],
                        payload["files"],
                        payload["targetDir"]
                    ])
                    .to_string()
                })
                .collect::<Vec<_>>();
            fetches.sort();
            assert!(
                !fetches.is_empty(),
                "{model} {variant}: no text-encoder job"
            );
            text_encoder_fetches.push(fetches);
        }
        assert_eq!(
            text_encoder_fetches[0], text_encoder_fetches[1],
            "{variant}: H3-Image must plan H3's exact text-encoder fetch"
        );
        let image_repos = jobs
            .iter()
            .map(|job| &job["payload"])
            .filter(|payload| payload["modelId"] == "minimax_h3_image")
            .filter_map(|payload| payload["repo"].as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            image_repos.contains("MiniMaxAI/MiniMax-H3-Image"),
            "{variant}: H3-Image's own weights stay a separate fetch: {image_repos:?}"
        );
    }
}
