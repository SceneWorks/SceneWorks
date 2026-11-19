//! SW-7100 (feature SW-F363): MiniMax-H3-Image at the image-job door — the run-time licence gate
//! (E23) and this model's own size envelope (E25).
use super::support::*;
use sceneworks_core::contracts::{JobType, JsonObject};

/// Run the WORKER's portable guard selection against what the API admitted: for every frozen
/// route, the tier the guard re-derives for `model_id`'s carrier (`portable_guard_runtime_variant`)
/// must rebuild an artifact closure whose primary row is the bound tier and source — otherwise the
/// worker refuses the binding before any loader runs. Returns the bound tier, which every route
/// must agree on.
pub(crate) fn assert_worker_guard_accepts(
    job_type: &JobType,
    payload: &JsonObject,
    binding: &Value,
    model_id: &str,
) -> String {
    use sceneworks_core::model_artifacts::portable_binding::{
        PortableJobBinding, PortableSourceIdentity,
    };
    use sceneworks_core::model_artifacts::ArtifactMemberRole;
    let binding: PortableJobBinding = serde_json::from_value(binding.clone()).unwrap();
    assert!(!binding.routes.is_empty(), "the job is bound");
    let mut tiers = std::collections::BTreeSet::new();
    for (key, route) in &binding.routes {
        let model = route
            .models
            .iter()
            .find(|model| model.model_id == model_id)
            .unwrap_or_else(|| panic!("{key} binds {model_id}"));
        let entry = Value::Object(payload.clone())
            .pointer(&format!("/{}", model.slot))
            .cloned()
            .unwrap_or_else(|| panic!("{key}: payload carries {}", model.slot));
        let variant = sceneworks_worker::portable_guard_runtime_variant(
            job_type,
            payload,
            &model.slot,
            &entry,
            Some(model),
        )
        .unwrap();
        let closure =
            sceneworks_core::model_artifacts::artifact_selection::selected_portable_model_artifact_closure(
                &entry,
                &route.backend,
                &route.platform,
                variant.as_deref(),
                payload,
            )
            .unwrap();
        let primary_row = closure["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row.get("coRequisite").and_then(Value::as_bool) != Some(true))
            .unwrap_or_else(|| panic!("{key}: the worker's closure has a primary"))
            .clone();
        assert_eq!(
            primary_row["variant"], model.variant,
            "{key}: the worker guard rebuilds the bound tier"
        );
        let primary = model
            .members
            .iter()
            .find(|member| member.role == ArtifactMemberRole::Primary)
            .unwrap();
        let PortableSourceIdentity::Revision { identity } = &primary.source else {
            panic!("{key}: a pinned primary");
        };
        assert_eq!(primary_row["repo"], identity.repository.as_str(), "{key}");
        assert_eq!(primary_row["revision"], identity.revision.as_str(), "{key}");
        tiers.insert(model.variant.clone());
    }
    assert_eq!(tiers.len(), 1, "every route binds one tier: {tiers:?}");
    tiers.into_iter().next().unwrap()
}

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
    // The 21:9 16 MP preset's 6272 px edge and the ~36 MP row's 9408 px edge are this model's own
    // (SW-7206); one past the declared 9408 edge is refused.
    for (width, height) in [(6272, 2688), (9408, 4032)] {
        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            job(width, height),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{width}x{height}: {created}");
    }
    let (status, over) = request(app.clone(), "POST", "/api/v1/image/jobs", job(9409, 2688)).await;
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
    let stored = state
        .jobs_store
        .get_job(job["id"].as_str().unwrap())
        .expect("job reads");
    assert_eq!(
        assert_worker_guard_accepts(
            &JobType::PromptRefine,
            &stored.payload,
            &binding,
            "minimax_h3_image_pe"
        ),
        "q4",
        "the worker's portable guard accepts the admitted q4 expander"
    );
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

/// PR #3418 review: a headless image request that names no tier, with only H3-Image's q4 tier
/// installed, freezes q4 as its explicit `advanced.mlxQuantize` (H3-Image declares no
/// `mlx.standardTierLayout`, so admission's implicit rule would bind bf16). The binding, the
/// worker's portable guard and the render loader then all select q4. The request that names a tier
/// keeps its own.
#[tokio::test]
async fn an_unselected_h3_image_request_freezes_the_installed_tier_end_to_end() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    let entry = shipped_entry();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [entry.clone(), Value::Object(
            sceneworks_core::builtin_manifests::builtin_model_entry("z_image_turbo")
                .expect("z_image_turbo is registered")
                .clone(),
        )] })
        .to_string(),
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
    // A q4-only install: the upstream snapshot's VAE/tokenizer/processor, the re-host's q4 DiT and
    // MiniMax-H3's q4 text encoder. No bf16 DiT.
    let snapshot = |repo: &str, revision: &str| {
        let root = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
            .unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs").join("main"), revision).unwrap();
        root.join("snapshots").join(revision)
    };
    let touch = |path: std::path::PathBuf| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    };
    let upstream = snapshot(
        "MiniMaxAI/MiniMax-H3-Image",
        "492fd8c4eca666f1085f57e30738c1e0d86b3439",
    );
    for file in [
        "vae_diffusers/config.json",
        "tokenizer/tokenizer.json",
        "processor/processor_config.json",
    ] {
        touch(upstream.join(file));
    }
    let rehost = snapshot(
        "SceneWorks/minimax-h3-image-mlx",
        "6f62669b919bccdbf51b08c696fbb84542000a87",
    );
    touch(rehost.join("q4/transformer_diffusers/config.json"));
    touch(
        snapshot(
            "SceneWorks/minimax-h3-mlx",
            "137ce668c55a20bc0935fd1cf2a3de8448abb7f4",
        )
        .join("q4/text_encoder/config.json"),
    );
    let (app, state) = crate::create_app_with_state(settings.clone()).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 q4").await;
    let job = |advanced: Value| {
        json!({
            "projectId": project["id"], "model": "minimax_h3_image", "prompt": "a lighthouse",
            "width": 2048, "height": 2048, "advanced": advanced,
        })
    };

    let (status, created) =
        request(app.clone(), "POST", "/api/v1/image/jobs", job(json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().unwrap();
    let stored = state.jobs_store.get_job(id).expect("job reads");
    assert_eq!(
        stored.payload["advanced"]["mlxQuantize"],
        json!(4),
        "{}",
        stored.payload["advanced"]
    );
    assert_eq!(stored.payload["advanced"]["mlxQuantizeFrozen"], json!(true));
    let binding = state
        .jobs_store
        .portable_model_binding(id)
        .expect("binding reads")
        .expect("the render is bound");
    assert_eq!(
        assert_worker_guard_accepts(
            &JobType::ImageGenerate,
            &stored.payload,
            &binding,
            "minimax_h3_image"
        ),
        "q4"
    );
    let dir =
        sceneworks_worker::minimax_h3_image_render_tier_dir(&settings.data_dir, &stored.payload)
            .expect("the loader resolves")
            .expect("an installed tier");
    assert_eq!(
        std::fs::canonicalize(dir).unwrap(),
        std::fs::canonicalize(rehost.join("q4")).unwrap()
    );

    // A request that names its tier keeps it (here an uninstalled q8, refused later by name).
    let (status, named) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job(json!({ "mlxQuantize": 8 })),
    )
    .await;
    if status == StatusCode::CREATED {
        let stored = state
            .jobs_store
            .get_job(named["id"].as_str().unwrap())
            .expect("job reads");
        assert_eq!(stored.payload["advanced"]["mlxQuantize"], json!(8));
        assert!(stored.payload["advanced"]
            .get("mlxQuantizeFrozen")
            .is_none());
    }
    // The marker is server-owned: a client that sends it does not get its value kept; the request
    // is judged as unselected and the installed tier is frozen.
    let (status, forged) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job(json!({ "mlxQuantize": 8, "mlxQuantizeFrozen": true })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{forged}");
    let stored = state
        .jobs_store
        .get_job(forged["id"].as_str().unwrap())
        .expect("job reads");
    assert_eq!(stored.payload["advanced"]["mlxQuantize"], json!(4));

    // A duplicate that switches to another model drops the frozen tier: it was never the user's.
    let (status, switched) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{id}/duplicate"),
        json!({ "payloadChanges": { "model": "z_image_turbo" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{switched}");
    let stored = state
        .jobs_store
        .get_job(switched["id"].as_str().unwrap())
        .expect("job reads");
    assert!(
        stored.payload["advanced"].get("mlxQuantize").is_none(),
        "{}",
        stored.payload["advanced"]
    );
    assert!(stored.payload["advanced"]
        .get("mlxQuantizeFrozen")
        .is_none());

    // After q4 is removed and q8 installed, a duplicate re-freezes to q8 instead of being refused
    // for an "explicit" q4 the user never chose.
    std::fs::remove_dir_all(rehost.join("q4")).unwrap();
    touch(rehost.join("q8/transformer_diffusers/config.json"));
    touch(
        snapshot(
            "SceneWorks/minimax-h3-mlx",
            "137ce668c55a20bc0935fd1cf2a3de8448abb7f4",
        )
        .join("q8/text_encoder/config.json"),
    );
    let (status, duplicate) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{id}/duplicate"),
        json!({ "payloadChanges": {} }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{duplicate}");
    let stored = state
        .jobs_store
        .get_job(duplicate["id"].as_str().unwrap())
        .expect("job reads");
    assert_eq!(stored.payload["advanced"]["mlxQuantize"], json!(8));
    assert_eq!(stored.payload["advanced"]["mlxQuantizeFrozen"], json!(true));

    // A partial bf16 install (its DiT config but not the fused heads its row declares) is not
    // installed to the loader either: the frozen tier and the loader's own unselected choice agree.
    touch(upstream.join("transformer_diffusers/config.json"));
    let installed = sceneworks_worker::minimax_h3_image_installed_render_tier(
        &settings.data_dir,
        &stored.payload["modelManifestEntry"],
    );
    assert_eq!(installed, Some("q8"));
    let mut unselected = stored.payload.clone();
    unselected.remove("advanced");
    let loader =
        sceneworks_worker::minimax_h3_image_render_tier_dir(&settings.data_dir, &unselected)
            .expect("the loader resolves")
            .expect("an installed tier");
    assert_eq!(
        std::fs::canonicalize(loader).unwrap(),
        std::fs::canonicalize(rehost.join("q8")).unwrap(),
        "the loader skips the partial bf16 exactly as the freeze does"
    );
}

/// PR #3418 review: a request admitted at q4 by name (`advanced.quantTier`), with the denser q8 also
/// installed, loads q4 — and so does a request whose only selection is the tier the worker's
/// portable guard verified and froze into the dispatch (`advanced.__portableBoundTier`). Before,
/// the render loader read only `mlxQuantize` and took the densest installed tier (q8).
#[tokio::test]
async fn a_named_or_bound_h3_image_tier_is_the_tier_the_loader_loads() {
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
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
        &entry,
        "test",
        "2026-11-10T00:00:00Z",
    )
    .unwrap();
    let snapshot = |repo: &str, revision: &str| {
        let root = sceneworks_core::hf_home::huggingface_repo_cache_path(&settings.data_dir, repo)
            .unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs").join("main"), revision).unwrap();
        root.join("snapshots").join(revision)
    };
    let touch = |path: std::path::PathBuf| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    };
    let upstream = snapshot(
        "MiniMaxAI/MiniMax-H3-Image",
        "492fd8c4eca666f1085f57e30738c1e0d86b3439",
    );
    for file in [
        "vae_diffusers/config.json",
        "tokenizer/tokenizer.json",
        "processor/processor_config.json",
    ] {
        touch(upstream.join(file));
    }
    let rehost = snapshot(
        "SceneWorks/minimax-h3-image-mlx",
        "6f62669b919bccdbf51b08c696fbb84542000a87",
    );
    let text_encoder = snapshot(
        "SceneWorks/minimax-h3-mlx",
        "137ce668c55a20bc0935fd1cf2a3de8448abb7f4",
    );
    for tier in ["q8", "q4"] {
        touch(rehost.join(format!("{tier}/transformer_diffusers/config.json")));
        touch(text_encoder.join(format!("{tier}/text_encoder/config.json")));
    }
    let (app, state) = crate::create_app_with_state(settings.clone()).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 named").await;
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project["id"], "model": "minimax_h3_image", "prompt": "a lighthouse",
            "width": 2048, "height": 2048, "advanced": { "quantTier": "q4" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let stored = state
        .jobs_store
        .get_job(created["id"].as_str().unwrap())
        .expect("job reads");
    // A named tier is the user's: nothing is frozen over it.
    assert!(
        stored.payload["advanced"].get("mlxQuantize").is_none(),
        "{}",
        stored.payload["advanced"]
    );
    let binding = state
        .jobs_store
        .portable_model_binding(created["id"].as_str().unwrap())
        .expect("binding reads")
        .expect("the render is bound");
    assert_eq!(
        assert_worker_guard_accepts(
            &JobType::ImageGenerate,
            &stored.payload,
            &binding,
            "minimax_h3_image"
        ),
        "q4"
    );
    let q4 = std::fs::canonicalize(rehost.join("q4")).unwrap();
    let loads = |payload: &JsonObject| {
        std::fs::canonicalize(
            sceneworks_worker::minimax_h3_image_render_tier_dir(&settings.data_dir, payload)
                .expect("the loader resolves")
                .expect("an installed tier"),
        )
        .unwrap()
    };
    assert_eq!(
        loads(&stored.payload),
        q4,
        "the named tier is loaded, not the denser q8"
    );

    // The guard writes the binding's verified tier into the dispatch as `__portableBoundTier`; a
    // request that carries only that selection loads it too.
    let bound_tier = binding["routes"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .and_then(|route| route["models"].as_array())
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["slot"] == "modelManifestEntry")
        })
        .and_then(|model| model["tier"].as_str())
        .expect("the bound tier")
        .to_owned();
    let mut dispatch = stored.payload.clone();
    dispatch.insert(
        "advanced".to_owned(),
        json!({ "__portableBoundTier": bound_tier }),
    );
    assert_eq!(loads(&dispatch), q4, "the verified bound tier is loaded");
    // Nothing named and nothing bound: the densest installed tier.
    dispatch.insert("advanced".to_owned(), json!({}));
    assert_eq!(
        loads(&dispatch),
        std::fs::canonicalize(rehost.join("q8")).unwrap()
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
    // gate, and the default tier installs: the published NVFP4 tier off-Mac, the published bf16
    // conversion on macOS (where the Blackwell-only NVFP4 row is narrowed away).
    let (status, job) = request(
        app.clone(),
        "POST",
        &download("minimax_h3_image_pe"),
        json!({ "requestedGpu": "auto", "licenseAcknowledged": true }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "model_download");
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

/// SW-7202 AC3 through the real download route on this host: installing q4 or q8 of the prompt
/// expander or of the Qwen3.8 27B describer queues exactly that tier's own folder row (this OS's
/// layout) plus the entry's co-requisites — the expander's upstream `prompt/` files and both
/// models' block drafts — and no job names another tier's folder. The all-OS twin of this plan is
/// `installing_a_qwen38_tier_plans_only_that_tier_and_its_co_requisites_on_every_os` in
/// sceneworks-core's `artifact_selection`.
#[tokio::test]
async fn installing_a_qwen38_tier_queues_only_that_tiers_folder_and_its_co_requisites() {
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
    let models = ["minimax_h3_image_pe", "vision_describe_qwen3_8_27b"];
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config/manifests");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": models.map(shipped) }).to_string(),
    )
    .unwrap();
    write_empty_sibling_manifests(&config);
    let app = create_app(test_settings(&temp)).unwrap();
    let mac = cfg!(target_os = "macos");

    let mut seen = std::collections::BTreeSet::new();
    for id in models {
        let mut host = shipped(id);
        crate::models::retain_downloads_for_os(&mut host, std::env::consts::OS);
        let co_requisites: Vec<Value> = host["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["coRequisite"] == true)
            .map(|row| json!([row["repo"], row["revision"], row["files"]]))
            .collect();
        for tier in ["q4", "q8"] {
            let other = if tier == "q4" { "q8" } else { "q4" };
            let folder = |tier: &str| match (id, mac) {
                ("minimax_h3_image_pe", false) => format!("candle-{tier}-full"),
                _ => format!("{tier}-full"),
            };
            let (status, job) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/models/{id}/download"),
                json!({ "requestedGpu": "auto", "variant": tier, "licenseAcknowledged": true }),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{id} {tier}: {job}");
            let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
            let queued: Vec<Value> = jobs
                .as_array()
                .unwrap()
                .iter()
                .filter(|job| seen.insert(job["id"].as_str().unwrap().to_owned()))
                .map(|job| job["payload"].clone())
                .collect();
            let fetches: Vec<Value> = queued
                .iter()
                .map(|payload| json!([payload["repo"], payload["revision"], payload["files"]]))
                .collect();
            let tier_jobs: Vec<&Value> = queued
                .iter()
                .filter(|payload| {
                    payload["files"]
                        .to_string()
                        .contains(&format!("\"{}/", folder(tier)))
                })
                .collect();
            assert_eq!(
                tier_jobs.len(),
                1,
                "{id} {tier}: one tier job: {fetches:#?}"
            );
            let row = crate::models::model_download_for_variant(&host, tier).unwrap();
            assert_eq!(
                fetches
                    .iter()
                    .filter(|fetch| **fetch == json!([row["repo"], row["revision"], row["files"]]))
                    .count(),
                1,
                "{id} {tier}: the tier job is the catalog's {tier} row: {fetches:#?}"
            );
            for co_requisite in &co_requisites {
                assert!(
                    fetches.contains(co_requisite),
                    "{id} {tier}: co-requisite {co_requisite} queued: {fetches:#?}"
                );
            }
            assert_eq!(
                fetches.len(),
                1 + co_requisites.len(),
                "{id} {tier}: the tier and its co-requisites, nothing else: {fetches:#?}"
            );
            assert!(
                !fetches
                    .iter()
                    .any(|fetch| fetch.to_string().contains(&format!("\"{}/", folder(other)))),
                "{id} {tier}: a job names {other}'s folder: {fetches:#?}"
            );
        }
    }
}

fn shipped_describer_entry() -> Value {
    Value::Object(
        sceneworks_core::builtin_manifests::builtin_model_entry("vision_describe_qwen3_8_27b")
            .expect("the describer is a builtin")
            .clone(),
    )
}

/// The three entries an expansion reads (the image model, the expander and the describer) as the
/// test install's catalog.
fn write_expansion_catalog(config: &std::path::Path) {
    single_model_manifest(config, "minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image");
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [
            shipped_entry(), shipped_pe_entry(), shipped_describer_entry()
        ] })
        .to_string(),
    )
    .unwrap();
}

/// A pinned snapshot directory of `repo` at `revision` in `data_dir`'s hub.
fn pinned_snapshot(data_dir: &std::path::Path, repo: &str, revision: &str) -> std::path::PathBuf {
    let root = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, repo).unwrap();
    std::fs::create_dir_all(root.join("refs")).unwrap();
    std::fs::write(root.join("refs").join("main"), revision).unwrap();
    root.join("snapshots").join(revision)
}

/// Every file the host's `tier` row of the describer names, at its pinned revision (a shard glob as
/// the one shard its index names).
fn install_describer_tier(data_dir: &std::path::Path, tier: &str) {
    let mut host = shipped_describer_entry();
    crate::models::retain_downloads_for_os(&mut host, std::env::consts::OS);
    let row = crate::models::model_download_for_variant(&host, tier).expect("describer tier row");
    let snapshot = pinned_snapshot(
        data_dir,
        row["repo"].as_str().unwrap(),
        row["revision"].as_str().unwrap(),
    );
    for file in row["files"].as_array().unwrap() {
        let file = file.as_str().unwrap();
        let (path, body) = if file.contains('*') {
            (file.replace('*', "00001-of-00001"), String::new())
        } else if file.ends_with("model.safetensors.index.json") {
            (
                file.to_owned(),
                json!({ "weight_map": { "a": "model-00001-of-00001.safetensors" } }).to_string(),
            )
        } else {
            (file.to_owned(), "{}".to_owned())
        };
        let path = snapshot.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }
}

/// MiniMax's five prompt files at the expander's pinned upstream revision.
fn install_prompt_files(data_dir: &std::path::Path) {
    let upstream = pinned_snapshot(
        data_dir,
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
}

/// The catalog row the models route serves for `id`.
async fn served_model(app: axum::Router, id: &str) -> Value {
    let (_, catalog) = request(app, "GET", "/api/v1/models", Value::Null).await;
    catalog
        .as_array()
        .or_else(|| catalog["models"].as_array())
        .expect("the models route lists models")
        .iter()
        .find(|model| model["id"] == id)
        .cloned()
        .unwrap_or_else(|| panic!("{id} is served"))
}

/// SW-7208 (AC1, AC3): the expander runs the model the request names, else the saved choice, else
/// MiniMax-H3-Image-PE. The abliterated choice stays behind the expander's MiniMax licence, and its
/// job carries the describer's entry with MiniMax's prompt files as a co-requisite.
#[tokio::test]
async fn an_expansion_runs_the_chosen_model_and_the_saved_choice_behind_the_expanders_licence() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    write_expansion_catalog(&temp.path().join("config/manifests"));
    let settings = test_settings(&temp);
    let ack_path =
        sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir);
    let app = create_app(settings).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Expand").await;
    let expand = |extra: Value| {
        let mut body = json!({
            "task": "minimax_h3_image_expand",
            "prompt": "a lighthouse in fog",
            "modelId": "minimax_h3_image",
            "projectId": project["id"],
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    let describer = json!({ "expanderModelId": "vision_describe_qwen3_8_27b" });

    // The abliterated model reads MiniMax's licensed prompts: refused until the licence is accepted.
    let (status, refused) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(describer.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{refused}");
    assert_eq!(refused["code"], "license_acknowledgment_required");
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &ack_path,
        &shipped_pe_entry(),
        "test",
        "2026-11-19T00:00:00Z",
    )
    .unwrap();

    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(describer.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["expanderModelId"],
        "vision_describe_qwen3_8_27b"
    );
    let entry = &job["payload"]["modelManifestEntry"];
    assert_eq!(entry["id"], "vision_describe_qwen3_8_27b", "{job}");
    let prompts: Vec<&Value> = entry["downloads"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["coRequisite"] == true && row["componentId"] == "prompt")
        .collect();
    assert_eq!(prompts.len(), 1, "{entry}");
    assert_eq!(prompts[0]["repo"], "MiniMaxAI/MiniMax-H3-Image-PE");
    assert_eq!(
        prompts[0]["revision"],
        "35603d4b63949afa12cb098227fffefedf081afa"
    );
    assert_eq!(prompts[0]["files"].as_array().unwrap().len(), 5);

    // No choice and nothing saved: MiniMax-H3-Image-PE.
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["expanderModelId"], "minimax_h3_image_pe");
    assert_eq!(
        job["payload"]["modelManifestEntry"]["id"],
        "minimax_h3_image_pe"
    );

    // A model that is not an expander is refused by name, the dataset captioner included.
    let (status, bad) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "expanderModelId": "vision_caption_qwen3vl_8b" })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");

    // The saved choice is what a request without one runs, and it survives a fresh read.
    let (status, saved) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "h3ImageExpanderModel": "vision_describe_qwen3_8_27b" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{saved}");
    let (_, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(prefs["h3ImageExpanderModel"], "vision_describe_qwen3_8_27b");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["expanderModelId"],
        "vision_describe_qwen3_8_27b"
    );
    // An unknown value never replaces the saved choice; an explicit request still wins over it.
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "h3ImageExpanderModel": "minimax_h3" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, prefs) = request(app.clone(), "GET", "/api/v1/ui-preferences", Value::Null).await;
    assert_eq!(prefs["h3ImageExpanderModel"], "vision_describe_qwen3_8_27b");
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "expanderModelId": "minimax_h3_image_pe" })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["modelManifestEntry"]["id"],
        "minimax_h3_image_pe"
    );
}

/// SW-7208 (AC3): the abliterated choice needs only an installed describer tier plus MiniMax's
/// prompt files — no MiniMax-H3-Image-PE weights. The prompt files download on their own behind the
/// expander's licence. With the tier alone no tier is frozen; with the prompt files too the job
/// freezes that tier, binds the describer's own pinned folder, and the worker's portable guard
/// accepts the binding.
#[tokio::test]
async fn an_abliterated_expansion_needs_only_a_describer_tier_and_the_prompt_files() {
    let _hf = isolate_hf_cache();
    std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
    let temp = tempfile::tempdir().unwrap();
    write_expansion_catalog(&temp.path().join("config/manifests"));
    let settings = test_settings(&temp);
    let (app, state) = crate::create_app_with_state(settings.clone()).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Expand").await;
    let body = json!({
        "task": "minimax_h3_image_expand",
        "prompt": "a lighthouse in fog",
        "modelId": "minimax_h3_image",
        "projectId": project["id"],
        "expanderModelId": "vision_describe_qwen3_8_27b",
    });
    let prompts_route = "/api/v1/models/minimax_h3_image_pe/conditional-components/\
                         describer_prompt_expansion/download";

    // The prompt files are MiniMax Materials: their own download waits on the expander's licence.
    let (status, refused) = request(
        app.clone(),
        "POST",
        prompts_route,
        json!({ "requestedGpu": "auto" }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{refused}");
    assert_eq!(refused["code"], "license_acknowledgment_required");
    let (status, components) = request(
        app.clone(),
        "POST",
        prompts_route,
        json!({ "requestedGpu": "auto", "licenseAcknowledged": true }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{components}");
    let queued = components["components"].as_array().unwrap();
    assert_eq!(queued.len(), 1, "only the prompt files: {components}");
    assert_eq!(queued[0]["repo"], "MiniMaxAI/MiniMax-H3-Image-PE");
    assert_eq!(queued[0]["status"], "queued");
    let pe = served_model(app.clone(), "minimax_h3_image_pe").await;
    assert_eq!(
        pe["conditionalPurposes"]["describer_prompt_expansion"]["installState"], "missing",
        "{pe}"
    );

    install_describer_tier(&settings.data_dir, "q4");
    let (status, job) = request(app.clone(), "POST", "/api/v1/prompts/refine", body.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert!(
        job["payload"].get("peTier").is_none(),
        "the prompt files are absent: {job}"
    );

    install_prompt_files(&settings.data_dir);
    // A finished download refreshes the catalog; these files were written directly.
    state.model_catalog_cache.invalidate();
    let pe = served_model(app.clone(), "minimax_h3_image_pe").await;
    assert_eq!(
        pe["conditionalPurposes"]["describer_prompt_expansion"]["installState"], "installed",
        "{pe}"
    );
    let (status, job) = request(app.clone(), "POST", "/api/v1/prompts/refine", body).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["peTier"], "q4", "{job}");
    let binding = state
        .jobs_store
        .portable_model_binding(job["id"].as_str().unwrap())
        .expect("binding reads")
        .expect("the expansion is bound");
    let stored = state
        .jobs_store
        .get_job(job["id"].as_str().unwrap())
        .expect("job reads");
    assert_eq!(
        assert_worker_guard_accepts(
            &JobType::PromptRefine,
            &stored.payload,
            &binding,
            "vision_describe_qwen3_8_27b"
        ),
        "q4"
    );
}

/// SW-7208 (E7): MiniMax-H3-Image-PE is never a dataset describer — its licence forbids using its
/// outputs to improve other models. The describe route refuses it under every name it has.
#[test]
fn the_dataset_describer_never_accepts_minimax_h3_image_pe() {
    for model in [
        "minimax_h3_image_pe",
        "MiniMaxAI/MiniMax-H3-Image-PE",
        "SceneWorks/minimax-h3-image-pe",
    ] {
        assert!(
            crate::training::resolve_describe_model(Some(model)).is_err(),
            "{model} must not describe a dataset"
        );
    }
    assert!(crate::training::resolve_describe_model(Some(
        sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_REPO
    ))
    .is_ok());
}

/// SW-7208 review: a retry or duplicate of an expansion re-checks the expander's MiniMax licence
/// against the catalog's current terms, whichever model runs it. The abliterated job carries the
/// describer's entry, which has no licence of its own, so before this its replay skipped the gate.
/// Both models' replays are refused once the acceptance is no longer on record.
#[tokio::test]
async fn retrying_an_expansion_rechecks_the_expanders_current_licence_for_either_model() {
    let _hf = isolate_hf_cache();
    for expander in ["vision_describe_qwen3_8_27b", "minimax_h3_image_pe"] {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config/manifests");
        write_expansion_catalog(&config);
        let settings = test_settings(&temp);
        sceneworks_core::license_acknowledgments::record_acknowledgment(
            &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
            &shipped_pe_entry(),
            "test",
            "2026-11-19T00:00:00Z",
        )
        .unwrap();
        let app = create_app(settings.clone()).unwrap();
        let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Retry").await;
        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/prompts/refine",
            json!({
                "task": "minimax_h3_image_expand",
                "prompt": "a lighthouse in fog",
                "modelId": "minimax_h3_image",
                "projectId": project["id"],
                "expanderModelId": expander,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{expander}: {job}");
        let id = job["id"].as_str().unwrap().to_owned();

        // The recorded acceptance is withdrawn after admission.
        std::fs::remove_file(
            sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
        )
        .unwrap();
        for action in ["retry", "duplicate"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{id}/{action}"),
                json!({ "payloadChanges": {} }),
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{expander} {action}: {body}");
            assert_eq!(body["code"], "license_acknowledgment_required", "{body}");
        }
    }
}

/// SW-7208 review: a request that names no model runs the saved choice only while it can run
/// here, else the first model that can (MiniMax-H3-Image-PE first), as the web UI does. A model
/// the request names is never substituted.
#[tokio::test]
async fn an_unnamed_expansion_falls_back_to_a_runnable_model() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    write_expansion_catalog(&temp.path().join("config/manifests"));
    let settings = test_settings(&temp);
    sceneworks_core::license_acknowledgments::record_acknowledgment(
        &sceneworks_core::license_acknowledgments::acknowledgments_file(&settings.config_dir),
        &shipped_pe_entry(),
        "test",
        "2026-11-19T00:00:00Z",
    )
    .unwrap();
    let app = create_app(settings.clone()).unwrap();
    let (_, project) = crate::tests::support::create_project(app.clone(), &"H3 Fallback").await;
    let expand = |extra: Value| {
        let mut body = json!({
            "task": "minimax_h3_image_expand",
            "prompt": "a lighthouse in fog",
            "modelId": "minimax_h3_image",
            "projectId": project["id"],
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    let (status, _) = request(
        app.clone(),
        "PUT",
        "/api/v1/ui-preferences",
        json!({ "h3ImageExpanderModel": "minimax_h3_image_pe" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Only the describer and the prompt files: the saved MiniMax-H3-Image-PE cannot run, so the
    // describer does, on its installed tier.
    install_describer_tier(&settings.data_dir, "q4");
    install_prompt_files(&settings.data_dir);
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["expanderModelId"],
        "vision_describe_qwen3_8_27b"
    );
    assert_eq!(job["payload"]["peTier"], "q4", "{job}");

    // A named model is never substituted, even when it cannot run.
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/prompts/refine",
        expand(json!({ "expanderModelId": "minimax_h3_image_pe" })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["expanderModelId"], "minimax_h3_image_pe");
    assert!(job["payload"].get("peTier").is_none(), "{job}");
}
