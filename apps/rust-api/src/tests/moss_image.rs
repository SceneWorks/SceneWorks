use super::support::*;
fn setup(temp: &tempfile::TempDir) {
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "moss_image_sft", "OpenMOSS-Team/MOSS-Image-SFT");
    let mut model = json!({"id":"moss_image_sft","name":"MOSS SFT","type":"image","family":"moss_image","capabilities":["text_to_image"],"downloads":[{"provider":"huggingface","repo":"OpenMOSS-Team/MOSS-Image-SFT","revision":"fc5de0d56fc910cbca6f6625a70f5f582b26deb1","files":["transformer/config.json"]}],"image":{"supportsSchedulerShift":true,"maxPromptBatchImages":8},"defaults":{"resolution":"1024x1024","count":1,"steps":50,"guidanceScale":3,"schedulerShift":5},"limits":{"minDimension":16,"maxDimension":4096,"requiresDimensionsMultipleOf":16},"conditionalComponents":[{"provider":"huggingface","repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"5ff314b4c1470876b114b810146085db8a181a6e","componentId":"moss-dmd-dataspace","files":["dataspace.safetensors"],"requiredFor":["moss_dmd_unmerged_bf16","moss_dmd_full_fp32"]}]});
    model["conditionalComponents"].as_array_mut().unwrap().push(json!({"provider":"huggingface","repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","componentId":"moss-image-edit","files":["adapter_config.json","adapter_model.safetensors"],"license":"unspecified","requiredFor":["moss_image_edit"]}));
    let mut base = model.clone();
    base["id"] = json!("moss_image_base");
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({"schemaVersion":1,"models":[model,base]}).to_string(),
    )
    .unwrap();
}

#[tokio::test]
async fn moss_edit_catalog_projection_tracks_actual_backend_even_with_a_shared_snapshot_cache() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let path = temp.path().join("config/manifests/builtin.models.jsonc");
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["models"].as_array_mut().unwrap().push(json!({"id":"krea_2_raw","name":"Krea","type":"image","family":"krea","capabilities":["text_to_image","edit_image"],"downloads":[],"paths":{}}));
    std::fs::write(path, manifest.to_string()).unwrap();
    let (_, mut state) = create_app_with_state(test_settings(&temp)).unwrap();
    let mut previous: Option<(&str, std::sync::Arc<Vec<Value>>)> = None;
    for candle_required in [false, true, false] {
        state.settings.candle_required = candle_required;
        let backend = crate::generation::enqueue_backend(&state);
        let models = crate::models::model_catalog_snapshot(&state).await.unwrap();
        let same_backend = crate::models::model_catalog_snapshot(&state).await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&models, &same_backend));
        if let Some((previous_backend, previous_models)) = previous.as_ref() {
            assert_eq!(
                std::sync::Arc::ptr_eq(previous_models, &models),
                *previous_backend == backend,
                "a backend transition must rebuild the shared catalog snapshot"
            );
        }
        previous = Some((backend, models.clone()));
        for model in models.iter().filter(|model| {
            sceneworks_core::moss_image::is_moss(model["id"].as_str().unwrap_or(""))
        }) {
            let caps = model["capabilities"].as_array().unwrap();
            assert!(caps.contains(&json!("text_to_image")));
            assert!(caps.contains(&json!("edit_image")));
            assert_eq!(model["defaults"]["steps"], 50);
            assert_eq!(model["defaults"]["guidanceScale"], 3);
        }
        let krea = models
            .iter()
            .find(|model| model["id"] == "krea_2_raw")
            .unwrap();
        assert_eq!(krea["capabilities"], json!(["text_to_image", "edit_image"]));
        assert_eq!(
            backend,
            if !cfg!(target_os = "macos") || candle_required {
                "candle"
            } else {
                "mlx"
            }
        );
    }
}
#[tokio::test]
async fn moss_component_download_repairs_wrong_revision_and_skips_exact_cached_revision() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let selected = json!({"repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"5ff314b4c1470876b114b810146085db8a181a6e","files":["dataspace.safetensors"]});
    let mut sibling = selected.clone();
    sibling["revision"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    super::yue2_catalog::seed_snapshot(&temp, &sibling);
    let app = create_app(test_settings(&temp)).unwrap();
    for mode in ["unmerged_bf16", "full_fp32"] {
        let route = format!(
            "/api/v1/models/moss_image_sft/conditional-components/moss_dmd_{mode}/download"
        );
        let (status, response) = request(
            app.clone(),
            "POST",
            &route,
            json!({"licenseAcknowledged":true}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{response}");
        let jobs = response["jobs"].as_array().unwrap();
        assert_eq!(jobs.len(), 1, "{response}");
        for key in ["repo", "revision", "files"] {
            assert_eq!(jobs[0]["payload"][key], selected[key], "{response}");
        }
        assert_eq!(response["components"][0]["status"], "queued");
    }
    super::yue2_catalog::seed_snapshot(&temp, &selected);
    for mode in ["unmerged_bf16", "full_fp32"] {
        let route = format!(
            "/api/v1/models/moss_image_sft/conditional-components/moss_dmd_{mode}/download"
        );
        let (status, response) = request(
            app.clone(),
            "POST",
            &route,
            json!({"licenseAcknowledged":true}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{response}");
        assert!(
            response["jobs"].as_array().unwrap().is_empty(),
            "{response}"
        );
        assert_eq!(response["components"][0]["status"], "installed");
    }
}
#[tokio::test]
async fn moss_typed_enqueue_defaults_off_preserves_prompt_and_scrubs_reports() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS"}),
    )
    .await;
    for advanced in [
        json!({"mossImageExecutionReport":{"forged":true}}),
        json!({"mossImage":{"acceleration":"off","uncond_skip_layers":[]}}),
    ] {
        let (status,job)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"日本語 <tag> {\"items\":[]}","width":16,"height":16,"advanced":advanced})).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["payload"]["prompt"], "日本語 <tag> {\"items\":[]}");
        assert_eq!(
            job["payload"]["advanced"]["mossImage"]["acceleration"],
            "off"
        );
        assert!(job["payload"]["advanced"]
            .get("mossImageExecutionReport")
            .is_none());
    }
}
#[tokio::test]
async fn moss_typed_and_generic_refuse_unsupported_controls_before_job_creation() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS"}),
    )
    .await;
    for advanced in [
        json!({"mossImage":{"acceleration":"full_fp32"},"guidanceScale":3}),
        json!({"mossImage":{"acceleration":"unmerged_bf16"},"steps":9}),
        json!({"mossImage":{"unknown":1}}),
        json!({"controlImage":"asset"}),
    ] {
        let (status,body)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"subject","advanced":advanced})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    for model in ["moss_image_sft", "flux_schnell"] {
        for kind in ["image_generate", "image_edit"] {
            let (status,body)=request(app.clone(),"POST","/api/v1/jobs",json!({"type":kind,"requestedGpu":"auto","projectId":project["id"],"payload":{"model":model,"prompt":"subject"}})).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert!(
                body.to_string().contains("must be created via POST"),
                "{body}"
            );
        }
    }
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(jobs.as_array().unwrap().len(), 0, "{jobs}");
    let (status,body)=request(app.clone(),"POST","/api/v1/jobs",json!({"type":"image_detail","requestedGpu":"auto","projectId":project["id"],"payload":{"model":"moss_image_sft","prompt":"subject"}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("text-to-image"), "{body}");
}
#[tokio::test]
async fn moss_both_optional_precision_modes_fail_before_enqueue_with_install_action() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS"}),
    )
    .await;
    for mode in ["unmerged_bf16", "full_fp32"] {
        let (status,body)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"subject","advanced":{"mossImage":{"acceleration":mode}}})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body.to_string().contains("conditional-components"),
            "{body}"
        );
    }
}
#[tokio::test]
async fn moss_duplicate_revalidates_merged_controls_and_resets_inherited_mode() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS"}),
    )
    .await;
    let (status,job)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"subject","advanced":{"mossImage":{"acceleration":"off"}}})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let route = format!("/api/v1/jobs/{}/duplicate", job["id"].as_str().unwrap());
    let (status,body)=request(app.clone(),"POST",&route,json!({"payloadChanges":{"advanced":{"mossImage":{"acceleration":"full_fp32"},"schedulerShift":5}}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = request(
        app.clone(),
        "POST",
        &route,
        json!({"payloadChanges":{"model":"moss_image_base"}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(
        body["payload"]["advanced"]["mossImage"]["acceleration"],
        "off"
    );
}

#[tokio::test]
async fn moss_bare_dmd_real_enqueue_and_replay_keep_omitted_generic_controls_omitted() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    // Tiny CPU fixture proves catalog availability only; the native worker separately authenticates
    // real frozen bytes and cannot execute this fixture as model weights.
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"5ff314b4c1470876b114b810146085db8a181a6e","files":["dataspace.safetensors"]}),
    );
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS"}),
    )
    .await;
    for mode in ["unmerged_bf16", "full_fp32"] {
        let (status,job)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"subject","advanced":{"mossImage":{"acceleration":mode}}})).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        let payload = &job["payload"];
        assert_eq!(payload["advanced"]["mossImage"]["acceleration"], mode);
        for key in [
            "steps",
            "guidanceScale",
            "schedulerShift",
            "sampler",
            "scheduler",
        ] {
            assert!(payload["advanced"].get(key).is_none(), "{key}: {payload}");
        }
        let route = format!("/api/v1/jobs/{}/duplicate", job["id"].as_str().unwrap());
        let (status, replay) =
            request(app.clone(), "POST", &route, json!({"payloadChanges":{}})).await;
        assert_eq!(status, StatusCode::CREATED, "{replay}");
        assert_eq!(
            replay["payload"]["advanced"]["mossImage"]["acceleration"],
            mode
        );
        let (status, ordinary) = request(
            app.clone(),
            "POST",
            &route,
            json!({"payloadChanges":{"advanced":{"mossImage":{"acceleration":"off"}}}}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{ordinary}");
        assert_eq!(
            ordinary["payload"]["advanced"]["mossImage"]["acceleration"],
            "off"
        );
    }
}

#[tokio::test]
async fn moss_post_preset_expansion_and_retry_use_the_same_canonical_validator() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"5ff314b4c1470876b114b810146085db8a181a6e","files":["dataspace.safetensors"]}),
    );
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS presets"}),
    )
    .await;
    for mode in ["unmerged_bf16", "full_fp32"] {
        let (status,preset)=request(app.clone(),"POST","/api/v1/recipe-presets",json!({"name":format!("Moss {mode}"),"model":"moss_image_sft","workflow":"text_to_image","defaults":{"mossImage":{"acceleration":mode}}})).await;
        assert_eq!(status, StatusCode::CREATED, "{preset}");
        let body = json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"日本語 <tag> {\"items\":[]}","recipePresetId":preset["id"]});
        let (status, job) = request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(
            job["payload"]["advanced"]["mossImage"]["acceleration"],
            mode
        );
        assert!(job["payload"]["advanced"].get("guidanceScale").is_none());
        let (status, retry) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{}/retry", job["id"].as_str().unwrap()),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{retry}");
        assert_eq!(
            retry["payload"]["advanced"]["mossImage"]["acceleration"],
            mode
        );
        let mut ordinary = body.clone();
        ordinary["advanced"] = json!({"mossImage":{"acceleration":"off","uncond_skip_layers":[],"slg":{"layers":[],"scale":0.5,"start":0,"stop":1}},"guidanceScale":0});
        ordinary["negativePrompt"] = json!("unconditional-only negative 日本語");
        let (status, off) = request(app.clone(), "POST", "/api/v1/image/jobs", ordinary).await;
        assert_eq!(status, StatusCode::CREATED, "{off}");
        assert_eq!(
            off["payload"]["advanced"]["mossImage"]["acceleration"],
            "off"
        );
        assert_eq!(
            off["payload"]["advanced"]["mossImage"]["uncond_skip_layers"],
            json!([])
        );
        assert_eq!(
            off["payload"]["advanced"]["mossImage"]["slg"]["layers"],
            json!([])
        );
        assert_eq!(
            off["payload"]["negativePrompt"],
            "unconditional-only negative 日本語"
        );
        for advanced in [
            json!({"guidanceScale":3}),
            json!({"schedulerShift":5}),
            json!({"mossImage":{"acceleration":mode,"slg":{"layers":[22],"scale":1,"start":0,"stop":1}}}),
        ] {
            let mut conflict = body.clone();
            conflict["advanced"] = advanced;
            let (status, error) =
                request(app.clone(), "POST", "/api/v1/image/jobs", conflict).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        }
        let mut negative = body;
        negative["negativePrompt"] = json!("explicit conflict");
        let (status, error) = request(app.clone(), "POST", "/api/v1/image/jobs", negative).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    }
}

#[tokio::test]
async fn moss_batch_prompts_are_verbatim_and_wrong_optional_revision_cannot_satisfy_availability() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":["dataspace.safetensors"]}),
    );
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS exact source"}),
    )
    .await;
    let prompts = json!([
        "日本語 <tag>",
        "{\"summary\":{\"caption\":\"scene\"},\"items\":[]}"
    ]);
    let (status,job)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":prompts[0],"prompts":prompts,"count":2,"width":16,"height":16,"advanced":{"mossImage":{"uncond_skip_layers":null}}})).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["prompts"], prompts);
    assert_eq!(job["payload"]["count"], 2);
    for mode in ["unmerged_bf16", "full_fp32"] {
        let (status,error)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","prompt":"subject","advanced":{"mossImage":{"acceleration":mode}}})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(
            error.to_string().contains("conditional-components"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn moss_edit_candle_enqueue_requires_exact_package_and_is_actually_claimable() {
    let _hf = isolate_hf_cache();
    for model in ["moss_image_base", "moss_image_sft"] {
        let temp = tempfile::tempdir().unwrap();
        setup(&temp);
        let mut settings = test_settings(&temp);
        settings.candle_required = true;
        let (app, state) = create_app_with_state(settings).unwrap();
        assert_eq!(crate::generation::enqueue_backend(&state), "candle");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({"name":"Candle edit"}),
        )
        .await;
        let route = format!(
            "/api/v1/projects/{}/assets",
            project["id"].as_str().unwrap()
        );
        let (_, asset) =
            request_multipart_upload(app.clone(), &route, "ref.png", "image/png", b"png-bytes")
                .await;
        let body = json!({"projectId":project["id"],"model":model,"mode":"edit_image","prompt":"blue cup","referenceAssetIds":[asset["id"]]});
        let mut selected = json!({"repo":"chinoll/Moss-Image-Edit","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":["adapter_config.json","adapter_model.safetensors"]});
        for seed_wrong_revision in [false, true] {
            if seed_wrong_revision {
                super::yue2_catalog::seed_snapshot(&temp, &selected);
            }
            let (status, error) =
                request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
            assert!(error.to_string().contains("moss_image_edit"), "{error}");
        }
        selected["revision"] = json!("e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11");
        super::yue2_catalog::seed_snapshot(&temp, &selected);
        let (status, job) = request(app.clone(), "POST", "/api/v1/image/jobs", body).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        let downloads = job["payload"]["modelManifestEntry"]["downloads"]
            .as_array()
            .unwrap();
        assert!(downloads
            .iter()
            .any(|row| row["repo"] == selected["repo"] && row["revision"] == selected["revision"]));
        assert!(!downloads
            .iter()
            .any(|row| row["repo"] == "lodestones/MOSS-Image-SFT-8-steps-dmd"));
        state
            .jobs_store
            .register_worker(sceneworks_core::jobs_store::RegisterWorker {
                worker_id: "candle-edit-fixture".into(),
                gpu_id: "0".into(),
                gpu_name: None,
                capabilities: vec![
                    WorkerCapability::Gpu,
                    WorkerCapability::ImageGenerate,
                    WorkerCapability::ImageEdit,
                    WorkerCapability::Unknown("candle".into()),
                ],
                loaded_models: vec![],
                utilization: None,
            })
            .unwrap();
        let claimed = state
            .jobs_store
            .claim_next_job("candle-edit-fixture")
            .unwrap()
            .expect("Candle worker claims typed edit");
        assert_eq!(claimed.id, job["id"].as_str().unwrap());
        assert_eq!(claimed.payload["referenceAssetIds"], json!([asset["id"]]));
        assert_eq!(
            claimed.job_type,
            sceneworks_core::contracts::JobType::ImageEdit
        );
        let (status, ordinary) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({"projectId":project["id"],"model":model,"prompt":"ordinary cup"}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{ordinary}");
        assert!(ordinary["payload"]["modelManifestEntry"]["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["repo"] != "chinoll/Moss-Image-Edit"
                && row["repo"] != "lodestones/MOSS-Image-SFT-8-steps-dmd"));
    }
}

#[tokio::test]
async fn moss_edit_typed_api_defaults_receipt_scrubbing_reference_ownership_and_replay() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let selected = json!({"repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","files":["adapter_config.json","adapter_model.safetensors"]});
    super::yue2_catalog::seed_snapshot(&temp, &selected);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"Edit"}),
    )
    .await;
    let route = format!(
        "/api/v1/projects/{}/assets",
        project["id"].as_str().unwrap()
    );
    let (status, asset) =
        request_multipart_upload(app.clone(), &route, "ref.png", "image/png", b"png-bytes").await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    for model in ["moss_image_base", "moss_image_sft"] {
        let body = json!({"projectId":project["id"],"model":model,"mode":"edit_image","prompt":"make it blue","negativePrompt":"blur","referenceAssetIds":[asset["id"]],"advanced":{"mossImageEdit":{"reference_max_edge":256,"use_kv_cache":false},"mossImageEditExecutionReport":{"forged":true}}});
        let (status, job) = request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
        assert_eq!(status, StatusCode::CREATED, "{job}");
        assert_eq!(job["type"], "image_edit");
        assert_eq!(job["payload"]["referenceAssetIds"], json!([asset["id"]]));
        assert_eq!(job["payload"]["width"], 512);
        assert_eq!(job["payload"]["height"], 512);
        assert_eq!(
            job["payload"]["advanced"]["mossImage"]["uncond_skip_layers"],
            json!([])
        );
        assert_eq!(
            job["payload"]["advanced"]["mossImageEdit"]["use_kv_cache"],
            false
        );
        assert!(job["payload"]["advanced"]
            .get("mossImageEditExecutionReport")
            .is_none());
        assert!(job["payload"]["modelManifestEntry"]["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["componentId"] == "moss-image-edit"));
        for action in ["duplicate", "retry"] {
            let route = format!("/api/v1/jobs/{}/{action}", job["id"].as_str().unwrap());
            let (status, replay) =
                request(app.clone(), "POST", &route, json!({"payloadChanges":{}})).await;
            assert_eq!(status, StatusCode::CREATED, "{replay}");
            assert_eq!(replay["payload"]["referenceAssetIds"], json!([asset["id"]]));
            let (status,error)=request(app.clone(),"POST",&route,json!({"payloadChanges":{"advanced":{"mossImage":{"acceleration":"unmerged_bf16"}}}})).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        }
        let mut foreign = body.clone();
        foreign["referenceAssetIds"] = json!(["foreign-reference"]);
        let (status, error) = request(app.clone(), "POST", "/api/v1/image/jobs", foreign).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    }
}

async fn assert_moss_edit_replay_transitions(action: &str) {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let manifest_path = temp.path().join("config/manifests/builtin.models.jsonc");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["models"].as_array_mut().unwrap().push(json!({
        "id":"flux2_dev", "name":"FLUX.2 Dev", "type":"image", "family":"flux2",
        "adapter":"flux2_diffusers", "capabilities":["text_to_image","edit_image"],
        "downloads":[], "paths":{}, "defaults":{"resolution":"512x512","count":1},
        "limits":{"minDimension":16,"maxDimension":4096,"requiresDimensionsMultipleOf":16}, "ui":{}
    }));
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","files":["adapter_config.json","adapter_model.safetensors"]}),
    );
    let app = create_app(test_settings(&temp)).unwrap();
    let (status, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS replay transitions"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{project}");
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(16, 16)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!(
            "/api/v1/projects/{}/assets",
            project["id"].as_str().unwrap()
        ),
        "reference.png",
        "image/png",
        png.get_ref(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    let body = json!({"projectId":project["id"],"model":"flux2_dev","mode":"edit_image","prompt":"make it blue","width":512,"height":512,"referenceAssetIds":[asset["id"]]});
    // A fresh equivalent target edit must succeed: an unrelated target-model validation error
    // cannot masquerade as a regression test for inherited MOSS options.
    let (status, fresh_target) =
        request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{fresh_target}");
    for model in ["moss_image_base", "moss_image_sft"] {
        for authored_options in [
            None,
            Some(json!({"reference_max_edge":256,"use_kv_cache":false})),
        ] {
            let mut moss_body = body.clone();
            moss_body["model"] = json!(model);
            let expected_options = authored_options
                .clone()
                .unwrap_or_else(|| json!({"reference_max_edge":512,"use_kv_cache":true}));
            if let Some(options) = authored_options {
                moss_body["advanced"] = json!({"mossImageEdit":options});
            }
            let (status, original) =
                request(app.clone(), "POST", "/api/v1/image/jobs", moss_body).await;
            assert_eq!(status, StatusCode::CREATED, "{original}");
            let options = &original["payload"]["advanced"]["mossImageEdit"];
            assert_eq!(
                *options, expected_options,
                "server must preserve authored options or insert defaults"
            );
            let route = format!("/api/v1/jobs/{}/{action}", original["id"].as_str().unwrap());
            let (status, switched) = request(
                app.clone(),
                "POST",
                &route,
                json!({"payloadChanges":{"model":"flux2_dev","mossImageEditExecutionReport":{"forged":true},"moss_image_edit_execution_report":{"forged":true}}}),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{action} to a valid target: {switched}"
            );
            assert_eq!(switched["payload"]["model"], "flux2_dev");
            assert_eq!(switched["payload"]["modelManifestEntry"]["id"], "flux2_dev");
            assert_eq!(
                switched["payload"]["referenceAssetIds"],
                json!([asset["id"]])
            );
            for key in ["mossImage", "mossImageEdit"] {
                assert!(
                    switched["payload"]["advanced"].get(key).is_none(),
                    "{switched}"
                );
            }
            for key in [
                "mossImageEditExecutionReport",
                "moss_image_edit_execution_report",
            ] {
                assert!(switched["payload"].get(key).is_none(), "{switched}");
            }
            for target in [
                model,
                if model == "moss_image_base" {
                    "moss_image_sft"
                } else {
                    "moss_image_base"
                },
            ] {
                let (status, preserved) = request(
                    app.clone(),
                    "POST",
                    &route,
                    json!({"payloadChanges":{"model":target}}),
                )
                .await;
                assert_eq!(status, StatusCode::CREATED, "{preserved}");
                assert_eq!(preserved["payload"]["advanced"]["mossImageEdit"], *options);
                assert_eq!(
                    preserved["payload"]["referenceAssetIds"],
                    json!([asset["id"]])
                );
            }
            let clear_references = json!({"mode":"text_to_image","sourceAssetId":null,"referenceAssetId":null,"referenceAssetIds":[],"maskAssetId":null});
            let (status, ordinary) = request(
                app.clone(),
                "POST",
                &route,
                json!({"payloadChanges":clear_references}),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::CREATED,
                "{action} to ordinary MOSS: {ordinary}"
            );
            assert_eq!(ordinary["payload"]["model"], model);
            assert_eq!(ordinary["payload"]["mode"], "text_to_image");
            assert_eq!(ordinary["payload"]["referenceAssetIds"], json!([]));
            assert!(ordinary["payload"]["advanced"]
                .get("mossImageEdit")
                .is_none());
            let (status, returned) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{}/{action}", ordinary["id"].as_str().unwrap()),
                json!({"payloadChanges":{"mode":"edit_image","referenceAssetIds":[asset["id"]]}}),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{returned}");
            assert_eq!(
                returned["payload"]["advanced"]["mossImageEdit"],
                json!({"reference_max_edge":512,"use_kv_cache":true})
            );
            for (changes, expected_detail) in [
                (
                    json!({"model":"flux2_dev","advanced":{"mossImageEdit":options}}),
                    "only supported by MOSS-Image",
                ),
                (
                    json!({"mode":"text_to_image","sourceAssetId":null,"referenceAssetId":null,"referenceAssetIds":[],"maskAssetId":null,"advanced":{"mossImageEdit":options}}),
                    "mossImageEdit requires edit_image",
                ),
            ] {
                let (status, rejected) = request(
                    app.clone(),
                    "POST",
                    &route,
                    json!({"payloadChanges":changes}),
                )
                .await;
                assert_eq!(
                    status,
                    StatusCode::BAD_REQUEST,
                    "explicit options must remain validated: {rejected}"
                );
                assert!(
                    rejected["detail"]
                        .as_str()
                        .unwrap()
                        .contains(expected_detail),
                    "{rejected}"
                );
            }
            let (status, explicit_target) = request(
                app.clone(),
                "POST",
                &route,
                json!({"payloadChanges":{"model":"flux2_dev","advanced":{}}}),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED, "{explicit_target}");
            assert!(explicit_target["payload"]["advanced"]
                .get("mossImageEdit")
                .is_none());
        }
    }
}

#[tokio::test]
async fn moss_edit_retry_clears_only_inherited_options_when_leaving_moss_edit() {
    assert_moss_edit_replay_transitions("retry").await;
}

#[tokio::test]
async fn moss_edit_duplicate_clears_only_inherited_options_when_leaving_moss_edit() {
    assert_moss_edit_replay_transitions("duplicate").await;
}

async fn moss_edit_mcp_round_trip(reference_count: usize) {
    use rmcp::model::CallToolRequestParams;
    use rmcp::transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
    };
    use rmcp::ServiceExt;
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","files":["adapter_config.json","adapter_model.safetensors"]}),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut settings = test_settings(&temp);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{addr}");
    let (app, state) = create_app_with_state(settings).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS MCP"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let mut ids = Vec::new();
    for (index, (width, height)) in [(16, 32), (48, 16), (32, 32)].into_iter().enumerate() {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(width, height)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            &format!("ref-{index}.png"),
            "image/png",
            bytes.get_ref(),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{asset}");
        ids.push(asset["id"].as_str().unwrap().to_owned());
    }
    let references = if reference_count == 1 {
        vec![ids[0].clone()]
    } else {
        vec![
            ids[0].clone(),
            ids[1].clone(),
            ids[0].clone(),
            ids[2].clone(),
        ]
    };
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let client = rmcp::model::ClientInfo::default()
        .serve(StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp")),
        ))
        .await
        .unwrap();
    for model in ["moss_image_base", "moss_image_sft"] {
        for custom in [false, true] {
            let mut args = json!({"projectId":project_id,"mode":"edit_image","model":model,"prompt":"make image 1 blue 日本語","negativePrompt":"blur","referenceAssetIds":references,"seed":0,"width":512,"height":768,"steps":41,"guidanceScale":4.5,"schedulerShift":6,"mossImageEdit":{"reference_max_edge":1024,"use_kv_cache":false},"mossImageEditExecutionReport":{"forged":true}});
            if !custom {
                for key in [
                    "width",
                    "height",
                    "steps",
                    "guidanceScale",
                    "schedulerShift",
                    "mossImageEdit",
                ] {
                    args.as_object_mut().unwrap().remove(key);
                }
            }
            let call = client.call_tool(
                CallToolRequestParams::new("generate_image")
                    .with_arguments(args.as_object().unwrap().clone()),
            );
            let observe = async {
                for _ in 0..100 {
                    let jobs = state
                        .jobs_store
                        .list_jobs(Some(project_id), None, 100)
                        .unwrap();
                    if let Some(job) = jobs.iter().find(|job| {
                        job.payload.get("model").and_then(Value::as_str) == Some(model)
                            && job.status.as_str() != "canceled"
                    }) {
                        let payload = job.payload.clone();
                        state.jobs_store.cancel_job(&job.id).unwrap();
                        return Some(payload);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
                None
            };
            let (result, payload) =
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    tokio::join!(call, observe)
                })
                .await
                .unwrap();
            let payload=payload.unwrap_or_else(||panic!("MCP must enqueue {reference_count} references through the real REST route; response: {result:?}"));
            assert_eq!(payload["referenceAssetIds"], json!(references));
            assert_eq!(payload["seed"], 0);
            assert_eq!(payload["width"], 512);
            assert_eq!(payload["height"], if custom { 768 } else { 512 });
            assert_eq!(
                payload["advanced"]["mossImageEdit"],
                if custom {
                    json!({"reference_max_edge":1024,"use_kv_cache":false})
                } else {
                    json!({"reference_max_edge":512,"use_kv_cache":true})
                }
            );
            assert_eq!(payload["advanced"]["steps"], if custom { 41 } else { 30 });
            assert_eq!(
                payload["advanced"]["guidanceScale"].as_f64(),
                Some(if custom { 4.5 } else { 4.0 })
            );
            assert_eq!(
                payload["advanced"]["schedulerShift"].as_f64(),
                Some(if custom { 6.0 } else { 5.0 })
            );
            assert!(payload["advanced"]
                .get("mossImageEditExecutionReport")
                .is_none());
            let facts = sceneworks_core::workflow_share::WorkflowAssetFacts {
                mode: "edit_image".into(),
                model: model.into(),
                prompt: payload["prompt"].as_str().unwrap().into(),
                negative_prompt: "blur".into(),
                seed: 0,
                width: Some(512),
                height: Some(if custom { 768 } else { 512 }),
            };
            let share =
                sceneworks_core::workflow_share::build_workflow_share_from(&facts, &payload);
            for key in [
                "steps",
                "guidanceScale",
                "schedulerShift",
                "mossImage",
                "mossImageEdit",
            ] {
                assert_eq!(
                    share.advanced[key], payload["advanced"][key],
                    "durable MCP replay {key}"
                );
            }
            assert_eq!(
                result.unwrap().is_error,
                Some(true),
                "the owned CPU test cancels after observing enqueue; no native worker runs"
            );
        }
    }
    for changes in [
        json!({"mossImageEdit":{"reference_max_edge":15}}),
        json!({"mossImageEdit":{"use_kv_cache":"false"}}),
        json!({"mossImageEdit":{"report":{}}}),
        json!({"mode":"generate","mossImageEdit":{}}),
        json!({"model":"flux2_dev","mossImageEdit":{}}),
        json!({"mossImage":{"acceleration":"unmerged_bf16"},"mossImageEdit":{}}),
    ] {
        let mut args = json!({"projectId":project_id,"mode":"edit_image","model":"moss_image_sft","prompt":"edit","referenceAssetIds":[ids[0]]});
        args.as_object_mut()
            .unwrap()
            .extend(changes.as_object().unwrap().clone());
        let result = client
            .call_tool(
                CallToolRequestParams::new("generate_image")
                    .with_arguments(args.as_object().unwrap().clone()),
            )
            .await;
        assert!(
            result.is_err(),
            "incompatible controls must fail strict MCP/shared parsing: {result:?}"
        );
    }
    let _ = client.cancel().await;
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn moss_edit_mcp_to_real_rest_preserves_one_reference_and_controls() {
    moss_edit_mcp_round_trip(1).await;
}

#[tokio::test]
async fn moss_edit_mcp_to_real_rest_preserves_four_ordered_reference_slots_and_controls() {
    moss_edit_mcp_round_trip(4).await;
}

async fn moss_edit_stored_payload_workflow_round_trip(count: usize) {
    use sceneworks_core::workflow_share::{build_workflow_share_from, WorkflowAssetFacts};
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","files":["adapter_config.json","adapter_model.safetensors"]}),
    );
    let (app, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"MOSS semantic replay"}),
    )
    .await;
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(16, 32)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let (status, asset) = request_multipart_upload(
        app.clone(),
        &format!(
            "/api/v1/projects/{}/assets",
            project["id"].as_str().unwrap()
        ),
        "owned-reference.png",
        "image/png",
        png.get_ref(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{asset}");
    let ids = vec![asset["id"].clone(); count];
    let (status,queued)=request(app.clone(),"POST","/api/v1/image/jobs",json!({"projectId":project["id"],"model":"moss_image_sft","mode":"edit_image","prompt":"blue cup","sourceAssetId":asset["id"],"referenceAssetId":asset["id"],"referenceAssetIds":ids,"advanced":{"mossImageEdit":{"reference_max_edge":768,"use_kv_cache":false}}})).await;
    assert_eq!(status, StatusCode::CREATED, "{queued}");
    let stored = state
        .jobs_store
        .get_job(queued["id"].as_str().unwrap())
        .unwrap();
    assert_eq!(stored.payload["width"], 512);
    assert_eq!(stored.payload["height"], 512);
    assert_eq!(stored.payload["negativePrompt"], "");
    assert_eq!(stored.payload.get("seed"), Some(&Value::Null));
    let facts = WorkflowAssetFacts {
        mode: "edit_image".into(),
        model: "moss_image_sft".into(),
        prompt: "blue cup".into(),
        negative_prompt: String::new(),
        seed: 7,
        width: Some(512),
        height: Some(512),
    };
    let share = build_workflow_share_from(&facts, &stored.payload);
    assert_eq!(share.inputs.len(), 1);
    assert_eq!(share.inputs[0].kind, "reference");
    assert_eq!(share.inputs[0].count as usize, count);
    let export = temp.path().join("exported-workflow.png");
    sceneworks_core::workflow_png::write_workflow_chunk(
        &image::RgbImage::new(2, 2),
        &export,
        Some(&share),
    )
    .unwrap();
    let bytes = std::fs::read(export).unwrap();
    let (status, inspected) = request_multipart_upload(
        app.clone(),
        "/api/v1/workflows/inspect",
        "shared.png",
        "image/png",
        &bytes,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{inspected}");
    assert_eq!(inspected["resolution"]["inputImagesRequired"], count);
    assert_eq!(
        inspected["resolution"]["components"][0]["state"],
        "resolved"
    );
    assert_eq!(
        inspected["workflow"]["advanced"]["mossImageEdit"],
        json!({"reference_max_edge":768,"use_kv_cache":false})
    );
    assert_eq!(inspected["workflow"]["advanced"]["steps"], 30);
    assert_eq!(
        inspected["workflow"]["advanced"]["guidanceScale"].as_f64(),
        Some(4.0)
    );
    assert_eq!(
        inspected["workflow"]["advanced"]["schedulerShift"].as_f64(),
        Some(5.0)
    );
    assert!(inspected["workflow"].get("sourceAssetId").is_none());
    assert!(inspected["workflow"].get("referenceAssetIds").is_none());
    state.jobs_store.cancel_job(&stored.id).unwrap();
}

#[tokio::test]
async fn moss_edit_stored_payload_share_and_inspect_count_matching_aliases_once() {
    moss_edit_stored_payload_workflow_round_trip(1).await;
}

#[tokio::test]
async fn moss_edit_stored_payload_share_and_inspect_preserve_four_repeated_slots() {
    moss_edit_stored_payload_workflow_round_trip(4).await;
}

#[tokio::test]
async fn moss_advanced_owned_npy_mcp_rest_and_portable_reacquisition_preserve_native_batch() {
    use rmcp::{
        model::CallToolRequestParams,
        transport::{
            streamable_http_client::StreamableHttpClientTransportConfig,
            StreamableHttpClientTransport,
        },
        ServiceExt,
    };
    use sceneworks_core::{moss_image_advanced::OwnedInputs, tensor_artifact};
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    super::yue2_catalog::seed_snapshot(
        &temp,
        &json!({"repo":"chinoll/Moss-Image-Edit","revision":"e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11","files":["adapter_config.json","adapter_model.safetensors"]}),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut settings = test_settings(&temp);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{address}");
    let (app, state) = create_app_with_state(settings).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name":"advanced native edit"}),
    )
    .await;
    let project_id = project["id"].as_str().unwrap();
    let route = format!("/api/v1/projects/{project_id}/assets");
    let mut rgb = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(16, 32)
        .write_to(&mut rgb, image::ImageFormat::Png)
        .unwrap();
    let (status, image) = request_multipart_upload(
        app.clone(),
        &route,
        "reference.png",
        "image/png",
        rgb.get_ref(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{image}");
    let mut noise = vec![0f32; 4 * 128];
    noise[0] = -0.;
    noise[1] = f32::from_bits(1);
    noise[128] = 0.25;
    let positive = tensor_artifact::encode_f32(&[2, 2, 4096], &vec![0.; 2 * 2 * 4096]).unwrap();
    let noise_bytes = tensor_artifact::encode_f32(&[4, 128, 1, 1], &noise).unwrap();
    let mut references = Vec::new();
    for (name, bytes) in [("positive.npy", positive), ("noise.npy", noise_bytes)] {
        let (status, asset) =
            request_multipart_upload(app.clone(), &route, name, tensor_artifact::MIME, &bytes)
                .await;
        assert_eq!(status, StatusCode::CREATED, "{asset}");
        assert_eq!(asset["type"], "tensor");
        references.push(json!({"assetId":asset["id"],"sha256":asset["tensor"]["sha256"]}));
    }
    let advanced = json!({"positive":{"values":references[0]},"noise":references[1],"negativePrompts":["blur","text"],"output":"latents"});
    let body = json!({"projectId":project_id,"model":"moss_image_sft","mode":"edit_image","referenceAssetIds":[image["id"]],"count":2,"width":16,"height":16,"seed":99,"mossImageAdvanced":advanced,"advanced":{"mossImage":{"max_sequence_length":2,"training_shift":2.5}}});
    let (status, job) = request(app.clone(), "POST", "/api/v1/image/jobs", body.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let stored = state
        .jobs_store
        .get_job(job["id"].as_str().unwrap())
        .unwrap();
    assert_eq!(stored.payload["seed"], Value::Null);
    assert_eq!(
        stored.payload["advanced"]["mossImageAdvanced"]["positive"]["values"]["metadata"]["shape"],
        json!([2, 2, 4096])
    );
    let model = stored.payload["modelManifestEntry"].as_object().unwrap();
    let store =
        sceneworks_core::project_store::ProjectStore::new(state.settings.data_dir.clone(), "test");
    let owned = OwnedInputs::load(&store, &stored.payload, model).unwrap();
    assert_eq!(
        owned
            .inputs()
            .noise
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        noise.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let native = sceneworks_core::moss_image::native_request(&stored.payload, model)
        .unwrap()
        .unwrap();
    assert!(native.negative_prompt.is_none());
    assert_eq!(native.count, 2);
    assert_eq!(owned.inputs().positive.unwrap().batch, 2);
    assert_eq!(
        sceneworks_core::image_request::ImageRequest::from_payload(&stored.payload).image_count(),
        4
    );
    let facts = sceneworks_core::workflow_share::WorkflowAssetFacts {
        mode: "edit_image".into(),
        model: "moss_image_sft".into(),
        prompt: "".into(),
        negative_prompt: "".into(),
        seed: 0,
        width: Some(16),
        height: Some(16),
    };
    let share = sceneworks_core::workflow_share::build_workflow_share_from(&facts, &stored.payload);
    assert_eq!(share.seed, None);
    assert_eq!(share.count, Some(2));
    let bytes = serde_json::to_vec(&share).unwrap();
    let (status, inspected) = request_multipart_upload(
        app.clone(),
        "/api/v1/workflows/inspect",
        "output.sceneworks-workflow.json",
        "application/json",
        &bytes,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{inspected}");
    assert_eq!(
        inspected["resolution"]["tensors"].as_array().unwrap().len(),
        2
    );
    assert!(
        inspected["workflow"]["advanced"]["mossImageAdvanced"]["noise"]
            .get("assetId")
            .is_none()
    );
    let mut replay = body.clone();
    replay["mossImageAdvanced"] = inspected["workflow"]["advanced"]["mossImageAdvanced"].clone();
    let (status, missing) =
        request(app.clone(), "POST", "/api/v1/image/jobs", replay.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{missing}");
    replay["mossImageAdvanced"]["positive"]["values"]["assetId"] = references[0]["assetId"].clone();
    replay["mossImageAdvanced"]["noise"]["assetId"] = references[1]["assetId"].clone();
    let (status, requeued) =
        request(app.clone(), "POST", "/api/v1/image/jobs", replay.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{requeued}");
    assert_eq!(requeued["payload"]["count"], 2);
    assert_eq!(requeued["payload"]["seed"], Value::Null);
    replay["mossImageAdvanced"]["noise"]["sha256"] = json!("f".repeat(64));
    let (status, mismatch) = request(app.clone(), "POST", "/api/v1/image/jobs", replay).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{mismatch}");
    for job in state
        .jobs_store
        .list_jobs(Some(project_id), None, 100)
        .unwrap()
    {
        state.jobs_store.cancel_job(&job.id).unwrap();
    }
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let client = rmcp::model::ClientInfo::default()
        .serve(StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{address}/mcp")),
        ))
        .await
        .unwrap();
    let discovery = client
        .call_tool(
            CallToolRequestParams::new("list_tensor_assets")
                .with_arguments(json!({"projectId":project_id}).as_object().unwrap().clone()),
        )
        .await
        .unwrap();
    assert_ne!(discovery.is_error, Some(true));
    let text = serde_json::to_string(&discovery).unwrap();
    assert!(text.contains(references[0]["sha256"].as_str().unwrap()));
    let args = json!({"projectId":project_id,"model":"moss_image_sft","mode":"edit_image","referenceAssetIds":[image["id"]],"count":2,"width":16,"height":16,"seed":99,"mossImage":{"max_sequence_length":2,"training_shift":2.5},"mossImageAdvanced":advanced});
    let call = client.call_tool(
        CallToolRequestParams::new("generate_image")
            .with_arguments(args.as_object().unwrap().clone()),
    );
    let observe = async {
        for _ in 0..100 {
            if let Some(job) = state
                .jobs_store
                .list_jobs(Some(project_id), None, 100)
                .unwrap()
                .into_iter()
                .find(|job| job.status.as_str() == "queued")
            {
                state.jobs_store.cancel_job(&job.id).unwrap();
                return job.payload;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("MCP advanced request never reached real REST enqueue")
    };
    let (result, payload) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        tokio::join!(call, observe)
    })
    .await
    .unwrap();
    assert_eq!(payload["count"], 2);
    assert_eq!(payload["seed"], Value::Null);
    assert_eq!(
        payload["advanced"]["mossImageAdvanced"],
        stored.payload["advanced"]["mossImageAdvanced"]
    );
    assert_eq!(
        result.unwrap().is_error,
        Some(true),
        "CPU fixture cancels after enqueue; no native worker runs"
    );
    let _ = client.cancel().await;
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn moss_output_pending_handoff_blocks_every_public_artifact_until_exact_recovery() {
    use sceneworks_core::{contracts::JobType, jobs_store::CreateJob};
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let (app, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let project = state
        .project_store
        .create_project("Atomic MOSS publication")
        .unwrap();
    let job=state.jobs_store.create_job(CreateJob{job_type:JobType::ImageEdit,project_id:Some(project.id.clone()),project_name:Some(project.name.clone()),payload:json!({"model":"moss_image_sft","mode":"edit_image","prompt":"edit","referenceAssetIds":["original"],"count":2,"width":16,"height":16,"advanced":{"mossImageAdvanced":{"output":"latents"}}}).as_object().unwrap().clone(),requested_gpu:"auto".into(),source_job_id:None,duplicate_of_job_id:None,attempts:1,initial_status:None}).unwrap();
    claim_job_as_worker(
        &app,
        &job.id,
        "moss-publisher",
        &[
            "image_edit",
            "gpu",
            if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "candle"
            },
        ],
    )
    .await;
    let set = "genset_publication";
    let root = std::path::Path::new(&project.path);
    let directory = root.join(format!("assets/tensors/moss-edit/{set}"));
    std::fs::create_dir_all(&directory).unwrap();
    let mut facts = Vec::new();
    for index in 0..2 {
        let relative = format!("assets/tensors/moss-edit/{set}/{index:04}.npy");
        std::fs::write(
            root.join(&relative),
            sceneworks_core::tensor_artifact::encode_f32(&[1, 128, 1, 1], &vec![index as f32; 128])
                .unwrap(),
        )
        .unwrap();
        let tensor = sceneworks_core::tensor_artifact::inspect(&root.join(&relative)).unwrap();
        facts.push(json!({"assetId":format!("asset_publication{index}"),"displayName":format!("MOSS latent {index}"),"type":"tensor","mimeType":"application/x-npy","tensor":tensor,"mediaPath":relative,"mode":"edit_image","model":"moss_image_sft","seed":null,"createdAt":"2026-10-23T00:00:00Z","rawAdapterSettings":{"mossImageEditExecutionReport":{"actual":true}}}));
    }
    let result = json!({"generationSetId":set,"expectedCount":2,"generationSet":{"id":set,"model":"moss_image_sft","mode":"edit_image","count":2},"assetWrites":facts});
    let completion = json!({"status":"completed","stage":"completed","progress":1,"message":"Done","workerId":"moss-publisher","result":result});
    state
        .progress_side_effects_fail_job_ids
        .lock()
        .insert(job.id.clone());
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/progress", job.id),
        completion.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let (_, pending) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/jobs/{}", job.id),
        Value::Null,
    )
    .await;
    assert_eq!(pending["status"], "completed");
    assert_eq!(pending["result"]["outputPublication"]["state"], "pending");
    assert!(pending["result"].get("assetWrites").is_none());
    assert!(!pending.to_string().contains("mossImageEditExecutionReport"));
    let file_route = format!(
        "/api/v1/projects/{}/files/assets/tensors/moss-edit/{set}/0000.npy",
        project.id
    );
    let (status, _) = request(app.clone(), "GET", &file_route, Value::Null).await;
    assert_eq!(status, StatusCode::CONFLICT);
    for alias in [
        format!("assets/tensors/./moss-edit/{set}/0000.npy"),
        format!("assets/tensors//moss-edit/{set}/0000.npy"),
    ] {
        assert!(
            crate::moss_output_publication::require_path(&state, &project.id, &alias)
                .await
                .is_err(),
            "{alias}"
        );
    }
    #[cfg(unix)]
    {
        let alias = root.join("assets/uploads/moss-output-alias.npy");
        std::os::unix::fs::symlink(directory.join("0000.npy"), &alias).unwrap();
        assert!(crate::moss_output_publication::require_path(
            &state,
            &project.id,
            "assets/uploads/moss-output-alias.npy"
        )
        .await
        .is_err());
    }
    // Simulate an interrupted old per-file side effect: same-job partial metadata is not public.
    state
        .project_store
        .write_generation_set(
            &project.id,
            &job.id,
            &result["generationSet"],
            facts.first(),
        )
        .unwrap();
    let mut partial = facts[0].clone();
    partial["extra"] =
        json!({"mossPublication":{"jobId":job.id,"attempt":job.attempts,"generationSetId":set}});
    state
        .project_store
        .persist_generated_asset(&project.id, &job.id, set, &partial)
        .unwrap();
    let (status, list) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{}/assets?scope=all", project.id),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list, json!([]));
    for suffix in ["", "/workflow"] {
        let (status, _) = request(
            app.clone(),
            "GET",
            &format!(
                "/api/v1/projects/{}/assets/asset_publication0{suffix}",
                project.id
            ),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
    }
    state
        .progress_side_effects_fail_job_ids
        .lock()
        .remove(&job.id);
    let second = directory.join("0001.npy");
    let second_bytes = std::fs::read(&second).unwrap();
    std::fs::remove_file(&second).unwrap();
    let (status, _) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/progress", job.id),
        completion.clone(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "the actual API refuses an unavailable second artifact"
    );
    let (_, hidden) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{}/assets?scope=all", project.id),
        Value::Null,
    )
    .await;
    assert_eq!(hidden, json!([]));
    std::fs::write(second, second_bytes).unwrap();
    // Reconciliation uses the original durable handoff. A replacement retry payload is ignored.
    state
        .progress_side_effects_fail_job_ids
        .lock()
        .remove(&job.id);
    let mut retry = completion;
    retry["result"] = json!({"forged":true});
    let (status, ready) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/progress", job.id),
        retry,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{ready}");
    assert_eq!(ready["result"]["assets"].as_array().unwrap().len(), 2);
    assert!(ready["result"].get("outputPublication").is_none());
    assert!(ready["result"].get("forged").is_none());
    assert_eq!(ready["result"]["assets"][0]["recipe"]["seed"], Value::Null);
    let (status, listed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{}/assets?scope=all", project.id),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed.as_array().unwrap().len(), 2);
    let mut mismatched = ready["result"]["assets"][0].clone();
    mismatched["id"] = json!("asset_other");
    assert!(
        crate::moss_output_publication::require_asset(&state, &project.id, &mismatched)
            .await
            .is_err()
    );
    // Exact canonical artifacts are readable; sidecars/temp paths never become public outputs.
    crate::moss_output_publication::require_path(
        &state,
        &project.id,
        &format!("assets/tensors/moss-edit/{set}/0000.npy"),
    )
    .await
    .unwrap();
    for name in [
        "0000.sceneworks.json",
        "0000.tmp.npy",
        "0000.sceneworks-workflow.json",
    ] {
        assert!(crate::moss_output_publication::require_path(
            &state,
            &project.id,
            &format!("assets/tensors/moss-edit/{set}/{name}")
        )
        .await
        .is_err());
    }
    state.jobs_store.clear_job(&job.id).unwrap();
    crate::moss_output_publication::require_path(
        &state,
        &project.id,
        &format!("assets/tensors/moss-edit/{set}/0000.npy"),
    )
    .await
    .unwrap();
    assert!(crate::moss_output_publication::require_path(
        &state,
        &project.id,
        "assets/tensors/moss-edit/genset_missing/0000.npy"
    )
    .await
    .is_err());
}
