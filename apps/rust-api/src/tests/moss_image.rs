use super::support::*;
fn setup(temp: &tempfile::TempDir) {
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "moss_image_sft", "OpenMOSS-Team/MOSS-Image-SFT");
    let model = json!({"id":"moss_image_sft","name":"MOSS SFT","type":"image","family":"moss_image","capabilities":["text_to_image"],"downloads":[{"provider":"huggingface","repo":"OpenMOSS-Team/MOSS-Image-SFT","revision":"fc5de0d56fc910cbca6f6625a70f5f582b26deb1","files":["transformer/config.json"]}],"image":{"supportsSchedulerShift":true,"maxPromptBatchImages":8},"defaults":{"resolution":"1024x1024","count":1,"steps":50,"guidanceScale":3,"schedulerShift":5},"limits":{"minDimension":16,"maxDimension":4096,"requiresDimensionsMultipleOf":16},"conditionalComponents":[{"provider":"huggingface","repo":"lodestones/MOSS-Image-SFT-8-steps-dmd","revision":"5ff314b4c1470876b114b810146085db8a181a6e","componentId":"moss-dmd-dataspace","files":["dataspace.safetensors"],"requiredFor":["moss_dmd_unmerged_bf16","moss_dmd_full_fp32"]}]});
    let mut base = model.clone();
    base["id"] = json!("moss_image_base");
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({"schemaVersion":1,"models":[model,base]}).to_string(),
    )
    .unwrap();
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
