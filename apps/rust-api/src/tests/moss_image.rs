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
