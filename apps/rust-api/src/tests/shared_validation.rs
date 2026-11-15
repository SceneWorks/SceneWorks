//! Static boundary tests only: no worker or inference execution.
use super::support::*;

fn preview_settings(temp: &tempfile::TempDir) -> Settings {
    let settings = test_settings(temp);
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .unwrap();
    settings
}

#[tokio::test]
async fn options_preview_uses_authenticated_server_facts_and_preserves_result_order() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = preview_settings(&temp);
    settings.access_token = "preview-secret".into();
    let app = create_app(settings).unwrap();
    let body = json!({
        "validatorSchemaVersion": 1,
        "kind": "image",
        "requests": [
            {"projectId":"p","model":"flux_dev","prompt":"a tree"},
            {"projectId":"p","model":"flux_dev","prompt":"a tree","advanced":{"steps":"bad"}},
            {"projectId":"p","model":"forged_model","prompt":"a tree"},
            {"projectId":"p","prompt":"a tree"},
            {"projectId":"p","modelId":"flux_dev","prompt":"a tree"}
        ]
    });
    let (status, _) = request(
        app.clone(),
        "POST",
        "/api/v1/options/validate",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, response) = request_with_headers(
        app,
        "POST",
        "/api/v1/options/validate",
        body,
        &[("authorization", "Bearer preview-secret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["validatorSchemaVersion"], 1);
    assert_eq!(response["results"].as_array().unwrap().len(), 5);
    assert_eq!(response["results"][0]["ok"], true, "{response}");
    assert_eq!(response["results"][0]["resolved"]["model"], "flux_dev");
    assert_eq!(response["results"][1]["ok"], false);
    assert_eq!(response["results"][1]["refusal"]["field"], "advanced.steps");
    assert_eq!(response["results"][2]["ok"], false);
    assert_eq!(response["results"][2]["refusal"]["field"], "model");
    assert_eq!(response["results"][3]["ok"], true, "{response}");
    assert_eq!(response["results"][4]["ok"], true, "{response}");
}

#[tokio::test]
async fn options_preview_rejects_unsupported_contracts_and_keeps_interleave_shifts_distinct() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(preview_settings(&temp)).unwrap();
    let path = "/api/v1/options/validate";
    let (status, mismatch) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":2,"kind":"image","requests":[{}]}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(mismatch["code"], "validator_version_mismatch");
    for body in [
        json!({"validatorSchemaVersion":1,"kind":"arbitrary","requests":[{}]}),
        json!({"validatorSchemaVersion":1,"kind":"image","requests":[]}),
        json!({"validatorSchemaVersion":1,"kind":"image","requests":vec![json!({});33]}),
    ] {
        let (status, _) = request(app.clone(), "POST", path, body).await;
        assert!(
            !status.is_success(),
            "invalid validation envelope passed: {status}"
        );
    }
    let forged = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(json!({"validatorSchemaVersion":1,"kind":"image","requests":[{}],"modelEntry":{"id":"forged"}}).to_string()))
        .unwrap();
    let response = app.clone().oneshot(forged).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let (status, response) = request(
        app,
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"interleave","requests":[{
            "projectId":"p","model":"sensenova_u1_8b","prompt":"a tree",
            "advanced":{"schedulerShift":9,"timestepShift":1.5}
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["results"][0]["ok"], true, "{response}");
    assert_eq!(
        response["results"][0]["resolved"]["advanced"]["schedulerShift"],
        9
    );
    assert_eq!(
        response["results"][0]["resolved"]["advanced"]["timestepShift"],
        1.5
    );
}

#[tokio::test]
async fn options_preview_resolves_vector_training_and_model_free_policies() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(preview_settings(&temp)).unwrap();
    let path = "/api/v1/options/validate";
    let (status, vector) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"vectorPrompt","requests":[{
            "projectId":"p","prompt":"a tree","rasterModel":"flux_dev","vectorModel":"starvector_1b"
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{vector}");
    assert_eq!(vector["results"][0]["ok"], true, "{vector}");

    let (status, detail) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"imageDetail","requests":[
            {"projectId":"p","sourceAssetId":"source"},
            {"projectId":"p","sourceAssetId":"source","model":null},
            {"projectId":"p","sourceAssetId":"source","model":42},
            {"projectId":"p","sourceAssetId":"source","model":" flux_dev "}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detail}");
    assert_eq!(detail["results"][0]["ok"], true, "{detail}");
    assert_eq!(detail["results"][1]["refusal"]["field"], "model");
    assert_eq!(detail["results"][2]["refusal"]["field"], "model");
    assert_eq!(detail["results"][3]["ok"], true, "{detail}");

    let (status, utility) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"utility","requests":[{
            "modelId":42,"advanced":{"mlxQuantize":4}
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{utility}");
    assert_eq!(utility["results"][0]["refusal"]["field"], "modelId");

    let target = crate::training::effective_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .unwrap();
    let mut invalid_config = serde_json::to_value(target.defaults).unwrap();
    invalid_config["steps"] = json!(0);
    let (status, training) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"training","requests":[{
            "targetId":target.id,"datasetId":"dataset","outputName":"adapter",
            "config":invalid_config
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["results"][0]["ok"], false, "{training}");
    assert!(
        training["results"][0]["refusal"]["field"]
            .as_str()
            .is_some_and(|field| field.contains("steps")),
        "{training}"
    );

    let (status, yue2) = request(
        app,
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"yue2","requests":[{
            "kind":"create","lyrics":"verse","computePolicy":"auto"
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{yue2}");
    assert_eq!(yue2["results"][0]["ok"], true, "{yue2}");
}

#[tokio::test]
async fn options_preview_preserves_ltx_and_iris_baseline_refusals() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(preview_settings(&temp)).unwrap();
    let path = "/api/v1/options/validate";
    let (status, video) = request(
        app.clone(),
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"video","requests":[{
            "projectId":"p","model":"ltx_2_3","mode":"extend_clip",
            "sourceClipAssetId":"clip","prompt":"A lighthouse in heavy fog"
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{video}");
    assert_eq!(video["results"][0]["ok"], false, "{video}");
    assert_eq!(video["results"][0]["refusal"]["field"], "mode", "{video}");

    let target = crate::training::effective_training_targets()
        .targets
        .into_iter()
        .find(|target| target.id == "iris_3b_depth")
        .unwrap();
    let mut config = serde_json::to_value(target.defaults).unwrap();
    config["saveEvery"] = json!(32001);
    let (status, training) = request(
        app,
        "POST",
        path,
        json!({"validatorSchemaVersion":1,"kind":"training","requests":[{
            "targetId":target.id,"datasetId":"dataset","outputName":"adapter",
            "config":config
        }]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{training}");
    assert_eq!(training["results"][0]["ok"], false, "{training}");
}

#[tokio::test]
async fn shared_refusals_survive_http_raw_deserialization() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    for (path, kind, payload) in [
        (
            "/api/v1/image/jobs",
            "image",
            json!({"projectId":"p","model":"flux_dev","prompt":"tree","width":"1024"}),
        ),
        (
            "/api/v1/image/jobs",
            "image",
            json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"steps":"bad"}}),
        ),
        (
            "/api/v1/image/jobs",
            "image",
            json!({"projectId":"p","model":"flux_dev","prompt":"tree","advanced":{"enhanceTemperature":1.0}}),
        ),
        (
            "/api/v1/video/jobs",
            "video",
            json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":"tree","sourceClipAssetIds":["clip",5]}),
        ),
        (
            "/api/v1/image/interleave/jobs",
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","advanced":{"guidanceScale":"NaN"}}),
        ),
        (
            "/api/v1/audio/jobs",
            "audio",
            json!({"projectId":"p","model":"kokoro_82m","prompt":"hello","steps":"bad"}),
        ),
    ] {
        let expected =
            sceneworks_core::request_validation::validate_shape(kind, payload.as_object().unwrap())
                .unwrap_err();
        let (status, body) = request(app.clone(), "POST", path, payload).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["code"], expected.code);
        assert_eq!(body["field"], expected.field);
        assert_eq!(body["context"], *expected.context);
    }
}

#[tokio::test]
async fn shared_video_upscale_refusals_match_raw_create_retry_duplicate() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = create_project(app.clone(), "Static utility boundary").await;
    let payload = json!({"sourceAssetId":"source-clip","factor":2});
    let (status, job) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({"type":"video_upscale","requestedGpu":"auto","projectId":project["id"],"payload":payload}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let invalid = json!({"sourceAssetId":"source-clip","factor":3});
    let expected = sceneworks_core::request_validation::validate(
        "videoUpscale",
        invalid.as_object().unwrap(),
        &Default::default(),
    )
    .unwrap_err();
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({"type":"video_upscale","requestedGpu":"auto","projectId":project["id"],"payload":invalid}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], expected.code);
    for operation in ["retry", "duplicate"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{}/{operation}", job["id"].as_str().unwrap()),
            json!({"payloadChanges":{"factor":3}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{operation}: {body}");
        assert_eq!(body["code"], expected.code);
        assert_eq!(body["field"], expected.field);
        assert_eq!(body["context"], *expected.context);
    }
}

#[tokio::test]
async fn shared_person_and_prompt_refusals_match_typed_raw_retry_and_duplicate() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = create_project(app.clone(), "Person static boundary").await;
    let project_id = project["id"].as_str().unwrap();
    for (kind, job_type, path, valid, invalids) in [
        (
            "personDetect",
            "person_detect",
            format!("/api/v1/projects/{project_id}/person-tracks/detections"),
            json!({"sourceAssetId":"clip"}),
            vec![
                json!({}),
                json!({"sourceAssetId":null}),
                json!({"sourceAssetId":"clip","sourceTimestamp":-1}),
                json!({"sourceAssetId":"clip","advanced":null}),
            ],
        ),
        (
            "personTrack",
            "person_track",
            format!("/api/v1/projects/{project_id}/person-tracks/jobs"),
            json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{"id":"person"}}),
            vec![
                json!({}),
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{}}),
                json!({"sourceAssetId":"clip","representativeFrameAssetId":null,"detection":{"id":"person"}}),
                json!({"sourceAssetId":"clip","representativeFrameAssetId":"frame","detection":{"id":"person"},"trackName":null}),
            ],
        ),
        (
            "promptRefine",
            "prompt_refine",
            "/api/v1/prompts/refine".to_owned(),
            json!({"prompt":"tree"}),
            vec![
                json!({"prompt":"tree","sourceAssetIds":null}),
                json!({"task":"image_caption","sourceAssetId":"source","prompt":null}),
                json!({"task":"image_caption","imagePaths":["",42]}),
                json!({"task":"image_caption","imagePaths":["image.png",42]}),
                json!({"task":"image_caption","imagePaths":null}),
                json!({"task":"image_caption","sourceAssetIds":["asset"],"imagePaths":[]}),
                json!({"task":"image_caption","imagePaths":vec!["image.png";7]}),
            ],
        ),
    ] {
        for invalid in &invalids {
            let expected = sceneworks_core::request_validation::validate_shape(
                kind,
                invalid.as_object().unwrap(),
            )
            .unwrap_err();
            for (route, body) in [
                (path.as_str(), invalid.clone()),
                (
                    "/api/v1/jobs",
                    json!({"type":job_type,"requestedGpu":"auto","projectId":project_id,"payload":invalid}),
                ),
            ] {
                let (status, body) = request(app.clone(), "POST", route, body).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{route}: {body}");
                assert_eq!(body["code"], expected.code);
                assert_eq!(body["field"], expected.field);
                assert_eq!(body["context"], *expected.context);
            }
        }
        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({"type":job_type,"requestedGpu":"auto","projectId":project_id,"payload":valid}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{job_type}: {job}");
        let changes = if kind == "promptRefine" {
            json!({"sourceAssetIds":null})
        } else {
            json!({"sourceAssetId":""})
        };
        let mut patched = valid.as_object().unwrap().clone();
        patched.extend(changes.as_object().unwrap().clone());
        let expected =
            sceneworks_core::request_validation::validate_shape(kind, &patched).unwrap_err();
        for operation in ["retry", "duplicate"] {
            let (status, body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/jobs/{}/{operation}", job["id"].as_str().unwrap()),
                json!({"payloadChanges":changes}),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{job_type}/{operation}: {body}"
            );
            assert_eq!(body["code"], expected.code);
            assert_eq!(body["field"], expected.field);
            assert_eq!(body["context"], *expected.context);
        }
    }
}

#[tokio::test]
async fn shared_promptless_svd_matches_create_retry_duplicate_and_preserves_raw_guard() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = create_project(app.clone(), "Promptless video").await;
    let payload = json!({"projectId":project["id"],"model":"svd","mode":"image_to_video","prompt":"","sourceAssetId":"image"});
    let (status, job) = request(app.clone(), "POST", "/api/v1/video/jobs", payload.clone()).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["prompt"], "");
    for operation in ["retry", "duplicate"] {
        let (status, replay) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{}/{operation}", job["id"].as_str().unwrap()),
            json!({"payloadChanges":{}}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{operation}: {replay}");
        assert_eq!(replay["payload"]["prompt"], "");
    }
    let (status, refusal) = request(app.clone(), "POST", "/api/v1/jobs", json!({"type":"video_generate","requestedGpu":"auto","projectId":project["id"],"payload":payload})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(refusal["detail"]
        .as_str()
        .unwrap()
        .contains("must be created via POST /api/v1/video/jobs"));
    let ordinary =
        json!({"projectId":project["id"],"model":"ltx_2_5","mode":"text_to_video","prompt":""});
    let (status, refusal) = request(app, "POST", "/api/v1/video/jobs", ordinary).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refusal}");
    assert_eq!(refusal["field"], "prompt");
}

#[tokio::test]
async fn sensenova_http_extractor_preserves_authored_numbers_and_shared_effective_values() {
    use axum::extract::FromRequest;
    use sceneworks_core::request_validation::resolve;
    let facts = sceneworks_core::registry::builtin_model_spec("sensenova_u1_8b")
        .unwrap()
        .project(None);
    for (kind, payload) in [
        (
            "vqa",
            json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What?","maxNewTokens":"4096","maxImagePixels":2.5}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"9999","height":12.9,"advanced":{"timestepShift":1.5,"schedulerShift":9}}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree"}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","maxImages":2}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","maxImages":99}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","maxImages":2,"advanced":{"maxImages":4}}),
        ),
        (
            "interleave",
            json!({"projectId":"p","model":"sensenova_u1_8b","prompt":"tree","width":"2.5","height":"1e100","maxImages":"bad"}),
        ),
        (
            "vqa",
            json!({"projectId":"p","model":"sensenova_u1_8b","sourceAssetId":"source","question":"What?","maxNewTokens":null,"maxImagePixels":"bad"}),
        ),
    ] {
        let request = axum::http::Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from(payload.to_string()))
            .unwrap();
        let typed = if kind == "vqa" {
            let crate::ApiJson(value) =
                crate::ApiJson::<crate::VqaJobRequest>::from_request(request, &())
                    .await
                    .unwrap();
            serde_json::to_value(value).unwrap()
        } else {
            let crate::ApiJson(value) =
                crate::ApiJson::<crate::InterleaveJobRequest>::from_request(request, &())
                    .await
                    .unwrap();
            serde_json::to_value(value).unwrap()
        };
        for key in [
            "width",
            "height",
            "maxNewTokens",
            "maxImagePixels",
            "maxImages",
        ] {
            assert_eq!(typed.get(key), payload.get(key));
        }
        assert_eq!(
            resolve(kind, typed.as_object().unwrap(), facts.as_object().unwrap()).unwrap(),
            resolve(
                kind,
                payload.as_object().unwrap(),
                facts.as_object().unwrap()
            )
            .unwrap()
        );
    }
}

#[tokio::test]
async fn interleave_distinct_shifts_survive_typed_and_raw_http_ingress() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = create_project(app.clone(), "Interleave shift carriers").await;
    let payload = json!({
        "projectId": project["id"],
        "model": "sensenova_u1_8b",
        "prompt": "tree",
        "advanced": { "schedulerShift": 9, "timestepShift": 1.5 },
    });
    for (path, body) in [
        ("/api/v1/image/interleave/jobs", payload.clone()),
        (
            "/api/v1/jobs",
            json!({
                "type": "image_interleave",
                "projectId": project["id"],
                "requestedGpu": "auto",
                "payload": payload,
            }),
        ),
    ] {
        let (status, response) = request(app.clone(), "POST", path, body).await;
        assert_eq!(status, StatusCode::CREATED, "{path}: {response}");
        assert_eq!(response["payload"]["advanced"]["schedulerShift"], 9);
        assert_eq!(response["payload"]["advanced"]["timestepShift"], 1.5);
    }
}

#[tokio::test]
async fn image_detail_registered_controls_match_raw_http_and_worker_kind() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    let payload =
        json!({"model":"flux_dev","sourceAssetId":"source","advanced":{"tileControlNetRepo":42}});
    let expected = sceneworks_core::request_validation::validate_shape(
        "imageDetail",
        payload.as_object().unwrap(),
    )
    .unwrap_err();
    let (status, body) = request(
        app,
        "POST",
        "/api/v1/jobs",
        json!({"type":"image_detail","projectId":"p","requestedGpu":"auto","payload":payload}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["field"], expected.field);
    assert_eq!(body["code"], expected.code);
    assert_eq!(
        sceneworks_core::request_validation::kind_for_job(
            &sceneworks_core::contracts::JobType::ImageDetail
        ),
        Some("imageDetail")
    );
}

#[tokio::test]
async fn source_utility_registered_fields_refuse_before_raw_job_parsing() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    for job_type in ["model_convert", "dataset_analysis", "training_caption"] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({"type":job_type,"projectId":"p","requestedGpu":"auto","payload":{"advanced":{"mlxQuantize":"bad"}}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["field"], "advanced.mlxQuantize", "{body}");
    }
}

#[tokio::test]
async fn shared_source_utility_refusal_precedes_local_paths_on_create_retry_duplicate() {
    let temp = tempfile::tempdir().unwrap();
    let (app, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let valid_options = json!({"modelId":"model-1","advanced":{"mlxQuantize":"4"}});
    let job = state
        .jobs_store
        .create_job(crate::CreateJob {
            job_type: crate::JobType::ModelConvert,
            project_id: None,
            project_name: None,
            payload: valid_options.as_object().unwrap().clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    for output in [Value::Null, json!("/outside-conversion-root/model")] {
        for malformed in [true, false] {
            let mut payload = valid_options.clone();
            payload["outputDir"] = output.clone();
            if malformed {
                payload["advanced"]["mlxQuantize"] = json!("bad");
            }
            let expected = sceneworks_core::request_validation::validate_shape(
                "utility",
                payload.as_object().unwrap(),
            )
            .err();
            let mut paths = vec![(
                "/api/v1/jobs".to_owned(),
                json!({"type":"model_convert","requestedGpu":"auto","payload":payload}),
            )];
            for operation in ["retry", "duplicate"] {
                paths.push((
                    format!("/api/v1/jobs/{}/{operation}", job.id),
                    json!({"payloadChanges":payload}),
                ));
            }
            for (path, body) in paths {
                let (status, refusal) = request(app.clone(), "POST", &path, body).await;
                assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {refusal}");
                if let Some(expected) = expected.as_ref() {
                    assert_eq!(refusal["code"], expected.code, "{path}: {refusal}");
                    assert_eq!(refusal["field"], expected.field, "{path}: {refusal}");
                    assert_eq!(refusal["context"], *expected.context, "{path}: {refusal}");
                } else {
                    assert!(refusal["field"].is_null(), "{path}: {refusal}");
                    let detail = refusal["detail"].as_str().unwrap();
                    assert!(
                        detail.contains("outputDir")
                            || detail.contains("conversion")
                            || detail.contains("models/mlx"),
                        "{path}: {refusal}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn shared_malformed_image_inputs_render_identically_over_http_and_mcp() {
    use rmcp::transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
    };
    use rmcp::{model::CallToolRequestParams, ServiceExt};
    let temp = tempfile::tempdir().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut settings = test_settings(&temp);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{addr}");
    let app = create_app(settings).unwrap();
    let served = app.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            served.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let client = rmcp::model::ClientConfig::default()
        .serve(StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp")),
        ))
        .await
        .unwrap();
    let mut cases = Vec::new();
    for member in [Value::Null, json!(42), json!(" bad")] {
        cases.push(("referenceAssetIds", json!({"projectId":"p","model":"moss_image_base","mode":"edit_image","prompt":"edit","referenceAssetIds":["source",member]})));
    }
    for carrier in [json!(42), json!("invalid"), json!([]), json!(false)] {
        cases.push(("mossImageAdvanced", json!({"projectId":"p","model":"moss_image_base","mode":"edit_image","prompt":"edit","referenceAssetIds":["source"],"mossImageAdvanced":carrier})));
    }
    for (field, payload) in cases {
        let expected = serde_json::to_value(
            sceneworks_core::request_validation::validate_shape(
                "image",
                payload.as_object().unwrap(),
            )
            .unwrap_err(),
        )
        .unwrap();
        let (status, body) =
            request(app.clone(), "POST", "/api/v1/image/jobs", payload.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body, expected);
        assert_eq!(body["code"], "invalid_generation_option");
        assert_eq!(body["field"], field);
        assert_eq!(
            body["context"],
            json!({"kind":"image","model":"moss_image_base","mode":"edit_image","field":field})
        );
        let error = client
            .call_tool(
                CallToolRequestParams::new("generate_image")
                    .with_arguments(payload.as_object().unwrap().clone()),
            )
            .await
            .unwrap_err();
        let rmcp::ServiceError::McpError(error) = error else {
            panic!("unexpected MCP failure: {error:?}");
        };
        assert_eq!(error.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert_eq!(error.data, Some(expected));
    }
    let _ = client.cancel().await;
    server.abort();
    let _ = server.await;
}
