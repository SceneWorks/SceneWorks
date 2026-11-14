//! Static boundary tests only: no worker or inference execution.
use super::support::*;

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
