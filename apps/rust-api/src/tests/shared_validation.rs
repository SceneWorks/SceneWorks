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
            "/api/v1/video/jobs",
            "video",
            json!({"projectId":"p","model":"ltx_2_5","mode":"text_to_video","prompt":"tree","sourceClipAssetIds":["clip",5]}),
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
        json!({"jobType":"video_upscale","projectId":project["id"],"payload":payload}),
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
        json!({"jobType":"video_upscale","projectId":project["id"],"payload":invalid}),
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
