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
