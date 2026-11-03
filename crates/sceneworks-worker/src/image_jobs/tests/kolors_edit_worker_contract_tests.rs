use super::resolve_candle_kolors_edit_init;
use super::super::routing::resolve_candle_image_route;
use super::super::routing::CandleImageRoute;
use super::super::ImageRequest;
use crate::Path;
use crate::PathBuf;
use crate::ProjectStore;
use crate::Settings;
use crate::Value;
use crate::DEFAULT_HUGGINGFACE_BASE_URL;
use crate::DEFAULT_MAX_LORA_URL_BYTES;
use crate::DEFAULT_MAX_MODEL_URL_BYTES;
use serde_json::json;

fn settings(data_dir: &Path) -> Settings {
    Settings {
        api_url: "http://127.0.0.1".to_owned(),
        config_dir: data_dir.join("config"),
        worker_id: "test-worker".to_owned(),
        gpu_id: "gpu-0".to_owned(),
        huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
        max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
        backend_candle_enabled: true,
        ..crate::Settings::for_test(data_dir.to_path_buf())
    }
}

#[test]
fn kolors_edit_route_resolves_the_submitted_source_and_strength() {
    let data = tempfile::tempdir().expect("data dir");
    let settings = settings(data.path());
    let weights = data.path().join("models/kolors-test");
    std::fs::create_dir_all(&weights).expect("model path");

    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .create_project("Kolors edit contract")
        .expect("project");
    let project_path = PathBuf::from(&project.path);
    let source_file = data.path().join("kolors-source.png");
    image::RgbImage::from_pixel(48, 32, image::Rgb([180, 40, 90]))
        .save(&source_file)
        .expect("source PNG");
    let asset = store
        .import_asset(
            &project.id,
            sceneworks_core::project_store::UploadAsset {
                filename: "kolors-source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source_file,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("source asset");
    let asset_id = asset["id"].as_str().expect("asset id");
    let request = ImageRequest::from_payload(
        json!({
            "projectId": project.id,
            "model": "kolors",
            "mode": "edit_image",
            "sourceAssetId": asset_id,
            "fitMode": "stretch",
            "advanced": {
                "modelPath": weights,
                "strength": "0.72"
            }
        })
        .as_object()
        .expect("request object"),
    );

    assert_eq!(
        resolve_candle_image_route(&request, &settings),
        Some(CandleImageRoute::KolorsEdit)
    );
    let (source, strength) = resolve_candle_kolors_edit_init(&request, &settings, &project_path)
        .expect("source resolution")
        .expect("Kolors edit init");
    assert_eq!((source.width, source.height), (48, 32));
    assert_eq!(source.pixels[0..3], [180, 40, 90]);
    assert!((strength - 0.72).abs() < f32::EPSILON);

    let missing_source = ImageRequest::from_payload(
        json!({
            "projectId": project.id,
            "model": "kolors",
            "mode": "edit_image",
            "advanced": { "modelPath": weights }
        })
        .as_object()
        .expect("request object"),
    );
    assert_ne!(
        resolve_candle_image_route(&missing_source, &settings),
        Some(CandleImageRoute::KolorsEdit)
    );
    assert!(
        resolve_candle_kolors_edit_init(&missing_source, &settings, &project_path)
            .expect("missing source is not an I/O error")
            .is_none()
    );
}

#[test]
fn kolors_ip_composites_refuse_before_provider_precedence() {
    let data = tempfile::tempdir().expect("data dir");
    let settings = settings(data.path());
    let weights = data.path().join("models/kolors-test");
    std::fs::create_dir_all(&weights).expect("model path");
    let request = |advanced: Value| {
        ImageRequest::from_payload(
            json!({
                "projectId": "project",
                "model": "kolors",
                "mode": "character_image",
                "referenceAssetId": "identity-reference",
                "advanced": advanced,
            })
            .as_object()
            .expect("request object"),
        )
    };
    let ip_and_pid = request(json!({"modelPath": weights, "usePid": true}));
    assert_eq!(
        resolve_candle_image_route(&ip_and_pid, &settings),
        Some(CandleImageRoute::KolorsCompositeReject)
    );
    let ip_and_pose = request(json!({
        "modelPath": weights,
        "poses": [{"keypoints": []}],
    }));
    assert_eq!(
        resolve_candle_image_route(&ip_and_pose, &settings),
        Some(CandleImageRoute::KolorsCompositeReject)
    );
}
