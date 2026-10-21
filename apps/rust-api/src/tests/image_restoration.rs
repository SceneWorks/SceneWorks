//! Iris-3B restoration (sc-25683): the pre-submission plan preview endpoint and the typed
//! upscale-job / Studio post-processing validation for the `iris` engine.
use super::support::*;

/// A JPEG carrying an EXIF `Orientation = 6` (Rotate90) APP1 segment.
fn rotated_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut plain = Vec::new();
    image::RgbImage::from_pixel(width, height, image::Rgb([40, 80, 120]))
        .write_to(
            &mut std::io::Cursor::new(&mut plain),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08".to_vec();
    tiff.extend_from_slice(&[0x00, 0x01]);
    tiff.extend_from_slice(&[
        0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00,
    ]);
    tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    let mut app1 = b"Exif\x00\x00".to_vec();
    app1.extend_from_slice(&tiff);
    let length = (app1.len() + 2) as u16;
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&length.to_be_bytes());
    jpeg.extend_from_slice(&app1);
    jpeg.extend_from_slice(&plain[2..]);
    jpeg
}

#[tokio::test]
async fn plan_preview_by_size_uses_the_engine_planner_and_labels_budgets_honestly() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "width": 512, "height": 384 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let plan = &body["items"][0]["plan"];
    assert_eq!(plan["scale"], json!(4.0), "the default is the model's 4x");
    assert_eq!(plan["output"], json!({ "width": 2048, "height": 1536 }));
    assert_eq!(plan["tileCount"], json!(6));
    assert_eq!(plan["budgetApplied"], json!(false));
    assert_eq!(body["items"][0]["error"], Value::Null);

    let (_, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "width": 4032, "height": 3024, "scale": 4, "inputSizing": "budgeted" }),
    )
    .await;
    let summary = body["items"][0]["plan"]["summary"].as_str().unwrap();
    assert!(
        summary.starts_with("budgeted: 683x512 input → 2732x2048"),
        "{summary}"
    );
    assert!(!summary.contains("of the original"), "{summary}");

    // 1x, odd size, fractional scale, colour fix off — echoed as applied.
    let (_, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "width": 1001, "height": 777, "scale": 1.3, "inputSizing": "original", "colorFix": false }),
    )
    .await;
    let plan = &body["items"][0]["plan"];
    assert_eq!(plan["output"], json!({ "width": 1301, "height": 1010 }));
    assert_eq!(plan["colorFix"], json!(false));
    assert_eq!(plan["inputSizing"], json!("original"));

    // An over-bound original-mode request is a per-item refusal, never a silent budget fallback.
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "width": 4032, "height": 3024, "scale": 4, "inputSizing": "original" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["items"][0]["plan"], Value::Null);
    assert!(body["items"][0]["error"]
        .as_str()
        .unwrap()
        .contains("16128x12096"));
}

/// S2: a Studio prompt batch whose prompts carry their own `[WxH]` directives plans EVERY distinct
/// render size in one call, so the Run gate sees the size the worker would refuse (768x1344 at 8x
/// original sizing is 6144x10752 — over the per-side bound) instead of only the Studio default.
#[tokio::test]
async fn plan_preview_by_sizes_plans_every_batch_render_size() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({
            "sizes": [{ "width": 512, "height": 384 }, { "width": 768, "height": 1344 }],
            "scale": 8,
            "inputSizing": "original",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0]["plan"]["output"],
        json!({ "width": 4096, "height": 3072 })
    );
    assert_eq!(items[0]["error"], Value::Null);
    assert_eq!(items[1]["plan"], Value::Null);
    assert!(
        items[1]["error"].as_str().unwrap().contains("6144x10752"),
        "{body}"
    );
}

/// S9b: a trained restoration model that is not installed is the job's refusal on every item —
/// never a silent plan with the release geometry.
#[tokio::test]
async fn plan_preview_with_a_missing_trained_model_is_refused_per_item() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "width": 512, "height": 384, "irisModel": "absent-restore-run" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["items"][0]["plan"], Value::Null);
    let error = body["items"][0]["error"].as_str().unwrap();
    assert!(error.contains("absent-restore-run"), "{error}");
}

#[tokio::test]
async fn plan_preview_rejects_malformed_controls_and_ambiguous_sources() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    for body in [
        json!({ "width": 512, "height": 384, "scale": 0 }),
        json!({ "width": 512, "height": 384, "scale": "4" }),
        json!({ "width": 512, "height": 384, "inputSizing": "tiny" }),
        json!({ "width": 512, "height": 384, "colorFix": "yes" }),
        json!({ "width": 512 }),
        json!({}),
        json!({ "width": 512, "height": 384, "projectId": "p", "assetIds": ["a"] }),
        json!({ "assetIds": ["a"] }),
        json!({ "sizes": [{ "width": 512 }] }),
        json!({ "sizes": "512x384" }),
        json!({ "width": 512, "height": 384, "sizes": [{ "width": 512, "height": 384 }] }),
        json!({ "width": 512, "height": 384, "irisModel": "../escape" }),
    ] {
        let (status, response) = request(
            app.clone(),
            "POST",
            "/api/v1/image-restoration/plan",
            body.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} → {response}");
    }
}

/// Existing assets plan against their UPRIGHT size: a 90°-tagged 40x20 phone JPEG plans as 20x40.
#[tokio::test]
async fn plan_preview_by_asset_measures_the_exif_corrected_size() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let store =
        sceneworks_core::project_store::ProjectStore::new(&settings.data_dir, "test-version");
    let project = store.create_project("Restore preview").unwrap();
    let mut ids = Vec::new();
    for (name, bytes) in [
        ("phone.jpg", rotated_jpeg(40, 20)),
        ("plain.png", {
            let mut png = Vec::new();
            image::RgbImage::new(64, 32)
                .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
            png
        }),
    ] {
        let file = temp_dir.path().join(name);
        std::fs::write(&file, bytes).unwrap();
        let asset = store
            .import_asset(
                &project.id,
                sceneworks_core::project_store::UploadAsset {
                    filename: name.to_owned(),
                    content_type: None,
                    source_path: file,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .unwrap();
        ids.push(asset["id"].as_str().unwrap().to_owned());
    }
    let app = create_app(settings).expect("app creates");
    let (status, body) = request(
        app,
        "POST",
        "/api/v1/image-restoration/plan",
        json!({ "projectId": project.id, "assetIds": ids, "scale": 1 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let items = body["items"].as_array().unwrap();
    assert_eq!(items[0]["assetId"], json!(ids[0]));
    assert_eq!(
        items[0]["plan"]["source"],
        json!({ "width": 20, "height": 40 })
    );
    assert_eq!(
        items[1]["plan"]["source"],
        json!({ "width": 64, "height": 32 })
    );
}

/// The raw `image_upscale` job with `engine: "iris"` carries its own task entry (never the
/// generation model or its text encoder), and every control that is not Iris's is refused by name
/// at submission — as are the Iris-only keys on the other engines.
#[tokio::test]
async fn iris_upscale_jobs_carry_their_task_entry_and_refuse_foreign_controls() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (status, body) = request(
        app.clone(),
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "image_upscale",
            "requestedGpu": "auto",
            "payload": { "sourceAssetId": "asset-1", "engine": "iris", "scale": 1.5, "inputSizing": "original", "colorFix": false },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let stamped = body["payload"]["modelManifestEntries"]
        .as_array()
        .expect("carries its model entry")
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(stamped, vec!["iris_3b_restore"]);
    assert_eq!(body["payload"]["scale"], json!(1.5));

    for (payload, field) in [
        (json!({ "engine": "iris", "factor": 4 }), "factor"),
        (json!({ "engine": "iris", "softness": 0.5 }), "softness"),
        (json!({ "engine": "iris", "seed": 7 }), "seed"),
        (json!({ "engine": "iris", "scale": -1 }), "scale"),
        (
            json!({ "engine": "iris", "inputSizing": "huge" }),
            "inputSizing",
        ),
        (json!({ "engine": "seedvr2", "scale": 2 }), "scale"),
        (
            json!({ "engine": "real-esrgan", "colorFix": true }),
            "colorFix",
        ),
    ] {
        let mut payload = payload;
        payload["sourceAssetId"] = json!("asset-1");
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({ "type": "image_upscale", "requestedGpu": "auto", "payload": payload }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {body}");
        assert!(body.to_string().contains(field), "{field}: {body}");
    }
}

fn image_job(upscale: Value) -> crate::dto::ImageJobRequest {
    serde_json::from_value(json!({
        "projectId": "project-1",
        "model": "z_image_turbo",
        "prompt": "a lighthouse",
        "mode": "text_to_image",
        "upscale": upscale,
    }))
    .expect("image job request parses")
}

/// The Image Studio post-processing block: Iris is sized by `scale` (any positive value), a
/// non-default `factor` or a `softness` is refused rather than ignored, and the factor engines keep
/// their 2x/4x contract and refuse the Iris-only controls.
#[test]
fn studio_post_processing_validates_the_iris_controls() {
    for ok in [
        json!({ "enabled": true, "engine": "iris" }),
        json!({ "enabled": true, "engine": "iris", "scale": 1 }),
        json!({ "enabled": true, "engine": "iris", "scale": 3.7, "inputSizing": "original", "colorFix": false }),
        json!({ "enabled": true, "engine": "real-esrgan", "factor": 4 }),
    ] {
        assert!(
            crate::validate_image_job(&image_job(ok.clone())).is_ok(),
            "{ok}"
        );
    }
    for bad in [
        json!({ "enabled": true, "engine": "iris", "factor": 4 }),
        json!({ "enabled": true, "engine": "iris", "softness": 0.2 }),
        json!({ "enabled": true, "engine": "iris", "scale": 0 }),
        json!({ "enabled": true, "engine": "iris", "inputSizing": "x" }),
        json!({ "enabled": true, "engine": "seedvr2", "factor": 2, "scale": 2 }),
        json!({ "enabled": true, "engine": "real-esrgan", "factor": 3 }),
    ] {
        assert!(
            crate::validate_image_job(&image_job(bad.clone())).is_err(),
            "{bad}"
        );
    }
}
