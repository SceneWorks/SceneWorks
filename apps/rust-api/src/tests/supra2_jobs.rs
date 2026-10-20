//! sc-24215 (epic 24212): Supra2-IMG's published interface at the API — the shipped catalog row
//! accepts prompt / seed / guidance / steps / image count at 256x256 and refuses everything else
//! with a 400 BEFORE a job exists.
use super::support::*;

fn shipped_supra2_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "supra2_img"))
        .cloned()
        .expect("supra2_img is in the builtin catalog")
}

fn setup(temp: &tempfile::TempDir) {
    let config = temp.path().join("config/manifests");
    single_model_manifest(&config, "supra2_img", "SupraLabs/Supra2-IMG");
    std::fs::write(
        config.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [shipped_supra2_entry()] }).to_string(),
    )
    .unwrap();
}

#[tokio::test]
async fn supra2_accepts_the_published_controls_and_refuses_the_rest_before_enqueue() {
    let _hf = isolate_hf_cache();
    let temp = tempfile::tempdir().unwrap();
    setup(&temp);
    let app = create_app(test_settings(&temp)).unwrap();
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Supra2" }),
    )
    .await;
    let job = |extra: Value| {
        let mut body =
            json!({ "projectId": project["id"], "model": "supra2_img", "prompt": "a red fox" });
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        body
    };

    // A bare call renders the release defaults at the native size; every control is carried.
    let (status, bare) = request(app.clone(), "POST", "/api/v1/image/jobs", job(json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{bare}");
    assert_eq!(
        (
            &bare["payload"]["width"],
            &bare["payload"]["height"],
            &bare["payload"]["count"]
        ),
        (&json!(256), &json!(256), &json!(1)),
        "{bare}"
    );
    let (status, full) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        job(json!({
            "width": 256, "height": 256, "count": 8, "seed": 7,
            "advanced": { "steps": 100_000, "guidanceScale": 1.0 }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{full}");
    assert_eq!(full["payload"]["count"], 8, "{full}");
    assert_eq!(full["payload"]["seed"], 7, "{full}");
    // The declared ceiling (gen-core's sanity cap) is accepted as-is, never clamped to 80.
    assert_eq!(full["payload"]["advanced"]["steps"], 100_000, "{full}");
    assert_eq!(full["payload"]["advanced"]["guidanceScale"], 1.0, "{full}");

    for (extra, needle) in [
        (json!({ "width": 512, "height": 512 }), "256"),
        (json!({ "negativePrompt": "blurry" }), "negative prompt"),
        (
            json!({ "mode": "edit_image", "sourceAssetId": "asset_1" }),
            "text-to-image",
        ),
        (
            json!({ "referenceAssetIds": ["asset_1"] }),
            "referenceAssetIds",
        ),
        (json!({ "maskAssetId": "asset_1" }), "maskAssetId"),
        (json!({ "hiresFix": { "enabled": true } }), "hires fix"),
        (json!({ "advanced": { "steps": 100_001 } }), "100000"),
        (json!({ "advanced": { "steps": 0 } }), "steps"),
        (json!({ "advanced": { "sampler": "dpmpp_2m" } }), "sampler"),
        (json!({ "advanced": { "quantTier": "q4" } }), "quantTier"),
        (json!({ "advanced": { "poses": [{ "id": "p" }] } }), "poses"),
    ] {
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            job(extra.clone()),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {body}");
        assert!(body.to_string().contains(needle), "{extra}: {body}");
    }
    let (_, jobs) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
    assert_eq!(
        jobs.as_array().unwrap().len(),
        2,
        "only the two honourable jobs exist: {jobs}"
    );
}
