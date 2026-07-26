//! Dynamic API response-compression policy (SC-14798).

use super::support::*;
use axum::http::header::{ACCEPT_RANGES, CONTENT_ENCODING, CONTENT_RANGE, CONTENT_TYPE, VARY};

fn vary_contains(headers: &axum::http::HeaderMap, token: &str) -> bool {
    headers
        .get_all(VARY)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|value| value.trim().eq_ignore_ascii_case(token))
}

fn write_real_catalog(config_dir: &std::path::Path) {
    std::fs::create_dir_all(config_dir).expect("manifest directory creates");
    std::fs::write(
        config_dir.join("builtin.models.jsonc"),
        include_str!("../../../../config/manifests/builtin.models.jsonc"),
    )
    .expect("builtin model catalog writes");
    write_empty_sibling_manifests(config_dir);
}

#[tokio::test]
async fn large_json_negotiates_brotli_gzip_and_identity() {
    let _env = isolate_hf_cache();
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    write_real_catalog(&temp_dir.path().join("config/manifests"));
    let (app, state) =
        create_app_with_state(test_settings(&temp_dir)).expect("app and state create");
    *state.model_size_estimate_disabled_override.lock() = Some(true);

    let (identity_status, identity_headers, identity_body) =
        request_raw(app.clone(), "GET", "/api/v1/models", Body::empty(), &[]).await;
    assert_eq!(identity_status, StatusCode::OK);
    assert_eq!(
        identity_headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert!(!identity_headers.contains_key(CONTENT_ENCODING));
    assert!(
        identity_body.len() >= usize::from(crate::API_COMPRESSION_MIN_BYTES),
        "the real catalog must exercise the configured size threshold"
    );
    serde_json::from_slice::<Value>(&identity_body).expect("identity catalog is JSON");

    let (brotli_status, brotli_headers, brotli_body) = request_raw(
        app.clone(),
        "GET",
        "/api/v1/models",
        Body::empty(),
        &[("accept-encoding", "gzip, br")],
    )
    .await;
    assert_eq!(brotli_status, StatusCode::OK);
    assert_eq!(
        brotli_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("br"),
        "Brotli wins an equal-quality tie, matching the embedded-web policy"
    );
    assert!(vary_contains(&brotli_headers, "Accept-Encoding"));
    assert!(brotli_body.len() < identity_body.len());

    let (gzip_status, gzip_headers, gzip_body) = request_raw(
        app.clone(),
        "GET",
        "/api/v1/models",
        Body::empty(),
        &[("accept-encoding", "br;q=0.4, gzip;q=1")],
    )
    .await;
    assert_eq!(gzip_status, StatusCode::OK);
    assert_eq!(
        gzip_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("gzip")
    );
    assert!(vary_contains(&gzip_headers, "Accept-Encoding"));
    assert_eq!(
        gzip_body.get(..2),
        Some([0x1f, 0x8b].as_slice()),
        "gzip response has the standard stream signature"
    );
    assert!(gzip_body.len() < identity_body.len());

    let (unsupported_status, unsupported_headers, unsupported_body) = request_raw(
        app,
        "GET",
        "/api/v1/models",
        Body::empty(),
        &[("accept-encoding", "zstd")],
    )
    .await;
    assert_eq!(unsupported_status, StatusCode::OK);
    assert!(!unsupported_headers.contains_key(CONTENT_ENCODING));
    assert!(vary_contains(&unsupported_headers, "Accept-Encoding"));
    let identity_catalog =
        serde_json::from_slice::<Value>(&identity_body).expect("identity catalog parses");
    let unsupported_catalog =
        serde_json::from_slice::<Value>(&unsupported_body).expect("fallback catalog parses");
    assert_eq!(
        unsupported_catalog.as_array().map(Vec::len),
        identity_catalog.as_array().map(Vec::len),
        "an unsupported coding safely returns the complete identity catalog"
    );
}

#[tokio::test]
async fn tiny_json_stays_identity_below_the_cpu_threshold() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (status, headers, body) = request_raw(
        app,
        "GET",
        "/api/v1/access",
        Body::empty(),
        &[("accept-encoding", "br, gzip")],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.len() < usize::from(crate::API_COMPRESSION_MIN_BYTES));
    assert!(!headers.contains_key(CONTENT_ENCODING));
    assert!(
        !vary_contains(&headers, "Accept-Encoding"),
        "tiny responses bypass the compressor rather than paying per-request CPU"
    );
    serde_json::from_slice::<Value>(&body).expect("tiny identity response is JSON");
}

#[tokio::test]
async fn media_and_ranges_are_never_recompressed() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Compression media exclusion" }),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let media = b"already-compressed-video-payload";
    let (upload_status, asset) = request_multipart_upload(
        app.clone(),
        &format!("/api/v1/projects/{project_id}/assets"),
        "clip.mp4",
        "video/mp4",
        media,
    )
    .await;
    assert_eq!(upload_status, StatusCode::CREATED);
    let url = asset["url"].as_str().expect("asset URL");

    let (full_status, full_headers, full_body) = request_raw(
        app.clone(),
        "GET",
        url,
        Body::empty(),
        &[("accept-encoding", "br, gzip")],
    )
    .await;
    assert_eq!(full_status, StatusCode::OK);
    assert_eq!(full_body, media);
    assert!(!full_headers.contains_key(CONTENT_ENCODING));
    assert_eq!(
        full_headers
            .get(ACCEPT_RANGES)
            .and_then(|value| value.to_str().ok()),
        Some("bytes")
    );

    let (range_status, range_headers, range_body) = request_raw(
        app,
        "GET",
        url,
        Body::empty(),
        &[("accept-encoding", "br, gzip"), ("range", "bytes=2-8")],
    )
    .await;
    assert_eq!(range_status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(range_body, &media[2..=8]);
    assert!(!range_headers.contains_key(CONTENT_ENCODING));
    let expected_content_range = format!("bytes 2-8/{}", media.len());
    assert_eq!(
        range_headers
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok()),
        Some(expected_content_range.as_str())
    );
}

#[tokio::test]
async fn sse_is_uncompressed_and_first_event_is_not_buffered() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let request = Request::builder()
        .method("GET")
        .uri("/api/v1/jobs/events")
        .header("accept-encoding", "br, gzip")
        .body(Body::empty())
        .expect("request builds");
    let response = app.oneshot(request).await.expect("response returns");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    assert!(!response.headers().contains_key(CONTENT_ENCODING));

    let mut stream = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .expect("the ready event is not buffered by compression")
        .expect("SSE stream remains open")
        .expect("SSE frame reads");
    let first = std::str::from_utf8(&first).expect("SSE is UTF-8");
    assert!(first.contains("event: ready"));
    assert!(first.contains("data: {\"status\":\"connected\"}"));
}

/// Manual, exact-route performance evidence for SC-14798. It is ignored in the
/// ordinary gate because seeding hundreds of SQLite-backed assets is deliberate
/// benchmark setup, not a correctness requirement. Run with:
///
/// `cargo test -p sceneworks-rust-api large_assets_response_compression_timing -- --ignored --nocapture`
#[tokio::test]
#[ignore = "manual ~2 MB API compression timing"]
async fn large_assets_response_compression_timing() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let settings = test_settings(&temp_dir);
    let store =
        sceneworks_core::project_store::ProjectStore::new(&settings.data_dir, "test-version");
    let project = store.create_project("Large compression timing").unwrap();
    let generation_set_id = "compression_timing_set";
    let repeated_prompt = "cinematic portrait with soft practical lighting ".repeat(90);
    store
        .write_generation_set(
            &project.id,
            "compression-timing-job",
            &json!({
                "id": generation_set_id,
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": repeated_prompt,
                "negativePrompt": "",
                "count": 420,
                "createdAt": "2026-07-26T00:00:00Z",
            }),
            None,
        )
        .expect("generation set writes");
    let media_dir = std::path::Path::new(&project.path)
        .join("assets/images")
        .join(generation_set_id);
    std::fs::create_dir_all(&media_dir).expect("media directory creates");
    for index in 0..420 {
        let media_path = format!("assets/images/{generation_set_id}/{index:04}.png");
        std::fs::write(media_dir.join(format!("{index:04}.png")), b"png")
            .expect("representative media file writes");
        store
            .persist_generated_asset(
                &project.id,
                "compression-timing-job",
                generation_set_id,
                &json!({
                    "assetId": format!("compression_asset_{index:04}"),
                    "mediaPath": media_path,
                    "mimeType": "image/png",
                    "displayName": format!("Compression sample {index:04}"),
                    "createdAt": "2026-07-26T00:00:00Z",
                    "mode": "text_to_image",
                    "model": "z_image_turbo",
                    "adapter": "z_image_diffusers",
                    "prompt": "cinematic portrait",
                }),
            )
            .expect("asset persists");
    }

    let app = create_app(settings).expect("app creates");
    let uri = format!("/api/v1/projects/{}/assets", project.id);
    // Warm the SQLite/index and filesystem caches before recording either
    // representation so the comparison isolates response delivery.
    let _ = request_raw(app.clone(), "GET", &uri, Body::empty(), &[]).await;

    let identity_started = std::time::Instant::now();
    let (identity_status, _, identity_body) =
        request_raw(app.clone(), "GET", &uri, Body::empty(), &[]).await;
    let identity_elapsed = identity_started.elapsed();
    assert_eq!(identity_status, StatusCode::OK);
    assert!(
        (1_500_000..=3_500_000).contains(&identity_body.len()),
        "fixture must stay representative of the measured ~2 MB assets response; got {} bytes",
        identity_body.len()
    );

    let gzip_started = std::time::Instant::now();
    let (gzip_status, gzip_headers, gzip_body) = tokio::time::timeout(
        Duration::from_secs(10),
        request_raw(
            app.clone(),
            "GET",
            &uri,
            Body::empty(),
            &[("accept-encoding", "gzip")],
        ),
    )
    .await
    .expect("gzip delivery stays bounded");
    let gzip_elapsed = gzip_started.elapsed();
    assert_eq!(gzip_status, StatusCode::OK);
    assert_eq!(gzip_headers[CONTENT_ENCODING], "gzip");
    assert!(gzip_body.len() < identity_body.len());

    let brotli_started = std::time::Instant::now();
    let (brotli_status, brotli_headers, brotli_body) = tokio::time::timeout(
        Duration::from_secs(10),
        request_raw(
            app,
            "GET",
            &uri,
            Body::empty(),
            &[("accept-encoding", "br")],
        ),
    )
    .await
    .expect("Brotli delivery stays bounded");
    let brotli_elapsed = brotli_started.elapsed();
    assert_eq!(brotli_status, StatusCode::OK);
    assert_eq!(brotli_headers[CONTENT_ENCODING], "br");
    assert!(brotli_body.len() < gzip_body.len());

    eprintln!(
        "SC-14798 ~2 MB /assets timing: identity={} bytes/{identity_elapsed:?}, gzip={} bytes/{gzip_elapsed:?}, br={} bytes/{brotli_elapsed:?}",
        identity_body.len(),
        gzip_body.len(),
        brotli_body.len(),
    );
}
