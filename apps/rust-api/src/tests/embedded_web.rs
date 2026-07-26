use super::support::*;
use axum::http::header::{
    CACHE_CONTROL, CONTENT_ENCODING, CONTENT_SECURITY_POLICY, CONTENT_TYPE, ETAG, VARY,
};

#[tokio::test]
async fn embedded_spa_and_api_share_the_router() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");

    let (root_status, root_headers, root_body) =
        request_raw(app.clone(), "GET", "/", Body::empty(), &[]).await;
    assert_eq!(root_status, StatusCode::OK);
    assert_eq!(
        root_headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/html")
    );
    assert!(root_headers.contains_key(CONTENT_SECURITY_POLICY));
    assert!(
        String::from_utf8_lossy(&root_body).contains("<div id=\"root\">"),
        "root serves the built React entrypoint"
    );

    let (fallback_status, fallback_headers, fallback_body) = request_raw(
        app.clone(),
        "GET",
        "/projects/client-side-route",
        Body::empty(),
        &[],
    )
    .await;
    assert_eq!(fallback_status, StatusCode::OK);
    assert_eq!(
        fallback_headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    assert_eq!(fallback_body, root_body, "SPA fallback returns index.html");

    let asset_path = format!("/{}", crate::web_assets::first_static_asset_path());
    let (asset_status, asset_headers, asset_body) =
        request_raw(app.clone(), "GET", &asset_path, Body::empty(), &[]).await;
    assert_eq!(asset_status, StatusCode::OK);
    assert!(
        asset_headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value != "text/html; charset=utf-8"),
        "static asset keeps its own MIME type"
    );
    assert!(!asset_body.is_empty(), "static asset body is embedded");

    let (health_status, health) = request(app.clone(), "GET", "/api/v1/health", Value::Null).await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(health["service"], "sceneworks-api");

    let (missing_api_status, missing_api) =
        request(app, "GET", "/api/v1/not-a-route", Value::Null).await;
    assert_eq!(missing_api_status, StatusCode::NOT_FOUND);
    assert_eq!(missing_api["detail"], "Not Found");
}

#[tokio::test]
async fn embedded_assets_negotiate_precompressed_representations() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let asset_path = format!(
        "/{}",
        crate::web_assets::first_compressible_static_asset_path()
    );

    let (identity_status, identity_headers, identity_body) =
        request_raw(app.clone(), "GET", &asset_path, Body::empty(), &[]).await;
    assert_eq!(identity_status, StatusCode::OK);
    assert!(!identity_headers.contains_key(CONTENT_ENCODING));
    assert_eq!(
        identity_headers
            .get(VARY)
            .and_then(|value| value.to_str().ok()),
        Some("Accept-Encoding")
    );

    let (brotli_status, brotli_headers, brotli_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "gzip, br")],
    )
    .await;
    assert_eq!(brotli_status, StatusCode::OK);
    assert_eq!(
        brotli_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("br")
    );
    assert_eq!(
        brotli_headers.get(CONTENT_TYPE),
        identity_headers.get(CONTENT_TYPE)
    );
    assert!(
        brotli_body.len() < identity_body.len(),
        "Brotli transfer is smaller than the identity representation"
    );

    let (gzip_status, gzip_headers, gzip_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
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
    assert!(
        gzip_body.len() < identity_body.len(),
        "gzip transfer is smaller than the identity representation"
    );

    let (disabled_status, disabled_headers, disabled_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "br;q=0, gzip;q=0")],
    )
    .await;
    assert_eq!(disabled_status, StatusCode::OK);
    assert!(!disabled_headers.contains_key(CONTENT_ENCODING));
    assert_eq!(disabled_body, identity_body);

    let (unsupported_status, unsupported_headers, unsupported_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "zstd")],
    )
    .await;
    assert_eq!(unsupported_status, StatusCode::OK);
    assert!(!unsupported_headers.contains_key(CONTENT_ENCODING));
    assert_eq!(unsupported_body, identity_body);

    let (identity_preferred_status, identity_preferred_headers, identity_preferred_body) =
        request_raw(
            app.clone(),
            "GET",
            &asset_path,
            Body::empty(),
            &[("accept-encoding", "gzip;q=0.4")],
        )
        .await;
    assert_eq!(identity_preferred_status, StatusCode::OK);
    assert!(!identity_preferred_headers.contains_key(CONTENT_ENCODING));
    assert_eq!(identity_preferred_body, identity_body);

    let (identity_rejected_status, identity_rejected_headers, _) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "gzip;q=0.4, identity;q=0")],
    )
    .await;
    assert_eq!(identity_rejected_status, StatusCode::OK);
    assert_eq!(
        identity_rejected_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("gzip")
    );

    let (wildcard_status, wildcard_headers, _) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "br;q=0, *;q=0.7, identity;q=0.1")],
    )
    .await;
    assert_eq!(wildcard_status, StatusCode::OK);
    assert_eq!(
        wildcard_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("gzip")
    );

    for rejected in [
        "br;q=0, gzip;q=0, identity;q=0",
        "*;q=0",
        "zstd, identity;q=0",
    ] {
        let (status, headers, body) = request_raw(
            app.clone(),
            "GET",
            &asset_path,
            Body::empty(),
            &[("accept-encoding", rejected)],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::NOT_ACCEPTABLE,
            "{rejected} rejects every available representation"
        );
        assert!(body.is_empty());
        assert_eq!(
            headers.get(VARY).and_then(|value| value.to_str().ok()),
            Some("Accept-Encoding")
        );
    }

    let uncompressed_path = format!(
        "/{}",
        crate::web_assets::first_uncompressed_static_asset_path()
    );
    let (unavailable_status, _, unavailable_body) = request_raw(
        app,
        "GET",
        &uncompressed_path,
        Body::empty(),
        &[("accept-encoding", "gzip, identity;q=0")],
    )
    .await;
    assert_eq!(
        unavailable_status,
        StatusCode::NOT_ACCEPTABLE,
        "an advertised coding is not acceptable when that representation is unavailable"
    );
    assert!(unavailable_body.is_empty());
}

#[tokio::test]
async fn embedded_cache_contract_separates_hashed_and_mutable_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let asset_path = format!(
        "/{}",
        crate::web_assets::first_compressible_static_asset_path()
    );

    let (asset_status, asset_headers, asset_body) =
        request_raw(app.clone(), "GET", &asset_path, Body::empty(), &[]).await;
    assert_eq!(asset_status, StatusCode::OK);
    assert!(!asset_body.is_empty());
    assert_eq!(
        asset_headers
            .get(CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("public, max-age=31536000, immutable")
    );
    let asset_etag = asset_headers
        .get(ETAG)
        .and_then(|value| value.to_str().ok())
        .expect("hashed asset has an ETag")
        .to_owned();
    let (reused_status, reused_headers, reused_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("if-none-match", &asset_etag)],
    )
    .await;
    assert_eq!(reused_status, StatusCode::NOT_MODIFIED);
    assert!(reused_body.is_empty());
    assert_eq!(reused_headers.get(ETAG), asset_headers.get(ETAG));

    for path in ["/", "/index.html", "/theme-init.js", "/projects/deep-link"] {
        let (status, headers, body) =
            request_raw(app.clone(), "GET", path, Body::empty(), &[]).await;
        assert_eq!(status, StatusCode::OK, "{path} serves");
        assert!(!body.is_empty(), "{path} has a body");
        assert_eq!(
            headers
                .get(CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-cache"),
            "{path} must revalidate"
        );
        assert!(headers.contains_key(ETAG), "{path} has a validator");
        assert_eq!(
            headers.get(VARY).and_then(|value| value.to_str().ok()),
            Some("Accept-Encoding")
        );
    }
}

#[tokio::test]
async fn validators_are_representation_specific_and_spa_fallback_revalidates() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let asset_path = format!(
        "/{}",
        crate::web_assets::first_compressible_static_asset_path()
    );

    let (_, identity_headers, _) =
        request_raw(app.clone(), "GET", &asset_path, Body::empty(), &[]).await;
    let identity_etag = identity_headers
        .get(ETAG)
        .and_then(|value| value.to_str().ok())
        .expect("identity representation has an ETag");
    let (encoded_status, encoded_headers, encoded_body) = request_raw(
        app.clone(),
        "GET",
        &asset_path,
        Body::empty(),
        &[("accept-encoding", "br"), ("if-none-match", identity_etag)],
    )
    .await;
    assert_eq!(encoded_status, StatusCode::OK);
    assert!(!encoded_body.is_empty());
    assert_ne!(encoded_headers.get(ETAG), identity_headers.get(ETAG));

    let (_, fallback_headers, fallback_body) = request_raw(
        app.clone(),
        "GET",
        "/projects/client-route",
        Body::empty(),
        &[("accept-encoding", "gzip")],
    )
    .await;
    let fallback_etag = fallback_headers
        .get(ETAG)
        .and_then(|value| value.to_str().ok())
        .expect("SPA fallback has an ETag");
    assert_eq!(
        fallback_headers
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
        Some("gzip")
    );
    assert!(!fallback_body.is_empty());
    let (revalidated_status, revalidated_headers, revalidated_body) = request_raw(
        app,
        "GET",
        "/projects/client-route",
        Body::empty(),
        &[
            ("accept-encoding", "gzip"),
            ("if-none-match", fallback_etag),
        ],
    )
    .await;
    assert_eq!(revalidated_status, StatusCode::NOT_MODIFIED);
    assert!(revalidated_body.is_empty());
    assert_eq!(
        revalidated_headers
            .get(CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-cache")
    );
}

#[tokio::test]
async fn internal_precompressed_siblings_and_cache_manifest_are_not_public_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let app = create_app(test_settings(&temp_dir)).expect("app creates");
    let asset_path = crate::web_assets::first_compressible_static_asset_path();
    let (_, _, root_body) = request_raw(app.clone(), "GET", "/", Body::empty(), &[]).await;

    for internal_path in [
        format!("/{asset_path}.br"),
        format!("/{asset_path}.gz"),
        format!("/{}", crate::web_assets::immutable_asset_manifest_path()),
    ] {
        let (status, headers, body) =
            request_raw(app.clone(), "GET", &internal_path, Body::empty(), &[]).await;
        assert_eq!(status, StatusCode::OK, "{internal_path} uses SPA fallback");
        assert_eq!(
            headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/html; charset=utf-8")
        );
        assert!(!headers.contains_key(CONTENT_ENCODING));
        assert_eq!(
            headers
                .get(CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-cache")
        );
        assert_eq!(
            body, root_body,
            "{internal_path} must not expose build representation bytes"
        );
    }
}
