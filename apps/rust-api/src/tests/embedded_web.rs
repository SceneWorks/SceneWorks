use super::support::*;
use axum::http::header::{CONTENT_SECURITY_POLICY, CONTENT_TYPE};

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
