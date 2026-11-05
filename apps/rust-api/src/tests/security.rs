//! SW-F327 F-001/002/011/012/098/099/152/156: CPU-only boundary regressions.
use super::support::*;
use std::net::SocketAddr;

#[tokio::test]
async fn browser_trust_precedes_loopback_token_and_public_bypasses() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = test_settings(&temp);
    settings.host = "0.0.0.0".into();
    settings.access_token = "secret".into();
    settings.trust_loopback = true;
    settings.mcp_allowed_hosts_extra = vec![
        "studio.local:8000".into(),
        "192.168.4.97:8000".into(),
        "api".into(),
    ];
    settings.mcp_api_url = "http://configured.local:8000".into();
    settings.cors_origins.push("https://app.example".into());
    let app = create_app(settings).unwrap();
    for path in ["/api/v1/jobs", "/api/v1/health", "/mcp"] {
        for (host, origin) in [
            ("attacker.example:8000", "http://attacker.example:8000"),
            ("localhost:8000", "https://attacker.example"),
            ("localhost:8000", "null"),
            ("localhost:8000", "http://localhost:8001"),
            ("localhost:8000", "http://localhost:8000/evil"),
        ] {
            let (status, _) = request_with_peer_headers(
                app.clone(),
                "GET",
                path,
                Value::Null,
                "127.0.0.1:50000",
                &[
                    ("host", host),
                    ("origin", origin),
                    ("x-sceneworks-token", "secret"),
                ],
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{path} {host} {origin}");
        }
    }
    for (host, origin, peer, token) in [
        (
            "localhost:8000",
            "http://localhost:8000",
            "127.0.0.1:50000",
            "",
        ),
        ("[::1]:8000", "http://[::1]:8000", "[::1]:50000", ""),
        (
            "studio.local:8000",
            "http://studio.local:8000",
            "192.168.4.2:50000",
            "secret",
        ),
        (
            "192.168.4.97:8000",
            "http://192.168.4.97:8000",
            "192.168.4.2:50000",
            "secret",
        ),
        (
            "configured.local:8000",
            "http://configured.local:8000",
            "192.168.4.2:50000",
            "secret",
        ),
        (
            "localhost:8000",
            "https://app.example",
            "127.0.0.1:50000",
            "",
        ),
    ] {
        let (status, body) = request_with_peer_headers(
            app.clone(),
            "GET",
            "/api/v1/jobs",
            Value::Null,
            peer,
            &[
                ("host", host),
                ("origin", origin),
                ("x-sceneworks-token", token),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{host} {origin} {body}");
    }
    let (status, _) = request_with_peer_headers(
        app.clone(),
        "GET",
        "/api/v1/jobs",
        Value::Null,
        "192.168.4.2:50000",
        &[("host", "api:8010"), ("x-sceneworks-token", "secret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "non-browser token client");
    for headers in [
        vec![("host", "localhost:8000"), ("host", "attacker.example")],
        vec![
            ("host", "localhost:8000"),
            ("origin", "http://localhost:8000"),
            ("origin", "http://attacker.example"),
        ],
        vec![("host", "user@localhost:8000")],
        vec![("host", "localhost:bad")],
    ] {
        let (status, _, _) = request_raw(
            app.clone(),
            "GET",
            "/api/v1/health",
            Body::empty(),
            &headers,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{headers:?}");
    }
    let (status, _, _) = request_raw(
        app,
        "OPTIONS",
        "/api/v1/jobs",
        Body::empty(),
        &[
            ("host", "localhost:8000"),
            ("origin", "http://attacker.example"),
            ("access-control-request-method", "POST"),
        ],
    )
    .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "preflight cannot bypass browser trust"
    );
}

#[tokio::test]
async fn optional_json_bodies_reject_browser_simple_content_types() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    for path in ["/api/v1/jobs/events/ticket", "/api/v1/jobs/missing/retry"] {
        for content_type in [
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ] {
            let (status, _, _) = request_raw(
                app.clone(),
                "POST",
                path,
                "{}",
                &[("content-type", content_type)],
            )
            .await;
            assert_eq!(
                status,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "{path} {content_type}"
            );
        }
        let (status, _, _) = request_raw(app.clone(), "POST", path, "{}", &[]).await;
        assert_eq!(
            status,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "{path} missing type"
        );
        let (status, _, _) = request_raw(
            app.clone(),
            "POST",
            path,
            "{}",
            &[("content-type", "application/json")],
        )
        .await;
        assert_eq!(
            status,
            if path.ends_with("retry") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::OK
            },
            "{path}"
        );
    }
    assert_eq!(
        request_raw(
            app,
            "POST",
            "/api/v1/jobs/events/ticket",
            Body::empty(),
            &[]
        )
        .await
        .0,
        StatusCode::OK
    );
}

#[tokio::test]
async fn remote_and_unknown_peers_cannot_select_host_paths_even_with_token() {
    let temp = tempfile::tempdir().unwrap();
    let mut settings = test_settings(&temp);
    settings.access_token = "secret".into();
    let app = create_app(settings).unwrap();
    let new_root = temp.path().join("new-catalog");
    for (path, payload) in [
        (
            "/api/v1/catalogs",
            json!({"path": new_root, "name": "Local"}),
        ),
        (
            "/api/v1/catalogs/attach",
            json!({"path": temp.path().join("nonexistent")}),
        ),
        (
            "/api/v1/projects/missing/training/datasets/missing/parquet-import-jobs",
            json!({"sourcePath": temp.path().join("private")}),
        ),
    ] {
        let (status, _) = request_with_peer_headers(
            app.clone(),
            "POST",
            path,
            payload.clone(),
            "192.168.4.2:50000",
            &[("x-sceneworks-token", "secret")],
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
        let (status, _) = request_with_headers(
            app.clone(),
            "POST",
            path,
            payload,
            &[("x-sceneworks-token", "secret")],
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "unknown peer {path}");
    }
    assert!(
        !new_root.exists(),
        "rejected selection cannot create a directory"
    );
    let (status, created) = request_with_peer_headers(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({"path": new_root, "name": "Local"}),
        "[::1]:50000",
        &[("x-sceneworks-token", "secret")],
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let (status, attached) = request_with_peer_headers(
        app,
        "POST",
        "/api/v1/catalogs/attach",
        json!({"path": new_root}),
        "127.0.0.1:50000",
        &[("x-sceneworks-token", "secret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{attached}");
}

#[test]
fn source_path_errors_do_not_disclose_existence_or_paths() {
    let temp = tempfile::tempdir().unwrap();
    let approved = temp.path().join("approved");
    std::fs::create_dir(&approved).unwrap();
    let valid = approved.join("model.safetensors");
    let outside = temp.path().join("private.safetensors");
    std::fs::write(&valid, b"fixture").unwrap();
    std::fs::write(&outside, b"secret").unwrap();
    assert!(crate::validate_import_source_path(
        valid.to_str().unwrap(),
        std::slice::from_ref(&approved)
    )
    .is_ok());
    assert!(crate::validate_import_source_path(
        approved.to_str().unwrap(),
        std::slice::from_ref(&approved)
    )
    .is_ok());
    let absent = temp.path().join("absent");
    for source in [
        "relative",
        outside.to_str().unwrap(),
        absent.to_str().unwrap(),
    ] {
        let error = crate::validate_import_source_path(source, std::slice::from_ref(&approved))
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.detail, "sourcePath must be inside an approved folder");
    }
    #[cfg(unix)]
    {
        let link = approved.join("escape.safetensors");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        assert_eq!(
            crate::validate_import_source_path(link.to_str().unwrap(), &[approved])
                .unwrap_err()
                .detail,
            "sourcePath must be inside an approved folder"
        );
    }
}

#[tokio::test]
async fn internal_errors_are_generic_correlated_and_typed_rejections_remain_actionable() {
    use axum::response::IntoResponse;
    for error in [
        crate::ApiError::internal("/Users/private/secret: SQLite missing table"),
        crate::ApiError::typed(
            StatusCode::INTERNAL_SERVER_ERROR,
            "private DB failure",
            "internal_fixture",
            json!({"path":"/private/database.sqlite"}),
        ),
        crate::ApiError::from(sceneworks_core::jobs_store::JobsStoreError::Sqlite(
            rusqlite::Error::InvalidQuery,
        )),
    ] {
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["detail"], "An internal error occurred");
        assert_eq!(body["context"], Value::Null);
        assert!(uuid::Uuid::parse_str(body["correlationId"].as_str().unwrap()).is_ok());
        assert!(!String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains("private"));
    }
    for (code, summary) in [
        (
            crate::workflows::ASSET_WORKFLOW_CODE_UNREADABLE,
            crate::workflows::ASSET_WORKFLOW_UNREADABLE_DETAIL,
        ),
        (
            crate::workflows::INSPECT_CODE_READ_FAILED,
            crate::workflows::INSPECT_READ_FAILED_DETAIL,
        ),
    ] {
        let error = crate::ApiError::typed(
            StatusCode::INTERNAL_SERVER_ERROR,
            "/private/diagnostic.sqlite: raw error",
            code,
            json!({"path":"/private/diagnostic.sqlite"}),
        );
        let bytes = to_bytes(error.into_response().into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["detail"], summary);
        assert_eq!(body["context"], Value::Null);
        assert!(uuid::Uuid::parse_str(body["correlationId"].as_str().unwrap()).is_ok());
        assert!(!String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains("/private"));
    }
    let error = crate::ApiError::typed(
        StatusCode::CONFLICT,
        "Reconnect the model library",
        "library_unavailable",
        json!({"reason":"disconnected"}),
    );
    let bytes = to_bytes(error.into_response().into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        body,
        json!({"detail":"Reconnect the model library", "code":"library_unavailable", "context":{"reason":"disconnected"}})
    );
}

#[tokio::test]
async fn all_migrated_json_extractors_preserve_the_error_contract() {
    let temp = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp)).unwrap();
    for (method, path) in [
        ("POST", "/api/v1/model-library/relocate"),
        ("POST", "/api/v1/models/library-roots"),
        ("PATCH", "/api/v1/models/library-roots/root"),
        ("POST", "/api/v1/models/library-roots/root/rescan"),
        ("POST", "/api/v1/model-cache/removal-preview"),
        ("POST", "/api/v1/model-cache/remove"),
        ("POST", "/api/v1/model-cache/pin"),
        ("POST", "/api/v1/training/iris/plan-check"),
    ] {
        let (status, _, bytes) = request_raw(
            app.clone(),
            method,
            path,
            "{",
            &[("content-type", "application/json")],
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{method} {path}");
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["detail"][0]["type"], "json_invalid", "{path} {body}");
        assert_eq!(body["detail"][0]["msg"], "JSON decode error");
    }
}

#[cfg(unix)]
#[tokio::test]
async fn asset_resolution_reuses_canonical_project_confinement() {
    let temp = tempfile::tempdir().unwrap();
    let (_, state) = create_app_with_state(test_settings(&temp)).unwrap();
    let project = state
        .project_store
        .create_project("Confined assets")
        .unwrap();
    let root = std::path::PathBuf::from(&project.path);
    let media = root.join("assets/images/safe.png");
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, PNG_32X32).unwrap();
    state.project_store.persist_generated_asset(&project.id, "job", "set", &json!({
        "assetId":"safe", "mediaPath":"assets/images/safe.png", "mimeType":"image/png", "type":"image", "displayName":"Safe", "createdAt":"2026-11-03T00:00:00Z", "mode":"text_to_image", "model":"fixture", "adapter":"fixture", "prompt":"fixture"
    })).unwrap();
    let resolved =
        crate::resolve_project_confined_asset_path(state.clone(), &project.id, "safe", &root)
            .await
            .unwrap();
    assert_eq!(resolved, std::fs::canonicalize(&media).unwrap());
    let outside = temp.path().join("private.png");
    std::fs::write(&outside, PNG_32X32).unwrap();
    std::fs::remove_file(&media).unwrap();
    std::os::unix::fs::symlink(outside, &media).unwrap();
    let error = crate::resolve_project_confined_asset_path(state, &project.id, "safe", &root)
        .await
        .unwrap_err();
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert_eq!(error.detail, "Invalid project file path");
}

#[test]
fn shared_helpers_preserve_slug_defaults_and_fail_closed_local_policy() {
    for (input, expected) in [
        (" My New LoRA! ", "my_new_lora"),
        ("_a__B-", "a_b"),
        ("!!!", "lora"),
    ] {
        assert_eq!(crate::slugify_lora_id(input), expected);
    }
    assert_eq!(crate::recipe_presets::slugify_preset_id("!!!"), "preset");
    assert_eq!(crate::recipe_presets::slugify_preset_id("lora"), "preset");
    for peer in [
        None,
        Some("192.168.1.1:1234".parse::<SocketAddr>().unwrap()),
        Some("[::ffff:192.168.1.1]:1234".parse().unwrap()),
        Some("[::ffff:0.0.0.0]:1234".parse().unwrap()),
        Some("[::]:1234".parse().unwrap()),
    ] {
        assert_eq!(
            crate::auth::local_only(peer).unwrap_err().status,
            StatusCode::FORBIDDEN
        );
    }
    assert!(crate::auth::local_only(Some("[::1]:1234".parse().unwrap())).is_ok());
    assert!(crate::auth::local_only(Some("127.0.0.1:1234".parse().unwrap())).is_ok());
    let mapped_local = Some("[::ffff:127.0.0.1]:1234".parse().unwrap());
    assert!(crate::auth::local_only(mapped_local).is_ok());
    crate::auth::require_local_peer(mapped_local, || panic!("a local peer cannot be refused"))
        .unwrap();
    let error =
        crate::auth::require_local_peer(Some("[::ffff:192.168.1.1]:1234".parse().unwrap()), || {
            crate::ApiError::bad_request("caller-owned local-path refusal")
        })
        .unwrap_err();
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert_eq!(error.detail, "caller-owned local-path refusal");
    for value in [
        json!({}),
        json!({"gated":true}),
        json!({"requiresLicenseAcknowledgment":false}),
    ] {
        assert!(!crate::models::model_requires_license_acknowledgment(
            &value
        ));
    }
    assert!(crate::models::model_requires_license_acknowledgment(
        &json!({"requiresLicenseAcknowledgment":true})
    ));
}
