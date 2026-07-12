//! rust-api mcp tests (split from tests.rs, sc-11217 F-030).
use super::support::*;

// ---------------------------------------------------------------------------
// MCP mount (epic 10231, sc-10233): /mcp rides the same access_control gate as
// /api/v1, and a real MCP streamable-HTTP client can round-trip the catalog
// tools against the live app.
// ---------------------------------------------------------------------------

#[test]
fn requires_token_gates_the_mcp_mount_for_every_method() {
    use axum::http::Method;
    // The MCP mount is token-gated exactly like a gated /api/v1 route — for every
    // method the transport uses (POST messages, GET SSE stream, DELETE session).
    for method in [Method::GET, Method::POST, Method::DELETE] {
        assert!(
            requires_token(&method, "/mcp"),
            "{method} /mcp must be gated"
        );
        assert!(
            requires_token(&method, "/mcp/anything"),
            "{method} /mcp/* must be gated"
        );
    }
    // No prefix bleed: an unrelated path starting with "mcp" is still SPA fallback.
    assert!(!requires_token(&Method::GET, "/mcpx"));
}

#[tokio::test]
async fn mcp_rejects_unauthenticated_lan_request_exactly_like_api_v1() {
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");

    // Same peer, no token: /mcp and a gated /api/v1 route must answer identically.
    let (mcp_status, mcp_body) =
        request_with_peer(app.clone(), "POST", "/mcp", json!({}), "192.168.1.9:50000").await;
    let (api_status, api_body) =
        request_with_peer(app, "GET", "/api/v1/jobs", Value::Null, "192.168.1.9:50001").await;
    assert_eq!(mcp_status, StatusCode::UNAUTHORIZED);
    assert_eq!(api_status, StatusCode::UNAUTHORIZED);
    assert_eq!(
        mcp_body, api_body,
        "rejection body must match the /api/v1 shape"
    );
    assert_eq!(mcp_body["authRequired"], true);
}

#[tokio::test]
async fn mcp_passes_auth_with_valid_token_or_loopback_trust() {
    // A request that clears auth reaches the rmcp transport, which answers 406
    // ("must accept application/json and text/event-stream") for a bare POST —
    // distinct from the middleware's 401, so it proves the gate opened.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");

    // Valid header token from a LAN peer.
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    let app = create_app(settings).expect("app creates");
    let status = mcp_status_with_peer(
        app.clone(),
        "192.168.1.9:50000",
        &[("x-sceneworks-token", "secret-token")],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_ACCEPTABLE);

    // Loopback-trusted peer with no token (desktop mode, SCENEWORKS_TRUST_LOOPBACK).
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "secret-token".to_owned();
    settings.trust_loopback = true;
    let app = create_app(settings).expect("app creates");
    let status = mcp_status_with_peer(app.clone(), "127.0.0.1:50000", &[]).await;
    assert_eq!(status, StatusCode::NOT_ACCEPTABLE);
    // ... but loopback trust must not leak to a LAN peer.
    let status = mcp_status_with_peer(app, "192.168.1.9:50000", &[]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn mcp_client_round_trips_catalog_tools_via_loopback_trust() {
    use rmcp::model::CallToolRequestParams;
    use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
    use rmcp::transport::StreamableHttpClientTransport;
    use rmcp::ServiceExt;

    // Desktop-style deployment: loopback trusted, no token. The MCP self-client
    // (settings.mcp_api_url) points back at this same live listener.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral listener");
    let addr = listener.local_addr().expect("listener addr");
    let mut settings = test_settings(&temp_dir);
    settings.trust_loopback = true;
    settings.mcp_api_url = format!("http://{addr}");
    let (app, state) = create_app_with_state(settings).expect("app creates");
    state
        .project_store
        .create_project("MCP Round Trip")
        .expect("project creates");
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });

    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp")),
    );
    let client = rmcp::model::ClientInfo::default()
        .serve(transport)
        .await
        .expect("MCP client initializes against /mcp");

    // Tool discovery.
    let tools = client.list_tools(None).await.expect("tools/list succeeds");
    let names: Vec<&str> = tools.tools.iter().map(|tool| tool.name.as_ref()).collect();
    for expected in [
        "list_projects",
        "list_models",
        "list_loras",
        "generate_image",
    ] {
        assert!(
            names.contains(&expected),
            "missing tool {expected}: {names:?}"
        );
    }

    // list_projects round-trips through the real /api/v1/projects route.
    let result = client
        .call_tool(CallToolRequestParams::new("list_projects"))
        .await
        .expect("list_projects succeeds");
    assert_ne!(result.is_error, Some(true));
    let projects = mcp_tool_content_json(&result);
    let project_names: Vec<&str> = projects
        .as_array()
        .expect("projects array")
        .iter()
        .filter_map(|project| project["name"].as_str())
        .collect();
    assert!(
        project_names.contains(&"MCP Round Trip"),
        "created project must be listed: {project_names:?}"
    );

    // list_models round-trips through the real /api/v1/models route (empty temp
    // config dir → an array; the round trip is what's under test).
    let result = client
        .call_tool(CallToolRequestParams::new("list_models"))
        .await
        .expect("list_models succeeds");
    assert_ne!(result.is_error, Some(true));
    assert!(mcp_tool_content_json(&result).is_array());

    let _ = client.cancel().await;
}

#[tokio::test]
async fn mcp_client_round_trips_with_access_token_and_no_loopback_trust() {
    use rmcp::model::CallToolRequestParams;
    use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
    use rmcp::transport::StreamableHttpClientTransport;
    use rmcp::ServiceExt;
    use std::collections::HashMap;

    // LAN-style deployment: token required, loopback NOT trusted. The MCP client
    // must present the token on /mcp, and the MCP self-client must present it on
    // its /api/v1 calls (it gets settings.access_token) — both gates are real here.
    let temp_dir = tempfile::tempdir().expect("temp dir creates");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral listener");
    let addr = listener.local_addr().expect("listener addr");
    let mut settings = test_settings(&temp_dir);
    settings.access_token = "round-trip-token".to_owned();
    settings.mcp_api_url = format!("http://{addr}");
    let (app, state) = create_app_with_state(settings).expect("app creates");
    state
        .project_store
        .create_project("Tokened Round Trip")
        .expect("project creates");
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });

    let mut headers = HashMap::new();
    headers.insert(
        axum::http::HeaderName::from_static("x-sceneworks-token"),
        axum::http::HeaderValue::from_static("round-trip-token"),
    );
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}/mcp"))
            .custom_headers(headers),
    );
    let client = rmcp::model::ClientInfo::default()
        .serve(transport)
        .await
        .expect("tokened MCP client initializes against /mcp");

    let result = client
        .call_tool(CallToolRequestParams::new("list_projects"))
        .await
        .expect("list_projects succeeds");
    assert_ne!(result.is_error, Some(true));
    let projects = mcp_tool_content_json(&result);
    assert!(
        projects
            .as_array()
            .expect("projects array")
            .iter()
            .any(|project| project["name"] == "Tokened Round Trip"),
        "created project must be listed: {projects}"
    );

    let _ = client.cancel().await;
}
