//! MCP round-trip tests (sc-10233): a REAL rmcp streamable-HTTP client speaks to
//! the mounted `/mcp` service (initialize → tools/list → tools/call), with the
//! tool handlers' [`ApiClient`] pointed at a stub `/api/v1` axum server on an
//! ephemeral port. Proves the whole chain the acceptance criteria describe —
//! session handshake, tool discovery, catalog calls, the `X-SceneWorks-Token`
//! header on the upstream API call, and error surfacing on an upstream 401.

mod common;

use axum::extract::Query;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::get;
use axum::{Json, Router};
use rmcp::model::CallToolRequestParams;
use sceneworks_mcp::JobWaitConfig;
use serde_json::{json, Value};

use common::{call_args, connect_mcp, result_json, spawn, TestClient};

const STUB_TOKEN: &str = "test-secret-token";

/// Stub of the three `/api/v1` catalog routes. Requires `X-SceneWorks-Token:
/// test-secret-token` exactly like the real `access_control` gate, and echoes the
/// `list_loras` query filters into the fixture so the test can assert they were
/// forwarded.
fn stub_api_router() -> Router {
    fn authorized(headers: &HeaderMap) -> bool {
        headers
            .get("x-sceneworks-token")
            .and_then(|value| value.to_str().ok())
            == Some(STUB_TOKEN)
    }

    Router::new()
        .route(
            "/api/v1/projects",
            get(|headers: HeaderMap| async move {
                if !authorized(&headers) {
                    return Err(StatusCode::UNAUTHORIZED);
                }
                Ok(Json(json!([{
                    "id": "p1",
                    "name": "My Film",
                    "path": "/data/projects/p1",
                    "createdAt": "2026-07-07T00:00:00Z"
                }])))
            }),
        )
        .route(
            "/api/v1/models",
            get(|headers: HeaderMap| async move {
                if !authorized(&headers) {
                    return Err(StatusCode::UNAUTHORIZED);
                }
                Ok(Json(json!([{
                    "id": "z_image_turbo",
                    "name": "Z-Image-Turbo",
                    "family": "z-image",
                    "type": "image",
                    "installState": "installed",
                    "defaults": { "resolution": "1024x1024", "steps": 8 },
                    "limits": { "resolutions": ["1024x1024"], "samplers": ["default"] },
                    "downloads": [{ "repo": "SceneWorks/z-image-turbo-mlx" }]
                }])))
            }),
        )
        .route(
            "/api/v1/loras",
            get(
                |headers: HeaderMap, Query(query): Query<Value>| async move {
                    if !authorized(&headers) {
                        return Err(StatusCode::UNAUTHORIZED);
                    }
                    let family = query
                        .get("modelFamily")
                        .and_then(Value::as_str)
                        .unwrap_or("none")
                        .to_owned();
                    Ok(Json(json!([{
                        "id": format!("lora-for-{family}"),
                        "name": "Fixture LoRA",
                        "family": family,
                        "compatibility": { "families": [family] },
                        "source": { "provider": "huggingface", "repo": "x/y" }
                    }])))
                },
            ),
        )
}

/// Spin up stub API + mounted MCP service, return a connected MCP client.
async fn connect_client(
    access_token: Option<String>,
) -> rmcp::service::RunningService<rmcp::RoleClient, TestClient> {
    let api_base = spawn(stub_api_router()).await;
    connect_mcp(api_base, access_token, JobWaitConfig::default(), TestClient)
        .await
        .1
}

#[tokio::test]
async fn mcp_client_lists_tools_and_calls_catalog_tools() {
    let client = connect_client(Some(STUB_TOKEN.to_owned())).await;

    // Tool discovery: the three catalog tools are advertised.
    let tools = client
        .list_tools(Default::default())
        .await
        .expect("tools/list succeeds");
    let names: Vec<&str> = tools.tools.iter().map(|tool| tool.name.as_ref()).collect();
    for expected in ["list_projects", "list_models", "list_loras"] {
        assert!(
            names.contains(&expected),
            "missing tool {expected}: {names:?}"
        );
    }

    // list_projects → compact rows (id/name/createdAt only — no path).
    let result = client
        .call_tool(CallToolRequestParams::new("list_projects"))
        .await
        .expect("list_projects succeeds");
    assert_ne!(result.is_error, Some(true));
    assert_eq!(
        result_json(&result),
        json!([{ "id": "p1", "name": "My Film", "createdAt": "2026-07-07T00:00:00Z" }])
    );

    // list_models → compact rows with the job-request fields, verbose keys dropped.
    let result = client
        .call_tool(CallToolRequestParams::new("list_models"))
        .await
        .expect("list_models succeeds");
    assert_ne!(result.is_error, Some(true));
    let models = result_json(&result);
    assert_eq!(models[0]["id"], "z_image_turbo");
    assert_eq!(models[0]["defaults"]["steps"], 8);
    assert_eq!(models[0]["resolutions"], json!(["1024x1024"]));
    assert!(
        models[0].get("downloads").is_none(),
        "verbose keys must be dropped"
    );

    // list_loras forwards the modelFamily filter to the API's query string.
    let result = client
        .call_tool(
            CallToolRequestParams::new("list_loras")
                .with_arguments(call_args(json!({ "modelFamily": "sdxl" }))),
        )
        .await
        .expect("list_loras succeeds");
    assert_ne!(result.is_error, Some(true));
    let loras = result_json(&result);
    assert_eq!(loras[0]["id"], "lora-for-sdxl");
    assert_eq!(loras[0]["compatibleFamilies"], json!(["sdxl"]));

    let _ = client.cancel().await;
}

#[tokio::test]
async fn upstream_api_rejection_surfaces_as_tool_error_not_success() {
    // No token configured on the MCP side → the stub API 401s → the tool call must
    // fail loudly (JSON-RPC error), never return an empty-but-"successful" list.
    let client = connect_client(None).await;

    let outcome = client
        .call_tool(CallToolRequestParams::new("list_projects"))
        .await;
    match outcome {
        Err(_) => {}
        Ok(result) => assert_eq!(
            result.is_error,
            Some(true),
            "a 401 from the API must not look like success: {result:?}"
        ),
    }

    let _ = client.cancel().await;
}
