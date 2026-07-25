#![allow(dead_code)]

use std::time::Duration;

use axum::Router;
use rmcp::handler::client::ClientHandler;
use rmcp::model::{CallToolResult, ClientInfo};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use sceneworks_mcp::{ApiClientConfig, JobWaitConfig};
use serde_json::Value;

pub async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub listener");
    let addr = listener.local_addr().expect("stub addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

#[derive(Clone, Default)]
pub struct TestClient;

impl ClientHandler for TestClient {
    fn get_info(&self) -> ClientInfo {
        ClientInfo::default()
    }
}

pub async fn connect_mcp<H>(
    api_base: String,
    access_token: Option<String>,
    job_wait: JobWaitConfig,
    handler: H,
) -> (String, RunningService<RoleClient, H>)
where
    H: ClientHandler,
{
    let mcp_service = sceneworks_mcp::streamable_http_service_with(
        ApiClientConfig {
            base_url: api_base,
            access_token,
        },
        job_wait,
    );
    let mcp_base = spawn(Router::new().nest_service("/mcp", mcp_service)).await;
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(format!("{mcp_base}/mcp")),
    );
    let client = handler
        .serve(transport)
        .await
        .expect("MCP client initializes against the mounted /mcp service");
    (mcp_base, client)
}

pub fn fast_job_wait() -> JobWaitConfig {
    JobWaitConfig {
        poll_interval: Duration::from_millis(10),
        timeout: Duration::from_secs(10),
    }
}

pub fn call_args(value: Value) -> serde_json::Map<String, Value> {
    value.as_object().expect("args are an object").clone()
}

/// Parse the last text block. Job-result tools append their JSON summary after resource links,
/// while catalog tools carry a single text block, so this works for both harness families.
pub fn result_json(result: &CallToolResult) -> Value {
    result
        .content
        .iter()
        .rev()
        .find_map(|block| block.as_text())
        .map(|text| serde_json::from_str(&text.text).expect("tool result text is JSON"))
        .expect("tool result carries a text block")
}

pub fn error_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| block.as_text())
        .map(|text| text.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}
