//! SceneWorks MCP server (epic 10231, sc-10233).
//!
//! A Model Context Protocol server over the official `rmcp` SDK's
//! streamable-HTTP transport. [`streamable_http_service`] returns a tower
//! service the API app nests at `/mcp` inside its existing axum router
//! (`apps/rust-api/src/lib.rs`, `create_app_with_state`), so every MCP request
//! rides the existing `access_control` middleware (X-SceneWorks-Token / Bearer,
//! `SCENEWORKS_TRUST_LOOPBACK`, brute-force throttle) — the MCP layer adds NO
//! auth of its own.
//!
//! The tools themselves are a thin client over `/api/v1/*` (see
//! [`api_client`]): the MCP process calls back into the API over HTTP exactly
//! like the Rust worker does, so there is one behavior surface to maintain.

pub mod api_client;
pub mod server;

use std::sync::Arc;

use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

pub use api_client::{ApiClient, ApiClientConfig};
pub use server::{JobWaitConfig, SceneWorksMcp};

/// The concrete tower service type the API mounts at `/mcp`.
pub type McpHttpService = StreamableHttpService<SceneWorksMcp, LocalSessionManager>;

/// Build the streamable-HTTP MCP service. One service instance per app; rmcp
/// spins up a fresh [`SceneWorksMcp`] (sharing the one [`ApiClient`]) per MCP
/// session.
///
/// rmcp's own `allowed_hosts` (DNS-rebinding) check is disabled deliberately:
/// its loopback-only default would 403 the supported LAN deployment
/// (`SCENEWORKS_API_HOST=0.0.0.0` + access token), and the real access control
/// for `/mcp` is the surrounding `access_control` middleware — identical to
/// every `/api/v1` route (sc-10233 acceptance).
pub fn streamable_http_service(config: ApiClientConfig) -> McpHttpService {
    streamable_http_service_with(config, JobWaitConfig::default())
}

/// [`streamable_http_service`] with an explicit blocking-job wait policy
/// (generate_image's poll interval + overall deadline). Tests shrink both so
/// submit→poll round trips run in milliseconds.
pub fn streamable_http_service_with(
    config: ApiClientConfig,
    job_wait: JobWaitConfig,
) -> McpHttpService {
    let api = ApiClient::new(config);
    StreamableHttpService::new(
        move || Ok(SceneWorksMcp::new(api.clone()).with_job_wait(job_wait.clone())),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}
