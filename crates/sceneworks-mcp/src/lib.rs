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
/// Defaults to the loopback-only DNS-rebinding [`allowed_hosts`](mcp_allowed_hosts)
/// posture (`localhost`/`127.0.0.1`/`::1`). Production wires
/// [`streamable_http_service_with_hosts`] with the set derived from the API's
/// bind config; the loopback default here keeps in-crate tests (which dial
/// `127.0.0.1`) working with the defense ON.
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
    streamable_http_service_with_hosts(config, job_wait, loopback_allowed_hosts())
}

/// The full constructor: also takes the [`allowed_hosts`](mcp_allowed_hosts)
/// list for the transport's Host-header (DNS-rebinding) validation.
///
/// **F-040 (sc-11236).** `/mcp` is mounted inside the API's axum router and, in
/// the default desktop posture (loopback bind, `SCENEWORKS_TRUST_LOOPBACK`, no
/// token), the surrounding `access_control` middleware performs NO Host/Origin
/// validation — so a malicious web page could use DNS rebinding to reach `/mcp`
/// from the victim's browser and drive job submission / ticketed file reads. rmcp
/// re-validates the `Host` header against this list and 403s a mismatch (the
/// canonical DNS-rebinding defense). An **empty** list disables the check
/// (rmcp's allow-all): used only for the wildcard LAN bind, where the interface
/// address can't be enumerated and a token already gates every request — see
/// [`mcp_allowed_hosts`].
pub fn streamable_http_service_with_hosts(
    config: ApiClientConfig,
    job_wait: JobWaitConfig,
    allowed_hosts: Vec<String>,
) -> McpHttpService {
    let api = ApiClient::new(config);
    let transport_config = if allowed_hosts.is_empty() {
        // Empty ⇒ allow any Host (rmcp semantics). Only reached for the wildcard
        // LAN bind with no operator override; the access token is the control there.
        StreamableHttpServerConfig::default().disable_allowed_hosts()
    } else {
        StreamableHttpServerConfig::default().with_allowed_hosts(allowed_hosts)
    };
    StreamableHttpService::new(
        move || Ok(SceneWorksMcp::new(api.clone()).with_job_wait(job_wait.clone())),
        Arc::new(LocalSessionManager::default()),
        transport_config,
    )
}

/// The loopback host set always accepted for `/mcp`: `localhost`, `127.0.0.1`,
/// `::1`. Bare host entries (no port) match ANY port, so the desktop's
/// OS-assigned dynamic loopback port is covered.
pub fn loopback_allowed_hosts() -> Vec<String> {
    vec![
        "localhost".to_owned(),
        "127.0.0.1".to_owned(),
        "::1".to_owned(),
    ]
}

/// Derive the `/mcp` Host-header allow-list (F-040, sc-11236) from the API's bind
/// configuration — the SAME `SCENEWORKS_API_HOST`/`SCENEWORKS_API_PORT` that
/// decide where the API listens.
///
/// - **Loopback / default desktop** (`host` is a loopback name/IP): the returned
///   set is [`loopback_allowed_hosts`] plus any `extra` operator entries. This
///   restores the DNS-rebinding defense for the posture the finding names — a
///   rebinding page sending `Host: attacker.example` is 403'd, while the local UI
///   and worker (`127.0.0.1`/`localhost`, any port) pass.
/// - **Concrete non-loopback interface** (e.g. `SCENEWORKS_API_HOST=192.168.4.97`):
///   loopback set + that host, both bare and `host:port`, + `extra`. Legit LAN
///   clients dialing that address pass; other Hosts are rejected.
/// - **Wildcard LAN bind** (`0.0.0.0` / `::`, i.e. the desktop's remote-access
///   mode): the reachable interface addresses cannot be enumerated from a
///   wildcard, so the operator declares them via `extra`
///   (`SCENEWORKS_MCP_ALLOWED_HOSTS`). When `extra` is set the returned set is
///   loopback + extra (defense ON). When `extra` is empty this returns an
///   **empty** vec, which the constructor treats as "disable Host validation" so
///   legitimate LAN clients are never locked out — safe because remote mode
///   ALWAYS binds with an access token, and a browser doing DNS rebinding is a
///   remote (non-loopback) peer that cannot present that token.
///
/// `port` is included on the concrete-host entry; loopback and `extra` entries
/// are passed through verbatim. Result is de-duplicated preserving order.
pub fn mcp_allowed_hosts(host: &str, port: u16, extra: &[String]) -> Vec<String> {
    let host = host.trim();
    let is_wildcard = host.is_empty() || host == "0.0.0.0" || host == "::" || host == "[::]";
    let is_loopback = host == "127.0.0.1"
        || host == "::1"
        || host == "[::1]"
        || host.eq_ignore_ascii_case("localhost");

    let extra: Vec<String> = extra
        .iter()
        .map(|entry| entry.trim().to_owned())
        .filter(|entry| !entry.is_empty())
        .collect();

    // Wildcard LAN bind with no operator override: cannot enumerate reachable
    // hosts, so disable the Host check (token-gated) rather than 403 real clients.
    if is_wildcard && extra.is_empty() {
        return Vec::new();
    }

    let mut hosts = loopback_allowed_hosts();
    if !is_wildcard && !is_loopback {
        // A concrete interface address the operator bound to: allow it (any port
        // and the specific `host:port`), so a client dialing it directly passes.
        hosts.push(host.to_owned());
        hosts.push(format!("{host}:{port}"));
    }
    hosts.extend(extra);
    // Order-preserving de-dup (an `extra` entry may repeat a loopback name).
    let mut seen = std::collections::HashSet::new();
    hosts.retain(|entry| seen.insert(entry.clone()));
    hosts
}

#[cfg(test)]
mod tests {
    use super::mcp_allowed_hosts;

    const LOOPBACK: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

    #[test]
    fn loopback_bind_is_loopback_only() {
        for host in ["127.0.0.1", "::1", "localhost", "LocalHost"] {
            let allowed = mcp_allowed_hosts(host, 0, &[]);
            assert_eq!(allowed, LOOPBACK, "loopback bind {host} → loopback-only");
        }
    }

    #[test]
    fn wildcard_bind_without_extra_disables_check() {
        for host in ["0.0.0.0", "::", "[::]", ""] {
            assert!(
                mcp_allowed_hosts(host, 8000, &[]).is_empty(),
                "wildcard bind {host} with no override disables the check"
            );
        }
    }

    #[test]
    fn wildcard_bind_with_extra_enforces_loopback_plus_extra() {
        let allowed = mcp_allowed_hosts("0.0.0.0", 8000, &["scenebox.local:8000".to_owned()]);
        assert_eq!(
            allowed,
            ["localhost", "127.0.0.1", "::1", "scenebox.local:8000"],
            "operator override re-enables the check with the LAN host allowed"
        );
    }

    #[test]
    fn concrete_interface_host_is_added_with_and_without_port() {
        let allowed = mcp_allowed_hosts("192.168.4.97", 8000, &[]);
        assert_eq!(
            allowed,
            [
                "localhost",
                "127.0.0.1",
                "::1",
                "192.168.4.97",
                "192.168.4.97:8000"
            ],
        );
    }

    #[test]
    fn extra_entries_are_trimmed_and_deduped_against_loopback() {
        let allowed = mcp_allowed_hosts(
            "192.168.4.97",
            8000,
            &[
                " localhost ".to_owned(),
                "".to_owned(),
                "box.lan".to_owned(),
            ],
        );
        // `localhost` already present (trimmed) → not duplicated; empty dropped.
        assert_eq!(
            allowed,
            [
                "localhost",
                "127.0.0.1",
                "::1",
                "192.168.4.97",
                "192.168.4.97:8000",
                "box.lan"
            ],
        );
    }
}
