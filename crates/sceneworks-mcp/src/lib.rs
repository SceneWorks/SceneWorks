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
pub mod yue2;

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
/// [`streamable_http_service_with_origins`] with the set derived from the API's
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
/// Host validation is always enabled, including wildcard LAN mode. The API's
/// access-control middleware additionally validates Origin before token/loopback bypass.
pub fn streamable_http_service_with_hosts(
    config: ApiClientConfig,
    job_wait: JobWaitConfig,
    allowed_hosts: Vec<String>,
) -> McpHttpService {
    streamable_http_service_with_origins(config, job_wait, allowed_hosts, Vec::new())
}

/// Full browser policy: explicit app origins plus HTTP(S) origins on allowed hosts.
pub fn streamable_http_service_with_origins(
    config: ApiClientConfig,
    job_wait: JobWaitConfig,
    allowed_hosts: Vec<String>,
    mut allowed_origins: Vec<String>,
) -> McpHttpService {
    let api = ApiClient::new(config);
    // Empty input is fail-closed to loopback, never rmcp's allow-all posture.
    let allowed_hosts = if allowed_hosts.is_empty() {
        loopback_allowed_hosts()
    } else {
        allowed_hosts
    };
    let forwarded_hosts = allowed_hosts.clone();
    for host in &allowed_hosts {
        let host = if host.parse::<std::net::Ipv6Addr>().is_ok() {
            format!("[{host}]")
        } else {
            host.clone()
        };
        allowed_origins.extend([format!("http://{host}"), format!("https://{host}")]);
    }
    let transport_config = StreamableHttpServerConfig::default()
        .with_allowed_hosts(allowed_hosts)
        .with_allowed_origins(allowed_origins);
    StreamableHttpService::new(
        move || {
            Ok(SceneWorksMcp::new(api.clone())
                .with_job_wait(job_wait.clone())
                .with_allowed_hosts(forwarded_hosts.clone()))
        },
        Arc::new(LocalSessionManager::default()),
        transport_config,
    )
}

/// Match an HTTP authority against the MCP Host allow-list using the same
/// host/optional-port semantics as rmcp. Unlike rmcp's inbound-host fallback,
/// an empty list admits no forwarded authority. Only enumerated or explicitly
/// configured hosts can influence ticket-bearing URLs.
pub fn forwarded_authority_is_allowed(authority: &str, allowed_hosts: &[String]) -> bool {
    let Ok(authority) = http::uri::Authority::try_from(authority.trim()) else {
        return false;
    };
    let suffix = &authority.as_str()[authority.host().len()..];
    if authority.as_str().contains('@')
        || (!suffix.is_empty() && (!suffix.starts_with(':') || suffix[1..].parse::<u16>().is_err()))
    {
        return false;
    }
    let host = authority
        .host()
        .trim_matches(['[', ']'])
        .to_ascii_lowercase();
    let port = authority.port_u16();
    !allowed_hosts.is_empty()
        && allowed_hosts.iter().any(|allowed| {
            let allowed = allowed.trim();
            let (allowed_host, allowed_port) =
                if let Ok(authority) = http::uri::Authority::try_from(allowed) {
                    (
                        authority
                            .host()
                            .trim_matches(['[', ']'])
                            .to_ascii_lowercase(),
                        authority.port_u16(),
                    )
                } else {
                    (allowed.trim_matches(['[', ']']).to_ascii_lowercase(), None)
                };
            allowed_host == host && allowed_port.is_none_or(|allowed| port == Some(allowed))
        })
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
/// - **Wildcard LAN bind**: loopback plus explicit entries; production calls
///   [`discovered_allowed_hosts`] to add the machine's concrete interface addresses.
///   An empty override never disables rebinding protection.
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

    let mut hosts = loopback_allowed_hosts();
    if !is_wildcard && !is_loopback {
        // A concrete interface address the operator bound to: allow it (any port
        // and the specific `host:port`), so a client dialing it directly passes.
        hosts.push(host.to_owned());
        let authority_host = if host.parse::<std::net::Ipv6Addr>().is_ok() {
            format!("[{host}]")
        } else {
            host.to_owned()
        };
        hosts.push(format!("{authority_host}:{port}"));
    }
    hosts.extend(extra);
    // Order-preserving de-dup (an `extra` entry may repeat a loopback name).
    let mut seen = std::collections::HashSet::new();
    hosts.retain(|entry| seen.insert(entry.clone()));
    hosts
}

/// Enumerate concrete interface addresses once at API startup for wildcard listeners.
/// Enumeration failure retains the loopback and explicit operator entries (fail closed).
pub fn discovered_allowed_hosts(host: &str, port: u16, extra: &[String]) -> Vec<String> {
    let mut extra = extra.to_vec();
    if matches!(host.trim(), "" | "0.0.0.0" | "::" | "[::]") {
        match if_addrs::get_if_addrs() {
            Ok(interfaces) => extra.extend(
                interfaces
                    .into_iter()
                    .map(|interface| interface.ip().to_string()),
            ),
            Err(error) => tracing::warn!(%error, "could not enumerate allowed API hosts"),
        }
    }
    mcp_allowed_hosts(host, port, &extra)
}

#[cfg(test)]
mod tests {
    use super::{forwarded_authority_is_allowed, mcp_allowed_hosts};

    const LOOPBACK: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

    #[test]
    fn loopback_bind_is_loopback_only() {
        for host in ["127.0.0.1", "::1", "localhost", "LocalHost"] {
            let allowed = mcp_allowed_hosts(host, 0, &[]);
            assert_eq!(allowed, LOOPBACK, "loopback bind {host} → loopback-only");
        }
    }

    #[test]
    fn wildcard_bind_without_extra_keeps_check() {
        for host in ["0.0.0.0", "::", "[::]", ""] {
            assert!(
                mcp_allowed_hosts(host, 8000, &[]) == LOOPBACK,
                "wildcard bind {host} with no override retains loopback protection"
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
    fn concrete_ipv6_host_uses_bracketed_authority() {
        let allowed = mcp_allowed_hosts("fd00::42", 8000, &[]);
        assert!(allowed.iter().any(|host| host == "[fd00::42]:8000"));
        assert!(!allowed.iter().any(|host| host == "fd00::42:8000"));
        assert!(forwarded_authority_is_allowed("[fd00::42]:8000", &allowed));
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

    #[test]
    fn forwarded_authority_requires_an_exact_configured_host() {
        let allowed = vec![
            "studio.example.com:443".to_owned(),
            "scenebox.local".to_owned(),
        ];
        assert!(forwarded_authority_is_allowed(
            "studio.example.com:443",
            &allowed
        ));
        assert!(!forwarded_authority_is_allowed(
            "studio.example.com:444",
            &allowed
        ));
        assert!(forwarded_authority_is_allowed(
            "scenebox.local:8000",
            &allowed
        ));
        assert!(!forwarded_authority_is_allowed(
            "attacker.example",
            &allowed
        ));
        assert!(!forwarded_authority_is_allowed("studio.example.com", &[]));
        for malformed in [
            "user@scenebox.local",
            "scenebox.local:bad",
            "scenebox.local:99999",
            "scenebox.local:",
        ] {
            assert!(
                !forwarded_authority_is_allowed(malformed, &allowed),
                "{malformed}"
            );
        }
    }
}
