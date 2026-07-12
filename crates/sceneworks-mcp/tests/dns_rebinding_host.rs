//! F-040 (sc-11236): the `/mcp` streamable-HTTP transport must re-validate the
//! `Host` header (DNS-rebinding defense). `/mcp` rides the API's `access_control`
//! middleware, but that gate performs NO Host/Origin validation — so in the
//! loopback / loopback-trust / no-token desktop posture a malicious web page could
//! DNS-rebind a victim's browser onto `/mcp`. These tests drive the mounted
//! service with raw HTTP/1.1 requests carrying spoofed / legitimate Host headers
//! and assert the transport 403s a disallowed Host while accepting the loopback
//! (and, in remote mode, the configured LAN) host.

use axum::Router;
use sceneworks_mcp::{mcp_allowed_hosts, streamable_http_service_with_hosts, ApiClientConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Mount the MCP service (with `allowed_hosts`) at `/mcp` behind a live listener,
/// returning its `host:port` authority. The API base URL is never dialed here (no
/// tool is called), so a placeholder is fine.
async fn serve_mcp(allowed_hosts: Vec<String>) -> String {
    let service = streamable_http_service_with_hosts(
        ApiClientConfig {
            base_url: "http://127.0.0.1:0".to_owned(),
            access_token: None,
        },
        sceneworks_mcp::JobWaitConfig::default(),
        allowed_hosts,
    );
    let router = Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mcp listener");
    let addr = listener.local_addr().expect("mcp addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr.to_string()
}

/// Send a raw `GET /mcp` with an explicit `Host` header (raw TCP so the Host is
/// exactly what we choose — a client library would derive it from the URL) and
/// return the HTTP status code. The DNS-rebinding check runs before method
/// dispatch, so a GET is enough to exercise it without a full MCP handshake.
async fn get_mcp_status(authority: &str, host_header: &str) -> u16 {
    let mut stream = TcpStream::connect(authority)
        .await
        .expect("connect to mcp listener");
    let request = format!(
        "GET /mcp HTTP/1.1\r\nHost: {host_header}\r\nAccept: text/event-stream\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read response");
    let response = String::from_utf8_lossy(&buf);
    let status_line = response.lines().next().expect("status line");
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status code in response: {status_line:?}"))
}

#[tokio::test]
async fn loopback_bind_rejects_spoofed_host_but_accepts_loopback() {
    // Default desktop posture: loopback bind → loopback-only allow-list.
    let allowed = mcp_allowed_hosts("127.0.0.1", 0, &[]);
    let authority = serve_mcp(allowed).await;
    let port = authority.rsplit(':').next().unwrap();

    // A DNS-rebinding page reaches 127.0.0.1 but sends an attacker Host → 403.
    assert_eq!(
        get_mcp_status(&authority, "attacker.example").await,
        403,
        "a disallowed Host must be rejected (DNS-rebinding defense)"
    );

    // The legitimate local UI / MCP client dials 127.0.0.1 or localhost → allowed
    // (bare loopback entries match any port, incl. the OS-assigned desktop port).
    for host in [format!("127.0.0.1:{port}"), format!("localhost:{port}")] {
        assert_ne!(
            get_mcp_status(&authority, &host).await,
            403,
            "loopback Host {host} must be accepted"
        );
    }
}

#[tokio::test]
async fn remote_bind_accepts_configured_lan_host_and_rejects_others() {
    // LAN remote posture: wildcard bind (0.0.0.0) + operator-declared LAN host via
    // SCENEWORKS_MCP_ALLOWED_HOSTS. The defense stays ON with the LAN host allowed.
    let allowed = mcp_allowed_hosts("0.0.0.0", 8000, &["scenebox.local:8000".to_owned()]);
    let authority = serve_mcp(allowed).await;
    let port = authority.rsplit(':').next().unwrap();

    // The declared LAN authority passes; loopback still passes.
    assert_ne!(
        get_mcp_status(&authority, "scenebox.local:8000").await,
        403,
        "the configured LAN Host must be accepted in remote mode"
    );
    assert_ne!(
        get_mcp_status(&authority, &format!("127.0.0.1:{port}")).await,
        403,
        "loopback stays allowed in remote mode"
    );

    // A different host — or the right host on the wrong port — is rejected.
    assert_eq!(
        get_mcp_status(&authority, "attacker.example").await,
        403,
        "an unrelated Host must be rejected even in remote mode"
    );
    assert_eq!(
        get_mcp_status(&authority, "scenebox.local:9999").await,
        403,
        "the LAN host on a non-configured port must be rejected"
    );
}

#[tokio::test]
async fn wildcard_bind_without_operator_hosts_disables_check() {
    // Wildcard bind with no override: the reachable interface IPs can't be
    // enumerated, so the Host check is disabled (empty allow-list ⇒ allow all) to
    // avoid locking out LAN clients — the mandatory LAN access token is the control
    // there. This documents/guards that deliberate fallback.
    assert!(
        mcp_allowed_hosts("0.0.0.0", 8000, &[]).is_empty(),
        "wildcard bind without operator hosts yields an empty (disabled) allow-list"
    );
    let authority = serve_mcp(mcp_allowed_hosts("0.0.0.0", 8000, &[])).await;
    assert_ne!(
        get_mcp_status(&authority, "anything.example").await,
        403,
        "with the check disabled, any Host is accepted (token-gated deployment)"
    );
}
