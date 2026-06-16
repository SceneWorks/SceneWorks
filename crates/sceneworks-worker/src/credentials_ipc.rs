//! Lazy, on-demand download-credential resolution for the macOS desktop build
//! (sc-5891).
//!
//! On macOS the desktop app is the only process that can read the OS keychain, and
//! reading it eagerly at worker spawn (to inject `HF_TOKEN`/`SCENEWORKS_CREDENTIALS`)
//! is what triggered a keychain-password prompt at every launch. Instead the desktop
//! runs a tiny Unix-domain-socket credential server and injects only its socket path
//! (`SCENEWORKS_CRED_IPC_SOCKET`), a per-launch token (`SCENEWORKS_CRED_IPC_TOKEN`),
//! and the *non-secret* list of hosts that have a credential recorded
//! (`SCENEWORKS_CREDENTIAL_HOSTS`). This module pulls a host's secret from that
//! socket the first time a download actually needs it — so the single keychain prompt
//! happens then, not at launch — caching it per process for the rest of the session.
//!
//! Off the desktop (server/Docker, the Windows candle worker) none of the
//! `SCENEWORKS_CRED_IPC_*` env vars are set, so the resolver is inert and the worker
//! keeps using the eager env/file-store credentials exactly as before. The socket
//! pull itself is `cfg(unix)`; on other targets it compiles to a no-op (the desktop
//! that injects the env vars only exists on macOS anyway).

use std::collections::HashMap;
use std::sync::OnceLock;

use tokio::sync::Mutex;

use crate::{credential_for_host, Settings, WorkerCredential};
// Only the `cfg(unix)` response parser (and its tests) name the scheme enum; gating
// the import keeps the Windows build free of an unused-import warning.
#[cfg(unix)]
use crate::CredentialScheme;

/// Path of the desktop credential socket (macOS desktop only).
const ENV_SOCKET: &str = "SCENEWORKS_CRED_IPC_SOCKET";
/// Per-launch shared token the worker presents to the socket (defense-in-depth on
/// top of the socket file's `0600` same-user restriction).
const ENV_TOKEN: &str = "SCENEWORKS_CRED_IPC_TOKEN";
/// Comma-separated, non-secret list of hosts that have a credential recorded in the
/// desktop keychain. The worker only ever asks the socket for hosts in this list, so
/// an install with nothing recorded never causes a keychain touch — the
/// no-credential invariant holds end to end.
const ENV_HOSTS: &str = "SCENEWORKS_CREDENTIAL_HOSTS";

struct IpcConfig {
    // Read only by the `cfg(unix)` socket pull; off unix the desktop never injects
    // these, so the fields are constructed-but-unread there.
    #[cfg_attr(not(unix), allow(dead_code))]
    socket: std::path::PathBuf,
    #[cfg_attr(not(unix), allow(dead_code))]
    token: String,
    recorded_hosts: Vec<String>,
}

impl IpcConfig {
    fn from_env() -> Option<Self> {
        let socket = std::env::var(ENV_SOCKET)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .map(std::path::PathBuf::from)?;
        let token = std::env::var(ENV_TOKEN)
            .unwrap_or_default()
            .trim()
            .to_owned();
        let recorded_hosts = std::env::var(ENV_HOSTS)
            .unwrap_or_default()
            .split(',')
            .map(|host| host.trim().to_ascii_lowercase())
            .filter(|host| !host.is_empty())
            .collect();
        Some(Self {
            socket,
            token,
            recorded_hosts,
        })
    }

    fn has_host(&self, host: &str) -> bool {
        self.recorded_hosts.iter().any(|recorded| recorded == host)
    }
}

struct Resolver {
    /// `None` off the macOS desktop (no `SCENEWORKS_CRED_IPC_SOCKET`), making every
    /// lazy pull a no-op so the static env/file-store path is used unchanged.
    config: Option<IpcConfig>,
    /// Per-process cache of pulled secrets, so a multi-file gated download — and any
    /// worker restart within the same desktop session (the desktop server caches
    /// too) — prompts at most once.
    cache: Mutex<HashMap<String, WorkerCredential>>,
}

impl Resolver {
    fn from_env() -> Self {
        Self {
            config: IpcConfig::from_env(),
            cache: Mutex::new(HashMap::new()),
        }
    }
}

static RESOLVER: OnceLock<Resolver> = OnceLock::new();

fn resolver() -> &'static Resolver {
    RESOLVER.get_or_init(Resolver::from_env)
}

/// The download credential for `host`: the statically-injected env/file-store
/// credential first (server/Docker, the Windows candle worker), else — on the macOS
/// desktop — a lazy pull from the desktop credential socket (cached per process).
/// `None` when no credential is available for the host.
pub(crate) async fn resolve_credential_for_host(
    settings: &Settings,
    host: &str,
) -> Option<WorkerCredential> {
    let host = host.trim().to_ascii_lowercase();
    if host.is_empty() {
        return None;
    }
    if let Some(credential) = credential_for_host(settings, &host) {
        return Some(credential.clone());
    }
    fetch_via_ipc(&host).await
}

/// The Hugging Face token for authenticating gated HF downloads: the operator/env
/// `HF_TOKEN` first (server/Docker, Windows), else the recorded `huggingface.co`
/// credential pulled lazily from the desktop socket.
pub(crate) async fn resolve_hf_token(settings: &Settings) -> Option<String> {
    if let Some(token) = &settings.huggingface_token {
        return Some(token.clone());
    }
    resolve_credential_for_host(settings, "huggingface.co")
        .await
        .map(|credential| credential.token)
}

async fn fetch_via_ipc(host: &str) -> Option<WorkerCredential> {
    let resolver = resolver();
    let config = resolver.config.as_ref()?;
    // Only ask for hosts the desktop says are recorded — never trigger a keychain
    // touch for something that isn't stored.
    if !config.has_host(host) {
        return None;
    }
    if let Some(credential) = resolver.cache.lock().await.get(host).cloned() {
        return Some(credential);
    }
    let credential = request_credential(config, host).await?;
    resolver
        .cache
        .lock()
        .await
        .insert(host.to_owned(), credential.clone());
    Some(credential)
}

/// Parse the socket's response line into a credential. The server replies with a
/// `{ "token": "...", "scheme": "bearer" | "query" }` JSON object, or `ERR`/empty
/// when the host has no secret. Only the `cfg(unix)` pull (and its tests) parse
/// responses.
#[cfg(unix)]
fn parse_response(host: &str, body: &str) -> Option<WorkerCredential> {
    let body = body.trim();
    if body.is_empty() || body == "ERR" {
        return None;
    }
    #[derive(serde::Deserialize)]
    struct Response {
        token: String,
        #[serde(default)]
        scheme: Option<String>,
    }
    let response: Response = serde_json::from_str(body).ok()?;
    let token = response.token.trim().to_owned();
    if token.is_empty() {
        return None;
    }
    let scheme = match response.scheme.as_deref() {
        Some("query") => CredentialScheme::Query,
        _ => CredentialScheme::Bearer,
    };
    Some(WorkerCredential {
        host: host.to_owned(),
        token,
        scheme,
    })
}

#[cfg(unix)]
async fn request_credential(config: &IpcConfig, host: &str) -> Option<WorkerCredential> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    let mut stream = UnixStream::connect(&config.socket).await.ok()?;
    // One request line: "<token> <host>\n". The server reads the line, replies, and
    // closes, so reading to EOF yields the full response.
    let request = format!("{} {}\n", config.token, host);
    stream.write_all(request.as_bytes()).await.ok()?;
    let _ = stream.shutdown().await;
    let mut response = String::new();
    stream.read_to_string(&mut response).await.ok()?;
    parse_response(host, &response)
}

#[cfg(not(unix))]
async fn request_credential(_config: &IpcConfig, _host: &str) -> Option<WorkerCredential> {
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    #[test]
    fn parse_response_reads_token_and_scheme() {
        let bearer = parse_response("huggingface.co", r#"{"token":"  hf_x ","scheme":"bearer"}"#)
            .expect("bearer");
        assert_eq!(bearer.token, "hf_x");
        assert_eq!(bearer.scheme, CredentialScheme::Bearer);

        let query =
            parse_response("civitai.com", r#"{"token":"k","scheme":"query"}"#).expect("query");
        assert_eq!(query.scheme, CredentialScheme::Query);

        // Unknown/missing scheme defaults to bearer.
        assert_eq!(
            parse_response("h", r#"{"token":"k"}"#).unwrap().scheme,
            CredentialScheme::Bearer
        );
        // Sentinels / empties yield nothing.
        assert!(parse_response("h", "ERR").is_none());
        assert!(parse_response("h", "").is_none());
        assert!(parse_response("h", r#"{"token":"  "}"#).is_none());
    }

    /// End-to-end wire test of the socket pull against a one-shot fake server: the
    /// worker sends "<token> <host>\n" and parses the JSON reply.
    #[tokio::test]
    async fn request_credential_round_trips_over_uds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let socket = dir.path().join("cred.sock");
        let listener = UnixListener::bind(&socket).expect("bind");

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut request = String::new();
            stream
                .read_to_string(&mut request)
                .await
                .expect("read request");
            assert_eq!(request.trim(), "tok huggingface.co");
            stream
                .write_all(br#"{"token":"hf_secret","scheme":"bearer"}"#)
                .await
                .expect("write response");
        });

        let config = IpcConfig {
            socket,
            token: "tok".to_owned(),
            recorded_hosts: vec!["huggingface.co".to_owned()],
        };
        let credential = request_credential(&config, "huggingface.co")
            .await
            .expect("credential");
        assert_eq!(credential.token, "hf_secret");
        assert_eq!(credential.scheme, CredentialScheme::Bearer);
        server.await.expect("server task");
    }
}
