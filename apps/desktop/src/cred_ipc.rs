//! On-demand download-credential socket for the macOS desktop (sc-5891).
//!
//! The desktop is the only process that can read the OS keychain. Reading it
//! eagerly at worker spawn (to inject `HF_TOKEN`/`SCENEWORKS_CREDENTIALS`) is what
//! prompted for the keychain password at every launch. Instead this module hosts a
//! tiny Unix-domain-socket server in the desktop process: the MLX worker is given
//! the socket path + a per-launch token, and pulls a host's secret from here the
//! first time a download actually needs it. The keychain is therefore read lazily —
//! the single prompt happens at download time, not at launch — and the result is
//! cached for the rest of the desktop session so worker restarts don't re-prompt.
//!
//! Security: the socket is created `0600` (same-user only) and the worker must
//! present the per-launch token. Only credentials recorded in `settings.json`
//! metadata are ever read (`settings::resolve_credential_secret` gates on that), so
//! an install with nothing stored never causes a keychain touch.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::settings::{self, CredentialScheme};

/// Cached, already-resolved secret for a host, so repeat pulls (a multi-file gated
/// download, or a worker restart within this session) don't re-read the keychain.
#[derive(Clone)]
struct CachedSecret {
    token: String,
    scheme: &'static str,
}

type CredCache = Arc<Mutex<HashMap<String, CachedSecret>>>;

/// Handle to the running credential socket, stored in `Managed` so the MLX worker
/// spawn site can inject the socket path/token and the credential commands can
/// invalidate the cache.
pub struct CredIpc {
    pub socket: PathBuf,
    pub token: String,
    cache: CredCache,
}

impl CredIpc {
    /// Drop a host's cached secret so a later pull re-reads the keychain — used when
    /// the user updates or removes that credential mid-session (revocation must take
    /// effect without an app restart).
    pub fn invalidate(&self, host: &str) {
        let host = host.trim().to_ascii_lowercase();
        if let Ok(mut cache) = self.cache.lock() {
            cache.remove(&host);
        }
    }
}

fn scheme_str(scheme: CredentialScheme) -> &'static str {
    match scheme {
        CredentialScheme::Bearer => "bearer",
        CredentialScheme::Query => "query",
    }
}

/// A per-launch random token (hex). Defense-in-depth on top of the socket's `0600`
/// same-user restriction. Falls back to a process/time-derived value if
/// `/dev/urandom` is somehow unreadable — the file-mode is the real guard.
fn random_token() -> String {
    use std::io::Read;
    let mut bytes = [0u8; 24];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .is_ok()
    {
        return bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    }
    format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    )
}

/// Bind the credential socket and spawn its server thread. Returns the handle to
/// store in `Managed`, or `None` if the socket couldn't be created (the worker then
/// simply gets no credentials — a gated download fails with a clear auth error
/// rather than the app prompting at launch).
pub fn start(socket: PathBuf) -> Option<CredIpc> {
    // Remove any stale socket from a prior (crashed) launch so bind succeeds.
    let _ = std::fs::remove_file(&socket);
    if let Some(parent) = socket.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let listener = match UnixListener::bind(&socket) {
        Ok(listener) => listener,
        Err(_) => return None,
    };
    // Restrict to the current user; the per-launch token is the second factor.
    let _ = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600));

    let token = random_token();
    let cache: CredCache = Arc::new(Mutex::new(HashMap::new()));
    let server_token = token.clone();
    let server_cache = Arc::clone(&cache);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            handle_connection(stream, &server_token, &server_cache);
        }
    });
    Some(CredIpc {
        socket,
        token,
        cache,
    })
}

/// Serve one request: read `"<token> <host>\n"`, validate the token, then reply with
/// `{ "token": "...", "scheme": "bearer" | "query" }` for a recorded+present host, or
/// `ERR` otherwise. One request per connection; the worker opens a fresh connection
/// each time.
fn handle_connection(stream: UnixStream, token: &str, cache: &CredCache) {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let mut parts = line.trim().splitn(2, ' ');
    let presented = parts.next().unwrap_or_default();
    let host = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
    let response = if presented != token || host.is_empty() {
        "ERR".to_owned()
    } else {
        resolve(&host, cache).unwrap_or_else(|| "ERR".to_owned())
    };
    let mut stream = reader.into_inner();
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// The cached or freshly-read secret for `host`, serialized as the response JSON.
/// Only recorded+present credentials resolve; the keychain read happens here (lazy),
/// and a successful read is cached. Absent secrets are not cached, so adding a
/// credential later this session works without restarting the server.
fn resolve(host: &str, cache: &CredCache) -> Option<String> {
    if let Some(cached) = cache.lock().ok().and_then(|cache| cache.get(host).cloned()) {
        return Some(response_json(&cached.token, cached.scheme));
    }
    let (token, scheme) = settings::resolve_credential_secret(host)?;
    let scheme = scheme_str(scheme);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(
            host.to_owned(),
            CachedSecret {
                token: token.clone(),
                scheme,
            },
        );
    }
    Some(response_json(&token, scheme))
}

fn response_json(token: &str, scheme: &str) -> String {
    serde_json::json!({ "token": token, "scheme": scheme }).to_string()
}
