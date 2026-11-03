//! On-demand OS-credential bridge for the desktop (sc-5891, sc-23730).
//!
//! The desktop is the only process that reads the OS secret facility. The API and,
//! on macOS, the MLX worker receive only an ephemeral endpoint plus a per-launch
//! capability and pull a recorded host's secret when an operation needs it. Unix
//! desktops use a `0600` Unix socket. Windows uses a loopback-only, OS-assigned TCP
//! port because it has no Unix socket with the same deployment contract.
//!
//! Only credentials recorded in `settings.json` metadata are read
//! (`settings::resolve_credential_secret` enforces that gate). Successful reads are
//! cached in this process and `set_credential` / `delete_credential` invalidate the
//! affected host, so save, rotation, and removal take effect without restarting the
//! API. No secret or IPC capability is persisted.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(any(windows, test))]
use std::net::{SocketAddr, TcpListener};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::settings::{self, CredentialScheme};

#[derive(Clone)]
struct CachedSecret {
    token: String,
    scheme: &'static str,
}

type CredCache = Arc<Mutex<HashMap<String, CachedSecret>>>;
type SecretResolver = Arc<dyn Fn(&str) -> Option<(String, CredentialScheme)> + Send + Sync>;
const MAX_REQUEST_BYTES: u64 = 4 * 1024;

/// The platform transport handed to a child process. Both forms are local-only;
/// the per-launch token is required in either case.
pub enum CredIpcEndpoint {
    #[cfg(unix)]
    Unix(PathBuf),
    #[cfg(any(windows, test))]
    Tcp(SocketAddr),
}

pub struct CredIpc {
    pub endpoint: CredIpcEndpoint,
    pub token: String,
    cache: CredCache,
    #[cfg(unix)]
    private_directory: Option<tempfile::TempDir>,
}

impl CredIpc {
    pub fn invalidate(&self, host: &str) {
        invalidate_cached_secret(&self.cache, host);
    }

    /// Add the transport and capability to a sidecar command without exposing a
    /// secret. A configured bridge is authoritative in the API, including an
    /// `ERR` response after credential removal.
    pub fn inject_env(
        &self,
        mut command: tauri_plugin_shell::process::Command,
    ) -> tauri_plugin_shell::process::Command {
        command = command.env("SCENEWORKS_CRED_IPC_TOKEN", &self.token);
        match &self.endpoint {
            #[cfg(unix)]
            CredIpcEndpoint::Unix(socket) => command.env(
                "SCENEWORKS_CRED_IPC_SOCKET",
                socket.to_string_lossy().to_string(),
            ),
            #[cfg(any(windows, test))]
            CredIpcEndpoint::Tcp(address) => {
                command.env("SCENEWORKS_CRED_IPC_TCP", address.to_string())
            }
        }
    }

    pub fn cleanup(&self) {
        #[cfg(unix)]
        match &self.endpoint {
            CredIpcEndpoint::Unix(socket) => {
                let _ = std::fs::remove_file(socket);
                if let Some(directory) = &self.private_directory {
                    let _ = std::fs::remove_dir(directory.path());
                }
            }
            #[cfg(test)]
            CredIpcEndpoint::Tcp(_) => {}
        }
    }
}

fn invalidate_cached_secret(cache: &CredCache, host: &str) {
    let host = host.trim().to_ascii_lowercase();
    if let Ok(mut cache) = cache.lock() {
        cache.remove(&host);
    }
}

fn scheme_str(scheme: CredentialScheme) -> &'static str {
    match scheme {
        CredentialScheme::Bearer => "bearer",
        CredentialScheme::Query => "query",
    }
}

/// A cryptographically random per-launch capability. UUID v4 is backed by the OS
/// random source on every supported desktop platform; two values retain the prior
/// token's entropy without relying on a platform-specific random-device path.
fn random_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Start the production transport for this platform: a user-only Unix socket on
/// macOS/Linux, or an authenticated ephemeral loopback listener on Windows.
pub fn start() -> Option<CredIpc> {
    let resolver: SecretResolver = Arc::new(settings::resolve_credential_secret);
    #[cfg(unix)]
    {
        start_unix(resolver)
    }
    #[cfg(windows)]
    {
        start_tcp(resolver)
    }
}

#[cfg(unix)]
fn start_unix(resolver: SecretResolver) -> Option<CredIpc> {
    // Configure directory creation as 0700 before bind. A short ephemeral path
    // keeps Unix sockaddr limits independent of the configured data/config root.
    // Both supported Unix desktop platforms provide /tmp; the private directory
    // owns cleanup on graceful drop, and OS temporary-file cleanup handles crashes.
    let private_directory = tempfile::Builder::new()
        .prefix("scw-cred-")
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir_in("/tmp")
        .ok()?;
    let socket = private_directory.path().join("c");
    let listener = UnixListener::bind(&socket).ok()?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)).ok()?;
    let (token, cache) = serve_connections(
        move || {
            let (stream, _) = listener.accept()?;
            let timeout = Some(std::time::Duration::from_secs(2));
            stream.set_read_timeout(timeout)?;
            stream.set_write_timeout(timeout)?;
            Ok(stream)
        },
        resolver,
    );
    Some(CredIpc {
        endpoint: CredIpcEndpoint::Unix(socket),
        token,
        cache,
        private_directory: Some(private_directory),
    })
}

#[cfg(any(windows, test))]
fn start_tcp(resolver: SecretResolver) -> Option<CredIpc> {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).ok()?;
    let address = listener.local_addr().ok()?;
    debug_assert!(address.ip().is_loopback());
    let (token, cache) = serve_connections(
        move || {
            let (stream, _) = listener.accept()?;
            let timeout = Some(std::time::Duration::from_secs(2));
            stream.set_read_timeout(timeout)?;
            stream.set_write_timeout(timeout)?;
            Ok(stream)
        },
        resolver,
    );
    Some(CredIpc {
        endpoint: CredIpcEndpoint::Tcp(address),
        token,
        cache,
        #[cfg(unix)]
        private_directory: None,
    })
}

/// Bounded worker pool shared by Unix and TCP. A partial request cannot serialize
/// the accept loop, and an unauthenticated flood cannot spawn unbounded threads.
fn serve_connections<S: Read + Write + Send + 'static>(
    mut accept: impl FnMut() -> std::io::Result<S> + Send + 'static,
    resolver: SecretResolver,
) -> (String, CredCache) {
    let token = random_token();
    let cache: CredCache = Arc::new(Mutex::new(HashMap::new()));
    let (sender, receiver) = std::sync::mpsc::sync_channel::<S>(16);
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..8 {
        let receiver = receiver.clone();
        let token = token.clone();
        let cache = cache.clone();
        let resolver = resolver.clone();
        std::thread::spawn(move || loop {
            let stream = match receiver.lock().expect("credential queue").recv() {
                Ok(stream) => stream,
                Err(_) => return,
            };
            handle_connection(stream, &token, &cache, &resolver);
        });
    }
    std::thread::spawn(move || loop {
        if let Ok(stream) = accept() {
            let _ = sender.try_send(stream);
        }
    });
    (token, cache)
}

fn capability_matches(presented: &str, expected: &str) -> bool {
    use subtle::ConstantTimeEq;
    // Production tokens have one fixed public length (64 ASCII bytes). Rejecting
    // a different length reveals no secret; equal-length values use constant time.
    presented.as_bytes().ct_eq(expected.as_bytes()).into()
}

fn handle_connection<S: Read + Write>(
    stream: S,
    token: &str,
    cache: &CredCache,
    resolver: &SecretResolver,
) {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let read_result = reader
        .by_ref()
        .take(MAX_REQUEST_BYTES + 1)
        .read_line(&mut line);
    if read_result.is_err() || line.len() as u64 > MAX_REQUEST_BYTES || !line.ends_with('\n') {
        return;
    }
    let mut parts = line.trim().splitn(2, ' ');
    let presented = parts.next().unwrap_or_default();
    let host = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
    let response = if !capability_matches(presented, token) || host.is_empty() {
        "ERR".to_owned()
    } else {
        resolve(&host, cache, resolver).unwrap_or_else(|| "ERR".to_owned())
    };
    let mut stream = reader.into_inner();
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn resolve(host: &str, cache: &CredCache, resolver: &SecretResolver) -> Option<String> {
    // Resolution and insertion share the invalidation mutex. Otherwise a rotation
    // could remove the cache while an earlier keychain read was in flight, then
    // that read could insert the old value after `set_credential` returned.
    let mut cache = cache.lock().ok()?;
    if let Some(cached) = cache.get(host).cloned() {
        return Some(response_json(&cached.token, cached.scheme));
    }
    let (token, scheme) = resolver(host)?;
    let scheme = scheme_str(scheme);
    cache.insert(
        host.to_owned(),
        CachedSecret {
            token: token.clone(),
            scheme,
        },
    );
    Some(response_json(&token, scheme))
}

fn response_json(token: &str, scheme: &str) -> String {
    serde_json::json!({ "token": token, "scheme": scheme }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;

    fn request(address: SocketAddr, auth: &str, host: &str) -> String {
        let mut stream = TcpStream::connect_timeout(&address, std::time::Duration::from_secs(3))
            .expect("connect to credential bridge");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        stream
            .write_all(format!("{auth} {host}\n").as_bytes())
            .expect("write request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    }

    #[test]
    fn windows_loopback_bridge_refreshes_save_rotate_and_delete_without_restart() {
        let secrets = Arc::new(Mutex::new(HashMap::<String, String>::new()));
        let resolver_secrets = Arc::clone(&secrets);
        let resolver: SecretResolver = Arc::new(move |host| {
            resolver_secrets
                .lock()
                .expect("secret lock")
                .get(host)
                .cloned()
                .map(|token| (token, CredentialScheme::Bearer))
        });
        let ipc = start_tcp(resolver).expect("start loopback bridge");
        let address = match &ipc.endpoint {
            CredIpcEndpoint::Tcp(address) => *address,
            #[cfg(unix)]
            CredIpcEndpoint::Unix(_) => panic!("test bridge must use TCP"),
        };
        assert!(address.ip().is_loopback());
        assert_eq!(
            request(address, "wrong-capability", "planner.example"),
            "ERR"
        );
        assert_eq!(request(address, &ipc.token, "planner.example"), "ERR");

        secrets
            .lock()
            .unwrap()
            .insert("planner.example".to_owned(), "first-token".to_owned());
        assert!(request(address, &ipc.token, "planner.example").contains("first-token"));

        secrets
            .lock()
            .unwrap()
            .insert("planner.example".to_owned(), "rotated-token".to_owned());
        ipc.invalidate("planner.example");
        let rotated = request(address, &ipc.token, "planner.example");
        assert!(rotated.contains("rotated-token"));
        assert!(!rotated.contains("first-token"));

        secrets.lock().unwrap().remove("planner.example");
        ipc.invalidate("planner.example");
        assert_eq!(request(address, &ipc.token, "planner.example"), "ERR");
    }

    #[test]
    fn invalidation_cannot_be_overtaken_by_an_in_flight_old_secret_read() {
        let secrets = Arc::new(Mutex::new(HashMap::from([(
            "planner.example".to_owned(),
            "old-token".to_owned(),
        )])));
        let read_started = Arc::new(Barrier::new(2));
        let release_read = Arc::new(Barrier::new(2));
        let first_read = Arc::new(AtomicBool::new(true));
        let resolver_secrets = Arc::clone(&secrets);
        let resolver_started = Arc::clone(&read_started);
        let resolver_release = Arc::clone(&release_read);
        let resolver_first = Arc::clone(&first_read);
        let resolver: SecretResolver = Arc::new(move |host| {
            let token = resolver_secrets.lock().unwrap().get(host).cloned()?;
            if resolver_first.swap(false, Ordering::SeqCst) {
                resolver_started.wait();
                resolver_release.wait();
            }
            Some((token, CredentialScheme::Bearer))
        });
        let cache: CredCache = Arc::new(Mutex::new(HashMap::new()));
        let lookup_cache = Arc::clone(&cache);
        let lookup_resolver = Arc::clone(&resolver);
        let lookup =
            std::thread::spawn(move || resolve("planner.example", &lookup_cache, &lookup_resolver));

        read_started.wait();
        assert!(
            cache.try_lock().is_err(),
            "the cache mutex must remain held until the old keychain read is inserted"
        );
        secrets
            .lock()
            .unwrap()
            .insert("planner.example".to_owned(), "new-token".to_owned());
        release_read.wait();
        assert!(lookup.join().unwrap().unwrap().contains("old-token"));

        invalidate_cached_secret(&cache, "planner.example");
        let refreshed = resolve("planner.example", &cache, &resolver).unwrap();
        assert!(refreshed.contains("new-token"));
        assert!(!refreshed.contains("old-token"));
    }
    #[test]
    fn partial_and_oversized_peers_do_not_block_other_credential_clients() {
        let resolver: SecretResolver =
            Arc::new(|_| Some(("secret".to_owned(), CredentialScheme::Bearer)));
        let ipc = start_tcp(resolver).unwrap();
        let address = match ipc.endpoint {
            CredIpcEndpoint::Tcp(address) => address,
            #[cfg(unix)]
            CredIpcEndpoint::Unix(_) => panic!("tcp"),
        };
        let mut slow = TcpStream::connect(address).unwrap();
        slow.write_all(b"partial").unwrap();
        let start = std::time::Instant::now();
        assert!(request(address, &ipc.token, "valid.example").contains("secret"));
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        let mut oversized = TcpStream::connect(address).unwrap();
        oversized
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        oversized
            .write_all(&vec![b'x'; MAX_REQUEST_BYTES as usize + 2])
            .unwrap();
        let mut reply = String::new();
        let closed = oversized.read_to_string(&mut reply);
        assert!(
            closed.is_ok()
                || closed.is_err_and(|error| error.kind() == std::io::ErrorKind::ConnectionReset)
        );
        assert!(reply.is_empty());
        assert_eq!(request(address, "bad", "valid.example"), "ERR");
        assert!(capability_matches(&ipc.token, &ipc.token));
        let mut wrong = ipc.token.clone().into_bytes();
        wrong[63] ^= 1;
        assert!(!capability_matches(
            std::str::from_utf8(&wrong).unwrap(),
            &ipc.token
        ));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_is_born_inside_private_directory() {
        let resolver: SecretResolver = Arc::new(|_| None);
        let ipc = start_unix(resolver).unwrap();
        let CredIpcEndpoint::Unix(ref path) = ipc.endpoint else {
            panic!("unix")
        };
        assert_eq!(
            std::fs::metadata(path.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let parent = path.parent().unwrap().to_owned();
        ipc.cleanup();
        assert!(!parent.exists());
    }
}
