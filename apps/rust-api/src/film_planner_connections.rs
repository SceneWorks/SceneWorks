//! Saved OpenAI-compatible planning endpoints.
//!
//! Connection descriptors are non-secret settings. Tokens continue to live in the existing
//! server credential store or desktop keychain and are resolved only inside the backend.

use super::*;

use sceneworks_core::credentials::{normalize_host, CredentialFileStore};

const CONNECTIONS_SCHEMA_VERSION: u32 = 1;
const CONNECTIONS_FILENAME: &str = "film-planner-connections.json";
const ENDPOINT_POLICY_ENV: &str = "SCENEWORKS_FILM_PLANNER_ENDPOINT_POLICY";
const CRED_IPC_SOCKET_ENV: &str = "SCENEWORKS_CRED_IPC_SOCKET";
const CRED_IPC_TCP_ENV: &str = "SCENEWORKS_CRED_IPC_TCP";
const CRED_IPC_TOKEN_ENV: &str = "SCENEWORKS_CRED_IPC_TOKEN";
const MAX_CRED_IPC_RESPONSE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FilmPlannerConnection {
    pub schema_version: u32,
    pub id: String,
    pub label: String,
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_host: Option<String>,
    #[serde(default = "default_true")]
    pub supports_model_listing: bool,
    #[serde(default)]
    pub supports_image_input: bool,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
}

fn default_true() -> bool {
    true
}

fn default_timeout_seconds() -> u64 {
    60
}

fn default_max_output_tokens() -> u32 {
    8_192
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConnectionFile {
    #[serde(default = "connection_file_version")]
    schema_version: u32,
    #[serde(default)]
    connections: Vec<FilmPlannerConnection>,
}

fn connection_file_version() -> u32 {
    CONNECTIONS_SCHEMA_VERSION
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConnectionTestResult {
    ok: bool,
    capability: &'static str,
    detail: String,
    models: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SaveConnectionRequest {
    label: String,
    base_url: String,
    #[serde(default)]
    credential_host: Option<String>,
    #[serde(default = "default_true")]
    supports_model_listing: bool,
    #[serde(default)]
    supports_image_input: bool,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: u64,
    #[serde(default = "default_max_output_tokens")]
    max_output_tokens: u32,
}

pub(crate) async fn list_film_planner_connections(
    State(state): State<AppState>,
) -> Result<Json<Vec<FilmPlannerConnection>>, ApiError> {
    Ok(Json(read_connections(&state)?))
}

pub(crate) async fn save_film_planner_connection(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(payload): ApiJson<SaveConnectionRequest>,
) -> Result<Json<FilmPlannerConnection>, ApiError> {
    validate_id(&id)?;
    let base_url = validate_base_url(&payload.base_url)?;
    let credential_host = payload
        .credential_host
        .as_deref()
        .map(normalize_host)
        .filter(|host| !host.is_empty());
    if credential_host
        .as_deref()
        .is_some_and(|host| host != normalize_host(&base_url))
    {
        return Err(ApiError::bad_request(
            "A planning credential must belong to the connection base URL host",
        ));
    }
    if payload.label.trim().is_empty() {
        return Err(ApiError::bad_request("A connection label is required"));
    }
    if !(5..=300).contains(&payload.timeout_seconds) {
        return Err(ApiError::bad_request(
            "Connection timeout must be between 5 and 300 seconds",
        ));
    }
    if !(256..=65_536).contains(&payload.max_output_tokens) {
        return Err(ApiError::bad_request(
            "Maximum output tokens must be between 256 and 65536",
        ));
    }
    let connection = FilmPlannerConnection {
        schema_version: CONNECTIONS_SCHEMA_VERSION,
        id,
        label: payload.label.trim().to_owned(),
        base_url,
        credential_host,
        supports_model_listing: payload.supports_model_listing,
        supports_image_input: payload.supports_image_input,
        timeout_seconds: payload.timeout_seconds,
        max_output_tokens: payload.max_output_tokens,
    };
    let mut connections = read_connections(&state)?;
    connections.retain(|saved| saved.id != connection.id);
    connections.push(connection.clone());
    connections.sort_by(|left, right| left.label.cmp(&right.label).then(left.id.cmp(&right.id)));
    write_connections(&state, &connections)?;
    Ok(Json(connection))
}

pub(crate) async fn test_film_planner_connection(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ConnectionTestResult>, ApiError> {
    let connection = find_connection(&state, &id)?;
    if !connection.supports_model_listing {
        return Ok(Json(ConnectionTestResult {
            ok: true,
            capability: "manual_model",
            detail: "This endpoint is saved for manual model entry; model discovery is disabled."
                .to_owned(),
            models: Vec::new(),
        }));
    }
    let credential = resolve_connection_credential(&state, &connection).await?;
    let models =
        crate::openai_planner::list_models(&state.http_client, &connection, credential).await?;
    Ok(Json(ConnectionTestResult {
        ok: true,
        capability: "model_listing",
        detail: format!(
            "Model listing succeeded and exposed {} model{}. Chat Completions is validated when planning starts.",
            models.len(),
            if models.len() == 1 { "" } else { "s" }
        ),
        models,
    }))
}

pub(crate) fn find_connection(
    state: &AppState,
    id: &str,
) -> Result<FilmPlannerConnection, ApiError> {
    validate_id(id)?;
    let mut connection = read_connections(state)?
        .into_iter()
        .find(|connection| connection.id == id)
        .ok_or_else(|| {
            ApiError::bad_request(format!("Saved planning connection {id:?} was not found"))
        })?;
    connection.base_url = validate_base_url(&connection.base_url)?;
    Ok(connection)
}

pub(crate) async fn resolve_connection_credential(
    state: &AppState,
    connection: &FilmPlannerConnection,
) -> Result<Option<String>, ApiError> {
    validate_base_url(&connection.base_url)?;
    let Some(host) = connection.credential_host.as_deref() else {
        return Ok(None);
    };
    // A desktop-launched API must use the desktop's live OS-secret source as the
    // authority. Checking it before standalone sources ensures a keychain removal
    // cannot expose an older server-file or process-env value from the same config.
    if let Some(ipc) = DesktopCredentialIpc::from_env() {
        return Ok(credential_from_desktop_ipc(ipc, host).await);
    }
    if let Some(entry) = CredentialFileStore::new(&state.settings.credentials_dir)
        .load()
        .map_err(|error| {
            ApiError::internal(format!("Failed to read planning credential: {error}"))
        })?
        .remove(host)
    {
        return Ok((!entry.token.trim().is_empty()).then(|| entry.token.trim().to_owned()));
    }
    Ok(resolve_runtime_credential(
        host,
        None,
        std::env::var("SCENEWORKS_CREDENTIALS").ok().as_deref(),
    )
    .await)
}

/// Resolve a runtime credential after the server file store has been checked. A
/// configured desktop bridge is authoritative: missing, deleted, unauthorized, or
/// unavailable all fail closed instead of falling through to a stale process-env
/// snapshot. Standalone server mode has no bridge and retains its env behavior.
async fn resolve_runtime_credential(
    host: &str,
    desktop_ipc: Option<DesktopCredentialIpc>,
    credentials_env: Option<&str>,
) -> Option<String> {
    if let Some(ipc) = desktop_ipc {
        return credential_from_desktop_ipc(ipc, host).await;
    }
    credential_from_env(host, credentials_env)
}

fn credential_from_env(host: &str, credentials_env: Option<&str>) -> Option<String> {
    let value: Value = serde_json::from_str(credentials_env?).ok()?;
    value
        .get(host)
        .and_then(|entry| entry.get("token"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

enum DesktopCredentialEndpoint {
    #[cfg(unix)]
    Unix(PathBuf),
    Tcp(SocketAddr),
    Unavailable,
}

struct DesktopCredentialIpc {
    endpoint: DesktopCredentialEndpoint,
    auth: String,
}

impl DesktopCredentialIpc {
    fn from_env() -> Option<Self> {
        let socket = std::env::var(CRED_IPC_SOCKET_ENV).ok();
        let tcp = std::env::var(CRED_IPC_TCP_ENV).ok();
        let auth = std::env::var(CRED_IPC_TOKEN_ENV).ok();
        if socket.is_none() && tcp.is_none() && auth.is_none() {
            return None;
        }
        let auth = auth.unwrap_or_default().trim().to_owned();
        if auth.is_empty() {
            return Some(Self {
                endpoint: DesktopCredentialEndpoint::Unavailable,
                auth,
            });
        }
        #[cfg(unix)]
        if let Some(socket) = socket
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
        {
            return Some(Self {
                endpoint: DesktopCredentialEndpoint::Unix(PathBuf::from(socket)),
                auth,
            });
        }
        let Some(address) = tcp
            .as_deref()
            .and_then(|value| value.trim().parse::<SocketAddr>().ok())
        else {
            return Some(Self {
                endpoint: DesktopCredentialEndpoint::Unavailable,
                auth,
            });
        };
        if !address.ip().is_loopback() {
            return Some(Self {
                endpoint: DesktopCredentialEndpoint::Unavailable,
                auth,
            });
        }
        Some(Self {
            endpoint: DesktopCredentialEndpoint::Tcp(address),
            auth,
        })
    }
}

async fn credential_from_desktop_ipc(ipc: DesktopCredentialIpc, host: &str) -> Option<String> {
    let host = host.to_owned();
    tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let timeout = Some(Duration::from_secs(2));
        let request = format!("{} {host}\n", ipc.auth);
        let mut response = String::new();
        match ipc.endpoint {
            DesktopCredentialEndpoint::Unavailable => return None,
            #[cfg(unix)]
            DesktopCredentialEndpoint::Unix(socket) => {
                use std::os::unix::net::UnixStream;
                let mut stream = UnixStream::connect(socket).ok()?;
                let _ = stream.set_read_timeout(timeout);
                let _ = stream.set_write_timeout(timeout);
                stream.write_all(request.as_bytes()).ok()?;
                stream
                    .take(MAX_CRED_IPC_RESPONSE_BYTES + 1)
                    .read_to_string(&mut response)
                    .ok()?;
            }
            DesktopCredentialEndpoint::Tcp(address) => {
                let mut stream =
                    std::net::TcpStream::connect_timeout(&address, Duration::from_secs(2)).ok()?;
                let _ = stream.set_read_timeout(timeout);
                let _ = stream.set_write_timeout(timeout);
                stream.write_all(request.as_bytes()).ok()?;
                stream
                    .take(MAX_CRED_IPC_RESPONSE_BYTES + 1)
                    .read_to_string(&mut response)
                    .ok()?;
            }
        }
        if response.len() as u64 > MAX_CRED_IPC_RESPONSE_BYTES {
            return None;
        }
        let value: Value = serde_json::from_str(&response).ok()?;
        value
            .get("token")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_owned)
    })
    .await
    .ok()
    .flatten()
}

fn connections_path(state: &AppState) -> PathBuf {
    state.settings.config_dir.join(CONNECTIONS_FILENAME)
}

fn read_connections(state: &AppState) -> Result<Vec<FilmPlannerConnection>, ApiError> {
    let path = connections_path(state);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to read planning connections: {error}"
            )))
        }
    };
    let file: ConnectionFile = serde_json::from_slice(&bytes).map_err(|error| {
        ApiError::internal(format!("Planning connection settings are invalid: {error}"))
    })?;
    if file.schema_version != CONNECTIONS_SCHEMA_VERSION {
        return Err(ApiError::internal(format!(
            "Unsupported planning connection schema version {}",
            file.schema_version
        )));
    }
    Ok(file.connections)
}

fn write_connections(
    state: &AppState,
    connections: &[FilmPlannerConnection],
) -> Result<(), ApiError> {
    let path = connections_path(state);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            ApiError::internal(format!("Failed to create settings directory: {error}"))
        })?;
    }
    let body = serde_json::to_vec_pretty(&ConnectionFile {
        schema_version: CONNECTIONS_SCHEMA_VERSION,
        connections: connections.to_vec(),
    })
    .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp = path.with_extension(format!("json.{}.tmp", Uuid::new_v4().simple()));
    std::fs::write(&temp, body).map_err(|error| {
        ApiError::internal(format!("Failed to save planning connections: {error}"))
    })?;
    std::fs::rename(&temp, &path).map_err(|error| {
        ApiError::internal(format!("Failed to install planning connections: {error}"))
    })
}

fn validate_id(id: &str) -> Result<(), ApiError> {
    if id.is_empty()
        || id.len() > 80
        || !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(ApiError::bad_request("Invalid planning connection ID"));
    }
    Ok(())
}

pub(crate) fn validate_base_url(input: &str) -> Result<String, ApiError> {
    let mut url = reqwest::Url::parse(input.trim())
        .map_err(|_| ApiError::bad_request("Connection base URL must be a valid HTTP(S) URL"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ApiError::bad_request(
            "Connection base URL must use HTTP or HTTPS and include a host",
        ));
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ApiError::bad_request(
            "Connection base URL must not contain credentials, query parameters, or a fragment",
        ));
    }
    let local = url
        .host_str()
        .is_some_and(crate::film_planner::is_local_host);
    let local_only = std::env::var(ENDPOINT_POLICY_ENV)
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("local-only"));
    if local_only && !local {
        return Err(ApiError::bad_request(
            "The configured film-planner endpoint policy allows local/LAN hosts only",
        ));
    }
    if url.scheme() == "http" && !local {
        return Err(ApiError::bad_request(
            "Plain HTTP planning endpoints are allowed only on local or private-network hosts",
        ));
    }
    while url.path().ends_with('/') && url.path() != "/" {
        let trimmed = url.path().trim_end_matches('/').to_owned();
        url.set_path(&trimmed);
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{Ipv4Addr, TcpListener};
    use std::sync::{Arc, Mutex};

    #[test]
    fn connection_url_rejects_embedded_secrets_and_insecure_public_http() {
        assert!(validate_base_url("https://token@example.com/v1").is_err());
        assert!(validate_base_url("http://example.com/v1").is_err());
        assert_eq!(
            validate_base_url("http://127.0.0.1:8080/v1/").unwrap(),
            "http://127.0.0.1:8080/v1"
        );
    }

    fn start_live_credential_bridge(
        auth: &'static str,
        secrets: Arc<Mutex<HashMap<String, String>>>,
    ) -> SocketAddr {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind test bridge");
        let address = listener.local_addr().expect("test bridge address");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let mut reader = BufReader::new(stream);
                let mut request = String::new();
                if reader.read_line(&mut request).is_err() {
                    continue;
                }
                let mut parts = request.trim().splitn(2, ' ');
                let presented = parts.next().unwrap_or_default();
                let host = parts.next().unwrap_or_default();
                let response = if presented != auth {
                    "ERR".to_owned()
                } else if let Some(token) = secrets.lock().unwrap().get(host).cloned() {
                    serde_json::json!({ "token": token, "scheme": "bearer" }).to_string()
                } else {
                    "ERR".to_owned()
                };
                let mut stream = reader.into_inner();
                let _ = stream.write_all(response.as_bytes());
            }
        });
        address
    }

    fn tcp_ipc(address: SocketAddr, auth: &str) -> DesktopCredentialIpc {
        DesktopCredentialIpc {
            endpoint: DesktopCredentialEndpoint::Tcp(address),
            auth: auth.to_owned(),
        }
    }

    #[tokio::test]
    async fn live_desktop_resolver_is_authoritative_through_save_rotate_and_delete() {
        let secrets = Arc::new(Mutex::new(HashMap::new()));
        let address = start_live_credential_bridge("live-capability", Arc::clone(&secrets));
        let stale = r#"{"planner.example":{"token":"startup-token","scheme":"bearer"}}"#;

        assert_eq!(
            resolve_runtime_credential(
                "planner.example",
                Some(tcp_ipc(address, "live-capability")),
                Some(stale),
            )
            .await,
            None,
            "authoritative missing must not fall back to the startup snapshot"
        );

        secrets
            .lock()
            .unwrap()
            .insert("planner.example".to_owned(), "saved-token".to_owned());
        assert_eq!(
            resolve_runtime_credential(
                "planner.example",
                Some(tcp_ipc(address, "live-capability")),
                Some(stale),
            )
            .await
            .as_deref(),
            Some("saved-token")
        );

        secrets
            .lock()
            .unwrap()
            .insert("planner.example".to_owned(), "rotated-token".to_owned());
        assert_eq!(
            resolve_runtime_credential(
                "planner.example",
                Some(tcp_ipc(address, "live-capability")),
                Some(stale),
            )
            .await
            .as_deref(),
            Some("rotated-token")
        );

        assert_eq!(
            resolve_runtime_credential(
                "planner.example",
                Some(tcp_ipc(address, "wrong-capability")),
                Some(stale),
            )
            .await,
            None,
            "unauthorized bridge requests must fail closed"
        );

        secrets.lock().unwrap().remove("planner.example");
        assert_eq!(
            resolve_runtime_credential(
                "planner.example",
                Some(tcp_ipc(address, "live-capability")),
                Some(stale),
            )
            .await,
            None,
            "delete must stop the stale startup token without an API restart"
        );
    }
}
