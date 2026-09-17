//! Saved OpenAI-compatible planning endpoints.
//!
//! Connection descriptors are non-secret settings. Tokens continue to live in the existing
//! server credential store or desktop keychain and are resolved only inside the backend.

use super::*;

use sceneworks_core::credentials::{normalize_host, CredentialFileStore};

const CONNECTIONS_SCHEMA_VERSION: u32 = 1;
const CONNECTIONS_FILENAME: &str = "film-planner-connections.json";
const ENDPOINT_POLICY_ENV: &str = "SCENEWORKS_FILM_PLANNER_ENDPOINT_POLICY";

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
    if let Some(entry) = CredentialFileStore::new(&state.settings.credentials_dir)
        .load()
        .map_err(|error| {
            ApiError::internal(format!("Failed to read planning credential: {error}"))
        })?
        .remove(host)
    {
        return Ok((!entry.token.trim().is_empty()).then(|| entry.token.trim().to_owned()));
    }
    if let Some(token) = credential_from_env(host) {
        return Ok(Some(token));
    }
    #[cfg(unix)]
    if let Some(token) = credential_from_desktop_ipc(host).await {
        return Ok(Some(token));
    }
    Ok(None)
}

fn credential_from_env(host: &str) -> Option<String> {
    let value: Value = serde_json::from_str(&std::env::var("SCENEWORKS_CREDENTIALS").ok()?).ok()?;
    value
        .get(host)
        .and_then(|entry| entry.get("token"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

#[cfg(unix)]
async fn credential_from_desktop_ipc(host: &str) -> Option<String> {
    let socket = std::env::var("SCENEWORKS_CRED_IPC_SOCKET").ok()?;
    let auth = std::env::var("SCENEWORKS_CRED_IPC_TOKEN").ok()?;
    let host = host.to_owned();
    tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        let mut stream = UnixStream::connect(socket).ok()?;
        let timeout = Some(Duration::from_secs(2));
        let _ = stream.set_read_timeout(timeout);
        let _ = stream.set_write_timeout(timeout);
        stream
            .write_all(format!("{auth} {host}\n").as_bytes())
            .ok()?;
        let mut response = String::new();
        stream.read_to_string(&mut response).ok()?;
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

    #[test]
    fn connection_url_rejects_embedded_secrets_and_insecure_public_http() {
        assert!(validate_base_url("https://token@example.com/v1").is_err());
        assert!(validate_base_url("http://example.com/v1").is_err());
        assert_eq!(
            validate_base_url("http://127.0.0.1:8080/v1/").unwrap(),
            "http://127.0.0.1:8080/v1"
        );
    }
}
