//! Server/Docker credential management (sc-1893). Backs the same Settings
//! "Service credentials" screen used on desktop, but over HTTP against a `0600`
//! file (in `Settings::credentials_dir`, sc-16540) instead of the OS keychain. The
//! store deliberately does NOT live in the config dir — see
//! [`sceneworks_core::credentials::resolve_credentials_dir`]. The token is write-only —
//! `GET` never returns it, only host/label/scheme/present. Routes sit behind the
//! standard access-token middleware like the rest of `/api/*`.

use super::*;

use sceneworks_core::credentials::{normalize_host, CredentialFileStore, CredentialStatus};

fn credential_store(state: &AppState) -> CredentialFileStore {
    CredentialFileStore::new(&state.settings.credentials_dir)
}

/// Redacted listing of stored credentials — never includes tokens.
pub(crate) async fn list_credentials(
    State(state): State<AppState>,
) -> Result<Json<Vec<CredentialStatus>>, ApiError> {
    let store = credential_store(&state);
    let list = credential_blocking(move || {
        store
            .list()
            .map_err(|error| ApiError::internal(format!("Failed to read credentials: {error}")))
    })
    .await?;
    Ok(Json(list))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SetCredentialRequest {
    host: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    scheme: String,
    token: String,
}

/// Create or overwrite the credential for a host, returning the updated redacted
/// listing. The token is stored but never read back out over the API.
pub(crate) async fn set_credential(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<SetCredentialRequest>,
) -> Result<Json<Vec<CredentialStatus>>, ApiError> {
    let host = normalize_host(&payload.host);
    if host.is_empty() {
        return Err(ApiError::bad_request(
            "A host is required (e.g. huggingface.co).",
        ));
    }
    if payload.token.trim().is_empty() {
        return Err(ApiError::bad_request(
            "A token is required; use DELETE to remove a credential.",
        ));
    }
    let store = credential_store(&state);
    let list = credential_blocking(move || {
        store
            .set(&host, &payload.label, &payload.scheme, &payload.token)
            .map_err(|error| ApiError::internal(format!("Failed to save credential: {error}")))?;
        store
            .list()
            .map_err(|error| ApiError::internal(format!("Failed to read credentials: {error}")))
    })
    .await?;
    Ok(Json(list))
}

/// Remove a host's credential, returning the updated redacted listing. Removing an
/// absent host is a no-op.
pub(crate) async fn delete_credential(
    State(state): State<AppState>,
    Path(host): Path<String>,
) -> Result<Json<Vec<CredentialStatus>>, ApiError> {
    let store = credential_store(&state);
    let list = credential_blocking(move || {
        store
            .delete(&host)
            .map_err(|error| ApiError::internal(format!("Failed to delete credential: {error}")))?;
        store
            .list()
            .map_err(|error| ApiError::internal(format!("Failed to read credentials: {error}")))
    })
    .await?;
    Ok(Json(list))
}

async fn credential_blocking<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| ApiError::internal(format!("Credential operation failed: {error}")))?
}
