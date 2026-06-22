//! Per-host download credentials: parsing `SCENEWORKS_CREDENTIALS` and the file/env merge.
use super::*;

/// How a stored download credential authenticates to its host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialScheme {
    /// `Authorization: Bearer <token>`.
    Bearer,
    /// `?token=<token>` query parameter.
    Query,
}

/// A per-host download credential injected via `SCENEWORKS_CREDENTIALS`, matched
/// against LoRA/model `sourceUrl` hosts.
#[derive(Debug, Clone)]
pub struct WorkerCredential {
    pub host: String,
    pub token: String,
    pub scheme: CredentialScheme,
}

/// Parse the `SCENEWORKS_CREDENTIALS` env value: a JSON object mapping host to
/// `{ "token": "...", "scheme": "bearer" | "query" }`. Empty entries are skipped,
/// an unrecognized/absent scheme defaults to bearer, and invalid JSON yields none.
pub(crate) fn parse_credentials_env(raw: &str) -> Vec<WorkerCredential> {
    #[derive(serde::Deserialize)]
    struct RawCredential {
        token: String,
        #[serde(default)]
        scheme: Option<String>,
    }
    let parsed: std::collections::HashMap<String, RawCredential> =
        serde_json::from_str(raw).unwrap_or_default();
    parsed
        .into_iter()
        .filter_map(|(host, credential)| {
            let host = host.trim().to_ascii_lowercase();
            let token = credential.token.trim().to_owned();
            if host.is_empty() || token.is_empty() {
                return None;
            }
            let scheme = match credential.scheme.as_deref() {
                Some("query") => CredentialScheme::Query,
                _ => CredentialScheme::Bearer,
            };
            Some(WorkerCredential {
                host,
                token,
                scheme,
            })
        })
        .collect()
}

/// Merge two credential sets keyed by host, with `env` overriding `file` per host.
/// Desktop injects credentials via the env (from the keychain); server/Docker reads
/// the config-dir file store; an operator env override wins over the file.
pub(crate) fn merge_credentials(
    file_credentials: Vec<WorkerCredential>,
    env_credentials: Vec<WorkerCredential>,
) -> Vec<WorkerCredential> {
    let mut by_host: std::collections::HashMap<String, WorkerCredential> =
        std::collections::HashMap::new();
    for credential in file_credentials {
        by_host.insert(credential.host.clone(), credential);
    }
    for credential in env_credentials {
        by_host.insert(credential.host.clone(), credential);
    }
    by_host.into_values().collect()
}

/// Worker credentials from the server/Docker file store (`<config>/credentials.json`)
/// overlaid with the `SCENEWORKS_CREDENTIALS` env (desktop injection / operator
/// override). Same parser for both (the file carries an extra `label` the worker
/// ignores). Picked up at startup, so changing credentials needs a worker restart —
/// consistent with the desktop, which already re-injects on restart.
pub(crate) fn load_worker_credentials(config_dir: &Path) -> Vec<WorkerCredential> {
    let file = config_dir.join(sceneworks_core::credentials::CREDENTIALS_FILENAME);
    let file_credentials = std::fs::read_to_string(&file)
        .ok()
        .map(|body| parse_credentials_env(&body))
        .unwrap_or_default();
    let env_credentials = std::env::var("SCENEWORKS_CREDENTIALS")
        .ok()
        .map(|raw| parse_credentials_env(&raw))
        .unwrap_or_default();
    merge_credentials(file_credentials, env_credentials)
}
