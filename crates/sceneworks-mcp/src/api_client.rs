//! Thin HTTP client over the existing `/api/v1/*` surface (epic 10231).
//!
//! Modeled on `crates/sceneworks-worker/src/api_client.rs`: every MCP tool call
//! becomes a plain authenticated HTTP request against the SceneWorks API — there
//! is deliberately NO direct path into the engine/DB from the MCP server, so the
//! tools inherit exactly the behavior (and fixes) of the routes the web UI uses.

use serde_json::Value;

/// How the MCP tools reach the SceneWorks API. `base_url` normally comes from
/// `SCENEWORKS_API_URL` (the same variable the Rust worker uses) and the token is
/// the API's own `SCENEWORKS_ACCESS_TOKEN` — sent as `X-SceneWorks-Token`, the
/// header `access_control` accepts.
#[derive(Debug, Clone)]
pub struct ApiClientConfig {
    pub base_url: String,
    pub access_token: Option<String>,
}

#[derive(Clone)]
pub struct ApiClient {
    client: reqwest::Client,
    base_url: String,
    access_token: Option<String>,
}

#[derive(Debug)]
pub enum ApiClientError {
    /// Transport-level failure (connection refused, timeout, invalid body …).
    Http(reqwest::Error),
    /// The API answered with a non-2xx status; `detail` carries its body text.
    Api {
        status: reqwest::StatusCode,
        detail: String,
    },
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(error) => write!(f, "SceneWorks API request failed: {error}"),
            Self::Api { status, detail } => {
                write!(f, "SceneWorks API returned {status}: {detail}")
            }
        }
    }
}

impl std::error::Error for ApiClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(error) => Some(error),
            Self::Api { .. } => None,
        }
    }
}

impl From<reqwest::Error> for ApiClientError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl ApiClient {
    pub fn new(config: ApiClientConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: config.base_url.trim_end_matches('/').to_owned(),
            // An empty/whitespace token means auth is off — send no header, exactly
            // like the worker's `Settings::from_env` filter.
            access_token: config
                .access_token
                .map(|token| token.trim().to_owned())
                .filter(|token| !token.is_empty()),
        }
    }

    /// GET `path` (an absolute `/api/v1/...` path) with optional query pairs and
    /// decode the JSON body. Empty query values are skipped so optional tool
    /// arguments don't turn into `?modelFamily=` noise.
    pub async fn get_json(
        &self,
        path: &str,
        query: &[(&str, Option<&str>)],
    ) -> Result<Value, ApiClientError> {
        let mut request = self.client.get(format!("{}{}", self.base_url, path));
        for (key, value) in query {
            if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
                request = request.query(&[(*key, value)]);
            }
        }
        if let Some(token) = &self.access_token {
            request = request.header("X-SceneWorks-Token", token);
        }
        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "request failed".to_owned());
            return Err(ApiClientError::Api { status, detail });
        }
        Ok(response.json::<Value>().await?)
    }
}
