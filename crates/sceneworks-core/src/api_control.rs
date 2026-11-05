//! Shared bounded API control-plane JSON transport. The desktop loopback client
//! disables proxies and redirects; worker adapters retain their existing client
//! policy. Callers retain endpoint authorization and response meaning.
use std::sync::OnceLock;
use std::time::Duration;

pub async fn post_json(
    api_url: &str,
    path: &str,
    body: &str,
    access_token: Option<&str>,
) -> Result<serde_json::Value, String> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let client = CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(5))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)?;
    // Retain the native loopback adapter's historical stricter header policy.
    // The worker adapter below keeps ordinary HeaderValue validation instead.
    let access_token = access_token.filter(|token| !token.chars().any(char::is_control));
    post_json_with_client(client, api_url, path, body, access_token).await
}

/// Share framing, authentication and response bounds while preserving a caller's
/// existing client/proxy/timeout policy (for workers connected to remote APIs).
pub async fn post_json_with_client(
    client: &reqwest::Client,
    api_url: &str,
    path: &str,
    body: &str,
    access_token: Option<&str>,
) -> Result<serde_json::Value, String> {
    if !path.starts_with('/') || path.starts_with("//") || path.chars().any(char::is_control) {
        return Err("Invalid API control path".to_owned());
    }
    let mut request = client
        .post(format!("{}{path}", api_url.trim_end_matches('/')))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_owned());
    if let Some(token) = access_token {
        request = request.header("X-SceneWorks-Token", token);
    }
    let mut response = request.send().await.map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("api responded {}", response.status().as_u16()));
    }
    const MAX_RESPONSE: usize = 64 * 1024;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        if chunk.len() > MAX_RESPONSE.saturating_sub(bytes.len()) {
            return Err("API control response exceeds 64 KiB".to_owned());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid API control JSON response".to_owned())
}
