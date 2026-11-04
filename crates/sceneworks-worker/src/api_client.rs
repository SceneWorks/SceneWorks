use super::*;
use crate::downloads::HTTP_CONNECT_TIMEOUT;

/// Total-request timeout for non-streaming control-plane calls (register / claim /
/// progress). Safe here because these carry no large bodies, and it guarantees a hung
/// API server can never wedge the worker at `send().await` (sc-11149). Streaming
/// downloads deliberately use `downloads::HTTP_READ_TIMEOUT` instead of a total cap.
use sceneworks_core::network_policy::CONTROL_REQUEST_TIMEOUT as API_REQUEST_TIMEOUT;

#[derive(Clone)]
pub(crate) struct ApiClient {
    client: reqwest::Client,
    api_url: String,
    access_token: Option<String>,
    /// This worker's id, stamped onto every progress report so the server can
    /// reject writes from a worker that no longer owns the job (sc-4172).
    pub(crate) worker_id: String,
}

impl ApiClient {
    pub(crate) fn new(settings: &Settings) -> Self {
        Self {
            // Non-streaming control-plane calls (register / claim / progress). A total
            // `timeout` is safe here (no large bodies) and guarantees a hung API server
            // can never wedge the worker at `send().await` — reqwest's default is *no*
            // timeout (sc-11149).
            client: reqwest::Client::builder()
                .connect_timeout(HTTP_CONNECT_TIMEOUT)
                .timeout(API_REQUEST_TIMEOUT)
                .build()
                .expect("static API client config is always valid"),
            api_url: settings.api_url.trim_end_matches('/').to_owned(),
            access_token: settings.access_token.clone(),
            worker_id: settings.worker_id.clone(),
        }
    }

    /// Test-support dispatch reuses the existing bounded control-plane transport;
    /// it must not introduce another HTTP client factory outside this module.
    #[cfg(all(feature = "test-support", native_lane))]
    pub(crate) fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    pub(crate) async fn get_json<T>(&self, path: &str) -> WorkerResult<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let response = self
            .with_auth(self.client.get(self.url(path)))
            .send()
            .await?;
        decode_api_response(response).await
    }

    pub(crate) async fn post_json<T, U>(&self, path: &str, payload: &T) -> WorkerResult<U>
    where
        T: serde::Serialize + ?Sized,
        U: for<'de> Deserialize<'de>,
    {
        let response = self
            .with_auth(self.client.post(self.url(path)).json(payload))
            .send()
            .await?;
        decode_api_response(response).await
    }

    pub(crate) async fn post_control_json(&self, path: &str, body: &str) -> Result<Value, String> {
        sceneworks_core::api_control::post_json_with_client(
            &self.client,
            &self.api_url,
            path,
            body,
            self.access_token.as_deref(),
        )
        .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_url, path)
    }

    fn with_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.access_token {
            Some(token) => request.header("X-SceneWorks-Token", token),
            None => request,
        }
    }
}

async fn decode_api_response<T>(response: reqwest::Response) -> WorkerResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "request failed".to_owned());
        let envelope = serde_json::from_str::<Value>(&body).ok();
        let detail = envelope
            .as_ref()
            .and_then(|value| value.get("detail"))
            .and_then(Value::as_str)
            .unwrap_or(&body)
            .to_owned();
        let code = envelope
            .as_ref()
            .and_then(|value| value.get("code"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        return Err(WorkerError::Api {
            status,
            detail,
            code,
        });
    }
    Ok(response.json::<T>().await?)
}

#[cfg(test)]
mod control_tests {
    use std::io::{Read, Write};
    use std::time::Duration;

    #[tokio::test]
    async fn shared_control_post_preserves_worker_header_policy() {
        for desktop_policy in [false, true] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut chunk = [0; 1024];
                    let read = stream.read(&mut chunk).unwrap();
                    assert!(read > 0 && bytes.len() < 16 * 1024);
                    bytes.extend_from_slice(&chunk[..read]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        if bytes.len() >= end + 4 + 2 {
                            break;
                        }
                    }
                }
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnull",
                    )
                    .unwrap();
                String::from_utf8(bytes).unwrap().to_ascii_lowercase()
            });
            let token = "worker\ttoken";
            let result = if desktop_policy {
                sceneworks_core::api_control::post_json(&url, "/api/test", "{}", Some(token)).await
            } else {
                let client = reqwest::Client::builder()
                    .no_proxy()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .unwrap();
                sceneworks_core::api_control::post_json_with_client(
                    &client,
                    &url,
                    "/api/test",
                    "{}",
                    Some(token),
                )
                .await
            };
            assert!(result.unwrap().is_null());
            let request = server.join().unwrap();
            assert_eq!(
                request.contains("x-sceneworks-token: worker\ttoken"),
                !desktop_policy
            );
        }
    }
}
