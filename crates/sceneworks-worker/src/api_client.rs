use super::*;

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
            client: reqwest::Client::new(),
            api_url: settings.api_url.trim_end_matches('/').to_owned(),
            access_token: settings.access_token.clone(),
            worker_id: settings.worker_id.clone(),
        }
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
        let detail = response
            .text()
            .await
            .unwrap_or_else(|_| "request failed".to_owned());
        return Err(WorkerError::Api { status, detail });
    }
    Ok(response.json::<T>().await?)
}
