//! OpenAI Chat Completions adapter for the shared film planner.
//!
//! This module owns protocol compatibility only. Validation, bounded repair, compilation, and
//! durable operation state remain in `film_planner`/`film_planning`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::{json, Value};

use sceneworks_core::film_compile::{PlannerExecutionRecord, PlannerUsageRecord};

use crate::film_harness::HarnessError;
use crate::film_planner::{LlmFuture, LlmReply, LlmRequest, PlannerLlm, PlannerReferenceImage};
use crate::film_planner_connections::FilmPlannerConnection;
use crate::ApiError;

const MAX_REFERENCE_IMAGES: usize = 8;
const MAX_REFERENCE_BYTES: u64 = 20 * 1024 * 1024;

type CancelRequested = Arc<dyn Fn() -> bool + Send + Sync>;

pub(crate) struct OpenAiPlannerOptions {
    pub model: String,
    pub thinking_mode: String,
    pub source_script: String,
    pub send_reference_pixels: bool,
}

pub(crate) struct OpenAiPlannerLlm {
    client: reqwest::Client,
    connection: FilmPlannerConnection,
    credential: Option<String>,
    model: String,
    thinking_mode: String,
    source_script: String,
    send_reference_pixels: bool,
    cancel_requested: CancelRequested,
}

impl OpenAiPlannerLlm {
    pub(crate) fn new(
        client: reqwest::Client,
        connection: FilmPlannerConnection,
        credential: Option<String>,
        options: OpenAiPlannerOptions,
        cancel_requested: CancelRequested,
    ) -> Result<Self, HarnessError> {
        if options.model.trim().is_empty() {
            return Err(HarnessError::Refused(
                "External planning requires an explicit model ID".to_owned(),
            ));
        }
        if options.send_reference_pixels && !connection.supports_image_input {
            return Err(HarnessError::Refused(
                "Reference pixels were enabled, but the selected connection is not marked as image-capable"
                    .to_owned(),
            ));
        }
        Ok(Self {
            client,
            connection,
            credential,
            model: options.model.trim().to_owned(),
            thinking_mode: options.thinking_mode,
            source_script: options.source_script,
            send_reference_pixels: options.send_reference_pixels,
            cancel_requested,
        })
    }
}

impl PlannerLlm for OpenAiPlannerLlm {
    fn complete(&self, request: LlmRequest) -> LlmFuture<'_> {
        Box::pin(async move {
            if (self.cancel_requested)() {
                return Err(HarnessError::Refused(
                    "planning canceled before the next external provider request".to_owned(),
                ));
            }
            let started = Instant::now();
            let outbound_prompt = format!(
                "# Original user script\n\n{}\n\n{}",
                self.source_script, request.prompt
            );
            let user_content = if self.send_reference_pixels {
                multimodal_content(&outbound_prompt, &request.reference_images)?
            } else {
                Value::String(outbound_prompt)
            };
            let mut body = json!({
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return the requested film plan as one JSON object. Keep any reasoning separate from the answer content."
                    },
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.2,
                "max_tokens": self.connection.max_output_tokens,
            });
            if self.thinking_mode == "enabled" {
                body["reasoning_effort"] = json!("medium");
            }
            let url = format!("{}/chat/completions", self.connection.base_url);
            let mut builder = self
                .client
                .post(url)
                .timeout(Duration::from_secs(self.connection.timeout_seconds));
            if let Some(token) = self.credential.as_deref() {
                builder = builder.bearer_auth(token);
            }
            let mut pending = Box::pin(builder.json(&body).send());
            let response = loop {
                tokio::select! {
                    result = &mut pending => break result.map_err(classify_transport_error)?,
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        if (self.cancel_requested)() {
                            return Err(HarnessError::Refused("planning canceled by user".to_owned()));
                        }
                    }
                }
            };
            let status = response.status();
            if !status.is_success() {
                return Err(classify_status(status));
            }
            let mut pending_body = Box::pin(response.json::<Value>());
            let value = loop {
                tokio::select! {
                    result = &mut pending_body => {
                        break result.map_err(|_| HarnessError::Transport(
                            "The external planner returned a malformed JSON protocol response".to_owned(),
                        ))?;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        if (self.cancel_requested)() {
                            return Err(HarnessError::Refused("planning canceled by user".to_owned()));
                        }
                    }
                }
            };
            let message = value.pointer("/choices/0/message").ok_or_else(|| {
                HarnessError::Transport(
                    "The external planner response has no choices[0].message".to_owned(),
                )
            })?;
            let text = message_text(message).ok_or_else(|| {
                HarnessError::Transport(
                    "The external planner response has no textual plan content".to_owned(),
                )
            })?;
            let thinking = message
                .get("reasoning_content")
                .or_else(|| message.get("thinking"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .filter(|value| !value.trim().is_empty());
            let usage = usage_record(value.get("usage"));
            Ok(LlmReply {
                text,
                thinking: thinking.clone(),
                job_id: None,
                execution: Some(PlannerExecutionRecord {
                    job_id: None,
                    provider: "openai_compatible".to_owned(),
                    model: self.model.clone(),
                    backend: Some(self.connection.id.clone()),
                    target_video_model_id: request.model_id.unwrap_or_default(),
                    thinking_mode: self.thinking_mode.clone(),
                    thinking,
                    usage,
                }),
                elapsed_seconds: started.elapsed().as_secs_f64(),
                peak_memory_bytes: None,
            })
        })
    }
}

pub(crate) async fn list_models(
    client: &reqwest::Client,
    connection: &FilmPlannerConnection,
    credential: Option<String>,
) -> Result<Vec<String>, ApiError> {
    let url = format!("{}/models", connection.base_url);
    let mut request = client
        .get(url)
        .timeout(Duration::from_secs(connection.timeout_seconds.min(30)));
    if let Some(token) = credential {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(|error| {
        if error.is_timeout() {
            ApiError::bad_request("The planning connection timed out while listing models")
        } else {
            ApiError::bad_request("The planning connection could not be reached")
        }
    })?;
    if !response.status().is_success() {
        return Err(ApiError::bad_request(status_detail(response.status())));
    }
    let value: Value = response.json().await.map_err(|_| {
        ApiError::bad_request("The planning endpoint returned a malformed model-list response")
    })?;
    let data = value.get("data").and_then(Value::as_array).ok_or_else(|| {
        ApiError::bad_request(
            "The planning endpoint does not expose the OpenAI-compatible /models capability; use manual model entry",
        )
    })?;
    let mut models = data
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str))
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    Ok(models)
}

fn multimodal_content(
    prompt: &str,
    images: &[PlannerReferenceImage],
) -> Result<Value, HarnessError> {
    if images.len() > MAX_REFERENCE_IMAGES {
        return Err(HarnessError::Refused(format!(
            "External planning allows at most {MAX_REFERENCE_IMAGES} reference images per request"
        )));
    }
    let mut content = vec![json!({"type": "text", "text": prompt})];
    let mut total = 0_u64;
    for image in images {
        let metadata = std::fs::metadata(&image.path)?;
        total = total.saturating_add(metadata.len());
        if total > MAX_REFERENCE_BYTES {
            return Err(HarnessError::Refused(format!(
                "External planning reference pixels exceed the {} MiB request limit",
                MAX_REFERENCE_BYTES / (1024 * 1024)
            )));
        }
        let bytes = std::fs::read(&image.path)?;
        let mime = match image
            .path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            _ => "image/png",
        };
        content.push(json!({
            "type": "text",
            "text": format!("Approved reference role: {}", image.role),
        }));
        content.push(json!({
            "type": "image_url",
            "image_url": {
                "url": format!(
                    "data:{mime};base64,{}",
                    base64::engine::general_purpose::STANDARD.encode(bytes)
                )
            }
        }));
    }
    Ok(Value::Array(content))
}

fn message_text(message: &Value) -> Option<String> {
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        return Some(text.to_owned()).filter(|text| !text.trim().is_empty());
    }
    let text = message
        .get("content")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    (!text.trim().is_empty()).then_some(text)
}

fn usage_record(usage: Option<&Value>) -> Option<PlannerUsageRecord> {
    let usage = usage?;
    let record = PlannerUsageRecord {
        input_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
    };
    (record.input_tokens.is_some()
        || record.output_tokens.is_some()
        || record.total_tokens.is_some())
    .then_some(record)
}

fn classify_transport_error(error: reqwest::Error) -> HarnessError {
    if error.is_timeout() {
        HarnessError::Transport(
            "The external planner timed out within the configured bound".to_owned(),
        )
    } else if error.is_connect() {
        HarnessError::Transport("The external planner could not be reached".to_owned())
    } else {
        HarnessError::Transport("The external planner request failed".to_owned())
    }
}

fn classify_status(status: reqwest::StatusCode) -> HarnessError {
    HarnessError::Transport(status_detail(status))
}

fn status_detail(status: reqwest::StatusCode) -> String {
    match status.as_u16() {
        401 | 403 => "The external planner rejected its credential (authentication failed)".to_owned(),
        400 | 404 | 405 | 415 | 422 => format!(
            "The external planner does not support the requested Chat Completions capability (HTTP {})",
            status.as_u16()
        ),
        408 | 504 => "The external planner timed out".to_owned(),
        429 => "The external planner is rate limited; no local provider was used instead".to_owned(),
        code => format!("The external planner failed with HTTP {code}; no local provider was used instead"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::{get, post};
    use axum::{Json, Router};

    #[derive(Clone)]
    struct Fixture {
        status: StatusCode,
        response: Value,
        delay: Duration,
        seen: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
    }

    async fn fixture_handler(
        State(fixture): State<Fixture>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        fixture.seen.lock().unwrap().push((headers, body));
        tokio::time::sleep(fixture.delay).await;
        (fixture.status, Json(fixture.response))
    }

    async fn fixture_models_handler(
        State(fixture): State<Fixture>,
        headers: HeaderMap,
    ) -> (StatusCode, Json<Value>) {
        fixture.seen.lock().unwrap().push((headers, Value::Null));
        tokio::time::sleep(fixture.delay).await;
        (fixture.status, Json(fixture.response))
    }

    async fn fixture(
        status: StatusCode,
        response: Value,
        delay: Duration,
    ) -> (String, Arc<Mutex<Vec<(HeaderMap, Value)>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(fixture_handler))
            .route("/v1/models", get(fixture_models_handler))
            .with_state(Fixture {
                status,
                response,
                delay,
                seen: Arc::clone(&seen),
            });
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}/v1"), seen)
    }

    fn connection(base_url: String) -> FilmPlannerConnection {
        FilmPlannerConnection {
            schema_version: 1,
            id: "fixture".to_owned(),
            label: "Fixture".to_owned(),
            base_url,
            credential_host: Some("fixture.test".to_owned()),
            supports_model_listing: true,
            supports_image_input: true,
            timeout_seconds: 5,
            max_output_tokens: 4096,
        }
    }

    fn request(images: Vec<PlannerReferenceImage>) -> LlmRequest {
        LlmRequest {
            task: Some("film_plan".to_owned()),
            prompt: "Return a plan".to_owned(),
            model_id: Some("minimax_h3".to_owned()),
            workflow: "video".to_owned(),
            guide: None,
            reference_images: images,
        }
    }

    fn options(thinking_mode: &str, send_reference_pixels: bool) -> OpenAiPlannerOptions {
        OpenAiPlannerOptions {
            model: "planner-model".to_owned(),
            thinking_mode: thinking_mode.to_owned(),
            source_script: "fixture script".to_owned(),
            send_reference_pixels,
        }
    }

    #[test]
    fn usage_is_optional_and_never_contains_provider_response_fields() {
        assert_eq!(usage_record(None), None);
        assert_eq!(
            usage_record(Some(
                &json!({"prompt_tokens": 12, "completion_tokens": 34, "secret": "no"})
            )),
            Some(PlannerUsageRecord {
                input_tokens: Some(12),
                output_tokens: Some(34),
                total_tokens: None,
            })
        );
    }

    #[test]
    fn thinking_is_not_part_of_plan_text() {
        let message = json!({"content": "{\"shots\":[]}", "reasoning_content": "private chain"});
        assert_eq!(message_text(&message).unwrap(), "{\"shots\":[]}");
        assert_eq!(message["reasoning_content"], "private chain");
    }

    #[tokio::test]
    async fn chat_completions_keeps_secret_out_of_body_and_records_sanitized_usage() {
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({
                "choices": [{"message": {"content": "{\"schemaVersion\":2}", "reasoning_content": "separate"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            }),
            Duration::ZERO,
        )
        .await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            Some("fixture-secret".to_owned()),
            OpenAiPlannerOptions {
                model: "planner-model".to_owned(),
                thinking_mode: "enabled".to_owned(),
                source_script: "A courier enters the workshop.".to_owned(),
                send_reference_pixels: false,
            },
            Arc::new(|| false),
        )
        .unwrap();
        let reply = llm.complete(request(Vec::new())).await.unwrap();
        assert_eq!(reply.text, "{\"schemaVersion\":2}");
        assert_eq!(reply.thinking.as_deref(), Some("separate"));
        let execution = reply.execution.unwrap();
        assert_eq!(execution.provider, "openai_compatible");
        assert_eq!(execution.backend.as_deref(), Some("fixture"));
        assert_eq!(execution.target_video_model_id, "minimax_h3");
        assert_eq!(execution.usage.unwrap().total_tokens, Some(18));
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen[0].0.get("authorization").unwrap(),
            "Bearer fixture-secret"
        );
        assert!(seen[0].1.to_string().contains("planner-model"));
        assert!(seen[0]
            .1
            .to_string()
            .contains("A courier enters the workshop."));
        assert!(!seen[0].1.to_string().contains("fixture-secret"));
    }

    #[tokio::test]
    async fn chat_completions_runs_through_shared_validation_and_compile() {
        use crate::tests::film_harness::{draft_text, full_draft, planner_options, Harness};

        let harness = Harness::start(false, vec![]).await;
        let draft = full_draft();
        let expected_shots = draft["shots"].as_array().unwrap().len();
        let (base_url, _) = fixture(
            StatusCode::OK,
            json!({
                "choices": [{"message": {"content": draft_text(&draft)}}],
                "usage": {"prompt_tokens": 101, "completion_tokens": 202, "total_tokens": 303}
            }),
            Duration::ZERO,
        )
        .await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        let mut planner = planner_options(&harness, "external-planned");
        planner.require_local_planner = false;
        planner.max_repair_rounds = 1;

        let artifacts = crate::film_planner::generate(&harness.transport, &llm, &planner)
            .await
            .expect("external Chat Completions draft validates and compiles");
        assert_eq!(artifacts.repair_rounds, 0);
        assert_eq!(artifacts.plan.shots.len(), expected_shots);
        assert!(artifacts.plan_path.is_file());
        assert!(artifacts.compiled_path.is_file());
        let execution = &artifacts.compiled.planner.as_ref().unwrap().executions[0];
        assert_eq!(execution.provider, "openai_compatible");
        assert_eq!(execution.model, "planner-model");
        assert_eq!(execution.target_video_model_id, "minimax_h3");
        assert_eq!(execution.usage.as_ref().unwrap().total_tokens, Some(303));
    }

    #[tokio::test]
    async fn model_listing_is_optional_and_returns_stable_manual_choices() {
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({"data": [{"id": "z-model"}, {"id": "a-model"}, {"id": "a-model"}]}),
            Duration::ZERO,
        )
        .await;
        let models = list_models(
            &reqwest::Client::new(),
            &connection(base_url),
            Some("fixture-secret".to_owned()),
        )
        .await
        .unwrap();
        assert_eq!(models, vec!["a-model", "z-model"]);
        assert_eq!(
            seen.lock().unwrap()[0].0.get("authorization").unwrap(),
            "Bearer fixture-secret"
        );
    }

    #[tokio::test]
    async fn reference_pixels_require_both_user_opt_in_and_image_capability() {
        let image = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(image.path(), b"pixels").unwrap();
        let image_request = || {
            request(vec![PlannerReferenceImage {
                role: "hero".to_owned(),
                path: image.path().to_path_buf(),
            }])
        };
        let response = json!({"choices": [{"message": {"content": "{}"}}]});

        let (base_url, seen) = fixture(StatusCode::OK, response.clone(), Duration::ZERO).await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        llm.complete(image_request()).await.unwrap();
        assert!(!seen.lock().unwrap()[0].1.to_string().contains("data:image"));

        let (base_url, seen) = fixture(StatusCode::OK, response, Duration::ZERO).await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", true),
            Arc::new(|| false),
        )
        .unwrap();
        llm.complete(image_request()).await.unwrap();
        assert!(seen.lock().unwrap()[0].1.to_string().contains("data:image"));

        let mut incapable = connection("http://127.0.0.1:1/v1".to_owned());
        incapable.supports_image_input = false;
        assert!(OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            incapable,
            None,
            options("disabled", true),
            Arc::new(|| false),
        )
        .is_err());
    }

    #[tokio::test]
    async fn authentication_capability_timeout_malformed_and_cancel_are_bounded() {
        for (status, expected) in [
            (StatusCode::UNAUTHORIZED, "authentication failed"),
            (StatusCode::UNPROCESSABLE_ENTITY, "does not support"),
        ] {
            let (base_url, _) =
                fixture(status, json!({"secret": "never echoed"}), Duration::ZERO).await;
            let llm = OpenAiPlannerLlm::new(
                reqwest::Client::new(),
                connection(base_url),
                None,
                options("disabled", false),
                Arc::new(|| false),
            )
            .unwrap();
            let error = llm.complete(request(Vec::new())).await.unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert!(!error.to_string().contains("never echoed"));
        }

        let (base_url, _) =
            fixture(StatusCode::OK, json!({"notChoices": []}), Duration::ZERO).await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        assert!(llm
            .complete(request(Vec::new()))
            .await
            .unwrap_err()
            .to_string()
            .contains("no choices"));

        let (base_url, _) = fixture(
            StatusCode::OK,
            json!({"choices": [{"message": {"content": "{}"}}]}),
            Duration::from_secs(2),
        )
        .await;
        let mut timed = connection(base_url);
        timed.timeout_seconds = 1;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            timed,
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        assert!(llm
            .complete(request(Vec::new()))
            .await
            .unwrap_err()
            .to_string()
            .contains("timed out"));

        let canceled = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection("http://127.0.0.1:1/v1".to_owned()),
            None,
            options("disabled", false),
            Arc::new(|| true),
        )
        .unwrap();
        assert!(canceled
            .complete(request(Vec::new()))
            .await
            .unwrap_err()
            .to_string()
            .contains("canceled before"));

        let (base_url, _) = fixture(
            StatusCode::OK,
            json!({"choices": [{"message": {"content": "{}"}}]}),
            Duration::from_secs(2),
        )
        .await;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_check = Arc::clone(&cancel);
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", false),
            Arc::new(move || cancel_check.load(Ordering::SeqCst)),
        )
        .unwrap();
        let cancel_later = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            cancel_later.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        assert!(llm
            .complete(request(Vec::new()))
            .await
            .unwrap_err()
            .to_string()
            .contains("canceled by user"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
