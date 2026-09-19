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
type RequestStarted = Arc<dyn Fn() + Send + Sync>;
type ExecutionUpdated =
    Arc<dyn Fn(&PlannerExecutionRecord) -> Result<(), HarnessError> + Send + Sync>;

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
    request_started: Option<RequestStarted>,
    execution_updated: Option<ExecutionUpdated>,
}

impl OpenAiPlannerLlm {
    pub(crate) fn new(
        client: reqwest::Client,
        connection: FilmPlannerConnection,
        credential: Option<String>,
        options: OpenAiPlannerOptions,
        cancel_requested: CancelRequested,
    ) -> Result<Self, HarnessError> {
        crate::film_planner_connections::validate_base_url(&connection.base_url)
            .map_err(|error| HarnessError::Refused(error.detail))?;
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
            request_started: None,
            execution_updated: None,
        })
    }

    pub(crate) fn on_execution_updated(mut self, callback: ExecutionUpdated) -> Self {
        self.execution_updated = Some(callback);
        self
    }

    fn record_execution(&self, execution: &PlannerExecutionRecord) -> Result<(), HarnessError> {
        if let Some(callback) = &self.execution_updated {
            callback(execution)?;
        }
        Ok(())
    }

    pub(crate) fn on_request_started(mut self, callback: RequestStarted) -> Self {
        self.request_started = Some(callback);
        self
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
            crate::film_planner_connections::validate_base_url(&self.connection.base_url)
                .map_err(|error| HarnessError::Refused(error.detail))?;
            if request.task.as_deref() != Some(crate::film_planner::FILM_PLAN_TASK) {
                return Err(HarnessError::Refused(
                    "The external film planner cannot perform target-model prompt refinement"
                        .to_owned(),
                ));
            }
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
            // Recheck immediately before dispatch, after optional file reads. A canceled request
            // has neither a POST nor an invented attempt receipt.
            if (self.cancel_requested)() {
                return Err(HarnessError::Refused(
                    "planning canceled before external dispatch".to_owned(),
                ));
            }
            let started = Instant::now();
            let mut execution = PlannerExecutionRecord {
                provider: "openai_compatible".to_owned(),
                model: self.model.clone(),
                backend: Some(self.connection.id.clone()),
                target_video_model_id: request.model_id.unwrap_or_default(),
                thinking_mode: self.thinking_mode.clone(),
                max_output_tokens: Some(self.connection.max_output_tokens),
                request_timeout_seconds: Some(self.connection.timeout_seconds),
                temperature: Some(0.2),
                reference_pixels_sent: Some(
                    self.send_reference_pixels && !request.reference_images.is_empty(),
                ),
                duration_seconds: Some(0.0),
                failure_code: Some("dispatch_started".to_owned()),
                ..PlannerExecutionRecord::default()
            };
            // This records a client dispatch attempt, not evidence that the server received it.
            self.record_execution(&execution)?;
            if let Some(callback) = &self.request_started {
                callback();
            }
            let result: Result<String, (&str, HarnessError)> = async {
                let mut pending = Box::pin(builder.json(&body).send());
                let response = loop {
                    tokio::select! {
                        result = &mut pending => break result.map_err(|error| {
                            let code = if error.is_timeout() { "transport_timeout" }
                                else if error.is_connect() { "connection_failed" }
                                else { "transport_error" };
                            (code, classify_transport_error(error))
                        })?,
                        _ = tokio::time::sleep(Duration::from_millis(25)) => {
                            if (self.cancel_requested)() {
                                return Err(("canceled", HarnessError::Refused("planning canceled by user".to_owned())));
                            }
                        }
                    }
                };
                let status = response.status();
                let http_failure = (!status.is_success()).then(|| {
                    match status.as_u16() {
                        401 | 403 => "authentication_failed",
                        400 | 404 | 405 | 415 | 422 => "unsupported_capability",
                        408 | 504 => "provider_timeout",
                        429 => "rate_limited",
                        _ => "http_error",
                    }
                });
                let mut pending_body = Box::pin(response.json::<Value>());
                let value = loop {
                    tokio::select! {
                        result = &mut pending_body => break result.map_err(|error| {
                            if error.is_timeout() {
                                ("transport_timeout", classify_transport_error(error))
                            } else if let Some(code) = http_failure {
                                (code, classify_status(status))
                            } else {
                                ("malformed_protocol_json", HarnessError::Transport(
                                    "The external planner returned a malformed JSON protocol response".to_owned()))
                            }
                        })?,
                        _ = tokio::time::sleep(Duration::from_millis(25)) => {
                            if (self.cancel_requested)() {
                                return Err(("canceled", HarnessError::Refused("planning canceled by user".to_owned())));
                            }
                        }
                    }
                };
                execution.usage = usage_record(value.get("usage"));
                if let Some(code) = http_failure { return Err((code, classify_status(status))); }
                execution.finish_reason = sanitized_finish_reason(value.pointer("/choices/0/finish_reason"));
                let message = value.pointer("/choices/0/message").ok_or_else(|| (
                    "missing_message", HarnessError::Transport(
                        "The external planner response has no choices[0].message".to_owned())))?;
                execution.thinking = message.get("reasoning_content")
                    .or_else(|| message.get("thinking"))
                    .and_then(Value::as_str).map(str::to_owned)
                    .filter(|value| !value.trim().is_empty());
                message_text(message).ok_or_else(|| (
                    "no_textual_plan_content", HarnessError::Transport(
                        "The external planner response has no textual plan content".to_owned())))
            }.await;
            execution.duration_seconds = Some(started.elapsed().as_secs_f64());
            execution.failure_code = result.as_ref().err().map(|(code, _)| (*code).to_owned());
            if let Err(source) = self.record_execution(&execution) {
                return Err(HarnessError::PlannerExecutionFailure {
                    source: Box::new(source),
                    executions: vec![execution],
                });
            }
            match result {
                Ok(text) => Ok(LlmReply {
                    text,
                    thinking: execution.thinking.clone(),
                    elapsed_seconds: execution.duration_seconds.unwrap_or_default(),
                    execution: Some(execution),
                    ..LlmReply::default()
                }),
                Err(("no_textual_plan_content", source)) => Err(HarnessError::PlannerResponse {
                    detail: match source {
                        HarnessError::Transport(detail) => detail,
                        _ => unreachable!(),
                    },
                    execution: Box::new(execution),
                }),
                Err((_, source)) => Err(HarnessError::PlannerExecutionFailure {
                    source: Box::new(source),
                    executions: vec![execution],
                }),
            }
        })
    }
}

pub(crate) async fn list_models(
    client: &reqwest::Client,
    connection: &FilmPlannerConnection,
    credential: Option<String>,
) -> Result<Vec<String>, ApiError> {
    let base_url = crate::film_planner_connections::validate_base_url(&connection.base_url)?;
    let url = format!("{base_url}/models");
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
        // Every role this ONE image carries (sc-24024): a photograph holding two people is one
        // image labelled for both subjects, each with the locator that picks it out, rather than
        // the same bytes sent twice under two role names.
        let roles = image
            .roles
            .iter()
            .map(|role| match role.locator.as_deref() {
                Some(locator) => format!("{} ({locator})", role.role),
                None => role.role.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        content.push(json!({
            "type": "text",
            "text": format!(
                "Approved reference role{}: {roles}",
                if image.roles.len() == 1 { "" } else { "s" }
            ),
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

fn sanitized_finish_reason(value: Option<&Value>) -> Option<String> {
    let reason = value?.as_str()?.trim();
    if reason.is_empty() {
        return None;
    }
    Some(
        match reason {
            "stop" | "length" | "tool_calls" | "function_call" | "content_filter" => reason,
            _ => "other",
        }
        .to_owned(),
    )
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
    use crate::film_planner::PlannerReferenceRole;

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
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        fixture.seen.lock().unwrap().push((headers, body));
        tokio::time::sleep(fixture.delay).await;
        if fixture.response == json!("malformed-protocol-fixture") {
            (fixture.status, "{invalid protocol").into_response()
        } else {
            (fixture.status, Json(fixture.response)).into_response()
        }
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
        assert_eq!(execution.max_output_tokens, Some(4096));
        assert_eq!(execution.reference_pixels_sent, Some(false));
        assert!(execution.duration_seconds.is_some());
        assert_eq!(execution.failure_code, None);
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
    async fn empty_text_fails_once_and_preserves_sanitized_response_provenance() {
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": null, "reasoning_content": "bounded reasoning"}
                }],
                "usage": {"prompt_tokens": 19, "completion_tokens": 4096, "total_tokens": 4115},
                "provider_debug": {"credential": "must-not-survive"}
            }),
            Duration::ZERO,
        )
        .await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            Some("fixture-secret".to_owned()),
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();

        let error = llm.complete(request(Vec::new())).await.unwrap_err();
        let HarnessError::PlannerResponse { detail, execution } = error else {
            panic!("expected sanitized planner-response failure, got {error}");
        };
        assert_eq!(
            detail,
            "The external planner response has no textual plan content"
        );
        assert_eq!(execution.provider, "openai_compatible");
        assert_eq!(execution.model, "planner-model");
        assert_eq!(execution.backend.as_deref(), Some("fixture"));
        assert_eq!(execution.target_video_model_id, "minimax_h3");
        assert_eq!(execution.thinking_mode, "disabled");
        assert_eq!(execution.max_output_tokens, Some(4096));
        assert_eq!(execution.reference_pixels_sent, Some(false));
        assert!(execution.duration_seconds.is_some_and(|value| value >= 0.0));
        assert_eq!(execution.finish_reason.as_deref(), Some("length"));
        assert_eq!(
            execution.failure_code.as_deref(),
            Some("no_textual_plan_content")
        );
        assert_eq!(execution.thinking.as_deref(), Some("bounded reasoning"));
        assert_eq!(execution.usage.as_ref().unwrap().total_tokens, Some(4115));
        let serialized = serde_json::to_string(&execution).unwrap();
        assert!(!serialized.contains("must-not-survive"));
        assert!(!serialized.contains("fixture-secret"));

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "empty content must not retry or fall back");
        assert_eq!(seen[0].1["model"], "planner-model");
        assert!(seen[0].1["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Return a plan"));
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
    async fn external_plan_uses_native_model_keyed_refinement_and_recovers_without_resending() {
        use crate::tests::film_harness::{
            draft_text, full_draft, planner_llm, planner_options, refine_job_payloads, Harness,
        };
        let harness = Harness::start(true, vec![]).await;
        harness.script.lock().refine_template = Some("Native shot rewrite: {prompt}".to_owned());
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({"choices": [{"message": {"content": draft_text(&full_draft())}}]}),
            Duration::ZERO,
        )
        .await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("enabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        let mut config = planner_options(&harness, "external-refined");
        config.require_local_planner = false;
        config.refine_prompts = true;
        let guide = harness.temp_dir.path().join("guide.txt");
        std::fs::write(&guide, "Exact target model guide").unwrap();
        config.prompt_guide_path = Some(guide);
        let artifacts = crate::film_planner::generate_with_refiner(
            &harness.transport,
            &llm,
            &planner_llm(&harness),
            &config,
        )
        .await
        .unwrap();
        let count = artifacts.plan.shots.len();
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "only film planning reaches the external provider"
        );
        let payloads = refine_job_payloads(&harness, false);
        assert_eq!(payloads.len(), count);
        for payload in &payloads {
            assert_ne!(payload["task"], "film_plan");
            assert_eq!(payload["modelId"], "minimax_h3");
            assert_eq!(payload["guide"], "Exact target model guide");
            assert_eq!(payload["workflow"], "video");
        }
        let cost = artifacts.compiled.planner.as_ref().unwrap();
        assert_eq!(cost.executions.len(), 1 + count);
        assert_eq!(cost.executions[0].provider, "openai_compatible");
        for execution in &cost.executions[1..] {
            assert_eq!(execution.provider, "native");
            assert_eq!(execution.model, "fixture/model-keyed-refiner");
            assert_eq!(execution.target_video_model_id, "minimax_h3");
            assert_eq!(execution.request_timeout_seconds, Some(30));
        }
        for request in &artifacts.compiled.requests {
            // CONTAINED, not leading: since sc-24025 the compiler's identity text leads any shot
            // that names a continuity role it does not bind to an image, and these shots bind
            // nothing. What this test is about is that the NATIVE refiner produced the text, so it
            // asserts the rewrite survived and that only recorded inserted text precedes it.
            let rewrite_at = request
                .prompt
                .find("Native shot rewrite:")
                .unwrap_or_else(|| panic!("{}: {}", request.shot_id, request.prompt));
            let leading: String = request
                .inserted_text
                .iter()
                .filter(|piece| {
                    piece.kind.placement()
                        == sceneworks_core::film_compile::InsertedTextPlacement::Leading
                })
                .map(|piece| piece.text.clone())
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(
                request.prompt[..rewrite_at].trim(),
                leading.trim(),
                "{}: only the compiler's own leading text precedes the rewrite",
                request.shot_id
            );
        }
        let recovered = crate::film_planner::compile_existing_with_executions(
            &harness.transport,
            &planner_llm(&harness).adopt_jobs(cost.job_ids.clone()),
            &config,
            &config.plan_path(),
            vec![cost.executions[0].clone()],
        )
        .await
        .unwrap();
        assert_eq!(
            refine_job_payloads(&harness, false).len(),
            count,
            "recovery adopts exact native jobs"
        );
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "recovery never resends external planning"
        );
        assert_eq!(recovered.compiled.requests, artifacts.compiled.requests);
        assert_eq!(
            recovered.compiled.planner.unwrap().executions.len(),
            count + 1
        );
    }

    #[tokio::test]
    async fn unavailable_native_refiner_preserves_external_plan_and_receipt_without_fallback() {
        use crate::tests::film_harness::{
            draft_text, full_draft, planner_llm, planner_options, Harness,
        };
        let harness = Harness::start(false, vec![]).await;
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({"choices": [{"message": {"content": draft_text(&full_draft())}}]}),
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
        let mut config = planner_options(&harness, "external-unavailable-refiner");
        config.require_local_planner = false;
        config.refine_prompts = true;
        let error = crate::film_planner::generate_with_refiner(
            &harness.transport,
            &llm,
            &planner_llm(&harness),
            &config,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("prompt_refine"), "{error}");
        let HarnessError::PlannerExecutionFailure { executions, .. } = error else {
            panic!("missing receipt")
        };
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].provider, "openai_compatible");
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert!(
            config.plan_path().exists(),
            "validated plan remains manually editable"
        );
        assert!(!config.compiled_path().exists());
    }

    #[tokio::test]
    async fn persisted_connections_obey_current_local_only_policy_before_credentials_or_network() {
        const CHILD: &str = "SCENEWORKS_POLICY_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "openai_planner::tests::persisted_connections_obey_current_local_only_policy_before_credentials_or_network", "--nocapture"])
                .env(CHILD, "1").env("SCENEWORKS_FILM_PLANNER_ENDPOINT_POLICY", "local-only")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        use crate::tests::support::{request as api_request, test_settings};
        let temp = tempfile::tempdir().unwrap();
        let settings = test_settings(&temp);
        let (app, state) = crate::create_app_with_state(settings.clone()).unwrap();
        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({"data": [{"id": "local-model"}]}),
            Duration::ZERO,
        )
        .await;
        let mut local = connection(base_url);
        local.credential_host = None;
        let mut public = connection("https://public.fixture.test/v1".to_owned());
        public.id = "public".to_owned();
        std::fs::create_dir_all(&settings.config_dir).unwrap();
        std::fs::write(
            settings.config_dir.join("film-planner-connections.json"),
            serde_json::to_vec(&json!({"schemaVersion": 1, "connections": [local, public]}))
                .unwrap(),
        )
        .unwrap();
        // Corrupt secret storage proves policy rejection precedes even credential resolution.
        std::fs::create_dir_all(&settings.credentials_dir).unwrap();
        std::fs::write(
            settings.credentials_dir.join("credentials.json"),
            b"broken secret store",
        )
        .unwrap();
        let error = crate::film_planner_connections::resolve_connection_credential(&state, &public)
            .await
            .unwrap_err();
        assert!(error.detail.contains("local/LAN"));
        let (status, error) = api_request(
            app.clone(),
            "POST",
            "/api/v1/film-planner-connections/public/test",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"].as_str().unwrap().contains("local/LAN"));
        assert!(list_models(
            &reqwest::Client::new(),
            &public,
            Some("must-not-send".to_owned())
        )
        .await
        .unwrap_err()
        .detail
        .contains("local/LAN"));
        assert!(
            matches!(OpenAiPlannerLlm::new(reqwest::Client::new(), public, Some("must-not-send".to_owned()),
            options("enabled", true), Arc::new(|| false)), Err(HarnessError::Refused(detail)) if detail.contains("local/LAN"))
        );
        // A descriptor already held by an adapter is checked again before every use.
        let mut held = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            local.clone(),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        held.connection = connection("https://public.fixture.test/v1".to_owned());
        held.credential = Some("must-not-send".to_owned());
        held = held.on_execution_updated(Arc::new(|_| panic!("policy refusal precedes dispatch")));
        assert!(matches!(held.complete(request(Vec::new())).await,
            Err(HarnessError::Refused(detail)) if detail.contains("local/LAN")));
        let (_, project) = api_request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({"name": "Policy test"}),
        )
        .await;
        let project_id = project["id"].as_str().unwrap();
        let (_, mut draft) = api_request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/films"),
            json!({"title": "Policy draft"}),
        )
        .await;
        let draft_id = draft["id"].as_str().unwrap().to_owned();
        draft["originalScript"] = json!("Private script must stay local");
        draft["planning"] = json!({"provider":"openai_compatible", "connectionId":"public", "modelId":"public-model", "refinePrompts":false, "sendReferencePixels":true, "thinkingMode":"disabled"});
        let (status, saved) = api_request(
            app.clone(),
            "PUT",
            &format!("/api/v1/projects/{project_id}/films/{draft_id}"),
            draft,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        let (status, error) = api_request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/films/{draft_id}/planning"),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(error["detail"].as_str().unwrap().contains("local/LAN"));
        assert!(
            seen.lock().unwrap().is_empty(),
            "refusals issue zero network requests"
        );
        let (status, response) = api_request(
            app,
            "POST",
            "/api/v1/film-planner-connections/fixture/test",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(response["models"], json!(["local-model"]));
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
                roles: vec![PlannerReferenceRole {
                    role: "hero".to_owned(),
                    locator: None,
                }],
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

    /// One photograph of two people is sent ONCE, labelled for both subjects with the locator that
    /// picks each out (sc-24024). The label is the only thing telling the external planner that
    /// the two roles are two subjects in one image rather than two pictures, so it is asserted
    /// verbatim — including the plural, which is what says a single-role image is labelled
    /// differently from a shared one.
    #[tokio::test]
    async fn a_shared_reference_image_is_sent_once_and_labelled_for_every_role_it_carries() {
        let image = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(image.path(), b"pixels").unwrap();
        let lone = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(lone.path(), b"pixels").unwrap();

        let (base_url, seen) = fixture(
            StatusCode::OK,
            json!({"choices": [{"message": {"content": "{}"}}]}),
            Duration::ZERO,
        )
        .await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", true),
            Arc::new(|| false),
        )
        .unwrap();
        llm.complete(request(vec![
            PlannerReferenceImage {
                roles: vec![
                    PlannerReferenceRole {
                        role: "courier".to_owned(),
                        locator: Some("the woman on the left".to_owned()),
                    },
                    PlannerReferenceRole {
                        role: "recipient".to_owned(),
                        locator: Some("the man on the right".to_owned()),
                    },
                ],
                path: image.path().to_path_buf(),
            },
            PlannerReferenceImage {
                roles: vec![PlannerReferenceRole {
                    role: "red_parcel".to_owned(),
                    locator: None,
                }],
                path: lone.path().to_path_buf(),
            },
        ]))
        .await
        .unwrap();

        let body = seen.lock().unwrap()[0].1.to_string();
        assert!(
            body.contains(
                "Approved reference roles: courier (the woman on the left), recipient (the man \
                 on the right)"
            ),
            "{body}"
        );
        assert!(
            body.contains("Approved reference role: red_parcel"),
            "a lone role is labelled in the singular and carries no locator: {body}"
        );
        assert_eq!(
            body.matches("data:image").count(),
            2,
            "two files, two images — the shared photograph is sent once: {body}"
        );
    }

    #[tokio::test]
    async fn every_dispatched_failure_has_a_sanitized_durable_attempt_and_cancel_class() {
        for (status, body, delay, cancel, code) in [
            (
                StatusCode::UNAUTHORIZED,
                json!({"secret": "never-retain", "usage": {"prompt_tokens": 17}}),
                Duration::ZERO,
                false,
                "authentication_failed",
            ),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({}),
                Duration::ZERO,
                false,
                "unsupported_capability",
            ),
            (
                StatusCode::OK,
                json!({"usage": {"prompt_tokens": 17}}),
                Duration::ZERO,
                false,
                "missing_message",
            ),
            (
                StatusCode::OK,
                json!("malformed-protocol-fixture"),
                Duration::ZERO,
                false,
                "malformed_protocol_json",
            ),
            (
                StatusCode::OK,
                json!({}),
                Duration::from_secs(2),
                false,
                "transport_timeout",
            ),
            (
                StatusCode::OK,
                json!({}),
                Duration::from_secs(2),
                true,
                "canceled",
            ),
        ] {
            let (base_url, seen) = fixture(status, body, delay).await;
            let mut connection = connection(base_url);
            connection.timeout_seconds = 1;
            let canceled = Arc::new(AtomicBool::new(false));
            let check = canceled.clone();
            let temp = tempfile::tempdir().unwrap();
            let image = temp.path().join("fixture.png");
            std::fs::write(&image, b"pixels").unwrap();
            let receipt_path = temp.path().join("receipt.json");
            let durable_path = receipt_path.clone();
            let events = Arc::new(Mutex::new(Vec::new()));
            let recorded = events.clone();
            let llm = OpenAiPlannerLlm::new(
                reqwest::Client::new(),
                connection,
                Some("fixture-secret".to_owned()),
                options("enabled", true),
                Arc::new(move || check.load(Ordering::SeqCst)),
            )
            .unwrap()
            .on_execution_updated(Arc::new(move |execution| {
                recorded.lock().unwrap().push(execution.clone());
                std::fs::write(&durable_path, serde_json::to_vec(execution).unwrap())?;
                Ok(())
            }));
            if cancel {
                let ledger = seen.clone();
                tokio::spawn(async move {
                    while ledger.lock().unwrap().is_empty() {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    canceled.store(true, Ordering::SeqCst);
                });
            }
            let error = llm
                .complete(request(vec![PlannerReferenceImage {
                    roles: vec![PlannerReferenceRole {
                        role: "hero".to_owned(),
                        locator: None,
                    }],
                    path: image,
                }]))
                .await
                .unwrap_err();
            let HarnessError::PlannerExecutionFailure { source, executions } = error else {
                panic!("missing receipt")
            };
            assert_eq!(
                matches!(*source, HarnessError::Refused(_)),
                cancel,
                "cancel retains its CLI refusal class"
            );
            assert_eq!(executions.len(), 1);
            let receipt = &executions[0];
            assert_eq!(receipt.failure_code.as_deref(), Some(code));
            assert_eq!(receipt.provider, "openai_compatible");
            assert_eq!(receipt.backend.as_deref(), Some("fixture"));
            assert_eq!(receipt.model, "planner-model");
            assert_eq!(receipt.max_output_tokens, Some(4096));
            assert_eq!(receipt.request_timeout_seconds, Some(1));
            assert_eq!(receipt.temperature, Some(0.2));
            assert_eq!(receipt.reference_pixels_sent, Some(true));
            assert!(receipt.duration_seconds.unwrap() > 0.0);
            let bytes = std::fs::read_to_string(&receipt_path).unwrap();
            assert_eq!(
                serde_json::from_str::<PlannerExecutionRecord>(&bytes).unwrap(),
                *receipt
            );
            assert!(!bytes.contains("fixture-secret"));
            assert!(!bytes.contains("never-retain"));
            assert_eq!(events.lock().unwrap().len(), 2);
            assert_eq!(
                events.lock().unwrap()[0].failure_code.as_deref(),
                Some("dispatch_started")
            );
            assert_eq!(seen.lock().unwrap().len(), 1);
            if matches!(code, "missing_message" | "authentication_failed") {
                assert_eq!(receipt.usage.as_ref().unwrap().input_tokens, Some(17));
            }
        }
    }

    #[tokio::test]
    async fn predispatch_cancel_and_receipt_storage_failure_send_no_request() {
        let (base_url, seen) = fixture(StatusCode::OK, json!({}), Duration::ZERO).await;
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url.clone()),
            None,
            options("disabled", false),
            Arc::new(|| true),
        )
        .unwrap()
        .on_execution_updated(Arc::new(|_| {
            panic!("cancellation must not fabricate a receipt")
        }));
        assert!(matches!(
            llm.complete(request(Vec::new())).await,
            Err(HarnessError::Refused(_))
        ));
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(base_url),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap()
        .on_execution_updated(Arc::new(|_| {
            Err(HarnessError::Io("receipt storage unavailable".to_owned()))
        }));
        assert!(matches!(
            llm.complete(request(Vec::new())).await,
            Err(HarnessError::Io(_))
        ));
        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn connection_failure_records_attempt_without_claiming_server_receipt() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let llm = OpenAiPlannerLlm::new(
            reqwest::Client::new(),
            connection(format!("http://{address}/v1")),
            None,
            options("disabled", false),
            Arc::new(|| false),
        )
        .unwrap();
        let HarnessError::PlannerExecutionFailure { source, executions } =
            llm.complete(request(Vec::new())).await.unwrap_err()
        else {
            panic!("expected attempted connection receipt")
        };
        assert!(matches!(*source, HarnessError::Transport(_)));
        assert_eq!(
            executions[0].failure_code.as_deref(),
            Some("connection_failed")
        );
        assert_eq!(executions[0].usage, None);
        assert_eq!(executions[0].finish_reason, None);
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
