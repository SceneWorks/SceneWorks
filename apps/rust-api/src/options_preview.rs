//! Server-backed draft feedback when the pure browser validator exceeds its byte budget.
//! This route uses the same upgrade and static rule owner as submission. It never enqueues.
use std::borrow::Cow;

use axum::{extract::State, http::StatusCode, Json};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::request_validation::{self, ValidationRefusal};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    models::model_catalog_snapshot, training::effective_training_targets, ApiError, AppState,
};

const MAX_PREVIEW_REQUESTS: usize = 32;
const MODEL_FREE_KINDS: &[&str] = &[
    "characterTest",
    "personDetect",
    "personTrack",
    "promptRefine",
    "videoUpscale",
    "yue2",
    "yue2Execution",
];
const SUPPORTED_KINDS: &[&str] = &[
    "audio",
    "characterTest",
    "control_training",
    "image",
    "imageDetail",
    "interleave",
    "lora_train",
    "personDetect",
    "personTrack",
    "promptRefine",
    "training",
    "utility",
    "vector",
    "vectorPrompt",
    "video",
    "videoUpscale",
    "vqa",
    "yue2",
    "yue2Execution",
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct OptionsPreviewRequest {
    validator_schema_version: u32,
    kind: String,
    requests: Vec<JsonObject>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OptionsPreviewResponse {
    validator_schema_version: u32,
    results: Vec<OptionsPreviewResult>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum OptionsPreviewResult {
    Valid {
        ok: bool,
        resolved: JsonObject,
    },
    Invalid {
        ok: bool,
        refusal: ValidationRefusal,
    },
}

impl OptionsPreviewResult {
    fn valid(resolved: JsonObject) -> Self {
        Self::Valid { ok: true, resolved }
    }

    fn invalid(refusal: ValidationRefusal) -> Self {
        Self::Invalid { ok: false, refusal }
    }
}

fn missing_fact(field: &str, id: &str) -> ValidationRefusal {
    ValidationRefusal::new(
        field,
        format!("No authoritative catalog entry exists for '{id}'."),
    )
}

fn model_entry<'a>(models: &'a [Value], id: &str) -> Option<&'a JsonObject> {
    models
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(id))
        .and_then(Value::as_object)
}

fn explicit_model(payload: &JsonObject) -> Option<&str> {
    payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

fn validate_optional_identity(kind: &str, payload: &JsonObject) -> Result<(), ValidationRefusal> {
    let field = match kind {
        "imageDetail" => "model",
        "utility" => "modelId",
        _ => return Ok(()),
    };
    if let Some(value) = payload.get(field) {
        let id = value
            .as_str()
            .ok_or_else(|| ValidationRefusal::new(field, format!("{field} must be a string")))?;
        crate::validate_model_id(id)
            .map_err(|error| ValidationRefusal::new(field, error.detail))?;
    }
    Ok(())
}

fn facts_for<'a>(
    kind: &str,
    payload: &JsonObject,
    models: &'a [Value],
    training_facts: &'a JsonObject,
) -> Result<Cow<'a, JsonObject>, ValidationRefusal> {
    if matches!(kind, "training" | "lora_train" | "control_training") {
        return Ok(Cow::Borrowed(training_facts));
    }
    if kind == "vectorPrompt" {
        let raster_id = payload
            .get("rasterModel")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let vector_id = payload
            .get("vectorModel")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let raster =
            model_entry(models, raster_id).ok_or_else(|| missing_fact("rasterModel", raster_id))?;
        let vector =
            model_entry(models, vector_id).ok_or_else(|| missing_fact("vectorModel", vector_id))?;
        return Ok(Cow::Owned(JsonObject::from_iter([
            ("raster".into(), Value::Object(raster.clone())),
            ("vector".into(), Value::Object(vector.clone())),
        ])));
    }
    if MODEL_FREE_KINDS.contains(&kind)
        || (kind == "utility" && request_validation::model_id_for_request(kind, payload).is_none())
        || (kind == "imageDetail" && explicit_model(payload).is_none())
    {
        return Ok(Cow::Owned(JsonObject::new()));
    }
    let id = request_validation::model_id_for_request(kind, payload).ok_or_else(|| {
        ValidationRefusal::new("model", "Choose a model before validating this request.")
    })?;
    model_entry(models, &id)
        .map(Cow::Borrowed)
        .ok_or_else(|| missing_fact("model", &id))
}

pub(crate) async fn validate_options_preview(
    State(state): State<AppState>,
    Json(request): Json<OptionsPreviewRequest>,
) -> Result<Json<OptionsPreviewResponse>, ApiError> {
    let version = request_validation::VALIDATOR_SCHEMA_VERSION;
    if request.validator_schema_version != version {
        return Err(ApiError::typed(
            StatusCode::CONFLICT,
            "The browser and server validator schema versions differ.",
            "validator_version_mismatch",
            json!({ "serverValidatorSchemaVersion": version }),
        ));
    }
    if !SUPPORTED_KINDS.contains(&request.kind.as_str()) {
        return Err(ApiError::bad_request("Unsupported validation kind."));
    }
    if request.requests.is_empty() || request.requests.len() > MAX_PREVIEW_REQUESTS {
        return Err(ApiError::bad_request(format!(
            "Validate between 1 and {MAX_PREVIEW_REQUESTS} requests per call."
        )));
    }

    let needs_models = !MODEL_FREE_KINDS.contains(&request.kind.as_str())
        && !matches!(
            request.kind.as_str(),
            "training" | "lora_train" | "control_training"
        )
        && match request.kind.as_str() {
            "utility" => request.requests.iter().any(|payload| {
                request_validation::model_id_for_request("utility", payload).is_some()
            }),
            "imageDetail" => request
                .requests
                .iter()
                .any(|payload| explicit_model(payload).is_some()),
            _ => true,
        };
    let models = if needs_models {
        Some(model_catalog_snapshot(&state).await?)
    } else {
        None
    };
    let training_facts = if matches!(
        request.kind.as_str(),
        "training" | "lora_train" | "control_training"
    ) {
        serde_json::to_value(effective_training_targets())
            .expect("effective training targets serialize")
            .as_object()
            .cloned()
            .expect("training registry is an object")
    } else {
        JsonObject::new()
    };
    let model_rows = models.as_deref().map(Vec::as_slice).unwrap_or(&[]);
    let results = request
        .requests
        .into_iter()
        .map(|payload| {
            let canonical = sceneworks_core::options::upgrade::upgrade_object(
                &payload,
                sceneworks_core::options::upgrade::request_carrier(&request.kind),
            )
            .map_err(|error| ValidationRefusal {
                code: error.code,
                field: error.field,
                detail: error.message,
                context: Box::new(json!({})),
            });
            let result = canonical.and_then(|canonical| {
                validate_optional_identity(&request.kind, &canonical)?;
                request_validation::validate_shape(&request.kind, &canonical)?;
                let entry = facts_for(&request.kind, &canonical, model_rows, &training_facts)?;
                request_validation::resolve(&request.kind, &canonical, &entry)
            });
            match result {
                Ok(resolved) => OptionsPreviewResult::valid(resolved),
                Err(refusal) => OptionsPreviewResult::invalid(refusal),
            }
        })
        .collect();
    Ok(Json(OptionsPreviewResponse {
        validator_schema_version: version,
        results,
    }))
}
