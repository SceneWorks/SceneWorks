// Prompt-batch persistence (sc-9954, epic 9952 — Batch Prompt Processing).
// A prompt batch is a saved, named list of prompt templates plus their variable
// definitions ({key, values[]}). It is a sibling of a recipe preset — same
// global/project JSONC-manifest storage and the same scope/write-location model —
// but deliberately NOT a recipe: it has no model, workflow, or LoRA, so none of the
// recipe validators apply. Storage: `user.prompt-batches.jsonc` (global) and a
// per-project `recipes/prompt-batches.jsonc`, mutated through the shared
// manifest read→merge→save helpers. Generic slug/duplicate helpers are reused from
// `recipe_presets`; batch-specific shape validation lives here.

use super::*;
// Generic (non-recipe-specific) helpers reused from the recipe-preset module: slug,
// duplicate-id/name suffixing, and the shared payload/field validators.
use super::recipe_presets::{
    next_duplicate_preset_id, next_duplicate_preset_name, slugify_preset_id, take_string_field,
    validate_required_string_field,
};

const MANIFEST_FIELD: &str = "batches";

pub(crate) async fn list_prompt_batches(
    State(state): State<AppState>,
    Query(query): Query<PromptBatchesQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    validate_prompt_batch_query(&query)?;
    let mut batches = prompt_batch_catalog(&state, query.project_id.as_deref()).await?;
    if !query.include_archived.unwrap_or(false) {
        batches.retain(|batch| !prompt_batch_archived(batch));
    }
    if let Some(scope) = query.scope.as_deref() {
        batches.retain(|batch| batch.get("scope").and_then(Value::as_str) == Some(scope));
    }
    Ok(Json(batches))
}

pub(crate) async fn get_prompt_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
    Query(query): Query<PromptBatchesQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_prompt_batch_query(&query)?;
    let batch = prompt_batch_catalog(&state, query.project_id.as_deref())
        .await?
        .into_iter()
        .find(|batch| batch.get("id").and_then(Value::as_str) == Some(batch_id.as_str()))
        .filter(|batch| {
            query.scope.as_deref().map_or(true, |scope| {
                batch.get("scope").and_then(Value::as_str) == Some(scope)
            })
        })
        .filter(|batch| query.include_archived.unwrap_or(false) || !prompt_batch_archived(batch))
        .ok_or_else(prompt_batch_not_found)?;
    Ok(Json(batch))
}

pub(crate) async fn create_prompt_batch(
    State(state): State<AppState>,
    Query(query): Query<PromptBatchesQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_prompt_batch_query(&query)?;
    let mut batch = prompt_batch_from_payload(payload)?;
    let scope = prompt_batch_write_scope(query.scope.as_deref(), prompt_batch_scope(&batch))?;
    let project_id = prompt_batch_context_project_id(&query, &mut batch);
    let manifest_path =
        prompt_batch_write_manifest_path(&state, &scope, project_id.as_deref()).await?;
    let object = batch
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Prompt batch must be an object"))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            object
                .get("name")
                .and_then(Value::as_str)
                .map(slugify_preset_id)
        })
        .ok_or_else(|| ApiError::bad_request("Prompt batch name is required"))?;
    object.insert("id".to_owned(), Value::String(id.clone()));
    let timestamp = now_rfc3339();
    object
        .entry("createdAt".to_owned())
        .or_insert_with(|| Value::String(timestamp.clone()));
    object.insert("updatedAt".to_owned(), Value::String(timestamp));
    let batch = mutate_manifest_entries(&state, &manifest_path, MANIFEST_FIELD, |mut entries| {
        let batch = normalize_prompt_batch_for_write(batch, &scope, true)?;
        if entries
            .iter()
            .any(|entry| entry.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            return Err(ApiError::bad_request("Prompt batch already exists"));
        }
        entries.push(batch.clone());
        Ok((entries, batch))
    })
    .await?;
    Ok((StatusCode::CREATED, Json(batch)))
}

pub(crate) async fn update_prompt_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
    Query(query): Query<PromptBatchesQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<Json<Value>, ApiError> {
    validate_prompt_batch_query(&query)?;
    let mut patch = prompt_batch_from_payload(payload)?;
    let project_id = prompt_batch_context_project_id(&query, &mut patch);
    strip_prompt_batch_write_context(&mut patch);
    let location = find_prompt_batch_write_location(
        &state,
        &batch_id,
        project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let batch = mutate_manifest_entries(
        &state,
        &location.manifest_path,
        MANIFEST_FIELD,
        |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(batch_id.as_str())
            }) else {
                return Err(prompt_batch_not_found());
            };
            let mut batch = entries[index].clone();
            merge_object(&mut batch, patch);
            if let Some(object) = batch.as_object_mut() {
                object.insert("id".to_owned(), Value::String(batch_id.clone()));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let batch = normalize_prompt_batch_for_write(batch, &location.scope, false)?;
            entries[index] = batch.clone();
            Ok((entries, batch))
        },
    )
    .await?;
    Ok(Json(batch))
}

pub(crate) async fn delete_prompt_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
    Query(query): Query<PromptBatchesQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_prompt_batch_query(&query)?;
    let location = find_prompt_batch_write_location(
        &state,
        &batch_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let batch = mutate_manifest_entries(
        &state,
        &location.manifest_path,
        MANIFEST_FIELD,
        |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(batch_id.as_str())
            }) else {
                return Err(prompt_batch_not_found());
            };
            let mut batch = entries[index].clone();
            if let Some(object) = batch.as_object_mut() {
                object.insert("archived".to_owned(), Value::Bool(true));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let batch = normalize_prompt_batch_for_write(batch, &location.scope, false)?;
            entries[index] = batch.clone();
            Ok((entries, batch))
        },
    )
    .await?;
    Ok(Json(batch))
}

pub(crate) async fn duplicate_prompt_batch(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
    Query(query): Query<PromptBatchesQuery>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_prompt_batch_query(&query)?;
    let location = find_prompt_batch_write_location(
        &state,
        &batch_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let batch = mutate_manifest_entries(
        &state,
        &location.manifest_path,
        MANIFEST_FIELD,
        |mut entries| {
            let Some(source) = entries
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(batch_id.as_str()))
                .cloned()
            else {
                return Err(prompt_batch_not_found());
            };
            let mut duplicate = source;
            if let Some(object) = duplicate.as_object_mut() {
                object.remove("manifestPath");
            }
            let base_id = duplicate
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(batch_id.as_str());
            let duplicate_id = next_duplicate_preset_id(&entries, base_id);
            let duplicate_name = next_duplicate_preset_name(
                &entries,
                duplicate
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(base_id),
            );
            let timestamp = now_rfc3339();
            if let Some(object) = duplicate.as_object_mut() {
                object.insert("id".to_owned(), Value::String(duplicate_id));
                object.insert("name".to_owned(), Value::String(duplicate_name));
                object.insert("scope".to_owned(), Value::String(location.scope.clone()));
                object.insert("archived".to_owned(), Value::Bool(false));
                object.insert("createdAt".to_owned(), Value::String(timestamp.clone()));
                object.insert("updatedAt".to_owned(), Value::String(timestamp));
            }
            let duplicate = normalize_prompt_batch_for_write(duplicate, &location.scope, true)?;
            entries.push(duplicate.clone());
            Ok((entries, duplicate))
        },
    )
    .await?;
    Ok((StatusCode::CREATED, Json(batch)))
}

pub(crate) async fn prompt_batch_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let user_manifest = manifest_dir.join("user.prompt-batches.jsonc");
    let mut batches = load_manifest_entries(state, &user_manifest, MANIFEST_FIELD)
        .await?
        .into_iter()
        .map(|batch| normalize_prompt_batch_entry(batch, "global", &user_manifest))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("recipes").join("prompt-batches.jsonc");
        let project_batches = load_manifest_entries(state, &project_manifest, MANIFEST_FIELD)
            .await?
            .into_iter()
            .map(|batch| normalize_prompt_batch_entry(batch, "project", &project_manifest))
            .collect::<Result<Vec<_>, _>>()?;
        batches = merge_entries_by_id(batches, project_batches);
    }
    batches.sort_by(|left, right| {
        let key = |batch: &Value| {
            (
                prompt_batch_scope_order(batch.get("scope").and_then(Value::as_str)),
                batch
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_lowercase(),
            )
        };
        key(left).cmp(&key(right))
    });
    Ok(batches)
}

fn prompt_batch_scope_order(scope: Option<&str>) -> u8 {
    match scope {
        Some("global") => 0,
        Some("project") => 1,
        _ => 2,
    }
}

fn normalize_prompt_batch_entry(
    mut batch: Value,
    scope: &str,
    manifest_path: &FsPath,
) -> Result<Value, ApiError> {
    let object = batch
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Prompt batch manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    object
        .entry("prompts".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    object
        .entry("variables".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    Ok(batch)
}

fn prompt_batch_archived(batch: &Value) -> bool {
    batch
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn prompt_batch_from_payload(payload: Value) -> Result<Value, ApiError> {
    match payload {
        Value::Null => Ok(Value::Object(JsonObject::new())),
        Value::Object(_) => Ok(payload),
        _ => Err(ApiError::bad_request(
            "Prompt batch payload must be an object",
        )),
    }
}

fn prompt_batch_scope(batch: &Value) -> Option<&str> {
    batch.get("scope").and_then(Value::as_str)
}

fn prompt_batch_context_project_id(
    query: &PromptBatchesQuery,
    payload: &mut Value,
) -> Option<String> {
    query
        .project_id
        .clone()
        .or_else(|| take_string_field(payload, "projectId"))
}

fn strip_prompt_batch_write_context(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("projectId");
        object.remove("scope");
        object.remove("manifestPath");
    }
}

fn prompt_batch_write_scope(
    query_scope: Option<&str>,
    payload_scope: Option<&str>,
) -> Result<String, ApiError> {
    let scope = query_scope.or(payload_scope).unwrap_or("global").trim();
    match scope {
        "global" | "project" => Ok(scope.to_owned()),
        _ => Err(ApiError::bad_request(
            "Prompt batch scope must be global or project",
        )),
    }
}

fn validate_prompt_batch_query(query: &PromptBatchesQuery) -> Result<(), ApiError> {
    if let Some(scope) = query.scope.as_deref() {
        match scope {
            "global" | "project" => {}
            _ => return Err(ApiError::bad_request("Unsupported prompt batch scope")),
        }
    }
    Ok(())
}

async fn prompt_batch_write_manifest_path(
    state: &AppState,
    scope: &str,
    project_id: Option<&str>,
) -> Result<PathBuf, ApiError> {
    match scope {
        "global" => Ok(state
            .settings
            .config_dir
            .join("manifests")
            .join("user.prompt-batches.jsonc")),
        "project" => {
            let Some(project_id) = project_id else {
                return Err(ApiError::bad_request(
                    "Project prompt batches require projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            Ok(project_path.join("recipes").join("prompt-batches.jsonc"))
        }
        _ => Err(ApiError::bad_request(
            "Prompt batch scope must be global or project",
        )),
    }
}

struct PromptBatchWriteLocation {
    scope: String,
    manifest_path: PathBuf,
}

fn prompt_batch_not_found() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: "Prompt batch not found".to_owned(),
    }
}

async fn find_prompt_batch_write_location(
    state: &AppState,
    batch_id: &str,
    project_id: Option<&str>,
    scope: Option<&str>,
) -> Result<PromptBatchWriteLocation, ApiError> {
    match scope {
        Some("global") => {
            return prompt_batch_location_if_present(state, batch_id, "global", project_id).await
        }
        Some("project") => {
            return prompt_batch_location_if_present(state, batch_id, "project", project_id).await
        }
        Some(_) => return Err(ApiError::bad_request("Unsupported prompt batch scope")),
        None => {}
    }

    if project_id.is_some() {
        match prompt_batch_location_if_present(state, batch_id, "project", project_id).await {
            Ok(location) => return Ok(location),
            Err(error) if error.status == StatusCode::NOT_FOUND => {}
            Err(error) => return Err(error),
        }
    }
    prompt_batch_location_if_present(state, batch_id, "global", project_id).await
}

async fn prompt_batch_location_if_present(
    state: &AppState,
    batch_id: &str,
    scope: &str,
    project_id: Option<&str>,
) -> Result<PromptBatchWriteLocation, ApiError> {
    let manifest_path = prompt_batch_write_manifest_path(state, scope, project_id).await?;
    let entries = load_manifest_entries(state, &manifest_path, MANIFEST_FIELD).await?;
    if entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(batch_id))
    {
        Ok(PromptBatchWriteLocation {
            scope: scope.to_owned(),
            manifest_path,
        })
    } else {
        Err(prompt_batch_not_found())
    }
}

fn normalize_prompt_batch_for_write(
    mut batch: Value,
    scope: &str,
    require_all: bool,
) -> Result<Value, ApiError> {
    let object = batch
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Prompt batch must be an object"))?;
    object.insert("scope".to_owned(), Value::String(scope.to_owned()));
    validate_prompt_batch_id(object.get("id").and_then(Value::as_str))?;
    validate_required_string_field(object, "name", require_all, "Prompt batch name is required")?;
    validate_prompt_batch_prompts(object.get("prompts"))?;
    validate_prompt_batch_variables(object.get("variables"))?;
    validate_prompt_batch_last_values(object.get("lastValues"))?;
    Ok(batch)
}

fn validate_prompt_batch_id(value: Option<&str>) -> Result<(), ApiError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Err(ApiError::bad_request("Prompt batch id is required"));
    };
    let valid = value.chars().enumerate().all(|(index, character)| {
        character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || (index > 0 && matches!(character, '_' | '-'))
    });
    if !valid {
        return Err(ApiError::bad_request(
            "Prompt batch id must use lowercase letters, numbers, dashes, or underscores",
        ));
    }
    Ok(())
}

fn validate_prompt_batch_prompts(value: Option<&Value>) -> Result<(), ApiError> {
    let Some(prompts) = value else {
        return Ok(());
    };
    let items = prompts
        .as_array()
        .ok_or_else(|| ApiError::bad_request("Prompt batch prompts must be an array"))?;
    if items.iter().any(|item| !item.is_string()) {
        return Err(ApiError::bad_request(
            "Prompt batch prompts must be an array of strings",
        ));
    }
    Ok(())
}

fn validate_prompt_batch_variables(value: Option<&Value>) -> Result<(), ApiError> {
    let Some(variables) = value else {
        return Ok(());
    };
    let items = variables
        .as_array()
        .ok_or_else(|| ApiError::bad_request("Prompt batch variables must be an array"))?;
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| ApiError::bad_request("Prompt batch variable must be an object"))?;
        match object.get("key").and_then(Value::as_str).map(str::trim) {
            Some(key) if !key.is_empty() => {}
            _ => {
                return Err(ApiError::bad_request(
                    "Prompt batch variable key is required",
                ))
            }
        }
        if let Some(values) = object.get("values") {
            let values = values.as_array().ok_or_else(|| {
                ApiError::bad_request("Prompt batch variable values must be an array")
            })?;
            if values.iter().any(|value| !value.is_string()) {
                return Err(ApiError::bad_request(
                    "Prompt batch variable values must be an array of strings",
                ));
            }
        }
    }
    Ok(())
}

fn validate_prompt_batch_last_values(value: Option<&Value>) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_object()) {
        return Err(ApiError::bad_request(
            "Prompt batch lastValues must be an object",
        ));
    }
    Ok(())
}
