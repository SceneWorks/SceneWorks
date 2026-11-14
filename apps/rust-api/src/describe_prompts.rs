// The describing prompt for the dataset prompt-batch describe job (SW-7085, feature SW-F362).
//
// Training Studio's "Create Prompt Batch from Dataset" dialog prefills its editable prompt with the
// selected model's default instruction before the first run, so the API serves that text. ONE
// source of truth: the text is a section of the worker's own prompt asset (`[SYSTEM]` for the
// Vision Captioner, `[JOYCAPTION_SYSTEM]` for JoyCaption), embedded here with `include_str!` from
// the worker crate's source tree and read with the same core parser and section choice the worker
// uses, so the default the dialog shows and the default the worker describes with are the same
// bytes. The limit is core's `max_describe_prompt_chars`, the one the job validation applies. The
// JSON contracts (`[OUTPUT]`, `[JOYCAPTION_OUTPUT]`) are never served: the worker always appends
// the model's contract, and the user never edits it, so a preset written before a field was added
// still gets a reply with every field.

use super::*;
use sceneworks_core::dataset_batch_describe::{
    max_describe_prompt_chars, MAX_DESCRIBE_PROMPT_CHARS,
};

const IMAGE_BATCH_DESCRIBE_ASSET: &str =
    include_str!("../../../crates/sceneworks-worker/src/image_batch_describe_v1.txt");

/// `model`'s default editable describing instruction (`model` is a resolved describer repo).
pub(crate) fn default_describe_instruction(model: &str) -> String {
    sceneworks_core::dataset_batch_describe::default_describe_instruction(
        IMAGE_BATCH_DESCRIBE_ASSET,
        Some(model),
    )
}

/// `GET /api/v1/training/describe-prompt/default?model=<repo>`: the model's default describing
/// instruction and the longest instruction a describe job for that model accepts. Absent `model`
/// is the Vision Captioner; an unknown model is refused like the describe job refuses it.
pub(crate) async fn get_default_describe_prompt(
    Query(query): Query<DescribePromptQuery>,
) -> Result<Json<Value>, ApiError> {
    let model = super::training::resolve_describe_model(query.model.as_deref())?;
    Ok(Json(json!({
        "model": model,
        "prompt": default_describe_instruction(model),
        "maxChars": max_describe_prompt_chars(Some(model)),
    })))
}

// ---------------------------------------------------------------------------------------------
// Named describing-prompt presets (SW-7086). A small sibling of the prompt-batch catalog: the same
// JSONC-manifest storage under `config/manifests/` and the same shared manifest-entity create /
// update / delete-as-archive helpers, global scope only (a describing prompt is a personal
// instruction, not project content). The built-in default is never stored: the list always leads
// with it, built from the same worker asset as the default route, and every write that names its
// id is refused.
// ---------------------------------------------------------------------------------------------

const PRESET_MANIFEST_FILE: &str = "user.describe-prompt-presets.jsonc";
const PRESET_MANIFEST_FIELD: &str = "presets";
/// The built-in default preset's id. Reserved: no stored preset may take it.
pub(crate) const BUILTIN_DESCRIBE_PRESET_ID: &str = "default";

fn describe_preset_manifest_path(state: &AppState) -> PathBuf {
    state
        .settings
        .config_dir
        .join("manifests")
        .join(PRESET_MANIFEST_FILE)
}

/// The built-in default preset for `model`: its prompt is that model's default instruction.
fn builtin_describe_preset(model: &str) -> Value {
    json!({
        "id": BUILTIN_DESCRIBE_PRESET_ID,
        "name": "Default",
        "prompt": default_describe_instruction(model),
        "scope": "builtin",
        "builtin": true,
    })
}

fn describe_preset_not_found() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: "Describing prompt preset not found".to_owned(),
        context: None,
        code: None,
    }
}

fn builtin_describe_preset_refused() -> ApiError {
    ApiError::forbidden("The built-in default describing prompt cannot be changed or deleted.")
}

fn is_builtin_describe_preset_id(id: &str) -> bool {
    id.trim() == BUILTIN_DESCRIBE_PRESET_ID
}

fn describe_preset_payload(payload: Value) -> Result<Value, ApiError> {
    let mut payload = match payload {
        Value::Object(_) => payload,
        _ => {
            return Err(ApiError::bad_request(
                "Describing prompt preset must be an object",
            ))
        }
    };
    if let Some(object) = payload.as_object_mut() {
        // Server-managed fields: a client can never mark a stored preset built-in or move it.
        object.remove("builtin");
        object.remove("scope");
        object.remove("archived");
    }
    Ok(payload)
}

/// Normalize and validate a stored preset: a valid id, a name, and a non-blank prompt within the
/// describe job's limit. The prompt is stored exactly as given (no trimming), so selecting the
/// preset restores the exact text.
fn normalize_describe_preset_for_write(
    mut preset: Value,
    scope: &str,
    require_all: bool,
) -> Result<Value, ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Describing prompt preset must be an object"))?;
    object.insert("scope".to_owned(), Value::String(scope.to_owned()));
    let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    if is_builtin_describe_preset_id(id) {
        return Err(builtin_describe_preset_refused());
    }
    let id_valid = !id.is_empty()
        && id.chars().enumerate().all(|(index, character)| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || (index > 0 && matches!(character, '_' | '-'))
        });
    if !id_valid {
        return Err(ApiError::bad_request(
            "Describing prompt preset id must use lowercase letters, numbers, dashes, or underscores",
        ));
    }
    super::recipe_presets::validate_required_string_field(
        object,
        "name",
        require_all,
        "Describing prompt preset name is required",
    )?;
    match object.get("prompt") {
        None if !require_all => {}
        Some(Value::String(prompt)) if !prompt.trim().is_empty() => {
            if prompt.chars().count() > MAX_DESCRIBE_PROMPT_CHARS {
                return Err(ApiError::bad_request(format!(
                    "The describing prompt must be at most {MAX_DESCRIBE_PROMPT_CHARS} characters."
                )));
            }
        }
        _ => {
            return Err(ApiError::bad_request(
                "Describing prompt preset prompt is required",
            ))
        }
    }
    Ok(preset)
}

/// `GET /api/v1/training/describe-prompt/presets?model=<repo>`: the built-in default first (the
/// prompt of `model`, absent meaning the Vision Captioner), then the stored presets (not archived)
/// by name. A stored preset is model-independent; its length is checked against the selected
/// model's limit when a job is submitted.
pub(crate) async fn list_describe_prompt_presets(
    State(state): State<AppState>,
    Query(query): Query<DescribePromptQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let model = super::training::resolve_describe_model(query.model.as_deref())?;
    let manifest_path = describe_preset_manifest_path(&state);
    let mut stored = load_manifest_entries(&state, &manifest_path, PRESET_MANIFEST_FIELD)
        .await?
        .into_iter()
        .filter(|preset| !super::manifest_entity::manifest_entry_archived(preset))
        .collect::<Vec<_>>();
    stored.sort_by_key(|preset| {
        preset
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_lowercase()
    });
    let mut presets = vec![builtin_describe_preset(model)];
    presets.extend(stored);
    Ok(Json(presets))
}

/// `POST /api/v1/training/describe-prompt/presets` `{name, prompt}`: store a named preset. The id
/// is slugified from the name unless given; the built-in default's id is refused.
pub(crate) async fn create_describe_prompt_preset(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let mut preset = describe_preset_payload(payload)?;
    // Same id rule as the shared create helper: an explicit id, else the slugified name.
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Describing prompt preset must be an object"))?;
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
                .map(super::recipe_presets::slugify_preset_id)
        })
        .ok_or_else(|| ApiError::bad_request("Describing prompt preset name is required"))?;
    object.insert("id".to_owned(), Value::String(id.clone()));
    let timestamp = now_rfc3339();
    object.insert("createdAt".to_owned(), Value::String(timestamp.clone()));
    object.insert("updatedAt".to_owned(), Value::String(timestamp));
    let preset = normalize_describe_preset_for_write(preset, "global", true)?;
    let manifest_path = describe_preset_manifest_path(&state);
    // A deleted preset is archived, not removed, but its name reads as free to the user: saving a
    // new preset under it replaces the archived entry. A live preset with the id still conflicts.
    let preset = mutate_manifest_entries(
        &state,
        &manifest_path,
        PRESET_MANIFEST_FIELD,
        move |mut entries| {
            let existing = entries
                .iter()
                .position(|entry| entry.get("id").and_then(Value::as_str) == Some(id.as_str()));
            match existing {
                Some(index) if super::manifest_entity::manifest_entry_archived(&entries[index]) => {
                    entries[index] = preset.clone();
                }
                Some(_) => {
                    return Err(ApiError::bad_request(
                        "A describing prompt preset with that name already exists",
                    ))
                }
                None => entries.push(preset.clone()),
            }
            Ok((entries, preset))
        },
    )
    .await?;
    Ok((StatusCode::CREATED, Json(preset)))
}

/// `PATCH /api/v1/training/describe-prompt/presets/:preset_id` `{name?, prompt?}`.
pub(crate) async fn update_describe_prompt_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<Json<Value>, ApiError> {
    if is_builtin_describe_preset_id(&preset_id) {
        return Err(builtin_describe_preset_refused());
    }
    let mut patch = describe_preset_payload(payload)?;
    if let Some(object) = patch.as_object_mut() {
        object.remove("id");
    }
    let manifest_path = describe_preset_manifest_path(&state);
    let preset = super::manifest_entity::update_manifest_entry(
        &state,
        &manifest_path,
        PRESET_MANIFEST_FIELD,
        &preset_id,
        "global",
        patch,
        describe_preset_not_found,
        normalize_describe_preset_for_write,
    )
    .await?;
    Ok(Json(preset))
}

/// `DELETE /api/v1/training/describe-prompt/presets/:preset_id`: archive a stored preset (the
/// shared delete-as-archive semantics); the built-in default is refused.
pub(crate) async fn delete_describe_prompt_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if is_builtin_describe_preset_id(&preset_id) {
        return Err(builtin_describe_preset_refused());
    }
    let manifest_path = describe_preset_manifest_path(&state);
    let preset = super::manifest_entity::delete_manifest_entry(
        &state,
        &manifest_path,
        PRESET_MANIFEST_FIELD,
        &preset_id,
        "global",
        describe_preset_not_found,
        normalize_describe_preset_for_write,
    )
    .await?;
    Ok(Json(preset))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use sceneworks_core::dataset_batch_describe::{
        prompt_asset_section, JOYCAPTION_REPO, MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS,
        VISION_CAPTIONER_REPO,
    };

    #[test]
    fn each_model_default_is_exactly_its_own_asset_section() {
        let vision = default_describe_instruction(VISION_CAPTIONER_REPO);
        let joy = default_describe_instruction(JOYCAPTION_REPO);
        // The worker's own section text, read from the worker's asset file.
        assert_eq!(
            vision,
            prompt_asset_section(IMAGE_BATCH_DESCRIBE_ASSET, "SYSTEM")
        );
        assert_eq!(
            joy,
            prompt_asset_section(IMAGE_BATCH_DESCRIBE_ASSET, "JOYCAPTION_SYSTEM")
        );
        assert!(vision.starts_with("You are a meticulous scene describer."));
        assert!(vision.contains("## LIGHTING"));
        assert!(
            vision.trim_end().ends_with("and where the shadows fall."),
            "the Vision default stops before [JOYCAPTION_SYSTEM]"
        );
        assert!(!vision.contains("JOYCAPTION") && !vision.contains("about 500 words"));
        for prompt in [&vision, &joy] {
            assert!(
                !prompt.contains("## OUTPUT"),
                "the JSON contract is never editable"
            );
            assert!(!prompt.contains("[OUTPUT]") && !prompt.contains("[META]"));
        }
        assert!(joy.contains("about 500 words") && !joy.contains("## LABELS"));
        assert!(vision.chars().count() <= MAX_DESCRIBE_PROMPT_CHARS);
        assert!(joy.chars().count() <= MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS);
    }
}
