// The describing prompt for the dataset prompt-batch describe job (SW-7085, feature SW-F362).
//
// Training Studio's "Create Prompt Batch from Dataset" dialog prefills its editable prompt with the
// worker's default instruction before the first run, so the API serves that text. ONE source of
// truth: the text is the `[SYSTEM]` section of the worker's own prompt asset, embedded here with
// `include_str!` from the worker crate's source tree, so the default the dialog shows and the
// default the worker describes with are the same bytes. The asset stays where the worker owns it;
// the API only reads it. The `[OUTPUT]` JSON contract is never served: the worker always appends
// it, and the user never edits it.

use super::*;
use sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS;

const IMAGE_BATCH_DESCRIBE_ASSET: &str =
    include_str!("../../../crates/sceneworks-worker/src/image_batch_describe_v1.txt");

/// One `[NAME]` section of a prompt asset, trimmed. Same marker rule as the worker's
/// `magic_section_from`: a marker is a whole line `[WORD]` with no spaces, and a section runs to the
/// next marker.
fn asset_section(source: &str, name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let is_marker = trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.contains(' ');
        if is_marker {
            if capturing {
                break;
            }
            capturing = trimmed[1..trimmed.len() - 1].eq_ignore_ascii_case(name);
            continue;
        }
        if capturing {
            body.push(line);
        }
    }
    body.join("\n").trim().to_owned()
}

/// The default editable describing instruction (the asset's `[SYSTEM]` section).
pub(crate) fn default_describe_instruction() -> String {
    asset_section(IMAGE_BATCH_DESCRIBE_ASSET, "SYSTEM")
}

/// `GET /api/v1/training/describe-prompt/default`: the default describing instruction and the
/// longest instruction a describe job accepts.
pub(crate) async fn get_default_describe_prompt() -> Json<Value> {
    Json(json!({
        "prompt": default_describe_instruction(),
        "maxChars": MAX_DESCRIBE_PROMPT_CHARS,
    }))
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

fn builtin_describe_preset() -> Value {
    json!({
        "id": BUILTIN_DESCRIBE_PRESET_ID,
        "name": "Default",
        "prompt": default_describe_instruction(),
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

/// `GET /api/v1/training/describe-prompt/presets`: the built-in default first, then the stored
/// presets (not archived) by name.
pub(crate) async fn list_describe_prompt_presets(
    State(state): State<AppState>,
) -> Result<Json<Vec<Value>>, ApiError> {
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
    let mut presets = vec![builtin_describe_preset()];
    presets.extend(stored);
    Ok(Json(presets))
}

/// `POST /api/v1/training/describe-prompt/presets` `{name, prompt}`: store a named preset. The id
/// is slugified from the name unless given; the built-in default's id is refused.
pub(crate) async fn create_describe_prompt_preset(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let preset = describe_preset_payload(payload)?;
    let manifest_path = describe_preset_manifest_path(&state);
    let preset = super::manifest_entity::create_manifest_entry(
        &state,
        &manifest_path,
        PRESET_MANIFEST_FIELD,
        preset,
        "global",
        "Describing prompt preset must be an object",
        "Describing prompt preset name is required",
        "A describing prompt preset with that name already exists",
        normalize_describe_preset_for_write,
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

    #[test]
    fn default_instruction_is_the_system_section_without_the_output_contract() {
        let prompt = default_describe_instruction();
        assert!(prompt.starts_with("You are a meticulous scene describer."));
        assert!(prompt.contains("## LIGHTING"));
        assert!(
            !prompt.contains("## OUTPUT"),
            "the JSON contract is never editable"
        );
        assert!(!prompt.contains("[OUTPUT]") && !prompt.contains("[META]"));
        assert!(prompt.chars().count() <= MAX_DESCRIBE_PROMPT_CHARS);
    }

    #[test]
    fn asset_section_stops_at_the_next_marker() {
        let source = "[A]\none\n[B]\ntwo\n  three  \n[C]\nfour";
        assert_eq!(asset_section(source, "B"), "two\n  three");
        assert_eq!(asset_section(source, "missing"), "");
    }
}
