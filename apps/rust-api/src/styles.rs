use super::*;

use sceneworks_core::style_composer::compose_styled_prompt;

/// `GET /api/v1/styles` (sc-13134). Serves the built-in Style catalog — the SAME catalog the web
/// app reads from `styles.json` (both are mechanical derivations of `documents/style.txt`) — so a
/// headless/MCP client can list the groups + sub-styles and pick a `styleId` to send with a job.
pub(crate) async fn list_styles(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    Ok(Json(styles_catalog(&state).await?))
}

/// Load the built-in Style catalog object (`schemaVersion`, `source`, `promptTemplate`, `groups`)
/// from `config_dir/manifests/builtin.styles.jsonc`. A missing manifest yields an empty catalog so
/// the endpoint/fold degrade gracefully rather than 500 (the builtin is seeded on startup, so this
/// is only the never-seeded edge). JSONC comments are stripped before parsing, matching the other
/// manifest loaders.
pub(crate) async fn styles_catalog(state: &AppState) -> Result<Value, ApiError> {
    let path = state
        .settings
        .config_dir
        .join("manifests")
        .join("builtin.styles.jsonc");
    let payload = match tokio::fs::read_to_string(&path).await {
        Ok(payload) => payload,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(json!({ "schemaVersion": 1, "groups": [] }));
        }
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to load styles manifest {}: {error}",
                path.display()
            )));
        }
    };
    serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(&payload)).map_err(|error| {
        ApiError::internal(format!(
            "Failed to parse styles manifest {}: {error}",
            path.display()
        ))
    })
}

/// Resolve a `styleId` to its catalog style text — the exact bridge the web's `styleTextForId`
/// implements (`apps/web/src/data/styleCatalog.js`): a SUB-STYLE id resolves to its `prompt`; a
/// GROUP id resolves to that group's `description` (the group's "overall" style, sc-13171). Returns
/// `None` when the id matches neither, so the caller can 400 an unknown style rather than compose
/// against empty text. Sub-style ids are globally unique and cannot collide with a group id (the
/// generator de-dupes into one id-space), so a sub-style match is preferred deterministically.
pub(crate) fn style_text_for_id(catalog: &Value, id: &str) -> Option<String> {
    let groups = catalog.get("groups").and_then(Value::as_array)?;
    // Sub-style id → its prompt.
    for group in groups {
        let Some(styles) = group.get("styles").and_then(Value::as_array) else {
            continue;
        };
        for style in styles {
            if style.get("id").and_then(Value::as_str) == Some(id) {
                return style
                    .get("prompt")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
        }
    }
    // Group id → its description.
    for group in groups {
        if group.get("id").and_then(Value::as_str) == Some(id) {
            return group
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
    }
    None
}

/// Server-side Style fold for an image job (sc-13134): when the request carries a top-level
/// `styleId` AND the client did NOT already compose the prompt (mirrors the
/// `presetPromptResolvedClientSide` skip in `apply_recipe_preset_to_image_payload`), resolve the id
/// to its catalog style text and splice the current `prompt` into the `Style:`/`Description:`
/// template with `compose_styled_prompt` — the Rust port of the web composer, so the composed
/// prompt is byte-identical to the web path.
///
/// A no-op when no `styleId` is present, or when `presetPromptResolvedClientSide` is set (the web
/// path sends the already-composed prompt + that flag and omits the top-level id, so it is passed
/// through unchanged and never double-folded). Runs AFTER the recipe-preset fold so it wraps the
/// preset-composed prompt exactly as the web's `composeStyledPrompt` runs as the LAST wrap.
pub(crate) async fn apply_style_to_image_payload(
    state: &AppState,
    payload: &ImageJobRequest,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(style_id) = payload
        .style_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(());
    };
    // The client already composed the full prompt client-side (web path) — take it verbatim so we
    // never double-fold. Mirrors the preset skip.
    if payload.preset_prompt_resolved_client_side.unwrap_or(false) {
        return Ok(());
    }

    let catalog = styles_catalog(state).await?;
    let style_text = style_text_for_id(&catalog, style_id)
        .ok_or_else(|| ApiError::bad_request(format!("Style not found: {style_id}")))?;

    // Compose over the CURRENT job prompt (post recipe-preset fold), not the raw DTO prompt, so the
    // style wraps the preset-composed text just as the web layers style over preset.
    let current_prompt = job_payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or(payload.prompt.as_str());
    let composed = compose_styled_prompt(&style_text, current_prompt);
    job_payload.insert("prompt".to_owned(), Value::String(composed));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Value {
        json!({
            "schemaVersion": 1,
            "groups": [
                {
                    "id": "anime-style",
                    "name": "Anime Style",
                    "description": "broad anime look",
                    "styles": [
                        { "id": "ghibli-style", "name": "Ghibli", "prompt": "gentle hand-painted" }
                    ]
                }
            ]
        })
    }

    #[test]
    fn resolves_sub_style_id_to_its_prompt() {
        assert_eq!(
            style_text_for_id(&catalog(), "ghibli-style").as_deref(),
            Some("gentle hand-painted")
        );
    }

    #[test]
    fn resolves_group_id_to_its_description() {
        assert_eq!(
            style_text_for_id(&catalog(), "anime-style").as_deref(),
            Some("broad anime look")
        );
    }

    #[test]
    fn unknown_id_resolves_to_none() {
        assert!(style_text_for_id(&catalog(), "does-not-exist").is_none());
    }
}
