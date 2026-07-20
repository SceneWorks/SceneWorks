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

/// Server-side Style fold for an image job (sc-13134, extended for structured captions in sc-13224):
/// when the request carries a top-level `styleId` AND the client did NOT already compose the prompt
/// (mirrors the `presetPromptResolvedClientSide` skip in `apply_recipe_preset_to_image_payload`),
/// resolve the id to its catalog style text and apply it to the current `prompt`. A prose prompt is
/// spliced into the `Subject:`/`Style:` template with `compose_styled_prompt`; a structured
/// JSON-caption prompt (Ideogram 4) has the style MERGED into `style_description.aesthetics` via
/// `merge_style_into_caption` and re-serialized. Both are the Rust twins of the web path, so a
/// headless/MCP client gets a byte-identical result to the studio.
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
    let composed = apply_style_text_to_prompt(&style_text, current_prompt);
    job_payload.insert("prompt".to_owned(), Value::String(composed));
    Ok(())
}

/// Apply a resolved catalog style text to a single prompt — the pure core of the fold, shared and
/// unit-tested without an `AppState`. A structured JSON-caption prompt (Ideogram 4, sc-13224) gets the
/// style MERGED into `style_description.aesthetics` and re-serialized (the server twin of the web's
/// `injectStyleIntoCaption` → `serializeCaption`); a prose prompt is spliced into the
/// `Subject:`/`Style:` template. Byte-identical to the web path in either branch.
fn apply_style_text_to_prompt(style_text: &str, prompt: &str) -> String {
    if let Ok(caption) = serde_json::from_str::<Value>(prompt) {
        if sceneworks_core::ideogram_caption::is_caption(&caption) {
            let injected =
                sceneworks_core::ideogram_caption::merge_style_into_caption(&caption, style_text);
            return sceneworks_core::ideogram_caption::serialize_caption(&injected, false);
        }
    }
    compose_styled_prompt(style_text, prompt)
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

    #[test]
    fn prose_prompt_gets_the_subject_style_template() {
        // A non-caption prompt keeps the sc-13134 prose fold: Subject leads, Style trails.
        assert_eq!(
            apply_style_text_to_prompt("cinematic watercolor", "a fox in the snow"),
            "Subject: a fox in the snow\nStyle: cinematic watercolor"
        );
    }

    #[test]
    fn caption_prompt_merges_into_aesthetics() {
        // A structured JSON-caption prompt (sc-13224): the style merges into aesthetics (user words
        // first) and the caption is re-serialized in canonical order — NOT wrapped in a prose template.
        let caption = r#"{"style_description": {"aesthetics": "moody", "lighting": "low key", "photo": "f/1.8"}, "compositional_deconstruction": {"background": "an alley", "elements": []}}"#;
        let out = apply_style_text_to_prompt("cinematic watercolor", caption);
        assert_eq!(
            out,
            r#"{"style_description": {"aesthetics": "moody. cinematic watercolor", "lighting": "low key", "photo": "f/1.8"}, "compositional_deconstruction": {"background": "an alley", "elements": []}}"#
        );
        // The result is still a caption and the discriminator is untouched.
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(sceneworks_core::ideogram_caption::is_caption(&parsed));
        let style = parsed
            .get("style_description")
            .and_then(Value::as_object)
            .unwrap();
        assert!(style.contains_key("photo"));
        assert!(!style.contains_key("art_style"));
    }

    #[test]
    fn caption_prompt_with_absent_aesthetics_sets_it_as_first_key() {
        // art_style (non-photo) block with no aesthetics → aesthetics is set as the first style key,
        // canonical order (aesthetics, lighting, medium, art_style, color_palette).
        let caption = r#"{"style_description": {"medium": "paint", "art_style": "watercolor"}, "compositional_deconstruction": {"background": "a meadow", "elements": []}}"#;
        let out = apply_style_text_to_prompt("bold ink linework", caption);
        assert_eq!(
            out,
            r#"{"style_description": {"aesthetics": "bold ink linework", "medium": "paint", "art_style": "watercolor"}, "compositional_deconstruction": {"background": "a meadow", "elements": []}}"#
        );
    }
}
