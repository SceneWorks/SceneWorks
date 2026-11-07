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
/// is only the never-seeded edge). The shared manifest loader parses JSONC and rejects non-object
/// roots; the on-disk catalog is reloaded so local overrides remain fresh.
pub(crate) async fn styles_catalog(state: &AppState) -> Result<Value, ApiError> {
    let path = state
        .settings
        .config_dir
        .join("manifests")
        .join("builtin.styles.jsonc");
    let catalog = crate::manifest::load_manifest_root(&path).await?;
    if catalog.is_empty() && !path.exists() {
        return Ok(json!({ "schemaVersion": 1, "groups": [] }));
    }
    Ok(Value::Object(catalog))
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
/// composed with the Description/Style fallback or existing directives by `compose_styled_prompt`;
/// a structured JSON-caption prompt (Ideogram 4) has the style MERGED into `style_description.aesthetics` via
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
    if let Some(selection) = payload
        .appearance
        .as_ref()
        .filter(|value| !value.is_empty())
    {
        return apply_appearance_to_payload(
            state,
            &payload.model,
            selection,
            payload.style_id.as_deref(),
            payload.preset_prompt_resolved_client_side.unwrap_or(false),
            job_payload,
        )
        .await;
    }
    let Some(style_id) = payload
        .style_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(());
    };

    // Booru-tag models (Anima, Illustrious — `captionStyle: "tags"`) take comma-separated tags, not
    // prose, and every catalog style is a 600–900-char English prose paragraph. The studio hides the
    // axis for them, so a `styleId` can only reach here from a headless/MCP caller: reject it by name
    // rather than silently dropping the field, matching the unknown-style 400 below and the audio
    // route's type gate (`generation.rs`).
    //
    // This runs BEFORE the client-side-resolved skip on purpose. That skip is a *fold* detail — it
    // says "the prompt is already composed" — whereas this is a capability gate on the model, so a
    // caller that sets both the flag and a styleId still gets told rather than quietly rendering a
    // prose-wrapped prompt through a tag checkpoint.
    //
    // Keyed off the POST-preset `job_payload["model"]`, NOT `payload.model`: the recipe-preset fold
    // that ran before us may have substituted the preset's own model when the caller named none, and
    // reading the stale DTO field would resolve the DEFAULT model's entry (sc-12300). An unknown id
    // resolves to `{}`, so it falls through to the normal path instead of a spurious rejection.
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(payload.model.as_str())
        .to_owned();
    let model_entry = crate::models::resolve_model_manifest_entry(state, &model_id).await?;
    if model_entry.get("captionStyle").and_then(Value::as_str) == Some("tags") {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} uses tag-style prompts, so the Style catalog does not apply (remove styleId)"
        )));
    }

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
/// Description/Style fallback or structured directives. Byte-identical to the web path in either branch.
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

/// The same built-in catalog is imported by the browser and available to headless clients.
pub(crate) async fn list_appearance() -> Result<Json<Value>, ApiError> {
    Ok(Json(
        serde_json::from_str(sceneworks_core::appearance::CATALOG_JSON)
            .map_err(|err| ApiError::internal(err.to_string()))?,
    ))
}

pub(crate) async fn apply_appearance_to_payload(
    state: &AppState,
    fallback_model: &str,
    selection: &sceneworks_core::appearance::AppearanceSelection,
    style_id: Option<&str>,
    already_resolved: bool,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    use sceneworks_core::{
        appearance::{compose_negative_prompt, resolve_appearance_for_family},
        style_composer::compose_prompt_with_directives,
    };
    if selection.is_empty() {
        return Ok(());
    }
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_owned();
    let model_entry = crate::models::resolve_model_manifest_entry(state, &model_id).await?;
    if model_entry.get("captionStyle").and_then(Value::as_str) == Some("tags") {
        return Err(ApiError::bad_request(
            "Appearance recipes require a prose-prompt model.",
        ));
    }
    let resolved = resolve_appearance_for_family(
        &model_id,
        model_entry.get("family").and_then(Value::as_str),
        selection,
    )
    .map_err(ApiError::bad_request)?;
    if let Some(inputs) = job_payload
        .get("advanced")
        .and_then(|value| value.get("mossImageAdvanced"))
    {
        if ["positive", "negative"]
            .iter()
            .any(|key| inputs.get(*key).is_some_and(|value| !value.is_null()))
            || inputs
                .get("negativePrompts")
                .and_then(Value::as_array)
                .is_some_and(|values| !values.is_empty())
        {
            return Err(ApiError::bad_request("Appearance recipes cannot replace supplied MOSS embedding or per-prompt negative inputs."));
        }
    }

    // Validate model/IDs even on already composed web requests; skip only the text fold.
    if already_resolved {
        return Ok(());
    }
    let style = if let Some(id) = style_id.filter(|id| !id.trim().is_empty()) {
        style_text_for_id(&styles_catalog(state).await?, id)
            .ok_or_else(|| ApiError::bad_request(format!("Style not found: {id}")))?
    } else {
        String::new()
    };
    let raw_prompt = job_payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let raw_batch = job_payload.get("prompts").cloned();
    let raw_negative = job_payload
        .get("negativePrompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let compose = |prompt: &str| -> Result<Value, ApiError> {
        let text = compose_prompt_with_directives(&style, prompt, &resolved.directives);
        if text.chars().count() > 4000 {
            return Err(ApiError::bad_request(
                "Appearance-composed prompt must be at most 4000 characters.",
            ));
        }
        Ok(Value::String(text))
    };
    job_payload.insert("prompt".into(), compose(&raw_prompt)?);
    if let Some(Value::Array(prompts)) = job_payload.get_mut("prompts") {
        for prompt in prompts {
            if let Some(text) = prompt.as_str() {
                *prompt = compose(text)?;
            }
        }
    }
    job_payload.insert(
        "negativePrompt".into(),
        Value::String(compose_negative_prompt(
            &raw_negative,
            if job_payload
                .get("advanced")
                .and_then(|value| value.get("mossImage"))
                .and_then(|value| value.get("acceleration"))
                .and_then(Value::as_str)
                .is_some_and(|mode| mode != "off")
            {
                ""
            } else {
                &resolved.negative
            },
        )),
    );
    let advanced = job_payload.entry("advanced").or_insert_with(|| json!({}));
    let advanced = advanced
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("advanced must be an object"))?;
    advanced.insert(
        "appearance".into(),
        serde_json::to_value(selection).map_err(|err| ApiError::internal(err.to_string()))?,
    );
    if let Some(batch) = raw_batch {
        advanced.insert("appearancePromptBatch".into(), batch);
    }
    if let Some(id) = style_id {
        advanced.insert("styleId".into(), Value::String(id.to_owned()));
        advanced.insert("stylePrompt".into(), Value::String(raw_prompt.clone()));
    }
    advanced.insert("appearancePrompt".into(), Value::String(raw_prompt));
    advanced.insert(
        "appearanceNegativePrompt".into(),
        Value::String(raw_negative),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn filesystem_style_catalog_stays_fresh_and_rejects_invalid_roots() {
        let temp = tempfile::tempdir().unwrap();
        let (_, state) =
            create_app_with_state(crate::tests::support::test_settings(&temp)).unwrap();
        let path = state
            .settings
            .config_dir
            .join("manifests/builtin.styles.jsonc");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, catalog().to_string()).unwrap();
        assert_eq!(styles_catalog(&state).await.unwrap(), catalog());
        let changed = json!({"schemaVersion": 2, "groups": []});
        std::fs::write(&path, format!("// local override\n{changed}")).unwrap();
        assert_eq!(styles_catalog(&state).await.unwrap(), changed);
        for raw in ["[]", "null", "not JSON"] {
            std::fs::write(&path, raw).unwrap();
            assert!(styles_catalog(&state).await.is_err());
        }
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            styles_catalog(&state).await.unwrap(),
            json!({"schemaVersion":1,"groups":[]})
        );
    }

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
    fn prose_prompt_gets_the_description_style_template() {
        // An unstructured prose prompt uses Description followed by the selected Style.
        assert_eq!(
            apply_style_text_to_prompt("cinematic watercolor", "a fox in the snow"),
            "Description: a fox in the snow\nStyle: cinematic watercolor"
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
    #[tokio::test]
    async fn appearance_headless_fold_matches_core_and_records_raw_inputs() {
        let temp = tempfile::tempdir().unwrap();
        let (_, state) =
            create_app_with_state(crate::tests::support::test_settings(&temp)).unwrap();
        let selection = sceneworks_core::appearance::AppearanceSelection {
            film: Some("film-kodak-tri-x-400".into()),
            looks: vec!["look-moody-dark".into()],
            ..Default::default()
        };
        let mut body = json!({"model":"moss_image_base","prompt":"a dancer","prompts":["a dancer","a fox"],"negativePrompt":"authored","advanced":{}}).as_object().unwrap().clone();
        apply_appearance_to_payload(
            &state,
            "moss_image_base",
            &selection,
            None,
            false,
            &mut body,
        )
        .await
        .unwrap();
        assert!(body["prompt"]
            .as_str()
            .unwrap()
            .starts_with("Description: a dancer\nFilm:"));
        assert_eq!(body["prompt"], body["prompts"][0]);
        assert!(body["prompts"][1]
            .as_str()
            .unwrap()
            .starts_with("Description: a fox\nFilm:"));
        assert!(body["negativePrompt"]
            .as_str()
            .unwrap()
            .starts_with("authored, chromatic color"));
        assert_eq!(body["advanced"]["appearancePrompt"], "a dancer");
        assert_eq!(body["advanced"]["appearanceNegativePrompt"], "authored");
        assert_eq!(
            body["advanced"]["appearancePromptBatch"],
            json!(["a dancer", "a fox"])
        );
        let mut too_long = body.clone();
        too_long.insert("prompt".into(), json!("x".repeat(3999)));
        assert!(apply_appearance_to_payload(
            &state,
            "moss_image_base",
            &selection,
            None,
            false,
            &mut too_long
        )
        .await
        .is_err());
        let before = body.clone();
        apply_appearance_to_payload(&state, "moss_image_base", &selection, None, true, &mut body)
            .await
            .unwrap();
        assert_eq!(body, before);
        body.insert("model".into(), json!("z_image_turbo"));
        let incompatible = sceneworks_core::appearance::AppearanceSelection {
            looks: vec!["look-light-leaks".into()],
            ..Default::default()
        };
        assert!(apply_appearance_to_payload(
            &state,
            "z_image_turbo",
            &incompatible,
            None,
            true,
            &mut body
        )
        .await
        .is_err());
        body.insert(
            "advanced".into(),
            json!({"mossImageAdvanced":{"positive":{}}}),
        );
        body.insert("model".into(), json!("moss_image_base"));
        assert!(apply_appearance_to_payload(
            &state,
            "moss_image_base",
            &selection,
            None,
            true,
            &mut body
        )
        .await
        .is_err());
    }
}
