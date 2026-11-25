//! Prompt-authored adapters use the same catalog and admission checks as selected adapters.
use super::*;

#[derive(Debug, PartialEq)]
pub(crate) struct Directive {
    name: String,
    strength: f64,
}

/// Remove complete directives, preserving all other text (including whitespace and JSON).
/// An incomplete `<lora:` remains literal text. Invalid complete directives are ignored.
fn take_from_text(text: &str, directives: &mut Vec<Directive>) -> String {
    let mut rest = text;
    let mut cleaned = String::with_capacity(text.len());
    while let Some(start) = rest.find("<lora:") {
        cleaned.push_str(&rest[..start]);
        let body = &rest[start + "<lora:".len()..];
        // Do not consume a later valid tag as part of an incomplete one.
        let Some(end) = body
            .find(['<', '>'])
            .filter(|&end| body.as_bytes()[end] == b'>')
        else {
            cleaned.push_str("<lora:");
            rest = body;
            continue;
        };
        let directive = body[..end].rsplit_once(':').and_then(|(name, strength)| {
            let name = name.trim();
            let strength = strength.trim().parse::<f64>().ok()?;
            // Engines consume f32 scales. Never admit NaN, infinity, or an overflowing scale.
            (!name.is_empty() && strength.is_finite() && (strength as f32).is_finite()).then(|| {
                Directive {
                    name: name.to_owned(),
                    strength,
                }
            })
        });
        match directive {
            Some(directive) => directives.push(directive),
            None => tracing::warn!(
                event = "prompt_lora_ignored",
                reason = "invalid name or strength",
                "Ignoring malformed prompt LoRA directive"
            ),
        }
        rest = &body[end + 1..];
    }
    cleaned.push_str(rest);
    cleaned
}

pub(crate) fn take_from_payload(payload: &mut JsonObject) -> Vec<Directive> {
    let mut directives = Vec::new();
    for field in ["prompt", "negativePrompt"] {
        if let Some(Value::String(text)) = payload.get_mut(field) {
            if text.contains("<lora:") {
                *text = take_from_text(text, &mut directives);
            }
        }
    }
    // Native prompt batches share one adapter set; ordinary studio batches submit separate jobs.
    if let Some(prompts) = payload.get_mut("prompts").and_then(Value::as_array_mut) {
        for prompt in prompts {
            if let Some(text) = prompt.as_str().filter(|text| text.contains("<lora:")) {
                *prompt = Value::String(take_from_text(text, &mut directives));
            }
        }
    }
    directives
}

/// Called on the blocking pool: the shared validator may inspect adapter headers.
pub(crate) fn append_resolved(
    models: &[Value],
    catalog: &[Value],
    model_id: &str,
    directives: &[Directive],
    selected: &mut Vec<Value>,
) {
    let mut attempted = std::collections::HashSet::new();
    for directive in directives {
        if !attempted.insert(directive.name.to_ascii_lowercase()) {
            continue;
        }
        let matches = |lora: &&Value| {
            ["name", "id"].iter().any(|key| {
                lora.get(*key)
                    .and_then(Value::as_str)
                    .is_some_and(|name| name.trim().eq_ignore_ascii_case(&directive.name))
            })
        };
        // A selected catalog identity always wins, even if multiple adapters share a name.
        if catalog.iter().filter(matches).any(|candidate| {
            selected
                .iter()
                .any(|lora| job_lora_id(lora) == job_lora_id(candidate))
        }) || selected.iter().any(|lora| {
            job_lora_id(lora).is_some_and(|id| id.eq_ignore_ascii_case(&directive.name))
                || lora
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name.trim().eq_ignore_ascii_case(&directive.name))
        }) {
            continue;
        }
        let mut resolved = None;
        for candidate in catalog.iter().filter(matches) {
            let Some(id) = candidate.get("id").and_then(Value::as_str) else {
                continue;
            };
            let spec = json!({"id": id, "weight": directive.strength});
            if let Ok(mut normalized) =
                validate_lora_specs_for_model(models, catalog, model_id, &[spec], false, "LoRA")
            {
                if let Some(lora) = normalized.pop() {
                    resolved = Some(lora);
                    break;
                }
            }
        }
        if let Some(lora) = resolved {
            selected.push(lora);
        } else {
            tracing::warn!(
                event = "prompt_lora_ignored", lora = %directive.name, model = model_id,
                "Ignoring prompt LoRA: no installed compatible adapter found for this model"
            );
        }
    }
}

/// Replay partitions trusted inline adapters before validating them. Resolve prompt additions
/// first, so new names still require catalog backing and cannot inherit an inline path permit.
pub(crate) async fn resolve_for_inline_replay(
    state: &AppState,
    project_id: Option<&str>,
    payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let directives = take_from_payload(payload);
    if directives.is_empty() {
        return Ok(());
    }
    let model_id = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let models = model_catalog(state).await?;
    let catalog = lora_catalog(state, project_id).await?;
    let mut selected = payload
        .get("loras")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let selected = tokio::task::spawn_blocking(move || {
        append_resolved(&models, &catalog, &model_id, &directives, &mut selected);
        selected
    })
    .await
    .map_err(|error| ApiError::internal(format!("Prompt LoRA resolution failed: {error}")))?;
    payload.insert("loras".to_owned(), Value::Array(selected));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_scales_without_rewriting_surrounding_text() {
        let mut directives = Vec::new();
        assert_eq!(
            take_from_text(
                "café <lora: Direct Flash :2>雨<lora:soft:-.5><lora:zero:0><lora:exp:1e-2>",
                &mut directives
            ),
            "café 雨"
        );
        assert_eq!(
            directives,
            vec![
                Directive {
                    name: "Direct Flash".into(),
                    strength: 2.0
                },
                Directive {
                    name: "soft".into(),
                    strength: -0.5
                },
                Directive {
                    name: "zero".into(),
                    strength: 0.0
                },
                Directive {
                    name: "exp".into(),
                    strength: 0.01
                },
            ]
        );
    }

    #[test]
    fn ignores_invalid_directives_and_preserves_incomplete_ones() {
        let mut directives = Vec::new();
        assert_eq!(take_from_text("<lora:x:NaN><lora:x:inf><lora:x:1e100><lora:x:bad><lora::1><lora:x><lora:unfinished", &mut directives), "<lora:unfinished");
        assert!(directives.is_empty());
        assert_eq!(
            take_from_text("<lora:unfinished <lora:ok:1>", &mut directives),
            "<lora:unfinished "
        );
        assert_eq!(directives.len(), 1);
    }

    #[test]
    fn cleans_all_prompt_carriers_and_is_idempotent() {
        let mut payload = json!({
            "prompt": "a <lora:first:2>", "negativePrompt": "b<lora:second:.5>",
            "prompts": ["a <lora:first:2>", "c<lora:third:0>"],
        })
        .as_object()
        .unwrap()
        .clone();
        assert_eq!(take_from_payload(&mut payload).len(), 4);
        assert_eq!(payload["prompt"], "a ");
        assert_eq!(payload["negativePrompt"], "b");
        assert_eq!(payload["prompts"], json!(["a ", "c"]));
        let entry = json!({"image":{"maxPromptBatchImages":8}});
        assert!(
            sceneworks_core::image_sampling_controls::prompt_batch_error(
                "batch_model",
                payload["prompt"].as_str().unwrap(),
                &["a ".into(), "c".into()],
                1,
                entry.as_object().unwrap(),
            )
            .is_none()
        );
        assert!(take_from_payload(&mut payload).is_empty());
    }

    #[test]
    fn preserves_structured_captions() {
        let mut directives = Vec::new();
        let cleaned = take_from_text(
            r#"{"scene":"rain<lora:Flash:2>","objects":["umbrella"]}"#,
            &mut directives,
        );
        assert_eq!(
            serde_json::from_str::<Value>(&cleaned).unwrap(),
            json!({"scene":"rain","objects":["umbrella"]})
        );
        assert_eq!(directives.len(), 1);
    }

    #[test]
    fn resolves_for_model_and_keeps_selected_strength_and_first_directive() {
        let models = vec![
            json!({"id":"image", "family":"z-image", "loraCompatibility":{"families":["z-image"]}}),
        ];
        let catalog = vec![
            json!({"id":"wrong", "name":"DirectFlash", "family":"qwen-image", "installState":"installed"}),
            json!({"id":"partition", "name":"DirectFlash", "family":"z-image", "modelIds":["other"], "installState":"installed"}),
            json!({"id":"flash", "name":"DirectFlash", "family":"z-image", "installState":"installed"}),
            json!({"id":"soft", "name":"Soft Light", "family":"z-image", "installState":"installed"}),
            json!({"id":"absent", "name":"Absent", "family":"z-image", "installState":"not_installed"}),
        ];
        let mut directives = Vec::new();
        take_from_text("<lora:directflash:2><lora:flash:3><lora:Soft Light:-.5><lora:soft:1><lora:Absent:1><lora:missing:1>", &mut directives);
        let mut selected = vec![json!({"id":"flash", "weight":0.25})];
        append_resolved(&models, &catalog, "image", &directives, &mut selected);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0]["weight"], 0.25);
        assert_eq!(selected[1]["id"], "soft");
        assert_eq!(selected[1]["weight"], -0.5);
        let mut selected = Vec::new();
        append_resolved(&models, &catalog, "image", &directives, &mut selected);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0]["id"], "flash");
        assert_eq!(selected[0]["weight"], 2.0);
    }
}
