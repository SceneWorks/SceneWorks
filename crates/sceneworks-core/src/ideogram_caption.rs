//! Canonical Ideogram 4 JSON-caption serialization — the Rust twin of the web's
//! `apps/web/src/ideogramCaption.js` `orderCaption` + `serializeCaption` + `parseMagicPromptCaption`
//! (epic 4725). Ideogram 4 is trained EXCLUSIVELY on structured JSON captions whose key order is
//! quality-relevant (its `CaptionVerifier`), so a caption handed to the engine must be re-emitted in
//! the canonical key order, with non-schema keys dropped, in Python `json.dumps(ensure_ascii=False)`
//! spacing (`", "` / `": "`) — the exact byte format the model saw in training.
//!
//! The API's headless auto-caption (sc-6519) cleans the magic-prompt utility model's reply through
//! [`serialize_magic_prompt_caption`] so the headless engine payload is byte-for-byte the one the web
//! produces: the stray top-level `aspect_ratio` the prompt emits is dropped, the model's unreliable
//! full-image element bboxes are stripped, and canonical order is imposed.

use serde_json::Value;

/// Canonical top-level caption key order. `compositional_deconstruction` is the only required section;
/// `high_level_description` / `style_description` are optional.
const TOP_LEVEL_KEYS: &[&str] = &[
    "high_level_description",
    "style_description",
    "compositional_deconstruction",
];
/// Style-block key order when the `photo` discriminator is present.
const STYLE_KEY_ORDER_PHOTO: &[&str] =
    &["aesthetics", "lighting", "photo", "medium", "color_palette"];
/// Style-block key order for an `art_style` block — note `art_style` sits AFTER `medium`, not in
/// photo's discriminator slot (the verifier's actual rule).
const STYLE_KEY_ORDER_NON_PHOTO: &[&str] = &[
    "aesthetics",
    "lighting",
    "medium",
    "art_style",
    "color_palette",
];
const ELEMENT_KEY_ORDER_OBJ: &[&str] = &["type", "bbox", "desc", "color_palette"];
const ELEMENT_KEY_ORDER_TEXT: &[&str] = &["type", "bbox", "text", "desc", "color_palette"];

/// True when `value` is a structured caption — a JSON object carrying the `compositional_deconstruction`
/// object the model's `CaptionVerifier` requires. Mirrors the web validator's required-section rule.
pub fn is_caption(value: &Value) -> bool {
    value.as_object().is_some_and(|map| {
        map.get("compositional_deconstruction")
            .is_some_and(Value::is_object)
    })
}

/// Clean + canonicalize a magic-prompt reply for the engine (the web's `parseMagicPromptCaption` +
/// `serializeCaption`): `None` unless `value` is a structured caption; otherwise the canonical-order
/// string with non-schema keys dropped (the stray top-level `aspect_ratio`) and element bboxes
/// stripped (the model's full-image box guesses are unreliable — the reference strips them by
/// default). Byte-for-byte the string the web hands the engine.
pub fn serialize_magic_prompt_caption(value: &Value) -> Option<String> {
    is_caption(value).then(|| serialize_caption(value, true))
}

/// Re-emit a caption in canonical key order with `json.dumps(ensure_ascii=False)` spacing.
/// `strip_bboxes` drops element bboxes (the magic-prompt path); pass `false` to preserve a
/// user-authored caption's boxes verbatim.
pub fn serialize_caption(value: &Value, strip_bboxes: bool) -> String {
    let Some(obj) = value.as_object() else {
        return emit_value(value);
    };
    let mut parts: Vec<String> = Vec::new();
    for &key in TOP_LEVEL_KEYS {
        let Some(field) = obj.get(key) else { continue };
        let emitted = match key {
            "style_description" => emit_ordered(field, style_order(field), &[]),
            "compositional_deconstruction" => emit_composition(field, strip_bboxes),
            _ => emit_value(field),
        };
        parts.push(format!("{}: {emitted}", emit_key(key)));
    }
    format!("{{{}}}", parts.join(", "))
}

/// Emit the `compositional_deconstruction` section: `background` then `elements`, each element keyed
/// in its obj/text canonical order (and bboxes stripped when `strip_bboxes`).
fn emit_composition(cd: &Value, strip_bboxes: bool) -> String {
    let Some(obj) = cd.as_object() else {
        return emit_value(cd);
    };
    let mut parts: Vec<String> = Vec::new();
    if let Some(background) = obj.get("background") {
        parts.push(format!(
            "{}: {}",
            emit_key("background"),
            emit_value(background)
        ));
    }
    if let Some(elements) = obj.get("elements") {
        let emitted = match elements.as_array() {
            Some(items) => {
                let els: Vec<String> = items
                    .iter()
                    .map(|el| emit_element(el, strip_bboxes))
                    .collect();
                format!("[{}]", els.join(", "))
            }
            None => emit_value(elements),
        };
        parts.push(format!("{}: {emitted}", emit_key("elements")));
    }
    format!("{{{}}}", parts.join(", "))
}

fn emit_element(el: &Value, strip_bboxes: bool) -> String {
    let order = if el.get("type").and_then(Value::as_str) == Some("text") {
        ELEMENT_KEY_ORDER_TEXT
    } else {
        ELEMENT_KEY_ORDER_OBJ
    };
    let skip: &[&str] = if strip_bboxes { &["bbox"] } else { &[] };
    emit_ordered(el, order, skip)
}

/// Emit an object's keys in `order`, skipping `skip` and any key not present.
fn emit_ordered(value: &Value, order: &[&str], skip: &[&str]) -> String {
    let Some(obj) = value.as_object() else {
        return emit_value(value);
    };
    let parts: Vec<String> = order
        .iter()
        .filter(|key| !skip.contains(*key))
        .filter_map(|&key| {
            obj.get(key)
                .map(|field| format!("{}: {}", emit_key(key), emit_value(field)))
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// Photo discriminator wins (both present, or photo-only); an `art_style`-only block uses the
/// non-photo order — mirrors the web's `styleOrderFor`.
fn style_order(style: &Value) -> &'static [&'static str] {
    if style.get("photo").is_some() {
        STYLE_KEY_ORDER_PHOTO
    } else {
        STYLE_KEY_ORDER_NON_PHOTO
    }
}

/// Emit an arbitrary JSON value with `json.dumps(ensure_ascii=False)` default spacing (`", "` / `": "`),
/// preserving array order and (for an opaque nested object) serde's key order. `serde_json::to_string`
/// already escapes strings the ensure_ascii=False way (literal UTF-8, escaped control / `"` / `\`); we
/// only add the separator spaces.
fn emit_value(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(emit_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, field)| format!("{}: {}", emit_key(key), emit_value(field)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_owned()),
    }
}

fn emit_key(key: &str) -> String {
    serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn is_caption_requires_the_composition_section() {
        assert!(is_caption(
            &json!({"compositional_deconstruction": {"background": "x", "elements": []}})
        ));
        assert!(!is_caption(&json!({"high_level_description": "x"})));
        assert!(!is_caption(
            &json!({"compositional_deconstruction": "not an object"})
        ));
        assert!(!is_caption(&json!("plain text")));
    }

    #[test]
    fn magic_prompt_caption_drops_aspect_ratio_strips_bboxes_and_canonicalizes() {
        // A real magic-prompt reply shape (sc-6519): non-schema `aspect_ratio` first, scrambled
        // top-level order, an obj element with a full-image bbox.
        let raw = json!({
            "aspect_ratio": "1:1",
            "compositional_deconstruction": {
                "background": "a snowy forest",
                "elements": [{"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": "a red fox"}]
            },
            "high_level_description": "a red fox in snow"
        });
        let out = serialize_magic_prompt_caption(&raw).expect("a valid caption");
        assert_eq!(
            out,
            r#"{"high_level_description": "a red fox in snow", "compositional_deconstruction": {"background": "a snowy forest", "elements": [{"type": "obj", "desc": "a red fox"}]}}"#
        );
    }

    #[test]
    fn magic_prompt_caption_rejects_non_captions() {
        assert!(serialize_magic_prompt_caption(&json!({"high_level_description": "x"})).is_none());
        assert!(serialize_magic_prompt_caption(&json!("plain text")).is_none());
    }

    #[test]
    fn style_block_uses_photo_then_non_photo_order() {
        // A photo style block keeps the photo discriminator order; key insertion order is irrelevant.
        let photo = json!({
            "style_description": {"medium": "DSLR", "photo": "f/2.8", "aesthetics": "moody", "lighting": "golden hour"},
            "compositional_deconstruction": {"background": "snow", "elements": []}
        });
        let out = serialize_caption(&photo, true);
        assert_eq!(
            out,
            r#"{"style_description": {"aesthetics": "moody", "lighting": "golden hour", "photo": "f/2.8", "medium": "DSLR"}, "compositional_deconstruction": {"background": "snow", "elements": []}}"#
        );

        // An art_style block puts art_style AFTER medium.
        let art = json!({
            "style_description": {"art_style": "watercolor", "medium": "paint", "aesthetics": "soft", "lighting": "diffuse"},
            "compositional_deconstruction": {"background": "meadow", "elements": []}
        });
        let out = serialize_caption(&art, true);
        assert!(
            out.contains(r#""lighting": "diffuse", "medium": "paint", "art_style": "watercolor""#)
        );
    }

    #[test]
    fn user_caption_can_preserve_bboxes() {
        let caption = json!({
            "compositional_deconstruction": {
                "background": "a beach",
                "elements": [{"type": "obj", "bbox": [100, 100, 200, 200], "desc": "a shell"}]
            }
        });
        let out = serialize_caption(&caption, false);
        assert!(out.contains(r#""bbox": [100, 100, 200, 200]"#));
    }

    #[test]
    fn emit_spacing_matches_python_json_dumps() {
        // Compact with a space after ',' and ':' — and literal (non-escaped) UTF-8.
        let value =
            json!({"compositional_deconstruction": {"background": "café — déjà", "elements": []}});
        let out = serialize_caption(&value, true);
        assert_eq!(
            out,
            r#"{"compositional_deconstruction": {"background": "café — déjà", "elements": []}}"#
        );
    }
}
