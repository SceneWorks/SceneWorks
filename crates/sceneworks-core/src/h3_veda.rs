//! Shared API/worker validation for the H3 Veda wire controls. Core stays backend-neutral.
use crate::contracts::JsonObject;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum H3VedaBudget {
    Percent(f64),
    KeyTiles(u32),
}
#[derive(Clone, Debug, PartialEq)]
pub struct H3VedaSettings {
    pub enabled: bool,
    pub generated: H3VedaBudget,
    pub reference: H3VedaBudget,
    pub full_attention_layers: Vec<usize>,
    pub full_attention_steps: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "budget_default")]
    generated: String,
    #[serde(default = "budget_default")]
    reference: String,
    #[serde(default)]
    full_attention_layers: String,
    #[serde(default)]
    full_attention_steps: String,
}
fn budget_default() -> String {
    "90%".into()
}
fn budget(text: &str) -> Result<H3VedaBudget, String> {
    let value = text.trim();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if let Some(percent) = value.strip_suffix(['%', '％']) {
        let mut parts = percent.trim().split('.');
        if !digits(parts.next().unwrap_or_default())
            || parts.next().is_some_and(|s| !digits(s))
            || parts.next().is_some()
        {
            return Err(
                "expected a percentage in [0, 100) or a positive integer key-tile count".into(),
            );
        }
        let p: f64 = percent.trim().parse().map_err(|_| "invalid percentage")?;
        if p.is_finite() && (0.0..100.0).contains(&p) {
            return Ok(H3VedaBudget::Percent(p));
        }
    } else if digits(value) {
        if let Ok(n) = value.parse::<u32>() {
            if n > 0 {
                return Ok(H3VedaBudget::KeyTiles(n));
            }
        }
    }
    Err("expected a percentage in [0, 100) or a positive u32 key-tile count".into())
}
fn indices(text: &str, domain: usize) -> Result<Vec<usize>, String> {
    let chars: Vec<char> = text.chars().collect();
    let separator = |c: char| c.is_whitespace() || matches!(c, ',' | ';' | '，' | '；' | '、');
    let dash = |c: char| matches!(c, '-' | '~' | '–' | '—');
    let number = |cursor: &mut usize| -> Result<usize, String> {
        let start = *cursor;
        while *cursor < chars.len() && chars[*cursor].is_ascii_digit() {
            *cursor += 1;
        }
        if start == *cursor {
            return Err("expected zero-based indices or ranges".into());
        }
        chars[start..*cursor]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| "index exceeds usize".into())
    };
    let mut result = std::collections::BTreeSet::new();
    let mut cursor = 0;
    while cursor < chars.len() {
        if separator(chars[cursor]) {
            cursor += 1;
            continue;
        }
        let first = number(&mut cursor)?;
        let mut look = cursor;
        while look < chars.len() && chars[look].is_whitespace() {
            look += 1;
        }
        let last = if look < chars.len() && dash(chars[look]) {
            cursor = look + 1;
            while cursor < chars.len() && chars[cursor].is_whitespace() {
                cursor += 1;
            }
            number(&mut cursor)?
        } else {
            first
        };
        if cursor < chars.len() && !separator(chars[cursor]) {
            return Err("expected zero-based indices or ranges".into());
        }
        if first.max(last) >= domain {
            return Err(format!(
                "zero-based index {} outside 0..{domain}",
                first.max(last)
            ));
        }
        result.extend(first.min(last)..=first.max(last));
    }
    Ok(result.into_iter().collect())
}
/// Validate present controls, including disabled values, before any native load/allocation.
/// `effective_steps` is the selected accelerator recipe plus any explicit step override.
pub fn requested_h3_veda(
    model: &str,
    advanced: &JsonObject,
    effective_steps: u32,
) -> Result<Option<H3VedaSettings>, String> {
    let Some(value) = advanced.get("veda") else {
        return Ok(None);
    };
    if !matches!(model, "minimax_h3" | "minimax_h3_ref") {
        return Err("advanced.veda is only supported by MiniMax-H3".into());
    }
    // Both native H3 providers declare MAX_STEPS=200 (model.rs). Check the actual
    // schedule before expanding ranges, including disabled persisted controls.
    if !(1..=200).contains(&effective_steps) {
        return Err("advanced.veda: H3 sampling steps must be in 1..=200".into());
    }
    let wire: Wire =
        serde_json::from_value(value.clone()).map_err(|e| format!("advanced.veda: {e}"))?;
    let field = |name: &str, e: String| format!("advanced.veda.{name}: {e}");
    Ok(Some(H3VedaSettings {
        enabled: wire.enabled,
        generated: budget(&wire.generated).map_err(|e| field("generated", e))?,
        reference: budget(&wire.reference).map_err(|e| field("reference", e))?,
        full_attention_layers: indices(&wire.full_attention_layers, 50)
            .map_err(|e| field("fullAttentionLayers", e))?,
        full_attention_steps: indices(&wire.full_attention_steps, effective_steps as usize)
            .map_err(|e| field("fullAttentionSteps", e))?,
    }))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn advanced(v: serde_json::Value) -> JsonObject {
        json!({"veda":v}).as_object().unwrap().clone()
    }
    #[test]
    fn absent_legacy_and_complete_controls() {
        assert_eq!(
            requested_h3_veda("wan", &JsonObject::new(), 50).unwrap(),
            None
        );
        let legacy = requested_h3_veda("minimax_h3", &advanced(json!({"enabled":true})), 8)
            .unwrap()
            .unwrap();
        assert_eq!(legacy.generated, H3VedaBudget::Percent(90.0));
        assert_eq!(legacy.reference, H3VedaBudget::Percent(90.0));
        for model in ["minimax_h3", "minimax_h3_ref"] {
            let v=requested_h3_veda(model,&advanced(json!({"enabled":true,"generated":"32","reference":"0%","fullAttentionLayers":"49, 2–0","fullAttentionSteps":"0; 3"})),4).unwrap().unwrap();
            assert_eq!(v.generated, H3VedaBudget::KeyTiles(32));
            assert_eq!(v.reference, H3VedaBudget::Percent(0.0));
            assert_eq!(v.full_attention_layers, vec![0, 1, 2, 49]);
            assert_eq!(v.full_attention_steps, vec![0, 3]);
        }
    }
    #[test]
    fn malformed_disabled_incompatible_and_schedule_bounds_refused() {
        for wire in [
            json!(null),
            json!(true),
            json!({"enabled":"true"}),
            json!({"unknown":1}),
            json!({"generated":32}),
            json!({"reference":"100%"}),
            json!({"generated":"0"}),
            json!({"generated":"4294967296"}),
            json!({"generated":"1.5"}),
            json!({"generated":"1e1%"}),
            json!({"fullAttentionLayers":"50"}),
            json!({"fullAttentionSteps":"4"}),
            json!({"fullAttentionSteps":"0-999999999"}),
            json!({"fullAttentionSteps":"-1"}),
        ] {
            assert!(
                requested_h3_veda("minimax_h3", &advanced(wire.clone()), 4).is_err(),
                "{wire}"
            );
        }
        let huge = advanced(json!({"enabled":true,"fullAttentionSteps":"0-4294967294"}));
        assert!(requested_h3_veda("minimax_h3", &huge, u32::MAX).is_err());
        assert!(requested_h3_veda(
            "minimax_h3",
            &advanced(json!({"fullAttentionSteps":"0-199"})),
            200
        )
        .is_ok());
        assert!(requested_h3_veda("wan", &advanced(json!({"enabled":false})), 50).is_err());
        assert!(requested_h3_veda(
            "minimax_h3",
            &advanced(json!({"fullAttentionSteps":"7"})),
            8
        )
        .is_ok());
        assert!(requested_h3_veda(
            "minimax_h3",
            &advanced(json!({"fullAttentionSteps":"49"})),
            50
        )
        .is_ok());
    }
}
