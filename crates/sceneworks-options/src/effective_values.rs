//! Lower accepted authored values into the effective binding view. The caller keeps the original
//! request for provenance. Provider defaults are deliberately resolved by their portable owners,
//! not inferred from prose in the declaration catalog.
use crate::{contracts::JsonObject, DefaultRule, OptionContext, OptionSpec, Shape};
use serde_json::{Map, Value};

/// Syntax-only compatibility helpers for bare route probes as well as full resolution.
/// Callers retain absence/null/invalid as distinct states; these helpers insert no defaults.
pub fn legacy_integer(value: &Value) -> Option<Value> {
    value
        .as_i64()
        .map(Value::from)
        .or_else(|| value.as_u64().map(Value::from))
        .or_else(|| {
            let text = value.as_str()?.trim();
            text.parse::<i64>()
                .ok()
                .map(Value::from)
                .or_else(|| text.parse::<u64>().ok().map(Value::from))
        })
}
pub fn legacy_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())
        .filter(|value| value.is_finite())
}

/// Legacy image-flag truthiness only; full validation separately refuses unsupported shapes.
/// Null and objects are false, as in the historical bare routing probes.
pub fn legacy_image_flag(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        _ => false,
    }
}

pub fn resolve_advanced(payload: &mut JsonObject, entry: &JsonObject) {
    resolve_advanced_inner(payload, entry, true);
}
/// Utility carriers lower authored registered fields without inventing image-generation defaults.
pub fn resolve_present_advanced(payload: &mut JsonObject, entry: &JsonObject) {
    resolve_advanced_inner(payload, entry, false);
}
fn resolve_advanced_inner(payload: &mut JsonObject, entry: &JsonObject, defaults: bool) {
    let model = Value::Object(entry.clone());
    let operation = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let context = OptionContext {
        operation: &operation,
        model: &model,
        features: model.get("features").unwrap_or(&Value::Null),
        target: &Value::Null,
    };
    let advanced = payload
        .entry("advanced")
        .or_insert_with(|| Value::Object(Map::new()));
    if advanced.is_null() {
        *advanced = Value::Object(Map::new());
    }
    for spec in crate::registry()
        .filter(|spec| spec.location == "advanced" || spec.location.starts_with("advanced."))
    {
        if !spec.user_settable {
            continue;
        }
        let path = spec.location.strip_prefix("advanced.").unwrap_or("");
        let segments: Vec<_> = path.split('.').filter(|part| !part.is_empty()).collect();
        visit(advanced, &segments, spec, &context, defaults);
    }
}
fn visit(
    value: &mut Value,
    path: &[&str],
    spec: &OptionSpec,
    context: &OptionContext<'_>,
    defaults: bool,
) {
    if let Some((segment, rest)) = path.split_first() {
        let key = segment.trim_end_matches("[]");
        if let Some(child) = value.get_mut(key) {
            if segment.ends_with("[]") {
                if let Some(values) = child.as_array_mut() {
                    for child in values {
                        visit(child, rest, spec, context, defaults);
                    }
                }
            } else {
                visit(child, rest, spec, context, defaults);
            }
        }
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    // A present null blocks the verified bound-tier fallback. Absence must remain absent until
    // source binding selects its tier; inserting a model default here would override that fact.
    if spec.id == "advanced.mlxQuantize" && object.get(spec.key).is_none_or(Value::is_null) {
        return;
    }
    let mut effective = object.get(spec.key).cloned().or_else(|| {
        spec.aliases
            .iter()
            .find_map(|key| object.get(*key).cloned())
    });
    if effective.as_ref().is_none_or(Value::is_null) {
        effective = if defaults
            && spec.applies_when.matches(context)
            && !matches!(
                context.model.get("family").and_then(Value::as_str),
                Some("moss_image" | "supra2")
            )
            && context.model.get("id").and_then(Value::as_str) != Some("supra2_img")
        {
            match spec.default {
                DefaultRule::Literal(json) => serde_json::from_str(json).ok(),
                DefaultRule::Model(path) => path
                    .split('.')
                    .try_fold(context.model, |value, key| value.get(key))
                    .cloned(),
                _ => None,
            }
        } else {
            None
        };
        // Some declaration paths identify a compound selection rather than its resolved object.
        // Leave those to the owning structured resolver instead of inserting the wrong wire type.
        effective = effective.filter(|value| matches_shape(value, spec.shape));
    }
    if spec.id == "advanced.captionOverflow" {
        effective = crate::image_sampling_controls::requested_caption_overflow(object)
            .expect("validated caption policy")
            .map(|policy| Value::from(policy.name()));
    }
    if let Some(mut effective) = effective {
        if spec.compatibility.contains("numeric string") && effective.is_string() {
            let parsed = match spec.shape {
                Shape::Integer => legacy_integer(&effective),
                Shape::Number => legacy_number(&effective).map(Value::from),
                _ => None,
            };
            if let Some(parsed) = parsed {
                effective = parsed;
            }
        }
        if spec.shape == Shape::Bool && spec.compatibility.contains("Legacy image flag") {
            effective = Value::Bool(legacy_image_flag(&effective));
        }
        object.insert(spec.key.into(), effective);
    } else {
        object.remove(spec.key);
    }
    for alias in spec.aliases {
        object.remove(*alias);
    }
}
fn matches_shape(value: &Value, shape: Shape) -> bool {
    match shape {
        Shape::Bool => value.is_boolean(),
        Shape::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        Shape::Number => value.is_number(),
        Shape::Text | Shape::OneOf(_) => value.is_string(),
        Shape::Object(_) => value.is_object(),
        Shape::List(_) | Shape::Numbers | Shape::Strings => value.is_array(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn legacy_image_flag_retains_bare_route_truthiness() {
        for value in [
            json!(false),
            json!(0),
            json!(""),
            json!([]),
            json!(null),
            json!({}),
        ] {
            assert!(!legacy_image_flag(&value), "{value}");
        }
        for value in [json!(true), json!(1), json!(-1), json!("false"), json!([0])] {
            assert!(legacy_image_flag(&value), "{value}");
        }
    }
    #[test]
    fn bare_route_numeric_syntax_matches_effective_resolution() {
        assert_eq!(legacy_integer(&json!(" 8 ")), Some(json!(8)));
        assert_eq!(
            legacy_integer(&json!("18446744073709551615")),
            Some(json!(u64::MAX))
        );
        assert_eq!(legacy_integer(&json!("2.5")), None);
        assert_eq!(legacy_number(&json!(" 2.5 ")), Some(2.5));
        for invalid in [json!(null), json!(""), json!("NaN"), json!(true), json!({})] {
            assert!(legacy_integer(&invalid).is_none());
            assert!(legacy_number(&invalid).is_none());
        }
    }
    #[test]
    fn effective_values_lower_accepted_values_without_changing_authored_values() {
        let authored = json!({"advanced":{"steps":"20","guidanceScale":"3.5","usePid":0,"timestepShift":"2","strength":null}});
        let mut effective = authored.as_object().unwrap().clone();
        resolve_advanced(
            &mut effective,
            json!({"type":"image","defaults":{"steps":10},"ui":{"pid":true}})
                .as_object()
                .unwrap(),
        );
        assert_eq!(effective["advanced"]["steps"], 20);
        assert_eq!(effective["advanced"]["guidanceScale"], 3.5);
        assert_eq!(effective["advanced"]["usePid"], false);
        assert_eq!(effective["advanced"]["schedulerShift"], 2.0);
        assert!(effective["advanced"].get("timestepShift").is_none());
        assert!(effective["advanced"].get("strength").is_none());
        assert_eq!(authored["advanced"]["steps"], "20");
    }
    #[test]
    fn quant_null_and_absence_preserve_bound_tier_selection() {
        let entry = json!({"type":"image","mlx":{"quantize":8}});
        for advanced in [
            json!({"mlxQuantize":null,"__portableBoundTier":"q4"}),
            json!({"__portableBoundTier":"q4"}),
        ] {
            let mut payload = json!({"advanced":advanced}).as_object().unwrap().clone();
            resolve_advanced(&mut payload, entry.as_object().unwrap());
            assert_eq!(
                payload["advanced"].get("mlxQuantize"),
                advanced.get("mlxQuantize")
            );
        }
    }
    #[test]
    fn defaults_require_applicable_facts_and_do_not_invent_provider_choices() {
        let mut payload = JsonObject::new();
        resolve_advanced(
            &mut payload,
            json!({"type":"image","defaults":{"steps":28}})
                .as_object()
                .unwrap(),
        );
        assert_eq!(payload["advanced"]["steps"], 28);
        assert!(payload["advanced"].get("vaeDecoder").is_none());
        assert!(payload["advanced"].get("sampler").is_none());
        assert!(payload["advanced"].get("mossImage").is_none());
    }
}
