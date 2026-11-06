//! Shared model-aware photography recipes. This module only composes text; it runs no inference.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

pub const CATALOG_JSON: &str = include_str!("../../../config/manifests/builtin.appearance.json");
static CATALOG: LazyLock<Value> =
    LazyLock::new(|| serde_json::from_str(CATALOG_JSON).expect("built-in appearance catalog"));

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppearanceSelection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub film: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub looks: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lighting: Option<String>,
}
impl AppearanceSelection {
    pub fn is_empty(&self) -> bool {
        self.camera.is_none()
            && self.film.is_none()
            && self.lighting.is_none()
            && self.looks.is_empty()
    }
}
#[derive(Debug, Default)]
pub struct ResolvedAppearance {
    pub directives: BTreeMap<String, String>,
    pub negative: String,
}

pub fn resolve_appearance(
    model: &str,
    selection: &AppearanceSelection,
) -> Result<ResolvedAppearance, String> {
    resolve_appearance_for_family(model, None, selection)
}

/// Registered fine-tunes inherit their canonical model family's recipe profile.
pub fn resolve_appearance_for_family(
    model: &str,
    family: Option<&str>,
    selection: &AppearanceSelection,
) -> Result<ResolvedAppearance, String> {
    if selection.is_empty() {
        return Ok(ResolvedAppearance::default());
    }
    let profiles = CATALOG["profiles"].as_object().expect("profiles object");
    let (profile_key, profile) = profiles
        .iter()
        .find(|(_, profile)| {
            profile["models"]
                .as_array()
                .is_some_and(|models| models.iter().any(|id| id.as_str() == Some(model)))
                || family.is_some_and(|family| {
                    profile["families"].as_array().is_some_and(|families| {
                        families.iter().any(|value| value.as_str() == Some(family))
                    })
                })
        })
        .ok_or_else(|| {
            "Camera, Film, Looks and Lighting recipes are unavailable for this model.".to_owned()
        })?;
    let entries = CATALOG["entries"].as_array().expect("catalog entries");
    let mut resolved = ResolvedAppearance::default();
    let mut negatives: Vec<String> = Vec::new();
    for (axis, key, ids) in [
        (
            "camera",
            "Camera",
            selection.camera.iter().collect::<Vec<_>>(),
        ),
        ("film", "Film", selection.film.iter().collect()),
        ("look", "Look", selection.looks.iter().collect()),
        ("lighting", "Lighting", selection.lighting.iter().collect()),
    ] {
        let mut seen = HashSet::new();
        let mut prompts = Vec::new();
        for id in ids {
            if !seen.insert(id) {
                continue;
            }
            let variant = entries
                .iter()
                .find(|entry| {
                    entry["id"].as_str() == Some(id.as_str())
                        && entry["axis"].as_str() == Some(axis)
                })
                .and_then(|entry| entry["variants"].get(profile_key))
                .ok_or_else(|| format!("Appearance recipe {id} is unavailable for this model."))?;
            prompts.push(
                variant["prompt"]
                    .as_str()
                    .ok_or_else(|| "Appearance recipe has no prompt.".to_owned())?
                    .to_owned(),
            );
            if profile["negative"].as_bool() == Some(true) {
                if let Some(negative) = variant["negative"].as_str().filter(|text| !text.is_empty())
                {
                    if !negatives.iter().any(|part| part == negative) {
                        negatives.push(negative.to_owned());
                    }
                }
            }
        }
        if !prompts.is_empty() {
            resolved
                .directives
                .insert(key.to_owned(), prompts.join(" "));
        }
    }
    resolved.negative = negatives.join(", ");
    Ok(resolved)
}

pub fn compose_negative_prompt(user_prompt: &str, recipe_negative: &str) -> String {
    if recipe_negative.is_empty() {
        user_prompt.to_owned()
    } else if user_prompt.trim().is_empty() {
        recipe_negative.to_owned()
    } else {
        format!("{user_prompt}, {recipe_negative}")
    }
}

/// Public tool schema for the same typed request accepted by the REST API.
pub fn selection_schema() -> Value {
    serde_json::json!({"type":"object","additionalProperties":false,"properties":{
        "camera":{"type":["string","null"]},"film":{"type":["string","null"]},"lighting":{"type":["string","null"]},
        "looks":{"type":"array","items":{"type":"string"},"uniqueItems":true}
    }})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_catalog_coverage_and_model_transfer() {
        let entries = CATALOG["entries"].as_array().unwrap();
        for (profile, expected) in [("moss", 52), ("z_image", 42)] {
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry["axis"] != "lighting"
                        && entry["variants"].get(profile).is_some())
                    .count(),
                expected
            );
            assert_eq!(
                entries
                    .iter()
                    .filter(|entry| entry["axis"] == "lighting"
                        && entry["variants"].get(profile).is_some())
                    .count(),
                27
            );
        }
        let selected = AppearanceSelection {
            film: Some("film-kodak-portra-400".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_appearance_for_family("my-finetune", Some("krea_2"), &selected)
                .unwrap()
                .directives,
            resolve_appearance("krea_2_turbo", &selected)
                .unwrap()
                .directives
        );
        assert_eq!(
            resolve_appearance("z_image_turbo", &selected)
                .unwrap()
                .directives,
            resolve_appearance("krea_2_turbo", &selected)
                .unwrap()
                .directives
        );
    }
    #[test]
    fn moss_negative_and_multiple_looks_are_resolved_once() {
        let selected = AppearanceSelection {
            film: Some("film-kodak-tri-x-400".into()),
            looks: vec!["look-moody-dark".into(), "look-moody-dark".into()],
            ..Default::default()
        };
        let resolved = resolve_appearance("moss_image_base", &selected).unwrap();
        assert_eq!(
            resolved.directives["Look"]
                .matches("Photographic rendering")
                .count(),
            1
        );
        assert!(resolved.negative.starts_with("chromatic color"));
        assert_eq!(
            compose_negative_prompt("authored", "recipe"),
            "authored, recipe"
        );
        assert_eq!(compose_negative_prompt("  authored  ", ""), "  authored  ");
    }
    #[test]
    fn invalid_models_ids_and_axes_are_refused() {
        for (model, selected) in [
            (
                "ltx_2_3",
                AppearanceSelection {
                    film: Some("film-kodak-portra-400".into()),
                    ..Default::default()
                },
            ),
            (
                "z_image_turbo",
                AppearanceSelection {
                    camera: Some("film-kodak-portra-400".into()),
                    ..Default::default()
                },
            ),
            (
                "z_image_turbo",
                AppearanceSelection {
                    looks: vec!["look-light-leaks".into()],
                    ..Default::default()
                },
            ),
        ] {
            assert!(resolve_appearance(model, &selected).is_err());
        }
        assert!(serde_json::from_str::<AppearanceSelection>(r#"{"films":["unknown"]}"#).is_err());
    }
}
