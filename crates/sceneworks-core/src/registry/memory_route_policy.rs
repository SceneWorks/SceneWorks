//! Typed load-route policy for declaration-owned deferred materialization.
//! The worker keeps historical legacy shapers separately; these rows are the single authored
//! source for its declaration-owned provider/tier/mode/load-profile selectors.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryRoutePolicyDocument {
    pub schema_version: u32,
    pub declared_routes: Vec<DeclaredMemoryRoute>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DeclaredMemoryRoute {
    pub backend: String,
    pub provider: String,
    pub tiers: Vec<String>,
    pub modes: Vec<String>,
    pub load_profiles: Vec<String>,
    pub requires_sequential_selection: bool,
    /// Required only when the runtime provider is a composed identity absent from the native
    /// generator descriptor and manifest memory contract (currently Candle Kolors bespoke routes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

pub fn parse(raw: &str) -> Result<MemoryRoutePolicyDocument, String> {
    crate::jsonc::reject_duplicate_keys(raw).map_err(|error| error.to_string())?;
    let document: MemoryRoutePolicyDocument =
        serde_json::from_str(raw).map_err(|error| error.to_string())?;
    if document.schema_version != 1 || document.declared_routes.is_empty() {
        return Err("memory route policy requires schemaVersion 1 and nonempty routes".into());
    }
    let mut coordinates = BTreeSet::new();
    for rule in &document.declared_routes {
        if !matches!(rule.backend.as_str(), "mlx" | "candle") || rule.provider.is_empty() {
            return Err(format!(
                "invalid memory route backend/provider: {}",
                rule.provider
            ));
        }
        for (name, values, allowed) in [
            ("tiers", &rule.tiers, TIERS),
            ("modes", &rule.modes, MODES),
            ("loadProfiles", &rule.load_profiles, LOAD_PROFILES),
        ] {
            if values.is_empty()
                || values
                    .iter()
                    .any(|value| !allowed.contains(&value.as_str()))
            {
                return Err(format!("{} has invalid or empty {name}", rule.provider));
            }
            if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
                return Err(format!("{} has duplicate {name}", rule.provider));
            }
        }
        for tier in &rule.tiers {
            for mode in &rule.modes {
                for profile in &rule.load_profiles {
                    if !coordinates.insert((
                        rule.backend.as_str(),
                        rule.provider.as_str(),
                        tier.as_str(),
                        mode.as_str(),
                        profile.as_str(),
                    )) {
                        return Err(format!(
                            "duplicate memory route coordinate for {}",
                            rule.provider
                        ));
                    }
                }
            }
        }
    }
    Ok(document)
}

const TIERS: &[&str] = &["bf16", "q4", "q8", "nvfp4"];
const MODES: &[&str] = &[
    "text_to_image",
    "style_variations",
    "edit_image",
    "image_to_image",
    "image_inpaint",
    "image_detail",
    "character_image",
];
const LOAD_PROFILES: &[&str] = &[
    "plain",
    "lora",
    "lora_pid",
    "lora_single_control",
    "lora_single_control_pid",
    "lora_ip_adapter",
    "ip_adapter_pid",
    "lora_ip_adapter_pid",
    "single_control",
    "single_control_pid",
    "multi_control",
    "ip_adapter",
    "pid",
    "identity",
];
