//! Usage policy declared by a catalog entry (sc-22998, epic sc-22988 E2 / E7): whether a model may
//! run on a commercial-use route, and how a purpose-specific component closure is acquired.
//!
//! Both answers are read from the manifest entry itself — `nonCommercial`, `commercialUse` and
//! `conditionalComponents` — never from a model id or name, so the policy for a model travels with
//! its declaration and a sibling model can never inherit it by resemblance.
//!
//! # No substitution
//!
//! A refusal NAMES alternatives; it never picks one. [`commercial_use_verdict`] resolves the model a
//! route asked for by its exact id (an unknown id is [`CommercialUseError::UnknownModel`], not the
//! nearest match), and a refused model's `alternatives` are the ids the user may choose instead. The
//! route that asked must refuse the request; nothing here returns a model to run in its place.

use serde::Serialize;
use serde_json::Value;

/// A commercial-use route's answer for one catalog entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "verdict")]
pub enum CommercialUseVerdict {
    /// The entry declares nothing that restricts commercial use of its weights.
    Eligible { model_id: String },
    /// The entry's weights may not be used on a commercial-use route.
    Refused {
        model_id: String,
        /// The declared restriction (`commercialUse.reason`), or the generic non-commercial
        /// restriction for a `nonCommercial: true` entry that declares no `commercialUse` block.
        reason: String,
        /// The family the entry points commercial use to (`commercialUse.alternativeFamily`).
        #[serde(skip_serializing_if = "Option::is_none")]
        alternative_family: Option<String>,
        /// The catalog ids of that family that are themselves commercially eligible, in catalog
        /// order. Empty when the family has no eligible entry in this catalog.
        alternatives: Vec<String>,
        /// Shown beside the pointer (`commercialUse.alternativeNote`).
        #[serde(skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
}

/// Why no verdict could be given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommercialUseError {
    /// No catalog entry has this exact id. Never answered with a different model.
    UnknownModel(String),
    /// Several catalog entries claim this id; which one the route meant is undecidable.
    AmbiguousModel(String),
}

impl std::fmt::Display for CommercialUseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownModel(id) => write!(f, "no catalog model has the id '{id}'"),
            Self::AmbiguousModel(id) => {
                write!(f, "more than one catalog model has the id '{id}'")
            }
        }
    }
}

impl std::error::Error for CommercialUseError {}

const GENERIC_NON_COMMERCIAL_REASON: &str =
    "This model's licence restricts its weights to non-commercial use.";

fn model_id(model: &Value) -> Option<&str> {
    model.get("id").and_then(Value::as_str)
}

fn non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

/// The refusal reason for one entry, or `None` when its weights are commercially eligible.
///
/// `commercialUse.eligible: false` refuses; so does `nonCommercial: true` whatever the block says —
/// an explicit `eligible: true` cannot un-restrict a model the catalog flags non-commercial.
fn refusal_reason(model: &Value) -> Option<String> {
    let block = model.get("commercialUse");
    let declared_ineligible = block
        .and_then(|block| block.get("eligible"))
        .and_then(Value::as_bool)
        == Some(false);
    let non_commercial = model.get("nonCommercial").and_then(Value::as_bool) == Some(true);
    if !declared_ineligible && !non_commercial {
        return None;
    }
    Some(
        non_empty_string(block.and_then(|block| block.get("reason")))
            .unwrap_or_else(|| GENERIC_NON_COMMERCIAL_REASON.to_owned()),
    )
}

/// The verdict a commercial-use route must apply to the catalog entry `id`.
///
/// `catalog` is the full model list the route selects from; alternatives are resolved against it,
/// so a pointer only ever names a model that exists and is itself eligible.
pub fn commercial_use_verdict(
    catalog: &[Value],
    id: &str,
) -> Result<CommercialUseVerdict, CommercialUseError> {
    let matches = catalog
        .iter()
        .filter(|model| model_id(model) == Some(id))
        .collect::<Vec<_>>();
    let model = match matches.as_slice() {
        [model] => *model,
        [] => return Err(CommercialUseError::UnknownModel(id.to_owned())),
        _ => return Err(CommercialUseError::AmbiguousModel(id.to_owned())),
    };
    let Some(reason) = refusal_reason(model) else {
        return Ok(CommercialUseVerdict::Eligible {
            model_id: id.to_owned(),
        });
    };
    let block = model.get("commercialUse");
    let alternative_family =
        non_empty_string(block.and_then(|block| block.get("alternativeFamily")));
    let own_family = model.get("family").and_then(Value::as_str);
    let alternatives = match alternative_family.as_deref() {
        // A family never points at itself: that would offer the refused weights back.
        Some(family) if Some(family) != own_family => catalog
            .iter()
            .filter(|candidate| candidate.get("family").and_then(Value::as_str) == Some(family))
            .filter(|candidate| refusal_reason(candidate).is_none())
            .filter_map(model_id)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    };
    Ok(CommercialUseVerdict::Refused {
        model_id: id.to_owned(),
        reason,
        alternative_family,
        alternatives,
        note: non_empty_string(block.and_then(|block| block.get("alternativeNote"))),
    })
}

/// Why a purpose's conditional components cannot be acquired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConditionalComponentsError {
    /// The model declares no conditional component for this purpose.
    NotDeclared { purpose: String },
    /// At least one component the purpose needs is blocked; nothing is returned for any of them.
    Blocked {
        purpose: String,
        /// `(componentId, reason, unblock)` for every blocked component, in declaration order.
        blocked: Vec<(String, String, String)>,
    },
}

impl std::fmt::Display for ConditionalComponentsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotDeclared { purpose } => {
                write!(f, "this model declares no components for '{purpose}'")
            }
            Self::Blocked { purpose, blocked } => {
                write!(f, "'{purpose}' is unavailable: ")?;
                let parts = blocked
                    .iter()
                    .map(|(component, reason, unblock)| {
                        format!("{component}: {reason} (unblock: {unblock})")
                    })
                    .collect::<Vec<_>>();
                f.write_str(&parts.join("; "))
            }
        }
    }
}

impl std::error::Error for ConditionalComponentsError {}

/// The pinned download rows `purpose` (for example `cover`) needs, from the entry's
/// `conditionalComponents`.
///
/// This is the only acquisition seam for those components — they are never `downloads[]` rows, so
/// no install or tier selection fetches them. All-or-nothing: if ANY component the purpose needs
/// carries `blocked`, the whole purpose is refused with every blocked component's reason and unblock
/// condition, because a partial closure cannot serve the purpose either. The returned rows carry
/// `provider` / `repo` / `revision` / `files` / `componentId` in the shape a download job reads, and
/// are marked `coRequisite: true` so they can never be mistaken for the model's primary download.
pub fn conditional_component_downloads(
    model: &Value,
    purpose: &str,
) -> Result<Vec<Value>, ConditionalComponentsError> {
    let components = model
        .get("conditionalComponents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|component| {
            component
                .get("requiredFor")
                .and_then(Value::as_array)
                .is_some_and(|purposes| purposes.iter().any(|p| p.as_str() == Some(purpose)))
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        return Err(ConditionalComponentsError::NotDeclared {
            purpose: purpose.to_owned(),
        });
    }
    let blocked = components
        .iter()
        .filter_map(|component| {
            let block = component.get("blocked")?;
            Some((
                non_empty_string(component.get("componentId")).unwrap_or_default(),
                non_empty_string(block.get("reason")).unwrap_or_default(),
                non_empty_string(block.get("unblock")).unwrap_or_default(),
            ))
        })
        .collect::<Vec<_>>();
    if !blocked.is_empty() {
        return Err(ConditionalComponentsError::Blocked {
            purpose: purpose.to_owned(),
            blocked,
        });
    }
    Ok(components
        .into_iter()
        .map(|component| {
            let mut row = serde_json::Map::new();
            for key in [
                "provider",
                "repo",
                "revision",
                "files",
                "componentId",
                "estimatedSizeBytes",
            ] {
                if let Some(value) = component.get(key) {
                    row.insert(key.to_owned(), value.clone());
                }
            }
            row.insert("coRequisite".to_owned(), Value::Bool(true));
            Value::Object(row)
        })
        .collect())
}

#[cfg(test)]
#[path = "model_usage_policy_tests.rs"]
mod tests;
