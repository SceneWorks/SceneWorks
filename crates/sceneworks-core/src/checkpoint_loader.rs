//! Bind imported weights to the same shipped model policy and component closure as a base model.
//!
//! The provider adapter owns whether a loader accepts replacement components. The product registry
//! supplies defaults and companion downloads; imported identity, primary paths and provenance never
//! come from the parent.
use serde_json::{Map, Value};

pub fn adapter(family: &str) -> Option<&'static gen_core::CheckpointAdapterRegistration> {
    let family = crate::model_family::exact_identity(family)?;
    gen_core::CHECKPOINT_ADAPTERS.iter().find(|adapter| {
        crate::model_family::exact_identity(adapter.compatibility_projection.family) == Some(family)
    })
}

pub fn bound_base(entry: &Map<String, Value>) -> Result<Option<Value>, String> {
    let Some(family) = entry.get("family").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(reuse) = adapter(family).and_then(|adapter| adapter.loader_reuse) else {
        return Ok(None);
    };
    let id = entry
        .get("baseModel")
        .and_then(Value::as_str)
        .unwrap_or(reuse.provider_id);
    let base = crate::registry::builtin_model_spec(id)
        .ok_or_else(|| format!("The compatible base model {id:?} is not available"))?;
    if crate::model_family::exact_identity(&base.product.family)
        != crate::model_family::exact_identity(family)
    {
        return Err(format!(
            "The base model {id:?} is incompatible with checkpoint family {family:?}"
        ));
    }
    // A binding to a different provider can select a different numerical/configuration policy even
    // within one architecture. Require the provider declared by the adapter rather than guessing.
    if id != reuse.provider_id {
        return Err(format!(
            "This checkpoint loader requires compatible base {:?}, not {id:?}",
            reuse.provider_id
        ));
    }
    if entry
        .get("policyModel")
        .is_some_and(|policy| policy.as_str() != Some(id))
    {
        return Err(format!(
            "Checkpoint policyModel must match the registered compatible base {id:?}"
        ));
    }
    Ok(Some(
        serde_json::to_value(&base.product).expect("serializable base product"),
    ))
}

pub fn apply_bound_base_defaults(entry: &mut Map<String, Value>) -> Result<(), String> {
    let Some(base) = bound_base(entry)? else {
        return Ok(());
    };
    let base = base.as_object().expect("base product object");
    entry.insert("baseModel".into(), base["id"].clone());
    entry.insert("policyModel".into(), base["id"].clone());
    for key in [
        "defaults",
        "limits",
        "image",
        "ui",
        "loraCompatibility",
        "capabilities",
        "adapter",
    ] {
        if let Some(value) = base.get(key) {
            match (entry.get_mut(key), value) {
                (Some(Value::Object(existing)), Value::Object(defaults)) => {
                    for (field, value) in defaults {
                        existing
                            .entry(field.clone())
                            .or_insert_with(|| value.clone());
                    }
                }
                (None, _) => {
                    entry.insert(key.into(), value.clone());
                }
                _ => {}
            }
        }
    }
    // Companions are immutable product declarations, not author supplied host paths. Never copy
    // the base's primary download or paths: the imported checkpoint owns the transformer bytes.
    entry.insert(
        "downloads".into(),
        Value::Array(component_downloads(&Value::Object(base.clone()))),
    );
    Ok(())
}

/// Root files retained when a checkpoint substitutes the base's learned component.
pub fn is_base_descriptor(name: &str) -> bool {
    !name.ends_with(".index.json")
        && ![".safetensors", ".gguf", ".bin", ".pt", ".pth"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

/// The existing base component closure plus the architecture descriptors needed by its loader.
/// Used both by import acquisition and by persisted repair/portable declarations.
pub fn component_downloads(base: &Value) -> Vec<Value> {
    let mut rows: Vec<_> = base
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| row.get("coRequisite").and_then(Value::as_bool) == Some(true))
        .cloned()
        .collect();
    if let Some(mut primary) = crate::model_artifacts::artifact_selection::model_download(base) {
        let files: Vec<_> = primary
            .get("files")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|file| file.as_str().is_some_and(is_base_descriptor))
            .cloned()
            .collect();
        if !files.is_empty() {
            primary["files"] = Value::Array(files);
            primary["coRequisite"] = Value::Bool(true);
            primary["componentId"] = Value::String("import_base_config".into());
            primary
                .as_object_mut()
                .expect("primary download object")
                .remove("estimatedSizeBytes");
            rows.push(primary);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn imported_identity_and_primary_survive_base_policy_inheritance() {
        let mut entry = json!({"id":"serenity","family":"moss-image","paths":{"model":"owned/model"},"source":{"provider":"local"},"defaults":{"steps":20}}).as_object().unwrap().clone();
        apply_bound_base_defaults(&mut entry).unwrap();
        assert_eq!(entry["id"], "serenity");
        assert_eq!(entry["paths"]["model"], "owned/model");
        assert_eq!(entry["source"]["provider"], "local");
        assert_eq!(entry["baseModel"], "moss_image_base");
        assert_eq!(entry["defaults"]["steps"], 20);
        assert_eq!(entry["defaults"]["schedulerShift"], 5);
        assert!(entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["coRequisite"] == true));
    }
    #[test]
    fn incompatible_base_cannot_supply_components_or_policy() {
        let entry = json!({"family":"moss-image","baseModel":"krea_2_raw"})
            .as_object()
            .unwrap()
            .clone();
        assert!(bound_base(&entry).unwrap_err().contains("incompatible"));
        let entry =
            json!({"family":"moss-image","baseModel":"moss_image_base","policyModel":"krea_2_raw"})
                .as_object()
                .unwrap()
                .clone();
        assert!(bound_base(&entry).unwrap_err().contains("policyModel"));
    }
}
