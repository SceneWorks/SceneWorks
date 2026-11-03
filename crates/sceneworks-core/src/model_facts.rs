//! Shared model facts from resolved manifests, with shipped rows as the ID-only fallback.
//! Explicit supplied facts are authoritative. Older catalog/replay rows can omit newer metadata:
//! a matching builtin identity (or a compatible recorded base) supplies missing adapter facts.
//! Unknown custom IDs retain only their established legacy compatibility conventions.

use crate::contracts::JsonObject;
use serde_json::Value;

pub fn entry<'a>(id: &str, supplied: Option<&'a JsonObject>) -> Option<&'a JsonObject> {
    supplied.or_else(|| crate::builtin_manifests::builtin_model_entry(id))
}

pub fn family(entry: &JsonObject) -> Option<String> {
    entry
        .get("family")
        .and_then(Value::as_str)
        .map(crate::model_family::canonical)
}

pub fn adapter(entry: &JsonObject) -> Option<&str> {
    entry.get("adapter").and_then(Value::as_str)
}

pub fn lora_fact<'a>(id: &str, supplied: Option<&'a JsonObject>, key: &str) -> Option<&'a Value> {
    if let Some(value) = supplied
        .and_then(|row| row.get("loraCompatibility"))
        .and_then(|compatibility| compatibility.get(key))
    {
        return Some(value);
    }
    let builtin = crate::builtin_manifests::builtin_model_entry(id).or_else(|| {
        supplied?
            .get("baseModel")?
            .as_str()
            .and_then(crate::builtin_manifests::builtin_model_entry)
    })?;
    if let Some(supplied) = supplied {
        // An explicit incompatible family never borrows a known ID's adapter metadata. A legacy
        // minimal row without a family can still identify the shipped model by exact ID.
        if let Some(supplied_family) = family(supplied) {
            if family(builtin).as_deref() != Some(&supplied_family) {
                return None;
            }
        }
    }
    builtin.get("loraCompatibility")?.get(key)
}

pub fn lora_generation<'a>(id: &str, supplied: Option<&'a JsonObject>) -> Option<&'a str> {
    lora_fact(id, supplied, "generation")?.as_str()
}

pub fn lora_size_class<'a>(id: &str, supplied: Option<&'a JsonObject>) -> Option<&'a str> {
    lora_fact(id, supplied, "sizeClass")?.as_str()
}

pub fn lora_image_conditioned_backbone(id: &str, supplied: Option<&JsonObject>) -> Option<bool> {
    lora_fact(id, supplied, "imageConditionedBackbone")?.as_bool()
}

/// Shared ID-only family predicate. Known builtin rows use their actual family; unknown IDs
/// retain the historical prefix convention for custom/legacy request-shaping callers only.
/// This is not an engine/provider eligibility predicate and must not authorize an unknown route.
pub fn is_family_with_legacy_prefix(id: &str, expected: &str, legacy_prefix: &str) -> bool {
    match entry(id, None) {
        Some(entry) => {
            family(entry).is_some_and(|family| family == crate::model_family::canonical(expected))
        }
        None => id.starts_with(legacy_prefix),
    }
}

/// Import lineage defaults for the families previously supported by the import surface. The
/// builtin adapter field owns concrete identities; SD3/Anima retain their historical imported
/// lineage labels, which are distinct from the builtin MLX dispatch adapter names.
pub fn import_adapter_for_family(family: &str) -> Option<&'static str> {
    let family = crate::model_family::canonical(family);
    if !matches!(
        family.as_str(),
        "z-image"
            | "qwen-image"
            | "qwen-image-2-1"
            | "lens"
            | "sensenova-u1"
            | "flux"
            | "flux3"
            | "chroma"
            | "kolors"
            | "sdxl"
            | "mage-flow"
            | "sd3"
            | "ltx-video"
            | "wan-video"
            | "svd"
            | "bernini"
            | "scail2"
            | "anima"
            | "krea_2"
    ) {
        return None;
    }
    match family.as_str() {
        "sd3" => return Some("sd3"),
        "anima" => return Some("anima"),
        _ => {}
    }
    crate::builtin_manifests::builtin_model_entries()
        .iter()
        .filter_map(Value::as_object)
        .find(|entry| self::family(entry).as_deref() == Some(&family))
        .and_then(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn supplied_facts_override_builtin_and_legacy_rows_fill_only_missing_adapter_metadata() {
        let custom = json!({"id":"ltx_2_5", "family":"custom", "adapter":"custom_adapter",
            "loraCompatibility":{"generation":"custom", "sizeClass":"3b", "imageConditionedBackbone":false}});
        let custom = custom.as_object().unwrap();
        assert_eq!(family(custom).as_deref(), Some("custom"));
        assert_eq!(adapter(custom), Some("custom_adapter"));
        assert_eq!(lora_generation("ltx_2_5", Some(custom)), Some("custom"));
        assert_eq!(lora_size_class("ltx_2_5", Some(custom)), Some("3b"));
        assert_eq!(
            lora_image_conditioned_backbone("ltx_2_5", Some(custom)),
            Some(false)
        );
        assert_eq!(
            lora_generation("ltx_2_5", Some(&JsonObject::new())),
            Some("2.5")
        );
        let old = json!({"id":"ltx_2_5", "family":"ltx-video"});
        assert_eq!(lora_generation("ltx_2_5", old.as_object()), Some("2.5"));
        let renamed = json!({"id":"renamed", "family":"ltx-video", "baseModel":"ltx_2_3"});
        assert_eq!(lora_generation("renamed", renamed.as_object()), Some("2.3"));
        let incompatible = json!({"family":"custom"});
        assert_eq!(lora_generation("ltx_2_5", incompatible.as_object()), None);
        assert_eq!(lora_generation("ltx_2_5", None), Some("2.5"));
    }

    #[test]
    fn every_builtin_gate_entry_declares_its_generation_size_and_conditioning() {
        for row in crate::builtin_manifests::builtin_model_entries() {
            let row = row.as_object().unwrap();
            let id = row["id"].as_str().unwrap();
            match family(row).as_deref() {
                Some("ltx-video") => assert!(lora_generation(id, None).is_some(), "{id}"),
                Some("wan-video" | "krea-realtime" | "scail2") => {
                    assert!(lora_size_class(id, None).is_some(), "{id}");
                    assert!(lora_image_conditioned_backbone(id, None).is_some(), "{id}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn import_adapter_defaults_preserve_the_existing_lineage_surface() {
        for (family, expected) in [
            ("z-image", "z_image_diffusers"),
            ("qwen-image", "qwen_image"),
            ("qwen-image-2-1", "qwen_image_2_1"),
            ("lens", "lens_turbo"),
            ("sensenova-u1", "sensenova_u1"),
            ("flux", "flux_diffusers"),
            ("flux3", "mlx_flux3"),
            ("chroma", "chroma_diffusers"),
            ("kolors", "kolors_diffusers"),
            ("sdxl", "sdxl_diffusers"),
            ("mage-flow", "mlx_mage"),
            ("sd3", "sd3"),
            ("ltx-video", "ltx_video"),
            ("wan-video", "wan_video"),
            ("svd", "svd_video"),
            ("bernini", "bernini"),
            ("scail2", "scail2"),
            ("anima", "anima"),
            ("krea2", "mlx_krea"),
        ] {
            assert_eq!(
                import_adapter_for_family(family),
                Some(expected),
                "{family}"
            );
        }
        for family in ["unknown", "moss-image", "minimax-h3", "flux2-dev"] {
            assert_eq!(import_adapter_for_family(family), None, "{family}");
        }
    }
}
