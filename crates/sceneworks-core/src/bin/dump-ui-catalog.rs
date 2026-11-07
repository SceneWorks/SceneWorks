//! Weights-free inputs for the pre-registry product UI comparison. Host installation is supplied
//! by the jsdom fixture, not inferred from this machine's model cache.
use sceneworks_core::{builtin_manifests::builtin_manifest, jobs_store, training};
use serde_json::{json, Value};

fn catalog() -> Result<Value, Box<dyn std::error::Error>> {
    let mut models = builtin_manifest("builtin.models.jsonc")?["models"]
        .as_array()
        .ok_or("builtin models must be an array")?
        .clone();
    let mut added_capabilities = serde_json::Map::new();
    for model in &mut models {
        let id = model["id"].as_str().ok_or("model id missing")?;
        let kind = model["type"].as_str().ok_or("model type missing")?;
        let family = model["family"].as_str();
        let mac = serde_json::to_value(jobs_store::model_mac_support(id, kind, family))?;
        let candle = serde_json::to_value(jobs_store::model_candle_support(id, kind))?;
        // The API's project_moss_edit_surface and apply_last_only_video_capability are host-free
        // projections for this pinned builtin population. Use their owning pure predicates.
        let moss_edit = sceneworks_core::moss_image::edit_backend_eligible(id, "mlx");
        if moss_edit != sceneworks_core::moss_image::edit_backend_eligible(id, "candle") {
            return Err("MOSS backend surfaces diverged; capture per-platform catalogs".into());
        }
        let last_only = sceneworks_core::h3_keyframes::accepts_last_only_resolved_image_to_video(
            id,
            "image_to_video",
            true,
            None,
        );
        if moss_edit {
            added_capabilities.insert(id.to_owned(), json!(["edit_image"]));
        }
        let object = model.as_object_mut().ok_or("model must be an object")?;
        object.insert("supportsLastOnlyImageToVideo".into(), json!(last_only));
        if moss_edit {
            let capabilities = object
                .get_mut("capabilities")
                .and_then(Value::as_array_mut)
                .ok_or("MOSS capabilities must be an array")?;
            capabilities.retain(|mode| mode != "edit_image" && mode != "image_edit");
            capabilities.push(json!("edit_image"));
        }
        object.insert("macSupport".into(), mac);
        object.insert("candleSupport".into(), candle);
        sceneworks_core::preview_support::apply_to_model_entry(object);
        sceneworks_core::decoder_support::apply_to_model_entry(object);
    }
    models.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let output: Value = json!({
        "models": models,
        "apiAddedCapabilities": added_capabilities,
        "platforms": {
            "macos": jobs_store::mac_capabilities("macos", true),
            "windows": jobs_store::mac_capabilities("windows", false),
            "linux": jobs_store::mac_capabilities("linux", false),
        },
        "trainingTargets": training::builtin_training_targets(),
        "trainingPresets": training::builtin_training_presets(),
    });
    Ok(output)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&catalog()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    fn compare(actual: &serde_json::Value, expected: &serde_json::Value, path: &str) {
        use serde_json::Value;
        match (actual, expected) {
            // The browser serializes 1.0 as 1. Compare JSON numeric meaning across that boundary.
            (Value::Number(a), Value::Number(b)) => {
                assert_eq!(a.as_f64(), b.as_f64(), "{path}");
            }
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(a.len(), b.len(), "{path}: object fields");
                for (key, value) in a {
                    compare(
                        value,
                        b.get(key).expect("baseline field"),
                        &format!("{path}/{key}"),
                    );
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{path}: array length");
                for (index, (left, right)) in a.iter().zip(b).enumerate() {
                    compare(left, right, &format!("{path}/{index}"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    #[test]
    fn native_product_inputs_match_the_pinned_before_side() {
        let old: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../apps/web/src/testUtils/catalogBaseline/old-catalog.json"
        ))
        .expect("pinned catalog fixture");
        compare(
            &super::catalog().expect("metadata-only catalog"),
            &old["catalog"],
            "catalog",
        );
    }
}
