//! Weights-free current native inputs for the product UI comparison. Host installation is supplied
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
        let minimax_h3_turbo = sceneworks_core::video_request::is_minimax_h3_model(id);
        if moss_edit {
            added_capabilities.insert(id.to_owned(), json!(["edit_image"]));
        }
        let object = model.as_object_mut().ok_or("model must be an object")?;
        object.insert("supportsLastOnlyImageToVideo".into(), json!(last_only));
        object.insert("supportsMinimaxH3Turbo".into(), json!(minimax_h3_turbo));
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
        sceneworks_core::memory_guidance::apply_to_model_entry(object);
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
            // serde_json's default float parser is not correctly rounded, so a non-integral
            // fixture value (the served memory-guidance thresholds) can read back one ULP away.
            // Only two non-integral float leaves may differ, and by at most one unit in the last
            // place; every integer, string, bool, null and shape compares exactly.
            (Value::Number(a), Value::Number(b)) if !a.is_f64() && !b.is_f64() => {
                assert_eq!(a, b, "{path}");
            }
            (Value::Number(a), Value::Number(b)) => {
                let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
                let float_leaf = |value: f64| value.is_finite() && value.fract() != 0.0;
                let one_ulp = float_leaf(x)
                    && float_leaf(y)
                    && x.is_sign_positive() == y.is_sign_positive()
                    && x.to_bits().abs_diff(y.to_bits()) <= 1;
                assert!(x == y || one_ulp, "{path}: {x} != {y}");
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

    /// The fixture comparison's only tolerance: a non-integral float leaf one ULP away passes; two
    /// ULPs, an integer off by one and any non-numeric difference fail.
    #[test]
    fn the_fixture_comparison_tolerates_exactly_one_float_ulp() {
        use serde_json::json;
        let x = 42.22222222222222_f64;
        let next = |value: f64, steps: u64| f64::from_bits(value.to_bits() + steps);
        compare(&json!(next(x, 1)), &json!(x), "one-ulp");
        compare(&json!({"a": [1.0]}), &json!({"a": [1]}), "integral");
        let fails = |actual: serde_json::Value, expected: serde_json::Value| {
            std::panic::catch_unwind(|| compare(&actual, &expected, "case")).is_err()
        };
        assert!(fails(json!(next(x, 2)), json!(x)), "two ULPs fail");
        assert!(fails(json!(17), json!(16)), "an integer off by one fails");
        assert!(
            fails(
                json!(9_007_199_254_740_993_u64),
                json!(9_007_199_254_740_992_u64)
            ),
            "integers never take the float tolerance"
        );
        assert!(fails(json!("a"), json!("b")), "strings compare exactly");
        assert!(fails(json!(true), json!(false)), "booleans compare exactly");
    }

    #[test]
    fn native_product_inputs_match_the_current_candidate_fixture() {
        let candidate: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../apps/web/src/testUtils/catalogBaseline/candidate-catalog.json"
        ))
        .expect("current native catalog fixture");
        compare(
            &super::catalog().expect("metadata-only catalog"),
            &candidate,
            "catalog",
        );
    }
}
