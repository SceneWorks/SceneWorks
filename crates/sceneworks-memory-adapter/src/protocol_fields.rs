//! Borrowed access to the already-parsed plan. Values are never trimmed, normalized or inferred.
use super::*;

fn label(path: &str) -> String {
    format!("planned{}", path.replace('/', "."))
}

pub fn plan_optional_str<'a>(plan: &'a Value, path: &str) -> Option<&'a str> {
    plan.pointer(path).and_then(Value::as_str)
}
pub fn planned_optional_str<'a>(request: &'a Value, path: &str) -> Result<Option<&'a str>, String> {
    Ok(plan_optional_str(crate::planned(request)?, path))
}
pub fn plan_str<'a>(plan: &'a Value, path: &str) -> Result<&'a str, String> {
    plan_optional_str(plan, path).ok_or_else(|| format!("{} must be a string", label(path)))
}
pub fn planned_str<'a>(request: &'a Value, path: &str) -> Result<&'a str, String> {
    plan_str(crate::planned(request)?, path)
}
pub fn plan_optional_u64(plan: &Value, path: &str) -> Option<u64> {
    plan.pointer(path).and_then(Value::as_u64)
}
pub fn plan_optional_bool(plan: &Value, path: &str) -> Option<bool> {
    plan.pointer(path).and_then(Value::as_bool)
}

/// A declared geometry axis retains the capture protocol's integer range and error spelling.
pub fn geometry_axis_u32(geometry: &Map<String, Value>, name: &str) -> Result<u32, String> {
    geometry
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
}

pub fn plan_object<'a>(plan: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    plan.pointer(path)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{} must be an object", label(path)))
}
pub fn planned_object<'a>(
    request: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, String> {
    plan_object(crate::planned(request)?, path)
}

pub fn planned_provider(request: &Value) -> Result<&str, String> {
    planned_str(request, "/target/provider")
}
pub fn planned_model_id(request: &Value) -> Result<&str, String> {
    planned_str(request, "/target/modelId")
}
pub fn planned_mode(request: &Value) -> Result<&str, String> {
    planned_str(request, "/target/mode")
}
pub fn planned_tier(request: &Value) -> Result<&str, String> {
    planned_str(request, "/target/tier")
}
pub fn planned_load_shape(request: &Value) -> Result<&str, String> {
    planned_str(request, "/loadShape")
}
pub fn planned_fixture(request: &Value) -> Result<&str, String> {
    planned_str(request, "/fixture")
}
pub fn planned_calibration_fingerprint(request: &Value) -> Result<&str, String> {
    planned_str(request, "/calibrationFingerprint")
}

/// Read a field after the caller's target-object check, preserving its earlier failure boundary.
pub fn target_optional_str<'a>(target: &'a Map<String, Value>, field: &str) -> Option<&'a str> {
    target.get(field).and_then(Value::as_str)
}
pub fn target_str<'a>(target: &'a Map<String, Value>, field: &str) -> Result<&'a str, String> {
    target_optional_str(target, field)
        .ok_or_else(|| format!("planned.target.{field} must be a string"))
}
pub fn target_optional_u64(target: &Map<String, Value>, field: &str) -> Option<u64> {
    target.get(field).and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_fields_keep_borrowing_types_spelling_and_missing_plan_precedence() {
        let value = json!({"planned":{"target":{"provider":" custom ","modelId":"imported/id","mode":"edit_image","tier":"q8"},"loadShape":"deferred_materialization","fixture":"fixture","calibrationFingerprint":"identity","count":u64::MAX,"float":2.0,"stringCount":"2"}});
        assert_eq!(planned_provider(&value).unwrap(), " custom ");
        assert_eq!(planned_model_id(&value).unwrap(), "imported/id");
        assert_eq!(planned_mode(&value).unwrap(), "edit_image");
        assert_eq!(planned_tier(&value).unwrap(), "q8");
        assert_eq!(
            planned_load_shape(&value).unwrap(),
            "deferred_materialization"
        );
        assert_eq!(planned_fixture(&value).unwrap(), "fixture");
        assert_eq!(planned_calibration_fingerprint(&value).unwrap(), "identity");
        assert_eq!(
            plan_optional_u64(&value["planned"], "/count"),
            Some(u64::MAX)
        );
        assert_eq!(plan_optional_u64(&value["planned"], "/float"), None);
        assert_eq!(plan_optional_u64(&value["planned"], "/stringCount"), None);
        assert_eq!(
            planned_str(&json!({}), "/target/mode").unwrap_err(),
            "run request is missing planned"
        );
        assert_eq!(
            planned_str(&json!({"planned":null}), "/target/mode").unwrap_err(),
            "planned.target.mode must be a string"
        );
        assert_eq!(planned_object(&value, "/target").unwrap().len(), 4);
        assert_eq!(planned_optional_str(&value, "/absent").unwrap(), None);
        assert_eq!(
            plan_optional_bool(&json!({"flag":false}), "/flag"),
            Some(false)
        );
        assert_eq!(plan_optional_bool(&json!({"flag":"false"}), "/flag"), None);
        let axes = json!({"width":u32::MAX,"height":u64::from(u32::MAX)+1,"frames":1.5});
        let axes = axes.as_object().unwrap();
        assert_eq!(geometry_axis_u32(axes, "width").unwrap(), u32::MAX);
        for field in ["height", "frames", "missing"] {
            assert_eq!(
                geometry_axis_u32(axes, field).unwrap_err(),
                format!("planned.target.geometry.{field} must fit u32")
            );
        }
    }
}
