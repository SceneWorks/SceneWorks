//! Portable admission routes follow the same request predicates as worker claim.

use sceneworks_core::contracts::{JobType, JsonObject};
use sceneworks_core::jobs_store::{native_route_claims_request, NON_GPU_JOB_TYPES};
use serde_json::Value;

use crate::ApiError;

/// Backend and platform pairs whose full source closure must be frozen at admission. The
/// worker claim still applies the requested GPU id and capability filters; this set never
/// invents a route merely because the catalog advertises some other request shape.
pub(crate) fn eligible_portable_routes(
    job_type: &JobType,
    payload: &JsonObject,
    entries: &[(String, &Value)],
    requested_gpu: &str,
) -> Result<Vec<(String, String)>, ApiError> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    if NON_GPU_JOB_TYPES.contains(&job_type.as_str()) {
        return Ok(["macos", "linux", "windows"]
            .into_iter()
            .map(|platform| ("cpu".to_owned(), platform.to_owned()))
            .collect());
    }
    if requested_gpu.eq_ignore_ascii_case("cpu") {
        return Err(ApiError::bad_request(
            "GPU model work cannot be assigned to the CPU utility worker",
        ));
    }
    let mut routes = Vec::new();
    for (backend, platforms) in [
        ("mlx", &["macos"][..]),
        ("candle", &["linux", "windows"][..]),
    ] {
        let claims =
            native_route_claims_request(backend, job_type, payload).map_err(ApiError::internal)?;
        if !claims || !entries_support_backend(entries, backend) {
            continue;
        }
        routes.extend(
            platforms
                .iter()
                .map(|platform| (backend.to_owned(), (*platform).to_owned())),
        );
    }
    if routes.is_empty() {
        return Err(ApiError::bad_request(
            "No native route can claim this model request and complete source closure",
        ));
    }
    Ok(routes)
}

fn entries_support_backend(entries: &[(String, &Value)], backend: &str) -> bool {
    entries.iter().all(|(_, entry)| {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            return false;
        };
        if let Some(spec) = sceneworks_core::registry::builtin_model_spec(id) {
            return spec.routes.contains_key(backend);
        }
        let key = if backend == "mlx" {
            "macSupport"
        } else {
            "candleSupport"
        };
        entry
            .pointer(&format!("/{key}/supported"))
            .and_then(Value::as_bool)
            != Some(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn routes(job_type: JobType, payload: Value, entry: Value) -> Vec<(String, String)> {
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        eligible_portable_routes(
            &job_type,
            payload.as_object().expect("payload object"),
            &entry_ref,
            "auto",
        )
        .expect("routable job")
    }

    #[test]
    fn auto_image_routes_follow_both_native_claim_predicates() {
        let entry = json!({"id":"flux_schnell"});
        let actual = routes(
            JobType::ImageGenerate,
            json!({"model":"flux_schnell"}),
            entry,
        );
        assert_eq!(
            actual,
            vec![
                ("mlx".to_owned(), "macos".to_owned()),
                ("candle".to_owned(), "linux".to_owned()),
                ("candle".to_owned(), "windows".to_owned()),
            ]
        );
    }

    #[test]
    fn no_route_is_invented_for_unsupported_conditioned_request() {
        let entry = json!({"id":"chroma","type":"image"});
        let payload = json!({"model":"chroma","advanced":{"poses":[{}]}});
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        assert!(eligible_portable_routes(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entry_ref,
            "auto",
        )
        .is_err());
    }

    #[test]
    fn explicit_cpu_cannot_claim_generation() {
        let entry = json!({"id":"flux_schnell"});
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"flux_schnell"});
        assert!(eligible_portable_routes(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entry_ref,
            "cpu",
        )
        .is_err());
    }
}
