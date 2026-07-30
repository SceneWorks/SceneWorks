//! Shared, weight-free protocol helpers for the real SC-15508 backend adapters.
//!
//! Backend binaries deliberately return `gated` until every required measurement and lifecycle
//! scenario has actually executed. A successful model call is not silently promoted into a complete
//! calibration record.

use serde_json::{json, Map, Value};
use std::io::{self, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const INFERENCE_PIN: &str = "da57f59cf10f8d95f90d67dcfa2e0d6dc76789e3";
pub const QWEN_REPOSITORY: &str = "SceneWorks/qwen-image-mlx";
pub const KREA_REPOSITORY: &str = "SceneWorks/krea-2-turbo-mlx";
pub const COMPARISON_OUTPUT_BIAS_PARAMETER: &str = "comparisonOutputBias";

pub fn request_from_stdin() -> Result<Value, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("read provider request: {error}"))?;
    serde_json::from_str(&input).map_err(|error| format!("parse provider request JSON: {error}"))
}

pub fn write_response(response: &Value) -> Result<(), String> {
    serde_json::to_writer(io::stdout(), response)
        .map_err(|error| format!("write provider response JSON: {error}"))
}

pub fn action(request: &Value) -> Result<&str, String> {
    request
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| "provider request.action must be a string".to_owned())
}

pub fn planned(request: &Value) -> Result<&Value, String> {
    request
        .get("planned")
        .ok_or_else(|| "run request is missing planned".to_owned())
}

pub fn strategy_parameters(request: &Value) -> Result<&Map<String, Value>, String> {
    planned(request)?
        .pointer("/strategy/parameters")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.strategy.parameters must be an object".to_owned())
}

pub fn parameter(request: &Value, name: &str) -> Result<u32, String> {
    let value = strategy_parameters(request)?
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("planned.strategy.parameters.{name} must be an integer"))?;
    u32::try_from(value).map_err(|_| format!("planned.strategy.parameters.{name} exceeds u32"))
}

pub fn comparison_output_bias(
    parameters: &Map<String, Value>,
    expected_failure: bool,
) -> Result<Option<f64>, String> {
    let bias = parameters
        .get(COMPARISON_OUTPUT_BIAS_PARAMETER)
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                format!(
                    "planned.strategy.parameters.{COMPARISON_OUTPUT_BIAS_PARAMETER} must be a number"
                )
            })
        })
        .transpose()?;
    match (expected_failure, bias) {
        (false, None) => Ok(None),
        (false, Some(_)) => Err(format!(
            "{COMPARISON_OUTPUT_BIAS_PARAMETER} is reserved for an expected-failure case"
        )),
        (true, None) => Err(format!(
            "expected-failure case must declare {COMPARISON_OUTPUT_BIAS_PARAMETER}"
        )),
        (true, Some(value)) if value.is_finite() && value > 0.0 => Ok(Some(value)),
        (true, Some(_)) => Err(format!(
            "planned.strategy.parameters.{COMPARISON_OUTPUT_BIAS_PARAMETER} must be finite and greater than zero"
        )),
    }
}

pub fn max_mean_abs(
    left: &[f32],
    right: &[f32],
    comparison_output_bias: Option<f64>,
) -> Result<(f64, f64), String> {
    if left.len() != right.len() || left.is_empty() {
        return Err(format!(
            "decode output length mismatch: baseline={} tiled={}",
            left.len(),
            right.len()
        ));
    }
    if comparison_output_bias.is_some_and(|bias| !bias.is_finite() || bias <= 0.0) {
        return Err("comparison output bias must be finite and greater than zero".to_owned());
    }
    let mut maximum = 0.0_f64;
    let mut sum = 0.0_f64;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            return Err(format!(
                "decode output contains a non-finite sample at index {index}"
            ));
        }
        let baseline = f64::from(*left);
        let tiled = f64::from(*right);
        let compared = comparison_output_bias.map_or(tiled, |bias| {
            if tiled >= baseline {
                tiled + bias
            } else {
                tiled - bias
            }
        });
        let difference = (baseline - compared).abs();
        if !compared.is_finite() || !difference.is_finite() {
            return Err(format!(
                "decode comparison produced a non-finite result at index {index}"
            ));
        }
        maximum = maximum.max(difference);
        sum += difference;
        if !sum.is_finite() {
            return Err("decode comparison mean accumulator became non-finite".to_owned());
        }
    }
    let mean = sum / left.len() as f64;
    if !maximum.is_finite() || !mean.is_finite() {
        return Err("decode comparison metrics must be finite".to_owned());
    }
    Ok((maximum, mean))
}

pub fn validate_comparison_shapes(left: &[i32], right: &[i32]) -> Result<(), String> {
    if left != right {
        return Err(format!(
            "decode output shape mismatch: baseline={left:?} tiled={right:?}"
        ));
    }
    Ok(())
}

pub fn expected_failure(request: &Value) -> bool {
    planned(request)
        .ok()
        .and_then(|planned| planned.get("expectedResult").and_then(Value::as_str))
        == Some("failed")
}

pub fn target_geometry(request: &Value) -> Result<(u32, u32), String> {
    let geometry = planned(request)?
        .pointer("/target/geometry")
        .and_then(Value::as_object)
        .ok_or_else(|| "planned.target.geometry must be an object".to_owned())?;
    let dimension = |name: &str| {
        geometry
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("planned.target.geometry.{name} must fit u32"))
    };
    Ok((dimension("width")?, dimension("height")?))
}

pub fn required_env(name: &str) -> Result<String, String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("required environment variable {name} is not set"))
}

pub fn validate_artifact_identity(
    repository: &str,
    revision: &str,
    expected_repository: &str,
) -> Result<(), String> {
    if repository != expected_repository {
        return Err(format!(
            "artifact repository must be the fixed {expected_repository} calibration artifact"
        ));
    }
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("artifact revision must be an exact lowercase 40-hex commit".to_owned());
    }
    Ok(())
}

pub fn validate_huggingface_snapshot_root(
    canonical_root: &Path,
    repository: &str,
    revision: &str,
    variant: &str,
    expected_repository: &str,
) -> Result<(), String> {
    validate_artifact_identity(repository, revision, expected_repository)?;
    let repository_component = format!("models--{}", expected_repository.replace('/', "--"));
    let expected = [
        repository_component.as_str(),
        "snapshots",
        revision,
        variant,
    ];
    let components = canonical_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    if !components.ends_with(&expected) {
        return Err(format!(
            "artifact root must end with /{repository_component}/snapshots/{revision}/{variant}"
        ));
    }
    Ok(())
}

pub fn not_run_scenarios(blocker: &str) -> Value {
    Value::Array(
        [
            "exact_fit",
            "unknown_budget",
            "stale_evidence",
            "warm_repeat",
            "cancel",
            "error",
            "loadability",
            "overlay",
        ]
        .into_iter()
        .map(|name| json!({ "name": name, "result": "not_run", "reason": blocker }))
        .collect(),
    )
}

pub fn diagnostics(
    adapter: &str,
    execution: &str,
    blockers: impl IntoIterator<Item = String>,
    measurements: impl IntoIterator<Item = (&'static str, &'static str, u64)>,
) -> Value {
    json!({
        "adapter": adapter,
        "execution": execution,
        "blockers": blockers.into_iter().collect::<Vec<_>>(),
        "measurements": measurements
            .into_iter()
            .map(|(name, unit, value)| json!({ "name": name, "unit": unit, "value": value }))
            .collect::<Vec<_>>(),
    })
}

pub fn captured_at() -> String {
    // UTC conversion without adding a time-formatting dependency to the calibration-only crate.
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (seconds / 86_400) as i64;
    let day_seconds = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        (day_seconds % 3_600) / 60,
        day_seconds % 60
    )
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    // Howard Hinnant's proleptic-Gregorian civil_from_days, shifted from 1970-01-01.
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

pub fn gated_fragment(
    artifact: Value,
    sweep: Value,
    blocker: &str,
    quality: Value,
    negative_mutation: Value,
    loadability: Value,
    diagnostics: Value,
) -> Value {
    json!({
        "status": "gated",
        "artifact": artifact,
        "sweep": sweep,
        "scenarios": not_run_scenarios(blocker),
        "predictedPeakBytes": null,
        "observedMemory": null,
        "quality": quality,
        "negativeMutation": negative_mutation,
        "loadability": loadability,
        "diagnostics": diagnostics,
        "capturedAt": captured_at(),
    })
}

pub fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("memory-strategy provider adapter: {}", message.as_ref());
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_epoch_formats_as_rfc3339() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_662), (2026, 7, 28));
        assert!(captured_at().ends_with('Z'));
    }

    #[test]
    fn parameter_rejects_missing_or_non_integer_values() {
        let request = json!({
            "planned": { "strategy": { "parameters": { "decodeTileEdge": 512 } } }
        });
        assert_eq!(parameter(&request, "decodeTileEdge").unwrap(), 512);
        assert!(parameter(&request, "decodeOverlap").is_err());
    }

    #[test]
    fn identical_output_comparison_is_unmodified_without_a_negative_bias() {
        let left = [0.25_f32, -0.5, 1.0];
        let right = [0.25_f32, -0.49, 0.98];
        let (maximum, mean) = max_mean_abs(&left, &right, None).unwrap();
        assert!((maximum - 0.02).abs() < 1e-6);
        assert!((mean - 0.01).abs() < 1e-6);
    }

    #[test]
    fn deterministic_output_bias_forces_a_measured_parity_breach() {
        let left = [0.25_f32, -0.5, 1.0];
        let right = [0.25_f32, -0.49, 0.98];
        let (maximum, mean) = max_mean_abs(&left, &right, Some(0.05)).unwrap();
        assert!(maximum > 0.03);
        assert!(mean > 0.003);
        assert!((maximum - 0.07).abs() < 1e-6);
        assert!((mean - 0.06).abs() < 1e-6);
    }

    #[test]
    fn comparison_rejects_non_finite_samples() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_mean_abs(&[non_finite], &[0.0], None)
                .unwrap_err()
                .contains("non-finite sample"));
            assert!(max_mean_abs(&[0.0], &[non_finite], Some(0.05))
                .unwrap_err()
                .contains("non-finite sample"));
        }
    }

    #[test]
    fn comparison_rejects_a_non_finite_computed_accumulator() {
        let error = max_mean_abs(&[0.0, 0.0], &[0.0, 0.0], Some(f64::MAX)).unwrap_err();
        assert!(error.contains("mean accumulator became non-finite"));
    }

    #[test]
    fn comparison_shapes_must_match_exactly_before_flattening() {
        assert!(validate_comparison_shapes(&[1, 4, 4, 3], &[1, 4, 4, 3]).is_ok());
        let error = validate_comparison_shapes(&[1, 4, 4, 3], &[1, 8, 2, 3]).unwrap_err();
        assert!(error.contains("shape mismatch"));
    }

    #[test]
    fn comparison_bias_is_required_only_for_expected_failures() {
        let positive = json!({ "decodeTileEdge": 256, "decodeOverlap": 32 })
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(comparison_output_bias(&positive, false).unwrap(), None);
        assert!(comparison_output_bias(&positive, true).is_err());

        let negative = json!({
            "decodeTileEdge": 256,
            "decodeOverlap": 32,
            "comparisonOutputBias": 0.05
        })
        .as_object()
        .unwrap()
        .clone();
        assert_eq!(comparison_output_bias(&negative, true).unwrap(), Some(0.05));
        assert!(comparison_output_bias(&negative, false).is_err());
    }

    #[test]
    fn qwen_identity_rejects_wrong_repository_revision_and_root() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let root = Path::new(
            "/cache/models--SceneWorks--qwen-image-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/bf16",
        );
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_snapshot_root(
            root,
            "Qwen/Qwen-Image",
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789ABCDEF0123456789abcdef01234567",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789abcdef0123456789abcdef0123456g",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            QWEN_REPOSITORY,
            "0123456789abcdef",
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            Path::new(
                "/cache/models--SceneWorks--qwen-image-mlx/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/bf16",
            ),
            QWEN_REPOSITORY,
            revision,
            "bf16",
            QWEN_REPOSITORY
        )
        .is_err());
    }

    #[test]
    fn krea_identity_rejects_wrong_repository_revision_and_root() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let root = Path::new(
            "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/q4",
        );
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_ok());
        assert!(validate_huggingface_snapshot_root(
            root,
            "SceneWorks/krea-2-turbo-candle",
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789abcdef0123456789abcdef0123456g",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789ABCDEF0123456789abcdef01234567",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            root,
            KREA_REPOSITORY,
            "0123456789abcdef",
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
        assert!(validate_huggingface_snapshot_root(
            Path::new(
                "/cache/models--SceneWorks--krea-2-turbo-mlx/snapshots/0123456789abcdef0123456789abcdef01234567/bf16",
            ),
            KREA_REPOSITORY,
            revision,
            "q4",
            KREA_REPOSITORY
        )
        .is_err());
    }
}
