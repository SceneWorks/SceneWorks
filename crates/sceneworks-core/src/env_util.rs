//! Environment text policies. API strings retain surrounding whitespace; worker strings and
//! filesystem paths trim it. All three fall back on absent, non-Unicode or blank values.
use std::path::{Path, PathBuf};

fn nonempty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

/// Read nonblank environment text verbatim (the API's historical string policy).
pub fn env_string(key: &str, default: &str) -> String {
    nonempty(std::env::var(key).ok()).unwrap_or_else(|| default.to_owned())
}

/// Read trimmed nonblank environment text (the worker's historical string policy).
pub fn env_trimmed_string(key: &str, default: &str) -> String {
    nonempty(std::env::var(key).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| default.to_owned())
}

/// Read a trimmed path; preserve even non-Unicode caller defaults when falling back.
pub fn env_path_or(key: &str, default: &Path) -> PathBuf {
    nonempty(std::env::var(key).ok())
        .map(|value| PathBuf::from(value.trim()))
        .unwrap_or_else(|| default.to_path_buf())
}

/// Parse SceneWorks boolean text. Values are trimmed and ASCII case-insensitive;
/// 1/true/yes/on are true, 0/false/no/off are false, all other values invalid.
pub fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Absent or blank values use the caller's default; other invalid values emit the
/// same key-only diagnostic and use that default. Strict callers use `parse_bool`
/// and return their existing configuration error instead.
pub fn flag_value(key: &str, value: Option<&str>, default: bool) -> bool {
    match value {
        None => default,
        Some(value) if value.trim().is_empty() => default,
        Some(value) => parse_bool(value).unwrap_or_else(|| {
            tracing::warn!(
                event = "invalid_env",
                key,
                "invalid boolean environment value; using configured default"
            );
            default
        }),
    }
}

/// Variant for OS-string inputs, diagnosing non-Unicode values consistently.
pub fn flag_os(key: &str, value: Option<&std::ffi::OsStr>, default: bool) -> bool {
    match value {
        None => default,
        Some(value) => match value.to_str() {
            Some(value) => flag_value(key, Some(value), default),
            None => {
                tracing::warn!(
                    event = "invalid_env",
                    key,
                    "non-Unicode boolean environment value; using configured default"
                );
                default
            }
        },
    }
}

pub fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => flag_value(key, Some(&value), default),
        Err(std::env::VarError::NotPresent) => default,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                event = "invalid_env",
                key,
                "non-Unicode boolean environment value; using configured default"
            );
            default
        }
    }
}

#[cfg(test)]
mod bool_tests {
    use super::*;

    #[test]
    fn boolean_policy_accepts_common_values_and_preserves_defaults() {
        for value in ["1", " true ", "YES", "On", "tRuE"] {
            assert_eq!(parse_bool(value), Some(true));
        }
        for value in ["0", " false ", "NO", "Off", "fAlSe"] {
            assert_eq!(parse_bool(value), Some(false));
        }
        for value in ["", "  ", "maybe", "2"] {
            assert_eq!(parse_bool(value), None);
        }
        for default in [false, true] {
            for value in [None, Some(""), Some(" "), Some("invalid")] {
                assert_eq!(flag_value("TEST_FLAG", value, default), default);
            }
        }
    }
}

/// Read the first present numeric override, refusing malformed or non-Unicode
/// values instead of falling through to an alias or default. `trim` preserves
/// the calling setting's established whitespace policy.
pub fn env_u64_any(keys: &[&str], default: u64, trim: bool) -> Result<u64, String> {
    env_u64_any_with(keys, default, trim, |key| std::env::var(key))
}

pub fn env_u64_any_with(
    keys: &[&str],
    default: u64,
    trim: bool,
    mut read: impl FnMut(&str) -> Result<String, std::env::VarError>,
) -> Result<u64, String> {
    for &key in keys {
        let raw = match read(key) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => continue,
            Err(std::env::VarError::NotUnicode(_)) => return Err(invalid_u64(key)),
        };
        let value = if trim { raw.trim() } else { raw.as_str() };
        return value.parse::<u64>().map_err(|_| invalid_u64(key));
    }
    Ok(default)
}

fn invalid_u64(key: &str) -> String {
    tracing::warn!(
        event = "invalid_env",
        key,
        "invalid unsigned integer environment value"
    );
    format!("{key} must be an unsigned 64-bit integer")
}
