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
