//! Shared Hugging Face repository grammar. Case is significant for display and is preserved.
//! Short repository names remain supported by the native downloader; persisted provenance and
//! share envelopes require an explicit namespace. Whitespace is removed before use or storage.

/// Parse a relative `name` or `namespace/name`, returning its normalized spelling.
/// Underscores and the historically supported `.git` suffix are intentionally retained.
pub fn parse(value: &str) -> Result<&str, &'static str> {
    let value = value.trim();
    let mut count = 0;
    for part in value.split('/') {
        count += 1;
        if part.is_empty()
            || part.starts_with(['-', '.'])
            || part.ends_with(['-', '.'])
            || part.contains("--")
            || part.contains("..")
            || !part
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
        {
            return Err("Hugging Face repo components must contain letters, numbers, `.`, `_`, or `-`, without leading/trailing dots or dashes, `..`, or `--`");
        }
    }
    if count > 2 {
        return Err("Hugging Face repo must be `name` or `namespace/name`");
    }
    Ok(value)
}

/// The same grammar with the explicit namespace required by catalog/provenance contracts.
pub fn namespaced(value: &str) -> Result<&str, &'static str> {
    let value = parse(value)?;
    value
        .contains('/')
        .then_some(value)
        .ok_or("Hugging Face repo must be in owner/name form")
}

#[cfg(test)]
mod tests {
    #[test]
    fn grammar_preserves_supported_names_and_normalizes_storage() {
        for value in ["repo", "Owner/Model.v2", "_owner/_model", "owner/model.git"] {
            assert_eq!(super::parse(&format!("  {value}  ")), Ok(value));
        }
        assert!(super::namespaced("repo").is_err());
        assert_eq!(super::namespaced(" Owner/Model "), Ok("Owner/Model"));
        for value in [
            "",
            "/a/b",
            "a/b/",
            "a/b/c",
            "../x",
            "a/..",
            "a\\b",
            "a/b--c",
            "a/b..c",
            "a/-b",
            "a/b.",
            "a/b c",
            "a/é",
            "https://hf.co/a/b",
            "a/b\nc",
        ] {
            assert!(super::parse(value).is_err(), "{value:?}");
        }
    }
}
