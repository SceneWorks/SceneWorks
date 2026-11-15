//! Portable filesystem and route identifiers. Domain-specific limits belong to callers.
pub fn is_safe_id(value: &str) -> bool {
    is_safe_id_with_max(value, usize::MAX)
}

pub fn is_safe_id_with_max(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifiers_preserve_bounds_and_reject_route_escapes() {
        assert!(is_safe_id("aA09_-"));
        assert!(is_safe_id_with_max("abc", 3));
        assert!(!is_safe_id_with_max("abcd", 3));
        for bad in ["", " ", "a.b", "../a", "a/b", "a\\b", "%2f", "é", "a\n"] {
            assert!(!is_safe_id(bad), "{bad:?}");
        }
    }
}
