pub fn slugify(value: &str, fallback: &str, max_length: Option<usize>) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !slug.is_empty() {
            slug.push('-');
            previous_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = fallback.to_owned();
    }
    if let Some(max_length) = max_length {
        slug.truncate(max_length);
        // Truncating can split mid-separator (e.g. "my project"/max 3 -> "my-"), so
        // re-run the trailing-dash trim afterwards to keep the slug clean (sc-8951).
        while slug.ends_with('-') {
            slug.pop();
        }
    }
    slug
}

#[cfg(test)]
mod tests {
    use super::slugify;

    #[test]
    fn lowercases_and_collapses_separators() {
        assert_eq!(slugify("My Project", "fallback", None), "my-project");
        assert_eq!(slugify("  a  b  ", "fallback", None), "a-b");
        assert_eq!(slugify("a__b--c", "fallback", None), "a-b-c");
    }

    #[test]
    fn trims_leading_and_trailing_separators() {
        assert_eq!(slugify("-hello-", "fallback", None), "hello");
        assert_eq!(slugify("!!!name!!!", "fallback", None), "name");
    }

    #[test]
    fn falls_back_when_nothing_survives() {
        assert_eq!(slugify("", "fallback", None), "fallback");
        assert_eq!(slugify("---", "fallback", None), "fallback");
        assert_eq!(slugify("!@#", "fallback", None), "fallback");
    }

    /// sc-8951 / F-149: truncation must not reintroduce a trailing dash. `"my project"`
    /// slugs to `"my-project"`; truncated to 3 that would be `"my-"`, which the
    /// post-truncate trim cleans back to `"my"`.
    #[test]
    fn truncation_does_not_leave_a_trailing_dash() {
        assert_eq!(slugify("my project", "fallback", Some(3)), "my");
        // A mid-word boundary keeps its characters (no separator to trim).
        assert_eq!(slugify("my project", "fallback", Some(4)), "my-p");
        // Truncating exactly onto a separator strips it too.
        assert_eq!(slugify("ab cd ef", "fallback", Some(3)), "ab");
    }

    #[test]
    fn truncation_shorter_than_the_full_slug_still_yields_a_clean_prefix() {
        assert_eq!(slugify("hello world", "fallback", Some(5)), "hello");
        assert_eq!(slugify("hello world", "fallback", Some(100)), "hello-world");
    }
}
