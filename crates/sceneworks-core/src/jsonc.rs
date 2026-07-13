//! Minimal JSONC (`.jsonc`) comment stripping for the manifest readers shared by
//! the rust-api and the rust-worker (sc-4279 / F-MLXW-15). Both crates read the
//! same `.jsonc` model/LoRA manifests, so the stripper lives here once rather
//! than byte-identically in each.

/// Strip `//` line and `/* */` block comments from a JSONC string, leaving the
/// JSON otherwise byte-for-byte (newlines inside line comments are preserved so
/// downstream error spans keep their line numbers). String literals — including
/// escaped quotes — are passed through untouched so a `//` inside a string is not
/// mistaken for a comment.
pub fn strip_jsonc_comments(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(character) = chars.next() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        if character == '"' {
            in_string = true;
            output.push(character);
            continue;
        }
        if character == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\r' || next == '\n' {
                    output.push(next);
                    break;
                }
            }
            continue;
        }
        if character == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut previous = '\0';
            for next in chars.by_ref() {
                if previous == '*' && next == '/' {
                    break;
                }
                previous = next;
            }
            continue;
        }
        output.push(character);
    }
    output
}

/// Reject any object that declares the same key twice, anywhere in `json`
/// (recursively). Returns `Err` naming the first offending key.
///
/// serde_json / JSON are last-key-wins: a duplicate key in an object is *valid*
/// JSON — it produces no parse error and silently discards the earlier value.
/// That once shipped as a real regression: a second `ui` block in a model entry
/// dropped its `img2img` flag, so the "Image reference" slider silently never
/// worked and cost several debugging rounds to find (sc-10198, fixed in #1249).
/// The manifests are hand-edited JSONC, so the same "add a field that already
/// exists in another block" mistake can recur; this guard turns it into a hard
/// error a test/build can gate on instead of a silent data loss (sc-10199).
///
/// `json` must already be comment-free — run [`strip_jsonc_comments`] first.
pub fn reject_duplicate_keys(json: &str) -> Result<(), String> {
    use serde::de::Deserialize;
    let mut deserializer = serde_json::Deserializer::from_str(json);
    UniqueKeys::deserialize(&mut deserializer).map_err(|error| error.to_string())?;
    // Ensure the whole document was consumed (no trailing garbage after the value).
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(())
}

/// Zero-sized witness that a JSON value's objects all had unique keys. Its
/// `Deserialize` walks the serde_json token stream — which yields *every* key,
/// duplicates included, since dedup only happens when building a `Map` — and
/// errors on the first repeat within a single object.
struct UniqueKeys;

impl<'de> serde::de::Deserialize<'de> for UniqueKeys {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueKeysVisitor)
    }
}

struct UniqueKeysVisitor;

impl<'de> serde::de::Visitor<'de> for UniqueKeysVisitor {
    type Value = UniqueKeys;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        use serde::de::Error;
        let mut seen = std::collections::HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            // Recurse so a duplicate nested inside this key's value is caught too.
            map.next_value::<UniqueKeys>()?;
        }
        Ok(UniqueKeys)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        while seq.next_element::<UniqueKeys>()?.is_some() {}
        Ok(UniqueKeys)
    }

    // Scalars carry no keys — accept them so `deserialize_any` covers every leaf.
    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueKeys)
    }
}

#[cfg(test)]
mod tests {
    use super::{reject_duplicate_keys, strip_jsonc_comments};

    #[test]
    fn strips_line_and_block_comments_but_preserves_strings() {
        let input = r#"{
            // a line comment
            "url": "https://example.com", /* trailing block */
            "note": "a // b /* c */ d"
        }"#;
        let stripped = strip_jsonc_comments(input);
        // Comments gone, but the string literal (incl. its // and /* */) survives.
        assert!(!stripped.contains("a line comment"));
        assert!(!stripped.contains("trailing block"));
        assert!(stripped.contains(r#""note": "a // b /* c */ d""#));
        assert!(stripped.contains(r#""url": "https://example.com""#));
        // Still valid JSON after stripping.
        let parsed: serde_json::Value = serde_json::from_str(&stripped).expect("valid json");
        assert_eq!(parsed["url"], "https://example.com");
    }

    #[test]
    fn accepts_documents_with_only_unique_keys() {
        let json = r#"{
            "a": 1,
            "b": { "c": true, "d": [ { "e": null }, { "e": "ok" } ] },
            "f": [1, 2, 3]
        }"#;
        // Repeating a key across DIFFERENT objects (both `e`) is fine — last-key-wins
        // only bites within a single object.
        assert!(reject_duplicate_keys(json).is_ok());
    }

    #[test]
    fn rejects_a_duplicate_key_at_the_top_level() {
        // The exact shape of the img2img regression: the second block silently wins.
        let json = r#"{ "ui": { "img2img": true }, "ui": {} }"#;
        let error = reject_duplicate_keys(json).expect_err("duplicate top-level key rejected");
        assert!(
            error.contains("ui"),
            "error names the offending key: {error}"
        );
    }

    #[test]
    fn rejects_a_duplicate_key_nested_deep() {
        // A duplicate buried inside an array element must still be caught.
        let json = r#"{ "models": [ { "id": "x", "id": "y" } ] }"#;
        let error = reject_duplicate_keys(json).expect_err("nested duplicate key rejected");
        assert!(
            error.contains("id"),
            "error names the offending key: {error}"
        );
    }

    #[test]
    fn strips_comments_then_rejects_the_duplicate_underneath() {
        // Comments can hide a duplicate from a casual read; strip first, then guard.
        let jsonc = r#"{
            "flag": true, // first
            "flag": false // oops, second wins silently
        }"#;
        let stripped = strip_jsonc_comments(jsonc);
        assert!(reject_duplicate_keys(&stripped).is_err());
    }
}
