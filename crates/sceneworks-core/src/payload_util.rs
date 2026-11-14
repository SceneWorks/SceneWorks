pub use sceneworks_options::payload_util::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::JsonObject;
    use serde_json::{json, Value};

    fn payload(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    // --- Characterization tests capturing the CURRENT per-lane behavior (sc-8817). ---
    // The image lane historically used `string_or` = RAW and `nonempty_string_or` =
    // trim+empty-filter; the video lane's `string_or` was actually the trim+filter variant.
    // These lock the raw-vs-trim distinction so the dedup is provably behavior-preserving.

    #[test]
    fn image_and_video_quant_bits_share_exact_integer_parsing() {
        use crate::{image_request::ImageRequest, video_request::VideoRequest};
        for (value, expected) in [
            (serde_json::json!(4), Some(4)),
            (serde_json::json!(" 8 "), Some(8)),
            (serde_json::json!(0), Some(0)),
            (serde_json::json!(-1), Some(-1)),
            (serde_json::json!(4.5), None),
            (serde_json::json!(4.0), None),
            (serde_json::json!(true), None),
            (serde_json::json!(null), None),
            (serde_json::json!(u64::MAX), None),
            (serde_json::json!("9223372036854775808"), None),
        ] {
            let payload = serde_json::json!({"advanced":{"mlxQuantize":value}})
                .as_object()
                .unwrap()
                .clone();
            assert_eq!(ImageRequest::from_payload(&payload).quant_bits(), expected);
            assert_eq!(VideoRequest::from_payload(&payload).quant_bits(), expected);
        }
        let empty = JsonObject::new();
        assert_eq!(ImageRequest::from_payload(&empty).quant_bits(), None);
        assert_eq!(VideoRequest::from_payload(&empty).quant_bits(), None);
    }

    #[test]
    fn string_or_returns_raw_value_including_present_but_empty() {
        // Present-but-empty stays empty (does NOT fall back to the default).
        let p = payload(json!({ "k": "" }));
        assert_eq!(string_or(&p, "k", "def"), "");
        // Leading/trailing whitespace is preserved verbatim.
        let p = payload(json!({ "k": "  spaced  " }));
        assert_eq!(string_or(&p, "k", "def"), "  spaced  ");
        // Absent / non-string → default.
        let p = payload(json!({ "n": 5 }));
        assert_eq!(string_or(&p, "k", "def"), "def");
        assert_eq!(string_or(&p, "n", "def"), "def");
    }

    #[test]
    fn nonempty_string_or_trims_and_empty_filters() {
        // Present-but-empty / whitespace-only → default.
        let p = payload(json!({ "k": "" }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
        let p = payload(json!({ "k": "   " }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
        // Non-blank values are trimmed.
        let p = payload(json!({ "k": "  spaced  " }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "spaced");
        // Absent → default.
        let p = payload(json!({}));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
    }

    #[test]
    fn optional_id_trims_and_drops_blanks() {
        let p = payload(json!({ "id": "  x  ", "blank": "  ", "n": 3 }));
        assert_eq!(optional_id(&p, "id").as_deref(), Some("x"));
        assert!(optional_id(&p, "blank").is_none());
        assert!(optional_id(&p, "n").is_none());
        assert!(optional_id(&p, "missing").is_none());
    }

    #[test]
    fn quant_selection_prefers_explicit_then_the_bound_tier_and_reports_malformed_values() {
        let advanced = |value: Value| value.as_object().unwrap().clone();
        assert_eq!(
            quant_selection(&advanced(json!({"__portableBoundTier": "q4"}))),
            QuantSelection::Bound(4)
        );
        assert_eq!(
            quant_selection(&advanced(
                json!({"mlxQuantize": null, "__portableBoundTier": "bf16"})
            )),
            QuantSelection::Null,
            "a present null is never replaced by the bound tier"
        );
        assert_eq!(
            request_quant_bits(&advanced(
                json!({"mlxQuantize": null, "__portableBoundTier": "q4"})
            )),
            None
        );
        assert_eq!(
            quant_selection(&advanced(
                json!({"mlxQuantize": "8", "__portableBoundTier": "q4"})
            )),
            QuantSelection::Explicit(8)
        );
        let malformed = advanced(json!({"mlxQuantize": 4.5, "__portableBoundTier": "q4"}));
        assert!(matches!(
            quant_selection(&malformed),
            QuantSelection::Malformed(_)
        ));
        assert_eq!(request_quant_bits(&malformed), None);
        assert_eq!(
            quant_selection(&advanced(json!({"__portableBoundTier": "default"}))),
            QuantSelection::Unselected
        );
        assert_eq!(
            quant_selection(&advanced(json!({}))),
            QuantSelection::Unselected
        );
    }

    #[test]
    fn optional_i64_reads_numbers_and_numeric_strings() {
        let p = payload(json!({ "a": 5, "b": "42", "c": "nope", "d": null }));
        assert_eq!(optional_i64(&p, "a"), Some(5));
        assert_eq!(optional_i64(&p, "b"), Some(42));
        assert_eq!(optional_i64(&p, "c"), None);
        assert_eq!(optional_i64(&p, "d"), None);
        assert_eq!(optional_i64(&p, "missing"), None);
    }

    #[test]
    fn clamped_u32_parses_and_clamps() {
        let p = payload(json!({ "a": 99, "b": "3", "c": "bad", "d": -1 }));
        assert_eq!(clamped_u32(p.get("a"), 4, 1, 8), 8);
        assert_eq!(clamped_u32(p.get("b"), 4, 1, 8), 3);
        // Unparseable / absent → default (then clamped).
        assert_eq!(clamped_u32(p.get("c"), 4, 1, 8), 4);
        assert_eq!(clamped_u32(None, 4, 1, 8), 4);
        // Negative can't become u32 → default.
        assert_eq!(clamped_u32(p.get("d"), 4, 1, 8), 4);
    }

    #[test]
    fn int_array_reads_numbers_and_numeric_strings_dropping_others() {
        let p = payload(json!({ "seeds": [1, "2", null, "bad", 3] }));
        assert_eq!(int_array(&p, "seeds"), vec![1, 2, 3]);
        let p = payload(json!({ "seeds": "notarray" }));
        assert!(int_array(&p, "seeds").is_empty());
        assert!(int_array(&payload(json!({})), "seeds").is_empty());
    }

    #[test]
    fn string_list_trims_and_drops_blanks_and_nonstrings() {
        let p = payload(json!({ "ids": ["a", "  ", "  b  ", 42, null] }));
        assert_eq!(string_list(&p, "ids"), vec!["a", "b"]);
        assert!(string_list(&payload(json!({ "ids": "x" })), "ids").is_empty());
        assert!(string_list(&payload(json!({})), "ids").is_empty());
    }

    #[test]
    fn array_and_object_or_empty_clone_or_default() {
        let p = payload(json!({ "arr": [1, 2], "obj": { "k": 1 }, "wrong": 5 }));
        assert_eq!(array_or_empty(&p, "arr"), vec![json!(1), json!(2)]);
        assert!(array_or_empty(&p, "wrong").is_empty());
        assert!(array_or_empty(&p, "missing").is_empty());
        assert_eq!(object_or_empty(&p, "obj").get("k"), Some(&json!(1)));
        assert!(object_or_empty(&p, "wrong").is_empty());
        assert!(object_or_empty(&p, "missing").is_empty());
    }
}
