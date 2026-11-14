//! Compile the external binder-facing API, independently of the private implementation module.
use sceneworks_options::{legacy_image_flag, legacy_integer, legacy_number};
use serde_json::{json, Value};

#[test]
fn typed_route_probes_can_use_the_public_syntax_parsers() {
    assert_eq!(legacy_integer(&json!(" 4 ")), Some(json!(4)));
    assert_eq!(legacy_number(&json!(" 0.75 ")), Some(0.75));
    assert!(
        legacy_image_flag(&json!("false")),
        "generic image flags retain nonempty-string truthiness; PID has its own narrower contract"
    );
    assert_eq!(legacy_integer(&Value::Null), None);
    assert_eq!(legacy_number(&json!("NaN")), None);
    assert!(!legacy_image_flag(&Value::Null));
}
