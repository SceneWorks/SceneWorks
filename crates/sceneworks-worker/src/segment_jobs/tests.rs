use super::*;
use serde_json::json;

fn obj(value: Value) -> JsonObject {
    value.as_object().unwrap().clone()
}

#[test]
fn parse_box_accepts_xyxy_array() {
    let payload = obj(json!({ "box": [10.0, 20.0, 110.0, 220.0] }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_accepts_rect_object() {
    // {x,y,width,height} → [x, y, x+w, y+h] (the editor's rect shape).
    let payload = obj(json!({ "box": { "x": 10.0, "y": 20.0, "width": 100.0, "height": 200.0 } }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_accepts_integer_json_numbers() {
    let payload = obj(json!({ "box": [10, 20, 110, 220] }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_rejects_missing_or_malformed() {
    assert!(parse_box(&obj(json!({}))).is_err());
    assert!(parse_box(&obj(json!({ "box": [1.0, 2.0, 3.0] }))).is_err());
    assert!(parse_box(&obj(json!({ "box": "nope" }))).is_err());
    assert!(parse_box(&obj(json!({ "box": { "x": 1.0, "y": 2.0 } }))).is_err());
}

// --- sc-6346: fg/bg point prompts (SAM3 tracker path) ---

#[test]
fn parse_points_absent_or_empty_is_none() {
    // Absent → None (caller falls back to the box path).
    assert_eq!(
        parse_points(&obj(json!({ "box": [1, 2, 3, 4] }))).unwrap(),
        None
    );
    // Empty array → None too.
    assert_eq!(parse_points(&obj(json!({ "points": [] }))).unwrap(), None);
}

#[test]
fn parse_points_reads_xy_and_label() {
    // Explicit int label 1=fg / 0=bg, and a default (no label) = fg.
    let payload = obj(json!({
        "points": [
            { "x": 10.0, "y": 20.0, "label": 1 },
            { "x": 30, "y": 40, "label": 0 },
            { "x": 5.5, "y": 6.5 }
        ]
    }));
    let pts = parse_points(&payload).unwrap().unwrap();
    assert_eq!(pts, vec![(10.0, 20.0, 1), (30.0, 40.0, 0), (5.5, 6.5, 1)]);
}

#[test]
fn parse_points_accepts_fg_bool_and_clamps_label() {
    // `fg: false` → background; any non-zero `label` → foreground (1).
    let payload = obj(json!({
        "points": [
            { "x": 1.0, "y": 2.0, "fg": false },
            { "x": 3.0, "y": 4.0, "fg": true },
            { "x": 5.0, "y": 6.0, "label": 7 }
        ]
    }));
    let pts = parse_points(&payload).unwrap().unwrap();
    assert_eq!(pts, vec![(1.0, 2.0, 0), (3.0, 4.0, 1), (5.0, 6.0, 1)]);
}

#[test]
fn parse_points_rejects_malformed() {
    // Not an array.
    assert!(parse_points(&obj(json!({ "points": "nope" }))).is_err());
    // Entry not an object.
    assert!(parse_points(&obj(json!({ "points": [[1.0, 2.0]] }))).is_err());
    // Missing y.
    assert!(parse_points(&obj(json!({ "points": [{ "x": 1.0 }] }))).is_err());
}
