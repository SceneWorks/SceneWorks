//! Canonical built-in angle landmark presets (epic 4422).
//!
//! The single source of truth for the 11 Character-Studio angle views: each is a 5-point
//! face landmark set (`[left_eye, right_eye, nose, mouth_left, mouth_right]`) normalized to a
//! square `0.0..=1.0` canvas. The worker passes these to the InstantID engine via
//! `generate_with_kps` (sc-4424/sc-4425); the API serves them as the **built-in presets** of the
//! Key Point Library (sc-4434). Tuned for LoRA-training coverage — head-and-shoulders fill the
//! frame, shoulders retained — and validated on real weights (`docs/sc-4422-kps-experiment/`,
//! sc-4427). Built-ins are always-present and protected: they live here, not as deletable
//! library assets (mirrors the bundled built-in pose set).
//!
//! The Python worker keeps its own `instantid_adapter.VIEW_ANGLE_KPS` copy (cross-language, the
//! Win/Linux path) — these values must stay byte-identical to it.

use serde_json::{json, Value};

/// The 11 canonical Character-Studio angle names, in display/generation order. Built-in
/// collections (and the default set) iterate this order.
pub const BUILTIN_ANGLE_SET_ORDER: [&str; 11] = [
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "left_profile",
    "right_profile",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
];

/// The validated default 5-point landmark presets, index-aligned with
/// [`BUILTIN_ANGLE_SET_ORDER`]. Some coordinates sit near math constants (e.g. 0.3927 ≈ π/8) but
/// must stay verbatim — they are empirical (sc-4424/sc-4427), not derived.
#[allow(clippy::approx_constant)]
pub const BUILTIN_ANGLE_KPS: [(&str, [(f32, f32); 5]); 11] = [
    (
        "front",
        [
            (0.4033, 0.3420),
            (0.5938, 0.3393),
            (0.5014, 0.4317),
            (0.4298, 0.5353),
            (0.5771, 0.5334),
        ],
    ),
    (
        "three_quarter_left",
        [
            (0.3973, 0.3380),
            (0.5057, 0.3432),
            (0.3734, 0.4268),
            (0.4024, 0.5330),
            (0.4785, 0.5357),
        ],
    ),
    (
        "three_quarter_right",
        [
            (0.4867, 0.3434),
            (0.6117, 0.3379),
            (0.6241, 0.4225),
            (0.5249, 0.5366),
            (0.6149, 0.5322),
        ],
    ),
    (
        "left_profile",
        [
            (0.3022, 0.3339),
            (0.3421, 0.3474),
            (0.2381, 0.4301),
            (0.3020, 0.5304),
            (0.3200, 0.5384),
        ],
    ),
    (
        "right_profile",
        [
            (0.6591, 0.3487),
            (0.7145, 0.3325),
            (0.7786, 0.4309),
            (0.6893, 0.5397),
            (0.7235, 0.5290),
        ],
    ),
    (
        "up",
        [
            (0.3973, 0.3404),
            (0.6069, 0.3408),
            (0.5036, 0.3826),
            (0.4155, 0.5343),
            (0.5919, 0.5344),
        ],
    ),
    (
        "down",
        [
            (0.4024, 0.4030),
            (0.6046, 0.3982),
            (0.5075, 0.5214),
            (0.4272, 0.5957),
            (0.5807, 0.5930),
        ],
    ),
    (
        "up_left",
        [
            (0.4025, 0.3298),
            (0.5575, 0.3461),
            (0.4018, 0.3944),
            (0.3686, 0.5243),
            (0.4840, 0.5391),
        ],
    ),
    (
        "up_right",
        [
            (0.4607, 0.3508),
            (0.6113, 0.3305),
            (0.6086, 0.4063),
            (0.5228, 0.5431),
            (0.6368, 0.5257),
        ],
    ),
    (
        "down_left",
        [
            (0.3677, 0.4103),
            (0.5168, 0.3910),
            (0.3988, 0.5100),
            (0.4260, 0.6021),
            (0.5312, 0.5866),
        ],
    ),
    (
        "down_right",
        [
            (0.4793, 0.3987),
            (0.6326, 0.4026),
            (0.6159, 0.4961),
            (0.4945, 0.5928),
            (0.6096, 0.5959),
        ],
    ),
];

/// The stable preset id for a built-in angle (`front` → `builtin_front`). Used by collections to
/// reference built-ins without persisting their kps (they live in this module).
pub fn builtin_preset_id(angle: &str) -> String {
    format!("builtin_{angle}")
}

/// The id of the seeded default collection (the 11 built-ins in [`BUILTIN_ANGLE_SET_ORDER`]).
pub const BUILTIN_DEFAULT_COLLECTION_ID: &str = "builtin_default";

/// The canonical kps preset for an angle name, or `None` for an unknown name.
pub fn angle_kps(angle: &str) -> Option<[(f32, f32); 5]> {
    BUILTIN_ANGLE_KPS
        .iter()
        .find(|(name, _)| *name == angle)
        .map(|(_, kps)| *kps)
}

/// Human-readable label for a canonical angle (`three_quarter_left` → `Three-Quarter Left`).
pub fn builtin_angle_display_name(angle: &str) -> String {
    match angle {
        "front" => "Front",
        "three_quarter_left" => "Three-Quarter Left",
        "three_quarter_right" => "Three-Quarter Right",
        "left_profile" => "Left Profile",
        "right_profile" => "Right Profile",
        "up" => "Up",
        "down" => "Down",
        "up_left" => "Up-Left",
        "up_right" => "Up-Right",
        "down_left" => "Down-Left",
        "down_right" => "Down-Right",
        other => other,
    }
    .to_owned()
}

/// One built-in preset as a library record: `{ id, name, angle, kps[[x,y]×5], builtin: true,
/// sourceImageRef: null }`. `sourceImageRef` is null because the built-ins' framing was tuned,
/// not extracted from a single committed photo — the UI renders the kps overlay for display.
pub fn builtin_preset_record(angle: &str, kps: &[(f32, f32); 5]) -> Value {
    json!({
        "id": builtin_preset_id(angle),
        "name": builtin_angle_display_name(angle),
        "angle": angle,
        "kps": kps.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>(),
        "builtin": true,
        "sourceImageRef": Value::Null,
    })
}

/// All 11 built-in presets as library records, in [`BUILTIN_ANGLE_SET_ORDER`].
pub fn builtin_preset_records() -> Vec<Value> {
    BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| builtin_preset_record(angle, kps))
        .collect()
}

/// The seeded default collection: the 11 built-ins in order, marked default + builtin.
pub fn builtin_default_collection() -> Value {
    json!({
        "id": BUILTIN_DEFAULT_COLLECTION_ID,
        "name": "Default angles",
        "orderedPresetIds": BUILTIN_ANGLE_SET_ORDER
            .iter()
            .map(|angle| Value::String(builtin_preset_id(angle)))
            .collect::<Vec<_>>(),
        "isDefault": true,
        "builtin": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_and_table_are_index_aligned() {
        assert_eq!(BUILTIN_ANGLE_KPS.len(), BUILTIN_ANGLE_SET_ORDER.len());
        for (i, angle) in BUILTIN_ANGLE_SET_ORDER.iter().enumerate() {
            assert_eq!(
                BUILTIN_ANGLE_KPS[i].0, *angle,
                "order/table mismatch at {i}"
            );
        }
    }

    #[test]
    fn every_preset_is_5_points_in_unit_square() {
        for (angle, kps) in BUILTIN_ANGLE_KPS.iter() {
            assert_eq!(kps.len(), 5, "{angle}");
            for (x, y) in kps {
                assert!(
                    (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y),
                    "{angle} {x},{y}"
                );
            }
        }
    }

    #[test]
    fn angle_kps_lookup() {
        assert_eq!(angle_kps("front"), Some(BUILTIN_ANGLE_KPS[0].1));
        assert!(angle_kps("nope").is_none());
    }

    #[test]
    fn builtin_records_cover_all_angles_with_stable_ids() {
        let records = builtin_preset_records();
        assert_eq!(records.len(), 11);
        assert_eq!(records[0]["id"], "builtin_front");
        assert_eq!(records[0]["name"], "Front");
        assert_eq!(records[0]["builtin"], true);
        assert!(records[0]["sourceImageRef"].is_null());
        assert_eq!(records[0]["kps"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn default_collection_orders_all_builtins() {
        let collection = builtin_default_collection();
        assert_eq!(collection["isDefault"], true);
        let ids = collection["orderedPresetIds"].as_array().unwrap();
        assert_eq!(ids.len(), 11);
        assert_eq!(ids[0], "builtin_front");
    }
}
