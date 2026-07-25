//! Backend routing / gating / catalog logic split out of the `jobs_store` god module
//! (sc-8816). This is a pure code move: the SQLite jobs/workers store and the SQL-coupled
//! dispatch stay in `jobs_store.rs`, while the backend-eligibility predicates, the Mac
//! support/capability probes, the routed-model/kernel catalog, and the gap classifiers live
//! here. No routing decision, catalog membership, or public API changed.

pub(crate) mod candle;
pub(crate) mod catalog;
pub(crate) mod gaps;
pub(crate) mod mlx;

use serde_json::{Map, Value};

/// Every built-in SenseNova-U1 variant. Keep routing, understanding, and gap
/// classification on one list so adding an Infographic/distilled tier cannot
/// silently fall through to a generic unsupported reason.
pub(crate) const SENSENOVA_MODEL_IDS: &[&str] = &[
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
];

/// True when a payload key contains a non-blank string.
pub(super) fn has_nonempty_string(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// True when a payload key contains a non-empty JSON array.
pub(super) fn has_nonempty_array(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}

/// True when a payload array contains at least one non-blank string id.
pub(super) fn has_nonempty_string_array(payload: &Map<String, Value>, key: &str) -> bool {
    payload
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .any(|value| !value.trim().is_empty())
        })
}

/// True when `payload[object_key][array_key]` is a non-empty array.
pub(super) fn has_nonempty_nested_array(
    payload: &Map<String, Value>,
    object_key: &str,
    array_key: &str,
) -> bool {
    payload
        .get(object_key)
        .and_then(Value::as_object)
        .and_then(|object| object.get(array_key))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty())
}
