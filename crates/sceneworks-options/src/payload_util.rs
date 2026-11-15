//! Shared payload-parsing helpers for the image- and video-generation job requests
//! (sc-8817). Both [`crate::image_request`] and [`crate::video_request`] parse a job
//! `payload` (a [`JsonObject`]) into a typed request, and both need the same small set
//! of "read this key, coerce it, fall back to a default" primitives.
//!
//! Historically these helpers were copy-pasted into each module and had already drifted:
//! `image_request::string_or` returned the raw string value, while
//! `video_request::string_or` trimmed and empty-filtered it (the semantics
//! `image_request` spelled `nonempty_string_or`). Collapsing them into one function would
//! silently change one lane's behavior, so this module deliberately exposes **two**
//! clearly-named string variants:
//!
//! * [`string_or`] — the RAW value (no trimming, a present-but-empty string stays empty).
//! * [`nonempty_string_or`] — trimmed, with a present-but-empty/whitespace value falling
//!   back to the default.
//!
//! Each existing call site is wired to the variant that preserves its prior behavior.

use serde_json::Value;

use crate::contracts::JsonObject;

/// Read a string key, returning the RAW value (no trimming); a present-but-empty string
/// is preserved. Falls back to `default` only when the key is absent or non-string.
pub fn string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

/// Read a string key, trimmed, where a present-but-empty (or whitespace-only) value also
/// falls back to `default` (matches the Python `.get(key, default)` where the UI never
/// sends an empty model/mode/etc.).
pub fn nonempty_string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

/// Read an optional string id: trimmed, `None` when absent, non-string, or blank.
pub fn optional_id(payload: &JsonObject, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Read an optional int that may arrive as a JSON number or a numeric string. `None`
/// when absent or unparseable.
/// What a tier selector must honour for `advanced.mlxQuantize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantSelection<'a> {
    /// The request named these bits.
    Explicit(i64),
    /// The request has no `mlxQuantize` key; the verified portable binding froze this tier.
    Bound(i64),
    /// No `mlxQuantize` key and no frozen tier: the selector's own default applies.
    Unselected,
    /// The key is present with JSON `null`. Selectors differ on whether that means "unselected" or
    /// is refused, so it is reported as its own case and never replaced by the bound tier.
    Null,
    /// The request named a value that is not an integer bit count: refuse it.
    Malformed(&'a Value),
}

/// The tier the worker's source guard froze for this job (`advanced.__portableBoundTier`, written
/// only after the route was verified), as quantization bits. A pinned Hugging Face member is
/// served in place from the worker's hub snapshot, which can hold other installed tiers beside the
/// frozen one; a selector that falls back to its own default could load an unverified one. Any
/// other bound tier name (a non-quant variant) selects nothing.
pub fn portable_bound_tier_bits(advanced: &JsonObject) -> Option<i64> {
    match advanced
        .get("__portableBoundTier")
        .and_then(Value::as_str)?
    {
        "q4" => Some(4),
        "q8" => Some(8),
        "bf16" => Some(0),
        _ => None,
    }
}

/// Classify `advanced.mlxQuantize` for a tier selector. Only an ABSENT key takes the verified
/// binding's frozen tier; a present key (an integer, JSON `null`, or a malformed value) is reported
/// as it is, so each selector keeps its own handling of it.
pub fn quant_selection(advanced: &JsonObject) -> QuantSelection<'_> {
    match advanced.get("mlxQuantize") {
        Some(Value::Null) => QuantSelection::Null,
        Some(value) => match json_integer(value) {
            Some(bits) => QuantSelection::Explicit(bits),
            None => QuantSelection::Malformed(value),
        },
        None => match portable_bound_tier_bits(advanced) {
            Some(bits) => QuantSelection::Bound(bits),
            None => QuantSelection::Unselected,
        },
    }
}

/// The quantization bits a job runs: the request's explicit `mlxQuantize`, else (key absent) the
/// tier its verified portable binding froze ([`quant_selection`]). A `null` or malformed explicit
/// value is `None`, as before the binding existed, and is never replaced by the bound tier.
pub fn request_quant_bits(advanced: &JsonObject) -> Option<i64> {
    match quant_selection(advanced) {
        QuantSelection::Explicit(bits) | QuantSelection::Bound(bits) => Some(bits),
        QuantSelection::Unselected | QuantSelection::Null | QuantSelection::Malformed(_) => None,
    }
}

pub fn optional_i64(payload: &JsonObject, key: &str) -> Option<i64> {
    payload.get(key).and_then(json_integer)
}

/// A JSON integer or trimmed integer string. Never truncates floating-point numbers.
pub fn json_integer(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// The `safe_int` READ with no default applied: a JSON number or a numeric string, `None` when
/// absent or unparseable. Split out of [`clamped_u32`] so a caller whose default is not a constant
/// can tell "the caller named nothing" from "the caller named a value" — video's fps resolves an
/// omission from the model's declared `defaults.fps`, which [`clamped_u32`]'s eager `unwrap_or`
/// collapses (sc-12347).
pub fn parse_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .and_then(|value| u32::try_from(value).ok())
}

/// Parse an int (JSON number or numeric string), clamp to `[min, max]`, default when
/// absent/unparseable — the `safe_int` contract. Takes the raw `Option<&Value>` so both
/// the `payload.get(key)` (image) and pre-fetched `payload.get("width")` (video) call
/// sites share one primitive.
pub fn clamped_u32(value: Option<&Value>, default: u32, min: u32, max: u32) -> u32 {
    parse_u32(value).unwrap_or(default).clamp(min, max)
}

/// A resolved manifest entry's `defaults.resolution` (`"848x480"`) as `(width, height)`, or `None`
/// when the model declares none or declares one the caller's `[min, max]` cannot honor.
///
/// Shared by both lanes because the *parse* is generic while the honorable range is not: video
/// clamps dimensions to 256..=1920 and images to 256..=4096, so image models legitimately declare
/// 2048x2048 (the sensenova U1 family) — a value video's bound would reject. Passing the bounds in
/// keeps one parser without pretending the two lanes share a geometry envelope.
///
/// Never-invent-from-a-typo posture, like every other manifest reader: anything that is not
/// `W x H` with both sides inside `[min, max]` yields `None`, so the caller falls back to its own
/// blanket rather than emitting a degenerate size. The value is deliberately NOT snapped to any
/// stride here — each lane's normalization floors declared and blanket geometry alike (sc-12400).
pub fn declared_resolution(
    model_manifest_entry: &JsonObject,
    min: u32,
    max: u32,
) -> Option<(u32, u32)> {
    let (width, height) = model_manifest_entry
        .get("defaults")
        .and_then(Value::as_object)
        .and_then(|defaults| defaults.get("resolution"))
        .and_then(Value::as_str)?
        .split_once(['x', 'X'])?;
    let honorable = |raw: &str| {
        raw.trim()
            .parse::<u32>()
            .ok()
            .filter(|side| (min..=max).contains(side))
    };
    Some((honorable(width)?, honorable(height)?))
}

/// Collect a JSON int array (numbers or numeric strings), dropping non-numeric entries.
/// Absent / non-array → empty.
pub fn int_array(payload: &JsonObject, key: &str) -> Vec<i64> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| {
                    value
                        .as_i64()
                        .or_else(|| value.as_str()?.trim().parse().ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Collect a JSON string array into trimmed, non-empty owned strings (blanks and
/// non-strings are dropped). Absent / non-array / all-blank → empty.
pub fn string_list(payload: &JsonObject, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Clone a JSON array key verbatim; absent / non-array → empty.
pub fn array_or_empty(payload: &JsonObject, key: &str) -> Vec<Value> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Clone a JSON object key verbatim; absent / non-object → empty.
pub fn object_or_empty(payload: &JsonObject, key: &str) -> JsonObject {
    payload
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// A byte count from a JSON unsigned integer or an untrimmed integer string.
/// Floats, negative values, surrounding whitespace and overflow are rejected.
pub fn json_size_to_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}
