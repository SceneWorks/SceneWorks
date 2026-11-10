//! Shared contract of the `dataset_batch_describe` job (SW-7082, SW-7083): the values the API
//! validates a request against, the scheduler routes it by, and the worker describes with, kept in
//! one place so the three cannot drift.

use serde_json::{Map, Value};

/// The Vision Captioner's repository (catalog id `vision_caption_qwen3vl_8b`). The default `model`.
pub const VISION_CAPTIONER_REPO: &str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";

/// JoyCaption's repository (catalog id `joycaption_beta_one`, the `training_caption` captioner).
pub const JOYCAPTION_REPO: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";

/// The six description fields every described image carries, in output-contract order.
pub const DESCRIBE_FIELDS: [&str; 6] = [
    "description",
    "style",
    "camera",
    "composition",
    "lighting",
    "environment",
];

/// Longest describing instruction a Vision Captioner request may carry, in characters. The default
/// instruction is about 4,800 characters; this leaves room for a much longer edit without letting
/// a request push an unbounded system prompt at the model.
pub const MAX_DESCRIBE_PROMPT_CHARS: usize = 32_000;

/// Longest describing instruction a JoyCaption request may carry, in characters. JoyCaption takes
/// one user prompt and its providers (mlx-gen and candle-gen) refuse a prompt over 4,000
/// characters. The worker sends the instruction, the fixed output contract and the user line in
/// that one prompt, which leaves about 3,000 characters for the instruction.
pub const MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS: usize = 3_000;

/// Whether `model` (the request's `model` value) selects JoyCaption. Absent, blank or any other
/// value is the Vision Captioner (or is refused before it is queued).
pub fn is_joycaption_model(model: Option<&str>) -> bool {
    model.map(str::trim) == Some(JOYCAPTION_REPO)
}

/// Whether a queued job payload describes with JoyCaption. A JoyCaption job routes by the
/// `training_caption` capability (the worker that links the JoyCaption captioner), the Vision
/// Captioner job by `prompt_refine`.
pub fn payload_uses_joycaption(payload: &Map<String, Value>) -> bool {
    is_joycaption_model(payload.get("model").and_then(Value::as_str))
}

/// The instruction length limit for `model`.
pub fn max_describe_prompt_chars(model: Option<&str>) -> usize {
    if is_joycaption_model(model) {
        MAX_JOYCAPTION_DESCRIBE_PROMPT_CHARS
    } else {
        MAX_DESCRIBE_PROMPT_CHARS
    }
}
