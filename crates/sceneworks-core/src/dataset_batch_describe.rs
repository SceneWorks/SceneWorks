//! Shared contract of the `dataset_batch_describe` job (SW-7082): the values the API validates a
//! request against and the worker describes with, kept in one place so the two cannot drift.

/// The Vision Captioner's repository (catalog id `vision_caption_qwen3vl_8b`). The only `model`
/// value the job accepts today.
pub const VISION_CAPTIONER_REPO: &str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";

/// The six description fields every described image carries, in output-contract order.
pub const DESCRIBE_FIELDS: [&str; 6] = [
    "description",
    "style",
    "camera",
    "composition",
    "lighting",
    "environment",
];

/// Longest describing instruction a request may carry, in characters. The default instruction is
/// about 5,500 characters; this leaves room for a much longer edit without letting a request push
/// an unbounded system prompt at the model.
pub const MAX_DESCRIBE_PROMPT_CHARS: usize = 32_000;
