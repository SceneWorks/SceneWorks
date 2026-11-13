//! Shared contract of the `dataset_batch_describe` job (SW-7082, SW-7083): the values the API
//! validates a request against, the scheduler routes it by, and the worker describes with, kept in
//! one place so the three cannot drift.

use serde_json::{Map, Value};

/// The Vision Captioner's repository (catalog id `vision_caption_qwen3vl_8b`). The default `model`.
pub const VISION_CAPTIONER_REPO: &str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";

/// JoyCaption's repository (catalog id `joycaption_beta_one`, the `training_caption` captioner).
pub const JOYCAPTION_REPO: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";

/// The Qwen3.8 27B describer's upstream repository (SW-F365): an abliterated Qwen3.8-27B that
/// keeps its vision encoder. The request's `model` value for this describer on every tier. Its
/// full-precision tier loads from this repository; the quantised tiers load from the re-host.
pub const QWEN38_27B_DESCRIBER_REPO: &str = "huihui-ai/Huihui-Qwen3.8-27B-abliterated";

/// The repositories the Qwen3.8 27B describer's quantised tiers are served from, one per engine.
/// Each holds a `q4/` and a `q8/` snapshot folder quantised ahead of time, so a quantised tier never
/// downloads or load-quantises the 57 GB full-precision files.
pub const QWEN38_27B_DESCRIBER_MLX_REPO: &str = "SceneWorks/qwen3.8-27b-abliterated-mlx";
pub const QWEN38_27B_DESCRIBER_CANDLE_REPO: &str = "SceneWorks/qwen3.8-27b-abliterated-candle";

/// The Qwen3.8 27B describer's tiers, smallest first. `bf16` is full precision.
pub const QWEN38_27B_DESCRIBER_TIERS: [&str; 3] = ["q4", "q8", "bf16"];

/// The tier a Qwen3.8 27B request without a `tier` describes with.
pub const QWEN38_27B_DESCRIBER_DEFAULT_TIER: &str = "q4";

/// Whether `model` (the request's `model` value) selects the Qwen3.8 27B describer.
pub fn is_qwen38_27b_describer(model: Option<&str>) -> bool {
    model.map(str::trim) == Some(QWEN38_27B_DESCRIBER_REPO)
}

/// The tier a request describes with. Only the Qwen3.8 27B describer has tiers: for it an absent
/// or blank `tier` is the default and an unknown one is refused; for every other model the answer
/// is `None` and naming a tier is refused, so a typo never silently runs a different model size.
pub fn resolve_describe_tier(
    model: Option<&str>,
    tier: Option<&str>,
) -> Result<Option<&'static str>, String> {
    let tier = tier.map(str::trim).filter(|tier| !tier.is_empty());
    if !is_qwen38_27b_describer(model) {
        return match tier {
            None => Ok(None),
            Some(tier) => Err(format!(
                "Describe tier {tier:?} is not available for this model; only the Qwen3.8 27B \
                 describer ({QWEN38_27B_DESCRIBER_REPO}) has tiers."
            )),
        };
    }
    match tier {
        None => Ok(Some(QWEN38_27B_DESCRIBER_DEFAULT_TIER)),
        Some(tier) => QWEN38_27B_DESCRIBER_TIERS
            .iter()
            .copied()
            .find(|known| *known == tier)
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "Unsupported describe tier {tier:?}. Use one of {}.",
                    QWEN38_27B_DESCRIBER_TIERS.join(", ")
                )
            }),
    }
}

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

/// The section of the describe prompt asset (`image_batch_describe_v1.txt`, owned by the worker)
/// that holds `model`'s default editable instruction: `[SYSTEM]` for the Vision Captioner,
/// `[JOYCAPTION_SYSTEM]` (the same rules condensed to JoyCaption's prompt limit) for JoyCaption.
pub fn default_instruction_section(model: Option<&str>) -> &'static str {
    if is_joycaption_model(model) {
        "JOYCAPTION_SYSTEM"
    } else {
        "SYSTEM"
    }
}

/// Body of one `[NAME]` section of a bracketed-marker prompt asset, trimmed; empty when the
/// section is absent. A marker is a bracketed single word alone on a line, and a section runs to
/// the next marker. The one parser the worker (every prompt asset) and the API (the describing
/// prompt it serves) share, so the default the dialog shows is the text the worker describes with.
pub fn prompt_asset_section(source: &str, name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let is_marker = trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.contains(' ');
        if is_marker {
            if capturing {
                break;
            }
            capturing = trimmed[1..trimmed.len() - 1].eq_ignore_ascii_case(name);
            continue;
        }
        if capturing {
            body.push(line);
        }
    }
    body.join("\n").trim().to_owned()
}

/// `model`'s default editable describing instruction, read from the describe prompt asset text.
pub fn default_describe_instruction(asset: &str, model: Option<&str>) -> String {
    prompt_asset_section(asset, default_instruction_section(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_stops_at_the_next_marker() {
        let source = "[A]\none\n[B]\ntwo\n  three  \n[C]\nfour";
        assert_eq!(prompt_asset_section(source, "B"), "two\n  three");
        assert_eq!(prompt_asset_section(source, "b"), "two\n  three");
        assert_eq!(prompt_asset_section(source, "missing"), "");
    }

    #[test]
    fn each_model_reads_its_own_default_section() {
        let asset = "[SYSTEM]\nvision rules\n[JOYCAPTION_SYSTEM]\njoy rules\n[OUTPUT]\n{}";
        assert_eq!(default_describe_instruction(asset, None), "vision rules");
        assert_eq!(
            default_describe_instruction(asset, Some(VISION_CAPTIONER_REPO)),
            "vision rules"
        );
        assert_eq!(
            default_describe_instruction(asset, Some(JOYCAPTION_REPO)),
            "joy rules"
        );
        assert_eq!(max_describe_prompt_chars(Some(JOYCAPTION_REPO)), 3_000);
        assert_eq!(max_describe_prompt_chars(None), 32_000);
    }

    #[test]
    fn only_the_qwen38_27b_describer_has_tiers() {
        let big = Some(QWEN38_27B_DESCRIBER_REPO);
        assert_eq!(resolve_describe_tier(big, None), Ok(Some("q4")));
        assert_eq!(resolve_describe_tier(big, Some("  ")), Ok(Some("q4")));
        for tier in QWEN38_27B_DESCRIBER_TIERS {
            assert_eq!(resolve_describe_tier(big, Some(tier)), Ok(Some(tier)));
        }
        assert!(resolve_describe_tier(big, Some("q2")).is_err());
        assert_eq!(resolve_describe_tier(None, None), Ok(None));
        assert_eq!(resolve_describe_tier(Some(JOYCAPTION_REPO), None), Ok(None));
        assert!(resolve_describe_tier(Some(VISION_CAPTIONER_REPO), Some("q4")).is_err());
        // The 27B is a Vision Captioner-class describer: full prompt, full limit.
        assert_eq!(max_describe_prompt_chars(big), MAX_DESCRIBE_PROMPT_CHARS);
        assert_eq!(default_instruction_section(big), "SYSTEM");
    }

    /// The source a describe job is admitted against is the tier it loads: the catalog row the
    /// model-source seam selects for the job's `variant` is the re-host's tier folder on macOS and
    /// the upstream repository for full precision, never a sibling tier.
    #[test]
    fn each_qwen38_tier_selects_its_own_catalog_download() {
        use crate::model_artifacts::artifact_selection::selected_model_artifact_closure;
        use serde_json::Value;

        let entry = Value::Object(
            crate::builtin_manifests::builtin_model_entry("vision_describe_qwen3_8_27b")
                .expect("the 27B describer is a builtin")
                .clone(),
        );
        let selected = |platform: &str, variant: Option<&str>| {
            let closure = selected_model_artifact_closure(&entry, platform, variant);
            let downloads = closure["downloads"].as_array().expect("downloads").clone();
            assert_eq!(downloads.len(), 1, "{platform} {variant:?}: {downloads:?}");
            (
                downloads[0]["repo"].as_str().expect("repo").to_owned(),
                downloads[0]["files"][0].as_str().expect("file").to_owned(),
            )
        };
        // Each engine stores a quantised tier in its own format, so each platform selects its
        // engine's re-host: the repository the worker on that platform loads from.
        for (platform, rehost) in [
            ("macos", QWEN38_27B_DESCRIBER_MLX_REPO),
            ("windows", QWEN38_27B_DESCRIBER_CANDLE_REPO),
            ("linux", QWEN38_27B_DESCRIBER_CANDLE_REPO),
        ] {
            for tier in ["q4", "q8"] {
                let (repo, files) = selected(platform, Some(tier));
                assert_eq!(repo, rehost, "{platform} {tier}");
                assert_eq!(files, format!("{tier}/*"));
            }
            assert_eq!(
                selected(platform, Some("bf16")).0,
                QWEN38_27B_DESCRIBER_REPO,
                "{platform}"
            );
            // A request without a variant gets the default tier's row, the tier the job
            // defaults to.
            assert_eq!(
                selected(platform, None).1,
                format!("{QWEN38_27B_DESCRIBER_DEFAULT_TIER}/*"),
                "{platform}"
            );
            // Every tier in the request vocabulary has a row.
            for tier in QWEN38_27B_DESCRIBER_TIERS {
                let closure = selected_model_artifact_closure(&entry, platform, Some(tier));
                assert_eq!(closure["downloads"][0]["variant"], tier, "{platform}");
            }
        }
    }
}
