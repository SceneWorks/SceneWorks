//! Shared contract of the `dataset_batch_describe` job (SW-7082, SW-7083): the values the API
//! validates a request against, the scheduler routes it by, and the worker describes with, kept in
//! one place so the three cannot drift. With a "Fit to" target the worker also condenses each
//! image's seven fields into one prompt that fits the target image model's prompt token limit; the
//! bounds of that step live here too.

use serde_json::{Map, Value};

/// The Vision Captioner's repository (catalog id `vision_caption_qwen3vl_8b`). The default `model`.
pub const VISION_CAPTIONER_REPO: &str = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";

/// JoyCaption's repository (catalog id `joycaption_beta_one`, the `training_caption` captioner).
pub const JOYCAPTION_REPO: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";

/// The Qwen3.8 27B describer's upstream repository (SW-F365): an abliterated Qwen3.8-27B that
/// keeps its vision encoder. The request's `model` value for this describer on every tier. Its
/// full-precision tier loads from this repository; the quantised tiers load from the re-host.
pub const QWEN38_27B_DESCRIBER_REPO: &str = "huihui-ai/Huihui-Qwen3.8-27B-abliterated";

/// The Qwen3.8 27B describer's catalog id: the entry whose download rows are its tiers and whose
/// `mlx` / `candle` blocks carry each tier's memory floor.
pub const QWEN38_27B_DESCRIBER_MODEL_ID: &str = "vision_describe_qwen3_8_27b";

/// The repositories the Qwen3.8 27B describer's quantised tiers are served from, one per engine.
/// Each holds a `q4/` and a `q8/` snapshot folder quantised ahead of time, so a quantised tier never
/// downloads or load-quantises the 57 GB full-precision files.
pub const QWEN38_27B_DESCRIBER_MLX_REPO: &str = "SceneWorks/qwen3.8-27b-abliterated-mlx";
pub const QWEN38_27B_DESCRIBER_CANDLE_REPO: &str = "SceneWorks/qwen3.8-27b-abliterated-candle";

/// The Qwen3.8 27B describer's tiers, smallest first. `bf16` is full precision.
pub const QWEN38_27B_DESCRIBER_TIERS: [&str; 3] = ["q4", "q8", "bf16"];

/// The tier a Qwen3.8 27B request without a `tier` describes with.
pub const QWEN38_27B_DESCRIBER_DEFAULT_TIER: &str = "q4";

/// The file selectors the shipped catalog row of the Qwen3.8 27B `tier` served from `repo` names:
/// the exact files of a quantised tier's folder, or full precision's named files plus its
/// `model-*.safetensors` shard glob. Empty when the catalog has no such row. The API's install
/// state, the job's tier check and the worker all require every one of these to be present.
pub fn qwen38_tier_files(repo: &str, tier: &str) -> Vec<String> {
    crate::builtin_manifests::builtin_model_entry(QWEN38_27B_DESCRIBER_MODEL_ID)
        .and_then(|entry| entry.get("downloads")?.as_array().cloned())
        .into_iter()
        .flatten()
        .find(|row| {
            row.get("repo").and_then(Value::as_str) == Some(repo)
                && row.get("variant").and_then(Value::as_str) == Some(tier)
        })
        .and_then(|row| row.get("files")?.as_array().cloned())
        .into_iter()
        .flatten()
        .filter_map(|file| file.as_str().map(str::to_owned))
        .collect()
}

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
    if model.map(str::trim) == Some(crate::loop_utility::MODEL_REPO) {
        return crate::loop_utility::tier(tier).map(Some);
    }
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

/// The seven description fields every described image carries, in output-contract order. Each is
/// stored as text: the worker flattens the Vision Captioner's structured `pose` (named slots per
/// person) into one line per person. A result stored before `pose` existed has only the other six.
pub const DESCRIBE_FIELDS: [&str; 7] = [
    "description",
    "pose",
    "style",
    "camera",
    "composition",
    "lighting",
    "environment",
];

/// Longest text kept in, or accepted for, one description field. The worker caps each parsed
/// field at this; a condense-only request (whose rows the user may have edited) is refused above it.
pub const MAX_DESCRIBE_FIELD_CHARS: usize = 16_000;

/// Characters the batch block's `[W:H]` line and its newline take at most (`[21:9]` plus `\n`,
/// with one to spare). The condensed prompt shares the route's character limit with that line.
pub const RATIO_LINE_CHARS: usize = 8;

/// Characters the batch formatter adds around each person label when it wraps it as a `{{label}}`
/// batch variable.
pub const LABEL_WRAP_CHARS: usize = 4;

/// The share of the target model's limit a condensed prompt must stay within when its size is
/// only estimated (counted with the describer's own tokenizer, not the target's): the margin
/// covers the difference between the two tokenizers.
pub const CONDENSE_ESTIMATE_PERCENT: usize = 85;

/// How much stricter the one retry's word target is than the first attempt's, as a percentage.
pub const CONDENSE_RETRY_PERCENT: usize = 85;

/// The room a condensed prompt keeps for each person-label mention (`woman_1`): Image Studio
/// replaces each `{{label}}` with the user's own text before it sends the prompt, so a prompt
/// that only just fits with bare labels would be refused once they are filled in. A prompt fits
/// only when `tokens + mentions × VARIABLE_VALUE_TOKEN_RESERVE <= limit` and `characters +
/// mentions × VARIABLE_VALUE_CHAR_RESERVE <=` the character bound: each filled-in value may be up
/// to about 8 tokens or 40 characters long ("a tall woman in a red coat").
pub const VARIABLE_VALUE_TOKEN_RESERVE: usize = 8;
/// The character room kept per label mention; see [`VARIABLE_VALUE_TOKEN_RESERVE`].
pub const VARIABLE_VALUE_CHAR_RESERVE: usize = 40;

/// The smallest prompt limit a "Fit to" target may declare. The condensing instruction must keep
/// every limb of every person; below about 512 tokens (about 340 words with the variable reserve)
/// that cannot be met for an ordinary image, so most rows would end over the limit.
pub const MIN_FIT_TO_PROMPT_TOKENS: usize = 512;

/// The most characters a condense-only row may carry across its seven fields (each field is
/// also held to [`MAX_DESCRIBE_FIELD_CHARS`]).
pub const MAX_CONDENSE_ROW_CHARS: usize = 32_000;

/// Whether the describer `model` names runs the "Fit to" condensing step: every Vision
/// Captioner-class describer has a text-only generation on its loaded model; JoyCaption (a
/// one-shot captioner) has none. The one place this property is decided.
pub fn describer_condenses(model: Option<&str>) -> bool {
    !is_joycaption_model(model)
}

/// The prompt context, in text-encoder tokens, a catalog entry declares for its image model
/// (`image.maxPromptTokens`); `None` when it declares none. For Krea 2 it is the text-to-image
/// limit (its image edit encodes with a separate, far larger one). The "Fit to" targets are the
/// entries for which this is at least [`MIN_FIT_TO_PROMPT_TOKENS`].
pub fn prompt_token_limit(entry: &Value) -> Option<usize> {
    entry
        .get("image")?
        .get("maxPromptTokens")?
        .as_u64()
        .filter(|tokens| *tokens > 0)
        .and_then(|tokens| usize::try_from(tokens).ok())
}

/// The token count a condensed prompt must not exceed for a model with `limit`: the limit itself
/// when the count is exact (measured with the target model's own tokenizer, template included),
/// [`CONDENSE_ESTIMATE_PERCENT`] of it when the count is an estimate.
pub fn condense_token_bound(limit: usize, exact: bool) -> usize {
    if exact {
        limit
    } else {
        limit * CONDENSE_ESTIMATE_PERCENT / 100
    }
}

/// The character count a condensed prompt's batch block must not exceed: the route's prompt
/// limit less the `[W:H]` line.
pub fn condense_char_bound() -> usize {
    crate::MAX_PROMPT_CHARS - RATIO_LINE_CHARS
}

/// The word count the condensing rewrite is told to stay under for a token and character bound
/// and the expected number of label `mentions` (whose variable-value reserve comes off both
/// bounds first). English prose runs about 4 words to 3 tokens and about 6 characters a word with
/// its space, so the target is the tighter of the two bounds expressed in words, less a 10% margin
/// so a reply that overshoots its word count a little still fits.
pub fn condense_word_target(token_bound: usize, char_bound: usize, mentions: usize) -> usize {
    let tokens = token_bound.saturating_sub(mentions * VARIABLE_VALUE_TOKEN_RESERVE);
    let chars = char_bound.saturating_sub(mentions * VARIABLE_VALUE_CHAR_RESERVE);
    (tokens * 3 / 4).min(chars / 6) * 9 / 10
}

/// Longest describing instruction a Vision Captioner request may carry, in characters. The default
/// instruction is about 9,500 characters; this leaves room for a much longer edit without letting
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
            crate::builtin_manifests::builtin_model_entry(QWEN38_27B_DESCRIBER_MODEL_ID)
                .expect("the 27B describer is a builtin")
                .clone(),
        );
        let selected = |platform: &str, variant: Option<&str>| {
            let closure = selected_model_artifact_closure(&entry, platform, variant);
            let downloads = closure["downloads"].as_array().expect("downloads").clone();
            assert_eq!(downloads.len(), 1, "{platform} {variant:?}: {downloads:?}");
            (
                downloads[0]["repo"].as_str().expect("repo").to_owned(),
                downloads[0]["files"]
                    .as_array()
                    .expect("files")
                    .iter()
                    .map(|file| file.as_str().expect("file").to_owned())
                    .collect::<Vec<_>>(),
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
                // Every file of the tier's folder, named one by one (never a `<tier>/*` glob, which
                // one file satisfies), the list every install check requires.
                let expected = [
                    "config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                ]
                .map(|name| format!("{tier}/{name}"));
                assert_eq!(files, expected, "{platform} {tier}");
                assert_eq!(files, qwen38_tier_files(rehost, tier));
            }
            assert_eq!(
                selected(platform, Some("bf16")).0,
                QWEN38_27B_DESCRIBER_REPO,
                "{platform}"
            );
            // A request without a variant gets the default tier's row, the tier the job
            // defaults to.
            assert!(
                selected(platform, None)
                    .1
                    .iter()
                    .all(|file| file.starts_with(&format!("{QWEN38_27B_DESCRIBER_DEFAULT_TIER}/"))),
                "{platform}"
            );
            // Every tier in the request vocabulary has a row.
            for tier in QWEN38_27B_DESCRIBER_TIERS {
                let closure = selected_model_artifact_closure(&entry, platform, Some(tier));
                assert_eq!(closure["downloads"][0]["variant"], tier, "{platform}");
            }
        }
    }

    #[test]
    fn a_fit_to_limit_is_the_declared_prompt_token_count_only() {
        use serde_json::json;
        let declared = json!({ "image": { "maxPromptTokens": 1024 } });
        assert_eq!(prompt_token_limit(&declared), Some(1024));
        for undeclared in [
            json!({}),
            json!({ "image": {} }),
            json!({ "image": { "maxPromptTokens": 0 } }),
            json!({ "image": { "maxPromptTokens": "1024" } }),
        ] {
            assert_eq!(prompt_token_limit(&undeclared), None, "{undeclared}");
        }
    }

    #[test]
    fn an_estimated_count_keeps_a_margin_and_the_word_target_obeys_both_bounds() {
        assert_eq!(condense_token_bound(1024, true), 1024);
        assert_eq!(condense_token_bound(1024, false), 870);
        assert_eq!(condense_char_bound(), 3_992);
        // Krea 2: the 4,000-character route limit binds before 1,024 tokens of English.
        assert_eq!(condense_word_target(1024, condense_char_bound(), 0), 598);
        // Eight label mentions reserve 320 characters (and 64 tokens) for their filled-in values.
        assert_eq!(condense_word_target(1024, condense_char_bound(), 8), 550);
        // A small token context binds before the character limit.
        assert_eq!(condense_word_target(128, condense_char_bound(), 0), 86);
        assert!(describer_condenses(Some(VISION_CAPTIONER_REPO)));
        assert!(describer_condenses(None));
        // The Qwen3.8 27B is a Vision Captioner-class describer: it condenses on its own model.
        assert!(describer_condenses(Some(QWEN38_27B_DESCRIBER_REPO)));
        assert!(!describer_condenses(Some(JOYCAPTION_REPO)));
    }

    #[test]
    fn pose_follows_description_and_the_other_six_keep_their_order() {
        assert_eq!(
            DESCRIBE_FIELDS,
            [
                "description",
                "pose",
                "style",
                "camera",
                "composition",
                "lighting",
                "environment"
            ]
        );
    }
}
