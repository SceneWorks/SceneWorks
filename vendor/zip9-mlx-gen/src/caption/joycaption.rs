//! ATTRIBUTION: The caption-type taxonomy, prompt template bodies, and name option below are
//! reproduced from fpgaminer/joycaption (<https://github.com/fpgaminer/joycaption>),
//! `gradio-app/app.py`'s `CAPTION_TYPE_MAP` and `NAME_OPTION`, commit
//! 8445b2e55db7856d522e44ae84e7415fcf3413f6, licensed under the Apache License 2.0
//! (Copyright 2024 fpgaminer@bitcoin-mining.com). The license text ships in SceneWorks' About ->
//! Licenses corpus as the `joycaption-source` component.
//!
//! MODIFIED BY SCENEWORKS (Apache-2.0 section 4(b)): ported from Python to Rust; renamed the
//! prompt-table and name-option symbols; normalized prompt punctuation (em dashes, curly quotes,
//! ellipses, backticks, quoted tag prefixes, and `(if any)`); removed `very long` from the caption
//! lengths; and changed the three Straightforward prompts from requiring watermark, signature, and
//! compression-artifact mentions to forbidding them.
//!
//! JoyCaption caption **product policy**.
//!
//! The JoyCaption model (SigLIP vision tower + LLaVA projector + image splice + Llama-3.1 decode +
//! the model's LLaVA chat-input format) lives in the unified LLM engine
//! ([`mlx-llm`](https://github.com/SceneWorks/mlx-llm)) as the `mlx-joycaption`
//! `core_llm::TextLlm` vision provider. This module keeps only the SceneWorks-side caption product
//! surface — the caption-type/length prompt templates, the advertised capability bounds, and the
//! trigger-word post-processing — that a consumer uses to build the prompt text before calling the
//! engine (epic 7153, sc-7265). It contains no model-specific code.

use crate::caption::{CaptionCapabilities, CaptionOptions};
use crate::gen_core;

pub const JOY_CAPTION_MODEL_ID: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";
pub const JOY_CAPTION_FAMILY: &str = "joycaption";

pub const JOY_NAME_OPTION: &str =
    "If there is a person/character in the image you must refer to them as {name}.";

pub const CAPTION_TYPES: &[&str] = &[
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
];

pub const CAPTION_LENGTHS: &[&str] = &["any", "very short", "short", "medium-length", "long"];

const PROMPT_TEMPLATES: &[(&str, [&str; 3])] = &[
    (
        "Descriptive",
        [
            "Write a detailed description for this image.",
            "Write a detailed description for this image in {word_count} words or less.",
            "Write a {length} detailed description for this image.",
        ],
    ),
    (
        "Descriptive (Casual)",
        [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
    ),
    (
        "Straightforward",
        [
            "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
        ],
    ),
    (
        "Stable Diffusion Prompt",
        [
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
            "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        ],
    ),
    (
        "MidJourney",
        [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
    ),
    (
        "Danbooru tag list",
        [
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
        ],
    ),
    (
        "e621 tag list",
        [
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
        ],
    ),
    (
        "Rule34 tag list",
        [
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
        ],
    ),
    (
        "Booru-like tag list",
        [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
    ),
    (
        "Art Critic",
        [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
    ),
    (
        "Product Listing",
        [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
    ),
    (
        "Social Media Post",
        [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    ),
];

pub fn capabilities() -> CaptionCapabilities {
    CaptionCapabilities {
        caption_types: CAPTION_TYPES.to_vec(),
        caption_lengths: CAPTION_LENGTHS.to_vec(),
        supports_custom_prompt: true,
        supports_low_vram: true,
        min_image_size: 1,
        max_image_size: 8192,
        max_prompt_chars: 4000,
        max_name_chars: 120,
        max_extra_options: 16,
        max_extra_option_chars: 500,
        max_trigger_words: 32,
        max_trigger_word_chars: 120,
        max_new_tokens: 1024,
        mac_only: true,
    }
}

pub fn build_prompt(options: &CaptionOptions) -> String {
    let custom = options.custom_prompt.trim();
    if !custom.is_empty() {
        return custom.to_owned();
    }

    let caption_length = options.caption_length.as_str();
    let template_index = if caption_length == "any" {
        0
    } else if !caption_length.is_empty() && caption_length.chars().all(|c| c.is_ascii_digit()) {
        1
    } else {
        2
    };
    let mut prompt = templates_for(&options.caption_type)[template_index].to_owned();
    if !options.extra_options.is_empty() {
        prompt.push(' ');
        prompt.push_str(&options.extra_options.join(" "));
    }
    prompt
        .replace("{name}", name_or_placeholder(options))
        .replace("{length}", caption_length)
        .replace("{word_count}", caption_length)
}

/// Compatibility export for backend callers. The policy itself is backend-neutral and lives in
/// `gen-core`, shared with the SceneWorks worker.
pub use gen_core::apply_caption_trigger_words as apply_trigger_words;

fn templates_for(caption_type: &str) -> &'static [&'static str; 3] {
    PROMPT_TEMPLATES
        .iter()
        .find(|(kind, _)| *kind == caption_type)
        .map(|(_, templates)| templates)
        .unwrap_or(&PROMPT_TEMPLATES[0].1)
}

fn name_or_placeholder(options: &CaptionOptions) -> &str {
    if options.name_input.is_empty() {
        "{NAME}"
    } else {
        &options.name_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(kind: &str, length: &str) -> CaptionOptions {
        CaptionOptions {
            caption_type: kind.to_owned(),
            caption_length: length.to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn prompt_selection_matrix_matches_upstream_and_lane_policy() {
        assert_eq!(
            build_prompt(&CaptionOptions::default()),
            "Write a long detailed description for this image."
        );
        for (kind, length, expected) in [
            (
                "Descriptive",
                "any",
                "Write a detailed description for this image.",
            ),
            (
                "Descriptive",
                "85",
                "Write a detailed description for this image in 85 words or less.",
            ),
            (
                "Descriptive",
                "short",
                "Write a short detailed description for this image.",
            ),
            (
                "Descriptive",
                "",
                "Write a  detailed description for this image.",
            ),
            (
                "Not a real type",
                "short",
                "Write a short detailed description for this image.",
            ),
        ] {
            assert_eq!(build_prompt(&options(kind, length)), expected);
        }
    }

    #[test]
    fn prompt_appends_options_and_replaces_name() {
        let prompt = build_prompt(&CaptionOptions {
            caption_type: "Descriptive".to_owned(),
            caption_length: "long".to_owned(),
            extra_options: vec![
                JOY_NAME_OPTION.to_owned(),
                "Mention the setting.".to_owned(),
            ],
            name_input: "Mika".to_owned(),
            ..Default::default()
        });
        assert_eq!(
            prompt,
            "Write a long detailed description for this image. If there is a person/character in the image you must refer to them as Mika. Mention the setting."
        );
    }

    #[test]
    fn custom_prompt_overrides_template() {
        let prompt = build_prompt(&CaptionOptions {
            custom_prompt: "  Describe only the outfit.  ".to_owned(),
            ..Default::default()
        });
        assert_eq!(prompt, "Describe only the outfit.");
    }

    #[test]
    fn trigger_words_are_prepended_only_when_missing() {
        let trigger_words = vec!["mika_token".to_owned(), "hat".to_owned()];
        assert_eq!(
            apply_trigger_words("A portrait of Mika wearing a hat.", &trigger_words),
            "mika_token, A portrait of Mika wearing a hat."
        );
        assert_eq!(
            apply_trigger_words("   ", &trigger_words),
            "mika_token, hat"
        );
    }
}
