//! Optional content exclusions shared by prompt refinement and image description.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PromptExclusion {
    IdentifyingCharacteristics,
    ClothingAndAccessories,
    ColorMode,
    Style,
}

/// Compact instruction for providers such as JoyCaption with a 4,000-character input limit.
pub fn compact_exclusion_instruction(exclusions: &[PromptExclusion]) -> String {
    let selected: Vec<_> = [
        (
            PromptExclusion::IdentifyingCharacteristics,
            "identifying traits (hair, skin, tattoos)",
        ),
        (
            PromptExclusion::ClothingAndAccessories,
            "all clothing/accessories",
        ),
        (PromptExclusion::ColorMode, "color/B&W mode"),
        (PromptExclusion::Style, "medium/artistic style"),
    ]
    .into_iter()
    .filter_map(|(key, rule)| exclusions.contains(&key).then_some(rule))
    .collect();
    if selected.is_empty() {
        return String::new();
    }
    format!(
        "Omit throughout, overriding prior content rules: {}. Keep output format; use empty fields if excluded. Never infer nudity or state omissions.",
        selected.join("; ")
    )
}

/// Explicit content rules for general language/vision models. Repeat these on the user
/// turn too: real-image runs showed the 8B captioner ignoring a short system-only suffix.
pub fn exclusion_instruction(exclusions: &[PromptExclusion]) -> String {
    let rules: Vec<_> = [
        (PromptExclusion::IdentifyingCharacteristics, "Do not mention ANY identifying or physical characteristics of a person: no hair color, length or style, facial hair, skin color or tone, ethnicity, eye color, age, build, tattoos, piercings or makeup. Delete these details from an original prompt instead of paraphrasing them. Even uncolored references to hair or skin are excluded. Keep their pose, gaze and expression; use neutral nouns or the existing person labels."),
        (PromptExclusion::ClothingAndAccessories, "Do not mention or describe ANY clothing or accessories at all, in any sentence or field. Omit garments, underwear, footwear, jewelry, necklaces, hats, bags, straps, and worn or carried equipment, including saddles, stirrups and harnesses and their colors, materials and fit. Also omit equipment draped across or resting on a person. Remove these details even from pose, lighting highlights and reflections. Describe body positions without referring to what is worn. Do not infer nudity."),
        (PromptExclusion::ColorMode, "Do not say or imply whether the image is color or black-and-white. Do not mention monochrome, grayscale, sepia, black-and-white palettes, color film or the absence of color. Object colors may be described unless another exclusion forbids them."),
        (PromptExclusion::Style, "Do not mention the medium or artistic style. Do not use photograph, photographic, illustration, painting, drawing, render, sketch, cinematic or editorial style, film look, studio aesthetic or an artist/style attribution. If the format has a style field, return \"style\": \"\"; do not move its content to other fields. Describe the visible scene, pose, framing and lighting directly."),
    ].into_iter().filter_map(|(key, rule)| exclusions.contains(&key).then_some(rule)).collect();
    if rules.is_empty() {
        return String::new();
    }
    format!("MANDATORY CONTENT EXCLUSIONS — these override all earlier requests for content, including the model guide. Apply them throughout every field, including pose, camera, lighting and style.\n{}\nKeep the required output format. Leave a field empty when its content is excluded. Do not announce exclusions, quote these rules or invent replacement details. Check the entire answer for forbidden content before returning it.", rules.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusions_are_optional_independent_and_bounded() {
        assert!(exclusion_instruction(&[]).is_empty());
        let all = [
            PromptExclusion::IdentifyingCharacteristics,
            PromptExclusion::ClothingAndAccessories,
            PromptExclusion::ColorMode,
            PromptExclusion::Style,
        ];
        for (key, expected) in all.into_iter().zip([
            "hair color",
            "ANY clothing or accessories",
            "color or black-and-white",
            "medium or artistic style",
        ]) {
            let instruction = exclusion_instruction(&[key]);
            assert!(instruction.contains(expected));
            assert_eq!(instruction, exclusion_instruction(&[key, key]));
        }
        assert!(compact_exclusion_instruction(&all).chars().count() <= 253);
        assert!(serde_json::from_str::<Vec<PromptExclusion>>(r#"["unknown"]"#).is_err());
    }

    #[test]
    fn all_option_combinations_fit_joycaptions_actual_batch_contract() {
        let asset = include_str!("../../sceneworks-worker/src/image_batch_describe_v1.txt")
            .replace("\r\n", "\n");
        let section = |name: &str| {
            asset
                .split_once(&format!("[{name}]\n"))
                .unwrap()
                .1
                .split("\n[")
                .next()
                .unwrap()
                .trim()
        };
        let all = [
            PromptExclusion::IdentifyingCharacteristics,
            PromptExclusion::ClothingAndAccessories,
            PromptExclusion::ColorMode,
            PromptExclusion::Style,
        ];
        for mask in 1..16 {
            let selected: Vec<_> = all
                .iter()
                .enumerate()
                .filter_map(|(index, key)| (mask & (1 << index) != 0).then_some(*key))
                .collect();
            let user = compact_exclusion_instruction(&selected);
            let overhead = section("JOYCAPTION_OUTPUT").chars().count() + user.chars().count() + 4;
            assert!(3000 + overhead <= 4000);
            assert!(section("JOYCAPTION_SYSTEM").chars().count() + overhead <= 4000);
        }
    }
}
