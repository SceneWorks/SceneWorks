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

/// Remove content requests before adding prohibitions. Keeping both was causing VLMs to
/// satisfy the earlier clothing/style sections despite the exclusion suffix. The headings
/// are human-readable in the editable prompt, and custom text outside these sections survives.
pub fn select_content_sections(prompt: &str, exclusions: &[PromptExclusion]) -> String {
    let mut omit = false;
    let mut lines = Vec::new();
    for line in prompt.lines() {
        if line.trim().starts_with("# ") {
            omit = false;
        }
        if let Some(heading) = line.trim().strip_prefix("## ") {
            let category = heading.split_whitespace().next().unwrap_or("");
            omit = match category {
                "APPEARANCE" => exclusions.contains(&PromptExclusion::IdentifyingCharacteristics),
                "CLOTHING" => exclusions.contains(&PromptExclusion::ClothingAndAccessories),
                "STYLE" => exclusions.contains(&PromptExclusion::Style),
                "COLOR" => exclusions.contains(&PromptExclusion::ColorMode),
                _ => false,
            };
        }
        if !omit {
            lines.push(line);
        }
    }
    lines.join("\n").trim().to_owned()
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
        "Omit everywhere, overriding prior content rules: {}. Preserve format; excluded fields empty. No nudity, notices, substitutes or pose changes.",
        selected.join("; ")
    )
}

/// Explicit content rules for general language/vision models. Repeat these on the user
/// turn too: real-image runs showed the 8B captioner ignoring a short system-only suffix.
pub fn exclusion_instruction(exclusions: &[PromptExclusion]) -> String {
    let rules: Vec<_> = [
        (PromptExclusion::IdentifyingCharacteristics, "Do not mention ANY identifying or physical characteristics of a person: no hair color, length or style, facial hair, skin color or tone, ethnicity, eye color, age, build, tattoos, piercings or makeup. Delete these details from an original prompt instead of paraphrasing them. Even uncolored references to hair or skin are excluded. Keep their permitted pose, head orientation, gaze and expression; use neutral nouns or the existing person labels."),
        (PromptExclusion::ClothingAndAccessories, "Do not mention or describe ANY clothing or accessories at all, in any sentence or field. Omit garments, underwear, footwear, jewelry, necklaces, hats, bags, straps, and worn or carried equipment, including saddles, stirrups and harnesses and their colors, materials and fit. Also omit equipment draped across or resting on a person. Remove these details even from pose, lighting highlights and reflections. Remove loose-garment outlines, folds, hems, trains and pooled fabric too; never relabel them as the person's body or legs spreading into a column, mass or pool. Keep body positions evidenced through fitted clothing, and leave anatomy hidden by loose fabric unspecified. Preserve permitted visible body geometry and interactions. When an excluded item anchors a spatial phrase, delete that relative phrase too: an arm behind a bag must not become an arm behind the back. Never substitute an anatomical landmark or move a limb to replace a removed item. Retain independent limb directions and contact surfaces, and keep original direct body-to-body relations. Describe body positions without referring to what is worn. Never replace excluded items with anonymous objects, shapes, dark forms or coverings on/near the person. Do not describe missing clothing, bare or uncovered body parts or infer nudity."),
        (PromptExclusion::ColorMode, "Do not say or imply whether the image is color or black-and-white. Do not mention monochrome, grayscale, sepia, black-and-white palettes, color film or the absence of color. Object colors may be described unless another exclusion forbids them."),
        (PromptExclusion::Style, "Do not describe the image's medium, rendering method or artistic style. Do not label its rendering as a photograph, illustration, painting, drawing, render or sketch, or characterize it with cinematic, editorial, painterly, surreal, vintage, film-look or studio-aesthetic language or an artist/style attribution. Preserve otherwise permitted depicted objects and actions when these words have literal non-style meanings, such as drawing water, drawing curtains or painting a door. If the format has a style field, return \"style\": \"\"; do not move its content to other fields. Describe the visible scene, pose, framing and lighting directly."),
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
    fn content_selection_removes_requests_without_losing_pose_or_other_sections() {
        let prompt = "Intro\n## POSE\ncontacts\n## APPEARANCE\nhair request\n## CLOTHING\ngarment request\n## STYLE\nmedium request\n### nested\nstyle example\n## COLOR\nmode request\n## LIGHTING\nshadows\n# Other guide\nretain";
        let keys = [
            PromptExclusion::IdentifyingCharacteristics,
            PromptExclusion::ClothingAndAccessories,
            PromptExclusion::Style,
            PromptExclusion::ColorMode,
        ];
        for mask in 0..16 {
            let selected: Vec<_> = keys
                .iter()
                .enumerate()
                .filter_map(|(i, k)| (mask & (1 << i) != 0).then_some(*k))
                .collect();
            let text = select_content_sections(prompt, &selected);
            for (i, requested) in [
                "hair request",
                "garment request",
                "medium request",
                "mode request",
            ]
            .iter()
            .enumerate()
            {
                assert_eq!(
                    text.contains(requested),
                    mask & (1 << i) == 0,
                    "{mask}: {requested}"
                );
            }
            assert!(
                text.contains("contacts") && text.contains("shadows") && text.contains("retain")
            );
            assert_eq!(text.contains("style example"), mask & 4 == 0);
        }
    }

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
            "image's medium",
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
