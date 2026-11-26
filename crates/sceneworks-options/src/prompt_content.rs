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

/// Compact enough to replace the batch user turn even at JoyCaption's maximum instruction size.
/// Content rules yield to these exclusions; the caller's output format is preserved.
pub fn exclusion_instruction(exclusions: &[PromptExclusion]) -> String {
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
            "hair, skin, tattoos",
            "all clothing/accessories",
            "color/B&W mode",
            "medium/artistic style",
        ]) {
            let instruction = exclusion_instruction(&[key]);
            assert!(instruction.contains(expected));
            assert_eq!(instruction, exclusion_instruction(&[key, key]));
        }
        assert!(exclusion_instruction(&all).chars().count() <= 253);
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
            let user = exclusion_instruction(&selected);
            let overhead = section("JOYCAPTION_OUTPUT").chars().count() + user.chars().count() + 4;
            assert!(3000 + overhead <= 4000);
            assert!(section("JOYCAPTION_SYSTEM").chars().count() + overhead <= 4000);
        }
    }
}
