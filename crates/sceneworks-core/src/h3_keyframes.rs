//! The existing H3 FL2VA task accepts a first frame, a last frame, or both.
//! Other model families keep their source-image requirement.
pub fn accepts_last_only_image_to_video(model: &str, mode: &str, has_last: bool) -> bool {
    model == "minimax_h3" && mode == "image_to_video" && has_last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_last_frame_is_an_image_to_video_input() {
        assert!(accepts_last_only_image_to_video(
            "minimax_h3",
            "image_to_video",
            true
        ));
        assert!(!accepts_last_only_image_to_video(
            "minimax_h3",
            "image_to_video",
            false
        ));
    }

    #[test]
    fn other_partitions_families_and_modes_keep_their_input_rules() {
        for model in ["minimax_h3_ref", "ltx_2_3", "wan_2_2_i2v", "unknown"] {
            assert!(!accepts_last_only_image_to_video(
                model,
                "image_to_video",
                true
            ));
        }
        for mode in ["text_to_video", "first_last_frame", "reference_to_video"] {
            assert!(!accepts_last_only_image_to_video("minimax_h3", mode, true));
        }
    }
}
