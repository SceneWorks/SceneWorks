//! The existing H3 FL2VA task accepts a first frame, a last frame, or both.
//! Other model families keep their source-image requirement.
pub fn accepts_last_only_image_to_video(model: &str, mode: &str, has_last: bool) -> bool {
    model == "minimax_h3" && mode == "image_to_video" && has_last
}

/// `imported_partition` must come from the server-resolved, verified imported H3 contract,
/// never from request metadata or a model-id/family guess.
pub fn accepts_last_only_resolved_image_to_video(
    model: &str,
    mode: &str,
    has_last: bool,
    imported_partition: Option<&str>,
) -> bool {
    accepts_last_only_image_to_video(model, mode, has_last)
        || (mode == "image_to_video"
            && has_last
            && matches!(imported_partition, Some("fl2va" | "hybrid")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_imported_partitions_keep_last_only_scoped_to_fl2va() {
        for partition in ["fl2va", "hybrid"] {
            assert!(accepts_last_only_resolved_image_to_video(
                "user_checkpoint",
                "image_to_video",
                true,
                Some(partition)
            ));
            assert!(!accepts_last_only_resolved_image_to_video(
                "user_checkpoint",
                "image_to_video",
                false,
                Some(partition)
            ));
            assert!(!accepts_last_only_resolved_image_to_video(
                "user_checkpoint",
                "reference_to_video",
                true,
                Some(partition)
            ));
        }
        for partition in [None, Some("ref2va"), Some("unknown")] {
            assert!(!accepts_last_only_resolved_image_to_video(
                "user_checkpoint",
                "image_to_video",
                true,
                partition
            ));
        }
    }

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
