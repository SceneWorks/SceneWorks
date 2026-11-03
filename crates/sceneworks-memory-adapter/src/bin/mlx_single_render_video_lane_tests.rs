use super::*;

/// The catalog plan, one request per MLX row, in the shape every arm reads (`planned.target`).
fn planned_mlx_rows() -> Vec<(String, Value, Value)> {
    let plan: Value = serde_json::from_str(include_str!(
        "../../../../config/memory-calibration-plan.json"
    ))
    .expect("the anchor plan is valid JSON");
    plan["anchors"]
        .as_object()
        .expect("anchors is an object")
        .iter()
        .filter(|(key, _)| key.ends_with(":mlx"))
        .map(|(key, row)| {
            let parts: Vec<&str> = key.split(':').collect();
            let request = json!({
                "planned": {
                    "target": {
                        "provider": row["provider"],
                        "modelId": parts[0],
                        "tier": parts[1],
                        "mode": row["mode"],
                        "overlay": row["overlay"],
                        "geometry": row["geometry"],
                    },
                }
            });
            (key.clone(), row.clone(), request)
        })
        .collect()
}

/// Every changed arm, by the provider id its plan rows carry, and the video members each one
/// captures. The still `bernini_image` member is served by the SAME `run_bernini` and is the
/// one row that proves the branch is the lane's and not the arm's.
const SINGLE_RENDER_VIDEO_ARMS: [(&str, &str); 8] = [
    ("run_ltx_with_admission (Ordinary)", LTX_PROVIDER),
    ("run_minimax_h3", MINIMAX_PROVIDER),
    ("run_bernini (video member)", BERNINI_PROVIDER),
    ("run_krea_realtime", KREA_REALTIME_PROVIDER),
    ("mlx_ltx25::run", "ltx_2_5"),
    ("mlx_wan_scail2::run (ti2v 5b)", "wan2_2_ti2v_5b"),
    ("mlx_wan_scail2::run (t2v a14b)", "wan2_2_t2v_14b"),
    ("mlx_wan_scail2::run (scail2)", "scail2_14b"),
];

/// Per changed arm: every planned video row resolves to exactly ONE `generate` (no warm pass),
/// and every planned image row — the still Bernini member included — to THREE. Mutations that
/// fail this: restoring the warm passes on the video lane (`VIDEO_WARM_PASSES = 2`), deriving
/// the lane from the provider instead of the frame count, or dropping the image lane's two.
#[test]
fn every_changed_video_arm_captures_one_render_and_the_image_lane_keeps_three() {
    let rows = planned_mlx_rows();
    for (arm, provider) in SINGLE_RENDER_VIDEO_ARMS {
        let video_rows: Vec<_> = rows
            .iter()
            .filter(|(_, row, _)| {
                row["provider"] == provider && row["geometry"]["frames"].as_u64() > Some(1)
            })
            .collect();
        assert!(
            !video_rows.is_empty(),
            "{arm}: no planned video row for {provider}"
        );
        for (key, _, request) in video_rows {
            let policy =
                protocol::capture_policy(request).unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(policy.lane, protocol::CaptureLane::Video, "{arm}: {key}");
            assert_eq!(policy.warm_passes, 0, "{arm}: {key} must run no warm pass");
            assert_eq!(policy.renders(), 1, "{arm}: {key} is one measured render");
            assert!(policy.require_video(arm).is_ok(), "{arm}: {key}");
            let scenario = policy.not_run_warm_repeat_scenario().unwrap();
            assert_eq!(scenario["result"], "not_run", "{arm}: {key}");
            let quality = policy.not_run_quality("c", (1.0, 1.0, 1.0)).unwrap();
            assert_eq!(
                (quality["warmPasses"].as_u64(), quality["result"].as_str()),
                (Some(0), Some("not_run")),
                "{arm}: {key}"
            );
        }
    }
    // The image lane, unchanged: three renders, and no `not_run` receipt may be filed for it.
    let mut image_rows = 0;
    let mut still_bernini = 0;
    for (key, row, request) in &rows {
        if row["geometry"]["frames"].as_u64() != Some(1) {
            continue;
        }
        image_rows += 1;
        let policy =
            protocol::capture_policy(request).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(policy.lane, protocol::CaptureLane::Image, "{key}");
        assert_eq!(
            policy.renders(),
            3,
            "{key}: an image capture is measured + two warm passes"
        );
        assert!(
            policy.not_run_quality("c", (1.0, 1.0, 1.0)).is_err(),
            "{key}"
        );
        assert!(policy.require_video("image arm").is_err(), "{key}");
        if row["provider"] == BERNINI_PROVIDER {
            still_bernini += 1;
            assert_eq!(key.split(':').next(), Some(BERNINI_IMAGE_MODEL_ID), "{key}");
        }
    }
    assert!(
        image_rows > 100,
        "{image_rows} image rows: the plan changed shape"
    );
    assert_eq!(
        still_bernini,
        BERNINI_TIERS.len(),
        "one still Bernini row per tier keeps three renders"
    );
}

/// The Bernini arm's lane is the plan's frame count checked against the member it names: the
/// still member on one frame is the image lane (three renders), the video member is one.
#[test]
fn the_bernini_lane_check_is_the_plans_frame_count_against_the_members() {
    for arm in BERNINI_ARMS {
        let policy = protocol::capture_policy(&json!({
            "planned": {
                "target": {
                    "geometry": { "width": 848, "height": 480, "batch": 1, "frames": arm.frames }
                }
            }
        }))
        .unwrap();
        assert_eq!(
            policy.lane == protocol::CaptureLane::Video,
            arm.frames > 1,
            "{}",
            arm.model_id
        );
        assert_eq!(
            policy.renders(),
            if arm.frames > 1 { 1 } else { 3 },
            "{}",
            arm.model_id
        );
    }
}
