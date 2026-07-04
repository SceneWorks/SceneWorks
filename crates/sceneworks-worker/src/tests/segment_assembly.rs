/// sc-8833 (F-031) — the macOS and off-Mac candle `segment_assembly_frames` twins were collapsed
/// into ONE cfg-free orchestrator parameterized by a segmenter-backend closure. These tests drive
/// that shared orchestrator with a FAKE segmenter (no weights / GPU / network into the model), so
/// the assembly/rollup contract, the OOB bounds guard, and the disabled/missing early-outs are all
/// covered on any lane the orchestrator compiles on. The real model path stays in the `#[ignore]`
/// E2E tests. Compiled only where the orchestrator exists (macOS or `-F backend-candle`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod segment_assembly_frames_tests {
    use super::*;
    use crate::person_track::{NormalizedBox, TrackFrame};
    use std::cell::Cell;

    /// A `TrackFrame` at `timestamp` with the given detection state. Detected frames carry a plausible
    /// box + confidence; gap frames carry `detected=false` (the predictor fills them from memory).
    fn frame(timestamp: f64, detected: bool) -> TrackFrame {
        TrackFrame {
            timestamp,
            box_: NormalizedBox {
                x: 0.25,
                y: 0.10,
                width: 0.30,
                height: 0.70,
            },
            confidence: if detected { 0.9 } else { 0.0 },
            detected,
            flags: Vec::new(),
        }
    }

    /// A `(1280×720)`-sized L8 mask whose first `on_pixels` bytes are 255 (foreground). Zero
    /// `on_pixels` → an all-empty mask (filtered out by the `p > 127` threshold, so no PNG written).
    fn mask(on_pixels: usize) -> Vec<u8> {
        let (w, h) = (1280usize, 720usize);
        let mut buf = vec![0u8; w * h];
        for byte in buf.iter_mut().take(on_pixels) {
            *byte = 255;
        }
        buf
    }

    /// Stub the `check_cancel` GET the orchestrator issues before the segmenter runs (no cancel),
    /// returning an `ApiClient` pointed at it plus a matching `JobSnapshot`.
    async fn api_and_job() -> (ApiClient, JobSnapshot) {
        let base_url = spawn_cancel_poll_stub(CancelPollStubState {
            get_status: AxumStatusCode::OK,
            cancel_requested: false,
            post_status: AxumStatusCode::OK,
        })
        .await;
        let mut settings = test_settings(base_url.clone(), None);
        settings.api_url = base_url;
        let api = ApiClient::new(&settings);
        let job: JobSnapshot = serde_json::from_value(job_snapshot_json("job-1", false))
            .expect("job snapshot deserializes");
        (api, job)
    }

    /// `segment_enabled=false` short-circuits to `missing` before any segmenter or network call —
    /// the fake segmenter must never fire.
    #[tokio::test]
    async fn disabled_segmentation_rolls_up_to_missing_without_running_segmenter() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, true), frame(0.5, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            false,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(
            state, "missing",
            "segmentation disabled → maskState=missing"
        );
        assert!(!ran.get(), "segmenter must not run when disabled");
        assert!(
            frames_json.iter().all(|f| f["mask"].is_null()),
            "no masks written when disabled"
        );
    }

    /// No detected frame → `missing`, again before the segmenter runs.
    #[tokio::test]
    async fn no_detected_frames_rolls_up_to_missing() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, false), frame(0.5, false)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "missing");
        assert!(
            !ran.get(),
            "segmenter must not run with zero detected frames"
        );
    }

    /// OOB bounds guard: `frame_paths` shorter than the last detected assembly index would slice
    /// `frame_paths[first..=last]` past the end and panic. The guard must return `degraded` before
    /// the clip is built, and the fake segmenter must never run. Regression for the twin drift where
    /// only one backend carried the guard.
    #[tokio::test]
    async fn short_frame_paths_hits_bounds_guard_and_degrades_without_panicking() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Three assembly frames, last detected at index 2 — but only ONE rendered frame path.
        let frames = vec![frame(0.0, false), frame(0.5, false), frame(1.0, true)];
        let frame_paths = vec![project.path().join("frame_0.png")];
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let ran = Cell::new(false);
        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| {
                ran.set(true);
                async { SegmentOutcome::Masks(Vec::new()) }
            },
        )
        .await
        .expect("orchestrator ok (guard returns Ok(degraded), never panics)");

        assert_eq!(
            state, "degraded",
            "short frame_paths → degraded via the bounds guard"
        );
        assert!(
            !ran.get(),
            "the OOB guard must short-circuit before the segmenter"
        );
    }

    /// Happy path: every detected frame masked → `active`, masks written to disk, and the clip the
    /// backend receives spans `first..=last` with box anchors on detected frames and `None` on gaps.
    #[tokio::test]
    async fn all_detected_masked_writes_pngs_and_rolls_up_active() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Detected at 0 and 2, a gap at 1 inside the span → clip is the full first..=last (3 frames).
        let frames = vec![frame(0.0, true), frame(0.5, false), frame(1.0, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_x",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |clip: SegmentClip| {
                // The orchestrator handed us the whole first..=last span with per-frame anchors.
                assert_eq!(
                    clip.clip_paths.len(),
                    3,
                    "clip covers first..=last inclusive"
                );
                assert_eq!(clip.anchors.len(), 3);
                assert!(clip.anchors[0].is_some(), "detected frame has a box anchor");
                assert!(clip.anchors[1].is_none(), "gap frame anchor is None");
                assert!(clip.anchors[2].is_some());
                // One non-empty mask per clip frame → every frame gets a PNG.
                async { SegmentOutcome::Masks(vec![mask(100), mask(100), mask(100)]) }
            },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "active", "every detected frame masked → active");
        // Masks written for all three assembly indices (1-based file names).
        for idx in 1..=3 {
            let path = project
                .path()
                .join(format!("person-tracks/track_x/masks/frame_{idx:06}.png"));
            assert!(path.exists(), "mask PNG {idx} written to disk");
        }
        // Each frame's sidecar `mask` is set to its relative path.
        for (i, f) in frames_json.iter().enumerate() {
            let rel = format!("person-tracks/track_x/masks/frame_{:06}.png", i + 1);
            assert_eq!(f["mask"], serde_json::Value::String(rel));
        }
    }

    /// A partial subset masked (some detected frames empty) → `generated`, and empty masks (all
    /// pixels below the `p > 127` threshold) write no PNG.
    #[tokio::test]
    async fn partial_masking_rolls_up_generated_and_empty_masks_skip_pngs() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        // Three detected frames; the middle mask is empty → 2 of 3 masked → generated.
        let frames = vec![frame(0.0, true), frame(0.5, true), frame(1.0, true)];
        let frame_paths: Vec<PathBuf> = frames
            .iter()
            .enumerate()
            .map(|(i, _)| project.path().join(format!("frame_{i}.png")))
            .collect();
        let mut frames_json = crate::person_track::frames_to_json(&frames);

        let state = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_y",
            &frames,
            &frame_paths,
            &mut frames_json,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Masks(vec![mask(50), mask(0), mask(50)]) },
        )
        .await
        .expect("orchestrator ok");

        assert_eq!(state, "generated", "2 of 3 detected masked → generated");
        // The empty middle mask writes no PNG and leaves its sidecar `mask` null.
        assert!(frames_json[1]["mask"].is_null(), "empty mask writes no PNG");
        assert!(!project
            .path()
            .join("person-tracks/track_y/masks/frame_000002.png")
            .exists());
        assert!(frames_json[0]["mask"].is_string());
        assert!(frames_json[2]["mask"].is_string());
    }

    /// A backend cancel is terminal for the whole job (never a degrade); a backend failure degrades
    /// to box masks. Both mappings live once, in the shared orchestrator.
    #[tokio::test]
    async fn backend_canceled_propagates_and_degraded_maps_to_degraded() {
        let project = tempdir().expect("tempdir");
        let (api, job) = api_and_job().await;
        let frames = vec![frame(0.0, true)];
        let frame_paths = vec![project.path().join("frame_0.png")];

        // Canceled → terminal WorkerError::Canceled.
        let mut json_a = crate::person_track::frames_to_json(&frames);
        let canceled = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_c",
            &frames,
            &frame_paths,
            &mut json_a,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Canceled("user canceled".to_owned()) },
        )
        .await;
        assert!(
            matches!(canceled, Err(WorkerError::Canceled(ref m)) if m == "user canceled"),
            "backend cancel is terminal, got {canceled:?}"
        );

        // Degraded → Ok("degraded"), box-mask fallback at replacement time.
        let mut json_b = crate::person_track::frames_to_json(&frames);
        let degraded = segment_assembly_frames(
            &api,
            project.path(),
            &job,
            "track_d",
            &frames,
            &frame_paths,
            &mut json_b,
            true,
            |_clip: SegmentClip| async { SegmentOutcome::Degraded },
        )
        .await
        .expect("degraded is Ok");
        assert_eq!(degraded, "degraded");
    }

    /// The rollup helper both backends now share (Python `segment_track` contract).
    #[test]
    fn mask_rollup_state_matches_python_segment_track() {
        assert_eq!(mask_rollup_state(0, 4), "degraded");
        assert_eq!(mask_rollup_state(2, 4), "generated");
        assert_eq!(mask_rollup_state(4, 4), "active");
        assert_eq!(mask_rollup_state(1, 1), "active");
        // generated capped at detected_total → still active.
        assert_eq!(mask_rollup_state(6, 5), "active");
    }
}
