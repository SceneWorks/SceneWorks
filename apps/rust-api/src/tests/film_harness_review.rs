//! Vision-assisted review, the human accept/reject/repair loop, and the labeled evaluation
//! (sc-22714).
//!
//! Same shape as the sc-22710/22711 suite: the REAL API routes in-process, and a scripted fake
//! worker claiming the jobs through the worker API. What the fake replaces is only the model — the
//! `frame_extract` job writes a placeholder still where FFmpeg would and the `image_vqa` job
//! answers from a table in the shape SenseNova-U1 posts. Asset persistence, timeline validation,
//! job routing and every enqueue gate are production code paths.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sceneworks_core::film_plan::{RunOutcome, RunRecord};
use sceneworks_core::film_review::{
    parse_eval_set, parse_review_plan, validate_eval_set, validate_review_plan, EvalSet,
    ObservedState, ReviewPlan, Verdict, ASSISTIVE_NOTICE,
};
use serde_json::Value;

use crate::film_harness::review::{
    self, Decision, EvalOptions, ReviewOptions, ScriptedVision, VqaVision,
};
use crate::film_harness::{self, RunControl};
use crate::tests::film_harness::{fast, harness_record, Harness, FIXTURE_DIR};

const REVIEW_PLAN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/courier-workshop/review.jsonc"
);
const EVAL_SET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/review-eval/labels.jsonc"
);
const REAL_TAKES_SET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/film-harness/review-eval/real-takes.jsonc"
);

fn plan_path() -> PathBuf {
    PathBuf::from(FIXTURE_DIR).join("plan.jsonc")
}

fn pack_path() -> PathBuf {
    PathBuf::from(FIXTURE_DIR).join("references.jsonc")
}

fn shipped_review_plan() -> ReviewPlan {
    parse_review_plan(&std::fs::read_to_string(REVIEW_PLAN).expect("review plan reads"))
        .expect("review plan parses")
}

/// A run of SH010 + SH020 that completed, ready to review.
async fn rendered_two_shots() -> (Harness, RunRecord) {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let options = harness.options(plan_path(), pack_path(), Some(&["SH010", "SH020"]));
    let record = film_harness::run(&harness.transport, &options)
        .await
        .expect("the two-shot run completes");
    assert_eq!(record.outcome, RunOutcome::Completed, "{record:#?}");
    (harness, record)
}

fn review_options(harness: &Harness, shots: &[&str]) -> ReviewOptions {
    let mut options = ReviewOptions::new(harness.out_dir());
    options.review_plan_path = Some(PathBuf::from(REVIEW_PLAN));
    options.poll_interval = Duration::from_millis(100);
    options.control = RunControl::new();
    options.shot_ids = shots.iter().map(|id| (*id).to_owned()).collect();
    options
}

fn script_answers(harness: &Harness, answers: &[(&str, &str)]) {
    let mut script = harness.script.lock();
    for (key, answer) in answers {
        script
            .vqa_answers
            .insert((*key).to_owned(), (*answer).to_owned());
    }
}

/// Answers that make every SH010 and SH020 question agree with the plan.
fn agreeing_answers() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "sh010_location",
            "Yes. This is a cluttered woodworking workshop with a long wooden workbench.",
        ),
        (
            "sh010_courier",
            "Yes. A person — a courier — is standing in the doorway.",
        ),
        ("sh010_courier_jacket", "The jacket is blue."),
        ("sh010_parcel", "Yes, a small box. It is red."),
        ("sh010_parcel_custody", "It is in a person's hands."),
        (
            "sh010_action",
            "The door is open and the workbench is clear.",
        ),
        (
            "sh020_location",
            "Yes. This is a cluttered woodworking workshop with a long wooden workbench.",
        ),
        (
            "sh020_courier",
            "Yes. A person in a jacket is walking across the room.",
        ),
        ("sh020_courier_jacket", "The jacket is blue."),
        ("sh020_parcel", "Yes, a small box. It is red."),
        ("sh020_parcel_custody", "It is still in the hands."),
        ("sh020_action", "The person is at the workbench."),
        (
            "sh020_cut",
            "A woodworking workshop; the person wears a blue jacket.",
        ),
    ]
}

fn observed_for(record: &RunRecord, out_dir: &Path, shot_id: &str) -> ObservedState {
    let shot = record.shot(shot_id).expect("shot is in the record");
    let summary = shot.latest_review().expect("the shot has a review");
    review::read_observed_state(&out_dir.join(&summary.record_path))
        .expect("the observed-state document reads")
}

// ---------------------------------------------------------------------------------------------
// The documents themselves
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shipped_review_plan_and_labeled_sets_validate_against_the_fixture_plan() {
    let plan = sceneworks_core::film_plan::read_plan_file(&plan_path()).expect("plan reads");
    let review = shipped_review_plan();
    assert!(
        validate_review_plan(&review, &plan).is_empty(),
        "{:#?}",
        validate_review_plan(&review, &plan)
    );
    // Every shot of the plan is reviewable, and every review topic the benchmark tests appears.
    for shot in &plan.shots {
        assert!(
            review.shots.contains_key(&shot.id),
            "shot {} has no review questions",
            shot.id
        );
    }
    let topics: std::collections::BTreeSet<&str> = review
        .shots
        .values()
        .flat_map(|spec| spec.questions.iter())
        .map(|question| question.topic.as_str())
        .collect();
    for required in sceneworks_core::film_review::REVIEW_TOPICS {
        assert!(topics.contains(required), "no question covers {required}");
    }
    // Every handoff / action-completion question must be mustObserve: an action nobody saw is the
    // exact thing that must not be able to read as completed.
    for (shot_id, spec) in &review.shots {
        for question in &spec.questions {
            if matches!(
                question.topic.as_str(),
                "parcel_custody" | "action_completion"
            ) {
                assert!(
                    question.must_observe,
                    "{shot_id}.{} is a custody/action question and must be mustObserve",
                    question.id
                );
            }
        }
    }

    for path in [EVAL_SET, REAL_TAKES_SET] {
        let set: EvalSet = parse_eval_set(&std::fs::read_to_string(path).expect("set reads"))
            .unwrap_or_else(|error| panic!("{path}: {error}"));
        let findings = validate_eval_set(&set, &review);
        assert!(findings.is_empty(), "{path}: {findings:#?}");
    }
}

#[test]
fn the_checked_in_labeled_frames_match_the_generator_byte_for_byte() {
    let set_path = PathBuf::from(EVAL_SET);
    let temp = tempfile::tempdir().expect("temp dir");
    let written = review::write_review_fixture_frames(&set_path, Some(temp.path()))
        .expect("fixture frames write");
    assert!(!written.is_empty());
    let shipped_root = set_path.parent().expect("set dir").join("frames");
    for path in &written {
        let name = path.file_name().expect("file name");
        let shipped = shipped_root.join(name);
        let expected = std::fs::read(&shipped)
            .unwrap_or_else(|error| panic!("{} is not checked in: {error}", shipped.display()));
        let actual = std::fs::read(path).expect("generated frame reads");
        assert_eq!(
            actual,
            expected,
            "{} drifted from the generator; re-run `film-harness review-fixtures`",
            shipped.display()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Review
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_review_writes_observed_state_beside_the_run_and_only_points_at_the_intended_state() {
    let (harness, record) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    vision
        .preflight()
        .await
        .expect("the fake worker advertises image_vqa");
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");

    let out_dir = harness.out_dir();
    let shot = reviewed.shot("SH020").expect("SH020 is in the record");
    let summary = shot.latest_review().expect("SH020 was reviewed");
    assert_eq!(summary.attempt, shot.selected_attempt.expect("selected"));
    assert_eq!(summary.backend, "image_vqa");

    // The record holds a POINTER and counts — not a single observed value.
    let summary_json = serde_json::to_string(summary).expect("summary serializes");
    for leaked in ["blue", "red", "workshop", "woodworking"] {
        assert!(
            !summary_json.contains(leaked),
            "the run record's review index leaked an observed value ({leaked}): {summary_json}"
        );
    }

    let observed = observed_for(&reviewed, &out_dir, "SH020");
    assert_eq!(observed.notice, ASSISTIVE_NOTICE);
    assert_eq!(observed.shot_id, "SH020");
    // Timestamped frame evidence, one per declared sample position, each an openable asset.
    let review_plan = shipped_review_plan();
    let own_frames: Vec<_> = observed
        .frames
        .iter()
        .filter(|frame| frame.shot_id == "SH020")
        .collect();
    assert_eq!(own_frames.len(), review_plan.sampling.positions.len());
    for frame in &own_frames {
        assert!(!frame.asset_id.is_empty(), "{frame:?}");
        assert!(frame.path.ends_with(".png"), "{frame:?}");
        assert_eq!(frame.source, "frame_extract");
        assert!(frame.job_id.is_some(), "{frame:?}");
        assert!(frame.timestamp_seconds >= 0.0);
    }
    let timestamps: Vec<f64> = own_frames.iter().map(|f| f.timestamp_seconds).collect();
    let mut sorted = timestamps.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(timestamps, sorted, "frames must be in timestamp order");

    // Every question answered, each with a confidence and an explicit unobserved flag.
    assert_eq!(
        observed.observations.len(),
        review_plan.shots["SH020"].questions.len()
    );
    for observation in &observed.observations {
        assert!(observation.is_well_formed(), "{observation:?}");
        assert!((0.0..=1.0).contains(&observation.confidence));
        assert!(!observation.answers.is_empty(), "{observation:?}");
    }
    assert!(
        observed.mismatches.is_empty(),
        "agreeing answers raise no flags: {:#?}",
        observed.mismatches
    );

    // The intended state is REFERENCED, and the pointer resolves to the run record's own intent.
    assert_eq!(observed.intended.run_id, reviewed.run_id);
    assert_eq!(observed.intended.plan_sha256, reviewed.plan.sha256);
    let record_json = serde_json::to_value(&reviewed).expect("record serializes");
    let pointed = record_json
        .pointer(&observed.intended.record_pointer)
        .unwrap_or_else(|| panic!("{} does not resolve", observed.intended.record_pointer));
    assert_eq!(
        pointed,
        &serde_json::to_value(&shot.intended).expect("intended serializes"),
        "the pointer must name THIS shot's intended state"
    );
    // And the observed document carries no copy of it.
    let observed_json = serde_json::to_value(observed.to_json()).expect("observed serializes");
    assert!(
        observed_json.pointer("/intended/startState").is_none()
            && observed_json.pointer("/intended/endState").is_none(),
        "the observed-state document must reference the intent, never copy it: {observed_json}"
    );
    // The backend is honest about the fake: no weights ran.
    assert!(!observed.backend.real_model_inference);
    assert_eq!(observed.backend.model, review::VQA_MODEL_ID);
    assert_eq!(observed.backend.route, review::VQA_ROUTE);
    let _ = record;
}

#[tokio::test]
async fn an_unobserved_handoff_is_flagged_unobserved_and_never_recorded_as_completed() {
    let (harness, _) = rendered_two_shots().await;
    let mut answers = agreeing_answers();
    // The camera never shows whether the parcel left the courier's hands.
    answers.retain(|(key, _)| !key.starts_with("sh020_parcel_custody"));
    answers.push((
        "sh020_parcel_custody",
        "I cannot tell — the parcel is out of frame behind the person.",
    ));
    script_answers(&harness, &answers);
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");

    let observed = observed_for(&reviewed, &harness.out_dir(), "SH020");
    let custody = observed
        .observations
        .iter()
        .find(|observation| observation.question_id == "sh020_parcel_custody")
        .expect("the custody question was asked");
    assert_eq!(custody.verdict, Verdict::Unobserved);
    assert!(custody.unobserved);
    assert!(
        custody.observed.is_none(),
        "an unobserved handoff must carry NO value: {custody:?}"
    );
    assert_eq!(custody.confidence, 0.0);

    // And on disk, the serialized document has no `observed` key at all for it.
    let raw: Value = serde_json::from_str(
        &std::fs::read_to_string(
            harness.out_dir().join(
                &reviewed
                    .shot("SH020")
                    .expect("shot")
                    .latest_review()
                    .expect("review")
                    .record_path,
            ),
        )
        .expect("document reads"),
    )
    .expect("document parses");
    let on_disk = raw["observations"]
        .as_array()
        .expect("observations array")
        .iter()
        .find(|observation| observation["questionId"] == "sh020_parcel_custody")
        .expect("the custody observation is on disk");
    assert!(on_disk.get("observed").is_none(), "{on_disk}");
    assert_eq!(on_disk["unobserved"], Value::Bool(true));
    for forbidden in ["completed", "complete", "done"] {
        assert!(
            !on_disk.to_string().to_ascii_lowercase().contains(forbidden),
            "an unobserved handoff must never read as {forbidden}: {on_disk}"
        );
    }

    // It IS an actionable flag, because the question is mustObserve.
    let flag = observed
        .mismatches
        .iter()
        .find(|flag| flag.question_id == "sh020_parcel_custody")
        .expect("a mustObserve question that was not observed raises a flag");
    assert_eq!(flag.severity, "unobserved");
    assert_eq!(flag.observed, "unobserved");
    assert!(flag.detail.contains("NOT as completed"), "{}", flag.detail);
    assert_eq!(flag.shot_id, "SH020");
    assert_eq!(flag.topic, "parcel_custody");
}

#[tokio::test]
async fn a_disagreeing_answer_becomes_an_actionable_mismatch_naming_intended_and_observed() {
    let (harness, _) = rendered_two_shots().await;
    let mut answers = agreeing_answers();
    answers.retain(|(key, _)| *key != "sh020_parcel");
    answers.push(("sh020_parcel", "Yes, there is a small blue parcel."));
    script_answers(&harness, &answers);
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH020");
    let flag = observed
        .mismatches
        .iter()
        .find(|flag| flag.question_id == "sh020_parcel")
        .expect("the wrong parcel colour is flagged");
    assert_eq!(flag.severity, "mismatch");
    assert_eq!(flag.topic, "parcel_identity");
    for flag in &observed.mismatches {
        assert!(
            sceneworks_core::film_review::MISMATCH_SEVERITIES.contains(&flag.severity.as_str()),
            "{:?} is outside the declared severity vocabulary",
            flag.severity
        );
    }
    assert!(flag.intended.contains("red"), "{}", flag.intended);
    assert!(flag.observed.contains("blue"), "{}", flag.observed);
    assert!(
        !flag.evidence_frame_ids.is_empty(),
        "a flag must cite the frames it came off"
    );
    assert_eq!(
        reviewed
            .shot("SH020")
            .expect("shot")
            .latest_review()
            .expect("review")
            .topics_flagged,
        vec!["parcel_identity".to_owned()]
    );
}

#[tokio::test]
async fn a_hedged_contradiction_is_uncertain_and_never_an_actionable_flag() {
    let (harness, _) = rendered_two_shots().await;
    let mut answers = agreeing_answers();
    answers.retain(|(key, _)| *key != "sh020_parcel");
    answers.push((
        "sh020_parcel",
        "It appears to be a blue parcel, though the light is very warm.",
    ));
    script_answers(&harness, &answers);
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH020");
    let flag = observed
        .mismatches
        .iter()
        .find(|flag| flag.question_id == "sh020_parcel")
        .expect("the hedged contradiction is still recorded");
    assert_eq!(flag.severity, "uncertain");
    assert!(
        observed
            .actionable()
            .iter()
            .all(|flag| flag.question_id != "sh020_parcel"),
        "a hedge must not be promoted to something a person is told to act on"
    );
    assert_eq!(
        reviewed
            .shot("SH020")
            .expect("shot")
            .latest_review()
            .expect("review")
            .actionable_flags,
        0
    );
}

#[tokio::test]
async fn a_review_never_writes_intended_state_or_conditioning_and_never_reaches_a_render() {
    let (harness, before) = rendered_two_shots().await;
    let mut answers = agreeing_answers();
    // Deliberately contradictory readings, so there is plenty of observed text to leak.
    answers.retain(|(key, _)| *key != "sh020_parcel" && *key != "sh020_location");
    answers.push(("sh020_parcel", "Yes, there is a small blue parcel."));
    answers.push(("sh020_location", "No — this is a kitchen, not a workshop."));
    script_answers(&harness, &answers);
    let options = review_options(&harness, &["SH010", "SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let after = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");

    for shot in &before.shots {
        let reviewed = after.shot(&shot.shot_id).expect("same shots");
        assert_eq!(
            reviewed.intended, shot.intended,
            "a review must not touch {}'s intended state",
            shot.shot_id
        );
        assert_eq!(
            reviewed.conditioning_assets, shot.conditioning_assets,
            "a review must not touch {}'s conditioning",
            shot.shot_id
        );
        assert_eq!(
            reviewed.selected_attempt, shot.selected_attempt,
            "a review must not move {}'s selection",
            shot.shot_id
        );
        assert_eq!(
            reviewed.attempts, shot.attempts,
            "a review must not disturb {}'s takes",
            shot.shot_id
        );
    }

    // Nothing the reviewer read reached a generation job — before OR after. The kitchen reading is
    // the canary: it exists only in the observed-state document.
    let jobs = harness.jobs().await;
    let generation: Vec<&Value> = jobs
        .iter()
        .filter(|job| job["type"] == "video_generate")
        .collect();
    assert_eq!(generation.len(), 2, "no extra render was dispatched");
    for job in generation {
        let payload = job["payload"].to_string().to_ascii_lowercase();
        for leaked in ["kitchen", "blue parcel", "i cannot tell", "unobserved"] {
            assert!(
                !payload.contains(leaked),
                "a generation payload carried an observation ({leaked}): {payload}"
            );
        }
    }
    // And the observed values ARE in the documents, so the assertion above is not vacuous.
    let observed = observed_for(&after, &harness.out_dir(), "SH020");
    assert!(serde_json::to_string(&observed.to_json())
        .expect("serializes")
        .to_ascii_lowercase()
        .contains("kitchen"));
}

#[tokio::test]
async fn a_second_review_appends_and_keeps_the_first_documents_evidence() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let first = review::review(&harness.transport, &options, &vision)
        .await
        .expect("first review");
    let first_path = harness.out_dir().join(
        &first
            .shot("SH020")
            .expect("shot")
            .latest_review()
            .expect("review")
            .record_path,
    );
    let first_bytes = std::fs::read(&first_path).expect("first document reads");

    script_answers(
        &harness,
        &[("sh020_parcel", "Yes, there is a small blue parcel.")],
    );
    let second = review::review(&harness.transport, &options, &vision)
        .await
        .expect("second review");
    let shot = second.shot("SH020").expect("shot");
    assert_eq!(shot.reviews.len(), 2, "reviews are append-only");
    assert_ne!(shot.reviews[0].record_path, shot.reviews[1].record_path);
    assert_eq!(
        std::fs::read(&first_path).expect("first document still reads"),
        first_bytes,
        "an earlier review's evidence must survive a later one"
    );
    assert_eq!(shot.reviews[0].actionable_flags, 0);
    assert_eq!(shot.reviews[1].actionable_flags, 1);
}

#[tokio::test]
async fn review_refuses_a_shot_with_no_take_and_a_shot_the_review_plan_asks_nothing_about() {
    let harness = Harness::start(true, fast(&["SH010"])).await;
    let options = harness.options(plan_path(), pack_path(), Some(&["SH010"]));
    film_harness::run(&harness.transport, &options)
        .await
        .expect("the one-shot run completes");
    let vision = ScriptedVision::new();

    let mut review_options = review_options(&harness, &["SH020"]);
    let error = review::review(&harness.transport, &review_options, &vision)
        .await
        .expect_err("SH020 was never rendered");
    assert!(format!("{error}").contains("no selected take"), "{error}");

    // A review plan that asks nothing about a shot refuses rather than reviewing it against
    // nothing and reporting a clean bill of health.
    let empty = tempfile::tempdir().expect("temp dir");
    let mut document = shipped_review_plan();
    document.shots.remove("SH010");
    let path = empty.path().join("review.jsonc");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&document).expect("serializes"),
    )
    .expect("writes");
    review_options.review_plan_path = Some(path);
    review_options.shot_ids = vec!["SH010".to_owned()];
    let error = review::review(&harness.transport, &review_options, &vision)
        .await
        .expect_err("a shot with no questions is refused");
    assert!(format!("{error}").contains("asks no questions"), "{error}");
}

#[tokio::test]
async fn a_review_that_hits_its_frame_budget_stops_and_keeps_the_partial_evidence() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    let temp = tempfile::tempdir().expect("temp dir");
    let mut document = shipped_review_plan();
    document.limits.max_frames_per_shot = 2;
    let path = temp.path().join("review.jsonc");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&document).expect("serializes"),
    )
    .expect("writes");
    let mut options = review_options(&harness, &["SH020"]);
    options.review_plan_path = Some(path);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("a bounded review still returns");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH020");
    assert_eq!(
        observed
            .frames
            .iter()
            .filter(|frame| frame.shot_id == "SH020")
            .count(),
        2,
        "the frame budget bounded the sampling"
    );
    let stop = observed
        .stop
        .as_deref()
        .expect("a bounded review says why it stopped");
    assert!(stop.contains("frame_budget"), "{stop}");
    assert!(
        !observed.observations.is_empty(),
        "the partial evidence is kept, not discarded"
    );
    assert_eq!(
        reviewed
            .shot("SH020")
            .expect("shot")
            .latest_review()
            .expect("review")
            .stop
            .as_deref(),
        Some(stop)
    );
}

#[test]
fn a_review_plan_that_declares_more_questions_than_it_budgets_for_is_refused_up_front() {
    let plan = sceneworks_core::film_plan::read_plan_file(&plan_path()).expect("plan reads");
    let mut document = shipped_review_plan();
    document.limits.max_questions_per_shot = 2;
    let findings = validate_review_plan(&document, &plan);
    assert!(
        findings
            .iter()
            .any(|finding| finding.message.contains("maxQuestionsPerShot")),
        "a document whose questions cannot all be asked is a document error, not a silent \
         truncation at review time: {findings:#?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The human loop
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn accepting_a_take_clears_only_that_shots_flags_and_records_the_decision() {
    let (harness, _) = rendered_two_shots().await;
    let out_dir = harness.out_dir();
    // Replace SH010's take so SH020 (which declares a continuity edge to it) is flagged.
    film_harness::replace_take(
        &harness.transport,
        &harness.resume_options(),
        "SH010",
        "the doorway is too dark",
    )
    .await
    .expect("the replacement lands");
    let flagged = harness_record(&harness);
    assert_eq!(
        flagged.shot("SH020").expect("shot").needs_review.len(),
        1,
        "SH020 declares a continuity edge to SH010"
    );
    let sh010_before = flagged.shot("SH010").expect("shot").clone();

    let accepted = review::decide_take(&out_dir, "SH020", Decision::Accept, "looks fine to me")
        .expect("accept records");
    let sh020 = accepted.shot("SH020").expect("shot");
    assert!(
        sh020.needs_review.is_empty(),
        "accepting resolves the flags the person just read"
    );
    let decision = sh020
        .human_decision
        .as_ref()
        .expect("a decision is recorded");
    assert_eq!(decision.state, "accepted");
    assert_eq!(decision.reason, "looks fine to me");
    assert_eq!(decision.attempt, sh020.selected_attempt.expect("selected"));
    assert_eq!(
        *accepted.shot("SH010").expect("shot"),
        sh010_before,
        "accepting SH020 must not touch SH010"
    );
    let logged = accepted.decisions.last().expect("the decision log grew");
    assert_eq!(logged.action, "accept_take");
    assert_eq!(logged.shot_id.as_deref(), Some("SH020"));
    assert!(
        logged.detail.contains("cleared 1 needsReview"),
        "{logged:?}"
    );
    // And it survives on disk.
    assert_eq!(harness_record(&harness), accepted);
}

#[tokio::test]
async fn rejecting_a_take_keeps_it_flags_dependents_and_leaves_unrelated_accepted_work_alone() {
    let (harness, _) = rendered_two_shots().await;
    let out_dir = harness.out_dir();
    let accepted = review::decide_take(&out_dir, "SH020", Decision::Accept, "SH020 is good")
        .expect("accept records");
    let sh020_before = accepted.shot("SH020").expect("shot").clone();
    let sh010_before = accepted.shot("SH010").expect("shot").clone();

    let rejected = review::decide_take(
        &out_dir,
        "SH010",
        Decision::Reject,
        "the courier's jacket is the wrong blue",
    )
    .expect("reject records");

    let sh010 = rejected.shot("SH010").expect("shot");
    // The take, its job and its asset are all still there.
    assert_eq!(sh010.attempts.len(), sh010_before.attempts.len());
    let rejected_attempt = sh010
        .attempts
        .iter()
        .find(|attempt| attempt.attempt == sh010_before.selected_attempt.expect("selected"))
        .expect("the take is still recorded");
    assert!(rejected_attempt.take.is_some(), "the take is kept");
    assert!(rejected_attempt.job_id.is_some(), "its job is kept");
    let rejection = rejected_attempt
        .rejection
        .as_ref()
        .expect("the rejection is recorded");
    assert_eq!(rejection.reason, "the courier's jacket is the wrong blue");
    assert!(sh010.selected_attempt.is_none(), "nothing is selected now");
    assert_eq!(
        sh010.human_decision.as_ref().map(|d| d.state.as_str()),
        Some("rejected")
    );

    // The declared dependent is flagged, not re-rendered, and its accepted work is intact.
    let sh020 = rejected.shot("SH020").expect("shot");
    assert_eq!(sh020.needs_review.len(), 1, "SH020 is flagged");
    assert_eq!(sh020.needs_review[0].source_shot_id, "SH010");
    assert_eq!(sh020.needs_review[0].dependency, "continuity");
    assert_eq!(
        sh020.selected_attempt, sh020_before.selected_attempt,
        "the accepted take of an unrelated shot does not move"
    );
    assert_eq!(sh020.attempts, sh020_before.attempts);
    assert_eq!(
        sh020.human_decision.as_ref().map(|d| d.state.as_str()),
        Some("accepted"),
        "the earlier acceptance survives"
    );
    if let Some(export) = &rejected.export {
        assert!(export.stale, "the MP4 no longer matches the selected takes");
    }
    // Nothing was rendered.
    assert_eq!(
        harness
            .jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "video_generate")
            .count(),
        2
    );
    let logged = rejected.decisions.last().expect("logged");
    assert_eq!(logged.action, "reject_take");
    assert!(
        logged.detail.contains("nothing was re-rendered"),
        "{logged:?}"
    );
}

#[tokio::test]
async fn reject_refuses_a_shot_that_has_no_take_at_all() {
    let (harness, _) = rendered_two_shots().await;
    let out_dir = harness.out_dir();
    let error = review::decide_take(&out_dir, "SH050", Decision::Reject, "never rendered")
        .expect_err("SH050 was not in the selection");
    assert!(format!("{error}").contains("no take"), "{error}");
}

#[tokio::test]
async fn a_repair_renders_exactly_one_more_take_with_the_reviews_flags_in_the_reason() {
    let (harness, _) = rendered_two_shots().await;
    let mut answers = agreeing_answers();
    answers.retain(|(key, _)| *key != "sh020_parcel" && *key != "sh020_parcel_custody");
    answers.push(("sh020_parcel", "Yes, there is a small blue parcel."));
    answers.push(("sh020_parcel_custody", "I cannot tell from this frame."));
    script_answers(&harness, &answers);
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");
    let before = reviewed.shot("SH020").expect("shot").clone();
    assert_eq!(before.latest_review().expect("review").actionable_flags, 2);

    let mut resume = harness.resume_options();
    resume.export = false;
    let repaired = review::request_repair(&harness.transport, &resume, "SH020", "fix the parcel")
        .await
        .expect("the repair runs");

    let sh020 = repaired.shot("SH020").expect("shot");
    assert_eq!(
        sh020.attempts.len(),
        before.attempts.len() + 1,
        "exactly ONE more attempt"
    );
    let new_attempt = sh020.attempts.last().expect("the repair attempt");
    assert!(
        new_attempt.human_requested,
        "a repair is a human decision, not a retry"
    );
    assert_eq!(sh020.selected_attempt, Some(new_attempt.attempt));
    // The rejected take is kept beside the new one.
    let old = sh020
        .attempts
        .iter()
        .find(|attempt| attempt.attempt == before.selected_attempt.expect("selected"))
        .expect("the old take is kept");
    assert!(old.take.is_some());
    let reason = &old.rejection.as_ref().expect("rejection").reason;
    assert!(reason.contains("fix the parcel"), "{reason}");
    assert!(reason.contains("parcel_identity"), "{reason}");
    assert!(reason.contains("parcel_custody"), "{reason}");
    assert!(reason.contains("[mismatch]"), "{reason}");
    assert!(reason.contains("[unobserved]"), "{reason}");

    // The decision log says a repair was authorised, before the dispatch.
    let repair = repaired
        .decisions
        .iter()
        .find(|decision| decision.action == "request_repair")
        .expect("the repair is in the decision log");
    assert_eq!(repair.shot_id.as_deref(), Some("SH020"));
    assert!(
        repair.detail.contains("one bounded repair attempt"),
        "{repair:?}"
    );
    // One extra render, and no loop.
    assert_eq!(
        harness
            .jobs()
            .await
            .iter()
            .filter(|job| job["type"] == "video_generate")
            .count(),
        3
    );
    // The review of the take that was just replaced stays in the record as evidence.
    assert_eq!(sh020.reviews.len(), 1);
    assert!(
        sh020.review_of_selected().is_none(),
        "the surviving review is of the REPLACED take, so it says nothing about the new one"
    );
    // And the replacement's prompt is the plan's, not the review's.
    let dispatched = harness.jobs().await;
    let newest = dispatched
        .iter()
        .filter(|job| job["type"] == "video_generate")
        .find(|job| {
            job.pointer("/payload/advanced/filmHarness/attempt")
                .and_then(Value::as_u64)
                == Some(u64::from(new_attempt.attempt))
        })
        .expect("the repair job");
    let payload = newest["payload"].to_string().to_ascii_lowercase();
    for leaked in ["blue parcel", "i cannot tell", "mismatch", "unobserved"] {
        assert!(
            !payload.contains(leaked),
            "the repair prompt carried an observation ({leaked}): {payload}"
        );
    }
}

#[tokio::test]
async fn a_repair_of_an_unreviewed_shot_carries_only_the_persons_words() {
    let (harness, _) = rendered_two_shots().await;
    let mut resume = harness.resume_options();
    resume.export = false;
    let repaired = review::request_repair(
        &harness.transport,
        &resume,
        "SH010",
        "the doorway is too dark",
    )
    .await
    .expect("the repair runs");
    let sh010 = repaired.shot("SH010").expect("shot");
    let rejected = sh010
        .attempts
        .iter()
        .find(|attempt| attempt.rejection.is_some())
        .expect("the old take is rejected");
    assert_eq!(
        rejected.rejection.as_ref().expect("rejection").reason,
        "the doorway is too dark"
    );
}

// ---------------------------------------------------------------------------------------------
// The labeled evaluation
// ---------------------------------------------------------------------------------------------

/// Answers for the labeled set that are RIGHT everywhere except three deliberate reviewer errors:
/// a miss (the wrong parcel colour read as red), a false alarm (a correct location read as a
/// kitchen) and an overclaim (an occluded handoff answered confidently). The evaluation has to
/// report each as its own kind.
fn eval_answers() -> BTreeMap<String, String> {
    let mut answers = BTreeMap::new();
    let mut set = |key: &str, answer: &str| {
        answers.insert(key.to_owned(), answer.to_owned());
    };
    // Generic per-question answers that agree with the plan.
    set(
        "sh020_location",
        "Yes, a cluttered woodworking workshop with a long wooden workbench.",
    );
    set("sh020_courier", "Yes, a person is walking across the room.");
    set("sh020_courier_jacket", "The jacket is blue.");
    set("sh020_parcel", "Yes, a small box. It is red.");
    set("sh020_parcel_custody", "It is still in the hands.");
    set("sh020_action", "The person is at the workbench.");
    set(
        "sh020_cut",
        "A woodworking workshop; the person wears a blue jacket.",
    );
    set(
        "sh030_bench",
        "Yes, this is a wooden workbench covered in sawdust.",
    );
    set("sh030_sleeve", "The sleeves are blue.");
    set("sh030_parcel", "Yes, a small box. It is red.");
    set(
        "sh030_parcel_custody",
        "It is resting on the workbench on its own.",
    );
    set("sh030_action", "The hands have let go and left the frame.");
    set("sh030_cut", "A wooden workbench with a red parcel on it.");
    set(
        "sh050_location",
        "Yes, a cluttered woodworking workshop with a long wooden workbench.",
    );
    set("sh050_recipient", "Yes, a woman is in the room.");
    set("sh050_recipient_apron", "She is wearing a grey apron.");
    set("sh050_parcel", "Yes, a small box. It is red.");
    set(
        "sh050_action",
        "She is standing at the workbench looking down at the parcel.",
    );
    set(
        "sh050_cut",
        "A woodworking workshop with a red parcel on the bench.",
    );
    set("sh060_bench", "Yes, this is a wooden workbench.");
    set("sh060_apron_sleeve", "The sleeves are grey.");
    set("sh060_parcel", "Yes, a small box. It is red.");
    set(
        "sh060_parcel_custody",
        "The hands wear a grey apron sleeve.",
    );
    set(
        "sh060_action",
        "The parcel is open with its flaps folded back.",
    );
    set("sh060_cut", "A wooden workbench with a red parcel on it.");
    answers
}

/// Rewrite the per-question answers for the frames of one case.
fn case_answers(
    answers: &mut BTreeMap<String, String>,
    case_id: &str,
    frames: usize,
    overrides: &[(&str, &str)],
) {
    for (question, answer) in overrides {
        for index in 1..=frames {
            answers.insert(
                format!("{question}@{case_id}-f{index}"),
                (*answer).to_owned(),
            );
        }
    }
}

fn eval_backend() -> ScriptedVision {
    let mut answers = eval_answers();

    // --- correct cases keep the agreeing answers, except one deliberate FALSE ALARM ---
    case_answers(
        &mut answers,
        "sh020_correct",
        3,
        &[("sh020_location", "No, this looks like a kitchen to me.")],
    );

    // --- wrong parcel colour: the reviewer catches the cut question but MISSES the colour ---
    case_answers(
        &mut answers,
        "sh030_wrong_parcel_colour",
        3,
        &[
            ("sh030_parcel", "Yes, a small box. It is red."),
            ("sh030_cut", "A wooden workbench with a blue parcel on it."),
        ],
    );

    // --- missing character ---
    case_answers(
        &mut answers,
        "sh020_missing_courier",
        3,
        &[
            ("sh020_courier", "There is no person in this frame."),
            (
                "sh020_courier_jacket",
                "I cannot tell — there is nobody here.",
            ),
            ("sh020_parcel", "There is no parcel in this frame."),
            (
                "sh020_parcel_custody",
                "I cannot tell — there is nothing to hold.",
            ),
            (
                "sh020_action",
                "There is no person, so they have not reached the bench.",
            ),
            ("sh020_cut", "A woodworking workshop with nobody in it."),
        ],
    );

    // --- wrong location ---
    case_answers(
        &mut answers,
        "sh020_wrong_location",
        3,
        &[
            ("sh020_location", "No, this is a domestic kitchen."),
            ("sh020_action", "The person has not reached any workbench."),
            ("sh020_cut", "A kitchen; the person wears a blue jacket."),
        ],
    );

    // --- unfinished action ---
    case_answers(
        &mut answers,
        "sh030_unfinished_action",
        3,
        &[
            ("sh030_parcel_custody", "Someone is still holding it."),
            (
                "sh030_action",
                "No, the hands are still gripping the parcel.",
            ),
        ],
    );

    // --- occluded handoff: custody is honestly unobserved, but the action is OVERCLAIMED ---
    case_answers(
        &mut answers,
        "sh030_occluded_handoff",
        3,
        &[
            (
                "sh030_parcel_custody",
                "I cannot tell — the parcel is behind the courier.",
            ),
            ("sh030_action", "The hands have let go and left the frame."),
        ],
    );

    // --- cut discontinuity ---
    case_answers(
        &mut answers,
        "sh030_cut_discontinuity",
        3,
        &[("sh030_cut", "A wooden workbench with no parcel on it.")],
    );

    // --- wrong costume ---
    case_answers(
        &mut answers,
        "sh050_wrong_costume",
        3,
        &[("sh050_recipient_apron", "She is wearing a blue jacket.")],
    );

    let mut vision = ScriptedVision::new();
    for (key, answer) in &answers {
        vision = vision.answer(key, answer);
    }
    vision
}

#[tokio::test]
async fn the_labeled_evaluation_reports_detections_misses_false_alarms_and_overclaims() {
    let harness = Harness::start(false, Vec::new()).await;
    let temp = tempfile::tempdir().expect("temp dir");
    let options = EvalOptions::new(PathBuf::from(EVAL_SET), temp.path().join("eval"));
    let results = review::review_eval(&harness.transport, &options, &eval_backend())
        .await
        .expect("the evaluation runs");

    assert_eq!(results.set_id, "courier-review-eval");
    assert!(!results.backend.real_model_inference);
    assert!(results.totals.scored > 0);

    let question = |id: &str| {
        results
            .per_question
            .iter()
            .find(|score| score.question_id == id)
            .unwrap_or_else(|| panic!("{id} is not in the report"))
    };
    let case = |id: &str| {
        results
            .per_case
            .iter()
            .find(|case| case.case_id == id)
            .unwrap_or_else(|| panic!("{id} is not in the report"))
    };

    // The deliberate faults the reviewer DID catch.
    assert_eq!(case("sh020_missing_courier").counts.detections, 4);
    assert_eq!(case("sh020_wrong_location").counts.detections, 3);
    assert_eq!(case("sh030_unfinished_action").counts.detections, 2);
    assert_eq!(case("sh030_cut_discontinuity").counts.detections, 1);
    assert_eq!(case("sh050_wrong_costume").counts.detections, 1);

    // The miss: the wrong parcel colour read as red.
    assert_eq!(question("sh030_parcel").counts.misses, 1);
    assert_eq!(case("sh030_wrong_parcel_colour").counts.misses, 1);
    assert_eq!(case("sh030_wrong_parcel_colour").counts.detections, 1);

    // The false alarm: a correct location called a kitchen.
    assert_eq!(question("sh020_location").counts.false_alarms, 1);
    assert_eq!(case("sh020_correct").counts.false_alarms, 1);

    // The overclaim: an occluded handoff answered confidently. This is the count that exists so an
    // uncertain observation quietly becoming a fact cannot hide inside "detections".
    assert_eq!(question("sh030_action").counts.overclaims, 1);
    assert_eq!(case("sh030_occluded_handoff").counts.overclaims, 1);
    // And the honest half of that same case is scored correct — including the custody question the
    // reviewer DID abstain on, which is agreement with an `unobserved` label, not a failure.
    assert_eq!(case("sh030_occluded_handoff").counts.correct, 5);
    assert_eq!(case("sh030_occluded_handoff").counts.scored, 6);

    assert_eq!(results.totals.misses, 1);
    assert_eq!(results.totals.false_alarms, 1);
    assert_eq!(results.totals.overclaims, 1);
    assert_eq!(
        results.totals.detections,
        results
            .per_case
            .iter()
            .map(|c| c.counts.detections)
            .sum::<u32>()
    );
    assert_eq!(
        results.totals.scored,
        results
            .per_case
            .iter()
            .map(|c| c.counts.scored)
            .sum::<u32>()
    );

    // Per-topic rollup covers every topic the questions use.
    assert!(results.per_topic.contains_key("parcel_custody"));
    assert!(results.per_topic.contains_key("action_completion"));
    assert!(results.per_topic.contains_key("cut_continuity"));

    // Files on disk: the results, the report, and one observed-state document per case, all kept
    // as the evaluation's evidence.
    let report = std::fs::read_to_string(temp.path().join("eval/review-eval.txt"))
        .expect("the report is written");
    assert!(
        report.contains("ASSISTIVE, not quality assurance"),
        "the report must never read as a quality bar: {report}"
    );
    assert!(report.contains("no model weights ran"), "{report}");
    assert!(report.contains("detections"), "{report}");
    assert!(
        report.contains("overclaims") || report.contains("over="),
        "{report}"
    );
    assert!(temp.path().join("eval/review-eval.json").exists());
    for case in &results.per_case {
        let observed = review::read_observed_state(Path::new(&case.observed_state_path))
            .expect("the per-case observed state reads");
        assert_eq!(observed.notice, ASSISTIVE_NOTICE);
        for observation in &observed.observations {
            assert!(observation.is_well_formed(), "{observation:?}");
        }
    }
}

#[tokio::test]
async fn the_evaluation_refuses_a_labeled_set_whose_media_is_not_on_this_host() {
    let harness = Harness::start(false, Vec::new()).await;
    let temp = tempfile::tempdir().expect("temp dir");
    let mut options = EvalOptions::new(PathBuf::from(EVAL_SET), temp.path().join("eval"));
    options.media_root = Some(temp.path().join("nowhere"));
    let error = review::review_eval(&harness.transport, &options, &ScriptedVision::new())
        .await
        .expect_err("missing media is a refusal, not a silent zero");
    assert!(format!("{error}").contains("--media-root"), "{error}");
}

#[tokio::test]
async fn an_evaluation_backend_that_answers_nothing_scores_misses_never_matches() {
    let harness = Harness::start(false, Vec::new()).await;
    let temp = tempfile::tempdir().expect("temp dir");
    let options = EvalOptions::new(PathBuf::from(EVAL_SET), temp.path().join("eval"));
    // The default scripted fallback is an explicit "I cannot tell".
    let results = review::review_eval(&harness.transport, &options, &ScriptedVision::new())
        .await
        .expect("the evaluation runs");
    assert_eq!(
        results.totals.detections, 0,
        "a backend that sees nothing catches nothing"
    );
    assert_eq!(
        results.totals.false_alarms, 0,
        "and it cries wolf about nothing either"
    );
    assert_eq!(
        results.totals.overclaims, 0,
        "silence is never an overclaim"
    );
    assert!(results.totals.misses > 0, "every planted fault is missed");
    assert!(results.totals.abstentions > 0);
    for case in &results.per_case {
        let observed = review::read_observed_state(Path::new(&case.observed_state_path))
            .expect("observed state reads");
        assert!(observed
            .observations
            .iter()
            .all(|observation| observation.unobserved && observation.observed.is_none()));
    }
}
