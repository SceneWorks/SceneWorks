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
use crate::tests::support::{huggingface_repo_cache_path, isolate_hf_cache, request, StatusCode};

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

/// The shipped plan and pack with their SOUND stripped, copied into the harness temp dir.
///
/// Nothing in this suite is about sound, and importing a sound clip transcodes it through an
/// ffmpeg the hosted macOS lane does not have (`ProjectStore::import_asset` -> `transcode_to_wav_pcm16`,
/// sc-22712) — a review test that needs a rendered run would otherwise fail on the upload before
/// it ever put a question to anything. The pictures, the shots, the dependsOn edges and every
/// review question are unchanged, so these tests mean exactly what they meant before.
fn sound_free_documents(harness: &Harness) -> (PathBuf, PathBuf) {
    (
        harness.edited_plan(|_| {}),
        harness.fixture_pack_without_sound(),
    )
}

fn shipped_review_plan() -> ReviewPlan {
    parse_review_plan(&std::fs::read_to_string(REVIEW_PLAN).expect("review plan reads"))
        .expect("review plan parses")
}

/// Seed the smallest structurally complete SenseNova tier and the MiniMax partition needed by a
/// routed repair. Review tests exercise routing and lifecycle with a scripted CPU worker; they
/// must not borrow a developer's installed weights or become red when CI supplies both HF cache
/// variables as empty directories.
fn seed_review_models(harness: &Harness) {
    let write = |path: &std::path::Path, bytes: &[u8]| {
        std::fs::create_dir_all(path.parent().unwrap()).expect("fixture model directory");
        std::fs::write(path, bytes).expect("fixture model file");
    };
    let review_root = huggingface_repo_cache_path(
        &harness.temp_dir.path().join("data"),
        "SceneWorks/sensenova-u1-8b-mlx",
    )
    .expect("fixture repo path")
    .join("snapshots/fixture/q4");
    for file in ["model.safetensors", "config.json", "tokenizer.json"] {
        write(&review_root.join(file), b"fixture");
    }
    let tier_root = huggingface_repo_cache_path(
        &harness.temp_dir.path().join("data"),
        "SceneWorks/minimax-h3-mlx",
    )
    .expect("fixture video repo path")
    .join("snapshots/fixture/q4");
    let index = br#"{"weight_map":{"fixture":"model-00001-of-00001.safetensors"}}"#;
    for partition in ["transformer", "transformer_ref"] {
        write(&tier_root.join(partition).join("config.json"), b"{}");
        write(
            &tier_root
                .join(partition)
                .join("diffusion_pytorch_model.safetensors.index.json"),
            index,
        );
        write(
            &tier_root
                .join(partition)
                .join("model-00001-of-00001.safetensors"),
            b"fixture",
        );
    }
    write(&tier_root.join("text_encoder/config.json"), b"{}");
    write(
        &tier_root.join("text_encoder/model.safetensors.index.json"),
        index,
    );

    let shared_root = huggingface_repo_cache_path(
        &harness.temp_dir.path().join("data"),
        "MiniMaxAI/MiniMax-H3",
    )
    .expect("fixture video shared repo path")
    .join("snapshots/fixture");
    for file in sceneworks_core::mlx_tier_completeness::MINIMAX_H3_SHARED_PROBED_FILES {
        write(&shared_root.join(file), b"fixture");
    }
    for file in sceneworks_core::mlx_tier_completeness::MINIMAX_H3_AUDIO_VAE_CONFIG_FILES {
        write(&shared_root.join("FL2VA/audio_vae").join(file), b"fixture");
    }
}

/// A run of SH010 + SH020 that completed, ready to review.
async fn rendered_two_shots() -> (Harness, RunRecord) {
    let harness = Harness::start(true, fast(&["SH010", "SH020"])).await;
    let (plan, pack) = sound_free_documents(&harness);
    let options = harness.options(plan, pack, Some(&["SH010", "SH020"]));
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
    // Every shipped question is CLOSED with a declared answer vocabulary (sc-22714, after the
    // real-weights smoke), so these are the short answers the model actually gives.
    vec![
        ("sh010_location", "workshop"),
        ("sh010_courier", "Yes, a person is standing in the doorway."),
        ("sh010_courier_jacket", "blue"),
        ("sh010_parcel", "red"),
        ("sh010_parcel_custody", "hands"),
        ("sh010_action", "open"),
        ("sh020_location", "workshop"),
        ("sh020_courier", "Yes, a person is walking across the room."),
        ("sh020_courier_jacket", "blue"),
        ("sh020_parcel", "red"),
        ("sh020_parcel_custody", "hands"),
        ("sh020_action", "Yes, they are standing at the workbench."),
        // The comparative cut question: the same answer on both sides means the cut holds.
        ("sh020_cut", "yes"),
    ]
}

fn observed_for(record: &RunRecord, out_dir: &Path, shot_id: &str) -> ObservedState {
    let shot = record.shot(shot_id).expect("shot is in the record");
    let summary = shot.latest_review().expect("the shot has a review");
    review::read_observed_state(&out_dir.join(&summary.record_path))
        .expect("the observed-state document reads")
}

fn register_run_for_routes(harness: &Harness, record: &RunRecord) -> (String, String) {
    let project_id = record.project_id.clone().expect("run has project");
    let project_path = PathBuf::from(record.project_path.as_ref().expect("run has project path"));
    let run_id = record.run_id.clone();
    let relative = format!("films/runs/{run_id}");
    let run_dir = project_path.join(&relative);
    std::fs::create_dir_all(run_dir.parent().expect("run parent")).expect("run parent creates");
    std::fs::rename(harness.out_dir(), &run_dir).expect("completed run moves into project");
    std::fs::copy(REVIEW_PLAN, run_dir.join("review.jsonc")).expect("review plan pins");
    std::fs::write(
        run_dir.join("locator.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schemaVersion": 1,
            "id": run_id,
            "projectId": project_id,
            "draftId": "film_route_fixture",
            "draftRevision": 1,
            "selectedShotIds": record.selected_shot_ids,
            "recordDirectory": relative,
            "createdAt": record.created_at,
        }))
        .expect("locator serializes"),
    )
    .expect("locator writes");
    (project_id, run_id)
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
    // ...and EVERY shot asks one. The parcel is the thing the whole sequence is about, so a shot
    // with no custody question is a shot in which it can change hands unasked — SH050 was exactly
    // that, and the gap is invisible in a report that only counts the questions that exist.
    for (shot_id, spec) in &review.shots {
        assert!(
            spec.questions
                .iter()
                .any(|question| question.topic == "parcel_custody"),
            "{shot_id} asks no parcel_custody question, so nothing checks who has the parcel"
        );
    }
    // Both token and memory ceilings are declared, not inherited from a constant somewhere.
    assert!(review.limits.max_new_tokens >= 1);
    assert_eq!(
        review.limits.max_memory_gb, 16.0,
        "the review declares SenseNova-U1-8B's own minMemoryGb as its ceiling"
    );

    for path in [EVAL_SET, REAL_TAKES_SET] {
        let set: EvalSet = parse_eval_set(&std::fs::read_to_string(path).expect("set reads"))
            .unwrap_or_else(|error| panic!("{path}: {error}"));
        let findings = validate_eval_set(&set, &review);
        assert!(findings.is_empty(), "{path}: {findings:#?}");
    }
}

/// EVERY labeled set checked into `config/film-harness/review-eval/` parses and validates against
/// the review plan it declares (sc-22715 adversarial review).
///
/// The test above names two files by hand, so the two evaluation sets added by this story — and
/// any set added later — were checked in with nothing loading them: a typo, a question id that
/// stopped existing, or a frame path that climbs out of the media root would have been found by
/// the first person who ran `review-eval`, months later. The set is the DIRECTORY, and the count
/// is asserted so an enumeration that silently finds nothing cannot pass.
#[test]
fn every_checked_in_labeled_set_parses_and_validates_against_the_plan_it_declares() {
    let dir = PathBuf::from(EVAL_SET)
        .parent()
        .expect("the review-eval directory")
        .to_path_buf();
    let mut sets: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("{} does not read: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonc"))
        .collect();
    sets.sort();
    let names: Vec<String> = sets
        .iter()
        .map(|path| {
            path.file_name()
                .expect("a name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    for required in [
        "evaluation-2026-09-14-planner-takes.jsonc",
        "evaluation-2026-09-14-takes.jsonc",
        "labels.jsonc",
        "real-takes.jsonc",
    ] {
        assert!(
            names.iter().any(|name| name == required),
            "{required} is checked in but the enumeration did not reach it: {names:?}"
        );
    }

    for path in &sets {
        let set: EvalSet = parse_eval_set(&std::fs::read_to_string(path).expect("the set reads"))
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        // Each set names its OWN review plan, relative to the document — the same resolution
        // `review-eval` does — so a set written against a different plan is still checked against
        // the questions it actually scores.
        let plan_path = path
            .parent()
            .expect("the set's directory")
            .join(&set.review_plan);
        let plan_text = std::fs::read_to_string(&plan_path).unwrap_or_else(|error| {
            panic!(
                "{}: its reviewPlan {:?} does not read ({}): {error}",
                path.display(),
                set.review_plan,
                plan_path.display()
            )
        });
        let plan = parse_review_plan(&plan_text)
            .unwrap_or_else(|error| panic!("{}: {error}", plan_path.display()));
        let findings = validate_eval_set(&set, &plan);
        assert!(findings.is_empty(), "{}: {findings:#?}", path.display());
        // A labeled frame is a NAME under the set's media root, never a path of its own; the
        // frames themselves live outside the repository for the evaluation sets, so this is the
        // part of "the set is usable" that can be asserted without the media.
        for case in &set.cases {
            for frame in case.frames.iter().chain(case.adjacent_frames.iter()) {
                assert!(
                    sceneworks_core::film_review::unsafe_media_path(&frame.file).is_none(),
                    "{}: case {:?} names frame {:?}, which is not a plain name under the set's \
                     mediaRoot",
                    path.display(),
                    case.id,
                    frame.file
                );
            }
        }
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
        .preflight(shipped_review_plan().limits)
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
    let (plan, pack) = sound_free_documents(&harness);
    let options = harness.options(plan, pack, Some(&["SH010"]));
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
async fn an_across_cut_question_with_no_neighbour_is_recorded_unobserved_with_its_reason() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    // SH010's take is rejected, so its selection is cleared and SH020 — which declares a continuity
    // edge to it — has nothing on the other side of its cut to be compared against.
    review::decide_take(
        &harness.out_dir(),
        "SH010",
        Decision::Reject,
        "the doorway is too dark",
    )
    .expect("reject records");

    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let record = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");
    let observed = observed_for(&record, &harness.out_dir(), "SH020");
    assert!(
        observed.adjacent.is_none(),
        "there is no adjacent selected take: {:?}",
        observed.adjacent
    );

    // The declared question is in the document, saying WHY nothing was read. Skipping it silently
    // left the document indistinguishable from one where nobody declared the question at all.
    let asked: Vec<&str> = observed
        .observations
        .iter()
        .map(|observation| observation.question_id.as_str())
        .collect();
    let plan = shipped_review_plan();
    let declared: Vec<&str> = plan.shots["SH020"]
        .questions
        .iter()
        .map(|question| question.id.as_str())
        .collect();
    assert_eq!(asked, declared, "every declared question is accounted for");
    let cut = observed
        .observations
        .iter()
        .find(|observation| observation.question_id == "sh020_cut")
        .expect("the cut question is recorded even though nobody could ask it");
    assert_eq!(cut.verdict, Verdict::Unobserved);
    assert!(cut.unobserved);
    assert!(cut.observed.is_none(), "and it claims nothing: {cut:?}");
    assert_eq!(cut.confidence, 0.0);
    assert!(
        cut.answers.is_empty(),
        "nothing was asked of any backend: {cut:?}"
    );
    assert!(
        cut.note
            .as_deref()
            .unwrap_or_default()
            .contains("no adjacent selected take"),
        "the reason has to be in the document: {cut:?}"
    );
    // It is not a mismatch either — a cut nobody could look at is not a discontinuous cut.
    assert!(
        !observed
            .mismatches
            .iter()
            .any(|flag| flag.question_id == "sh020_cut"),
        "{:#?}",
        observed.mismatches
    );
    // And the whole document still holds the invariant the module exists for.
    for observation in &observed.observations {
        assert!(observation.is_well_formed(), "{observation:?}");
    }
    // The serialized form carries the note and no value.
    let raw: Value = serde_json::from_str(
        &std::fs::read_to_string(
            harness.out_dir().join(
                &record
                    .shot("SH020")
                    .expect("shot")
                    .latest_review()
                    .expect("review")
                    .record_path,
            ),
        )
        .expect("the document reads"),
    )
    .expect("the document is json");
    let cut_json = raw["observations"]
        .as_array()
        .expect("observations")
        .iter()
        .find(|observation| observation["questionId"] == "sh020_cut")
        .expect("the cut question is written out");
    assert!(cut_json.get("observed").is_none(), "{cut_json}");
    assert_eq!(cut_json["unobserved"], serde_json::json!(true));
    assert!(cut_json["note"].as_str().is_some(), "{cut_json}");
}

#[tokio::test]
async fn a_malformed_observed_state_document_is_refused_rather_than_read() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    let options = review_options(&harness, &["SH020"]);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let record = review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");
    let path = harness.out_dir().join(
        &record
            .shot("SH020")
            .expect("shot")
            .latest_review()
            .expect("review")
            .record_path,
    );
    review::read_observed_state(&path).expect("a document this build wrote reads back");

    // The pair (unobserved, observed) is independently settable in the serialized shape, so a
    // hand-edited or foreign document CAN claim that an unseen handoff completed. Reading one and
    // folding its flags into a repair would launder that into a fact.
    let mut document: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("reads")).expect("json");
    let observation = document["observations"]
        .as_array_mut()
        .expect("observations")
        .iter_mut()
        .find(|observation| observation["questionId"] == "sh020_parcel_custody")
        .expect("the custody question");
    observation["unobserved"] = serde_json::json!(true);
    observation["verdict"] = serde_json::json!("unobserved");
    observation["observed"] = serde_json::json!("the courier let go of the parcel");
    std::fs::write(&path, document.to_string()).expect("the tampered document writes");

    let error = review::read_observed_state(&path)
        .expect_err("an unobserved observation carrying a value is not a usable document");
    let message = format!("{error}");
    assert!(
        message.contains("no such thing as an unobserved value"),
        "{message}"
    );
    assert!(
        message.contains("sh020_parcel_custody"),
        "the refusal must name the observation: {message}"
    );
}

#[tokio::test]
async fn the_declared_token_ceiling_is_what_reaches_the_vqa_job() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    // A review plan identical to the shipped one except for its declared token ceiling. The value
    // the backend asks for must come from the DOCUMENT — it used to be a const in the caller, which
    // no reader of the review plan could see and no author could change.
    let temp = tempfile::tempdir().expect("temp dir");
    let plan_text = std::fs::read_to_string(REVIEW_PLAN).expect("review plan reads");
    let narrowed = plan_text.replace("\"maxNewTokens\": 192", "\"maxNewTokens\": 48");
    assert_ne!(
        narrowed, plan_text,
        "the shipped plan declares maxNewTokens"
    );
    let narrowed_path = temp.path().join("review.jsonc");
    std::fs::write(&narrowed_path, narrowed).expect("the narrowed plan writes");

    let mut options = review_options(&harness, &["SH020"]);
    options.review_plan_path = Some(narrowed_path);
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    review::review(&harness.transport, &options, &vision)
        .await
        .expect("the review runs");

    let vqa: Vec<Value> = harness
        .jobs()
        .await
        .into_iter()
        .filter(|job| job["type"] == "image_vqa")
        .collect();
    assert!(!vqa.is_empty(), "the review asked something");
    for job in &vqa {
        assert_eq!(
            job.pointer("/payload/maxNewTokens").and_then(Value::as_u64),
            Some(48),
            "every question is bounded by the document's own ceiling: {job}"
        );
    }
}

#[tokio::test]
async fn a_labeled_set_whose_frames_climb_out_of_the_media_root_is_refused_before_any_file_is_touched(
) {
    let harness = Harness::start(false, Vec::new()).await;
    let temp = tempfile::tempdir().expect("temp dir");
    let set_path = temp.path().join("escaping.jsonc");
    let review_plan = PathBuf::from(REVIEW_PLAN);
    std::fs::write(
        &set_path,
        serde_json::json!({
            "schemaVersion": 1, "id": "escaping-set", "version": 1,
            "mediaRoot": temp.path().join("frames").display().to_string(),
            "reviewPlan": review_plan.display().to_string(),
            "cases": [
                {
                    "id": "escapes", "shotId": "SH030", "label": "wrong_parcel_colour",
                    "frames": [{ "file": "../../../../etc/passwd", "timestampSeconds": 0.0 }],
                    "expected": { "sh030_parcel": "mismatch" }
                },
                {
                    "id": "honest", "shotId": "SH030", "label": "correct",
                    "frames": [{ "file": "a.png", "timestampSeconds": 0.0 }],
                    "expected": { "sh030_parcel": "match" }
                }
            ]
        })
        .to_string(),
    )
    .expect("the set writes");

    // Reading side: the evaluation refuses the document rather than opening the file it names.
    let options = EvalOptions::new(set_path.clone(), temp.path().join("eval"));
    let error = review::review_eval(&harness.transport, &options, &ScriptedVision::new())
        .await
        .expect_err("a frame path that climbs out of the media root is not a frame path");
    let message = format!("{error}");
    assert!(message.contains("escapes"), "{message}");
    assert!(message.contains("`..` component"), "{message}");

    // Writing side, which matters more: `review-fixtures` creates a file per named frame.
    let error = review::write_review_fixture_frames(&set_path, None)
        .expect_err("review-fixtures must not write outside the media root either");
    assert!(format!("{error}").contains("escapes"), "{error}");
    assert!(
        !Path::new("/etc/passwd.png").exists(),
        "nothing was written outside the root"
    );

    // The same set with an absolute frame is refused too...
    let absolute = std::fs::read_to_string(&set_path)
        .expect("reads")
        .replace("../../../../etc/passwd", "/etc/hosts");
    std::fs::write(&set_path, absolute).expect("writes");
    let error = review::write_review_fixture_frames(&set_path, None)
        .expect_err("an absolute frame path is not a name under the root");
    assert!(format!("{error}").contains("not an absolute"), "{error}");

    // ...and the shipped set, whose frames are plain relative names, still writes.
    review::write_review_fixture_frames(&PathBuf::from(EVAL_SET), Some(temp.path()))
        .expect("the shipped labeled set is fine");
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
async fn review_routes_expose_decisions_preflight_and_one_bounded_repair_without_auto_acceptance() {
    let _env = isolate_hf_cache();
    let harness = Harness::start_http(true, fast(&["SH010", "SH020"])).await;
    seed_review_models(&harness);
    let (status, models) = request(harness.app.clone(), "GET", "/api/v1/models", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{models}");
    for model_id in ["minimax_h3", "sensenova_u1_8b"] {
        let model = models
            .as_array()
            .expect("model catalog")
            .iter()
            .find(|model| model["id"] == model_id)
            .unwrap_or_else(|| panic!("missing {model_id}: {models}"));
        assert_eq!(model["installState"], "installed", "{model}");
    }
    let (plan, pack) = sound_free_documents(&harness);
    let record = film_harness::run(
        &harness.transport,
        &harness.options(plan, pack, Some(&["SH010", "SH020"])),
    )
    .await
    .expect("the two-shot run completes");
    script_answers(&harness, &agreeing_answers());
    let original_attempts = record.shot("SH010").expect("shot").attempts.len();
    let (project_id, run_id) = register_run_for_routes(&harness, &record);
    let review_path = format!("/api/v1/projects/{project_id}/film-runs/{run_id}/review");

    let (status, view) = request(harness.app.clone(), "GET", &review_path, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{view}");
    assert_eq!(view["selections"][0]["state"], "aligned");
    assert_eq!(
        view["run"]["record"]["shots"][0]["attempts"]
            .as_array()
            .expect("take history")
            .len(),
        original_attempts
    );

    let (status, accepted) = request(
        harness.app.clone(),
        "POST",
        &format!("{review_path}/decision"),
        serde_json::json!({"shotId": "SH010", "decision": "accept", "reason": "Human approved"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{accepted}");
    assert_eq!(
        accepted["run"]["record"]["shots"][0]["humanDecision"]["state"],
        "accepted"
    );

    let (status, reviewing) = request(
        harness.app.clone(),
        "POST",
        &review_path,
        serde_json::json!({"shotIds": ["SH010"]}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{reviewing}");
    assert!(reviewing["actionDisabledReason"].is_string(), "{reviewing}");
    let mut reviewed = Value::Null;
    for _ in 0..300 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let (status, current) =
            request(harness.app.clone(), "GET", &review_path, Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{current}");
        if current["run"]["controllerActive"] == serde_json::json!(false) {
            reviewed = current;
            break;
        }
    }
    assert_ne!(reviewed, Value::Null, "bounded review did not settle");
    assert_eq!(
        reviewed["observations"]
            .as_array()
            .expect("observations")
            .len(),
        1
    );
    assert!(reviewed["reviewTimelineId"].is_string(), "{reviewed}");
    assert_eq!(
        reviewed["run"]["record"]["shots"][0]["humanDecision"]["state"], "accepted",
        "advisory review cannot overwrite the human decision"
    );

    let (status, started) = request(
        harness.app.clone(),
        "POST",
        &format!("{review_path}/repair"),
        serde_json::json!({"shotId": "SH010", "reason": "Repair the parcel handoff"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{started}");
    assert!(started["actionDisabledReason"].is_string(), "{started}");

    let mut settled = Value::Null;
    for _ in 0..80 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let (status, current) =
            request(harness.app.clone(), "GET", &review_path, Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{current}");
        if current["run"]["controllerActive"] == serde_json::json!(false) {
            settled = current;
            break;
        }
    }
    assert_ne!(settled, Value::Null, "bounded repair did not settle");
    let shot = &settled["run"]["record"]["shots"][0];
    assert_eq!(
        shot["attempts"].as_array().expect("attempt history").len(),
        original_attempts + 1,
        "repair appends exactly one attempt: {settled}"
    );
    assert!(
        shot.get("humanDecision").is_none(),
        "repair cannot auto-accept: {shot}"
    );
    assert_eq!(
        settled["run"]["record"]["decisions"]
            .as_array()
            .expect("decision history")
            .iter()
            .filter(|decision| decision["action"] == "request_repair")
            .count(),
        1
    );
}

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
    assert!(
        rejected
            .export
            .as_ref()
            .expect("the two-shot run exported an MP4")
            .stale,
        "the MP4 no longer matches the selected takes"
    );
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
async fn accepting_a_rejected_take_is_refused_and_changes_nothing() {
    let (harness, _) = rendered_two_shots().await;
    let out_dir = harness.out_dir();
    let rejected = review::decide_take(
        &out_dir,
        "SH010",
        Decision::Reject,
        "the courier's jacket is the wrong blue",
    )
    .expect("reject records");
    assert!(rejected
        .shot("SH010")
        .expect("shot")
        .selected_attempt
        .is_none());

    // Accepting it would re-select the very attempt this run threw away — rejection record and all
    // — clear THIS shot's needsReview flags, and leave its dependents flagged and the export stale.
    let error = review::decide_take(&out_dir, "SH010", Decision::Accept, "actually it's fine")
        .expect_err("a rejected take cannot be accepted back into the cut");
    let message = format!("{error}");
    assert!(message.contains("was rejected at"), "{message}");
    assert!(
        message.contains("the courier's jacket is the wrong blue"),
        "the refusal must quote the rejection it is refusing to undo: {message}"
    );
    assert!(
        message.contains("replace-take") && message.contains("swap-take"),
        "the refusal must say what to do instead: {message}"
    );

    // Nothing moved: not the selection, not the decision, not the dependents, not the export.
    let after = harness_record(&harness);
    assert_eq!(after, rejected, "a refused decision writes nothing");
    let sh010 = after.shot("SH010").expect("shot");
    assert!(sh010.selected_attempt.is_none());
    assert_eq!(
        sh010.human_decision.as_ref().map(|d| d.state.as_str()),
        Some("rejected"),
        "the rejection stands"
    );
    assert_eq!(after.shot("SH020").expect("shot").needs_review.len(), 1);
    assert!(after.export.as_ref().expect("the run exported").stale);

    // The bounded repair — the verb the refusal points at — still works from here.
    let mut resume = harness.resume_options();
    resume.export = false;
    let repaired = review::request_repair(&harness.transport, &resume, "SH010", "re-render it")
        .await
        .expect("the repair runs");
    let sh010 = repaired.shot("SH010").expect("shot");
    let selected = sh010.selected_attempt.expect("the repair selects its take");
    assert!(
        sh010
            .attempts
            .iter()
            .find(|attempt| attempt.attempt == selected)
            .expect("the selected attempt")
            .rejection
            .is_none(),
        "the take now in the cut is not a rejected one"
    );
    // And accepting THAT one is fine.
    review::decide_take(&out_dir, "SH010", Decision::Accept, "the new take is good")
        .expect("a live take accepts");
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
    // The shipped questions are closed with declared answer vocabularies (sc-22714, after the
    // real-weights smoke), so these are one-word answers agreeing with the plan. Anything a case
    // does not override keeps these.
    set("sh020_location", "workshop");
    set("sh020_courier", "yes");
    set("sh020_courier_jacket", "blue");
    set("sh020_parcel", "red");
    set("sh020_parcel_custody", "hands");
    set("sh020_action", "yes");
    set("sh020_cut", "yes");
    set("sh030_bench", "wood");
    set("sh030_sleeve", "blue");
    set("sh030_parcel", "red");
    set("sh030_parcel_custody", "surface");
    set("sh030_action", "no");
    set("sh030_cut", "red");
    set("sh050_location", "workshop");
    set("sh050_recipient", "yes");
    set("sh050_recipient_apron", "apron");
    set("sh050_parcel", "red");
    set("sh050_parcel_custody", "surface");
    set("sh050_action", "yes");
    set("sh050_cut", "yes");
    set("sh060_bench", "wood");
    set("sh060_apron_sleeve", "grey");
    set("sh060_parcel", "red");
    set("sh060_parcel_custody", "yes");
    set("sh060_action", "open");
    set("sh060_cut", "red");
    answers
}

/// Rewrite the per-question answers for the frames of one case. `frames` covers the case's OWN
/// frames; the adjacent frame of a comparative cut question is keyed separately, so a test can
/// make the two sides of a cut disagree.
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

/// What the take on the OTHER side of the cut answers for a comparative question.
fn adjacent_answer(
    answers: &mut BTreeMap<String, String>,
    case_id: &str,
    question: &str,
    answer: &str,
) {
    answers.insert(format!("{question}@{case_id}-adjacent"), answer.to_owned());
}

/// The whole answer table the labeled evaluation is scored against, keyed exactly as BOTH backends
/// key a question: `<question id>@<frame id>` first, then `<question id>`. `ScriptedVision` reads
/// it directly; the fake worker's `image_vqa` job reads the same keys off the tag the reviewer
/// stamps into the question text, so the two backends can be driven with one table (sc-22715).
fn eval_answer_table() -> BTreeMap<String, String> {
    let mut answers = eval_answers();
    // Every comparative cut question needs its OTHER side answered. By default the neighbour
    // agrees, so only the case that plants a discontinuity disagrees.
    for (case, question, answer) in [
        ("sh020_correct", "sh020_cut", "yes"),
        ("sh020_missing_courier", "sh020_cut", "yes"),
        ("sh020_wrong_location", "sh020_cut", "yes"),
        ("sh030_correct", "sh030_cut", "red"),
        ("sh030_wrong_parcel_colour", "sh030_cut", "red"),
        ("sh030_unfinished_action", "sh030_cut", "red"),
        ("sh030_occluded_handoff", "sh030_cut", "red"),
        ("sh030_cut_discontinuity", "sh030_cut", "red"),
        ("sh050_correct", "sh050_cut", "yes"),
        ("sh050_wrong_costume", "sh050_cut", "yes"),
        ("sh060_correct", "sh060_cut", "red"),
    ] {
        adjacent_answer(&mut answers, case, question, answer);
    }

    // --- correct cases keep the agreeing answers, except one deliberate FALSE ALARM ---
    case_answers(
        &mut answers,
        "sh020_correct",
        3,
        &[("sh020_location", "kitchen")],
    );

    // --- wrong parcel colour: the reviewer catches the cut but MISSES the colour itself ---
    case_answers(
        &mut answers,
        "sh030_wrong_parcel_colour",
        3,
        &[("sh030_parcel", "red"), ("sh030_cut", "blue")],
    );

    // --- missing character ---
    case_answers(
        &mut answers,
        "sh020_missing_courier",
        3,
        &[
            ("sh020_courier", "no"),
            (
                "sh020_courier_jacket",
                "I cannot tell — there is nobody here.",
            ),
            ("sh020_parcel", "none"),
            (
                "sh020_parcel_custody",
                "I cannot tell — there is nothing to hold.",
            ),
            ("sh020_action", "no"),
            ("sh020_cut", "no"),
        ],
    );

    // --- wrong location ---
    case_answers(
        &mut answers,
        "sh020_wrong_location",
        3,
        &[
            ("sh020_location", "kitchen"),
            ("sh020_action", "no"),
            ("sh020_cut", "no"),
        ],
    );

    // --- unfinished action ---
    case_answers(
        &mut answers,
        "sh030_unfinished_action",
        3,
        &[("sh030_parcel_custody", "hands"), ("sh030_action", "yes")],
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
            ("sh030_action", "no"),
        ],
    );

    // --- cut discontinuity: the take is internally fine, but the parcel it opens on is not the
    //     one the previous take carried to the bench. ONLY the comparison can see this — asking
    //     each frame independently whether it holds a red parcel answers "yes" on both sides.
    case_answers(
        &mut answers,
        "sh030_cut_discontinuity",
        3,
        &[("sh030_cut", "none")],
    );

    // --- wrong costume ---
    case_answers(
        &mut answers,
        "sh050_wrong_costume",
        3,
        &[("sh050_recipient_apron", "jacket")],
    );

    answers
}

fn eval_backend() -> ScriptedVision {
    let mut vision = ScriptedVision::new();
    for (key, answer) in &eval_answer_table() {
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
    // ...and that detection came from COMPARING the two sides of the cut, not from judging one
    // frame. Every frame of this case answers "red" on its own; only the neighbour disagrees.
    {
        let observed = review::read_observed_state(Path::new(
            &case("sh030_cut_discontinuity").observed_state_path,
        ))
        .expect("observed state reads");
        let cut = observed
            .observations
            .iter()
            .find(|observation| observation.question_id == "sh030_cut")
            .expect("the cut question was asked");
        assert_eq!(cut.verdict, Verdict::Mismatch);
        let named = cut.observed.as_deref().expect("a named difference");
        assert!(named.contains("this take reads"), "{named}");
        assert!(named.contains("the take it cuts from reads"), "{named}");
        assert!(
            cut.evidence_frame_ids
                .iter()
                .any(|id| id.ends_with("-adjacent")),
            "the comparison must cite the neighbour's frame: {:?}",
            cut.evidence_frame_ids
        );
        // Every graded answer records the token it matched and the polarity it was read with, so
        // a mis-grade is debuggable from the document alone.
        for answer in &cut.answers {
            assert!(
                ["affirmed", "negated", "none"].contains(&answer.polarity.as_str()),
                "{answer:?}"
            );
            // A decisive answer names the token it was decided by. An unobserved one names nothing
            // — unless it was SELF-CONTRADICTORY, where the conflict itself is the finding and the
            // record has to say which two values collided.
            match (answer.verdict, answer.matched.as_deref()) {
                (Verdict::Unobserved, None) => {}
                (Verdict::Unobserved, Some(matched)) => assert!(
                    matched.starts_with("conflict:"),
                    "an unobserved answer names no value unless it is a conflict: {answer:?}"
                ),
                (_, Some(_)) => {}
                (_, None) => panic!("a decisive answer must name its token: {answer:?}"),
            }
        }
    }
    assert_eq!(case("sh050_wrong_costume").counts.detections, 1);
    // SH050's custody question is labeled on BOTH its cases. It was the one shot of six with no
    // custody question at all, and an unlabeled question is not scored — so a coverage gap there is
    // invisible in a report that only counts what it was asked.
    assert_eq!(
        question("sh050_parcel_custody").counts.scored,
        2,
        "both SH050 cases label the custody question"
    );
    assert_eq!(
        question("sh050_parcel_custody").counts.correct,
        2,
        "the parcel sits on the bench in both, and the reviewer says so"
    );

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
async fn preflight_refuses_a_worker_row_that_advertises_image_vqa_but_is_offline() {
    // The sc-22714 smoke ran against a data dir seeded from an earlier run. Its GPU worker row
    // still advertised `image_vqa` with `status: "offline"`, the capability-only preflight passed
    // it, and the review then queued questions nothing would ever claim — the exact failure the
    // preflight exists to prevent.
    let harness = Harness::start(false, Vec::new()).await;
    let (status, _) = crate::tests::support::request(
        harness.app.clone(),
        "POST",
        "/api/v1/workers/register",
        serde_json::json!({
            "workerId": "stale-gpu",
            "gpuId": "mlx",
            "gpuName": "Apple M-series (stale)",
            "capabilities": ["image_vqa"],
            "loadedModels": [],
            // A registered worker reports its host's memory on every register AND heartbeat
            // (`sceneworks_worker`: `utilization: gpu_utilization(...)`), and that report is the
            // only thing `GET /api/v1/host-capabilities` has to answer with — so the memory half of
            // the preflight reads it from here.
            "utilization": { "memoryTotalMb": 128 * 1024 },
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let (status, _) = crate::tests::support::request(
        harness.app.clone(),
        "POST",
        "/api/v1/workers/stale-gpu/heartbeat",
        serde_json::json!({
            "status": "offline", "loadedModels": [],
            "utilization": { "memoryTotalMb": 128 * 1024 },
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);

    let vision = VqaVision::new(
        &harness.transport,
        Duration::from_millis(50),
        RunControl::new(),
    );
    let limits = shipped_review_plan().limits;
    let error = vision
        .preflight(limits)
        .await
        .expect_err("an offline worker cannot answer anything");
    let message = format!("{error}");
    assert!(message.contains("no live registered worker"), "{message}");
    assert!(
        message.contains("stale-gpu (offline)"),
        "the refusal must name the row that fooled the old check: {message}"
    );

    // A live row of the same shape passes.
    let (status, _) = crate::tests::support::request(
        harness.app.clone(),
        "POST",
        "/api/v1/workers/stale-gpu/heartbeat",
        serde_json::json!({
            "status": "idle", "loadedModels": [],
            "utilization": { "memoryTotalMb": 128 * 1024 },
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    vision
        .preflight(limits)
        .await
        .expect("an idle worker advertising image_vqa is accepted");

    // ...and the same live worker is refused when the review declares a ceiling the host cannot
    // meet. A review that starts on a host too small dies inside the loader AFTER creating a
    // project and extracting frames, or times out on every question — which records the whole take
    // as unobserved and reads like evidence.
    let mut hungry = limits;
    hungry.max_memory_gb = 4096.0;
    let error = vision
        .preflight(hungry)
        .await
        .expect_err("a 4 TB ceiling cannot be met by any host this test runs on");
    let message = format!("{error}");
    assert!(message.contains("limits.maxMemoryGb"), "{message}");
    assert!(
        message.contains("4096.0") && message.contains("the API host reports"),
        "the refusal must name both numbers: {message}"
    );

    // And a live worker whose heartbeat carries no utilization at all leaves the host reporting
    // nothing: an UNCHECKED ceiling is not a checked one, so that is a refusal too.
    let (status, _) = crate::tests::support::request(
        harness.app.clone(),
        "POST",
        "/api/v1/workers/stale-gpu/heartbeat",
        serde_json::json!({ "status": "idle", "loadedModels": [] }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    let error = vision
        .preflight(limits)
        .await
        .expect_err("a ceiling nobody can check is not a checked ceiling");
    assert!(
        format!("{error}").contains("no registered worker reports host memory"),
        "{error}"
    );
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

// ---------------------------------------------------------------------------------------------
// sc-22715: the two review limits that no test ever tripped, and the evaluation over the REAL
// image_vqa route.
// ---------------------------------------------------------------------------------------------

/// A review plan copy with `edit` applied to its limits, written beside the harness.
fn review_plan_with_limits(harness: &Harness, edit: impl FnOnce(&mut ReviewPlan)) -> PathBuf {
    let mut document = shipped_review_plan();
    edit(&mut document);
    let path = harness.temp_dir.path().join("review-limits.jsonc");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&document).expect("serializes"),
    )
    .expect("writes");
    path
}

/// (C) `limits.maxSeconds`: a review whose backend is slow spends its wall-clock budget partway
/// through the questions, stops with `review_budget`, and KEEPS the evidence it already has —
/// the frames it sampled and the questions it answered — rather than erroring out.
#[tokio::test]
async fn a_review_that_spends_its_wall_clock_budget_stops_with_review_budget_and_keeps_partial_evidence(
) {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    // 2 s per answer against an 8 s budget. The two bounds this test needs are both wide:
    // the frames plus ONE answer must fit (frames have ~6 s of room, against the fraction of a
    // second three extractions take even on a loaded runner), and all six answers must NOT
    // (6 x 2 s = 12 s, half again over the budget) — so the deadline always falls between
    // questions with evidence already in hand.
    //
    // It used to be 0.6 s answers against a 2 s budget, which left the frames ~1.4 s: enough on an
    // idle machine, and not enough on a busy one, where this test measured 0 of 6 answers and
    // failed (sc-23403). Its sibling above tolerates a loaded runner in its claims; this one
    // cannot, because "partial evidence is KEPT" is the thing it exists to prove — so the margin
    // has to be in the timings instead.
    harness.script.lock().vqa_delay = Some(Duration::from_secs(2));
    let mut options = review_options(&harness, &["SH010"]);
    options.review_plan_path = Some(review_plan_with_limits(&harness, |plan| {
        plan.limits.max_seconds = 8;
    }));
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("a review that runs out of time still returns, with what it has");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH010");
    let stop = observed
        .stop
        .as_deref()
        .expect("a review that spent its budget says so");
    assert!(
        stop.starts_with("review_budget: limits.maxSeconds is 8s"),
        "{stop}"
    );
    let declared = shipped_review_plan().shots["SH010"].questions.len();
    assert!(
        !observed.observations.is_empty() && observed.observations.len() < declared,
        "partial evidence is kept: {} of {declared} questions",
        observed.observations.len()
    );
    assert!(
        observed.frames.iter().any(|frame| frame.shot_id == "SH010"),
        "the sampled frames are kept"
    );
    assert!(
        observed
            .observations
            .iter()
            .all(|observation| observation.is_well_formed()),
        "{:#?}",
        observed.observations
    );
    assert_eq!(
        reviewed
            .shot("SH010")
            .expect("shot")
            .latest_review()
            .expect("review recorded")
            .stop
            .as_deref(),
        Some(stop),
        "the run record's review index carries the same stop"
    );
    assert!(
        harness.script.lock().vqa_asked.len() < declared * 3,
        "the review did not go on asking after its budget ran out"
    );
}

/// (C) `limits.maxSeconds` running out DURING a frame extraction is a `review_budget` stop that
/// keeps the frames already sampled — it used to surface as a transport error from the cancelled
/// extraction job, with no document written at all.
#[tokio::test]
async fn a_budget_that_runs_out_mid_extraction_stops_with_the_frames_already_sampled() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    // Frames take ~0.9 s each against a 2 s budget: the first one or two land, and the budget
    // runs out while a later one is still being extracted. Under a loaded runner even the first
    // can miss, which is the same stop with fewer frames — the claims below hold either way.
    harness.script.lock().frame_delay = Some(Duration::from_millis(900));
    let mut options = review_options(&harness, &["SH010"]);
    options.review_plan_path = Some(review_plan_with_limits(&harness, |plan| {
        plan.limits.max_seconds = 2;
    }));
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("a budget spent mid-extraction is a stop, not an error");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH010");
    let stop = observed
        .stop
        .as_deref()
        .expect("the review says why it stopped");
    assert!(
        stop.starts_with("review_budget: limits.maxSeconds is 2s"),
        "{stop}"
    );
    let sampled = observed
        .frames
        .iter()
        .filter(|frame| frame.shot_id == "SH010")
        .count();
    assert!(
        sampled < 3,
        "the extraction the budget interrupted must not have produced a frame: {sampled}"
    );
    assert_eq!(
        observed.frames.len(),
        sampled,
        "only the frames that landed before the budget ran out are kept: {:?}",
        observed.frames
    );
    assert!(
        observed.observations.is_empty(),
        "no question was put after the budget ran out: {:#?}",
        observed.observations
    );
    let jobs = harness.jobs().await;
    assert!(
        jobs.iter()
            .any(|job| job["type"] == "frame_extract" && job["status"] == "canceled"),
        "the extraction the budget interrupted was cancelled through the API"
    );
}

/// (C) `limits.maxAnswerSeconds`: an answer that does not arrive in time is recorded as an
/// `unobserved` observation carrying the timeout as its note — never as a value, never as an
/// error that throws the take's other evidence away — and the job it was waiting on is cancelled.
#[tokio::test]
async fn an_answer_that_times_out_is_recorded_unobserved_with_the_timeout_never_as_a_value() {
    let (harness, _) = rendered_two_shots().await;
    script_answers(&harness, &agreeing_answers());
    harness.script.lock().vqa_delay = Some(Duration::from_secs(30));
    let mut options = review_options(&harness, &["SH010"]);
    options.review_plan_path = Some(review_plan_with_limits(&harness, |plan| {
        plan.limits.max_answer_seconds = 1;
        plan.limits.max_seconds = 600;
    }));
    let vision = VqaVision::new(
        &harness.transport,
        options.poll_interval,
        options.control.clone(),
    );
    let reviewed = review::review(&harness.transport, &options, &vision)
        .await
        .expect("a timed-out answer is not an error");
    let observed = observed_for(&reviewed, &harness.out_dir(), "SH010");
    let declared = shipped_review_plan().shots["SH010"].questions.len();
    assert_eq!(
        observed.observations.len(),
        declared,
        "every declared question is in the document"
    );
    for observation in &observed.observations {
        assert!(observation.unobserved, "{observation:?}");
        assert_eq!(observation.verdict, Verdict::Unobserved, "{observation:?}");
        assert!(
            observation.observed.is_none(),
            "a timeout never becomes a value: {observation:?}"
        );
        assert!(observation.answers.is_empty(), "{observation:?}");
        let note = observation
            .note
            .as_deref()
            .expect("the timeout is the note");
        assert!(
            note.starts_with("answer_timeout: limits.maxAnswerSeconds is 1s"),
            "{note}"
        );
        assert!(note.contains("was cancelled"), "{note}");
        assert!(observation.is_well_formed(), "{observation:?}");
    }
    assert!(
        observed.stop.is_none(),
        "a per-answer timeout is not a review stop: {:?}",
        observed.stop
    );
    assert!(
        !observed.backend.real_model_inference,
        "no answer came from a model run, so the document must not claim one did"
    );
    // A mustObserve question that timed out raises its unobserved flag exactly as an honest
    // "I cannot tell" does.
    let shipped = shipped_review_plan();
    let must_observe: Vec<&str> = shipped.shots["SH010"]
        .questions
        .iter()
        .filter(|question| question.must_observe)
        .map(|question| question.id.as_str())
        .collect();
    for question_id in must_observe {
        assert!(
            observed
                .mismatches
                .iter()
                .any(|flag| flag.question_id == question_id),
            "{question_id} is mustObserve and was not observed: {:#?}",
            observed.mismatches
        );
    }
    // Every VQA job the reviewer gave up on was cancelled through the API, not abandoned — and a
    // question whose first frame timed out is not put to its remaining frames (one job each).
    let jobs = harness.jobs().await;
    let vqa: Vec<&Value> = jobs
        .iter()
        .filter(|job| job["type"] == "image_vqa")
        .collect();
    assert_eq!(
        vqa.len(),
        declared,
        "one cancelled job per question, none for the frames after the timeout: {}",
        vqa.len()
    );
    assert!(
        vqa.iter().all(|job| job["status"] == "canceled"),
        "{:?}",
        vqa.iter()
            .map(|job| job["status"].clone())
            .collect::<Vec<_>>()
    );
}

/// (G) The labeled evaluation over `VqaVision` — the real `image_vqa` route, the real frame import
/// through `POST /assets`, the fake worker answering from the SAME table the scripted backend
/// reads — scores exactly as the scripted run does. A route-level regression can no longer pass
/// the evaluation on the strength of a backend that never touches the route.
#[tokio::test]
async fn the_labeled_evaluation_over_the_real_vqa_route_scores_exactly_as_the_scripted_backend() {
    let harness = Harness::start(true, Vec::new()).await;
    harness.script.lock().vqa_answers = eval_answer_table();
    let temp = tempfile::tempdir().expect("temp dir");

    let scripted = review::review_eval(
        &harness.transport,
        &EvalOptions::new(PathBuf::from(EVAL_SET), temp.path().join("scripted")),
        &eval_backend(),
    )
    .await
    .expect("the scripted evaluation runs");

    let vision = VqaVision::new(
        &harness.transport,
        Duration::from_millis(50),
        RunControl::new(),
    );
    let routed = review::review_eval(
        &harness.transport,
        &EvalOptions::new(PathBuf::from(EVAL_SET), temp.path().join("routed")),
        &vision,
    )
    .await
    .expect("the evaluation runs over the real route");

    assert_eq!(routed.backend.kind, "image_vqa");
    assert_eq!(routed.backend.route, review::VQA_ROUTE);
    assert!(
        !routed.backend.real_model_inference,
        "the fake worker reports no weights ran, and the report says so"
    );
    assert_eq!(
        routed.totals, scripted.totals,
        "the verdicts are the route's, not the table's"
    );
    assert_eq!(routed.per_question, scripted.per_question);
    assert_eq!(routed.per_topic, scripted.per_topic);
    assert_eq!(routed.per_case.len(), scripted.per_case.len());
    for (routed_case, scripted_case) in routed.per_case.iter().zip(&scripted.per_case) {
        assert_eq!(routed_case.case_id, scripted_case.case_id);
        assert_eq!(
            routed_case.counts, scripted_case.counts,
            "{}",
            routed_case.case_id
        );
        assert_eq!(
            routed_case.verdicts, scripted_case.verdicts,
            "{}",
            routed_case.case_id
        );
    }
    // And the route really was driven: every graded answer came through an `image_vqa` job the
    // fake worker claimed, over frames imported as project assets.
    let asked = harness.script.lock().vqa_asked.len();
    assert!(asked > 0);
    // `GET /api/v1/jobs` pages at 100, so count the jobs the fake worker CLAIMED instead.
    let claimed_vqa = harness
        .script
        .lock()
        .claimed
        .iter()
        .filter(|(kind, _, _)| kind == "image_vqa")
        .count();
    assert_eq!(claimed_vqa, asked);
    let scored: u32 = routed.per_case.iter().map(|case| case.counts.scored).sum();
    assert!(
        asked as u32 >= scored,
        "every scored question rode at least one job: {asked} jobs, {scored} scored"
    );
    for case in &routed.per_case {
        let observed = review::read_observed_state(Path::new(&case.observed_state_path))
            .expect("observed state reads");
        assert!(
            observed
                .frames
                .iter()
                .all(|frame| !frame.asset_id.is_empty()),
            "every frame the evidence cites is a project asset: {:?}",
            observed.frames
        );
        for frame in &observed.frames {
            let (status, asset) = crate::tests::support::request(
                harness.app.clone(),
                "GET",
                &format!(
                    "/api/v1/projects/{}/assets/{}",
                    routed_project_id(&harness).await,
                    frame.asset_id
                ),
                Value::Null,
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::OK, "{asset}");
            assert_eq!(
                asset["extra"]["filmHarness"]["kind"], "review_frame",
                "{asset}"
            );
        }
    }
}

/// The project the routed evaluation imported its frames into (`film-harness review-eval (…)`).
async fn routed_project_id(harness: &Harness) -> String {
    let (_, projects) =
        crate::tests::support::request(harness.app.clone(), "GET", "/api/v1/projects", Value::Null)
            .await;
    projects
        .as_array()
        .into_iter()
        .flatten()
        .find(|project| {
            project["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("film-harness review-eval ("))
        })
        .and_then(|project| project["id"].as_str())
        .expect("the evaluation created its project")
        .to_owned()
}

// ---------------------------------------------------------------------------------------------
// sc-23405 (S4) — one shared workshop
// ---------------------------------------------------------------------------------------------

/// AC3, first half. The shipped review plan validates against BOTH courier plans — the phase-1
/// `plan.jsonc` baseline and the reference-conditioned `plan.v2.jsonc` — so the same questions
/// grade the same film either way and the two are comparable answer for answer.
#[test]
fn the_shipped_review_plan_validates_against_both_courier_plans() {
    let review = shipped_review_plan();
    assert_eq!(review.version, 3);
    for name in ["plan.jsonc", "plan.v2.jsonc"] {
        let plan =
            sceneworks_core::film_plan::read_plan_file(&PathBuf::from(FIXTURE_DIR).join(name))
                .unwrap_or_else(|error| panic!("{name} reads: {error}"));
        let findings = validate_review_plan(&review, &plan);
        assert!(
            findings.is_empty(),
            "{name}: {:?}",
            findings
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        );
    }
}

/// AC3, second half. Every `acrossCut` question compares a feature of a reference BOTH sides of the
/// cut are conditioned on in `plan.v2.jsonc`.
///
/// That is the whole change of premise this story makes to the review. In phase 1 a cut question
/// could only ask whether two independently invented rooms happened to agree — the documented miss
/// of the first smoke, where both sides answered "a woodworking workshop" while rendering different
/// ones. In plan.v2 every shot binds the approved `workshop_location`, `workbench_table` and
/// `red_parcel`, so each cut question names in its `intended` the role the comparison rests on, and
/// this test holds the document to it: the named role must be approved in the pack and bound by the
/// shot AND by the shot its `dependsOn` edge points at. A question whose `intended` names a role
/// only one side binds is comparing nothing.
#[test]
fn every_cut_question_compares_a_reference_both_sides_of_the_cut_bind() {
    let review = shipped_review_plan();
    let plan = sceneworks_core::film_plan::read_plan_file(
        &PathBuf::from(FIXTURE_DIR).join("plan.v2.jsonc"),
    )
    .expect("plan.v2.jsonc reads");
    let pack = sceneworks_core::film_plan::read_reference_pack_file(
        &PathBuf::from(FIXTURE_DIR).join("references.jsonc"),
    )
    .expect("the pack reads");
    let approved: Vec<&str> = pack
        .references
        .iter()
        .filter(|entry| entry.approved)
        .map(|entry| entry.role.as_str())
        .collect();
    let bound = |shot_id: &str| -> Vec<String> {
        let shot = plan
            .shots
            .iter()
            .find(|shot| shot.id == shot_id)
            .unwrap_or_else(|| panic!("{shot_id} is in plan.v2.jsonc"));
        // CONDITIONING only. `continuityRoles` is a continuity DECLARATION — it says the role is
        // meant to stay the same across the cut — and does not put the image on either request, so
        // chaining it in would let a question pass on two shots that were both conditioned on
        // nothing. That is the documented miss this test exists to catch, so the set it compares is
        // exactly the roles the compiled requests carry (sc-23405 review).
        shot.conditioning.reference_roles.clone()
    };

    let mut checked = 0_usize;
    for (shot_id, spec) in &review.shots {
        for question in &spec.questions {
            if !question.across_cut {
                continue;
            }
            checked += 1;
            let shot = plan
                .shots
                .iter()
                .find(|shot| &shot.id == shot_id)
                .unwrap_or_else(|| panic!("{shot_id} is in plan.v2.jsonc"));
            let source = shot
                .depends_on
                .first()
                .unwrap_or_else(|| panic!("{shot_id}'s acrossCut question needs a dependsOn edge"));
            let here = bound(shot_id);
            let there = bound(&source.shot_id);
            let shared: Vec<&str> = approved
                .iter()
                .copied()
                .filter(|role| {
                    question.intended.contains(*role)
                        && here.iter().any(|bound| bound == role)
                        && there.iter().any(|bound| bound == role)
                })
                .collect();
            assert!(
                !shared.is_empty(),
                "{}: its `intended` must name an approved role that BOTH {shot_id} and {} bind, \
                 or the question compares nothing. intended={:?}; {shot_id} binds {here:?}; {} \
                 binds {there:?}",
                question.id,
                source.shot_id,
                question.intended,
                source.shot_id,
            );
        }
    }
    assert_eq!(checked, 5, "one cut question per cut, SH020 through SH060");

    // Where the room is in frame, the shared feature graded IS the workshop — the pegboard of hand
    // tools the approved `workshop_location` plate declares, asked identically on both sides.
    for shot_id in ["SH020", "SH040", "SH050"] {
        let question = review.shots[shot_id]
            .questions
            .iter()
            .find(|question| question.across_cut)
            .expect("a cut question");
        assert!(
            question.intended.contains("workshop_location")
                && question.ask.contains("pegboard")
                && question.expect.iter().any(|word| word == "yes"),
            "{shot_id}: {question:?}"
        );
    }
    // On the two CLOSE-UPS the room is not in frame, so the shared reference graded is the parcel.
    // A frame cannot be asked about a wall it does not contain: that answer would be `unobserved`,
    // which is a flag rather than evidence.
    for shot_id in ["SH030", "SH060"] {
        let question = review.shots[shot_id]
            .questions
            .iter()
            .find(|question| question.across_cut)
            .expect("a cut question");
        assert!(
            question.intended.contains("red_parcel") && question.ask.contains("parcel"),
            "{shot_id}: {question:?}"
        );
    }

    // And the LOCATION questions name the approved plate they are grading against, rather than
    // "the same room as SH010" — which under plan.v2 is a fact about the pack, not about a take.
    for (shot_id, role) in [
        ("SH010", "workshop_location"),
        ("SH020", "workshop_location"),
        ("SH030", "workbench_table"),
        ("SH040", "workshop_location"),
        ("SH050", "workshop_location"),
        ("SH060", "workbench_table"),
    ] {
        let question = review.shots[shot_id]
            .questions
            .iter()
            .find(|question| question.topic == "location")
            .unwrap_or_else(|| panic!("{shot_id} has a location question"));
        assert!(
            question.intended.contains(role),
            "{shot_id}'s location question must name the approved {role} it grades against: {:?}",
            question.intended
        );
    }
}
